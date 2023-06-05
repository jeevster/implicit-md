import numpy as np
import gsd.hoomd
import torch
import torch.nn as nn
from nff.utils.scatter import compute_grad
from itertools import product
from nff.nn.layers import GaussianSmearing
from YParams import YParams
import argparse
import os
from tqdm import tqdm
import pstats
import pdb
import random
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import generate_vol_bins, DifferentiableRDF, DifferentiableVelHist, DifferentiableVACF, msd, DiffusionCoefficient
import torchopt
from torchopt.nn import ImplicitMetaGradientModule
from contextlib import nullcontext
import time
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
from sys import getrefcount
from functorch import vmap
from utils import radii_to_dists, fcc_positions, initialize_velocities, \
                    dump_params_to_yml, powerlaw_inv_cdf, print_active_torch_tensors, plot_pair, mean_across_lists, subtract_across_lists, multiply_across_lists


class ImplicitMDSimulator():
    def __init__(self, params, model, radii_0, velocities_0, rdf_0):
        super(ImplicitMDSimulator, self).__init__()

        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        #Set random seeds
        np.random.seed(seed=params.seed)
        torch.manual_seed(params.seed)
        random.seed(params.seed)
        self.params = params
        self.n_particle = params.n_particle
        self.temp = params.temp
        self.kbt0 = params.kbt0
        self.box = params.box
        self.pbc = not params.no_pbc
        self.dt = params.dt
        self.t_total = params.t_total
        self.n_replicas = params.n_replicas

        #Nose-Hoover Thermostat stuff
        self.nvt_time = params.nvt_time
        self.targeEkin = 0.5 * (3.0 * self.n_particle) * self.temp
        #self.Q = 3.0 * self.n_particle * self.temp * (self.t_total * self.dt)**2
        self.Q = 3.0 * self.n_particle * self.temp * (50 * self.dt)**2 #found that this setting works well
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)

        self.diameter_viz = params.diameter_viz
        self.epsilon = params.epsilon
        self.rep_power = params.rep_power
        self.attr_power = params.attr_power
        self.poly = params.poly
        self.poly_power = params.poly_power
        self.min_sigma = params.min_sigma
        self.sigma = params.sigma
        
        self.dr = params.dr
        self.n_nvt_steps = np.rint(self.nvt_time/self.dt).astype(np.int32)
        self.nsteps = np.rint(self.t_total/self.dt).astype(np.int32)
        self.burn_in_frac = params.burn_in_frac
        self.nn = params.nn
        self.save_intermediate_rdf = params.save_intermediate_rdf
        self.exp_name = params.exp_name

        self.diffusion_window = params.diffusion_window
        self.vacf_window = params.vacf_window

        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear

        self.c_0 = -28*self.epsilon / (self.cutoff**12)
        self.c_2 = 48*self.epsilon / (self.cutoff**14)
        self.c_4 = -21*self.epsilon / (self.cutoff**16)


        # Constant box properties
        self.vol = self.box**3.0
        self.rho = self.n_particle/self.vol

        self.rdf_loss_weight = params.rdf_loss_weight
        self.diffusion_loss_weight = params.diffusion_loss_weight
        self.vacf_loss_weight = params.vacf_loss_weight

        self.zeros = torch.zeros((1, 1, 3)).to(self.device)
        
        #limit CPU usage
        torch.set_num_threads(10)

        #Register inner parameters
        self.model = model#.to(self.device)
        #model.train()
        self.radii = nn.Parameter(radii_0.detach_().clone(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities_0.detach_().clone(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(rdf_0.detach_().clone(), requires_grad=True).to(self.device)
        #if self.diffusion_loss_weight!=0:
        self.diff_coeff = nn.Parameter(torch.zeros((self.n_replicas,)), requires_grad=True).to(self.device)
        #if self.vacf_loss_weight !=0:   
        self.vacf = nn.Parameter(torch.zeros((self.n_replicas,self.vacf_window)), requires_grad=True).to(self.device)

        if self.poly: #get per-particle sigmas
            u = torch.rand((self.n_replicas, self.n_particle))
            self.particle_sigmas = powerlaw_inv_cdf(u, power = self.poly_power, y_min = self.min_sigma).clip(max = 1/0.45 * self.min_sigma)
            sigma1 = self.particle_sigmas.unsqueeze(1)
            sigma2 = self.particle_sigmas.unsqueeze(2)
            self.sigma_pairs = 1/2 * (sigma1 + sigma2)*(1 - self.epsilon*torch.abs(sigma1 - sigma2))
            self.sigma_pairs = self.sigma_pairs[:, ~torch.eye(self.sigma_pairs.shape[1],dtype=bool)].reshape(self.n_replicas, self.sigma_pairs.shape[1], -1, 1).to(self.device)
        else:
            #TODO: come up with a general function for polydisperse LJ systems
            self.energy_fn = LJFamily(epsilon=self.epsilon, sigma=self.sigma, rep_pow=self.rep_power, attr_pow=self.attr_power)

        #define vectorized differentiable rdf and velhist
        self.diff_rdf = vmap(DifferentiableRDF(params, self.device), -1)
        self.diff_rdf_cpu = vmap(DifferentiableRDF(params, "cpu"), -1)
        self.diff_vel_hist = vmap(DifferentiableVelHist(params, self.device), 0)
        self.diff_vel_hist_cpu = vmap(DifferentiableVelHist(params, "cpu"), 0)

        #define vectorized differentiable vacf and diffusion coefficients
        self.diff_vacf = vmap(DifferentiableVACF(params), 0)
        #vectorize over dim 1 (the input will be of shape [diffusion_window , n_replicas])
        self.diffusion_coefficient = vmap(DiffusionCoefficient(params, self.device) , 1)
        

        if self.nn:
            add = "polylj_" if self.poly else ""
            results = 'results_polylj' if self.poly else 'results'
            self.save_dir = os.path.join(results, f"IMPLICIT_{add}_{self.exp_name}_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_dt={self.dt}_ttotal={self.t_total}")
        else: 
            add = "_polylj" if self.poly else ""
            self.save_dir = os.path.join('ground_truth' + add, f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}")

        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)

        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")
        self.t = gsd.hoomd.open(name=f'{self.save_dir}/sim_temp{self.temp}.gsd', mode='wb') 
        self.n_dump = params.n_dump # dump for configuration

    '''memory cleanups'''
    def cleanup(self):
        #detach the radii and velocities - saves some memory
        self.last_h_radii = [radii.detach() for radii in self.last_h_radii]
        self.last_h_velocities = [vels.detach() for vels in self.last_h_velocities]
        last_radii = self.radii
        last_velocities = self.velocities
        last_rdf = self.rdf
        #de-register all the parameters to kill the gradients
        for param in ['radii', 'velocities', 'rdf', 'diff_coeff', 'vacf']:
                self.register_parameter(param, None)
        return last_radii, last_velocities, last_rdf

    def reset(self, radii, velocities, rdf):
        self.radii = nn.Parameter(radii.detach_().clone(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities.detach_().clone(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(rdf.detach_().clone(), requires_grad=True).to(self.device)
        self.diff_coeff = nn.Parameter(torch.zeros((self.n_replicas,)), requires_grad=True).to(self.device)
        self.vacf = nn.Parameter(torch.zeros((self.n_replicas,self.vacf_window)), requires_grad=True).to(self.device)



    def check_symmetric(self, a, mode, tol=1e-4):
        if mode == 'opposite':
            return np.all(np.abs(a + a.T) < tol)
        
        return np.all(np.abs(a - a.T) < tol)

    # Function to dump simulation frame that is readable in Ovito
    # Also stores radii and velocities in a compressed format which is nice
    def create_frame(self, frame):
        # Particle positions, velocities, diameter
        radii = self.radii[0].detach()
        partpos = radii.tolist()
        velocities = self.velocities[0].detach().tolist()
        sigmas = self.particle_sigmas[0] if self.poly else self.sigma
        diameter = self.diameter_viz*sigmas*np.ones((self.n_particle,))
        diameter = diameter.tolist()

        # Now make gsd file
        s = gsd.hoomd.Frame()
        s.configuration.step = frame
        s.particles.N=self.n_particle
        s.particles.position = partpos
        s.particles.velocity = velocities
        s.particles.diameter = diameter
        s.configuration.box=[self.box,self.box,self.box,0,0,0]

        return s

    def calc_properties(self, pe):
        #TODO
        # Calculate properties of interest in this function
        p_dof = 3*self.n_particle-3
        vel_squared = torch.sum(torch.square(self.velocities))
        ke = vel_squared/2
        temp = 2*ke/p_dof
        #w = -1/6*torch.sum(self.internal_virial)
        #pressure = w/self.vol + self.rho*self.kbt0
        pressure = torch.Tensor(0)
        return {"Temperature": temp.item(),
                #"Pressure": pressure,
                "Potential Energy": pe.item(),
                "Total Energy": (ke+pe).item(),
                "Momentum Magnitude": torch.norm(torch.sum(self.velocities, axis =-2)).item()}


    '''CORE MD OPERATIONS'''
    def poly_potential(self, dists, sigma_pairs = None):
        s = self.sigma_pairs if sigma_pairs is None else sigma_pairs
        parenth = (s/dists)
        rep_term = parenth ** 12
        energy = torch.sum(rep_term + self.c_0 + self.c_2*parenth**-2 + self.c_4*parenth**-4, dim = -1)
        #TODO: look at this - not quite right
        energy[(dists > self.cutoff).squeeze(-1)] = 0
        return energy

    def force_calc(self, radii, retain_grad = False):
        
        #Get rij matrix
        with torch.enable_grad():
            r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
            if not r.requires_grad:
                r.requires_grad = True

            #Enforce minimum image convention
            r = -1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r))

            #get rid of diagonal 0 entries of r matrix (for gradient stability)
            r = r[:, ~torch.eye(r.shape[1],dtype=bool)].reshape(r.shape[0], r.shape[1], -1, 3)
            
            #compute distance matrix:
            dists = torch.sqrt(torch.sum(r**2, axis=-1)).unsqueeze(-1)

            #compute energy
            if self.nn:
                energy = self.model((dists/self.sigma_pairs)) if self.poly else self.model(dists)
                forces = -compute_grad(inputs=r, output=energy) if retain_grad else -compute_grad(inputs=r, output=energy).detach()
                
            #LJ potential
            else:
                if self.poly: #repulsive
                    energy = self.poly_potential(dists)
                    forces = -compute_grad(inputs=r, output=energy) if retain_grad else -compute_grad(inputs=r, output=energy).detach()
                
                else:
                    energy = self.energy_fn(dists)
                    # #reuse components of potential to calculate virial and forces
                    # internal_virial = -48*self.epsilon*r6i*(r6i - 0.5)/(self.sigma**2)
                    # forces = -internal_virial*r*r2i
                    forces = -compute_grad(inputs=r, output=energy) if retain_grad else -compute_grad(inputs=r, output=energy).detach()

                #calculate which forces to keep
                keep = dists/self.sigma_pairs <= self.cutoff if self.poly else dists/self.sigma <= self.cutoff

                #apply cutoff
                forces = torch.where(~keep, self.zeros, forces)

        #Ensure no NaNs
        assert(not torch.any(torch.isnan(forces)))
        
        #sum forces across particles
        return energy, torch.sum(forces, axis = -2)#.to(self.device)
        
    def forward_nvt(self, radii, velocities, forces, zeta, calc_rdf = False, calc_diffusion = False, calc_vacf = False, retain_grad = False):
        # get current accelerations (assume unit mass)
        accel = forces

        # make full step in position 
        radii = radii + velocities * self.dt + \
            (accel - zeta * velocities) * (0.5 * self.dt ** 2)

        #PBC correction
        if self.pbc:
            radii = radii/self.box 
            radii = self.box*torch.where(radii-torch.round(radii) >= 0, \
                        (radii-torch.round(radii)), (radii - torch.floor(radii)-1))

        # record current velocities
        
        KE_0 = torch.sum(torch.square(velocities), axis = (1,2), keepdims=True) / 2
        
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)

        # make a full step in accelerations
        energy, forces = self.force_calc(radii.to(self.device), retain_grad=retain_grad)
        accel = forces

        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * (1/self.Q) * (KE_0 - self.targeEkin)

        #get updated KE
        ke = torch.sum(torch.square(velocities), axis = (1,2), keepdims=True) / 2

        # make another halfstep in self.zeta
        zeta = zeta + 0.5 * self.dt * \
            (1/self.Q) * (ke - self.targeEkin)

        # make another half step in velocity
        velocities = (velocities + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * zeta)

        if calc_rdf:
            new_dists = radii_to_dists(radii, self.params)
            if self.poly:
                #normalize distances by sigma pairs
                new_dists = new_dists / self.sigma_pairs

        new_velhist = self.diff_vel_hist(torch.linalg.norm(velocities, dim=-1).permute((1,0))) if calc_rdf else 0 #calculate velocity histogram from a single frame
        
        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device).permute((1, 2, 3, 0)))) if calc_rdf else 0 #calculate the RDF from a single frame
        #new_rdf = 0

        if calc_diffusion:
            self.last_h_radii.append(radii.unsqueeze(0))
        if calc_vacf:
            self.last_h_velocities.append(velocities.unsqueeze(0))

        # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not self.nn:
                new_dists = radii_to_dists(radii, self.params)
                if self.poly:
                    #normalize distances by sigma pairs
                    new_dists = new_dists / self.sigma_pairs
                self.running_dists.append(new_dists.cpu().detach())
                self.running_vels.append(torch.linalg.norm(velocities, dim = -1).cpu().detach())

        return radii, velocities, forces, zeta, new_rdf, new_velhist
        
        
    def forward(self, radii, velocities, forces, calc_rdf = False, calc_diffusion = False, calc_vacf = False, retain_grad = False):
        # Forward process - 1 MD step with Velocity-Verlet integration
        #half-step in velocity
        
        velocities = velocities + 0.5*self.dt*forces

        #full step in position
        radii = radii + self.dt*velocities

        #PBC correction
        if self.pbc:
            radii = radii/self.box 
            radii = self.box*torch.where(radii-torch.round(radii) >= 0, \
                        (radii-torch.round(radii)), (radii - torch.floor(radii)-1))

        #calculate force at new position
        energy, forces = self.force_calc(radii.to(self.device), retain_grad=retain_grad)
        
        #another half-step in velocity
        velocities = (velocities + 0.5*self.dt*forces) 
        #props = self.calc_properties()

        if calc_rdf:
            new_dists = radii_to_dists(radii, self.params)
            if self.poly:
                #normalize distances by sigma pairs
                new_dists = new_dists/ self.sigma_pairs

        new_velhist = self.diff_vel_hist(torch.linalg.norm(velocities, dim=-1).permute((1,0))) if calc_rdf else 0 #calculate velocity histogram from a single frame
        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device).permute((1, 2, 3, 0)))) if calc_rdf else 0 #calculate the RDF from a single frame
        #new_rdf = 0
        if calc_diffusion:
            self.last_h_radii.append(radii.unsqueeze(0))
        if calc_vacf:
            self.last_h_velocities.append(velocities.unsqueeze(0))

        # dump frame
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not self.nn:
                new_dists = radii_to_dists(radii, self.params)
                if self.poly:
                    #normalize distances by sigma pairs
                    new_dists = new_dists / self.sigma_pairs
                self.running_dists.append(new_dists.cpu().detach())
                self.running_vels.append(torch.linalg.norm(velocities, dim = -1).cpu().detach())

        return radii, velocities, forces, new_rdf, new_velhist # return the new distance matrix 


    '''Stationary condition construction for calculating implicit gradient'''
    def optimality(self, enable_grad = True):
        #get current forces - treat as a constant (since it's coming from before the fixed point)
        forces = self.force_calc(self.radii, retain_grad=False)[1]
        
        #hacky fix for non-square matrix stuff
        if type(enable_grad) != bool:
            enable_grad = False

        with torch.enable_grad() if enable_grad else nullcontext():
            if self.vacf_loss_weight != 0:
                old_vacf = self.vacf

            #make an MD step - retain grads
            new_radii, new_velocities, new_forces, new_zeta, new_rdf, _ = self.forward_nvt(self.radii, self.velocities, forces, self.zeta, calc_rdf = True, calc_diffusion = self.diffusion_loss_weight!=0, calc_vacf = self.vacf_loss_weight!=0, retain_grad = True)
            
            #compute residuals
            radii_residual  = self.radii - new_radii
            velocity_residual  = self.velocities - new_velocities
            rdf_residual = self.rdf - new_rdf

            #compute diffusion coefficient residual
            if self.diffusion_loss_weight != 0:
                new_msd_data = msd(torch.cat(self.last_h_radii[1:], dim=0), self.box)
                new_diffusion_coeff = self.diffusion_coefficient(new_msd_data)
                diffusion_residual = (new_diffusion_coeff - self.diff_coeff)
                zeta_residual = (self.zeta - new_zeta)

            #compute VACF residual
            if self.vacf_loss_weight != 0:
                last_h_vels = torch.cat(self.last_h_velocities[1:], dim = 0).permute((1,0,2,3))
                new_vacf = self.diff_vacf(last_h_vels)
                vacf_residual = (new_vacf - self.vacf)
            
            #for now assume rdf + only one dynamical observable: either diffusion coefficient or VACF, not both
            if self.diffusion_loss_weight != 0:
                return (radii_residual, velocity_residual, rdf_residual, diffusion_residual, zeta_residual)
            elif self.vacf_loss_weight != 0:
                return (radii_residual, velocity_residual, rdf_residual, vacf_residual)

            return (radii_residual, velocity_residual, rdf_residual)


    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        self.running_dists = []
        self.running_vels = []
        self.last_h_radii = []
        self.last_h_velocities = []
        self.running_radii = []
        #Initialize forces/potential of starting configuration
        with torch.no_grad():
            
            _, forces = self.force_calc(self.radii)
            #Run MD
            print("Start MD trajectory", file=self.f)
            
            zeta = self.zeta
            radii = self.radii
            velocities = self.velocities
            diffusion_grad_start = self.nsteps - self.diffusion_window if self.diffusion_loss_weight !=0 else self.nsteps
            vacf_grad_start = self.nsteps - self.vacf_window if self.vacf_loss_weight !=0 else self.nsteps
            no_grad_override = 0 if self.save_intermediate_rdf or not self.nn else self.nsteps
            num_no_grad_steps = min(self.nsteps, diffusion_grad_start, vacf_grad_start, no_grad_override)
            #start no grad steps
            for step in tqdm(range(num_no_grad_steps)):
                self.step = step
                
                calc_rdf = step ==  self.nsteps -1 or self.save_intermediate_rdf or not self.nn
                calc_diffusion = (step >= self.nsteps - self.diffusion_window) or self.save_intermediate_rdf or not self.nn #calculate diffusion coefficient if within window from the end
                calc_vacf = (step >= self.nsteps - self.vacf_window) or self.save_intermediate_rdf or not self.nn #calculate VACF if within window from the end

                #MD step
                radii, velocities, forces, zeta, rdf, velhist = self.forward_nvt(radii, velocities, forces, zeta, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion, calc_vacf = calc_vacf)
                
                #save equilibriated radiis for backward pass
                if step > self.eq_steps and step % self.n_dump == 0: 
                    self.running_radii.append(radii.detach())
                
                if not self.nn and self.save_intermediate_rdf and step % self.n_dump == 0:
                    filename = f"step{step+1}_rdf.npy"
                    np.save(os.path.join(self.save_dir, filename), rdf.mean(dim = 0).cpu().detach().numpy())

                    filename = f"step{step+1}_velhist.npy"
                    np.save(os.path.join(self.save_dir, filename), velhist.mean(dim = 0).cpu().detach().numpy())

                    if step > self.vacf_window:
                        last_h_vels = torch.cat(self.last_h_velocities[-self.vacf_window:], dim = 0).permute((1,0,2,3))
                        vacf = self.diff_vacf(last_h_vels)
                        filename = f"step{step+1}_vacf.npy"
                        np.save(os.path.join(self.save_dir, filename), vacf.mean(dim = 0).cpu().detach().numpy())
        #now do grad steps
        print("Switching over to gradient enabled steps")
        with torch.enable_grad():
            self.radii = radii.requires_grad_(True)
            self.velocities = velocities.requires_grad_(True)

            for step in tqdm(range(num_no_grad_steps, self.nsteps)):
                self.step = step
                retain_grad = step >= self.nsteps -2
                
                calc_rdf = step ==  self.nsteps -1 or self.save_intermediate_rdf or not self.nn
                calc_diffusion = (step >= self.nsteps - self.diffusion_window) or self.save_intermediate_rdf or not self.nn #calculate diffusion coefficient if within window from the end
                calc_vacf = (step >= self.nsteps - self.vacf_window) or self.save_intermediate_rdf or not self.nn #calculate VACF if within window from the end

                #MD step
                self.radii, self.velocities, forces, zeta, self.rdf, velhist = self.forward_nvt(self.radii, self.velocities, forces, zeta, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion, calc_vacf = calc_vacf, retain_grad = retain_grad)
                
                #save equilibriated radiis for backward pass
                if step > self.eq_steps and step % self.n_dump == 0: 
                    self.running_radii.append(self.radii.detach())
                
                if not self.nn and self.save_intermediate_rdf and step % self.n_dump == 0:
                    filename = f"step{step+1}_rdf.npy"
                    np.save(os.path.join(self.save_dir, filename), self.rdf.mean(dim = 0).cpu().detach().numpy())

                    filename = f"step{step+1}_velhist.npy"
                    np.save(os.path.join(self.save_dir, filename), velhist.mean(dim = 0).cpu().detach().numpy())

                    if step > self.vacf_window:
                        last_h_vels = torch.cat(self.last_h_velocities[-self.vacf_window:], dim = 0).permute((1,0,2,3))
                        vacf = self.diff_vacf(last_h_vels)
                        filename = f"step{step+1}_vacf.npy"
                        np.save(os.path.join(self.save_dir, filename), vacf.mean(dim = 0).cpu().detach().numpy())

        #postprocessing stuff
        length = len(self.running_dists)
        self.zeta = zeta
        #compute diffusion coefficient
        #if self.diffusion_loss_weight != 0 or not self.nn:
        msd_data = msd(torch.cat(self.last_h_radii, dim=0), self.box)
        diffusion_coeff = self.diffusion_coefficient(msd_data)
        self.diff_coeff.copy_(diffusion_coeff)
        
        filename ="gt_diff_coeff.npy" if not self.nn else f"diff_coeff_epoch{epoch+1}.npy"
        np.save(os.path.join(self.save_dir, filename), diffusion_coeff.mean().cpu().detach().numpy())

        #compute VACF
        #if self.vacf_loss_weight != 0 or not self.nn:
        last_h_vels = torch.cat(self.last_h_velocities, dim = 0).permute((1,0,2,3))
        vacf = self.diff_vacf(last_h_vels)
        self.vacf.copy_(vacf)
        filename ="gt_vacf.npy" if not self.nn else f"vacf_epoch{epoch+1}.npy"
        np.save(os.path.join(self.save_dir, filename), vacf.mean(dim=0).cpu().detach().numpy())

        #compute ground truth rdf over entire trajectory (do it on CPU to avoid memory issues)
        save_velhist = self.diff_vel_hist_cpu(torch.stack(self.running_vels[int(self.burn_in_frac*length):], dim = 1)) if not self.nn else velhist
        save_rdf = self.diff_rdf_cpu(self.running_dists[int(self.burn_in_frac*length):]) if not self.nn else self.rdf
        filename ="gt_rdf.npy" if not self.nn else f"rdf_epoch{epoch+1}.npy"
        np.save(os.path.join(self.save_dir, filename), save_rdf.mean(dim=0).cpu().detach().numpy())

        filename ="gt_velhist.npy" if not self.nn else f"velhist_epoch{epoch+1}.npy"
        np.save(os.path.join(self.save_dir, filename), save_velhist.mean(dim=0).cpu().detach().numpy())

        #plot true and current energy functions
        plot_pair(epoch, self.save_dir, self.model, self.device, end=2.5, target_pot=lambda dists: self.poly_potential(dists, sigma_pairs = 1) if self.poly else self.energy_fn)

        return self

class Stochastic_IFT(torch.autograd.Function):
    def __init__(self):
        super(Stochastic_IFT, self).__init__()
        
    @staticmethod
    def forward(ctx, *args):
        simulator = args[0]
        gt_rdf = args[1] 
        params = args[2]
        def outer_loss(rdf):
            rdf_loss = (rdf - gt_rdf).pow(2).mean()
            return params.rdf_loss_weight*rdf_loss
        
        simulator.eq_steps = 1000
        simulator.nsteps = 3000
        equilibriated_simulator = simulator.solve()
        ctx.save_for_backward(equilibriated_simulator)
        model = equilibriated_simulator.model
        radii = equilibriated_simulator.running_radii
        dists = [radii_to_dists(r, simulator.params) for r in radii]
        if equilibriated_simulator.poly:
            #normalize distances by sigma pairs
            dists = [d / equilibriated_simulator.sigma_pairs for d in dists]
        rdfs = [equilibriated_simulator.diff_rdf(tuple(d.permute((1, 2, 3, 0)))) for d in dists]
        losses = [outer_loss(rdf) for rdf in rdfs]
        #compute energies at each of the positions and then compute gradients wrt model parameters
        with torch.enable_grad():
            energies = [model((d/simulator.sigma_pairs)) if simulator.poly else model(d) for d in dists]
            grads = [compute_grad(inputs=model.parameters(), output=energy) for energy in energies]
        
        #compute the estimator on every pair of positions
        gradients = [multiply_across_lists(losses[j], subtract_across_lists(grads[i], grads[j])) for i,j in product(list(range(len(grads))),repeat=2) if i!=j]
        gradient_estimator = mean_across_lists(gradients)
        final_grad_norms = [torch.linalg.norm(g, dim =-1).max().item() for g in gradient_estimator]
        
        return equilibriated_simulator, gradient_estimator

    @staticmethod
    def backward(ctx, *grad_output):
        import pdb; pdb.set_trace()
        equilibriated_simulator = ctx.saved_tensors
        #gradient calculation using Fabian's method
        model = equilibriated_simulator.model
        radii = equilibriated_simulator.running_radii
        with torch.enable_grad():
            radii = [r.requires_grad_(True) for r in radii]
            dists = [radii_to_dists(r.requires_grad_(True), simulator.params) for r in radii]
            energies = [model((d/simulator.sigma_pairs)) if simulator.poly else model(d) for d in dists]
            grads = [compute_grad(inputs=model.parameters(), output=energy / simulator.temp) for energy in energies]
            forces = [-compute_grad(inputs=r, output=energy) for r, energy in zip(radii, energies)]
            grads_except = lambda i: grads[:i] + grads[i+1:]
            #not done yet - need to multiple each element by the radiis themselves
            gradient_estimator = mean_across_lists([subtract_across_lists(grad, mean_across_lists(grads_except(i))) for i, grad in enumerate(grads)])

        return None
        
if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument('--n_particle', type=int, default=256, help='number of particles')
    parser.add_argument('--temp', type=float, default=1, help='temperature in reduced units')
    parser.add_argument('--seed', type=int, default=123, help='random seed used to initialize velocities')
    parser.add_argument('--kbt0', type=float, default=1.8, help='multiplier for pressure calculation')
    parser.add_argument('--box', type=int, default=7, help='box size')
    parser.add_argument('--no_pbc', action='store_true', help='do not do periodic boundary condition')

    parser.add_argument('--epsilon', type=float, default=1.0, help='LJ epsilon')
    parser.add_argument('--rep_power', type=float, default=12, help='LJ repulsive exponent')
    parser.add_argument('--attr_power', type=float, default=0, help='LJ attractive exponent')
    parser.add_argument('--poly', action='store_true', help='vary sigma (diameter) of each particle according to power law')
    parser.add_argument('--poly_power', type=float, default = -3.0, help='power of power law')
    parser.add_argument('--min_sigma', type=float, default = 0.5, help='minimum sigma for power law distribution')
    parser.add_argument('--sigma', type=float, default=1.0, help='LJ sigma for non-poly systems')
    parser.add_argument('--dt', type=float, default=0.005, help='time step for integration')
    parser.add_argument('--dr', type=float, default=0.01, help='bin size for RDF calculation')
    parser.add_argument('--dv', type=float, default=0.1, help='bin size for velocity histogram calculation')

    parser.add_argument('--rdf_sample_frac', type=float, default=1, help='fraction of particles to sample for RDF')
    parser.add_argument('--diffusion_window', type=int, default=100, help='number of timesteps on which to fit linear regression for estimating diffusion coefficient')
    parser.add_argument('--vacf_window', type=int, default=50, help='number of timesteps on which to calculate the velocity autocorrelation function')

    parser.add_argument('--t_total', type=float, default=5, help='total time')
    parser.add_argument('--nvt_time', type=float, default=5, help='time for NVT equilibration')

    parser.add_argument('--diameter_viz', type=float, default=0.3, help='particle diameter for Ovito visualization')
    parser.add_argument('--n_dump', type=int, default=10, help='save frequency of configurations (also frequency of frames used for ground truth RDF calculation)')
    parser.add_argument('--save_intermediate_rdf', action = 'store_true', help='Whether to store the RDF along the trajectory for the ground truth')
    parser.add_argument('--burn_in_frac', type=float, default=0.2, help='initial fraction of trajectory to discount when calculating ground truth rdf')
    parser.add_argument('--exp_name', type=str, default = "", help='name of experiment - used as prefix of results folder name')

    #learnable potential stuff
    parser.add_argument('--n_epochs', type=int, default=30, help='number of outer loop training epochs')
    parser.add_argument('--n_replicas', type=int, default=5, help='number of simulations to parallelize over')

    parser.add_argument('--nn', action='store_true', help='use neural network potential')
    parser.add_argument('--cutoff', type=float, default=2.5, help='LJ cutoff distance')
    parser.add_argument('--gaussian_width', type=float, default=0.1, help='width of the Gaussian used in the RDF')
    parser.add_argument('--n_width', type=int, default=128, help='number of Gaussian functions used in the RDF')
    parser.add_argument('--n_layers', type=int, default=3, help='number of hidden layers in the neural network potential')
    parser.add_argument('--nonlinear', type=str, default='ELU', help='type of nonlinearity used in the neural network potential')

    parser.add_argument('--rdf_loss_weight', type=float, default=1, help='coefficient in front of RDF loss term')
    parser.add_argument('--diffusion_loss_weight', type=float, default=100, help='coefficient in front of diffusion coefficient loss term')
    parser.add_argument('--vacf_loss_weight', type=float, default=100, help='coefficient in front of VACF loss term')
    parser.add_argument('--batch_size', type=int, default=1, help='number of points along trajectory at which to measure loss')
    parser.add_argument('--loss_measure_freq', type=int, default=100, help='gap with which to measure loss along trajectory')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--restart_probability', type=float, default=1.0, help='probability of restarting from FCC vs continuing simulation')



    params = parser.parse_args()

    #GPU
    try:
        device = torch.device(torch.cuda.current_device())
    except:
        device = "cpu"

    
    #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)#, sample_frac = params.rdf_sample_frac)


    #initialize model
    mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
                'r_start': 0.0,
                'r_end': params.cutoff, 
                'n_width': params.n_width,
                'n_layers': params.n_layers,
                'nonlinear': params.nonlinear}


    NN = pairMLP(**mlp_params)

    #prior potential only contains repulsive term
    prior = LJFamily(epsilon=params.epsilon, sigma=params.sigma, rep_pow=6, attr_pow=0)

    model = Stack({'nn': NN, 'prior': prior}).to(device)
    radii_0 = fcc_positions(params.n_particle, params.box, device).unsqueeze(0).repeat(params.n_replicas, 1, 1)
    velocities_0  = initialize_velocities(params.n_particle, params.temp, n_replicas = params.n_replicas)
    rdf_0  = diff_rdf(tuple(radii_to_dists(radii_0[0].unsqueeze(0).to(device), params))).unsqueeze(0).repeat(params.n_replicas, 1)

    #load ground truth rdf and diffusion coefficient
    if params.nn:
        add = "_polylj" if params.poly else ""
        gt_dir = os.path.join('ground_truth' + add, f"n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}")
        gt_rdf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_rdf.npy"))).to(device)
        gt_diff_coeff = torch.Tensor(np.load(os.path.join(gt_dir, "gt_diff_coeff.npy"))).to(device)
        gt_vacf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_vacf.npy"))).to(device)[0:params.vacf_window]
        add = "polylj_" if params.poly else ""
        results = 'results_polylj' if params.poly else 'results'
        results_dir = os.path.join(results, f"IMPLICIT_{add}_{params.exp_name}_n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}_dt={params.dt}_ttotal={params.t_total}")

    #initialize outer loop optimizer/scheduler
    optimizer = torch.optim.Adam(list(NN.parameters()), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

    if not params.nn:
        params.n_epochs = 1

    #outer training loop
    losses = []
    rdf_losses = []
    diffusion_losses = []
    vacf_losses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    if params.nn:
        writer = SummaryWriter(log_dir = results_dir)
    
    for epoch in range(params.n_epochs):
        rdf_loss = 0
        vacf_loss = 0
        diffusion_loss = 0
        print(f"Epoch {epoch+1}")
        restart_override = epoch==0 or (torch.rand(size=(1,)) < params.restart_probability).item()
        

        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        for i in range(params.batch_size):
            #if continuing simulation, initialize with equilibriated NVT coupling constant 
            restart = i == 0 and restart_override
            #initialize simulator parameterized by a NN model
            if restart: #start from FCC lattice
                print("Initialize from FCC lattice")
                simulator = ImplicitMDSimulator(params, model, radii_0, velocities_0, rdf_0)
            else: #continue from where we left off in the last epoch/batch
                simulator.reset(last_radii, last_velocities, last_rdf)
            
            top_level = Stochastic_IFT()
            start = time.time()
            equilibriated_simulator, grads = top_level.apply(simulator, gt_rdf, params)
            end = time.time()
            #import pdb; pdb.set_trace()
            #manual gradient update
            for param, grad in zip(model.parameters(), grads):
                param.data -= params.lr*grad
            # simulator.nsteps = np.rint(params.t_total/params.dt).astype(np.int32) if restart else max(params.vacf_window, params.loss_measure_freq)
            # if not restart:
            #     simulator.zeta = equilibriated_simulator.zeta
            # start = time.time()
            # equilibriated_simulator = simulator.solve()
            # end = time.time()
            sim_time = end-start
            print("MD simulation time (s): ", sim_time)
            
            #compute loss at the end of the trajectory
            if params.nn:
                rdf_loss += (equilibriated_simulator.rdf - gt_rdf).pow(2).mean() / params.batch_size
                diffusion_loss += (equilibriated_simulator.diff_coeff - gt_diff_coeff).pow(2).mean() / params.batch_size# if params.diffusion_loss_weight != 0 else torch.Tensor([0.]).to(device)
                vacf_loss += (equilibriated_simulator.vacf - gt_vacf).pow(2).mean() / params.batch_size# if params.vacf_loss_weight != 0 else torch.Tensor([0.]).to(device)

            #memory cleanup
            #last_radii, last_velocities, last_rdf = equilibriated_simulator.cleanup()
        simulator.f.close()
        
        outer_loss = params.rdf_loss_weight*rdf_loss + \
                    params.diffusion_loss_weight*diffusion_loss + \
                    params.vacf_loss_weight*vacf_loss
        print(f"Loss: RDF={params.rdf_loss_weight*rdf_loss.item()}+Diffusion={params.diffusion_loss_weight*diffusion_loss.item()}+VACF={params.vacf_loss_weight*vacf_loss.item()}={outer_loss.item()}")
        #compute (implicit) gradient of outer loss wrt model parameters
        start = time.time()
        #torch.autograd.backward(tensors = outer_loss, inputs = list(model.parameters()))
        end = time.time()
        grad_time = end-start
        print("gradient calculation time (s): ",  grad_time)

        #equilibriated_simulator.cleanup()

        #print_active_torch_tensors()
        torch.cuda.empty_cache()
        gc.collect()
        max_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
                if  norm > max_norm:
                    max_norm = norm
        try:
            print("Max norm: ", max_norm.item())
        except AttributeError:
            print("Max norm: ", max_norm)

        #optimizer.step()
        scheduler.step(outer_loss)
        #log stats
        losses.append(outer_loss.item())
        rdf_losses.append(rdf_loss.item())
        diffusion_losses.append((diffusion_loss.sqrt()/gt_diff_coeff).item())
        vacf_losses.append(vacf_loss.item())
        sim_times.append(sim_time)
        grad_times.append(grad_time)
        try:
            grad_norms.append(max_norm.item())
        except:
            grad_norms.append(max_norm)

        writer.add_scalar('Loss', losses[-1], global_step=epoch+1)
        writer.add_scalar('RDF Loss', rdf_losses[-1], global_step=epoch+1)
        writer.add_scalar('Relative Diffusion Loss', diffusion_losses[-1], global_step=epoch+1)
        writer.add_scalar('VACF Loss', vacf_losses[-1], global_step=epoch+1)
        writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Time', grad_times[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
    
            

    
    if params.nn:
        stats_write_file = os.path.join(simulator.save_dir, 'stats.txt')
        with open(stats_write_file, "w") as output:
            output.write("Losses: " + str(losses) + "\n")
            output.write("Simulation times: " +  str(sim_times) + "\n")
            output.write("Gradient calculation times: " +  str(grad_times) + "\n")
            output.write("Max gradient norms: " + str(grad_norms))

        writer.close()
    print('Done!')
    


