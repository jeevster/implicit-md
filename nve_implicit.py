from itertools import product
import concurrent.futures

import numpy as np
import gsd.hoomd
import torch
import torch.nn as nn
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from YParams import YParams
import argparse
import threading
import os
from tqdm import tqdm
import pstats
import pdb
import random
from contextlib import nullcontext
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF, msd, DiffusionCoefficient
import torchopt
from torchopt.nn import ImplicitMetaGradientModule
import time
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
from functorch import vmap



from utils import radii_to_dists, fcc_positions, initialize_velocities, dump_params_to_yml, powerlaw_inv_cdf



class ImplicitMDSimulator(ImplicitMetaGradientModule, linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0)):
    def __init__(self, params, model, radii_0, velocities_0, rdf_0):
        super(ImplicitMDSimulator, self).__init__()
        
        

        self.params = params
        self.n_particle = params.n_particle
        self.temp = params.temp
        self.kbt0 = params.kbt0
        self.box = params.box
        self.pbc = not params.no_pbc
        self.dt = params.dt
        self.t_total = params.t_total

        #Nose-Hoover Thermostat stuff
        self.nvt_time = params.nvt_time
        self.targeEkin = 0.5 * (3.0 * self.n_particle) * self.temp
        #self.Q = 3.0 * self.n_particle * self.temp * (self.t_total * self.dt)**2
        self.Q = 3.0 * self.n_particle * self.temp * (50 * self.dt)**2 #found that this setting works well


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

        self.cutoff = params.cutoff
        self.gaussian_width = params.gaussian_width
        self.n_width = params.n_width
        self.n_layers = params.n_layers
        self.nonlinear = params.nonlinear

        self.c_0 = -28*self.epsilon / (self.cutoff**12)
        self.c_2 = 48*self.epsilon / (self.cutoff**14)
        self.c_4 = -21*self.epsilon / (self.cutoff**16)


        #parallelization stuff
        self.num_threads = params.num_threads
        self.bin_size = self.cutoff
        self.num_cols = int(np.ceil(self.box / self.bin_size))
        self.num_bins = self.num_cols**3
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)
       
        self.neighbor_bin_mappings = self.get_neighbor_bin_mappings()
        

        # Constant box properties
        self.vol = self.box**3.0
        self.rho = self.n_particle/self.vol

        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        self.zeros = torch.zeros((1, 1, 3)).to(self.device)
        
        #limit CPU usage
        torch.set_num_threads(1)

        #Register inner parameters
        self.model = model.to(self.device)
        #model.train()
        self.radii = nn.Parameter(radii_0.clone().detach_(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities_0.clone().detach_(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(rdf_0.clone().detach_(), requires_grad=True).to(self.device)
        self.diff_coeff = nn.Parameter(torch.Tensor([0.]), requires_grad=True).to(self.device)

        

        if self.poly: #get per-particle sigmas
            u = torch.rand(self.n_particle)
            self.particle_sigmas = powerlaw_inv_cdf(u, power = self.poly_power, y_min = self.min_sigma).clip(max = 1/0.45 * self.min_sigma).to(self.device)
            self.particle_sigmas_cpu = powerlaw_inv_cdf(u, power = self.poly_power, y_min = self.min_sigma).clip(max = 1/0.45 * self.min_sigma).cpu()
            sigma1 = self.particle_sigmas.unsqueeze(0)
            sigma2 = self.particle_sigmas.unsqueeze(1)
            self.get_sigma_pairs = lambda sigma1, sigma2: (1/2 * (sigma1 + sigma2)*(1 - self.epsilon*torch.abs(sigma1 - sigma2)))
            self.sigma_pairs = (1/2 * (sigma1 + sigma2)*(1 - self.epsilon*torch.abs(sigma1 - sigma2)))
            #self.sigma_pairs = self.sigma_pairs[~torch.eye(self.sigma_pairs.shape[0],dtype=bool)].reshape(self.sigma_pairs.shape[0], -1, 1).to(self.device)
            
        

        #define differentiable rdf and diffusion coefficient functions
        self.diff_rdf = DifferentiableRDF(params, self.device)
        self.diff_rdf_cpu = DifferentiableRDF(params, "cpu")

        self.diffusion_coefficient = DiffusionCoefficient(params, self.device)

        
        #check if inital momentum is zero
        #initial_props = self.calc_properties()
        

        if self.nn:
            add = "polylj_" if self.poly else ""
            self.save_dir = os.path.join('results', f"IMPLICIT_{add}_{self.exp_name}_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_dt={self.dt}_ttotal={self.t_total}")
        else: 
            add = "_polylj" if self.poly else ""
            self.save_dir = os.path.join('ground_truth_parallel' + add, f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}")


        

        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)

        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")
        self.t = gsd.hoomd.open(name=f'{self.save_dir}/sim_temp{self.temp}.gsd', mode='wb') 
        self.n_dump = params.n_dump # dump for configuration

    def check_symmetric(self, a, mode, tol=1e-4):
        if mode == 'opposite':
            return np.all(np.abs(a + a.T) < tol)
        
        return np.all(np.abs(a - a.T) < tol)

    def divide_items(self, bins, threads, exact=False):
        """Divide the given items among threads threads or processes. If exact,
        threads must evenly divide the number of items. Returns a list of particle
        lists for each thread.

        >>> divide_items([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> divide_items([1, 2, 3, 4, 5, 6, 7, 8], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]
        """
        num = len(bins) // threads
        rem = len(bins) % threads
        if exact and rem:
            raise ValueError("threads don't evenly divide particles")

        divided = []
        for i in range(threads):
            start = num * i + (i if i < rem else rem)
            end = start + num + (1 if i < rem else 0)
            divided.append(bins[start:end])

        return divided

    # Function to dump simulation frame that is readable in Ovito
    # Also stores radii and velocities in a compressed format which is nice
    def create_frame(self, frame):
        # Particle positions, velocities, diameter
            
        radii = self.radii.detach()
        partpos = radii.tolist()
        velocities = self.velocities.detach().tolist()
        sigmas = self.particle_sigmas_cpu if self.poly else self.sigma
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
                "Momentum Magnitude": torch.norm(torch.sum(self.velocities, axis =0)).item()}


    '''Returns bin indices of all particles '''
    def bin_index(self, radii):
        temp_radii = radii + self.box / 2
        bins = torch.floor(temp_radii / self.bin_size)
        return (bins[:, 0] + self.num_cols*bins[:, 1] + self.num_cols**2 * bins[:, 2]).to(int)
    
    """ Returns list mapping bin id to particles for in the bin's domain """
    def assign_bins(self, radii):
        bin_idxs = self.bin_index(radii)
        bin_to_particles = []
        for i in range(self.num_bins):
            bin_to_particles.append(torch.nonzero(bin_idxs == i).squeeze(-1))

        return bin_to_particles

    ''''Gets the neighbor bin indices for every bin'''
    def get_neighbor_bin_mappings(self):
        neighbor_bin_mappings = {}
        for bin in range(self.num_bins):
            x = bin % self.num_cols
            y = (bin % self.num_cols**2 - x) / self.num_cols
            z = (bin - x - self.num_cols * y) / self.num_cols**2

            neighbor_coords = torch.Tensor([[x - 1, x, x + 1], [y - 1, y, y + 1], [z - 1, z, z + 1]]) % self.num_cols
            neighbor_bin_idxs = [int(x_ + self.num_cols * y_ + self.num_cols**2 * z_) for x_, y_, z_ in product(*neighbor_coords)]
            neighbor_bin_mappings[bin] = list(set(neighbor_bin_idxs))
        return neighbor_bin_mappings
        
        
    '''CORE MD OPERATIONS'''

    def top_level_force_calc(self, radii, bins):

        #partition the radii, sigma pairs, and bin_idxs among threads
        divided_radii = [radii[idxs] for idxs in bins]
        thread_partitioned_radii = self.divide_items(divided_radii, self.num_threads)
        divided_sigmas = [self.particle_sigmas[idxs] for idxs in bins] if self.poly else [self.sigma * torch.ones((len(idxs),)) for idxs in bins]
        thread_partitioned_sigmas = self.divide_items(divided_sigmas, self.num_threads)
        thread_partitioned_bin_idxs = self.divide_items(range(self.num_bins), self.num_threads)

        # Define a list to store the future objects representing the results of each thread
        futures = []

        # Submit tasks to the thread pool and store the future objects
        for i in range(self.num_threads):

            radii = thread_partitioned_radii[i]
            neighbor_radii = [torch.concat([divided_radii[j] \
                                for j in self.neighbor_bin_mappings[bin_idx]], dim=0) \
                                 for bin_idx in thread_partitioned_bin_idxs[i]]

            sigmas = thread_partitioned_sigmas[i]
            neighbor_sigmas = [torch.concat([divided_sigmas[j] \
                                for j in self.neighbor_bin_mappings[bin_idx]], dim=0) \
                                 for bin_idx in thread_partitioned_bin_idxs[i]]

            future = self.executor.submit(self.thread_force_calc, radii, neighbor_radii, sigmas, neighbor_sigmas)
                                    
            futures.append(future)

        # Retrieve the results from the future objects
        results = [future.result() for future in futures]

        #Scatter the forces in the correct order, sum the energies
        scatter_idxs = torch.cat(bins)
        forces_aggregated = torch.cat([result[1] for result in results])
        forces = torch.scatter(forces_aggregated, 0, scatter_idxs.unsqueeze(-1), forces_aggregated)
        energy = torch.cat([result[0].unsqueeze(-1) for result in results]).sum()

        return energy, forces

    def thread_force_calc(self, all_current_radii, all_neighbor_radii, all_current_sigmas, all_neighbor_sigmas):
        #import pdb; pdb.set_trace()
        with torch.enable_grad():
            
            rs = [current_radii.unsqueeze(1) - neighbor_radii.unsqueeze(0) for current_radii, neighbor_radii in zip(all_current_radii, all_neighbor_radii)]
            rs = [-1*torch.where(r > 0.5*self.box, r-self.box, torch.where(r<-0.5*self.box, r+self.box, r)) for r in rs]
            r = torch.stack([torch.block_diag(*[r[:, :, i] for r in rs]) for i in range(3)], dim=2)
            if not r.requires_grad:
                    r.requires_grad = True

            vectorized_dists = torch.sqrt(torch.sum(r**2, axis=2))
            mask = (vectorized_dists == 0) | (vectorized_dists > self.cutoff) #remove particles beyond cutoff and remove ourself
            vectorized_dists = vectorized_dists[~mask].unsqueeze(-1)

            if self.poly:
                sigma_pairs = [self.get_sigma_pairs(current_sigmas.unsqueeze(1), neighbor_sigmas.unsqueeze(0)) for current_sigmas, neighbor_sigmas in zip(all_current_sigmas, all_neighbor_sigmas)]
                vectorized_sigma_pairs = torch.block_diag(*sigma_pairs)
                vectorized_sigma_pairs = vectorized_sigma_pairs[~mask].unsqueeze(-1)

            #compute energy
            if self.nn:
                energy = self.model(vectorized_dists/vectorized_sigma_pairs) if self.poly else self.model(vectorized_dists)
                
            #LJ potential
            else:
                if self.poly: #repulsive
                    parenth = (vectorized_sigma_pairs/vectorized_dists)                    
                    rep_term = parenth ** self.rep_power
                    energy = torch.sum(rep_term + self.c_0 + self.c_2*parenth**-2 + self.c_4*parenth**-4)
                else:
                    r2i = (self.sigma/vectorized_dists)**2
                    r6i = r2i**3
                    energy = (2*self.epsilon*torch.sum(r6i*(r6i - 1))).unsqueeze(-1)
                    
        #not sure why we have to do this
        forces = 2*compute_grad(inputs=r, output=energy)
        
        #assert no nans except for the mask places
        assert(not torch.any(torch.isnan(forces[~mask])))

        #remove nans
        forces[forces != forces] = 0.


        #sum forces across particles
        return energy, torch.sum(forces, axis = 1)#.to(self.device)
        

    def forward_nvt(self, radii, velocities, forces, zeta, calc_rdf = False, calc_diffusion = False):
        # get current acceleration
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
        KE_0 = torch.sum(torch.square(velocities)) / 2
        
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)

        #assign new bins
        bins = self.assign_bins(radii)
        #compute energy and forces and aggregate them
        
        #breaking here on the second iteration
        energy, forces = self.top_level_force_calc(radii, bins)
        
        accel = forces

        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * \
            (1/self.Q) * (KE_0 - self.targeEkin)

        #get updated KE
        ke = torch.sum(torch.square(velocities)) / 2

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
                new_dists = new_dists/ self.sigma_pairs[~torch.eye(new_dists.shape[0],dtype=bool)].reshape(new_dists.shape[0], -1, 1)

        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device))) if calc_rdf else 0 #calculate the RDF from a single frame
        #new_rdf = 0

        if calc_diffusion:
            self.last_h_radii.append(radii.unsqueeze(0))

        # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not self.nn:
                new_dists = radii_to_dists(radii, self.params)
                if self.poly:
                    #normalize distances by sigma pairs
                    new_dists = new_dists/ self.sigma_pairs[~torch.eye(new_dists.shape[0],dtype=bool)].reshape(new_dists.shape[0], -1, 1)
                self.running_dists.append(new_dists.cpu().detach())



        return radii, velocities, forces, zeta, new_rdf
        
        
    def forward(self, radii, velocities, forces, calc_rdf = False, calc_diffusion = False):
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

        #assign new bins
        bins = self.assign_bins(radii)

        #compute energy and forces and aggregate them
        energy, forces = self.top_level_force_calc(radii, bins)

    
        #another half-step in velocity
        velocities = (velocities + 0.5*self.dt*forces) 
        #props = self.calc_properties()

        if calc_rdf:
            new_dists = radii_to_dists(radii, self.params)
            if self.poly:
                #normalize distances by sigma pairs
                new_dists = new_dists/ self.sigma_pairs[~torch.eye(new_dists.shape[0],dtype=bool)].reshape(new_dists.shape[0], -1, 1)
        
        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device))) if calc_rdf else 0 #calculate the RDF from a single frame
        #new_rdf = 0
        if calc_diffusion:
            self.last_h_radii.append(radii.unsqueeze(0))

        # dump frame
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not self.nn:
                new_dists = radii_to_dists(radii, self.params)
                if self.poly:
                    #normalize distances by sigma pairs
                    new_dists = new_dists/ self.sigma_pairs[~torch.eye(new_dists.shape[0],dtype=bool)].reshape(new_dists.shape[0], -1, 1)
                self.running_dists.append(new_dists.cpu().detach())

        return radii, velocities, forces, new_rdf # return the new distance matrix 



    def optimality(self):
        # Stationary condition construction for calculating implicit gradient
        #print("optimality")
        #Stationarity of the RDF - doesn't change if we do another step of MD
        
        with torch.enable_grad():
            #compute current diffusion coefficient
            msd_data = msd(torch.cat(self.last_h_radii, dim=0), self.box)
            old_diffusion_coeff = self.diffusion_coefficient(msd_data)

            #make an MD step
            forces = self.force_calc(self.radii)[1]
            new_radii, new_velocities, _, _, new_rdf = self.forward_nvt(self.radii, self.velocities, forces, self.zeta_final, calc_rdf = True, calc_diffusion = True)

            #compute new diffusion coefficient
            new_msd_data = msd(torch.cat(self.last_h_radii[1:], dim=0), self.box)
            new_diffusion_coeff = self.diffusion_coefficient(new_msd_data)

        radii_residual  = self.radii - new_radii
        velocity_residual  = self.velocities - new_velocities
        rdf_residual = self.rdf - new_rdf
        diffusion_residual = (new_diffusion_coeff - old_diffusion_coeff).unsqueeze(0)
        
        return (radii_residual, velocity_residual, rdf_residual, diffusion_residual)


    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        self.running_dists = []
        self.last_h_radii = []
        
        #Initialize forces/potential of starting configuration
        with torch.no_grad():
            
            #assign new bins
            bins = self.assign_bins(self.radii)

            #compute energy and forces and aggregate them
            energy, forces = self.top_level_force_calc(self.radii, bins)
            self.forces = forces
            #
            #Run MD
            print("Start MD trajectory", file=self.f)
            zeta = 0
            for step in tqdm(range(self.nsteps)):
                self.step = step
                
                calc_rdf = step ==  self.nsteps -1 or (self.save_intermediate_rdf and not self.nn)
                calc_diffusion = step >= self.nsteps - self.diffusion_window #calculate diffusion coefficient if within window from the end

                with torch.enable_grad() if calc_diffusion else nullcontext():
                    if step < self.n_nvt_steps: #NVT step
                        radii, velocities, forces, zeta, rdf = self.forward_nvt(self.radii, self.velocities, forces, zeta, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion)
                    else: # NVE step
                        if step == self.n_nvt_steps:
                            print("NVT Equilibration Complete, switching to NVE", file=self.f)
                        radii, velocities, forces, rdf = self.forward(self.radii, self.velocities, forces, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion)

                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.rdf.copy_(rdf)
                self.diff_coeff.copy_(0) #placeholder for now, store final value at the end
                self.forces = forces # temp for debugging
                if not self.nn and self.save_intermediate_rdf and step % self.n_dump == 0:
                    filename = f"step{step+1}_rdf.npy"
                    np.save(os.path.join(self.save_dir, filename), self.rdf.cpu().detach().numpy())

        
        self.f.close()
        length = len(self.running_dists)

        #compute diffusion coefficient
        msd_data = msd(torch.cat(self.last_h_radii, dim=0), self.box)
        diffusion_coeff = self.diffusion_coefficient(msd_data)
        self.diff_coeff.copy_(diffusion_coeff)
        self.zeta_final = zeta
        # filename ="gt_diff_coeff.npy" if not self.nn else f"diff_coeff_epoch{epoch+1}.npy"
        # np.save(os.path.join(self.save_dir, filename), diffusion_coeff.cpu().detach().numpy())

        # #compute ground truth rdf over entire trajectory (do it on CPU to avoid memory issues)
        # save_rdf = self.diff_rdf_cpu(self.running_dists[int(self.burn_in_frac*length):]) if not self.nn else self.rdf
        # filename ="gt_rdf.npy" if not self.nn else f"rdf_epoch{epoch+1}.npy"
        # np.save(os.path.join(self.save_dir, filename), save_rdf.cpu().detach().numpy())

        return self

    
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
    parser.add_argument('--dr', type=float, default=0.01, help='bin size for RDF calculation (non-differentiable version)')
    parser.add_argument('--rdf_sample_frac', type=float, default=1, help='fraction of particles to sample for RDF')
    parser.add_argument('--diffusion_window', type=int, default=100, help='number of timesteps on which to fit linear regression for estimating diffusion coefficient')

    parser.add_argument('--t_total', type=float, default=5, help='total time')
    parser.add_argument('--nvt_time', type=float, default=5, help='time for NVT equilibration')

    parser.add_argument('--diameter_viz', type=float, default=0.3, help='particle diameter for Ovito visualization')
    parser.add_argument('--n_dump', type=int, default=10, help='save frequency of configurations (also frequency of frames used for ground truth RDF calculation)')
    parser.add_argument('--save_intermediate_rdf', action = 'store_true', help='Whether to store the RDF along the trajectory for the ground truth')
    parser.add_argument('--burn_in_frac', type=float, default=0.2, help='initial fraction of trajectory to discount when calculating ground truth rdf')
    parser.add_argument('--exp_name', type=str, default = "", help='name of experiment - used as prefix of results folder name')

    #learnable potential stuff
    parser.add_argument('--n_epochs', type=int, default=30, help='number of outer loop training epochs')
    parser.add_argument('--nn', action='store_true', help='use neural network potential')
    parser.add_argument('--cutoff', type=float, default=2.5, help='LJ cutoff distance')
    parser.add_argument('--gaussian_width', type=float, default=0.1, help='width of the Gaussian used in the RDF')
    parser.add_argument('--n_width', type=int, default=128, help='number of Gaussian functions used in the RDF')
    parser.add_argument('--n_layers', type=int, default=3, help='number of hidden layers in the neural network potential')
    parser.add_argument('--nonlinear', type=str, default='ELU', help='type of nonlinearity used in the neural network potential')

    parser.add_argument('--rdf_loss_weight', type=float, default=1, help='coefficient in front of RDF loss term')
    parser.add_argument('--diffusion_loss_weight', type=float, default=100, help='coefficient in front of diffusion coefficient loss term')

    parser.add_argument('--num_threads', type=int, default=1, help='# of threads')


    params = parser.parse_args()

    #Set random seeds
    np.random.seed(seed=params.seed)
    torch.manual_seed(params.seed)
    random.seed(params.seed)

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

    model = Stack({'nn': NN, 'prior': prior})
    radii_0 = fcc_positions(params.n_particle, params.box, device)
    velocities_0  = initialize_velocities(params.n_particle, params.temp)
    rdf_0  = diff_rdf(tuple(radii_to_dists(radii_0.to(device), params)))

    #load ground truth rdf and diffusion coefficient
    if params.nn:
        add = "_polylj" if params.poly else ""
        gt_dir = os.path.join('ground_truth_parallel' + add, f"n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}")
        gt_rdf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_rdf.npy"))).to(device)
        gt_diff_coeff = torch.Tensor(np.load(os.path.join(gt_dir, "gt_diff_coeff.npy"))).to(device)
        add = "polylj_" if params.poly else ""
        results_dir = os.path.join('results', f"IMPLICIT_{add}_{params.exp_name}_n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}_dt={params.dt}_ttotal={params.t_total}")

    #initialize outer loop optimizer/scheduler
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    if not params.nn:
        params.n_epochs = 1

    #outer training loop
    losses = []
    rdf_losses = []
    diffusion_losses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    if params.nn:
        writer = SummaryWriter(log_dir = results_dir)
    
    for epoch in range(params.n_epochs):
        print(f"Epoch {epoch+1}")
        #initialize simulator parameterized by a NN model
        simulator = ImplicitMDSimulator(params, model, radii_0, velocities_0, rdf_0)

        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        start = time.time()
        equilibriated_simulator = simulator.solve()
        end = time.time()
        sim_time = end-start
        print("MD simulation time (s): ",  sim_time)
        
        #compute loss at the end of the trajectory
        if params.nn:
            rdf_loss = (equilibriated_simulator.rdf - gt_rdf).pow(2).mean()
            diffusion_loss = (equilibriated_simulator.diff_coeff - gt_diff_coeff).pow(2).mean()
            outer_loss = params.rdf_loss_weight*rdf_loss + params.diffusion_loss_weight*diffusion_loss
                            
            print("Loss: ", outer_loss.item())
            #compute (implicit) gradient of outer loss wrt model parameters
            start = time.time()
    #         torch.autograd.backward(tensors = outer_loss, inputs = list(model.parameters()))
    #         end = time.time()
    #         grad_time = end-start
    #         print("gradient calculation time (s): ",  grad_time)

    #         max_norm = 0
    #         for param in model.parameters():
    #             if param.grad is not None:
    #                 norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
    #                 if  norm > max_norm:
    #                     max_norm = norm
    #         try:
    #             print("Max norm: ", max_norm.item())
    #         except AttributeError:
    #             print("Max norm: ", max_norm)

    #         optimizer.step()
    #         scheduler.step(outer_loss)
    #         #log stats
    #         losses.append(outer_loss.item())
    #         rdf_losses.append(rdf_loss.item())
    #         diffusion_losses.append((diffusion_loss.sqrt()/gt_diff_coeff).item())
    #         sim_times.append(sim_time)
    #         grad_times.append(grad_time)
    #         grad_norms.append(max_norm.item())

    #         writer.add_scalar('Loss', losses[-1], global_step=epoch+1)
    #         writer.add_scalar('RDF Loss', rdf_losses[-1], global_step=epoch+1)
    #         writer.add_scalar('Relative Diffusion Loss', diffusion_losses[-1], global_step=epoch+1)
    #         writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
    #         writer.add_scalar('Gradient Time', grad_times[-1], global_step=epoch+1)
    #         writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
    
    # if params.nn:
    #     stats_write_file = os.path.join(simulator.save_dir, 'stats.txt')
    #     with open(stats_write_file, "w") as output:
    #         output.write("Losses: " + str(losses) + "\n")
    #         output.write("Simulation times: " +  str(sim_times) + "\n")
    #         output.write("Gradient calculation times: " +  str(grad_times) + "\n")
    #         output.write("Max gradient norms: " + str(grad_norms))

    #     writer.close()
    print('Done!')
    
    

