import numpy as np
import gsd.hoomd
import torch
from pathlib import Path
import torch.nn as nn
import math
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from YParams import YParams
from nequip.train.trainer import Trainer
import types

import argparse
import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
from itertools import product
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
from functorch import vmap, vjp
from utils import radii_to_dists, fcc_positions, initialize_velocities, \
                    dump_params_to_yml, powerlaw_inv_cdf, print_active_torch_tensors, plot_pair, solve_continuity_system, find_hr_from_file, distance_pbc

import warnings
warnings.filterwarnings("ignore")

#NNIP stuff:
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from nequip.ase.nequip_calculator import nequip_calculator

from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, Trajectory
from ase.neighborlist import natural_cutoffs, NeighborList

import mdsim.md.integrator as md_integrator
from mdsim.common.registry import registry
from mdsim.common.utils import setup_imports, setup_logging, compute_bond_lengths, load_schnet_model, data_to_atoms, atoms_to_batch, atoms_to_state_dict
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.datasets import data_list_collater
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater

from mdsim.common.utils import load_config

from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg
)
from mdsim.common.flags import flags


class ImplicitMDSimulator():
    def __init__(self, config, params, model, model_config):
        super(ImplicitMDSimulator, self).__init__()
        self.params = params

    
        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

    
        self.name = config['dataset']['name']
        self.molecule = config['dataset']['molecule']
        self.size = config['dataset']['size']
        self.model_type = config['dataset']['model']

        logging.info("Initializing MD simulation environment")
        self.config = config
        self.data_dir = config['dataset']['src']
        self.model_dir = os.path.join(config['dataset']["model_dir"], self.model_type, f"{self.name}-{self.molecule}_{self.size}_{self.model_type}")
        self.save_name = config['dataset']["save_name"]
        self.train = config['mode'] == 'train'
        self.n_replicas = config["ift"]["n_replicas"]
        self.minibatch_size = config["ift"]['minibatch_size']
        self.shuffle = config["ift"]['shuffle']
        self.vacf_window = config["ift"]["vacf_window"]
        self.no_ift = config["ift"]["no_ift"]

        #initialize datasets
        self.train_dataset = LmdbDataset({'src': os.path.join(self.data_dir, self.name, self.molecule, self.size, 'train')})
        self.valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], self.name, self.molecule, self.size, 'val')})

        #get random initial condition from dataset
        length = self.train_dataset.__len__()
        init_data = self.train_dataset.__getitem__(10)#np.random.randint(0, length))
        self.n_atoms = init_data['pos'].shape[0]
        self.atoms = data_to_atoms(init_data)

        #extract bond and atom type information
        NL = NeighborList(natural_cutoffs(self.atoms), self_interaction=False)
        NL.update(self.atoms)
        self.bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(self.device).T
        self.atom_types_list = list(set(self.atoms.get_chemical_symbols()))
        self.atom_types = self.atoms.get_chemical_symbols()
        type_to_index = {value: index for index, value in enumerate(self.atom_types_list)}
        self.typeid = np.zeros(self.n_atoms, dtype=int)
        for i, _type in enumerate(self.atom_types):
            self.typeid[i] = type_to_index[_type]    

        #bond length deviation
        DATAPATH = f'{self.data_dir}/{self.name}/{self.molecule}/{self.size}/test/nequip_npz.npz'
        gt_data = np.load(DATAPATH)
        self.gt_traj = torch.FloatTensor(gt_data.f.R).to(self.device)
        self.gt_energies = torch.FloatTensor(gt_data.f.E).to(self.device)
        #normalize
        self.gt_energies = (self.gt_energies - self.gt_energies.mean()) / self.gt_energies.std()
        self.gt_forces = torch.FloatTensor(gt_data.f.F).to(self.device)
        self.mean_bond_lens = distance_pbc(
        self.gt_traj[:, self.bonds[:, 0]], self.gt_traj[:, self.bonds[:, 1]], \
                    torch.FloatTensor([30., 30., 30.]).to(self.device)).mean(dim=0)
        
        #Nose-Hoover Thermostat stuff
        self.dt = config["ift"]['integrator_config']["timestep"] * units.fs
        self.temp = config["ift"]['integrator_config']["temperature"]
        self.integrator = self.config['ift']["integrator"]
        # adjust units.
        if self.integrator in ['NoseHoover', 'NoseHooverChain', 'Langevin']:
            self.temp *= units.kB
        self.targeEkin = 0.5 * (3.0 * self.n_atoms) * self.temp
        self.ttime = config["ift"]["integrator_config"]["ttime"]
        self.Q = 3.0 * self.n_atoms * self.temp * (self.ttime * self.dt)**2
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
        self.masses = torch.Tensor(self.atoms.get_masses().reshape(-1, 1)).repeat(self.n_replicas, 1, 1).to(self.device)


        #Langevin thermostat stuff
        self.gamma = config["ift"]["integrator_config"]["gamma"] / (1000*units.fs)
        self.noise_f = (2.0 * self.gamma/self.masses * self.temp * self.dt).sqrt().to(self.device)

        self.nsteps = params.steps
        self.eq_steps = params.eq_steps
        #ensure that the number of logged steps is a multiple of the vacf window (for chopping up the trajectory)
        self.nsteps -= (self.nsteps - self.eq_steps) % self.vacf_window
        if (self.nsteps - self.eq_steps) < self.vacf_window:
            self.nsteps = self.eq_steps + 2*self.vacf_window #at least two windows
        while self.nsteps < params.steps: #nsteps should be at least as long as what was requested
            self.nsteps += self.vacf_window
        
        #Initialize model (passed in as an argument to make it a meta parameter)
        self.model = model
    
        self.model_config = model_config

        self.atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(self.n_replicas)
        self.batch = torch.arange(self.n_replicas).repeat_interleave(self.n_atoms).to(self.device)
        self.ic_stddev = params.ic_stddev

        #Register inner parameters
        samples = np.random.choice(np.arange(self.train_dataset.__len__()), self.n_replicas)
        actual_atoms = [self.train_dataset.__getitem__(i) for i in samples]
        radii = torch.stack([torch.Tensor(data_to_atoms(atoms).get_positions()) for atoms in actual_atoms])
        self.dummy_param = nn.Parameter(torch.Tensor([0.]))
        self.radii = (radii + torch.normal(torch.zeros_like(radii), self.ic_stddev)).to(self.device)
        self.velocities = torch.Tensor(initialize_velocities(self.n_atoms, self.masses, self.temp, self.n_replicas)).to(self.device)

        self.diameter_viz = params.diameter_viz
        
        self.nn = params.nn
        self.save_intermediate_rdf = params.save_intermediate_rdf
        self.exp_name = params.exp_name

        self.rdf_loss_weight = params.rdf_loss_weight
        self.diffusion_loss_weight = params.diffusion_loss_weight
        self.vacf_loss_weight = params.vacf_loss_weight
        
        #limit CPU usage
        torch.set_num_threads(10)

        #define vectorized differentiable rdf and vacf
        self.diff_rdf = vmap(DifferentiableRDF(params, self.device), -1)
        self.diff_vacf = vmap(DifferentiableVACF(params, self.device))
        # #vectorize over dim 1 (the input will be of shape [diffusion_window , n_replicas])
        # self.diffusion_coefficient = vmap(DiffusionCoefficient(params, self.device) , 1)
    
        if self.nn:
            results = 'results_nnip'
            name = f"{self.molecule}_{params.exp_name}" if self.no_ift else f"IMPLICIT_{self.molecule}_{params.exp_name}"
            self.save_dir = os.path.join(results, name)
        
        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)

        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")
        self.t = gsd.hoomd.open(name=f'{self.save_dir}/sim_temp.gsd', mode='w') 
        self.n_dump = params.n_dump # dump for configuration

    '''compute energy/force error on held-out test set'''
    def energy_force_error(self, batch_size):
        with torch.no_grad():
            num_batches = math.ceil(self.gt_traj.shape[0]/ batch_size)
            energies = []
            forces = []
            for i in range(num_batches):
                #print_active_torch_tensors()
                start = batch_size*i
                end = batch_size*(i+1)
                actual_batch_size = min(end, self.gt_traj.shape[0]) - start        
                atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(actual_batch_size)
                batch = torch.arange(actual_batch_size).repeat_interleave(self.n_atoms).to(self.device)
                energy, force = self.force_calc(self.gt_traj[start:end], atomic_numbers, batch) 
                energies.append(energy.detach())
                forces.append(force.detach())
            energies = torch.cat(energies)
            #normalize energies
            energies = (energies - energies.mean()) / energies.std()
            forces = torch.cat(forces)
            energy_mae = (self.gt_energies - energies).abs().mean()
            energy_rmse = (self.gt_energies - energies).pow(2).mean().sqrt()
            force_rmse = (self.gt_forces - forces).pow(2).mean().sqrt()
            force_mae = (self.gt_forces - forces).abs().mean()
            energies = 0
            forces = 0
            return energy_rmse, force_rmse

    '''memory cleanups'''
    def cleanup(self):        
        last_radii = self.radii
        last_velocities = self.velocities
        #de-register all the parameters to kill the gradients
        for name, _ in self.named_parameters():
            self.register_parameter(name, None)
        return last_radii, last_velocities

    def reset(self, radii, velocities):
        self.radii = nn.Parameter(radii.detach_().clone(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(velocities.detach_().clone(), requires_grad=True).to(self.device)
        # self.diff_coeff = nn.Parameter(torch.zeros((self.n_replicas,)), requires_grad=True).to(self.device)
        # self.vacf = nn.Parameter(torch.zeros((self.n_replicas,self.vacf_window)), requires_grad=True).to(self.device)

    def force_calc(self, radii, atomic_numbers = None, batch = None, retain_grad = False):
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        if batch is None:
            batch = self.batch
        with torch.enable_grad():
            if not radii.requires_grad:
                    radii.requires_grad = True
            energy = self.model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)

            forces = -compute_grad(inputs = radii, output = energy) if retain_grad else -compute_grad(inputs = radii, output = energy).detach()
            assert(not torch.any(torch.isnan(forces)))
            return energy, forces


    def forward_nvt(self, radii, velocities, forces, zeta, retain_grad=False):
        # get current accelerations
        accel = forces / self.masses

        # make full step in position 
        radii = radii + velocities * self.dt + \
            (accel - zeta * velocities) * (0.5 * self.dt ** 2)

        # record current KE
        KE_0 = 1/2 * (self.masses*torch.square(velocities)).sum(axis = (1,2), keepdims=True)
        
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)

        # make a full step in accelerations
        energy, forces = self.force_calc(radii, retain_grad=retain_grad)
        accel = forces / self.masses

        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * (1/self.Q) * (KE_0 - self.targeEkin)

        #get updated KE
        ke = 1/2 * (self.masses*torch.square(velocities)).sum(axis = (1,2), keepdims=True)

        # make another halfstep in self.zeta
        zeta = zeta + 0.5 * self.dt * \
            (1/self.Q) * (ke - self.targeEkin)

        # make another half step in velocity
        velocities = (velocities + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * zeta)

        # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))

        return radii, velocities, forces, zeta
    
    def forward_langevin(self, radii, velocities, forces, retain_grad = False):
        
        #full step in position
        radii = radii.detach() + self.dt*velocities

        #calculate force at new position
        energy, forces = self.force_calc(radii, retain_grad=retain_grad)

        #full step in velocities
        velocities = velocities + self.dt*(forces/self.masses - self.gamma * velocities) + self.noise_f * torch.randn_like(velocities)

        # # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            self.t.append(self.create_frame(frame = self.step/self.n_dump))
        return radii, velocities, forces

    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        
        self.running_dists = []
        self.running_vels = []
        self.running_accs = []
        self.last_h_radii = []
        self.last_h_velocities = []
        self.running_radii = []
        self.running_energy = []
        #Initialize forces/potential of starting configuration
        with torch.enable_grad() if self.no_ift else torch.no_grad():
            self.step = -1
            _, forces = self.force_calc(self.radii, retain_grad = self.no_ift)
            if self.integrator == 'Langevin':
                #half-step outside loop to ensure symplecticity
                self.velocities = self.velocities + self.dt/2*(forces/self.masses - self.gamma*self.velocities) + self.noise_f/torch.sqrt(torch.tensor(2.0).to(self.device))*torch.randn_like(self.velocities)
            zeta = self.zeta
            #Run MD
            print("Start MD trajectory", file=self.f)

            for step in tqdm(range(self.nsteps)):
                self.step = step
                #MD Step
                if self.integrator == 'NoseHoover':
                    radii, velocities, forces, zeta = self.forward_nvt(self.radii, self.velocities, forces, zeta, retain_grad = self.no_ift)
                elif self.integrator == 'Langevin':
                    radii, velocities, forces = self.forward_langevin(self.radii, self.velocities, forces, retain_grad = self.no_ift)
                else:
                    RuntimeError("Must choose either NoseHoover or Langevin as integrator")
                
                #save trajectory for gradient calculation
                if step >= self.eq_steps:# and step % self.n_dump == 0:
                    self.running_radii.append(radii if self.no_ift else radii.detach().clone())
                    self.running_vels.append(velocities if self.no_ift else velocities.detach().clone())
                    self.running_accs.append((forces/self.masses) if self.no_ift else (forces/self.masses).detach().clone())
                if self.no_ift:
                    self.radii = radii
                    self.velocities = velocities
                else:
                    self.radii.copy_(radii)
                    self.velocities.copy_(velocities)
                    
            self.zeta = zeta
            self.forces = forces

            #compute bond length deviation
            self.stacked_radii = torch.cat(self.running_radii)
            bond_lens = distance_pbc(self.stacked_radii[:, self.bonds[:, 0]], self.stacked_radii[:, self.bonds[:, 1]], torch.FloatTensor([30., 30., 30.]).to(self.device))
            self.max_dev = (bond_lens - self.mean_bond_lens).abs().max(dim=-1)[0].mean().detach()
            self.stacked_vels = torch.cat(self.running_vels)
         
            energies = torch.cat(self.running_energy)
            np.save(os.path.join(self.save_dir, f'energies_{self.integrator}_{self.ic_stddev}.npy'), energies.cpu())

        return self 

    def save_checkpoint(self, best=False):
        name = "best_ckpt.pt" if best else "ckpt.pt"
        checkpoint_path = os.path.join(self.save_dir, name)
        torch.save({'model_state': self.model.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, best=False):
        name = "best_ckpt.pt" if best else "ckpt.pt"
        checkpoint_path = os.path.join(self.save_dir, name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])

    def create_frame(self, frame):
        # Particle positions, velocities, diameter
        partpos = detach_numpy(self.radii[0]).tolist()
        velocities = detach_numpy(self.velocities[0]).tolist()
        diameter = 10*self.diameter_viz*np.ones((self.n_atoms,))
        diameter = diameter.tolist()
        # Now make gsd file
        s = gsd.hoomd.Frame()
        s.configuration.step = frame
        s.particles.N=self.n_atoms
        s.particles.position = partpos
        s.particles.velocity = velocities
        s.particles.diameter = diameter
        s.configuration.box=[10.0, 10.0, 10.0,0,0,0]
        s.configuration.step = self.dt

        s.bonds.N = self.bonds.shape[0]
        s.bonds.group = detach_numpy(self.bonds)
        return s
    
    def calc_properties(self, pe):
        # Calculate properties of interest in this function
        p_dof = 3*self.n_atoms
        ke = 1/2 * (self.masses*torch.square(self.velocities)).sum(axis = (1,2)).unsqueeze(-1)
        temp = (2*ke/p_dof).mean() / units.kB

        #max bond length deviation calculation (mean over replicas)
        bond_lens = distance_pbc(self.radii[:, self.bonds[:, 0]], self.radii[:, self.bonds[:, 1]], torch.FloatTensor([30., 30., 30.]).to(self.device))
        max_dev = (bond_lens - self.mean_bond_lens).abs().max(dim=-1)[0].mean()
        self.running_energy.append((ke+pe).detach())
        return {"Temperature": temp.item(),
                "Potential Energy": pe.mean().item(),
                "Total Energy": (ke+pe).mean().item(),
                "Momentum Magnitude": torch.norm(torch.sum(self.masses*self.velocities, axis =-2)).item(),
                "Max Bond Length Deviation": max_dev.item()}

    def get_adjoints(self, pos_traj, vel_traj, grad_outputs, force_fn = None):
        
        with torch.no_grad():
            a_dt = self.dt*1 #save frequency = 1 for now
            M = self.masses[0].unsqueeze(0)
            adjoints = []
            adjoint_norms = []
            testR = []
    
            #initial adjoints
            grad_outputs *= a_dt**2/M #pre-multiply the grad outputs to make norms closer to naive backprop
            a = grad_outputs[:, -1]
            for i in tqdm(range(self.vacf_window)):
                #work backwards from the final state
                R = pos_traj[:, -i -1].detach().to(self.device)
                
                testR.append(R.detach())
                adjoints.append(a.detach())
                adjoint_norms.append(a.norm(dim = (-2, -1)).detach())
                #compute VJP between adjoint (a) and df/dR which is the time-derivative of the adjoint state
                #verified that this does the same thing as torch.autograd.grad(forces, R, a, create_graph = True)
                #where forces = force_fn(R)
                _, vjp_a = torch.autograd.functional.vjp(force_fn, R, a)
                #update adjoint state
                a = a + a_dt**2 * vjp_a /M - a_dt*self.gamma * a
                #adjust in the direction of the next grad outputs
                if i != self.vacf_window -1:   
                    a = a + grad_outputs[:, -i - 2]
                
            adjoints = torch.stack(adjoints, axis=1)
            adjoint_norms = torch.stack(adjoint_norms)
            testR = torch.stack(testR, axis=1)
        return adjoints, adjoint_norms, testR

class Stochastic_IFT(torch.autograd.Function):
    def __init__(self, params, device):
        super(Stochastic_IFT, self).__init__()
        self.diff_rdf = DifferentiableRDF(params, device)
        
    @staticmethod
    def forward(ctx, *args):
        with torch.no_grad():
            simulator = args[0]
            gt_rdf = args[1] 
            gt_vacf = args[2]
            params = args[3]
            MINIBATCH_SIZE = simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
            diff_rdf = DifferentiableRDF(params, simulator.device)
            diff_vacf = DifferentiableVACF(params, simulator.device)
            def rdf_loss(rdf):
                return (rdf - gt_rdf).pow(2).mean()
            def vacf_loss(vacf):    
                return (vacf - gt_vacf).pow(2).mean()

            print('Collect MD Simulation Data')
            equilibriated_simulator = simulator.solve()
            ctx.save_for_backward(equilibriated_simulator)
            
            model = equilibriated_simulator.model
            #store original shapes of model parameters
            original_numel = [param.data.numel() for param in model.parameters()]
            original_shapes = [param.data.shape for param in model.parameters()]
            
            #get continuous trajectories (permute to make replica dimension come first)
            radii_traj = torch.stack(equilibriated_simulator.running_radii)
            stacked_radii = radii_traj[::simulator.n_dump] #take i.i.d samples for RDF loss
            radii_traj = radii_traj.permute(1,0,2,3)
            velocities_traj = torch.stack(equilibriated_simulator.running_vels).permute(1,0,2,3)
            accel_traj = torch.stack(equilibriated_simulator.running_accs).permute(1,0,2,3)
            
            vacfs = vmap(diff_vacf)(velocities_traj)
            mean_vacf = vacfs.mean(dim = 0)
            
            if params.vacf_loss_weight == 0:
                vacf_gradient_estimators = None
                vacf_package = (vacf_gradient_estimators, mean_vacf, vacf_loss(mean_vacf).to(simulator.device))
            else:
                #define force function - expects input of shape (batch, N, 3)
                def get_forces(radii):
                    batch_size = radii.shape[0]
                    batch = torch.arange(batch_size).repeat_interleave(simulator.n_atoms).to(simulator.device)
                    atomic_numbers = torch.Tensor(simulator.atoms.get_atomic_numbers()).to(torch.long).to(simulator.device).repeat(batch_size)
                    energy = model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)
                    forces = -compute_grad(inputs = radii, output = energy)
                    #forces = simulator.force_calc(radii, atomic_numbers, batch, retain_grad = True)
                    return forces

                #define Onsager-Machlup Action ("energy" of each trajectory)
                def om_action(vel_traj, acc_traj):
                    v_tp1 = vel_traj[:, :, 1:]
                    v_t = vel_traj[:, :, :-1]
                    a_tp1 = acc_traj[:, :, 1:]
                    diff = (v_tp1 - v_t - a_tp1*simulator.dt + simulator.gamma*v_t*simulator.dt)
                    om_action = diff**2 / simulator.noise_f.unsqueeze(1).unsqueeze(1) #this is exponentially distributed
                    #sum over euclidean dimensions, atoms, and vacf window: TODO: this ruins the exponential property
                    return diff/simulator.noise_f.unsqueeze(1).unsqueeze(1), om_action.sum((-3, -2, -1))
                
                #split into sub-trajectories of length = vacf_window
                radii_traj = radii_traj.reshape(radii_traj.shape[0], -1, simulator.vacf_window,simulator.n_atoms, 3)
                velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)
                accel_traj = accel_traj.reshape(accel_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)
                vacf_loss_tensor = vmap(vmap(vacf_loss))(vacfs).reshape(-1, 1, 1)
            
                '''DEBUGGING: compare gradients from naive backprop with gradients from adjoint method'''
                
                #choose initial condition
                start_radii = radii_traj[:, 0, 0]
                start_vel = velocities_traj[:, 0, 0]

                #run MD
                radii = start_radii.requires_grad_(True)
                velocities = start_vel.requires_grad_(True)
                running_radii = []
                running_vels = []
                running_accs = []
                with torch.enable_grad():
                    forces = get_forces(radii)
                    velocities = velocities + simulator.dt/2*(forces/simulator.masses - simulator.gamma * velocities) + simulator.noise_f/torch.sqrt(torch.tensor(2.0).to(simulator.device)) * torch.randn_like(velocities) 
                    for i in tqdm(range(100)):
                        #new_radii, new_velocities, new_forces = simulator.forward_langevin(radii, velocities, forces, retain_grad = True)
                        new_radii = radii.detach() + simulator.dt*velocities
                        #calculate force at new position
                        new_forces = get_forces(new_radii)
                        #full step in velocities
                        new_velocities = velocities + simulator.dt*(new_forces/simulator.masses - simulator.gamma * velocities) + simulator.noise_f * torch.randn_like(velocities)
                        running_radii.append(new_radii)
                        running_vels.append(new_velocities)
                        running_accs.append(new_forces/simulator.masses)
                        #update
                        radii = new_radii
                        velocities = new_velocities
                        forces = new_forces

                    radii_traj = torch.stack(running_radii).permute(1,0,2,3)
                    velocities_traj = torch.stack(running_vels).permute(1,0,2,3)
                    accel_traj = torch.stack(running_accs).permute(1,0,2,3)
                    velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)
                    accel_traj = accel_traj.reshape(accel_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)

                    diff, om_act = om_action(velocities_traj, accel_traj)
                    #assert(torch.allclose(diff, noise_traj[:, :, 1:], atol = 1e-4)) #make sure the diffs match the stored noises along the trajectory
                    naive_backprop_grads = [[g.detach() if g is not None else torch.Tensor([0.]).to(simulator.device) for g in torch.autograd.grad(o, model.parameters(), create_graph = True, allow_unused = True)] for o in tqdm(om_act.flatten())]
                
                #get initial adjoint states
                grad_outputs = compute_grad(inputs = velocities_traj, output = om_act).detach()
                #print_active_torch_tensors()
                #reshape to join replica and sample dimensions
                velocities_traj = velocities_traj.reshape(radii_traj.shape)
                grad_outputs = grad_outputs.reshape(radii_traj.shape)
                
                #run backward dynamics
                print(f"Run backward dynamics to calculate adjoints:")
                start = time.time()
                final_adjoints, adjoint_norms, R = simulator.get_adjoints(radii_traj, velocities_traj, grad_outputs, force_fn = get_forces)
                end = time.time()

                #log adjoint norms
                np.save(os.path.join(simulator.save_dir, f'adjoint_norms_epoch{simulator.epoch}'), adjoint_norms.cpu().numpy())
                #print_active_torch_tensors()
                
                #print(f"Adjoint calculation time: {end - start} s")
                #now get dO/dtheta (where O is the OM action)
                #Loop over trajectories for now
                def calc_grads(adjoints, radii):
                    with torch.enable_grad():
                        radii.requires_grad=True
                        forces = get_forces(radii)
                    #compute gradient of force w.r.t model params
                    #some of the model gradients are zero - have to use allow_unused - I believe these correspond to the bias parameters which don't affect the force calculation (only the energies)?
                    grads = [g.detach() if g is not None else torch.Tensor([0.]).to(simulator.device) \
                                for g in torch.autograd.grad(forces, model.parameters(), \
                                        adjoints, create_graph = True, allow_unused = True)]
                    return grads
                print("Calculate gradients of Onsager-Machlup action")
                #vmap isn't working for some reason - weird shape error in get_forces
                # vmap_calc_grads = vmap(calc_grads)
                # grads = vmap_calc_grads(final_adjoints, R)
                grads = [calc_grads(adj, r) for adj, r in tqdm(zip(final_adjoints, R))]
                #print_active_torch_tensors()
                #flatten out the grads
                num_params = len(list(model.parameters()))
                num_samples = final_adjoints.shape[0]

                
                naive_backprop_grads_flattened = torch.stack([torch.cat([naive_backprop_grads[i][j].flatten().detach() \
                                        for j in range(num_params)]) for i in range(num_samples)])
                vacf_grads_flattened = torch.stack([torch.cat([grads[i][j].flatten().detach() \
                                        for j in range(num_params)]) for i in range(num_samples)])
                #compare naive backprop and adjoint grads
                # - both have zero grads for parameters in the same places (about 5% of parameters)
                # - the adjoint grads are larger (~2 orders of magnitude) than the naive backprop grads - but not by a constant factor
                # 95% of the nonzero gradients have the same sign with naive backprop and adjoint method 
                # - by premultiplying the grad_outputs (as opposed to the adjoints) by a_dt**2/M, we make the naive_backprop_grads bigger by a factor of ~13 on average
                
                ratios = (naive_backprop_grads_flattened + 1e-8) / (vacf_grads_flattened + 1e-8)
                mask = ratios != 1
                ratios = ratios[mask].reshape(final_adjoints.shape[0], -1)
                import pdb; pdb.set_trace()
                if simulator.shuffle:   
                    shuffle_idx = torch.randperm(vacf_grads_flattened.shape[0])
                    vacf_grads_flattened = vacf_grads_flattened[shuffle_idx]
                    vacf_loss_tensor = vacf_loss_tensor[shuffle_idx]
                
                #now calculate Fabian estimator
                #scale VACF minibatch size to have a similar number of gradient updates as RDF
                #vacf_minibatch_size = math.ceil(MINIBATCH_SIZE / simulator.vacf_window * simulator.n_dump)
                vacf_minibatch_size = MINIBATCH_SIZE
                num_blocks = math.ceil(vacf_grads_flattened.shape[0]/ vacf_minibatch_size)
                start_time = time.time()
                vacf_gradient_estimators = []
                #print(f"Computing VACF gradients in minibatches of {vacf_minibatch_size} trajectories")
                for i in tqdm(range(num_blocks)):
                    start = vacf_minibatch_size*i
                    end = vacf_minibatch_size*(i+1)
                    grads_batch = vacf_grads_flattened[start:end]
                    grad_diffs = grads_batch.unsqueeze(0) - grads_batch.unsqueeze(1)
                    product = vacf_loss_tensor[start:end] * grad_diffs
                    mean = product.mean(dim=(0,1))
                    #re-assemble flattened gradients into correct shape
                    gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(mean.split(original_numel), original_shapes)])
                    vacf_gradient_estimators.append(gradient_estimator)
                vacf_package = (vacf_gradient_estimators, mean_vacf, vacf_loss(mean_vacf).to(simulator.device))
            #print_active_torch_tensors()

            ###RDF Stuff
            r2d = lambda r: radii_to_dists(r, simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(1, -1, simulator.n_atoms, simulator.n_atoms-1, 1)
            rdfs = vmap(diff_rdf)(tuple(dists))
            mean_rdf = rdfs.mean(dim=0)
            if params.rdf_loss_weight ==0:
                rdf_gradient_estimators = None
                rdf_package = (rdf_gradient_estimators, mean_rdf, rdf_loss(mean_rdf).to(simulator.device))              
            else:
                rdf_loss_tensor = vmap(rdf_loss)(rdfs).unsqueeze(-1).unsqueeze(-1)
            
                #TODO: scale the estimator by temperature
                start = time.time()
                stacked_radii = stacked_radii.reshape(-1, simulator.n_atoms, 3)
                #shuffle the radii and losses
                if simulator.shuffle:   
                    shuffle_idx = torch.randperm(stacked_radii.shape[0])
                    stacked_radii = stacked_radii[shuffle_idx]
                    rdf_loss_tensor = rdf_loss_tensor[shuffle_idx]
                
                num_blocks = math.ceil(stacked_radii.shape[0]/ (MINIBATCH_SIZE))
                start_time = time.time()
                rdf_gradient_estimators = []
                print(f"Computing RDF gradients in minibatches of {MINIBATCH_SIZE} structures")
                
                running_energy = []
                for i in tqdm(range(num_blocks)):
                    #print_active_torch_tensors()
                    start = MINIBATCH_SIZE*i
                    end = MINIBATCH_SIZE*(i+1)
                    actual_batch_size = min(end, stacked_radii.shape[0]) - start
                    batch = torch.arange(actual_batch_size).repeat_interleave(simulator.n_atoms).to(simulator.device)
                    atomic_numbers = torch.Tensor(simulator.atoms.get_atomic_numbers()).to(torch.long).to(simulator.device).repeat(actual_batch_size)
                    with torch.enable_grad():
                        radii_in = stacked_radii[start:end].reshape(-1, 3)
                        radii_in.requires_grad = True
                        energy = model(pos = radii_in, z = atomic_numbers, batch = batch)
                        running_energy.append(energy)
                    def get_vjp(v):
                        return compute_grad(inputs = list(model.parameters()), output = energy, grad_outputs = v, create_graph = False)
                    vectorized_vjp = vmap(get_vjp)
                    I_N = torch.eye(energy.shape[0]).unsqueeze(-1).to(simulator.device)
                    grads_vectorized = vectorized_vjp(I_N)
                    #flatten the gradients for vectorization
                    num_params = len(list(model.parameters()))
                    num_samples = energy.shape[0]
                    grads_flattened= torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_params)]) for j in range(num_samples)])
                    grad_diffs = grads_flattened.unsqueeze(0) - grads_flattened.unsqueeze(1)
                    product = rdf_loss_tensor[start:end] * grad_diffs
                    mean = product.mean(dim=(0,1))
                    #re-assemble flattened gradients into correct shape
                    gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(mean.split(original_numel), original_shapes)])
                    rdf_gradient_estimators.append(gradient_estimator)
                #print_active_torch_tensors()
                # energies = torch.cat(running_energy)
                # np.save(os.path.join(simulator.save_dir, f'energies_{simulator.integrator}_{simulator.ic_stddev}.npy'), energies.cpu())
                end_time = time.time()
                #print(f"gradient calculation time: {end_time-start_time} seconds")
                rdf_package = (rdf_gradient_estimators, mean_rdf, rdf_loss(mean_rdf).to(simulator.device))
            
            return equilibriated_simulator, rdf_package, vacf_package

    #TODO: trigger this custom backwards pass
    @staticmethod
    def backward(ctx, *grad_output):
        import pdb; pdb.set_trace()
        equilibriated_simulator = ctx.saved_tensors
        return None

if __name__ == "__main__":
    setup_logging() 
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    params = types.SimpleNamespace(**config["ift"])

    #Set random seeds
    np.random.seed(seed=params.seed)
    torch.manual_seed(params.seed)
    random.seed(params.seed)
    
    #GPU
    try:
        device = torch.device(torch.cuda.current_device())
    except:
        device = "cpu"

    #set up model
    data_path = config['dataset']['src']
    name = config['dataset']['name']
    molecule = config['dataset']['molecule']
    size = config['dataset']['size']
    model_type = config['dataset']['model']
    
    logging.info(f"Loading pretrained {model_type} model")
    pretrained_model_path = os.path.join(config["dataset"]['model_dir'], model_type, f"{name}-{molecule}_{size}_{model_type}") 

    if model_type == "nequip":
        model, model_config = Trainer.load_model_from_training_session(pretrained_model_path, \
                        device =  torch.device(device))
    elif model_type == "schnet":
        model, model_config = load_schnet_model(path = pretrained_model_path, ckpt_epoch = config["dataset"]['checkpoint_epoch'], device = torch.device(device))
    #count number of trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"{num_params} trainable parameters in {model_type} model")

    # #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)#, sample_frac = params.rdf_sample_frac)
    
    if params.nn:
        results = 'results_nnip'
        timestep = config['ift']["integrator_config"]["timestep"]
        ttime = config['ift']["integrator_config"]["ttime"]
        results_dir = os.path.join(results, f"IMPLICIT_{molecule}_{params.exp_name}")
        os.makedirs(results_dir, exist_ok = True)
    # #load ground truth rdf and VACF
    temp_size = '10k' if name == 'md17' else 'all'
    gt_rdf = torch.Tensor(find_hr_from_file(data_path, name, molecule, temp_size, params, device)).to(device)
    contiguous_path = f'{data_path}/contiguous-{name}/{molecule}/{temp_size}/val/nequip_npz.npz'
    gt_data = np.load(contiguous_path)
    gt_traj = torch.FloatTensor(gt_data.f.R).to(device)
    gt_vels = gt_traj[1:] - gt_traj[:-1] #finite difference approx for now TODO: calculate precisely based on forces and positions
    gt_vacf = DifferentiableVACF(params, device)(gt_vels)
    np.save(os.path.join(results_dir, 'gt_rdf.npy'), gt_rdf.cpu())
    np.save(os.path.join(results_dir, 'gt_vacf.npy'), gt_vacf.cpu())
    #initialize outer loop optimizer/scheduler
    optimizer = torch.optim.Adam(list(model.parameters()), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

    if not params.nn:
        params.n_epochs = 1
    
    #outer training loop
    losses = []
    rdf_losses = []
    diffusion_losses = []
    vacf_losses = []
    max_bond_len_devs = []
    energy_rmses = []
    force_rmses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    best_outer_loss = 100
    if params.nn:
        writer = SummaryWriter(log_dir = results_dir)

    #top level simulator
    top_level = Stochastic_IFT(params, device)
    
    for epoch in range(params.n_epochs):
        rdf = torch.zeros_like(gt_rdf).to(device)
        rdf_loss = torch.Tensor([0]).to(device)
        vacf_loss = torch.Tensor([0]).to(device)
        diffusion_loss = torch.Tensor([0]).to(device)
        best = False
        print(f"Epoch {epoch+1}")
        restart= epoch==0 or (torch.rand(size=(1,)) <= params.restart_probability).item()
        
        optimizer.zero_grad()

        if restart: #draw IC from dataset
            print("Initialize from random IC")
            #initialize simulator parameterized by a NN model
            simulator = ImplicitMDSimulator(config, params, model, model_config)
        else: #continue from where we left off in the last epoch/batch
            simulator.reset(last_radii, last_velocities)
            simulator.zeta = equilibriated_simulator.zeta
        simulator.epoch = epoch

        start = time.time()
        if simulator.no_ift: #full backprop through MD simulation - baseline
            with torch.enable_grad():
                #run MD with gradients enabled
                equilibriated_simulator = simulator.solve()
                diff_rdf = DifferentiableRDF(simulator.params, simulator.device)
                diff_vacf = DifferentiableVACF(simulator.params, simulator.device)
                r2d = lambda r: radii_to_dists(r, simulator.params)
                dists = r2d(equilibriated_simulator.stacked_radii).reshape(1, -1, simulator.n_atoms, simulator.n_atoms-1, 1)
                rdfs = vmap(diff_rdf)(tuple(dists))
                mean_rdf = rdfs.mean(dim=0)[0]
                velocities_traj = torch.stack(equilibriated_simulator.running_vels).permute(1,0,2,3)
                vacfs = vmap(diff_vacf)(velocities_traj)
                mean_vacf = vacfs.mean(dim = 0)
                rdf_loss = (mean_rdf - gt_rdf).pow(2).mean()
                vacf_loss = (mean_vacf - gt_vacf).pow(2).mean()
                loss = params.rdf_loss_weight*rdf_loss + params.vacf_loss_weight*vacf_loss
                #this line is currently giving zero gradients
                torch.autograd.backward(tensors = loss, inputs = list(model.parameters()))
                optimizer.step()

        else:
            #run simulation and get gradients via Fabian method/adjoint
            equilibriated_simulator, rdf_package, vacf_package = top_level.apply(simulator, gt_rdf, gt_vacf, params)
            #unpack results
            rdf_grad_batches, mean_rdf, rdf_loss = rdf_package
            vacf_grad_batches, mean_vacf, vacf_loss = vacf_package
            
            #make lengths match for iteration
            if rdf_grad_batches is None:
                rdf_grad_batches = vacf_grad_batches
            if vacf_grad_batches is None:
                vacf_grad_batches = rdf_grad_batches
            
            #manual SGD for now
            for rdf_grads, vacf_grads in zip(rdf_grad_batches, vacf_grad_batches): #loop through minibatches
                for param, rdf_grad, vacf_grad in zip(model.parameters(), rdf_grads, vacf_grads):
                    grad = params.rdf_loss_weight*rdf_grad + params.vacf_loss_weight*vacf_grad
                    param.grad = grad#for grad norm tracking
                    param.data -= optimizer.param_groups[0]['lr']*grad

        end = time.time()
        sim_time = end - start

        
        #save rdf and vacf at the end of the trajectory
        filename = f"rdf_epoch{epoch+1}.npy"
        np.save(os.path.join(results_dir, filename), mean_rdf.cpu().detach().numpy())

        filename = f"vacf_epoch{epoch+1}.npy"
        np.save(os.path.join(results_dir, filename), mean_vacf.cpu().detach().numpy())
        
        
        if params.nn:
            outer_loss = params.rdf_loss_weight*rdf_loss + diffusion_loss + params.vacf_loss_weight*vacf_loss
            print(f"Loss: RDF={rdf_loss.item()}+Diffusion={diffusion_loss.item()}+VACF={vacf_loss.item()}={outer_loss.item()}")

            #checkpointing
            if outer_loss < best_outer_loss:
                best_outer_loss = outer_loss
                best = True
            simulator.save_checkpoint(best = best)

            #equilibriated_simulator.cleanup()

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
            #diffusion_losses.append((diffusion_loss.sqrt()/gt_diff_coeff).item())
            vacf_losses.append(vacf_loss.item())
            max_bond_len_devs.append(equilibriated_simulator.max_dev)
            #energy/force error
            print("Logging energy/force error")
            energy_rmse, force_rmse = simulator.energy_force_error(params.test_batch_size)
            energy_rmses.append(energy_rmse)
            force_rmses.append(force_rmse)

            sim_times.append(sim_time)
            #grad_times.append(grad_time)
            try:
                grad_norms.append(max_norm.item())
            except:
                grad_norms.append(max_norm)

            writer.add_scalar('Loss', losses[-1], global_step=epoch+1)
            writer.add_scalar('RDF Loss', rdf_losses[-1], global_step=epoch+1)
            #writer.add_scalar('Relative Diffusion Loss', diffusion_losses[-1], global_step=epoch+1)
            writer.add_scalar('VACF Loss', vacf_losses[-1], global_step=epoch+1)
            writer.add_scalar('Max Bond Length Deviation', max_bond_len_devs[-1], global_step=epoch+1)
            writer.add_scalar('Energy RMSE', energy_rmses[-1], global_step=epoch+1)
            writer.add_scalar('Force RMSE', force_rmses[-1], global_step=epoch+1)
            writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
            #writer.add_scalar('Gradient Time', grad_times[-1], global_step=epoch+1)
            writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
        
    simulator.f.close()
    simulator.t.close()
    np.save(os.path.join(results_dir, 'gt_rdf.npy'), gt_rdf.cpu())


    if params.nn:
        stats_write_file = os.path.join(simulator.save_dir, 'stats.txt')
        with open(stats_write_file, "w") as output:
            output.write("Losses: " + str(losses) + "\n")
            output.write("Simulation times: " +  str(sim_times) + "\n")
            output.write("Gradient calculation times: " +  str(grad_times) + "\n")
            output.write("Max gradient norms: " + str(grad_norms))

        writer.close()
    print('Done!')
    


