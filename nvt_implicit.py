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
from configs.md22.integrator_configs import INTEGRATOR_CONFIGS
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
from itertools import product
from tqdm import tqdm
import pstats
import pdb
import random
from torch_geometric.nn import MessagePassing, radius_graph
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import generate_vol_bins, DifferentiableRDF, DifferentiableADF, DifferentiableVelHist, DifferentiableVACF, msd, DiffusionCoefficient
from contextlib import nullcontext
import time
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
from sys import getrefcount
from functorch import vmap, vjp
from utils import process_gradient, compare_gradients, radii_to_dists, fcc_positions, initialize_velocities, load_schnet_model, \
                    dump_params_to_yml, powerlaw_inv_cdf, print_active_torch_tensors, plot_pair, solve_continuity_system, find_hr_adf_from_file, distance_pbc
import warnings
warnings.filterwarnings("ignore")
#NNIP stuff:
import ase
from ase import Atoms, units
from ase.calculators.calculator import Calculator
import nequip.scripts.deploy
from nequip.ase.nequip_calculator import nequip_calculator
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.AtomicData import neighbor_list_and_relative_vec
from nequip.utils import atomic_write
from nequip.utils.torch_geometric import Batch, Dataset
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, Trajectory
from ase.neighborlist import natural_cutoffs, NeighborList
import mdsim.md.integrator as md_integrator
from mdsim.common.registry import registry
from mdsim.common.utils import setup_imports, setup_logging, compute_bond_lengths, data_to_atoms, atoms_to_batch, atoms_to_state_dict, convert_atomic_numbers_to_types
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.datasets import data_list_collater
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.common.utils import load_config
from mdsim.modules.evaluator import Evaluator
from mdsim.modules.normalizer import Normalizer

from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg
)
from mdsim.common.flags import flags

MAX_SIZES = {'md17': '50k', 'md22': '100percent', 'water': '90k', 'lips': '20k'}

class ImplicitMDSimulator():
    def __init__(self, config, params, model, model_config):
        super(ImplicitMDSimulator, self).__init__()
        print("Initializing MD simulation environment")
        self.params = params
        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"
        self.name = config['name']
        self.pbc = (self.name == 'water' or self.name == 'lips')
        self.molecule = config['molecule'] if 'molecule' in config else ""
        self.size = config['size']
        self.model_type = config['model']
        self.l_max = config["l_max"]
        self.all_unstable = False
        self.first_simulation = True

        
        self.config = config
        self.data_dir = config['src']
        lmax_string = f"lmax={self.l_max}_" if model_type == "nequip" else ""
        self.train = params.train
        self.n_replicas = config["n_replicas"]
        self.minibatch_size = config['minibatch_size']
        self.allow_off_policy_updates = config['allow_off_policy_updates']
        self.gradient_clipping = config["gradient_clipping"]
        self.shuffle = config['shuffle']
        self.reset_probability = config["reset_probability"]
        self.vacf_window = config["vacf_window"]
        self.no_ift = config["no_ift"]
        self.optimizer = config["optimizer"]
        self.use_mse_gradient = config["use_mse_gradient"]
        self.bond_dev_tol = config["bond_dev_tol"]
        self.max_frac_unstable_threshold = config["max_frac_unstable_threshold"]
        self.min_frac_unstable_threshold = config["min_frac_unstable_threshold"]
        self.results_dir = os.path.join(config["log_dir"], config["results_dir"])
        self.eval_model = config["eval_model"]
        self.n_dump = config["n_dump"]
        self.n_dump_vacf = config["n_dump_vacf"]

        #Initialize model
        self.model = model
        self.model_config = model_config

        #initialize datasets
        
        train_src = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'test')
        valid_src = os.path.join(self.data_dir, self.name, self.molecule, MAX_SIZES[self.name], 'val')
        
        self.train_dataset = LmdbDataset({'src': train_src})
        self.valid_dataset = LmdbDataset({'src': valid_src})

        #get random initial condition from dataset
        length = self.train_dataset.__len__()
        init_data = self.train_dataset.__getitem__(10)
        self.n_atoms = init_data['pos'].shape[0]
        self.atoms = data_to_atoms(init_data)

        #extract bond and atom type information
        NL = NeighborList(natural_cutoffs(self.atoms), self_interaction=False)
        NL.update(self.atoms)
        self.bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(self.device).T
        self.atom_types = self.atoms.get_chemical_symbols()
        #atom type mapping
        if self.model_type == "nequip":
            type_names = self.model_config[nequip.scripts.deploy.TYPE_NAMES_KEY]
            species_to_type_name = {s: s for s in ase.data.chemical_symbols}
            type_name_to_index = {n: i for i, n in enumerate(type_names)}
            chemical_symbol_to_type = {
                sym: type_name_to_index[species_to_type_name[sym]]
                for sym in ase.data.chemical_symbols
                if sym in type_name_to_index
            }
            self.typeid = np.zeros(self.n_atoms, dtype=int)
            for i, _type in enumerate(self.atom_types):
                self.typeid[i] = chemical_symbol_to_type[_type]  
            self.final_atom_types = torch.Tensor(self.typeid).repeat(self.n_replicas).to(self.device).to(torch.long).unsqueeze(-1)

        #extract ground truth energies, forces, and bond length deviation
        DATAPATH_TRAIN = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'train/nequip_npz.npz')
        DATAPATH_TEST = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'test/nequip_npz.npz')
        
        gt_data_test = np.load(DATAPATH_TEST)
        self.gt_traj_test = torch.FloatTensor(gt_data_test.f.wrapped_coords if self.name == 'water' else gt_data_test.f.R).to(self.device)
        self.gt_energies_test = torch.FloatTensor(gt_data_test.f.energy if self.name == 'water' else gt_data_test.f.E).to(self.device)
        #TODO: force vs forces
        self.gt_forces_test = torch.FloatTensor(gt_data_test.f.forces if self.name == 'water' else gt_data_test.f.F).to(self.device)
        self.target_test = {'energy': self.gt_energies_test, 'forces': self.gt_forces_test.reshape(-1, 3), \
                            'natoms': torch.Tensor([self.n_atoms]).repeat(self.gt_energies_test.shape[0]).to(self.device)}
        self.normalizer_test = Normalizer(tensor = self.gt_energies_test, device = self.device)
        
        gt_data_train = np.load(DATAPATH_TRAIN)
        self.gt_traj_train = torch.FloatTensor(gt_data_train.f.wrapped_coords if self.name == 'water' else gt_data_train.f.R).to(self.device)
        self.gt_energies_train = torch.FloatTensor(gt_data_train.f.energy if self.name == 'water' else gt_data_train.f.E).to(self.device)
        #TODO: force vs forces
        self.gt_forces_train = torch.FloatTensor(gt_data_train.f.forces if self.name == 'water' else gt_data_train.f.F).to(self.device)
        self.target_train = {'energy': self.gt_energies_train, 'forces': self.gt_forces_train.reshape(-1, 3), \
                                'natoms': torch.Tensor([self.n_atoms]).repeat(self.gt_energies_train.shape[0]).to(self.device)}
        self.normalizer_train = Normalizer(tensor = self.gt_energies_train, device = self.device)
        self.mean_bond_lens = distance_pbc(
        self.gt_traj_train[:, self.bonds[:, 0]], self.gt_traj_train[:, self.bonds[:, 1]], \
                    torch.FloatTensor([30., 30., 30.]).to(self.device)).mean(dim=0)
        self.max_bond_dev_per_replica = torch.zeros((self.n_replicas,)).to(self.device)
        self.stable_time = torch.zeros((self.n_replicas,)).to(self.device)
        
        self.integrator = self.config["integrator"]
        #Nose-Hoover Thermostat stuff
        self.integrator_config = INTEGRATOR_CONFIGS[self.molecule] if self.name == 'md22' else config['integrator_config']
        self.dt = self.integrator_config["timestep"] * units.fs
        self.temp = self.integrator_config["temperature"]
        
        # adjust units.
        if self.integrator in ['NoseHoover', 'NoseHooverChain', 'Langevin']:
            self.temp *= units.kB
        self.targeEkin = 0.5 * (3.0 * self.n_atoms) * self.temp
        
        self.ttime = self.integrator_config["ttime"]
        self.Q = 3.0 * self.n_atoms * self.temp * (self.ttime * self.dt)**2
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
        self.masses = torch.Tensor(self.atoms.get_masses().reshape(1, -1, 1)).to(self.device)


        #Langevin thermostat stuff
        self.gamma = self.integrator_config["gamma"] / (1000*units.fs)
        self.noise_f = (2.0 * self.gamma/self.masses * self.temp * self.dt).sqrt().to(self.device)

        self.nsteps = params.steps
        self.eq_steps = params.eq_steps
        #ensure that the number of logged steps is a multiple of the vacf window (for chopping up the trajectory)
        self.nsteps -= (self.nsteps - self.eq_steps) % self.vacf_window
        if (self.nsteps - self.eq_steps) < self.vacf_window:
            self.nsteps = self.eq_steps + 2*self.vacf_window #at least two windows
        while self.nsteps < params.steps: #nsteps should be at least as long as what was requested
            self.nsteps += self.vacf_window
        self.ps_per_epoch = self.nsteps * self.integrator_config["timestep"] // 1000.
        

        self.atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(self.n_replicas)
        self.batch = torch.arange(self.n_replicas).repeat_interleave(self.n_atoms).to(self.device)
        self.ic_stddev = params.ic_stddev

        dataset = self.train_dataset if self.train else self.valid_dataset
        #samples = np.random.choice(np.arange(dataset.__len__()), self.n_replicas, replace=False)
        samples = [0]
        self.raw_atoms = [data_to_atoms(dataset.__getitem__(i)) for i in samples]
        self.cell = torch.Tensor(self.raw_atoms[0].cell).to(self.device)
        radii = torch.stack([torch.Tensor(atoms.get_positions()) for atoms in self.raw_atoms])
        self.radii = (radii + torch.normal(torch.zeros_like(radii), self.ic_stddev)).to(self.device)
        self.velocities = torch.Tensor(initialize_velocities(self.n_atoms, self.masses, self.temp, self.n_replicas)).to(self.device)
        #initialize checkpoint states for resetting
        self.checkpoint_radii = []
        self.checkpoint_radii.append(self.radii)
        self.checkpoint_velocities = []
        self.checkpoint_velocities.append(self.velocities)
        self.checkpoint_zetas = []
        self.checkpoint_zetas.append(self.zeta)
        self.original_radii = self.radii.clone()
        self.original_velocities = self.velocities.clone()
        self.original_zeta = self.zeta.clone()
        
        if self.model_type == 'nequip':
            #assign velocities to atoms
            for i in range(len(self.raw_atoms)):
                self.raw_atoms[i].set_velocities(self.velocities[i].cpu().numpy())
            #create batch of atoms to be operated on
            self.atoms_batch = [AtomicData.from_ase(atoms=a, r_max= self.model_config["r_max"]) for a in self.raw_atoms]
            self.atoms_batch = AtomicData.to_AtomicDataDict(Batch.from_data_list(self.atoms_batch))
            self.atoms_batch['atom_types'] = self.final_atom_types
            del self.atoms_batch['ptr']
            del self.atoms_batch['atomic_numbers']
            self.atoms_batch = {k: v.to(self.device) for k, v in self.atoms_batch.items()}
            # import pdb; pdb.set_trace()
            # del self.atoms_batch['cell']
            # del self.atoms_batch['pbc']
            # del self.atoms_batch['edge_cell_shift']
            
        self.diameter_viz = params.diameter_viz
        self.exp_name = params.exp_name
        self.rdf_loss_weight = params.rdf_loss_weight
        self.diffusion_loss_weight = params.diffusion_loss_weight
        self.vacf_loss_weight = params.vacf_loss_weight

        #energy/force stuff
        self.evaluator = Evaluator(task="s2ef") #s2ef: structure to energies/forces
        self.energy_loss = torch.nn.MSELoss()
        self.force_loss = torch.nn.MSELoss()
        self.energy_loss_weight = params.energy_loss_weight
        self.force_loss_weight = params.force_loss_weight
        
        #limit CPU usage
        torch.set_num_threads(10)

        #define vectorized differentiable rdf and vacf
        self.diff_rdf = vmap(DifferentiableRDF(params, self.device), -1)
        self.diff_vacf = vmap(DifferentiableVACF(params, self.device))
    
        molecule_for_name = "water" if self.name == 'water' else self.molecule
        name = f"{molecule_for_name}_{params.exp_name}" if self.no_ift else f"IMPLICIT_{molecule_for_name}_{params.exp_name}"
        self.save_dir = os.path.join(self.results_dir, name) if self.train else os.path.join(self.results_dir, name, 'inference', self.eval_model)
        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)
        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")
         

    '''compute energy/force error on test set'''
    def energy_force_error(self, batch_size):
        with torch.no_grad():
            
            num_batches = math.ceil(self.gt_traj_test.shape[0]/ batch_size)
            energies = []
            forces = []
            print(f"Computing bottom-up (energy-force) error on held-out test set of {num_batches * batch_size} samples")
            for i in tqdm(range(num_batches)):
                start = batch_size*i
                end = batch_size*(i+1)
                actual_batch_size = min(end, self.gt_traj_test.shape[0]) - start        
                atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(actual_batch_size)
                batch = torch.arange(actual_batch_size).repeat_interleave(self.n_atoms).to(self.device)
                
                energy, force = self.force_calc(self.gt_traj_test[start:end], atomic_numbers, batch)
                energies.append(energy.detach())
                forces.append(force.detach())
            
            energies = torch.cat(energies)
            forces = torch.cat(forces)
            prediction = {'energy': self.normalizer_train.denorm(energies), 'forces': forces.reshape(-1, 3), 'natoms': torch.Tensor([self.n_atoms]).repeat(energies.shape[0]).to(self.device)}
            metrics = self.evaluator.eval(prediction, self.target_test)
            metrics = {k: v['metric'] for k, v in metrics.items()}
            return metrics['energy_rmse'], metrics['forces_rmse']

    def energy_force_gradient(self, batch_size):
        #num_batches = math.ceil(self.gt_traj_train.shape[0]/ batch_size)
        num_batches = math.ceil(1000/ batch_size)
        energy_gradients = []
        force_gradients = []
        #store original shapes of model parameters
        original_numel = [param.data.numel() for param in self.model.parameters()]
        original_shapes = [param.data.shape for param in self.model.parameters()]
        print(f"Computing gradients of bottom-up (energy-force) objective on {num_batches * batch_size} samples")
        with torch.enable_grad():
            for i in tqdm(range(num_batches)):
                start = batch_size*i
                end = batch_size*(i+1)
                actual_batch_size = min(end, self.gt_traj_train.shape[0]) - start        
                atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(actual_batch_size)
                batch = torch.arange(actual_batch_size).repeat_interleave(self.n_atoms).to(self.device)
                energy, force = self.force_calc(self.gt_traj_train[start:end], atomic_numbers, batch, retain_grad = True)
                
                #compute losses
                energy_loss = self.energy_loss(self.normalizer_train.norm(self.gt_energies_train[start:end]), energy).mean()
                force_loss = self.force_loss(self.gt_forces_train[start:end].reshape(-1, 3), force.reshape(-1, 3)).mean() 
                energy_gradients.append(process_gradient(torch.autograd.grad(energy_loss, self.model.parameters(), retain_graph = True), self.device))
                force_gradients.append(process_gradient(torch.autograd.grad(force_loss, self.model.parameters(), allow_unused = True), self.device))
            num_params = len(list(self.model.parameters()))
            #energy
            energy_grads_flattened = torch.stack([torch.cat([energy_gradients[j][i].flatten().detach() for i in range(num_params)]) for j in range(num_batches)])
            mean_energy_grads = energy_grads_flattened.mean(0)
            final_energy_grads = tuple([g.reshape(shape) for g, shape in zip(mean_energy_grads.split(original_numel), original_shapes)])
            #force
            force_grads_flattened = torch.stack([torch.cat([force_gradients[j][i].flatten().detach() for i in range(num_params)]) for j in range(num_batches)])
            mean_force_grads = force_grads_flattened.mean(0)
            final_force_grads = tuple([g.reshape(shape) for g, shape in zip(mean_force_grads.split(original_numel), original_shapes)])
            return final_energy_grads, final_force_grads


    def resume(self):
        #reset replicas which exceeded the max bond dev criteria
        reset_replicas = self.max_bond_dev_per_replica > self.bond_dev_tol
        num_resets = reset_replicas.count_nonzero().item()
        if num_resets / self.n_replicas >= self.max_frac_unstable_threshold: #threshold of unstable replicas reached
            if not self.all_unstable:
                print("Threshold of unstable replicas has been reached... Start Learning")
            self.all_unstable = True
        
        if self.all_unstable: #reset all replicas to the initial values
            self.radii = self.original_radii
            self.velocities = self.original_velocities
            self.zeta = self.original_zeta

        else:
            if self.first_simulation: #random replica reset
                #pick replicas to reset (only pick from stable replicas)
                stable_replicas = (~reset_replicas).nonzero()
                random_reset_idxs = torch.randperm(stable_replicas.shape[0]) \
                                        [0:math.ceil(self.reset_probability * stable_replicas.shape[0])].to(self.device)
                #uniformly sample times to reset each replica to
                reset_times = torch.randint(0, len(self.checkpoint_radii), (len(random_reset_idxs), )).to(self.device)
                #reset them 
                self.radii.requires_grad = False
                self.velocities.requires_grad = False
                self.zeta.requires_grad = False
                for idx, time in zip(random_reset_idxs, reset_times):
                    self.radii[idx] = self.checkpoint_radii[time][idx]
                    self.velocities[idx] = self.checkpoint_velocities[time][idx]
                    self.zeta[idx] = self.checkpoint_zetas[time][idx]

            #reset the replicas which are unstable
            exp_reset_replicas = (reset_replicas).unsqueeze(-1).unsqueeze(-1).expand_as(self.radii)
            self.radii = torch.where(exp_reset_replicas, self.original_radii.detach().clone(), self.radii.detach().clone()).requires_grad_(True)
            self.velocities = torch.where(exp_reset_replicas, self.original_velocities.detach().clone(), self.velocities.detach().clone()).requires_grad_(True)
            self.zeta = torch.where(reset_replicas.unsqueeze(-1).unsqueeze(-1), self.original_zeta.detach().clone(), self.zeta.detach().clone()).requires_grad_(True)
            #update stability times for each replica
            if not self.train:
                self.stable_time = torch.where(reset_replicas, self.stable_time, self.stable_time + self.ps_per_epoch)
        self.first_simulation = False
        return num_resets / self.n_replicas


    def force_calc(self, radii, atomic_numbers = None, batch = None, retain_grad = False):
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers
        if batch is None:
            batch = self.batch
        with torch.enable_grad():
            if not radii.requires_grad:
                radii.requires_grad = True
            if self.model_type == "schnet":
                import pdb; pdb.set_trace()
                energy = self.model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)
                forces = -compute_grad(inputs = radii, output = energy, create_graph = retain_grad)
            elif self.model_type == "nequip":
                #assign radii
                self.atoms_batch['pos'] = radii.reshape(-1, 3)
                self.atoms_batch['batch'] = batch
                self.atoms_batch['atom_types'] = self.final_atom_types
                #make these match the number of replicas (different from n_replicas when doing bottom-up stuff)
                self.atoms_batch['cell'] = self.atoms_batch['cell'][0].unsqueeze(0).repeat(radii.shape[0], 1, 1)
                self.atoms_batch['pbc'] = self.atoms_batch['pbc'][0].unsqueeze(0).repeat(radii.shape[0], 1)
                self.atoms_batch['atom_types'] = self.atoms_batch['atom_types'][0:self.n_atoms].repeat(radii.shape[0], 1)
                
                #recompute neighbor list
                self.atoms_batch['edge_index'] = radius_graph(radii.reshape(-1, 3), r=self.model_config["r_max"], batch=batch, max_num_neighbors=32)
                self.atoms_batch['edge_cell_shift'] = torch.zeros((self.atoms_batch['edge_index'].shape[1], 3)).to(self.device)
                atoms_updated = self.model(self.atoms_batch)
                del self.atoms_batch['node_features']
                del self.atoms_batch['node_attrs']
                energy = atoms_updated[AtomicDataDict.TOTAL_ENERGY_KEY]
                forces = atoms_updated[AtomicDataDict.FORCE_KEY].reshape(-1, self.n_atoms, 3) if retain_grad else atoms_updated[AtomicDataDict.FORCE_KEY].reshape(-1, self.n_atoms, 3).detach()
            assert(not torch.any(torch.isnan(forces)))
            return energy, forces


    def forward_nosehoover(self, radii, velocities, forces, zeta, retain_grad=False):
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
            step  = self.step if self.train else (self.epoch+1) * self.step #don't overwrite previous epochs at inference time
            try:   
                self.t.append(self.create_frame(frame = step/self.n_dump))
            except:
                pass

        return radii, velocities, forces, zeta
    
    def forward_langevin(self, radii, velocities, forces, retain_grad = False):
        
        #full step in position
        radii = radii.detach() + self.dt*velocities
        #calculate force at new position
        energy, forces = self.force_calc(radii, retain_grad=retain_grad)
        noise = torch.randn_like(velocities)
        #full step in velocities
        velocities = velocities + self.dt*(forces/self.masses - self.gamma * velocities) + self.noise_f * noise
        # # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.calc_properties(energy), file=self.f)
            try:    
                self.t.append(self.create_frame(frame = self.step/self.n_dump))
            except:
                pass
        return radii, velocities, forces, noise

      
    #top level MD simulation code (i.e the "solver") that returns the optimal "parameter" -aka the equilibriated radii
    def solve(self):
        self.mode = 'learning' if self.all_unstable else 'simulation'
        self.running_dists = []
        self.running_vels = []
        self.running_accs = []
        self.last_h_radii = []
        self.last_h_velocities = []
        self.running_radii = []
        self.running_noise = []
        self.running_energy = []
        self.original_radii = self.radii.clone()
        self.original_velocities = self.velocities.clone()
        self.original_zeta = self.zeta.clone()
        #log checkpoint states for resetting
        if self.mode == 'simulation':
            self.checkpoint_radii.append(self.original_radii)
            self.checkpoint_velocities.append(self.original_velocities)
            self.checkpoint_zetas.append(self.original_zeta)
        #File dump
        if self.train or self.epoch == 0: #create one long simulation for inference
            try:
                self.t = gsd.hoomd.open(name=f'{self.save_dir}/sim_epoch{self.epoch+1}_{self.mode}.gsd', mode='w')
            except:
                pass
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
                    radii, velocities, forces, zeta = self.forward_nosehoover(self.radii, self.velocities, forces, zeta, retain_grad = self.no_ift)
                elif self.integrator == 'Langevin':
                    radii, velocities, forces, noise = self.forward_langevin(self.radii, self.velocities, forces, retain_grad = self.no_ift)
                else:
                    RuntimeError("Must choose either NoseHoover or Langevin as integrator")
                
                #save trajectory for gradient calculation
                if step >= self.eq_steps:# and step % self.n_dump == 0:
                    self.running_radii.append(radii if self.no_ift else radii.detach().clone())
                    self.running_vels.append(velocities if self.no_ift else velocities.detach().clone())
                    self.running_accs.append((forces/self.masses) if self.no_ift else (forces/self.masses).detach().clone())
                    if self.integrator == "Langevin":
                        self.running_noise.append(noise)
                if self.no_ift:
                    self.radii = radii
                    self.velocities = velocities
                else:
                    self.radii.copy_(radii)
                    self.velocities.copy_(velocities)
                    
            self.zeta = zeta
            self.forces = forces

            #compute bond length deviation
            self.stacked_radii = torch.stack(self.running_radii)
            bond_lens = distance_pbc(self.stacked_radii[:, :, self.bonds[:, 0]], self.stacked_radii[:,:, self.bonds[:, 1]], torch.FloatTensor([30., 30., 30.]).to(self.device))
            
            #max over bonds and timesteps
            self.max_bond_dev_per_replica = (bond_lens - self.mean_bond_lens).abs().max(dim=-1)[0].max(dim=0)[0].detach()
            self.max_dev = self.max_bond_dev_per_replica.mean()
            self.stacked_vels = torch.cat(self.running_vels)
        
        # self.f.close()
        if self.train:
            try:
                self.t.close()
            except:
                pass
        return self 

    def save_checkpoint(self, best=False):
        name = "best_ckpt.pth" if best else "ckpt.pth"
        checkpoint_path = os.path.join(self.save_dir, name)
        try:
            torch.save({'model_state': self.model.state_dict(), 'config': self.model_config}, checkpoint_path)
        except:
            pass
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

class Stochastic_IFT(torch.autograd.Function):
    def __init__(self, params, device):
        super(Stochastic_IFT, self).__init__()
        self.diff_rdf = DifferentiableRDF(params, device)
        
    @staticmethod
    def forward(ctx, *args):
        with torch.no_grad():
            simulator = args[0]
            gt_rdf = args[1] 
            gt_adf = args[2]
            gt_vacf = args[3]
            params = args[4]

            if simulator.vacf_loss_weight !=0 and simulator.integrator != "Langevin":
                raise RuntimeError("Must use stochastic (Langevin) dynamics for VACF training")
            
            MINIBATCH_SIZE = simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
            diff_rdf = DifferentiableRDF(params, simulator.device)
            diff_adf = DifferentiableADF(simulator.n_atoms, simulator.bonds, simulator.cell, params, simulator.device)
            diff_vacf = DifferentiableVACF(params, simulator.device)
            
            def rdf_loss(rdf):
                return (rdf - gt_rdf).pow(2).mean()
            def adf_loss(adf):
                return (adf - gt_adf).pow(2).mean()
            def vacf_loss(vacf):    
                return (vacf - gt_vacf).pow(2).mean()

            print('Collect MD Simulation Data')
            equilibriated_simulator = simulator.solve()
            running_radii = simulator.running_radii if simulator.all_unstable else simulator.running_radii[0:2]
            #ctx.save_for_backward(equilibriated_simulator)
            
            model = equilibriated_simulator.model
            #find which replicas are unstable
            stable_replicas = simulator.max_bond_dev_per_replica <= simulator.bond_dev_tol
            #store original shapes of model parameters
            original_numel = [param.data.numel() for param in model.parameters()]
            original_shapes = [param.data.shape for param in model.parameters()]
            
            #get continuous trajectories (permute to make replica dimension come first)
            radii_traj = torch.stack(running_radii)
            
            stacked_radii = radii_traj[::simulator.n_dump] #take i.i.d samples for RDF loss
            velocities_traj = torch.stack(equilibriated_simulator.running_vels).permute(1,0,2,3)
            #split into sub-trajectories of length = vacf_window
            velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)
            velocities_traj = velocities_traj[:, ::simulator.n_dump_vacf] #sample i.i.d paths
            vacfs = vmap(vmap(diff_vacf))(velocities_traj)
            mean_vacf = vacfs[stable_replicas].mean(dim = (0,1)) #only compute loss on stable replicas
            vacfs = vacfs.reshape(-1, simulator.vacf_window)
            mean_vacf_loss = vacf_loss(mean_vacf)   

            #energy/force loss
            if (simulator.energy_loss_weight != 0 or simulator.force_loss_weight!=0) and simulator.train and simulator.all_unstable:
                energy_grads, force_grads = simulator.energy_force_gradient(batch_size = simulator.n_replicas)
                energy_force_package = ([energy_grads], [force_grads])
            else:
                energy_force_package = (None, None)

            if simulator.vacf_loss_weight !=0 and simulator.train and simulator.all_unstable:
                radii_traj = radii_traj.permute(1,0,2,3)
                accel_traj = torch.stack(equilibriated_simulator.running_accs).permute(1,0,2,3)
                noise_traj = torch.stack(equilibriated_simulator.running_noise).permute(1,0,2,3)
                #split into sub-trajectories of length = vacf_window
                radii_traj = radii_traj.reshape(radii_traj.shape[0], -1, simulator.vacf_window,simulator.n_atoms, 3)
                radii_traj = radii_traj[:, ::simulator.n_dump_vacf] #sample i.i.d paths
                noise_traj = noise_traj.reshape(noise_traj.shape[0], -1, simulator.vacf_window, simulator.n_atoms, 3)
                noise_traj = noise_traj[:, ::simulator.n_dump_vacf] #sample i.i.d paths
            else:
                del radii_traj
                del velocities_traj
                del simulator.running_radii
                del simulator.running_accs
                del simulator.running_noise
                del simulator.running_vels
                
            
            if params.vacf_loss_weight == 0 or not simulator.train or not simulator.all_unstable:
                vacf_gradient_estimators = None
                vacf_package = (vacf_gradient_estimators, mean_vacf, vacf_loss(mean_vacf).to(simulator.device))
            else:
                vacf_loss_tensor = vmap(vmap(vacf_loss))(vacfs).reshape(-1, 1, 1)
                #define force function - expects input of shape (batch, N, 3)
                def get_forces(radii):
                    batch_size = radii.shape[0]
                    batch = torch.arange(batch_size).repeat_interleave(simulator.n_atoms).to(simulator.device)
                    atomic_numbers = torch.Tensor(simulator.atoms.get_atomic_numbers()).to(torch.long).to(simulator.device).repeat(batch_size)
                    if simulator.model_type == "schnet":
                        energy = model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)
                        grad = compute_grad(inputs = radii, output = energy)
                        assert(not grad.is_leaf)
                        return -grad
                    elif simulator.model_type == "nequip":
                        #recompute neighbor list
                        #assign radii
                        simulator.atoms_batch['pos'] = radii.reshape(-1, 3)
                        simulator.atoms_batch['batch'] = batch
                        simulator.atoms_batch['atom_types'] = simulator.final_atom_types
                        #recompute neighbor list
                        simulator.atoms_batch['edge_index'] = radius_graph(radii.reshape(-1, 3), r=simulator.model_config["r_max"], batch=batch, max_num_neighbors=32)
                        simulator.atoms_batch['edge_cell_shift'] = torch.zeros((simulator.atoms_batch['edge_index'].shape[1], 3)).to(simulator.device)
                        atoms_updated = simulator.model(simulator.atoms_batch)
                        del simulator.atoms_batch['node_features']
                        del simulator.atoms_batch['node_attrs']
                        energy = atoms_updated[AtomicDataDict.TOTAL_ENERGY_KEY]
                        forces = atoms_updated[AtomicDataDict.FORCE_KEY].reshape(-1, simulator.n_atoms, 3)
                        assert(not forces.is_leaf)
                        return forces

                #define Onsager-Machlup Action ("energy" of each trajectory)
                #TODO: make this a torch.nn.Module in observable.py
                def om_action(vel_traj, radii_traj):
                    v_tp1 = vel_traj[:, :, 1:]
                    v_t = vel_traj[:, :, :-1]
                    f_tp1 = get_forces(radii_traj[:, :, 1:].reshape(-1, simulator.n_atoms, 3)).reshape(v_t.shape)
                    a_tp1 = f_tp1/simulator.masses.unsqueeze(1).unsqueeze(1)
                    diff = (v_tp1 - v_t - a_tp1*simulator.dt + simulator.gamma*v_t*simulator.dt)
                    #pre-divide by auxiliary temperature (noise_f**2)
                    om_action = diff**2 / (simulator.noise_f**2).unsqueeze(1).unsqueeze(1) #this is exponentially distributed
                    #sum over euclidean dimensions, atoms, and vacf window: TODO: this ruins the exponential property
                    return (diff/simulator.noise_f.unsqueeze(1).unsqueeze(1)).detach(), om_action.sum((-3, -2, -1))
                
                #compute OM action - do it in mini-batches to avoid OOM issues
                batch_size = 5
                print(f"Calculate gradients of Onsager-Machlup action of {velocities_traj.shape[0] * velocities_traj.shape[1]} paths in minibatches of size {batch_size*velocities_traj.shape[1]}")
                num_blocks = math.ceil(velocities_traj.shape[0]/ batch_size)
                diffs = []
                om_acts = []
                grads = []
                quick_grads = []
                
                with torch.enable_grad():
                    velocities_traj = velocities_traj.detach().requires_grad_(True)
                    radii_traj = radii_traj.detach().requires_grad_(True)
                    for i in tqdm(range(num_blocks)):
                        start = batch_size*i
                        end = batch_size*(i+1)
                        velocities = velocities_traj[start:end]
                        radii = radii_traj[start:end]
                        diff, om_act = om_action(velocities, radii)
                        #noises = noise_traj[start:end].reshape(-1, simulator.vacf_window, simulator.n_atoms, 3)
                        # with torch.enable_grad():
                        #     forces = get_forces(radii.reshape(-1, simulator.n_atoms, 3)).reshape(-1, simulator.vacf_window, simulator.n_atoms, 3)
                        
                        #make sure the diffs match the stored noises along the trajectory
                        #assert(torch.allclose(diff, noise_traj[start:end, :, 1:], atol = 1e-3))
                        
                        #doesn't work because vmap detaches the force tensor before doing the vectorization
                        def get_grads(forces_, noises_):
                            return torch.autograd.grad(-1*forces_, list(model.parameters()), noises_, allow_unused = True)
                       
                        #loop over samples
                        #VJP between df/dtheta and noise - thought this method would be faster but it's the same
                        #verified that gradients match the OM action way
                        # for i in tqdm(range(forces.shape[0])):
                        #     grad = [simulator.process_gradient(g) for g in torch.autograd.grad(-1*forces[i], list(model.parameters()), 2*simulator.dt/simulator.masses* simulator.noise_f * noises[i], retain_graph = True, allow_unused = True)]
                        #     quick_grads.append(grad)
                        # I_N = torch.eye(om_act.flatten().shape[0]).to(simulator.device)
                        #the vmap leads to a seg fault for some reason
                        #grad = get_grads_vmaped(I_N)
                        #grad = [[simulator.process_gradient(g) for g in get_grads(v)] for v in I_N]
                        #this explicit loop is very slow though (10 seconds per iteration)
                        #OM-action method
                        grad = [process_gradient(torch.autograd.grad(o, model.parameters(), create_graph = False, retain_graph = True, allow_unused = True), simulator.device) for o in om_act.flatten()]
                        
                        om_act = om_act.detach()
                        diffs.append(diff)
                        om_acts.append(om_act)
                        grads.append(grad)

                #recombine batches
                diff = torch.cat(diffs)
                om_act = torch.cat(om_acts)
                grads = sum(grads, [])
                #log OM stats
                np.save(os.path.join(simulator.save_dir, f'om_diffs_epoch{simulator.epoch}'), diff.flatten().cpu().numpy())
                np.save(os.path.join(simulator.save_dir, f'om_action_epoch{simulator.epoch}'), om_act.detach().flatten().cpu().numpy())
                    
                #flatten out the grads
                num_params = len(list(model.parameters()))
                num_samples = len(grads)

                vacf_grads_flattened = torch.stack([torch.cat([grads[i][j].flatten().detach() for j in range(num_params)]) for i in range(num_samples)])
                if simulator.shuffle:   
                    shuffle_idx = torch.randperm(vacf_grads_flattened.shape[0])
                    vacf_grads_flattened = vacf_grads_flattened[shuffle_idx]
                    vacf_loss_tensor = vacf_loss_tensor[shuffle_idx]
                    vacfs = vacfs[shuffle_idx]
                    
                #now calculate Fabian estimator
                #scale VACF minibatch size to have a similar number of gradient updates as RDF
                #vacf_minibatch_size = math.ceil(MINIBATCH_SIZE / simulator.vacf_window * simulator.n_dump)
                vacf_minibatch_size = MINIBATCH_SIZE
                num_blocks = math.ceil(vacf_grads_flattened.shape[0]/ vacf_minibatch_size)
                start_time = time.time()
                vacf_gradient_estimators = []
                raw_grads = []
                print(f"Computing VACF gradients from {vacf_grads_flattened.shape[0]} trajectories in minibatches of size {vacf_minibatch_size}")
                for i in tqdm(range(num_blocks)):
                    start = vacf_minibatch_size*i
                    end = vacf_minibatch_size*(i+1)
                    vacf_grads_batch = vacf_grads_flattened[start:end]
                    if simulator.use_mse_gradient:
                        #compute VJP with MSE gradient
                        vacf_batch = vacfs[start:end]
                        gradient_estimator = (vacf_grads_batch.mean(0).unsqueeze(0)*vacf_batch.mean(0).unsqueeze(-1) - vacf_grads_batch.unsqueeze(1) * vacf_batch.unsqueeze(-1)).mean(dim=0)
                        grad_outputs = 2*(vacf_batch.mean(0) - gt_vacf).unsqueeze(0) #MSE gradient
                        final_vjp = torch.mm(grad_outputs, gradient_estimator)[0]
                    else:
                        #use loss directly
                        vacf_loss_batch = vacf_loss_tensor[start:end].squeeze(-1)
                        final_vjp = vacf_grads_batch.mean(0)*vacf_loss_batch.mean(0) \
                                            - (vacf_grads_batch*vacf_loss_batch).mean(dim=0)

                    if not simulator.allow_off_policy_updates:
                        raw_grads.append(final_vjp)
                    else:
                        #re-assemble flattened gradients into correct shape
                        gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(final_vjp.split(original_numel), original_shapes)])
                        vacf_gradient_estimators.append(gradient_estimator)
                if not simulator.allow_off_policy_updates:
                    mean_grads = torch.stack(raw_grads).mean(dim=0)
                    #re-assemble flattened gradients into correct shape
                    gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(mean_grads.split(original_numel), original_shapes)])
                    vacf_gradient_estimators.append(gradient_estimator)
                vacf_package = (vacf_gradient_estimators, mean_vacf, mean_vacf_loss.to(simulator.device))
            ###RDF/ADF Stuff
            
            r2d = lambda r: radii_to_dists(r, simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(-1, simulator.n_atoms, simulator.n_atoms-1, 1)
            rdfs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(-1, simulator.n_replicas, gt_rdf.shape[-1]) #this way of calculating uses less memory
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, simulator.n_atoms, 3)]).reshape(-1, simulator.n_replicas, gt_adf.shape[-1]) #this way of calculating uses less memory
            #rdfs = vmap(diff_rdf)(tuple(dists))
            
            #compute mean quantities only on stable replicas
            mean_rdf = rdfs[:, stable_replicas].mean(dim=(0, 1))
            mean_adf = adfs[:, stable_replicas].mean(dim=(0, 1))
            mean_rdf_loss = rdf_loss(mean_rdf)
            mean_adf_loss = adf_loss(mean_adf)
            
            if params.rdf_loss_weight ==0 or not simulator.train or not simulator.all_unstable:
                rdf_gradient_estimators = None
                rdf_package = (rdf_gradient_estimators, mean_rdf, rdf_loss(mean_rdf).to(simulator.device), mean_adf, adf_loss(mean_adf).to(simulator.device))              
            else:
                #only keep the unstable replicas
                mask = ~stable_replicas if params.only_train_on_unstable_replicas \
                        else torch.ones((simulator.n_replicas), dtype=torch.bool).to(simulator.device)
                rdfs = rdfs[:, mask].reshape(-1, rdfs.shape[-1])
                adfs = adfs[:, mask].reshape(-1, adfs.shape[-1])
                stacked_radii = stacked_radii[:, mask]
                
                rdf_loss_tensor = vmap(rdf_loss)(rdfs).unsqueeze(-1).unsqueeze(-1)
                adf_loss_tensor = vmap(adf_loss)(adfs).unsqueeze(-1).unsqueeze(-1)
            
                #TODO: scale the estimator by temperature
                start = time.time()
                stacked_radii = stacked_radii.reshape(-1, simulator.n_atoms, 3)
                #shuffle the radii, rdfs, and losses
                if simulator.shuffle:   
                    shuffle_idx = torch.randperm(stacked_radii.shape[0])
                    stacked_radii = stacked_radii[shuffle_idx]
                    rdf_loss_tensor = rdf_loss_tensor[shuffle_idx]
                    rdfs = rdfs[shuffle_idx]
                    adf_loss_tensor = adf_loss_tensor[shuffle_idx]
                    adfs = adfs[shuffle_idx]
                
                num_blocks = math.ceil(stacked_radii.shape[0]/ (MINIBATCH_SIZE))
                start_time = time.time()
                rdf_gradient_estimators = []
                adf_gradient_estimators = []
                raw_grads = []
                print(f"Computing RDF/ADF gradients from {stacked_radii.shape[0]} structures in minibatches of size {MINIBATCH_SIZE}")
                
                for i in tqdm(range(num_blocks)):
                    if simulator.model_type == "nequip":
                        temp_atoms_batch = simulator.atoms_batch
                    start = MINIBATCH_SIZE*i
                    end = MINIBATCH_SIZE*(i+1)
                    actual_batch_size = min(end, stacked_radii.shape[0]) - start
                    batch = torch.arange(actual_batch_size).repeat_interleave(simulator.n_atoms).to(simulator.device)
                    atomic_numbers = torch.Tensor(simulator.atoms.get_atomic_numbers()).to(torch.long).to(simulator.device).repeat(actual_batch_size)
                    with torch.enable_grad():
                        radii_in = stacked_radii[start:end]
                        radii_in.requires_grad = True
                        if simulator.model_type == "schnet":
                            energy = model(pos = radii_in.reshape(-1, 3), z = atomic_numbers, batch = batch)
                        elif simulator.model_type == "nequip":
                            #construct a batch
                            temp_atoms_batch['pos'] = radii_in.reshape(-1, 3)
                            temp_atoms_batch['batch'] = batch
                            temp_atoms_batch['edge_index'] = radius_graph(radii_in.reshape(-1, 3), r=simulator.model_config["r_max"], batch=batch, max_num_neighbors=32)
                            temp_atoms_batch['edge_cell_shift'] = torch.zeros((temp_atoms_batch['edge_index'].shape[1], 3)).to(simulator.device)
                            #adjust shapes
                            simulator.atoms_batch['cell'] = simulator.atoms_batch['cell'][0].unsqueeze(0).repeat(radii_in.shape[0], 1, 1)
                            simulator.atoms_batch['pbc'] = simulator.atoms_batch['pbc'][0].unsqueeze(0).repeat(radii_in.shape[0], 1)
                            temp_atoms_batch['atom_types'] = simulator.final_atom_types[0:simulator.n_atoms].repeat(radii_in.shape[0], 1)
                            energy = model(temp_atoms_batch)[AtomicDataDict.TOTAL_ENERGY_KEY]

                    def get_vjp(v):
                        return compute_grad(inputs = list(model.parameters()), output = energy, grad_outputs = v, create_graph = False)
                    vectorized_vjp = vmap(get_vjp)
                    I_N = torch.eye(energy.shape[0]).unsqueeze(-1).to(simulator.device)
                    grads_vectorized = vectorized_vjp(I_N)
                    #flatten the gradients for vectorization
                    num_samples = energy.shape[0]
                    num_params = len(list(model.parameters()))
                    grads_flattened= torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_params)]) for j in range(num_samples)])
                    
                    if simulator.use_mse_gradient:
                        
                        rdf_batch = rdfs[start:end]
                        adf_batch = adfs[start:end]
                        gradient_estimator_rdf = (grads_flattened.mean(0).unsqueeze(0)*rdf_batch.mean(0).unsqueeze(-1) - grads_flattened.unsqueeze(1) * rdf_batch.unsqueeze(-1)).mean(dim=0)
                        if params.adf_loss_weight !=0:
                            gradient_estimator_adf = (grads_flattened.mean(0).unsqueeze(0)*adf_batch.mean(0).unsqueeze(-1) - grads_flattened.unsqueeze(1) * adf_batch.unsqueeze(-1)).mean(dim=0)
                            grad_outputs_adf = 2*(adf_batch.mean(0) - gt_adf).unsqueeze(0)
                        #MSE gradient
                        grad_outputs_rdf = 2*(rdf_batch.mean(0) - gt_rdf).unsqueeze(0)
                        #compute VJP with MSE gradient
                        final_vjp = torch.mm(grad_outputs_rdf, gradient_estimator_rdf)[0]
                        if params.adf_loss_weight !=0:
                            final_vjp+= params.adf_loss_weight*torch.mm(grad_outputs_adf, gradient_estimator_adf)[0]
                                        
                    else:
                        #use loss directly
                        rdf_loss_batch = rdf_loss_tensor[start:end].squeeze(-1)
                        adf_loss_batch = adf_loss_tensor[start:end].squeeze(-1)
                        loss_batch = rdf_loss_batch + params.adf_loss_weight*adf_loss_batch
                        final_vjp = grads_flattened.mean(0)*loss_batch.mean(0) \
                                            - (grads_flattened*loss_batch).mean(dim=0)

                    if not simulator.allow_off_policy_updates:
                        raw_grads.append(final_vjp)
                    else:
                        #re-assemble flattened gradients into correct shape
                        final_vjp = tuple([g.reshape(shape) for g, shape in zip(final_vjp.split(original_numel), original_shapes)])
                        rdf_gradient_estimators.append(final_vjp)

                if not simulator.allow_off_policy_updates:
                    mean_vjps = torch.stack(raw_grads).mean(dim=0)
                    #re-assemble flattened gradients into correct shape
                    mean_vjps = tuple([g.reshape(shape) for g, shape in zip(mean_vjps.split(original_numel), original_shapes)])
                    rdf_gradient_estimators.append(mean_vjps)

                # end_time = time.time()
                #print(f"gradient calculation time: {end_time-start_time} seconds")
                rdf_package = (rdf_gradient_estimators, mean_rdf, mean_rdf_loss.to(simulator.device), mean_adf, mean_adf_loss.to(simulator.device))
            
            return equilibriated_simulator, rdf_package, vacf_package, energy_force_package

if __name__ == "__main__":
    setup_logging() 
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    params = types.SimpleNamespace(**config)
    
    params.results_dir = os.path.join(params.log_dir, params.results_dir)

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
    data_path = config['src']
    name = config['name']
    molecule = f"{config['molecule']}" if name == 'md17' or name == 'md22' else ""
    size = config['size']
    model_type = config['model']
    
    print(f"Loading pretrained {model_type} model")
    lmax_string = f"lmax={params.l_max}_" if model_type == "nequip" else ""
    #load the correct checkpoint based on whether we're doing train or val
    if params.train or config["eval_model"] == 'pre': #load energies/forces trained model
        pretrained_model_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{size}_{lmax_string}{model_type}") 
    
    elif 'k' in config["eval_model"]:#load energies/forces model trained on a different dataset size
        new_size = config["eval_model"]
        pretrained_model_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{new_size}_{lmax_string}{model_type}") 

    else: #load observable-finetuned model
        pretrained_model_path = os.path.join(params.results_dir, f"IMPLICIT_{molecule}_{params.exp_name}")

    if model_type == "nequip":
        ckpt_epoch = config['checkpoint_epoch']
        cname = 'best_model.pth' if ckpt_epoch == -1 else f"ckpt{ckpt_epoch}.pth"
        model, model_config = Trainer.load_model_from_training_session(pretrained_model_path, \
                                model_name = cname, device =  torch.device(device))
    elif model_type == "schnet":
        model, model_config = load_schnet_model(path = pretrained_model_path, ckpt_epoch = config['checkpoint_epoch'], device = torch.device(device), train = params.train or params.eval_model == 'pre')
    #count number of trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"{num_params} trainable parameters in {model_type} model")

    # #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)

    integrator_config = INTEGRATOR_CONFIGS[molecule] if name == 'md22' else config['integrator_config']
    timestep = integrator_config["timestep"]
    ttime = integrator_config["ttime"]
    results_dir = os.path.join(params.results_dir, f"IMPLICIT_{molecule}_{params.exp_name}") \
                if params.train else os.path.join(params.results_dir, f"IMPLICIT_{molecule}_{params.exp_name}", "inference", params.eval_model)
    os.makedirs(results_dir, exist_ok = True)

    #load ground truth rdf and VACF
    print("Computing ground truth observables from datasets")
    gt_rdf, gt_adf = find_hr_adf_from_file(data_path, name, molecule, MAX_SIZES[name], params, device)
    contiguous_path = os.path.join(data_path, f'contiguous-{name}', molecule, MAX_SIZES[name], 'val/nequip_npz.npz')
    gt_data = np.load(contiguous_path)
    #TODO: gt vacf doesn't look right
    if name == 'water':
        gt_vels = torch.FloatTensor(gt_data.f.velocities).to(device)
    else:
        gt_traj = torch.FloatTensor(gt_data.f.R).to(device)
        gt_vels = gt_traj[1:] - gt_traj[:-1] #finite difference approx
    
    gt_vacf = DifferentiableVACF(params, device)(gt_vels)
    if params.train:
        np.save(os.path.join(results_dir, 'gt_rdf.npy'), gt_rdf.cpu())
        np.save(os.path.join(results_dir, 'gt_adf.npy'), gt_adf.cpu())
        np.save(os.path.join(results_dir, 'gt_vacf.npy'), gt_vacf.cpu())
    
    min_lr = params.lr / (5 * params.max_times_reduce_lr)

    #outer training loop
    losses = []
    rdf_losses = []
    adf_losses = []
    diffusion_losses = []
    vacf_losses = []
    max_bond_len_devs = []
    energy_rmses = []
    force_rmses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    lrs = []
    resets = []
    best_outer_loss = 100
    writer = SummaryWriter(log_dir = results_dir)

    #top level simulator
    top_level = Stochastic_IFT(params, device)
    changed_lr = False

    for epoch in range(params.n_epochs):
        
        rdf = torch.zeros_like(gt_rdf).to(device)
        rdf_loss = torch.Tensor([0]).to(device)
        vacf_loss = torch.Tensor([0]).to(device)
        diffusion_loss = torch.Tensor([0]).to(device)
        best = False
        grad_cosine_similarity = 0
        ratios = 0
        print(f"Epoch {epoch+1}")
        
        if epoch==0: #draw IC from dataset
            #initialize simulator parameterized by a NN model
            simulator = ImplicitMDSimulator(config, params, model, model_config)
            #initialize outer loop optimizer/scheduler
            if params.optimizer == 'Adam':
                optimizer = torch.optim.Adam(list(simulator.model.parameters()), lr=0)
            elif params.optimizer == 'SGD':
                optimizer = torch.optim.SGD(list(simulator.model.parameters()), lr=0)
            else:
                raise RuntimeError("Optimizer must be either Adam or SGD")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
            print(f"Initialize {simulator.n_replicas} random ICs in parallel")
        #continue from where we left off in the last epoch/batch
        simulator.epoch = epoch
        num_resets = simulator.resume()
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
            equilibriated_simulator, rdf_package, vacf_package, energy_force_package = top_level.apply(simulator, gt_rdf, gt_adf, gt_vacf, params)
            
            #unpack results
            rdf_grad_batches, mean_rdf, rdf_loss, mean_adf, adf_loss = rdf_package
            vacf_grad_batches, mean_vacf, vacf_loss = vacf_package
            energy_grad_batches, force_grad_batches = energy_force_package

            #TODO: figure out why ADF loss is becoming NaN in some cases
            if torch.isnan(adf_loss):
                adf_loss = torch.zeros_like(adf_loss).to(device)

            outer_loss = params.rdf_loss_weight*rdf_loss + params.adf_loss_weight*adf_loss + diffusion_loss + params.vacf_loss_weight*vacf_loss
            print(f"Loss: RDF={rdf_loss.item()}+ADF={adf_loss.item()}+Diffusion={diffusion_loss.item()}+VACF={vacf_loss.item()}={outer_loss.item()}")
            
            #make lengths match for iteration
            if rdf_grad_batches is None:
                rdf_grad_batches = vacf_grad_batches
            if vacf_grad_batches is None:
                vacf_grad_batches = rdf_grad_batches
            if energy_grad_batches is None and force_grad_batches is None:
                energy_grad_batches = rdf_grad_batches
                force_grad_batches = rdf_grad_batches
            
            #manual gradient updates for now
            if vacf_grad_batches or rdf_grad_batches:
                grad_cosine_similarity = []
                ratios = []
                for rdf_grads, vacf_grads, energy_grads, force_grads in zip(rdf_grad_batches, vacf_grad_batches, energy_grad_batches, force_grad_batches): #loop through minibatches
                    optimizer.zero_grad()
                    add_lists = lambda list1, list2, w1, w2: tuple([w1*l1 + w2*l2 \
                                                        for l1, l2 in zip(list1, list2)])
                    obs_grads = add_lists(rdf_grads, vacf_grads, params.rdf_loss_weight, params.vacf_loss_weight)
                    ef_grads = add_lists(energy_grads, force_grads, params.energy_loss_weight, params.force_loss_weight)
                    cosine_similarity, ratio = compare_gradients(obs_grads, ef_grads)
                    grad_cosine_similarity.append(cosine_similarity)#compute gradient similarities
                    ratios.append(ratio)
                    
                    #Loop through each group of parameters and set gradients
                    for param, obs_grad, ef_grad in zip(model.parameters(), obs_grads, ef_grads):
                        param.grad = obs_grad + ef_grad
                        
                    if params.gradient_clipping: #gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if simulator.all_unstable and not changed_lr: #start learning
                        optimizer.param_groups[0]['lr'] = params.lr
                        changed_lr = True
                        if not params.train:
                            np.save(os.path.join(results_dir, f'replicas_stable_time.npy'), simulator.stable_time.cpu().numpy())
                            simulator.stable_time = torch.zeros((simulator.n_replicas,)).to(simulator.device)
                    if optimizer.param_groups[0]['lr'] > 0:
                        optimizer.step()
                scheduler.step(outer_loss) #adjust LR according to observable loss on stable replicas
                grad_cosine_similarity = sum(grad_cosine_similarity) / len(grad_cosine_similarity)
                ratios = sum(ratios) / len(ratios)
            
            if simulator.all_unstable and params.train and (optimizer.param_groups[0]['lr'] < min_lr or num_resets <= params.min_frac_unstable_threshold):
                print(f"Back to data collection")
                simulator.all_unstable = False
                simulator.first_simulation = True
                changed_lr = False
                #reinitialize optimizer and scheduler with LR = 0
                if params.optimizer == 'Adam':
                    optimizer = torch.optim.Adam(list(simulator.model.parameters()), lr=0)
                elif params.optimizer == 'SGD':
                    optimizer = torch.optim.SGD(list(simulator.model.parameters()), lr=0)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

                    
        end = time.time()
        sim_time = end - start
        
        #save rdf, adf, and vacf at the end of the trajectory
        filename = f"rdf_epoch{epoch+1}.npy"
        np.save(os.path.join(results_dir, filename), mean_rdf.cpu().detach().numpy())
        filename = f"adf_epoch{epoch+1}.npy"
        np.save(os.path.join(results_dir, filename), mean_adf.cpu().detach().numpy())
        filename = f"vacf_epoch{epoch+1}.npy"
        np.save(os.path.join(results_dir, filename), mean_vacf.cpu().detach().numpy())
        
        #checkpointing
        if outer_loss < best_outer_loss:
            best_outer_loss = outer_loss
            best = True
        simulator.save_checkpoint(best = best)
        
        torch.cuda.empty_cache()
        gc.collect()
        max_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                norm = torch.linalg.vector_norm(param.grad, dim=-1).max()
                if  norm > max_norm:
                    max_norm = norm
        
        #log stats
        losses.append(outer_loss.item())
        rdf_losses.append(rdf_loss.item())
        adf_losses.append(adf_loss.item())
        vacf_losses.append(vacf_loss.item())
        max_bond_len_devs.append(equilibriated_simulator.max_dev)
        resets.append(num_resets)
        lrs.append(optimizer.param_groups[0]['lr'])
        #energy/force error
        energy_rmse, force_rmse = simulator.energy_force_error(params.n_replicas)
        energy_rmses.append(energy_rmse)
        force_rmses.append(force_rmse)
        # energy_rmses.append(0)
        # force_rmses.append(0)

        sim_times.append(sim_time)
        try:
            grad_norms.append(max_norm.item())
        except:
            grad_norms.append(max_norm)
        if simulator.all_unstable and not params.train:
            #reached instability point, can stop
            break
        writer.add_scalar('Loss', losses[-1], global_step=epoch+1)
        writer.add_scalar('RDF Loss', rdf_losses[-1], global_step=epoch+1)
        writer.add_scalar('ADF Loss', adf_losses[-1], global_step=epoch+1)
        writer.add_scalar('VACF Loss', vacf_losses[-1], global_step=epoch+1)
        writer.add_scalar('Max Bond Length Deviation', max_bond_len_devs[-1], global_step=epoch+1)
        writer.add_scalar('Fraction of Unstable Replicas', resets[-1], global_step=epoch+1)
        writer.add_scalar('Learning Rate', lrs[-1], global_step=epoch+1)
        writer.add_scalar('Energy RMSE', energy_rmses[-1], global_step=epoch+1)
        writer.add_scalar('Force RMSE', force_rmses[-1], global_step=epoch+1)
        writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Cosine Similarity (Observable vs Energy-Force)', grad_cosine_similarity, global_step=epoch+1)
        writer.add_scalar('Gradient Ratios (Observable vs Energy-Force)', ratios, global_step=epoch+1)

    # simulator.f.close()
    # simulator.t.close()

    stats_write_file = os.path.join(simulator.save_dir, 'stats.txt')
    with open(stats_write_file, "w") as output:
        output.write("Losses: " + str(losses) + "\n")
        output.write("Simulation times: " +  str(sim_times) + "\n")
        output.write("Gradient calculation times: " +  str(grad_times) + "\n")
        output.write("Max gradient norms: " + str(grad_norms))

    writer.close()
    print('Done!')
    


