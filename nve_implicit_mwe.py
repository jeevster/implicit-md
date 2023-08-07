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
        
        #Initialize model
        self.model = model
        self.model_config = model_config

        self.atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(self.n_replicas)
        self.batch = torch.arange(self.n_replicas).repeat_interleave(self.n_atoms).to(self.device)
        self.ic_stddev = params.ic_stddev

        #Register inner parameters
        samples = np.random.choice(np.arange(self.train_dataset.__len__()), self.n_replicas)
        actual_atoms = [self.train_dataset.__getitem__(i) for i in samples]
        radii = torch.stack([torch.Tensor(data_to_atoms(atoms).get_positions()) for atoms in actual_atoms])
        self.radii = (radii + torch.normal(torch.zeros_like(radii), self.ic_stddev)).to(self.device).requires_grad_(True)
        self.velocities = torch.Tensor(initialize_velocities(self.n_atoms, self.masses, self.temp, self.n_replicas)).to(self.device).requires_grad_(True)

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


    def force_calc(self, radii, atomic_numbers = None, batch = None, retain_grad = True):
        batch_size = radii.shape[0]
        batch = torch.arange(batch_size).repeat_interleave(self.n_atoms).to(self.device)
        atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(batch_size)
        energy = self.model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)
        forces = -compute_grad(inputs = radii, output = energy) if retain_grad else -compute_grad(inputs = radii, output = energy).detach()
        assert(not torch.any(torch.isnan(forces)))
        return forces

    #define Onsager-Machlup Action ("energy" of each trajectory)
    def om_action(self, vel_traj, radii_traj, force_traj = None):
        v_tp1 = vel_traj[:, :, 1:]
        v_t = vel_traj[:, :, :-1]
        if force_traj is None:
            r = radii_traj[:, :, 1:].reshape(-1, self.n_atoms, 3)
            batch_size = r.shape[0]
            batch = torch.arange(batch_size).repeat_interleave(self.n_atoms).to(self.device)
            atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device).repeat(batch_size)
            a_tp1 = self.force_calc(r, atomic_numbers = atomic_numbers, batch = batch).reshape(v_t.shape) / self.masses.unsqueeze(1).unsqueeze(1)
        else:
            a_tp1 = force_traj[:, :, 1:] / self.masses.unsqueeze(1).unsqueeze(1)
        diff = (v_tp1 - v_t - a_tp1*self.dt + self.gamma*v_t*self.dt)
        om_action = diff**2 / self.noise_f.unsqueeze(1).unsqueeze(1) #this is exponentially distributed
        #sum over euclidean dimensions, atoms, and vacf window: TODO: this ruins the exponential property
        return diff/self.noise_f.unsqueeze(1).unsqueeze(1), om_action.sum((-3, -2, -1))
                
    def solve(self):
        '''DEBUGGING: compare gradients from naive backprop with gradients from adjoint method'''
                
        running_radii = []
        running_vels = []
        running_accs = []
        running_noise = []
        running_forces = []
        with torch.enable_grad():
            radii = self.radii
            velocities = self.velocities
            assert(radii.requires_grad and velocities.requires_grad)
            forces = self.force_calc(radii)
            velocities = velocities + self.dt/2*(forces/self.masses - self.gamma * velocities) + self.noise_f/torch.sqrt(torch.tensor(2.0).to(self.device)) * torch.randn_like(velocities) 
            print("Collect MD Simulation Data")
            for i in tqdm(range(self.vacf_window)):
                new_radii = radii.detach() + self.dt*velocities
                new_forces = self.force_calc(new_radii)
                noise = torch.randn_like(velocities)
                new_velocities = velocities + self.dt*(new_forces/self.masses - self.gamma * velocities) + self.noise_f * noise
                running_radii.append(new_radii)
                running_vels.append(new_velocities)
                running_accs.append(new_forces/self.masses)
                running_noise.append(noise)
                running_forces.append(new_forces)
                #update
                radii = new_radii
                velocities = new_velocities
                forces = new_forces

            #tensorize the saved trajectories
            radii_traj = torch.stack(running_radii).permute(1,0,2,3)
            velocities_traj = torch.stack(running_vels).permute(1,0,2,3)
            accel_traj = torch.stack(running_accs).permute(1,0,2,3)
            noise_traj = torch.stack(running_noise).permute(1,0,2,3)
            force_traj = torch.stack(running_forces).permute(1,0,2,3)

            #reshape based on dynamics window
            radii_traj = radii_traj.reshape(velocities_traj.shape[0], -1, self.vacf_window, self.n_atoms, 3)
            velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, self.vacf_window, self.n_atoms, 3)
            noise_traj = noise_traj.reshape(noise_traj.shape[0], -1, self.vacf_window, self.n_atoms, 3)
            force_traj = force_traj.reshape(force_traj.shape[0], -1, self.vacf_window, self.n_atoms, 3)
            diff, om_act = self.om_action(velocities_traj, radii_traj)
            assert(torch.allclose(diff, noise_traj[:, :, 1:], atol = 1e-3)) #make sure the diffs match the stored noises along the trajectory
            
            print("Compute gradients by naive backprop")
            naive_backprop_grads = [[g.detach() if g is not None else torch.Tensor([0.]).to(self.device) for g in torch.autograd.grad(o, model.parameters(), create_graph = True, allow_unused = True)] for o in tqdm(om_act.flatten())]
            
            print("Compute adjoints by naive backprop")
            #r adjoints are pretty tiny, v adjoints have reasonable values (as expected due to detach?)
            naive_r_adjoints = torch.stack([compute_grad(inputs = r, output = om_act).detach() for r in tqdm(reversed(running_radii))])
            naive_v_adjoints = torch.stack([compute_grad(inputs = v, output = om_act).detach()  for v in tqdm(reversed(running_vels))])
            naive_adjoint_norms = naive_v_adjoints.norm(dim = (-2, -1))
        #get initial adjoint states
        import pdb; pdb.set_trace()
        pure_partials = compute_grad(inputs = velocities_traj, output = om_act).detach()
        partials_via_x = compute_grad(inputs = radii_traj, output = om_act).detach()
        zeros = torch.zeros_like(partials_via_x[:, :, 0]).unsqueeze(2).to(self.device)
        partials_via_x = torch.cat([partials_via_x, zeros], dim = 2)
        grad_outputs = pure_partials + self.dt*partials_via_x[:, :, 1:]
        #reshape to join replica and sample dimensions
        radii_traj = radii_traj.reshape(accel_traj.shape)
        velocities_traj = velocities_traj.reshape(accel_traj.shape)
        grad_outputs = grad_outputs.reshape(accel_traj.shape)
        
        #run backward dynamics
        print(f"Run backward dynamics to calculate adjoints:")
        start = time.time()
        final_adjoints, adjoint_norms, R = self.get_adjoints(radii_traj, velocities_traj, grad_outputs, force_fn = self.force_calc)
        end = time.time()

        #log adjoint norms
        np.save(os.path.join(self.save_dir, f'adjoint_norms_epoch0'), adjoint_norms.cpu().numpy())
        np.save(os.path.join(self.save_dir, f'naive_adjoint_norms_epoch0'), naive_adjoint_norms.cpu().numpy())
        #print(f"Adjoint calculation time: {end - start} s")
        #now get dO/dtheta (where O is the OM action)
        #Loop over trajectories for now
        def calc_grads(adjoints, radii):
            with torch.enable_grad():
                radii.requires_grad=True
                forces = self.force_calc(radii)
            #compute gradient of force w.r.t model params
            #some of the model gradients are zero - have to use allow_unused - I believe these correspond to the bias parameters which don't affect the force calculation (only the energies)?
            grads = [g.detach() if g is not None else torch.Tensor([0.]).to(self.device) \
                        for g in torch.autograd.grad(forces, model.parameters(), \
                                adjoints, create_graph = True, allow_unused = True)]
            return grads
        print(f"Calculate gradients of Onsager-Machlup action for {len(final_adjoints)} trajectories")
        grads = [calc_grads(adj, r) for adj, r in tqdm(zip(final_adjoints, R))]
        #flatten out the grads
        num_params = len(list(model.parameters()))
        num_samples = final_adjoints.shape[0]

        
        naive_backprop_grads_flattened = torch.stack([torch.cat([naive_backprop_grads[i][j].flatten().detach() \
                                for j in range(num_params)]) for i in range(num_samples)])
        vacf_grads_flattened = torch.stack([torch.cat([grads[i][j].flatten().detach() \
                                for j in range(num_params)]) for i in range(num_samples)])
        ratios = (naive_backprop_grads_flattened + 1e-8) / (vacf_grads_flattened + 1e-8)
        mask = ratios != 1
        ratios = ratios[mask].reshape(final_adjoints.shape[0], -1)
        
        return naive_backprop_grads_flattened, vacf_grads_flattened
        

    def get_adjoints(self, pos_traj, vel_traj, grad_outputs, force_fn = None):
        with torch.no_grad():
            a_dt = self.dt*1 #save frequency = 1 for now
            M = self.masses[0].unsqueeze(0)
            adjoints = []
            adjoint_norms = []
            testR = []
    
            #initial adjoints
            #grad_outputs *= a_dt**2/M #pre-multiply the grad outputs to make norms closer to naive backprop (still not sure why this is needed)
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
    
    simulator = ImplicitMDSimulator(config, params, model, model_config)
    naive_backprop_grads, vacf_grads = simulator.solve()

       
    


