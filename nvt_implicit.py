import numpy as np
import gsd.hoomd
import torch
import logging
import gc
import json
from pathlib import Path
import yaml
import torch.nn as nn
import math
import shutil
from nff.utils.scatter import compute_grad
from YParams import YParams
import argparse
import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
from tqdm import tqdm
import pstats
import random
import types
from torch_geometric.nn import MessagePassing, radius_graph
from torchmd.observable import generate_vol_bins, DifferentiableRDF, DifferentiableADF, DifferentiableVelHist, DifferentiableVACF, SelfIntermediateScattering, msd, DiffusionCoefficient
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.utils.data import DataLoader
from functorch import vmap, vjp
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
#NNIP stuff:
import ase
from ase import Atoms, units
import nequip.scripts.deploy
from nequip.train.trainer import Trainer
from nequip.train.loss import Loss
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.AtomicData import neighbor_list_and_relative_vec
from nequip.utils.torch_geometric import Batch, Dataset
from nequip.utils import atomic_write, load_file
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.md.ase_utils import OCPCalculator
from mdsim.md.integrator import get_stress
from mdsim.common.registry import registry
from mdsim.common.utils import extract_cycle_epoch, save_checkpoint, cleanup_atoms_batch, setup_imports, setup_logging, compute_bond_lengths, data_to_atoms, atoms_to_batch, atoms_to_state_dict, convert_atomic_numbers_to_types, process_gradient, compare_gradients, initialize_velocities, dump_params_to_yml
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.common.utils import load_config
from mdsim.modules.evaluator import Evaluator
from mdsim.modules.normalizer import Normalizer
from utils import calculate_final_metrics
from mdsim.observables.common import distance_pbc, BondLengthDeviation, radii_to_dists, compute_distance_matrix_batch
from mdsim.observables.md17_22 import find_hr_adf_from_file, get_hr
from mdsim.observables.water import WaterRDFMAE, MinimumIntermolecularDistance, find_water_rdfs_diffusivity_from_file, get_water_rdfs, get_smoothed_diffusivity, n_closest_molecules
from mdsim.observables.lips import LiPSRDFMAE, find_lips_rdfs_diffusivity_from_file, cart2frac, frac2cart
from mdsim.models.load_models import load_pretrained_model
from mdsim.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
    compose_data_cfg
)
from mdsim.common.flags import flags
from boltzmann_estimator import BoltzmannEstimator
from npt_temp import NPT

MAX_SIZES = {'md17': '10k', 'md22': '100percent', 'water': '10k', 'lips': '20k'}

class ImplicitMDSimulator():
    def __init__(self, config, params, model, model_path, model_config, gt_rdf):
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
        self.optimizer = config["optimizer"]
        self.use_mse_gradient = config["use_mse_gradient"]
        self.adjoint = config["adjoint"]
        #set stability criterion
        self.stability_tol = config["imd_tol"] if self.pbc else config["bond_dev_tol"]
        self.max_frac_unstable_threshold = config["max_frac_unstable_threshold"]
        self.min_frac_unstable_threshold = config["min_frac_unstable_threshold"]
        self.results_dir = os.path.join(config["log_dir"], config["results_dir"])
        self.eval_model = config["eval_model"]
        self.n_dump = config["n_dump"]
        if self.name == 'water':
            self.n_local_neighborhoods = config["n_local_neighborhoods"]
            self.n_closest_molecules = config["n_closest_molecules"]
            self.n_atoms_local = 3*(self.n_closest_molecules + 1)
        self.n_dump_vacf = config["n_dump_vacf"]

        #Initialize model
        self.curr_model_path = model_path
        self.model = model
        if self.model_type == "nequip":
            self.rescale_layers = []
            outer_layer = self.model
            while hasattr(outer_layer, "unscale"):
                self.rescale_layers.append(outer_layer)
                outer_layer = getattr(outer_layer, "model", None)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model_config = model_config

        #initialize datasets
        train_src = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'train')
        test_src = os.path.join(self.data_dir, self.name, self.molecule, MAX_SIZES[self.name], 'test')
        
        self.train_dataset = LmdbDataset({'src': train_src})
        self.test_dataset = LmdbDataset({'src': test_src})
        self.train_dataloader = DataLoader(self.train_dataset, collate_fn=data_list_collater, \
                                batch_size = self.minibatch_size)
 
        #get random initial condition from dataset
        init_data = self.train_dataset.__getitem__(10)
        self.n_atoms = init_data['pos'].shape[0]
        self.atoms = data_to_atoms(init_data)

        #extract bond and atom type information
        NL = NeighborList(natural_cutoffs(self.atoms), self_interaction=False)
        NL.update(self.atoms)
        self.bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(self.device).T
        #filter out extra edges (don't know why they're there)
        if self.name == 'water':
            mask = torch.abs(self.bonds[:, 0] - self.bonds[:, 1]) <=2
            self.bonds = self.bonds[mask]
        
        self.atom_types = self.atoms.get_chemical_symbols()
        #atom type mapping
        if self.model_type == "nequip":
            type_names = self.model_config['model'][nequip.scripts.deploy.TYPE_NAMES_KEY]
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
        self.DATAPATH_TRAIN = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'train') 
        self.DATAPATH_TEST = os.path.join(self.data_dir, self.name, self.molecule, self.size, 'test')
        
        gt_data_train = np.load(os.path.join(self.DATAPATH_TRAIN, 'nequip_npz.npz'))
        if self.name == 'water':
            pos_field_train = gt_data_train.f.wrapped_coords
            self.gt_data_spacing_fs = 10
        elif self.name == 'lips':
            pos_field_train = gt_data_train.f.pos
            self.gt_data_spacing_fs = 2.0 #TODO: check this
        else:
            pos_field_train = gt_data_train.f.R
            if self.name == "md17":
                self.gt_data_spacing_fs = 0.5
            else:
                self.gt_data_spacing_fs = 1
        
        self.gt_traj_train = torch.FloatTensor(pos_field_train).to(self.device)
        
        
        self.instability_per_replica = 100*torch.ones((self.n_replicas,)).to(self.device) if params.stability_criterion == 'imd' \
                                                    else torch.zeros((self.n_replicas,)).to(self.device)
        self.stable_time = torch.zeros((self.n_replicas,)).to(self.device)
        
        self.integrator = self.config["integrator"]
        #Nose-Hoover Thermostat stuff
        self.integrator_config =  config['integrator_config']
        self.dt = config["timestep"] * units.fs
        self.temp = config["temperature"]
        
        print(f"Simulation Temperature: {self.temp}")
        
        # adjust units.
        if self.integrator in ['NoseHoover', 'NoseHooverChain', 'Langevin', 'Berendsen', 'NPT']:
            self.temp *= units.kB
        self.targeEkin = 0.5 * (3.0 * self.n_atoms) * self.temp
        
        self.ttime = self.integrator_config["ttime"]
        self.Q = 3.0 * self.n_atoms * self.temp * (self.ttime * self.dt)**2
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
        self.masses = torch.Tensor(self.atoms.get_masses().reshape(1, -1, 1)).to(self.device)


        #Langevin thermostat stuff
        self.gamma = self.integrator_config["gamma"] / (1000*units.fs)
        self.noise_f = (2.0 * self.gamma/self.masses * self.temp * self.dt).sqrt().to(self.device)

        # Berendsen thermostat stuff
        self.taup = self.integrator_config["taup"] * units.fs
        self.taut = self.integrator_config["taut"] * units.fs
        self.compressibility_au = self.integrator_config["compressibility_au"]
        self.pressure_au = torch.tensor([self.integrator_config["pressure_au"]]).to(self.device)
        self.fix_com = self.integrator_config["fix_com"]

        # NPT thermostat Stuff
        self.ptime = self.integrator_config["ptime"] * units.fs
        bulk_modulus = 2 * units.GPa # bulk modulus of water
        self.pfactor = self.ptime **2 * bulk_modulus

        self.nsteps = params.steps
        self.eq_steps = params.eq_steps
        #ensure that the number of logged steps is a multiple of the vacf window (for chopping up the trajectory)
        self.nsteps -= (self.nsteps - self.eq_steps) % self.vacf_window
        if (self.nsteps - self.eq_steps) < self.vacf_window:
            self.nsteps = self.eq_steps + 2*self.vacf_window #at least two windows
        while self.nsteps < params.steps: #nsteps should be at least as long as what was requested
            self.nsteps += self.vacf_window
        self.ps_per_epoch = self.nsteps * self.config["timestep"] // 1000.
        

        self.atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        self.batch = torch.arange(self.n_replicas).repeat_interleave(self.n_atoms).to(self.device)
        self.ic_stddev = params.ic_stddev

        dataset = self.train_dataset if self.train else self.test_dataset
        #pick the first n_replicas ICs if doing simulation (so we can directly compare replicas' stabilities across models)
        samples = np.random.choice(np.arange(dataset.__len__()), self.n_replicas, replace=False) \
                                    if self.train else np.arange(self.n_replicas) 

        self.raw_atoms = [data_to_atoms(dataset.__getitem__(i)) for i in samples]
        self.cell = torch.Tensor(self.raw_atoms[0].cell).to(self.device)
        if self.name == "lips":
            dists = compute_distance_matrix_batch(self.cell,self.gt_traj_train)
            self.mean_bond_lens = dists[:, self.bonds[:, 0], self.bonds[:, 1]].mean(dim=0)
        else:
            bond_lens = torch.stack([distance_pbc(
            gt_traj_train.unsqueeze(0)[:, self.bonds[:, 0]], gt_traj_train.unsqueeze(0)[:, self.bonds[:, 1]], \
                        torch.diag(self.cell).to(self.device)).mean(dim=0) for gt_traj_train in self.gt_traj_train])
            self.mean_bond_lens = bond_lens.mean(0)
            self.bond_lens_var = bond_lens.var(0)
            

        
        self.gt_rdf = gt_rdf
        #choose the appropriate stability criterion based on the type of system
        if self.name == 'water':
            if params.stability_criterion == 'imd':
                self.stability_criterion = MinimumIntermolecularDistance(self.bonds, self.cell, self.device)
            else:
                self.stability_criterion = WaterRDFMAE(self.data_dir, self.gt_rdf, self.n_atoms, self.params, self.device)
            self.rdf_mae = WaterRDFMAE(self.data_dir, self.gt_rdf, self.n_atoms, self.params, self.device)
            self.min_imd = MinimumIntermolecularDistance(self.bonds, self.cell, self.device)
            self.bond_length_dev = BondLengthDeviation(self.name, self.bonds,self.mean_bond_lens,self.cell,self.device)
        elif self.name == 'lips':
            if params.stability_criterion == 'imd':
                self.stability_criterion = MinimumIntermolecularDistance(self.bonds, self.cell, self.device)
            else:
                self.stability_criterion = LiPSRDFMAE(self.data_dir, self.gt_rdf, self.n_atoms, self.n_replicas, self.params, self.device)
            self.rdf_mae = LiPSRDFMAE(self.data_dir, self.gt_rdf, self.n_atoms, self.n_replicas, self.params, self.device)
            self.bond_length_dev = BondLengthDeviation(self.name, self.bonds,self.mean_bond_lens,self.cell,self.device)
        else:
            self.stability_criterion = BondLengthDeviation(self.name, self.bonds,self.mean_bond_lens,self.cell,self.device)
        radii = torch.stack([torch.Tensor(atoms.get_positions()) for atoms in self.raw_atoms])
        self.radii = (radii + torch.normal(torch.zeros_like(radii), self.ic_stddev)).to(self.device)
        self.velocities = torch.Tensor(initialize_velocities(self.n_atoms, self.masses, self.temp, self.n_replicas)).to(self.device)

        #assign velocities to atoms
        for i in range(len(self.raw_atoms)):
            self.raw_atoms[i].set_velocities(self.velocities[i].cpu().numpy())
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
        self.all_radii = []
        
        #create batch of atoms to be operated on
        self.r_max_key = "r_max" if self.model_type == "nequip" else "cutoff"
        self.atoms_batch = [AtomicData.from_ase(atoms=a, r_max=self.model_config['model'][self.r_max_key]) for a in self.raw_atoms]
        self.atoms_batch = Batch.from_data_list(self.atoms_batch) #DataBatch for non-Nequip models
        self.atoms_batch['natoms'] = torch.Tensor([self.n_atoms]).repeat(self.n_replicas).to(self.device)
        self.atoms_batch['cell'] = self.atoms_batch['cell'].to(self.device)
        self.atoms_batch['atomic_numbers'] =self.atoms_batch['atomic_numbers'].squeeze().to(torch.long).to(self.device)
        #convert to dict for Nequip
        if self.model_type == "nequip":
            self.atoms_batch = AtomicData.to_AtomicDataDict(self.atoms_batch) 
            self.atoms_batch = {k: v.to(self.device) for k, v in self.atoms_batch.items()}
            self.atoms_batch['atom_types'] = self.final_atom_types
            del self.atoms_batch['ptr']
            del self.atoms_batch['atomic_numbers']

        # NPT Integrator
        cell = self.cell.unsqueeze(0).repeat(self.n_replicas, 1, 1)
        if self.integrator == 'NPT':
            self.npt_integrator = NPT(self.atoms_batch, 
                                    self.radii, 
                                    self.velocities, 
                                    self.masses, 
                                    cell, 
                                    self.pbc, 
                                    self.atomic_numbers, 
                                    self.force_calc, 
                                    self.dt, 
                                    self.temp, 
                                    self.pressure_au, 
                                    self.ttime * units.fs, 
                                    self.pfactor)
            
        self.diameter_viz = params.diameter_viz
        self.exp_name = params.exp_name
        self.training_observable = params.training_observable
        self.rdf_loss_weight = params.rdf_loss_weight
        self.diffusion_loss_weight = params.diffusion_loss_weight
        self.vacf_loss_weight = params.vacf_loss_weight
        self.energy_force_loss_weight = params.energy_force_loss_weight
        if self.vacf_loss_weight != 0:
            self.integrator = 'Langevin'
        
        #limit CPU usage
        torch.set_num_threads(1)

        #define vectorized differentiable rdf and vacf
        self.diff_rdf = vmap(DifferentiableRDF(params, self.device), -1)
        self.diff_vacf = vmap(DifferentiableVACF(params, self.device))
    
        molecule_for_name = self.name if self.name =='water' or self.name == 'lips' else self.molecule
        name = f"IMPLICIT_{self.model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}"
        self.save_dir = os.path.join(self.results_dir, name) if self.train else os.path.join(self.results_dir, name, 'inference', self.eval_model)
        os.makedirs(self.save_dir, exist_ok = True)
        dump_params_to_yml(self.params, self.save_dir)
        #File dump stuff
        self.f = open(f"{self.save_dir}/log.txt", "a+")

        #initialize trainer to calculate energy/force gradients
        if self.train:
            if self.model_type == "nequip":
                self.train_dict = load_file(
                supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
                filename=os.path.join(self.save_dir, "trainer.pth"),
                enforced_format=None,
                )
                self.nequip_loss = Loss(coeffs = self.train_dict['loss_coeffs'])
            else:
                self.trainer = OCPCalculator(config_yml=self.model_config, checkpoint=self.curr_model_path, 
                                        test_data_src=self.DATAPATH_TEST, 
                                        energy_units_to_eV=1.).trainer
         
    def stability_per_replica(self):
        stability = (self.instability_per_replica > self.stability_tol) if self.params.stability_criterion == 'imd' else (self.instability_per_replica < self.stability_tol)
        return stability

    '''compute energy/force error on test set'''
    def energy_force_error(self):
        self.model_config['model']['name'] = self.model_type
        if self.model_type == 'nequip':
            data_config = f"configs/{self.name}/nequip_data_cfg/{self.molecule}.yml"
            # call nequip evaluation script
            os.system(f'nequip-evaluate --train-dir {os.path.dirname(self.curr_model_path)} \
                        --model {self.curr_model_path} --dataset-config {data_config} \
                            --log {os.path.dirname(self.curr_model_path)}/test_metric.log --batch-size 4')
            with open(f'{os.path.dirname(self.curr_model_path)}/test_metric.log', 'r') as f:
                test_log = f.read().splitlines()
                for i, line in enumerate(test_log):
                    if 'Final result' in line:
                        test_log = test_log[(i+1):]
                        break
                test_metrics = {}
                for line in test_log:
                    k, v = line.split('=')
                    k = k.strip()
                    v = float(v.strip())
                    test_metrics[k] = v
            return test_metrics['e_mae'], test_metrics['f_rmse'], test_metrics['e_mae'], test_metrics['f_mae']
        #non-Nequip models use OCP calculator
        else:
            self.calculator = OCPCalculator(config_yml=self.model_config, checkpoint=self.curr_model_path, 
                                    test_data_src=self.DATAPATH_TEST, 
                                    energy_units_to_eV=1.) 
            print(f"Computing bottom-up (energy-force) error on test set")
            test_metrics = self.calculator.trainer.validate('test', max_points=1000)
            test_metrics = {k: v['metric'] for k, v in test_metrics.items()}
            return test_metrics['energy_rmse'], test_metrics['forces_rmse'], \
                    test_metrics['energy_mae'], test_metrics['forces_mae']

    def energy_force_gradient(self):
        #store original shapes of model parameters
        original_numel = [param.data.numel() for param in self.model.parameters()]
        original_shapes = [param.data.shape for param in self.model.parameters()]
        print(f"Computing gradients of bottom-up (energy-force) objective on {self.train_dataset.__len__()} samples")
        gradients = []
        losses = []
        if self.model_type == "nequip":
            with torch.enable_grad():
                for data in tqdm(self.train_dataloader):
                    # Do any target rescaling
                    data = data.to(self.device)
                    data = AtomicData.to_AtomicDataDict(data)
                    actual_batch_size = int(data['pos'].shape[0]/self.n_atoms)
                    data['cell'] = data['cell'][0].unsqueeze(0).repeat(actual_batch_size, 1, 1)
                    data['pbc'] = self.atoms_batch['pbc'][0].unsqueeze(0).repeat(actual_batch_size, 1)
                    data['atom_types'] = self.atoms_batch['atom_types'][0:self.n_atoms].repeat(actual_batch_size, 1)

                    data_unscaled = data
                    for layer in self.rescale_layers:
                        # This means that self.model is RescaleOutputs
                        # this will normalize the targets
                        # in validation (eval mode), it does nothing
                        # in train mode, it normalizes the targets
                        data_unscaled = layer.unscale(data_unscaled)    
                    # Run model
                    data_unscaled['edge_index'] = radius_graph(data_unscaled['pos'].reshape(-1, 3), r=self.model_config['model'][self.r_max_key], batch=data_unscaled['batch'], max_num_neighbors=32)
                    data_unscaled['edge_cell_shift'] = torch.zeros((data_unscaled['edge_index'].shape[1], 3)).to(self.device)
                    out = self.model(data_unscaled)
                    data_unscaled['forces'] = data_unscaled['force']
                    data_unscaled['total_energy'] = data_unscaled['y'].unsqueeze(-1)
                    loss, _ = self.nequip_loss(pred=out, ref=data_unscaled)
                    grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused = True)
                    gradients.append(process_gradient(self.model.parameters(), grads, self.device))
                    losses.append(loss.detach())
        else:
            self.trainer.model = self.model
            with torch.enable_grad():
                for batch in tqdm(self.train_dataloader):
                    with torch.cuda.amp.autocast(enabled=self.trainer.scaler is not None):
                        for key in batch.keys:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(self.device)
                        out = self.trainer._forward(batch)
                        loss = self.trainer._compute_loss(out, [batch])
                        loss = self.trainer.scaler.scale(loss) if self.trainer.scaler else loss
                        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused = True)
                        gradients.append(process_gradient(self.model.parameters(), grads, self.device))
                        losses.append(loss.detach())
        grads_flattened = torch.stack([torch.cat([param.flatten().detach()\
                                    for param in grad]) for grad in gradients])
        mean_grads = grads_flattened.mean(0)
        final_grads = tuple([g.reshape(shape) for g, shape in zip(mean_grads.split(original_numel), original_shapes)])
        return final_grads

    def set_starting_states(self):
        #find replicas which violate the stability criterion
        reset_replicas = ~self.stability_per_replica()
        num_unstable_replicas = reset_replicas.count_nonzero().item()
        if num_unstable_replicas / self.n_replicas >= self.max_frac_unstable_threshold: #threshold of unstable replicas reached
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
        return num_unstable_replicas / self.n_replicas


    def force_calc(self, radii, cell = None, retain_grad = False, output_individual_energies = False):
        batch_size = radii.shape[0]
        batch = torch.arange(batch_size).repeat_interleave(self.n_atoms).to(self.device)
        with torch.enable_grad():
            if not radii.requires_grad:
                radii.requires_grad = True
            
            if self.pbc:
                # wrap
                if cell is not None:
                    diag = vmap(torch.diag)(cell).unsqueeze(1)
                else:
                    diag = torch.diag(self.cell)
                radii = ((radii / diag) % 1) * diag  - diag/2
            
            #assign radii and batch
            self.atoms_batch['pos'] = radii.reshape(-1, 3)
            self.atoms_batch['batch'] = batch
            #make these match the number of replicas (different from n_replicas when doing bottom-up stuff)
            self.atoms_batch['cell'] = self.atoms_batch['cell'][0].unsqueeze(0).repeat(batch_size, 1, 1)
            self.atoms_batch['pbc'] = self.atoms_batch['pbc'][0].unsqueeze(0).repeat(batch_size, 1)
            all_energies = None
            if self.model_type == "nequip":
                self.atoms_batch['atom_types'] = self.atoms_batch['atom_types'][0:self.n_atoms].repeat(batch_size, 1)
                #recompute neighbor list
                self.atoms_batch['edge_index'] = radius_graph(radii.reshape(-1, 3), r=self.model_config['model'][self.r_max_key], batch=batch, max_num_neighbors=32)
                #TODO: edge cell shift is nonzero for LiPS (non cubic cell)
                self.atoms_batch['edge_cell_shift'] = torch.zeros((self.atoms_batch['edge_index'].shape[1], 3)).to(self.device)
                atoms_updated = self.model(self.atoms_batch)
                energy = atoms_updated[AtomicDataDict.TOTAL_ENERGY_KEY]
                forces = atoms_updated[AtomicDataDict.FORCE_KEY].reshape(-1, self.n_atoms, 3)
            else:
                self.atoms_batch = cleanup_atoms_batch(self.atoms_batch)
                self.atoms_batch['natoms'] = torch.Tensor([self.n_atoms]).repeat(batch_size).to(self.device)
                self.atoms_batch['atomic_numbers'] = self.atomic_numbers.repeat(batch_size)
                if output_individual_energies:
                    if self.model_type == 'gemnet_t':
                        
                        energy, all_energies, forces = self.model(self.atoms_batch, output_individual_energies = True)
                    else:
                        raise RuntimeError(f"Outputting individual energies is only supported for gemnet_t, not {self.model_type}")
                else:
                    energy, forces = self.model(self.atoms_batch)
                forces = forces.reshape(-1, self.n_atoms, 3)
            assert(not torch.any(torch.isnan(forces)))
            energy = energy.detach() if not retain_grad else energy
            forces = forces.detach() if not retain_grad else forces
            if all_energies is None:
                return energy, forces
            else:
                return energy, all_energies, forces

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
        energy, forces = self.force_calc(radii, retain_grad)
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
            print(self.step, self.thermo_log(energy, forces), file=self.f)
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
        energy, forces = self.force_calc(radii, retain_grad)
        noise = torch.randn_like(velocities)
        #full step in velocities
        velocities = velocities + self.dt*(forces/self.masses - self.gamma * velocities) + self.noise_f * noise
        # # dump frames
        if self.step%self.n_dump == 0:
            print(self.step, self.thermo_log(energy, forces), file=self.f)
            step  = self.step if self.train else (self.epoch+1) * self.step #don't overwrite previous epochs at inference time
            try:    
                self.t.append(self.create_frame(frame = self.step/self.n_dump))
            except:
                pass
        return radii, velocities, forces, noise

    
    def forward_berendsen(self, radii, velocities, forces, cell, retain_grad = False):
        """
        Berendsen thermostat for NPT simulation. Adapted from https://wiki.fysik.dtu.dk/ase/_modules/ase/md/nptberendsen.html#NPTBerendsen.
        """

        # scale velocities based on current temperature
        
        tautscl = self.dt / self.taut
        p_dof = 3*self.n_atoms
        ke = 1/2 * (self.masses*torch.square(velocities)).sum(axis = (1,2)).unsqueeze(-1)
        old_temperature = (2*ke/p_dof) / units.kB

        temp = torch.tensor(self.temp).to(self.device)
        scl_temperature = torch.sqrt(1.0 + ((temp/units.kB) / old_temperature - 1.0) * tautscl).unsqueeze(-1)
        
        # Limit the velocity scaling to reasonable values
        scl_temperature = torch.clamp(scl_temperature, 0.9, 1.1)
        
        p = self.masses * velocities
        p = scl_temperature * p
        velocities = p / self.masses

        # scale positions and cell
        taupscl = self.dt / self.taup
        volume = vmap(torch.diag)(cell).prod(dim = 1) # assumes cubic cell
        # calculate stress
        stress = get_stress(radii, velocities, forces, self.masses, volume)
        # stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True) # TODO: replace
        
        old_pressure = -vmap(torch.diag)(stress).mean(dim = 1)
        scl_pressure = (1.0 - taupscl * self.compressibility_au / 3.0 * (self.pressure_au - old_pressure))

        # TODO: make sure we are doing this correctly - updating cell with atom position scaling
        new_cell = cell
        new_cell = scl_pressure.unsqueeze(-1).unsqueeze(-1) * new_cell
        M = torch.linalg.solve(cell, new_cell)
        radii = torch.bmm(radii, M)
        cell = new_cell
        # self.atoms.set_cell(cell, scale_atoms=True)

        # one step velocity verlet

        p = self.masses * velocities
        p += 0.5 * self.dt * forces

        if self.fix_com:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=1, keepdims=True) / p.shape[1]
            p = p - psum

        radii = radii + self.dt * p / self.masses

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        velocities = p / self.masses
        _, forces = self.force_calc(radii, retain_grad)
        accel = forces / self.masses
        
        velocities = velocities + 0.5 * self.dt * accel

        return radii, velocities, forces, cell

    #main MD loop
    def solve(self):
        self.mode = 'learning' if self.optimizer.param_groups[0]['lr'] > 0 else 'simulation'
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
        if not self.all_unstable:
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
        with torch.no_grad():
            self.step = -1
            _, forces = self.force_calc(self.radii)
            # if self.integrator == 'Langevin':
            #     #half-step outside loop to ensure symplecticity
            #     self.velocities = self.velocities + self.dt/2*(forces/self.masses - self.gamma*self.velocities) + self.noise_f/torch.sqrt(torch.tensor(2.0).to(self.device))*torch.randn_like(self.velocities)
            zeta = self.zeta
            cell = self.cell.unsqueeze(0).repeat(self.n_replicas, 1, 1)
            #Run MD
            print("Start MD trajectory", file=self.f)
            for step in tqdm(range(self.nsteps)):
                self.step = step
                #MD Step
                if self.integrator == 'NoseHoover':
                    radii, velocities, forces, zeta = self.forward_nosehoover(self.radii, self.velocities, forces, zeta, retain_grad = False)
            
                elif self.integrator == 'Langevin':
                    radii, velocities, forces, noise = self.forward_langevin(self.radii, self.velocities, forces, retain_grad = False)
                elif self.integrator == 'Berendsen':
                    radii, velocities, forces, cell = self.forward_berendsen(self.radii, self.velocities, forces, cell, retain_grad = False)
                elif self.integrator == 'NPT':
                    radii, velocities, forces, cell = self.npt_integrator.step(retain_grad = False)
                    self.npt_integrator.log()
                    step  = self.step if self.train else (self.epoch+1) * self.step #don't overwrite previous epochs at inference time
                    self.t.append(self.create_frame(frame = step/self.n_dump, cell = cell[0]))
                    
                else:
                    raise RuntimeError("Must choose either NoseHoover, Langevin, or Berendsen as integrator")
                
                #save trajectory for gradient calculation
                if step >= self.eq_steps:# and step % self.n_dump == 0:
                    self.running_radii.append(radii.detach().clone())
                    self.running_vels.append(velocities.detach().clone())
                    self.running_accs.append((forces/self.masses).detach().clone())
                    if self.integrator == "Langevin":
                        self.running_noise.append(noise)
                
                    if step % self.n_dump == 0 and not self.train:
                        self.all_radii.append(radii.detach().cpu()) #save whole trajectory without resetting at inference time
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                    
            self.zeta = zeta
            self.forces = forces
            self.stacked_radii = torch.stack(self.running_radii[::self.n_dump])
            # compute instability metric (either bond length deviation, min intermolecular distance, or RDF MAE)
            self.instability_per_replica = self.stability_criterion(self.stacked_radii)
            if isinstance(self.instability_per_replica, tuple):
                self.instability_per_replica = self.instability_per_replica[-1]
            self.mean_instability = self.instability_per_replica.mean()
            if self.pbc:
                self.mean_bond_length_dev = self.bond_length_dev(self.stacked_radii)[1].mean()
                self.mean_rdf_mae = self.rdf_mae(self.stacked_radii)[-1].mean()
            self.stacked_vels = torch.cat(self.running_vels)
        
        if self.train:
            try:
                self.t.close()
            except:
                pass
        return self 

    def save_checkpoint(self, best=False, name_=None):
        if self.model_type == "nequip":
            if name_ is not None:
                name = f"{name_}.pth"
            else:
                name = "best_ckpt.pth" if best else "ckpt.pth"
            checkpoint_path = os.path.join(self.save_dir, name)
            with atomic_write(checkpoint_path, blocking=True, binary=True) as write_to:
                torch.save(self.model.state_dict(), write_to)
        else:
            if name_ is not None:
                name = f"{name_}.pt"
            else:
                name = "best_ckpt.pt" if best else "ckpt.pt"
            checkpoint_path = os.path.join(self.save_dir, name)
            new_state_dict = OrderedDict(("module."+k if "module" not in k else k, v) for k, v in self.model.state_dict().items())
            torch.save({
                        "epoch": self.epoch,
                        "step": self.epoch,
                        "state_dict": new_state_dict,
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.trainer.normalizers.items()
                        },
                        "config": self.model_config,
                        "ema": self.trainer.ema.state_dict() if self.trainer.ema else None,
                        "amp": self.trainer.scaler.state_dict()
                        if self.trainer.scaler
                        else None,
                    }, checkpoint_path)
        #also save in 'ckpt.pt'
        shutil.copyfile(checkpoint_path, os.path.join(self.save_dir, 'ckpt.pth' if self.model_type == 'nequip' else 'ckpt.pt'))
        self.curr_model_path = checkpoint_path

    def create_frame(self, frame, cell = None):
        # Particle positions, velocities, diameter
        radii = self.radii[0]
        if self.pbc:
            c = self.cell if cell is None else cell
            #wrap for visualization purposes
            if self.name == 'lips': #account for non-cubic cell
                frac_radii = cart2frac(radii, c)
                frac_radii = frac_radii % 1.0 #wrap
                radii = frac2cart(frac_radii, c)
            else:
                radii = ((radii / torch.diag(c)) % 1) * torch.diag(c)  - torch.diag(c)/2 #wrap coords (last subtraction is for cell alignment in Ovito)
        partpos = detach_numpy(radii).tolist()
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
        
        cell = torch.Tensor(c).cpu()
        s.configuration.box=[cell[0][0], cell[1][1], cell[2][2], 0, 0, 0]
        s.configuration.step = self.dt

        if self.name != 'lips': #don't show bonds if lips
            s.bonds.N = self.bonds.shape[0]
            s.bonds.group = detach_numpy(self.bonds)
        return s
    
    def thermo_log(self, pe, forces):
        #Log energies and instabilities
        p_dof = 3*self.n_atoms
        ke = 1/2 * (self.masses*torch.square(self.velocities)).sum(axis = (1,2)).unsqueeze(-1)
        temp = (2*ke/p_dof).mean() / units.kB

        # pressure calculation
        B, N, _ = self.radii.shape
        V = torch.linalg.det(self.cell.unsqueeze(0).repeat(self.n_replicas, 1, 1)).unsqueeze(-1).unsqueeze(-1)

        # Potential (Virial) Contribution
        outer_product = self.radii.unsqueeze(-1) * forces.unsqueeze(-2)
        stress_tensor_potential = outer_product.sum(dim=1) / V

        # Kinetic Contribution
        kinetic_contrib = (
            self.masses.unsqueeze(-1)
            * self.velocities.unsqueeze(-1)
            * self.velocities.unsqueeze(-2)
        ).sum(dim=1) / V

        # Total Stress Tensor
        stress_tensor = stress_tensor_potential + kinetic_contrib
        pressure = - vmap(torch.diag)(stress_tensor).mean(dim = 1)
    
        instability = self.stability_criterion(self.radii.unsqueeze(0))
        if isinstance(instability, tuple):
            instability = instability[-1]
        results_dict = {"Temperature": temp.item(),
                        "Pressure": pressure.mean().item(),
                        "Potential Energy": pe.mean().item(),
                        "Total Energy": (ke+pe).mean().item(),
                        "Momentum Magnitude": torch.norm(torch.sum(self.masses*self.velocities, axis =-2)).item(),
                        'Max Bond Length Deviation': self.bond_length_dev(self.radii.unsqueeze(0))[1].mean().item() \
                                                      if self.pbc else instability.mean().item()}
        if self.pbc:
            results_dict['Minimum Intermolecular Distance'] = instability.mean().item()
            results_dict['RDF MAE'] = self.rdf_mae(self.radii.unsqueeze(0))[-1].mean().item()
        return results_dict

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
    molecule_for_name = name if name =='water' or name == 'lips' else molecule
    size = config['size']
    model_type = config['model']

    #make directories
    results_dir = os.path.join(params.results_dir, f"IMPLICIT_{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}") \
                if params.train else os.path.join(params.results_dir, f"IMPLICIT_{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}", "inference", params.eval_model)
    os.makedirs(results_dir, exist_ok = True)
    
    print(f"Loading pretrained {model_type} model")
    lmax_string = f"lmax={params.l_max}_" if model_type == "nequip" else ""
    #load the correct checkpoint based on whether we're doing train or val
    load_cycle = None
    load_epoch = None
    
    print("Pretrained model:", config["eval_model"])
    if params.train or 'pre' in config["eval_model"]: #load energies/forces trained model
        pretrained_model_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{size}_{lmax_string}{model_type}") 
    
    elif 'k' in config["eval_model"] or 'percent' in config["eval_model"]:#load energies/forces model trained on a different dataset size
        new_size = config["eval_model"].split("k")[0] + "k"
        pretrained_model_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{new_size}_{lmax_string}{model_type}")

    elif 'lmax' in config["eval_model"]:#load energies/forces model trained with a different lmax
        assert model_type == 'nequip', f"l option is only valid with nequip models, you have selected {model_type}"
        new_lmax_string = config["eval_model"] + "_"
        pretrained_model_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{size}_{new_lmax_string}{model_type}")  

    elif 'post' in config["eval_model"]: #load observable finetuned model at some point in training
        pretrained_model_path = os.path.join(params.results_dir, f"IMPLICIT_{model_type}_{molecule_for_name}_{params.exp_name}_lr={params.lr}_efweight={params.energy_force_loss_weight}")
        if 'cycle' in config["eval_model"]: #load observable finetuned model at specified cycle
            load_cycle, load_epoch = extract_cycle_epoch(config["eval_model"])
    else:
        RuntimeError("Invalid eval model choice")
    
    if model_type == "nequip":
        if "post" in config['eval_model'] and not params.train:
            #get config from pretrained directory
            if load_cycle is not None:
                if load_epoch is not None:
                    cname = f"cycle{load_cycle}_epoch{load_epoch}.pth"
                else:
                    cname = f"end_of_cycle{load_cycle}.pth"
            else:
                cname = "ckpt.pth"
            print(f'Loading model weights from {os.path.join(pretrained_model_path, cname)}')
            pre_path = os.path.join(config['model_dir'], model_type, f"{name}-{molecule}_{size}_{lmax_string}{model_type}")
            _, model_config = Trainer.load_model_from_training_session(pre_path, \
                                    model_name = 'best_model.pth', device =  torch.device(device))
            #get model from finetuned directory
            model, _ = Trainer.load_model_from_training_session(pretrained_model_path, \
                                    config_dictionary=model_config, model_name = cname, device =  torch.device(device))
            #model = torch.load(os.path.join(pretrained_model_path, cname), map_location = torch.device(device))
        else:
            ckpt_epoch = config['checkpoint_epoch']
            cname = 'best_model.pth' if ckpt_epoch == -1 else f"ckpt{ckpt_epoch}.pth"
            print(f'Loading model weights from {os.path.join(pretrained_model_path, cname)}')
            model, model_config = Trainer.load_model_from_training_session(pretrained_model_path, \
                                    model_name = cname, device =  torch.device(device))
            shutil.copy(os.path.join(pretrained_model_path, cname), os.path.join(results_dir, 'best_model.pth'))
            shutil.copy(os.path.join(pretrained_model_path,'config.yaml'), os.path.join(results_dir, 'config.yaml'))
            shutil.copy(os.path.join(pretrained_model_path,'trainer.pth'), os.path.join(results_dir, 'trainer.pth'))
        model_path = os.path.join(pretrained_model_path, cname)
        model_config = {'model': model_config}
    else:
        model, model_path, model_config = load_pretrained_model(model_type, path = pretrained_model_path, \
                                                                ckpt_epoch = config['checkpoint_epoch'], cycle = load_cycle, post_epoch = load_epoch, \
                                                                device = torch.device(device), train = params.train or 'post' not in params.eval_model)
        #copy original model config to results directory
        if params.train:
            shutil.copy(os.path.join(pretrained_model_path, "checkpoints", 'config.yml'), os.path.join(results_dir, 'config.yml'))
    #count number of trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"{num_params} trainable parameters in {model_type} model")

    # #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)

    integrator_config = config['integrator_config']
    timestep = config["timestep"]
    ttime = integrator_config["ttime"]

    #load ground truth rdf and VACF
    print("Computing ground truth observables from datasets")
    if name == 'water':
        gt_rdf_package, gt_rdf_local_package, gt_diffusivity, gt_msd, gt_adf, oxygen_atoms_mask = find_water_rdfs_diffusivity_from_file(data_path, MAX_SIZES[name], params, device)
        gt_rdf, gt_rdf_var = gt_rdf_package
        gt_rdf_local, gt_rdf_var_local = gt_rdf_local_package
        # temp = torch.tensor([1]).to(device)
        # gt_rdf, gt_rdf_var, gt_rdf_local, gt_rdf_var_local, gt_diffusivity, gt_msd, gt_adf = temp, temp, temp, temp, temp, temp, temp
        # gt_rdf_package = (gt_rdf, gt_rdf_var)
        # gt_rdf_local_package = (gt_rdf_local, gt_rdf_var_local)
    elif name == 'lips':
        gt_rdf, gt_diffusivity = find_lips_rdfs_diffusivity_from_file(data_path, MAX_SIZES[name], params, device)
        gt_rdf_var = torch.ones_like(gt_rdf)
        gt_adf = torch.zeros((100,1)).to(device) #TODO: temporary
        
    else:
        gt_rdf, gt_adf = find_hr_adf_from_file(data_path, name, molecule, MAX_SIZES[name], params, device)
        gt_rdf_var = torch.ones_like(gt_rdf)
        gt_rdf_package = (gt_rdf, gt_rdf_var)
        gt_rdf_local_package = (gt_rdf, gt_rdf_var)
    contiguous_path = os.path.join(data_path, f'contiguous-{name}', molecule, MAX_SIZES[name], 'val/nequip_npz.npz')
    gt_data = np.load(contiguous_path)
    #TODO: gt vacf doesn't look right - it's because the recording frequency of the data is 10 fs, not 0.5 as in MD17
    if name == 'water':
        gt_vels = torch.FloatTensor(gt_data.f.velocities).to(device)
    elif name == 'lips':
        gt_traj = torch.FloatTensor(gt_data.f.pos).to(device)
        gt_vels = gt_traj[1:] - gt_traj[:-1] #finite difference approx
    else:
        gt_traj = torch.FloatTensor(gt_data.f.R).to(device)
        gt_vels = gt_traj[1:] - gt_traj[:-1] #finite difference approx


    gt_vacf = DifferentiableVACF(params, device)(gt_vels)
    if isinstance(gt_rdf, dict):
        for _type, _rdf in gt_rdf.items():
            np.save(os.path.join(results_dir, f'gt_{_type}_rdf.npy'), _rdf[0].cpu())
            np.save(os.path.join(results_dir, f'gt_{_type}_rdf_var.npy'), gt_rdf_var[_type].cpu())
            np.save(os.path.join(results_dir, f'gt_{_type}_rdf_local.npy'), gt_rdf_local[_type][0].cpu())
            np.save(os.path.join(results_dir, f'gt_{_type}_rdf_var_local.npy'), gt_rdf_var_local[_type].cpu())
    else:
        np.save(os.path.join(results_dir, 'gt_rdf.npy'), gt_rdf.cpu())
    np.save(os.path.join(results_dir, 'gt_adf.npy'), gt_adf.cpu())
    np.save(os.path.join(results_dir, 'gt_vacf.npy'), gt_vacf.cpu())
    if params.name == 'lips' or params.name == 'water':
        np.save(os.path.join(results_dir, 'gt_diffusivity.npy'), gt_diffusivity.cpu())
        np.save(os.path.join(results_dir, 'gt_msd.npy'), gt_msd.cpu())
    
    min_lr = params.lr / (5 ** params.max_times_reduce_lr) #LR reduction factor is 0.2 each time
    #outer training loop
    losses = []
    rdf_losses = []
    adf_losses = []
    diffusion_losses = []
    all_vacfs_per_replica = []
    vacf_losses = []
    mean_instabilities = []
    bond_length_devs = []
    rdf_maes = []
    energy_rmses = []
    force_rmses = []
    energy_maes = []
    force_maes = []
    grad_times = []
    sim_times = []
    grad_norms = []
    lrs = []
    resets = []
    writer = SummaryWriter(log_dir = results_dir)
    changed_lr = False
    cycle = 0
    learning_epochs_in_cycle = 0
    #function to add gradients
    add_lists = lambda list1, list2, w1, w2: tuple([w1*l1 + w2*l2 \
                                                    for l1, l2 in zip(list1, list2)])

    #Begin Main Training Loop
    for epoch in range(params.n_epochs):
        
        #rdf = torch.zeros_like(gt_rdf).to(device)
        rdf_loss = torch.Tensor([0]).to(device)
        vacf_loss = torch.Tensor([0]).to(device)
        diffusion_loss = torch.Tensor([0]).to(device)
        best = False
        grad_cosine_similarity = 0
        ratios = 0
        print(f"Epoch {epoch+1}")
        
        if epoch==0: #draw IC from dataset
            #initialize simulator parameterized by a NN model
            simulator = ImplicitMDSimulator(config, params, model, model_path, model_config, gt_rdf)
            #initialize Boltzmann_estimator
            boltzmann_estimator = BoltzmannEstimator(gt_rdf_package, gt_rdf_local_package, simulator.mean_bond_lens, gt_vacf, gt_adf, params, device)
            #initialize outer loop optimizer/scheduler
            if params.optimizer == 'Adam':
                simulator.optimizer = torch.optim.Adam(list(simulator.model.parameters()), \
                                            0 if params.only_learn_if_unstable_threshold_reached else params.lr)
            elif params.optimizer == 'SGD':
                simulator.optimizer = torch.optim.SGD(list(simulator.model.parameters()), \
                                            0 if params.only_learn_if_unstable_threshold_reached else params.lr)
            else:
                raise RuntimeError("Optimizer must be either Adam or SGD")
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(simulator.optimizer, mode='min', factor=0.2, patience=10)
            print(f"Initialize {simulator.n_replicas} random ICs in parallel")

        simulator.optimizer.zero_grad()
        simulator.epoch = epoch
        #set starting states based on instability metric
        num_unstable_replicas = simulator.set_starting_states()

        #start learning with focus on instability
        if simulator.all_unstable and not changed_lr: 
            simulator.optimizer.param_groups[0]['lr'] = params.lr
            #reset scheduler
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(simulator.optimizer, mode='min', factor=0.2, patience=10)
            changed_lr = True

        start = time.time()
        #run simulation 
        print('Collect MD Simulation Data')
        equilibriated_simulator = simulator.solve()
        #estimate gradients via Fabian method/adjoint
        rdf_package, vacf_package, energy_force_grad_batches = \
                    boltzmann_estimator.compute(equilibriated_simulator)
                
        #unpack results
        rdf_grad_batches, mean_rdf, rdf_loss, mean_adf, adf_loss = rdf_package
        vacf_grad_batches, vacfs_per_replica, vacf_loss = vacf_package
        if not params.train:
            all_vacfs_per_replica.append(vacfs_per_replica)

        outer_loss = params.rdf_loss_weight*rdf_loss + params.adf_loss_weight*adf_loss + diffusion_loss + params.vacf_loss_weight*vacf_loss
        print(f"Loss: RDF={rdf_loss.item()}+ADF={adf_loss.item()}+Diffusion={diffusion_loss.item()}+VACF={vacf_loss.item()}={outer_loss.item()}")
        
        #make lengths match for iteration
        if rdf_grad_batches is None:
            rdf_grad_batches = vacf_grad_batches
        if vacf_grad_batches is None:
            vacf_grad_batches = rdf_grad_batches
        if energy_force_grad_batches is None:
            energy_force_grad_batches = rdf_grad_batches
        
        #manual gradient updates for now
        if vacf_grad_batches or rdf_grad_batches:
            grad_cosine_similarity = []
            ratios = []
            for rdf_grads, vacf_grads, energy_force_grads in zip(rdf_grad_batches, vacf_grad_batches, energy_force_grad_batches): #loop through minibatches
                simulator.optimizer.zero_grad()
                
                #modify gradients according to loss weights
                obs_grads = add_lists(rdf_grads, vacf_grads, params.rdf_loss_weight, params.vacf_loss_weight)
                ef_grads = tuple([params.energy_force_loss_weight * ef_grad for ef_grad in energy_force_grads])
                cosine_similarity, ratio = compare_gradients(obs_grads, ef_grads)
                grad_cosine_similarity.append(cosine_similarity)#compute gradient similarities
                ratios.append(ratio)
                
                #Loop through each group of parameters and set gradients
                for param, obs_grad, ef_grad in zip(model.parameters(), obs_grads, ef_grads):
                    param.grad = obs_grad + ef_grad
                    
                if params.gradient_clipping: #gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_norm_threshold)
                
                simulator.optimizer.step()
            #If we are focusing on accuracy, step based on observable loss. If we are focusing on stability, step based on number of unstable replicas
            simulator.scheduler.step(num_unstable_replicas if simulator.all_unstable else outer_loss)
            grad_cosine_similarity = sum(grad_cosine_similarity) / len(grad_cosine_similarity)
            ratios = sum(ratios) / len(ratios)
        
        if simulator.all_unstable and params.train and (simulator.optimizer.param_groups[0]['lr'] < min_lr or num_unstable_replicas <= params.min_frac_unstable_threshold):
            print(f"Back to data collection")
            simulator.all_unstable = False
            simulator.first_simulation = True
            changed_lr = False
            cycle = cycle + 1
            learning_epochs_in_cycle = 0
            #save checkpoint at the end of learning cycle
            if params.train:
                simulator.save_checkpoint(name_ = f"end_of_cycle{cycle}")
            #reinitialize optimizer and scheduler with LR = 0
            if params.optimizer == 'Adam':
                simulator.optimizer = torch.optim.Adam(list(simulator.model.parameters()), \
                                            0 if params.only_learn_if_unstable_threshold_reached else params.lr)
            elif params.optimizer == 'SGD':
                simulator.optimizer = torch.optim.SGD(list(simulator.model.parameters()), \
                                            0 if params.only_learn_if_unstable_threshold_reached else params.lr)
            simulator.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(simulator.optimizer, mode='min', factor=0.2, patience=10)

        end = time.time()
        sim_time = end - start
        
        #checkpointing
        if params.train and simulator.mode =='learning':
            simulator.save_checkpoint(name_ = f"cycle{cycle+1}_epoch{learning_epochs_in_cycle+1}")
            learning_epochs_in_cycle +=1
        
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
        mean_instabilities.append(equilibriated_simulator.mean_instability)
        if simulator.pbc:
            bond_length_devs.append(equilibriated_simulator.mean_bond_length_dev)
            rdf_maes.append(equilibriated_simulator.mean_rdf_mae)
        resets.append(num_unstable_replicas)
        lrs.append(simulator.optimizer.param_groups[0]['lr'])
        #energy/force error
        if epoch == 0 or (simulator.optimizer.param_groups[0]['lr'] > 0 and params.train): #don't compute it unless we are in the learning phase
            energy_rmse, force_rmse, energy_mae, force_mae = simulator.energy_force_error()
            # energy_rmse = 0
            # force_rmse = 0
            # energy_mae = 0
            # force_mae = 0
            energy_rmses.append(energy_rmse)
            force_rmses.append(force_rmse)
            energy_maes.append(energy_mae)
            force_maes.append(force_mae)
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
        if params.stability_criterion == "imd":
            writer.add_scalar('Min Intermolecular Distances' ,mean_instabilities[-1], global_step=epoch+1)
        elif params.stability_criterion == "bond_length_deviation":
            writer.add_scalar('Max Bond Length Deviation' ,mean_instabilities[-1], global_step=epoch+1)
        
        if simulator.pbc:
            writer.add_scalar('Max Bond Length Deviation' , bond_length_devs[-1], global_step=epoch+1)
            writer.add_scalar('RDF MAE' , rdf_maes[-1], global_step=epoch+1)
            
        writer.add_scalar('Fraction of Unstable Replicas', resets[-1], global_step=epoch+1)
        writer.add_scalar('Learning Rate', lrs[-1], global_step=epoch+1)
        writer.add_scalar('Energy RMSE', energy_rmses[-1], global_step=epoch+1)
        writer.add_scalar('Force RMSE', force_rmses[-1], global_step=epoch+1)
        writer.add_scalar('Energy MAE', energy_maes[-1], global_step=epoch+1)
        writer.add_scalar('Force MAE', force_maes[-1], global_step=epoch+1)
        writer.add_scalar('Simulation Time', sim_times[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Norm', grad_norms[-1], global_step=epoch+1)
        writer.add_scalar('Gradient Cosine Similarity (Observable vs Energy-Force)', grad_cosine_similarity, global_step=epoch+1)
        writer.add_scalar('Gradient Ratios (Observable vs Energy-Force)', ratios, global_step=epoch+1)


        #add hyperparams and final metrics at inference time (do it every 5 epochs in case we time-out before the end)
        if not params.train:
            # save trajectory every epoch
            full_traj = torch.stack(simulator.all_radii)
            np.save(os.path.join(results_dir, 'full_traj.npy'), full_traj)
            
            if epoch % 5 == 0 and epoch > 175: #to save time
                if name == "md17" or name == "md22":
                    hparams_logging = calculate_final_metrics(simulator, params, device, results_dir, energy_maes, force_maes, gt_rdf, gt_adf, gt_vacf, all_vacfs_per_replica = all_vacfs_per_replica)
                elif name == "water":
                    hparams_logging = calculate_final_metrics(simulator, params, device, results_dir, energy_maes, force_maes, gt_rdf, gt_adf, gt_diffusivity = gt_diffusivity, oxygen_atoms_mask = oxygen_atoms_mask)
                for i in hparams_logging:
                    writer.file_writer.add_summary(i)
    
    #save metrics at end too
    if not params.train:
        if name == "md17" or name == "md22":
            hparams_logging = calculate_final_metrics(simulator, params, device, results_dir, energy_maes, force_maes, gt_rdf, gt_adf, gt_vacf, all_vacfs_per_replica = all_vacfs_per_replica)
        elif name == "water":
            hparams_logging = calculate_final_metrics(simulator, params, device, results_dir, energy_maes, force_maes, gt_rdf, gt_adf, gt_diffusivity = gt_diffusivity, oxygen_atoms_mask = oxygen_atoms_mask)
        for i in hparams_logging:
            writer.file_writer.add_summary(i)

    writer.close()      
    if not params.train:
        #close simulation file
        if hasattr(equilibriated_simulator, 't'):
            equilibriated_simulator.t.close()  
    print('Done!')
    


