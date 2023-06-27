from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo import PPO, PPOTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.policy.torch_mixins import ValueNetworkMixin


import numpy as np
import torch
import torch.nn as nn
import itertools
import wandb
import logging
from tqdm import tqdm
from torch_scatter import scatter_add
import os
from nequip.ase.nequip_calculator import nequip_calculator

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.AtomicData import neighbor_list_and_relative_vec
from nequip.train.trainer import Trainer
from mdsim.models.schnet import SchNetPolicy
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.common.registry import registry


from mdsim.common.utils import get_atomic_types, dictdata_to_atoms, compute_bond_lengths, data_to_atoms, load_schnet_model, atoms_to_batch
import mdsim.common.pytorch_util as ptu
from ase.data import atomic_masses


class ForceModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, config, name,  **model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, config, name)
        nn.Module.__init__(self)


        self.pretrained_model_path = model_kwargs['model_path']
        self.schnet_pretrained_model_path = model_kwargs['schnet_model_path']
        self.model_type = model_kwargs['model_type']
        try:
            self.device = torch.cuda.current_device()
            logging.info("Training with GPU")    
        except: 
            self.device = "cpu"
            logging.info("No GPUs found, Training with CPU")
            
        self.action_dim = self.action_space.shape[0]
        self.n_atoms = int(self.action_dim/ 6)
        self.value_state_dim = int(6*self.n_atoms) # value_state contains only pos and velocities

        # load model
        logging.info(f"Loading pretrained {self.model_type} model")
        if self.model_type == "nequip":
            self.force_network, self.model_config = Trainer.load_model_from_training_session(self.pretrained_model_path, \
                            device =  torch.device(self.device))
        elif self.model_type == "schnet":
            self.force_network, self.model_config = load_schnet_model(path = self.schnet_pretrained_model_path, num_atoms = self.n_atoms, device = torch.device(self.device))

        
        #initialize variance network
        self.variance_mode = model_kwargs['variance_mode']
        assert self.variance_mode in ["fixed", "diag", "full"], "Variance mode must be fixed, diag, or full"
        
        #converts d-dimensional atomic reps from SchNet to 3-d diagonal force variances
        self.variance_network = ptu.build_mlp(input_size = self.model_config["hidden_channels"], \
                                output_size=3, n_layers = 2, size = 2*self.model_config["hidden_channels"]).to(self.device)
        
        
        #some simulation parameters (need to access them in forward pass)
        self.dt = model_kwargs['timestep']
        self.T = model_kwargs['temperature']
        self.ttime = model_kwargs['ttime']

        #initialize critic network
        self.value_network = ptu.build_mlp(input_size = self.value_state_dim, \
                                output_size=1, n_layers = 2, size = 64).to(self.device)

        #define functions accessed by RLLIB
        #make all networks trainable - they currently all have the same very low LR though
        self.parameters = lambda: itertools.chain(self.force_network.parameters(), self.variance_network.parameters(), self.value_network.parameters())
        self.train = lambda : self.force_network.train(); self.variance_network.train() ; self.value_network.train()
        self.eval = lambda: self.force_network.eval(); self.variance_network.eval(); self.value_network.eval()
        
        #Monitor models in Weights and Biases
        if model_kwargs["log_to_wandb"]:
            wandb.watch(self.force_network)
            wandb.watch(self.value_network)
            wandb.watch(self.variance_network)

            
    def last_state(self):
        return self._last_state

    def value_function(self):
        state = self.last_state().to(self.device)
        return self.value_network(state).squeeze(-1).cpu()
    
    def forward(self, obs_batch, dummy_state=None, seq_lens=None):
        atoms = obs_batch['obs'] # already in dict format 
        original_vel = atoms['velocities']
        original_zeta = atoms['zeta']
        batch_size = atoms['pos'].shape[0]

        if torch.count_nonzero(atoms['pos']).item() == 0: #dummy loss computation - short-cut forward computation
            #set last value state
            self._last_state = torch.zeros((batch_size, self.value_state_dim))
            dummy_forces_mean = torch.zeros((batch_size, self.action_dim))
            dummy_forces_var = 0.001*torch.ones_like(dummy_forces_mean)
            return torch.cat([dummy_forces_mean, dummy_forces_var], dim = -1), []

        
        #batching
        atoms = atoms_to_batch(dictdata_to_atoms(atoms), device = self.device)
        atoms['velocities'] = original_vel.to(self.device)
        atoms['zeta'] = original_zeta.to(self.device)
                         
        
        masses = torch.tensor(atomic_masses)[atoms['atomic_numbers'].to(torch.long)].unsqueeze(-1).to(self.device)
        
        #set target kinetic energy
        target_EKin = 0.5 * (3.0 * self.n_atoms) * self.T
        Q = 3.0 * self.n_atoms * self.T * (self.ttime * self.dt)**2
        
        #predict forces
        with torch.enable_grad():
            _, f_t, rep_t = self.force_network(atoms)
            
        accel = f_t / masses
        vel = atoms['velocities'].reshape(-1, 3)
        
        zeta = torch.repeat_interleave(atoms['zeta'], self.n_atoms, 0)

        # make full step in position
        a_minus_zetavel = accel - (zeta * vel)
        x = atoms['pos'] + vel * self.dt + a_minus_zetavel * (0.5 * self.dt ** 2)

        atoms['pos'] = x.to(torch.float32)

        #get kinetic energy
        kinetic_energy = scatter_add(0.5 * masses * vel**2, atoms.batch, dim=0).sum(dim=-1, keepdim=True)

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * a_minus_zetavel
        atoms['velocities'] = vel_half.to(torch.float32)

        #get next forces
        with torch.enable_grad():
            _, f_tp1, rep_tp1 = self.force_network(atoms)

        # make a full step in accelerations
        accel = f_tp1 /masses

        # make a half step in self.zeta
        atoms['zeta'] = zeta[::self.n_atoms] + 0.5 * self.dt * \
            (1/Q) * (kinetic_energy - target_EKin)

        #update kinetic energy 
        kinetic_energy = scatter_add(0.5 * masses * vel**2, atoms.batch, dim=0).sum(dim=-1, keepdim=True)

        # make another halfstep in self.zeta
        atoms['zeta'] += 0.5 * self.dt * \
            (1/Q) * (kinetic_energy - target_EKin)
        
        # make another half step in velocity
        vel = (vel_half + 0.5 * self.dt * accel) / (1 + 0.5 * self.dt * torch.repeat_interleave(atoms['zeta'], self.n_atoms, 0))
        atoms['velocities'] = vel

        forces_mean = torch.cat((f_t, f_tp1), dim=1).reshape(batch_size, self.action_dim) #flatten the forces

        #compute variance using learned SchNet embeddings
        if self.variance_mode == "diag":
            #learn an independent 3d variance for each atom
            final_var_rep = torch.cat([rep_t, rep_tp1], dim=0)
            forces_var = self.variance_network(final_var_rep).reshape(batch_size, -1)
        else:
            forces_var = 0.001*torch.ones_like(forces_mean)

        #set last state for value computation
        self._last_state = torch.cat([atoms['pos'].reshape(batch_size, -1), atoms['velocities'].reshape(batch_size, -1)], dim=-1).float()
        return torch.cat([forces_mean, forces_var], dim = -1).cpu(), []

        


    

 