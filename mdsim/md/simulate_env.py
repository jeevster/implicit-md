import gym
from gym.spaces import Box, Tuple, Dict
from pathlib import Path
import copy
import logging
import os
import yaml
import torch
import numpy as np
import random
from tqdm import tqdm
from torch_geometric.data import Data
from nequip.data import AtomicData, AtomicDataDict
from sklearn.datasets import make_spd_matrix

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from nequip.ase.nequip_calculator import nequip_calculator

from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory

import mdsim.md.integrator as md_integrator
from mdsim.common.registry import registry
from mdsim.common.utils import setup_imports, setup_logging, compute_bond_lengths, load_schnet_model, data_to_atoms, atoms_to_batch, atoms_to_state_dict
from mdsim.datasets import data_list_collater
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.models.gmm.gmm import GaussianMixture



class MDSimulationEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        # deploy trained Nequip model.
        logging.info("Initializing MD simulation environment")
        self.config = config
        self.model_dir = config['dataset']["model_dir"]
        self.save_name = config['dataset']["save_name"]

        self.max_rollout_length = config['rl']["rollout_fragment_length"]
        
        self.train = config['mode'] == 'train'
        # adjust units.
        self.config['simulate']["integrator_config"]["timestep"] *= units.fs
        if self.config['simulate']["integrator"] in ['NoseHoover', 'NoseHooverChain']:
            self.config['simulate']["integrator_config"]["temperature"] *= units.kB

        self.integrator = None
        self.simulator = None
        self.calculator = None

        #initialize device
        try:
            self.device = torch.cuda.current_device()
        except: 
            self.device = "cpu"
            

        #create model save path
        (Path(self.model_dir) / self.save_name).mkdir(parents=True, exist_ok=True)

        #initialize datasets
        self.train_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], 'train')})
        self.valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], 'val')})

        self.train_length = self.train_dataset.__len__()
        self.valid_length = self.valid_dataset.__len__()

        #initialize dataset iterators
        self.train_idx=0
        self.valid_idx=0

        #initialize mean bond length target based on first point in training data
        self.r_max = self.config["task"]["r_max_reward"]
        init_data = self.getitem() 
        self.n_atoms = init_data['pos'].shape[0]
        atoms = data_to_atoms(init_data)
        self.gail = config['task']['reward_mode'] == 'gail'
        #set up reward
        if self.gail:
            path = os.path.join(config['dataset']['schnet_model_dir'], 'discriminator_checkpoints', config['task']['discriminator_name'])
            self.discriminator, discriminator_config = load_schnet_model(path = path, num_atoms = self.n_atoms, device = self.device, \
                                                    mode = "discriminator", from_pretrained=True)

        elif config['task']['reward_mode'] == 'gmm':
            #rew_bond_indices are the bonds that we're using to compute the reward (fixed)
            _, self.rew_bond_indices = compute_bond_lengths(atoms, r_max = self.r_max)
            self.n_bonds = self.rew_bond_indices.shape[1]
            #fit a GMM to the ground truth data
            self.bond_length_gmm = GaussianMixture(n_components=1, n_features=self.rew_bond_indices.shape[1])
            logging.info("Making bond length dataset")
            self.bond_length_data = self.make_bond_length_dataset(n_samples=1000)
            logging.info(f"Fitting {self.rew_bond_indices.shape[1]}-dimensional Gaussian Mixture Model with 1 mixture component to bond length data")
            self.bond_length_gmm.fit(self.bond_length_data)
            #Look at Bayesian Information Criterion (BIC) to determine that 1-component GMM was sufficient
            # bics = []
            # for comp in range(20):
            #     self.bond_length_gmm = GaussianMixture(n_components=comp+1, n_features=self.rew_bond_indices.shape[1])
            #     self.bond_length_gmm.fit(self.bond_length_data)
            #     bics.append(self.bond_length_gmm.bic(self.bond_length_data))
        else:
            logging.error("Reward mode must either be 'gmm' or 'gail'")

        #reset dataset iterators
        self.train_idx=0
        self.valid_idx=0

        #initialize Gym spaces
        # self.observation_space = Dict({
        #                            "edge_index": Box(low = -1000, high=1000, shape = (2,self.n_bonds)), #edge indices
        #                             "pos": Box(low=-1000, high = 1000, shape = (self.n_atoms,3)), #atomic positions
        #                             "atomic_numbers": Box(low = -1000, high=1000, shape = (self.n_atoms,1)), #atomic numbers
        #                             "cell": Box(low = -1000, high = 1000, shape = (3,3)), #cell
        #                             "edge_cell_shift": Box(low = -1000, high = 1000, shape = (self.n_bonds,3)), #edge cell shift 
        #                             "pbc": Box(low = -1000, high=1000, shape = (3,)), #pbc
        #                             "velocities": Box(low=-1000, high = 1000, shape = (self.n_atoms,3)), #velocity
        #                             "zeta": Box(low=-1000, high = 1000, shape = (1,)) #zeta for integrator
        #                         })
        
        self.observation_space = Dict({
                                    "pos": Box(low=-1000, high = 1000, shape = (self.n_atoms,3)), #atomic positions
                                    "atomic_numbers": Box(low = -1000, high=1000, shape = (self.n_atoms,1)), #atomic numbers
                                    "cell": Box(low = -1000, high = 1000, shape = (3,3)), #cell
                                    "velocities": Box(low=-1000, high = 1000, shape = (self.n_atoms,3)), #velocity
                                    "zeta": Box(low=-1000, high = 1000, shape = (1,)) #zeta for integrator
                                })

        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, dtype=np.float32, shape = (2*self.n_atoms*3,))

    def make_bond_length_dataset(self, n_samples):
        dataset = torch.zeros((n_samples, self.rew_bond_indices.shape[1]))
        for i in tqdm(range(n_samples)):
            data = self.getitem() 
            atoms = data_to_atoms(data)
            dataset[i] = compute_bond_lengths(atoms, bonds = self.rew_bond_indices)[0]
        return dataset


    def reward(self, f_t):
        if self.gail:
            atom = atoms_to_batch([self.atoms], self.device)
            atom.force = torch.as_tensor(f_t).to(self.device)
            #call discriminator on state-action pair
            disc_out = self.discriminator(atom)
            reward = (torch.log(disc_out + 1e-8) - torch.log(1-disc_out + 1e-8)).item()

        else:
            #log likelihood under trained GMM model
            bond_lengths, _ = compute_bond_lengths(self.atoms, bonds = self.rew_bond_indices)
            reward = self.bond_length_gmm.score_samples(bond_lengths).item()

        #normalize by length of rollout
        return reward / self.max_rollout_length

    
    def step(self, forces):
        #logging.info("Taking a step in the environment")
        #unflatten forces and split into current and next
        forces = forces.reshape(-1, 3)
        f_t, f_tp1 = forces[0:self.n_atoms], forces[self.n_atoms:]
        self.episode_len += 1
        #run one MD step
        self.simulator.integrator.step_rl(f_t, f_tp1)
        
        #calculate reward
        reward = self.reward(f_t)

        #update state with new positions
        self.atoms = self.simulator.atoms
        self.state = atoms_to_state_dict(self.atoms, self.r_max, zeta = self.simulator.integrator.zeta) 

        #keep these constant for purposes of keeping the state dimension constant
        # self.state['edge_index'] = self.rew_bond_indices.numpy() 
        # self.state['edge_cell_shift'] = np.zeros((self.n_bonds, 3))
        self.state.pop('edge_index')
        self.state.pop('edge_cell_shift')
        self.state.pop('pbc')
           
        self.ekin = self.simulator.atoms.get_kinetic_energy()
        self.temp = self.ekin / (1.5 * units.kB * self.simulator.natoms)
        early_stop = False
        if self.temp < self.simulator.min_temp or self.temp > self.simulator.max_temp:
            logging.info(f'Temperature {self.temp:.2f} is out of range: \
                    [{self.simulator.min_temp:.2f}, {self.simulator.max_temp:.2f}]. \
                    Early stopping the simulation.')
            early_stop = True
        
        done = early_stop or self.episode_len >= self.max_rollout_length

         
        return self.state, reward, done, {}

    def reset(self):

        """Resets the episode and returns the initial observation of the new one.
        """
        
        #logging.info("Resetting environment")
        # Reset the episode len.
        self.episode_len = 0

        #Sample datapoint from dataset and set current state
        init_data = self.getitem()

        self.atoms = data_to_atoms(init_data)

        #Set up simulator and integrator
        simulated_time = 0
        self.integrator = getattr(md_integrator, self.config['simulate']["integrator"])(
            self.atoms, **self.config['simulate']["integrator_config"])
        self.simulator = Simulator(self.atoms, self.integrator, self.config['simulate']["T_init"], 
                            restart=False, #TODO: check this
                            start_time=simulated_time,
                            save_dir=Path(self.model_dir) / self.save_name, 
                            save_frequency=self.config['simulate']["save_freq"])

        self.state = atoms_to_state_dict(self.atoms, self.r_max, self.simulator.integrator.zeta)

        #keep these constant for purposes of keeping the state dimension constant
        self.state.pop('edge_index')
        self.state.pop('edge_cell_shift')
        self.state.pop('pbc')
        
        return self.state

    #wrapper around __getitem__ function of dataset
    def getitem(self):
        if self.train:
            data = self.train_dataset.__getitem__(self.train_idx % self.train_length)
            
            self.train_idx+=1
        else:
            data = self.valid_dataset.__getitem__(self.valid_idx % self.valid_length)
            self.valid_idx+=1

        return data
        
        
class Simulator:
    def __init__(self, 
                 atoms, 
                 integrator,
                 T_init,
                 start_time=0,
                 save_dir='./log',
                 restart=False,
                 save_frequency=100,
                 min_temp=0.1,
                 max_temp=100000):
        self.atoms = atoms
        self.integrator = integrator
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.natoms = self.atoms.get_number_of_atoms()
        

        # intialize system momentum 
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, T_init * units.kB)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.save_dir / 'atoms.traj', 'a', self.atoms)
        self.integrator.attach(self.traj.write, interval=save_frequency)
        
        # attach log file
        # self.integrator.attach(NeuralMDLogger(self.integrator, self.atoms, 
        #                                 self.save_dir / 'thermo.log', 
        #                                 start_time=start_time, mode='a'), 
        #                        interval=save_frequency)


    
class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 start_time=0,
                 verbose=True,
                 **kwargs):
        if start_time == 0:
            header = True
        else:
            header = False
        super().__init__(header=header, *args, **kwargs)
        """
        Logger uses ps units.
        """
        self.start_time = start_time
        self.verbose = verbose
        if verbose:
            print(self.hdr)
        self.natoms = self.atoms.get_number_of_atoms()

    def __call__(self):
        if self.start_time > 0 and self.dyn.get_time() == 0:
            return
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs) + self.start_time
            dat = (t,)
        else:
            dat = ()
        dat += (epot+ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)




