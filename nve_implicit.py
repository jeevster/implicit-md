import numpy as np
import gsd.hoomd
import torch
from pathlib import Path
import torch.nn as nn
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from YParams import YParams
from nequip.train.trainer import Trainer
import types

import argparse
import logging
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
                    dump_params_to_yml, powerlaw_inv_cdf, print_active_torch_tensors, plot_pair, solve_continuity_system, find_hr_from_file


#NNIP stuff:
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
        #   


class ImplicitMDSimulator(ImplicitMetaGradientModule, linear_solve=torchopt.linear_solve.solve_normal_cg(maxiter=5, atol=0)):
    def __init__(self, config, params, model, model_config):
        super(ImplicitMDSimulator, self).__init__()
        self.params = params
        #GPU
        try:
            self.device = torch.device(torch.cuda.current_device())
        except:
            self.device = "cpu"

        #Set random seeds
        np.random.seed(seed=config['optim']['seed'])
        torch.manual_seed(config['optim']['seed'])
        random.seed(config['optim']['seed'])

        self.name = config['dataset']['name']
        self.molecule = config['dataset']['molecule']
        self.size = config['dataset']['size']
        self.model_type = config['dataset']['model']

        logging.info("Initializing MD simulation environment")
        self.config = config
        self.model_dir = os.path.join(config['dataset']["model_dir"], self.model_type, f"{self.name}-{self.molecule}_{self.size}_{self.model_type}")
        self.save_name = config['dataset']["save_name"]

        
        self.train = config['mode'] == 'train'
        # adjust units.
        self.config['ift']["integrator_config"]["timestep"] *= units.fs
        if self.config['ift']["integrator"] in ['NoseHoover', 'NoseHooverChain']:
            self.config['ift']["integrator_config"]["temperature"] *= units.kB

        self.nsteps = np.rint(config['ift']["integrator_config"]["ttime"]/config['ift']["integrator_config"]["timestep"]).astype(np.int32)

        #create model save path
        (Path(self.model_dir) / self.save_name).mkdir(parents=True, exist_ok=True)

        #initialize datasets
        self.train_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], self.name, self.molecule, self.size, 'train')})
        self.valid_dataset = LmdbDataset({'src': os.path.join(config['dataset']['src'], self.name, self.molecule, self.size, 'val')})

        #get first configuration from dataset
        init_data = self.train_dataset.__getitem__(0)
        self.n_atoms = init_data['pos'].shape[0]
        self.atoms = data_to_atoms(init_data)

        #Initialize model (passed in as an argument to make it a meta parameter)
        self.model = model
        mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
                'r_start': 0.0,
                'r_end': params.cutoff, 
                'n_width': params.n_width,
                'n_layers': params.n_layers,
                'nonlinear': params.nonlinear}


        self.mlp_model = pairMLP(**mlp_params)
        self.model_config = model_config

        #Set up simulator and integrator
        # simulated_time = 0
        # self.integrator = getattr(md_integrator, self.config['ift']["integrator"])(
        #     self.atoms, self.model,device = self.device, **self.config['ift']["integrator_config"])
        # self.simulator = Simulator(self.atoms, self.integrator, self.config['ift']["T_init"], 
        #                     restart=False, #TODO: check this
        #                     start_time=simulated_time,
        #                     save_dir=Path(self.model_dir) / self.save_name, 
        #                     save_frequency=self.config['ift']["n_dump"])
        
        self.atomic_numbers = torch.Tensor(self.atoms.get_atomic_numbers()).to(torch.long).to(self.device)
        self.masses = torch.Tensor(self.atoms.get_masses().reshape(-1, 1)).to(self.device)

        self.n_replicas = config["ift"]["n_replicas"]
        self.vacf_window = config["ift"]["vacf_window"]

        #Register inner parameters
        self.radii = nn.Parameter(torch.Tensor(self.atoms.get_positions()).clone(), requires_grad=True).to(self.device)
        self.velocities = nn.Parameter(torch.Tensor(self.atoms.get_velocities()).clone(), requires_grad=True).to(self.device)
        self.rdf = nn.Parameter(torch.zeros((int(self.params.max_rdf_dist/self.params.dr),)), requires_grad=True).to(self.device)
        # self.diff_coeff = nn.Parameter(torch.zeros((self.n_replicas,)), requires_grad=True).to(self.device)
        # self.vacf = nn.Parameter(torch.zeros((self.n_replicas,self.vacf_window)), requires_grad=True).to(self.device)

        # self.n_particle = params.n_particle
        # self.temp = params.temp
        # self.kbt0 = params.kbt0
        # self.box = params.box
        # self.pbc = not params.no_pbc
        # self.dt = params.dt
        # self.t_total = params.t_total

        #Nose-Hoover Thermostat stuff
        self.dt = config["ift"]['integrator_config']["timestep"]
        self.temp = config["ift"]['integrator_config']["temperature"]
        self.targeEkin = 0.5 * (3.0 * self.n_atoms) * self.temp
        self.ttime = config["ift"]["integrator_config"]["ttime"]  # * units.fs
        self.Q = 3.0 * self.n_atoms * self.temp * (self.ttime * self.dt)**2
        self.zeta = torch.Tensor([0.0]).to(self.device)

        # self.diameter_viz = params.diameter_viz
        # self.epsilon = params.epsilon
        # self.rep_power = params.rep_power
        # self.attr_power = params.attr_power
        # self.poly = params.poly
        # self.poly_power = params.poly_power
        # self.min_sigma = params.min_sigma
        # self.sigma = params.sigma
        
        # self.dr = params.dr
        # self.n_nvt_steps = np.rint(self.nvt_time/self.dt).astype(np.int32)
        # self.nsteps = np.rint(self.t_total/self.dt).astype(np.int32)
        # self.burn_in_frac = params.burn_in_frac
        self.nn = params.nn
        self.save_intermediate_rdf = params.save_intermediate_rdf
        self.exp_name = params.exp_name

        # self.diffusion_window = params.diffusion_window
        # self.vacf_window = params.vacf_window

        # self.rdf_loss_weight = params.rdf_loss_weight
        # self.diffusion_loss_weight = params.diffusion_loss_weight
        # self.vacf_loss_weight = params.vacf_loss_weight
        
        #limit CPU usage
        torch.set_num_threads(10)

        #define vectorized differentiable rdf and velhist
        self.diff_rdf = DifferentiableRDF(params, self.device)
        # self.diff_rdf_cpu = vmap(DifferentiableRDF(params, "cpu"), -1)
        # self.diff_vel_hist = vmap(DifferentiableVelHist(params, self.device), 0)
        # self.diff_vel_hist_cpu = vmap(DifferentiableVelHist(params, "cpu"), 0)
        # #define vectorized differentiable vacf and diffusion coefficients
        # self.diff_vacf = vmap(DifferentiableVACF(params), 0)
        # #vectorize over dim 1 (the input will be of shape [diffusion_window , n_replicas])
        # self.diffusion_coefficient = vmap(DiffusionCoefficient(params, self.device) , 1)
        

        # if self.nn:
        #     add = "polylj_" if self.poly else ""
        #     results = 'results_polylj' if self.poly else 'results'
        #     self.save_dir = os.path.join(results, f"IMPLICIT_{add}_{self.exp_name}_n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_rep_power={self.rep_power}_attr_power={self.attr_power}_dt={self.dt}_ttotal={self.t_total}")
        # else: 
        #     add = "_polylj" if self.poly else ""
        #     self.save_dir = os.path.join('ground_truth' + add, f"n={self.n_particle}_box={self.box}_temp={self.temp}_eps={self.epsilon}_sigma={self.sigma}_rep_power={self.rep_power}_attr_power={self.attr_power}")

        # os.makedirs(self.save_dir, exist_ok = True)
        # dump_params_to_yml(self.params, self.save_dir)

        # #File dump stuff
        # self.f = open(f"{self.save_dir}/log.txt", "a+")
        # self.t = gsd.hoomd.open(name=f'{self.save_dir}/sim_temp{self.temp}.gsd', mode='wb') 
        # self.n_dump = params.n_dump # dump for configuration

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

    def force_calc(self, radii, retain_grad = False):
        if not radii.requires_grad:
                radii.requires_grad = True
        with torch.enable_grad():
            energy = self.model(pos = radii, z = self.atomic_numbers)
        forces = -compute_grad(inputs = radii, output = energy) if retain_grad else -compute_grad(inputs = radii, output = energy).detach()
        return energy, forces


    def forward_nvt(self, radii, velocities, forces, zeta, calc_rdf = False):
        # get current accelerations
        accel = forces / self.masses

        # make full step in position 
        radii = radii + velocities * self.dt + \
            (accel - zeta * velocities) * (0.5 * self.dt ** 2)


        # record current velocities
        KE_0 = torch.sum(torch.square(velocities)) / 2
        
        # make half a step in velocity
        velocities = velocities + 0.5 * self.dt * (accel - zeta * velocities)

        # make a full step in accelerations
        energy, forces = self.force_calc(radii.to(self.device), retain_grad=False)
        accel = forces / self.masses

        # make a half step in self.zeta
        zeta = zeta + 0.5 * self.dt * (1/self.Q) * (KE_0 - self.targeEkin)

        #get updated KE
        ke = torch.sum(torch.square(velocities))/ 2

        # make another halfstep in self.zeta
        zeta = zeta + 0.5 * self.dt * \
            (1/self.Q) * (ke - self.targeEkin)

        # make another half step in velocity
        velocities = (velocities + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * zeta)

        
        if calc_rdf:
            new_dists = radii_to_dists(radii, self.params)
            
        # new_velhist = self.diff_vel_hist(torch.linalg.norm(velocities, dim=-1).permute((1,0))) if calc_rdf else 0 #calculate velocity histogram from a single frame
        new_rdf = self.diff_rdf(tuple(new_dists.to(self.device))) if calc_rdf else 0 #calculate the RDF from a single frame
        # #new_rdf = 0

        # if calc_diffusion:
        #     self.last_h_radii.append(radii.unsqueeze(0))
        # if calc_vacf:
        #     self.last_h_velocities.append(velocities.unsqueeze(0))

        # # dump frames
        # if self.step%self.n_dump == 0:
        #     print(self.step, self.calc_properties(energy), file=self.f)
        #     self.t.append(self.create_frame(frame = self.step/self.n_dump))
        #     #append dists to running_dists for RDF calculation (remove diagonal entries)
        #     if not self.nn:
        #         new_dists = radii_to_dists(radii, self.params)
        #         if self.poly:
        #             #normalize distances by sigma pairs
        #             new_dists = new_dists / self.sigma_pairs
        #         self.running_dists.append(new_dists.cpu().detach())
        #         self.running_vels.append(torch.linalg.norm(velocities, dim = -1).cpu().detach())

        return radii, velocities, forces, new_rdf, zeta

    '''Stationary condition construction for calculating implicit gradient'''
    def optimality(self, enable_grad = True):
        import pdb; pdb.set_trace()
        #get current forces - treat as a constant (since it's coming from before the fixed point)
        forces = self.force_calc(self.radii.detach().cpu().cuda(), retain_grad=False)[1]
        
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
        #Initialize forces/potential of starting configuration
        with torch.no_grad():
            
            _, forces = self.force_calc(self.radii)
            zeta = self.zeta
            #Run MD
            print("Start MD trajectory")#, file=self.f)

            for step in tqdm(range(self.nsteps)):
                # self.step = step
                
                calc_rdf = step ==  self.nsteps -1 or self.save_intermediate_rdf or not self.nn
                # calc_diffusion = (step >= self.nsteps - self.diffusion_window) or self.save_intermediate_rdf or not self.nn #calculate diffusion coefficient if within window from the end
                # calc_vacf = (step >= self.nsteps - self.vacf_window) or self.save_intermediate_rdf or not self.nn #calculate VACF if within window from the end

                # with torch.enable_grad() if (calc_diffusion and self.diffusion_loss_weight!=0) or (calc_vacf and self.diffusion_loss_weight!=0) else nullcontext():
                #     if step < self.n_nvt_steps: #NVT step
                #         radii, velocities, forces, zeta, rdf, velhist = self.forward_nvt(self.radii, self.velocities, forces, zeta, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion, calc_vacf = calc_vacf)
                #     else: # NVE step
                #         if step == self.n_nvt_steps:
                #             print("NVT Equilibration Complete, switching to NVE", file=self.f)
                #         radii, velocities, forces, rdf, velhist = self.forward(self.radii, self.velocities, forces, calc_rdf = calc_rdf, calc_diffusion=calc_diffusion, calc_vacf = calc_vacf)
                
                #MD Step
                radii, velocities, forces, rdf, zeta = self.forward_nvt(self.radii, self.velocities, forces, zeta, calc_rdf = calc_rdf)
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.rdf.copy_(rdf)
                
                
                # if not self.nn and self.save_intermediate_rdf and step % self.n_dump == 0:
                #     filename = f"step{step+1}_rdf.npy"
                #     np.save(os.path.join(self.save_dir, filename), self.rdf.mean(dim = 0).cpu().detach().numpy())

                #     filename = f"step{step+1}_velhist.npy"
                #     np.save(os.path.join(self.save_dir, filename), velhist.mean(dim = 0).cpu().detach().numpy())

                #     if step > self.vacf_window:
                #         last_h_vels = torch.cat(self.last_h_velocities[-self.vacf_window:], dim = 0).permute((1,0,2,3))
                #         vacf = self.diff_vacf(last_h_vels)
                #         filename = f"step{step+1}_vacf.npy"
                #         np.save(os.path.join(self.save_dir, filename), vacf.mean(dim = 0).cpu().detach().numpy())
            
            
            
            # length = len(self.running_dists)
            # self.zeta = zeta
            # #compute diffusion coefficient
            # #if self.diffusion_loss_weight != 0 or not self.nn:
            # msd_data = msd(torch.cat(self.last_h_radii, dim=0), self.box)
            # diffusion_coeff = self.diffusion_coefficient(msd_data)
            # self.diff_coeff.copy_(diffusion_coeff)
            
            # filename ="gt_diff_coeff.npy" if not self.nn else f"diff_coeff_epoch{epoch+1}.npy"
            # np.save(os.path.join(self.save_dir, filename), diffusion_coeff.mean().cpu().detach().numpy())

            # #compute VACF
            # #if self.vacf_loss_weight != 0 or not self.nn:
            # last_h_vels = torch.cat(self.last_h_velocities, dim = 0).permute((1,0,2,3))
            # vacf = self.diff_vacf(last_h_vels)
            # self.vacf.copy_(vacf)
            # filename ="gt_vacf.npy" if not self.nn else f"vacf_epoch{epoch+1}.npy"
            # np.save(os.path.join(self.save_dir, filename), vacf.mean(dim=0).cpu().detach().numpy())

            # #compute ground truth rdf over entire trajectory (do it on CPU to avoid memory issues)
            # save_velhist = self.diff_vel_hist_cpu(torch.stack(self.running_vels[int(self.burn_in_frac*length):], dim = 1)) if not self.nn else velhist
            # save_rdf = self.diff_rdf_cpu(self.running_dists[int(self.burn_in_frac*length):]) if not self.nn else self.rdf
            # filename ="gt_rdf.npy" if not self.nn else f"rdf_epoch{epoch+1}.npy"
            # np.save(os.path.join(self.save_dir, filename), save_rdf.mean(dim=0).cpu().detach().numpy())

            # filename ="gt_velhist.npy" if not self.nn else f"velhist_epoch{epoch+1}.npy"
            # np.save(os.path.join(self.save_dir, filename), save_velhist.mean(dim=0).cpu().detach().numpy())

            # #plot true and current energy functions
            # if self.poly:
            #     energy_fn = lambda dists: self.poly_potential(dists, sigma_pairs = 1)
            # else:
            #     energy_fn = lambda dists: self.lj_potential(dists)
            # plot_pair(epoch, self.save_dir, self.model, self.device, end=self.cutoff, target_pot=energy_fn)
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
        
if __name__ == "__main__":
    setup_logging() 
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    params = types.SimpleNamespace(**config["ift"])
    
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
        model, model_config = load_schnet_model(path = pretrained_model_path, device = torch.device(device))
        
    
    # #initialize RDF calculator
    diff_rdf = DifferentiableRDF(params, device)#, sample_frac = params.rdf_sample_frac)


    # #initialize model
    # mlp_params = {'n_gauss': int(params.cutoff//params.gaussian_width), 
    #             'r_start': 0.0,
    #             'r_end': params.cutoff, 
    #             'n_width': params.n_width,
    #             'n_layers': params.n_layers,
    #             'nonlinear': params.nonlinear}


    # NN = pairMLP(**mlp_params)

    # #prior potential only contains repulsive term
    # prior = LJFamily(epsilon=params.prior_epsilon, sigma=params.prior_sigma, rep_pow=params.prior_rep_power, attr_pow=params.prior_attr_power)
    

    # model = Stack({'nn': NN, 'prior': prior}).to(device)
    # radii_0 = fcc_positions(params.n_particle, params.box, device).unsqueeze(0).repeat(params.n_replicas, 1, 1)
    # velocities_0  = initialize_velocities(params.n_particle, params.temp, n_replicas = params.n_replicas)
    # rdf_0  = diff_rdf(tuple(radii_to_dists(radii_0[0].unsqueeze(0).to(device), params))).unsqueeze(0).repeat(params.n_replicas, 1)

    # #load ground truth rdf and diffusion coefficient
    gt_rdf = torch.Tensor(find_hr_from_file(data_path, molecule, size, params)).to(device)
    # if params.nn:
    #     add = "_polylj" if params.poly else ""
    #     gt_dir = os.path.join('ground_truth' + add, f"n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}")
    #     gt_rdf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_rdf.npy"))).to(device)
    #     gt_diff_coeff = torch.Tensor(np.load(os.path.join(gt_dir, "gt_diff_coeff.npy"))).to(device)
    #     gt_vacf = torch.Tensor(np.load(os.path.join(gt_dir, "gt_vacf.npy"))).to(device)[0:params.vacf_window]
    #     add = "polylj_" if params.poly else ""
    #     results = 'results_polylj' if params.poly else 'results'
    #     results_dir = os.path.join(results, f"IMPLICIT_{add}_{params.exp_name}_n={params.n_particle}_box={params.box}_temp={params.temp}_eps={params.epsilon}_sigma={params.sigma}_rep_power={params.rep_power}_attr_power={params.attr_power}_dt={params.dt}_ttotal={params.t_total}")

    #initialize outer loop optimizer/scheduler
    optimizer = torch.optim.Adam(list(model.parameters()), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)

    if not params.nn:
        params.n_epochs = 1
        params.batch_size = 1

    #outer training loop
    losses = []
    rdf_losses = []
    diffusion_losses = []
    vacf_losses = []
    grad_times = []
    sim_times = []
    grad_norms = []
    best_outer_loss = 100
    # if params.nn:
    #     writer = SummaryWriter(log_dir = results_dir)
    
    for epoch in range(params.n_epochs):
        rdf_loss = torch.Tensor([0]).to(device)
        vacf_loss = torch.Tensor([0]).to(device)
        diffusion_loss = torch.Tensor([0]).to(device)
        best = False
        print(f"Epoch {epoch+1}")
        # restart_override = epoch==0 or (torch.rand(size=(1,)) < params.restart_probability).item()
        

        optimizer.zero_grad()

        #run MD simulation to get equilibriated radii
        for i in range(params.batch_size):
            #if continuing simulation, initialize with equilibriated NVT coupling constant 
            restart = True #i == 0 and restart_override
            #initialize simulator parameterized by a NN model
            if restart: #start from FCC lattice
                print("Initialize from FCC lattice")
                simulator = ImplicitMDSimulator(config, params, model, model_config)
            else: #continue from where we left off in the last epoch/batch
                simulator.reset(last_radii, last_velocities, last_rdf)
                
            if not restart:
                simulator.zeta = equilibriated_simulator.zeta
            start = time.time()
            equilibriated_simulator = simulator.solve()
            end = time.time()
            sim_time = end-start
            print("MD simulation time (s): ", sim_time)
            
            #compute loss at the end of the trajectory
            if params.nn:
                rdf_loss += (equilibriated_simulator.rdf - gt_rdf).pow(2).mean() / params.batch_size
                # diffusion_loss += (equilibriated_simulator.diff_coeff - gt_diff_coeff).pow(2).mean() / params.batch_size# if params.diffusion_loss_weight != 0 else torch.Tensor([0.]).to(device)
                # vacf_loss += (equilibriated_simulator.vacf - gt_vacf).pow(2).mean() / params.batch_size# if params.vacf_loss_weight != 0 else torch.Tensor([0.]).to(device)

            #memory cleanup
            last_radii, last_velocities, last_rdf = equilibriated_simulator.cleanup()
        #simulator.f.close()
        
        if params.nn:
            outer_loss = params.rdf_loss_weight*rdf_loss + \
                        params.diffusion_loss_weight*diffusion_loss + \
                        params.vacf_loss_weight*vacf_loss
            print(f"Loss: RDF={params.rdf_loss_weight*rdf_loss.item()}+Diffusion={params.diffusion_loss_weight*diffusion_loss.item()}+VACF={params.vacf_loss_weight*vacf_loss.item()}={outer_loss.item()}")
            #compute (implicit) gradient of outer loss wrt model parameters
            start = time.time()
            torch.autograd.backward(tensors = outer_loss, inputs = list(model.parameters()))
            end = time.time()
            grad_time = end-start
            print("gradient calculation time (s): ",  grad_time)

            #checkpointing
            if outer_loss < best_outer_loss:
                best_outer_loss = outer_loss
                best = True
            simulator.save_checkpoint(best = best)

            equilibriated_simulator.cleanup()

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

            optimizer.step()
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
    


