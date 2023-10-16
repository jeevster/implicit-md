import numpy as np
import torch
import torch.nn as nn
import math
from nff.utils.scatter import compute_grad
import os
from configs.md22.integrator_configs import INTEGRATOR_CONFIGS
from tqdm import tqdm
import random
from torch_geometric.nn import MessagePassing, radius_graph
from torchmd.observable import generate_vol_bins, DifferentiableRDF, DifferentiableADF, DifferentiableVelHist, DifferentiableVACF, msd, DiffusionCoefficient
import time
from functorch import vmap, vjp
import warnings
warnings.filterwarnings("ignore")
#NNIP stuff:
import ase
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from nequip.data import AtomicData, AtomicDataDict
from mdsim.common.utils import setup_imports, setup_logging, compute_bond_lengths, data_to_atoms, atoms_to_batch, atoms_to_state_dict, convert_atomic_numbers_to_types, process_gradient, compare_gradients, initialize_velocities, dump_params_to_yml
from mdsim.observables.md17_22 import radii_to_dists, distance_pbc
from mdsim.observables.water import WaterRDFMAE, find_water_rdfs_diffusivity_from_file

#TODO: have separate functions for 

class BoltzmannEstimator():
    def __init__(self, gt_rdf, gt_vacf, gt_adf, params, device):
        super(BoltzmannEstimator, self).__init__()
        self.params = params
        self.diff_rdf = DifferentiableRDF(self.params, device)
        self.gt_rdf = gt_rdf
        if isinstance(self.gt_rdf, dict):
            self.gt_rdf = torch.cat([rdf.flatten() for rdf in self.gt_rdf.values()])
        
        self.gt_vacf = gt_vacf
        self.gt_adf = gt_adf
    
    def rdf_loss(self, rdf):
        return (rdf - self.gt_rdf).pow(2).mean()
    def adf_loss(self, adf):
        return (adf - self.gt_adf).pow(2).mean()
    def vacf_loss(self, vacf):    
        return (vacf - self.gt_vacf).pow(2).mean()
    
    #define Onsager-Machlup Action ("energy" of each trajectory)
    #TODO: make this a torch.nn.Module in observable.py
    def om_action(self, vel_traj, radii_traj):
        v_tp1 = vel_traj[:, :, 1:]
        v_t = vel_traj[:, :, :-1]
        f_tp1 = self.simulator.force_calc(radii_traj[:, :, 1:].reshape(-1, self.simulator.n_atoms, 3), retain_grad = True)[1].reshape(v_t.shape)
        a_tp1 = f_tp1/self.simulator.masses.unsqueeze(1).unsqueeze(1)
        diff = (v_tp1 - v_t - a_tp1*self.simulator.dt + self.simulator.gamma*v_t*self.simulator.dt)
        #pre-divide by auxiliary temperature (noise_f**2)
        om_action = diff**2 / (self.simulator.noise_f**2).unsqueeze(1).unsqueeze(1) #this is exponentially distributed
        #sum over euclidean dimensions, atoms, and vacf window: TODO: this ruins the exponential property
        return (diff/self.simulator.noise_f.unsqueeze(1).unsqueeze(1)).detach(), om_action.sum((-3, -2, -1))
    
    #Function to compute the gradient estimator
    def estimator(self, g, df_dtheta, grad_outputs = None):
        estimator = \
            (df_dtheta.mean(0).unsqueeze(0)*g.mean(0).unsqueeze(-1) - \
                df_dtheta.unsqueeze(1) * g.unsqueeze(-1)).mean(dim=0)
        #compute VJP with grad output
        if grad_outputs is not None:
            estimator = torch.mm(grad_outputs, estimator)[0] 
        return estimator

    def compute(self, simulator):
        self.simulator = simulator

        if self.simulator.vacf_loss_weight !=0 and self.simulator.integrator != "Langevin":
            raise RuntimeError("Must use stochastic (Langevin) dynamics for VACF training")
        
        MINIBATCH_SIZE = self.simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
        diff_rdf = DifferentiableRDF(self.params, self.simulator.device)
        diff_adf = DifferentiableADF(self.simulator.n_atoms, self.simulator.bonds, self.simulator.cell, self.params, self.simulator.device)
        diff_vacf = DifferentiableVACF(self.params, self.simulator.device)
        
        running_radii = self.simulator.running_radii if self.simulator.optimizer.param_groups[0]['lr'] > 0 else self.simulator.running_radii[0:2]
        
        model = simulator.model
        #find which replicas are unstable
        stable_replicas = self.simulator.instability_per_replica <= self.simulator.stability_tol
        #if focusing on accuracy, always only keep stable replicas for gradient calculation
        if not self.simulator.all_unstable:
            mask = stable_replicas
        #if focusing on stability, option to only keep unstable replicas for gradient calculation
        else:
            mask = ~stable_replicas if (self.params.only_train_on_unstable_replicas) \
                    else torch.ones((self.simulator.n_replicas), dtype=torch.bool).to(self.simulator.device)
        #store original shapes of model parameters
        original_numel = [param.data.numel() for param in model.parameters()]
        original_shapes = [param.data.shape for param in model.parameters()]
        
        #get continuous trajectories (permute to make replica dimension come first)
        radii_traj = torch.stack(running_radii)
        
        stacked_radii = radii_traj[::self.simulator.n_dump] #take i.i.d samples for RDF loss
        velocities_traj = torch.stack(simulator.running_vels).permute(1,0,2,3)
        #split into sub-trajectories of length = vacf_window
        velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
        velocities_traj = velocities_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
        vacfs = vmap(vmap(diff_vacf))(velocities_traj)
        mean_vacf = vacfs[stable_replicas].mean(dim = (0,1)) #only compute loss on stable replicas
        mean_vacf_loss = self.vacf_loss(mean_vacf)   

        #energy/force loss
        if (self.simulator.energy_loss_weight != 0 or self.simulator.force_loss_weight!=0) and self.simulator.train and simulator.optimizer.param_groups[0]['lr'] > 0:
            energy_grads, force_grads = self.simulator.energy_force_gradient(batch_size = self.simulator.n_replicas)
            energy_force_package = ([energy_grads], [force_grads])
        else:
            energy_force_package = (None, None)
        
        #VACF stuff
        if self.params.vacf_loss_weight == 0 or not self.simulator.train or simulator.optimizer.param_groups[0]['lr'] == 0:
            vacf_gradient_estimators = None
            vacf_package = (vacf_gradient_estimators, mean_vacf, self.vacf_loss(mean_vacf).to(self.simulator.device))
        else:
            radii_traj = radii_traj.permute(1,0,2,3)
            accel_traj = torch.stack(simulator.running_accs).permute(1,0,2,3)
            noise_traj = torch.stack(simulator.running_noise).permute(1,0,2,3)
            #split into sub-trajectories of length = vacf_window
            radii_traj = radii_traj.reshape(radii_traj.shape[0], -1, self.simulator.vacf_window,self.simulator.n_atoms, 3)
            radii_traj = radii_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
            noise_traj = noise_traj.reshape(noise_traj.shape[0], -1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
            noise_traj = noise_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
            
            vacfs = vacfs[mask].reshape(-1, self.simulator.vacf_window)
            radii_traj = radii_traj[mask]
            velocities_traj = velocities_traj[mask]
            accel_traj = accel_traj[mask]
            noise_traj = noise_traj[mask]

            vacf_loss_tensor = vmap(self.vacf_loss)(vacfs).reshape(-1, 1, 1)
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
                    diff, om_act = self.om_action(velocities, radii)
                    #noises = noise_traj[start:end].reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
                    # with torch.enable_grad():
                    #     forces = self.simulator.force_calc(radii.reshape(-1, self.simulator.n_atoms, 3), retain_grad=True).reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
                    
                    #make sure the diffs match the stored noises along the trajectory
                    #assert(torch.allclose(diff, noise_traj[start:end, :, 1:], atol = 1e-3))
                    
                    #doesn't work because vmap detaches the force tensor before doing the vectorization
                    def get_grads(forces_, noises_):
                        return torch.autograd.grad(-1*forces_, list(model.parameters()), noises_, allow_unused = True)
                    
                    #loop over samples
                    #VJP between df/dtheta and noise - thought this method would be faster but it's the same
                    #verified that gradients match the OM action way
                    # for i in tqdm(range(forces.shape[0])):
                    #     grad = [self.simulator.process_gradient(g) for g in torch.autograd.grad(-1*forces[i], list(model.parameters()), 2*self.simulator.dt/self.simulator.masses* self.simulator.noise_f * noises[i], retain_graph = True, allow_unused = True)]
                    #     quick_grads.append(grad)
                    # I_N = torch.eye(om_act.flatten().shape[0]).to(self.simulator.device)
                    #the vmap leads to a seg fault for some reason
                    #grad = get_grads_vmaped(I_N)
                    #grad = [[self.simulator.process_gradient(g) for g in get_grads(v)] for v in I_N]
                    #this explicit loop is very slow though (10 seconds per iteration)
                    #OM-action method
                    grad = [process_gradient(model.parameters(), torch.autograd.grad(o, model.parameters(), create_graph = False, retain_graph = True, allow_unused = True), self.simulator.device) for o in om_act.flatten()]
                    
                    om_act = om_act.detach()
                    diffs.append(diff)
                    om_acts.append(om_act)
                    grads.append(grad)

            #recombine batches
            diff = torch.cat(diffs)
            om_act = torch.cat(om_acts)
            import pdb; pdb.set_trace()
            grads = sum(grads, [])
            #log OM stats
            np.save(os.path.join(self.simulator.save_dir, f'om_diffs_epoch{self.simulator.epoch}'), diff.flatten().cpu().numpy())
            np.save(os.path.join(self.simulator.save_dir, f'om_action_epoch{self.simulator.epoch}'), om_act.detach().flatten().cpu().numpy())
                
            #flatten out the grads
            num_params = len(list(model.parameters()))
            num_samples = len(grads)

            vacf_grads_flattened = torch.stack([torch.cat([grads[i][j].flatten().detach() for j in range(num_params)]) for i in range(num_samples)])
            if self.simulator.shuffle:   
                shuffle_idx = torch.randperm(vacf_grads_flattened.shape[0])
                vacf_grads_flattened = vacf_grads_flattened[shuffle_idx]
                vacf_loss_tensor = vacf_loss_tensor[shuffle_idx]
                vacfs = vacfs[shuffle_idx]
                
            #now calculate Fabian estimator
            #scale VACF minibatch size to have a similar number of gradient updates as RDF
            #vacf_minibatch_size = math.ceil(MINIBATCH_SIZE / self.simulator.vacf_window * self.simulator.n_dump)
            vacf_minibatch_size = MINIBATCH_SIZE
            num_blocks = math.ceil(vacf_grads_flattened.shape[0]/ vacf_minibatch_size)
            start_time = time.time()
            vacf_gradient_estimators = []
            raw_grads = []
            for i in range(num_blocks):
                start = vacf_minibatch_size*i
                end = vacf_minibatch_size*(i+1)
                vacf_grads_batch = vacf_grads_flattened[start:end]
                if self.simulator.use_mse_gradient:
                    #compute VJP with MSE gradient
                    vacf_batch = vacfs[start:end]
                    gradient_estimator = (vacf_grads_batch.mean(0).unsqueeze(0)*vacf_batch.mean(0).unsqueeze(-1) - vacf_grads_batch.unsqueeze(1) * vacf_batch.unsqueeze(-1)).mean(dim=0)
                    grad_outputs = 2*(vacf_batch.mean(0) - self.gt_vacf).unsqueeze(0) #MSE gradient
                    final_vjp = torch.mm(grad_outputs, gradient_estimator)[0]
                else:
                    #use loss directly
                    vacf_loss_batch = vacf_loss_tensor[start:end].squeeze(-1)
                    final_vjp = vacf_grads_batch.mean(0)*vacf_loss_batch.mean(0) \
                                        - (vacf_grads_batch*vacf_loss_batch).mean(dim=0)

                if not self.simulator.allow_off_policy_updates:
                    raw_grads.append(final_vjp)
                else:
                    #re-assemble flattened gradients into correct shape
                    gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(final_vjp.split(original_numel), original_shapes)])
                    vacf_gradient_estimators.append(gradient_estimator)
            if not self.simulator.allow_off_policy_updates:
                mean_grads = torch.stack(raw_grads).mean(dim=0)
                #re-assemble flattened gradients into correct shape
                gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(mean_grads.split(original_numel), original_shapes)])
                vacf_gradient_estimators.append(gradient_estimator)
            vacf_package = (vacf_gradient_estimators, mean_vacf, mean_vacf_loss.to(self.simulator.device))
       
        ###RDF/ADF Stuff ###
        if self.simulator.name == "water":
            rdfs = torch.cat([self.simulator.stability_criterion(s.unsqueeze(0))[0] for s in stacked_radii])
            rdfs = rdfs.reshape(-1, self.simulator.n_replicas, rdfs.shape[-1])
            adfs = torch.zeros_like(rdfs) #TODO: fix
        else:
            r2d = lambda r: radii_to_dists(r, self.simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(-1, self.simulator.n_atoms, self.simulator.n_atoms-1, 1)
            rdfs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(-1, self.simulator.n_replicas, self.gt_rdf.shape[-1]) #this way of calculating uses less memory
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1]) #this way of calculating uses less memory
        
        #compute mean quantities only on stable replicas
        mean_rdf = rdfs[:, stable_replicas].mean(dim=(0, 1))
        mean_adf = adfs[:, stable_replicas].mean(dim=(0, 1))
        mean_rdf_loss = self.rdf_loss(mean_rdf)
        mean_adf_loss = self.adf_loss(mean_adf)
        
        if self.params.rdf_loss_weight ==0 or not self.simulator.train or simulator.optimizer.param_groups[0]['lr'] == 0:
            rdf_gradient_estimators = None
            rdf_package = (rdf_gradient_estimators, mean_rdf, self.rdf_loss(mean_rdf).to(self.simulator.device), mean_adf, self.adf_loss(mean_adf).to(self.simulator.device))              
        else:
            rdfs = rdfs[:, mask].reshape(-1, rdfs.shape[-1])
            adfs = adfs[:, mask].reshape(-1, adfs.shape[-1])
            stacked_radii = stacked_radii[:, mask]
            
            rdf_loss_tensor = vmap(self.rdf_loss)(rdfs).unsqueeze(-1).unsqueeze(-1)
            adf_loss_tensor = vmap(self.adf_loss)(adfs).unsqueeze(-1).unsqueeze(-1)
        
            #TODO: scale the estimator by temperature
            start = time.time()
            stacked_radii = stacked_radii.reshape(-1, self.simulator.n_atoms, 3)
            #shuffle the radii, rdfs, and losses
            if self.simulator.shuffle:   
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
                start = MINIBATCH_SIZE*i
                end = MINIBATCH_SIZE*(i+1)
                with torch.enable_grad():
                    radii_in = stacked_radii[start:end]
                    radii_in.requires_grad = True
                    energy, _ = self.simulator.force_calc(radii_in, retain_grad = True)
                def get_vjp(v):
                    return compute_grad(inputs = list(model.parameters()), output = energy, grad_outputs = v, allow_unused = True, create_graph = False)
                vectorized_vjp = vmap(get_vjp)
                I_N = torch.eye(energy.shape[0]).unsqueeze(-1).to(self.simulator.device)
                grads_vectorized = vectorized_vjp(I_N)
                #flatten the gradients for vectorization
                num_samples = energy.shape[0]
                num_params = len(list(model.parameters()))
                grads_flattened= torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_params)]) for j in range(num_samples)])
                rdf_batch = rdfs[start:end].chunk(3 \
                            if self.simulator.name == 'water' else 1, dim=1) # 3 separate RDFs for water
                adf_batch = adfs[start:end]
                
                #TODO: fix the case where we don't use the MSE gradient
                grad_outputs_rdf = [2*(rdf.mean(0) - gt_rdf).unsqueeze(0) \
                                    if self.simulator.use_mse_gradient else None \
                                    for rdf, gt_rdf in zip(rdf_batch, \
                                    self.gt_rdf.chunk(len(rdf_batch)))]
                grad_outputs_adf = 2*(adf_batch.mean(0) - self.gt_adf).unsqueeze(0) \
                                    if self.simulator.use_mse_gradient else None
                final_vjp = [self.estimator(rdf, grads_flattened, grad_output_rdf) for rdf, grad_output_rdf in zip(rdf_batch, grad_outputs_rdf)]
                final_vjp = torch.stack(final_vjp).mean(0)
                if self.params.adf_loss_weight !=0:
                    gradient_estimator_adf = self.estimate(adf_batch, grads_flattened, grad_outputs_adf)
                    final_vjp+= self.params.adf_loss_weight*gradient_estimator_adf             
                # else:
                #     #use loss directly
                #     self.rdf_loss_batch = rdf_loss_tensor[start:end].squeeze(-1)
                #     self.adf_loss_batch = adf_loss_tensor[start:end].squeeze(-1)
                #     loss_batch = self.rdf_loss_batch + self.params.adf_loss_weight*self.adf_loss_batch
                #     final_vjp = grads_flattened.mean(0)*loss_batch.mean(0) \
                #                         - (grads_flattened*loss_batch).mean(dim=0)

                if not self.simulator.allow_off_policy_updates:
                    raw_grads.append(final_vjp)
                else:
                    #re-assemble flattened gradients into correct shape
                    final_vjp = tuple([g.reshape(shape) for g, shape in zip(final_vjp.split(original_numel), original_shapes)])
                    rdf_gradient_estimators.append(final_vjp)

            if not self.simulator.allow_off_policy_updates:
                mean_vjps = torch.stack(raw_grads).mean(dim=0)
                #re-assemble flattened gradients into correct shape
                mean_vjps = tuple([g.reshape(shape) for g, shape in zip(mean_vjps.split(original_numel), original_shapes)])
                rdf_gradient_estimators.append(mean_vjps)

            rdf_package = (rdf_gradient_estimators, mean_rdf, mean_rdf_loss.to(self.simulator.device), mean_adf, mean_adf_loss.to(self.simulator.device))
        
        return rdf_package, vacf_package, energy_force_package