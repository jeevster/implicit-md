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
from mdsim.observables.water import find_water_rdfs_diffusivity_from_file


class BoltzmannEstimator():
    def __init__(self, simulator, gt_rdf, gt_vacf, gt_adf, params, device):
        super(BoltzmannEstimator, self).__init__()
        self.simulator = simulator
        self.params = params
        self.diff_rdf = DifferentiableRDF(self.params, device)
        self.gt_rdf = gt_rdf
        self.gt_vacf = gt_vacf
        self.gt_adf = gt_adf
    
    def rdf_loss(self, rdf):
        return (rdf - self.gt_rdf).pow(2).mean()
    def adf_loss(self, adf):
        return (adf - self.gt_adf).pow(2).mean()
    def vacf_loss(self, vacf):    
        return (vacf - self.gt_vacf).pow(2).mean()
    
    def estimate(self):

        if self.simulator.vacf_loss_weight !=0 and self.simulator.integrator != "Langevin":
            raise RuntimeError("Must use stochastic (Langevin) dynamics for VACF training")
        
        MINIBATCH_SIZE = self.simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
        diff_rdf = DifferentiableRDF(self.params, self.simulator.device)
        diff_adf = DifferentiableADF(self.simulator.n_atoms, self.simulator.bonds, self.simulator.cell, self.params, self.simulator.device)
        diff_vacf = DifferentiableVACF(self.params, self.simulator.device)
        
        print('Collect MD Simulation Data')
        equilibriated_simulator = self.simulator.solve()
        running_radii = self.simulator.running_radii if self.simulator.all_unstable else self.simulator.running_radii[0:2]
        #ctx.save_for_backward(equilibriated_self.simulator)
        
        model = equilibriated_simulator.model
        #find which replicas are unstable
        stable_replicas = self.simulator.instability_per_replica <= self.simulator.stability_tol
        #store original shapes of model parameters
        original_numel = [param.data.numel() for param in model.parameters()]
        original_shapes = [param.data.shape for param in model.parameters()]
        
        #get continuous trajectories (permute to make replica dimension come first)
        radii_traj = torch.stack(running_radii)
        
        stacked_radii = radii_traj[::self.simulator.n_dump] #take i.i.d samples for RDF loss
        velocities_traj = torch.stack(equilibriated_simulator.running_vels).permute(1,0,2,3)
        #split into sub-trajectories of length = vacf_window
        velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
        velocities_traj = velocities_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
        vacfs = vmap(vmap(diff_vacf))(velocities_traj)
        mean_vacf = vacfs[stable_replicas].mean(dim = (0,1)) #only compute loss on stable replicas
        vacfs = vacfs.reshape(-1, self.simulator.vacf_window)
        mean_vacf_loss = self.vacf_loss(mean_vacf)   

        #energy/force loss
        if (self.simulator.energy_loss_weight != 0 or self.simulator.force_loss_weight!=0) and self.simulator.train and self.simulator.all_unstable:
            energy_grads, force_grads = self.simulator.energy_force_gradient(batch_size = self.simulator.n_replicas)
            energy_force_package = ([energy_grads], [force_grads])
        else:
            energy_force_package = (None, None)

        if self.simulator.vacf_loss_weight !=0 and self.simulator.train and self.simulator.all_unstable:
            radii_traj = radii_traj.permute(1,0,2,3)
            accel_traj = torch.stack(equilibriated_simulator.running_accs).permute(1,0,2,3)
            noise_traj = torch.stack(equilibriated_simulator.running_noise).permute(1,0,2,3)
            #split into sub-trajectories of length = vacf_window
            radii_traj = radii_traj.reshape(radii_traj.shape[0], -1, self.simulator.vacf_window,self.simulator.n_atoms, 3)
            radii_traj = radii_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
            noise_traj = noise_traj.reshape(noise_traj.shape[0], -1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
            noise_traj = noise_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
        else:
            del radii_traj
            del velocities_traj
            del self.simulator.running_radii
            del self.simulator.running_accs
            del self.simulator.running_noise
            del self.simulator.running_vels
            
        
        if self.params.vacf_loss_weight == 0 or not self.simulator.train or not self.simulator.all_unstable:
            vacf_gradient_estimators = None
            vacf_package = (vacf_gradient_estimators, mean_vacf, self.vacf_loss(mean_vacf).to(self.simulator.device))
        else:
            vacf_loss_tensor = vmap(vmap(self.vacf_loss))(vacfs).reshape(-1, 1, 1)
            #define force function - expects input of shape (batch, N, 3)
            def get_forces(radii):
                batch_size = radii.shape[0]
                batch = torch.arange(batch_size).repeat_interleave(self.simulator.n_atoms).to(self.simulator.device)
                atomic_numbers = torch.Tensor(self.simulator.atoms.get_atomic_numbers()).to(torch.long).to(self.simulator.device).repeat(batch_size)
                if self.simulator.model_type == "schnet":
                    energy = model(pos = radii.reshape(-1,3), z = atomic_numbers, batch = batch)
                    grad = compute_grad(inputs = radii, output = energy)
                    assert(not grad.is_leaf)
                    return -grad
                elif self.simulator.model_type == "nequip":
                    #recompute neighbor list
                    #assign radii
                    self.simulator.atoms_batch['pos'] = radii.reshape(-1, 3)
                    self.simulator.atoms_batch['batch'] = batch
                    self.simulator.atoms_batch['atom_types'] = self.simulator.final_atom_types
                    #recompute neighbor list
                    self.simulator.atoms_batch['edge_index'] = radius_graph(radii.reshape(-1, 3), r=self.simulator.model_config[self.simulator.r_max_key], batch=batch, max_num_neighbors=32)
                    self.simulator.atoms_batch['edge_cell_shift'] = torch.zeros((self.simulator.atoms_batch['edge_index'].shape[1], 3)).to(self.simulator.device)
                    atoms_updated = self.simulator.model(self.simulator.atoms_batch)
                    del self.simulator.atoms_batch['node_features']
                    del self.simulator.atoms_batch['node_attrs']
                    energy = atoms_updated[AtomicDataDict.TOTAL_ENERGY_KEY]
                    forces = atoms_updated[AtomicDataDict.FORCE_KEY].reshape(-1, self.simulator.n_atoms, 3)
                    assert(not forces.is_leaf)
                    return forces

            #define Onsager-Machlup Action ("energy" of each trajectory)
            #TODO: make this a torch.nn.Module in observable.py
            def om_action(vel_traj, radii_traj):
                v_tp1 = vel_traj[:, :, 1:]
                v_t = vel_traj[:, :, :-1]
                f_tp1 = get_forces(radii_traj[:, :, 1:].reshape(-1, self.simulator.n_atoms, 3)).reshape(v_t.shape)
                a_tp1 = f_tp1/self.simulator.masses.unsqueeze(1).unsqueeze(1)
                diff = (v_tp1 - v_t - a_tp1*self.simulator.dt + self.simulator.gamma*v_t*self.simulator.dt)
                #pre-divide by auxiliary temperature (noise_f**2)
                om_action = diff**2 / (self.simulator.noise_f**2).unsqueeze(1).unsqueeze(1) #this is exponentially distributed
                #sum over euclidean dimensions, atoms, and vacf window: TODO: this ruins the exponential property
                return (diff/self.simulator.noise_f.unsqueeze(1).unsqueeze(1)).detach(), om_action.sum((-3, -2, -1))
            
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
                    #noises = noise_traj[start:end].reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
                    # with torch.enable_grad():
                    #     forces = get_forces(radii.reshape(-1, self.simulator.n_atoms, 3)).reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
                    
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
                    grad = [process_gradient(torch.autograd.grad(o, model.parameters(), create_graph = False, retain_graph = True, allow_unused = True), self.simulator.device) for o in om_act.flatten()]
                    
                    om_act = om_act.detach()
                    diffs.append(diff)
                    om_acts.append(om_act)
                    grads.append(grad)

            #recombine batches
            diff = torch.cat(diffs)
            om_act = torch.cat(om_acts)
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
            print(f"Computing VACF gradients from {vacf_grads_flattened.shape[0]} trajectories in minibatches of size {vacf_minibatch_size}")
            for i in tqdm(range(num_blocks)):
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
        ###RDF/ADF Stuff
        
        r2d = lambda r: radii_to_dists(r, self.simulator.params)
        dists = vmap(r2d)(stacked_radii).reshape(-1, self.simulator.n_atoms, self.simulator.n_atoms-1, 1)
        rdfs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(-1, self.simulator.n_replicas, self.gt_rdf.shape[-1]) #this way of calculating uses less memory
        adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1]) #this way of calculating uses less memory
        
        #compute mean quantities only on stable replicas
        mean_rdf = rdfs[:, stable_replicas].mean(dim=(0, 1))
        mean_adf = adfs[:, stable_replicas].mean(dim=(0, 1))
        mean_rdf_loss = self.rdf_loss(mean_rdf)
        mean_adf_loss = self.adf_loss(mean_adf)
        
        if self.params.rdf_loss_weight ==0 or not self.simulator.train or not self.simulator.all_unstable:
            rdf_gradient_estimators = None
            rdf_package = (rdf_gradient_estimators, mean_rdf, self.rdf_loss(mean_rdf).to(self.simulator.device), mean_adf, self.adf_loss(mean_adf).to(self.simulator.device))              
        else:
            #only keep the unstable replicas
            mask = ~stable_replicas if self.params.only_train_on_unstable_replicas \
                    else torch.ones((self.simulator.n_replicas), dtype=torch.bool).to(self.simulator.device)
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
                if self.simulator.model_type == "nequip":
                    temp_atoms_batch = self.simulator.atoms_batch
                start = MINIBATCH_SIZE*i
                end = MINIBATCH_SIZE*(i+1)
                actual_batch_size = min(end, stacked_radii.shape[0]) - start
                batch = torch.arange(actual_batch_size).repeat_interleave(self.simulator.n_atoms).to(self.simulator.device)
                atomic_numbers = torch.Tensor(self.simulator.atoms.get_atomic_numbers()).to(torch.long).to(self.simulator.device).repeat(actual_batch_size)
                with torch.enable_grad():
                    radii_in = stacked_radii[start:end]
                    radii_in.requires_grad = True
                    if self.simulator.model_type == "schnet":
                        energy = model(pos = radii_in.reshape(-1, 3), z = atomic_numbers, batch = batch)
                    elif self.simulator.model_type == "nequip":
                        #construct a batch
                        temp_atoms_batch['pos'] = radii_in.reshape(-1, 3)
                        temp_atoms_batch['batch'] = batch
                        temp_atoms_batch['edge_index'] = radius_graph(radii_in.reshape(-1, 3), r=self.simulator.model_config[self.simulator.r_max_key], batch=batch, max_num_neighbors=32)
                        temp_atoms_batch['edge_cell_shift'] = torch.zeros((temp_atoms_batch['edge_index'].shape[1], 3)).to(self.simulator.device)
                        #adjust shapes
                        self.simulator.atoms_batch['cell'] = self.simulator.atoms_batch['cell'][0].unsqueeze(0).repeat(radii_in.shape[0], 1, 1)
                        self.simulator.atoms_batch['pbc'] = self.simulator.atoms_batch['pbc'][0].unsqueeze(0).repeat(radii_in.shape[0], 1)
                        temp_atoms_batch['atom_types'] = self.simulator.final_atom_types[0:self.simulator.n_atoms].repeat(radii_in.shape[0], 1)
                        energy = model(temp_atoms_batch)[AtomicDataDict.TOTAL_ENERGY_KEY]

                def get_vjp(v):
                    return compute_grad(inputs = list(model.parameters()), output = energy, grad_outputs = v, create_graph = False)
                vectorized_vjp = vmap(get_vjp)
                I_N = torch.eye(energy.shape[0]).unsqueeze(-1).to(self.simulator.device)
                grads_vectorized = vectorized_vjp(I_N)
                #flatten the gradients for vectorization
                num_samples = energy.shape[0]
                num_params = len(list(model.parameters()))
                grads_flattened= torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_self.params)]) for j in range(num_samples)])
                
                if self.simulator.use_mse_gradient:
                    
                    rdf_batch = rdfs[start:end]
                    adf_batch = adfs[start:end]
                    gradient_estimator_rdf = (grads_flattened.mean(0).unsqueeze(0)*rdf_batch.mean(0).unsqueeze(-1) - grads_flattened.unsqueeze(1) * rdf_batch.unsqueeze(-1)).mean(dim=0)
                    if self.params.adf_loss_weight !=0:
                        gradient_estimator_adf = (grads_flattened.mean(0).unsqueeze(0)*adf_batch.mean(0).unsqueeze(-1) - grads_flattened.unsqueeze(1) * adf_batch.unsqueeze(-1)).mean(dim=0)
                        grad_outputs_adf = 2*(adf_batch.mean(0) - self.gt_adf).unsqueeze(0)
                    #MSE gradient
                    grad_outputs_rdf = 2*(rdf_batch.mean(0) - self.gt_rdf).unsqueeze(0)
                    #compute VJP with MSE gradient
                    final_vjp = torch.mm(grad_outputs_rdf, gradient_estimator_rdf)[0]
                    if self.params.adf_loss_weight !=0:
                        final_vjp+= self.params.adf_loss_weight*torch.mm(grad_outputs_adf, gradient_estimator_adf)[0]
                                    
                else:
                    #use loss directly
                    self.rdf_loss_batch = rdf_loss_tensor[start:end].squeeze(-1)
                    self.adf_loss_batch = adf_loss_tensor[start:end].squeeze(-1)
                    loss_batch = self.rdf_loss_batch + self.params.adf_loss_weight*self.adf_loss_batch
                    final_vjp = grads_flattened.mean(0)*loss_batch.mean(0) \
                                        - (grads_flattened*loss_batch).mean(dim=0)

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

            rdf_package = (rdf_gradient_estimators, mean_rdf, mean_self.rdf_loss.to(self.simulator.device), mean_adf, mean_self.adf_loss.to(self.simulator.device))
        
        return equilibriated_simulator, rdf_package, vacf_package, energy_force_package