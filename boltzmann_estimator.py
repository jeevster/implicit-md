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
from torchmd.observable import generate_vol_bins, DifferentiableRDF, DifferentiableADF, DifferentiableVelHist, DifferentiableVACF, SelfIntermediateScattering, msd, DiffusionCoefficient
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
from mdsim.observables.common import radii_to_dists, distance_pbc, ObservableMAELoss, ObservableMSELoss, IMDHingeLoss
from mdsim.observables.water import WaterRDFMAE, find_water_rdfs_diffusivity_from_file
from adjoints import get_adjoints, get_model_grads

#TODO: have separate functions for 

class BoltzmannEstimator():
    def __init__(self, gt_rdf, gt_rdf_var, gt_vacf, gt_adf, params, device):
        super(BoltzmannEstimator, self).__init__()
        self.params = params
        self.device = device
        self.diff_rdf = DifferentiableRDF(self.params, device)
        self.gt_rdf = gt_rdf
        self.gt_rdf_var = gt_rdf_var
        if isinstance(self.gt_rdf, dict):
            if params.training_observable == 'rdf':
                self.gt_rdf = torch.cat([rdf.flatten() for rdf in self.gt_rdf.values()]) #combine RDFs together
                self.gt_rdf_var = torch.cat([var.flatten() for var in self.gt_rdf_var.values()]) + 1e-5 #combine RDF variances together
            else:
                self.gt_rdf = torch.Tensor([1.44]).to(self.device) #ground truth min IMD
        
        self.gt_vacf = gt_vacf
        self.gt_adf = gt_adf
    
        #define losses - need to set up config options to use others if needed
        self.rdf_loss = IMDHingeLoss(self.gt_rdf) if params.training_observable == "imd" else ObservableMSELoss(self.gt_rdf)
        self.adf_loss = ObservableMSELoss(self.gt_adf)
        self.vacf_loss = ObservableMSELoss(self.gt_vacf)
    
    #define Onsager-Machlup Action ("energy" of each trajectory)
    #TODO: make this a torch.nn.Module in observable.py
    def om_action(self, vel_traj, radii_traj):
        #expects vel_traj and radii_traj to be of shape T X N X 3
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
        return estimator.detach()

    def compute(self, simulator):
        self.simulator = simulator
        if self.simulator.vacf_loss_weight !=0 and self.simulator.integrator != "Langevin":
            raise RuntimeError("Must use stochastic (Langevin) dynamics for VACF training")
        
        MINIBATCH_SIZE = self.simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
        diff_rdf = DifferentiableRDF(self.params, self.device)
        diff_adf = DifferentiableADF(self.simulator.n_atoms, self.simulator.bonds, self.simulator.cell, self.params, self.device)
        diff_vacf = DifferentiableVACF(self.params, self.device)
        #diff_sisf = SelfIntermediateScattering(self.params, self.device)
        
        model = simulator.model
        #find which replicas are unstable
        stable_replicas = self.simulator.stability_per_replica()
        #if focusing on accuracy, always only keep stable replicas for gradient calculation
        if not self.simulator.all_unstable:
            mask = stable_replicas
        #if focusing on stability, option to only keep unstable replicas for gradient calculation
        else:
            mask = ~stable_replicas if (self.params.only_train_on_unstable_replicas) \
                    else torch.ones((self.simulator.n_replicas), dtype=torch.bool).to(self.device)
        #store original shapes of model parameters
        original_numel = [param.data.numel() for param in model.parameters()]
        original_shapes = [param.data.shape for param in model.parameters()]
        
        #get continuous trajectories (permute to make replica dimension come first)
        radii_traj = torch.stack(self.simulator.running_radii)
        
        stacked_radii = radii_traj[::self.simulator.n_dump] #take i.i.d samples for RDF loss
        if self.simulator.mode == 'learning':
            #save replicas
            np.save(os.path.join(self.simulator.save_dir, f'stacked_radii_epoch{self.simulator.epoch}.npy'), stacked_radii.cpu())
            np.save(os.path.join(self.simulator.save_dir, f'stable_replicas_epoch{self.simulator.epoch}.npy'), stable_replicas.cpu())
        
        velocity_subsample_ratio = int(self.simulator.gt_data_spacing_fs / (self.simulator.dt / units.fs)) #make the vacf spacing the same as the underlying GT data
        velocities_traj = torch.stack(simulator.running_vels).permute(1,0,2,3)[:, ::velocity_subsample_ratio]
        vacfs_per_replica = vmap(diff_vacf)(velocities_traj)
        vacfs_per_replica[~stable_replicas] = torch.zeros(1, 100).to(self.device) #zero out the unstable replica vacfs
        #split into sub-trajectories of length = vacf_window
        velocities_traj = velocities_traj.reshape(velocities_traj.shape[0], -1, self.simulator.vacf_window, self.simulator.n_atoms, 3)
        velocities_traj = velocities_traj[:, ::self.simulator.n_dump_vacf] #sample i.i.d paths
        
        vacfs = vmap(vmap(diff_vacf))(velocities_traj)
        mean_vacf = vacfs.mean(dim = (0,1))
        mean_vacf_loss = self.vacf_loss(mean_vacf) 
        
        #energy/force loss
        if (self.simulator.energy_force_loss_weight != 0 and self.simulator.train and simulator.optimizer.param_groups[0]['lr'] > 0):
            energy_force_package = (self.simulator.energy_force_gradient(),)
        else:
            energy_force_package = None
        
        #VACF stuff
        if self.params.vacf_loss_weight == 0 or not self.simulator.train or simulator.optimizer.param_groups[0]['lr'] == 0:
            vacf_gradient_estimators = None
            vacf_package = (vacf_gradient_estimators, vacfs_per_replica, self.vacf_loss(mean_vacf).to(self.device))
        else:
            vacf_gradient_estimators = []
            if self.params.adjoint:
                with torch.enable_grad():
                    #reshape into independent paths
                    short_vel_traj = torch.stack(simulator.running_vels).permute(1,0,2,3)[stable_replicas].reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)[::self.simulator.n_dump_vacf]
                    short_pos_traj = torch.stack(simulator.running_radii).permute(1,0,2,3)[stable_replicas].reshape(-1, self.simulator.vacf_window, self.simulator.n_atoms, 3)[::self.simulator.n_dump_vacf]
                    short_vel_traj.requires_grad = True
                    short_pos_traj.requires_grad = True
                    short_vacfs = vmap(diff_vacf)(short_vel_traj)
                    mean_short_vacf = (short_vacfs).mean(dim = 0)
                    mean_vacf_loss = self.vacf_loss(mean_short_vacf)
                    grad_outputs, = torch.autograd.grad(mean_vacf_loss, short_vel_traj, allow_unused = True)
                    force_fn = lambda r: simulator.force_calc(r, retain_grad = True)[1]
                    print(f"Computing gradients of VACF loss via adjoint method on {short_pos_traj.shape[0]} paths of length {short_pos_traj.shape[1]}")
                    adjoints, _, R = get_adjoints(simulator, short_pos_traj, grad_outputs, force_fn)
                    grads = [get_model_grads(adj, r, simulator) for adj, r in tqdm(zip(adjoints, R))]
                    num_params = len(list(simulator.model.parameters()))
                    num_samples = len(grads)
                    grads_final = torch.stack([torch.cat([grads[i][j].flatten().detach() for j in range(num_params)]) for i in range(num_samples)]).sum(0)
                    gradient_estimator = tuple([g.reshape(shape) for g, shape in zip(grads_final.split(original_numel), original_shapes)])
                    vacf_gradient_estimators.append(gradient_estimator)
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
                batch_size = 1
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
                        grad = [process_gradient(model.parameters(), torch.autograd.grad(o, model.parameters(), create_graph = False, retain_graph = True, allow_unused = True), self.device) for o in om_act.flatten()]
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
            vacf_package = (vacf_gradient_estimators, vacfs_per_replica, mean_vacf_loss.to(self.device))
       
        ###RDF/ADF Stuff ###
        if self.simulator.pbc: #LiPS or Water
            #replace RDF with IMD (didn't change variable name yet)
            if self.simulator.training_observable == 'imd':
                rdfs = torch.cat([self.simulator.stability_criterion(s.unsqueeze(0)) for s in stacked_radii])
            elif self.simulator.training_observable == 'rdf':
                rdfs = torch.cat([self.simulator.rdf_mae(s.unsqueeze(0))[0] for s in stacked_radii])
            else:
                raise RuntimeError(f"Must choose rdf or imd as target observable, not{self.simulator.training_observable}")
            rdfs = rdfs.reshape(-1, simulator.n_replicas, self.gt_rdf.shape[-1])
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1])
        else:
            r2d = lambda r: radii_to_dists(r, self.simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(-1, self.simulator.n_atoms, self.simulator.n_atoms-1, 1)
            rdfs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(-1, self.simulator.n_replicas, self.gt_rdf.shape[-1]) #this way of calculating uses less memory
            
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1]) #this way of calculating uses less memory
        
        # if self.simulator.mode == 'learning':
        #     import pdb; pdb.set_trace()
        #     x = 0
        np.save(os.path.join(self.simulator.save_dir, f'rdfs_epoch{self.simulator.epoch}.npy'), rdfs.cpu())
        mean_rdf = rdfs.mean(dim=(0, 1))
        mean_adf = adfs.mean(dim=(0, 1))
        mean_rdf_loss = self.rdf_loss(mean_rdf)
        mean_adf_loss = self.adf_loss(mean_adf)
        
        if self.params.rdf_loss_weight ==0 or not self.simulator.train or simulator.optimizer.param_groups[0]['lr'] == 0:
            rdf_gradient_estimators = None
            rdf_package = (rdf_gradient_estimators, mean_rdf, self.rdf_loss(mean_rdf).to(self.device), mean_adf, self.adf_loss(mean_adf).to(self.device))              
        else:
            rdfs = rdfs[:, mask].reshape(-1, rdfs.shape[-1])
            adfs = adfs[:, mask].reshape(-1, adfs.shape[-1])
            stacked_radii = stacked_radii[:, mask]
            rdfs.requires_grad = True
            adfs.requires_grad = True
            
            #TODO: scale the estimator by Kb*T
            start = time.time()
            stacked_radii = stacked_radii.reshape(-1, self.simulator.n_atoms, 3)
            
            #shuffle the radii, rdfs, and losses
            if self.simulator.shuffle:   
                shuffle_idx = torch.randperm(stacked_radii.shape[0])
                stacked_radii = stacked_radii[shuffle_idx]
                rdfs = rdfs[shuffle_idx]
                adfs = adfs[shuffle_idx]
            
            bsize=MINIBATCH_SIZE
            num_blocks = math.ceil(stacked_radii.shape[0]/ bsize)
            start_time = time.time()
            rdf_gradient_estimators = []
            adf_gradient_estimators = []
            raw_grads = []
            pe_grads_flattened = []
            print(f"Computing RDF/ADF gradients from {stacked_radii.shape[0]} structures in minibatches of size {MINIBATCH_SIZE}")
            num_params = len(list(model.parameters()))
            #first compute gradients of potential energy (in batches of size n_replicas)
            for i in tqdm(range(num_blocks)):
                start = bsize*i
                end = bsize*(i+1)
                with torch.enable_grad():
                    radii_in = stacked_radii[start:end]
                    radii_in.requires_grad = True
                    energy, _ = self.simulator.force_calc(radii_in, retain_grad = True)
                def get_vjp(v):
                    return compute_grad(inputs = list(model.parameters()), output = energy, grad_outputs = v, allow_unused = True, create_graph = False)
                vectorized_vjp = vmap(get_vjp)
                I_N = torch.eye(energy.shape[0]).unsqueeze(-1).to(self.device)
                num_samples = energy.shape[0]
                if self.simulator.model_type == 'forcenet': #dealing with device mismatch error
                    grads_vectorized = [process_gradient(model.parameters(), \
                                        compute_grad(inputs = list(model.parameters()), \
                                        output = e, allow_unused = True, create_graph = False), \
                                        self.device) for e in energy]
                    grads_flattened = torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for j in range(num_params)]) for i in range(num_samples)])
                else:
                    grads_vectorized = vectorized_vjp(I_N)
                    #flatten the gradients for vectorization
                    grads_flattened= torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_params)]) for j in range(num_samples)])
                pe_grads_flattened.append(grads_flattened)

            pe_grads_flattened = torch.cat(pe_grads_flattened)
            #Now compute final gradient estimators in batches of size MINIBATCH_SIZE
            num_blocks = math.ceil(stacked_radii.shape[0]/ (MINIBATCH_SIZE))
            for i in tqdm(range(num_blocks)):
                start = MINIBATCH_SIZE*i
                end = MINIBATCH_SIZE*(i+1)
                rdf_batch = [rdfs[start:end]] if self.simulator.training_observ1e-able == 'imd' else \
                            rdfs[start:end].chunk(3 if self.simulator.name == 'water' else 1, dim=1) # 3 separate RDFs for water
                pe_grads_batch = pe_grads_flattened[start:end]
                adf_batch = adfs[start:end]
                #TODO: fix the case where we don't use the MSE gradient
                grad_outputs_rdf = [2/gt_rdf_var*(rdf.mean(0) - gt_rdf).unsqueeze(0) \
                                    if self.simulator.use_mse_gradient else None \
                                    for rdf, gt_rdf, gt_rdf_var in zip(rdf_batch, \
                                    self.gt_rdf.chunk(len(rdf_batch)), self.gt_rdf_var.chunk(len(rdf_batch)))]
                grad_outputs_adf = 2*(adf_batch.mean(0) - self.gt_adf).unsqueeze(0) \
                                    if self.simulator.use_mse_gradient else None
                                
                # with torch.enable_grad():
                #     rdfs_mean = rdfs[start:end].mean(0)
                #     adfs_mean = adfs[start:end].mean(0)
                #     rdf_loss = self.rdf_loss(rdfs_mean)
                #     adf_loss = self.adf_loss(adfs_mean)
                #     grad_outputs_rdf = [torch.autograd.grad(rdf_loss, rdfs_mean)[0].detach().unsqueeze(0)]
                #     grad_outputs_adf = [torch.autograd.grad(adf_loss, adfs_mean)[0].detach().unsqueeze(0)]
                
                final_vjp = [self.estimator(rdf, pe_grads_batch, grad_output_rdf) for rdf, grad_output_rdf in zip(rdf_batch, grad_outputs_rdf)]
                final_vjp = torch.stack(final_vjp).mean(0)
                if self.params.adf_loss_weight !=0:
                    gradient_estimator_adf = self.estimate(adf_batch, grads_flattened, grad_outputs_adf)
                    final_vjp+= self.params.adf_loss_weight*gradient_estimator_adf             

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

            rdf_package = (rdf_gradient_estimators, mean_rdf, mean_rdf_loss.to(self.device), mean_adf, mean_adf_loss.to(self.device))
        
        return rdf_package, vacf_package, energy_force_package