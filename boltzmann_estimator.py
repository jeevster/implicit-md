import numpy as np
import torch
import math
from nff.utils.scatter import compute_grad
import os
from tqdm import tqdm
from torchmd.observable import DifferentiableRDF, DifferentiableADF, DifferentiableVACF
from functorch import vmap, vjp
import warnings
from ase import units
from mdsim.common.utils import process_gradient
from mdsim.observables.common import radii_to_dists, ObservableMAELoss, ObservableMSELoss, IMDHingeLoss
from mdsim.observables.water import n_closest_molecules


class BoltzmannEstimator():
    def __init__(self, gt_rdf_package, gt_rdf_local_package, mean_bond_lens, gt_vacf, gt_adf, params, device):
        super(BoltzmannEstimator, self).__init__()
        self.params = params
        self.device = device
        self.diff_rdf = DifferentiableRDF(self.params, device)
        self.gt_rdf, self.gt_rdf_var = gt_rdf_package
        self.gt_rdf_local, self.gt_rdf_var_local = gt_rdf_local_package
        
        if isinstance(self.gt_rdf, dict):
            if params.training_observable == 'rdf':
                self.gt_rdf = torch.cat([rdf.flatten() for rdf in self.gt_rdf.values()]) #combine RDFs together
                self.gt_rdf_local = torch.cat([rdf.flatten() for rdf in self.gt_rdf_local.values()]) #combine RDFs together
                self.gt_rdf_var = torch.cat([var.flatten() for var in self.gt_rdf_var.values()]) #combine RDF variances together
                self.gt_rdf_var = torch.where(self.gt_rdf_var == 0, 1e-2, self.gt_rdf_var) #add "hyperprior" to gt variance
                self.gt_rdf_var_local = torch.cat([var.flatten() for var in self.gt_rdf_var_local.values()]) #combine RDF variances together
                self.gt_rdf_var_local = torch.where(self.gt_rdf_var_local == 0, 1e-2, self.gt_rdf_var_local) #add "hyperprior" to gt variance
            else:
                if params.training_observable == 'imd':
                    self.gt_rdf = torch.Tensor([1.44]).to(self.device) #ground truth min IMD
                    self.gt_rdf_var_local = torch.Tensor([1.0]).to(self.device) #identity
                elif params.training_observable == 'bond_length_dev':
                    self.gt_rdf = mean_bond_lens[:2].to(self.device) #ground truth bond length dev
                    self.gt_rdf_var_local = torch.Tensor([1.0]).to(self.device) #identity for now
                self.gt_rdf_local = self.gt_rdf
                
        
        self.gt_vacf = gt_vacf
        self.gt_adf = gt_adf
    
        #define losses - need to set up config options to use others if needed
        self.rdf_loss = IMDHingeLoss(self.gt_rdf) if params.training_observable == "imd" else ObservableMSELoss(self.gt_rdf)
        self.adf_loss = ObservableMSELoss(self.gt_adf)
        self.vacf_loss = ObservableMSELoss(self.gt_vacf)
    
    #Function to compute the gradient estimator
    def estimator(self, g, df_dtheta, grad_outputs = None):
        estimator = \
            (df_dtheta.mean(0).unsqueeze(0)*g.mean(0).unsqueeze(-1) - \
                df_dtheta.unsqueeze(1) * g.unsqueeze(-1)).mean(dim=0)
        #compute VJP with grad output
        if grad_outputs is not None:
            estimator = torch.mm(grad_outputs.to(torch.float32), estimator.to(torch.float32))[0] 
        return estimator.detach()

    def compute(self, simulator):
        self.simulator = simulator
        if self.simulator.vacf_loss_weight !=0 and self.simulator.integrator != "Langevin":
            raise RuntimeError("Must use stochastic (Langevin) dynamics for VACF training")
        
        MINIBATCH_SIZE = self.simulator.minibatch_size #how many structures to include at a time (match rare events sampling paper for now)
        diff_rdf = DifferentiableRDF(self.params, self.device)
        diff_adf = DifferentiableADF(self.simulator.n_atoms, self.simulator.bonds, self.simulator.cell, self.params, self.device)
        diff_vacf = DifferentiableVACF(self.params, self.device)
        
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
        
        velocity_subsample_ratio = math.ceil(self.simulator.gt_data_spacing_fs / (self.simulator.dt / units.fs)) #make the vacf spacing the same as the underlying GT data
        velocities_traj = torch.stack(simulator.running_vels).permute(1,0,2,3)[:, ::velocity_subsample_ratio]
        vacfs_per_replica = vmap(diff_vacf)(velocities_traj)
        vacfs_per_replica[~stable_replicas] = torch.zeros(1, 100).to(self.device) #zero out the unstable replica vacfs
        #split into sub-trajectories of length = vacf_window
        velocities_traj = velocities_traj[:, :math.floor(velocities_traj.shape[1]/self.simulator.vacf_window) * self.simulator.vacf_window]
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
        vacf_gradient_estimators = None
        vacf_package = (vacf_gradient_estimators, vacfs_per_replica, self.vacf_loss(mean_vacf).to(self.device))
        
        ###RDF/ADF Stuff ###
        if self.simulator.pbc: #LiPS or Water
            #replace RDF with IMD (didn't change variable name yet)
            if self.simulator.training_observable == 'imd':
                rdfs = torch.cat([self.simulator.min_imd(s.unsqueeze(0)) for s in stacked_radii])
            elif self.simulator.training_observable == 'rdf':
                rdfs = torch.cat([self.simulator.rdf_mae(s.unsqueeze(0))[0] for s in stacked_radii])
            elif self.simulator.training_observable == 'bond_length_dev':
                rdfs = torch.cat([self.simulator.bond_length_dev(s.unsqueeze(0))[0] for s in stacked_radii])
                
            rdfs = rdfs.reshape(-1, simulator.n_replicas, self.gt_rdf_local.shape[-1])
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1])
        else:
            r2d = lambda r: radii_to_dists(r, self.simulator.params)
            dists = vmap(r2d)(stacked_radii).reshape(-1, self.simulator.n_atoms, self.simulator.n_atoms-1, 1)
            rdfs = torch.stack([diff_rdf(tuple(dist)) for dist in dists]).reshape(-1, self.simulator.n_replicas, self.gt_rdf.shape[-1]) #this way of calculating uses less memory
            
            adfs = torch.stack([diff_adf(rad) for rad in stacked_radii.reshape(-1, self.simulator.n_atoms, 3)]).reshape(-1, self.simulator.n_replicas, self.gt_adf.shape[-1]) #this way of calculating uses less memory
        
        np.save(os.path.join(self.simulator.save_dir, f'rdfs_epoch{self.simulator.epoch}.npy'), rdfs.cpu())
        mean_rdf = rdfs.mean(dim=(0, 1))
        mean_adf = adfs.mean(dim=(0, 1))
        mean_rdf_loss = self.rdf_loss(mean_rdf)
        mean_adf_loss = self.adf_loss(mean_adf)
        
        if self.params.rdf_loss_weight==0 or not self.simulator.train or simulator.optimizer.param_groups[0]['lr'] == 0:
            rdf_gradient_estimators = None
            rdf_package = (rdf_gradient_estimators, mean_rdf, self.rdf_loss(mean_rdf).to(self.device), mean_adf, self.adf_loss(mean_adf).to(self.device))              
            return (rdf_package, vacf_package, energy_force_package)
        
        stacked_radii = stacked_radii[:, mask]
        stacked_radii = stacked_radii.reshape(-1, self.simulator.n_atoms, 3)
        if self.simulator.name == 'water':
            #extract all local neighborhoods of n_molecules molecules (centered around each atom)
    
            #pick centers of each local neighborhood (always an Oxygen atom)
            center_atoms = 3* np.random.choice(np.arange(int(self.simulator.n_atoms / 3)), self.simulator.n_local_neighborhoods, replace=False)
            local_neighborhoods = [n_closest_molecules(stacked_radii, \
                                                                    center_atom, self.simulator.n_closest_molecules, \
                                                                    self.simulator.cell) \
                                                                    for center_atom in center_atoms]
            
            local_stacked_radii = torch.stack([local_neighborhood[0] for local_neighborhood in local_neighborhoods], dim = 1)
            atomic_indices = torch.stack([local_neighborhood[1] for local_neighborhood in local_neighborhoods], dim=1)
            
            #compute observables on local environments
            # rdfs = torch.cat([self.simulator.rdf_mae(s.unsqueeze(0).unsqueeze(0))[0] for s in local_stacked_radii.reshape(-1, self.simulator.n_atoms_local, 3)])
            # rdfs = rdfs.reshape(local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1)
            # adfs = torch.zeros(rdfs.shape[0], rdfs.shape[1], 180).to(self.simulator.device) #TODO: temporary
            imds = torch.cat([self.simulator.min_imd(s.unsqueeze(0).unsqueeze(0)) for s in local_stacked_radii.reshape(-1, self.simulator.n_atoms_local, 3)])
            imds = imds.reshape(local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1)
            bond_lens = torch.cat([self.simulator.bond_length_dev(s.unsqueeze(0).unsqueeze(0))[0] for s in local_stacked_radii.reshape(-1, self.simulator.n_atoms_local, 3)])
            bond_len_devs = torch.cat([self.simulator.bond_length_dev(s.unsqueeze(0).unsqueeze(0))[1] for s in local_stacked_radii.reshape(-1, self.simulator.n_atoms_local, 3)])
            bond_lens = bond_lens.reshape(local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1)
            bond_len_devs = bond_len_devs.reshape(local_stacked_radii.shape[0], local_stacked_radii.shape[1], -1)

            #scheme to subsample local neighborhoods
            bond_len_devs_temp = bond_len_devs.reshape(-1, 1).cpu()
            bond_lens_temp = bond_lens.reshape(-1, bond_lens.shape[-1]).cpu()
            hist, bins = np.histogram(bond_len_devs_temp, bins = 100)
            bin_assignments = np.digitize(bond_len_devs_temp, bins)
            samples_per_bin = 5 * self.simulator.n_replicas

            full_list = []
            for bin_index in range(1, 101):
                # Filter elements belonging to the current bin
                bin_elements = bond_lens_temp[(bin_assignments == bin_index).squeeze(-1)]
                
                # Randomly sample elements from the bin, if available
                if len(bin_elements) >= samples_per_bin:
                    shuffle_idx = torch.randperm(bin_elements.shape[0])
                    sampled_elements = bin_elements[shuffle_idx][:samples_per_bin]
                else:
                    # If less than 5 elements, take all available
                    sampled_elements = torch.Tensor(bin_elements)

                full_list = full_list + [sampled_elements]
            full_list = torch.cat(full_list).to(self.simulator.device)
            #subsample the relevant frames
            
            indices = [torch.cat([x[0].unsqueeze(0).to(self.simulator.device) for x in torch.where(bond_lens == value)[0:2]]) for value in full_list]
            local_stacked_radii = torch.stack([local_stacked_radii[idx[0], idx[1]] for idx in indices])
            stacked_radii = torch.stack([stacked_radii[idx[0]] for idx in indices])
            atomic_indices = torch.stack([atomic_indices[idx[0], idx[1]] for idx in indices])
            bond_lens = full_list

            np.save(os.path.join(self.simulator.save_dir, f'local_stacked_radii_epoch{self.simulator.epoch}.npy'), local_stacked_radii.cpu())
            np.save(os.path.join(self.simulator.save_dir, f'local_rdfs_epoch{self.simulator.epoch}.npy'), rdfs.cpu())
            np.save(os.path.join(self.simulator.save_dir, f'local_imds_epoch{self.simulator.epoch}.npy'), imds.cpu())
            np.save(os.path.join(self.simulator.save_dir, f'local_bond_len_devs_epoch{self.simulator.epoch}.npy'), bond_len_devs.cpu())
        
        else:
            rdfs = rdfs[:, mask].reshape(-1, rdfs.shape[-1])
            adfs = adfs[:, mask].reshape(-1, adfs.shape[-1])
        
        
        rdfs.requires_grad = True
        adfs.requires_grad = True

        #TODO: scale the estimator by Kb*T
        #shuffle the radii, rdfs, and losses
        if self.simulator.shuffle:   
            shuffle_idx = torch.randperm(stacked_radii.shape[0])
            stacked_radii = stacked_radii[shuffle_idx]
            if self.simulator.name == 'water':
                atomic_indices = atomic_indices[shuffle_idx]
                bond_lens = bond_lens[shuffle_idx]
            else:
                rdfs = rdfs[shuffle_idx]
                adfs = adfs[shuffle_idx]
        
        bsize=MINIBATCH_SIZE
        num_blocks = math.ceil(stacked_radii.shape[0]/ bsize)
        rdf_gradient_estimators = []
        adf_gradient_estimators = []
        raw_grads = []
        pe_grads_flattened = []
        if self.simulator.name == 'water':
            print(f"Computing RDF/ADF gradients from {stacked_radii.shape[0]} local environments of {local_stacked_radii.shape[-2]} atoms in minibatches of size {MINIBATCH_SIZE}")
        else:
            print(f"Computing RDF/ADF gradients from {stacked_radii.shape[0]} structures in minibatches of size {MINIBATCH_SIZE}")
        num_params = len(list(model.parameters()))
        #first compute gradients of potential energy (in batches of size n_replicas)
        
        for i in tqdm(range(num_blocks)):
            start = bsize*i
            end = bsize*(i+1)
            with torch.enable_grad():
                radii_in = stacked_radii[start:end]
                if self.simulator.name == 'water':
                    atomic_indices_in = atomic_indices[start:end]
                radii_in.requires_grad = True
                energy_force_output = self.simulator.force_calc(radii_in, retain_grad = True, output_individual_energies = self.simulator.name == 'water')
                if len(energy_force_output) == 2:
                    #global energy
                    energy = energy_force_output[0] 
                elif len(energy_force_output) == 3:
                    #individual atomic energies
                    energy = energy_force_output[1].reshape(radii_in.shape[0], -1, 1)
                    #sum atomic energies within local neighborhoods
                    energy = torch.stack([energy[i, atomic_index].sum() for i, atomic_index in enumerate(atomic_indices_in)])
                    energy = energy.reshape(-1, 1)

            #need to do another loop here over batches of local neighborhoods
            num_local_blocks = math.ceil(energy.shape[0]/ bsize)
            for i in range(num_local_blocks):
                start_inner = bsize*i
                end_inner = bsize*(i+1)
                local_energy = energy[start_inner:end_inner]
                def get_vjp(v):
                    return compute_grad(inputs = list(model.parameters()), output = local_energy, grad_outputs = v, allow_unused = True, create_graph = False)
                vectorized_vjp = vmap(get_vjp)
                I_N = torch.eye(local_energy.shape[0]).unsqueeze(-1).to(self.device)
                num_samples = local_energy.shape[0]
                if self.simulator.model_type == 'forcenet': #dealing with device mismatch error
                    grads_vectorized = [process_gradient(model.parameters(), \
                                        compute_grad(inputs = list(model.parameters()), \
                                        output = e, allow_unused = True, create_graph = False), \
                                        self.device) for e in local_energy]
                    grads_flattened = torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for j in range(num_params)]) for i in range(num_samples)])
                else:
                    grads_vectorized = vectorized_vjp(I_N)
                    #flatten the gradients for vectorization
                    grads_flattened=torch.stack([torch.cat([grads_vectorized[i][j].flatten().detach() for i in range(num_params)]) for j in range(num_samples)])
            
                pe_grads_flattened.append(grads_flattened)

        
        pe_grads_flattened = torch.cat(pe_grads_flattened)
        #Now compute final gradient estimators in batches of size MINIBATCH_SIZE
        num_blocks = math.ceil(stacked_radii.shape[0]/ (MINIBATCH_SIZE))
        
        for i in tqdm(range(num_blocks)):
            start = MINIBATCH_SIZE*i
            end = MINIBATCH_SIZE*(i+1)
            if self.simulator.training_observable == 'rdf':
                obs = rdfs
            elif self.simulator.training_observable == 'imd':
                obs = imds
            elif self.simulator.training_observable == 'bond_length_dev':
                obs = bond_lens
                
            obs = obs.reshape(pe_grads_flattened.shape[0], -1)                    
            obs_batch = [obs[start:end]]
            pe_grads_batch = pe_grads_flattened[start:end]
            #TODO: fix the case where we don't use the MSE gradient
            grad_outputs_obs = [2/gt_rdf_var_local*(ob.mean(0) - gt_ob).unsqueeze(0) \
                                if self.simulator.use_mse_gradient else None \
                                for ob, gt_ob, gt_rdf_var_local in zip(obs_batch, \
                                self.gt_rdf_local.chunk(len(obs_batch)), self.gt_rdf_var_local.chunk(len(obs_batch)))]
            final_vjp = [self.estimator(obs, pe_grads_batch, grad_output_obs) for obs, grad_output_obs in zip(obs_batch, grad_outputs_obs)]
            final_vjp = torch.stack(final_vjp).mean(0)
            if self.params.adf_loss_weight !=0 and self.simulator.name != 'water':
                adfs = adfs.reshape(pe_grads_flattened.shape[0], -1)
                adf_batch = adfs[start:end]
                grad_outputs_adf = 2*(adf_batch.mean(0) - self.gt_adf).unsqueeze(0) \
                                if self.simulator.use_mse_gradient else None
                gradient_estimator_adf = self.estimate(adf_batch, grads_flattened, grad_outputs_adf)
                final_vjp+=self.params.adf_loss_weight*gradient_estimator_adf             

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