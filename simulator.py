import numpy as np
import gsd.hoomd
import torch
import math
import os
from tqdm import tqdm
from torch_geometric.nn import radius_graph
from torch.utils.data import DataLoader
import ase
from ase import Atoms, units
import nequip.scripts.deploy
from nequip.train.trainer import Trainer
from nequip.train.loss import Loss
from nequip.data import AtomicData, AtomicDataDict
from nequip.utils.torch_geometric import Batch
from nequip.utils import atomic_write, load_file
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.md.ase_utils import OCPCalculator
from mdsim.md.integrator import NoseHoover, Langevin
from mdsim.md.calculator import ForceCalculator
from mdsim.common.registry import registry
from mdsim.common.utils import data_to_atoms, process_gradient, initialize_velocities, dump_params_to_yml
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.datasets.lmdb_dataset import LmdbDataset, data_list_collater
from mdsim.observables.common import distance_pbc, BondLengthDeviation, compute_distance_matrix_batch
from mdsim.observables.water import WaterRDFMAE, MinimumIntermolecularDistance
from mdsim.observables.lips import LiPSRDFMAE, cart2frac, frac2cart

MAX_SIZES = {'md17': '10k', 'md22': '100percent', 'water': '10k', 'lips': '20k'}

class Simulator():
    def __init__(self, config, params, model, model_path, model_config, gt_rdf):
        super(Simulator, self).__init__()
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
        lmax_string = f"lmax={self.l_max}_" if self.model_type == "nequip" else ""
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

        self.masses = torch.Tensor(self.atoms.get_masses().reshape(1, -1, 1)).to(self.device)
        self.r_max_key = "r_max" if self.model_type == "nequip" else "cutoff"

    
        self.nsteps = params.steps
        self.eq_steps = params.eq_steps
        #ensure that the number of logged steps is a multiple of the vacf window (for chopping up the trajectory)
        self.nsteps -= (self.nsteps - self.eq_steps) % self.vacf_window
        if (self.nsteps - self.eq_steps) < self.vacf_window:
            self.nsteps = self.eq_steps + 2*self.vacf_window #at least two windows
        while self.nsteps < params.steps: #nsteps should be at least as long as what was requested
            self.nsteps += self.vacf_window
        self.ps_per_epoch = self.nsteps * self.config["timestep"] // 1000.
        
        self.temp = self.config['temperature'] * units.kB
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
        self.zeta = torch.zeros((self.n_replicas, 1, 1)).to(self.device)
        self.checkpoint_zetas.append(self.zeta)
        self.original_radii = self.radii.clone()
        self.original_velocities = self.velocities.clone()
        self.original_zeta = self.zeta.clone()
        self.all_radii = []
        
        #create batch of atoms to be operated on
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

        #Initialize calculator
        self.calculator = ForceCalculator(self.model, self.model_type, self.model_config, self.r_max_key, self.n_atoms, self.atomic_numbers, self.atoms_batch, self.device)

        #Initialize integrator
        self.dt = config['timestep']
        self.integrator_type = self.config["integrator"]
        self.integrator = registry.get_integrator_class(self.integrator_type)(self.calculator, self.masses, self.n_replicas, self.n_atoms, self.config, self.device)
            
        self.diameter_viz = params.diameter_viz
        self.exp_name = params.exp_name
        self.training_observable = params.training_observable
        self.rdf_loss_weight = params.rdf_loss_weight
        self.diffusion_loss_weight = params.diffusion_loss_weight
        self.vacf_loss_weight = params.vacf_loss_weight
        self.energy_force_loss_weight = params.energy_force_loss_weight

        
        #limit CPU usage
        torch.set_num_threads(1)
    
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
                        # normalizes the targets
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
            _, self.forces = self.integrator.calculator.calculate_energy_force(self.radii)
            #Run MD
            print("Start MD trajectory", file=self.f)
            for step in tqdm(range(self.nsteps)):
                self.step = step
                #MD Step
                radii, velocities, energy, forces, zeta = self.integrator.step(self.radii, self.velocities, self.forces, self.zeta, retain_grad = False)
                # dump frames
                if self.step%self.n_dump == 0:
                    print(self.step, self.thermo_log(energy), file=self.f)
                    step  = self.step if self.train else (self.epoch+1) * self.step #don't overwrite previous epochs at inference time
                    try:   
                        self.t.append(self.create_frame(frame = step/self.n_dump))
                    except:
                        pass
                
                #save trajectory for gradient calculation
                if step >= self.eq_steps:
                    self.running_radii.append(radii.detach().clone())
                    self.running_vels.append(velocities.detach().clone())
                    self.running_accs.append((forces/self.masses).detach().clone())
                    
                    if step % self.n_dump == 0 and not self.train:
                        self.all_radii.append(radii.detach().cpu()) #save whole trajectory without resetting at inference time
                
                self.radii.copy_(radii)
                self.velocities.copy_(velocities)
                self.zeta.copy_(zeta)
                self.forces.copy_(forces)
            self.stacked_radii = torch.stack(self.running_radii[::self.n_dump])

            #compute instability metric (either bond length deviation, min intermolecular distance, or RDF MAE)
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


    def create_frame(self, frame):
        # Particle positions, velocities, diameter
        radii = self.radii[0]
        if self.pbc:
            #wrap for visualization purposes
            if self.name == 'lips': #account for non-cubic cell
                frac_radii = cart2frac(radii, self.cell)
                frac_radii = frac_radii % 1.0 #wrap
                radii = frac2cart(frac_radii, self.cell)
            else:
                radii = ((radii / torch.diag(self.cell)) % 1) * torch.diag(self.cell)  - torch.diag(self.cell)/2 #wrap coords (last subtraction is for cell alignment in Ovito)
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
        
        cell = torch.Tensor(self.atoms.cell)
        s.configuration.box=[cell[0][0], cell[1][1], cell[2][2], 0, 0, 0]
        s.configuration.step = self.dt

        if self.name != 'lips': #don't show bonds if lips
            s.bonds.N = self.bonds.shape[0]
            s.bonds.group = detach_numpy(self.bonds)
        return s
    
    def thermo_log(self, pe):
        #Log energies and instabilities
        p_dof = 3*self.n_atoms
        ke = 1/2 * (self.masses*torch.square(self.velocities)).sum(axis = (1,2)).unsqueeze(-1)
        temp = (2*ke/p_dof).mean() / units.kB
        instability = self.stability_criterion(self.radii.unsqueeze(0))
        if isinstance(instability, tuple):
            instability = instability[-1]
        results_dict = {"Temperature": temp.item(),
                        "Potential Energy": pe.mean().item(),
                        "Total Energy": (ke+pe).mean().item(),
                        "Momentum Magnitude": torch.norm(torch.sum(self.masses*self.velocities, axis =-2)).item(),
                        'Max Bond Length Deviation': self.bond_length_dev(self.radii.unsqueeze(0))[1].mean().item() \
                                                      if self.pbc else instability.mean().item()}
        if self.pbc:
            results_dict['Minimum Intermolecular Distance'] = instability.mean().item()
            results_dict['RDF MAE'] = self.rdf_mae(self.radii.unsqueeze(0))[-1].mean().item()
        return results_dict
