import torch
import numpy as np
from itertools import product
from scipy.stats import maxwell
import math
import yaml
import os
import gc
import matplotlib.pyplot as plt
from torchmd.observable import DifferentiableRDF, DifferentiableADF
from mdsim.common.custom_radius_graph import detach_numpy
from mdsim.common.utils import data_to_atoms
from mdsim.observables.common import distance_pbc
from mdsim.datasets.lmdb_dataset import LmdbDataset
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.observables.common import radii_to_dists, get_smoothed_diffusivity

#Water utils
class WaterRDFMAE(torch.nn.Module):
    def __init__(self, base_path, gt_rdfs, n_atoms, n_replicas, params, device):
        super(WaterRDFMAE, self).__init__()
        self.gt_rdfs = gt_rdfs
        self.n_atoms = n_atoms
        self.n_replicas = n_replicas
        self.device = device
        self.params = params
        self.xlim = params.max_rdf_dist
        n_bins = int(self.xlim/params.dr)
        self.bins = np.linspace(1e-6, self.xlim, n_bins + 1) # for computing RDF
        # get ground truth data
        DATAPATH = os.path.join(base_path, 'water', '1k', 'val/nequip_npz.npz')
        gt_data = np.load(DATAPATH, allow_pickle=True)
        self.ptypes = torch.tensor(gt_data.f.atom_types)
        self.lattices = torch.tensor(gt_data.f.lengths[0]).float()


    def forward(self, stacked_radii):
        max_maes = []
        rdf_list = []
        for i in range(self.n_replicas): #explicit loop since vmap makes some numpy things weird
            rdfs, _ = get_water_rdfs(stacked_radii[:, i], self.ptypes, self.lattices, self.bins, self.device)
            #compute MAEs of all element-conditioned RDFs
            max_maes.append(torch.max(torch.cat([self.xlim*torch.abs(rdf-gt_rdf).mean().unsqueeze(-1) for rdf, gt_rdf in zip(rdfs.values(), self.gt_rdfs.values())])))
            rdf_list.append(torch.cat([rdf.flatten() for rdf in rdfs.values()]))
        return torch.stack(rdf_list).to(self.device), torch.stack(max_maes).to(self.device)
  

#minimum distance between two atoms on different water molecules
class MinimumIntermolecularDistance(torch.nn.Module):
    def __init__(self, bonds, cell, device, element_mask = None):
        super(MinimumIntermolecularDistance, self).__init__()
        self.cell = cell
        self.device = device
        #construct a tensor containing all the intermolecular bonds 
        num_atoms = 192
        missing_edges = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                if not ((i % 3 == 0 and (j == i + 1 or j == i + 2)) or (j % 3 == 0 and (i == j + 1 or i == j + 2))): #i and j are not on the same atom
                    if element_mask == 'O':
                        if i % 3 == 0 and j % 3 == 0: #both oxygen
                            missing_edges.append([i, j])
                    elif element_mask == 'H':
                        if i % 3 != 0 and j%3 != 0: #both hydrogen
                            missing_edges.append([i, j])
                    elif element_mask is None:
                        missing_edges.append([i, j])
        self.intermolecular_edges = torch.Tensor(missing_edges).to(torch.long)

    def forward(self, stacked_radii):
        stacked_radii = ((stacked_radii / torch.diag(self.cell)) % 1) * torch.diag(self.cell) #wrap coords
        intermolecular_distances = distance_pbc(stacked_radii[:, :, self.intermolecular_edges[:, 0]], \
                                                stacked_radii[:,:, self.intermolecular_edges[:, 1]], \
                                                torch.diag(self.cell)).to(self.device) #compute distances under minimum image convention
        return intermolecular_distances.min(dim=-1)[0].min(dim=0)[0].detach()


def distance_pbc_select(x, lattices, indices0, indices1):
    x0 = x[:, indices0]
    x1 = x[:, indices1]
    x0_size = x0.shape[1]
    x1_size = x1.shape[1]
    x0 = x0.repeat([1, x1_size, 1])
    x1 = x1.repeat_interleave(x0_size, dim=1)
    delta = torch.abs(x0 - x1)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(axis=-1))

def get_water_rdfs(data_seq, ptypes, lattices, bins, device='cpu'):
    """
    get atom-type conditioned water RDF curves.
    """
    data_seq = data_seq.to(device).float()
    lattices = lattices.to(device).float()
    
    type2indices = {
        'H': ptypes == 1,
        'O': ptypes == 8
    }
    pairs = [('O', 'O'), ('H', 'H'), ('H', 'O')]
    
    data_seq = ((data_seq / lattices) % 1) * lattices #coords are wrapped
    all_rdfs = {}
    all_rdfs_vars = {}
    n_rdfs = 3
    for idx in range(n_rdfs):
        type1, type2 = pairs[idx]    
        indices0 = type2indices[type1].to(device)
        indices1 = type2indices[type2].to(device)
        #Original Method
        data_pdist = distance_pbc_select(data_seq, lattices, indices0, indices1)
        data_pdist = data_pdist.flatten().cpu().numpy()
        data_shape = data_pdist.shape[0]
        data_pdist = data_pdist[data_pdist != 0]
        data_hist, _ = np.histogram(data_pdist, bins)
        rho_data = data_shape / torch.prod(lattices).cpu().numpy()
        Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        data_rdf = data_hist / Z_data

        #New Method
        data_pdist = distance_pbc_select(data_seq, lattices, indices0, indices1)
        data_pdist = data_pdist.cpu().numpy()
        data_shape = data_pdist.shape[0]
        data_hists = np.stack([np.histogram(dist, bins)[0] for dist in data_pdist])
        rho_data = data_shape / torch.prod(lattices).cpu().numpy()
        Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        data_rdfs = data_hists / Z_data
        data_rdfs = data_rdfs / data_rdfs.sum(1, keepdims=True) * data_rdf.sum()#normalize to match original sum
        data_rdf_mean = data_rdfs.mean(0)
        data_rdf_var = data_rdfs.var(0)
        all_rdfs[type1 + type2] = torch.Tensor([data_rdf_mean]).to(device)
        all_rdfs_vars[type1 + type2] = torch.Tensor([data_rdf_var]).to(device)
    return all_rdfs, all_rdfs_vars


def find_water_rdfs_diffusivity_from_file(base_path: str, size: str, params, device):
    xlim = params.max_rdf_dist
    n_bins = int(xlim/params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing RDF

    # get ground truth data
    DATAPATH = os.path.join(base_path, 'water', size, 'test/nequip_npz.npz')
    gt_data = np.load(DATAPATH, allow_pickle=True)
    atom_types = torch.tensor(gt_data.f.atom_types)
    oxygen_atoms_mask = atom_types==8
    lattices = torch.tensor(gt_data.f.lengths[0]).float()
    gt_traj = torch.tensor(gt_data.f.unwrapped_coords)
    gt_data_continuous = np.load(os.path.join(base_path, 'contiguous-water', '90k', 'train/nequip_npz.npz'))
    gt_traj_continuous = torch.tensor(gt_data_continuous.f.unwrapped_coords)
    gt_diffusivity = get_smoothed_diffusivity(gt_traj_continuous[0::100, atom_types==8])[:100].to(device) # track diffusivity of oxygen atoms, unit is A^2/ps
    #recording frequency of underlying data is 10 fs. 
    #Want to match frequency of our data collection which is params.n_dump*params.integrator_config["dt"]
    keep_freq = math.ceil(params.n_dump*params.integrator_config["timestep"] / 10)
    gt_rdfs = get_water_rdfs(gt_traj[::keep_freq], atom_types, lattices, bins, device)
    
    #ADF
    temp_data = LmdbDataset({'src': os.path.join(base_path, 'water', size, 'train')})
    init_data = temp_data.__getitem__(0)
    atoms = data_to_atoms(init_data)
    #extract bond and atom type information
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T
    gt_adf = DifferentiableADF(gt_traj.shape[-2], bonds, torch.diag(lattices).to(device), params, device)(gt_traj[0:200][::keep_freq].to(torch.float).to(device))
    #TODO: O-O conditioned RDF using oxygen_atoms_mask
    #gt_adf = DifferentiableADF(gt_traj.shape[-2], bonds, torch.diag(lattices).to(device), params, device)(gt_traj[0:2000, oxygen_atoms_mask][::keep_freq].to(torch.float).to(device))
    return gt_rdfs, gt_diffusivity, gt_adf, oxygen_atoms_mask