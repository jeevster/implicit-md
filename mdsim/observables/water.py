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
from mdsim.common.utils import data_to_atoms
from mdsim.datasets.lmdb_dataset import LmdbDataset
from ase.neighborlist import natural_cutoffs, NeighborList

#Water utils
def get_diffusivity_traj(pos_seq, dilation=1):
    """
    Input: B x N x T x 3
    Output: B x T
    """
    # substract CoM
    bsize, time_steps = pos_seq.shape[0], pos_seq.shape[2]
    pos_seq = pos_seq - pos_seq.mean(1, keepdims=True)
    msd = (pos_seq[:, :, 1:] - pos_seq[:, :, 0].unsqueeze(2)).pow(2).sum(dim=-1).mean(dim=1)
    diff = msd / (torch.arange(1, time_steps)*dilation) / 6
    return diff.view(bsize, time_steps-1)

def get_smoothed_diffusivity(xyz):
    seq_len = xyz.shape[0] - 1
    diff = torch.zeros(seq_len)
    for i in range(seq_len):
        diff[:seq_len-i] += get_diffusivity_traj(xyz[i:].transpose(0, 1).unsqueeze(0)).flatten()
    diff = diff / torch.flip(torch.arange(seq_len),dims=[0])
    return diff

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
    
    data_seq = ((data_seq / lattices) % 1) * lattices
    all_rdfs = {}
    n_rdfs = 3
    for idx in range(n_rdfs):
        type1, type2 = pairs[idx]    
        indices0 = type2indices[type1].to(device)
        indices1 = type2indices[type2].to(device)
        data_pdist = distance_pbc_select(data_seq, lattices, indices0, indices1)
        
        data_pdist = data_pdist.flatten().cpu().numpy()
        data_shape = data_pdist.shape[0]
            
        data_pdist = data_pdist[data_pdist != 0]
        data_hist, _ = np.histogram(data_pdist, bins)
        rho_data = data_shape / torch.prod(lattices).cpu().numpy() 
        Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        data_rdf = data_hist / Z_data
        all_rdfs[type1 + type2] = torch.Tensor([data_rdf]).to(device)
        
    return all_rdfs


def find_water_rdfs_diffusivity_from_file(base_path: str, size: str, params, device):
    xlim = params.max_rdf_dist
    n_bins = int(xlim/params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing RDF

    # get ground truth data
    DATAPATH = os.path.join(base_path, 'water', size, 'val/nequip_npz.npz')
    gt_data = np.load(DATAPATH, allow_pickle=True)
    atom_types = torch.tensor(gt_data.f.atom_types)
    lattices = torch.tensor(gt_data.f.lengths[0]).float()
    gt_traj = torch.tensor(gt_data.f.unwrapped_coords)
    gt_diffusivity = get_smoothed_diffusivity(gt_traj[0::100, atom_types==8])[:100].to(device) # track diffusivity of oxygen atoms, unit is A^2/ps
    #recording frequency of underlying data is 10 fs. 
    #Want to match frequency of our data collection which is params.n_dump*params.integrator_config["dt"]
    keep_freq = math.ceil(params.n_dump*params.integrator_config["timestep"] / 10)
    gt_rdfs = get_water_rdfs(gt_traj[::keep_freq], atom_types, lattices, bins, device)
    return gt_rdfs, gt_diffusivity