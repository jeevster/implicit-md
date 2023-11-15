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

#MD17/MD22 utils
def get_hr(traj, bins):
    '''
    compute h(r) (the RDF) for MD17 and MD22 simulations.
    traj: T x N_atoms x 3
    '''
    pdist = torch.cdist(traj, traj).flatten()
    hist, _ = np.histogram(pdist[:].flatten().numpy(), bins, density=True)
    return hist
    
def find_hr_adf_from_file(base_path: str, name: str, molecule: str, size: str, params, device):
    #RDF plotting parameters
    xlim = params.max_rdf_dist
    n_bins = int(xlim/params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing h(r)
    # load ground truth data
    DATAPATH = os.path.join(base_path, name, molecule, size, 'train/nequip_npz.npz')
    gt_data = np.load(DATAPATH)
    gt_traj = torch.FloatTensor(gt_data.f.R)
    hist_gt = get_hr(gt_traj, bins)
    hist_gt = 100*hist_gt/ hist_gt.sum()

    #ADF
    temp_data = LmdbDataset({'src': os.path.join(base_path, name, molecule, size, 'train')})
    init_data = temp_data.__getitem__(0)
    n_atoms = init_data['pos'].shape[0]
    atoms = data_to_atoms(init_data)
    #extract bond and atom type information
    NL = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    NL.update(atoms)
    bonds = torch.tensor(NL.get_connectivity_matrix().todense().nonzero()).to(device).T
    raw_atoms = data_to_atoms(temp_data.__getitem__(0))
    cell = torch.Tensor(raw_atoms.cell).to(device)
    diff_adf = DifferentiableADF(n_atoms, bonds, cell, params, device)
    hist_adf = diff_adf(gt_traj[:3000].to(device))
    return torch.Tensor(hist_gt).to(device), torch.Tensor(hist_adf).to(device)