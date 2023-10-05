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
class BondLengthDeviation(torch.nn.Module):
    def __init__(self, bonds, mean_bond_lens, cell, device):
        super(BondLengthDeviation, self).__init__()
        self.bonds = bonds
        self.mean_bond_lens = mean_bond_lens
        self.cell = cell
        self.device = device

    def forward(self, stacked_radii):
        bond_lens = distance_pbc(stacked_radii[:, :, self.bonds[:, 0]], self.stacked_radii[:,:, self.bonds[:, 1]], cell).to(self.device)
        max_bond_dev_per_replica = (bond_lens - self.mean_bond_lens).abs().max(dim=-1)[0].max(dim=0)[0].detach()
        return max_bond_dev_per_replica



def radii_to_dists(radii, params):
    #Get rij matrix
    r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
    
    # #Enforce minimum image convention
    # r = -1*torch.where(r > 0.5*params.box, r-params.box, torch.where(r<-0.5*params.box, r+params.box, r))

    #get rid of diagonal 0 entries of r matrix (for gradient stability)
    r = r[:, ~torch.eye(r.shape[1],dtype=bool)].reshape(r.shape[0], r.shape[1], -1, 3)
    try:
        r.requires_grad = True
    except RuntimeError:
        pass

    #compute distance matrix:
    return torch.sqrt(torch.sum(r**2, axis=-1)).unsqueeze(-1)

def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1,1,3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(dim=-1))

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
    DATAPATH = os.path.join(base_path, name, molecule, size, 'val/nequip_npz.npz')
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
    hist_adf = diff_adf(gt_traj[0:1].to(device))
    return torch.Tensor(hist_gt).to(device), torch.Tensor(hist_adf).to(device)