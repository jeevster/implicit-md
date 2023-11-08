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

class BondLengthDeviation(torch.nn.Module):
    def __init__(self, bonds, mean_bond_lens, cell, device):
        super(BondLengthDeviation, self).__init__()
        self.bonds = bonds
        self.mean_bond_lens = mean_bond_lens
        self.cell = cell
        self.device = device

    def forward(self, stacked_radii):
        bond_lens = distance_pbc(stacked_radii[:, :, self.bonds[:, 0]], stacked_radii[:,:, self.bonds[:, 1]], torch.diag(self.cell)).to(self.device)
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