import torch
from torch_cluster import radius_graph
import numpy as np
import os
from ase.io import read
from mdsim.observables.water import get_smoothed_diffusivity
from tqdm import tqdm
import itertools

class LiPSRDFMAE(torch.nn.Module):
    def __init__(self, base_path, gt_rdf, n_atoms, n_replicas, params, device):
        super(LiPSRDFMAE, self).__init__()
        self.gt_rdf = gt_rdf
        self.n_atoms = n_atoms
        self.n_replicas = n_replicas
        self.device = device
        self.params = params
        self.xlim = params.max_rdf_dist
        n_bins = int(self.xlim/params.dr)
        self.bins = np.linspace(1e-6, self.xlim, n_bins + 1) # for computing RDF
        # get ground truth data
        atoms = read(os.path.join(base_path, 'lips', 'lips.xyz'), index=':', format='extxyz')
        n_points = len(atoms)
        positions, cell, atomic_numbers = [], [], []
        for i in tqdm(range(n_points)):
            positions.append(atoms[i].get_positions())
            cell.append(atoms[i].get_cell())
            atomic_numbers.append(atoms[i].get_atomic_numbers())
        positions = torch.from_numpy(np.array(positions))
        self.cell = torch.from_numpy(np.array(cell)[0])


    def forward(self, stacked_radii):
        maes = []
        rdf_list = []
        for i in range(self.n_replicas): #explicit loop since vmap makes some numpy things weird
            rdf = get_lips_rdf(stacked_radii[:, i], self.cell, self.bins).to(self.device)
            #compute MAEs of all element-conditioned RDFs
            maes.append(self.xlim*torch.abs(rdf-self.gt_rdf).mean().unsqueeze(-1))
            rdf_list.append(rdf)
        return torch.stack(rdf_list).to(self.device), torch.stack(maes).to(self.device)
  

def compute_image_flag(cell, fcoord1, fcoord2):
    supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3))).to(cell.device)
    fcoords = fcoord2[:, None] + supercells
    coords = fcoords @ cell
    coord1 = fcoord1 @ cell
    dists = torch.cdist(coord1[:, None], coords).squeeze()
    image = dists.argmin(dim=-1)
    return supercells[image].long()

def frac2cart(fcoord, cell):
    return fcoord @ cell

def cart2frac(coord, cell):
    invcell = torch.linalg.inv(cell)
    return coord @ invcell

# the source data is in wrapped coordinates. need to unwrap it for computing diffusivity.
def unwrap(pos0, pos1, cell):
    fcoords1 = cart2frac(pos0, cell)
    fcoords2 = cart2frac(pos1, cell)
    flags = compute_image_flag(cell, fcoords1, fcoords2)
    remapped_frac_coords = cart2frac(pos1, cell) + flags
    return frac2cart(remapped_frac_coords, cell)

# different from previous functions, now needs to deal with non-cubic cells. 
def compute_distance_matrix_batch(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells+1, 1).to(cell.device)
    combos = torch.stack(
        torch.meshgrid(pos, pos, pos, indexing='xy')
            ).permute(3, 2, 1, 0).reshape(-1, 3).to(cell.device)
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    # NxNxCells distance array
    shifted = cart_coords.unsqueeze(2) + shifts.unsqueeze(0).unsqueeze(0)
    dist = cart_coords.unsqueeze(2).unsqueeze(2) - shifted.unsqueeze(1)
    dist = dist.pow(2).sum(dim=-1).sqrt()
    # But we want only min
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix

    
    
def get_lips_rdf(data_seq, lattices, bins, device='cpu'):
    data_seq = data_seq.to(device).float()
    lattices = lattices.to(device).float()
    
    lattice_np = lattices.cpu().numpy()
    volume = float(abs(np.dot(np.cross(lattice_np[0], lattice_np[1]), lattice_np[2])))
    data_dist = compute_distance_matrix_batch(lattices, data_seq)

    data_pdist = data_dist.flatten().cpu().numpy()
    data_shape = data_pdist.shape[0]

    data_pdist = data_pdist[data_pdist != 0]
    data_hist, _ = np.histogram(data_pdist, bins)

    rho_data = data_shape / volume
    Z_data = rho_data * 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rdf = data_hist / Z_data
        
    return torch.Tensor(rdf).to(device)

def find_lips_rdfs_diffusivity_from_file(base_path: str, size: str, params, device):
    xlim = params.max_rdf_dist
    n_bins = int(xlim/params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing RDF

    atoms = read(os.path.join(base_path, 'lips', 'lips.xyz'), index=':', format='extxyz')
    n_points = len(atoms)
    positions, cell, atomic_numbers = [], [], []
    for i in tqdm(range(n_points)):
        positions.append(atoms[i].get_positions())
        cell.append(atoms[i].get_cell())
        atomic_numbers.append(atoms[i].get_atomic_numbers())
    positions = torch.from_numpy(np.array(positions))
    cell = torch.from_numpy(np.array(cell)[0])
    atomic_numbers = torch.from_numpy(np.array(atomic_numbers)[0])

    # unwrap positions
    all_displacements = []
    for i in tqdm((range(1, len(positions)))):
        next_pos = unwrap(positions[i-1], positions[i], cell)
        displacements = next_pos - positions[i-1]
        all_displacements.append(displacements)
    displacements = torch.stack(all_displacements)
    accum_displacements = torch.cumsum(displacements, dim=0)
    positions = torch.cat([positions[0].unsqueeze(0), positions[0] + accum_displacements], dim=0)

    gt_rdf = get_lips_rdf(positions[::], cell, bins, device='cpu')
    # Li diffusivity unit in m^2/s. remove the first 5 ps as equilibrium.
    # Desirably, we want longer trajectories for computing diffusivity.
    gt_diffusivity = get_smoothed_diffusivity((positions[2500:None:25, atomic_numbers == 3])) * 20 * 1e-8
    return torch.Tensor(gt_rdf).to(device), torch.Tensor(gt_diffusivity).to(device)

