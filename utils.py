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


# Initialize configuration
# Radii
# FCC lattice
def fcc_positions(n_particle, box, device):
    
    # round-up to nearest fcc box
    cells = np.ceil((n_particle/4.0)**(1.0/3.0)).astype(np.int32) #cells in each dimension (assume 4 particles per unit cell)
    cell_size = box/cells 
    radius_ = np.empty((n_particle,3)) #initial positions of particles
    r_fcc = np.array ([[0.25,0.25,0.25],[0.25,0.75,0.75],[0.75,0.75,0.25],[0.75,0.25,0.75]], dtype=np.float64)
    i = 0
    for ix, iy, iz in product(list(range(cells)),repeat=3): # triple loop over unit cells
        for a in range(4): # 4 atoms in a unit cell
            radius_[i,:] = r_fcc[a,:] + np.array([ix,iy,iz]).astype(np.float64) # 0..nc space
            radius_[i,:] = radius_[i,:]*cell_size#/self.box # normalize to [0,1]
            i = i+1
            if(i==n_particle): #  break when we have n_particle in our box
                return torch.Tensor(radius_ - box/2).to(device) # convert to -L/2 to L/2 space for ease with PBC


   
# Procedure to initialize velocities
def initialize_velocities(n_particle, masses, temp, n_replicas):

    masses = masses.cpu().numpy()
    vel_dist = maxwell()
    momenta = masses * vel_dist.rvs(size = (n_replicas, n_particle, 3))
    #shift so that initial momentum is zero
    momenta -= np.mean(momenta, axis = -2, keepdims=True)

    #scale velocities to match desired temperature
    ke = (momenta**2 / (2*masses)).sum(axis = (1,2), keepdims=True)
    targeEkin = 0.5 * (3.0 * n_particle) * temp
    correction_factor = np.sqrt(targeEkin / ke)
    momenta *= correction_factor
    velocities = momenta/masses
    return torch.Tensor(velocities)

#inverse cdf for power law with exponent 'power' and min value y_min
def powerlaw_inv_cdf(y, power, y_min):
    return y_min*((1-y)**(1/(1-power)))
    

def dump_params_to_yml(params, filepath):
    with open(os.path.join(filepath, "config.yaml"), 'w') as f:
        yaml.dump(params, f)


def print_active_torch_tensors():
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #print(type(obj), obj.size())
                count +=1
                del obj
        except:
            pass
    print(f"{count} tensors in memory")

'''Helper function to compare true underlying potential to the learned one'''
def plot_pair(epoch, path, model, device, end, target_pot): 

    x = torch.linspace(0.1, end, 250)[:, None].to(device)
    u_fit = torch.Tensor([model(i) for i in x]).detach().cpu().numpy()
    u_fit = u_fit - u_fit[-1] #shift the prior so that it has a value of zero at the cutoff point
    u_target = torch.Tensor([target_pot(i) for i in x]).detach().cpu().numpy()

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)
    
    plt.plot( x.detach().cpu().numpy(), 
              u_target,
               label='truth', 
               linewidth=2,linestyle='--', c='black')
    plt.ylim(-2.0, 6.0)
    plt.legend()      
    plt.show()
    plt.savefig(os.path.join(path, 'potential_{}.jpg'.format(epoch)), bbox_inches='tight')
    plt.close()


def solve_continuity_system(device, x_c, n, m, epsilon):
    A = torch.Tensor([[1, x_c**2, x_c**4],
                  [0, 2*x_c, 4*x_c**3],
                  [0, 2, 12*x_c**2]]).to(device)
    if m ==0:
        B = epsilon * torch.Tensor([[- x_c**-n],
                            [n*x_c**-(n+1)],
                            [ - n*(n+1)*x_c**-(n+2)]]).to(device)
    else:
        B = epsilon * torch.Tensor([[x_c**-m - x_c**-n],
                                [n*x_c**-(n+1) - m*x_c**-(m+1)],
                                [m*(m+1)*x_c**-(m+2) - n*(n+1)*x_c**-(n+2)]]).to(device)
    c = torch.linalg.solve(A, B)

    return c[0], c[1], c[2]

#MD17 utils
def get_hr(traj, bins):
    '''
    compute h(r) (the RDF) for MD17 simulations.
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
    DATAPATH = f'{base_path}/{name}/{molecule}/{size}/test/nequip_npz.npz'
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
    hist_adf = diff_adf(gt_traj.to(device))
    return torch.Tensor(hist_gt).to(device), torch.Tensor(hist_adf).to(device)


def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1,1,3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(dim=-1))



        
