"""Summary
"""
import torch
import torchmd
from nff.nn.layers import GaussianSmearing
import numpy as np
from torchmd.system import check_system
from torchmd.topology import generate_nbr_list, get_offsets, generate_angle_list

def generate_vol_bins(start, end, nbins, dim):
    bins = torch.linspace(start, end, nbins + 1)
    
    # compute volume differential 
    if dim == 3:
        Vbins = 4 * np.pi /3*(bins[1:]**3 - bins[:-1]**3)
        V = (4/3)* np.pi * (end) ** 3
    elif dim == 2:
        Vbins = np.pi * (bins[1:]**2 - bins[:-1]**2)
        V = np.pi * (end) ** 2
        
    return V, torch.Tensor(Vbins), bins


def compute_angle(xyz, angle_list, cell, N):
    device = xyz.device
    xyz = xyz.reshape(-1, N, 3)
    bond_vec1 = xyz[angle_list[:,0], angle_list[:,1]] - xyz[angle_list[:,0], angle_list[:, 2]]
    bond_vec2 = xyz[angle_list[:,0], angle_list[:,3]] - xyz[angle_list[:,0], angle_list[:, 2]]
    #issue here with shape of cell
    # bond_vec1 = bond_vec1 + get_offsets(bond_vec1, cell, device) * cell
    # bond_vec2 = bond_vec2 + get_offsets(bond_vec2, cell, device) * cell  
    
    angle_dot = (bond_vec1 * bond_vec2).sum(-1)
    norm = ( bond_vec1.pow(2).sum(-1) * bond_vec2.pow(2).sum(-1) ).sqrt()
    cos = angle_dot / norm
    
    return cos

class DifferentiableRDF(torch.nn.Module):
    def __init__(self, params, device):
        super(DifferentiableRDF, self).__init__()
        start = 1e-6
        end =  params.max_rdf_dist #torch.max(self.running_dists)
        nbins = int((end-start)/params.dr) + 1
        self.cutoff_boundary = end + 5e-1
        self.index_tuple = None

        #GPU
        self.device = device

        V, vol_bins, bins = generate_vol_bins(start, end, nbins, dim=3)

        self.V = V
        self.vol_bins = vol_bins.to(self.device)
        #self.device = system.device
        self.bins = bins

        self.smear = GaussianSmearing(
            start=start,
            stop=bins[-1],
            n_gaussians=nbins,
            width=params.gaussian_width,
            trainable=False
        ).to(self.device)

    def forward(self, running_dists):
        running_dists = torch.cat(running_dists)
        count = self.smear(running_dists.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        return 100*count

#differentiable angular distribution function
class DifferentiableADF(torch.nn.Module):
    def __init__(self, n_atoms, bonds, cell, params, device):
        super(DifferentiableADF, self).__init__()
        #GPU
        self.device = device
        self.n_atoms = n_atoms
        self.bonds = bonds
        #create bond mask
        self.bonds_mask = torch.zeros(self.n_atoms, self.n_atoms).to(self.device)
        self.bonds_mask[self.bonds[:, 0], self.bonds[:, 1]] = 1
        self.bonds_mask[self.bonds[:, 1], self.bonds[:, 0]] = 1
        start = params.angle_range[0]
        end = params.angle_range[1]
        self.nbins = 180 # 1 bin for each angle
        self.bins = torch.linspace(start, end, self.nbins + 1).to(self.device)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=self.nbins,
            width=1.0,
            trainable=False
        ).to(self.device)
        self.width = (self.smear.width[0]).item()
        self.cutoff = 1.25 if n_atoms > 100 else 3.0 #heuristic
        self.cell = cell
        
    def forward(self, xyz):
        xyz = xyz.reshape(-1, self.n_atoms, 3)

        nbr_list, _ = generate_nbr_list(xyz, self.cutoff,
                                           self.cell, 
                                           self.bonds_mask, 
                                           get_dis=False)
        nbr_list = nbr_list.to("cpu")
        angle_list = generate_angle_list(nbr_list).to(self.device)
        cos_angles = compute_angle(xyz, angle_list, self.cell, N=self.n_atoms)  
        angles = cos_angles.acos() * 180/np.pi
        count = self.smear(angles.reshape(-1).squeeze()[..., None]).sum(0) 
        
        norm = count.sum()   # normalization factor for histogram 
        count = count / (norm)  # normalize 
        return count

class DifferentiableVelHist(torch.nn.Module):
    def __init__(self, params, device):
        super(DifferentiableVelHist, self).__init__()
        start = 0
        range =  5*params.temp #torch.max(self.running_dists)
        nbins = int(range/params.dv)

        #GPU
        self.device = device

        V, vol_bins, bins = generate_vol_bins(start, range, nbins, dim=3)

        self.V = V
        self.vol_bins = vol_bins.to(self.device)
        self.bins = bins

        self.smear = GaussianSmearing(
            start=start,
            stop=bins[-1],
            n_gaussians=nbins,
            width=params.gaussian_width,
            trainable=False
        ).to(self.device)

    def forward(self, vels):
        
        count = self.smear(vels.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        velhist =  count# / (self.vol_bins / self.V )  
        return 100*velhist

class DifferentiableVACF(torch.nn.Module):
    def __init__(self, params, device):
        super(DifferentiableVACF, self).__init__()
        self.device = device
        self.t_window = [i for i in range(1, params.vacf_window, 1)]

    def forward(self, vel):
        vacf = [torch.Tensor([1.]).to(self.device)]
        average_vel_sq = (vel * vel).mean() + 1e-6

        # can be implemented in parallel
        vacf += [((vel[t:] * vel[:-t]).mean() / average_vel_sq)[None] for t in self.t_window]
        return torch.stack(vacf).reshape(-1)

''''Explicitly BATCHED over replicas'''
def msd(positions, box):
    #Input: positions tensor of shape [N_timesteps, N_replicas, N_particles, 3]
    msd = torch.zeros(positions.shape[0], positions.shape[1])
    total_displacements = torch.zeros_like(positions[0])
    # Loop over time steps
    for step in range(1, positions.shape[0]):
        # Compute displacement vector for each particle
        displacements = positions[step] - positions[step - 1]
        displacements = torch.where(displacements > 0.5*box, displacements-box, torch.where(displacements<-0.5*box, displacements+box, displacements))
        total_displacements = total_displacements + displacements
        # Calculate squared displacements
        total_squared_displacements = torch.linalg.norm(total_displacements, axis=-1) ** 2

        # Accumulate squared displacements and update number of displacements
        msd[step] = total_squared_displacements.mean(axis = 1)

    # Optionally, calculate standard deviation for error estimates
    #std_msd = (msd.var() / len(msd)).sqrt()
    return msd


class DiffusionCoefficient(torch.nn.Module):
    def __init__(self, params, device):
        super(DiffusionCoefficient, self).__init__()
        self.dt = params.dt
        self.device = device
        
    def forward(self, msd_data):
        msd_data = msd_data.to(self.device)
        n = msd_data.shape[0] # Number of training examples.
        X = torch.linspace(0, self.dt*n, n)
        X = torch.stack([X, torch.ones((n,))], dim = 1).to(self.device) #append column of ones for bias term
        # Compute slope using Normal Equation
        X_T = torch.transpose(X, 0, 1)
        X_T_X = torch.matmul(X_T, X)
        X_T_X_inv = torch.inverse(X_T_X)
        X_T_y = torch.matmul(X_T, msd_data)
        theta = torch.matmul(X_T_X_inv, X_T_y)

        #Einstein relation
        return 1/6 * theta[0]

class SelfIntermediateScattering(torch.nn.Module):
    # Note: k_mag should be the distance in Angstroms of the first RDF peak
    def __init__(self, params, device, n_vectors=30):
        super(SelfIntermediateScattering, self).__init__()
        self.device = device
        self.k_mag = params.rdf_peak #distance in Angstroms of the first RDF peak
        # Generate random unit vectors
        random_vectors = torch.randn(n_vectors, 3, device=device)
        # Normalize the vectors to have unit length, then scale by k_mag
        self.k_vectors = (random_vectors / torch.norm(random_vectors, dim=1, keepdim=True)) * self.k_mag
    def forward(self, trajectory):
        # Dimensions: timestep x replica x n_atoms x 3
        trajectory = trajectory.to(self.device)
        dr = trajectory - trajectory[0, :, :, :].unsqueeze(0)
        dr_dot_k = torch.sum(dr[:, : , : , None, :] *  self.k_vectors[None, None, None, :, :], dim = -1)
        exponential_term = torch.exp(1j * dr_dot_k).real
        return torch.mean(exponential_term, dim=(1,2,3))


