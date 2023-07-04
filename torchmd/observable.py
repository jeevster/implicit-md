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

'''Sanjeev's DiffRDF implementation - doesn't use system stuff'''
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
        # nbr_list, pair_dis, _ = generate_nbr_list(xyz, 
        #                                        self.cutoff_boundary, 
        #                                        self.cell, 
        #                                        index_tuple=self.index_tuple, 
        #                                        get_dis=True)
        running_dists = torch.cat(running_dists)
        
        count = self.smear(running_dists.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        # gr =  count / (self.vol_bins / self.V)  
        # return gr #to match with MD17 RDF computation
        return 100*count

'''Sanjeev's differentiable velocity histogram implementation - doesn't use system stuff'''
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

'''Sanjeev's differentiable VACF implementation - doesn't use system stuff'''
class DifferentiableVACF(torch.nn.Module):
    def __init__(self, params):
        super(DifferentiableVACF, self).__init__()
        self.t_window = [i for i in range(1, params.vacf_window, 1)]

    def forward(self, vel):
        vacf = [(vel * vel).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in self.t_window]

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


'''Sanjeev's Diffusion Coefficient implementation - doesn't use system stuff'''
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