import torch
import numpy as np
from itertools import product
from scipy.stats import maxwell
import math


def radii_to_dists(radii, box_size):
    #Get rij matrix
    r = radii.unsqueeze(0) - radii.unsqueeze(1)
    
    #Enforce minimum image convention
    r = -1*torch.where(r > 0.5*box_size, r-box_size, torch.where(r<-0.5*box_size, r+box_size, r))

    #get rid of diagonal 0 entries of r matrix (for gradient stability )
    r = r[~torch.eye(r.shape[0],dtype=bool)].reshape(r.shape[0], -1, 3)
    try:
        r.requires_grad = True
    except RuntimeError:
        pass

    #compute distance matrix:
    return torch.sqrt(torch.sum(r**2, axis=2)).unsqueeze(-1)


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
def initialize_velocities(n_particle, temp):
    
    vel_dist = maxwell()
    velocities = vel_dist.rvs(size = (n_particle, 3))
    #shift so that initial momentum is zero
    velocities -= np.mean(velocities, axis = 0)

    #scale velocities to match desired temperature
    sum_vsq = np.sum(np.square(velocities))
    p_dof = 3*(n_particle-1)
    correction_factor = math.sqrt(p_dof*temp/sum_vsq)
    velocities *= correction_factor
    return torch.Tensor(velocities)