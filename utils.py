import torch
import numpy as np
from itertools import product
from scipy.stats import maxwell
import math
import yaml
import os
import gc
import matplotlib.pyplot as plt


def radii_to_dists(radii, params):
    #Get rij matrix
    r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
    
    #Enforce minimum image convention
    r = -1*torch.where(r > 0.5*params.box, r-params.box, torch.where(r<-0.5*params.box, r+params.box, r))

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
def initialize_velocities(n_particle, temp, n_replicas):
    
    vel_dist = maxwell()
    velocities = vel_dist.rvs(size = (n_replicas, n_particle, 3))
    #shift so that initial momentum is zero
    velocities -= np.mean(velocities, axis = 1, keepdims=True)

    #scale velocities to match desired temperature
    sum_vsq = np.sum(np.square(velocities), axis = (1,2), keepdims=True)
    p_dof = 3*(n_particle-1)
    correction_factor = np.sqrt(p_dof*temp/sum_vsq)
    velocities *= correction_factor
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
                print(type(obj), obj.size())
                count +=1
                del obj
        except:
            pass
    print(f"{count} tensors in memory")

'''Helper function to compare true underlying potential to the learned one'''
def plot_pair(epoch, path, model, device, end, target_pot): 

    x = torch.linspace(0.1, end, 250)[:, None].to(device)
    u_fit = torch.Tensor([model(i) for i in x]).detach().cpu().numpy()
    u_target = torch.Tensor([target_pot(i) for i in x]).detach().cpu().numpy()

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)
    
    plt.plot( x.detach().cpu().numpy(), 
              u_target,
               label='truth', 
               linewidth=2,linestyle='--', c='black')
    plt.ylim(-6.0, 6.0)
    plt.legend()      
    plt.show()
    plt.savefig(os.path.join(path, 'potential_{}.jpg'.format(epoch)), bbox_inches='tight')
    plt.close()

