import torch

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