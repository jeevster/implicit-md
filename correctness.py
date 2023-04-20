import sys
import gsd.hoomd
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF, msd, DiffusionCoefficient
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":


    if len(sys.argv) != 2:
        print("Usage: python program_name.py dir")
        sys.exit(1)

    dir = sys.argv[1]

    f_true = gsd.hoomd.open(name=f'{dir}/sim_TRUE_temp0.8.gsd', mode='rb')
    f = gsd.hoomd.open(name=f'{dir}/sim_temp0.8.gsd', mode='rb')


    pos_true = torch.cat([torch.tensor(frame.particles.position).unsqueeze(0) for frame in f_true], dim = 0)
    pos = torch.cat([torch.tensor(frame.particles.position).unsqueeze(0) for frame in f], dim = 0)

    vel_true = torch.cat([torch.tensor(frame.particles.velocity).unsqueeze(0) for frame in f_true], dim = 0)
    vel = torch.cat([torch.tensor(frame.particles.velocity).unsqueeze(0) for frame in f], dim = 0)

    

   

    for i in tqdm(range(pos_true.shape[0])):

        try:
            assert(torch.all(pos_true[i] == pos[i]))  
        except:
            print(f"Position {i+1} failed")
            break
        
        try:
            assert(torch.all(vel_true[i] == vel[i])) 
        except:
            print(f"Velocity {i+1} failed")
            break

        