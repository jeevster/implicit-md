from itertools import product
import concurrent.futures
from pypapi import events, papi_high as high
import numpy as np
import gsd.hoomd
import torch
import math
import torch.nn as nn
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from torch.nn.utils.rnn import pad_sequence
from YParams import YParams
import argparse
import threading
import os
from tqdm import tqdm
import pstats
import pdb
import random
from contextlib import nullcontext
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from torchmd.observable import rdf, generate_vol_bins, DifferentiableRDF, msd, DiffusionCoefficient
import torchopt
from torchopt.nn import ImplicitMetaGradientModule
import time
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import shutil
from torch.utils.tensorboard import SummaryWriter
from functorch import vmap
from utils import radii_to_dists, fcc_positions, initialize_velocities, dump_params_to_yml, powerlaw_inv_cdf


'''CORE MD OPERATIONS'''
def serial_force_calc(sim, radii, retain_grad = False):
    
    #Get rij matrix
    with torch.enable_grad():
        r = radii.unsqueeze(-3) - radii.unsqueeze(-2)
        if not r.requires_grad:
            r.requires_grad = True

        #Enforce minimum image convention
        r = -1*torch.where(r > 0.5*sim.box, r-sim.box, torch.where(r<-0.5*sim.box, r+sim.box, r))

        #get rid of diagonal 0 entries of r matrix (for gradient stability)
        r = r[~torch.eye(r.shape[0],dtype=bool)].reshape(r.shape[0], -1, 3)
        
        #compute distance matrix:
        dists = torch.sqrt(torch.sum(r**2, axis=-1)).unsqueeze(-1)

        #compute energy
        if sim.nn:
            energy = sim.model((dists/sim.sigma_pairs)) if sim.poly else sim.model(dists)
            forces = -compute_grad(inputs=r, output=energy) if retain_grad else -compute_grad(inputs=r, output=energy).detach()
            
        #LJ potential
        else:
            if sim.poly: #repulsive
                parenth = (sim.sigma_pairs/dists)
                # r2i = (1/dists)**2
                
                rep_term = parenth ** sim.rep_power
                # attr_term = parenth ** sim.attr_power
                # r_multiplier = r2i * (sim.rep_power * rep_term - \
                #                 sim.attr_power * attr_term) - 2*c_2/(sim.sigma_pairs**2)

                # r3_multiplier = 4*c_4 / (sim.sigma_pairs**4)
                
                energy = torch.sum(rep_term + sim.c_0 + sim.c_2*parenth**-2 + sim.c_4*parenth**-4)
                
                forces = -compute_grad(inputs=r, output=energy) if retain_grad else -compute_grad(inputs=r, output=energy).detach()
            
            else:
                r2i = (sim.sigma/dists)**2
                r6i = r2i**3
                energy = 2*sim.epsilon*torch.sum(r6i*(r6i - 1))
                #reuse components of potential to calculate virial and forces
                internal_virial = -48*sim.epsilon*r6i*(r6i - 0.5)/(sim.sigma**2)
                forces = -internal_virial*r*r2i

            #calculate which forces to keep
            keep = dists/sim.sigma_pairs <= sim.cutoff if sim.poly else dists/sim.sigma <= sim.cutoff

            #apply cutoff
            forces = torch.where(~keep, sim.zeros, forces)

                
    # #Ensure no NaNs
    assert(not torch.any(torch.isnan(forces)))
    
    #sum forces across particles
    return energy, torch.sum(forces, axis = -2)#.to(sim.device)


def serial_forward_nvt(sim, radii, velocities, forces, zeta, calc_rdf = False, calc_diffusion = False, calc_vacf = False, retain_grad = False):
        # get current accelerations (assume unit mass)
        accel = forces

        # make full step in position 
        radii = radii + velocities * sim.dt + \
            (accel - zeta * velocities) * (0.5 * sim.dt ** 2)

        #PBC correction
        if sim.pbc:
            radii = radii/sim.box 
            radii = sim.box*torch.where(radii-torch.round(radii) >= 0, \
                        (radii-torch.round(radii)), (radii - torch.floor(radii)-1))

        # record current velocities
        KE_0 = torch.sum(torch.square(velocities)) / 2
        
        # make half a step in velocity
        velocities = velocities + 0.5 * sim.dt * (accel - zeta * velocities)

        # make a full step in accelerations
        energy, forces = serial_force_calc(sim, radii.to(sim.device), retain_grad=retain_grad)
        accel = forces

        # make a half step in sim.zeta
        zeta = zeta + 0.5 * sim.dt * (1/sim.Q) * (KE_0 - sim.targeEkin)

        #get updated KE
        ke = torch.sum(torch.square(velocities)) / 2

        # make another halfstep in sim.zeta
        zeta = zeta + 0.5 * sim.dt * \
            (1/sim.Q) * (ke - sim.targeEkin)

        # make another half step in velocity
        velocities = (velocities + 0.5 * sim.dt * accel) / \
            (1 + 0.5 * sim.dt * zeta)

        if calc_rdf:
            new_dists = radii_to_dists(radii, sim.params)
            if sim.poly:
                #normalize distances by sigma pairs
                new_dists = new_dists / sim.sigma_pairs

        new_rdf = sim.diff_rdf(tuple(new_dists.to(sim.device))) if calc_rdf else 0 #calculate the RDF from a single frame
        #new_rdf = 0

        if calc_diffusion:
            sim.last_h_radii.append(radii.unsqueeze(0))
        if calc_vacf:
            sim.last_h_velocities.append(velocities.unsqueeze(0))

        # dump frames
        if sim.step%sim.n_dump == 0:
            print(sim.step, sim.calc_properties(energy), file=sim.f)
            sim.t.append(sim.create_frame(frame = sim.step/sim.n_dump))
            #append dists to running_dists for RDF calculation (remove diagonal entries)
            if not sim.nn:
                new_dists = radii_to_dists(radii, sim.params)
                if sim.poly:
                    #normalize distances by sigma pairs
                    new_dists = new_dists / sim.sigma_pairs
                sim.running_dists.append(new_dists.cpu().detach())
                sim.running_vels.append(torch.linalg.norm(velocities, dim = -1).cpu().detach())


        return radii, velocities, forces, zeta, new_rdf