import torch
import numpy as np
from itertools import product
from scipy.stats import maxwell
import math
import yaml
import os
import json
import gc
import matplotlib.pyplot as plt
from torchmd.observable import DifferentiableRDF, DifferentiableADF
from mdsim.common.utils import data_to_atoms
from mdsim.datasets.lmdb_dataset import LmdbDataset
from ase.neighborlist import natural_cutoffs, NeighborList
from mdsim.models.schnet import SchNetWrap
from mdsim.observables.md17_22 import get_hr
from mdsim.observables.water import get_water_rdfs
from mdsim.observables.common import get_smoothed_diffusivity
from torch.utils.tensorboard.summary import hparams

#calculate simulation metrics from stable parts of all trajectories
def calculate_final_metrics(simulator, params, device, results_dir, energy_maes, force_maes, gt_rdf, gt_adf=None, gt_vacf=None, gt_diffusivity=None, oxygen_atoms_mask=None, all_vacfs_per_replica=None):
    np.save(os.path.join(results_dir, f'replicas_stable_time.npy'), simulator.stable_time.cpu().numpy())
    full_traj = torch.stack(simulator.all_radii)
    np.save(os.path.join(results_dir, 'full_traj.npy'), full_traj)
    hyperparameters = {
        'lr': params.lr,
        'ef_loss_weight': params.energy_force_loss_weight
    }
    steps_per_epoch = int(simulator.nsteps/simulator.n_dump)
    stable_steps = simulator.stable_time * steps_per_epoch
    stable_trajs = [full_traj[:int(upper_step_limit), i] for i, upper_step_limit in enumerate(stable_steps)]
    stable_trajs_stacked = torch.cat(stable_trajs)
    xlim = params.max_rdf_dist
    n_bins = int(xlim/params.dr)
    bins = np.linspace(1e-6, xlim, n_bins + 1) # for computing h(r)
    if params.name == 'md17' or params.name == 'md22':
        final_rdfs = torch.stack([torch.Tensor(get_hr(traj, bins)).to(device) for traj in stable_trajs])
        gt_rdf = final_rdfs[0].sum()*gt_rdf/ gt_rdf.sum() #normalize to be on the same scale
        final_rdf_maes = xlim * torch.abs(gt_rdf.unsqueeze(0) -final_rdfs).mean(-1)
        adf = DifferentiableADF(simulator.n_atoms, simulator.bonds, simulator.cell, params, device)
        final_adfs = torch.stack([adf(traj[::5].to(device)) for traj in stable_trajs])
        final_adf_maes = torch.abs(gt_adf.unsqueeze(0) - final_adfs).mean(-1)
        count_per_replica = torch.stack(all_vacfs_per_replica).sum(0)[:, 0].unsqueeze(-1)+1e-8
        final_vacfs = (torch.stack(all_vacfs_per_replica).sum(0) /count_per_replica).to(device)
        final_vacf_maes = torch.abs(gt_vacf.unsqueeze(0) - final_vacfs).mean(-1)
        final_metrics = {
            'Energy MAE': energy_maes[-1],
            'Force MAE': force_maes[-1],
            'Mean Stability (ps)': simulator.stable_time.mean().item(),
            'Std Dev Stability (ps)': simulator.stable_time.std().item(),
            'Mean RDF MAE': final_rdf_maes.mean().item(),
            'Mean ADF MAE': final_adf_maes.mean().item(),
            'Mean VACF MAE': final_vacf_maes.mean().item(),
            'Std Dev RDF MAE': final_rdf_maes.std().item(),
            'Std Dev ADF MAE': final_adf_maes.std().item(),
            'Std Dev VACF MAE': final_vacf_maes.std().item()
        }
        #save rdfs, adfs, and vacfs at the end of the trajectory
        np.save(os.path.join(results_dir, "final_rdfs.npy"), final_rdfs.cpu())
        np.save(os.path.join(results_dir, "final_adfs.npy"), final_adfs.cpu())
        np.save(os.path.join(results_dir, "final_vacfs.npy"), final_vacfs.cpu())
        np.save(os.path.join(results_dir, "final_rdf_maes.npy"), final_rdf_maes.cpu())
        np.save(os.path.join(results_dir, "final_adf_maes.npy"), final_adf_maes.cpu())
        np.save(os.path.join(results_dir, "final_vacf_maes.npy"), final_vacf_maes.cpu())

    elif params.name == "water":
        final_rdfs = [get_water_rdfs(traj, simulator.rdf_mae.ptypes, simulator.rdf_mae.lattices, simulator.rdf_mae.bins, device)[0] for traj in stable_trajs]
        final_rdfs_by_key = {k: torch.stack([final_rdf[k] for final_rdf in final_rdfs]) for k in gt_rdf.keys()}
        final_rdf_maes = {k: xlim* torch.abs(gt_rdf[k] - final_rdfs_by_key[k]).mean(-1).squeeze(-1) for k in gt_rdf.keys()}
        #Recording frequency is 1 ps for diffusion coefficient
        all_diffusivities = [get_smoothed_diffusivity(traj[::int(1000/params.n_dump), oxygen_atoms_mask])[0] for traj in stable_trajs]
        all_msds = [get_smoothed_diffusivity(traj[::int(1000/params.n_dump), oxygen_atoms_mask])[1] for traj in stable_trajs]
        last_diffusivities = torch.cat([diff[-1].unsqueeze(-1) if len(diff) > 0 else torch.Tensor([0.]) for diff in all_diffusivities])
        diffusivity_maes = 10*(gt_diffusivity[-1].to(device) - last_diffusivities.to(device)).abs()
        
        #save full diffusivity trajectory
        all_diffusivities = [diff.cpu() for diff in all_diffusivities]
        np.save(os.path.join(results_dir, "all_diffusivities.npy"), np.array(all_diffusivities, dtype=object), allow_pickle=True)
        np.save(os.path.join(results_dir, "all_msds.npy"), np.array(all_msds, dtype=object), allow_pickle=True)

        #TODO: compute O-O-conditioned ADF instead of full ADF
        final_adfs = torch.stack([DifferentiableADF(simulator.n_atoms, simulator.bonds, simulator.cell, params, device)(traj[::2].to(device)) for traj in stable_trajs])
        final_adf_maes = torch.abs(gt_adf-final_adfs).mean(-1)
        final_metrics = {
            'Energy MAE': energy_maes[-1],
            'Force MAE': force_maes[-1],
            'Mean Stability (ps)': simulator.stable_time.median().item(),
            'Std Dev Stability (ps)': simulator.stable_time.std().item(),
            'Mean OO RDF MAE': final_rdf_maes['OO'].mean().item(),
            'Mean HO RDF MAE': final_rdf_maes['HO'].mean().item(),
            'Mean HH RDF MAE': final_rdf_maes['HH'].mean().item(),
            'Std Dev OO RDF MAE': final_rdf_maes['OO'].std().item(),
            'Std Dev HO RDF MAE': final_rdf_maes['HO'].std().item(),
            'Std Dev HH RDF MAE': final_rdf_maes['HH'].std().item(),
            'Mean ADF MAE': final_adf_maes.mean().item(),
            'Std Dev ADF MAE': final_adf_maes.std().item(),
            'Mean Diffusivity MAE (10^-9 m^2/s)': diffusivity_maes.mean().item(),
            'Std Dev Diffusivity MAE (10^-9 m^2/s)': diffusivity_maes.std().item()
        }
        #save rdf, adf, and diffusivity at the end of the traj
        for key, final_rdfs in final_rdfs_by_key.items():
            np.save(os.path.join(results_dir, f"final_{key}_rdfs.npy"), final_rdfs.squeeze(1).cpu())
            np.save(os.path.join(results_dir, f"final_{key}_rdf_maes.npy"), final_rdf_maes[key].cpu())
        np.save(os.path.join(results_dir, "final_adfs.npy"), final_adfs.cpu().detach().numpy())
        np.save(os.path.join(results_dir, "final_diffusivities.npy"), last_diffusivities.cpu().detach().numpy())

    #save final metrics to JSON
    with open(os.path.join(results_dir, 'final_metrics.json'), 'w') as fp:
        json.dump(final_metrics, fp, indent=4, separators=(',', ': '))
    return hparams(hyperparameters, final_metrics)
