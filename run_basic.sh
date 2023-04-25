#!/bin/bash
python nve_implicit.py \
--n_particle 512 \
--temp 0.2 \
--seed 123 \
--kbt0 1.8 \
--box 15 \
--epsilon 0.2 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.001 \
--dr 0.01 \
--exp_name "rdf_longconvergence_lr0.001_withdiffusion" \
--diffusion_window 1000 \
--lr 0.001 \
--t_total 9 \
--nvt_time 9 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 100 \
--burn_in_frac 0.8 \
--n_epochs 50 \
--diffusion_loss_weight 100 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

python nve_implicit.py \
--n_particle 512 \
--temp 0.2 \
--seed 123 \
--kbt0 1.8 \
--box 15 \
--epsilon 0.2 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.001 \
--dr 0.01 \
--exp_name "rdf_longconvergence_lr0.005_withdiffusion" \
--diffusion_window 1000 \
--lr 0.005 \
--t_total 9 \
--nvt_time 9 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 100 \
--burn_in_frac 0.8 \
--n_epochs 50 \
--diffusion_loss_weight 100 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

python nve_implicit.py \
--n_particle 512 \
--temp 0.2 \
--seed 123 \
--kbt0 1.8 \
--box 15 \
--epsilon 0.2 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.001 \
--dr 0.01 \
--exp_name "rdf_longconvergence_lr0.01_withdiffusion" \
--diffusion_window 1000 \
--lr 0.01 \
--t_total 9 \
--nvt_time 9 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 100 \
--burn_in_frac 0.8 \
--n_epochs 50 \
--diffusion_loss_weight 100 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'












