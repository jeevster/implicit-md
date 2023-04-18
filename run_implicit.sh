#!/bin/bash

python nve_implicit.py \
--n_particle 256 \
--temp 0.8 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 0.2 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--exp_name "test_latest_code_retainforcegrad" \
--diffusion_window 1000 \
--t_total 15 \
--nvt_time 15 \
--diameter_viz 0.3 \
--n_dump 20 \
--burn_in_frac 0.9 \
--n_epochs 50 \
--diffusion_loss_weight 0 \
--rdf_loss_weight 1 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

# python nve_implicit.py \
# --n_particle 1024 \
# --temp 0.2 \
# --seed 123 \
# --kbt0 1.8 \
# --box 15 \
# --epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.001 \
# --dr 0.01 \
# --exp_name "rdfonly_longconverge" \
# --diffusion_window 100 \
# --t_total 10 \
# --nvt_time 10 \
# --diameter_viz 0.3 \
# --n_dump 200 \
# --burn_in_frac 0.9 \
# --n_epochs 50 \
# --diffusion_loss_weight 0 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'


# python nve_implicit.py \
# --n_particle 1024 \
# --temp 0.2 \
# --seed 123 \
# --kbt0 1.8 \
# --box 15 \
# --epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.001 \
# --dr 0.01 \
# --exp_name "rdfonly_longconverge" \
# --diffusion_window 100 \
# --t_total 20 \
# --nvt_time 20 \
# --diameter_viz 0.3 \
# --n_dump 200 \
# --burn_in_frac 0.9 \
# --n_epochs 50 \
# --diffusion_loss_weight 0 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'
