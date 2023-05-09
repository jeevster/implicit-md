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
--dv 0.1 \
--exp_name "TEST_vacf" \
--diffusion_window 1000 \
--vacf_window 100 \
--lr 0.001 \
--t_total 2 \
--nvt_time 2 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 20 \
--burn_in_frac 0.8 \
--n_epochs 50 \
--diffusion_loss_weight 0 \
--vacf_loss_weight 100 \
--rdf_loss_weight 1 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'


# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --exp_name "newtest_replicas4" \
# --diffusion_window 1000 \
# --lr 0.001 \
# --t_total 15 \
# --nvt_time 15 \
# --n_replicas 4 \
# --diameter_viz 0.3 \
# --n_dump 20 \
# --burn_in_frac 0.8 \
# --n_epochs 50 \
# --diffusion_loss_weight 0 \
# --rdf_loss_weight 1 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'


# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --exp_name "newtest_replicas8" \
# --diffusion_window 1000 \
# --lr 0.001 \
# --t_total 15 \
# --nvt_time 15 \
# --n_replicas 8 \
# --diameter_viz 0.3 \
# --n_dump 20 \
# --burn_in_frac 0.8 \
# --n_epochs 50 \
# --diffusion_loss_weight 0 \
# --rdf_loss_weight 1 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'

# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --exp_name "newtest_replicas16" \
# --diffusion_window 1000 \
# --lr 0.001 \
# --t_total 15 \
# --nvt_time 15 \
# --n_replicas 16 \
# --diameter_viz 0.3 \
# --n_dump 20 \
# --burn_in_frac 0.8 \
# --n_epochs 50 \
# --diffusion_loss_weight 0 \
# --rdf_loss_weight 1 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'