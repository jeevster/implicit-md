#!/bin/bash
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
# --exp_name "with_diffusion_givepolyinfo" \
# --diffusion_window 5000 \
# --t_total 3 \
# --nvt_time 3 \
# --diameter_viz 0.3 \
# --n_dump 100 \
# --burn_in_frac 0.8 \
# --n_epochs 100 \
# --diffusion_loss_weight 100 \
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
# --exp_name "with_diffusion_givepolyinfo" \
# --diffusion_window 5000 \
# --t_total 5 \
# --nvt_time 5 \
# --diameter_viz 0.3 \
# --n_dump 100 \
# --burn_in_frac 0.8 \
# --n_epochs 100 \
# --diffusion_loss_weight 100 \
# --nn \
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'

python -m cProfile -o out.prof old_nve_implicit.py \
--n_particle 256 \
--temp 0.8 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 0.2 \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--exp_name "test_parallel" \
--diffusion_window 10 \
--t_total 1 \
--nvt_time 1 \
--diameter_viz 0.3 \
--n_dump 1 \
--burn_in_frac 0.9 \
--n_epochs 1 \
--diffusion_loss_weight 1 \
--rdf_loss_weight 0 \
--cutoff 7 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU' \
--num_threads 1


# python -m cProfile -o out.prof old_nve_implicit.py \
# --n_particle 8 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 2 \
# --epsilon 0.2 \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --exp_name "test_parallel" \
# --diffusion_window 10 \
# --t_total 1 \
# --nvt_time 1 \
# --diameter_viz 0.3 \
# --n_dump 1 \
# --burn_in_frac 0.9 \
# --n_epochs 1 \
# --diffusion_loss_weight 1 \
# --rdf_loss_weight 0 \
# --cutoff 2 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU' \
# --num_threads 1
