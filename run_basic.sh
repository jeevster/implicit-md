#!/bin/bash

python nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 0.5 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--nn \
--exp_name "cutoff10" \
--n_epochs 30 \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

python nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 1.0 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--nn \
--exp_name "cutoff10" \
--n_epochs 30 \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

python nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 2.0 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--nn \
--exp_name "cutoff10" \
--n_epochs 30 \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'


python nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 3.5 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--nn \
--exp_name "cutoff10" \
--n_epochs 30 \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

python nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 5.0 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--nn \
--exp_name "cutoff10" \
--n_epochs 30 \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'