#!/bin/bash

python nve_implicit.py \
--n_particle 1024 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 15 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 5.0 \
--diameter_viz 0.3 \
--n_dump 1 \
--burn_in_frac 0.2 \
--n_epochs 30 \
--exp_name "TEST" \
--cutoff 10 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear "ELU"
        