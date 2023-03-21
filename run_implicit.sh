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
--t_total 1.1 \
--diameter_viz 0.3 \
--save_intermediate_rdf \
--n_dump 50 \
--exp_name 'profile' \
--burn_in_frac 0.2 \
--n_epochs 50 \
--nn \
--cutoff 2.5 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

