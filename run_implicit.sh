#!/bin/bash

python nve_implicit.py \
--n_particle 1024 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.0005 \
--dr 0.01 \
--t_total 5 \
--save_intermediate_rdf \
--diameter_viz 0.3 \
--n_dump 50 \
--burn_in_frac 0.2 \
--n_epochs 30 \
--cutoff 2.5 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'



