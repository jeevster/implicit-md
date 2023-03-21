#!/bin/bash

python nve_implicit.py \
--n_particle 255 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 10 \
--epsilon 0.2 \
--poly \
--poly_power 3.0 \
--min_sigma 0.5 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--t_total 5 \
--diameter_viz 0.3 \
--n_dump 1 \
--save_intermediate_rdf \
--burn_in_frac 0.2 \
--n_epochs 30 \
--cutoff 1.5 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'