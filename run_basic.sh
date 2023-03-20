#!/bin/bash

python -m cProfile -o out.prof nve_basic.py \
--n_particle 256 \
--temp 1 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 1.0 \
--sigma 1.0 \
--dt 0.001 \
--dr 0.01 \
--t_total 0.2 \
--diameter_viz 0.3 \
--n_dump 10 \
--burn_in_frac 0.2 \
--n_epochs 1 \
--nn \
--cutoff 2.5 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'