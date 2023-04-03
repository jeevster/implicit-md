#!/bin/bash
ttotal_values=(0.5 1.0 2.0 3.5 5.0)

for ttotal in "${ttotal_values[@]}"
do
    python nve_implicit.py \
    --n_particle 256 \
    --temp 1 \
    --seed 123 \
    --kbt0 1.8 \
    --box 7 \
    --epsilon 1.0 \
    --sigma 1.0 \
    --dt 0.005 \
    --dr 0.01 \
    --t_total "$ttotal" \
    --diameter_viz 0.3 \
    --n_dump 10 \
    --burn_in_frac 0.2 \
    --nn \
    --exp_name "ckpt_cutoff10" \
    --n_epochs 30 \
    --cutoff 10 \
    --gaussian_width 0.1 \
    --n_width 128 \
    --n_layers 3 \
    --nonlinear 'ELU'



