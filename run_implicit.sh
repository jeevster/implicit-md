#!/bin/bash

ttotal_values=(0.5 1.0 2.0 3.5 5.0)
temp_values=(0.5 1.0 2.0)
box_values=(5 7 10)

pretrained_model_dir="/home/sanjeevr/implicit-md/results/IMPLICIT_ckpt_n=256_box=7_temp=1.0_eps=1.0_sigma=1.0_dt=0.005_ttotal="

for temp in "${temp_values[@]}"
do
    for box in "${box_values[@]}"
    do
        for ttotal in "${ttotal_values[@]}"
        do
            python nve_basic.py \
            --n_particle 256 \
            --temp "$temp" \
            --seed 123 \
            --kbt0 1.8 \
            --box "$box" \
            --epsilon 1.0 \
            --sigma 1.0 \
            --dt 0.005 \
            --dr 0.01 \
            --t_total 5.0 \
            --diameter_viz 0.3 \
            --n_dump 10 \
            --burn_in_frac 0.2 \
            --n_epochs 30 \
            --save_intermediate_rdf \
            --exp_name "generalization" \
            --nn \
            --inference \
            --pretrained_model_dir "${pretrained_model_dir}$ttotal" \
            --cutoff 2.5 \
            --gaussian_width 0.1 \
            --n_width 128 \
            --n_layers 3 \
            --nonlinear "ELU"
        done
    done
done
