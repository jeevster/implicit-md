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
--exp_name "TEST_fabianmethod_longer_eqtime" \
--diffusion_window 1000 \
--vacf_window 100 \
--batch_size 1 \
--lr 0.001 \
--t_total 3 \
--nvt_time 3 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 500 \
--burn_in_frac 0.8 \
--n_epochs 100 \
--diffusion_loss_weight 0 \
--vacf_loss_weight 1 \
--rdf_loss_weight 1 \
--nn \
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'