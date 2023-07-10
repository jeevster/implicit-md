#!/bin/bash
#12 6 experiments
# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --sigma 1 \
# --prior_epsilon 1 \
# --prior_sigma 1 \
# --prior_rep_power 6 \
# --prior_attr_power 3 \
# --rep_power 9 \
# --attr_power 3 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --dt 0.005 \
# --dr 0.01 \
# --dv 0.1 \
# --exp_name "TEST_vacf+rdf_aggregategrads_bs10_9_3_potential_prior6_3" \
# --diffusion_window 1000 \
# --vacf_window 100 \
# --batch_size 10 \
# --lr 0.001 \
# --t_total 15 \
# --nvt_time 15 \
# --n_replicas 1 \
# --diameter_viz 0.3 \
# --n_dump 20 \
# --burn_in_frac 0.8 \
# --n_epochs 100 \
# --diffusion_loss_weight 0 \
# --vacf_loss_weight 1 \
# --rdf_loss_weight 1 \
# --nn \
# --cutoff 2.5 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'


python nve_implicit.py \
--n_particle 256 \
--temp 0.8 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 0.2 \
--sigma 1 \
--prior_epsilon 1 \
--prior_sigma 1 \
--prior_rep_power 6 \
--prior_attr_power 0 \
--rep_power 12 \
--attr_power 6 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--dt 0.005 \
--dr 0.01 \
--dv 0.1 \
--exp_name "TEST_sanity" \
--diffusion_window 1000 \
--vacf_window 100 \
--batch_size 32 \
--lr 0.001 \
--t_total 15 \
--nvt_time 15 \
--n_replicas 1 \
--diameter_viz 0.3 \
--n_dump 20 \
--burn_in_frac 0.8 \
--n_epochs 100 \
--diffusion_loss_weight 0 \
--vacf_loss_weight 1 \
--rdf_loss_weight 1 \
--nn \
--cutoff 2.5 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'



