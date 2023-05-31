#!/bin/bash
python nve_implicit.py \
--n_particle 256 \
--temp 0.8 \
--seed 123 \
--kbt0 1.8 \
--box 7 \
--epsilon 0.2 \
--prior_rep_power 6 \
--prior_attr_power 0 \
--poly \
--poly_power 3.0 \
--min_sigma 0.73 \
--sigma 1.0 \
--dt 0.005 \
--dr 0.01 \
--dv 0.1 \
--exp_name "TEST" \
--diffusion_window 1000 \
--vacf_window 100 \
--batch_size 10 \
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
--cutoff 1.25 \
--gaussian_width 0.1 \
--n_width 128 \
--n_layers 3 \
--nonlinear 'ELU'

# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --prior_rep_power 12 \
# --prior_attr_power 6 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --dv 0.1 \
# --exp_name "TEST_vacf+rdf_aggregategrads_bs10_plotpair_prior12_6" \
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
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'

# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --prior_rep_power 9 \
# --prior_attr_power 6 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --dv 0.1 \
# --exp_name "TEST_vacf+rdf_aggregategrads_bs10_plotpair_prior9_6" \
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
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'


# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --prior_rep_power 3 \
# --prior_attr_power 0 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --dv 0.1 \
# --exp_name "TEST_vacf+rdf_aggregategrads_bs10_plotpair_prior3_0" \
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
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'

# python nve_implicit.py \
# --n_particle 256 \
# --temp 0.8 \
# --seed 123 \
# --kbt0 1.8 \
# --box 7 \
# --epsilon 0.2 \
# --prior_rep_power 6 \
# --prior_attr_power 0 \
# --prior_epsilon 0.2 \
# --poly \
# --poly_power 3.0 \
# --min_sigma 0.73 \
# --sigma 1.0 \
# --dt 0.005 \
# --dr 0.01 \
# --dv 0.1 \
# --exp_name "TEST_vacf+rdf_aggregategrads_bs10_plotpair_prior6_0_oldeps" \
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
# --cutoff 1.25 \
# --gaussian_width 0.1 \
# --n_width 128 \
# --n_layers 3 \
# --nonlinear 'ELU'