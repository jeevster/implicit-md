#!/bin/bash

# Simulation Arguments:
# --system=$1
# --molecule=$2 
# --lr=$3 
# --rdf_loss_weight=$4 
# --vacf_loss_weight=$5 
# --energy_force_loss_weight=$6
# --eval_model=$7 
# --temperature=$8
# --integrator=$9
# --n_epochs=$10

##################### MD17 ############################
# Model checkpoint path: /global/cfs/projectdirs/m4319/sanjeevr/results_md17_correctedlogging/IMPLICIT_schnet_aspirin_TEST_correctedlogging_lr=0.001_efweight=10
# Simulation results are in the "inference/post" subfolder

system="md17"
molecule='aspirin'
ef_weight=10
size='1k'
lr=0.001

# Longer Nose Hoover sims for MD17 (StABlE Trained model)
# sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_nhlong' 500 'NoseHoover' 1000 # works

# # Longer Langevin sims for MD17 (pre and post StABlE training models)
# sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_langevinlong' 500 'Langevin' 1000 # works
# sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_langevinlong' 500 'Langevin' 1000 # works
sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight '10k_langevinlong' 500 'Langevin' 1000 # works
sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight '50k_langevinlong' 500 'Langevin' 1000 # works

# ##################### MD22 ############################
# # Model checkpoint path: /global/cfs/projectdirs/m4319/sanjeevr/results_md22_final_l=1_25percent/IMPLICIT_nequip_ac_Ala3_NHMe_FINAL_lr=0.001_efweight=10
# # Simulation results are in the "inference/post_cycle1" subfolder

# # Longer Langevin sims for Ac-Ala3 (pre and post)
# system="md22"
# molecule='ac_Ala3_NHMe'
# ef_weight=10
# size='25percent'
# lr=0.001

# sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_langevinlong' 500 'Langevin' 1000
# sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle1_langevinlong' 500 'Langevin' 1000

# ##################### Water ############################
# # Model checkpoint path: /global/cfs/projectdirs/m4319/sanjeevr/results_water_localneighborhoods/IMPLICIT_gemnet_t_water_TEST_bond_length_dev_singlemolecule_COPY_lr=0.003_efweight=0
# # Simulation results are in the "inference/post_cycle1" subfolder


# system="water"
# model="gemnet_t"
# ef_weight=0
# size='1k'
# lr=0.003

# # Longer Langevin sims for water (pre and post)
# sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_langevinlong' 300 'Langevin' 1000
# sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_langevinlong' 300 'Langevin' 1000


