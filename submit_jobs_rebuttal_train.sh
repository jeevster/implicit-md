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


# Training arguments:
# --system=$1
# --model=$2 
# --lr=$3 
# --rdf_loss_weight=$4 
# --vacf_loss_weight=$5 
# --energy_force_loss_weight=$6 
# --only_learn_if_unstable_threshold_reached=$7 
# --size=$8
# --integrator=$9

#Training water with NPT

system="water"
model="gemnet_t"
ef_weight=0
size='1k'
lr=0.003

# Normal NVT Training (with PBC Fix)
jid1=$(sbatch --parsable run_implicit.sh $system $model $lr 1 0 $ef_weight True $size 'NoseHoover') 
sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre' 300 'NoseHoover' 100
sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1' 300 'NoseHoover' 100


# # NPT Training
# jid1=$(sbatch --parsable run_implicit.sh $system $model $lr 1 0 $ef_weight True $size 'Berendsen') 
# sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre' 300 'Berendsen' 100
# sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1' 300 'Berendsen' 100