#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

# system="md17"
# molecules_md22=('aspirin') # 'salicylic_acid' 'uracil')
# lrs=(0.001) # 0.0003 0.001)
# ef_weights=(10) # 1) # 3 10 30 100)
# size='1k'

# for molecule in "${molecules_md22[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for ef_weight in "${ef_weights[@]}"; do
#             jid1=$(sbatch --parsable run_implicit.sh $system $molecule $lr 1 0 $ef_weight True $size) 
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post' 500
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre' 500
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_temp350' 350
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_temp350' 350
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_temp700' 700
#             sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_temp700' 700
#         done
#     done
# done

# system="water"
# model="gemnet_t"
# lrs=(0.001)
# ef_weights=(0) # 1 10) # 1) # 3 10 30 100)
# for lr in "${lrs[@]}"; do
#     for ef_weight in "${ef_weights[@]}"; do
#         sbatch run_implicit_simulate.sh $system $model 0.001 1 0 $ef_weight 'pre'
#         sbatch run_implicit_simulate.sh $system $model 0.001 1 0 $ef_weight 'post_cycle1'
#         # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_dt1' 1 100 1000
#         # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dtp1' 0.1 1000 10000
#         # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dtp3' 0.3 333 3333
#         # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dt1' 1 100 1000
#     done
# done
   
