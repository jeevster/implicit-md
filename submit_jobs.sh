#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

# system="md17"
# dts=(0.005)
# molecules_md22=('aspirin') # 'salicylic_acid' 'uracil')
# lrs=(0.001) # 0.0003 0.001)
# ef_weights=(10) # 1) # 3 10 30 100)
# for dt in "${dts[@]}"; do
#     for molecule in "${molecules_md22[@]}"; do
#         for lr in "${lrs[@]}"; do
#             for ef_weight in "${ef_weights[@]}"; do
#                 # jid1=$(sbatch --parsable run_implicit.sh $system $molecule $lr 1 0 $ef_weight True $size) 
#                 # post inference - for every HP combination - can only start once training finishes
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_dt005' 0.005 10000 200000
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_dt016' 0.016 3000 62500
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_dt05' 0.05 1000 20000
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_dt16' 0.16 300 6250
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_dt5' 0.5 100 2000

#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_dt005' 0.005 10000 200000
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_dt016' 0.016 3000 62500
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_dt05' 0.05 1000 20000
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_dt16' 0.16 300 6250
#                 sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_dt5' 0.5 100 2000
#             done
#         done
#     done
    
# done

system="water"
model="gemnet_t"
lrs=(0.003)
ef_weights=(0) # 1 10) # 1) # 3 10 30 100)
for lr in "${lrs[@]}"; do
    for ef_weight in "${ef_weights[@]}"; do
        sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_dtp1' 0.1 1000 10000
        sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_dtp3' 0.3 333 3333
        # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_dt1' 1 100 1000
        # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dtp1' 0.1 1000 10000
        # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dtp3' 0.3 333 3333
        # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_dt1' 1 100 1000
    done
done
   
