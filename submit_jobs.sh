#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

system="md22"
size="25percent"
molecules_md22=('ac_Ala3_NHMe')
models=('gemnet_t')
lrs=(0.00003) # 0.0003 0.001)
ef_weights=(100) # 1) # 3 10 30 100)
for molecule in "${molecules_md22[@]}"; do
    for lr in "${lrs[@]}"; do
        for ef_weight in "${ef_weights[@]}"; do
            jid1=$(sbatch --parsable run_implicit.sh $system $molecule $lr 1 0 $ef_weight True $size) 
            # post inference - for every HP combination - can only start once training finishes
            sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post' $size
            sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_epoch10' $size
            sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_epoch20' $size
            sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle1' $size
            # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle2'
            # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle3'
        done
    done
    #non-post inferences - once per model (can start after training starts)
    lrtemp=0.00003
    ef_weighttemp=100
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp 'pre'
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '10percent'
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '25percent'
done



