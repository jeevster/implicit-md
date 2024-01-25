#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

system="md17"
sizes=("1k")
molecules_md22=('aspirin' 'salicylic_acid' 'uracil')
lrs=(0.001) # 0.0003 0.001)
ef_weights=(10) # 1) # 3 10 30 100)
for size in "${sizes[@]}"; do
    for molecule in "${molecules_md22[@]}"; do
        for lr in "${lrs[@]}"; do
            for ef_weight in "${ef_weights[@]}"; do
                x=0
                # jid1=$(sbatch --parsable run_implicit.sh $system $molecule $lr 1 0 $ef_weight True $size) 
                # post inference - for every HP combination - can only start once training finishes
                # sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_temp250' $size 250
                # sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_temp250' $size 250
                # sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'pre_temp700' $size 700
                # sbatch run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_temp700' $size 700
                # sbatch run_implicit.sh $system $molecule $lr 1 0 $ef_weight True $size
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle1_epoch10' $size
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle1_epoch20' $size
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle1' $size
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle2' $size
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle2'
                # sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post_cycle3'
            done
        done
    done
    #non-post inferences - once per model (can start after training starts)
    # lrtemp=0.001
    # ef_weighttemp=10
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp 'pre' $size
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '10percent'
    # sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '25percent'
done

system="water"
model="gemnet_t"
lrs=(0.003)
ef_weights=(0) # 1 10) # 1) # 3 10 30 100)
for lr in "${lrs[@]}"; do
    for ef_weight in "${ef_weights[@]}"; do
        # jid1=$(sbatch --parsable run_implicit.sh $system $model $lr 1 0 $ef_weight True 1k)
        sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'pre_copy2'
        # sbatch run_implicit_simulate.sh $system $model $lr 1 0 $ef_weight 'post_cycle1_copy2'
    done
done
   
