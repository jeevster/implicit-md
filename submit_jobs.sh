#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

system="md17"
molecules_md17=('aspirin') # 'naphthalene' 'salicylic_acid')
models=('schnet') #  'forcenet')
lrs=(0.00001 0.00003 0.0001 0.0003 0.001)
ef_weights=(1 3 10 30 100)
# Always learn experiments
for molecule in "${molecules_md17[@]}"; do
    for lr in "${lrs[@]}"; do
        for ef_weight in "${ef_weights[@]}"; do
            jid1=$(sbatch --parsable run_implicit.sh $system $molecule $lr 1 0 $ef_weight False)
            # post inference - for every HP combination - can only start once training finishes
            sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh $system $molecule $lr 1 0 $ef_weight 'post'
        done
    done
    #non-post inferences - once per model (can start after training starts)
    lrtemp=0.00001
    ef_weighttemp=1
    sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp 'pre'
    sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '10k'
    sbatch --dependency=after:$jid1 run_implicit_simulate.sh $system $molecule $lrtemp 1 0 $ef_weighttemp '50k'
done



