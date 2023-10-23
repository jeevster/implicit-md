#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy/force loss weight
# $5: exp_name

molecules_md17=('aspirin' 'naphthalene' 'salicylic_acid')
models=('schnet') #  'forcenet')
eval_modes=('pre') # 'post' '10k')
# # Iterate through the molecules and run the scripts
for model in "${models[@]}"; do
    jid1=$(sbatch --parsable run_implicit.sh water $model 1 0 0 False 'TEST_1k_noenergyforce_alwayslearn')
    for eval_mode in "${eval_modes[@]}"; do
        sbatch --dependency=afterany:$jid1 run_implicit_simulate.sh 'water' $model 1 0 0 $eval_mode 'TEST_1k_noenergyforce_alwayslearn'
    done
done

# sbatch run_implicit.sh $model 1 0 1 True 'TEST_1k_energyforce'
# sbatch run_implicit.sh $model 1 0 2 True 'TEST_1k_strongerenergyforce'
# sbatch run_implicit.sh $model 1 1 1 True 'TEST_1k_rdf+vacf_energyforce'
# sbatch run_implicit.sh $model 1 0 0 False 'TEST_1k_noenergyforce_alwayslearn'
# sbatch run_implicit.sh $model 1 0 1 False 'TEST_1k_energyforce_alwayslearn'
# sbatch run_implicit.sh $model 1 0 2 False 'TEST_1k_strongerenergyforce_alwayslearn'
# sbatch run_implicit.sh $model 1 1 1 False 'TEST_1k_rdf+vacf_energyforce_alwayslearn'


