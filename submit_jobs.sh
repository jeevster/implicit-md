#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy loss weight
# $5: force loss weight
# $6: reset probability
# $7: exp_name

molecules_md22=('ac_Ala3_NHMe' 'DHA' 'AT_AT')
# eval_modes=('post') # 'post' 'lmax=2' 'lmax=3' '50percent' '100percent')
# # Iterate through the molecules and run the scripts
# for mol in "${molecules_md22[@]}"; do
#     for eval_mode in "${eval_modes[@]}"; do
#         sbatch run_implicit.sh $mol 1 0 0 0 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_noenergyforce' $eval_mode
#         sbatch run_implicit.sh $mol 1 0 0.05 0.95 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous' $eval_mode
#         # sbatch run_implicit.sh $mol 1 0 0.10 1.9 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_strongerenergyforce' $eval_mode
#     done
# done

eval_modes=('lmax=2' 'lmax=3')
# Iterate through the molecules and run the scripts
for mol in "${molecules_md22[@]}"; do
    for eval_mode in "${eval_modes[@]}"; do
        # sbatch run_implicit.sh $mol 1 0 0 0 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_noenergyforce' $eval_mode
        sbatch run_implicit.sh $mol 1 0 0.05 0.95 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous' $eval_mode
        # sbatch run_implicit.sh $mol 1 0 0.10 1.9 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_strongerenergyforce' $eval_mode
    done
done

# models=('schnet' 'dimenetplusplus')
# # Iterate through the molecules and run the scripts
# for model in "${models[@]}"; do
#     sbatch run_implicit.sh $model 1 0 0 0 0.0 'TEST_resumestrategy0.8_learn_continuous_noenergyforce'
#     sbatch run_implicit.sh $model 1 0 0.05 0.95 0.0 'TEST_resumestrategy0.8_learn_continuous'
#     sbatch run_implicit.sh $model 1 0 0.10 1.9 0.0 'TEST_resumestrategy0.8_learn_continuous_strongerenergyforce'
# done

