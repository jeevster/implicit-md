#!/bin/bash

# Arguments:
# $1: model
# $2: rdf loss weight
# $3: vacf loss weight
# $4: energy loss weight
# $5: force loss weight
# $6: reset probability
# $7: exp_name

# molecules_md22=('ac_Ala3_NHMe' 'DHA' 'AT_AT')
# # Iterate through the molecules and run the scripts
# for mol in "${molecules_md22[@]}"; do
#     sbatch run_implicit.sh $mol 1 0 0 0 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_noenergyforce'
#     sbatch run_implicit.sh $mol 1 0 0.05 0.95 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous'
#     sbatch run_implicit.sh $mol 1 0 0.10 1.9 0.0 'TEST_lmax1_10percent_resumestrategy0.25_learn_continuous_strongerenergyforce'
# done

models=('schnet' 'forcenet')
# Iterate through the molecules and run the scripts
for model in "${models[@]}"; do
    sbatch run_implicit.sh $model 1 0 0 0 0.0 'TEST_resumestrategy0.25_learn_continuous_noenergyforce'
    sbatch run_implicit.sh $model 1 0 0.05 0.95 0.0 'TEST_resumestrategy0.25_learn_continuous'
    sbatch run_implicit.sh $model 1 0 0.10 1.9 0.0 'TEST_resumestrategy0.25_learn_continuous_strongerenergyforce'
done

