#!/bin/bash

# Arguments:
# $1: molecule
# $2: dataset name
# $3: rdf loss weight
# $4: vacf loss weight
# $5: energy loss weight
# $6: force loss weight
# $7: exp_name
# molecules_md17=('naphthalene' 'salicylic_acid' 'aspirin')
# # Iterate through the molecules and run the scripts
# for mol in "${molecules_md17[@]}"; do
#     sbatch run_implicit.sh $mol 'md17' 1 0 0.1 1.9 'TEST_resumestrategy0.25_learn_continuous_stronger_energyforce'
# done

#molecules_md17=('naphthalene' 'salicylic_acid' 'aspirin' 'ethanol' 'toluene' 'benzene' 'uracil' 'malonaldehyde')
molecules_md17=('ethanol' 'toluene' 'benzene' 'uracil' 'malonaldehyde')
# Iterate through the molecules and run the scripts
for mol in "${molecules_md17[@]}"; do
    #sbatch run_implicit.sh $mol 'md17' 1 0 0 0 'TEST_resumestrategy0.25_learn_continuous_no_energyforce' 
    sbatch run_implicit.sh $mol 'md17' 1 0 0.05 0.95 'TEST_resumestrategy0.25_learn_continuous'
    #sbatch run_implicit.sh $mol 'md17' 1 0 0.1 1.9 'TEST_resumestrategy0.25_learn_continuous_stronger_energyforce'
done

# # Iterate through the molecules and run the scripts
# for mol in "${molecules_md17[@]}"; do
#     sbatch run_implicit.sh $mol 'md17' 1 1 0.05 0.95 'TEST_resumestrategy0.25_learn_continuous_rdf+vacf_energyforce'
#     sbatch run_implicit.sh $mol 'md17' 0 1 0.05 0.95 'TEST_resumestrategy0.25_learn_continuous_vacfonly_energyforce'
# done
