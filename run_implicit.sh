#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g             # Request 80GB GPU
#SBATCH -G 1               # Request 1 GPU
#SBATCH -q debug
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -o /global/cfs/projectdirs/m4319/sanjeevr/logs/implicit-md-%j.out
#SBATCH -t 00:03:00
#SBATCH -A m4319_g

# Define the path to your Python script
script="/global/homes/s/sanjeevr/implicit-md/nvt_implicit.py"
config_yml="/global/homes/s/sanjeevr/implicit-md/configs/$1/train.yml"
command_train="python $script --config-yml $config_yml --model=$2 --rdf_loss_weight=$3 --vacf_loss_weight=$4 \
               --energy_force_loss_weight=$5 --only_learn_if_unstable_threshold_reached=$6 --exp_name=$7"
srun $command_train

    

