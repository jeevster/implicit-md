#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g      # Request 80GB GPU
#SBATCH -G 1               # Request 1 GPU
#SBATCH -q regular
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -o /global/cfs/projectdirs/m4319/sanjeevr/logs/implicit-md-%j.out
#SBATCH -t 06:00:00
#SBATCH -A m4319_g

# Define the path to your Python script
script="/global/homes/s/sanjeevr/implicit-md/nve_implicit.py"

command="python $script --config-yml /global/homes/s/sanjeevr/implicit-md/configs/md17/base.yml --molecule=$1 --name=$2 --size=1k --rdf_loss_weight=$3 --vacf_loss_weight=$4 --minibatch_size=40 \
             --energy_loss_weight=$5 --force_loss_weight=$6 --exp_name=$7"

srun $command
    

