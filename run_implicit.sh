#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu             # Request 80GB GPU
#SBATCH -G 1               # Request 1 GPU
#SBATCH -q regular
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -o /global/cfs/projectdirs/m4319/sanjeevr/logs/implicit-md-%j.out
#SBATCH -t 09:00:00
#SBATCH -A m4319_g

# Define the path to your Python script
script="/global/homes/s/sanjeevr/implicit-md/nve_implicit.py"

command_train="python $script --config-yml /global/homes/s/sanjeevr/implicit-md/configs/md17/base.yml --model=schnet --molecule=$1 --name=$2 --size=1k --rdf_loss_weight=$3 --vacf_loss_weight=$4 --minibatch_size=40 \
             --energy_loss_weight=$5 --force_loss_weight=$6 --reset_probability=$7 --exp_name=$8"
command_inference_pre="python $script --config-yml /global/homes/s/sanjeevr/implicit-md/configs/md17/base.yml --model=schnet --molecule=$1 --name=$2 --size=1k --rdf_loss_weight=$3 --vacf_loss_weight=$4 --minibatch_size=40 \
            --energy_loss_weight=$5 --force_loss_weight=$6 --exp_name=$8 --train=False --bond_dev_tol=0.5 --n_epochs=300 --eval_mode=pre"
command_inference_post="python $script --config-yml /global/homes/s/sanjeevr/implicit-md/configs/md17/base.yml --model=schnet --molecule=$1 --name=$2 --size=1k --rdf_loss_weight=$3 --vacf_loss_weight=$4 --minibatch_size=40 \
            --energy_loss_weight=$5 --force_loss_weight=$6 --exp_name=$8 --train=False --bond_dev_tol=0.5 --n_epochs=300 --eval_mode=post"

srun $command_train
srun $command_inference_pre
srun $command_inference_post
    

