#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g             # Request 80GB GPU
#SBATCH -G 1               # Request 1 GPU
#SBATCH -q regular
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -o /global/cfs/projectdirs/m4319/sanjeevr/logs/implicit-md-%j.out
#SBATCH -t 09:00:00
#SBATCH -A m4319_g

# Define the path to your Python script
script="/global/homes/s/sanjeevr/implicit-md/nvt_implicit.py"
config_yml="/global/homes/s/sanjeevr/implicit-md/configs/water/water.yml"
command_train="python $script --config-yml $config_yml --model=$1 --rdf_loss_weight=$2 --vacf_loss_weight=$3 \
             --energy_loss_weight=$4 --force_loss_weight=$5 --reset_probability=$6 --exp_name=$7"
# command_inference_pre="python $script --config-yml $config_yml --model=nequip --molecule=$1 --size=10percent --rdf_loss_weight=$2 --vacf_loss_weight=$3 --minibatch_size=8 \
#             --energy_loss_weight=$4 --force_loss_weight=$5 --exp_name=$7 --train=False --bond_dev_tol=0.5 --n_epochs=300 --eval_mode=pre"
# command_inference_post="python $script --config-yml $config_yml --model=nequip --molecule=$1 --size=10percent --rdf_loss_weight=$2 --vacf_loss_weight=$3 --minibatch_size=8 \
#             --energy_loss_weight=$4 --force_loss_weight=$5 --exp_name=$7 --train=False --bond_dev_tol=0.5 --n_epochs=300 --eval_mode=post"
srun $command_train
# srun $command_inference_pre
# srun $command_inference_post
    

