#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g             # Request 80GB GPU
#SBATCH -G 1               # Request 1 GPU
#SBATCH -q regular
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH -o /global/cfs/projectdirs/m4319/sanjeevr/logs/implicit-md-%j.out
#SBATCH -t 24:00:00
#SBATCH -A m4319_g

# Define the path to your Python script
script="/global/homes/s/sanjeevr/implicit-md/nvt_implicit.py"
config_yml="/global/homes/s/sanjeevr/implicit-md/configs/$1/simulate.yml"
if [ "$1" == "water" ]; then
    command_inference="python $script --config-yml $config_yml --model=$2 --lr=$3 --rdf_loss_weight=$4 --vacf_loss_weight=$5 \
               --energy_force_loss_weight=$6 --eval_model=$7 --results_dir=results_water_localneighborhoods --timestep=$8 --n_dump=$9 --steps=${10}"
else 
    command_inference="python $script --config-yml $config_yml --molecule=$2 --lr=$3 --rdf_loss_weight=$4 --vacf_loss_weight=$5 \
               --energy_force_loss_weight=$6 --eval_model=$7 --results_dir=results_md17_correctedlogging --timestep=$8 --n_dump=$9 --steps=${10}"
fi
echo $command_inference
srun -u $command_inference
    

