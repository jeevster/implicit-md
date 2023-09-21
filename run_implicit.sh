#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J Test
#SBATCH --mail-user=sanjeevr@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00
#SBATCH -A mp4319

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  /global/homes/s/sanjeevr/implicit-md/nve_implicit.py