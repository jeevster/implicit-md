#!/bin/bash
#BSUB -J "nequip_3BPA_RUN_NAME_test_err"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 1:00
#BSUB -alloc_flags ipisolate

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# Activate conda environment
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# just to record each node we're using in the job output
jsrun -r 1 hostname
 
# get hostname of node that jsrun considers to be first (where rank 0 will run)
firsthost=`jsrun --nrs 1 -r 1 hostname`
echo "first host: $firsthost"

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any used port number
export MASTER_ADDR=$firsthost
export MASTER_PORT=23556

# Runtime settings
NUM_NODES=1
GPUS_PER_NODE=1
CPUS_PER_GPU=1
CPUS_PER_NODE=$(( $GPUS_PER_NODE*$CPUS_PER_GPU ))

# -r: number of resource sets per node
# -a: number of "tasks" per resource set (default is 1 task = 1 process)
# -c: number of CPUs per resource set
# -g: number of GPUs per resource set
# --bind=none: allow each task to use all of its allocated cpus
# jsrun -r 1 -a $GPUS_PER_NODE -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.landscape \
jsrun --smpiargs='off' -r 1 -a $GPUS_PER_NODE -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.landscape \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --batch-size 1 \
    --loss-type 'both' \
    --landscape-type 'lines' \
    --steps 81 \
    --distance 0.5 \
    --model-type 'nequip' \
    --n-lines 20 \
    --model-path '/g/g20/vita1/ws/projects/nequip/results/3BPA/dsize_redo_nval_amsgrad_rescale/RUN_NAME' \
    --database-path '/g/g20/vita1/ws/projects/nequip/results/3BPA/dsize_redo_nval_amsgrad_rescale/RUN_NAME' \
    --additional-datamodule-kwargs "train_filename:/g/g20/vita1/ws/projects/data/3BPA/dataset_3BPA/test_full.xyz" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/3BPA/nequip/dsize_redo_nval_amsgrad_rescale/RUN_NAME_test_full" \
    --no-compute-landscape \
    # --no-compute-initial-losses \
