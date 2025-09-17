#!/bin/bash

#SBATCH --job-name byol_in1k
#SBATCH --time=0-19:59:00
#SBATCH --nodes=1
# SBATCH --cpus-per-task=64               # increase CPU cores for data loading
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=internet
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output byol_in1k.txt

# Set environment variables for optimal DDP and data loading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK       # for CPU operations / numpy
export NCCL_DEBUG=INFO                             # to check DDP communication
export PYTHONFAULTHANDLER=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name byol.yaml