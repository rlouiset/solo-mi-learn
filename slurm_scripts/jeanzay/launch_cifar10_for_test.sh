#!/bin/bash

#SBATCH --job-name byol
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint a100
#SBATCH --account haj@a100
#SBATCH --output byol.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
srun python3 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol.yaml