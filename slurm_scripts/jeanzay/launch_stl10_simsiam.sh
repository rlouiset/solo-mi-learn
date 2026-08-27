#!/bin/bash

#SBATCH --job-name simsiam_stl
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output simsiam_stl_x2lr2.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
srun python3 main_pretrain.py --config-path scripts/pretrain/stl/ --config-name simsiam.yaml
