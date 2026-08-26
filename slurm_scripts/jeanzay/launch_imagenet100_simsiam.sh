#!/bin/bash

#SBATCH --job-name simsiam_in100
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --constraint a100
#SBATCH --account haj@a100
#SBATCH --output simsiam_in100.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name simsiam.yaml
