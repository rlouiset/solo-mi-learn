#!/bin/bash

#SBATCH --job-name byol
#SBATCH --time=00-22:59:00
#SBATCH --nodes=1
#SBATCH --mem 80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition gpu-p2
#SBATCH --output byol.txt

source activate py39
srun python3 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name byol.yaml