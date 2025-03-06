#!/bin/bash

#SBATCH --job-name byol-mc
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition V100
#SBATCH --output byol-mc.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simsiam.yaml