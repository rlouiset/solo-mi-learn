#!/bin/bash

#SBATCH --job-name dino_bs_512_CIFAR100
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition A40
#SBATCH --output dino_bs_512_CIFAR10.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino.yaml