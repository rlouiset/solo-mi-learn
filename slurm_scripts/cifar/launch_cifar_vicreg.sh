#!/bin/bash

#SBATCH --job-name vicreg_bs_512_CIFAR10
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition V100
#SBATCH --output vicreg_bs_512_CIFAR10.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name vicreg.yaml