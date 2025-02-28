#!/bin/bash

#SBATCH --job-name simclr_bs_512_bloodmnist
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --partition V100
#SBATCH --output simclr_bs_512_bloodmnist.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name simclr_blood.yaml