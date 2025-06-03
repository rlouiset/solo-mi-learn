#!/bin/bash

#SBATCH --job-name simclr_bs_128_octmnist
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --partition A100
#SBATCH --output simclr_octmnist.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name simclr_oct.yaml