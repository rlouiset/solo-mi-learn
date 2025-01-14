#!/bin/bash

#SBATCH --job-name ssl_bs_512_STL10
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 50G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition A100
#SBATCH --output ssl_bs_512_STL10.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/stl/ --config-name simclr.yaml
srun python3 main_pretrain.py --config-path scripts/pretrain/stl/ --config-name dino.yaml
srun python3 main_pretrain.py --config-path scripts/pretrain/stl/ --config-name barlow.yaml



