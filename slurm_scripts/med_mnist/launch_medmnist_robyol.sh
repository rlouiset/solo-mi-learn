#!/bin/bash

#SBATCH --job-name robyol_bs_512_pathmnist
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition A100
#SBATCH --output robyol_bs_512_bloodmnist.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name byol_blood.yaml
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name byol_path.yaml





