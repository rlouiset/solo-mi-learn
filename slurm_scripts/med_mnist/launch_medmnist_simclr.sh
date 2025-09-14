#!/bin/bash

#SBATCH --job-name dino_bs_512_medmnist
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 80G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition A100
#SBATCH --output dino_bs_512_medmnist.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name dino_blood.yaml
srun python3 main_pretrain.py --config-path scripts/pretrain/med-mnist/ --config-name dino_path.yaml