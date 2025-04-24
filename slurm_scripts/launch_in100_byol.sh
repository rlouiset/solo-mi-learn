#!/bin/bash

#SBATCH --job-name rosiam_bs_512_A100_IN100
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --partition A100
#SBATCH --output rosiam_bs_512_A100_IN100.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
nvidia-smi
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name simsiam_vit.yaml