#!/bin/bash

#SBATCH --job-name byol_512_H100_IN1K
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition H100
#SBATCH --output byol_512_H100_IN1K.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
nvidia-smi
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name byol.yaml