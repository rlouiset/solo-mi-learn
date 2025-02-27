#!/bin/bash

#SBATCH --job-name byol_bs_512_IN100
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 100G
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --partition V100
#SBATCH --output byol_bs_512_IN100.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name robyol.yaml