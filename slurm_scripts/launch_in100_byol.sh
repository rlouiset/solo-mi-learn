#!/bin/bash

#SBATCH --job-name robyol_A100_IN100
#SBATCH --time=00-23:59:00
#SBATCH --nodes=1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition A100
#SBATCH --output robyol_A100_IN100.txt

export PATH=/home/ids/rareme/miniconda3/bin:$PATH
source activate base
nvidia-smi
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name simclr.yaml