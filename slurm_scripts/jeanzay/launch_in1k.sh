#!/bin/bash

#SBATCH --job-name byol_in1k
#SBATCH --time=0-19:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --constraint a100
#SBATCH --account haj@a100
#SBATCH --output byol_in1k.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
srun python3 main_pretrain.py --config-path scripts/pretrain/imagenet/ --config-name byol.yaml