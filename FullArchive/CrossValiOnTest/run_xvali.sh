#!/usr/bin/bash
#
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --output=log_train_%A_%a.out
#SBATCH --error=log_train_%A_%a.err
#SBATCH --time=48:00:00

python3 -m x-validation
