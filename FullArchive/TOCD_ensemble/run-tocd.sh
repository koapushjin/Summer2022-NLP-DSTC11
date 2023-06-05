#!/usr/bin/bash
#
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --output=log_train_%A_%a.out
#SBATCH --error=log_train_%A_%a.err
#SBATCH --time=96:00:00

# install dependencies
#pip3 install -r requirements.txt

python3 -m train \
    --num_train_epochs 10 \
    --window_size 5 \
    --starting_epoch 0 \
    --domain banking

python3 -m test \
    --num_train_epochs 10 \
    --window_size 5 \
    --starting_epoch 0 \
    --num_trials 20 \
    --domain banking
