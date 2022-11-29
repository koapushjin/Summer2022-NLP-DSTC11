#!/usr/bin/bash
#
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --output=log_roberta_train_%A_%a.out
#SBATCH --error=log_roberta_train_%A_%a.err
#SBATCH --time=48:00:00

module purge

source $HOME/.bashrc

conda activate dstc11

module load cuda/11.2

export TASK_NAME=dstc10 # dataset is defined in ./glue.py
export TOKENIZERS_PARALLELISM=true

# train a QA cross-encoder BERT model
python run_glue.py \
  --model_name_or_path ./roberta-base-local \
  --tokenizer_name ./roberta-base-local \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./routput >& rtrain.log
