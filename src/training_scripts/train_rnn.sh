#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=rnn_cont_4k
#SBATCH --mem-per-cpu=42G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate EmonetLab
python src/training_scripts/train_rnn_baseline.py