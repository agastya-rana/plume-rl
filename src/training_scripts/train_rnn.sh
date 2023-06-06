#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=rnn_shape_upwind
#SBATCH --mem-per-cpu=42G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-9

eval "$(conda shell.bash hook)"
conda activate EmonetLab
python src/training_scripts/train_rnn_baseline.py $SLURM_ARRAY_TASK_ID