#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=dqn_best
#SBATCH --mem-per-cpu=42G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-9

eval "$(conda shell.bash hook)"
conda activate EmonetLab
python ../src/training_scripts/train_dqn_history.py $SLURM_ARRAY_TASK_ID