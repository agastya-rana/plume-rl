#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=dqn_hist
#SBATCH --mem-per-cpu=42G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

eval "$(conda shell.bash hook)"
conda activate EmonetLab
python ../src/training_scripts/train_dqn_history.py $SLURM_ARRAY_TASK_ID