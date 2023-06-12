#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=plume_summary
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --open-mode=append
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate EmonetLab
python ../src/training_scripts/plume_summary_test.py