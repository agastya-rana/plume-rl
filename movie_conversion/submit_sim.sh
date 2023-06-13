#!/bin/bash
#SBATCH --job-name=convert_movie
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-0
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
 
python3 run_big_movie_conversion.py $SLURM_ARRAY_TASK_ID
 
exit 0
