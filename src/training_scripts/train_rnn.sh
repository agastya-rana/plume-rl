#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=rnn_training_vec
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=20
#SBATCH --open-mode=append
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL

module load StdEnv
module load miniconda
source /gpfs/loomis/apps/avx/software/miniconda/23.1.0/etc/profile.d/conda.sh
conda activate EmonetLab
python -m src.training_scripts.train_rnn_baseline