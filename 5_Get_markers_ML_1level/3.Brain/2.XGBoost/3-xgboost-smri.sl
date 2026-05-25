#!/bin/bash
#SBATCH --job-name=xgboost-smri-adapt-grid
#SBATCH --account=buiir
#SBATCH --partition=aoraki
##SBATCH --nodelist=aoraki10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/sciences/psychology/UKBiobank/brainbody/scripts/output/%x/%x_%j_%a.out
# Activate Anaconda work environment for OpenDrift
#SBATCH --cpus-per-task=8
#SBATCH --mem=800M
#SBATCH --time=144:00:00
#SBATCH --array=1-59
## Command(s) to run:

# Activate Anaconda work environment for OpenDrift

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ib

python3 3-xgboost-smri.py ${SLURM_ARRAY_TASK_ID}

echo "finished"
