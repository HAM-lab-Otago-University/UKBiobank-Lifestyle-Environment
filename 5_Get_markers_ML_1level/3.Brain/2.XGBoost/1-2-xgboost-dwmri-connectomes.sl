#!/bin/bash
#SBATCH --job-name=xgb-dwi-conn-schaef1000
#SBATCH --account=buiir
#SBATCH --partition=aoraki
#SBATCH --auks=yes
##SBATCH --nodelist=aoraki10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/sciences/psychology/UKBiobank/brainbody/scripts/output/%x/%x_%j_%a.out
# Activate Anaconda work environment for OpenDrift
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=200:00:00
#SBATCH --array=12-17
## Command(s) to run:

# Activate Anaconda work environment for OpenDrift

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ib

python3 1-2-xgboost-dwmri-connectomes.py ${SLURM_ARRAY_TASK_ID}

echo "finished"
