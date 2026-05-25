#!/bin/bash
#SBATCH --job-name=xgboost-dwmri-idp-adapt-grid
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
#SBATCH --array=0-90
## Command(s) to run:

# Activate Anaconda work environment for OpenDrift

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ib

python3 1-1-xgboost-dwmri-idps.py ${SLURM_ARRAY_TASK_ID}

echo "finished"
