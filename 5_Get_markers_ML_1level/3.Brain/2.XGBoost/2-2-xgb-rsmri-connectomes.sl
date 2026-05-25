#!/bin/bash
#SBATCH --job-name=xgb-rs-conn-part
#SBATCH --account=buiir163
#SBATCH --partition=aoraki
#SBATCH --auks=yes
##SBATCH --nodelist=aoraki10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/sciences/psychology/narunpat-lab/UKBiobank/brainbody/scripts/output/%x/%x_%j_%a.out
# Activate Anaconda work environment for OpenDrift
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=200:00:00
#SBATCH --array=0-24
## Command(s) to run:

# Activate Anaconda work environment for OpenDrift

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ib

python3 2-2-xgb-rsmri-connectomes.py ${SLURM_ARRAY_TASK_ID}

echo "finished"
