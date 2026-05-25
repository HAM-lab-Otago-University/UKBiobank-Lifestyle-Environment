#!/bin/bash
#SBATCH --job-name=rs-nilearn-corr-sch
#SBATCH --account=buiir
#SBATCH --partition=aoraki
#SBATCH --auks=yes
##SBATCH --nodelist=aoraki10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/sciences/psychology/UKBiobank/brainbody/scripts/output/%x/%x_%j_%a.out
# Activate Anaconda work environment for OpenDrift
#SBATCH --cpus-per-task=8
#SBATCH --mem=450G
#SBATCH --time=144:00:00
#SBATCH --array=0
## Command(s) to run:

# Activate Anaconda work environment for OpenDrift

source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate ib

python3 rs-nilearn-corr-sch.py ${SLURM_ARRAY_TASK_ID}

echo "finished"
