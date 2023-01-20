#!/bin/sh

#SBATCH --job-name="fdtd-run"
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus-per-task=1

module load 2022r2
module load python
module load cuda/11.6
module load openmpi
module load miniconda3
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dhpc
srun python ./full_sweep.py
conda deactivate