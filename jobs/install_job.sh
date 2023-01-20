#!/bin/sh

#SBATCH --job-name="fdtd-install"
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

module load 2022r2
module load python
module load cuda/11.6
module load openmpi
module load miniconda3
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p /scratch/${USER}/.conda
conda update -n base -c defaults conda
conda create -y -p /scratch/${USER}/.conda
conda activate /scratch/${USER}/.conda
conda install -y numpy numba scipy matplotlib
conda install -y -c conda-forge pyopencl
