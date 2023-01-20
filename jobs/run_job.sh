#!/bin/sh

#SBATCH --job-name="fdtd-run"
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
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
conda create -y -p /scratch/${USER}/.conda
conda activate /scratch/${USER}/.conda
conda install -y numpy numba scipy matplotlib
conda install -y -c conda-forge pyopencl

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

echo ${USER}
echo /scratch/${USER}/.conda
srun python /home/dflipse/python-opencl-fdtd/full_sweep.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
conda deactivate