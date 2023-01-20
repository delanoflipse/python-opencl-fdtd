#!/bin/sh

#SBATCH --job-name="fdtd-run"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

module load 2022r2
module load python
module load cuda/11.6
module load openmpi
module load py-pip

pip install numpy numba scipy pyopencl matplotlib

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python /home/dflipse/python-opencl-fdtd/full_sweep.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
