#!/bin/bash

#SBATCH --job-name="fdtd-run"
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

if [[ $# -eq 0 ]] ; then
    echo "Usage: sbatch $0 -m fdtd run [options]"
    echo "   or: sbatch $0 path/to/script.py [options]"
    echo 'Argument is required!'
    exit 1
fi

module load 2022r2
module load python
module load cuda/11.6
module load openmpi
module load py-pip

# install the fdtd package and its dependencies from the submit directory
pip install --user --quiet .

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

echo "Running: python $*"
srun python "$@"

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
