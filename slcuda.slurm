#!/bin/bash

#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --qos=soc-gpu-kp
#SBATCH --export=ALL
#SBATCH -o cuda/results.txt
#SBATCH -e cuda/err.txt
ulimit -c unlimited -s

module load gcc
module load cuda

make runcuda
