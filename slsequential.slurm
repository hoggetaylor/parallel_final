#!/bin/bash

#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --qos=soc-gpu-kp
#SBATCH --export=ALL
#SBATCH -o sequential/results.txt
#SBATCH -e sequential/err.txt
ulimit -c unlimited -s

module load gcc
module load cuda

make runseq
