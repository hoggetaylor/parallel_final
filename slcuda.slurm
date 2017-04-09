#!/bin/bash

#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH -o cuda/results.txt
#SBATCH -e cuda/err.txt

module load gcc
module load cuda

make runcuda
