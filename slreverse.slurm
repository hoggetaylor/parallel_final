#!/bin/bash

#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH -o reverse/results.txt
#SBATCH -e reverse/err.txt

module load gcc

make runrev
