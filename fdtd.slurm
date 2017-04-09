#!/bin/bash

#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=taylor.hogge@utah.edu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH -o fdtd.out

module load gcc

make || exit 1
make run
