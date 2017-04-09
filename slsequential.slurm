#!/bin/bash

#SBATCH --account=soc-kp
#SBATCH --partition=soc-kp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=taylor.hogge@utah.edu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH -o sequential/results.txt
#SBATCH -e sequential/err.txt


module load gcc

make runseq
