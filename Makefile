#  SEQUENTIAL PROCESS

buildseq : sequential/FDTD3D.c
	cd sequential; gcc -o FDTD3D FDTD3D.c -lm -O3

runseq : buildseq 
	./sequential/FDTD3D

batchseq : slsequential.slurm
	sbatch slsequential.slurm

editseq : sequential/FDTD3D.c
	vim sequential/FDTD3D.c

# REVERSED ITERATION

buildrev : reverse/reviter.c
	cd reverse; gcc -o reviter reviter.c -lm -O3

runrev : buildrev
	./reverse/reviter

batchrev : slreverse.slurm
	sbatch slreverse.slurm

editrev : reverse/reviter.c
	vim reverse/reviter.c

# CUDA IMPROVEMENTS ONLY

buildcuda : cuda/cudatiming.cu
	cd cuda; nvcc -o cudatime cudatiming.cu

runcuda : buildcuda
	./cuda/cudatime

batchcuda : slcuda.slurm
	sbatch slcuda.slurm

editcuda : cuda/cudatiming.cu
	vim cuda/cudatiming.cu

# OTHER HELPERS

batchall :
	sbatch slsequential.slurm; sbatch slreverse.slurm; sbatch slcuda.slurm

clean :
	rm -f sequential/FDTD3D; rm -f reverse/reviter; rm -f cuda/cudatime
