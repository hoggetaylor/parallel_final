
INCLUDE=-I/usr/local/cuda/8.0/cuda/include

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

buildrev : reverse/reviter.cu
	cd reverse; nvcc $(INCLUDE) -o reviter reviter.cu

runrev : buildrev
	./reverse/reviter

batchrev : slreverse.slurm
	sbatch slreverse.slurm

editrev : reverse/reviter.cu
	vim reverse/reviter.cu

# CUDA IMPROVEMENTS ONLY

buildcuda : cuda/cudatiming.cu
	cd cuda; nvcc $(INCLUDE) -o cudatime cudatiming.cu

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
