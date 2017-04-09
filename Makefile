buildseq : sequential/FDTD3D.c
	cd sequential; gcc -o FDTD3D FDTD3D.c -lm -O3

runseq : buildseq 
	./sequential/FDTD3D

buildrev : reverse/reviter.c
	cd reverse; gcc -o reviter reviter.c -lm -O3

runrev : buildrev
	./reverse/reviter

buildcuda : cuda/cudatiming.cu
	cd cuda; nvcc -o cudatime cudatiming.cu

runcuda : buildcuda
	./cuda/cudatime

clean :
	rm -f sequential/FDTD3D; rm -f reverse/reviter; rm -f cuda/cudatime
