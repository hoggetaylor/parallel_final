FDTD3D : FDTD3D.c
	gcc -o FDTD3D FDTD3D.c -lm -O3

clean :
	rm -f FDTD3D

run : FDTD3D
	./FDTD3D

buildrev : reviter.c
	gcc -o reviter reviter.c -lm -03

runrev : buildrev
	./reviter
