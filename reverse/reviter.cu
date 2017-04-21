// To compile - gcc -o 3dFDTD FDTD3D.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

int main() {
    int imax = 100, jmax = 100, nmax = 1000, nhalf = 20, no = nhalf*3, kmax = 100;
    int i, j, n,k;
    double c = 2.99792458e8, pi = 3.141592654, sigma = 0, mu = 4.0 * pi * 1.0e-7, eps = 8.85418782e-12;
    double delta = 1e-3;
    double dt = delta/(c*1.41421356237);

    double ***Ex, ***Ey, ***Ez, ***Hy, ***Hx, ***Hz;

    cudaEvent_t start_event, stop_event;
    float elapsed_time;

    Ex = (double ***)malloc((imax+1)*sizeof(double **));
    Ey = (double ***)malloc((imax+1)*sizeof(double **));
    Ez = (double ***)malloc((imax+1)*sizeof(double **));
    Hx = (double ***)malloc((imax+1)*sizeof(double **));
    Hy = (double ***)malloc((imax+1)*sizeof(double **));
    Hz = (double ***)malloc((imax+1)*sizeof(double **));

    for(i=0;i<(imax+1);i++) {
        Ex[i] = (double **)malloc((jmax+1)*sizeof(double *));
        Ey[i] = (double **)malloc((jmax+1)*sizeof(double *));
        Ez[i] = (double **)malloc((jmax+1)*sizeof(double *));
        Hx[i] = (double **)malloc((jmax+1)*sizeof(double *));
        Hy[i] = (double **)malloc((jmax+1)*sizeof(double *));
        Hz[i] = (double **)malloc((jmax+1)*sizeof(double *));
        
        for(j=0;j<(jmax+1);j++) {
            Ex[i][j] = (double *)malloc((kmax+1)*sizeof(double));
            Ey[i][j] = (double *)malloc((kmax+1)*sizeof(double));
            Ez[i][j] = (double *)malloc((kmax+1)*sizeof(double));
            Hx[i][j] = (double *)malloc((kmax+1)*sizeof(double));
            Hy[i][j] = (double *)malloc((kmax+1)*sizeof(double));
            Hz[i][j] = (double *)malloc((kmax+1)*sizeof(double));
        }
    }	

    for(k=0;k<(kmax+1);k++){
	for(j=0;j<(jmax+1);j++){
	    for(i=0;i<(imax+1);i++){
                Ex[i][j][k] = 0.0;
                Ey[i][j][k] = 0.0;
                Ez[i][j][k] = 0.0;
                Hx[i][j][k] = 0.0;
                Hy[i][j][k] = 0.0;
                Hz[i][j][k] = 0.0;
	    }
	}
    }	

    double Ca,Cb,Da,Db;

    char buf[18];

    Ca = (1-((sigma*dt)/(2*eps)))/(1+((sigma*dt)/(2*eps)));
    Cb = (dt/(eps*delta))/(1+((sigma*dt)/(2*eps)));
    Da = (1-((sigma*dt)/(2*mu)))/(1+((sigma*dt)/(2*mu)));
    Db = (dt/(mu*delta))/(1+((sigma*dt)/(2*mu)));

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    for (n = 0; n < nmax; n++) {
        for (i = 0; i < imax; i++) {
            for (j = 1; j < jmax; j++) {
                for (k = 1; k < kmax; k++) {
                    Ex[i][j][k] = Ca*Ex[i][j][k] + Cb*((Hz[i][j][k] - Hy[i][j-1][k]) + (Hy[i][j][k-1] - Hy[i][j][k]));
                }
            }
        }
        for (i = 1; i < imax; i++) {
            for (j = 0; j < jmax; j++) {
                for (k = 1; k < kmax; k++) {
                    Ey[i][j][k] = Ca*Ey[i][j][k] + Cb*((Hz[i-1][j][k] - Hy[i][j][k]) + (Hy[i][j][k] - Hy[i][j][k-1]));
                }
            }
        }
        for (i = 1; i < imax; i++) {
            for (j = 1; j < jmax; j++) {
                for (k = 0; k < kmax; k++) {
                    Ez[i][j][k] = Ca*Ez[i][j][k] + Cb*((Hz[i][j][k] - Hy[i-1][j][k]) + (Hy[i][j-1][k] - Hy[i][j][k]));
                }
            }
        }
        Ez[imax/2][jmax/2][kmax/2] = exp(-(pow(((n-no)/(double)nhalf),2.0)));
        for (i = 1; i < imax; i++) {
           for (j = 0; j < jmax-1; j++) {
               for (k = 0; k < kmax-1; k++) {
                   Hx[i][j][k] = Da*Hx[i][j][k] + Db*((Ez[i][j][k] - Ez[i][j+1][k]) + (Ez[i][j][k+1]-Ez[i][j][k]));
               }
           }
        }
        for (i = 0; i < imax; i++) {
           for (j = 1; j < jmax-1; j++) {
               for (k = 0; k < kmax-1; k++) {
                   Hy[i][j][k] = Da*Hy[i][j][k] + Db*((Ez[i+1][j][k] - Ez[i][j][k]) + (Ez[i][j][k]-Ez[i][j][k+1]));
               }
           }
        }
        for (i = 0; i < imax; i++) {
           for (j = 0; j < jmax-1; j++) {
               for (k = 1; k < kmax-1; k++) {
                   Hz[i][j][k] = Da*Hz[i][j][k] + Db*((Ez[i][j][k] - Ez[i+1][j][k]) + (Ez[i][j+1][k]-Ez[i][j][k]));
               }
           }
        }
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    FILE * fPointer;
    fPointer = fopen("parlleloutput.dat","w");
    int x, y, z;
    for (x=0; x<kmax; x++) {
      for (y=1; y<jmax; y++) {
        for (z=1; z<imax; z++) {
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Ez[(z*imax) + (y*jmax) + x]);
          fputs(buf, fPointer);
        }
      }
    }
    fclose(fPointer);

    // calculate run time

    printf("GPU Time: %.2f\n", elapsed_time);
    return 0;
}
