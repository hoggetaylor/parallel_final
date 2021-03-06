// To compile - gcc -o 3dFDTD FDTD3D.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IMAX 100
#define JMAX 100
#define KMAX 100

// This was taken from stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPU error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 *  First half of the total loop circuit
 */
extern __global__ void loop1_GPU(double (*Ex)[IMAX][JMAX], double (*Ey)[IMAX][JMAX], double (*Ez)[IMAX][JMAX], double (*Hy)[IMAX][JMAX], double (*Hz)[IMAX][JMAX], double Cb, double Ca, int n, int no, int nhalf) {
  int i, j;
  int k = blockIdx.x * 32 + threadIdx.x;
  
  if (k < KMAX && k > 0) {
    for (j=1; j<JMAX; j++) {
      for (i=0; i<IMAX; i++) {
	Ex[i][j][k] = Ca*Ex[i][j][k] + Cb*((Hz[i][j][k] - Hy[i][j-1][k]) + (Hy[i][j][k-1] - Hy[i][j][k]));
      }
    }
    for (j=0; j<JMAX; j++) {
      for (i=1; i<IMAX; i++) {
	Ey[i][j][k] = Ca*Ey[i][j][k] + Cb*((Hz[i-1][j][k] - Hy[i][j][k]) + (Hy[i][j][k] - Hy[i][j][k-1]));
      }
    }
  } // end if k
  if (k < KMAX) {
    for (j=1; j<JMAX; j++) {
      for (i=1; i<IMAX; i++) {
	Ez[i][j][k] = Ca*Ez[i][j][k] + Cb*((Hz[i][j][k] - Hy[i-1][j][k]) + (Hy[i][j-1][k] - Hy[i][j][k]));
      }
    }
  } // end if k
  if (k==(KMAX/2)) {
    Ez[IMAX/2][JMAX/2][k] = exp(-(pow(((n-no)/(double)nhalf),2.0)));
  }
}

/**
 *  Second half of the total loop circuit.
 */
extern __global__ void loop2_GPU(double (*Ez)[IMAX][JMAX], double (*Hx)[IMAX][JMAX], double (*Hy)[IMAX][JMAX], double (*Hz)[IMAX][JMAX], double Da, double Db) {
    int i, j;
    int k = blockIdx.x * 32 + threadIdx.x;

    if (k < KMAX) {
        for (j = 0; j < JMAX-1; j++) {
            for (i = 1; i < IMAX-1; i++) {
                Hx[i][j][k] = Da*Hx[i][j][k] + Db*((Ez[i][j][k] - Ez[i][j+1][k]) + (Ez[i][j][k+1]-Ez[i][j][k]));
            }
        }
       for (j = 1; j < JMAX-1; j++) {
           for (i = 0; i < IMAX-1; i++) {
               Hy[i][j][k] = Da*Hy[i][j][k] + Db*((Ez[i+1][j][k] - Ez[i][j][k]) + (Ez[i][j][k]-Ez[i][j][k+1]));
           }
       }
    } // end if k
    if (k < KMAX && k > 0) {
       for (j = 0; j < JMAX-1; j++) {
           for (i = 0; i < IMAX-1; i++) {
               Hz[i][j][k] = Da*Hz[i][j][k] + Db*((Ez[i][j][k] - Ez[i+1][j][k]) + (Ez[i][j+1][k]-Ez[i][j][k]));
           }
       }
    } // end if k
}

int main() {
    int nmax = 400, nhalf = 20, no = nhalf*3;
    int n;
    double c = 2.99792458e8, pi = 3.141592654, sigma = 0, mu = 4.0 * pi * 1.0e-7, eps = 8.85418782e-12;
    double delta = 1e-3;
    double dt = delta/(c*1.41421356237);

    int counter = 0;

    double *Ex, *Ey, *Ez, *Hy, *Hx, *Hz;

    cudaEvent_t start_event, stop_event;
    float elapsed_time;

    Ex = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));
    Ey = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));
    Ez = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));
    Hx = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));
    Hy = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));
    Hz = (double *)calloc((IMAX+1) * (JMAX+1) * (KMAX+1), sizeof(double));

    double (*g_Ex)[IMAX][JMAX];
    double (*g_Ey)[IMAX][JMAX];
    double (*g_Ez)[IMAX][JMAX];
    double (*g_Hx)[IMAX][JMAX];
    double (*g_Hy)[IMAX][JMAX];
    double (*g_Hz)[IMAX][JMAX];
    CHECK_ERROR(cudaMalloc((void**)&g_Ex, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Ey, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Ez, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hx, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hy, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hz, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double)));

    double Ca,Cb,Da,Db;

    Ca = (1-((sigma*dt)/(2*eps)))/(1+((sigma*dt)/(2*eps)));
    Cb = (dt/(eps*delta))/(1+((sigma*dt)/(2*eps)));
    Da = (1-((sigma*dt)/(2*mu)))/(1+((sigma*dt)/(2*mu)));
    Db = (dt/(mu*delta))/(1+((sigma*dt)/(2*mu)));

    CHECK_ERROR(cudaEventCreate(&start_event));
    CHECK_ERROR(cudaEventCreate(&stop_event));
    CHECK_ERROR(cudaEventRecord(start_event, 0));

    CHECK_ERROR(cudaMemcpy(g_Ex, Ex, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Ey, Ey, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Ez, Ez, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hx, Hx, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hy, Hy, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hz, Hz, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32);
    dim3 numBlocks((KMAX + 31) / 32);

    dim3 singleThread(1);
    dim3 singleBlock(1);

    for (n = 0; n < nmax; n++) {
      // loop 1
      loop1_GPU<<<numBlocks, threadsPerBlock>>>(g_Ex, g_Ey, g_Ez, g_Hy, g_Hz, Cb, Ca, n, no, nhalf);
      CHECK_ERROR(cudaPeekAtLastError());
      CHECK_ERROR(cudaDeviceSynchronize());
      // error checking  
      //CHECK_ERROR(cudaMemcpy(Ez, g_Ez, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
      //CHECK_ERROR(cudaPeekAtLastError());
      //printf("%d EZ: %.12f\t%.12f\n", counter++, /*Ez[(JMAX*KMAX)+(KMAX)], Ez[(JMAX*KMAX)+(KMAX)+1]); */ Ez[((IMAX/2)*JMAX*KMAX)+((JMAX/2)*KMAX)+(KMAX/2)], 0.0);
      // loop 2
      loop2_GPU<<<numBlocks, threadsPerBlock>>>(g_Ez, g_Hx, g_Hy, g_Hz, Da, Db);
      CHECK_ERROR(cudaPeekAtLastError());
      CHECK_ERROR(cudaDeviceSynchronize());
    }

    CHECK_ERROR(cudaMemcpy(Ex, g_Ex, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Ey, g_Ey, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Ez, g_Ez, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hx, g_Hx, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hy, g_Hy, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hz, g_Hz, (IMAX+1) * (JMAX+1) * (KMAX+1) * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(g_Ex));
    CHECK_ERROR(cudaFree(g_Ey));
    CHECK_ERROR(cudaFree(g_Ez));
    CHECK_ERROR(cudaFree(g_Hx));
    CHECK_ERROR(cudaFree(g_Hy));
    CHECK_ERROR(cudaFree(g_Hz));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    printf("GPU Time: %.2f\n", elapsed_time);

    FILE * fPointer;
    fPointer = fopen("parlleloutput.dat", "w");
    char buf[18];
    int x, y, z;
    for (x=1; x<IMAX; x++) {
      for (y=1; y<JMAX; y++) {
	for (z=0; z<KMAX; z++) {
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Ex[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Ey[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Ez[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Hx[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Hy[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
	  memset(buf, 0, 18);
	  sprintf(buf, "%e\n", Hz[(x*JMAX*KMAX) + (y*KMAX) + z]);
	  fputs(buf, fPointer);
          memset(buf, 0, 18);
	  sprintf(buf, "                 \n");
	  fputs(buf, fPointer);
	}
      }
    }
    fclose(fPointer);

    return 0;
}
