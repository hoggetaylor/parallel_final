// To compile - gcc -o 3dFDTD FDTD3D.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// This was taken from stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPU error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

extern __global__ void loop4_GPU(double*** Hx, double*** Ez, double Da, double Db, int kmax, int jmax, int imax) {
    int i, j;
    int k = blockIdx.x * 32 + threadIdx.x;
    if (k < kmax) {
        for (j = 0; j < jmax-1; j++) {
            for (i = 1; i < imax-1; i++) {
                Hx[i][j][k] = Da*Hx[i][j][k] + Db*((Ez[i][j][k] - Ez[i][j+1][k]) + (Ez[i][j][k+1]-Ez[i][j][k]));
            }
        }
    }
}

extern __global__ void loop5_GPU(double*** Hy, double*** Ez, double Da, double Db, int kmax, int jmax, int imax) {
    int i, j;
    int k = blockIdx.x * 32 + threadIdx.x;
    if (k < kmax) {
       for (j = 1; j < jmax-1; j++) {
           for (i = 0; i < imax-1; i++) {
               Hy[i][j][k] = Da*Hy[i][j][k] + Db*((Ez[i+1][j][k] - Ez[i][j][k]) + (Ez[i][j][k]-Ez[i][j][k+1]));
           }
       }
    }
}

extern __global__ void loop6_GPU(double*** Hz, double*** Ez, double Da, double Db, int kmax, int jmax, int imax) {
    int i, j;
    int k = (blockIdx.x * 32 + threadIdx.x) + 1; // this loop starts at k=1 so we add 1
    if (k < kmax) {
       for (j = 0; j < jmax-1; j++) {
           for (i = 0; i < imax-1; i++) {
               Hz[i][j][k] = Da*Hz[i][j][k] + Db*((Ez[i][j][k] - Ez[i+1][j][k]) + (Ez[i][j+1][k]-Ez[i][j][k]));
           }
       }
    }
}

int main() {
    printf("Running main\n");
    int imax = 100, jmax = 100, nmax = 1000, nhalf = 20, no = nhalf*3, kmax = 100;
    int i, j, n,k;
    double c = 2.99792458e8, pi = 3.141592654, sigma = 0, mu = 4.0 * pi * 1.0e-7, eps = 8.85418782e-12;
    double delta = 1e-3;
    double dt = delta/(c*1.41421356237);

    double ***Ex, ***Ey, ***Ez, ***Hy, ***Hx, ***Hz;

    //struct timeval tstart,tend;
    //int sec,usec;
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
	double*** g_Hx;
        double*** g_Hy;
        double*** g_Hz;
        double*** g_Ez;
	//fprintf(fPointer, "allocating memory on GPU\n");
        CHECK_ERROR(cudaMalloc((void**)&g_Hx, (imax+1)*sizeof(double**)));
        CHECK_ERROR(cudaMalloc((void**)&g_Hy, (imax+1)*sizeof(double**)));
        CHECK_ERROR(cudaMalloc((void**)&g_Hz, (imax+1)*sizeof(double**)));
        CHECK_ERROR(cudaMalloc((void**)&g_Ez, (imax+1)*sizeof(double**)));
        for(i=0;i<(imax+1);i++) {
            CHECK_ERROR(cudaMalloc((void**)&g_Hx[i], (jmax+1)*sizeof(double*)));
            CHECK_ERROR(cudaMalloc((void**)&g_Hy[i], (jmax+1)*sizeof(double*)));
            CHECK_ERROR(cudaMalloc((void**)&g_Hz[i], (jmax+1)*sizeof(double*)));
            CHECK_ERROR(cudaMalloc((void**)&g_Ez[i], (jmax+1)*sizeof(double*)));
            for(j=0;j<(jmax+1);j++) {
                CHECK_ERROR(cudaMalloc((void**)&g_Hx[i][j], (kmax+1)*sizeof(double)));
                CHECK_ERROR(cudaMalloc((void**)&g_Hy[i][j], (kmax+1)*sizeof(double)));
                CHECK_ERROR(cudaMalloc((void**)&g_Hz[i][j], (kmax+1)*sizeof(double)));
                CHECK_ERROR(cudaMalloc((void**)&g_Ez[i][j], (kmax+1)*sizeof(double)));
            }
        }

    double Ca,Cb,Da,Db;

    Ca = (1-((sigma*dt)/(2*eps)))/(1+((sigma*dt)/(2*eps)));
    Cb = (dt/(eps*delta))/(1+((sigma*dt)/(2*eps)));
    Da = (1-((sigma*dt)/(2*mu)))/(1+((sigma*dt)/(2*mu)));
    Db = (dt/(mu*delta))/(1+((sigma*dt)/(2*mu)));

    FILE * fPointer;
    fPointer = fopen("myoutput3d.dat","w");

    CHECK_ERROR(cudaEventCreate(&start_event));
    CHECK_ERROR(cudaEventCreate(&stop_event));
    CHECK_ERROR(cudaEventRecord(start_event, 0));

    for (n = 0; n < nmax; n++) {
	char buf[18];
	memset(buf, 0, 18);
	sprintf(buf, "inside n loop\n");
	fputs(buf, fPointer);
        for (k = 1; k < kmax; k++) {
            for (j = 1; j < jmax; j++) {
                for (i = 0; i < imax; i++) {
                    Ex[i][j][k] = Ca*Ex[i][j][k] + Cb*((Hz[i][j][k] - Hy[i][j-1][k]) + (Hy[i][j][k-1] - Hy[i][j][k]));
                }
            }
        }
        for (k = 1; k < kmax; k++) {
            for (j = 0; j < jmax; j++) {
                for (i = 1; i < imax; i++) {
                    Ey[i][j][k] = Ca*Ey[i][j][k] + Cb*((Hz[i-1][j][k] - Hy[i][j][k]) + (Hy[i][j][k] - Hy[i][j][k-1]));
                }
            }
        }
        for (k = 0; k < kmax; k++) {
            for (j = 1; j < jmax; j++) {
                for (i = 1; i < imax; i++) {
                    Ez[i][j][k] = Ca*Ez[i][j][k] + Cb*((Hz[i][j][k] - Hy[i-1][j][k]) + (Hy[i][j-1][k] - Hy[i][j][k]));
                }
            }
        }
        Ez[imax/2][jmax/2][kmax/2] = exp(-(pow(((n-no)/(double)nhalf),2.0)));
	 
	fprintf(fPointer, "Copying memory to GPU\n");
        for(i=0;i<(imax+1);i++) {
            for(j=0;j<(jmax+1);j++) {
                CHECK_ERROR(cudaMemcpy(g_Hx[i][j], Hx[i][j], (kmax+1)*sizeof(double), cudaMemcpyHostToDevice));
                CHECK_ERROR(cudaMemcpy(g_Hy[i][j], Hy[i][j], (kmax+1)*sizeof(double), cudaMemcpyHostToDevice));
                CHECK_ERROR(cudaMemcpy(g_Hz[i][j], Hz[i][j], (kmax+1)*sizeof(double), cudaMemcpyHostToDevice));
                CHECK_ERROR(cudaMemcpy(g_Ez[i][j], Ez[i][j], (kmax+1)*sizeof(double), cudaMemcpyHostToDevice));
            }
        }

	fprintf(fPointer, "Running loops on GPU\n");
        dim3 threadsPerBlock(32);
        dim3 numBlocks((kmax + threadsPerBlock.x-1) / threadsPerBlock.x);
        loop4_GPU<<<numBlocks, threadsPerBlock>>>(g_Hx, g_Ez, Da, Db, kmax, jmax, imax);
        loop5_GPU<<<numBlocks, threadsPerBlock>>>(g_Hy, g_Ez, Da, Db, kmax, jmax, imax);
        loop6_GPU<<<numBlocks, threadsPerBlock>>>(g_Hz, g_Ez, Da, Db, kmax, jmax, imax);

	fprintf(fPointer, "Copying results back to host\n");
        for(i=0;i<(imax+1);i++) {
            for(j=0;j<(jmax+1);j++) {
                CHECK_ERROR(cudaMemcpy(Hx[i][j], g_Hx[i][j], (kmax+1)*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_ERROR(cudaMemcpy(Hy[i][j], g_Hy[i][j], (kmax+1)*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_ERROR(cudaMemcpy(Hz[i][j], g_Hz[i][j], (kmax+1)*sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_ERROR(cudaMemcpy(Ez[i][j], g_Ez[i][j], (kmax+1)*sizeof(double), cudaMemcpyDeviceToHost));
            }
        }

	}

	fprintf(fPointer, "Freeing memory on GPU\n");
        for(i=0;i<(imax+1);i++) {
            for(j=0;j<(jmax+1);j++) {
                CHECK_ERROR(cudaFree(g_Hx[i][j]));
                CHECK_ERROR(cudaFree(g_Hy[i][j]));
                CHECK_ERROR(cudaFree(g_Hz[i][j]));
                CHECK_ERROR(cudaFree(g_Ez[i][j]));
            }
            CHECK_ERROR(cudaFree(g_Hx[i]));
            CHECK_ERROR(cudaFree(g_Hy[i]));
            CHECK_ERROR(cudaFree(g_Hz[i]));
            CHECK_ERROR(cudaFree(g_Ez[i]));
        }
        CHECK_ERROR(cudaFree(g_Hx));
        CHECK_ERROR(cudaFree(g_Hy));
        CHECK_ERROR(cudaFree(g_Hz));
        CHECK_ERROR(cudaFree(g_Ez));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    fclose(fPointer);

    printf("GPU Time: %.2f\n", elapsed_time);

    return 0;
}
