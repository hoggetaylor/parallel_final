// To compile - gcc -o 3dFDTD FDTD3D.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define IMAX 100;
#define JMAX 100;
#define KMAX 100;


// This was taken from stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPU error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/**
 *  Second half of the total loop circuit.
 */
extern __global__ void loop2_GPU(
        double (*Ez)[IMAX][JMAX], 
        double (*Hx)[IMAX][JMAX], 
        double (*Hy)[IMAX][JMAX], 
        double (*Hz)[IMAX][JMAX],
        double Da, double Db) {
    int i, j;
    int k = blockIdx.x * 32 + threadIdx.x;

    if (k < KMAX && k > 0) {
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
    }
    if (k < KMAX) {
       for (j = 0; j < JMAX-1; j++) {
           for (i = 0; i < IMAX-1; i++) {
               Hz[i][j][k] = Da*Hz[i][j][k] + Db*((Ez[i][j][k] - Ez[i+1][j][k]) + (Ez[i][j+1][k]-Ez[i][j][k]));
           }
       }
    }
}

int main() {
    int nmax = 1000, nhalf = 20, no = nhalf*3;
    int i, j, n,k;
    double c = 2.99792458e8, pi = 3.141592654, sigma = 0, mu = 4.0 * pi * 1.0e-7, eps = 8.85418782e-12;
    double delta = 1e-3;
    double dt = delta/(c*1.41421356237);

    double *Ex, *Ey, *Ez, *Hy, *Hx, *Hz;

    cudaEvent_t start_event, stop_event;
    float elapsed_time;

    Ex = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));
    Ey = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));
    Ez = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));
    Hx = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));
    Hy = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));
    Hz = (double *)calloc((imax+1) * (jmax+1) * (kmax+1), sizeof(double));

    double* g_Ex;
    double* g_Ey;
    double* g_Ez;
    double* g_Hx;
    double* g_Hy;
    double* g_Hz;
    CHECK_ERROR(cudaMalloc((void**)&g_Ex, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Ey, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Ez, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hx, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hy, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&g_Hz, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double)));

    double Ca,Cb,Da,Db;

    Ca = (1-((sigma*dt)/(2*eps)))/(1+((sigma*dt)/(2*eps)));
    Cb = (dt/(eps*delta))/(1+((sigma*dt)/(2*eps)));
    Da = (1-((sigma*dt)/(2*mu)))/(1+((sigma*dt)/(2*mu)));
    Db = (dt/(mu*delta))/(1+((sigma*dt)/(2*mu)));

    CHECK_ERROR(cudaEventCreate(&start_event));
    CHECK_ERROR(cudaEventCreate(&stop_event));
    CHECK_ERROR(cudaEventRecord(start_event, 0));

    CHECK_ERROR(cudaMemcpy(g_Ex, Ex, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Ey, Ey, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Ez, Ez, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hx, Hx, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hy, Hy, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(g_Hz, Hz, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyHostToDevice));

    for (n = 0; n < nmax; n++) {
        //for (k = 1; k < kmax; k++) {
        //    for (j = 1; j < jmax; j++) {
        //        for (i = 0; i < imax; i++) {
        //            Ex[i][j][k] = Ca*Ex[i][j][k] + Cb*((Hz[i][j][k] - Hy[i][j-1][k]) + (Hy[i][j][k-1] - Hy[i][j][k]));
        //        }
        //    }
        //}
        //for (k = 1; k < kmax; k++) {
        //    for (j = 0; j < jmax; j++) {
        //        for (i = 1; i < imax; i++) {
        //            Ey[i][j][k] = Ca*Ey[i][j][k] + Cb*((Hz[i-1][j][k] - Hy[i][j][k]) + (Hy[i][j][k] - Hy[i][j][k-1]));
        //        }
        //    }
        //}
        //for (k = 0; k < kmax; k++) {
        //    for (j = 1; j < jmax; j++) {
        //        for (i = 1; i < imax; i++) {
        //            Ez[i][j][k] = Ca*Ez[i][j][k] + Cb*((Hz[i][j][k] - Hy[i-1][j][k]) + (Hy[i][j-1][k] - Hy[i][j][k]));
        //        }
        //    }
        //}
        //Ez[imax/2][jmax/2][kmax/2] = exp(-(pow(((n-no)/(double)nhalf),2.0)));

        dim3 threadsPerBlock(32);
        dim3 numBlocks((kmax + threadsPerBlock.x-1) / threadsPerBlock.x);
        loop2_GPU<<<numBlocks, threadsPerBlock>>>(g_Ez, g_Hx, g_Hy, g_Hz, Da, Db);
    }

    CHECK_ERROR(cudaMemcpy(Ex, g_Ex, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Ey, g_Ey, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Ez, g_Ez, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hx, g_Hx, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hy, g_Hy, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(Hz, g_Hz, (imax+1) * (jmax+1) * (kmax+1) * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(g_Hx));
    CHECK_ERROR(cudaFree(g_Hy));
    CHECK_ERROR(cudaFree(g_Hz));
    CHECK_ERROR(cudaFree(g_Ez));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    printf("GPU Time: %.2f\n", elapsed_time);

    return 0;
}
