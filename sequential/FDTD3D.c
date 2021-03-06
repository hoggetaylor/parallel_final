// To compile - gcc -o 3dFDTD FDTD3D.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

int main() { 
    int imax = 100, jmax = 100, kmax = 100, nmax = 400, nhalf = 20, no = (nhalf*3);
    int i, j, n,k;
    double c = 2.99792458e8, pi = 3.141592654, sigma = 0, mu = 4.0 * pi * 1.0e-7, eps = 8.85418782e-12;
    double delta = 1e-3;
    double dt = delta/(c*1.41421356237);
    int counter = 0;

    double ***Ex, ***Ey, ***Ez, ***Hy, ***Hx, ***Hz;

    struct timeval tstart,tend;
    int sec,usec;

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


    if(gettimeofday(&tstart, NULL) != 0){
        printf("Error in gettimeofday()\n");
        exit(1);
    }

    for (n = 0; n < nmax; n++) {
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
        //printf("%d EZ %.12f\t%.12f\n", counter++, /*Ez[1][1][0], Ez[1][1][1]); */Ez[imax/2][jmax/2][kmax/2], 0.0);
        for (k = 0; k < kmax; k++) {
           for (j = 0; j < jmax-1; j++) {
               for (i = 1; i < imax-1; i++) {
                   Hx[i][j][k] = Da*Hx[i][j][k] + Db*((Ez[i][j][k] - Ez[i][j+1][k]) + (Ez[i][j][k+1]-Ez[i][j][k]));
               }
           }
        }
        for (k = 0; k < kmax; k++) {
           for (j = 1; j < jmax-1; j++) {
               for (i = 0; i < imax-1; i++) {
                   Hy[i][j][k] = Da*Hy[i][j][k] + Db*((Ez[i+1][j][k] - Ez[i][j][k]) + (Ez[i][j][k]-Ez[i][j][k+1]));
               }
           }
        }
        for (k = 1; k < kmax; k++) {
           for (j = 0; j < jmax-1; j++) {
               for (i = 0; i < imax-1; i++) {
                   Hz[i][j][k] = Da*Hz[i][j][k] + Db*((Ez[i][j][k] - Ez[i+1][j][k]) + (Ez[i][j+1][k]-Ez[i][j][k]));
               }
           }
        }
    }

    if(gettimeofday(&tend, NULL) != 0){
        printf("Error in gettimeofday()\n");
        exit(1);
    }

    FILE * fPointer;
    fPointer = fopen("sequentialoutput.dat","w");
    int x, y, z;
    for (x=1; x<imax; x++) {
      for (y=1; y<jmax; y++) {
        for (z=0; z<kmax; z++) {
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Ex[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Ey[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Ez[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Hx[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Hy[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "%e\n", Hz[x][y][z]);
          fputs(buf, fPointer);
          memset(buf, 0, 18);
          sprintf(buf, "                 \n");
          fputs(buf, fPointer);
        }
      }
    }
    fclose(fPointer);

    // calculate run time
    if(tstart.tv_usec > tend.tv_usec){
        tend.tv_usec += 1000000;
        tend.tv_sec--;
    }
    usec = tend.tv_usec - tstart.tv_usec;
    sec = tend.tv_sec - tstart.tv_sec;

    printf("Run Time for run = %d sec and %d usec\n",sec,usec);
    return 0;
}
