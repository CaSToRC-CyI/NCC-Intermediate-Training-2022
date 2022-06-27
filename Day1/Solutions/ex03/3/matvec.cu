#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_THR 1024
/***
 * Print usage
 ***/
void
usage(char *argv[])
{
  fprintf(stderr, "usage: %s M N\n", argv[0]);
  return;
}

/***
 * Allocate memory; print error if NULL is returned
 ***/
void *
ualloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, "malloc() returned null; quitting...\n");
    exit(-2);
  }
  return ptr;
}


/***
 * Allocate memory on GPU; print error if not successful
 ***/
void *
gpu_alloc(size_t size)
{
  void *ptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if(err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() returned %d; quitting...\n", err);
    exit(-2);
  } 
  return ptr;
}

/***
 * Return a random number in [0, 1)
 ***/
double
urand(void)
{
  double x = (double)rand()/(double)RAND_MAX;
  return x;
}

/***
 * Return seconds elapsed since t0, with t0 = 0 the epoch
 ***/
double
stop_watch(double t0)
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec/1e6 - t0;
}

/***
 * Do y <- A*x on the CPU using OpenMP, y: m, A: mxn, x: n
 ***/
void
Ax(int m, int n, float *y, float *A, float *x)
{
#pragma omp parallel for
  for(int i=0; i<m; i++) {
    y[i] = 0.;
    for(int j=0; j<n; j++)
      y[i] += A[i*n + j]*x[j];
  }
  return;
}

/***
 * Do y <- A*x on the GPU using CUDA, y: m, A: mxn, x: n
 ***/
__global__ void
gpu_Ax(int m, int n, float *y, float *A, float *x)
{
  int nx = blockDim.x;
  int ny = blockDim.y;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int nb = n/nx;
  float yb = 0;
  __shared__ float Ab[MAX_THR];
  __shared__ float xb[MAX_THR];

  for(int k=0; k<nb; k++) {
    Ab[ty*nx + tx] = A[(ty+ny*by)*n + (tx+nx*k)];
    if(ty == 0)
      xb[tx] = x[tx + nx*k];
    __syncthreads();
    
    if(tx == 0) {
      for(int i=0; i<nx; i++)
	yb += Ab[ty*nx+i]*xb[i];
    }
    __syncthreads();
  }

  if(tx == 0)
    y[ty + by*ny] = yb;
  return;
}

int
main(int argc, char *argv[])
{
  /*
   * If number of arguments are not as expected, print usage and exit
   */
  if(argc != 3) {
    usage(argv);
    return 1;
  }

  unsigned long int m = atol(argv[1]);
  unsigned long int n = atol(argv[2]);

  float *x = (float *)ualloc(sizeof(float)*n);
  float *A = (float *)ualloc(sizeof(float)*n*m);
  float *y0 = (float *)ualloc(sizeof(float)*m);
  float *y1 = (float *)ualloc(sizeof(float)*m);

  /*
   * Initialize a and arrays
   */
  srand(2147483647);
  for(int i=0; i<n; i++) {
    x[i] = urand();
    for(int j=0; j<m; j++)
      A[i*m + j] = urand();
  }

  /*
   * A: Run Ax(), return to y0, report performance
   */
  {
    double t0 = stop_watch(0);
    Ax(m, n, y0, A, x);
    t0 = stop_watch(t0);

    double n_flop = 2.0*n*m /* TODO */;
    double n_io = sizeof(float)*(n*m + m + n) /* TODO */;
#pragma omp parallel
    {
#pragma omp single
      {
	int nthr = omp_get_num_threads();
	printf(" CPU: nthr = %4d        t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	       nthr, t0, n_flop/1e9/t0, n_io/1e9/t0);
      }
    }
  }

  /*
   * B: Run Ax(), return to y1, report performance
   */
  {
    /* Allocate GPU memory */
    float *d_x = (float *)gpu_alloc(n*sizeof(float));
    float *d_y = (float *)gpu_alloc(m*sizeof(float));
    float *d_A = (float *)gpu_alloc(m*n*sizeof(float));

    cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, sizeof(float)*n*m, cudaMemcpyHostToDevice);
    
    double t0 = stop_watch(0);
    int tx = 16;
    int ty =  8;
    dim3 blck(1, m/ty, 1);
    dim3 thrd(tx, ty, 1);
    gpu_Ax<<<blck, thrd>>>(m, n, d_y, d_A, d_x);
    cudaDeviceSynchronize();
    t0 = stop_watch(t0);

    cudaMemcpy(y1, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);    

    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_A);
    
    double n_flop = 2.0*n*m /* TODO */;
    double n_io = sizeof(float)*(n*m + m + n) /* TODO */;
    printf(" GPU: nthr = (%3d,%3d)   t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	   tx, ty, t0, n_flop/1e9/t0, n_io/1e9/t0);
  }

  
  /* Compare y1 and y0 */
  double diff = 0;
  double norm = 0;
  for(int i=0; i<m; i++) {
    double d = y0[i]-y1[i];
    diff += d*d;
    norm += y0[i]*y0[i];
  }
  printf(" Diff = %e\n", diff/norm);
  /*
   * Don't need arrays anymore
   */
  free(x);
  free(A);
  free(y0);
  free(y1);
  return 0;
}
