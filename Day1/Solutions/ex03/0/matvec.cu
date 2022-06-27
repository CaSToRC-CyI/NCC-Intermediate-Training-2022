#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

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

    double n_flop = 2*m*n;
    double n_io = sizeof(float)*(m*n + n + m);
#pragma omp parallel
    {
#pragma omp single
      {
	int nthr = omp_get_num_threads();
	printf(" CPU: nthr = %4d   t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	       nthr, t0, n_flop/1e9/t0, n_io/1e9/t0);
      }
    }
  }

  /*
   * B: Run Ax(), return to y1, report performance
   */
  {
    double t0 = stop_watch(0);
    Ax(m, n, y1, A, x);
    t0 = stop_watch(t0);

    double n_flop = 2*m*n;
    double n_io = sizeof(float)*(m*n + n + m);
#pragma omp parallel
    {
#pragma omp single
      {
	int nthr = omp_get_num_threads();
	printf(" CPU: nthr = %4d   t0 = %6.4lf sec   P = %7.3lf Gflop/s   B = %7.3lf GB/s\n",
	       nthr, t0, n_flop/1e9/t0, n_io/1e9/t0);
      }
    }
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
