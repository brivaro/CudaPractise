
#include <stdio.h>
#define N 10

__global__ void media(double *A, double *sal)
{
    int tid = blockIdx.x;
    sal[tid] = (A[tid] + A[tid+1] + A[tid+2]) / 3;
}

__global__ void media2(int *A, int *sal)
{
    int tid = blockIdx.x;
    sal[tid] = (A[tid] + A[tid+1] + A[tid+2]) / 3;
}


int main()
{
  double a[N], c[N - 2];
  double *dev_a, *dev_c;
  int i;
  cudaMalloc((void **)&dev_a, N * sizeof(double));
  cudaMalloc((void **)&dev_c, (N - 2) * sizeof(double));
  // rellenar vectores en CPU
  for (i = 0; i < N; i++)
  {
    a[i] = i * i;
  }
  cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);

  media<<<N - 2, 1>>>(dev_a, dev_c);
  cudaMemcpy(c, dev_c, (N - 2) * sizeof(double), cudaMemcpyDeviceToHost);
  for (i = 0; i < N - 2; i++)
    printf("  %f\n", c[i]);

  int aa[N], cc[N];
  for (i = 0; i < N; i++)
  {
    aa[i] = i;
  }

  int *dev_aa, *dev_cc;
  cudaMalloc((void **)&dev_aa, (N) * sizeof(int));
  cudaMalloc((void **)&dev_cc, (N - 2) * sizeof(int));
  cudaMemcpy(dev_aa, aa, N * sizeof(int), cudaMemcpyHostToDevice);
  media2<<<N - 2, 1>>>(dev_aa, dev_cc);
  cudaMemcpy(cc, dev_cc, (N - 2) * sizeof(int), cudaMemcpyDeviceToHost);
  for (i = 0; i < N - 2; i++)
    printf("  %d\n", cc[i]);

  
}