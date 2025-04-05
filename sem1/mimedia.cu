#include <stdio.h>
#define N 10

////////////////////
/*
 * MI MEDIA
 */
////////////////////
#include <stdio.h>
#define N 10

void media(double *a, double *c)
{
  int i;
  for (i = 0; i < N - 2; i++)
    c[i] = (a[i] + a[i + 1] + a[i + 2]) / 3.0f;
}

__global__ void mediakernel(double *a, double *c)
{
  int tid = blockIdx.x; // indice local coincide con global
  while (tid < N - 2)
  {
    c[tid] = (a[tid] + a[tid + 1] + a[tid + 2]) / 3.0f;
    // tid+=N; // podia hacer un if mas sencillo, o += gridDim.x porque cada hilo hace lo suyo y sale
    tid += gridDim.x;
  }
}

int main()
{
  double a[N], c[N], c_host[N];
  double *dev_a, *dev_c;
  int i;

  // rellenar vectores en CPU
  for (i = 0; i < N; i++)
  {
    a[i] = i * i;
  }

  media(a, c);

  for (i = 0; i < N - 2; i++)
    printf("  %f\n", c[i]);

  cudaMalloc((void **)&dev_a, N * sizeof(double));
  cudaMalloc((void **)&dev_c, (N - 2) * sizeof(double));

  cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c_host, (N - 2) * sizeof(double), cudaMemcpyHostToDevice);

  mediakernel<<<N - 2, 1>>>(dev_a, dev_c);

  cudaMemcpy(c_host, dev_c, (N - 2) * sizeof(double), cudaMemcpyDeviceToHost);

  for (i = 0; i < N - 2; i++)
    printf("  %f\n", c_host[i]);

  cudaFree(dev_a);
  cudaFree(dev_c);
}
