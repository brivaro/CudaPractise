#include <stdio.h>
#define N 8  // filas
#define M 12 // columnas

__global__ void add(int *a, int *b, int *c)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   c[tid] = a[tid] + b[tid];
}

void Print_matrix(int C[], int n, int m)
{
   int i, j;

   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
         printf("%d ", C[i + j * m]);
      printf("\n");
   }
   printf("\n");
} /* Print_matrix */

int main()
{
   int a[N * M], b[N * M], c[N * M];
   int *dev_a, *dev_b, *dev_c, i, j;
   // reservar memoria en GPU
   cudaMalloc((void **)&dev_a, N * M * sizeof(int));
   cudaMalloc((void **)&dev_b, N * M * sizeof(int));
   cudaMalloc((void **)&dev_c, N * M * sizeof(int));
   // rellenar matriz en CPU
   for (j = 0; j < M; j++)
   {
      for (i = 0; i < N; i++)
      {
         a[i + N * j] = i + j;
         b[i + N * j] = i + j;
      }
   }
   Print_matrix(a, N, M);
   Print_matrix(b, N, M);
   // enviar vectores a GPU
   cudaMemcpy(dev_a, a, N * M * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, N * M * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_c, c, N * M * sizeof(int), cudaMemcpyHostToDevice);

   // llamar al Kernel
   add<<<M, N>>>(dev_a, dev_b, dev_c);
   // obtener el resultado de vuelta en la CPU
   cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);
   Print_matrix(c, N, M);

   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFree(dev_c);
}
