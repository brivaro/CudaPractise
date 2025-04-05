#include <stdio.h>

#define N 16
#define BLOCKSIZE 4

void Print_matrix(int C[], int n)
{
   int i, j;

   for (i = 0; i < n; i++)
   {
      for (j = 0; j < n; j++)
         printf("%d ", C[i + j * n]);
      printf("\n");
   }
} /* Print_matrix */

void contar_int(int *A, int *sal, int num1, int num2)
{
   int i, j, cant = 0;
   for (j = 0; j < N; j++)
      for (i = 0; i < N - 1; i++)
         if ((A[i + j * N] == num1) && (A[i + 1 + j * N] == num2))
            cant++;

   *sal = cant;
}

__global__ void kernel(int *A, int *sal, int num1, int num2)
{
   int cant = 0;
   int i = 0;
   int tid = blockIdx.x;
   for (i = 0; i < N - 1; i++)
      if ((A[i + tid * N] == num1) && (A[i + 1 + tid * N] == num2))
         cant++;

   sal[tid] = cant;
}

__global__ void kernel2(int *A, int *sal, int num1, int num2)
{
   int cant = 0;
   __shared__ int cache[BLOCKSIZE];
   int tid = threadIdx.x;
   int col = blockIdx.x;
   int cacheindex = threadIdx.x;
   while (tid < N)
   {
      if ((A[tid + col * N] == num1) && (A[tid + 1 + col * N] == num2))
         cant++;
      tid += BLOCKSIZE;
   }
   cache[cacheindex] = cant;
   __syncthreads();
   int i = blockDim.x / 2;
   while (i != 0)
   {
      if (cacheindex < i)
         cache[cacheindex] += cache[cacheindex + i];
      __syncthreads();
      i = i / 2;
   }
   if (cacheindex == 0)
      sal[col] = cache[0];
}

int main()
{

   int i, j;

   int *A = (int *)malloc(N * N * sizeof(int));
   int salcpu;

   // rellenar matriz de caracteres en CPU
   for (j = 0; j < N; j++)
      for (i = 0; i < N; i++)
      {
         A[i + N * j] = rand() % 10;
      }
   Print_matrix(A, N);
   contar_int(A, &salcpu, 6, 3);
   printf(" \n En cpu se cuentan %d secuencias %d %d ", salcpu, 6, 3);

   // Aqui pon el código para reservar memoria, copiar matriz, llamar kernel, traer resultados,
   //  y lo que sea necesario

   int *dev_a, *dev_sal;
   int *sal = (int *)malloc(N * sizeof(int)); // variable para copiar resultado de gpu a cpu
   cudaMalloc((void **)&dev_a, N * N * sizeof(int));
   cudaMalloc((void **)&dev_sal, N * sizeof(int));

   cudaMemcpy(dev_a, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_sal, sal, N * sizeof(int), cudaMemcpyHostToDevice);

   kernel<<<N, 1>>>(dev_a, dev_sal, 6, 3);

   cudaMemcpy(sal, dev_sal, N * sizeof(int), cudaMemcpyDeviceToHost);

   int res = 0;
   for (i = 0; i < N; i++)
      res += sal[i];

   printf(" \n En gpu se cuentan %d secuencias %d %d ", res, 6, 3);

   ////////////////////////////////////////////
   // version 2
   
   int *dev_A, *dev_SAL;
   int *SAL = (int *)malloc(N * sizeof(int)); // variable para copiar resultado de gpu a cpu
   cudaMalloc((void **)&dev_A, N * N * sizeof(int));
   cudaMalloc((void **)&dev_SAL, N * sizeof(int));

   cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_SAL, SAL, N * sizeof(int), cudaMemcpyHostToDevice);

   kernel2<<<N, BLOCKSIZE>>>(dev_A, dev_SAL, 6, 3);

   cudaMemcpy(SAL, dev_SAL, N * sizeof(int), cudaMemcpyDeviceToHost);

   res = 0;
   for (i = 0; i < N; i++)
      res += SAL[i];

   printf(" \n En gpu se cuentan %d secuencias %d %d ", res, 6, 3);
}
