
#include <stdio.h>
#define N 12
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

void comprobar_cpu(int *A, int *sal)
{
   int i, j, res = 1;
   for (j = 0; j < N - 1; j++)
      for (i = 0; i < N - 1; i++)
         if (A[i + j * N] != A[i + 1 + (j + 1) * N])
            res = 0;

   *sal = res;
}

int comprobar_cpu_vector(int *sal)
{
   int i, res = 1;
   for (i = 0; i < N-1; i++)
      if (sal[i] != sal[i + 1])
         res = 0;

   return res;
}

__global__ void kernel(int *A, int *sal)
{
   int i;
   __shared__ int cache[BLOCKSIZE];
   int tid = threadIdx.x;
   int cacheindex = threadIdx.x;
   int col = blockIdx.x;
   int res = 1;
   while (tid < N - 1 && col < N - 1)
   {
     if ((A[tid + col * N] != A[tid + 1 + (col+1) * N]))
       res=0;
     tid += BLOCKSIZE;
   }
   cache[cacheindex] = res;
   __syncthreads();
   i = blockDim.x / 2;
   while (i != 0)
   {
      if (cacheindex < i){
         if ((cache[cacheindex + i] != cache[cacheindex])){
           cache[cacheindex] = 0;
         }
       }
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
   int *sal = (int *)malloc(N * sizeof(int));
   int salcpu;

   // rellenar matriz de numeros en CPU
   for (j = 0; j < N; j++)
      for (i = 0; i < N; i++)
      {
         A[i + N * j] = j - i;
      }
   // A[3+N*4]=77;
   Print_matrix(A, N);
   comprobar_cpu(A, &salcpu);
   if (salcpu == 1)
      printf(" \n CPU La matriz es toeplitz \n");
   else
      printf(" \n CPU La matriz no es toeplitz \n");

   // Aqui pon el código para reservar memoria, copiar matriz, llamar kernel, traer resultados,
   //  y lo que sea necesario

   // Comienzo parte GPU
   int *dev_A, *dev_sal;

   cudaMalloc((void **)&dev_A, N * N * sizeof(int));
   cudaMalloc((void **)&dev_sal, N * sizeof(int));
   cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);

   kernel<<<N,BLOCKSIZE>>>(dev_A,dev_sal);

   cudaMemcpy(sal, dev_sal, N * sizeof(int), cudaMemcpyDeviceToHost);

   int res = comprobar_cpu_vector(sal);
   if (res == 1)
      printf(" \n GPU La matriz es toeplitz \n");
   else
      printf(" \n GPU La matriz no es toeplitz \n");

}
