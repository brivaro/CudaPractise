#include <stdio.h>

#define N 8
#define BLOCKSIZE 4

void Print_matrix(int C[])
{
  int i, j;

  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
      printf("%d ", C[i + j * N]);
    printf("\n");
  }
} /* Print_matrix */

int calcula_max_vector(int *A)
{
  int i, maximo;
  maximo = A[0];
  for (i = 1; i < N; i++)
    if ((A[i] > maximo))
      maximo = A[i];

  return maximo;
}

void calcula_max(int *A, int *sal)
{
  int i, j, maximo;
  maximo = A[0];
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if ((A[i + j * N] > maximo))
        maximo = A[i + j * N];

  *sal = maximo;
}

__global__ void kernel(int *A, int *sal)
{
  int i;
  __shared__ int cache[BLOCKSIZE];
  int tid = threadIdx.x;
  int cacheindex = threadIdx.x;
  int col = blockIdx.x;
  int maximo = A[tid + col * N];
  while (tid < N)
  {
    if ((A[tid + col * N] > maximo))
      maximo = A[tid + col * N];
    tid += BLOCKSIZE;
  }
  cache[cacheindex] = maximo;
  __syncthreads();
  i = blockDim.x / 2;
  while (i != 0)
  {
     if (cacheindex < i){
        if ((cache[cacheindex + i] > cache[cacheindex])){
          cache[cacheindex] = cache[cacheindex + i];
        }
      }
     __syncthreads();
     i = i / 2;
  }
  if (cacheindex == 0)
     sal[col] = cache[0];
}

__global__ void kernel2(int *A, int *sal)
{
  int i, tid = blockIdx.x;
  int maximo = A[0 + tid * N];;
  for (i = 0; i < N; i++)
    if ((A[i + tid * N] > maximo))
      maximo = A[i + tid * N];

  sal[tid] = maximo;

}

int main()
{

  int i, j;

  int *A = (int *)malloc(N * N * sizeof(int));
  int salcpu;

  // rellenar matriz de enteros en CPU
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
    {
      A[i + N * j] = rand() % 1000;
    }
  Print_matrix(A);
  calcula_max(A, &salcpu);
  printf(" \n El maximo calculado en cpu es %d ", salcpu);

  // Aqui pon el código para reservar memoria, copiar matriz, llamar kernel, traer resultados,
  //  y lo que sea necesario

  // Comienzo parte GPU

  int *sal = (int *)malloc(N * sizeof(int)); // variable para copiar resultado parcial de gpu a cpu
  // variables para gpu
  int *dev_A;
  int *dev_sal;

  cudaMalloc((void **)&dev_A, N * N * sizeof(int));
  cudaMalloc((void **)&dev_sal, N * sizeof(int));
  cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);

  kernel<<<N, BLOCKSIZE>>>(dev_A, dev_sal);

  cudaMemcpy(sal, dev_sal, N * sizeof(int), cudaMemcpyDeviceToHost);

  int cont = calcula_max_vector(sal);
  printf(" \n El maximo calculado en gpu con cache es %d ", cont);


  ////////////////////////////////////////////////////
  /////////////////////// VERSION 2

  // Comienzo parte GPU

  int *sal2 = (int *)malloc(N * sizeof(int)); // variable para copiar resultado parcial de gpu a cpu
  // variables para gpu
  int *dev_A2;
  int *dev_sal2;

  cudaMalloc((void **)&dev_A2, N * N * sizeof(int));
  cudaMalloc((void **)&dev_sal2, N * sizeof(int));
  cudaMemcpy(dev_A2, A, N * N * sizeof(int), cudaMemcpyHostToDevice);

  kernel2<<<N, 1>>>(dev_A2, dev_sal2);

  cudaMemcpy(sal2, dev_sal2, N * sizeof(int), cudaMemcpyDeviceToHost);

  cont = calcula_max_vector(sal2);
  printf(" \n El maximo calculado en gpu sin cache es %d ", cont);

  free(A);
  free(sal);
  cudaFree(dev_A);
  cudaFree(dev_sal);
  cudaFree(dev_A2);
  cudaFree(dev_sal2);
}
