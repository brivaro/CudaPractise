#include <stdio.h>
#define N 10

__global__ void add(int *a, int *b, int *c)
{
  int tid=blockIdx.x; // tid = threadIdx.x + blockIdx.x * blockDim.x 
  if (tid <N) {
  c[tid]=a[tid]+b[tid];
  }
 }
 
 int main() {
 int a[N], b[N], c[N];
 int *dev_a, *dev_b, *dev_c,i;
 //reservar memoria en GPU
 cudaMalloc((void **) &dev_a, N*sizeof(int) );
 cudaMalloc((void **) &dev_b, N*sizeof(int) );
 cudaMalloc((void **) &dev_c, N*sizeof(int) );
 //rellenar vectores en CPU
  for (i=0;i<N;i++)
   {
     a[i]=-i;
     b[i]=i*i;
    }
//enviar vectores a GPU
cudaMemcpy( dev_a, a, N*sizeof(int) , cudaMemcpyHostToDevice );
cudaMemcpy( dev_b, b, N*sizeof(int) , cudaMemcpyHostToDevice );
cudaMemcpy( dev_c, c, N*sizeof(int) , cudaMemcpyHostToDevice );

//llamar al Kernel
 add<<<N,1>>>(dev_a,dev_b,dev_c);
 //obtener el resultado de vuelta en la CPU
 cudaMemcpy( c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost );
 for (i=0;i<N;i++)
  printf(" %d + %d = %d\n", a[i],b[i], c[i]);
  
  cudaFree(dev_a) ;
  cudaFree(dev_b) ;
  cudaFree(dev_c) ;
  }
	
	
