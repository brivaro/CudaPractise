#include <stdio.h>
#define N 8 //columnas
#define M 12 //filas


void Print_matrix(float C[], int n, int m) {
   int i, j;

   for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++)
         printf("%.2e ", C[i+j*n]);
      printf("\n");
   }
}  /* Print_matrix */

void filtrocpu(float *a, float *c)
{
	int i,j;
	for (i=0;i<M-2;i++)
	  for (j=0;j<N-2;j++)
	   c[i+j*(M-2)]=(a[i+(j+1)*M]+a[i+1+(j+1)*M]+a[i+2+(j+1)*M]+a[i+1+(j)*M]+a[i+1+(j+2)*M])/5.0;
	
}

__global__ void filtrogpu(float *a, float *c)
{
   int tidx = threadIdx.x + blockIdx.x * blockDim.x; //indice local
   int tidy= threadIdx.y + blockIdx.y*blockDim.y;
   c[tidx+tidy*(M-2)]=(a[tidx+(tidy+1)*M]+a[tidx+1+(tidy+1)*M]+a[tidx+2+(tidy+1)*M]+a[tidx+1+(tidy)*M]+a[tidx+1+(tidy+2)*M])/5.0;
     
}

 int main() {
 float A[N*M], C[(N-2)*(M-2)], c_host[(N-2)*(M-2)];
 float *dev_a, *dev_c;
 int i,j;

  for (i=0;i<M;i++)
  for (j=0;j<N;j++)
   {
     A[i+j*M]=i+j;
     
    }
printf("A \n");
Print_matrix(A,M,N);
filtrocpu(A,C);

printf("\n C \n");
Print_matrix(C,M-2,N-2);

cudaMalloc((void **) &dev_a, M*N*sizeof(float) );
cudaMalloc((void **) &dev_c, (M-2)*(N-2)*sizeof(float) );

cudaMemcpy(dev_a, A, M*N*sizeof(float), cudaMemcpyHostToDevice);

dim3 threadsXblock(M-2,N-2);

filtrogpu<<<1,threadsXblock>>>(dev_a,dev_c);

cudaMemcpy(c_host, dev_c,  (M-2)*(N-2)*sizeof(float) , cudaMemcpyDeviceToHost);

printf("\n C GPU \n");
Print_matrix(c_host,M-2,N-2);


  }
	
