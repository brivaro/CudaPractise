#include <stdio.h>
#define N 8
#define M 12

__global__ void filtro_gpu(float *a, float *c)
{
 // int  tidx=threadIdx.x+blockIdx.x*blockDim.x;
 // int tidy= threadIdx.y + blockIdx.y*blockDim.y;
  int  tidx=threadIdx.x;
  int tidy= threadIdx.y;
 
 
 c[tidx+tidy*(M-2)]=(a[tidx+(tidy+1)*M]+a[tidx+1+(tidy+1)*M]+a[tidx+2+(tidy+1)*M]+a[tidx+1+tidy*M]+a[tidx+1+(tidy+2)*M])/5.0;
 }


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
 
 int main() {
 float A[N*M], C1[(N-2)*(M-2)], C2[(N-2)*(M-2)];

 int i,j;

  for (i=0;i<M;i++)
  for (j=0;j<N;j++)
   {
     A[i+j*M]=i+j;
     
    }
printf("A \n");
Print_matrix(A,M,N);
filtrocpu(A,C1);

printf("\n C cpu \n");
Print_matrix(C1,M-2,N-2);

float *dev_A,  *dev_C;
 //reservar memoria en GPU
 cudaMalloc((void **) &dev_A, M*N*sizeof(float) );
 cudaMalloc((void **) &dev_C, (M-2)*(N-2)*sizeof(float) );
cudaMemcpy( dev_A, A, M*N*sizeof(float) , cudaMemcpyHostToDevice );
//dim3 block_p_grd(2,3);
dim3 thr_p_block(M-2,N-2);

filtro_gpu<<<1, thr_p_block>>>(dev_A,dev_C);
cudaMemcpy( C2, dev_C, (M-2)*(N-2)*sizeof(float), cudaMemcpyDeviceToHost );

printf("\n C gpu \n");
Print_matrix(C2,M-2,N-2);





  }
	
	
