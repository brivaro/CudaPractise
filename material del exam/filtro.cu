#include <stdio.h>
#define M 8
#define N 12

void Print_matrix(double C[], int n, int m) {
   int i, j;

   for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++)
         printf("%.2e ", C[i+j*n]);
      printf("\n");
   }
}  /* Print_matrix */


__global__ void filtro(double *a, double *b)
{
  int tidx, tidy,tidxp,tidyp;
  tidx=threadIdx.x;
  tidxp=tidx+1;
  tidy=threadIdx.y;
  tidyp=tidy+1;
  if ((tidx <(M-2))&&(tidy<(N-2)))
 {
  b[tidx+tidy*(M-2)]=(a[tidxp+tidyp*M]+a[tidxp-1+tidyp*M]+a[tidxp+1+tidyp*M]+a[tidxp+(tidyp-1)*M]+a[tidxp+(tidyp+1)*M])/5.0f;
  }
 }
 
 int main() {
 double a[N*M], b[(N-2)*(M-2)];
 double *dev_a, *dev_b;
 int i,j;
 //reservar memoria en GPU
 cudaMalloc((void **) &dev_a, N*M*sizeof(double) );
 cudaMalloc((void **) &dev_b, (N-2)*(M-2)*sizeof(double) );
 
 //rellenar vectores en CPU
  for (i=0;i<M;i++)
  for (j=0;j<N;j++)
   {
     a[i+j*M]=i+j;
     
    }
Print_matrix(a,M,N);
printf("esta era A \n");
//enviar vectores a GPU
cudaMemcpy( dev_a, a, N*M*sizeof(double) , cudaMemcpyHostToDevice );
//cudaMemcpy( dev_b, b, (N-2)*(M-2)*sizeof(double) , cudaMemcpyHostToDevice );
//cudaMemcpy( dev_c, c, N*sizeof(int) , cudaMemcpyHostToDevice );
//dim3 block_p_grd(2,3);
dim3 thr_p_block((M-2),(N-2));
//llamar al Kernel
 filtro<<<1,thr_p_block>>>(dev_a,dev_b);
 //obtener el resultado de vuelta en la CPU
 cudaMemcpy( b, dev_b, (N-2)*(M-2)*sizeof(double), cudaMemcpyDeviceToHost );

Print_matrix(b,M-2,N-2);
printf("esta era b \n");

  cudaFree(dev_a) ;
  cudaFree(dev_b) ;
  
  }
	
	
