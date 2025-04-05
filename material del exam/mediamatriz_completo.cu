#include <stdio.h>


#define M 5
#define N 6


__global__ void mediasmatrizgpu(double *A,  double *sal)
{
  int i; 
  int tid=threadIdx.x;
  double suma;

    suma=0;
    for(i=0;i<M;i++)
       suma=suma+A[i+tid*M];
    sal[tid]=suma/double(M);
   
 }

void Print_matrix(double C[], int m, int n) {
   int i, j;

   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%.2e ", C[i+j*m]);
      printf("\n");
   }
}  /* Print_matrix */
 
 int main() {
  int i,j;
double *dev_A, *dev_sal1;
 
  double *A = (double *) malloc( N*M*sizeof(double) );
  double *sal1 = (double *) malloc( N*sizeof(double) );
cudaMalloc((void **) &dev_A, M*N*sizeof(double) );
 cudaMalloc((void **) &dev_sal1, N*sizeof(double) );
 
 //rellenar matriz en CPU
  for (j=0;j<N;j++)
    for(i=0;i<M;i++)
   {
      A[i+M*j]=i+j ;
    }

  Print_matrix(A,M,N);
cudaMemcpy( dev_A, A, M*N*sizeof(double), cudaMemcpyHostToDevice );
  mediasmatrizgpu<<<1,N>>>(dev_A,dev_sal1);
cudaMemcpy( sal1, dev_sal1, N*sizeof(double), cudaMemcpyDeviceToHost );
   for (j=0;j<N;j++)
    printf("media columna %d = %f  \n",j,sal1[j]);

  free(A);
  free(sal1);

  }
	
	
