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
 
 int main() {
 float A[N*M], C[(N-2)*(M-2)];
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

  }
	
	
