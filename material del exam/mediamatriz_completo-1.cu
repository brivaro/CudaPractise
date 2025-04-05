#include <stdio.h>

#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }

#define N 10240
#define M 1024
#define	BLOCKSIZE 32



__global__ void mediasmatriznaive(double *A,  double *sal)
{
  int i,tid=blockIdx.x; 
  double suma=0;
  if (tid <M) {
    for(i=0;i<N;i++)
       suma=suma+A[i+tid*N];
    sal[tid]=suma/double(N);
  }
 }

__global__ void mediasmatrizfast(double *A,  double *sal)
{
__shared__ double cache[BLOCKSIZE];
int tid=threadIdx.x;
int col=blockIdx.x;
int cacheindex=threadIdx.x;
double suma=0.0,temp=0.0;
while (tid<N)
{temp +=A[tid+col*N];
tid+= BLOCKSIZE;
}
cache[cacheindex]=temp; 
__syncthreads();
int i=blockDim.x/2;
while (i!=0)
{ if (cacheindex<i)
   cache[cacheindex]+=cache[cacheindex+i];
    __syncthreads();
   i=i/2;
}
if (cacheindex==0)
sal[blockIdx.x]=cache[0]/N;
 }
 
 int main() {
  int i,j;
 double *dev_A,  *dev_sal;
 
  double *A = (double *) malloc( N*M*sizeof(double) );
  double *sal1 = (double *) malloc( M*sizeof(double) );
double *sal2 = (double *) malloc( M*sizeof(double) );
 //reservar memoria en GPU
 cudaMalloc((void **) &dev_A, N*M*sizeof(double) );
 cudaMalloc((void **) &dev_sal, M*sizeof(double) );
 
 //rellenar matriz en CPU
  for (j=0;j<M;j++)
    for(i=0;i<N;i++)
   {
      A[i+N*j]=2.0f * ( (double) rand() / RAND_MAX ) ;
    }
//enviar vectores a GPU
cudaMemcpy( dev_A, A, N*M*sizeof(double) , cudaMemcpyHostToDevice );


//cudaMemcpy( dev_sal, sal, N*sizeof(double) , cudaMemcpyHostToDevice );

//llamar al Kernel1
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );
//
CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event

 mediasmatrizfast<<<M,BLOCKSIZE>>>(dev_A,dev_sal);
 //obtener el resultado de vuelta en la CPU
 cudaMemcpy( sal1, dev_sal, M*sizeof(double), cudaMemcpyDeviceToHost );

CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  float msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );
  printf("GPU time1 = %.2f msec.\n",msecGPU);
//
// kernel 2
//
CUDA_SAFE_CALL( cudaEventRecord(start, NULL) ); // Record the start event

 mediasmatriznaive<<<M,1>>>(dev_A,dev_sal);
 //obtener el resultado de vuelta en la CPU
 cudaMemcpy( sal2, dev_sal, M*sizeof(double), cudaMemcpyDeviceToHost );

  CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );  // Record the stop event
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );   // Wait for the stop event to complete
  msecGPU = 0.0f;
  CUDA_SAFE_CALL( cudaEventElapsedTime(&msecGPU, start, stop) );
  printf("GPU time2 = %.2f msec.\n",msecGPU);

//
  double error=fabs(sal1[0]-sal2[0]);
  double aux;
  for (i=1;i<M;i++)
      { aux=fabs(sal1[i]-sal2[i]);
        if (aux>error)
           error=aux;
      }
  printf("error %f  \n",error);
  cudaFree(dev_A) ;
  cudaFree(dev_sal) ;
  free(A);
  free(sal1);
 free(sal2);
  }
	
	
