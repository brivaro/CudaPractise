#include <stdio.h>
#define N 10

__global__ void suma(int *a, int *b, int *c)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x; //indice local
   while(tid < N){
	c[tid]=a[tid]+b[tid];
	tid += gridDim.x * blockDim.x;
   }

}
int main()
{
	int a[N],b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**) &dev_a, N*sizeof(int));
	cudaMalloc((void**) &dev_b, N*sizeof(int));
	cudaMalloc((void**) &dev_c, N*sizeof(int));

	for (int i=0;i<N;i++)
	{
	  a[i]=-i;
	  b[i]=i*i;
	 }

	 for (int i=0;i<N;i++)
       printf(" %d + %d = %d\n", a[i],b[i], a[i]+b[i]);

	 cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	 cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	 suma<<<2,3>>>(dev_a, dev_b, dev_c);

	 cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	 for (int i=0;i<N;i++)
	    printf(" %d + %d = %d\n", a[i],b[i], c[i]);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
