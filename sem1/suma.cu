#include <stdio.h>
__global__ void suma(int a, int b, int *c)
{
   *c=a+b;
}
int main()
{
	int c;
	int *dev_c;
	cudaMalloc( (void**)&dev_c,sizeof(int) );
	suma<<<1,1>>>(2,7,dev_c);
	cudaMemcpy( &c,dev_c, sizeof(int),cudaMemcpyDeviceToHost);
	printf("2+7 = %d\n",c);
	cudaFree(dev_c);
	return 0;
}
