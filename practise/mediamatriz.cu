#include <stdio.h>

#define M 5
#define N 6

void mediasmatrizcpu(double *A, double *sal)
{
    int i, j;
    double suma;
    for (j = 0; j < N; j++)
    {
        suma = 0;
        for (i = 0; i < M; i++)
            suma = suma + A[i + j * M];
        sal[j] = suma / double(M);
    }
}

__global__ void kernel(double *A, double *sal)
{
    int i;
    int tid = blockIdx.x;
    double suma;
    suma = 0;
    for (i = 0; i < M; i++)
        suma = suma + A[i + tid * M];
    sal[tid] = suma / double(M);
}

void Print_matrix(double C[], int m, int n)
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf("%.2e ", C[i + j * m]);
        printf("\n");
    }
} /* Print_matrix */

int main()
{
    int i, j;

    double *A = (double *)malloc(N * M * sizeof(double));
    double *sal1 = (double *)malloc(N * sizeof(double));

    // rellenar matriz en CPU
    for (j = 0; j < N; j++)
        for (i = 0; i < M; i++)
        {
            A[i + M * j] = i + j;
        }

    Print_matrix(A, M, N);

    mediasmatrizcpu(A, sal1);

    for (j = 0; j < N; j++)
        printf("media columna %d = %f  \n", j, sal1[j]);



    double *dev_a, *dev_sal11;
    double *sal11 = (double *)malloc(N * sizeof(double));
    cudaMalloc((void **)&dev_a, M * N * sizeof(double));
    cudaMalloc((void **)&dev_sal11, N * sizeof(double));

    cudaMemcpy(dev_a, A, M * N * sizeof(double), cudaMemcpyHostToDevice);

    kernel<<<M,1>>>(dev_a, dev_sal11);

    cudaMemcpy(sal11, dev_sal11, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (j = 0; j < N; j++)
        printf("media columna %d = %f  \n", j, sal11[j]);

    free(A);
    free(sal1);
    free(sal11);
    cudaFree(dev_sal11);
    cudaFree(dev_a);
}
