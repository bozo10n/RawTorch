# include <iostream>
# define N 100
# define blocksPerGrid 32

# define threadsPerBlock 256

__global__ void vector_sum(float *a, float *b, float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while(tid < N)
    {
        c[tid] = b[tid] + a[tid];
        tid += blockDim.x * gridDim.x;
    }
}



int main(void)
{
    float *a, *b, *c;

    float *device_a, *device_b, *device_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(N * sizeof(float));

    cudaMalloc((void**)&device_a, N * sizeof(float));
    cudaMalloc((void**)&device_b, N * sizeof(float));
    cudaMalloc((void**)&device_c, N * sizeof(float));

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_sum<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c);

    cudaMemcpy(c, device_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
    {
        printf("%f + %f = %f\n", a[i], b[i], c[i]);
    }
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(a);
    free(b);
    free(c);

    return 0;
}