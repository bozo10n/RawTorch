# include <iostream>
# define N 10000 * 33
# define blocksPerGrid 32

# define threadsPerBlock 256

__global__ void matMult(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while(id < N)
    {
        temp += a[id] * b[id];
        id += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;

    while(i != 0)
    {
        if(cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();

        i = i / 2;
    }

    if(cacheIndex == 0)
    {
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    float *A, *B, *partial_c;

    float *device_a, *device_b, *device_c;

    float C = 0;

    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&device_a, N * sizeof(float));
    cudaMalloc((void**)&device_b, N * sizeof(float));
    cudaMalloc((void**)&device_c, blocksPerGrid * sizeof(float));

    for(int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    cudaMemcpy(device_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    matMult<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c);

    cudaMemcpy(partial_c, device_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < blocksPerGrid; i++)
    {
        C += partial_c[i];
    }

    printf("result: %f\n", C);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(A);
    free(B);
    free(partial_c);
    return 0;
}