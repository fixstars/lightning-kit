
#include <stdint.h>
#include <stdlib.h>

#include <iostream>

#include <cuda_runtime.h>

__global__ void test_memcpyasync(int64_t* res, int64_t* tar, int64_t* dst, size_t iter, size_t chunk, size_t chunk_num)
{

    int64_t start_time, end_time;
    volatile int64_t sum_time;


    sum_time = 0;
    for (size_t i = 0; i < iter; ++i) {
        auto tmp = dst + chunk * ((i + threadIdx.x) % chunk_num);
        start_time = clock();
        cudaMemcpyAsync(tmp, tar, chunk, cudaMemcpyDeviceToDevice);
        end_time = clock();
        sum_time += (end_time - start_time);
    }

    printf("%llu average cudaMemcpyAsync overhead\n", sum_time / iter/ blockDim.x);
    res[threadIdx.x + 1] = sum_time;
    res[0] = (int64_t)dst[0];
}

int cuda_main()
{
    size_t chunk = 4000;
    size_t chunk_num = 1024 * 512;
    int64_t* dst;
    cudaMalloc(&dst, chunk * chunk_num * sizeof(int64_t));

    size_t elem = 1024 * 1024 * 1024;
    size_t size = sizeof(int64_t*) * elem + chunk;

    int64_t* gpu;
    cudaMalloc(&gpu, size);
    int64_t* res;
    cudaMalloc(&res, sizeof(int64_t) * 33);

    size_t itr = 10;

#define TH 64

    test_memcpyasync<<<1, TH>>>(res, gpu, dst, itr, chunk, chunk_num);

    cudaDeviceSynchronize();

    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << " 	cudaPeekAtLastError" << std::endl;

    return 0;
}
