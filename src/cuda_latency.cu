
#include <stdint.h>
#include <stdlib.h>

#include <iostream>

#include <cuda_runtime.h>

#define REP4(a) a a a a
#define REP16(a) REP4(a) REP4(a) REP4(a) REP4(a)
#define REP64(a) REP16(a) REP16(a) REP16(a) REP16(a)
#define REP128(a) REP64(a) REP64(a)
#define REP256(a) REP64(a) REP64(a) REP64(a) REP64(a)

// __global__ void test(int64_t* res, volatile int64_t** tar, size_t iter)
// {

//     int64_t start_time, end_time;
//     volatile int64_t** j = tar;
//     volatile int64_t sum_time;

//     // ignore
//     for (size_t i = 0; i < 100; ++i) {
//         start_time = clock();
//         REP256(j = (volatile int64_t**)(*j);)
//         end_time = clock();
//         sum_time += (end_time - start_time);
//     }

//     sum_time = 0;
//     for (size_t i = 0; i < iter; ++i) {
//         start_time = clock();
//         REP256(j = (volatile int64_t**)(*j);)
//         end_time = clock();
//         sum_time += (end_time - start_time);
//     }

//     res[0] = sum_time;
//     res[1] = (int64_t)j;
// }

// __global__ void test_memcpy(int64_t* res, int64_t** tar, int64_t* dst, size_t iter, size_t chunk, size_t chunk_num)
// {

//     int64_t start_time, end_time;
//     int64_t** j = tar;
//     volatile int64_t sum_time;

//     // ignore
//     for (size_t i = 0; i < 100; ++i) {
//         start_time = clock();
//         REP256(j = (int64_t**)(*j);)
//         end_time = clock();
//         sum_time += (end_time - start_time);
//     }

//     sum_time = 0;
//     for (size_t i = 0; i < iter; ++i) {
//         auto tmp = dst + chunk * (i % chunk_num);
//         start_time = clock();
//         cudaMemcpyAsync(tmp, tar, chunk, cudaMemcpyDeviceToDevice);
//         // REP128(cudaMemcpyAsync(tmp, tar, chunk, cudaMemcpyDeviceToDevice);)
//         // REP4(j = (int64_t**)(*j); cudaMemcpyAsync(tmp, *j, chunk, cudaMemcpyDeviceToDevice);)
//         end_time = clock();
//         sum_time += (end_time - start_time);
//     }

//     res[0] = sum_time;
//     res[1] = (int64_t)j;
// }

__global__ void self_memcpy(int64_t* dst, int64_t* tar, size_t size)
{
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = tar[i];
    }
}

__global__ void test_memcpyasync(int64_t* res, int64_t* tar, int64_t* dst, size_t iter, size_t chunk, size_t chunk_num)
{

    int64_t start_time, end_time;
    volatile int64_t sum_time;

    // ignore
    // for (size_t i = 0; i < 100; ++i) {
    //     start_time = clock();
    //     REP256(j = (int64_t**)(*j);)
    //     end_time = clock();
    //     sum_time += (end_time - start_time);
    // }

    sum_time = 0;
    for (size_t i = 0; i < iter; ++i) {
        auto tmp = dst + chunk * ((i + threadIdx.x) % chunk_num);
        start_time = clock();
        self_memcpy<<<1, 32>>>(tmp, tar, chunk);
        // cudaMemcpyAsync(tmp, tar, chunk, cudaMemcpyDeviceToDevice);
        // cudaMemcpyAsync(tmp + threadIdx.x * chunk, tar + threadIdx.x * chunk, chunk, cudaMemcpyDeviceToDevice);
        // REP64(cudaMemcpyAsync(tmp + chunk * threadIdx.x, tar + chunk * threadIdx.x, chunk, cudaMemcpyDeviceToDevice);)
        // REP4(j = (int64_t**)(*j); cudaMemcpyAsync(tmp, *j, chunk, cudaMemcpyDeviceToDevice);)
        end_time = clock();
        sum_time += (end_time - start_time);
    }

    printf("%llu sum_time\n", sum_time / iter);
    res[threadIdx.x + 1] = sum_time;
    res[0] = (int64_t)dst[0];
}

int cuda_main()
{

    size_t stride = 15; //(16*1024*1024 + 7);

    size_t chunk = 4000;
    size_t chunk_num = 1024 * 512;
    int64_t* dst;
    cudaMalloc(&dst, chunk * chunk_num * sizeof(int64_t));

    size_t elem = 1024 * 1024 * 1024; // 1024*1024*1024;
    size_t size = sizeof(int64_t*) * elem + chunk;

    int64_t* gpu;
    cudaMalloc(&gpu, size);
    int64_t* res;
    cudaMalloc(&res, sizeof(int64_t) * 33);

    size_t itr = 10;

#define TH 32

    test_memcpyasync<<<1, TH>>>(res, gpu, dst, itr, chunk, chunk_num);
    // test_memcpy<<<1, 1>>>(res, gpu, dst, itr, chunk, chunk_num);
    // test<<<1, 1>>>(res, gpu, itr);

    cudaDeviceSynchronize();

    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << " 	cudaPeekAtLastError" << std::endl;

    int64_t res_cpu[TH];
    cudaMemcpy(res_cpu, res + 1, sizeof(int64_t) * TH, cudaMemcpyHostToDevice);
    for (int i = 0; i < TH; ++i) {
        std::cout << (res_cpu[i] / itr / 64) << std::endl;
    }
}

// int cuda_main2()
// {

//     size_t stride = 15; //(16*1024*1024 + 7);

//     size_t chunk = 8000;
//     size_t chunk_num = 1024 * 512;
//     int64_t* dst;
//     cudaMalloc(&dst, chunk * chunk_num * sizeof(int64_t*));

//     size_t elem = 1024 * 1024 * 1024; // 1024*1024*1024;
//     size_t size = sizeof(int64_t*) * elem + chunk;

//     int64_t** cpu = (int64_t**)malloc(size);
//     int64_t** gpu;
//     cudaMalloc(&gpu, size);
//     int64_t* res;
//     cudaMalloc(&res, sizeof(int64_t) * 2);

//     for (size_t i = 0; i < elem; ++i) {
//         cpu[i] = (int64_t*)(cpu + ((i + stride) % elem));
//     }
//     cudaMemcpy(gpu, cpu, size, cudaMemcpyHostToDevice);

//     size_t itr = 1;

//     // test_memcpyasync<<<1, 1>>>(res, gpu, dst, itr, chunk, chunk_num);
//     test_memcpy<<<1, 1>>>(res, gpu, dst, itr, chunk, chunk_num);
//     // test<<<1, 1>>>(res, gpu, itr);

//     cudaDeviceSynchronize();

//     int64_t res_cpu;
//     cudaMemcpy(&res_cpu, res, sizeof(int64_t), cudaMemcpyHostToDevice);
//     std::cout << (res_cpu / itr / 128) << std::endl;
// }
