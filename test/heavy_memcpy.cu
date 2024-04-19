#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void heavy_memcpy(uint8_t* dst, uint8_t* src, size_t chunk, size_t frame_size)
{

    size_t cnt = 0;
    while (true) {
        cnt++;
        if (cnt % 1000 && threadIdx.x == 0) {
            printf("copying %d\n", cnt);
        }
        for (int i = threadIdx.x; i < frame_size / chunk - 1; i += blockDim.x) {
            cudaMemcpyAsync(dst + i * chunk, src + i * chunk, chunk, cudaMemcpyDeviceToDevice);
        }
    }
}

void heavy_memcpy_cpu()
{
    uint8_t* dst;
    uint8_t* src;

    size_t frame_size = (size_t)4 * 1024 * 1024 * 1024;
    size_t chunk = 8000;

    cudaMalloc((void**)&dst, frame_size);
    cudaMalloc((void**)&src, frame_size);

    heavy_memcpy<<<1, 1024>>>(dst, src, chunk, frame_size);
    cudaDeviceSynchronize();
}
