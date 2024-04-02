#ifndef CUDA_REDUCE_H
#define CUDA_REDUCE_H

#include <cuda_runtime.h>

template <typename T>
__inline__ __device__ T warpMax(T localMax)
{
    localMax = max(localMax, __shfl_xor_sync(0xffffffff, localMax, 16));
    localMax = max(localMax, __shfl_xor_sync(0xffffffff, localMax, 8));
    localMax = max(localMax, __shfl_xor_sync(0xffffffff, localMax, 4));
    localMax = max(localMax, __shfl_xor_sync(0xffffffff, localMax, 2));
    localMax = max(localMax, __shfl_xor_sync(0xffffffff, localMax, 1));

    return localMax;
}

#endif
