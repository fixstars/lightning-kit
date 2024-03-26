#ifndef LNG_DOCA_KERNELS_H
#define LNG_DOCA_KERNELS_H

#include <vector>

#include <cuda_runtime_api.h>

#include "stream.h"

namespace lng{

void init_udp_kernels(struct rx_queue* rxq, struct semaphore* sem,
    std::vector<cudaStream_t>& streams);
void launch_udp_kernels(struct rx_queue* rxq, struct semaphore* sem,
    std::vector<cudaStream_t>& streams);
}

#endif
