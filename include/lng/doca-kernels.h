#ifndef LNG_DOCA_KERNELS_H
#define LNG_DOCA_KERNELS_H

#include <vector>

#include <cuda_runtime_api.h>

#include "stream.h"

namespace lng {

void init_udp_kernels(std::vector<cudaStream_t>& streams);
void launch_udp_kernels(struct rx_queue* rxq,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_fr,
    struct semaphore* sem_reply,
    std::vector<cudaStream_t>& streams);
}

#endif
