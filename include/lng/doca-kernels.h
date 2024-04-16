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

void init_tcp_kernels(std::vector<cudaStream_t>& streams);
void launch_tcp_kernels(struct rx_queue* rxqs, int rxq_num,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_pay,
    struct semaphore* sem_fr,
    uint8_t* tar_bufs, size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn, int* is_fin,
    struct work_buffers* work_buf,
    std::vector<cudaStream_t>& streams);
}

#endif
