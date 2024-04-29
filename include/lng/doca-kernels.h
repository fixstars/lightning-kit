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
void launch_tcp_kernels(struct rx_queue* rxq,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_fr,
    uint8_t* tar_bufs, size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn, int* is_fin,
    std::vector<cudaStream_t>& streams);

// temporary here
void init_dpdk_udp_framebuilding_kernels(std::vector<cudaStream_t>& streams);
void launch_dpdk_udp_framebuilding_kernels(
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    // struct semaphore* sem_fr,
    uint32_t* quit_flag_ptr,
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    std::vector<cudaStream_t>& streams);

void init_dpdk_tcp_framebuilding_kernels(std::vector<cudaStream_t>& streams);
void launch_dpdk_tcp_framebuilding_kernels(
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_recv_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    // struct semaphore* sem_fr,
    uint32_t* quit_flag_ptr,
    uint32_t* seqn,
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    std::vector<cudaStream_t>& streams);
void cpu_3way_handshake(
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_recv_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    uint32_t* quit_flag_ptr,
    uint32_t* seqn);
// void print_header_cpu(uint8_t* ack);
// void set_header_cpu(uint8_t* ack);
}

#endif
