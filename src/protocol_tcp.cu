#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

#include "lng/doca-util.h"

#include <vector>

#include <iostream>

DOCA_LOG_REGISTER(DOCA2CU);

#define ACK_MASK (0x00 | TCP_FLAG_ACK)

namespace lng {

__device__ __inline__ int
raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr** hdr, uint8_t** payload)
{
    (*hdr) = (struct eth_ip_tcp_hdr*)buf_addr;
    (*payload) = (uint8_t*)(buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

    return 0;
}

__device__ __inline__ int
wipe_packet_32b(uint8_t* payload)
{
#pragma unroll
    for (int idx = 0; idx < 32; idx++)
        payload[idx] = 0;

    return 0;
}

__device__ __inline__ int
filter_is_tcp_syn(const struct tcp_hdr* l4_hdr)
{
    return l4_hdr->tcp_flags & TCP_FLAG_SYN;
}

__device__ __inline__ int
filter_is_tcp_fin(const struct tcp_hdr* l4_hdr)
{
    return l4_hdr->tcp_flags & TCP_FLAG_FIN;
}

__device__ __inline__ int
filter_is_tcp_ack(const struct tcp_hdr* l4_hdr)
{
    return l4_hdr->tcp_flags & ACK_MASK;
}

static __device__ void http_set_mac_addr(struct eth_ip_tcp_hdr* hdr, const uint16_t* src_bytes, const uint16_t* dst_bytes)
{
    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[0] = src_bytes[0];
    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[1] = src_bytes[1];
    ((uint16_t*)hdr->l2_hdr.s_addr_bytes)[2] = src_bytes[2];

    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[0] = dst_bytes[0];
    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[1] = dst_bytes[1];
    ((uint16_t*)hdr->l2_hdr.d_addr_bytes)[2] = dst_bytes[2];
}

__global__ void cuda_kernel_wait_3wayhandshake(
    uint32_t* out_ackn,
    struct doca_gpu_eth_rxq* rxq,
    struct doca_gpu_eth_txq* txq,
    struct doca_gpu_buf_arr* buf_arr_gpu)
{
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;
    __shared__ int is_3wayhandshake;
    __shared__ uint32_t total_payload_size;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    // struct stats_tcp* stats_global;
    struct eth_ip_tcp_hdr* hdr;
    uint32_t base_pkt_len = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr);
    uintptr_t buf_addr;
    uint64_t buf_idx = 0;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t sem_stats_idx = 0;
    uint8_t* payload;
    uint32_t max_pkts;
    uint64_t timeout_ns;
    uint64_t doca_gpu_buf_idx = laneId;

    max_pkts = MAX_RX_NUM_PKTS;
    timeout_ns = MAX_RX_TIMEOUT_NS;

    if (blockIdx.x != 0) {
        return;
    }

    if (threadIdx.x == 0) {
        is_3wayhandshake = true;
    }
    __syncthreads();

    while (is_3wayhandshake) {

        ret = doca_gpu_dev_eth_rxq_receive_block(rxq, max_pkts, timeout_ns, &rx_pkt_num, &rx_buf_idx);
        /* If any thread returns receive error, the whole execution stops */
        if (ret != DOCA_SUCCESS) {
            if (threadIdx.x == 0) {
                /*
                 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
                 * If application prints this message on the console, something bad happened and
                 * applications needs to exit
                 */
                printf("Receive TCP kernel error %d Block %d rxpkts %d error %d\n", ret, blockIdx.x, rx_pkt_num, ret);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
            }
            break;
        }

        if (rx_pkt_num == 0)
            continue;

        uint32_t sent_seq = 0;

        __syncthreads();

        if (threadIdx.x == 0 && rx_pkt_num > 0) {

            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, (rx_buf_idx + rx_pkt_num - 1) % MAX_PKT_NUM, &buf_ptr);
            if (ret != DOCA_SUCCESS) {
                printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            raw_to_tcp(buf_addr, &hdr, &payload);
            sent_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
            total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

            struct doca_gpu_buf* ack_buf = NULL;
            ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu, doca_gpu_buf_idx, &ack_buf);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            uintptr_t ack_buf_addr;
            ret = doca_gpu_dev_buf_get_addr(ack_buf, &ack_buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            struct eth_ip_tcp_hdr* ack_hdr;
            uint8_t* ack_payload;
            raw_to_tcp(ack_buf_addr, &ack_hdr, &ack_payload);

            http_set_mac_addr(ack_hdr, (uint16_t*)hdr->l2_hdr.d_addr_bytes, (uint16_t*)hdr->l2_hdr.s_addr_bytes);
            ack_hdr->l3_hdr.src_addr = hdr->l3_hdr.dst_addr;
            ack_hdr->l3_hdr.dst_addr = hdr->l3_hdr.src_addr;
            ack_hdr->l4_hdr.src_port = hdr->l4_hdr.dst_port;
            ack_hdr->l4_hdr.dst_port = hdr->l4_hdr.src_port;
            ack_hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr));

            uint32_t prev_pkt_sz = 1;
            uint32_t ackn = sent_seq + prev_pkt_sz;
            DOCA_GPUNETIO_VOLATILE(*out_ackn) = DOCA_GPUNETIO_VOLATILE(ackn);
            ack_hdr->l4_hdr.recv_ack = BYTE_SWAP32(sent_seq + prev_pkt_sz); // BYTE_SWAP32(hdr->l4_hdr.sent_seq)
            ack_hdr->l4_hdr.sent_seq = hdr->l4_hdr.recv_ack;
            ack_hdr->l4_hdr.cksum = 0;
            ack_hdr->l4_hdr.tcp_flags = TCP_FLAG_SYN | TCP_FLAG_ACK; //| TCP_FLAG_FIN;

            ret = doca_gpu_dev_eth_txq_send_enqueue_weak(txq, ack_buf, base_pkt_len + 1, 0, 0);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            doca_gpu_dev_eth_txq_commit_weak(txq, 1);
            doca_gpu_dev_eth_txq_push(txq);

            is_3wayhandshake = false;
        }

        __syncthreads();
    }

    // while (true) {
    //     ret = doca_gpu_dev_eth_rxq_receive_block(rxq, 1, timeout_ns, &rx_pkt_num, &rx_buf_idx);
    //     /* If any thread returns receive error, the whole execution stops */
    //     if (ret != DOCA_SUCCESS) {
    //         if (threadIdx.x == 0) {
    //             /*
    //              * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
    //              * If application prints this message on the console, something bad happened and
    //              * applications needs to exit
    //              */
    //             printf("Receive TCP kernel error %d Block %d rxpkts %d error %d\n", ret, blockIdx.x, rx_pkt_num, ret);
    //             // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
    //         }
    //         break;
    //     }

    //     if (rx_pkt_num == 0)
    //         continue;
    //     else
    //         break;
    // }
}

__global__ void cuda_kernel_receive_tcp(
    struct doca_gpu_eth_rxq* rxq,
    int sem_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int* is_fin, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_receive_tcp\n");
        }
        return;
    }
    __shared__ int32_t rx_pkt_num;
    __shared__ int64_t rx_buf_idx;

    // __shared__ bool is_fin;
    // uint32_t clock_count = 0;

    doca_error_t ret;
    struct rx_info* rx_info_global;
    // struct doca_gpu_buf* buf_ptr;
    // struct eth_ip_tcp_hdr* hdr;
    // uint32_t base_pkt_len = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr);
    // uintptr_t buf_addr;
    // uint64_t buf_idx = 0;
    // uint32_t laneId = threadIdx.x % WARP_SIZE;
    // uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t sem_stats_idx = 0;
    // uint8_t* payload;
    uint32_t max_pkts;
    uint64_t timeout_ns;
    // uint64_t doca_gpu_buf_idx = laneId;

    max_pkts = MAX_RX_NUM_PKTS;
    timeout_ns = MAX_RX_TIMEOUT_NS;

    __syncthreads();

    size_t heart_beat = 0;

    if (blockIdx.x == 0) {

        while (!DOCA_GPUNETIO_VOLATILE(*is_fin)) {

            // if (threadIdx.x == 0 && (heart_beat % ((size_t)1500 * 128) == 0)) {
            //     printf("heartbeat recv\n");
            // }
            // heart_beat++;

            ret = doca_gpu_dev_eth_rxq_receive_warp(rxq, max_pkts, timeout_ns, (uint32_t*)&rx_pkt_num, (uint64_t*)&rx_buf_idx);
            /* If any thread returns receive error, the whole execution stops */
            if (ret != DOCA_SUCCESS) {
                if (threadIdx.x == 0) {
                    /*
                     * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
                     * If application prints this message on the console, something bad happened and
                     * applications needs to exit
                     */
                    printf("Receive TCP kernel error %d Block %d rxpkts %d error %d\n", ret, blockIdx.x, rx_pkt_num, ret);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                }
                break;
            }
            // }

            if (threadIdx.x == 0 && rx_pkt_num > 0) {

                // if (threadIdx.x == 0 && (heart_beat % ((size_t)50) == 0)) {
                //     printf("%d heartbeat recv\n", sem_stats_idx);
                // }
                // heart_beat++;

                ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo, sem_stats_idx, (void**)&rx_info_global);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_pkt_num);
                DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx);
                // printf("%d rx_buf_idx recv\n", rx_buf_idx);
                // printf("%d cur_ackn recv\n", cur_ackn);

                // __threadfence();
                doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_stats_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP semaphore recv error\n");
                    return;
                }
                __threadfence_system();

                sem_stats_idx = (sem_stats_idx + 1) % sem_num;
            }
        }
    }
}

__global__ void send_ack(
    struct doca_gpu_eth_rxq* rxq,
    struct doca_gpu_eth_txq* txq,
    struct doca_gpu_buf_arr* buf_arr_gpu,
    int sem_rx_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int sem_pay_num, struct doca_gpu_semaphore_gpu* sem_payinfo,
    int* is_fin, bool is_warmup, int id)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup send_ack\n");
        }
        return;
    }

    __shared__ int32_t rx_pkt_num;
    __shared__ int64_t rx_buf_idx;

    // __shared__ bool is_fin;
    // uint32_t clock_count = 0;

    doca_error_t ret;
    struct rx_info* rx_info_global;
    struct pay_info* pay_info_global;
    struct doca_gpu_buf* buf_ptr;
    struct eth_ip_tcp_hdr* hdr;
    uint32_t base_pkt_len = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr);
    uintptr_t buf_addr;
    uint64_t buf_idx = 0;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t sem_stats_idx = 0;
    uint32_t sem_pay_idx = 0;
    uint8_t* payload;
    uint64_t doca_gpu_buf_idx = laneId;

    __syncthreads();

    enum doca_gpu_semaphore_status status;

    while (!(*is_fin)) {

        uint32_t cur_ackn = 0;

        ret = doca_gpu_dev_semaphore_get_status(sem_recvinfo, sem_stats_idx, &status);
        if (ret != DOCA_SUCCESS) {
            printf("TCP semaphore error");
            return;
        }

        if (status != DOCA_GPU_SEMAPHORE_STATUS_READY) {
            continue;
        }

        ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo, sem_stats_idx, (void**)&(rx_info_global));
        if (ret != DOCA_SUCCESS) {
            printf("TCP semaphore get address error\n");
            return;
        }

        DOCA_GPUNETIO_VOLATILE(rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx);
        DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num);

        __threadfence();

        // printf("%d rx_pkt_num frame \n", rx_pkt_num);
        // printf("%d rx_buf_idx frame \n", rx_buf_idx);
        // printf("%d cur_ackn frame \n", cur_ackn);

        ret = doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_stats_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
        if (ret != DOCA_SUCCESS) {
            printf("TCP semaphore error\n");
            return;
        }

        sem_stats_idx = (sem_stats_idx + 1) % sem_rx_num;

        if (threadIdx.x == 0 && rx_pkt_num > 0) {

            // if (threadIdx.x == 0 && (heart_beat % ((size_t)10) == 0)) {
            //     printf("%d heartbeat recv\n", rx_buf_idx);
            // }
            // heart_beat++;

            int64_t idx = (rx_buf_idx + rx_pkt_num - 1) % MAX_PKT_NUM;

            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx, &buf_ptr);
            if (ret != DOCA_SUCCESS) {
                printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            raw_to_tcp(buf_addr, &hdr, &payload);
            uint32_t sent_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
            uint32_t total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

            struct doca_gpu_buf* ack_buf = NULL;
            ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu, doca_gpu_buf_idx, &ack_buf);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            uintptr_t ack_buf_addr;
            ret = doca_gpu_dev_buf_get_addr(ack_buf, &ack_buf_addr);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }

            struct eth_ip_tcp_hdr* ack_hdr;
            uint8_t* ack_payload;
            raw_to_tcp(ack_buf_addr, &ack_hdr, &ack_payload);

            int is_fin_tmp = hdr->l4_hdr.tcp_flags & TCP_FLAG_FIN;

            DOCA_GPUNETIO_VOLATILE(*is_fin) = DOCA_GPUNETIO_VOLATILE(is_fin_tmp);

            http_set_mac_addr(ack_hdr, (uint16_t*)hdr->l2_hdr.d_addr_bytes, (uint16_t*)hdr->l2_hdr.s_addr_bytes);
            ack_hdr->l3_hdr.src_addr = hdr->l3_hdr.dst_addr;
            ack_hdr->l3_hdr.dst_addr = hdr->l3_hdr.src_addr;
            ack_hdr->l4_hdr.src_port = hdr->l4_hdr.dst_port;
            ack_hdr->l4_hdr.dst_port = hdr->l4_hdr.src_port;

            uint32_t prev_pkt_sz = total_payload_size;
            cur_ackn = sent_seq + prev_pkt_sz;
            // if (rx_buf_idx + rx_pkt_num >= MAX_PKT_NUM) {
            //     printf("++++++++++++++++++++++++++\n");
            //     printf("%d idx recv\n", idx);
            //     printf("%d rx_buf_idx recv\n", rx_buf_idx);
            //     printf("%d rx_pkt_num recv\n", rx_pkt_num);
            //     printf("%u sent_seq recv\n", sent_seq);
            // }
            // printf("%d rx_buf_idx\n", rx_buf_idx);
            // printf("%d rx_pkt_num\n", rx_pkt_num);
            // printf("%u sent_seq recv\n", sent_seq);
            // printf("%u cur_ackn recv\n", cur_ackn);
            // printf("%d totalpay\n", total_payload_size);
            // printf("%u sent_seq\n", prev_pkt_sz);
            // printf("%u sent_seq\n", BYTE_SWAP16(hdr->l3_hdr.total_length));
            ack_hdr->l4_hdr.recv_ack = BYTE_SWAP32(sent_seq + prev_pkt_sz);
            ack_hdr->l4_hdr.sent_seq = hdr->l4_hdr.recv_ack;
            ack_hdr->l4_hdr.cksum = 0;
            ack_hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK;
            ack_hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr));

            ret = doca_gpu_dev_eth_txq_send_enqueue_weak(txq, ack_buf, base_pkt_len, 0, 0);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            doca_gpu_dev_eth_txq_commit_weak(txq, 1);
            doca_gpu_dev_eth_txq_push(txq);
            // printf("%d doca_gpu_dev_eth_txq_push\n", id);
            // auto tx_ed_clock = clock64();
        }

        if (threadIdx.x == 0 && rx_pkt_num > 0) {

            // if (threadIdx.x == 0 && (heart_beat % ((size_t)50) == 0)) {
            //     printf("%d heartbeat recv\n", sem_stats_idx);
            // }
            // heart_beat++;

            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_payinfo, sem_pay_idx, (void**)&pay_info_global);
            if (ret != DOCA_SUCCESS) {
                printf("TCP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            DOCA_GPUNETIO_VOLATILE(pay_info_global->rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_pkt_num);
            DOCA_GPUNETIO_VOLATILE(pay_info_global->rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx);
            DOCA_GPUNETIO_VOLATILE(pay_info_global->cur_ackn) = DOCA_GPUNETIO_VOLATILE(cur_ackn);
            // printf("%d rx_buf_idx recv\n", rx_buf_idx);
            // printf("%d cur_ackn recv\n", cur_ackn);

            __threadfence();
            doca_gpu_dev_semaphore_set_status(sem_payinfo, sem_pay_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore recv error\n");
                return;
            }
            __threadfence_system();

            sem_pay_idx = (sem_pay_idx + 1) % sem_pay_num;
        }
        __syncthreads();
    }
}

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

__global__ void cuda_kernel_makeframe(
    size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn,
    struct doca_gpu_eth_rxq* rxq,
    int sem_pkt_num, struct doca_gpu_semaphore_gpu* sem_pktinfo,
    uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    int* is_fin, bool is_warmup, int id)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_makeframe\n");
        }
        return;
    }
    if (threadIdx.x == 0) {
        printf("cuda_kernel_makeframe performance\n");
    }

    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    struct fr_info* fr_global;

    __shared__ uint32_t sem_frame_idx;

    __shared__ uint64_t frame_head;
    __shared__ uint8_t* cur_tar_buf;
    __shared__ uint8_t* prev_tar_buf;

    __shared__ int64_t rx_buf_idx_head;
    __shared__ int64_t rx_buf_idx_tail;
    __shared__ uint32_t cur_ackn;

    __shared__ uint32_t prev_ackn;
    struct pay_dist_info* pay_dist_info_global;

    struct doca_gpu_buf* buf_ptr;
    struct eth_ip_tcp_hdr* hdr;
    uintptr_t buf_addr;
    uint8_t* payload;

    __shared__ int64_t remain_buf_st;

    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(prev_ackn) = DOCA_GPUNETIO_VOLATILE(*first_ackn);
        cur_tar_buf = nullptr;
        sem_frame_idx = 0;
        prev_tar_buf = nullptr;
        remain_buf_st = 0;
    }

    __syncthreads();

    uint32_t sem_pktinfo_idx = 0;
    enum doca_gpu_semaphore_status status;

    while (!DOCA_GPUNETIO_VOLATILE(*is_fin)) {

        if (threadIdx.x == 0) {
            while (true) {
                auto sem_idx = (sem_pktinfo_idx) % sem_pkt_num;
                auto ret = doca_gpu_dev_semaphore_get_status(sem_pktinfo, sem_idx, &status);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP semaphore error");
                    return;
                }
                if (status != DOCA_GPU_SEMAPHORE_STATUS_READY) {
                    continue;
                }

                ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_pktinfo, sem_idx, (void**)&(pay_dist_info_global));

                DOCA_GPUNETIO_VOLATILE(rx_buf_idx_head) = DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->rx_buf_idx_head);
                DOCA_GPUNETIO_VOLATILE(rx_buf_idx_tail) = DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->rx_buf_idx_tail);
                DOCA_GPUNETIO_VOLATILE(cur_tar_buf) = DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->cur_tar_buf);

                // printf("%llu head makeframe\n", rx_buf_idx_head);
                // printf("%llu tail makeframe\n", rx_buf_idx_tail);

                __threadfence_system();

                break;
            }
        }
        __syncthreads();

        int64_t pkt_num = rx_buf_idx_tail >= rx_buf_idx_head ? rx_buf_idx_tail - rx_buf_idx_head : rx_buf_idx_tail - rx_buf_idx_head + MAX_PKT_NUM;
        int64_t pkt_remain = pkt_num % blockDim.x;
        int64_t pkt_num_local = pkt_num / blockDim.x;
        int64_t rx_buf_st = pkt_num_local * threadIdx.x;
        if (threadIdx.x < pkt_remain) {
            rx_buf_st += threadIdx.x;
            pkt_num_local++;
        } else {
            rx_buf_st += pkt_remain;
        }
        rx_buf_st += rx_buf_idx_head;

        if ((cur_tar_buf != prev_tar_buf) && threadIdx.x == 0) {
            cudaMemcpyAsync(cur_tar_buf + max(remain_buf_st, (int64_t)0), tmp_buf, frame_head, cudaMemcpyDeviceToDevice);
            prev_tar_buf = cur_tar_buf;
            // if (heart_beat % 50 == 0) {
            //     printf("%d rx_buf_idx_head\n", rx_buf_idx_head);
            //     printf("%d rx_buf_idx_tail\n", rx_buf_idx_tail);
            // }
            // if (rx_buf_idx_tail >= MAX_PKT_NUM) {
            //     printf("-----------------------\n");
            //     printf("%d rx_buf_idx_head\n", rx_buf_idx_head);
            //     printf("%d rx_buf_idx_tail\n", rx_buf_idx_tail);
            // }
        }

        if (cur_tar_buf) {
            bool is_printed = false;
            for (int64_t idx = rx_buf_st; idx < rx_buf_st + pkt_num_local; ++idx) {

                auto ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx % (int64_t)MAX_PKT_NUM, &buf_ptr);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                raw_to_tcp(buf_addr, &hdr, &payload);
                volatile uint32_t raw_sent_seq = hdr->l4_hdr.sent_seq;
                uint32_t sent_seq = BYTE_SWAP32(raw_sent_seq);
                uint32_t total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

                uint32_t offset = sent_seq - prev_ackn;
                uint64_t cur_head = frame_head + offset;
                if (idx == rx_buf_idx_head) {
                    remain_buf_st = cur_head;
                    // printf("%d sent_seq bufst\n", sent_seq);
                    // printf("%d prev_ackn bufst\n", prev_ackn);
                }
                if (idx == rx_buf_idx_tail - 1) {
                    cur_ackn = sent_seq + total_payload_size;
                    // printf("%d cur_ackn bufst\n", cur_ackn);
                }

                if (cur_head + total_payload_size <= frame_size) {
                    uint32_t write_byte = total_payload_size;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    cudaMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                } else if (cur_head < frame_size) {
                    uint32_t write_byte = frame_size - cur_head;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    cudaMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                    // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                    //     printf("kokokoko\n");
                    // }
                } else {
                    cudaMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                    if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        printf("%" PRIx64 " idx\n", idx);
                        printf("%" PRIu64 " frame_head\n", frame_head);
                        printf("%" PRIu64 " sent_seq\n", sent_seq);
                        printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                        is_printed = true;
                    }
                }
            }
        }

        // for (int th_num = (blockDim.x + warpSize - 1) / warpSize; th_num > 1; th_num = (th_num + warpSize - 1) / warpSize) {
        //     if (threadIdx.x < warpSize * ((th_num + warpSize - 1) / warpSize)) {
        //         local_max = threadIdx.x < th_num ? packet_reached_thidx_share[threadIdx.x] : 0;
        //         packet_reached_thidx_share[threadIdx.x / warpSize] = warpMax(local_max);
        //     }
        //     __syncthreads();
        // }
        __syncthreads();

        if (threadIdx.x == 0) {
            auto sem_idx = (sem_pktinfo_idx) % sem_pkt_num;
            doca_gpu_dev_semaphore_set_status(sem_pktinfo, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
            sem_pktinfo_idx++;
        }

        if (warp_id == 1 && lane_id == 0) {
            uint64_t bytes = (cur_ackn - prev_ackn);
            // printf("%" PRIu64 " cur_ackn build\n", cur_ackn);
            // printf("%" PRIu64 " prev_ackn build\n", prev_ackn);
            // printf("%" PRIu64 " frame_head build\n", frame_head);
            // printf("%" PRIu64 " bytes build\n", bytes);
            // bytes_local += bytes;
            // if (heart_beat % 50 == 0) {
            //     auto cl_end = clock();
            //     // printf("%" PRIu64 " bytes\n", bytes_local);
            //     printf("%lf %d Gbps\n", 8 * bytes_local / ((cl_end - cl_start) / (1.5)), (cl_end - cl_start));
            //     cl_start = clock();
            //     bytes_local = 0;
            // }
            frame_head += bytes;
            if (frame_head > 2 * frame_size) {
                printf("error\n");
            }
            if (frame_head > frame_size && frame_size >= remain_buf_st) {
                auto ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_frame, sem_frame_idx, (void**)&(fr_global));
                DOCA_GPUNETIO_VOLATILE(fr_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(cur_tar_buf);
                __threadfence_system();

                ret = doca_gpu_dev_semaphore_set_status(sem_frame, sem_frame_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                __threadfence_system();
                printf("%llu frame_head send\n", frame_head);
                printf("%u %d pkt_num\n", pkt_num, id);
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                cur_tar_buf = nullptr;
                frame_head -= frame_size;
                remain_buf_st -= frame_size;
                // quit = true;
            }
            prev_ackn = cur_ackn;
        }
    }
}

#define MAX_THREAD_NUM (1024)
#define MAX_WARP_NUM (MAX_THREAD_NUM / 32)

__global__ void cuda_kernel_dist_pkt(
    uint8_t* tar_buf, size_t frame_size,
    uint32_t* first_ackn,
    int sem_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int sem_pkt_num, struct doca_gpu_semaphore_gpu* sem_pktinfo,
    uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    int* is_fin, bool is_warmup, int id)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_dist_pkt\n");
        }
        return;
    }
    if (threadIdx.x == 0) {
        printf("cuda_kernel_dist_pkt performance\n");
    }

    __shared__ uint32_t sem_frame_idx;
    __shared__ uint64_t frame_head;
    __shared__ uint32_t sem_recvinfo_idx;
    __shared__ uint16_t packet_reached_thidx_share[MAX_WARP_NUM];
    __shared__ int64_t rx_buf_idx_head;
    __shared__ int64_t rx_buf_idx_tail;
    uint16_t packet_reached_thidx = 0;

    uint32_t sem_pay_idx = 0;
    __shared__ uint32_t cur_ackn;

    doca_error_t ret;
    struct pay_info* pay_info_global;
    __shared__ bool quit;
    __shared__ uint8_t* cur_tar_buf;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    uint32_t prev_ackn;

    frame_head = 0;
    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(prev_ackn) = DOCA_GPUNETIO_VOLATILE(*first_ackn);
        cur_tar_buf = nullptr;
        quit = false;
        sem_frame_idx = 0;
        sem_recvinfo_idx = 0;
    }

    if (blockIdx.x != 0) {
        return;
    }

    __syncthreads();

    enum doca_gpu_semaphore_status status;
    __shared__ enum doca_gpu_semaphore_status status_frame;

    // size_t heart_beat = 0;

    // size_t is_first = 0;

    // auto cl_start = clock();
    size_t bytes_local = 0;
    while ((!quit) && (!DOCA_GPUNETIO_VOLATILE(*is_fin))) {

        // if (threadIdx.x == 0 && (heart_beat % ((size_t)100) == 0)) {
        //     printf("heartbeat frame %d\n", threadIdx.x);
        // }
        // heart_beat++;

        // auto cl_start = clock();
        while (true) {

            packet_reached_thidx = 0;

            for (int idx = threadIdx.x; idx < sem_num; idx += blockDim.x) {
                ret = doca_gpu_dev_semaphore_get_status(sem_recvinfo, (sem_recvinfo_idx + idx) % sem_num, &status);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP semaphore error");
                    return;
                }

                if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
                    packet_reached_thidx = idx + 1;
                } else {
                    break;
                }
            }

            __syncthreads();

            uint16_t local_max = warpMax(packet_reached_thidx);

            if (lane_id == 0) {
                packet_reached_thidx_share[warp_id] = local_max;
            }
            __syncthreads();

            if (threadIdx.x < warpSize) {
                local_max = threadIdx.x < MAX_WARP_NUM ? packet_reached_thidx_share[threadIdx.x] : 0;
                packet_reached_thidx_share[0] = warpMax(local_max);
            }

            __syncthreads();

            if (packet_reached_thidx_share[0] > 0) {

                if (warp_id == 0 && lane_id == 0) {

                    // printf("%d hoonto\n", packet_reached_thidx_share[0]);

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo,
                        sem_recvinfo_idx, (void**)&(pay_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx_head) = DOCA_GPUNETIO_VOLATILE(pay_info_global->rx_buf_idx);

                    __threadfence();

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);
                    // printf("%d cur_ackn frame \n", cur_ackn);

                    ret = doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_recvinfo_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore error\n");
                        return;
                    }
                    __threadfence_system();
                } else if (warp_id == 1 && lane_id == 0) {

                    size_t sem_idx = (sem_recvinfo_idx + packet_reached_thidx_share[0] - 1) % sem_num;

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo,
                        sem_idx, (void**)&(pay_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    int64_t rx_buf_idx;
                    int32_t rx_pkt_num;

                    DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(pay_info_global->rx_pkt_num);
                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(pay_info_global->rx_buf_idx);
                    DOCA_GPUNETIO_VOLATILE(cur_ackn) = DOCA_GPUNETIO_VOLATILE(pay_info_global->cur_ackn);

                    __threadfence();
                    rx_buf_idx_tail = rx_buf_idx + rx_pkt_num;

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);
                    // printf("%d cur_ackn frame \n", cur_ackn);

                    ret = doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore error\n");
                        return;
                    }
                    __threadfence_system();
                } else if (warp_id >= 2) {
                    for (size_t i = threadIdx.x + 1 - 2 * warpSize; i < packet_reached_thidx_share[0] - 1; i += (blockDim.x - 2 * warpSize)) {
                        ret = doca_gpu_dev_semaphore_set_status(sem_recvinfo, (sem_recvinfo_idx + i) % sem_num, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                        if (ret != DOCA_SUCCESS) {
                            printf("TCP semaphore error\n");
                            return;
                        }
                        __threadfence_system();
                    }
                }
                break;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            sem_recvinfo_idx = (sem_recvinfo_idx + packet_reached_thidx_share[0]) % sem_num;
        }

        if ((!cur_tar_buf) && threadIdx.x == 0) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, sem_frame_idx, &status_frame);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status_frame == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
                printf("%d %lld set buf\n", sem_frame_idx, frame_head);
                cur_tar_buf = tar_buf + sem_frame_idx * frame_size;
            }
        }

        if (threadIdx.x == 0) {
            struct pay_dist_info* pay_dist_info_global;
            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_pktinfo, sem_pay_idx, (void**)&(pay_dist_info_global));

            DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->rx_buf_idx_head) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx_head);
            DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->rx_buf_idx_tail) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx_tail);
            DOCA_GPUNETIO_VOLATILE(pay_dist_info_global->cur_tar_buf) = DOCA_GPUNETIO_VOLATILE(cur_tar_buf);
            __threadfence();
            doca_gpu_dev_semaphore_set_status(sem_pktinfo, sem_pay_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);

            while (true) {
                doca_gpu_dev_semaphore_get_status(sem_pktinfo, sem_pay_idx, &status);
                if (status != DOCA_GPU_SEMAPHORE_STATUS_READY)
                    break;
            }

            sem_pay_idx = (sem_pay_idx + 1) % sem_pkt_num;

            uint64_t bytes = (cur_ackn - prev_ackn);
            // printf("%llu prev_ackn dist\n", prev_ackn);
            // printf("%llu cur_ackn dist\n", cur_ackn);
            // printf("%llu bytes dist\n", bytes);
            frame_head += bytes;
            if (frame_head > frame_size) {
                cur_tar_buf = nullptr;
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                frame_head -= frame_size;
            }
            prev_ackn = cur_ackn;
        }
    }
}

__global__ void frame_notice(
    uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    int* is_fin, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup frame_notice\n");
        }
        return;
    }
    doca_error_t ret;
    __shared__ enum doca_gpu_semaphore_status status;
    int reached_frame = 0;
    if (threadIdx.x == 0) {
        int frame_counter = 0;
        while ((!DOCA_GPUNETIO_VOLATILE(*is_fin))) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, frame_counter, &status);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
                printf("%d kitayo\n", reached_frame++);
                ret = doca_gpu_dev_semaphore_set_status(sem_frame, frame_counter, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                // fin = (frame_counter == 4);
                frame_counter = (frame_counter + 1) % frame_num;
            }
        }
    }
}

void init_tcp_kernels(std::vector<cudaStream_t>& streams)
{

    cuda_kernel_receive_tcp<<<1, CUDA_THREADS>>>(
        nullptr,
        0, nullptr, nullptr, true);

    send_ack<<<1, CUDA_THREADS>>>(
        nullptr, nullptr, nullptr,
        0, nullptr, 0, nullptr, nullptr, true, 0);

    cuda_kernel_dist_pkt<<<1, 32>>>(
        nullptr, 0, nullptr,
        0, nullptr,
        0, nullptr,
        0, nullptr,
        nullptr, true, 0);

    cuda_kernel_makeframe<<<1, 32>>>(
        0,
        nullptr,
        nullptr,
        nullptr,
        0, nullptr,
        0, nullptr,
        nullptr, true, 0);

    frame_notice<<<1, CUDA_THREADS>>>(0, nullptr, nullptr, true);

    streams.resize(5);

    // int leastPriority;
    // int greatestPriority;
    // cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    // std::cout << leastPriority << " " << greatestPriority << " greatestPriority" << std::endl;
    // cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking, greatestPriority);
    // cudaStreamCreateWithPriority(&streams[1], cudaStreamNonBlocking, leastPriority);
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);
    cudaStreamCreate(&streams[4]);
}

void launch_tcp_kernels(struct rx_queue* rxq,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_pay,
    struct semaphore* sem_pkt,
    struct semaphore* sem_fr,
    uint8_t* tar_bufs, size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn, int* is_fin,
    std::vector<cudaStream_t>& streams, int id)
{

    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << " 	cudaPeekAtLastError" << id << std::endl;
    cuda_kernel_wait_3wayhandshake<<<1, CUDA_THREADS, 0, streams[1]>>>(
        first_ackn,
        rxq->eth_rxq_gpu,
        txq->eth_txq_gpu,
        tx_buf_arr->buf_arr_gpu);
    std::cout << "cuda_kernel_wait_3wayhandshake" << std::endl;
    cudaDeviceSynchronize();

    cuda_kernel_receive_tcp<<<1, 32, 0, streams[0]>>>(
        rxq->eth_rxq_gpu,
        sem_rx->sem_num, sem_rx->sem_gpu,
        is_fin, false);

    send_ack<<<1, 32, 0, streams[3]>>>(
        rxq->eth_rxq_gpu,
        txq->eth_txq_gpu,
        tx_buf_arr->buf_arr_gpu,
        sem_rx->sem_num, sem_rx->sem_gpu,
        sem_pay->sem_num, sem_pay->sem_gpu,
        is_fin, false, id);

    cuda_kernel_dist_pkt<<<1, MAX_THREAD_NUM, 0, streams[4]>>>(
        tar_bufs, frame_size,
        first_ackn,
        sem_pay->sem_num, sem_pay->sem_gpu,
        sem_pkt->sem_num, sem_pkt->sem_gpu,
        sem_fr->sem_num, sem_fr->sem_gpu,
        is_fin, false, id);

    cuda_kernel_makeframe<<<1, MAX_THREAD_NUM, 0, streams[1]>>>(
        frame_size,
        tmp_buf,
        first_ackn,
        rxq->eth_rxq_gpu,
        sem_pkt->sem_num, sem_pkt->sem_gpu,
        sem_fr->sem_num, sem_fr->sem_gpu,
        is_fin, false, id);

    frame_notice<<<1, 32, 0, streams[2]>>>(sem_fr->sem_num, sem_fr->sem_gpu, is_fin, false);
}

}
