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

            ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, ack_buf, base_pkt_len + 1, 0);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            doca_gpu_dev_eth_txq_commit_strong(txq);
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
                // printf("%d rx_pkt_num recv\n", rx_pkt_num);
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

#define MEASURE_JITTER
#define INTERVAL 4
#define NUM_INTERVALS (16 * 25)
#define HZ 1950

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

#ifdef MEASURE_JITTER
    __shared__ int jitter[NUM_INTERVALS];
#endif

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

#ifdef MEASURE_JITTER
    uint64_t prev_rdtsc_ = 0;
#endif

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

            ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, ack_buf, base_pkt_len, 0);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, laneId);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
            doca_gpu_dev_eth_txq_commit_strong(txq);
            doca_gpu_dev_eth_txq_push(txq);
            // printf("%d doca_gpu_dev_eth_txq_push\n", id);
            // auto tx_ed_clock = clock64();
#ifdef MEASURE_JITTER
            uint64_t current = clock64();
            uint64_t duration = current - prev_rdtsc_;
            prev_rdtsc_ = current;
            int idx_du = duration / (INTERVAL * HZ);
            if (idx_du >= NUM_INTERVALS) {
                jitter[NUM_INTERVALS - 1]++;
            } else if (idx_du >= 0) {
                jitter[idx_du]++;
            }
#endif
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

#ifdef MEASURE_JITTER
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_INTERVALS; ++i) {
            printf("under %d sec %d times\n", (i + 1) * 4, jitter[i]);
        }
    }
#endif
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

#define MAX_THREAD_NUM (512)
#define MAX_WARP_NUM (MAX_THREAD_NUM / 32)

#define lngMemcpy2DAsync(...)                                                   \
    if (cudaMemcpy2DAsync(__VA_ARGS__) != cudaSuccess) {                        \
        printf("%d %s\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); \
    }
#define lngMemcpyAsync(...)                                                     \
    if (cudaMemcpyAsync(__VA_ARGS__) != cudaSuccess) {                          \
        printf("%d %s\n", __LINE__, cudaGetErrorString(cudaPeekAtLastError())); \
    }

__global__ void cuda_kernel_makeframe(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn,
    struct doca_gpu_eth_rxq* rxq,
    int sem_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
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

    __shared__ int64_t rx_buf_idx_head;
    __shared__ int64_t rx_buf_idx_tail;

    __shared__ bool packet_reached;
    __shared__ uint16_t packet_reached_thidx_share[MAX_WARP_NUM];
    uint16_t packet_reached_thidx = 0;

    __shared__ uint64_t frame_head;
    __shared__ uint32_t prev_ackn;

    __shared__ uint8_t* cur_tar_buf;

    uint32_t cur_ackn;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct doca_gpu_buf* buf_ptr2;
    struct pay_info* pay_info_global;
    struct fr_info* fr_global;
    struct eth_ip_tcp_hdr* hdr;
    struct eth_ip_tcp_hdr* hdr2;
    uintptr_t buf_addr;
    uintptr_t buf_addr2;
    __shared__ uint32_t sem_recvinfo_idx;
    __shared__ uint32_t sem_frame_idx;
    uint8_t* payload;
    uint8_t* payload2;
    __shared__ bool quit;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    frame_head = 0;
    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(prev_ackn) = DOCA_GPUNETIO_VOLATILE(*first_ackn);
        packet_reached = false;
        cur_tar_buf = nullptr;
        quit = false;
        sem_frame_idx = 0;
        sem_recvinfo_idx = 0;

        printf("%d prev_ackn initial\n", prev_ackn);
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

        auto cl_start = clock();
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

        bool is_head_copy = false;

        if ((!cur_tar_buf) && threadIdx.x == 0) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, sem_frame_idx, &status_frame);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status_frame == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
                printf("%d %lld set buf\n", sem_frame_idx, frame_head);
                cur_tar_buf = tar_buf + sem_frame_idx * frame_size;
                is_head_copy = true;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            sem_recvinfo_idx = (sem_recvinfo_idx + packet_reached_thidx_share[0]) % sem_num;
            if (is_head_copy)
                cudaMemcpyAsync(cur_tar_buf, tmp_buf, frame_head, cudaMemcpyDeviceToDevice);
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

        int64_t pkt_num = rx_buf_idx_tail >= rx_buf_idx_head ? rx_buf_idx_tail - rx_buf_idx_head : rx_buf_idx_tail - rx_buf_idx_head + MAX_PKT_NUM;
        int64_t pkt_num2 = pkt_num / 2;
        int64_t pkt_remain = pkt_num2 % blockDim.x;
        int64_t pkt_num_local = pkt_num2 / blockDim.x * 2;
        int64_t rx_buf_st = pkt_num_local * threadIdx.x;
        if (threadIdx.x < pkt_remain) {
            rx_buf_st += threadIdx.x * 2;
            pkt_num_local += 2;
        } else {
            rx_buf_st += pkt_remain * 2;
        }
        rx_buf_st += rx_buf_idx_head;
        if (threadIdx.x == blockDim.x - 1) {
            pkt_num_local += pkt_num % 2;
        }

        // if (threadIdx.x == 0) {
        //     printf("%d rx_buf_st st\n", rx_buf_st);
        //     printf("%" PRIu64 " rx_buf_idx_head\n", rx_buf_idx_head);
        // } else if (threadIdx.x == blockDim.x - 1) {
        //     printf("%d rx_buf_st ed\n", rx_buf_st + pkt_num_local);
        // }

        // if (is_first == 4) {
        //     printf("%" PRIu64 " rx_buf_st\n", rx_buf_st);
        // }
        // is_first++;

        if (cur_tar_buf) {
            bool is_printed = false;
            for (int64_t idx = rx_buf_st; idx < rx_buf_st + (pkt_num_local % 2 == 0 ? pkt_num_local : pkt_num_local - 1); idx += 2) {

                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx % (int64_t)MAX_PKT_NUM, &buf_ptr);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, (idx + 1) % (int64_t)MAX_PKT_NUM, &buf_ptr2);
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
                ret = doca_gpu_dev_buf_get_addr(buf_ptr2, &buf_addr2);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                raw_to_tcp(buf_addr, &hdr, &payload);
                raw_to_tcp(buf_addr2, &hdr2, &payload2);
                uint32_t sent_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
                uint32_t sent_seq2 = BYTE_SWAP32(hdr2->l4_hdr.sent_seq);
                uint32_t total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);
                uint32_t total_payload_size2 = BYTE_SWAP16(hdr2->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

                uint32_t offset = sent_seq - prev_ackn;
                uint32_t offset2 = sent_seq2 - prev_ackn;
                uint64_t cur_head = frame_head + offset;
                uint64_t cur_head2 = frame_head + offset2;

                if (total_payload_size == total_payload_size2) {
                    if ((uintptr_t)payload < (uintptr_t)payload2) {
                        if (cur_head2 + total_payload_size2 <= frame_size) {
                            uint32_t write_byte = total_payload_size;
                            uint8_t* data_head = cur_tar_buf + cur_head;
                            lngMemcpy2DAsync(data_head, write_byte,
                                payload, (uintptr_t)payload2 - (uintptr_t)payload,
                                write_byte, 2, cudaMemcpyDeviceToDevice);
                        } else {
                            if (cur_head + total_payload_size <= frame_size) {
                                uint32_t write_byte = total_payload_size;
                                uint8_t* data_head = cur_tar_buf + cur_head;
                                lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                            } else if (cur_head < frame_size) {
                                uint32_t write_byte = frame_size - cur_head;
                                uint8_t* data_head = cur_tar_buf + cur_head;
                                lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                                lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                                // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                //     printf("kokokoko\n");
                                // }
                            } else {
                                lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                                if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                    printf("%" PRIx64 " idx\n", idx);
                                    printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                                    printf("%" PRIu64 " sent_seq\n", sent_seq);
                                    printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                                    is_printed = true;
                                }
                            }
                            if (cur_head2 + total_payload_size2 <= frame_size) {
                                uint32_t write_byte = total_payload_size2;
                                uint8_t* data_head = cur_tar_buf + cur_head2;
                                lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                            } else if (cur_head2 < frame_size) {
                                uint32_t write_byte = frame_size - cur_head2;
                                uint8_t* data_head = cur_tar_buf + cur_head2;
                                lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                                lngMemcpyAsync(tmp_buf, payload2 + write_byte, total_payload_size2 - write_byte, cudaMemcpyDeviceToDevice);
                                // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                //     printf("kokokoko\n");
                                // }
                            } else {
                                lngMemcpyAsync(tmp_buf + cur_head2 - frame_size, payload2, total_payload_size2, cudaMemcpyDeviceToDevice);
                                if ((!is_printed) && cur_head2 - frame_size + total_payload_size2 > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                    printf("%" PRIx64 " idx\n", idx);
                                    printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                                    printf("%" PRIu64 " sent_seq\n", sent_seq);
                                    printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                                    is_printed = true;
                                }
                            }
                        }
                    } else {
                        if (cur_head + total_payload_size <= frame_size) {
                            uint32_t write_byte = total_payload_size;
                            uint8_t* data_head = cur_tar_buf + cur_head;
                            lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                        } else if (cur_head < frame_size) {
                            uint32_t write_byte = frame_size - cur_head;
                            uint8_t* data_head = cur_tar_buf + cur_head;
                            lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                            lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                            // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                            //     printf("kokokoko\n");
                            // }
                        } else {
                            lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                            if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                printf("%" PRIx64 " idx\n", idx);
                                printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                                printf("%" PRIu64 " sent_seq\n", sent_seq);
                                printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                                is_printed = true;
                            }
                        }
                        if (cur_head2 + total_payload_size2 <= frame_size) {
                            uint32_t write_byte = total_payload_size2;
                            uint8_t* data_head = cur_tar_buf + cur_head2;
                            lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                        } else if (cur_head2 < frame_size) {
                            uint32_t write_byte = frame_size - cur_head2;
                            uint8_t* data_head = cur_tar_buf + cur_head2;
                            lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                            lngMemcpyAsync(tmp_buf, payload2 + write_byte, total_payload_size2 - write_byte, cudaMemcpyDeviceToDevice);
                            // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                            //     printf("kokokoko\n");
                            // }
                        } else {
                            lngMemcpyAsync(tmp_buf + cur_head2 - frame_size, payload2, total_payload_size2, cudaMemcpyDeviceToDevice);
                            if ((!is_printed) && cur_head2 - frame_size + total_payload_size2 > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                                printf("%" PRIx64 " idx\n", idx);
                                printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                                printf("%" PRIu64 " sent_seq\n", sent_seq);
                                printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                                is_printed = true;
                            }
                        }
                    }
                } else {
                    if (cur_head + total_payload_size <= frame_size) {
                        uint32_t write_byte = total_payload_size;
                        uint8_t* data_head = cur_tar_buf + cur_head;
                        lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                    } else if (cur_head < frame_size) {
                        uint32_t write_byte = frame_size - cur_head;
                        uint8_t* data_head = cur_tar_buf + cur_head;
                        lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                        lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                        // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        //     printf("kokokoko\n");
                        // }
                    } else {
                        lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                        if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                            printf("%" PRIx64 " idx\n", idx);
                            printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                            printf("%" PRIu64 " sent_seq\n", sent_seq);
                            printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                            is_printed = true;
                        }
                    }
                    if (cur_head2 + total_payload_size2 <= frame_size) {
                        uint32_t write_byte = total_payload_size2;
                        uint8_t* data_head = cur_tar_buf + cur_head2;
                        lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                    } else if (cur_head2 < frame_size) {
                        uint32_t write_byte = frame_size - cur_head2;
                        uint8_t* data_head = cur_tar_buf + cur_head2;
                        lngMemcpyAsync(data_head, payload2, write_byte, cudaMemcpyDeviceToDevice);
                        lngMemcpyAsync(tmp_buf, payload2 + write_byte, total_payload_size2 - write_byte, cudaMemcpyDeviceToDevice);
                        // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        //     printf("kokokoko\n");
                        // }
                    } else {
                        lngMemcpyAsync(tmp_buf + cur_head2 - frame_size, payload2, total_payload_size2, cudaMemcpyDeviceToDevice);
                        if ((!is_printed) && cur_head2 - frame_size + total_payload_size2 > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                            printf("%" PRIx64 " idx\n", idx);
                            printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                            printf("%" PRIu64 " sent_seq\n", sent_seq);
                            printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                            is_printed = true;
                        }
                    }
                }
            }
            if (pkt_num_local % 2 == 1) {
                int64_t idx = rx_buf_st + pkt_num_local - 1;

                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx % (int64_t)MAX_PKT_NUM, &buf_ptr);
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

                if (cur_head + total_payload_size <= frame_size) {
                    uint32_t write_byte = total_payload_size;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                } else if (cur_head < frame_size) {
                    uint32_t write_byte = frame_size - cur_head;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                    lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                    // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                    //     printf("kokokoko\n");
                    // }
                } else {
                    lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                    if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        printf("%" PRIx64 " idx\n", idx);
                        printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
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

        if (warp_id == 1 && lane_id == 0) {
            uint64_t bytes = (cur_ackn - prev_ackn);
            // printf("%" PRIu64 " cur_ackn_fin\n", cur_ackn);
            // printf("%" PRIu64 " prev_ackn\n", prev_ackn);
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
            if (frame_head > frame_size) {
                ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_frame, sem_frame_idx, (void**)&(fr_global));
                DOCA_GPUNETIO_VOLATILE(fr_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(cur_tar_buf);
                __threadfence_system();

                ret = doca_gpu_dev_semaphore_set_status(sem_frame, sem_frame_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                __threadfence_system();
                auto cl_end = clock();
                printf("%llu %u frame_head send\n", frame_head, packet_reached_thidx_share[0]);
                printf("%u %d pkt_num\n", pkt_num, id);
                printf("%llu clock\n", cl_end - cl_start);
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                cur_tar_buf = nullptr;
                frame_head -= frame_size;
                // quit = true;
            }
            prev_ackn = cur_ackn;
        }

        __syncthreads();
        packet_reached = false;
    }
}

#define MAX_THREAD_RING_NUM (512)
#define MAX_WARP_RING_NUM (MAX_THREAD_RING_NUM / 32)

__global__ void cuda_kernel_makeframe_assumering(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    uint32_t* first_ackn,
    struct doca_gpu_eth_rxq* rxq,
    int sem_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
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

    __shared__ int64_t rx_buf_idx_head;
    __shared__ int64_t rx_buf_idx_tail;

    __shared__ bool packet_reached;
    __shared__ uint16_t packet_reached_thidx_share[MAX_WARP_RING_NUM];
    uint16_t packet_reached_thidx = 0;

    __shared__ uint64_t frame_head;
    __shared__ uint32_t prev_ackn;

    __shared__ uint8_t* cur_tar_buf;

    __shared__ uintptr_t next_paydiff[MAX_THREAD_RING_NUM];
    __shared__ uint8_t next_same[MAX_THREAD_RING_NUM];

    uint32_t cur_ackn;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct pay_info* pay_info_global;
    struct fr_info* fr_global;
    struct eth_ip_tcp_hdr* hdr;
    uintptr_t buf_addr;
    __shared__ uint32_t sem_recvinfo_idx;
    __shared__ uint32_t sem_frame_idx;
    uint8_t* payload;
    __shared__ bool quit;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // int count = 0;

    frame_head = 0;
    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(prev_ackn) = DOCA_GPUNETIO_VOLATILE(*first_ackn);
        packet_reached = false;
        cur_tar_buf = nullptr;
        quit = false;
        sem_frame_idx = 0;
        sem_recvinfo_idx = 0;
        rx_buf_idx_head = -1;

        printf("%d prev_ackn initial\n", prev_ackn);
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

        auto cl_start = clock();
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
                local_max = threadIdx.x < MAX_WARP_RING_NUM ? packet_reached_thidx_share[threadIdx.x] : 0;
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

                    // printf("%d rx_buf_idx_head frame \n", rx_buf_idx_head);
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

        bool is_head_copy = false;

        if ((!cur_tar_buf) && threadIdx.x == 0) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, sem_frame_idx, &status_frame);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status_frame == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
                printf("%d %lld set buf\n", sem_frame_idx, frame_head);
                cur_tar_buf = tar_buf + sem_frame_idx * frame_size;
                is_head_copy = true;
            }
        }

        if (threadIdx.x == 0) {
            sem_recvinfo_idx = (sem_recvinfo_idx + packet_reached_thidx_share[0]) % sem_num;
            if (is_head_copy)
                cudaMemcpyAsync(cur_tar_buf, tmp_buf, frame_head, cudaMemcpyDeviceToDevice);
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

        __syncthreads();

        int64_t pkt_num = rx_buf_idx_tail >= rx_buf_idx_head ? rx_buf_idx_tail - rx_buf_idx_head : rx_buf_idx_tail - rx_buf_idx_head + MAX_PKT_NUM;

        if (cur_tar_buf) {
            bool is_printed = false;
            // count++;
            // if (threadIdx.x == 0 && rx_buf_idx_head == 0) {
            //     printf("%d %d rx_buf_idx_head\n", rx_buf_idx_head, count);
            // }

            for (int64_t idx = rx_buf_idx_head + threadIdx.x; idx < rx_buf_idx_head + ((pkt_num / warpSize) * warpSize); idx += blockDim.x) {

                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx % (int64_t)MAX_PKT_NUM, &buf_ptr);
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

                uint32_t offset = sent_seq - prev_ackn;
                uint64_t cur_head = frame_head + offset;

                auto warp_tail = __shfl_sync(0xffffffff, cur_head + total_payload_size, warpSize - 1);

                if (warp_tail <= frame_size) {
                    auto next_payload = __shfl_down_sync(0xffffffff, (uintptr_t)payload, 1);
                    auto next_payload_diff = next_payload - (uintptr_t)payload;
                    next_paydiff[threadIdx.x] = next_payload_diff;
                    if (lane_id == 0) {
                        next_same[threadIdx.x] = 1;
                        next_same[threadIdx.x + 1] = ((uintptr_t)payload) > next_payload;
                        for (int i = threadIdx.x + 2; i < threadIdx.x + warpSize; ++i) {
                            if (next_paydiff[i - 1] == next_paydiff[i - 2]) {
                                next_same[i] = 0;
                            } else {
                                next_same[i] = 1;
                                if (i != threadIdx.x + warpSize - 1) {
                                    next_same[i + 1] = 0;
                                    i++;
                                }
                            }
                        }
                    }

                    auto next_size = __shfl_up_sync(0xffffffff, total_payload_size, 1);
                    if (lane_id == 0)
                        next_same[threadIdx.x] = 1;
                    else
                        next_same[threadIdx.x] |= (next_size == total_payload_size ? 0 : 1);

                    if (next_same[threadIdx.x] != 0) {
                        int count_pkt = 1;
                        for (auto i = threadIdx.x + 1; i < (warp_id + 1) * warpSize; ++i) {
                            if (next_same[i])
                                break;
                            count_pkt++;
                        }
                        uint32_t write_byte = total_payload_size;
                        uint8_t* data_head = cur_tar_buf + cur_head;
                        // if (lane_id != 0)
                        // {
                        //     printf("%llu cur_head\n", cur_head);
                        //     printf("%d count_pkt\n", count_pkt);
                        //     printf("%p count_pkt\n", payload);
                        // }

                        if (count_pkt > 1) {
                            lngMemcpy2DAsync(data_head, write_byte,
                                payload, next_payload_diff,
                                write_byte, count_pkt, cudaMemcpyDeviceToDevice);
                        } else {
                            lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                        }
                    }
                } else {
                    if (threadIdx.x == 0) {
                        printf("%llu tail\n", cur_head + total_payload_size);
                    }
                    if (cur_head + total_payload_size <= frame_size) {
                        uint32_t write_byte = total_payload_size;
                        uint8_t* data_head = cur_tar_buf + cur_head;
                        lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                    } else if (cur_head < frame_size) {
                        uint32_t write_byte = frame_size - cur_head;
                        uint8_t* data_head = cur_tar_buf + cur_head;
                        lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                        lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                        // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        //     printf("kokokoko\n");
                        // }
                    } else {
                        lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                        if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                            printf("%" PRIx64 " idx0\n", idx);
                            printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
                            printf("%" PRIu64 " sent_seq\n", sent_seq);
                            printf("%" PRIu64 " prev_ackn\n", prev_ackn);
                            is_printed = true;
                        }
                    }
                }
            }
            for (int64_t idx = rx_buf_idx_head + (pkt_num / warpSize * warpSize) + threadIdx.x; idx < rx_buf_idx_head + pkt_num; idx += blockDim.x) {

                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, idx % (int64_t)MAX_PKT_NUM, &buf_ptr);
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

                if (cur_head + total_payload_size <= frame_size) {
                    uint32_t write_byte = total_payload_size;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                } else if (cur_head < frame_size) {
                    uint32_t write_byte = frame_size - cur_head;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    lngMemcpyAsync(data_head, payload, write_byte, cudaMemcpyDeviceToDevice);
                    lngMemcpyAsync(tmp_buf, payload + write_byte, total_payload_size - write_byte, cudaMemcpyDeviceToDevice);
                    // if (total_payload_size - write_byte > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                    //     printf("kokokoko\n");
                    // }
                } else {
                    lngMemcpyAsync(tmp_buf + cur_head - frame_size, payload, total_payload_size, cudaMemcpyDeviceToDevice);
                    if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)1024 * 1024 * 1024) {
                        printf("%d pkt_num\n", pkt_num);
                        printf("%d rx_buf_idx_head\n", rx_buf_idx_head);
                        printf("%" PRIx64 " idx1\n", idx);
                        printf("%" PRIu64 " idx_round\n", idx % MAX_PKT_NUM);
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

        if (warp_id == 1 && lane_id == 0) {
            uint64_t bytes = (cur_ackn - prev_ackn);
            // printf("%" PRIu64 " cur_ackn_fin\n", cur_ackn);
            // printf("%" PRIu64 " prev_ackn\n", prev_ackn);
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
            if (frame_head > frame_size) {
                ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_frame, sem_frame_idx, (void**)&(fr_global));
                DOCA_GPUNETIO_VOLATILE(fr_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(cur_tar_buf);
                __threadfence_system();

                ret = doca_gpu_dev_semaphore_set_status(sem_frame, sem_frame_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                __threadfence_system();
                auto cl_end = clock();
                printf("%llu %u frame_head send\n", frame_head, packet_reached_thidx_share[0]);
                printf("%u %d pkt_num\n", pkt_num, id);
                printf("%llu clock\n", cl_end - cl_start);
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                cur_tar_buf = nullptr;
                frame_head -= frame_size;
                // quit = true;
            }
            prev_ackn = cur_ackn;
        }

        __syncthreads();
        packet_reached = false;
    }
}

__global__ void frame_check_kernel(
    uint8_t* frame, size_t frame_size, int* res, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup frame_check_kernel\n");
        }
        return;
    }

    __shared__ bool is_ok;
    is_ok = true;
    __syncthreads();

    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < frame_size; i += blockDim.x * gridDim.x) {
        if (frame[i] != (uint8_t)i) {
            is_ok = false;
            printf("%d actu\n", frame[i]);
            printf("%d exp\n", (uint8_t)i);
        }
        if (!is_ok) {
            break;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAnd(res, is_ok);
    }
}

__global__ void frame_check_print(int* res, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup frame_check_print\n");
        }
        return;
    }

    if (*res) {
        printf("kitayo ok\n");
    } else {
        printf("kitayo error\n");
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

    cuda_kernel_makeframe_assumering<<<1, CUDA_THREADS>>>(
        nullptr, 0,
        nullptr,
        nullptr,
        nullptr,
        0, nullptr,
        0, nullptr,
        nullptr, true, 0);

    // cuda_kernel_makeframe<<<1, CUDA_THREADS>>>(
    //     nullptr, 0,
    //     nullptr,
    //     nullptr,
    //     nullptr,
    //     0, nullptr,
    //     0, nullptr,
    //     nullptr, true, 0);

    frame_notice<<<1, CUDA_THREADS>>>(0, nullptr, nullptr, true);

    frame_check_kernel<<<1, 32>>>(nullptr, 0, nullptr, true);
    frame_check_print<<<1, 32>>>(nullptr, true);

    streams.resize(4);

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
}

void launch_tcp_kernels(struct rx_queue* rxq,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_pay,
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
    cudaStreamSynchronize(streams[1]);

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

    cuda_kernel_makeframe_assumering<<<1, MAX_THREAD_RING_NUM, 0, streams[1]>>>(
        tar_bufs, frame_size,
        tmp_buf,
        first_ackn,
        rxq->eth_rxq_gpu,
        sem_pay->sem_num, sem_pay->sem_gpu,
        sem_fr->sem_num, sem_fr->sem_gpu,
        is_fin, false, id);

    // cuda_kernel_makeframe<<<1, MAX_THREAD_NUM, 0, streams[1]>>>(
    //     tar_bufs, frame_size,
    //     tmp_buf,
    //     first_ackn,
    //     rxq->eth_rxq_gpu,
    //     sem_pay->sem_num, sem_pay->sem_gpu,
    //     sem_fr->sem_num, sem_fr->sem_gpu,
    //     is_fin, false, id);

    // frame_notice<<<1, 32, 0, streams[2]>>>(sem_fr->sem_num, sem_fr->sem_gpu, is_fin, false);
}

void frame_check(uint8_t* frame, size_t frame_size, int* res, cudaStream_t stream)
{
    cudaMemsetAsync(res, 1, sizeof(int), stream);
    frame_check_kernel<<<16, 1024, 0, stream>>>(frame, frame_size, res, false);
    cudaMemsetAsync(frame, 0, frame_size, stream); // reset
    frame_check_print<<<1, 1, 0, stream>>>(res, false);
}

}
