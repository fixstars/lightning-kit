#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

#include "lng/doca-util.h"

DOCA_LOG_REGISTER(DOCA2CU);

#define ACK_MASK (0x00 | TCP_FLAG_ACK)

using namespace lng;

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

__global__ void cuda_kernel_receive_tcp(
    struct doca_gpu_eth_rxq* rxq,
    struct doca_gpu_eth_txq* txq,
    int sem_num,
    struct doca_gpu_buf_arr* buf_arr_gpu, uint32_t nbytes,
    struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int* is_fin, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_receive_tcp\n");
        }
        return;
    }
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;

    // __shared__ bool is_fin;
    uint32_t clock_count = 0;

    doca_error_t ret;
    struct rx_info* rx_info_global;
    struct doca_gpu_buf* buf_ptr;
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

    if (blockIdx.x >= 2) {
        return;
    }

    __syncthreads();

    if (blockIdx.x == 0) {

        while (!(*is_fin)) {

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
            // }

            if (rx_pkt_num == 0)
                continue;

            __syncthreads();
            uint32_t cur_ackn = 0;

            if (threadIdx.x == 0 && rx_pkt_num > 0) {

                ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + rx_pkt_num - 1, &buf_ptr);
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
                ack_hdr->l4_hdr.recv_ack = BYTE_SWAP32(sent_seq + prev_pkt_sz);
                ack_hdr->l4_hdr.sent_seq = hdr->l4_hdr.recv_ack;
                ack_hdr->l4_hdr.cksum = 0;
                ack_hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK;

                ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, ack_buf, base_pkt_len + nbytes, 0);
                if (ret != DOCA_SUCCESS) {
                    printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, laneId);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                doca_gpu_dev_eth_txq_commit_strong(txq);
                doca_gpu_dev_eth_txq_push(txq);
                printf("doca_gpu_dev_eth_txq_push\n");
                // auto tx_ed_clock = clock64();
            }

            if (threadIdx.x == 0 && rx_pkt_num > 0) {

                ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo, sem_stats_idx, (void**)&rx_info_global);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP Error %d doca_gpu_dev_semaphore_get_custom_info_addr block %d thread %d\n", ret, blockIdx.x, threadIdx.x);
                    // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                    break;
                }
                DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_pkt_num);
                DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_buf_idx);
                DOCA_GPUNETIO_VOLATILE(rx_info_global->cur_ackn) = DOCA_GPUNETIO_VOLATILE(cur_ackn);
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
            __syncthreads();
        }
    }
}

__global__ void cuda_kernel_makeframe(
    uint8_t* tar_buf, uint64_t tar_buf_total_size, uint64_t pitch, uint32_t* first_ackn,
    struct doca_gpu_eth_rxq* rxq,
    int sem_num,
    struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int* is_fin,
    struct doca_gpu_semaphore_gpu* sem_frame, uint64_t frame_num, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_makeframe\n");
        }
        return;
    }

    printf("cuda_kernel_makeframe run\n");
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;

    __shared__ bool packet_reached;

    __shared__ uint64_t frame_head;
    __shared__ uint32_t prev_ackn;

    __shared__ uint8_t* cur_tar_buf;
    // __shared__ uint64_t tar_buf_total_size;

    __shared__ uint32_t cur_ackn;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct rx_info* rx_info_global;
    struct store_buf_info* store_buf_global;
    struct ready_buf_info* ready_buf_global;
    struct eth_ip_tcp_hdr* hdr;
    uintptr_t buf_addr;
    uint32_t sem_recvinfo_idx = 0;
    uint32_t sem_frame_idx = 0;
    uint8_t* payload;
    __shared__ bool quit;

    frame_head = 0;
    if (threadIdx.x == 0) {
        DOCA_GPUNETIO_VOLATILE(prev_ackn) = DOCA_GPUNETIO_VOLATILE(*first_ackn);
        packet_reached = false;
        cur_tar_buf = nullptr;
        // tar_buf = nullptr;
        // tar_buf_total_size = 0;
        quit = false;
    }

    if (blockIdx.x != 0) {
        return;
    }

    __syncthreads();

    __shared__ enum doca_gpu_semaphore_status status;
    __shared__ enum doca_gpu_semaphore_status status_frame;

    while ((!quit) && (!DOCA_GPUNETIO_VOLATILE(*is_fin))) {

        if (threadIdx.x == 0) {
            while (!packet_reached) {
                ret = doca_gpu_dev_semaphore_get_status(sem_recvinfo, sem_recvinfo_idx, &status);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP semaphore error");
                    return;
                }
                if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo, sem_recvinfo_idx, (void**)&(rx_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num);
                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx);
                    DOCA_GPUNETIO_VOLATILE(cur_ackn) = DOCA_GPUNETIO_VOLATILE(rx_info_global->cur_ackn);

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
                    packet_reached = true;

                    sem_recvinfo_idx = (sem_recvinfo_idx + 1) % sem_num;
                } else {
                    rx_pkt_num = 0;
                }
            }
        }

        __syncthreads();

        if (!packet_reached)
            continue;

        if ((!cur_tar_buf) && threadIdx.x == 0) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, sem_frame_idx, &status_frame);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status_frame == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
                printf("%d set buf\n", sem_frame_idx);
                cur_tar_buf = tar_buf + sem_frame_idx * pitch;
            }
        }

        __syncthreads();

        if (cur_tar_buf) {

            for (uint64_t idx = rx_buf_idx + threadIdx.x; idx < rx_buf_idx + rx_pkt_num; idx += blockDim.x) {
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

                uint32_t offset = sent_seq - prev_ackn;
                uint64_t cur_head = frame_head + offset;

                if (cur_head < tar_buf_total_size) {
                    uint32_t write_byte = min(cur_head + total_payload_size, tar_buf_total_size) - cur_head;
                    uint8_t* data_head = cur_tar_buf + cur_head;
                    memcpy(data_head, payload, write_byte);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && rx_pkt_num > 0) {
            uint64_t bytes = (cur_ackn - prev_ackn);
            frame_head += bytes;
            if (frame_head > tar_buf_total_size) {
                ret = doca_gpu_dev_semaphore_set_status(sem_frame, sem_frame_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                __threadfence_system();
                printf("%llu frame_head send\n", frame_head);
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                cur_tar_buf = nullptr;
                // quit = true;
            }
            prev_ackn = cur_ackn;
        }

        __syncthreads();
        packet_reached = false;
    }
}

__global__ void frame_notice(
    struct doca_gpu_semaphore_gpu* sem_frame, uint64_t frame_num, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup frame_notice\n");
        }
        return;
    }
    doca_error_t ret;
    __shared__ enum doca_gpu_semaphore_status status;
    if (threadIdx.x == 0) {
        bool fin = false;
        int frame_counter = 0;
        while (!fin) {
            ret = doca_gpu_dev_semaphore_get_status(sem_frame, frame_counter, &status);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }
            if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
                printf("%d kitayo\n", frame_counter);
                ret = doca_gpu_dev_semaphore_set_status(sem_frame, frame_counter, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                fin = (frame_counter == 4);
                frame_counter = (frame_counter + 1) % frame_num;
            }
        }
    }
}

__global__ void cuda_kernel_wait_3wayhandshake(
    uint32_t* out_ackn,
    struct doca_gpu_eth_rxq* rxq,
    struct doca_gpu_eth_txq* txq,
    struct doca_gpu_buf_arr* buf_arr_gpu, uint32_t nbytes)
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

            ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + rx_pkt_num - 1, &buf_ptr);
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

            uint32_t prev_pkt_sz = 1;
            uint32_t ackn = sent_seq + prev_pkt_sz;
            DOCA_GPUNETIO_VOLATILE(*out_ackn) = DOCA_GPUNETIO_VOLATILE(ackn);
            ack_hdr->l4_hdr.recv_ack = BYTE_SWAP32(sent_seq + prev_pkt_sz); // BYTE_SWAP32(hdr->l4_hdr.sent_seq)
            ack_hdr->l4_hdr.sent_seq = hdr->l4_hdr.recv_ack;
            ack_hdr->l4_hdr.cksum = 0;
            ack_hdr->l4_hdr.tcp_flags = TCP_FLAG_SYN | TCP_FLAG_ACK; //| TCP_FLAG_FIN;

            ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, ack_buf, base_pkt_len + nbytes, 0);
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
}

extern "C" {

doca_error_t kernel_receive_tcp(struct rxq_tcp_queues* tcp_queues,
    uint8_t* cpu_tar_buf, uint64_t size, uint64_t pitch, struct sem_pair* sem_frame)
{
    cudaError_t result = cudaSuccess;

    if (tcp_queues == NULL || tcp_queues->numq == 0) {
        DOCA_LOG_ERR("kernel_receive_tcp invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    uint8_t* tar_buf;
    cudaMalloc((void**)&tar_buf, size * sem_frame->nums);
    uint8_t* tmp_buf;
    printf("%d size\n", static_cast<int>(size));
    cudaMalloc((void**)&tmp_buf, size);
    uint32_t* first_ackn;
    cudaMalloc((void**)&first_ackn, sizeof(uint32_t));
    int* is_fin;
    cudaMalloc((void**)&is_fin, sizeof(int));
    cudaMemset(is_fin, 0, sizeof(int));

    cudaStream_t streams[3];

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    cuda_kernel_wait_3wayhandshake<<<1, CUDA_THREADS>>>(
        first_ackn,
        tcp_queues->eth_rxq_gpu[0],
        tcp_queues->eth_txq_gpu[0],
        tcp_queues->tx_buf_arr.buf_arr_gpu, tcp_queues->tx_buf_arr.pkt_nbytes);

    cuda_kernel_receive_tcp<<<2, CUDA_THREADS>>>(
        tcp_queues->eth_rxq_gpu[0],
        tcp_queues->eth_txq_gpu[0],
        tcp_queues->nums,
        tcp_queues->tx_buf_arr.buf_arr_gpu, tcp_queues->tx_buf_arr.pkt_nbytes,
        tcp_queues->sem_gpu[0], is_fin, true);
    cuda_kernel_makeframe<<<1, CUDA_THREADS>>>(
        tar_buf, size, pitch, first_ackn,
        tcp_queues->eth_rxq_gpu[0],
        tcp_queues->nums,
        tcp_queues->sem_gpu[0], is_fin,
        sem_frame->sem_gpu, sem_frame->nums, true);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    frame_notice<<<1, CUDA_THREADS>>>(sem_frame->sem_gpu, sem_frame->nums, true);

    cudaDeviceSynchronize();
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    } else {
        DOCA_LOG_INFO("3wayhandshake\n");
    }

    /* Assume MAX_QUEUES == 4 */
    DOCA_LOG_INFO("kernel_receive_tcp block %d thread %d %d", tcp_queues->numq, CUDA_THREADS, static_cast<int>(sizeof(struct tcp_hdr) + sizeof(struct ipv4_hdr)));
    cuda_kernel_receive_tcp<<<2, CUDA_THREADS, 0, streams[0]>>>(
        tcp_queues->eth_rxq_gpu[0],
        tcp_queues->eth_txq_gpu[0],
        tcp_queues->nums,
        tcp_queues->tx_buf_arr.buf_arr_gpu, tcp_queues->tx_buf_arr.pkt_nbytes,
        tcp_queues->sem_gpu[0], is_fin, false);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cuda_kernel_makeframe<<<1, CUDA_THREADS, 0, streams[1]>>>(
        tar_buf, size, pitch, first_ackn,
        tcp_queues->eth_rxq_gpu[0],
        tcp_queues->nums,
        tcp_queues->sem_gpu[0], is_fin,
        sem_frame->sem_gpu, sem_frame->nums, false);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    frame_notice<<<1, CUDA_THREADS, 0, streams[2]>>>(sem_frame->sem_gpu, sem_frame->nums, false);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_tar_buf, tar_buf, size, cudaMemcpyDeviceToHost);

    return DOCA_SUCCESS;
}

} /* extern C */
