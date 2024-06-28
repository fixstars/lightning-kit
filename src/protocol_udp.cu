

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

#include "lng/doca-util.h"

#include <vector>

DOCA_LOG_REGISTER(PROT_UDP);

namespace lng {

__device__ __inline__ int
raw_to_udp(const uintptr_t buf_addr, struct eth_ip_udp_hdr** hdr, uint8_t** payload)
{
    (*hdr) = (struct eth_ip_udp_hdr*)buf_addr;
    (*payload) = (uint8_t*)(buf_addr + sizeof(struct eth_ip_udp_hdr));

    return 0;
}

__global__ void cuda_kernel_receive_udp(
    struct doca_gpu_eth_rxq* rxq,
    int sem_num,
    struct doca_gpu_semaphore_gpu* sem_recvinfo,
    bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_receive_udp\n");
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
    struct eth_ip_udp_hdr* hdr;
    uintptr_t buf_addr;
    uint64_t buf_idx = 0;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t sem_stats_idx = 0;
    uint8_t* payload;
    uint32_t max_pkts;
    uint64_t timeout_ns;
    uint64_t doca_gpu_buf_idx = laneId;

    __shared__ enum doca_gpu_semaphore_status rx_status;

    max_pkts = MAX_RX_NUM_PKTS;
    timeout_ns = MAX_RX_TIMEOUT_NS;

    if (blockIdx.x >= 2) {
        return;
    }

    __syncthreads();

    if (blockIdx.x == 0) {

        while (true) {

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

            if (threadIdx.x == 0 && rx_pkt_num > 0) {

                while (true) {
                    doca_gpu_dev_semaphore_get_status(sem_recvinfo, sem_stats_idx, &rx_status);

                    if (rx_status == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
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

                        // __threadfence();
                        doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_stats_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                        if (ret != DOCA_SUCCESS) {
                            printf("TCP semaphore recv error\n");
                            return;
                        }
                        __threadfence_system();
                        break;
                    } else {
                        printf("not good to reach here\n");
                    }
                }

                sem_stats_idx = (sem_stats_idx + 1) % sem_num;
            }
            __syncthreads();
        }
    }
}

__global__ void cuda_kernel_makeframe_echo_server(
    struct doca_gpu_eth_rxq* rxq,
    int sem_rx_num, struct doca_gpu_semaphore_gpu* sem_rx_recvinfo,
    int sem_fr_num, struct doca_gpu_semaphore_gpu* sem_fr_recvinfo,
    bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_makeframe\n");
        }
        return;
    }
    // printf("cuda_kernel_makeframe run\n");
    __shared__ uint32_t rx_pkt_num;
    __shared__ uint64_t rx_buf_idx;

    __shared__ bool packet_reached;

    __shared__ uint64_t frame_head;

    // __shared__ uint64_t tar_buf_total_size;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct rx_info* rx_info_global;
    struct fr_info* fr_info_global;
    struct eth_ip_udp_hdr* hdr;
    uintptr_t buf_addr;
    uint32_t sem_rx_recvinfo_idx = 0;
    __shared__ uint32_t sem_fr_recvinfo_idx;
    uint8_t* payload;
    __shared__ bool all_frame_done;

    if (threadIdx.x == 0) {
        frame_head = 0;
        packet_reached = false;
        all_frame_done = false;
        sem_fr_recvinfo_idx = 0;
    }

    if (blockIdx.x != 0) {
        return;
    }

    __syncthreads();

    __shared__ enum doca_gpu_semaphore_status rx_status;
    __shared__ enum doca_gpu_semaphore_status fr_status;

    while (true) {

        if (threadIdx.x == 0) {
            while (!packet_reached) {
                ret = doca_gpu_dev_semaphore_get_status(sem_rx_recvinfo, sem_rx_recvinfo_idx, &rx_status);
                if (ret != DOCA_SUCCESS) {
                    printf("TCP semaphore error");
                    return;
                }
                if (rx_status == DOCA_GPU_SEMAPHORE_STATUS_READY) {

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_rx_recvinfo, sem_rx_recvinfo_idx, (void**)&(rx_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num);
                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx);

                    __threadfence();

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);

                    ret = doca_gpu_dev_semaphore_set_status(sem_rx_recvinfo, sem_rx_recvinfo_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore error\n");
                        return;
                    }
                    __threadfence_system();
                    packet_reached = true;

                    sem_rx_recvinfo_idx = (sem_rx_recvinfo_idx + 1) % sem_rx_num;
                } else {
                    rx_pkt_num = 0;
                }
            }
        }

        __syncthreads();

        if (!packet_reached)
            continue;

        __syncthreads();

        uint32_t tail_sem_idx = (sem_fr_recvinfo_idx + rx_pkt_num - 1) % sem_fr_num;

        while (!all_frame_done) {
            ret = doca_gpu_dev_semaphore_get_status(sem_fr_recvinfo, tail_sem_idx, &fr_status);
            if (ret != DOCA_SUCCESS) {
                printf("fr semaphore failed.");
                return;
            }
            if (fr_status == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
                for (uint64_t idx = rx_buf_idx + threadIdx.x,
                              sem_idx = sem_fr_recvinfo_idx + threadIdx.x;
                     idx < rx_buf_idx + rx_pkt_num;
                     idx += blockDim.x, sem_idx += blockDim.x) {
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

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_fr_recvinfo, sem_idx % sem_fr_num, (void**)&(fr_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }
                    uint8_t* payload = (uint8_t*)buf_addr;

                    DOCA_GPUNETIO_VOLATILE(fr_info_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(payload);

                    __threadfence();

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);

                    ret = doca_gpu_dev_semaphore_set_status(sem_fr_recvinfo, sem_idx % sem_fr_num, DOCA_GPU_SEMAPHORE_STATUS_READY);

                    raw_to_udp(buf_addr, &hdr, &payload);
                    // printf("%d l4_hdr bytes recv\n", BYTE_SWAP16(hdr->l4_hdr.dgram_len));
                    // printf("%d l3_hdr\n", BYTE_SWAP16(hdr->l3_hdr.total_length));
                }
                all_frame_done = true;
            }
        }

        __syncthreads();
        packet_reached = false;
        all_frame_done = false;
        if (threadIdx.x == 0) {
            sem_fr_recvinfo_idx = (sem_fr_recvinfo_idx + rx_pkt_num) % sem_fr_num;
        }
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

#define MAX_THREAD_NUM (1024)
#define MAX_WARP_NUM (MAX_THREAD_NUM / 32)

__inline__ __device__ struct udp_payload_header get_seqn(uint8_t* p)
{
    struct udp_payload_header ret;
    memcpy(&(ret.seqn), p, sizeof(udp_payload_header));
    return ret;
}

__global__ void cuda_kernel_makeframe(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    struct doca_gpu_eth_rxq* rxq,
    int sem_num, struct doca_gpu_semaphore_gpu* sem_recvinfo,
    uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    bool is_warmup, int id)
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
    __shared__ uint32_t next_prev_ackn;

    __shared__ uint8_t* cur_tar_buf;

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct rx_info* rx_info_global;
    struct fr_info* fr_global;
    struct eth_ip_udp_hdr* hdr;
    uintptr_t buf_addr;
    __shared__ uint32_t sem_recvinfo_idx;
    __shared__ uint32_t sem_frame_idx;
    uint8_t* payload;
    __shared__ bool quit;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    frame_head = 0;
    if (threadIdx.x == 0) {
        prev_ackn = 0; // shared with client
        next_prev_ackn = 0;
        packet_reached = false;
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
    while ((!quit)) {

        // if (threadIdx.x == 0 && (heart_beat % ((size_t)100) == 0)) {
        //     printf("heartbeat frame %d\n", threadIdx.x);
        // }
        // heart_beat++;

        // auto cl_start = clock();
        while (true) {

            ret = doca_gpu_dev_semaphore_get_status(sem_recvinfo, (sem_recvinfo_idx + threadIdx.x) % sem_num, &status);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error");
                return;
            }

            if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
                packet_reached_thidx = threadIdx.x + 1;
            } else {
                packet_reached_thidx = 0;
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
                        sem_recvinfo_idx, (void**)&(rx_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx_head) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx);

                    __threadfence();

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);

                    ret = doca_gpu_dev_semaphore_set_status(sem_recvinfo, sem_recvinfo_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore error\n");
                        return;
                    }
                    __threadfence_system();
                } else if (warp_id == 1 && lane_id == 0) {

                    size_t sem_idx = (sem_recvinfo_idx + packet_reached_thidx_share[0] - 1) % sem_num;

                    ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_recvinfo,
                        sem_idx, (void**)&(rx_info_global));
                    if (ret != DOCA_SUCCESS) {
                        printf("TCP semaphore get address error\n");
                        return;
                    }

                    int64_t rx_buf_idx;
                    int32_t rx_pkt_num;

                    DOCA_GPUNETIO_VOLATILE(rx_pkt_num) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_pkt_num);
                    DOCA_GPUNETIO_VOLATILE(rx_buf_idx) = DOCA_GPUNETIO_VOLATILE(rx_info_global->rx_buf_idx);

                    __threadfence();
                    rx_buf_idx_tail = rx_buf_idx + rx_pkt_num;

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);

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

        // if (threadIdx.x == 0) {
        //     printf("%" PRIu64 " rx_buf_idx_head\n", rx_buf_idx_head);
        //     printf("%" PRIu64 " rx_buf_idx_tail\n", rx_buf_idx_tail);
        // }

        // if (is_first == 4) {
        //     printf("%" PRIu64 " rx_buf_st\n", rx_buf_st);
        // }
        // is_first++;

        if (cur_tar_buf) {
            bool is_printed = false;
            for (int64_t idx = rx_buf_st; idx < rx_buf_st + pkt_num_local; ++idx) {

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
                raw_to_udp(buf_addr, &hdr, &payload);
                uint32_t sent_seq = get_seqn(payload).seqn;
                payload += sizeof(struct udp_payload_header);
                uint32_t total_payload_size = BYTE_SWAP16(hdr->l4_hdr.dgram_len) - sizeof(struct udp_hdr) - sizeof(struct udp_payload_header);

                if (idx == rx_buf_idx_tail - 1) {
                    next_prev_ackn = sent_seq + total_payload_size;
                }

                uint32_t offset = sent_seq - prev_ackn;
                uint64_t cur_head = frame_head + offset;

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
                    if ((!is_printed) && cur_head - frame_size + total_payload_size > (size_t)1 * (size_t)512 * 1024 * 1024) {
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
            uint64_t bytes = (next_prev_ackn - prev_ackn);
            // printf("%" PRIu64 " prev_ackn\n", bytes);
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
                printf("%llu %u frame_head send\n", frame_head, packet_reached_thidx_share[0]);
                printf("%u %d pkt_num\n", pkt_num, id);
                sem_frame_idx = (sem_frame_idx + 1) % frame_num;
                cur_tar_buf = nullptr;
                frame_head -= frame_size;
                // quit = true;
            }
            prev_ackn = next_prev_ackn;
        }

        __syncthreads();
        packet_reached = false;
    }
}

__inline__ __device__ void swap_eth(struct ether_hdr* eth)
{
    uint8_t tmp_addr[ETHER_ADDR_LEN];
    memcpy(tmp_addr, eth->d_addr_bytes, ETHER_ADDR_LEN);
    memcpy(eth->d_addr_bytes, eth->s_addr_bytes, ETHER_ADDR_LEN);
    memcpy(eth->s_addr_bytes, tmp_addr, ETHER_ADDR_LEN);
}

__global__ void cuda_kernel_send_packets(
    struct doca_gpu_eth_txq* txq,
    struct doca_gpu_buf_arr* tx_buf_arr,
    int sem_reply_num, struct doca_gpu_semaphore_gpu* sem_reply_recvinfo,
    bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_send_packets\n");
        }
        return;
    }

    doca_error_t ret;
    struct doca_gpu_buf* buf_ptr;
    struct reply_info* reply_info_global;
    struct eth_ip_udp_hdr* hdr;
    uint8_t* buf_addr;
    __shared__ uint32_t sem_reply_recvinfo_idx;
    enum doca_gpu_semaphore_status reply_status;
    __shared__ int32_t max_sent_id[32];
    const uint32_t base_pkt_len = sizeof(struct eth_ip_udp_hdr);

    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_num = blockDim.x / warpSize;
    int th_id = threadIdx.x;

    if (threadIdx.x == 0) {
        sem_reply_recvinfo_idx = 0;
    }

    __syncthreads();

    while (true) {
        uint32_t sem_idx = (sem_reply_recvinfo_idx + threadIdx.x) % sem_reply_num;
        ret = doca_gpu_dev_semaphore_get_status(sem_reply_recvinfo, sem_idx, &reply_status);
        if (ret != DOCA_SUCCESS) {
            printf("TCP semaphore error");
            return;
        }
        int32_t sent_id = -1;
        if (reply_status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
            sent_id = threadIdx.x;
            ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_reply_recvinfo, sem_idx, (void**)&(reply_info_global));
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore get address error\n");
                return;
            }

            DOCA_GPUNETIO_VOLATILE(buf_addr) = DOCA_GPUNETIO_VOLATILE(reply_info_global->eth_payload);

            __threadfence();

            // printf("%d rx_pkt_num frame \n", rx_pkt_num);
            // printf("%d rx_buf_idx frame \n", rx_buf_idx);

            ret = doca_gpu_dev_semaphore_set_status(sem_reply_recvinfo, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
            if (ret != DOCA_SUCCESS) {
                printf("TCP semaphore error\n");
                return;
            }
            __threadfence_system();

            struct doca_gpu_buf* reply_buf = NULL;
            ret = doca_gpu_dev_buf_get_buf(tx_buf_arr, threadIdx.x, &reply_buf);

            uintptr_t reply_buf_addr;
            ret = doca_gpu_dev_buf_get_addr(reply_buf, &reply_buf_addr);

            hdr = (struct eth_ip_udp_hdr*)buf_addr;

            memcpy((uint8_t*)reply_buf_addr, buf_addr, BYTE_SWAP16(hdr->l3_hdr.total_length) + sizeof(struct ether_hdr));

            hdr = (struct eth_ip_udp_hdr*)reply_buf_addr;

            swap_eth(&(hdr->l2_hdr));

            auto tmp_src_addr = hdr->l3_hdr.src_addr;
            hdr->l3_hdr.src_addr = hdr->l3_hdr.dst_addr;
            hdr->l3_hdr.dst_addr = tmp_src_addr;
            auto tmp_src_port = hdr->l4_hdr.src_port;
            hdr->l4_hdr.src_port = hdr->l4_hdr.dst_port;
            hdr->l4_hdr.dst_port = tmp_src_port;

            // printf("%d l4_hdr bytes send", BYTE_SWAP16(hdr->l4_hdr.dgram_len));

            ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, reply_buf, base_pkt_len + BYTE_SWAP16(hdr->l4_hdr.dgram_len) - sizeof(udp_hdr), 0);
            if (ret != DOCA_SUCCESS) {
                printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, lane_id);
                // DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                break;
            }
        }

        sent_id = warpMax(sent_id);
        if (lane_id == 0) {
            max_sent_id[warp_id] = sent_id;
        }
        __syncthreads();
        if (th_id < warpSize) {
            sent_id = th_id < warp_num ? max_sent_id[th_id] : -1;
        }
        if (warp_id == 0) {
            sent_id = warpMax(sent_id);
        }

        if (threadIdx.x == 0 && sent_id >= 0) {
            doca_gpu_dev_eth_txq_commit_strong(txq);
            doca_gpu_dev_eth_txq_push(txq);
        }

        if (threadIdx.x == 0 && sent_id > 0) {
            sem_reply_recvinfo_idx += sent_id;
        }

        __syncthreads();
    }
}

void init_udp_echo_kernels(std::vector<cudaStream_t>& streams)
{
    cuda_kernel_receive_udp<<<1, CUDA_THREADS>>>(
        nullptr, 0, nullptr, true);
    cuda_kernel_makeframe_echo_server<<<1, CUDA_THREADS>>>(
        nullptr, 0, nullptr, 0, nullptr, true);
    cuda_kernel_send_packets<<<1, CUDA_THREADS>>>(
        nullptr, nullptr, 0, nullptr, true);

    streams.resize(3);

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
}

void launch_udp_echo_kernels(struct rx_queue* rxq,
    struct tx_queue* txq,
    struct tx_buf* tx_buf_arr,
    struct semaphore* sem_rx,
    struct semaphore* sem_fr,
    struct semaphore* sem_reply,
    std::vector<cudaStream_t>& streams)
{
    cuda_kernel_receive_udp<<<1, CUDA_THREADS, 0, streams.at(0)>>>(
        rxq->eth_rxq_gpu,
        sem_rx->sem_num,
        sem_rx->sem_gpu, false);

    cuda_kernel_makeframe_echo_server<<<1, CUDA_THREADS, 0, streams.at(1)>>>(
        rxq->eth_rxq_gpu,
        sem_rx->sem_num, sem_rx->sem_gpu,
        sem_fr->sem_num, sem_fr->sem_gpu,
        false);

    cuda_kernel_send_packets<<<1, CUDA_THREADS, 0, streams.at(2)>>>(
        txq->eth_txq_gpu, tx_buf_arr->buf_arr_gpu, sem_reply->sem_num, sem_reply->sem_gpu,
        false);
}

void init_udp_framebuilding_kernels(std::vector<cudaStream_t>& streams)
{
    cuda_kernel_receive_udp<<<1, CUDA_THREADS>>>(
        nullptr, 0, nullptr, true);

    cuda_kernel_makeframe<<<1, CUDA_THREADS>>>(
        nullptr, 0, nullptr,
        nullptr, 0, nullptr, 0, nullptr, true, 0);

    streams.resize(2);

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
}

void launch_udp_framebuilding_kernels(struct rx_queue* rxq,
    struct semaphore* sem_rx,
    struct semaphore* sem_fr,
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    std::vector<cudaStream_t>& streams)
{
    cuda_kernel_receive_udp<<<1, 32, 0, streams.at(0)>>>(
        rxq->eth_rxq_gpu,
        sem_rx->sem_num,
        sem_rx->sem_gpu, false);

    cuda_kernel_makeframe<<<1, MAX_THREAD_NUM, 0, streams.at(1)>>>(
        tar_buf, frame_size,
        tmp_buf,
        rxq->eth_rxq_gpu,
        sem_rx->sem_num, sem_rx->sem_gpu,
        sem_fr->sem_num, sem_fr->sem_gpu,
        false, 0);
}

}
