

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

#include "lng/doca-util.h"

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
    int* is_fin, bool is_warmup)
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

            if (threadIdx.x == 0 && rx_pkt_num > 0) {

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

                sem_stats_idx = (sem_stats_idx + 1) % sem_num;
            }
            __syncthreads();
        }
    }
}

__global__ void cuda_kernel_makeframe(
    uint8_t* tar_buf, uint64_t tar_buf_total_size, uint64_t pitch, uint32_t* first_ackn,
    uint8_t* tmp_buf,
    struct doca_gpu_eth_rxq* rxq,
    int sem_num,
    struct doca_gpu_semaphore_gpu* sem_recvinfo,
    int* is_fin, bool is_warmup)
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
    struct eth_ip_udp_hdr* hdr;
    uintptr_t buf_addr;
    uint32_t sem_recvinfo_idx = 0;
    uint8_t* payload;
    __shared__ bool quit;

    if (threadIdx.x == 0) {
        frame_head = 0;
        packet_reached = false;
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

                    __threadfence();

                    // printf("%d rx_pkt_num frame \n", rx_pkt_num);
                    // printf("%d rx_buf_idx frame \n", rx_buf_idx);

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

        __syncthreads();

        {

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
                raw_to_udp(buf_addr, &hdr, &payload);
                printf("%d l4_hdr\n", BYTE_SWAP16(hdr->l4_hdr.dgram_len));
                printf("%d l3_hdr\n", BYTE_SWAP16(hdr->l3_hdr.total_length));
                // uint32_t sent_seq = BYTE_SWAP32(hdr->l4_hdr.sent_seq);
                // uint32_t total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

                // uint32_t offset = sent_seq - prev_ackn;
                // uint64_t cur_head = frame_head + offset;

                // if (cur_head + total_payload_size < tar_buf_total_size) {
                //     uint32_t write_byte = total_payload_size;
                //     uint8_t* data_head = tar_buf + cur_head;
                //     memcpy(data_head, payload, write_byte);
                // } else if (cur_head < tar_buf_total_size) {
                //     uint32_t write_byte = tar_buf_total_size - cur_head;
                //     uint8_t* data_head = tar_buf + cur_head;
                //     memcpy(data_head, payload, write_byte);
                //     memcpy(tmp_buf, payload + write_byte, total_payload_size - write_byte);
                // } else {
                //     uint32_t write_byte = total_payload_size;
                //     uint8_t* data_head = tmp_buf + cur_head - tar_buf_total_size;
                //     // printf("%d koko\n", (cur_head - tar_buf_total_size));
                //     // memcpy(data_head, payload, write_byte);
                // }
            }
        }
        // __syncthreads();
        // if (threadIdx.x == 0 && rx_pkt_num > 0) {
        //     uint64_t bytes = (cur_ackn - prev_ackn);
        //     frame_head += bytes;
        //     if (frame_head > tar_buf_total_size) {
        //         printf("%d frame made\n", frame_head);
        //     }
        //     prev_ackn = cur_ackn;
        // }

        __syncthreads();
        packet_reached = false;
    }
}

extern "C" {

doca_error_t kernel_receive_udp(struct rxq_udp_queues* udp_queues,
    uint8_t* cpu_tar_buf, uint64_t size, uint64_t pitch)
{
    cudaError_t result = cudaSuccess;

    if (udp_queues == NULL || udp_queues->numq == 0) {
        DOCA_LOG_ERR("kernel_receive_udp invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    uint8_t* tar_buf;
    cudaMalloc((void**)&tar_buf, size);
    uint8_t* tmp_buf;
    printf("%d size\n", static_cast<int>(size));
    cudaMalloc((void**)&tmp_buf, size);
    uint32_t* first_ackn;
    cudaMalloc((void**)&first_ackn, sizeof(uint32_t));
    int* is_fin;
    cudaMalloc((void**)&is_fin, sizeof(int));
    cudaMemset(is_fin, 0, sizeof(int));

    cuda_kernel_receive_udp<<<2, CUDA_THREADS>>>(
        udp_queues->eth_rxq_gpu[0],
        udp_queues->nums,
        udp_queues->sem_gpu[0], is_fin, true);
    cuda_kernel_makeframe<<<1, CUDA_THREADS>>>(
        tar_buf, size, pitch, first_ackn,
        tmp_buf,
        udp_queues->eth_rxq_gpu[0],
        udp_queues->nums,
        udp_queues->sem_gpu[0], is_fin, true);

    cudaStream_t streams[2];

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    /* Assume MAX_QUEUES == 4 */
    DOCA_LOG_INFO("kernel_receive_udp block %d thread %d %d", udp_queues->numq, CUDA_THREADS, static_cast<int>(sizeof(struct tcp_hdr) + sizeof(struct ipv4_hdr)));
    cuda_kernel_receive_udp<<<2, CUDA_THREADS, 0, streams[0]>>>(
        udp_queues->eth_rxq_gpu[0],
        udp_queues->nums,
        udp_queues->sem_gpu[0], is_fin, false);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cuda_kernel_makeframe<<<1, CUDA_THREADS, 0, streams[1]>>>(
        tar_buf, size, pitch, first_ackn,
        tmp_buf,
        udp_queues->eth_rxq_gpu[0],
        udp_queues->nums,
        udp_queues->sem_gpu[0], is_fin, false);
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaDeviceSynchronize();
    cudaMemcpy(cpu_tar_buf, tar_buf, size, cudaMemcpyDeviceToHost);

    return DOCA_SUCCESS;
}

} /* extern C */
}
