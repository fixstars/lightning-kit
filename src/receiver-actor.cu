
#include <cuda_runtime.h>

// #include <doca_gpunetio_dev_sem.cuh>

#include <rte_gpudev.h>

#include <stdio.h>
#include <vector>

#include "lng/net-header.h"

// #include "lng/doca-util.h"

namespace lng {

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

template <typename T>
__inline__ __device__ T warpAcc(T sum, int lane_id)
{
    auto local = __shfl_up_sync(0xffffffff, sum, 1);
    sum += lane_id < 1 ? 0 : local;
    local = __shfl_up_sync(0xffffffff, sum, 2);
    sum += lane_id < 2 ? 0 : local;
    local = __shfl_up_sync(0xffffffff, sum, 4);
    sum += lane_id < 4 ? 0 : local;
    local = __shfl_up_sync(0xffffffff, sum, 8);
    sum += lane_id < 8 ? 0 : local;
    local = __shfl_up_sync(0xffffffff, sum, 16);
    sum += lane_id < 16 ? 0 : local;

    return sum;
}

#define MAX_THREAD_NUM (1024)
#define MAX_WARP_NUM (MAX_THREAD_NUM / 32)

#define NUM_PAYLOADS 4096

struct udp_payload_header {
    uint32_t seqn;
};

__device__ __inline__ int
raw_to_udp(const uintptr_t buf_addr, struct eth_ip_udp_hdr** hdr, uint8_t** payload)
{
    (*hdr) = (struct eth_ip_udp_hdr*)buf_addr;
    (*payload) = (uint8_t*)(buf_addr + sizeof(struct eth_ip_udp_hdr));

    return 0;
}
__inline__ __device__ struct udp_payload_header get_seqn(uint8_t* p)
{
    struct udp_payload_header ret;
    memcpy(&(ret.seqn), p, sizeof(udp_payload_header));
    return ret;
}

__global__ void cuda_kernel_makeframe(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    // uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    uint32_t* quit_flag_ptr, bool is_warmup, int id)
{
    int frame_num = 2;

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
    __shared__ uint32_t offset_share[MAX_WARP_NUM];
    uint16_t packet_reached_thidx = 0;

    __shared__ uint64_t frame_head;
    __shared__ uint32_t prev_ackn;
    __shared__ uint32_t next_prev_ackn;

    __shared__ uint8_t* cur_tar_buf;

    __shared__ uintptr_t payloads[NUM_PAYLOADS];

    uint32_t cur_ackn;

    // doca_error_t ret;
    // struct fr_info* fr_global;
    struct eth_ip_udp_hdr* hdr;
    uintptr_t buf_addr;
    __shared__ uint32_t comm_list_idx;
    __shared__ uint32_t sem_frame_idx;
    uint8_t* payload;
    __shared__ bool quit;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    frame_head = 0;
    if (threadIdx.x == 0) {
        prev_ackn = 0;
        next_prev_ackn = 0;
        packet_reached = false;
        cur_tar_buf = nullptr;
        quit = false;
        sem_frame_idx = 0;
        comm_list_idx = 0;
    }

    if (blockIdx.x != 0) {
        return;
    }

    __syncthreads();

    // __shared__ enum doca_gpu_semaphore_status status_frame;

    // size_t heart_beat = 0;

    // size_t is_first = 0;

    // auto cl_start = clock();
    size_t bytes_local = 0;
    while ((!quit) && (*quit_flag_ptr == 0)) {

        // if (threadIdx.x == 0 && (heart_beat % ((size_t)100) == 0)) {
        //     printf("heartbeat frame %d\n", threadIdx.x);
        // }
        // heart_beat++;

        // auto cl_start = clock();
        while (true) {

            struct rte_gpu_comm_list* cur_comm_list = &comm_list[(comm_list_idx + threadIdx.x) % comm_list_entries];
            rte_gpu_comm_list_status* p_status = &(cur_comm_list->status_d[0]);
            auto num_pkts = cur_comm_list->num_pkts;

            if (*p_status == RTE_GPU_COMM_LIST_READY) {
                packet_reached_thidx = threadIdx.x + 1;
            } else {
                packet_reached_thidx = 0;
                num_pkts = 0;
            }
            __threadfence();

            uint16_t local_max = warpMax(packet_reached_thidx);
            auto local_offset = warpAcc(num_pkts, lane_id);

            if (lane_id == warpSize - 1) {
                packet_reached_thidx_share[warp_id] = local_max;
                offset_share[warp_id] = local_offset;
            }
            __syncthreads();

            if (threadIdx.x < warpSize) {
                local_max = threadIdx.x < MAX_WARP_NUM ? packet_reached_thidx_share[threadIdx.x] : 0;
                auto tmp = threadIdx.x < MAX_WARP_NUM ? offset_share[threadIdx.x] : 0;
                packet_reached_thidx_share[0] = warpMax(local_max);
                offset_share[threadIdx.x] = warpAcc(tmp, lane_id);
            }

            __syncthreads();

            local_offset += (warp_id == 0 ? 0 : offset_share[warp_id - 1]);
            {
                auto tmp = __shfl_up_sync(0xffffffff, local_offset, 1);
                local_offset = threadIdx.x == 0 ? 0 : (lane_id == 0 ? offset_share[warp_id - 1] : tmp);
            }

            for (int pkt = 0; pkt < num_pkts; ++pkt) {
                payloads[local_offset + pkt] = cur_comm_list->pkt_list[pkt].addr;
            }

            rx_buf_idx_head = 0;
            rx_buf_idx_tail = offset_share[warpSize - 1];

            if (packet_reached_thidx_share[0] > 0) {
                break;
            }
        }

        bool is_head_copy = false;

        if ((!cur_tar_buf) && threadIdx.x == 0) {
            // printf("%d rx_buf_idx_tail\n", rx_buf_idx_tail);

            // ret = doca_gpu_dev_semaphore_get_status(sem_frame, sem_frame_idx, &status_frame);
            // if (ret != DOCA_SUCCESS) {
            //     printf("TCP semaphore error");
            //     return;
            // }
            // if (status_frame == DOCA_GPU_SEMAPHORE_STATUS_FREE) {
            printf("%d %lld set buf\n", sem_frame_idx, frame_head);
            cur_tar_buf = tar_buf + sem_frame_idx * frame_size;
            is_head_copy = true;
            // }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
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

        int64_t pkt_num = rx_buf_idx_tail - rx_buf_idx_head;
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
            for (int64_t idx = rx_buf_st; idx < rx_buf_st + pkt_num_local; ++idx) {
                raw_to_udp(payloads[idx], &hdr, &payload);
                uint32_t sent_seq = get_seqn(payload).seqn;
                payload += sizeof(struct udp_payload_header);
                uint32_t total_payload_size = BYTE_SWAP16(hdr->l4_hdr.dgram_len) - sizeof(struct udp_hdr) - sizeof(struct udp_payload_header);

                // printf("%u sent_seq\n", sent_seq);
                // printf("%u total_payload_size\n", total_payload_size);
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
        if (threadIdx.x < packet_reached_thidx_share[0]) {
            struct rte_gpu_comm_list* cur_comm_list = &comm_list[(comm_list_idx + threadIdx.x) % comm_list_entries];
            cur_comm_list->status_d[0] = RTE_GPU_COMM_LIST_FREE;
        }
        __threadfence();

        if (warp_id == 1 && lane_id == 0) {
            comm_list_idx = (comm_list_idx + packet_reached_thidx_share[0]) % comm_list_entries;
            uint64_t bytes = (next_prev_ackn - prev_ackn);
            // printf("%" PRIu64 " frame_head\n", frame_head);
            // printf("%" PRIu64 " cur_ackn_fin\n", cur_ackn);
            // printf("%" PRIu64 " next_prev_ackn\n", next_prev_ackn);
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
                // ret = doca_gpu_dev_semaphore_get_custom_info_addr(sem_frame, sem_frame_idx, (void**)&(fr_global));
                // DOCA_GPUNETIO_VOLATILE(fr_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(cur_tar_buf);
                // __threadfence_system();

                // ret = doca_gpu_dev_semaphore_set_status(sem_frame, sem_frame_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
                // __threadfence_system();
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
void init_dpdk_udp_framebuilding_kernels(std::vector<cudaStream_t>& streams)
{
    cuda_kernel_makeframe<<<1, 32>>>(
        nullptr, 0, nullptr,
        nullptr, 0,
        // 0, nullptr,
        nullptr, true, 0);

    streams.resize(1);

    cudaStreamCreate(&streams[0]);
}

void launch_dpdk_udp_framebuilding_kernels(
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    // struct semaphore* sem_fr,
    uint32_t* quit_flag_ptr,
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    std::vector<cudaStream_t>& streams)
{
    cuda_kernel_makeframe<<<1, MAX_THREAD_NUM, 0, streams.at(0)>>>(
        tar_buf, frame_size,
        tmp_buf,
        comm_list, comm_list_entries,
        // sem_fr->sem_num, sem_fr->sem_gpu,
        quit_flag_ptr,
        false, 0);
}

}
