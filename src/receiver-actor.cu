
#include <cuda_runtime.h>

// #include <doca_gpunetio_dev_sem.cuh>

#include <rte_gpudev.h>
#include <rte_tcp.h>

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
__inline__ __device__ T warpSum(T localMax)
{
    localMax += __shfl_xor_sync(0xffffffff, localMax, 16);
    localMax += __shfl_xor_sync(0xffffffff, localMax, 8);
    localMax += __shfl_xor_sync(0xffffffff, localMax, 4);
    localMax += __shfl_xor_sync(0xffffffff, localMax, 2);
    localMax += __shfl_xor_sync(0xffffffff, localMax, 1);

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

__global__ void cuda_kernel_udp_makeframe(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    // uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    uint32_t* quit_flag_ptr, bool is_warmup, int id)
{
    int frame_num = 2;

    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_udp_makeframe\n");
        }
        return;
    }
    if (threadIdx.x == 0) {
        printf("cuda_kernel_udp_makeframe performance\n");
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
    cuda_kernel_udp_makeframe<<<1, 32>>>(
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
    cuda_kernel_udp_makeframe<<<1, MAX_THREAD_NUM, 0, streams.at(0)>>>(
        tar_buf, frame_size,
        tmp_buf,
        comm_list, comm_list_entries,
        // sem_fr->sem_num, sem_fr->sem_gpu,
        quit_flag_ptr,
        false, 0);
}

__device__ __inline__ int
raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr** hdr, uint8_t** payload)
{
    (*hdr) = (struct eth_ip_tcp_hdr*)buf_addr;
    (*payload) = (uint8_t*)(buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

    return 0;
}

__device__ void print_header_cuda(uint8_t* ack)
{
    printf("kokoko\n");
    struct ether_hdr* ack_eth = (struct ether_hdr*)ack;
    printf("Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
        ((uint8_t*)(ack_eth->s_addr_bytes))[0], ((uint8_t*)(ack_eth->s_addr_bytes))[1],
        ((uint8_t*)(ack_eth->s_addr_bytes))[2], ((uint8_t*)(ack_eth->s_addr_bytes))[3],
        ((uint8_t*)(ack_eth->s_addr_bytes))[4], ((uint8_t*)(ack_eth->s_addr_bytes))[5],
        ((uint8_t*)(ack_eth->d_addr_bytes))[0], ((uint8_t*)(ack_eth->d_addr_bytes))[1],
        ((uint8_t*)(ack_eth->d_addr_bytes))[2], ((uint8_t*)(ack_eth->d_addr_bytes))[3],
        ((uint8_t*)(ack_eth->d_addr_bytes))[4], ((uint8_t*)(ack_eth->d_addr_bytes))[5]);
    struct ipv4_hdr* ack_ipv4 = (struct ipv4_hdr*)(ack + sizeof(struct ether_hdr));
    printf("addr %x %x\n", ack_ipv4->src_addr, ack_ipv4->dst_addr);
    struct tcp_hdr* ack_tcp = (struct tcp_hdr*)(ack + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr));
    printf("tcp %d %d %u %u\n",
        ack_tcp->src_port,
        ack_tcp->dst_port,
        ack_tcp->sent_seq,
        ack_tcp->recv_ack);
}

// __global__ void set_header_cuda(uint8_t* ack)
// {
//     struct ether_hdr* ack_eth = (struct ether_hdr*)ack;
//     ack_eth->s_addr_bytes[0] = 0xe8;
//     ack_eth->s_addr_bytes[1] = 0xeb;
//     ack_eth->s_addr_bytes[2] = 0xd3;
//     ack_eth->s_addr_bytes[3] = 0xa7;
//     ack_eth->s_addr_bytes[4] = 0x25;
//     ack_eth->s_addr_bytes[5] = 0xef;
//     ack_eth->d_addr_bytes[0] = 0xa0;
//     ack_eth->d_addr_bytes[1] = 0x88;
//     ack_eth->d_addr_bytes[2] = 0xc2;
//     ack_eth->d_addr_bytes[3] = 0x34;
//     ack_eth->d_addr_bytes[4] = 0xac;
//     ack_eth->d_addr_bytes[5] = 0xf6;
//     struct ipv4_hdr* ack_ipv4 = (struct ipv4_hdr*)(ack + sizeof(struct ether_hdr));
//     ack_ipv4->src_addr = 0x2e1a8c0;
//     ack_ipv4->dst_addr = 0x1e1a8c0;
//     ack_ipv4->total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + 100);
//     struct tcp_hdr* ack_tcp = (struct tcp_hdr*)(ack + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr));
//     ack_tcp->src_port = 53764;
//     ack_tcp->dst_port = 53764;
//     ack_tcp->sent_seq = 0;
//     ack_tcp->recv_ack = 0;
// }

// void set_header_cpu(uint8_t* ack)
// {
//     set_header_cuda<<<1, 1>>>(ack);
// }

// void print_header_cpu(uint8_t* ack)
// {
//     // print_header_cuda<<<1, 1>>>(ack);
// }

__device__ uint32_t make_tcp_headers(uint8_t* ack, uint8_t* org, uint8_t tcp_flags, int len)
{
    memcpy(ack, org, sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr));

    struct ether_hdr* ack_eth = (struct ether_hdr*)ack;
    struct ether_hdr* org_eth = (struct ether_hdr*)org;
    memcpy(ack_eth->d_addr_bytes, org_eth->s_addr_bytes, ETHER_ADDR_LEN);
    memcpy(ack_eth->s_addr_bytes, org_eth->d_addr_bytes, ETHER_ADDR_LEN);

    struct ipv4_hdr* ack_ipv4 = (struct ipv4_hdr*)(ack + sizeof(struct ether_hdr));
    struct ipv4_hdr* org_ipv4 = (struct ipv4_hdr*)(org + sizeof(struct ether_hdr));
    ack_ipv4->src_addr = org_ipv4->dst_addr;
    ack_ipv4->dst_addr = org_ipv4->src_addr;
    ack_ipv4->total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr));

    struct tcp_hdr* ack_tcp = (struct tcp_hdr*)(ack + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr));
    struct tcp_hdr* org_tcp = (struct tcp_hdr*)(org + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr));
    uint32_t ackn = BYTE_SWAP32(org_tcp->sent_seq) + (len > 0 ? len : BYTE_SWAP16(org_ipv4->total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr) + ((org_tcp->tcp_flags & RTE_TCP_FIN_FLAG) ? 1 : 0));

    ack_tcp->src_port = org_tcp->dst_port;
    ack_tcp->dst_port = org_tcp->src_port;
    ack_tcp->sent_seq = org_tcp->recv_ack;
    ack_tcp->recv_ack = BYTE_SWAP32(ackn);
    ack_tcp->tcp_flags = tcp_flags;

    return ackn;
    // print_header_cuda(ack);
}

// __device__ void swap_tcp_headers(uint8_t* ack, uint8_t tcp_flags)
// {

//     struct ether_hdr* ack_eth = (struct ether_hdr*)ack;
//     uint8_t tmp[ETHER_ADDR_LEN];
//     memcpy(tmp, ack_eth->d_addr_bytes, ETHER_ADDR_LEN);
//     memcpy(ack_eth->d_addr_bytes, ack_eth->s_addr_bytes, ETHER_ADDR_LEN);
//     memcpy(ack_eth->s_addr_bytes, tmp, ETHER_ADDR_LEN);

//     struct ipv4_hdr* ack_ipv4 = (struct ipv4_hdr*)(ack + sizeof(struct ether_hdr));
//     auto addr = ack_ipv4->src_addr;
//     ack_ipv4->src_addr = ack_ipv4->dst_addr;
//     ack_ipv4->dst_addr = addr;
//     ack_ipv4->total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr));

//     struct tcp_hdr* ack_tcp = (struct tcp_hdr*)(ack + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr));
//     uint32_t ackn = BYTE_SWAP32(ack_tcp->sent_seq) + BYTE_SWAP16(ack_ipv4->total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr) + ((ack_tcp->tcp_flags & RTE_TCP_FIN_FLAG) ? 1 : 0);

//     auto port = ack_tcp->src_port;
//     ack_tcp->src_port = ack_tcp->dst_port;
//     ack_tcp->dst_port = port;
//     ack_tcp->sent_seq = ack_tcp->recv_ack;
//     ack_tcp->recv_ack = BYTE_SWAP32(ackn);
//     ack_tcp->tcp_flags = tcp_flags;

//     // print_header_cuda(ack);
// }

__global__ void cuda_kernel_tcp_ack(
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    // uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    uint32_t* quit_flag_ptr, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("cuda_kernel_tcp_ack\n");
        }
        return;
    }

    size_t comm_list_idx = 0;
    size_t comm_list_ack_idx = 0;

    uint32_t seqn = 0;
    // size_t heart_beat = 0;

    while ((*quit_flag_ptr == 0) && threadIdx.x == 0) {
        struct rte_gpu_comm_list* cur_comm_list = &comm_list_recv[(comm_list_idx) % comm_list_entries];
        rte_gpu_comm_list_status* p_status = &(cur_comm_list->status_d[0]);
        rte_gpu_comm_list_status is_ready = *p_status;
        __threadfence();

        // if (heart_beat % ((size_t)100000) == 0) {
        //     printf("%lld comm_list_idx\n", comm_list_idx);
        //     printf("%u latest seqn\n", seqn);
        // }
        // heart_beat++;

        if (is_ready == RTE_GPU_COMM_LIST_READY) {

            struct rte_gpu_comm_list* cur_comm_list_ack = &comm_list_ack[(comm_list_ack_idx) % comm_list_ack_entries];
            uint8_t* org = (uint8_t*)cur_comm_list->pkt_list[0].addr;
            uint8_t* ack = (uint8_t*)cur_comm_list_ack->pkt_list[0].addr;

            seqn = make_tcp_headers(ack, org, RTE_TCP_ACK_FLAG, 0);
            // printf("%d comm_list_ack_idx\n", comm_list_ack_idx);
            // printf("%u ACK\n", seqn);

            *p_status = RTE_GPU_COMM_LIST_FREE;
            cur_comm_list_ack->status_d[0] = RTE_GPU_COMM_LIST_FREE;
            __threadfence();

            comm_list_idx++;
            comm_list_ack_idx++;
        }
    }
}

__global__ void cuda_kernel_3way_handshake(
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    uint32_t* seqn,
    uint32_t* quit_flag_ptr, bool is_warmup)
{
    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("cuda_kernel_3way_handshake\n");
        }
        return;
    }

    while ((*quit_flag_ptr == 0) && threadIdx.x == 0) {
        struct rte_gpu_comm_list* cur_comm_list = &comm_list_recv[0];
        rte_gpu_comm_list_status* p_status = &(cur_comm_list->status_d[0]);
        rte_gpu_comm_list_status is_ready = *p_status;
        __threadfence();
        if (is_ready == RTE_GPU_COMM_LIST_READY) {
            struct rte_gpu_comm_list* cur_comm_list_ack = &comm_list_ack[0];
            uint8_t* org = (uint8_t*)cur_comm_list->pkt_list[0].addr;
            uint8_t* ack = (uint8_t*)cur_comm_list_ack->pkt_list[0].addr;

            *seqn = make_tcp_headers(ack, org, RTE_TCP_ACK_FLAG | RTE_TCP_SYN_FLAG, 1);
            printf("send SYN ACK\n");

            cur_comm_list_ack->status_d[0] = RTE_GPU_COMM_LIST_FREE;
            *p_status = RTE_GPU_COMM_LIST_FREE;
            __threadfence();

            break;
        }
    }
}

#define MAX_TCP_THREAD_NUM (512)
#define MAX_TCP_WARP_NUM (MAX_TCP_THREAD_NUM / 32)

__global__ void cuda_kernel_tcp_makeframe(
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    struct rte_gpu_comm_list* comm_list_notify_frame, int frame_num,
    uint32_t* seqn,
    // uint64_t frame_num, struct doca_gpu_semaphore_gpu* sem_frame,
    uint32_t* quit_flag_ptr, bool is_warmup, int id)
{

    if (is_warmup) {
        if (threadIdx.x == 0) {
            printf("warmup cuda_kernel_tcp_makeframe\n");
        }
        return;
    }
    if (threadIdx.x == 0) {
        printf("cuda_kernel_tcp_makeframe performance\n");
    }

    __shared__ int64_t rx_buf_idx_head;
    __shared__ int64_t rx_buf_idx_tail;

    __shared__ bool packet_reached;
    __shared__ uint16_t packet_reached_thidx_share[MAX_TCP_WARP_NUM];
    uint16_t packet_reached_thidx = 0;

    __shared__ uint64_t frame_head;
    __shared__ uint32_t prev_ackn;
    __shared__ uint32_t next_prev_ackn;

    __shared__ uint8_t* cur_tar_buf;

    __shared__ struct rte_gpu_comm_list* comm_list_addr[1024];

    uint32_t cur_ackn;

    // doca_error_t ret;
    // struct fr_info* fr_global;
    struct eth_ip_tcp_hdr* hdr;
    uintptr_t buf_addr;
    __shared__ uint32_t comm_list_idx;
    __shared__ uint32_t sem_frame_idx;
    uint8_t* payload;
    __shared__ bool quit;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    frame_head = 0;
    if (threadIdx.x == 0) {
        prev_ackn = *seqn;
        printf("%d prev_ackn\n", prev_ackn);
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

        // auto cl_start = clock();
        while (true) {

            // if (threadIdx.x == 0 && (heart_beat % ((size_t)1000) == 0)) {
            //     printf("heartbeat frame %d\n", threadIdx.x);
            // }
            // heart_beat++;

            struct rte_gpu_comm_list* cur_comm_list = &comm_list[(comm_list_idx + threadIdx.x) % comm_list_entries];
            rte_gpu_comm_list_status* p_status = &(cur_comm_list->status_d[0]);

            if (*p_status == RTE_GPU_COMM_LIST_READY) {
                packet_reached_thidx = threadIdx.x + 1;
                comm_list_addr[threadIdx.x] = cur_comm_list;
            } else {
                packet_reached_thidx = 0;
            }
            __threadfence();

            uint16_t local_max = warpMax(packet_reached_thidx);

            if (lane_id == warpSize - 1) {
                packet_reached_thidx_share[warp_id] = local_max;
            }
            __syncthreads();

            if (threadIdx.x < warpSize) {
                local_max = threadIdx.x < MAX_TCP_WARP_NUM ? packet_reached_thidx_share[threadIdx.x] : 0;
                packet_reached_thidx_share[0] = warpMax(local_max);
            }
            __syncthreads();

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

        auto commlist_num = packet_reached_thidx_share[0];
        int64_t ceil = (blockDim.x + commlist_num - 1) / commlist_num;
        int64_t floor = blockDim.x / commlist_num;
        int64_t remain = blockDim.x % commlist_num;
        // struct rte_gpu_comm_list* assign_commlist;
        int64_t pkt_offset;
        int64_t pkt_num;
        int64_t commlist_id;
        // if (threadIdx.x == 0) {
        //     printf("%d commlist_num\n", commlist_num);
        // }

        if (threadIdx.x < remain * ceil) {
            commlist_id = threadIdx.x / ceil;
            auto assign_commlist = comm_list_addr[commlist_id];
            auto num_pkts = assign_commlist->num_pkts;
            pkt_num = num_pkts / ceil;
            auto pkt_remain = num_pkts % ceil;
            auto base_th_id = ceil * (threadIdx.x / ceil);
            auto relative_th_id = threadIdx.x - base_th_id;
            pkt_offset = pkt_num * relative_th_id;
            if (relative_th_id < pkt_remain) {
                pkt_num++;
                pkt_offset += relative_th_id;
            } else {
                pkt_offset += pkt_remain;
            }
        } else if (floor > 0) {
            auto relative_th = threadIdx.x - remain * ceil;
            commlist_id = relative_th / floor + remain;
            auto assign_commlist = comm_list_addr[commlist_id];
            auto num_pkts = assign_commlist->num_pkts;
            pkt_num = num_pkts / floor;
            auto pkt_remain = num_pkts % floor;
            auto base_th_id = floor * (relative_th / floor);
            auto relative_th_id = relative_th - base_th_id;
            pkt_offset = pkt_num * relative_th_id;
            if (relative_th_id < pkt_remain) {
                pkt_num++;
                pkt_offset += relative_th_id;
            } else {
                pkt_offset += pkt_remain;
            }
        } else {
            commlist_id = 0;
            pkt_num = 0;
            pkt_offset = 0;
        }

        // printf("%d commlist_id\n", commlist_id);
        // printf("%d pkt_offset\n", pkt_offset);
        // printf("%d pkt_num\n", pkt_num);

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
            for (int64_t idx = pkt_offset; idx < pkt_offset + pkt_num; ++idx) {
                raw_to_tcp(comm_list_addr[commlist_id]->pkt_list[idx].addr, &hdr, &payload);
                volatile uint32_t raw_sent_seq = hdr->l4_hdr.sent_seq;
                uint32_t sent_seq = BYTE_SWAP32(raw_sent_seq);
                uint32_t total_payload_size = BYTE_SWAP16(hdr->l3_hdr.total_length) - sizeof(struct ipv4_hdr) - sizeof(struct tcp_hdr);

                uint32_t offset = sent_seq - prev_ackn;
                uint64_t cur_head = frame_head + offset;

                if (commlist_id == commlist_num - 1 && idx == pkt_offset + pkt_num - 1) {
                    next_prev_ackn = sent_seq + total_payload_size;
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

                comm_list_notify_frame[sem_frame_idx].status_d[0] = RTE_GPU_COMM_LIST_READY;
                __threadfence();

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

void init_dpdk_tcp_framebuilding_kernels(std::vector<cudaStream_t>& streams)
{

    cuda_kernel_3way_handshake<<<1, 32>>>(
        nullptr, 0,
        nullptr, 0,
        nullptr,
        nullptr, true);

    cuda_kernel_tcp_ack<<<1, 32>>>(
        nullptr, 0,
        nullptr, 0,
        nullptr, true);

    cuda_kernel_tcp_makeframe<<<1, 32>>>(
        nullptr, 0, nullptr,
        nullptr, 0,
        nullptr, 0,
        // 0, nullptr,
        nullptr,
        nullptr, true, 0);

    streams.resize(2);

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
}

void cpu_3way_handshake(
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_recv_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    uint32_t* quit_flag_ptr,
    uint32_t* seqn)
{
    cuda_kernel_3way_handshake<<<1, 32>>>(
        comm_list_recv, comm_list_recv_entries,
        comm_list_ack, comm_list_ack_entries,
        seqn,
        quit_flag_ptr, false);
}

void launch_dpdk_tcp_framebuilding_kernels(
    struct rte_gpu_comm_list* comm_list, int comm_list_entries,
    struct rte_gpu_comm_list* comm_list_recv, int comm_list_recv_entries,
    struct rte_gpu_comm_list* comm_list_ack, int comm_list_ack_entries,
    struct rte_gpu_comm_list* comm_list_notify_frame, int frame_entries,
    // struct semaphore* sem_fr,
    uint32_t* quit_flag_ptr,
    uint32_t* seqn,
    uint8_t* tar_buf, size_t frame_size,
    uint8_t* tmp_buf,
    std::vector<cudaStream_t>& streams)
{

    cuda_kernel_tcp_ack<<<1, 32, 0, streams.at(1)>>>(
        comm_list_recv, comm_list_recv_entries,
        comm_list_ack, comm_list_ack_entries,
        quit_flag_ptr, false);

    cuda_kernel_tcp_makeframe<<<1, MAX_TCP_THREAD_NUM, 0, streams.at(0)>>>(
        tar_buf, frame_size,
        tmp_buf,
        comm_list, comm_list_entries,
        comm_list_notify_frame, frame_entries,
        // sem_fr->sem_num, sem_fr->sem_gpu,
        seqn,
        quit_flag_ptr,
        false, 0);
}

}
