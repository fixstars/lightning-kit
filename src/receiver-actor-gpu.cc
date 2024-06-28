#include <rte_ether.h>
#include <rte_gpudev.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_tcp.h>

// #include <doca_gpunetio.h>

#include <cuda_runtime_api.h>

#include <fstream>

#include "lng/doca-kernels.h" // temporary
// #include "lng/doca-util.h" // temporary
#include "lng/net-header.h"
#include "lng/receiver-actor-gpu.h"

#include "log.h"
namespace lng {

void ReceiverGPUUDP::setup()
{
    init_dpdk_udp_framebuilding_kernels(streams_);

    // auto result = doca_gpu_create("03:02.0", &gpu_dev_);
    // if (result != DOCA_SUCCESS) {
    //     throw std::runtime_error("Function doca_gpu_create returned " + std::string(doca_error_get_descr(result)));
    // }

    int gpu_dev_id = 0; // TODO share with runtime.cc

    quit_flag_.reset(new struct rte_gpu_comm_flag);

    mbufs.resize(num_entries);
#define PACKT_NUM_AT_ONCE 4096
    for (auto& mbuf : mbufs) {
        mbuf.reset(new rte_mbuf*[PACKT_NUM_AT_ONCE]);
    }

    mbufs_num.resize(num_entries);

    rte_gpu_comm_create_flag(gpu_dev_id, quit_flag_.get(), RTE_GPU_COMM_FLAG_CPU);
    rte_gpu_comm_set_flag(quit_flag_.get(), 0);

    comm_list_ = rte_gpu_comm_create_list(gpu_dev_id, num_entries);

    // sem_fr.reset(new struct semaphore);
    // create_semaphore(sem_fr.get(), gpu_dev_, SEMAPHORES_PER_QUEUE, sizeof(struct fr_info), DOCA_GPU_MEM_TYPE_GPU_CPU);

    cudaMalloc((void**)&tar_bufs_, FRAME_SIZE * FRAME_NUM);
    cudaMalloc((void**)&tmp_buf_, TMP_FRAME_SIZE);

    launch_dpdk_udp_framebuilding_kernels(
        comm_list_, num_entries,
        // sem_fr.get(),
        quit_flag_->ptr,
        tar_bufs_, FRAME_SIZE,
        tmp_buf_,
        streams_);
}

void ReceiverGPUUDP::main()
{
    // if (!payload_) {
    //     if (!ready_payload_stream_->get(&payload_, 1)) {
    //         return;
    //     }
    //     payload_->Clear();
    // }

    int nb;

    rte_mbuf** v = mbufs.at(comm_list_idx_ % num_entries).get();

    if ((nb = nic_stream_->get(v, PACKT_NUM_AT_ONCE)) == 0) {
        return;
    }

    // log::info("{} comm_list_idx_", comm_list_idx_);

    rte_gpu_comm_populate_list_pkts(comm_list_ + (comm_list_idx_ % num_entries), v, nb);
    mbufs_num.at(comm_list_idx_ % num_entries) = nb;

    comm_list_idx_++;

    // log::info("kokotootta {} {}", nb, nic_stream_->count());

    while (comm_list_free_idx_ < comm_list_idx_) {

        rte_gpu_comm_list_status status;
        rte_gpu_comm_get_status(comm_list_ + (comm_list_free_idx_ % num_entries), &status);
        if (status == RTE_GPU_COMM_LIST_READY) {
            break;
        }

        static int hoge = -1;
        hoge++;
        if (hoge % 100 == 0)
            log::info("{} {} {} num", comm_list_free_idx_, comm_list_idx_, mbufs_num.at(comm_list_free_idx_ % num_entries));
        rte_pktmbuf_free_bulk(mbufs.at(comm_list_free_idx_ % num_entries).get(), mbufs_num.at(comm_list_free_idx_ % num_entries));
        rte_gpu_comm_cleanup_list(comm_list_ + (comm_list_free_idx_ % num_entries));
        comm_list_free_idx_++;
    }

    // if (!nic_stream_->check_target_packet(v)) {
    //     return;
    // }

    // TODO detect FIN and quit
    // auto len = payload_->ExtractPayload(v);

    // nic_stream_->send_ack(v, len);

    // valid_payload_stream_->put(&payload_, 1);

    // payload_ = nullptr;
}

static inline void set_affinity(int cpu_id)
{
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(cpu_id, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

// #define ONLY_ACK

void ReceiverGPUTCP::setup()
{
    init_dpdk_tcp_framebuilding_kernels(streams_);

    // auto result = doca_gpu_create("03:02.0", &gpu_dev_);
    // if (result != DOCA_SUCCESS) {
    //     throw std::runtime_error("Function doca_gpu_create returned " + std::string(doca_error_get_descr(result)));
    // }

    int gpu_dev_id = 0; // TODO share with runtime.cc

    quit_flag_.reset(new struct rte_gpu_comm_flag);

    mbufs.resize(num_entries);
#define TCP_PACKT_NUM_AT_ONCE 256
    for (auto& mbuf : mbufs) {
        mbuf.reset(new rte_mbuf*[TCP_PACKT_NUM_AT_ONCE]);
    }

    mbufs_num.resize(num_entries);

    rte_gpu_comm_create_flag(gpu_dev_id, quit_flag_.get(), RTE_GPU_COMM_FLAG_CPU);
    rte_gpu_comm_set_flag(quit_flag_.get(), 0);

    comm_list_ = rte_gpu_comm_create_list(gpu_dev_id, num_entries);
    comm_list_ack_ref_ = rte_gpu_comm_create_list(gpu_dev_id, num_ack_entries);
    comm_list_ack_pkt_ = rte_gpu_comm_create_list(gpu_dev_id, num_ack_entries);
    comm_list_notify_frame_ = rte_gpu_comm_create_list(gpu_dev_id, FRAME_NUM);

    for (int i = 0; i < num_ack_entries; ++i) {
        auto buf = nic_stream_->alloc_ack_mbuf();
        rte_gpu_comm_populate_list_pkts(comm_list_ack_pkt_ + i, &buf, 1);

        // log::info("{} num_ack_en", i);
        // set_header_cpu(rte_pktmbuf_mtod_offset(buf, uint8_t*, 0));
        // cudaDeviceSynchronize();
        // nic_stream_->put(&buf, 1);
    }

    for (int i = 0; i < FRAME_NUM; ++i) {
        rte_gpu_comm_set_status(comm_list_notify_frame_ + i, RTE_GPU_COMM_LIST_FREE);
    }

    comm_list_idx_ = std::make_shared<std::atomic<size_t>>(0);

    ack_thread.reset(new std::thread([=]() {
        size_t idx = 0;
        bool is_3way_handshake = true; // TODO
        set_affinity(0);
        // size_t heart_beat = 0;
        while (true) {
            rte_gpu_comm_list_status status;
            struct rte_gpu_comm_list* cur_comm = comm_list_ack_pkt_ + (idx % num_ack_entries);
            rte_gpu_comm_get_status(cur_comm, &status);

            // if (heart_beat % ((size_t)100000) == 0) {
            //     printf("%lld ack\n", idx);
            // }
            // heart_beat++;

            if (status == RTE_GPU_COMM_LIST_READY) {
                continue;
            }

            // log::info("{} tx", idx);

            nic_stream_->put(cur_comm->mbufs, 1);
            rte_wmb();
            // rte_pktmbuf_free(cur_comm->mbufs[0]);

            rte_gpu_comm_cleanup_list(cur_comm);

            rte_mbuf* new_tmp = nic_stream_->alloc_ack_mbuf();
            rte_gpu_comm_populate_list_pkts(cur_comm, &new_tmp, 1);

#ifdef ONLY_ACK
            rte_pktmbuf_free_bulk(comm_list_ack_ref_[idx % num_ack_entries].mbufs, 1);
#endif

            // rte_pktmbuf_free(comm_list_ack_ref_[idx % num_ack_entries].mbufs[0]);
            rte_gpu_comm_cleanup_list(comm_list_ack_ref_ + (idx % num_ack_entries));

            if (!is_3way_handshake)
                idx++;
            is_3way_handshake = false;
        }
    }));

#ifndef ONLY_ACK
    free_thread.reset(new std::thread([=]() {
        set_affinity(2);
        while (true) {
            while (comm_list_free_idx_ < *comm_list_idx_) {

                rte_gpu_comm_list_status status;
                rte_gpu_comm_get_status(comm_list_ + (comm_list_free_idx_ % num_entries), &status);
                if (status == RTE_GPU_COMM_LIST_READY) {
                    break;
                }

                static int hoge = -1;
                hoge++;
                if (hoge % 1000 == 0) {
                    size_t a = *comm_list_idx_;
                    log::info("{} {} {} num", comm_list_free_idx_, a, comm_list_[comm_list_free_idx_ % num_entries].num_pkts);
                }
                rte_pktmbuf_free_bulk(comm_list_[comm_list_free_idx_ % num_entries].mbufs, comm_list_[comm_list_free_idx_ % num_entries].num_pkts);
                rte_gpu_comm_cleanup_list(comm_list_ + (comm_list_free_idx_ % num_entries));
                comm_list_free_idx_++;
            }
        }
    }));
#endif

    // sem_fr.reset(new struct semaphore);
    // create_semaphore(sem_fr.get(), gpu_dev_, SEMAPHORES_PER_QUEUE, sizeof(struct fr_info), DOCA_GPU_MEM_TYPE_GPU_CPU);

    cudaMalloc((void**)&tar_bufs_, FRAME_SIZE * FRAME_NUM);
    cudaMalloc((void**)&tmp_buf_, TMP_FRAME_SIZE);
    cudaMalloc((void**)&seqn_, sizeof(uint32_t));
    tar_bufs_cpu_ = (uint8_t*)malloc(FRAME_SIZE);

    wait_3wayhandshake();

    cudaStreamCreate(&stream_cpy_);

    launch_dpdk_tcp_framebuilding_kernels(
        comm_list_, num_entries,
        comm_list_ack_ref_, num_ack_entries,
        comm_list_ack_pkt_, num_ack_entries,
        comm_list_notify_frame_, FRAME_NUM,
        // sem_fr.get(),
        quit_flag_->ptr,
        seqn_,
        tar_bufs_, FRAME_SIZE,
        tmp_buf_,
        streams_);
}

void ReceiverGPUTCP::wait_3wayhandshake()
{
    cpu_3way_handshake(
        comm_list_ack_ref_, num_ack_entries,
        comm_list_ack_pkt_, num_ack_entries,
        quit_flag_->ptr,
        seqn_);

    rte_mbuf* v[16];
    while (true) {
        int nb;
        // SYN
        if ((nb = nic_stream_->get(v, TCP_PACKT_NUM_AT_ONCE)) == 0) {
            continue;
        }
        rte_gpu_comm_populate_list_pkts(comm_list_ack_ref_, v + nb - 1, 1);
        break;
    }
    log::info("SYN");
    while (true) {
        int nb;
        // ACK
        if ((nb = nic_stream_->get(v, TCP_PACKT_NUM_AT_ONCE)) == 0) {
            continue;
        }
        rte_pktmbuf_free_bulk(v, nb);
        break;
    }
    log::info("ACK");
    cudaDeviceSynchronize();
}

void ReceiverGPUTCP::main()
{
    // if (!payload_) {
    //     if (!ready_payload_stream_->get(&payload_, 1)) {
    //         return;
    //     }
    //     payload_->Clear();
    // }

    int nb;

    rte_mbuf** v = mbufs.at((*comm_list_idx_) % num_entries).get();

    // static int prev = nic_stream_->count();

    if ((nb = nic_stream_->get(v, TCP_PACKT_NUM_AT_ONCE)) == 0) {
        return;
    }

    // auto prev_clone = nic_stream_->count();

    rte_mbuf* ack_ref = v[nb - 1]; // nic_stream_->clone_ack_mbuf(v[nb - 1]);
    // log::info("koko");
    rte_gpu_comm_populate_list_pkts(comm_list_ack_ref_ + ((*comm_list_idx_) % num_ack_entries), &ack_ref, 1);
#ifdef ONLY_ACK
    rte_pktmbuf_free_bulk(v, nb - 1);
    (*comm_list_idx_)++;
#else
    rte_gpu_comm_populate_list_pkts(comm_list_ + ((*comm_list_idx_) % num_entries), v, nb);
    // mbufs_num.at(comm_list_idx_ % num_entries) = nb;

    // log::info("{} {} {} {} {} comm_list_idx_", comm_list_idx_, prev, prev_clone, nic_stream_->count(), nb);
    // log::info("{} {} {} comm_list_idx_", comm_list_idx_, nic_stream_->count(), nb);

    (*comm_list_idx_)++;

    // log::info("kokotootta {} {}", nb, nic_stream_->count());

    rte_gpu_comm_list_status status;
    rte_gpu_comm_get_status(comm_list_notify_frame_ + (comm_list_frame_idx_ % FRAME_NUM), &status);
    if (status == RTE_GPU_COMM_LIST_READY) {

        if (OUTPUT_TO_FILE && comm_list_frame_idx_ == 8) {
            log::info("outputfile");
            auto result_buf = tar_bufs_ + FRAME_SIZE * (comm_list_frame_idx_ % FRAME_NUM);
            cudaMemcpyAsync(tar_bufs_cpu_, result_buf, FRAME_SIZE, cudaMemcpyDeviceToHost, stream_cpy_);
            cudaStreamSynchronize(stream_cpy_);
            std::ofstream ofs("outfile.dat");
            log::info("ofs write");
            ofs.write((char*)tar_bufs_cpu_, FRAME_SIZE);
        }

        rte_gpu_comm_set_status(comm_list_notify_frame_ + (comm_list_frame_idx_ % FRAME_NUM), RTE_GPU_COMM_LIST_FREE);
        comm_list_frame_idx_++;
    }
#endif

    // prev = nic_stream_->count();

    // if (!nic_stream_->check_target_packet(v)) {
    //     return;
    // }

    // TODO detect FIN and quit
    // auto len = payload_->ExtractPayload(v);

    // nic_stream_->send_ack(v, len);

    // valid_payload_stream_->put(&payload_, 1);

    // payload_ = nullptr;
}

}
