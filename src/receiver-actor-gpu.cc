#include <rte_ether.h>
#include <rte_gpudev.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_tcp.h>

// #include <doca_gpunetio.h>

#include <cuda_runtime_api.h>

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
}
