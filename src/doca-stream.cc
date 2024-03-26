#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)

#include "lng/lng.h"
#include "lng/stream.h"

#include "lng/doca-kernels.h"

#include "log.h"

#include <doca_buf_array.h>
#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_version.h>

#include <rte_ethdev.h>

namespace lng {

DOCAStream::Impl::Impl(std::string nic_addr, std::string gpu_addr)
    : sem_fr_idx(0)
{

    doca_error_t result;
    result = init_doca_device(nic_addr.c_str(), &ddev, &port_id);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error("Function init_doca_device returned " + std::string(doca_error_get_descr(result)));
    }

    /* Initialize DOCA GPU instance */
    result = doca_gpu_create(gpu_addr.c_str(), &gpu_dev);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error("Function doca_gpu_create returned " + std::string(doca_error_get_descr(result)));
    }

    int queue_num = 1;

    df_port = init_doca_udp_flow(port_id, queue_num);
    if (df_port == NULL) {
        throw std::runtime_error("FAILED: init_doca_flow");
    }

    rxq.reset(new struct rx_queue);
    sem_rx.reset(new struct semaphore);
    sem_fr.reset(new struct semaphore);

    create_rx_queue(rxq.get(), gpu_dev, ddev);
    create_semaphore(sem_rx.get(), gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct rx_info), DOCA_GPU_MEM_TYPE_GPU);
    create_semaphore(sem_fr.get(), gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct fr_info), DOCA_GPU_MEM_TYPE_GPU_CPU);
    create_udp_pipe(&rxq_pipe, rxq.get(), df_port, queue_num);

    /* Create root control pipe to route tcp/udp/OS packets */
    result = create_udp_root_pipe(&root_pipe, &root_udp_entry, rxq_pipe, df_port);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error("Function create_root_pipe returned " + std::string(doca_error_get_descr(result)));
    }

    std::vector<cudaStream_t> streams;

    init_udp_kernels(streams);
    launch_udp_kernels(rxq.get(), sem_rx.get(), sem_fr.get(), streams);
}

DOCAStream::Impl::~Impl()
{
    doca_error_t result;
    // result = destroy_udp_flow_queue(port_id, df_port, udp_queues.get());
    // if (result != DOCA_SUCCESS) {
    //     throw std::runtime_error("Function finialize_doca_flow returned " + std::string(doca_error_get_descr(result)));
    // }

    result = doca_gpu_destroy(gpu_dev);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error("Failed to destroy GPU: " + std::string(doca_error_get_descr(result)));
    }
}

bool DOCAStream::Impl::put(uint8_t** v, size_t count)
{
    // auto nb = rte_eth_tx_burst(impl_->port_id, 0, &v, 1);
    // if (nb != 1) {
    //     throw std::runtime_error("rte_eth_tx_burst");
    // }
    return false;
}

size_t DOCAStream::Impl::get(uint8_t*** vp, size_t max)
{
    size_t ret = 0;
    struct fr_info* fr_info_global;
    enum doca_gpu_semaphore_status status;

    for (; ret < max; ++ret) {
        uint32_t sem_idx = (sem_fr_idx + ret) % sem_fr->sem_num;
        doca_gpu_semaphore_get_status(sem_fr->sem_cpu, sem_idx, &status);
        if (status != DOCA_GPU_SEMAPHORE_STATUS_READY) {
            break;
        } else {
            doca_gpu_semaphore_get_custom_info_addr(sem_fr->sem_cpu, sem_idx, (void**)&(fr_info_global));
            vp_internal[ret] = fr_info_global->eth_payload;
            doca_gpu_semaphore_set_status(sem_fr->sem_cpu, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
        }
    }

    *vp = vp_internal;

    sem_fr_idx = (sem_fr_idx + ret) % sem_fr->sem_num;

    return ret;
}

/**

Impl plan

put get model does not suit for doca semaphore model.
I think the reason why nvidia implements semaphoe is that launching kernel is costly and they wants to avoid it.
But we don't want semaphore model as our interface.
So I will use semaphore model in performance sensitive part, and use put get model in other part.

recv_kernel(gpu) <--semaphore--> frame_builder(gpu) <--semaphore--> get(cpu)

put(cpu) <--semaphore--> send_kernel(gpu)

*/

} // lng

#endif
