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

/**

Impl plan

put get model does not suit for doca semaphore model.
I think the reason why nvidia implements semaphoe is that launching kernel is costly and they wants to avoid it.
But we don't want semaphore model as our interface.
So I will use semaphore model in performance sensitive part, and use put get model in other part.

recv_kernel(gpu) <--semaphore--> frame_builder(gpu) <--semaphore--> get(cpu)

put(cpu) <--semaphore--> send_kernel(gpu)

*/

DOCAUDPEchoStream::Impl::Impl(std::string nic_addr, std::string gpu_addr)
    : sem_fr_idx(0)
    , sem_reply_idx(0)
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
    sem_reply.reset(new struct semaphore);
    txq.reset(new struct tx_queue);
    tx_buf_arr.reset(new struct tx_buf);

    create_rx_queue(rxq.get(), gpu_dev, ddev);
    create_semaphore(sem_rx.get(), gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct rx_info), DOCA_GPU_MEM_TYPE_GPU);
    create_semaphore(sem_fr.get(), gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct fr_info), DOCA_GPU_MEM_TYPE_GPU_CPU);
    create_semaphore(sem_reply.get(), gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct reply_info), DOCA_GPU_MEM_TYPE_CPU_GPU);
    create_udp_pipe(&rxq_pipe, rxq.get(), df_port, queue_num);

    std::vector<uint16_t> dst_ports = { 1234 };

    /* Create root control pipe to route tcp/udp/OS packets */
    create_udp_root_pipe(&root_pipe, &root_udp_entry, &rxq_pipe, dst_ports.data(), 1, df_port);

    create_tx_queue(txq.get(), gpu_dev, ddev);

    create_tx_buf(tx_buf_arr.get(), gpu_dev, ddev, TX_BUF_NUM, TX_BUF_MAX_SZ);
    prepare_udp_tx_buf(tx_buf_arr.get());

    std::vector<cudaStream_t> streams;

    init_udp_echo_kernels(streams);
    launch_udp_echo_kernels(rxq.get(), txq.get(), tx_buf_arr.get(), sem_rx.get(), sem_fr.get(), sem_reply.get(), streams);
}

DOCAUDPEchoStream::Impl::~Impl()
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

bool DOCAUDPEchoStream::Impl::put(uint8_t** v, size_t count)
{
    size_t ret = 0;
    struct reply_info* reply_info_global;
    enum doca_gpu_semaphore_status status = DOCA_GPU_SEMAPHORE_STATUS_READY;

    for (; ret < count; ++ret) {
        uint32_t sem_idx = (sem_reply_idx + ret) % sem_reply->sem_num;
        while (status != DOCA_GPU_SEMAPHORE_STATUS_FREE) {
            doca_gpu_semaphore_get_status(sem_reply->sem_cpu, sem_idx, &status);
        }
        doca_gpu_semaphore_get_custom_info_addr(sem_reply->sem_cpu, sem_idx, (void**)&(reply_info_global));
        DOCA_GPUNETIO_VOLATILE(reply_info_global->eth_payload) = DOCA_GPUNETIO_VOLATILE(v[ret]);
        // printf("put %p\n", v[ret]);
        doca_gpu_semaphore_set_status(sem_reply->sem_cpu, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_READY);
    }
    sem_reply_idx = (sem_reply_idx + ret) % sem_reply->sem_num;

    return ret;
}

size_t DOCAUDPEchoStream::Impl::get(uint8_t** vp, size_t max)
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
            DOCA_GPUNETIO_VOLATILE(vp[ret]) = DOCA_GPUNETIO_VOLATILE(fr_info_global->eth_payload);
            // printf("get %p\n", vp[ret]);
            doca_gpu_semaphore_set_status(sem_fr->sem_cpu, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
        }
    }

    sem_fr_idx = (sem_fr_idx + ret) % sem_fr->sem_num;
    // if (ret)
    //     printf("sem_fr_idx %d\n", sem_fr_idx);

    return ret;
}

size_t DOCAUDPEchoStream::count()
{
    // TBD
    return 0;
}

DOCAUDPFrameBuilderStream::Impl::Impl(std::string nic_addr, std::string gpu_addr)
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
    create_semaphore(sem_fr.get(), gpu_dev, frame_num, sizeof(struct fr_info), DOCA_GPU_MEM_TYPE_GPU_CPU);
    create_udp_pipe(&rxq_pipe, rxq.get(), df_port, queue_num);

    std::vector<uint16_t> dst_ports = { 1234 };

    /* Create root control pipe to route tcp/udp/OS packets */
    create_udp_root_pipe(&root_pipe, &root_udp_entry, &rxq_pipe, dst_ports.data(), 1, df_port);

    std::vector<cudaStream_t> streams;

    cudaMalloc((void**)&tar_buf, frame_size * frame_num);
    cudaMalloc((void**)&tmp_buf, tmp_size);

    init_udp_framebuilding_kernels(streams);
    launch_udp_framebuilding_kernels(rxq.get(), sem_rx.get(), sem_fr.get(),
        tar_buf, frame_size, tmp_buf,
        streams);
}

DOCAUDPFrameBuilderStream::Impl::~Impl()
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

bool DOCAUDPFrameBuilderStream::Impl::put(uint8_t** v, size_t count)
{
    size_t ret = 0;

    return ret;
}

size_t DOCAUDPFrameBuilderStream::Impl::get(uint8_t** vp, size_t max)
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
            DOCA_GPUNETIO_VOLATILE(vp[ret]) = DOCA_GPUNETIO_VOLATILE(fr_info_global->eth_payload);
            printf("get %p\n", vp[ret]);
            doca_gpu_semaphore_set_status(sem_fr->sem_cpu, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
        }
    }

    sem_fr_idx = (sem_fr_idx + ret) % sem_fr->sem_num;
    // if (ret)
    //     printf("sem_fr_idx %d\n", sem_fr_idx);

    return ret;
}

size_t DOCAUDPFrameBuilderStream::count()
{
    // TBD
    return 0;
}

DOCATCPStream::Impl::Impl(std::string nic_addr, std::string gpu_addr)
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

    df_port = init_doca_tcp_flow(port_id, queue_num);
    if (df_port == NULL) {
        throw std::runtime_error("FAILED: init_doca_flow");
    }

    rxq.resize(rxq_num);
    sem_rx.resize(rxq_num);
    sem_pay.resize(rxq_num);
    sem_pkt.resize(rxq_num);
    sem_pkt2.resize(rxq_num);
    sem_pkt3.resize(rxq_num);
    sem_pkt4.resize(rxq_num);
    sem_fr.resize(rxq_num);
    txq.resize(rxq_num);
    tx_buf_arr.resize(rxq_num);
    rxq_pipe.resize(rxq_num);
    tar_bufs.resize(rxq_num);
    tmp_buf.resize(rxq_num);

    std::vector<uint16_t> dst_ports = { 1234, 1235 };

    for (int i = 0; i < rxq_num; ++i) {
        create_rx_queue(&rxq[i], gpu_dev, ddev);
        create_semaphore(&sem_rx[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct rx_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_pay[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct pay_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_pkt[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct pay_dist_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_pkt2[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct pay_dist_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_pkt3[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct pay_dist_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_pkt4[i], gpu_dev, SEMAPHORES_PER_QUEUE, sizeof(struct pay_dist_info), DOCA_GPU_MEM_TYPE_GPU);
        create_semaphore(&sem_fr[i], gpu_dev, FRAME_NUM, sizeof(struct tcp_frame_info), DOCA_GPU_MEM_TYPE_GPU_CPU);
        create_tcp_pipe(&rxq_pipe[i], &rxq[i], df_port, queue_num);
    }

    /* Create root control pipe to route tcp/udp/OS packets */
    create_tcp_root_pipe(&root_pipe, &root_udp_entry, rxq_pipe.data(), dst_ports.data(), rxq_num, df_port);

    for (int i = 0; i < rxq_num; ++i) {
        create_tx_queue(&txq[i], gpu_dev, ddev);

        // tx buf size must be over eth/ip/tcp header size
        create_tx_buf(&tx_buf_arr[i], gpu_dev, ddev, TX_BUF_NUM, TX_BUF_MAX_SZ);
        prepare_tcp_tx_buf(&tx_buf_arr[i]);
    }

    cudaMalloc((void**)&is_fin, sizeof(int) * rxq_num);
    cudaMalloc((void**)&first_ackn, sizeof(uint32_t) * rxq_num);
    for (int i = 0; i < rxq_num; ++i) {
        cudaMalloc((void**)&tar_bufs[i], FRAME_SIZE * FRAME_NUM);
        cudaMalloc((void**)&tmp_buf[i], TMP_FRAME_SIZE);
    }

    std::vector<std::vector<cudaStream_t>> streams(rxq_num);

    for (int i = 0; i < rxq_num; ++i) {
        init_tcp_kernels(streams[i]);
    }

    std::vector<std::thread> th(rxq_num);
    cudaMemset(is_fin, 0, sizeof(int) * 2);

    for (int i = 0; i < rxq_num; ++i) {
        th[i] = std::thread([&, i]() {
            launch_tcp_kernels(
                &rxq[i], &txq[i], &tx_buf_arr[i],
                &sem_rx[i], &sem_pay[i], &sem_pkt[i], &sem_pkt2[i], &sem_pkt3[i], &sem_pkt4[i], &sem_fr[i],
                tar_bufs[i], FRAME_SIZE,
                tmp_buf[i],
                first_ackn + i, is_fin + i, streams[i], i);
            for (auto& st : streams[i]) {
                cudaStreamSynchronize(st);
            }
        });
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < rxq_num; ++i) {
        th[i].join();
    }
    exit(0);
}

DOCATCPStream::Impl::~Impl()
{
    doca_error_t result;
    // result = destroy_udp_flow_queue(port_id, df_port, udp_queues.get());
    // if (result != DOCA_SUCCESS) {
    //     throw std::runtime_error("Function finialize_doca_flow returned " + std::string(doca_error_get_descr(result)));
    // }

    cudaFree(first_ackn);
    cudaFree(is_fin);
    for (int i = 0; i < rxq_num; ++i) {
        cudaFree(tar_bufs[i]);
        cudaFree(tmp_buf[i]);
    }

    result = doca_gpu_destroy(gpu_dev);
    if (result != DOCA_SUCCESS) {
        throw std::runtime_error("Failed to destroy GPU: " + std::string(doca_error_get_descr(result)));
    }
}

bool DOCATCPStream::Impl::put(uint8_t** v, size_t count)
{
    size_t ret = 0;
    return ret;
}

size_t DOCATCPStream::Impl::get(uint8_t** vp, size_t max)
{
    size_t ret = 0;
    struct fr_info* fr_info_global;
    enum doca_gpu_semaphore_status status;

    // for (; ret < max; ++ret) {
    //     uint32_t sem_idx = (sem_fr_idx + ret) % sem_fr->sem_num;
    //     doca_gpu_semaphore_get_status(sem_fr->sem_cpu, sem_idx, &status);
    //     if (status != DOCA_GPU_SEMAPHORE_STATUS_READY) {
    //         break;
    //     } else {
    //         doca_gpu_semaphore_get_custom_info_addr(sem_fr->sem_cpu, sem_idx, (void**)&(fr_info_global));
    //         DOCA_GPUNETIO_VOLATILE(vp[ret]) = DOCA_GPUNETIO_VOLATILE(fr_info_global->eth_payload);
    //         printf("get %p\n", vp[ret]);
    //         doca_gpu_semaphore_set_status(sem_fr->sem_cpu, sem_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
    //     }
    // }

    // sem_fr_idx = (sem_fr_idx + ret) % sem_fr->sem_num;
    // if (ret)
    //     printf("sem_fr_idx %d\n", sem_fr_idx);

    return ret;
}

size_t DOCATCPStream::count()
{
    // TBD
    return 0;
}

} // lng

#endif
