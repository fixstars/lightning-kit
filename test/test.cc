#include <iostream>
#include <stdexcept>
#include <vector>

#include "lng/lng.h"

using namespace lng;

int main()
{
    try {
        doca_error_t result;

        Actor actor;

        struct doca_dev* ddev = nullptr;
        uint16_t dpdk_dev_port_id;
        result = init_doca_device("a1:00.0", &ddev, &dpdk_dev_port_id);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function init_doca_device returned " + std::string(doca_error_get_descr(result)));
        }

        struct doca_gpu* gpu_dev = nullptr;
        result = doca_gpu_create("81:00.0", &gpu_dev);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function doca_gpu_create returned " + std::string(doca_error_get_descr(result)));
        }

        constexpr int queue_num = 1;

        auto df_port = init_doca_flow(dpdk_dev_port_id, queue_num);
        if (df_port == nullptr) {
            throw std::runtime_error("Function init_doca_flow failed");
        }

        struct rxq_tcp_queues tcp_queues;
        result = create_tcp_queues(&tcp_queues, df_port, gpu_dev, ddev, queue_num, SEMAPHORES_PER_QUEUE);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function create_tp_queues returned " + std::string(doca_error_get_descr(result)));
        }

        struct sem_pair sem_frame;
#define FRAME_NUM 16
        result = create_sem(gpu_dev, &sem_frame, FRAME_NUM);
        // TODO: Handle semaphore
        // if (result != DOCA_SUCCESS) {
        //     throw std::runtime_error("Function create_sem returned " + std::string(doca_error_get_descr(result)));
        // }

        /* Create root control pipe to route tcp/udp/OS packets */
        result = create_root_pipe(&tcp_queues, df_port);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function create_root_pipe returned " + std::string(doca_error_get_descr(result)));
        }

#define RECV_BYTES MINIMUM_TARBUF_SIZE
        std::vector<uint8_t> tar_buf(RECV_BYTES * FRAME_NUM, 0);

        kernel_receive_tcp(&tcp_queues, tar_buf.data(), RECV_BYTES, RECV_BYTES, &sem_frame);

        result = destroy_flow_queue(dpdk_dev_port_id, df_port, &tcp_queues);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function finalize_doca_flow returned " + std::string(doca_error_get_descr(result)));
        }

        result = doca_gpu_destroy(gpu_dev);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function doca_gpu_destroy returned " + std::string(doca_error_get_descr(result)));
        }

        std::cout << "Passed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
