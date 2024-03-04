#include <iostream>
#include <stdexcept>
#include <vector>

#include "lng/lng.h"

using namespace lng;

int main()
{
    try {
        doca_error_t result;

        /* Register a logger backend */
        result = doca_log_backend_create_standard();
        if (result != DOCA_SUCCESS)
            return EXIT_FAILURE;

        static struct doca_gpu* gpu_dev;
        static struct doca_dev* ddev;
        static uint16_t dpdk_dev_port_id;
        static struct doca_flow_port* df_port;
        static struct rxq_udp_queues udp_queues;

        result = init_doca_device("a1:00.0", &ddev, &dpdk_dev_port_id);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function init_doca_device returned " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

        /* Initialize DOCA GPU instance */
        result = doca_gpu_create("81:00.0", &gpu_dev);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function doca_gpu_create returned " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

        int queue_num = 1;

        df_port = init_doca_udp_flow(dpdk_dev_port_id, queue_num);
        if (df_port == NULL) {
            throw std::runtime_error("FAILED: init_doca_flow");
            return EXIT_FAILURE;
        }

        result = create_udp_queues(&udp_queues, df_port, gpu_dev, ddev, queue_num, SEMAPHORES_PER_QUEUE);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function create_udp_queues returned " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

        /* Create root control pipe to route tcp/udp/OS packets */
        result = create_udp_root_pipe(&udp_queues, df_port);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function create_root_pipe returned " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

#define RECV_BYTES MINIMUM_TARBUF_SIZE
        uint8_t* tar_buf = (uint8_t*)malloc(RECV_BYTES);
        memset(tar_buf, 0, RECV_BYTES);

        kernel_receive_udp(&udp_queues, tar_buf, RECV_BYTES, RECV_BYTES);
        // printf("%s\n", (char*)tar_buf);

        result = destroy_udp_flow_queue(dpdk_dev_port_id, df_port, &udp_queues);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Function finialize_doca_flow returned " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

        result = doca_gpu_destroy(gpu_dev);
        if (result != DOCA_SUCCESS) {
            throw std::runtime_error("Failed to destroy GPU: " + std::string(doca_error_get_descr(result)));
            return EXIT_FAILURE;
        }

        std::cout << "Application finished successfully";
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return EXIT_SUCCESS;
}
