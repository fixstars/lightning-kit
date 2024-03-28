

#include "lng/doca-util.h"

DOCA_LOG_REGISTER(DOCA_COMMON_UTIL_INTERNAL);

namespace lng {

doca_error_t destroy_rx_queue(rx_queue* rxq)
{
    doca_error_t result;
    result = doca_ctx_stop(rxq->eth_rxq_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_destroy(rxq->eth_rxq_cpu);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_mmap_destroy(rxq->pkt_buff_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_mem_free(rxq->gpu_dev, rxq->gpu_pkt_addr);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to free gpu memory: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

doca_error_t destroy_semaphore(semaphore* sem)
{
    doca_error_t result;
    result = doca_gpu_semaphore_stop(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_destroy(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_destroy: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

}
