#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "doca-common-util-internal.h"
#include "lng/doca-util.h"

DOCA_LOG_REGISTER(DOCA_COMMON_UTIL);

namespace lng {
static doca_error_t
open_doca_device_with_pci(const char* pcie_value, struct doca_dev** retval)
{
    struct doca_devinfo** dev_list;
    uint32_t nb_devs;
    doca_error_t res;
    size_t i;
    uint8_t is_addr_equal = 0;

    /* Set default return value */
    *retval = NULL;

    res = doca_devinfo_create_list(&dev_list, &nb_devs);
    if (res != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
        return res;
    }

    /* Search */
    for (i = 0; i < nb_devs; i++) {
        res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
        if (res == DOCA_SUCCESS && is_addr_equal) {
            /* if device can be opened */
            res = doca_dev_open(dev_list[i], retval);
            if (res == DOCA_SUCCESS) {
                doca_devinfo_destroy_list(dev_list);
                return res;
            }
        }
    }

    DOCA_LOG_ERR("Matching device not found");
    res = DOCA_ERROR_NOT_FOUND;

    doca_devinfo_destroy_list(dev_list);
    return res;
}

doca_error_t
init_doca_device(const char* nic_pcie_addr, struct doca_dev** ddev, uint16_t* dpdk_port_id)
{
    doca_error_t result;
    int ret;

    std::vector<std::string> eal_param = { "", "-a", "00:00.0" };
    std::vector<char*> eal_param_;
    for (auto& p : eal_param) {
        eal_param_.push_back(&p[0]);
    }
    eal_param_.push_back(nullptr);

    if (nic_pcie_addr == NULL || ddev == NULL || dpdk_port_id == NULL)
        return DOCA_ERROR_INVALID_VALUE;

    if (strlen(nic_pcie_addr) >= DOCA_DEVINFO_PCI_ADDR_SIZE)
        return DOCA_ERROR_INVALID_VALUE;

    result = open_doca_device_with_pci(nic_pcie_addr, ddev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to open NIC device based on PCI address");
        return result;
    }

    ret = rte_eal_init(eal_param_.size() - 1, eal_param_.data());
    if (ret < 0) {
        DOCA_LOG_ERR("DPDK init failed: %d", ret);
        return DOCA_ERROR_DRIVER;
    }

    /* Enable DOCA Flow HWS mode */
    result = doca_dpdk_port_probe(*ddev, "dv_flow_en=2");
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_dpdk_port_probe returned %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_dpdk_get_first_port_id(*ddev, dpdk_port_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_dpdk_get_first_port_id returned %s", doca_error_get_descr(result));
        return result;
    }

    return DOCA_SUCCESS;
}

doca_error_t create_semaphore(semaphore* sem, struct doca_gpu* gpu_dev, uint32_t sem_num, int element_size, enum doca_gpu_mem_type mem_type)
{
    doca_error_t result;
    result = doca_gpu_semaphore_create(gpu_dev, &(sem->sem_cpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }

    sem->sem_num = sem_num;

    /*
     * Semaphore memory reside on CPU visibile from GPU.
     * CPU will poll in busy wait on this semaphore (multiple reads)
     * while GPU access each item only once to update values.
     */
    result = doca_gpu_semaphore_set_memory_type(sem->sem_cpu, mem_type);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_set_items_num(sem->sem_cpu, sem_num);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }

    /*
     * Semaphore memory reside on CPU visibile from GPU.
     * The CPU reads packets info from this structure.
     * The GPU access each item only once to update values.
     */
    result = doca_gpu_semaphore_set_custom_info(sem->sem_cpu, element_size, DOCA_GPU_MEM_TYPE_CPU_GPU);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_start(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_get_gpu_handle(sem->sem_cpu, &(sem->sem_gpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_error_get_descr(result));
        destroy_semaphore(sem);
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

doca_error_t create_rx_queue(struct rx_queue* rxq, struct doca_gpu* gpu_dev, struct doca_dev* ddev)
{
    uint32_t cyclic_buffer_size = 0;

    rxq->gpu_dev = gpu_dev;

    doca_error_t result;
    result = doca_eth_rxq_create(ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &(rxq->eth_rxq_cpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_set_type(rxq->eth_rxq_cpu, DOCA_ETH_RXQ_TYPE_CYCLIC);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_estimate_packet_buf_size(DOCA_ETH_RXQ_TYPE_CYCLIC, 0, 0, MAX_PKT_SIZE, MAX_PKT_NUM, 0, &cyclic_buffer_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_mmap_create(&rxq->pkt_buff_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_mmap_add_dev(rxq->pkt_buff_mmap, ddev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_mem_alloc(gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_GPU, &rxq->gpu_pkt_addr, NULL);
    if (result != DOCA_SUCCESS || rxq->gpu_pkt_addr == NULL) {
        DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    /* Map GPU memory buffer used to receive packets with DMABuf */
    result = doca_gpu_dmabuf_fd(gpu_dev, rxq->gpu_pkt_addr, cyclic_buffer_size, &(rxq->dmabuf_fd));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB) with nvidia-peermem mode",
            rxq->gpu_pkt_addr, cyclic_buffer_size);

        /* If failed, use nvidia-peermem method */
        result = doca_mmap_set_memrange(rxq->pkt_buff_mmap, rxq->gpu_pkt_addr, cyclic_buffer_size);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_error_get_descr(result));
            destroy_rx_queue(rxq);
            return DOCA_ERROR_BAD_STATE;
        }
    } else {
        DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
            rxq->gpu_pkt_addr, cyclic_buffer_size, rxq->dmabuf_fd);

        result = doca_mmap_set_dmabuf_memrange(rxq->pkt_buff_mmap, rxq->dmabuf_fd, rxq->gpu_pkt_addr, 0, cyclic_buffer_size);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set dmabuf memrange for mmap %s", doca_error_get_descr(result));
            destroy_rx_queue(rxq);
            return DOCA_ERROR_BAD_STATE;
        }
    }

    result = doca_mmap_set_permissions(rxq->pkt_buff_mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set permissions for mmap %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_mmap_start(rxq->pkt_buff_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start mmap %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_set_pkt_buf(rxq->eth_rxq_cpu, rxq->pkt_buff_mmap, 0, cyclic_buffer_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set cyclic buffer  %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    rxq->eth_rxq_ctx = doca_eth_rxq_as_doca_ctx(rxq->eth_rxq_cpu);
    if (rxq->eth_rxq_ctx == NULL) {
        DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_ctx_set_datapath_on_gpu(rxq->eth_rxq_ctx, gpu_dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_ctx_start(rxq->eth_rxq_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_get_gpu_handle(rxq->eth_rxq_cpu, &(rxq->eth_rxq_gpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_error_get_descr(result));
        destroy_rx_queue(rxq);
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

doca_error_t create_tx_queue(struct tx_queue* txq, struct doca_gpu* gpu_dev, struct doca_dev* ddev)
{
    doca_error_t result;
    result = doca_eth_txq_create(ddev, MAX_SQ_DESCR_NUM, &(txq->eth_txq_cpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_txq_set_l3_chksum_offload(txq->eth_txq_cpu, 1);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set eth_txq l3 offloads: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_txq_set_l4_chksum_offload(txq->eth_txq_cpu, 1);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set eth_txq l4 offloads: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    txq->eth_txq_ctx = doca_eth_txq_as_doca_ctx(txq->eth_txq_cpu);
    if (txq->eth_txq_ctx == NULL) {
        DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_ctx_set_datapath_on_gpu(txq->eth_txq_ctx, gpu_dev);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_ctx_start(txq->eth_txq_ctx);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_txq_get_gpu_handle(txq->eth_txq_cpu, &(txq->eth_txq_gpu));
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

doca_error_t create_tx_buf(struct tx_buf* buf, struct doca_gpu* gpu_dev, struct doca_dev* ddev, uint32_t num_packets, uint32_t max_pkt_sz)
{
    doca_error_t status;

    if (buf == NULL || gpu_dev == NULL || ddev == NULL || num_packets == 0 || max_pkt_sz == 0) {
        printf("Invalid input arguments");
        return DOCA_ERROR_INVALID_VALUE;
    }

    buf->gpu_dev = gpu_dev;
    buf->ddev = ddev;
    buf->num_packets = num_packets;
    buf->max_pkt_sz = max_pkt_sz;

    status = doca_mmap_create(&(buf->mmap));
    if (status != DOCA_SUCCESS) {
        printf("Unable to create doca_buf: failed to create mmap");
        return status;
    }

    status = doca_mmap_add_dev(buf->mmap, buf->ddev);
    if (status != DOCA_SUCCESS) {
        printf("Unable to add dev to buf: doca mmap internal error");
        return status;
    }

    auto buf_size = buf->num_packets * buf->max_pkt_sz;

    status = doca_gpu_mem_alloc(buf->gpu_dev, buf_size, 4096, DOCA_GPU_MEM_TYPE_GPU, (void**)(&buf->gpu_pkt_addr), NULL);
    if ((status != DOCA_SUCCESS) || (buf->gpu_pkt_addr == NULL)) {
        printf("Unable to alloc txbuf: failed to allocate gpu memory");
        return status;
    }

    status = doca_gpu_dmabuf_fd(buf->gpu_dev, buf->gpu_pkt_addr, buf_size, &(buf->dmabuf_fd));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB) with nvidia-peermem mode",
            buf->gpu_pkt_addr, buf_size);

        /* If failed, use nvidia-peermem method */
        status = doca_mmap_set_memrange(buf->mmap, buf->gpu_pkt_addr, buf_size);
        if (status != DOCA_SUCCESS) {
            printf("Failed to set memrange for mmap %s", doca_error_get_descr(status));
            return DOCA_ERROR_BAD_STATE;
        }
    } else {
        DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
            buf->gpu_pkt_addr, buf_size, buf->dmabuf_fd);

        status = doca_mmap_set_dmabuf_memrange(buf->mmap, buf->dmabuf_fd, buf->gpu_pkt_addr, 0, buf_size);
        if (status != DOCA_SUCCESS) {
            printf("Failed to set dmabuf memrange for mmap %s", doca_error_get_descr(status));
            return DOCA_ERROR_BAD_STATE;
        }
    }
    // status = doca_mmap_set_memrange(buf->mmap, buf->gpu_pkt_addr, (buf->num_packets * buf->max_pkt_sz));
    // if (status != DOCA_SUCCESS) {
    //     printf("doca_mmap_set_memrange %s", doca_error_get_descr(status));
    //     return status;
    // }

    status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
    if (status != DOCA_SUCCESS) {
        printf("doca_mmap_set_permissions error %s", doca_error_get_descr(status));
        return status;
    }

    status = doca_mmap_start(buf->mmap);
    if (status != DOCA_SUCCESS) {
        printf("doca_mmap_start %s", doca_error_get_descr(status));
        return status;
    }

    status = doca_buf_arr_create(buf->mmap, &buf->buf_arr);
    if (status != DOCA_SUCCESS) {
        printf("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_set_target_gpu(buf->buf_arr, buf->gpu_dev);
    if (status != DOCA_SUCCESS) {
        printf("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_set_params(buf->buf_arr, buf->max_pkt_sz, buf->num_packets, 0);
    if (status != DOCA_SUCCESS) {
        printf("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_start(buf->buf_arr);
    if (status != DOCA_SUCCESS) {
        printf("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_get_gpu_handle(buf->buf_arr, &(buf->buf_arr_gpu));
    if (status != DOCA_SUCCESS) {
        printf("Unable to get buff_arr GPU handle: %s", doca_error_get_descr(status));
        return status;
    }

    return DOCA_SUCCESS;
}

}
