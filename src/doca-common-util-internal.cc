

#include "lng/doca-util.h"

#include "log.h"

DOCA_LOG_REGISTER(DOCA_COMMON_UTIL_INTERNAL);

namespace lng {

static uint64_t default_flow_timeout_usec;

doca_error_t destroy_rx_queue(rx_queue* rxq)
{
    doca_error_t result;
    result = doca_ctx_stop(rxq->eth_rxq_ctx);
    if (result != DOCA_SUCCESS) {
        log::error("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_eth_rxq_destroy(rxq->eth_rxq_cpu);
    if (result != DOCA_SUCCESS) {
        log::error("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_mmap_destroy(rxq->pkt_buff_mmap);
    if (result != DOCA_SUCCESS) {
        log::error("Failed to destroy mmap: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_mem_free(rxq->gpu_dev, rxq->gpu_pkt_addr);
    if (result != DOCA_SUCCESS) {
        log::error("Failed to free gpu memory: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

doca_error_t destroy_semaphore(semaphore* sem)
{
    doca_error_t result;
    result = doca_gpu_semaphore_stop(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
        log::error("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_destroy(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
        log::error("Failed doca_gpu_semaphore_destroy: %s", doca_error_get_descr(result));
        return DOCA_ERROR_BAD_STATE;
    }
    return result;
}

struct doca_flow_port*
init_doca_flow(uint16_t port_id, uint8_t rxq_num, uint64_t offload_flags)
{
    doca_error_t result;
    char port_id_str[MAX_PORT_STR_LEN];
    struct doca_flow_port_cfg port_cfg = { 0 };
    struct doca_flow_port* df_port;
    struct doca_flow_cfg rxq_flow_cfg = { 0 };
    int ret = 0;
    struct rte_eth_dev_info dev_info = { 0 };
    struct rte_eth_conf eth_conf = {
        .rxmode = {
            .mtu = 2048, /* Not really used, just to initialize DPDK */
        },
        .txmode = {
            .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | offload_flags,
        },
    };
    struct rte_mempool* mp = NULL;
    struct rte_eth_txconf tx_conf;
    struct rte_flow_error error;

    /*
     * DPDK should be initialized and started before DOCA Flow.
     * DPDK doesn't start the device without, at least, one DPDK Rx queue.
     * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
     *
     * Following lines of code can be considered the minimum WAR for this issue.
     */

    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret) {
        log::error("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
        return NULL;
    }

    ret = rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf);
    if (ret) {
        log::error("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
        return NULL;
    }

#define PKTMBUF_CELLSIZE 20480
    // MAX_PKT_SIZE PKTMBUF_CELLSIZE + RTE_PKTMBUF_HEADROOM
    mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, 8246, rte_eth_dev_socket_id(port_id));
    if (mp == NULL) {
        log::error("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
        return NULL;
    }

    tx_conf = dev_info.default_txconf;
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | offload_flags;
    // struct rte_eth_rxconf rxconf = dev_info.default_rxconf;
    // rxconf.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;

    for (int idx = 0; idx < rxq_num; idx++) {
        ret = rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), NULL, mp); // &rxconf
        if (ret) {
            log::error("Failed rte_eth_rx_queue_setup with: %s", rte_strerror(-ret));
            return NULL;
        }

        ret = rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf);
        if (ret) {
            log::error("Failed rte_eth_tx_queue_setup with: %s", rte_strerror(-ret));
            return NULL;
        }
    }

    ret = rte_flow_isolate(port_id, 1, &error);
    if (ret) {
        log::error("Failed rte_flow_isolate with: %s", error.message);
        return NULL;
    }

    ret = rte_eth_dev_start(port_id);
    if (ret) {
        log::error("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
        return NULL;
    }

    /* Initialize doca flow framework */
    rxq_flow_cfg.pipe_queues = rxq_num;
    /*
     * HWS: Hardware steering
     * Isolated: don't create RSS rule for DPDK created RX queues
     */
    rxq_flow_cfg.mode_args = "vnf,hws,isolated";
    rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

    result = doca_flow_init(&rxq_flow_cfg);
    if (result != DOCA_SUCCESS) {
        log::error("Failed to init doca flow with: %s", doca_error_get_descr(result));
        return NULL;
    }

    /* Start doca flow port */
    port_cfg.port_id = port_id;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str;
    result = doca_flow_port_start(&port_cfg, &df_port);
    if (result != DOCA_SUCCESS) {
        log::error("Failed to start doca flow port with: %s", doca_error_get_descr(result));
        return NULL;
    }

    return df_port;
}

doca_error_t
create_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_entry, struct doca_flow_pipe** rxq_pipe, uint16_t* dst_ports, int rxq_num, struct doca_flow_port* port, doca_flow_l4_type_ext l4_type_ext)
{
    uint32_t priority_high = 1;
    uint32_t priority_low = 3;
    doca_error_t result;
    struct doca_flow_match match_mask = { 0 };
    struct doca_flow_monitor monitor = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
    };

    if (root_pipe == NULL || root_entry == NULL || rxq_pipe == NULL || port == NULL)
        return DOCA_ERROR_INVALID_VALUE;

    struct doca_flow_pipe_cfg pipe_cfg = { 0 };
    pipe_cfg.attr.name = "ROOT_PIPE";
    pipe_cfg.attr.enable_strict_matching = true;
    pipe_cfg.attr.is_root = true;
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
    pipe_cfg.port = port;
    pipe_cfg.monitor = &monitor;
    pipe_cfg.match_mask = &match_mask;

    result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, root_pipe);
    if (result != DOCA_SUCCESS) {
        log::error("Root pipe creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    for (int r = 0; r < rxq_num; ++r) {

        struct doca_flow_match match_gpu = { 0 };
        match_gpu.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
        match_gpu.outer.l4_type_ext = l4_type_ext;
        match_gpu.outer.tcp.l4_port.dst_port = rte_cpu_to_be_16(dst_ports[r]);

        struct doca_flow_fwd fwd_gpu = {
            .type = DOCA_FLOW_FWD_PIPE,
            .next_pipe = rxq_pipe[r],
        };

        result = doca_flow_pipe_control_add_entry(0, 0, *root_pipe, &match_gpu, NULL, NULL, NULL, NULL, NULL, NULL,
            &fwd_gpu, NULL, root_entry);
        if (result != DOCA_SUCCESS) {
            log::error("Root pipe entry creation failed with: %s", doca_error_get_descr(result));
            return result;
        }
    }

    result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
        log::error("Root pipe entry process failed with: %s", doca_error_get_descr(result));
        return result;
    }

    log::debug("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

}
