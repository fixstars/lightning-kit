#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "doca-common-util-internal.h"
#include "lng/doca-util.h"

DOCA_LOG_REGISTER(DOCA_UDP_UTIL);

namespace lng {

static uint64_t default_flow_timeout_usec;

struct doca_flow_port*
init_doca_udp_flow(uint16_t port_id, uint8_t rxq_num)
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
            .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM,
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
        DOCA_LOG_ERR("Failed rte_eth_dev_info_get with: %s", rte_strerror(-ret));
        return NULL;
    }

    ret = rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf);
    if (ret) {
        DOCA_LOG_ERR("Failed rte_eth_dev_configure with: %s", rte_strerror(-ret));
        return NULL;
    }

#define PKTMBUF_CELLSIZE 20480
    // MAX_PKT_SIZE PKTMBUF_CELLSIZE + RTE_PKTMBUF_HEADROOM
    mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, 8246, rte_eth_dev_socket_id(port_id));
    if (mp == NULL) {
        DOCA_LOG_ERR("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
        return NULL;
    }

    tx_conf = dev_info.default_txconf;
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM;
    // struct rte_eth_rxconf rxconf = dev_info.default_rxconf;
    // rxconf.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;

    for (int idx = 0; idx < rxq_num; idx++) {
        ret = rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), NULL, mp); // &rxconf
        if (ret) {
            DOCA_LOG_ERR("Failed rte_eth_rx_queue_setup with: %s", rte_strerror(-ret));
            return NULL;
        }

        ret = rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf);
        if (ret) {
            DOCA_LOG_ERR("Failed rte_eth_tx_queue_setup with: %s", rte_strerror(-ret));
            return NULL;
        }
    }

    ret = rte_flow_isolate(port_id, 1, &error);
    if (ret) {
        DOCA_LOG_ERR("Failed rte_flow_isolate with: %s", error.message);
        return NULL;
    }

    ret = rte_eth_dev_start(port_id);
    if (ret) {
        DOCA_LOG_ERR("Failed rte_eth_dev_start with: %s", rte_strerror(-ret));
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
        DOCA_LOG_ERR("Failed to init doca flow with: %s", doca_error_get_descr(result));
        return NULL;
    }

    /* Start doca flow port */
    port_cfg.port_id = port_id;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str;
    result = doca_flow_port_start(&port_cfg, &df_port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start doca flow port with: %s", doca_error_get_descr(result));
        return NULL;
    }

    return df_port;
}

doca_error_t
create_udp_pipe(struct doca_flow_pipe** pipe, struct rx_queue* rxq, struct doca_flow_port* port, int numq)
{
    doca_error_t result;
    struct doca_flow_match match = { 0 };
    struct doca_flow_match match_mask = { 0 };
    struct doca_flow_fwd fwd = {};
    struct doca_flow_fwd miss_fwd = {};
    struct doca_flow_pipe_entry* entry;
    uint16_t flow_queue_id;
    uint16_t rss_queues[MAX_QUEUES];
    struct doca_flow_monitor monitor = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
    };

    if (pipe == NULL || rxq == NULL || port == NULL || numq > MAX_QUEUES)
        return DOCA_ERROR_INVALID_VALUE;

    struct doca_flow_pipe_cfg pipe_cfg = { 0 };
    pipe_cfg.attr.name = "GPU_RXQ_TCP_PIPE";
    pipe_cfg.attr.enable_strict_matching = true;
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.nb_actions = 0;
    pipe_cfg.attr.is_root = false;
    pipe_cfg.match = &match;
    pipe_cfg.match_mask = &match_mask;
    pipe_cfg.monitor = &monitor;
    pipe_cfg.port = port;

    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

    for (int idx = 0; idx < numq; idx++) {
        doca_eth_rxq_get_flow_queue_id(rxq[idx].eth_rxq_cpu, &flow_queue_id);
        rss_queues[idx] = flow_queue_id;
    }

    fwd.type = DOCA_FLOW_FWD_RSS;
    fwd.rss_queues = rss_queues;
    fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
    fwd.num_of_queues = numq;

    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, pipe);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    /* Add HW offload */
    result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, NULL, &entry);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("RxQ pipe entry creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

// doca_error_t
// destroy_udp_queues(struct rxq_udp_queues* udp_queues)
// {
//     doca_error_t result;

//     if (udp_queues == NULL) {
//         DOCA_LOG_ERR("Can't destroy UDP queues, invalid input");
//         return DOCA_ERROR_INVALID_VALUE;
//     }

//     for (int idx = 0; idx < udp_queues->numq; idx++) {

//         DOCA_LOG_INFO("Destroying UDP queue %d", idx);

//         destroy_semaphore(&(udp_queues->sem[idx]));

//         destroy_rx_queue(&(udp_queues->rxq[idx]));
//     }

//     return DOCA_SUCCESS;
// }

// doca_error_t
// destroy_udp_flow_queue(uint16_t port_id, struct doca_flow_port* port_df,
//     struct rxq_udp_queues* udp_queues)
// {
//     int ret = 0;

//     doca_flow_port_stop(port_df);
//     doca_flow_destroy();

//     destroy_udp_queues(udp_queues);

//     ret = rte_eth_dev_stop(port_id);
//     if (ret != 0) {
//         DOCA_LOG_ERR("Couldn't stop DPDK port %d err %d", port_id, ret);
//         return DOCA_ERROR_DRIVER;
//     }

//     return DOCA_SUCCESS;
// }

doca_error_t create_udp_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_udp_entry, struct doca_flow_pipe* rxq_pipe, struct doca_flow_port* port)
{
    uint32_t priority_high = 1;
    uint32_t priority_low = 3;
    doca_error_t result;
    struct doca_flow_match match_mask = { 0 };
    struct doca_flow_monitor monitor = {};
    monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

    if (root_pipe == NULL || root_udp_entry == NULL || rxq_pipe == NULL || port == NULL)
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
        DOCA_LOG_ERR("Root pipe creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    struct doca_flow_match udp_match = { 0 };
    udp_match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    udp_match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;

    struct doca_flow_fwd udp_fwd = {};
    udp_fwd.type = DOCA_FLOW_FWD_PIPE;
    udp_fwd.next_pipe = rxq_pipe;

    result = doca_flow_pipe_control_add_entry(0, 0, *root_pipe, &udp_match, NULL, NULL, NULL, NULL, NULL, NULL,
        &udp_fwd, NULL, root_udp_entry);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Root pipe entry process failed with: %s", doca_error_get_descr(result));
        return result;
    }

    DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

doca_error_t prepare_udp_tx_buf(struct tx_buf* buf)
{
    uint8_t* cpu_pkt_addr;
    uint8_t* pkt;
    struct eth_ip_udp_hdr* hdr;
    const char* payload = "";
    cudaError_t res_cuda;

    buf->pkt_nbytes = strlen(payload);

    cpu_pkt_addr = (uint8_t*)calloc(buf->num_packets * buf->max_pkt_sz, sizeof(uint8_t));
    if (cpu_pkt_addr == NULL) {
        DOCA_LOG_ERR("Error in txbuf preparation, failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    for (int idx = 0; idx < buf->num_packets; idx++) {
        pkt = cpu_pkt_addr + (idx * buf->max_pkt_sz);
        hdr = (struct eth_ip_udp_hdr*)pkt;

        hdr->l2_hdr.ether_type = rte_cpu_to_be_16(DOCA_ETHER_TYPE_IPV4);

        hdr->l3_hdr.version_ihl = 0x45;
        hdr->l3_hdr.type_of_service = 0x0;
        hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct udp_hdr) + buf->pkt_nbytes);
        hdr->l3_hdr.packet_id = 0;
        hdr->l3_hdr.fragment_offset = rte_cpu_to_be_16(0x4000); // 0;
        hdr->l3_hdr.time_to_live = 0x40; // 60;
        hdr->l3_hdr.next_proto_id = IPPROTO_UDP;
        hdr->l3_hdr.hdr_checksum = 0;
        hdr->l3_hdr.src_addr = 0;
        hdr->l3_hdr.dst_addr = 0;

        hdr->l4_hdr.src_port = 0;
        hdr->l4_hdr.dst_port = 0;
        hdr->l4_hdr.dgram_len = BYTE_SWAP16(sizeof(struct udp_hdr) + buf->pkt_nbytes);
        hdr->l4_hdr.dgram_cksum = 0;
    }

    /* Copy the whole list of packets into GPU memory buffer */
    res_cuda = cudaMemcpy(buf->gpu_pkt_addr, cpu_pkt_addr, buf->num_packets * buf->max_pkt_sz, cudaMemcpyDefault);
    free(cpu_pkt_addr);
    if (res_cuda != cudaSuccess) {
        DOCA_LOG_ERR("Function CUDA Memcpy cqe_addr failed with %s", cudaGetErrorString(res_cuda));
        return DOCA_ERROR_DRIVER;
    }

    return DOCA_SUCCESS;
}

}
