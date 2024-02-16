#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "lng/doca-util.h"

DOCA_LOG_REGISTER(DOCA2);

namespace lng {

static uint64_t default_flow_timeout_usec;

#ifdef DOCA22
static doca_error_t get_dpdk_port_id_doca_dev(struct doca_dev* dev_input, uint16_t* port_id)
{
    struct doca_dev* dev_local = NULL;
    char pci_addr_input[DOCA_DEVINFO_PCI_ADDR_SIZE];
    doca_error_t result;
    uint16_t dpdk_port_id;
    uint8_t is_addr_equal = 0;

    if (dev_input == NULL || port_id == NULL)
        return DOCA_ERROR_INVALID_VALUE;

    *port_id = RTE_MAX_ETHPORTS;

    result = doca_devinfo_get_pci_addr_str(doca_dev_as_devinfo(dev_input), pci_addr_input);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_error_get_descr(result));
#endif
        return result;
    }

    for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; dpdk_port_id++) {
        /* search for the probed devices */
        if (!rte_eth_dev_is_valid_port(dpdk_port_id))
            continue;

        result = doca_dpdk_port_as_dev(dpdk_port_id, &dev_local);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to find DOCA device associated with port ID %d: %s", dpdk_port_id, doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to find DOCA device associated with port ID %d: %s", dpdk_port_id, doca_error_get_descr(result));
#endif
            return result;
        }

        result = doca_devinfo_get_is_pci_addr_equal(doca_dev_as_devinfo(dev_local), pci_addr_input, &is_addr_equal);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to get device PCI address: %s", doca_error_get_descr(result));
#endif
            return result;
        }

        if (is_addr_equal) {
            *port_id = dpdk_port_id;
            break;
        }
    }

    DOCA_LOG_DBG("DPDK port id %d", *port_id);

    if (*port_id == RTE_MAX_ETHPORTS) {
        DOCA_LOG_ERR("No DPDK port matches the DOCA device");
        return DOCA_ERROR_NOT_FOUND;
    }

    return DOCA_SUCCESS;
}
#endif

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

#ifdef DOCA22
    res = doca_devinfo_list_create(&dev_list, &nb_devs);
    if (res != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
        return res;
    }
#else
    res = doca_devinfo_create_list(&dev_list, &nb_devs);
    if (res != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
        return res;
    }
#endif

    /* Search */
    for (i = 0; i < nb_devs; i++) {
#ifdef DOCA22
        res = doca_devinfo_get_is_pci_addr_equal(dev_list[i], pcie_value, &is_addr_equal);
#else
        res = doca_devinfo_is_equal_pci_addr(dev_list[i], pcie_value, &is_addr_equal);
#endif
        if (res == DOCA_SUCCESS && is_addr_equal) {
            /* if device can be opened */
            res = doca_dev_open(dev_list[i], retval);
            if (res == DOCA_SUCCESS) {
#ifdef DOCA22
                doca_devinfo_list_destroy(dev_list);
#else
                doca_devinfo_destroy_list(dev_list);
#endif
                return res;
            }
        }
    }

    DOCA_LOG_ERR("Matching device not found");
    res = DOCA_ERROR_NOT_FOUND;

#ifdef DOCA22
    doca_devinfo_list_destroy(dev_list);
#else
    doca_devinfo_destroy_list(dev_list);
#endif
    return res;
}

doca_error_t
init_doca_device(const char* nic_pcie_addr, struct doca_dev** ddev, uint16_t* dpdk_port_id)
{
    doca_error_t result;
    int ret;

    std::vector<std::string> eal_param = { "", "-a", "00:00.0" };
    std::vector<char *> eal_param_;
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

    ret = rte_eal_init(eal_param_.size()-1, eal_param_.data());
    if (ret < 0) {
        DOCA_LOG_ERR("DPDK init failed: %d", ret);
        return DOCA_ERROR_DRIVER;
    }

    /* Enable DOCA Flow HWS mode */
    result = doca_dpdk_port_probe(*ddev, "dv_flow_en=2");
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Function doca_dpdk_port_probe returned %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Function doca_dpdk_port_probe returned %s", doca_error_get_descr(result));
#endif
        return result;
    }

#ifdef DOCA22
    result = get_dpdk_port_id_doca_dev(*ddev, dpdk_port_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function get_dpdk_port_id_doca_dev returned %s", doca_get_error_string(result));
        return result;
    }
#else
    result = doca_dpdk_get_first_port_id(*ddev, dpdk_port_id);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Function doca_dpdk_get_first_port_id returned %s", doca_error_get_descr(result));
        return result;
    }
#endif

    return DOCA_SUCCESS;
}

struct doca_flow_port*
init_doca_flow(uint16_t port_id, uint8_t rxq_num)
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
            .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
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
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;
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
    rxq_flow_cfg.queues = rxq_num;
    /*
     * HWS: Hardware steering
     * Isolated: don't create RSS rule for DPDK created RX queues
     */
    rxq_flow_cfg.mode_args = "vnf,hws,isolated";
    rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

    result = doca_flow_init(&rxq_flow_cfg);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed to init doca flow with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed to init doca flow with: %s", doca_error_get_descr(result));
#endif
        return NULL;
    }

    /* Start doca flow port */
    port_cfg.port_id = port_id;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str;
    result = doca_flow_port_start(&port_cfg, &df_port);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed to start doca flow port with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed to start doca flow port with: %s", doca_error_get_descr(result));
#endif
        return NULL;
    }

    return df_port;
}

doca_error_t
create_tcp_gpu_pipe(struct rxq_tcp_queues* tcp_queues, struct doca_flow_port* port)
{
    uint16_t flow_queue_id;
    uint16_t rss_queues[MAX_QUEUES];
    doca_error_t result;
    struct doca_flow_pipe_entry* dummy_entry = NULL;
    struct doca_flow_match match_mask = { 0 };

    /* The GPU TCP pipe should only forward known flows to the GPU. Others will be dropped */

    if (tcp_queues == NULL || port == NULL || tcp_queues->numq > MAX_QUEUES)
        return DOCA_ERROR_INVALID_VALUE;

    struct doca_flow_match match = { 0 };
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.ip4.next_proto = IPPROTO_TCP;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;

    // if (connection_based_flows) {
    //     match.outer.ip4.src_ip = 0xffffffff;
    //     match.outer.ip4.dst_ip = 0xffffffff;
    //     match.outer.tcp.l4_port.src_port = 0xffff;
    //     match.outer.tcp.l4_port.dst_port = 0xffff;
    // };

    for (int idx = 0; idx < tcp_queues->numq; idx++) {
        doca_eth_rxq_get_flow_queue_id(tcp_queues->eth_rxq_cpu[idx], &flow_queue_id);
        rss_queues[idx] = flow_queue_id;
    }

    struct doca_flow_fwd fwd = {
        .type = DOCA_FLOW_FWD_RSS,
        .rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP,
        .rss_queues = rss_queues,
        .num_of_queues = tcp_queues->numq,
    };

    struct doca_flow_fwd miss_fwd = {
        .type = DOCA_FLOW_FWD_DROP,
    };

    struct doca_flow_monitor monitor = {
#ifdef DOCA22
        .flags = DOCA_FLOW_MONITOR_COUNT,
#else
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
#endif
    };

#ifdef DOCA22
    struct doca_flow_pipe_cfg pipe_cfg = { 0 };
    pipe_cfg.attr.name = "GPU_RXQ_TCP_PIPE";
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.nb_actions = 0;
    pipe_cfg.attr.is_root = false;
    pipe_cfg.match = &match;
    pipe_cfg.match_mask = &match_mask;
    pipe_cfg.monitor = &monitor;
    pipe_cfg.port = port;
#else
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
#endif

    result = doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &tcp_queues->rxq_pipe_gpu);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
#endif
        return result;
    }

    {
        // if (!connection_based_flows) {
        // For the non-connection-based configuration, create a dummy flow entry which will enable
        // any TCP packets to be forwarded.
        result = doca_flow_pipe_add_entry(0, tcp_queues->rxq_pipe_gpu, NULL, NULL, NULL, NULL, 0, NULL, &dummy_entry);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("RxQ pipe-entry creation failed with: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("RxQ pipe-entry creation failed with: %s", doca_error_get_descr(result));
#endif
            // DOCA_GPUNETIO_VOLATILE(force_quit) = true;
            return result;
        }

        default_flow_timeout_usec = 0;

        result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
#endif
            return result;
        }
    }

    DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

doca_error_t
destroy_tcp_queues(struct rxq_tcp_queues* tcp_queues)
{
    doca_error_t result;

    if (tcp_queues == NULL) {
        DOCA_LOG_ERR("Can't destroy TCP queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    for (int idx = 0; idx < tcp_queues->numq; idx++) {

        DOCA_LOG_INFO("Destroying TCP queue %d", idx);

        if (tcp_queues->sem_cpu[idx]) {
            result = doca_gpu_semaphore_stop(tcp_queues->sem_cpu[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_gpu_semaphore_destroy(tcp_queues->sem_cpu[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_destroy: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed doca_gpu_semaphore_destroy: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }
        }

        if (tcp_queues->eth_rxq_ctx[idx]) {
            result = doca_ctx_stop(tcp_queues->eth_rxq_ctx[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }

#ifdef DOCA22
            result = doca_ctx_dev_rm(tcp_queues->eth_rxq_ctx[idx], tcp_queues->ddev);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed doca_ctx_dev_rm: %s", doca_get_error_string(result));
                return DOCA_ERROR_BAD_STATE;
            }
#endif
        }

        if (tcp_queues->pkt_buff_mmap[idx]) {
            result = doca_mmap_stop(tcp_queues->pkt_buff_mmap[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed to start mmap %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed to start mmap %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }

            result = doca_mmap_destroy(tcp_queues->pkt_buff_mmap[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }
        }

        if (tcp_queues->gpu_pkt_addr[idx]) {
            result = doca_gpu_mem_free(tcp_queues->gpu_dev, tcp_queues->gpu_pkt_addr[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed to free gpu memory: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed to free gpu memory: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }
        }

        if (tcp_queues->eth_rxq_cpu[idx]) {
            result = doca_eth_rxq_destroy(tcp_queues->eth_rxq_cpu[idx]);
            if (result != DOCA_SUCCESS) {
#ifdef DOCA22
                DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_get_error_string(result));
#else
                DOCA_LOG_ERR("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
#endif
                return DOCA_ERROR_BAD_STATE;
            }
        }
    }

    return DOCA_SUCCESS;
}

doca_error_t
destroy_flow_queue(uint16_t port_id, struct doca_flow_port* port_df,
    struct rxq_tcp_queues* tcp_queues)
{
    int ret = 0;

    doca_flow_port_stop(port_df);
    doca_flow_destroy();

    destroy_tcp_queues(tcp_queues);

    ret = rte_eth_dev_stop(port_id);
    if (ret != 0) {
        DOCA_LOG_ERR("Couldn't stop DPDK port %d err %d", port_id, ret);
        return DOCA_ERROR_DRIVER;
    }

    return DOCA_SUCCESS;
}

doca_error_t prepare_tx_buf(struct tx_buf* buf)
{
    uint8_t* cpu_pkt_addr;
    uint8_t* pkt;
    struct eth_ip_tcp_hdr* hdr;
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
        hdr = (struct eth_ip_tcp_hdr*)pkt;

        hdr->l2_hdr.ether_type = rte_cpu_to_be_16(DOCA_ETHER_TYPE_IPV4);

        hdr->l3_hdr.version_ihl = 0x45;
        hdr->l3_hdr.type_of_service = 0x0;
        hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + buf->pkt_nbytes);
        hdr->l3_hdr.packet_id = 0;
        hdr->l3_hdr.fragment_offset = rte_cpu_to_be_16(0x4000); // 0;
        hdr->l3_hdr.time_to_live = 0x40; // 60;
        hdr->l3_hdr.next_proto_id = 6;
        hdr->l3_hdr.hdr_checksum = 0;
        hdr->l3_hdr.src_addr = 0;
        hdr->l3_hdr.dst_addr = 0;

        hdr->l4_hdr.src_port = 0;
        hdr->l4_hdr.dst_port = 0;
        hdr->l4_hdr.sent_seq = 0;
        hdr->l4_hdr.recv_ack = 0;
        /* Assuming no TCP flags needed */
        const uint32_t l4_len = sizeof(struct tcp_hdr);
        hdr->l4_hdr.dt_off = (l4_len << 2) & 0xf0; // 0x50; // 5 << 4;
        /* Assuming no TCP flags needed */
        hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK; //| TCP_FLAG_FIN;
        hdr->l4_hdr.rx_win = rte_cpu_to_be_16(0xFFFF); // BYTE_SWAP16(6000);
        hdr->l4_hdr.cksum = 0;
        hdr->l4_hdr.tcp_urp = 0;

        /* Assuming no TCP flags needed */
        pkt = pkt + sizeof(struct eth_ip_tcp_hdr);

        memcpy(pkt, payload, buf->pkt_nbytes);
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

doca_error_t create_tx_buf(struct tx_buf* buf, struct doca_gpu* gpu_dev, struct doca_dev* ddev, uint32_t num_packets, uint32_t max_pkt_sz)
{
    doca_error_t status;

    if (buf == NULL || gpu_dev == NULL || ddev == NULL || num_packets == 0 || max_pkt_sz == 0) {
        DOCA_LOG_ERR("Invalid input arguments");
        return DOCA_ERROR_INVALID_VALUE;
    }

    buf->gpu_dev = gpu_dev;
    buf->ddev = ddev;
    buf->num_packets = num_packets;
    buf->max_pkt_sz = max_pkt_sz;

#ifdef DOCA22
    status = doca_mmap_create(NULL, &(buf->mmap));
#else
    status = doca_mmap_create(&(buf->mmap));
#endif
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to create doca_buf: failed to create mmap");
        return status;
    }

#ifdef DOCA22
    status = doca_mmap_dev_add(buf->mmap, buf->ddev);
#else
    status = doca_mmap_add_dev(buf->mmap, buf->ddev);
#endif
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to add dev to buf: doca mmap internal error");
        return status;
    }

#ifdef DOCA22
    status = doca_gpu_mem_alloc(buf->gpu_dev, buf->num_packets * buf->max_pkt_sz, 4096, DOCA_GPU_MEM_GPU, (void**)&(buf->gpu_pkt_addr), NULL);
#else
    status = doca_gpu_mem_alloc(buf->gpu_dev, buf->num_packets * buf->max_pkt_sz, 4096, DOCA_GPU_MEM_TYPE_GPU, (void**)(&buf->gpu_pkt_addr), NULL);
#endif
    if ((status != DOCA_SUCCESS) || (buf->gpu_pkt_addr == NULL)) {
        DOCA_LOG_ERR("Unable to alloc txbuf: failed to allocate gpu memory");
        return status;
    }

    status = doca_mmap_set_memrange(buf->mmap, buf->gpu_pkt_addr, (buf->num_packets * buf->max_pkt_sz));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
        return status;
    }

#ifdef DOCA22
    status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_LOCAL_READ_WRITE);
#else
    status = doca_mmap_set_permissions(buf->mmap, DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
#endif
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
        return status;
    }

    status = doca_mmap_start(buf->mmap);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca mmap internal error");
        return status;
    }

    status = doca_buf_arr_create(buf->mmap, &buf->buf_arr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_set_target_gpu(buf->buf_arr, buf->gpu_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_set_params(buf->buf_arr, buf->max_pkt_sz, buf->num_packets, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_start(buf->buf_arr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to start buf: doca buf_arr internal error");
        return status;
    }

    status = doca_buf_arr_get_gpu_handle(buf->buf_arr, &(buf->buf_arr_gpu));
    if (status != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Unable to get buff_arr GPU handle: %s", doca_get_error_string(status));
#else
        DOCA_LOG_ERR("Unable to get buff_arr GPU handle: %s", doca_error_get_descr(status));
#endif
        return status;
    }

    return DOCA_SUCCESS;
}

doca_error_t
create_tcp_queues(struct rxq_tcp_queues* tcp_queues, struct doca_flow_port* df_port, struct doca_gpu* gpu_dev, struct doca_dev* ddev, uint32_t queue_num, uint32_t sem_num)
{
    doca_error_t result;
    uint32_t cyclic_buffer_size = 0;

    if (tcp_queues == NULL || df_port == NULL || gpu_dev == NULL || ddev == NULL || queue_num == 0 || sem_num == 0) {
        DOCA_LOG_ERR("Can't create TCP queues, invalid input");
        return DOCA_ERROR_INVALID_VALUE;
    }

    tcp_queues->ddev = ddev;
    tcp_queues->gpu_dev = gpu_dev;
    tcp_queues->port = df_port;
    tcp_queues->numq = queue_num;
    tcp_queues->numq_cpu_rss = queue_num;
    tcp_queues->nums = sem_num;

    for (int idx = 0; idx < (int)queue_num; idx++) {
        DOCA_LOG_INFO("Creating TCP Eth Rxq %d", idx);

#ifdef DOCA22
        result = doca_eth_rxq_create(&(tcp_queues->eth_rxq_cpu[idx]));
#else
        result = doca_eth_rxq_create(tcp_queues->ddev, MAX_PKT_NUM, MAX_PKT_SIZE, &(tcp_queues->eth_rxq_cpu[idx]));
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_rxq_create: %s", doca_error_get_descr(result));
#endif
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_eth_rxq_set_num_packets(tcp_queues->eth_rxq_cpu[idx], MAX_PKT_NUM);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set eth_rxq num packets: %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_set_max_packet_size(tcp_queues->eth_rxq_cpu[idx], MAX_PKT_SIZE);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set eth_rxq max packet size: %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
#endif

#ifndef DOCA22
        result = doca_eth_rxq_set_type(tcp_queues->eth_rxq_cpu[idx], DOCA_ETH_RXQ_TYPE_CYCLIC);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed doca_eth_rxq_set_type: %s", doca_error_get_descr(result));
            return DOCA_ERROR_BAD_STATE;
        }
#endif

        result = doca_eth_rxq_get_pkt_buffer_size(tcp_queues->eth_rxq_cpu[idx], &cyclic_buffer_size);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to get eth_rxq cyclic buffer size: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
        DOCA_LOG_INFO("cyclic_buffer_size: %d", cyclic_buffer_size);

#ifdef DOCA22
        result = doca_mmap_create(NULL, &tcp_queues->pkt_buff_mmap[idx]);
#else
        result = doca_mmap_create(&tcp_queues->pkt_buff_mmap[idx]);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to create mmap: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to create mmap: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_mmap_dev_add(tcp_queues->pkt_buff_mmap[idx], tcp_queues->ddev);
#else
        result = doca_mmap_add_dev(tcp_queues->pkt_buff_mmap[idx], tcp_queues->ddev);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to add dev to mmap: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_gpu_mem_alloc(tcp_queues->gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_GPU, &tcp_queues->gpu_pkt_addr[idx], NULL);
#else
        result = doca_gpu_mem_alloc(tcp_queues->gpu_dev, cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_TYPE_GPU, &tcp_queues->gpu_pkt_addr[idx], NULL);
#endif
        if (result != DOCA_SUCCESS || tcp_queues->gpu_pkt_addr[idx] == NULL) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to allocate gpu memory %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_mmap_set_memrange(tcp_queues->pkt_buff_mmap[idx], tcp_queues->gpu_pkt_addr[idx], cyclic_buffer_size);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
#else
        /* Map GPU memory buffer used to receive packets with DMABuf */
        result = doca_gpu_dmabuf_fd(tcp_queues->gpu_dev, tcp_queues->gpu_pkt_addr[idx], cyclic_buffer_size, &(tcp_queues->dmabuf_fd[idx]));
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB) with nvidia-peermem mode",
                tcp_queues->gpu_pkt_addr[idx], cyclic_buffer_size);

            /* If failed, use nvidia-peermem method */
            result = doca_mmap_set_memrange(tcp_queues->pkt_buff_mmap[idx], tcp_queues->gpu_pkt_addr[idx], cyclic_buffer_size);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to set memrange for mmap %s", doca_error_get_descr(result));
                destroy_tcp_queues(tcp_queues);
                return DOCA_ERROR_BAD_STATE;
            }
        } else {
            DOCA_LOG_INFO("Mapping receive queue buffer (0x%p size %dB dmabuf fd %d) with dmabuf mode",
                tcp_queues->gpu_pkt_addr[idx], cyclic_buffer_size, tcp_queues->dmabuf_fd[idx]);

            result = doca_mmap_set_dmabuf_memrange(tcp_queues->pkt_buff_mmap[idx], tcp_queues->dmabuf_fd[idx], tcp_queues->gpu_pkt_addr[idx], 0, cyclic_buffer_size);
            if (result != DOCA_SUCCESS) {
                DOCA_LOG_ERR("Failed to set dmabuf memrange for mmap %s", doca_error_get_descr(result));
                destroy_tcp_queues(tcp_queues);
                return DOCA_ERROR_BAD_STATE;
            }
        }
#endif

#ifdef DOCA22
        result = doca_mmap_set_permissions(tcp_queues->pkt_buff_mmap[idx], DOCA_ACCESS_LOCAL_READ_WRITE);
#else
        result = doca_mmap_set_permissions(tcp_queues->pkt_buff_mmap[idx], DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to set permissions for mmap %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to set permissions for mmap %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_mmap_start(tcp_queues->pkt_buff_mmap[idx]);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to start mmap %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to start mmap %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_set_pkt_buffer(tcp_queues->eth_rxq_cpu[idx], tcp_queues->pkt_buff_mmap[idx], 0, cyclic_buffer_size);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to set cyclic buffer  %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to set cyclic buffer  %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        tcp_queues->eth_rxq_ctx[idx] = doca_eth_rxq_as_doca_ctx(tcp_queues->eth_rxq_cpu[idx]);
        if (tcp_queues->eth_rxq_ctx[idx] == NULL) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_rxq_as_doca_ctx: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_ctx_dev_add(tcp_queues->eth_rxq_ctx[idx], tcp_queues->ddev);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Unable to register device with lib context: %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
#endif

        result = doca_ctx_set_datapath_on_gpu(tcp_queues->eth_rxq_ctx[idx], tcp_queues->gpu_dev);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_start(tcp_queues->eth_rxq_ctx[idx]);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_rxq_get_gpu_handle(tcp_queues->eth_rxq_cpu[idx], &(tcp_queues->eth_rxq_gpu[idx]));
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_rxq_get_gpu_handle: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_create(tcp_queues->gpu_dev, &(tcp_queues->sem_cpu[idx]));
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        /*
         * Semaphore memory reside on CPU visibile from GPU.
         * CPU will poll in busy wait on this semaphore (multiple reads)
         * while GPU access each item only once to update values.
         */
#ifdef DOCA22
        result = doca_gpu_semaphore_set_memory_type(tcp_queues->sem_cpu[idx], DOCA_GPU_MEM_CPU_GPU);
#else
        result = doca_gpu_semaphore_set_memory_type(tcp_queues->sem_cpu[idx], DOCA_GPU_MEM_TYPE_GPU);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_set_items_num(tcp_queues->sem_cpu[idx], tcp_queues->nums);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        /*
         * Semaphore memory reside on CPU visibile from GPU.
         * The CPU reads packets info from this structure.
         * The GPU access each item only once to update values.
         */
#ifdef DOCA22
        result = doca_gpu_semaphore_set_custom_info(tcp_queues->sem_cpu[idx], sizeof(struct rx_info), DOCA_GPU_MEM_GPU);
#else
        result = doca_gpu_semaphore_set_custom_info(tcp_queues->sem_cpu[idx], sizeof(struct rx_info), DOCA_GPU_MEM_TYPE_CPU_GPU);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_start(tcp_queues->sem_cpu[idx]);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_gpu_semaphore_get_gpu_handle(tcp_queues->sem_cpu[idx], &(tcp_queues->sem_gpu[idx]));
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        DOCA_LOG_INFO("Creating TCP Eth Txq %d", idx);

#ifdef DOCA22
        result = doca_eth_txq_create(&(tcp_queues->eth_txq_cpu[idx]));
#else
        result = doca_eth_txq_create(tcp_queues->ddev, MAX_SQ_DESCR_NUM,
            &(tcp_queues->eth_txq_cpu[idx]));
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_txq_create: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_eth_txq_set_queue_size(tcp_queues->eth_txq_cpu[idx], MAX_SQ_DESCR_NUM);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Failed to set eth_txq num packets: %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
#endif

#ifdef DOCA22
        result = doca_eth_txq_set_l3_chksum_offload(tcp_queues->eth_txq_cpu[idx]);
#else
        result = doca_eth_txq_set_l3_chksum_offload(tcp_queues->eth_txq_cpu[idx], 1);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to set eth_txq l3 offloads: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to set eth_txq l3 offloads: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_eth_txq_set_l4_chksum_offload(tcp_queues->eth_txq_cpu[idx]);
#else
        result = doca_eth_txq_set_l4_chksum_offload(tcp_queues->eth_txq_cpu[idx], 1);
#endif
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed to set eth_txq l4 offloads: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed to set eth_txq l4 offloads: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        tcp_queues->eth_txq_ctx[idx] = doca_eth_txq_as_doca_ctx(tcp_queues->eth_txq_cpu[idx]);
        if (tcp_queues->eth_txq_ctx[idx] == NULL) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_txq_as_doca_ctx: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

#ifdef DOCA22
        result = doca_ctx_dev_add(tcp_queues->eth_txq_ctx[idx], tcp_queues->ddev);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Unable to register device with lib context: %s", doca_get_error_string(result));
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
#endif

        result = doca_ctx_set_datapath_on_gpu(tcp_queues->eth_txq_ctx[idx], tcp_queues->gpu_dev);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_ctx_set_datapath_on_gpu: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_ctx_start(tcp_queues->eth_txq_ctx[idx]);
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_ctx_start: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }

        result = doca_eth_txq_get_gpu_handle(tcp_queues->eth_txq_cpu[idx], &(tcp_queues->eth_txq_gpu[idx]));
        if (result != DOCA_SUCCESS) {
#ifdef DOCA22
            DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_get_error_string(result));
#else
            DOCA_LOG_ERR("Failed doca_eth_txq_get_gpu_handle: %s", doca_error_get_descr(result));
#endif
            destroy_tcp_queues(tcp_queues);
            return DOCA_ERROR_BAD_STATE;
        }
    }

    result = create_tx_buf(&tcp_queues->tx_buf_arr, tcp_queues->gpu_dev, tcp_queues->ddev, TX_BUF_NUM, TX_BUF_MAX_SZ);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed create buf_page_index: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed create buf_page_index: %s", doca_error_get_descr(result));
#endif
        destroy_tcp_queues(tcp_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    result = prepare_tx_buf(&tcp_queues->tx_buf_arr);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed prepare buf_page_index: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed prepare buf_page_index: %s", doca_error_get_descr(result));
#endif
        destroy_tcp_queues(tcp_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    /* Create UDP based flow pipe */
    result = create_tcp_gpu_pipe(tcp_queues, df_port);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Function create_tcp_gpu_pipe returned %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Function create_tcp_gpu_pipe returned %s", doca_error_get_descr(result));
#endif
        destroy_tcp_queues(tcp_queues);
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t create_sem(struct doca_gpu* gpu_dev, struct sem_pair* sem, uint16_t sem_num)
{
    sem->nums = sem_num;

    doca_error_t result;

    result = doca_gpu_semaphore_create(gpu_dev, &(sem->sem_cpu));
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_create: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    /*
     * Semaphore memory reside on CPU visibile from GPU.
     * CPU will poll in busy wait on this semaphore (multiple reads)
     * while GPU access each item only once to update values.
     */
#ifdef DOCA22
    result = doca_gpu_semaphore_set_memory_type(sem->sem_cpu, DOCA_GPU_MEM_CPU_GPU);
#else
    result = doca_gpu_semaphore_set_memory_type(sem->sem_cpu, DOCA_GPU_MEM_TYPE_GPU);
#endif
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_memory_type: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_set_items_num(sem->sem_cpu, sem->nums);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_items_num: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    /*
     * Semaphore memory reside on CPU visibile from GPU.
     * The CPU reads packets info from this structure.
     * The GPU access each item only once to update values.
     */
#ifdef DOCA22
    result = doca_gpu_semaphore_set_custom_info(sem->sem_cpu, sizeof(struct rx_info), DOCA_GPU_MEM_GPU);
#else
    result = doca_gpu_semaphore_set_custom_info(sem->sem_cpu, sizeof(struct stats_tcp), DOCA_GPU_MEM_TYPE_CPU_GPU);
#endif
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_set_custom_info: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_start(sem->sem_cpu);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    result = doca_gpu_semaphore_get_gpu_handle(sem->sem_cpu, &(sem->sem_gpu));
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Failed doca_gpu_semaphore_get_gpu_handle: %s", doca_error_get_descr(result));
#endif
        return DOCA_ERROR_BAD_STATE;
    }

    return result;
}

doca_error_t
create_root_pipe(struct rxq_tcp_queues* tcp_queues, struct doca_flow_port* port)
{
    uint32_t priority_high = 1;
    uint32_t priority_low = 3;
    doca_error_t result;
    struct doca_flow_match match_mask = { 0 };
    struct doca_flow_monitor monitor = {
#ifdef DOCA22
        .flags = DOCA_FLOW_MONITOR_COUNT,
#else
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
#endif
    };

    if (tcp_queues == NULL)
        return DOCA_ERROR_INVALID_VALUE;

#ifdef DOCA22
    struct doca_flow_pipe_cfg pipe_cfg = { 0 };
    pipe_cfg.attr.name = "ROOT_PIPE";
    pipe_cfg.attr.is_root = true;
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
    pipe_cfg.port = port;
    pipe_cfg.monitor = &monitor;
    pipe_cfg.match_mask = &match_mask;
#else
    struct doca_flow_pipe_cfg pipe_cfg = { 0 };
    pipe_cfg.attr.name = "ROOT_PIPE";
    pipe_cfg.attr.enable_strict_matching = true;
    pipe_cfg.attr.is_root = true;
    pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
    pipe_cfg.port = port;
    pipe_cfg.monitor = &monitor;
    pipe_cfg.match_mask = &match_mask;
#endif

    result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, &tcp_queues->root_pipe);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Root pipe creation failed with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Root pipe creation failed with: %s", doca_error_get_descr(result));
#endif
        return result;
    }

    struct doca_flow_match tcp_match_gpu = { 0 };
    tcp_match_gpu.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    tcp_match_gpu.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;

    struct doca_flow_fwd tcp_fwd_gpu = {
        .type = DOCA_FLOW_FWD_PIPE,
        .next_pipe = tcp_queues->rxq_pipe_gpu,
    };

    result = doca_flow_pipe_control_add_entry(0, 0, tcp_queues->root_pipe, &tcp_match_gpu, NULL, NULL, NULL, NULL, NULL,
        &tcp_fwd_gpu, NULL, &tcp_queues->root_tcp_entry_gpu);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Root pipe UDP entry creation failed with: %s", doca_error_get_descr(result));
#endif
        return result;
    }

    result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
#ifdef DOCA22
        DOCA_LOG_ERR("Root pipe entry process failed with: %s", doca_get_error_string(result));
#else
        DOCA_LOG_ERR("Root pipe entry process failed with: %s", doca_error_get_descr(result));
#endif
        return result;
    }

    DOCA_LOG_DBG("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

std::string doca_error_get_descr(doca_error_t err)
{
#ifdef DOCA22
    return doca_get_error_string(err);
#else
    return doca_error_get_descr(err);
#endif
}

} // lng
