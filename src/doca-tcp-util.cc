#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "doca-common-util-internal.h"
#include "lng/doca-util.h"

#include "log.h"

DOCA_LOG_REGISTER(DOCA_TCP_UTIL);

namespace lng {

static uint64_t default_flow_timeout_usec;

struct doca_flow_port* init_doca_tcp_flow(uint16_t port_id, uint8_t rxq_num)
{
    return init_doca_flow(port_id, rxq_num, RTE_ETH_TX_OFFLOAD_TCP_CKSUM);
}

doca_error_t
create_tcp_pipe(struct doca_flow_pipe** pipe, struct rx_queue* rxq, struct doca_flow_port* port, int numq)
{
    uint16_t flow_queue_id;
    uint16_t rss_queues[MAX_QUEUES];
    doca_error_t result;
    struct doca_flow_pipe_entry* dummy_entry = NULL;
    struct doca_flow_match match_mask = { 0 };

    /* The GPU TCP pipe should only forward known flows to the GPU. Others will be dropped */

    if (pipe == NULL || rxq == NULL || port == NULL || numq > MAX_QUEUES)
        return DOCA_ERROR_INVALID_VALUE;

    struct doca_flow_match match = { 0 };
    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.ip4.next_proto = IPPROTO_TCP;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;

    for (int idx = 0; idx < numq; idx++) {
        doca_eth_rxq_get_flow_queue_id(rxq[idx].eth_rxq_cpu, &flow_queue_id);
        rss_queues[idx] = flow_queue_id;
    }

    struct doca_flow_fwd fwd = {
        .type = DOCA_FLOW_FWD_RSS,
        .rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP,
        .rss_queues = rss_queues,
        .num_of_queues = numq,
    };

    struct doca_flow_fwd miss_fwd = {
        .type = DOCA_FLOW_FWD_DROP,
    };

    struct doca_flow_monitor monitor = {
        .counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED,
    };

    struct doca_flow_pipe_cfg* pipe_cfg;

    result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_name(pipe_cfg, "GPU_RXQ_TCP_PIPE");
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg name: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_enable_strict_matching(pipe_cfg, true);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg enable_strict_matching: %s",
            doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_type(pipe_cfg, DOCA_FLOW_PIPE_BASIC);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg type: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_is_root(pipe_cfg, false);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg is_root: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, &match_mask);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
        return result;
    }
    result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
        return result;
    }

    result = doca_flow_pipe_create(pipe_cfg, &fwd, &miss_fwd, pipe);
    if (result != DOCA_SUCCESS) {
        log::error("RxQ pipe creation failed with: %s", doca_error_get_descr(result));
        return result;
    }

    // any TCP packets to be forwarded.
    result = doca_flow_pipe_add_entry(0, *pipe, NULL, NULL, NULL, NULL, 0, NULL, &dummy_entry);
    if (result != DOCA_SUCCESS) {
        log::error("RxQ pipe-entry creation failed with: %s", doca_error_get_descr(result));
        // DOCA_GPUNETIO_VOLATILE(force_quit) = true;
        return result;
    }

    default_flow_timeout_usec = 0;

    result = doca_flow_entries_process(port, 0, default_flow_timeout_usec, 0);
    if (result != DOCA_SUCCESS) {
        log::error("RxQ pipe entry process failed with: %s", doca_error_get_descr(result));
        return result;
    }

    // log::debug("Created Pipe %s", pipe_cfg.attr.name);

    return DOCA_SUCCESS;
}

// doca_error_t
// destroy_tcp_queues(struct rxq_tcp_queues* tcp_queues)
// {
//     doca_error_t result;

//     if (tcp_queues == NULL) {
//         log::error("Can't destroy TCP queues, invalid input");
//         return DOCA_ERROR_INVALID_VALUE;
//     }

//     for (int idx = 0; idx < tcp_queues->numq; idx++) {

//         DOCA_LOG_INFO("Destroying TCP queue %d", idx);

//         if (tcp_queues->sem_cpu[idx]) {
//             result = doca_gpu_semaphore_stop(tcp_queues->sem_cpu[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed doca_gpu_semaphore_start: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }

//             result = doca_gpu_semaphore_destroy(tcp_queues->sem_cpu[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed doca_gpu_semaphore_destroy: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }
//         }

//         if (tcp_queues->eth_rxq_ctx[idx]) {
//             result = doca_ctx_stop(tcp_queues->eth_rxq_ctx[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed doca_ctx_stop: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }
//         }

//         if (tcp_queues->pkt_buff_mmap[idx]) {
//             result = doca_mmap_stop(tcp_queues->pkt_buff_mmap[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed to start mmap %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }

//             result = doca_mmap_destroy(tcp_queues->pkt_buff_mmap[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed to destroy mmap: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }
//         }

//         if (tcp_queues->gpu_pkt_addr[idx]) {
//             result = doca_gpu_mem_free(tcp_queues->gpu_dev, tcp_queues->gpu_pkt_addr[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed to free gpu memory: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }
//         }

//         if (tcp_queues->eth_rxq_cpu[idx]) {
//             result = doca_eth_rxq_destroy(tcp_queues->eth_rxq_cpu[idx]);
//             if (result != DOCA_SUCCESS) {
//                 log::error("Failed doca_eth_rxq_destroy: %s", doca_error_get_descr(result));
//                 return DOCA_ERROR_BAD_STATE;
//             }
//         }
//     }

//     return DOCA_SUCCESS;
// }

// doca_error_t
// destroy_tcp_flow_queue(uint16_t port_id, struct doca_flow_port* port_df,
//     struct rxq_tcp_queues* tcp_queues)
// {
//     int ret = 0;

//     doca_flow_port_stop(port_df);
//     doca_flow_destroy();

//     destroy_tcp_queues(tcp_queues);

//     ret = rte_eth_dev_stop(port_id);
//     if (ret != 0) {
//         log::error("Couldn't stop DPDK port %d err %d", port_id, ret);
//         return DOCA_ERROR_DRIVER;
//     }

//     return DOCA_SUCCESS;
// }

// to be deleted
doca_error_t prepare_tcp_tx_buf(struct tx_buf* buf)
{
    uint8_t* cpu_pkt_addr;
    uint8_t* pkt;
    struct eth_ip_tcp_hdr* hdr;
    const char* payload = "";
    cudaError_t res_cuda;

    buf->pkt_nbytes = strlen(payload);

    cpu_pkt_addr = (uint8_t*)calloc(buf->num_packets * buf->max_pkt_sz, sizeof(uint8_t));
    if (cpu_pkt_addr == NULL) {
        log::error("Error in txbuf preparation, failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    for (int idx = 0; idx < buf->num_packets; idx++) {
        pkt = cpu_pkt_addr + (idx * buf->max_pkt_sz);
        hdr = (struct eth_ip_tcp_hdr*)pkt;

        hdr->l2_hdr.ether_type = rte_cpu_to_be_16(DOCA_FLOW_ETHER_TYPE_IPV4);

        hdr->l3_hdr.version_ihl = 0x45;
        hdr->l3_hdr.type_of_service = 0x0;
        hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + buf->pkt_nbytes);
        hdr->l3_hdr.packet_id = 0;
        hdr->l3_hdr.fragment_offset = rte_cpu_to_be_16(0x4000); // 0;
        hdr->l3_hdr.time_to_live = 0x40; // 60;
        hdr->l3_hdr.next_proto_id = IPPROTO_TCP;
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
        log::error("Function CUDA Memcpy cqe_addr failed with %s", cudaGetErrorString(res_cuda));
        return DOCA_ERROR_DRIVER;
    }

    return DOCA_SUCCESS;
}

doca_error_t
create_tcp_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_entry, struct doca_flow_pipe** rxq_pipe, uint16_t* dst_ports, int rxq_num, struct doca_flow_port* port)
{
    return create_root_pipe(root_pipe, root_entry, rxq_pipe, dst_ports, rxq_num, port, DOCA_FLOW_L4_TYPE_EXT_TCP);
}

} // lng
