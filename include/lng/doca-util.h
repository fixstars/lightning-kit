#ifndef LNG_DOCA_UTIL_H
#define LNG_DOCA_UTIL_H

#define GPU_PAGE_SIZE (1UL << 16)
#define MAX_QUEUES 1
#define MAX_PORT_STR_LEN 128 /* Maximal length of port name */
#define MAX_PKT_NUM 65536
#define MAX_PKT_SIZE 8192
#define MAX_RX_NUM_PKTS 2048
#define MAX_RX_TIMEOUT_NS 10000 /* 10us */ // 1000000 /* 1ms */
#define SEMAPHORES_PER_QUEUE 1024
#define CUDA_THREADS 512
#define ETHER_ADDR_LEN 6
#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define DPDK_DEFAULT_PORT 0
#define WARP_SIZE 32
#define WARP_FULL_MASK 0xFFFFFFFF
#define MAX_SQ_DESCR_NUM 4096
#define TX_BUF_NUM 1024 /* 32 x 32 */
#define TX_BUF_MAX_SZ MAX_PKT_SIZE // 512
#define MINIMUM_TARBUF_SIZE (50 * 1024 * 1024)

#include <cstdint>
#include <string>

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

enum tcp_flags {
    TCP_FLAG_FIN = (1 << 0),
    /* set tcp packet with Fin flag */
    TCP_FLAG_SYN = (1 << 1),
    /* set tcp packet with Syn flag */
    TCP_FLAG_RST = (1 << 2),
    /* set tcp packet with Rst flag */
    TCP_FLAG_PSH = (1 << 3),
    /* set tcp packet with Psh flag */
    TCP_FLAG_ACK = (1 << 4),
    /* set tcp packet with Ack flag */
    TCP_FLAG_URG = (1 << 5),
    /* set tcp packet with Urg flag */
    TCP_FLAG_ECE = (1 << 6),
    /* set tcp packet with ECE flag */
    TCP_FLAG_CWR = (1 << 7),
    /* set tcp packet with CQE flag */
};

#define BYTE_SWAP16(v) \
    ((((uint16_t)(v)&UINT16_C(0x00ff)) << 8) | (((uint16_t)(v)&UINT16_C(0xff00)) >> 8))

#define BYTE_SWAP32(x) \
    ((((x)&0xff000000) >> 24) | (((x)&0x00ff0000) >> 8) | (((x)&0x0000ff00) << 8) | (((x)&0x000000ff) << 24))

struct ether_hdr {
    uint8_t d_addr_bytes[ETHER_ADDR_LEN]; /* Destination addr bytes in tx order */
    uint8_t s_addr_bytes[ETHER_ADDR_LEN]; /* Source addr bytes in tx order */
    uint16_t ether_type; /* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr {
    uint8_t version_ihl; /* version and header length */
    uint8_t type_of_service; /* type of service */
    uint16_t total_length; /* length of packet */
    uint16_t packet_id; /* packet ID */
    uint16_t fragment_offset; /* fragmentation offset */
    uint8_t time_to_live; /* time to live */
    uint8_t next_proto_id; /* protocol ID */
    uint16_t hdr_checksum; /* header checksum */
    uint32_t src_addr; /* source address */
    uint32_t dst_addr; /* destination address */
} __attribute__((__packed__));

struct tcp_hdr {
    uint16_t src_port; /* TCP source port */
    uint16_t dst_port; /* TCP destination port */
    uint32_t sent_seq; /* TX data sequence number */
    uint32_t recv_ack; /* RX data acknowledgment sequence number */
    uint8_t dt_off; /* Data offset */
    uint8_t tcp_flags; /* TCP flags */
    uint16_t rx_win; /* RX flow control window */
    uint16_t cksum; /* TCP checksum */
    uint16_t tcp_urp; /* TCP urgent pointer, if any */
} __attribute__((__packed__));

struct eth_ip_tcp_hdr {
    struct ether_hdr l2_hdr; /* Ethernet header */
    struct ipv4_hdr l3_hdr; /* IP header */
    struct tcp_hdr l4_hdr; /* TCP header */
} __attribute__((__packed__));

struct tx_buf {
    struct doca_gpu* gpu_dev; /* GPU device */
    struct doca_dev* ddev; /* Network DOCA device */
    uint32_t num_packets; /* Number of packets in the buffer */
    uint32_t max_pkt_sz; /* Max size of each packet in the buffer */
    uint32_t pkt_nbytes; /* Effective bytes in each packet */
    uint8_t* gpu_pkt_addr; /* GPU memory address of the buffer */
    int dmabuf_fd;
    struct doca_mmap* mmap; /* DOCA mmap around GPU memory buffer for the DOCA device */
    struct doca_buf_arr* buf_arr; /* DOCA buffer array object around GPU memory buffer */
    struct doca_gpu_buf_arr* buf_arr_gpu; /* DOCA buffer array GPU handle */
};

struct semaphore {
    struct doca_gpu_semaphore* sem_cpu;
    struct doca_gpu_semaphore_gpu* sem_gpu;
    int sem_num;
};

struct rx_queue {
    struct doca_ctx* eth_rxq_ctx;
    struct doca_eth_rxq* eth_rxq_cpu;
    struct doca_gpu_eth_rxq* eth_rxq_gpu;
    struct doca_mmap* pkt_buff_mmap;
    void* gpu_pkt_addr;
    int dmabuf_fd;
    struct doca_gpu* gpu_dev;
};

struct tx_queue {
    struct doca_ctx* eth_txq_ctx;
    struct doca_eth_txq* eth_txq_cpu;
    struct doca_gpu_eth_txq* eth_txq_gpu;
};

struct udp_hdr {
    uint16_t src_port; /* UDP source port */
    uint16_t dst_port; /* UDP destination port */
    uint16_t dgram_len; /* UDP datagram length */
    uint16_t dgram_cksum; /* UDP datagram checksum */
} __attribute__((__packed__));

struct eth_ip_udp_hdr {
    struct ether_hdr l2_hdr; /* Ethernet header */
    struct ipv4_hdr l3_hdr; /* IP header */
    struct udp_hdr l4_hdr; /* UDP header */
} __attribute__((__packed__));

// to be deleted
struct sem_pair {
    uint16_t nums; /* Number of semaphores items */
    struct doca_gpu_semaphore* sem_cpu; /* One semaphore per queue to report stats, CPU handler*/
    struct doca_gpu_semaphore_gpu* sem_gpu; /* One semaphore per queue to report stats, GPU handler*/
};

// to be deleted
struct stats_tcp {
    uint32_t tcp_syn; /* TCP with SYN flag */
    uint32_t tcp_fin; /* TCP with FIN flag */
    uint32_t tcp_ack; /* TCP with ACK flag */
    uint32_t others; /* Other TCP packets */
    uint32_t total; /* Total TCP packets */
};

struct rx_info {
    uint32_t rx_pkt_num;
    uint64_t rx_buf_idx;
    uint32_t cur_ackn;
};

struct fr_info {
    uint8_t* eth_payload;
};

struct tcp_frame_info {
    uint8_t* body;
};

struct reply_info {
    uint8_t* eth_payload;
};

// to be deleted
struct store_buf_info {
    uint8_t* buf;
    uint64_t size;
};

// to be deleted
struct ready_buf_info {
    uint64_t is_ready;
};

#define FRAME_SIZE (128 * 1024 * 1024)

// to be deleted
struct Frame {
    uint8_t data[FRAME_SIZE];
};

doca_error_t
init_doca_device(const char* nic_pcie_addr, struct doca_dev** ddev, uint16_t* dpdk_port_id);

struct doca_flow_port*
init_doca_tcp_flow(uint16_t port_id, uint8_t rxq_num);

struct doca_flow_port*
init_doca_udp_flow(uint16_t port_id, uint8_t rxq_num);

doca_error_t
create_tcp_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_udp_entry, struct doca_flow_pipe* rxq_pipe, struct doca_flow_port* port);

doca_error_t
create_udp_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_udp_entry, struct doca_flow_pipe* rxq_pipe, struct doca_flow_port* port);

doca_error_t create_rx_queue(struct rx_queue* rxq, struct doca_gpu* gpu_dev, struct doca_dev* ddev);
doca_error_t create_tx_queue(struct tx_queue* txq, struct doca_gpu* gpu_dev, struct doca_dev* ddev);
doca_error_t create_tx_buf(struct tx_buf* buf, struct doca_gpu* gpu_dev, struct doca_dev* ddev, uint32_t num_packets, uint32_t max_pkt_sz);
doca_error_t prepare_udp_tx_buf(struct tx_buf* buf);
doca_error_t prepare_tcp_tx_buf(struct tx_buf* buf);
doca_error_t create_semaphore(semaphore* sem, struct doca_gpu* gpu_dev, uint32_t sem_num, int element_size, enum doca_gpu_mem_type mem_type);
doca_error_t create_tcp_pipe(struct doca_flow_pipe** pipe, struct rx_queue* rxq, struct doca_flow_port* port, int numq);
doca_error_t create_udp_pipe(struct doca_flow_pipe** pipe, struct rx_queue* rxq, struct doca_flow_port* port, int numq);

// doca_error_t
// destroy_tcp_flow_queue(uint16_t port_id, struct doca_flow_port* port_df,
//     struct rxq_tcp_queues* tcp_queues);

// doca_error_t
// destroy_udp_flow_queue(uint16_t port_id, struct doca_flow_port* port_df,
//     struct rxq_udp_queues* udp_queues);

// extern "C" {

// // to be deleted
// doca_error_t kernel_receive_tcp(struct rxq_tcp_queues* tcp_queues,
//     uint8_t* cpu_tar_buf, uint64_t size, uint64_t pitch, struct sem_pair* sem_frame);
// }

} // lng

#endif // LNG_DOCA_UTIL_H
