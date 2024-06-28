#ifndef NET_HEADER_H
#define NET_HEADER_H

#include <stdint.h>

#define ETHER_ADDR_LEN 6
#define MAX_PKT_NUM (65536 * 4)
#define GPU_PAGE_SIZE (1UL << 16)

#define BYTE_SWAP16(v) \
    ((((uint16_t)(v)&UINT16_C(0x00ff)) << 8) | (((uint16_t)(v)&UINT16_C(0xff00)) >> 8))

#define BYTE_SWAP32(x) \
    ((((x)&0xff000000) >> 24) | (((x)&0x00ff0000) >> 8) | (((x)&0x0000ff00) << 8) | (((x)&0x000000ff) << 24))

namespace lng {

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

}

#endif
