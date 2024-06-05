#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <regex>
#include <vector>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_net.h>

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#define FMT_HEADER_ONLY
#include "argparse/argparse.hpp"
#include "fmt/core.h"

#define RX_RING_SIZE 1028
#define TX_RING_SIZE 1028
#define MTU 8000
#define NUM_MBUFS (4096 * 24 - 1) // 98303
#define MBUF_CACHE_SIZE 250
#define PKTMBUF_CELLSIZE 10240

namespace fs = std::filesystem;

bool g_running = true;
void catch_int(int sig_num)
{
    signal(SIGINT, catch_int);
    fmt::print("Interrupted");
    g_running = false;
}

std::vector<uint8_t> parse_ipv4_addr(const std::string& addr)
{
    const char* addr_ptr = addr.c_str();
    std::vector<uint8_t> vs { 0, 0, 0, 0 };
    size_t index = 0;

    while (*addr_ptr) {
        if (isdigit(static_cast<unsigned char>(*addr_ptr))) {
            vs[index] *= 10;
            vs[index] += *addr_ptr - '0';
        } else {
            index++;
        }
        addr_ptr++;
    }

    return vs;
}

rte_be32_t convert_ipv4_addr(const std::vector<uint8_t>& vs)
{
    return rte_cpu_to_be_32(RTE_IPV4(vs[0], vs[1], vs[2], vs[3]));
}

class TCPPacketMaker {

public:
    TCPPacketMaker(rte_mempool* mbuf_pool,
        rte_eth_dev_info dev_info,
        const std::string& src_ether_addr,
        const std::string& dst_ether_addr,
        const std::string& src_addr,
        const std::string& dst_addr,
        uint16_t src_port,
        uint16_t dst_port)
        : mbuf_pool_(mbuf_pool)
        , dev_info_(dev_info)
        , src_port_(rte_cpu_to_be_16(src_port))
        , dst_port_(rte_cpu_to_be_16(dst_port))
    {
        rte_ether_unformat_addr(src_ether_addr.c_str(), &src_ether_addr_);
        rte_ether_unformat_addr(dst_ether_addr.c_str(), &dst_ether_addr_);

        src_addr_ = convert_ipv4_addr(parse_ipv4_addr(src_addr));
        dst_addr_ = convert_ipv4_addr(parse_ipv4_addr(dst_addr));
    }

    std::vector<struct rte_mbuf*> build(uint32_t tcp_seqn, uint32_t tcp_ackn, uint8_t tcp_flags, bool notify_options, void* payload_data, size_t payload_size)
    {
        const uint16_t mtu = MTU;
        const uint32_t l2_len = sizeof(struct rte_ether_hdr);
        const uint32_t l3_len = sizeof(struct rte_ipv4_hdr);
        const size_t tcp_option_size = notify_options ? 12 : 0;
        const uint32_t l4_len = sizeof(rte_tcp_hdr) + tcp_option_size;
        const uint32_t hdr_len = l2_len + l3_len + l4_len;
        const uint16_t mss = mtu - l4_len - l3_len;

        std::vector<struct rte_mbuf*> bufs;
        uint8_t* payload_ptr = reinterpret_cast<uint8_t*>(payload_data);
        int32_t remaining = payload_size;

        do {
            auto buf = rte_pktmbuf_alloc(mbuf_pool_);
            if (buf == nullptr) {
                throw std::runtime_error("rte_pktmbuf_alloc error\n");
            }
            buf->packet_type = RTE_PTYPE_L2_ETHER | (notify_options ? RTE_PTYPE_L3_IPV4_EXT : RTE_PTYPE_L3_IPV4) | RTE_PTYPE_L4_TCP;

            uint8_t* head = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(buf, hdr_len));

            // L2
            auto ether = reinterpret_cast<struct rte_ether_hdr*>(head);
            rte_ether_addr_copy(&dst_ether_addr_, &ether->dst_addr);
            rte_ether_addr_copy(&src_ether_addr_, &ether->src_addr);
            ether->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

            // L3
            auto* ipv4 = reinterpret_cast<struct rte_ipv4_hdr*>(head + l2_len);
            ipv4->version_ihl = RTE_IPV4_VHL_DEF;
            ipv4->packet_id = rte_cpu_to_be_16(static_cast<uint16_t>(rand()));
            ipv4->fragment_offset = rte_cpu_to_be_16(0x4000);
            ipv4->time_to_live = 0x40;
            ipv4->next_proto_id = IPPROTO_TCP;
            ipv4->src_addr = src_addr_;
            ipv4->dst_addr = dst_addr_;
            ipv4->hdr_checksum = 0;

            // L4
            auto* tcp = reinterpret_cast<struct rte_tcp_hdr*>(head + l2_len + l3_len);
            tcp->src_port = src_port_;
            tcp->dst_port = dst_port_;
            tcp->sent_seq = rte_cpu_to_be_32(tcp_seqn);
            tcp->recv_ack = rte_cpu_to_be_32(tcp_ackn);
            tcp->data_off = (l4_len << 2) & 0xf0;
            tcp->tcp_flags = tcp_flags;
            tcp->rx_win = rte_cpu_to_be_16(0xFFFF); // window size
            if (notify_options) {
                reinterpret_cast<uint8_t*>(tcp)[20] = 0x02; // Option Kind is MSS(0x02)
                reinterpret_cast<uint8_t*>(tcp)[21] = 0x04; // Option Length is 4 byte
                rte_be16_t v = rte_cpu_to_be_16(mss);
                memcpy(reinterpret_cast<uint8_t*>(tcp) + 22, &v, sizeof(v));

                reinterpret_cast<uint8_t*>(tcp)[24] = 0x01; // Option Kind is NOP
                reinterpret_cast<uint8_t*>(tcp)[25] = 0x01; // Option Kind is NOP
                reinterpret_cast<uint8_t*>(tcp)[26] = 0x04; // Option Kind is Sack Permitted(0x04)
                reinterpret_cast<uint8_t*>(tcp)[27] = 0x02; // Option Length is 2 byte

                reinterpret_cast<uint8_t*>(tcp)[28] = 0x01; // Option Kind is NOP
                reinterpret_cast<uint8_t*>(tcp)[29] = 0x03; // Option Kind is Window Scale (0x03)
                reinterpret_cast<uint8_t*>(tcp)[30] = 0x03; // Option Length is 3 byte
                uint8_t window_scale = 0x07;
                memcpy(reinterpret_cast<uint8_t*>(tcp) + 31, &window_scale, sizeof(window_scale));
            }
            tcp->cksum = 0;

            size_t segmented_payload_size = std::min(static_cast<size_t>(remaining), static_cast<size_t>(rte_pktmbuf_tailroom(buf)));
            uint8_t* body = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(buf, segmented_payload_size));
            memcpy(body, payload_ptr, segmented_payload_size);

            remaining -= segmented_payload_size;
            payload_ptr += segmented_payload_size;
            tcp_seqn += segmented_payload_size;

            buf->ol_flags = RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;
            buf->l2_len = l2_len;
            buf->l3_len = l3_len;
            buf->l4_len = l4_len;

            if (remaining) {
                buf->ol_flags |= RTE_MBUF_F_TX_TCP_SEG;
                buf->tso_segsz = mss;
            }

            while (remaining && buf->nb_segs < dev_info_.tx_desc_lim.nb_seg_max) {
                auto b = rte_pktmbuf_alloc(mbuf_pool_);
                if (b == nullptr) {
                    throw std::runtime_error("rte_pktmbuf_alloc seg");
                }

                size_t segmented_payload_size = std::min(remaining, static_cast<int32_t>(b->buf_len));
                memcpy(b->buf_addr, payload_ptr, segmented_payload_size);

                remaining -= segmented_payload_size;
                payload_ptr += segmented_payload_size;
                tcp_seqn += segmented_payload_size;

                b->data_len = segmented_payload_size;
                b->pkt_len = b->data_len;
                b->data_off = 0;

                if (rte_pktmbuf_chain(buf, b)) {
                    throw std::runtime_error("rte_pktmbuf_chain error\n");
                }
            }

            ipv4->total_length = rte_cpu_to_be_16(buf->pkt_len - l2_len);
            ipv4->hdr_checksum = 0;
            tcp->cksum = 0;

            bufs.push_back(buf);
        } while (remaining);

        return bufs;
    }

private:
    rte_mempool* mbuf_pool_;
    rte_eth_dev_info dev_info_;

    rte_ether_addr src_ether_addr_;
    rte_ether_addr dst_ether_addr_;

    rte_be32_t src_addr_;
    rte_be32_t dst_addr_;

    rte_be16_t src_port_;
    rte_be16_t dst_port_;
};

struct client_args {
    int lcore_id;
    struct rte_mempool* mbuf_pool;
    uint16_t dev_port_id;
    struct rte_eth_dev_info dev_info;
    std::string client_eth_addr;
    std::string server_eth_addr;
    std::string client_ip_addr;
    std::string server_ip_addr;
    uint16_t client_port;
    uint16_t server_port;
    uint32_t bandwidth_in_gbps;
    uint16_t nb_txd;
    uint32_t seqn;
    uint32_t ackn;
    std::unique_ptr<TCPPacketMaker> tcp_pkt_maker;
    std::vector<uint8_t> send_buf;
    uint32_t send_buf_num;
    uint32_t chunk_size;
    uint32_t check_ack_freq;
    bool ignore_ack;
};

int init_rte_env(void* arg1)
{
    struct client_args* arg = (struct client_args*)arg1;
    const unsigned nb_ports = 1;

    uint16_t dev_port_id = arg->dev_port_id;

    fmt::print("dev_port_id({}) works on {}th core on {}th socket\n", dev_port_id, rte_lcore_id(), rte_socket_id());

    std::string pool_name = "CLIENT_MBUF_POOL " + std::to_string(dev_port_id);
    arg->mbuf_pool = rte_pktmbuf_pool_create(pool_name.c_str(), NUM_MBUFS * nb_ports,
        MBUF_CACHE_SIZE, 0, PKTMBUF_CELLSIZE + RTE_PKTMBUF_HEADROOM, rte_socket_id());
    struct rte_mempool* mbuf_pool = arg->mbuf_pool;
    if (mbuf_pool == NULL) {
        throw std::runtime_error("Failed to build mbuf_pool\n");
    }

    /* Initializing all ports. 8< */
    const uint16_t rx_rings = 1, tx_rings = 1;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    struct rte_eth_dev_info& dev_info = arg->dev_info;
    struct rte_eth_rxconf rxconf;
    struct rte_eth_txconf txconf;

    if (!rte_eth_dev_is_valid_port(dev_port_id)) {
        throw std::runtime_error(fmt::format("dev_port_id({}) is not valid port\n", dev_port_id));
    }

    struct rte_eth_conf port_conf;
    memset(&port_conf, 0, sizeof(struct rte_eth_conf));

    int ret = rte_eth_dev_info_get(dev_port_id, &dev_info);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Error during getting device (port {}) info: {}\n", dev_port_id, strerror(-ret)));
    }

    port_conf.link_speeds = RTE_ETH_LINK_SPEED_FIXED | RTE_ETH_LINK_SPEED_100G;
    port_conf.rxmode.mtu = MTU;
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TCP_LRO) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;
        port_conf.rxmode.max_lro_pkt_size = dev_info.max_lro_pkt_size;
    }
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_SCATTER) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_SCATTER;
    }

    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
    }
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MULTI_SEGS) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MULTI_SEGS;
    }
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_IPV4_CKSUM) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM;
    }
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_TCP_CKSUM) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_CKSUM;
    }
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_TCP_TSO) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_TSO;
    }

    /* Configure the Ethernet device. */
    ret = rte_eth_dev_configure(dev_port_id, rx_rings, tx_rings, &port_conf);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("rte_eth_dev_configure (port {}) {}\n", dev_port_id, strerror(-ret)));
    }

    ret = rte_eth_dev_set_mtu(dev_port_id, MTU);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("rte_eth_dev_set_mtu error (port {}) {}\n", dev_port_id, strerror(-ret)));
    }

    ret = rte_eth_dev_adjust_nb_rx_tx_desc(dev_port_id, &nb_rxd, &nb_txd);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("rte_eth_dev_adjust_nb_rx_tx_desc (port {}) {}\n", dev_port_id, strerror(-ret)));
    }

    /* Allocate and set up 1 RX queue per Ethernet port. */
    rxconf = dev_info.default_rxconf;
    rxconf.offloads = port_conf.rxmode.offloads;
    for (uint16_t q = 0; q < rx_rings; q++) {
        ret = rte_eth_rx_queue_setup(dev_port_id, q, nb_rxd, rte_eth_dev_socket_id(dev_port_id), &rxconf, mbuf_pool);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("rte_eth_rx_queue_setup (port {}) {}\n", dev_port_id, strerror(-ret)));
        }
    }

    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    /* Allocate and set up 1 TX queue per Ethernet port. */
    for (uint16_t q = 0; q < tx_rings; q++) {
        ret = rte_eth_tx_queue_setup(dev_port_id, q, nb_txd, rte_eth_dev_socket_id(dev_port_id), &txconf);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("rte_eth_tx_queue_setup (port {}) {}\n", dev_port_id, strerror(-ret)));
        }
    }
    arg->nb_txd = nb_txd;

    ret = rte_eth_dev_start(dev_port_id);
    if (ret < 0) {
        throw std::runtime_error(fmt::format("rte_eth_dev_start (port {}) {}\n", dev_port_id, strerror(-ret)));
    }

    /* Enable RX in promiscuous mode for the Ethernet device. */
    ret = rte_eth_promiscuous_enable(dev_port_id);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("rte_eth_promiscuous_enable (port {}) {}\n", dev_port_id, strerror(-ret)));
    }

    // Link status
    struct rte_eth_link link;
    ret = rte_eth_link_get(dev_port_id, &link);
    if (ret < 0) {
        throw std::runtime_error(fmt::format("rte_eth_link_get (port {}) {}\n", dev_port_id, rte_strerror(-ret)));
    }
    if (link.link_status == RTE_ETH_LINK_DOWN) {
        throw std::runtime_error("Link is down\n");
    }

    /* print link status if flag set */
    char link_status_text[RTE_ETH_LINK_MAX_STR_LEN];
    rte_eth_link_to_str(link_status_text, sizeof(link_status_text), &link);
    fmt::print("port {} {}\n", dev_port_id, link_status_text);

    arg->tcp_pkt_maker = std::make_unique<TCPPacketMaker>(mbuf_pool,
        arg->dev_info,
        arg->client_eth_addr,
        arg->server_eth_addr,
        arg->client_ip_addr,
        arg->server_ip_addr,
        arg->client_port,
        arg->server_port);

    return 0;
}

int conv_lcore_to_idx(int lcore_id, std::vector<client_args>& params)
{
    auto it = std::find_if(params.begin(), params.end(), [&](const client_args& p) { return p.lcore_id == lcore_id; });
    int i = it - params.begin();
    return i;
}

void init(
    std::vector<client_args>& client_argses,
    const std::vector<std::string>& dev_pci_addrs,
    const std::string& lcores,
    const std::string& socket_mem,
    const std::string& iova_mode,
    const uint32_t bandwidth_in_gbps,
    const std::string& log_level)
{

    std::vector<std::string> arguments = { ".", "--lcores", lcores, "--socket-mem", socket_mem, "--iova-mode", iova_mode, "--log-level", log_level, "--file-prefix", dev_pci_addrs[0] + lcores };

    for (auto& dev_pci_addr : dev_pci_addrs) {
        arguments.push_back("-a");
        arguments.push_back(dev_pci_addr);
    }

    std::vector<char*> args;
    for (auto& a : arguments) {
        args.push_back(&a[0]);
    }
    args.push_back(nullptr);

    int ret = rte_eal_init(args.size() - 1, args.data());
    if (ret < 0) {
        fmt::print("error rte_eal_init\n");
        return;
    }

    int lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id)
    {
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        rte_eal_remote_launch(init_rte_env, (void*)&client_argses.at(i), lcore_id);
    }

    {
        lcore_id = rte_get_main_lcore();
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        init_rte_env((void*)&client_argses.at(i));
    }

    rte_eal_mp_wait_lcore();
}

std::tuple<uint32_t, uint32_t> wait_packet(uint16_t portid,
    const std::function<bool()>& timeout_func,
    const std::function<bool(const rte_ipv4_hdr* ipv4, const rte_tcp_hdr* tcp)>& packet_func)
{

    uint32_t sent_seq = 0;
    uint32_t recv_ack = 0;
    bool expected_packet_reached = false;

    while (timeout_func() && (!expected_packet_reached)) {
        rte_mbuf* buf;
        if (rte_eth_rx_burst(portid, 0, &buf, 1)) {
            rte_net_hdr_lens hdr_lens;
            auto packet_type = rte_net_get_ptype(buf, &hdr_lens, RTE_PTYPE_ALL_MASK);
            if (RTE_ETH_IS_IPV4_HDR(packet_type) && ((packet_type & RTE_PTYPE_L4_MASK) == RTE_PTYPE_L4_TCP)) {
                auto ipv4 = rte_pktmbuf_mtod_offset(buf, rte_ipv4_hdr*, hdr_lens.l2_len);
                auto tcp = rte_pktmbuf_mtod_offset(buf, rte_tcp_hdr*, hdr_lens.l2_len + hdr_lens.l3_len);
                if (packet_func(ipv4, tcp)) {
                    sent_seq = rte_be_to_cpu_32(tcp->sent_seq);
                    recv_ack = rte_be_to_cpu_32(tcp->recv_ack);
                    expected_packet_reached = true;
                }
            }
            rte_pktmbuf_free(buf);
        }
    }

    return std::make_tuple(sent_seq, recv_ack);
}

int wait_3wayhandshake_lcore(void* arg1)
{
    struct client_args* arg = (struct client_args*)arg1;

    uint16_t dev_port_id = arg->dev_port_id;

    uint32_t bandwidth_in_gbps = arg->bandwidth_in_gbps;

    srand(time(0));

    //
    // 3-way handshake
    //
    // Send SYN
    uint32_t seqn;
    uint32_t ackn = 0;
    while (g_running) {
        seqn = rand();
        {
            auto bs = arg->tcp_pkt_maker->build(seqn, 0, RTE_TCP_SYN_FLAG, true, nullptr, 0);
            auto nb = rte_eth_tx_burst(dev_port_id, 0, bs.data(), bs.size());
            if (nb != bs.size()) {
                throw std::runtime_error("Failed to send SYN");
            }
            seqn += 1;
        }

        // Wait SYN/ACK
        auto start_tp = std::chrono::high_resolution_clock::now();
        auto ns = wait_packet(
            dev_port_id,
            [&]() {
                auto check_tp = std::chrono::high_resolution_clock::now();
                const double du = std::chrono::duration<double>(check_tp - start_tp).count();
                const double timeout = 1.0; // sec
                return g_running && (du < timeout);
            },
            [&](const rte_ipv4_hdr*, const rte_tcp_hdr* tcp) {
                return (tcp->tcp_flags & RTE_TCP_SYN_FLAG) && (tcp->tcp_flags & RTE_TCP_ACK_FLAG) && (tcp->dst_port == rte_cpu_to_be_16(arg->client_port));
            });

        if (seqn == std::get<1>(ns)) {
            ackn = std::get<0>(ns) + 1;
            break;
        } else {
            fmt::print("SYN/ACK didn't reach. Sending SYN....\n");
        }
    }

    // Send ACK
    {
        auto bs = arg->tcp_pkt_maker->build(seqn, ackn, RTE_TCP_ACK_FLAG, false, nullptr, 0);
        auto nb = rte_eth_tx_burst(dev_port_id, 0, bs.data(), bs.size());
        if (nb != bs.size()) {
            throw std::runtime_error("Failed to send ACK");
        }
    }

    arg->seqn = seqn;
    arg->ackn = ackn;

    return 0;
}

void print_statistics(const double elapsed, const double size)
{
    double total_gbytes = size / (1024.0 * 1024.0 * 1024.0);
    double total_gbits = total_gbytes * 8;

    printf("************************************\n");
    printf("%lf [GB], %lf sec\n", total_gbytes, elapsed);
    printf("Bandwidth: %lf [Gb/s]\n", total_gbits / elapsed);
    printf("************************************\n");
}

int sending_tcp_data(void* arg1)
{
    struct client_args* arg = (struct client_args*)arg1;
    struct rte_mempool* mbuf_pool = arg->mbuf_pool;

    uint16_t dev_port_id = arg->dev_port_id;

    uint32_t bandwidth_in_gbps = arg->bandwidth_in_gbps;

    uint16_t nb_txd = arg->nb_txd;

    uint32_t seqn = arg->seqn;
    uint32_t wait_seqn = arg->seqn;
    uint32_t ackn = arg->ackn;

    uint32_t check_ack_freq = arg->check_ack_freq;

    const size_t chunk_size = arg->chunk_size; // 16MiB

    bool ignore_ack = arg->ignore_ack;

    bool is_first = true;

    // NOTE: Assume frame index is always zero to make things easy
    std::vector<std::tuple<std::vector<struct rte_mbuf*>, size_t>> bss;
    {
        auto& raw_buffer = arg->send_buf;
        for (size_t i = 0; i < raw_buffer.size(); i += chunk_size) {
            size_t payload_size = std::min(chunk_size, raw_buffer.size() - i);
            auto bs = arg->tcp_pkt_maker->build(0, ackn, RTE_TCP_PSH_FLAG | RTE_TCP_ACK_FLAG, false, &raw_buffer[i], payload_size);
            bss.push_back(std::make_tuple(bs, payload_size));
        }
    }

    std::vector<struct rte_mbuf*> bs;

    int32_t n = 0;

    uint64_t sent_in_bytes = 0;

    auto ts1 = std::chrono::high_resolution_clock::now();

    size_t tx_time = 0;
    size_t rtt_measured = 0;

    double sum_rtt = 0;
    double sum2_rtt = 0;
    double max_rtt = -1;
    double min_rtt = 1000000;

    while (g_running) {

        auto tx_st = std::chrono::high_resolution_clock::now();

        for (const auto& bs_info : bss) {
            const auto& reference_bs = std::get<0>(bs_info);
            const auto& payload_size = std::get<1>(bs_info);

            // Clone mbufs
            bs.resize(reference_bs.size());
            for (auto i = 0; i < bs.size(); ++i) {
                bs[i] = rte_pktmbuf_clone(reference_bs[i], mbuf_pool);

                auto b = bs[i];

                auto* tcp = rte_pktmbuf_mtod_offset(b, struct rte_tcp_hdr*, b->l2_len + b->l3_len);
                tcp->sent_seq = rte_cpu_to_be_32(seqn);

                seqn += rte_pktmbuf_pkt_len(b) - b->l2_len - b->l3_len - b->l4_len;
            }

            if (!is_first) {
                if (!ignore_ack && ((tx_time + 1) % check_ack_freq == 0)) {
                    auto ns = wait_packet(
                        dev_port_id,
                        [&]() {
                            return g_running;
                        },
                        [&](const rte_ipv4_hdr*, const rte_tcp_hdr* tcp) {
                            return (tcp->tcp_flags & RTE_TCP_ACK_FLAG) && (tcp->dst_port == rte_cpu_to_be_16(arg->client_port)) && (rte_be_to_c\
pu_32(tcp->recv_ack) == wait_seqn);
                        });
                    auto tx_ed = std::chrono::high_resolution_clock::now();

                    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(tx_ed - tx_st).count() / 1000.0;
                    if (tx_time > 10) {
                        rtt_measured++;
                        sum_rtt += elapsed;
                        sum2_rtt += elapsed * elapsed;
                        max_rtt = std::max(max_rtt, elapsed);
                        min_rtt = std::min(min_rtt, elapsed);
                    }
                }
            } else {
                is_first = false;
            }
            wait_seqn = seqn;

            // Transmit
            if (tx_time % check_ack_freq == 0) {
                tx_st = std::chrono::high_resolution_clock::now();
            }

            size_t transmitted_num = 0;
            while (transmitted_num < bs.size()) {
                auto nb = rte_eth_tx_burst(dev_port_id, 0, bs.data() + transmitted_num, bs.size() - transmitted_num);
                transmitted_num += nb;
            }

            tx_time++;

            if (!g_running) {
                break;
            }

            if (bandwidth_in_gbps) {
                sent_in_bytes += payload_size;

                auto stop_time = (8 * sent_in_bytes / (bandwidth_in_gbps * 1024.0 * 1024.0 * 1024.0)) - (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ts1).count() / 1000.0);

                if (stop_time > 0) {
                    usleep(static_cast<int>(stop_time * 1000 * 1000));
                }
            }
        }

        if ((n++ % 100) == 0) {
            std::cout << "." << std::flush;
        }

        if (0 < arg->send_buf_num && arg->send_buf_num <= n) {
            break;
        }
    }

    auto ts2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ = ts2 - ts1;
    double elapsed = elapsed_.count();

    // FIN
    {
        auto bs = arg->tcp_pkt_maker->build(seqn, ackn, RTE_TCP_FIN_FLAG, false, nullptr, 0);
        auto nb = rte_eth_tx_burst(dev_port_id, 0, bs.data(), bs.size());
        if (nb != bs.size()) {
            throw std::runtime_error("Failed to send ACK");
        }
    }

    fmt::print("Sent all {} frames\n", n);

    struct rte_eth_stats stats;
    if (!rte_eth_stats_get(dev_port_id, &stats)) {
        print_statistics(elapsed, stats.obytes);
    }

    if (n > 0) {
        printf("*****************************\n");
        printf("average rtt : %f usec \n", sum_rtt / rtt_measured);
        printf("std_dev rtt : %f usec \n", std::sqrt(sum2_rtt / rtt_measured - (sum_rtt / rtt_measured) * (sum_rtt / rtt_measured)));
        printf("minimum rtt : %f usec \n", (double)min_rtt);
        printf("maximum rtt : %f usec \n", (double)max_rtt);
        printf("*****************************\n");
    }

    return 0;
}

void run_tcp(
    std::vector<client_args>& client_argses)
{
    int lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id)
    {
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        rte_eal_remote_launch(wait_3wayhandshake_lcore, (void*)&client_argses.at(i), lcore_id);
    }

    {
        lcore_id = rte_get_main_lcore();
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        wait_3wayhandshake_lcore((void*)&client_argses.at(i));
    }

    rte_eal_mp_wait_lcore();

    RTE_LCORE_FOREACH_WORKER(lcore_id)
    {
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        rte_eal_remote_launch(sending_tcp_data, (void*)&client_argses.at(i), lcore_id);
    }

    {
        lcore_id = rte_get_main_lcore();
        int i = conv_lcore_to_idx(lcore_id, client_argses);
        sending_tcp_data((void*)&client_argses.at(i));
    }

    rte_eal_mp_wait_lcore();
}

static inline std::vector<std::string> split(std::string str)
{
    std::vector<std::string> ret;
    std::stringstream ss(str);
    std::string word;
    while (!ss.eof()) {
        std::getline(ss, word, ',');
        ret.push_back(word);
    }
    return ret;
}

std::vector<int> split_int(std::string str)
{
    std::vector<int> ret;
    std::stringstream ss(str);
    std::string word;
    while (!ss.eof()) {
        std::getline(ss, word, ',');
        ret.push_back(std::atoi(word.c_str()));
    }
    return ret;
}

static inline std::vector<std::string> convert_to_pci_addr(std::vector<int> idx)
{
    std::vector<std::string> mlx_path;
    for (const fs::directory_entry& dir_entry : fs::recursive_directory_iterator("/sys/devices")) {
        if (dir_entry.is_regular_file()) {
            if (dir_entry.path().filename() == "vendor") {
                std::ifstream ifs(dir_entry.path().c_str());
                std::string data;
                ifs >> data;
                if (data == "0x15b3") {
                    mlx_path.push_back(dir_entry.path().parent_path().filename());
                }
            }
        }
    }

    std::sort(mlx_path.begin(), mlx_path.end());

    std::vector<std::string> ret;

    for (int i = 0; i < idx.size(); ++i) {
        if (mlx_path.size() <= idx.at(i)) {
            throw std::runtime_error("get_pci_name out of index\n");
        }

        auto& pci_addr = mlx_path.at(idx.at(i));
        ret.push_back(pci_addr);
        fmt::print("PCI device {} is selected\n", pci_addr);
    }

    return ret;
}

static inline std::string get_socket_mem(const std::string& core_list)
{
    int max_socket_num = 0;

    std::regex re("cpu[0-9]+");

    for (const fs::directory_entry& dir_entry : fs::directory_iterator("/sys/devices/system/cpu")) {
        if (dir_entry.is_directory()) {
            if (std::regex_match(dir_entry.path().filename().c_str(), re)) {
                std::string path = dir_entry.path().string() + "/topology/physical_package_id";
                std::ifstream ifs(path);
                int pys_id;
                ifs >> pys_id;
                max_socket_num = std::max(max_socket_num, pys_id);
            }
        }
    }

    std::vector<std::string> tmp(max_socket_num + 1, "0");
    for (auto&& core : split(core_list)) {
        std::string path = "/sys/devices/system/cpu/cpu" + core + "/topology/physical_package_id";

        std::ifstream ifs(path);
        int pys_id;
        ifs >> pys_id;

        tmp.at(pys_id) = "4096";
    }

    std::string ret = tmp[0];
    for (int i = 1; i < tmp.size(); ++i) {
        ret += "," + tmp.at(i);
    }

    fmt::print("socket_mem is {}\n", ret);
    return ret;
}

std::vector<int> calc_virtual_dev_port_ids(std::vector<int>& mlx_port_id)
{
    std::vector<int> sorted_idx(mlx_port_id.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [&mlx_port_id](int i1, int i2) { return mlx_port_id[i1] < mlx_port_id[i2]; });

    std::vector<int> res(mlx_port_id.size());
    for (int i = 0; i < sorted_idx.size(); ++i) {
        res.at(sorted_idx.at(i)) = i;
    }
    return res;
}

void init_send_buf(std::vector<uint8_t>& send_buf)
{
    for (size_t i = 0; i < send_buf.size(); ++i) {
        send_buf[i] = (uint8_t)i;
    }
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("program_name");

    program.add_argument("--dev_port_id")
        .required()
        .help("specify the device port ids.");

    program.add_argument("--lcores")
        .required()
        .help("specify the lcores");

    program.add_argument("--iova_mode")
        .default_value<std::string>("pa")
        .help("specify the iova mode");

    program.add_argument("--server_eth_addr")
        .required()
        .help("specify the server ether address");

    program.add_argument("--client_eth_addr")
        .required()
        .help("specify the client ether address");

    program.add_argument("--server_ip_addr")
        .required()
        .help("specify the server ip address");

    program.add_argument("--client_ip_addr")
        .required()
        .help("specify the client ip address");

    program.add_argument("--server_tcp_port")
        .required()
        .help("specify the server tcp port");

    program.add_argument("--client_tcp_port")
        .required()
        .help("specify the client tcp port");

    program.add_argument("--bandwidth")
        .default_value<uint32_t>(0)
        .help("specify the bandwidth")
        .scan<'u', uint32_t>();

    program.add_argument("--log_level")
        .default_value<std::string>(".*,0")
        .help("specify the log level");

    program.add_argument("--frame_size")
        .default_value<size_t>(256 * 1024 * 1024)
        .help("specify the one frame size")
        .scan<'u', size_t>();

    program.add_argument("--frame_num")
        .default_value<uint32_t>(128)
        .help("specify the # of frame")
        .scan<'u', uint32_t>();

    program.add_argument("--chunk_size")
        .default_value<uint32_t>(2 * 1024 * 1024)
        .help("specify the chunk size for one rx")
        .scan<'u', uint32_t>();

    program.add_argument("--check_ack_interval")
        .default_value<uint32_t>(1)
        .help("specify checking ack frequency")
        .scan<'u', uint32_t>();

    program.add_argument("--output_sent_file")
        .default_value<std::string>("")
        .help("specify the filename");

    program.add_argument("--ignore_ack")
        .help("switch ignore ack")
        .flag();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<int> dev_mlx_port_ids = split_int(program.get<std::string>("--dev_port_id"));
    std::vector<std::string> dev_pci_addrs = convert_to_pci_addr(dev_mlx_port_ids);
    std::string lcores = program.get<std::string>("--lcores");
    std::string iova_mode = program.get<std::string>("--iova_mode");
    std::vector<std::string> server_eth_addrs = split(program.get<std::string>("--server_eth_addr"));
    std::vector<std::string> client_eth_addrs = split(program.get<std::string>("--client_eth_addr"));
    std::vector<std::string> server_ip_addrs = split(program.get<std::string>("--server_ip_addr"));
    std::vector<std::string> client_ip_addrs = split(program.get<std::string>("--client_ip_addr"));
    std::vector<std::string> server_tcp_ports = split(program.get<std::string>("--server_tcp_port"));
    std::vector<std::string> client_tcp_ports = split(program.get<std::string>("--client_tcp_port"));
    uint32_t bandwidth_in_gbps = program.get<uint32_t>("--bandwidth");
    std::string log_level = program.get<std::string>("--log_level");
    size_t frame_size = program.get<size_t>("--frame_size");
    uint32_t frame_num = program.get<uint32_t>("--frame_num");
    uint32_t chunk_size = program.get<uint32_t>("--chunk_size");
    std::string output_file = program.get<std::string>("--output_sent_file");
    uint32_t check_ack_freq = program.get<uint32_t>("--check_ack_interval");
    bool ignore_ack = (program["--ignore_ack"] == true);

    auto socket_mem = get_socket_mem(lcores);

    // dpdk uses 0-indexed port id.
    // when dev_mlx_port_ids is 1,4,3, dpdk rename it to 0,2,1
    std::vector<int> virtual_dev_port_ids = calc_virtual_dev_port_ids(dev_mlx_port_ids);
    auto core_v = split_int(lcores);

    std::vector<uint8_t> send_buf(frame_size);
    init_send_buf(send_buf);

    if (output_file != "") {
        std::ofstream ofs(output_file.c_str());
        ofs.write((char*)(&send_buf[0]), send_buf.size());
    }

    std::vector<client_args> client_argses(core_v.size());
    for (int i = 0; i < client_argses.size(); ++i) {
        client_argses.at(i).lcore_id = core_v.at(i);
        client_argses.at(i).dev_port_id = virtual_dev_port_ids.at(i);
        client_argses.at(i).server_eth_addr = server_eth_addrs.at(i);
        client_argses.at(i).client_eth_addr = client_eth_addrs.at(i);
        client_argses.at(i).server_ip_addr = server_ip_addrs.at(i);
        client_argses.at(i).client_ip_addr = client_ip_addrs.at(i);
        client_argses.at(i).server_port = std::stoi(server_tcp_ports.at(i));
        client_argses.at(i).client_port = std::stoi(client_tcp_ports.at(i));
        client_argses.at(i).bandwidth_in_gbps = bandwidth_in_gbps;
        client_argses.at(i).send_buf = send_buf;
        client_argses.at(i).send_buf_num = frame_num;
        client_argses.at(i).chunk_size = chunk_size;
        client_argses.at(i).check_ack_freq = check_ack_freq;
        client_argses.at(i).ignore_ack = ignore_ack;
    }

    std::sort(client_argses.begin(), client_argses.end(),
        [](client_args& i1, client_args& i2) { return i1.lcore_id < i2.lcore_id; });

    try {
        signal(SIGINT, catch_int);
        init(client_argses, dev_pci_addrs, lcores,
            socket_mem, iova_mode, bandwidth_in_gbps, log_level);

        run_tcp(client_argses);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        rte_exit(EXIT_FAILURE, "%s", e.what());
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        rte_exit(EXIT_FAILURE, "Unknown error");
    }
}
