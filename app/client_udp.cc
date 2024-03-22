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
    fmt::print("Interrupted\n");
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

class UDPPacketMaker {

public:
    UDPPacketMaker(rte_mempool* mbuf_pool,
        rte_eth_dev_info dev_info,
        const std::string& src_ether_addr,
        const std::string& dst_ether_addr,
        const std::string& src_addr,
        const std::string& dst_addr,
        uint16_t src_port,
        std::vector<uint16_t>& dst_ports)
        : mbuf_pool_(mbuf_pool)
        , dev_info_(dev_info)
        , src_port_(rte_cpu_to_be_16(src_port))
    {
        rte_ether_unformat_addr(src_ether_addr.c_str(), &src_ether_addr_);
        rte_ether_unformat_addr(dst_ether_addr.c_str(), &dst_ether_addr_);

        src_addr_ = convert_ipv4_addr(parse_ipv4_addr(src_addr));
        dst_addr_ = convert_ipv4_addr(parse_ipv4_addr(dst_addr));

        for (auto& dst_port : dst_ports) {
            dst_ports_.push_back(rte_cpu_to_be_16(dst_port));
        }
    }

    std::vector<struct rte_mbuf*> build(void* payload_data, size_t payload_size)
    {
        const uint16_t mtu = MTU;
        const uint32_t l2_len = sizeof(struct rte_ether_hdr);
        const uint32_t l3_len = sizeof(struct rte_ipv4_hdr);
        const uint32_t l4_len = sizeof(rte_udp_hdr);
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
            buf->packet_type = RTE_PTYPE_L2_ETHER | RTE_PTYPE_L3_IPV4 | RTE_PTYPE_L4_UDP;

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
            ipv4->next_proto_id = IPPROTO_UDP;
            ipv4->src_addr = src_addr_;
            ipv4->dst_addr = dst_addr_;
            ipv4->hdr_checksum = 0;

            // L4
            auto* udp = reinterpret_cast<struct rte_udp_hdr*>(head + l2_len + l3_len);
            udp->src_port = src_port_;
            udp->dst_port = dst_ports_.at(0);
            udp->dgram_cksum = 0;

            size_t segmented_payload_size = std::min(std::min(static_cast<size_t>(remaining), static_cast<size_t>(rte_pktmbuf_tailroom(buf))), static_cast<size_t>(mss));
            uint8_t* body = reinterpret_cast<uint8_t*>(rte_pktmbuf_append(buf, segmented_payload_size));
            memcpy(body, payload_ptr, segmented_payload_size);

            uint16_t dgram_len = segmented_payload_size + sizeof(struct rte_udp_hdr);

            remaining -= segmented_payload_size;
            payload_ptr += segmented_payload_size;

            buf->ol_flags = RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_UDP_CKSUM;
            buf->l2_len = l2_len;
            buf->l3_len = l3_len;
            buf->l4_len = l4_len;

            ipv4->total_length = rte_cpu_to_be_16(buf->pkt_len - l2_len);
            ipv4->hdr_checksum = 0;
            udp->dgram_len = rte_cpu_to_be_16(dgram_len);

            bufs.push_back(buf);
        } while (remaining);

        return bufs;
    }

    std::vector<rte_be16_t> dst_ports_;

private:
    rte_mempool* mbuf_pool_;
    rte_eth_dev_info dev_info_;

    rte_ether_addr src_ether_addr_;
    rte_ether_addr dst_ether_addr_;

    rte_be32_t src_addr_;
    rte_be32_t dst_addr_;

    rte_be16_t src_port_;
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
    std::vector<uint16_t> server_ports;
    uint32_t bandwidth_in_gbps;
    uint16_t nb_txd;
    std::unique_ptr<UDPPacketMaker> udp_pkt_maker;
    std::vector<uint8_t> send_buf;
    uint32_t send_buf_num;
    uint32_t chunk_size;
    uint32_t port_change_interval;
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
    // if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_UDP_LRO) {
    //     port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_UDP_LRO;
    //     port_conf.rxmode.max_lro_pkt_size = dev_info.max_lro_pkt_size;
    // }
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
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_UDP_CKSUM) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_UDP_CKSUM;
    }
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_UDP_TSO) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_UDP_TSO;
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

    arg->udp_pkt_maker = std::make_unique<UDPPacketMaker>(mbuf_pool,
        arg->dev_info,
        arg->client_eth_addr,
        arg->server_eth_addr,
        arg->client_ip_addr,
        arg->server_ip_addr,
        arg->client_port,
        arg->server_ports);

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

    std::vector<std::string> arguments = { ".", "--lcores", lcores, "--socket-mem", socket_mem, "--iova-mode", iova_mode, "--log-level", log_level, "--file-prefix", dev_pci_addrs[0] };

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
        fmt::print("rte_eal_init error\n");
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

void print_statistics(const double elapsed, const double size)
{
    double total_gbytes = size / (1024.0 * 1024.0 * 1024.0);
    double total_gbits = total_gbytes * 8;

    printf("************************************\n");
    printf("%lf [GB], %lf sec\n", total_gbytes, elapsed);
    printf("Bandwidth: %lf [Gb/s]\n", total_gbits / elapsed);
    printf("************************************\n");
}

#define NUM_DUP 2048

int sending_udp_data(void* arg1)
{
    struct client_args* arg = (struct client_args*)arg1;
    struct rte_mempool* mbuf_pool = arg->mbuf_pool;

    uint32_t port_change_interval = arg->port_change_interval;

    uint16_t dev_port_id = arg->dev_port_id;

    uint32_t bandwidth_in_gbps = arg->bandwidth_in_gbps;

    uint16_t nb_txd = arg->nb_txd;

    const size_t chunk_size = arg->chunk_size; // 16MiB

    // NOTE: Assume frame index is always zero to make things easy
    std::vector<std::tuple<std::vector<struct rte_mbuf*>, size_t>> bss;
    {
        auto& raw_buffer = arg->send_buf;
        for (int j = 0; j < NUM_DUP; j++) {
            for (size_t i = 0; i < raw_buffer.size(); i += chunk_size) {
                size_t payload_size = std::min(chunk_size, raw_buffer.size() - i);
                auto bs = arg->udp_pkt_maker->build(&raw_buffer[i], payload_size);
                bss.push_back(std::make_tuple(bs, payload_size));
            }
        }
    }

    std::vector<struct rte_mbuf*> bs;

    int32_t n = 0;

    uint64_t sent_in_bytes = 0;

    auto ts1 = std::chrono::high_resolution_clock::now();

    uint32_t itr = 0;
    int ports_num = arg->udp_pkt_maker->dst_ports_.size();
    std::vector<uint64_t> count_bytes(ports_num, 0);
    std::vector<uint8_t> send_pkt_id(ports_num, 0);

    uint32_t num_itr = (NUM_DUP - 1 + arg->send_buf_num) / NUM_DUP;

    while (g_running) {

        for (const auto& bs_info : bss) {
            const auto& reference_bs = std::get<0>(bs_info);
            const auto& payload_size = std::get<1>(bs_info);

            // Clone mbufs
            bs.resize(reference_bs.size());
            for (auto i = 0; i < bs.size(); ++i) {
                bs[i] = rte_pktmbuf_clone(reference_bs[i], mbuf_pool);
                // bs[i] = rte_pktmbuf_copy(reference_bs[i], mbuf_pool, 0, 7972);

                auto* udp = rte_pktmbuf_mtod_offset(bs[i], struct rte_udp_hdr*, bs[i]->l2_len + bs[i]->l3_len);
                int idx = (itr / port_change_interval) % ports_num;
                count_bytes.at(idx) += rte_cpu_to_be_16(udp->dgram_len);
                udp->dst_port = arg->udp_pkt_maker->dst_ports_.at(idx);

                auto* body = rte_pktmbuf_mtod_offset(bs[i], uint8_t*, bs[i]->l2_len + bs[i]->l3_len + bs[i]->l4_len);
                body[0] = send_pkt_id.at(idx);
                send_pkt_id.at(idx)++;
                // std::cout << bs[i]->l2_len << " " << bs[i]->l3_len << " " << bs[i]->l4_len << " l2 l3 l4" << std::endl;
                itr++;
            }

            while (rte_eth_tx_descriptor_status(dev_port_id, 0, nb_txd * 3 / 4) != RTE_ETH_TX_DESC_DONE && g_running) {
            }

            // Transmit
            auto nb = rte_eth_tx_burst(dev_port_id, 0, bs.data(), bs.size());
            if (nb != bs.size()) {
                throw std::runtime_error("Failed to send data");
            }

            if (!g_running) {
                break;
            }
        }

        if ((n++ % 1000) == 0) {
            std::cout << "." << std::flush;
        }

        if (0 < num_itr && num_itr <= n) {
            break;
        }
    }

    auto ts2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_ = ts2 - ts1;
    double elapsed = elapsed_.count();

    fmt::print("Sent all {} frames\n", n);

    for (int i = 0; i < ports_num; ++i) {
        fmt::print("port {} sent {} bytes\n", rte_cpu_to_be_16(arg->udp_pkt_maker->dst_ports_.at(i)), count_bytes.at(i));
    }

    struct rte_eth_stats stats;
    if (!rte_eth_stats_get(dev_port_id, &stats)) {
        print_statistics(elapsed, stats.obytes);
    }
    return 0;
}

void wait_packet(uint16_t portid,
    const std::function<bool()>& timeout_func,
    const std::function<bool(const rte_udp_hdr* udp, const uint8_t* payload)>& packet_func)
{

    uint32_t sent_seq = 0;
    uint32_t recv_ack = 0;
    bool expected_packet_reached = false;

    while (timeout_func() && (!expected_packet_reached)) {
        rte_mbuf* buf;
        if (rte_eth_rx_burst(portid, 0, &buf, 16)) {
            rte_net_hdr_lens hdr_lens;
            auto packet_type = rte_net_get_ptype(buf, &hdr_lens, RTE_PTYPE_ALL_MASK);
            if (RTE_ETH_IS_IPV4_HDR(packet_type) && ((packet_type & RTE_PTYPE_L4_MASK) == RTE_PTYPE_L4_UDP)) {
                auto udp = rte_pktmbuf_mtod_offset(buf, rte_udp_hdr*, hdr_lens.l2_len + hdr_lens.l3_len);
                auto payload = rte_pktmbuf_mtod_offset(buf, uint8_t*, hdr_lens.l2_len + hdr_lens.l3_len + hdr_lens.l4_len);
                if (packet_func(udp, payload)) {
                    expected_packet_reached = true;
                }
            }
            rte_pktmbuf_free(buf);
        }
    }

    return;
}

int measure_udp_rtt(void* arg1)
{
    struct client_args* arg = (struct client_args*)arg1;
    struct rte_mempool* mbuf_pool = arg->mbuf_pool;

    uint16_t dev_port_id = arg->dev_port_id;

    uint32_t bandwidth_in_gbps = arg->bandwidth_in_gbps;

    uint16_t nb_txd = arg->nb_txd;

    const size_t chunk_size = arg->chunk_size; // 16MiB

    // NOTE: Assume frame index is always zero to make things easy
    std::vector<std::tuple<std::vector<struct rte_mbuf*>, size_t>> bss;
    {
        auto& raw_buffer = arg->send_buf;
        for (size_t i = 0; i < raw_buffer.size(); i += chunk_size) {
            size_t payload_size = std::min(chunk_size, raw_buffer.size() - i);
            auto bs = arg->udp_pkt_maker->build(&raw_buffer[i], payload_size);
            bss.push_back(std::make_tuple(bs, payload_size));
        }
    }

    std::vector<struct rte_mbuf*> bs;

    int32_t n = 0;

    uint64_t sent_in_bytes = 0;

    uint8_t payload[1];

    double sum_rtt = 0;
    double sum2_rtt = 0;
    double max_rtt = -1;
    double min_rtt = 1000000;

    int64_t rtt_count = 0;

    while (g_running) {

        auto bs = arg->udp_pkt_maker->build(payload, 1)[0];

        auto ts1 = std::chrono::high_resolution_clock::now();
        auto nb = rte_eth_tx_burst(dev_port_id, 0, &bs, 1);
        if (nb != 1) {
            throw std::runtime_error("Failed to send data");
        }

        wait_packet(
            dev_port_id,
            [&]() {
                return g_running;
            },
            [&](const rte_udp_hdr* udp, const uint8_t* recved_payload) {
                return (udp->dst_port == rte_cpu_to_be_16(arg->client_port) && recved_payload[0] == payload[0]);
            });
        auto ts2 = std::chrono::high_resolution_clock::now();

        if ((n++ % 10000) == 0) {
            std::cout << "." << std::flush;
        }

        double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count() / 1000.0;
        if (n > 50) {
            sum_rtt += elapsed;
            sum2_rtt += elapsed * elapsed;
            max_rtt = std::max(max_rtt, elapsed);
            min_rtt = std::min(min_rtt, elapsed);
            rtt_count++;
            if (elapsed > 1000) {
                printf("%d %f rtt\n", n, elapsed);
            }
        }

        payload[0]++;

        if (0 < arg->send_buf_num && arg->send_buf_num <= n) {
            break;
        }
    }

    if (rtt_count > 0) {
        printf("*****************************\n");
        printf("average rtt : %f usec \n", sum_rtt / rtt_count);
        printf("std_dev rtt : %f usec \n", std::sqrt(sum2_rtt / rtt_count - (sum_rtt / rtt_count) * (sum_rtt / rtt_count)));
        printf("minimum rtt : %f usec \n", (double)min_rtt);
        printf("maximum rtt : %f usec \n", (double)max_rtt);
        printf("*****************************\n");
    }

    return 0;
}

void run_udp(
    std::vector<client_args>& client_argses,
    bool is_rtt)
{
    int lcore_id;

    if (is_rtt) {
        RTE_LCORE_FOREACH_WORKER(lcore_id)
        {
            int i = conv_lcore_to_idx(lcore_id, client_argses);
            rte_eal_remote_launch(measure_udp_rtt, (void*)&client_argses.at(i), lcore_id);
        }

        {
            lcore_id = rte_get_main_lcore();
            int i = conv_lcore_to_idx(lcore_id, client_argses);
            measure_udp_rtt((void*)&client_argses.at(i));
        }

        rte_eal_mp_wait_lcore();
    } else {
        RTE_LCORE_FOREACH_WORKER(lcore_id)
        {
            int i = conv_lcore_to_idx(lcore_id, client_argses);
            rte_eal_remote_launch(sending_udp_data, (void*)&client_argses.at(i), lcore_id);
        }

        {
            lcore_id = rte_get_main_lcore();
            int i = conv_lcore_to_idx(lcore_id, client_argses);
            sending_udp_data((void*)&client_argses.at(i));
        }

        rte_eal_mp_wait_lcore();
    }
}

static inline std::vector<std::string> split(std::string str, char delim = ',')
{
    std::vector<std::string> ret;
    std::stringstream ss(str);
    std::string word;
    while (!ss.eof()) {
        std::getline(ss, word, delim);
        ret.push_back(word);
    }
    return ret;
}

template <class T>
std::vector<T> split_int(std::string str)
{
    std::vector<T> ret;
    std::stringstream ss(str);
    std::string word;
    while (!ss.eof()) {
        std::getline(ss, word, ',');
        ret.push_back((T)std::atoi(word.c_str()));
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

    program.add_argument("--server_udp_port")
        .required()
        .help("specify the server udp port");

    program.add_argument("--client_udp_port")
        .required()
        .help("specify the client udp port");

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
    ;

    program.add_argument("--frame_num")
        .default_value<uint32_t>(128)
        .help("specify the # of frame")
        .scan<'u', uint32_t>();

    program.add_argument("--chunk_size")
        .default_value<uint32_t>(2 * 1024 * 1024)
        .help("specify the chunk size for one rx")
        .scan<'u', uint32_t>();

    program.add_argument("--is_rtt")
        .default_value(false)
        .implicit_value(true)
        .help("measure rtt");

    program.add_argument("--port_change_interval")
        .default_value<uint32_t>(1024)
        .help("specify the port changing interval")
        .scan<'u', uint32_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::vector<int> dev_mlx_port_ids = split_int<int>(program.get<std::string>("--dev_port_id"));
    std::vector<std::string> dev_pci_addrs = convert_to_pci_addr(dev_mlx_port_ids);
    std::string lcores = program.get<std::string>("--lcores");
    std::string iova_mode = program.get<std::string>("--iova_mode");
    std::vector<std::string> server_eth_addrs = split(program.get<std::string>("--server_eth_addr"));
    std::vector<std::string> client_eth_addrs = split(program.get<std::string>("--client_eth_addr"));
    std::vector<std::string> server_ip_addrs = split(program.get<std::string>("--server_ip_addr"));
    std::vector<std::string> client_ip_addrs = split(program.get<std::string>("--client_ip_addr"));
    std::vector<std::string> server_udp_ports = split(program.get<std::string>("--server_udp_port"), ':');
    std::vector<std::string> client_udp_ports = split(program.get<std::string>("--client_udp_port"));
    uint32_t bandwidth_in_gbps = program.get<uint32_t>("--bandwidth");
    std::string log_level = program.get<std::string>("--log_level");
    size_t frame_size = program.get<size_t>("--frame_size");
    uint32_t frame_num = program.get<uint32_t>("--frame_num");
    uint32_t chunk_size = program.get<uint32_t>("--chunk_size");
    uint32_t port_change_interval = program.get<uint32_t>("--port_change_interval");
    bool is_rtt = program.get<bool>("--is_rtt");

    auto socket_mem = get_socket_mem(lcores);

    // dpdk uses 0-indexed port id.
    // when dev_mlx_port_ids is 1,4,3, dpdk rename it to 0,2,1
    std::vector<int> virtual_dev_port_ids = calc_virtual_dev_port_ids(dev_mlx_port_ids);
    auto core_v = split_int<int>(lcores);

    std::vector<client_args> client_argses(core_v.size());
    // char* aba = "123456789abcdefghijklmnopqrstuvwxyz\n";
    // frame_size = 37;
    for (int i = 0; i < client_argses.size(); ++i) {
        client_argses.at(i).lcore_id = core_v.at(i);
        client_argses.at(i).dev_port_id = virtual_dev_port_ids.at(i);
        client_argses.at(i).server_eth_addr = server_eth_addrs.at(i);
        client_argses.at(i).client_eth_addr = client_eth_addrs.at(i);
        client_argses.at(i).server_ip_addr = server_ip_addrs.at(i);
        client_argses.at(i).client_ip_addr = client_ip_addrs.at(i);
        client_argses.at(i).server_ports = split_int<uint16_t>(server_udp_ports.at(i));
        client_argses.at(i).client_port = std::stoi(client_udp_ports.at(i));
        client_argses.at(i).bandwidth_in_gbps = bandwidth_in_gbps;
        client_argses.at(i).send_buf = std::vector<uint8_t>(frame_size);
        // memcpy(&client_argses.at(i).send_buf[0], aba, frame_size);
        client_argses.at(i).send_buf_num = frame_num;
        client_argses.at(i).chunk_size = chunk_size;
        client_argses.at(i).port_change_interval = port_change_interval;
    }

    std::sort(client_argses.begin(), client_argses.end(),
        [](client_args& i1, client_args& i2) { return i1.lcore_id < i2.lcore_id; });

    try {
        signal(SIGINT, catch_int);
        init(client_argses, dev_pci_addrs, lcores,
            socket_mem, iova_mode, bandwidth_in_gbps, log_level);

        run_udp(client_argses, is_rtt);

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        rte_exit(EXIT_FAILURE, "%s", e.what());
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        rte_exit(EXIT_FAILURE, "Unknown error");
    }
}
