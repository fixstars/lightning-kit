#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)
#include <rte_arp.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#endif

#include "lng/stream.h"

#include "log.h"

namespace lng {

namespace {

void make_arp_reply(rte_mbuf *buf, rte_ether_hdr *eth_h)
{
    auto arp_h = (rte_arp_hdr *) ((char *)eth_h + sizeof(rte_ether_hdr));
    auto arp_op = rte_be_to_cpu_16(arp_h->arp_opcode);
    auto arp_pro = rte_be_to_cpu_16(arp_h->arp_protocol);

    if ((rte_be_to_cpu_16(arp_h->arp_hardware) != RTE_ARP_HRD_ETHER) ||
        (arp_pro != RTE_ETHER_TYPE_IPV4) ||
        (arp_h->arp_hlen != 6) ||
        (arp_h->arp_plen != 4)
       ) {
        rte_pktmbuf_free(buf);
        auto msg = "Unexpected ARP packet header";
        log::error(msg);
        throw std::runtime_error(msg);
    }

    if (arp_op != RTE_ARP_OP_REQUEST) {
        rte_pktmbuf_free(buf);
        auto msg = fmt::format("DPDK: Unexpected ARP operation : %d", arp_op);
        log::error(msg);
        throw std::runtime_error(msg);
    }

    /* Use source MAC address as destination MAC address. */
    rte_ether_addr_copy(&eth_h->src_addr, &eth_h->dst_addr);
    /* Set source MAC address with MAC address of TX port */
    rte_ether_addr_copy(&server_eth_addr, &eth_h->src_addr);

    arp_h->arp_opcode = rte_cpu_to_be_16(RTE_ARP_OP_REPLY);

    rte_ether_addr_copy(&arp_h->arp_data.arp_sha, &arp_h->arp_data.arp_tha);
    rte_ether_addr_copy(&eth_h->src_addr, &arp_h->arp_data.arp_sha);

    /* Swap IP addresses in ARP payload */
    std::swap(arp_h->arp_data.arp_sip, arp_h->arp_data.arp_tip);
}

std::vector<std::tuple<std::string, rte_ether_addr>> get_netdev() {

    std::vector<std::tuple<std::string, std::string>> netdevs;

    auto if_nidxs = if_nameindex();
    if (if_nidxs == NULL ) {
        return netdevs;
    }

    for (auto if_itr = if_nidxs; if_itr->if_index != 0 || if_itr->if_name != NULL; ++if_itr) {

        auto fd = socket(AF_INET, SOCK_DGRAM, 0);

        //Type of address to retrieve - IPv4 IP address
        ifreq if_req;
        if_req.ifr_addr.sa_family = AF_INET;

        //Copy the interface name in the ifreq structure
        strncpy(if_req.ifr_name, if_itr->if_name , IFNAMSIZ-1);

        if (ioctl(fd, SIOCGIFADDR, &ifr) == -1) {
            close(fd);
            auto msg = "SIOCGIFADDR";
            log::error(msg);
            throw std::runtime_error(msg);
        }

        std::string ip_addr(inet_ntoa(( (struct sockaddr_in *)&if_req.ifr_addr )->sin_addr));

        if (ioctl(fd, SIOCGIFHWADDR, &ifr) == -1) {
            close(fd);
            auto msg = "SIOCGIFHWADDR";
            log::error(msg);
            throw std::runtime_error(msg);
        }

        close(fd);

        auto *eth_addr = reinterpret_cast<uint8_t*>(if_req.ifr_hwaddr.sa_data);

        // printf("%s -> %s, %02x:%02x:%02x:%02x:%02x:%02x\n", if_itr->if_name, ip_addr.c_str(), eth_addr[0], eth_addr[1], eth_addr[2], eth_addr[3], eth_addr[4], eth_addr[5]);

        netdevs.push_back(std::make_tuple(ip_addr, ));
    }

    if_freenameindex(if_nidxs);

    return 0;
}

}

DPDKStream::from_eth_addr(const std::string& eth_addr) {

}

DPDKStream::Impl::Impl(uint16_t port_id)
    : port_id(port_id)
{
    constexpr uint32_t mtu = 9000;

    // Initializion the environment abstraction layer
    std::vector<std::string> arguments = {"."};
    std::vector<char *>args;
    for (auto& a : arguments) {
        args.push_back(&a[0]);
    }
    args.push_back(nullptr);

    int ret = rte_eal_init(args.size()-1, args.data());
    if (ret < 0) {
        throw std::runtime_error("Cannot initialize DPDK");
    }

    // Allocates mempool to hold the mbufs
    constexpr uint32_t n = 8192 - 1;
    constexpr uint32_t cache_size = 256;
    constexpr uint32_t data_room_size = RTE_PKTMBUF_HEADROOM + 10 * 1024;

    mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", n, cache_size, 0, data_room_size, rte_socket_id());
    if (mbuf_pool == nullptr) {
        throw std::runtime_error(fmt::format("Cannot create mbuf pool, n={}, cache_size={}, priv_size=0, data_room_size={}",
                                     n, cache_size, data_room_size));
    }

    // Initializing all ports
    if (!rte_eth_dev_is_valid_port(port_id)) {
        throw std::runtime_error(fmt::format("Port {} is not valid", port_id));
    }

    rte_eth_conf port_conf;
    memset(&port_conf, 0, sizeof(rte_eth_conf));

    rte_eth_dev_info dev_info;
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to get ethernet device (port {}) info: {}", port_id, strerror(-ret)));
    }

    port_conf.link_speeds = RTE_ETH_LINK_SPEED_FIXED | RTE_ETH_LINK_SPEED_100G;
    port_conf.rxmode.mtu = mtu;

    // RX offload
    // NOTE: Disabling LRO may occur that rx_burst does not receive any packets
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TCP_LRO) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;
        port_conf.rxmode.max_lro_pkt_size = dev_info.max_lro_pkt_size;
    }
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_SCATTER) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_SCATTER;
    }
    if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_CHECKSUM) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_CHECKSUM;
    }

    // TX offload
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

    // Configure the ethernet device
    const uint16_t rx_rings = 1;
    const uint16_t tx_rings = 1;
    ret = rte_eth_dev_configure(port_id, rx_rings, tx_rings, &port_conf);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to configure device (port {}): {}", port_id, strerror(-ret)));
    }

    ret = rte_eth_dev_set_mtu(port_id, mtu);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to set mtu (port {}), mtu={}: {}", port_id, mtu, strerror(-ret)));
    }

    uint16_t rx_desc_size = 1024;
    uint16_t tx_desc_size = 1024;
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &rx_desc_size, &tx_desc_size);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to adjust Tx/Rx description (port {}): {}", port_id, strerror(-ret)));
    }

    // Allocate and set up 1 rx queue per ethernet port
    rte_eth_rxconf rxconf = dev_info.default_rxconf;
    // rxconf.offloads = port_conf.rxmode.offloads;
    for (auto q = 0; q < rx_rings; q++) {
        ret = rte_eth_rx_queue_setup(port_id, q, rx_desc_size, rte_eth_dev_socket_id(port_id), &rxconf, mbuf_pool);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("Failed to setup Rx queue: {}", strerror(-ret)));
        }
    }

    rte_eth_txconf txconf = dev_info.default_txconf;
    // txconf.offloads = port_conf.txmode.offloads;

    // Allocate and set up 1 tx queue per ethernet port
    for (auto q = 0; q < tx_rings; q++) {
        ret = rte_eth_tx_queue_setup(port_id, q, tx_desc_size, rte_eth_dev_socket_id(port_id), &txconf);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("Failed to setup Tx queue: {}", strerror(-ret)));
        }
    }

    // Starting ethernet port
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        throw std::runtime_error(fmt::format("Failed to start device: {}", strerror(-ret)));
    }

    // Retrieve the port mac address
    ret = rte_eth_macaddr_get(port_id, &self_eth_addr);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to get mac address: {}", strerror(-ret)));
    }
    log::info("Port {} MAC: {:x}::{:x}::{:x}::{:x}::{:x}::{:x}", port_id, RTE_ETHER_ADDR_BYTES(&self_eth_addr));

    ret = rte_eth_promiscuous_enable(port_id);
    if (ret < 0) {
        throw std::runtime_error(fmt::format("Failed to set promiscuous mode: {}", strerror(-ret)));
    }

    // Recreive link status
    rte_eth_link link;
    ret = rte_eth_link_get(port_id, &link);
    if (ret < 0) {
        throw std::runtime_error(fmt::format("Failed to get link status: {}", strerror(-ret)));
    }

    log::info("Link status is {}", link.link_status ? "up" : "down");

    process_arp(self_eth_addr);
}

DPDKStream::Impl::~Impl() {
    auto ret = rte_eth_dev_stop(port_id);
    if (ret < 0) {
        log::error("Failed to stop device: {}", strerror(-ret));
    }

    struct rte_eth_stats stats;
    if (!rte_eth_stats_get(port_id, &stats)) {
        // std::chrono::duration<double> elapsed_ = m_end_tp - m_start_tp;
        // auto elapsed = elapsed_.count();
        // double total_gbytes = stats.ibytes * 1e-9;
        // double total_gbits = total_gbytes * 8;
        log::debug("Total Rx bytes   : {}", stats.ibytes);
        log::debug("Total Rx packets : {}", stats.ipackets);
        // std::cout << "Wrong Rx packets : " << m_wrong_packets << std::endl;
        // std::cout << "Packet loss rate : " << static_cast<double>(m_wrong_packets) / static_cast<double>(stats.ipackets) << std::endl;
        // std::cout << "Elapsed time     : " << elapsed << " sec" << std::endl;
        // std::cout << "Bandwidth        : " << total_gbits/elapsed << " Gbps" << std::endl;
    }

    rte_eal_cleanup();
}

void DPDKStream::put(rte_mbuf *v) {
    auto nb = rte_eth_tx_burst(impl_->port_id, 0, &v, 1);
    if (nb != 1) {
        throw std::runtime_error("rte_eth_tx_burst");
    }
}

bool DPDKStream::get(rte_mbuf **vp) {
    if (!rte_eth_rx_burst(impl_->port_id, 0, vp, 1)) {
        return false;
    }
    return true;
}

} // lng
