#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#endif

#include "lng/stream.h"

#include "log.h"

namespace lng {

DPDKStream::Impl::Impl(uint16_t port_id)
    : port_id(port_id)
{

    constexpr uint32_t mtu = 8000;

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
    if (!rte_eth_dev_is_valid_port(0)) {
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
    // if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_TCP_LRO) {
    //     port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;
    //     port_conf.rxmode.max_lro_pkt_size = dev_info.max_lro_pkt_size;
    // }
    // if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_SCATTER) {
    //     port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_SCATTER;
    // }
    // if (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_CHECKSUM) {
    //     port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_CHECKSUM;
    // }

    // if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE) {
    //     port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
    // }
    // if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MULTI_SEGS) {
    //     port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MULTI_SEGS;
    // }
    // if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_IPV4_CKSUM) {
    //     port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM;
    // }
    // if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_TCP_CKSUM) {
    //     port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_CKSUM;
    // }
    // if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_TCP_TSO) {
    //     port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_TSO;
    // }

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
    rte_ether_addr addr;
    ret = rte_eth_macaddr_get(port_id, &addr);
    if (ret != 0) {
        throw std::runtime_error(fmt::format("Failed to get mac address: {}", strerror(-ret)));
    }
    log::info("Port {} MAC: {:x}::{:x}::{:x}::{:x}::{:x}::{:x}", port_id, RTE_ETHER_ADDR_BYTES(&addr));

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


}

DPDKStream::Impl::~Impl() {
    auto ret = rte_eth_dev_stop(port_id);
    if (ret < 0) {
        log::error("Failed to stop device: {}", strerror(-ret));
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
