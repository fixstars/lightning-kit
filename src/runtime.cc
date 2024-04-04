#if defined(LNG_WITH_DOCA)
#include "lng/doca-util.h"
#endif

#if defined(LNG_WITH_DPDK)
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_net.h>
#endif

#include "lng/runtime.h"

#include "log.h"

namespace lng {

#if defined(LNG_WITH_DOCA)
DOCARuntime::DOCARuntime() {
    doca_log_backend_create_standard();
}
#endif

#if defined(LNG_WITH_DPDK)
void DPDKRuntime::start() {
    // Initializion the environment abstraction layer
    std::vector<std::string> arguments = { "." };
    std::vector<char*> args;
    for (auto& a : arguments) {
        args.push_back(&a[0]);
    }
    args.push_back(nullptr);

    int ret = rte_eal_init(args.size() - 1, args.data());
    if (ret < 0) {
        throw std::runtime_error("Cannot initialize DPDK");
    }

    // Allocates mempool to hold the mbufs
    constexpr uint32_t n = 8192 - 1;
    constexpr uint32_t cache_size = 256;
    constexpr uint32_t data_room_size = RTE_PKTMBUF_HEADROOM + 10 * 1024;

    mbuf_pool_ = rte_pktmbuf_pool_create("mbuf_pool", n, cache_size, 0, data_room_size, rte_socket_id());
    if (mbuf_pool_ == nullptr) {
        throw std::runtime_error(fmt::format("Cannot create mbuf pool, n={}, cache_size={}, priv_size=0, data_room_size={}",
                                             n, cache_size, data_room_size));
    }

}

void DPDKRuntime::stop() {
    rte_mempool_free(mbuf_pool_);
    rte_eal_cleanup();
}

#endif

} // lng
