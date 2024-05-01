#if defined(LNG_WITH_DOCA)
#include "lng/doca-util.h"
#endif

#if defined(LNG_WITH_DPDK)
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_gpudev.h>
#include <rte_mbuf.h>
#include <rte_net.h>

#endif

#include "lng/runtime.h"

#include "lng/net-header.h"

#include "log.h"

namespace lng {

#if defined(LNG_WITH_DOCA)
DOCARuntime::DOCARuntime()
{
    doca_log_backend_create_standard();
}
#endif

#if defined(LNG_WITH_DPDK)
void DPDKRuntime::start()
{
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

    log::info("Trying to create mempool");

    mbuf_pool_ = rte_pktmbuf_pool_create("mbuf_pool", n, cache_size, 0, data_room_size, rte_socket_id());
    if (mbuf_pool_ == nullptr) {
        log::error("Failed to create mempool");
        throw std::runtime_error(fmt::format("Cannot create mbuf pool, n={}, cache_size={}, priv_size=0, data_room_size={}",
            n, cache_size, data_room_size));
    }
}

void DPDKRuntime::stop()
{
    rte_mempool_free(mbuf_pool_);
    rte_eal_cleanup();
}

#define CPU_PAGE_SIZE 4096

void DPDKGPURuntime::start()
{
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
    constexpr uint32_t n = 8192 * 16 - 1;
    constexpr uint32_t cache_size = 256;
    constexpr uint32_t data_room_size = RTE_PKTMBUF_HEADROOM + 10 * 1024;

    log::info("Trying to create mempool");

    struct rte_pktmbuf_extmem ext_mem;
    int16_t gpu_dev_id = 0;
    struct rte_eth_dev_info dev_info;

    ext_mem.elt_size = data_room_size; // mbufs_headroom_size;
    ext_mem.buf_len = RTE_ALIGN_CEIL(n * ext_mem.elt_size, GPU_PAGE_SIZE);
    ext_mem.buf_iova = RTE_BAD_IOVA;
    ext_mem.buf_ptr = rte_gpu_mem_alloc(gpu_dev_id, ext_mem.buf_len, CPU_PAGE_SIZE);
    if (ext_mem.buf_ptr == NULL) {
        throw std::runtime_error("rte_gpu_mem_alloc fail");
    }

    if (rte_extmem_register(ext_mem.buf_ptr, ext_mem.buf_len, NULL, ext_mem.buf_iova, GPU_PAGE_SIZE) < 0) {
        throw std::runtime_error("rte_extmem_register fail");
    }
    log::info("{} port_id_", port_id_);

    rte_eth_dev_info_get(port_id_, &dev_info);

    if (rte_dev_dma_map(dev_info.device, ext_mem.buf_ptr, ext_mem.buf_iova, ext_mem.buf_len) < 0) {
        throw std::runtime_error("rte_dev_dma_map fail");
    }
    mbuf_pool_ = rte_pktmbuf_pool_create_extbuf("gpu_mempool", n,
        cache_size, 0, ext_mem.elt_size,
        rte_socket_id(), &ext_mem, 1);
}

void DPDKGPURuntime::stop()
{
    rte_mempool_free(mbuf_pool_);
    rte_eal_cleanup();
}

#endif

} // lng
