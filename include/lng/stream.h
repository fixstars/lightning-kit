#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>
#include <vector>

#include "concurrentqueue/blockingconcurrentqueue.h"

struct rte_mbuf;
struct rte_mempool;

struct doca_gpu;
struct doca_dev;
struct doca_flow_port;
struct doca_flow_pipe;
struct doca_flow_pipe_entry;

namespace lng {

struct rx_queue;
struct semaphore;

template <typename T>
class Stream {
public:
    virtual void put(T v) = 0;
    virtual bool get(T* vp) = 0;
};

template <typename T>
class MemoryStream : public Stream<T> {

    struct Impl {
        moodycamel::ConcurrentQueue<T> queue;
    };

public:
    MemoryStream()
        : impl_(new Impl)
    {
    }

    virtual void put(T v)
    {
        impl_->queue.enqueue(v);
    }

    virtual bool get(T* vp)
    {
        return impl_->queue.try_dequeue(*vp);
    }

private:
    std::shared_ptr<Impl> impl_;
};

#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)

class DPDKStream : public Stream<rte_mbuf*> {

    struct Impl {
        rte_mempool* mbuf_pool;
        uint16_t port_id;

        Impl(uint16_t port_id);
        ~Impl();
    };

public:
    DPDKStream(uint16_t port_id)
        : impl_(new Impl(port_id))
    {
    }

    virtual void put(rte_mbuf* v);

    virtual bool get(rte_mbuf** vp);

private:
    std::shared_ptr<Impl> impl_;
};

#endif

#if defined(LNG_WITH_DOCA)

class DOCAStream : public Stream<rte_mbuf*> {

    struct Impl {
        struct doca_gpu* gpu_dev;
        struct doca_dev* ddev;
        struct doca_flow_port* df_port;
        std::unique_ptr<struct rx_queue> rxq;
        std::unique_ptr<struct semaphore> sem;
        uint16_t port_id;
        struct doca_flow_pipe* rxq_pipe;
        struct doca_flow_pipe* root_pipe;
        struct doca_flow_pipe_entry* root_udp_entry;

        Impl(std::string nic_addr, std::string gpu_addr);
        ~Impl();
    };

public:
    DOCAStream(std::string nic_addr, std::string gpu_addr)
        : impl_(new Impl(nic_addr, gpu_addr))
    {
    }

    virtual void put(rte_mbuf* v);

    virtual bool get(rte_mbuf** vp);

private:
    std::shared_ptr<Impl> impl_;
};

#endif

} // lng

#endif
