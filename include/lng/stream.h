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
struct tx_queue;
struct semaphore;

template <typename T>
class Stream {
public:
    virtual bool put(T* v, size_t count) = 0;
    virtual size_t get(T* vp, size_t max) = 0;
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

    virtual bool put(T* v, size_t count)
    {
        return impl_->queue.enqueue_bulk(v, count);
    }

    virtual size_t get(T* vp, size_t max)
    {
        return impl_->queue.try_dequeue_bulk(vp, max);
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

    virtual bool put(rte_mbuf** v, size_t count);

    virtual size_t get(rte_mbuf** vp, size_t max);

private:
    std::shared_ptr<Impl> impl_;
};

#endif

#if defined(LNG_WITH_DOCA)

class DOCAStream : public Stream<uint8_t*> {

    struct Impl {
        struct doca_gpu* gpu_dev;
        struct doca_dev* ddev;
        struct doca_flow_port* df_port;
        std::unique_ptr<struct rx_queue> rxq;
        std::unique_ptr<struct tx_queue> txq;
        std::unique_ptr<struct semaphore> sem_rx;
        std::unique_ptr<struct semaphore> sem_fr;
        std::unique_ptr<struct semaphore> sem_reply;
        uint32_t sem_fr_idx;
        uint32_t sem_reply_idx;
        uint16_t port_id;
        struct doca_flow_pipe* rxq_pipe;
        struct doca_flow_pipe* root_pipe;
        struct doca_flow_pipe_entry* root_udp_entry;
        std::unique_ptr<struct tx_buf> tx_buf_arr;

        Impl(std::string nic_addr, std::string gpu_addr);
        ~Impl();
        size_t get(uint8_t** vp, size_t max);
        bool put(uint8_t** v, size_t count);
    };

public:
    DOCAStream(std::string nic_addr, std::string gpu_addr)
        : impl_(new Impl(nic_addr, gpu_addr))
    {
    }

    virtual bool put(uint8_t** v, size_t count)
    {
        return impl_->put(v, count);
    }

    virtual size_t get(uint8_t** vp, size_t max)
    {
        return impl_->get(vp, max);
    }

private:
    std::shared_ptr<Impl> impl_;
};

#endif

} // lng

#endif
