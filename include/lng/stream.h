#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>
#include <vector>

#include "concurrentqueue/blockingconcurrentqueue.h"

struct rte_mbuf;
struct rte_mempool;

namespace lng {

template<typename T>
class Stream {
public:
    virtual void put(T v) = 0;
    virtual bool get(T *vp) = 0;
};


template<typename T>
class MemoryStream {

    struct Impl {
        moodycamel::ConcurrentQueue<T> queue;
    };

public:
    MemoryStream()
        : impl_(new Impl)
    {}

    virtual void put(T v) {
        impl_->queue.enqueue(v);
    }

    virtual bool get(T *vp) {
        return impl_->queue.try_dequeue(*vp);
    }

private:
    std::shared_ptr<Impl> impl_;
};


#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)

class DPDKStream : public Stream<rte_mbuf*> {

    struct Impl {
        rte_mempool *mbuf_pool;
        uint16_t port_id;

        rte_ether_addr self_eth_addr;
        rte_ether_addr peer_eth_addr;

        Impl(uint16_t port_id);
        ~Impl();
    };

public:
    static from_eth_addr(const std::string& eth_addr);
    static from_ip_addr(const std::string& ip_addr);

    DPDKStream(uint16_t port_id) : impl_(new Impl(port_id)) { }

    virtual void put(rte_mbuf *v);

    virtual bool get(rte_mbuf **vp);

private:
    std::shared_ptr<Impl> impl_;
};

#endif

} // lng

#endif
