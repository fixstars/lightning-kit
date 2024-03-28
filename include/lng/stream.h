#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>
#include <vector>

#include "concurrentqueue/blockingconcurrentqueue.h"

struct rte_mbuf;
struct rte_mempool;

namespace lng {

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

} // lng

#endif
