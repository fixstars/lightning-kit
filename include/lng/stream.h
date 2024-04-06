#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>
#include <vector>

#include "concurrentqueue/blockingconcurrentqueue.h"

struct rte_mbuf;
struct rte_mempool;

namespace lng {

class DPDKRuntime;

class Stream {
public:
    virtual void start() = 0;
    virtual void stop() = 0;
};

template<typename T>
class Queueable {
 public:
    virtual void put(T v) = 0;
    virtual bool get(T* vp) = 0;
};

template <typename T>
class MemoryStream : public Stream, public Queueable<T> {

    struct Impl {
#ifdef BLOCKINGQUEUE
        moodycamel::BlockingConcurrentQueue<T> queue;
#else
        moodycamel::ConcurrentQueue<T> queue;
#endif
    };

public:
    MemoryStream()
        : impl_(new Impl)
    {
    }

    virtual void start() {}
    virtual void stop() {}

    virtual void put(T v)
    {
        impl_->queue.enqueue(v);
    }

    virtual bool get(T* vp)
    {
#ifdef BLOCKINGQUEUE
        T res;
        impl_->queue.wait_dequeue(res);
        *vp = res;
        return true;
#else
        return impl_->queue.try_dequeue(*vp);
#endif
    }

private:
    std::shared_ptr<Impl> impl_;
};

#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)

class DPDKStream : public Stream, public Queueable<rte_mbuf*> {

    struct Impl {
        std::shared_ptr<DPDKRuntime> rt;
        uint16_t port_id;
        uint16_t tcp_port;
        bool send_ack(rte_mbuf* recv_mbuf, size_t length);
        bool send_synack(rte_mbuf* tar);
        void wait_for_3wayhandshake();
        bool check_target_packet(rte_mbuf* recv_mbuf);

        Impl(const std::shared_ptr<DPDKRuntime>& rt, uint16_t port_id)
            : rt(rt), port_id(port_id)
        {}

    private:
        bool send_flag_packet(rte_mbuf* tar, size_t length, uint8_t tcp_flags);
    };

public:
    DPDKStream(const std::shared_ptr<DPDKRuntime>& rt, uint16_t port_id)
        : impl_(new Impl(rt, port_id))
    {
    }

    virtual void start();
    virtual void stop();

    virtual void put(rte_mbuf* v);

    virtual bool get(rte_mbuf** vp);

    bool send_ack(rte_mbuf* recv_mbuf, size_t length)
    {
        return impl_->send_ack(recv_mbuf, length);
    }
    bool check_target_packet(rte_mbuf* recv_mbuf)
    {
        return impl_->check_target_packet(recv_mbuf);
    }

private:
    std::shared_ptr<Impl> impl_;
};

#endif

struct Segment {
    uint8_t* payload;
    uint16_t length;
};

struct Payloads {
    static constexpr size_t max_payloads = 10;
    rte_mbuf* buf = nullptr;
    uint8_t no_of_payload = 0;
    Segment segments[max_payloads];
    void Clear();
    size_t ExtractPayloads(rte_mbuf* mbuf);
};

struct Frame {
    static constexpr size_t frame_size = 64 * 1024 * 1024;
    size_t frame_id;
    uint8_t body[frame_size];
};

} // lng

#endif
