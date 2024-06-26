#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>
#include <vector>

#include "concurrentqueue/concurrentqueue.h"

struct rte_mbuf;
struct rte_mempool;

struct doca_gpu;
struct doca_dev;
struct doca_flow_port;
struct doca_flow_pipe;
struct doca_flow_pipe_entry;

namespace lng {

class DPDKRuntime;
class DPDKGPURuntime;
struct rx_queue;
struct tx_queue;
struct semaphore;

class Stream {
public:
    virtual void start() = 0;
    virtual void stop() = 0;
};

template <typename T>
class Queueable {
public:
    virtual bool put(T* v, size_t count) = 0;
    virtual size_t get(T* vp, size_t max) = 0;
    virtual size_t count() = 0;
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

    virtual void start() { }
    virtual void stop() { }

    virtual bool put(T* v, size_t count)
    {
        return impl_->queue.enqueue_bulk(v, count);
    }

    virtual size_t get(T* vp, size_t max)
    {
        return impl_->queue.try_dequeue_bulk(vp, max);
    }

    virtual size_t count()
    {
        return impl_->queue.size_approx();
    }

private:
    std::shared_ptr<Impl> impl_;
};

#if defined(LNG_WITH_DOCA) || defined(LNG_WITH_DPDK)

class DPDKStream : public Stream, public Queueable<rte_mbuf*> {

public:
    enum PKTType {
        FIN,
        TCP,
        OTHER
    };

private:
    struct Impl {
        std::shared_ptr<DPDKRuntime> rt;
        uint16_t port_id;
        uint16_t tcp_port;
        bool send_ack(rte_mbuf* recv_mbuf, uint32_t length);
        bool send_ack_from_tmp(rte_mbuf* recv_mbuf, uint32_t length);
        bool send_synack(rte_mbuf* tar);
        rte_mbuf* wait_for_3wayhandshake();
        void prepare_ack_tmp_pkt(rte_mbuf* ref);
        PKTType check_target_packet(rte_mbuf* recv_mbuf);

        Impl(const std::shared_ptr<DPDKRuntime>& rt, uint16_t port_id)
            : rt(rt)
            , port_id(port_id)
            , ack_tmp_count(0)
        {
        }

    private:
        bool send_flag_packet(rte_mbuf* tar, uint32_t length, uint8_t tcp_flags);
        bool send_flag_packet_from_tmp(rte_mbuf* tar, uint32_t length, uint8_t tcp_flags);

        static constexpr int ACK_TMP_NUM = 128;
        int ack_tmp_count;
        rte_mbuf* ack_tmp_pkt[ACK_TMP_NUM];
        uint32_t prev_ack = 0;
    };

public:
    DPDKStream(const std::shared_ptr<DPDKRuntime>& rt, uint16_t port_id)
        : impl_(new Impl(rt, port_id))
    {
    }

    virtual void start();
    virtual void stop();

    virtual bool put(rte_mbuf** v, size_t count);

    virtual size_t get(rte_mbuf** vp, size_t max);

    virtual size_t count();

    bool send_ack(rte_mbuf* recv_mbuf, uint32_t length)
    {
        return impl_->send_ack(recv_mbuf, length);
    }

    bool send_ack_from_tmp(rte_mbuf* recv_mbuf, uint32_t length)
    {
        return impl_->send_ack_from_tmp(recv_mbuf, length);
    }

    PKTType check_target_packet(rte_mbuf* recv_mbuf)
    {
        return impl_->check_target_packet(recv_mbuf);
    }

    rte_mbuf* wait_for_3wayhandshake()
    {
        return impl_->wait_for_3wayhandshake();
    }

    void prepare_ack_tmp_pkt(rte_mbuf* ref)
    {
        return impl_->prepare_ack_tmp_pkt(ref);
    }

private:
    std::shared_ptr<Impl> impl_;
};

class DPDKGPUUDPStream : public Stream, public Queueable<rte_mbuf*> {

    struct Impl {
        std::shared_ptr<DPDKGPURuntime> rt;
        uint16_t port_id;
        uint16_t tcp_port;

        Impl(const std::shared_ptr<DPDKGPURuntime>& rt, uint16_t port_id)
            : rt(rt)
            , port_id(port_id)
        {
        }
    };

public:
    DPDKGPUUDPStream(const std::shared_ptr<DPDKGPURuntime>& rt, uint16_t port_id)
        : impl_(new Impl(rt, port_id))
    {
    }

    virtual void start();
    virtual void stop();

    virtual bool put(rte_mbuf** v, size_t count);

    virtual size_t get(rte_mbuf** vp, size_t max);

    virtual size_t count();

private:
    std::shared_ptr<Impl> impl_;
};

class DPDKGPUTCPStream : public Stream, public Queueable<rte_mbuf*> {

    struct Impl {
        std::shared_ptr<DPDKGPURuntime> rt;
        uint16_t port_id;
        uint16_t tcp_port;

        Impl(const std::shared_ptr<DPDKGPURuntime>& rt, uint16_t port_id)
            : rt(rt)
            , port_id(port_id)
        {
        }
    };

public:
    DPDKGPUTCPStream(const std::shared_ptr<DPDKGPURuntime>& rt, uint16_t port_id)
        : impl_(new Impl(rt, port_id))
    {
    }

    virtual void start();
    virtual void stop();

    virtual bool put(rte_mbuf** v, size_t count);

    virtual size_t get(rte_mbuf** vp, size_t max);

    virtual size_t count();

    rte_mbuf* alloc_ack_mbuf();
    rte_mbuf* clone_ack_mbuf(rte_mbuf* tmp);

private:
    std::shared_ptr<Impl> impl_;
};

#endif

#if defined(LNG_WITH_DOCA)

class DOCAUDPEchoStream : public Stream, public Queueable<uint8_t*> {

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
    DOCAUDPEchoStream(std::string nic_addr, std::string gpu_addr)
        : impl_(new Impl(nic_addr, gpu_addr))
    {
    }

    virtual void start()
    { /*TBD*/
    }
    virtual void stop()
    { /*TBD*/
    }

    virtual bool put(uint8_t** v, size_t count)
    {
        return impl_->put(v, count);
    }

    virtual size_t get(uint8_t** vp, size_t max)
    {
        return impl_->get(vp, max);
    }

    virtual size_t count();

private:
    std::shared_ptr<Impl> impl_;
};

class DOCAUDPFrameBuilderStream : public Stream, public Queueable<uint8_t*> {

    struct Impl {
        struct doca_gpu* gpu_dev;
        struct doca_dev* ddev;
        struct doca_flow_port* df_port;
        std::unique_ptr<struct rx_queue> rxq;
        std::unique_ptr<struct tx_queue> txq;
        std::unique_ptr<struct semaphore> sem_rx;
        std::unique_ptr<struct semaphore> sem_fr;
        uint32_t sem_fr_idx;
        uint16_t port_id;
        struct doca_flow_pipe* rxq_pipe;
        struct doca_flow_pipe* root_pipe;
        struct doca_flow_pipe_entry* root_udp_entry;

        uint8_t* tar_buf;
        uint8_t* tmp_buf;
        static constexpr size_t frame_size = (size_t)512 * 1024 * 1024;
        static constexpr size_t frame_num = 2;
        static constexpr size_t tmp_size = (size_t)512 * 1024 * 1024;

        Impl(std::string nic_addr, std::string gpu_addr);
        ~Impl();
        size_t get(uint8_t** vp, size_t max);
        bool put(uint8_t** v, size_t count);
    };

public:
    DOCAUDPFrameBuilderStream(std::string nic_addr, std::string gpu_addr)
        : impl_(new Impl(nic_addr, gpu_addr))
    {
    }

    virtual void start()
    { /*TBD*/
    }
    virtual void stop()
    { /*TBD*/
    }

    virtual bool put(uint8_t** v, size_t count)
    {
        return impl_->put(v, count);
    }

    virtual size_t get(uint8_t** vp, size_t max)
    {
        return impl_->get(vp, max);
    }

    virtual size_t count();

private:
    std::shared_ptr<Impl> impl_;
};

class DOCATCPStream : public Stream, public Queueable<uint8_t*> {
    struct Impl;

public:
    DOCATCPStream(std::string nic_addr, std::string gpu_addr);

    virtual void start()
    { /*TBD*/
    }
    virtual void stop()
    { /*TBD*/
    }

    virtual bool put(uint8_t** v, size_t count);

    virtual size_t get(uint8_t** vp, size_t max);

    virtual size_t count();

private:
    std::shared_ptr<Impl> impl_;
};

#endif
#if 0
struct Segment {
    uint8_t* payload;
    uint16_t length;
};

struct Payload {
    static constexpr size_t max_Payload = 10;
    rte_mbuf* buf = nullptr;
    uint8_t segments_num = 0;
    Segment segments[max_Payload];
    size_t dropped_bytes = 0;
    void Clear();
    uint32_t ExtractPayload(rte_mbuf* mbuf);
};
#else
struct Segment {
    uint8_t* addr;
    uint16_t size;
};

struct Payload {
    static constexpr size_t segments_max = 10;
    rte_mbuf* buf = nullptr;
    uint8_t segments_num = 0;
    Segment segments[segments_max];
    size_t dropped_bytes = 0;
    void Clear();
    uint32_t ExtractPayload(rte_mbuf* mbuf);
};
#endif
struct Frame {
    static constexpr size_t frame_size = 512 * 1024 * 1024;
    size_t frame_id;
    uint8_t body[frame_size];
};

} // lng

#endif
