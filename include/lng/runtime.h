#ifndef LNG_RUNTIME_H
#define LNG_RUNTIME_H

#if defined(LNG_WITH_DPDK)
struct rte_mempool;
#endif

namespace lng {

class Runtime {
public:
    enum Type {
        DPDK,
        DOCA
    };

    virtual void start() = 0;
    virtual void stop() = 0;
};

#if defined(LNG_WITH_DOCA)
class DOCARuntime : public Runtime {
public:
    DOCARuntime();

    virtual void start() { }

    virtual void stop() { }
};
#endif

#if defined(LNG_WITH_DPDK)

class DPDKRuntime : public Runtime {
public:
    virtual void start();

    virtual void stop();

    rte_mempool* get_mempool() {
        return mbuf_pool_;
    }

private:
    rte_mempool* mbuf_pool_;
};
#endif

} // lng

#endif
