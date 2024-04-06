#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <signal.h>

#include <rte_mbuf.h>

#include "lng/lng.h"

using namespace std::chrono_literals;
using namespace lng;

bool done = false;
void handler_sigint(int sig)
{
    done = true;
}

class Receiver : public Actor {
public:
    Receiver(const std::string& id, int cpu_id, const std::shared_ptr<Queueable<rte_mbuf*>>& is, const std::shared_ptr<Queueable<rte_mbuf*>>& os)
        : Actor(id, cpu_id)
        , inner_stream_(is)
        , outer_stream_(os)
    {
    }

protected:
    virtual void main() override
    {
        rte_mbuf* v;
        if (outer_stream_->get(&v)) {
            std::cout << "received " << v->pkt_len << " bytes" << std::endl;
            inner_stream_->put(v);
        }
    }

private:
    std::shared_ptr<Queueable<rte_mbuf*>> inner_stream_;
    std::shared_ptr<Queueable<rte_mbuf*>> outer_stream_;
};

class Sender : public Actor {
public:
    Sender(const std::string& id, int cpu_id, const std::shared_ptr<Queueable<rte_mbuf*>>& is, const std::shared_ptr<Queueable<rte_mbuf*>>& os)
        : Actor(id, cpu_id)
        , inner_stream_(is)
        , outer_stream_(os)
    {
    }

protected:
    virtual void main() override
    {
        rte_mbuf* v;
        if (inner_stream_->get(&v)) {
            std::cout << "send " << v->pkt_len << " bytes" << std::endl;
            outer_stream_->put(v);
        }
    }

private:
    std::shared_ptr<Queueable<rte_mbuf*>> inner_stream_;
    std::shared_ptr<Queueable<rte_mbuf*>> outer_stream_;
};

//                 +----------+                    +--------+
// outer_stream -> | receiver | -> inner_stream -> | sender | -> outer_stream
//                 +----------+                    +--------+
int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        auto outer_stream(sys.create_stream<DPDKStream>(2));
        auto inner_stream(sys.create_stream<MemoryStream<rte_mbuf*>>());

        auto receiver(sys.create_actor<Receiver>("/receiver", 4, inner_stream, outer_stream));
        auto sender(sys.create_actor<Sender>("/sender", 5, inner_stream, outer_stream));

        sys.start();

        while (!done) {
            std::this_thread::sleep_for(10ms);
        }

        sys.stop();

        sys.terminate();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
