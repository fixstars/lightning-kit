#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <signal.h>

#include <rte_mbuf_core.h>

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
    Receiver(const std::string& id, Stream<rte_mbuf*> s)
        : Actor(id), stream_(s)
    {}

protected:
    virtual void main() override {
        // TODO: Implement receiving
        //
        stream_.put(nullptr);
    }

private:
    Stream<rte_mbuf *> stream_;
};

class Sender : public Actor {
public:
    Sender(const std::string& id, Stream<rte_mbuf *> s)
        : Actor(id), stream_(s)
    {}

protected:
    virtual void main() override {
        auto buf = stream_.get();
        // TODO: Implement sending
    }

private:
    Stream<rte_mbuf *> stream_;
};

int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        Stream<rte_mbuf *> stream;

        auto receiver(sys.create_actor<Receiver>("/receiver", stream));
        auto sender(sys.create_actor<Sender>("/sender", stream));

        sys.start();

        while (!done) {
            std::this_thread::sleep_for(10ms);
        }

        sys.stop();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
