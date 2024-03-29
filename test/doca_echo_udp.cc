#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <signal.h>

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
    Receiver(const std::string& id, Stream<uint8_t*>* is, Stream<uint8_t*>* os)
        : Actor(id)
        , inner_stream_(is)
        , outer_stream_(os)
    {
    }

protected:
    virtual void main() override
    {
        uint8_t* v[1000];
        if (size_t num = outer_stream_->get((uint8_t**)(v), 1000)) {
            // std::cout << "received " << num << " " << ((float*)v[0]) << " packets" << std::endl;
            inner_stream_->put(v, num);
        }
    }

private:
    Stream<uint8_t*>* inner_stream_;
    Stream<uint8_t*>* outer_stream_;
};

class Sender : public Actor {
public:
    Sender(const std::string& id, Stream<uint8_t*>* is, Stream<uint8_t*>* os)
        : Actor(id)
        , inner_stream_(is)
        , outer_stream_(os)
    {
    }

protected:
    virtual void main() override
    {
        uint8_t* v[1000];
        if (size_t num = inner_stream_->get((uint8_t**)(v), 1000)) {
            // std::cout << "send " << num << " " << ((float*)v[0]) << " packets" << std::endl;
            outer_stream_->put(v, num);
        }
    }

private:
    Stream<uint8_t*>* inner_stream_;
    Stream<uint8_t*>* outer_stream_;
};

//                 +----------+                    +--------+
// outer_stream -> | receiver | -> inner_stream -> | sender | -> outer_stream
//                 +----------+                    +--------+
int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        DOCAUDPStream outer_stream("a1:00.1", "81:00.0");
        MemoryStream<uint8_t*> inner_stream;

        auto receiver(sys.create_actor<Receiver>("/receiver",
            &inner_stream,
            &outer_stream));
        auto sender(sys.create_actor<Sender>("/sender",
            &inner_stream,
            &outer_stream));

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
