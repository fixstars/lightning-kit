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
    Receiver(const std::string& id, int cpu_id, const std::shared_ptr<Queueable<uint8_t*>>& is, const std::shared_ptr<Queueable<uint8_t*>>& os)
        : Actor(id, cpu_id)
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
    std::shared_ptr<Queueable<uint8_t*>> inner_stream_;
    std::shared_ptr<Queueable<uint8_t*>> outer_stream_;
};

class Sender : public Actor {
public:
    Sender(const std::string& id, int cpu_id, const std::shared_ptr<Queueable<uint8_t*>>& is, const std::shared_ptr<Queueable<uint8_t*>>& os)
        : Actor(id, cpu_id)
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
    std::shared_ptr<Queueable<uint8_t*>> inner_stream_;
    std::shared_ptr<Queueable<uint8_t*>> outer_stream_;
};

//                 +----------+                    +--------+
// outer_stream -> | receiver | -> inner_stream -> | sender | -> outer_stream
//                 +----------+                    +--------+
int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        auto outer_stream(sys.create_stream<DOCATCPStream>("c1:00.0", "81:00.0"));
        auto inner_stream(sys.create_stream<MemoryStream<uint8_t*>>());

        auto receiver(sys.create_actor<Receiver>("/receiver", 20,
            inner_stream,
            outer_stream));
        auto sender(sys.create_actor<Sender>("/sender", 21,
            inner_stream,
            outer_stream));

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
