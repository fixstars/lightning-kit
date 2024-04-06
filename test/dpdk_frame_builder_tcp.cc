#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <signal.h>

#include <rte_mbuf.h>

#include "lng/lng.h"
#include "lng/receiver-actor.h"

using namespace std::chrono_literals;
using namespace lng;

bool done = false;
void handler_sigint(int sig)
{
    done = true;
}

class FrameReceiver : public Actor {
public:
    FrameReceiver(const std::string& id, int cpu_id,
                  const std::shared_ptr<Queueable<Frame*>>& valid_frame,
                  const std::shared_ptr<Queueable<Frame*>>& ready_frame)
        : Actor(id, cpu_id)
        , valid_stream_(valid_frame)
        , ready_stream_(ready_frame)
    {
    }

protected:
    virtual void main() override
    {
        Frame* v;
        if (valid_stream_->get(&v)) {
            std::cout << "received " << v->frame_id << " frame" << std::endl;
            ready_stream_->put(v);
        }
    }

private:
    std::shared_ptr<Queueable<Frame*>> valid_stream_;
    std::shared_ptr<Queueable<Frame*>> ready_stream_;
};

//                 +----------+                    +--------+
// outer_stream -> | receiver | -> inner_stream -> | sender | -> outer_stream
//                 +----------+                    +--------+
int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        auto dpdk_stream = sys.create_stream<DPDKStream>(1);
        auto valid_frame_stream = sys.create_stream<MemoryStream<Frame*>>();
        auto ready_frame_stream = sys.create_stream<MemoryStream<Frame*>>();
        auto valid_payload_stream = sys.create_stream<MemoryStream<Payloads*>>();
        auto ready_payload_stream = sys.create_stream<MemoryStream<Payloads*>>();

        const int num_pays = 1024;
        std::unique_ptr<Payloads> pays[num_pays];
        for (int i = 0; i < num_pays; ++i) {
            pays[i].reset(new Payloads);
            ready_payload_stream->put(pays[i].get());
        }
        const int num_frames = 8;
        std::unique_ptr<Frame> frames[num_frames];
        for (int i = 0; i < num_frames; ++i) {
            frames[i].reset(new Frame);
            ready_frame_stream->put(frames[i].get());
        }

        auto receiver(sys.create_actor<Receiver>("/frame/build/eth", 4,
            dpdk_stream,
            valid_payload_stream,
            ready_payload_stream));
        auto frame_builder(sys.create_actor<FrameBuilder>("/frame/build", 5,
            valid_payload_stream,
            ready_payload_stream,
            valid_frame_stream,
            ready_frame_stream));
        auto frame_receiver(sys.create_actor<FrameReceiver>("/frame", 6,
            valid_frame_stream,
            ready_frame_stream));

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
