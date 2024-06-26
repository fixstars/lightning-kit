#include <fstream>
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
        const std::shared_ptr<Queueable<Frame*>>& ready_frame,
        const std::string output_file)
        : Actor(id, cpu_id)
        , valid_stream_(valid_frame)
        , ready_stream_(ready_frame)
        , output_file_(output_file)
    {
    }

protected:
    virtual void main() override
    {
        Frame* v;
        if (valid_stream_->get(&v, 1)) {
            std::cout << "received " << v->frame_id << " frame" << std::endl;
            if (!output_first_file_ && output_file_ != "") {
                std::ofstream ofs_(output_file_.c_str());
                ofs_.write(reinterpret_cast<char*>(v->body), Frame::frame_size);
                output_first_file_ = true;
            }
            ready_stream_->put(&v, 1);
        }
    }

private:
    std::shared_ptr<Queueable<Frame*>> valid_stream_;
    std::shared_ptr<Queueable<Frame*>> ready_stream_;
    bool output_first_file_ = false;
    std::string output_file_;
};

//                 +----------+                    +--------+
// outer_stream -> | receiver | -> inner_stream -> | sender | -> outer_stream
//                 +----------+                    +--------+
int main()
{
    try {
        signal(SIGINT, handler_sigint);

        System sys;

        auto dpdk_stream = sys.create_stream<DPDKStream>(2);
        auto valid_frame_stream = sys.create_stream<MemoryStream<Frame*>>();
        auto ready_frame_stream = sys.create_stream<MemoryStream<Frame*>>();
        auto valid_payload_stream = sys.create_stream<MemoryStream<Payload*>>();
        auto ready_payload_stream = sys.create_stream<MemoryStream<Payload*>>();

        const int num_pays = (5 * 1024);
        std::unique_ptr<Payload> pays[num_pays];
        for (int i = 0; i < num_pays; ++i) {
            pays[i].reset(new Payload);
            auto p = pays[i].get();
            ready_payload_stream->put(&p, 1);
        }
        const int num_frames = 8;
        std::unique_ptr<Frame> frames[num_frames];
        for (int i = 0; i < num_frames; ++i) {
            frames[i].reset(new Frame);
            auto p = frames[i].get();
            ready_frame_stream->put(&p, 1);
        }

        auto receiver(sys.create_actor<Receiver>("/frame/build/eth", 16,
            dpdk_stream,
            valid_payload_stream,
            ready_payload_stream));
        auto frame_builder(sys.create_actor<FrameBuilder>("/frame/build", 17,
            valid_payload_stream,
            ready_payload_stream,
            valid_frame_stream,
            ready_frame_stream));
        auto frame_receiver(sys.create_actor<FrameReceiver>("/frame", 18,
            valid_frame_stream,
            ready_frame_stream,
            ""));

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
