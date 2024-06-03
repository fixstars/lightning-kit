
#include "actor.h"

#include "stream.h"

namespace lng {

class Receiver : public Actor {
public:
    Receiver(const std::string& id,
        int cpu_id,
        const std::shared_ptr<DPDKStream>& dpdk_st,
        const std::shared_ptr<Queueable<Payload*>>& valid,
        const std::shared_ptr<Queueable<Payload*>>& ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , valid_payload_stream_(valid)
        , ready_payload_stream_(ready)
        , payload_(nullptr)
        , TIMING()
    {
    }

protected:
    virtual void setup() override;
    virtual void main() override;

private:
    std::shared_ptr<DPDKStream> nic_stream_;
    std::shared_ptr<Queueable<Payload*>> valid_payload_stream_;
    std::shared_ptr<Queueable<Payload*>> ready_payload_stream_;

    Payload* payload_;

    static constexpr int NUM_RDTSC = 16;
    uint64_t prev_rdtsc_ = 0;

    std::array<uint64_t, NUM_RDTSC> TIMING;
};

class FrameBuilder : public Actor {
public:
    FrameBuilder(const std::string& id,
        int cpu_id,
        const std::shared_ptr<Queueable<Payload*>>& valid_payload,
        const std::shared_ptr<Queueable<Payload*>>& ready_payload,
        const std::shared_ptr<Queueable<Frame*>>& valid_frame,
        const std::shared_ptr<Queueable<Frame*>>& ready_frame)
        : Actor(id, cpu_id)
        , valid_payload_stream_(valid_payload)
        , ready_payload_stream_(ready_payload)
        , valid_frame_stream_(valid_frame)
        , ready_frame_stream_(ready_frame)
        , payload_(nullptr)
        , payload_segment_id_(0)
        , payload_segment_read_offset_(0)
        , frame_(nullptr)
        , frame_id_(0)
        , frame_write_offset_(0)
    {
    }

protected:
    virtual void main() override;

private:
    std::shared_ptr<Queueable<Payload*>> valid_payload_stream_;
    std::shared_ptr<Queueable<Payload*>> ready_payload_stream_;
    std::shared_ptr<Queueable<Frame*>> valid_frame_stream_;
    std::shared_ptr<Queueable<Frame*>> ready_frame_stream_;
#if 0
    Frame* next_frame_;
    size_t frame_id_;
    size_t write_head_;
#else
    Payload* payload_;
    size_t payload_segment_id_;
    size_t payload_segment_read_offset_;
    Frame* frame_;
    size_t frame_id_;
    size_t frame_write_offset_;
#endif
};
}
