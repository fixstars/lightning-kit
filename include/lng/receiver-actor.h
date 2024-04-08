
#include "actor.h"

#include "stream.h"

namespace lng {

class Receiver : public Actor {
public:
    Receiver(const std::string& id,
        int cpu_id,
        const std::shared_ptr<DPDKStream>& dpdk_st,
        const std::shared_ptr<Queueable<Payloads*>>& valid,
        const std::shared_ptr<Queueable<Payloads*>>& ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , vaild_payload_stream_(valid)
        , ready_payload_stream_(ready)
    {
    }

protected:
    virtual void setup() override;
    virtual void main() override;

private:
    std::shared_ptr<DPDKStream> nic_stream_;
    std::shared_ptr<Queueable<Payloads*>> vaild_payload_stream_;
    std::shared_ptr<Queueable<Payloads*>> ready_payload_stream_;
};

class FrameBuilder : public Actor {
public:
    FrameBuilder(const std::string& id,
        int cpu_id,
        const std::shared_ptr<Queueable<Payloads*>>& valid_payload,
        const std::shared_ptr<Queueable<Payloads*>>& ready_payload,
        const std::shared_ptr<Queueable<Frame*>>& valid_frame,
        const std::shared_ptr<Queueable<Frame*>>& ready_frame)
        : Actor(id, cpu_id)
        , vaild_payload_stream_(valid_payload)
        , ready_payload_stream_(ready_payload)
        , vaild_frame_stream_(valid_frame)
        , ready_frame_stream_(ready_frame)
        , frame_id_(0)
        , write_head_(0)
        , next_frame_(nullptr)
    {
    }

protected:
    virtual void main() override;

private:
    std::shared_ptr<Queueable<Payloads*>> vaild_payload_stream_;
    std::shared_ptr<Queueable<Payloads*>> ready_payload_stream_;
    std::shared_ptr<Queueable<Frame*>> vaild_frame_stream_;
    std::shared_ptr<Queueable<Frame*>> ready_frame_stream_;
    size_t frame_id_;
    size_t write_head_;
    Frame* next_frame_;
};
}
