
#include "actor.h"

#include "stream.h"

namespace lng {

class Receiver : public Actor {
public:
    Receiver(const std::string& id,
        int cpu_id,
        DPDKStream* dpdk_st,
        Stream<Payloads*>* valid,
        Stream<Payloads*>* ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , vaild_payload_stream_(valid)
        , ready_payload_stream_(ready)
    {
    }

protected:
    virtual void main() override;

private:
    DPDKStream* nic_stream_;
    Stream<Payloads*>* vaild_payload_stream_;
    Stream<Payloads*>* ready_payload_stream_;
};

class FrameBuilder : public Actor {
public:
    FrameBuilder(const std::string& id,
        int cpu_id,
        Stream<Payloads*>* valid_payload,
        Stream<Payloads*>* ready_payload,
        Stream<Frame*>* valid_frame,
        Stream<Frame*>* ready_frame)
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
    Stream<Payloads*>* vaild_payload_stream_;
    Stream<Payloads*>* ready_payload_stream_;
    Stream<Frame*>* vaild_frame_stream_;
    Stream<Frame*>* ready_frame_stream_;
    size_t frame_id_;
    size_t write_head_;
    Frame* next_frame_;
};
}
