
#include "actor.h"

#include "stream.h"

#include <cuda_runtime_api.h>

struct doca_gpu;
struct rte_gpu_comm_flag;
struct rte_gpu_comm_list;

namespace lng {

struct semaphore;

class ReceiverGPU : public Actor {
public:
    ReceiverGPU(const std::string& id,
        int cpu_id,
        const std::shared_ptr<DPDKGPUUDPStream>& dpdk_st,
        const std::shared_ptr<Queueable<Payload*>>& valid,
        const std::shared_ptr<Queueable<Payload*>>& ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , valid_payload_stream_(valid)
        , ready_payload_stream_(ready)
        , comm_list_idx_(0)
        , comm_list_(nullptr)
    {
    }

protected:
    virtual void setup() override;
    virtual void main() override;

private:
    std::shared_ptr<DPDKGPUUDPStream> nic_stream_;
    std::shared_ptr<Queueable<Payload*>> valid_payload_stream_;
    std::shared_ptr<Queueable<Payload*>> ready_payload_stream_;

    std::vector<cudaStream_t> streams_;

    static constexpr int num_entries = 1024;
    int comm_list_idx_;

    struct rte_gpu_comm_list* comm_list_;
    struct doca_gpu* gpu_dev_;
    std::shared_ptr<struct rte_gpu_comm_flag> quit_flag_;

    std::shared_ptr<struct semaphore> sem_fr;

    static constexpr uint32_t FRAME_NUM = 2;
    static constexpr size_t FRAME_SIZE = 512 * 1024 * 1024;
    static constexpr size_t TMP_FRAME_SIZE = 1024 * 1024 * 1024;

    uint8_t* tar_bufs_;
    uint8_t* tmp_buf_;
};
}
