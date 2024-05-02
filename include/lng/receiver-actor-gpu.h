
#include "actor.h"

#include "stream.h"

#include <cuda_runtime_api.h>

struct doca_gpu;
struct rte_gpu_comm_flag;
struct rte_gpu_comm_list;

namespace lng {

struct semaphore;

class ReceiverGPUUDP : public Actor {
public:
    ReceiverGPUUDP(const std::string& id,
        int cpu_id,
        const std::shared_ptr<DPDKGPUUDPStream>& dpdk_st,
        const std::shared_ptr<Queueable<Payload*>>& valid,
        const std::shared_ptr<Queueable<Payload*>>& ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , valid_payload_stream_(valid)
        , ready_payload_stream_(ready)
        , comm_list_idx_(0)
        , comm_list_free_idx_(0)
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
    size_t comm_list_idx_;
    size_t comm_list_free_idx_;
    std::vector<std::shared_ptr<rte_mbuf*>> mbufs;
    std::vector<size_t> mbufs_num;

    struct rte_gpu_comm_list* comm_list_;
    struct doca_gpu* gpu_dev_;
    std::shared_ptr<struct rte_gpu_comm_flag> quit_flag_;

    // std::shared_ptr<struct semaphore> sem_fr;

    static constexpr uint32_t FRAME_NUM = 2;
    static constexpr size_t FRAME_SIZE = 512 * 1024 * 1024;
    static constexpr size_t TMP_FRAME_SIZE = 1024 * 1024 * 1024;

    uint8_t* tar_bufs_;
    uint8_t* tmp_buf_;
};

class ReceiverGPUTCP : public Actor {
public:
    ReceiverGPUTCP(const std::string& id,
        int cpu_id,
        const std::shared_ptr<DPDKGPUTCPStream>& dpdk_st,
        const std::shared_ptr<Queueable<Payload*>>& valid,
        const std::shared_ptr<Queueable<Payload*>>& ready)
        : Actor(id, cpu_id)
        , nic_stream_(dpdk_st)
        , valid_payload_stream_(valid)
        , ready_payload_stream_(ready)
        , comm_list_idx_(0)
        , comm_list_free_idx_(0)
        , comm_list_frame_idx_(0)
        , comm_list_(nullptr)
        , comm_list_ack_ref_(nullptr)
        , comm_list_ack_pkt_(nullptr)
        , comm_list_notify_frame_(nullptr)
    {
    }

protected:
    virtual void setup() override;
    virtual void main() override;

private:
    void wait_3wayhandshake();

    std::shared_ptr<DPDKGPUTCPStream> nic_stream_;
    std::shared_ptr<Queueable<Payload*>> valid_payload_stream_;
    std::shared_ptr<Queueable<Payload*>> ready_payload_stream_;

    std::vector<cudaStream_t> streams_;
    cudaStream_t stream_cpy_;

    static constexpr int num_entries = 1024;
    static constexpr int num_ack_entries = 1024;
    size_t comm_list_idx_;
    size_t comm_list_free_idx_;
    size_t comm_list_frame_idx_;
    std::vector<std::shared_ptr<rte_mbuf*>> mbufs;
    std::vector<size_t> mbufs_num;
    std::vector<rte_mbuf*> ack_tmp_mbufs_;

    std::shared_ptr<std::thread> ack_thread;

    struct rte_gpu_comm_list* comm_list_;
    struct rte_gpu_comm_list* comm_list_ack_ref_;
    struct rte_gpu_comm_list* comm_list_ack_pkt_;
    struct rte_gpu_comm_list* comm_list_notify_frame_;
    struct doca_gpu* gpu_dev_;
    std::shared_ptr<struct rte_gpu_comm_flag> quit_flag_;

    // std::shared_ptr<struct semaphore> sem_fr;
    static constexpr bool OUTPUT_TO_FILE = false;

    static constexpr uint32_t FRAME_NUM = 2;
    static constexpr size_t FRAME_SIZE = 512 * 1024 * 1024;
    static constexpr size_t TMP_FRAME_SIZE = 1024 * 1024 * 1024;

    uint8_t* tar_bufs_;
    uint8_t* tar_bufs_cpu_;
    uint8_t* tmp_buf_;
    uint32_t* seqn_;
};
}
