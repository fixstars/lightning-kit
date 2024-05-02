#ifndef LNG_DOCA_COMMON_UTIL_INTERNAL_H
#define LNG_DOCA_COMMON_UTIL_INTERNAL_H

#include "lng/doca-util.h"

namespace lng {

doca_error_t destroy_rx_queue(rx_queue* rxq);

doca_error_t destroy_semaphore(semaphore* sem);

struct doca_flow_port*
init_doca_flow(uint16_t port_id, uint8_t rxq_num, uint64_t offload_flags);

doca_error_t
create_root_pipe(struct doca_flow_pipe** root_pipe, struct doca_flow_pipe_entry** root_entry, struct doca_flow_pipe** rxq_pipe, uint16_t* dst_ports, int rxq_num, struct doca_flow_port* port, doca_flow_l4_type_ext l4_type_ext);

}

#endif
