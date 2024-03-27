#ifndef LNG_DOCA_COMMON_UTIL_INTERNAL_H
#define LNG_DOCA_COMMON_UTIL_INTERNAL_H

#include "lng/doca-util.h"

namespace lng{

doca_error_t destroy_rx_queue(rx_queue* rxq);

doca_error_t destroy_semaphore(semaphore* sem);

}

#endif
