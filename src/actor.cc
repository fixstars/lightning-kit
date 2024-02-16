#include "lng/doca-util.h"
#include "lng/actor.h"

namespace lng {

Actor::Actor() {
#ifdef DOCA22
    doca_log_create_standard_backend();
#else
    doca_log_backend_create_standard();
#endif
}

}
