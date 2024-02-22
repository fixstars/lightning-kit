#include "lng/actor.h"
#include "lng/doca-util.h"

namespace lng {

Actor::Actor()
{
    doca_log_backend_create_standard();
}

}
