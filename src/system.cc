#include "lng/system.h"

#if defined(LNG_WITH_NV)
#include "lng/doca-util.h"
#endif

#include "log.h"

namespace lng {

System::System()
{
    log::debug("System is initialized");

#if defined(LNG_WITH_NV)
    doca_log_backend_create_standard();
#endif
}

} // lng
