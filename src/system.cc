#include "lng/system.h"

#if defined(LNG_WITH_NV)
#include "lng/doca-util.h"
#endif

namespace lng {

System::System()
{
#if defined(LNG_WITH_NV)
    doca_log_backend_create_standard();
#endif
}

}
