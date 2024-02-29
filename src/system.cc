#include "lng/system.h"
#include "lng/doca-util.h"

namespace lng {

System::System()
{
    doca_log_backend_create_standard();
}

}
