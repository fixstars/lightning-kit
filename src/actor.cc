#include "lng/actor.h"

#include "log.h"

namespace lng {

Actor::Actor(const std::string& id)
    : impl_(new Impl(this, id))
{
}

void Actor::entry_point(Actor *obj)
{
    log::debug("Actor {} is initiated", obj->impl_->id);
    while (true) {
        try {
            // TODO: WIP
            // scoped_lock lock;
            // impl_->wait_for
            obj->main();
        } catch (const std::exception& e) {
            log::error("Exception on {} : {}", obj->impl_->id, e.what());
        } catch (...) {
            log::error("Unknwon error on {}", obj->impl_->id);
        }
    }
    log::debug("Actor {} is ended", obj->impl_->id);
}

}
