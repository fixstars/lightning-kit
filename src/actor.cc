#include "lng/actor.h"

#include "log.h"

namespace lng {

Actor::Actor(const std::string& id)
    : impl_(new Impl(this, id))
{
}

void Actor::start() {
    {
        std::unique_lock lock(impl_->mutex);
        impl_->cvar.wait(lock, [&] { return impl_->state == State::Ready; });
        impl_->state = State::Running;
        log::debug("Actor {} is started", impl_->id);
    }
    impl_->cvar.notify_all();
}

void Actor::stop() {
    {
        std::unique_lock lock(impl_->mutex);
        impl_->cvar.wait(lock, [&] { return impl_->state == State::Running; });
        impl_->state = State::Ready;
        log::debug("Actor {} is stopped", impl_->id);
    }
    impl_->cvar.notify_all();
}

void Actor::entry_point(Actor *obj)
{
    log::debug("Actor {} is initialized", obj->impl_->id);
    {
        std::unique_lock lock(obj->impl_->mutex);
        obj->impl_->state = State::Ready;
    }
    obj->impl_->cvar.notify_all();

    while (true) {
        try {

            State state;
            {
                std::unique_lock lock(obj->impl_->mutex);
                obj->impl_->cvar.wait(lock, [&] { return obj->impl_->state == State::Running |
                                                         obj->impl_->state == State::Terminated; });
                state = obj->impl_->state;
            }

            if (state == State::Running) {
                obj->main();
            } else if (state == State::Terminated) {
                break;
            }
        } catch (const std::exception& e) {
            log::error("Exception on {} : {}", obj->impl_->id, e.what());
        } catch (...) {
            log::error("Unknwon error on {}", obj->impl_->id);
        }
    }
    log::debug("Actor {} is ended", obj->impl_->id);
}


}
