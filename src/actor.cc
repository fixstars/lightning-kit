#include "lng/actor.h"

#include "log.h"

namespace lng {

namespace {
    std::string to_string(const Actor::State state)
    {
        const char* table[] = {
            "Init",
            "Ready",
            "Started",
            "Running",
            "Stopped",
            "Terminated",
            "Fin"
        };
        return table[static_cast<int>(state)];
    }
} // anonymous

Actor::Actor(const std::string& id, int cpu_id)
    : impl_(new Impl(this, id, cpu_id))
{
    log::debug("{} is initialized", impl_->id);
}

void Actor::start()
{
    if (impl_->state == State::Init) {
        impl_->th = std::move(std::thread(entry_point, this));
        wait_until(State::Ready);
    }

    transit(State::Ready, State::Started);
}

void Actor::stop()
{
    transit(State::Running, State::Stopped);
}

void Actor::terminate()
{
    transit(State::Ready, State::Terminated);
}

void Actor::wait_until(State to)
{
    std::unique_lock lock(impl_->mutex);
    log::debug("{} wait until : {} -> {}", impl_->id, to_string(impl_->state), to_string(to));
    impl_->cvar.wait(lock, [&] { return impl_->state == to; });
}

static inline void set_affinity(int cpu_id)
{
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(cpu_id, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

void Actor::entry_point(Actor* obj)
{
    set_affinity(obj->impl_->cpu_id);
    
    obj->transit(State::Init, State::Ready);
    
    while (true) {
        try {

            State from;
            State to;
            {
                std::unique_lock lock(obj->impl_->mutex);
                obj->impl_->cvar.wait(lock, [&] { return obj->impl_->state == State::Running | obj->impl_->state == State::Started | obj->impl_->state == State::Stopped | obj->impl_->state == State::Terminated; });
                from = obj->impl_->state;
                if (from == State::Running) {
                    to = from;
                } else if (from == State::Started) {
                    to = State::Running;
                } else if (from == State::Stopped) {
                    to = State::Ready;
                } else if (from == State::Terminated) {
                    to = State::Fin;
                }
                obj->impl_->state = to;
            }

            if (to != from) {
                log::debug("{} transit from {} to {}", obj->impl_->id, to_string(from), to_string(to));
                obj->impl_->cvar.notify_all();
            }

            if (from == State::Started && to == State::Running) {
                obj->setup();
            }

            if (to == State::Running) {
                obj->main();
            } else if (to == State::Fin) {
                return;
            }

            if (from == State::Stopped && to == State::Ready) {
                obj->teardown();
            }

        } catch (const std::exception& e) {
            log::error("Exception on {} : {}", obj->impl_->id, e.what());
        } catch (...) {
            log::error("Unknwon error on {}", obj->impl_->id);
        }
    }
}

void Actor::transit(State from, State to)
{
    {
        std::unique_lock lock(impl_->mutex);
        impl_->cvar.wait(lock, [&] { return impl_->state == from; });
        impl_->state = to;
        log::debug("{} transit from {} to {}", impl_->id, to_string(from), to_string(to));
    }
    impl_->cvar.notify_all();
}

}
