#include "lng/system.h"

#if defined(LNG_WITH_DOCA)
#include "lng/doca-util.h"
#endif

#if defined(LNG_WITH_DPDK)
#include "dpdk-runtime.h"
#endif

#include "log.h"

namespace lng {

struct System::Impl {

    std::unordered_map<std::string, std::shared_ptr<Actor>> actors;

#if defined(LNG_WITH_DPDK)
    DPDKRuntime dpdk_rt;
#endif
};

System::System()
    : impl_(new Impl)
{
    log::debug("System is initialized");

#if defined(LNG_WITH_DOCA)
    doca_log_backend_create_standard();
#endif
}

void System::start()
{
    // TODO: Considering tree dependency
    for (auto& [id, actor] : impl_->actors) {
        actor->start();
    }

    // Make sure to be running all actors
    for (auto& [n, actor] : impl_->actors) {
        actor->wait_until(Actor::State::Running);
    }
}

void System::stop()
{
    // TODO: Considering tree dependency
    for (auto& [id, actor] : impl_->actors) {
        actor->stop();
    }

    // Make sure to be ready all actors
    for (auto& [n, actor] : impl_->actors) {
        actor->wait_until(Actor::State::Ready);
    }
}

void System::terminate()
{
    // TODO: Considering tree dependency
    for (auto& [n, actor] : impl_->actors) {
        actor->terminate();
    }

    // Make sure to finalize all actors
    for (auto& [n, actor] : impl_->actors) {
        actor->wait_until(Actor::State::Fin);
    }
}

void System::register_actor(const std::string& id, const std::shared_ptr<Actor>& actor) {
    impl_->actors[id] = actor;
}

} // lng
