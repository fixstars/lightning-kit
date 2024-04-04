#include "lng/system.h"

#include "lng/runtime.h"
#include "log.h"

namespace lng {

struct System::Impl {
    std::unordered_map<Runtime::Type, std::shared_ptr<Runtime>> runtimes;
    std::unordered_map<std::string, std::shared_ptr<Actor>> actors;
};

System::System()
    : impl_(new Impl)
{
    log::debug("System is initialized");

#if defined(LNG_WITH_DOCA)
    impl_->runtimes[Runtime::DOCA] = std::make_shared<DOCARuntime>();
#endif

#if defined(LNG_WITH_DPDK)
    impl_->runtimes[Runtime::DPDK] = std::make_shared<DPDKRuntime>();
#endif

}

void System::start()
{
    for (auto& [_, rt] : impl_->runtimes) {
        rt->start();
    }

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

    for (auto& [_, rt] : impl_->runtimes) {
        rt->stop();
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

std::shared_ptr<Runtime> System::select_runtime(Runtime::Type type) {
    return impl_->runtimes[type];
}

} // lng
