#ifndef LNG_ACTOR_H
#define LNG_ACTOR_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <memory>

namespace lng {

class Actor {

    enum class State {
        Initialized,
        Ready,
        Running,
        Terminated,
    };

    struct Impl {
        std::thread th;
        std::mutex mutex;
        std::condion_variable cvar;

        std::string id;
        State state;

        Impl(Actor *obj, const std::string& id)
            : th(entry_point, obj), mutex(), cvar(), id(id), state(State::Initialized)
        {
        }

        ~Impl() {
            if (th.joinable()) {
                th.join();
            }
        }
    };

public:
    Actor(const std::string& id);

    virtual void main() {}

    static void entry_point(Actor *obj);

    void start() {
        std::scoped_lock lock(impl_->mutex);
        impl_->state = State::Running;
    }

    void stop() {
        std::scoped_lock lock(impl_->mutex);
        impl_->state = State::Ready;
    }

private:

    std::shared_ptr<Impl> impl_;
};

}

#endif
