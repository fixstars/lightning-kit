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
        std::condition_variable cvar;

        std::string id;
        State state;

        Impl(Actor *obj, const std::string& id)
            : th(entry_point, obj), mutex(), cvar(), id(id), state(State::Initialized)
        {
        }

        ~Impl() {
            {
                std::unique_lock lock(mutex);
                state = State::Terminated;
            }
            cvar.notify_all();
            if (th.joinable()) {
                th.join();
            }
        }
    };

public:
    Actor(const std::string& id);

    void start();
    void stop();

protected:
    virtual void main() = 0;

private:

    static void entry_point(Actor *obj);

    std::shared_ptr<Impl> impl_;
};

}

#endif
