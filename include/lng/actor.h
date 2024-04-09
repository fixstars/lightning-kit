#ifndef LNG_ACTOR_H
#define LNG_ACTOR_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace lng {

class Actor {
public:
    //
    // -> : Self transition
    // => : Enforced transition
    //
    //           +-------------------------------+
    //           |                               |
    //           v                               |
    // Init -> Ready => Started -> Running => Stopped
    //           |
    //           +=> Terminated -> Fin
    enum class State {
        Init = 0,
        Ready,
        Started,
        Running,
        Stopped,
        Terminated,
        Fin,
    };

private:
    struct Impl {
        std::thread th;
        std::mutex mutex;
        std::condition_variable cvar;

        std::string id;
        State state;

        int cpu_id;

        Impl(Actor* obj, const std::string& id, int cpu)
            : th(entry_point, obj)
            , mutex()
            , cvar()
            , id(id)
            , state(State::Init)
            , cpu_id(cpu)
        {
        }

        ~Impl()
        {
            if (th.joinable()) {
                th.join();
            }
        }
    };

public:
    Actor(const std::string& id, int cpu_id);

    void start();
    void stop();
    void terminate();
    void wait_until(State to);

protected:
    virtual void main() = 0;

private:
    static void entry_point(Actor* obj);
    void transit(State from, State to);

    std::shared_ptr<Impl> impl_;
};

}

#endif
