#ifndef LNG_SYSTEM_H
#define LNG_SYSTEM_H

#include <string>
#include <type_traits>
#include <unordered_map>

#include "lng/actor.h"

namespace lng {

class System {
public:
    System();

    template <typename T, typename... Args,
        typename std::enable_if<std::is_base_of<Actor, T>::value>::type* = nullptr>
    T create_actor(const std::string id, int cpu_id, Args... args)
    {
        auto actor(std::make_shared<T>(id, cpu_id, args...));
        actors_[id] = actor;
        return *actor;
    }

    void start()
    {
        // TODO: Considering tree dependency
        for (auto& [id, actor] : actors_) {
            actor->start();
        }

        // Make sure to be running all actors
        for (auto& [n, actor] : actors_) {
            actor->wait_until(Actor::State::Running);
        }
    }

    void stop()
    {
        // TODO: Considering tree dependency
        for (auto& [id, actor] : actors_) {
            actor->stop();
        }

        // Make sure to be ready all actors
        for (auto& [n, actor] : actors_) {
            actor->wait_until(Actor::State::Ready);
        }
    }

    void terminate()
    {
        // TODO: Considering tree dependency
        for (auto& [n, actor] : actors_) {
            actor->terminate();
        }

        // Make sure to finalize all actors
        for (auto& [n, actor] : actors_) {
            actor->wait_until(Actor::State::Fin);
        }
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Actor>> actors_;
};

}

#endif
