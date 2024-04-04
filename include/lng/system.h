#ifndef LNG_SYSTEM_H
#define LNG_SYSTEM_H

#include <string>
#include <type_traits>
#include <unordered_map>

#include "lng/actor.h"

namespace lng {

class System {

    struct Impl;

public:
    System();
    System(const System&) = delete;
    System(const System&&) = delete;

    template <typename T, typename... Args,
        typename std::enable_if<std::is_base_of<Actor, T>::value>::type* = nullptr>
    T create_actor(const std::string id, int cpu_id, Args... args)
    {
        auto actor(std::make_shared<T>(id, cpu_id, args...));
        register_actor(id, actor);
        return *actor;
    }

    void start();

    void stop();

    void terminate();


private:
    void register_actor(const std::string& id, const std::shared_ptr<Actor>& actor);

    std::shared_ptr<Impl> impl_;
};

}

#endif
