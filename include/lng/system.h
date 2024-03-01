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

     template<typename T, typename... Args,
              typename std::enable_if<std::is_base_of<Actor, T>::value>::type* = nullptr>
     T create_actor(const std::string id, Args ...args) {
         auto actor(std::make_shared<T>(args...));
         actors_[id] = actor;
         return *actor;
     }

     void run() {
         for (auto& [id, actor] : actors_) {
             actor->run();
         }
     }

 private:
     std::unordered_map<std::string, std::shared_ptr<Actor>> actors_;
};

}

#endif
