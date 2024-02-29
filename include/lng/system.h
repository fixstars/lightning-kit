#ifndef LNG_SYSTEM_H
#define LNG_SYSTEM_H

#include <unordered_map>

#include "lng/actor.h"

namespace lng {

class System {
 public:
     System();

     template<typename T>
     Actor create_actor(const std::string id) {
     }

     void run() {
     }

 private:
     std::unordered_map<const std::string&, Actor> actors_;
};

}

#endif
