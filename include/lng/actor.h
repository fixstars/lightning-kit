#ifndef LNG_ACTOR_H
#define LNG_ACTOR_H

#include <thread>
#include <memory>

namespace lng {

class Actor {
    struct Impl {
        std::thread th;

        ~Impl() {
            if (th.joinable()) {
                th.join();
            }
        }
    };

public:
    Actor();

    virtual void main() {}

    static void entry_point(Actor *obj)
    {
        obj->main();
    }

    void run() {
        impl_->th = std::thread(entry_point, this);
    }

private:

    std::shared_ptr<Impl> impl_;
};

}

#endif
