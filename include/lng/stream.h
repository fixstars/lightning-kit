#ifndef LNG_STREAM_H
#define LNG_STREAM_H

#include <memory>

#include "concurrentqueue/blockingconcurrentqueue.h"

namespace lng {

template<typename T>
class Stream {

    struct Impl {
        moodycamel::BlockingConcurrentQueue<T> queue;
    };

public:
    Stream()
        : impl_(new Impl)
    {}

    void put(T v) {
        impl_->queue.enqueue(v);
    }

    T get() {
        T v;
        impl_->queue.wait_dequeue(v);
        return v;
    }

private:
    std::shared_ptr<Impl> impl_;
};

}

#endif
