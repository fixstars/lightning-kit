#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "lng/lng.h"

using namespace std::chrono_literals;
using namespace lng;

template <typename T>
class Producer : public Actor {
public:
    Producer(const std::string& id, int cpu_id, Stream<T>* s)
        : Actor(id, cpu_id)
        , stream_(s)
        , v_(0)
    {
    }

protected:
    virtual void main() override
    {
        stream_->put(&v_, 1);
        v_++;
    }

private:
    Stream<T>* stream_;
    int v_;
};

template <typename T>
class Consumer : public Actor {
public:
    Consumer(const std::string& id, int cpu_id, Stream<T>* s)
        : Actor(id, cpu_id)
        , stream_(s)
    {
    }

protected:
    virtual void main() override
    {
        int v[1];
        v[0] = 0;
        if (stream_->get(v, 1)) {
            std::cout << "Consuming " << v << std::endl;
        }
    }

private:
    Stream<T>* stream_;
};

int main()
{
    try {
        System sys;

        MemoryStream<int> stream;

        auto consumer(sys.create_actor<Consumer<int>>("/consumer", 4, reinterpret_cast<Stream<int>*>(&stream)));
        auto producer(sys.create_actor<Producer<int>>("/producer", 5, reinterpret_cast<Stream<int>*>(&stream)));

        sys.start();

        std::this_thread::sleep_for(1ms);

        sys.stop();

        std::this_thread::sleep_for(1s);

        sys.terminate();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
