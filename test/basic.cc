#include <iostream>
#include <stdexcept>
#include <vector>

#include "lng/lng.h"

using namespace lng;

template<typename T>
class Producer : public Actor {
public:
    Producer(Stream<T> s)
        : stream_(s)
    {}

    virtual void main() override {
        int v = 1;
        std::cout << "Producing " << v << std::endl;
        stream_.put(v);
    }

private:
    Stream<T> stream_;
};

template<typename T>
class Consumer : public Actor {
public:
    Consumer(Stream<T> s)
        : stream_(s)
    {}

    virtual void main() override {
        int v = stream_.get();
        std::cout << "Consuming " << v << std::endl;
    }

private:
    Stream<T> stream_;
};

int main()
{
    try {
        System sys;

        Stream<int> stream;

        Actor consumer = sys.create_actor<Consumer<int>>("/consumer", stream);
        Actor producer = sys.create_actor<Producer<int>>("/producer", stream);

        sys.run();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Passed" << std::endl;

    return 0;
}
