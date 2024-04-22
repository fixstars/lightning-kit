#ifndef LNG_SYSTEM_H
#define LNG_SYSTEM_H

#include <string>
#include <type_traits>
#include <unordered_map>

#include "lng/actor.h"
#include "lng/runtime.h"
#include "lng/stream.h"

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

    template <typename T, typename... Args,
        typename std::enable_if<!std::is_same<DPDKStream, T>::value>::type* = nullptr,
        typename std::enable_if<!std::is_same<DPDKGPUUDPStream, T>::value>::type* = nullptr>
    std::shared_ptr<T> create_stream(Args... args)
    {
        auto stream(std::make_shared<T>(args...));
        register_stream(stream);
        return stream;
    }

    template <typename T, typename... Args,
        typename std::enable_if<std::is_same<DPDKStream, T>::value>::type* = nullptr>
    std::shared_ptr<T> create_stream(Args... args)
    {
        register_runtime(Runtime::DPDK);

        auto stream(std::make_shared<T>(std::dynamic_pointer_cast<DPDKRuntime>(select_runtime(Runtime::DPDK)), args...));
        register_stream(stream);
        return stream;
    }

    template <typename T, typename... Args,
        typename std::enable_if<std::is_same<DPDKGPUUDPStream, T>::value>::type* = nullptr>
    std::shared_ptr<T> create_stream(Args... args)
    {
        register_runtime(Runtime::DPDKGPU);

        auto stream(std::make_shared<T>(std::dynamic_pointer_cast<DPDKGPURuntime>(select_runtime(Runtime::DPDKGPU)), args...));
        register_stream(stream);
        return stream;
    }

    void start();

    void stop();

    void terminate();

private:
    void register_actor(const std::string& id, const std::shared_ptr<Actor>& actor);
    void register_stream(const std::shared_ptr<Stream>& stream);
    void register_runtime(Runtime::Type type);

    std::shared_ptr<Runtime> select_runtime(Runtime::Type type);

    std::shared_ptr<Impl> impl_;
};

}

#endif
