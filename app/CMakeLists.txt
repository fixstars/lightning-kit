cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_ARCHITECTURES 80)
project (lightning-kit LANGUAGES CXX)

#
# CMake common settings
#
set(CMAKE_CXX_STANDARD 20)

#
# Dependent libraries
#
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDPDK REQUIRED libdpdk)

add_executable(client_tcp client_tcp.cc)
target_link_libraries(client_tcp ${LIBDPDK_STATIC_LDFLAGS})
target_compile_options(client_tcp
    PRIVATE ${LIBDPDK_STATIC_CFLAGS} "-Wno-volatile" "-Wno-deprecated-declarations")

add_executable(client_udp client_udp.cc)
target_link_libraries(client_udp ${LIBDPDK_STATIC_LDFLAGS})
target_compile_options(client_udp
    PRIVATE ${LIBDPDK_STATIC_CFLAGS} "-Wno-volatile" "-Wno-deprecated-declarations")


