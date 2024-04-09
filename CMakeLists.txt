cmake_minimum_required(VERSION 3.17)
project(lightning-kit)

#
# CMake common settings
#
set(CMAKE_CXX_STANDARD 20)

#
# Dependent libraries
#
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# DPDK
pkg_check_modules(LIBDPDK libdpdk)

# DOCA
pkg_check_modules(LIBDOCAGPU doca-gpu)

# CUDA
find_package(CUDAToolkit)

# AVX
include(FindAVX.cmake)

if(CUDAToolkit_FOUND)
    set(LNG_WITH_CUDA 1)
    add_compile_definitions(LNG_WITH_CUDA=1)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    enable_language(CUDA)
endif()

if(LIBDOCAGPU_FOUND)
    set(LNG_WITH_DOCA 1)
    add_compile_definitions(LNG_WITH_DOCA=1)
elseif(LIBDPDK_FOUND)
    set(LNG_WITH_DPDK 1)
    add_compile_definitions(LNG_WITH_DPDK=1)
endif()

#
# Version
#
execute_process(COMMAND git describe --tags
    OUTPUT_VARIABLE GIT_DESCRIBE_RESULT
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
string(STRIP "${GIT_DESCRIBE_RESULT}" LNG_KIT_VERSION_S)
message(STATUS "Version: ${LNG_KIT_VERSION_S}")
add_definitions("-DLNG_KIT_VERSION=\"${LNG_KIT_VERSION_S}\"")
string(REPLACE "v" "" LNG_KIT_VERSION "${LNG_KIT_VERSION_S}")
file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/VERSION CONTENT "${LNG_KIT_VERSION_S}")

#
# Build main library
#
set(LNG_CORE_SRC
    src/actor.cc
    src/receiver-actor.cc
    src/event.cc
    src/log.cc
    src/stream.cc
    src/system.cc
    )
if(LNG_WITH_CUDA)
    list(APPEND LNG_CORE_SRC
        src/protocol_tcp.cu
        src/protocol_udp.cu)
endif()
if(LNG_WITH_DOCA)
    list(APPEND LNG_CORE_SRC
        src/doca-common-util.cc
        src/doca-common-util-internal.cc
        src/doca-stream.cc
        src/doca-tcp-util.cc
        src/doca-udp-util.cc)
endif()
add_library(lng-core ${LNG_CORE_SRC})

set(LNG_CORE_INCLUDE_DIRECTORIES_PUBLIC ${PROJECT_SOURCE_DIR}/include)
set(LNG_CORE_INCLUDE_DIRECTORIES_PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(LNG_CORE_COMPILE_OPTIONS_PUBLIC ${LATEST_AVX_FLAGS})
set(LNG_CORE_COMPILE_OPTIONS_PRIVATE)
set(LNG_CORE_LINK_LIBRARIES_PUBLIC Threads::Threads)
set(LNG_CORE_LINK_LIBRARIES_PRIVATE)
if(LNG_WITH_CUDA)
    list(APPEND LNG_CORE_COMPILE_OPTIONS_PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --default-stream per-thread --diag-suppress 177,550,1217,3010>)
    list(APPEND LNG_CORE_LINK_LIBRARIES_PRIVATE CUDA::cudart)
    set_target_properties(lng-core
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
endif()
if(LNG_WITH_DOCA)
    list(APPEND LNG_CORE_COMPILE_OPTIONS_PUBLIC ${LIBDOCAGPU_STATIC_CFLAGS} "-Wno-volatile" "-Wno-deprecated-declarations")
    list(APPEND LNG_CORE_LINK_LIBRARIES_PUBLIC ${LIBDOCAGPU_STATIC_LDFLAGS})
    list(APPEND LNG_CORE_LINK_LIBRARIES_PRIVATE ${LIBDOCAGPU_LIBDIR}/libdoca_gpunetio_device.a)
endif()
if(LNG_WITH_DPDK)
    list(APPEND LNG_CORE_COMPILE_OPTIONS_PUBLIC ${LIBDPDK_STATIC_CFLAGS})
    list(APPEND LNG_CORE_LINK_LIBRARIES_PUBLIC ${LIBDPDK_STATIC_LDFLAGS})
endif()
target_include_directories(lng-core
    PUBLIC ${LNG_CORE_INCLUDE_DIRECTORIES_PUBLIC}
    PRIVATE ${LNG_CORE_INCLUDE_DIRECTORIES_PRIVATE})
target_compile_options(lng-core
    PUBLIC ${LNG_CORE_COMPILE_OPTIONS_PUBLIC}
    PRIVATE ${LNG_CORE_COMPILE_OPTIONS_PRIVATE})
target_link_libraries(lng-core
    PUBLIC ${LNG_CORE_LINK_LIBRARIES_PUBLIC}
    PRIVATE ${LNG_CORE_LINK_LIBRARIES_PRIVATE})

#
# Test
#
add_executable(test_basic test/basic.cc)
target_link_libraries(test_basic lng-core)

if(LNG_WITH_DOCA OR LNG_WITH_DPDK)
    add_executable(test_dpdk_echo_udp test/dpdk_echo_udp.cc)
    target_link_libraries(test_dpdk_echo_udp lng-core ${LIBDPDK_STATIC_LDFLAGS})
    add_executable(dpdk_frame_builder_tcp test/dpdk_frame_builder_tcp.cc)
    target_link_libraries(dpdk_frame_builder_tcp lng-core ${LIBDPDK_STATIC_LDFLAGS})
endif()

if(LNG_WITH_DOCA)
    add_executable(test_doca_echo_udp test/doca_echo_udp.cc)
    target_link_libraries(test_doca_echo_udp lng-core ${LIBDPDK_STATIC_LDFLAGS})
    add_executable(test_doca_build_frame_tcp test/doca_build_frame_tcp.cc)
    target_link_libraries(test_doca_build_frame_tcp lng-core ${LIBDPDK_STATIC_LDFLAGS})
endif()

# if(LNG_WITH_NVIDIA)
#     add_executable(test_tcp test/test_tcp.cc)
#     target_link_libraries(test_tcp lng-core ${LIBDOCAGPU_STATIC_LDFLAGS})
# endif()

#
# Install
#
install(TARGETS lng-core DESTINATION lib)
install(DIRECTORY include DESTINATION .)
install(FILES
    ${PROJECT_BINARY_DIR}/VERSION
    DESTINATION .)
install(FILES
    ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION license)
install(FILES
    ${PROJECT_SOURCE_DIR}/thirdparty_notice.txt
    DESTINATION license)

#
# Packaging
#
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${LNG_KIT_VERSION})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-unknown)
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-x86-64-linux)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-arm-64-linux)
endif()

include(CPack)
