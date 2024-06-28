cmake_minimum_required(VERSION 3.17)
project(perf CXX CUDA)

#
# CMake common settings
#
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 80)

# CUDA
find_package(CUDAToolkit)

list(APPEND PERF_SRC
  cuda_latency.cu)
add_library(perf-core ${PERF_SRC})

list(APPEND PERF_COMPILE_OPTIONS_PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --default-stream per-thread --diag-suppress 177,550,1217,3010>)
list(APPEND PERF_LINK_LIBRARIES_PRIVATE CUDA::cudart)
set_target_properties(perf-core
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
target_compile_options(perf-core
    PRIVATE ${PERF_COMPILE_OPTIONS_PRIVATE})
target_link_libraries(perf-core
    PRIVATE ${PERF_LINK_LIBRARIES_PRIVATE})

#
# Test
#
add_executable(test_latency cuda_latency.cpp)
target_link_libraries(test_latency perf-core)
