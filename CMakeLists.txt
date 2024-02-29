cmake_minimum_required(VERSION 3.17)
project(lightning-kit)

#
# CMake common settings
#
set(CMAKE_CXX_STANDARD 20)

#
# Dependent libraries
#
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDOCAGPU doca-gpu)
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    enable_language(CUDA)
endif()

#
# Flags
#
set(LNG_WITH_NV ${CUDAToolkit_FOUND} AND ${LIBDOCAGPU_FOUND})

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
    src/event.cc
    src/log.cc
    src/signal.cc
    src/system.cc
    )
if(${LNG_WITH_NV})
    list(APPEND LNG_CORE_SRC src/protocol.cu src/doca-util.cc)
endif()

add_library(lng-core ${LNG_CORE_SRC})
target_include_directories(lng-core
    PRIVATE ${PROJECT_SOURCE_DIR}/src
    PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(${LNG_WITH_NV})
    set_target_properties(lng-core
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    target_compile_options(lng-core
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --default-stream per-thread --diag-suppress 177,550,1217,3010>
        PUBLIC ${LIBDOCAGPU_STATIC_CFLAGS} "-Wno-volatile" "-Wno-deprecated-declarations")
    target_link_libraries(lng-core
        PRIVATE CUDA::cudart ${LIBDOCAGPU_LIBDIR}/libdoca_gpunetio_device.a
        PUBLIC ${LIBDOCAGPU_STATIC_LDFLAGS})
endif()

#
# Test
#
if(${LNG_WITH_NV})
    add_executable(test_gpunetio test/gpunetio.cc)
    target_link_libraries(test_gpunetio lng-core ${LIBDOCAGPU_STATIC_LDFLAGS})
endif()

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
