INCLUDE(CheckCXXSourceRuns)

set(FIND_AVX2)
set(FIND_AVX512F)
set(AVX2_FLAGS)
set(AVX512F_FLAGS)
set(LATEST_AVX_FLAGS)

SET(CMAKE_REQUIRED_FLAGS "-mavx2")
CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m256i a;
    __m256i result = _mm256_loadu_si256(&a);
    return 0;
}" FIND_AVX2)

SET(CMAKE_REQUIRED_FLAGS "-mavx512f")
CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m512i a;
    __m512i result = _mm512_loadu_si512(&a);
    return 0;
}" FIND_AVX512F)

if(${FIND_AVX2})
  set(AVX2_FLAGS "-mavx2")
  set(LATEST_AVX_FLAGS "-mavx2")
endif()

if(${FIND_AVX512F})
  set(AVX512F_FLAGS "-mavx512f")
  set(LATEST_AVX_FLAGS "-mavx512f")
endif()
