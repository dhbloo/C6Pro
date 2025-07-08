cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(C6Pro)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==========================================================
# Auto-detect SIMD/ISA capability of the host
include(CheckCXXSourceCompiles)

function(bool_to_onoff in_var out_var)
    if(${in_var})
        set(${out_var} ON  PARENT_SCOPE)
    else()
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

set(DEFAULT_USE_SSE               OFF)
set(DEFAULT_USE_AVX2              OFF)
set(DEFAULT_USE_AVX512            OFF)
set(DEFAULT_USE_BMI2              OFF)
set(DEFAULT_USE_VNNI              OFF)
set(DEFAULT_USE_NEON              OFF)
set(DEFAULT_USE_NEON_DOTPROD      OFF)
set(DEFAULT_USE_WASM_SIMD         OFF)
set(DEFAULT_USE_WASM_SIMD_RELAXED OFF)

if(NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    set(CMAKE_REQUIRED_QUIET TRUE)

    # ---- x86/AMD64 ----
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m128i a = _mm_setzero_si128(); (void)a; return 0; }"
        HAS_SSE)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m256i a = _mm256_setzero_si256(); (void)a; return 0; }"
        HAS_AVX2)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m512i a = _mm512_setzero_si512(); (void)a; return 0; }"
        HAS_AVX512)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ unsigned long long r = _pext_u64(0xFFFFULL, 0xAAULL); (void)r; return 0; }"
        HAS_BMI2)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){
            __m512i z = _mm512_setzero_si512();
            __m512i r = _mm512_dpbusd_epi32(z, z, z);
            (void)r; return 0;
        }"
        HAS_AVX512_VNNI)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){
            __m256i z = _mm256_setzero_si256();
            __m256i r = _mm256_dpbusd_avx_epi32(z, z, z);
            (void)r; return 0;
        }"
        HAS_AVX_VNNI)
    set(HAS_VNNI FALSE)
    if(HAS_AVX512_VNNI OR HAS_AVX_VNNI)
        set(HAS_VNNI TRUE)
    endif()

    # ---- ARM/ARM64 ----
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main(){ int32x4_t a = vdupq_n_s32(0); (void)a; return 0; }"
        HAS_NEON)
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main(){
            int32x4_t acc = vdupq_n_s32(0);
            int8x16_t  a  = vdupq_n_s8(0);
            int8x16_t  b  = vdupq_n_s8(0);
            int32x4_t  r  = vdotq_s32(acc, a, b);
            (void)r; return 0;
        }"
        HAS_NEON_DOTPROD)

    # ---- WebAssembly ----
    check_cxx_source_compiles("
        #include <wasm_simd128.h>
        int main(){ v128_t x = wasm_i32x4_splat(0); (void)x; return 0; }"
        HAS_WASM_SIMD)
    check_cxx_source_compiles("
        #include <wasm_simd128.h>
        int main(){
            v128_t acc = wasm_i32x4_splat(0);
            v128_t  a  = wasm_i8x16_splat(0);
            v128_t  b  = wasm_i8x16_splat(0);
            v128_t  r  = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(acc, a, b);
            (void)r; return 0;
        }"
        HAS_WASM_SIMD_RELAXED)

    bool_to_onoff(HAS_SSE               DEFAULT_USE_SSE)
    bool_to_onoff(HAS_AVX2              DEFAULT_USE_AVX2)
    bool_to_onoff(HAS_AVX512            DEFAULT_USE_AVX512)
    bool_to_onoff(HAS_BMI2              DEFAULT_USE_BMI2)
    bool_to_onoff(HAS_VNNI              DEFAULT_USE_VNNI)
    bool_to_onoff(HAS_NEON              DEFAULT_USE_NEON)
    bool_to_onoff(HAS_NEON_DOTPROD      DEFAULT_USE_NEON_DOTPROD)
    bool_to_onoff(HAS_WASM_SIMD         DEFAULT_USE_WASM_SIMD)
    bool_to_onoff(HAS_WASM_SIMD_RELAXED DEFAULT_USE_WASM_SIMD_RELAXED)

    message(STATUS "Detected SIMD/ISA capability of the host:")
    message(STATUS "  SSE:               ${DEFAULT_USE_SSE}")
    message(STATUS "  AVX2:              ${DEFAULT_USE_AVX2}")
    message(STATUS "  AVX512:            ${DEFAULT_USE_AVX512}")
    message(STATUS "  BMI2:              ${DEFAULT_USE_BMI2}")
    message(STATUS "  VNNI:              ${DEFAULT_USE_VNNI}")
    message(STATUS "  NEON:              ${DEFAULT_USE_NEON}")
    message(STATUS "  NEON_DOTPROD:      ${DEFAULT_USE_NEON_DOTPROD}")
    message(STATUS "  WASM_SIMD:         ${DEFAULT_USE_WASM_SIMD}")
    message(STATUS "  WASM_SIMD_RELAXED: ${DEFAULT_USE_WASM_SIMD_RELAXED}")
else()
    message(STATUS
        "Cross-compiling detected. Please specify the SIMD/ISA capability manually like -DUSE_AVX2=ON.")
endif()

#==========================================================
# CMake Options

option(USE_SSE  "Enable SSE2/SSSE3/SSE4.1 instruction" ${DEFAULT_USE_SSE})
option(USE_AVX2 "Enable AVX2/FMA instruction" ${DEFAULT_USE_AVX2})
option(USE_AVX512 "Enable AVX512 instruction" ${DEFAULT_USE_AVX512})
option(USE_BMI2 "Enable BMI2 instruction" ${DEFAULT_USE_BMI2})
option(USE_VNNI "Enable AVX512_VNNI/AVX_VNNI instruction" ${DEFAULT_USE_VNNI})
option(USE_NEON "Enable NEON instruction" ${DEFAULT_USE_NEON})
option(USE_NEON_DOTPROD "Enable NEON DotProd instruction" ${DEFAULT_USE_NEON_DOTPROD})
option(USE_WASM_SIMD "Enable WebAssembly SIMD instruction" ${DEFAULT_USE_WASM_SIMD})
option(USE_WASM_SIMD_RELAXED "Enable WebAssembly Relaxed SIMD instruction" ${DEFAULT_USE_WASM_SIMD_RELAXED})

option(NO_PREFETCH "Disable prefetch in search" OFF)
option(ENABLE_LTO "Enable link time optimization" OFF)
option(FORCE_ASSERTION "Enable force assertion for all build type" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "No build type selected, default to Release" FORCE)
endif()

#==========================================================

enable_testing()

# Source files
file(GLOB_RECURSE HEADERS "src/*.h")
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Core library
add_library(C6ProCore STATIC ${SOURCES} ${HEADERS})
target_include_directories(C6ProCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(external/simde)
target_link_libraries(C6ProCore PUBLIC simde)

# Main executable
add_executable(C6Pro src/main.cpp)
target_link_libraries(C6Pro PRIVATE C6ProCore)

#==========================================================

# Test executable
file(GLOB_RECURSE TEST_SOURCES "test/*.cpp")
add_executable(C6ProTest ${TEST_SOURCES})
add_subdirectory(external/doctest)
target_link_libraries(C6ProTest PRIVATE C6ProCore doctest::doctest)
include(${CMAKE_CURRENT_SOURCE_DIR}/external/doctest/scripts/cmake/doctest.cmake)
doctest_discover_tests(C6ProTest)

#==========================================================
# Compile flags

if(NO_PREFETCH)
    target_compile_definitions(C6ProCore PRIVATE NO_PREFETCH)
endif()
if(USE_SSE)
    target_compile_definitions(C6ProCore PRIVATE USE_SSE)
endif()
if(USE_AVX2)
    target_compile_definitions(C6ProCore PRIVATE USE_AVX2)
endif()
if(USE_AVX512)
    target_compile_definitions(C6ProCore PRIVATE USE_AVX512)
endif()
if(USE_BMI2 OR USE_AVX512)
    target_compile_definitions(C6ProCore PRIVATE USE_BMI2)
endif()
if(USE_VNNI)
	target_compile_definitions(C6ProCore PRIVATE USE_VNNI)
endif()
if(USE_NEON OR USE_NEON_DOTPROD)
    target_compile_definitions(C6ProCore PRIVATE USE_NEON)
endif()
if(USE_NEON_DOTPROD)
	target_compile_definitions(C6ProCore PRIVATE USE_NEON_DOTPROD)
endif()

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPOSupported OUTPUT error)
    if(IPOSupported)
        set_property(TARGET C6ProCore PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "LTO not supported: <${error}>")
    endif()
endif()

if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Setting up build for MSVC Clang-CL.")
        string(REGEX REPLACE "/O2" "/clang:-O3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        message(STATUS "Setting up build for MSVC.")
        string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()

	if(USE_AVX512)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA, AVX512F, AVX512DQ, AVX512BW flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__ /D__AVX512F__ /D__AVX512DQ__ /D__AVX512BW__")
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512f /clang:-mavx512dq /clang:-mavx512bw")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__AVX512VNNI__")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512vnni")
            endif()
        endif()
    elseif(USE_AVX2)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx2 /clang:-mfma")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavxvnni")
            endif()
        endif()
    elseif(USE_SSE)
        # Fix SSE3, SSSE3, SSE4.1 flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-msse /clang:-msse2 /clang:-msse3 /clang:-mssse3 /clang:-msse4 /clang:-msse4.1")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(USE_NEON_DOTPROD)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8.2-a+dotprod")
	    elseif(USE_NEON)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8-a+simd")
        endif()
    endif()

    if(FORCE_ASSERTION)
        string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
        string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
        set_target_properties(C6ProCore PROPERTIES LINK_FLAGS "/PROFILE")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL Release)
        set_property(TARGET C6ProCore PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Setting up build for GNU or Clang.")

	if(USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw")
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vnni -mavx512vl")
        endif()
    elseif(USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavxvnni")
        endif()
    elseif(USE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1")
    endif()

    if(USE_BMI2 OR USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")
    endif()

    if(USE_NEON_DOTPROD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
	elseif(USE_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    endif()

    if(FORCE_ASSERTION)
        string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
        string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -Wno-missing-braces -Wno-logical-op-parentheses -Wno-parentheses -Wno-sign-compare")

    if(NOT WIN32)
        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
            target_link_libraries(C6ProCore PRIVATE pthread)
        else()
		    target_link_libraries(C6ProCore PRIVATE pthread atomic)
        endif()
    endif()
endif()
