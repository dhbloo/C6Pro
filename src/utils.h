#pragma once

#include <cassert>
#include <cstdint>
#include <istream>
#include <memory>
#include <new>
#include <ostream>
#include <string>

#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    #include <bit>
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

// --------------------------------------------------------

// Define some macros for platform specific optimization hint
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #define FORCE_INLINE inline __attribute__((always_inline))
    #define NO_INLINE    __attribute__((noinline))
    #define RESTRICT     __restrict__
    #define LIKELY(x)    __builtin_expect(!!(x), 1)
    #define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
    #define NO_INLINE    __declspec(noinline)
    #define RESTRICT     __restrict
    #define LIKELY(x)    (x)
    #define UNLIKELY(x)  (x)
#else
    #define FORCE_INLINE inline
    #define NO_INLINE
    #define RESTRICT
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

// Define a macro to align data to cache line size
// This is useful for avoiding false sharing in multi-threaded applications.
#define ALIGN_CACHELINE alignas(std::hardware_destructive_interference_size)

// --------------------------------------------------------
// Common bit operations

// Counts the number of non-zero bits in a 32-bit bitboard.
inline int Popcount(uint32_t b)
{
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::popcount(b);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_popcount(b);  // Assumed gcc or compatible compiler
#elif defined(_MSC_VER)
    return _mm_popcnt_u32(b);  // MSVC
#else
    #error "Compiler not supported."
#endif
}

// Counts the number of non-zero bits in a 64-bit bitboard.
inline int Popcount(uint64_t b)
{
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::popcount(b);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_popcountll(b);  // Assumed gcc or compatible compiler
#elif defined(_MSC_VER)
    return _mm_popcnt_u64(b);  // MSVC
#else
    #error "Compiler not supported."
#endif
}

// Returns the least significant bit in a non-zero 32-bit bitboard.
inline int GetLSB(uint32_t b)
{
    assert(b);
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::countr_zero(b);
#elif defined(__clang__) || defined(__GNUC__)
    return int(__builtin_ctz(b));  // GCC, Clang, ICX
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, b);
    return int(idx);
#else
    #error "Compiler not supported."
#endif
}

// Returns the least significant bit in a non-zero 64-bit bitboard.
inline int GetLSB(uint64_t b)
{
    assert(b);
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::countr_zero(b);
#elif defined(__clang__) || defined(__GNUC__)
    return int(__builtin_ctzll(b));  // GCC, Clang, ICX
#elif defined(_MSC_VER)
    #ifdef _WIN64  // MSVC, WIN64
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return int(idx);
    #else          // MSVC, WIN32
    unsigned long idx;
    if (b & 0xffffffff) {
        _BitScanForward(&idx, int32_t(b));
        return int(idx);
    }
    else {
        _BitScanForward(&idx, int32_t(b >> 32));
        return int(idx + 32);
    }
    #endif
#else
    #error "Compiler not supported."
#endif
}

// Finds and clears the least significant bit in a non-zero 32-bit bitboard.
inline int PopLSB(uint32_t &b)
{
    assert(b);
    const int s = GetLSB(b);
    b &= b - 1;
    return s;
}

// Finds and clears the least significant bit in a non-zero 64-bit bitboard.
inline int PopLSB(uint64_t &b)
{
    assert(b);
    const int s = GetLSB(b);
    b &= b - 1;
    return s;
}

/// A right logical shift function that supports negetive shamt.
/// It might be implemented as rotr64 to avoid conditional branch.
inline uint64_t rotr(uint64_t x, int shamt)
{
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::rotr(x, shamt);
#elif defined(__clang__)
    return __builtin_rotateright64(x, shamt);
#elif defined(_MSC_VER)
    return _rotr64(x, shamt);
#else
    shamt &= 63;
    return (x << (64 - shamt)) | (x >> shamt);
#endif
}

// -------------------------------------------------

/// mulhi64() returns the higher 64 bits from two 64 bits multiply.
/// @see stackoverflow "getting-the-high-part-of-64-bit-integer-multiplication".

#ifdef __SIZEOF_INT128__  // GNU C

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    return ((unsigned __int128)a * (unsigned __int128)b) >> 64;
}

#elif defined(_M_X64) || defined(_M_ARM64)  // MSVC for x86-64 or AArch64

    #define mulhi64 __umulh

#else

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    uint64_t aL = (uint32_t)a, aH = a >> 32;
    uint64_t bL = (uint32_t)b, bH = b >> 32;
    uint64_t c1 = (aL * bL) >> 32;
    uint64_t c2 = aH * bL + c1;
    uint64_t c3 = aL * bH + (uint32_t)c2;
    return aH * bH + (c2 >> 32) + (c3 >> 32);
}

#endif

// -------------------------------------------------

/// Preloads the given address in L1/L2 cache. This is a non-blocking
/// function that doesn't stall the CPU waiting for data to be loaded
/// from memory, which can be quite slow.
inline void prefetch(const void *addr)
{
#ifndef NO_PREFETCH
    #if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(addr);
    #elif defined(_M_ARM) || defined(_M_ARM64)
    __prefetch(addr);
    #else
    _mm_prefetch((char *)addr, _MM_HINT_T0);
    #endif
#endif
}

namespace _PrefetchImpl {

template <int N>
struct PrefetchImpl
{};

template <>
struct PrefetchImpl<1>
{
    inline static void call(const char *addr) { ::prefetch(addr); }
};

template <>
struct PrefetchImpl<2>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<1>::call(addr);
        PrefetchImpl<1>::call(addr + 64);
    }
};

template <>
struct PrefetchImpl<3>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<2>::call(addr);
        PrefetchImpl<1>::call(addr + 128);
    }
};

template <>
struct PrefetchImpl<4>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<2>::call(addr);
        PrefetchImpl<2>::call(addr + 128);
    }
};

template <>
struct PrefetchImpl<5>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<4>::call(addr);
        PrefetchImpl<1>::call(addr + 256);
    }
};

template <>
struct PrefetchImpl<6>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<4>::call(addr);
        PrefetchImpl<2>::call(addr + 256);
    }
};

template <>
struct PrefetchImpl<7>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<4>::call(addr);
        PrefetchImpl<3>::call(addr + 256);
    }
};

template <>
struct PrefetchImpl<8>
{
    inline static void call(const char *addr)
    {
        PrefetchImpl<4>::call(addr);
        PrefetchImpl<4>::call(addr + 256);
    }
};

}  // namespace _PrefetchImpl

template <int NumBytes>
inline void multiPrefetch(const void *addr)
{
    constexpr int CacheLineSize = 64;
    constexpr int NumCacheLines = (NumBytes + CacheLineSize - 1) / CacheLineSize;
    _PrefetchImpl::PrefetchImpl<NumCacheLines>::call(reinterpret_cast<const char *>(addr));
}

// -------------------------------------------------
// Math/Template helper functions

/// Returns the size of a static array at compile time.
template <typename ElemType, int Size>
constexpr int arraySize(ElemType (&arr)[Size])
{
    return Size;
}

/// Get the result of base^exponent at compile time.
template <typename T>
constexpr T power(T base, unsigned exponent)
{
    return (exponent == 0) ? 1
           : (exponent % 2 == 0)
               ? power(base, exponent / 2) * power(base, exponent / 2)
               : base * power(base, (exponent - 1) / 2) * power(base, (exponent - 1) / 2);
}

/// Returns true if x is a power of two.
constexpr bool isPowerOfTwo(uint64_t x)
{
    return (x & (x - 1)) == 0;
}

/// Returns the log2 of x, rounded down to the nearest integer.
constexpr uint64_t floorLog2(uint64_t x)
{
    return x == 1 ? 0 : 1 + floorLog2(x >> 1);
}

/// Returns the nearest power of two less than x
constexpr uint64_t floorPowerOfTwo(uint64_t x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x ^ (x >> 1);
}

// --------------------------------------------------------

// Linear congruential hash
constexpr uint64_t LCHash(uint64_t x)
{
    return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

// --------------------------------------------------------
// Fast Pesudo Random Number Generator

/// PRNG struct is a fast generator based on SplitMix64
/// See <https://xoroshiro.di.unimi.it/splitmix64.c>
class PRNG
{
public:
    typedef uint64_t result_type;

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

    PRNG(uint64_t seed) : x(seed) {}

    /// Generates a random unsigned int64 number.
    uint64_t operator()();

private:
    uint64_t x;
};

// --------------------------------------------------------
// Time utility functions

typedef uint64_t Time;

/// Get the current time stamp in milliseconds.
Time Now();

// --------------------------------------------------------
// Container helpers

template <class T, size_t Size, size_t... Sizes>
struct MultiDimNativeArray
{
    using Nested = typename MultiDimNativeArray<T, Sizes...>::type;
    using type   = Nested[Size];
};

template <class T, size_t Size>
struct MultiDimNativeArray<T, Size>
{
    using type = T[Size];
};

/// Type alias to a multi-dimentional native array.
template <class T, size_t... Sizes>
using MDNativeArray = typename MultiDimNativeArray<T, Sizes...>::type;

// --------------------------------------------------------
// String helper functions

std::string timeText(Time time);
std::string nodesText(uint64_t nodes);
std::string speedText(uint64_t nodesPerSecond);

// --------------------------------------------------------
// Stream helper functions

class FormatGuard
{
public:
    FormatGuard(std::ostream &s) : os(s), oldState(nullptr) { oldState.copyfmt(os); }
    ~FormatGuard() { os.copyfmt(oldState); }

private:
    std::ostream &os;
    std::ios      oldState;
};

// --------------------------------------------------------
// NUMA-aware helper

namespace Numa {

/// Default NUMA node ID, used when NUMA is not supported or not available.
constexpr int DefaultNumaNodeId = 0;

/// The threshold for the number of bind groups. If the number of threads is
/// greater than this threshold, we will bind threads to NUMA groups to
/// improve performance. This is a heuristic value to determine when to
/// use NUMA-aware logic.
constexpr int BindGroupThreshold = 8;

/// Under Windows it is not possible for a process to run on more than one
/// logical processor group. This usually means to be limited to use max 64
/// cores. To overcome this, some special platform specific API should be
/// called to set group affinity for each thread. Original code from Texel by
/// Peter Österlund. We also let this function return the numa node ID for
/// the thread, to allow NUMA-aware logics in the thread.
/// @param idx The index of the thread to bind.
/// @return The NUMA node ID for the thread.
int bindThisThread(uint32_t idx);

}  // namespace Numa

// --------------------------------------------------------
// Memory allocation utility

namespace MemAlloc {

/// A warpper around std::aligned_alloc, which uses system calls if available.
/// Memory allocated using this function should be freed with alignedFree().
void *alignedAlloc(size_t alignment, size_t size);
/// Free memory allocated by alignedAlloc().
void alignedFree(void *ptr);
/// A helper to alloc an aligned array of type T. Ptr should be freed with alignedFree().
template <typename T, size_t Alignment = alignof(T)>
T *alignedArrayAlloc(size_t arraySize)
{
    return static_cast<T *>(alignedAlloc(Alignment, sizeof(T) * arraySize));
}

/// Allocate large page memory, with min alignment 4KiB. Memory allocated
/// using this function should be freed with alignedLargePageFree().
void *alignedLargePageAlloc(size_t size);

/// Free memory allocated by alignedLargePageAlloc().
void alignedLargePageFree(void *ptr);

/// A custom deleter for unique_ptr large page
template <typename T>
struct LargePageDeleter
{
    void operator()(T *ptr) const
    {
        if (!ptr)
            return;

        // Explicitly needed to call the destructor
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();

        alignedLargePageFree(ptr);
    }
};

}  // namespace MemAlloc

/// A unique_ptr type for managing large page pointer
template <typename T>
using LargePagePtr = std::unique_ptr<T, MemAlloc::LargePageDeleter<T>>;

// Make LargePagePtr by constructing single objects with the given arguments
template <typename T, typename... Args>
LargePagePtr<T> make_large_page(Args &&...args)
{
    static_assert(alignof(T) <= 4096,
                  "alignedLargePageAlloc() may fail for such a big alignment requirement of T");
    void *raw_memory = MemAlloc::alignedLargePageAlloc(sizeof(T));
    T    *obj        = new (raw_memory) T(std::forward<Args>(args)...);
    return LargePagePtr<T>(obj);
}

// --------------------------------------------------------
// Compression helper

class Compressor
{
public:
    enum class Type { NO_COMPRESS, LZ4_DEFAULT, ZIP_DEFAULT };

    /// Create a compressor with the given algorithm type.
    Compressor(std::ostream &ostream, Type type);
    /// Create a decompressor with the given algorithm type.
    Compressor(std::istream &istream, Type type);
    ~Compressor();

    /// Open an output stream by entry name.
    /// Entry name only work for ZIP type. Any type else should leave
    /// the entry name to empty string.
    /// @return Pointer to output stream, or nullptr if failed to open.
    std::ostream *openOutputStream(std::string entryName = "");

    /// Open an input stream by entry name.
    /// Entry name only work for ZIP type. Any type else should leave
    /// the entry name to empty string.
    /// @return Pointer to input stream, or nullptr if failed to open.
    std::istream *openInputStream(std::string entryName = "");

    /// Close current opened stream in advance. Opened stream will
    /// also be closed by the time of Compressor's destructor called.
    void closeStream(std::ios &stream);

private:
    class CompressorData;
    CompressorData *data;
};
