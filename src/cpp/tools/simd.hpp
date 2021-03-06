#include <algorithm>
#include <limits>
#include <cstdlib>
#include <x86intrin.h>

#ifdef __GNUC__
typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
#elif __clang__
typedef float float8_t __attribute__ ((ext_vector_type(8)));
#else
#error "SIMD helpers currently typedef'd only for Clang and GNU GCC"
#endif

// Allocate memory for a 256-bit vector of floats and return the pointer
static float8_t* float8_alloc(size_t n) {
    void *ptr = 0;
    if (posix_memalign(&ptr, sizeof(float8_t), n * sizeof(float8_t))) {
        throw std::bad_alloc();
    }
    return (float8_t*)ptr;
}

constexpr float infty = std::numeric_limits<float>::infinity();
constexpr float8_t f8infty {
    infty, infty, infty, infty,
    infty, infty, infty, infty
};

// Return the value of the smallest element in a float8 vector
inline float hmin8(const float8_t& v) {
    float res = infty;
    for (int i = 0; i < 8; ++i) {
        res = std::min(v[i], res);
    }
    return res;
}

// Return a vector of the minimum elements for each pair of elements of two float8 vectors
inline float8_t min8(const float8_t& v, const float8_t& w) {
#ifdef __clang__
    return _mm256_min_ps(v, w);
#else
    return v < w ? v : w;
#endif
}

inline float8_t swap4(float8_t x) {
    return _mm256_permute2f128_ps(x, x, 0b00000001);
}

inline float8_t swap2(float8_t x) {
    return _mm256_permute_ps(x, 0b01001110);
}

inline float8_t swap1(float8_t x) {
    return _mm256_permute_ps(x, 0b10110001);
}
