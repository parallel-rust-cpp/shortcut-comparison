#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include "step.hpp"

float next_float() {
    static std::random_device rd;
    static std::default_random_engine e(rd());
    static std::uniform_real_distribution<float> floats(0.0, 1.0);
    return floats(e);
}

void benchmark(const unsigned n) noexcept {
    std::vector<float> result;
    result.reserve(n*n);
    std::vector<float> data;
    data.reserve(n*n);
    std::generate(data.begin(), data.end(), next_float);

    const auto time_start = std::chrono::high_resolution_clock::now();
    step(result.data(), data.data(), n);
    const auto time_end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> delta_seconds = time_end - time_start;
    std::cout << delta_seconds.count() << std::endl;
}

int main(int argc, const char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " N [ITERATIONS]" << std::endl;
        return EXIT_FAILURE;
    }
    const unsigned n = std::atoi(argv[1]);
    const unsigned iterations = argc == 3 ? std::atoi(argv[2]) : 1;
    std::cout << "benchmarking " << argv[0]
              << " with input containing "
              << n*n << " elements" << std::endl;

    for (auto i = 0u; i < iterations; ++i) {
        benchmark(n);
    }
}
