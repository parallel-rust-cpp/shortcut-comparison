#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include "step.hpp"
#include "step_reference.hpp"

float next_float() {
    static std::random_device rd;
    static std::default_random_engine e(rd());
    static std::uniform_real_distribution<float> floats(0.0, 1.0);
    return floats(e);
}

void benchmark(const unsigned n) noexcept {
    std::vector<float> data(n*n);
    std::generate(data.begin(), data.end(), next_float);
    std::vector<float> result(n*n);

    const auto time_start = std::chrono::high_resolution_clock::now();
    step(result.data(), data.data(), n);
    const auto time_end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> delta_seconds = time_end - time_start;
    std::cout << std::setprecision(7) << delta_seconds.count() << std::endl;
}

void test(const unsigned n) noexcept {
    std::vector<float> data(n*n);
    std::generate(data.begin(), data.end(), next_float);
    std::vector<float> result_correct(n*n);
    std::vector<float> result_testing(n*n);

    step_reference(result_correct.data(), data.data(), n);
    step(result_testing.data(), data.data(), n);

    for (auto i = 0u; i < n*n; ++i) {
        if (std::abs(result_testing[i] - result_correct[i]) > 1e-6) {
            std::cerr << "\nERROR: step function produced unexpected value: "
                      << result_testing[i]
                      << ", at index " << i
                      << ", while the reference solution produced "
                      << result_correct[i] << std::endl;
            break;
        }
    }
}

void run_benchmark(const unsigned n, const unsigned iterations) {
    std::cout << "for " << iterations << " iterations"
              << " with input containing "
              << n*n << " elements" << std::endl;
    for (auto i = 0u; i < iterations; ++i) {
        benchmark(n);
    }
}

void run_test(const unsigned n, const unsigned iterations) {
    std::cout << "for " << iterations << " iterations"
              << " with input containing "
              << n*n << " elements" << std::endl;
    for (auto i = 0u; i < iterations; ++i) {
        test(n);
        std::cout << '.' << std::flush;
    }
    std::cout << std::endl;
}


static const std::vector<std::string> VALID_COMMANDS {
    "benchmark",
    "test"
};

bool is_valid(const std::string& command) {
    return (std::find(VALID_COMMANDS.begin(), VALID_COMMANDS.end(), command)
            != VALID_COMMANDS.end());
}

void usage(const char* script_name) {
    std::cerr << "usage: " << script_name << " <command> N [ITERATIONS]\n"
              << "where command is one of:" << std::endl;
    for (const auto& c : VALID_COMMANDS) {
        std::cerr << "  " << c << std::endl;
    }
}

int main(int argc, const char** argv) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const std::string command(argv[1]);
    if (not is_valid(command)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const unsigned n = std::atoi(argv[2]);
    const unsigned iterations = argc == 4 ? std::atoi(argv[3]) : 1;

    if (command == "benchmark") {
        std::cout << "benchmarking " << argv[0] << ' ' << std::flush;
        run_benchmark(n, iterations);
    } else if (command == "test") {
        std::cout << "testing " << argv[0] << ' ' << std::flush;
        run_test(n, iterations);
    }
}