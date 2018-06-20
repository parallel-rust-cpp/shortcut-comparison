cmake_minimum_required(VERSION 3.10)
project(ShortcutBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -Wall -pedantic")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/target)
set(BENCHMARK_SOURCES common/benchmark.cpp)

list(APPEND step_implementations
    v0_baseline)

foreach(step_impl ${step_implementations})
    set(SOURCES ${BENCHMARK_SOURCES} "${step_impl}/step.cpp")
    add_executable(${step_impl} ${SOURCES})
    target_include_directories(${step_impl} PRIVATE common)
endforeach()
