cmake_minimum_required(VERSION 3.10)
project(ShortcutBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native -g -O3 -Wall -pedantic")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BENCHMARK_SOURCES main/main.cpp main/step_reference.cpp)

# The Unix module in Rust depends on pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Read step implementations from each line into list
file(STRINGS step_implementations.txt STEP_IMPLEMENTATIONS)

foreach(STEP_IMPL ${STEP_IMPLEMENTATIONS})
    set(CPP_STEP_LIB "${STEP_IMPL}_cpp_lib")
    # Compile C++ step implementation into static library
    add_library(${CPP_STEP_LIB} STATIC "cpp/${STEP_IMPL}/step.cpp")
    # Path to Rust step implementation as a static library; TODO call cargo build from here
    set(RUST_STEP_LIB "${CMAKE_BINARY_DIR}/rust_cargo/release/lib${STEP_IMPL}.a")

    # Create C++ benchmark executable
    set(CPP_BENCHMARK_BIN "${STEP_IMPL}_cpp")
    add_executable(${CPP_BENCHMARK_BIN} ${BENCHMARK_SOURCES})
    target_include_directories(${CPP_BENCHMARK_BIN} PRIVATE main)
    target_include_directories(${CPP_STEP_LIB} PRIVATE main)
    target_include_directories(${CPP_STEP_LIB} PRIVATE cpp/tools)
    target_link_libraries(${CPP_BENCHMARK_BIN} ${CPP_STEP_LIB})

    # Create Rust benchmark executable
    set(RUST_BENCHMARK_BIN "${STEP_IMPL}_rust")
    add_executable(${RUST_BENCHMARK_BIN} ${BENCHMARK_SOURCES})
    target_include_directories(${RUST_BENCHMARK_BIN} PRIVATE main)
    target_link_libraries(${RUST_BENCHMARK_BIN}
        Threads::Threads
        ${RUST_STEP_LIB}
        ${CMAKE_DL_LIBS})
endforeach()
