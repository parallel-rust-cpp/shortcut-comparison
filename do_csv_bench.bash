#!/usr/bin/env bash
# Exit immediately if some command exits with a non-zero code
set -e

BUILD_DIR=./build
REPORT_DIR=./reports
BENCHMARK_SIZE=6000
THREADS=4
TEST_ITERATIONS=1
DO_SINGLE_THREAD=1
DO_MULTI_THREAD=1
export CXX=$(type -p clang++)

function echo_red {
    echo -e "\e[31m$@\e[0m"
}

function echo_header {
    echo
    echo -e "\e[1m$@\e[0m"
    echo
}

function clean {
    if [ -d "$BUILD_DIR" ]; then
        echo "removing $BUILD_DIR"
        rm --recursive "$BUILD_DIR"
    fi
}

function check_dependencies {
    local all_ok=1
    local executables='python3 clang++ make cmake cargo rustc'
    for dep in $executables; do
        local where=$(type -p $dep)
        if [ -z $where ]; then
            echo_red "$dep not found"
            all_ok=0
        else
            echo "$dep is '$($where --version | head -1)' at $where"
        fi
    done
    if [ "$all_ok" != "1" ]; then
        echo_red "Some dependencies are missing"
        return 1
    fi
}

echo_header "Check all dependencies"
check_dependencies
echo

if [ $DO_SINGLE_THREAD -ne 0 ]; then
    echo_header "Single-thread benchmark"
    echo "Building all libraries in debug mode"
    clean
    ./build.py --debug --no-multi-thread --build_dir $BUILD_DIR
    echo "Testing all libraries"
    ./test.py --iterations $TEST_ITERATIONS \
        --build_dir $BUILD_DIR \
        --threads 1
    echo "Building all libraries"
    clean
    ./build.py --no-multi-thread --build_dir $BUILD_DIR
    echo "Running all benchmarks"
    ./bench.py --reporter_out csv \
        --no-perf \
        --report_dir "$REPORT_DIR/single_core" \
        --build_dir $BUILD_DIR \
        --input_size $BENCHMARK_SIZE \
        --threads 1
    echo_header "Single-thread benchmark complete"
fi

if [ $DO_MULTI_THREAD -ne 0 ]; then
    echo_header "Multi-thread benchmark"
    echo "Building all libraries in debug mode"
    clean
    ./build.py --debug --build_dir $BUILD_DIR
    echo "Testing all libraries"
    ./test.py --iterations $TEST_ITERATIONS \
        --build_dir $BUILD_DIR \
        --threads $THREADS
    echo "Building all libraries"
    clean
    ./build.py --build_dir $BUILD_DIR
    echo "Running all benchmarks"
    ./bench.py --reporter_out csv \
        --no-perf \
        --report_dir "$REPORT_DIR/multi_core" \
        --build_dir $BUILD_DIR \
        --input_size $BENCHMARK_SIZE \
        --threads $THREADS
    echo_header "Multi-thread benchmark complete"
fi
