#!/usr/bin/env bash
set -e

BUILD_DIR=./build
REPORT_DIR=./reports
STEP_IMPLEMENTATIONS_LIST=./src/step_implementations.txt
BENCHMARK_SIZE=6000
BENCHMARK_ITERATIONS=20
THREADS=4
TEST_ITERATIONS=1

function echo_red {
    echo -e "\e[31m$@\e[0m"
}

function echo_header {
    echo -e "\e[1m$@\e[0m"
}

function clean {
    if [ -d "$BUILD_DIR" ]; then
        echo "removing $BUILD_DIR"
        rm --recursive "$BUILD_DIR"
    fi
}

function check_dependencies {
    local all_ok=1
    local executables='python3 g++ clang++ make cmake cargo rustc'
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
        echo_red "some dependencies are missing"
        return 1
    fi
}

echo_header "checking dependencies"
check_dependencies
echo

step_implementations=()
while read -r step; do
    step_implementations+=("$step")
done < $STEP_IMPLEMENTATIONS_LIST

mkdir --verbose --parents ${REPORT_DIR}/{single-thread,multi-thread}/{gcc,clang,rustc}

for threads in 1 $THREADS; do
    if [ $threads -eq 1 ]; then
        mode=single-thread
        cpu_list=0
    else
        mode=multi-thread
        cpu_list=$(seq -s, 0 $(($threads - 1)))
    fi
    echo_header "$mode, gcc and rustc"
    clean
    export CXX=$(type -p g++)
    export {OMP,RAYON}_NUM_THREADS=$threads
    echo "cpu list '$cpu_list', CXX '$CXX', omp threads '$OMP_NUM_THREADS', rayon threads '$RAYON_NUM_THREADS'"
    if [ $threads -eq 1 ]; then
        ./build.py --debug --no-multi-thread --build_dir $BUILD_DIR
    else
        ./build.py --debug --build_dir $BUILD_DIR
    fi
    echo "testing"
    ./test.py --iterations $TEST_ITERATIONS --build_dir $BUILD_DIR --threads $threads
    clean
    if [ $threads -eq 1 ]; then
        ./build.py --no-multi-thread --build_dir $BUILD_DIR
    else
        ./build.py --build_dir $BUILD_DIR
    fi
    for step in ${step_implementations[*]}; do
        echo_header "$step c++ gcc"
        perf stat --field-separator=, taskset --cpu-list $cpu_list $BUILD_DIR/bin/${step}_cpp benchmark $BENCHMARK_SIZE $BENCHMARK_ITERATIONS &> ${REPORT_DIR}/$mode/gcc/${step}.txt
        echo_header "$step rustc"
        perf stat --field-separator=, taskset --cpu-list $cpu_list $BUILD_DIR/bin/${step}_rust benchmark $BENCHMARK_SIZE $BENCHMARK_ITERATIONS &> ${REPORT_DIR}/$mode/rustc/${step}.txt
    done
    echo_header "$mode, clang"
    clean
    export CXX=$(type -p clang++)
    if [ $threads -eq 1 ]; then
        ./build.py --debug --no-multi-thread --build_dir $BUILD_DIR
    else
        ./build.py --debug --build_dir $BUILD_DIR
    fi
    echo "testing"
    ./test.py --no-rust --iterations $TEST_ITERATIONS --build_dir $BUILD_DIR --threads $threads
    clean
    if [ $threads -eq 1 ]; then
        ./build.py --no-multi-thread --build_dir $BUILD_DIR
    else
        ./build.py --build_dir $BUILD_DIR
    fi
    for step in ${step_implementations[*]}; do
        echo_header "$step c++ clang"
        perf stat --field-separator=, taskset --cpu-list $cpu_list $BUILD_DIR/bin/${step}_cpp benchmark $BENCHMARK_SIZE $BENCHMARK_ITERATIONS &> ${REPORT_DIR}/$mode/clang/${step}.txt
    done
done
