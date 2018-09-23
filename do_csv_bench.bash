#!/usr/bin/env bash

BUILD_DIR=/tmp/scbuild
REPORT_DIR=reports
BENCHMARK_SIZE=500
THREADS=4

# Exit immediately if some command exits with a non-zero code
set -e

function echo_header {
    echo
    echo -e "\e[1m$@\e[0m"
    echo
}

function clean {
    rm -rf $BUILD_DIR $REPORT_DIR
}

# Single thread

echo_header "==== SINGLE THREAD BENCHMARK ===="

echo_header "Building all libraries"

./build.py --verbose \
           --no-multi-thread \
           --build_dir $BUILD_DIR

echo_header "Testing all libraries"

./test.py --input_size 500 \
          --iterations 5 \
          --build_dir $BUILD_DIR 

echo_header "Running all benchmarks"

./bench.py --reporter_out csv \
           --report_dir "$REPORT_DIR/single_core" \
           --build_dir $BUILD_DIR \
           --input_size $BENCHMARK_SIZE \
           --threads $THREADS \
           --iterations 5

echo_header "==== SINGLE THREAD BENCHMARK COMPLETE ===="

rm -rf $BUILD_DIR

# Multi thread

echo_header "==== MULTI THREAD BENCHMARK ===="

echo_header "Building all libraries"

./build.py --verbose \
           --build_dir $BUILD_DIR

echo_header "Testing all libraries"

./test.py --input_size 500 \
          --iterations 5 \
          --build_dir $BUILD_DIR 

echo_header "Running all benchmarks"

./bench.py --reporter_out csv \
           --report_dir "$REPORT_DIR/multi_core" \
           --build_dir $BUILD_DIR \
           --input_size $BENCHMARK_SIZE \
           --threads $THREADS \
           --iterations 5

echo_header "==== MULTI THREAD BENCHMARK COMPLETE ===="

TOPOLOGY_FILE="$REPORT_DIR/cputopology.xml"
echo_header "Analyze CPU topology"
lstopo --no-io \
       --physical \
       --output-format console
lstopo --no-io \
       --physical \
       --force \
       $TOPOLOGY_FILE
echo "Wrote topology to $TOPOLOGY_FILE"
