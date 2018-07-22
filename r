#!/bin/bash
# Test everything
./build.py && ./test.py && ./bench.py --limit_input_size_end 5
