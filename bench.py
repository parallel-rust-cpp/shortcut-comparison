#!/usr/bin/python3
"""
measure function f(n) where n is input size and f is:
    * instructions per second (billions)
    * instructions per cycle (around 1.0)
    * cycles (billions)
    * total and max heap usage bytes
    * total amount of allocs
"""
import argparse
import os
import subprocess

from build import print_header, STEP_IMPLEMENTATIONS

INPUT_SIZES = [100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300]
ITERATIONS = [1]


def parse_perf_csv(s):
    def get_value(key):
        return s.partition(key)[0].splitlines()[-1].rstrip(',')
    return {
        "time": float(s.splitlines()[1]),
        "instructions": int(get_value("instructions")),
        "cycles": int(get_value("cycles")),
    }


def run_perf(cmd, num_threads=1):
    """
    Run given command string with perf-stat and return results in dict.
    """
    perf_cmd = "perf stat --detailed --detailed --field-separator ,".split(' ')
    env = {
        "OMP_NUM_THREADS": str(num_threads),
        "OMP_PROC_BIND": "true"
    }
    result = subprocess.run(
        perf_cmd + cmd.split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    return parse_perf_csv(result.stdout.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir",
        type=str,
        help="Path to the benchmark binaries")
    parser.add_argument("input_element_bytes",
        type=int,
        help="Amount of bytes of a single element in the generated input")

    args = parser.parse_args()
    build_dir = os.path.abspath(args.build_dir)
    input_bytes = args.input_element_bytes

    print_header("Running perf-stat for all implementations", end="\n\n")
    for lang in ("cpp", "rust"):
        for step_impl in STEP_IMPLEMENTATIONS:
            print_header(lang + ' ' + step_impl)
            bench_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            print("N", 4*' ', "1e9 instr / sec")
            for iterations in ITERATIONS:
                for input_size in INPUT_SIZES[:7]:
                    bench_args = 'benchmark {} {}'.format(input_size, iterations)
                    cmd = bench_cmd + ' ' + bench_args
                    result = run_perf(cmd)
                    bns_ops_per_sec = 1e-9 * result["instructions"]/result["time"]
                    print("{:4d}{:8.2f}".format(input_size, bns_ops_per_sec))

            print()

