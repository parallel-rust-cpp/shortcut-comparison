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


def run_perf(cmd, num_threads):
    """
    Run given command string with perf-stat and return results in dict.
    """
    perf_cmd = "perf stat --detailed --detailed --field-separator ,".split(' ')
    env = {
        "OMP_NUM_THREADS": str(num_threads),
        "RAYON_NUM_THREADS": str(num_threads)
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
    parser.add_argument("--limit_input_size_begin", "-n",
        type=int,
        default=len(INPUT_SIZES),
        help="n in sizes[:n], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--limit_input_size_end", "-m",
        type=int,
        default=0,
        help="n in sizes[n:], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--threads", "-t",
        type=int,
        default=1,
        help="Value for environment variables controlling number of threads")
    parser.add_argument("--implementation", "-i",
        type=str,
        help="Filter implementations by prefix, e.g '-i v0' runs only v0_baseline.")

    args = parser.parse_args()
    build_dir = os.path.abspath(args.build_dir)
    impl_filter = args.implementation
    input_sizes = INPUT_SIZES[args.limit_input_size_begin:args.limit_input_size_end]

    print_header("Running perf-stat for all implementations", end="\n\n")
    for step_impl in STEP_IMPLEMENTATIONS:
        if impl_filter and not step_impl.startswith(impl_filter):
            continue
        for lang in ("cpp", "rust"):
            print_header(lang + ' ' + step_impl)
            bench_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            print("{:8s}{:10s}{:15s}{:15s}{:8s}".format("N", "time", "instructions", "cycles", "insn/cyc"))
            for iterations in ITERATIONS:
                for input_size in input_sizes:
                    bench_args = 'benchmark {} {}'.format(input_size, iterations)
                    cmd = bench_cmd + ' ' + bench_args
                    result = run_perf(cmd, args.threads)
                    insn_per_cycle = result["instructions"]/result["cycles"]
                    print("{:4d}{:8.3f}{:15d}{:15d}{:12.3f}".format(
                        input_size,
                        result["time"],
                        result["instructions"],
                        result["cycles"],
                        insn_per_cycle))

            print()

