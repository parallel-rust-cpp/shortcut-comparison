#!/usr/bin/python3
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
    env = dict(os.environ.copy(),
        OMP_NUM_THREADS=str(num_threads),
        RAYON_NUM_THREADS=str(num_threads))
    result = subprocess.run(
        perf_cmd + cmd.split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    return parse_perf_csv(result.stdout.decode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark binaries for comparing C++ and Rust implementations of the step function")
    parser.add_argument("--build_dir", "-b",
        type=str,
        default=os.path.join(os.path.curdir, "build", "bin"),
        help="Path to the benchmark binaries, if not the default from build.py.")
    parser.add_argument("--limit_input_size_begin", "-n",
        type=int,
        default=0,
        help="n in sizes[n:], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--limit_input_size_end", "-m",
        type=int,
        default=len(INPUT_SIZES),
        help="m in sizes[:m], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--threads", "-t",
        type=int,
        default=1,
        help="Value for environment variables controlling number of threads, defaults to 1")
    parser.add_argument("--implementation", "-i",
        type=str,
        help="Filter implementations by prefix, e.g '-i v0' runs only v0_baseline.")
    parser.add_argument("--iterations", "-c",
        type=int,
        default=1,
        help="Amount of iterations for each input size")
    parser.add_argument("--no-cpp",
            action='store_true',
            help="Skip all C++ benchmarks")
    parser.add_argument("--no-rust",
            action='store_true',
            help="Skip all Rust benchmarks")

    args = parser.parse_args()
    build_dir = os.path.abspath(args.build_dir)
    impl_filter = args.implementation
    input_sizes = INPUT_SIZES[args.limit_input_size_begin:args.limit_input_size_end]
    benchmark_langs = []
    if not args.no_cpp:
        benchmark_langs.append("cpp")
    if not args.no_rust:
        benchmark_langs.append("rust")

    print_header("Running perf-stat for all implementations", end="\n\n")
    for step_impl in STEP_IMPLEMENTATIONS:
        if impl_filter and not step_impl.startswith(impl_filter):
            continue
        for lang in benchmark_langs:
            print_header(lang + ' ' + step_impl)
            bench_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            print("{:8s}{:10s}{:15s}{:15s}{:8s}".format("N", "time", "instructions", "cycles", "insn/cyc"))
            for input_size in input_sizes:
                for iter_n in range(args.iterations):
                    bench_args = 'benchmark {} 1'.format(input_size)
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

