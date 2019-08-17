#!/usr/bin/python3
import argparse
import collections
import csv
import os
import subprocess
import sys

from build import print_header, print_error, STEP_IMPLEMENTATIONS

INPUT_SIZES = [100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300]
STEP_ITERATIONS = [2, 3, 5, 10, 15, 20, 20, 20]
assert len(STEP_ITERATIONS) == len(STEP_IMPLEMENTATIONS)

# Prefix the benchmark command with taskset, binding the process to 4 physical cores
CPU_BIND_CMD = "taskset --cpu-list 0-3"

class PerfToolException(BaseException): pass


def get_gflops(n, secs):
    total_float_ops = 2 * n ** 3
    return 1e-9 * total_float_ops / secs


class Reporter:
    supported = ("stdout", "csv")

    def __init__(self, out="stdout", output_path=None, use_perf=False):
        assert out in self.__class__.supported, "unsupported report format: {}".format(out)
        self.out = out
        self.use_perf = use_perf
        if use_perf:
            self.fieldnames = ["N (rows)", "time (us)", "instructions", "cycles", "GFLOP/s"]
        else:
            self.fieldnames = ["N (rows)", "iterations", "time (us)", "GFLOP/s"]
        self.output_path = output_path

    def print_row(self, row):
        if self.out == "stdout":
            if self.use_perf:
                insn_per_cycle = row["instructions"]/row["cycles"]
                print("{:8d}{:12d}{:15d}{:15d}{:10.2f}{:12.2f}".format(*row.values(), insn_per_cycle))
            else:
                print("{:8d}{:12d}{:15d}{:10.2f}".format(*row.values()))
        elif self.out == "csv":
            with open(self.output_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)

    def print_header(self, clear_file=False):
        if self.out == "stdout":
            if self.use_perf:
                print("{:>8s}{:>12s}{:>15s}{:>15s}{:>10s}{:>12s}".format(*self.fieldnames, "insn/cycle"))
            else:
                print("{:>8s}{:>12s}{:>15s}{:>10s}".format(*self.fieldnames))
        elif self.out == "csv":
            with open(self.output_path, "w" if clear_file else "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()


def parse_perf_csv(s):
    if "You may not have permission to collect stats" in s:
        raise PerfToolException
    def get_value(key):
        return s.partition(key)[0].splitlines()[-1].rstrip(',')
    return collections.OrderedDict((
        ("time (us)", int(1e6*float(s.splitlines()[1]))),
        ("instructions", int(get_value("instructions"))),
        ("cycles", int(get_value("cycles"))),
    ))


def run(cmd, input_size, no_perf=False, num_threads=None):
    """
    Run given command string with perf-stat and return results in dict.
    """
    cmd_prefix = ''
    if not no_perf:
        cmd_prefix = "perf stat --detailed --detailed --field-separator ,"
    env = None
    if num_threads:
        env = dict(os.environ.copy(), OMP_NUM_THREADS=str(num_threads), RAYON_NUM_THREADS=str(num_threads))
    result = subprocess.run(
        (cmd_prefix + cmd).split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    out = result.stdout.decode("utf-8")
    return out if no_perf else parse_perf_csv(out)


def do_benchmark(build_dir, iterations_per_step_impl, benchmark_langs, threads, report_dir=None, reporter_out=None, no_perf=False):
    if no_perf:
        print_header("Benchmarking", end="\n\n")
    else:
        print_header("Benchmarking with perf-stat", end="\n\n")
    for step_impl in STEP_IMPLEMENTATIONS:
        if impl_filter and not any(step_impl.startswith(prefix) for prefix in impl_filter):
            continue
        iterations = iterations_per_step_impl[step_impl]
        for lang in benchmark_langs:
            print_header(lang + ' ' + step_impl)
            report_path = None
            if report_dir:
                report_name = step_impl[:2] + '.' + reporter_out
                report_path = os.path.join(report_dir, lang, report_name)
            reporter = Reporter(reporter_out, report_path, use_perf=not no_perf)
            reporter.print_header(clear_file=True)
            bench_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            for input_size in input_sizes:
                if no_perf:
                    bench_args = 'benchmark {} {}'.format(input_size, iterations)
                    cmd = ' '.join((CPU_BIND_CMD, bench_cmd, bench_args))
                    output = run(cmd, input_size, no_perf=no_perf, num_threads=threads)
                    result = collections.OrderedDict()
                    result["N (rows)"] = input_size
                    result["iterations"] = iterations
                    result["time (us)"] = int(1e6 * sum(float(line.strip()) for line in output.splitlines()[1:] if line.strip()))
                    result["GFLOP/s"] = get_gflops(input_size, 1e-6 * result["time (us)"] / iterations)
                    reporter.print_row(result)
                else:
                    bench_args = 'benchmark {} 1'.format(input_size)
                    cmd = ' '.join((CPU_BIND_CMD, bench_cmd, bench_args))
                    for iter_n in range(iterations):
                        result = run(cmd, input_size, no_perf=no_perf, num_threads=threads)
                        result["N (rows)"] = input_size
                        result.move_to_end("N (rows)", last=False)
                        result["GFLOP/s"] = get_gflops(input_size, 1e-6 * result["time (us)"])
                        reporter.print_row(result)
            if report_dir:
                print("Wrote {} report to {}".format(reporter_out, report_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark binaries for comparing C++ and Rust implementations of the step function")
    parser.add_argument("--build_dir", "-b",
        type=str,
        default=os.path.join(os.path.curdir, "build"),
        help="Path to build output directory from build.py, defaults to the default of build.py.")
    parser.add_argument("--limit_input_size_begin", "-n",
        type=int,
        default=0,
        help="n in sizes[n:], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--limit_input_size_end", "-m",
        type=int,
        default=len(INPUT_SIZES),
        help="m in sizes[:m], where sizes is {}".format(INPUT_SIZES))
    parser.add_argument("--input_size",
        type=int,
        help="Override input size with a single value. Takes precedence over --limit_input_size{begin,end}.")
    parser.add_argument("--threads", "-t",
        type=int,
        help="Value for environment variables controlling number of threads, defaults to 1")
    parser.add_argument("--implementation", "-i",
        action='append',
        type=str,
        help="Specify implementations to be run, e.g '-i v0' runs only v0_baseline. Can be specified multiple times.")
    parser.add_argument("--reporter_out",
        choices=Reporter.supported,
        default="stdout",
        help="Reporter output type")
    parser.add_argument("--report_dir",
        type=str,
        help="Directory to create reports")
    parser.add_argument("--iterations", "-c",
        type=int,
        help="Amount of iterations for each input size")
    parser.add_argument("--no-perf",
        action='store_true',
        help="Do not use perf, just benchmark and get the running time and compute gflops heuristic")
    parser.add_argument("--no-cpp",
        action='store_true',
        help="Skip all C++ benchmarks")
    parser.add_argument("--no-rust",
        action='store_true',
        help="Skip all Rust benchmarks")

    args = parser.parse_args()
    build_dir = os.path.abspath(os.path.join(args.build_dir, "bin"))
    if not os.path.exists(build_dir):
        print_error("Build directory {}, that should contain the test executables, does not exist".format(build_dir))
        sys.exit(1)
    impl_filter = tuple(set(args.implementation)) if args.implementation else ()
    if args.input_size:
        input_sizes = [args.input_size]
    else:
        input_sizes = INPUT_SIZES[args.limit_input_size_begin:args.limit_input_size_end]

    benchmark_langs = []
    if not args.no_cpp:
        benchmark_langs.append("cpp")
    if not args.no_rust:
        benchmark_langs.append("rust")

    if args.reporter_out == "csv":
        if args.report_dir:
            if not os.path.exists(args.report_dir):
                os.makedirs(args.report_dir)
                for lang in benchmark_langs:
                    os.mkdir(os.path.join(args.report_dir, lang))
        else:
            print_error("Reporter type 'csv' needs an output directory, please specify one with --report_dir")
            sys.exit(1)

    if args.iterations:
        iterations = {k: args.iterations for k in STEP_IMPLEMENTATIONS}
    else:
        iterations = {k: iterations for k, iterations in zip(STEP_IMPLEMENTATIONS, STEP_ITERATIONS)}
    try:
        do_benchmark(build_dir, iterations, benchmark_langs, args.threads, args.report_dir, args.reporter_out, args.no_perf)
    except PerfToolException:
        print()
        print_error("Failed to run perf due to low privileges. Consider e.g. decreasing the integer value in /proc/sys/kernel/perf_event_paranoid")
        sys.exit(1)
