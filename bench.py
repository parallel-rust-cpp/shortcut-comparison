#!/usr/bin/python3
import argparse
import collections
import csv
import os
import subprocess
import sys

from build import print_header, print_error, STEP_IMPLEMENTATIONS

INPUT_SIZES = [100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300]

# Use hwloc to pin 4 threads to 4 physical processing units on the first CPU
CPU_BIND_CMD = "hwloc-bind --cpubind --physical package:0.pu:0-3 --"
# This assumes the physical processing units are indexed 0, 1, 2, and 3.
# To verify this is the case on your platform, run the following in a shell:
# lstopo --physical --no-io --output-format ascii
# The PU indexes for each core in package P#0 should be numbered P#0, P#1, P#2, and P#3.
# If not, change the above command accordingly.

class PerfToolException(BaseException): pass


def get_gflops(n, secs):
    total_float_ops = 2 * n ** 3
    return 1e-9 * total_float_ops / secs


class Reporter:
    supported = ("stdout", "csv")

    def __init__(self, out="stdout", output_path=None):
        assert out in self.__class__.supported, "unsupported report format: {}".format(out)
        self.out = out
        self.fieldnames = ["N (rows)", "time (us)", "instructions", "cycles", "GFLOP/s"]
        self.output_path = output_path

    def print_row(self, row):
        if self.out == "stdout":
            insn_per_cycle = row["instructions"]/row["cycles"]
            print("{:8d}{:12d}{:15d}{:15d}{:10.2f}{:12.2f}".format(*row.values(), insn_per_cycle))
        elif self.out == "csv":
            with open(self.output_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)

    def print_header(self, clear_file=False):
        if self.out == "stdout":
            print("{:>8s}{:>12s}{:>15s}{:>15s}{:>10s}{:>12s}".format(*self.fieldnames, "insn/cycle"))
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


def run_perf(cmd, input_size, num_threads=None):
    """
    Run given command string with perf-stat and return results in dict.
    """
    perf_cmd = "perf stat --detailed --detailed --field-separator ,".split(' ')
    env = None
    if num_threads:
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


def do_benchmark(build_dir, iterations, benchmark_langs, threads, report_dir=None, reporter_out=None):
    print_header("Running perf-stat for all implementations", end="\n\n")
    for step_impl in STEP_IMPLEMENTATIONS:
        if impl_filter and not any(step_impl.startswith(prefix) for prefix in impl_filter):
            continue
        for lang in benchmark_langs:
            print_header(lang + ' ' + step_impl)
            report_path = None
            if report_dir:
                report_name = step_impl[:2] + '.' + reporter_out
                report_path = os.path.join(report_dir, lang, report_name)
            reporter = Reporter(reporter_out, report_path)
            reporter.print_header(clear_file=True)
            bench_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            for input_size in input_sizes:
                bench_args = 'benchmark {} 1'.format(input_size)
                cmd = ' '.join((CPU_BIND_CMD, bench_cmd, bench_args))
                for iter_n in range(iterations):
                    result = run_perf(cmd, input_size, threads)
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
        default=1,
        help="Amount of iterations for each input size")
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

    try:
        do_benchmark(build_dir, args.iterations, benchmark_langs, args.threads, args.report_dir, args.reporter_out)
    except PerfToolException:
        print()
        print_error("Failed to run perf due to low privileges. Consider e.g. decreasing the integer value in /proc/sys/kernel/perf_event_paranoid")
        sys.exit(1)
