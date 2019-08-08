#!/usr/bin/python3
import argparse
import os
import subprocess
import sys

from build import print_header, STEP_IMPLEMENTATIONS


INPUT_SIZES = [1, 10, 100, 200]


def run(cmd, num_threads):
    env = None
    if num_threads:
        env = {
            "OMP_NUM_THREADS": str(num_threads),
            "RAYON_NUM_THREADS": str(num_threads)
        }
    result = subprocess.run(
        cmd.split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )
    return result.stdout.decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_dir", "-b",
        type=str,
        default=os.path.join(os.path.curdir, "build"),
        help="Path to the benchmark binaries, if not the default from build.py.")
    parser.add_argument("--threads", "-t",
        type=int,
        default=1,
        help="Value for environment variables controlling number of threads")
    parser.add_argument("--input_size", "-n",
        type=int,
        help="Run tests with this value for n")
    parser.add_argument("--iterations", "-c",
        type=int,
        default=10,
        help="Run each test this many times")
    parser.add_argument("--implementation", "-i",
        type=str,
        help="Filter implementations by prefix, e.g '-i v0' runs only v0_baseline.")
    parser.add_argument("--no-cpp",
        action='store_true')
    parser.add_argument("--no-rust",
        action='store_true')
    parser.add_argument("--verbose", "-v",
        action='store_true',
        default=False)

    args = parser.parse_args()
    build_dir = os.path.abspath(os.path.join(args.build_dir, "bin"))
    impl_filter = args.implementation

    if args.input_size:
        INPUT_SIZES = [args.input_size]

    all_ok = True

    langs = []
    if not args.no_cpp:
        langs.append("cpp")
    if not args.no_rust:
        langs.append("rust")

    for lang in langs:
        for step_impl in STEP_IMPLEMENTATIONS:
            if impl_filter and not step_impl.startswith(impl_filter):
                continue
            print_header(lang + ' ' + step_impl + ' ...', end=' ')
            test_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            failed = False
            for input_size in INPUT_SIZES:
                test_args = 'test {} {}'.format(input_size, args.iterations)
                cmd = test_cmd + ' ' + test_args
                output = run(cmd, args.threads)
                failed = "ERROR" in output
                if args.verbose:
                    print(output)
                if all_ok and failed:
                    all_ok = False
            print("! fail" if failed else "ok")

    if not all_ok:
        print()
        print_header("ERROR: at least one test failed")
        sys.exit(1)
