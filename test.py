#!/usr/bin/python3
import argparse
import os
import subprocess

from build import print_header, STEP_IMPLEMENTATIONS


INPUT_SIZES = [1, 10, 100, 200]


def run(cmd, num_threads):
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
    parser.add_argument("build_dir",
        type=str,
        help="Path to the benchmark binaries")
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

    print_header("Running tests for all implementations", end="\n\n")
    for lang in ("cpp", "rust"):
        for step_impl in STEP_IMPLEMENTATIONS:
            if impl_filter and not step_impl.startswith(impl_filter):
                continue
            print_header(lang + ' ' + step_impl)
            test_cmd = os.path.join(build_dir, step_impl + "_" + lang)
            iterations = 10
            for input_size in INPUT_SIZES:
                test_args = 'test {} {}'.format(input_size, iterations)
                cmd = test_cmd + ' ' + test_args
                print(run(cmd, args.threads))
            print()

