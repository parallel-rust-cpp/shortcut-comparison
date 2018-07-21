#!/usr/bin/python3
import argparse
import subprocess
import os


COMMANDS = {
    "cmake-generate": ["cmake", "-G", "Unix Makefiles"],
    "cargo-build": ["cargo", "build", "--release"],
}

STEP_IMPLEMENTATIONS = [
    "v0_baseline",
    "v1_linear_reading",
    "v2_instr_level_parallelism",
    "v3_simd",
]


def run(cmd, cwd, verbose=False):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    for line in popen.stdout:
        if verbose:
            print(line.decode(), end='')

class color:
    bold = '\033[1m'
    endc = '\033[0m'

def print_header(s, **kwargs):
    print(color.bold + s + color.endc, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", "-d",
            type=str,
            help="Source root directory",
            default="src")
    parser.add_argument("--build_dir", "-o",
            type=str,
            help="Directory for build output",
            default="build")
    parser.add_argument("--verbose", "-v",
            action='store_true',
            help="Show all output from build commands",
            default=False)

    args = parser.parse_args()
    root_dir =  os.path.abspath(args.source_root)
    build_dir = os.path.abspath(args.build_dir)
    cargo_target_dir = os.path.join(build_dir, "rust_cargo")

    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    if not os.path.exists(cargo_target_dir):
        os.mkdir(cargo_target_dir)

    print_header("Generating makefiles")
    run(COMMANDS["cmake-generate"] + [root_dir], build_dir, args.verbose)

    print_header("Building Rust libraries")
    os.environ["CARGO_TARGET_DIR"] = cargo_target_dir
    for step_impl in STEP_IMPLEMENTATIONS:
        crate_dir = os.path.join(root_dir, "rust", step_impl)
        run(COMMANDS["cargo-build"], crate_dir, args.verbose)

    print_header("Building C++ libraries and benchmarks")
    run(["make"], build_dir, args.verbose)

