#!/usr/bin/python3
"""
Awkward build script, should be in Rust.
"""
import argparse
import subprocess
import os
import sys


COMMANDS = {
    "cmake-generate": {
        "cmd": ["cmake", "-D", "SC_NO_MULTI_THREAD=0", "-G", "Unix Makefiles"],
    },
    "cargo-build": {
        "env": {"RUSTFLAGS": "-C target-cpu=native"},
        "cmd": ["cargo", "build", "--release"],
    },
    "make": {
        "cmd": ["make", "--jobs", "8"],
    },
}

with open(os.path.join("src", "step_implementations.txt")) as f:
    STEP_IMPLEMENTATIONS = f.read().splitlines()


def run(cmd, cwd, verbose=False):
    newenv = dict(os.environ.copy(), **cmd.get("env", {}))
    proc = subprocess.Popen(
        cmd["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=newenv,
    )
    for line in proc.stdout:
        if verbose:
            print(line.decode("utf-8"), end='')
    return proc.wait(timeout=1)

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
            help="Show all output from build commands")
    parser.add_argument("--no-multi-thread",
            action='store_true',
            help="Explicitly exclude multithreading support from all builds, i.e. OpenMP and Rayon syntax.")
    parser.add_argument("--cmake",
            type=str,
            help="Specify cmake binary to use instead of 'cmake'")
    parser.add_argument("--cxx",
            type=str,
            help="Specify CXX environment variable to be used when running cmake")

    args = parser.parse_args()
    root_dir =  os.path.abspath(args.source_root)
    build_dir = os.path.abspath(args.build_dir)
    cargo_target_dir = os.path.join(build_dir, "rust_cargo")

    if args.no_multi_thread:
        cmake_cmd = COMMANDS["cmake-generate"]["cmd"]
        cmake_cmd[2] = cmake_cmd[2][:-1] + "1"
    if args.cmake is not None:
        COMMANDS["cmake-generate"]["cmd"][0] = args.cmake
    if args.cxx is not None:
        cmake_env = COMMANDS["cmake-generate"].get("env", {})
        COMMANDS["cmake-generate"]["env"] = dict(cmake_env, CXX=args.cxx)

    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    if not os.path.exists(cargo_target_dir):
        os.mkdir(cargo_target_dir)

    print_header("Generating makefiles")
    cmake_gen = COMMANDS["cmake-generate"]
    cmake_gen["cmd"] += [root_dir]
    returncode = run(cmake_gen, build_dir, args.verbose)
    if returncode > 0:
        sys.exit(returncode)

    print_header("Building Rust libraries")
    cargo_build = COMMANDS["cargo-build"]
    cargo_build["env"]["CARGO_TARGET_DIR"] = cargo_target_dir
    for step_impl in STEP_IMPLEMENTATIONS:
        crate_dir = os.path.join(root_dir, "rust", step_impl)
        returncode = run(cargo_build, crate_dir, args.verbose)
        if returncode > 0:
            sys.exit(returncode)

    print_header("Building C++ libraries and benchmarks")
    returncode = run(COMMANDS["make"], build_dir, args.verbose)
    if returncode > 0:
        sys.exit(returncode)
