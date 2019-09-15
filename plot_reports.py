#!/usr/bin/env python3
import argparse
import collections
import csv
import os

import matplotlib.pyplot as plt
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 25,
})

COMPILERS = [
    {"key": "gcc",
     "display_name": "C++ (GCC)"},
    {"key": "clang",
     "display_name": "C++ (Clang)"},
    {"key": "rustc",
     "display_name": "Rust"},
]
colormap = plt.get_cmap("Accent", len(COMPILERS))
for i, compiler in enumerate(COMPILERS):
    compiler["color"] = colormap(i/len(COMPILERS))

with open("src/step_implementations.txt") as f:
    STEP_IMPLEMENTATIONS = [line.strip() for line in f]


def get_gflops(n, secs):
    total_float_ops = 2 * (n ** 3)
    return 1e-9 * total_float_ops / secs


def parse_data_from_log(log_path):
    with open(log_path) as f:
        header = f.readline().strip()
        assert header.startswith("benchmarking")
        num_elems = int(header.split(" elements")[0].split(" ")[-1])
        time = 0.0
        num_iter = 0
        for line in f:
            line = line.strip()
            if len(line.split(",")) > 1:
                # No more iteration results
                break
            time += float(line)
            num_iter += 1
        assert num_iter > 0, "Unexpected amount of iterations: {}".format(num_iter)
        # context-switches
        f.readline()
        # cpu-migrations
        f.readline()
        # page-faults
        f.readline()
        # cycles
        cycles_line = f.readline().strip().split(",")
        assert cycles_line[-1] == "GHz"
        # instructions
        instructions_line = f.readline().strip().split(",")
        assert instructions_line[-1] == "insn per cycle"
        return {"seconds": round(time/num_iter, 6),
                "cycles": int(cycles_line[0]),
                "instructions": int(instructions_line[0]),
                "n": round(num_elems ** 0.5)}


def plot_reports(title, report_path, metric):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.subplots()
    index = list(range(len(STEP_IMPLEMENTATIONS)))
    bar_width = 0.2
    bar_sep = bar_width/6
    opacity = 1.0
    compilers = COMPILERS[:3]
    num_compilers = len(compilers)

    if metric == "gflops":
        get_metric = lambda res: get_gflops(res["n"], res["seconds"])
        ax.set_ylabel("GFLOP/s")
    elif metric == "seconds":
        get_metric = lambda res: res["seconds"]
        ax.set_ylabel("Running time (seconds)")
    elif metric == "instructions-per-cycle":
        get_metric = lambda res: res["instructions"] / res["cycles"]
        ax.set_ylabel("Instructions per CPU cycle")
    elif metric == "instructions":
        get_metric = lambda res: 1e-9 * res["instructions"]
        ax.set_ylabel("Billions (1e9) of instructions")
    elif metric == "cycles":
        get_metric = lambda res: 1e-9 * res["cycles"]
        ax.set_ylabel("Billions (1e9) of CPU cycles")

    for i, compiler in enumerate(compilers):
        data = [
            parse_data_from_log(os.path.join(report_dir, compiler["key"], step + ".txt"))
            for step in STEP_IMPLEMENTATIONS
        ]
        index_with_offset = [
            idx + i*bar_width + i*bar_sep
            for idx in index
        ]
        metrics = [get_metric(res) for res in data]
        ax.bar(
            index_with_offset,
            metrics,
            bar_width,
            alpha=opacity,
            label=compiler["display_name"],
            color=compiler["color"]
       )

    ax.set_xlabel("Implementation")
    ax.set_xticks([idx + (num_compilers - 1) * (bar_width + bar_sep) / 2 for idx in index])
    ax.set_xticklabels([step[:2] for step in STEP_IMPLEMENTATIONS])
    if title is not None:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots from benchmark output.")
    parser.add_argument("report_output_dir",
        type=str,
        help="Directory to load reports from.")
    parser.add_argument("--output-path",
        type=str,
        help="Where to write output image. Defaults to plot.svg in the report directory.")
    parser.add_argument("--type",
        type=str,
        default="svg",
        help="File extension for default output path. Defaults to 'svg'.")
    parser.add_argument("--title", type=str)
    parser.add_argument("--metric",
        type=str,
        choices=(
            "cycles",
            "gflops",
            "instructions",
            "instructions-per-cycle",
            "seconds",
        ),
        default="gflops",
        help="Metric to be used on the vertical axis.")
    args = parser.parse_args()
    report_dir = args.report_output_dir
    plot_reports(args.title, report_dir, args.metric)
    if not args.output_path:
        output_img_path = os.path.join(report_dir, "plot." + args.type)
    else:
        output_img_path = args.output_path
    plt.savefig(output_img_path)
    print("Wrote plot to {}".format(output_img_path))
