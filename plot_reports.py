#!/usr/bin/env python3
import argparse
import collections
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt


implementation_labels = tuple('v{}'.format(i) for i in range(7))
langs = (
    {
        "key": "cpp",
        "display_name": "C++",
        "github_color": "#f34b7d",
    },
    {
        "key": "rust",
        "display_name": "Rust",
        "github_color": "#dea584",
    },
)


def parse_data_from_csv(report_path, lang, reduction="mean"):
    data = collections.OrderedDict()
    for label in implementation_labels:
        with open(os.path.join(report_path, lang, label + ".csv")) as f:
            data[label] = list(csv.DictReader(f))
        if reduction == "mean":
            results = data[label]
            assert all(results[0]["N (rows)"] == res["N (rows)"] for res in results), "Unexpected differing input dataset sizes"
            data[label] = collections.OrderedDict([
                ("N (rows)", int(results[0]["N (rows)"])),
                ("GFLOP/s", sum(float(result["GFLOP/s"]) for result in results)/len(results)),
                ("time (us)", sum(int(result["time (us)"]) for result in results)/len(results)),
                ("instructions", sum(int(result["instructions"]) for result in results)/len(results)),
                ("cycles", sum(int(result["cycles"]) for result in results)/len(results)),
            ])
    return data


def plot_reports(report_path, metric):
    fig, ax = plt.subplots()

    index = np.arange(len(implementation_labels))
    bar_width = 0.35
    bar_sep = bar_width/8

    opacity = 1.0

    lang_count = len(langs)

    if metric == "gflops":
        get_metric = lambda res: res["GFLOP/s"]
        ax.set_ylabel('GFLOP/s')
    elif metric == "seconds":
        get_metric = lambda res: 1e-6 * res["time (us)"]
        ax.set_ylabel('Running time (seconds)')
    elif metric == "instructions-per-cycle":
        get_metric = lambda res: res["instructions"] / res["cycles"]
        ax.set_ylabel('Instructions per CPU cycle')
    elif metric == "instructions":
        get_metric = lambda res: 1e-9 * res["instructions"]
        ax.set_ylabel('Instructions (10^9)')
    elif metric == "cycles":
        get_metric = lambda res: 1e-9 * res["cycles"]
        ax.set_ylabel('CPU cycles (10^9)')

    for i, lang in enumerate(langs):
        data = parse_data_from_csv(report_path, lang["key"])
        index_with_offset = index + i*bar_width + (i - lang_count//2)*bar_sep
        gflops = tuple(get_metric(res) for res in data.values())
        ax.bar(index_with_offset, gflops, bar_width,
               alpha=opacity, color=lang["github_color"], label=lang["display_name"])

    ax.set_xlabel('Version')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(implementation_labels)
    ax.set_title("Throughput of floating point instructions (more is better)")
    ax.legend()

    fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots from benchmark csv reports.")
    parser.add_argument("report_dir",
        type=str,
        help="Directory to load reports from.")
    parser.add_argument("--output_dir",
        type=str,
        help="Directory where to write the resulting plot. Defaults to report dir.")
    parser.add_argument("--metric",
        type=str,
        choices=(
            "gflops",
            "seconds",
            "instructions-per-cycle",
            "instructions",
            "cycles",
        ),
        default="gflops",
        help="Metric to be used on the vertical axis. Defaults to GFLOP per second.")
    args = parser.parse_args()
    plot_reports(args.report_dir, args.metric)
    if not args.output_dir:
        output_img_path = os.path.join(args.report_dir, "plot.png")
    else:
        output_img_path = os.path.join(args.output_dir, "plot.png")

    plt.savefig(output_img_path)
    print("Wrote plot to {}".format(output_img_path))
