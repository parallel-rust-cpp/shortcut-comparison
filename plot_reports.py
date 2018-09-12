import argparse
import collections
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt


implementation_labels = tuple('v{}'.format(i) for i in range(8))
github_colors = {
    "rust": "#dea584",
    "cpp": "#f34b7d",
}
langs = (
    ("rust", "Rust"),
    ("cpp", "C++"),
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
            ])
    return data


def plot_reports(report_path):
    fig, ax = plt.subplots()

    index = np.arange(len(implementation_labels))
    bar_width = 0.35
    bar_sep = bar_width/8

    opacity = 1.0

    lang_count = len(langs)

    for i, lang in enumerate(langs):
        data = parse_data_from_csv(report_path, lang[0])
        index_with_offset = index + i*bar_width + (i - lang_count//2)*bar_sep
        gflops = tuple(res["GFLOP/s"] for res in data.values())
        ax.bar(index_with_offset, gflops, bar_width,
               alpha=opacity, color=github_colors[lang[0]], label=lang[1])

    ax.set_xlabel('Version')
    ax.set_ylabel('GFLOP/s')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(implementation_labels)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bar plots from benchmark csv reports.")
    parser.add_argument("report_dir",
        type=str,
        help="Directory to load reports from.")
    args = parser.parse_args()
    plot_reports(args.report_dir)
