#!/usr/bin/env bash
set -e

cpus=(i5-4690k Xeon-E3-1230-v5)
output_dir=plots

for cpu in ${cpus[*]}; do
	mkdir --parents --verbose ${output_dir}/${cpu}
	python3 plot_reports.py \
		reports/${cpu}/single-thread \
		--title Single-core \
		--output-path ${output_dir}/${cpu}/single_core.png
	python3 plot_reports.py \
		reports/${cpu}/multi-thread \
		--title Multi-core \
		--output-path ${output_dir}/${cpu}/multi_core.png
done
