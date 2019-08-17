#!/usr/bin/env bash
set -e

cpus=(Xeon_E3-1230_v5)
output_dir=$HOME/code/shortcut-comparison-web/img

for cpu in ${cpus[*]}; do
	python3 plot_reports.py \
		reports/${cpu}/single_core \
		--title Sequential \
		--output-path ${output_dir}/${cpu}_single_core.png
	python3 plot_reports.py \
		reports/${cpu}/multi_core \
		--title 'Parallel, 4 threads' \
		--output-path ${output_dir}/${cpu}_multi_core.png
done
