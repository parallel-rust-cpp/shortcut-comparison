#!/usr/bin/env sh
set -e

model_name=$(lscpu | grep 'Model name:' | sed 's/Model name:[[:space:]]*//g')
cat << __END__ >> README.md

### CPU: $model_name
#### Topology
![CPU architecture sketch](cpu.png)
__END__

lstopo --fontsize 20 \
       --gridsize 20 \
       --no-icaches \
       --no-io \
       --no-legend \
       --output-format png > cpu.png
