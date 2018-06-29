FROM base/archlinux:2018.06.01
RUN pacman --sync --refresh --sysupgrade --noconfirm
RUN pacman --sync --noconfirm perf gcc make cmake git python3 rustup
RUN rustup install nightly
RUN rustup default nightly
RUN rustup update
RUN git clone --depth 1 https://github.com/matiaslindgren/shortcut-comparison.git
