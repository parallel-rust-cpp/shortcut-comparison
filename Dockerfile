FROM base/archlinux:2018.06.01
RUN pacman --sync --refresh --sysupgrade --noconfirm
    && pacman --sync --noconfirm curl tar perf gcc make cmake python3 rustup \
    && rustup install nightly \
    && rustup default nightly \
    && rustup update \
    && curl --location https://github.com/matiaslindgren/shortcut-comparison/archive/master.tar.gz | tar --extract --gunzip --strip-components 1
