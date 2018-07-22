FROM debian:buster-slim
ARG SHORTCUT_URL=https://github.com/matiaslindgren/shortcut-comparison/archive/master.tar.gz
ARG RUSTUP_URL=https://sh.rustup.rs
WORKDIR /usr/share/shortcut
RUN apt update \
    && apt install -y curl tar linux-perf g++ make cmake python3 \
    # Remove perf script that detects perf version from uname
    && rm /usr/bin/perf \
    && ln -s /usr/bin/perf_4.16 /usr/bin/perf
RUN echo "Installing Rust" \
    && curl --fail-early --location $RUSTUP_URL > /tmp/rustup \
    && chmod 700 /tmp/rustup \
    && /tmp/rustup --verbose -y --default-toolchain nightly \
    && rm -rf /root/.rustup/toolchains/*/share/doc \
    && rm -rf /root/.rustup/toolchains/*/share/man
RUN echo "Downloading shortcut-comparison repository" \
    && curl --fail-early --location $SHORTCUT_URL | tar --extract --gunzip --strip-components 1
ENV PATH="${PATH}:/root/.cargo/bin"
