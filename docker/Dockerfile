# Linux validation environment for the native Vulkan backend (rlx-vulkan).
#
# Ships Mesa **lavapipe** — a software (CPU) Vulkan 1.x implementation — so the
# backend's SPIR-V kernels can be validated on Linux with **no physical GPU**,
# which makes this usable in CI. With a real GPU + the right ICD mounted, the
# same image validates against hardware.
#
# The rlx source tree is NOT copied in; mount it at /work so the image stays
# tiny and always tests the current checkout:
#
#   docker build -t rlx-vulkan-linux ./docker
#   docker run --rm -v "$PWD":/work rlx-vulkan-linux
#
# See ./README.md for caching volumes and hardware-GPU usage.
FROM rust:1-bookworm

# Mesa Vulkan drivers (includes lavapipe / "lvp"), the loader, and vulkaninfo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers \
        libvulkan1 \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# lavapipe needs a runtime dir; compute needs no display. (The ICD is selected
# explicitly in run-validation.sh — VK_LOADER_DRIVERS_SELECT is intentionally
# NOT set here; it filters by driver name and suppresses auto-discovery.)
ENV XDG_RUNTIME_DIR=/tmp \
    CARGO_TERM_COLOR=always

WORKDIR /work

COPY run-validation.sh /usr/local/bin/run-validation
RUN chmod +x /usr/local/bin/run-validation

CMD ["run-validation"]
