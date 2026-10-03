# Linux/Vulkan validation image for rlx-wgpu.
#
# macOS can only exercise wgpu's Metal backend, so the Vulkan kernel variants
# (`matmul_wide_nv`, `matmul_coop_f32_portable`, the `*_coop_f16_vulkan*`
# family) never run there. This image gives them a Linux Vulkan device via
# Mesa's lavapipe software rasterizer — slow, but a real Vulkan ICD with real
# SPIR-V validation, which is what the portability questions are about.
#
# Build for the host arch (arm64 on Apple silicon) so it runs natively:
#   docker build -t rlx-wgpu-vk crates/backends/rlx-wgpu/docker
#   docker run --rm -v "$PWD:/rlx" -v rlx-vk-target:/target rlx-wgpu-vk \
#       cargo test -p rlx-wgpu --tests -- --test-threads=1
FROM rust:1-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers \
        vulkan-tools \
        libvulkan-dev \
        libopenblas-dev \
        pkg-config \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Pin the software ICD explicitly: with no physical GPU, an unset
# VK_ICD_FILENAMES can still enumerate zero devices depending on what else the
# image picked up. `*` covers both lvp_icd.aarch64.json and lvp_icd.x86_64.json.
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.aarch64.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
# Keep Linux object files out of the host's macOS target/ dir.
ENV CARGO_TARGET_DIR=/target
WORKDIR /rlx
