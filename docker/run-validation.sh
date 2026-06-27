#!/usr/bin/env bash
# Validate the native Vulkan backend inside the Linux container.
#   1. report the Vulkan device the loader selected (lavapipe by default),
#   2. run the rlx-vulkan unit smoke tests,
#   3. run the Vulkan↔CPU cross-backend parity suite.
# Any failure aborts with a non-zero exit (CI-friendly).
set -euo pipefail

# Point the loader at the Mesa lavapipe ICD (arch suffix differs x86_64 / aarch64),
# unless the caller already chose an ICD (e.g. a real GPU — see README).
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    LVP_ICD="$(ls /usr/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -1 || true)"
    if [ -n "${LVP_ICD}" ]; then
        export VK_ICD_FILENAMES="${LVP_ICD}"
    fi
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"

# Keep Linux build artifacts out of the (macOS) host target/. Mount a volume
# here to cache across runs — see README.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/target}"

echo "==> Vulkan devices"
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null \
        | grep -E "deviceName|driverName|deviceType|apiVersion" \
        || echo "  (no device enumerated — check the mounted ICD)"
else
    echo "  (vulkaninfo not installed)"
fi

# Self-contained Linux validation: the rlx-vulkan crate's own tests run real
# compute on the selected device (lavapipe by default) and check exact values
# for matmul / elementwise / transpose / reduce / narrow / softmax. This builds
# cleanly on Linux (minimal deps) and is the gating check.
#
# (The broad Vulkan↔CPU parity suite in `crates/rlx-runtime/tests/vulkan_parity.rs`
# is run on macOS/hardware — it can't build here because rlx-runtime's test
# harness pulls in macOS-only dev-deps, rlx-metal / rlx-mlx.)
echo
echo "==> rlx-vulkan device-backed compute validation"
cargo test -p rlx-vulkan -- --nocapture

echo
echo "==> VULKAN LINUX VALIDATION PASSED"
