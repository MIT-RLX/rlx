#!/usr/bin/env bash
# Compile-check rlx-cuda inside an NVIDIA CUDA Docker image.
#
# Run from the workspace root:
#   ./rlx-cuda/check-compile.sh
#
# Verifies the crate compiles + tests link against a real CUDA toolchain.
# No GPU is required (Docker on Mac works fine) — actual kernel dispatch
# only runs on hosts with NVIDIA hardware + driver.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Building rlx-cuda compile-check image..."
docker build -f rlx-cuda/Dockerfile.compile-check -t rlx-cuda-check .

echo
echo "==> Compile check passed. Image: rlx-cuda-check"
echo "    To bench on a real CUDA box, see rlx-cuda/README.md"
