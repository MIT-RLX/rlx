#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Fetch HIP-CPU into docker/vendor (not a git submodule; Docker-only).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="$root/rlx-cuda/docker/vendor/HIP-CPU"
if [[ -f "$dest/include/hip/hip_runtime.h" ]]; then
  exit 0
fi
mkdir -p "$(dirname "$dest")"
git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP-CPU.git "$dest"
