#!/usr/bin/env bash
# RLX — FFT bench on remote CUDA rig (Windows MSVC + WSL Ubuntu).
#
# Usage (from repo root on Mac):
#   ./scripts/bench_fft_rig.sh
#
# Requires rig SSH access (see rig.sh / scripts/rig/local.env).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Syncing workspace to rig..."
./rig.sh sync

BENCH='cargo run -p rlx-bench --release --example bench_fft --features cpu,gpu,cuda'

echo "Running FFT bench on Windows + WSL..."
./rig.sh --both bash -lc "$BENCH"
