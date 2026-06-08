#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
