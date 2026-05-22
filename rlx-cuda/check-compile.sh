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
