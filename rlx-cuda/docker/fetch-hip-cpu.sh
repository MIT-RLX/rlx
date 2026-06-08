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

# Fetch HIP-CPU into docker/vendor (not a git submodule; Docker-only).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="$root/rlx-cuda/docker/vendor/HIP-CPU"
if [[ -f "$dest/include/hip/hip_runtime.h" ]]; then
  exit 0
fi
mkdir -p "$(dirname "$dest")"
git clone --depth 1 https://github.com/ROCm-Developer-Tools/HIP-CPU.git "$dest"
