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

#  RLX dev recipes (plan #67).
#
#  Borrowed from MAX's pixi-tasks pattern: each common dev command
#  lives here so onboarding doesn't have to remember --features
#  combinations. Install just from https://just.systems if you don't
#  have it; everything also works as plain cargo invocations.
#
#  Run with `just <recipe>` or `just --list` to see all recipes.

# Default recipe — list available commands.
default:
    @just --list

# Run the throttle gate before benching. CI-friendly --warn variant
# never exits non-zero.
[no-cd]
throttle:
    {{justfile_directory()}}/scripts/check-throttle.sh

throttle-warn:
    {{justfile_directory()}}/scripts/check-throttle.sh --warn

# Build whole workspace (release).
build:
    cargo build --release

# Build with everything turned on (Metal, kernel-trace, nan-check).
build-all:
    cargo build --release -p rlx-runtime --features "cpu,metal,kernel-trace,nan-check,blas-accelerate"

# Build rlx-mlx. First build compiles MLX from source (~minutes).
# Requires `git submodule update --init rlx-mlx-sys/vendor/mlx`.
build-mlx:
    cargo build --release -p rlx-mlx

# Run rlx-mlx tests (matmul+add parity check, both eager and lazy modes).
test-mlx:
    cargo test --release -p rlx-mlx

# FKL region fusion parity (docs/fk-fusion.md). Metal MPS tests skip off macOS.
test-fk:
    cargo test -p rlx-fusion fk_
    cargo test -p rlx-compile --lib fusion_pipeline::tests
    cargo test -p rlx-tpu --test fk_pipeline --test hlo_match batch_elementwise
    cargo test -p rlx-runtime --features cpu,metal,gpu,tpu --test fk_prologue_parity
    cargo test -p rlx-metal --test mps_graph_batch_region_lower --test mps_graph_prologue_region_lower
    cargo test -p rlx-mlx --test basic batch_elementwise_region_matches_atomic

# Run all unit tests.
test:
    cargo test --release

# pyrlx: build extension into pyrlx/.venv (first run) and run pytest.
test-pyrlx:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}/pyrlx"
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
        .venv/bin/pip install -q maturin numpy pytest
        .venv/bin/maturin develop --features cpu
    fi
    .venv/bin/python -m pytest tests/ -q

# Run a specific filter; use as `just testf narrow_attention`.
testf FILTER:
    cargo test --release {{FILTER}}

# Format check (no rewrite). Mirrors what CI should run.
fmt-check:
    cargo fmt --all -- --check

# Auto-format.
fmt:
    cargo fmt --all

# Clippy with warnings as errors.
lint:
    cargo clippy --all-targets -- -D warnings

# Run burnembed bench for a single model. `just bench minilm6`.
bench MODEL:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cd ../burnembed && cargo run --release \
        --example bench_rlx_single \
        --features "ndarray,blas-accelerate,rlx,hf-download" \
        -- --model {{MODEL}}

# Run burnembed Nomic Metal vs CPU comparison.
bench-nomic-metal:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cd ../burnembed && cargo run --release \
        --example bench_nomic_metal_vs_cpu \
        --features "rlx,rlx-metal,ndarray,blas-accelerate,hf-download"

# Verbose run — exposes [rlx] / [ktrace] log lines.
run-verbose CMD:
    RLX_VERBOSE=1 {{CMD}}

# Quick basic test of the workspace: build + test + lint + fast smokes.
ci: build test lint test-pyrlx test-third-order-gpu test-rocm

# ROCm compile check + graph_devices parity (tests skip when HIP unavailable).
test-rocm:
    cargo check -p rlx-runtime --features cpu,rocm
    cargo test -p rlx-rocm --lib
    cargo test -p rlx-rocm
    cargo test -p rlx-runtime --features cpu,rocm --test graph_devices_parity
    cargo test -p rlx-runtime --features cpu,rocm --test rocm_op_parity

# HIP-CPU kernel validation (linux-gnu Docker only). Clones HIP-CPU into docker/vendor/.
test-hip-cpu-validate:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    docker build -f "$root/rlx-cuda/docker/Dockerfile.hip-cpu-validate" -t rlx-hip-cpu-validate "$root"
    docker run --rm -v "$root:/work" -w /work rlx-hip-cpu-validate \
        bash -c 'set -euo pipefail
            bash rlx-cuda/docker/fetch-hip-cpu.sh
            cargo test -p rlx-cuda --features hip-cpu-validate
            cargo test -p rlx-rocm --features hip-cpu-validate'

# Higher-order AD: CPU tests + GPU parity (Apple backends on macOS, wgpu elsewhere).
test-third-order-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo test -p rlx-runtime --release --features cpu --test nth_order_grad
    case "$(uname -s)" in
      Darwin)
        cargo test -p rlx-runtime --release --features cpu,apple \
          --test higher_order_low_precision_parity
        cargo test -p rlx-runtime --release --features cpu,apple \
          --test third_order_gpu_parity --test directional_nth_gpu_parity
        cargo test -p rlx-runtime --release --features cpu,apple \
          --test higher_order_decompose_parity -- --test-threads=1 ;;
      *)
        cargo test -p rlx-runtime --release --features cpu,gpu \
          --test higher_order_low_precision_parity
        cargo test -p rlx-runtime --release --features cpu,gpu \
          --test third_order_gpu_parity --test directional_nth_gpu_parity
        cargo test -p rlx-runtime --release --features cpu,gpu \
          --test higher_order_decompose_parity -- --test-threads=1
        if cargo check -p rlx-runtime --features cpu,cuda,gpu --quiet 2>/dev/null; then
          cargo test -p rlx-runtime --release --features cpu,cuda,gpu \
            --test higher_order_decompose_parity -- --test-threads=1
        fi
        if cargo check -p rlx-runtime --features cpu,rocm --quiet 2>/dev/null; then
          cargo test -p rlx-runtime --release --features cpu,rocm \
            --test rocm_op_parity --test graph_devices_parity
          cargo test -p rlx-runtime --release --features cpu,rocm \
            --test third_order_gpu_parity --test directional_nth_gpu_parity \
            --test higher_order_low_precision_parity
          cargo test -p rlx-runtime --release --features cpu,rocm \
            --test higher_order_decompose_parity -- --test-threads=1
        fi ;;
    esac

# Update the Cargo.lock (pinned dep refresh; commit the lockfile).
update-lock:
    cargo update --workspace

# Run a CPU kernel micro-bench (plan #52). `just micro sgemm`.
micro NAME:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cargo bench -p rlx-cpu --bench {{NAME}}

# Run all CPU kernel micro-benches.
micro-all:
    {{justfile_directory()}}/scripts/check-throttle.sh
    cargo bench -p rlx-cpu
