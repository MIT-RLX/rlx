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
    cargo test --release -p rlx-runtime --features cpu,mlx --test mlx_attention_parity

# Run rlx-cerebras tests (CSL codegen + matmul oracle parity). Pure Rust, no SDK.
test-cerebras:
    cargo test -p rlx-cerebras

# Emit CSL artifacts for an MxKxN matmul (default 32x64x32) into OUT.
# Compile + run on a Linux host with the Cerebras SDK container:
#   cd OUT && bash commands_wse2.sh
cerebras-emit M="32" K="64" N="32" OUT="cerebras-out":
    cargo run -q -p rlx-cerebras --bin rlx-cerebras-emit -- {{M}} {{K}} {{N}} {{OUT}}

# Emit + (if cslc is on PATH, i.e. inside the SDK container) compile & simulate.
cerebras-sim M="32" K="64" N="32" OUT="cerebras-out": (cerebras-emit M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v cslc >/dev/null 2>&1; then
        cd {{OUT}} && bash commands_wse2.sh
    else
        echo "cslc not found — run inside the Cerebras SDK container (Linux host)."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash commands_wse2.sh"
    fi

# Run rlx-qnn tests (QNN model-C++ codegen + matmul oracle parity). Pure Rust, no SDK.
test-qnn:
    cargo test -p rlx-qnn

# Emit QNN model artifacts for an MxKxN matmul (default 32x64x32) into OUT.
# Build + run on a Linux host with the QNN SDK (QNN_SDK_ROOT set):
#   cd OUT && bash run_qnn.sh
qnn-emit M="32" K="64" N="32" OUT="qnn-out":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- {{M}} {{K}} {{N}} {{OUT}}

# Emit + (if the QNN SDK tools are on PATH) build & run on the x86 reference backend.
qnn-run M="32" K="64" N="32" OUT="qnn-out": (qnn-emit M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v qnn-net-run >/dev/null 2>&1; then
        cd {{OUT}} && bash run_qnn.sh
    else
        echo "qnn-net-run not found — install the Qualcomm AI Engine Direct SDK (Linux host)."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash run_qnn.sh"
    fi

# SDK-free Docker self-test of the QNN host harness (verify.py + plumbing).
# Needs only Docker; uses a numpy stand-in for qnn-net-run.
qnn-docker-test M="8" K="16" N="4":
    python3 crates/backends/rlx-qnn/docker/validate.py harness-test --dims {{M}} {{K}} {{N}}

# Real Docker validation: build the model lib + run on libQnnCpu.so.
# Needs Docker AND the proprietary QNN SDK (set QNN_SDK_ROOT).
qnn-docker-run M="32" K="64" N="32":
    python3 crates/backends/rlx-qnn/docker/validate.py run --dims {{M}} {{K}} {{N}}

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

# GGUF grouped MoE integration — serial when multiple GPU backends link in.
test-gguf-grouped:
    cargo test -p rlx-runtime --test dequant_grouped_matmul_gguf -- --test-threads=1

# pyrlx: build extension into crates/bindings/pyrlx/.venv (first run) and run pytest.
test-pyrlx:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}/crates/bindings/pyrlx"
    if [[ ! -d .venv ]]; then
        python3 -m venv .venv
        .venv/bin/pip install -q maturin numpy pytest safetensors
        .venv/bin/maturin develop --features cpu,gguf-convert
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

# Cross-compile gate: the CPU + WebGPU stack must build for the browser
# (wasm32-unknown-unknown). Compile-only — running models in a browser is
# done via `just serve-web`.
check-wasm:
    rustup target add wasm32-unknown-unknown
    cargo check -p rlx-cpu -p rlx-wgpu -p rlx-webgl --target wasm32-unknown-unknown
    cargo check -p rlx-web --target wasm32-unknown-unknown
    cargo check -p rlx-web --target wasm32-unknown-unknown --features webgpu,webgl
    # rlx-webgl's planner + CPU executor are verified natively against autodiff.
    cargo test -p rlx-webgl

# Cross-compile gate: the Apple on-device stack must build for every Apple
# platform. The native backends are rlx-cpu (Accelerate/AMX), rlx-metal
# (Metal + MPS + MPSGraph) and rlx-coreml (ANE) — each compiles the *real*
# backend, not the non-Apple stub. Platform support matrix:
#   macOS / iOS / tvOS / visionOS → CPU + Metal + CoreML
#   watchOS                        → CPU/Accelerate only (no Metal API; CoreML
#                                    runtime model-compilation is unavailable)
# Compile-only — shipping to a device needs Xcode packaging (XCFramework).
# `check-ios` kept as an alias for the common iPhone/iPad case.
check-ios: check-apple
check-apple:
    # iOS is Rust tier-2 (prebuilt std): plain stable cargo. Driving the full
    # runtime pulls in the backend crates transitively. The `apple` umbrella
    # also exercises wgpu (Metal-on-Apple) + the MLX stub.
    rustup target add aarch64-apple-ios aarch64-apple-ios-sim
    cargo check -p rlx-runtime --features apple --target aarch64-apple-ios
    cargo check -p rlx-runtime --no-default-features --features cpu,metal,coreml --target aarch64-apple-ios-sim
    # tvOS / watchOS / visionOS are Rust tier-3 — build std from source (nightly).
    rustup component add rust-src --toolchain nightly
    # tvOS + visionOS ship Metal + CoreML, same surface as iOS.
    cargo +nightly check -Zbuild-std -p rlx-runtime --no-default-features --features cpu,metal,coreml --target aarch64-apple-tvos
    cargo +nightly check -Zbuild-std -p rlx-runtime --no-default-features --features cpu,metal,coreml --target aarch64-apple-visionos
    # watchOS: CPU/Accelerate backend only (no Metal API; no CoreML runtime compile).
    cargo +nightly check -Zbuild-std -p rlx-runtime --no-default-features --features cpu --target aarch64-apple-watchos

# Run the Apple backend smoke + parity test ON an iOS simulator. Boots a sim
# (override with RLX_SIM_DEVICE=<name|udid>) and runs the test binary inside it
# via `simctl spawn` — real on-simulator execution, not just a cross-compile.
# Needs Xcode + the iOS simulator runtime.
#
# Backends: cpu,metal,coreml. MLX is intentionally excluded from the *sim test*:
# a headless `simctl spawn` exposes no Metal device (so MLX/Metal can't run
# there anyway), and statically linking MLX pulls in newest-SDK Metal symbols
# (MTLTensor) that need a high link deployment target. MLX-on-iOS compile is
# covered by `just check-apple` (the `apple` umbrella includes mlx) + the host
# parity test in this same file.
test-apple-sim:
    rustup target add aarch64-apple-ios-sim
    CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUNNER={{justfile_directory()}}/scripts/apple-sim-runner.sh \
        cargo test -p rlx-runtime --no-default-features --features cpu,metal,coreml \
        --target aarch64-apple-ios-sim \
        --test apple_backends_sim -- --nocapture --test-threads=1

# Build the browser bundle (wasm + JS bindings) into crates/bindings/rlx-web/web/pkg.
# Add `--webgpu` to also bring up a WebGPU device. One command, all platforms.
build-web *ARGS:
    python3 crates/bindings/rlx-web/build.py {{ARGS}}

# Build + serve the demo at http://localhost:8000 (Ctrl-C to stop).
serve-web *ARGS:
    python3 crates/bindings/rlx-web/build.py --serve {{ARGS}}

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
ci: build test lint check-wasm test-pyrlx test-third-order-gpu test-rocm

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
