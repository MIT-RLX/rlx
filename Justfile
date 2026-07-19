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

# Static graph checker (`cargo rlx check`) — dispatch, fusion, shape/dtype and
# numeric diagnostics. Device-free: CPU legality + all-target fusion, no GPU.
# Pass a graph JSON path or a demo, e.g. `just check-graph "--demo swiglu"`.
check-graph ARGS="--list-demos":
    cargo run -q -p rlx-check --bin cargo-rlx -- rlx check {{ARGS}}

# Install `cargo-rlx` so `cargo rlx check …` works from any RLX crate.
install-check:
    cargo install --path crates/tooling/rlx-check

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

# Emit a MatMul+Softmax QNN model for offline qnn-net-run.
qnn-emit-matmul-softmax M="8" K="16" N="4" OUT="qnn-mmsm":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --matmul-softmax {{M}} {{K}} {{N}} {{OUT}}

# Emit a two-layer MLP (LinearRelu → Linear) QNN model for offline qnn-net-run.
qnn-emit-mlp2 M="8" K="16" H="32" N="4" OUT="qnn-mlp2":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --mlp2 {{M}} {{K}} {{H}} {{N}} {{OUT}}

# Emit a Linear with STATIC weight/bias (activation-only input) for offline qnn-net-run.
qnn-emit-linear-static M="8" K="16" N="4" OUT="qnn-linstatic":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-static {{M}} {{K}} {{N}} {{OUT}}

# Emit LinearStatic then run the offline context-binary path (needs QNN SDK on PATH).
qnn-run-context M="8" K="16" N="4" OUT="qnn-linstatic": (qnn-emit-linear-static M K N OUT)
    #!/usr/bin/env bash
    set -e
    if command -v qnn-context-binary-generator >/dev/null 2>&1; then
        cd {{OUT}} && bash run_qnn_context.sh
    else
        echo "qnn-context-binary-generator not found — install the Qualcomm AI Engine Direct SDK."
        echo "artifacts are in {{OUT}}/; then: cd {{OUT}} && bash run_qnn_context.sh"
    fi

# Emit a Linear (in0·in1+in2) QNN model for offline qnn-net-run.
qnn-emit-linear M="8" K="16" N="4" OUT="qnn-linear":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear {{M}} {{K}} {{N}} {{OUT}}

# Emit a LinearRelu (relu(in0·in1+in2)) QNN model for offline qnn-net-run.
qnn-emit-linear-relu M="8" K="16" N="4" OUT="qnn-linrelu":
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-relu {{M}} {{K}} {{N}} {{OUT}}

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

# Run rlx-fpga tests (Verilog codegen + INT8 reference parity). Pure Rust.
test-fpga:
    cargo test -p rlx-fpga --release
    cargo test -p rlx-runtime --features cpu,fpga --lib export::

# Emit target-agnostic SystemVerilog for TinyConv-MNIST.
# TARGET = latency|size|energy|precision|bandwidth
# HW     = generic|ecp5|ice40|xilinx7:PART
fpga-emit TARGET="precision" HW="generic" OUT="":
    #!/usr/bin/env bash
    set -e
    if [ -n "{{OUT}}" ]; then
        cargo run -q -p rlx-fpga --release --bin rlx-fpga-emit -- --target {{TARGET}} --hw {{HW}} --out {{OUT}}
    else
        cargo run -q -p rlx-fpga --release --bin rlx-fpga-emit -- --target {{TARGET}} --hw {{HW}}
    fi

# Refresh the checked-in MNIST SystemVerilog demo (examples/mnist_sv/).
fpga-mnist-demo:
    cargo run -q -p rlx-fpga --release --example export_mnist

# SDK-free Docker self-test of the QNN host harness (verify.py + plumbing).
# Needs only Docker; uses a numpy stand-in for qnn-net-run.
qnn-docker-test M="8" K="16" N="4":
    python3 crates/backends/rlx-qnn/docker/validate.py harness-test --dims {{M}} {{K}} {{N}}

# Real Docker validation: build the model lib + run on libQnnCpu.so.
# Needs Docker AND the proprietary QNN SDK (set QNN_SDK_ROOT).
qnn-docker-run M="32" K="64" N="32":
    python3 crates/backends/rlx-qnn/docker/validate.py run --dims {{M}} {{K}} {{N}}

# Native (no Docker) QNN FFI + Session validation on a Linux host with the SDK.
# Expects QNN_SDK_ROOT (and typically LD_LIBRARY_PATH for libc++). Skips cleanly
# without the backend lib. Default backend: libQnnCpu.so.
qnn-ffi:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT / QNN SDK root}"
    export RLX_QNN_BACKEND_LIB="${RLX_QNN_BACKEND_LIB:-$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so}"
    export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cargo test -p rlx-qnn --features runtime -- --nocapture
    cargo test -p rlx-runtime --features qnn --test qnn_hexagon_matmul -- --nocapture
    cargo test -p rlx-runtime --features "cpu,qnn" --test fused_attention_block_parity -- --nocapture
    cargo test -p rlx-runtime --features "cpu,qnn" --test qnn_dequant_matmul -- --nocapture
    cargo test -p rlx-runtime --features qnn --test qnn_int8_matmul -- --nocapture
    cargo test -p rlx-runtime --features qnn --test qnn_int4_matmul -- --nocapture

# x86 HTP *functional simulator* (libQnnHtp.so) — no Snapdragon silicon.
# Re-runs CPU soft-skip probes (sfixed8×sfixed8) plus int4/int8 MatMul and a
# LinearStatic offline model.so + context-binary path under HTP prepare.
# Forces HTP even if RLX_QNN_BACKEND_LIB already points at libQnnCpu.so
# (common in env.sh); override with RLX_QNN_HTP_LIB if needed.
qnn-htp-sim:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT / QNN SDK root}"
    export PYTHONPATH="${PYTHONPATH:-}"
    TARGET=x86_64-linux-clang
    export RLX_QNN_BACKEND_LIB="${RLX_QNN_HTP_LIB:-$QNN_SDK_ROOT/lib/$TARGET/libQnnHtp.so}"
    export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/$TARGET${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    test -f "$RLX_QNN_BACKEND_LIB" || { echo "missing HTP backend: $RLX_QNN_BACKEND_LIB"; exit 1; }
    echo "HTP functional sim: $RLX_QNN_BACKEND_LIB"
    # FFI probes that CPU soft-skips or that exercise quantized weight paths.
    # cargo test accepts one filter; run each name separately.
    for t in \
        ffi_sfixed8_matmul_probe \
        ffi_int4_static_matmul \
        ffi_int8_static_matmul \
        ffi_int8_per_channel_matmul \
        ffi_int8_param_matmul
    do
        echo "=== $t ==="
        cargo test -p rlx-qnn --features runtime --lib "$t" -- --nocapture
    done
    # Offline LinearStatic on HTP sim (model.so + context binary).
    OUT=$(mktemp -d)
    cargo run -q -p rlx-qnn --bin rlx-qnn-emit -- --linear-static 4 8 2 "$OUT"
    (cd "$OUT" && bash run_qnn.sh)
    (cd "$OUT" && bash run_qnn_context.sh)
    echo "qnn-htp-sim OK"

# FKL region fusion parity (docs/fk-fusion.md). Metal MPS tests skip off macOS.
test-fk:
    cargo test -p rlx-fusion fk_
    cargo test -p rlx-compile --lib fusion_pipeline::tests
    cargo test -p rlx-tpu --test fk_pipeline --test hlo_match batch_elementwise
    cargo test -p rlx-runtime --features cpu,metal,gpu,tpu --test fk_prologue_parity
    cargo test -p rlx-metal --test mps_graph_batch_region_lower --test mps_graph_prologue_region_lower
    cargo test -p rlx-mlx --test basic batch_elementwise_region_matches_atomic

# Logical CPUs for cargo `-j` / libtest `--test-threads` (macOS, else Linux).
cpus := `sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4`

# Run all unit tests at high parallelism.
# On Darwin, Metal / MoltenVK / MLX share one GPU — cap concurrent cargo
# jobs so crates don't deadlock the device; within a binary, libtest still
# uses the default (CPU count) thread pool. Linux keeps full -j / threads.
# Keep `--test-threads=1` only on recipes that share a GPU runtime unsafely.
test:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    if [[ "$(uname -s)" == Darwin ]]; then
        # A few parallel test binaries is enough; more contended Apple GPU.
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test --release -j "$JOBS"
    else
        cargo test --release -j "$CPUS" -- --test-threads="$CPUS"
    fi

# GPU backends + runtime feature tests for the host platform.
# Darwin: Metal, MLX, wgpu, Vulkan (MoltenVK), `cpu,apple` runtime, third-order.
# Linux: wgpu, Vulkan, optional CUDA/ROCm via `test-third-order-gpu` / `test-rocm`.
test-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    JOBS=$(( CPUS < 4 ? CPUS : 4 ))
    case "$(uname -s)" in
      Darwin)
        cargo test --release -p rlx-metal -j "$JOBS"
        cargo test --release -p rlx-mlx -j "$JOBS"
        cargo test --release -p rlx-wgpu -j "$JOBS"
        cargo test --release -p rlx-vulkan -j "$JOBS"
        cargo test --release -p rlx-runtime --features cpu,apple -j "$JOBS"
        just test-mlx
        just test-third-order-gpu
        ;;
      *)
        cargo test --release -p rlx-wgpu -j "$CPUS" -- --test-threads="$CPUS"
        cargo test --release -p rlx-vulkan -j "$CPUS" -- --test-threads="$CPUS"
        cargo test --release -p rlx-runtime --features cpu,gpu -j "$CPUS" -- --test-threads="$CPUS"
        just test-third-order-gpu
        if cargo check -p rlx-runtime --features cpu,cuda,gpu --quiet 2>/dev/null; then
          cargo test --release -p rlx-cuda -j "$CPUS" -- --test-threads="$CPUS"
          cargo test --release -p rlx-runtime --features cpu,cuda,gpu -j "$CPUS" -- --test-threads=1
        fi
        just test-rocm
        ;;
    esac

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

# PyTorch → RLX: convert a torch model file to an RLX bundle + generated crate.
# `MODEL` is a .py exposing `model` + `example_inputs` (or get_model()/build()).
# Usage: just torch-import path/to/model.py out/
torch-import MODEL OUT:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}/crates/bindings/pyrlx"
    if [[ ! -d .venv-torch ]]; then
        python3 -m venv .venv-torch
        .venv-torch/bin/pip install -q torch safetensors numpy
    fi
    # Run the front-end as a standalone script (no compiled _pyrlx needed); it
    # shells to the Rust `rlx-torch-import` worker via the cargo fallback.
    .venv-torch/bin/python python/pyrlx/torch_import.py "{{MODEL}}" -o "{{OUT}}"

# rlx-torch-import: run the aten→rlx importer's Rust tests.
test-torch-import:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    if [[ "$(uname -s)" == Darwin ]]; then
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test -p rlx-torch-import -j "$JOBS"
    else
        cargo test -p rlx-torch-import -j "$CPUS" -- --test-threads="$CPUS"
    fi

# Run a specific filter; use as `just testf narrow_attention`.
testf FILTER:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    if [[ "$(uname -s)" == Darwin ]]; then
        JOBS=$(( CPUS < 4 ? CPUS : 4 ))
        cargo test --release -j "$JOBS" {{FILTER}}
    else
        cargo test --release -j "$CPUS" {{FILTER}} -- --test-threads="$CPUS"
    fi

# Format check (no rewrite). Mirrors what CI should run.
fmt-check:
    cargo fmt --all -- --check

# Auto-format.
fmt:
    cargo fmt --all

# Clippy with warnings as errors.
lint:
    cargo clippy --all-targets -- -D warnings

# Refresh docs/op-coverage.md checkmarks from backend SUPPORTED_OPS claims.
gen-op-coverage:
    python3 scripts/gen-op-coverage.py

# Fail if docs/op-coverage.md drifts from backend SUPPORTED_OPS claims.
check-op-coverage:
    python3 scripts/gen-op-coverage.py --check

# Print the curated RLX_* environment catalog (high-signal options only).
env-catalog:
    cargo run -q -p rlx-ir --example env_catalog

# Print the checklist / stub paths for adding a new Op (does not edit the tree).
# Usage: `just new-op MyOp` or `just new-op MyOp --write` to create empty stub files.
new-op NAME *ARGS:
    python3 scripts/new-op.py {{NAME}} {{ARGS}}

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
# `--serve-with npx` uses `npx serve`; `miniserve` / `basic-http-server` need
# `cargo install miniserve` or `cargo install basic-http-server`.
serve-web *ARGS:
    python3 crates/bindings/rlx-web/build.py --serve {{ARGS}}

# Serve only (no wasm rebuild). Requires `just build-web` first.
serve-web-static BACKEND="python" PORT="8000":
    python3 crates/bindings/rlx-web/serve.py --backend {{BACKEND}} --port {{PORT}}

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
