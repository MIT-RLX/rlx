# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

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

# A GPU recipe that found no GPU has proved nothing, but a skipped test still
# prints `ok` — which is how a rig whose card had fallen off the PCIe bus
# reported a green suite while executing nothing on it. `RLX_REQUIRE_DEVICE`
# turns a skip into a failure for any backend that IS compiled in and still
# could not be instantiated; a backend the build does not contain is unaffected.
#
# On for the GPU recipes below, because asking for `just test-gpu` is asserting
# a GPU is present. A host without one wants `just require_device=0 test-gpu`.
require_device := "1"

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
    # Opt-in C++ quantized_matmul path for mxfp (must also pass default host path).
    cargo test --release -p rlx-mlx --features native-mxfp --test mxfp_path

# Run rlx-cerebras tests (CSL codegen + matmul oracle parity). Pure Rust, no SDK.
test-cerebras:
    cargo test -p rlx-cerebras

# Run rlx-lbm tests INCLUDING the graph path. `ir` is off by default (it pulls
# rlx-ir + rlx-runtime), so a bare `cargo test -p rlx-lbm` runs 18 of 26 tests
# and silently skips `graph_stream` / `lane_reset` — the two that check the
# solver as an rlx Graph rather than as host arithmetic.
test-lbm:
    cargo test -p rlx-lbm --features ir

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

# Run rlx-egpu tests + the Device::Egpu runtime seam. Pure Rust, no SDK, no device.
test-egpu:
    cargo test -p rlx-egpu --all-features
    cargo test -p rlx-runtime --features egpu --test egpu_seam

# Report what the PCIe tunnel is carrying (see docs/egpu.md).
egpu-probe:
    cargo run -q -p rlx-egpu --example egpu_probe

# Read a baked kernel pack or code object — needs no GPU toolchain.
egpu-inspect ARTIFACT:
    cargo run -q -p rlx-egpu --example egpu_inspect -- {{ARTIFACT}}

# Bake an ahead-of-time kernel pack. Needs hipcc (--hip) or ptxas (--ptx) on this host.
#   just egpu-bake "--hip kernels.hip:gfx1100,gfx1201" k.rlxisa
egpu-bake SPEC OUT="kernels.rlxisa":
    cargo run -q -p rlx-egpu --example egpu_bake -- -o {{OUT}} {{SPEC}}

# Fetch + verify the signed vendor firmware a bring-up hands to the card.
# FAMILIES: vendor (amd/nvidia) or IP family (gc_12_0 …); empty = everything.
egpu-firmware FAMILIES="":
    scripts/pull_gpu_firmware.sh {{FAMILIES}}
    cargo run -q -p rlx-egpu --features firmware --example egpu_firmware -- {{FAMILIES}}

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
    # A workspace `cargo test` builds every crate with its DEFAULT features, so
    # a test target gated behind a non-default feature is skipped without
    # saying so. rlx-lbm's graph path is the case in tree; keep it in the gate.
    cargo test --release -p rlx-lbm --features ir
    just test-debug-verifier

# Run the IR-verifier-bearing tests in a DEBUG profile.
#
# `rlx_ir::debug_assert_valid!` re-verifies the graph after every fusion pass
# that changed it — and it is `debug_assert`, so **`--release` never runs it**.
# Every other recipe here is `--release`, which meant the verifier was compiled
# out of the entire gate. Two real defects were sitting behind that:
#
#   * `Op::FusedTransformerLayer` declared 10 inputs while its lowering reads 8,
#     so a correctly-built no-bias node failed verification;
#   * rlx-vulkan emitted a `Step` with no matching `StepDep` on the `KvAppend`
#     path, tripping a `debug_assert_eq!` in its scheduler.
#
# Both passed `just test` and `just test-gpu` the whole time. Debug is slow, so
# this is scoped to the crates whose passes and schedulers carry the asserts
# rather than the whole workspace — the point is to run the checks at all.
test-debug-verifier:
    #!/usr/bin/env bash
    set -euo pipefail
    case "$(uname -s)" in
      Darwin) FEATURES="cpu,apple,vulkan" ;;
      *)      FEATURES="cpu,gpu,vulkan" ;;
    esac
    cargo test -p rlx-ir -p rlx-fusion -p rlx-compile
    cargo test -p rlx-runtime --features "$FEATURES" -j 4 --no-fail-fast

# GPU backends + runtime feature tests for the host platform.
# Darwin: Metal, MLX, wgpu, Vulkan (MoltenVK), `cpu,apple` runtime, third-order.
# Linux: wgpu, Vulkan, optional CUDA/ROCm via `test-third-order-gpu` / `test-rocm`.
# When `$RLX_HF_CACHE` (or `~/.cache/rlx/hf`) already holds an mlx-community
# checkout, Metal's `metal_hf_mlx_one_linear` runs without `RLX_HF_MLX=1`.
# Cold download: `RLX_HF_MLX=1 just test-gpu`.
test-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    CPUS={{cpus}}
    JOBS=$(( CPUS < 4 ? CPUS : 4 ))
    # A backend that skips for want of a device must not report `ok` here.
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      Darwin)
        cargo test --release -p rlx-metal -j "$JOBS"
        cargo test --release -p rlx-mlx -j "$JOBS"
        cargo test --release -p rlx-wgpu -j "$JOBS"
        cargo test --release -p rlx-vulkan -j "$JOBS"
        cargo test --release -p rlx-runtime --features cpu,apple -j "$JOBS"
        just test-mlx
        just test-third-order-gpu
        just check-corpus-device
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
        just check-corpus-device
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
    # Set up on what is actually MISSING, not on whether `.venv/` exists. The
    # directory test meant a venv left half-built by an interrupted run — or one
    # predating a dependency being added to this list — was never repaired, and
    # the recipe then failed forever on that machine with `No module named
    # pytest`. That is how `just ci` was failing here.
    [[ -x .venv/bin/python ]] || python3 -m venv .venv
    .venv/bin/python -m pytest --version >/dev/null 2>&1 \
        || .venv/bin/pip install -q maturin numpy pytest safetensors
    .venv/bin/python -c 'import pyrlx' >/dev/null 2>&1 \
        || .venv/bin/maturin develop --features cpu,gguf-convert
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
#
# `--all-targets` covers every target of every crate, but only under DEFAULT
# features — so a `#[cfg(feature = "vulkan")]` test or a backend behind an
# optional dep is invisible to it. That is not a hypothetical gap: it is why
# feature-gated code has landed unlinted here before. `lint-features` covers the
# combinations this host can actually build.
lint: lint-features
    cargo clippy --all-targets -- -D warnings

# Clippy the feature combinations `--all-targets` alone cannot reach.
#
# Host-appropriate by design: linting `apple` on Linux or `cuda` without a
# toolkit fails for reasons that have nothing to do with the code. CUDA/ROCm are
# probed first and skipped when they cannot build, so this is safe to run
# anywhere — and says which combinations it actually checked, because a lint
# gate that silently covered two of five reads exactly like one that covered all
# five.
lint-features:
    #!/usr/bin/env bash
    set -euo pipefail
    checked=(); skipped=()
    lint() {  # lint <label> <cargo args...>
        local label="$1"; shift
        if cargo clippy "$@" --all-targets -- -D warnings; then
            checked+=("$label")
        else
            echo "FAILED: $label" >&2; return 1
        fi
    }
    case "$(uname -s)" in
      Darwin)
        lint "rlx-runtime cpu,apple,vulkan" -p rlx-runtime --features cpu,apple,vulkan
        lint "rlx-corpus apple,vulkan"      -p rlx-corpus  --features apple,vulkan
        lint "rlx-ir test-support"          -p rlx-ir      --features test-support
        # The AMX/SME/BNNS paths are opt-in and default-OFF, so nothing else
        # here compiles them. `amx-bnns` shipped with a Result-vs-Option arm
        # that did not build at all, for exactly that reason.
        lint "rlx-cpu amx+splat"            -p rlx-cpu     --features amx-bnns,amx-dense,amx-sme,splat,parity-gemm
        ;;
      *)
        lint "rlx-runtime cpu,gpu,vulkan"   -p rlx-runtime --features cpu,gpu,vulkan
        lint "rlx-corpus gpu,vulkan"        -p rlx-corpus  --features gpu,vulkan
        lint "rlx-ir test-support"          -p rlx-ir      --features test-support
        ;;
    esac
    for f in cuda rocm; do
        if cargo check -p rlx-runtime --features "cpu,$f" --quiet 2>/dev/null; then
            lint "rlx-runtime cpu,$f" -p rlx-runtime --features "cpu,$f"
        else
            skipped+=("rlx-runtime cpu,$f (does not build on this host)")
        fi
    done
    echo "lint-features checked: ${checked[*]}"
    [[ ${#skipped[@]} -eq 0 ]] || printf 'lint-features SKIPPED: %s\n' "${skipped[@]}"

# Cross-check every Metal dispatch's buffer bindings against the kernel's
# declared parameters. Buffers are bound by integer index against MSL signatures
# in kernels.rs, and a stale index is not a compile error, not a crash and not a
# GPU fault — it reads zero, so a kernel whose `len` moved does nothing at all.
# Off by default (a relaxed atomic load); this turns it on for the suite.
validate-metal-bindings *ARGS:
    RLX_METAL_VALIDATE_BINDINGS=1 cargo test --release -j 4 -p rlx-metal --no-fail-fast {{ARGS}}

# Objective-C refcount gate for the Metal backend (macOS; no-ops elsewhere).
# The test suite cannot see a leak — an over-retained object still computes the
# right answer — so this runs Metal test binaries under `leaks --atExit`.
leak-check *TESTS:
    {{justfile_directory()}}/crates/backends/rlx-metal/scripts/leak-check.sh {{TESTS}}

# Install repo git hooks (auto-fmt + clippy on commit). Safe to re-run.
install-git-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    ROOT="$(git rev-parse --show-toplevel)"
    mkdir -p "$ROOT/.git/hooks"
    ln -sfn ../../scripts/git-hooks/pre-commit "$ROOT/.git/hooks/pre-commit"
    chmod +x "$ROOT/scripts/git-hooks/pre-commit"
    echo "installed .git/hooks/pre-commit → scripts/git-hooks/pre-commit"

# Refresh docs/op-coverage.md checkmarks from backend SUPPORTED_OPS claims.
gen-op-coverage:
    python3 scripts/gen-op-coverage.py

# Fail if docs/op-coverage.md drifts from backend SUPPORTED_OPS claims.
check-op-coverage:
    python3 scripts/gen-op-coverage.py --check

# Print the curated RLX_* environment catalog (high-signal options only).
env-catalog:
    cargo run -q -p rlx-ir --example env_catalog

# Regenerate docs/rlx-env-vars.md from the env registry (+ leftover mentions).
gen-rlx-env-vars:
    python3 scripts/gen-rlx-env-vars.py

# Named kernel corpus: every family through the device-free gate stack
# (verify -> repr -> memory-plan program-safety/schedule-semantics) under every
# planner configuration a backend uses. Answers "which families did my compiler
# change break", which the per-crate suites cannot.
check-corpus:
    cargo test -p rlx-corpus -- --nocapture

# The same corpus, executed on every device this host can reach, scored against
# the INDEPENDENT f64 oracle rather than against rlx's own CPU path — a defect
# both share (rms_norm_backward's extra 1/r, in all seven impls at once) is
# invisible to backend-vs-backend parity. Rolled up per family per device.
#
# Backends are opt-in per host because cargo unifies features across a workspace
# build: an Apple-only backend enabled by default here would be enabled for
# every crate on Linux too.
check-corpus-device:
    #!/usr/bin/env bash
    set -euo pipefail
    export RLX_REQUIRE_DEVICE="{{require_device}}"
    case "$(uname -s)" in
      Darwin) FEATURES="apple,vulkan" ;;
      *)      FEATURES="gpu,vulkan" ;;
    esac
    cargo test -p rlx-corpus --features "$FEATURES" --test device_gate -- --nocapture

# Env-registry gate, wired into `just ci`. Fails on any of:
#   * docs/rlx-env-vars.md drifting from the registry
#   * an RLX_* read with no registry entry
#   * an RLX_* read that bypasses the rlx_ir::env shim (std::env::var)
# The last one was a soft note until 208 call sites accumulated under it; the
# shim is what lets a test or an in-process A/B set a knob at all.
check-rlx-env-vars:
    python3 scripts/gen-rlx-env-vars.py --check

# Example-name gate, wired into `just ci`.
#
# Cargo writes every package's examples into one `target/<profile>/examples/`
# directory keyed by the bare target name, so two packages naming an example
# the same thing overwrite each other's binary and a harness run by that path
# measures whichever built last — under both labels. Prefix the example with
# its backend (`wgpu_schedule_matmul_ab`), or state in the script why the two
# can never share a target/ directory.
check-example-names:
    python3 scripts/check-example-names.py

# Markdown link gate, wired into `just ci`.
#
# Relative links rot silently: nothing renders an error, GitHub 404s only on a
# click, and `cargo doc` never reads a README. Regrouping the crates under
# crates/<group>/<crate>/ broke 59 links at once, and no gate noticed. This
# checks that every relative link resolves and that same-file `#anchor` links
# match a real heading.
check-doc-links:
    python3 scripts/check-doc-links.py

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

# Gates keyed on the WHOLE op set, not the subset some test happened to build.
#
# `rlx-ir`'s `test-support` feature exposes `sample_ops`: one constructible
# `Op` per `OpKind`, behind a `match` that fails to compile when an op is
# added. That is what lets a gate cover all 187 kinds — `rlx-corpus` reaches
# 13. The feature is off by default (it is scaffolding, not IR surface), so
# nothing else in the tree runs these.
check-ops:
    cargo test -p rlx-ir --features test-support --test arity_exhaustive
    # Declared arity vs. what backends actually index — the only check here
    # that compares the table against reality rather than against itself.
    # `--lib` runs the scanner's own regression tests: every early version of
    # it passed while silently finding nothing, so the parser is pinned too.
    cargo test -p rlx-check --features op-gates --lib op_scan
    cargo test -p rlx-check --features op-gates --test backend_operand_indexing
    cargo clippy -p rlx-ir --features test-support --all-targets -- -D warnings
    cargo clippy -p rlx-check --features op-gates --all-targets -- -D warnings

# Desktop coordinator for the mobile node demos. Defaults to `--self-test`,
# which runs the worker ranks here on loopback — so the desktop half is
# verifiable before a handset is involved.
#
#   just demo-coordinator                                  # loopback self-test
#   just demo-coordinator "--world 2 --peers 0.0.0.0:29500" # wait for a phone
demo-coordinator ARGS="--world 2 --self-test":
    cargo run -p rlx-ffi --example node_coordinator -- {{ARGS}}

# Build the iOS node demo app (xcframework → XcodeGen project → simulator
# build). Needs Xcode + `brew install xcodegen`.
demo-ios:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}
    ios/build-xcframework.sh
    cd ios/Demo
    xcodegen generate
    xcodebuild -project RlxDemo.xcodeproj -scheme RlxDemo \
        -sdk iphonesimulator -configuration Debug \
        CODE_SIGNING_ALLOWED=NO build
    echo "open ios/Demo/RlxDemo.xcodeproj, or install the .app with simctl"

# Build the Android node demo APK (cross-build the .so, then Gradle).
# Honors ANDROID_HOME / JAVA_HOME if already set; otherwise probes Homebrew.
demo-android:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}/android
    ./build.sh
    brew_prefix="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    export ANDROID_HOME="${ANDROID_HOME:-$brew_prefix/share/android-commandlinetools}"
    if [[ -z "${JAVA_HOME:-}" ]]; then
      for jdk in "$brew_prefix/opt/openjdk@17" "$brew_prefix/opt/openjdk"; do
        [[ -d "$jdk" ]] && export JAVA_HOME="$jdk" && break
      done
    fi
    if [[ ! -d "$ANDROID_HOME" ]]; then
      echo "Android SDK not found; set ANDROID_HOME." >&2; exit 1
    fi
    ./gradlew assembleDebug --console=plain
    echo "APK: android/app/build/outputs/apk/debug/app-debug.apk"

# Cross-compile gate: a distributed **node** must build for every non-desktop
# target we claim can join a mesh. Nodes are what make a cluster heterogeneous
# — a phone, an SBC, or the host driving an FPGA board is a rank like any
# other — and nothing else in the tree compiles the node stack off-desktop, so
# without this gate that support regresses silently.
#
# Layers, innermost first:
#   rlx-driver + rlx-collectives  transport + in-graph collectives (no platform gating)
#   rlx-runtime (dist::node)      the node driver + ship-graph worker
#   rlx-ffi                       C ABI shell (iOS xcframework, embedded C hosts)
#
# Android is gated on the NDK being installed: `cargo check` still runs C build
# scripts (zstd/bzip2 via the GGUF reader), which need the NDK's clang. The
# recipe reports the skip rather than passing quietly or failing on a machine
# that simply has no NDK.
check-nodes:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{justfile_directory()}}

    echo "==> iOS (aarch64-apple-ios)"
    rustup target add aarch64-apple-ios >/dev/null
    cargo check -p rlx-driver -p rlx-collectives --target aarch64-apple-ios
    cargo check -p rlx-runtime --no-default-features --features cpu --target aarch64-apple-ios
    # staticlib only: a cdylib must resolve every symbol at link time and
    # zstd-sys does not on iOS. This is the archive the xcframework wraps.
    cargo build -p rlx-ffi --lib --target aarch64-apple-ios

    echo "==> Android (aarch64-linux-android)"
    if ndk_env="$(android/ndk-env.sh 2>/dev/null)"; then
      eval "$ndk_env"
      rustup target add aarch64-linux-android >/dev/null
      cargo check -p rlx-driver -p rlx-collectives --target aarch64-linux-android
      cargo check -p rlx-runtime --no-default-features --features cpu --target aarch64-linux-android
      ( cd android && cargo check -p rlx-jni --target aarch64-linux-android )
    else
      echo "SKIPPED: no Android NDK (set ANDROID_NDK_HOME or ANDROID_HOME)." >&2
      echo "         Android node support was NOT verified by this run." >&2
    fi

    echo "==> node driver tests (host)"
    cargo test -p rlx-runtime --no-default-features --features cpu --test dist_node_fixed_function
    cargo test -p rlx-ffi

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

# Android cross-compile gate — CPU (NEON) + wgpu via the `android` feature.
# Needs NDK (ANDROID_NDK_HOME / ANDROID_HOME) for C deps (bzip2-sys, etc.).
android-check:
    #!/usr/bin/env bash
    set -euo pipefail
    rustup target add aarch64-linux-android
    eval "$("{{justfile_directory()}}/android/ndk-env.sh")"
    cargo check -p rlx-runtime --no-default-features --features android --target aarch64-linux-android
    cargo check -p rlx --no-default-features --features android --target aarch64-linux-android
    cargo check --manifest-path "{{justfile_directory()}}/android/rlx-jni/Cargo.toml" --target aarch64-linux-android

# Cross-build librlx_jni.so for the Android demo app (stages under jniLibs/).
# Pass --blas after ./android/build-openblas.sh for static OpenBLAS.
android-build *ARGS:
    {{justfile_directory()}}/android/build.sh {{ARGS}}

# End-to-end Android gate (emulator + instrumented tests). See android/e2e.sh.
android-e2e *ARGS:
    {{justfile_directory()}}/android/e2e.sh {{ARGS}}

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

# Verbose run — exposes [rlx] / [ktrace] log lines.
run-verbose CMD:
    RLX_VERBOSE=1 {{CMD}}

# Quick basic test of the workspace: build + test + lint + fast smokes.
ci: build test fmt-check lint check-rlx-env-vars check-example-names check-doc-links check-corpus check-wasm test-pyrlx test-third-order-gpu test-rocm leak-check validate-metal-bindings

# ROCm compile check + graph_devices parity (tests skip when HIP unavailable).
#
# Deliberately does NOT set `require_device`: this recipe is in `just ci`, which
# runs on developer machines where rlx-rocm compiles fine and no AMD GPU exists
# — exactly the case the flag is designed to fail. A rig gets it from rig.sh
# (RIG_REQUIRE_DEVICE), where "the device should be here" is actually true.
test-rocm:
    cargo check -p rlx-runtime --features cpu,rocm
    cargo test -p rlx-rocm --lib
    # Kernel argument-count check on. HIP reads exactly as many pointers as the
    # kernel declares and cannot see how many were passed, so a mismatch is
    # silent — this repo shipped one (see `gguf_gpu::launch_dequant_gguf`).
    RLX_GPU_VALIDATE_PARAMS=1 cargo test -p rlx-rocm
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
    export RLX_REQUIRE_DEVICE="{{require_device}}"
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
