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
# Requires `git submodule update --init` for vendor/mlx.
build-mlx:
    cargo build --release -p rlx-mlx

# Run rlx-mlx tests (matmul+add parity smoke, both eager and lazy modes).
test-mlx:
    cargo test --release -p rlx-mlx

# Run all unit tests.
test:
    cargo test --release

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

# Quick smoke test of the workspace: build + test + lint.
ci: build test lint

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
