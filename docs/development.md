# Development guide

Day-to-day workflow for the RLX workspace. See also [`AGENTS.md`](../AGENTS.md) and
[`llms.txt`](../llms.txt).

## Prerequisites

- Rust toolchain from `rust-toolchain.toml`
- `just` for recipes (`just --list`)
- Python 3.9+ + `maturin` for `pyrlx`

## Common commands

```sh
just build          # workspace build
just test           # cargo test (workspace)
just lint           # clippy
just fmt            # rustfmt
just ci             # build + tests + clippy + pyrlx pytest
just throttle       # thermal gate before benchmarks
```

Before benchmarks: run `just throttle` or set `RLX_ALLOW_THROTTLE=1` for one-offs.
Use `rlx_ir::Tick` for sub-ms timing in hot paths, not `Instant::now()`.

## pyrlx

```sh
cd crates/pyrlx
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest safetensors
maturin develop --features cpu,gguf-convert,metal   # add backends as needed
pytest tests/ -q
# or: just test-pyrlx
```

Backend feature matrix: [`crates/pyrlx/docs/backends.md`](../crates/pyrlx/docs/backends.md).

Python DSL (`graph` / `Node`, scalar literals): [`crates/pyrlx/docs/dsl.md`](../crates/pyrlx/docs/dsl.md).
GGUF helpers: `quantize`, `load_gguf`, `convert_to_gguf` — see [`gguf-backend-paths.md`](gguf-backend-paths.md).
Runnable demo: `python crates/pyrlx/examples/dsl_quickstart.py` (after `maturin develop`).

Grouped MoE GGUF tests (multi-GPU backends): `just test-gguf-grouped`.

## Multi-backend runtime

Rust helpers live in `rlx-runtime` (`GraphDevices`, `DeviceRouter`, `DevicePolicy`).
Full API reference: [`backend-selection.md`](backend-selection.md).

```sh
cargo run --example graph_devices_demo -p rlx-runtime --features cpu
cargo test -p rlx-runtime --test graph_devices_parity
```

Env vars for pick/fallback: `RLX_DEVICE`, `RLX_DEVICE_CHAIN`, `RLX_DEVICES`,
`RLX_BENCHMARK_PICK` (see backend-selection doc).

## ROCm

```sh
just test-rocm                    # compile check + parity tests (skip without HIP)
just test-hip-cpu-validate        # HIP-CPU kernel tests in Docker (linux-gnu only)
cargo test -p rlx-runtime --features cpu,rocm --test rocm_op_parity
```

Pinned host I/O: `RLX_ROCM_PINNED_IO=1` (default on in graph exec mode, mirrors CUDA).

HIP-CPU headers: `rlx-cuda/docker/vendor/HIP-CPU` (cloned inside Docker; gitignored). Run `just test-hip-cpu-validate` — not on the macOS host.

## Dispatch probes

```sh
RLX_DISPATCH_REPORT=1 cargo test -p rlx-runtime --test some_test -- --nocapture
```

Or `dispatch_report_for_device` in Rust (see `rlx-runtime/src/device_ext.rs`).

## FKL region fusion

Resize prologue and batch region fusion: [`fk-fusion.md`](fk-fusion.md). Parity tests:

```sh
cargo test -p rlx-runtime --features cpu,metal,gpu,tpu --test fk_prologue_parity
cargo test -p rlx-fusion fk_
cargo test -p rlx-compile --lib fusion_pipeline::tests
cargo test -p rlx-tpu --test fk_pipeline --test hlo_match batch_elementwise
cargo test -p rlx-metal --test mps_graph_batch_region_lower
cd crates/pyrlx && python3 -m pytest tests/test_fk_batch_native.py tests/test_fk_batch_primitive.py -q
```

Or `just test-fk` from the repo root.

## Adding an op

1. `rlx-ir` — `Op`, inference, verifier
2. Every backend that should run it — thunk / executor, `supported_ops`
3. `rlx-fusion` / `rlx-compile` if fusion or legalization applies
4. Parity test in `rlx-runtime/tests/` or downstream

## Calibration caches

GPU backend ranking uses on-disk JSON under `~/.cache/rlx/` (`*-calib-*.json`).
Delete a file to force re-measurement. See [`backend-selection.md`](backend-selection.md).

## Docs index

| Path | Contents |
|------|----------|
| [`docs/README.md`](README.md) | Doc index |
| [`docs/backend-selection.md`](backend-selection.md) | Multi-backend API |
| [`docs/benchmarks/higher-order-ad.md`](benchmarks/higher-order-ad.md) | HO AD benches |
## License

GPL-3.0-only.
