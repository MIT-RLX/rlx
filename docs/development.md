# Development guide

Day-to-day workflow for the RLX workspace. See also [`llms.txt`](../llms.txt)
for the workspace map, and [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the
review and ground rules.

## Prerequisites

- Rust toolchain from `rust-toolchain.toml`
- `just` for recipes (`just --list`)
- Python 3.9+ + `maturin` for `pyrlx`
- Optional: `sccache` — see [Build cache](#build-cache)

## Build cache

The MLX C++/CUDA build already routes through sccache or ccache; pick which
with `RLX_MLX_CCACHE` (see [`rlx-env-vars.md`](rlx-env-vars.md)). **Rust**
compilation is not cached by default. To enable it, put the wrapper in your
*user* config, `~/.cargo/config.toml` — not in this repo, where a committed
`rustc-wrapper` would be an absolute path on one machine and a missing binary
on the next (this workspace builds on macOS, CUDA, ROCm and aarch64 boxes):

```toml
[build]
rustc-wrapper = "/path/to/sccache"
incremental = false
```

Two rules decide whether that does anything at all. Both are easy to get
wrong silently, because a cache that never hits looks exactly like a cache
that is not installed:

- **`incremental = false` is required, not a preference.** sccache cannot cache
  a crate compiled with `-C incremental`; it forwards the call to rustc, which
  shows up under `non-cacheable reasons: incremental` in `sccache --show-stats`.
  Cargo's dev profile enables incremental by default, so without this the cache
  misses on exactly the builds that run most.
- **Never `export CARGO_TARGET_DIR`.** The environment variable is part of
  sccache's Rust cache key, so varying it misses every single time — while the
  same directory change passed as `--target-dir` hits. Measured 2026-10-16:
  identical crate, dir varied by flag → hit, dir varied by env var → miss. A
  machine that had been running sccache for months through
  `CARGO_TARGET_DIR`-exporting scripts sat at a 0.00% Rust hit rate against 83%
  for C/C++. A *constant* value is fine (`docker/docker-compose.yml` sets one,
  and `[build] target-dir` in a config file is fine too) — it is the varying
  that defeats the cache. `tests/codegen_compile.rs` uses the flag for this
  reason.

`sccache --show-stats` ends with the live cache location and size cap. The cap
is read when the server starts, so run `sccache --stop-server` after changing
it, or wait out the 600 s idle timeout.

## Common commands

```sh
just build          # workspace build
just test           # cargo test (workspace; Darwin caps -j for shared GPU)
just test-gpu       # Metal/MLX/wgpu/Vulkan (+ apple/cuda/rocm runtime) on host

just lint           # clippy (-D warnings)
just fmt            # rustfmt
just fmt-check      # rustfmt --check
just check-no-std   # no_std crates (rlx-ir core, cortexm, fpga) for thumbv7em
just install-git-hooks  # pre-commit auto-fmt + clippy
just ci             # build + tests + fmt-check + clippy + pyrlx pytest
just throttle       # thermal gate before benchmarks
```

Fmt/clippy are gated by git: `just install-git-hooks` installs a pre-commit
hook (`scripts/git-hooks/pre-commit`) that runs `cargo fmt --all`, re-stages
touched staged `.rs` files, then clippy; `just ci` runs `fmt-check` + `lint`.

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

Backend feature matrix: [`crates/bindings/pyrlx/docs/backends.md`](../crates/bindings/pyrlx/docs/backends.md).

Python DSL (`graph` / `Node`, scalar literals): [`crates/bindings/pyrlx/docs/dsl.md`](../crates/bindings/pyrlx/docs/dsl.md).
GGUF helpers: `quantize`, `load_gguf`, `convert_to_gguf` — see [`gguf-backend-paths.md`](gguf-backend-paths.md).
Runnable demo: `python crates/bindings/pyrlx/examples/dsl_quickstart.py` (after `maturin develop`).

Grouped MoE GGUF tests (multi-GPU backends): `just test-gguf-grouped`.

## Multi-backend runtime

Rust helpers live in `rlx-runtime` (`GraphDevices`, `DeviceRouter`, `DevicePolicy`).
Full API reference: [`backend-selection.md`](backend-selection.md).

```sh
cargo run --example graph_devices_demo -p rlx-runtime --features cpu
cargo test -p rlx-runtime --test graph_devices_parity
```

Env vars for pick/fallback: `RLX_DEVICE`, `RLX_DEVICE_CHAIN`, `RLX_DEVICES`,
`RLX_BENCHMARK_PICK` (see backend-selection doc). Curated catalog:
`just env-catalog` (or `cargo run -p rlx-ir --example env_catalog`).
Exhaustive `RLX_*` inventory: [`docs/rlx-env-vars.md`](rlx-env-vars.md)
(`just gen-rlx-env-vars`).

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

Optional executable features (MoE, GPU handles, typed I/O, …) are declared via
`ExecutableGraph::capabilities()` → [`ExecutableCapabilities`]. Method return
values remain authoritative when a backend forgets to flip a bit.

## Diagnosing a device failure

`cargo rlx debug` runs a graph on a device and on CPU with the same
deterministic inputs and params, then classifies the result (`unsupported-op`,
`compile-panic`, `run-panic`, `non-finite`, `mismatch`, …):

```sh
cargo run -p rlx-check --features metal --bin cargo-rlx -- \
    debug graph.json --device metal --probe --bisect
```

- `--probe` reruns with one compile option changed at a time (fusion off,
  regions unfused, F32, common-IR kernels, no constant folding, no MPSGraph on
  Metal). A probe that passes names the subsystem the defect is in.
- `--bisect` exposes every op as an output and names the first one that
  diverges, and whether it reads only graph inputs (suspect upload / layout).
  If nothing diverges once exposed, a fusion the exposure blocked is wrong.

Rust: `rlx_runtime::debug::debug_graph`. `RLX_POISON_ARENA=1` starts CPU
scratch as NaN instead of zeros (at allocation and before each run), so a slot
nothing wrote — an unfed input, an unwritten constant, a no-op op — reads NaN
rather than a plausible zero. GPU host-fallback buffers are not covered.

## Toolchain versions and known issues

External toolchains (mlir-aie + Peano, XRT, CUDA driver/NVRTC, ROCm) change under
us. Three pieces keep that from reaching users first:

- **Version matrix** — `scripts/toolchain_matrix.toml` lists one run per
  (device, toolchain version, rig), each version installed side by side (its own
  venv / toolkit dir). `just toolchain-matrix [--only xdna] [--update]` runs them
  in a fresh `TMPDIR` each (cached kernel builds cannot leak between versions) and
  `--update` records the result in
  `crates/core/rlx-runtime/src/doctor/tested_toolchains.tsv`, which
  `cargo rlx doctor` shows as "tested". To cover a new release: install it next to
  the others, add a `[[run]]`, run with `--update`.
- **Upstream watch** — `just toolchain-upstream` exits 1 when a project in the
  spec has a release newer than anything tested. Run it nightly / before a release.
- **Known issues** — `crates/core/rlx-runtime/src/doctor/known_issues.rs`: every
  resolved setup problem becomes an entry (symptom, cause, fix, and a `detect`
  check over host facts where it can be diagnosed). Each `detect` needs a fake-host
  test in the same file; `just gen-known-issues` regenerates
  [known-issues.md](known-issues.md) (a unit test fails while it is stale). Users
  see matches in `cargo rlx doctor` and paste `cargo rlx doctor --report` into
  issues (template in `.github/ISSUE_TEMPLATE/bug_report.yml`).

## Known failures

A case a backend is known to get wrong goes in
`tests/known_failures/<test>.txt` (`case device[@os] issue [note]`) instead of
a skip: it keeps running, its failure is reported as known, and once it passes
the test fails until the entry is removed. Format and options:
`rlx_ir::known_failures`; decision: [ADR 0005](adr/0005-known-failures-as-data.md).

```sh
just known-failures list
RLX_KNOWN_FAILURES_REPORT=/tmp/kf.jsonl just test-gpu
just known-failures update /tmp/kf.jsonl --dry-run
```

`fd_backward_gate` and `gpu_filters_parity` are wired; another harness opts in
by routing its per-case result through `KnownFailures::record` or `check`.

Fusions that must fire on real block graphs, per registered device, are pinned
by `rlx-runtime/tests/fusions_fire_on_model_blocks.rs` (with producer and
fusion-off controls, so the check cannot pass vacuously).

## `Op::Scan` device preference

GPU / MLX / Vulkan / OneAPI backends prefer **short** Scans as ordinary on-device
ops and keep **long** Scans for the shared host packed body:

| Path | When | Mechanism |
|------|------|-----------|
| On-device IR | `length ≤ scan_unroll_max_length` (default **64**) | `maybe_unroll_scans` / `rlx_maybe_unroll_scans!` |
| On-device IR | `length × body_nodes ≤ 4096` | `maybe_unroll_scans_budget` |
| Host fallback | otherwise | `ScanHostDesc` + D2H/H2D or UM (`rlx_scan_stage_d2h!` / packed f32) |

Set the threshold with `CompileOptions::new().scan_unroll_max_length(n)` (`0`
disables length unroll). Nested AD uses `Op::ScanBackward*` via the shared
[`HostOpDesc`](../crates/backends/rlx-cpu/src/thunk/host_op.rs) contract
(mirrors `ScanHostDesc`): `rlx_host_op_desc!` / `rlx_execute_host_op_on_bytes!`
/ `rlx_host_op_stage_d2h!`, plus value-map helpers `run_scan_node_f32` /
`run_host_op_node_f32`. Discrete GPUs share `rlx_arena_stage_d2h!` for Scan /
HostOp / Spd staging. wgpu rebases with `ScanHostSpan` / `HostOpSpan`.

**On-device Scan:** only via IR unroll (body as ordinary kernels). There is no
nested Metal/CUDA body scheduler / body-ISA interpreter — long Scans stay on
the host packed loop.

Parity: `scan_unroll_parity`, `scan_backward_parity`,
`gpu_filters_parity::…::iirfilt`.

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

## GPU backend source layout

Large compile/run surfaces are split for navigation (public APIs unchanged):

| Crate | Layout |
|-------|--------|
| `rlx-metal` | `backend/mod.rs` (`MetalExecutable`); `backend/encode/{mod,ops}.rs` |
| `rlx-cuda` / `rlx-rocm` | `backend/{mod,step,helpers}.rs` (+ CUDA `bwd_launch.rs`); `compile` / `run` / … |
| `rlx-wgpu` | `backend/{mod,step,helpers}.rs`; `compile/{mod,lower}.rs` |
| `rlx-mlx` | `lower/{mod,env,subgraph,helpers}.rs` (`lower_with_env` in `env.rs`) |

Host-staged ops shared across discrete GPUs: `rlx-gpu-host`.

## Adding an op

1. `rlx-ir` — `Op`, inference, verifier
2. Every backend that should run it — thunk / executor, and that crate's
   `SUPPORTED_OPS` (`crates/backends/rlx-*/src/supported_ops.rs`)
3. `rlx-fusion` / `rlx-compile` if fusion or legalization applies
4. Parity test in `rlx-runtime/tests/` or downstream
5. `just gen-op-coverage` (or `python3 scripts/gen-op-coverage.py`) so
   [`docs/op-coverage.md`](op-coverage.md) stays in sync

## Calibration caches

GPU backend ranking uses on-disk JSON under `~/.cache/rlx/` (`*-calib-*.json`).
Delete a file to force re-measurement. See [`backend-selection.md`](backend-selection.md).

## Android

JNI demo app: [`android/README.md`](../android/README.md).

```sh
just android-check          # cargo check for aarch64-linux-android (+ rlx-jni)
just android-build          # NDK cross-build → app/src/main/jniLibs/arm64-v8a/
just android-build -- --blas  # after ./android/build-openblas.sh
just android-e2e            # emulator + instrumented tests
```

The `android` / `android-blas` features on `rlx` / `rlx-runtime` enable `cpu` +
`gpu` (wgpu/Vulkan). OpenBLAS is optional via `OPENBLAS_LIB_DIR` (NDK has no
system CBLAS).

## Docs index

| Path | Contents |
|------|----------|
| [`docs/README.md`](README.md) | Doc index |
| [`docs/backend-selection.md`](backend-selection.md) | Multi-backend API |
| [`docs/benchmarks/higher-order-ad.md`](benchmarks/higher-order-ad.md) | HO AD benches |
## License

MIT OR Apache-2.0.
