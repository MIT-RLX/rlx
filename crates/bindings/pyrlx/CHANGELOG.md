# Changelog

All notable changes to `pyrlx` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.14] — 2026-07-24

- Aligned with RLX workspace **0.2.14**.


## [0.2.12] — 2026-07-06

### Changed

- Dependency pins aligned with RLX workspace **0.2.12**.

## [0.2.11] — 2026-07-05

### Added

- **GGUF Python surface** — `quantize` / `dequant`, `load_gguf` / `write_gguf`,
  `convert_to_gguf` (safetensors → GGUF), and `GgufFile` metadata helpers.
- Cargo features `gguf-convert` (default), `gguf-onnx`, `gguf-pt` for optional
  checkpoint readers.

### Changed

- Dependency pins aligned with RLX workspace **0.2.11**.

## [0.2.7] — 2026-06

### Changed

- Dependency pins aligned with RLX workspace **0.2.7** (in-graph RNG via
  upstream `rlx-runtime` / `rlx-ir`; ONNX `Random*` import in `rlx-onnx-import`).

## [0.2.6] — 2026-06

### Changed

- Dependency pins aligned with RLX workspace **0.2.6** (MLX GELU-approx and Metal
  MPSGraph GELU parity fixes from upstream `rlx-mlx` / `rlx-metal`).

## [0.2.4] — 2026-06-08

### Added
- **DSL** — `pyrlx.graph()` context manager, `pyrlx.Node` operator overloads
  (`+`, `-`, `*`, `/`, `@`, `**`, comparisons), scalar literal promotion
  (`x * 2.0`), and `g.outputs = [...]` sugar. Implementation lives in
  `python/pyrlx/dsl.py`; see `docs/dsl.md`.
- **`Graph.constant`** — broadcastable rank-0 literals; validated on the
  Python binding before IR insertion (via `GraphExt::try_constant`).
- **Graph ops** — `gelu_approx`, `stop_gradient`, `conv2d`, `conv_transpose2d`,
  `layer_norm2d`, `group_norm`, `rope_n` on the native `Graph` surface.
- **Comparison shortcuts** — `Graph.gt_`, `Graph.ge_`, `Graph.ne_` alongside
  `eq_` / `lt_`.
- **`rlx-ir::GraphExt::try_constant`** — fallible scalar literal builder for
  Rust callers.
- **`examples/dsl_quickstart.py`** — minimal runnable DSL example.
- **`py.typed`** marker for PEP 561 typed-package metadata.
- Integer literal OOB rules: range, integrality, float-on-int-dtype promotion,
  rejection of `abs(int) > 2**53`.
- Expanded tests: DSL execution parity for compare/where, OOB guards,
  `test_ir_parity.py`, `test_bindings.py`, `rlx-runtime/tests/cpu_scalar_constant.rs`.

### Changed
- Type stubs (`_pyrlx.pyi`) synced with the full native binding surface.
- `jacfwd` docstring updated — `vmap` is available; use it for large Jacobians.
- Module docs in `python/pyrlx/__init__.py`, `src/graph.rs`, and `rlx` prelude
  document scalar literals and the DSL entry points.
- `Graph.constant` Python binding delegates to `GraphExt::try_constant`.

### Removed
- Stale `pyrlx.Embed` mention from the 0.1.0 notes below — that API was never
  shipped in this crate.

## [0.2.3]

### Added
- Multi-backend runtime: `GraphDevices`, `DeviceRouter`, `DevicePolicy`,
  `FlexibleSession`, `backends_manifest()`.
- FKL fusion: `FusionOptions`, `Session.compile_with`, `batch_narrow_relu_graph`.
- FFT helpers, autodiff transforms (`grad`, `jvp`, `hvp`, `vmap`,
  `nth_order_grad`), typed I/O (`run_typed`, `set_param_typed`).
- Dtype-aware Python helpers: `pyrlx.set_param`, `pyrlx.run`, `numpy_dtype`,
  `dtype_str`.

## [0.1.0] — initial release

### Added
- `pyrlx.available_devices()` / `pyrlx.is_available()` — query the
  build's registered backends.
- `pyrlx.Graph` — Python graph builder over `rlx_ir::Graph`.
- `pyrlx.Session(device, precision)` + `pyrlx.CompiledGraph` — compile +
  execute with NumPy I/O.
- Build via `maturin` or `uv pip install -e . --config-settings=build-args=...`.
- Backend selection via cargo features mirroring `rlx-runtime`:
  `cpu`, `blas-accelerate`, `blas-mkl`, `blas-openblas`, `metal`,
  `mlx`, `gpu` (wgpu), `cuda`, `rocm`.

[Unreleased]: https://github.com/MIT-RLX/rlx/compare/pyrlx-v0.2.14...HEAD
[0.2.14]:     https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.14
[0.2.13]:     https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.13
[0.2.12]:     https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.12
[0.2.11]:     https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.11
[0.2.7]:      https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.7
[0.2.6]:      https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.6
[0.2.4]:      https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.4
[0.2.3]:      https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.2.3
[0.1.0]:      https://github.com/MIT-RLX/rlx/releases/tag/pyrlx-v0.1.0
