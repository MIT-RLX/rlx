# Changelog

All notable changes to `pyrlx` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — initial release

### Added
- `pyrlx.available_devices()` / `pyrlx.is_available()` — query the
  build's registered backends.
- `pyrlx.Graph` — minimal Python graph builder over `rlx_ir::Graph`
  (input / param / matmul / binary / activation / set_outputs).
- `pyrlx.Session(device, precision)` + `pyrlx.CompiledGraph` — the
  hot-path execution surface, with NumPy I/O.
- `pyrlx.Embed` — load BERT / NomicBERT / NomicVision and run on any
  registered backend (gated on the `embed` cargo feature).
- Type stubs (`_pyrlx.pyi`) and `py.typed` marker for full IDE
  autocompletion + mypy support.
- Build via `maturin` or `uv pip install -e . --config-settings=build-args=...`.
- Backend selection via cargo features mirroring `rlx-runtime`:
  `cpu`, `blas-accelerate`, `blas-mkl`, `blas-openblas`, `metal`,
  `mlx`, `gpu` (wgpu), `cuda`, `rocm`, plus `embed` / `hf-download`.

[Unreleased]: https://github.com/anthropics/rlx/compare/pyrlx-v0.1.0...HEAD
[0.1.0]:      https://github.com/anthropics/rlx/releases/tag/pyrlx-v0.1.0
