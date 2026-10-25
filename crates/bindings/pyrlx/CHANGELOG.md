# Changelog

All notable changes to `pyrlx` will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.19] — 2026-10-24

### Added

- **`Group` — distributed nodes** (`connect`, `from_env`, `send` / `recv`,
  `all_reduce`, `all_gather`, `broadcast`, `barrier`, `collect_caps`,
  `ship_stage` / `serve_worker`, `send_activation` / `recv_activation`,
  `ship_train` / `run_train` / `serve_trainer`) and `loopback_peers`. Every
  blocking call releases the GIL, so a local mesh can be Python threads.
- `Graph.clone()` / `Graph.is_consumed()`, `load_weights(format, path)` (also
  `"safetensors"`), `to_rlxp(from_, src, dst, options)`,
  `GraphDevices.resolve` / `run_on` / `set_params`, `DeviceRouter.set_params` —
  the rows the shared runtime API table (`rlx-bind/src/api.idl`) found missing;
  `tests/test_api_contract.py` now checks the module against that table both
  ways.
- Five runtime examples — `examples/{inference,training,loading,distributed,
  backends}.py` — the same five as the JS and Swift bindings, run by
  `tests/test_examples.py`.

- **The rest of the `rlx_ir::Graph` builder surface** (~140 methods, now ~200
  in all): conv 1d/3d and transposed variants, `im2col`, pooling,
  interpolation, `attention_bias` / `attention_kind_vdim` / `attention_opts`
  (score scale, logit softcap), `rope(style="gptj")`, `axial_rope2d`, batch /
  adaLN / fused residual norms, LSTM / GRU / RNN, selective scan, Mamba-2,
  Gated DeltaNet, `scan` over a body graph, `custom_fn_multi`, `custom_op`,
  `expand`, scatter / gather ND, `pad`, `slice`, `tile`, `trilu`, `roll`,
  ordering ops, Cholesky / QR / SVD / eigh / det / triangular solve,
  per-channel and block-scaled quantization, `dequant_matmul` and its packed /
  grouped / NVFP4 forms, `q_matmul` / `q_conv2d`, `synth_matmul`, VQ and
  residual VQ, KAN splines, N-D FFT, FIR / IIR / biquad / SOS filters, Hilbert,
  resampling, spectrogram / band power / log-mel, SPD manifold layers, and
  `param_names` / `op_kinds` / `outputs`. Output shapes are inferred.
- `check` / `check_backends` — device-free static diagnostics.
- Precision policies: `Session(policy=...)`, `compile_with(policy=...)`,
  `FlexibleSession.compile_with_resolved(policy=...)`.
- `CompiledGraph.set_params`, `output_shapes`, `output_dtypes`;
  `FusionOptions.disable_conv_bias_act_fusion`; `verify_rlxp`.
- `Trainer` / `Optimizer` (feature `training`) and `load_tokenizer` /
  `Tokenizer` / `Sampler` / `sample_next` / `argmax` / `render_chat`
  (feature `text`), ported from rlx-js. Trainer checkpoints use rlx-js's GGUF
  layout, so either binding resumes the other's.

### Changed

- Aligned with RLX workspace **0.2.19**.
- The `Graph` builders and the runtime (`Session`, `CompiledGraph`, `Trainer`,
  `Optimizer`, tokenizer / sampler, `check`, GGUF, RLXP, weight loaders) are
  now generated from / delegated to `rlx-bind`, the core rlx-js and the C ABI
  share. Every public name and signature is kept; a trailing options group
  (`conv2d`'s `kernel_size` … `groups`, `sample`'s `top_k` … `dtype`) now has
  defaults, and pair options accept an int (`stride=2`).
- `CompiledGraph.run` refuses non-f32 outputs instead of reinterpreting their
  bytes; use `run_typed`, or `cast` to f32 in the graph.
- A Rust panic inside any call raises `RuntimeError` (it was pyo3's
  `PanicException`, which `except Exception` does not catch).
- `dequant_gguf` handles every GGUF type (F16 / BF16 were refused);
  `GgufFile.metadata()` returns arrays as lists (shortened past 64 entries —
  `metadata_value(key)` returns one in full) rather than a debug string.
- New spellings from the shared table: `ifft`, `le_`, `node_count`, `name`,
  `fused_residual_layer_norm` (`fused_residual_ln` stays).

- Default cargo features now include `training` and `text`, matching rlx-js.
- `activation` / `binary` read `Activation::ALL` / `BinaryOp::ALL`, so all 29
  activations and 14 binary ops (`softplus`, `erf`, `mish`, `atan2`, `mod`,
  bitwise, shifts, …) are reachable by name; before, 13 and 7 were.

### Fixed

- The `.pyi` stubs typed `grad` / `jvp` / `hvp`'s `wrt` as names and gave
  `nth_order_grad` / `directional_nth_grad` the wrong parameter order; they
  take node ids, and `(graph, wrt_name, order)` respectively.
- `load_rlxp`, `convert_*_to_rlxp`, `load_mlx` / `load_dduf` / `load_nemo` /
  `load_pt` and (with `fpga`) `export_fpga` were compiled into the extension
  but never re-exported, so `rlx.load_rlxp` — as the README documents it —
  raised `AttributeError`.

## [0.2.18] — 2026-10-15

Covers 0.2.17 as well — `pyproject.toml` had stayed at 0.2.16 there.

### Added

- `export_fpga` accepts `quant="ternary"` (BitNet b1.58) and
  `hw="xc7:BOARD"` (openXC7: `arty-a7-35`, `arty-a7-100`, `basys3`,
  `nexys-a7-100`).

### Changed

- Aligned with RLX workspace **0.2.18**.

## [0.2.16] — 2026-09-15

Covers 0.2.15 as well — neither went out to PyPI.

### Fixed

- **A GGUF dtype name pyrlx printed could not be handed back to it.** The
  dtype-name parser kept its own table, and it had drifted from the printing
  counterpart: it rejected seven names `ggml_type_name` emits, so the obvious
  round-trip failed on those tensors. It now delegates to `GgmlType::from_name`,
  so there is one table instead of two.

### Changed

- Aligned with RLX workspace **0.2.16**.

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
## License

MIT OR Apache-2.0.
