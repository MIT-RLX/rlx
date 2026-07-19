# rlx-coreml

Apple **CoreML / Neural Engine (ANE)** backend for RLX.

GGUF on-device dequant, hybrid host segments, and env toggles:
[docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md) (ANE section).

It lowers an RLX IR graph to a CoreML **ML Program** (the MIL dialect),
serialises it into a `.mlpackage`, and runs it through `CoreML.framework`.
CoreML's planner then schedules each op across the CPU, GPU, and Neural
Engine.

## Layout

| path | role |
|------|------|
| `proto/coreml.proto` | focused subset of Apple's CoreML protobuf schema (exact field numbers) |
| `src/mil/` | IR → MIL `Program` lowering (host-portable, no FFI) |
| `src/mlpackage.rs` | `.mlpackage` bundle writer |
| `src/hybrid.rs` | segment-based host + CoreML execution planner |
| `src/host_exec.rs` | host ops (FFT, RNN, sampling, custom kernels, …) |
| `src/op_registry.rs` | `Op::Custom` kernel registry |
| `src/chip.rs` | ANE / chip introspection (`ane_available`, `chip_info`) |
| `csrc/coreml_shim.m` | Objective-C bridge over `CoreML.framework` (compiled by `build.rs`) |
| `src/ffi.rs`, `src/backend.rs` | execution (Apple platforms only) |

MIL emission and `.mlpackage` writing are pure Rust and build on every
host. Only execution is gated behind `target_os = "macos"`/`"ios"`.

## Usage

```rust
use rlx_runtime::{Device, Precision, Session};

// F16 via Session precision
let session = Session::new_with_precision(Device::Ane, Precision::F16);

let mut compiled = session.compile(graph);
compiled.set_param("W", &weights);
let out = compiled.run(&[("x", &input)]);
```

Custom op on ANE (host segment):

```rust
rlx_coreml::register_coreml_kernel(Arc::new(MyKernel));
```

Environment knobs:

| Variable | Effect |
|----------|--------|
| `RLX_COREML_UNITS` | `cpu` / `gpu` / `all` / `ane`. **Default (unset):** fp32 graphs → CPU+GPU; f16 graphs → CPU+Neural Engine. Prefer `gpu` for large imported ONNX (TTS, etc.) — Neural-Engine BNNS AOT can SIGSEGV on multi-k-node ML Programs. |
| `RLX_COREML_HOST_DEQUANT=1` | bake full f32 weights at compile (legacy path) |
| `RLX_COREML_FLEXIBLE_INPUTS=1` | emit CoreML `ShapeRange` on dynamic inputs |
| `RLX_COREML_NATIVE_FLEX=1` | ANE: one model + runtime shapes (skip `DeferredExecutable`) |

### Compute units (BNNS note)

`Device::Ane` always goes through CoreML; which silicon runs the graph is
`MLComputeUnits`. Defaulting fp32 inference to **CPU+GPU** avoids a known Apple
crash: `bnns::GraphCompile` null-deref on large Shape-heavy programs (LuxTTS
`encoder_body` / `fm_decoder`, similar VITS/CFM graphs). Force Neural Engine with
`RLX_COREML_UNITS=ane` or by compiling an f16 graph (AFP). See
[`default_compute_units`](src/backend.rs).

## Status

**65** op kinds declared in `SUPPORTED_OPS` — the complete forward
inference surface for transformer + vision + MoE + quantized + SSM graphs.

### MIL-lowered ops

Element-wise, matmul, norms, attention (causal / bias / sliding window),
vision (conv / pool / resize), MoE `grouped_matmul`, SSM (`selective_scan`,
`gated_delta_net`), quantized `dequant_*` (with on-device block dequant for
Q8_0 / Q4_0 / IQ4NL), `quantize` / `dequantize`, reductions, gather, rope,
and the rest of the primitive set listed in `rlx_coreml::SUPPORTED_OPS`.

On-device constexpr dequant also covers **K-quants** Q4_K / Q5_K / Q8_K and
**Q2_K / Q3_K / Q6_K** (`mul` + optional `sub`; Q2/Q3/Q6 use per-element
`[nb,32]` scale/offset tensors when sub-block scales vary within a 32-chunk).
Legacy **Q4_1** (scale + min per block) is included alongside Q4_0 / Q8_0 /
IQ4NL.

### Host segments (hybrid runner)

Ops with no stable MIL lowering run on CPU between CoreML segments:

- `fft`, `log_mel`, `welch_peaks`
- `sample`, `rng_normal`, `rng_uniform`
- **native RNN / SSM:** `lstm`, `gru`, `rnn`, `mamba2` (via `rlx_cpu` reference kernels)
- `custom` (via `register_coreml_kernel`)

`gru` / `rnn` / `lstm` avoid unfusing to huge MIL graphs.

### P1–P5 infrastructure (2026-06)

- **FP16:** `LowerOptions::float_dtype`, f16 blob weights, f16 CoreML I/O in shim, `Precision::F16` via `CompileOptions`
- **Flexible shapes:** MIL `UnknownDimension` + `ShapeRange`; runtime shape inference at predict; optional `RLX_COREML_NATIVE_FLEX=1` skips deferred recompile on ANE
- **On-device dequant:** Q8_0 / Q4_0 / IQ4NL / Q4_K / Q5_K / Q8_K in MIL; MoE grouped matmul too
- **Custom registry:** `op_registry.rs` + hybrid dispatch

K-quants and IQ/TQ/MX/NV families without a MIL `mul`+`sub` lowering still
**host-dequant** to f32 at bake time (hybrid segment or
`RLX_COREML_HOST_DEQUANT=1`).

Weights with ≥ 10 elements go to `weights/weight.bin` (MILBlob format).

Introspection: `chip_info()`, `ane_available()`, `MLComputePlan` routing
(macOS 14.4+).

### Not in scope for ANE (inference claim)

Control flow beyond host `Scan*`, fusion internals (`ElementwiseRegion`),
`CustomFn`, `QMatMul` / `QConv2d` (int8 I/O), Gaussian splat family.

Under the `training` feature, packed / primitive `*Backward` ops are claimed for
device selection via `COREML_BACKWARD_OPS` (decompose to MIL primitives) or
`NATIVE_BACKWARD_OPS` (dedicated MIL arms). DiT modulation forward
(`AdaLayerNorm` / `GatedResidual`) and packed reverse
(`AdaLayerNormBackward` / `GatedResidualBackward`) use native composed MIL
(implicit broadcast + concat pack for reverse).

### Output layout note

CoreML may pad rank-4 outputs for ANE alignment; the shim copies via
stride-aware indexing, not flat `memcpy`.

## License

GPL-3.0-only.
