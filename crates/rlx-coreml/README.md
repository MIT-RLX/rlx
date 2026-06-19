# rlx-coreml

Apple **CoreML / Neural Engine (ANE)** backend for RLX.

It lowers an RLX IR graph to a CoreML **ML Program** (the MIL dialect),
serialises it into a `.mlpackage`, and runs it through `CoreML.framework`.
CoreML's planner then schedules each op across the CPU, GPU, and Neural
Engine.

## Layout

| path | role |
|------|------|
| `proto/coreml.proto` | focused subset of Apple's CoreML protobuf schema (exact field numbers) |
| `src/mil.rs` | IR → MIL `Program` lowering (host-portable, no FFI) |
| `src/mlpackage.rs` | `.mlpackage` bundle writer |
| `src/chip.rs` | ANE / chip introspection (`ane_available`, `chip_info`) |
| `csrc/coreml_shim.m` | Objective-C bridge over `CoreML.framework` (compiled by `build.rs`) |
| `src/ffi.rs`, `src/backend.rs` | execution (Apple platforms only) |

MIL emission and `.mlpackage` writing are pure Rust and build on every
host. Only execution is gated behind `target_os = "macos"`/`"ios"`.

## Usage

```rust
use rlx_runtime::{Device, Session};

let mut compiled = Session::new(Device::Ane).compile(graph);
compiled.set_param("W", &weights);
let out = compiled.run(&[("x", &input)]);
```

or directly:

```rust
let mut exe = rlx_coreml::CoremlExecutable::compile(graph);
exe.set_param("W", &weights);
let out = exe.run(&[("x", &input)])?;
```

## Status

Ops lowered to MIL today:

- `const` / `param` (weights baked as inline immediate values)
- `matmul`
- element-wise binary: `add` / `sub` / `mul` / `real_div` / `maximum` /
  `minimum` / `pow`
- activations: `relu` / `sigmoid` / `tanh` / `gelu` (exact + tanh-approx) /
  `silu` / `exp` / `log` / `sqrt` / `rsqrt` / `abs` / `neg` / `sin` /
  `cos` / `tan` / `atan` / `round`
- `softmax`, `layer_norm`, `rms_norm`, `group_norm`, `layer_norm_2d`,
  `batch_norm` (inference) — composed where MIL lacks a native op
- `rope` (NeoX split-halves) + `axial_rope_2d` (SAM2-style, baked tables)
- `attention` (causal / bias / none masks, scale, logit softcap) composed
  from primitives — a full transformer block runs end-to-end
- `compare` (all 6), `where`/select, `expand`, `cumsum`, `scatter_add`,
  `top_k`, `lora_matmul`, `stop_gradient`
- MoE: `grouped_matmul` (gather-then-batched-matmul)
- quantized weights: `dequant_matmul`, `dequant_grouped_matmul` (MoE),
  `dequant_moe_weights` — GGUF schemes Q8_0 / Q4_0 / Q2_K / Q3_K / Q4_K /
  Q5_K / Q6_K / Q8_K, **host-dequantized to f32 at model-build time**
- `quantize` / `dequantize` (per-tensor & per-channel int8 fake-quant)
- SSM (recurrent, unrolled over the sequence): `selective_scan` (Mamba),
  `gated_delta_net` (Qwen3.5) — verified against the CPU backend
- vision: `conv`, `conv_transpose_2d`, `pool` (max/avg), `resize_nearest_2x`
  — NCHW
- `reduce` (sum/mean/max/min/prod), `concat`, `gather`, `narrow`, `cast`
- `reshape`, `transpose`

That's **42 of the IR's 106 op kinds** — the complete inference compute
surface (transformer + vision + MoE + quantized + SSM + elementwise).

GGUF-quantized weights arrive via `set_param_typed` (raw bytes) and are
dequantized on the host when the `.mlpackage` is baked. This makes
quantized models *run*; the proto then carries f32 weights (on-device
`constexpr` dequant is a later size optimization).

Weights with ≥ 10 elements are written to `weights/weight.bin` in CoreML's
**MILBlob format** and referenced from the proto by offset; smaller consts
stay inline. This keeps the protobuf small enough for CoreML to parse even
for LLM-scale models (an all-inline proto would blow past its limits).

If the serialized `model.mlmodel` still exceeds protobuf's ~2 GiB message
cap (too many ops / large inline constants), packaging fails fast with an
explicit `CoremlError::TooLarge { bytes, limit }` rather than a downstream
CoreML parse error. `mlpackage::check_model_size` exposes the check.

Introspection: `chip_info()`, `ane_available()`, and per-op device
routing via `MLComputePlan` (macOS 14.4+) on the loaded model.

### Not yet implemented (extension surface)

The 64 remaining op kinds are **not** a tail of inference compute — every
op a model executes during a forward pass is now covered. What's left is
categorically different work:

- **Training / backward (~30)** — every `*Backward*` op, `SoftmaxCross
  Entropy*`, the `FakeQuantize` (QAT) family. CoreML is an inference
  runtime; these cannot be lowered to a static prediction model.
- **Fusion-internal (~9)** — `Fused*`, `ElementwiseRegion`,
  `TransformRegion`. Created only by the fusion pass, which CoreML doesn't
  run, so the backend never sees them (it lowers the primitives instead).
- **Host / runtime-loop** — `Sample` + `Rng*` (decode loop), control flow
  (`If`/`While`/`Scan`), `Custom`/`CustomFn` (need a CoreML op registry).
- **Int8-activation matmul** — `QMatMul`, `QConv2d`. Lowerable in principle
  (fp32-integer + host weight dequant), but their I/O is int8, which can't
  be a CoreML model input *or* output — only usable buried inside a fully
  int8 `Quantize→…→Dequantize` subgraph (a niche onnx-static-quant path;
  the common quant path is GGUF weight-quant, which is done).
- **No reference / no MIL op** — `DotGeneral` (the CPU backend itself
  leaves it unimplemented), `Fft` (no MIL FFT), `DenseSolve` (no MIL
  solver), `Im2Col` (a conv-internal detail), domain ops (`GaussianSplat*`,
  `LogMel`, `WelchPeaks`).

Other limits: GGUF dequant covers the 8 k-quant schemes wired in the
dispatcher (IQ / ternary / microscaling fall through) and is
host-dequantized to f32 (no on-device `constexpr` dequant yet); attention
is rank-4 `[B,H,S,D]` only; RoPE requires last-dim == `head_dim`; SSM ops
are unrolled (static seq); input shapes must be static.

### Note on output layout

CoreML pads the inner dimension of rank-4 outputs for ANE alignment, so
output `MLMultiArray`s can be non-contiguous; the shim copies them via
logical (stride-aware) indexing rather than a flat `memcpy`.
