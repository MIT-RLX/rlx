# rlx-ir

Tensor IR for the RLX ML compiler — standalone, serializable, optimizable.

The **leaf** of the workspace dep graph — adding deps here ripples
through every backend. `rlx-ir` owns:

- **`DType`** — F32, F16, BF16, F64, I8, I16, I32, I64, U8, U32, Bool, C64.
  Promotion rules.
- **`Shape`** — static or dynamic dimensions, dtype-aware. No symbolic
  dims yet.
- **`Op`** — the IR vocabulary. JAX-shaped: `MatMul`, `DotGeneral`, `Scan`,
  `SelectiveScan`, `DenseSolve`, `Fft`, `LayerNorm`, `Attention` (with
  `MaskKind`), `Quantize` / `Dequantize` / `FakeQuantize`, `If` / `While`,
  `Custom` / `CustomFn` extension points, plus the full reverse-mode AD
  op set (`*Backward` variants). Adding an op means: this file,
  `infer.rs`, `graph.rs` builder, both backends' thunks, MPSGraph
  lowering, fusion patterns, cost model. ~6 files per new op.
- **`fft`** — shared `FftNorm`, GPU launch plans, and NumPy-shaped helpers
  (`fftfreq`, `gpu_fft_native_eligible`, …). Graph builders in
  `ops::fft_ops` compose `rfft`, `irfft`, `stft`, `psd_real`, and
  `fft_conv1d` from primitive ops.
- **`MaskKind`** — attention mask kind (None / Causal / SlidingWindow /
  Custom). Added Apr 2026 per plan #20.
- **`Graph`** — DAG builder + topo iterator.
- **`infer.rs`** — shape inference per op. Drives fusion and arena
  planning.
- **`verify.rs`** — graph-level invariant checks. Run after every
  optimizer pass in debug builds (plan #50).
- **`OpExtension` / `op_registry`** — the custom-op scaffold downstream
  crates register against (see [`rlx-sparse`], [`rlx-linalg`]).
- **Logical kernels** — one `OpKind`, native or common-IR lowering; see
  [README-logical-kernels.md](README-logical-kernels.md).
- **`Tick`** — `CNTVCT_EL0`-direct cycle counter for sub-millisecond
  measurement on Apple Silicon. Falls back to `Instant`. Use
  `Tick::now()` / `tick.elapsed_ns(start)` for any sub-ms measurement.

## Install

```toml
[dependencies]
rlx-ir = "0.2"
```

Most users should depend on the [`rlx`](https://crates.io/crates/rlx)
prelude crate instead, which re-exports `rlx_ir` as `rlx::ir`.

## IR levels (HIR / MIR / LIR)

| Level | Module | Role |
|-------|--------|------|
| **HIR** | `rlx_ir::hir` | Block ops for model builders (`SwiGLU`, `Linear`, …) |
| **MIR** | `rlx_ir::mir` | Fused tensor DAG — optimizer input (`Graph` alias) |
| **LIR** | `rlx_ir::lir` | MIR + arena buffer plan — backend input |

[`GraphModule`] tracks which stage you are in. Use [`Graph::define`] for
fusion-first HIR builders; use [`GraphModule::mir`] for primitive MIR.

Pipeline: `CompilePipeline::compile_module(module)` in `rlx_opt`.

## Quickstart

```rust
use rlx_ir::{DType, Graph, GraphModule, Shape};

// Fusion-first (recommended)
let module = Graph::define("layer", |m| {
    let x = m.input("x", Shape::new(&[2, 128], DType::F32));
    let w = m.param("w", Shape::new(&[128, 128], DType::F32));
    m.linear(x, w, None, None, Shape::new(&[2, 128], DType::F32))
});

// Primitive MIR (legacy)
let mut g = Graph::new("hello");
let x = g.input("x", Shape::new(&[1, 4], DType::F32));
let w = g.param("w", Shape::new(&[4, 2], DType::F32));
let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
g.set_outputs(vec![y]);
let module = g.module();
```

## Build / test

```sh
cargo build -p rlx-ir
cargo test  -p rlx-ir       # 11 unit tests
```

## Status

The `Op` enum is stable in shape but not in surface — minor 0.x bumps
may add variants or refine attributes. Pin exact versions in production
until 1.0.

## `QuantScheme` and `Op::DequantMatMul`

`rlx_ir::quant::QuantScheme` describes how a packed weight tensor is
laid out so the dequant kernel can decode it without a side lookup.
Variants:

| Variant | Block size | Bits/elem (×10) | Notes |
|---|---|---|---|
| `Int8Block { block_size }` | configurable | 80 | Symmetric INT8, GPTQ-style |
| `Int8BlockAsym { block_size }` | configurable | 80 | + per-block zero-point |
| `Int4Block { block_size }` | configurable | 40 | INT4 packed two-per-byte |
| `Fp8E4m3`, `Fp8E5m2` | n/a | 80 | per-tensor FP8 |
| `GgufQ4K`, `GgufQ5K`, `GgufQ6K`, `GgufQ8K` | 256 | 45 / 55 / 66 / 91 | llama.cpp K-quant super-blocks; scales / mins live inside the packed bytes |

`Op::DequantMatMul { scheme }` takes 4 inputs for the legacy Int8
schemes (`x`, `w_q`, `scale`, `zp`) or 2 for the GGUF schemes (`x`,
`packed_w_bytes`) — `num_inputs()` switches on `scheme.is_gguf()`.

The CPU backend handles all listed schemes today; Metal lowering for
the GGUF schemes is on the roadmap (`Op::DequantMatMul` falls through
to the per-op thunk path, which dequants the weight to F32 scratch
once before matmul — correct but doesn't keep packed bytes on the
GPU).

## Gotchas

- `Op::Attention` input count is variable: `MaskKind::Custom` → 4
  inputs (Q, K, V, mask); other kinds → 3. `num_inputs()` knows.
- `Op::DequantMatMul` input count is variable too: 4 for legacy Int8,
  2 for GGUF K-quant schemes (see table above).
- `Op` is not `Copy` (some variants own `Vec`/`Box<Graph>`). Match
  destructures must use `..` for non-Copy fields.
- `MaskKind` IS `Copy`; `*mask_kind` is fine inside a match arm bound
  to `&MaskKind`.
- Don't add backend-flavored types here — keep `rlx-ir` portable.

## License

GPL-3.0-only.
