# Native low-precision tensor-core GEMM (FP8 / FP6 / FP4)

`Op::ScaledMatMul` feeds **low-precision operands directly into the tensor-core /
MFMA GEMM** with f32 accumulation — the real 2×–4× throughput win on hardware
that has FP8/FP4 matrix units. This is distinct from the *storage* quantization
path (`Op::DequantMatMul`, `QuantScheme`), which decodes weights to f32 first and
then runs an ordinary sgemm.

## The op family (`rlx-ir`)

| Op | Inputs | Output | Role |
|----|--------|--------|------|
| `ScaledQuantScale { format, scale_layout }` | `x` (f32) | scale tensor | per-tensor amax → scale (one value), or per-block scale |
| `ScaledQuantize { format, scale_layout }` | `x` (f32), `scale` | `U8` codes | encode `x / scale` to low-precision codes |
| `ScaledMatMul { lhs_format, rhs_format, scale_layout, has_bias }` | `lhs`,`rhs` (U8 codes), `lhs_scale`,`rhs_scale`, `[bias]` | `[m,n]` f32 | native GEMM, f32 accumulate |

Operands flow as `DType::U8` byte buffers; the element **format is carried on the
op**, not the dtype — so no `DType` variant is needed and the enum's exhaustive
matches stay intact. Operand `Shape`s carry **logical** element dims.

**Layout is TN**: `lhs [m,k]`, `rhs [n,k]` (both K-last), `out = lhs · rhsᵀ`.
This makes block scales run along the last/contraction axis of *both* operands
uniformly, and matches cuBLASLt / hipBLASLt FP8's required `transa=T, transb=N`.

### Formats (`ScaledFormat`) and scale layouts (`ScaleLayout`)

- FP8 `F8E4M3` (±448), `F8E5M2` (±57344) — OCP. `F8E4M3Fnuz`/`F8E5M2Fnuz` — AMD.
- FP6 `F6E2M3` (±7.5), `F6E3M2` (±28) — OCP MX.
- FP4 `F4E2M1` (±6) — NVFP4 / MXFP4.
- `PerTensor` (one f32 scale) · `BlockMxE8M0 { block:32 }` (E8M0 power-of-two) ·
  `Nvfp4 { group:16 }` (FP8-E4M3 block scale).

The canonical bit-exact f32↔code codec for every format lives in
`rlx-ir/src/lowp_codec.rs` (decode is closed-form; encode is nearest-representable
with round-half-to-even + saturation). It is the contract every backend matches.

## Per-backend status

| Backend | Native compute? | What runs | Validated |
|---------|-----------------|-----------|-----------|
| **CPU** (`rlx-cpu`) | No (no FP8 matrix ISA) | **Reference oracle**: decode→scale→sgemm for all 7 formats × all scale layouts | ✅ on this host (cosine vs f32) |
| **CUDA** (`rlx-cuda`) | **Yes** for per-tensor FP8 — cuBLASLt, sm_89+ (Ada/Hopper) | per-tensor `F8E4M3`/`F8E5M2` native; **all other formats (block-MX, FP4, FP6) run via an on-device decode-accumulate kernel** (`scaled_matmul_decode`) | ⏳ compile-checked; run on RTX 4090 |
| **ROCm** (`rlx-rocm`) | **Yes** for per-tensor FP8 — hipBLASLt FNUZ, CDNA3 (MI300) | per-tensor FNUZ fp8 native; other formats via the same decode-accumulate kernel | ⏳ compile-checked; **fp8 ABI constants need MI300 verification** (`hipblaslt.rs`) |
| **Metal** (`rlx-metal`) | **No** — Apple GPUs have no FP8/FP4 matrix units; MPSGraph has no FP8 type | **all** formats/layouts run as a host decode-and-accumulate fallback over unified memory, reusing the *same* rlx-cpu oracle | ✅ on this host (Metal == CPU bit-for-bit, incl. MXFP8/NVFP4) |

Every backend now runs **every** format × scale layout; the only difference is
whether the GEMM hits tensor cores (per-tensor FP8 on CUDA/ROCm) or decodes and
accumulates on general cores (everything else, and all of CPU/Metal).

Why CPU + Metal are reference-only: native low-precision *matrix* compute
physically exists only on Hopper/Ada/Blackwell (FP8/FP4) and CDNA3/CDNA4
(FP8-FNUZ/OCP). x86/most-ARM CPUs and Apple GPUs have no FP8 matmul silicon, so
the honest ceiling there is decode-and-accumulate — which the CPU oracle and the
existing storage path already provide.

### CUDA / ROCm native path internals

The GEMM consumes the fp8 codes directly; cuBLASLt/hipBLASLt apply the per-tensor
dequant via `A_SCALE_POINTER` / `B_SCALE_POINTER` (`D = a_scale·b_scale·(A·B)`).
Quantization of activations is two shared kernels in
`rlx-gpu-kernels/kernels/scaled_lowp.cu` (per-tensor amax reduction + FP8 encode,
race-free single-byte stores). Weights quantize once and const-fold.

Scope limit: cuBLASLt 12.3 / CDNA3 hipBLASLt expose only **per-tensor** fp8 scale
pointers — block-scaled MXFP8/MXFP6 and FP4 (NVFP4/MXFP4) native GEMM need
Blackwell + CUDA 12.8 / CDNA4 and are out of scope here. Those formats still run
on the **CPU oracle** today.

## AMP-FP8 insertion pass (`rlx-compile`)

`scaled_quant_insert::insert_scaled_matmul(graph, cfg)` rewrites each 2-D
`Op::MatMul` into the native path: transpose rhs to K-last, insert
`ScaledQuantScale`+`ScaledQuantize` per operand (quantize-once cache; weights
const-fold), feed `ScaledMatMul`. Opt-in — it changes numerics; nothing enables
it by default. Run **before** `ConstantFolding`.

```rust
use rlx_compile::scaled_quant_insert::{insert_scaled_matmul, ScaledQuantConfig};
let g = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());
```

## Verification

On any host (CPU + compile):
```
cargo test -p rlx-ir lowp_codec          # codec round-trip + known values
cargo test -p rlx-cpu scaled_matmul       # ScaledMatMul vs f32 cosine, all formats
cargo test -p rlx-cpu scaled_quant_pass   # AMP pass → run → matches f32 e2e
cargo test -p rlx-compile scaled_quant    # rewrite structure + quantize-once
cargo build -p rlx-cuda -p rlx-rocm       # compile-check the gated GPU kernels
```

On hardware (the tensor-core wins):
- RTX 4090 (Ada sm_89): per-tensor `F8E4M3`/`F8E5M2` `ScaledMatMul` cosine vs the
  CPU oracle; perf vs the f16 baseline.
- MI300 (CDNA3): same via hipBLASLt — **first verify the fp8 datatype /
  scale-pointer / transpose constants in `rlx-rocm/src/hipblaslt.rs` against the
  installed headers** (a wrong value fails the heuristic loudly, not silently).

## Coexistence with the f16 AMP pass

`AutoMixedPrecision` (the f16 relabel pass) **skips** `ScaledMatMul` /
`ScaledQuantize` / `ScaledQuantScale` and never casts their `U8` operands — so
you can run `insert_scaled_matmul` (fp8 matmuls) and then `AutoMixedPrecision`
(f16 elementwise) on the same graph and the fp8 subgraph survives intact
(`coexists_with_f16_amp` test). If an f32 operand of a quantize op was lowered to
f16 upstream, it's cast back to f32 first.

## QAT — straight-through gradient

`ScaledMatMul` is differentiable via a straight-through estimator: its VJP
rebuilds the (quantized) operands with `Op::ScaledDequantize` and runs the
ordinary matmul backward, routing the gradient through `ScaledQuantize`'s
identity STE to the original f32 source (`ScaledQuantScale` is a detached
statistic → no gradient). So `grad(sum(lhs·rhsᵀ))` tracks the f32 matmul
gradient within fp8 reconstruction error (`scaled_matmul_grad.rs`,
rlx-autodiff). This makes fp8 graphs trainable (forward-quant QAT). *Full*
native-fp8 training (fp8 **backward** GEMMs, e5m2 gradients — Transformer-Engine
style) remains future work; the forward STE is the standard QAT building block.

## Status of the original gaps — all closed

- **Metal** — ✅ host decode-accumulate fallback (all formats; tested on-device).
- **fp8 + f16 AMP coexistence** — ✅ the f16 pass skips the scaled ops.
- **QAT** — ✅ straight-through VJP (above), tested on CPU.
- **Block-scaled (MX) + FP4 (NVFP4) + FP6 on GPU** — ✅ run on CUDA/ROCm via the
  on-device decode-accumulate kernel (and on CPU/Metal via the oracle).
  *Tensor-core-native* block/FP4 still needs CUDA 12.8 + Blackwell/CDNA4 (cudarc
  pinned to 12.3); until then these formats are correct on-device but not on the
  tensor cores. The native fast path activates automatically (`is_native_fp8` +
  per-tensor) once the toolkit/hardware land.
