# Native low-precision GEMM — FP8 / FP6 / FP4 and parameterized `fNeXmY` minifloats

`Op::ScaledMatMul` feeds **low-precision operands directly into the tensor-core /
MFMA GEMM** with f32 accumulation — the real 2×–4× throughput win on hardware
that has FP8/FP4 matrix units. This is distinct from the *storage* quantization
path (`Op::DequantMatMul`, `QuantScheme`), which decodes weights to f32 first and
then runs an ordinary sgemm.

Beyond the named hardware formats, the element type is a **parameterized
minifloat**: any `fNeXmY` split (`ScaledFormat::Custom { exp_bits, mant_bits,
bias }`) — e.g. `f4e3m0`, a signed power-of-two 4-bit grid — runs everywhere the
named formats do.

## The op family (`rlx-ir`)

| Op | Inputs | Output | Role |
|----|--------|--------|------|
| `ScaledQuantScale { format, scale_layout }` | `x` (f32) | scale tensor | per-tensor amax → scale (one value), or per-block scale |
| `ScaledQuantize { format, scale_layout }` | `x` (f32), `scale` | `U8` codes | encode `x / scale` to low-precision codes |
| `ScaledMatMul { lhs_format, rhs_format, scale_layout, has_bias }` | `lhs`,`rhs` (U8 codes), `lhs_scale`,`rhs_scale`, `[bias]` | `[m,n]` f32 | native GEMM, f32 accumulate |
| `ScaledDequantize { format, scale_layout }` | `codes` (U8), `scale` | f32 | inverse of quantize (QAT backward, standalone dequant) |

Operands flow as `DType::U8` byte buffers; the element **format is carried on the
op**, not the dtype — so no `DType` variant is needed and the enum's exhaustive
matches stay intact. Operand `Shape`s carry **logical** element dims.

**Layout is TN**: `lhs [m,k]`, `rhs [n,k]` (both K-last), `out = lhs · rhsᵀ`.
This makes block scales run along the last/contraction axis of *both* operands
uniformly, and matches cuBLASLt / hipBLASLt FP8's required `transa=T, transb=N`.

## Formats — `ScaledFormat`

### Named hardware formats

- FP8 `F8E4M3` (±448), `F8E5M2` (±57344) — OCP. `F8E4M3Fnuz`/`F8E5M2Fnuz` — AMD.
- FP6 `F6E2M3` (±7.5), `F6E3M2` (±28) — OCP MX.
- FP4 `F4E2M1` (±6) — NVFP4 / MXFP4.
- `ScaledFormat::NAMED` is the `[ScaledFormat; 7]` array of these, for sweeps.

### Parameterized `fNeXmY` family (`ScaledFormat::Custom`)

`fN` = total bits = `1` sign + `X` exp + `Y` mant. An all-finite minifloat (no
inf/NaN/FNUZ) whose whole code fits in a byte, so **`exp ≥ 1`, `mant ≥ 0`,
`1 + exp + mant ≤ 8`** → exactly **28** formats. `custom(e, m)` uses the IEEE bias
`2^(e-1) − 1` (the convention every named format follows); `custom_with_bias`
overrides it.

| e\\m | m0 | m1 | m2 | m3 | m4 | m5 | m6 |
|-----|----|----|----|----|----|----|----|
| **e1** (bias 0, range ≈2) | `f2e1m0` | `f3e1m1` | `f4e1m2` | `f5e1m3` | `f6e1m4` | `f7e1m5` | `f8e1m6` |
| **e2** (bias 1) | `f3e2m0` | `f4e2m1`◆ | `f5e2m2` | `f6e2m3`◆ | `f7e2m4` | `f8e2m5` | |
| **e3** (bias 3) | `f4e3m0` | `f5e3m1` | `f6e3m2`◆ | `f7e3m3` | `f8e3m4` | | |
| **e4** (bias 7) | `f5e4m0` | `f6e4m1` | `f7e4m2` | `f8e4m3`◆ | | | |
| **e5** (bias 15) | `f6e5m0` | `f7e5m1` | `f8e5m2`◆ | | | | |
| **e6** (bias 31) | `f7e6m0` | `f8e6m1` | | | | | |
| **e7** (bias 63) | `f8e7m0` | | | | | | |

Max finite grows with `e` (dynamic range): `f4e3m0`→16, `f8e4m3`→480,
`f8e5m2`→114688, `f8e7m0`→1.8e19. Precision grows with `m`. Example — `f4e3m0`
(3 exp, 0 mant) has the grid `±0` and `±{0.25, 0.5, 1, 2, 4, 8, 16}`.

**◆ = the name collides with a named hardware format.** These differ:
`ScaledFormat::custom(4, 3)` is the **all-finite** research format (255 finite
values, max 480), whereas `"f8e4m3".parse()` / `ScaledFormat::F8E4M3` is the
**hardware OCP** format (254 finite + 1 NaN, max 448). The string parser
deliberately returns the hardware-accurate named variant for those 5 names.

The 28 formats and their properties are enumerated + asserted in
`rlx-ir` `enumerate_all_fnexmy` (`cargo test -p rlx-ir enumerate_all_fnexmy -- --nocapture`).

### `ScaledFormat` Rust API / DX

```rust
use rlx_ir::ScaledFormat;

const F: ScaledFormat = ScaledFormat::custom(3, 0);   // const fn; f4e3m0
let f: ScaledFormat = "f4e3m0".parse().unwrap();       // FromStr; Display round-trips
assert_eq!(f.to_string(), "f4e3m0");

f.exp_bits(); f.mant_bits(); f.bias();                 // const accessors
f.is_custom(); f.is_named(); ScaledFormat::NAMED;      // classification / sweeps
f.max_finite(); f.bit_width();

f.encode(1.4);            // -> nearest code (round-half-to-even, saturating)
f.decode(code);           // -> f32 (bit-exact; ±inf/NaN for formats with them)
f.quantize(1.4);          // == 1.0  (round-trip a value to its nearest representable)
f.representable_values(); // the whole grid, ascending
```

### Scale layouts — `ScaleLayout`

- `PerTensor` (one f32 scale) · `BlockMxE8M0 { block: 32 }` (E8M0 power-of-two) ·
  `Nvfp4 { group: 16 }` (FP8-E4M3 block scale).
- Constructors `ScaleLayout::mx()` / `nvfp4()`, and `FromStr`: `"per_tensor"`,
  `"mx"`, `"nvfp4"`, or an explicit `"mx/<block>"`.

The canonical bit-exact f32↔code codec for every format lives in
`rlx-ir/src/lowp_codec.rs` — decode is closed-form from the `(exp, mant, bias)`
fields (so a new `fNeXmY` needs no new code); encode is nearest-representable with
round-half-to-even, saturating overflow **and ±inf** to `±max_finite`, `NaN → 0`.
It is the contract every backend matches.

## Specifying a format at the high level

The same `ScaledFormat` flows through every composition and execution surface —
no hand-wiring the `ScaledQuantScale → ScaledQuantize → ScaledMatMul` chain.

**Compose ops** (low-level `Graph`, and the mirror `HirGraphExt` on the HIR):

```rust
// rhs must be K-last ([n, k]); fmt is any ScaledFormat incl. Custom.
let y = g.scaled_matmul(lhs, rhs, ScaledFormat::custom(3, 0), ScaleLayout::mx());
let (codes, scale) = g.scaled_quantize(x, fmt, layout);   // + scaled_dequantize, scaled_matmul_bias
```

**Tensor DSL** (`rlx-tensor`):

```rust
let y = a.scaled_matmul(&w, ScaledFormat::custom(3, 0), ScaleLayout::mx()); // lazy; .to_vec() to eval
```

**Execute a flow** — rewrite *every* 2-D matmul in a graph at compile time:

```rust
use rlx_runtime::{CompileOptions, ScaledQuantConfig, Session, Device};
let opts = CompileOptions::new().scaled_quant(ScaledQuantConfig {
    lhs_format: ScaledFormat::custom(3, 0),
    rhs_format: ScaledFormat::custom(3, 0),
    scale_layout: ScaleLayout::mx(),
}); // or ScaledQuantConfig::fp8_e4m3() / mxfp8_e4m3()
let compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
```

**Python** (`pyrlx`):

```python
y = g.scaled_matmul(lhs, rhs, format="f4e3m0", layout="mx")   # parses fNeXmY; ValueError on bad names
```

## Per-backend status

| Backend | Native compute? | What runs | Validated |
|---------|-----------------|-----------|-----------|
| **CPU** (`rlx-cpu`) | No (no FP8 matrix ISA) | **Reference oracle**: decode→scale→sgemm for every named + `Custom` format × every scale layout | ✅ on this host (bit-exact grid, cosine vs f32) |
| **CUDA** (`rlx-cuda`) | **Yes** for per-tensor FP8 — cuBLASLt, sm_89+ (Ada/Hopper) | per-tensor `F8E4M3`/`F8E5M2` native GEMM; **all other formats (custom, block-MX, FP4, FP6) run the on-device decode-accumulate kernel** `scaled_matmul_decode` | ✅ **on RTX 3080 Ti** (12-format sweep bit-exact vs oracle; `f4e3m0` grid GEMM bit-exact vs f32; native fp8 *quantize* kernels validated). Tensor-core fp8 GEMM itself still needs sm_89+ |
| **ROCm** (`rlx-rocm`) | **Yes** for per-tensor FP8 — hipBLASLt FNUZ, CDNA3 (MI300) | per-tensor FNUZ fp8 native; other formats via the same decode-accumulate kernel | ⏳ kernels **compile under `hipcc` for gfx90a/CDNA2**; on-device run pending AMD hardware |
| **Metal** (`rlx-metal`) | **No** — Apple GPUs have no FP8/FP4 matrix units | **all** formats/layouts run as a host decode-and-accumulate fallback over unified memory, reusing the *same* rlx-cpu oracle | ✅ **on Apple GPU** (Metal == CPU bit-for-bit, incl. `f4e3m0`, MXFP8, NVFP4) |
| **Vulkan** (`rlx-vulkan`) | **No** — SPIR-V compute path has no FP8/FP4 matrix units | **all** formats run as a CPU host-fallback against the mapped host-visible arena (same rlx-cpu oracle); the four scaled ops are in `SUPPORTED_OPS` + `is_host_fallback`, and the host path writes U8 code/scale outputs as raw bytes | ✅ **on a native NVIDIA Vulkan driver** (RTX 3080 Ti): `f4e3m0` grid GEMM bit-exact vs f32 |

Every backend runs **every** format × scale layout; the only difference is
whether the GEMM hits tensor cores (per-tensor FP8 on CUDA/ROCm with the right
silicon) or decodes and accumulates on general cores / the CPU oracle.

Why CPU + Metal + Vulkan are reference-only: native low-precision *matrix* compute
physically exists only on Hopper/Ada/Blackwell (FP8/FP4) and CDNA3/CDNA4. x86 /
most-ARM CPUs, Apple GPUs, and the generic SPIR-V compute path have no FP8 matmul
silicon, so the honest ceiling there is decode-and-accumulate.

### CUDA / ROCm native path internals

The GEMM consumes the fp8 codes directly; cuBLASLt/hipBLASLt apply the per-tensor
dequant via `A_SCALE_POINTER` / `B_SCALE_POINTER` (`D = a_scale·b_scale·(A·B)`).
Activation quantization is two kernels in
`rlx-gpu-kernels/kernels/scaled_lowp.cu` (per-tensor amax reduction + fp8 encode,
race-free single-byte stores). The f32→fp8 encode is done **in closed form**
(matching the oracle bit-for-bit) so the kernel NVRTC/hipRTC-compiles without the
toolkit's `<cuda_fp8.h>` / `<hip/hip_fp8.h>` (which have no include search path
under the runtime compilers). Weights quantize once and const-fold.

The **decode-accumulate kernel** (`scaled_lowp_general.cu`, the non-tensor-core
fallback for every custom / block / FP4 / FP6 format) is **16×16 shared-memory
tiled** — each code is decoded once per tile instead of once per output element:
**~5.4× faster** on the RTX 3080 Ti at 1024³ (74.9 → 405 GFLOP/s). Its generic
path unpacks `(exp, mant, bias)` from a packed `kernel_id()` descriptor (top-bit
sentinel), so a new `fNeXmY` needs **no kernel edit**; the seven named ids
(`0..=6`) keep the existing `switch`, leaving the hardware path byte-identical.

Scope limit: cuBLASLt 12.3 / CDNA3 hipBLASLt expose only **per-tensor** fp8 scale
pointers — block-scaled MXFP8/MXFP6 and FP4 (NVFP4/MXFP4) native GEMM need
Blackwell + CUDA 12.8 / CDNA4 and are out of scope here. Those formats still run
via the decode-accumulate kernel today.

## AMP / scaled-quant insertion pass (`rlx-compile`) + execution wiring

`scaled_quant_insert::insert_scaled_matmul(graph, cfg)` rewrites each 2-D
`Op::MatMul` into the native path: transpose rhs to K-last, insert
`ScaledQuantScale`+`ScaledQuantize` per operand (quantize-once cache; weights
const-fold), feed `ScaledMatMul`. Opt-in — it changes numerics; nothing enables
it by default. Run **before** `ConstantFolding`. It accepts **any** `ScaledFormat`
(incl. `Custom`) via `ScaledQuantConfig`.

```rust
use rlx_compile::scaled_quant_insert::{insert_scaled_matmul, ScaledQuantConfig};
let g = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());
```

At the high level this pass runs automatically when
`CompileOptions::scaled_quant(cfg)` is set — `Session::compile_with` applies it
before the rest of the pipeline (see *Specifying a format at the high level*).

## Verification

On any host (CPU + compile):
```
cargo test -p rlx-ir lowp_codec              # codec round-trip + known values (incl. custom)
cargo test -p rlx-ir enumerate_all_fnexmy    # the 28 fNeXmY formats + properties (--nocapture)
cargo test -p rlx-cpu scaled                 # ScaledMatMul/quantize vs f32, all formats + f4e3m0 grid
cargo test -p rlx-cpu scaled_matmul_builder  # Graph::scaled_matmul builder end-to-end
cargo test -p rlx-runtime --test scaled_quant_policy   # CompileOptions::scaled_quant e2e
cargo test -p rlx-tensor --features eval --test tensor scaled_matmul  # Tensor DSL e2e
cargo test -p rlx-compile scaled_quant       # rewrite structure + quantize-once
cargo build -p rlx-cuda -p rlx-rocm -p rlx-vulkan      # compile-check the GPU kernels
```

On hardware (validated):
- **CUDA — RTX 3080 Ti** (`rlx-cuda/tests/cuda_scaled_custom.rs`): `f4e3m0` grid
  GEMM bit-exact vs f32; 12-format quantize→dequantize sweep bit-for-bit == the
  CPU oracle; tiled decode GEMM 405 GFLOP/s (5.4× the naive kernel). The
  tensor-core fp8 *GEMM* still needs sm_89+ (Ada/Hopper); the 3080 Ti (Ampere)
  validates the quantize kernels + decode path only.
- **Metal — Apple GPU** (`rlx-metal/tests/metal_scaled_matmul_parity.rs`): Metal
  == CPU bit-for-bit for `f4e3m0` and the named formats.
- **Vulkan — native NVIDIA** (`rlx-vulkan/tests/vulkan_scaled_custom.rs`):
  `f4e3m0` grid GEMM bit-exact vs f32 on the RTX 3080 Ti driver.
- **ROCm — MI300 (CDNA3)**: pending AMD hardware. The shared kernels compile
  under `hipcc` for gfx90a; **verify the fp8 datatype / scale-pointer / transpose
  constants in `rlx-rocm/src/hipblaslt.rs` against the installed headers** before
  trusting the native GEMM.

## QAT — straight-through gradient

`ScaledMatMul` is differentiable via a straight-through estimator: its VJP
rebuilds the (quantized) operands with `Op::ScaledDequantize` and runs the
ordinary matmul backward, routing the gradient through `ScaledQuantize`'s
identity STE to the original f32 source (`ScaledQuantScale` is a detached
statistic → no gradient). So `grad(sum(lhs·rhsᵀ))` tracks the f32 matmul
gradient within reconstruction error (`scaled_matmul_grad.rs`, rlx-autodiff).
This makes low-precision graphs trainable (forward-quant QAT). *Full* native-fp8
training (fp8 **backward** GEMMs, e5m2 gradients — Transformer-Engine style)
remains future work; the forward STE is the standard QAT building block.

## Coexistence with the f16 AMP pass

`AutoMixedPrecision` (the f16 relabel pass) **skips** `ScaledMatMul` /
`ScaledQuantize` / `ScaledQuantScale` and never casts their `U8` operands — so
you can run `insert_scaled_matmul` (fp8/custom matmuls) then `AutoMixedPrecision`
(f16 elementwise) on the same graph and the low-precision subgraph survives
intact (`coexists_with_f16_amp` test). If an f32 operand of a quantize op was
lowered to f16 upstream, it's cast back to f32 first.

## History / discovered issues (fixed)

Running the scaled path on real hardware for the first time surfaced two latent
NVRTC bugs — the scaled kernels had never actually NVRTC-compiled before:
1. the decode kernel used the `INFINITY` macro (undefined under NVRTC) → now
   `__int_as_float`, `#ifndef`-guarded so nvcc/hipcc are unaffected;
2. the native fp8 quantize kernel `#include <cuda_fp8.h>` / `<hip/hip_fp8.h>`
   (no include path under NVRTC/hipRTC) → the f32→fp8 conversion is now closed
   form, matching the oracle bit-for-bit and removing the header dependency.

Also fixed: `encode(±inf)` now saturates to `±max_finite` (was code `0`) on CPU
and GPU in lockstep; the Vulkan generic host path now writes U8 outputs as raw
bytes instead of reinterpreting them as f32.
