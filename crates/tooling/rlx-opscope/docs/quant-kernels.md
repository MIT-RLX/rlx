# Quantized matmul kernels — speed, precision, and the AMX reality check

This documents the int8 matmul kernels in [`kernels.rs`](../src/kernels.rs)
(`matmul_f32`, `matmul_w8a16`, `matmul_w8a8`, plus the `gemv_*_dot` cores and the
`quantize_*` helpers) and — importantly — the **honest limits** of the speedups
they show. Everything here is measured, not assumed. Build/run:

```sh
# portable scalar (runs anywhere)
cargo run --release -p rlx-opscope --features decompose  --bin opscope-kernels
# NEON f32/W8A16 + SDOT W8A8 (aarch64)
cargo run --release -p rlx-opscope --features dotprod    --bin opscope-kernels
# everything an Apple-Silicon build wants (NEON + SDOT + AMX/Accelerate)
cargo run --release -p rlx-opscope --features apple      --bin opscope-kernels
```

## Feature flags (opt-in; optimizations stay out of the default build)

| feature | what it compiles | implies |
|---|---|---|
| `decompose` | portable **scalar** kernels + all decompositions (runs anywhere) | — |
| `neon`      | aarch64 **NEON** weight-stationary GEMV/GEMM cores (f32 + W8A16 widen) | `decompose` |
| `dotprod`   | + ARMv8.2 **`SDOT`** for the W8A8 fast path (inline asm) | `neon` |
| `amx`       | Apple **AMX** via the Accelerate framework (Apple targets only) | `decompose` |
| `apple`     | convenience: `dotprod` + `amx` | — |

Without any hardware flag, `decompose` gives correct scalar kernels; each flag
lights up one hardware path and is compiled **only when requested** (so a build
that never touches int8 doesn't carry NEON/asm, and a non-Apple/cross build never
links Accelerate). The bin prints which path is active in its header.

## The three precisions

| kernel | weights | activations | how the MACs run |
|---|---|---|---|
| `matmul_f32`   | f32  | f32  | NEON `fmla` (f32×f32→f32) |
| `matmul_w8a16` | int8 | f32  | NEON: load int8, **widen** i8→f32, then f32 `fmla` |
| `matmul_w8a8`  | int8 | int8 | ARM **`SDOT`** (i8×i8→i32, 16 MACs/instr) |

Weights are pre-transposed to output-major `[n,k]` (`transpose` /
`quantize_cols_t`) so each output's weights are contiguous — a *weight-stationary*
GEMV per row, looped over `m` rows for the GEMM. This makes weight traffic the
dominant term and keeps the accumulator in registers.

Quantization is symmetric, per-**output-channel** for weights
(`quantize_cols_t`) and per-**row** for W8A8 activations (`quantize_rows_i8`).
`SDOT` is unstable as an intrinsic (`stdarch_neon_dotprod`), so it's emitted with
stable inline `asm!("sdot …")` under `#[target_feature(enable="dotprod")]`, with a
runtime `is_aarch64_feature_detected!("dotprod")` dispatch and a scalar fallback.

## Measured: speed + precision (Apple Silicon, qwen-0.6B layer shapes)

Speedup is **vs the hand-NEON `matmul_f32`**; error is rel-L2 vs the f32 dense
reference. `m=1` is token-by-token decode; `m=32` is prefill.

| shape (m×k×n)          | f32     | W8A16 (×, err)   | W8A8 (×, err)   |
|------------------------|---------|------------------|-----------------|
| attn decode 1×1024×1024   | 0.068ms | 0.97×, **0.008** | 3.80×, 0.012    |
| attn prefill 32×1024×1024 | 2.264ms | 1.01×, 0.008     | **4.36×**, 0.011|
| mlp-up decode 1×1024×3072 | 0.203ms | 0.99×, 0.008     | 4.11×, 0.011    |
| mlp-up prefill 32×…×3072   | 6.384ms | 1.00×, 0.008     | 3.61×, 0.011    |
| mlp-down decode 1×3072×1024| 0.210ms | 0.99×, 0.008     | 3.78×, 0.012    |
| mlp-down prefill 32×3072×… | 6.117ms | 0.93×, 0.009     | 3.38×, 0.012    |

Reading it:

- **W8A16 = size, not speed.** ~1× (sometimes slightly slower). These layers are
  4–12 MB and fit in the M-series L2/SLC, so the GEMV is compute-bound, not
  DRAM-bound — the i8→f32 widen is pure overhead that eats the byte savings. Its
  win is **4× smaller weights** at the **lowest error (0.008)**.
- **W8A8 = speed, at a precision cost.** ~3.4–4.4× over the NEON f32 baseline
  (both decode *and* prefill) because `SDOT` does 16 int8 MACs per instruction
  (~4× the f32 `fmla` MAC throughput). Error is ~0.011–0.012 — ~50% more than
  W8A16 because activations are quantized too, but still ~1%.

### Why the shape matters

`W8A16` widening cost only shows once f32 is fast; with a single accumulator the
f32 dot is FMA-latency-bound and int8 *looks* competitive. Four independent
accumulators break that chain (both kernels), and only then does f32 hit its
throughput and the widen become visible overhead. `W8A8`'s `SDOT` win is
compute-throughput, so it holds across decode and prefill.

## The AMX reality check (this is the important part)

Apple's **AMX** matrix coprocessor is not portably programmable (undocumented
opcodes); it's reached through **Accelerate** (`cblas_sgemm`) or **BNNS** (int8).
`cblas_sgemm` runs on AMX, so it's the honest yardstick. Measured, k=n=1024:

| m    | NEON-f32 | AMX-sgemm | vs NEON | W8A8-SDOT | vs NEON |
|------|----------|-----------|---------|-----------|---------|
| 1   (decode)  | 0.058ms | 0.006ms | **10.1×** | 0.014ms | 4.0× |
| 32  (prefill) | 1.985ms | 0.053ms | **37.5×** | 0.559ms | 3.6× |
| 128 (prefill) | 8.205ms | 0.165ms | **49.6×** | 2.171ms | 3.8× |

(AMX `sgemm` is f32; rel-err vs dense `2.1e-7`.)

**Accelerate/AMX f32 beats every hand kernel here by 10–50× — and its f32 is
2–10× faster than the hand-written W8A8 int8.** Part of the prefill factor is
multi-core (Accelerate threads; the hand kernels are single-thread), part is the
AMX array. This reframes the whole exercise:

> On Apple Silicon the "3–4× W8A8 speedup" is only true **relative to a naive
> single-thread NEON baseline**. Against a real vendor BLAS, hand-written int8 is
> *slower* than AMX f32.

So the earlier hypothesis — "AMX helps prefill but not decode (GEMV)" — was
**wrong**: AMX wins even at `m=1` (10×). A well-tuned matrix unit + good blocking
beats a naive GEMV loop even without matrix reuse.

## Decision guide

- **Apple Silicon, f32 matmul:** use **Accelerate (`cblas_sgemm`, AMX)**. Don't
  ship the hand-NEON `matmul_f32` as a fast path — it's a portable reference.
- **Apple Silicon, want smaller:** **W8A16** for 4× smaller weights at ~0.008
  error and ~1× speed (footprint win — fit a bigger model / more KV cache). For
  int8 *speed* on Apple you need **BNNS-on-AMX** (unmeasured here), not the
  hand-NEON `SDOT`.
- **CPUs without a vendor matrix unit / BLAS:** the hand kernels stand on their
  own — **W8A8 (`SDOT`)** is the ~3–4× speed path, **W8A16** the size path.
- **Any target:** pick precision by *end-to-end* quality, not per-matmul error
  (below).

## Caveat analysis (measured where possible)

The speedups above come with caveats. Rather than list them, here's each one
*analyzed* — measured when cheap, reasoned when not.

### 1. Per-matmul error ≠ model quality — **measured, closed** ✅

The per-matmul rel-errors are one layer's output; errors **compound across 28
layers**. Measured end-to-end (`qwen_quant_bench`, real qwen3-0.6B, next-token
agreement vs f32 on 16 positions):

| recipe | top-1 | top-5 | cosine | KL |
|---|---|---|---|---|
| **W8A16** (int8 weights) | **100%** | 100% | 0.996 | 0.05 |
| **W8A8** (int8 wt + int8 act) | **81%** | 100% | 0.968 | 0.11 |
| int4 plain | 75% | 75% | 0.734 | 1.31 |
| int4 grouped-32 | 69% | 94% | 0.895 | 0.50 |

**W8A16 is lossless for greedy decoding (100% next-token) — ship it. W8A8
measurably degrades (81%).** Quantizing activations too (naive per-token int8)
adds enough error to flip ~19% of next tokens — better than plain int4 (top-5
stays 100%, so the right token is still ranked high) but not shippable as-is;
it needs better activation quant (per-channel / SmoothQuant / outlier handling),
the same naive→grouped lesson as int4. So the W8A8 *speed* win (§ "how much
faster", and int8-on-AMX for prefill) carries a real *quality* cost — measure,
don't assume. (W8A8 activations are simulated end-to-end via
`rlx_opscope::inject_activation_fakequant`, a graph rewrite that inserts a
per-token int8 round-trip on every matmul's activation input; unit-tested to
match a per-row int8 round-trip.)

### 2. int8-on-AMX vs f32-on-AMX — **measured, a crossover** ✅

`cblas_sgemm` is f32. int8 on AMX needs BNNS — and *not* the general matmul:
`BNNSMatMul` is **float-only** (int8 descriptors → `rc = -1`, verified). The int8
path is the **quantized fully-connected layer** (`BNNSFilterCreateLayerFullyConnected`
→ `BNNSFilterApplyBatch`, int8 in/weights → f32 out, per-tensor scales). Wired and
validated (rc 0, err 0.014). Measured, k=n=1024, **int8-FC-AMX vs f32-sgemm-AMX**:

| m | decode/prefill | int8-AMX vs f32-AMX |
|---|---|---|
| 1   | decode  | **0.48× — loses** (int8 ~2× *slower*) |
| 32  | prefill | 0.84× — ~tie |
| 128 | prefill | **1.98× — wins** (int8 ~2× *faster*) |

So it's a **batch crossover**: at decode (`m=1`) the GEMV underutilizes the int8
array and quant overhead dominates → **f32-AMX wins**; at large prefill the int8
matrix throughput (2× MACs/cycle) pays off → **int8-AMX wins ~2×**. For
decode-bound LLM serving, f32-AMX is best; int8-AMX helps prefill/throughput.
(Absolute AMX timings are noisy under machine load; the within-run int8/f32 ratio
is the reliable signal.)

### 3. Threading — **measured, largely closed** ✅

Concern: Accelerate may use multiple cores while the hand kernels are
single-threaded, so some of the 37–50× could be parallelism, not AMX. Measured
with `VECLIB_MAXIMUM_THREADS=1` (single-thread Accelerate):

| m | AMX multi-thread | AMX single-thread | threading factor |
|---|---|---|---|
| 1   (decode)  | 0.007ms (8.7×)  | 0.006ms (10.2×) | ~1× (threads *hurt* tiny GEMV) |
| 32  (prefill) | 0.044ms (43.9×) | 0.057ms (36.4×) | ~1.3× |
| 128 (prefill) | 0.141ms (53.5×) | 0.191ms (40.3×) | ~1.35× |

**Single-thread AMX is still 10× (decode) to 40× (prefill) over single-thread
NEON.** So the AMX dominance is the *matrix unit*, not multicore — threading adds
at most ~1.35× at large `m` and nothing at decode. Caveat resolved.

### 4. Cache residency — **measured, quantified** ✅

Concern: these layers fit L2/SLC so it's compute-bound; int8's 4× fewer bytes
might matter more when weights spill to DRAM. Measured, m=1:

| shape | f32 weight | location | W8A16 | W8A8 |
|---|---|---|---|---|
| 2048×2048 | 17 MB  | cache | 0.90× | 3.04× |
| 6144×6144 | 151 MB | DRAM  | **1.00×** | 3.22× |

**W8A16 moves from 0.90× to 1.00× as the weight spills to DRAM** — the byte
saving becomes relevant when bandwidth-bound, but only reaches *break-even* at
151 MB (the f32 FMA + i8→f32 widen still gates it; it never becomes fully
DRAM-bound in this naive kernel). So for qwen-0.6B's 4–12 MB (cache-resident)
layers, W8A16 is a **pure footprint win**. W8A8 stays ~3× regardless because its
win is compute (`SDOT`), not bandwidth.

### 5. Portable-demo status — **by design** ℹ️

No cache tiling, no `m`-blocking, no prefetch, single-threaded. These kernels
exist to *quantify* the schemes and to serve targets without a vendor BLAS — not
to be the fastest path on hardware that has one (on Apple, that's AMX; see §
"AMX reality check"). The `f32` numbers here are a *floor*, not the ceiling.
