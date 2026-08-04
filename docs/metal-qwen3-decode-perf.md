# Metal qwen3 decode performance — findings

How batch-1 (KV-cache) decode of qwen3-0.6B on the Metal backend went from
**24 → 82 tok/s (+242%)** on an M4 Pro, token-identical throughout, and — just
as usefully — which "obvious" optimizations turned out to be **neutral or
negative** and why.

Bench: `rlx-models` `qwen_quant_bench kvbench metal` (median per-step decode
tps). All flags are opt-in / default-off unless noted.

## TL;DR — what actually moved the needle

| stage | tok/s | how |
|---|---|---|
| f32 baseline | 24 | — |
| **F16-resident weights** | 33 | `RLX_QWEN3_F16_WEIGHTS` — store matmul weights f16 (native bf16 ⇒ ≈lossless) |
| **K-split M=1 GEMV** | 55 | `gemv_f16w_splitk` (KSPLIT=32 + half2) — saturate more bandwidth on small-N gemv |
| **Bake the weight concat** | 82 | `RLX_QWEN3_BAKE_WEIGHTS` — stop re-concatenating the fused QKV/gate-up weight every token |
| **GQA-native attention** | **89** | `RLX_QWEN3_GQA_NATIVE` — drop `repeat_kv`'s Expand; attention reads un-expanded KV in place |

Each is a **bytes-moved** reduction. Nothing that only reduced *dispatch count*
helped (see below).

## The one true bottleneck: DRAM bytes, not dispatches

Decode is **weight-read-bandwidth bound**. The step time is set by how many
weight-bytes stream from DRAM per token, full stop. Evidence:

- `RLX_METAL_TRACE` splits the step: `encode ≈ 0.25 ms`, `wait ≈ GPU time`. It's
  ~99% GPU-side — CPU encode is noise, so nothing that only saves encode helps.
- **MPS matmul** (`RLX_METAL_SGEMM_MPS=1`, Apple's tuned kernel) gave 24.6 vs
  24.4 — *not* compute-bound.
- **Removing dispatches changed nothing** (three independent tries, all
  token-identical, all neutral-to-negative — see "What didn't work").

The mental model: on the single serial command buffer the GPU's command
processor overlaps the *next* dispatch's setup with the memory stalls of the
current one, so a launch costs ~0 when there's always a DRAM transaction in
flight. Fusion / dispatch-reduction only helps a **launch-bound** workload;
this one is **bandwidth-bound**. The only lever is bytes.

## Lever 1 — F16-resident weights (24 → 33)

Weights are natively bf16 but rlx-metal was dequantizing to f32 in the arena, so
decode read ~2.2 GB/token. The **Param node's `shape.dtype()` is the single
source of truth**: declare the 7 decode matmul weights (+ tied lm_head) `F16` at
graph-build time and everything follows automatically — arena sizing, the
bind-time f32→f16 conversion in `set_param` (`arena.write_from_f32`), and
`Sgemm { b_f16 } → metal_sgemm_f16w`. Contained entirely in `rlx-flow`
(`context.rs::load_param_typed`, `blocks/qwen3_decode_layer.rs`,
`blocks/lm_head.rs`). No Metal-backend or upload-loop change. AMP does **not**
do this — it keeps weight Params f32.

## Lever 2 — K-split M=1 GEMV kernel (33 → 55)

`sgemm_f16w_small_m` launched one thread per output column → only ~1–3k threads
on the decode projections (N≈1–3k), far too few to saturate memory (~37 GB/s of
~273 peak). `gemv_f16w_splitk` splits the K-sum **KSPLIT=32** ways (threadgroup =
32 cols × 32 splits = 1024 threads, Metal's max): a simdgroup is the 32 columns
for one split so `B` stays coalesced, and partials reduce in threadgroup memory.
KSPLIT sweep: 8→45.7, 16→50.4, 32→54.8. half2 loads (2 cols/thread, 128-byte
transactions, even-N) → 55.4. Effective ~110 GB/s (40% peak) — near this
kernel's ceiling. Default-on within the f16 path; opt out `RLX_METAL_GEMV_SPLITK=0`.

## Lever 3 — bake the fused-weight concat (55 → 82) — the surprise

`RLX_METAL_DUMP_BYTES` (analytic per-op DRAM traffic) exposed what dispatch
counting and per-op profiling both missed: the top line wasn't the matmul, it
was **`concat`, at 49% (1.18 GB/step)**. Those aren't KV concats (the KV concat
is a tiny 15 MB) — they're the **weight concatenations** that
`FuseSharedInputMatMul` (QKV) and `FuseSwiGLUDualMatmul` (gate/up) insert to
build their combined weight. Those weights are **constant**, yet the graph
re-concatenates them **every token**, reading the individual weights + writing
the fused layout — ≈doubling the weight bandwidth (total 2434 ≈ 2 × 1193 MB).

It isn't constant-folded because the concat's inputs are `Param` nodes
(runtime-settable via `set_param`), so the compiler conservatively re-runs it.

Two fixes were built and A/B-tested (both token-identical, same window):

| | tok/s | wait | bytes/step |
|---|---|---|---|
| baseline (concat every step) | 55 | 17.4 ms | 2434 MB |
| **B** — read weights in place (`RLX_NO_WEIGHT_CONCAT_FUSION`; disable the two weight-concat fusions, keep activation `FuseSwiGLU`) | 80 | 11.9 ms | 1260 MB |
| **A** — bake once (`RLX_QWEN3_BAKE_WEIGHTS`; `Concat.weight_const` set at lowering when all inputs are Param/Constant → compute on first step, skip after) | **82** | **11.2 ms** | 1260 MB |

**A wins**: it keeps the single fused matmul (one big N=4096 GEMV → better
occupancy than B's separate smaller ones) *and* keeps the fusion's benefit on
prefill, while paying the concat once instead of per-token. The baked fused
weight lives in its arena slot, which (verified) isn't reused across steps.

**Recommended permanent form:** teach `rlx-compile/const_fold` to fold
`Concat`-of-constants (it currently hits `_ => None` for Concat), and run
`SpecializeParams` + `ConstantFolding` on the inference graph. That makes the
bake automatic, on by default, on every backend and model — not a qwen3/Metal
env flag. This is a graph-level bug, not a Metal one.

## Lever 4 — GQA-native attention (82 → 89)

`repeat_kv` expands the K/V from `nkv=8` to `nh=16` heads so attention has a
matching KV head per query head. But the SDPA decode kernel already does GQA
internally (`qkv_kv_offset` maps query head → `hi/group`), so the Expand is
avoidable: pass the **un-expanded** nkv-head K/V and skip it (`RLX_QWEN3_GQA_NATIVE`).
The Expand writes 2× the KV (30 MB) which attention then reads (~75 MB of KV
traffic); reading the base KV in place is ~30 MB, and the `group×` re-reads of
each shared head hit L2 (8 heads ≈ 160 KB, cache-resident). Measured
**11.4 → 10.4 ms, 82 → 89 tok/s**, token-identical. Same idea as baking the
weight concat, applied to KV: don't materialize a bigger copy the kernel can
read from the original. (Shared-graph change, so gated to Metal via the flag
until validated on the other backends' attention.)

> **Cautionary tale:** this was earlier *rejected* as "slower (17.1 → 18.4 ms)"
> from a best-of-5 run under heavy machine contention, with a plausible-sounding
> coalescing rationalization. It was noise. The byte math (75 vs 30 MB) said it
> should be faster all along; a clean re-measurement confirmed it. Trust the
> bytes over a noisy timer.

## What did NOT work (and why) — all token-identical, all neutral/negative

The recurring lesson: **dispatch count and per-op-sync profiles are misleading;
only bytes moved matter on a bandwidth-bound decode.**

- **Drop 56 residual-add dispatches (f32):** identical wait. The adds move ~12 KB
  each; removing them is below noise.
- **`FusedMatMulResidual`** (fold residual into the matmul store): 0 benefit, and
  its f32-only epilogue *conflicts* with f16 weights (forces o/down to f32).
  Implemented but **not claimed by Metal**.
- **Dual-output residual+RMSNorm fusion** (`RLX_METAL_FUSE_RESIDUAL_DUAL`): fires
  on all 55 per-layer residuals, correct — but neutral (identical wait). Fewer
  launches, same bytes.
- **F16 KV cache** (`RLX_QWEN3_F16_KV`): neutral at short context — the KV is
  only ~6% of traffic (~77 MB) vs the 94% weight stream. It's a **long-context**
  lever (KV grows O(context); at 1k+ tokens it dominates). Also currently WIP —
  the attention kernel + f16 readback/transpose were built, but the round-trip
  still corrupts due to f32 assumptions in `rlx-runtime`'s bucketed-decode KV
  padding/readback; not worth completing for short-context.
- **Full `RLX_METAL_NO_FUSION`:** 79 tok/s — a blunt proof of the weight-concat
  cost (it also drops the good activation fusions; A/B are the targeted fixes).

## Methodology: record bytes, not dispatches

`RLX_METAL_DUMP_BYTES` (in `rlx-metal/src/thunk/mod.rs::dump_thunk_bytes`) sums
analytic read+write bytes per op type for one decode step. Gate: fires on a
decode step (>10 `m==1` sgemms), not prefill (prefill's only m==1 sgemm is the
last-token lm_head). This is the tool that found lever 3 — the per-thunk-sync
profiler (`RLX_METAL_THUNK_PROFILE`) inflates by dispatch count and pointed at
the wrong ops (it ranked `concat` high for the right reason but the wrong
metric, and buried the weight-vs-KV distinction).

## Flag reference

| flag | default | effect |
|---|---|---|
| `RLX_QWEN3_F16_WEIGHTS` | off | f16-resident decode matmul weights + lm_head (lever 1) |
| `RLX_METAL_GEMV_SPLITK` | **on** (within f16 path) | K-split M=1 gemv (lever 2); `=0` reverts to `sgemm_f16w_small_m` |
| `RLX_QWEN3_BAKE_WEIGHTS` | off | compute weight-only concats once, skip after (lever 3, option A) |
| `RLX_NO_WEIGHT_CONCAT_FUSION` | off | disable the two weight-concat fusions; read weights in place (lever 3, option B) |
| `RLX_QWEN3_GQA_NATIVE` | off | drop `repeat_kv`; attention reads un-expanded KV (lever 4) |
| `RLX_METAL_DUMP_BYTES` | off | dump per-op DRAM traffic for one decode step |
| `RLX_METAL_TRACE` | off | per-step encode / commit / wait µs split |
| `RLX_METAL_FUSE_RESIDUAL_DUAL` | off | dual-output residual+RMSNorm fusion (neutral) |
| `RLX_QWEN3_F16_KV` | off | f16 KV cache (WIP / long-context only) |

## Bottom line

**24 → 89 tok/s, +271%, ~2.8× past MLX (32), token-identical.** The wins were all
byte reductions (f16 weights; a gemv that reads them faster; stop copying the
weights twice; stop expanding the KV). Every dispatch/fusion idea was neutral
because the workload has a byte problem, not a launch problem. The single biggest
miss — a 1.18 GB/token redundant weight copy — was invisible to dispatch-counting
and per-op profiling and obvious the instant we recorded **bytes moved**. And the
one lever we wrongly rejected (GQA-native) failed only a noisy timer, not the byte
math — which is the whole lesson twice over: **measure bytes moved.**
