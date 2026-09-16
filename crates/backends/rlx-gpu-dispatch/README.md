# rlx-gpu-dispatch

The shape-keyed GPU dispatch table, shared by every RLX GPU backend.

`rlx-cpu`'s `dispatch.rs` states the principle — *kernel-variant selection is a
data lookup, not scattered match arms in dispatch sites* — with compile-time
per-arch defaults and runtime overrides filled by measurement. This is the GPU
twin, plus the one thing the CPU table doesn't need: a **shape bucket** in the
key, because a GEMM at `m=1` (decode) and one at `m=4096` (prefill) want
different physical schedules and a single threshold can't say so.

Three modules:

- [`dispatch`] — `(arch, op, shape-bucket) -> Choice`, with exact compile-time
  defaults, measured runtime overrides, and a tab-separated persistence format.
- [`tiles`] — the tile parameter space for the shared `matmul` kernel and the
  legality rules a tile must satisfy before it is compiled.
- [`cost`] — an analytical model that *ranks* candidate tiles without running
  them, so the tuner can spend device time only on plausible winners.

## What the cost model may and may not be used for

`cost` exists because measuring the full candidate cross product needs an idle
GPU and a JIT compile per entry. It narrows the set; it never picks the winner.

The distinction is not stylistic. Held out one shape at a time against real
sm_86 timings, the model reproduces the *measured ordering* of six tiles at
Spearman ρ ≥ 0.94. Ordering is all it can do: the whole model is a single
constant — one shared-memory staging barrier costs about as many cycles as
25,464 padded MACs — and scores are denominated in padded-MAC equivalents so
they cannot be mistaken for milliseconds. `tests/cost_model_ranking.rs` carries
the timings, so both the constant and the ρ claim are re-derived on every test
run rather than taken on faith.

Four consequences are wired in rather than documented and hoped for:

- **No opinion is not a rejection.** `estimate` returns `None` for workloads
  outside its coverage, and `prefilter` sorts those *first* so an unscored
  candidate always survives.
- **Nothing is dropped silently.** `Prefilter::dropped` names what was removed
  and `disclosure()` says on whose authority, so a narrowed sweep cannot read
  like an exhaustive one.
- **Uncalibrated models say so.** `CostProvenance::Structural` is usable for
  ordering on an arch nobody has measured, and its disclosure line marks it
  `UNCALIBRATED`. Same discipline as `rlx_runtime`'s `CostCalibration`: an
  invented number may inform a search, never a claim.
- **Extrapolation is marked.** Held-out validation licenses interpolation only.
  The sm_86 fit spans `m ∈ [1, 32]`, yet the model changes its recommendation
  at `m ≥ 128` — in a regime no calibration point touches. `CostProvenance`
  carries the measured shape box and `TileCost::extrapolated` flags answers
  from outside it, so the guess does not read like the validated part.

`rlx-cuda`'s `tune_dispatch` runs the model on every bucket whether or not the
prefilter is enabled (`RLX_TUNE_PREFILTER=N`, default off), prints its rank
beside each measurement, and scores it at the end. With the prefilter off, that
score is independent evidence from shapes the model was never fitted on — which
is what would justify turning it on later.

## Why this crate exists separately

It holds *decisions*, not *sources*. `rlx-gpu-kernels` owns the CUDA/HIP `.cu`
text and is rightly a CUDA/ROCm-only dependency; Metal and wgpu need the table
without any of that. Keeping the two apart is what lets all five GPU backends
share one routing policy.

The crate is dependency-free and touches neither the filesystem nor the
environment — each backend owns where its tuning cache lives (`RLX_GPU_TUNING_CACHE`).

## Layering

```
rlx-gpu-dispatch  ──▶  rlx-gpu-kernels  ──▶  rlx-cuda / rlx-rocm
       │
       ├────────────────────────────────▶  rlx-metal
       └────────────────────────────────▶  rlx-wgpu
```
## License

MIT OR Apache-2.0.
