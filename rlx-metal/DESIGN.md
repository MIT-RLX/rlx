# rlx-metal DESIGN

## What this crate owns

Apple GPU execution. Three coexisting strategies:

1. **Per-thunk MSL kernels** (`thunk.rs` + `kernels.rs`) — every
   op has a hand-written MSL kernel; the executor encodes one
   compute dispatch per thunk. Default.
2. **MPSGraph lowering** (`mps_graph.rs` + `mps_graph_lower.rs`) —
   subgraphs are translated to MPSGraph ops and Metal's optimizer
   schedules them. Opt-in per op (e.g. `RLX_MPSGRAPH_ATTENTION=1`).
   Phases F/J extended this to Concat / FusedSwiGLU / RoPE
   cos-sin slicing.
3. **ICB batching** (`icb.rs`) — Indirect Command Buffer wrapping
   the per-thunk encodes. The cross-cutting throughput unlock:
   `wait_until_completed` overhead amortizes across more ops.

## Decisions

**Pretty much all hot paths still go via thunks.** MPSGraph buys
schedule optimization but we measured per-op encode cost is small
relative to the ~150 µs `wait_until_completed` floor — chaining
many MSL kernels in one ICB beats pushing fewer MPSGraph ops at
the same latency.

**f16 dispatch via `HalfFlag`.** Phase F threaded dtype through
encoders + kernels; every variant that exists in f32 has an `_h`
sibling. Phase G eliminated cast tax inside AutoMixedPrecision
(no f32↔f16 round-trips between consecutive f16 ops).

**Strided source on Rope.** Mirrors the CPU change for plan #45 —
`src_row_stride` parameter lets the Narrow→Rope fusion rewrite
Rope to read directly from the parent QKV. Both `rope` and
`rope_h` MSL kernels accept it as buffer 8.

**Calibration is cached on disk.** `~/.cache/rlx/metal-calib-<hwid>.json`
keyed by GPU registry id; runs ~50 ms of probes once per machine.
Hardware change → cache miss → re-measure automatically.

## What doesn't work / why

- **No strided Q/K/V on Metal Attention.** The CPU did it for #46
  deep; Metal kernel still walks Q/K/V with hardcoded `hs` row
  stride. Mirroring requires editing the Attention MSL kernel and
  touching encode + ICB paths — bigger surface than the Rope
  mirror, deferred.
- **No fused dequant + matmul.** Same as CPU side; QuantMap exists
  but kernels.rs doesn't dispatch on it.
- **`encoder-switch overhead`.** Some ops (Reshape / view-only
  Cast) used to switch from compute encoder to blit encoder. The
  switch is expensive (~60 µs) so we route those through `copy_f32`
  / `copy_h` MSL kernels even though a blit copy would be O(memcpy).
  Acceptable as long as no one emits these ops in tight loops.

## Cross-crate contract

- Same `Op` → `Thunk` pattern as CPU; same `is_pure_view` check.
- `Thunk::Rope.src_row_stride` and `Thunk::Attention.q/k/v_row_stride`
  (CPU-only for now) are the public knobs the fusion passes flip
  to elide narrow copies.
- Calibration values feed `cost.rs`; `rlx_runtime::pick_best_device`
  consumes both backends' cost models to choose CPU vs Metal per
  graph.
