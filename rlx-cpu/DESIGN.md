# rlx-cpu DESIGN

## What this crate owns

CPU execution. SIMD kernels, BLAS dispatch, thread pool, arena
executor. Two coexisting execution paths in `thunk.rs`:

1. **Closure-based** — every Thunk variant builds a
   `Box<dyn Fn(*mut u8) + Send + Sync>` at compile time; the
   executor calls the boxed closure per thunk. Uniform but
   pays one indirect call per op.
2. **Direct execution** — single big `match` over `&Thunk` in
   `execute_thunks`. Zero closure overhead. The hot kernels
   (Attention, FusedAttnBlock, FusedNomicLayer, matmul) live here.

Both paths must stay in sync when changing a Thunk variant.

## Decisions

**Two execution paths, not one.** The closure path is the legacy
default and we kept it for ops where dispatch cost doesn't matter
(small element-wise stuff). The direct path is where perf-critical
ops have moved. Picking one would be a refactor; doing it right
needs benchmarks per op to verify the closure call cost.

**FusedAttnBlock and FusedNomicLayer are thunk-level, not IR-level.**
The fusion patterns scan the post-lowering thunk schedule because
the matchers need positional access to neighboring thunks (e.g.
"three Narrows then maybe two Ropes then Attention then matmul").
Doing this on the graph would require richer graph-walking
infrastructure than rlx-opt has today.

**Strided Q/K/V on Thunk::Attention (default = hidden, fused = parent
row stride).** Lets the Narrow×3→Attention fusion (#46 deep) elide
per-head buffers without rewriting the kernel — same code path,
different stride values.

**Persistent thread pool, no per-call init.** `pool::par_for(total,
grain, &|off, cnt| ...)` is the parallel primitive. Fixed worker
count from `RuntimeConfig::pool_workers`. Workers spin briefly
before parking — better tail latency than a wakeup-per-call pool
on Apple Silicon's P/E asymmetric cores.

## What doesn't work / why

- **`autotune.rs` only tunes `par_threshold` and `sdpa_seq_threshold`.**
  Pool worker count can't be retuned after init (the pool is
  constructed once). To tune workers we'd need a dispose-and-rebuild
  pattern.
- **No fused dequant + matmul.** `QuantMap` exists in `rlx-ir` (#57)
  but blas.rs doesn't read it. The big LLM-bandwidth win on Apple
  Silicon is still on the table (plan #5).
- **Thunk-level fusion patterns are fragile.** Adding new ops
  upstream can subtly break the matchers without compile errors.
  The unit tests (e.g. `narrow_attention_fuses_in_unfused_path`)
  catch the common case but a regression test per fusion-pattern
  shape would be worth more than they currently are.

## Cross-crate contract

- `compile_thunks(graph, arena)` → `ThunkSchedule`. The arena's
  `MemoryPlan` (from rlx-opt) tells us byte offsets; we never
  allocate. The schedule is the executable artifact.
- Hot ops dispatch via `match thunk { ... }`; new variants need
  arms in BOTH execution paths and the `thunk_read_offsets` helper
  if they read any arena buffer (used by view / fusion safety
  checks).
