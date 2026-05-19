# rlx-runtime DESIGN

## What this crate owns

The user-facing API. Glues IR + opt + backends. Public surface:
`Session`, `CompiledGraph`, `Device`, `Backend` trait, `Tick`,
`CacheBuster`, `ktrace!`, `nan_check`.

## Decisions

**Backend selection is feature-gated, not runtime.** Adding `metal`
to the feature set both compiles the Metal backend in AND makes
`Device::Metal` constructible. Without the feature, instantiating
`Device::Metal` panics at the registry lookup. This keeps binaries
small for users who only need CPU.

**Compile cache uses graph fingerprint + precision policy.** Bumping
precision invalidates entries — the AOT-compiled artifact is
specialized to a precision policy. Right call: same graph at
different precisions wants different fused kernels.

**`CompiledGraph::run` is sync.** Async / streaming was tempting but
the sync API is simpler and the per-call cost (~150 µs Metal
roundtrip floor) is the dominant latency anyway. A future
serving-oriented `run_async` can wrap sync via the existing thread
pool.

**Trace tensors live in `trace.rs`; the `kernel-trace` macro is
separate.** The two "tracing" things have different jobs — one is
graph construction (record ops on TracedTensor → emit Graph), the
other is debug logging at runtime. Putting them in different
modules avoids the naming collision MAX hit by overloading "trace."

## What doesn't work / why

- **`run_if` / `run_while` exist but the executor doesn't lower
  `Op::If` / `While`.** Both fall through to `Thunk::Nop`. Wiring
  it up needs recursive subgraph compilation at parent-compile
  time. No model in tree exercises these ops, so the work doesn't
  have a validation target. Estimated 4-6 hours when there's a
  consumer.
- **`weights.rs` is just a trait + a bytes loader.** The named
  weights registry pattern (#24 in PLAN.md) — addressable handles,
  ref counts, hot-swap LoRA — is the natural extension; not yet
  needed.
- **No worker isolation.** Long-running serving paths probably
  want a subprocess worker per request (#36 in PLAN.md). Today
  everything runs in-process; one bad model load takes the
  process down.

## Cross-crate contract

- `Session::new(Device).compile(graph)` runs:
  1. `rlx_opt` passes (precision, fusion, memory plan, view alias).
  2. Backend's `compile_thunks` to lower to backend-specific
     `ThunkSchedule`.
  3. Wrap in `CompiledGraph` with the arena.
- Re-exported helpers (`Tick`, `CacheBuster`) come from `rlx_ir` so
  callers don't need a direct `rlx_ir` dep.
