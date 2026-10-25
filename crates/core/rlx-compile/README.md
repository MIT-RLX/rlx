# rlx-compile

HIR → MIR → LIR compile pipeline for RLX: [`CompilePipeline`], backend
legalization, memory planning, precision / PTQ passes.

Depends on [`rlx-ir`] and [`rlx-fusion`].

## What's here

- **`compiler`** — `CompilePipeline`, `CompileResult`, pipeline inspect hooks.
- **`fusion_pipeline`** — `fusion_passes`, `fusion_passes_for_supported`,
  backend-aware pass lists.
- **`legalize`** / **`legalize_broadcast`** — rewrite graphs for a target device.
- **`memory`** — liveness analysis and arena slot assignment.
- **`precision`** — auto-mixed precision policy (f32 ↔ f16/bf16 around matmul).
- **`quant_insert`** / **`quant_propagate`** — PTQ Q/DQ insertion and propagation.
- **`const_fold`**, **`dce`**, **`inline`**, **`promote_params`**, **`svg`**.
- **`param_specialize`** — bake fixed `Op::Param` tensors into `Op::Constant`
  before DCE / constant folding (`CompileOptions::param_bindings` in runtime).
- **`algebraic_simplify`** — same-shape `x*0`, `x*1`, `x+0` rewrites (safe
  only when constant rank matches output; no scalar-broadcast folding).
- **`cse`**, **`sccp`** — common-subexpression elimination; sparse
  conditional constant propagation.
- **`simplify_transposes`** — remove transpose pairs that cancel (importer
  layout conversions, e.g. ONNX from TensorFlow).
- **`param_hoist`** — split a param-invariant *prepare* graph off the per-run
  graph.
- **`plan_check`** — program-safety and schedule-semantics gates over a
  `MemoryPlan` (`schedule_hazards`).
- **`numeric_lint`** — provable compile-time NaN/Inf sources.
- **`rewrite`** / **`dispatch_report`** — decompose ops a backend doesn't
  claim; report native vs common-IR vs missing.
- **`scaled_quant_insert`** (FP8/FP4 `ScaledMatMul`), **`bitnet_qat`**
  (BitNet b1.58 QAT rewrite), **`training_compile`**, **`inspect`** (dump every
  HIR / MIR / LIR stage).

## Number formats

`CompileOptions::number_format` (in `rlx-runtime`) runs an f32 graph in
another format, by graph rewrite:

| format | pass | what happens |
|---|---|---|
| `F64` | `promote_f64` | every f32 value becomes f64; inputs and outputs stay f32, params are widened once on upload. Native f64 where the backend has it. |
| `F64` on a GPU without f64, or `F64x2` | `decompose_f64` → `lower_f64` | composites are decomposed to the primitives with double-single kernels, then each f64 tensor `[…]` becomes an f32 tensor `[…, 2]` (`Op::DoubleSingle`, ~48 significant bits) |
| `Float { … }` / `Mixed { … }` | `emulate_format` | `Op::RoundToFormat` after the weights and each op, so the graph computes as if values were stored in a narrower float format |

## Kernel dispatch transparency

`prepare_graph_for_backend_with_report` and friends produce a
`KernelDispatchReport` (native vs common-IR vs rewritten vs unsupported).
The runtime uses the same legalization path as compile; see the root
[`README.md`](../../../README.md#kernel-dispatch-and-transparency) for env vars
(`RLX_DISPATCH_REPORT`, `RLX_KERNEL_DISPATCH`).

## Op sharding — decompose, price the IO, choose where

[`shard`](src/shard.rs) cuts a single op into pieces that have no data
dependency on each other, so they can run on separate threads, devices or
nodes:

```rust
use rlx_compile::shard::{choose, shardings, LinkCost, Residency, WorkerCost};

let options = shardings(&graph, node, 4);          // ways to cut it
let pick = choose(&options, &workers, link, Residency { shared_resident: true }, local_rate);
```

Three questions, and nothing else:

1. **Which axis, into what pieces.** A conv can be cut by batch or by output
   channel; a matmul by rows or columns. Ranges are contiguous, disjoint and
   cover the axis exactly — a gap or an overlap would silently drop or
   double-count samples.
2. **What it costs in bytes.** The axes differ in *which operand gets
   replicated*, and that is usually the whole story: a batch split replicates
   the weights, an out-channel split replicates the activations. Sliced
   operands cross a link once in total; shared ones cross once per worker,
   unless `Residency { shared_resident: true }` says they are already there —
   which is the data-parallel case, and it changes the answer.
3. **Where to run it.** `choose` scores the slowest shard's compute against the
   transfer, and returns `None` for "run it whole" when no split beats that. A
   worker whose cost model is not rankable is **excluded and reported**, by the
   same `CostCalibration::is_rankable` rule device ranking already follows.

The module performs no graph surgery: `ShardOption` names the operand, axis and
ranges, and the caller emits narrows and a concat, or ships ranges to workers.
It also defines its own `LinkCost`/`WorkerCost` rather than importing
`rlx_collectives::planner::Link` or `rlx_runtime::BackendCostModel`, which both
sit above this crate — `rlx_runtime::shard_placement` is the adapter.

**Exact, but not always bit-identical.** Every option preserves the maths, yet
the backend may block a narrower shape differently: batch and row splits came
back bit-identical, while out-channel and column splits agreed to ~5e-7. A
cohort that relies on every worker holding identical values must shard along
batch/rows. Executed and checked in
`rlx-runtime/tests/shard_equivalence.rs` — each option is run the way a worker
would be fed it (given only its slice) and recombined, because a shard handed
the wrong slice still returns numbers.

## Install

```toml
[dependencies]
rlx-compile = "0.2"
```

Usually via [`rlx-opt`](../rlx-opt/README.md) or [`rlx`](https://docs.rs/rlx).

## Build / test

```sh
cargo test -p rlx-compile
```

## License

MIT OR Apache-2.0.

[`CompilePipeline`]: https://docs.rs/rlx-compile/latest/rlx_compile/struct.CompilePipeline.html
[`rlx-ir`]: ../rlx-ir/README.md
[`rlx-fusion`]: ../rlx-fusion/README.md
