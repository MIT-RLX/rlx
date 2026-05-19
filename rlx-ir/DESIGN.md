# rlx-ir DESIGN

## What this crate owns

The IR vocabulary. Once an op is in `Op` and shape inference works, it
travels through every other crate unchanged.

## Decisions

**Op is one big enum, not a trait.** Means O(n_op_kinds) match arms in
every backend, but pattern matching on the enum is what makes fusion
patterns readable. A trait would force dynamic dispatch on the hot
schedule path.

**Shape carries dtype.** Avoids the "did I forget to also pass the
dtype?" bug class. Cost: some ops technically don't need a dtype on
their output (Bool from compare) but pay for the field anyway.

**`Dim::Dynamic` exists but is not yet used by any model.** Symbolic
dims (#54 in PLAN.md) would let one compiled graph serve any seq
length; today everyone passes `Dim::Static`. Keeping the variant
costs nothing and unblocks the future.

**`MaskKind` lives on `Op::Attention`, not as a separate Op.** One
attention kernel handles all variants by branching on the kind in
the inner loop. Adding a new mask kind = one match arm, not a new op.

**`QuantMap` is graph-level, not per-Node.** Quantization metadata is
sparse; putting it on Node would bloat every node for the rare
annotation case.

## What doesn't work / why

- **No `serde::Serialize` on Graph.** The IR was designed for it
  (everything is plain data) but nothing in tree consumes a
  serialized graph yet, and `Box<Graph>` inside `Op::If` / `While`
  needs custom handling we haven't done. AOT compilation (#16) would
  push us into this.
- **`verify.rs` doesn't check shape inference exhaustively.** It
  catches structural bugs (input count, DAG-ness) but trusts each
  builder to set the right output shape. A future pass should
  re-derive expected shape and diff.

## Cross-crate contract

- Adding a new `Op` variant ripples to: `infer.rs` (shape),
  `graph.rs` (builder), `verify.rs` (input count rule), both
  backends' `compile_thunks`, fusion patterns, cost model. ~6 files.
  No way around this without runtime dispatch.
