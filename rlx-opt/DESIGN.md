# rlx-opt DESIGN

## What this crate owns

Graph rewrites between IR construction and backend lowering. Nothing
here knows about kernels — the output is still a `Graph`, just a
better-shaped one.

## Decisions

**Pattern-match fusion, not a generic rewriter.** Each fusion pass
hard-codes the IR shape it recognizes (e.g., FuseMatMulBiasAct knows
`MatMul → Add → Activation`). A generic e-graph would be more
elegant but the patterns are stable and debugging concrete matchers
is easier when fusion goes wrong.

**Memory planner is greedy first-fit on size-sorted buffers.** Not
optimal — the textbook solution is GlobalDecreasingSizeBestFitHeap
(XLA). In practice the workloads we benched (BERT, Nomic) don't
allocate enough distinct tensors for the difference to matter.
Revisit when arena waste shows up in profiles.

**View aliasing is a post-pass.** The planner allocates root buffers
first, then `pure_view_offset` walks view nodes (Reshape / same-dtype
Cast / axis-0 Narrow) and aliases their slots to the root + offset.
Doing it in one pass would be more elegant but harder to reason
about live-range overlap.

**`run_passes` verifies between every pass in debug.** Catches
optimizer bugs at the boundary that introduced them; release builds
compile the verify call out (zero overhead).

## What doesn't work / why

- **`fuse_attention_block` (the IR-level pass) is a no-op.** The
  actual fusion happens at the *thunk* level inside `rlx-cpu` /
  `rlx-metal`'s `compile_thunks` because positional matching across
  variable Narrow / Rope counts is awkward in graph rewriters. The
  IR-level placeholder remains so the public API (FusedAttnBlock)
  exists for future cleanup.
- **No cost-model-driven pass ordering.** Passes run in a fixed
  sequence (DCE last, fusion before memory planning). For larger
  optimization spaces (e.g., LLM serving) we'd want an actual cost
  model picking pass order. Not needed for the current workloads.

## Cross-crate contract

- Memory planner emits `MemoryPlan { arena_size, assignments,
  schedule }`. Both backends consume this; arena allocates exactly
  `arena_size` bytes once.
- `is_pure_view(graph, node)` is the single predicate backends use
  to decide "emit Nop, the planner aliased this slot." Don't
  duplicate the view-detection logic in backends.
