# rlx-fusion

MIR fusion passes and [`unfuse_fused_for_autodiff`] for RLX. Depends on
[`rlx-ir`] only — no runtime or backend deps.

## What's here

- **`pass`** — `Pass` trait and canonical pass ordering helpers.
- **`fusion`** — pattern-match fusion (`FuseMatMulBiasAct`, `FuseSwiGLU`,
  `FuseAttentionBlock`, residual + norm fusions, …).
- **`fusion_report`** — missed-fusion diagnostics (`MissReason`, `MissedFusion`).
- **`unfuse`** — split fused ops for backends or AD (`unfuse_fused_for_autodiff`).
- **`lower_dot_general`** — XLA-style `DotGeneral` → `MatMul` + reshapes.
- **`control_flow`** — `LowerControlFlow`, while unrolling helpers.
- **`fk_fusion`** — FKL-style passes (`FuseRegionPrologue`, `FuseBatchPreprocess`,
  `MarkBatchSliceRegions`, `DecomposeFusionRegions`). See [`docs/fk-fusion.md`](../docs/fk-fusion.md).
- **`fk_graphs`** — shared test/builder graphs (`batch_narrow_relu_primitive_graph`, …).

## Consumers

- [`rlx-compile`](../rlx-compile/README.md) — orchestrates fusion in
  `CompilePipeline` / `fusion_passes_for_supported`.
- [`rlx-autodiff`](../rlx-autodiff/README.md) — unfuses before reverse-mode AD
  when a fused op has no VJP rule.

## Build / test

```sh
cargo test -p rlx-fusion
```

## License

MIT OR Apache-2.0.

[`unfuse_fused_for_autodiff`]: https://docs.rs/rlx-fusion/latest/rlx_fusion/fn.unfuse_fused_for_autodiff.html
[`rlx-ir`]: ../rlx-ir/README.md
