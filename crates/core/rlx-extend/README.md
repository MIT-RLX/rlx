# rlx-extend

The stable extension surface for extending RLX **from outside the workspace** —
a model crate in [`rlx-models`](https://github.com/MIT-RLX/rlx-models), a
numerics crate, or a one-off research op — **without editing core**.

```rust
use rlx_extend::prelude::*;
```

`rlx-extend` is a thin, slow-moving facade: it only re-exports the public
extension traits and builders from [`rlx-ir`](../rlx-ir) and
[`rlx-flow`](../rlx-flow). Depend on it (rather than reaching into the internal
crates directly) so the layout underneath can be refactored without breaking
your downstream code.

## The seams

- **Block seam** — implement [`LayerStage`](../rlx-flow) (compose primitives via
  `FlowCtx`'s builders: `ctx.matmul`, `ctx.linear`, `ctx.rms_norm`, …) and drop
  it into a flow with `ModelFlow::layer_stage`. No `FlowStage` enum variant, no
  core edit. Publish auxiliary outputs with `FlowCtx::publish_side_output`.
- **Op seam** — implement [`OpExtension`](../rlx-ir) and `register_op`; build
  nodes with `Graph::custom_op` / `try_custom_op`. Provide an `OpExtension::lower`
  rule to decompose to primitives (runs on every backend with no kernel), or
  register a per-backend kernel for a native fast path. Shape inference, autodiff
  (VJP / JVP), and vmap hooks are all part of the trait.
- **Backend / codegen seam** — lives in `rlx-runtime` (a heavier dep, not
  re-exported here): `rlx_runtime::register_backend` against an
  `rlx_runtime::Device`, or consume an `rlx_ir::Graph` directly for an
  ahead-of-time codegen target (the `rlx-fpga` / `rlx-cerebras` pattern).

## What's re-exported

- `rlx_extend::prelude::*` — the flow DSL + block seam (`ModelFlow`,
  `LayerStage`, `FlowCtx`, `BlockAsLayer`, `DynStage`, …) and the op-extension
  seam (`Graph`, `Op`, `OpKind`, `OpExtension`, `register_op`, the
  `Jvp/Vjp/Vmap/Lower` contexts, `Shape`, `DType`, `Node`, …).
- `rlx_extend::rlx_ir` / `rlx_extend::rlx_flow` — the underlying crates, for
  fully-qualified access when the glob is too broad.

The crate is `#![forbid(unsafe_code)]` and carries a test asserting the prelude
surface still resolves, so an upstream rename breaks here first.

See [`docs/extending.md`](../../../docs/extending.md) for worked examples.
## License

MIT OR Apache-2.0.
