# rlx-opt

Facade crate that re-exports [`rlx-fusion`], [`rlx-autodiff`], and [`rlx-compile`]
with the historical `rlx_opt::` module paths. Most callers use [`rlx`] and
import `rlx::opt::*`; depend on `rlx-opt` directly only when you want the
optimizer stack without the runtime.

## Crate split (0.2)

| Crate | Role |
|-------|------|
| [`rlx-fusion`] | MIR fusion passes, `Pass` trait, `unfuse_fused_for_autodiff` |
| [`rlx-autodiff`] | `grad_with_loss`, `jvp`, `hvp`, `vmap`, `prepare_graph_for_ad` |
| [`rlx-compile`] | `CompilePipeline`, legalization, memory plan, precision / PTQ |

Implementation lives in those crates; `rlx-opt` only wires backward-compatible
`rlx_opt::fusion`, `rlx_opt::autodiff`, `rlx_opt::legalize`, etc.

## Features

| Feature | Enables |
|---------|---------|
| `compile` *(default)* | `rlx-compile` — HIR → MIR → LIR pipeline |
| `training` *(default)* | `rlx-autodiff` — reverse / forward AD, vmap |
| `full` | both |

## Fusion pipeline (via `rlx-compile`)

Fusion is **backend-aware**: `fusion_passes_for_supported` selects passes from a
backend's `OpKind` claim set so the optimizer never emits fused ops the target
cannot lower (e.g. Metal may skip `FuseAttentionBlock`).

Typical order:

1. **Constant folding** — fold compile-time-known subgraphs.
2. **Fusion passes** — gated pattern fusions + elementwise regions.
3. **Legalize** — broadcast materialization, backend-specific rewrites.
4. **Memory plan** — liveness → arena buffer assignment.
5. **DCE** — dead-code elimination (last in the fusion pipeline).

See [`rlx-fusion/README.md`](../rlx-fusion/README.md) and
[`rlx-compile/README.md`](../rlx-compile/README.md) for crate-local detail.

## Install

```toml
[dependencies]
rlx-opt = "0.2"
```

## Build / test

```sh
cargo build -p rlx-opt
cargo test  -p rlx-fusion -p rlx-autodiff -p rlx-compile
```

## Gotchas

- Pass order matters: const-fold before fusion; memory plan after fusion.
- New fused ops need `Op` + infer + verifier + every backend thunk you target.
- `precision` inserts `Cast` nodes; some are eliminated by fusion peepholes.

## License

GPL-3.0-only.

[`rlx`]: https://docs.rs/rlx
[`rlx-fusion`]: ../rlx-fusion/README.md
[`rlx-autodiff`]: ../rlx-autodiff/README.md
[`rlx-compile`]: ../rlx-compile/README.md
