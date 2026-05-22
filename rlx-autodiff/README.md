# rlx-autodiff

JAX-shaped transforms on RLX MIR: reverse-mode [`grad_with_loss`], forward-mode
[`jvp`] / [`hvp`], and [`vmap`] (leading-axis batching).

Depends on [`rlx-ir`] and [`rlx-fusion`] (unfuse fused ops before AD when needed).

## What's here

- **`autodiff`** — reverse-mode AD; fused-op VJPs, control flow (`If` / `While` /
  `Scan`), custom-fn inlining for AD.
- **`autodiff_fwd`** — forward-mode AD.
- **`prepare_ad`** — `prepare_graph_for_ad`, MIR/module preparation.
- **`vmap`** — batched function transform.
- **`legalize_reduce`** — reduce legalization helpers for training graphs.

## Feature

Enable via `rlx-opt` with feature `training` (default), or depend on this crate
directly for a minimal AD-only dep tree.

## Build / test

```sh
cargo test -p rlx-autodiff
```

## License

GPL-3.0-only.

[`grad_with_loss`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.grad_with_loss.html
[`jvp`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.jvp.html
[`hvp`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.hvp.html
[`vmap`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.vmap.html
[`rlx-ir`]: ../rlx-ir/README.md
