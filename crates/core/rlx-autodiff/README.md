# rlx-autodiff

JAX-shaped transforms on RLX MIR: reverse-mode [`grad_with_loss`], forward-mode
[`jvp`] / [`hvp`], and [`vmap`] (leading-axis batching).

Depends on [`rlx-ir`] and [`rlx-fusion`] (unfuse fused ops before AD when needed).

## What's here

- **`autodiff`** — reverse-mode AD; fused-op VJPs, control flow (`If` / `While` /
  `Scan`), custom-fn inlining for AD.
- **`autodiff_fwd`** — forward-mode AD; [`hvp`] runs `jvp` on a decomposed backward graph
  (`prepare_grad_graph_for_jvp` — no caller `d_output` seed).
- **`compose`** / **`decompose_backward`** / **`higher_order`** — stack reverse-mode
  for 2nd/3rd+ derivatives (`nth_order_grad`, `directional_nth_grad`, `cse`,
  `HigherOrderOptions`, `nth_order_grad_with_options`).
- Set `RLX_HIGHER_ORDER_NO_FUSE=1` to disable elementwise-region fusion after each
  differentiation layer (marginal on broadcast-heavy grad graphs; try on wgpu).
- **`mlip`** — `grad_subgraph`, `build_force_energy_loss` for force+energy training.
- **`activation_deriv`** — closed-form `f'(x)` for backward decomposition.
- **`prepare_ad`** — `prepare_graph_for_ad`, MIR/module preparation.
- **`vmap`** — batched function transform.
- **`legalize_reduce`** — reduce legalization helpers for training graphs.
- **FFT AD** — VJP/JVP rules for `Op::Fft` (unitary / norm-aware).

## Feature

Enable via `rlx-opt` with feature `training` (default), or depend on this crate
directly for a minimal AD-only dep tree.

## Build / test

```sh
cargo test -p rlx-autodiff
```

## Benchmarks

Third-order cubic-sum sweep across backends and batch sizes:
[`docs/benchmarks/higher-order-ad.md`](../docs/benchmarks/higher-order-ad.md).

```sh
cargo run -p rlx-bench --release --example bench_nth_order --features metal,mlx,gpu
./rig.sh bench-nth-order both
```

## License

MIT OR Apache-2.0.

[`grad_with_loss`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.grad_with_loss.html
[`jvp`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.jvp.html
[`hvp`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.hvp.html
[`vmap`]: https://docs.rs/rlx-autodiff/latest/rlx_autodiff/fn.vmap.html
[`rlx-ir`]: ../rlx-ir/README.md
