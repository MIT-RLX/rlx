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

## Kernel dispatch transparency

`prepare_graph_for_backend_with_report` and friends produce a
`KernelDispatchReport` (native vs common-IR vs rewritten vs unsupported).
The runtime uses the same legalization path as compile; see the root
[`README.md`](../README.md#kernel-dispatch-and-transparency) for env vars
(`RLX_DISPATCH_REPORT`, `RLX_KERNEL_DISPATCH`).

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

GPL-3.0-only.

[`CompilePipeline`]: https://docs.rs/rlx-compile/latest/rlx_compile/struct.CompilePipeline.html
[`rlx-ir`]: ../rlx-ir/README.md
[`rlx-fusion`]: ../rlx-fusion/README.md
