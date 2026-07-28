# rlx-vq

Fused vector-quantization kernel for RLX — nearest-codebook assignment
(`cdist → argmin`) that beats the `matmul + argmin` composition.
**Downstream package**: registers against rlx's custom-op scaffold
without touching `rlx-ir`, `rlx-opt`, or `rlx-runtime` core.

## What's here

- **`vector_quantize(g, x, codebook, metric, target)`** — nearest-codebook
  assignment. Returns `(indices, quantized)`. The fused CPU path computes
  `argmin_j (‖C_j‖² − 2·x·C_jᵀ)` with a running per-row reduction — it never
  materializes the `[N, K]` distance matrix and parallelizes over rows.
- **`Metric`** — `L2` (squared Euclidean) nearest-codebook distance.
- **`Target`** — device-aware lowering (see below).
- **`register()`** — call once per process to publish `Op::Custom("rlx.vq_assign")`
  via the framework kernel registries (`OpExtension` + CPU / Metal kernels).

## Device-aware lowering (`Target`)

VQ's core is a matmul, so the fastest implementation differs by backend:

- **`Target::Cpu`** → the fused custom op (no `[N,K]` matrix, rayon over rows).
  Measured **1.5–5.4×** faster than the `matmul + argmin` composition, the win
  growing with codebook size `K`.
- **`Target::Gpu`** → the on-device composition (MPS matmul + argmin). On Metal
  the matrix units make this **~10×** faster than the CPU path, so it is the
  right choice on Metal / wgpu.

`rlx-metal` additionally recognizes `Op::Custom("rlx.vq_assign")` and dispatches
a native on-GPU MSL kernel (one threadgroup / row, `float4` cooperative argmin),
so a `Target::Cpu` graph accidentally run on Metal is only ~2–4× slower than the
composition rather than ~100× of a host-callback ABI.

## Features

- `cpu` *(default)* — registers the fused `CpuKernel`.
- `metal` — registers the Metal host-callback `MetalKernel` (macOS only); the
  native MSL kernel in `rlx-metal` is preferred when available.

## Install

```toml
[dependencies]
rlx-vq = "0.2"
```

## Quickstart

```rust
rlx_vq::register();   // once per process

let (idx, q) = rlx_vq::vector_quantize(
    &mut g, x, codebook, rlx_vq::Metric::L2, rlx_vq::Target::Gpu,
);
```

## License

MIT OR Apache-2.0.
