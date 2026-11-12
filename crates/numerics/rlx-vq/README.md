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
- `emeans` — the e-means trainer (`rlx_vq::emeans`, see below).
- `emeans-device` — adds the device-resident trainer (`emeans::device`;
  depends on `rlx-runtime`).

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

## e-means (`emeans` feature)

`rlx_vq::emeans` implements **e-means**, the mini-batch k-means of

> S. Bessa, P.-E. Mazaré, G. Szilvasy, H. Touvron, D. Picard, M. Douze and
> H. Jégou, *e-means: outperforming k-means (in one-tenth the epochs)*,
> arXiv:2611.09501, 2026. <https://arxiv.org/abs/2611.09501>

Each batch updates every cluster's mass `N_j` and position `c_j` with an
exponential moving average (`N ← γN + αn`, `c ← c + (αn/N)(s/n − c)`), with
`α` cosine-annealed to `0.1·α₀`, dead clusters revived by splitting
mass-weighted donors every 5 steps, and exact closing means over the last
epoch. Batch size and `α₀` come from the paper's fitted scaling laws, so the
epoch budget is the only setting.

```rust
use rlx_vq::emeans::{Emeans, EmeansConfig};

let mut em = Emeans::new(d, k, EmeansConfig::default()); // 5 epochs
let (centroids, labels) = em.train(&x, None);           // x: [n, d] f32
let (dist, nearest) = em.assign(&x_new);
```

- **Host trainer** (`Emeans`) — fused nearest-centroid search (rayon, no
  `[B, k]` matrix) plus the EMA, revival and closing on the host. It follows
  faiss's `contrib/e_means` `NumpyEmeans` step for step, including its NumPy
  MT19937 streams via `rlx-rng`: on the same data and seed the per-epoch
  objectives agree to 6 digits, and at `d = 8` the centroids and labels are
  bit-identical. `scripts/emeans_faiss_reference.py` runs that comparison.
- **Step graph** (`step_graph`, `step_graph_with`) — one step as an rlx graph
  (matmul → ArgMin → ScatterAdd of `[x | 1]` → EMA), so any backend runs it;
  `Emeans::update_from_step` advances the schedule from its outputs.
  `step_graph_with` runs the assignment in f64, f16 or bf16. The assignment
  centres rows and centroids on the batch mean first: the argmin is the
  same, but far from the origin `‖c‖²` and `x·c` swamp the gaps between
  centroids, so f16 overflowed and an XDNA bf16 matmul labelled 8% of rows
  right at offset 300.
- **Initialization** — `EmeansConfig::init` is `Init::RandomRows` (the
  reference's, and the default) or `Init::KMeansPlusPlus`. Random rows can
  leave a well-separated region without a centroid, which revival does not
  repair (seed 1 on 20 blobs ends at 4.8× the MSE of other seeds, in faiss
  too). Over 5 seeds on 4 blob sets, k-means++ lowered the mean MSE 5–37% and
  the worst case up to 74%, for `O(32·k²·d)` seeding (29 ms at k = 256,
  d = 32).
- **Device-resident trainer** (`emeans::device::DeviceEmeans`, feature
  `emeans-device`) — the dataset, centroids, masses and epoch accumulators
  live in GPU handles that the step graph feeds in place; a step uploads only
  its row indices, row weights and `α` and reads nothing back. The host reads
  the `k` masses every 5 steps for revival (with the host trainer's RNG, so
  the same clusters split), and the accumulators once per epoch. Resident on
  CPU, Metal, MLX, wgpu, Vulkan, CUDA and ROCm; elsewhere the same graph runs
  with the state passed through. On the XDNA NPU, `device_options` picks the
  bf16 matmul over the default INT8.

  ```rust
  use rlx_vq::emeans::{EmeansConfig, device::DeviceEmeans};
  let mut em = DeviceEmeans::new(rlx_runtime::Device::Metal, d, k, EmeansConfig::default());
  let (centroids, labels) = em.train(&x, None);
  ```
- **Playground datasets** — `tests/emeans_playground.rs` ports the 24
  generators of the paper's [browser playground](https://e-means.github.io/playground.html)
  (all but the Meta logo) and races 5 e-means epochs against 25 Lloyd
  iterations on held-out rows, as the page does, plus the device trainer with
  the page's bf16 assignment on every backend. At the page's size
  (`EMEANS_PLAYGROUND_N=100000 EMEANS_PLAYGROUND_K=500`) e-means wins 24/24.
- **Parity** — `tests/emeans.rs` runs the step graph and a full graph-driven
  run on every available backend at every precision (`CompileOptions::
  number_format` f64/f64x2/f32/f16/bf16/f8, plus native casts), each case in
  its own process. Enable backends through `rlx-runtime` features:

  ```sh
  cargo test -p rlx-vq --release --test emeans \
      --features emeans,metal,rlx-runtime/vulkan,rlx-runtime/coreml -- --nocapture
  cargo test -p rlx-vq --release --test emeans \
      --features emeans,rlx-runtime/cuda,rlx-runtime/rocm,rlx-runtime/xdna -- --nocapture
  ```

  `RLX_REQUIRE_DEVICE=1` turns unsupported combinations into failures.

## License

MIT OR Apache-2.0.
