# rlx-xdna

AMD **XDNA / Ryzen AI NPU** backend for RLX — `Device::Xdna`.

The XDNA NPU is the AI Engine (`aie2`) tile array on AMD Ryzen AI SoCs, driven on
Linux by the in-kernel `amdxdna` driver via `/dev/accel*`. This crate makes it a
first-class RLX device that **runs graphs on the NPU** — the forward-inference
surface of a transformer and a CNN, plus gradient training — validated bit-exact
(cosine for the quantized matmul) against the CPU backend on real hardware (a Ryzen
**Phoenix `npu1`** APU, AIE version 1.1).

## What runs on the NPU

Graph dispatch for `Device::Xdna` lives in `rlx-runtime`'s `XdnaBackend`, which
drives the pieces in this crate. `XdnaBackend::supported_ops()` covers ~39 op kinds:

- **INT8 GEMM** — the fast matmul, ~638 GOP/s peak on Phoenix, via the vendor
  `aie::mmul` microkernel overlay. The AIE array is an INT8/BF16 MAC engine (no
  native f32 datapath), so f32 matmuls are per-row/col quantized (cosine-close).
- **Transformer** — multi-head causal attention, RoPE (NeoX/GptJ), RMS/Layer/
  GroupNorm, softmax, 26 activations, elementwise / binary / reduce / scan.
- **Data movement** — reshape, transpose, narrow, slice, reverse, tile, expand,
  trilu, concat, gather, pad, where, fma, clamp, cast, argmax/argmin, compare.
- **Vision** — 2-D pooling + im2col (a conv is `im2col → INT8 GEMM` on the NPU).
- **Quantization** — Quantize / Dequantize / FakeQuantize (dtype boundary + QAT).
- **Training** — backward graphs decompose to these primitives and run on the NPU
  (including a dynamic-weight GEMM for `xᵀ @ dy`); a host-optimizer SGD loop trains
  with the gradient computed on-device (loss ↓ over steps, validated).

## The pieces

| Module | Role |
|---|---|
| `aie` | **AIE-MLIR emitter** — pure Rust; rlx emits the per-op AIE kernels itself (no Python). |
| `compile` | **Python-free overlay compilation** — drives the native `aiecc` binary → xclbin + instruction stream. |
| `npu_gemm` | **XRT INT8 GEMM executor** — the fast-matmul path. |
| `xrt` | Bindings to AMD **XRT** + the `amdxdna` shim; the default execution path (`xrt` feature). |
| `direct` | **Closest to the metal**: the `amdxdna` DRM-accel ioctl ABI on `/dev/accel*`, no XRT / no C++ shim (`direct` feature, Linux-only). See below. |

## Prerequisites

A Linux host with the `amdxdna` driver (Ryzen AI), plus two userspace toolchains:

1. **AMD XRT** + the `amdxdna` shim (`libxrt_driver_xdna.so`) — the runtime.
2. The native **MLIR-AIE** compiler (`aiecc`) + **Peano** (`llvm-aie`) — used to
   compile op overlays on demand (no Python).

Point the env at them:

```sh
export XILINX_XRT=…/xrt LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
export RLX_XDNA_SHIM=…/libxrt_driver_xdna.so
export AIECC=…/mlir-aie/bin/aiecc PEANO=…/llvm-aie
# pip `mlir_aie` installs: point at the include tree (holds aie_kernels + aie_api),
# since bin/aiecc there isn't at <mlir_aie>/bin:
export RLX_XDNA_AIE_INCLUDE=…/site-packages/mlir_aie/include
```

Sanity-check the two bundled parity suites (each result is compared to the CPU
backend and prints `all NPU graph ops match CPU ✓`):

```sh
cargo run -p rlx-runtime --features xdna --example xdna_graph   # op parity vs CPU
cargo run -p rlx-runtime --features xdna --example xdna_train   # backward + SGD on the NPU
```

## Simple example — one INT8 GEMM (a `Linear` layer)

Build a graph, target `Device::Xdna`, and run. The f32 weight is quantized per
output-channel on the NPU; the activation is quantized per row at run time:

```rust
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

// y = x @ W    (x:[64,64], W:[64,64])
let mut g = Graph::new("linear");
let x = g.input("x", Shape::new(&[64, 64], DType::F32));
let w = g.param("W", Shape::new(&[64, 64], DType::F32));
let y = g.matmul(x, w, Shape::new(&[64, 64], DType::F32));
g.set_outputs(vec![y]);

let mut sess = Session::new(Device::Xdna).compile(g); // compiles the overlay once
sess.set_param("W", &weights);                        // pre-tiled INT8, cached
let out = sess.run(&[("x", &input)]);                 // INT8 GEMM on the AIE array
```

Matmul dims are arbitrary: the backend host-blocks any `M×K×N` over a fixed
`64 × (≤512) × (≤256)` overlay tile (see *Sizes* below). Because the GEMM is
INT8-quantized, the result matches CPU f32 by **cosine ≈ 0.99**, not bit-exactly;
the elementwise / norm / attention ops run in scalar f32 and **are** bit-exact.

## Advanced example — a decoder layer, and training

A full transformer decoder layer runs in one graph — RMSNorm → multi-head causal
attention → residual → RMSNorm → INT8 matmul → activation → residual — dispatched
op-by-op through a lazily-pooled chain of NPU contexts. See
`examples/xdna_graph.rs` (`layer·decoder`, cos 1.0 vs CPU).

**Training on the NPU** (gradient on-device, host optimizer). `grad` produces a
backward graph that decomposes to NPU primitives (including a dynamic-weight GEMM
for `xᵀ @ dy`); compile it once and step:

```rust
use rlx_opt::autodiff::grad;
use rlx_runtime::{Device, Session};

// forward loss graph `g` with param "W"; keep the loss rank-1 (keep_dim) — see limits
let bwd = grad(&g, &[w_id]);                       // dL/dW graph
let mut sess = Session::new(Device::Xdna).compile(bwd);
for _step in 0..epochs {
    sess.set_param("W", &w);
    let grads = sess.run(&[("x", &x), ("t", &t), ("d_output", &[1.0])]);
    for (wi, gi) in w.iter_mut().zip(&grads[0]) { *wi -= lr * gi; } // SGD on host
}
```

`examples/xdna_train.rs` runs this end-to-end (linear regression: loss 1.32 → 0.005
over 40 steps, gradient computed on the NPU).

## Sizes & performance

- **Overlay tile:** one dispatch computes `64 × (KT·64) × (COLS·64)` with
  `KT = clamp(⌈K/64⌉, 1, 8)` and `COLS = clamp(⌈N/64⌉, 1, 4)` → up to
  `64 × 512 × 256`. The host blocks arbitrary `M/K/N` over it; the peak config is
  `DIM=64 KT=8 COLS=4` (K-accumulation across 4 AIE columns).
- **Throughput:** ~638 GOP/s peak INT8 (raw kernel). End-to-end is lower on small
  tiles because host per-row/col quantization dominates (e.g. ~130 GOP/s at 64³);
  it amortizes as `M` grows and across a resident chain.
- **TURBO:** `RLX_XDNA_TURBO=1` clocks the array to maximum DPM — **+11%** measured
  on the INT8 GEMM. Needs root / `CAP_SYS_ADMIN` (a clear one-line warning +
  default DPM otherwise). Benchmark it with `scripts/xdna_turbo_bench.sh`.
- **Hardware:** Phoenix `npu1` ≈ 16 INT8 TOPS across 4 usable AIE columns; static
  weights are pre-tiled + cached in `set_param`, dynamic weights (backward) re-tiled
  per run.
- **Elementwise tiles** stream in chunks ≤ 2048 elems (64 KB tile memory).

## Limitations

- **INT8/BF16 only for matmul** — no native f32 datapath, so f32 GEMMs are
  quantized (cosine ≈ 0.99). Everything else runs in scalar f32 and is bit-exact.
- **Reduce is last-axis only.** Use `keep_dim: true` so a scalar loss/reduce stays
  rank-1 — a **rank-0** tensor has no last axis and the kernels reject it.
- **Chains** are limited by the NPU's concurrent hardware-context count (~5–6): a
  deep graph runs through a bounded LRU pool of contexts (`RLX_XDNA_CHAIN_CAP`,
  default 4), re-opening evicted contexts from cache.
- **Pool / Im2Col / Quantize / Dequantize** are host-computed (coverage so vision
  and int8 graphs *run*); the perf-critical GEMM stays on the NPU.
- **Training:** one `wrt` per compiled graph (a multi-output grad returns only the
  first gradient — use single-`wrt` `grad`); the backward matmul is INT8 so grads
  are approximate (cosine ≈ 0.99, fine for SGD); the optimizer step is host-side.
- **Attention** is standard multi-head + causal; GQA/MQA is handled by upstream
  KV-expansion (no separate `num_kv_heads` on the op).
- **No CPU fallback.** A missing runtime/overlay is a hard `XdnaError`, never a
  silent CPU masquerade — the device either does the work or says it can't.
- **`direct` path is parked** — see below.

## The `direct` path (parked)

`direct` owns the raw `amdxdna` ioctl ABI end-to-end — hwctx / BO / exec / syncobj,
AXLF-PDI extraction, and the TURBO power mode — with **no XRT and no C++ shim**. The
`EXEC_CMD` + syncobj GEMM path is code-complete and byte-verified against XRT, but
it is **parked**: on Phoenix `npu1` (a kernel-managed-queue part with no user-mode
doorbell) the firmware accepts the submission yet won't execute a command XRT runs
fine, and the cause is undiagnosable under Secure Boot lockdown. XRT is the working
execution path; `direct` is retained as the foundation for future hardware
(user-mode-queue / XDNA2) and for the TURBO ioctl.

## License

MIT OR Apache-2.0.
