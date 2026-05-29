# rlx-wgpu

Cross-platform GPU backend via the [wgpu](https://wgpu.rs/) crate.
Single backend serves Metal (macOS), Vulkan (Linux), DirectX 12
(Windows), and WebGPU (browsers). WGSL kernels, pure Rust deps —
no FFI, no submodules.

## What's here

- **WGSL kernels** — fp32 matmul (8×8 tile), cooperative-matrix
  matmul (32×32 tile, `simdgroup_matrix` / `KHR_cooperative_matrix`),
  f16-storage matmul.
- **`device.rs`** — wgpu instance/adapter/device singleton. Sync wrapper
  via `pollster::block_on` so the rest of the backend matches the
  rlx-cpu / rlx-metal / rlx-mlx synchronous shape.
- **`buffer.rs` / `Arena`** — single contiguous storage buffer; per-
  node offsets from `rlx-opt::memory::plan_memory_aligned`. f32 host
  I/O via `queue.write_buffer` / staging-buffer-mapped readback.
- **`kernels/matmul.wgsl`** — fp32 matmul, one workgroup per 8×8 output
  tile. Functional, not optimized.
- **`kernels/mod.rs`** — `OnceLock`-cached pipeline + bind-group layout.
  First dispatch pays the WGSL → SPIR-V/MSL/HLSL translation cost
  (~ms); subsequent dispatches reuse the compiled pipeline.
- **`backend.rs`** — `WgpuExecutable`. Anything not in the supported op
  set panics at compile time with a clear "fall back to CPU/Metal/MLX"
  diagnostic.
- **FFT** — `fft_gpu.wgsl` multi-kernel pow-2 dispatch (in-pass with
  per-op uniforms). Non-pow2 / f64 / C64 use `fft_host.rs` partial sync.
  `RLX_BENCH_DISPATCH_ONLY=1` skips output readback for micro-benchmarks.
  `RLX_DISPATCH_REPORT=1` logs `fft_gpu` vs `fft_host` step counts.

## Op coverage

Today: `MatMul` (2D), `Op::Input`, `Op::Param`, `Op::Constant`.
Anything else fails at compile time with a clear "fall back to
CPU/Metal/MLX" diagnostic.

The roadmap is to land ops in BERT-shaped order: element-wise binary,
layer norm, softmax, attention, gather, transpose. Adding an op means:
WGSL source, a `MatmulPipeline`-style cache entry, a `Step` variant, a
dispatch in `run`. PRs welcome.

## Install

```toml
[dependencies]
rlx-wgpu = "0.2"
```

Or via [`rlx`](https://crates.io/crates/rlx)'s `gpu` feature.

## Build / test

```sh
cargo build -p rlx-wgpu --release
cargo test  -p rlx-wgpu --release
```

Through `rlx-runtime`:

```sh
cargo build -p rlx-runtime --features gpu --release
```

## Status

Functional, less battle-tested than `rlx-metal` / `rlx-mlx` on Apple
Silicon. Coop-matrix paths under active validation. The matmul kernel
is correctness-first — order of magnitude slower than what's possible.

## Gotchas

- Wgpu is async; we wrap with `pollster::block_on` for sync semantics.
  Future work: an async `commit_no_wait`-style API to amortize submit
  latency, mirroring rlx-metal.
- The matmul kernel is correctness-first. It loops over K per thread
  with no register blocking or shared-memory tiling — order of
  magnitude slower than what's possible. Optimization comes after
  the op set is broad enough to run a real model.
- Shader compilation is lazy + cached via `OnceLock`. First dispatch
  pays the WGSL → SPIR-V/MSL/HLSL translation cost (~ms); subsequent
  dispatches reuse the compiled pipeline.

## License

GPL-3.0-only.
