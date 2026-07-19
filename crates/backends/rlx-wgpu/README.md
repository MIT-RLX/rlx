# rlx-wgpu

Cross-platform GPU backend via the [wgpu](https://wgpu.rs/) crate.
Single backend serves Metal (macOS), Vulkan (Linux), DirectX 12
(Windows), and WebGPU (browsers). WGSL kernels, pure Rust deps —
no FFI, no submodules.

## What's here

- **WGSL kernels** — fp32 matmul (8×8 tile), cooperative-matrix
  matmul (32×32 tile, `simdgroup_matrix` / `KHR_cooperative_matrix`),
  f16-storage matmul.
- **`device.rs`** — wgpu instance/adapter/device singleton. Platform
  defaults: DX12 (+ Vulkan fallback) on Windows, Vulkan on Linux/WSL,
  Metal (+ MoltenVK) on macOS. Override with `WGPU_BACKEND`. Sync wrapper
  via `pollster::block_on` so the rest of the backend matches the
  rlx-cpu / rlx-metal / rlx-mlx synchronous shape.
- **`buffer.rs` / `Arena`** — logical activation arena with per-node offsets
  from `rlx-opt::memory::plan_memory_aligned`. When the plan exceeds wgpu
  `max_buffer_size` (~4 GiB), the arena is **striped** across multiple ≤4 GiB
  buffers (`extra_shards`); each stripe keeps a 64 MiB stage reserve so
  cross-shard operand copies do not clobber live activations. f32 host I/O
  via `queue.write_buffer` / pooled MAP_READ staging readback
  (`ReadbackStaging`, `read_f32_many_pooled`). Env: `RLX_WGPU_SHARD_LOG=1`.
  Packed quant weights may live in a separate `weight_buffer` (`from_plan_split`).
- **`kernels/matmul.wgsl`** — fp32 matmul, one workgroup per 8×8 output
  tile. Functional, not optimized.
- **`kernels/mod.rs`** — `OnceLock`-cached pipeline + bind-group layout.
  First dispatch pays the WGSL → SPIR-V/MSL/HLSL translation cost
  (~ms); subsequent dispatches reuse the compiled pipeline.
- **`backend/`** — `WgpuExecutable`, decomposed into `compile`/`run`/`set`/
  `dispatch` submodules (each a sibling `impl WgpuExecutable`; see `mod.rs`).
  Unsupported ops panic at compile time with a clear "fall back to
  CPU/Metal/MLX" diagnostic.
- **GGUF dequant** — `kernels/dequant_gguf.wgsl` + `gguf_gpu.rs`: fused
  windowed GEMV for Q4_K/Q6_K/Q1_0 (scratch-free, shard-safe). Other schemes
  dequant into scratch then `matmul_bt` when the arena is a single buffer;
  sharded / split-weight arenas fall back to `gguf_host`. Grouped MoE GGUF
  uses the GPU path when scratch fits and the arena is unsharded. Scheme ids
  match Metal/CUDA. Details:
  [docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md).
- **FFT** — `fft_gpu.wgsl` multi-kernel pow-2 dispatch, all stages in one
  compute pass with a **per-stage uniform pool** (each stage binds its own
  slot; a *shared* uniform would alias across stages because `write_buffer`
  lands before the pass runs). Batch (`outer`) spills across the y/z grid dims
  past wgpu's 65535 per-dimension cap. Non-pow2 / **sharded arenas** use
  `fft_host.rs` (host sync); f64 / C64 are rejected — the wgpu arena is f32-only.
  `RLX_BENCH_DISPATCH_ONLY=1` skips output readback for micro-benchmarks.
  `RLX_DISPATCH_REPORT=1` logs `fft_gpu` vs `fft_host` step counts.

## Op coverage

Broad inference surface (matmul, norms, attention, conv, vision, RNN/SSM,
quantized matmul, etc.) — see [docs/op-coverage.md](../../docs/op-coverage.md).
GGUF `DequantMatMul` uses the GPU path when arena scratch fits; otherwise
CPU dequant via `gguf_host`. MoE grouped GGUF remains host-only on WGPU.

## Install

```toml
[dependencies]
rlx-wgpu = "0.2"
```

Or via [`rlx`](https://crates.io/crates/rlx)'s `gpu` feature.

## Cost-model calibration

When a wgpu adapter is available, `rlx_wgpu::calibrate::Calibration::load_or_measure()`
benchmarks a 512³ matmul and caches results at `~/.cache/rlx/wgpu-calib-<adapter>.json`.
Feeds `WgpuCostModel` in `rlx-runtime` for backend ranking.

## Cooperative-matrix matmul (Vulkan / DX12 / Metal)

On discrete GPUs (Vulkan, DX12), aligned f32 matmul with Param weights
auto-selects **CoopF16Vk** when the adapter reports 16×16 f16 cooperative
matrix support:

- **`matmul_coop_f16_vulkan.wgsl`** — 16×16 f16 tensor-core tiles,
  `coopLoad` on A + `coopLoadT` on B when N ≤ 768; optional widen variant uses
  `coopLoad` on B for N > 768 when `RLX_WGPU_COOP_F16_VK_LARGE_N=1`. By default
  N > 768 dispatches `matmul_wide_nv` instead (see auto wide fallback).
- **`matmul_qkv_coop_f16_vk.wgsl`** — split-write Q/K/V variant (same
  numerics, one dispatch instead of matmul + Narrow×3).
- **Auto wide fallback** — at `set_param`, an oscillation score on Param B
  selects wide f32 matmul for stress weights; **N > 768** also defaults to wide
  f32 (`matmul_wide_nv`) because RTX coop B-load at large N stays ~4e-3 vs f16-ref.
  Opt back into coop for large N with `RLX_WGPU_COOP_F16_VK_LARGE_N=1`.
- Fallback wide path: **`matmul_wide_nv.wgsl`** (64×64 tiles).

When a matmul operand is an `Activation`, the lowering **host-mirrors** it
each `run()` (`apply_activation` + `write_f32`, same f16 shadow as Params)
and routes the matmul through **F32 wide** (`matmul_wide_nv`). CoopF16Vk
remains on Input/Param-only operands (BERT QKV weights, etc.). A GPU
`cast_f32_to_f16` pre-pass still handles non-unary computed tensors when
CoopF16Vk applies. Pass flushes between unary → cast → coop matmul keep
Vulkan/DX12 visibility correct.

### Environment flags

| Variable | Effect |
|----------|--------|
| `WGPU_BACKEND` | `vulkan` (default Linux/Windows), `dx12` (Windows), `metal` (macOS) |
| `RLX_WGPU_NO_COOP_F16_VK=1` | Force `matmul_wide_nv` instead of CoopF16Vk |
| `RLX_WGPU_COOP_F16_VK_OSC_THRESH` | Oscillation score threshold for auto wide fallback (default `0.35`) |
| `RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE=1` | Disable oscillation-based wide fallback |
| `RLX_WGPU_COOP_F16_VK_NO_F32ACC=1` | Disable f16×f16→f32 coop accumulator kernels |
| `RLX_WGPU_COOP_F16_VK_FORCE_WIDE=1` | Always wide-fallback CoopF16Vk matmul at dispatch |
| `RLX_WGPU_COOP_F16_VK_LARGE_N=1` | Keep CoopF16Vk for N > 768 (default: wide f32 for accuracy) |
| `RLX_WGPU_COOP_F16_VK_LOAD_T=1` | Force `coopLoadT` on B when `RLX_WGPU_COOP_F16_VK_LARGE_N=1` and N > 768 |
| `RLX_WGPU_NO_COOP_F32=1` | Disable CoopF32 on Metal |
| `RLX_WGPU_FORCE_COOP_F32=1` | Opt-in CoopF32 on Vulkan (portable 8×8; correctness not validated) |
| `RLX_WGPU_FORCE_INPUT_UPLOAD=1` | Always upload inputs (disable hash-based skip) |
| `RLX_BENCH_DISPATCH_ONLY=1` | Skip output readback (micro-benchmarks) |
| `RLX_WGPU_F16_WEIGHTS=1` | Legacy f16-storage matmul experiment |
| `RLX_WGPU_SCHEDULE=1` | Log compiled dispatch schedule |
| `RLX_DISPATCH_REPORT=1` | Per-step dispatch report |
| `RLX_WGPU_NO_PACKED_BSHD_ATTN=1` | Disable packed QKV attention stride path |
| `RLX_WGPU_DUMP_NODES=1` | Per-node max-abs dump after run (debug) |

**EEG-DINO parity notes:** compile runs `LegalizeBroadcast` before fusion
(mid-axis `[1,C,1,D]+[1,C,P,D]` needs `Expand`) and unfuses
`ElementwiseRegion` on wgpu (region kernel only supports trailing broadcast).
`Activation::Gelu` uses exact `erf` in `unary.wgsl` (not the tanh approx).

## Build / test

```sh
cargo build -p rlx-wgpu --release
cargo test  -p rlx-wgpu --release
cargo test  -p rlx-wgpu --test coop_f16_vk_correctness --release -- --nocapture
cargo test  -p rlx-wgpu --test coop_mat_probe --release -- --nocapture
# DX12-only (set WGPU_BACKEND=dx12 on Windows):
cargo test  -p rlx-wgpu --test coop_f16_vk_dx12 --release -- --nocapture
```

Through `rlx-runtime`:

```sh
cargo build -p rlx-runtime --features gpu --release
```

## Status

Functional, less battle-tested than `rlx-metal` / `rlx-mlx` on Apple
Silicon. CoopF16Vk on Vulkan/DX12 is validated on RTX-class GPUs via
`coop_f16_vk_correctness` and `coop_f16_vk_bert_baseline` (gentle BERT
weights vs f16-ref, oscillating weights auto-wide to f32). The legacy fp32
tile matmul kernel remains correctness-first for unaligned shapes.

## Gotchas

- **Scalar-output latency:** End-to-end `run()` on tiny graphs is often
  readback-bound (~1.3 ms on MoltenVK; ~100–150 µs on DX12/Vulkan).
  Kernels may be sub-µs; use `RLX_BENCH_DISPATCH_ONLY=1` to time dispatch
  without readback. See [`docs/benchmarks/higher-order-ad.md`](../docs/benchmarks/higher-order-ad.md).
- **WSL:** Linux cargo on the rig uses `~/rlx-workspace-mirror/rlx` (ext4),
  not virtio `D:` directly. `rig.sh bench-nth-order` syncs new examples there.
- Wgpu is async; we wrap with `pollster::block_on` for sync semantics.
  `run()` fuses output readback copies into the final compute encoder
  submit and schedules host mapping via `CommandEncoder::map_buffer_on_submit`
  (wgpu 29+), then reuses a pooled MAP_READ staging buffer across runs.
- The matmul kernel is correctness-first. It loops over K per thread
  with no register blocking or shared-memory tiling — order of
  magnitude slower than what's possible. Optimization comes after
  the op set is broad enough to run a real model.
- Shader compilation is lazy + cached via `OnceLock`. First dispatch
  pays the WGSL → SPIR-V/MSL/HLSL translation cost (~ms); subsequent
  dispatches reuse the compiled pipeline.

## License

GPL-3.0-only.
