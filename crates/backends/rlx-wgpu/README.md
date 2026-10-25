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
  [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
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
quantized matmul, etc.) — see [docs/op-coverage.md](../../../docs/op-coverage.md).
GGUF `DequantMatMul` uses the GPU path when arena scratch fits; otherwise
CPU dequant via `gguf_host`. MoE grouped GGUF remains host-only on WGPU.
`Op::DoubleSingle` (f64 as `(hi, lo)` f32 pairs, `double_single.wgsl`) runs
native.

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

## Matmul kernel selection

Plain WGSL, no extensions, before any cooperative-matrix path:

- **Decode (`m ≤ 4`) → `matmul_gemv`.** A 256-thread workgroup owns 32
  adjacent columns and splits K eight ways, so B is read in coalesced row
  segments and up to four A rows share each load. The 32×32-tile `matmul`
  spent 31/32 of its work on padding rows at `m = 1`. Columns per workgroup
  follow the output width (`RLX_WGPU_GEMV_COLS=32|16|8` pins it);
  `RLX_WGPU_NO_GEMV=1` turns the path off.
- **Wide tiles → `matmul_wide_vec4`** (default): the 64×64 tile with `vec4`
  tile storage and a `vec4` FMA inner loop, measured 2.8× `matmul_wide` on an
  M4 Pro and bit-identical to it. `RLX_WGPU_WIDE_STD=1` / `RLX_WGPU_WIDE_NV=1`
  pin the older scalar kernels.
- **GGUF.** Packed decode GEMVs (Q4_K / Q5_K / Q6_K, the 32-element block
  formats, BF16) read the weight in place in the arena; Q8_0 with 8 or more rows takes a tiled GEMM that reads
  each weight block once per 64-row tile (`RLX_WGPU_Q8_0_GEMM_DISABLE=1` keeps
  the per-row GEMV).
- **f16 weight mirror.** A large f32 weight is read from its f16 shadow only
  when it can't share a storage binding with its activations (VirtIO / DX12 cap
  bindings near 2 GiB); `RLX_WGPU_F16_MIRROR=1` uses the shadow anyway, for the
  bandwidth.

Per-shape choices come from `rlx-gpu-dispatch`; shipped per-arch rows load
below the user cache (`RLX_GPU_NO_SHIPPED_TUNING=1` skips them), and
`examples/wgpu_tune_dispatch.rs` measures new ones.

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
| `RLX_WGPU_FORCE_INPUT_UPLOAD=1` | No-op (inputs are always re-uploaded; kept for older scripts) |
| `RLX_BENCH_DISPATCH_ONLY=1` | Skip output readback (micro-benchmarks) |
| `RLX_WGPU_F16_WEIGHTS=1` | Legacy f16-storage matmul experiment |
| `RLX_WGPU_SCHEDULE=1` | Log compiled dispatch schedule |
| `RLX_DISPATCH_REPORT=1` | Per-step dispatch report |
| `RLX_WGPU_NO_PACKED_BSHD_ATTN=1` | Disable packed QKV attention stride path |
| `RLX_WGPU_DUMP_NODES=1` | Per-node max-abs dump after run (debug) |
| `RLX_WGPU_FORCE_HOST=1` | Host-fallback all risky SPIR-V paths |
| `RLX_WGPU_SHARD_GPU=1` | Opt out of discrete/sharded host packing (unsafe for Kitten NSF) |
| `RLX_WGPU_DISCRETE_HOST=1` | Extra discrete hosting via `wgpu_prefer_host_fallback` |
| `RLX_WGPU_HOST_BUFFER_COPY=1` | Force host round-trip for BufferCopy |
| `RLX_WGPU_HOST_EAGER_H2D=1` | Write every HostOp/Conv output to the device immediately (default: defer until Expand/Concat/GPU/readback) |
| `RLX_WGPU_HOST_MATMUL=1` | Pack MatMul as HostOp |
| `RLX_WGPU_CONV_HOST=1` | Force Conv2d host path |
| `RLX_WGPU_STEP_PROFILE=1` | Per-step GPU timestamps, one op per pass; prints a table every `RLX_WGPU_STEP_PROFILE_EVERY` runs (default 32). Ranks steps; not production timing |

**Discrete Vulkan/DX12 (Kitten NSF):** Binary/Activation/Where/Fma, norms,
Expand/Concat/Transpose/Narrow, and Conv stay on the host (SPIR-V elementwise
and structure paths collapse audio). MatMul and unsharded BufferCopy stay on
GPU. Host→Host streaks reuse a tensor mirror (`HostTensorCache`) and skip empty
compute encodes between host steps; deferred H2D batches uploads before the
next device-reading step.

**Encoder parity notes:** compile runs `LegalizeBroadcast` before fusion
(mid-axis `[1,C,1,D]+[1,C,P,D]` needs `Expand`) and unfuses
`ElementwiseRegion` on wgpu (region kernel only supports trailing broadcast).
`Activation::Gelu` uses exact `erf` in `unary.wgsl` (not the tanh approx).

## WGSL uniform control flow (browser vs native)

WGSL requires `workgroupBarrier()` to be reached in **uniform control flow**. A
`return` gated on `global_invocation_id` breaks that, and who notices depends on
the toolchain:

| Toolchain | Behaviour |
|-----------|-----------|
| naga → Metal/Vulkan | accepts it — native tests pass |
| Dawn (every browser) | rejects the **shader module**: *"'workgroupBarrier' must only be called from uniform control flow"* |
| FXC (DirectX) | rejects it as X4026 |

The browser failure mode is the dangerous one. Dawn invalidates the module, so
every pipeline built from it is invalid and every dispatch fails — but
`run_async` still resolves, and the arena keeps whatever was in it. A browser
rank therefore reported `webgpu`, trained on a loss of **exactly 0.0000**, and
fed that into a cluster's gradient average. No panic, no error, no wrong-looking
numbers until you ask what ln(10) should be.

`matmul.wgsl` already carried the fix and a comment about FXC;
`matmul_f16w`/`matmul_f16_compute` were written later and missed it, and
`reduce` acquired it with its tree-reduction path. The pattern:

```wgsl
// Wrong: a `gid`-gated return makes every later barrier non-uniform.
let bz = gid.z;
if (bz >= params.batch) { return; }

// Right: workgroup_id is workgroup-uniform, and out-of-range invocations
// still reach the barriers with their stores guarded.
let bz = wid.z;
let in_batch = bz < params.batch;
let bz_safe = select(0u, bz, in_batch);
```

Where the index is genuinely per-invocation (`reduce`'s small path), branch
instead of returning: WGSL's uniformity analysis is flow-based and will not
reason about two arms being mutually exclusive, so one non-uniform `return`
anywhere before a barrier condemns the whole entry point — including barriers
in the arm that invocation never enters.

A comment in one shader did not stop this spreading to three, so
`tests/wgsl_uniform_control_flow.rs` checks every kernel: it strips comments,
resolves which names derive from `global_invocation_id`, and walks each
function's enclosing `if` conditions. It also self-tests — one case asserting it
catches the exact shape that broke Chrome, one asserting it does not flag the
legal workgroup-uniform forms. In `just check-wasm`.

## Host-executed steps, and planning around them

Some ops lower to a **host step**: the executor ends its compute pass, submits,
reads the step's inputs back, computes on the CPU, and writes the result into
the arena before the next pass. Each one is a sync point. A convolution is one
of them (`Step::Conv2dHost`), so a conv graph is not a pure GPU schedule even
though it compiles like one.

[`HostPlan`](src/backend/host_plan.rs) describes that side of a compiled
schedule — reachable as `WgpuExecutable::host_plan()`, or as one line from
`host_summary()`:

```text
2 of 57 steps run on the host in 1 sync point (Conv2dHost×2); 2 are
independent (2 sync points if run one at a time)
```

It answers two questions the executor previously could not:

- **What is on the host.** Callers had a single bit
  (`requires_host_execution()`), which is why the browser WebGPU refusal could
  only list the ops it *might* have been — "GGUF dequant, LSTM, collectives,
  FFT-host, …" — when the real answer was one `Conv2dHost` per convolution.
  The refusal now names the steps, so a caller knows whether to change backend
  or change model. Surfaced through `ExecutableGraph::wgpu_host_summary()` →
  `CompiledGraph::host_execution_summary()`, native as well as wasm.
- **Which of them are independent.** Consecutive host steps pay for one sync
  point each even when they have nothing to do with each other. Two can share
  one only if neither reads a byte range the other writes, so the plan groups a
  run of host steps into batches and marks a batch `parallel` when that is
  *proven*.

Region knowledge is deliberately partial: `Conv2dHost` and `BufferCopy` carry
explicit arena offsets and full shapes, so their reads and writes are exact;
every other host step reports `HostIo::Unknown` and plans as a barrier. Adding
a kind is one match arm. Until then the conservative answer is the correct one
— a wrong independence claim reorders reads and writes into the same arena
bytes and produces plausible, wrong numbers, which is the failure mode this
backend's host-cache mirror has already produced once (see *Gotchas*).

A single host step that is too big for one worker can be cut further:
[`rlx_compile::shard`](../../core/rlx-compile/src/shard.rs) splits one conv or
matmul into independent pieces, prices each axis by bytes moved, and picks a
placement. `HostPlan` says *which steps* can share a sync point; `shard` says
*how to divide one step* across workers.

**The plan is not yet acted on.** Execution still runs host steps one at a
time; `parallel: true` marks what *could* share a sync point. Doing it means
gathering a batch's reads once, computing concurrently, and writing back once —
inside the host-step arm of `run_inner`, whose host-cache flush ordering is
load-bearing.

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
  without readback. See [`docs/benchmarks/higher-order-ad.md`](../../../docs/benchmarks/higher-order-ad.md).
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

MIT OR Apache-2.0.
