# rlx-vulkan

Native **Vulkan compute** backend for [RLX](https://github.com/MIT-RLX/rlx) —
`Device::Vulkan`.

Unlike `rlx-wgpu` (which can reach Vulkan through the wgpu portability layer),
this crate talks to Vulkan directly via [`ash`](https://crates.io/crates/ash):
its own instance / physical-device / logical-device / compute queue, a single
f32 arena `VkBuffer`, descriptor sets, and compute pipelines built from
hand-written GLSL kernels. It is the dedicated Vulkan path where a real Vulkan
ICD exists: **desktop Linux/Windows NVIDIA/AMD/Intel and Android Adreno/Mali.**

> **Apple Silicon is not a Vulkan target.** macOS has no Vulkan loader by
> default — the GPU path here is Metal (`rlx-metal`, or wgpu-on-Metal for
> `Device::Gpu`). `Device::Vulkan` only comes alive on a Mac if you install the
> loader + MoltenVK (`brew install vulkan-loader molten-vk`) and point the
> loader at them (`VK_ICD_FILENAMES` + `DYLD_FALLBACK_LIBRARY_PATH`, since
> Homebrew installs to `/opt/homebrew/lib`, off the default dlopen path).
> MoltenVK is a Vulkan→Metal *translation layer*, so even then SPIR-V runs on
> Metal — useful for kernel-correctness checks, **not** a native Vulkan driver.
> Without that setup `is_available(Device::Vulkan)` is `false` and the backend
> never dispatches (by design).

## How it builds without a Vulkan SDK

- **Kernels**: GLSL `shaders/*.comp` are compiled to **SPIR-V at build time**
  by `build.rs` using [`naga`](https://crates.io/crates/naga) (pure Rust — no
  glslang / shaderc / Vulkan SDK on the build host). The `.spv` words are
  embedded with `include_bytes!`; there is **no runtime shader compilation**.
- **Loader**: `ash` is used in dynamic-loading mode (`Entry::load()`), so the
  crate compiles and links on hosts with no Vulkan driver. With no loader (e.g.
  macOS without MoltenVK), `is_available()` returns `false` and the runtime
  registry simply doesn't dispatch to `Device::Vulkan` — same graceful pattern
  as `rlx-cuda` / `rlx-rocm`.

## Architecture

| file | role |
|------|------|
| `device.rs`  | instance / device / queue singleton, memory-type query, one-shot submit |
| `buffer.rs`  | host-visible f32 arena (+ multi-buffer sharding when over `maxStorageBufferRange`) |
| `host_stage.rs` | [`DeviceArena`](../rlx-gpu-host) adapter for shard-safe Scan / HostOp / indexing |
| `kernels.rs` | shared descriptor-set + pipeline layout, lazy per-kernel pipeline cache |
| `shaders.rs` | embedded SPIR-V blobs (generated from `shaders/`) |
| `backend.rs` | `VulkanExecutable`: legalize → schedule of dispatches → run |

Every tensor is an f32 slot in the activation arena; each schedule step is one
compute pipeline + push constants + a workgroup count. A global shader-memory
barrier separates dispatches; one submit per `run`; outputs read back from the
persistently-mapped arena.

### Sharded arenas

When the planned act arena exceeds `maxStorageBufferRange` (~4 GiB on many NVIDIA
cards), activations are striped across multiple `VkBuffer`s. Host paths that
used a single `mapped_ptr()` (shard 0 only) SIGSEGV'd on large graphs (e.g.
KittenTTS wave ≥64 k before the fix). Indexing / Scan / HostOp now go through
`host_stage::VulkanArena` + `rlx_gpu_host::run_*` (per-region D2H when sharded).

`snap_plan_to_shards` mirrors wgpu: scratch may live in the per-stripe stage
reserve — it does **not** open an empty extra shard for tail bytes. Override
stage size with `RLX_VULKAN_SHARD_STAGE_MIB` (bytes = MiB×2²⁰, capped at the
576 MiB compile-time default). KittenTTS sets `64` so long-wave plans stay
unsharded longer.

**Note:** sharded KittenTTS still produces near-silent audio; keep wave compile
width under the unsharded window (see Kitten `device_policy`). The host-staging
fix only removes the crash.

## Op coverage

Claims the full **153/`OpKind`** surface ([`docs/op-coverage.md`](../../docs/op-coverage.md)).
Native SPIR-V covers the transformer / vision / training hot path:

- **Transformer**: matmul, attention (online softmax; causal / sliding-window /
  custom / bias masks; `[B,S,H,D]` & `[B,H,S,D]`), RoPE (NeoX **and** GptJ,
  full / partial), RMS/Layer/Group norm (+ bwd), softmax, SoftmaxCrossEntropy*.
- **Elementwise**: binary (×7), unary/activation (×16), compare (×6), where,
  Fma, ActivationBackward.
- **Reduction / shape**: reduce, cumsum (+ bwd), argmax/argmin, transpose,
  narrow, concat, expand, gather (+ bwd), reverse (one strided-copy kernel
  backs the shape ops).
- **Vision**: conv2d/3d (+ bwd), ConvTranspose2d/3d, pool2d (+ MaxPool2d bwd),
  im2col, layernorm2d, nearest-2× resize, FusedConvBiasAct.
- **MoE / SSM / generation**: grouped matmul, selective-scan (Mamba),
  Gru/Rnn/Mamba2/Lstm (size-capped; oversized → host), top-k,
  FftButterflyStage.
- **QAT / I8**: FakeQuantize Fixed/PerBatch/LSQ (+ STE bwd), packed-I8
  Quantize/Dequantize/QMatMul/QConv2d.
- **C64**: ComplexNormSq / Backward / Conjugate (interleaved).

Fused residual norms / SwiGLU, `FusedMatMulBiasAct`, fusion regions,
`LoraMatMul` / `FusedTransformerLayer` / `If` / `While` / `DotGeneral` /
`GatedDeltaNet` deepen via native SPIR-V or `rlx-unfuse` before schedule.
`DotGeneral`, non-last-axis reduce, etc. also decompose via
`legalize_or_rewrite_for_backend`.

**Host-fallback** (no native SPIR-V yet, or exceeds a native size cap —
run on the CPU reference against the host-visible mapped arena): oversized
RNN/SSM, `Fft`, quantized matmul (`DequantMatMul` + GGUF block decode),
`DequantGroupedMatMul`/`DequantMoEWeights`, `RngNormal`/`RngUniform`/`Sample`,
DenseSolve (HostOpDesc → LAPACK), `CustomFn`, splat prepare/rasterize.
The run loop submits GPU dispatches in segments and flushes around each host op.

Packed DiT reverse (`AdaLayerNormBackward` / `GatedResidualBackward`) has
dedicated SPIR-V shaders; forward `AdaLayerNorm` / `GatedResidual` still
claim-then-`unfuse_dit_modulation`. Host `Scan*` / SPD / Eigh backward stay
as arena CPU fallbacks.

## Status

**Kernel-correctness validated** on two *non-native* Vulkan implementations —
enough to prove the SPIR-V and push-constant layout are right, but neither is a
real GPU driver:
- **MoltenVK** (Vulkan→Metal translation, manual setup — see the Apple-Silicon
  note above) — a Vulkan↔CPU parity suite
  (`crates/rlx-runtime/tests/vulkan_parity.rs`, ~99 op-variant checks across 50
  tests): every Binary/Activation/Compare/Reduce variant, attention across mask
  kinds + both head layouts, RoPE (NeoX/GptJ × full/partial), matmul, norms,
  softmax, vision (conv/pool/im2col/resize), MoE grouped-matmul, Mamba SSM, and
  the host-fallback families (oversized RNN/SSM, FFT, GGUF dequant, RNG).
- **lavapipe** (Mesa CPU/software Vulkan, Linux) — exact-value device tests via
  the `./docker` container.

This validation caught two real bugs (push-constant array layout in the
strided-copy kernel; a missing RMSNorm `beta` term), both fixed.

**Not yet validated on a native Vulkan driver / real GPU** — that's the pending
step on the Linux CUDA rig. Kernels are also still naive (correctness first):
shared-memory tiling, a device-local arena + staging, grid-stride dispatch, and
promoting remaining hot host ops (esp. `DequantMatMul` / GGUF) to native SPIR-V
are the perf follow-ups, best done there.

Run the parity suite locally with a Vulkan driver present:

```bash
# macOS (MoltenVK via Homebrew: vulkan-loader + molten-vk)
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
cargo test -p rlx-runtime --features vulkan --test vulkan_parity -- --test-threads=1
```

## Usage

```rust
use rlx::{Session, Device};

let mut session = Session::new(Device::Vulkan);
// … compile + run a graph, same API as every other backend …
```

Enable via the umbrella crate's `vulkan` feature: `cargo build -p rlx --features vulkan`.

## License

GPL-3.0-only.
