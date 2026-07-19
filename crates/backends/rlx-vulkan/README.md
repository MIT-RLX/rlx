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
| `buffer.rs`  | host-visible f32 arena + the f32-uniform memory plan |
| `kernels.rs` | shared descriptor-set + pipeline layout, lazy per-kernel pipeline cache |
| `shaders.rs` | embedded SPIR-V blobs (generated from `shaders/`) |
| `backend.rs` | `VulkanExecutable`: legalize → schedule of dispatches → run |

Every tensor is an f32 slot in one arena buffer; each schedule step is one
compute pipeline + push constants + a workgroup count. A global shader-memory
barrier separates dispatches; one submit per `run`; outputs read back from the
persistently-mapped arena.

## Op coverage

~29 native compute kernels spanning the forward-inference surface:

- **Transformer**: matmul, attention (online softmax; causal / sliding-window /
  custom / bias masks; `[B,S,H,D]` & `[B,H,S,D]`), RoPE (NeoX **and** GptJ,
  full / partial), RMS/Layer norm, softmax.
- **Elementwise**: binary (×7), unary/activation (×16), compare (×6), where.
- **Reduction / shape**: reduce, cumsum, argmax/argmin, transpose, narrow,
  concat, expand, gather, reverse (one strided-copy kernel backs the shape ops).
- **Vision**: conv2d (stride/pad/dilation/groups incl. depthwise), pool2d
  (max/avg), im2col, layernorm2d, nearest-2× resize.
- **MoE / SSM / generation**: grouped matmul, selective-scan (Mamba), top-k.

Fused ops, `DotGeneral`, `Fma`, non-last-axis reduce, `GroupNorm`,
`BatchNormInference`, `LoraMatMul`, etc. are decomposed into the native set by
`legalize_or_rewrite_for_backend`.

**Host-fallback** (no native SPIR-V kernel yet, run on the CPU reference against
the host-visible mapped arena — same design as `rlx-wgpu`'s host ops, so results
are bit-for-bit the reference): the RNN/SSM families
(`Lstm`/`Gru`/`Rnn`/`Mamba2`/`GatedDeltaNet`), `ConvTranspose2d`, `Fft`,
quantized matmul (`DequantMatMul` + GGUF block decode reads the packed U8 weight
straight from the arena), `DequantGroupedMatMul`/`DequantMoEWeights`, and
`RngNormal`/`RngUniform`/`Sample`. The run loop submits GPU dispatches in
segments and flushes around each host op. These are correctness-complete and
parity-validated; promoting the hot ones to native SPIR-V is a perf follow-up.

Out of scope for a forward-inference kernel set: int8-I/O `QMatMul`/`QConv2d`,
and domain custom ops (Gaussian splat). Packed DiT reverse
(`AdaLayerNormBackward` / `GatedResidualBackward`) has dedicated SPIR-V shaders
(`ada_layer_norm_backward.comp` / `gated_residual_backward.comp`); forward
`AdaLayerNorm` / `GatedResidual` still claim-then-`unfuse_dit_modulation`. Host
`Scan*` / SPD / Eigh backward stay as arena CPU fallbacks.

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
  the host-fallback families (RNN, ConvTranspose2d, FFT, GGUF dequant, RNG).
- **lavapipe** (Mesa CPU/software Vulkan, Linux) — exact-value device tests via
  the `./docker` container.

This validation caught two real bugs (push-constant array layout in the
strided-copy kernel; a missing RMSNorm `beta` term), both fixed.

**Not yet validated on a native Vulkan driver / real GPU** — that's the pending
step on the Linux CUDA rig. Kernels are also still naive (correctness first):
shared-memory tiling, a device-local arena + staging, grid-stride dispatch, and
promoting hot host-fallback ops (esp. `DequantMatMul`) to native SPIR-V are the
perf follow-ups, best done there.

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
