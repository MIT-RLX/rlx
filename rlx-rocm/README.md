# rlx-rocm

AMD ROCm / HIP backend for RLX. Sister crate to
[`rlx-cuda`](https://crates.io/crates/rlx-cuda) — same `.cu` kernel
sources, same dispatch-tier pattern, dispatched via HIP instead of the
CUDA driver API.

## Stack

- **Matmul** — hipBLAS / hipBLASLt (with `GemmEx` for mixed precision).
- **Convolution / pooling** — MIOpen, including 4D and N-D primitives.
- **Custom kernels** — hipRTC-compiled, cached on disk.
- **hipGraph** — capture + replay.
- **Multi-stream** — dependency-aware scheduling via hipEvent fences.
- **rocTX** — span markers parallel to CUDA's NVTX path.

## Install

> **Not on crates.io for 0.1.0.** `src/kernels/sources.rs` does
> `include_str!("../../../rlx-cuda/src/kernels/*.cu")` to share kernel
> sources with rlx-cuda — the workspace-relative paths aren't in a
> published `.crate`. Distributed via the workspace git tree:

```toml
[dependencies]
rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["rocm"] }
# or directly:
rlx-rocm = { git = "https://github.com/MIT-RLX/rlx" }
```

A working ROCm install (libhipruntime / libhipblas / libMIOpen) must be
on the loader path at runtime.

A host-side HIP-CPU shim is bundled for off-GPU validation; see
`rlx-rocm/tests/hip_cpu_validate.rs`.

## What's here

* **Hand-rolled HIP runtime shim** (`src/hip.rs`) — libloading-based
  dispatch to `libamdhip64.so` / `libhiprtc.so`. Resolves the 30
  HIP API + 7 hipRTC functions we need at runtime so the crate
  compiles + tests cleanly on hosts without HIP installed.
  `HipRuntime::load()` returns `None` cleanly on missing libs.
* **`HipBuffer<T>` / `HipKernel`** wrapper types matching cudarc's
  `CudaSlice<T>` / `CudaKernel` shape: owned device memory with
  RAII `hipFree` on drop, kernel modules with `hipModuleUnload`.
* **`RocmContext`** singleton that initializes HIP + creates a
  context on device 0 + a default stream.
* **Arena** (`src/arena.rs`) — port of `rlx-cuda::arena`. f32 main
  buffer + optional u16 half-precision side-buffer. `set_param` /
  `set_param_half` upload paths fully wired against the HIP shim.
* **Kernel cache** (`src/kernels/`) — hipRTC compile + per-kernel
  `OnceLock<HipKernel>` cache + persistent `.hsaco` disk cache at
  `$RLX_ROCM_HSACO_CACHE` / `$XDG_CACHE_HOME/rlx-rocm/hsaco-rocm`.
  All 32 kernels registered (matmul_wmma intentionally excluded —
  needs MFMA/WMMA AMD intrinsics, not nvcuda::wmma).
* **`unfuse.rs`** — copied verbatim from `rlx-cuda` (IR-level, no
  backend types).
* **`Step` enum** — full 33-variant copy from `rlx-cuda`.
* **`CompileMode`** (Jit/Aot) + **`ExecMode`** (Stream/Eager/Graph/MultiStream).
* **`compile_with()` body** — full IR walk from `rlx-cuda` ported with
  `cudarc` → `HipBuffer` type swaps. All 33 Step variants emitted.
* **`run()` body** — kernel-only dispatch loop using `launch_kernel!`
  to hand-pack kernel params for `hipModuleLaunchKernel`.

## Library tier ladder (parity with rlx-cuda)

* **hipBLAS sgemm + strided-batched** — Step::MatMul / DotGeneral
  fall through to `hipblasSgemm` and `hipblasSgemmStridedBatched`
  with the row-major-as-column-major A↔B swap. TF32-equivalent
  via `HIPBLAS_XF32_XDL_MATH` math mode.
* **hipBLASLt fused epilogue** — Step::FusedMatMulBiasAct lowers to
  `hipblasLtMatmul` with bias + relu/gelu epilogue. Workspace
  pre-allocated, descriptors cached.
* **MIOpen forward conv** — Step::Conv1d (degenerate 2D), Conv2d,
  and Conv3d (via nd-tensor descriptors) lower through MIOpen's
  forward-find heuristic + workspace, with custom-kernel fallback.
* **hipBLAS GemmEx mixed-precision** — half-arena consumer; same
  cast→GemmEx pattern as rlx-cuda.
* **hipGraph capture/replay** — ExecMode::Graph wired via
  `hipStreamBeginCapture` / `hipGraphLaunch`.
* **Multi-stream + dependency-aware scheduling** —
  ExecMode::MultiStream(n) dispatches across a stream pool with
  hipEvent fences. `HipblasContext::set_stream` re-binds the
  hipBLAS handle per-step.
* **rocTX scoped ranges** — NVTX-equivalent annotations for
  rocprof / rocm-profiler. libloading-resolved.

**Native ElementwiseRegion (PLAN L2).** `Op::ElementwiseRegion` is
lowered by a hipRTC interpreted-chain kernel — kernel source
`elementwise_region.cu` shared with rlx-cuda via the `include_str!`
chain in `kernels/sources.rs`, compiled into `.hsaco` on first
dispatch. One thread per output element walks a runtime chain
encoding (4 u32s per step: `op_kind`/`op_sub`/`lhs_enc`/`rhs_enc`)
into a private `float scratch[16]`. Caps: 16 steps, 8 inputs.
op_sub numbering matches the cross-backend convention (Metal MSL /
wgpu WGSL / rlx-cuda) so the encoder produces one byte stream all
four backends interpret identically.

What's **not** here yet:

* **MFMA / WMMA matmul kernel** — equivalent of rlx-cuda's
  matmul_wmma.cu but using `__builtin_amdgcn_mfma_*` (CDNA) or
  `__builtin_amdgcn_wmma_*` (RDNA3+) intrinsics. Skip until
  real GPU access is in the picture.

## Status

Sister-crate parity with `rlx-cuda`. Build-clean, clippy-clean, 8 unit +
2 smoke tests pass on Mac. Runtime correctness on real AMD hardware is
**unverified** — first cloud-GPU run on MI300X / RX 7900 XTX is the
validation gate. All library tiers fall through gracefully to the
kernel-only path when their `.so` isn't loadable.

## Why scaffold-first?

Same reason `rlx-cuda` started out as a "minimum viable" crate without
cuBLAS — gets the workspace integration, IR plumbing, test harness, and
conventions in place so that real dispatch work lands as drop-in
additions instead of a big ball of intermingled "new crate + new
bindings + new dispatch" all at once.

## Dev: HIP-CPU validation path

`--features hip-cpu-validate` runs the same `.cu` kernel sources on
CPU threads via [HIP-CPU](https://github.com/ROCm-Developer-Tools/HIP-CPU)
— literally the AMD-shipped HIP-on-CPU runtime. Useful for catching
kernel-logic and dispatch bugs on Mac (or any host without an AMD
driver) before paying for cloud-GPU time.

**Off by default. Never enabled in production builds.**

### Code-sharing strategy

The `.cu` kernel sources, the C++ wrapper layer, and the Rust FFI
bindings are **all shared with rlx-cuda** rather than duplicated:

| Layer | Source of truth | rlx-rocm reference |
|---|---|---|
| `.cu` kernels | `rlx-cuda/src/kernels/*.cu` | `kernels::sources` via `include_str!` |
| C++ wrapper layer (`launch_*` fns) | `rlx-cuda/cpp/cpu_dispatch.cpp` | `cpp/cpu_dispatch.cpp` (one-line `#include`) |
| Rust FFI bindings (`run_*` fns) | `rlx-cuda/src/cpu_dispatch.rs` | `src/cpu_dispatch.rs` (one-line `#[path]`) |
| HIP-CPU headers | `rlx-cuda/vendor/HIP-CPU` (submodule) | reused — single submodule, both crates |
| Comprehensive kernel tests | `rlx-cuda/tests/hip_cpu_validate.rs` (38) | covered upstream |

So any kernel improvement, FFI signature change, or wrapper fix in
rlx-cuda flows through to rlx-rocm automatically.

### Workflow

```sh
# One-time: pull HIP-CPU as a submodule (shared with rlx-cuda).
git submodule add https://github.com/ROCm-Developer-Tools/HIP-CPU.git \
    rlx-cuda/vendor/HIP-CPU
git submodule update --init

# Compile + smoke-test the CPU-execution path from rlx-rocm.
cargo test -p rlx-rocm --features hip-cpu-validate

# In Docker (any architecture, no GPU needed):
docker run --rm -v $PWD:/work -w /work rust:1.76 \
    bash -c "apt-get update && apt-get install -y g++ && \
             cargo test -p rlx-rocm --features hip-cpu-validate"
```

## Build / test

```sh
cargo build -p rlx-rocm --release          # compile-check on any host
cargo test  -p rlx-rocm --release          # 2 smoke tests
```

## License

GPL-3.0-only.
