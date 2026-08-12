# rlx-rocm

AMD ROCm / HIP backend for RLX. Sister crate to
[`rlx-cuda`](https://crates.io/crates/rlx-cuda) — same `.cu` kernel
sources, same dispatch-tier pattern, dispatched via HIP instead of the
CUDA driver API.

## Stack

- **Matmul** — hipBLAS / hipBLASLt (with `GemmEx` for mixed precision).
- **Eigensolver** — hipSOLVER `SsyevjBatched` for `Op::Eigh` / `Op::EighBatch`
  (`n ≤ 32`); larger `n` uses the CPU host path.
- **DenseSolve** — hipSOLVER `Sgetrf`/`Sgetrs` + hipBLAS batched LU for
  `DenseSolve` / `BatchedDenseSolve` (F32); other dtypes HostOp → LAPACK.
- **Convolution / pooling** — MIOpen, including 4D and N-D primitives.
- **Custom kernels** — hipRTC-compiled, cached on disk.
- **hipGraph** — capture + replay.
- **Multi-stream** — dependency-aware scheduling via hipEvent fences.
- **rocTX** — span markers parallel to CUDA's NVTX path.

## Install

Kernel sources ship in [`rlx-gpu-kernels`](../rlx-gpu-kernels) on crates.io.
`rlx-rocm` depends on it (with the `rocm` feature for `matmul_mfma.cu`).

```toml
[dependencies]
rlx = { version = "0.2", features = ["rocm"] }
# or directly:
rlx-rocm = "0.2"
rlx-gpu-kernels = { version = "0.2", features = ["rocm"] }
```

A working ROCm install (libhipruntime / libhipblas / libMIOpen / libhipsolver)
must be on the loader path at runtime. hipSOLVER is optional for most ops but
required for the native `Eigh` / `EighBatch` path (`n ≤ 32`).

A host-side HIP-CPU shim is bundled for off-GPU validation; see
`rlx-rocm/tests/hip_cpu_validate.rs`.

## Cost-model calibration

When HIP is available, `rlx_rocm::calibrate::Calibration::load_or_measure()` runs a
1024³ matmul micro-benchmark and writes `~/.cache/rlx/rocm-calib-<device>.json`.
Feeds `RocmCostModel` in `rlx-runtime` for backend ranking.

## What's here

* **Hand-rolled HIP runtime shim** (`src/hip.rs`) — libloading-based
  dispatch to `libamdhip64.so` / `libhiprtc.so`. Resolves the 30
  HIP API + 7 hipRTC functions we need at runtime so the crate
  compiles + tests cleanly on hosts without HIP installed.
  `HipRuntime::load()` returns `None` cleanly on missing libs.
* **ROCm-SMI telemetry/control shim** (`src/rsmi.rs`) — same libloading
  pattern over `librocm_smi64.so`. Read-only `sample(index)` (edge /
  junction / memory temp, power, cap, fan, util) plus root-only
  `set_power_cap` / `set_fan_percent` / `reset_fan` and `power_cap_range`.
  Surfaced cross-backend via `rlx_runtime::device_thermal`; drives the
  `rlx-gpu` CLI. Clock control is not wired (use the power cap).
* **FFT** — same `rlx-gpu-kernels/fft.cu` plan as CUDA; `fft_host.rs`
  for partial sync on non-native shapes/dtypes.
* **`HipBuffer<T>` / `HipKernel`** wrapper types matching cudarc's
  `CudaSlice<T>` / `CudaKernel` shape: owned device memory with
  RAII `hipFree` on drop, kernel modules with `hipModuleUnload`.
* **`RocmContext`** singleton that initializes HIP + creates a
  context on device 0 + a default stream.
* **Arena** (`src/arena.rs`) — port of `rlx-cuda::arena`. f32 main
  buffer + optional u16 half-precision side-buffer. `set_param` /
  `set_param_half` upload paths fully wired against the HIP shim.
* **`host_staging.rs`** — pageable or pinned host slots for input upload
  and output download (`RLX_ROCM_PINNED_IO`, always on in graph exec mode).
* **Attention** — BSHD `[B,S,H,D]` and BHSD both use tiled flash
  (`attention_kernel`) when `head_dim ≤ 128`; `RLX_ROCM_FORCE_ATTENTION_ROW=1`
  forces `attention_row_kernel`. Packed QKV: `RLX_ROCM_NO_PACKED_BSHD_ATTN`.
  `run_slots` + `arena_ptr` mirror `rlx-cuda`.
* **Kernel cache** (`src/kernels/`) — hipRTC compile + per-kernel
  `OnceLock<HipKernel>` cache + persistent `.hsaco` disk cache at
  `$RLX_ROCM_HSACO_CACHE` / `$XDG_CACHE_HOME/rlx-rocm/hsaco-rocm`.
  All 48 kernels registered (matmul_mfma intentionally excluded —
  needs MFMA/WMMA AMD intrinsics, not nvcuda::wmma).
* **`unfuse.rs`** — copied verbatim from `rlx-cuda` (IR-level, no
  backend types).
* **`Step` enum** — full variant set copied from `rlx-cuda`.
* **`CompileMode`** (Jit/Aot) + **`ExecMode`** (Stream/Eager/Graph/MultiStream).
* **`compile_with()` / `run()`** — IR walk + kernel dispatch via `launch_kernel!`.

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
* **hipSOLVER SsyevjBatched** — `Op::Eigh` / `Op::EighBatch` with
  `n ≤ 32` → `Step::EighNative` (on-device; see `eigh_native.rs`).
  Missing `libhipsolver` or larger `n` keeps `Step::SpdHost`.
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

Sister-crate parity with `rlx-cuda` for the supported op set. Build-clean
on Mac via libloading; runtime correctness on AMD hardware should be
validated on MI300X / RX 7900 XTX class GPUs. Library tiers fall through
to custom kernels when their `.so` isn't loadable.

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
| `.cu` kernels | `rlx-gpu-kernels/kernels/*.cu` | `rlx-gpu-kernels` (`rocm` feature for MFMA) |
| C++ wrapper layer (`launch_*` fns) | `rlx-cuda/cpp/cpu_dispatch.cpp` | `cpp/cpu_dispatch.cpp` (one-line `#include`) |
| Rust FFI bindings (`run_*` fns) | `rlx-cuda/src/cpu_dispatch.rs` | `src/cpu_dispatch.rs` (one-line `#[path]`) |
| HIP-CPU headers | `rlx-cuda/docker/vendor/HIP-CPU` (Docker clone) | reused — not in git |
| Comprehensive kernel tests | `rlx-cuda/tests/hip_cpu_validate.rs` (38) | covered upstream |

So any kernel improvement, FFI signature change, or wrapper fix in
rlx-cuda flows through to rlx-rocm automatically.

### Workflow

```sh
just test-hip-cpu-validate
```

## Build / test

```sh
cargo build -p rlx-rocm --release          # compile-check on any host
cargo test  -p rlx-rocm --release          # basic + unit tests
```

## License

MIT OR Apache-2.0.
