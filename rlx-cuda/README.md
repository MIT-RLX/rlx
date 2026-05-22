# rlx-cuda

NVIDIA CUDA backend for RLX. cuBLAS / cuBLASLt for matmul, NVRTC-
compiled kernels for everything else, via the pure-Rust
[`cudarc`](https://crates.io/crates/cudarc) crate — no nvcc at workspace
build time, no CUDA SDK install on developer machines. CUDA C++ kernel
sources live as `&'static str` and are JIT-compiled to PTX via NVRTC on
first dispatch (same pattern as rlx-wgpu's WGSL kernels).

## Stack

- **Matmul** — cuBLAS (FP32), cuBLASLt (mixed precision via `GemmEx`).
- **Convolution / pooling** — cuDNN.
- **Tensor cores** — WMMA path for FP16 / BF16 GEMM on Volta+.
- **Custom kernels** — NVRTC-compiled `.cu` sources, cached on disk
  by graph fingerprint.
- **CUDA Graphs** — capture + replay for inference-shaped workloads.
- **Multi-stream** — async copy + compute overlap.
- **NVTX** — span markers wired through Perfetto export.

## Install

A working CUDA toolkit (libcudart / libcublas / libcudnn) must be on
the loader path. The crate is feature-gated in `rlx-runtime`:

```toml
[dependencies]
rlx = { version = "0.1", features = ["cuda"] }
```

## Mac-side iteration

`cudarc`'s `dynamic-loading` feature loads `libcuda` via `dlopen` at
first FFI call. On Mac there's no libcuda, so:

1. **`cargo build -p rlx-cuda --release`** — compiles cleanly. The
   crate links against cudarc's stub bindings; libcuda is only
   resolved at runtime.

2. **`cargo test -p rlx-cuda --release`** — runs the basic tests. Each
   test checks `is_available()` first; on Mac that returns false (the
   libcuda load fails inside cudarc and we catch the panic), so
   tests no-op cleanly.

3. **`./rlx-cuda/check-compile.sh`** — builds the crate inside an
   `nvidia/cuda:12.6.0-devel-ubuntu22.04` Docker image. Validates that
   our CUDA C++ sources compile against a real NVRTC + that cudarc
   links against the real libcuda. Apple Silicon runs the amd64 image
   under qemu emulation; takes a few minutes on first build, much
   faster on cache hits.

There's no path to actually *run* CUDA kernels on Mac — Apple Silicon
has no NVIDIA GPU, and Docker Desktop's VM has no GPU passthrough even
when running on a hypothetical Intel Mac with NVIDIA hardware. For
benchmarks: use a cloud GPU (vast.ai, Lambda Labs, RunPod) or a
self-hosted Linux box.

## What's here

- `device.rs` — `CudaContext` singleton with panic-catching init so a
  missing libcuda returns `None` instead of crashing.
- `arena.rs` — single device buffer + per-node offsets, mirroring the
  rlx-wgpu f32-uniform arena. Reshape and Cast alias the input slot.
- `kernels/*.cu` — CUDA C++ sources (binary, unary, copy, matmul,
  attention, conv, etc.). Compiled via NVRTC at first dispatch and
  cached behind `OnceLock`s.
- `kernels/mod.rs` — NVRTC compile + module/function loader.
- `backend.rs` — `CudaExecutable`. Full IR coverage via the dispatch
  tier ladder below.

## Matmul dispatch tier decision tree

`Step::Matmul` walks down a tier ladder; each tier checks its
preconditions and either dispatches or falls through. With
`RLX_CUDA_LOG_FALLBACK=1` you'll see exactly which tier ran.

```
                       Step::Matmul(m, k, n, …)
                                │
            ┌───────────────────┴────────────────────┐
            │ Is weight (B) in half-arena?           │
            │ (set_param_half was called for B)      │
            └───────────────────┬────────────────────┘
                                │
                ┌──── yes ──────┴───── no ──────┐
                ▼                               ▼
   ┌─────────────────────────┐    ┌────────────────────────────┐
   │ Tier 0: mixed-precision │    │ Tier 1: cublasLt fused     │
   │ cast f32 act → f16/bf16 │    │ matmul + bias + relu/gelu  │
   │ scratch; cublasGemmEx;  │    │ in one launch              │
   │ epilogue kernel for     │    │ — only when act ∈ {Relu,   │
   │ bias/act (any kind)     │    │   Gelu, none}              │
   │                         │    │                            │
   │ ✓ 2× weight memory      │    │ ✓ Saves epilogue launch    │
   │ ✓ Tensor Core compute   │    │ ✓ Bias broadcast inline    │
   └─────────────────────────┘    └────────────┬───────────────┘
                                                │ act not relu/gelu
                                                ▼
                                  ┌────────────────────────────┐
                                  │ Tier 2: cublasSgemm        │
                                  │ + matmul_epilogue.cu       │
                                  │   if has_bias || act ≠ id  │
                                  │                            │
                                  │ ✓ TF32 Tensor Core (auto)  │
                                  │ ✓ Handles all 12 acts      │
                                  └────────────┬───────────────┘
                                                │ blas unavailable
                                                ▼
                                  ┌────────────────────────────┐
                                  │ Tier 3: WMMA Tensor Core   │
                                  │ kernel (matmul_wmma.cu)    │
                                  │ — only if RLX_CUDA_WMMA=1  │
                                  │ + SM 70+ NVRTC compile OK  │
                                  └────────────┬───────────────┘
                                                │ env not set / SM<70
                                                ▼
                                  ┌────────────────────────────┐
                                  │ Tier 4: scalar SGEMM       │
                                  │ 64×64 block + 4×4 reg tile │
                                  │ float4 vec loads when      │
                                  │ K%4==0 && N%4==0           │
                                  └────────────────────────────┘
```

### Concrete examples

| Shape | Bias | Act | Half-arena? | Tier picked |
|-------|-----:|----:|------------:|-------------|
| 1024×4096×4096 | yes | gelu | yes (f16) | **0** mixed-precision GemmEx + epilogue |
| 1024×4096×4096 | yes | gelu | no | **1** cublasLt fused |
| 1024×4096×4096 | yes | silu | no | **2** sgemm + epilogue (silu not in cublasLt) |
| 1×3×2 (test) | no | — | no | **2** sgemm (cuBLAS handles tiny shapes fine) |
| any | any | any | no, no driver | **4** scalar fallback |

## Conv dispatch

`Step::Conv1d / Conv2d / Conv3d` are simpler: cuDNN if libcudnn
loaded → custom direct-conv otherwise. Conv1d uses the conv2d helper
with `H=kh=sh=1, ph=0, dh=1` (degenerate 2-D); Conv3d uses cuDNN's
nd-descriptor APIs.

## Compile + execution modes

`CudaExecutable::compile_with(graph, compile_mode, exec_mode)` selects:

- **`CompileMode::Jit`** (default) — kernels NVRTC-compile on first
  dispatch, then live in the cuModule cache for the rest of the
  process. First `run()` pays the JIT cost (~10-100ms × 32 kernels).
- **`CompileMode::Aot`** — pre-compile every kernel (32 of them) at
  executable construction. Moves JIT cost out of the critical path
  at the cost of ~1-3s upfront. Good for inference servers that
  build the executable once and run forever.
- **Persistent PTX disk cache.** All NVRTC compiles cache their PTX
  to `$RLX_CUDA_PTX_CACHE` (or `$XDG_CACHE_HOME/rlx-cuda` /
  `~/.cache/rlx-cuda`), namespaced by the cuda toolkit version. Cache
  key is `<entry>-<fnv1a64(source)>.ptx`; FNV-1a is just for filename
  uniqueness — a stale cache hit is impossible because mismatched
  source recompiles. Atomic via tmp + rename. Across-process
  cold-start drops from ~1-3s to ~50ms after first run.

- **TF32 fast math in cublasLt.** Compute type is
  `CUBLAS_COMPUTE_32F_FAST_TF32` for f32 matmul — uses Tensor Cores
  on Ampere+ for ~2× speedup with a 10-bit-mantissa intermediate.
  Matches what `cublasSgemm` does by default (since CUDA 11) and is
  well within transformer-inference precision tolerance.

- **NVTX profiling ranges.** Each `Step` dispatch is wrapped in an
  `nvtx::scoped_range` named `rlx::<StepKind>`. Negligible overhead
  when no profiler is attached; nsight-systems / nvprof traces show
  step boundaries cleanly so devs can see where time goes.

- **Backend-level element-wise fusion.** `fuse_elementwise_chains`
  runs after the schedule is built and merges adjacent
  `Binary → Unary` pairs into a single `FusedBinaryUnary` step when
  the intermediate offset has exactly one consumer in the schedule.

- **Half-precision params side-buffer + mixed-precision matmul.**
  `Arena.half_buffer` is an optional `CudaSlice<u16>` (raw bits —
  `f16` or `bf16` per-node tag via `HalfDtype`) for storing weights.
  Activations stay f32 in the main `buffer`. Use
  `CudaExecutable::set_param_half(name, dtype, &[u16])` to upload
  weights in half-precision instead of `set_param`. The matmul
  dispatch detects half-stored weights via `Arena.half_by_f32_off`
  and:
    1. Casts the f32 activations to f16/bf16 into a scratch buffer
       (`cast_f32_to_half.cu` kernel).
    2. Calls `cublasGemmEx` with both inputs f16/bf16, compute type
       `CUBLAS_COMPUTE_32F_FAST_16F` / `CUBLAS_COMPUTE_32F_FAST_16BF`,
       and a f32 accumulator that writes back to the main arena.
    3. Optional bias / activation epilogue runs as a separate
       `matmul_epilogue.cu` pass after.

- **`ExecMode::Stream`** (default) — every `run()` dispatches each
  step on the default stream.
- **`ExecMode::Graph`** — first `run()` captures the schedule into
  a CUDA Graph; subsequent runs replay the captured graph. Saves
  per-launch dispatch overhead (~10-20% on small-batch decode).
- **`ExecMode::Eager`** — `CudaExecutable::eager(graph, inputs)`
  one-shot helper that compiles + runs + drops in one call.
- **`ExecMode::MultiStream(n)`** — allocate a pool of `n` streams
  and assign each `Step` based on producer-consumer relations on
  arena offsets (computed by `step_offsets`). Independent ops run
  in parallel; cross-stream sync is via CUDA events at fork/join
  points. Incompatible with `ExecMode::Graph`.

## Build / test

```sh
cargo build -p rlx-cuda --release          # compile-check on any host
cargo test  -p rlx-cuda --release          # 3 basic tests; no-op on Mac
./rlx-cuda/check-compile.sh                # docker compile validation
```

## Status

Functional; less battle-tested than the Apple Silicon path. The kernel
sources are shared with `rlx-rocm` (sister crate) so coverage moves in
lock-step.

## Dev: HIP-CPU validation path

`--features hip-cpu-validate` is an **opt-in dev feature** that lets us
run the same `.cu` kernel sources on CPU threads via [HIP-CPU](https://github.com/ROCm-Developer-Tools/HIP-CPU).
Useful for catching kernel-logic and IR-lowering bugs on Mac (or any
host without an NVIDIA driver) before paying for cloud-GPU time.

**Off by default. Never enabled in production builds.**

### Workflow

```sh
# One-time: pull HIP-CPU as a submodule.
git submodule add https://github.com/ROCm-Developer-Tools/HIP-CPU.git \
    rlx-cuda/vendor/HIP-CPU
git submodule update --init

# Compile + test the CPU-execution path.
cargo test -p rlx-cuda --features hip-cpu-validate

# In Docker (any architecture, no GPU needed):
docker run --rm -v $PWD:/work -w /work rust:1.76 \
    bash -c "apt-get update && apt-get install -y g++ && \
             cargo test -p rlx-cuda --features hip-cpu-validate"
```

### Architecture

```
                ┌─── shared sources: src/kernels/*.cu ───┐
                │                                        │
       cudarc + libcuda                          HIP-CPU + cc::Build
                │                                        │
       NVIDIA GPU dispatch                       CPU thread dispatch
       (production: rlx-cuda)                    (dev: hip-cpu-validate)
```

`build.rs` compiles `cpp/cpu_dispatch.cpp` against HIP-CPU headers when
the feature is on. The TU `#include`s each `.cu` file directly and
exposes one `extern "C" launch_<kernel>` wrapper per kernel using
`hipLaunchKernelGGL`. Rust calls those via FFI in `src/cpu_dispatch.rs`.

### Coverage

All 32 kernel entry points are wired end-to-end (= 30 `.cu` files +
matmul/scatter_add contributing extras). Each one has:

1. `#include "<kernel>.cu"` in `cpp/cpu_dispatch.cpp` plus a
   `extern "C" launch_<kernel>(...)` wrapper that calls
   `hipLaunchKernelGGL` with the kernel's argument tuple.
2. The matching `extern "C"` declaration + safe Rust wrapper in
   `src/cpu_dispatch.rs` (one `run_<kernel>(...)` fn per family).
3. A unit test under `tests/hip_cpu_validate.rs` that exercises the
   FFI dispatch on a tiny representative shape.

### Caveats

- HIP-CPU is **CPU emulation** of CUDA semantics, not a full
  reimplementation. `__shared__` works, `__syncthreads()` is a barrier,
  atomics use `std::atomic`. We avoid `__shfl_*` warp-level primitives
  because HIP-CPU's wavefront size differs from CUDA's 32-thread warp.
- Translation differences between NVCC and clang (sign extension,
  FMA fusion ordering, intrinsic lowering) won't surface here. Real
  CUDA validation requires a real CUDA box.
- HIP-CPU's perf is wildly slower than a real GPU (~1000×). Don't
  bench against it; only use it for correctness.

## Gotchas

- **`dynamic-loading` panics on missing libcuda.** Even calling
  `cudarc::driver::CudaContext::new(0)` panics rather than returning
  an `Err` when libcuda can't be `dlopen`'d. We wrap the first call in
  `panic::catch_unwind` so `is_available()` returns false cleanly.

- **FlashAttention-1 KV blocking.** `attention.cu` is a one-block-per
  -(batch, head, q-tile) kernel. BR=16 query rows × BC=32 KV-tile,
  128 threads/block. K and V tiles are loaded into shared memory once
  per tile and reused for both QK and PV passes. Online softmax across
  KV tiles maintains row_max/row_sum and rescales the running V
  accumulator on every tile. Static head_dim cap of 128 (covers Llama
  70B); larger head_dim early-returns.

- **cuDNN conv dispatch.** `Conv1d`/`Conv2d`/`Conv3d` all route
  through cuDNN's v7 heuristic-picked forward conv when libcudnn is
  available. Workspace is a 32 MiB scratch buffer per executable.

- **Grouped matmul (MoE) sorted-batch path.** `Step::GroupedMatmul`
  downloads the expert-id buffer to host, detects runs of identical
  consecutive ids, and issues one `cublasSgemm` per run when the run
  count is ≤ m/4. Falls back to the per-token expert-lookup kernel for
  random idx, where the cuBLAS launch overhead would dominate.

- **Kernels JIT-compile on first dispatch.** First `run()` per kernel
  pays an NVRTC compile (~10-100ms each); subsequent calls reuse the
  cached `cuModule`. Pre-warming all kernels at compile time would
  amortize this, but it'd hit the cold path during compile rather
  than first-run.

- **Native ElementwiseRegion (PLAN L2).** `Op::ElementwiseRegion` is
  lowered by an NVRTC interpreted-chain kernel
  (`kernels/elementwise_region.cu`). One thread per output element
  walks a runtime chain encoding (4 u32s per step:
  `op_kind`/`op_sub`/`lhs_enc`/`rhs_enc`) into a private
  `float scratch[16]` register array and writes the last step's
  result to `arena[dst_off + i]`. Operand bit 31 picks the source
  (0=Input → `arena[input_offs[idx]+i]`, 1=Step → `scratch[idx]`).
  Caps: 16 chain steps, 8 inputs — same as the Metal MSL / wgpu WGSL
  kernels so the encoder in `rlx-opt` produces one byte stream all
  three backends interpret identically.

## License

GPL-3.0-only.
