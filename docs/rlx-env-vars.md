# RLX environment variables (`RLX_*`)

Generated from [`env_registry`](../crates/core/rlx-ir/src/env_registry.rs)
(source of truth). Prefer `CompileOptions` when a setting changes compile
semantics. Curated Public list: `just env-catalog`.

## Legend

| Stability | Meaning |
|-----------|---------|
| Public | Stable / documented (`just env-catalog`) |
| Bisect | Escape hatch / parity |
| Internal | Bench / tooling |
| Deprecated | Use replace_with |

**Registered names:** 335  
**Unregistered mentions (migration leftovers):** 88

## Groups

- [compile](#compile) — 9
- [coreml](#coreml) — 8
- [cpu](#cpu) — 6
- [cuda](#cuda) — 38
- [debug](#debug) — 12
- [device](#device) — 4
- [fft](#fft) — 10
- [gpu](#gpu) — 4
- [metal](#metal) — 88
- [misc](#misc) — 65
- [mlx](#mlx) — 17
- [oneapi](#oneapi) — 4
- [onnx](#onnx) — 4
- [profile](#profile) — 1
- [qnn](#qnn) — 1
- [rocm](#rocm) — 6
- [tpu](#tpu) — 1
- [vulkan](#vulkan) — 14
- [wgpu](#wgpu) — 43

## compile

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_CACHE_PARAM_INVARIANT` | Public | Bool | Compile | Hoist param-invariant subgraph into prepare-once graph |
| `RLX_DISABLE_CONV_BIAS_ACT_FUSION` | Public | Bool | Compile | Skip Conv+Bias+Act fusion |
| `RLX_FUSE_ATTN_THRESHOLD` | Bisect | U64 | Compile | See call sites for `RLX_FUSE_ATTN_THRESHOLD` |
| `RLX_FUSE_BATCH_PREPROCESS` | Bisect | Bool | Compile | See call sites for `RLX_FUSE_BATCH_PREPROCESS` |
| `RLX_FUSE_REGION_PROLOGUE` | Bisect | Bool | Compile | See call sites for `RLX_FUSE_REGION_PROLOGUE` |
| `RLX_FUSION_REPORT` | Public | Bool | Compile | Print fusion pass before/after report |
| `RLX_KERNEL_DISPATCH` | Public | Enum | Compile | common|native|force_common|force_native kernel dispatch policy |
| `RLX_LINT_NUMERICS` | Public | Bool | Compile | Static provable NaN/Inf lint during compile |
| `RLX_NO_IO_PEAKS_OUTPUT` | Public | Bool | Compile | Disable compile-time IO-gated peaks-only fusion |

## coreml

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_COREML_F16` | Bisect | Bool | Backend(coreml) | See call sites for `RLX_COREML_F16` |
| `RLX_COREML_FLEXIBLE_INPUTS` | Bisect | Bool | Backend(coreml) | See call sites for `RLX_COREML_FLEXIBLE_INPUTS` |
| `RLX_COREML_HOST_DEQUANT` | Public | Bool | Backend(coreml) | Force CoreML hybrid host dequant segments |
| `RLX_COREML_NATIVE_FLEX` | Bisect | Bool | Backend(coreml) | See call sites for `RLX_COREML_NATIVE_FLEX` |
| `RLX_COREML_NATIVE_SCAN` | Bisect | Bool | Backend(coreml) | See call sites for `RLX_COREML_NATIVE_SCAN` |
| `RLX_COREML_Q1_MODE` | Bisect | String | Backend(coreml) | See call sites for `RLX_COREML_Q1_MODE` |
| `RLX_COREML_SEG_REPORT` | Bisect | Bool | Backend(coreml) | See call sites for `RLX_COREML_SEG_REPORT` |
| `RLX_COREML_UNITS` | Bisect | String | Backend(coreml) | See call sites for `RLX_COREML_UNITS` |

## cpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ARENA_ALIGN` | Bisect | U64 | Backend(cpu) | See call sites for `RLX_ARENA_ALIGN` |
| `RLX_ARENA_NO_REUSE` | Bisect | Bool | Backend(cpu) | See call sites for `RLX_ARENA_NO_REUSE` |
| `RLX_FAST_CONV` | Public | BoolOr | Backend(cpu) | CPU Conv2d im2col+BLAS path (default on; set 0 for scalar nested loops) |
| `RLX_PAR_THRESHOLD` | Bisect | U64 | Backend(cpu) | See call sites for `RLX_PAR_THRESHOLD` |
| `RLX_SDPA_THRESHOLD` | Bisect | U64 | Backend(cpu) | See call sites for `RLX_SDPA_THRESHOLD` |
| `RLX_WORKERS` | Bisect | U64 | Backend(cpu) | See call sites for `RLX_WORKERS` |

## cuda

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_CUDA_ARENA_DEBUG` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_ARENA_DEBUG` |
| `RLX_CUDA_ARENA_NO_REUSE` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_ARENA_NO_REUSE` |
| `RLX_CUDA_ARENA_POOL` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_ARENA_POOL` |
| `RLX_CUDA_ARENA_POOL_CHUNK_BYTES` | Bisect | U64 | Backend(cuda) | See call sites for `RLX_CUDA_ARENA_POOL_CHUNK_BYTES` |
| `RLX_CUDA_ARENA_POOL_MAX` | Bisect | U64 | Backend(cuda) | See call sites for `RLX_CUDA_ARENA_POOL_MAX` |
| `RLX_CUDA_COMPILE_MODE` | Bisect | String | Backend(cuda) | See call sites for `RLX_CUDA_COMPILE_MODE` |
| `RLX_CUDA_COMPILE_TIMING` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_COMPILE_TIMING` |
| `RLX_CUDA_CONV_BWD_CUDNN` | Public | Bool | Backend(cuda) | Allow cuDNN for grouped/degenerate Conv2d backward shapes |
| `RLX_CUDA_CONV_BWD_HOST` | Public | Bool | Backend(cuda) | Force CUDA Conv2d backward through CPU host-fallback (parity/debug) |
| `RLX_CUDA_CONV_FWD_CUDNN` | Public | Bool | Backend(cuda) | Force CUDA Conv2d forward through cuDNN (override host/default routing) |
| `RLX_CUDA_CONV_FWD_HOST` | Public | Bool | Backend(cuda) | Force CUDA Conv2d forward through CPU host-fallback (bisect) |
| `RLX_CUDA_CONV_STABLE_BWD` | Public | BoolOr | Backend(cuda) | Prefer cuDNN IMPLICIT_GEMM (ALGO_1) for conv backward (default on; set 0 to opt out) |
| `RLX_CUDA_CONV_TF32` | Public | Bool | Backend(cuda) | Enable TF32 tensor-core math for cuDNN conv (default is FMA) |
| `RLX_CUDA_DUMP_INTERMEDIATE` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_DUMP_INTERMEDIATE` |
| `RLX_CUDA_DUMP_NODES` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_DUMP_NODES` |
| `RLX_CUDA_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(cuda) | See call sites for `RLX_CUDA_DUMP_NODES_LIMIT` |
| `RLX_CUDA_EXEC_MODE` | Bisect | String | Backend(cuda) | See call sites for `RLX_CUDA_EXEC_MODE` |
| `RLX_CUDA_GDN_HOST` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_GDN_HOST` |
| `RLX_CUDA_GGUF_FUSED_M1` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_GGUF_FUSED_M1` |
| `RLX_CUDA_GGUF_HOST` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_GGUF_HOST` |
| `RLX_CUDA_IM2COL_HOST` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_IM2COL_HOST` |
| `RLX_CUDA_INDEXING_FULL_ARENA` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_INDEXING_FULL_ARENA` |
| `RLX_CUDA_LOG_CONV_PATH` | Bisect | Path | Backend(cuda) | See call sites for `RLX_CUDA_LOG_CONV_PATH` |
| `RLX_CUDA_LOG_FALLBACK` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_LOG_FALLBACK` |
| `RLX_CUDA_MATMUL_PRECISE` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_MATMUL_PRECISE` |
| `RLX_CUDA_NONDET_CONV` | Public | Bool | Backend(cuda) | Allow non-deterministic cuDNN conv backward algos (atomicAdd; faster, noisy) |
| `RLX_CUDA_NO_CUBLASLT` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_NO_CUBLASLT` |
| `RLX_CUDA_NO_CUDNN` | Public | Bool | Backend(cuda) | Skip cuDNN entirely (im2col / custom kernels only; silence missing-lib warning) |
| `RLX_CUDA_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_NO_PACKED_BSHD_ATTN` |
| `RLX_CUDA_NO_TF32` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_NO_TF32` |
| `RLX_CUDA_NO_ZERO_ARENA` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_NO_ZERO_ARENA` |
| `RLX_CUDA_PARITY` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_PARITY` |
| `RLX_CUDA_PATH_TRACE` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_PATH_TRACE` |
| `RLX_CUDA_PINNED_IO` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_PINNED_IO` |
| `RLX_CUDA_PTX_CACHE` | Bisect | Path | Backend(cuda) | See call sites for `RLX_CUDA_PTX_CACHE` |
| `RLX_CUDA_TRACE_FAB` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_TRACE_FAB` |
| `RLX_CUDA_WMMA` | Bisect | Bool | Backend(cuda) | See call sites for `RLX_CUDA_WMMA` |
| `RLX_CUDNN_DIR` | Public | Path | Backend(cuda) | Directory containing libcudnn.so* to preload (pip/conda wheel failsafe) |

## debug

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ALLOW_THROTTLE` | Public | Bool | Tooling | Skip thermal gate for one-off benches (prefer `just throttle`) |
| `RLX_CHECK` | Bisect | String | Runtime | See call sites for `RLX_CHECK` |
| `RLX_DBG_BINF` | Bisect | Bool | Runtime | See call sites for `RLX_DBG_BINF` |
| `RLX_DBG_CONV` | Bisect | Bool | Runtime | See call sites for `RLX_DBG_CONV` |
| `RLX_DBG_CUSTOM` | Public | Bool | Runtime | Log host custom-op staging (onnx.* dtype bridge) on GPU backends |
| `RLX_DBG_SHAPES` | Bisect | Bool | Runtime | See call sites for `RLX_DBG_SHAPES` |
| `RLX_DBG_STEP` | Bisect | Bool | Runtime | See call sites for `RLX_DBG_STEP` |
| `RLX_DEBUG_NANS` | Public | Enum | Runtime | Runtime NaN/Inf localizer (1 or abort) |
| `RLX_DEBUG_TRIP` | Bisect | Bool | Runtime | See call sites for `RLX_DEBUG_TRIP` |
| `RLX_DISPATCH_REPORT` | Public | Bool | Compile | Print legalize/dispatch report during compile (1 = on) |
| `RLX_ENV_DEPRECATIONS` | Public | Bool | Runtime | Emit one-shot messages when deprecated RLX_* aliases are used |
| `RLX_VERBOSE` | Public | Bool | Runtime | Extra runtime logging |

## device

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_BENCHMARK_PICK` | Public | U64 | Device | Micro-benchmark N runs to pick the fastest device (needs inputs) |
| `RLX_DEVICE` | Public | String | Device | Default device hint for resolved runs (cpu, metal, mlx, cuda, gpu, …) |
| `RLX_DEVICES` | Public | String | Device | Allow-list of devices for DevicePolicy::from_env |
| `RLX_DEVICE_CHAIN` | Public | String | Device | Fallback order when a preferred device fails (e.g. cuda,gpu,cpu) |

## fft

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_FFT_CPU_PARALLEL` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_CPU_PARALLEL` |
| `RLX_FFT_CUFFT` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_CUFFT` |
| `RLX_FFT_FORCE_MIXED` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_FORCE_MIXED` |
| `RLX_FFT_FUSE_DEBUG` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_FUSE_DEBUG` |
| `RLX_FFT_FUSE_REAL` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_FUSE_REAL` |
| `RLX_FFT_GEN` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_GEN` |
| `RLX_FFT_MULTIROW` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_MULTIROW` |
| `RLX_FFT_NATIVE` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_NATIVE` |
| `RLX_FFT_RADIX` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_RADIX` |
| `RLX_FFT_RADIX4` | Bisect | Bool | Tooling | See call sites for `RLX_FFT_RADIX4` |

## gpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_FFT_FAST` | Public | Bool | Runtime | Enable native on-chip GPU FFT when compiled (0 disables) |
| `RLX_INDEXING_FULL_ARENA` | Public | Bool | Runtime | Force full-arena mirror for indexing host-fallback (bisect; slow on discrete GPUs) |
| `RLX_QMATMUL_GPU_MIN_FLOPS` | Bisect | U64 | Runtime | See call sites for `RLX_QMATMUL_GPU_MIN_FLOPS` |
| `RLX_QMATMUL_INGRAPH` | Bisect | Bool | Runtime | See call sites for `RLX_QMATMUL_INGRAPH` |

## metal

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_DISABLE_METAL_DEQUANT_GPU` | Deprecated → `RLX_METAL_DEQUANT_GPU_DISABLE` | Bool | Backend(metal) | Deprecated alias of `RLX_METAL_DEQUANT_GPU_DISABLE` |
| `RLX_DISABLE_MPSGRAPH` | Public | Bool | Compile | Force Metal thunk path instead of MPSGraph regions |
| `RLX_DISABLE_MPSGRAPH_EXECUTABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_DISABLE_MPSGRAPH_EXECUTABLE` |
| `RLX_METAL_ATTN_BWD_GPU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_ATTN_BWD_GPU` |
| `RLX_METAL_ATTN_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_ATTN_TRACE` |
| `RLX_METAL_CMDBUF_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_CMDBUF_TRACE` |
| `RLX_METAL_CONCAT_HOST` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_CONCAT_HOST` |
| `RLX_METAL_CONCAT_MULTI` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_CONCAT_MULTI` |
| `RLX_METAL_CONV_BWD_IMPLICIT` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_CONV_BWD_IMPLICIT` |
| `RLX_METAL_DEBUG` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_DEBUG` |
| `RLX_METAL_DEQUANT_GPU_DISABLE` | Public | Bool | Backend(metal) | Disable Metal GPU GGUF dequant (host / legacy path) aliases: `RLX_DISABLE_METAL_DEQUANT_GPU` |
| `RLX_METAL_DEQUANT_MATMUL_LEGACY` | Public | Bool | Backend(metal) | Use pre-fused dequant+matmul path (materializes weights) |
| `RLX_METAL_DISABLE_NARROW_ROPE_FUSE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_DISABLE_NARROW_ROPE_FUSE` |
| `RLX_METAL_DUMP_NODES` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_DUMP_NODES` |
| `RLX_METAL_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(metal) | See call sites for `RLX_METAL_DUMP_NODES_LIMIT` |
| `RLX_METAL_EXTERNALIZE_QUANT` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_EXTERNALIZE_QUANT` |
| `RLX_METAL_EXT_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_EXT_TRACE` |
| `RLX_METAL_FA` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FA` |
| `RLX_METAL_FFT_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FFT_HOST_FALLBACK` |
| `RLX_METAL_FORCE_INLINE_PARAMS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FORCE_INLINE_PARAMS` |
| `RLX_METAL_FORCE_PIN_OUTPUT_ANCESTORS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FORCE_PIN_OUTPUT_ANCESTORS` |
| `RLX_METAL_FORCE_UNPIN_OUTPUT_ANCESTORS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FORCE_UNPIN_OUTPUT_ANCESTORS` |
| `RLX_METAL_FUSE_DECODE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_DECODE` |
| `RLX_METAL_FUSE_DECODE_GELU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_DECODE_GELU` |
| `RLX_METAL_FUSE_DECODE_LOG` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_DECODE_LOG` |
| `RLX_METAL_FUSE_DEPTHWISE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_DEPTHWISE` |
| `RLX_METAL_FUSE_GDN_NORM` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_GDN_NORM` |
| `RLX_METAL_FUSE_RESIDUAL_RMS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_FUSE_RESIDUAL_RMS` |
| `RLX_METAL_GDN_CPU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_GDN_CPU` |
| `RLX_METAL_GDN_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_GDN_HOST_FALLBACK` |
| `RLX_METAL_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_HOST_FALLBACK` |
| `RLX_METAL_HOST_SLICE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_HOST_SLICE` |
| `RLX_METAL_HYBRID_BIG_ARENA` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_HYBRID_BIG_ARENA` |
| `RLX_METAL_IQ1M_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ1M_FUSED_DISABLE` |
| `RLX_METAL_IQ1S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ1S_FUSED_DISABLE` |
| `RLX_METAL_IQ2S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ2S_FUSED_DISABLE` |
| `RLX_METAL_IQ2XS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ2XS_FUSED_DISABLE` |
| `RLX_METAL_IQ2XXS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ2XXS_FUSED_DISABLE` |
| `RLX_METAL_IQ3S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ3S_FUSED_DISABLE` |
| `RLX_METAL_IQ3XXS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ3XXS_FUSED_DISABLE` |
| `RLX_METAL_IQ4NL_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_IQ4NL_FUSED_DISABLE` |
| `RLX_METAL_LSTM_CPU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_LSTM_CPU` |
| `RLX_METAL_LSTM_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_LSTM_HOST_FALLBACK` |
| `RLX_METAL_MPSGRAPH_BIG_ARENA` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_MPSGRAPH_BIG_ARENA` |
| `RLX_METAL_MPS_PROFILE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_MPS_PROFILE` |
| `RLX_METAL_MPS_SDPA` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_MPS_SDPA` |
| `RLX_METAL_NARROW_BATCH` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_NARROW_BATCH` |
| `RLX_METAL_NO_FUSION` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_NO_FUSION` |
| `RLX_METAL_NO_SHARE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_NO_SHARE` |
| `RLX_METAL_ONNX_QMATMUL_GPU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_ONNX_QMATMUL_GPU` |
| `RLX_METAL_ONNX_QMATMUL_MIN_FLOPS` | Bisect | U64 | Backend(metal) | See call sites for `RLX_METAL_ONNX_QMATMUL_MIN_FLOPS` |
| `RLX_METAL_OUTPUT_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_OUTPUT_TRACE` |
| `RLX_METAL_PIPELINE_CACHE` | Bisect | Path | Backend(metal) | See call sites for `RLX_METAL_PIPELINE_CACHE` |
| `RLX_METAL_PRECISE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_PRECISE` |
| `RLX_METAL_Q1_0_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q1_0_FUSED_DISABLE` |
| `RLX_METAL_Q1_0_SG_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q1_0_SG_DISABLE` |
| `RLX_METAL_Q1_DUAL_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q1_DUAL_DISABLE` |
| `RLX_METAL_Q2_0_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q2_0_FUSED_DISABLE` |
| `RLX_METAL_Q2_0_SG_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q2_0_SG_DISABLE` |
| `RLX_METAL_Q2_DUAL_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q2_DUAL_DISABLE` |
| `RLX_METAL_Q40_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q40_FUSED_DISABLE` |
| `RLX_METAL_Q41_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q41_FUSED_DISABLE` |
| `RLX_METAL_Q4K_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q4K_FUSED_DISABLE` |
| `RLX_METAL_Q4K_GEMM_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q4K_GEMM_DISABLE` |
| `RLX_METAL_Q4K_SG_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q4K_SG_DISABLE` |
| `RLX_METAL_Q6K_GEMM_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q6K_GEMM_DISABLE` |
| `RLX_METAL_Q80_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_Q80_FUSED_DISABLE` |
| `RLX_METAL_RNN_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_RNN_HOST_FALLBACK` |
| `RLX_METAL_SAMPLE_HOST` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SAMPLE_HOST` |
| `RLX_METAL_SDPA_DECODE_M1` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SDPA_DECODE_M1` |
| `RLX_METAL_SGEMM_MPS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SGEMM_MPS` |
| `RLX_METAL_SGEMM_PRECISE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SGEMM_PRECISE` |
| `RLX_METAL_SGEMM_VARIANT` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SGEMM_VARIANT` |
| `RLX_METAL_SOFTMAX_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SOFTMAX_TRACE` |
| `RLX_METAL_SSM_CPU` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SSM_CPU` |
| `RLX_METAL_SSM_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_SSM_HOST_FALLBACK` |
| `RLX_METAL_THUNK_PROFILE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_THUNK_PROFILE` |
| `RLX_METAL_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_TRACE` |
| `RLX_METAL_TRACE_FAB` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_TRACE_FAB` |
| `RLX_METAL_UNFUSE_REGIONS` | Bisect | Bool | Backend(metal) | See call sites for `RLX_METAL_UNFUSE_REGIONS` |
| `RLX_MPSGRAPH_FORCE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_MPSGRAPH_FORCE` |
| `RLX_MPSGRAPH_MIN_FLOPS` | Bisect | U64 | Backend(metal) | See call sites for `RLX_MPSGRAPH_MIN_FLOPS` |
| `RLX_MPSGRAPH_PARAM_CONST` | Bisect | Bool | Backend(metal) | See call sites for `RLX_MPSGRAPH_PARAM_CONST` |
| `RLX_MPSGRAPH_PARAM_CONST_CAP` | Bisect | U64 | Backend(metal) | See call sites for `RLX_MPSGRAPH_PARAM_CONST_CAP` |
| `RLX_MPSGRAPH_TRACE` | Bisect | Bool | Backend(metal) | See call sites for `RLX_MPSGRAPH_TRACE` |
| `RLX_MPS_ALIGN_DEBUG` | Bisect | U64 | Backend(metal) | See call sites for `RLX_MPS_ALIGN_DEBUG` |
| `RLX_MPS_FP16` | Bisect | Bool | Backend(metal) | See call sites for `RLX_MPS_FP16` |
| `RLX_MPS_THRESHOLD_FLOP` | Bisect | U64 | Backend(metal) | See call sites for `RLX_MPS_THRESHOLD_FLOP` |

## misc

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ARCH` | Internal | Bool | Tooling | See call sites for `RLX_ARCH` |
| `RLX_ATTN_DEBUG` | Bisect | Bool | Tooling | See call sites for `RLX_ATTN_DEBUG` |
| `RLX_BENCH_CSV` | Internal | Bool | Tooling | See call sites for `RLX_BENCH_CSV` |
| `RLX_BENCH_DISPATCH_ONLY` | Internal | String | Tooling | See call sites for `RLX_BENCH_DISPATCH_ONLY` |
| `RLX_BENCH_ITERS` | Internal | Bool | Tooling | See call sites for `RLX_BENCH_ITERS` |
| `RLX_CPU_DUMP_FLAT` | Bisect | Bool | Tooling | See call sites for `RLX_CPU_DUMP_FLAT` |
| `RLX_CPU_DUMP_NODES` | Bisect | Bool | Tooling | See call sites for `RLX_CPU_DUMP_NODES` |
| `RLX_CPU_DUMP_NODES_LIMIT` | Bisect | U64 | Tooling | See call sites for `RLX_CPU_DUMP_NODES_LIMIT` |
| `RLX_DECODE_BUCKET_PEAK_BYTES` | Bisect | U64 | Tooling | See call sites for `RLX_DECODE_BUCKET_PEAK_BYTES` |
| `RLX_DECODE_BUCKET_RESIDENT_BYTES` | Bisect | U64 | Tooling | See call sites for `RLX_DECODE_BUCKET_RESIDENT_BYTES` |
| `RLX_DECODE_ONESHOT_PEAK_BYTES` | Bisect | U64 | Tooling | See call sites for `RLX_DECODE_ONESHOT_PEAK_BYTES` |
| `RLX_DECOMPOSE_FUSION_REGIONS` | Bisect | Bool | Tooling | See call sites for `RLX_DECOMPOSE_FUSION_REGIONS` |
| `RLX_DEQUANT_CACHE` | Bisect | Path | Tooling | See call sites for `RLX_DEQUANT_CACHE` |
| `RLX_DIRECT_CONV` | Bisect | Bool | Tooling | See call sites for `RLX_DIRECT_CONV` |
| `RLX_DISABLE_MPS` | Bisect | Bool | Tooling | See call sites for `RLX_DISABLE_MPS` |
| `RLX_DISABLE_NOMIC_FUSION` | Bisect | Bool | Tooling | See call sites for `RLX_DISABLE_NOMIC_FUSION` |
| `RLX_DUMP_SCHED` | Bisect | Bool | Tooling | See call sites for `RLX_DUMP_SCHED` |
| `RLX_ENABLE_FUSE_TRANSFORMER_LAYER` | Bisect | Bool | Tooling | See call sites for `RLX_ENABLE_FUSE_TRANSFORMER_LAYER` |
| `RLX_F32_DUMP` | Bisect | Bool | Tooling | See call sites for `RLX_F32_DUMP` |
| `RLX_FK_BATCH_SINGLE_KERNEL` | Bisect | Bool | Tooling | See call sites for `RLX_FK_BATCH_SINGLE_KERNEL` |
| `RLX_FORCE_DEVICE` | Bisect | Bool | Tooling | See call sites for `RLX_FORCE_DEVICE` |
| `RLX_FUSED_GPU_CONIC_SCANLINE` | Bisect | Bool | Compile | See call sites for `RLX_FUSED_GPU_CONIC_SCANLINE` |
| `RLX_GGUF_MATMUL_LEGACY` | Bisect | Bool | Tooling | See call sites for `RLX_GGUF_MATMUL_LEGACY` |
| `RLX_GGUF_TRACE` | Bisect | Bool | Tooling | See call sites for `RLX_GGUF_TRACE` |
| `RLX_GPU_HANDLE_HOST_MIRROR` | Bisect | Bool | Tooling | See call sites for `RLX_GPU_HANDLE_HOST_MIRROR` |
| `RLX_HIGHER_ORDER_NO_FUSE` | Bisect | Bool | Tooling | See call sites for `RLX_HIGHER_ORDER_NO_FUSE` |
| `RLX_HYBRID_K` | Bisect | Bool | Tooling | See call sites for `RLX_HYBRID_K` |
| `RLX_HYBRID_M` | Bisect | Bool | Tooling | See call sites for `RLX_HYBRID_M` |
| `RLX_ICB_TRACE` | Bisect | Bool | Tooling | See call sites for `RLX_ICB_TRACE` |
| `RLX_IMP_DBG` | Bisect | Bool | Tooling | See call sites for `RLX_IMP_DBG` |
| `RLX_IQ_TEST_DIR` | Bisect | Path | Tooling | See call sites for `RLX_IQ_TEST_DIR` |
| `RLX_IR_DUMP` | Bisect | Bool | Tooling | See call sites for `RLX_IR_DUMP` |
| `RLX_KEEP_ELEMENTWISE_REGIONS` | Bisect | Bool | Tooling | See call sites for `RLX_KEEP_ELEMENTWISE_REGIONS` |
| `RLX_KV_CACHE_DBG` | Bisect | Bool | Tooling | See call sites for `RLX_KV_CACHE_DBG` |
| `RLX_KV_CACHE_MAX_RESIDENT` | Bisect | U64 | Tooling | See call sites for `RLX_KV_CACHE_MAX_RESIDENT` |
| `RLX_KV_CACHE_NO_EVICT` | Bisect | Bool | Tooling | See call sites for `RLX_KV_CACHE_NO_EVICT` |
| `RLX_LM_HEAD_PARALLEL` | Bisect | Bool | Tooling | See call sites for `RLX_LM_HEAD_PARALLEL` |
| `RLX_LOG_EPOCH_LOSS` | Bisect | Bool | Tooling | See call sites for `RLX_LOG_EPOCH_LOSS` |
| `RLX_LSTM_DEBUG` | Bisect | Bool | Tooling | See call sites for `RLX_LSTM_DEBUG` |
| `RLX_MNIST_DIR` | Bisect | Path | Tooling | See call sites for `RLX_MNIST_DIR` |
| `RLX_MPSG_TRACE` | Bisect | Bool | Tooling | See call sites for `RLX_MPSG_TRACE` |
| `RLX_NATIVE_FK_REGIONS` | Bisect | Bool | Compile | See call sites for `RLX_NATIVE_FK_REGIONS` |
| `RLX_NEMO_TEST_FILE` | Bisect | Bool | Tooling | See call sites for `RLX_NEMO_TEST_FILE` |
| `RLX_NO_FK_FUSION` | Bisect | Bool | Compile | See call sites for `RLX_NO_FK_FUSION` |
| `RLX_NO_NATIVE_FK_REGIONS` | Bisect | Bool | Tooling | See call sites for `RLX_NO_NATIVE_FK_REGIONS` |
| `RLX_NO_SHARED_INPUT_MATMUL` | Bisect | Bool | Tooling | See call sites for `RLX_NO_SHARED_INPUT_MATMUL` |
| `RLX_NO_SHUFFLE` | Internal | Bool | Tooling | See call sites for `RLX_NO_SHUFFLE` |
| `RLX_ORT_CUDA_GRAPH` | Bisect | Bool | Tooling | See call sites for `RLX_ORT_CUDA_GRAPH` |
| `RLX_PARITY_DEVICE` | Bisect | Bool | Tooling | See call sites for `RLX_PARITY_DEVICE` |
| `RLX_PHASE_TIMING` | Bisect | Bool | Tooling | See call sites for `RLX_PHASE_TIMING` |
| `RLX_PROBE_DYNAMIC` | Bisect | Bool | Tooling | See call sites for `RLX_PROBE_DYNAMIC` |
| `RLX_PROBE_FEATURE_DIM` | Bisect | Bool | Tooling | See call sites for `RLX_PROBE_FEATURE_DIM` |
| `RLX_PROFILE_COMPILE` | Bisect | Bool | Tooling | See call sites for `RLX_PROFILE_COMPILE` |
| `RLX_PROFILE_THUNKS` | Bisect | Bool | Tooling | See call sites for `RLX_PROFILE_THUNKS` |
| `RLX_RIG_RUNTIME` | Bisect | Bool | Tooling | See call sites for `RLX_RIG_RUNTIME` |
| `RLX_ROPE_DEBUG` | Bisect | Bool | Tooling | See call sites for `RLX_ROPE_DEBUG` |
| `RLX_SOFT_MEMORY_BUDGET_BYTES` | Bisect | U64 | Tooling | See call sites for `RLX_SOFT_MEMORY_BUDGET_BYTES` |
| `RLX_SOFT_MEMORY_FRACTION` | Bisect | Bool | Tooling | See call sites for `RLX_SOFT_MEMORY_FRACTION` |
| `RLX_SPD_JACOBI_SWEEPS` | Bisect | U64 | Tooling | See call sites for `RLX_SPD_JACOBI_SWEEPS` |
| `RLX_SPD_UNROLL` | Bisect | Bool | Tooling | See call sites for `RLX_SPD_UNROLL` |
| `RLX_TRACE_PERFETTO` | Bisect | Bool | Tooling | See call sites for `RLX_TRACE_PERFETTO` |
| `RLX_TRACE_THUNK` | Bisect | Bool | Tooling | See call sites for `RLX_TRACE_THUNK` |
| `RLX_UMAP_CUDA_FUSED_KNN` | Bisect | Bool | Tooling | See call sites for `RLX_UMAP_CUDA_FUSED_KNN` |
| `RLX_USE_ICB` | Bisect | Bool | Tooling | See call sites for `RLX_USE_ICB` |
| `RLX_WINOGRAD` | Bisect | Bool | Tooling | See call sites for `RLX_WINOGRAD` |

## mlx

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_MLX_COMPILE_MAX_NODES` | Bisect | U64 | Backend(mlx) | See call sites for `RLX_MLX_COMPILE_MAX_NODES` |
| `RLX_MLX_CUDA_ARCH` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_CUDA_ARCH` |
| `RLX_MLX_DEBUG_EVAL` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_DEBUG_EVAL` |
| `RLX_MLX_DEQUANT_CACHE_BYTES` | Bisect | U64 | Backend(mlx) | See call sites for `RLX_MLX_DEQUANT_CACHE_BYTES` |
| `RLX_MLX_DEQUANT_CACHE_DISABLE` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_DEQUANT_CACHE_DISABLE` |
| `RLX_MLX_DEVICE` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_DEVICE` |
| `RLX_MLX_FUSE_CAP` | Bisect | U64 | Backend(mlx) | See call sites for `RLX_MLX_FUSE_CAP` |
| `RLX_MLX_GGUF_HOST_FALLBACK` | Public | Bool | Backend(mlx) | Force host GGUF dequant on MLX |
| `RLX_MLX_JOBS` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_JOBS` |
| `RLX_MLX_MODE` | Public | Enum | Backend(mlx) | eager | lazy | compiled execution mode |
| `RLX_MLX_PARAM_VIEW` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_PARAM_VIEW` |
| `RLX_MLX_PROFILE` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_PROFILE` |
| `RLX_MLX_Q1_HOST` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_Q1_HOST` |
| `RLX_MLX_Q1_MV_DISABLE` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_Q1_MV_DISABLE` |
| `RLX_MLX_RNN_F16` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_RNN_F16` |
| `RLX_MLX_SDPA_REFERENCE` | Public | Bool | Backend(mlx) | Use reference SDPA composition for bisects |
| `RLX_MLX_WARN_LAZY` | Bisect | Bool | Backend(mlx) | See call sites for `RLX_MLX_WARN_LAZY` |

## oneapi

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ONEAPI_BUILD_KERNELS` | Bisect | Bool | Backend(oneapi) | See call sites for `RLX_ONEAPI_BUILD_KERNELS` |
| `RLX_ONEAPI_LOADER` | Bisect | Bool | Backend(oneapi) | See call sites for `RLX_ONEAPI_LOADER` |
| `RLX_ONEAPI_OCLOC` | Bisect | Bool | Backend(oneapi) | See call sites for `RLX_ONEAPI_OCLOC` |
| `RLX_ONEAPI_OCLOC_DEVICE` | Bisect | Bool | Backend(oneapi) | See call sites for `RLX_ONEAPI_OCLOC_DEVICE` |

## onnx

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ONNX_BUNDLE` | Bisect | Bool | Tooling | See call sites for `RLX_ONNX_BUNDLE` |
| `RLX_ONNX_SEQUENCE_LENGTH` | Bisect | Bool | Tooling | See call sites for `RLX_ONNX_SEQUENCE_LENGTH` |
| `RLX_ONNX_TAP` | Bisect | Bool | Tooling | See call sites for `RLX_ONNX_TAP` |
| `RLX_ONNX_TEST_MODEL` | Bisect | String | Tooling | See call sites for `RLX_ONNX_TEST_MODEL` |

## profile

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_TIMING` | Bisect | Bool | Runtime | See call sites for `RLX_TIMING` |

## qnn

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_QNN_BACKEND_LIB` | Bisect | Bool | Backend(qnn) | See call sites for `RLX_QNN_BACKEND_LIB` |

## rocm

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ROCM_HSACO_CACHE` | Bisect | Path | Backend(rocm) | See call sites for `RLX_ROCM_HSACO_CACHE` |
| `RLX_ROCM_IM2COL_HOST` | Bisect | Bool | Backend(rocm) | See call sites for `RLX_ROCM_IM2COL_HOST` |
| `RLX_ROCM_LOG_FALLBACK` | Bisect | Bool | Backend(rocm) | See call sites for `RLX_ROCM_LOG_FALLBACK` |
| `RLX_ROCM_MFMA` | Bisect | Bool | Backend(rocm) | See call sites for `RLX_ROCM_MFMA` |
| `RLX_ROCM_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(rocm) | See call sites for `RLX_ROCM_NO_PACKED_BSHD_ATTN` |
| `RLX_ROCM_PINNED_IO` | Public | Bool | Backend(rocm) | Use pinned host I/O for ROCm graph exec (default on in graph mode) |

## tpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_TPU_HLO_DUMP` | Bisect | Bool | Backend(tpu) | See call sites for `RLX_TPU_HLO_DUMP` |

## vulkan

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_VULKAN_ARENA_DEBUG` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_ARENA_DEBUG` |
| `RLX_VULKAN_CHECK_BCAST` | Bisect | String | Backend(vulkan) | See call sites for `RLX_VULKAN_CHECK_BCAST` |
| `RLX_VULKAN_CHECK_CAST` | Bisect | String | Backend(vulkan) | See call sites for `RLX_VULKAN_CHECK_CAST` |
| `RLX_VULKAN_DEBUG` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_DEBUG` |
| `RLX_VULKAN_DUMP_OPS` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_DUMP_OPS` |
| `RLX_VULKAN_FULLBARRIER` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_FULLBARRIER` |
| `RLX_VULKAN_HOST_CONV` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_HOST_CONV` |
| `RLX_VULKAN_HOST_OPS` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_HOST_OPS` |
| `RLX_VULKAN_MATMUL` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_MATMUL` |
| `RLX_VULKAN_NOBARRIER` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_NOBARRIER` |
| `RLX_VULKAN_NOCACHE` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_NOCACHE` |
| `RLX_VULKAN_SCAN_ALL` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_SCAN_ALL` |
| `RLX_VULKAN_SHARD_LOG` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_SHARD_LOG` |
| `RLX_VULKAN_VALIDATION` | Bisect | Bool | Backend(vulkan) | See call sites for `RLX_VULKAN_VALIDATION` |

## wgpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_WGPU_ACTIVE_TRACE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_ACTIVE_TRACE` |
| `RLX_WGPU_CONCAT_HOST` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_CONCAT_HOST` |
| `RLX_WGPU_CONV_IM2COL` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_CONV_IM2COL` |
| `RLX_WGPU_COOP_F16_VK_DISABLE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_DISABLE` |
| `RLX_WGPU_COOP_F16_VK_ENABLE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_ENABLE` |
| `RLX_WGPU_COOP_F16_VK_FORCE_WIDE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_FORCE_WIDE` |
| `RLX_WGPU_COOP_F16_VK_LARGE_N` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_LARGE_N` |
| `RLX_WGPU_COOP_F16_VK_LOAD_T` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_LOAD_T` |
| `RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE` |
| `RLX_WGPU_COOP_F16_VK_NO_F32ACC` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_NO_F32ACC` |
| `RLX_WGPU_COOP_F16_VK_OSC_THRESH` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_COOP_F16_VK_OSC_THRESH` |
| `RLX_WGPU_DEBUG` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DEBUG` |
| `RLX_WGPU_DEBUG_ATTN_ALIAS` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DEBUG_ATTN_ALIAS` |
| `RLX_WGPU_DEBUG_ATTN_MASK` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DEBUG_ATTN_MASK` |
| `RLX_WGPU_DEBUG_OP_HIST` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DEBUG_OP_HIST` |
| `RLX_WGPU_DUMP_FLAT` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_FLAT` |
| `RLX_WGPU_DUMP_IDS` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_IDS` |
| `RLX_WGPU_DUMP_INPUTS` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_INPUTS` |
| `RLX_WGPU_DUMP_NODES` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_NODES` |
| `RLX_WGPU_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_NODES_LIMIT` |
| `RLX_WGPU_DUMP_TAIL` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_DUMP_TAIL` |
| `RLX_WGPU_F16_WEIGHTS` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_F16_WEIGHTS` |
| `RLX_WGPU_FORCE_COOP_F32` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_FORCE_COOP_F32` |
| `RLX_WGPU_FORCE_INPUT_UPLOAD` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_FORCE_INPUT_UPLOAD` |
| `RLX_WGPU_GDN_HOST` | Public | Bool | Backend(wgpu) | Force GatedDeltaNet host fallback on wgpu (skip WGSL) |
| `RLX_WGPU_HOST_BUFFER_COPY` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_HOST_BUFFER_COPY` |
| `RLX_WGPU_IM2COL_MIN_COUT` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_IM2COL_MIN_COUT` |
| `RLX_WGPU_IM2COL_MIN_K` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_IM2COL_MIN_K` |
| `RLX_WGPU_IM2COL_MIN_SPATIAL` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_IM2COL_MIN_SPATIAL` |
| `RLX_WGPU_LARGE_BUFFERS` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_LARGE_BUFFERS` |
| `RLX_WGPU_MATMUL_F32_ONLY` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_MATMUL_F32_ONLY` |
| `RLX_WGPU_NAN_TRACE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NAN_TRACE` |
| `RLX_WGPU_NO_COOP_F16_VK` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_COOP_F16_VK` |
| `RLX_WGPU_NO_COOP_F32` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_COOP_F32` |
| `RLX_WGPU_NO_F16_MIRROR` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_F16_MIRROR` |
| `RLX_WGPU_NO_F16_SHADOW` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_F16_SHADOW` |
| `RLX_WGPU_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_PACKED_BSHD_ATTN` |
| `RLX_WGPU_NO_TILED_CONV` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_NO_TILED_CONV` |
| `RLX_WGPU_PRINT_LIMITS` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_PRINT_LIMITS` |
| `RLX_WGPU_Q1_0_GEMM_DISABLE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_Q1_0_GEMM_DISABLE` |
| `RLX_WGPU_SCHEDULE` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_SCHEDULE` |
| `RLX_WGPU_SHARD_LOG` | Bisect | Bool | Backend(wgpu) | See call sites for `RLX_WGPU_SHARD_LOG` |
| `RLX_WGPU_TILED_MIN_SPATIAL` | Bisect | U64 | Backend(wgpu) | See call sites for `RLX_WGPU_TILED_MIN_SPATIAL` |

## Unregistered mentions

Identifiers still appearing in the tree but not yet in the registry (docs, benches, or pending migration). Prefer registering or deleting.

| Name | Example path |
|------|--------------|
| `RLX_BACKENDS_MANIFEST_PATH` | `crates/core/rlx-runtime/src/backends_manifest.rs` |
| `RLX_BATCH` | `CHANGELOG.md` |
| `RLX_BENCH` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_COMPILE_OUTPUT_CAP` | `crates/backends/rlx-mlx/src/config.rs` |
| `RLX_CUDA_FORCE_ATTENTION_ROW` | `crates/core/rlx-ir/src/attention_layout.rs` |
| `RLX_CUDA_FULL_KV_READBACK` | `crates/backends/rlx-cuda/README.md` |
| `RLX_DECOMPOSE_BENCH_RUNS` | `rig.sh` |
| `RLX_DECOMPOSE_BENCH_WARMUP` | `rig.sh` |
| `RLX_DENY_DEVICES` | `docs/backend-selection.md` |
| `RLX_DETERMINISTIC_REDUCE` | `crates/core/rlx-collectives/src/lib.rs` |
| `RLX_DISABLE_MPSGRAPH_HYBRID` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_DISABLE_MPSGRAPH_PARAM_CONST` | `crates/backends/rlx-metal/src/backend/mod.rs` |
| `RLX_FFT_E2E_APPLE_FEATURES` | `rig.sh` |
| `RLX_FFT_E2E_APPLE_JSON` | `rig.sh` |
| `RLX_FFT_E2E_CUDA_JSON` | `rig.sh` |
| `RLX_FFT_E2E_DISTILL_STEPS` | `rig.sh` |
| `RLX_FFT_E2E_FEATURES` | `rig.sh` |
| `RLX_FFT_E2E_HTML` | `rig.sh` |
| `RLX_FFT_E2E_ITERS` | `rig.sh` |
| `RLX_FFT_E2E_STEPS` | `rig.sh` |
| `RLX_FFT_MODELS_ROOT` | `rig.sh` |
| `RLX_FFT_PICKER_TRACE` | `rig.sh` |
| `RLX_FFT_RIG_RUNTIME` | `rig.sh` |
| `RLX_FFT_WELCH_DEBUG` | `rig.sh` |
| `RLX_FFT_WELCH_ITERS` | `rig.sh` |
| `RLX_FFT_WELCH_TRAIN_STEPS` | `rig.sh` |
| `RLX_FFT_WGPU_BIG` | `crates/backends/rlx-wgpu/src/fft_dispatch.rs` |
| `RLX_FFT_WGPU_ONCHIP` | `crates/backends/rlx-wgpu/src/fft_dispatch.rs` |
| `RLX_GEMMA3_GGUF` | `rig.sh` |
| `RLX_GRAPH_FUSED` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_IROH_ALPN` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_PEERS` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_SECRET` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_SEED` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_KERNELS_MSL` | `crates/backends/rlx-metal/src/icb.rs` |
| `RLX_KERNELS_MSL_DEQUANT` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_FFT_GPU` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_SPLAT` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_SPLAT_CONIC` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_LOCATEANYTHING_DIR` | `rig.sh` |
| `RLX_LR` | `docs/benchmarks/coreml-training.md` |
| `RLX_METAL_IQ` | `crates/backends/rlx-metal/README.md` |
| `RLX_MINICPM5_GGUF_DIR` | `rig.sh` |
| `RLX_MINICPM5_GGUF_Q4_K_M` | `rig.sh` |
| `RLX_MLX_BENCH_PROFILE` | `rig.sh` |
| `RLX_MLX_COMPILE_OUTPUT_CAP` | `crates/backends/rlx-mlx/src/config.rs` |
| `RLX_MLX_CUDA` | `crates/backends/rlx-mlx-sys/build.rs` |
| `RLX_MLX_NO_CCACHE` | `crates/backends/rlx-mlx-sys/build.rs` |
| `RLX_MLX_OK` | `crates/backends/rlx-mlx/src/array.rs` |
| `RLX_MODELS_BUILD_WIN` | `rig.sh` |
| `RLX_MODELS_SRC` | `rig.sh` |
| `RLX_MODELS_WIN` | `rig.sh` |
| `RLX_MODELS_WSL` | `rig.sh` |
| `RLX_MPSGRAPH_ATTENTION` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_NTH_ORDER_RUNS` | `rig.sh` |
| `RLX_NTH_ORDER_SIZES` | `rig.sh` |
| `RLX_NTH_ORDER_WARMUP` | `rig.sh` |
| `RLX_ORT_INTRA_THREADS` | `crates/io/rlx-onnx/src/backend.rs` |
| `RLX_PIPELINE_ALPN` | `crates/core/rlx-driver/src/lib.rs` |
| `RLX_PREFER_DEVICES` | `docs/backend-selection.md` |
| `RLX_QNN_HTP_LIB` | `Justfile` |
| `RLX_QWEN25_GGUF` | `rig.sh` |
| `RLX_QWEN3_F16_LM_HEAD` | `CHANGELOG.md` |
| `RLX_QWEN3_PARITY` | `CHANGELOG.md` |
| `RLX_QWEN3_TTS_DIR` | `rig.sh` |
| `RLX_REGIONS` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_RESIDENT` | `docs/benchmarks/frameworks-and-backends.md` |
| `RLX_RIG_BUILD` | `rig.sh` |
| `RLX_RIG_CUDA_BENCH` | `rig.sh` |
| `RLX_RIG_DEST` | `scripts/sync-to-rig.sh` |
| `RLX_RIG_HOST` | `scripts/sync-to-rig.sh` |
| `RLX_RIG_MODELS_FEATURES` | `rig.sh` |
| `RLX_RIG_ROOT` | `rig.sh` |
| `RLX_RIG_SKIP_SYNC` | `rig.sh` |
| `RLX_RIG_SYNC_NO_PRUNE` | `rig.sh` |
| `RLX_RIG_WORKSPACE` | `rig.sh` |
| `RLX_ROCM_FORCE_ATTENTION_ROW` | `crates/backends/rlx-rocm/src/backend/run.rs` |
| `RLX_SIM_DEVICE` | `Justfile` |
| `RLX_SKIP_WRITEBACK` | `docs/benchmarks/frameworks-and-backends.md` |
| `RLX_SMOLLM2_GGUF` | `rig.sh` |
| `RLX_TAP_L0` | `crates/backends/rlx-metal/src/backend/encode/mod.rs` |
| `RLX_TORCH_IMPORT_BIN` | `crates/bindings/pyrlx/pyproject.toml` |
| `RLX_TPU_BENCH` | `crates/backends/rlx-tpu/tests/pjrt_bench.rs` |
| `RLX_TPU_BENCH_SWEEP` | `crates/backends/rlx-tpu/tests/pjrt_bench.rs` |
| `RLX_TRANSPORT` | `docs/iroh-transport.md` |
| `RLX_USE_MPSGRAPH` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_USE_MPS_GRAPH` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_WHISPER_DIR` | `rig.sh` |

## Maintenance

```sh
just gen-rlx-env-vars
# or: python3 scripts/gen-rlx-env-vars.py
```

Add new names to `env_registry_data.inc.rs`. Unregistered `env::flag("RLX_…")` call sites fail `just check-rlx-env-vars`.
## License

MIT OR Apache-2.0.
