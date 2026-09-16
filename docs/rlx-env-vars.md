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

**Registered names:** 626  
**Unregistered mentions (migration leftovers):** 136

## Groups

- [compile](#compile) — 22
- [coreml](#coreml) — 14
- [cpu](#cpu) — 29
- [cuda](#cuda) — 70
- [debug](#debug) — 12
- [device](#device) — 4
- [egpu](#egpu) — 6
- [fft](#fft) — 10
- [gpu](#gpu) — 4
- [gpu-host](#gpu-host) — 1
- [metal](#metal) — 184
- [misc](#misc) — 97
- [mlx](#mlx) — 20
- [oneapi](#oneapi) — 5
- [onnx](#onnx) — 5
- [qnn](#qnn) — 1
- [quant](#quant) — 1
- [rocm](#rocm) — 31
- [tpu](#tpu) — 4
- [vulkan](#vulkan) — 19
- [wgpu](#wgpu) — 77
- [xdna](#xdna) — 10

## compile

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ARENA_PIN_HOST_STRUCTURE` | Internal | Bool | Compile | Pin host-visible structure across planner entry points |
| `RLX_CACHE_PARAM_INVARIANT` | Public | Bool | Compile | Hoist param-invariant subgraph into prepare-once graph |
| `RLX_DISABLE_CONV_BIAS_ACT_FUSION` | Public | Bool | Compile | Skip Conv+Bias+Act fusion |
| `RLX_DISABLE_GATED_RESIDUAL_FUSION` | Internal | Bool | Compile | Keep gated residuals decomposed (fusion ablation) |
| `RLX_DISABLE_MATMUL_BIAS_FUSION` | Internal | Bool | Compile | Disable matmul+bias+activation fusion (ablation) |
| `RLX_DISABLE_RESIDUAL_LN_FUSION` | Internal | Bool | Compile | Disable residual+LayerNorm fusion (ablation) |
| `RLX_DISABLE_SCCP` | Bisect | Bool | Compile | Disable sparse conditional constant propagation (A/B the pass) |
| `RLX_FOLD_LIVENESS_DEBUG` | Internal | Bool | Compile | Report how many Conv3D epilogue folds extended a live range (absorbed / read-through) |
| `RLX_FUSE_ATTN_THRESHOLD` | Bisect | U64 | Compile | Sequence length above which attention fusion engages |
| `RLX_FUSE_BATCH_PREPROCESS` | Bisect | Bool | Compile | Merge parallel region slices into BatchElementwiseRegion; 0 disables |
| `RLX_FUSE_REGION_PROLOGUE` | Bisect | Bool | Compile | Fold ResizeNearest2x into the ElementwiseRegion prologue; 0 disables |
| `RLX_FUSION_REPORT` | Public | Bool | Compile | Print fusion pass before/after report |
| `RLX_GDN_UNFUSE_FOR_AD` | Bisect | Bool | Compile | Unroll Op::GatedDeltaNet's time loop for autodiff instead of emitting the fused backward (~32x slower) |
| `RLX_INSPECT_BINS` | Internal | String | Compile | Histogram bin count for the op tap (default 24) |
| `RLX_INSPECT_OPS` | Internal | String | Compile | Tag every op output for the inspection tap |
| `RLX_KERNEL_DISPATCH` | Public | Enum | Compile | common|native|force_common|force_native kernel dispatch policy |
| `RLX_LINT_NUMERICS` | Public | Bool | Compile | Static provable NaN/Inf lint during compile |
| `RLX_MEM_VERIFY` | Internal | Bool | Compile | Run the memory-planner invariant self-check |
| `RLX_NO_IO_PEAKS_OUTPUT` | Public | Bool | Compile | Disable compile-time IO-gated peaks-only fusion |
| `RLX_NO_WEIGHT_CONCAT_FUSION` | Internal | Bool | Compile | Skip the fusions that concatenate weights into one buffer |
| `RLX_PIN_OUTPUT_ANCESTORS` | Internal | Bool | Compile | Give every output ancestor its own arena slot |
| `RLX_PLAN_VERIFY` | Bisect | Bool | Compile | Re-derive every memory-plan offset with the O(N^3) candidate scan and assert the sweep agrees |

## coreml

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_COREML_DBG_CONV` | Internal | Bool | Backend(coreml) | Trace CoreML conv placement decisions |
| `RLX_COREML_DEBUG_IO` | Internal | String | Backend(coreml) | Dump CoreML model inputs and outputs at finalize |
| `RLX_COREML_F16` | Bisect | Bool | Backend(coreml) | Store CoreML activations and dequant output in f16 (~half bandwidth) |
| `RLX_COREML_FLEXIBLE_INPUTS` | Bisect | Bool | Backend(coreml) | Declare CoreML inputs as flexible-shape |
| `RLX_COREML_HOST_DEQUANT` | Public | Bool | Backend(coreml) | Force CoreML hybrid host dequant segments |
| `RLX_COREML_HOST_RESHAPED_REDUCE` | Internal | Bool | Backend(coreml) | Host-execute Reduce over a reshaped Concat (MIL miscompiles it) |
| `RLX_COREML_MAX_CONV_KERNEL` | Internal | U64 | Backend(coreml) | Widest conv kernel left on MIL; wider convs go host |
| `RLX_COREML_MAX_SCAN_LEN` | Internal | U64 | Backend(coreml) | Longest SelectiveScan left on MIL before host fallback |
| `RLX_COREML_NATIVE_FLEX` | Bisect | Bool | Backend(coreml) | Compile dynamic graphs once with CoreML ShapeRange instead of per shape |
| `RLX_COREML_NATIVE_SCAN` | Bisect | Bool | Backend(coreml) | Lower Scan natively on CoreML; set 0 to host it |
| `RLX_COREML_Q1_MODE` | Bisect | String | Backend(coreml) | Q1_0 on-device lowering mode: lut (default) | f32 (legacy unfold) |
| `RLX_COREML_SEG_REPORT` | Bisect | Bool | Backend(coreml) | Report the CoreML/host segmentation plan |
| `RLX_COREML_UNITS` | Bisect | String | Backend(coreml) | CoreML compute units: cpu | gpu | all | ane |
| `RLX_COREML_VERIFY` | Bisect | String | Backend(coreml) | MIL program verifier: `strict` rejects a malformed program at build time |

## cpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ARENA_ALIGN` | Bisect | U64 | Backend(cpu) | Arena buffer alignment in bytes (default 64) |
| `RLX_ARENA_CHECK` | Internal | String | Backend(cpu) | Verify CPU arena plan invariants when building from a plan |
| `RLX_ARENA_NO_REUSE` | Bisect | Bool | Backend(cpu) | Give every tensor its own arena slot; disables slot reuse |
| `RLX_ARENA_WRITE_TRACE` | Bisect | Bool | Backend(cpu) | Log the arena byte range each CPU thunk dirties (finds slot aliasing/overruns) |
| `RLX_ATTN_BWD_SERIAL` | Internal | String | Backend(cpu) | Run CPU attention backward serially instead of with Rayon |
| `RLX_BLAS_LINK` | Internal | String | Backend(cpu) | Escape hatch naming the BLAS/LAPACK libraries to link |
| `RLX_BLAS_SEARCH` | Internal | String | Backend(cpu) | Extra directories searched for the RLX_BLAS_LINK libraries |
| `RLX_CPU_ARENA_REPORT` | Internal | Bool | Backend(cpu) | Attribute the CPU arena's resident bytes to the nodes holding them |
| `RLX_CPU_ATTN_BWD_FUSE` | Internal | String | Backend(cpu) | Opt in to the fused CPU attention-backward path (default off) |
| `RLX_CPU_BNNS_BF16` | Internal | String | Backend(cpu) | Opt in to the BNNS bf16 GEMM path (lossy f32 to bf16 downcast) |
| `RLX_CPU_BNNS_F16` | Internal | String | Backend(cpu) | Opt in to the BNNS f16 GEMM path (half bandwidth) |
| `RLX_CPU_DUMP_DIR` | Internal | Path | Backend(cpu) | Directory for RLX_CPU_DUMP_NODE_DATA output |
| `RLX_CPU_DUMP_NODE_DATA` | Internal | String | Backend(cpu) | Write listed nodes' raw f32 arena buffers to files |
| `RLX_CPU_DUMP_THUNKS` | Internal | String | Backend(cpu) | Print compiled thunks in a <lo>:<hi> index range |
| `RLX_CPU_MATMUL_F64_ACCUM` | Public | Bool | Backend(cpu) | Accumulate the CPU matmul K-reduction in f64 (precision/validated mode; default-off, vendor-BLAS fast path untouched) |
| `RLX_CPU_ROW_TABS` | Internal | U64 | Backend(cpu) | Per-row sum of |x| for one node id |
| `RLX_CPU_SME` | Internal | String | Backend(cpu) | Opt in to the direct ARM SME2 GEMM path (Apple M4+) |
| `RLX_CPU_SME_BF16` | Internal | String | Backend(cpu) | Opt in to the native SME bf16 path (Apple M4+) |
| `RLX_CPU_SME_W8A8` | Internal | String | Backend(cpu) | Opt in to the wired SME W8A8 integer path (Apple M4+) |
| `RLX_CPU_WATCH` | Internal | String | Backend(cpu) | Report every thunk that changes <byte_offset>:<n_floats> |
| `RLX_DEQUANT_CACHE_MAX_BYTES` | Internal | String | Backend(cpu) | Total byte budget for the dequant cache |
| `RLX_DEQUANT_PARALLEL` | Internal | String | Backend(cpu) | Dequantize GGUF blocks in parallel; set 0 to serialize |
| `RLX_FAST_CONV` | Public | BoolOr | Backend(cpu) | CPU Conv2d im2col+BLAS path (default on; set 0 for scalar nested loops) |
| `RLX_PAR_THRESHOLD` | Bisect | U64 | Backend(cpu) | Minimum element count before an op dispatches in parallel |
| `RLX_Q4K_FUSED_MIN_N` | Internal | String | Backend(cpu) | n above which the fused Q4_K kernel is used instead of dequant |
| `RLX_Q4K_NO_DOTPROD` | Internal | String | Backend(cpu) | Force the baseline Q4_K path instead of the dotprod intrinsics |
| `RLX_SDPA_THRESHOLD` | Bisect | U64 | Backend(cpu) | Sequence length above which SDPA switches from NEON dots to BLAS sgemm |
| `RLX_VMATH_ACCURATE` | Public | BoolOr | Backend(cpu) | CPU vmath exp/tanh/log/sqrt: use Accelerate/libm accurate path (default 0 = SIMD *_fast) |
| `RLX_WORKERS` | Bisect | U64 | Backend(cpu) | Thread-pool size (0 = auto) |

## cuda

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_CUDA_ARENA_DEBUG` | Bisect | Bool | Backend(cuda) | Log CUDA arena allocation and slot reuse |
| `RLX_CUDA_ARENA_NO_REUSE` | Bisect | Bool | Backend(cuda) | Give every CUDA tensor its own arena slot; disables slot reuse |
| `RLX_CUDA_ARENA_POOL` | Bisect | Bool | Backend(cuda) | Pool CUDA arena allocations across compiles |
| `RLX_CUDA_ARENA_POOL_CHUNK_BYTES` | Bisect | U64 | Backend(cuda) | Max bytes per pooled CUDA arena chunk |
| `RLX_CUDA_ARENA_POOL_MAX` | Bisect | U64 | Backend(cuda) | Max buffers retained in the CUDA arena pool |
| `RLX_CUDA_ATTN_RANK4` | Internal | Bool | Backend(cuda) | Force the rank-4 CUDA attention-backward promotion |
| `RLX_CUDA_CAPTURE_DEBUG` | Internal | Bool | Backend(cuda) | Check CUDA graph capture state once per step after dispatch |
| `RLX_CUDA_COMPILE_MODE` | Bisect | String | Backend(cuda) | CUDA compile mode: jit (default) | aot |
| `RLX_CUDA_COMPILE_TIMING` | Bisect | Bool | Backend(cuda) | Report NVRTC compile time per kernel |
| `RLX_CUDA_CONV_BWD_CUDNN` | Public | Bool | Backend(cuda) | Allow cuDNN for grouped/degenerate Conv2d backward shapes |
| `RLX_CUDA_CONV_BWD_HOST` | Public | Bool | Backend(cuda) | Force CUDA Conv2d backward through CPU host-fallback (parity/debug) |
| `RLX_CUDA_CONV_FORCE_GATHER` | Internal | Bool | Backend(cuda) | Force the gather-based conv path instead of cuDNN |
| `RLX_CUDA_CONV_FWD_CUDNN` | Public | Bool | Backend(cuda) | Force CUDA Conv2d forward through cuDNN (override host/default routing) |
| `RLX_CUDA_CONV_FWD_HOST` | Public | Bool | Backend(cuda) | Force CUDA Conv2d forward through CPU host-fallback (bisect) |
| `RLX_CUDA_CONV_STABLE_BWD` | Public | BoolOr | Backend(cuda) | Prefer cuDNN IMPLICIT_GEMM (ALGO_1) for conv backward (default on; set 0 to opt out) |
| `RLX_CUDA_CONV_TF32` | Public | Bool | Backend(cuda) | Enable TF32 tensor-core math for cuDNN conv (default is FMA) |
| `RLX_CUDA_CONV_TRACE` | Internal | Bool | Backend(cuda) | Trace which convolution path each conv takes |
| `RLX_CUDA_CONV_T_KERNEL` | Internal | Bool | Backend(cuda) | Opt out of cuDNN for transposed conv; use the rlx kernel |
| `RLX_CUDA_DUMP_INTERMEDIATE` | Bisect | Bool | Backend(cuda) | Dump intermediate node buffers (see RLX_CUDA_DUMP_NODES_LIMIT) |
| `RLX_CUDA_DUMP_IO` | Internal | Bool | Backend(cuda) | Also dump Input/Param/Constant slots when dumping nodes |
| `RLX_CUDA_DUMP_NODES` | Bisect | Bool | Backend(cuda) | Dump output buffers per node for Metal/CPU cross-diff |
| `RLX_CUDA_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(cuda) | Cap how many nodes RLX_CUDA_DUMP_INTERMEDIATE writes |
| `RLX_CUDA_DYN_LSTM_HOST` | Internal | Bool | Backend(cuda) | Run the dynamically-quantized LSTM on the host |
| `RLX_CUDA_DYN_LSTM_TRACE` | Internal | Bool | Backend(cuda) | Trace the dynamically-quantized LSTM path |
| `RLX_CUDA_EXEC_MODE` | Bisect | String | Backend(cuda) | CUDA execution mode: stream (default) | graph | multistream:N |
| `RLX_CUDA_GDN_HOST` | Bisect | Bool | Backend(cuda) | Force GatedDeltaNet through the host D2H-CPU-H2D path |
| `RLX_CUDA_GGUF_FUSED_M1` | Bisect | Bool | Backend(cuda) | Fused m=1 GGUF decode GEMV; set 0 to keep planning scratch |
| `RLX_CUDA_GGUF_HOST` | Bisect | Bool | Backend(cuda) | Dequantize GGUF weights on the host instead of on device |
| `RLX_CUDA_IM2COL_HOST` | Bisect | Bool | Backend(cuda) | Force the host im2col fallback instead of the GPU kernel |
| `RLX_CUDA_INDEXING_FULL_ARENA` | Bisect | Bool | Backend(cuda) | Give indexing ops the full arena instead of a window |
| `RLX_CUDA_INDEXING_HOST` | Bisect | Bool | Backend(cuda) | Force ND indexing (GatherND/GatherElements/Scatter*) to the CPU host route |
| `RLX_CUDA_INDEXING_TRACE` | Internal | Bool | Backend(cuda) | Trace indexing-op dispatch |
| `RLX_CUDA_INPUT_DIAG` | Internal | Bool | Backend(cuda) | Print diagnostics for each graph input before the run |
| `RLX_CUDA_KDA_CHUNK` | Internal | String | Backend(cuda) | Enable the chunked Kimi delta-attention kernel (read at launch) |
| `RLX_CUDA_LEGACY_ATTENTION_ROW` | Internal | Bool | Backend(cuda) | Use the legacy thread-per-row attention kernel instead of warp-per-row |
| `RLX_CUDA_LOG_CONV_PATH` | Bisect | Path | Backend(cuda) | Log which convolution implementation was selected |
| `RLX_CUDA_LOG_FALLBACK` | Bisect | Bool | Backend(cuda) | Log every op that falls back off its native CUDA path |
| `RLX_CUDA_LSTM_CUBLAS_TRACE` | Internal | Bool | Backend(cuda) | Trace the cuBLAS LSTM gemm path |
| `RLX_CUDA_LSTM_CUDNN` | Internal | String | Backend(cuda) | Use cuDNN for LSTM inference; set 0 to opt out |
| `RLX_CUDA_LSTM_CUDNN_TRACE` | Internal | Bool | Backend(cuda) | Trace cuDNN LSTM descriptor setup |
| `RLX_CUDA_LSTM_DEBUG` | Internal | Bool | Backend(cuda) | Print LSTM shapes and gate values during execution |
| `RLX_CUDA_MATMUL_PRECISE` | Bisect | Bool | Backend(cuda) | Compensated k-reduction for matmul, improving precision at long K |
| `RLX_CUDA_MATMUL_PRECISE_MIN_K` | Internal | String | Backend(cuda) | K above which RLX_CUDA_MATMUL_PRECISE compensation engages |
| `RLX_CUDA_NONDET_CONV` | Public | Bool | Backend(cuda) | Allow non-deterministic cuDNN conv backward algos (atomicAdd; faster, noisy) |
| `RLX_CUDA_NO_BCAST_BINARY` | Internal | Bool | Backend(cuda) | Disable the broadcasting binary kernel; materialize instead |
| `RLX_CUDA_NO_CUBLAS` | Bisect | Bool | Backend(cuda) | Skip both cuBLAS tiers so dense GEMM runs rlx's own tiled `matmul` kernel (A/B vs the vendor library; also how the dispatch tuner reaches the tile it tunes) |
| `RLX_CUDA_NO_CUBLASLT` | Bisect | Bool | Backend(cuda) | Disable cuBLASLt and use plain cuBLAS |
| `RLX_CUDA_NO_CUDNN` | Public | Bool | Backend(cuda) | Skip cuDNN entirely (im2col / custom kernels only; silence missing-lib warning) |
| `RLX_CUDA_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(cuda) | Disable the packed BSHD attention kernel |
| `RLX_CUDA_NO_TF32` | Bisect | Bool | Backend(cuda) | Disable TF32 tensor cores (costs ~1e-4 relative error otherwise) |
| `RLX_CUDA_NO_ZERO_ARENA` | Bisect | Bool | Backend(cuda) | Skip zeroing the CUDA arena on allocation |
| `RLX_CUDA_PARITY` | Bisect | Bool | Backend(cuda) | Tighten CUDA numerics for cross-backend parity (implies no TF32) |
| `RLX_CUDA_PATH_TRACE` | Bisect | Bool | Backend(cuda) | Trace which implementation path each op takes |
| `RLX_CUDA_PINNED_IO` | Bisect | Bool | Backend(cuda) | Pinned host staging for faster D2H; on by default, set 0 to disable |
| `RLX_CUDA_PTX_CACHE` | Bisect | Path | Backend(cuda) | Directory for the compiled PTX cache (overrides XDG_CACHE_HOME) |
| `RLX_CUDA_Q4K_GEMV_COOP` | Internal | Bool | Backend(cuda) | Opt in to the cooperative block-per-row Q4_K GEMV |
| `RLX_CUDA_Q4K_GEMV_WARP` | Bisect | Bool | Backend(cuda) | Warp-per-row Q4_K decode GEMV; full occupancy at small k, unlike the block-per-row coop kernel |
| `RLX_CUDA_RNN_HOST_FALLBACK` | Internal | Bool | Backend(cuda) | Force RNN ops onto the host D2H-CPU-H2D path |
| `RLX_CUDA_SEGMENTED_CAPTURE` | Internal | Bool | Backend(cuda) | Opt in to segmented CUDA graph capture |
| `RLX_CUDA_SEGMENTED_CAPTURE_ENGAGE` | Internal | Bool | Backend(cuda) | Actually replay segmented captures (needs RLX_CUDA_SEGMENTED_CAPTURE) |
| `RLX_CUDA_SHARED_PARAMS` | Internal | Bool | Backend(cuda) | Back shared parameters with one CUDA VMM region to cut HtoD traffic |
| `RLX_CUDA_SSM_HOST_FALLBACK` | Internal | Bool | Backend(cuda) | Force SSM ops onto the host D2H-CPU-H2D path |
| `RLX_CUDA_STEP_PROFILE` | Internal | Bool | Backend(cuda) | Print per-op ms/call for each execution step |
| `RLX_CUDA_TRACE_FAB` | Bisect | Bool | Backend(cuda) | Trace fused-attention-block construction |
| `RLX_CUDA_TRANSPOSE_SHAPES` | Internal | Bool | Backend(cuda) | Log the shapes of every transpose dispatched |
| `RLX_CUDA_UNIFIED` | Internal | Bool | Backend(cuda) | Use CUDA unified memory so models larger than VRAM page over PCIe |
| `RLX_CUDA_WHOLE_GRAPH_CAPTURE` | Internal | Bool | Backend(cuda) | Capture the whole graph as one CUDA graph (needs RLX_CUDA_EXEC_MODE=graph) |
| `RLX_CUDA_WMMA` | Bisect | Bool | Backend(cuda) | Opt in to the WMMA tensor-core matmul kernel |
| `RLX_CUDNN_DIR` | Public | Path | Backend(cuda) | Directory containing libcudnn.so* to preload (pip/conda wheel failsafe) |
| `RLX_TUNE_PREFILTER` | Internal | String | Backend(cuda) | Narrow the tuner's candidate tiles to the N cheapest by cost model |

## debug

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ALLOW_THROTTLE` | Public | Bool | Tooling | Skip thermal gate for one-off benches (prefer `just throttle`) |
| `RLX_CHECK` | Bisect | String | Runtime | Static-check breadth: off | all | strict |
| `RLX_DBG_BINF` | Bisect | Bool | Runtime | Print binary-op shape inference during ONNX import |
| `RLX_DBG_CONV` | Bisect | Bool | Runtime | Print convolution lowering details |
| `RLX_DBG_CUSTOM` | Public | Bool | Runtime | Log host custom-op staging (onnx.* dtype bridge) on GPU backends |
| `RLX_DBG_SHAPES` | Bisect | Bool | Runtime | Print inferred shapes during ONNX import |
| `RLX_DBG_STEP` | Bisect | Bool | Runtime | Print each wgpu execution step as it runs |
| `RLX_DEBUG_NANS` | Public | Enum | Runtime | Runtime NaN/Inf localizer (1 or abort) |
| `RLX_DISPATCH_REPORT` | Public | Bool | Compile | Print legalize/dispatch report during compile (1 = on) |
| `RLX_ENV_DEPRECATIONS` | Public | Bool | Runtime | Emit one-shot messages when deprecated RLX_* aliases are used |
| `RLX_REQUIRE_DEVICE` | Public | Bool | Tooling | Make a missing GPU a test FAILURE instead of a silent skip (rig runs) |
| `RLX_VERBOSE` | Public | Bool | Runtime | Extra runtime logging |

## device

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_BENCHMARK_PICK` | Public | U64 | Device | Micro-benchmark N runs to pick the fastest device (needs inputs) |
| `RLX_DEVICE` | Public | String | Device | Default device hint for resolved runs (cpu, metal, mlx, cuda, gpu, …) |
| `RLX_DEVICES` | Public | String | Device | Allow-list of devices for DevicePolicy::from_env |
| `RLX_DEVICE_CHAIN` | Public | String | Device | Fallback order when a preferred device fails (e.g. cuda,gpu,cpu) |

## egpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_EGPU_CONFIG` | Internal | String | Backend(egpu) | Path to the eGPU config file (overrides XDG_CONFIG_HOME/rlx/egpu.conf) |
| `RLX_EGPU_SERVICE` | Internal | String | Backend(egpu) | eGPU service endpoint used instead of the assumed default |
| `RLX_FW_DIR` | Internal | String | Backend(egpu) | Where firmware is written (overrides XDG_CACHE_HOME/rlx/firmware) |
| `RLX_HIPCC` | Internal | String | Backend(egpu) | Path to the hipcc compiler (default hipcc) |
| `RLX_NVRTC` | Internal | String | Backend(egpu) | Path to the NVRTC library for ahead-of-time baking |
| `RLX_PTXAS` | Internal | String | Backend(egpu) | Path to ptxas for ahead-of-time baking (default ptxas) |

## fft

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_FFT_CPU_PARALLEL` | Bisect | Bool | Tooling | Rayon-parallelize the CPU FFT batch loop; set 0 to serialize |
| `RLX_FFT_CUFFT` | Bisect | Bool | Tooling | Use cuFFT; set 0 to fall back to rlx's own FFT kernels |
| `RLX_FFT_FORCE_MIXED` | Bisect | Bool | Tooling | Force the precompiled mixed-radix FFT instead of the generated kernel |
| `RLX_FFT_FUSE_DEBUG` | Bisect | Bool | Tooling | Print FFT fusion decisions |
| `RLX_FFT_FUSE_REAL` | Bisect | Bool | Tooling | Fuse the real-to-complex FFT prologue; set 0 to disable |
| `RLX_FFT_GEN` | Bisect | Bool | Tooling | Use the per-size generated FFT kernel; set 0 for the mixed-radix fallback |
| `RLX_FFT_MULTIROW` | Bisect | Bool | Tooling | Opt in to the multi-row small-n FFT path (default off) |
| `RLX_FFT_NATIVE` | Bisect | Bool | Tooling | Native Stockham FFT path; set 0 to disable it |
| `RLX_FFT_RADIX` | Bisect | Bool | Tooling | Cap the FFT radix for an A/B: 2 | 4 (default 8) |
| `RLX_FFT_RADIX4` | Bisect | Bool | Tooling | Radix-4 CPU FFT for pure powers of four; set 0 to disable |

## gpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_FFT_FAST` | Public | Bool | Runtime | Enable native on-chip GPU FFT when compiled (0 disables) |
| `RLX_INDEXING_FULL_ARENA` | Public | Bool | Runtime | Force full-arena mirror for indexing host-fallback (bisect; slow on discrete GPUs) |
| `RLX_STATIC_WEIGHT_PACK` | Public | Bool | Compile | Materialise step-invariant weight packs once and skip after; `=0` opts out |
| `RLX_WGPU_LSTM_WIDE` | Bisect | Bool | Backend(gpu) | Re-enable the wide-hidden native WGSL LSTM (debugging the Apple exp defect) |

## gpu-host

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_HD_PROFILE` | Internal | Bool | Backend(gpu-host) | Accumulate nanoseconds spent in each host-delegate phase |

## metal

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_DISABLE_METAL_DEQUANT_GPU` | Deprecated → `RLX_METAL_DEQUANT_GPU_DISABLE` | Bool | Backend(metal) | Deprecated alias of `RLX_METAL_DEQUANT_GPU_DISABLE` |
| `RLX_DISABLE_MPSGRAPH` | Public | Bool | Compile | Force Metal thunk path instead of MPSGraph regions |
| `RLX_DISABLE_MPSGRAPH_EXECUTABLE` | Bisect | Bool | Backend(metal) | Disable the precompiled MPSGraphExecutable path |
| `RLX_DISABLE_MPSGRAPH_HYBRID` | Internal | Bool | Backend(metal) | Disable hybrid MPSGraph/native segmentation |
| `RLX_DUMP_SCHED_DECODE` | Internal | Bool | Backend(metal) | Dump the encode schedule for decode steps only |
| `RLX_HF_MLX` | Internal | String | Backend(metal) | Allow downloading MLX weights from HuggingFace on a cold cache |
| `RLX_METAL_ARENA_DIAG` | Internal | Bool | Backend(metal) | Report Metal arena slot assignment and reuse |
| `RLX_METAL_ATTN_BWD_FUSE` | Internal | String | Backend(metal) | Fuse Metal attention backward; set 0 to opt out |
| `RLX_METAL_ATTN_BWD_FUSED` | Internal | Bool | Backend(metal) | Use the fused Metal attention-backward kernel |
| `RLX_METAL_ATTN_BWD_FUSED_ZEROONLY` | Internal | Bool | Backend(metal) | Run only the zeroing half of the fused attention backward (isolation aid) |
| `RLX_METAL_ATTN_BWD_GPU` | Bisect | Bool | Backend(metal) | Run attention backward on the GPU; set 0 to host it |
| `RLX_METAL_ATTN_TRACE` | Bisect | Bool | Backend(metal) | Trace Metal attention kernel selection |
| `RLX_METAL_CMDBUF_TRACE` | Bisect | Bool | Backend(metal) | Trace Metal command-buffer status transitions |
| `RLX_METAL_CONCAT_HOST` | Bisect | Bool | Backend(metal) | Opt in to the host-staged Metal Concat fallback |
| `RLX_METAL_CONCAT_MULTI` | Bisect | Bool | Backend(metal) | Disable the multi-input Concat kernel; encode one input at a time |
| `RLX_METAL_CONCURRENT` | Bisect | Bool | Backend(metal) | Open compute encoders in Concurrent dispatch so independent decode dispatches overlap |
| `RLX_METAL_CONCURRENT_BARRIER_FRESH` | Bisect | Bool | Backend(metal) | Also barrier when no encoder is open yet, by opening it first (a Concurrent encoder is not implicitly ordered after the previous one) |
| `RLX_METAL_CONCURRENT_FENCE_ALL` | Bisect | Bool | Backend(metal) | Fence every concurrent dispatch instead of only the data-dependent ones |
| `RLX_METAL_CONCURRENT_IDX_HI` | Bisect | U64 | Backend(metal) | Upper bound (exclusive) of the thunk index range allowed to use Concurrent dispatch |
| `RLX_METAL_CONCURRENT_IDX_LO` | Bisect | U64 | Backend(metal) | Lower bound (inclusive) of the thunk index range allowed to use Concurrent dispatch |
| `RLX_METAL_CONCURRENT_NOBARRIER` | Internal | Bool | Backend(metal) | Drop barriers between concurrent Metal dispatches |
| `RLX_METAL_CONCURRENT_OPAQUE` | Bisect | String | Backend(metal) | Comma-separated thunk kinds to force-fence, to bisect which op is sensitive to the encoder dispatch type |
| `RLX_METAL_CONCURRENT_SPLIT_ENC` | Bisect | Bool | Backend(metal) | Give every thunk its own encoder so the dispatch type can be chosen per thunk (bisection aid) |
| `RLX_METAL_CONCURRENT_STATS` | Internal | Bool | Backend(metal) | Report encoders opened vs thunks dispatched, to tell whether Concurrent can overlap anything |
| `RLX_METAL_CONV3D` | Bisect | Enum | Backend(metal) | Force a Conv3D kernel: naive (scalar) or gemm (skip the c-tiled variant) |
| `RLX_METAL_CONV3D_FOLDS` | Bisect | Bool | Backend(metal) | Enable the Conv3D concat/upsample/leaky thunk folds (off: they regress arena liveness) |
| `RLX_METAL_CONV3D_MIN_M` | Bisect | U64 | Backend(metal) | Minimum Conv3D output-position count before the tiled implicit-GEMM kernel is used (default 8) |
| `RLX_METAL_CONV_BWD_IMPLICIT` | Bisect | Bool | Backend(metal) | Use implicit GEMM for conv backward-weight; set 0 for explicit |
| `RLX_METAL_DEBUG` | Bisect | Bool | Backend(metal) | Extra Metal validity checks and diagnostics |
| `RLX_METAL_DEQUANT_FORCE_GEMV` | Internal | Bool | Backend(metal) | Force the GEMV dequant kernel even when m > 1 |
| `RLX_METAL_DEQUANT_GPU_DISABLE` | Public | Bool | Backend(metal) | Disable Metal GPU GGUF/MLX dequant (host / legacy path) aliases: `RLX_DISABLE_METAL_DEQUANT_GPU` |
| `RLX_METAL_DEQUANT_MATMUL_LEGACY` | Public | Bool | Backend(metal) | Use pre-fused dequant+matmul path (materializes weights) |
| `RLX_METAL_DISABLE_NARROW_ROPE_FUSE` | Bisect | Bool | Backend(metal) | Disable fusing Narrow into RoPE |
| `RLX_METAL_DUMP_BYTES` | Internal | Bool | Backend(metal) | Sum bytes moved per op type (cache excluded) |
| `RLX_METAL_DUMP_MSL` | Internal | String | Backend(metal) | Write every generated MSL source to this path |
| `RLX_METAL_DUMP_NODES` | Bisect | Bool | Backend(metal) | Dump per-node output buffers on Metal |
| `RLX_METAL_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(metal) | Cap how many nodes the Metal dump writes (default 4000) |
| `RLX_METAL_DW_SUM` | Internal | Bool | Backend(metal) | Correctly-rounded near-f64 dW reduction, at ~3-4x the flops |
| `RLX_METAL_EXPAND_CONV3D_BIAS` | Bisect | Bool | Backend(metal) | Let a rank-5 no-activation FusedConvBiasAct keep its expanded bias instead of declining the fusion |
| `RLX_METAL_EXTERNALIZE_QUANT` | Bisect | Bool | Backend(metal) | Keep quantized weights external rather than baking them in |
| `RLX_METAL_EXT_TRACE` | Bisect | Bool | Backend(metal) | Trace Metal op-extension lowering |
| `RLX_METAL_FA` | Bisect | Bool | Backend(metal) | Opt in to the Metal flash-attention kernel |
| `RLX_METAL_FFT_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Run FFT on the host instead of the Metal kernel |
| `RLX_METAL_FOLD_AUDIT` | Internal | Bool | Backend(metal) | Assert no two arena-sharing nodes have overlapping FOLD-ADJUSTED live ranges |
| `RLX_METAL_FORCE_INLINE_PARAMS` | Bisect | Bool | Backend(metal) | Inline parameters into the encoder rather than binding buffers |
| `RLX_METAL_FORCE_PIN_OUTPUT_ANCESTORS` | Bisect | Bool | Backend(metal) | Force output-ancestor buffers to stay pinned |
| `RLX_METAL_FORCE_UNPIN_OUTPUT_ANCESTORS` | Bisect | Bool | Backend(metal) | Force output-ancestor buffers to be unpinned |
| `RLX_METAL_FUSE_DECODE` | Bisect | Bool | Backend(metal) | Fuse the decode MLP on Metal; on by default, set 0 to opt out |
| `RLX_METAL_FUSE_DECODE_GELU` | Bisect | Bool | Backend(metal) | Combined GeGLU decode fusion; set 0 to opt out |
| `RLX_METAL_FUSE_DECODE_LOG` | Bisect | Bool | Backend(metal) | Log which decode fusions fired |
| `RLX_METAL_FUSE_DEPTHWISE` | Bisect | Bool | Backend(metal) | Fuse depthwise conv1d in BSC layout; set 0 to disable |
| `RLX_METAL_FUSE_GDN_NORM` | Bisect | Bool | Backend(metal) | Fuse the GatedDeltaNet gated norm; set 0 to disable |
| `RLX_METAL_FUSE_L2NORM` | Bisect | BoolOr | Backend(metal) | Fuse the ggml L2_NORM op chain into one L2NormLastDim dispatch (default on; =0 disables) |
| `RLX_METAL_FUSE_RESIDUAL_DUAL` | Internal | Bool | Backend(metal) | Opt in to the dual-output fused residual |
| `RLX_METAL_FUSE_RESIDUAL_RMS` | Bisect | Bool | Backend(metal) | Fuse residual add into RmsNorm; on by default, set 0 to opt out |
| `RLX_METAL_G8_0_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused G8_0 decode GEMV and dequantize to a scratch buffer first |
| `RLX_METAL_G8_0_SCALAR` | Bisect | Bool | Backend(metal) | Fall back to the scalar byte inner loop in the G8_0 GEMV (A/B only) |
| `RLX_METAL_GDN_CPU` | Bisect | Bool | Backend(metal) | Run GatedDeltaNet on the CPU instead of Metal |
| `RLX_METAL_GDN_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Force the host fallback for GatedDeltaNet on Metal |
| `RLX_METAL_GDN_SG_DISABLE` | Internal | Bool | Backend(metal) | Disable the simdgroup GatedDeltaNet kernel (state size 128) |
| `RLX_METAL_GEMV_KPART` | Internal | String | Backend(metal) | K-partition small-N GEMVs across threadgroups (ceiling probe) |
| `RLX_METAL_GEMV_SPLITK` | Internal | String | Backend(metal) | Split-K GEMV for m=1, n>=64; set 0 to disable |
| `RLX_METAL_GROUPED_GEMV_DISABLE` | Internal | Bool | Backend(metal) | Disable the grouped GEMV kernel and fall back to MPS |
| `RLX_METAL_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Run unsupported Metal ops on the host instead of failing |
| `RLX_METAL_HOST_SLICE` | Bisect | Bool | Backend(metal) | Run Slice on the host instead of a Metal kernel |
| `RLX_METAL_HYBRID_BIG_ARENA` | Bisect | Bool | Backend(metal) | Allow the large hybrid arena instead of all-thunks (can OOM) |
| `RLX_METAL_IQ1M_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ1_M matmul kernel |
| `RLX_METAL_IQ1S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ1_S matmul kernel |
| `RLX_METAL_IQ2S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ2_S matmul kernel |
| `RLX_METAL_IQ2XS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ2_XS matmul kernel |
| `RLX_METAL_IQ2XXS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ2_XXS matmul kernel |
| `RLX_METAL_IQ3S_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ3_S matmul kernel |
| `RLX_METAL_IQ3XXS_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ3_XXS matmul kernel |
| `RLX_METAL_IQ4NL_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused IQ4_NL matmul kernel |
| `RLX_METAL_KERNEL_STATS` | Internal | Bool | Backend(metal) | Report per-kernel dispatch statistics |
| `RLX_METAL_LN_GAMMA_SIMD` | Internal | Bool | Backend(metal) | Use the simdgroup reduction for LayerNorm gamma backward |
| `RLX_METAL_LSTM_CPU` | Bisect | Bool | Backend(metal) | Run LSTM on the CPU instead of Metal |
| `RLX_METAL_LSTM_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Run LSTM on the host instead of Metal |
| `RLX_METAL_LSTM_NATIVE_WIDE` | Internal | Bool | Backend(metal) | Allow the native LSTM kernel at wide hidden sizes |
| `RLX_METAL_MATMUL_TRANSPOSE_FOLD` | Internal | String | Backend(metal) | Opt in to folding transposes into Metal matmul (default off) |
| `RLX_METAL_MATMUL_TRANSPOSE_FOLD_WEIGHTS` | Internal | String | Backend(metal) | Fold transposes into weight operands of Metal matmul |
| `RLX_METAL_MLP_SG` | Internal | String | Backend(metal) | Simdgroup MLP kernel; set 0 to disable |
| `RLX_METAL_MPSGRAPH_BIG_ARENA` | Bisect | Bool | Backend(metal) | Allow MPSGraph to use the large arena |
| `RLX_METAL_MPS_PROFILE` | Bisect | Bool | Backend(metal) | Time MPSGraph and hybrid dispatches |
| `RLX_METAL_MPS_SDPA` | Bisect | Bool | Backend(metal) | Route SDPA through MPSGraph |
| `RLX_METAL_NARROW_BATCH` | Bisect | Bool | Backend(metal) | Disable batching of adjacent Narrow ops |
| `RLX_METAL_NO_AUTO_WIDE_GEMM` | Internal | Bool | Backend(metal) | Disable automatic selection of the wide GEMM path |
| `RLX_METAL_NO_CONV3D_CONCAT_FOLD` | Bisect | Bool | Backend(metal) | Disable folding a channel-wise Concat into the Conv3D gather |
| `RLX_METAL_NO_CONV3D_FOLDS` | Bisect | Bool | Backend(metal) | Disable every Conv3D epilogue thunk fold at once (bisect the fold path as a whole) |
| `RLX_METAL_NO_CONV3D_LEAKY_FOLD` | Bisect | Bool | Backend(metal) | Disable folding LeakyReLU into the Conv3D kernel |
| `RLX_METAL_NO_CONV3D_UPSAMPLE_FOLD` | Bisect | Bool | Backend(metal) | Disable folding a nearest-neighbour Upsample into the Conv3D gather |
| `RLX_METAL_NO_FUSION` | Bisect | Bool | Backend(metal) | Skip every Metal pattern fusion |
| `RLX_METAL_NO_SGEMM64` | Bisect | Bool | Backend(metal) | Disable the default 64x64-tile Simd64 sgemm (fall back to Simd4x4) for aligned tall/short-K matmuls. |
| `RLX_METAL_NO_SHARE` | Bisect | Bool | Backend(metal) | Disable sharing parameter buffers between compiled graphs |
| `RLX_METAL_ONNX_QMATMUL_GPU` | Bisect | Bool | Backend(metal) | Run ONNX quantized matmul on the GPU; set 0 to host it |
| `RLX_METAL_ONNX_QMATMUL_MIN_FLOPS` | Bisect | U64 | Backend(metal) | FLOP threshold above which ONNX qmatmul goes to the GPU |
| `RLX_METAL_OUTPUT_TRACE` | Bisect | Bool | Backend(metal) | Trace each graph output read back from Metal |
| `RLX_METAL_PARAMS` | Public | String | Backend(metal) | Apple kernel tuning spec, `k=v,...` (stages/sync/precision/tile/encode) |
| `RLX_METAL_PARAM_DIAG` | Internal | String | Backend(metal) | Diagnose parameter uploads larger than 1 MB |
| `RLX_METAL_PIPELINE_CACHE` | Bisect | Path | Backend(metal) | On-disk Metal pipeline-state cache directory |
| `RLX_METAL_PRECISE` | Bisect | Bool | Backend(metal) | Force the scalar fp32 Metal path instead of reduced-precision tensor units |
| `RLX_METAL_PREFILL_COUNTERS` | Internal | Bool | Backend(metal) | Report Metal prefill dispatch counters |
| `RLX_METAL_PREFILL_FA` | Internal | String | Backend(metal) | Flash-attention prefill kernel; set 0 to disable |
| `RLX_METAL_PREFILL_FA_MMA` | Internal | String | Backend(metal) | Opt in to the MMA flash-attention prefill kernel |
| `RLX_METAL_PREFILL_HD256` | Internal | String | Backend(metal) | Head-dim-256 prefill kernel; set 0 to disable |
| `RLX_METAL_PREFILL_TRACE` | Internal | Bool | Backend(metal) | Trace Metal prefill matmul dispatches (m > 1) |
| `RLX_METAL_Q1_0_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q1_0 matmul kernel |
| `RLX_METAL_Q1_0_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q1_0 kernel |
| `RLX_METAL_Q1_DUAL_DISABLE` | Bisect | Bool | Backend(metal) | Disable the dual-output Q1 decode kernel |
| `RLX_METAL_Q2K_FUSED_DISABLE` | Internal | Bool | Backend(metal) | Disable the fused Q2_K matmul kernel |
| `RLX_METAL_Q2_0_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q2_0 matmul kernel |
| `RLX_METAL_Q2_0_SCALAR` | Bisect | Bool | Backend(metal) | Fall back to the scalar byte inner loop in the Q2_0 GEMVs instead of the shared q2_0_dot16 (A/B only) |
| `RLX_METAL_Q2_0_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q2_0 kernel |
| `RLX_METAL_Q2_DUAL_DISABLE` | Bisect | Bool | Backend(metal) | Disable the dual-output Q2 decode kernel |
| `RLX_METAL_Q3K_FUSED_DISABLE` | Internal | Bool | Backend(metal) | Disable the fused Q3_K matmul kernel |
| `RLX_METAL_Q3K_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q3_K decode GEMV, using the one-thread-per-row kernel |
| `RLX_METAL_Q40_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q4_0 matmul kernel |
| `RLX_METAL_Q40_SG_DISABLE` | Internal | Bool | Backend(metal) | Disable the simdgroup Q4_0 kernels; use one thread per row |
| `RLX_METAL_Q41_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q4_1 matmul kernel |
| `RLX_METAL_Q41_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q4_1 decode GEMV, using the one-thread-per-row kernel |
| `RLX_METAL_Q4K_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q4_K matmul kernel |
| `RLX_METAL_Q4K_GEMM_DISABLE` | Bisect | Bool | Backend(metal) | Disable the Q4_K GEMM kernel; use GEMV |
| `RLX_METAL_Q4K_GEMM_MAX_M` | Internal | String | Backend(metal) | m above which Q4_K routes to GEMM instead of GEMV (default 32) |
| `RLX_METAL_Q4K_GEMM_XS_DISABLE` | Internal | Bool | Backend(metal) | Disable the extra-small-tile Q4_K GEMM kernel |
| `RLX_METAL_Q4K_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q4_K kernel |
| `RLX_METAL_Q5K_FUSED_DISABLE` | Internal | Bool | Backend(metal) | Disable the fused Q5_K matmul kernel |
| `RLX_METAL_Q5K_GEMM_DISABLE` | Internal | Bool | Backend(metal) | Disable the Q5_K GEMM kernel; use GEMV |
| `RLX_METAL_Q5K_SG_DISABLE` | Bisect | Bool | Backend(metal) | Disable the simdgroup Q5_K decode GEMV, using the one-thread-per-row kernel |
| `RLX_METAL_Q6K_FUSED_DISABLE` | Internal | Bool | Backend(metal) | Disable fused Q6_K matmul, falling back to dequant-to-scratch |
| `RLX_METAL_Q6K_GEMM_DISABLE` | Bisect | Bool | Backend(metal) | Disable the Q6_K GEMM kernel; use GEMV |
| `RLX_METAL_Q6K_GEMM_XS_DISABLE` | Internal | Bool | Backend(metal) | Disable the extra-small-tile Q6_K GEMM kernel |
| `RLX_METAL_Q6K_SG_DISABLE` | Internal | Bool | Backend(metal) | Disable the simdgroup Q6_K/Q8_0 decode kernels |
| `RLX_METAL_Q80_FUSED_DISABLE` | Bisect | Bool | Backend(metal) | Disable the fused Q8_0 matmul kernel |
| `RLX_METAL_Q8_0_GEMM_DISABLE` | Internal | Bool | Backend(metal) | Disable the Q8_0 GEMM kernel; use GEMV |
| `RLX_METAL_Q8_0_SG_DISABLE` | Internal | Bool | Backend(metal) | Disable the simdgroup Q8_0 kernel |
| `RLX_METAL_RB_DEBUG` | Internal | Bool | Backend(metal) | Log register-blocked GEMM selection |
| `RLX_METAL_RB_FUSED_GEMM` | Internal | Bool | Backend(metal) | Use the register-blocked fused GEMM kernel |
| `RLX_METAL_REDUCE_SIMD` | Internal | Bool | Backend(metal) | Use the simdgroup reduction kernel for axis reductions |
| `RLX_METAL_RNN_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Run RNN ops on the host instead of Metal |
| `RLX_METAL_SAMPLE_HOST` | Bisect | Bool | Backend(metal) | Run token sampling on the host instead of on device |
| `RLX_METAL_SDPA_DECODE_M1` | Bisect | Bool | Backend(metal) | Dedicated m=1 SDPA decode kernel; set 0 to disable |
| `RLX_METAL_SDPA_FA2` | Internal | Bool | Backend(metal) | Use the FlashAttention-2 style SDPA kernel |
| `RLX_METAL_SDPA_H16` | Internal | Bool | Backend(metal) | Opt in to the f16-scores SDPA kernel (implies SIMD softmax) |
| `RLX_METAL_SDPA_HDSPLIT` | Internal | String | Backend(metal) | Opt in to the head-dim-split SDPA variant |
| `RLX_METAL_SDPA_MMA` | Internal | Bool | Backend(metal) | Use the MMA (simdgroup matrix) SDPA kernel |
| `RLX_METAL_SDPA_OCCPAD` | Internal | Bool | Backend(metal) | Pad SDPA dispatches for occupancy at seq > 1 |
| `RLX_METAL_SDPA_SIMD` | Internal | Bool | Backend(metal) | Use the simdgroup softmax SDPA kernel (f32 only) |
| `RLX_METAL_SDPA_SPLITK` | Internal | Bool | Backend(metal) | Split SDPA along K across threadgroups |
| `RLX_METAL_SGEMM_MPS` | Bisect | Bool | Backend(metal) | Force the MPS sgemm path at shapes the cost model would route elsewhere |
| `RLX_METAL_SGEMM_PRECISE` | Bisect | Bool | Backend(metal) | Force the scalar fp32 sgemm path for precision-critical work |
| `RLX_METAL_SGEMM_SPLITK` | Bisect | Bool | Backend(metal) | Enable the split-K 64x64 sgemm for fat-K/small-MN shapes (dW=xT.dq); atomic-accumulate into a pre-zeroed C. |
| `RLX_METAL_SGEMM_VARIANT` | Bisect | Enum | Backend(metal) | Pin one Metal sgemm variant for an A/B (refused where ineligible) |
| `RLX_METAL_SOFTMAX_TRACE` | Bisect | Bool | Backend(metal) | Trace softmax kernel selection |
| `RLX_METAL_SSM_CPU` | Bisect | Bool | Backend(metal) | Run the SSM recurrence on the CPU instead of Metal |
| `RLX_METAL_SSM_HOST_FALLBACK` | Bisect | Bool | Backend(metal) | Run the SSM recurrence on the host instead of Metal |
| `RLX_METAL_SYNTH_MPS_DISABLE` | Bisect | Bool | Backend(metal) | SynthMatMul m>8 prefill uses the fused kernel instead of reconstruct→MPS (A/B) |
| `RLX_METAL_SYNTH_RECON_F16` | Public | Bool | Backend(metal) | SynthMatMul m>8 prefill reconstructs the weight in f16 → MPS hgemm (~1.3×, 2× smaller scratch, relaxed precision) |
| `RLX_METAL_SYNTH_TILED` | Bisect | Bool | Backend(metal) | SynthMatMul m>8 uses the threadgroup-tiled fused kernel (zero-scratch/capturable; slower than recon→MPS) |
| `RLX_METAL_SYNTH_TILED_F16` | Bisect | Bool | Backend(metal) | With RLX_METAL_SYNTH_TILED, use the f16 (simdgroup_half8x8) tiled kernel variant |
| `RLX_METAL_THUNK_PROFILE` | Bisect | Bool | Backend(metal) | Time each Metal schedule thunk individually |
| `RLX_METAL_TRACE` | Bisect | Bool | Backend(metal) | Log every Metal dispatch as it is encoded |
| `RLX_METAL_TRACE_FAB` | Bisect | Bool | Backend(metal) | Trace fused-attention-block construction on Metal |
| `RLX_METAL_UNFUSE_REGIONS` | Bisect | Bool | Backend(metal) | Unfuse ElementwiseRegion nodes before Metal lowering |
| `RLX_METAL_UNPIN_ALL` | Internal | Bool | Backend(metal) | Do not pin output-ancestor buffers |
| `RLX_METAL_VALIDATE_BINDINGS` | Internal | Bool | Backend(metal) | Check each Metal dispatch's buffer bindings against the kernel signature |
| `RLX_METAL_W8A8_ATTN` | Internal | Bool | Backend(metal) | W8A8 decode attention: int8 Q-K integer dot with int8 V |
| `RLX_METAL_W8A8_BLOCK` | Internal | Bool | Backend(metal) | Use block-wise W8A8 quantization scales |
| `RLX_METAL_W8A8_INCR` | Internal | String | Backend(metal) | Token count for the incremental-quantize timing probe |
| `RLX_METAL_W8A8_KMODE` | Internal | String | Backend(metal) | W8A8 K source: i8 (default) or f32 for diagnostic isolation |
| `RLX_METAL_W8A8_QMODE` | Internal | String | Backend(metal) | W8A8 Q source: i8 integer dot (default) or f32 |
| `RLX_METAL_W8A8_VMODE` | Internal | String | Backend(metal) | W8A8 V source: i8 (default) or f32 to isolate K error |
| `RLX_METAL_WIDE_GEMM` | Internal | Bool | Backend(metal) | Force the wide GEMM path wherever M makes it eligible |
| `RLX_MPSGRAPH_FORCE` | Bisect | Bool | Backend(metal) | Force this op through MPSGraph regardless of the cost model |
| `RLX_MPSGRAPH_MIN_FLOPS` | Bisect | U64 | Backend(metal) | FLOP threshold above which MPSGraph is used |
| `RLX_MPSGRAPH_NO_SYNC_COMPILE` | Internal | Bool | Backend(metal) | Restore the old nil-descriptor asynchronous MPSGraph compile |
| `RLX_MPSGRAPH_PARAM_CONST` | Bisect | Bool | Backend(metal) | Bake parameters into MPSGraph as constants (opt-in) |
| `RLX_MPSGRAPH_PARAM_CONST_CAP` | Bisect | U64 | Backend(metal) | Byte cap on parameters frozen into MPSGraph constants |
| `RLX_MPSGRAPH_TRACE` | Bisect | Bool | Backend(metal) | Log when the MPSGraph path fires |
| `RLX_MPS_ALIGN_DEBUG` | Bisect | U64 | Backend(metal) | Report MPS buffer alignment decisions |
| `RLX_MPS_FP16` | Bisect | Bool | Backend(metal) | Cast both inputs of a 2-input MPS op to fp16 |
| `RLX_MPS_LOWP_ACT` | Internal | String | Backend(metal) | Low-precision MPS activations: less bandwidth, some accuracy cost |
| `RLX_MPS_THRESHOLD_FLOP` | Bisect | U64 | Backend(metal) | FLOP cutoff above which matmul routes to MPS |
| `RLX_QWEN3_BAKE_WEIGHTS` | Deprecated → `RLX_STATIC_WEIGHT_PACK` | BoolOr | Backend(metal) | Legacy Metal-only alias for RLX_STATIC_WEIGHT_PACK (default on; `=0` opts out) |
| `RLX_SDPA_CASE` | Internal | String | Backend(metal) | Case selector for the SDPA canonicalization crash test |

## misc

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_AD_FUSED_MMBA_VJP` | Internal | String | Tooling | Opt in to the fused matmul+bias+activation VJP in autodiff |
| `RLX_ARCH` | Internal | Bool | Tooling | Trainer architecture to build: mlp | cnn |
| `RLX_ATTN_DEBUG` | Bisect | Bool | Tooling | Print CPU attention thunk selection and shapes |
| `RLX_BAKE_OUT` | Internal | String | Tooling | Output path for the baked .rlx artifact |
| `RLX_BAKE_PASSWORD` | Internal | String | Tooling | Password for an encrypted bake (never passed on the CLI) |
| `RLX_BENCH_CSV` | Internal | Bool | Tooling | Append each bench row to this CSV file |
| `RLX_BENCH_DISPATCH_ONLY` | Internal | String | Tooling | Bench dispatch overhead only, skipping the compute |
| `RLX_BENCH_ITERS` | Internal | Bool | Tooling | Iteration count for the op benchmarks |
| `RLX_CONVINT_FLOAT_ACT` | Internal | String | Tooling | Route ConvInteger through the pre-quant float activation |
| `RLX_CPU_DUMP_FLAT` | Bisect | Bool | Tooling | Also print this flat element index when dumping a diverging node |
| `RLX_CPU_DUMP_NODES` | Bisect | Bool | Tooling | Print per-node max|x| and nonzero counts during execution |
| `RLX_CPU_DUMP_NODES_LIMIT` | Bisect | U64 | Tooling | Cap how many nodes the CPU dump writes (default 2000) |
| `RLX_DECODE_BUCKET_PEAK_BYTES` | Bisect | U64 | Tooling | Override the estimated compile peak bytes for bucketed decode |
| `RLX_DECODE_BUCKET_RESIDENT_BYTES` | Bisect | U64 | Tooling | Override the assumed resident bytes per decode bucket |
| `RLX_DECODE_ONESHOT_PEAK_BYTES` | Bisect | U64 | Tooling | Override the estimated compile peak bytes for one-shot decode |
| `RLX_DECOMPOSE_FUSION_REGIONS` | Bisect | Bool | Tooling | Decompose fusion regions back to primitives before lowering |
| `RLX_DECOMPOSE_SPLINE_BWD` | Bisect | Bool | Tooling | Force the KAN spline VJP to the decomposed exp/mul/reduce chain instead of the fused SplineActivationBackwardX/Coeff ops (debug/parity). |
| `RLX_DEQUANT_CACHE` | Bisect | Path | Tooling | Cache dequantized weights; set 0 to disable caching entirely |
| `RLX_DIM_DBG` | Internal | String | Tooling | Print dimension resolution during ONNX runtime emission |
| `RLX_DIRECT_CONV` | Bisect | Bool | Tooling | Opt in to the direct (non-im2col) convolution path |
| `RLX_DISABLE_CSE` | Internal | String | Runtime | Disable common-subexpression elimination for an A/B |
| `RLX_DISABLE_MPS` | Bisect | Bool | Tooling | Disable MPSMatrixMultiplication; use rlx's own MSL sgemm |
| `RLX_DISABLE_NOMIC_FUSION` | Bisect | Bool | Tooling | Disable the Nomic-embed fusion pattern for an A/B |
| `RLX_DISABLE_TRANSPOSE_ELISION` | Internal | Bool | Tooling | Keep redundant transposes instead of eliding them (A/B + safety valve) |
| `RLX_DUMP_SCHED` | Bisect | Bool | Tooling | Dump the encode schedule |
| `RLX_ENABLE_FUSE_TRANSFORMER_LAYER` | Bisect | Bool | Tooling | Enable whole-transformer-layer fusion |
| `RLX_F32_DUMP` | Bisect | Bool | Tooling | Directory the trainer writes f32 weight dumps to |
| `RLX_FK_BATCH_SINGLE_KERNEL` | Bisect | Bool | Tooling | Emit one fused kernel for the batched FK path |
| `RLX_FORCE_DEVICE` | Bisect | Bool | Tooling | Hard-pin the backend, overriding the cost model's device choice |
| `RLX_FUSED_GPU_CONIC_SCANLINE` | Bisect | Bool | Compile | Experimental GPU conic scanline splat rasterizer |
| `RLX_GGUF_MATMUL_LEGACY` | Bisect | Bool | Tooling | Use the legacy GGUF matmul dispatch instead of gguf_matmul_bt_dispatch |
| `RLX_GGUF_TRACE` | Bisect | Bool | Tooling | Trace GGUF header and tensor parsing |
| `RLX_GPU_HANDLE_HOST_MIRROR` | Bisect | Bool | Tooling | Mirror GPU handle outputs to the host on every read |
| `RLX_GPU_TUNING_CACHE` | Public | Path | Tooling | Path to the persisted GPU dispatch tuning cache (measured kernel-variant + tile winners per arch/op/shape-bucket); unset = under $XDG_CACHE_HOME |
| `RLX_HF_CACHE` | Internal | String | Tooling | HuggingFace cache directory (default $HOME/.cache/rlx/hf) |
| `RLX_HF_MLX_REPO` | Internal | String | Tooling | HuggingFace repo id to pull MLX weights from |
| `RLX_HF_REVISION` | Internal | String | Tooling | HuggingFace revision to fetch (default main) |
| `RLX_HIGHER_ORDER_NO_FUSE` | Bisect | Bool | Tooling | Disable elementwise fusion for higher-order derivatives |
| `RLX_HYBRID_K` | Bisect | Bool | Tooling | Override the k dimension in the hybrid dequant-matmul parity test |
| `RLX_HYBRID_M` | Bisect | Bool | Tooling | Override the hybrid-split m dimension (default 96) |
| `RLX_ICB_TRACE` | Bisect | Bool | Tooling | Trace indirect-command-buffer construction |
| `RLX_IMP_DBG` | Bisect | Bool | Tooling | Print per-op decisions during ONNX import |
| `RLX_IQ_TEST_DIR` | Bisect | Path | Tooling | Directory holding the IQ/TQ test GGUFs |
| `RLX_IR_DUMP` | Bisect | Bool | Tooling | Write a full pipeline IR dump to this path prefix or directory |
| `RLX_KEEP_ELEMENTWISE_REGIONS` | Bisect | Bool | Tooling | Keep fused ElementwiseRegion nodes through lowering |
| `RLX_KITTEN_IF_STUB_META` | Internal | String | Tooling | Stub the KittenTTS If-branch metadata (debug aid) |
| `RLX_KITTEN_INORM_ACTIVE` | Internal | String | Tooling | Treat rank-3 channel-axis-1 norms as KittenTTS instance norm |
| `RLX_KVSTORE_PREAD` | Internal | String | Runtime | Read the KV store with pread instead of mmap (opt-in) |
| `RLX_KVSTORE_READ_STATS` | Internal | String | Runtime | Report KV store batched-read statistics |
| `RLX_KV_CACHE_DBG` | Bisect | Bool | Tooling | Log KV cache eviction decisions |
| `RLX_KV_CACHE_MAX_RESIDENT` | Bisect | U64 | Tooling | Byte ceiling above which KV cache buckets become evictable |
| `RLX_KV_CACHE_NO_EVICT` | Bisect | Bool | Tooling | Never evict KV cache buckets |
| `RLX_LM_HEAD_PARALLEL` | Bisect | Bool | Tooling | Scan vocab rows in parallel via Rayon; set 0 to serialize |
| `RLX_LOG_EPOCH_LOSS` | Bisect | Bool | Tooling | Print the training loss after each epoch |
| `RLX_LSTM_DEBUG` | Bisect | Bool | Tooling | Print LSTM lowering details during ONNX import |
| `RLX_MADV` | Internal | String | Runtime | madvise policy for arena pages, trading footprint against refault cost |
| `RLX_MAX_RAM_BYTES` | Internal | String | Runtime | Hard RAM budget in bytes (else RLX_SOFT_MEMORY_BUDGET_BYTES or physical RAM) |
| `RLX_MAX_RESIDENT_EXPERTS` | Internal | String | Runtime | Max resident MoE experts per layer (else derived from the RAM budget) |
| `RLX_MNIST_DIR` | Bisect | Path | Tooling | Directory holding the raw MNIST files |
| `RLX_MODELS_DIR` | Internal | String | Tooling | Directory searched for test model weights |
| `RLX_MPSG_TRACE` | Bisect | Bool | Tooling | Verbose MPSGraph construction trace |
| `RLX_NATIVE_FK_REGIONS` | Bisect | Bool | Compile | Keep TransformRegion/BatchElementwiseRegion in MIR for native lowering |
| `RLX_NEMO_TEST_FILE` | Bisect | Bool | Tooling | Path to a .nemo file for the NeMo tests |
| `RLX_NO_FK_FUSION` | Bisect | Bool | Compile | Disable the FKL fusion passes |
| `RLX_NO_NATIVE_FK_REGIONS` | Bisect | Bool | Tooling | Force FK regions to decompose rather than lower natively |
| `RLX_NO_SHARED_INPUT_MATMUL` | Bisect | Bool | Tooling | Opt out of sharing one matmul across consumers of the same input |
| `RLX_NO_SHUFFLE` | Internal | Bool | Tooling | Do not shuffle the training set between epochs |
| `RLX_ORT_CUDA_GRAPH` | Bisect | Bool | Tooling | Enable CUDA graph capture in the ONNX Runtime CUDA provider |
| `RLX_PARITY_DEVICE` | Bisect | Bool | Tooling | Device the parity harness compares against: cuda (default) | rocm |
| `RLX_PHASE_TIMING` | Bisect | Bool | Tooling | Print wall time for each compiler phase |
| `RLX_PRECISION` | Internal | String | Tooling | Mixed-precision policy for the graph (default F32) |
| `RLX_PROBE_DEVICE` | Internal | String | Runtime | Device for the probe examples: cpu (default) | cuda | rocm | metal | gpu |
| `RLX_PROBE_DYNAMIC` | Bisect | Bool | Tooling | Import the probe model with dynamic shapes |
| `RLX_PROBE_FEATURE_DIM` | Bisect | Bool | Tooling | Feature dimension for the ONNX import probe |
| `RLX_PROFILE_COMPILE` | Bisect | Bool | Tooling | Report time spent compiling each LIR kernel |
| `RLX_PROFILE_THUNKS` | Bisect | Bool | Tooling | Time each thunk to see which ops dominate a step |
| `RLX_QWEN3_F16_KV` | Internal | Bool | Tooling | Keep the Qwen3 KV cache resident in f16 |
| `RLX_QWEN3_F16_WEIGHTS` | Internal | Bool | Tooling | Store Qwen3 weights in f16 to halve decode weight bandwidth |
| `RLX_QWEN3_FUSED_QKV` | Internal | Bool | Tooling | Fuse Q/K/V into one DequantMatMul (packed weights only) |
| `RLX_QWEN3_GQA_NATIVE` | Internal | Bool | Tooling | Index K/V natively in the SDPA kernels instead of expanding for GQA |
| `RLX_QWEN3_NO_INPLACE_KV` | Internal | Bool | Tooling | Concat the new KV row instead of appending it in place (A/B lever) |
| `RLX_RANK` | Internal | String | Runtime | This process's rank in a distributed run |
| `RLX_RIG_RUNTIME` | Bisect | Bool | Tooling | Where the device sweep runs: local (default) or a rig host |
| `RLX_ROPE_DEBUG` | Bisect | Bool | Tooling | Print RoPE table strides and rotation indices |
| `RLX_SELSCAN_LEGACY_UNROLL` | Internal | String | Tooling | Use the legacy unrolled selective-scan VJP for an A/B |
| `RLX_SOFT_MEMORY_BUDGET_BYTES` | Bisect | U64 | Tooling | Soft RAM budget in bytes, used when RLX_MAX_RAM_BYTES is unset |
| `RLX_SOFT_MEMORY_FRACTION` | Bisect | Bool | Tooling | Fraction of physical RAM used as the soft memory budget |
| `RLX_SPD_JACOBI_SWEEPS` | Bisect | U64 | Tooling | Jacobi sweep count for the SPD eigensolver |
| `RLX_SPD_UNROLL` | Bisect | Bool | Tooling | Use the natively unrolled SPD eigensolver rounds |
| `RLX_TEST_DEVICE` | Internal | String | Runtime | Device the cross-backend tests run against |
| `RLX_TRACE_PERFETTO` | Bisect | Bool | Tooling | Write a Perfetto trace to this path |
| `RLX_TRACE_THUNK` | Bisect | Bool | Tooling | Print each CPU thunk as it executes |
| `RLX_UMAP_CUDA_FUSED_KNN` | Bisect | Bool | Tooling | Use the fused GPU kNN path in UMAP |
| `RLX_USE_ICB` | Bisect | Bool | Tooling | Opt in to Metal indirect command buffers |
| `RLX_WG_LAYERS` | Internal | String | Runtime | Layer count for the whole-graph capture probe |
| `RLX_WINOGRAD` | Bisect | Bool | Tooling | Opt in to the Winograd convolution path |
| `RLX_WORKER_ERR_DIR` | Internal | String | Runtime | Directory distributed workers write captured errors to |

## mlx

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_MLX_COMPILE_MAX_NODES` | Bisect | U64 | Backend(mlx) | Max graph nodes MLX will compile as one Compiled unit (default 1536) |
| `RLX_MLX_CUDA_ARCH` | Bisect | Bool | Backend(mlx) | CUDA arch to build MLX for |
| `RLX_MLX_DEBUG_EVAL` | Bisect | Bool | Backend(mlx) | Log every MLX eval boundary |
| `RLX_MLX_DEQUANT_CACHE_BYTES` | Bisect | U64 | Backend(mlx) | Byte budget for the MLX dequantized-weight cache |
| `RLX_MLX_DEQUANT_CACHE_DISABLE` | Bisect | Bool | Backend(mlx) | Disable the MLX dequantized-weight cache |
| `RLX_MLX_DEVICE` | Bisect | Bool | Backend(mlx) | MLX device label for the device benchmark |
| `RLX_MLX_FUSE_CAP` | Bisect | U64 | Backend(mlx) | Max ops MLX will fuse into one kernel (default 12) |
| `RLX_MLX_GGUF_HOST_FALLBACK` | Public | Bool | Backend(mlx) | Force host GGUF dequant on MLX |
| `RLX_MLX_GROUPED_ONDEVICE` | Internal | Bool | Backend(mlx) | Keep grouped-MoE gather on device instead of host-staging (experiment) |
| `RLX_MLX_GROUPED_ONDEVICE_MAX_BYTES` | Internal | U64 | Backend(mlx) | Byte budget guarding RLX_MLX_GROUPED_ONDEVICE (default 2e9) |
| `RLX_MLX_JOBS` | Bisect | Bool | Backend(mlx) | Parallel job count for the MLX CMake build |
| `RLX_MLX_KEEP_WARM` | Internal | String | Backend(mlx) | Skip the per-tensor WILLNEED/DONTNEED madvise on MLX loads |
| `RLX_MLX_MODE` | Public | Enum | Backend(mlx) | eager | lazy | compiled execution mode |
| `RLX_MLX_PARAM_VIEW` | Bisect | Bool | Backend(mlx) | Hold MLX parameters as views rather than owned arrays |
| `RLX_MLX_PROFILE` | Bisect | Bool | Backend(mlx) | Dump the per-op-kind wall-time breakdown once |
| `RLX_MLX_Q1_HOST` | Bisect | Bool | Backend(mlx) | Dequantize Q1 weights on the host instead of through MLX |
| `RLX_MLX_Q1_MV_DISABLE` | Bisect | Bool | Backend(mlx) | Disable the Q1 matrix-vector kernel; use one-shot dequant plus sgemm |
| `RLX_MLX_RNN_F16` | Bisect | Bool | Backend(mlx) | Run the RNN recurrence in fp16 compute on the Metal GPU |
| `RLX_MLX_SDPA_REFERENCE` | Public | Bool | Backend(mlx) | Use reference SDPA composition for bisects |
| `RLX_MLX_WARN_LAZY` | Bisect | Bool | Backend(mlx) | Lazy-evaluation warning level; set all for per-executable warnings |

## oneapi

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_OCLOC` | Internal | Path | Backend(oneapi) | Path to Intel `ocloc` for OpenCL C ahead-of-time compilation |
| `RLX_ONEAPI_BUILD_KERNELS` | Bisect | Bool | Backend(oneapi) | Build native oneAPI SPIR-V kernels at build time (needs ocloc) |
| `RLX_ONEAPI_LOADER` | Bisect | Bool | Backend(oneapi) | Path to the Level Zero loader library |
| `RLX_ONEAPI_OCLOC` | Bisect | Bool | Backend(oneapi) | Path to the ocloc kernel compiler (default ocloc) |
| `RLX_ONEAPI_OCLOC_DEVICE` | Bisect | Bool | Backend(oneapi) | Target device token passed to ocloc (default pvc) |

## onnx

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_ONNX_ALLOW_RESIZE_STUB` | Bisect | Bool | Tooling | Import an unlowerable ONNX Resize as zeros instead of failing (runs, output is silently wrong) |
| `RLX_ONNX_BUNDLE` | Bisect | Bool | Tooling | Path to an exported ONNX bundle directory |
| `RLX_ONNX_SEQUENCE_LENGTH` | Bisect | Bool | Tooling | Sequence length to restore and propagate for ONNX import |
| `RLX_ONNX_TAP` | Bisect | Bool | Tooling | Comma-separated ONNX tensor names to tap during import |
| `RLX_ONNX_TEST_MODEL` | Bisect | String | Tooling | Path to the ONNX model used by the import tests |

## qnn

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_QNN_BACKEND_LIB` | Bisect | Bool | Backend(qnn) | Explicit path to the QNN backend library (libQnn*.so) |

## quant

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_MLX_DEQUANT_GPU_DISABLE` | Public | Bool | Runtime | Force host-staged MLX DequantMatMul on all GPU backends |

## rocm

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_DUMP_KERNELS` | Internal | String | Backend(rocm) | Snapshot generated GPU kernel sources into this directory |
| `RLX_GPU_VALIDATE_PARAMS` | Internal | Bool | Backend(rocm) | Check each launch's argument count against the kernel signature |
| `RLX_ROCM_ARCH` | Internal | String | Backend(rocm) | Explicit ROCm target arch override, e.g. gfx1100 |
| `RLX_ROCM_ARENA_DUMP` | Internal | String | Backend(rocm) | Dump the ROCm arena layout |
| `RLX_ROCM_ARENA_NO_REUSE` | Internal | String | Backend(rocm) | Give every ROCm tensor one permanent arena slot; disables reuse |
| `RLX_ROCM_DISABLE_MIOPEN` | Internal | String | Backend(rocm) | Disable MIOpen whether or not its kernel database is present |
| `RLX_ROCM_EXEC` | Internal | String | Backend(rocm) | ROCm execution mode: stream | graph | ms<N> |
| `RLX_ROCM_FAST_FP_ATOMICS` | Internal | Bool | Backend(rocm) | Compile HIP kernels with unsafe fast fp atomics |
| `RLX_ROCM_GEMV` | Internal | Bool | Backend(rocm) | Opt in to the skinny-m split-K ROCm GEMV kernel |
| `RLX_ROCM_GGUF_DIAG` | Internal | Bool | Backend(rocm) | Diagnose GGUF dequant-matmul dispatch on ROCm |
| `RLX_ROCM_GGUF_FUSED_DISABLE` | Internal | Bool | Backend(rocm) | Disable the fused GGUF matmul kernel on ROCm |
| `RLX_ROCM_GGUF_HOST` | Internal | Bool | Backend(rocm) | Dequantize GGUF weights on the CPU instead of on device |
| `RLX_ROCM_HSACO_CACHE` | Bisect | Path | Backend(rocm) | Directory for the compiled HSACO cache |
| `RLX_ROCM_IM2COL_HOST` | Bisect | Bool | Backend(rocm) | Run im2col on the host instead of the ROCm kernel |
| `RLX_ROCM_INDEXING_HOST` | Bisect | Bool | Backend(rocm) | Force ND indexing (GatherND/GatherElements/Scatter*) to the CPU host route |
| `RLX_ROCM_LEGACY_ATTENTION_ROW` | Internal | Bool | Backend(rocm) | Use the legacy thread-per-row attention kernel instead of warp-per-row |
| `RLX_ROCM_LOG_FALLBACK` | Bisect | Bool | Backend(rocm) | Log every op that falls back off its native ROCm path |
| `RLX_ROCM_MFMA` | Bisect | Bool | Backend(rocm) | Use MFMA matrix cores where the arch supports them |
| `RLX_ROCM_MIOPEN_FORCE` | Internal | String | Backend(rocm) | Use MIOpen even with no kernel database present (for measuring) |
| `RLX_ROCM_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(rocm) | Disable the packed BSHD attention kernel on ROCm |
| `RLX_ROCM_NO_TF32` | Internal | Bool | Backend(rocm) | Disable XF32/TF32 on ROCm and compute in true f32 |
| `RLX_ROCM_NO_VENDOR_GEMM` | Internal | Bool | Backend(rocm) | Use rlx's own GEMM instead of rocBLAS |
| `RLX_ROCM_PARAM_DIAG` | Internal | Bool | Backend(rocm) | Diagnose parameter uploads on ROCm |
| `RLX_ROCM_PARITY` | Internal | Bool | Backend(rocm) | Tighten ROCm numerics for cross-backend parity |
| `RLX_ROCM_PINNED_IO` | Public | Bool | Backend(rocm) | Use pinned host I/O for ROCm graph exec (default on in graph mode) |
| `RLX_ROCM_PROFILE_STEPS` | Internal | Bool | Backend(rocm) | Attribute wall time to each op kind by syncing between steps |
| `RLX_ROCM_RNN_HOST_FALLBACK` | Internal | Bool | Backend(rocm) | Force RNN ops onto the host on ROCm |
| `RLX_ROCM_SCHEDULE_MATMUL` | Internal | String | Backend(rocm) | Source for the HIP matmul kernel: schedule-IR emitted vs handwritten |
| `RLX_ROCM_SMI_INDEX` | Internal | U64 | Backend(rocm) | rocm-smi card index for contention gating (follows HIP_VISIBLE_DEVICES) |
| `RLX_ROCM_SSM_HOST_FALLBACK` | Internal | Bool | Backend(rocm) | Force SSM ops onto the host on ROCm |
| `RLX_ROCM_TRACE_STEPS` | Internal | Bool | Backend(rocm) | Print each step synchronously so a crashing kernel is identifiable |

## tpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_TPU_BENCH` | Internal | Bool | Backend(tpu) | Run the TPU benchmarks (off during a normal cargo test) |
| `RLX_TPU_BENCH_SWEEP` | Internal | Bool | Backend(tpu) | Run the slower TPU benchmark sweep |
| `RLX_TPU_HLO_DUMP` | Bisect | Bool | Backend(tpu) | Dump generated HLO for inspection |
| `RLX_TPU_MATMUL_HIGHEST` | Internal | Bool | Backend(tpu) | Pin TPU matmul to highest precision instead of bf16 passes |

## vulkan

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_GLSLANG` | Internal | Path | Backend(vulkan) | Path to `glslangValidator` for GLSL->SPIR-V (naga lacks memoryBarrierShared) |
| `RLX_VULKAN_ARENA_DEBUG` | Bisect | Bool | Backend(vulkan) | Report Vulkan arena allocation and slot reuse |
| `RLX_VULKAN_CHECK_BCAST` | Bisect | String | Backend(vulkan) | Verify broadcast strides before each Vulkan dispatch |
| `RLX_VULKAN_CHECK_CAST` | Bisect | String | Backend(vulkan) | Verify dtype casts during Vulkan pipeline build |
| `RLX_VULKAN_DEBUG` | Bisect | Bool | Backend(vulkan) | Extra Vulkan diagnostics and object labelling |
| `RLX_VULKAN_DUMP_OPS` | Bisect | Bool | Backend(vulkan) | Dump the Vulkan dispatch schedule |
| `RLX_VULKAN_FULLBARRIER` | Bisect | Bool | Backend(vulkan) | Insert a full barrier between every pair of dispatches |
| `RLX_VULKAN_HOST_CONV` | Bisect | Bool | Backend(vulkan) | Run convolution on the host instead of a Vulkan kernel |
| `RLX_VULKAN_HOST_OPS` | Bisect | Bool | Backend(vulkan) | Comma-separated op classes forced onto the host, or all |
| `RLX_VULKAN_INDEXING_HOST` | Bisect | Bool | Backend(vulkan) | Force ND indexing (GatherND/GatherElements/Scatter*) to the CPU host route |
| `RLX_VULKAN_MATMUL` | Bisect | Bool | Backend(vulkan) | Vulkan matmul kernel: auto (default) | scalar | tiled |
| `RLX_VULKAN_MLX_NATIVE` | Internal | Bool | Backend(vulkan) | Route dense MLX matmuls back onto the SPIR-V kernels |
| `RLX_VULKAN_NOBARRIER` | Bisect | Bool | Backend(vulkan) | Drop all pipeline barriers (unsafe; for isolating sync bugs) |
| `RLX_VULKAN_NOCACHE` | Bisect | Bool | Backend(vulkan) | Disable the command-buffer cache; re-record every run |
| `RLX_VULKAN_SCAN_ALL` | Bisect | Bool | Backend(vulkan) | Scan every buffer for NaN/Inf after each run |
| `RLX_VULKAN_SCHEDULE_MATMUL` | Internal | String | Backend(vulkan) | Source for the GLSL matmul kernel: schedule-IR emitted vs handwritten |
| `RLX_VULKAN_SHARD_LOG` | Bisect | Bool | Backend(vulkan) | Log Vulkan buffer sharding decisions |
| `RLX_VULKAN_SHARD_STAGE_MIB` | Internal | String | Backend(vulkan) | Staging-window size in MiB for sharded Vulkan uploads |
| `RLX_VULKAN_VALIDATION` | Bisect | Bool | Backend(vulkan) | Enable the Khronos validation layer |

## wgpu

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_WGPU_ACTIVE_TRACE` | Bisect | Bool | Backend(wgpu) | Trace each wgpu dispatch as it is submitted |
| `RLX_WGPU_ALLOW_SHARD` | Bisect | Bool | Backend(wgpu) | Warn instead of refusing when the activation arena would be striped across buffers (wrong results) |
| `RLX_WGPU_ALL_PARAMS_WEIGHT` | Internal | Bool | Backend(wgpu) | Park every parameter in the weight arena rather than splitting |
| `RLX_WGPU_ARENA_REPORT` | Internal | Bool | Backend(wgpu) | Report which tensors dominate the activation arena after planning |
| `RLX_WGPU_CONCAT_HOST` | Bisect | Bool | Backend(wgpu) | Run Concat on the host instead of a wgpu kernel |
| `RLX_WGPU_CONV3D_SCALAR` | Bisect | Bool | Backend(wgpu) | Force the one-element-per-thread Conv3D kernel instead of the 4-channel tiled one |
| `RLX_WGPU_CONV_HOST` | Internal | Bool | Backend(wgpu) | Run convolution on the host instead of a wgpu kernel |
| `RLX_WGPU_CONV_IM2COL` | Bisect | Bool | Backend(wgpu) | Use the im2col convolution path on wgpu |
| `RLX_WGPU_COOP_F16_VK_DISABLE` | Bisect | Bool | Backend(wgpu) | Disable f16 cooperative-matrix matmul on Vulkan |
| `RLX_WGPU_COOP_F16_VK_ENABLE` | Bisect | Bool | Backend(wgpu) | Re-enable f16 cooperative-matrix matmul on Vulkan (at your risk) |
| `RLX_WGPU_COOP_F16_VK_FORCE_WIDE` | Bisect | Bool | Backend(wgpu) | Force the wide cooperative-matrix tile |
| `RLX_WGPU_COOP_F16_VK_LARGE_N` | Bisect | Bool | Backend(wgpu) | Keep cooperative tensor cores at large N instead of falling back |
| `RLX_WGPU_COOP_F16_VK_LOAD_T` | Bisect | Bool | Backend(wgpu) | Use coopLoadT on B at N > 768 instead of coopLoad |
| `RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE` | Bisect | Bool | Backend(wgpu) | Disable automatic selection of the wide cooperative tile |
| `RLX_WGPU_COOP_F16_VK_NO_F32ACC` | Bisect | Bool | Backend(wgpu) | Accumulate cooperative f16 matmul in f16 instead of f32 |
| `RLX_WGPU_COOP_F16_VK_OSC_THRESH` | Bisect | Bool | Backend(wgpu) | Oscillation threshold for cooperative tile selection (default 0.35) |
| `RLX_WGPU_DBG_COUNTERS` | Internal | Bool | Backend(wgpu) | Print step/host/uniform/bind-group counters |
| `RLX_WGPU_DBG_HOST_OP` | Internal | Bool | Backend(wgpu) | Log each op routed to the host fallback |
| `RLX_WGPU_DBG_INT8_HOST` | Internal | Bool | Backend(wgpu) | Log the host int8 dequant-matmul path |
| `RLX_WGPU_DBG_TRANSPOSE` | Internal | Bool | Backend(wgpu) | Trace transpose lowering decisions |
| `RLX_WGPU_DEBUG` | Bisect | Bool | Backend(wgpu) | Extra wgpu diagnostics and validation |
| `RLX_WGPU_DEBUG_ATTN_ALIAS` | Bisect | Bool | Backend(wgpu) | Report when an attention output aliases its Q/K/V input |
| `RLX_WGPU_DEBUG_ATTN_MASK` | Bisect | Bool | Backend(wgpu) | Report the attention mask shape and kind chosen |
| `RLX_WGPU_DEBUG_OP_HIST` | Bisect | Bool | Backend(wgpu) | Print a histogram of op kinds after compilation |
| `RLX_WGPU_DISCRETE_HOST` | Internal | Bool | Backend(wgpu) | Force full host execution on discrete adapters |
| `RLX_WGPU_DUMP_FLAT` | Bisect | Bool | Backend(wgpu) | Also print this flat element index when dumping node stats |
| `RLX_WGPU_DUMP_IDS` | Bisect | Bool | Backend(wgpu) | Comma-separated node ids to dump |
| `RLX_WGPU_DUMP_INPUTS` | Bisect | Bool | Backend(wgpu) | Dump graph inputs before running |
| `RLX_WGPU_DUMP_NODES` | Bisect | Bool | Backend(wgpu) | Dump per-node output buffers for cross-backend diffing |
| `RLX_WGPU_DUMP_NODES_LIMIT` | Bisect | U64 | Backend(wgpu) | Cap how many nodes the wgpu dump writes (default 40) |
| `RLX_WGPU_DUMP_TAIL` | Bisect | Bool | Backend(wgpu) | Dump the tail of each buffer rather than the head |
| `RLX_WGPU_EXPAND_HOST` | Internal | Bool | Backend(wgpu) | Run Expand on the host instead of a wgpu kernel |
| `RLX_WGPU_F16_WEIGHTS` | Bisect | Bool | Backend(wgpu) | Store wgpu weights in f16 to halve weight bandwidth |
| `RLX_WGPU_FORCE_COOP_F32` | Bisect | Bool | Backend(wgpu) | Force f32 cooperative matrices (unreliable on RTX, which lacks 8x8 f32) |
| `RLX_WGPU_FORCE_HOST` | Internal | Bool | Backend(wgpu) | Host every op, reproducing discrete-adapter behaviour on any adapter |
| `RLX_WGPU_FORCE_INPUT_UPLOAD` | Bisect | Bool | Backend(wgpu) | Re-upload graph inputs every run instead of reusing device copies |
| `RLX_WGPU_GATHER_SPLIT` | Internal | Bool | Backend(wgpu) | Split Gather into per-slice dispatches |
| `RLX_WGPU_GDN_HOST` | Public | Bool | Backend(wgpu) | Force GatedDeltaNet host fallback on wgpu (skip WGSL) |
| `RLX_WGPU_GPU_NORM` | Internal | Bool | Backend(wgpu) | Force norms onto the GPU (opposite of RLX_WGPU_HOST_NORM) |
| `RLX_WGPU_HAZARD_REPORT` | Internal | Bool | Backend(wgpu) | Report arena slot read/write hazards after planning |
| `RLX_WGPU_HOST_BUFFER_COPY` | Bisect | Bool | Backend(wgpu) | Route buffer-to-buffer copies through the host |
| `RLX_WGPU_HOST_EAGER_H2D` | Internal | Bool | Backend(wgpu) | Flush deferred host-to-device copies eagerly rather than batching |
| `RLX_WGPU_HOST_MATMUL` | Internal | Bool | Backend(wgpu) | Run matmul on the host instead of a wgpu kernel |
| `RLX_WGPU_HOST_NORM` | Internal | Bool | Backend(wgpu) | Force norms onto the host |
| `RLX_WGPU_IM2COL_MIN_COUT` | Bisect | U64 | Backend(wgpu) | Minimum output channels before im2col is used (default 64) |
| `RLX_WGPU_IM2COL_MIN_K` | Bisect | U64 | Backend(wgpu) | Minimum K before im2col is used (default 256) |
| `RLX_WGPU_IM2COL_MIN_SPATIAL` | Bisect | U64 | Backend(wgpu) | Minimum spatial extent before im2col is used (default 2048) |
| `RLX_WGPU_INDEXING_HOST` | Bisect | Bool | Backend(wgpu) | Force ND indexing (GatherND/GatherElements/Scatter*) to the CPU host route |
| `RLX_WGPU_LARGE_BUFFERS` | Bisect | Bool | Backend(wgpu) | Allow buffers past the conservative wgpu size limit |
| `RLX_WGPU_MATMUL_F32_ONLY` | Bisect | Bool | Backend(wgpu) | Restrict wgpu matmul to f32, disabling f16 paths |
| `RLX_WGPU_MAX_BIND_MB` | Internal | String | Backend(wgpu) | Cap the bind-group size in MiB to exercise the sharding path |
| `RLX_WGPU_NAN_TRACE` | Bisect | Bool | Backend(wgpu) | Legacy wgpu NaN trace (prefer RLX_DEBUG_NANS) |
| `RLX_WGPU_NO_COOP_F16_VK` | Bisect | Bool | Backend(wgpu) | Disable f16 cooperative matrices in the wgpu Vulkan backend |
| `RLX_WGPU_NO_COOP_F32` | Bisect | Bool | Backend(wgpu) | Disable f32 cooperative matrices |
| `RLX_WGPU_NO_F16_MIRROR` | Bisect | Bool | Backend(wgpu) | Do not keep an f16 mirror of f32 weights |
| `RLX_WGPU_NO_F16_SHADOW` | Bisect | Bool | Backend(wgpu) | Do not allocate the f16 shadow buffer |
| `RLX_WGPU_NO_FOLD_MMBA` | Internal | Bool | Backend(wgpu) | Disable folding matmul+bias+activation into one node |
| `RLX_WGPU_NO_FOLD_RESLN` | Internal | Bool | Backend(wgpu) | Disable folding residual add into LayerNorm |
| `RLX_WGPU_NO_PACKED_BSHD_ATTN` | Bisect | Bool | Backend(wgpu) | Disable the packed BSHD attention kernel on wgpu |
| `RLX_WGPU_NO_TILED_CONV` | Bisect | Bool | Backend(wgpu) | Disable the tiled convolution kernel |
| `RLX_WGPU_ONE_OP_PER_PASS` | Internal | Bool | Backend(wgpu) | Submit one op per compute pass (isolates sync bugs) |
| `RLX_WGPU_PRINT_LIMITS` | Bisect | U64 | Backend(wgpu) | Print the adapter's reported limits at startup |
| `RLX_WGPU_Q1_0_GEMM_DISABLE` | Bisect | Bool | Backend(wgpu) | Disable the Q1_0 GEMM kernel on wgpu |
| `RLX_WGPU_QUIET_SHARD` | Internal | Bool | Backend(wgpu) | Suppress the buffer-sharding log |
| `RLX_WGPU_ROW_TABS` | Internal | U64 | Backend(wgpu) | Per-row sum of |x| for one snapshot step |
| `RLX_WGPU_SCHEDULE` | Bisect | Bool | Backend(wgpu) | Print the wgpu execution schedule |
| `RLX_WGPU_SHARD_CAP_MIB` | Bisect | U64 | Backend(wgpu) | Lower the per-shard activation-arena cap below the adapter's max_buffer_size (clamps smaller only) so the striping path is testable without multi-GiB allocations |
| `RLX_WGPU_SHARD_GPU` | Internal | Bool | Backend(wgpu) | Keep sharded ops on the GPU instead of hosting them |
| `RLX_WGPU_SHARD_LOG` | Bisect | Bool | Backend(wgpu) | Log wgpu buffer sharding decisions |
| `RLX_WGPU_SHARD_STAGE_MIB` | Internal | String | Backend(wgpu) | Staging-window size in MiB for sharded wgpu uploads |
| `RLX_WGPU_SHARE_WEIGHTS` | Public | Bool | Backend(wgpu) | Reuse GPU weight buffers across compiles with matching named layouts (default on; set 0 to disable) |
| `RLX_WGPU_SNAPSHOT` | Internal | Bool | Backend(wgpu) | Replay the compiled schedule truncated at each step, reading every node as produced |
| `RLX_WGPU_SNAPSHOT_WATCH` | Internal | U64 | Backend(wgpu) | Also report one chosen node id after every snapshot step |
| `RLX_WGPU_STRIPE_VERBOSE` | Internal | Bool | Backend(wgpu) | Print every op whose operands straddle an arena stripe boundary |
| `RLX_WGPU_TEE_VETO_LOG` | Internal | Bool | Backend(wgpu) | Log residual-LN tee folds vetoed as arena-unsafe |
| `RLX_WGPU_TILED_MIN_SPATIAL` | Bisect | U64 | Backend(wgpu) | Minimum spatial extent before the tiled conv path is used |
| `RLX_WGPU_TRANSPOSE_HOST` | Internal | Bool | Backend(wgpu) | Run Transpose on the host instead of a wgpu kernel |

## xdna

| Name | Stability | Kind | Layer | Summary |
|------|-----------|------|-------|---------|
| `RLX_XDNA_AIE_INCLUDE` | Internal | String | Backend(xdna) | Include directory for the MLIR-AIE toolchain |
| `RLX_XDNA_CHAIN_CAP` | Internal | String | Backend(xdna) | Max ops chained into one XDNA submission |
| `RLX_XDNA_GEMM` | Internal | String | Backend(xdna) | GEMM dimensions for the XDNA example, as m,k,n |
| `RLX_XDNA_NO_RESIDENT` | Internal | String | Backend(xdna) | Re-upload weights each call instead of keeping them resident on the NPU |
| `RLX_XDNA_SHIM` | Internal | String | Backend(xdna) | Path to the XDNA shim library |
| `RLX_XDNA_SKIP_PROBE` | Bisect | Bool | Backend(xdna) | Trust the toolchain paths instead of compiling a probe design during device selection |
| `RLX_XDNA_TURBO` | Internal | String | Backend(xdna) | Request NPU turbo clocks (needs root/DRM-master) |
| `RLX_XDNA_UMQ` | Internal | String | Backend(xdna) | Request a user-mode queue with a ring buffer object |
| `RLX_XDNA_UMQ_RING` | Internal | String | Backend(xdna) | Ring-buffer size in bytes for the XDNA user-mode queue |
| `RLX_XDNA_XRT_LIB` | Internal | String | Backend(xdna) | Explicit path to the XRT library (which ships only versioned sonames) |

## Unregistered mentions

Identifiers still appearing in the tree but not yet in the registry (docs, benches, or pending migration). Prefer registering or deleting.

| Name | Example path |
|------|--------------|
| `RLX_ACTIVATION_BACKWARD` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_ACT_INPLACE_H` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_ANDROID_AVD` | `android/e2e.sh` |
| `RLX_BACKENDS_MANIFEST_PATH` | `crates/core/rlx-runtime/src/backends_manifest.rs` |
| `RLX_BARRIER` | `crates/backends/rlx-metal/src/apple_params.rs` |
| `RLX_BATCH` | `CHANGELOG.md` |
| `RLX_BENCH` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_BINARY_FN` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_CODESIGN_IDENTITY` | `crates/backends/rlx-egpu/dext/build.sh` |
| `RLX_COMPARE_FN` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_COMPILE_OUTPUT_CAP` | `crates/backends/rlx-mlx/src/config.rs` |
| `RLX_CONCAT_MIDAXIS` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_CUDA_ATTENTION` | `crates/backends/rlx-cuda/src/config.rs` |
| `RLX_CUDA_ATTENTION_WMMA` | `crates/backends/rlx-cuda/src/config.rs` |
| `RLX_CUDA_ATTENTION_WMMA_MIN_WORK` | `crates/backends/rlx-cuda/src/config.rs` |
| `RLX_CUDA_CONV_T_CUDNN` | `crates/backends/rlx-cuda/README.md` |
| `RLX_CUDA_FORCE_ATTENTION_ROW` | `crates/core/rlx-ir/src/attention_layout.rs` |
| `RLX_CUDA_FULL_KV_READBACK` | `crates/backends/rlx-cuda/README.md` |
| `RLX_CUDA_SCHEDULE_MATMUL` | `crates/backends/rlx-cuda/src/config.rs` |
| `RLX_CUDA_TMA` | `crates/backends/rlx-cuda/src/config.rs` |
| `RLX_DEBUG_TRIP` | `scripts/gen-rlx-env-vars.py` |
| `RLX_DECOMPOSE_BENCH_RUNS` | `rig.sh` |
| `RLX_DECOMPOSE_BENCH_WARMUP` | `rig.sh` |
| `RLX_DENY_DEVICES` | `docs/backend-selection.md` |
| `RLX_DETERMINISTIC_REDUCE` | `crates/core/rlx-collectives/src/lib.rs` |
| `RLX_DEXT_BUNDLE_ID` | `crates/backends/rlx-egpu/dext/build.sh` |
| `RLX_DEXT_OUT` | `crates/backends/rlx-egpu/dext/build.sh` |
| `RLX_DISABLE_MPSGRAPH_PARAM_CONST` | `crates/backends/rlx-metal/src/backend/mod.rs` |
| `RLX_EGPU_HELPER` | `crates/backends/rlx-egpu/src/dext.rs` |
| `RLX_EGPU_SOCK` | `crates/backends/rlx-egpu/src/dext.rs` |
| `RLX_ENC_MAGIC` | `crates/io/rlx-bake/src/lib.rs` |
| `RLX_ENC_VERSION` | `crates/io/rlx-bake/src/lib.rs` |
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
| `RLX_FORMAT_VERSION` | `crates/io/rlx-bake/src/format.rs` |
| `RLX_FW_MANIFEST` | `scripts/pull_gpu_firmware.sh` |
| `RLX_GELU_INPLACE_F32` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_GEMMA3_GGUF` | `rig.sh` |
| `RLX_GRAPH_FUSED` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_IROH_ALPN` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_PEERS` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_SECRET` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_IROH_SEED` | `crates/core/rlx-driver/src/iroh_transport.rs` |
| `RLX_JNI_FEATURES` | `android/build.sh` |
| `RLX_KERNELS_MSL` | `crates/backends/rlx-metal/src/icb.rs` |
| `RLX_KERNELS_MSL_DEQUANT` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_FFT_GPU` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_SPLAT` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_KERNELS_MSL_SPLAT_CONIC` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_LOCATEANYTHING_DIR` | `rig.sh` |
| `RLX_LR` | `docs/benchmarks/coreml-training.md` |
| `RLX_MAGIC` | `crates/io/rlx-bake/src/format.rs` |
| `RLX_METAL_GDN_NATIVE` | `crates/core/rlx-runtime/tests/cpu_gated_delta_net_parity.rs` |
| `RLX_METAL_GDN_SG` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_METAL_IQ` | `crates/backends/rlx-metal/README.md` |
| `RLX_METAL_SDPA_FLASH_DECODE` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_FLASH_P` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_TUNE_CACHE` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_TUNE_CACHE_EVICTION` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_TUNE_CACHE_LOAD` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_TUNE_CACHE_MAX_ENTRIES` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_METAL_SDPA_TUNE_CACHE_PERSIST` | `crates/backends/rlx-metal/src/config.rs` |
| `RLX_MINICPM5_GGUF_DIR` | `rig.sh` |
| `RLX_MINICPM5_GGUF_Q4_K_M` | `rig.sh` |
| `RLX_MLX_BENCH_PROFILE` | `rig.sh` |
| `RLX_MLX_COMPILE_OUTPUT_CAP` | `crates/backends/rlx-mlx/src/config.rs` |
| `RLX_MLX_CUDA` | `crates/backends/rlx-mlx-sys/build.rs` |
| `RLX_MLX_NO_CCACHE` | `crates/backends/rlx-mlx-sys/build.rs` |
| `RLX_MLX_OK` | `crates/backends/rlx-mlx/src/array.rs` |
| `RLX_MPSGRAPH_ATTENTION` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_NODE_ERR_ARG` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NODE_ERR_BUSY` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NODE_ERR_CONFIG` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NODE_ERR_MODE` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NODE_ERR_SPAWN` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NODE_OK` | `crates/bindings/rlx-ffi/src/lib.rs` |
| `RLX_NTH_ORDER_RUNS` | `rig.sh` |
| `RLX_NTH_ORDER_SIZES` | `rig.sh` |
| `RLX_NTH_ORDER_WARMUP` | `rig.sh` |
| `RLX_ORT_INTRA_THREADS` | `crates/io/rlx-onnx/src/backend.rs` |
| `RLX_PIPELINE_ALPN` | `crates/core/rlx-driver/src/lib.rs` |
| `RLX_POW_SCALAR_FN` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_PREFER_DEVICES` | `docs/backend-selection.md` |
| `RLX_QNN_HTP_LIB` | `Justfile` |
| `RLX_QWEN25_GGUF` | `rig.sh` |
| `RLX_QWEN3_F16_LM_HEAD` | `CHANGELOG.md` |
| `RLX_QWEN3_INPLACE_KV` | `crates/core/rlx-flow/src/blocks/qwen3_decode_layer.rs` |
| `RLX_QWEN3_PARITY` | `CHANGELOG.md` |
| `RLX_QWEN3_RETENTION` | `docs/README.md` |
| `RLX_QWEN3_RETENTION_DEBUG` | `docs/kv-retention.md` |
| `RLX_QWEN3_TTS_DIR` | `rig.sh` |
| `RLX_REGIONS` | `crates/backends/rlx-cortexm/trainer/src/train.rs` |
| `RLX_RESIDENT` | `docs/benchmarks/frameworks-and-backends.md` |
| `RLX_RIG_BUILD` | `rig.sh` |
| `RLX_RIG_CUDA_BENCH` | `rig.sh` |
| `RLX_RIG_DEST` | `scripts/sync-to-rig.sh` |
| `RLX_RIG_ENV` | `rig.sh` |
| `RLX_RIG_HOST` | `scripts/sync-to-rig.sh` |
| `RLX_RIG_MODELS_FEATURES` | `rig.sh` |
| `RLX_RIG_ROOT` | `rig.sh` |
| `RLX_RIG_SKIP_SYNC` | `rig.sh` |
| `RLX_RIG_SYNC_NO_PRUNE` | `rig.sh` |
| `RLX_RIG_WORKSPACE` | `rig.sh` |
| `RLX_ROCM_FORCE_ATTENTION_ROW` | `crates/backends/rlx-rocm/src/backend/run.rs` |
| `RLX_SCALAR_ACT_FNS` | `crates/backends/rlx-metal/src/icb.rs` |
| `RLX_SDPA_DECODE_M1` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_SGEMM_TILES` | `crates/backends/rlx-metal/src/kernels.rs` |
| `RLX_SHFL_ALL_LANES` | `crates/backends/rlx-rocm/src/backend/run.rs` |
| `RLX_SIM_DEVICE` | `Justfile` |
| `RLX_SKIP_WRITEBACK` | `docs/benchmarks/frameworks-and-backends.md` |
| `RLX_SMOLLM2_GGUF` | `rig.sh` |
| `RLX_STAGES` | `crates/backends/rlx-metal/src/apple_params.rs` |
| `RLX_STAGE_T` | `crates/backends/rlx-metal/src/apple_params.rs` |
| `RLX_TAP_L0` | `crates/backends/rlx-metal/src/backend/encode/mod.rs` |
| `RLX_TORCH_IMPORT_BIN` | `crates/bindings/pyrlx/pyproject.toml` |
| `RLX_TRANSPORT` | `docs/iroh-transport.md` |
| `RLX_TS` | `crates/backends/rlx-metal/src/apple_params.rs` |
| `RLX_USE_MPSGRAPH` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_USE_MPS_GRAPH` | `crates/backends/rlx-metal/src/mps_graph.rs` |
| `RLX_WHISPER_DIR` | `rig.sh` |
| `RLX_XDNA_INSTS` | `crates/backends/rlx-xdna/src/lib.rs` |
| `RLX_XDNA_XCLBIN` | `crates/backends/rlx-xdna/src/lib.rs` |

## Maintenance

```sh
just gen-rlx-env-vars
# or: python3 scripts/gen-rlx-env-vars.py
```

Add new names to `env_registry_data.inc.rs`. Unregistered `env::flag("RLX_…")` call sites fail `just check-rlx-env-vars`.
