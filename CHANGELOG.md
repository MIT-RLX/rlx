# Changelog

All notable changes to RLX. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
tracks SemVer with the understanding that any `0.x → 0.(x+1)`
bump may carry breaking changes per `0.x`-semver convention.

## [Unreleased]

## [0.2.13] - 2026-07-18

### Added

- **CUDA: fused conv + bias + activation (+ residual) via cuDNN.**
  `Op::FusedConvBiasAct` folds a convolution's bias + `relu` — and an
  optional ResNet residual (cuDNN `z` operand) — into one
  `cudnnConvolutionBiasActivationForward`. `FuseConvBiasAct` matches
  `conv→bias→relu`; `FuseConvAffineAct` folds a host-pre-folded BatchNorm
  block `conv→Mul(scale)→Add(shift)→[Add(residual)]→relu` (per-channel scale
  folds into the weights). Fires only for cuDNN-friendly shapes
  (`groups==1, k>1`) + identity/relu — **1.6–2.2×** vs unfused at batch 1 on an
  NVIDIA GPU. cuDNN stays optional: falls back to the direct-conv kernel +
  `conv_bias_act_epilogue.cu` (bit-exact) when libcudnn is absent. Diagnostics:
  `RLX_CUDA_LOG_CONV_PATH`, `RLX_CUDA_LOG_FALLBACK`,
  `RLX_DISABLE_CONV_BIAS_ACT_FUSION`. See
  [`crates/backends/rlx-cuda/README.md`](crates/backends/rlx-cuda/README.md).

- **Compute weight-derived tensors once, not every forward** — two mechanisms.
  Compile-time: `CompileOptions::param_bindings` bakes weights to constants and
  `ConstantFolding` (now with NumPy **broadcasting**) folds per-channel weight
  math away. Run time: `CompileOptions::cache_param_invariant` /
  `RLX_CACHE_PARAM_INVARIANT=1` splits the param-invariant closure into a
  *prepare* graph run once (`rlx_compile::split_param_invariant`), injected into
  the main graph via persistent `bind_handle` (CPU) or a feed fallback (CUDA).
  Complementary + transparent; validated CPU + CUDA. See
  [`docs/weight-compute-caching.md`](docs/weight-compute-caching.md).

- **QNN / Hexagon: x86 HTP functional simulator recipes** (`just qnn-htp-sim`).
  Points `RLX_QNN_BACKEND_LIB` at `libQnnHtp.so` (no Snapdragon silicon).
  Re-runs quantized MatMul probes plus LinearStatic `run_qnn.sh` /
  `run_qnn_context.sh`. Emitted run scripts honor `RLX_QNN_BACKEND_LIB`
  (default still `libQnnCpu.so`). I8×I8 MatMul IR lowers as Dequantize both →
  f32 MatMul (portable across CPU / HTP sim).

- **QNN / Hexagon: offline context-binary path** (`run_qnn_context.sh`).
  Emit → `qnn-model-lib-generator` → `qnn-context-binary-generator` →
  `qnn-net-run --retrieve_context model.bin`. Complements the FFI
  `export_context_binary` / `reload_from_context_binary` path.

- **QNN / Hexagon: `from_graph` → `LinearStatic` with Constant weights.**
  When `Add(MatMul(Input, Constant), Constant)` is recognized, weight/bias
  f32 payloads are baked into the emitted `qnn_model.cpp` (not seed-0).

- **QNN / Hexagon: codegen `LinearStatic`** (STATIC weight/bias packing).
  Activation-only `APP_WRITE`; `W`/`b` baked into `qnn_model.cpp`. CLI
  `--linear-static` / `just qnn-emit-linear-static`.

- **QNN / Hexagon: codegen `Mlp2`** (two-layer `LinearRelu → Linear`).
  Offline `qnn-net-run` path; CLI `--mlp2 M K H N` / `just qnn-emit-mlp2`.

- **QNN / Hexagon: per-channel int8 MatMul weights** (`AXIS_SCALE_OFFSET`).
  `Dequantize` with `axis=Some` + `scales.len() > 1` stages STATIC
  `SFIXED_POINT_8` with QNN `AXIS_SCALE_OFFSET` (e.g. Linear `[K,N]` axis=1).

- **QNN / Hexagon: codegen `MatMulSoftmax`** (`softmax(in0·in1, axis=1)`).
  Offline `qnn-net-run` path; CLI `--matmul-softmax` / `just qnn-emit-matmul-softmax`.

- **QNN / Hexagon: bidirectional `QMatMul` ↔ QNN bridges.**
  Host INT8 accumulate mixes with QNN in either direction:
  `QMatMul → Dequantize → Relu`, and
  `Quantize → QMatMul → Dequantize → Relu` (pre-session → host → post-session).

- **QNN / Hexagon: on-device int4 weights** (BW_SCALE_OFFSET bitwidth=4).
  IR may store tightly packed nibbles (`(K·N+1)/2` bytes); the runtime unpacks
  to 1 byte/elem because `libQnnCpu` rejects native `SFIXED_POINT_4`
  (`UNSUPPORTED_TENSOR_PARAM`). Dequantize → f32 MatMul on device.

- **QNN / Hexagon: on-device `MatMul(I8, I8)` lower.**
  `Quantize(x)` × STATIC I8 weight without Dequantize in IR. Runtime inserts
  Dequantize on both operands → f32 MatMul (libQnnCpu rejects direct
  sfixed8×sfixed8 with `0xc26`; HTP prepare accepts but execute fails on
  MatMul_bias). Host `Op::QMatMul` remains the fully-quantized path.

- **QNN / Hexagon: `Op::QMatMul` (fully quantized INT8, no f32 weight bake).**
  Integer accumulate + requantize on the host (same kernel as `rlx-cpu`);
  weights stay I8. Plus multi-op codegen `Linear` / `LinearRelu` for the
  offline `qnn-net-run` path.

- **QNN / Hexagon: persistent session + context-binary save/load (M3).**
  Shim `rlx_qnn_session_*` finalizes once and reuses across `run`s. Export via
  `contextGetBinary` / reload via `createFromBinary` + `libQnnSystem` metadata.
  `QnnExecutable::{export_context_binary,reload_from_context_binary}`.

- **QNN / Hexagon: on-device int8 MatMul weights** (`MatMul(x, Dequantize(I8_w))`).
  I8 `Param`/`Constant` weights stay STATIC `SFIXED_POINT_8`; QNN runs
  Dequantize → f32 MatMul (mixed f32×int8 MatMul is rejected by the CPU
  backend). Shim `clientBuf.dataSize` uses 1 byte/elem for sfixed8.
  `set_param_typed(I8)` fills deferred weights. FFI + Session parity vs CPU.

- **QNN / Hexagon: host-dequant `DequantMatMul` (GGUF → f32 MatMul).**
  Packed GGUF weights (`Q8_0`/`Q4_0`/K/IQ/…) decode on the host (same posture
  as CoreML's off-device path), transpose `[N,K]→[K,N]`, and run a plain QNN
  MatMul. `set_param_typed(U8)` fills deferred weights. Plus earlier FAB /
  Expand / Silu / Custom Attention work; `just qnn-ffi` validates on Linux.

- **QNN / Hexagon: `FusedAttentionBlock`, Expand, Silu, Custom rank-4 Attention.**
  `QnnBackend` now claims `SUPPORTED_OPS` (legalize) and runs
  `unfuse_attention_block` before FFI lower. New lowers: `Expand` (Reshape+Tile),
  `Silu` (Sigmoid·x), rank-4 `Attention` with additive `MaskKind::Custom` (FAB
  shape), and NeoX RoPE broadcast from compact `[S,D/2]` tables. Parity via
  `fused_attention_block_parity` (`Device::Hexagon`) and `just qnn-ffi`
  (native Linux, no Docker). Shim includes `<stdlib.h>` for newer gcc.

- **Static graph checker (`rlx_runtime::check`) + `cargo rlx check` + `#[rlx_model(check)]`.**
  Folds the analyses rlx already computes — shape/dtype verification (`verify_all`),
  backend dispatch (native / common-IR / unsupported via each backend's real op
  claim), missed fusions (`FusionReport.missed` with fix hints), and the
  provable-NaN/Inf lint (`lint_numerics`) — into one structured `CheckReport`
  (human or `--json`). Fusion/shape/numeric run with no GPU or driver; CPU
  execution legality is always available, other backends' legality is opt-in
  behind per-backend features (op claim read driver-free from the registry —
  factories build unit structs, `supported_ops()` is a const).
  - The checker lives in `rlx_runtime::check::check_graph` (single source of
    truth, no new deps for consumers).
  - **`#[rlx_model(check)]`** injects a `model_self_check(&graph)` call right
    after tracing, so building a model surfaces findings on stderr — tuned by
    `RLX_CHECK` (`off` / `all` / `strict`). The generated code already routes
    through `::rlx_runtime`, so no extra dependency is needed.
  - **`cargo rlx check`** (`crates/tooling/rlx-check`): a `cargo-rlx` subcommand
    over the same `check_graph`, with built-in `--demo` graphs, `--json`, backend
    filtering, and `just check-graph` / `just install-check`. Non-zero exit on any
    error-level finding for CI/pre-commit gating.

- **Vulkan / OneAPI packed DiT reverse SPIR-V.** Dedicated
  `ada_layer_norm_backward` / `gated_residual_backward` compute kernels
  (Vulkan GLSL via naga; OneAPI OpenCL-C via `ocloc` when
  `RLX_ONEAPI_BUILD_KERNELS=1`). `compile_rng` no longer decomposes these
  ops; OneAPI host-fallbacks through `rlx-cpu` when kernels are not embedded.

- **CoreML ANE native MIL for DiT modulation forward (`AdaLayerNorm` /
  `GatedResidual`).** Composed lowering (implicit broadcast, no `Expand`) in
  `rlx_coreml::mil`; `compile_with_options` no longer calls
  `unfuse_dit_modulation`. Host-portable `mil_lower` tests; op-coverage ANE ✅.

- **DiT modulation ops on all backends (`Op::AdaLayerNorm` /
  `Op::GatedResidual`).** Claimed in every `*_SUPPORTED_OPS` set and fused via
  `FuseAdaLayerNorm` / `FuseGatedResidual` on all fusion targets. Native
  kernels: CPU, Metal (MSL), CUDA/ROCm (shared `.cu` in `rlx-gpu-kernels`),
  wgpu (WGSL). Composed: MLX (`ops::layer_norm` / `rms_norm` + broadcast),
  TPU (HLO). Claim-then-`unfuse_dit_modulation` (broadcast Mul/Add, no Expand):
  Vulkan, OneAPI, WebGL (CoreML ANE uses native composed MIL for forward +
  packed reverse). Shared `rlx_ir::ada_modulation_lead_pack` for `[B,1,D]`
  broadcast metadata. Parity: CPU Session + Metal vs CPU.

- **DiT modulation autodiff (fused + unfuse).** Default AD keeps
  `AdaLayerNorm` / `GatedResidual` fused and emits packed
  `AdaLayerNormBackward` / `GatedResidualBackward` (`[dx ∥ dscale ∥ dshift]` /
  `[dx ∥ dy ∥ dgate]`), avoiding Expand of `[B,1,D]` modulation in the
  backward graph. Native reverse kernels on CPU, Metal, CUDA/ROCm (shared
  `.cu`), and wgpu; CoreML ANE uses native composed MIL for packed reverse
  (`COREML_NATIVE_BACKWARD_OPS`); MLX and TPU use native composed lowering in
  `lower.rs`; Vulkan / OneAPI ship dedicated SPIR-V kernels
  (`ada_layer_norm_backward` / `gated_residual_backward`).
  Import-shaped LN+Expand+Mul/Add graphs fuse
  via `FuseAdaLayerNorm` / `FuseGatedResidual` under Session (FLUX adaLN-Zero +
  F5 identity/RMS fixtures in `dit_import_fuse`). Exact JVP for
  LayerNorm / RmsNorm / Ada (full mean/var / RMS pushforward). Microbench:
  `rlx-bench` `bench_dit_modulation`. Flow: `dit_ada_gated_linear` train step.
  The unfuse-for-AD path remains available through
  `rlx_fusion::unfuse_dit_modulation` before `grad_with_loss` (primitive
  VJPs). JVP rules for both forward ops; `vmap` auto-unfuses then lifts
  primitives. FD coverage for fused/unfused reverse, decompose_backward, JVP,
  and vmap; Metal/CUDA/wgpu/ROCm/MLX↔CPU grad parity; tiny DiT-block train
  step on CPU.

- **DiT ONNX adaLN fixture + multi-block train.** Real ONNX `dit_adaln.onnx`
  (affine-free `LayerNormalization` + `Expand` modulation) imports strictly and
  fuses to `AdaLayerNorm` after param specialization; 3-block FLUX-style train
  graph asserts six `AdaLayerNormBackward` / six `GatedResidualBackward` ops.
  `bench_dit_modulation` median numbers recorded in `docs/op-coverage.md`.

- **Differentiable symmetric eigendecomposition (`Op::Eigh` / `Op::EighBatch`).**
  First-class `eigh` as a graph op — `A [n,n] → (λ [n] ascending, U [n,n]`
  columns = eigenvectors, `A = U diag(λ) Uᵀ)`, packed `[λ ∥ U]` internally —
  plus the batched `[B,n,n]` form. Full reverse mode via `Op::EighBackward` /
  `Op::EighBatchBackward` (the standard symmetric-eigendecomposition adjoint
  `Ā = sym(U (diag(λ̄) + F∘(Uᵀ Ū)) Uᵀ)`, `F_ij = 1/(λ_j−λ_i)`, degeneracy-
  guarded). Builders `Graph::eigh` / `Graph::eigh_batch`; VJP wired in autodiff.
  Runs on **all backends**: CPU-native (LAPACK `dsyevd`, f64) and F64 CPU
  host-fallback on Metal / CUDA / ROCm / wgpu / Vulkan / oneAPI (`is_spd_host`
  + every `*_SUPPORTED_OPS` list). This is the differentiable primitive the SPD
  spectral functions build on, and the seam a native batched eigensolver
  (cuSOLVER `syevjBatched` &c.) plugs into. Verified: forward `A = U diag(λ) Uᵀ`
  reconstruction, VJP finite-difference checked (incl. the eigenvector gradient,
  sign-aligned), the exact `Σλ = trace ⇒ ∂/∂A = I` end-to-end gradient through
  `Session`, and gradient parity on real CUDA hardware (bit-exact vs CPU).
- **Native CUDA batched eigensolver (`Step::EighNative`, cuSOLVER
  `SsyevjBatched`).** `Op::Eigh` / `Op::EighBatch` with `n ≤ 32` now run
  **on-device** on the CUDA backend via cuSOLVER's batched Jacobi eigensolver —
  no D2H→CPU→H2D round-trip. An `eigh_assemble` NVRTC kernel transposes
  cuSOLVER's column-major eigenvectors into the packed `[λ ∥ U]` layout and
  interleaves the eigenvalues; `n > 32` falls back to the CPU host path. f32
  (matches the widened SPD arena; f32 Jacobi ≈ f64 LAPACK to cos≈1.0 on these
  small SPD blocks). Measured on an NVIDIA GPU: ~0.5 µs/matrix at n=32,
  batch≥4096 (a single kernel launch for the whole batch) vs ~7.6 µs on the
  rayon CPU path. Requires cudarc's `cusolver` feature (bumped to 0.19.8 for the
  CUDA-13 cuSOLVER symbol set). Verified on CUDA hardware: native forward
  reconstruction (single + batched) and gradient parity vs CPU; full rlx-cuda
  suite green.
- **Native ROCm batched eigensolver (`Step::EighNative`, hipSOLVER
  `SsyevjBatched`).** Same on-device path for AMD: `Op::Eigh` / `Op::EighBatch`
  with `n ≤ 32` when `libhipsolver` is loadable. Hand-rolled libloading shim
  (`hipsolver.rs`) + hipRTC `eigh_assemble` kernel mirroring the CUDA layout.
  Missing hipSOLVER or `n > 32` keeps the existing CPU host-fallback.

- **SPD-manifold Riemannian primitives — host helpers + first-class ops on
  every backend.** `rlx_cpu::spd` gains `karcher_mean_weighted` (the true AIRM
  barycentre `argmin_M Σ wᵢ δ²(M, Cᵢ)` — what a barycentric-OT projection /
  soft-clustering / weighted-MDM needs, replacing the log-Euclidean
  `expm(Σ w̄ᵢ logm Cᵢ)` shortcut), the arbitrary-base `log_map` / `exp_map` and
  `parallel_transport` (AIRM `Log_P`/`Exp_P` and the isometric transport of
  Yair et al. 2019 — `geodesic_interp(A,B,t)` is now just
  `exp_map(A, t·log_map(A,B))`), and rayon-batched `logm_batch` / `expm_batch` /
  `sqrtm_batch` / `invsqrtm_batch`. These are also promoted to graph ops
  `Op::SpdKarcherMeanWeighted` / `SpdLogMap` / `SpdExpMap` /
  `SpdParallelTransport` / `SpdMatrixFnBatch { kind }` (builders
  `Graph::spd_karcher_mean_weighted` / `spd_log_map` / `spd_exp_map` /
  `spd_parallel_transport` / `spd_{logm,expm,sqrtm,invsqrtm}_batch`), so they run
  on **all backends**: CPU-native (F64), and F64 CPU host-fallback on Metal /
  CUDA / ROCm / wgpu / Vulkan / oneAPI (the same eigendecomposition-free
  host-delegation path as the existing SPDNet ops — claimed in every
  `*_SUPPORTED_OPS` list + `is_spd_host`). **Fully differentiable**: the maps
  carry analytic Riemannian VJPs (`SpdLogMapBackward` / `SpdExpMapBackward` /
  `SpdParallelTransportBackward` / `SpdMatrixFnBatchBackward`, packed per-input
  gradients) — the base point is differentiated too, so a learned base trains
  correctly, not just the moving argument (`SpdKarcherMean{,Weighted}` stay
  detached statistics). Verified: weighted-barycentre stationarity, `exp∘log`
  round-trip, transport isometry, batch-vs-scalar parity, and every VJP
  finite-difference checked (rlx-cpu unit tests); end-to-end gradients through
  `Session` on CPU; forward parity on Metal / wgpu / Vulkan / oneAPI /
  CUDA(hardware) / ROCm; and a gradient parity check on the GPU host path (wgpu).
- **First-class `ScatterElements` / `GatherNd` / `GatherElements`.** ONNX
  import emits `Op::ScatterElements { axis, reduction }`,
  `Op::GatherNd { batch_dims }`, and `Op::GatherElements { axis }` (no longer
  Custom / flatten decompositions). CPU thunks are the reference; Metal /
  wgpu / CUDA / ROCm / Vulkan / MLX host-delegate; CoreML hybrid host; TPU
  XLA gather/scatter. Autodiff VJP for all three (plus existing ScatterNd);
  finite-diff checked.
- **First-class `Op::ScatterNd`.** ONNX ScatterND lowers to
  `Op::ScatterNd { reduction }` (none/add/mul/max/min) instead of
  `Op::Custom("onnx.ScatterND")`. CPU thunk is the reference; Metal /
  wgpu / CUDA / ROCm / Vulkan / MLX host-delegate via `HostOpDesc`;
  CoreML MIL `scatter_nd`; TPU XLA scatter. Claimed in every runtime
  `*_SUPPORTED_OPS` list. Autodiff VJP covers all five reductions
  (finite-diff checked).
- **FPGA export legalizer + DX.** `ExportQuantMode::{Int8,Int4,Fp4}`,
  `prepare_model` / `LegalizeOptions`, split requant side tables
  (`*_requant_m0` / `*_requant_shift`), `board_top.sv` shells, bias-free
  Dense / logits-only output, `ExportSession`, and `pyrlx.export_fpga`
  (feature `fpga`). Soft scalar **sidebands** (`SidebandSpec`,
  `bind_sideband_inputs`, CLI `--sideband`). See
  [`docs/fpga-export.md`](docs/fpga-export.md).
- **First-class FPGA / SystemVerilog export.** `Model::from_graph`,
  `FpgaExportConfig` + `HwTarget` (default **target-agnostic** soft-port RTL;
  optional ECP5 / iCE40 / Xilinx7 synth scripts), `rlx_fpga::export_graph`,
  and `rlx_runtime::export` (`ExportTarget::Fpga`, feature `fpga`).
  Docs: [`docs/fpga-export.md`](docs/fpga-export.md). Recipes: `just fpga-emit`,
  `just test-fpga`. `rlx` feature `edge` now includes `fpga`.
- **Unified `Op::Scan` host contract across GPU backends.** Long scans that survive
  legalize run through shared `rlx-cpu` macros (`rlx_scan_host_desc!`,
  `rlx_execute_scan_on_bytes!`, `rlx_scan_stage_d2h!`, `rlx_maybe_unroll_scans!`)
  and `ScanHostDesc` / `run_scan_packed_f32` — Metal / CUDA / ROCm / wgpu /
  Vulkan / OneAPI / CoreML / MLX. Short scans prefer on-device IR via
  `CompileOptions::scan_unroll_max_length` (default **64**) plus
  `maybe_unroll_scans_budget(4096)` (`length × body_nodes`).
- **`Op::ScanBackward` / `ScanBackwardXs` GPU host path** via shared
  `HostOpDesc` (byte offsets) + macros (`rlx_host_op_desc!`,
  `rlx_execute_host_op_on_bytes!`, `rlx_host_op_stage_d2h!`) and value-map
  helpers `run_scan_node_f32` / `run_host_op_node_f32` — Metal / CUDA / ROCm /
  wgpu / Vulkan / CoreML / MLX / OneAPI. Discrete GPUs also share
  `rlx_arena_stage_d2h!` for Scan / HostOp / Spd; wgpu uses `HostOpSpan`.
  Parity: `scan_backward_parity`.
- **MLX `RLX_MLX_PROFILE`:** per-op-kind wall-time dump on executable drop
  (restored after lower.rs host-path work).
- **TPU `QuantScheme::GgufQ1_0`** host dequant at HLO emit
  (`rlx_gguf::q1_dequant::dequant_q1_0`).

### Changed

- **De-duplicated GPU-backend infrastructure into two shared crates.** Three
  copy-pasted-across-backends surfaces now live once:
  - **`rlx-gpu-host`** — host-fallback staging (`D2H → CPU → H2D`) for ops with
    no native kernel (RNN/SSM, im2col, deformable attention, UMAP kNN, log-mel,
    welch-peaks, splat, rms/rope/cumsum/gather backward, Scan/HostOp/indexing,
    RNG fill, SPD manifold eval, GGUF dequant-matmul CPU fallback, maxpool/conv
    training compact-scratch, …). Ops that were duplicated across GPU backends
    are now single-source generic fns over a `DeviceArena` staging trait; each
    backend keeps a ~15-line adapter (`CudaArena`/`RocmArena`/`WgpuArena`) and
    thin per-op wrappers, so call sites are unchanged. SPD `eval` /
    `is_spd_host` are re-exported from Metal/Vulkan/oneAPI/CUDA/ROCm/wgpu.
    Vision/RNN/collective host paths and compact conv/pool training also live
    here. `device_report` rows include advisory `capabilities`; `just new-op`
    prints the AGENTS checklist. ONNX custom-op f32-slot dtype bridging and
    SAM debug hosts also share `rlx-gpu-host`. Verified: CUDA
    `gated_delta_net` / `welch_peaks` parity on an NVIDIA GPU + CPU
    equivalence guard tests.
  - **`rlx-unfuse`** — the IR-level unfuse/decompose pass (`FusedAttentionBlock`,
    `FusedTransformerLayer`, `FusedSwiGLU`, `LoraMatMul`, `DotGeneral`, rank-3
    `Attention`, control flow) shared by `rlx-cuda`/`rlx-rocm`/`rlx-wgpu` behind
    a per-backend `DecomposePolicy` (capability flags). The ~80%-identical logic
    lives once; the CUDA-native-FAB gate, attention-backward promotion, and
    wgpu's fused-op/rank-3 variants are policy flags. Behavior-preserving
    (safety-critical gates confirmed byte-identical; wgpu decompose parity green
    on Metal, CUDA attention parity green on the NVIDIA GPU).
- **Split `rlx-cuda` `backend.rs` (12.5k lines) into a `backend/` module dir**
  (`mod`/`run`/`compile`/`set`/`fill`/`output`), mirroring `rlx-rocm`. Pure code
  move — all methods preserved, public API unchanged.
- **Peel mega backend / lower files into navigable modules** (pure moves; public
  APIs unchanged):
  - **Metal** — `backend/encode/{mod,ops}.rs` (`encode_and_run` + `encode_*`
    helpers); `backend/mod.rs` keeps `MetalExecutable`.
  - **CUDA / ROCm** — `backend/{step,helpers}.rs` (+ CUDA `bwd_launch.rs`);
    `Step` / op-id / schedule helpers off `mod.rs`.
  - **wgpu** — `backend/{step,helpers}.rs`; `compile/{mod,lower}.rs` (match body
    in `lower.rs`).
  - **MLX** — `lower/{mod,env,subgraph,helpers}.rs` (`lower_with_env` match in
    `env.rs`).
- **CPU `RLX_FAST_CONV` defaults on.** Forward Conv2D uses im2col+BLAS (and
  rayon fans for pool / elementwise / bwd) unless `RLX_FAST_CONV=0`. Fixes
  orders-of-magnitude-slow MNIST CNN training that previously fell through to
  the scalar nested-loop kernel. When the fast path is on, OpenBLAS/MKL/
  Accelerate inner threads are capped to 1 (unless the user already set
  `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / …) so Rayon outer parallelism
  does not nest into an N² thread storm. Cortex-M trainer labels stay correct
  (`rlx-fused` by default; set `=0` for the unfused `rlx` bar).
- **Vulkan / oneAPI claim `OpKind::Custom`.** In-graph collectives
  (`collective.all_reduce`, …) used by `rlx-vision-bench` data-parallel graphs
  no longer fail Session legalize on Vulkan; any Custom routes to the existing
  host-fallback. `rlx_oneapi::is_available()` is now always true (CPU reference
  when no Level Zero GPU / kernels) so `RLX_DEVICE=oneapi` works without
  `ze_intel_gpu`. OneAPI `host::eval` now returns dtype-aware `HostOut`
  (Bool/`U8`/`I8` as bytes) so SoftmaxCE→`Compare`→`Where` no longer panics
  reading a 1-byte mask as f32. `CompilePipeline::backend_label` keeps Vulkan /
  OneAPI legalize errors from mislabeling as `"wgpu"`.
- Clippy cleanup: drop unused `CnnGeom::final_hw` / `ident_mat`; split Vulkan
  instance-ext cfg so non-Apple builds don't trip `unused_mut`.
- Codegen backends (QNN / Cerebras) and TPU rewrite route nested Scan through
  `LowerScan` / `LowerControlFlow` so arbitrary-body Scan does not need a
  device body-ISA interpreter.

## [0.2.12] — 2026-07-06

### Added

- **FIR / RIR / IIR digital filters across every backend.** New `Graph` builders
  that compose the existing FFT + elementwise + `Op::Scan` primitives, so they
  lower on all backends with **no new kernels**:
  - **FIR** — `fir_conv1d(x, taps, FirMode)` with `Full`/`Same`/`Valid`/`Causal`
    output modes; `≤ 64` taps use an exact direct time-domain shift-and-add,
    longer filters the FFT convolution theorem (auto-selected).
  - **RIR / convolution reverb** — `partitioned_conv1d(x, ir, block)` and
    `conv_reverb(x, &ir, block)` implement **uniform-partitioned overlap-save
    (UPOLS)**: the impulse response is split into `⌈M/B⌉` partitions summed in a
    frequency-domain delay line, keeping every FFT a fixed `2B` points so long
    room impulse responses stay on the native Metal/WGPU FFT kernels instead of
    one giant `L+M−1` transform.
  - **IIR** — `iirfilt(x, b, a)` (arbitrary-order Direct-Form-II-Transposed via
    `Op::Scan`; native on CPU/MLX, host-fallback on Metal/wgpu) plus
    `iir_as_fir(..)` / `iir_impulse_response(..)`, which reduce a stable IIR to a
    truncated impulse response applied as an FIR so IIR runs natively on **every**
    backend.
  - **Fused `Op::PartitionedConv`** — a first-class op for the partitioned
    convolution. `partitioned_conv(x, ir, block)` builds it; it decomposes (via
    the shared `unfuse` rewrite, before any backend sees it — no per-backend
    kernel) into a **batched complex matmul over the partition axis**
    (`partitioned_conv1d_gemm`), routing the frequency-domain delay line through
    the native batched-GEMM kernels (cuBLAS / rocBLAS / MPS) rather than `P`
    elementwise multiply-accumulates.
  - Validated: CPU numeric (vs direct-convolution / DF2T references) and
    CPU-parity on **Metal, MLX, wgpu, CUDA, and Vulkan** — the last two on a
    discrete NVIDIA GPU (incl. `Op::PartitionedConv`); CoreML compile-checked.
    (Vulkan required the native-FFT kernel below; before it, the FFT filters
    segfaulted on discrete GPUs via host-fallback.)
  - Fixed a latent **batched `irfft`** bug uncovered by the framed convolution:
    the Hermitian mirror reversed the *flattened* last axis via `Op::Gather`,
    corrupting every batch row but the first (any rank ≥ 2 `rfft`/`irfft`,
    including multi-channel `fft_conv1d`); it now uses the batch-general
    `Op::Reverse`, which every backend lowers.
- **Native Vulkan FFT kernel (`Op::Fft` on-device).** Vulkan previously ran
  `Op::Fft` via CPU host-fallback, which **crashes on discrete GPUs** (it
  assumes a host-visible mapped arena the CPU can read the op's inputs from).
  A new `fft` kernel (radix-2 Cooley-Tukey, one workgroup of 256 threads per
  batch row, shared-memory butterflies) runs the forward/inverse f32 FFT
  on-device for power-of-two `n ≤ 1024` (larger `n` / non-f32 still fall back to
  host); dispatch mirrors the wgpu native-FFT guard. This fixes the discrete-GPU
  crash and lets the FIR/RIR/IIR filters run on discrete Vulkan (validated on an
  NVIDIA GPU). Like `matmul_tiled`/`matmul_coop`, it is **precompiled offline
  with glslang** to `shaders/precompiled/fft.spv` — naga's GLSL frontend has no
  `memoryBarrierShared`/`groupMemoryBarrier` and its bare `barrier()` doesn't
  enforce cross-subgroup shared visibility on NVIDIA, so the shared-memory
  stages would race. Also adds an opt-in `RLX_VULKAN_VALIDATION=1` env to enable
  the Khronos validation layer.
- **Parameterized `fNeXmY` minifloats (`ScaledFormat::Custom`).** Beyond the
  seven named tensor-core formats (FP8 e4m3/e5m2 + FNUZ, FP6 e2m3/e3m2, FP4
  e2m1), `ScaledFormat` now carries a `Custom { exp_bits, mant_bits, bias }`
  variant for an arbitrary all-finite minifloat whose whole code fits in a byte
  (`1 + exp + mant ≤ 8`). Build one with `ScaledFormat::custom(exp, mant)` (IEEE
  bias `2^(exp-1)-1`), `custom_with_bias(..)`, or `"f4e3m0".parse()`; `Display`
  round-trips the `fNeXmY` name. Example — **`f4e3m0`** (3 exp, 0 mant): a signed
  power-of-two 4-bit grid, `±0` and `±{0.25, 0.5, 1, 2, 4, 8, 16}`.
  - The f32↔code codec (`lowp_codec`) was already generic over
    `(exp, mant, bias)`, so `ScaledQuantScale` / `ScaledQuantize` / `ScaledMatMul`
    / `ScaledDequantize` accept a `Custom` format with **no new kernel** on **CPU**
    and **Metal** (host decode-and-accumulate reference). Bit-exact CPU round-trip
    test on the `f4e3m0` grid; codec parity tests (a `Custom{2,1}` decodes
    identically to the named `F4E2M1`). **Metal hardware-validated** (Apple GPU):
    `f4e3m0` is bit-for-bit == the CPU oracle (`max_abs = 0`), cosine-vs-f32 0.998.
  - **CUDA / ROCm**: the decode kernel (`scaled_lowp_general.cu`) gains a generic
    path — `kernel_id()` packs `(exp, mant, bias)` into the `fmt` word with a
    top-bit sentinel, which the kernel unpacks and decodes generically. The seven
    named ids (`0..=6`) keep the existing `switch`, so the hardware FP8 path is
    byte-for-byte unchanged. No hardware tensor core exists for these research
    formats — they always take the decode fallback.
    - **Hardware-validated on CUDA** (NVIDIA GPU, NVRTC,
      `crates/backends/rlx-cuda/tests/cuda_scaled_custom.rs`): `f4e3m0` grid GEMM
      **bit-exact vs f32** (`max_abs = 0`), mx-block cosine 0.998, multi-tile
      (37×80×45) cosine 0.998; and a **12-format sweep** (6 custom splits + native
      fp8 + fnuz/fp6/fp4) where on-device quantize→dequantize is **bit-for-bit ==
      the CPU oracle** for every format.
    - The decode GEMM (`scaled_matmul_decode`) is now **shared-memory tiled**
      (16×16 — each code decoded once per tile instead of once per output
      element): **~5.4× faster** on the NVIDIA GPU at 1024³ (74.9 → 405 GFLOP/s),
      same launch config, correctness unchanged.
    - **ROCm/HIP**: the shared kernels compile cleanly under `hipcc` for a real
      AMD target (gfx90a / CDNA2); on-device run pending AMD hardware (none
      reachable here — the rig is NVIDIA).
    - Fixed two latent NVRTC bugs uncovered by these first on-device runs (the
      scaled kernels had never actually NVRTC-compiled before): (1) the decode
      kernel used the `INFINITY` macro, undefined under NVRTC → now
      `__int_as_float` (`#ifndef`-guarded, so nvcc/hipcc unaffected); (2) the
      native per-tensor fp8 quantize kernel did `#include <cuda_fp8.h>` /
      `<hip/hip_fp8.h>` (no NVRTC/hipRTC include path) → the f32→fp8 conversion is
      now closed-form, matching the oracle bit-for-bit, removing the toolkit-header
      dependency entirely.
  - **Vulkan**: the four scaled ops are now wired into `rlx-vulkan` as CPU
    host-fallbacks (the same `rlx-cpu` oracle Metal uses) against the mapped
    host-visible arena — added to `SUPPORTED_OPS` + `is_host_fallback`, and the
    generic host path now writes **U8 outputs** (quant codes / block scales) as
    raw bytes instead of reinterpreting them as f32. **Validated on a native
    NVIDIA Vulkan driver** (NVIDIA GPU): `f4e3m0` grid GEMM bit-exact vs f32,
    mx-block cosine 0.998 (`tests/vulkan_scaled_custom.rs`). (First native-driver
    Vulkan validation of any scaled path.)

- **Pick a minifloat format at the high level** — no more hand-wiring the
  `ScaledQuantScale → ScaledQuantize → ScaledMatMul` chain. The same
  `ScaledFormat` (any named format or a parameterized `Custom` like `f4e3m0`)
  now flows through every composition/execution surface:
  - **Compose ops** — `Graph::scaled_matmul(lhs, rhs, fmt, layout)` (+
    `scaled_quantize` / `scaled_dequantize` / `scaled_matmul_bias`) on the
    low-level graph, and the mirror `HirGraphExt::scaled_matmul` on the HIR
    builder. One call emits the whole quantize→GEMM chain.
  - **Tensor DSL** — `Tensor::scaled_matmul(&rhs, fmt, layout)` on the lazy
    `rlx-tensor` tensors (e.g. `a.scaled_matmul(&w, ScaledFormat::custom(3,0),
    ScaleLayout::mx())`).
  - **Execute a flow** — `CompileOptions::scaled_quant(ScaledQuantConfig{..})`
    (re-exported as `rlx_runtime::ScaledQuantConfig`); `Session::compile_with`
    runs the existing `insert_scaled_matmul` pass so *every* 2-D matmul in a
    graph is rewritten to the chosen format at compile time.
  - **Python** — `pyrlx.Graph.scaled_matmul(lhs, rhs, format="f4e3m0",
    layout="mx")` parses the `fNeXmY` name (via `ScaledFormat: FromStr`) and
    rejects invalid splits with a clear `ValueError`.
  - Verified end-to-end on CPU at every layer (builder, Session policy, Tensor
    DSL, and a pyrlx→CPU run: `f4e3m0` cosine 0.997 vs numpy).

- **`ScaledFormat` Rust API / DX.** Working with the different float formats is
  now ergonomic without reaching for the free-function codec or tuple `fields()`:
  - `const` constructors — `ScaledFormat::custom(3, 0)` / `custom_with_bias(..)`
    are `const fn`, usable in `const` items and pattern-free construction.
  - Introspection accessors — `exp_bits()`, `mant_bits()`, `bias()`,
    `is_custom()`, `is_named()`, and a `ScaledFormat::NAMED: [_; 7]` array for
    format sweeps (all `const`).
  - Codec methods on the format itself — `fmt.decode(code)`, `fmt.encode(x)`,
    `fmt.quantize(x)` (round-trip a single f32 to its nearest representable
    value, e.g. `custom(3,0).quantize(1.4) == 1.0`), and
    `fmt.representable_values()` to inspect a format's whole grid.
  - `ScaleLayout: FromStr` (`"mx"`, `"nvfp4"`, `"per_tensor"`, `"mx/<block>"`)
    to match `ScaledFormat: FromStr` (`"f4e3m0"`); the pyrlx layout arg now
    delegates to it (one source of truth). Doc-tested.

- **Low-precision `encode(±inf)` now saturates to `±max_finite`** (`lowp_codec`
  + the GPU `rlx_encode_lowp`), matching the codec's documented contract — it
  previously returned code `0` because every finite candidate is equidistant from
  a huge value in f64. Applies to all formats, incl. E5M2 (whose inf code is never
  emitted). CPU + GPU kept in lockstep.

- **`rlx-torch-import` diffusion-model op coverage.** New aten→rlx lowerings so
  UNet / DiT image models import cleanly:
  - `group_norm` / `native_group_norm` → `Op::GroupNorm`.
  - `upsample_nearest2d` / `_upsample_nearest_exact2d` (`.default` + `.vec`, any
    2ⁿ× scale) → chained `Op::ResizeNearest2x`.
  - `constant_pad_nd` (constant mode) → `concat` with `Op::Constant` fills.
  - `baddbmm` → `beta·input + alpha·(b1@b2)` (the non-SDPA attention score path;
    `beta = 0` fast path drops the bias).
  - `split.Tensor` / `chunk` → narrow tuples (GEGLU / adaLN modulation).
  - `zeros` / `ones` / `empty` (+ `_like` / `new_`) → `Op::Constant`.
  - `pixel_shuffle` / `pixel_unshuffle` → reshape + permute.
  - `_scaled_dot_product_{flash,efficient,math}_attention` → same `Op::Attention`
    as the public op (routes each overload's own arg layout; `decomposition=core`).
  - `leaky_relu`, `hardswish`, `hardsigmoid`, `hardtanh` → clamp/mul decompositions.
  - `masked_fill` → `x + mask·(value − x)` (arithmetic; avoids bool-cond `Where`
    on the f32 arena).
  - `upsample_{bilinear,bicubic}2d` and their antialiased `_aa` overloads (any
    output size, `align_corners` either way) via new
    `HirMut::resize_{bilinear,bicubic}2d[_aa]` builders — every filter is
    separable, so they lower to two constant-matrix interpolation matmuls
    (`MatMul`/`reshape`/`transpose`). Bit-exact PyTorch parity on **every backend
    with no new kernel** (verified CPU + Metal + MLX on Apple).

  Unblocks Stable Diffusion / SDXL (UNet + VAE) and Sana (linear-attention
  ones-padding). 18 exact-parity CPU tests (+ Metal/MLX resize parity) + README.

- **`rlx-torch-import` dynamic shapes.** A model exported with
  `torch.export(dynamic_shapes=…)` now imports with symbolic input dims instead of
  raising. The front-end (`pyrlx.from_torch(..., dynamic_shapes=…)`) emits a
  per-axis `dynamic` marker (stable symbol id per `SymInt`); the importer builds
  `Dim::Dynamic(sym)` inputs (`InputDef.dyn_dims` / `hir_shape`), the compile pass
  re-infers the symbolic graph, and `DimBinding` specializes it per run — so a
  model is imported **once** and run at any batch/seq (`run_dynamic`; `verify`
  binds the reference shape). End-to-end verified (Linear+GELU with dynamic batch
  imports + parity), plus a Rust test running one dynamic-batch HIR at batch 2 & 4.
- **CPU interpreter `Op::Expand` now broadcasts.** The executor treated `Expand`
  like `Reshape` (a plain `input.len()` copy), leaving stride-0 axes unfilled — so
  any imported broadcast (e.g. a `torch.eye` built from `arange`/`eq`) produced
  garbage. Now does a proper strided broadcast walk.
- **Importer `sum`/`mean` with an empty dim list reduce all dims.** aten's
  `sum.dim_IntList(x, [])` / `mean.dim(x, [])` — how bare `Tensor.sum()`/`.mean()`
  decompose — reduce over *every* dim (→ scalar); the importer reduced *nothing*
  (returned the input), so e.g. `torch.eye(n) + x.sum()*0` left a rank-2 tensor and
  broke the downstream broadcast. Now an empty (or absent) dim list reduces all axes.
- **Importer int/bool intermediates on the f32 arena.** A byte-sized
  `arange`/compare *intermediate* was mis-read (the f32-uniform arena under-sizes a
  bool node — `9 bytes → 2 f32 slots` — so a comparison feeding a matmul went OOB).
  Fixes: integer `arange` is materialized F32 (values exact); comparisons emit
  `Compare → Cast(F32)` so the consumed value is a properly-sized f32 `{0,1}`
  tensor (same pattern as grid_sample); non-float node dtypes are tracked as F32.
  The `torch.eye` (`arange`/`eq`/`mm`) identity now imports with exact parity — the
  case that surfaced this.
- **`pyrlx.from_torch` auto-decompose fallback.** When the Rust importer reports
  ops the RLX registry doesn't cover, the front-end re-exports with those ops
  decomposed (via torch's full decomposition registry) and retries, up to
  `max_decompose_rounds` — so torch-decomposable ops are handled automatically
  (reported in `summary["auto_decomposed_ops"]`). Decompositions that emit
  `prims.*` (breaking functionalization) are bisected out one at a time, so a bad
  op degrades to a clean "unsupported op" report instead of a crash. On by default
  (`auto_decompose=False` to disable). Pure-logic tested in
  `tests/test_torch_import_autodecomp.py`.
- **`grid_sampler` / `grid_sampler_2d` — all variations.** New
  `HirMut::grid_sample2d` decomposes `grid_sample` into universal ops (transpose →
  axis-0 `Gather` + `Round`-based floor + arithmetic weights, batch unrolled):
  every interpolation mode (nearest / bilinear / bicubic) × padding
  (zeros / border / reflection) × `align_corners`. Exact PyTorch parity verified
  on **CPU, Metal, and MLX** for all mode/padding combos. Also fixed the CPU
  interpreter `Op::Gather` general-axis path (previously silently zeroed for
  `axis != 0`).
- **rlx-mlx fusion cap.** The deep grid_sample decomposition made `mlx::compile`
  fuse an elementwise region into one Metal kernel that exhausted the
  argument-buffer limit. Fixed with (a) eval barriers in the Lazy lowering
  (`lower_with_env`) that materialize a fusable chain every `RLX_MLX_FUSE_CAP`
  (default 12) ops — non-elementwise ops reset the counter, so ordinary models
  are untouched, and it's disabled inside the `mlx::compile` trace; and (b) a
  retry in `run_read_outputs` that, on an over-fused-kernel failure, disables
  compile and re-runs in barrier'd Lazy.

- **`rlx-ir::HirMut::resize_{bilinear,bicubic}2d[_aa]`.** Separable NCHW
  bilinear/bicubic resize built from universal ops (constant interpolation matrix
  + `MatMul`); no per-backend kernel. Bicubic uses PyTorch's Keys cubic kernel
  (`a = -0.75`) with edge clamping; the `_aa` path widens the filter by the
  downsampling ratio and renormalizes (and matches plain when upsampling, as
  PyTorch documents). Unit-tested for shape, partition of unity, sample
  interpolation, antialias downsample weights, and the aa==plain upsample identity.

## [0.2.11] — 2026-07-05

### Added

- **GGUF IQ / TQ / MX end-to-end.** Encoders in `rlx-gguf`; `rlx-gguf-convert` scheme
  enum; GPU dequant parity on Metal / WGPU / CUDA / ROCm; CoreML MIL on-device splits
  for IQ / TQ / MX and K-quants; TPU compile-time and runtime Param bake paths.
- **Metal fused IQ GEMV (`m = 1`).** `iq4_nl`, `iq2_xxs/xs/s`, `iq3_xxs/s`, `iq1_s/m`
  MV kernels with per-scheme disable env vars.
- **Grouped MoE GGUF tests.** Q4_0 / Q8_K / IQ2 / IQ3 / TQ2 / IQ1 expert stacks on
  CPU / Metal / WGPU / CUDA / ROCm (`dequant_grouped_matmul_gguf.rs`).
- **pyrlx GGUF.** `quantize` / `dequant`, `load_gguf` / `write_gguf`, `convert_to_gguf`
  (safetensors → GGUF via `rlx-gguf-convert`); tests in `crates/pyrlx/tests/test_gguf_*.py`.
- **Docs.** Canonical GGUF backend matrix in `docs/gguf-backend-paths.md`; op coverage
  updates in `docs/op-coverage.md`; `just test-gguf-grouped` recipe.
- **`FusedAttentionBlock` is first-class on every inference backend.** All backends now
  declare `OpKind::FusedAttentionBlock` and lower it — CPU/MLX natively, everyone else
  by decomposing to the primitive chain (matmul → narrow → reshape/transpose → \[rope\]
  → attention → matmul). New FAB-only `rlx_fusion::unfuse::unfuse_attention_block` pass
  (shared decomposition with the autodiff unfuse): CUDA/ROCm/TPU/WGPU decompose through
  their own crate `unfuse`, Metal/Vulkan/oneAPI/CoreML through an explicit pre-lowering
  pass (FAB-only, so each backend's native `FusedMatMulBiasAct` / `FusedResidualLN` /
  `LoraMatMul` survive). Cross-backend parity test
  `crates/rlx-runtime/tests/fused_attention_block_parity.rs`. WebGL stays
  excluded — it cannot lower the resulting `Op::Attention`. QNN claims FAB and
  decomposes via `unfuse_attention_block` before the FFI lower.
- **Native CUDA + Metal fused-attention kernels.** A `fused_attn_block` kernel
  (CUDA `fused_attn.cu`; Metal MSL in `kernels.rs`) fuses inline NeoX RoPE + softmax
  SDPA over the packed QKV projection — one block/threadgroup per batch·head, score
  matrix in shared/threadgroup memory — collapsing the decompose chain's narrow×3 +
  transpose×3 + rope×2 + attention into a single launch; the QKV / output projections
  stay as GEMMs into appended arena scratch. Each backend keeps the block native when
  the `[seq,seq]` scores fit (`seq ≤ 96` CUDA / `≤ 64` Metal; Metal additionally gates
  to f32 + no-bias) and otherwise falls back to the primitive decomposition. Validated
  vs the CPU reference: CUDA on an NVIDIA GPU (identity / bias / rope), Metal on Apple
  Silicon (identity / rope native; bias decomposes).

### Performance

- **`rlx-vulkan` dependency-aware barriers.** The scheduler emitted a global
  shader-memory barrier between *every* dispatch; it now emits one only on a real
  read/write hazard, tracked by arena slot offset (safe given the unique-slot bump
  allocator — aliasing shares an offset). On MoltenVK, where each barrier forces a
  Metal compute-encoder restart, this is ~25% faster on the resident-MLP MNIST
  step (135 dispatches, most independent) with bit-identical results. New env
  knobs: `RLX_VULKAN_DEBUG=1` (per-step dispatch histogram), `RLX_VULKAN_FULLBARRIER=1`
  (restore the old between-every-pair behaviour), `RLX_VULKAN_NOBARRIER=1` (drop all
  — unsafe, diagnostic only).
- **`rlx-vulkan` pre-recorded command buffers.** The static schedule (kernels,
  push constants, workgroup counts are fixed; inputs flow through the host-visible
  arena, not the command stream) is now recorded once into reusable command
  buffers and resubmitted with a persistent fence, instead of allocating a command
  buffer + recording + creating/destroying a fence every step. Neutral on MoltenVK
  (re-encodes per submit) but a real latency win on native Vulkan drivers
  (Linux/NVIDIA). `RLX_VULKAN_NOCACHE=1` restores the per-step record path.
- Combined with a larger batch (the `rlx-mnist-device` bench gained an `RLX_BATCH`
  knob), native Vulkan resident-MLP MNIST goes from ~55–87k to ~340–395k img/s on
  an M4 Pro at *higher* accuracy (0.944 → 0.963).

### Fixed

- **`dequant_gguf` routing for legacy 32-byte schemes.** Q4_1 / Q5_0 / Q5_1 no longer
  fall into the 256-element K-quant branch (MSL / CUDA / WGSL).
- **IQ3 fused GEMV element order.** MV kernels now match dequant layout (g1 block then
  g2 block per sub-quant).
- **MLX GGUF dequant cache.** Cache key includes scheme and packed bytes hash.
- **WGPU TQ1_0 / IQ1_M dequant.** Trit u8-wrap and scale-word fixes in WGSL.

## [0.2.10] — 2026-06-25

### Added

- **Distributed / multi-node transport (`rlx-driver`).** New `transport.rs` +
  `net.rs` add a two-sided point-to-point `Transport` trait and a `ProcessGroup`
  layering tensor-shaped `all_reduce` / `all_gather` / `broadcast` / `barrier` on
  top, plus a full-mesh TCP `NetTransport` (one connection per rank pair, a demux
  reader thread per connection routing `SEND` / `PUT` / `GETREQ` / `GETRESP`
  frames). `NetTransport` implements both the two-sided `Transport`
  (pipeline-parallel hidden-state handoff) and the one-sided `SymmetricTransport`
  (put/get/barrier) surfaces, so the existing symmetric collectives run over it
  unchanged. `TcpTransport` and a `ThunderboltTransport` (same wire protocol,
  intended for the macOS Thunderbolt Bridge link, with a `looks_like_thunderbolt`
  IP heuristic) are exposed; re-exported from `rlx-runtime`. Verified by
  multi-rank loopback tests (pipeline handoff, all-reduce / broadcast / barrier
  over TCP, remote put/get). Real multi-node hardware not exercised in this
  environment.
- **`rlx-driver::ring_all_reduce`** — bandwidth-optimal ring all-reduce over a
  symmetric heap (reduce-scatter ring + all-gather ring, Baidu / NCCL pattern)
  expressed with one-sided `put` + `barrier`; each rank moves ~`2(N-1)/N` of the
  vector vs the naïve gather-to-root `O(N²)`. Verified multi-threaded (Sum + Mean)
  against the serial reference.
- **New crate `rlx-collectives`** — an in-graph `collective.all_reduce` custom op
  for tensor-parallel layers: the op carries a `u64` group id in its attrs, each
  rank registers its `ProcessGroup` under an id, and the CPU kernel resolves it
  and sums across ranks at execution time (blocking rendezvous). Validated
  end-to-end by tensor-parallel matmul, Megatron-style SwiGLU MLP, and a Qwen3
  decoder-layer shard test against hand-computed references.
- **`rlx-mlx::MlxTransport` (`distributed.rs`)** — a `Transport` backed by MLX's
  distributed module (`jaccl` RDMA-over-Thunderbolt / `ring` TCP / `mpi`,
  auto-selected), with native `all_sum` / `all_gather` plus two-sided
  length-prefixed byte send/recv and barrier over the MLX C ABI. Also registers a
  **device-resident** `collective.all_reduce` MLX kernel that composes
  `mc::distributed::all_sum` on the lazy device array (no host round-trip), so a
  tensor-parallel layer's all-reduce stays on-GPU. New C-ABI shim entry points in
  `rlx-mlx-sys` (`rlx_mlx_dist_*`). Singleton (no-launcher) path tested; multi-rank
  jaccl / ring needs MLX's launcher.
- **`Op::Reverse { axes }`** — batch-general flip along listed axes (output shape
  unchanged; non-listed axes pass through). Native kernels on CPU, Metal, MLX, and
  WGPU; host-staged on CUDA / ROCm. Reverses a `[batch, seq, …]` sequence without
  a `batch == 1` assumption.
- **`Op::Gru`, `Op::Rnn`, `Op::Mamba2` native GPU kernels.** Single-layer /
  unidirectional / no-carry kernels: Metal MSL `gru` / `rnn` / `mamba2` (hidden
  ≤ 1024, Mamba2 state ≤ 128) and native WGSL `gru` / `rnn` / `mamba2` (hidden /
  state ≤ 256), each with a host-staged CPU fallback for the multi-layer /
  bidirectional / carry / oversized cases. The CPU references
  (`execute_gru_f32` / `execute_rnn_f32` / `execute_mamba2_f32`) back the
  fallbacks. HIR gains a `Gru` op + lowering. Validated by `metal_rnn_native` /
  `wgpu_rnn_native` parity tests against the CPU reference.
- **Native Metal `selective_scan` MSL kernel** (f32, state ≤ 128) replacing the
  host fallback for Mamba S6 on Metal, plus native Metal `argreduce`, `sample`,
  and `reverse` paths. Their CPU references (`execute_selective_scan_f32`,
  `execute_sample_f32`, `execute_argreduce_f32`) are now shared host-delegate
  functions.
- **Native Metal block-quantized matmul** (`dequant_matmul_int8` /
  `dequant_matmul_int4` MSL kernels) — dequant-on-the-fly over the unified-memory
  arena for int8 / int4 block schemes, matching the CPU reference (new
  `metal_dequant_matmul_int_parity` test).
- **WGPU vision + recurrent op coverage.** Host-staged `ConvTranspose2d`,
  `GroupNorm`, `LayerNorm2d`, `ResizeNearest2x`, `Reverse`, and `ArgMax` / `ArgMin`
  (readback → verified CPU kernel → writeback, mirroring the existing
  `im2col_host` pattern; new `vision_host.rs` / `conv_transpose2d_host.rs`),
  closing the correctness gap for SAM / U-Net decoders on cross-platform GPU. New
  `wgpu_vision_ops_parity` test.
- **CUDA / ROCm host-staged `Reverse`, `ArgMax`, `ArgMin`, `AxialRope2d`, and
  `StopGradient`** (`host_misc.rs` on both backends: sync → dtoh → verified
  rlx-cpu kernel → htod). TPU also gains `StopGradient` (forward-identity HLO
  alias). **Compile-verified only; not run on NVIDIA / AMD / TPU hardware** (no
  device in this environment).
- **MLX op coverage** — native `Reverse`, `ArgMax`, `ArgMin`, `Im2Col`
  (NCHW → rows), `GroupNorm` (NCHW), and `GroupNorm` backward (input / gamma /
  beta) lowerings.
- **CoreML / ANE fused-attention layouts** — `lower_attention` now disambiguates
  the `[B,S,H,D]` (heads at axis 2) vs `[B,H,S,D]` (canonical) operand layouts via
  `attention_geom`, transposing the former to canonical, attending, and
  transposing back — without it CoreML would attend over the heads axis and fail
  once `s_q != s_k` (KV-cache decode).
- **Dynamic-shape support on every backend (`rlx-runtime::deferred`).**
  `Session::compile` now detects `Dim::Dynamic` graphs and wraps them in a
  `DeferredExecutable` that infers the concrete shape from input lengths on each
  `run`, specializes to a static graph, and caches the most recent specialization
  — giving CPU / Metal / CUDA / ROCm the multi-shape support wgpu / MLX already
  had internally, with no recompile when a shape repeats.
- **`rlx-text` streaming detokenization + tool-call parsing.** New
  `StreamingDetokenizer` (`detokenize.rs`) re-decodes the full id sequence each
  step and emits only the newly-stable suffix, holding back trailing U+FFFD runs —
  fixes byte-level-BPE multi-byte splits and SentencePiece context-dependent
  spacing that per-token decode mangles. New `tool_parse.rs` parses model-emitted
  tool/function calls (Hermes / Qwen `<tool_call>` blocks, bare JSON, and Pythonic
  `[fn(a="x")]`) into `{name, arguments}`, with a `detect_and_parse` helper.
- **Min-p sampling + logit bias (`rlx-runtime`).** New `MinP` sampler (keep tokens
  with prob ≥ `p · p_max`, with a `min_keep` floor; Nguyen et al. 2024) wired
  through `SampleOpts::min_p`, and a host-side `apply_logit_bias` (OpenAI
  `logit_bias` semantics, bounds-checked, additive).
- **Host-driven logits decode + prompt-cache session reuse (`LmRunner`).** New
  default-implemented `prefill_logits` / `decode_logits` (caller owns sampling,
  logit bias, log-probs, stop detection — for the HTTP server), plus
  `export_session` / `restore_session` / `prefill_logits_reusing` over a new
  `SessionSnapshot` (KV cache + token history) for prefix reuse.
- **Streaming attention over a quantized KV cache (`rlx-runtime::quantized_kv`).**
  `attend_quantized` does single-query GQA-aware attention directly over the
  quantized layer, dequantizing one row at a time so peak extra memory is
  `O(kv_dim)` not `O(past_len · kv_dim)`; new `read_rows(start, count)`
  generalizes `read_window`. Validated against full-dequant attention.
- **Sliding-window decode masking.** `attn_mask::bucket_decode_mask_windowed`
  masks cached keys outside `[past_seq − window, past_seq]` for incremental decode
  (reduces to the causal mask when the window is wide), and the Qwen3 decoder block
  (`rlx-flow`) now takes a configurable `MaskKind` (`Causal` or
  `SlidingWindow(w)`) instead of hard-coded causal — enabling Mistral / Gemma-style
  sliding-window models.
- **`Device::as_arg`** — canonical lowercase CLI token that round-trips through
  `FromStr` (unlike the human-facing `name`, e.g. `"GPU (wgpu)"`).
- **New parity / coverage tests** across backends (argreduce, conv2d-groups,
  conv-bias, conv-transpose2d, GRU, group-norm-backward, LoRA-matmul-decompose,
  LSQ-quant, native RNN on Metal/WGPU, im2col on MLX, multi-shape / dynamic,
  reverse, sample, vision-ops, sliding-window-attn, dequant-matmul-int, expand,
  CPU selective-scan), a `bench_new_ops.rs` example, and a new
  [`docs/op-coverage.md`](docs/op-coverage.md) — the single-source-of-truth
  op × backend matrix (113 `OpKind`s).

### Performance

- **CPU executor: per-run arena reset is now O(scratch), not O(params).**
  `restore_arena_baseline` previously cloned and rewrote the entire (multi-GB)
  weight region on every `run()`, making large models swap-thrash. Params /
  constants now live solely in their dedicated never-aliased arena slots (no
  redundant CPU-side copy that doubled the weight footprint), and only the
  complement of the persistent byte ranges is zeroed each run. Constants are
  written once.
- **`Op::GroupedMatMul` on CPU is now a real segmented GEMM** — counting-sort
  tokens by expert, one GEMM per expert, then unpermute — replacing the naive
  per-token implementation. (GPU backends dispatch a dedicated grouped kernel.)
- **`OpKind::LoraMatMul` added to `FUSED_KINDS`** so its `unfuse` decomposition
  actually fires on backends without a native LoRA kernel (Metal / WGPU / CUDA /
  ROCm / TPU). Standalone LoRA previously failed legalization unless another fused
  op happened to be present (verified exact vs CPU).

### Fixed

- **Per-token (ragged) RoPE on CPU and Metal.** The RoPE kernels now index the
  cos/sin table per token (one row per batch·seq element) when the table has
  `total_tokens` rows, so ragged batched decode — each sequence in the batch at a
  different absolute position — gets its own RoPE row instead of collapsing to row
  0. New Metal `cos_per_token` kernel path; `device_ext::supports_ragged_rope`
  gates this to CPU + Metal (other GPU RoPE kernels still index by seq position, so
  callers fall back to per-length uniform grouping there). Validated by the new
  ragged `metal_rope_parity` test against CPU.
- **`ElementwiseRegion` output dtype mis-inference.** A fused elementwise chain now
  takes the dtype of its final chain step (walking `Compare → Bool`,
  `Cast → its dtype`, …) rather than input 0's — input 0 may be a bool `Where`
  condition (`where(cond, a, b) + …`), which previously mis-typed the whole region
  as bool.
- **`conv_transpose2d` dynamic-batch shape inference** now preserves a dynamic
  batch dim (mirroring `conv2d_output_shape`) instead of force-unwrapping it to a
  static value.
- **Metal uninitialized-buffer reads.** `new_buffer` (shared storage) is now zeroed
  on allocation — ops that read unwritten arena regions (e.g. conv halo padding)
  previously picked up per-process garbage, a nondeterminism / correctness bug.
- **`rlx-coreml` stale `.mlmodelc` cache** — a version-incompatible or corrupt
  compiled-model cache no longer permanently breaks loading; a load failure
  discards the cache entry and recompiles from the `.mlpackage`.
- **`moe_residency` GroupedMatMul ordinal made thread-local.** The process-global
  atomic let one thread's `reset` / `next` ordinal sequence clobber another
  in-flight forward's (corrupting layer/matrix decode → wrong TIDE host-expert
  weights + residency accounting); forwards run per-thread, so the counter is now
  thread-local.
- **`LocalTransport` barrier is now a real rendezvous** (`std::sync::Barrier`,
  auto-reset) instead of an arrival counter, so multi-step collectives (ring
  all-reduce) synchronize correctly across threads.
- **`memory_estimate::would_exceed_soft_budget`** boundary logic split into a pure,
  deterministically-testable `exceeds_budget` predicate (the prior test was flaky
  against live fluctuating RSS).

### Changed

- **CPU now declares `Reverse`, `Gru`, `Rnn`, `Mamba2`, `FakeQuantizeLSQ` (+ LSQ
  backward X/scale), and `GroupNorm` backward (input/gamma/beta)** in
  `CPU_SUPPORTED_OPS` — the kernels already existed in `thunk.rs` / training-bwd,
  but the legalization const omitted them, so compiled CPU graphs couldn't use
  them. CPU is once again the reference that lowers every `OpKind` any backend does.
- Backend `supported_ops()` consts updated to reflect the new native / host
  kernels above; the canonical per-backend op counts now live in
  [`docs/op-coverage.md`](docs/op-coverage.md) (CPU 104, MLX 84, MTL 76, WGPU 75,
  CUDA 71, ROCm 68, TPU 50 of 113 `OpKind`s).
- New env opt-out: `RLX_METAL_RNN_HOST_FALLBACK` forces the Metal GRU / RNN host
  path.

## [0.2.9] — 2026-06-22

### Performance

- **`rlx_cpu::ms_deform_attn`** (the shared host-delegate behind the fused
  `Op::Custom("gdino.ms_deform_attn")` on CPU/Metal/MLX/CUDA/WGPU) now routes its
  value/offset/attention/output projections through `blas::sgemm_bt` instead of a
  naive triple loop. The projections dominate at full token counts (~18k); the
  GPU host-delegates were ~7× slower than the CPU backend (which already used
  BLAS) purely from this. Grounding DINO MLX enhancer 8.8→1.0s, decoder 2.6→0.1s.
- **`rlx_cpu::conv_fwd::conv2d_forward_nchw_f32`** rewritten as im2col +
  `blas::sgemm` (groups/stride/pad/dilation preserved); replaces a naive 6-deep
  loop. Benefits CNN-backbone models on CPU and the GPU conv host-delegates.
  Validated by the existing conv fwd/bwd/1×1/q_conv2d tests.

### Added

- **`rlx-coreml` fused multi-head attention & RoPE layouts.** `lower_attention`
  and the RoPE lowering now dispatch on operand layout: the original split
  `[..,S,D]` / last-dim-==-`head_dim` path is byte-for-byte unchanged, and a new
  fused `[B,S,H·D]` path (heads packed in the last axis, as in Qwen3 / Qwen3-ASR
  fused-QKV) reshapes+transposes to canonical `[B,H,S,D]`, runs the shared
  `attention_core`, and folds the result back. RoPE gains the same per-head view
  (cos/sin broadcast over a singleton head axis). Expands the set of transformer
  models lowerable to CoreML/ANE.

### Fixed

- **`rlx-wgpu` attention additive-bias mask.** `MaskKind::Bias` now lowers to its
  own kernel path (mask kind `4`, `score += mask`) instead of being folded into
  the binary key-padding path (kind `2`, `mask < 0.5 → -inf`). The two are not
  interchangeable — an additive block-diagonal window bias (e.g. the encoder
  winmask) was silently corrupted by the binary path.
- **`rlx-wgpu` decode-step causal/sliding-window masking.** The attention kernel
  now compares against the absolute query position `qi + (seq_k − seq_q)` rather
  than the local `qi`, so causal and sliding-window masks are correct when
  `seq_q == 1` during incremental decode (past KV precedes the query). Prefill
  (`seq_q == seq_k`) is unaffected.
- **`rlx-wgpu` matmul arena-window assertion** no longer spuriously panics for
  models whose entire arena fits within `max_binding` (whole-arena bind reports
  `param_anchor = false`, but the large param B is trivially in-window); the
  assertion now keys on actual addressability.
- **`rlx-cuda` / `rlx-rocm` attention kernels** (`attention.cu`,
  `attention_row.cu` in `rlx-gpu-kernels`, shared by both backends) gained
  parity with the Metal/WGPU fixes above: (1) an additive-bias path for
  `MaskKind::Bias` (kernel mask kind `4`, `score += mask`) — the backends
  already bound the bias tensor for kind `4` but the kernels had no branch for
  it, so the bias was silently dropped (ALiBi / block-diagonal window bias);
  (2) decode-step causal/sliding-window masking now compares against the
  absolute query position `qi + (seq_k − seq_q)` instead of the local `qi`, so
  causality is correct when `seq_q < seq_k` during incremental decode. Masking
  logic validated by a host-C++ harness; **not yet verified on CUDA/ROCm
  hardware** (no device available in this environment).
- **`rlx-mlx` concurrent free/eval crash.** `Array::drop` freed its MLX handle
  (`rlx_mlx_array_free`) without the runtime lock, so a result array freed on
  one thread could race a guarded `eval()` on another and SIGSEGV (intermittent,
  release + multi-threaded). The runtime lock is now reentrant (thread-local
  depth over the existing mutex) and is held by `Array::drop`, `eval`,
  `async_eval`, `synchronize`, and `clone_handle` — so cross-thread frees
  serialize against in-flight eval, while intermediate drops inside a guarded
  `run_*` on the same thread don't deadlock. Single-threaded inference (the hot
  path) only pays a thread-id check.

### Changed

- **`rlx-metal` built-in custom-op kernels auto-register.** The bundled
  host-delegate kernels (e.g. `ms_deform_attn`) register themselves on first
  custom-op lookup — no explicit `register()` call or extra cargo feature
  required. (`llada2_gate` stays consumer-registered to avoid double-registration.)
- **`rlx-mlx` Lazy-fallback warning** now fires at most once per distinct reason
  per process (was once per executable — models with many graphs sharing a
  host-eval op flooded the log). `RLX_MLX_WARN_LAZY=all` restores per-executable
  warnings. Logging-only; execution is unchanged.

## [0.2.8] — 2026-06

### Added — PyTorch `.pt` weight loading

- **`rlx_nemo::PtModel`** — standalone loader for plain PyTorch `.pt` /
  `.pth` / `pytorch_model.bin` `torch.save` checkpoints, reusing the
  non-executing pickle VM + STORED-zip reader that already backs the
  `.nemo` loader. Tensors materialize on demand as contiguous f32
  regardless of on-disk dtype (fp32 / fp16 / bf16 / int). Modern (≥ PyTorch
  1.6) zip format only; legacy non-zip pickles are rejected with a clear
  error.
- **`rlx-gguf-convert` `pt` feature** — `Converter::from_pt(...)` + a
  `PtReader` `TensorReader`, so `.pt` checkpoints convert/quantize to GGUF
  like safetensors/ONNX. The `convert` example now dispatches by input
  extension (`.safetensors` / `.pt` / `.pth` / `.bin` / `.onnx`).

### Added — Recurrent / sequence ops & complex matmul

- **`Op::Lstm`** — multi-layer, optionally bidirectional, optional decode
  carry (`h0`/`c0` threaded in place). Packed weights; gate order i,f,g,o.
  Real CPU kernel (`execute_lstm_f32`) shared by the CUDA / ROCm / wgpu /
  Metal host fallbacks; **native Metal MSL kernel** for the single-layer
  unidirectional path (`hidden ≤ 1024`, opt out via `RLX_METAL_LSTM_CPU=1`).
  Decomposes via `unfuse` for MLX / CoreML / TPU and backprop-through-time
  (verified against central finite differences).
- **`Op::Gru`** (PyTorch r/z/n, separate `b_ih`/`b_hh`) and **`Op::Rnn`**
  (Elman, tanh/ReLU) — multi-layer / bidirectional / carry, via the same
  `unfuse`-for-autodiff decomposition.
- **`Op::Mamba2`** — Mamba-2 / SSD scalar-decay structured state-space scan
  (sibling of `SelectiveScan` / `GatedDeltaNet`); `unfuse` decomposition.
- **Complex (C64) matmul** — `Thunk::CgemmC64` CPU kernel + a `MatMul`
  lowering branch, completing the deferred piece of C64 support. `MatMul`
  VJP now inserts Wirtinger conjugates for C64
  (`dA = upstream·conj(B)ᵀ`, `dB = conj(A)ᵀ·upstream`).
- **ONNX coverage ops** — GatherND, ScatterND, OneHot, NonZero, CumProd,
  and Einsum (with a real equation parser) as `Op::Custom("onnx.*")` CPU
  reference kernels wired through the ONNX import path.

### Changed

- **`rlx-mlx-sys`**: bumped the vendored MLX submodule to **0.32.0** (from
  0.31.2). The C-ABI shim builds unchanged against the new upstream API;
  all `rlx-mlx` (128) and runtime-MLX integration (455) tests pass.

### Added — `rlx-tensor` symbolic Tensor DSL

- **New crate `rlx-tensor`**: a native `ndarray` alternative — NumPy-style,
  operator-overloaded `Tensor` handles that trace into `rlx-ir` instead of
  executing eagerly. The graph stays lazy (so fusion + memory planning see
  the whole expression) until forced. Re-exported through the prelude as
  `rlx::tensor` / `rlx::prelude::{Tensor, graph, s, shape, ...}` behind the
  umbrella `tensor` feature (**on by default**; `--no-default-features` drops
  it). The umbrella backend flags (`cpu`/`metal`/`cuda`/…) now also enable the
  matching `rlx-tensor` `eval` backend via weak features, so
  `rlx::tensor::Tensor::to_vec` / `.on(Device::…)` materialize out of the box.
- **Op surface**: arithmetic + scalar ops, activations (`relu`/`gelu`/`silu`/
  …), reductions (`sum`/`mean`/`var`/`logsumexp`/`cumsum`/`argmax`/…), shape &
  view ops (`reshape`/`narrow`/`slice` via `s![]`/`split`/`cat`/`stack`/…),
  indexing (`gather`/`where_`/`masked_fill`/comparisons), and NN/linalg
  (`matmul`/`softmax`/`layer_norm`/`rms_norm`/`conv2d`/`attention`/`rope`/
  `fft`/`inv`/`solve`). First-class `Dim::Dynamic` for variable batch/seq.
- **Opt-in features**: `eval` (materialize via `rlx_runtime::Session`, default
  CPU; `eval-metal`/`eval-mlx`/`eval-cuda`/`eval-rocm`/`eval-gpu`/`eval-coreml`/
  `eval-apple`/`eval-blas` for other backends), `grad`/`transforms` (reverse-mode
  AD + composable `Func` transforms `vmap`/`jvp`/`hvp`), `optim`
  (`Func::train_step` + `rlx_optim` optimizers), and `ndarray` interop
  (`Tensor::from(array)` / `to_ndarray`). Base crate stays a pure `rlx-ir`
  graph builder with no backend pulled in.

### Added — IQ / TQ / MX dequant family

- **`rlx-gguf`**: dequant kernels for every llama.cpp scheme — IQ4_NL,
  IQ4_XS, IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M, TQ1_0/TQ2_0, MXFP4, NVFP4.
  Grid LUTs auto-extracted from `ggml-common.h` and shipped in
  `src/iq_grids.rs`. Real-weight parity tests against `llama-quantize`
  output on Qwen3-0.6B (`tests/iq_tq_real_weights.rs`).
- **Q2_K / Q3_K layout fix** (`rlx-gguf`, `rlx-cpu/gguf_matmul`,
  `rlx-metal/dequant_gguf.msl`, `rlx-cuda/dequant_gguf.cu`): pre-existing
  encoder/decoder put f16 `d`/`dmin` at the front of the block, but
  llama.cpp's actual layout is `scales | qs | d` (Q2_K) and
  `hmask | qs | scales | d` (Q3_K). Constant-value round-trip tests
  passed because the encoder used the same flipped layout; real GGUFs
  produced NaN. Decoder and encoder now match `ggml-common.h` exactly.
- **GPU dequant**: native MSL (rlx-metal) + CUDA (rlx-cuda) kernels for
  all 19 schemes. IQ-family grids staged into a ~33 KB device buffer
  per session/context via `Kernels::iq_grid_buffer` (Metal) /
  `cuda_iq_grid_buffer` (CUDA), bound as the 6th kernel argument.
  `has_metal_dequant_kernel(QuantScheme)` reports coverage.
- **`rlx-ir`**: 13 new `QuantScheme` variants (`GgufIQ4NL`, `GgufIQ4XS`,
  `GgufIQ2XXS`, `GgufIQ2XS`, `GgufIQ2S`, `GgufIQ3XXS`, `GgufIQ3S`,
  `GgufIQ1S`, `GgufIQ1M`, `GgufTQ1_0`, `GgufTQ2_0`, `GgufMXFP4`,
  `GgufNVFP4`) with byte-counts wired into `gguf_block_size` /
  `gguf_block_bytes` / `is_gguf` / `bits_per_element_x10`.
- **`rlx-gguf` encoders**: IQ/TQ/MX quantize path (`iq_quantize`,
  `iq2_encode`, `iq3_encode`, `iq1_encode`, `tq_quantize`, `mx_quantize`).
  IQ2 uses llama.cpp kmap + sign-extraction; parallel block encoding via
  `rayon`. CoreML on-device dequant for NVFP4 + IQ2/3/1
  (`split_*_ondevice` in `rlx-coreml/src/mil/helpers.rs`).

### Added — Samplers

- **`rlx-runtime::samplers`** module: backend-agnostic `Sampler` trait +
  `SamplerChain`. Implements Temperature, DynamicTemperature, TopK,
  TopP, TopNSigma, TypicalP, Mirostat v1 / v2, XTC, DRY, RepetitionPenalty
  in the canonical llama.cpp order.
- **`SampleOpts` extended** (`rlx-runtime::lm`, `rlx_qwen3::sampling`):
  fields for every advanced sampler default to off; `into_chain()`
  builds the chain; `is_classic()` lets legacy callers stay on the
  fast inline path. `MirostatMode` enum exposed at the runtime top
  level. `sample_token_with_history()` and `sample_token_stateful()`
  added to rlx-qwen3 for chain-aware decoders.

### Added — Quantized KV cache

- **`rlx-runtime::quantized_kv`**: per-layer K/V history stored as
  `KvQuant::{F16, Q8_0, Q5_0, Q4_0}` GGUF blocks instead of f16/f32.
  ~2–4× memory cut on long decodes. `QuantizedKvLayer::{append_rows,
  read_all, read_window, drop_front}` + `QuantizedKvCache` aggregate.
- **`mmap-kv` feature**: optional `MmapKvLayer` / `MmapKvCache` backed by
  `memmap2` for anonymous or file-backed mappings. Pages cold history
  in/out via the OS page cache; `prefetch_window` issues madvise
  WILLNEED. Use case: 100k-token contexts that exceed RAM.

## [0.2.7] — 2026-06

### Added

- **In-graph RNG** (`rlx-ir`): `Op::RngNormal` / `Op::RngUniform` for ONNX
  `Random*` / `Random*Like`; `RngOptions` / `RngBackend` (Philox default,
  Ort CPU parity, Zero for deterministic tests). `CompileOptions::rng` and
  `CompiledGraph::set_rng` override policy without recompiling.
- **RNG backends**: CPU Philox + ORT reference (`rlx-cpu`); host-fill on Metal/MLX;
  D2H→fill→H2D on CUDA/ROCm/wgpu; XLA `rng` lowering on TPU (`rlx-tpu`).
- **ONNX `Random*` import** (`rlx-onnx-import`): native lowering to
  `Op::RngNormal` / `Op::RngUniform` (direct import + codegen); shared
  `random.rs` helpers; conformance harness coverage (`rlx-onnx-conformance`).
- **Autodiff / vmap**: RNG ops treated as stateless w.r.t. gradients (`rlx-autodiff`).

### Changed

- Patch bumps for all workspace crates in this release train to **0.2.7**
  (`rlx-ir`, `rlx-compile`, `rlx-gpu-kernels`, `rlx-cuda`, `rlx-wgpu`, `rlx-metal`,
  `rlx-mlx`, `rlx-runtime`, `rlx`, …); `rlx-mlx-sys` and `pyrlx` at **0.2.7**.

## [0.2.6] — 2026-06

### Added

- **Native GPU `Op::WelchPeaks`** (`rlx-cuda`, `rlx-wgpu`, `rlx-rocm`, `rlx-gpu-kernels`):
  in-arena Welch PSD top-K when eligible (`rlx-ir::welch_peaks_gpu_native_eligible`,
  f32 spectrum, ≤512 one-sided bins, K≤64); host CPU path unchanged for out-of-range
  shapes.
- **`rlx-compile` IO-gated fusion**: `SelectPeaksOnlyOutputs` drops FFT spectrum from
  graph outputs when peaks-only readback wins the per-target IO gate; compile-time
  `profile_graph_io` / `profile_graph_io_outputs`; thread-local `FusionTarget` for
  gated passes. Opt out with `RLX_NO_IO_PEAKS_OUTPUT=1`.

### Fixed

- **`rlx-mlx` `Activation::GeluApprox`**: lower through `ops::gelu_approx` (tanh
  form matching `rlx-cpu`) instead of exact `gelu`. Fixes ~3% Brain-JEPA predictor
  drift vs CPU while SDPA stayed within tolerance. Optional `RLX_MLX_SDPA_REFERENCE=1`
  composes unfused matmul+softmax for SDPA bisects.
- **`rlx-metal` MPSGraph `Activation::Gelu`**: use `erfWithTensor` + the CPU
  erf GELU formula (`0.5·x·(1+erf(x/√2))`) instead of the tanh approximation.
  Fixes large CPU/Metal drift on REVE-style GEGLU blocks (~0.08 max abs → ~5e-3
  on a single transformer layer; full-model parity restored with MPSGraph enabled).

### Changed

- Patch bumps for all workspace crates in this release train to **0.2.6**
  (`rlx-ir`, `rlx-compile`, `rlx-gpu-kernels`, `rlx-cuda`, `rlx-wgpu`, `rlx-metal`,
  `rlx-mlx`, `rlx-runtime`, `rlx`, …); `rlx-mlx-sys` remains **0.2.6**.

## [0.2.5] — 2026-06

### Fixed

- **`rlx-runtime`**: import `rlx_opt::pass::Pass` in CUDA and ROCm `Backend::compile`
  so `LegalizeBroadcast` / `AutoMixedPrecision` compile on Rust ≥1.87 (crates.io
  0.2.4 tarball missed this in `compile()` while `compile_lir()` had it).

### Added

- **`Op::WelchPeaks`**: Welch PSD top-K spikes from block-layout FFT segment spectra
  (`rlx-ir`, CPU + Metal + MLX lowering, CUDA/wgpu host sidecars, runtime supported-op lists).
- **`rlx-runtime::graph_io`**: static IO / sync profiling for compile-time fusion
  planning (`GraphIoProfile`, `profile_graph_io`, peaks-only output sizing).
- **`rlx-compile::fusion_benefit`**: IO-aware fusion benefit scoring and per-target
  gates (`io_fusion_gate_for_target`).

## [0.2.3] — 2026-06

### Added

- **Multi-backend runtime** (`rlx-runtime` 0.2.3): `DevicePolicy`,
  `GraphDevices`, `FlexibleSession`, `DeviceRouter`, env-driven resolve /
  fallback (`RLX_DEVICE`, `RLX_DEVICE_CHAIN`, `RLX_BENCHMARK_PICK`),
  `BackendsManifest`, `warm_all` / `benchmark_devices`, typed param sync
  across cached backends.
- **Prelude** (`rlx` 0.2.3): re-exports above + `register_backends!` macro.
- **Python** (`pyrlx` 0.2.3): `GraphDevices`, `DeviceRouter`, `DevicePolicy`,
  `FlexibleSession`, `parse_device`, `backends_manifest`, `fastest_device_for`,
  `device_report`, `set_param_typed` on multi-backend runners.
- **GPU calibrators**: on-disk matmul micro-bench caches for CUDA
  (`rlx-cuda` 0.2.3), ROCm (`rlx-rocm` 0.2.3), wgpu (`rlx-wgpu` 0.2.3);
  feed heterogeneous cost-model ranking.
- **ROCm full CUDA parity** (`rlx-rocm` 0.2.3): all 48 hipRTC kernels, Session-path
  `GroupNorm` / `ResizeNearest2x`, GPU backward ops, GGUF GPU dequant, splat prepare/rasterize,
  im2col, pinned host I/O (`host_staging.rs`, `RLX_ROCM_PINNED_IO`).
- **Runtime ROCm** (`rlx-runtime` 0.2.3): ROCm supported-op parity, `rocm_op_parity` tests,
  ROCm arms in higher-order / autodiff GPU parity suites.
- **FKL-style region fusion** ([`docs/fk-fusion.md`](docs/fk-fusion.md)): resize prologue
  (`FuseRegionPrologue`), batch preprocess (`FuseBatchPreprocess` /
  `BatchElementwiseRegion`), `MarkBatchSliceRegions`, `apply_native_fk_defaults` on
  GPU-class targets and TPU. CUDA/ROCm/Metal/wgpu single-launch batch kernel via
  `RLX_FK_BATCH_SINGLE_KERNEL=1`. TPU HLO lowering in `rlx-tpu` (`prepare_graph_for_hlo`,
  `fk_pipeline`). Parity: `rlx-runtime/tests/fk_prologue_parity.rs`, `pyrlx` FK tests,
  `rlx-bench` `bench_fk_fusion`.
- **HIP-CPU**: Docker-only fetch into `rlx-cuda/docker/vendor/HIP-CPU` via
  `just test-hip-cpu-validate` (linux-gnu; not a git submodule).
- **Autodiff**: `prepare_graph_for_ad` runs `DecomposeFusionRegions` so FKL batch/transform
  ops decompose before reverse-mode AD.
- **Docs**: [`docs/backend-selection.md`](docs/backend-selection.md),
  [`docs/development.md`](docs/development.md), [`docs/README.md`](docs/README.md).
- **Examples**: `rlx-runtime/examples/graph_devices_demo.rs`.
- **Tests**: full `hip_cpu_validate` suite (38 kernel families), `rlx-rocm/tests/basic.rs`
  GatedDeltaNet, `rlx-runtime/tests/rocm_op_parity.rs`,
  `rlx-runtime/tests/graph_devices_parity.rs`, `crates/pyrlx/tests/test_graph_devices.py`,
  ROCm suites in higher-order / autodiff GPU parity tests, `prologue_input` on region op
  literals in Metal/MLX/wgpu parity tests.
- **CI**: `just test-rocm`, `just test-hip-cpu-validate`, ROCm arm in `just ci` /
  `test-third-order-gpu`.

### Changed

- Patch bumps for all crates in this release train (`rlx-ir`, `rlx-opt`,
  `rlx-fusion`, `rlx-compile`, `rlx-autodiff`, `rlx-cpu`, `rlx-cuda`, `rlx-metal`,
  `rlx-mlx`, `rlx-mlx-sys`, `rlx-gpu-kernels`, `rlx-bench`, `rlx-wgpu`); workspace
  dependency pins leveled to 0.2.3.

### Fixed

- **`rlx-gguf`**: `dequant_q6_k_block` now casts per-sub-block scales as
  `i8` (matching `dequant_q6_k`). The old `as f32` path misread bytes
  ≥128 (e.g. `0xFF` → 255 instead of −1), breaking `Op::DequantMatMul`
  on Q6_K tensors such as MiniCPM5 `v_proj` / `down_proj`.

## [0.2.2] — 2026-05

### Added

- **`rlx-umap`** crate — UMAP / fast-umap custom ops (k-NN from pairwise distances).
- **`rlx-gpu-kernels`** crate — shared CUDA/HIP `.cu` sources for `rlx-cuda` + `rlx-rocm`.
- **`rlx-cpu`** kernel and executor improvements.

## [0.2.1] — 2026-05

### Changed

- Workspace/crate version bump only.

### Added

- **`rlx::run` runner API** (`model builders` crate, `run` module):
  builder-style entry points for the supported model families,
  re-exported in the prelude under the `models` cargo feature.
  - `Qwen3Runner::builder()` — `.weights(p)`, `.device(d)`,
    `.max_seq(n)`, `.precision(F32 | F16LmHead)`,
    `.max_memory_gb(g)`, `.stream(bool)`, `.use_mtp(bool)`,
    `.sample(opts)`, `.config(ConfigSource::…)`,
    `.format(WeightFormat::…)`, `.build()`.
  - `SamRunner::builder(SamArch::Sam1 | Sam2 | Sam3)` — uniform
    builder shape, `.predict_image(...)` method dispatches to the
    per-arch `Sam{,2,3}::from_safetensors_on` + forward call.
  - Helpers `open_loader(path)`, `list_mtp_keys(path)`,
    `debug_resolve_name(hf_name)`.
- **`rlx-run` CLI** (`model builders` crate, `rlx-run` binary): subcommands
  `qwen3`, `sam1`, `sam2`, `sam3`, `inspect`, `help`. Hand-rolled
  arg parser — no clap dep. Mirrors the builder API 1:1.
- **`Op::DequantMatMul` GGUF schemes** (`rlx-ir/src/quant.rs`):
  `QuantScheme::GgufQ4K`, `GgufQ5K`, `GgufQ6K`, `GgufQ8K`. CPU
  implementation in `rlx-cpu` dequants the packed bytes to f32
  scratch then sgemm — keeps the arena footprint small (Q4_K ≈
  4.5 bpe vs F32's 32 bpe) at the cost of per-call dequant. Metal
  lowering is on the roadmap (per-op thunk path still dequants at
  load time today).
- **GGUF K-quant decoders** (`rlx-gguf`): Q4_K, Q5_K, Q6_K, Q8_K
  block decoders, mirroring llama.cpp's `ggml-quants.c` reference.
  Made `pub` so `rlx-cpu`'s `DequantMatMul` GGUF arm can call them.
- **`GgufLoader`** (`model builders::weight_loader`): pluggable
  `WeightLoader` for `.gguf` files with transparent
  HF↔GGUF name resolution (`hf_to_gguf_name` /
  `gguf_to_hf_name`), MTP-head isolation (`is_mtp_weight`,
  `mtp_keys`), and shape normalization (innermost-first GGUF dims
  reversed to safetensors order without byte movement).
- **Qwen3 graph builder** (`model builders` crate, `qwen3`): GQA via
  graph-level KV head repetition, QK-norm, RoPE, SwiGLU,
  tied-embedding LM head with build-time weight pre-transpose
  (eliminates 600 MB per-call Transpose op), prefill + cached
  decode generators with bucketed compile cache.
- **MPSGraph Metal fast path** (`rlx-metal`):
  - `rms_norm` via `normalizationWithTensor:mean=0:variance=mean(x²):`
    (uses Apple's fused norm kernel).
  - `attention_causal` via `scaledDotProductAttention` builtin with
    in-graph constant causal mask — bypasses the slice-of-computed
    MPSGraph optimizer bug that hits the BERT QKV-split pattern.
  - `ElementwiseRegion` chain replay for fused SwiGLU.
  - Pre-compiled `MPSGraphExecutable` with feed/result permutation
    recovered from `executable.feedTensors`/`targetTensors`; per-call
    dispatch is one ObjC call with the input/output `NSArray`s built
    once at compile (`bind_arena` + `run_cached`).
  - Default-on whenever lowering succeeds; opt out via
    `RLX_DISABLE_MPSGRAPH` / `RLX_DISABLE_MPSGRAPH_EXECUTABLE`.
  - Opt-in `RLX_MPSGRAPH_PARAM_CONST=1` bakes weights as graph
    constants (production single-shape callers).
- **F16 LM-head path** (opt-in via `RLX_QWEN3_F16_LM_HEAD=1`): casts
  hidden + lm-head weight to F16 before the final matmul. Wins
  1.3-1.45× on B≥2, L≥64 `last` cells.
- **Examples per model family** (`model builders` repo `examples/`):
  `run_qwen3_safetensors.rs`, `run_qwen3_gguf.rs`, `run_sam1.rs`,
  `run_sam2.rs`, `run_sam3.rs`, plus `qwen3_gguf_inference.rs` and
  `gguf_qwen3_probe.rs` for deeper walk-throughs.
- **Publish script** (`scripts/publish.sh`): tier-ordered workspace
  publisher with active sparse-index polling, HTTP 429 backoff, and
  live countdown timers. See `--help`.

### Changed

- **`Op::DequantMatMul::num_inputs()` is now scheme-dependent**
  (was always 4). Returns 2 for GGUF schemes (`[x, packed_w]`),
  4 for legacy Int8 schemes (`[x, w_q, scale, zp]`). **Breaking**
  for any downstream code that hard-coded the input count — match
  on `scheme.is_gguf()` before reading inputs.
- **`GgufLoader::take_transposed` now actually transposes**
  (was a buggy no-op that returned GGUF native bytes with the GGUF
  shape label, silently producing wrong logits when the builder
  expected `[in, out]` row-major). The fix routes through
  `GgufLoader::take` which now normalizes GGUF's innermost-first
  shape convention to safetensors' outermost-first ordering (no
  byte movement — only the shape label flips). **Breaking** for
  any downstream code that compensated for the old buggy
  behavior; drop the workaround.
- **`Qwen3Generator::from_loader`** canonicalizes cache keys to the
  HF naming convention (via `gguf_to_hf_name`) so the same generator
  works against safetensors OR GGUF loaders without builder changes.
- **`set_param_typed`** on the f32-arena backends (CPU, Metal, wgpu)
  now accepts `DType::U8` and `DType::I8` via the existing
  `set_param_bytes` path. Needed by the GGUF `Op::DequantMatMul`
  path to hand raw packed bytes to the arena. Behavior for
  F32/F16/BF16 is unchanged.
- **Pre-transposed tied LM-head embedding** in the qwen3 builder:
  computed once at graph-build time as a distinct param of shape
  `[hidden, vocab]`. The earlier scheme emitted a runtime
  `Transpose(embed_w, [1,0])` op that materialized ~600 MB per
  forward. CPU `last`-mode prefill drops from ~970 ms → ~70 ms on
  this fix alone.
- **MPSGraph lowering is opt-out** (was opt-in). Env-var name
  changed from `RLX_USE_MPSGRAPH=1` to **`RLX_DISABLE_MPSGRAPH=1`**.
  The matrix harness `model builders` example `qwen3_matrix.rs` no longer needs to
  set anything to engage the fast path.
- `WeightFormat::from_path` / `ConfigSource` / `Precision` /
  `SamArch` enums + the runner builders are re-exported as
  `rlx::run::*` (under the `models` feature).
- `rlx::QuantScheme` flat re-export added to the prelude.
- Workspace version bumped from `0.1.0` → `0.2.0` (all 23
  crates).

### Fixed

- **`Q8K` block byte count off by 16** in `QuantScheme::gguf_block_bytes()`
  (was 276, should be 292 = 4 + 256 + 32). Caught by the new
  `dequant_matmul_q8k_matches_dequant_then_matmul` integration test.
- **MPSGraph attention `MaskKind::Causal`** is now lowered correctly
  (was returning `None` from `try_lower` and falling back to the
  per-op encoder path; now uses Apple's fused SDPA with an in-graph
  constant causal mask).
- **`Op::DequantMatMul` `scheme` field** is now used by the
  CPU lowerer to dispatch to the right kernel; previously the
  GGUF schemes panicked with "scheme not implemented".
- Three pre-existing `model builders` warnings (unused
  `multihead_attention` import in `sam3/detector_decoder.rs`,
  unused `data` arg in `sam3/detector_encoder_ir.rs:add_param`,
  dead `sigmoid` fn in `sam3/tensor.rs`) cleaned up so the
  publish script's `clippy -- -D warnings` gate passes.

### Performance

- **Qwen3-0.6B prefill on Apple Silicon (Metal):** RLX beats
  Python+PyTorch+MPS in 11/23 (B, L, mode) cells, ties in 6,
  with the win margin growing from ~5% at L=32 to 1.45× at L=128.
  Beats Candle CPU on every cell tested (2.6×–9×).
- **Qwen3-0.6B Q4_K_M GGUF on Metal end-to-end:** cosine 0.976 vs
  F32 safetensors — textbook Q4_K_M loss, no NaN, top-1 plausible.

### Docs

- New `CHANGELOG.md` (this file).
- `model builders` repo README: added a Qwen3 section, runner DX section,
  per-example table, env-var matrix for the MPSGraph fast path.
- `rlx-ir/README.md`: added a `QuantScheme` table covering legacy
  Int8 + new GGUF schemes, and a Gotchas note about
  `Op::DequantMatMul`'s variable input count.
- `rlx-gguf/README.md`: replaced overclaiming feature list with an
  honest per-format table; documents the shape-convention quirk
  callers need to know about.
- Root `README.md`: new runner section, Status-by-area entries for
  Qwen3 LM + Op::DequantMatMul GGUF schemes + rlx::run.

### Performance / memory

- **Packed-weights qwen3 builder** (`build_qwen3_graph_sized_packed`):
  K-quant matmul weights stay packed in the arena and the graph
  emits `Op::DequantMatMul { scheme }` per projection. On
  Qwen3-0.6B Q4_K_M: arena drops from 2.22 GB → 1.42 GB end-to-end
  with **bit-exact parity** against the F32-load path (cosine
  1.00000, max\|Δ\| 0.000, top-1 match). End-to-end example at
  `model builders` example `qwen3_packed_inference.rs`; set
  `RLX_QWEN3_PARITY=1` to also build the F32 reference for the
  same file and report cosine.
- **`Qwen3RunnerBuilder::packed_weights(true)`** + CLI `--packed`
  flag — high-level entry to the packed-weights path. Builds the
  packed prefill graph, uploads K-quant params as U8 byte tensors
  via `set_param_typed`, exposes `Qwen3Runner::predict_logits` for
  a single forward AND `Qwen3Runner::generate_packed` for
  streaming via repeated prefills. `generate(...)` auto-routes to
  the packed path in packed mode, so the same caller-side code
  works in both modes. Trade-off: each generated token costs one
  full prefill (no decode-graph KV cache in packed mode yet —
  bucketed decode-graph machinery is still F32-only); throughput
  is ~`max_seq` × slower than the F32 streaming path but memory
  stays packed — the only path that fits 14 B+ Q4_K_M GGUFs on
  commodity Macs today.
- **Layout bug fixed** in CPU `Op::DequantMatMulGguf`: the dequant
  output is `[n, k]` row-major (GGUF byte order), not `[k, n]` —
  the original arm called `sgemm` which silently produced wrong
  outputs for `n > 1` cells. Now uses `sgemm_bt` (B transposed).
  Pinned by a new `dequant_matmul_q8k_correct_layout_for_n_gt_1`
  regression test that's specifically picked to fail under the
  old layout.

### Known limitations

- **Qwen3.5 / Qwen3.6 (`qwen35`) hybrid gated-DeltaNet + attention**:
  the unsloth/froggeric `Qwen3.5-0.8B-MTP-GGUF` and
  `Qwen3.6-27B-MTP-GGUF` files both tag `general.architecture =
  "qwen35"` (Qwen3-Next style: gated DeltaNet "linear attention"
  trunk layers interspersed with standard attention every
  `full_attention_interval`, plus an MTP head). End-to-end forward
  pipeline shipped this release:
  - `Op::GatedDeltaNet { state_size }` — new IR op + CPU
    autoregressive scan kernel mirroring
    `delta-net-base.cpp::build_delta_net_autoregressive`. Parity-
    tested against a scalar reference + per-batch state-reset test
    (`rlx-runtime/tests/cpu_gated_delta_net_parity.rs`, 2/2 green).
  - `Qwen35Config::from_gguf` + `Qwen35Weights::from_loader{,_packed}`
    — full per-layer tensor bundle. Auto-detects linear-attn vs
    full-attn layers from `full_attention_interval`; loads the MTP
    layer's NextN `eh_proj` / `enorm` / `hnorm` / optional
    `embed_tokens` / `shared_head_*`. `MatWeight::{F32, Packed}`
    enum routes K-quant matmul weights through `Op::DequantMatMul`
    when `from_loader_packed` is used.
  - `build_qwen35_graph_sized` — full prefill IR: gated-DeltaNet
    trunk (norm → joint qkv+gate split → α/β/dt + softplus gate
    → unrolled k=4 depthwise causal conv → SiLU → q/k/v split →
    L2-norm → GQA-repeat → `Op::GatedDeltaNet` → silu(z)-gated
    norm → `ssm_out`) + every-`full_attention_interval` standard
    attention block (joint Q+gate, sigmoid-gated attn output) +
    optional MTP head. 2/2 basic tests green (graph builds,
    executes, produces finite logits on both trunk + MTP outputs).
  - `Qwen35Runner` / `Qwen35RunnerBuilder` — mirrors the
    `Qwen3Runner` API; `.packed_weights(true)` opts into the K-
    quant in-arena path. `.generate(prompt_ids, n_new, on_token)`
    runs autoregressive greedy generation via repeated prefills.
  - `rlx-run qwen35` CLI subcommand + `examples/run_qwen35.rs`
    end-to-end. Flags: `--packed`, `--mtp`, `--max-tokens N`,
    `--prompt-ids 1,2,3`, `--max-seq N`.
  - Deviations from the llama.cpp reference (flagged for the
    next-slice parity oracle): standard per-axis RoPE substituted
    for the rope-sections MRoPE; depthwise k=4 conv unrolled into
    narrow+mul+add (no `Op::Conv`); per-batch state reset (no
    decode-time state cache).
  Memory: F32 dequant path needs ~1.5 GB for 0.8B (fits) /
  ~65 GB for 27B (doesn't fit). Packed path drops 27B to ~16 GB
  (fits) by keeping K-quant bytes in the arena. Numerical parity
  vs llama.cpp on a real GGUF is the next milestone.

  **Packed-loader perf**: zero-copy upload path. `take_packed`
  used to `.to_vec()` each K-quant tensor's bytes (~16 GB of
  memcpy on 27 B Q4_K_M). New flow:
  `take_packed_metadata` records `(scheme, shape)` only,
  `MatWeight::Packed` holds the loader key, and the runner uploads
  via `loader.tensor_bytes_borrowed(key) → compiled.set_param_typed`
  — bytes flow straight from mmap into the arena, no intermediate
  Vec. Also: reuse the loader's already-parsed `GgufFile` for
  `Qwen35Config::from_gguf` (was re-parsing 800+ tensor headers,
  ~10 s saved on 27 B). Builder/runner now log per-phase timing
  via `eprintln!` so future regressions surface.
- **`Op::DequantMatMul` on Metal** still falls through to the
  per-op thunk path; the GGUF schemes only have CPU lowerings
  today. On Apple GPUs the F32-load path is the working option
  until the native Metal Dequant kernel lands.
- **Streaming decode tok/s** in `Qwen3Runner::generate` recompiles
  per token in `stream(true)` mode — the bucketed compile cache
  doesn't get hit until the second pass. Fix in 0.2.0: callback
  threaded through `Qwen3Generator::generate_cached` so a single
  compile covers the whole `n_new` decode loop.
- **Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q1_0**
  GGUF formats are not decoded. Files containing them raise a
  clean "dequant for {type} not implemented yet" error.
- **27 B-class GGUF on Mac**: requires the Metal `Op::DequantMatMul`
  kernel above (108 GB F32-dequant footprint doesn't fit anywhere
  affordable). Models up to ~8 B Q4_K_M load and run today on a
  32 GB unified-memory Mac.
- **MTP heads** are now loadable end-to-end on
  `unsloth/Qwen3.6-27B-MTP-GGUF`-style files: pass
  `--use-mtp` (CLI) or `.use_mtp(true)` (runner builder) to flip
  the `GgufLoader::include_mtp` visibility; MTP tensors are drained
  into the generator's weights cache and a diagnostic logs how many
  heads were captured. Direct access via `GgufLoader::take_mtp(name)`
  is also exposed. The base generation path still runs single-token
  decode (the speculative + verify loop that would *use* the heads
  is the follow-up); inference succeeds either way.

### Internal

- `Op::DequantMatMulGguf` thunk variant added in `rlx-cpu` to
  carry the GGUF scheme through scheduling + VJP recompute paths
  cleanly.
- Workspace member layout unchanged.

---

## [0.2.0] — 2026-05

The first release with end-to-end **Qwen3 LM inference** on Apple
Silicon (safetensors + GGUF, F32, parity-checked against the
HuggingFace reference), a high-level **`rlx::run`** runner API, a
**`rlx-run`** CLI, and **GGUF K-quant dequantization** baked into
`Op::DequantMatMul`.

## [0.1.0] — 2026-04

Initial release. Tracked at [git history root].

[Unreleased]: https://github.com/MIT-RLX/rlx/compare/v0.2.13...HEAD
[0.2.13]: https://github.com/MIT-RLX/rlx/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.12
[0.2.11]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.11
[0.2.10]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.10
[0.2.9]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.9
[0.2.8]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.8
[0.2.7]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.7
[0.2.6]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.6
[0.2.5]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.5
[0.2.3]: https://github.com/MIT-RLX/rlx/releases/tag/v0.2.3

## License

GPL-3.0-only.
