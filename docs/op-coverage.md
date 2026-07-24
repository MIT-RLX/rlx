# RLX op coverage — single source of truth

This document tracks **every IR op that exists**, what each one does, its
variations, and **which backend can lower it**.

## Where the truth lives

| Concept | Source of truth |
|---------|-----------------|
| The canonical op list (`OpKind`) | [`crates/core/rlx-ir/src/op.rs`](../crates/core/rlx-ir/src/op.rs) — `pub enum OpKind` |
| The full op payloads (`Op`) + doc comments | same file — `pub enum Op` |
| Op *variations* (Activation, BinaryOp, …) | same file — the small enums near the top |
| Quant schemes | [`crates/core/rlx-ir/src/quant.rs`](../crates/core/rlx-ir/src/quant.rs) — `QuantScheme` |
| Per-backend **legalization contract** | each backend crate's `SUPPORTED_OPS` (`crates/backends/rlx-*/src/supported_ops.rs`, Vulkan/OneAPI in their `backend.rs`) — returned by `Backend::supported_ops()` |
| Per-op `supports(device, op)` heuristic | [`crates/core/rlx-runtime/src/device_ext.rs`](../crates/core/rlx-runtime/src/device_ext.rs) |

A backend only lowers ops listed in its crate-level `SUPPORTED_OPS` const. The
`LegalizeForBackend` pass in `rlx-opt` checks a graph against this set and
**fails the compile** (no silent CPU fallback) when an op isn't claimed. So the
matrix below *is* the contract — not aspirational.

> **Claim columns are generated/verified from backend `SUPPORTED_OPS`.**
> Run `just gen-op-coverage` (or `python3 scripts/gen-op-coverage.py`) after
> changing a claim; `just check-op-coverage` fails if the doc drifts.

## Backends

The 8 columns are the backends registered in the `rlx-runtime` `Backend`
registry:

| Col | Backend | Crate | Target |
|-----|---------|-------|--------|
| **CPU**  | Reference / native | `rlx-cpu`    | x86-64 / aarch64 (Accelerate AMX, AVX) — ground truth |
| **MTL**  | Apple GPU | `rlx-metal`  | Metal / MPSGraph on macOS & iOS |
| **MLX**  | Apple unified | `rlx-mlx`    | Apple Silicon unified memory (MLX) |
| **WGPU** | Cross-platform GPU | `rlx-wgpu`   | Vulkan / Metal / DX12 / WebGPU |
| **ANE**  | Apple Neural Engine | `rlx-coreml` | CoreML ML Program (static inference compiler) |
| **CUDA** | NVIDIA GPU | `rlx-cuda`   | CUDA / cuBLAS |
| **ROCm** | AMD GPU | `rlx-rocm`   | HIP / rocBLAS / MIOpen |
| **TPU**  | Tensor accelerator | `rlx-tpu`    | XLA-style INT8 path |

Not in the matrix (specialized crates that are **not** registered runtime
backends): `rlx-cortexm` (Cortex-M MCU codegen), `rlx-fpga`. They consume a
narrower hand-picked op set documented in their own crates.

Legend: ✅ = backend declares this op in `supported_ops()`. **ᵁ** = claimed then
unfused / lowered before the backend's native path. **ᴰ** = claimed for device
selection / legalize then `decompose_backward_ops` (training). **ᴴ** = host
fallback. Blank = not lowered (graph fails legalization on that device).

### Coverage at a glance

| Backend | Ops claimed | of 157 |
|---------|------------:|-------:|
| CPU  | **157** | reference (full OpKind surface; fused/control expand before thunks) |
| MLX  | **157** | broadest GPU surface (control flow + scan + conv-bwd + QAT + GroupNorm fwd+bwd + Im2Col + ArgMax/Min) |
| MTL  | **157** | Apple GPU inference + core training-bwd (Mamba `SelectiveScan`, `Sample`, `Reverse`, `ArgMax/Min`, **native fused `Gru`/`Rnn`/`Mamba2`**) |
| WGPU  | **157** | cross-platform inference + partial training-bwd (vision trio + `Reverse` + `ArgMax/Min` + **native WGSL `Gru`/`Rnn`/`Mamba2`**) |
| CUDA  | **157** | full OpKind surface (+ native `Mamba2`/`Gru`/`Rnn`/`FftButterflyStage`/`QMatMul`/`QConv2d`; DenseSolve via cuSOLVER) |
| ROCm  | **157** | mirrors CUDA (shared `.cu` + hipSOLVER DenseSolve) |
| TPU  | **154** | full OpKind surface (HLO compose for norms/QAT/conv-bwd/MaxPool/Attention bwd/AxialRope/Im2Col/ConvTranspose/PerTensor-FP8 Scaled* + host for SPD / splat / FftButterfly / DenseSolve) |
| ANE  | **157** | static inference compiler + hybrid host segments |

*(Total **157** `OpKind`s. `Mamba2` still unfuses on ANE; `Gru`/`Rnn`/`Lstm` are native host.)*

**Also at 153 (EXTRA backends, not in the 8-column matrix):** **Vulkan** and **OneAPI** — claim parity with CUDA; native SPIR-V/OpenCL for norms/fused/RNN/vision-bwd/FFT/I8 quant + host/unfuse for specialty ops.

> **Verification note (this revision):** CPU/Metal/MLX/WGPU additions are
> parity-tested on-device (Apple Silicon) and benched (see below). CUDA/ROCm
> `StopGradient` and the host-staged `Reverse`/`ArgMax`/`ArgMin`/`AxialRope2d`
> (sync → dtoh span → verified rlx-cpu kernel → htod, the existing `im2col_host`
> pattern) are *compile-verified* (`cargo check` clean) but **not run on
> NVIDIA/AMD/TPU hardware** — it mirrors the verified `Reshape`/`Cast`
> slot-aliasing identity path, so it is correct by construction, but flag it
> before relying on it. Also fixed while adding these: three CPU batch-stride
> bugs (`GroupNorm`/`ResizeNearest2x`/`AxialRope2d` under-advanced 4× for
> batch>1 — only n=1 was ever tested), a Metal `TransformRegion(ResizeNearest2x)`
> unimplemented-panic, and an MLX `LayerNorm2d` NCHW axis-grouping bug.

---

## Coverage matrix (by category)

### Sources & leaves

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Input` | Runtime-fed input placeholder | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Param` | Loadable/trainable weight tensor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Constant` | Compile-time constant tensor | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Elementwise & predication

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Activation` | Unary activation — see [Activation variants](#activation) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Cast` | DType conversion | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `StopGradient` | Identity forward; blocks gradient | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Binary` | Elementwise binary — see [BinaryOp](#binaryop) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Compare` | Elementwise compare → bool — see [CmpOp](#cmpop) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Where` | Elementwise `select(cond, a, b)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Fusion regions

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `ElementwiseRegion` | Fused elementwise chain (one kernel) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `TransformRegion` | Fused sampling/geometry chain (FKL-style) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchElementwiseRegion` | Same chain over N batch planes (horizontal fusion) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Linear algebra

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `MatMul` | Batched matrix multiply | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DotGeneral` | XLA-style general contraction (arbitrary batch/contract dims) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DenseSolve` | Dense linear solve `Ax=b` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchedDenseSolve` | Batched dense linear solve | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GroupedMatMul` | MoE grouped matmul (per-token expert routing) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LoraMatMul` | Base matmul + low-rank `A·B` LoRA update (native CPU/MLX/ANE; **all backends via decomposition**) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Normalization

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `LayerNorm` | Layer norm (last-axis) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LayerNorm2d` | Channel-wise LayerNorm on NCHW | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GroupNorm` | Group normalization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchNormInference` | Inference batch norm (frozen stats) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RmsNorm` | RMS normalization | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Attention & positional

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Attention` | Fused scaled-dot-product attention — see [MaskKind](#maskkind) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Rope` | Rotary position embedding (NeoX/GPT) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `AxialRope2d` | 2D axial rotary embedding (vision) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Shape & data movement

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Reshape` | Reshape (no data movement) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Transpose` | Permute axes | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Narrow` | Slice along an axis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Concat` | Concatenate along an axis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Expand` | Broadcast-expand singleton dims | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Gather` | Gather rows/elements by index | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Reverse` | Batch-general flip along axes (`[batch,seq,…]` seq-reverse) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ScatterAdd` | Scatter-add into output by index | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ScatterNd` | ONNX ScatterND (data+indices+updates, reduction) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ScatterElements` | ONNX ScatterElements (axis + reduction) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatherNd` | ONNX GatherND (batch_dims) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatherElements` | ONNX GatherElements / take_along_axis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ResizeNearest2x` | 2× nearest-neighbour upsample | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Interpolate3d` | Nearest NCDHW resample to explicit size | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Reduction & indexing

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Reduce` | Axis reduction — see [ReduceOp](#reduceop) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Softmax` | Softmax along an axis | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Cumsum` | Cumulative sum | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ArgMax` | Index of max along axis (f32-encoded) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ArgMin` | Index of min along axis (f32-encoded) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `TopK` | Top-k values/indices | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Sample` | Categorical / logit sampling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Random number generation

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `RngNormal` | Random-normal fill (`RandomNormalLike`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RngUniform` | Random-uniform fill (`RandomUniformLike`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Convolution & pooling

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Conv` | 2D convolution (NCHW, groups) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Im2Col` | Image→column expansion (conv lowering) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ConvTranspose2d` | Transposed conv2d (deconv) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Pool` | 2D pooling (max/avg) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Quantization — inference

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Quantize` | Float → integer quantize | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Dequantize` | Integer → float dequantize | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DequantMatMul` | MatMul w/ packed quantized weights (dequant-on-fly) — see [QuantScheme](#quantscheme) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DequantGroupedMatMul` | Grouped/MoE matmul over packed quantized expert weights | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DequantMoEWeights` | Dequantize a packed MoE expert weight bank | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `QMatMul` | Real INT8-domain matmul (int8 in/out) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `QConv2d` | Real INT8-domain conv2d | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Quantization — QAT (fake-quant)

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `FakeQuantize` | Simulated quant for QAT — see [ScaleMode](#scalemode) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FakeQuantizeBackward` | STE backward for FakeQuantize — see [SteKind](#stekind) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FakeQuantizeLSQ` | Learned-step-size fake quant (learnable scale) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FakeQuantizeLSQBackwardX` | LSQ gradient w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FakeQuantizeLSQBackwardScale` | LSQ gradient w.r.t. scale | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Sequence models (SSM / RNN / linear-attention)

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `SelectiveScan` | Mamba selective scan (S6) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatedDeltaNet` | Qwen3.5 gated delta-net linear attention | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Lstm` | LSTM recurrence | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Gru` | GRU recurrence (native CPU/Metal/WGPU/ANE host; MLX via decomposition) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Rnn` | Elman RNN recurrence (native CPU/Metal/WGPU/ANE host; MLX via decomposition) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Mamba2` | Mamba-2 SSD block (native CPU/Metal/WGPU/CUDA/ROCm; MLX/ANE via decomposition) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Fused composites

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `FusedSwiGLU` | Fused SwiGLU MLP (gate·up → down) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FusedMatMulBiasAct` | Fused matmul + bias + activation | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FusedResidualLN` | Fused residual-add + LayerNorm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FusedResidualRmsNorm` | Fused residual-add + RMSNorm | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `AdaLayerNorm` | DiT adaLN-Zero `norm(x)·(1+scale)+shift` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatedResidual` | DiT gated residual `x + gate·y` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FusedAttentionBlock` | Fused attention sub-block | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FusedTransformerLayer` | Fused full transformer layer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`AdaLayerNorm` / `GatedResidual`: native kernels on CPU / Metal / CUDA / ROCm /
wgpu / ANE (composed MIL, implicit broadcast); composed on MLX / TPU; claimed
then `unfuse_dit_modulation` on Vulkan / OneAPI / WebGL (primitive LayerNorm +
Mul/Add with NumPy broadcast — no Expand materialization of `[B,1,D]`).

Packed reverse (`AdaLayerNormBackward` / `GatedResidualBackward`): native on
CPU / Metal / CUDA / ROCm / wgpu / ANE (composed MIL) / MLX / TPU / Vulkan /
OneAPI (SPIR-V).

### Control flow

Short `Op::Scan` graphs prefer **on-device IR** via
`CompileOptions::scan_unroll_max_length` (default 64) and
`maybe_unroll_scans_budget(4096)` (`length × body_nodes`). Longer / nested Scans
and `ScanBackward*` use the shared host contract (`ScanHostDesc` /
`HostOpDesc`) — host-fallback on GPU backends, packed eval on MLX/CoreML/OneAPI.
There is **no** nested device body-ISA interpreter; the on-device path is IR
unroll so body ops run as ordinary kernels. See
[`development.md`](development.md#opscan-device-preference).

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `If` | Conditional sub-graph | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `While` | Bounded while loop | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Scan` | Scan/fold over leading axis (carry + ys) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ScanBackward` | Reverse-mode scan backward (carry grads) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ScanBackwardXs` | Scan backward w.r.t. scanned inputs `xs` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Legend: **ᴴ** = host-fallback (`ScanHostDesc` / `HostOpDesc` or packed f32);
**ᵁ** = lowered by `LowerScan` / backward decompose before HLO (must not
escape legalize). Vulkan / OneAPI also claim Scan* (host / packed).

### Complex (C64 / Wirtinger AD)

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `ComplexNormSq` | `\ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ComplexNormSqBackward` | Backward of `ComplexNormSq` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Conjugate` | Complex conjugate (Wirtinger VJP) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### FFT & signal

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Fft` | 1D FFT (forward/inverse) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FftButterflyStage` | Ternary-pruned radix-2 butterfly stage | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LogMel` | Log-mel spectrogram from FFT spectrum (Whisper) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LogMelBackward` | Backward of `LogMel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `WelchPeaks` | Welch PSD top-k peaks | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 3D Gaussian splatting

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `GaussianSplatRender` | 3DGS rasterizer (project → bin → sort → raster) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GaussianSplatRenderBackward` | 3DGS backward (scene-param grads) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GaussianSplatPrepare` | 3DGS stage 1 (project + tile-bin + sort) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GaussianSplatRasterize` | 3DGS stage 2 (per-pixel raster) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### User-extensible

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `Custom` | User-registered op via `op_registry` (FFT/eigensolve/Sparse-LU/…) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CustomFn` | User sub-graph with override AD rules (`custom_vjp`/`custom_jvp`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Backward / training ops

| Op | Description | CPU | MTL | MLX | WGPU | ANE | CUDA | ROCm | TPU |
|----|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `ReluBackward` | ReLU backward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ActivationBackward` | Generic activation backward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxPool2dBackward` | Max-pool backward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `MaxPool3dBackward` | Max-pool 3D backward (NCDHW) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Conv2dBackwardInput` | Conv2d grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Conv2dBackwardWeight` | Conv2d grad w.r.t. weight | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Conv3dBackwardInput` | Conv3d grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `Conv3dBackwardWeight` | Conv3d grad w.r.t. weight | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SoftmaxCrossEntropy` | Dense/soft-label softmax cross-entropy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SoftmaxCrossEntropyWithLogits` | Fused softmax + cross-entropy loss | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SoftmaxCrossEntropyBackward` | Backward of softmax cross-entropy | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `AttentionBackward` | Attention backward — see [AttentionBwdWrt](#attentionbwdwrt) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LayerNormBackwardInput` | LayerNorm grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LayerNormBackwardGamma` | LayerNorm grad w.r.t. gamma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RmsNormBackwardInput` | RMSNorm grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RmsNormBackwardGamma` | RMSNorm grad w.r.t. gamma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RmsNormBackwardBeta` | RMSNorm grad w.r.t. beta | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RopeBackward` | RoPE backward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GroupNormBackwardInput` | GroupNorm grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GroupNormBackwardGamma` | GroupNorm grad w.r.t. gamma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GroupNormBackwardBeta` | GroupNorm grad w.r.t. beta | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchNormInferenceBackwardInput` | BatchNorm-inference grad w.r.t. input | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchNormInferenceBackwardGamma` | BatchNorm-inference grad w.r.t. gamma | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `BatchNormInferenceBackwardBeta` | BatchNorm-inference grad w.r.t. beta | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CumsumBackward` | Cumsum backward (reverse cumsum) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatherBackward` | Gather backward (scatter-add of grads) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `AdaLayerNormBackward` | Packed DiT adaLN reverse `[dx∥dscale∥dshift]` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `GatedResidualBackward` | Packed DiT gated residual reverse `[dx∥dy∥dgate]` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`AdaLayerNormBackward` / `GatedResidualBackward` on **ANE**: native composed MIL
in `COREML_NATIVE_BACKWARD_OPS` (legalize keeps the packed op; lowering emits
implicit-broadcast + concat pack). Native composed lowering on **MLX** /
**TPU**. Native SPIR-V on **Vulkan** (`shaders/*_backward.comp`) and **OneAPI**
(`kernels/*_backward.cl`, host-fallback when kernels are not embedded).

---

## Notable gaps (read this before assuming an op runs somewhere)

**Defined in IR but lowered by *no* backend** — *(none — every OpKind now lowers
on at least CPU as of this revision).*

> **Decomposition gating (`FUSED_KINDS`).** An op without a native kernel runs
> only if the rewrite pipeline decomposes it — and the `unfuse` pass fires only
> for `OpKind`s in `rewrite.rs::FUSED_KINDS`. `LoraMatMul` *had* an `unfuse`
> decomposition arm but was **missing from that set**, so standalone LoRA
> hard-failed on Metal/WGPU/CUDA/ROCm/TPU (it only decomposed when some other
> fused op happened to be present). Adding it to `FUSED_KINDS` closes it on every
> backend (verified exact vs CPU). `AxialRope2d` and `FakeQuantize` still
> hard-fail where not native — they have **no** decomposition arm; `FakeQuantize`
> was prototyped but the primitive chain diverged per-backend (Round-tie /
> Expand), so it needs a native kernel or a backend round-fix, not a quick rule.

> **`Gru` / `Rnn` / `Mamba2`** — native fused kernels on **CPU** (`execute_{gru,rnn,mamba2}_f32`), **Metal** (native MSL), **WGPU** (native WGSL), and **CUDA/ROCm** (shared `.cu`; `state_size`/`hidden` ≤ 256 with host-staged fallback). Verified by `metal_rnn_native` / `wgpu_rnn_native`. On **MLX** they run via the `unfuse` → primitives decomposition (which *is* MLX's on-GPU path). All paths match the PyTorch/ONNX / SSD reference.

**Host-by-design specialty** (claimed on GPU backends via packed HostOp; pin to `Device::Cpu` for the reference path):
`CustomFn`.

> **`DenseSolve` / `BatchedDenseSolve`** — native on **CUDA** (cuSOLVER / cuBLAS batched LU) and **ROCm** (hipSOLVER / hipBLAS batched). **Vulkan** / **OneAPI** / **wgpu** use packed [`HostOpDesc`](../crates/backends/rlx-cpu/src/thunk/host_op.rs) → rlx-cpu LAPACK (`sgesv`) on the mapped / USM f32 arena (same contract as ScanBackward). There is **no** device-side oneMKL LAPACK link on OneAPI — DenseSolve stays host LAPACK there. HostOp fallback for non-F32 on all GPU backends.

> **`QMatMul` / `QConv2d`** — native shared `.cu` on **CUDA** / **ROCm** (packed I8 arena); also on **CPU** / Apple / wgpu / TPU.

> **`ComplexNormSq` / `ComplexNormSqBackward` / `Conjugate`** — native on
> **CPU** / **Metal** (MSL) / **WGPU** (WGSL) / **CUDA** / **ROCm**
> (`complex_wirtinger.cu`). **MLX** / **ANE** still host-eval interleaved C64.

**Single-backend exclusives / notes:**

- `QMatMul`, `QConv2d` — real INT8 I/O; native on CUDA/ROCm (packed-byte arena) and TPU; other GPUs may host-stage.
- `If`, `While` — native bounded-unroll on **MLX**; CUDA/ROCm/wgpu expand via `rlx-unfuse` (`expand_if` / `expand_while`).

**Inference-only accelerators:** ANE (CoreML) and TPU declare **no native
backward kernels** in the base inference claim — ANE training uses
`COREML_BACKWARD_OPS` (decompose) plus `COREML_NATIVE_BACKWARD_OPS` (MIL) under
the `training` feature; TPU still omits reverse. ANE also omits
`RngNormal`/`RngUniform` from some paths, fusion internals
(`ElementwiseRegion`/`TransformRegion`/`BatchElementwiseRegion`), and the splat
family.

**ROCm ≈ CUDA** for host GPU training ops that share `.cu` kernels
(including `MaxPool2dBackward`, `Conv2dBackwardInput` /
`Conv2dBackwardWeight`, and `FusedConvBiasAct` via the epilogue fallback).

> **CPU caveat:** `supports(Device::Cpu, _)` in `device_ext.rs` returns `true`
> unconditionally ("reference is ground truth"), but the *legalization* contract
> is `CPU_SUPPORTED_OPS` (94 ops). Where the two disagree (e.g. `Gru` may have a
> reference thunk under active development but isn't in the const), the const
> wins for compilation — the op must be added to `CPU_SUPPORTED_OPS` before a
> compiled graph can use it on CPU.

---

## Op variations

These are the enum payloads that multiply a single `OpKind` into many concrete
behaviors. They live in `crates/core/rlx-ir/src/op.rs` (and `quant.rs`).

### Activation

`Op::Activation(Activation)` — every backend that supports `Activation` supports
the full set:

`Gelu`, `GeluApprox`, `Silu`, `Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt`,
`Rsqrt`, `Neg`, `Abs`, `Sin`, `Cos`, `Tan`, `Atan`, `Round`.

> `Round` is half-to-even with STE (identity) backward — a primitive for
> hand-rolled quant chains.

### BinaryOp

`Op::Binary(BinaryOp)`: `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `Pow`.

### CmpOp

`Op::Compare(CmpOp)` → bool tensor: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`.

### ReduceOp

`Op::Reduce { op: ReduceOp, .. }`: `Sum`, `Mean`, `Max`, `Min`, `Prod`.

### MaskKind

`Op::Attention { mask: MaskKind, .. }`:

- `None` — full bidirectional, no mask load.
- `Causal` — autoregressive; upper-triangle generated in-kernel, no `seq²` tensor.
- `SlidingWindow(w)` — `qi` attends to `ki ∈ [qi-w, qi]`.
- `Custom` — read mask values from the 4th input (`[batch, key_len]`, BERT padding).
- `Bias` — additive `[batch, heads, q, k]` bias added to scores (DETR boxRPB, ALiBi).

### AttentionBwdWrt

`Op::AttentionBackward { wrt: AttentionBwdWrt, .. }`: `Query`, `Key`, `Value`.

### ScaleMode

`Op::FakeQuantize { scale_mode: ScaleMode, .. }`:

- `PerBatch` — recompute `s[c] = max(|x|)/q_max` each call (1 input).
- `EMA { decay }` — running scale in a state tensor (2 inputs); typical `decay=0.99`.
- `Fixed` — use the pre-calibrated state tensor as-is (2 inputs).

> MLX supports `PerBatch` + `Fixed`; `EMA` returns a clear lowering error.

### SteKind

`Op::FakeQuantizeBackward { ste: SteKind, .. }` — STE for the round step:
`Identity`, `ClippedIdentity`, `Tanh`, `HardTanh`.

### QuantScheme

`Op::DequantMatMul { scheme }` / `DequantGroupedMatMul` / `DequantMoEWeights`
(`crates/core/rlx-ir/src/quant.rs`). GGUF schemes pack scales/mins inside the weight
bytes (2 tensor inputs); non-GGUF schemes take separate scale/zp (4 inputs).

**Canonical backend matrix:** [gguf-backend-paths.md](gguf-backend-paths.md)
(scheme ids, env toggles, P0–P5 code map). Summary also inlined below.

| Family | Schemes |
|--------|---------|
| Linear int | `Int8Block{block_size}`, `Int8BlockAsym{block_size}`, `Int4Block{block_size}` |
| FP8 / FP4 | `Fp8E4m3`, `Fp8E5m2`, `Nvfp4Block` |
| GGUF K-quant | `GgufQ2K`, `GgufQ3K`, `GgufQ4K`, `GgufQ5K`, `GgufQ6K`, `GgufQ8K` |
| GGUF legacy | `GgufQ4_0`, `GgufQ4_1`, `GgufQ5_0`, `GgufQ5_1`, `GgufQ8_0` |
| GGUF IQ (LUT) | `GgufIQ4NL`, `GgufIQ4XS`, `GgufIQ2XXS`, `GgufIQ2XS`, `GgufIQ2S`, `GgufIQ3XXS`, `GgufIQ3S`, `GgufIQ1S`, `GgufIQ1M` |
| GGUF ternary | `GgufTQ1_0`, `GgufTQ2_0` |
| GGUF FP4 micro | `GgufMXFP4`, `GgufNVFP4` |

> **Per-backend GGUF execution paths** (scheme ids 0–23 shared across GPU
> backends via `gguf_scheme_id`):
>
> | Backend | GPU dequant | Fused GEMV (`m=1`) | On-device constexpr (ANE) | Host fallback |
> |---------|-------------|--------------------|---------------------------|---------------|
> | **CPU** | — | block-wise fused matmul in `rlx-cpu` | — | always available |
> | **Metal** | `dequant_gguf` MSL → MPS matmul | Q4_K, Q4_0, Q4_1, Q8_0, IQ4NL, IQ2_XXS/XS/S, IQ3_XXS/S, **IQ1_S/M** | Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4NL, Q4/5/8_K, **Q2/3/6_K** (`mul`+`sub`) | `RLX_METAL_DEQUANT_GPU_DISABLE=1`; TQ/MX without MSL branch |
> | **CUDA / ROCm** | `dequant_gguf` → cuBLAS/rocBLAS | — | — | same disable flag pattern as Metal |
> | **WGPU** | `dequant_gguf.wgsl` → `matmul_bt` when arena scratch fits; grouped MoE GPU when scratch fits | — | — | `gguf_host` when scratch exceeds `max_buffer_size` |
> | **ANE (CoreML)** | hybrid host segments for exotic schemes | — | see Metal ANE column; K with per-element scales use `[nb,32]` tensors | `RLX_COREML_HOST_DEQUANT=1` bakes full f32 weights |
> | **TPU** | — | — | — | **compile-time** host dequant of `Constant` weights → f32 HLO dot; `Param` weights must be pre-baked |
> | **MLX** | host dequant + cache | — | — | primary Apple path when MLX feature enabled |
>
> **Scheme id map (legacy tail):** Q4_0 = 19, Q8_0 = 20, Q4_1 = 21, Q5_0 = 22, Q5_1 = 23.
> **GgmlType → IR:** `rlx_cpu::quant_scheme_for_ggml`.
>
> **Metal FP8 / NVFP4** (`Fp8E4m3`, `Fp8E5m2`, `Nvfp4Block`) use native
> `dequant_matmul_fp8` / `dequant_matmul_nvfp4` MSL when the graph has no
> pending deferred host ops; otherwise CPU thunks in `rlx-cpu`.
>
> **WGPU caveats:** byte offsets are u32 (arenas ≥ 4 GB need host path);
> `DequantGroupedMatMul` uses a GPU path when arena scratch fits (see
> [gguf-backend-paths.md](gguf-backend-paths.md)); encode→dequant parity
> tests cover IQ/TQ/MX in `rlx-wgpu/tests/gguf_dequant_parity.rs`.
>
> **TPU caveats:** `lower_dequant_matmul_gguf` materialises f32 weights at
> HLO emission time — runtime `Param` uploads are not dequantized on device.
> Bake weights as `Op::Constant` or pre-dequantize before `set_param`.
>
> The op-level ✅ column only states that `DequantMatMul` is lowerable on that
> backend; individual `QuantScheme` variants may still route through host
> dequant for a given chip or graph shape.

---

## Benchmarking the ops

`crates/tooling/rlx-bench/examples/bench_new_ops.rs` measures each new op across
CPU/Metal/MLX/WGPU — **validity** (max abs diff vs the CPU reference),
**latency** (median, synchronous), **throughput** (Gelem/s), **bandwidth**
(effective GB/s), and a **RAM/size-limit sweep** (largest working set that runs
per device). Run:

```sh
cargo run -p rlx-bench --release --example bench_new_ops --features metal,mlx,gpu
```

DiT packed reverse fused-vs-unfused (`AdaLayerNormBackward` /
`GatedResidualBackward`):

```sh
just throttle
cargo run -p rlx-bench --release --example bench_dit_modulation
cargo run -p rlx-bench --release --example bench_dit_modulation --features metal
```

Median backward latency (µs, warmup=5, runs=25, `RLX_ALLOW_THROTTLE=1`) on
Apple M4 Pro / aarch64 (2026-07-17):

| Device | Shape `[B,S,D]` | AdaLayerNormBackward fused | unfused | speedup | GatedResidualBackward fused | unfused | speedup |
|--------|-----------------|---------------------------:|--------:|--------:|----------------------------:|--------:|--------:|
| CPU | `[2,128,64]` | 106 | 199 | 1.88× | 77 | 134 | 1.74× |
| CPU | `[4,256,128]` | 512 | 765 | 1.49× | 263 | 350 | 1.33× |
| CPU | `[8,512,256]` | 3154 | 4587 | 1.45× | 1599 | 2173 | 1.36× |
| Metal | `[2,128,64]` | 911 | 354 | 0.39× | 229 | 219 | 0.96× |
| Metal | `[4,256,128]` | 1127 | 618 | 0.55× | 390 | 373 | 0.96× |
| Metal | `[8,512,256]` | 2620 | 2930 | 1.12× | 1433 | 1535 | 1.07× |

Packed reverse wins on CPU at all shapes; Metal crosses over near `[8,512,256]`
(AdaLN fused faster, gate roughly even).

## Maintenance checklist

When adding or wiring an op:

1. Add the variant to `OpKind` **and** `Op` in `crates/core/rlx-ir/src/op.rs`; map it in `Op::kind()`.
2. Add it to each backend's `*_SUPPORTED_OPS` const in `crates/core/rlx-runtime/src/backend/` (per-backend `*_backend.rs`) as kernels land.
3. If a backend rejects specific *variants* (e.g. MLX `ScaleMode::EMA`), keep the per-op guard in `device_ext.rs`.
4. Refresh this table (op list, category table, the gaps section, and the "at a glance" counts).
5. Counts to keep honest: **113 `OpKind`s total** as of this revision (added `Reverse`).
