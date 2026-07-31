// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Operation types — every tensor op in the RLX IR.
//!
//! Designed for pattern-matching fusion: ops are grouped by category so
//! fusion passes can reason about them structurally.

use crate::DType;

/// Unary element-wise activation functions.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    Gelu,
    GeluApprox,
    Silu, // SwiGLU gate activation
    Relu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Neg,
    Abs,
    /// `sin(x)`. Backward: `dx = upstream · cos(x)`.
    Sin,
    /// `cos(x)`. Backward: `dx = -upstream · sin(x)`.
    Cos,
    /// `tan(x)`. Backward: `dx = upstream · sec²(x) = upstream · (1 + tan²(x))`.
    Tan,
    /// `atan(x)`. Backward: `dx = upstream · (1 / (1 + x²))`.
    Atan,
    /// Reciprocal `1 / x`.
    Recip,
    /// Round to nearest integer (half-to-even), in f32.
    /// Forward: `x.round()`. Backward: STE — treats as identity, so
    /// the gradient passes through unchanged. Useful as a primitive
    /// for composing custom quantization schemes (Mul-by-recip-scale
    /// → Round → Clamp → Mul-by-scale = a hand-rolled FakeQuantize
    /// that the elementwise-region pass can fuse into a single kernel).
    Round,
    /// `floor(x)`. Backward: 0 (piecewise-constant).
    Floor,
    /// `ceil(x)`. Backward: 0.
    Ceil,
    /// `sign(x)` = -1/0/+1 (sign of zero is 0). Backward: 0.
    Sign,
    /// `softplus(x) = ln(1 + eˣ)`. Backward: `dx = u·sigmoid(x)`.
    Softplus,
    /// ELU (α=1): `x if x>0 else eˣ-1`. Backward: `u·(1 if x>0 else eˣ)`.
    Elu,
    /// `erf(x)` (Gauss error function). Backward: `dx = u·(2/√π)·e^(-x²)`.
    Erf,
    /// HardSwish `x·clamp(x+3,0,6)/6`. Backward: 0/1/`(2x+3)/6` piecewise.
    HardSwish,
    /// HardSigmoid `clamp(x/6+0.5,0,1)`. Backward: `1/6` on `|x|<3` else 0.
    HardSigmoid,
    /// Mish `x·tanh(softplus(x))`.
    Mish,
    /// Softsign `x/(1+|x|)`. Backward: `1/(1+|x|)²`.
    Softsign,
    /// LogSigmoid `log(σ(x)) = min(x,0) − ln(1+e^(−|x|))`. Backward: `σ(−x)`.
    LogSigmoid,
}

impl Activation {
    /// Activations that may be folded into a fused `ElementwiseRegion` /
    /// `TransformRegion` chain. The newer scalar activations
    /// (`Floor`/`Ceil`/`Sign`/`Softplus`/`Elu`) run only on the standalone
    /// kernel path for now — the per-backend fused-region kernels don't yet
    /// carry their opcodes, and their `default` passthrough would silently
    /// mis-evaluate them. Keeping them unfused is correct; a fused kernel is a
    /// perf follow-up.
    pub fn region_fusable(self) -> bool {
        !matches!(
            self,
            Activation::Floor
                | Activation::Ceil
                | Activation::Sign
                | Activation::Softplus
                | Activation::Elu
                | Activation::Erf
                | Activation::HardSwish
                | Activation::HardSigmoid
                | Activation::Mish
                | Activation::Softsign
                | Activation::LogSigmoid
        )
    }
}

/// Scale-tracking strategy for `Op::FakeQuantize`. Determines how
/// the per-channel `s[c]` is computed each forward pass.
///
/// * `PerBatch` — recompute `s[c] = max(|x|) / q_max` from the
///   current data on every call. Simple, no extra inputs, but
///   noisy for activations (max-abs jumps batch-to-batch).
///
/// * `EMA { decay }` — keep a running `s[c]` in a state tensor
///   (passed as a second op input). On each call, blend the
///   current per-batch max-abs into the state via
///   `state' = decay·state + (1-decay)·max_abs`. Smooth scale
///   over training, makes activation-QAT actually trainable.
///   Typical `decay = 0.99`.
///
/// * `Fixed` — never recompute. The state tensor's value is
///   used as-is each call (set once at construction or by the
///   caller). Useful when scales are pre-calibrated.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub enum ScaleMode {
    #[default]
    PerBatch,
    EMA {
        decay: f32,
    },
    Fixed,
}

impl Eq for ScaleMode {}
impl std::hash::Hash for ScaleMode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ScaleMode::PerBatch => state.write_u8(0),
            ScaleMode::EMA { decay } => {
                state.write_u8(1);
                state.write_u32(decay.to_bits());
            }
            ScaleMode::Fixed => state.write_u8(2),
        }
    }
}

/// Straight-through estimator variants for `Op::FakeQuantize`'s
/// backward. The forward is the same regardless: discrete
/// `clamp(round(x/s)) * s`. The choice here affects only the
/// gradient w.r.t. `x` during training.
///
/// * `Identity` — `dx = upstream`. The original STE; treats the
///   round as identity in the backward direction. Simplest, fine
///   for moderate bit widths (i4 / i8).
///
/// * `ClippedIdentity` — `dx = upstream * (|x| ≤ q_max·s)`. Zero
///   the gradient when the input was outside the quantization
///   range (i.e. the clamp activated). Stops the optimizer from
///   pushing weights further into saturation.
///
/// * `Tanh` — `dx = upstream * (1 - tanh²(x/s))`. Smooth surrogate
///   for the round step. Slowly attenuates the gradient as `|x|`
///   approaches `q_max·s`. Often best on tight bit widths (i2).
///
/// * `HardTanh` — `dx = upstream * (1 - |x/(q_max·s)|).max(0)`.
///   Piecewise-linear cousin of `Tanh`; cheaper to compute and
///   nearly as effective.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SteKind {
    #[default]
    Identity,
    ClippedIdentity,
    Tanh,
    HardTanh,
}

/// Binary element-wise operations.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Pow,
    /// C `fmod` (sign of dividend) — matches Rust `%`. Differentiable in `a`
    /// (`∂/∂a = 1`); `b` treated as the constant modulus (`∂/∂b = 0`).
    Mod,
    /// Bitwise ops on integer-valued operands (GPU: `(int)a op (int)b`).
    /// Non-differentiable (zero gradient).
    BitAnd,
    BitOr,
    BitXor,
    /// Left/right shift: `a << b` / `a >> b` on integer-valued operands.
    Shl,
    Shr,
    /// `atan2(a, b)` — quadrant-aware arctangent (angle of the point `(b, a)`,
    /// i.e. Rust's `a.atan2(b)`). Smooth and differentiable in both operands:
    /// `∂/∂a = b/(a²+b²)`, `∂/∂b = -a/(a²+b²)`. Standalone kernel only (not
    /// region-fusable — the fused-region kernels don't carry its opcode).
    Atan2,
}

impl BinaryOp {
    /// Ops that may fold into a fused `ElementwiseRegion`. The newer ops
    /// (`Mod` + bitwise) run only on the standalone binary kernel for now — the
    /// per-backend fused-region kernels don't carry their opcodes yet (a
    /// perf follow-up), and their `default` would silently mis-evaluate them.
    pub fn region_fusable(self) -> bool {
        matches!(
            self,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Max
                | BinaryOp::Min
                | BinaryOp::Pow
        )
    }
}

/// Comparison operations (return Bool tensor).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Reduction applied when ONNX-style [`Op::ScatterNd`] writes updates.
///
/// Matches the ONNX ScatterND `reduction` attribute (opset 16+): `none`
/// overwrites; `add`/`mul`/`max`/`min` combine with the existing value
/// (required for duplicate indices).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScatterNdReduction {
    #[default]
    None,
    Add,
    Mul,
    Max,
    Min,
}

/// Which factor of a thin SVD `A = U·diag(S)·Vᵀ` an [`Op::Svd`] node yields.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SvdPart {
    /// `U` `[m, k]` (left singular vectors, `k = min(m,n)`).
    U,
    /// `S` `[k]` (singular values, descending).
    S,
    /// `Vᵀ` `[k, n]` (right singular vectors, transposed).
    Vt,
}

/// Which factor of a thin QR `A = Q·R` an [`Op::Qr`] node yields.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QrPart {
    /// `Q` `[m, k]` (orthonormal columns, `k = min(m,n)`).
    Q,
    /// `R` `[k, n]` (upper-triangular).
    R,
}

/// What kind of attention mask the kernel should apply.
///
/// Borrowed from MAX's `nn/attention/mha_mask.mojo` pattern: one
/// attention kernel handles all variants by branching on
/// the mask kind, instead of forcing every caller to materialize a mask
/// tensor. The win is two-fold:
///   1. **`None`** — single unpadded sequence: no mask load, no per-key
///      compare in the inner loop.
///   2. **`Causal`** — autoregressive decode: kernel generates the upper-
///      triangular fill from `(qi, ki)` directly; no `seq²` mask tensor
///      ever exists.
///
/// `Custom` is the existing path — read mask values from the 4th input.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskKind {
    /// No masking — every position attends to every position.
    None,
    /// Causal (autoregressive) — position `qi` attends only to `ki <= qi`.
    Causal,
    /// Sliding window — position `qi` attends to `ki ∈ [qi - w, qi]`.
    SlidingWindow(usize),
    /// Read mask values from the input tensor (default; matches BERT
    /// padding-mask behavior). Tensor shape `[batch, key_len]` with
    /// `1.0` = valid, `<0.5` = ignored.
    Custom,
    /// Additive per-head, per-query bias tensor
    /// `[batch, num_heads, query_len, key_len]` added to the
    /// `QK^T · scale` scores before softmax. Lets DETR-style boxRPB
    /// and other learned position biases reuse the fast `Op::Attention`
    /// path instead of decomposing into matmul + add + softmax + matmul.
    Bias,
}

/// Which forward input an [`Op::AttentionBackward`] node differentiates.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionBwdWrt {
    Query,
    Key,
    Value,
}

/// Reduction operations along specified axes.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

/// PLAN L4: discriminant for each [`Op`] variant. Used by
/// [`Op::kind`] + the `Backend::supported_ops` trait method to declare
/// which ops a backend can lower; the `LegalizeForBackend` pass in
/// `rlx-opt` checks the graph against this set and fails the compile
/// when an unsupported op is present (instead of silent fallback).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Input,
    Param,
    Constant,
    Activation,
    Cast,
    StopGradient,
    Quantize,
    Dequantize,
    FakeQuantize,
    FakeQuantizeLSQ,
    FakeQuantizeLSQBackwardX,
    FakeQuantizeLSQBackwardScale,
    Binary,
    Compare,
    Where,
    Fma,
    ElementwiseRegion,
    /// Fused sampling / geometry chain (FKL-style transform region).
    TransformRegion,
    /// Same element-wise chain over multiple batch planes (horizontal fusion).
    BatchElementwiseRegion,
    MatMul,
    DotGeneral,
    DenseSolve,
    BatchedDenseSolve,
    Cholesky,
    TriangularSolve,
    Det,
    LogDet,
    Sort,
    ArgSort,
    Svd,
    Qr,
    LayerNorm,
    LayerNorm2d,
    GroupNorm,
    BatchNormInference,
    RmsNorm,
    ResizeNearest2x,
    /// Nearest-neighbor resample of NCDHW spatial dims to an explicit size.
    Interpolate3d,
    Attention,
    Rope,
    AxialRope2d,
    Reshape,
    Transpose,
    Narrow,
    Concat,
    Expand,
    Gather,
    Reverse,
    Pad,
    Slice,
    Clamp,
    Tile,
    Trilu,
    Reduce,
    Softmax,
    Cumsum,
    CumProd,
    CumMax,
    ArgMax,
    ArgMin,
    TopK,
    Sample,
    /// ONNX `RandomNormalLike` — shape from input 0, output filled at runtime.
    RngNormal,
    /// ONNX `RandomUniformLike`.
    RngUniform,
    Conv,
    Im2Col,
    ConvTranspose2d,
    Conv3d,
    ConvTranspose3d,
    Pool,
    ReluBackward,
    ActivationBackward,
    FakeQuantizeBackward,
    ComplexNormSq,
    ComplexNormSqBackward,
    Conjugate,
    MaxPool2dBackward,
    Conv2dBackwardInput,
    Conv2dBackwardWeight,
    MaxPool3dBackward,
    Conv3dBackwardInput,
    Conv3dBackwardWeight,
    SoftmaxCrossEntropy,
    SoftmaxCrossEntropyWithLogits,
    SoftmaxCrossEntropyBackward,
    AttentionBackward,
    LayerNormBackwardInput,
    LayerNormBackwardGamma,
    RmsNormBackwardInput,
    RmsNormBackwardGamma,
    RmsNormBackwardBeta,
    RopeBackward,
    GroupNormBackwardInput,
    GroupNormBackwardGamma,
    GroupNormBackwardBeta,
    BatchNormInferenceBackwardInput,
    BatchNormInferenceBackwardGamma,
    BatchNormInferenceBackwardBeta,
    CumsumBackward,
    GatherBackward,
    GroupedMatMul,
    DequantGroupedMatMul,
    DequantGroupedMatMulMlx,
    DequantMoEWeights,
    ScatterAdd,
    ScatterNd,
    ScatterElements,
    GatherNd,
    GatherElements,
    LoraMatMul,
    PartitionedConv,
    DequantMatMul,
    QMatMul,
    QConv2d,
    ScaledMatMul,
    ScaledQuantize,
    ScaledQuantScale,
    ScaledDequantize,
    SelectiveScan,
    GatedDeltaNet,
    Lstm,
    Gru,
    Rnn,
    Mamba2,
    FusedSwiGLU,
    FusedMatMulBiasAct,
    FusedConvBiasAct,
    FusedResidualLN,
    FusedResidualRmsNorm,
    FusedAttentionBlock,
    FusedTransformerLayer,
    If,
    While,
    Scan,
    ScanBackward,
    ScanBackwardXs,
    /// CPU reference 3D Gaussian splat raster (project → bin → sort → raster).
    /// See [`Op::GaussianSplatRender`].
    GaussianSplatRender,
    /// Backward of [`Op::GaussianSplatRender`] — packed scene parameter gradients.
    GaussianSplatRenderBackward,
    /// Project + tile bin + sort + ray grid (strict IR splat stage 1).
    GaussianSplatPrepare,
    /// Per-pixel raster from prepared buffers (strict IR splat stage 2).
    GaussianSplatRasterize,
    /// User-registered op dispatched through `op_registry`. All
    /// custom ops (Sparse-LU, FFT, eigensolve, ...) share this kind;
    /// the per-op identity lives in `Op::Custom::name`.
    Custom,
    /// User-defined sub-graph with optional override AD rules. See
    /// [`Op::CustomFn`] / [`crate::Graph::custom_fn`].
    CustomFn,
    /// 1D FFT primitive (forward or inverse) — see [`Op::Fft`].
    Fft,
    /// Ternary pruned radix-2 butterfly stage — see [`Op::FftButterflyStage`].
    FftButterflyStage,
    /// Whisper-style log-mel from block-layout FFT spectrum — see [`Op::LogMel`].
    LogMel,
    /// Backward of [`Op::LogMel`] w.r.t. block-layout spectrum input 0.
    LogMelBackward,
    /// Welch PSD top-K spikes from block-layout FFT spectra — see [`Op::WelchPeaks`].
    WelchPeaks,

    // ── Riemannian / SPD-manifold layers ────────────────────────
    /// SPDNet bilinear mapping — see [`Op::BiMap`].
    BiMap,
    /// SPDNet eigenvalue rectification — see [`Op::ReEig`].
    ReEig,
    /// SPDNet matrix-log layer — see [`Op::LogEig`].
    LogEig,
    /// SPD batch-norm affine transport — see [`Op::SpdBatchNorm`].
    SpdBatchNorm,
    /// Batch Fréchet/Karcher mean of SPD matrices — see [`Op::SpdKarcherMean`].
    SpdKarcherMean,
    /// Backward of [`Op::ReEig`].
    ReEigBackward,
    /// Backward of [`Op::LogEig`].
    LogEigBackward,
    /// Backward of [`Op::SpdBatchNorm`] w.r.t. inputs.
    SpdBatchNormBackwardX,
    /// Backward of [`Op::SpdBatchNorm`] w.r.t. bias G.
    SpdBatchNormBackwardG,
    /// Weighted batch Fréchet/Karcher mean — see [`Op::SpdKarcherMeanWeighted`].
    SpdKarcherMeanWeighted,
    /// AIRM logarithm map at an arbitrary base — see [`Op::SpdLogMap`].
    SpdLogMap,
    /// AIRM exponential map at an arbitrary base — see [`Op::SpdExpMap`].
    SpdExpMap,
    /// AIRM parallel transport of a tangent vector — see [`Op::SpdParallelTransport`].
    SpdParallelTransport,
    /// Batched SPD spectral matrix function — see [`Op::SpdMatrixFnBatch`].
    SpdMatrixFnBatch,
    /// Backward of [`Op::SpdLogMap`].
    SpdLogMapBackward,
    /// Backward of [`Op::SpdExpMap`].
    SpdExpMapBackward,
    /// Backward of [`Op::SpdParallelTransport`].
    SpdParallelTransportBackward,
    /// Backward of [`Op::SpdMatrixFnBatch`].
    SpdMatrixFnBatchBackward,
    /// Symmetric eigendecomposition — see [`Op::Eigh`].
    Eigh,
    /// Backward of [`Op::Eigh`].
    EighBackward,
    /// Batched symmetric eigendecomposition — see [`Op::EighBatch`].
    EighBatch,
    /// Backward of [`Op::EighBatch`].
    EighBatchBackward,

    // ── Diffusion Transformer (DiT) modulation ──────────────────
    /// Fused adaptive-norm modulation `norm(x)·(1+scale)+shift` — see
    /// [`Op::AdaLayerNorm`].
    AdaLayerNorm,
    /// Backward of [`Op::AdaLayerNorm`] — packed `[dx ∥ dscale ∥ dshift]`.
    AdaLayerNormBackward,
    /// Gated residual `x + gate·y` with broadcast gate — see
    /// [`Op::GatedResidual`].
    GatedResidual,
    /// Backward of [`Op::GatedResidual`] — packed `[dx ∥ dy ∥ dgate]`.
    GatedResidualBackward,
}

/// An operand inside a fused [`ChainStep`] — either a graph-level input
/// to the [`Op::ElementwiseRegion`] (by index 0..num_inputs) or the
/// result of a previous step in the chain (by index 0..step_position).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainOperand {
    Input(u32),
    Step(u32),
}

/// One step in a fused element-wise chain. Each step produces exactly
/// one scalar result (per element); later steps can refer to it via
/// [`ChainOperand::Step`]. The whole chain runs per element in registers.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum ChainStep {
    Activation(Activation, ChainOperand),
    Cast(DType, ChainOperand),
    Binary(BinaryOp, ChainOperand, ChainOperand),
    Compare(CmpOp, ChainOperand, ChainOperand),
    /// 3-input element-wise select: `cond ? on_true : on_false`. Mirrors
    /// `Op::Where` inside a chain. `cond` is treated as truthy iff
    /// non-zero. Lets the optimizer fold attention masks / clamp-style
    /// patterns into a single region kernel instead of breaking the
    /// chain at the first `Op::Where`.
    Where(ChainOperand, ChainOperand, ChainOperand),
}

/// Pre-region memory transform fused into [`Op::ElementwiseRegion`].
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RegionPrologue {
    #[default]
    None,
    /// Input is half-resolution NCHW; output shape is 2× H×W (nearest 2×).
    ResizeNearest2x,
}

/// One sampling step in [`Op::TransformRegion`].
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum TransformStep {
    ResizeNearest2x(ChainOperand),
}

/// Pairing convention for [`Op::Rope`]. Different model families bake RoPE
/// into their weights differently, so the rotation kernel needs a flavor:
///
/// - [`RopeStyle::NeoX`] (default): HF / GPT-NeoX "rotate-half" — dimension `i`
///   pairs with `i + n_rot/2`. Used by HF-safetensors checkpoints (Llama, Qwen,
///   Gemma, …).
/// - [`RopeStyle::GptJ`]: GPT-J / llama.cpp `NORM` — adjacent pairs `(2i, 2i+1)`.
///   GGUF Llama weights are permuted by the HF→GGUF converter for this layout,
///   so GGUF-backed Llama inference must rotate with this flavor.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RopeStyle {
    #[default]
    NeoX,
    GptJ,
}

/// Pre-modulation normalization flavor for [`Op::AdaLayerNorm`] (DiT adaLN-Zero).
/// Both normalize over the last axis with **no learnable affine** — the affine
/// comes from the per-conditioning `scale`/`shift` inputs instead.
///
/// - [`AdaNormKind::LayerNorm`]: mean-subtract + variance-normalize (DiT / PixArt
///   / SD3 / FLUX adaLN blocks — `LayerNorm(elementwise_affine=False)`).
/// - [`AdaNormKind::RmsNorm`]: root-mean-square normalize only (adaLN variants
///   that pre-norm with RMSNorm).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AdaNormKind {
    #[default]
    LayerNorm,
    RmsNorm,
}

/// Which spectral matrix function [`Op::SpdMatrixFnBatch`] applies to each SPD
/// matrix of a batch — the batched counterparts of `rlx_cpu::spd`'s
/// `logm` / `expm` / `sqrtm` / `invsqrtm` (spectral: `V·diag(f(λ))·Vᵀ`).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpdMatFn {
    /// Matrix logarithm of an SPD matrix.
    Logm,
    /// Matrix exponential of a symmetric matrix.
    Expm,
    /// Matrix square root of an SPD matrix.
    Sqrtm,
    /// Inverse matrix square root of an SPD matrix (`A^{-1/2}`).
    Invsqrtm,
}

/// An operation in the RLX IR graph.
///
/// Operations are categorized for fusion analysis:
/// - Element-wise ops fuse with anything reading their output
/// How [`Op::Pad`] fills the region added around the input.
///
/// Semantics match NumPy `pad` / PyTorch `F.pad`:
/// - `Constant(v)` — fill with the scalar `v`.
/// - `Reflect` — mirror across the edge, **excluding** the edge element
///   (`abcd` → `cb|abcd|cb`). Requires `pad < axis_len`.
/// - `Replicate` — repeat the edge element (`abcd` → `aa|abcd|dd`).
/// - `Circular` — wrap around (`abcd` → `cd|abcd|ab`). Requires `pad ≤ axis_len`.
///
/// Not `Eq`/`Hash` because `Constant` carries an `f32` — mirrors [`Op`], which
/// already holds `f32` attributes (e.g. `eps`) and is only `PartialEq`.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    /// Fill padded positions with a constant scalar.
    Constant(f32),
    /// Mirror across the edge, excluding the edge element.
    Reflect,
    /// Repeat the edge element.
    Replicate,
    /// Wrap around (periodic).
    Circular,
}

/// - Matmul/Conv are BLAS-dispatched and form fusion boundaries
/// - Reductions are fusion roots (drive the loop iteration)
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // ── Graph inputs ────────────────────────────────────────────
    /// Model input with a name (shape on the Node).
    Input {
        name: String,
    },

    /// Model parameter (weight/bias) with a name.
    Param {
        name: String,
    },

    /// Constant tensor embedded in the graph.
    Constant {
        data: Vec<u8>,
    },

    // ── Element-wise unary ──────────────────────────────────────
    /// Unary activation: one input, same shape output.
    Activation(Activation),

    /// Cast to a different dtype.
    Cast {
        to: DType,
    },

    /// Stop-gradient (a.k.a. `detach` / `lax.stop_gradient`). Forward is
    /// identity; the reverse-mode autodiff rule returns **no** gradient
    /// contribution for the input. Single input, same shape & dtype
    /// output. Used to build a Gradient-Reverse-Layer with identity
    /// forward semantics in user code (see maet-rs `dat_loss`).
    StopGradient,

    /// INT8 quantization. Input f32; output i8 same shape.
    ///   `q[i] = saturate_i8(round(x[i] / scale[c]) + zero_point[c])`
    /// where `c` selects the per-channel scale/zp when `axis = Some(d)`
    /// (`c = idx[d]`), or always uses index 0 when `axis = None`
    /// (per-tensor). The `scales` / `zero_points` payload length must
    /// match `1` for per-tensor and `input.dim(d)` for per-channel.
    /// Static — typically produced at calibration time and baked
    /// into the loaded model. Use `Op::Dequantize` for the inverse.
    Quantize {
        axis: Option<usize>,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    },

    /// INT8 dequantization (inverse of `Op::Quantize`). Input i8;
    /// output f32 same shape.
    ///   `x[i] = (q[i] - zero_point[c]) · scale[c]`
    /// where `c` is selected by `axis` exactly as in `Op::Quantize`.
    Dequantize {
        axis: Option<usize>,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    },

    /// "Fake-quantize" op for **quantization-aware training** (QAT).
    /// Input f32; output f32 same shape. Forward computes a per-axis
    /// (or per-tensor when `axis = None`) max-abs scale on the fly:
    ///   `s[c] = max(|x[..., c, ...]|) / q_max(bits)`
    /// then quantizes-then-dequantizes:
    ///   `out[i] = clamp(round(x[i] / s[c]), -q_max, q_max) * s[c]`
    /// where `q_max` is `127` for `bits=8`, `7` for `bits=4`, `1` for
    /// `bits=2` (ternary). Symmetric only — zero-point is always 0.
    ///
    /// The point of this op is to make the SGD optimizer "see" the
    /// deployment-time rounding during training. Backward is the
    /// **straight-through estimator** (STE): the gradient passes
    /// through (variant chosen by `ste`), ignoring the discontinuity
    /// at the round. Without STE the rounding would have zero
    /// gradient almost everywhere and learning would stop.
    ///
    /// Inserted by the trainer on conv / FC weight tensors when
    /// `--qat` is on; the existing `Op::Quantize` / packing path at
    /// the end of training still handles the deployment-side
    /// conversion to `i8`/`i4`/`i2` codes.
    FakeQuantize {
        bits: u8,
        axis: Option<usize>,
        ste: SteKind,
        scale_mode: ScaleMode,
    },

    /// Learned Step Size Quantization (LSQ; Esser et al. 2020,
    /// `arXiv:1902.08153`). Like `FakeQuantize` but the per-channel
    /// `scale` is a *learned parameter*, passed as the second input.
    /// Forward is identical to `FakeQuantize` with a fixed scale:
    ///   `out[i] = clamp(round(x[i]/s[c]), -q_max, q_max) * s[c]`
    /// Backward computes both `dx` (STE) and `dscale[c]` via the
    /// closed-form gradient:
    ///   `dscale[c] = sum_i ψ(x[i]/s[c]) · upstream[i]`
    /// where `ψ(z) = -z + round(z)` if `|z| ≤ q_max` else
    /// `sign(z) · q_max`. Routinely beats per-batch and EMA at
    /// tight bit widths (i2 / i3).
    ///
    /// Inputs: `[x, scale]`. `scale` is `[chan_dim]` f32 (matches
    /// `axis`); for `axis = None` it's `[1]`.
    FakeQuantizeLSQ {
        bits: u8,
        axis: Option<usize>,
    },

    /// Backward pass for `Op::FakeQuantizeLSQ`. Computes BOTH the
    /// gradient w.r.t. `x` (STE) and the gradient w.r.t. `scale`
    /// (closed-form). Output shape matches `x`; the `scale` gradient
    /// is reduced separately by `LsqScaleGradient`.
    /// Inputs: `[x, scale, dy]`. Output: `dx`, same shape as `x`.
    FakeQuantizeLSQBackwardX {
        bits: u8,
        axis: Option<usize>,
    },

    /// Companion to `FakeQuantizeLSQBackwardX`: computes the
    /// `[chan_dim]` per-channel scale gradient. Inputs `[x, scale, dy]`.
    /// Output shape matches `scale`.
    FakeQuantizeLSQBackwardScale {
        bits: u8,
        axis: Option<usize>,
    },

    // ── Element-wise binary ─────────────────────────────────────
    /// Binary op with broadcasting: two inputs, output shape is broadcast result.
    Binary(BinaryOp),

    // ── Comparison ──────────────────────────────────────────────
    /// Element-wise comparison: two inputs, Bool output.
    Compare(CmpOp),

    /// Select elements: cond (Bool), on_true, on_false → output.
    Where,

    /// Fused multiply-add with a SINGLE rounding: `a*b + c`. Distinct from a
    /// `mul` then `add` (two roundings) — the single rounding is what makes
    /// error-free transforms (TwoProduct) and compensated/double-word
    /// arithmetic possible. Inputs: a, b, c.
    ///
    /// The native (single-rounding) kernel is **elementwise on equal-shaped
    /// operands** — broadcast a scalar to the common shape first. Backends
    /// without a native FMA fall back to `mul`+`add` via `LowerFma` (two
    /// roundings, but broadcasts), so this only constrains the precise path.
    Fma,

    /// Fused element-wise region (PLAN L2). Holds an N-step chain of
    /// element-wise operations. Inputs are referenced by index 0..num_inputs;
    /// each step's result can be referenced by later steps via
    /// `ChainOperand::Step(idx)`. The output is the last step's result.
    /// Emitted by `MarkElementwiseRegions` in `rlx-opt` from chains of
    /// Activation/Cast/Binary/Compare/Where ops with single-consumer
    /// intermediates and broadcast-compatible shapes. Backends that
    /// don't have a region kernel can decompose back to the original
    /// chain via `unfuse::unfuse_elementwise_regions`.
    ///
    /// `scalar_input_mask` is a per-input bitfield (bit `i` set ⇒
    /// input `i` is a scalar broadcast — has shape `[1]`). Kept as a
    /// fast-path indicator that lets kernels skip the modulo entirely
    /// when they detect a scalar.
    ///
    /// `input_modulus[i]` is the per-input element count, used by
    /// kernels to compute `arena[input_offs[i] + (gid % input_modulus[i])]`
    /// — the trailing-shape broadcast pattern. `0` means "no broadcast"
    /// (input matches the output element count; kernel reads `gid`
    /// directly). `1` means scalar; any other value means the input
    /// has fewer elements than the output and they tile by modulo.
    /// The encoder only allows broadcasts where `out_elems % in_elems
    /// == 0` so the modulo divides cleanly. Lets chains include bias /
    /// scale / eps / mask factors that previously broke the chain at
    /// a Binary op with mismatched shapes.
    ElementwiseRegion {
        chain: Vec<ChainStep>,
        num_inputs: u32,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
        /// FKL-style closed fusion: apply before the element-wise chain.
        prologue: RegionPrologue,
        /// External input index that supplies the prologue transform source (default 0).
        prologue_input: u32,
    },

    /// Fused transform chain (resize, future crop/color). Decompose via
    /// [`rlx_fusion::DecomposeFusionRegions`] when no native kernel exists.
    TransformRegion {
        steps: Vec<TransformStep>,
        num_inputs: u32,
    },

    /// Identical [`Op::ElementwiseRegion`] chain over `num_batch_inputs` tensors
    /// (horizontal / z-plane fusion). Inputs are separate batch slices.
    BatchElementwiseRegion {
        chain: Vec<ChainStep>,
        num_batch_inputs: u32,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
        prologue: RegionPrologue,
        prologue_input: u32,
    },

    // ── Linear algebra ──────────────────────────────────────────
    /// Matrix multiply. Inputs: [.., M, K] × [.., K, N] → [.., M, N].
    /// Batch dimensions are broadcast.
    MatMul,

    /// Matrix multiply with explicit dimension specification.
    /// Like XLA's DotGeneral — handles arbitrary batch/contracting dims.
    DotGeneral {
        lhs_contracting: Vec<usize>,
        rhs_contracting: Vec<usize>,
        lhs_batch: Vec<usize>,
        rhs_batch: Vec<usize>,
    },

    /// Batched dense linear solve. Inputs: `A [B, N, N]`,
    /// `b [B, N]` or `b [B, N, K]`. Output: same shape as `b`.
    ///
    /// Per-batch independent solve — each `A[i]` and `b[i]` are
    /// solved as a separate `Op::DenseSolve`. Emitted by vmap of
    /// `Op::DenseSolve`. The CPU lowering loops over the batch
    /// dimension calling `dgesv` per slice (LAPACK doesn't expose a
    /// batched solve on Accelerate; cuSOLVER does on NVIDIA).
    BatchedDenseSolve,

    /// Dense linear solve `x = A⁻¹ · b` via LU factorization.
    /// Inputs: `A [N, N]`, `b [N]` (or `b [N, K]` for multi-RHS).
    /// Output: same shape as `b`.
    ///
    /// VJP via the implicit-function theorem:
    ///   `dx = solve(Aᵀ, upstream)`
    ///   `dA = -outer(dx, x)`   (x is the forward output)
    ///   `db = dx`
    /// The rule is dtype-agnostic; lowering is per-backend (Accelerate
    /// `dgesv` / `sgesv`, cuSOLVER, etc.).
    DenseSolve,

    /// Cholesky factorization of a symmetric positive-definite matrix.
    /// Input `A` is `[n, n]` (SPD, row-major); output `L` is `[n, n]`
    /// lower-triangular with the strict upper triangle zeroed, so `A = L·Lᵀ`.
    /// One input. CPU: LAPACK `potrf` (promoted to f64). Backward (added with
    /// `TriangularSolve`) is `Ā = Φ(L⁻ᵀ · Φ(Lᵀ L̄) · L⁻¹)` symmetrized.
    Cholesky,

    /// Solve `op(A)·X = B` for triangular `A` (`[n, n]`), with `B` `[n]` or
    /// `[n, nrhs]`. `op(A)` is `Aᵀ` when `transpose` else `A`; `A` is read as
    /// lower- or upper-triangular per `lower` (non-unit diagonal). Two inputs
    /// `[A, B]`; output `X` matches `B`. CPU: BLAS `trsm`. Differentiable in
    /// both (grads are themselves triangular solves).
    TriangularSolve {
        lower: bool,
        transpose: bool,
    },

    /// Determinant of a square matrix `A` `[n, n]` → scalar `[]`. One input.
    /// CPU: LAPACK LU (`getrf`). Backward: `Ā = (det(A)·ū)·A⁻ᵀ`.
    Det,
    /// `log|det(A)|` of a square matrix `A` `[n, n]` → scalar `[]`. One input.
    /// Numerically stable (sum of `log|Uᵢᵢ|`). Backward: `Ā = ū·A⁻ᵀ`.
    LogDet,

    /// Sort values along `axis` (ascending, or `descending`). One input; output
    /// same shape/dtype. Backward scatters `ū` back via the argsort permutation.
    Sort {
        axis: usize,
        descending: bool,
    },
    /// Indices that would sort along `axis`, as f32 (like `ArgMax`). One input;
    /// output same shape. Non-differentiable.
    ArgSort {
        axis: usize,
        descending: bool,
    },

    /// One factor of a thin SVD `A = U·diag(S)·Vᵀ` (`A` `[m, n]`,
    /// `k = min(m, n)`), selected by `part`. One input. CPU: LAPACK `gesdd`.
    /// Only `SvdPart::S` is differentiable (`Ā = U·diag(ū)·Vᵀ`); `U`/`Vt` are
    /// forward-only.
    Svd {
        part: SvdPart,
    },

    /// One factor of a thin QR `A = Q·R` (`A` `[m, n]`, `k = min(m, n)`),
    /// selected by `part`: `Q` `[m, k]` or `R` `[k, n]`. One input. CPU: LAPACK
    /// `geqrf`+`orgqr`. Differentiable (the QR pullback via `copyltu` + `R⁻ᵀ`).
    Qr {
        part: QrPart,
    },

    // ── Normalization ───────────────────────────────────────────
    /// Layer normalization: input, gamma, beta → normalized output.
    /// `axis` is the feature dimension (usually -1).
    LayerNorm {
        axis: i32,
        eps: f32,
    },

    /// Group normalization on NCHW tensors: `input`, `gamma`, `beta` → same shape.
    /// Normalizes over `(C/num_groups) × H × W` per group.
    GroupNorm {
        num_groups: usize,
        eps: f32,
    },

    /// LayerNorm2d on NCHW: normalize across the channel axis at each spatial
    /// position (candle / SAM `LayerNorm2d` semantics — not PyTorch's H×W norm).
    LayerNorm2d {
        eps: f32,
    },

    /// Nearest-neighbor 2× upsample on NCHW (doubles spatial dims 2 and 3).
    ResizeNearest2x,

    /// Nearest-neighbor resample on NCDHW. `size` is `[D_out, H_out, W_out]`.
    /// Source mapping: `src = min(floor(dst * in / out), in - 1)`.
    Interpolate3d {
        size: Vec<usize>,
    },

    /// RMS normalization: input, gamma → normalized output.
    RmsNorm {
        axis: i32,
        eps: f32,
    },

    /// BatchNorm inference with frozen running statistics.
    /// Inputs: `x`, `gamma`, `beta`, `running_mean`, `running_var`.
    /// Feature dimension is the last axis of `x`; stats are 1-D `[C]`.
    BatchNormInference {
        eps: f32,
    },

    // ── Attention ───────────────────────────────────────────────
    /// Scaled dot-product attention: Q, K, V, \[mask\] → output.
    /// The compiler can lower this to fused SDPA or flash attention.
    /// `mask_kind` controls how masking is applied — `Custom` reads from
    /// the 4th input tensor; `None` / `Causal` / `SlidingWindow` skip the
    /// mask load and apply the mask directly in the inner loop. See
    /// `MaskKind` for the rationale.
    ///
    /// `score_scale`: when `Some(s)`, dot-product scores are multiplied by
    /// `s` instead of the default `1/sqrt(head_dim)` (Gemma uses `head_dim^-0.5`
    /// explicitly in config). `attn_logit_softcap`: when `Some(c)`, applies
    /// `c * tanh(s/c)` to scores before softmax (Gemma 2).
    ///
    /// ## Operand layout
    ///
    /// Q/K/V (and the output) carry the heads explicitly and are accepted in two
    /// rank-4 layouts — both end in `head_dim`, so backends disambiguate by which
    /// axis equals `num_heads`:
    ///
    /// - **`[B, S, H, D]`** (heads at axis 2) — the dominant convention, produced
    ///   by Llama/Moshi-style `reshape([B, S, H, D])` after the QKV projection.
    ///   The CPU/Metal/MLX/wgpu kernels attend over axis 1 (`S`); CoreML
    ///   transposes to `[B, H, S, D]` first (see `rlx-coreml`).
    /// - **`[B, H, S, D]`** (heads at axis 1) — already head-canonical.
    ///
    /// Rank-3 `[B, S, D]` is single-head and head-canonical. `K`/`V` may have a
    /// different sequence length than `Q` (`S_k >= S_q`) for KV-cache decode; the
    /// causal mask places the `S_q` queries at the **end** of the `S_k` keys
    /// (query `qi` is at absolute position `S_k - S_q + qi`).
    Attention {
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    },

    /// Rotary position embedding applied to one tensor: x, cos, sin → x_rotated.
    /// Apply separately to Q and K. `head_dim` is the per-head width; `n_rot`
    /// is how many leading dims get NeoX RoPE (pair offset `n_rot/2`). When
    /// `n_rot < head_dim`, trailing dims are copied unchanged (Qwen3.5 MRoPE).
    Rope {
        head_dim: usize,
        n_rot: usize,
        /// Rotation pairing convention (default [`RopeStyle::NeoX`]).
        style: RopeStyle,
    },

    /// SAM2 axial 2-D RoPE on `[batch, seq, num_heads * head_dim]`.
    AxialRope2d {
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        num_heads: usize,
        theta: f32,
        repeat_factor: usize,
    },

    // ── Shape manipulation ──────────────────────────────────────
    Reshape {
        new_shape: Vec<i64>,
    },
    Transpose {
        perm: Vec<usize>,
    },
    /// Select a contiguous slice along an axis.
    Narrow {
        axis: usize,
        start: usize,
        len: usize,
    },
    /// Strided slice along an axis: output element `j` reads input index
    /// `start + j*step` for `j` in `0..len`. `step` may be negative (reverse);
    /// `step != 0`. All accessed indices must lie in `[0, axis_len)`. The
    /// contiguous `step == 1` case is [`Op::Narrow`]; this covers `x[::2]`,
    /// `x[::-1]`, dilation, and general strided indexing. No backend has a
    /// native kernel except Metal/CUDA; others legalize to `narrow`/`reverse`/
    /// `gather` via `rlx_fusion::LowerSlice`.
    Slice {
        axis: usize,
        start: usize,
        len: usize,
        step: i64,
    },
    /// Concatenate along an axis.
    Concat {
        axis: usize,
    },
    /// Expand (broadcast) to a target shape.
    Expand {
        target_shape: Vec<i64>,
    },
    /// Gather elements by index along an axis (embedding lookup).
    Gather {
        axis: usize,
    },
    /// Reverse (flip) the element order along each listed axis. Output shape
    /// equals the input shape; element `[…, i, …]` on a reversed axis of size
    /// `d` reads input `[…, d-1-i, …]`. Batch-general — only the listed axes
    /// flip, all others pass through unchanged (the right primitive for
    /// reversing a `[batch, seq, …]` sequence without a batch=1 assumption).
    Reverse {
        axes: Vec<usize>,
    },
    /// Pad each axis by `pads[i] = [before, after]` (aligned to the input rank)
    /// per [`PadMode`]. Output axis `i` grows by `before + after`. Single tensor
    /// input; the constant fill value lives in the mode, not an input.
    ///
    /// No backend has a native kernel except Metal/CUDA; every other backend
    /// legalizes it to `narrow`/`reverse`/`concat`/`gather`/`full` via
    /// `rlx_fusion::LowerPad`, so the decomposition is the semantic oracle.
    Pad {
        pads: Vec<[usize; 2]>,
        mode: PadMode,
    },
    /// Elementwise `clamp(x, min, max)`. Decomposes to `max`/`min` (native on
    /// every backend); no dedicated kernel needed. Backward: pass-through where
    /// `min ≤ x ≤ max`, else 0.
    Clamp {
        min: f32,
        max: f32,
    },
    /// Tile (repeat) the input `reps[i]` times along axis `i` (`reps` aligned to
    /// the input rank). Decomposes to `concat`. Backward: reduce-sum the copies.
    Tile {
        reps: Vec<usize>,
    },
    /// Keep the upper (`upper=true`) or lower triangle relative to `diagonal`;
    /// zero the rest. Operates on the last two axes. Decomposes to a multiply by
    /// a constant triangular mask. Backward: same mask.
    Trilu {
        upper: bool,
        diagonal: i64,
    },

    // ── Reduction ───────────────────────────────────────────────
    /// Reduce along specified axes.
    Reduce {
        op: ReduceOp,
        axes: Vec<usize>,
        keep_dim: bool,
    },

    /// Selective scan (plan #15) — Mamba-style state-space model
    /// step. The recurrence:
    ///   `h[t] = exp(Δ[t] * A) * h[t-1] + Δ[t] * B[t] * x[t]`
    ///   `y[t] = C[t] * h[t]`
    /// where state `h` has dimension `state_size` and the input has
    /// `(batch, seq, hidden)`.
    ///
    /// Inputs (in order):
    ///   `x [b, s, h]`      f32 input
    ///   `delta [b, s, h]`  f32 step size (per-position, per-channel)
    ///   `a [h, n]`         f32 transition matrix (one per channel)
    ///   `b [b, s, n]`      f32 input projection
    ///   `c [b, s, n]`      f32 output projection
    /// Output: `[b, s, h]` f32. State `h` is implicit; the kernel
    /// scans through the seq dimension carrying it.
    ///
    /// `state_size` = `n` is exposed for the cost model.
    SelectiveScan {
        state_size: usize,
    },

    /// Gated DeltaNet linear-attention recurrence — the per-layer
    /// kernel used by Qwen3.5/3.6 trunk "linear attention" blocks
    /// (and Qwen3-Next, Kimi-Linear). Mirrors
    /// `llama.cpp / src/models/delta-net-base.cpp` autoregressive
    /// path; chunked + fused variants ride the same op identity.
    ///
    /// **Math (per token `t`, head `h`, state size `n`):**
    /// state matrix `S[h, i, j]` is implicit (reset per batch).
    /// ```text
    ///   S[h]     *= exp(g[t,h])                     # scalar gate
    ///   sk[h,j]   = Σ_i S[h,i,j] * k[t,h,i]
    ///   d[h,j]    = (v[t,h,j] - sk[h,j]) * b[t,h]   # b = beta
    ///   S[h,i,j] += k[t,h,i] * d[h,j]               # outer-prod
    ///   o[t,h,j]  = Σ_i S[h,i,j] * (q[t,h,i] / √n)
    /// ```
    ///
    /// Inputs:
    ///   `q     [b, s, h_v, n]`  f32 queries (L2-normed by caller)
    ///   `k     [b, s, h_v, n]`  f32 keys    (L2-normed by caller;
    ///                            GQA-repeated to match `h_v`)
    ///   `v     [b, s, h_v, n]`  f32 values
    ///   `g     [b, s, h_v]`     f32 log-gate (exp'd inside kernel)
    ///   `beta  [b, s, h_v]`     f32 delta-rule mixing factor
    ///
    /// Output: `[b, s, h_v, n]` f32.
    ///
    /// When `carry_state` is true, a sixth input `state [b, h_v, n, n]`
    /// provides the initial SSM matrix per head; the kernel updates it
    /// in place across the sequence and leaves the final state in the
    /// same buffer (same layout as the internal scan state:
    /// `state[h, i, j]` row-major over `(n, n)` per head).
    ///
    /// When `gate_per_channel` is true (Kimi-K3 "KDA"), the `g` input is
    /// `[b, s, h_v, n]` (one log-gate per key channel) instead of `[b, s, h_v]`,
    /// and the state decay is applied per key-row: `S[i, j] *= exp(g[i])`
    /// (vs the scalar `S *= exp(g)` of the per-head gate). Everything else is
    /// identical; Qwen3-Next / Qwen3.5 use the per-head gate (`false`).
    GatedDeltaNet {
        state_size: usize,
        carry_state: bool,
        gate_per_channel: bool,
    },

    /// Multi-layer (optionally bidirectional) LSTM over a
    /// `[batch, seq, input]` sequence. Gate order i, f, g, o (PyTorch);
    /// recurrence per step:
    /// ```text
    ///   z      = x_t · w_ihᵀ + h_{t-1} · w_hhᵀ + bias
    ///   i,f,o  = σ(z_i), σ(z_f), σ(z_o);   g = tanh(z_g)
    ///   c_t    = f · c_{t-1} + i · g;      h_t = o · tanh(c_t)
    /// ```
    /// `D = 2 if bidirectional else 1`. Inputs `[x, w_ih, w_hh, bias]`
    /// (`+ [h0, c0]` when `carry`):
    ///   * `x`:    `[batch, seq, input]`
    ///   * `w_ih`: packed, all `(layer, direction)` blocks concatenated in
    ///     `layer`-major then `direction` order. Block `(l,d)` is
    ///     `[4*hidden, in_l]` with `in_l = input` for `l=0` else `D*hidden`.
    ///   * `w_hh`: packed `L*D` blocks of `[4*hidden, hidden]`.
    ///   * `bias`: packed `L*D` blocks of `[4*hidden]` (combined `b_ih+b_hh`).
    ///   * `h0`, `c0` (carry only): `[L*D, batch, hidden]`; the final
    ///     `hn`/`cn` are written back **in place** (decode threading).
    ///
    /// Output `y`: `[batch, seq, D*hidden]` — last layer's hidden states
    /// (forward ‖ reverse concatenated on the feature axis). With
    /// `num_layers = 1, bidirectional = false, carry = false` this is the
    /// plain single-layer LSTM and the weight shapes reduce to
    /// `[4*hidden, input]`, `[4*hidden, hidden]`, `[4*hidden]`.
    Lstm {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        carry: bool,
    },

    /// Gated Recurrent Unit (PyTorch). `D = 2 if bidirectional else 1`;
    /// gate order r, z, n. Recurrence:
    /// ```text
    ///   r = σ(x·W_irᵀ + b_ir + h·W_hrᵀ + b_hr)
    ///   z = σ(x·W_izᵀ + b_iz + h·W_hzᵀ + b_hz)
    ///   n = tanh(x·W_inᵀ + b_in + r ⊙ (h·W_hnᵀ + b_hn))
    ///   h' = (1 - z) ⊙ n + z ⊙ h
    /// ```
    /// The new-gate reset is applied to the hidden term *after* its bias,
    /// so `b_ih`/`b_hh` cannot be merged. Inputs `[x, w_ih, w_hh, b_ih, b_hh]`
    /// (`+ [h0]` when carry); packing matches [`Op::Lstm`] with `3*hidden`
    /// gate rows. `h0` `[L*D, batch, hidden]`. Output `[batch, seq, D*hidden]`.
    Gru {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        carry: bool,
    },

    /// Elman RNN (PyTorch): `h' = act(x·w_ihᵀ + h·w_hhᵀ + bias)` with
    /// `act = ReLU` when `relu` else `tanh`. Packed `w_ih` `[hidden, in_l]`,
    /// `w_hh` `[hidden, hidden]`, merged `bias` `[hidden]` (per layer ×
    /// direction). Inputs `[x, w_ih, w_hh, bias]` (`+ [h0]` when carry).
    /// Output `[batch, seq, D*hidden]`.
    Rnn {
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        carry: bool,
        relu: bool,
    },

    /// Mamba-2 / SSD (structured state-space duality) scan — the
    /// scalar-decay SSM at the core of Mamba-2, sibling of
    /// [`Op::SelectiveScan`] / [`Op::GatedDeltaNet`]. Inputs
    /// `[x, dt, a, b, c]` (all `f32`):
    ///   * `x`:  `[batch, seq, heads, head_dim]`
    ///   * `dt`: `[batch, seq, heads]` — discretization step (already ≥ 0,
    ///     e.g. softplus'd by the caller)
    ///   * `a`:  `[heads]` — per-head decay rate (`dA = exp(dt·a)`)
    ///   * `b`:  `[batch, seq, heads, state_size]`
    ///   * `c`:  `[batch, seq, heads, state_size]`
    ///
    /// State `S [batch, heads, head_dim, state_size]` is zero-initialized
    /// per sequence. Recurrence per timestep `t`:
    /// ```text
    ///   dA_t = exp(dt_t · a)
    ///   S_t  = dA_t · S_{t-1} + (dt_t · x_t) ⊗ b_t
    ///   y_t  = Σ_n S_t[:, n] · c_t[n]
    /// ```
    /// Output `y`: `[batch, seq, heads, head_dim]` (same shape as `x`).
    Mamba2 {
        head_dim: usize,
        state_size: usize,
    },

    /// Fused dequant + matmul (plan #5). The biggest LLM-bandwidth
    /// win on Apple Silicon: dequantizes weights inside the matmul
    /// inner loop, never materializing f32 weights.
    ///
    /// **BREAKING CHANGE in 0.2.0:** `num_inputs()` is now
    /// scheme-dependent — **4** for legacy Int8 schemes, **2** for
    /// the new GGUF K-quant schemes (their scales/mins live inside
    /// the packed bytes, so no side-channel `scale` / `zp` tensors
    /// are fed in). Callers that assumed a fixed 4-input contract
    /// must inspect `scheme.is_gguf()` before reading inputs.
    ///
    /// Inputs (Int8 schemes — `scheme.is_gguf() == false`):
    ///   `x [m, k]`             f32 activations
    ///   `w_q [k, n]` packed    quantized weight bytes (i8 per
    ///                          element for Int8 schemes; 4-bit
    ///                          packed two-per-byte for Int4)
    ///   `scale [k/block, n]`   per-block f32 dequant scale
    ///   `zp    [k/block, n]`   per-block f32 zero-point
    ///                          (zero-tensor if symmetric)
    ///
    /// Inputs (`Nvfp4Block` — fixed group size 16 along K):
    ///   `x [m, k]`             f32 activations
    ///   `w_q [k,n/2]` packed   FP4 E2M1 codes (unsigned nibble 0..15)
    ///   `scale [k/16, n]` u8   FP8 E4M3 block scales (one byte / group)
    ///   `global_scale [1]` f32 per-tensor scale (pass `[1.0]` if unused)
    ///
    /// Inputs (GGUF schemes — `scheme.is_gguf() == true`):
    ///   `x [m, k]`             f32 activations
    ///   `packed_w [bytes]`     raw GGUF super-block bytes; the
    ///                          dequantizer reads the per-sub-block
    ///                          scales / mins / quants directly out
    ///                          of the buffer per the K-quant block
    ///                          layout (no side tensors).
    ///
    /// Output: `[m, n]` f32.
    ///
    /// `block_size` (on the Int8 schemes only) is the number of
    /// consecutive elements that share one (scale, zero_point) pair.
    /// The Op carries enough metadata that the kernel doesn't need
    /// a separate `QuantMap` lookup at run time.
    DequantMatMul {
        scheme: crate::quant::QuantScheme,
    },

    /// Real INT8-arithmetic matrix multiply with i32 accumulation.
    /// Inputs (in order):
    ///   `x      [M, K]`  i8 activations (zero-point = `x_zp`)
    ///   `w      [K, N]`  i8 weights     (zero-point = `w_zp`)
    ///   `bias   [N]`     i32 (in accumulator scale = `x_scale·w_scale`),
    ///                    pass a zeros tensor for "no bias"
    /// Output:  `[M, N]`  i8 (zero-point = `out_zp`)
    ///
    /// Per-element compute:
    ///   `out[m,n] = requantize(bias[n] + Σₖ (x[m,k]-x_zp)·(w[k,n]-w_zp), mult, out_zp)`
    /// where `mult = x_scale · w_scale / out_scale`.
    ///
    /// This is the same kernel shape `rlx-cortexm/src/dense.rs`
    /// uses for on-device int8 inference, lifted into the IR so the
    /// rlx-cpu backend can run a quantized graph directly (instead
    /// of round-tripping through fake-quant Dequantize → MatMul →
    /// Quantize). 2-D only — generalizing to batched comes when a
    /// real workload demands it.
    QMatMul {
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
    },

    /// Real INT8-arithmetic 2-D convolution with i32 accumulation.
    /// Inputs:
    ///   `x      [N, C_in, H, W]`              i8 (zero-point = `x_zp`)
    ///   `w      [C_out, C_in/groups, kH, kW]` i8 (zero-point = `w_zp`)
    ///   `bias   [C_out]`                      i32 in accumulator scale
    /// Output: `[N, C_out, H_out, W_out]` i8 (zero-point = `out_zp`).
    /// Same NCHW geometry contract as `Op::Conv`; same requantize
    /// math as `Op::QMatMul` (per-element `acc·mult` rounded to i8).
    QConv2d {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
    },

    /// **Native low-precision tensor-core GEMM** with f32 accumulation —
    /// both operands are sub-f16 codes fed *directly* into the matmul, the
    /// opposite of `DequantMatMul` (which decodes weights to f32 first and
    /// keeps `x` in f32). This is the path that wins on Hopper / Ada /
    /// Blackwell (cublasLt FP8/FP4) and CDNA3 / CDNA4 (hipBLASLt). On CPU /
    /// Metal it is the decode-and-accumulate *reference* (no fp8 matrix HW).
    ///
    /// Layout is **TN** — both operands are K-last (`out = lhs · rhsᵀ`) so
    /// every block scale runs along the last/contraction axis uniformly, which
    /// is also cuBLASLt / hipBLASLt FP8's preferred layout.
    ///
    /// Inputs (in order):
    ///   `lhs   [m, k]`  U8 — packed [`crate::ScaledFormat`] codes (logical dims)
    ///   `rhs   [n, k]`  U8 — packed codes (K-last, i.e. weight transposed)
    ///   `lhs_s        `  scale tensor, dtype per [`crate::ScaleLayout::scale_dtype`]
    ///   `rhs_s        `  scale tensor
    ///   `bias  [n]    `  f32, present iff `has_bias`
    /// Output: `[m, n]` f32. Reconstruction: `decode(code)·scale(block)`.
    ///
    /// Operand `Shape`s carry **logical** element dims with `dtype = U8`; the
    /// physical byte layout (1 code/byte on the CPU oracle, packed nibbles on
    /// GPU) is a backend-paired detail shared by `ScaledQuantize` and the
    /// matmul kernel, so it never needs a `DType` variant.
    ScaledMatMul {
        lhs_format: crate::quant::ScaledFormat,
        rhs_format: crate::quant::ScaledFormat,
        scale_layout: crate::quant::ScaleLayout,
        has_bias: bool,
    },

    /// Quantize an f32/f16 tensor to packed low-precision codes for
    /// [`Op::ScaledMatMul`]. Inputs: `x` (f32) and `scale` (from
    /// [`Op::ScaledQuantScale`]); output is `DType::U8` codes of the same
    /// logical shape as `x`. `code[i] = encode(format, x[i] / scale(block_of(i)))`.
    ScaledQuantize {
        format: crate::quant::ScaledFormat,
        scale_layout: crate::quant::ScaleLayout,
    },

    /// Compute the (per-tensor or per-block) scale tensor for a tensor about
    /// to be quantized to `format`. Input: `x` (f32). Output dtype follows
    /// [`crate::ScaleLayout::scale_dtype`]. Split from `ScaledQuantize` to keep
    /// the IR single-output (mirrors the `FakeQuantizeLSQBackwardX/Scale` split).
    ScaledQuantScale {
        format: crate::quant::ScaledFormat,
        scale_layout: crate::quant::ScaleLayout,
    },

    /// Reconstruct f32 from packed low-precision codes: `decode(code)·scale`.
    /// The inverse of [`Op::ScaledQuantize`]. Inputs: `codes` (U8), `scale`;
    /// output is f32 with the same logical shape as `codes`. Used by the
    /// straight-through VJP of [`Op::ScaledMatMul`] to rebuild operands in the
    /// backward graph (and as a standalone dequantizer).
    ScaledDequantize {
        format: crate::quant::ScaledFormat,
        scale_layout: crate::quant::ScaleLayout,
    },

    /// Fused LoRA matmul: `out = x·W + scale * x·A·B`.
    /// Inputs (in order): `x [m, k]`, `w [k, n]`, `a [k, r]`, `b [r, n]`.
    /// `r` is the LoRA rank (typically 4-64). `scale` is the
    /// per-adapter `alpha / rank` knob.
    /// Plan #9: lifts LoRA from "three matmuls + an add" into one
    /// kernel that keeps the rank-r intermediate in registers.
    LoraMatMul {
        scale: f32,
    },

    /// Uniform-partitioned overlap-save 1-D convolution (FIR / RIR).
    /// Inputs `[x, ir]`: signal `x [.., L]` and rank-1 impulse response
    /// `ir [M]`; output is the full linear convolution `[.., L + M − 1]`.
    /// `block` sets the partition / FFT size (`2·next_pow2(block)` points).
    /// Decomposes (via `unfuse`) to `rfft → batched complex matmul over the
    /// partition axis → irfft`, so the frequency-domain delay line runs on the
    /// native batched-GEMM kernels (cuBLAS / rocBLAS / MPS). See
    /// [`crate::Graph::partitioned_conv`].
    PartitionedConv {
        block: usize,
    },

    /// Fused sampling kernel: logits → optional top-k filter →
    /// optional top-p truncation → softmax → multinomial sample.
    /// One f32-encoded sampled token id per batch row (output
    /// shape `[batch]`).
    ///
    /// `temperature == 1.0` matches a plain argmax-of-softmax;
    /// lower → sharper, higher → flatter. `top_k == 0` disables.
    /// `top_p == 1.0` disables. `seed` is the Philox seed; pass 0
    /// for "use process-global counter" (still deterministic
    /// given the call order).
    /// Borrowed from MAX's nn/sampling.mojo.
    /// Latency-critical: never materializes the full softmax
    /// distribution on the host.
    Sample {
        top_k: usize,     // 0 = disabled
        top_p: f32,       // 1.0 = disabled
        temperature: f32, // 1.0 = neutral
        seed: u64,        // 0 = use thread-local counter
    },

    /// ONNX `RandomNormalLike` / `RandomNormal`: zero or one shape-template
    /// input (Like uses the template's shape; `RandomNormal` with a `shape`
    /// attribute needs no input). Output shape is fixed on the node.
    /// at compile/execute time. Optional ONNX `seed` attribute (f32) overrides
    /// the mixed seed on the Ort backend.
    RngNormal {
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },

    /// ONNX `RandomUniformLike`.
    RngUniform {
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },

    /// Inclusive cumulative sum along an axis. Same shape in/out.
    /// Underpins ragged-tensor offsets, sampling (top-p prefix sum),
    /// and sequence-position math.
    /// `exclusive=true` shifts the result so output\[0\] = 0 (useful
    /// for offset arrays where the first segment starts at 0).
    Cumsum {
        axis: i32,
        exclusive: bool,
    },

    /// Inclusive cumulative product along an axis. Same shape in/out.
    /// `exclusive=true` shifts so output\[0\] = 1 (the multiplicative identity).
    /// Mirrors [`Op::Cumsum`]; native on CPU/Metal/CUDA, decomposes to a
    /// masked reduce-product elsewhere.
    CumProd {
        axis: i32,
        exclusive: bool,
    },

    /// Inclusive cumulative maximum along an axis. Same shape in/out.
    /// `exclusive=true` shifts so output\[0\] = -inf. Native on CPU/Metal/CUDA,
    /// decomposes to a masked reduce-max elsewhere.
    CumMax {
        axis: i32,
        exclusive: bool,
    },

    /// Index of the maximum along `axis`. Output drops that axis (or keeps it
    /// as size 1 when `keep_dim`). Indices are **f32-encoded** (rlx is f32 at
    /// the I/O boundary, matching [`Op::TopK`]).
    ArgMax {
        axis: usize,
        keep_dim: bool,
    },

    /// Index of the minimum along `axis`; see [`Op::ArgMax`].
    ArgMin {
        axis: usize,
        keep_dim: bool,
    },

    /// Softmax along an axis (reduction + element-wise).
    Softmax {
        axis: i32,
    },

    /// Top-K **indices** along the last axis. Output shape `[..., k]`,
    /// f32-encoded indices (rlx is f32-only at the I/O boundary).
    /// To recover the values, follow with a `Gather` against the
    /// original tensor — works because Gather already supports any axis.
    /// Ties broken by smaller index (matches NumPy / PyTorch
    /// `torch.topk(..., largest=True, sorted=True)`).
    /// Used by MoE gating; also useful for beam search.
    TopK {
        k: usize,
    },

    /// Indexed batched matmul. The MoE GEMM primitive.
    /// Inputs: `[input, weight, expert_idx]`
    ///   input       : [M, K]                — per-token activations
    ///   weight      : [num_experts, K, N]   — stacked expert weights
    ///   expert_idx  : \[M\]                   — f32-encoded expert id per token
    /// Output         : [M, N]                — output\[i\] = input\[i\] @ weight[expert_idx\[i\]]
    /// CPU is a real segmented GEMM: counting-sort tokens by expert, one GEMM per
    /// expert, then unpermute. GPU backends dispatch a dedicated grouped kernel.
    GroupedMatMul,

    /// Fused GGUF K-quant dequant + [`Op::GroupedMatMul`]. Same three
    /// inputs as `GroupedMatMul`, but `weight` is a U8 tensor holding
    /// `num_experts` contiguous packed expert slabs (GGML layout, expert
    /// dimension outermost). Scales live inside the packed bytes.
    DequantGroupedMatMul {
        scheme: crate::quant::QuantScheme,
    },

    /// MLX-affine grouped expert matmul — the [`Op::DequantGroupedMatMul`]
    /// analogue for mlx-community MoE, whose experts carry SEPARATE `scales`
    /// and `biases` tensors (not embedded in the packed bytes). Five inputs:
    ///   input   : [M, K]                       — one row per token
    ///   weight  : [E · out · (K·bits/32)] U8   — stacked affine code words
    ///   scales  : [E, out, n_groups] f32       — per-expert per-group scales
    ///   biases  : [E, out, n_groups] f32       — per-expert per-group biases
    ///   expert_idx : \[M\]                       — f32-encoded expert id per row
    /// Output    : [M, out]                     — output\[i\] = input\[i\] @ dequant(expert_idx\[i\])
    DequantGroupedMatMulMlx {
        scheme: crate::quant::QuantScheme,
    },

    /// Dequant a packed MoE expert stack to F32 `[num_experts, K, N]` in
    /// GroupedMatMul layout. Input: U8 packed bytes; output shape is
    /// declared on the node (`[E, K, N]`).
    DequantMoEWeights {
        scheme: crate::quant::QuantScheme,
    },

    /// Scatter-add into a destination tensor. The "unpermute" half of
    /// MoE routing (also useful for embedding gradient updates).
    /// Inputs: `[updates, indices]`
    ///   updates : [num_updates, trailing]   — values to add
    ///   indices : \[num_updates\]             — f32-encoded destination row
    /// Output    : [out_dim, trailing]       — output[indices\[i\]] += updates\[i\]
    /// `out_dim` is taken from the node's declared output shape.
    /// Initial output is zero; multiple updates to the same row
    /// accumulate (sequentially on CPU; with atomic-add on Metal).
    ScatterAdd,

    /// ONNX ScatterND: copy `data`, then write `updates` at multi-index
    /// locations from `indices`.
    /// Inputs: `[data, indices, updates]`
    ///   data    : rank `r` tensor (output shape matches)
    ///   indices : rank `q`, trailing dim `k` indexes the first `k` axes of data
    ///   updates : shape `indices.shape[:-1] + data.shape[k:]`
    /// `reduction` selects overwrite vs accumulate (see [`ScatterNdReduction`]).
    ScatterNd {
        reduction: ScatterNdReduction,
    },

    /// ONNX ScatterElements: per-element scatter along `axis`.
    /// Inputs: `[data, indices, updates]` — output shape matches `data`.
    /// `indices` and `updates` share rank with `data`; along `axis` they
    /// select which coordinate to write.
    ScatterElements {
        axis: i32,
        reduction: ScatterNdReduction,
    },

    /// ONNX GatherND: gather slices from `data` indexed by the trailing
    /// axis of `indices`. Inputs: `[data, indices]`.
    /// `batch_dims` (default 0) skips leading shared batch axes.
    GatherNd {
        batch_dims: i32,
    },

    /// ONNX GatherElements / `take_along_axis`: output shape = `indices`
    /// shape; each output element is `data` with the `axis` coordinate
    /// replaced by the corresponding index. Inputs: `[data, indices]`.
    GatherElements {
        axis: i32,
    },

    // ── Convolution ─────────────────────────────────────────────
    /// 2D convolution on NCHW tensors. Also exposed as [`OpKind::Conv`] / `conv2d`.
    /// Weight layout: `[C_out, C_in / groups, kH, kW]`.
    Conv {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },

    /// NCHW im2col for conv backward-weight style matmul.
    /// Input `[N, C, H, W]`. Output `[M, C·kH·kW]` with
    /// `M = N · H_out · W_out`. When batch is [`dynamic::sym::BATCH`],
    /// output rows use [`dynamic::sym::ROWS`] (bind `N · H_out · W_out`).
    Im2Col {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },

    /// 2D transposed convolution on NCHW. Weight layout (PyTorch):
    /// `[C_in, C_out / groups, kH, kW]`.
    ConvTranspose2d {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        output_padding: Vec<usize>,
        groups: usize,
    },

    /// 3D convolution on NCDHW tensors (forward / cross-correlation).
    /// Weight layout: `[C_out, C_in / groups, kD, kH, kW]`. Kernel size is
    /// derived from the weight shape. Mirrors [`Op::Conv`] with a depth axis.
    Conv3d {
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    },

    /// 3D transposed convolution on NCDHW. Weight layout (PyTorch):
    /// `[C_in, C_out / groups, kD, kH, kW]`. Mirrors [`Op::ConvTranspose2d`].
    ConvTranspose3d {
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        output_padding: [usize; 3],
        groups: usize,
    },

    // ── Pooling ─────────────────────────────────────────────────
    Pool {
        kind: ReduceOp,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },

    // ── Backward / training ops ─────────────────────────────────
    //
    // Closed-form gradient nodes emitted by `rlx-opt::autodiff`.
    // Pairing a forward op with a dedicated backward op (instead of
    // composing it from primitives) keeps the gradient kernel simple
    // and lets the backend recompute argmax / masks / softmax inline.
    /// ReLU backward: `dx = dy where x > 0 else 0`.
    /// Inputs: `[x, dy]` — both same shape; output matches.
    ReluBackward,

    /// Element-wise complex squared-magnitude: `|z|² = z.re² + z.im²`.
    /// Input: 1 tensor with `DType::C64`. Output: same shape but
    /// `DType::F32`. The natural real-valued loss surface for
    /// Wirtinger reverse-mode AD on complex graphs — pair with
    /// [`Op::ComplexNormSqBackward`].
    ComplexNormSq,

    /// Element-wise complex conjugate: `z̄ = z.re - i·z.im` per element.
    /// Input: 1 tensor with `DType::C64`. Output: same shape, same dtype.
    /// Used by Wirtinger VJP rules on `Op::Binary` over C64 (the rule
    /// for `y = a·b` is `dL/dā = upstream · conj(b)`, etc.).
    Conjugate,

    /// Backward for [`Op::ComplexNormSq`] under Wirtinger calculus.
    /// `f(z) = |z|² = z·z̄`, so `∂f/∂z̄ = z`. Given upstream real
    /// cotangent `g` (same shape as the forward output), the C64
    /// gradient with respect to `z` is `g · z` element-wise, returned
    /// in C64 storage `[re_g·re_z, re_g·im_z]` per element.
    ///
    /// Inputs: `[z (C64), g (F32)]` — both same logical shape; output
    /// matches `z` (C64).
    ComplexNormSqBackward,

    /// LayerNorm backward w.r.t. the input. Computes
    ///   `d_x[..., d] = inv_std · (dy·γ - mean(dy·γ) - x̂·mean(dy·γ·x̂))`
    /// over the feature axis, where `x̂ = (x - mean)/std` is recomputed
    /// inline from `x`. Inputs: `[x, gamma, dy]`; output shape = `x.shape`.
    /// Currently lowers axis=-1 only (matches the forward thunk).
    LayerNormBackwardInput {
        axis: i32,
        eps: f32,
    },

    /// LayerNorm backward w.r.t. gamma. Computes
    ///   `d_gamma[d] = Σ_{batch} dy[..., d] · x̂[..., d]`
    /// — sums the per-element product of upstream and the (recomputed)
    /// normalized input over the leading axes. Inputs: `[x, dy]`;
    /// output shape = `gamma.shape` (= 1-D feature axis).
    LayerNormBackwardGamma {
        axis: i32,
        eps: f32,
    },

    /// RMSNorm backward w.r.t. input. Inputs `[x, gamma, beta, dy]`; output = `x.shape`.
    RmsNormBackwardInput {
        axis: i32,
        eps: f32,
    },

    /// RMSNorm backward w.r.t. gamma. Inputs `[x, gamma, beta, dy]`; output = `gamma.shape`.
    RmsNormBackwardGamma {
        axis: i32,
        eps: f32,
    },

    /// RMSNorm backward w.r.t. beta. Inputs `[x, gamma, beta, dy]`; output = `beta.shape`.
    RmsNormBackwardBeta {
        axis: i32,
        eps: f32,
    },

    /// RoPE backward w.r.t. `x`. Inputs `[dy, cos, sin]`; output = `dy.shape`.
    RopeBackward {
        head_dim: usize,
        n_rot: usize,
    },

    /// GroupNorm (NCHW) backward w.r.t. input. Inputs `[x, gamma, beta, dy]`.
    GroupNormBackwardInput {
        num_groups: usize,
        eps: f32,
    },

    /// GroupNorm backward w.r.t. gamma. Inputs `[x, dy]`; output = `gamma.shape`.
    GroupNormBackwardGamma {
        num_groups: usize,
        eps: f32,
    },

    /// GroupNorm backward w.r.t. beta. Inputs `[x, dy]`; output = `beta.shape`.
    GroupNormBackwardBeta {
        num_groups: usize,
        eps: f32,
    },

    /// BatchNorm inference backward w.r.t. `x`. Inputs `[x, gamma, mean, var, dy]`.
    BatchNormInferenceBackwardInput {
        eps: f32,
    },

    /// BatchNorm inference backward w.r.t. `gamma`. Inputs `[x, mean, var, dy]`.
    BatchNormInferenceBackwardGamma {
        eps: f32,
    },

    /// BatchNorm inference backward w.r.t. `beta`. Inputs `[dy]`; output = `beta.shape`.
    BatchNormInferenceBackwardBeta,

    /// Cumsum backward along `axis`. Inputs `[dy]`; output matches forward input shape.
    CumsumBackward {
        axis: i32,
        exclusive: bool,
    },

    /// Gather backward (scatter-add into table). Inputs `[dy, indices]`; output = table shape.
    /// `axis` matches forward [`Op::Gather`].
    GatherBackward {
        axis: i32,
    },

    /// Generic element-wise activation backward. `kind` selects the
    /// closed-form derivative `d/dx act(x)`. Inputs: `[x, dy]`; output
    /// shape matches `x`. The kernel computes `d/dx · dy` per element.
    ///
    /// Closed forms (all element-wise):
    /// * `Gelu`     — exact derivative of erf-based GELU.
    /// * `GeluApprox` — derivative of the tanh approximation
    ///   `0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`.
    /// * `Silu`     — `σ(x)·(1 + x·(1 - σ(x)))`.
    /// * `Sigmoid`  — `σ(x)·(1 - σ(x))`.
    /// * `Tanh`     — `1 - tanh(x)²`.
    /// * `Exp`      — `exp(x)`.
    /// * `Log`      — `1 / x`.
    /// * `Sqrt`     — `0.5 / sqrt(x)`.
    /// * `Rsqrt`    — `-0.5 · x^(-3/2)`.
    /// * `Neg`      — `-1`.
    /// * `Abs`      — `sign(x)` (zero at x=0).
    /// * `Sin`      — `cos(x)`.
    /// * `Cos`      — `-sin(x)`.
    /// * `Tan`      — `1 + tan²(x)`.
    /// * `Atan`     — `1 / (1 + x²)`.
    /// * `Relu`     — kept here for completeness; the dedicated
    ///   `ReluBackward` op is preferred for relu and is what the
    ///   autodiff pass actually emits.
    ActivationBackward {
        kind: Activation,
    },

    /// Backward for `Op::FakeQuantize` under a non-default STE.
    /// Inputs `[x, dy]`: the forward input and the upstream
    /// gradient. Output `dx` same shape. The `bits`/`axis`/`ste`
    /// fields must match the forward op so the kernel computes the
    /// same per-channel scale and applies the right STE attenuation.
    /// For `SteKind::Identity` this op is unnecessary — autodiff
    /// just routes `upstream` through unchanged.
    FakeQuantizeBackward {
        bits: u8,
        axis: Option<usize>,
        ste: SteKind,
    },

    /// 2D max-pool backward. Routes each element of `dy` back into the
    /// position in `x`'s window where the forward max was taken.
    /// Inputs: `[x, dy]` with `x [N, C, H, W]` and
    /// `dy [N, C, H_out, W_out]`. Output: same shape as `x`.
    /// Carries the forward pool's geometry so the kernel can recompute
    /// the argmax position per window without a saved-indices tensor.
    MaxPool2dBackward {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },

    /// 2D conv backward w.r.t. input. Computes `dx = conv_transpose(dy, w)`.
    /// Inputs: `[dy, w]` with `dy [N, C_out, H_out, W_out]` and
    /// `w [C_out, C_in/groups, kH, kW]`. Output: `[N, C_in, H, W]`
    /// (declared on the node — caller knows the original input shape).
    /// Geometry is the forward conv's parameters, not the transposed
    /// conv's.
    Conv2dBackwardInput {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },

    /// 2D conv backward w.r.t. weight. Computes
    /// `dw[c_out, c_in, kh, kw] = sum_{n,h_out,w_out} x[n,c_in,...] * dy[n,c_out,h_out,w_out]`.
    /// Inputs: `[x, dy]`. Output: `[C_out, C_in/groups, kH, kW]`.
    Conv2dBackwardWeight {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },

    /// 3D max-pool backward on NCDHW. Inputs: `[x, dy]`. Output: same as `x`.
    MaxPool3dBackward {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },

    /// 3D conv backward w.r.t. input (NCDHW). Inputs: `[dy, w]`.
    Conv3dBackwardInput {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },

    /// 3D conv backward w.r.t. weight (NCDHW). Inputs: `[x, dy]`.
    Conv3dBackwardWeight {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },

    /// Fused softmax + cross-entropy against a **dense** target
    /// distribution (soft labels / one-hot probabilities) — the
    /// companion to the sparse integer-label
    /// [`Op::SoftmaxCrossEntropyWithLogits`]. Per-row output:
    /// `loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]`
    ///         = -Σ_c targets[n,c]·log_softmax(logits[n])[c]`.
    /// Inputs: `[logits, targets]`, both `[N, C]`. Output: `[N]`.
    /// Caller does the `Reduce::Mean` if they want a scalar.
    /// Numerically stable (max-subtract before exp). Used for label
    /// smoothing and knowledge distillation where targets are full
    /// probability rows rather than class indices.
    SoftmaxCrossEntropy,

    /// Fused softmax + cross-entropy loss with integer (f32-encoded)
    /// targets — the standard classification loss. Per-row output:
    /// `loss[n] = -log(softmax(logits[n])[labels[n]])`.
    /// Inputs: `[logits, labels]` with `logits [N, C]` and
    /// `labels [N]` (f32-encoded class indices). Output: `[N]`.
    /// Caller does the `Reduce::Mean` if they want a scalar.
    SoftmaxCrossEntropyWithLogits,

    /// Backward of the fused loss above. Emits
    /// `dlogits[n,c] = (softmax(logits[n])[c] - one_hot(labels)[n,c]) * d_loss[n]`.
    /// Inputs: `[logits, labels, d_loss]`. Output: `[N, C]` (same shape
    /// as `logits`). Recomputes the softmax inline rather than threading
    /// it through from the forward node.
    SoftmaxCrossEntropyBackward,

    /// Backward of [`Op::Attention`]. Recomputes scaled `QK^T`, applies
    /// the same `mask_kind` as the forward op, softmaxes scores, then
    /// emits **one** of `dQ`, `dK`, or `dV` selected by [`AttentionBwdWrt`].
    /// Autodiff emits three nodes (one per `wrt`) so each output shape
    /// stays a normal single-output MIR node.
    ///
    /// Inputs: `[q, k, v, dy]` plus optional mask when `mask_kind` is
    /// [`MaskKind::Custom`] or [`MaskKind::Bias`] (same convention as
    /// forward). Output shape matches `q`, `k`, or `v` respectively.
    AttentionBackward {
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        wrt: AttentionBwdWrt,
    },

    // ── Fused operations (created by optimization passes) ──────
    /// Fused matmul + bias + activation. Created from MatMul → Add → Activation.
    FusedMatMulBiasAct {
        activation: Option<Activation>,
    },

    /// Fused convolution + bias + activation. Created from
    /// `Conv → Reshape(bias→[1,C,1,1]) → Expand → Add → [Activation]`.
    ///
    /// Inputs: `[input, weight, bias]` — bias is rank-1 `[C_out]`, mirroring
    /// [`Op::QConv2d`]'s bundled-bias convention (not a separate broadcast Add).
    /// Fields mirror [`Op::Conv`] plus the epilogue activation.
    ///
    /// Only backends that can apply the whole epilogue in one dispatch claim
    /// this op — today CUDA. Every other backend decomposes it back to
    /// primitives in `unfuse`, so the fusion is always safe to emit. On CUDA it
    /// lowers to cuDNN's fused `cudnnConvolutionBiasActivationForward`. The
    /// fusion pass only emits it for cuDNN-friendly shapes with an
    /// identity/ReLU epilogue (the combination that benchmarks faster than
    /// unfused); the `conv_bias_act_epilogue` kernel exists only as a
    /// correctness fallback when libcudnn is absent at runtime.
    ///
    /// `has_residual`: when true a 4th input `residual` (full `[N,C,H,W]` tensor)
    /// is added before the activation — `act(conv + bias + residual)` — mapping
    /// to cuDNN's `z` operand. Used for ResNet `conv→bias→add(residual)→relu`.
    FusedConvBiasAct {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        activation: Option<Activation>,
        has_residual: bool,
    },

    /// Fused residual + optional bias + layer norm.
    /// Created from Add(x, residual) → [Add(bias)] → LayerNorm.
    FusedResidualLN {
        has_bias: bool,
        eps: f32,
    },

    /// Fused residual + optional bias + RMS norm.
    /// Created from Add(x, residual) → [Add(bias)] → RmsNorm.
    FusedResidualRmsNorm {
        has_bias: bool,
        eps: f32,
    },

    /// Fused SwiGLU: split input into up/gate halves, silu(gate) * up.
    /// Created from Split → Silu → Mul when fed by a concatenated matmul.
    ///
    /// `cast_to`: optional output dtype — when `Some(dt)` the kernel casts
    /// its result from the input dtype to `dt` in-register, saving a
    /// separate cast pass. Reserved for future fp8/fp4 quantization paths;
    /// for f32→f16 mixed precision the AutoMixedPrecision pass already
    /// inserts a Cast node so this stays `None` in current pipelines.
    FusedSwiGLU {
        cast_to: Option<DType>,
        /// When `true`, the concatenated input stores gate in the low half
        /// `[..., 0..N)` and up in the high half `[..., N..2N)` — the layout
        /// produced when gate projection is emitted before up in the builder.
        /// Default `false`: up @ low, gate @ high (canonical concat order).
        gate_first: bool,
    },

    /// Fused full transformer layer: attention block + residual+LN + FFN + residual+LN.
    /// All intermediates resident in registers/threadgroup memory; one kernel
    /// per layer instead of ~30 (the CPU's batch=1 win, lifted to IR so any
    /// backend can implement it as a monolithic kernel).
    ///
    /// Inputs: hidden, qkv_w, qkv_b, out_w, out_b,
    ///         ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b, ln2_g, ln2_b, mask
    /// Output: same shape as hidden.
    ///
    /// **Backend status:** same as FusedAttentionBlock. CPU implements
    /// the L1-cache-resident merge at the thunk level. Metal deferred —
    /// requires a single MSL kernel for the whole layer to actually
    /// beat the unfused path. Multi-day work; revisit when there's a
    /// model whose Metal inference is bottlenecked here rather than on
    /// the wait latency floor.
    FusedTransformerLayer {
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        eps1: f32,
        eps2: f32,
        activation: Activation,
        has_bias: bool,
    },

    /// Fused attention block: QKV projection → split → \[RoPE\] → SDPA → output projection.
    /// Created by FuseAttentionBlock pass when batch*seq is small.
    /// All intermediates stay in L1 cache — no arena writes between ops.
    ///
    /// Inputs (in order):
    ///   hidden, qkv_w, out_w, mask,
    ///   [qkv_b, out_b]      if has_bias,
    ///   [rope_cos, rope_sin] if has_rope
    ///
    /// **Backend status (Phase C finalize):**
    ///   CPU  — implemented at the *thunk* level: the CPU schedule
    ///          recognizes the multi-thunk pattern and merges into
    ///          a single FusedAttnBlock that keeps Q/K/V in stack
    ///          buffers across stages (the L1-cache win).
    ///   Metal — **deferred**. A dispatch-wrapper version (chaining
    ///          existing kernels) buys nothing the unfused Metal path
    ///          doesn't already get, since per-run cost is dominated
    ///          by `wait_until_completed` (~150 µs), not encode. The
    ///          real win is a single MSL kernel keeping Q/K/V in
    ///          threadgroup memory across stages — multi-day work.
    ///          Until then, Metal runs the unfused chain (one matmul,
    ///          three narrows, two ropes, attention, one matmul) — all
    ///          covered in op_coverage and parity_harness.
    FusedAttentionBlock {
        num_heads: usize,
        head_dim: usize,
        has_bias: bool,
        has_rope: bool,
    },

    // ── Control flow (subgraphs as op payloads) ─────────────────
    //
    // Status: IR is defined; helper `run_if` / `run_while` exist in
    // rlx-runtime/src/subgraph.rs; **executor wiring is not yet
    // implemented** (both CPU thunk and Metal thunk fall through to
    // `Thunk::Nop` for these ops). Wiring requires:
    //   1. Recursive subgraph compile at parent-compile time.
    //   2. Per-subgraph input/output binding through the arena.
    //   3. Schedule-level dispatch when the predicate / loop cond is
    //      resolved at runtime.
    // Estimate: 4–6 hours of focused work + parity tests. Deferred
    // because no current in-tree model uses these ops;
    // surface area without a validation target invites silent bugs.
    /// Conditional: pick between two subgraphs based on a boolean predicate.
    /// Inputs: [predicate, ...captures (used inside both branches)].
    /// `then_branch` and `else_branch` are sub-graphs that share the
    /// captured inputs and must produce identically-shaped outputs.
    /// Used for: shape-dependent execution, batched inference of
    /// dynamic-length sequences with padding masks.
    If {
        then_branch: Box<crate::Graph>,
        else_branch: Box<crate::Graph>,
    },

    /// Loop: iterate `body` while `cond` evaluates true.
    /// Inputs: [...initial loop-carried values].
    /// `cond`'s single output is a Bool scalar.
    /// `body`'s outputs become the next iteration's loop-carried inputs.
    /// Outputs of While are the values after the final iteration.
    /// Used for: KV-cache-driven autoregressive generation, beam search.
    While {
        cond: Box<crate::Graph>,
        body: Box<crate::Graph>,
        max_iterations: Option<usize>,
    },

    /// Bounded-length loop with a fixed-shape carry, optional per-step
    /// inputs, and optional stacked output. Mirrors JAX's `lax.scan`.
    ///
    /// Body signature: `(carry, x_t_0, ..., x_t_{num_xs-1}) → carry_next`
    /// — `1 + num_xs` Op::Inputs in NodeId construction order (first
    /// declared is the carry; the remaining `num_xs` are per-step
    /// slices). Single output (the next carry).
    ///
    /// Outer Op::Scan inputs (in order):
    ///   `[init_carry, xs_0, xs_1, ..., xs_{num_xs-1}]`
    /// Each `xs_i` has shape `[length, *per_step_shape_i]`; the body
    /// sees `xs_i[t]` (a `per_step_shape_i` slice) on iteration `t`.
    ///
    /// Outer Op::Scan output:
    ///   * `save_trajectory == false` — final carry, shape `*carry`.
    ///   * `save_trajectory == true`  — stacked trajectory of carries,
    ///     shape `[length, *carry]`. Row `t` is the carry after step
    ///     `t+1`, so row `length-1` matches the no-trajectory case.
    ///
    /// Mirrors JAX's `lax.scan`. Common uses include time-stepping
    /// integrators with time-varying drives, Mamba-style SSM scans
    /// reading per-step inputs, and RNN-style sequence processing.
    Scan {
        body: Box<crate::Graph>,
        length: u32,
        save_trajectory: bool,
        /// Number of "broadcast" inputs — values that are constant
        /// across iterations. Outer scan inputs in order:
        ///   `[init, bcast_0..bcast_{B-1}, xs_0..xs_{X-1}]`
        /// Body Op::Inputs in NodeId order:
        ///   `[carry, bcast_0..bcast_{B-1}, x_t_0..x_t_{X-1}]`
        /// CPU executor fills bcast slots ONCE before the iteration
        /// loop (xs slots are filled per-step). The reverse-mode AD
        /// pre-pass materialises each bcast into an xs of shape
        /// `[length, *bcast]` via broadcast `Mul` so the rest of the
        /// VJP / executor pipeline can stay unchanged. `0` (default)
        /// keeps the original carry+xs scan shape.
        num_bcast: u32,
        /// Number of per-step `xs` inputs. Total outer Op::Scan
        /// inputs is `1 + num_bcast + num_xs`.
        num_xs: u32,
        /// Number of trajectory checkpoints when `save_trajectory ==
        /// true`. `0` means "save all `length` rows" (default). A
        /// positive value `K` means save only `K` evenly-spaced rows
        /// at indices `floor(t * length / K)` for `t in 0..K`. Used
        /// by recursive checkpointed AD: store O(√T) carries during
        /// forward, recompute the rest in the backward pass.
        ///
        /// When `0` (or `K == length`), the saved trajectory has
        /// shape `[length, *carry]` — same as the original behavior.
        /// When `0 < K < length`, the saved trajectory has shape
        /// `[K, *carry]`.
        num_checkpoints: u32,
    },

    /// Reverse-mode AD companion to `Op::Scan` — extracts the carry
    /// gradient `dinit`. Walks `t = length-1 .. 0`, applying `body_vjp`
    /// to thread `dcarry` back through the time loop.
    ///
    /// Inputs (in order):
    ///   `[init, trajectory, upstream, xs_0, ..., xs_{num_xs-1}]`
    /// Output: `dinit`, shape = carry shape.
    ///
    /// `body_vjp` is the result of
    /// `autodiff::grad(body, [carry_id, xs_0_id, ..., xs_{num_xs-1}_id])`
    /// — a graph with `1 + num_xs + 1` Inputs (carry + x_t_i for each
    /// xs + `"d_output"`) and `1 + num_xs` outputs
    /// (dcarry + dx_t_i for each xs). This op reads `outputs[0]` =
    /// dcarry; the sibling [`Self::ScanBackwardXs`] reads the
    /// `outputs[1 + xs_idx]` slot for each xs gradient.
    ScanBackward {
        body_vjp: Box<crate::Graph>,
        length: u32,
        save_trajectory: bool,
        num_xs: u32,
        /// When `0` or equal to `length`, the trajectory input has
        /// shape `[length, *carry]` — every step's carry is cached
        /// (`CheckpointStrategy::All`). When `0 < K < length`, the
        /// trajectory input has shape `[K, *carry]` and the executor
        /// recomputes intermediate carries via `forward_body` between
        /// checkpoints. `forward_body` must be `Some` whenever this
        /// is < length.
        num_checkpoints: u32,
        /// Forward body (the same `body` from the forward Op::Scan).
        /// Required when `num_checkpoints > 0 && < length` so the
        /// executor can recompute carries between saved checkpoints.
        /// `None` for the All strategy (no recompute needed).
        forward_body: Option<Box<crate::Graph>>,
    },

    /// Companion to [`Self::ScanBackward`] that extracts one stacked
    /// per-step `dxs_i` (shape `[length, *per_step_xs_i]`). Same inputs
    /// and same `body_vjp` graph as ScanBackward — `xs_idx` selects
    /// which body_vjp output to stack into the result.
    ///
    /// Note: each ScanBackwardXs runs its own backward loop. A future
    /// optimization can fuse them into a single multi-output backward
    /// kernel; for now it's `1 + num_xs` independent sweeps.
    ScanBackwardXs {
        body_vjp: Box<crate::Graph>,
        length: u32,
        save_trajectory: bool,
        num_xs: u32,
        xs_idx: u32,
        num_checkpoints: u32,
        forward_body: Option<Box<crate::Graph>>,
    },

    /// CPU reference 3D Gaussian splat forward render.
    ///
    /// Seven flat F32 inputs (scene buffers + camera/render meta):
    ///   0. positions `[N*3]`
    ///   1. scales `[N*3]` (log-space)
    ///   2. rotations `[N*4]` (xyzw)
    ///   3. opacities `[N]` (logit)
    ///   4. colors `[N*3]` (linear RGB)
    ///   5. sh_coeffs `[N * sh_coeff_count * 3]`
    ///   6. meta `[23]` — camera position/target/up/fov/near/far, background RGB,
    ///      then width/height/tile_size/radius_scale/alpha_cutoff/max_splat_steps/
    ///      transmittance_threshold/max_list_entries as f32 bit-patterns.
    ///
    /// Output: `[height * width * 4]` linear RGBA (display gamma baked in).
    /// Build via [`crate::Graph::gaussian_splat_render`].
    ///
    /// Differentiable backward is not implemented in v1; autodiff treats this
    /// op as non-differentiable (same as [`Op::Sample`]).
    GaussianSplatRender {
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },

    /// Backward pass for [`Self::GaussianSplatRender`].
    ///
    /// Eight inputs: the same seven as forward plus `d_loss_rgba` `[W*H*4]`
    /// (only RGB channels are used). Re-runs the training forward internally.
    ///
    /// Output: packed gradients
    /// `[positions(3N) | scales(3N) | rotations(4N) | opacities(N) | colors(3N) | sh(N*sh*3)]`.
    /// Unpack via [`crate::ops::splat::unpack_gaussian_splat_packed_grads`].
    GaussianSplatRenderBackward {
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
        loss_grad_clip: f32,
        sh_band: u32,
        max_anisotropy: f32,
    },

    /// Strict IR stage 1: project, bin, sort, build per-pixel rays.
    ///
    /// Seven inputs (same scene + meta as [`Self::GaussianSplatRender`]). Output: packed
    /// prepare buffer (see `rlx_splat::prep_layout::prep_packed_len`).
    GaussianSplatPrepare {
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },

    /// Strict IR stage 2: tile raster from [`Self::GaussianSplatPrepare`] output.
    ///
    /// Inputs: `prep` packed buffer, `meta` `[23]`. Output: `[width * height * 4]` RGBA.
    GaussianSplatRasterize {
        width: u32,
        height: u32,
        tile_size: u32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },

    /// User-registered custom op. `name` keys into the
    /// [`crate::op_registry`] for shape inference, autodiff, and
    /// per-backend execution. `attrs` is an opaque blob passed
    /// through to those callbacks (FFT direction, SparseLU
    /// reordering strategy, etc.). `num_inputs` is captured at
    /// construction time so [`Op::num_inputs`] stays infallible
    /// without a registry lookup. Build via [`crate::Graph::custom_op`].
    Custom {
        name: String,
        num_inputs: u32,
        attrs: Vec<u8>,
    },

    /// 1D Fast Fourier Transform along the last axis.
    ///
    /// **Layouts**
    /// - `F32` / `F64`: 2N real-block — last axis is `[re₀…re_{N-1}, im₀…im_{N-1}]`.
    /// - `C64`: interleaved `[re, im]` pairs per complex element along the last axis.
    ///
    /// **ND transforms** — use `Graph::fftn` / `Graph::ifftn`, which compose
    /// `fft_axis` (transpose → Fft → transpose). Multi-axis `fftn` requires
    /// `DType::C64`; the 2N-block layout describes a single complex axis.
    ///
    /// Default (`FftNorm::Backward`) is **unnormalized** on both directions:
    ///   `fft(x)[k] = Σ x[n]·exp(-2πi·nk/N)`
    ///   `ifft(y)[n] = Σ y[k]·exp(+2πi·nk/N)`
    /// so `ifft(fft(x)) = N·x`. Use `FftNorm::Forward` for gpu-fft-style
    /// `1/N` scaling on inverse, or `FftNorm::Ortho` for unitary scaling.
    ///
    /// AD: VJP(`fft`) = `ifft`, VJP(`ifft`) = `fft` when `norm=Backward`;
    /// other norms apply the chain rule via output scaling.
    Fft {
        inverse: bool,
        norm: crate::fft::FftNorm,
    },

    /// Ternary pruned radix-2 butterfly stage on interleaved complex state.
    ///
    /// Inputs:
    ///   0 — state `[batch, n_fft, 2]` (re/im on axis 2)
    ///   1 — gate `[half]` — 0 = identity, 1 = run butterfly (`half = n_fft/2`)
    ///   2 — rev `[half]` — 0 = forward, 1 = swap outputs when gate=1
    ///   3 — tw_re `[half]`
    ///   4 — tw_im `[half]`
    ///
    /// Output: `[batch, n_fft, 2]` same layout. Slots with gate=0 copy inputs
    /// without twiddle math.
    FftButterflyStage {
        stage: u32,
        n_fft: u32,
    },

    /// Log-mel spectrogram from RLX FFT block-layout spectrum.
    ///
    /// Inputs:
    ///   0 — spectrum `[..., 2*n_fft]` (re plane then im plane, same as `Op::Fft` output)
    ///   1 — mel filterbank `[n_mels, n_bins]` with `n_bins = n_fft/2 + 1`
    ///
    /// Output: `[..., n_mels]` with Whisper dynamic-range compression
    /// (`log10`, clamp to max−8 dB, `(x+4)/4`).
    LogMel,

    /// VJP of [`Op::LogMel`] w.r.t. spectrum (input 0).
    ///
    /// Inputs: spectrum block, mel filters, upstream `dy`.
    /// Output: `d_spectrum` (same shape as input 0).
    LogMelBackward,

    /// Top-K Welch peaks from block-layout segment spectra.
    ///
    /// Input 0: spectrum `[batch * n_segments, 2*n_fft]` (re ∥ im planes).
    /// Output: `[batch, k*2]` interleaved `(bin, power)` per spike.
    WelchPeaks {
        k: usize,
        n_segments: usize,
    },

    /// User-defined sub-graph with optional override AD rules.
    /// Mirrors JAX's `custom_vjp` / `custom_jvp` decorators: the
    /// caller wraps a forward computation and supplies its own
    /// reverse- and/or forward-mode AD bodies. Useful when:
    ///   * The forward is iterative (Newton, fixed-point) and
    ///     differentiating through the loop is wasteful — the
    ///     vjp_body computes the implicit-function gradient at the
    ///     converged point in one shot.
    ///   * The math has a closed-form gradient that's much cheaper
    ///     than autodiff.
    ///   * The forward op is non-differentiable by tracing
    ///     (sampling, argmax) and the user wants a smooth surrogate.
    ///
    /// **fwd_body**: `num_inputs` Op::Inputs in NodeId construction
    /// order, one Op::Output (the primal y). Forward execution
    /// inlines this body once.
    ///
    /// **vjp_body** (optional): Op::Inputs are `num_inputs` primal
    /// inputs in NodeId order, plus two special-named Inputs —
    /// `"primal_output"` (the y from forward) and `"d_output"` (the
    /// upstream gradient). Outputs: `num_inputs` tensors in
    /// `set_outputs` order, matching the gradients of each primal
    /// input. When `None`, reverse-mode AD recurses into fwd_body
    /// — same as if the op were inlined.
    ///
    /// **jvp_body** (optional): Op::Inputs are `num_inputs` primal
    /// inputs in NodeId order, `num_inputs` special-named Inputs
    /// `"tangent_0"..="tangent_{num_inputs-1}"` carrying each input's
    /// tangent, and an optional special-named `"primal_output"` Input
    /// (the y from forward, useful when the JVP must be evaluated at
    /// a converged / nonlinear point — e.g. IFT-style forward-mode AD
    /// of an iterative solver). Output: 1 tensor (the tangent of y).
    /// When `None`, forward-mode AD recurses into fwd_body.
    ///
    /// `num_inputs` is captured so [`Op::num_inputs`] stays
    /// infallible. Build via [`crate::Graph::custom_fn`].
    CustomFn {
        fwd_body: Box<crate::Graph>,
        vjp_body: Option<Box<crate::Graph>>,
        jvp_body: Option<Box<crate::Graph>>,
        num_inputs: u32,
    },

    // ── Riemannian / SPD-manifold layers ────────────────────────
    //
    // Building blocks of SPDNet (Huang & Van Gool, AAAI 2017) and
    // Riemannian batch-norm (Brooks et al., NeurIPS 2019). They operate
    // on symmetric-positive-definite matrices (covariance descriptors:
    // EEG/BCI, diffusion tensors, second-order pooling). F64; compute
    // bodies live in `rlx_cpu::spd`. Executed natively on CPU and via CPU
    // host-fallback on Metal / Vulkan / wgpu / CUDA / ROCm / oneAPI (each
    // delegates to `rlx_cpu::spd` through the thunk path — widening its
    // f32 arena to f64 and back). Backends that compile to a closed f32
    // program (MLX / CoreML / TPU / WebGL / QNN) can't run F64
    // eigendecomposition and reject these ops cleanly at legalize
    // (→ pin such graphs to `Device::Cpu`, like `Fft`/`DenseSolve`).
    /// BiMap (bilinear mapping) SPDNet layer: `Y = W · X · Wᵀ`.
    ///
    /// Inputs: `[W [m,n], X [n,n]]` (X symmetric SPD). Output: `Y [m,m]`
    /// (SPD when W has full row rank). Reduces the manifold dimension.
    /// The Stiefel / semi-orthogonality constraint on `W` is the
    /// optimizer's responsibility, not this op's. VJP decomposes to
    /// matmuls: `dW = (Ḡ+Ḡᵀ)·W·X`, `dX = Wᵀ·Ḡ·W`.
    BiMap,

    /// ReEig (eigenvalue rectification) SPDNet nonlinearity:
    /// `Y = U · max(ε, Σ) · Uᵀ` where `X = U Σ Uᵀ`. Input `[X [n,n]]`
    /// symmetric → `Y [n,n]` SPD. The SPD analogue of ReLU: floors the
    /// spectrum at `eps` to keep the output well-conditioned.
    ReEig {
        eps: f32,
    },

    /// LogEig SPDNet layer: `Y = U · log(Σ) · Uᵀ = logm(X)`. Maps the
    /// SPD manifold to the tangent space at the identity (symmetric
    /// matrices) so a plain Euclidean head can classify. Input
    /// `[X [n,n]]` → `Y [n,n]`. `eps` floors the spectrum before `log`.
    LogEig {
        eps: f32,
    },

    /// SPD batch-norm affine transport (Brooks et al. 2019):
    /// `Y_i = G^{1/2} · (M^{-1/2} X_i M^{-1/2}) · G^{1/2}`.
    ///
    /// Inputs: `[X [B,n,n], M [n,n], G [n,n]]` — a batch of SPD matrices,
    /// the (detached) Fréchet mean `M`, and the learnable SPD bias `G`.
    /// Output: `Y [B,n,n]`. Centers the batch to the identity then biases
    /// toward `G`. `M` receives no gradient (detached statistic, like
    /// Euclidean batch-norm's running mean); grads flow to `X` and `G`.
    SpdBatchNorm {
        eps: f32,
    },

    /// Fréchet / Karcher mean of a batch of SPD matrices under the AIRM
    /// metric (iterative tangent averaging). Input `[X [B,n,n]]` →
    /// `M [n,n]`. Non-differentiable (a detached batch statistic — its
    /// VJP contributes no gradient), mirroring how batch-norm's running
    /// statistics live outside the autodiff graph.
    SpdKarcherMean {
        iters: u32,
        tol: f32,
    },

    /// Weighted Fréchet / Karcher mean of a batch of SPD matrices under the
    /// AIRM metric — the barycentre `argmin_M Σᵢ wᵢ · δ²(M, Cᵢ)`. Inputs
    /// `[X [B,n,n], w [B]]` → `M [n,n]`. The weighted counterpart of
    /// [`Op::SpdKarcherMean`]: the true AIRM barycentre a barycentric-OT
    /// projection / soft-clustering / weighted-MDM needs (not the
    /// log-Euclidean shortcut `expm(Σ w̄ᵢ logm Cᵢ)`). Non-differentiable — a
    /// detached statistic, exactly like [`Op::SpdKarcherMean`].
    SpdKarcherMeanWeighted {
        iters: u32,
        tol: f32,
    },

    /// AIRM Riemannian **logarithm** at an arbitrary base point:
    /// `Log_P(X) = P^{1/2} · logm(P^{-1/2} X P^{-1/2}) · P^{1/2}`. Inputs
    /// `[base [n,n], x [n,n]]` → tangent vector `[n,n]`. Generalises
    /// [`Op::LogEig`] (which is `Log_I`) to any base point — the Riemannian
    /// OT / domain-adaptation primitive (Yair et al. 2019).
    SpdLogMap,

    /// AIRM Riemannian **exponential** at an arbitrary base point:
    /// `Exp_P(V) = P^{1/2} · expm(P^{-1/2} V P^{-1/2}) · P^{1/2}`. Inputs
    /// `[base [n,n], v [n,n]]` → SPD point `[n,n]`. Inverse of
    /// [`Op::SpdLogMap`].
    SpdExpMap,

    /// AIRM **parallel transport** of a tangent vector between base points:
    /// `Γ_{P→Q}(V) = E · V · Eᵀ`, `E = (Q P^{-1})^{1/2}`. Inputs
    /// `[from [n,n], to [n,n], v [n,n]]` → transported tangent vector `[n,n]`.
    /// The AIRM isometry that moves tangent vectors between base points for
    /// Riemannian OT / domain adaptation (Yair et al. 2019).
    SpdParallelTransport,

    /// Batched SPD spectral matrix function — applies `kind` (logm / expm /
    /// sqrtm / invsqrtm) to each matrix of a batch. Input `[X [B,n,n]]` →
    /// `[B,n,n]`. The graph-level, any-backend counterpart of
    /// `rlx_cpu::spd::{logm,expm,sqrtm,invsqrtm}_batch`.
    SpdMatrixFnBatch {
        kind: SpdMatFn,
    },

    /// Backward of [`Op::ReEig`]: `(X [n,n], dY [n,n]) → dX [n,n]` via
    /// the Daleckii–Krein (Loewner) adjoint with `f(λ)=max(λ,ε)`.
    ReEigBackward {
        eps: f32,
    },

    /// Backward of [`Op::LogEig`]: `(X, dY) → dX` via the Loewner adjoint
    /// with `f=log`, `f′=1/λ`.
    LogEigBackward {
        eps: f32,
    },

    /// Backward of [`Op::SpdBatchNorm`] w.r.t. the inputs `X` (mean
    /// detached): `(M [n,n], G [n,n], dY [B,n,n]) → dX [B,n,n]`.
    SpdBatchNormBackwardX {
        eps: f32,
    },

    /// Backward of [`Op::SpdBatchNorm`] w.r.t. the learnable bias `G`
    /// (mean detached): `(X [B,n,n], M [n,n], G [n,n], dY [B,n,n]) →
    /// dG [n,n]`.
    SpdBatchNormBackwardG {
        eps: f32,
    },

    /// Backward of [`Op::SpdLogMap`]. Inputs `[base [n,n], x [n,n], dY [n,n]]` →
    /// packed `[d_base ∥ d_x]` of shape `[2, n, n]` (the builder narrows the two
    /// gradients out). Full AIRM adjoint — the base point receives a gradient.
    SpdLogMapBackward,

    /// Backward of [`Op::SpdExpMap`]. Inputs `[base [n,n], v [n,n], dY [n,n]]` →
    /// packed `[d_base ∥ d_v]` of shape `[2, n, n]`.
    SpdExpMapBackward,

    /// Backward of [`Op::SpdParallelTransport`].
    /// Inputs `[from [n,n], to [n,n], v [n,n], dY [n,n]]` → packed
    /// `[d_from ∥ d_to ∥ d_v]` of shape `[3, n, n]`.
    SpdParallelTransportBackward,

    /// Backward of [`Op::SpdMatrixFnBatch`]: per-slice Daleckii–Krein adjoint of
    /// `kind`. Inputs `[X [B,n,n], dY [B,n,n]]` → `dX [B,n,n]`.
    SpdMatrixFnBatchBackward {
        kind: SpdMatFn,
    },

    /// Symmetric **eigendecomposition** of an `[n,n]` matrix. Output is packed
    /// `[λ (n) ∥ U (n²)]` — ascending eigenvalues then the row-major eigenvector
    /// matrix (column `j` = eigenvector `j`, so `A = U diag(λ) Uᵀ`). The builder
    /// [`Graph::eigh`] narrows the two views out. Differentiable (see
    /// [`Op::EighBackward`]). The seam a native batched eigensolver (cuSOLVER
    /// `syevjBatched` &c.) plugs into.
    Eigh,

    /// Backward of [`Op::Eigh`]. Inputs `[fwd [n²+n], bar [n²+n]]` — the packed
    /// forward `[λ ∥ U]` and the upstream cotangent `[λ̄ ∥ Ū]` — → `Ā [n,n]`
    /// (the standard symmetric-eigendecomposition adjoint).
    EighBackward,

    /// Batched [`Op::Eigh`]: `[B,n,n] → [B, n²+n]` packed per slice.
    EighBatch,

    /// Backward of [`Op::EighBatch`]. Inputs `[fwd [B,n²+n], bar [B,n²+n]]` →
    /// `Ā [B,n,n]`.
    EighBatchBackward,

    // ── Diffusion Transformer (DiT) modulation ──────────────────
    /// Fused adaptive-norm modulation — the signature DiT / adaLN-Zero op.
    ///
    /// ```text
    /// out = norm(x) * (1 + scale) + shift
    /// ```
    ///
    /// `norm` normalizes over the **last axis** with no learnable affine
    /// ([`AdaNormKind`]); the affine is supplied per-conditioning by `scale`
    /// and `shift`, which broadcast over every axis except the last (the DiT
    /// `[B, 1, D]` modulation over `[B, S, D]`). Folding the broadcast into the
    /// op avoids the `Op::Expand` materialization the decomposed form needs
    /// (`[B,1,D]→[B,S,D]` can't ride an `ElementwiseRegion`'s trailing-modulo
    /// broadcast) and keeps the pre-norm activation off the heap — mirroring
    /// [`Op::FusedResidualLN`].
    ///
    /// Inputs: `[x, scale, shift]`
    ///   - `x`:     `[..., D]`
    ///   - `scale`: broadcast-compatible with `x`, last dim `D` (e.g. `[B, 1, D]`)
    ///   - `shift`: same shape as `scale`
    ///
    /// Output shape == `x`. Backends without a native kernel decompose via
    /// `unfuse` into `norm → mul(1+scale) → add(shift)`.
    AdaLayerNorm {
        norm: AdaNormKind,
        eps: f32,
    },

    /// Gated residual `out = x + gate * y` (the DiT post-sublayer combine).
    ///
    /// `gate` broadcasts over every axis except the last (the `[B, 1, D]`
    /// modulation gate over `[B, S, D]`), so — like [`Op::AdaLayerNorm`] — the
    /// built-in broadcast avoids an `Op::Expand` of the gate. Used for both the
    /// attention (`gate_msa`) and MLP (`gate_mlp`) residual joins.
    ///
    /// Inputs: `[x, y, gate]`
    ///   - `x`:    residual stream `[..., D]`
    ///   - `y`:    sublayer output, same shape as `x`
    ///   - `gate`: broadcast-compatible with `x`, last dim `D` (e.g. `[B, 1, D]`)
    ///
    /// Output shape == `x`. Backends without a native kernel decompose via
    /// `unfuse` into `mul(gate, y) → add(x)`.
    GatedResidual,

    /// Packed reverse of [`Op::AdaLayerNorm`].
    ///
    /// Inputs: `[x, scale, shift, dy]` (same shapes as the forward).
    /// Output: 1-D `[nx + 2·ns]` packing `dx` (numel `nx`), `dscale` (`ns`),
    /// `dshift` (`ns`) in that order — avoids materializing `Expand` of
    /// `[B,1,D]` modulation in the backward graph. Unpack with
    /// [`crate::ops::backward::unpack_ada_layer_norm_backward`].
    AdaLayerNormBackward {
        norm: AdaNormKind,
        eps: f32,
    },

    /// Packed reverse of [`Op::GatedResidual`].
    ///
    /// Inputs: `[x, y, gate, dy]`. Output: 1-D `[nx + ny + ng]` packing
    /// `dx`, `dy`, `dgate` (row-major). Unpack with
    /// [`crate::ops::backward::unpack_gated_residual_backward`].
    GatedResidualBackward,
}

impl Op {
    /// PLAN L4: discriminant for backend-supported-set checks.
    /// Stable, parameter-free identity per variant — `Op::Activation(_)`
    /// and `Op::Activation(Relu)` share the same `OpKind::Activation`.
    pub fn kind(&self) -> OpKind {
        match self {
            Op::Input { .. } => OpKind::Input,
            Op::Param { .. } => OpKind::Param,
            Op::Constant { .. } => OpKind::Constant,
            Op::Activation(_) => OpKind::Activation,
            Op::Cast { .. } => OpKind::Cast,
            Op::StopGradient => OpKind::StopGradient,
            Op::Quantize { .. } => OpKind::Quantize,
            Op::Dequantize { .. } => OpKind::Dequantize,
            Op::FakeQuantize { .. } => OpKind::FakeQuantize,
            Op::FakeQuantizeLSQ { .. } => OpKind::FakeQuantizeLSQ,
            Op::FakeQuantizeLSQBackwardX { .. } => OpKind::FakeQuantizeLSQBackwardX,
            Op::FakeQuantizeLSQBackwardScale { .. } => OpKind::FakeQuantizeLSQBackwardScale,
            Op::Binary(_) => OpKind::Binary,
            Op::Compare(_) => OpKind::Compare,
            Op::Where => OpKind::Where,
            Op::Fma => OpKind::Fma,
            Op::ElementwiseRegion { .. } => OpKind::ElementwiseRegion,
            Op::TransformRegion { .. } => OpKind::TransformRegion,
            Op::BatchElementwiseRegion { .. } => OpKind::BatchElementwiseRegion,
            Op::MatMul => OpKind::MatMul,
            Op::DotGeneral { .. } => OpKind::DotGeneral,
            Op::DenseSolve => OpKind::DenseSolve,
            Op::BatchedDenseSolve => OpKind::BatchedDenseSolve,
            Op::Cholesky => OpKind::Cholesky,
            Op::TriangularSolve { .. } => OpKind::TriangularSolve,
            Op::Det => OpKind::Det,
            Op::LogDet => OpKind::LogDet,
            Op::Sort { .. } => OpKind::Sort,
            Op::ArgSort { .. } => OpKind::ArgSort,
            Op::Svd { .. } => OpKind::Svd,
            Op::Qr { .. } => OpKind::Qr,
            Op::LayerNorm { .. } => OpKind::LayerNorm,
            Op::LayerNorm2d { .. } => OpKind::LayerNorm2d,
            Op::GroupNorm { .. } => OpKind::GroupNorm,
            Op::BatchNormInference { .. } => OpKind::BatchNormInference,
            Op::RmsNorm { .. } => OpKind::RmsNorm,
            Op::ResizeNearest2x => OpKind::ResizeNearest2x,
            Op::Interpolate3d { .. } => OpKind::Interpolate3d,
            Op::Attention { .. } => OpKind::Attention,
            Op::Rope { .. } => OpKind::Rope,
            Op::AxialRope2d { .. } => OpKind::AxialRope2d,
            Op::Reshape { .. } => OpKind::Reshape,
            Op::Transpose { .. } => OpKind::Transpose,
            Op::Narrow { .. } => OpKind::Narrow,
            Op::Concat { .. } => OpKind::Concat,
            Op::Expand { .. } => OpKind::Expand,
            Op::Gather { .. } => OpKind::Gather,
            Op::Reverse { .. } => OpKind::Reverse,
            Op::Pad { .. } => OpKind::Pad,
            Op::Slice { .. } => OpKind::Slice,
            Op::Clamp { .. } => OpKind::Clamp,
            Op::Tile { .. } => OpKind::Tile,
            Op::Trilu { .. } => OpKind::Trilu,
            Op::Reduce { .. } => OpKind::Reduce,
            Op::Softmax { .. } => OpKind::Softmax,
            Op::Cumsum { .. } => OpKind::Cumsum,
            Op::CumProd { .. } => OpKind::CumProd,
            Op::CumMax { .. } => OpKind::CumMax,
            Op::ArgMax { .. } => OpKind::ArgMax,
            Op::ArgMin { .. } => OpKind::ArgMin,
            Op::TopK { .. } => OpKind::TopK,
            Op::Sample { .. } => OpKind::Sample,
            Op::RngNormal { .. } => OpKind::RngNormal,
            Op::RngUniform { .. } => OpKind::RngUniform,
            Op::Conv { .. } => OpKind::Conv,
            Op::Im2Col { .. } => OpKind::Im2Col,
            Op::ConvTranspose2d { .. } => OpKind::ConvTranspose2d,
            Op::Conv3d { .. } => OpKind::Conv3d,
            Op::ConvTranspose3d { .. } => OpKind::ConvTranspose3d,
            Op::Pool { .. } => OpKind::Pool,
            Op::ReluBackward => OpKind::ReluBackward,
            Op::ActivationBackward { .. } => OpKind::ActivationBackward,
            Op::FakeQuantizeBackward { .. } => OpKind::FakeQuantizeBackward,
            Op::ComplexNormSq => OpKind::ComplexNormSq,
            Op::ComplexNormSqBackward => OpKind::ComplexNormSqBackward,
            Op::Conjugate => OpKind::Conjugate,
            Op::LayerNormBackwardInput { .. } => OpKind::LayerNormBackwardInput,
            Op::LayerNormBackwardGamma { .. } => OpKind::LayerNormBackwardGamma,
            Op::RmsNormBackwardInput { .. } => OpKind::RmsNormBackwardInput,
            Op::RmsNormBackwardGamma { .. } => OpKind::RmsNormBackwardGamma,
            Op::RmsNormBackwardBeta { .. } => OpKind::RmsNormBackwardBeta,
            Op::RopeBackward { .. } => OpKind::RopeBackward,
            Op::GroupNormBackwardInput { .. } => OpKind::GroupNormBackwardInput,
            Op::GroupNormBackwardGamma { .. } => OpKind::GroupNormBackwardGamma,
            Op::GroupNormBackwardBeta { .. } => OpKind::GroupNormBackwardBeta,
            Op::BatchNormInferenceBackwardInput { .. } => OpKind::BatchNormInferenceBackwardInput,
            Op::BatchNormInferenceBackwardGamma { .. } => OpKind::BatchNormInferenceBackwardGamma,
            Op::BatchNormInferenceBackwardBeta => OpKind::BatchNormInferenceBackwardBeta,
            Op::CumsumBackward { .. } => OpKind::CumsumBackward,
            Op::GatherBackward { .. } => OpKind::GatherBackward,
            Op::MaxPool2dBackward { .. } => OpKind::MaxPool2dBackward,
            Op::Conv2dBackwardInput { .. } => OpKind::Conv2dBackwardInput,
            Op::Conv2dBackwardWeight { .. } => OpKind::Conv2dBackwardWeight,
            Op::MaxPool3dBackward { .. } => OpKind::MaxPool3dBackward,
            Op::Conv3dBackwardInput { .. } => OpKind::Conv3dBackwardInput,
            Op::Conv3dBackwardWeight { .. } => OpKind::Conv3dBackwardWeight,
            Op::SoftmaxCrossEntropy => OpKind::SoftmaxCrossEntropy,
            Op::SoftmaxCrossEntropyWithLogits => OpKind::SoftmaxCrossEntropyWithLogits,
            Op::SoftmaxCrossEntropyBackward => OpKind::SoftmaxCrossEntropyBackward,
            Op::AttentionBackward { .. } => OpKind::AttentionBackward,
            Op::GroupedMatMul => OpKind::GroupedMatMul,
            Op::DequantGroupedMatMul { .. } => OpKind::DequantGroupedMatMul,
            Op::DequantGroupedMatMulMlx { .. } => OpKind::DequantGroupedMatMulMlx,
            Op::DequantMoEWeights { .. } => OpKind::DequantMoEWeights,
            Op::ScatterAdd => OpKind::ScatterAdd,
            Op::ScatterNd { .. } => OpKind::ScatterNd,
            Op::ScatterElements { .. } => OpKind::ScatterElements,
            Op::GatherNd { .. } => OpKind::GatherNd,
            Op::GatherElements { .. } => OpKind::GatherElements,
            Op::LoraMatMul { .. } => OpKind::LoraMatMul,
            Op::PartitionedConv { .. } => OpKind::PartitionedConv,
            Op::DequantMatMul { .. } => OpKind::DequantMatMul,
            Op::QMatMul { .. } => OpKind::QMatMul,
            Op::QConv2d { .. } => OpKind::QConv2d,
            Op::ScaledMatMul { .. } => OpKind::ScaledMatMul,
            Op::ScaledQuantize { .. } => OpKind::ScaledQuantize,
            Op::ScaledQuantScale { .. } => OpKind::ScaledQuantScale,
            Op::ScaledDequantize { .. } => OpKind::ScaledDequantize,
            Op::SelectiveScan { .. } => OpKind::SelectiveScan,
            Op::GatedDeltaNet { .. } => OpKind::GatedDeltaNet,
            Op::Lstm { .. } => OpKind::Lstm,
            Op::Gru { .. } => OpKind::Gru,
            Op::Rnn { .. } => OpKind::Rnn,
            Op::Mamba2 { .. } => OpKind::Mamba2,
            Op::FusedSwiGLU { .. } => OpKind::FusedSwiGLU,
            Op::FusedMatMulBiasAct { .. } => OpKind::FusedMatMulBiasAct,
            Op::FusedConvBiasAct { .. } => OpKind::FusedConvBiasAct,
            Op::FusedResidualLN { .. } => OpKind::FusedResidualLN,
            Op::FusedResidualRmsNorm { .. } => OpKind::FusedResidualRmsNorm,
            Op::FusedAttentionBlock { .. } => OpKind::FusedAttentionBlock,
            Op::FusedTransformerLayer { .. } => OpKind::FusedTransformerLayer,
            Op::If { .. } => OpKind::If,
            Op::While { .. } => OpKind::While,
            Op::Scan { .. } => OpKind::Scan,
            Op::ScanBackward { .. } => OpKind::ScanBackward,
            Op::ScanBackwardXs { .. } => OpKind::ScanBackwardXs,
            Op::GaussianSplatRender { .. } => OpKind::GaussianSplatRender,
            Op::GaussianSplatRenderBackward { .. } => OpKind::GaussianSplatRenderBackward,
            Op::GaussianSplatPrepare { .. } => OpKind::GaussianSplatPrepare,
            Op::GaussianSplatRasterize { .. } => OpKind::GaussianSplatRasterize,
            Op::Custom { .. } => OpKind::Custom,
            Op::CustomFn { .. } => OpKind::CustomFn,
            Op::Fft { .. } => OpKind::Fft,
            Op::FftButterflyStage { .. } => OpKind::FftButterflyStage,
            Op::LogMel => OpKind::LogMel,
            Op::LogMelBackward => OpKind::LogMelBackward,
            Op::WelchPeaks { .. } => OpKind::WelchPeaks,
            Op::BiMap => OpKind::BiMap,
            Op::ReEig { .. } => OpKind::ReEig,
            Op::LogEig { .. } => OpKind::LogEig,
            Op::SpdBatchNorm { .. } => OpKind::SpdBatchNorm,
            Op::SpdKarcherMean { .. } => OpKind::SpdKarcherMean,
            Op::SpdKarcherMeanWeighted { .. } => OpKind::SpdKarcherMeanWeighted,
            Op::SpdLogMap => OpKind::SpdLogMap,
            Op::SpdExpMap => OpKind::SpdExpMap,
            Op::SpdParallelTransport => OpKind::SpdParallelTransport,
            Op::SpdMatrixFnBatch { .. } => OpKind::SpdMatrixFnBatch,
            Op::ReEigBackward { .. } => OpKind::ReEigBackward,
            Op::LogEigBackward { .. } => OpKind::LogEigBackward,
            Op::SpdBatchNormBackwardX { .. } => OpKind::SpdBatchNormBackwardX,
            Op::SpdBatchNormBackwardG { .. } => OpKind::SpdBatchNormBackwardG,
            Op::SpdLogMapBackward => OpKind::SpdLogMapBackward,
            Op::SpdExpMapBackward => OpKind::SpdExpMapBackward,
            Op::SpdParallelTransportBackward => OpKind::SpdParallelTransportBackward,
            Op::SpdMatrixFnBatchBackward { .. } => OpKind::SpdMatrixFnBatchBackward,
            Op::Eigh => OpKind::Eigh,
            Op::EighBackward => OpKind::EighBackward,
            Op::EighBatch => OpKind::EighBatch,
            Op::EighBatchBackward => OpKind::EighBatchBackward,
            Op::AdaLayerNorm { .. } => OpKind::AdaLayerNorm,
            Op::AdaLayerNormBackward { .. } => OpKind::AdaLayerNormBackward,
            Op::GatedResidual => OpKind::GatedResidual,
            Op::GatedResidualBackward => OpKind::GatedResidualBackward,
        }
    }

    /// True if this op is element-wise (same shape in, same shape out).
    /// Element-wise ops are prime fusion candidates.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Activation(_)
                | Op::Cast { .. }
                | Op::StopGradient
                | Op::Binary(_)
                | Op::Compare(_)
                | Op::Where
                | Op::Fma
                | Op::ElementwiseRegion { .. }
                | Op::BatchElementwiseRegion { .. }
        )
    }

    /// True if this op may appear in a [`Op::TransformRegion`] chain.
    pub fn is_transform_eligible(&self) -> bool {
        matches!(self, Op::ResizeNearest2x)
    }

    /// True if this op is a BLAS/compute-intensive op that forms a fusion boundary.
    pub fn is_blas(&self) -> bool {
        matches!(
            self,
            Op::MatMul
                | Op::DotGeneral { .. }
                | Op::DenseSolve
                | Op::BatchedDenseSolve
                | Op::Conv { .. }
                | Op::Im2Col { .. }
                | Op::ConvTranspose2d { .. }
                | Op::Conv3d { .. }
                | Op::ConvTranspose3d { .. }
                | Op::FusedMatMulBiasAct { .. }
                | Op::FusedConvBiasAct { .. }
                | Op::GroupedMatMul
                | Op::DequantGroupedMatMul { .. }
                | Op::DequantGroupedMatMulMlx { .. }
                | Op::DequantMoEWeights { .. }
                | Op::LoraMatMul { .. }
                | Op::DequantMatMul { .. }
                | Op::QMatMul { .. }
                | Op::QConv2d { .. }
                | Op::ScaledMatMul { .. }
        )
    }

    /// True if element-wise fusion must not span across this op.
    pub fn is_fusion_boundary(&self) -> bool {
        self.is_blas()
            || matches!(
                self,
                Op::GaussianSplatRender { .. }
                    | Op::GaussianSplatRenderBackward { .. }
                    | Op::GaussianSplatPrepare { .. }
                    | Op::GaussianSplatRasterize { .. }
            )
    }

    /// True if this op is a reduction (drives loop iteration in fused kernels).
    pub fn is_reduction(&self) -> bool {
        matches!(
            self,
            Op::Reduce { .. } | Op::Softmax { .. } | Op::TopK { .. }
        )
    }

    /// Number of tensor inputs this op expects.
    pub fn num_inputs(&self) -> usize {
        match self {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => 0,
            Op::Activation(_)
            | Op::Cast { .. }
            | Op::StopGradient
            | Op::Reshape { .. }
            | Op::Quantize { .. }
            | Op::Dequantize { .. }
            | Op::Transpose { .. }
            | Op::Narrow { .. }
            | Op::Reverse { .. }
            | Op::Pad { .. }
            | Op::Slice { .. }
            | Op::Clamp { .. }
            | Op::Tile { .. }
            | Op::Trilu { .. }
            | Op::Expand { .. }
            | Op::Reduce { .. }
            | Op::Softmax { .. }
            | Op::FusedSwiGLU { .. }
            | Op::TopK { .. }
            | Op::Cumsum { .. }
            | Op::CumProd { .. }
            | Op::CumMax { .. }
            | Op::ArgMax { .. }
            | Op::ArgMin { .. }
            | Op::Sample { .. }
            | Op::ResizeNearest2x
            | Op::Interpolate3d { .. } => 1,
            Op::RngNormal { .. } | Op::RngUniform { .. } => 0, // 0 or 1 — see verify
            // EMA / Fixed scale modes carry a state tensor as a 2nd input;
            // PerBatch (default) doesn't need one.
            Op::FakeQuantize { scale_mode, .. } => match scale_mode {
                ScaleMode::PerBatch => 1,
                ScaleMode::EMA { .. } | ScaleMode::Fixed => 2,
            },
            Op::FakeQuantizeLSQ { .. } => 2, // x, scale (learned param)
            Op::FakeQuantizeLSQBackwardX { .. } | Op::FakeQuantizeLSQBackwardScale { .. } => 3, // x, scale, dy
            Op::Binary(_) | Op::Compare(_) | Op::Gather { .. } | Op::MatMul | Op::ScatterAdd => 2,
            Op::GatherNd { .. } | Op::GatherElements { .. } => 2,
            Op::ScatterNd { .. } | Op::ScatterElements { .. } => 3, // data, indices, updates
            Op::GroupedMatMul => 3,                                 // input, weight, expert_idx
            Op::DequantGroupedMatMul { .. } => 3,                   // input, packed_w, expert_idx
            Op::DequantGroupedMatMulMlx { .. } => 5, // input, w, scales, biases, expert_idx
            Op::DequantMoEWeights { .. } => 1,       // packed_w
            Op::LoraMatMul { .. } => 4,              // x, w, a, b
            Op::PartitionedConv { .. } => 2,         // x, ir
            // x, w_q, scale, zp — or x, packed_w_bytes for GGUF
            // schemes (their scales/mins live inside the packed bytes,
            // see `QuantScheme::is_gguf`).
            Op::DequantMatMul { scheme } => {
                if scheme.is_gguf() {
                    2
                } else if scheme.mxfp4x2_config().is_some() {
                    3 // x, w_q (packed [plane0|plane1]), scale ([s0|s1])
                } else {
                    4
                }
            }
            Op::QMatMul { .. } => 3, // x, w, bias
            Op::QConv2d { .. } => 3, // x, w, bias
            // lhs_codes, rhs_codes, lhs_scale, rhs_scale (+ bias)
            Op::ScaledMatMul { has_bias, .. } => 4 + usize::from(*has_bias),
            Op::ScaledQuantize { .. } => 2,   // x, scale
            Op::ScaledQuantScale { .. } => 1, // x
            Op::ScaledDequantize { .. } => 2, // codes, scale
            Op::SelectiveScan { .. } => 5,    // x, delta, a, b, c
            Op::GatedDeltaNet { carry_state, .. } if *carry_state => 6, // + state in/out
            Op::GatedDeltaNet { .. } => 5,    // q, k, v, g, beta
            Op::Lstm { carry, .. } => {
                if *carry { 6 } else { 4 } // x, w_ih, w_hh, bias (+ h0, c0)
            }
            Op::Gru { carry, .. } => {
                if *carry { 6 } else { 5 } // x, w_ih, w_hh, b_ih, b_hh (+ h0)
            }
            Op::Rnn { carry, .. } => {
                if *carry { 5 } else { 4 } // x, w_ih, w_hh, bias (+ h0)
            }
            Op::Mamba2 { .. } => 5, // x, dt, a, b, c
            Op::Where => 3,         // cond, on_true, on_false
            Op::Fma => 3,           // a, b, c  (a*b + c)
            Op::Attention { mask_kind, .. } => match mask_kind {
                MaskKind::Custom | MaskKind::Bias => 4, // Q, K, V, mask
                _ => 3,                                 // Q, K, V (mask synthesized in-kernel)
            },
            Op::AttentionBackward { mask_kind, .. } => match mask_kind {
                MaskKind::Custom | MaskKind::Bias => 5, // q, k, v, dy, mask
                _ => 4,                                 // q, k, v, dy
            },
            Op::Rope { .. } => 3, // x, cos, sin
            Op::AxialRope2d { .. } => 1,
            Op::LayerNorm { .. }
            | Op::LayerNorm2d { .. }
            | Op::GroupNorm { .. }
            | Op::RmsNorm { .. } => 3, // input, gamma, beta
            Op::BatchNormInference { .. } => 5, // x, gamma, beta, mean, var
            Op::FusedMatMulBiasAct { .. } => 3, // input, weight, bias
            Op::FusedConvBiasAct {
                has_residual: true, ..
            } => 4, // + residual (z)
            Op::FusedConvBiasAct { .. } => 3,   // input, weight, bias
            Op::FusedResidualLN { has_bias: true, .. } => 5, // x, residual, bias, gamma, beta
            Op::FusedResidualLN {
                has_bias: false, ..
            } => 4, // x, residual, gamma, beta
            Op::FusedResidualRmsNorm { has_bias: true, .. } => 5, // x, residual, bias, gamma, beta
            Op::FusedResidualRmsNorm {
                has_bias: false, ..
            } => 4, // x, residual, gamma, beta
            Op::Conv { .. } | Op::ConvTranspose2d { .. } => 2, // input, weight (bias via Add)
            Op::Conv3d { .. } | Op::ConvTranspose3d { .. } => 2, // input, weight (bias via Add)
            Op::Im2Col { .. } => 1,
            Op::Pool { .. } => 1,
            Op::ReluBackward => 2,                  // x, dy
            Op::ActivationBackward { .. } => 2,     // x, dy
            Op::FakeQuantizeBackward { .. } => 2,   // x, dy
            Op::ComplexNormSq => 1,                 // z (C64)
            Op::ComplexNormSqBackward => 2,         // z, g
            Op::Conjugate => 1,                     // z (C64)
            Op::LayerNormBackwardInput { .. } => 3, // x, gamma, dy
            Op::LayerNormBackwardGamma { .. } => 2, // x, dy
            Op::RmsNormBackwardInput { .. } => 4,   // x, gamma, beta, dy
            Op::RmsNormBackwardGamma { .. } => 4,
            Op::RmsNormBackwardBeta { .. } => 4,
            Op::RopeBackward { .. } => 3,           // dy, cos, sin
            Op::GroupNormBackwardInput { .. } => 4, // x, gamma, beta, dy
            Op::GroupNormBackwardGamma { .. } => 2, // x, dy
            Op::GroupNormBackwardBeta { .. } => 2,
            Op::BatchNormInferenceBackwardInput { .. } => 5, // x, gamma, mean, var, dy
            Op::BatchNormInferenceBackwardGamma { .. } => 4, // x, mean, var, dy
            Op::BatchNormInferenceBackwardBeta => 1,         // dy
            Op::CumsumBackward { .. } => 1,                  // dy
            Op::GatherBackward { .. } => 2,                  // dy, indices
            Op::MaxPool2dBackward { .. } => 2,               // x, dy
            Op::Conv2dBackwardInput { .. } => 2,             // dy, w
            Op::Conv2dBackwardWeight { .. } => 2,            // x, dy
            Op::MaxPool3dBackward { .. } => 2,               // x, dy
            Op::Conv3dBackwardInput { .. } => 2,             // dy, w
            Op::Conv3dBackwardWeight { .. } => 2,            // x, dy
            Op::SoftmaxCrossEntropy => 2,                    // logits, targets
            Op::SoftmaxCrossEntropyWithLogits => 2,          // logits, labels
            Op::SoftmaxCrossEntropyBackward => 3,            // logits, labels, d_loss
            Op::Concat { .. } => 0,                          // variadic — checked at graph level
            Op::DotGeneral { .. } => 2,
            Op::DenseSolve => 2,             // A, b
            Op::BatchedDenseSolve => 2,      // A [B,N,N], b [B,N] or [B,N,K]
            Op::Cholesky => 1,               // A (SPD)
            Op::TriangularSolve { .. } => 2, // A (triangular), B
            Op::Det | Op::LogDet => 1,       // A
            Op::Sort { .. } | Op::ArgSort { .. } => 1,
            Op::Svd { .. } => 1,
            Op::Qr { .. } => 1,
            Op::FusedAttentionBlock {
                has_bias, has_rope, ..
            } => 4 + if *has_bias { 2 } else { 0 } + if *has_rope { 2 } else { 0 },
            Op::If { .. } => 1,    // predicate (captures handled separately)
            Op::While { .. } => 0, // variadic loop-carried; checked at graph level
            Op::Scan {
                num_bcast, num_xs, ..
            } => 1 + *num_bcast as usize + *num_xs as usize,
            Op::ScanBackward { num_xs, .. } => 3 + *num_xs as usize, // init, trajectory, upstream, xs_0..
            Op::ScanBackwardXs { num_xs, .. } => 3 + *num_xs as usize, // same as ScanBackward
            Op::GaussianSplatRender { .. } => 7,
            Op::GaussianSplatRenderBackward { .. } => 8,
            Op::GaussianSplatPrepare { .. } => 7,
            Op::GaussianSplatRasterize { .. } => 2,
            Op::FusedTransformerLayer { has_bias, .. } => {
                // hidden + qkv_w + out_w + ln1_g + ln1_b + fc1_w + fc2_w + ln2_g + ln2_b + mask = 10
                // bias variant adds: qkv_b + out_b + fc1_b + fc2_b = 4 more
                10 + if *has_bias { 4 } else { 0 }
            }
            Op::ElementwiseRegion { num_inputs, .. } => *num_inputs as usize,
            Op::TransformRegion { num_inputs, .. } => *num_inputs as usize,
            Op::BatchElementwiseRegion {
                num_batch_inputs, ..
            } => *num_batch_inputs as usize,
            Op::Custom { num_inputs, .. } => *num_inputs as usize,
            Op::CustomFn { num_inputs, .. } => *num_inputs as usize,
            Op::Fft { .. } => 1,
            Op::FftButterflyStage { .. } => 5,
            Op::LogMel => 2,
            Op::LogMelBackward => 3,
            Op::WelchPeaks { .. } => 1,
            // ── Riemannian / SPD-manifold layers ─────────────────
            Op::BiMap => 2,                           // W, X
            Op::ReEig { .. } => 1,                    // X
            Op::LogEig { .. } => 1,                   // X
            Op::SpdBatchNorm { .. } => 3,             // X, mean, G
            Op::SpdKarcherMean { .. } => 1,           // X
            Op::SpdKarcherMeanWeighted { .. } => 2,   // X, weights
            Op::SpdLogMap => 2,                       // base, X
            Op::SpdExpMap => 2,                       // base, V
            Op::SpdParallelTransport => 3,            // from, to, V
            Op::SpdMatrixFnBatch { .. } => 1,         // X
            Op::ReEigBackward { .. } => 3,            // λ, U, dY  (reuses fwd eigendecomp)
            Op::LogEigBackward { .. } => 3,           // λ, U, dY
            Op::SpdBatchNormBackwardX { .. } => 3,    // mean, G, dY
            Op::SpdBatchNormBackwardG { .. } => 4,    // X, mean, G, dY
            Op::SpdLogMapBackward => 3,               // base, x, dY
            Op::SpdExpMapBackward => 3,               // base, v, dY
            Op::SpdParallelTransportBackward => 4,    // from, to, v, dY
            Op::SpdMatrixFnBatchBackward { .. } => 2, // X, dY
            Op::Eigh => 1,                            // A
            Op::EighBackward => 2,                    // fwd_packed, bar
            Op::EighBatch => 1,                       // A
            Op::EighBatchBackward => 2,               // fwd_packed, bar
            Op::AdaLayerNorm { .. } => 3,             // x, scale, shift
            Op::AdaLayerNormBackward { .. } => 4,     // x, scale, shift, dy
            Op::GatedResidual => 3,                   // x, y, gate
            Op::GatedResidualBackward => 4,           // x, y, gate, dy
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Input { name } => write!(f, "input(\"{name}\")"),
            Op::Param { name } => write!(f, "param(\"{name}\")"),
            Op::Constant { data } => write!(f, "const({}B)", data.len()),
            Op::Activation(a) => write!(f, "{a:?}"),
            Op::Quantize { axis, scales, .. } => match axis {
                None => write!(f, "quantize(s={})", scales[0]),
                Some(d) => write!(f, "quantize(axis={d},nch={})", scales.len()),
            },
            Op::Dequantize { axis, scales, .. } => match axis {
                None => write!(f, "dequantize(s={})", scales[0]),
                Some(d) => write!(f, "dequantize(axis={d},nch={})", scales.len()),
            },
            Op::FakeQuantize {
                bits,
                axis,
                ste,
                scale_mode,
            } => match axis {
                None => write!(
                    f,
                    "fake_quant(bits={bits},ste={ste:?},scale={scale_mode:?})"
                ),
                Some(d) => write!(
                    f,
                    "fake_quant(bits={bits},axis={d},ste={ste:?},scale={scale_mode:?})"
                ),
            },
            Op::FakeQuantizeLSQ { bits, axis } => match axis {
                None => write!(f, "fake_quant_lsq(bits={bits})"),
                Some(d) => write!(f, "fake_quant_lsq(bits={bits},axis={d})"),
            },
            Op::FakeQuantizeLSQBackwardX { bits, .. } => {
                write!(f, "fake_quant_lsq_bwd_x(bits={bits})")
            }
            Op::FakeQuantizeLSQBackwardScale { bits, .. } => {
                write!(f, "fake_quant_lsq_bwd_s(bits={bits})")
            }
            Op::Cast { to } => write!(f, "cast({to})"),
            Op::StopGradient => write!(f, "stop_gradient"),
            Op::Binary(op) => write!(f, "{op:?}"),
            Op::Compare(op) => write!(f, "{op:?}"),
            Op::Where => write!(f, "where"),
            Op::Fma => write!(f, "fma"),
            Op::MatMul => write!(f, "matmul"),
            Op::DotGeneral { .. } => write!(f, "dot_general"),
            Op::DenseSolve => write!(f, "dense_solve"),
            Op::BatchedDenseSolve => write!(f, "batched_dense_solve"),
            Op::Cholesky => write!(f, "cholesky"),
            Op::TriangularSolve { lower, transpose } => {
                write!(f, "trisolve(lower={lower},trans={transpose})")
            }
            Op::Det => write!(f, "det"),
            Op::LogDet => write!(f, "logdet"),
            Op::Sort { axis, descending } => write!(f, "sort(axis={axis},desc={descending})"),
            Op::ArgSort { axis, descending } => {
                write!(f, "argsort(axis={axis},desc={descending})")
            }
            Op::Svd { part } => write!(f, "svd({part:?})"),
            Op::Qr { part } => write!(f, "qr({part:?})"),
            Op::LayerNorm { eps, .. } => write!(f, "layer_norm(eps={eps})"),
            Op::GroupNorm { num_groups, eps } => {
                write!(f, "group_norm(groups={num_groups},eps={eps})")
            }
            Op::BatchNormInference { eps } => write!(f, "batch_norm_inference(eps={eps})"),
            Op::ResizeNearest2x => write!(f, "resize_nearest_2x"),
            Op::Interpolate3d { size } => write!(
                f,
                "interpolate3d(size=[{},{},{}])",
                size.first().copied().unwrap_or(0),
                size.get(1).copied().unwrap_or(0),
                size.get(2).copied().unwrap_or(0)
            ),
            Op::RmsNorm { eps, .. } => write!(f, "rms_norm(eps={eps})"),
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => {
                let mut s = match mask_kind {
                    MaskKind::Custom => format!("attention(h={num_heads},d={head_dim})"),
                    MaskKind::None => format!("attention(h={num_heads},d={head_dim},nomask)"),
                    MaskKind::Causal => format!("attention(h={num_heads},d={head_dim},causal)"),
                    MaskKind::SlidingWindow(w) => {
                        format!("attention(h={num_heads},d={head_dim},sw={w})")
                    }
                    MaskKind::Bias => format!("attention(h={num_heads},d={head_dim},bias)"),
                };
                if let Some(sc) = score_scale {
                    s.push_str(&format!(",scale={sc}"));
                }
                if let Some(cap) = attn_logit_softcap {
                    s.push_str(&format!(",softcap={cap}"));
                }
                write!(f, "{s}")
            }
            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => write!(f, "rope(d={head_dim}, n_rot={n_rot}, style={style:?})"),
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => write!(
                f,
                "axial_rope2d({end_x}x{end_y},h={num_heads},d={head_dim},θ={theta},r={repeat_factor})"
            ),
            Op::Reshape { new_shape } => write!(f, "reshape({new_shape:?})"),
            Op::Transpose { perm } => write!(f, "transpose({perm:?})"),
            Op::Narrow { axis, start, len } => write!(f, "narrow({axis},{start},{len})"),
            Op::Slice {
                axis,
                start,
                len,
                step,
            } => {
                write!(f, "slice({axis},{start},{len},step={step})")
            }
            Op::Clamp { min, max } => write!(f, "clamp({min},{max})"),
            Op::Tile { reps } => write!(f, "tile({reps:?})"),
            Op::Trilu { upper, diagonal } => write!(f, "trilu(upper={upper},diag={diagonal})"),
            Op::Reverse { axes } => write!(f, "reverse({axes:?})"),
            Op::Pad { pads, mode } => write!(f, "pad({pads:?},{mode:?})"),
            Op::Concat { axis } => write!(f, "concat(axis={axis})"),
            Op::Expand { .. } => write!(f, "expand"),
            Op::Gather { axis } => write!(f, "gather(axis={axis})"),
            Op::Reduce { op, axes, .. } => write!(f, "reduce_{op:?}({axes:?})"),
            Op::Softmax { axis } => write!(f, "softmax(axis={axis})"),
            Op::Cumsum { axis, exclusive } => {
                if *exclusive {
                    write!(f, "cumsum(axis={axis},excl)")
                } else {
                    write!(f, "cumsum(axis={axis})")
                }
            }
            Op::CumProd { axis, exclusive } => {
                if *exclusive {
                    write!(f, "cumprod(axis={axis},excl)")
                } else {
                    write!(f, "cumprod(axis={axis})")
                }
            }
            Op::CumMax { axis, exclusive } => {
                if *exclusive {
                    write!(f, "cummax(axis={axis},excl)")
                } else {
                    write!(f, "cummax(axis={axis})")
                }
            }
            Op::ArgMax { axis, keep_dim } => write!(f, "argmax(axis={axis},keep={keep_dim})"),
            Op::ArgMin { axis, keep_dim } => write!(f, "argmin(axis={axis},keep={keep_dim})"),
            Op::Sample {
                top_k,
                top_p,
                temperature,
                ..
            } => {
                write!(f, "sample(t={temperature}")?;
                if *top_k > 0 {
                    write!(f, ",k={top_k}")?;
                }
                if *top_p < 1.0 {
                    write!(f, ",p={top_p}")?;
                }
                write!(f, ")")
            }
            Op::RngNormal {
                mean,
                scale,
                key,
                op_seed,
            } => {
                write!(f, "rng_normal({mean},{scale},key={key}")?;
                if let Some(s) = op_seed {
                    write!(f, ",seed={s}")?;
                }
                write!(f, ")")
            }
            Op::RngUniform {
                low,
                high,
                key,
                op_seed,
            } => {
                write!(f, "rng_uniform({low},{high},key={key}")?;
                if let Some(s) = op_seed {
                    write!(f, ",seed={s}")?;
                }
                write!(f, ")")
            }
            Op::TopK { k } => write!(f, "topk(k={k})"),
            Op::GroupedMatMul => write!(f, "grouped_matmul"),
            Op::DequantGroupedMatMul { scheme } => {
                write!(f, "dequant_grouped_matmul({scheme})")
            }
            Op::DequantGroupedMatMulMlx { scheme } => {
                write!(f, "dequant_grouped_matmul_mlx({scheme})")
            }
            Op::DequantMoEWeights { scheme } => write!(f, "dequant_moe_weights({scheme})"),
            Op::LoraMatMul { scale } => write!(f, "lora_matmul(scale={scale})"),
            Op::PartitionedConv { block } => write!(f, "partitioned_conv(block={block})"),
            Op::DequantMatMul { scheme } => write!(f, "dequant_matmul({scheme})"),
            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => write!(
                f,
                "q_matmul(x_zp={x_zp},w_zp={w_zp},out_zp={out_zp},mult={mult})"
            ),
            Op::QConv2d { kernel_size, .. } => write!(f, "q_conv2d({kernel_size:?})"),
            Op::ScaledMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                has_bias,
            } => write!(
                f,
                "scaled_matmul({lhs_format}×{rhs_format},{scale_layout}{})",
                if *has_bias { ",bias" } else { "" }
            ),
            Op::ScaledQuantize {
                format,
                scale_layout,
            } => write!(f, "scaled_quantize({format},{scale_layout})"),
            Op::ScaledQuantScale {
                format,
                scale_layout,
            } => write!(f, "scaled_quant_scale({format},{scale_layout})"),
            Op::ScaledDequantize {
                format,
                scale_layout,
            } => write!(f, "scaled_dequantize({format},{scale_layout})"),
            Op::SelectiveScan { state_size } => write!(f, "ssm_scan(n={state_size})"),
            Op::GatedDeltaNet {
                state_size,
                carry_state,
                gate_per_channel,
            } => {
                let gpc = if *gate_per_channel { ",gpc" } else { "" };
                if *carry_state {
                    write!(f, "gated_delta_net(n={state_size},carry{gpc})")
                } else {
                    write!(f, "gated_delta_net(n={state_size}{gpc})")
                }
            }
            Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let dir = if *bidirectional { "bi" } else { "uni" };
                let c = if *carry { ",carry" } else { "" };
                write!(f, "lstm(h={hidden_size},layers={num_layers},{dir}{c})")
            }
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let dir = if *bidirectional { "bi" } else { "uni" };
                let c = if *carry { ",carry" } else { "" };
                write!(f, "gru(h={hidden_size},layers={num_layers},{dir}{c})")
            }
            Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
                relu,
            } => {
                let dir = if *bidirectional { "bi" } else { "uni" };
                let act = if *relu { "relu" } else { "tanh" };
                let c = if *carry { ",carry" } else { "" };
                write!(f, "rnn(h={hidden_size},layers={num_layers},{dir},{act}{c})")
            }
            Op::Mamba2 {
                head_dim,
                state_size,
            } => write!(f, "mamba2(p={head_dim},n={state_size})"),
            Op::ScatterAdd => write!(f, "scatter_add"),
            Op::ScatterNd { reduction } => write!(f, "scatter_nd({reduction:?})"),
            Op::ScatterElements { axis, reduction } => {
                write!(f, "scatter_elements(axis={axis},{reduction:?})")
            }
            Op::GatherNd { batch_dims } => write!(f, "gather_nd(batch_dims={batch_dims})"),
            Op::GatherElements { axis } => write!(f, "gather_elements(axis={axis})"),
            Op::Conv { kernel_size, .. } => write!(f, "conv2d({kernel_size:?})"),
            Op::Im2Col { kernel_size, .. } => write!(f, "im2col({kernel_size:?})"),
            Op::ConvTranspose2d { kernel_size, .. } => {
                write!(f, "conv_transpose2d({kernel_size:?})")
            }
            Op::Conv3d { stride, .. } => write!(f, "conv3d(stride={stride:?})"),
            Op::ConvTranspose3d { stride, .. } => {
                write!(f, "conv_transpose3d(stride={stride:?})")
            }
            Op::LayerNorm2d { eps } => write!(f, "layer_norm2d(eps={eps})"),
            Op::Pool {
                kind, kernel_size, ..
            } => write!(f, "pool_{kind:?}({kernel_size:?})"),
            Op::ReluBackward => write!(f, "relu_backward"),
            Op::ActivationBackward { kind } => write!(f, "{kind:?}_backward"),
            Op::ComplexNormSq => write!(f, "complex_norm_sq"),
            Op::ComplexNormSqBackward => write!(f, "complex_norm_sq_backward"),
            Op::Conjugate => write!(f, "conjugate"),
            Op::FakeQuantizeBackward { bits, ste, .. } => {
                write!(f, "fake_quant_backward(bits={bits},ste={ste:?})")
            }
            Op::MaxPool2dBackward { kernel_size, .. } => {
                write!(f, "maxpool2d_backward({kernel_size:?})")
            }
            Op::Conv2dBackwardInput { kernel_size, .. } => {
                write!(f, "conv2d_backward_input({kernel_size:?})")
            }
            Op::Conv2dBackwardWeight { kernel_size, .. } => {
                write!(f, "conv2d_backward_weight({kernel_size:?})")
            }
            Op::MaxPool3dBackward { kernel_size, .. } => {
                write!(f, "maxpool3d_backward({kernel_size:?})")
            }
            Op::Conv3dBackwardInput { kernel_size, .. } => {
                write!(f, "conv3d_backward_input({kernel_size:?})")
            }
            Op::Conv3dBackwardWeight { kernel_size, .. } => {
                write!(f, "conv3d_backward_weight({kernel_size:?})")
            }
            Op::SoftmaxCrossEntropy => write!(f, "sce"),
            Op::SoftmaxCrossEntropyWithLogits => write!(f, "sce_with_logits"),
            Op::SoftmaxCrossEntropyBackward => write!(f, "sce_backward"),
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => match mask_kind {
                MaskKind::None => write!(f, "attn_bwd_{wrt:?}(h={num_heads},d={head_dim},nomask)"),
                MaskKind::Causal => {
                    write!(f, "attn_bwd_{wrt:?}(h={num_heads},d={head_dim},causal)")
                }
                MaskKind::SlidingWindow(w) => {
                    write!(f, "attn_bwd_{wrt:?}(h={num_heads},d={head_dim},sw={w})")
                }
                MaskKind::Custom => {
                    write!(f, "attn_bwd_{wrt:?}(h={num_heads},d={head_dim},custom)")
                }
                MaskKind::Bias => write!(f, "attn_bwd_{wrt:?}(h={num_heads},d={head_dim},bias)"),
            },
            Op::FusedMatMulBiasAct { activation } => {
                write!(f, "fused_mm_bias")?;
                if let Some(a) = activation {
                    write!(f, "_{a:?}")?;
                }
                Ok(())
            }
            Op::FusedConvBiasAct {
                activation,
                has_residual,
                ..
            } => {
                write!(f, "fused_conv_bias")?;
                if *has_residual {
                    write!(f, "_res")?;
                }
                if let Some(a) = activation {
                    write!(f, "_{a:?}")?;
                }
                Ok(())
            }
            Op::FusedResidualLN { has_bias, eps } => {
                write!(f, "fused_residual")?;
                if *has_bias {
                    write!(f, "_bias")?;
                }
                write!(f, "_ln(eps={eps})")
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                write!(f, "fused_residual")?;
                if *has_bias {
                    write!(f, "_bias")?;
                }
                write!(f, "_rms(eps={eps})")
            }
            Op::FusedSwiGLU {
                cast_to,
                gate_first,
            } => {
                let mut s = match cast_to {
                    Some(dt) => format!("fused_swiglu(cast={dt}"),
                    None => "fused_swiglu(".to_string(),
                };
                if *gate_first {
                    s.push_str(",gate_first");
                }
                s.push(')');
                write!(f, "{s}")
            }
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => {
                write!(f, "fused_attn(h={num_heads},d={head_dim}")?;
                if *has_bias {
                    write!(f, ",bias")?;
                }
                if *has_rope {
                    write!(f, ",rope")?;
                }
                write!(f, ")")
            }
            Op::If { .. } => write!(f, "if(...)"),
            Op::While { max_iterations, .. } => match max_iterations {
                Some(n) => write!(f, "while(...max={n})"),
                None => write!(f, "while(...)"),
            },
            Op::Scan {
                length,
                save_trajectory,
                num_xs,
                ..
            } => {
                let traj = if *save_trajectory { ",traj" } else { "" };
                let xs = if *num_xs > 0 {
                    format!(",xs={}", num_xs)
                } else {
                    String::new()
                };
                write!(f, "scan(len={length}{xs}{traj})")
            }
            Op::ScanBackward {
                length,
                save_trajectory,
                num_xs,
                ..
            } => {
                let traj = if *save_trajectory { ",traj" } else { "" };
                let xs = if *num_xs > 0 {
                    format!(",xs={}", num_xs)
                } else {
                    String::new()
                };
                write!(f, "scan_bwd(len={length}{xs}{traj})")
            }
            Op::ScanBackwardXs {
                length,
                save_trajectory,
                num_xs,
                xs_idx,
                ..
            } => {
                let traj = if *save_trajectory { ",traj" } else { "" };
                write!(
                    f,
                    "scan_bwd_xs(len={length},xs={num_xs},idx={xs_idx}{traj})"
                )
            }
            Op::FusedTransformerLayer {
                num_heads,
                head_dim,
                intermediate_size,
                has_bias,
                ..
            } => {
                write!(
                    f,
                    "fused_layer(h={num_heads},d={head_dim},int={intermediate_size}"
                )?;
                if *has_bias {
                    write!(f, ",bias")?;
                }
                write!(f, ")")
            }
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus: _,
                prologue,
                prologue_input: _,
            } => {
                let pro = match prologue {
                    RegionPrologue::None => "",
                    RegionPrologue::ResizeNearest2x => ",prologue=resize2x",
                };
                if *scalar_input_mask != 0 {
                    write!(
                        f,
                        "ew_region(in={num_inputs},steps={},scalar_mask=0x{:x}{pro})",
                        chain.len(),
                        scalar_input_mask
                    )
                } else {
                    write!(f, "ew_region(in={num_inputs},steps={}{pro})", chain.len())
                }
            }
            Op::TransformRegion { steps, num_inputs } => {
                write!(f, "transform_region(in={num_inputs},steps={})", steps.len())
            }
            Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask,
                prologue,
                ..
            } => write!(
                f,
                "batch_ew_region(batch={num_batch_inputs},steps={},mask=0x{:x},prologue={prologue:?})",
                chain.len(),
                scalar_input_mask
            ),
            Op::LayerNormBackwardInput { eps, .. } => {
                write!(f, "layer_norm_backward_input(eps={eps})")
            }
            Op::LayerNormBackwardGamma { eps, .. } => {
                write!(f, "layer_norm_backward_gamma(eps={eps})")
            }
            Op::RmsNormBackwardInput { eps, .. } => write!(f, "rms_norm_backward_input(eps={eps})"),
            Op::RmsNormBackwardGamma { eps, .. } => write!(f, "rms_norm_backward_gamma(eps={eps})"),
            Op::RmsNormBackwardBeta { eps, .. } => write!(f, "rms_norm_backward_beta(eps={eps})"),
            Op::RopeBackward { head_dim, n_rot } => {
                write!(f, "rope_backward(d={head_dim},n_rot={n_rot})")
            }
            Op::GroupNormBackwardInput { num_groups, eps } => {
                write!(f, "group_norm_backward_input(g={num_groups},eps={eps})")
            }
            Op::GroupNormBackwardGamma { num_groups, eps } => {
                write!(f, "group_norm_backward_gamma(g={num_groups},eps={eps})")
            }
            Op::GroupNormBackwardBeta { num_groups, eps } => {
                write!(f, "group_norm_backward_beta(g={num_groups},eps={eps})")
            }
            Op::BatchNormInferenceBackwardInput { eps } => {
                write!(f, "batch_norm_inference_backward_input(eps={eps})")
            }
            Op::BatchNormInferenceBackwardGamma { eps } => {
                write!(f, "batch_norm_inference_backward_gamma(eps={eps})")
            }
            Op::BatchNormInferenceBackwardBeta => {
                write!(f, "batch_norm_inference_backward_beta")
            }
            Op::CumsumBackward { axis, exclusive } => {
                write!(f, "cumsum_backward(axis={axis},exclusive={exclusive})")
            }
            Op::GatherBackward { axis } => write!(f, "gather_backward(axis={axis})"),
            Op::GaussianSplatRender {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => write!(
                f,
                "gaussian_splat_render({width}x{height},tile={tile_size},r={radius_scale},a={alpha_cutoff},steps={max_splat_steps},t={transmittance_threshold},list={max_list_entries})"
            ),
            Op::GaussianSplatRenderBackward {
                width,
                height,
                loss_grad_clip,
                sh_band,
                ..
            } => write!(
                f,
                "gaussian_splat_render_bwd({width}x{height},clip={loss_grad_clip},sh={sh_band})"
            ),
            Op::GaussianSplatPrepare {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
                ..
            } => write!(
                f,
                "gaussian_splat_prepare({width}x{height},tile={tile_size},r={radius_scale},a={alpha_cutoff},steps={max_splat_steps},t={transmittance_threshold},list={max_list_entries})"
            ),
            Op::GaussianSplatRasterize {
                width,
                height,
                tile_size,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
                ..
            } => write!(
                f,
                "gaussian_splat_rasterize({width}x{height},tile={tile_size},a={alpha_cutoff},steps={max_splat_steps},t={transmittance_threshold},list={max_list_entries})"
            ),
            Op::Custom {
                name,
                num_inputs,
                attrs,
            } => write!(f, "custom({name},in={num_inputs},attrs={}B)", attrs.len()),
            Op::CustomFn {
                num_inputs,
                vjp_body,
                jvp_body,
                ..
            } => {
                let v = if vjp_body.is_some() { ",vjp" } else { "" };
                let j = if jvp_body.is_some() { ",jvp" } else { "" };
                write!(f, "custom_fn(in={num_inputs}{v}{j})")
            }
            Op::Fft { inverse, norm } => {
                write!(f, "fft(inverse={inverse}, norm={norm:?})")
            }
            Op::FftButterflyStage { stage, n_fft } => {
                write!(f, "fft_butterfly_stage(stage={stage}, n_fft={n_fft})")
            }
            Op::LogMel => write!(f, "log_mel()"),
            Op::LogMelBackward => write!(f, "log_mel_backward()"),
            Op::WelchPeaks { k, n_segments } => {
                write!(f, "welch_peaks(k={k}, n_segments={n_segments})")
            }
            Op::BiMap => write!(f, "bimap()"),
            Op::ReEig { eps } => write!(f, "reeig(eps={eps})"),
            Op::LogEig { eps } => write!(f, "logeig(eps={eps})"),
            Op::SpdBatchNorm { eps } => write!(f, "spd_batch_norm(eps={eps})"),
            Op::SpdKarcherMean { iters, tol } => {
                write!(f, "spd_karcher_mean(iters={iters}, tol={tol})")
            }
            Op::SpdKarcherMeanWeighted { iters, tol } => {
                write!(f, "spd_karcher_mean_weighted(iters={iters}, tol={tol})")
            }
            Op::SpdLogMap => write!(f, "spd_log_map()"),
            Op::SpdExpMap => write!(f, "spd_exp_map()"),
            Op::SpdParallelTransport => write!(f, "spd_parallel_transport()"),
            Op::SpdMatrixFnBatch { kind } => write!(f, "spd_matrix_fn_batch(kind={kind:?})"),
            Op::ReEigBackward { eps } => write!(f, "reeig_backward(eps={eps})"),
            Op::LogEigBackward { eps } => write!(f, "logeig_backward(eps={eps})"),
            Op::SpdBatchNormBackwardX { eps } => {
                write!(f, "spd_batch_norm_backward_x(eps={eps})")
            }
            Op::SpdBatchNormBackwardG { eps } => {
                write!(f, "spd_batch_norm_backward_g(eps={eps})")
            }
            Op::SpdLogMapBackward => write!(f, "spd_log_map_backward()"),
            Op::SpdExpMapBackward => write!(f, "spd_exp_map_backward()"),
            Op::SpdParallelTransportBackward => write!(f, "spd_parallel_transport_backward()"),
            Op::SpdMatrixFnBatchBackward { kind } => {
                write!(f, "spd_matrix_fn_batch_backward(kind={kind:?})")
            }
            Op::Eigh => write!(f, "eigh()"),
            Op::EighBackward => write!(f, "eigh_backward()"),
            Op::EighBatch => write!(f, "eigh_batch()"),
            Op::EighBatchBackward => write!(f, "eigh_batch_backward()"),
            Op::AdaLayerNorm { norm, eps } => write!(f, "ada_layer_norm({norm:?},eps={eps})"),
            Op::AdaLayerNormBackward { norm, eps } => {
                write!(f, "ada_layer_norm_backward({norm:?},eps={eps})")
            }
            Op::GatedResidual => write!(f, "gated_residual"),
            Op::GatedResidualBackward => write!(f, "gated_residual_backward"),
        }
    }
}
