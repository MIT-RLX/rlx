// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OneApiExecutable` — compile an IR graph for the Intel oneAPI Level Zero
//! backend and execute it.
//!
//! Two execution paths share one legalized graph (the rlx-vulkan primitive set,
//! so the same rewrite/legalize decompositions apply):
//!
//! - [`run_host`](OneApiExecutable::run_host) — a value-map interpreter that
//!   evaluates every node through the `rlx-cpu` reference. This is the path the
//!   macOS dev box / CI take (no Level Zero device), and it makes the backend
//!   fully correct without Intel hardware.
//! - [`run_l0`](OneApiExecutable::run_l0) — the native path: a USM-shared f32
//!   arena + per-op SPIR-V kernel dispatch (with a CPU host-fallback, against
//!   the same arena, for ops with no native kernel yet). Selected only when a
//!   live device *and* embedded kernels are both present — neither is true off
//!   an Intel build host, so it is compiled-but-dormant here, pending hardware
//!   validation on Arc / Data Center Max.

use crate::device::oneapi_device;
use crate::host::{self, HostBuf, HostOut};
use crate::kernels::kernels;
use rlx_compile::memory::{BufferSlot, MemoryPlan};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, RngOptions, Shape};
use std::collections::HashMap;
use std::ffi::c_void;

/// OpKinds this backend lowers (claim set). Closes gaps vs CUDA/wgpu via
/// `rlx_unfuse` + CPU host-fallback; native OpenCL-C when kernels are embedded.
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        Cast,
        StopGradient,
        Reshape, // structural / alias
        Binary,
        Compare,
        Where,
        Fma,
        Activation, // elementwise
        // Region fused forms — claimed so fusion may emit them; `compile_rng`
        // runs `DecomposeFusionRegions` then `UnfuseElementwiseRegions` before
        // legalize (Transform/Batch → Resize + ElementwiseRegion → primitives).
        ElementwiseRegion,
        TransformRegion,
        BatchElementwiseRegion,
        MatMul,
        // Scaled low-p GEMM / QAT — host-fallback (same as Vulkan).
        ScaledMatMul,
        ScaledQuantize,
        ScaledQuantScale,
        ScaledDequantize,
        // Claimed; `rlx_unfuse` expands to MatMul (+ reshape) before legalize.
        DotGeneral,
        // Host via rlx-cpu LAPACK (`sgesv`) on USM / value-map — no device
        // oneMKL LAPACK linked (same HostOpDesc contract as wgpu / Vulkan).
        DenseSolve,
        BatchedDenseSolve,
        // Host via rlx-cpu LAPACK (potrf / trsm / getrf) on the mapped arena,
        // same HostOpDesc contract as DenseSolve. No device oneMKL LAPACK linked.
        Cholesky,
        TriangularSolve,
        Det,
        LogDet,
        // Sort / ArgSort host-stage to CPU (stable strided sort) on the
        // mapped arena, same HostOpDesc contract as Det / LogDet.
        Sort,
        ArgSort,
        Reduce,
        Softmax, // contraction / reduction
        LayerNorm,
        RmsNorm,
        LayerNorm2d, // normalization
        GroupNorm,
        GroupNormBackwardInput,
        GroupNormBackwardGamma,
        GroupNormBackwardBeta,
        Rope,
        Attention, // transformer
        AttentionBackward,
        // Claimed first-class; `compile_rng` runs `crate::unfuse` /
        // `unfuse_attention_block` to lower to primitives before legalization.
        FusedAttentionBlock,
        FusedResidualLN,
        FusedResidualRmsNorm,
        FusedSwiGLU,
        // Native compose: matmul + Binary(Add) + optional Activation.
        FusedMatMulBiasAct,
        // DiT modulation — claimed for fusion; `unfuse_dit_modulation`
        // expands forward Ada/Gated before host / SPIR-V lowering.
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native OpenCL-C SPIR-V when kernels are
        // embedded (`RLX_ONEAPI_BUILD_KERNELS=1`); else CPU host-fallback.
        AdaLayerNormBackward,
        GatedResidualBackward,
        SoftmaxCrossEntropy,
        SoftmaxCrossEntropyWithLogits,
        SoftmaxCrossEntropyBackward,
        // C64 Wirtinger surface — native SPIR-V when kernels embedded.
        ComplexNormSq,
        ComplexNormSqBackward,
        Conjugate,
        // QAT: Fixed/PerBatch + INT8 Quantize/Dequantize native; EMA / LSQ host.
        FakeQuantize,
        FakeQuantizeLSQ,
        FakeQuantizeLSQBackwardX,
        FakeQuantizeLSQBackwardScale,
        FakeQuantizeBackward,
        Quantize,
        Dequantize,
        // Inference BN + bwd trio — native when kernels embedded.
        BatchNormInference,
        BatchNormInferenceBackwardInput,
        BatchNormInferenceBackwardGamma,
        BatchNormInferenceBackwardBeta,
        // 3-D conv + transpose — native when kernels embedded.
        Conv3d,
        ConvTranspose3d,
        // SAM2 axial 2-D RoPE — native when kernels embedded.
        AxialRope2d,
        // Norm reverse — native OpenCL when kernels embedded (GroupNorm bwd too).
        LayerNormBackwardInput,
        LayerNormBackwardGamma,
        RmsNormBackwardInput,
        RmsNormBackwardGamma,
        RmsNormBackwardBeta,
        // Activation / RoPE / vision reverse — native when kernels embedded.
        ReluBackward,
        ActivationBackward,
        RopeBackward,
        CumsumBackward, // native `cumsum_backward.cl` when embedded
        GatherBackward, // native `gather_backward.cl` when embedded
        MaxPool2dBackward,
        Conv2dBackwardInput,
        Conv2dBackwardWeight,
        // Fused conv+bias+act — native `fused_conv_bias_act.cl` when embedded.
        FusedConvBiasAct,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Cumsum,
        CumProd, // host-eval (CPU thunk), same path as Cumsum
        CumMax,
        Reverse, // shape / indexing
        ArgMax,
        ArgMin,
        Pool,
        ResizeNearest2x,
        Conv,          // reductions / vision
        GroupedMatMul, // MoE
        SelectiveScan, // SSM / Mamba
        Im2Col,
        ScatterAdd,
        ScatterNd,
        ScatterElements,
        GatherNd,
        GatherElements,
        TopK, // vision / indexing / generation
        // RNN family — native OpenCL when hidden/state ≤ 256 (else host).
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        // Expanded to MatMul/Mul/Add… before legalize (no dedicated kernel).
        GatedDeltaNet,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad):
        // no native kernel → routed to the rlx-cpu host fallback (USM-shared arena).
        Scan,
        ScanBackward,
        ScanBackwardXs,
        ConvTranspose2d, // native `conv_transpose2d.cl` when embedded
        Fft,
        FftButterflyStage, // native `fft_butterfly_stage.cl` when embedded
        LogMel,            // packed HostOpDesc (no OpenCL LogMel; same as wgpu)
        LogMelBackward,    // packed HostOpDesc
        WelchPeaks,        // native when eligible + embedded; else HostOpDesc
        DequantMatMul,
        DequantGroupedMatMul,
        DequantMoEWeights, // GGUF quant
        QMatMul,
        QConv2d,
        RngNormal,
        RngUniform,
        Sample, // RNG / generation
        // Gaussian splat CPU reference — host-fallback.
        GaussianSplatRender,
        GaussianSplatRenderBackward,
        GaussianSplatPrepare,
        GaussianSplatRasterize,
        // Decomposed by `crate::unfuse` (`expand_lora` / `expand_ftl` /
        // `expand_if` / `expand_while`) before lowering.
        LoraMatMul,
        FusedTransformerLayer,
        If,
        While,
        // PartitionedConv expanded by `expand_cpu_nop_fused` (CPU would Nop).
        PartitionedConv,
        CustomFn,
        // Core Riemannian / SPD-manifold ops (F64): no native kernel → routed
        // to the F64-aware CPU host fallback (`crate::spd`), on both the
        // value-map (`run_host`) and USM-arena (`run_l0`) paths.
        BiMap,
        ReEig,
        LogEig,
        SpdBatchNorm,
        SpdKarcherMean,
        SpdKarcherMeanWeighted,
        SpdLogMap,
        SpdExpMap,
        SpdParallelTransport,
        SpdMatrixFnBatch,
        ReEigBackward,
        LogEigBackward,
        SpdBatchNormBackwardX,
        SpdBatchNormBackwardG,
        SpdLogMapBackward,
        SpdExpMapBackward,
        SpdParallelTransportBackward,
        SpdMatrixFnBatchBackward,
        Eigh,
        EighBackward,
        EighBatch,
        EighBatchBackward,
        // In-graph collectives (`collective.*`) — claimed so the Session/stages
        // legalize pass lets them through; `run_host` / L0 host-fallback eval
        // via rlx-cpu (same as rlx-vulkan).
        Custom,
    ]
};

/// Ops with a native OpenCL-C SPIR-V kernel under `kernels/`. Everything else
/// routes to the CPU host-fallback on the native path. GRU/LSTM/RNN/Mamba2
/// require simple geometry and a ≤256 compile-time cap (mirrors Vulkan/wgpu).
/// `FusedMatMulBiasAct` is composed from matmul+binary[+unary] in `run_l0`.
fn native_kernel(op: &Op) -> Option<&'static str> {
    use rlx_ir::op::ScaleMode;
    match op {
        Op::Binary(_) => Some("binary"),
        Op::Activation(_) => Some("unary"),
        Op::MatMul => Some("matmul"),
        Op::Softmax { .. } => Some("softmax"),
        Op::RmsNorm { .. } => Some("rmsnorm"),
        Op::AdaLayerNormBackward { .. } => Some("ada_layer_norm_backward"),
        Op::GatedResidualBackward => Some("gated_residual_backward"),
        Op::GroupNorm { .. } => Some("group_norm"),
        Op::GroupNormBackwardInput { .. } => Some("group_norm_bwd_input"),
        Op::GroupNormBackwardGamma { .. } => Some("group_norm_bwd_gamma"),
        Op::GroupNormBackwardBeta { .. } => Some("group_norm_bwd_beta"),
        Op::FusedResidualLN { .. } => Some("fused_residual_ln"),
        Op::FusedResidualRmsNorm { .. } => Some("fused_residual_rms_norm"),
        Op::FusedSwiGLU { .. } => Some("fused_swiglu"),
        Op::SoftmaxCrossEntropy => Some("softmax_cross_entropy"),
        Op::SoftmaxCrossEntropyWithLogits => Some("softmax_cross_entropy_with_logits"),
        Op::SoftmaxCrossEntropyBackward => Some("softmax_cross_entropy_backward"),
        Op::Fma => Some("fma_elem"),
        Op::ComplexNormSq => Some("complex_norm_sq"),
        Op::ComplexNormSqBackward => Some("complex_norm_sq_backward"),
        Op::Conjugate => Some("conjugate_c64"),
        Op::FakeQuantize {
            scale_mode: ScaleMode::Fixed,
            ..
        } => Some("fake_quantize_fixed"),
        Op::FakeQuantize {
            scale_mode: ScaleMode::PerBatch,
            ..
        } => Some("fake_quantize_perbatch"),
        Op::Quantize { .. } => Some("quantize_i8"),
        Op::Dequantize { .. } => Some("dequantize_i8"),
        Op::CumsumBackward { .. } => Some("cumsum_backward"),
        Op::GatherBackward { .. } => Some("gather_backward"),
        Op::BatchNormInference { .. } => Some("batch_norm_inference"),
        Op::BatchNormInferenceBackwardInput { .. } => Some("batch_norm_inference_bwd_input"),
        Op::BatchNormInferenceBackwardGamma { .. } => Some("batch_norm_inference_bwd_gamma"),
        Op::BatchNormInferenceBackwardBeta => Some("batch_norm_inference_bwd_beta"),
        Op::ReluBackward | Op::ActivationBackward { .. } => Some("activation_backward"),
        Op::AxialRope2d { .. } => Some("axial_rope2d"),
        Op::LayerNormBackwardInput { .. } => Some("layer_norm_bwd_input"),
        Op::LayerNormBackwardGamma { .. } => Some("layer_norm_bwd_gamma"),
        Op::RmsNormBackwardInput { .. } => Some("rms_norm_bwd_input"),
        Op::RmsNormBackwardGamma { .. } | Op::RmsNormBackwardBeta { .. } => {
            Some("rms_norm_bwd_param")
        }
        Op::FftButterflyStage { .. } => Some("fft_butterfly_stage"),
        Op::Conv3d { .. } => Some("conv3d"),
        Op::ConvTranspose3d { .. } => Some("conv_transpose3d"),
        Op::ConvTranspose2d { .. } => Some("conv_transpose2d"),
        Op::Conv2dBackwardInput { .. } => Some("conv2d_backward_input"),
        Op::Conv2dBackwardWeight { .. } => Some("conv2d_backward_weight"),
        Op::MaxPool2dBackward { .. } => Some("maxpool2d_backward"),
        Op::FusedConvBiasAct { .. } => Some("fused_conv_bias_act"),
        Op::RopeBackward { .. } => Some("rope_backward"),
        Op::Lstm {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
        } => {
            if *num_layers == 1
                && !*bidirectional
                && !*carry
                && *hidden_size > 0
                && *hidden_size <= 256
            {
                Some("lstm")
            } else {
                None
            }
        }
        Op::Gru {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
        } => {
            if *num_layers == 1
                && !*bidirectional
                && !*carry
                && *hidden_size > 0
                && *hidden_size <= 256
            {
                Some("gru")
            } else {
                None
            }
        }
        Op::Rnn {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
            ..
        } => {
            if *num_layers == 1
                && !*bidirectional
                && !*carry
                && *hidden_size > 0
                && *hidden_size <= 256
            {
                Some("rnn")
            } else {
                None
            }
        }
        Op::Mamba2 { state_size, .. } => {
            if *state_size > 0 && *state_size <= 256 {
                Some("mamba2")
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Cast op ids for the `unary` OpenCL kernel (`unary.cl` cases 100–106). Kept
/// in sync with rlx-cuda / rlx-rocm (unary.cu) and rlx-vulkan (unary.comp).
const CAST_F32_TO_I8: u32 = 100;
const CAST_F32_TO_I16: u32 = 101;
const CAST_F32_TO_I32: u32 = 102;
const CAST_F32_TO_I64: u32 = 103;
const CAST_F32_TO_U8: u32 = 104;
const CAST_F32_TO_U32: u32 = 105;
const CAST_TO_BOOL: u32 = 106;

/// Op ids for `activation_backward.cl` — "relu-first" scheme (matches CUDA /
/// wgpu, not the forward unary switch). Canonical table in `rlx_ir::opcodes`.
fn activation_bwd_op_id(a: Activation) -> u32 {
    a.opcode_relu_first()
}

/// How an `Op::Cast` lowers on the f32-uniform arena.
enum CastLower {
    /// Value-preserving relabel — alias the input slot (no dispatch). Covers
    /// same-dtype, int→float, float→float (F16/BF16/F64 are all f32-stored
    /// here), int→int, and bool→int/float.
    Identity,
    /// A real elementwise conversion via the `unary` kernel with this op id
    /// (float→int trunc-saturate, or →Bool `x != 0`).
    Kernel(u32),
    /// A complex cast (real↔C64, real↔C128, C64↔C128) — pure f32-lane moves via
    /// the standalone `complex_cast` kernel. Carries the mode (0..5, see
    /// `kernels/complex_cast.cl`). Needs its own (complex-sized) slot, not an
    /// alias — the lane width changes even though the element count does not.
    Complex(u32),
    /// Not representable in an f32 arena (an F64 real component has no lane
    /// storage here) — reject at lowering.
    Reject,
}

/// Classify a `Cast(src → dst)` on the f32-uniform arena. float→int truncates
/// toward zero + saturates (Rust `as` / rlx-cpu); →Bool is `x != 0`. F16/BF16/
/// F64 are demoted to f32 storage so real casts to/from them are identity
/// relabels. Complex casts (real↔C64, real↔C128, C64↔C128) are pure f32-lane
/// moves on the simulated-complex arena (C64 = 2 lanes/elem, C128 = 4 lanes
/// df64); only a complex cast touching the one non-lane-storable real component
/// (F64, demoted to a single lossy lane) rejects. Mirrors rlx-vulkan / rlx-cuda
/// / rlx-wgpu.
fn classify_cast(src: DType, dst: DType) -> CastLower {
    if src == dst {
        return CastLower::Identity; // pure relabel (also covers C64→C64 / C128→C128)
    }
    if src.is_complex() || dst.is_complex() {
        // F64 is the one component type with no faithful f32-lane storage here
        // (it is demoted to a single lossy lane elsewhere), so a complex cast
        // touching it on the real side is still rejected.
        if src == DType::F64 || dst == DType::F64 {
            return CastLower::Reject;
        }
        let mode = match (src, dst) {
            (s, DType::C64) if !s.is_complex() => 0,  // real → C64
            (DType::C64, d) if !d.is_complex() => 1,  // C64 → real
            (s, DType::C128) if !s.is_complex() => 2, // real → C128
            (DType::C128, d) if !d.is_complex() => 3, // C128 → real
            (DType::C64, DType::C128) => 4,
            (DType::C128, DType::C64) => 5,
            _ => return CastLower::Reject,
        };
        return CastLower::Complex(mode);
    }
    if dst == DType::Bool {
        return CastLower::Kernel(CAST_TO_BOOL);
    }
    if src.is_float() && dst.is_int() {
        return CastLower::Kernel(match dst {
            DType::I8 => CAST_F32_TO_I8,
            DType::I16 => CAST_F32_TO_I16,
            DType::I32 => CAST_F32_TO_I32,
            DType::I64 => CAST_F32_TO_I64,
            DType::U8 => CAST_F32_TO_U8,
            DType::U32 => CAST_F32_TO_U32,
            _ => unreachable!("is_int() covers all integer dtypes"),
        });
    }
    CastLower::Identity
}

/// True when a Cast needs its own slot + a conversion kernel (float→int /
/// →Bool / complex lane-move) or must be rejected — i.e. not an identity
/// relabel. Complex casts change the lane width (real 1 → C64 2 → C128 4), so
/// they cannot alias the input slot.
fn cast_is_kernel(graph: &Graph, node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::Cast { to } => !matches!(
            classify_cast(graph.node(node.inputs[0]).shape.dtype(), *to),
            CastLower::Identity
        ),
        _ => false,
    }
}

/// Number of f32 lanes a node occupies in the f32-uniform arena. Complex is
/// simulated on f32 lanes — C64 = 2 lanes/elem, C128 = 4 lanes df64; every
/// OTHER dtype is exactly ONE f32 lane per element (I64/Bool/… are widened to a
/// single lane, so `size_bytes()/4` must NOT be blanket-applied — that would
/// make I64 two lanes and Bool zero). Drives slot sizing and lane-aware
/// readback (reading a complex output by element count would truncate it to the
/// real parts).
fn arena_lane_count(shape: &Shape) -> usize {
    let elems = shape.num_elements().unwrap_or(0);
    match shape.dtype() {
        DType::C64 => elems * 2,
        DType::C128 => elems * 4,
        _ => elems,
    }
}

/// Complex `Op::Cast` on f32 lanes (host-path mirror of `kernels/complex_cast.cl`,
/// same six lane-move modes). `n` is the complex-element count (cast-invariant).
/// Used by `run_host` so the CPU-reference path keeps the same df64 lane
/// convention as the on-device kernels (rather than routing through rlx-cpu's
/// native-f64 C128 storage, which is a different byte layout).
fn complex_cast_host(input: &[f32], n: usize, mode: u32) -> Vec<f32> {
    let ld = |j: usize| input.get(j).copied().unwrap_or(0.0);
    let out_lanes = match mode {
        0 | 5 => 2 * n, // → C64
        1 | 3 => n,     // → real
        2 | 4 => 4 * n, // → C128
        _ => n,
    };
    let mut out = vec![0.0f32; out_lanes];
    for k in 0..n {
        match mode {
            0 => out[2 * k] = ld(k), // real → C64 (im=0)
            1 => out[k] = ld(2 * k), // C64 → real
            2 => out[4 * k] = ld(k), // real → C128 (rest 0)
            3 => out[k] = ld(4 * k), // C128 → real
            4 => {
                out[4 * k] = ld(2 * k); // C64 → C128
                out[4 * k + 2] = ld(2 * k + 1);
            }
            5 => {
                out[2 * k] = ld(4 * k); // C128 → C64
                out[2 * k + 1] = ld(4 * k + 2);
            }
            _ => {}
        }
    }
    out
}

/// Element-wise C64 binary op on f32 lanes (host-path mirror of
/// `kernels/binary_c64.cl`). `op` 0=add/1=sub/2=mul/3=div; `n` is the output
/// complex-element count; `n_a`/`n_b` are operand complex-element counts for
/// modulo broadcast. Formulas match rlx-cpu `exec_binary_full_c64`.
fn binary_c64_host(a: &[f32], b: &[f32], n: usize, n_a: usize, n_b: usize, op: u32) -> Vec<f32> {
    let na = n_a.max(1);
    let nb = n_b.max(1);
    let la = |j: usize| a.get(j).copied().unwrap_or(0.0);
    let lb = |j: usize| b.get(j).copied().unwrap_or(0.0);
    let mut out = vec![0.0f32; 2 * n];
    for k in 0..n {
        let ka = k % na;
        let kb = k % nb;
        let (ar, ai) = (la(2 * ka), la(2 * ka + 1));
        let (br, bi) = (lb(2 * kb), lb(2 * kb + 1));
        let (cr, ci) = match op {
            0 => (ar + br, ai + bi),
            1 => (ar - br, ai - bi),
            2 => (ar * br - ai * bi, ar * bi + ai * br),
            3 => {
                let d = br * br + bi * bi;
                ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
            }
            _ => (0.0, 0.0),
        };
        out[2 * k] = cr;
        out[2 * k + 1] = ci;
    }
    out
}

/// Reject a complex `Op::Binary` that has no simulated path — C128 arithmetic
/// (rlx-cpu has none either) and C64 max/min/pow (undefined for complex).
/// Returns the C64 op code (0=add/1=sub/2=mul/3=div) when supported; panics
/// otherwise, matching the CPU reference's rejection (never a silently-wrong
/// result). Shared by `run_host` and the L0 dispatch builder.
fn c64_binary_opcode(dtype: DType, op: rlx_ir::op::BinaryOp) -> u32 {
    if dtype == DType::C128 {
        panic!(
            "rlx-oneapi: Binary on C128: complex-f64 arithmetic is unsupported \
             (rlx-cpu has none either) — only C64 add/sub/mul/div are wired"
        );
    }
    let code = binop_id(op);
    if code > 3 {
        panic!(
            "rlx-oneapi: C64 Binary: {op:?} is undefined for complex (only \
             Add/Sub/Mul/Div); matches rlx-cpu rejection"
        );
    }
    code
}

#[derive(Clone)]
enum ParamVal {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

pub struct OneApiExecutable {
    /// Post-legalize, f32-uniform graph.
    graph: Graph,
    params: HashMap<String, ParamVal>,
    output_ids: Vec<NodeId>,
    output_dtypes: Vec<DType>,
    rng: RngOptions,
    active_extent: Option<(usize, usize)>,
}

unsafe impl Send for OneApiExecutable {}

impl OneApiExecutable {
    pub fn compile(graph: Graph) -> Self {
        Self::compile_rng(graph, RngOptions::default())
    }

    /// Legalize the graph to the native primitive set, then capture I/O maps.
    pub fn compile_rng(graph: Graph, rng: RngOptions) -> Self {
        Self::compile_rng_with_options(graph, rng, 64)
    }

    pub fn compile_rng_with_options(
        graph: Graph,
        rng: RngOptions,
        scan_unroll_max_length: u32,
    ) -> Self {
        use rlx_opt::pass::Pass as _;

        let graph = rlx_opt::LowerControlFlow.run(graph);
        // Decompose composed ops (LoraMatMul, FusedTransformerLayer, FAB,
        // DotGeneral, If/While) while keeping native FusedSwiGLU / residual-norm.
        // Folds biased projections into FusedMatMulBiasAct (native compose).
        let graph = crate::unfuse::unfuse(graph);
        let graph = rlx_opt::unfuse::unfuse_dit_modulation(graph);
        // PartitionedConv: CPU HostOp would Nop — expand. FusedConvBiasAct
        // stays first-class for the native `fused_conv_bias_act` kernel.
        let graph = crate::unfuse::expand_cpu_nop_fused(graph);
        // GatedDeltaNet → MatMul/Mul/Add time-unroll (no dedicated GDN kernel).
        let graph = crate::unfuse::expand_gated_delta_net(graph);
        // TransformRegion / BatchElementwiseRegion → Resize + ElementwiseRegion;
        // then ElementwiseRegion → Binary / Activation / … primitives.
        let graph = rlx_opt::rlx_fusion::DecomposeFusionRegions.run(graph);
        let graph = rlx_opt::UnfuseElementwiseRegions::FOR_CPU.run(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, SUPPORTED_OPS)
            .unwrap_or_else(|errs| panic!("{}", rlx_opt::format_legalize_error("oneapi", &errs)));
        let graph = rlx_cpu::rlx_maybe_unroll_scans!(graph, scan_unroll_max_length);
        let graph = rlx_opt::maybe_unroll_scans_budget(graph, 4096);
        let graph = rlx_opt::LegalizeBroadcast.run(graph);

        let output_ids = graph.outputs.clone();
        let output_dtypes = output_ids
            .iter()
            .map(|&id| graph.node(id).shape.dtype())
            .collect();

        Self {
            graph,
            params: HashMap::new(),
            output_ids,
            output_dtypes,
            rng,
            active_extent: None,
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params
            .insert(name.to_string(), ParamVal::F32(data.to_vec()));
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        self.params
            .insert(name.to_string(), ParamVal::Bytes(data.to_vec()));
    }

    pub fn output_dtypes(&self) -> Vec<DType> {
        self.output_dtypes.clone()
    }

    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    pub fn set_rng(&mut self, rng: RngOptions) {
        self.rng = rng;
    }

    pub fn rng(&self) -> RngOptions {
        self.rng
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        // Native dispatch only when a live device AND embedded kernels exist;
        // otherwise the CPU-reference interpreter (the dev-box / CI path).
        if oneapi_device().is_some() && kernels().is_some() {
            self.run_l0(inputs, read_indices)
        } else {
            self.run_host(inputs, read_indices)
        }
    }

    // ── dev-box path: whole-graph CPU reference interpreter ────────────────

    fn run_host(&self, inputs: &[(&str, &[f32])], read_indices: Option<&[usize]>) -> Vec<Vec<f32>> {
        let in_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        let mut f32v: HashMap<NodeId, Vec<f32>> = HashMap::new();
        let mut bytev: HashMap<NodeId, Vec<u8>> = HashMap::new();

        for node in self.graph.nodes() {
            let numel = node.shape.num_elements().unwrap_or(0);
            match &node.op {
                Op::Input { name } => {
                    let v = in_map
                        .get(name.as_str())
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| vec![0.0; numel]);
                    f32v.insert(node.id, v);
                }
                Op::Param { name } => match self.params.get(name) {
                    Some(ParamVal::F32(v)) => {
                        f32v.insert(node.id, v.clone());
                    }
                    Some(ParamVal::Bytes(b)) => {
                        bytev.insert(node.id, b.clone());
                    }
                    None => {
                        f32v.insert(node.id, vec![0.0; numel]);
                    }
                },
                Op::Constant { data } => {
                    if matches!(node.shape.dtype(), DType::U8 | DType::I8) {
                        bytev.insert(node.id, data.clone());
                    } else {
                        f32v.insert(node.id, widen_const_to_f32(data, node.shape.dtype()));
                    }
                }
                // Core Riemannian / SPD-manifold ops (F64) go through the
                // F64-aware `spd::eval` (widens f32→f64, runs the CPU thunk,
                // narrows back), not the f32-only `host::eval` — same split as
                // rlx-vulkan. Delegating with each node's REAL declared
                // dtype/shape handles the packed `[2n²+n]` ReEig/LogEig forward
                // output + precomputed backward layout for free.
                op if crate::spd::is_spd_host(op) => {
                    let in_specs: Vec<(Shape, Vec<f32>)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            (sh, f32v.get(&id).cloned().unwrap_or_default())
                        })
                        .collect();
                    let out = crate::spd::eval(&node.op, &node.shape, &in_specs);
                    f32v.insert(node.id, out);
                }
                Op::Scan { .. } => {
                    let out = rlx_cpu::thunk::run_scan_node_f32(node, |id| {
                        f32v.get(&id).cloned().unwrap_or_default()
                    });
                    f32v.insert(node.id, out);
                }
                // Complex Cast (real↔C64, real↔C128, C64↔C128): pure f32-lane
                // moves in the df64 convention — handled directly rather than
                // via rlx-cpu (whose native-f64 C128 storage is a different byte
                // layout), so `run_host` keeps the SAME lane convention as the
                // on-device `complex_cast` kernel + the shared widen/narrow
                // boundary. `numel` is the cast-invariant complex-element count.
                Op::Cast { to }
                    if !matches!(
                        classify_cast(self.graph.node(node.inputs[0]).shape.dtype(), *to),
                        CastLower::Identity | CastLower::Kernel(_)
                    ) =>
                {
                    let src = self.graph.node(node.inputs[0]).shape.dtype();
                    match classify_cast(src, *to) {
                        CastLower::Complex(mode) => {
                            let input = f32v.get(&node.inputs[0]).cloned().unwrap_or_default();
                            f32v.insert(node.id, complex_cast_host(&input, numel, mode));
                        }
                        CastLower::Reject => panic!(
                            "rlx-oneapi: Cast {src:?} → {to:?} touches an F64 real \
                             component with no faithful f32-lane storage — run on CPU"
                        ),
                        _ => unreachable!("guard excludes Identity / Kernel"),
                    }
                }
                // Complex Binary (C64 add/sub/mul/div): reads both [re, im]
                // lanes per element, evaluated directly to match the on-device
                // `binary_c64` kernel. C128 arithmetic + C64 max/min/pow reject
                // (matches rlx-cpu). `numel` is the output complex-element count.
                Op::Binary(op) if node.shape.dtype().is_complex() => {
                    let code = c64_binary_opcode(node.shape.dtype(), *op);
                    let a = f32v.get(&node.inputs[0]).cloned().unwrap_or_default();
                    let b = f32v.get(&node.inputs[1]).cloned().unwrap_or_default();
                    let na = self
                        .graph
                        .node(node.inputs[0])
                        .shape
                        .num_elements()
                        .unwrap_or(0);
                    let nb = self
                        .graph
                        .node(node.inputs[1])
                        .shape
                        .num_elements()
                        .unwrap_or(0);
                    f32v.insert(node.id, binary_c64_host(&a, &b, numel, na, nb, code));
                }
                _ => {
                    let in_specs: Vec<(Shape, HostBuf)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let buf = if let Some(b) = bytev.get(&id) {
                                HostBuf::Bytes(b.clone())
                            } else {
                                HostBuf::F32(f32v.get(&id).cloned().unwrap_or_default())
                            };
                            (sh, buf)
                        })
                        .collect();
                    match host::eval(&node.op, &node.shape, &in_specs) {
                        HostOut::F32(out) => {
                            f32v.insert(node.id, out);
                        }
                        HostOut::Bytes(b) => {
                            bytev.insert(node.id, b);
                        }
                    }
                }
            }
        }

        self.read_outputs(read_indices, |id, n| {
            f32v.get(&id)
                .map(|v| v[..n.min(v.len())].to_vec())
                .unwrap_or_else(|| vec![0.0; n])
        })
    }

    // ── native path: USM arena + per-op SPIR-V dispatch (HW-pending) ───────

    fn run_l0(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        let dev = oneapi_device().expect("rlx-oneapi: no device");
        let kerns = kernels().expect("rlx-oneapi: no kernels");

        let plan = plan_f32_uniform(&self.graph, 64);
        let arena = match crate::arena::Arena::from_plan(&plan) {
            Ok(a) => a,
            // Allocation failed on the device — fall back to the CPU path so we
            // still return correct results rather than panic.
            Err(_) => return self.run_host(inputs, read_indices),
        };

        // Upload constants, params, inputs into the USM arena.
        for node in self.graph.nodes() {
            match &node.op {
                Op::Constant { data } if arena.has(node.id) && !data.is_empty() => {
                    if matches!(node.shape.dtype(), DType::U8 | DType::I8) {
                        arena.write_bytes(node.id, data);
                    } else {
                        arena.write_f32(node.id, &widen_const_to_f32(data, node.shape.dtype()));
                    }
                }
                Op::Param { name } => match self.params.get(name) {
                    Some(ParamVal::F32(v)) => arena.write_f32(node.id, v),
                    Some(ParamVal::Bytes(b)) => arena.write_bytes(node.id, b),
                    None => {}
                },
                _ => {}
            }
        }
        let in_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        for node in self.graph.nodes() {
            if let Op::Input { name } = &node.op {
                if let Some(data) = in_map.get(name.as_str()) {
                    arena.write_f32(node.id, data);
                }
            }
        }

        // Execute node-by-node: native kernel where available, else CPU
        // host-fallback against the (host-coherent) USM arena.
        let list = dev.create_command_list().expect("rlx-oneapi: command list");
        // Transient USM allocations (e.g. Quantize affine tables) live until
        // the command list finishes; freed after `execute_sync`.
        let mut scratch: Vec<*mut c_void> = Vec::new();
        for node in self.graph.nodes() {
            if matches!(
                node.op,
                Op::Input { .. }
                    | Op::Param { .. }
                    | Op::Constant { .. }
                    | Op::Reshape { .. }
                    | Op::StopGradient
            ) {
                continue;
            }
            // Cast: identity relabels are arena-aliased (skip); float→int /
            // →Bool casts got their own f32-sized slot and dispatch the `unary`
            // kernel (value stored as an f32 lane). Complex is rejected.
            if let Op::Cast { to } = &node.op {
                let src = self.graph.node(node.inputs[0]).shape.dtype();
                match classify_cast(src, *to) {
                    CastLower::Identity => continue,
                    CastLower::Kernel(_) => {
                        self.dispatch(dev, kerns, list, "unary", node, &arena, &mut scratch);
                        continue;
                    }
                    // real↔C64, real↔C128, C64↔C128 — pure f32-lane moves.
                    CastLower::Complex(_) => {
                        self.dispatch(dev, kerns, list, "complex_cast", node, &arena, &mut scratch);
                        continue;
                    }
                    CastLower::Reject => panic!(
                        "rlx-oneapi: Cast {src:?} → {to:?} touches an F64 real component \
                         with no faithful f32-lane storage in the uniform arena — run on CPU"
                    ),
                }
            }
            // Complex Binary (C64 add/sub/mul/div) reads BOTH [re, im] lanes per
            // element, so it lowers to the standalone `binary_c64` kernel (not
            // the scalar `binary`). C128 arithmetic + C64 max/min/pow are
            // rejected inside the dispatch arg builder (matches rlx-cpu).
            if let Op::Binary(_) = &node.op {
                if node.shape.dtype().is_complex() {
                    self.dispatch(dev, kerns, list, "binary_c64", node, &arena, &mut scratch);
                    continue;
                }
            }
            // SPD-manifold ops (F64, no native kernel) read the USM arena, run
            // the F64-aware `spd::eval` (widen f32→f64 → CPU thunk → narrow),
            // and write back — exactly rlx-vulkan's host-fallback split from
            // the f32-only `host::eval`.
            if crate::spd::is_spd_host(&node.op) {
                let in_specs: Vec<(Shape, Vec<f32>)> = node
                    .inputs
                    .iter()
                    .map(|&id| {
                        let sh = self.graph.node(id).shape.clone();
                        let nn = sh.num_elements().unwrap_or(0);
                        (sh, arena.read_f32(id, nn))
                    })
                    .collect();
                let out = crate::spd::eval(&node.op, &node.shape, &in_specs);
                arena.write_f32(node.id, &out);
                continue;
            }
            if matches!(node.op, Op::Scan { .. }) {
                let out = rlx_cpu::thunk::run_scan_node_f32(node, |id| {
                    let nn = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    arena.read_f32(id, nn)
                });
                arena.write_f32(node.id, &out);
                continue;
            }
            // Packed HostOpDesc on the USM arena (wgpu-shaped): DenseSolve →
            // rlx-cpu LAPACK; LogMel has no OpenCL twin. No device oneMKL LAPACK.
            if matches!(
                node.op,
                Op::DenseSolve
                    | Op::BatchedDenseSolve
                    | Op::Cholesky
                    | Op::TriangularSolve { .. }
                    | Op::Det
                    | Op::LogDet
                    | Op::Sort { .. }
                    | Op::Svd { .. }
                    | Op::Qr { .. }
                    | Op::ArgSort { .. }
                    | Op::LogMel
                    | Op::LogMelBackward
            ) {
                let desc = rlx_cpu::thunk::host_op_desc_from_node(&self.graph, node, |id| {
                    arena.byte_offset(id)
                });
                unsafe {
                    rlx_cpu::thunk::execute_host_op_on_bytes(arena.base_ptr() as *mut u8, &desc);
                }
                continue;
            }
            if let Op::WelchPeaks { k, n_segments } = &node.op {
                let spec_shape = self.graph.node(node.inputs[0]).shape.clone();
                let use_gpu =
                    rlx_ir::audio::welch_peaks_gpu_native_eligible(&spec_shape, *k, *n_segments)
                        .unwrap_or(false);
                if use_gpu && kerns.get("welch_peaks").is_some() {
                    self.dispatch(dev, kerns, list, "welch_peaks", node, &arena, &mut scratch);
                } else {
                    let desc = rlx_cpu::thunk::host_op_desc_from_node(&self.graph, node, |id| {
                        arena.byte_offset(id)
                    });
                    unsafe {
                        rlx_cpu::thunk::execute_host_op_on_bytes(
                            arena.base_ptr() as *mut u8,
                            &desc,
                        );
                    }
                }
                continue;
            }
            // FusedMatMulBiasAct: compose existing matmul + Binary(Add) +
            // optional Activation into `out` (mirrors Vulkan schedule compose).
            if let Op::FusedMatMulBiasAct { activation } = &node.op {
                let can_compose = kerns.get("matmul").is_some()
                    && kerns.get("binary").is_some()
                    && (activation.is_none() || kerns.get("unary").is_some());
                if can_compose {
                    self.dispatch_fused_matmul_bias_act(dev, kerns, list, node, &arena);
                    continue;
                }
            }
            match native_kernel(&node.op).filter(|name| kerns.get(name).is_some()) {
                Some(name) => self.dispatch(dev, kerns, list, name, node, &arena, &mut scratch),
                None => {
                    // Read inputs out of the arena, eval on CPU, write back.
                    let in_specs: Vec<(Shape, HostBuf)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let nn = sh.num_elements().unwrap_or(0);
                            let buf = if matches!(sh.dtype(), DType::U8 | DType::I8 | DType::Bool) {
                                HostBuf::Bytes(arena.read_bytes(id, nn))
                            } else {
                                HostBuf::F32(arena.read_f32(id, nn))
                            };
                            (sh, buf)
                        })
                        .collect();
                    match host::eval(&node.op, &node.shape, &in_specs) {
                        HostOut::F32(out) => arena.write_f32(node.id, &out),
                        HostOut::Bytes(b) => arena.write_bytes(node.id, &b),
                    }
                }
            }
        }
        dev.execute_sync(list).expect("rlx-oneapi: execute");
        unsafe {
            let _ = (dev.lib.command_list_destroy)(list);
        }
        for p in scratch {
            dev.free(p);
        }

        self.read_outputs(read_indices, |id, n| arena.read_f32(id, n))
    }

    /// Compose `FusedMatMulBiasAct` as matmul → Binary(Add bias) → optional
    /// Activation, writing through `out` (same schedule as rlx-vulkan).
    fn dispatch_fused_matmul_bias_act(
        &self,
        dev: &crate::device::OneApiDevice,
        kerns: &crate::kernels::Kernels,
        list: crate::level_zero::CommandListHandle,
        node: &rlx_ir::Node,
        arena: &crate::arena::Arena,
    ) {
        let Op::FusedMatMulBiasAct { activation } = &node.op else {
            return;
        };
        let off = |id: NodeId| arena.elem_offset(id);
        let out = node.id;
        let a = node.inputs[0];
        let b = node.inputs[1];
        let bias = node.inputs[2];
        let ad = dims(&self.graph, a);
        let bd = dims(&self.graph, b);
        let od = dims(&self.graph, out);
        let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
        let n = bd[bd.len() - 1];
        let batch = if od.len() > 2 {
            numel(&od[..od.len() - 2])
        } else {
            1
        };
        let a_batch = if ad.len() > 2 {
            numel(&ad[..ad.len() - 2])
        } else {
            1
        };
        let b_batch = if bd.len() > 2 {
            numel(&bd[..bd.len() - 2])
        } else {
            1
        };
        let a_bs = if a_batch <= 1 { 0 } else { m * k };
        let b_bs = if b_batch <= 1 { 0 } else { k * n };
        let base = arena.base_ptr();

        let matmul_args = [
            KArg::Ptr(base),
            KArg::U32(m as u32),
            KArg::U32(k as u32),
            KArg::U32(n as u32),
            KArg::U32(off(a)),
            KArg::U32(off(b)),
            KArg::U32(off(out)),
            KArg::U32(batch as u32),
            KArg::U32(a_bs as u32),
            KArg::U32(b_bs as u32),
            KArg::U32((m * n) as u32),
        ];
        if let Some(kernel) = kerns.get("matmul") {
            append_kernel_launch(dev, kernel, list, &matmul_args, batch.max(1) * m * n, 64);
        }

        let total = numel(&od);
        let bn = numel(&dims(&self.graph, bias));
        let add_args = [
            KArg::Ptr(base),
            KArg::U32(total as u32),
            KArg::U32(off(out)),
            KArg::U32(off(bias)),
            KArg::U32(off(out)),
            KArg::U32(0), // a contiguous (matmul result)
            KArg::U32(if bn == total { 0 } else { bn as u32 }),
            KArg::U32(binop_id(rlx_ir::op::BinaryOp::Add)),
        ];
        if let Some(kernel) = kerns.get("binary") {
            append_kernel_launch(dev, kernel, list, &add_args, total.max(1), 256);
        }

        if let Some(act) = activation {
            let act_args = [
                KArg::Ptr(base),
                KArg::U32(total as u32),
                KArg::U32(off(out)),
                KArg::U32(off(out)),
                KArg::U32(act_id(*act)),
            ];
            if let Some(kernel) = kerns.get("unary") {
                append_kernel_launch(dev, kernel, list, &act_args, total.max(1), 256);
            }
        }
    }

    /// Set kernel arguments (arg 0 = arena base pointer, then scalars) and
    /// append a launch onto `list`. Arg layouts match `kernels/<name>.cl`.
    /// `scratch` collects transient USM allocations (Quantize affine) freed by
    /// the caller after `execute_sync`.
    fn dispatch(
        &self,
        dev: &crate::device::OneApiDevice,
        kerns: &crate::kernels::Kernels,
        list: crate::level_zero::CommandListHandle,
        name: &str,
        node: &rlx_ir::Node,
        arena: &crate::arena::Arena,
        scratch: &mut Vec<*mut c_void>,
    ) {
        let Some(kernel) = kerns.get(name) else {
            return;
        };
        let off = |id: NodeId| arena.elem_offset(id);
        let out = node.id;
        let mut args: Vec<KArg> = vec![KArg::Ptr(arena.base_ptr())];
        let (global, local): (usize, u32) = match &node.op {
            // Standalone `binary_c64` kernel for complex output: n = output
            // complex-element count, n_a/n_b = operand complex-element counts
            // (>= 1) for modulo broadcast, offsets are f32-lane starts. Rejects
            // C128 arithmetic + C64 max/min/pow (matches rlx-cpu).
            Op::Binary(op) if node.shape.dtype().is_complex() => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                let na = numel(&dims(&self.graph, a));
                let nb = numel(&dims(&self.graph, b));
                let code = c64_binary_opcode(node.shape.dtype(), *op);
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(code),
                    KArg::U32(na.max(1) as u32),
                    KArg::U32(nb.max(1) as u32),
                ]);
                (n, 256)
            }
            Op::Binary(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                let an = numel(&dims(&self.graph, a));
                let bn = numel(&dims(&self.graph, b));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(if an == n { 0 } else { an as u32 }),
                    KArg::U32(if bn == n { 0 } else { bn as u32 }),
                    KArg::U32(binop_id(*op)),
                ]);
                (n, 256)
            }
            Op::Activation(act) => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                    KArg::U32(act_id(*act)),
                ]);
                (n, 256)
            }
            // float→int / →Bool cast via the `unary` kernel (op ids 100–106), or
            // a complex lane-move via the `complex_cast` kernel (mode 0..5). Both
            // share the (n, in_off, out_off, code) arg layout; the caller routes
            // to the matching kernel name. `n` is the (cast-invariant) element
            // count. The caller only routes here for `Kernel` / `Complex`.
            Op::Cast { to } => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                let src = self.graph.node(x).shape.dtype();
                let code = match classify_cast(src, *to) {
                    CastLower::Kernel(op) => op,      // unary conversion op id
                    CastLower::Complex(mode) => mode, // complex_cast lane-move mode
                    _ => unreachable!("dispatch(Cast) only called for kernel / complex casts"),
                };
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                    KArg::U32(code),
                ]);
                (n, 256)
            }
            Op::MatMul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let ad = dims(&self.graph, a);
                let bd = dims(&self.graph, b);
                let od = dims(&self.graph, out);
                let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
                let n = bd[bd.len() - 1];
                let batch = if od.len() > 2 {
                    numel(&od[..od.len() - 2])
                } else {
                    1
                };
                let a_batch = if ad.len() > 2 {
                    numel(&ad[..ad.len() - 2])
                } else {
                    1
                };
                let b_batch = if bd.len() > 2 {
                    numel(&bd[..bd.len() - 2])
                } else {
                    1
                };
                let a_bs = if a_batch <= 1 { 0 } else { m * k };
                let b_bs = if b_batch <= 1 { 0 } else { k * n };
                args.extend([
                    KArg::U32(m as u32),
                    KArg::U32(k as u32),
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(batch as u32),
                    KArg::U32(a_bs as u32),
                    KArg::U32(b_bs as u32),
                    KArg::U32((m * n) as u32),
                ]);
                (batch.max(1) * m * n, 64)
            }
            Op::Softmax { axis } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let axis_len = xd[ax];
                let outer = numel(&xd[..ax]);
                let inner = numel(&xd[ax + 1..]);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(axis_len as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                ]);
                (outer * inner, 256)
            }
            Op::RmsNorm { axis, eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let beta = node.inputs[2];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let n = xd[ax];
                let rows = numel(&xd) / n.max(1);
                args.extend([
                    KArg::U32(rows as u32),
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(beta)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                ]);
                (rows, 64)
            }
            Op::AdaLayerNormBackward { norm, eps } => {
                use rlx_ir::ada_modulation_launch;
                use rlx_ir::op::AdaNormKind;
                let x = node.inputs[0];
                let scale = node.inputs[1];
                let dy = node.inputs[3];
                let x_dims = dims(&self.graph, x);
                let mod_dims = dims(&self.graph, scale);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &mod_dims);
                let layer_norm = matches!(norm, AdaNormKind::LayerNorm) as u32;
                args.extend([
                    KArg::U32(mod_rows),
                    KArg::U32(seq_per_mod),
                    KArg::U32(inner),
                    KArg::U32(off(x)),
                    KArg::U32(off(scale)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::U32(layer_norm),
                    KArg::F32(*eps),
                ]);
                (mod_rows as usize, 64)
            }
            Op::GatedResidualBackward => {
                use rlx_ir::ada_modulation_launch;
                let y = node.inputs[1];
                let gate = node.inputs[2];
                let dy = node.inputs[3];
                let x_dims = dims(&self.graph, dy);
                let gate_dims = dims(&self.graph, gate);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &gate_dims);
                args.extend([
                    KArg::U32(mod_rows),
                    KArg::U32(seq_per_mod),
                    KArg::U32(inner),
                    KArg::U32(off(y)),
                    KArg::U32(off(gate)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (mod_rows as usize, 64)
            }
            Op::GroupNorm { num_groups, eps } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
                args.extend([
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[2])),
                    KArg::U32(off(out)),
                    KArg::U32(n as u32),
                    KArg::U32(c as u32),
                    KArg::U32(h as u32),
                    KArg::U32(w as u32),
                    KArg::U32(*num_groups as u32),
                    KArg::F32(*eps),
                ]);
                (n * *num_groups, 64)
            }
            Op::GroupNormBackwardInput { num_groups, eps } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
                args.extend([
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[3])),
                    KArg::U32(off(out)),
                    KArg::U32(n as u32),
                    KArg::U32(c as u32),
                    KArg::U32(h as u32),
                    KArg::U32(w as u32),
                    KArg::U32(*num_groups as u32),
                    KArg::F32(*eps),
                ]);
                (n * *num_groups, 64)
            }
            Op::GroupNormBackwardGamma { num_groups, eps } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
                args.extend([
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(out)),
                    KArg::U32(n as u32),
                    KArg::U32(c as u32),
                    KArg::U32(h as u32),
                    KArg::U32(w as u32),
                    KArg::U32(*num_groups as u32),
                    KArg::F32(*eps),
                ]);
                (1, 1)
            }
            Op::GroupNormBackwardBeta { .. } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (n, c, h, w) = (xd[0], xd[1], xd[2], xd[3]);
                args.extend([
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(out)),
                    KArg::U32(n as u32),
                    KArg::U32(c as u32),
                    KArg::U32(h as u32),
                    KArg::U32(w as u32),
                ]);
                (1, 1)
            }
            Op::FusedResidualLN { has_bias, eps } | Op::FusedResidualRmsNorm { has_bias, eps } => {
                let x = node.inputs[0];
                let residual = node.inputs[1];
                let (bias, gamma, beta) = if *has_bias {
                    (node.inputs[2], node.inputs[3], node.inputs[4])
                } else {
                    (x, node.inputs[2], node.inputs[3]) // bias unused
                };
                let xd = dims(&self.graph, out);
                let inner = *xd.last().unwrap_or(&1);
                let total = numel(&xd);
                let outer = total / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(residual)),
                    KArg::U32(off(bias)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(beta)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                    KArg::U32(if *has_bias { 1 } else { 0 }),
                ]);
                (outer, 64)
            }
            Op::FusedSwiGLU {
                cast_to: _,
                gate_first,
            } => {
                let x = node.inputs[0];
                let od = dims(&self.graph, out);
                let n_half = *od.last().unwrap_or(&1);
                let total = numel(&od);
                args.extend([
                    KArg::U32(n_half as u32),
                    KArg::U32(total as u32),
                    KArg::U32(if *gate_first { 1 } else { 0 }),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                ]);
                (total, 256)
            }
            Op::SoftmaxCrossEntropy => {
                let logits = node.inputs[0];
                let targets = node.inputs[1];
                let ld = dims(&self.graph, logits);
                let inner = *ld.last().unwrap_or(&1);
                let outer = numel(&ld) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(logits)),
                    KArg::U32(off(targets)),
                    KArg::U32(off(out)),
                ]);
                (outer, 64)
            }
            Op::SoftmaxCrossEntropyWithLogits => {
                let logits = node.inputs[0];
                let labels = node.inputs[1];
                let ld = dims(&self.graph, logits);
                let inner = *ld.last().unwrap_or(&1);
                let outer = numel(&ld) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(logits)),
                    KArg::U32(off(labels)),
                    KArg::U32(off(out)),
                ]);
                (outer, 64)
            }
            Op::SoftmaxCrossEntropyBackward => {
                let logits = node.inputs[0];
                let labels = node.inputs[1];
                let d_loss = node.inputs[2];
                let ld = dims(&self.graph, logits);
                let inner = *ld.last().unwrap_or(&1);
                let outer = numel(&ld) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(logits)),
                    KArg::U32(off(labels)),
                    KArg::U32(off(d_loss)),
                    KArg::U32(off(out)),
                ]);
                (outer, 64)
            }
            Op::Fma => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let c = node.inputs[2];
                let n = numel(&dims(&self.graph, out));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(c)),
                    KArg::U32(off(out)),
                ]);
                (n, 256)
            }
            Op::ComplexNormSq => {
                let z = node.inputs[0];
                let n = numel(&dims(&self.graph, out)); // complex-element count
                args.extend([KArg::U32(n as u32), KArg::U32(off(z)), KArg::U32(off(out))]);
                (n, 256)
            }
            Op::ComplexNormSqBackward => {
                let z = node.inputs[0];
                let g = node.inputs[1];
                let n = numel(&dims(&self.graph, z)); // complex-element count
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(z)),
                    KArg::U32(off(g)),
                    KArg::U32(off(out)),
                ]);
                (n, 256)
            }
            Op::Conjugate => {
                let z = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                args.extend([KArg::U32(n as u32), KArg::U32(off(z)), KArg::U32(off(out))]);
                (n, 256)
            }
            Op::FakeQuantize {
                bits,
                axis,
                scale_mode: rlx_ir::op::ScaleMode::Fixed,
                ..
            } => {
                let x = node.inputs[0];
                let scale = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                let (chan_dim, inner) = match *axis {
                    None => (1usize, n.max(1)),
                    Some(d) => {
                        let xd = dims(&self.graph, out);
                        let chan = xd[d];
                        let inn: usize = xd[d + 1..].iter().product::<usize>().max(1);
                        (chan, inn)
                    }
                };
                let q_max = match *bits {
                    8 => 127.0f32,
                    4 => 7.0,
                    2 => 1.0,
                    other => panic!("rlx-oneapi FakeQuantize Fixed: unsupported bits {other}"),
                };
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(chan_dim as u32),
                    KArg::U32(inner as u32),
                    KArg::F32(q_max),
                    KArg::U32(off(x)),
                    KArg::U32(off(scale)),
                    KArg::U32(off(out)),
                ]);
                (n, 256)
            }
            Op::FakeQuantize {
                bits,
                axis,
                scale_mode: rlx_ir::op::ScaleMode::PerBatch,
                ..
            } => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                let (chan_dim, inner) = match *axis {
                    None => (1usize, n.max(1)),
                    Some(d) => {
                        let xd = dims(&self.graph, out);
                        let chan = xd[d];
                        let inn: usize = xd[d + 1..].iter().product::<usize>().max(1);
                        (chan, inn)
                    }
                };
                let q_max = match *bits {
                    8 => 127.0f32,
                    4 => 7.0,
                    2 => 1.0,
                    other => panic!("rlx-oneapi FakeQuantize PerBatch: unsupported bits {other}"),
                };
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(chan_dim as u32),
                    KArg::U32(inner as u32),
                    KArg::F32(q_max),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                ]);
                (chan_dim, 64)
            }
            Op::BatchNormInference { eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let beta = node.inputs[2];
                let mean = node.inputs[3];
                let var = node.inputs[4];
                let xd = dims(&self.graph, x);
                let channels = *xd.last().unwrap_or(&1);
                let n = numel(&xd);
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(channels as u32),
                    KArg::F32(*eps),
                    KArg::U32(off(x)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(beta)),
                    KArg::U32(off(mean)),
                    KArg::U32(off(var)),
                    KArg::U32(off(out)),
                ]);
                (n, 256)
            }
            Op::BatchNormInferenceBackwardInput { eps } => {
                let gamma = node.inputs[1];
                let var = node.inputs[3];
                let dy = node.inputs[4];
                let xd = dims(&self.graph, dy);
                let channels = *xd.last().unwrap_or(&1);
                let n = numel(&xd);
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(channels as u32),
                    KArg::F32(*eps),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(var)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (n, 256)
            }
            Op::BatchNormInferenceBackwardGamma { eps } => {
                let x = node.inputs[0];
                let mean = node.inputs[1];
                let var = node.inputs[2];
                let dy = node.inputs[3];
                let xd = dims(&self.graph, x);
                let channels = *xd.last().unwrap_or(&1);
                let n = numel(&xd);
                let count = n / channels.max(1);
                args.extend([
                    KArg::U32(count as u32),
                    KArg::U32(channels as u32),
                    KArg::F32(*eps),
                    KArg::U32(off(x)),
                    KArg::U32(off(mean)),
                    KArg::U32(off(var)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (channels, 64)
            }
            Op::BatchNormInferenceBackwardBeta => {
                let dy = node.inputs[0];
                let xd = dims(&self.graph, dy);
                let channels = *xd.last().unwrap_or(&1);
                let n = numel(&xd);
                let count = n / channels.max(1);
                args.extend([
                    KArg::U32(count as u32),
                    KArg::U32(channels as u32),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (channels, 64)
            }
            Op::ReluBackward => {
                let x = node.inputs[0];
                let dy = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::U32(0), // relu
                ]);
                (n, 256)
            }
            Op::ActivationBackward { kind } => {
                let x = node.inputs[0];
                let dy = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::U32(activation_bwd_op_id(*kind)),
                ]);
                (n, 256)
            }
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (batch, seq, hidden) = if xd.len() >= 3 {
                    (xd[0], xd[1], xd[2])
                } else {
                    panic!("rlx-oneapi AxialRope2d: expected rank ≥ 3, got {xd:?}");
                };
                let n_total = batch * seq * hidden;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(hidden as u32),
                    KArg::U32(*end_x as u32),
                    KArg::U32(*end_y as u32),
                    KArg::U32(*head_dim as u32),
                    KArg::U32(*num_heads as u32),
                    KArg::U32(*repeat_factor as u32),
                    KArg::F32(*theta),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                    KArg::U32(n_total as u32),
                ]);
                (n_total, 256)
            }
            Op::LayerNormBackwardInput { axis, eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let dy = node.inputs[2];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let inner = xd[ax];
                let outer = numel(&xd) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                ]);
                (outer.max(1), 64)
            }
            Op::LayerNormBackwardGamma { axis, eps } => {
                let x = node.inputs[0];
                let dy = node.inputs[1];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let inner = xd[ax];
                let outer = numel(&xd) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                ]);
                (1, 1)
            }
            Op::RmsNormBackwardInput { axis, eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let dy = node.inputs[3];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let inner = xd[ax];
                let outer = numel(&xd) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                ]);
                (outer.max(1), 64)
            }
            Op::RmsNormBackwardGamma { axis, eps } => {
                let x = node.inputs[0];
                let dy = node.inputs[3];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let inner = xd[ax];
                let outer = numel(&xd) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                    KArg::U32(1), // wrt = dgamma
                ]);
                (1, 1)
            }
            Op::RmsNormBackwardBeta { axis, eps } => {
                let x = node.inputs[0];
                let dy = node.inputs[3];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let inner = xd[ax];
                let outer = numel(&xd) / inner.max(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                    KArg::U32(2), // wrt = dbeta
                ]);
                (1, 1)
            }
            Op::FftButterflyStage { stage, n_fft } => {
                let state = node.inputs[0];
                let gate = node.inputs[1];
                let rev = node.inputs[2];
                let tw_re = node.inputs[3];
                let tw_im = node.inputs[4];
                let sd = dims(&self.graph, state);
                let batch = if sd.is_empty() { 1 } else { sd[0] };
                let half = (*n_fft as usize / 2).max(1);
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(*n_fft),
                    KArg::U32(*stage),
                    KArg::U32(half as u32),
                    KArg::U32(off(state)),
                    KArg::U32(off(out)),
                    KArg::U32(off(gate)),
                    KArg::U32(off(rev)),
                    KArg::U32(off(tw_re)),
                    KArg::U32(off(tw_im)),
                ]);
                (batch * half, 64)
            }
            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                let x = node.inputs[0];
                let w = node.inputs[1];
                let xd = dims(&self.graph, x);
                let wd = dims(&self.graph, w);
                let od = dims(&self.graph, out);
                assert!(
                    xd.len() == 5 && wd.len() == 5 && od.len() == 5,
                    "rlx-oneapi Conv3d: expected NCDHW ranks, got x={xd:?} w={wd:?} o={od:?}"
                );
                let (n, c_in, d, h, ww) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
                let (c_out, _, kd, kh, kw) = (wd[0], wd[1], wd[2], wd[3], wd[4]);
                let (d_out, h_out, w_out) = (od[2], od[3], od[4]);
                let total = n * c_out * d_out * h_out * w_out;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c_in as u32),
                    KArg::U32(c_out as u32),
                    KArg::U32(d as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(d_out as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kd as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(stride[2] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(padding[2] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32(dilation[2] as u32),
                    KArg::U32(*groups as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(w)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 256)
            }
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                let x = node.inputs[0];
                let w = node.inputs[1];
                let xd = dims(&self.graph, x);
                let wd = dims(&self.graph, w);
                let od = dims(&self.graph, out);
                assert!(
                    xd.len() == 5 && wd.len() == 5 && od.len() == 5,
                    "rlx-oneapi ConvTranspose3d: expected NCDHW ranks, got x={xd:?} w={wd:?} o={od:?}"
                );
                let (n, c_in, d, h, ww) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
                let (_, c_out_pg, kd, kh, kw) = (wd[0], wd[1], wd[2], wd[3], wd[4]);
                let (d_out, h_out, w_out) = (od[2], od[3], od[4]);
                let c_out = od[1];
                let _ = c_out_pg;
                let total = n * c_out * d_out * h_out * w_out;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c_in as u32),
                    KArg::U32(c_out as u32),
                    KArg::U32(d as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(d_out as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kd as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(stride[2] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(padding[2] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32(dilation[2] as u32),
                    KArg::U32(*groups as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(w)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 256)
            }
            Op::ConvTranspose2d {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                let x = node.inputs[0];
                let w = node.inputs[1];
                let xd = dims(&self.graph, x);
                let wd = dims(&self.graph, w);
                let od = dims(&self.graph, out);
                assert!(
                    xd.len() == 4 && wd.len() == 4 && od.len() == 4,
                    "rlx-oneapi ConvTranspose2d: expected NCHW ranks, got x={xd:?} w={wd:?} o={od:?}"
                );
                let (nn, cin, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (_, _cout_pg, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
                let (cout, oh, ow) = (od[1], od[2], od[3]);
                let total = nn * cout * oh * ow;
                args.extend([
                    KArg::U32(nn as u32),
                    KArg::U32(cin as u32),
                    KArg::U32(hh as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(cout as u32),
                    KArg::U32(oh as u32),
                    KArg::U32(ow as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32((*groups).max(1) as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(w)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 64)
            }
            Op::Conv2dBackwardInput {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                let dy = node.inputs[0];
                let w = node.inputs[1];
                let dyd = dims(&self.graph, dy);
                let wd = dims(&self.graph, w);
                let od = dims(&self.graph, out); // dx
                let (n, c_out, h_out, w_out) = (dyd[0], dyd[1], dyd[2], dyd[3]);
                let (_, c_in_pg, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
                let (c_in, h, ww) = (od[1], od[2], od[3]);
                let _ = c_in_pg;
                let total = n * c_in * h * ww;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c_in as u32),
                    KArg::U32(c_out as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32(*groups as u32),
                    KArg::U32(off(dy)),
                    KArg::U32(off(w)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 256)
            }
            Op::Conv2dBackwardWeight {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                let x = node.inputs[0];
                let dy = node.inputs[1];
                let xd = dims(&self.graph, x);
                let dyd = dims(&self.graph, dy);
                let od = dims(&self.graph, out); // dw
                let (n, c_in, h, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (c_out, h_out, w_out) = (dyd[1], dyd[2], dyd[3]);
                let (_, _, kh, kw) = (od[0], od[1], od[2], od[3]);
                let c_in_per_g = c_in / (*groups).max(1);
                let total = c_out * c_in_per_g * kh * kw;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c_in as u32),
                    KArg::U32(c_out as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32(*groups as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 256)
            }
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let x = node.inputs[0];
                let dy = node.inputs[1];
                let xd = dims(&self.graph, x);
                let dyd = dims(&self.graph, dy);
                let (n, c, h, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (h_out, w_out) = (dyd[2], dyd[3]);
                let kh = kernel_size[0];
                let kw = kernel_size.get(1).copied().unwrap_or(kh);
                let sh = stride.first().copied().unwrap_or(1);
                let sw = stride.get(1).copied().unwrap_or(sh);
                let ph = padding.first().copied().unwrap_or(0);
                let pw = padding.get(1).copied().unwrap_or(ph);
                let total = n * c * h * ww;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(sh as u32),
                    KArg::U32(sw as u32),
                    KArg::U32(ph as u32),
                    KArg::U32(pw as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (total.max(1), 256)
            }
            Op::FusedConvBiasAct {
                stride,
                padding,
                dilation,
                groups,
                activation,
                has_residual,
                ..
            } => {
                let x = node.inputs[0];
                let w = node.inputs[1];
                let bias = node.inputs[2];
                let xd = dims(&self.graph, x);
                let wd = dims(&self.graph, w);
                let od = dims(&self.graph, out);
                let (n, c_in, h, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (c_out, _, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
                let (h_out, w_out) = (od[2], od[3]);
                let residual = if *has_residual {
                    node.inputs[3]
                } else {
                    x // unused
                };
                let act = match activation {
                    None => 0xFFFFu32,
                    Some(a) => act_id(*a),
                };
                let total = n * c_out * h_out * w_out;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(c_in as u32),
                    KArg::U32(c_out as u32),
                    KArg::U32(h as u32),
                    KArg::U32(ww as u32),
                    KArg::U32(h_out as u32),
                    KArg::U32(w_out as u32),
                    KArg::U32(kh as u32),
                    KArg::U32(kw as u32),
                    KArg::U32(stride[0] as u32),
                    KArg::U32(stride[1] as u32),
                    KArg::U32(padding[0] as u32),
                    KArg::U32(padding[1] as u32),
                    KArg::U32(dilation[0] as u32),
                    KArg::U32(dilation[1] as u32),
                    KArg::U32(*groups as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(w)),
                    KArg::U32(off(bias)),
                    KArg::U32(off(residual)),
                    KArg::U32(off(out)),
                    KArg::U32(if *has_residual { 1 } else { 0 }),
                    KArg::U32(act),
                ]);
                (total.max(1), 256)
            }
            Op::RopeBackward { head_dim, n_rot } => {
                let dy = node.inputs[0];
                let cos = node.inputs[1];
                let sin = node.inputs[2];
                let dyd = dims(&self.graph, dy);
                let (batch, seq, hidden) = if dyd.len() >= 3 {
                    (dyd[0], dyd[1], dyd[2])
                } else {
                    (1, dyd[0], dyd.get(1).copied().unwrap_or(1))
                };
                let cos_len = numel(&dims(&self.graph, cos));
                let total = batch * seq * hidden;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(hidden as u32),
                    KArg::U32(*head_dim as u32),
                    KArg::U32(*n_rot as u32),
                    KArg::U32(off(dy)),
                    KArg::U32(off(cos)),
                    KArg::U32(off(sin)),
                    KArg::U32(off(out)),
                    KArg::U32(cos_len as u32),
                ]);
                (total.max(1), 256)
            }
            Op::Lstm { hidden_size, .. } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (batch, seq, input_size) = (xd[0], xd[1], xd[2]);
                let hidden = *hidden_size;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(input_size as u32),
                    KArg::U32(hidden as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[2])),
                    KArg::U32(off(node.inputs[3])),
                    KArg::U32(off(out)),
                    KArg::U32(seq as u32), // seq_stride
                ]);
                // One work-group per batch item; local size = LSTM_MAX_H.
                (batch.max(1) * 256, 256)
            }
            Op::Gru { hidden_size, .. } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (batch, seq, input_size) = (xd[0], xd[1], xd[2]);
                let hidden = *hidden_size;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(input_size as u32),
                    KArg::U32(hidden as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[2])),
                    KArg::U32(off(node.inputs[3])),
                    KArg::U32(off(node.inputs[4])),
                    KArg::U32(off(out)),
                    KArg::U32(seq as u32), // seq_stride
                ]);
                // One work-group per batch item; local size = GRU_MAX_H.
                (batch.max(1) * 256, 256)
            }
            Op::Rnn {
                hidden_size, relu, ..
            } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let (batch, seq, input_size) = (xd[0], xd[1], xd[2]);
                let hidden = *hidden_size;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(input_size as u32),
                    KArg::U32(hidden as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[2])),
                    KArg::U32(off(node.inputs[3])),
                    KArg::U32(off(out)),
                    KArg::U32(seq as u32),
                    KArg::U32(u32::from(*relu)),
                ]);
                (batch.max(1) * 256, 256)
            }
            Op::Mamba2 {
                head_dim,
                state_size,
            } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x); // [B,S,H,P]
                let (batch, seq, heads) = (xd[0], xd[1], xd[2]);
                let total = batch * heads * *head_dim;
                args.extend([
                    KArg::U32(batch as u32),
                    KArg::U32(seq as u32),
                    KArg::U32(heads as u32),
                    KArg::U32(*head_dim as u32),
                    KArg::U32(*state_size as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(node.inputs[1])),
                    KArg::U32(off(node.inputs[2])),
                    KArg::U32(off(node.inputs[3])),
                    KArg::U32(off(node.inputs[4])),
                    KArg::U32(off(out)),
                    KArg::U32(seq as u32),
                ]);
                (total.max(1), 64)
            }
            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                let (chan_dim, inner) = match *axis {
                    None => (1usize, n.max(1)),
                    Some(d) => {
                        let xd = dims(&self.graph, out);
                        let chan = xd[d];
                        let inn: usize = xd[d + 1..].iter().product::<usize>().max(1);
                        (chan, inn)
                    }
                };
                let mut affine = Vec::with_capacity(chan_dim * 2);
                for c in 0..chan_dim {
                    affine.push(scales[c].to_bits());
                    affine.push(zero_points[c] as u32);
                }
                let bytes = affine.len() * 4;
                let ptr = dev
                    .alloc_shared(bytes)
                    .expect("rlx-oneapi: Quantize affine USM alloc failed");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        affine.as_ptr() as *const u8,
                        ptr as *mut u8,
                        bytes,
                    );
                }
                scratch.push(ptr);
                let q_byte_off = off(out) * 4; // packed i8 at start of f32 slot
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(chan_dim as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(q_byte_off),
                    KArg::Ptr(ptr),
                ]);
                (n.max(1), 256)
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let q = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                let (chan_dim, inner) = match *axis {
                    None => (1usize, n.max(1)),
                    Some(d) => {
                        let xd = dims(&self.graph, out);
                        let chan = xd[d];
                        let inn: usize = xd[d + 1..].iter().product::<usize>().max(1);
                        (chan, inn)
                    }
                };
                let mut affine = Vec::with_capacity(chan_dim * 2);
                for c in 0..chan_dim {
                    affine.push(scales[c].to_bits());
                    affine.push(zero_points[c] as u32);
                }
                let bytes = affine.len() * 4;
                let ptr = dev
                    .alloc_shared(bytes)
                    .expect("rlx-oneapi: Dequantize affine USM alloc failed");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        affine.as_ptr() as *const u8,
                        ptr as *mut u8,
                        bytes,
                    );
                }
                scratch.push(ptr);
                let q_byte_off = off(q) * 4;
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(chan_dim as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(q_byte_off),
                    KArg::U32(off(out)),
                    KArg::Ptr(ptr),
                ]);
                (n.max(1), 256)
            }
            Op::CumsumBackward { exclusive, .. } => {
                let dy = node.inputs[0];
                let xd = dims(&self.graph, dy);
                let cols = *xd.last().unwrap_or(&1);
                let rows = numel(&xd) / cols.max(1);
                args.extend([
                    KArg::U32(rows as u32),
                    KArg::U32(cols as u32),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::U32(if *exclusive { 1 } else { 0 }),
                ]);
                (rows.max(1), 64)
            }
            Op::GatherBackward { axis } => {
                let dy = node.inputs[0];
                let idx = node.inputs[1];
                let dy_d = dims(&self.graph, dy);
                let out_d = dims(&self.graph, out);
                let idx_d = dims(&self.graph, idx);
                let rank = out_d.len();
                let ax = if *axis < 0 {
                    (rank as i32 + *axis) as usize
                } else {
                    *axis as usize
                };
                let outer = numel(&dy_d[..ax]).max(1);
                let num_idx = idx_d.get(ax).copied().unwrap_or(1);
                let trailing = numel(&dy_d[ax + 1..]).max(1);
                let axis_dim = out_d.get(ax).copied().unwrap_or(1);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(axis_dim as u32),
                    KArg::U32(num_idx as u32),
                    KArg::U32(trailing as u32),
                    KArg::U32(off(dy)),
                    KArg::U32(off(idx)),
                    KArg::U32(off(out)),
                ]);
                (outer.max(1), 1)
            }
            Op::WelchPeaks { k, n_segments } => {
                let spec = node.inputs[0];
                let spec_shape = self.graph.node(spec).shape.clone();
                let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                    .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                args.extend([
                    KArg::U32(off(spec)),
                    KArg::U32(off(out)),
                    KArg::U32(meta.welch_batch as u32),
                    KArg::U32(meta.n_fft as u32),
                    KArg::U32(meta.n_segments as u32),
                    KArg::U32(meta.k as u32),
                    KArg::U32(meta.n_bins as u32),
                ]);
                (meta.welch_batch.max(1), 64)
            }
            _ => return,
        };

        append_kernel_launch(dev, kernel, list, &args, global, local);
    }

    fn read_outputs(
        &self,
        read_indices: Option<&[usize]>,
        mut read: impl FnMut(NodeId, usize) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let want: Vec<usize> = match read_indices {
            Some(ix) => ix.to_vec(),
            None => (0..self.output_ids.len()).collect(),
        };
        want.into_iter()
            .filter_map(|i| {
                let id = *self.output_ids.get(i)?;
                // Lane count, not element count: a complex output occupies 2 (C64)
                // / 4 (C128) f32 lanes per element, so reading `num_elements` would
                // truncate the readback to the real parts. One lane per element for
                // every other dtype, so this is `num_elements` there.
                let n = arena_lane_count(&self.graph.node(id).shape);
                Some(read(id, n))
            })
            .collect()
    }

    /// Deep copy for the runtime's executable cache: fresh state with the same
    /// legalized graph + uploaded params.
    pub fn clone_for_cache(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            params: self.params.clone(),
            output_ids: self.output_ids.clone(),
            output_dtypes: self.output_dtypes.clone(),
            rng: self.rng,
            active_extent: self.active_extent,
        }
    }
}

// ── kernel-argument helper ─────────────────────────────────────────────────

enum KArg {
    Ptr(*mut c_void),
    U32(u32),
    F32(f32),
}

impl KArg {
    /// `(argSize, pArgValue)` for `zeKernelSetArgumentValue`. The returned
    /// pointer borrows `self`, so it must be consumed before `self` drops —
    /// callers use it immediately inside the set-arg loop.
    fn as_arg(&self) -> (usize, *const c_void) {
        match self {
            KArg::Ptr(p) => (
                std::mem::size_of::<*mut c_void>(),
                p as *const *mut c_void as *const c_void,
            ),
            KArg::U32(v) => (4, v as *const u32 as *const c_void),
            KArg::F32(v) => (4, v as *const f32 as *const c_void),
        }
    }
}

fn append_kernel_launch(
    dev: &crate::device::OneApiDevice,
    kernel: crate::level_zero::KernelHandle,
    list: crate::level_zero::CommandListHandle,
    args: &[KArg],
    global: usize,
    local: u32,
) {
    unsafe {
        let _ = (dev.lib.kernel_set_group_size)(kernel, local, 1, 1);
        for (i, a) in args.iter().enumerate() {
            let (size, ptr) = a.as_arg();
            let _ = (dev.lib.kernel_set_argument_value)(kernel, i as u32, size, ptr);
        }
        let groups = crate::level_zero::GroupCount {
            group_count_x: ceil_div(global, local).max(1),
            group_count_y: 1,
            group_count_z: 1,
        };
        let _ = (dev.lib.command_list_append_launch_kernel)(
            list,
            kernel,
            &groups,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        );
        // Each kernel reads/writes the shared arena; barrier between launches.
        let _ = (dev.lib.command_list_append_barrier)(
            list,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        );
    }
}

// ── memory plan (f32-uniform bump allocator; same as rlx-vulkan) ───────────

fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        // Reshape / StopGradient, and identity Casts, alias the input slot.
        // float→int / →Bool casts get their own (f32-sized) slot + a kernel.
        let is_view = match &node.op {
            Op::Reshape { .. } | Op::StopGradient => true,
            Op::Cast { .. } => !cast_is_kernel(graph, node),
            _ => false,
        };
        if is_view {
            if let Some(in_id) = node.inputs.first() {
                if let Some(slot) = assignments.get(in_id) {
                    let aliased = slot.clone();
                    assignments.insert(node.id, aliased);
                    schedule.push(node.id);
                    continue;
                }
            }
        }
        // Slot length = (#f32 lanes) × 4. Real/int/bool tensors are ONE lane per
        // element; complex is simulated on lanes (C64 = 2, C128 = 4 df64), so a
        // complex slot must reserve 2N / 4N lanes or its kernels + readback would
        // overrun / truncate.
        let lanes = arena_lane_count(&node.shape);
        let bytes = (lanes * 4).max(4);
        let aligned = bytes.div_ceil(align) * align;
        assignments.insert(
            node.id,
            BufferSlot {
                offset: cursor,
                size: aligned,
            },
        );
        schedule.push(node.id);
        cursor += aligned;
    }
    MemoryPlan {
        arena_size: cursor.max(align),
        assignments,
        schedule,
    }
}

// ── small shape helpers (shared with the dispatch builder) ─────────────────

fn dims(graph: &Graph, id: NodeId) -> Vec<usize> {
    graph
        .node(id)
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(s) => *s,
            _ => 0,
        })
        .collect()
}

fn numel(d: &[usize]) -> usize {
    d.iter()
        .product::<usize>()
        .max(if d.is_empty() { 1 } else { 0 })
}

fn norm_axis(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis).max(0) as usize
    } else {
        (axis as usize).min(rank.saturating_sub(1))
    }
}

fn ceil_div(n: usize, d: u32) -> u32 {
    (n as u64).div_ceil(d as u64) as u32
}

/// Op ids for the forward `unary.cl` switch (Vulkan/oneAPI "gelu-first"
/// scheme). Canonical table lives in `rlx_ir::opcodes`.
fn act_id(a: Activation) -> u32 {
    a.opcode_gelu_first()
}

fn binop_id(op: rlx_ir::op::BinaryOp) -> u32 {
    op.opcode()
}

/// Widen a constant byte blob (any IR dtype) to f32 for the f32-uniform arena.
fn widen_const_to_f32(data: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I32 | DType::U32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::I16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        DType::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => data.iter().map(|&b| b as f32).collect(),
        // C64 = 2 interleaved f32 lanes `[re, im]`; the host already stores it as
        // f32 pairs, so widening is a pure reinterpret (N complex → 2N lanes).
        DType::C64 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // C128 = 4 f32 lanes df64 `[re_hi, re_lo, im_hi, im_lo]`, host-stored as
        // 2×f64 (16 B/elem). This is the df64 SPLIT boundary: each f64 `v` →
        // `hi=(f32)v` + `lo=(f32)(v-(f64)hi)`, so `(f64)hi + (f64)lo` reconstructs
        // `v` to double precision. Bit-identical to the shared
        // `rlx_runtime::backend::widen_bytes_to_f32` (the CPU↔GPU boundary the
        // complex-cast kernels round-trip against).
        DType::C128 => {
            let split = |v: f64| -> [f32; 2] {
                let hi = v as f32;
                let lo = (v - hi as f64) as f32;
                [hi, lo]
            };
            let mut out = Vec::with_capacity((data.len() / 16) * 4);
            for elem in data.chunks_exact(16) {
                let re = f64::from_le_bytes(elem[0..8].try_into().unwrap());
                let im = f64::from_le_bytes(elem[8..16].try_into().unwrap());
                out.extend_from_slice(&split(re));
                out.extend_from_slice(&split(im));
            }
            out
        }
    }
}
