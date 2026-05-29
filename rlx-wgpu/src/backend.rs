// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `WgpuExecutable` — compiles an rlx-ir Graph into a sequence of
//! kernel dispatches against a pre-allocated arena buffer.
//!
//! v2 op coverage: MatMul + element-wise families (Binary 7, Unary 12,
//! Compare 6, Where) + leaves. Anything else panics at compile time.

use std::collections::{HashMap, HashSet};

use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};

use crate::buffer::{Arena, plan_f32_uniform};
use crate::device::wgpu_device;
use crate::kernels::{
    ArgmaxParams, AttentionBwdParams, AttentionParams, BinaryParams, Conv1dParams, Conv2dParams,
    Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FusedResidualLnParams, FusedResidualLnTeeParams,
    FusedResidualRmsNormParams, GatherAxisParams, GatherBwdParams, GatherParams,
    GroupedMatmulParams, Kernel, LayerNormParams, MatmulParams, MatmulQkvParams,
    NarrowConcatParams, Pool1dParams, Pool2dParams, Pool3dParams, ReduceParams, RmsNormBwdParams,
    RopeBwdParams, RopeParams, SampleParams, ScatterAddParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WhereParams, argmax_kernel,
    attention_bwd_kernel, attention_kernel, binary_kernel, cast_f32_to_f16_kernel, compare_kernel,
    concat_kernel, conv1d_kernel, conv2d_kernel, conv3d_kernel, copy_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, elementwise_region_kernel,
    expand_kernel, fused_residual_ln_kernel, fused_residual_ln_tee_kernel,
    fused_residual_rms_norm_kernel, gather_axis_kernel, gather_backward_acc_kernel,
    gather_backward_zero_kernel, gather_kernel, grouped_matmul_kernel, layernorm_kernel,
    matmul_coop_f32_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel, matmul_f16w_kernel,
    matmul_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel, matmul_wide_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    rms_norm_backward_kernel, rms_norm_backward_param_kernel, rope_backward_kernel, rope_kernel,
    sample_kernel, scatter_add_kernel, selective_scan_kernel, softmax_kernel, topk_kernel,
    transpose_kernel, umap_knn_kernel, unary_kernel, where_kernel,
};
use rlx_ir::op::{ChainOperand, ChainStep};

/// Inner-FMA precision for matmul.
///   F32    — full f32 path (matmul.wgsl / matmul_wide.wgsl).
///   F16    — f16 multiply, f32 acc (matmul_f16_compute.wgsl).
///   Coop16 — cooperative-matrix 8×8 hardware GEMM
///            (matmul_coop16.wgsl, simdgroup_multiply_accumulate on
///             Apple, OpCooperativeMatrixMulAddKHR on Vulkan).
///            Requires M/N/K multiples of 8, b is a Param, and
///            both SHADER_F16 + EXPERIMENTAL_COOPERATIVE_MATRIX.
///            Caller must ensure A is mirrored to arena_f16 first
///            (the lowering inserts a `Step::CastF32ToF16` pre-pass).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulCompute {
    F32,
    F16,
    Coop16,
    /// Cooperative-matrix on Apple's `simdgroup_float8x8` — same hardware
    /// GEMM unit as Coop16 but with f32 operands and f32 accumulator.
    /// No precision loss vs F32 baseline; no f16 overflow risk in deep
    /// FFN sums. Used when alignment + features allow but the IR is f32.
    CoopF32,
}

/// f32 → f16 element-wise cast, mirroring an arena region into the
/// f16 shadow buffer. Used as a pre-pass before `matmul_coop16` so
/// the matmul's A operand (a runtime activation, not a Param) is
/// readable as f16.
///
/// Currently unused — the matmul_coop16 kernel stages A through
/// workgroup-shared memory directly from the f32 arena. Kept for
/// future paths that may want a one-shot cast (e.g. before a chain
/// of f16-only kernels operating on a fixed activation region).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct CastF32ToF16Params {
    pub src_off: u32, // f32-element offset into arena (also f16-element offset)
    pub len: u32,
    pub _p0: u32,
    pub _p1: u32,
}
unsafe impl bytemuck::Pod for CastF32ToF16Params {}
unsafe impl bytemuck::Zeroable for CastF32ToF16Params {}

/// One dispatch step in the compiled schedule.
///
/// `dead_code` is allowed at the enum level: several variants carry
/// fields (mask_buf, meta_idx, compute_precision discriminants) that
/// are only consulted at compile time during bind-group construction,
/// or are kept to extend buffer lifetimes (mask_buf). A few variants
/// (CastF32ToF16, Copy, the unreachable F16 compute_precision) are
/// retained for future paths.
#[allow(dead_code)]
enum Step {
    CastF32ToF16 {
        params: CastF32ToF16Params,
    },
    Matmul {
        m: u32,
        k: u32,
        n: u32,
        a_off_f32: u32,
        b_off_f32: u32,
        c_off_f32: u32,
        batch: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
        c_batch_stride: u32,
        has_bias: u32,
        bias_off_f32: u32,
        act_id: u32, // 0xFFFF = no activation
        // True iff input B is a Param node — i.e. a model weight that
        // doesn't change between `run()` calls. Read from the f16
        // shadow buffer (half memory bandwidth) when set + the device
        // exposes SHADER_F16. Set at compile time; consulted only by
        // the dispatch arm.
        b_is_param: bool,
        // Compute precision for the inner FMA. F32 = full precision
        // (the historical / default path). F16 = mixed-precision
        // (operands cast to f16, multiply in f16 for 2× ALU on Apple,
        // accumulator in f32). Set at compile time from the IR's
        // dtype after AutoMixedPrecision policy.
        compute_precision: MatmulCompute,
    },
    Binary {
        params: BinaryParams,
    },
    Compare {
        params: BinaryParams,
    },
    Unary {
        params: UnaryParams,
    },
    Where {
        params: WhereParams,
    },
    Reduce {
        params: ReduceParams,
    },
    Softmax {
        params: SoftmaxParams,
    },
    LayerNorm {
        params: LayerNormParams,
    },
    Cumsum {
        params: CumsumParams,
    },
    /// Native multi-kernel f32 FFT (gpu-fft dispatch strategy).
    FftGpu {
        src_off: u32,
        dst_off: u32,
        outer: u32,
        n: u32,
        inverse: u32,
        norm_scale: f32,
    },
    /// Explicit host FFT (D2H → rlx-cpu → H2D). Used when the native
    /// WGSL kernel cannot handle dtype / size / non-pow-2 constraints.
    FftHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype_tag: u32,
    },
    Copy {
        params: CopyParams,
    },
    /// PLAN L2 — fused N-ary element-wise region. Lowered from
    /// `Op::ElementwiseRegion` by `MarkElementwiseRegions`. Kernel
    /// interprets the chain encoding per-element (saves N kernel
    /// dispatches + N global-memory round-trips vs the decomposed
    /// atomic ops).
    ElementwiseRegion {
        params: ElementwiseRegionParams,
    },
    Transpose {
        params: TransposeParams,
        meta_idx: usize,
    },
    Narrow {
        params: NarrowConcatParams,
    },
    Concat {
        params: NarrowConcatParams,
    }, // one Step per input
    Gather {
        params: GatherParams,
    },
    GatherAxis {
        params: GatherAxisParams,
    },
    Attention {
        params: AttentionParams,
        mask_buf: Option<wgpu::Buffer>,
    },
    AttentionBackward {
        params: AttentionBwdParams,
        mask_buf: Option<wgpu::Buffer>,
    },
    Rope {
        params: RopeParams,
    },
    Expand {
        params: ExpandParams,
        meta_idx: usize,
    },
    Argmax {
        params: ArgmaxParams,
    },
    Pool2d {
        params: Pool2dParams,
    },
    Conv2d {
        params: Conv2dParams,
    },
    Pool1d {
        params: Pool1dParams,
    },
    Pool3d {
        params: Pool3dParams,
    },
    Conv1d {
        params: Conv1dParams,
    },
    Conv3d {
        params: Conv3dParams,
    },
    ScatterAdd {
        params: ScatterAddParams,
    },
    TopK {
        params: TopKParams,
    },
    GroupedMatmul {
        params: GroupedMatmulParams,
    },
    Sample {
        params: SampleParams,
    },
    SelectiveScan {
        params: SelectiveScanParams,
    },
    DequantMatmul {
        params: DequantMatmulParams,
    },
    /// GGUF K-quant — host fused dequant+matmul between GPU segments.
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        out_byte_off: u32,
    },
    /// GGUF K-quant — host fused dequant+grouped matmul between GPU segments.
    DequantGroupedMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        idx_byte_off: u32,
        out_byte_off: u32,
    },
    /// Gated-DeltaNet — host scan between GPU segments (qwen35 linear layers).
    GatedDeltaNet {
        q_byte_off: u32,
        k_byte_off: u32,
        v_byte_off: u32,
        g_byte_off: u32,
        beta_byte_off: u32,
        state_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
        use_carry: bool,
    },
    Llada2GroupLimitedGate {
        sig_byte_off: u32,
        route_byte_off: u32,
        out_byte_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
    },
    UmapKnn {
        params: UmapKnnParams,
    },
    /// Small-`n` host k-NN (partial arena read/write; avoids GPU launch overhead).
    UmapKnnHost {
        pairwise_byte_off: u32,
        out_byte_off: u32,
        n: u32,
        k: u32,
    },
    /// 3D Gaussian splat forward (CPU reference between segments).
    #[cfg(feature = "splat")]
    GaussianSplatRender {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        dst_byte_off: u32,
        dst_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    /// Backward splat — host round-trip via rlx-cpu/splat.
    #[cfg(feature = "splat")]
    GaussianSplatRenderBackward {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        d_loss_byte_off: u32,
        d_loss_len: u32,
        packed_byte_off: u32,
        packed_len: u32,
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
    #[cfg(feature = "splat")]
    GaussianSplatPrepare {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        meta_len: u32,
        prep_byte_off: u32,
        prep_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    #[cfg(feature = "splat")]
    GaussianSplatRasterize {
        prep_byte_off: u32,
        prep_len: u32,
        meta_byte_off: u32,
        meta_len: u32,
        dst_byte_off: u32,
        dst_len: u32,
        count: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    RmsNormBackwardInput {
        params: RmsNormBwdParams,
    },
    RmsNormBackwardGamma {
        params: RmsNormBwdParams,
    },
    RmsNormBackwardBeta {
        params: RmsNormBwdParams,
    },
    RopeBackward {
        params: RopeBwdParams,
    },
    CumsumBackward {
        params: CumsumBwdParams,
    },
    GatherBackward {
        params: GatherBwdParams,
    },
    FusedResidualLn {
        params: FusedResidualLnParams,
    },
    /// Split-write QKV matmul. Replaces a (FusedMatMulBiasAct → Narrow×3)
    /// pattern with one dispatch that writes Q, K, V into separate
    /// contiguous buffers from a single matmul pass. See
    /// `kernels/matmul_qkv.wgsl`.
    MatmulQkv {
        params: MatmulQkvParams,
        /// True → `matmul_qkv_coop_f32` (cooperative_matrix → simdgroup
        /// f32 hw GEMM). False → `matmul_qkv` (portable f32 tile).
        /// Both have identical bind groups and dispatch grid.
        coop: bool,
    },
    /// `fused_residual_ln_tee` — does (Add → LN) but writes the sum to
    /// a separate arena slot (the eliminated Add's old slot). Fires
    /// when the Add has multi-consumer downstream (vision pre-norm).
    FusedResidualLnTee {
        params: FusedResidualLnTeeParams,
    },
    FusedResidualRmsNorm {
        params: FusedResidualRmsNormParams,
    },
}

pub struct WgpuExecutable {
    graph: Graph,
    arena: Arena,
    schedule: Vec<Step>,
    input_offsets: HashMap<String, NodeId>,
    param_offsets: HashMap<String, NodeId>,
    /// One uniform buffer + bind group per dispatch step. Pre-allocated
    /// so run() just writes new bytes per step.
    uniforms: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    /// Per-step metadata storage buffers (only Transpose uses them).
    /// Indexed by `Step::Transpose.meta_idx`.
    meta_buffers: Vec<wgpu::Buffer>,

    // ── Lazy dynamic-shape state ─────────────────────────────────
    /// The originally-supplied graph (pre-resolution). Only set when
    /// the input graph contained `Dim::Dynamic` entries — otherwise
    /// `None` and the compiled fields above are authoritative. On each
    /// `run()` we infer a `DimBinding` from the live input data, and
    /// if it differs from `last_binding` we re-resolve + recompile.
    unresolved: Option<Graph>,
    last_binding: Option<DimBinding>,
    /// Buffered params written via `set_param` / `set_param_bytes`
    /// before the first `run()`. Replayed against the freshly compiled
    /// arena once shapes resolve.
    pending_params: HashMap<String, Vec<f32>>,
    pending_param_bytes: HashMap<String, Vec<u8>>,
    /// Active-extent hint (PLAN L1). When set + every Step in the
    /// safe set, both the uniform write and the dispatch workgroup
    /// count are scaled by `actual / upper`. Otherwise full-extent.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Skip-redundant-uniform-writes guard. Each `run()` would
    /// otherwise re-`queue.write_buffer` ~115 per-step uniforms (one
    /// per dispatched op in BERT) even when their bytes are identical
    /// to the previous call's. At small batches, that fixed write +
    /// staging-copy overhead is the dominant cost. We track the last
    /// active-extent value the uniforms were written for; subsequent
    /// `run()`s with the same `active_extent` (and `recompile`-clean
    /// schedule) skip the entire uniform-write loop. `None` ⇒ never
    /// written; `Some(x)` ⇒ uniforms hold params for active_extent=x.
    uniforms_active_extent: Option<Option<(usize, usize)>>,
    /// Per-`FftGpu` step: isolated uniform buffers + bind groups (one vec entry per op).
    fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources>,
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Coverage: simple element-wise + reductions + matmul + linalg
    /// + reductions/argmax/topk/sample + gather + conv + pool +
    /// scatter (zero output + scale num_updates) + macros gated to
    /// batch=1 (Attention, SelectiveScan).
    pub fn safe_for_active_extent(&self) -> bool {
        match self {
            Step::Binary { .. }
            | Step::Compare { .. }
            | Step::Unary { .. }
            | Step::Where { .. }
            | Step::Reduce { .. }
            | Step::Softmax { .. }
            | Step::LayerNorm { .. }
            | Step::FusedResidualLn { .. }
            | Step::FusedResidualLnTee { .. }
            | Step::FusedResidualRmsNorm { .. }
            | Step::Cumsum { .. }
            | Step::Copy { .. }
            | Step::ElementwiseRegion { .. }
            | Step::Argmax { .. }
            | Step::TopK { .. }
            | Step::Sample { .. }
            | Step::Gather { .. }
            | Step::GatherAxis { .. }
            | Step::GroupedMatmul { .. }
            | Step::DequantMatmul { .. }
            | Step::DequantMatmulGguf { .. }
            | Step::DequantGroupedMatmulGguf { .. }
            | Step::GatedDeltaNet { .. }
            | Step::Llada2GroupLimitedGate { .. }
            | Step::UmapKnn { .. }
            | Step::UmapKnnHost { .. }
            | Step::Conv1d { .. }
            | Step::Conv2d { .. }
            | Step::Conv3d { .. }
            | Step::Pool1d { .. }
            | Step::Pool2d { .. }
            | Step::Pool3d { .. }
            | Step::ScatterAdd { .. } => true,
            // FFT: full-extent transform per row, no active-extent
            // scaling. Marking true so a graph that mixes FFT with
            // active-extent-safe ops still gets the optimization for
            // the rest of the schedule.
            Step::FftGpu { .. } | Step::FftHost { .. } => true,
            // Matmul: c_batch_stride is set at compile time at full m,
            // independent of params.m. With scaled m, threads with
            // global_row >= m early-return; per-batch output offsets
            // stay correct. Safe at any batch.
            Step::Matmul { .. } => true,
            // Same active-extent reasoning as Matmul: per-batch output
            // strides are baked at compile time, scaling m only adjusts
            // the per-thread bound check.
            Step::MatmulQkv { .. } => true,
            Step::CastF32ToF16 { .. } => true,
            // Attention: WGSL kernel uses `seq_q_stride`/`seq_k_stride`
            // (full extent, set at compile time) for per-(batch, head)
            // offset math, and `params.seq_q`/`params.seq_k` for loop
            // bounds only. Scaling seq_q/seq_k shrinks the iteration
            // without corrupting per-head strides. Safe at any batch.
            Step::Attention { .. } => true,
            Step::AttentionBackward { .. } => true,
            // SelectiveScan: WGSL kernel uses `params.seq_stride`
            // (full extent, set at compile time) for per-batch stride
            // math; `params.seq` is the loop bound only. Safe at any
            // batch under active-extent scaling of seq.
            Step::SelectiveScan { .. } => true,
            // Narrow + Concat: kernel iterates `params.total` in
            // row-major order with outer as the leading dim. Scaling
            // total by actual/upper effectively scales outer by the
            // same factor (since total = outer * axis_size * inner).
            // Output positions past scaled_total stay untouched.
            // **Conservative assumption**: bucket axis is outer.
            // Cases where the bucket axis is the narrow/concat axis
            // itself are unsafe — fall back to full extent there.
            Step::Narrow { .. } => true,
            Step::Concat { .. } => true,
            // Rope: WGSL kernel uses `seq_stride` (full extent, set
            // at compile time) for per-batch buffer offset math and
            // explicit `batch` for index decomposition. `params.seq`
            // and `params.n_total` are runtime-scaled iteration
            // bounds. Safe at any batch.
            Step::Rope { .. } => true,
            // Transpose: precomputed `bucket_outermost` flag in
            // params (set to 1 at compile time iff `perm[0] == 0`).
            // Active path scales `out_total` by `actual / upper`
            // proportional to `out_dim_0`. Other transposes (where
            // bucket axis moves) fall back to full extent.
            Step::Transpose { params, .. } => params.bucket_outermost == 1,
            // Expand: same shape as Transpose. `bucket_outermost` is
            // 1 iff `in_dims[0] == out_dims[0]` (no broadcast at the
            // bucket axis).
            Step::Expand { params, .. } => params.bucket_outermost == 1,
            // Training backward ops: not used in inference; disable
            // active-extent fast path until individually audited.
            Step::RmsNormBackwardInput { .. }
            | Step::RmsNormBackwardGamma { .. }
            | Step::RmsNormBackwardBeta { .. }
            | Step::RopeBackward { .. }
            | Step::CumsumBackward { .. }
            | Step::GatherBackward { .. } => false,
            #[cfg(feature = "splat")]
            Step::GaussianSplatRender { .. }
            | Step::GaussianSplatRenderBackward { .. }
            | Step::GaussianSplatPrepare { .. }
            | Step::GaussianSplatRasterize { .. } => false,
        }
    }
}

/// Static-string label for each Step variant — used by the Perfetto
/// trace layer (PLAN L3) to mark per-step events without allocating.
fn fft_dtype_tag(dtype: rlx_ir::DType) -> u32 {
    match dtype {
        rlx_ir::DType::F32 => 0,
        rlx_ir::DType::F64 => 1,
        rlx_ir::DType::C64 => 2,
        other => panic!("rlx-wgpu Op::Fft: unsupported dtype {other:?}"),
    }
}

fn fft_dtype_from_tag(tag: u32) -> rlx_ir::DType {
    match tag {
        0 => rlx_ir::DType::F32,
        1 => rlx_ir::DType::F64,
        2 => rlx_ir::DType::C64,
        other => panic!("rlx-wgpu Op::Fft: bad dtype tag {other}"),
    }
}

fn step_name(step: &Step) -> &'static str {
    match step {
        Step::CastF32ToF16 { .. } => "cast_f32_to_f16",
        Step::Matmul { .. } => "matmul",
        Step::Binary { .. } => "binary",
        Step::Compare { .. } => "compare",
        Step::Unary { .. } => "unary",
        Step::Where { .. } => "where",
        Step::Reduce { .. } => "reduce",
        Step::Softmax { .. } => "softmax",
        Step::LayerNorm { .. } => "layer_norm",
        Step::Cumsum { .. } => "cumsum",
        Step::FftGpu { .. } => "fft_gpu",
        Step::FftHost { .. } => "fft_host",
        Step::Copy { .. } => "copy",
        Step::Transpose { .. } => "transpose",
        Step::Narrow { .. } => "narrow",
        Step::Concat { .. } => "concat",
        Step::Gather { .. } => "gather",
        Step::GatherAxis { .. } => "gather_axis",
        Step::Attention { .. } => "attention",
        Step::AttentionBackward { .. } => "attention_bwd",
        Step::Rope { .. } => "rope",
        Step::Expand { .. } => "expand",
        Step::Argmax { .. } => "argmax",
        Step::Pool2d { .. } => "pool2d",
        Step::Conv2d { .. } => "conv2d",
        Step::Pool1d { .. } => "pool1d",
        Step::Pool3d { .. } => "pool3d",
        Step::Conv1d { .. } => "conv1d",
        Step::Conv3d { .. } => "conv3d",
        Step::ScatterAdd { .. } => "scatter_add",
        Step::TopK { .. } => "topk",
        Step::GroupedMatmul { .. } => "grouped_matmul",
        Step::Sample { .. } => "sample",
        Step::SelectiveScan { .. } => "selective_scan",
        Step::DequantMatmul { .. } => "dequant_matmul",
        Step::DequantMatmulGguf { .. } => "dequant_matmul_gguf",
        Step::DequantGroupedMatmulGguf { .. } => "dequant_grouped_matmul_gguf",
        Step::GatedDeltaNet { .. } => "gated_delta_net",
        Step::Llada2GroupLimitedGate { .. } => "llada2_group_limited_gate",
        Step::UmapKnn { .. } => "umap_knn",
        Step::UmapKnnHost { .. } => "umap_knn_host",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRender { .. } => "gaussian_splat_render",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRenderBackward { .. } => "gaussian_splat_render_backward",
        #[cfg(feature = "splat")]
        Step::GaussianSplatPrepare { .. } => "gaussian_splat_prepare",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRasterize { .. } => "gaussian_splat_rasterize",
        Step::RmsNormBackwardInput { .. } => "rms_norm_backward_input",
        Step::RmsNormBackwardGamma { .. } => "rms_norm_backward_gamma",
        Step::RmsNormBackwardBeta { .. } => "rms_norm_backward_beta",
        Step::RopeBackward { .. } => "rope_backward",
        Step::CumsumBackward { .. } => "cumsum_backward",
        Step::GatherBackward { .. } => "gather_backward",
        Step::FusedResidualLn { .. } => "fused_residual_ln",
        Step::FusedResidualLnTee { .. } => "fused_residual_ln_tee",
        Step::FusedResidualRmsNorm { .. } => "fused_residual_rms_norm",
        Step::MatmulQkv { .. } => "matmul_qkv",
        Step::ElementwiseRegion { .. } => "elementwise_region",
    }
}

fn step_runs_on_host(step: &Step) -> bool {
    match step {
        Step::DequantMatmulGguf { .. }
        | Step::DequantGroupedMatmulGguf { .. }
        | Step::GatedDeltaNet { .. }
        | Step::Llada2GroupLimitedGate { .. }
        | Step::UmapKnnHost { .. }
        | Step::FftHost { .. } => true,
        #[cfg(feature = "splat")]
        Step::GaussianSplatRender { .. }
        | Step::GaussianSplatRenderBackward { .. }
        | Step::GaussianSplatPrepare { .. }
        | Step::GaussianSplatRasterize { .. } => true,
        _ => false,
    }
}

fn binary_op_id(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

fn compare_op_id(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

fn reduce_op_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

fn activation_op_id(act: Activation) -> u32 {
    match act {
        Activation::Relu => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::Exp => 3,
        Activation::Log => 4,
        Activation::Sqrt => 5,
        Activation::Rsqrt => 6,
        Activation::Neg => 7,
        Activation::Abs => 8,
        Activation::Gelu => 9,
        Activation::Silu => 10,
        Activation::GeluApprox => 11,
        Activation::Round => 12,
        Activation::Sin => 13,
        Activation::Cos => 14,
        Activation::Tan => 15,
        Activation::Atan => 16,
    }
}

impl WgpuExecutable {
    /// Resolve the deferred graph against bindings inferred from
    /// `inputs`, recompile the inner state if the bindings changed
    /// since the last call, and replay any pending params.
    fn lazy_compile_for_inputs(&mut self, inputs: &[(&str, &[f32])]) {
        let unresolved = self
            .unresolved
            .as_ref()
            .expect("lazy_compile_for_inputs called without an unresolved graph");
        let binding = infer_bindings_from_f32_inputs(unresolved, inputs)
            .expect("rlx-wgpu lazy compile: could not infer DimBinding from inputs");

        // No-op if shapes haven't changed since the last compile.
        if let Some(prev) = &self.last_binding
            && same_binding(prev, &binding)
        {
            return;
        }

        // Resolve and recompile.
        let resolved = bind_graph(unresolved, &binding);
        let original = self.unresolved.take();
        let pending_params = std::mem::take(&mut self.pending_params);
        let pending_bytes = std::mem::take(&mut self.pending_param_bytes);

        let fresh = Self::compile_static_inner(resolved);

        // Move the freshly-compiled fields into self, preserve the
        // unresolved+binding state for the next round.
        self.graph = fresh.graph;
        self.arena = fresh.arena;
        self.schedule = fresh.schedule;
        self.input_offsets = fresh.input_offsets;
        self.param_offsets = fresh.param_offsets;
        self.uniforms = fresh.uniforms;
        self.bind_groups = fresh.bind_groups;
        self.meta_buffers = fresh.meta_buffers;
        self.unresolved = original;
        self.last_binding = Some(binding);
        // Recompiled — uniforms are now empty buffers; force re-write
        // on next run().
        self.uniforms_active_extent = None;

        // Replay pending param uploads against the new arena.
        for (name, data) in pending_params {
            self.set_param(&name, &data);
        }
        for (name, data) in pending_bytes {
            self.set_param_bytes(&name, &data);
        }
    }

    /// Compile against an explicit `DimBinding`. Each `Dim::Dynamic`
    /// in the graph that maps to a symbol in `bindings` is replaced
    /// with `Dim::Static(size)` before the standard compile runs.
    /// Symbols not in the binding stay dynamic — and then `compile`
    /// will panic with the usual diagnostic.
    pub fn compile_with_bindings(graph: Graph, bindings: &DimBinding) -> Self {
        if bindings.is_empty() {
            return Self::compile(graph);
        }
        // Walk the graph and bind every node's shape.
        let mut fresh = Graph::new(&graph.name);
        for node in graph.nodes() {
            let bound = node.shape.bind(bindings);
            fresh.add_node(node.op.clone(), node.inputs.clone(), bound);
        }
        fresh.set_outputs(graph.outputs.clone());
        Self::compile(fresh)
    }

    pub fn compile(graph: Graph) -> Self {
        if has_dynamic_dims(&graph) {
            return Self::deferred(graph);
        }
        Self::compile_static_inner(graph)
    }

    /// Compile placeholder for a graph with `Dim::Dynamic` entries.
    /// The real compile happens on the first `run()` once input data
    /// reveals the symbol → size bindings. Buffered params (set via
    /// `set_param` / `set_param_bytes` before run) are replayed.
    fn deferred(graph: Graph) -> Self {
        let dev = wgpu_device().expect("rlx-wgpu: no compatible adapter found");
        // Minimal valid arena buffer. Replaced on first run().
        let placeholder = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu deferred placeholder"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let arena = Arena {
            buffer: placeholder,
            f16_buffer: None,
            offsets: HashMap::new(),
            lens: HashMap::new(),
            size: 0,
        };
        Self {
            graph: graph.clone(),
            arena,
            schedule: Vec::new(),
            input_offsets: HashMap::new(),
            param_offsets: HashMap::new(),
            uniforms: Vec::new(),
            bind_groups: Vec::new(),
            meta_buffers: Vec::new(),
            unresolved: Some(graph),
            last_binding: None,
            pending_params: HashMap::new(),
            pending_param_bytes: HashMap::new(),
            active_extent: None,
            uniforms_active_extent: None,
            fft_gpu_steps: Vec::new(),
        }
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every Step is in the safe set. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    fn compile_static_inner(graph: Graph) -> Self {
        let dev = wgpu_device().expect("rlx-wgpu: no compatible adapter found");

        // Decompose composed/fused ops (FusedMatMulBiasAct, LoraMatMul,
        // FusedAttentionBlock, FusedTransformerLayer, ...) into primitive
        // sequences before memory planning so every intermediate gets a
        // regular arena slot. CPU/Metal/MLX lower the fused variants
        // directly with bespoke kernels; we choose simplicity over peak
        // throughput here.
        let graph = crate::unfuse::unfuse(graph);

        // f32-uniform slots + liveness reuse (pairwise `[n,n]` graphs).
        let plan = plan_f32_uniform(&graph, 16);
        let mut arena = Arena::from_plan(&dev.device, &plan);
        // Override slot lengths with the actual elem*4 byte counts so
        // readback returns the right element count (slots may be
        // padded for alignment).
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0);
            arena.set_actual_len(node.id, elems * 4);
        }

        // Initialize Constants directly into the arena.
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let bytes_to_write = data.len().min(arena.len_of(node.id));
                dev.queue.write_buffer(
                    &arena.buffer,
                    arena.offset(node.id) as u64,
                    &data[..bytes_to_write],
                );
            }
        }

        let mut input_offsets = HashMap::new();
        let mut param_offsets = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_offsets.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_offsets.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        let mm_k = matmul_kernel(&dev.device);
        let mm_w = matmul_wide_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_kernel(&dev.device);
        let mm_cast = cast_f32_to_f16_kernel(&dev.device);
        let bk = binary_kernel(&dev.device);
        let uk = unary_kernel(&dev.device);
        let ck = compare_kernel(&dev.device);
        let wk = where_kernel(&dev.device);

        let mut schedule = Vec::new();
        let mut uniforms = Vec::new();
        let mut bind_groups = Vec::new();
        let mut fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources> = Vec::new();
        let mut gguf_host_pad: Option<(wgpu::Buffer, wgpu::BindGroup)> = None;
        let mut meta_buffers: Vec<wgpu::Buffer> = Vec::new();

        // Detect (FusedMatMulBiasAct → Narrow×3) split-QKV pattern. Returns
        // a map parent_node_id → (q_narrow_id, k_narrow_id, v_narrow_id).
        // The matmul_qkv kernel collapses the matmul + 3 narrows into one
        // dispatch by routing each output column to the right Q/K/V sink.
        //
        // CRITICAL: only mark a pattern site for elision when the parent
        // FMB will actually take the MatmulQkv path (which only fires
        // for F32 compute precision). For Coop16/CoopF32-eligible FMBs,
        // those kernels write to the FMB's *own* output slot, NOT the
        // 3 narrow slots — skipping the narrows would leave Q/K/V
        // uninitialized and attention would read garbage. Predict the
        // compute precision the FMB will receive; only skip when F32.
        let mut qkv_split: HashMap<NodeId, (NodeId, NodeId, NodeId)> = HashMap::new();
        for (parent_id, qkv) in detect_split_qkv_pattern(&graph) {
            let parent = graph.node(parent_id);
            // Mirror the lowering's precision derivation. FMB inputs:
            // [a, w, bias]; we need (m, k, n) to query.
            let a_id = parent.inputs[0];
            let b_id = parent.inputs[1];
            let a_dims = graph.node(a_id).shape.dims();
            let b_dims = graph.node(b_id).shape.dims();
            let out_dims = parent.shape.dims();
            let (m, k, n) =
                if a_dims.len() >= 2 && b_dims.len() == 2 && out_dims.len() == a_dims.len() {
                    let leading: usize = a_dims[..a_dims.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let m_inner = a_dims[a_dims.len() - 2].unwrap_static();
                    let k_inner = a_dims[a_dims.len() - 1].unwrap_static();
                    let n_inner = b_dims[1].unwrap_static();
                    ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
                } else if a_dims.len() == 2 && b_dims.len() == 2 {
                    (
                        a_dims[0].unwrap_static() as u32,
                        a_dims[1].unwrap_static() as u32,
                        b_dims[1].unwrap_static() as u32,
                    )
                } else {
                    continue; // unusual shape — let the regular FMB path handle
                };
            let cp = derive_matmul_compute(&dev.device, &graph, a_id, b_id, m, k, n);
            // F32 → matmul_qkv. CoopF32 → matmul_qkv_coop_f32. Both write
            // Q/K/V into the narrow output slots, so the narrows can be
            // elided. Coop16 still falls back to FMB+narrows (kernel
            // would need an f16-acc variant; deferred).
            if cp == MatmulCompute::F32 || cp == MatmulCompute::CoopF32 {
                qkv_split.insert(parent_id, qkv);
            }
        }
        let qkv_skip_narrows: HashSet<NodeId> = qkv_split
            .values()
            .flat_map(|&(q, k, v)| [q, k, v])
            .collect();

        // Detect (Add → LayerNorm) where Add has multi-consumer downstream.
        // The standard `FuseResidualLN` pass declines to fuse these (its
        // single-consumer guard forces materializing the sum); we collapse
        // them here at the wgpu lowering level via `Step::FusedResidualLnTee`.
        // Returns:
        //   ln_to_tee: ln_id  → (h, delta, gamma, beta, sum_arena_id)
        //   skip_adds: { add_id }  — these Add nodes are computed by the
        //                            tee step; their normal Step emission
        //                            is suppressed.
        let (ln_to_tee, skip_adds) = detect_residual_ln_tee_pattern(&graph);

        let emit_uniform = |size: usize| -> wgpu::Buffer {
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu uniform"),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        for node in graph.nodes() {
            // Helpers — capture device + arena into closures isn't
            // ergonomic in the loop, so inline the bind-group build
            // when each step is emitted below.
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::MatMul => {
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_shape = graph.node(a_id).shape.dims();
                    let b_shape = graph.node(b_id).shape.dims();
                    let out_shape = node.shape.dims();
                    // Three patterns:
                    //   • 2D×2D                              → batch=1
                    //   • [..,M,K] × [K,N]  (broadcast rhs)  → batch=1, flatten leading into M
                    //   • [..,M,K] × [..,K,N] (matched batch)→ batch=prod(leading), per-batch strides
                    let (m, k, n, batch, a_bs, b_bs, c_bs) = if a_shape.len() == 2
                        && b_shape.len() == 2
                        && out_shape.len() == 2
                    {
                        (
                            a_shape[0].unwrap_static() as u32,
                            a_shape[1].unwrap_static() as u32,
                            b_shape[1].unwrap_static() as u32,
                            1u32,
                            0u32,
                            0u32,
                            0u32,
                        )
                    } else if a_shape.len() >= 2
                        && b_shape.len() == 2
                        && out_shape.len() == a_shape.len()
                    {
                        let leading: usize = a_shape[..a_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .product();
                        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                        let n_inner = b_shape[1].unwrap_static();
                        (
                            (leading * m_inner) as u32,
                            k_inner as u32,
                            n_inner as u32,
                            1u32,
                            0u32,
                            0u32,
                            0u32,
                        )
                    } else if a_shape.len() == b_shape.len()
                        && a_shape.len() >= 3
                        && out_shape.len() == a_shape.len()
                    {
                        // True batched: leading dims must match.
                        let leading_a: Vec<usize> = a_shape[..a_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .collect();
                        let leading_b: Vec<usize> = b_shape[..b_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .collect();
                        if leading_a != leading_b {
                            panic!(
                                "rlx-wgpu MatMul: batched shape mismatch \
                                    a_leading={leading_a:?} b_leading={leading_b:?}"
                            );
                        }
                        let b_count: usize = leading_a.iter().product();
                        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
                        (
                            m_inner as u32,
                            k_inner as u32,
                            n_inner as u32,
                            b_count as u32,
                            (m_inner * k_inner) as u32,
                            (k_inner * n_inner) as u32,
                            (m_inner * n_inner) as u32,
                        )
                    } else {
                        panic!(
                            "rlx-wgpu MatMul: unsupported shapes a={a_shape:?} b={b_shape:?} \
                                out={out_shape:?} (supported: 2D×2D, [..,M,K]×[K,N], [..,M,K]×[..,K,N])"
                        );
                    };
                    let b_is_param = traces_to_param(&graph, b_id);
                    let compute_precision =
                        derive_matmul_compute(&dev.device, &graph, a_id, b_id, m, k, n);
                    // No cast pre-pass needed for Coop16 anymore — the
                    // kernel stages A through workgroup-shared memory
                    // directly from the f32 arena.
                    let _ = mm_cast;
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32: (arena.offset(a_id) / 4) as u32,
                        b_off_f32: (arena.offset(b_id) / 4) as u32,
                        c_off_f32: (arena.offset(node.id) / 4) as u32,
                        has_bias: 0,
                        bias_off_f32: 0,
                        act_id: 0xFFFF,
                        b_is_param,
                        compute_precision,
                    });
                    let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                    let bg = build_matmul_bind_group(
                        &dev.device,
                        mm_k,
                        mm_w,
                        &mm_f16w,
                        &mm_f16c,
                        &mm_coop,
                        &mm_coop_f32,
                        &arena,
                        &u,
                        b_is_param,
                        compute_precision,
                    );
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Binary(bop) => {
                    // Skip emit when this Add is consumed by a downstream
                    // FRLTee — the tee step writes the sum to this node's
                    // arena slot directly. Subsequent consumers read the
                    // same slot and find correct data.
                    if skip_adds.contains(&node.id) {
                        continue;
                    }
                    require_equal_shapes(&graph, &node.inputs, "Binary");
                    let p = BinaryParams {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: binary_op_id(*bop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Binary { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two(&dev.device, bk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Compare(cop) => {
                    require_equal_shapes(&graph, &node.inputs, "Compare");
                    let p = BinaryParams {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: compare_op_id(*cop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Compare { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two(&dev.device, ck, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Activation(act) => {
                    let p = UnaryParams {
                        n: elems,
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: activation_op_id(*act),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Unary { params: p });
                    let u = emit_uniform(std::mem::size_of::<UnaryParams>());
                    let bg = bind_two(&dev.device, uk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Where => {
                    let p = WhereParams {
                        n: elems,
                        cond_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        x_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        y_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Where { params: p });
                    let u = emit_uniform(std::mem::size_of::<WhereParams>());
                    let bg = bind_two(&dev.device, wk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // fixed-size u32 buffer; one uniform per region.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-wgpu ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Use UnfuseElementwiseRegions to fall back.",
                            chain.len()
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = (arena.offset(id) / 4) as u32;
                    }
                    let encode_operand = |op: &ChainOperand| -> u32 {
                        match *op {
                            ChainOperand::Input(i) => i & 0x7FFF_FFFFu32,
                            ChainOperand::Step(i) => 0x8000_0000u32 | (i & 0x7FFF_FFFFu32),
                        }
                    };
                    let act_sub = |a: Activation| match a {
                        Activation::Gelu => 0u32,
                        Activation::GeluApprox => 1,
                        Activation::Silu => 2,
                        Activation::Relu => 3,
                        Activation::Sigmoid => 4,
                        Activation::Tanh => 5,
                        Activation::Exp => 6,
                        Activation::Log => 7,
                        Activation::Sqrt => 8,
                        Activation::Rsqrt => 9,
                        Activation::Neg => 10,
                        Activation::Abs => 11,
                        Activation::Round => 12,
                        Activation::Sin => 13,
                        Activation::Cos => 14,
                        Activation::Tan => 15,
                        Activation::Atan => 16,
                    };
                    let bin_sub = |b: BinaryOp| match b {
                        BinaryOp::Add => 0u32,
                        BinaryOp::Sub => 1,
                        BinaryOp::Mul => 2,
                        BinaryOp::Div => 3,
                        BinaryOp::Max => 4,
                        BinaryOp::Min => 5,
                        BinaryOp::Pow => 6,
                    };
                    let cmp_sub = |c: CmpOp| match c {
                        CmpOp::Eq => 0u32,
                        CmpOp::Ne => 1,
                        CmpOp::Lt => 2,
                        CmpOp::Le => 3,
                        CmpOp::Gt => 4,
                        CmpOp::Ge => 5,
                    };
                    let mut chain_enc = [0u32; 128];
                    for (k, step) in chain.iter().enumerate() {
                        let base = k * 4;
                        let (kind, sub, lhs, rhs) = match step {
                            ChainStep::Activation(a, src) => {
                                (0u32, act_sub(*a), encode_operand(src), 0u32)
                            }
                            ChainStep::Cast(_, src) => (1u32, 0, encode_operand(src), 0u32),
                            ChainStep::Binary(op, l, r) => {
                                (2u32, bin_sub(*op), encode_operand(l), encode_operand(r))
                            }
                            ChainStep::Compare(op, l, r) => {
                                (3u32, cmp_sub(*op), encode_operand(l), encode_operand(r))
                            }
                            ChainStep::Where(c, t, f) =>
                            // Pack 3 operands into the 4-u32 step:
                            // op_sub=cond, lhs=on_true, rhs=on_false.
                            {
                                (
                                    4u32,
                                    encode_operand(c),
                                    encode_operand(t),
                                    encode_operand(f),
                                )
                            }
                        };
                        chain_enc[base] = kind;
                        chain_enc[base + 1] = sub;
                        chain_enc[base + 2] = lhs;
                        chain_enc[base + 3] = rhs;
                    }
                    let p = ElementwiseRegionParams {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                        input_modulus: *input_modulus,
                    };
                    schedule.push(Step::ElementwiseRegion { params: p });
                    let ek = elementwise_region_kernel(&dev.device);
                    // STORAGE (not UNIFORM) — the WGSL params struct
                    // contains `array<u32, N>` arrays whose 4-byte
                    // stride violates uniform's 16-byte stride rule.
                    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu region params"),
                        size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg = bind_two(&dev.device, ek, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Reduce {
                    op: rop,
                    axes,
                    keep_dim: _,
                } => {
                    // v3: only reduce-last-axis is supported. The
                    // kernel reads inner contiguously and writes one
                    // f32 per output row.
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let last = in_shape.len() - 1;
                    if axes.as_slice() != [last] {
                        panic!(
                            "rlx-wgpu Reduce: only last-axis is wired \
                             (got axes={axes:?}, rank={})",
                            in_shape.len()
                        );
                    }
                    let inner = in_shape[last].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = ReduceParams {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: reduce_op_id(*rop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Reduce { params: p });
                    let rk = reduce_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ReduceParams>());
                    let bg = bind_two(&dev.device, rk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Softmax { axis } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let last = (in_shape.len() - 1) as i32;
                    if *axis != -1 && *axis != last {
                        panic!("rlx-wgpu Softmax: only last-axis wired (got axis={axis})");
                    }
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = SoftmaxParams {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Softmax { params: p });
                    let sk = softmax_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SoftmaxParams>());
                    let bg = bind_two(&dev.device, sk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let is_layer_norm = matches!(&node.op, Op::LayerNorm { .. });

                    // FRLTee fast path: if this LN is the head of a
                    // (multi-consumer Add → LN) pattern, emit one
                    // `Step::FusedResidualLnTee` that writes the sum to
                    // the eliminated Add's arena slot AND the LN result
                    // to this LN's slot. The Add itself is skipped
                    // upstream (`skip_adds`).
                    if is_layer_norm
                        && let Some(&(h_id, delta_id, gamma_id, beta_id, sum_id)) =
                            ln_to_tee.get(&node.id)
                    {
                        let p = FusedResidualLnTeeParams {
                            outer,
                            inner,
                            in_off: (arena.offset(h_id) / 4) as u32,
                            residual_off: (arena.offset(delta_id) / 4) as u32,
                            bias_off: 0, // FRLTee currently no-bias only
                            gamma_off: (arena.offset(gamma_id) / 4) as u32,
                            beta_off: (arena.offset(beta_id) / 4) as u32,
                            sum_off: (arena.offset(sum_id) / 4) as u32,
                            ln_out_off: (arena.offset(node.id) / 4) as u32,
                            eps_bits: eps.to_bits(),
                            has_bias: 0,
                            _p0: 0,
                        };
                        schedule.push(Step::FusedResidualLnTee { params: p });
                        let frtk = fused_residual_ln_tee_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<FusedResidualLnTeeParams>());
                        let bg = bind_two(&dev.device, frtk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                        continue;
                    }

                    let gamma_id = node.inputs[1];
                    // beta is the third input for LayerNorm; RmsNorm
                    // ignores it (kernel branch on `op` skips the read).
                    let beta_id = if is_layer_norm && node.inputs.len() >= 3 {
                        node.inputs[2]
                    } else {
                        // Use gamma's offset as a benign placeholder;
                        // the RmsNorm kernel branch never reads it.
                        gamma_id
                    };
                    let p = LayerNormParams {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        gamma_off: (arena.offset(gamma_id) / 4) as u32,
                        beta_off: (arena.offset(beta_id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        op: if is_layer_norm { 0 } else { 1 },
                    };
                    schedule.push(Step::LayerNorm { params: p });
                    let lk = layernorm_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNormParams>());
                    let bg = bind_two(&dev.device, lk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Reshape { .. } | Op::Cast { .. } => {
                    // No-op: memory planner view-aliased this slot.
                }

                Op::Transpose { perm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let out_shape = node.shape.dims();
                    let rank = perm.len();
                    if rank != in_shape.len() || rank != out_shape.len() {
                        panic!("rlx-wgpu Transpose: rank mismatch");
                    }
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    let out_dims: Vec<u32> =
                        out_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Input cumulative strides (row-major).
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
                    }
                    // For each *output* axis i, the corresponding input
                    // axis is perm[i] — its stride is in_strides[perm[i]].
                    let strides_for_out: Vec<u32> =
                        (0..rank).map(|i| in_strides[perm[i]]).collect();

                    // Build meta buffer: dims (rank u32s) + strides (rank u32s).
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu transpose meta"),
                        size: (meta_data.len() * 4).max(4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue
                        .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta_buf);

                    // PLAN L1: precompute "bucket axis stays at out
                    // axis 0" flag from perm. When `perm[0] == 0`,
                    // active-extent scaling of `out_total` is safe.
                    let bucket_outermost = if perm[0] == 0 { 1u32 } else { 0u32 };
                    let p = TransposeParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        bucket_outermost,
                        out_dim_0: out_dims[0],
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Transpose {
                        params: p,
                        meta_idx,
                    });
                    let tk = transpose_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<TransposeParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu transpose bg"),
                        layout: &tk.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: arena.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Narrow { axis, start, len } => {
                    // Part of a split-QKV pattern: the parent FMB has been
                    // (or will be) replaced by Step::MatmulQkv that writes
                    // directly into this narrow's arena slot. Skip the
                    // narrow's own dispatch.
                    if qkv_skip_narrows.contains(&node.id) {
                        continue;
                    }
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let outer: u32 = in_shape[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = in_shape[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_in = in_shape[*axis].unwrap_static() as u32;
                    let p = NarrowConcatParams {
                        total: elems,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    };
                    schedule.push(Step::Narrow { params: p });
                    let nk = narrow_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                    let bg = bind_two(&dev.device, nk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Concat { axis } => {
                    let out_shape = node.shape.dims();
                    let outer: u32 = out_shape[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = out_shape[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_out = out_shape[*axis].unwrap_static() as u32;

                    let mut start_pos: u32 = 0;
                    for &in_id in &node.inputs {
                        let in_shape = graph.node(in_id).shape.dims();
                        let axis_in = in_shape[*axis].unwrap_static() as u32;
                        let in_total: u32 =
                            in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let p = NarrowConcatParams {
                            total: in_total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size: axis_out,
                            start: start_pos,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        };
                        schedule.push(Step::Concat { params: p });
                        let cck = concat_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                        let bg = bind_two(&dev.device, cck, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                        start_pos += axis_in;
                    }
                }

                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale: _,
                    attn_logit_softcap: _,
                } => {
                    // v5: rank-4 [B, H, S, D] inputs only. SlidingWindow
                    // synthesizes a Custom mask host-side.
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    // Accept either rank-4 [B, H, S, D] or rank-3 [B*H, S, D]
                    // (the latter is what BERT-flavored builders emit). For
                    // rank-3 we treat the leading dim as `batch * heads`,
                    // setting heads = num_heads from the Op so the kernel's
                    // (b, h) indexing folds back to the right offset.
                    let h = *num_heads as u32;
                    let hd = *head_dim as u32;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            q_shape[0].unwrap_static() as u32,
                            q_shape[1].unwrap_static() as u32,
                            q_shape[2].unwrap_static() as u32,
                            k_shape[2].unwrap_static() as u32,
                        ),
                        3 => {
                            // Two rank-3 layouts coexist:
                            //   [B, S, H·D] — transpose-elided layout
                            //   [B·H, S, D] — canonical compacted layout
                            // Distinguish by last-dim: if it equals H·D
                            // (the per-token feature width) it's [B, S, H·D];
                            // otherwise it's [B·H, S, D].
                            let last = q_shape[2].unwrap_static() as u32;
                            if last == h * hd {
                                // [B, S, H·D]: leading = B, seq = S
                                (
                                    q_shape[0].unwrap_static() as u32,
                                    h,
                                    q_shape[1].unwrap_static() as u32,
                                    k_shape[1].unwrap_static() as u32,
                                )
                            } else {
                                // [B·H, S, D]: leading must be divisible by H
                                let leading = q_shape[0].unwrap_static() as u32;
                                if !leading.is_multiple_of(h) {
                                    panic!(
                                        "rlx-wgpu Attention: rank-3 leading dim {leading} \
                                            not divisible by num_heads {h} (and last dim \
                                            {last} ≠ H·D = {})",
                                        h * hd
                                    );
                                }
                                (
                                    leading / h,
                                    h,
                                    q_shape[1].unwrap_static() as u32,
                                    k_shape[1].unwrap_static() as u32,
                                )
                            }
                        }
                        other => panic!(
                            "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V \
                                         inputs supported (got rank {other})"
                        ),
                    };
                    let scale = 1.0_f32 / (hd as f32).sqrt();

                    let (mask_kind_id, mask_off, mask_buf, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, None, 0u32),
                        MaskKind::Causal => (1u32, 0u32, None, 0u32),
                        MaskKind::Custom | MaskKind::Bias => {
                            let m_id = node.inputs[3];
                            (2u32, (arena.offset(m_id) / 4) as u32, None, 0u32)
                        }
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, None, *w as u32),
                    };

                    // Mask address strides. For Custom masks, derive from
                    // the mask's IR shape so the kernel can broadcast a
                    // [B, S] padding mask without materializing the full
                    // [B, H, S_q, S_k] expansion. Other mask kinds use
                    // canonical [B, H, S_q, S_k] strides (the kernel's
                    // mask_partial computation is harmless when not read).
                    struct MStrides {
                        b: u32,
                        h: u32,
                        q: u32,
                        k: u32,
                    }
                    let mask_strides = if mask_kind_id == 2u32 {
                        let m_dims = graph.node(node.inputs[3]).shape.dims();
                        let dim = |i: usize| m_dims[i].unwrap_static() as u32;
                        match m_dims.len() {
                            2 => MStrides {
                                b: dim(1),
                                h: 0,
                                q: 0,
                                k: 1,
                            },
                            3 => MStrides {
                                b: dim(1) * dim(2),
                                h: 0,
                                q: dim(2),
                                k: 1,
                            },
                            4 => MStrides {
                                b: dim(1) * dim(2) * dim(3),
                                h: dim(2) * dim(3),
                                q: dim(3),
                                k: 1,
                            },
                            _ => MStrides {
                                b: heads * seq_q * seq_k,
                                h: seq_q * seq_k,
                                q: seq_k,
                                k: 1,
                            },
                        }
                    } else {
                        MStrides {
                            b: heads * seq_q * seq_k,
                            h: seq_q * seq_k,
                            q: seq_k,
                            k: 1,
                        }
                    };

                    // Compute per-axis strides from input shape. Supports
                    // both [B, H, S, D] (rank-4) / [B*H, S, D] (rank-3)
                    // layouts (the canonical post-`unfuse` form) and the
                    // future [B, S, H, D] / [B, S, H·D] layout that
                    // skips the unfuse transposes. Detection: if the
                    // input shape's rank-3 last-dim equals H·D, treat
                    // as [B, S, H·D] = [B, S, H, D]; otherwise canonical.
                    let infer_strides =
                        |shape: &[rlx_ir::shape::Dim], seq_extent: u32| -> (u32, u32, u32) {
                            let last = shape[shape.len() - 1].unwrap_static() as u32;
                            if shape.len() == 3 && last == (heads * hd) {
                                // [B, S, H·D] viewed as [B, S, H, D]
                                let head_dim_total = heads * hd;
                                (seq_extent * head_dim_total, hd, head_dim_total)
                            } else {
                                // Canonical [B, H, S, D] (or rank-3 [B*H, S, D])
                                (heads * seq_extent * hd, seq_extent * hd, hd)
                            }
                        };
                    let (q_b, q_h, q_s) = infer_strides(q_shape, seq_q);
                    let (k_b, k_h, k_s) = infer_strides(k_shape, seq_k);
                    let v_shape = graph.node(v_id).shape.dims();
                    let (v_b, v_h, v_s) = infer_strides(v_shape, seq_k);
                    let out_shape = node.shape.dims();
                    let (o_b, o_h, o_s) = infer_strides(out_shape, seq_q);
                    let p = AttentionParams {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        // Mask strides — derive from the mask's IR shape:
                        //   [B, S]:           (mb=S,        mh=0,    mq=0,   mk=1)
                        //   [B, S_q, S_k]:    (mb=S_q·S_k,  mh=0,    mq=S_k, mk=1)
                        //   [B, H, S_q, S_k]: (mb=H·S_q·S_k mh=S_q·S_k mq=S_k mk=1)
                        // Stride 0 means the kernel broadcasts across that
                        // axis (reads the same element for every value of
                        // the index). Lets us skip the Expand pre-pass that
                        // unfuse used to emit per attention block.
                        seq_q_stride: mask_strides.q,
                        seq_k_stride: mask_strides.k,
                        mask_batch_stride: mask_strides.b,
                        mask_head_stride: mask_strides.h,
                        _pad_mask_0: 0,
                        _pad_mask_1: 0,
                        _pad_mask_2: 0,
                        q_batch_stride: q_b,
                        q_head_stride: q_h,
                        q_seq_stride: q_s,
                        _pad_q: 0,
                        k_batch_stride: k_b,
                        k_head_stride: k_h,
                        k_seq_stride: k_s,
                        _pad_k: 0,
                        v_batch_stride: v_b,
                        v_head_stride: v_h,
                        v_seq_stride: v_s,
                        _pad_v: 0,
                        o_batch_stride: o_b,
                        o_head_stride: o_h,
                        o_seq_stride: o_s,
                        _pad_o: 0,
                    };
                    let _ = num_heads;
                    schedule.push(Step::Attention {
                        params: p,
                        mask_buf,
                    });
                    let ak = attention_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AttentionParams>());
                    let bg = bind_two(&dev.device, ak, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::AttentionBackward {
                    num_heads: _,
                    head_dim,
                    mask_kind,
                    wrt,
                } => {
                    use rlx_ir::op::AttentionBwdWrt;
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    let hd = *head_dim as u32;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            q_shape[0].unwrap_static() as u32,
                            q_shape[1].unwrap_static() as u32,
                            q_shape[2].unwrap_static() as u32,
                            k_shape[2].unwrap_static() as u32,
                        ),
                        3 => {
                            let h = q_shape[2].unwrap_static() as u32 / hd;
                            (
                                q_shape[0].unwrap_static() as u32 / h,
                                h,
                                q_shape[1].unwrap_static() as u32,
                                k_shape[1].unwrap_static() as u32,
                            )
                        }
                        other => panic!(
                            "rlx-wgpu AttentionBackward: only rank-3/4 Q,K,V (got rank {other})"
                        ),
                    };
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, mask_buf, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, None, 0u32),
                        MaskKind::Causal => (1u32, 0u32, None, 0u32),
                        MaskKind::Custom => {
                            (2u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32)
                        }
                        MaskKind::Bias => {
                            (4u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32)
                        }
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, None, *w as u32),
                    };
                    struct MStrides {
                        b: u32,
                        h: u32,
                        q: u32,
                        k: u32,
                    }
                    let mask_strides = if mask_kind_id == 2 || mask_kind_id == 4 {
                        let m_dims = graph.node(node.inputs[4]).shape.dims();
                        let dim = |i: usize| m_dims[i].unwrap_static() as u32;
                        match m_dims.len() {
                            2 => MStrides {
                                b: dim(1),
                                h: 0,
                                q: 0,
                                k: 1,
                            },
                            3 => MStrides {
                                b: dim(1) * dim(2),
                                h: 0,
                                q: dim(2),
                                k: 1,
                            },
                            4 => MStrides {
                                b: dim(1) * dim(2) * dim(3),
                                h: dim(2) * dim(3),
                                q: dim(3),
                                k: 1,
                            },
                            _ => MStrides {
                                b: heads * seq_q * seq_k,
                                h: seq_q * seq_k,
                                q: seq_k,
                                k: 1,
                            },
                        }
                    } else {
                        MStrides {
                            b: heads * seq_q * seq_k,
                            h: seq_q * seq_k,
                            q: seq_k,
                            k: 1,
                        }
                    };
                    let infer_strides =
                        |shape: &[rlx_ir::shape::Dim], seq_extent: u32| -> (u32, u32, u32) {
                            let last = shape[shape.len() - 1].unwrap_static() as u32;
                            if shape.len() == 3 && last == (heads * hd) {
                                let head_dim_total = heads * hd;
                                (seq_extent * head_dim_total, hd, head_dim_total)
                            } else {
                                (heads * seq_extent * hd, seq_extent * hd, hd)
                            }
                        };
                    let (q_b, q_h, q_s) = infer_strides(q_shape, seq_q);
                    let (k_b, k_h, k_s) = infer_strides(k_shape, seq_k);
                    let v_shape = graph.node(v_id).shape.dims();
                    let (v_b, v_h, v_s) = infer_strides(v_shape, seq_k);
                    let out_shape = node.shape.dims();
                    let out_seq = match wrt {
                        AttentionBwdWrt::Query => seq_q,
                        AttentionBwdWrt::Key | AttentionBwdWrt::Value => seq_k,
                    };
                    let (o_b, o_h, o_s) = infer_strides(out_shape, out_seq);
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    let p = AttentionBwdParams {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        wrt: wrt_id,
                        seq_q_stride: mask_strides.q,
                        seq_k_stride: mask_strides.k,
                        mask_batch_stride: mask_strides.b,
                        mask_head_stride: mask_strides.h,
                        _pad_mask_0: 0,
                        _pad_mask_1: 0,
                        _pad_mask_2: 0,
                        q_batch_stride: q_b,
                        q_head_stride: q_h,
                        q_seq_stride: q_s,
                        _pad_q: 0,
                        k_batch_stride: k_b,
                        k_head_stride: k_h,
                        k_seq_stride: k_s,
                        _pad_k: 0,
                        v_batch_stride: v_b,
                        v_head_stride: v_h,
                        v_seq_stride: v_s,
                        _pad_v: 0,
                        o_batch_stride: o_b,
                        o_head_stride: o_h,
                        o_seq_stride: o_s,
                        _pad_o: 0,
                    };
                    schedule.push(Step::AttentionBackward {
                        params: p,
                        mask_buf,
                    });
                    let ak = attention_bwd_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AttentionBwdParams>());
                    let bg = bind_two(&dev.device, ak, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Rope { head_dim, n_rot: _ } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-wgpu Rope: last_dim ({last}) must be a multiple \
                                of head_dim ({head_dim})"
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-wgpu Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    // PLAN L1: derive batch from total / seq / last_dim
                    // (= product of leading dims). `seq_stride` stays at
                    // full seq for buffer offset math; `seq` becomes the
                    // runtime-scaled loop bound.
                    let batch = total / (seq * last as u32).max(1);
                    let p = RopeParams {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        cos_off: (arena.offset(cos_id) / 4) as u32,
                        sin_off: (arena.offset(sin_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        last_dim: last as u32,
                        batch,
                        seq_stride: seq,
                        _p2: 0,
                    };
                    schedule.push(Step::Rope { params: p });
                    let rk = rope_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<RopeParams>());
                    let bg = bind_two(&dev.device, rk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let rank = target_shape.len();
                    if rank != in_shape.len() {
                        panic!(
                            "rlx-wgpu Expand: rank mismatch \
                                (in_rank={}, target_rank={})",
                            in_shape.len(),
                            rank
                        );
                    }
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Cumulative input strides (row-major). When the
                    // input dim is 1 but target dim > 1, that axis
                    // broadcasts → stride = 0.
                    let mut in_strides_row = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides_row[i] = in_strides_row[i + 1] * in_dims[i + 1];
                    }
                    let strides_for_out: Vec<u32> = (0..rank)
                        .map(|i| {
                            if in_dims[i] == 1 && out_dims[i] != 1 {
                                0
                            } else {
                                in_strides_row[i]
                            }
                        })
                        .collect();

                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu expand meta"),
                        size: (meta_data.len() * 4).max(4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue
                        .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta_buf);

                    // PLAN L1: bucket axis stays at out axis 0 iff the
                    // expand at axis 0 isn't a broadcast (in_dims[0]
                    // matches out_dims[0]). When broadcast at axis 0
                    // (in_dims[0]==1, out_dims[0]>1), the bucket-axis
                    // contract doesn't apply — fall back to full extent.
                    let bucket_outermost = if in_dims[0] == out_dims[0] {
                        1u32
                    } else {
                        0u32
                    };
                    let p = ExpandParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        bucket_outermost,
                        out_dim_0: out_dims[0],
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Expand {
                        params: p,
                        meta_idx,
                    });
                    let ek = expand_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ExpandParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu expand bg"),
                        layout: &ek.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: arena.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Gather { axis } => {
                    let table_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    if *axis == 0 {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let vocab = table_shape[0].unwrap_static() as u32;
                        let dim: u32 = table_shape[1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let n_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let p = GatherParams {
                            n_out: elems,
                            n_idx,
                            dim,
                            vocab,
                            in_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                        };
                        schedule.push(Step::Gather { params: p });
                        let gk = gather_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherParams>());
                        let bg = bind_two(&dev.device, gk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let outer: u32 = table_shape[..*axis]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let trailing: u32 = table_shape[*axis + 1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let axis_dim = table_shape[*axis].unwrap_static() as u32;
                        let num_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let total = outer * num_idx * trailing;
                        let p = GatherAxisParams {
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        };
                        schedule.push(Step::GatherAxis { params: p });
                        let gk = gather_axis_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherAxisParams>());
                        let bg = bind_two(&dev.device, gk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }

                Op::FusedMatMulBiasAct { activation } => {
                    // Inputs: [x, w, bias]. We require 2D × 2D or
                    // [..,M,K] × [K,N] (broadcast bias). Bias is shape [N].
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let bias_id = node.inputs[2];
                    let a_shape = graph.node(a_id).shape.dims();
                    let b_shape = graph.node(b_id).shape.dims();
                    let out_shape = node.shape.dims();
                    let (m, k, n) =
                        if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2 {
                            (
                                a_shape[0].unwrap_static() as u32,
                                a_shape[1].unwrap_static() as u32,
                                b_shape[1].unwrap_static() as u32,
                            )
                        } else if a_shape.len() >= 2
                            && b_shape.len() == 2
                            && out_shape.len() == a_shape.len()
                        {
                            let leading: usize = a_shape[..a_shape.len() - 2]
                                .iter()
                                .map(|d| d.unwrap_static())
                                .product();
                            let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                            let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                            let n_inner = b_shape[1].unwrap_static();
                            ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
                        } else {
                            panic!(
                                "rlx-wgpu FusedMatMulBiasAct: unsupported shapes \
                                a={a_shape:?} b={b_shape:?}"
                            );
                        };
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
                    let b_is_param = traces_to_param(&graph, b_id);
                    let compute_precision =
                        derive_matmul_compute(&dev.device, &graph, a_id, b_id, m, k, n);

                    // Split-QKV pattern: matmul writes Q/K/V directly into
                    // 3 separate output buffers, eliminating the 3 Narrow
                    // dispatches that would otherwise follow. Two flavors:
                    //   F32     → matmul_qkv          (portable f32 tile)
                    //   CoopF32 → matmul_qkv_coop_f32 (simdgroup f32 GEMM)
                    // Coop16 is intentionally not handled here (the kernel
                    // would need an f16-acc variant — Naga 29 can't compile
                    // mixed-precision coop_mat).
                    let mqk_eligible = act_id == 0xFFFFu32
                        && (compute_precision == MatmulCompute::F32
                            || compute_precision == MatmulCompute::CoopF32);
                    if mqk_eligible && let Some(&(q_id, k_id_n, v_id)) = qkv_split.get(&node.id) {
                        let head_width = n / 3;
                        let coop = compute_precision == MatmulCompute::CoopF32;
                        let mqk_kernel = if coop {
                            matmul_qkv_coop_f32_kernel(&dev.device)
                                .expect("coop matmul_qkv kernel: hardware feature was checked but kernel missing")
                        } else {
                            matmul_qkv_kernel(&dev.device)
                        };
                        let p = MatmulQkvParams {
                            m,
                            k,
                            n,
                            a_off: (arena.offset(a_id) / 4) as u32,
                            b_off: (arena.offset(b_id) / 4) as u32,
                            q_off: (arena.offset(q_id) / 4) as u32,
                            k_off: (arena.offset(k_id_n) / 4) as u32,
                            v_off: (arena.offset(v_id) / 4) as u32,
                            head_width,
                            has_bias: 1,
                            bias_off: (arena.offset(bias_id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                        };
                        schedule.push(Step::MatmulQkv { params: p, coop });
                        let u = emit_uniform(std::mem::size_of::<MatmulQkvParams>());
                        let bg = bind_two(&dev.device, mqk_kernel, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        schedule.push(Step::Matmul {
                            m,
                            k,
                            n,
                            batch: 1,
                            a_batch_stride: 0,
                            b_batch_stride: 0,
                            c_batch_stride: 0,
                            a_off_f32: (arena.offset(a_id) / 4) as u32,
                            b_off_f32: (arena.offset(b_id) / 4) as u32,
                            c_off_f32: (arena.offset(node.id) / 4) as u32,
                            has_bias: 1,
                            bias_off_f32: (arena.offset(bias_id) / 4) as u32,
                            act_id,
                            b_is_param,
                            compute_precision,
                        });
                        let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                        let bg = build_matmul_bind_group(
                            &dev.device,
                            mm_k,
                            mm_w,
                            &mm_f16w,
                            &mm_f16c,
                            &mm_coop,
                            &mm_coop_f32,
                            &arena,
                            &u,
                            b_is_param,
                            compute_precision,
                        );
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }

                Op::DotGeneral { .. } => {
                    // Should be unreachable: DotGeneral is decomposed into
                    // MatMul + Transpose + Reshape by the unfusion pass
                    // before memory planning. If we hit this arm, the
                    // unfusion pass has a gap.
                    panic!(
                        "rlx-wgpu DotGeneral: leaked past unfusion pass — \
                            check unfuse.rs::expand_dot_general for missing patterns"
                    );
                }

                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    // Greedy fast-path: temperature == 1.0 with no top_k/top_p
                    // is an argmax — same numeric result, much cheaper kernel.
                    let is_greedy = *top_k == 0
                        && (*top_p - 1.0).abs() < 1e-6
                        && (*temperature - 1.0).abs() < 1e-6;
                    if is_greedy {
                        let p = ArgmaxParams {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                        };
                        schedule.push(Step::Argmax { params: p });
                        let amk = argmax_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<ArgmaxParams>());
                        let bg = bind_two(&dev.device, amk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let p = SampleParams {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            top_k: *top_k as u32,
                            top_p_bits: top_p.to_bits(),
                            temp_bits: temperature.to_bits(),
                            seed_lo: *seed as u32,
                            seed_hi: (*seed >> 32) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::Sample { params: p });
                        let sk = sample_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<SampleParams>());
                        let bg = bind_two(&dev.device, sk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_shape = graph.node(node.inputs[0]).shape.dims();
                    let out_shape = node.shape.dims();
                    let op_id: u32 = match kind {
                        ReduceOp::Sum => 0,
                        ReduceOp::Mean => 1,
                        ReduceOp::Max => 2,
                        ReduceOp::Min => 3,
                        ReduceOp::Prod => 4,
                    };
                    match (kernel_size.len(), in_shape.len(), out_shape.len()) {
                        (1, 3, 3) => {
                            let p = Pool1dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                l: in_shape[2].unwrap_static() as u32,
                                l_out: out_shape[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride.first().copied().unwrap_or(1) as u32,
                                pl: padding.first().copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                                _p3: 0,
                                _p4: 0,
                                _p5: 0,
                            };
                            schedule.push(Step::Pool1d { params: p });
                            let pk = pool1d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool1dParams>());
                            let bg = bind_two(&dev.device, pk, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (2, 4, 4) => {
                            let p = Pool2dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                h: in_shape[2].unwrap_static() as u32,
                                w: in_shape[3].unwrap_static() as u32,
                                h_out: out_shape[2].unwrap_static() as u32,
                                w_out: out_shape[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride.first().copied().unwrap_or(1) as u32,
                                sw: stride.get(1).copied().unwrap_or(1) as u32,
                                ph: padding.first().copied().unwrap_or(0) as u32,
                                pw: padding.get(1).copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::Pool2d { params: p });
                            let pk = pool2d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool2dParams>());
                            let bg = bind_two(&dev.device, pk, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (3, 5, 5) => {
                            let p = Pool3dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                d: in_shape[2].unwrap_static() as u32,
                                h: in_shape[3].unwrap_static() as u32,
                                w: in_shape[4].unwrap_static() as u32,
                                d_out: out_shape[2].unwrap_static() as u32,
                                h_out: out_shape[3].unwrap_static() as u32,
                                w_out: out_shape[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride.first().copied().unwrap_or(1) as u32,
                                sh: stride.get(1).copied().unwrap_or(1) as u32,
                                sw: stride.get(2).copied().unwrap_or(1) as u32,
                                pd: padding.first().copied().unwrap_or(0) as u32,
                                ph: padding.get(1).copied().unwrap_or(0) as u32,
                                pw: padding.get(2).copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                            };
                            schedule.push(Step::Pool3d { params: p });
                            let pk = pool3d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool3dParams>());
                            let bg = bind_two(&dev.device, pk, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (k, n, m) => panic!(
                            "rlx-wgpu Pool: kernel-rank {k} with input rank {n} / \
                             output rank {m} not supported (use 1D/2D/3D NCHW)"
                        ),
                    }
                }

                Op::Conv {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_shape = graph.node(node.inputs[0]).shape.dims();
                    let w_shape = graph.node(node.inputs[1]).shape.dims();
                    let out_shape = node.shape.dims();
                    let s = |i: usize| stride.get(i).copied().unwrap_or(1) as u32;
                    let p = |i: usize| padding.get(i).copied().unwrap_or(0) as u32;
                    let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as u32;
                    match (
                        kernel_size.len(),
                        in_shape.len(),
                        w_shape.len(),
                        out_shape.len(),
                    ) {
                        (1, 3, 3, 3) => {
                            let p1 = Conv1dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                l: in_shape[2].unwrap_static() as u32,
                                l_out: out_shape[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: s(0),
                                pl: p(0),
                                dl: d(0),
                                groups: *groups as u32,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::Conv1d { params: p1 });
                            let ck = conv1d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv1dParams>());
                            let bg = bind_two(&dev.device, ck, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (2, 4, 4, 4) => {
                            let p2 = Conv2dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                h: in_shape[2].unwrap_static() as u32,
                                w: in_shape[3].unwrap_static() as u32,
                                h_out: out_shape[2].unwrap_static() as u32,
                                w_out: out_shape[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: s(0),
                                sw: s(1),
                                ph: p(0),
                                pw: p(1),
                                dh: d(0),
                                dw: d(1),
                                groups: *groups as u32,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                            };
                            schedule.push(Step::Conv2d { params: p2 });
                            let ck = conv2d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv2dParams>());
                            let bg = bind_two(&dev.device, ck, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (3, 5, 5, 5) => {
                            let p3 = Conv3dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                d: in_shape[2].unwrap_static() as u32,
                                h: in_shape[3].unwrap_static() as u32,
                                w: in_shape[4].unwrap_static() as u32,
                                d_out: out_shape[2].unwrap_static() as u32,
                                h_out: out_shape[3].unwrap_static() as u32,
                                w_out: out_shape[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: s(0),
                                sh: s(1),
                                sw: s(2),
                                pd: p(0),
                                ph: p(1),
                                pw: p(2),
                                dd: d(0),
                                dh: d(1),
                                dw: d(2),
                                groups: *groups as u32,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                            };
                            schedule.push(Step::Conv3d { params: p3 });
                            let ck = conv3d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                            let bg = bind_two(&dev.device, ck, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (k, ni, wi, mi) => panic!(
                            "rlx-wgpu Conv: rank kernel={k} in={ni} weight={wi} out={mi} \
                             not supported (use 1D/2D/3D NCHW)"
                        ),
                    }
                }

                Op::Cumsum { axis, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let last = (in_shape.len() - 1) as i32;
                    if *axis != -1 && *axis != last {
                        panic!("rlx-wgpu Cumsum: only last-axis wired (got axis={axis})");
                    }
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = CumsumParams {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Cumsum { params: p });
                    let ck2 = cumsum_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<CumsumParams>());
                    let bg = bind_two(&dev.device, ck2, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Fft { inverse, norm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.clone();
                    let meta = rlx_ir::fft::fft_meta(&in_shape);
                    let dtype = in_shape.dtype();
                    let use_gpu = rlx_ir::fft::gpu_fft_native_eligible(dtype, meta.n_complex)
                        && meta.n_complex >= 2;
                    let scale = norm.output_scale(meta.n_complex, *inverse) as f32;
                    if use_gpu {
                        schedule.push(Step::FftGpu {
                            src_off: (arena.offset(in_id) / 4) as u32,
                            dst_off: (arena.offset(node.id) / 4) as u32,
                            outer: meta.outer as u32,
                            n: meta.n_complex as u32,
                            inverse: if *inverse { 1 } else { 0 },
                            norm_scale: scale,
                        });
                        fft_gpu_steps.push(crate::fft_dispatch::FftGpuResources::new(
                            &dev.device,
                            &arena.buffer,
                        ));
                    } else {
                        schedule.push(Step::FftHost {
                            src_byte_off: arena.offset(in_id) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            outer: meta.outer as u32,
                            n_complex: meta.n_complex as u32,
                            inverse: *inverse,
                            norm_tag: norm.tag(),
                            dtype_tag: fft_dtype_tag(dtype),
                        });
                    }
                }
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!(
                            "rlx-wgpu SelectiveScan: state_size {} exceeds compile-time \
                                cap of 256 (kernel uses fixed-size private array)",
                            state_size
                        );
                    }
                    let x_id = node.inputs[0];
                    let dt_id = node.inputs[1];
                    let a_id = node.inputs[2];
                    let b_id = node.inputs[3];
                    let c_id = node.inputs[4];
                    let in_dims = graph.node(x_id).shape.dims();
                    let seq = in_dims[1].unwrap_static() as u32;
                    let p = SelectiveScanParams {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq,
                        hidden: in_dims[2].unwrap_static() as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        delta_off: (arena.offset(dt_id) / 4) as u32,
                        a_off: (arena.offset(a_id) / 4) as u32,
                        b_off: (arena.offset(b_id) / 4) as u32,
                        c_off: (arena.offset(c_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        // PLAN L1: full-extent stride; safe under
                        // active-extent scaling of params.seq.
                        seq_stride: seq,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                        _p4: 0,
                        _p5: 0,
                    };
                    schedule.push(Step::SelectiveScan { params: p });
                    let ssk = selective_scan_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SelectiveScanParams>());
                    let bg = bind_two(&dev.device, ssk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-wgpu GatedDeltaNet: state_size {state_size} > {}",
                            rlx_cpu::gdn::GDN_MAX_STATE
                        );
                    }
                    let q_id = node.inputs[0];
                    let q_shape = &graph.node(q_id).shape;
                    let state_off = if *carry_state {
                        arena.offset(node.inputs[5])
                    } else {
                        0
                    };
                    schedule.push(Step::GatedDeltaNet {
                        q_byte_off: arena.offset(q_id) as u32,
                        k_byte_off: arena.offset(node.inputs[1]) as u32,
                        v_byte_off: arena.offset(node.inputs[2]) as u32,
                        g_byte_off: arena.offset(node.inputs[3]) as u32,
                        beta_byte_off: arena.offset(node.inputs[4]) as u32,
                        state_byte_off: state_off as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: q_shape.dim(0).unwrap_static() as u32,
                        seq: q_shape.dim(1).unwrap_static() as u32,
                        heads: q_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                        use_carry: *carry_state,
                    });
                    if gguf_host_pad.is_none() {
                        let bk = binary_kernel(&dev.device);
                        let u = emit_uniform(256);
                        gguf_host_pad =
                            Some((u.clone(), bind_two(&dev.device, bk, &arena.buffer, &u)));
                    }
                    let (u, bg) = gguf_host_pad.as_ref().unwrap();
                    uniforms.push(u.clone());
                    bind_groups.push(bg.clone());
                }
                Op::Custom { name, attrs, .. } => match name.as_str() {
                    "llada2.group_limited_gate" => {
                        let sig_id = node.inputs[0];
                        let route_id = node.inputs[1];
                        let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                        let mut attr_buf = [0u8; 20];
                        let n = attrs.len().min(20);
                        attr_buf[..n].copy_from_slice(&attrs[..n]);
                        schedule.push(Step::Llada2GroupLimitedGate {
                            sig_byte_off: arena.offset(sig_id) as u32,
                            route_byte_off: arena.offset(route_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            n_elems,
                            attrs: attr_buf,
                        });
                    }
                    "umap.knn" => {
                        let pw_id = node.inputs[0];
                        let pw_shape = graph.node(pw_id).shape.dims();
                        let n = pw_shape[0].unwrap_static() as u32;
                        let k = if attrs.len() >= 4 {
                            u32::from_le_bytes(attrs[..4].try_into().unwrap())
                        } else {
                            panic!("rlx-wgpu: umap.knn attrs missing k");
                        };
                        let pw_off = arena.offset(pw_id) as u32;
                        let out_off = arena.offset(node.id) as u32;
                        if n as usize >= crate::umap_knn_host::UMAP_KNN_GPU_MIN_N {
                            let p = UmapKnnParams {
                                n,
                                k,
                                pw_off: pw_off / 4,
                                out_off: out_off / 4,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::UmapKnn { params: p });
                            let uk = umap_knn_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<UmapKnnParams>());
                            let bg = bind_two(&dev.device, uk, &arena.buffer, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        } else {
                            schedule.push(Step::UmapKnnHost {
                                pairwise_byte_off: pw_off,
                                out_byte_off: out_off,
                                n,
                                k,
                            });
                        }
                    }
                    other => panic!("rlx-wgpu: unsupported Op::Custom('{other}')"),
                },
                Op::GroupedMatMul => {
                    // Inputs: input [M, K], weight [E, K, N], expert_idx [M]
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = w_dims[2].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    let p = GroupedMatmulParams {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    };
                    schedule.push(Step::GroupedMatmul { params: p });
                    let gk = grouped_matmul_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GroupedMatmulParams>());
                    let bg = bind_two(&dev.device, gk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::DequantGroupedMatMul { scheme } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let slab_bytes = (k as usize * n as usize) / block_elems * block_bytes;
                    let total_bytes = graph.node(w_id).shape.num_elements().unwrap();
                    let ne = (total_bytes / slab_bytes.max(1)) as u32;
                    schedule.push(Step::DequantGroupedMatmulGguf {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                        x_byte_off: arena.offset(in_id) as u32,
                        w_byte_off: arena.offset(w_id) as u32,
                        idx_byte_off: arena.offset(idx_id) as u32,
                        out_byte_off: arena.offset(node.id) as u32,
                    });
                    if gguf_host_pad.is_none() {
                        let bk = binary_kernel(&dev.device);
                        let u = emit_uniform(256);
                        gguf_host_pad =
                            Some((u.clone(), bind_two(&dev.device, bk, &arena.buffer, &u)));
                    }
                    let (u, bg) = gguf_host_pad.as_ref().unwrap();
                    uniforms.push(u.clone());
                    bind_groups.push(bg.clone());
                }
                Op::TopK { k } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer: u32 = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let p = TopKParams {
                        outer,
                        inner,
                        k: *k as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::TopK { params: p });
                    let tk = topk_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<TopKParams>());
                    let bg = bind_two(&dev.device, tk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::ScatterAdd => {
                    // Inputs: updates [num_updates, trailing], indices [num_updates].
                    // Output: [out_dim, trailing]. Implemented as two phases:
                    //   1. Zero `out_dim * trailing` slots.
                    //   2. CAS-loop atomic-accumulate `num_updates * trailing` updates.
                    let upd_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    let upd_dims = graph.node(upd_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let num_updates = upd_dims[0].unwrap_static() as u32;
                    let trailing: u32 = upd_dims
                        .iter()
                        .skip(1)
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let out_dim = out_dims[0].unwrap_static() as u32;
                    let out_total = out_dim * trailing;

                    let common = ScatterAddParams {
                        op: 0,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        upd_off: (arena.offset(upd_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_total,
                        num_updates,
                        trailing,
                        out_dim,
                    };
                    let sk = scatter_add_kernel(&dev.device);

                    // Phase 0: zero.
                    schedule.push(Step::ScatterAdd { params: common });
                    let u0 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                    let bg0 = bind_two(&dev.device, sk, &arena.buffer, &u0);
                    uniforms.push(u0);
                    bind_groups.push(bg0);

                    // Phase 1: accumulate.
                    let mut acc = common;
                    acc.op = 1;
                    schedule.push(Step::ScatterAdd { params: acc });
                    let u1 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                    let bg1 = bind_two(&dev.device, sk, &arena.buffer, &u1);
                    uniforms.push(u1);
                    bind_groups.push(bg1);
                }
                Op::FusedResidualLN { has_bias, eps } => {
                    // Inputs: [x, residual, [bias], gamma, beta].
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3]) // bias unused
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = FusedResidualLnParams {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::FusedResidualLn { params: p });
                    let frk = fused_residual_ln_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FusedResidualLnParams>());
                    let bg = bind_two(&dev.device, frk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::FusedResidualRmsNorm { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = FusedResidualRmsNormParams {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::FusedResidualRmsNorm { params: p });
                    let frk = fused_residual_rms_norm_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FusedResidualRmsNormParams>());
                    let bg = bind_two(&dev.device, frk, &arena.buffer, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::QuantScheme;
                    let x_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let out_dims = node.shape.dims();
                    let x_dims = graph.node(x_id).shape.dims();
                    let m = out_dims[0].unwrap_static() as u32;
                    let n = out_dims[1].unwrap_static() as u32;
                    let k = x_dims[1].unwrap_static() as u32;
                    if scheme.is_gguf() {
                        schedule.push(Step::DequantMatmulGguf {
                            m,
                            k,
                            n,
                            scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                            x_byte_off: arena.offset(x_id) as u32,
                            w_byte_off: arena.offset(w_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                        });
                        if gguf_host_pad.is_none() {
                            let bk = binary_kernel(&dev.device);
                            let u = emit_uniform(256);
                            gguf_host_pad =
                                Some((u.clone(), bind_two(&dev.device, bk, &arena.buffer, &u)));
                        }
                        let (u, bg) = gguf_host_pad.as_ref().unwrap();
                        uniforms.push(u.clone());
                        bind_groups.push(bg.clone());
                    } else {
                        let (block_size, scheme_id) = match scheme {
                            QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                            QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                            QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                            QuantScheme::Fp8E4m3 => (1, 3u32),
                            QuantScheme::Fp8E5m2 => (1, 4u32),
                            QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                            other => panic!("rlx-wgpu DequantMatMul: unsupported scheme {other:?}"),
                        };
                        let scale_id = node.inputs[2];
                        let zp_id = node.inputs[3];
                        let p = DequantMatmulParams {
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            w_off: (arena.offset(w_id) / 4) as u32,
                            scale_off: (arena.offset(scale_id) / 4) as u32,
                            zp_off: (arena.offset(zp_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                        };
                        schedule.push(Step::DequantMatmul { params: p });
                        let dk = dequant_matmul_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<DequantMatmulParams>());
                        let bg = bind_two(&dev.device, dk, &arena.buffer, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }
                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let foff = |i: usize| (arena.offset(node.inputs[i]) / 4) as u32;
                    let wrt = match &node.op {
                        Op::RmsNormBackwardInput { .. } => 0u32,
                        Op::RmsNormBackwardGamma { .. } => 1u32,
                        Op::RmsNormBackwardBeta { .. } => 2u32,
                        _ => unreachable!(),
                    };
                    let p = RmsNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: foff(0),
                        gamma_off: foff(1),
                        beta_off: foff(2),
                        dy_off: foff(3),
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        wrt,
                    };
                    let rk = if wrt == 0 {
                        rms_norm_backward_kernel(&dev.device)
                    } else {
                        rms_norm_backward_param_kernel(&dev.device)
                    };
                    let u = emit_uniform(std::mem::size_of::<RmsNormBwdParams>());
                    let bg = bind_two(&dev.device, rk, &arena.buffer, &u);
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => {
                            schedule.push(Step::RmsNormBackwardInput { params: p });
                        }
                        Op::RmsNormBackwardGamma { .. } => {
                            schedule.push(Step::RmsNormBackwardGamma { params: p });
                        }
                        Op::RmsNormBackwardBeta { .. } => {
                            schedule.push(Step::RmsNormBackwardBeta { params: p });
                        }
                        _ => unreachable!(),
                    }
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::RopeBackward { head_dim, n_rot } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                        (
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                            dy_shape.dim(2).unwrap_static() as u32,
                        )
                    } else {
                        (
                            1,
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                        )
                    };
                    let cos_len = graph.node(node.inputs[1]).shape.num_elements().unwrap() as u32;
                    let p = RopeBwdParams {
                        batch,
                        seq,
                        hidden,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        cos_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        sin_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        cos_len,
                    };
                    let rk = rope_backward_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<RopeBwdParams>());
                    let bg = bind_two(&dev.device, rk, &arena.buffer, &u);
                    schedule.push(Step::RopeBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::CumsumBackward { exclusive, .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                    let p = CumsumBwdParams {
                        outer: rows,
                        inner: cols,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    let ck = cumsum_backward_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<CumsumBwdParams>());
                    let bg = bind_two(&dev.device, ck, &arena.buffer, &u);
                    schedule.push(Step::CumsumBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::GatherBackward { .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    let rank = out_shape.rank();
                    let axis = match &node.op {
                        Op::GatherBackward { axis } => *axis,
                        _ => 0,
                    };
                    let axis_u = if axis < 0 {
                        (rank as i32 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let outer: usize = (0..axis_u)
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let num_idx = idx_shape.dim(axis_u).unwrap_static();
                    let trailing: usize = (axis_u + 1..dy_shape.rank())
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_dim = out_shape.dim(axis_u).unwrap_static();
                    let p = GatherBwdParams {
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        idx_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                    };
                    let zk = gather_backward_zero_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GatherBwdParams>());
                    let bg = bind_two(&dev.device, zk, &arena.buffer, &u);
                    schedule.push(Step::GatherBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                #[cfg(feature = "splat")]
                Op::GaussianSplatRender {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRender {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatRenderBackward {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRenderBackward {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        d_loss_byte_off: arena.offset(node.inputs[7]) as u32,
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_byte_off: arena.offset(node.id) as u32,
                        packed_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                        loss_grad_clip: *loss_grad_clip,
                        sh_band: *sh_band,
                        max_anisotropy: *max_anisotropy,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatPrepare {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatPrepare {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        meta_len: elem_len(node.inputs[6]),
                        prep_byte_off: arena.offset(node.id) as u32,
                        prep_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatRasterize {
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    let prep_id = node.inputs[0];
                    let count = match &graph.node(prep_id).op {
                        rlx_ir::Op::GaussianSplatPrepare { .. } => {
                            elem_len(graph.node(prep_id).inputs[0]) / 3
                        }
                        _ => 1,
                    };
                    schedule.push(Step::GaussianSplatRasterize {
                        prep_byte_off: arena.offset(prep_id) as u32,
                        prep_len: elem_len(prep_id),
                        meta_byte_off: arena.offset(node.inputs[1]) as u32,
                        meta_len: elem_len(node.inputs[1]),
                        dst_byte_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        count,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::If { .. } | Op::While { .. } => {
                    // Should be unreachable: unfuse.rs inlines both branches
                    // (If) or unrolls max_iterations (While) into the parent
                    // graph using primitive ops + Where for the gating. If
                    // we hit this arm, the unfusion pass has a gap.
                    panic!(
                        "rlx-wgpu: Op::If/While leaked past unfusion pass — \
                            check unfuse.rs::expand_if / expand_while"
                    );
                }
                other => panic!(
                    "rlx-wgpu: op {other:?} not yet lowered (v2 covers Matmul, \
                     Binary, Compare, Activation, Where — fall back to CPU/Metal/MLX)"
                ),
            }
        }

        if rlx_ir::env::flag("RLX_WGPU_SCHEDULE") || rlx_ir::env::flag("RLX_DISPATCH_REPORT") {
            let mut counts: std::collections::BTreeMap<&'static str, usize> =
                std::collections::BTreeMap::new();
            let mut fft_gpu = 0usize;
            let mut fft_host = 0usize;
            for s in &schedule {
                *counts.entry(step_name(s)).or_insert(0) += 1;
                match s {
                    Step::FftGpu { .. } => fft_gpu += 1,
                    Step::FftHost { .. } => fft_host += 1,
                    _ => {}
                }
            }
            let arena_mb = arena.size as f64 / (1u64 << 20) as f64;
            eprintln!(
                "[rlx-wgpu] schedule: {} steps, arena={arena_mb:.1} MiB, fft_gpu={fft_gpu}, fft_host={fft_host}",
                schedule.len()
            );
            for (n, c) in &counts {
                eprintln!("    {c:>4} × {n}");
            }
        }

        Self {
            graph,
            arena,
            schedule,
            input_offsets,
            param_offsets,
            uniforms,
            bind_groups,
            meta_buffers,
            unresolved: None,
            last_binding: None,
            pending_params: HashMap::new(),
            pending_param_bytes: HashMap::new(),
            active_extent: None,
            uniforms_active_extent: None,
            fft_gpu_steps,
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if self.unresolved.is_some() {
            self.pending_params.insert(name.to_string(), data.to_vec());
            return;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            self.arena.write_f32(&dev.queue, id, data);
        }
    }

    /// Debug helper: run forward, then read every node slot back and
    /// report the first node whose output contains a NaN, plus a
    /// summary of the *previous* finite node's value range so the
    /// caller can see the input that broke. Slow — diagnosis only.
    pub fn debug_first_nan_node(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Option<(usize, String, String)> {
        let _ = self.run(inputs);
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        let mut prev_summary = String::from("(none)");
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if !self.arena.has(node.id) {
                continue;
            }
            let elems = node.shape.num_elements().unwrap_or(0);
            if elems == 0 {
                continue;
            }
            let data = self.arena.read_f32(&dev.device, &dev.queue, node.id);
            let nan_count = data.iter().filter(|v| v.is_nan()).count();
            let inf_count = data.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                return Some((i, format!("{:?}", node.op), prev_summary));
            }
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let abs_max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            prev_summary = format!(
                "node #{i} {:?} shape={:?}  min={min:.6e} max={max:.6e} |max|={abs_max:.6e}",
                node.op,
                node.shape
                    .dims()
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect::<Vec<_>>()
            );
        }
        None
    }

    /// Declared output dtypes (one per graph output). Used by the
    /// runtime wrapper's `run_typed` to narrow F32 results back to
    /// F16/BF16 etc. on the way out.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

    /// Upload raw bytes for a Param. The bytes land tight-packed at
    /// the param's slot offset — no f32 round-trip. Used for quantized
    /// weights (int8 / int4) where the kernel reads the byte stream
    /// via `bitcast<u32>` from the f32-typed arena.
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if self.unresolved.is_some() {
            self.pending_param_bytes
                .insert(name.to_string(), data.to_vec());
            return;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            dev.queue
                .write_buffer(&self.arena.buffer, self.arena.offset(id) as u64, data);
        }
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        // Lazy compile path: if we deferred compile waiting for shapes,
        // infer the binding from input data lengths now and compile.
        if self.unresolved.is_some() {
            self.lazy_compile_for_inputs(inputs);
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                self.arena.write_f32(&dev.queue, id, data);
            }
        }

        // Active-extent (PLAN L1): scale safe Steps' primary dim by
        // actual/upper. Used in BOTH the uniform-write loop (so the
        // kernel sees the scaled count) AND the dispatch loop (so the
        // workgroup grid is shrunk).
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // Stage uniform writes — but skip the loop entirely when the
        // bytes already in the uniforms match this run's active extent.
        // BERT inference at fixed batch hits this path: 100+ tiny
        // queue.write_buffer calls (one per Step) collapse to zero,
        // saving milliseconds of staging-copy overhead.
        let need_uniform_writes = self.uniforms_active_extent != Some(active);
        if need_uniform_writes {
            let mut gpu_ui = 0usize;
            for step in self.schedule.iter() {
                if step_runs_on_host(step) {
                    continue;
                }
                match step {
                    Step::CastF32ToF16 { .. } => {
                        // Params are static for this step (offset+len), so the
                        // pre-pass write at compile time is sufficient. No
                        // active-extent scaling — len is the full element count.
                    }
                    Step::Matmul {
                        m,
                        k,
                        n,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32,
                        batch,
                        a_batch_stride,
                        b_batch_stride,
                        c_batch_stride,
                        has_bias,
                        bias_off_f32,
                        act_id,
                        b_is_param: _,
                        compute_precision: _,
                    } => {
                        // PLAN L1 (safe at any batch — c_batch_stride is
                        // pre-baked at compile time at FULL m, so scaling
                        // params.m only changes per-thread bound checks).
                        let m_scaled = scale(*m);
                        let p = MatmulParams {
                            m: m_scaled,
                            k: *k,
                            n: *n,
                            a_off: *a_off_f32,
                            b_off: *b_off_f32,
                            c_off: *c_off_f32,
                            batch: *batch,
                            a_batch_stride: *a_batch_stride,
                            b_batch_stride: *b_batch_stride,
                            c_batch_stride: *c_batch_stride,
                            has_bias: *has_bias,
                            bias_off: *bias_off_f32,
                            act_id: *act_id,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Binary { params } | Step::Compare { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Unary { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Where { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Reduce { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Softmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RmsNormBackwardInput { params }
                    | Step::RmsNormBackwardGamma { params }
                    | Step::RmsNormBackwardBeta { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::CumsumBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RopeBackward { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Cumsum { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FftGpu { .. } => {}
                    Step::Copy { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ElementwiseRegion { params } => {
                        // Active-extent: scale element count.
                        let mut p = *params;
                        p.len = scale(p.len);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Transpose { params, .. } => {
                        // PLAN L1: when bucket_outermost == 1, scale
                        // `out_total` proportional to scaling `out_dim_0`.
                        // Other transposes leave out_total at full extent
                        // (predicate prevents the active-extent path).
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Narrow { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Concat { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Gather { params } => {
                        let mut p = *params;
                        p.n_out = scale(p.n_out);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherAxis { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Attention { params, .. } => {
                        // PLAN L1: scale seq_q + seq_k. Stride fields
                        // (seq_q_stride / seq_k_stride) stay at the
                        // compile-time full extent, so per-(batch, head)
                        // offset math in the WGSL stays correct.
                        let mut p = *params;
                        p.seq_q = scale(p.seq_q);
                        p.seq_k = scale(p.seq_k);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AttentionBackward { params, .. } => {
                        let mut p = *params;
                        if p.wrt == 0 {
                            p.seq_q = scale(p.seq_q);
                        } else {
                            p.seq_k = scale(p.seq_k);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Rope { params } => {
                        // PLAN L1: scale `seq` and `n_total` proportionally.
                        // `seq_stride` and `batch` stay at compile-time
                        // values; the WGSL kernel uses them for buffer
                        // offsets while `seq` / `n_total` are loop bounds.
                        let mut p = *params;
                        let s_active = scale(p.seq);
                        p.seq = s_active;
                        p.n_total = p.batch * s_active * p.last_dim;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Expand { params, .. } => {
                        // PLAN L1: same pattern as Transpose.
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Argmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool2d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv2d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ScatterAdd { params } => {
                        // Two-phase: phase 0 zeros the FULL output (preserves
                        // accumulator semantics); phase 1 scatters first
                        // num_updates_active updates only.
                        let mut p = *params;
                        if p.op == 1 {
                            p.num_updates = scale(p.num_updates);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::TopK { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::UmapKnn { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupedMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Sample { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SelectiveScan { params } => {
                        // Predicate-gated to batch=1: scale seq.
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmulGguf { .. }
                    | Step::DequantGroupedMatmulGguf { .. }
                    | Step::GatedDeltaNet { .. }
                    | Step::Llada2GroupLimitedGate { .. }
                    | Step::UmapKnnHost { .. }
                    | Step::FftHost { .. } => {}
                    Step::FusedResidualLn { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualLnTee { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualRmsNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MatmulQkv { params, coop: _ } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    #[cfg(feature = "splat")]
                    Step::GaussianSplatRender { .. }
                    | Step::GaussianSplatRenderBackward { .. }
                    | Step::GaussianSplatPrepare { .. }
                    | Step::GaussianSplatRasterize { .. } => {}
                }
                if !matches!(step, Step::FftGpu { .. }) {
                    gpu_ui += 1;
                }
            }
            self.uniforms_active_extent = Some(active);
        }

        // Encode + submit.
        let mm_k = matmul_kernel(&dev.device);
        let mm_w = matmul_wide_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_kernel(&dev.device);
        let mm_cast = cast_f32_to_f16_kernel(&dev.device);
        let bk = binary_kernel(&dev.device);
        let uk = unary_kernel(&dev.device);
        let ck = compare_kernel(&dev.device);
        let wk = where_kernel(&dev.device);
        let mut step_i = 0;
        let mut gpu_bi = 0usize;
        let mut fft_i = 0usize;
        while step_i < self.schedule.len() {
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rlx-wgpu run"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rlx-wgpu compute pass"),
                    timestamp_writes: None,
                });
                while step_i < self.schedule.len() {
                    if step_runs_on_host(&self.schedule[step_i]) {
                        break;
                    }
                    let step = &self.schedule[step_i];
                    // PLAN L3: per-step Perfetto trace span; no-op when
                    // env var RLX_TRACE_PERFETTO unset.
                    let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "wgpu");
                    match step {
                        Step::CastF32ToF16 { params } => {
                            // Pre-pass for matmul_coop16: mirror f32 arena
                            // region into f16 shadow buffer so the matmul
                            // kernel can read A as f16. One thread per
                            // element; 64-thread workgroups.
                            if let Some(cast_k) = mm_cast {
                                pass.set_pipeline(&cast_k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(params.len, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                        }
                        Step::Matmul {
                            m,
                            n,
                            batch,
                            b_is_param,
                            compute_precision,
                            ..
                        } =>
                        // The dispatch branches below use a chain of
                        // `is_some() && …unwrap()` to pick a pipeline
                        // because each variant cares about a different
                        // Option<Pipeline>. `if let Some(p) = …` chains
                        // would require nesting per variant; the flat
                        // form is the readable shape here.
                        {
                            #[allow(clippy::unnecessary_unwrap)]
                            // Safe at any batch (see safe_for_active_extent
                            // comment); scale m, output rows past m_s per
                            // batch retain prior values via c_batch_stride.
                            let m_s = scale(*m);
                            if m_s == 0 {
                                continue;
                            }
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            // Kernel selection priority:
                            //   1. compute_precision == F16 + b_is_param +
                            //      SHADER_F16 → matmul_f16_compute
                            //      (f16 multiply, f32 acc — 2× ALU on Apple)
                            //   2. legacy RLX_WGPU_F16_WEIGHTS opt-in →
                            //      matmul_f16w (storage-only f16; experimental,
                            //      currently regresses on Apple)
                            //   3. wide-N (m≥32, n≥64)   → matmul_wide
                            //   4. otherwise            → matmul (small/skinny)
                            let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
                            if let Some(coop) = mm_coop.as_ref()
                                && *b_is_param
                                && *compute_precision == MatmulCompute::Coop16
                            {
                                // Hardware GEMM via simdgroup_matrix /
                                // KHR_cooperative_matrix. 32×32 output tile
                                // per workgroup (16 hardware-GEMM ops with
                                // shared A/B loads). Caller guaranteed m, n,
                                // k are multiples of 32/32/8.
                                pass.set_pipeline(&coop.pipeline);
                                pass.dispatch_workgroups(n / 32, m_s.div_ceil(32), *batch);
                            } else if let Some(coop_f32) = mm_coop_f32.as_ref()
                                && *b_is_param
                                && *compute_precision == MatmulCompute::CoopF32
                            {
                                // Pure-f32 cooperative-matrix path
                                // (`simdgroup_float8x8` on Apple). Same tile
                                // shape as Coop16; no precision loss.
                                pass.set_pipeline(&coop_f32.pipeline);
                                pass.dispatch_workgroups(n / 32, m_s.div_ceil(32), *batch);
                            } else if let Some(f16c) = mm_f16c.as_ref()
                                && *b_is_param
                                && *compute_precision == MatmulCompute::F16
                            {
                                pass.set_pipeline(&f16c.pipeline);
                                pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), *batch);
                            } else if let Some(f16w) = mm_f16w.as_ref()
                                && *b_is_param
                                && f16w_opt_in
                            {
                                pass.set_pipeline(&f16w.pipeline);
                                pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), *batch);
                            } else if m_s >= 32 && *n >= 64 {
                                pass.set_pipeline(&mm_w.pipeline);
                                pass.dispatch_workgroups(n.div_ceil(64), m_s.div_ceil(32), *batch);
                            } else {
                                pass.set_pipeline(&mm_k.pipeline);
                                pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), *batch);
                            }
                        }
                        Step::Binary { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            pass.set_pipeline(&bk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Compare { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            pass.set_pipeline(&ck.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Unary { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            pass.set_pipeline(&uk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Where { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            pass.set_pipeline(&wk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Reduce { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let rk = reduce_kernel(&dev.device);
                            pass.set_pipeline(&rk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Softmax { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let sk = softmax_kernel(&dev.device);
                            pass.set_pipeline(&sk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::LayerNorm { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let lk = layernorm_kernel(&dev.device);
                            pass.set_pipeline(&lk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::RmsNormBackwardInput { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let rk = rms_norm_backward_kernel(&dev.device);
                            pass.set_pipeline(&rk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(outer_s, 1, 1);
                        }
                        Step::RmsNormBackwardGamma { params }
                        | Step::RmsNormBackwardBeta { params } => {
                            if params.inner == 0 {
                                continue;
                            }
                            let rk = rms_norm_backward_param_kernel(&dev.device);
                            pass.set_pipeline(&rk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        Step::CumsumBackward { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let ck = cumsum_backward_kernel(&dev.device);
                            pass.set_pipeline(&ck.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::RopeBackward { params } => {
                            let seq_s = scale(params.seq);
                            if seq_s == 0 {
                                continue;
                            }
                            let rk = rope_backward_kernel(&dev.device);
                            pass.set_pipeline(&rk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = params.batch * seq_s * params.hidden;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::GatherBackward { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let total = outer_s * params.axis_dim * params.trailing;
                            if total > 0 {
                                let zk = gather_backward_zero_kernel(&dev.device);
                                pass.set_pipeline(&zk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, _, _) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, 1, 1);
                            }
                            let ak = gather_backward_acc_kernel(&dev.device);
                            pass.set_pipeline(&ak.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(outer_s, 1, 1);
                        }
                        Step::Cumsum { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let ck2 = cumsum_kernel(&dev.device);
                            pass.set_pipeline(&ck2.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::FftGpu {
                            src_off,
                            dst_off,
                            outer,
                            n,
                            inverse,
                            norm_scale,
                        } => {
                            let res = &self.fft_gpu_steps[fft_i];
                            fft_i += 1;
                            crate::fft_dispatch::dispatch_fft_gpu_in_pass(
                                &dev.device,
                                &dev.queue,
                                &mut pass,
                                res,
                                *src_off,
                                *dst_off,
                                *outer,
                                *n,
                                *inverse != 0,
                                *norm_scale,
                            );
                        }
                        Step::Copy { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let ck2 = copy_kernel(&dev.device);
                            pass.set_pipeline(&ck2.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::ElementwiseRegion { params } => {
                            let len_s = scale(params.len);
                            if len_s == 0 {
                                continue;
                            }
                            let ek = elementwise_region_kernel(&dev.device);
                            pass.set_pipeline(&ek.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(len_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Transpose { params, .. } => {
                            // Compute scaled grid count to match the
                            // uniform's scaled out_total when bucket axis
                            // is outermost.
                            let total_s = if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                let scaled_d0 = scale(params.out_dim_0);
                                let inner = params.out_total / params.out_dim_0;
                                scaled_d0 * inner
                            } else {
                                params.out_total
                            };
                            if total_s == 0 {
                                continue;
                            }
                            let tk = transpose_kernel(&dev.device);
                            pass.set_pipeline(&tk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Narrow { params } => {
                            let total_s = scale(params.total);
                            if total_s == 0 {
                                continue;
                            }
                            let nk = narrow_kernel(&dev.device);
                            pass.set_pipeline(&nk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Concat { params } => {
                            let total_s = scale(params.total);
                            if total_s == 0 {
                                continue;
                            }
                            let cck = concat_kernel(&dev.device);
                            pass.set_pipeline(&cck.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Gather { params } => {
                            let n_out_s = scale(params.n_out);
                            if n_out_s == 0 {
                                continue;
                            }
                            let gk = gather_kernel(&dev.device);
                            pass.set_pipeline(&gk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_out_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::GatherAxis { params } => {
                            let total_s = scale(params.total);
                            if total_s == 0 {
                                continue;
                            }
                            let gk = gather_axis_kernel(&dev.device);
                            pass.set_pipeline(&gk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Attention { params, .. } => {
                            // Scale seq_q for grid dim; per-head strides
                            // come from seq_q_stride / seq_k_stride (full
                            // extent) inside the WGSL.
                            let seq_q_s = scale(params.seq_q);
                            if seq_q_s == 0 {
                                continue;
                            }
                            let ak = attention_kernel(&dev.device);
                            pass.set_pipeline(&ak.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = params.batch * params.heads * seq_q_s;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::AttentionBackward { params, .. } => {
                            let axis = if params.wrt == 0 {
                                params.seq_q
                            } else {
                                params.seq_k
                            };
                            let axis_s = scale(axis);
                            if axis_s == 0 {
                                continue;
                            }
                            let ak = attention_bwd_kernel(&dev.device);
                            pass.set_pipeline(&ak.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = params.batch * params.heads * axis_s;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Rope { params } => {
                            // Multi-batch via stride-field WGSL fix:
                            // iterate `batch * scaled_seq * last_dim` items.
                            let s_active = scale(params.seq);
                            let total_s = params.batch * s_active * params.last_dim;
                            if total_s == 0 {
                                continue;
                            }
                            let rk = rope_kernel(&dev.device);
                            pass.set_pipeline(&rk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Expand { params, .. } => {
                            let total_s = if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                let scaled_d0 = scale(params.out_dim_0);
                                let inner = params.out_total / params.out_dim_0;
                                scaled_d0 * inner
                            } else {
                                params.out_total
                            };
                            if total_s == 0 {
                                continue;
                            }
                            let ek = expand_kernel(&dev.device);
                            pass.set_pipeline(&ek.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(total_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Argmax { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let amk = argmax_kernel(&dev.device);
                            pass.set_pipeline(&amk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Pool2d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let pk = pool2d_kernel(&dev.device);
                            pass.set_pipeline(&pk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = n_s * params.c * params.h_out * params.w_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Conv2d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let ck2 = conv2d_kernel(&dev.device);
                            pass.set_pipeline(&ck2.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = n_s * params.c_out * params.h_out * params.w_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Pool1d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let pk = pool1d_kernel(&dev.device);
                            pass.set_pipeline(&pk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = n_s * params.c * params.l_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Pool3d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let pk = pool3d_kernel(&dev.device);
                            pass.set_pipeline(&pk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = n_s * params.c * params.d_out * params.h_out * params.w_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Conv1d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let ck = conv1d_kernel(&dev.device);
                            pass.set_pipeline(&ck.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = n_s * params.c_out * params.l_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::Conv3d { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let ck = conv3d_kernel(&dev.device);
                            pass.set_pipeline(&ck.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total =
                                n_s * params.c_out * params.d_out * params.h_out * params.w_out;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::ScatterAdd { params } => {
                            let sk = scatter_add_kernel(&dev.device);
                            pass.set_pipeline(&sk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            // Phase 0 zeros the FULL output (preserves
                            // accumulator semantics). Phase 1 scatters first
                            // num_updates_active updates only; serial single
                            // workgroup either way (atomic CAS unsupported in
                            // naga's MSL emitter — see scatter_add.wgsl).
                            if params.op == 0 {
                                let (gx, gy, gz) = dispatch_dims(params.out_total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            } else {
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                        }
                        Step::TopK { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let tk = topk_kernel(&dev.device);
                            pass.set_pipeline(&tk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::UmapKnn { params } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            let uk = umap_knn_kernel(&dev.device);
                            pass.set_pipeline(&uk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(n_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::GroupedMatmul { params } => {
                            let m_s = scale(params.m);
                            if m_s == 0 {
                                continue;
                            }
                            let gk = grouped_matmul_kernel(&dev.device);
                            pass.set_pipeline(&gk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                        }
                        Step::Sample { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let sk = sample_kernel(&dev.device);
                            pass.set_pipeline(&sk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::SelectiveScan { params } => {
                            // Predicate-gated to batch=1; the seq scaling
                            // happens inside the kernel (uniform sees scaled
                            // seq). Dispatch grid here is per-(batch, hidden);
                            // unaffected by seq scaling.
                            let ssk = selective_scan_kernel(&dev.device);
                            pass.set_pipeline(&ssk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let total = params.batch * params.hidden;
                            let (gx, gy, gz) = dispatch_dims(total, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::DequantMatmul { params } => {
                            let m_s = scale(params.m);
                            if m_s == 0 {
                                continue;
                            }
                            let dk = dequant_matmul_kernel(&dev.device);
                            pass.set_pipeline(&dk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                        }
                        Step::FusedResidualLn { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let frk = fused_residual_ln_kernel(&dev.device);
                            pass.set_pipeline(&frk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::FusedResidualLnTee { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let frtk = fused_residual_ln_tee_kernel(&dev.device);
                            pass.set_pipeline(&frtk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::FusedResidualRmsNorm { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let frk = fused_residual_rms_norm_kernel(&dev.device);
                            pass.set_pipeline(&frk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                            pass.dispatch_workgroups(gx, gy, gz);
                        }
                        Step::MatmulQkv { params, coop } => {
                            let m_s = scale(params.m);
                            if m_s == 0 {
                                continue;
                            }
                            // Both kernels write to the same 32×32 output tile
                            // grid; only the inner GEMM strategy differs.
                            let pipe = if *coop {
                                &matmul_qkv_coop_f32_kernel(&dev.device)
                                    .expect("coop matmul_qkv kernel missing")
                                    .pipeline
                            } else {
                                &matmul_qkv_kernel(&dev.device).pipeline
                            };
                            pass.set_pipeline(pipe);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(params.n.div_ceil(32), m_s.div_ceil(32), 1);
                        }
                        Step::DequantMatmulGguf { .. }
                        | Step::DequantGroupedMatmulGguf { .. }
                        | Step::GatedDeltaNet { .. }
                        | Step::Llada2GroupLimitedGate { .. }
                        | Step::UmapKnnHost { .. }
                        | Step::FftHost { .. } => {}
                        #[cfg(feature = "splat")]
                        Step::GaussianSplatRender { .. }
                        | Step::GaussianSplatRenderBackward { .. }
                        | Step::GaussianSplatPrepare { .. }
                        | Step::GaussianSplatRasterize { .. } => {}
                    }
                    if !matches!(step, Step::FftGpu { .. }) {
                        gpu_bi += 1;
                    }
                    step_i += 1;
                }
            }
            dev.queue.submit(std::iter::once(enc.finish()));
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            if step_i >= self.schedule.len() {
                break;
            }
            match &self.schedule[step_i] {
                Step::DequantMatmulGguf {
                    m,
                    k,
                    n,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    crate::gguf_host::run_dequant_matmul_gguf(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *m as usize,
                        *k as usize,
                        *n as usize,
                        *scheme_id,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::DequantGroupedMatmulGguf {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    crate::gguf_host::run_dequant_grouped_matmul_gguf(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *m as usize,
                        *k as usize,
                        *n as usize,
                        *num_experts as usize,
                        *scheme_id,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *idx_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::GatedDeltaNet {
                    q_byte_off,
                    k_byte_off,
                    v_byte_off,
                    g_byte_off,
                    beta_byte_off,
                    state_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    heads,
                    state_size,
                    use_carry,
                } => {
                    crate::gdn_host::run_gated_delta_net(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *q_byte_off as usize,
                        *k_byte_off as usize,
                        *v_byte_off as usize,
                        *g_byte_off as usize,
                        *beta_byte_off as usize,
                        *state_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *heads as usize,
                        *state_size as usize,
                        *use_carry,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_byte_off,
                    route_byte_off,
                    out_byte_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *sig_byte_off as usize,
                        *route_byte_off as usize,
                        *out_byte_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::UmapKnnHost {
                    pairwise_byte_off,
                    out_byte_off,
                    n,
                    k,
                } => {
                    crate::umap_knn_host::run_umap_knn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *pairwise_byte_off as usize,
                        *out_byte_off as usize,
                        *n as usize,
                        *k as usize,
                    );
                }
                Step::FftHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype_tag,
                } => {
                    crate::fft_host::run_fft1d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        *norm_tag,
                        fft_dtype_from_tag(*dtype_tag),
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRender {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    dst_byte_off,
                    dst_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_render(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatPrepare {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    meta_len,
                    prep_byte_off,
                    prep_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_prepare(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRasterize {
                    prep_byte_off,
                    prep_len,
                    meta_byte_off,
                    meta_len,
                    dst_byte_off,
                    dst_len,
                    count,
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_rasterize(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *count as usize,
                        *width,
                        *height,
                        *tile_size,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRenderBackward {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    d_loss_byte_off,
                    d_loss_len,
                    packed_byte_off,
                    packed_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    crate::splat::run_gaussian_splat_render_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *d_loss_byte_off as usize,
                        *d_loss_len as usize,
                        *packed_byte_off as usize,
                        *packed_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                        *loss_grad_clip,
                        *sh_band,
                        *max_anisotropy,
                    );
                }
                _ => break,
            }
            step_i += 1;
        }

        // RLX_WGPU_NAN_TRACE=1: after submission, scan every node's
        // arena slot for NaN. Print the first N nodes whose output
        // contains NaN (in IR topo order). Used to bisect which kernel
        // first introduces NaN — once we know the producer, we know
        // which WGSL to look at.
        if rlx_ir::env::flag("RLX_WGPU_NAN_TRACE") {
            let mut bad_nodes = Vec::new();
            for node in self.graph.nodes() {
                if !self.arena.has(node.id) {
                    continue;
                }
                // Skip leaves — populated by host writes, not kernels.
                if matches!(
                    node.op,
                    rlx_ir::Op::Input { .. }
                        | rlx_ir::Op::Param { .. }
                        | rlx_ir::Op::Constant { .. }
                ) {
                    continue;
                }
                let data = self.arena.read_f32(&dev.device, &dev.queue, node.id);
                let nan_count = data.iter().filter(|v| v.is_nan()).count();
                let inf_count = data.iter().filter(|v| v.is_infinite()).count();
                if nan_count > 0 || inf_count > 0 {
                    // Capture first NaN index + the values around it.
                    let first_nan = data.iter().position(|v| v.is_nan());
                    if let Some(idx) = first_nan {
                        let lo = idx.saturating_sub(2);
                        let hi = (idx + 3).min(data.len());
                        eprintln!(
                            "  node {:?} op={:?} len={} nan={} inf={} \
                                   first_nan_idx={} ctx={:?}",
                            node.id,
                            node.op,
                            data.len(),
                            nan_count,
                            inf_count,
                            idx,
                            &data[lo..hi]
                        );
                    }
                    bad_nodes.push((node.id, data.len(), nan_count, inf_count));
                    if bad_nodes.len() >= 3 {
                        break;
                    }
                }
            }
            if bad_nodes.is_empty() {
                eprintln!("[wgpu-nan-trace] no NaN/Inf in any node — clean run");
            } else {
                eprintln!(
                    "[wgpu-nan-trace] first {} bad nodes (above)",
                    bad_nodes.len()
                );
            }
        }

        self.graph
            .outputs
            .iter()
            .map(|&id| {
                if rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") {
                    let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    vec![0.0; n]
                } else {
                    self.arena.read_f32(&dev.device, &dev.queue, id)
                }
            })
            .collect()
    }
}

/// Compute a (X, Y, 1) workgroup grid for a 1-D workload.
///
/// WebGPU caps `dispatch_workgroups` per-dimension at 65535. For
/// workloads beyond `65535 × workgroup_size_x` threads we split into
/// a 2-D grid; kernels recover the linear thread index via
/// `gid.x + gid.y * num_workgroups.x * 64u`.
fn dispatch_dims(threads_total: u32, workgroup_size: u32) -> (u32, u32, u32) {
    let groups = threads_total.div_ceil(workgroup_size);
    if groups <= 65535 {
        (groups, 1, 1)
    } else {
        let gx = 65535u32;
        let gy = groups.div_ceil(gx);
        (gx, gy, 1)
    }
}

fn require_equal_shapes(graph: &Graph, ids: &[NodeId], op_name: &str) {
    let s0 = graph.node(ids[0]).shape.num_elements().unwrap_or(0);
    for &id in &ids[1..] {
        let si = graph.node(id).shape.num_elements().unwrap_or(0);
        if si != s0 {
            panic!(
                "rlx-wgpu {op_name}: broadcasting not yet implemented; \
                    inputs must have the same element count (got {s0} vs {si})"
            );
        }
    }
}

fn bind_two(
    device: &wgpu::Device,
    kernel: &Kernel,
    buf0: &wgpu::Buffer,
    buf1: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf1.as_entire_binding(),
            },
        ],
    })
}

/// Compute precision selector: derive from IR dtypes of A and B and
/// the device features.
///
/// Priority:
///   1. Coop16 — if EXPERIMENTAL_COOPERATIVE_MATRIX + SHADER_F16 +
///      F16 IR tag + b traces to a Param + M/K/N are 32/8/32 aligned.
///      Unlocks Apple's `simdgroup_matrix` / Vulkan's KHR_cooperative
///      hardware GEMM units (~18× faster than f32 ALU on Apple M-series).
///   2. F32 — every other case, *including* when AutoMixedPrecision
///      tagged the matmul as F16 but it failed Coop16's alignment
///      check. The non-coop F16 path (`matmul_f16_compute.wgsl`) was
///      empirically measured 4-5× SLOWER than the f32 baseline on
///      Apple via wgpu/naga 29 — the WGSL→MSL emit doesn't unlock
///      Apple's f16 ALU through portable WGSL ALU. So at small /
///      unaligned shapes we lose nothing by ignoring the IR's f16
///      tag and using f32 — precision improves AND speed wins.
///
/// (The F16 variant of `MatmulCompute` and `matmul_f16_compute.wgsl`
/// remain for future use — e.g. when naga gains a portable subgroup-
/// matrix surface that lowers efficiently without needing the full
/// coop-matrix dance, or when bf16 hardware lands. Today no path
/// dispatches them.)
fn derive_matmul_compute(
    dev: &wgpu::Device,
    graph: &Graph,
    a_id: NodeId,
    b_id: NodeId,
    m: u32,
    k: u32,
    n: u32,
) -> MatmulCompute {
    use rlx_ir::DType;
    let a_dt = graph.node(a_id).shape.dtype();
    let b_dt = graph.node(b_id).shape.dtype();
    let any_low =
        matches!(a_dt, DType::F16 | DType::BF16) || matches!(b_dt, DType::F16 | DType::BF16);
    // CoopF32 (`simdgroup_float8x8`) needs K and N aligned to 8 and 32
    // (one micro-tile per K-iter, one 32-col workgroup per N-tile).
    // M can be arbitrary — the kernel pads to the next multiple of 32
    // and bounds-checks the output writes so out-of-range rows stay
    // untouched. (The Coop16 / matmul_qkv paths still require m%32==0;
    // their kernels don't have the same bounds check.)
    let coop16_aligned = m.is_multiple_of(32) && k.is_multiple_of(8) && n.is_multiple_of(32);
    let coop_f32_aligned = k.is_multiple_of(8) && n.is_multiple_of(32);
    let has_coop = dev
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // Coop16 has an f16 accumulator (Naga 29 can't compile the mixed
    // f32-acc / f16-operand form). Sums of 3072 BERT-FFN activations
    // overflow f16, so we only enter on F16/BF16 IR tags — AutoMixed
    // users have already opted into the precision tradeoff.
    if any_low
        && has_coop
        && dev.features().contains(wgpu::Features::SHADER_F16)
        && traces_to_param(graph, b_id)
        && coop16_aligned
    {
        return MatmulCompute::Coop16;
    }
    // CoopF32 (`simdgroup_float8x8` on Apple): the f32 hardware-GEMM
    // path. Used whenever cooperative-matrix is available, B is a
    // Param, and shapes align — gives ~5-10× speedup over the
    // tiled `matmul_wide` path with no precision loss vs the f32
    // baseline (BERT max|Δ| stays at 2.3e-3 vs CPU on Apple).
    //
    // Backend gate: only Metal validated. On Vulkan/NVIDIA the same
    // kernel produces wildly wrong output (BERT max|Δ| 3.4 vs CPU,
    // bench 2026-05 on RTX 4090) — naga 29's lowering of
    // `coop_mat<f32>` to KHR_cooperative_matrix doesn't agree with
    // the simdgroup_float8x8 path on layout or stride. Re-enable on
    // Vulkan/DX12 once the path is verified end-to-end. Override
    // with RLX_WGPU_FORCE_COOP_F32=1 to bench the broken path.
    let disabled = rlx_ir::env::flag("RLX_WGPU_NO_COOP_F32");
    let forced = rlx_ir::env::flag("RLX_WGPU_FORCE_COOP_F32");
    let backend_ok = forced
        || matches!(
            crate::device::wgpu_device().map(|d| d.backend),
            Some(wgpu::Backend::Metal)
        );
    if !disabled && backend_ok && has_coop && coop_f32_aligned && traces_to_param(graph, b_id) {
        return MatmulCompute::CoopF32;
    }
    MatmulCompute::F32
}

/// Detects the BERT-style fused-QKV-then-narrow-then-attention
/// pattern. When all three of an attention's Q/K/V inputs are
/// `Op::Narrow` of a single source tensor on the last axis with
/// sequential offsets `(0, H·D, 2·H·D)` and equal lengths `H·D`,
/// returns `Some((qkv_source_node, h_d))` — naming the source
/// tensor and per-slice width.
///
/// EMPIRICAL FINDING: the obvious "skip the narrow + read attention
/// directly from QKV with stride 3·H·D" optimization REGRESSED end-
/// to-end perf 7-15× on Apple M4 Pro. The narrow's apparent overhead
/// (~3 dispatches per attention block, ~150µs at small batch) is
/// dwarfed by the cost of strided attention reads — stepping by
/// 3·H·D = 4.6 KB between sequence positions defeats the hardware
/// prefetcher (prefetch distance maxes around 1-2 KB on M-series).
/// Cosine stayed 0.9999+ (output is correct, just slow).
///
/// Kept as a helper for future smarter fusions — e.g. a coop kernel
/// that reads Q/K/V cooperatively from QKV in a single pass over
/// the sequence dim, avoiding the random-access stride pattern.
#[allow(dead_code)]
fn detect_qkv_narrow_pattern(
    graph: &Graph,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
) -> Option<(NodeId, u32)> {
    let unwrap_narrow = |id: NodeId| -> Option<(NodeId, usize, usize, usize)> {
        let node = graph.node(id);
        match &node.op {
            Op::Narrow { axis, start, len } => Some((node.inputs[0], *axis, *start, *len)),
            _ => None,
        }
    };
    let (q_src, q_axis, q_start, q_len) = unwrap_narrow(q_id)?;
    let (k_src, k_axis, k_start, k_len) = unwrap_narrow(k_id)?;
    let (v_src, v_axis, v_start, v_len) = unwrap_narrow(v_id)?;
    // Same source tensor.
    if q_src != k_src || k_src != v_src {
        return None;
    }
    // Equal slice widths (= H · D).
    if q_len != k_len || k_len != v_len {
        return None;
    }
    // Sequential offsets 0, H·D, 2·H·D.
    if q_start != 0 || k_start != q_len || v_start != q_len * 2 {
        return None;
    }
    // All on the LAST axis of the source.
    let src_rank = graph.node(q_src).shape.dims().len();
    if q_axis + 1 != src_rank || k_axis + 1 != src_rank || v_axis + 1 != src_rank {
        return None;
    }
    Some((q_src, q_len as u32))
}

/// Detects the (FusedMatMulBiasAct → Narrow×3) split-QKV pattern that
/// shows up at the start of every BERT-style attention block. Returns
/// a map `parent_fmb_id → (q_narrow_id, k_narrow_id, v_narrow_id)`
/// for every site where the pattern can be replaced by one
/// `Step::MatmulQkv` dispatch.
///
/// Pattern requirements:
///   - Parent is `Op::FusedMatMulBiasAct { activation: None }` with
///     output shape `[..., 3·head_width]`.
///   - The parent's *only* consumers are exactly 3 `Op::Narrow` nodes,
///     all on the last axis, with offsets `(0, head_width, 2·head_width)`
///     and equal `len = head_width`.
///
/// The win is purely structural: same FMA work, but the 3 narrow
/// dispatches (and their full-tensor read+write of the QKV intermediate)
/// disappear. Different from the reverted "skip narrow + read attention
/// strided" approach because reads from each Q/K/V buffer remain
/// sequential — the prefetcher stays happy.
/// Detects (`Op::Binary(Add) → Op::LayerNorm`) where the Add has more
/// than one consumer in the graph — the case `FuseResidualLN` declines
/// because its single-consumer guard would force materializing the sum.
///
/// Returns:
///   - `ln_to_tee`: `ln_id → (h, delta, gamma, beta, sum_id)` so the
///     wgpu LayerNorm lowering can emit `Step::FusedResidualLnTee`
///     using the existing arena slot for the sum (= the Add's slot).
///   - `skip_adds`: the set of Add `NodeId`s whose normal Step emission
///     should be suppressed; their output value is written by the tee
///     step instead.
fn detect_residual_ln_tee_pattern(
    graph: &Graph,
) -> (
    HashMap<NodeId, (NodeId, NodeId, NodeId, NodeId, NodeId)>,
    HashSet<NodeId>,
) {
    use rlx_ir::op::BinaryOp;
    // Consumer counts (output references count once each).
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            *consumers.entry(input).or_insert(0) += 1;
        }
    }
    for &out in &graph.outputs {
        *consumers.entry(out).or_insert(0) += 1;
    }

    let mut ln_to_tee = HashMap::new();
    let mut skip_adds = HashSet::new();
    for node in graph.nodes() {
        let Op::LayerNorm { axis: _, eps: _ } = &node.op else {
            continue;
        };
        if node.inputs.len() < 3 {
            continue;
        } // need [in, gamma, beta]
        let in_id = node.inputs[0];
        let in_node = graph.node(in_id);
        if !matches!(in_node.op, Op::Binary(BinaryOp::Add)) {
            continue;
        }
        // Only fire when Add has >= 2 consumers (otherwise `FuseResidualLN`
        // already collapses it into Op::FusedResidualLN upstream).
        if consumers.get(&in_id).copied().unwrap_or(0) < 2 {
            continue;
        }
        // Add must be plain — both operands shape-equal to LN's input
        // and to each other.
        if in_node.inputs.len() != 2 {
            continue;
        }
        let h_id = in_node.inputs[0];
        let delta_id = in_node.inputs[1];
        if graph.node(h_id).shape.dims() != node.shape.dims() {
            continue;
        }
        if graph.node(delta_id).shape.dims() != node.shape.dims() {
            continue;
        }
        let gamma_id = node.inputs[1];
        let beta_id = node.inputs[2];
        ln_to_tee.insert(node.id, (h_id, delta_id, gamma_id, beta_id, in_id));
        skip_adds.insert(in_id);
    }
    (ln_to_tee, skip_adds)
}

fn detect_split_qkv_pattern(graph: &Graph) -> HashMap<NodeId, (NodeId, NodeId, NodeId)> {
    // consumers[parent] = list of node ids that read parent
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(node.id);
        }
    }
    // Output nodes also count as consumers — would prevent QKV elision
    // if the matmul output is ever read externally.
    for &out_id in &graph.outputs {
        consumers.entry(out_id).or_default().push(NodeId(u32::MAX));
    }

    let mut result = HashMap::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::FusedMatMulBiasAct { activation: None }) {
            continue;
        }
        let cs = match consumers.get(&node.id) {
            Some(c) if c.len() == 3 => c,
            _ => continue,
        };
        let dims = node.shape.dims();
        if dims.is_empty() {
            continue;
        }
        let last_axis = dims.len() - 1;
        let n = dims[last_axis].unwrap_static();
        if n % 3 != 0 {
            continue;
        }
        let head_width = n / 3;

        // Each consumer must be a Narrow on the last axis, len = head_width.
        let mut narrows: Vec<(usize, NodeId)> = Vec::with_capacity(3);
        let mut all_match = true;
        for &c in cs {
            let cn = graph.node(c);
            match cn.op {
                Op::Narrow { axis, start, len }
                    if axis == last_axis && len == head_width && cn.inputs[0] == node.id =>
                {
                    narrows.push((start, c));
                }
                _ => {
                    all_match = false;
                    break;
                }
            }
        }
        if !all_match {
            continue;
        }
        narrows.sort_by_key(|&(start, _)| start);
        if narrows[0].0 != 0 || narrows[1].0 != head_width || narrows[2].0 != 2 * head_width {
            continue;
        }
        result.insert(node.id, (narrows[0].1, narrows[1].1, narrows[2].1));
    }
    result
}

/// Walk through Cast/Reshape nodes (which alias the underlying arena
/// slot, per `plan_f32_uniform`) to find whether `id` ultimately
/// refers to an `Op::Param`. AutoMixedPrecision wraps params in
/// Cast(F32→F16) nodes, so a literal `matches!(node.op, Op::Param)`
/// check on the matmul's `b_id` would miss the Cast(Param) case.
fn traces_to_param(graph: &Graph, mut id: NodeId) -> bool {
    loop {
        let node = graph.node(id);
        match &node.op {
            Op::Param { .. } => return true,
            Op::Cast { .. } | Op::Reshape { .. } => {
                if node.inputs.is_empty() {
                    return false;
                }
                id = node.inputs[0];
            }
            _ => return false,
        }
    }
}

/// Per-Matmul-step bind group builder. Three branches:
///   1. compute_precision == F16 + b_is_param + SHADER_F16
///        → matmul_f16_compute (3-binding, f16 ALU)
///   2. legacy `RLX_WGPU_F16_WEIGHTS` env var + b_is_param + SHADER_F16
///        → matmul_f16w (3-binding, f32 ALU; experimental, see kernel
///         docstring for why this currently regresses perf)
///   3. otherwise → matmul (2-binding, f32 ALU)
/// Append a Coop16 pre-pass: mirrors `arena[off..off+len]` (f32) into
/// `arena_f16[off..off+len]` (f16) so the matmul kernel can read A
/// as f16. Caller is responsible for guaranteeing the arena has an
/// `f16_buffer` (should be true on any SHADER_F16-capable device).
///
/// Currently unused — superseded by the workgroup-staging path in
/// `matmul_coop16.wgsl`. Retained as the right primitive for future
/// kernels that operate on a f16-tagged activation region without
/// internal staging (e.g. a chain of f16-only ops).
#[allow(dead_code)]
fn push_cast_f32_to_f16_step(
    device: &wgpu::Device,
    arena: &Arena,
    schedule: &mut Vec<Step>,
    uniforms: &mut Vec<wgpu::Buffer>,
    bind_groups: &mut Vec<wgpu::BindGroup>,
    mm_cast: &Option<&'static Kernel>,
    src_off: u32,
    len: u32,
) {
    let kernel = match mm_cast {
        Some(k) => *k,
        None => return, // device lacks SHADER_F16; fall through, dispatch will skip
    };
    let f16_buf = match &arena.f16_buffer {
        Some(b) => b,
        None => return,
    };
    let p = CastF32ToF16Params {
        src_off,
        len,
        _p0: 0,
        _p1: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu cast_f32_to_f16 uniform"),
        size: std::mem::size_of::<CastF32ToF16Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Write params at compile (kernel doesn't depend on active extent).
    let dev = wgpu_device().expect("rlx-wgpu: device gone");
    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu cast_f32_to_f16 bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: f16_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: arena.buffer.as_entire_binding(),
            },
        ],
    });
    schedule.push(Step::CastF32ToF16 { params: p });
    uniforms.push(u);
    bind_groups.push(bg);
}

fn build_matmul_bind_group(
    device: &wgpu::Device,
    mm_k: &Kernel,
    _mm_w: &Kernel,
    mm_f16w: &Option<&'static Kernel>,
    mm_f16c: &Option<&'static Kernel>,
    mm_coop: &Option<&'static Kernel>,
    mm_coop_f32: &Option<&'static Kernel>,
    arena: &Arena,
    params: &wgpu::Buffer,
    b_is_param: bool,
    compute_precision: MatmulCompute,
) -> wgpu::BindGroup {
    if b_is_param
        && compute_precision == MatmulCompute::CoopF32
        && let Some(coop_f32) = mm_coop_f32
    {
        // 2-binding layout — both A and B come from the f32 arena
        // (no f16 shadow buffer needed for the pure-f32 path).
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_coop_f32 bg"),
            layout: &coop_f32.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    }
    if b_is_param
        && compute_precision == MatmulCompute::Coop16
        && let (Some(f16_buf), Some(coop)) = (&arena.f16_buffer, mm_coop)
    {
        // 3-binding layout — A is staged from arena (f32) through
        // workgroup-shared memory inside the kernel, no separate
        // f16 binding for A.
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_coop16 bg"),
            layout: &coop.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: f16_buf.as_entire_binding(),
                }, // weights
            ],
        });
    }
    if b_is_param
        && compute_precision == MatmulCompute::F16
        && let (Some(f16_buf), Some(f16c)) = (&arena.f16_buffer, mm_f16c)
    {
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_f16_compute bg"),
            layout: &f16c.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: f16_buf.as_entire_binding(),
                },
            ],
        });
    }
    let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
    if b_is_param
        && f16w_opt_in
        && let (Some(f16_buf), Some(f16w)) = (&arena.f16_buffer, mm_f16w)
    {
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_f16w bg"),
            layout: &f16w.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: f16_buf.as_entire_binding(),
                },
            ],
        });
    }
    bind_two(device, mm_k, &arena.buffer, params)
}
