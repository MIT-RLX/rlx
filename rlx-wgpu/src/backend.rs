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
use std::num::NonZeroU64;

use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ArgmaxParams, AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams,
    Conv1dParams, Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams,
    DequantMatmulParams, ElementwiseRegionParams, ExpandParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, Kernel, LayerNormBwdParams, LayerNormParams, MatmulParams,
    MatmulQkvParams, NarrowConcatParams, Pool1dParams, Pool2dParams, Pool3dParams, ReduceParams,
    RmsNormBwdParams, RopeBwdParams, RopeParams, SampleParams, ScatterAddParams,
    SelectiveScanParams, SoftmaxParams, TopKParams, TransposeParams, UmapKnnParams, UnaryParams,
    WhereParams, argmax_kernel, attention_bwd_kernel, attention_kernel,
    batch_elementwise_region_kernel, binary_kernel, cast_f32_to_f16_kernel, compare_kernel,
    concat_kernel, conv1d_kernel, conv2d_kernel, conv3d_kernel, copy_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, elementwise_region_kernel,
    elementwise_region_spatial_kernel, expand_kernel, fused_residual_ln_kernel,
    fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel, gather_axis_kernel,
    gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel, grouped_matmul_kernel,
    layer_norm_backward_gamma_partial_kernel, layer_norm_backward_gamma_reduce_kernel,
    layer_norm_backward_input_kernel, layernorm_kernel, matmul_coop_f16_vulkan_active_kernel,
    matmul_coop_f16_vulkan_kernel, matmul_coop_f32_active_kernel, matmul_coop16_kernel,
    matmul_f16_compute_kernel, matmul_f16w_kernel, matmul_kernel,
    matmul_qkv_coop_f16_vk_active_kernel, matmul_qkv_coop_f16_vk_kernel,
    matmul_qkv_coop_f32_kernel, matmul_qkv_kernel, matmul_wide_active_kernel, matmul_wide_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    rms_norm_backward_kernel, rms_norm_backward_param_kernel, rope_backward_kernel, rope_kernel,
    sample_kernel, scatter_add_kernel, selective_scan_kernel, softmax_kernel, topk_kernel,
    transpose_kernel, umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel, where_kernel,
};
/// Compute the maximum tail-scratch bytes any single op needs across
/// the graph. Currently only `Op::LayerNormBackwardGamma` uses scratch
/// — it stores `num_workgroups * H` f32 partial sums.
fn compute_scratch_bytes(graph: &rlx_ir::Graph) -> usize {
    const ROWS_PER_WG: u32 = 16;
    let mut max_bytes = 0usize;
    for node in graph.nodes() {
        // Norm staging: when params live far from activations in the arena,
        // wgpu's `max_storage_buffer_binding_size` can prevent binding a
        // single window that covers both. We reserve a small scratch tail
        // zone so we can copy gamma/beta next to activations via
        // `copy_buffer_to_buffer` and keep shader bindings local.
        if matches!(
            &node.op,
            rlx_ir::Op::LayerNorm { .. } | rlx_ir::Op::RmsNorm { .. }
        ) {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let h_dim = x_shape.dim(x_shape.rank() - 1);
            if h_dim.is_static() {
                let h = h_dim.unwrap_static();
                // gamma + beta, 256B-aligned for binding offsets.
                let bytes = ((h * 4).div_ceil(256) * 256) * 2;
                if bytes > max_bytes {
                    max_bytes = bytes;
                }
            }
        }
        if let rlx_ir::Op::LayerNormBackwardGamma { .. } = &node.op {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let Some(elems) = x_shape.num_elements() else {
                continue;
            };
            let h_dim = x_shape.dim(x_shape.rank() - 1);
            if !h_dim.is_static() {
                continue;
            }
            let h = h_dim.unwrap_static();
            if h == 0 {
                continue;
            }
            let rows = (elems / h) as u32;
            let num_workgroups = rows.div_ceil(ROWS_PER_WG.max(1));
            let bytes = (num_workgroups as usize) * h * 4;
            if bytes > max_bytes {
                max_bytes = bytes;
            }
        }
    }
    // Reserve extra scratch for staging small far-apart operands when the
    // arena exceeds wgpu's binding window. This keeps compile-time simple
    // and avoids per-op scratch sizing plumbing.
    max_bytes.max(64 * 1024 * 1024)
}

/// FNV-1a over f32 payload bytes — skips redundant `queue.write_buffer`
/// when bench/inference feeds identical input tensors across runs.
fn hash_f32_input(data: &[f32]) -> u64 {
    let bytes = bytemuck::cast_slice(data);
    let mut h: u64 = 0xcbf29ce484222325;
    h ^= data.len() as u64;
    h = h.wrapping_mul(0x100000001b3);
    for chunk in bytes.chunks(8) {
        let mut arr = [0u8; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        h ^= u64::from_le_bytes(arr);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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
    /// Vulkan/NVIDIA 16×16 f16 tensor-core matmul with K-slab f32
    /// reduction (avoids Naga mixed f16/f32 coop_mat bugs).
    CoopF16Vk,
}

/// Split-write QKV matmul kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulQkvKind {
    F32,
    CoopF32,
    CoopF16Vk,
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
        f16_mirror: bool,
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
    /// Welch PSD top-K — D2H → rlx-cpu → H2D.
    WelchPeaksHost {
        spec_byte_off: u32,
        dst_byte_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
    },
    LogMelHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    LogMelBackwardHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dy_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    /// NCHW im2col host path (D2H → rlx-cpu → H2D).
    Im2ColHost {
        x_byte_off: u32,
        col_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw_dil: u32,
    },
    /// Host-side buffer copy (recorded into a command encoder) used to
    /// stage small param tensors into the tail scratch region so kernels
    /// can bind a ≤4GiB window of the arena.
    BufferCopy {
        src_byte_off: u32,
        dst_byte_off: u32,
        bytes: u32,
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
    BatchElementwiseRegion {
        params: BatchElementwiseRegionParams,
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
    LayerNormBackwardInput {
        params: LayerNormBwdParams,
    },
    LayerNormBackwardGammaPartial {
        params: LayerNormBwdParams,
        num_workgroups: u32,
    },
    LayerNormBackwardGammaReduce {
        params: LayerNormBwdParams,
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
        kind: MatmulQkvKind,
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
    /// Last-upload fingerprint per input name; skips staging when unchanged.
    input_staging_hashes: HashMap<String, u64>,
    /// True when the schedule contains CoopF16Vk matmul (disables f32-only
    /// input upload skip — the f16 shadow must stay in sync each run).
    coop_f16_vk: bool,
    /// CoopF16Vk Param B offsets (f32 arena / 4) → param name for wide routing.
    coop_f16_b_param: HashMap<u32, String>,
    /// Param names flagged by the oscillation probe for wide f32 fallback.
    coop_f16_vk_wide_b: HashSet<String>,
    /// Wide f32 bind groups for CoopF16Vk steps (schedule index → bg).
    coop_f16_vk_wide_bind_groups: HashMap<usize, wgpu::BindGroup>,
    /// CoopF16Vk activation operands mirrored on the host each `run()` (f32+f16).
    coop_f16_host_activations: Vec<(NodeId, Activation, String)>,
    /// Last `set_param` f32 payload per name (for host activation mirrors).
    stashed_params: HashMap<String, Vec<f32>>,
    /// Reused output readback staging (avoids per-run buffer alloc).
    readback_staging: Option<ReadbackStaging>,
    /// Persistent tiny readback buffer for single scalar outputs.
    tiny_readback: Option<TinyReadbackStaging>,
    /// Per-`FftGpu` step: isolated uniform buffers + bind groups (one vec entry per op).
    fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources>,
    /// Persistent KV inputs (host staging uploaded each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Arena input slots authoritative — skip host KV mirror each decode step.
    gpu_handle_resident: HashSet<String>,
    pending_read_indices: Option<Vec<usize>>,
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
            | Step::BatchElementwiseRegion { .. }
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
            | Step::ScatterAdd { .. }
            | Step::BufferCopy { .. } => true,
            // FFT: full-extent transform per row, no active-extent
            // scaling. Marking true so a graph that mixes FFT with
            // active-extent-safe ops still gets the optimization for
            // the rest of the schedule.
            Step::FftGpu { .. } | Step::FftHost { .. } => true,
            Step::Im2ColHost { .. }
            | Step::WelchPeaksHost { .. }
            | Step::LogMelHost { .. }
            | Step::LogMelBackwardHost { .. } => true,
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
            | Step::LayerNormBackwardInput { .. }
            | Step::LayerNormBackwardGammaPartial { .. }
            | Step::LayerNormBackwardGammaReduce { .. }
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
        Step::WelchPeaksHost { .. } => "welch_peaks_host",
        Step::LogMelHost { .. } => "log_mel_host",
        Step::LogMelBackwardHost { .. } => "log_mel_backward_host",
        Step::Im2ColHost { .. } => "im2col_host",
        Step::BufferCopy { .. } => "buffer_copy",
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
        Step::LayerNormBackwardInput { .. } => "layer_norm_backward_input",
        Step::LayerNormBackwardGammaPartial { .. } => "layer_norm_backward_gamma_partial",
        Step::LayerNormBackwardGammaReduce { .. } => "layer_norm_backward_gamma_reduce",
        Step::RopeBackward { .. } => "rope_backward",
        Step::CumsumBackward { .. } => "cumsum_backward",
        Step::GatherBackward { .. } => "gather_backward",
        Step::FusedResidualLn { .. } => "fused_residual_ln",
        Step::FusedResidualLnTee { .. } => "fused_residual_ln_tee",
        Step::FusedResidualRmsNorm { .. } => "fused_residual_rms_norm",
        Step::MatmulQkv { .. } => "matmul_qkv",
        Step::ElementwiseRegion { .. } => "elementwise_region",
        Step::BatchElementwiseRegion { .. } => "batch_elementwise_region",
    }
}

fn step_is_tail_host(step: &Step) -> bool {
    matches!(
        step,
        Step::WelchPeaksHost { .. } | Step::LogMelHost { .. } | Step::LogMelBackwardHost { .. }
    )
}

fn step_runs_on_host(step: &Step) -> bool {
    match step {
        Step::DequantMatmulGguf { .. }
        | Step::DequantGroupedMatmulGguf { .. }
        | Step::GatedDeltaNet { .. }
        | Step::Llada2GroupLimitedGate { .. }
        | Step::UmapKnnHost { .. }
        | Step::FftHost { .. }
        | Step::Im2ColHost { .. }
        | Step::BufferCopy { .. } => true,
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
        self.input_staging_hashes.clear();
        self.coop_f16_vk = fresh.coop_f16_vk;
        self.coop_f16_b_param = fresh.coop_f16_b_param;
        self.coop_f16_vk_wide_bind_groups = fresh.coop_f16_vk_wide_bind_groups;
        self.coop_f16_host_activations = fresh.coop_f16_host_activations;

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

    /// Test hook: first `Step::Attention` Q sequence stride (600 = packed QKV).
    #[doc(hidden)]
    pub fn test_attn_q_seq_stride(&self) -> Option<u32> {
        self.schedule.iter().find_map(|s| {
            if let Step::Attention { params, .. } = s {
                Some(params.q_seq_stride)
            } else {
                None
            }
        })
    }

    /// Test hook: `(q_off, k_off, v_off, q_seq_stride)` for the first attention step.
    #[doc(hidden)]
    pub fn test_attn_offsets_and_stride(&self) -> Option<(u32, u32, u32, u32)> {
        self.schedule.iter().find_map(|s| {
            if let Step::Attention { params, .. } = s {
                Some((
                    params.q_off,
                    params.k_off,
                    params.v_off,
                    params.q_seq_stride,
                ))
            } else {
                None
            }
        })
    }

    /// Global arena offset in f32 elements (not bind-window-local).
    #[doc(hidden)]
    pub fn test_arena_offset_elems(&self, id: NodeId) -> u32 {
        (self.arena.offset(id) / 4) as u32
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
            scratch_off: 0,
            scratch_bytes: 0,
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
            input_staging_hashes: HashMap::new(),
            coop_f16_vk: false,
            coop_f16_b_param: HashMap::new(),
            coop_f16_vk_wide_b: HashSet::new(),
            coop_f16_vk_wide_bind_groups: HashMap::new(),
            coop_f16_host_activations: Vec::new(),
            stashed_params: HashMap::new(),
            readback_staging: None,
            tiny_readback: None,
            fft_gpu_steps: Vec::new(),
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            pending_read_indices: None,
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
        // Pre-walk to compute the max scratch any single op needs.
        // Currently only `Op::LayerNormBackwardGamma` uses scratch
        // (`num_workgroups * H * 4` bytes for the partial-sums buffer).
        let scratch_bytes = compute_scratch_bytes(&graph);
        let mut arena = Arena::from_plan_with_scratch(&dev.device, &plan, scratch_bytes);
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
        let _mm_w_active = matmul_wide_active_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
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
        let mut coop_f16_b_param: HashMap<u32, String> = HashMap::new();
        let mut coop_f16_vk_wide_bind_groups: HashMap<usize, wgpu::BindGroup> = HashMap::new();
        let mm_w_active_compile = matmul_wide_active_kernel(&dev.device);

        let coop_f16_vk_mirror_acts = collect_coop_f16_vk_mirror_activations(&graph, &dev.device);

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
            let cp = derive_matmul_compute(
                &dev.device,
                &graph,
                &coop_f16_vk_mirror_acts,
                a_id,
                b_id,
                m,
                k,
                n,
            );
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

        // EEG-DINO / packed QKV: FMB → [B,S,3,H,D] → Narrow×3 (axis 2) → Attention.
        // Match CPU `compile_thunks` fused_strided_attn: read Q/K/V from the
        // packed parent with seq stride 3·H·D instead of materializing narrows.
        let mut packed_bshd_attn: HashMap<NodeId, (NodeId, u32)> = HashMap::new();
        let mut packed_bshd_skip_narrows: HashSet<NodeId> = HashSet::new();
        if !rlx_ir::env::flag("RLX_WGPU_NO_PACKED_BSHD_ATTN") {
            for node in graph.nodes() {
                let Op::Attention { .. } = &node.op else {
                    continue;
                };
                if node.inputs.len() < 3 {
                    continue;
                }
                if let Some((parent, head_width, narrows)) =
                    rlx_ir::detect_packed_bshd_qkv_attention(
                        &graph,
                        node.inputs[0],
                        node.inputs[1],
                        node.inputs[2],
                    )
                {
                    packed_bshd_attn.insert(node.id, (parent, head_width as u32));
                    for narrow in narrows {
                        if rlx_ir::packed_bshd_narrow_elidable(&graph, narrow, node.id) {
                            packed_bshd_skip_narrows.insert(narrow);
                        }
                    }
                }
            }
        }

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

        let mut coop_f16_host_activations: Vec<(NodeId, Activation, String)> = Vec::new();

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
                    let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                    let b_bytes = arena.len_of(b_id) as u64;
                    let mut compute_precision = derive_matmul_compute(
                        &dev.device,
                        &graph,
                        &coop_f16_vk_mirror_acts,
                        a_id,
                        b_id,
                        m,
                        k,
                        n,
                    );
                    if b_is_param && b_bytes > ARENA_STAGE_CAP && arena.param_fits_f16_mirror(b_id)
                    {
                        compute_precision = MatmulCompute::F16;
                    }
                    let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        node.id,
                        a_id,
                        b_id,
                    );
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    arena_expand_bind_window(
                        &arena,
                        &[node.id, a_id, b_id],
                        &mut base,
                        &mut size,
                        max_binding,
                    );
                    let mut scratch = arena.scratch_off as u64;
                    if param_anchor {
                        arena_ensure_scratch_in_window(&mut scratch, base, size);
                    }
                    if b_is_param && b_bytes > ARENA_STAGE_CAP {
                        assert!(
                            param_anchor && arena_tensor_in_window(&arena, b_id, base, size),
                            "rlx-wgpu matmul: large param B {:?} off={} not in window base={base} size={size}",
                            b_id,
                            arena.offset(b_id),
                        );
                    }
                    let a_off_f32 = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off_f32 = if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena_tensor_in_window(&arena, b_id, base, size)
                    {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            b_id,
                            &mut base,
                            &mut size,
                        )
                    };
                    maybe_push_coop_f16_vk_casts(
                        &graph,
                        a_id,
                        b_id,
                        &coop_f16_vk_mirror_acts,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        &mm_cast,
                        compute_precision,
                        a_off_f32,
                        m,
                        k,
                        batch,
                        b_off_f32,
                        n,
                    );
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32: arena_local_off_f32(&arena, node.id, base),
                        has_bias: 0,
                        bias_off_f32: 0,
                        act_id: 0xFFFF,
                        b_is_param,
                        compute_precision,
                    });
                    let b_off_global = (arena.offset(b_id) / 4) as u32;
                    let b_off_bind = if b_is_param
                        && matches!(
                            compute_precision,
                            MatmulCompute::Coop16 | MatmulCompute::CoopF16Vk | MatmulCompute::F16
                        ) {
                        b_off_global
                    } else {
                        b_off_f32
                    };
                    register_coop_f16_vk_b_param(
                        &mut coop_f16_b_param,
                        &param_offsets,
                        b_id,
                        b_off_bind,
                        compute_precision,
                    );
                    let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                    let (bg, b_off_adj) = build_matmul_bind_group(
                        &dev.device,
                        mm_k,
                        mm_w,
                        &mm_f16w,
                        &mm_f16c,
                        &mm_coop,
                        &mm_coop_f32,
                        &arena,
                        base,
                        size,
                        &u,
                        b_is_param,
                        compute_precision,
                        k,
                        n,
                        batch,
                        b_off_bind,
                        b_bs,
                    );
                    if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                        *b_off_f32 = b_off_adj;
                    }
                    uniforms.push(u);
                    bind_groups.push(bg);
                    if compute_precision == MatmulCompute::CoopF16Vk {
                        coop_f16_vk_wide_bind_groups.insert(
                            schedule.len() - 1,
                            bind_two_buf0_window(
                                &dev.device,
                                mm_w_active_compile,
                                &arena.buffer,
                                base,
                                size,
                                &uniforms[uniforms.len() - 1],
                            ),
                        );
                    }
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
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let win_ids = [node.id, a_id, b_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let a_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        b_id,
                        &mut base,
                        &mut size,
                    );
                    let p = BinaryParams {
                        n: elems,
                        a_off,
                        b_off,
                        c_off: arena_local_off_f32(&arena, node.id, base),
                        op: binary_op_id(*bop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Binary { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, bk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Compare(cop) => {
                    require_equal_shapes(&graph, &node.inputs, "Compare");
                    let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_src = arena.offset(a_id) as u64;
                    let b_src = arena.offset(b_id) as u64;
                    let a_len = arena.len_of(a_id) as u64;
                    let b_len = arena.len_of(b_id) as u64;
                    let a_in = a_src >= base && a_src + a_len <= base + size;
                    let b_in = b_src >= base && b_src + b_len <= base + size;
                    let a_dst = arena.scratch_off as u64;
                    let a_aligned = a_len.div_ceil(256) * 256;
                    let b_dst = a_dst + a_aligned;
                    if a_dst < base || b_dst + b_len > base + size {
                        base = (arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let a_off = if a_in {
                        arena_local_off_f32(&arena, a_id, base)
                    } else {
                        if a_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Compare staging operand A too large ({a_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: a_src as u32,
                            dst_byte_off: a_dst as u32,
                            bytes: a_len as u32,
                        });
                        ((a_dst.saturating_sub(base)) / 4) as u32
                    };
                    let b_off = if b_in {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        if b_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Compare staging operand B too large ({b_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: b_src as u32,
                            dst_byte_off: b_dst as u32,
                            bytes: b_len as u32,
                        });
                        ((b_dst.saturating_sub(base)) / 4) as u32
                    };
                    let p = BinaryParams {
                        n: elems,
                        a_off,
                        b_off,
                        c_off: arena_local_off_f32(&arena, node.id, base),
                        op: compare_op_id(*cop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Compare { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, ck, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Activation(act) => {
                    if coop_f16_vk_mirror_acts.contains(&node.id) {
                        let src_name =
                            tensor_host_name(&input_offsets, &param_offsets, node.inputs[0]);
                        coop_f16_host_activations.push((node.id, *act, src_name));
                        continue;
                    }
                    let in_id = node.inputs[0];
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = UnaryParams {
                        n: elems,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        op: activation_op_id(*act),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Unary {
                        params: p,
                        f16_mirror: false,
                    });
                    let u = emit_uniform(std::mem::size_of::<UnaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, uk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Where => {
                    let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let cond_id = node.inputs[0];
                    let x_id = node.inputs[1];
                    let y_id = node.inputs[2];
                    let cond_src = arena.offset(cond_id) as u64;
                    let x_src = arena.offset(x_id) as u64;
                    let y_src = arena.offset(y_id) as u64;
                    let cond_len = arena.len_of(cond_id) as u64;
                    let x_len = arena.len_of(x_id) as u64;
                    let y_len = arena.len_of(y_id) as u64;
                    let cond_in = cond_src >= base && cond_src + cond_len <= base + size;
                    let x_in = x_src >= base && x_src + x_len <= base + size;
                    let y_in = y_src >= base && y_src + y_len <= base + size;
                    let cond_dst = arena.scratch_off as u64;
                    let cond_aligned = cond_len.div_ceil(256) * 256;
                    let x_dst = cond_dst + cond_aligned;
                    let x_aligned = x_len.div_ceil(256) * 256;
                    let y_dst = x_dst + x_aligned;
                    if cond_dst < base || y_dst + y_len > base + size {
                        base = (arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let cond_off = if cond_in {
                        arena_local_off_f32(&arena, cond_id, base)
                    } else {
                        if cond_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging cond too large ({cond_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: cond_src as u32,
                            dst_byte_off: cond_dst as u32,
                            bytes: cond_len as u32,
                        });
                        ((cond_dst.saturating_sub(base)) / 4) as u32
                    };
                    let x_off = if x_in {
                        arena_local_off_f32(&arena, x_id, base)
                    } else {
                        if x_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging x too large ({x_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: x_src as u32,
                            dst_byte_off: x_dst as u32,
                            bytes: x_len as u32,
                        });
                        ((x_dst.saturating_sub(base)) / 4) as u32
                    };
                    let y_off = if y_in {
                        arena_local_off_f32(&arena, y_id, base)
                    } else {
                        if y_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging y too large ({y_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: y_src as u32,
                            dst_byte_off: y_dst as u32,
                            bytes: y_len as u32,
                        });
                        ((y_dst.saturating_sub(base)) / 4) as u32
                    };
                    let p = WhereParams {
                        n: elems,
                        cond_off,
                        x_off,
                        y_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Where { params: p });
                    let u = emit_uniform(std::mem::size_of::<WhereParams>());
                    let bg = bind_two_buf0_window(&dev.device, wk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::BatchElementwiseRegion {
                    chain,
                    num_batch_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    let n = *num_batch_inputs as usize;
                    if n == 0 || chain.len() > 32 {
                        panic!(
                            "rlx-wgpu BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                            chain.len()
                        );
                    }
                    let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                    let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                        .expect("batch region static shape");
                    let mut win_ids: Vec<NodeId> = vec![node.id];
                    win_ids.extend(node.inputs.iter().copied());
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let chain_enc = rlx_ir::encode_chain_steps(chain);
                    let tail =
                        rlx_ir::encode_prologue_tail(*prologue, &slice_shape, *prologue_input);
                    let base_dst = arena_local_off_f32(&arena, node.id, base);
                    let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                    if use_single {
                        let mut batch_input_offs = [0u32; 64];
                        for i in 0..n {
                            batch_input_offs[i] = arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                node.inputs[i],
                                &mut base,
                                &mut size,
                            );
                        }
                        let p = BatchElementwiseRegionParams {
                            slice_len: slice_elems,
                            num_batch: n as u32,
                            num_steps: chain.len() as u32,
                            base_dst_off: base_dst,
                            slice_elems,
                            batch_input_offs,
                            chain: chain_enc,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                        };
                        schedule.push(Step::BatchElementwiseRegion { params: p });
                        let ek = batch_elementwise_region_kernel(&dev.device);
                        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("rlx-wgpu batch region params"),
                            size: std::mem::size_of::<BatchElementwiseRegionParams>() as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let bg =
                            bind_two_buf0_window(&dev.device, ek, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let spatial = tail[0] == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW;
                        let ek = if spatial {
                            elementwise_region_spatial_kernel(&dev.device)
                        } else {
                            elementwise_region_kernel(&dev.device)
                        };
                        for i in 0..n {
                            let mut input_offs = [0u32; 16];
                            input_offs[0] = arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                node.inputs[i],
                                &mut base,
                                &mut size,
                            );
                            let p = ElementwiseRegionParams {
                                len: slice_elems,
                                num_inputs: 1,
                                num_steps: chain.len() as u32,
                                dst_off: rlx_ir::batch_region_slice_dst_off_f32(
                                    base_dst,
                                    slice_elems,
                                    i,
                                ),
                                input_offs,
                                chain: chain_enc,
                                scalar_input_mask: *scalar_input_mask,
                                prologue: tail[0],
                                out_n: tail[1],
                                out_c: tail[2],
                                out_h: tail[3],
                                out_w: tail[4],
                                prologue_input: tail[5],
                                input_modulus: *input_modulus,
                            };
                            schedule.push(Step::ElementwiseRegion { params: p });
                            let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("rlx-wgpu batch region params"),
                                size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ek,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                    }
                }
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
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
                    let mut win_ids: Vec<NodeId> = vec![node.id];
                    win_ids.extend(node.inputs.iter().copied());
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            id,
                            &mut base,
                            &mut size,
                        );
                    }
                    let chain_enc = rlx_ir::encode_chain_steps(chain);
                    let tail =
                        rlx_ir::encode_prologue_tail(*prologue, &node.shape, *prologue_input);
                    let p = ElementwiseRegionParams {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: arena_local_off_f32(&arena, node.id, base),
                        input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        prologue: tail[0],
                        out_n: tail[1],
                        out_c: tail[2],
                        out_h: tail[3],
                        out_w: tail[4],
                        prologue_input: tail[5],
                        input_modulus: *input_modulus,
                    };
                    schedule.push(Step::ElementwiseRegion { params: p });
                    let ek = if p.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW {
                        elementwise_region_spatial_kernel(&dev.device)
                    } else {
                        elementwise_region_kernel(&dev.device)
                    };
                    // STORAGE (not UNIFORM) — the WGSL params struct
                    // contains `array<u32, N>` arrays whose 4-byte
                    // stride violates uniform's 16-byte stride rule.
                    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu region params"),
                        size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg = bind_two_buf0_window(&dev.device, ek, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Reduce {
                    op: rop,
                    axes,
                    keep_dim: _,
                } => {
                    // Single-axis reduce OR contiguous multi-axis reduce.
                    // The kernel walks the input as `[outer, reduce_dim,
                    // inner]` — for contiguous axes [k..k+m], we set
                    // `reduce_dim = product(dims[k..k+m])`.
                    // Non-contiguous reductions are not yet wired (no
                    // model has hit them); transposing into contiguous
                    // form first is the future fix.
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let mut sorted = axes.clone();
                    sorted.sort_unstable();
                    let contiguous = sorted.windows(2).all(|w| w[1] == w[0] + 1);
                    if !contiguous {
                        panic!(
                            "rlx-wgpu Reduce: non-contiguous axes not yet wired \
                             (got axes={axes:?}, rank={})",
                            in_shape.len()
                        );
                    }
                    let ax_first = sorted[0];
                    let ax_last = *sorted.last().unwrap();
                    let dims_u32: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    let outer: u32 = dims_u32[..ax_first].iter().product();
                    let reduce_dim: u32 = dims_u32[ax_first..=ax_last].iter().product();
                    let inner: u32 = dims_u32[ax_last + 1..].iter().product();
                    let red_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let red_fits = arena_span_bytes(&arena, &red_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &red_ids,
                    );
                    if !red_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = ReduceParams {
                        outer,
                        reduce_dim,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        op: reduce_op_id(*rop),
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::Reduce { params: p });
                    let rk = reduce_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ReduceParams>());
                    let bg = bind_two_buf0_window(&dev.device, rk, &arena.buffer, base, size, &u);
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
                    let sm_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let sm_fits = arena_span_bytes(&arena, &sm_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &sm_ids,
                    );
                    if !sm_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = SoftmaxParams {
                        outer,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Softmax { params: p });
                    let sk = softmax_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SoftmaxParams>());
                    let bg = bind_two_buf0_window(&dev.device, sk, &arena.buffer, base, size, &u);
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
                        let gamma_is_param =
                            tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                        let gamma_bytes = arena.len_of(gamma_id) as u64;
                        let frlt_win: Vec<NodeId> =
                            if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                                vec![gamma_id, node.id, h_id, delta_id, beta_id, sum_id]
                            } else {
                                vec![node.id, h_id, delta_id, gamma_id, beta_id, sum_id]
                            };
                        let mut scratch = arena.scratch_off as u64;
                        let (mut base, mut size, param_anchor) = arena_multi_op_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            &mut schedule,
                            &mut scratch,
                            &frlt_win,
                        );
                        if !param_anchor {
                            base = arena_bind_window_covering_scratch_if_needed(
                                &arena, base, size, scratch,
                            );
                        }
                        let in_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            h_id,
                            &mut base,
                            &mut size,
                        );
                        let residual_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            delta_id,
                            &mut base,
                            &mut size,
                        );
                        let sum_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            sum_id,
                            &mut base,
                            &mut size,
                        );
                        let gamma_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            gamma_id,
                            &mut base,
                            &mut size,
                        );
                        let beta_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            beta_id,
                            &mut base,
                            &mut size,
                        );
                        let p = FusedResidualLnTeeParams {
                            outer,
                            inner,
                            in_off,
                            residual_off,
                            bias_off: 0, // FRLTee currently no-bias only
                            gamma_off,
                            beta_off,
                            sum_off,
                            ln_out_off: arena_local_off_f32(&arena, node.id, base),
                            eps_bits: eps.to_bits(),
                            has_bias: 0,
                            _p0: 0,
                        };
                        schedule.push(Step::FusedResidualLnTee { params: p });
                        let frtk = fused_residual_ln_tee_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<FusedResidualLnTeeParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, frtk, &arena.buffer, base, size, &u);
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
                    let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                    let gamma_bytes = arena.len_of(gamma_id) as u64;
                    let ln_win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                        vec![gamma_id, node.id, in_id]
                    } else {
                        let mut v = vec![node.id, in_id];
                        if gamma_is_param {
                            v.push(gamma_id);
                        }
                        if is_layer_norm {
                            v.push(beta_id);
                        }
                        v
                    };
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let ln_fits = arena_span_bytes(&arena, &ln_win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &ln_win,
                    );
                    if !ln_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let gamma_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        gamma_id,
                        &mut base,
                        &mut size,
                    );
                    let beta_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        beta_id,
                        &mut base,
                        &mut size,
                    );
                    let p = LayerNormParams {
                        outer,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        gamma_off,
                        beta_off,
                        eps_bits: eps.to_bits(),
                        op: if is_layer_norm { 0 } else { 1 },
                    };
                    schedule.push(Step::LayerNorm { params: p });
                    let lk = layernorm_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNormParams>());
                    let bg = bind_two_buf0_window(&dev.device, lk, &arena.buffer, base, size, &u);
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
                    let tr_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let in_is_param = tensor_is_graph_param(&graph, &param_offsets, in_id);
                    let in_bytes = arena.len_of(in_id) as u64;
                    let (mut base, mut size) = if in_is_param && in_bytes <= max_binding {
                        arena_window_for_nodes(&dev.device, &arena, &[in_id])
                    } else if arena_span_bytes(&arena, &tr_ids) <= max_binding {
                        arena_window_for_nodes(&dev.device, &arena, &tr_ids)
                    } else {
                        arena_window_for_nodes(&dev.device, &arena, &[node.id])
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = TransposeParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off,
                        out_off,
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
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &arena.buffer,
                                    offset: base,
                                    size: NonZeroU64::new(size),
                                }),
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
                    if qkv_skip_narrows.contains(&node.id)
                        || packed_bshd_skip_narrows.contains(&node.id)
                    {
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
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = NarrowConcatParams {
                        total: elems,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off,
                        out_off,
                    };
                    schedule.push(Step::Narrow { params: p });
                    let nk = narrow_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                    let bg = bind_two_buf0_window(&dev.device, nk, &arena.buffer, base, size, &u);
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

                    let all_ids: Vec<NodeId> = std::iter::once(node.id)
                        .chain(node.inputs.iter().copied())
                        .collect();
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits_all = arena_span_bytes(&arena, &all_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &all_ids,
                    );
                    arena_expand_bind_window(&arena, &all_ids, &mut base, &mut size, max_binding);
                    if !fits_all && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let out_off = arena_local_off_f32(&arena, node.id, base);

                    let mut start_pos: u32 = 0;
                    for &in_id in &node.inputs {
                        let in_shape = graph.node(in_id).shape.dims();
                        let axis_in = in_shape[*axis].unwrap_static() as u32;
                        let in_total: u32 =
                            in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let _win_ids = [node.id, in_id];
                        let in_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            in_id,
                            &mut base,
                            &mut size,
                        );
                        let p = NarrowConcatParams {
                            total: in_total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size: axis_out,
                            start: start_pos,
                            in_off,
                            out_off,
                        };
                        schedule.push(Step::Concat { params: p });
                        let cck = concat_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, cck, &arena.buffer, base, size, &u);
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
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let bhsd = geom.bhsd;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            geom.batch as u32,
                            geom.heads as u32,
                            geom.seq_q as u32,
                            geom.seq_k as u32,
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

                    let (mask_kind_id, mask_buf, window) = match mask_kind {
                        MaskKind::None => (0u32, None, 0u32),
                        MaskKind::Causal => (1u32, None, 0u32),
                        MaskKind::Custom | MaskKind::Bias => (2u32, None, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, None, *w as u32),
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

                    let stride = |shape: &[rlx_ir::shape::Dim], seq_extent: u32| {
                        rlx_ir::strides_for_shape(shape, heads, hd, seq_extent, bhsd)
                    };
                    let packed_parent = packed_bshd_attn.get(&node.id).copied();
                    let (q_b, q_h, q_s, k_b, k_h, k_s, v_b, v_h, v_s) =
                        if let Some((_parent, head_width)) = packed_parent {
                            let (batch_stride, head_stride, pack_seq) =
                                rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
                            (
                                batch_stride,
                                head_stride,
                                pack_seq,
                                batch_stride,
                                head_stride,
                                pack_seq,
                                batch_stride,
                                head_stride,
                                pack_seq,
                            )
                        } else {
                            let (qb, qh, qs) = stride(q_shape, seq_q);
                            let (kb, kh, ks) = stride(k_shape, seq_k);
                            let v_shape = graph.node(v_id).shape.dims();
                            let (vb, vh, vs) = stride(v_shape, seq_k);
                            (qb, qh, qs, kb, kh, ks, vb, vh, vs)
                        };
                    let out_shape = node.shape.dims();
                    let (o_b, o_h, o_s) = stride(out_shape, seq_q);
                    let mut attn_ids = if let Some((parent, _)) = packed_parent {
                        vec![node.id, parent]
                    } else {
                        vec![node.id, q_id, k_id, v_id]
                    };
                    if mask_kind_id == 2 {
                        attn_ids.push(node.inputs[3]);
                    }
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &attn_ids,
                    );
                    if !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let (q_off, k_off, v_off) = if let Some((parent, head_width)) = packed_parent {
                        let parent_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            parent,
                            &mut base,
                            &mut size,
                        );
                        (
                            parent_off,
                            parent_off.saturating_add(head_width),
                            parent_off.saturating_add(head_width * 2),
                        )
                    } else {
                        let q_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            q_id,
                            &mut base,
                            &mut size,
                        );
                        let k_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            k_id,
                            &mut base,
                            &mut size,
                        );
                        let v_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            v_id,
                            &mut base,
                            &mut size,
                        );
                        (q_off, k_off, v_off)
                    };
                    let out_byte = arena.offset(node.id) as u64;
                    let out_len = arena.len_of(node.id) as u64;
                    let out_aliases_qkv = arena_tensors_overlap(&arena, node.id, q_id)
                        || arena_tensors_overlap(&arena, node.id, k_id)
                        || arena_tensors_overlap(&arena, node.id, v_id)
                        || packed_parent.is_some_and(|(parent, _)| {
                            arena_tensors_overlap(&arena, node.id, parent)
                        });
                    let mut kernel_out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let mut attn_scratch_copy: Option<(u64, u32)> = None;
                    if out_aliases_qkv && rlx_ir::env::flag("RLX_WGPU_DEBUG_ATTN_ALIAS") {
                        eprintln!(
                            "rlx-wgpu Attention alias: out={:?}@{}+{} q={:?}@{} k={:?}@{} v={:?}@{}",
                            node.id,
                            out_byte,
                            out_len,
                            q_id,
                            arena.offset(q_id),
                            k_id,
                            arena.offset(k_id),
                            v_id,
                            arena.offset(v_id),
                        );
                    }
                    if out_aliases_qkv {
                        let tmp_byte = scratch;
                        let tmp_aligned = out_len.div_ceil(256) * 256;
                        scratch = scratch.saturating_add(tmp_aligned);
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        } else {
                            base = arena_bind_window_covering_scratch_if_needed(
                                &arena, base, size, scratch,
                            );
                        }
                        kernel_out_off = ((tmp_byte.saturating_sub(base)) / 4) as u32;
                        attn_scratch_copy = Some((tmp_byte, out_len as u32));
                    }
                    let mask_off = if mask_kind_id == 2 {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            node.inputs[3],
                            &mut base,
                            &mut size,
                        )
                    } else {
                        0
                    };
                    let p = AttentionParams {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off,
                        k_off,
                        v_off,
                        out_off: kernel_out_off,
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
                    if let Some((tmp_byte, bytes)) = attn_scratch_copy {
                        schedule.push(Step::BufferCopy {
                            src_byte_off: tmp_byte as u32,
                            dst_byte_off: out_byte as u32,
                            bytes,
                        });
                    }
                    let ak = attention_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AttentionParams>());
                    let bg = bind_two_buf0_window(&dev.device, ak, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::AttentionBackward {
                    num_heads,
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
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let bhsd = geom.bhsd;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            geom.batch as u32,
                            geom.heads as u32,
                            geom.seq_q as u32,
                            geom.seq_k as u32,
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
                    let stride = |shape: &[rlx_ir::shape::Dim], seq_extent: u32| {
                        rlx_ir::strides_for_shape(shape, heads, hd, seq_extent, bhsd)
                    };
                    let (q_b, q_h, q_s) = stride(q_shape, seq_q);
                    let (k_b, k_h, k_s) = stride(k_shape, seq_k);
                    let v_shape = graph.node(v_id).shape.dims();
                    let (v_b, v_h, v_s) = stride(v_shape, seq_k);
                    let out_shape = node.shape.dims();
                    let out_seq = match wrt {
                        AttentionBwdWrt::Query => seq_q,
                        AttentionBwdWrt::Key | AttentionBwdWrt::Value => seq_k,
                    };
                    let (o_b, o_h, o_s) = stride(out_shape, out_seq);
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
                    let bg = bind_op_output_window(&dev.device, ak, &arena, node.id, &u);
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
                    let cos_is_param = tensor_is_graph_param(&graph, &param_offsets, cos_id);
                    let cos_bytes = arena.len_of(cos_id) as u64;
                    let rope_win: Vec<NodeId> = if cos_is_param && cos_bytes > ARENA_STAGE_CAP {
                        vec![cos_id, sin_id, node.id, x_id]
                    } else {
                        vec![node.id, x_id, cos_id, sin_id]
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &rope_win,
                    );
                    if !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        x_id,
                        &mut base,
                        &mut size,
                    );
                    let cos_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        cos_id,
                        &mut base,
                        &mut size,
                    );
                    let sin_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        sin_id,
                        &mut base,
                        &mut size,
                    );
                    let p = RopeParams {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        in_off,
                        cos_off,
                        sin_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        last_dim: last as u32,
                        batch,
                        seq_stride: seq,
                        _p2: 0,
                    };
                    schedule.push(Step::Rope { params: p });
                    let rk = rope_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<RopeParams>());
                    let bg = bind_two_buf0_window(&dev.device, rk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let in_rank = in_shape.len();
                    let rank = target_shape.len();
                    if in_rank > rank {
                        panic!(
                            "rlx-wgpu Expand: rank mismatch \
                                (in_rank={in_rank}, target_rank={rank})"
                        );
                    }
                    // Implicit leading 1s when input rank < target rank (e.g.
                    // scalar → vector from `LegalizeBroadcast`).
                    let pad = rank.saturating_sub(in_rank);
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> = (0..rank)
                        .map(|i| {
                            if i < pad {
                                1
                            } else {
                                in_shape[i - pad].unwrap_static() as u32
                            }
                        })
                        .collect();
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
                    let exp_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let exp_fits = arena_span_bytes(&arena, &exp_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &exp_ids,
                    );
                    if !exp_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = ExpandParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off,
                        out_off,
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
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &arena.buffer,
                                    offset: base,
                                    size: NonZeroU64::new(size),
                                }),
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
                    let table_is_param = tensor_is_graph_param(&graph, &param_offsets, table_id);
                    let table_bytes = arena.len_of(table_id) as u64;
                    let gather_win: Vec<NodeId> = if table_is_param && table_bytes > ARENA_STAGE_CAP
                    {
                        vec![table_id, node.id, idx_id]
                    } else {
                        vec![node.id, idx_id, table_id]
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, table_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &gather_win,
                    );
                    if !table_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off =
                        if table_anchor && arena_tensor_in_window(&arena, table_id, base, size) {
                            arena_local_off_f32(&arena, table_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                table_id,
                                &mut base,
                                &mut size,
                            )
                        };
                    let idx_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        idx_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_local_off_f32(&arena, node.id, base);
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
                            in_off,
                            idx_off,
                            out_off,
                            _p0: 0,
                        };
                        schedule.push(Step::Gather { params: p });
                        let gk = gather_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, gk, &arena.buffer, base, size, &u);
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
                            table_off: in_off,
                            idx_off,
                            out_off,
                        };
                        schedule.push(Step::GatherAxis { params: p });
                        let gk = gather_axis_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherAxisParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, gk, &arena.buffer, base, size, &u);
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
                    let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                    let b_bytes = arena.len_of(b_id) as u64;
                    let mut compute_precision = derive_matmul_compute(
                        &dev.device,
                        &graph,
                        &coop_f16_vk_mirror_acts,
                        a_id,
                        b_id,
                        m,
                        k,
                        n,
                    );
                    if b_is_param && b_bytes > ARENA_STAGE_CAP && arena.param_fits_f16_mirror(b_id)
                    {
                        compute_precision = MatmulCompute::F16;
                    }

                    // Split-QKV pattern: matmul writes Q/K/V directly into
                    // 3 separate output buffers, eliminating the 3 Narrow
                    // dispatches that would otherwise follow.
                    let mqk_eligible = act_id == 0xFFFFu32
                        && matches!(
                            compute_precision,
                            MatmulCompute::F32 | MatmulCompute::CoopF32 | MatmulCompute::CoopF16Vk
                        );
                    if mqk_eligible && let Some(&(q_id, k_id_n, v_id)) = qkv_split.get(&node.id) {
                        let head_width = n / 3;
                        let qkv_kind = match compute_precision {
                            MatmulCompute::CoopF16Vk => MatmulQkvKind::CoopF16Vk,
                            MatmulCompute::CoopF32 => MatmulQkvKind::CoopF32,
                            _ => MatmulQkvKind::F32,
                        };
                        let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            q_id,
                            a_id,
                            b_id,
                        );
                        let mut scratch = arena.scratch_off as u64;
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        }
                        if b_is_param && b_bytes > ARENA_STAGE_CAP {
                            assert!(
                                param_anchor && arena_tensor_in_window(&arena, b_id, base, size),
                                "rlx-wgpu FusedMatMul QKV: large param B {:?} not in bind window",
                                b_id,
                            );
                        }
                        let a_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            a_id,
                            &mut base,
                            &mut size,
                        );
                        let q_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            q_id,
                            &mut base,
                            &mut size,
                        );
                        let k_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            k_id_n,
                            &mut base,
                            &mut size,
                        );
                        let v_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            v_id,
                            &mut base,
                            &mut size,
                        );
                        let bias_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            bias_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_f32 = if b_is_param
                            && b_bytes > ARENA_STAGE_CAP
                            && arena_tensor_in_window(&arena, b_id, base, size)
                        {
                            arena_local_off_f32(&arena, b_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                b_id,
                                &mut base,
                                &mut size,
                            )
                        };
                        let b_off_global = (arena.offset(b_id) / 4) as u32;
                        maybe_push_coop_f16_vk_casts(
                            &graph,
                            a_id,
                            b_id,
                            &coop_f16_vk_mirror_acts,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mm_cast,
                            compute_precision,
                            a_off,
                            m,
                            k,
                            1,
                            if qkv_kind == MatmulQkvKind::CoopF16Vk {
                                b_off_global
                            } else {
                                b_off_f32
                            },
                            n,
                        );
                        let p = MatmulQkvParams {
                            m,
                            k,
                            n,
                            a_off,
                            b_off: if qkv_kind == MatmulQkvKind::CoopF16Vk {
                                b_off_global
                            } else {
                                b_off_f32
                            },
                            q_off,
                            k_off,
                            v_off,
                            head_width,
                            has_bias: 1,
                            bias_off,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                        };
                        schedule.push(Step::MatmulQkv {
                            params: p,
                            kind: qkv_kind,
                        });
                        register_coop_f16_vk_b_param(
                            &mut coop_f16_b_param,
                            &param_offsets,
                            b_id,
                            p.b_off,
                            match qkv_kind {
                                MatmulQkvKind::CoopF16Vk => MatmulCompute::CoopF16Vk,
                                MatmulQkvKind::CoopF32 => MatmulCompute::CoopF32,
                                MatmulQkvKind::F32 => MatmulCompute::F32,
                            },
                        );
                        let u = emit_uniform(std::mem::size_of::<MatmulQkvParams>());
                        let bg = match qkv_kind {
                            MatmulQkvKind::CoopF16Vk => {
                                let mqk = matmul_qkv_coop_f16_vk_kernel(&dev.device).expect(
                                    "coop f16 matmul_qkv kernel: feature was checked but missing",
                                );
                                let (bg, b_off_adj) = build_matmul_qkv_coop_f16_vk_bind_group(
                                    &dev.device,
                                    mqk,
                                    &arena,
                                    base,
                                    size,
                                    &u,
                                    k,
                                    n,
                                    p.b_off,
                                );
                                if let Some(Step::MatmulQkv { params, .. }) = schedule.last_mut() {
                                    params.b_off = b_off_adj;
                                }
                                bg
                            }
                            MatmulQkvKind::CoopF32 => bind_two_buf0_window(
                                &dev.device,
                                matmul_qkv_coop_f32_kernel(&dev.device).expect(
                                    "coop matmul_qkv kernel: hardware feature was checked but kernel missing",
                                ),
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            ),
                            MatmulQkvKind::F32 => bind_two_buf0_window(
                                &dev.device,
                                matmul_qkv_kernel(&dev.device),
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            ),
                        };
                        uniforms.push(u);
                        bind_groups.push(bg);
                        if qkv_kind == MatmulQkvKind::CoopF16Vk {
                            coop_f16_vk_wide_bind_groups.insert(
                                schedule.len() - 1,
                                bind_two_buf0_window(
                                    &dev.device,
                                    matmul_qkv_kernel(&dev.device),
                                    &arena.buffer,
                                    base,
                                    size,
                                    &uniforms[uniforms.len() - 1],
                                ),
                            );
                        }
                    } else {
                        let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            node.id,
                            a_id,
                            b_id,
                        );
                        let mut scratch = arena.scratch_off as u64;
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        }
                        if b_is_param && b_bytes > ARENA_STAGE_CAP {
                            assert!(
                                param_anchor && arena_tensor_in_window(&arena, b_id, base, size),
                                "rlx-wgpu FusedMatMul: large param B {:?} not in bind window",
                                b_id,
                            );
                        }
                        let a_off_f32 = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            a_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_f32 = if b_is_param
                            && b_bytes > ARENA_STAGE_CAP
                            && arena_tensor_in_window(&arena, b_id, base, size)
                        {
                            arena_local_off_f32(&arena, b_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                b_id,
                                &mut base,
                                &mut size,
                            )
                        };
                        let bias_off_f32 = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            bias_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_global = (arena.offset(b_id) / 4) as u32;
                        let b_off_bind = if b_is_param
                            && matches!(
                                compute_precision,
                                MatmulCompute::Coop16
                                    | MatmulCompute::CoopF16Vk
                                    | MatmulCompute::F16
                            ) {
                            b_off_global
                        } else {
                            b_off_f32
                        };
                        maybe_push_coop_f16_vk_casts(
                            &graph,
                            a_id,
                            b_id,
                            &coop_f16_vk_mirror_acts,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mm_cast,
                            compute_precision,
                            a_off_f32,
                            m,
                            k,
                            1,
                            b_off_bind,
                            n,
                        );
                        schedule.push(Step::Matmul {
                            m,
                            k,
                            n,
                            batch: 1,
                            a_batch_stride: 0,
                            b_batch_stride: 0,
                            c_batch_stride: 0,
                            a_off_f32,
                            b_off_f32,
                            c_off_f32: arena_local_off_f32(&arena, node.id, base),
                            has_bias: 1,
                            bias_off_f32,
                            act_id,
                            b_is_param,
                            compute_precision,
                        });
                        register_coop_f16_vk_b_param(
                            &mut coop_f16_b_param,
                            &param_offsets,
                            b_id,
                            b_off_bind,
                            compute_precision,
                        );
                        let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                        let (bg, b_off_adj) = build_matmul_bind_group(
                            &dev.device,
                            mm_k,
                            mm_w,
                            &mm_f16w,
                            &mm_f16c,
                            &mm_coop,
                            &mm_coop_f32,
                            &arena,
                            base,
                            size,
                            &u,
                            b_is_param,
                            compute_precision,
                            k,
                            n,
                            1,
                            b_off_bind,
                            0,
                        );
                        if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                            *b_off_f32 = b_off_adj;
                        }
                        uniforms.push(u);
                        bind_groups.push(bg);
                        if compute_precision == MatmulCompute::CoopF16Vk {
                            coop_f16_vk_wide_bind_groups.insert(
                                schedule.len() - 1,
                                bind_two_buf0_window(
                                    &dev.device,
                                    mm_w_active_compile,
                                    &arena.buffer,
                                    base,
                                    size,
                                    &uniforms[uniforms.len() - 1],
                                ),
                            );
                        }
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
                        let bg = bind_op_output_window(&dev.device, amk, &arena, node.id, &u);
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
                        let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
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
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
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
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
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
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
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
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let win_ids = [node.id, in_id, w_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    arena_expand_bind_window(&arena, &win_ids, &mut base, &mut size, max_binding);
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let w_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        w_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );

                    let in_shape = graph.node(in_id).shape.dims();
                    let w_shape = graph.node(w_id).shape.dims();
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
                                in_off,
                                w_off,
                                out_off,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::Conv1d { params: p1 });
                            let ck = conv1d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv1dParams>());
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ck,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
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
                                in_off,
                                w_off,
                                out_off,
                            };
                            schedule.push(Step::Conv2d { params: p2 });
                            let ck = conv2d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv2dParams>());
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ck,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
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
                                in_off,
                                w_off,
                                out_off,
                                _p0: 0,
                            };
                            schedule.push(Step::Conv3d { params: p3 });
                            let ck = conv3d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ck,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (k, ni, wi, mi) => panic!(
                            "rlx-wgpu Conv: rank kernel={k} in={ni} weight={wi} out={mi} \
                             not supported (use 1D/2D/3D NCHW)"
                        ),
                    }
                }

                Op::Im2Col {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    if kernel_size.len() != 2 || x_shape.rank() != 4 {
                        panic!("rlx-wgpu Im2Col: 2D NCHW only");
                    }
                    let n = match x_shape.dim(0) {
                        rlx_ir::shape::Dim::Static(v) => v as u32,
                        _ => 0,
                    };
                    let c_in = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    let kh = kernel_size[0] as u32;
                    let kw = kernel_size[1] as u32;
                    let sh = stride.first().copied().unwrap_or(1) as u32;
                    let sw = stride.get(1).copied().unwrap_or(1) as u32;
                    let ph = padding.first().copied().unwrap_or(0) as u32;
                    let pw = padding.get(1).copied().unwrap_or(0) as u32;
                    let dh = dilation.first().copied().unwrap_or(1) as u32;
                    let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
                    let h_out = rlx_ir::shape::conv2d_spatial_output(
                        h as usize,
                        kh as usize,
                        sh as usize,
                        ph as usize,
                        dh as usize,
                    ) as u32;
                    let w_out = rlx_ir::shape::conv2d_spatial_output(
                        w as usize,
                        kw as usize,
                        sw as usize,
                        pw as usize,
                        dw_dil as usize,
                    ) as u32;
                    schedule.push(Step::Im2ColHost {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        col_byte_off: arena.offset(node.id) as u32,
                        n,
                        c_in,
                        h,
                        w,
                        h_out,
                        w_out,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        dh,
                        dw_dil,
                    });
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
                    let bg = bind_op_output_window(&dev.device, ck2, &arena, node.id, &u);
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
                Op::WelchPeaks { k, n_segments } => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                        .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                    schedule.push(Step::WelchPeaksHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        welch_batch: meta.welch_batch as u32,
                        n_fft: meta.n_fft as u32,
                        n_segments: meta.n_segments as u32,
                        k: meta.k as u32,
                    });
                }
                Op::LogMel => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                    schedule.push(Step::LogMelHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::LogMelBackward => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                    schedule.push(Step::LogMelBackwardHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dy_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
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
                    let bg = bind_op_output_window(&dev.device, ssk, &arena, node.id, &u);
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
                        gguf_host_pad = Some((
                            u.clone(),
                            bind_op_output_window(&dev.device, bk, &arena, node.id, &u),
                        ));
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
                            let bg = bind_op_output_window(&dev.device, uk, &arena, node.id, &u);
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
                    let bg = bind_op_output_window(&dev.device, gk, &arena, node.id, &u);
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
                        gguf_host_pad = Some((
                            u.clone(),
                            bind_op_output_window(&dev.device, bk, &arena, node.id, &u),
                        ));
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
                    let bg = bind_op_output_window(&dev.device, tk, &arena, node.id, &u);
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
                    let bg0 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u0);
                    uniforms.push(u0);
                    bind_groups.push(bg0);

                    // Phase 1: accumulate.
                    let mut acc = common;
                    acc.op = 1;
                    schedule.push(Step::ScatterAdd { params: acc });
                    let u1 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                    let bg1 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u1);
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
                    let bg = bind_op_output_window(&dev.device, frk, &arena, node.id, &u);
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
                    let bg = bind_op_output_window(&dev.device, frk, &arena, node.id, &u);
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
                            gguf_host_pad = Some((
                                u.clone(),
                                bind_op_output_window(&dev.device, bk, &arena, node.id, &u),
                            ));
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
                        let bg = bind_op_output_window(&dev.device, dk, &arena, node.id, &u);
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
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
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
                Op::LayerNormBackwardInput { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let p = LayerNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        scratch_off: 0,
                    };
                    let rk = layer_norm_backward_input_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                    schedule.push(Step::LayerNormBackwardInput { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::LayerNormBackwardGamma { eps, .. } => {
                    // Inputs: [x, dy] — gamma_off is unused for this op.
                    // Emit two steps: a multi-workgroup partial that
                    // writes per-chunk dgamma to the tail scratch zone,
                    // and a single-workgroup reduce that sums chunks
                    // into the final dgamma slot.
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    const ROWS_PER_WG: u32 = 16;
                    let num_workgroups = rows.div_ceil(ROWS_PER_WG.max(1));
                    let scratch_off_words = (arena.scratch_off / 4) as u32;
                    let partial_params = LayerNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        gamma_off: 0,
                        dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        out_off: 0, // unused by the partial kernel
                        eps_bits: eps.to_bits(),
                        scratch_off: scratch_off_words,
                    };
                    let reduce_params = LayerNormBwdParams {
                        // `outer` for the reduce kernel carries the
                        // number of partial chunks we just emitted.
                        outer: num_workgroups,
                        inner: h,
                        x_off: 0,
                        gamma_off: 0,
                        dy_off: 0,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        scratch_off: scratch_off_words,
                    };
                    let p_k = layer_norm_backward_gamma_partial_kernel(&dev.device);
                    let r_k = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                    let p_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let r_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let p_bg = bind_op_output_window(&dev.device, p_k, &arena, node.id, &p_u);
                    let r_bg = bind_op_output_window(&dev.device, r_k, &arena, node.id, &r_u);
                    schedule.push(Step::LayerNormBackwardGammaPartial {
                        params: partial_params,
                        num_workgroups,
                    });
                    schedule.push(Step::LayerNormBackwardGammaReduce {
                        params: reduce_params,
                    });
                    uniforms.push(p_u);
                    uniforms.push(r_u);
                    bind_groups.push(p_bg);
                    bind_groups.push(r_bg);
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
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
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
                    let bg = bind_op_output_window(&dev.device, ck, &arena, node.id, &u);
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
                    let bg = bind_op_output_window(&dev.device, zk, &arena, node.id, &u);
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

        let coop_f16_vk = schedule_uses_coop_f16_vk(&schedule);

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
            input_staging_hashes: HashMap::new(),
            coop_f16_vk,
            coop_f16_b_param,
            coop_f16_vk_wide_b: HashSet::new(),
            coop_f16_vk_wide_bind_groups,
            coop_f16_host_activations,
            stashed_params: HashMap::new(),
            readback_staging: None,
            tiny_readback: None,
            fft_gpu_steps,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            pending_read_indices: None,
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        const STASH_MAX_BYTES: usize = 16 * 1024 * 1024;
        if data.len() * 4 <= STASH_MAX_BYTES {
            self.stashed_params.insert(name.to_string(), data.to_vec());
        }
        if self.coop_f16_vk {
            crate::coop_f16_vk::refresh_wide_b_flag(&mut self.coop_f16_vk_wide_b, name, data);
        }
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

    fn dump_node_stats_if_requested(&self, dev: &crate::device::WgpuDevice) {
        if !rlx_ir::env::flag("RLX_WGPU_DUMP_NODES") {
            return;
        }
        let flat_probe = rlx_ir::env::parse_or::<usize>("RLX_WGPU_DUMP_FLAT", usize::MAX);
        let limit = rlx_ir::env::parse_or("RLX_WGPU_DUMP_NODES_LIMIT", 40usize);
        eprintln!(
            "[rlx-wgpu-dump] per-node max |x| (topo order, limit={limit}{})",
            if flat_probe != usize::MAX {
                format!(", flat[{flat_probe}]")
            } else {
                String::new()
            }
        );
        let mut shown = 0usize;
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if !self.arena.has(node.id) {
                continue;
            }
            if matches!(
                node.op,
                rlx_ir::Op::Input { .. }
                    | rlx_ir::Op::Param { .. }
                    | rlx_ir::Op::Constant { .. }
                    | rlx_ir::Op::Reshape { .. }
                    | rlx_ir::Op::Cast { .. }
            ) {
                continue;
            }
            let data = self.arena.read_f32(&dev.device, &dev.queue, node.id);
            let max = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let nz = data.iter().filter(|&&v| v != 0.0).count();
            let flat_s = if flat_probe < data.len() {
                format!(" flat[{flat_probe}]={:.6}", data[flat_probe])
            } else {
                String::new()
            };
            eprintln!(
                "  [{i:>3}] {:?} max={max:.6} nonzero={}/{}{flat_s}",
                node.op,
                nz,
                data.len()
            );
            shown += 1;
            if shown >= limit {
                break;
            }
        }
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.pending_read_indices = read_indices.map(|s| s.to_vec());
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        outs
    }

    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name) {
            if out_idx < self.graph.outputs.len() {
                let id = self.graph.outputs[out_idx];
                if self.arena.has(id) {
                    let dev = wgpu_device().expect("rlx-wgpu: device gone");
                    return Some(self.arena.read_f32(&dev.device, &dev.queue, id));
                }
            }
        }
        if self.gpu_handle_resident.contains(name) {
            if let Some(&id) = self.input_offsets.get(name) {
                if self.arena.has(id) {
                    let dev = wgpu_device().expect("rlx-wgpu: device gone");
                    return Some(self.arena.read_f32(&dev.device, &dev.queue, id));
                }
            }
        }
        self.gpu_handles.get(name).cloned()
    }

    fn readback_plan(&self) -> Vec<usize> {
        let n = self.graph.outputs.len();
        if self.pending_read_indices.is_none() && self.gpu_handle_feeds.is_empty() {
            return (0..n).collect();
        }
        if let Some(ref want) = self.pending_read_indices {
            let mut v: Vec<_> = want.to_vec();
            v.sort_unstable();
            return v;
        }
        (0..n).collect()
    }

    fn propagate_gpu_handle_feeds_on_gpu(
        &mut self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
    ) {
        let extent = self.active_extent;
        let feeds: Vec<(String, usize)> = self
            .gpu_handle_feeds
            .iter()
            .map(|(n, &i)| (n.clone(), i))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_bytes = self.arena.len_of(out_id);
                let copy_bytes = match extent {
                    Some((actual, upper)) if upper > 0 => {
                        let stride = (out_bytes / (upper + 1)).max(4);
                        (actual * stride).min(out_bytes)
                    }
                    _ => out_bytes,
                };
                self.dispatch_arena_copy_bytes(dev, enc, out_id, in_id, copy_bytes);
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    fn dispatch_arena_copy_bytes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
        nbytes: usize,
    ) {
        if nbytes == 0 {
            return;
        }
        let src = self.arena.offset(src_id) as u64;
        let dst = self.arena.offset(dst_id) as u64;
        let nbytes = nbytes
            .min(self.arena.len_of(src_id))
            .min(self.arena.len_of(dst_id)) as u64;
        let elems = (nbytes / 4).max(1) as u32;
        let lo = src.min(dst);
        let hi = src.saturating_add(nbytes).max(dst.saturating_add(nbytes));
        let max_binding = dev.device.limits().max_storage_buffer_binding_size;
        let mut size = hi.saturating_sub(lo).div_ceil(256) * 256;
        size = size.max(256).min(max_binding);
        let mut base = (lo / 256) * 256;
        if base.saturating_add(size) > self.arena.size as u64 {
            base = (self.arena.size as u64).saturating_sub(size);
            base = (base / 256) * 256;
        }
        let p = CopyParams {
            n: elems,
            in_off: (src.saturating_sub(base) / 4) as u32,
            out_off: (dst.saturating_sub(base) / 4) as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            _p4: 0,
        };
        let ck = copy_kernel(&dev.device);
        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu kv_feed_copy uniform"),
            size: std::mem::size_of::<CopyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = bind_two_buf0_window(&dev.device, ck, &self.arena.buffer, base, size, &u);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu kv_feed_copy pass"),
            ..Default::default()
        });
        pass.set_pipeline(&ck.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(elems, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }

    #[allow(dead_code)]
    fn dispatch_arena_copy_between_nodes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
    ) {
        let nbytes = self.arena.len_of(src_id).min(self.arena.len_of(dst_id));
        self.dispatch_arena_copy_bytes(dev, enc, src_id, dst_id, nbytes);
    }

    fn stage_gpu_handle_inputs(
        &mut self,
        dev: &crate::device::WgpuDevice,
        inputs: &[(&str, &[f32])],
    ) {
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                self.arena.write_f32(&dev.queue, id, data);
                self.input_staging_hashes.remove(name);
            }
        }
    }

    fn pack_readback_outputs(&mut self, plan: &[usize], partial: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if self.pending_read_indices.is_none() {
            for (pos, &out_i) in plan.iter().enumerate() {
                if let Some(data) = partial.get(pos) {
                    for (name, &feed_i) in &self.gpu_handle_feeds {
                        if feed_i == out_i {
                            self.gpu_handles.insert(name.clone(), data.clone());
                        }
                    }
                }
            }
        }
        if self.pending_read_indices.is_none() && plan.len() == self.graph.outputs.len() {
            return partial;
        }
        let want = self.pending_read_indices.as_deref().unwrap_or(plan);
        let mut by_idx = std::collections::HashMap::new();
        for (pos, &i) in plan.iter().enumerate() {
            if let Some(d) = partial.get(pos) {
                by_idx.insert(i, d.clone());
            }
        }
        want.iter()
            .map(|&i| {
                by_idx
                    .get(&i)
                    .cloned()
                    .expect("readback plan missing output")
            })
            .collect()
    }

    fn run_tail_host_audio_ops(&self, dev: &crate::device::WgpuDevice) {
        if !self.schedule.iter().any(step_is_tail_host) {
            return;
        }
        for step in &self.schedule {
            if !step_is_tail_host(step) {
                continue;
            }
            match step {
                Step::WelchPeaksHost {
                    spec_byte_off,
                    dst_byte_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => {
                    crate::welch_peaks_host::run_welch_peaks(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *dst_byte_off as usize,
                        *welch_batch as usize,
                        *n_fft as usize,
                        *n_segments as usize,
                        *k as usize,
                    );
                }
                Step::LogMelHost {
                    spec_byte_off,
                    filt_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                Step::LogMelBackwardHost {
                    spec_byte_off,
                    filt_byte_off,
                    dy_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dy_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                _ => {}
            }
        }
    }

    fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        // Lazy compile path: if we deferred compile waiting for shapes,
        // infer the binding from input data lengths now and compile.
        if self.unresolved.is_some() {
            self.lazy_compile_for_inputs(inputs);
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        self.stage_gpu_handle_inputs(dev, inputs);
        let skip_input_upload =
            !rlx_ir::env::flag("RLX_WGPU_FORCE_INPUT_UPLOAD") && !self.coop_f16_vk;
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                if skip_input_upload {
                    let h = hash_f32_input(data);
                    if self.input_staging_hashes.get(name) == Some(&h) {
                        if self.arena.f16_buffer.is_some() {
                            self.arena.write_f16_shadow(&dev.queue, id, data);
                        }
                        continue;
                    }
                    self.arena.write_f32(&dev.queue, id, data);
                    self.input_staging_hashes.insert(name.to_string(), h);
                } else {
                    self.arena.write_f32(&dev.queue, id, data);
                }
            }
        }
        for &(act_id, act, ref src_name) in &self.coop_f16_host_activations {
            let src =
                host_tensor_f32(src_name, inputs, &self.stashed_params).unwrap_or_else(|| {
                    panic!("rlx-wgpu CoopF16Vk host activation: missing tensor {src_name:?}")
                });
            let mirrored = apply_activation_host(act, src);
            self.arena.write_f32(&dev.queue, act_id, &mirrored);
        }
        if !self.coop_f16_host_activations.is_empty() {
            // Ensure host staging writes are visible before CoopF16Vk reads f16.
            let flush = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rlx-wgpu host mirror flush"),
                });
            dev.queue.submit(std::iter::once(flush.finish()));
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
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
                    Step::Unary { params, .. } => {
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
                    Step::LayerNormBackwardInput { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaPartial { params, .. } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaReduce { params } => {
                        // `outer` here is the partial chunk count (not
                        // a batch dim) — do NOT apply active-extent
                        // scaling.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
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
                    Step::BufferCopy { .. } => {}
                    Step::ElementwiseRegion { params } => {
                        // Active-extent: scale element count.
                        let mut p = *params;
                        p.len = scale(p.len);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BatchElementwiseRegion { params } => {
                        let mut p = *params;
                        p.slice_len = scale(p.slice_len);
                        p.num_batch = scale(p.num_batch);
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
                    | Step::FftHost { .. }
                    | Step::Im2ColHost { .. }
                    | Step::WelchPeaksHost { .. }
                    | Step::LogMelHost { .. }
                    | Step::LogMelBackwardHost { .. } => {}
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
                    Step::MatmulQkv { params, kind: _ } => {
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
        let mm_w_active = matmul_wide_active_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f16_vk = matmul_coop_f16_vulkan_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
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
                let mut pass_dispatched = false;
                while step_i < self.schedule.len() {
                    if step_is_tail_host(&self.schedule[step_i]) {
                        step_i += 1;
                        continue;
                    }
                    if step_runs_on_host(&self.schedule[step_i]) {
                        break;
                    }
                    // Vulkan/DX12: end the pass after unary/cast so f32→f16
                    // mirrors are visible to the next step. Only split once
                    // we've dispatched in *this* pass — otherwise the step that
                    // needs the flush would never run (infinite empty passes).
                    if pass_dispatched
                        && step_i > 0
                        && step_needs_pass_flush(&self.schedule[step_i], &self.schedule[step_i - 1])
                    {
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
                            b_off_f32,
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
                            let coop_f16_wide = mm_coop_f16_vk.is_some()
                                && *compute_precision == MatmulCompute::CoopF16Vk
                                && crate::coop_f16_vk::use_wide_matmul(
                                    *b_off_f32,
                                    *n,
                                    &self.coop_f16_b_param,
                                    &self.coop_f16_vk_wide_b,
                                );
                            pass.set_bind_group(
                                0,
                                coop_f16_vk_bind_group(self, gpu_bi, coop_f16_wide),
                                &[],
                            );
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
                                pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), *batch);
                            } else if mm_coop_f16_vk.is_some()
                                && *compute_precision == MatmulCompute::CoopF16Vk
                            {
                                if coop_f16_wide {
                                    dispatch_wide_f32_matmul(
                                        &mut pass,
                                        mm_w_active,
                                        mm_k,
                                        m_s,
                                        *n,
                                        *batch,
                                    );
                                } else {
                                    let n_eff = scale(*n);
                                    let coop_vk =
                                        matmul_coop_f16_vulkan_active_kernel(&dev.device, n_eff)
                                            .expect("coop f16 vk kernel missing");
                                    pass.set_pipeline(&coop_vk.pipeline);
                                    pass.dispatch_workgroups(
                                        m_s.div_ceil(16),
                                        n.div_ceil(16),
                                        *batch,
                                    );
                                }
                            } else if let Some(coop_f32) = mm_coop_f32.as_ref()
                                && *b_is_param
                                && *compute_precision == MatmulCompute::CoopF32
                            {
                                // CoopF32: Metal uses 32×32 simdgroup tiles;
                                // Vulkan uses 8×8 coopLoadT portable kernel.
                                pass.set_pipeline(&coop_f32.pipeline);
                                let backend = wgpu_device()
                                    .map(|d| d.backend)
                                    .unwrap_or(wgpu::Backend::Noop);
                                let (gx, gy) = if backend == wgpu::Backend::Metal {
                                    (n.div_ceil(32), m_s.div_ceil(32))
                                } else {
                                    (m_s.div_ceil(8), n.div_ceil(8))
                                };
                                pass.dispatch_workgroups(gx, gy, *batch);
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
                                pass.set_pipeline(&mm_w_active.pipeline);
                                let backend = wgpu_device()
                                    .map(|d| d.backend)
                                    .unwrap_or(wgpu::Backend::Noop);
                                let (gx, gy) = if matches!(
                                    backend,
                                    wgpu::Backend::Vulkan | wgpu::Backend::Dx12
                                ) {
                                    (n.div_ceil(64), m_s.div_ceil(64))
                                } else {
                                    (n.div_ceil(64), m_s.div_ceil(32))
                                };
                                pass.dispatch_workgroups(gx, gy, *batch);
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
                        Step::Unary { params, f16_mirror } => {
                            let n_s = scale(params.n);
                            if n_s == 0 {
                                continue;
                            }
                            if *f16_mirror {
                                if let Some(uk_f16) = unary_f16_mirror_kernel(&dev.device) {
                                    pass.set_pipeline(&uk_f16.pipeline);
                                } else {
                                    pass.set_pipeline(&uk.pipeline);
                                }
                            } else {
                                pass.set_pipeline(&uk.pipeline);
                            }
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
                            let total_out = outer_s.saturating_mul(params.inner);
                            if params.reduce_dim <= 64 {
                                // Fast path: 1 thread per output cell.
                                let (gx, gy, gz) = dispatch_dims(total_out, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            } else {
                                // Tree-reduce path: 1 workgroup (64
                                // threads) per output cell, parallel
                                // reduction with shared scratch.
                                let (gx, gy, gz) = dispatch_dims(total_out, 1);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
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
                        Step::LayerNormBackwardInput { params } => {
                            let outer_s = scale(params.outer);
                            if outer_s == 0 {
                                continue;
                            }
                            let lk = layer_norm_backward_input_kernel(&dev.device);
                            pass.set_pipeline(&lk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(outer_s, 1, 1);
                        }
                        Step::LayerNormBackwardGammaPartial {
                            params,
                            num_workgroups,
                        } => {
                            if params.inner == 0 || *num_workgroups == 0 {
                                continue;
                            }
                            let lk = layer_norm_backward_gamma_partial_kernel(&dev.device);
                            pass.set_pipeline(&lk.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            pass.dispatch_workgroups(*num_workgroups, 1, 1);
                        }
                        Step::LayerNormBackwardGammaReduce { params } => {
                            if params.inner == 0 {
                                continue;
                            }
                            let lk = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                            pass.set_pipeline(&lk.pipeline);
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
                        Step::BufferCopy { .. } => {
                            // Host step: `copy_buffer_to_buffer` runs outside compute passes.
                        }
                        Step::ElementwiseRegion { params } => {
                            let len_s = scale(params.len);
                            if len_s == 0 {
                                continue;
                            }
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            if params.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW {
                                let ek = elementwise_region_spatial_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                let (gx, gy, gz) = dispatch_prologue_nchw(
                                    params.out_w,
                                    params.out_h,
                                    params.out_n * params.out_c,
                                );
                                pass.dispatch_workgroups(gx, gy, gz);
                            } else {
                                let ek = elementwise_region_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                let (gx, gy, gz) = dispatch_dims(len_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                        }
                        Step::BatchElementwiseRegion { params } => {
                            let slice_len_s = scale(params.slice_len);
                            let num_batch_s = scale(params.num_batch);
                            if slice_len_s == 0 || num_batch_s == 0 {
                                continue;
                            }
                            let ek = batch_elementwise_region_kernel(&dev.device);
                            pass.set_pipeline(&ek.pipeline);
                            pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                            let (gx, gy, _) = dispatch_dims(slice_len_s, 64);
                            pass.dispatch_workgroups(gx, gy, num_batch_s);
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
                        Step::MatmulQkv { params, kind } => {
                            let m_s = scale(params.m);
                            if m_s == 0 {
                                continue;
                            }
                            let qkv_coop_wide = matches!(kind, MatmulQkvKind::CoopF16Vk)
                                && crate::coop_f16_vk::use_wide_matmul(
                                    params.b_off,
                                    params.n,
                                    &self.coop_f16_b_param,
                                    &self.coop_f16_vk_wide_b,
                                );
                            pass.set_bind_group(
                                0,
                                coop_f16_vk_bind_group(self, gpu_bi, qkv_coop_wide),
                                &[],
                            );
                            match kind {
                                MatmulQkvKind::CoopF16Vk => {
                                    if qkv_coop_wide {
                                        pass.set_pipeline(&matmul_qkv_kernel(&dev.device).pipeline);
                                        pass.dispatch_workgroups(
                                            params.n.div_ceil(32),
                                            m_s.div_ceil(32),
                                            1,
                                        );
                                    } else {
                                        let n_eff = scale(params.n);
                                        let mqk = matmul_qkv_coop_f16_vk_active_kernel(
                                            &dev.device,
                                            n_eff,
                                        )
                                        .expect("coop f16 matmul_qkv kernel missing");
                                        pass.set_pipeline(&mqk.pipeline);
                                        pass.dispatch_workgroups(
                                            m_s.div_ceil(16),
                                            params.n.div_ceil(16),
                                            1,
                                        );
                                    }
                                }
                                MatmulQkvKind::CoopF32 => {
                                    pass.set_pipeline(
                                        &matmul_qkv_coop_f32_kernel(&dev.device)
                                            .expect("coop matmul_qkv kernel missing")
                                            .pipeline,
                                    );
                                    pass.dispatch_workgroups(
                                        params.n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        1,
                                    );
                                }
                                MatmulQkvKind::F32 => {
                                    pass.set_pipeline(&matmul_qkv_kernel(&dev.device).pipeline);
                                    pass.dispatch_workgroups(
                                        params.n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        1,
                                    );
                                }
                            }
                        }
                        Step::DequantMatmulGguf { .. }
                        | Step::DequantGroupedMatmulGguf { .. }
                        | Step::GatedDeltaNet { .. }
                        | Step::Llada2GroupLimitedGate { .. }
                        | Step::UmapKnnHost { .. }
                        | Step::FftHost { .. }
                        | Step::Im2ColHost { .. }
                        | Step::WelchPeaksHost { .. }
                        | Step::LogMelHost { .. }
                        | Step::LogMelBackwardHost { .. } => {}
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
                    pass_dispatched = true;
                }
            }
            let needs_f16_drain = step_i < self.schedule.len()
                && !step_runs_on_host(&self.schedule[step_i])
                && step_i > 0
                && step_needs_pass_flush(&self.schedule[step_i], &self.schedule[step_i - 1]);
            let gpu_schedule_done = step_i >= self.schedule.len();
            let skip_readback = rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY");
            let defer_tail = gpu_schedule_done && self.schedule.iter().any(step_is_tail_host);
            let mut fused_readback: Option<(
                ReadbackLayout,
                std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
                Vec<usize>,
            )> = None;
            if gpu_schedule_done && !skip_readback && !defer_tail {
                if !self.gpu_handle_feeds.is_empty() {
                    self.propagate_gpu_handle_feeds_on_gpu(dev, &mut enc);
                }
                let plan = self.readback_plan();
                let out_ids_all: Vec<_> = self.graph.outputs.clone();
                let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                    if self.tiny_readback.is_none() {
                        self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                    }
                    let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                    encode_readback_copies(&mut enc, &self.arena, tiny.buffer(), &out_ids, &layout);
                    let map_rx = schedule_readback_map(&mut enc, tiny.buffer(), &layout);
                    dev.queue.submit(std::iter::once(enc.finish()));
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    wait_readback_map(&dev.device, &map_rx, layout.total_bytes);
                    map_rx.recv().unwrap().unwrap();
                    return self.pack_readback_outputs(
                        &plan,
                        vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)],
                    );
                }
                ReadbackStaging::prepare(
                    &dev.device,
                    &mut self.readback_staging,
                    layout.total_bytes,
                );
                if let Some(staging) = self.readback_staging.as_ref() {
                    encode_readback_copies(
                        &mut enc,
                        &self.arena,
                        staging.buffer(),
                        &out_ids,
                        &layout,
                    );
                    let map_rx = schedule_readback_map(&mut enc, staging.buffer(), &layout);
                    fused_readback = Some((layout, map_rx, plan));
                }
            }
            dev.queue.submit(std::iter::once(enc.finish()));
            if defer_tail {
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                self.run_tail_host_audio_ops(dev);
                if !skip_readback {
                    let mut rb_enc =
                        dev.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rlx-wgpu readback after tail-host"),
                            });
                    if !self.gpu_handle_feeds.is_empty() {
                        self.propagate_gpu_handle_feeds_on_gpu(dev, &mut rb_enc);
                    }
                    let plan = self.readback_plan();
                    let out_ids_all: Vec<_> = self.graph.outputs.clone();
                    let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                    let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                    if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                        if self.tiny_readback.is_none() {
                            self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                        }
                        let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                        encode_readback_copies(
                            &mut rb_enc,
                            &self.arena,
                            tiny.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut rb_enc, tiny.buffer(), &layout);
                        dev.queue.submit(std::iter::once(rb_enc.finish()));
                        wait_readback_map(&dev.device, &map_rx, layout.total_bytes);
                        map_rx.recv().unwrap().unwrap();
                        return self.pack_readback_outputs(
                            &plan,
                            vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)],
                        );
                    }
                    ReadbackStaging::prepare(
                        &dev.device,
                        &mut self.readback_staging,
                        layout.total_bytes,
                    );
                    if let Some(staging) = self.readback_staging.as_ref() {
                        encode_readback_copies(
                            &mut rb_enc,
                            &self.arena,
                            staging.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut rb_enc, staging.buffer(), &layout);
                        dev.queue.submit(std::iter::once(rb_enc.finish()));
                        wait_readback_map(&dev.device, &map_rx, layout.total_bytes);
                        map_rx.recv().unwrap().unwrap();
                        self.dump_node_stats_if_requested(dev);
                        let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                        return self.pack_readback_outputs(&plan, partial);
                    }
                }
            }
            if needs_f16_drain {
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
            let need_host_sync =
                step_i < self.schedule.len() && step_runs_on_host(&self.schedule[step_i]);
            if need_host_sync {
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
            if gpu_schedule_done {
                if skip_readback || defer_tail {
                    return self
                        .graph
                        .outputs
                        .iter()
                        .map(|&id| {
                            let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                            vec![0.0; n]
                        })
                        .collect();
                }
                if let (Some((layout, map_rx, plan)), Some(staging)) =
                    (fused_readback, self.readback_staging.as_ref())
                {
                    wait_readback_map(&dev.device, &map_rx, layout.total_bytes);
                    map_rx.recv().unwrap().unwrap();
                    self.dump_node_stats_if_requested(dev);
                    let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                    return self.pack_readback_outputs(&plan, partial);
                }
                break;
            }
            match &self.schedule[step_i] {
                Step::BufferCopy {
                    src_byte_off,
                    dst_byte_off,
                    bytes,
                } => {
                    // wgpu forbids `copy_buffer_to_buffer` on the same buffer;
                    // use the generic copy compute kernel instead.
                    let src = *src_byte_off as u64;
                    let dst = *dst_byte_off as u64;
                    let nbytes = *bytes as u64;
                    let elems = (nbytes / 4).max(1) as u32;
                    let lo = src.min(dst);
                    let hi = src.saturating_add(nbytes).max(dst.saturating_add(nbytes));
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let span = hi.saturating_sub(lo).max(1);
                    let mut size = span.div_ceil(256) * 256;
                    size = size.max(256).min(max_binding);
                    let mut base = (lo / 256) * 256;
                    if base.saturating_add(size) > self.arena.size as u64 {
                        base = (self.arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let p = CopyParams {
                        n: elems,
                        in_off: (src.saturating_sub(base) / 4) as u32,
                        out_off: (dst.saturating_sub(base) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                        _p4: 0,
                    };
                    let ck = copy_kernel(&dev.device);
                    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu arena_copy uniform"),
                        size: std::mem::size_of::<CopyParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
                    let bg =
                        bind_two_buf0_window(&dev.device, ck, &self.arena.buffer, base, size, &u);
                    let mut enc =
                        dev.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rlx-wgpu arena_copy"),
                            });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("rlx-wgpu arena_copy pass"),
                            ..Default::default()
                        });
                        pass.set_pipeline(&ck.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        let (gx, gy, gz) = dispatch_dims(elems, 64);
                        pass.dispatch_workgroups(gx, gy, gz);
                    }
                    dev.queue.submit(std::iter::once(enc.finish()));
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                }
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
                Step::WelchPeaksHost { .. }
                | Step::LogMelHost { .. }
                | Step::LogMelBackwardHost { .. } => {
                    unreachable!("tail-host audio ops run after GPU wait")
                }
                Step::Im2ColHost {
                    x_byte_off,
                    col_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                } => {
                    crate::im2col_host::run_im2col(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off as usize,
                        *col_byte_off as usize,
                        *n,
                        *c_in,
                        *h,
                        *w,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
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

        self.dump_node_stats_if_requested(dev);

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

        if rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") {
            return self
                .graph
                .outputs
                .iter()
                .map(|&id| {
                    let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    vec![0.0; n]
                })
                .collect();
        }
        let out_ids: Vec<_> = self.graph.outputs.clone();
        read_f32_many_pooled(
            &self.arena,
            &dev.device,
            &dev.queue,
            &out_ids,
            &mut self.readback_staging,
        )
    }
}

/// Compute a (X, Y, 1) workgroup grid for a 1-D workload.
///
/// WebGPU caps `dispatch_workgroups` per-dimension at 65535. For
/// workloads beyond `65535 × workgroup_size_x` threads we split into
/// a 2-D grid; kernels recover the linear thread index via
/// `gid.x + gid.y * num_workgroups.x * 64u`.
fn dispatch_prologue_nchw(w: u32, h: u32, nc: u32) -> (u32, u32, u32) {
    (w.div_ceil(8).max(1), h.div_ceil(8).max(1), nc.max(1))
}

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

/// Shape/feature gate for CoopF16Vk (no operand tracing — avoids circular
/// dependency with compile-time f16 mirror planning).
///
/// **Default OFF.** The Vulkan/DX12 cooperative-matrix matmul path
/// silently produces wrong output on BERT-family attention chains on at
/// least RTX 4090 (verified empirically against Bio_ClinicalBERT:
/// encoder cosine collapses from ≈1.0 on the wide-F32 fallback to ≈0.09
/// when the coop path runs, regardless of whether the kernel uses
/// F16-acc or F32-acc accumulators). The root cause is upstream — likely
/// in how wgpu's `coopLoadT` / `coopMultiplyAdd` interact with strided
/// arena buffers on non-Apple drivers — and needs a focused
/// reproducer before it can be fixed in `rlx-wgpu`. Until then the
/// correctness-first default is to route Vulkan/DX12 matmuls through the
/// wide-F32 path, even though it's substantially slower (~80× on this
/// shape).
///
/// Opt back in (at the user's risk) with `RLX_WGPU_COOP_F16_VK_ENABLE=1`
/// — useful for measuring the perf headroom or for non-BERT models
/// where the precision loss may be acceptable. Legacy
/// `RLX_WGPU_NO_COOP_F16_VK=1` and explicit
/// `RLX_WGPU_COOP_F16_VK_DISABLE=1` are honored for completeness.
fn coop_f16_vk_eligible(dev: &wgpu::Device, m: u32, k: u32, n: u32) -> bool {
    if rlx_ir::env::flag("RLX_WGPU_NO_COOP_F16_VK")
        || rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_DISABLE")
    {
        return false;
    }
    if !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_ENABLE") {
        return false;
    }
    m.is_multiple_of(16)
        && k.is_multiple_of(16)
        && n.is_multiple_of(16)
        && dev
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && dev.features().contains(wgpu::Features::SHADER_F16)
        && crate::device::coop_discrete_backend()
        && crate::device::coop_f16_16x16_supported()
}

fn step_needs_pass_flush(step: &Step, prev: &Step) -> bool {
    match step {
        Step::CastF32ToF16 { .. } => matches!(
            prev,
            Step::Unary {
                f16_mirror: false,
                ..
            }
        ),
        Step::Matmul {
            compute_precision: MatmulCompute::CoopF16Vk,
            ..
        }
        | Step::MatmulQkv {
            kind: MatmulQkvKind::CoopF16Vk,
            ..
        } => matches!(prev, Step::Unary { .. } | Step::CastF32ToF16 { .. }),
        _ => false,
    }
}

fn dispatch_wide_f32_matmul(
    pass: &mut wgpu::ComputePass<'_>,
    mm_w_active: &Kernel,
    mm_k: &Kernel,
    m_s: u32,
    n: u32,
    batch: u32,
) {
    // Tile-size selection differs by GPU backend.
    //
    // **Vulkan / DX12** (`matmul_wide_nv`, 64×64 tile): when `m_s < 64`
    // the bottom rows of every workgroup's M-axis tile contain padded
    // zeros that the kernel still computes and writes back — pure
    // wasted work on small-M shapes like BERT-base prefill (m=32). The
    // regular 32×32-tile kernel sidesteps the M-axis padding and is
    // ~8% faster end-to-end on RTX 4090 (verified on Bio_ClinicalBERT:
    // encoder forward 58.9 ms → 54.1 ms at cosine 0.9999995 vs HF).
    //
    // **Metal / other** (`matmul_wide`, 64×64 tile): the wider tile
    // wins even on small M — Apple GPUs prefer the larger workgroup
    // and amortize the M-padding well. Forcing the 32×32 kernel here
    // regresses Mac WGPU encoder time (26.6 → 29.1 ms verified).
    let backend = wgpu_device()
        .map(|d| d.backend)
        .unwrap_or(wgpu::Backend::Noop);
    let is_vulkan_dx12 = matches!(backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12);
    let prefer_small_for_m = is_vulkan_dx12 && m_s < 64;
    let use_wide = !prefer_small_for_m && m_s >= 32 && n >= 64;
    if use_wide {
        pass.set_pipeline(&mm_w_active.pipeline);
        let (gx, gy) = if is_vulkan_dx12 {
            (n.div_ceil(64), m_s.div_ceil(64))
        } else {
            (n.div_ceil(64), m_s.div_ceil(32))
        };
        pass.dispatch_workgroups(gx, gy, batch);
    } else {
        pass.set_pipeline(&mm_k.pipeline);
        pass.dispatch_workgroups(n.div_ceil(32), m_s.div_ceil(32), batch);
    }
}

fn coop_f16_vk_bind_group(exe: &WgpuExecutable, gpu_bi: usize, use_wide: bool) -> &wgpu::BindGroup {
    if use_wide {
        exe.coop_f16_vk_wide_bind_groups
            .get(&gpu_bi)
            .unwrap_or(&exe.bind_groups[gpu_bi])
    } else {
        &exe.bind_groups[gpu_bi]
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

/// Bind the entire arena in one storage buffer range when it fits the device limit.
fn arena_whole_arena_bind(arena: &Arena, max_binding: u64) -> Option<(u64, u64)> {
    let need = arena.size as u64;
    if need > max_binding {
        return None;
    }
    // Bind size must not exceed the allocated buffer (planner may leave a small tail gap).
    let buf_bytes = arena.buffer.size();
    let size = need.min(buf_bytes).max(256);
    Some((0, size))
}

fn arena_window_for_nodes(dev: &wgpu::Device, arena: &Arena, ids: &[NodeId]) -> (u64, u64) {
    // wgpu requires storage buffer binding offsets aligned to 256 bytes.
    const ALIGN: u64 = 256;
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    if let Some(w) = arena_whole_arena_bind(arena, max_binding) {
        return w;
    }
    let mut lo: u64 = u64::MAX;
    let mut hi: u64 = 0;
    for &id in ids {
        let off = arena.offset(id) as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(off);
        hi = hi.max(off.saturating_add(len));
    }
    if lo == u64::MAX {
        return (0, max_binding.max(256));
    }
    let span = hi.saturating_sub(lo).max(1);
    if span > max_binding {
        let mut details = String::new();
        for &id in ids.iter().take(6) {
            let off = arena.offset(id);
            let len = arena.len_of(id);
            details.push_str(&format!(" id={id:?}@{off}+{len};"));
        }
        panic!(
            "rlx-wgpu: op needs {} bytes of arena span (>{});{}",
            span, max_binding, details
        );
    }
    let mut base = (lo / ALIGN) * ALIGN;
    // Bind only the byte span the op needs (not the full 4 GiB cap) so we
    // don't slide the window to the arena tail and drop low-offset tensors.
    let mut size = span.div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(max_binding);
    if base.saturating_add(size) > arena.size as u64 {
        base = (arena.size as u64).saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    if base > lo || base.saturating_add(size) < hi {
        base = (lo / ALIGN) * ALIGN;
        size = hi.saturating_sub(base).div_ceil(ALIGN) * ALIGN;
        size = size.max(256).min(max_binding);
        if base.saturating_add(size) > arena.size as u64 {
            base = hi.saturating_sub(size);
            base = (base / ALIGN) * ALIGN;
        }
    }
    (base, size)
}

fn arena_local_off_f32(arena: &Arena, id: NodeId, base: u64) -> u32 {
    (((arena.offset(id) as u64).saturating_sub(base)) / 4) as u32
}

fn arena_tensor_in_window(arena: &Arena, id: NodeId, base: u64, size: u64) -> bool {
    let src = arena.offset(id) as u64;
    let len = arena.len_of(id) as u64;
    src >= base && src.saturating_add(len) <= base.saturating_add(size)
}

/// True when two planned arena slots share any byte (memory planner reuse).
fn arena_tensors_overlap(arena: &Arena, a: NodeId, b: NodeId) -> bool {
    if a == b {
        return true;
    }
    let (a0, al) = (arena.offset(a) as u64, arena.len_of(a) as u64);
    let (b0, bl) = (arena.offset(b) as u64, arena.len_of(b) as u64);
    if al == 0 || bl == 0 {
        return false;
    }
    let a1 = a0.saturating_add(al);
    let b1 = b0.saturating_add(bl);
    a0 < b1 && b0 < a1
}

/// Arena bind window for matmul: when the weight alone fits the bind limit but
/// activations + weight do not, anchor on the param tensor (e.g. tied `LmHead`).
fn arena_matmul_bind_window(
    device: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    out_id: NodeId,
    a_id: NodeId,
    b_id: NodeId,
) -> (u64, u64, bool) {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if let Some((base, size)) = arena_whole_arena_bind(arena, max_binding) {
        return (base, size, false);
    }
    let ids = [out_id, a_id, b_id];
    let all_fits = arena_span_bytes(arena, &ids) <= max_binding;
    let b_bytes = arena.len_of(b_id) as u64;
    let b_is_param = tensor_is_graph_param(graph, param_offsets, b_id);
    let param_anchor =
        b_is_param && b_bytes <= max_binding && (!all_fits || b_bytes > ARENA_STAGE_CAP);
    let (mut base, mut size) = if param_anchor {
        arena_window_for_nodes(device, arena, &[b_id])
    } else if all_fits {
        arena_window_for_nodes(device, arena, &ids)
    } else {
        arena_window_for_nodes(device, arena, &[out_id])
    };
    let param_anchor = param_anchor
        || (b_is_param
            && b_bytes <= max_binding
            && !arena_tensor_in_window(arena, b_id, base, size));
    if param_anchor && !arena_tensor_in_window(arena, b_id, base, size) {
        (base, size) = arena_window_for_nodes(device, arena, &[b_id]);
    }
    (base, size, param_anchor)
}

/// Grow `[base, base+size)` to cover all listed tensors when the span still
/// fits `max_storage_buffer_binding_size` (avoids spurious staging copies).
fn arena_expand_bind_window(
    arena: &Arena,
    ids: &[NodeId],
    base: &mut u64,
    size: &mut u64,
    max_binding: u64,
) {
    const ALIGN: u64 = 256;
    let mut lo = *base;
    let mut hi = base.saturating_add(*size);
    for &id in ids {
        let off = arena.offset(id) as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(off);
        hi = hi.max(off.saturating_add(len));
    }
    let span = hi.saturating_sub(lo).max(1);
    if span > max_binding {
        return;
    }
    *base = (lo / ALIGN) * ALIGN;
    *size = span.div_ceil(ALIGN) * ALIGN;
    *size = (*size).max(256).min(max_binding);
    if (*base).saturating_add(*size) > arena.size as u64 {
        *base = (arena.size as u64).saturating_sub(*size);
        *base = (*base / ALIGN) * ALIGN;
    }
}

fn arena_off_in_bind_window(
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    device: &wgpu::Device,
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    id: NodeId,
    base: &mut u64,
    size: &mut u64,
) -> u32 {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if let Some((b, s)) = arena_whole_arena_bind(arena, max_binding) {
        *base = b;
        *size = s;
        return arena_local_off_f32(arena, id, b);
    }
    if arena_tensor_in_window(arena, id, *base, *size) {
        arena_local_off_f32(arena, id, *base)
    } else {
        let len = arena.len_of(id) as u64;
        if tensor_is_graph_param(graph, param_offsets, id) && len > max_binding {
            panic!(
                "rlx-wgpu: param node {:?} ({} bytes) exceeds max_storage_buffer_binding_size \
                 ({max_binding}); split weights or use f16 shadow binds",
                id, len
            );
        }
        if len > ARENA_STAGE_CAP {
            let op = &graph.node(id).op;
            panic!(
                "rlx-wgpu: bind_window would stage {} bytes for {:?} op={op:?} \
                 (off={}, base={}, bind_size={})",
                len,
                id,
                arena.offset(id),
                *base,
                *size,
            );
        }
        arena_off_in_window_or_stage(arena, schedule, scratch, base, size, max_binding, id)
    }
}

/// Bind window for ops that read/write multiple arena tensors (conv, concat, …).
/// Returns `(base, size)` and rebased f32 offsets; stages operands that fall outside
/// the window when the full span exceeds `max_storage_buffer_binding_size`.
fn arena_multi_op_window(
    dev: &wgpu::Device,
    arena: &Arena,
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    _schedule: &mut Vec<Step>,
    scratch: &mut u64,
    ids: &[NodeId],
) -> (u64, u64, bool) {
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    if let Some((base, size)) = arena_whole_arena_bind(arena, max_binding) {
        *scratch = arena.scratch_off as u64;
        return (base, size, false);
    }
    let param_anchor = if arena_span_bytes(arena, ids) > max_binding {
        ids.iter()
            .find(|&&id| {
                let nbytes = arena.len_of(id) as u64;
                tensor_is_graph_param(graph, param_offsets, id) && nbytes <= max_binding
            })
            .copied()
    } else {
        None
    };
    let mut param_anchored = param_anchor.is_some();
    let (mut base, mut size) = if arena_span_bytes(arena, ids) <= max_binding {
        arena_window_for_nodes(dev, arena, ids)
    } else if let Some(id) = param_anchor {
        arena_window_for_nodes(dev, arena, &[id])
    } else {
        arena_window_for_nodes(dev, arena, &[ids[0]])
    };
    if let Some(id) = param_anchor {
        if !arena_tensor_in_window(arena, id, base, size) {
            (base, size) = arena_window_for_nodes(dev, arena, &[id]);
        }
        param_anchored = true;
    } else {
        for &id in ids {
            let nbytes = arena.len_of(id) as u64;
            if tensor_is_graph_param(graph, param_offsets, id)
                && nbytes <= max_binding
                && !arena_tensor_in_window(arena, id, base, size)
            {
                (base, size) = arena_window_for_nodes(dev, arena, &[id]);
                param_anchored = true;
                break;
            }
        }
    }
    *scratch = arena.scratch_off as u64;
    if param_anchored {
        arena_ensure_scratch_in_window(scratch, base, size);
    }
    (base, size, param_anchored)
}

fn arena_bind_window_covering_scratch_if_needed(
    arena: &Arena,
    base: u64,
    size: u64,
    scratch: u64,
) -> u64 {
    // Planner places scratch at the arena tail; do not relocate the bind
    // window until this op has actually started staging into scratch.
    if scratch <= arena.scratch_off as u64 {
        return base;
    }
    if scratch >= base && scratch.saturating_add(ARENA_STAGE_CAP) <= base.saturating_add(size) {
        return base;
    }
    arena_window_covering_scratch(arena, base, size)
}

/// Keep staging writes inside `[base, base+size)` when the bind window is anchored on a
/// param far from the arena tail scratch zone.
fn arena_ensure_scratch_in_window(scratch: &mut u64, base: u64, size: u64) {
    let cap = ARENA_STAGE_CAP.min(size);
    let end = base.saturating_add(size);
    if *scratch < base || scratch.saturating_add(cap) > end {
        *scratch = end.saturating_sub(cap);
        *scratch = (*scratch / 256) * 256;
    }
}

#[allow(dead_code)]
fn arena_off_for_window(
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    id: NodeId,
    _window_ids: &[NodeId],
    mut base: u64,
    mut size: u64,
    max_binding: u64,
    _fits_in_one_binding: bool,
) -> u32 {
    let src = arena.offset(id) as u64;
    let len = arena.len_of(id) as u64;
    if src >= base && src.saturating_add(len) <= base.saturating_add(size) {
        arena_local_off_f32(arena, id, base)
    } else {
        arena_off_in_window_or_stage(
            arena,
            schedule,
            scratch,
            &mut base,
            &mut size,
            max_binding,
            id,
        )
    }
}

/// f16 shadow buffer window matching an f32 arena bind `[arena_base, arena_base+arena_size)`.
fn f16_shadow_bind_range(arena_base: u64, arena_size: u64, f16_buf_bytes: u64) -> (u64, u64) {
    const ALIGN: u64 = 256;
    let mut base = (arena_base / 2 / ALIGN) * ALIGN;
    let mut size = (arena_size / 2).div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(f16_buf_bytes);
    if base.saturating_add(size) > f16_buf_bytes {
        base = f16_buf_bytes.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    (base, size)
}

/// Window into `f16_buffer` for matmul weight reads (`params.b_off` is in
/// f16-element indices, matching the f32 arena word index).
fn f16_weight_bind_range(
    dev: &wgpu::Device,
    f16_buf_bytes: u64,
    b_off: u32,
    k: u32,
    n: u32,
    batch: u32,
    b_batch_stride: u32,
) -> (u64, u64, u32) {
    const ALIGN: u64 = 256;
    let max_binding = dev.limits().max_storage_buffer_binding_size;
    let b0 = b_off as u64;
    let span = (k as u64).saturating_mul(n as u64);
    let batch_n = batch.max(1) as u64;
    let stride = if batch_n > 1 {
        b_batch_stride as u64
    } else {
        span
    };
    let hi_elems = b0
        .saturating_add((batch_n - 1).saturating_mul(stride))
        .saturating_add(span);
    let lo_byte = b0.saturating_mul(2);
    let hi_byte = hi_elems.saturating_mul(2).saturating_add(8);
    let need = hi_byte.saturating_sub(lo_byte).max(1);
    if need > max_binding {
        panic!(
            "rlx-wgpu: f16 weight region needs {need} bytes (> {max_binding}); \
             matmul k={k} n={n} batch={batch}"
        );
    }
    let mut base = (lo_byte / ALIGN) * ALIGN;
    let mut size = need.div_ceil(ALIGN) * ALIGN;
    size = size.max(256).min(max_binding).min(f16_buf_bytes);
    if base.saturating_add(size) < hi_byte {
        base = hi_byte.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    if base.saturating_add(size) > f16_buf_bytes {
        base = f16_buf_bytes.saturating_sub(size);
        base = (base / ALIGN) * ALIGN;
    }
    let rebased = b_off.saturating_sub((base / 2) as u32);
    (base, size, rebased)
}

const ARENA_STAGE_CAP: u64 = 256 * 1024 * 1024;

/// Return a window-local f32 offset, staging into scratch when the tensor lies
/// outside the bind window (via `copy_buffer_to_buffer`).
fn arena_off_in_window_or_stage(
    arena: &Arena,
    schedule: &mut Vec<Step>,
    scratch: &mut u64,
    base: &mut u64,
    size: &mut u64,
    max_binding: u64,
    id: NodeId,
) -> u32 {
    let src = arena.offset(id) as u64;
    let len = arena.len_of(id) as u64;
    if src >= *base && src.saturating_add(len) <= (*base).saturating_add(*size) {
        return arena_local_off_f32(arena, id, *base);
    }
    if len > ARENA_STAGE_CAP {
        panic!(
            "rlx-wgpu: cannot stage {} bytes for node {:?} (cap {ARENA_STAGE_CAP})",
            len, id
        );
    }
    let aligned = len.div_ceil(256) * 256;
    let dst = *scratch;
    *scratch = scratch.saturating_add(aligned);
    schedule.push(Step::BufferCopy {
        src_byte_off: src as u32,
        dst_byte_off: dst as u32,
        bytes: len as u32,
    });
    let lo = (*base).min(dst);
    let hi = (*base)
        .saturating_add(*size)
        .max(dst.saturating_add(aligned));
    let span = hi.saturating_sub(lo).max(1);
    if span <= max_binding {
        const ALIGN: u64 = 256;
        *base = (lo / ALIGN) * ALIGN;
        *size = span.div_ceil(ALIGN) * ALIGN;
        *size = (*size).max(256).min(max_binding);
        if (*base).saturating_add(*size) > arena.size as u64 {
            *base = (arena.size as u64).saturating_sub(*size);
            *base = (*base / ALIGN) * ALIGN;
        }
    }
    if arena_tensor_in_window(arena, id, *base, *size) {
        arena_local_off_f32(arena, id, *base)
    } else {
        ((dst.saturating_sub(*base)) / 4) as u32
    }
}

/// If scratch does not fall inside `[base, base+size)`, slide the window to the tail.
fn arena_window_covering_scratch(arena: &Arena, base: u64, size: u64) -> u64 {
    let scratch = arena.scratch_off as u64;
    if scratch >= base && scratch.saturating_add(ARENA_STAGE_CAP) <= base.saturating_add(size) {
        return base;
    }
    let new_base = (arena.size as u64).saturating_sub(size);
    (new_base / 256) * 256
}

fn arena_span_bytes(arena: &Arena, ids: &[NodeId]) -> u64 {
    let mut lo: u64 = u64::MAX;
    let mut hi: u64 = 0;
    for &id in ids {
        let off = arena.offset(id) as u64;
        let len = arena.len_of(id) as u64;
        lo = lo.min(off);
        hi = hi.max(off.saturating_add(len));
    }
    if lo == u64::MAX {
        0
    } else {
        hi.saturating_sub(lo)
    }
}

#[allow(dead_code)]
fn bind_two(
    device: &wgpu::Device,
    kernel: &Kernel,
    buf0: &wgpu::Buffer,
    buf1: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if buf0.size() > max_binding {
        panic!(
            "rlx-wgpu: bind_two buffer {} bytes exceeds max_storage_buffer_binding_size {}; \
             use bind_two_buf0_window or bind_op_output_window",
            buf0.size(),
            max_binding
        );
    }
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

/// Windowed arena bind. When `operand_ids` is non-empty and their span with
/// `out_id` exceeds the binding limit, falls back to output-only window
/// (callers should stage operands and rebase offsets).
fn bind_op_output_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    out_id: NodeId,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    bind_op_window(device, kernel, arena, &[out_id], params)
}

fn bind_op_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    arena: &Arena,
    ids: &[NodeId],
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    let (base, size) = if arena_span_bytes(arena, ids) <= max_binding {
        arena_window_for_nodes(device, arena, ids)
    } else {
        arena_window_for_nodes(device, arena, &[ids[0]])
    };
    bind_two_buf0_window(device, kernel, &arena.buffer, base, size, params)
}

fn bind_two_buf0_window(
    device: &wgpu::Device,
    kernel: &Kernel,
    buf0: &wgpu::Buffer,
    buf0_base: u64,
    buf0_size: u64,
    buf1: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu bg window"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf0,
                    offset: buf0_base,
                    size: NonZeroU64::new(buf0_size),
                }),
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
    mirror_acts: &HashSet<NodeId>,
    a_id: NodeId,
    b_id: NodeId,
    m: u32,
    k: u32,
    n: u32,
) -> MatmulCompute {
    if rlx_ir::env::flag("RLX_WGPU_MATMUL_F32_ONLY") {
        return MatmulCompute::F32;
    }
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
    //
    // Vulkan uses `matmul_coop_f32_portable` (8×8 tiles, coopLoadT) which
    // only requires k%8 and n%8.
    let coop16_aligned = m.is_multiple_of(32) && k.is_multiple_of(8) && n.is_multiple_of(32);
    let coop_f32_metal_aligned = k.is_multiple_of(8) && n.is_multiple_of(32);
    let coop_f32_portable_aligned = k.is_multiple_of(8) && n.is_multiple_of(8);
    let has_coop = dev
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    let backend = crate::device::wgpu_device().map(|d| d.backend);
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
    if !any_low && coop_f16_vk_eligible(dev, m, k, n) {
        if traces_to_param(graph, b_id)
            && !mirror_acts.contains(&a_id)
            && !mirror_acts.contains(&b_id)
        {
            return MatmulCompute::CoopF16Vk;
        }
    }
    // CoopF32 (`simdgroup_float8x8` on Apple): the f32 hardware-GEMM
    // path. Used whenever cooperative-matrix is available, B is a
    // Param, and shapes align — gives ~5-10× speedup over the
    // tiled `matmul_wide` path with no precision loss vs the f32
    // baseline (BERT max|Δ| stays at 2.3e-3 vs CPU on Apple).
    //
    // CoopF32: Metal-only by default. Vulkan portable 8×8 is opt-in via
    // RLX_WGPU_FORCE_COOP_F32 (RTX lacks 8×8 f32 coop; output is unreliable).
    let disabled = rlx_ir::env::flag("RLX_WGPU_NO_COOP_F32");
    let forced = rlx_ir::env::flag("RLX_WGPU_FORCE_COOP_F32");
    let metal_coop = !disabled
        && has_coop
        && coop_f32_metal_aligned
        && traces_to_param(graph, b_id)
        && (forced || matches!(backend, Some(wgpu::Backend::Metal)));
    let vulkan_coop = !disabled
        && has_coop
        && coop_f32_portable_aligned
        && traces_to_param(graph, b_id)
        && crate::device::coop_discrete_backend()
        && crate::device::coop_f32_8x8_supported();
    if metal_coop
        || vulkan_coop
        || (forced
            && has_coop
            && traces_to_param(graph, b_id)
            && (coop_f32_metal_aligned || coop_f32_portable_aligned))
    {
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
fn node_is_arena_param(param_offsets: &HashMap<String, NodeId>, id: NodeId) -> bool {
    param_offsets.values().any(|&nid| nid == id)
}

fn traces_to_param(graph: &Graph, mut id: NodeId) -> bool {
    loop {
        let node = graph.node(id);
        match &node.op {
            Op::Param { .. } => return true,
            Op::Cast { .. } | Op::Reshape { .. } | Op::Transpose { .. } => {
                if node.inputs.is_empty() {
                    return false;
                }
                id = node.inputs[0];
            }
            _ => return false,
        }
    }
}

fn tensor_is_graph_param(
    graph: &Graph,
    param_offsets: &HashMap<String, NodeId>,
    id: NodeId,
) -> bool {
    node_is_arena_param(param_offsets, id) || traces_to_param(graph, id)
}

fn traces_to_input(graph: &Graph, mut id: NodeId) -> bool {
    loop {
        let node = graph.node(id);
        match &node.op {
            Op::Input { .. } => return true,
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

/// Mirror A/B into the f16 shadow buffer before CoopF16Vk when the operand
/// is not already mirrored (Inputs/Params are written via `write_f32`).
fn schedule_uses_coop_f16_vk(schedule: &[Step]) -> bool {
    schedule.iter().any(|s| {
        matches!(
            s,
            Step::Matmul {
                compute_precision: MatmulCompute::CoopF16Vk,
                ..
            } | Step::MatmulQkv {
                kind: MatmulQkvKind::CoopF16Vk,
                ..
            }
        )
    })
}

fn register_coop_f16_vk_b_param(
    map: &mut HashMap<u32, String>,
    param_offsets: &HashMap<String, NodeId>,
    b_id: NodeId,
    b_off_f32: u32,
    compute: MatmulCompute,
) {
    if compute != MatmulCompute::CoopF16Vk {
        return;
    }
    for (name, &id) in param_offsets {
        if id == b_id {
            map.insert(b_off_f32, name.clone());
            return;
        }
    }
}

fn tensor_host_name(
    input_offsets: &HashMap<String, NodeId>,
    param_offsets: &HashMap<String, NodeId>,
    id: NodeId,
) -> String {
    for (name, &nid) in input_offsets {
        if nid == id {
            return name.clone();
        }
    }
    for (name, &nid) in param_offsets {
        if nid == id {
            return name.clone();
        }
    }
    panic!("rlx-wgpu: CoopF16Vk host activation source {id} is not an input or param");
}

fn host_tensor_f32<'a>(
    name: &str,
    inputs: &'a [(&str, &[f32])],
    stashed_params: &'a HashMap<String, Vec<f32>>,
) -> Option<&'a [f32]> {
    inputs
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, d)| *d)
        .or_else(|| stashed_params.get(name).map(|v| v.as_slice()))
}

fn apply_activation_host(act: Activation, data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&x| match act {
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Exp => x.exp(),
            Activation::Log => x.ln(),
            Activation::Sqrt => x.sqrt(),
            Activation::Rsqrt => 1.0 / x.sqrt(),
            Activation::Neg => -x,
            Activation::Abs => x.abs(),
            Activation::Gelu | Activation::GeluApprox => {
                let c = 0.797_884_6_f32;
                let x3 = x * x * x;
                let inner = (c * (x + 0.044_715 * x3)).clamp(-15.0, 15.0);
                0.5 * x * (1.0 + inner.tanh())
            }
            Activation::Silu => {
                let nx = (-x).clamp(-88.0, 88.0);
                x / (1.0 + nx.exp())
            }
            Activation::Round => x.round(),
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            Activation::Tan => x.tan(),
            Activation::Atan => x.atan(),
        })
        .collect()
}

/// Activation node ids consumed as CoopF16Vk matmul A/B operands.
fn collect_coop_f16_vk_mirror_activations(graph: &Graph, dev: &wgpu::Device) -> HashSet<NodeId> {
    let mut acts = HashSet::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        let a_id = node.inputs[0];
        let b_id = node.inputs[1];
        let a_shape = graph.node(a_id).shape.dims();
        let b_shape = graph.node(b_id).shape.dims();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            continue;
        }
        let m = a_shape[0].unwrap_static() as u32;
        let k = a_shape[1].unwrap_static() as u32;
        let n = b_shape[1].unwrap_static() as u32;
        if !coop_f16_vk_eligible(dev, m, k, n) || !traces_to_param(graph, b_id) {
            continue;
        }
        if matches!(graph.node(a_id).op, Op::Activation(_)) {
            acts.insert(a_id);
        }
        if matches!(graph.node(b_id).op, Op::Activation(_)) {
            acts.insert(b_id);
        }
    }
    acts
}

/// When A/B are computed (not Input/Param), mirror f32 arena into f16 shadow
/// via `cast_f32_to_f16` before CoopF16Vk matmul (non-activation intermediates).
fn maybe_push_coop_f16_vk_casts(
    graph: &Graph,
    a_id: NodeId,
    b_id: NodeId,
    mirror_acts: &HashSet<NodeId>,
    device: &wgpu::Device,
    arena: &Arena,
    schedule: &mut Vec<Step>,
    uniforms: &mut Vec<wgpu::Buffer>,
    bind_groups: &mut Vec<wgpu::BindGroup>,
    mm_cast: &Option<&'static Kernel>,
    compute_precision: MatmulCompute,
    a_off_f32: u32,
    m: u32,
    k: u32,
    batch: u32,
    b_off_f32: u32,
    n: u32,
) {
    if compute_precision != MatmulCompute::CoopF16Vk {
        return;
    }
    let batch_n = batch.max(1);
    if !traces_to_input(graph, a_id)
        && !traces_to_param(graph, a_id)
        && !mirror_acts.contains(&a_id)
    {
        let a_elems = m.saturating_mul(k).saturating_mul(batch_n);
        let (base, size) = arena_window_for_nodes(device, arena, &[a_id]);
        push_cast_f32_to_f16_step(
            device,
            arena,
            base,
            size,
            schedule,
            uniforms,
            bind_groups,
            mm_cast,
            a_off_f32,
            a_elems,
        );
    }
    if !traces_to_input(graph, b_id)
        && !traces_to_param(graph, b_id)
        && !mirror_acts.contains(&b_id)
    {
        let b_elems = k.saturating_mul(n).saturating_mul(batch_n);
        let (base, size) = arena_window_for_nodes(device, arena, &[b_id]);
        push_cast_f32_to_f16_step(
            device,
            arena,
            base,
            size,
            schedule,
            uniforms,
            bind_groups,
            mm_cast,
            b_off_f32,
            b_elems,
        );
    }
}

fn build_matmul_qkv_coop_f16_vk_bind_group(
    device: &wgpu::Device,
    mqk: &Kernel,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    params: &wgpu::Buffer,
    k: u32,
    n: u32,
    b_off: u32,
) -> (wgpu::BindGroup, u32) {
    let f16_buf = arena
        .f16_buffer
        .as_ref()
        .expect("CoopF16Vk QKV requires SHADER_F16 f16 shadow arena");
    let (f16_res, rebased_b) = {
        let (base, size, rebased) =
            f16_weight_bind_range(device, f16_buf.size(), b_off, k, n, 1, 0);
        (
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: f16_buf,
                offset: base,
                size: NonZeroU64::new(size),
            }),
            rebased,
        )
    };
    (
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rlx-wgpu matmul_qkv_coop_f16_vk bg"),
            layout: &mqk.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: f16_res,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &arena.buffer,
                        offset: arena_base,
                        size: NonZeroU64::new(arena_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        }),
        rebased_b,
    )
}
/// Append a CastF32ToF16 pre-pass: mirrors `arena[off..off+len]` (f32) into
/// `arena_f16[off..off+len]` (f16) so coop matmul kernels can read operands
/// as f16. Used before CoopF16Vk when A/B are computed activations.
fn push_cast_f32_to_f16_step(
    device: &wgpu::Device,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
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
        src_off: src_off.saturating_sub((arena_base / 4) as u32),
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
    let (f16_base, f16_size) = f16_shadow_bind_range(arena_base, arena_size, f16_buf.size());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu cast_f32_to_f16 bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: f16_buf,
                    offset: f16_base,
                    size: NonZeroU64::new(f16_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &arena.buffer,
                    offset: arena_base,
                    size: NonZeroU64::new(arena_size),
                }),
            },
        ],
    });
    schedule.push(Step::CastF32ToF16 { params: p });
    uniforms.push(u);
    bind_groups.push(bg);
}

/// Per-Matmul-step bind group builder. Returns `(bind_group, rebased_b_off)`;
/// `rebased_b_off` adjusts `MatmulParams.b_off` when the f16 weight buffer is
/// window-bound.
fn build_matmul_bind_group(
    device: &wgpu::Device,
    mm_k: &Kernel,
    _mm_w: &Kernel,
    mm_f16w: &Option<&'static Kernel>,
    mm_f16c: &Option<&'static Kernel>,
    mm_coop: &Option<&'static Kernel>,
    mm_coop_f32: &Option<&'static Kernel>,
    arena: &Arena,
    arena_base: u64,
    arena_size: u64,
    params: &wgpu::Buffer,
    b_is_param: bool,
    compute_precision: MatmulCompute,
    k: u32,
    n: u32,
    batch: u32,
    b_off: u32,
    b_batch_stride: u32,
) -> (wgpu::BindGroup, u32) {
    let f16_bind = |b_off: u32| -> (wgpu::BindingResource<'_>, u32) {
        let f16_buf = arena
            .f16_buffer
            .as_ref()
            .expect("f16 weight bind without f16_buffer");
        let (base, size, rebased) =
            f16_weight_bind_range(device, f16_buf.size(), b_off, k, n, batch, b_batch_stride);
        (
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: f16_buf,
                offset: base,
                size: NonZeroU64::new(size),
            }),
            rebased,
        )
    };
    if compute_precision == MatmulCompute::CoopF16Vk
        && let (Some(coop_vk), Some(_f16_buf)) =
            (matmul_coop_f16_vulkan_kernel(device), &arena.f16_buffer)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f16_vulkan bg"),
                layout: &coop_vk.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: f16_res,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena.buffer,
                            offset: arena_base,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            }),
            rebased_b,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::CoopF32
        && let Some(coop_f32) = mm_coop_f32
    {
        // 2-binding layout — both A and B come from the f32 arena
        // (no f16 shadow buffer needed for the pure-f32 path).
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop_f32 bg"),
                layout: &coop_f32.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena.buffer,
                            offset: arena_base,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            }),
            b_off,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::Coop16
        && let (Some(_f16_buf), Some(coop)) = (&arena.f16_buffer, mm_coop)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        // 3-binding layout — A is staged from arena (f32) through
        // workgroup-shared memory inside the kernel, no separate
        // f16 binding for A.
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_coop16 bg"),
                layout: &coop.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena.buffer,
                            offset: arena_base,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    }, // weights
                ],
            }),
            rebased_b,
        );
    }
    if b_is_param
        && compute_precision == MatmulCompute::F16
        && let (Some(_f16_buf), Some(f16c)) = (&arena.f16_buffer, mm_f16c)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_f16_compute bg"),
                layout: &f16c.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena.buffer,
                            offset: arena_base,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    },
                ],
            }),
            rebased_b,
        );
    }
    let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
    if b_is_param
        && f16w_opt_in
        && let (Some(_f16_buf), Some(f16w)) = (&arena.f16_buffer, mm_f16w)
    {
        let (f16_res, rebased_b) = f16_bind(b_off);
        return (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rlx-wgpu matmul_f16w bg"),
                layout: &f16w.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena.buffer,
                            offset: arena_base,
                            size: NonZeroU64::new(arena_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f16_res,
                    },
                ],
            }),
            rebased_b,
        );
    }
    (
        bind_two_buf0_window(device, mm_k, &arena.buffer, arena_base, arena_size, params),
        b_off,
    )
}
