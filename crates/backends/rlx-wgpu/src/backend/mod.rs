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

use rlx_ir::dynamic::{bind_graph, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};

use crate::buffer::{
    Arena, ReadbackStaging, TinyReadbackStaging,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ArgmaxParams, AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams,
    Conv1dParams, Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams,
    DequantMatmulParams, ElementwiseRegionParams, ExpandParams, FmaParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, GruParams, Kernel, LayerNormBwdParams, LayerNormParams,
    Mamba2Params, MatmulQkvParams, NarrowConcatParams, Pool1dParams, Pool2dParams,
    Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams,
    SampleParams, ScatterAddParams, SceParams, SelectiveScanParams, SoftmaxParams, TopKParams,
    TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
    gather_split_kernel, matmul_coop_f16_vulkan_kernel,
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
    Fma {
        params: FmaParams,
    },
    Reduce {
        params: ReduceParams,
    },
    Softmax {
        params: SoftmaxParams,
    },
    SoftmaxCrossEntropy {
        params: SceParams,
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
    /// General `Op::Scan` recurrence — D2H the input span → run the compiled
    /// body loop on the CPU → H2D. Enables IIR (`biquad`/`sosfilt`) on wgpu.
    ScanHost {
        plan: std::sync::Arc<rlx_cpu::thunk::ScanBodyPlan>,
        outer_init_off: usize,
        outer_final_off: usize,
        length: u32,
        save_trajectory: bool,
        xs_outer: Vec<(usize, usize)>,
        bcast_outer: Vec<(usize, usize)>,
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
    /// Host fill for [`Op::RngNormal`] (fill → H2D).
    RngNormalHost {
        dst_byte_off: u32,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    /// Host fill for [`Op::RngUniform`] (fill → H2D).
    RngUniformHost {
        dst_byte_off: u32,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    /// Host-side buffer copy (recorded into a command encoder) used to
    /// stage small param tensors into the tail scratch region so kernels
    /// can bind a ≤4GiB window of the arena.
    BufferCopy {
        // u64: staging copies for >4 GiB GGUF decode arenas address tensors past
        // the 4 GiB mark; a u32 byte offset truncates and stages garbage.
        src_byte_off: u64,
        dst_byte_off: u64,
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
    WelchPeaksGpu {
        params: WelchPeaksGpuParams,
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
    /// Native WGSL Mamba-2 SSD scan (state_size ≤ 256).
    Mamba2 {
        params: Mamba2Params,
    },
    /// Native WGSL GRU (single-layer/unidir/no-carry, hidden ≤ 256).
    Gru {
        params: GruParams,
    },
    /// Native WGSL Elman RNN (single-layer/unidir/no-carry, hidden ≤ 256).
    Rnn {
        params: RnnParams,
    },
    /// Host-staged GRU fallback (multi-layer / bidir / carry / hidden > 256).
    GruHost {
        x: u32,
        w_ih: u32,
        w_hh: u32,
        b_ih: u32,
        b_hh: u32,
        h0: u32,
        dst: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// Host-staged Elman RNN fallback.
    RnnHost {
        x: u32,
        w_ih: u32,
        w_hh: u32,
        bias: u32,
        h0: u32,
        dst: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
        relu: bool,
    },
    DequantMatmul {
        params: DequantMatmulParams,
    },
    /// Split-binding embedding gather for >4 GiB arenas. The table and the
    /// idx/output slots are more than one ≤4 GiB binding window apart, so the
    /// single-arena-binding `Step::Gather` cannot reach the output. Runs as a
    /// host segment (its own submission + copy-back), like `DequantMatmulGguf`.
    /// BYTE offsets are u64 (arena exceeds 4 GiB).
    GatherSplit {
        n_out: u32,
        n_idx: u32,
        dim: u32,
        vocab: u32,
        table_byte_off: u64,
        idx_byte_off: u64,
        out_byte_off: u64,
    },
    /// GGUF K-quant — host fused dequant+matmul between GPU segments.
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        // Arena BYTE offsets must be u64: GGUF decode arenas exceed 4 GiB
        // (Orpheus-3B Q4_K_M is ~10 GiB), so a u32 byte offset truncates for
        // any tensor past the 4 GiB mark and the host dequant reads garbage.
        x_byte_off: u64,
        w_byte_off: u64,
        out_byte_off: u64,
    },
    /// GGUF K-quant — host fused dequant+grouped matmul between GPU segments.
    DequantGroupedMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme_id: u32,
        x_byte_off: u64,
        w_byte_off: u64,
        idx_byte_off: u64,
        out_byte_off: u64,
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
    Lstm {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        bias_byte_off: u32,
        h0_byte_off: u32,
        c0_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    ConvTranspose2d {
        src_byte_off: u32,
        weight_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
    },
    /// Host-staged NCHW GroupNorm (readback → CPU → writeback).
    GroupNormHost {
        src_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps: f32,
    },
    /// Host-staged NCHW LayerNorm2d (readback → CPU → writeback).
    LayerNorm2dHost {
        src_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps: f32,
    },
    /// Host-staged nearest 2× upsample on NCHW (readback → CPU → writeback).
    ResizeNearest2xHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Host-staged batch-general reverse/flip (readback → CPU → writeback).
    ReverseHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u32,
    },
    /// Host-staged ArgMax/ArgMin (readback → CPU → writeback).
    ArgReduceHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
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
    /// Fused multi-scale deformable attention (host compute over arena buffers).
    MsDeformAttnHost {
        in_offs: Vec<(u32, u32)>, // (byte_off, byte_len) per input
        out_byte_off: u32,
        out_bytes: u32,
        attrs: Vec<u8>,
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
    /// Byte offset of GGUF dequant scratch slab (0 when host fallback).
    dequant_scratch_off: usize,
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
    /// When set, `run_inner` dispatches + submits all compute but skips the
    /// blocking output readback (results stay in the arena). Used by the wasm
    /// `run_async` path, which then reads outputs back asynchronously — the
    /// browser event loop cannot be blocked. Always false on native.
    dispatch_only: bool,
    /// Per-`FftGpu` step: isolated uniform buffers + bind groups (one vec entry per op).
    fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources>,
    /// Persistent KV inputs (host staging uploaded each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Arena input slots authoritative — skip host KV mirror each decode step.
    gpu_handle_resident: HashSet<String>,
    pending_read_indices: Option<Vec<usize>>,
    /// Runtime-mutable RNG policy for [`Step::RngNormalHost`] / [`Step::RngUniformHost`].
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
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
            | Step::Fma { .. }
            | Step::Reduce { .. }
            | Step::Softmax { .. }
            | Step::SoftmaxCrossEntropy { .. }
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
            | Step::WelchPeaksGpu { .. }
            | Step::Sample { .. }
            | Step::Gather { .. }
            | Step::GatherAxis { .. }
            | Step::GatherSplit { .. }
            | Step::GroupedMatmul { .. }
            | Step::DequantMatmul { .. }
            | Step::DequantMatmulGguf { .. }
            | Step::DequantGroupedMatmulGguf { .. }
            | Step::GatedDeltaNet { .. }
            | Step::Lstm { .. }
            | Step::ConvTranspose2d { .. }
            | Step::GroupNormHost { .. }
            | Step::LayerNorm2dHost { .. }
            | Step::ResizeNearest2xHost { .. }
            | Step::ReverseHost { .. }
            | Step::ArgReduceHost { .. }
            | Step::GruHost { .. }
            | Step::RnnHost { .. }
            | Step::Llada2GroupLimitedGate { .. }
            | Step::UmapKnn { .. }
            | Step::UmapKnnHost { .. }
            | Step::MsDeformAttnHost { .. }
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
            Step::FftGpu { .. } | Step::FftHost { .. } | Step::ScanHost { .. } => true,
            Step::Im2ColHost { .. }
            | Step::RngNormalHost { .. }
            | Step::RngUniformHost { .. }
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
            // Mamba2: same seq_stride discipline as SelectiveScan.
            Step::Mamba2 { .. } => true,
            // GRU/RNN: per-batch workgroups; seq_stride is full-extent, seq is
            // the loop bound only. Safe under active-extent scaling.
            Step::Gru { .. } => true,
            Step::Rnn { .. } => true,
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
        Step::Fma { .. } => "fma",
        Step::Reduce { .. } => "reduce",
        Step::Softmax { .. } => "softmax",
        Step::SoftmaxCrossEntropy { .. } => "softmax_cross_entropy",
        Step::LayerNorm { .. } => "layer_norm",
        Step::Cumsum { .. } => "cumsum",
        Step::FftGpu { .. } => "fft_gpu",
        Step::FftHost { .. } => "fft_host",
        Step::WelchPeaksHost { .. } => "welch_peaks_host",
        Step::LogMelHost { .. } => "log_mel_host",
        Step::LogMelBackwardHost { .. } => "log_mel_backward_host",
        Step::Im2ColHost { .. } => "im2col_host",
        Step::RngNormalHost { .. } => "rng_normal_host",
        Step::RngUniformHost { .. } => "rng_uniform_host",
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
        Step::WelchPeaksGpu { .. } => "welch_peaks_gpu",
        Step::GroupedMatmul { .. } => "grouped_matmul",
        Step::Sample { .. } => "sample",
        Step::SelectiveScan { .. } => "selective_scan",
        Step::Mamba2 { .. } => "mamba2",
        Step::Gru { .. } => "gru",
        Step::Rnn { .. } => "rnn",
        Step::GruHost { .. } => "gru_host",
        Step::RnnHost { .. } => "rnn_host",
        Step::DequantMatmul { .. } => "dequant_matmul",
        Step::GatherSplit { .. } => "gather_split",
        Step::DequantMatmulGguf { .. } => "dequant_matmul_gguf",
        Step::DequantGroupedMatmulGguf { .. } => "dequant_grouped_matmul_gguf",
        Step::GatedDeltaNet { .. } => "gated_delta_net",
        Step::Lstm { .. } => "lstm",
        Step::ConvTranspose2d { .. } => "conv_transpose2d",
        Step::GroupNormHost { .. } => "group_norm_host",
        Step::LayerNorm2dHost { .. } => "layer_norm2d_host",
        Step::ResizeNearest2xHost { .. } => "resize_nearest2x_host",
        Step::ReverseHost { .. } => "reverse_host",
        Step::ArgReduceHost { .. } => "argreduce_host",
        Step::Llada2GroupLimitedGate { .. } => "llada2_group_limited_gate",
        Step::UmapKnn { .. } => "umap_knn",
        Step::UmapKnnHost { .. } => "umap_knn_host",
        Step::MsDeformAttnHost { .. } => "ms_deform_attn_host",
        Step::ScanHost { .. } => "scan_host",
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
        Step::GatherSplit { .. }
        | Step::DequantMatmulGguf { .. }
        | Step::DequantGroupedMatmulGguf { .. }
        | Step::GatedDeltaNet { .. }
        | Step::Lstm { .. }
        | Step::ConvTranspose2d { .. }
        | Step::GroupNormHost { .. }
        | Step::LayerNorm2dHost { .. }
        | Step::ResizeNearest2xHost { .. }
        | Step::ReverseHost { .. }
        | Step::ArgReduceHost { .. }
        | Step::GruHost { .. }
        | Step::RnnHost { .. }
        | Step::Llada2GroupLimitedGate { .. }
        | Step::UmapKnnHost { .. }
        | Step::MsDeformAttnHost { .. }
        | Step::FftHost { .. }
        | Step::ScanHost { .. }
        | Step::Im2ColHost { .. }
        | Step::RngNormalHost { .. }
        | Step::RngUniformHost { .. }
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

mod compile;
mod dispatch;
mod run;
mod set;
mod test;

impl WgpuExecutable {
    /// Resolve the deferred graph against bindings inferred from
    /// `inputs`, recompile the inner state if the bindings changed
    /// since the last call, and replay any pending params.
    pub(crate) fn lazy_compile_for_inputs(&mut self, inputs: &[(&str, &[f32])]) {
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

        let fresh = Self::compile_static_inner(resolved, self.rng.clone());

        // Move the freshly-compiled fields into self, preserve the
        // unresolved+binding state for the next round.
        self.graph = fresh.graph;
        self.arena = fresh.arena;
        self.dequant_scratch_off = fresh.dequant_scratch_off;
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


    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }


    /// Compile placeholder for a graph with `Dim::Dynamic` entries.
    /// The real compile happens on the first `run()` once input data
    /// reveals the symbol → size bindings. Buffered params (set via
    /// `set_param` / `set_param_bytes` before run) are replayed.
    pub(crate) fn deferred(graph: Graph, rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>) -> Self {
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
            dequant_scratch_off: 0,
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
            dispatch_only: false,
            fft_gpu_steps: Vec::new(),
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            pending_read_indices: None,
            rng,
        }
    }


    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
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


    pub(crate) fn dump_node_stats_if_requested(&self, dev: &crate::device::WgpuDevice) {
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


    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let graph = self
            .unresolved
            .clone()
            .unwrap_or_else(|| self.graph.clone());
        let mut exe = Self::compile_rng(graph, self.rng());
        for (k, v) in &self.stashed_params {
            exe.set_param(k, v);
        }
        for (k, v) in &self.pending_params {
            exe.set_param(k, v);
        }
        for (k, v) in &self.pending_param_bytes {
            exe.set_param_bytes(k, v);
        }
        for (k, v) in &self.gpu_handles {
            exe.bind_gpu_handle(k, v);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe.set_rng(self.rng());
        exe
    }


    pub(crate) fn readback_plan(&self) -> Vec<usize> {
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


    pub(crate) fn propagate_gpu_handle_feeds_on_gpu(
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


    pub(crate) fn stage_gpu_handle_inputs(
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


    pub(crate) fn pack_readback_outputs(&mut self, plan: &[usize], partial: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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

/// Split-binding embedding gather for >4 GiB arenas (see `Step::GatherSplit`).
///
/// The table and the idx/output slots can be more than one ≤4 GiB binding
/// window apart, so they're bound as separate read-only windows of the arena;
/// the output goes to a dedicated read-write buffer that is copied back into
/// the arena afterwards (the arena cannot also be bound read-write in the same
/// dispatch — wgpu treats STORAGE_READ_WRITE as exclusive per buffer). Mirrors
/// [`crate::gguf_gpu::run_dequant_matmul_gguf_gemv`].
#[allow(clippy::too_many_arguments)]
fn run_gather_split(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n_out: u32,
    n_idx: u32,
    dim: u32,
    vocab: u32,
    table_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
) {
    const ALIGN: u64 = 256;
    let arena_size = arena.size as u64;
    let max_bind = device.limits().max_storage_buffer_binding_size;

    // Table window: cover [table_byte_off, + vocab*dim*4).
    let t0 = table_byte_off as u64;
    let t_bytes = (vocab as u64) * (dim as u64) * 4;
    let t_base = (t0 / ALIGN) * ALIGN;
    let t_size = ((t0 + t_bytes - t_base).div_ceil(16) * 16).min(arena_size - t_base);

    // Index window: cover [idx_byte_off, + n_idx*4).
    let i0 = idx_byte_off as u64;
    let i_bytes = ((n_idx as u64) * 4).max(4);
    let i_base = (i0 / ALIGN) * ALIGN;
    let i_size = ((i0 + i_bytes - i_base).div_ceil(16) * 16).min(arena_size - i_base);

    assert!(
        t_size <= max_bind && i_size <= max_bind,
        "rlx-wgpu gather_split: window too large (table={t_size}, idx={i_size}, max={max_bind})"
    );

    // Separate output buffer (rw) — copied into the arena after the dispatch.
    let out_bytes = ((n_out as u64) * 4).max(4);
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gather_split out"),
        size: out_bytes.div_ceil(16) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let p = GatherParams {
        n_out,
        n_idx,
        dim,
        vocab,
        in_off: ((t0 - t_base) / 4) as u32,
        idx_off: ((i0 - i_base) / 4) as u32,
        out_off: 0,
        _p0: 0,
    };
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gather_split uniform"),
        size: std::mem::size_of::<GatherParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));

    let gk = gather_split_kernel(device);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu gather_split bg"),
        layout: &gk.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &arena.buffer,
                    offset: t_base,
                    size: wgpu::BufferSize::new(t_size),
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
                    offset: i_base,
                    size: wgpu::BufferSize::new(i_size),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu gather_split"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu gather_split pass"),
            ..Default::default()
        });
        pass.set_pipeline(&gk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(n_out, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    // Copy the embedding back into the arena (distinct buffers → legal).
    enc.copy_buffer_to_buffer(&out_buf, 0, &arena.buffer, out_byte_off as u64, out_bytes);
    queue.submit(std::iter::once(enc.finish()));
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

/// True when a matmul reads its weight `B` from the separate f16 shadow buffer
/// (so `B` is NOT bound through the arena binding). For these precisions the
/// arena window must cover only the activation + output, never the weight.
fn matmul_b_from_f16(precision: MatmulCompute, b_is_param: bool) -> bool {
    b_is_param
        && matches!(
            precision,
            MatmulCompute::F16 | MatmulCompute::Coop16 | MatmulCompute::CoopF16Vk
        )
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
    b_in_arena: bool,
) -> (u64, u64, bool) {
    let max_binding = device.limits().max_storage_buffer_binding_size;
    if let Some((base, size)) = arena_whole_arena_bind(arena, max_binding) {
        return (base, size, false);
    }
    if !b_in_arena {
        // B is read from the separate f16 shadow buffer (F16 / Coop16 /
        // CoopF16Vk), so the arena binding only carries the activation A and
        // the output C. Anchor on [out, a] — NOT on B. Anchoring on a >4 GiB
        // param B (e.g. the tied F32 lm_head/embedding) would push the window
        // onto the weight and leave C outside it, dropping the output write
        // (logits stayed mostly zero → argmax = STOP).
        let (base, size) = arena_window_for_nodes(device, arena, &[out_id, a_id]);
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

/// Output spatial positions computed per thread by `conv2d.wgsl` (register
/// tiling for weight reuse). MUST equal `TILE` in that kernel.
const CONV2D_TILE: u32 = 4;

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
        src_byte_off: src,
        dst_byte_off: dst,
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
    // Metal `simdgroup_float8x8` CoopF32 produced ORTHOGONAL GARBAGE (not mere
    // imprecision) on Gemma-4 vision/audio f32 param-weight matmuls — e.g.
    // scaled[1,64,768] @ input_proj[768,768] (all axes aligned) gave cos 0.016
    // vs CPU; forcing the plain F32 kernel restores cos 1.0. GGUF text models
    // dodged this via the DequantMatMul path. Until the kernel is root-caused,
    // Metal CoopF32 is opt-in via RLX_WGPU_FORCE_COOP_F32.
    let metal_coop = !disabled
        && has_coop
        && coop_f32_metal_aligned
        && traces_to_param(graph, b_id)
        && forced;
    let _ = backend;
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

