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

//! `CudaExecutable` — lowers an rlx-ir Graph into a sequence of CUDA
//! kernel launches against a pre-allocated device buffer.
//!
//! v2 op coverage: MatMul (tiled SGEMM), Binary, Compare, Activation, Where,
//! Reduce, Softmax, LayerNorm, RmsNorm, FusedResidualLN, Gather, Narrow,
//! Argmax, Reshape/Cast (no-op via slot aliasing), leaf nodes. Anything
//! else panics at compile time with a "fall back to CPU/Metal/MLX/WGPU"
//! diagnostic. Op coverage is grown incrementally — each new op is one
//! `.cu` source + one Step variant + one match arm.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};

use crate::arena::{Arena, plan_f32_uniform};
use crate::device::{cuda_blas, cuda_blas_lt_handle, cuda_context, cuda_dnn_handle};
use crate::kernels::{
    argmax_kernel, attention_bwd_kernel, attention_kernel, binary_kernel, compare_kernel,
    concat_kernel, conv_transpose2d_kernel, conv1d_kernel, conv2d_kernel, conv3d_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, dispatch_grid_1d,
    elementwise_region_kernel, expand_kernel, fused_binary_unary_kernel, fused_residual_ln_kernel,
    gather_axis_kernel, gather_backward_kernel, gather_kernel, group_norm_kernel,
    grouped_matmul_kernel, layer_norm2d_kernel, layernorm_kernel, matmul_epilogue_kernel,
    matmul_kernel, matmul_wmma_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel,
    reduce_kernel, resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};

/// Opt-in WMMA Tensor Core matmul. Reads `RLX_CUDA_WMMA=1` from env at
/// process start (cached behind a `OnceLock`). When true and cuBLAS is
/// unavailable, the scalar matmul kernel is replaced by the WMMA kernel
/// for plain (non-fused) matmul. Tensor Cores require SM 70+; on older
/// hardware NVRTC's `load_module` will fail and we fall back to scalar.
fn use_wmma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_WMMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// One launch step in the compiled schedule.
#[derive(Clone)]
enum Step {
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
        act_id: u32,
    },
    Binary {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Compare {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Unary {
        n: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Where {
        n: u32,
        cond_off: u32,
        x_off: u32,
        y_off: u32,
        out_off: u32,
    },
    Reduce {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Softmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    LayerNorm {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        gamma_off: u32,
        beta_off: u32,
        eps_bits: u32,
        op: u32,
    },
    FusedResidualLn {
        outer: u32,
        inner: u32,
        in_off: u32,
        residual_off: u32,
        bias_off: u32,
        gamma_off: u32,
        beta_off: u32,
        out_off: u32,
        eps_bits: u32,
        has_bias: u32,
    },
    Gather {
        n_out: u32,
        n_idx: u32,
        dim: u32,
        vocab: u32,
        in_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    GatherAxis {
        total: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
        table_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    Narrow {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Argmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    Transpose {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Expand {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Concat {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Attention {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        window: u32,
    },
    AttentionBackward {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        dy_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        window: u32,
        wrt: u32,
    },
    Rope {
        n_total: u32,
        seq: u32,
        head_dim: u32,
        half: u32,
        in_off: u32,
        cos_off: u32,
        sin_off: u32,
        out_off: u32,
        last_dim: u32,
    },
    Cumsum {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        exclusive: u32,
    },
    TopK {
        outer: u32,
        inner: u32,
        k: u32,
        in_off: u32,
        out_off: u32,
    },
    GroupedMatmul {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        in_off: u32,
        w_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    ScatterAddZero {
        out_off: u32,
        out_total: u32,
    },
    ScatterAddAcc {
        out_off: u32,
        upd_off: u32,
        idx_off: u32,
        num_updates: u32,
        trailing: u32,
        out_dim: u32,
    },
    DequantMatmul {
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        scheme_id: u32,
        x_off: u32,
        w_off: u32,
        scale_off: u32,
        zp_off: u32,
        out_off: u32,
    },
    /// GGUF K-quant weights — GPU dequant scratch + cuBLAS (host fallback).
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        out_byte_off: u32,
    },
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
    Sample {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        top_k: u32,
        top_p_bits: u32,
        temp_bits: u32,
        seed_lo: u32,
        seed_hi: u32,
    },
    SelectiveScan {
        batch: u32,
        seq: u32,
        hidden: u32,
        state_size: u32,
        x_off: u32,
        delta_off: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        out_off: u32,
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
    /// LLaDA2 / TIDE group-limited MoE gate (host TopK between GPU segments).
    Llada2GroupLimitedGate {
        sig_off: u32,
        route_off: u32,
        out_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
    },
    /// 3D Gaussian splat — host reference between GPU segments.
    GaussianSplatRender {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        dst_off: u32,
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
    GaussianSplatRenderBackward {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        d_loss_off: u32,
        d_loss_len: u32,
        packed_off: u32,
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
    GaussianSplatPrepare {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        meta_len: u32,
        prep_off: u32,
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
    GaussianSplatRasterize {
        prep_off: u32,
        prep_len: u32,
        meta_off: u32,
        meta_len: u32,
        dst_off: u32,
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
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardGamma {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dgamma_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardBeta {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dbeta_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RopeBackward {
        dy_byte_off: u32,
        cos_byte_off: u32,
        sin_byte_off: u32,
        dx_byte_off: u32,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        n_rot: u32,
        cos_len: u32,
    },
    CumsumBackward {
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    GatherBackward {
        dy_byte_off: u32,
        indices_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    Pool1d {
        n: u32,
        c: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool2d {
        n: u32,
        c: u32,
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
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool3d {
        n: u32,
        c: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Conv1d {
        n: u32,
        c_in: u32,
        c_out: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        dl: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    Conv2d {
        n: u32,
        c_in: u32,
        c_out: u32,
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
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    Conv3d {
        n: u32,
        c_in: u32,
        c_out: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        dd: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    /// NCHW LayerNorm2d (SAM semantics).
    LayerNorm2d {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps_bits: u32,
    },
    /// NCHW ConvTranspose2d (PyTorch weight layout).
    ConvTranspose2d {
        src_off: u32,
        w_off: u32,
        dst_off: u32,
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
    /// NCHW group norm.
    GroupNorm {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps_bits: u32,
    },
    /// Nearest-neighbor 2× upsample on NCHW.
    ResizeNearest2x {
        src_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Backend-level fusion of `Binary → Unary` element-wise chains.
    /// Emitted by `fuse_elementwise_chains` when the intermediate
    /// offset has exactly one consumer in the schedule. Avoids one
    /// kernel launch + one round-trip to global memory for the
    /// intermediate result.
    FusedBinaryUnary {
        n: u32,
        a_off: u32,
        b_off: u32,
        out_off: u32,
        bin_op: u32,
        un_op: u32,
    },
    /// PLAN L2 — interpreted N-ary element-wise chain. The chain
    /// encoding (input_offs[8] + chain[64]) lives in `meta_buffers`
    /// and is indexed via `meta_idx`. One thread per output element;
    /// each thread walks the chain in registers and writes the final
    /// result to `arena[dst_off + i]`. Caps: 16 steps, 8 inputs.
    /// Emitted from `Op::ElementwiseRegion` by `MarkElementwiseRegions`
    /// (replaces the prior `UnfuseElementwiseRegions` decomposer
    /// fallback). `input_offs` mirrors what's packed in `meta` and is
    /// kept in the Step so the multi-stream scheduler can resolve
    /// producer-consumer dependencies without unpacking metadata.
    ElementwiseRegion {
        len: u32,
        num_inputs: u32,
        num_steps: u32,
        dst_off: u32,
        input_offs: [u32; 16],
        /// PLAN L2 quality fast path: per-input scalar bitfield.
        /// Bit `i` ⇒ input `i` is a single-element broadcast.
        scalar_input_mask: u32,
        /// PLAN L2 quality general broadcast: per-input element count.
        /// `0` ⇒ no broadcast (kernel reads gid); `>0` ⇒ kernel reads
        /// `arena[input_offs[i] + (gid % input_modulus[i])]`.
        input_modulus: [u32; 16],
        meta_idx: usize,
    },
}

/// When kernels turn into PTX device code.
///
/// `Jit` is the default — each kernel NVRTC-compiles on first dispatch,
/// then the cuModule is cached for the rest of the process. `Aot`
/// pre-compiles every kernel at executable construction so the first
/// `run()` doesn't pay any compile latency. The full AOT pass is ~1-3s
/// (10-100ms × 32 kernels) but moves that cost out of the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    #[default]
    Jit,
    Aot,
}

/// How the schedule executes.
///
/// `Stream` (default) launches each Step on the default stream every
/// `run()`. `Graph` captures the full schedule into a CUDA Graph on
/// first run and replays the captured graph on subsequent runs —
/// eliminates per-launch dispatch overhead (~10-20% on small-batch
/// inference). `Eager` is a one-shot helper that compiles + runs +
/// drops the executable in one call; useful for interactive debugging.
/// `MultiStream(n)` allocates a pool of `n` streams and assigns each
/// `Step` to a stream based on data dependencies — independent ops
/// (e.g. unfused Q/K/V projections, FFN gate/up) run in parallel.
/// Cross-stream synchronization uses CUDA events at producer-consumer
/// boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    Stream,
    Graph,
    Eager,
    MultiStream(usize),
}

pub struct CudaExecutable {
    ctx: Arc<CudaContext>,
    /// cuBLAS handle bound to the same default stream as `ctx`. Used for
    /// plain matmul (no fused bias/activation); falls back to the custom
    /// kernel when cuBLAS isn't available (e.g., on Mac via the panic-
    /// catch probe).
    blas: Option<Arc<Mutex<CudaBlas>>>,
    /// cuBLASLt handle for fused matmul + bias + activation. Falls back
    /// to plain cuBLAS sgemm + epilogue kernel when unavailable.
    blas_lt: Option<cublaslt_sys::cublasLtHandle_t>,
    /// Scratch workspace for cuBLASLt heuristic-selected algorithms.
    /// 4 MiB is the standard recommendation in cuBLAS docs.
    blas_lt_workspace: Option<cudarc::driver::CudaSlice<u8>>,
    /// cuDNN handle for convolution dispatch (conv1d/2d/3d). Falls back
    /// to the custom direct-convolution kernels when unavailable.
    dnn: Option<cudnn_sys::cudnnHandle_t>,
    /// Scratch workspace for cuDNN-selected conv algorithms. Sized at
    /// 32 MiB which covers most modern conv shapes.
    dnn_workspace: Option<cudarc::driver::CudaSlice<u8>>,
    /// Scratch f16 buffer for casting activations on-the-fly when the
    /// matching weight is half-stored. Sized to fit the largest
    /// per-call M·K product seen in matmul dispatch; grown lazily.
    half_act_scratch: Option<cudarc::driver::CudaSlice<u16>>,
    /// Byte offset in the f32 arena for GGUF dequant scratch (max k×n f32).
    dequant_scratch_off: usize,
    graph: Graph,
    arena: Arena,
    schedule: Vec<Step>,
    input_offsets: HashMap<String, NodeId>,
    param_offsets: HashMap<String, NodeId>,
    /// Per-step side buffers for kernels that need per-axis u32 metadata
    /// (Transpose, Expand). Indexed via `Step::Transpose.meta_idx` etc.
    meta_buffers: Vec<cudarc::driver::CudaSlice<u32>>,
    exec_mode: ExecMode,
    /// Captured CUDA Graph (built on first `run()` when `exec_mode ==
    /// Graph`). Replayed on subsequent runs to skip per-launch dispatch.
    captured_graph: Option<cudarc::driver::CudaGraph>,
    /// Stream pool for `ExecMode::MultiStream(n)`. Empty for the other
    /// modes (which use the context's default stream).
    streams: Vec<Arc<cudarc::driver::CudaStream>>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every step in `schedule` is in the
    /// safe set, `run` bypasses the captured CUDA Graph (recorded at
    /// full extent) and dispatches per-step with scaled launch dims.
    /// Otherwise full-extent fallback. See PLAN L1.
    pub(crate) active_extent: Option<(usize, usize)>,
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Initial coverage: simple element-wise ops + reductions + softmax +
    /// LayerNorm + cumsum. Matmul, Attention, Conv, Pool, GroupedMatmul,
    /// DequantMatmul, Sample, SelectiveScan, Rope, ScatterAdd, Transpose,
    /// Expand, Concat, Narrow, Gather, GatherAxis, Argmax, TopK still
    /// default to unsafe — opt in once each Step's per-tier dispatch +
    /// kernel offset arithmetic has been verified to scale safely.
    pub fn safe_for_active_extent(&self) -> bool {
        matches!(
            self,
            Step::Binary { .. }
                | Step::Compare { .. }
                | Step::Unary { .. }
                | Step::Where { .. }
                | Step::Reduce { .. }
                | Step::Softmax { .. }
                | Step::LayerNorm { .. }
                | Step::FusedResidualLn { .. }
                | Step::Cumsum { .. }
                | Step::FusedBinaryUnary { .. }
                | Step::ElementwiseRegion { .. }
        )
    }
}

const CUBLASLT_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
const CUDNN_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// Map our internal activation id (matches the `unary` kernel table)
/// to a cuBLASLt epilogue activation, if it's natively fusable.
/// cuBLASLt only supports Relu and Gelu in the epilogue — anything else
/// (sigmoid, tanh, silu, abs, neg, sqrt) returns None and the caller
/// falls back to plain sgemm + the matmul_epilogue kernel.
fn cublaslt_act_for(act_id: u32) -> Option<cublaslt_sys::cublasLtEpilogue_t> {
    None.or(match act_id {
        // Identity
        0xFFFFu32 => Some(None),
        // Relu = 0; Gelu = 9; GeluApprox = 11 (treat as Gelu).
        0 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        )),
        9 | 11 => Some(Some(
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        )),
        _ => Some(None),
    })
    .flatten()
}

/// True when `act_id` is fusable in cuBLASLt's epilogue (or absent).
fn cublaslt_act_supported(act_id: u32) -> bool {
    matches!(act_id, 0xFFFFu32 | 0 | 9 | 11)
}

/// Single cuBLASLt fused matmul. Consumes one descriptor + three matrix
/// layouts + one preference object per call (descriptors are cheap to
/// create; future optimization could cache them by shape). Returns
/// `Err` on any setup failure so the caller can fall back to plain
/// cuBLAS sgemm + epilogue kernel.
unsafe fn cublaslt_matmul_fused(
    handle: cublaslt_sys::cublasLtHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    m: u32,
    k: u32,
    n: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
    has_bias: bool,
    bias_off_f32: u32,
    epilogue_act: Option<cublaslt_sys::cublasLtEpilogue_t>,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    cu_stream: cudarc::driver::sys::CUstream,
) -> Result<(), cublaslt_result::CublasError> {
    use core::ffi::c_void;
    use core::mem;

    // cuBLASLt is column-major. We swap A↔B so that "computing C^T =
    // B^T·A^T in column-major" matches "C = A·B in row-major".
    let a_ptr = (arena_dev_ptr + (b_off_f32 as u64) * 4) as *const c_void; // = our B
    let b_ptr = (arena_dev_ptr + (a_off_f32 as u64) * 4) as *const c_void; // = our A
    let c_ptr = (arena_dev_ptr + (c_off_f32 as u64) * 4) as *const c_void;
    let d_ptr = c_ptr as *mut c_void;

    let dt = cublaslt_sys::cudaDataType_t::CUDA_R_32F;

    // Layouts. After A↔B swap: cuBLASLt sees a [n,k] · [k,m] = [n,m].
    let a_layout = cublaslt_result::create_matrix_layout(dt, n as u64, k as u64, n as i64)?;
    let b_layout = cublaslt_result::create_matrix_layout(dt, k as u64, m as u64, k as i64)?;
    let c_layout = cublaslt_result::create_matrix_layout(dt, n as u64, m as u64, n as i64)?;

    if batch > 1 {
        unsafe {
            let bsz = batch as i32;
            for &layout in &[a_layout, b_layout, c_layout] {
                cublaslt_result::set_matrix_layout_attribute(
                layout,
                cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &bsz as *const _ as *const _,
                mem::size_of::<i32>(),
            )?;
            }
            let stride_b = b_batch_stride as i64;
            let stride_a = a_batch_stride as i64;
            let stride_c = c_batch_stride as i64;
            cublaslt_result::set_matrix_layout_attribute(
            a_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_b as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            b_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_a as *const _ as *const _, mem::size_of::<i64>())?;
            cublaslt_result::set_matrix_layout_attribute(
            c_layout,
            cublaslt_sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &stride_c as *const _ as *const _, mem::size_of::<i64>())?;
        }
    }

    // CUBLAS_COMPUTE_32F_FAST_TF32 enables Tensor-Core paths for f32
    // inputs on Ampere+ (10-bit mantissa intermediate vs 23-bit). ~2×
    // speedup; precision delta is well within transformer-inference
    // tolerance and matches what cublasSgemm already does by default.
    let matmul_desc = cublaslt_result::create_matmul_desc(
        cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        dt,
    )?;

    // Pick the epilogue mode. cuBLASLt fuses bias broadcast over the
    // M dimension (in cuBLASLt's view). With our A↔B swap, cuBLASLt's
    // M = our row-major N, so a bias[N] vector broadcasts across M
    // rows of row-major C — exactly what we want.
    let epilogue = match (has_bias, epilogue_act) {
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS
        }
        (true, Some(cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU)) => {
            cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        }
        (true, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        (false, Some(act)) => act,
        (false, None) => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        _ => cublaslt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
    };
    unsafe {
        cublaslt_result::set_matmul_desc_attribute(
            matmul_desc,
            cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue as *const _ as *const _,
            mem::size_of::<cublaslt_sys::cublasLtEpilogue_t>(),
        )?;
    }

    if has_bias {
        let bias_dev_ptr = arena_dev_ptr + (bias_off_f32 as u64) * 4;
        unsafe {
            cublaslt_result::set_matmul_desc_attribute(
                matmul_desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_dev_ptr as *const _ as *const _,
                mem::size_of::<u64>(),
            )?;
        }
    }

    let matmul_pref = cublaslt_result::create_matmul_pref()?;
    unsafe {
        cublaslt_result::set_matmul_pref_attribute(
            matmul_pref,
            cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size as *const _ as *const _,
            mem::size_of::<usize>(),
        )?;
    }

    let heuristic = unsafe {
        cublaslt_result::get_matmul_algo_heuristic(
            handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            matmul_pref,
        )
    }?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let workspace_ptr = workspace_dev_ptr as *mut c_void;

    let result = unsafe {
        cublaslt_result::matmul(
            handle,
            matmul_desc,
            &alpha as *const _ as *const c_void,
            &beta as *const _ as *const c_void,
            a_ptr,
            a_layout,
            b_ptr,
            b_layout,
            c_ptr,
            c_layout,
            d_ptr,
            c_layout,
            &heuristic.algo as *const _,
            workspace_ptr,
            workspace_size,
            cu_stream as cublaslt_sys::cudaStream_t,
        )
    };

    // Always destroy descriptors (success or fail).
    unsafe {
        let _ = cublaslt_result::destroy_matmul_pref(matmul_pref);
        let _ = cublaslt_result::destroy_matmul_desc(matmul_desc);
        let _ = cublaslt_result::destroy_matrix_layout(c_layout);
        let _ = cublaslt_result::destroy_matrix_layout(b_layout);
        let _ = cublaslt_result::destroy_matrix_layout(a_layout);
    }

    result
}

/// cuDNN forward 2D convolution against arena offsets. NCHW input,
/// KCRS filter, NCHW output. Uses the v7 algorithm heuristic to pick
/// the fastest algo that fits in the supplied workspace. Returns
/// `Err` on any setup failure so the caller can fall back to the
/// direct-convolution kernel.
unsafe fn cudnn_conv2d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
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
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;

    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    let setup = unsafe {
        cudnn_result::set_tensor4d_descriptor(
            x_desc,
            fmt,
            dt,
            [n as i32, c_in as i32, h as i32, w as i32],
        )?;
        cudnn_result::set_tensor4d_descriptor(
            y_desc,
            fmt,
            dt,
            [n as i32, c_out as i32, h_out as i32, w_out as i32],
        )?;
        cudnn_result::set_filter4d_descriptor(
            w_desc,
            dt,
            fmt,
            [
                c_out as i32,
                (c_in / groups.max(1)) as i32,
                kh as i32,
                kw as i32,
            ],
        )?;
        cudnn_result::set_convolution2d_descriptor(
            conv_desc,
            ph as i32,
            pw as i32,
            sh as i32,
            sw as i32,
            dh as i32,
            dw as i32,
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        // Pick the fastest fwd algo via the v7 heuristic.
        let mut returned_count: i32 = 0;
        let mut perf = std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;

        let needed = cudnn_result::get_convolution_forward_workspace_size(
            handle, x_desc, w_desc, conv_desc, y_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// cuDNN forward 3-D convolution. NCDHW input, KCDRS filter, NCDHW
/// output. Uses cuDNN's nd-descriptor APIs (set_tensornd / set_filternd
/// / set_convolutionnd) since the 4D versions only cover up to 2D conv.
unsafe fn cudnn_conv3d_forward(
    handle: cudnn_sys::cudnnHandle_t,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), cudnn_result::CudnnError> {
    use core::ffi::c_void;

    let dt = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
    let fmt = cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    let x_desc = cudnn_result::create_tensor_descriptor()?;
    let y_desc = cudnn_result::create_tensor_descriptor()?;
    let conv_desc = cudnn_result::create_convolution_descriptor()?;
    let w_desc = unsafe {
        let mut w_desc_uninit = std::mem::MaybeUninit::uninit();
        cudnn_sys::cudnnCreateFilterDescriptor(w_desc_uninit.as_mut_ptr()).result()?;
        w_desc_uninit.assume_init()
    };

    // 5-D tensor: [N, C, D, H, W] with row-major contiguous strides.
    let x_dims: [i32; 5] = [n as i32, c_in as i32, d as i32, h as i32, w as i32];
    let x_strides: [i32; 5] = [
        (c_in * d * h * w) as i32,
        (d * h * w) as i32,
        (h * w) as i32,
        w as i32,
        1,
    ];
    let y_dims: [i32; 5] = [
        n as i32,
        c_out as i32,
        d_out as i32,
        h_out as i32,
        w_out as i32,
    ];
    let y_strides: [i32; 5] = [
        (c_out * d_out * h_out * w_out) as i32,
        (d_out * h_out * w_out) as i32,
        (h_out * w_out) as i32,
        w_out as i32,
        1,
    ];
    let f_dims: [i32; 5] = [
        c_out as i32,
        (c_in / groups.max(1)) as i32,
        kd as i32,
        kh as i32,
        kw as i32,
    ];
    let pads: [i32; 3] = [pd as i32, ph as i32, pw as i32];
    let strides: [i32; 3] = [sd as i32, sh as i32, sw as i32];
    let dilations: [i32; 3] = [dd as i32, dh as i32, dw as i32];

    let setup = unsafe {
        cudnn_result::set_tensornd_descriptor(x_desc, dt, 5, x_dims.as_ptr(), x_strides.as_ptr())?;
        cudnn_result::set_tensornd_descriptor(y_desc, dt, 5, y_dims.as_ptr(), y_strides.as_ptr())?;
        cudnn_result::set_filternd_descriptor(w_desc, dt, fmt, 5, f_dims.as_ptr())?;
        cudnn_result::set_convolutionnd_descriptor(
            conv_desc,
            3,
            pads.as_ptr(),
            strides.as_ptr(),
            dilations.as_ptr(),
            cudnn_sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            dt,
        )?;
        if groups > 1 {
            cudnn_sys::cudnnSetConvolutionGroupCount(conv_desc, groups as i32).result()?;
        }
        Ok::<(), cudnn_result::CudnnError>(())
    };

    let result = setup.and_then(|()| unsafe {
        let mut returned_count: i32 = 0;
        let mut perf = std::mem::MaybeUninit::<cudnn_sys::cudnnConvolutionFwdAlgoPerf_t>::uninit();
        cudnn_result::get_convolution_forward_algorithm(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            &mut returned_count,
            perf.as_mut_ptr(),
        )?;
        if returned_count == 0 {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }
        let algo = perf.assume_init().algo;

        let needed = cudnn_result::get_convolution_forward_workspace_size(
            handle, x_desc, w_desc, conv_desc, y_desc, algo,
        )?;
        if needed > workspace_size {
            return Err(cudnn_result::CudnnError(
                cudnn_sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ));
        }

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
        let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
        let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
        let workspace_ptr = workspace_dev_ptr as *mut c_void;

        cudnn_result::convolution_forward(
            handle,
            &alpha as *const _ as *const c_void,
            x_desc,
            x_ptr,
            w_desc,
            w_ptr,
            conv_desc,
            algo,
            workspace_ptr,
            workspace_size,
            &beta as *const _ as *const c_void,
            y_desc,
            y_ptr,
        )
    });

    unsafe {
        let _ = cudnn_result::destroy_convolution_descriptor(conv_desc);
        let _ = cudnn_result::destroy_filter_descriptor(w_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(y_desc);
        let _ = cudnn_result::destroy_tensor_descriptor(x_desc);
    }

    result
}

/// Decode a Matmul/FusedMatMulBiasAct node's input shapes into the
/// (m, k, n, batch, a_stride, b_stride, c_stride, a_id, b_id) tuple
/// the kernel expects. Three patterns:
///   • 2D × 2D                       → batch=1, all strides 0
///   • [..,M,K] × [K,N] (broadcast)  → batch=1, leading dims flattened into M
///   • [..,M,K] × [..,K,N] (matched) → batch=prod(leading), per-batch strides
fn matmul_shape(
    graph: &Graph,
    node: &rlx_ir::Node,
    op_label: &str,
) -> (u32, u32, u32, u32, u32, u32, u32, NodeId, NodeId) {
    let a_id = node.inputs[0];
    let b_id = node.inputs[1];
    let a_shape = graph.node(a_id).shape.dims();
    let b_shape = graph.node(b_id).shape.dims();
    let out_shape = node.shape.dims();
    if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2 {
        let m = a_shape[0].unwrap_static() as u32;
        let k = a_shape[1].unwrap_static() as u32;
        let n = b_shape[1].unwrap_static() as u32;
        (m, k, n, 1, 0, 0, 0, a_id, b_id)
    } else if a_shape.len() >= 2 && b_shape.len() == 2 && out_shape.len() == a_shape.len() {
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
            1,
            0,
            0,
            0,
            a_id,
            b_id,
        )
    } else if a_shape.len() == b_shape.len() && a_shape.len() >= 3 {
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
                "rlx-cuda {op_label}: batched shape mismatch \
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
            a_id,
            b_id,
        )
    } else {
        panic!(
            "rlx-cuda {op_label}: unsupported shapes a={a_shape:?} b={b_shape:?} out={out_shape:?}"
        );
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

/// Mixed-precision matmul tier-0: when the weight (B input) is stored
/// in the half-arena, cast f32 activations to f16/bf16 in the scratch
/// buffer and run `cublasGemmEx` with both inputs half + f32
/// accumulator. Returns `true` on success.
///
/// Free function (rather than `&mut self` method) so the caller can
/// hold `&self.schedule` across the call without violating disjoint-
/// field borrow checks.
#[allow(clippy::too_many_arguments)]
fn try_mixed_precision_gemm(
    ctx: &Arc<CudaContext>,
    arena: &mut crate::arena::Arena,
    half_act_scratch: &mut Option<cudarc::driver::CudaSlice<u16>>,
    blas: Option<&Arc<Mutex<CudaBlas>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    a_off_f32: u32,
    b_off_f32: u32,
    c_off_f32: u32,
) -> bool {
    let (half_off, half_dtype) = match arena.half_by_f32_off.get(&b_off_f32).copied() {
        Some(v) => v,
        None => return false,
    };
    let blas = match blas {
        Some(b) => b,
        None => return false,
    };

    let act_elems = (m * k * batch.max(1)) as usize;
    let need_resize = half_act_scratch
        .as_ref()
        .is_none_or(|s| s.len() < act_elems);
    if need_resize {
        *half_act_scratch = stream.alloc_zeros::<u16>(act_elems.max(4)).ok();
    }
    if half_act_scratch.is_none() {
        return false;
    }

    // Phase 1: cast activations f32 → f16/bf16 into the scratch.
    let n_total = m * k * batch.max(1);
    let dtype_id: u32 = match half_dtype {
        crate::arena::HalfDtype::F16 => 0,
        crate::arena::HalfDtype::Bf16 => 1,
    };
    {
        let kernel = crate::kernels::cast_f32_to_half_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(n_total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let src_view = arena
            .buffer
            .slice(a_off_f32 as usize..a_off_f32 as usize + n_total as usize);
        let scratch_mut = half_act_scratch.as_mut().unwrap();
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&src_view)
            .arg(scratch_mut)
            .arg(&n_total)
            .arg(&dtype_id);
        if unsafe { launcher.launch(cfg) }.is_err() {
            return false;
        }
    }

    // Phase 2: cublasGemmEx with both inputs half + f32 output.
    let blas = blas.lock().unwrap();
    let (arena_ptr_u64, _ar) = arena.buffer.device_ptr_mut(stream);
    let (half_buf_ptr, _hb) = arena.half_buffer.as_mut().unwrap().device_ptr_mut(stream);
    let scratch_ptr_u64 = {
        let s = half_act_scratch.as_mut().unwrap();
        let (p, _r) = s.device_ptr_mut(stream);
        p
    };
    let weight_dev = half_buf_ptr + (half_off as u64) * 2; // u16 = 2 bytes
    let act_dev = scratch_ptr_u64;
    let c_dev = arena_ptr_u64 + (c_off_f32 as u64) * 4;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let cuda_dt = match half_dtype {
        crate::arena::HalfDtype::F16 => cublas_sys::cudaDataType_t::CUDA_R_16F,
        crate::arena::HalfDtype::Bf16 => cublas_sys::cudaDataType_t::CUDA_R_16BF,
    };
    let compute_ty = match half_dtype {
        crate::arena::HalfDtype::F16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F
        }
        crate::arena::HalfDtype::Bf16 => {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
        }
    };
    let result = unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32 as *const _,
            weight_dev as *const _,
            cuda_dt,
            n as i32,
            act_dev as *const _,
            cuda_dt,
            k as i32,
            &beta as *const f32 as *const _,
            c_dev as *mut _,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            compute_ty,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    if let Err(ref e) = result {
        log_fallback("matmul.gemmEx (mixed-precision)", e);
    }
    result.is_ok()
}

/// One-time-per-tier log when a fast-path dispatch silently falls
/// back. Helps cloud-GPU debugging see *why* the slow path took over —
/// otherwise the only signal is unexpectedly low throughput.
/// Gated behind `RLX_CUDA_LOG_FALLBACK=1` so production isn't spammed.
fn log_fallback(tier: &str, err: impl std::fmt::Debug) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        rlx_ir::env::var("RLX_CUDA_LOG_FALLBACK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    if enabled {
        eprintln!("rlx-cuda: tier '{tier}' fell back: {err:?}");
    }
}

/// Stable, profiler-friendly name for an NVTX range covering a Step
/// dispatch. Matches the variant name; nsight-systems / nvprof show
/// these as range boundaries in the timeline.
fn step_name(step: &Step) -> &'static str {
    match step {
        Step::Matmul { .. } => "rlx::Matmul",
        Step::Binary { .. } => "rlx::Binary",
        Step::Compare { .. } => "rlx::Compare",
        Step::Unary { .. } => "rlx::Unary",
        Step::Where { .. } => "rlx::Where",
        Step::Reduce { .. } => "rlx::Reduce",
        Step::Softmax { .. } => "rlx::Softmax",
        Step::LayerNorm { .. } => "rlx::LayerNorm",
        Step::FusedResidualLn { .. } => "rlx::FusedResidualLN",
        Step::Gather { .. } => "rlx::Gather",
        Step::GatherAxis { .. } => "rlx::GatherAxis",
        Step::Narrow { .. } => "rlx::Narrow",
        Step::Concat { .. } => "rlx::Concat",
        Step::Transpose { .. } => "rlx::Transpose",
        Step::Expand { .. } => "rlx::Expand",
        Step::Argmax { .. } => "rlx::Argmax",
        Step::Attention { .. } => "rlx::Attention",
        Step::AttentionBackward { .. } => "rlx::AttentionBackward",
        Step::Rope { .. } => "rlx::Rope",
        Step::Cumsum { .. } => "rlx::Cumsum",
        Step::TopK { .. } => "rlx::TopK",
        Step::GroupedMatmul { .. } => "rlx::GroupedMatmul",
        Step::ScatterAddZero { .. } => "rlx::ScatterAdd::zero",
        Step::ScatterAddAcc { .. } => "rlx::ScatterAdd::acc",
        Step::DequantMatmul { .. } => "rlx::DequantMatmul",
        Step::DequantMatmulGguf { .. } => "rlx::DequantMatmulGguf",
        Step::DequantGroupedMatmulGguf { .. } => "rlx::DequantGroupedMatmulGguf",
        Step::Sample { .. } => "rlx::Sample",
        Step::SelectiveScan { .. } => "rlx::SelectiveScan",
        Step::GatedDeltaNet { .. } => "rlx::GatedDeltaNet",
        Step::Llada2GroupLimitedGate { .. } => "rlx::Llada2GroupLimitedGate",
        Step::GaussianSplatRender { .. } => "rlx::GaussianSplatRender",
        Step::GaussianSplatRenderBackward { .. } => "rlx::GaussianSplatRenderBackward",
        Step::GaussianSplatPrepare { .. } => "rlx::GaussianSplatPrepare",
        Step::GaussianSplatRasterize { .. } => "rlx::GaussianSplatRasterize",
        Step::RmsNormBackwardInput { .. } => "rlx::RmsNormBackwardInput",
        Step::RmsNormBackwardGamma { .. } => "rlx::RmsNormBackwardGamma",
        Step::RmsNormBackwardBeta { .. } => "rlx::RmsNormBackwardBeta",
        Step::RopeBackward { .. } => "rlx::RopeBackward",
        Step::CumsumBackward { .. } => "rlx::CumsumBackward",
        Step::GatherBackward { .. } => "rlx::GatherBackward",
        Step::Pool1d { .. } => "rlx::Pool1d",
        Step::Pool2d { .. } => "rlx::Pool2d",
        Step::Pool3d { .. } => "rlx::Pool3d",
        Step::Conv1d { .. } => "rlx::Conv1d",
        Step::Conv2d { .. } => "rlx::Conv2d",
        Step::Conv3d { .. } => "rlx::Conv3d",
        Step::LayerNorm2d { .. } => "rlx::LayerNorm2d",
        Step::ConvTranspose2d { .. } => "rlx::ConvTranspose2d",
        Step::GroupNorm { .. } => "rlx::GroupNorm",
        Step::ResizeNearest2x { .. } => "rlx::ResizeNearest2x",
        Step::FusedBinaryUnary { .. } => "rlx::FusedBinaryUnary",
        Step::ElementwiseRegion { .. } => "rlx::ElementwiseRegion",
    }
}

/// Walk a freshly-built schedule and merge `Binary → Unary` element-wise
/// chains into `FusedBinaryUnary`. Conditions for fusion:
///   1. The pair has matching element count `n`.
///   2. The Unary's input offset == the Binary's output offset.
///   3. The intermediate offset has exactly one consumer in the
///      schedule (= no other Step reads it). This guarantees we can
///      drop the round-trip to global memory for the intermediate
///      without breaking any other Step's input.
fn fuse_elementwise_chains(schedule: Vec<Step>) -> Vec<Step> {
    // Tally consumer counts per offset: how many Steps in the schedule
    // read each offset.
    let mut consumer_counts: HashMap<u32, usize> = HashMap::new();
    for step in &schedule {
        let (reads, _) = step_offsets(step);
        for r in &reads {
            *consumer_counts.entry(*r).or_insert(0) += 1;
        }
    }

    let mut out = Vec::with_capacity(schedule.len());
    let mut i = 0;
    while i < schedule.len() {
        if i + 1 < schedule.len() {
            let pair = (&schedule[i], &schedule[i + 1]);
            if let (
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op: bin_op,
                },
                Step::Unary {
                    n: n2,
                    in_off,
                    out_off,
                    op: un_op,
                },
            ) = pair
            {
                let single_consumer = consumer_counts.get(c_off).copied() == Some(1);
                if n == n2 && c_off == in_off && single_consumer {
                    out.push(Step::FusedBinaryUnary {
                        n: *n,
                        a_off: *a_off,
                        b_off: *b_off,
                        out_off: *out_off,
                        bin_op: *bin_op,
                        un_op: *un_op,
                    });
                    i += 2;
                    continue;
                }
            }
        }
        out.push(schedule[i].clone());
        i += 1;
    }
    out
}

/// (read offsets, write offsets) for a Step. Used by the multi-stream
/// scheduler to decide which streams each step depends on. Offsets are
/// the leading f32-element offset of each input/output tensor — a
/// coarse approximation that's correct for our planner since each
/// node has its own slot (Reshape/Cast aliasing maps consumers to the
/// same slot, which is exactly what the dependency tracker wants).
fn step_offsets(step: &Step) -> (Vec<u32>, Vec<u32>) {
    match step {
        Step::Matmul {
            a_off_f32,
            b_off_f32,
            c_off_f32,
            has_bias,
            bias_off_f32,
            ..
        } => {
            let mut r = vec![*a_off_f32, *b_off_f32];
            if *has_bias != 0 {
                r.push(*bias_off_f32);
            }
            (r, vec![*c_off_f32])
        }
        Step::Binary {
            a_off,
            b_off,
            c_off,
            ..
        }
        | Step::Compare {
            a_off,
            b_off,
            c_off,
            ..
        } => (vec![*a_off, *b_off], vec![*c_off]),
        Step::Unary {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Where {
            cond_off,
            x_off,
            y_off,
            out_off,
            ..
        } => (vec![*cond_off, *x_off, *y_off], vec![*out_off]),
        Step::Reduce {
            in_off, out_off, ..
        }
        | Step::Softmax {
            in_off, out_off, ..
        }
        | Step::Argmax {
            in_off, out_off, ..
        }
        | Step::Cumsum {
            in_off, out_off, ..
        }
        | Step::Sample {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::TopK {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::LayerNorm {
            in_off,
            gamma_off,
            beta_off,
            out_off,
            ..
        } => (vec![*in_off, *gamma_off, *beta_off], vec![*out_off]),
        Step::FusedResidualLn {
            in_off,
            residual_off,
            bias_off,
            gamma_off,
            beta_off,
            out_off,
            has_bias,
            ..
        } => {
            let mut r = vec![*in_off, *residual_off, *gamma_off, *beta_off];
            if *has_bias != 0 {
                r.push(*bias_off);
            }
            (r, vec![*out_off])
        }
        Step::Gather {
            in_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *idx_off], vec![*out_off]),
        Step::GatherAxis {
            table_off,
            idx_off,
            out_off,
            ..
        } => (vec![*table_off, *idx_off], vec![*out_off]),
        Step::Narrow {
            in_off, out_off, ..
        }
        | Step::Concat {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Transpose {
            in_off, out_off, ..
        }
        | Step::Expand {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Attention {
            q_off,
            k_off,
            v_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::AttentionBackward {
            q_off,
            k_off,
            v_off,
            dy_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off, *dy_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::Rope {
            in_off,
            cos_off,
            sin_off,
            out_off,
            ..
        } => (vec![*in_off, *cos_off, *sin_off], vec![*out_off]),
        Step::GroupedMatmul {
            in_off,
            w_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off, *idx_off], vec![*out_off]),
        Step::ScatterAddZero { out_off, .. } => (vec![], vec![*out_off]),
        Step::ScatterAddAcc {
            upd_off,
            idx_off,
            out_off,
            ..
        } =>
        // out_off is read-modify-write — list it as both a read and
        // a write so the scheduler waits on the prior zero.
        {
            (vec![*upd_off, *idx_off, *out_off], vec![*out_off])
        }
        Step::DequantMatmul {
            x_off,
            w_off,
            scale_off,
            zp_off,
            out_off,
            scheme_id,
            ..
        } => {
            let mut r = vec![*x_off, *w_off, *scale_off];
            if *scheme_id == 1 {
                r.push(*zp_off);
            }
            (r, vec![*out_off])
        }
        Step::DequantMatmulGguf {
            x_byte_off,
            w_byte_off,
            out_byte_off,
            ..
        } => (vec![x_byte_off / 4, w_byte_off / 4], vec![out_byte_off / 4]),
        Step::DequantGroupedMatmulGguf {
            x_byte_off,
            w_byte_off,
            idx_byte_off,
            out_byte_off,
            ..
        } => (
            vec![x_byte_off / 4, w_byte_off / 4, idx_byte_off / 4],
            vec![out_byte_off / 4],
        ),
        Step::SelectiveScan {
            x_off,
            delta_off,
            a_off,
            b_off,
            c_off,
            out_off,
            ..
        } => (
            vec![*x_off, *delta_off, *a_off, *b_off, *c_off],
            vec![*out_off],
        ),
        Step::GatedDeltaNet {
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            state_byte_off,
            dst_byte_off,
            use_carry,
            ..
        } => {
            let mut reads = vec![
                q_byte_off / 4,
                k_byte_off / 4,
                v_byte_off / 4,
                g_byte_off / 4,
                beta_byte_off / 4,
            ];
            if *use_carry {
                reads.push(state_byte_off / 4);
            }
            let mut writes = vec![dst_byte_off / 4];
            if *use_carry {
                writes.push(state_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Llada2GroupLimitedGate {
            sig_off,
            route_off,
            out_off,
            ..
        } => (vec![*sig_off, *route_off], vec![*out_off]),
        Step::GaussianSplatRender {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            dst_off,
            dst_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![dst_off / 4],
        ),
        Step::GaussianSplatRenderBackward {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            d_loss_off,
            d_loss_len: _,
            packed_off,
            packed_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
                d_loss_off / 4,
            ],
            vec![packed_off / 4],
        ),
        Step::RmsNormBackwardInput {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dx_byte_off / 4],
        ),
        Step::RmsNormBackwardGamma {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dgamma_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dgamma_byte_off / 4],
        ),
        Step::RmsNormBackwardBeta {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dbeta_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dbeta_byte_off / 4],
        ),
        Step::RopeBackward {
            dy_byte_off,
            cos_byte_off,
            sin_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, cos_byte_off / 4, sin_byte_off / 4],
            vec![dx_byte_off / 4],
        ),
        Step::CumsumBackward {
            dy_byte_off,
            dx_byte_off,
            ..
        } => (vec![dy_byte_off / 4], vec![dx_byte_off / 4]),
        Step::GatherBackward {
            dy_byte_off,
            indices_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, indices_byte_off / 4],
            vec![dst_byte_off / 4],
        ),
        Step::Pool1d {
            in_off, out_off, ..
        }
        | Step::Pool2d {
            in_off, out_off, ..
        }
        | Step::Pool3d {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Conv1d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv2d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv3d {
            in_off,
            w_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off], vec![*out_off]),
        Step::LayerNorm2d {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::ConvTranspose2d {
            src_off,
            w_off,
            dst_off,
            ..
        } => (vec![*src_off, *w_off], vec![*dst_off]),
        Step::GroupNorm {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::ResizeNearest2x {
            src_off, dst_off, ..
        } => (vec![*src_off], vec![*dst_off]),
        Step::FusedBinaryUnary {
            a_off,
            b_off,
            out_off,
            ..
        } => (vec![*a_off, *b_off], vec![*out_off]),
        Step::ElementwiseRegion {
            dst_off,
            input_offs,
            num_inputs,
            ..
        } => {
            let n = (*num_inputs as usize).min(input_offs.len());
            (input_offs[..n].to_vec(), vec![*dst_off])
        }
        Step::GaussianSplatPrepare {
            positions_off,
            scales_off,
            rotations_off,
            opacities_off,
            colors_off,
            sh_coeffs_off,
            meta_off,
            prep_off,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![prep_off / 4],
        ),
        Step::GaussianSplatRasterize {
            prep_off,
            meta_off,
            dst_off,
            ..
        } => (vec![prep_off / 4, meta_off / 4], vec![dst_off / 4]),
    }
}

/// Pre-compile every NVRTC kernel against `ctx`. Used by AOT mode to
/// move JIT compile cost out of the first-run critical path.
fn prewarm_all(ctx: &Arc<CudaContext>) {
    use crate::kernels::*;
    let _ = binary_kernel(ctx);
    let _ = fused_binary_unary_kernel(ctx);
    let _ = unary_kernel(ctx);
    let _ = copy_kernel(ctx);
    let _ = matmul_kernel(ctx);
    let _ = matmul_epilogue_kernel(ctx);
    let _ = compare_kernel(ctx);
    let _ = where_kernel(ctx);
    let _ = reduce_kernel(ctx);
    let _ = softmax_kernel(ctx);
    let _ = layernorm_kernel(ctx);
    let _ = fused_residual_ln_kernel(ctx);
    let _ = gather_kernel(ctx);
    let _ = gather_axis_kernel(ctx);
    let _ = narrow_kernel(ctx);
    let _ = concat_kernel(ctx);
    let _ = transpose_kernel(ctx);
    let _ = expand_kernel(ctx);
    let _ = attention_kernel(ctx);
    let _ = attention_bwd_kernel(ctx);
    let _ = argmax_kernel(ctx);
    let _ = rope_kernel(ctx);
    let _ = cumsum_kernel(ctx);
    let _ = topk_kernel(ctx);
    let _ = grouped_matmul_kernel(ctx);
    let _ = scatter_add_zero_kernel(ctx);
    let _ = scatter_add_acc_kernel(ctx);
    let _ = dequant_matmul_kernel(ctx);
    let _ = dequant_gguf_kernel(ctx);
    let _ = sample_kernel(ctx);
    let _ = selective_scan_kernel(ctx);
    let _ = pool1d_kernel(ctx);
    let _ = pool2d_kernel(ctx);
    let _ = pool3d_kernel(ctx);
    let _ = conv1d_kernel(ctx);
    let _ = conv2d_kernel(ctx);
    let _ = conv3d_kernel(ctx);
    let _ = layer_norm2d_kernel(ctx);
    let _ = conv_transpose2d_kernel(ctx);
    let _ = group_norm_kernel(ctx);
    let _ = resize_nearest_2x_kernel(ctx);
    let _ = elementwise_region_kernel(ctx);
    // matmul_wmma deliberately excluded: requires SM 70+ and may fail
    // load_module on older GPUs. Compile lazily on first opt-in dispatch.
}

impl CudaExecutable {
    /// JIT compile, stream-mode execution. Default entry point.
    pub fn compile(graph: Graph) -> Self {
        Self::compile_with(graph, CompileMode::Jit, ExecMode::Stream)
    }

    /// One-shot eager run. Compiles, executes once with the given
    /// inputs, and drops the executable. No persistent state.
    pub fn eager(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut exec = Self::compile_with(graph, CompileMode::Jit, ExecMode::Eager);
        exec.run(inputs)
    }

    /// Full constructor with explicit compile + exec modes.
    pub fn compile_with(graph: Graph, compile_mode: CompileMode, exec_mode: ExecMode) -> Self {
        let ctx = cuda_context().expect("rlx-cuda: no CUDA driver available");

        if compile_mode == CompileMode::Aot {
            prewarm_all(&ctx);
        }

        // Decompose composed ops we don't yet have native kernels for
        // (FusedMatMulBiasAct, canonical DotGeneral) into primitives
        // before memory planning.
        let graph = crate::unfuse::unfuse(graph);

        let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
        let mut plan = plan_f32_uniform(&graph, 16);
        let dequant_scratch_off = if dequant_scratch > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            plan.arena_size = aligned + dequant_scratch;
            aligned
        } else {
            0
        };
        let mut arena = Arena::from_plan(&ctx, &plan);
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0);
            arena.set_actual_len(node.id, elems * 4);
        }

        // Initial param/input offset maps for fast lookup at run time.
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

        // Initialise Constants directly into the arena.
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let bytes_to_write = data.len().min(arena.len_of(node.id));
                let n_f32 = bytes_to_write / 4;
                let f32_view: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_f32) };
                let off_f32 = arena.offset(node.id) / 4;
                let stream = ctx.default_stream();
                let mut slot = arena.buffer.slice_mut(off_f32..off_f32 + n_f32);
                stream
                    .memcpy_htod(f32_view, &mut slot)
                    .expect("rlx-cuda: constant upload failed");
            }
        }

        let mut schedule = Vec::new();
        let mut meta_buffers: Vec<cudarc::driver::CudaSlice<u32>> = Vec::new();
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::Reshape { .. } | Op::Cast { .. } => {
                    // No-op: arena.plan_f32_uniform already aliased the
                    // output slot to the input. The same row-major bytes
                    // are visible under the new node ID.
                }
                Op::MatMul => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "MatMul");
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
                    });
                }
                Op::FusedMatMulBiasAct { activation } => {
                    let (m, k, n, batch, a_bs, b_bs, c_bs, a_id, b_id) =
                        matmul_shape(&graph, node, "FusedMatMulBiasAct");
                    let bias_id = node.inputs[2];
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
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
                        has_bias: 1,
                        bias_off_f32: (arena.offset(bias_id) / 4) as u32,
                        act_id,
                    });
                }
                Op::Binary(bop) => {
                    schedule.push(Step::Binary {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: binary_op_id(*bop),
                    });
                }
                Op::Activation(act) => {
                    schedule.push(Step::Unary {
                        n: elems,
                        in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: activation_op_id(*act),
                    });
                }
                Op::Compare(cop) => {
                    schedule.push(Step::Compare {
                        n: elems,
                        a_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        c_off: (arena.offset(node.id) / 4) as u32,
                        op: compare_op_id(*cop),
                    });
                }
                Op::Where => {
                    schedule.push(Step::Where {
                        n: elems,
                        cond_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        x_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        y_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // 72-u32 metadata buffer (8 input offsets + 16 steps *
                    // 4 u32s) uploaded once at compile time; the kernel
                    // walks the chain interpretively in registers. Caps
                    // match the cross-backend Metal MSL / wgpu WGSL
                    // encoders.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-cuda ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Run UnfuseElementwiseRegions to fall back \
                                to atomic ops.",
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
                    // op_sub mappings shared with rlx-metal MSL +
                    // rlx-wgpu WGSL chain kernels — the encoder
                    // produces one byte stream that all three backends
                    // interpret identically.
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
                    // meta layout: 16 input offsets + 32 steps × 4 u32s = 144 words.
                    let mut meta_data: Vec<u32> = Vec::with_capacity(144);
                    meta_data.extend_from_slice(&input_offs);
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
                    meta_data.extend_from_slice(&chain_enc);
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: elementwise_region meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::ElementwiseRegion {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        input_offs,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                        meta_idx,
                    });
                }
                Op::Reduce {
                    op,
                    axes,
                    keep_dim: _,
                } => {
                    // v2: reduce along the LAST axis only — same v1
                    // simplification rlx-wgpu had.
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    if axes.len() != 1 || axes[0] != in_dims.len() - 1 {
                        panic!(
                            "rlx-cuda Reduce: only single last-axis supported \
                                (got axes={axes:?}, rank={})",
                            in_dims.len()
                        );
                    }
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Reduce {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        op: reduce_op_id(*op),
                    });
                }
                Op::Softmax { axis: _ } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Softmax {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let is_layer = matches!(&node.op, Op::LayerNorm { .. });
                    let gamma_id = node.inputs[1];
                    let beta_id = if is_layer && node.inputs.len() >= 3 {
                        node.inputs[2]
                    } else {
                        gamma_id
                    };
                    schedule.push(Step::LayerNorm {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        gamma_off: (arena.offset(gamma_id) / 4) as u32,
                        beta_off: (arena.offset(beta_id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        op: if is_layer { 0 } else { 1 },
                    });
                }
                Op::FusedResidualLN { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    schedule.push(Step::FusedResidualLn {
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
                    });
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
                        schedule.push(Step::Gather {
                            n_out: elems,
                            n_idx,
                            dim,
                            vocab,
                            in_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
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
                        schedule.push(Step::GatherAxis {
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off: (arena.offset(table_id) / 4) as u32,
                            idx_off: (arena.offset(idx_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    }
                }
                Op::Narrow { axis, start, len } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let outer: u32 = in_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = in_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_in = in_dims[*axis].unwrap_static() as u32;
                    schedule.push(Step::Narrow {
                        total: elems,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::Transpose { perm } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let rank = perm.len();
                    let in_dims_u: Vec<u32> =
                        in_dims.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Cumulative input strides (row-major, innermost = 1).
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims_u[i + 1];
                    }
                    let out_dims_u: Vec<u32> = perm.iter().map(|&i| in_dims_u[i]).collect();
                    let strides_for_out: Vec<u32> = perm.iter().map(|&i| in_strides[i]).collect();
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims_u);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Transpose {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let rank = target_shape.len();
                    if rank != in_shape.len() {
                        panic!(
                            "rlx-cuda Expand: rank mismatch (in={}, target={})",
                            in_shape.len(),
                            rank
                        );
                    }
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
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
                    let meta = ctx
                        .default_stream()
                        .clone_htod(&meta_data)
                        .expect("rlx-cuda: meta upload failed");
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta);
                    schedule.push(Step::Expand {
                        rank: rank as u32,
                        out_total: elems,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        meta_idx,
                    });
                }
                Op::Concat { axis } => {
                    // Caller convention: one Step::Concat per input, copying
                    // each input's slice into the output at the right axis offset.
                    let mut start: u32 = 0;
                    let out_dims = node.shape.dims();
                    let outer: u32 = out_dims[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = out_dims[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_out_size = out_dims[*axis].unwrap_static() as u32;
                    for &in_id in &node.inputs {
                        let in_dims = graph.node(in_id).shape.dims();
                        let axis_in = in_dims[*axis].unwrap_static() as u32;
                        let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                        schedule.push(Step::Concat {
                            total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size,
                            start,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                        start += axis_in;
                    }
                }
                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale: _,
                    attn_logit_softcap: _,
                } => {
                    // Rank-3 inputs already promoted by unfuse; here we only
                    // see rank-4 [B, H, S, D].
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    if q_shape.len() != 4 {
                        panic!("rlx-cuda Attention: unfuse should have promoted to rank-4");
                    }
                    let batch = q_shape[0].unwrap_static() as u32;
                    let heads = q_shape[1].unwrap_static() as u32;
                    let seq_q = q_shape[2].unwrap_static() as u32;
                    let seq_k = k_shape[2].unwrap_static() as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[3]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, 0u32, 0u32),
                    };
                    let _ = num_heads;
                    schedule.push(Step::Attention {
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
                    });
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
                    if q_shape.len() != 4 {
                        panic!("rlx-cuda AttentionBackward: unfuse should have promoted to rank-4");
                    }
                    let batch = q_shape[0].unwrap_static() as u32;
                    let heads = q_shape[1].unwrap_static() as u32;
                    let seq_q = q_shape[2].unwrap_static() as u32;
                    let seq_k = k_shape[2].unwrap_static() as u32;
                    let hd = *head_dim as u32;
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, 0u32),
                        MaskKind::Causal => (1u32, 0u32, 0u32),
                        MaskKind::Custom => (2u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, *w as u32),
                        MaskKind::Bias => (4u32, (arena.offset(node.inputs[4]) / 4) as u32, 0u32),
                    };
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    schedule.push(Step::AttentionBackward {
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
                    });
                }
                Op::Rope { head_dim, n_rot: _ } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-cuda Rope: last_dim {} not multiple of head_dim {}",
                            last, head_dim
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-cuda Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    schedule.push(Step::Rope {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        cos_off: (arena.offset(cos_id) / 4) as u32,
                        sin_off: (arena.offset(sin_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        last_dim: last as u32,
                    });
                }
                Op::Cumsum { axis: _, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::Cumsum {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                    });
                }
                Op::TopK { k } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    schedule.push(Step::TopK {
                        outer,
                        inner,
                        k: *k as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::GroupedMatMul => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = w_dims[2].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    schedule.push(Step::GroupedMatmul {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
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
                }
                Op::ScatterAdd => {
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
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    schedule.push(Step::ScatterAddZero { out_off, out_total });
                    schedule.push(Step::ScatterAddAcc {
                        out_off,
                        upd_off: (arena.offset(upd_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        num_updates,
                        trailing,
                        out_dim,
                    });
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::quant::QuantScheme;
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
                    } else {
                        let (block_size, scheme_id) = match scheme {
                            QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                            QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                            QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                            QuantScheme::Fp8E4m3 => (1, 3u32),
                            QuantScheme::Fp8E5m2 => (1, 4u32),
                            QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                            other => panic!("rlx-cuda DequantMatMul: unsupported scheme {other:?}"),
                        };
                        let scale_id = node.inputs[2];
                        let zp_id = node.inputs[3];
                        schedule.push(Step::DequantMatmul {
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
                        });
                    }
                }
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!("rlx-cuda SelectiveScan: state_size {state_size} > 256 cap");
                    }
                    let x_id = node.inputs[0];
                    let dt_id = node.inputs[1];
                    let a_id = node.inputs[2];
                    let b_id = node.inputs[3];
                    let c_id = node.inputs[4];
                    let in_dims = graph.node(x_id).shape.dims();
                    schedule.push(Step::SelectiveScan {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq: in_dims[1].unwrap_static() as u32,
                        hidden: in_dims[2].unwrap_static() as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        delta_off: (arena.offset(dt_id) / 4) as u32,
                        a_off: (arena.offset(a_id) / 4) as u32,
                        b_off: (arena.offset(b_id) / 4) as u32,
                        c_off: (arena.offset(c_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    });
                }
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-cuda GatedDeltaNet: state_size {state_size} > {}",
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
                }
                Op::Custom { name, attrs, .. } => {
                    if name != "llada2.group_limited_gate" {
                        panic!("rlx-cuda: unsupported Op::Custom('{name}')");
                    }
                    let sig_id = node.inputs[0];
                    let route_id = node.inputs[1];
                    let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                    let mut attr_buf = [0u8; 20];
                    let n = attrs.len().min(20);
                    attr_buf[..n].copy_from_slice(&attrs[..n]);
                    schedule.push(Step::Llada2GroupLimitedGate {
                        sig_off: (arena.offset(sig_id) / 4) as u32,
                        route_off: (arena.offset(route_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        n_elems,
                        attrs: attr_buf,
                    });
                }

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
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        dst_off: arena.offset(node.id) as u32,
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
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        d_loss_off: arena.offset(node.inputs[7]) as u32,
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_off: arena.offset(node.id) as u32,
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
                        positions_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: arena.offset(node.inputs[6]) as u32,
                        meta_len: elem_len(node.inputs[6]),
                        prep_off: arena.offset(node.id) as u32,
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
                        prep_off: arena.offset(prep_id) as u32,
                        prep_len: elem_len(prep_id),
                        meta_off: arena.offset(node.inputs[1]) as u32,
                        meta_len: elem_len(node.inputs[1]),
                        dst_off: arena.offset(node.id) as u32,
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

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let op_id = reduce_op_id(*kind);
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => {
                            schedule.push(Step::Pool1d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                l: in_dims[2].unwrap_static() as u32,
                                l_out: out_dims[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride[0] as u32,
                                pl: padding[0] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        2 => {
                            schedule.push(Step::Pool2d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                h: in_dims[2].unwrap_static() as u32,
                                w: in_dims[3].unwrap_static() as u32,
                                h_out: out_dims[2].unwrap_static() as u32,
                                w_out: out_dims[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride[0] as u32,
                                sw: stride[1] as u32,
                                ph: padding[0] as u32,
                                pw: padding[1] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        3 => {
                            schedule.push(Step::Pool3d {
                                n: in_dims[0].unwrap_static() as u32,
                                c: in_dims[1].unwrap_static() as u32,
                                d: in_dims[2].unwrap_static() as u32,
                                h: in_dims[3].unwrap_static() as u32,
                                w: in_dims[4].unwrap_static() as u32,
                                d_out: out_dims[2].unwrap_static() as u32,
                                h_out: out_dims[3].unwrap_static() as u32,
                                w_out: out_dims[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride[0] as u32,
                                sh: stride[1] as u32,
                                sw: stride[2] as u32,
                                pd: padding[0] as u32,
                                ph: padding[1] as u32,
                                pw: padding[2] as u32,
                                op: op_id,
                                in_off,
                                out_off,
                            });
                        }
                        other => panic!("rlx-cuda Pool: unsupported kernel rank {other}"),
                    }
                }
                Op::LayerNorm2d { eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::LayerNorm2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::ConvTranspose2d {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::ConvTranspose2d {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        w_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w_in: in_shape.dim(3).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    });
                }
                Op::GroupNorm { num_groups, eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::GroupNorm {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        g_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        num_groups: *num_groups as u32,
                        eps_bits: eps.to_bits(),
                    });
                }
                Op::ResizeNearest2x => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2x {
                        src_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
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
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let in_off = (arena.offset(in_id) / 4) as u32;
                    let w_off = (arena.offset(w_id) / 4) as u32;
                    let out_off = (arena.offset(node.id) / 4) as u32;
                    match kernel_size.len() {
                        1 => {
                            schedule.push(Step::Conv1d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                l: in_dims[2].unwrap_static() as u32,
                                l_out: out_dims[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride[0] as u32,
                                pl: padding[0] as u32,
                                dl: dilation[0] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        2 => {
                            schedule.push(Step::Conv2d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                h: in_dims[2].unwrap_static() as u32,
                                w: in_dims[3].unwrap_static() as u32,
                                h_out: out_dims[2].unwrap_static() as u32,
                                w_out: out_dims[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride[0] as u32,
                                sw: stride[1] as u32,
                                ph: padding[0] as u32,
                                pw: padding[1] as u32,
                                dh: dilation[0] as u32,
                                dw: dilation[1] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        3 => {
                            schedule.push(Step::Conv3d {
                                n: in_dims[0].unwrap_static() as u32,
                                c_in: in_dims[1].unwrap_static() as u32,
                                c_out: w_dims[0].unwrap_static() as u32,
                                d: in_dims[2].unwrap_static() as u32,
                                h: in_dims[3].unwrap_static() as u32,
                                w: in_dims[4].unwrap_static() as u32,
                                d_out: out_dims[2].unwrap_static() as u32,
                                h_out: out_dims[3].unwrap_static() as u32,
                                w_out: out_dims[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride[0] as u32,
                                sh: stride[1] as u32,
                                sw: stride[2] as u32,
                                pd: padding[0] as u32,
                                ph: padding[1] as u32,
                                pw: padding[2] as u32,
                                dd: dilation[0] as u32,
                                dh: dilation[1] as u32,
                                dw: dilation[2] as u32,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            });
                        }
                        other => panic!("rlx-cuda Conv: unsupported kernel rank {other}"),
                    }
                }
                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let is_greedy = *top_k == 0
                        && (*top_p - 1.0).abs() < 1e-6
                        && (*temperature - 1.0).abs() < 1e-6;
                    if is_greedy {
                        schedule.push(Step::Argmax {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                        });
                    } else {
                        schedule.push(Step::Sample {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            top_k: *top_k as u32,
                            top_p_bits: top_p.to_bits(),
                            temp_bits: temperature.to_bits(),
                            seed_lo: *seed as u32,
                            seed_hi: (*seed >> 32) as u32,
                        });
                    }
                }
                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let eps_bits = eps.to_bits();
                    let off = |i: usize| arena.offset(node.inputs[i]) as u32;
                    let common = (off(0), off(1), off(2), off(3), rows, h, eps_bits);
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => {
                            schedule.push(Step::RmsNormBackwardInput {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dx_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardGamma { .. } => {
                            schedule.push(Step::RmsNormBackwardGamma {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dgamma_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        Op::RmsNormBackwardBeta { .. } => {
                            schedule.push(Step::RmsNormBackwardBeta {
                                x_byte_off: common.0,
                                gamma_byte_off: common.1,
                                beta_byte_off: common.2,
                                dy_byte_off: common.3,
                                dbeta_byte_off: arena.offset(node.id) as u32,
                                rows: common.4,
                                h: common.5,
                                eps_bits: common.6,
                            });
                        }
                        _ => unreachable!(),
                    }
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
                    schedule.push(Step::RopeBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        cos_byte_off: arena.offset(node.inputs[1]) as u32,
                        sin_byte_off: arena.offset(node.inputs[2]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        batch,
                        seq,
                        hidden,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        cos_len,
                    });
                }
                Op::CumsumBackward { exclusive, .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                    schedule.push(Step::CumsumBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        dx_byte_off: arena.offset(node.id) as u32,
                        rows,
                        cols,
                        exclusive: *exclusive,
                    });
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
                    schedule.push(Step::GatherBackward {
                        dy_byte_off: arena.offset(node.inputs[0]) as u32,
                        indices_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                    });
                }
                other => panic!(
                    "rlx-cuda: op {other:?} not yet lowered. \
                     Open a follow-up PR if you hit this — every other op \
                     in the IR is wired."
                ),
            }
        }

        let schedule = fuse_elementwise_chains(schedule);

        let blas = cuda_blas();
        let blas_lt = cuda_blas_lt_handle();
        let blas_lt_workspace = if blas_lt.is_some() {
            ctx.default_stream()
                .alloc_zeros::<u8>(CUBLASLT_WORKSPACE_BYTES)
                .ok()
        } else {
            None
        };
        let dnn = cuda_dnn_handle();
        let dnn_workspace = if dnn.is_some() {
            ctx.default_stream()
                .alloc_zeros::<u8>(CUDNN_WORKSPACE_BYTES)
                .ok()
        } else {
            None
        };

        let streams = match exec_mode {
            ExecMode::MultiStream(n) if n > 1 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if let Ok(s) = ctx.new_stream() {
                        v.push(s);
                    }
                }
                v
            }
            _ => Vec::new(),
        };

        Self {
            ctx,
            blas,
            blas_lt,
            blas_lt_workspace,
            dnn,
            dnn_workspace,
            half_act_scratch: None,
            dequant_scratch_off,
            graph,
            arena,
            schedule,
            input_offsets,
            param_offsets,
            meta_buffers,
            exec_mode,
            captured_graph: None,
            streams,
            active_extent: None,
        }
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every step in the schedule passes
    /// `Step::safe_for_active_extent`. Bypasses captured CUDA Graph
    /// (recorded at full extent) when active. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    /// Declared graph-output dtypes, in `graph.outputs` order. Used by
    /// the runtime wrapper's `run_typed` to narrow f32 outputs back to
    /// the declared dtype on the way out.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let off_f32 = self.arena.offset(id) / 4;
            let stream = self.ctx.default_stream();
            let mut slot = self.arena.buffer.slice_mut(off_f32..off_f32 + data.len());
            stream
                .memcpy_htod(data, &mut slot)
                .expect("rlx-cuda: param upload failed");
        }
    }

    /// Upload packed U8/I8 GGUF weights into the param slot (byte offset).
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let byte_off = self.arena.offset(id);
            let stream = self.ctx.default_stream();
            crate::gguf_host::upload_param_bytes(&stream, &mut self.arena.buffer, byte_off, data);
        }
    }

    /// Upload a param as packed half-precision bits (`u16` per element).
    /// Caller passes the raw IEEE-754 binary16 (`F16`) or BFloat16
    /// (`Bf16`) bit pattern; the backend stores it in the half-arena
    /// side-buffer and skips the f32 slot entirely. Use cases:
    /// 2× weight-memory savings for inference, plus Tensor Core matmul
    /// via `cublasGemmEx` when both A and B (or just B) are stored
    /// half-precision.
    ///
    /// When the same `name` is also `set_param`'d as f32, the
    /// half-arena entry takes precedence in the matmul dispatch. Use
    /// only one of the two for any given param.
    pub fn set_param_half(&mut self, name: &str, dtype: crate::arena::HalfDtype, bits: &[u16]) {
        let id = match self.param_offsets.get(name) {
            Some(&id) if self.arena.has(id) => id,
            _ => return,
        };
        let f32_off = (self.arena.offset(id) / 4) as u32;
        let off = self
            .arena
            .register_half_param(&self.ctx, id, f32_off, bits.len(), dtype);
        let stream = self.ctx.default_stream();
        if let Some(buf) = self.arena.half_buffer.as_mut() {
            let mut slot = buf.slice_mut(off..off + bits.len());
            stream
                .memcpy_htod(bits, &mut slot)
                .expect("rlx-cuda: half-param upload failed");
        }
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let default_stream = self.ctx.default_stream();
        let stream = default_stream.clone();

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into the
        // captured CUDA Graph.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let mut slot = self.arena.buffer.slice_mut(off_f32..off_f32 + data.len());
                stream
                    .memcpy_htod(data, &mut slot)
                    .expect("rlx-cuda: input upload failed");
            }
        }

        // Active-extent (PLAN L1): when set + every Step safe, bypass
        // captured CUDA Graph (recorded at full extent) and dispatch
        // per-step with scaled launch dims via the normal loop.
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        // Scale a count by actual/upper with ceiling-division, clamped to [0, full].
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // CUDA Graph fast path: replay a previously-captured schedule.
        // The first run with `ExecMode::Graph` falls through to the
        // normal dispatch loop with stream capture turned on; the
        // resulting graph is stashed in `self.captured_graph` and
        // replayed on every subsequent run.
        let do_replay =
            active.is_none() && self.exec_mode == ExecMode::Graph && self.captured_graph.is_some();
        let do_capture =
            active.is_none() && self.exec_mode == ExecMode::Graph && self.captured_graph.is_none();

        if do_replay {
            self.captured_graph
                .as_ref()
                .unwrap()
                .launch()
                .expect("rlx-cuda: graph replay failed");
            stream.synchronize().expect("rlx-cuda: stream sync failed");
            return self.read_outputs(&stream);
        }
        let _ = do_replay;

        if do_capture {
            stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .expect("rlx-cuda: begin_capture failed");
        }

        // Multi-stream scheduler state. When `exec_mode ==
        // MultiStream(n)`, each Step gets assigned to one of `n` pool
        // streams based on producer-consumer dependencies on arena
        // offsets. Independent ops (e.g. unfused Q/K/V matmuls)
        // parallelise; producer-consumer chains stay on one stream.
        let multi_stream =
            matches!(self.exec_mode, ExecMode::MultiStream(_)) && !self.streams.is_empty();
        let mut producer_of: HashMap<u32, usize> = HashMap::new();
        let mut last_event: HashMap<usize, cudarc::driver::CudaEvent> = HashMap::new();
        let mut rr_cursor: usize = 0;

        // Dispatch each step. Each iteration is wrapped in an NVTX
        // range so nsight-systems traces show step boundaries cleanly.
        // Gated behind the `nvtx` feature because CUDA 13 removed
        // `nvToolsExt.dll`; cudarc panics on first call when the lib
        // isn't loadable.
        for step in &self.schedule {
            #[cfg(feature = "nvtx")]
            let _nvtx = cudarc::nvtx::scoped_range(step_name(step));
            // PLAN L3: cross-backend Perfetto trace; no-op when env
            // var RLX_TRACE_PERFETTO unset.
            let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "cuda");

            // Per-step stream selection. In single-stream mode `stream`
            // shadows to the default stream; in multi-stream mode it
            // shadows to the assigned pool stream (and we cross-stream
            // event-wait on every producer not on the chosen stream).
            let assigned_idx: Option<usize> = if multi_stream {
                let (reads, _) = step_offsets(step);
                let mut producer_streams: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for r in &reads {
                    if let Some(&s) = producer_of.get(r) {
                        producer_streams.insert(s);
                    }
                }
                let chosen = if producer_streams.is_empty() {
                    let s = rr_cursor % self.streams.len();
                    rr_cursor += 1;
                    s
                } else if producer_streams.len() == 1 {
                    *producer_streams.iter().next().unwrap()
                } else {
                    // Multiple producers — keep the chosen one's queue
                    // intact and event-wait on the others.
                    let chosen = *producer_streams.iter().next().unwrap();
                    for s in &producer_streams {
                        if *s != chosen
                            && let Some(evt) = last_event.get(s)
                        {
                            let _ = self.streams[chosen].wait(evt);
                        }
                    }
                    chosen
                };
                Some(chosen)
            } else {
                None
            };
            let stream: Arc<cudarc::driver::CudaStream> = match assigned_idx {
                Some(i) => self.streams[i].clone(),
                None => default_stream.clone(),
            };
            // Re-bind cuBLAS / cuDNN handles to the active stream so
            // their internal kernel launches go to the right queue.
            if multi_stream {
                if let Some(blas) = self.blas.as_ref() {
                    let blas = blas.lock().unwrap();
                    unsafe {
                        let _ = cudarc::cublas::result::set_stream(
                            *blas.handle(),
                            stream.cu_stream() as _,
                        );
                    }
                }
                if let Some(handle) = self.dnn {
                    unsafe {
                        let _ = cudarc::cudnn::result::set_stream(
                            handle,
                            stream.cu_stream() as cudnn_sys::cudaStream_t,
                        );
                    }
                }
            }
            match step {
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
                } => {
                    // Tier 0: mixed-precision GemmEx — when B (the weight)
                    // is stored in the half-arena, cast activations to
                    // f16/bf16 in a scratch buffer and call cublasGemmEx
                    // with both inputs half + f32 accumulator. Falls
                    // through to cublasLt on any setup or runtime error.
                    let used_mixed = try_mixed_precision_gemm(
                        &self.ctx,
                        &mut self.arena,
                        &mut self.half_act_scratch,
                        self.blas.as_ref(),
                        &stream,
                        *m,
                        *k,
                        *n,
                        *batch,
                        *a_off_f32,
                        *b_off_f32,
                        *c_off_f32,
                    );
                    if used_mixed {
                        // Optional bias / activation epilogue.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(&mut self.arena.buffer)
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (mixed) failed");
                            }
                        }
                        // Multi-stream tail bookkeeping still runs at end of step.
                        if let Some(idx) = assigned_idx {
                            if let Ok(evt) = stream.record_event(None) {
                                last_event.insert(idx, evt);
                            }
                            let (_, writes) = step_offsets(step);
                            for w in &writes {
                                producer_of.insert(*w, idx);
                            }
                        }
                        continue;
                    }

                    // Tier 1: cublasLt fused (matmul + bias + relu/gelu in
                    // one launch). Only used when the activation is one of
                    // the two cublasLt natively fuses; other acts (silu,
                    // sigmoid, etc.) fall through to the sgemm + epilogue
                    // kernel path.
                    let try_cublaslt = self.blas_lt.is_some()
                        && self.blas_lt_workspace.is_some()
                        && cublaslt_act_supported(*act_id);
                    let used_cublaslt = if try_cublaslt {
                        let lt_handle = self.blas_lt.unwrap();
                        let workspace = self.blas_lt_workspace.as_mut().unwrap();
                        let (workspace_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _record) = self.arena.buffer.device_ptr_mut(&stream);
                        let cu_stream = stream.cu_stream();
                        let act = cublaslt_act_for(*act_id);
                        let r = unsafe {
                            cublaslt_matmul_fused(
                                lt_handle,
                                workspace_ptr,
                                CUBLASLT_WORKSPACE_BYTES,
                                arena_ptr,
                                *m,
                                *k,
                                *n,
                                *a_off_f32,
                                *b_off_f32,
                                *c_off_f32,
                                *has_bias != 0,
                                *bias_off_f32,
                                act,
                                *batch,
                                *a_batch_stride,
                                *b_batch_stride,
                                *c_batch_stride,
                                cu_stream,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("matmul.cublasLt", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cublaslt {
                        continue;
                    }

                    // Tier 2: cuBLAS sgemm via raw pointers (bypasses
                    // the borrow checker's same-buffer aliasing).
                    let used_cublas = if let Some(blas) = self.blas.as_ref() {
                        let blas = blas.lock().unwrap();
                        let (arena_ptr_u64, _record) = self.arena.buffer.device_ptr_mut(&stream);
                        let a_dev = arena_ptr_u64 + (*a_off_f32 as u64) * 4;
                        let b_dev = arena_ptr_u64 + (*b_off_f32 as u64) * 4;
                        let c_dev = arena_ptr_u64 + (*c_off_f32 as u64) * 4;
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        // cuBLAS is column-major; we have row-major. Trick:
                        // computing C = A·B (row-major) is the same as
                        // computing C^T = B^T · A^T (column-major), and
                        // viewing our row-major arrays as column-major
                        // automatically yields the transpose.
                        let result = unsafe {
                            if *batch == 1 {
                                cudarc::cublas::result::sgemm(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    a_dev as *const f32,
                                    *k as i32,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                )
                            } else {
                                cudarc::cublas::result::sgemm_strided_batched(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    *b_batch_stride as i64,
                                    a_dev as *const f32,
                                    *k as i32,
                                    *a_batch_stride as i64,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                    *c_batch_stride as i64,
                                    *batch as i32,
                                )
                            }
                        };
                        if let Err(ref e) = result {
                            log_fallback("matmul.cublasSgemm", e);
                        }
                        result.is_ok()
                    } else {
                        false
                    };

                    if used_cublas {
                        // Optional fused epilogue (bias + activation) as
                        // a separate element-wise kernel.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(&mut self.arena.buffer)
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue launch failed");
                            }
                        }
                    } else if use_wmma() {
                        // WMMA Tensor Core path: 32×64 block tile, 128 threads/block,
                        // SM 70+ only. Doesn't fuse bias/activation — those go to the
                        // shared epilogue kernel.
                        let kernel = matmul_wmma_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(32), *batch),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(&mut self.arena.buffer)
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul_wmma launch failed");
                        }
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(&mut self.arena.buffer)
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (post-wmma) failed");
                            }
                        }
                    } else {
                        // Custom scalar kernel fallback: 64×64 block tile, 4×4 register tile.
                        let kernel = matmul_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(&mut self.arena.buffer)
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul launch failed");
                        }
                    }
                }
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = binary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: binary launch failed");
                    }
                }
                Step::ElementwiseRegion {
                    len,
                    num_inputs,
                    num_steps,
                    dst_off,
                    input_offs: _,
                    scalar_input_mask,
                    input_modulus,
                    meta_idx,
                } => {
                    let len_s = scale(*len);
                    if len_s == 0 {
                        continue;
                    }
                    let kernel = elementwise_region_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(len_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    // input_modulus is passed by-value as a 64-byte
                    // const param (16 u32s). Could move to meta_buffer
                    // but a constant param keeps the kernel signature
                    // self-describing.
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&len_s)
                        .arg(num_inputs)
                        .arg(num_steps)
                        .arg(dst_off)
                        .arg(&self.meta_buffers[*meta_idx])
                        .arg(scalar_input_mask)
                        .arg(input_modulus);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: elementwise_region launch failed");
                    }
                }
                Step::FusedBinaryUnary {
                    n,
                    a_off,
                    b_off,
                    out_off,
                    bin_op,
                    un_op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = fused_binary_unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(out_off)
                        .arg(bin_op)
                        .arg(un_op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_binary_unary launch failed");
                    }
                }
                Step::Unary {
                    n,
                    in_off,
                    out_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&n_s)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: unary launch failed");
                    }
                }
                Step::Compare {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = compare_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: compare launch failed");
                    }
                }
                Step::Where {
                    n,
                    cond_off,
                    x_off,
                    y_off,
                    out_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = where_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&n_s)
                        .arg(cond_off)
                        .arg(x_off)
                        .arg(y_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: where launch failed");
                    }
                }
                Step::Reduce {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = reduce_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: reduce launch failed");
                    }
                }
                Step::Softmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax launch failed");
                    }
                }
                Step::LayerNorm {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    gamma_off,
                    beta_off,
                    eps_bits,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = layernorm_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(eps_bits)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layernorm launch failed");
                    }
                }
                Step::FusedResidualLn {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off,
                    eps_bits,
                    has_bias,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = fused_residual_ln_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(residual_off)
                        .arg(bias_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_residual_ln launch failed");
                    }
                }
                Step::Gather {
                    n_out,
                    n_idx,
                    dim,
                    vocab,
                    in_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_out, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n_out)
                        .arg(n_idx)
                        .arg(dim)
                        .arg(vocab)
                        .arg(in_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather launch failed");
                    }
                }
                Step::GatherAxis {
                    total,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                    table_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_axis_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(total)
                        .arg(outer)
                        .arg(axis_dim)
                        .arg(num_idx)
                        .arg(trailing)
                        .arg(table_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather_axis launch failed");
                    }
                }
                Step::Narrow {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = narrow_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: narrow launch failed");
                    }
                }
                Step::Argmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let kernel = argmax_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: argmax launch failed");
                    }
                }
                Step::Transpose {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = transpose_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: transpose launch failed");
                    }
                }
                Step::Expand {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = expand_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: expand launch failed");
                    }
                }
                Step::Concat {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = concat_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: concat launch failed");
                    }
                }
                Step::Attention {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    window,
                } => {
                    // FlashAttention-1: BR=16 q-rows per block, BC=32 KV-tile,
                    // 128 threads/block. grid=(q_blocks, batch*heads, 1).
                    let kernel = attention_kernel(&self.ctx);
                    let q_blocks = (*seq_q).div_ceil(16);
                    let cfg = LaunchConfig {
                        grid_dim: (q_blocks, batch * heads, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(batch)
                        .arg(heads)
                        .arg(seq_q)
                        .arg(seq_k)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention launch failed");
                    }
                }
                Step::AttentionBackward {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    dy_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    window,
                    wrt,
                } => {
                    let kernel = attention_bwd_kernel(&self.ctx);
                    let seq_axis = if *wrt == 0 { *seq_q } else { *seq_k };
                    let y_blocks = seq_axis.div_ceil(256);
                    let cfg = LaunchConfig {
                        grid_dim: (batch * heads, y_blocks, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(batch)
                        .arg(heads)
                        .arg(seq_q)
                        .arg(seq_k)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window)
                        .arg(wrt);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention_bwd launch failed");
                    }
                }
                Step::Rope {
                    n_total,
                    seq,
                    head_dim,
                    half,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off,
                    last_dim,
                } => {
                    let kernel = rope_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n_total)
                        .arg(seq)
                        .arg(head_dim)
                        .arg(half)
                        .arg(in_off)
                        .arg(cos_off)
                        .arg(sin_off)
                        .arg(out_off)
                        .arg(last_dim);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: rope launch failed");
                    }
                }
                Step::Cumsum {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    exclusive,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = cumsum_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(outer_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(exclusive);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: cumsum launch failed");
                    }
                }
                Step::TopK {
                    outer,
                    inner,
                    k,
                    in_off,
                    out_off,
                } => {
                    let kernel = topk_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(outer)
                        .arg(inner)
                        .arg(k)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: topk launch failed");
                    }
                }
                Step::GroupedMatmul {
                    m,
                    k,
                    n,
                    num_experts,
                    in_off,
                    w_off,
                    idx_off,
                    out_off,
                } => {
                    // Tier 1: sorted-batch dispatch via cuBLAS. Reads
                    // the idx buffer back to host, finds runs of
                    // identical consecutive expert ids, and issues one
                    // cublasSgemm per run. Wins big when tokens are
                    // pre-sorted by expert (the standard MoE upstream
                    // convention) — for random idx the run count is
                    // ~m and the launch overhead would negate the win,
                    // so we fall back to the kernel in that case.
                    let used_sorted = if let Some(blas) = self.blas.as_ref() {
                        // Sync first so prior writes to idx are visible.
                        stream
                            .synchronize()
                            .expect("rlx-cuda: stream sync before idx download");
                        let idx_host = {
                            let idx_slot = self
                                .arena
                                .buffer
                                .slice(*idx_off as usize..(idx_off + m) as usize);
                            stream.clone_dtoh(&idx_slot).ok()
                        };
                        match idx_host {
                            Some(idx_vec) => {
                                let mut runs: Vec<(u32, u32, u32)> = Vec::new();
                                let mut i = 0usize;
                                let mn = *m as usize;
                                while i < mn {
                                    let e = idx_vec[i] as u32;
                                    let mut j = i + 1;
                                    while j < mn && (idx_vec[j] as u32) == e {
                                        j += 1;
                                    }
                                    if e < *num_experts {
                                        runs.push((i as u32, j as u32, e));
                                    }
                                    i = j;
                                }
                                // Heuristic: bail when the run count
                                // exceeds m/4 (idx isn't usefully sorted).
                                let threshold = (mn / 4).max(2);
                                if !runs.is_empty() && runs.len() <= threshold {
                                    let blas = blas.lock().unwrap();
                                    let (arena_ptr, _record) =
                                        self.arena.buffer.device_ptr_mut(&stream);
                                    let alpha: f32 = 1.0;
                                    let beta: f32 = 0.0;
                                    let mut all_ok = true;
                                    for (lo, hi, e) in &runs {
                                        let rows = hi - lo;
                                        let a_dev = arena_ptr + ((*in_off + lo * k) as u64) * 4;
                                        let b_dev = arena_ptr + ((*w_off + e * k * n) as u64) * 4;
                                        let c_dev = arena_ptr + ((*out_off + lo * n) as u64) * 4;
                                        let r = unsafe {
                                            cudarc::cublas::result::sgemm(
                                                *blas.handle(),
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                                *n as i32,
                                                rows as i32,
                                                *k as i32,
                                                &alpha as *const f32,
                                                b_dev as *const f32,
                                                *n as i32,
                                                a_dev as *const f32,
                                                *k as i32,
                                                &beta as *const f32,
                                                c_dev as *mut f32,
                                                *n as i32,
                                            )
                                        };
                                        if r.is_err() {
                                            all_ok = false;
                                            break;
                                        }
                                    }
                                    all_ok
                                } else {
                                    false
                                }
                            }
                            None => false,
                        }
                    } else {
                        false
                    };
                    if used_sorted {
                        continue;
                    }

                    // Fallback: per-token expert lookup kernel.
                    let kernel = grouped_matmul_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(num_experts)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: grouped_matmul launch failed");
                    }
                }
                Step::ScatterAddZero { out_off, out_total } => {
                    let kernel = scatter_add_zero_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(out_off)
                        .arg(out_total);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_zero launch failed");
                    }
                }
                Step::ScatterAddAcc {
                    out_off,
                    upd_off,
                    idx_off,
                    num_updates,
                    trailing,
                    out_dim,
                } => {
                    let kernel = scatter_add_acc_kernel(&self.ctx);
                    let total = num_updates * trailing;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(out_off)
                        .arg(upd_off)
                        .arg(idx_off)
                        .arg(num_updates)
                        .arg(trailing)
                        .arg(out_dim);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_acc launch failed");
                    }
                }
                Step::DequantMatmul {
                    m,
                    k,
                    n,
                    block_size,
                    scheme_id,
                    x_off,
                    w_off,
                    scale_off,
                    zp_off,
                    out_off,
                } => {
                    let kernel = dequant_matmul_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(block_size)
                        .arg(scheme_id)
                        .arg(x_off)
                        .arg(w_off)
                        .arg(scale_off)
                        .arg(zp_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: dequant_matmul launch failed");
                    }
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                            &self.ctx,
                            &stream,
                            &mut self.arena.buffer,
                            blas,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_matmul_gguf(
                            &stream,
                            &mut self.arena.buffer,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                            &self.ctx,
                            &stream,
                            &mut self.arena.buffer,
                            blas,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &stream,
                            &mut self.arena.buffer,
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
                }
                Step::Sample {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    top_k,
                    top_p_bits,
                    temp_bits,
                    seed_lo,
                    seed_hi,
                } => {
                    let kernel = sample_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(top_k)
                        .arg(top_p_bits)
                        .arg(temp_bits)
                        .arg(seed_lo)
                        .arg(seed_hi);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: sample launch failed");
                    }
                }
                Step::SelectiveScan {
                    batch,
                    seq,
                    hidden,
                    state_size,
                    x_off,
                    delta_off,
                    a_off,
                    b_off,
                    c_off,
                    out_off,
                } => {
                    let kernel = selective_scan_kernel(&self.ctx);
                    let total = batch * hidden;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(batch)
                        .arg(seq)
                        .arg(hidden)
                        .arg(state_size)
                        .arg(x_off)
                        .arg(delta_off)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: selective_scan launch failed");
                    }
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
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
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
                    sig_off,
                    route_off,
                    out_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *sig_off as usize,
                        *route_off as usize,
                        *out_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::LayerNorm2d {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    eps_bits,
                } => {
                    let kernel = layer_norm2d_kernel(&self.ctx);
                    let total = n * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layer_norm2d launch failed");
                    }
                }
                Step::ConvTranspose2d {
                    src_off,
                    w_off,
                    dst_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                } => {
                    let kernel = conv_transpose2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(src_off)
                        .arg(w_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c_in)
                        .arg(h)
                        .arg(w_in)
                        .arg(c_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv_transpose2d launch failed");
                    }
                }
                Step::GroupNorm {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps_bits,
                } => {
                    let kernel = group_norm_kernel(&self.ctx);
                    let grid = n * num_groups;
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(num_groups)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm launch failed");
                    }
                }
                Step::ResizeNearest2x {
                    src_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    let kernel = resize_nearest_2x_kernel(&self.ctx);
                    let total = n * c * h * 2 * w * 2;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: resize_nearest_2x launch failed");
                    }
                }
                Step::GaussianSplatRender {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    dst_off,
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
                    #[cfg(feature = "native-splat")]
                    crate::splat_native::run_gaussian_splat_render_native(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                    #[cfg(not(feature = "native-splat"))]
                    crate::splat_host::run_gaussian_splat_render(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
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
                Step::GaussianSplatPrepare {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    meta_len,
                    prep_off,
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
                    crate::splat_host::run_gaussian_splat_prepare(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *prep_off as usize,
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
                Step::GaussianSplatRasterize {
                    prep_off,
                    prep_len,
                    meta_off,
                    meta_len,
                    dst_off,
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
                    crate::splat_host::run_gaussian_splat_rasterize(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *prep_off as usize,
                        *prep_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *dst_off as usize,
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
                Step::GaussianSplatRenderBackward {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    d_loss_off,
                    d_loss_len,
                    packed_off,
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
                    crate::splat_host::run_gaussian_splat_render_backward(
                        &stream,
                        &mut self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *d_loss_off as usize,
                        *d_loss_len as usize,
                        *packed_off as usize,
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
                Step::RmsNormBackwardInput {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dx_byte_off / 4,
                        *eps_bits,
                        0,
                    );
                }
                Step::RmsNormBackwardGamma {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dgamma_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dgamma_byte_off / 4,
                        *eps_bits,
                        1,
                    );
                }
                Step::RmsNormBackwardBeta {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dbeta_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *rows,
                        *h,
                        *x_byte_off / 4,
                        *gamma_byte_off / 4,
                        *beta_byte_off / 4,
                        *dy_byte_off / 4,
                        *dbeta_byte_off / 4,
                        *eps_bits,
                        2,
                    );
                }
                Step::RopeBackward {
                    dy_byte_off,
                    cos_byte_off,
                    sin_byte_off,
                    dx_byte_off,
                    batch,
                    seq,
                    hidden,
                    head_dim,
                    n_rot,
                    cos_len,
                } => {
                    launch_rope_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *batch,
                        *seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
                        *dy_byte_off / 4,
                        *cos_byte_off / 4,
                        *sin_byte_off / 4,
                        *dx_byte_off / 4,
                        *cos_len,
                    );
                }
                Step::CumsumBackward {
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    cols,
                    exclusive,
                } => {
                    launch_cumsum_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *rows,
                        *cols,
                        *dy_byte_off / 4,
                        *dx_byte_off / 4,
                        if *exclusive { 1 } else { 0 },
                    );
                }
                Step::GatherBackward {
                    dy_byte_off,
                    indices_byte_off,
                    dst_byte_off,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                } => {
                    launch_gather_bwd(
                        &self.ctx,
                        &stream,
                        &mut self.arena.buffer,
                        *outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
                        *dy_byte_off / 4,
                        *indices_byte_off / 4,
                        *dst_byte_off / 4,
                    );
                }
                Step::Pool1d {
                    n,
                    c,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool1d_kernel(&self.ctx);
                    let total = n * c * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool1d launch failed");
                    }
                }
                Step::Pool2d {
                    n,
                    c,
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
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool2d_kernel(&self.ctx);
                    let total = n * c * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool2d launch failed");
                    }
                }
                Step::Pool3d {
                    n,
                    c,
                    d,
                    h,
                    w,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool3d_kernel(&self.ctx);
                    let total = n * c * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool3d launch failed");
                    }
                }
                Step::Conv1d {
                    n,
                    c_in,
                    c_out,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    dl,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN — 1-D conv as a degenerate 2-D conv
                    // with H=1, kh=1, sh=1, ph=0, dh=1. Same descriptors
                    // as conv2d; the H axis just collapses to 1.
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_mut())
                    {
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) = self.arena.buffer.device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                /*h*/ 1,
                                *l,
                                /*h_out*/ 1,
                                *l_out,
                                /*kh*/ 1,
                                *kl,
                                /*sh*/ 1,
                                *sl,
                                /*ph*/ 0,
                                *pl,
                                /*dh*/ 1,
                                *dl,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv1d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv1d_kernel(&self.ctx);
                    let total = n * c_out * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(dl)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv1d launch failed");
                    }
                }
                Step::Conv2d {
                    n,
                    c_in,
                    c_out,
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
                    dw,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN — picks the fastest algo via the v7
                    // heuristic for the supplied shape + workspace size.
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_mut())
                    {
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) = self.arena.buffer.device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
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
                                *dw,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv2d launch failed");
                    }
                }
                Step::Conv3d {
                    n,
                    c_in,
                    c_out,
                    d,
                    h,
                    w,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: cuDNN nd-conv (NCDHW + 3-D pads/strides/dilations).
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_mut())
                    {
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) = self.arena.buffer.device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv3d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *d,
                                *h,
                                *w,
                                *d_out,
                                *h_out,
                                *w_out,
                                *kd,
                                *kh,
                                *kw,
                                *sd,
                                *sh,
                                *sw,
                                *pd,
                                *ph,
                                *pw,
                                *dd,
                                *dh,
                                *dw,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv3d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(&mut self.arena.buffer)
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(dd)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv3d launch failed");
                    }
                }
            }

            // Multi-stream tail: record an event so future steps can
            // wait on this one, then update producer_of with the
            // offsets this step wrote.
            if let Some(idx) = assigned_idx {
                if let Ok(evt) = stream.record_event(None) {
                    last_event.insert(idx, evt);
                }
                let (_, writes) = step_offsets(step);
                for w in &writes {
                    producer_of.insert(*w, idx);
                }
            }
        }

        // Multi-stream: sync every pool stream so output reads see all
        // produced data.
        if multi_stream {
            for s in &self.streams {
                let _ = s.synchronize();
            }
        }

        if do_capture {
            let cu_graph = stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags
                    ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
            ).expect("rlx-cuda: end_capture failed");
            if let Some(g) = cu_graph {
                g.upload().expect("rlx-cuda: graph upload failed");
                g.launch().expect("rlx-cuda: graph first launch failed");
                self.captured_graph = Some(g);
            }
        }

        stream.synchronize().expect("rlx-cuda: stream sync failed");
        self.read_outputs(&stream)
    }

    fn read_outputs(&self, stream: &Arc<cudarc::driver::CudaStream>) -> Vec<Vec<f32>> {
        self.graph
            .outputs
            .iter()
            .map(|&id| {
                let off_f32 = self.arena.offset(id) / 4;
                let elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                let slot = self.arena.buffer.slice(off_f32..off_f32 + elems);
                stream
                    .clone_dtoh(&slot)
                    .expect("rlx-cuda: output download failed")
                    .to_vec()
            })
            .collect()
    }
}

fn launch_cumsum_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    inner: u32,
    dy_off: u32,
    dx_off: u32,
    exclusive: u32,
) {
    let kernel = cumsum_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(outer, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&outer)
        .arg(&inner)
        .arg(&dy_off)
        .arg(&dx_off)
        .arg(&exclusive);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: cumsum_bwd launch failed");
    }
}

fn launch_rope_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dy_off: u32,
    cos_off: u32,
    sin_off: u32,
    dx_off: u32,
    cos_len: u32,
) {
    let total = batch * seq * hidden;
    let kernel = rope_backward_kernel(ctx);
    let (grid, block) = dispatch_grid_1d(total, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(buffer)
        .arg(&batch)
        .arg(&seq)
        .arg(&hidden)
        .arg(&head_dim)
        .arg(&n_rot)
        .arg(&dy_off)
        .arg(&cos_off)
        .arg(&sin_off)
        .arg(&dx_off)
        .arg(&cos_len);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rope_bwd launch failed");
    }
}

fn launch_gather_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    dy_off: u32,
    idx_off: u32,
    dst_off: u32,
) {
    let total = outer * axis_dim * trailing;
    if total > 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(total, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&dst_off).arg(&total);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: gather_bwd zero launch failed");
        }
    }
    let kernel = gather_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (outer, (num_idx * trailing).div_ceil(256), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&outer)
        .arg(&axis_dim)
        .arg(&num_idx)
        .arg(&trailing)
        .arg(&dy_off)
        .arg(&idx_off)
        .arg(&dst_off);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: gather_bwd launch failed");
    }
}

fn launch_rms_norm_bwd(
    ctx: &Arc<CudaContext>,
    stream: &cudarc::driver::CudaStream,
    buffer: &mut cudarc::driver::CudaSlice<f32>,
    rows: u32,
    inner: u32,
    x_off: u32,
    gamma_off: u32,
    beta_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    wrt: u32,
) {
    if wrt != 0 {
        let zk = rms_norm_bwd_zero_kernel(ctx);
        let (grid, block) = dispatch_grid_1d(inner, 256);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut zl = stream.launch_builder(&zk.function);
        zl.arg(&mut *buffer).arg(&out_off).arg(&inner);
        unsafe {
            zl.launch(cfg)
                .expect("rlx-cuda: rms_norm_bwd zero launch failed");
        }
    }
    let kernel = rms_norm_backward_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&rows)
        .arg(&inner)
        .arg(&x_off)
        .arg(&gamma_off)
        .arg(&beta_off)
        .arg(&dy_off)
        .arg(&out_off)
        .arg(&eps_bits)
        .arg(&wrt);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: rms_norm_bwd launch failed");
    }
}

#[cfg(test)]
mod tests {
    //! Pure-function tests for the multi-stream scheduler analysis and
    //! the element-wise fusion pass. Both are pure Rust against
    //! synthesized `Vec<Step>` inputs — no CUDA driver needed, so they
    //! run on Mac.
    use super::*;

    #[test]
    fn step_offsets_binary() {
        let s = Step::Binary {
            n: 8,
            a_off: 100,
            b_off: 200,
            c_off: 300,
            op: 0,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![100, 200]);
        assert_eq!(w, vec![300]);
    }

    #[test]
    fn step_offsets_matmul_with_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 1,
            bias_off_f32: 40,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20, 40]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_matmul_no_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 0,
            bias_off_f32: 0,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_attention_causal_no_mask_arg() {
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 1, // causal — mask_off ignored
            scale_bits: 0,
            window: 0,
        };
        let (r, _) = step_offsets(&s);
        assert!(!r.contains(&9999), "causal mask must not consume mask_off");
        assert_eq!(r, vec![0, 100, 200]);
    }

    #[test]
    fn step_offsets_attention_custom_mask_pulls_mask() {
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 2, // custom mask
            scale_bits: 0,
            window: 0,
        };
        let (r, _) = step_offsets(&s);
        assert!(r.contains(&9999));
    }

    #[test]
    fn step_offsets_scatter_add_acc_marks_out_as_rmw() {
        let s = Step::ScatterAddAcc {
            out_off: 100,
            upd_off: 200,
            idx_off: 300,
            num_updates: 4,
            trailing: 1,
            out_dim: 16,
        };
        let (r, w) = step_offsets(&s);
        // out is read-modify-write, so it appears in BOTH reads and writes
        // — this lets the multi-stream scheduler force the prior
        // ScatterAddZero to complete before the accumulate launches.
        assert!(r.contains(&100));
        assert!(w.contains(&100));
    }

    #[test]
    fn fuse_elementwise_merges_binary_then_unary() {
        let schedule = vec![
            // c = a + b
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            // d = relu(c)
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 1, "expected exactly one fused step");
        match &fused[0] {
            Step::FusedBinaryUnary {
                n,
                a_off,
                b_off,
                out_off,
                bin_op,
                un_op,
            } => {
                assert_eq!(*n, 4);
                assert_eq!(*a_off, 0);
                assert_eq!(*b_off, 4);
                assert_eq!(*out_off, 12);
                assert_eq!(*bin_op, 0);
                assert_eq!(*un_op, 0);
            }
            other => panic!("expected FusedBinaryUnary, got {}", step_name(other)),
        }
    }

    #[test]
    fn fuse_elementwise_skips_when_intermediate_has_two_consumers() {
        // c = a + b
        // d = relu(c)
        // e = c * c   ← second consumer of c, blocks fusion
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 8,
                b_off: 8,
                c_off: 16,
                op: 2,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 3, "no fusion: c has multiple consumers");
        assert!(matches!(&fused[0], Step::Binary { .. }));
        assert!(matches!(&fused[1], Step::Unary { .. }));
        assert!(matches!(&fused[2], Step::Binary { .. }));
    }

    #[test]
    fn fuse_elementwise_skips_when_n_mismatch() {
        // Different element counts → can't fuse (different launch grid).
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 8,
                in_off: 8,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_skips_when_unary_input_isnt_binary_output() {
        // Unary reads a different offset than what Binary wrote.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 99,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_handles_multiple_chains() {
        // Two independent Binary→Unary chains in a row — both should fuse.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 16,
                b_off: 20,
                c_off: 24,
                op: 2,
            },
            Step::Unary {
                n: 4,
                in_off: 24,
                out_off: 28,
                op: 9,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
        assert!(matches!(&fused[0], Step::FusedBinaryUnary { .. }));
        assert!(matches!(&fused[1], Step::FusedBinaryUnary { .. }));
    }
}
