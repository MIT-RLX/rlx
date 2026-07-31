// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pre-compiled command list — analog of rlx-cpu's Thunk.

const ARENA_LARGE_OFF: usize = 1usize << 32;

/// High bit marks an offset into [`MetalExecutable`]'s separate weight MTLBuffer
/// (params kept out of the activation arena so it stays under the 4 GiB MPS cliff).
pub(crate) const WEIGHT_BUF_TAG: usize = 1usize << 63;

#[inline]
pub(crate) fn tag_weight_off(off: usize) -> usize {
    debug_assert_eq!(off & WEIGHT_BUF_TAG, 0, "weight offset already tagged");
    off | WEIGHT_BUF_TAG
}

#[inline]
pub(crate) fn is_weight_off(off: usize) -> bool {
    off & WEIGHT_BUF_TAG != 0
}

#[inline]
pub(crate) fn raw_off(off: usize) -> usize {
    off & !WEIGHT_BUF_TAG
}

#[inline]
fn arena_off_large(off: usize) -> bool {
    !is_weight_off(off) && raw_off(off) >= ARENA_LARGE_OFF
}

#[inline]
fn metal_host_fallback_enabled() -> bool {
    matches!(
        std::env::var("RLX_METAL_HOST_SLICE").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    ) || rlx_ir::env::flag("RLX_METAL_HOST_FALLBACK")
}

/// Numpy-style broadcast strides for `in_dims` into the row-major
/// output of `out_dims`. Returns a length-`out_dims.len()` vector
/// where entry `d` is `0` if the input is size-1 (broadcast) at output
/// dim `d` (after left-padding with size-1 to match ranks), otherwise
/// the natural row-major stride into the input buffer.
fn broadcast_strides(in_dims: &[usize], out_dims: &[usize]) -> Vec<u32> {
    let r_out = out_dims.len();
    let r_in = in_dims.len();
    debug_assert!(r_in <= r_out, "broadcast in rank {r_in} > out rank {r_out}");
    let pad = r_out - r_in;
    let mut strides = vec![0u32; r_out];
    let mut acc: usize = 1;
    for d in (0..r_out).rev() {
        let in_size = if d < pad { 1 } else { in_dims[d - pad] };
        if in_size == 1 {
            strides[d] = 0;
        } else {
            debug_assert_eq!(
                in_size, out_dims[d],
                "broadcast: dim {in_size} vs out {} at {d}",
                out_dims[d]
            );
            strides[d] = acc as u32;
            acc *= in_size;
        }
    }
    strides
}

/// Pack leading-dim broadcast metadata for DiT modulation kernels.
/// Layout: `[lead_rank, x_lead[0..8], mod_lead[0..8]]`.
fn ada_lead_pack(x_dims: &[usize], mod_dims_in: &[usize]) -> [u32; 17] {
    rlx_ir::ada_modulation_lead_pack(x_dims, mod_dims_in)
}

/// Launch geometry for DiT modulation backward: one threadgroup per unique
/// modulation row; that TG loops `seq_per_mod` feature-rows that share it.
fn ada_mod_launch(x_dims: &[usize], mod_dims: &[usize]) -> (u32, u32) {
    debug_assert!(!x_dims.is_empty() && !mod_dims.is_empty());
    let xr = x_dims.len() - 1;
    let mr = mod_dims.len() - 1;
    let mut seq = 1u32;
    let mut mods = 1u32;
    for i in 0..xr {
        let xd = x_dims[i] as u32;
        let md = if i + mr >= xr {
            mod_dims[i - (xr - mr)] as u32
        } else {
            1
        };
        if md == 1 && xd > 1 {
            seq = seq.saturating_mul(xd);
        } else {
            mods = mods.saturating_mul(xd.max(1));
        }
    }
    (mods.max(1), seq.max(1))
}

/// True when the rhs is a *true* trailing broadcast of the lhs — i.e.
/// every rhs dim matches the corresponding lhs dim counting from the
/// right (no size-1 broadcasts *inside* the rhs). This is the only
/// case the cheap `BiasAdd` thunk is correct for. Mid-shape singletons
/// (e.g. SAM rel_pos `[bh, h, w, h, 1]` against `[bh, h, w, h, w]`)
/// are NOT trailing broadcasts and must go through `BinaryBroadcast`.
fn trailing_broadcast(lhs: &Shape, rhs: &Shape) -> bool {
    if rhs.rank() > lhs.rank() {
        return false;
    }
    let off = lhs.rank() - rhs.rank();
    for i in 0..rhs.rank() {
        let r = rhs.dim(i).unwrap_static();
        let l = lhs.dim(off + i).unwrap_static();
        if r != l {
            return false;
        }
    }
    true
}
use crate::op_registry::{MetalGpuKernel, MetalKernel};
use rlx_ir::op::{Activation, BinaryOp, CmpOp};
use rlx_ir::{DType, Shape};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfFlag {
    F32,
    F16,
}

impl From<DType> for HalfFlag {
    fn from(d: DType) -> Self {
        match d {
            DType::F16 => HalfFlag::F16,
            _ => HalfFlag::F32,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Thunk {
    Nop,
    /// Cast between f32 and f16 (same element count, dtype change).
    Cast {
        src: usize,
        dst: usize,
        len: u32,
        src_dt: HalfFlag,
        dst_dt: HalfFlag,
    },
    /// Integer / bool / mixed-width cast. Metal's f32/f16 kernels treat every
    /// non-F16 dtype as F32 (`HalfFlag`), which densifies float bytes into an
    /// I64 arena slot — ScatterElements then reads float pairs as garbage
    /// indices. This host path truncates / widens using the real dtypes.
    CastHost {
        src: usize,
        dst: usize,
        len: u32,
        src_dt: DType,
        dst_dt: DType,
    },
    /// Truncate F32 toward zero into an F32 slot. Used when Metal widens
    /// `Cast(f32→i64)` arenas to F32 but must keep ONNX truncation semantics
    /// (Vocos fringe mask / integer Gather indices as floats).
    CastTruncF32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    Sgemm {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
        dt: HalfFlag,
        /// RHS weight matrix stored as F16; promote to F32 via scratch before sgemm.
        b_f16: bool,
    },
    /// Batched f32 matmul — per-batch independent `Sgemm`. Used for 3-D
    /// `[batch, M, K] @ [batch, K, N]` where both operands carry a batch
    /// dim. The plain `Sgemm` flattens to 2-D (M·batch, K, N) which is
    /// only correct when the RHS has *no* batch dim. SAM's decomposed
    /// attention hits this and silently produces garbage (cascades to
    /// NaN) without this dedicated path.
    BatchedSgemm {
        a: usize,
        b: usize,
        c: usize,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
        dt: HalfFlag,
        /// Operand `a`/`b` has batch dim 1 and is BROADCAST across the output
        /// batch (its per-matrix stride is 0 — reuse matrix 0).
        a_bcast: bool,
        b_bcast: bool,
    },
    FusedMmBiasAct {
        a: usize,
        w: usize,
        bias: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
        act: Option<Activation>,
        dt: HalfFlag,
    },
    ActivationInPlace {
        data: usize,
        len: u32,
        act: Activation,
        dt: HalfFlag,
    },
    /// Out-of-place activation: `dst[i] = act(src[i])` (avoids Copy+InPlace).
    ActivationOut {
        src: usize,
        dst: usize,
        len: u32,
        act: Activation,
        dt: HalfFlag,
    },
    /// Out-of-place tanh-approx GELU: `dst[i] = gelu_approx(src[i])`.
    GeluApproxOut {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// CPU unified-memory gelu_approx(src→dst) for >4 GiB arenas (fused copy+act).
    GeluApproxHost {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// Fused `act(lhs op rhs)` in one dispatch (region lowering).
    FusedBinaryActivation {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: BinaryOp,
        act: Activation,
        dt: HalfFlag,
    },
    /// Fused `act((lhs op0 rhs0) op1 rhs1)` in one dispatch (region lowering).
    FusedTernaryActivation {
        lhs: usize,
        rhs0: usize,
        rhs1: usize,
        dst: usize,
        len: u32,
        op0: BinaryOp,
        op1: BinaryOp,
        act: Activation,
        dt: HalfFlag,
    },
    LayerNorm {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
        dt: HalfFlag,
    },
    /// NCHW group norm.
    GroupNorm {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps: f32,
        dt: HalfFlag,
    },
    /// NCHW LayerNorm2d (normalize across C at each spatial position).
    LayerNorm2d {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps: f32,
        dt: HalfFlag,
    },
    /// NCHW ConvTranspose2d (PyTorch layout, no bias).
    ConvTranspose2d {
        src: usize,
        weight: usize,
        dst: usize,
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
        dt: HalfFlag,
    },
    /// NCDHW Conv3d (PyTorch layout, no bias).
    /// Weight: `[C_out, C_in/groups, kD, kH, kW]`.
    Conv3d {
        src: usize,
        weight: usize,
        dst: usize,
        n: u32,
        c_in: u32,
        d: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
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
        dt: HalfFlag,
    },
    /// NCDHW ConvTranspose3d (PyTorch layout, no bias).
    /// Weight: `[C_in, C_out/groups, kD, kH, kW]`.
    ConvTranspose3d {
        src: usize,
        weight: usize,
        dst: usize,
        n: u32,
        c_in: u32,
        d: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
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
        dt: HalfFlag,
    },
    /// Nearest 2× upsample on NCHW.
    ResizeNearest2x {
        src: usize,
        dst: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        dt: HalfFlag,
    },
    /// RMSNorm: variance-only normalization. See CPU's Thunk::RmsNorm.
    RmsNorm {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
        dt: HalfFlag,
    },
    BinaryFull {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: BinaryOp,
        dt: HalfFlag,
    },
    /// Shape-aware broadcast binary op. Handles arbitrary-rank
    /// broadcasts including mid-shape singletons (e.g. `[bh, h, w, 1, w]
    /// + [bh, h, w, h, w]` from SAM's decomposed rel-pos). The legacy
    /// `BiasAdd`/`BinaryFull` only handle trailing-singleton/exact-size
    /// cases — anything else silently aliased to the wrong stride.
    BinaryBroadcast {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: BinaryOp,
        dt: HalfFlag,
        rank: u32,
        /// Output dims (length = rank). Stored inline as u32; SAM rel-pos
        /// uses rank ≤ 5.
        out_dims: Vec<u32>,
        /// Per-axis input strides (0 ⇒ broadcast / replicate).
        lhs_strides: Vec<u32>,
        rhs_strides: Vec<u32>,
    },
    BiasAdd {
        src: usize,
        bias: usize,
        dst: usize,
        m: u32,
        n: u32,
        dt: HalfFlag,
    },
    /// out = LN(x + residual + bias, gamma, beta) (bias=0 means no-bias variant)
    FusedResidualLN {
        x: usize,
        res: usize,
        bias: usize,
        g: usize,
        b: usize,
        out: usize,
        rows: u32,
        h: u32,
        eps: f32,
        has_bias: bool,
        dt: HalfFlag,
    },
    /// out = RmsNorm(x + residual + bias, gamma, beta)
    FusedResidualRmsNorm {
        x: usize,
        res: usize,
        bias: usize,
        g: usize,
        b: usize,
        out: usize,
        rows: u32,
        h: u32,
        eps: f32,
        has_bias: bool,
        dt: HalfFlag,
    },
    /// DiT adaLN-Zero: `out = norm(x)·(1+scale)+shift` with broadcast scale/shift.
    /// `lead_pack`: `[lead_rank, x_lead[8], mod_lead[8]]` (17 u32s).
    AdaLayerNorm {
        x: usize,
        scale: usize,
        shift: usize,
        out: usize,
        rows: u32,
        h: u32,
        eps: f32,
        layer_norm: bool,
        lead_pack: [u32; 17],
        dt: HalfFlag,
    },
    /// DiT gated residual: `out = x + gate·y` with broadcast gate.
    /// `lead_pack`: same layout as [`Thunk::AdaLayerNorm`].
    GatedResidual {
        x: usize,
        y: usize,
        gate: usize,
        out: usize,
        rows: u32,
        h: u32,
        lead_pack: [u32; 17],
        dt: HalfFlag,
    },
    /// Packed `[dx ∥ dscale ∥ dshift]` — see [`Op::AdaLayerNormBackward`].
    AdaLayerNormBackward {
        x: usize,
        scale: usize,
        dy: usize,
        out: usize,
        h: u32,
        eps: f32,
        layer_norm: bool,
        seq_per_mod: u32,
        mod_rows: u32,
        dt: HalfFlag,
    },
    /// Packed `[dx ∥ dy ∥ dgate]` — see [`Op::GatedResidualBackward`].
    GatedResidualBackward {
        y: usize,
        gate: usize,
        dy: usize,
        out: usize,
        h: u32,
        seq_per_mod: u32,
        mod_rows: u32,
        dt: HalfFlag,
    },
    /// GDN gated norm: `out = rms_norm(x, g, b) * silu(z)` (same shape).
    /// Pattern-merged from `silu(z) → rms_norm(scan) → mul` by `fuse_gdn_gated_norm`.
    FusedRmsNormMulSilu {
        x: usize,
        g: usize,
        b: usize,
        z: usize,
        out: usize,
        rows: u32,
        h: u32,
        eps: f32,
        dt: HalfFlag,
    },
    /// Depthwise 1-D conv on BSC `[B,W,C]` → `[B,out_seq,C]` (+ optional SiLU).
    /// Pattern-merged from Transpose→Copy→Conv2D→Copy→Transpose(+Silu) by
    /// `fuse_depthwise_conv1d_bsc` (GDN `ssm_conv1d`).
    FusedDepthwiseConv1dBsc {
        src: usize,
        weight: usize,
        dst: usize,
        batch: u32,
        width: u32,
        out_seq: u32,
        channels: u32,
        k: u32,
        silu: bool,
    },
    /// Gather along axis 0 (embedding lookup)
    Gather {
        table: usize,
        idx: usize,
        dst: usize,
        num_idx: u32,
        trailing: u32,
        dt: HalfFlag,
    },
    /// Narrow along last axis
    Narrow {
        src: usize,
        dst: usize,
        outer: u32,
        src_axis: u32,
        start: u32,
        len: u32,
        dt: HalfFlag,
    },
    /// Fused concat-VJP / multi-slice: one dispatch for many last-axis narrows
    /// from the same source buffer (see `fuse_narrow_clusters` in `compile_thunks`).
    SplitLastAxis {
        src: usize,
        outer: u32,
        src_axis: u32,
        dt: HalfFlag,
        segments: Vec<(usize, u32, u32)>,
    },
    /// Reshape / Cast / Expand: copy len elements
    Copy {
        src: usize,
        dst: usize,
        len: u32,
        dt: HalfFlag,
    },
    /// SDPA. `mask_kind` encodes how to apply masking inside the
    /// kernel:
    ///   0 = None           (no masking)
    ///   1 = Causal         (prefill: upper-triangular fill in-kernel)
    ///   2 = Custom         (read binary mask buffer `mask`)
    ///   3 = Bias           (additive per-head bias; `sdpa_long`-only)
    ///   4 = SlidingWindow  (causal + lookback `window`; lookback uses
    ///                       absolute positions so decode mode with
    ///                       cached K/V works correctly)
    Attention {
        q: usize,
        k: usize,
        v: usize,
        mask: usize,
        out: usize,
        batch: u32,
        seq: u32,    // query length (Lq)
        kv_seq: u32, // key/value length (Lk); == seq for self-attn
        heads: u32,
        /// Key/value head count. Equal to `heads` for MHA; smaller for GQA
        /// (K/V packed as `kv_heads * head_dim` without host-side repeat).
        kv_heads: u32,
        head_dim: u32,
        mask_kind: u32,
        /// Lookback distance for `mask_kind == 4` (SlidingWindow); 0
        /// for every other mask kind. Visible range per query at
        /// absolute position `abs_q` is `[abs_q - window, abs_q]`.
        window: u32,
        dt: HalfFlag,
        /// 1 iff Q/K/V are `[B, H, S, D]` (dim1 == num_heads).
        bhsd: u32,
        /// Op::Attention.score_scale. Sentinel `0.0` means "use the kernel default
        /// `1/sqrt(head_dim)`". Gemma 4 sets this to `1.0` because Q is per-head
        /// RMS-normed before attention — applying `1/sqrt(d)` on top crushes the
        /// scores 16× (head_dim=256) or 22× (head_dim=512).
        score_scale: f32,
        /// Op::Attention.attn_logit_softcap. Sentinel `0.0` disables. Gemma 2 uses
        /// 50.0, Gemma 4 has no attn softcap (handled at final logits).
        attn_logit_softcap: f32,
    },
    /// Native fused-attention core (`fused_attn_block` MSL kernel): inline
    /// NeoX RoPE + softmax SDPA over the packed QKV scratch `[B,S,3*inner]`
    /// → attn scratch `[B,S,inner]`. One threadgroup per (batch·head); score
    /// matrix in threadgroup memory. The QKV / out projections are separate
    /// `Sgemm` thunks emitted by the same `Op::FusedAttentionBlock` arm.
    /// `qkv` / `out` are byte offsets into the FAB scratch region. F32 only.
    FusedAttn {
        qkv: usize,
        mask: usize,
        cos: usize,
        sin: usize,
        out: usize,
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        mask_kind: u32,
        scale_bits: u32,
        has_rope: u32,
    },
    /// [`Op::AttentionBackward`] — GPU MSL when scratch fits; CPU fallback otherwise.
    /// on unified-memory arena (F32 only).
    AttentionBackward {
        q: usize,
        k: usize,
        v: usize,
        dy: usize,
        mask: usize,
        out: usize,
        batch: u32,
        seq: u32,
        kv_seq: u32,
        heads: u32,
        head_dim: u32,
        mask_kind: u32,
        window: u32,
        wrt: u32,
        /// 1 iff Q/K/V are `[B, H, S, D]` (dim1 == num_heads).
        bhsd: u32,
    },
    /// RoPE. `src_row_stride` is elements per source row (defaults to
    /// `hidden`); the Narrow→Rope thunk fusion at the end of Metal
    /// `compile_thunks` rewrites it when Rope reads directly from a
    /// wider parent like QKV. Mirrors the CPU change in plan #45.
    Rope {
        src: usize,
        cos: usize,
        sin: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        n_rot: u32,
        dt: HalfFlag,
        src_row_stride: u32,
        /// `true` when the cos/sin tables carry one row per (batch·seq) token
        /// (ragged batched decode) rather than one row per seq position. Drives
        /// per-token RoPE indexing in the kernel; `false` for every prefill /
        /// uniform-decode graph (byte-identical to the prior behavior).
        cos_per_token: bool,
        /// `true` = GPT-J / llama.cpp-NORM interleaved pairs `(2i, 2i+1)`;
        /// `false` = HF / NeoX rotate-half pairs `(i, i+n_rot/2)`. GGUF Llama
        /// weights need the interleaved flavor.
        interleaved: bool,
    },
    /// Softmax
    Softmax {
        data: usize,
        rows: u32,
        cols: u32,
        dt: HalfFlag,
    },
    /// Fused dense / soft-label softmax cross-entropy. `logits [N,C]`,
    /// `targets [N,C]` → per-row loss `[N]`. f32 only (matches the rlx
    /// SCE contract). One threadgroup per row.
    SoftmaxCrossEntropyDense {
        logits: usize,
        targets: usize,
        dst: usize,
        n: u32,
        c: u32,
    },
    /// Softmax cross-entropy with integer labels (forward).
    /// `loss[row] = logsumexp(logits[row]) - logits[row, label]`.
    SoftmaxCrossEntropyWithLogits {
        logits: usize,
        labels: usize,
        dst: usize,
        n: u32,
        c: u32,
    },
    /// Softmax cross-entropy backward (integer labels).
    /// `dlogits[row,k] = (softmax(logits[row])[k] - [k==label]) * d_loss[row]`.
    SoftmaxCrossEntropyBackward {
        logits: usize,
        labels: usize,
        d_loss: usize,
        dlogits: usize,
        n: u32,
        c: u32,
    },
    /// Inclusive (or exclusive) cumulative sum along the last axis.
    Cumsum {
        src: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    /// Native cumulative product / maximum (`is_max` selects max) along the
    /// last axis — mirrors [`Thunk::Cumsum`].
    CumScan {
        src: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        exclusive: bool,
        is_max: bool,
    },
    /// Fused SwiGLU: `out[r,i] = x[r,i] * silu(x[r, n_half+i])`.
    /// Optional output cast: when `cast_to != src_dt` the kernel writes
    /// the result in `cast_to` precision; otherwise plain f32/f16 path.
    FusedSwiGLU {
        src: usize,
        dst: usize,
        n_half: u32,
        total: u32,
        src_dt: HalfFlag,
        dst_dt: HalfFlag,
        gate_first: bool,
    },
    /// Concat along last axis: dispatches one segment kernel per input.
    /// Each entry in `inputs` is (src_offset, axis_len_for_that_input).
    Concat {
        dst: usize,
        outer: u32,
        dst_axis: u32,
        /// Trailing-dim product (= 1 for last-axis concat, > 1 for
        /// mid-shape concat). The kernel reads/writes `inner` elements
        /// per (outer, axis-slot) pair.
        inner: u32,
        dt: HalfFlag,
        inputs: Vec<(usize, u32)>,
    },
    /// Element-wise comparison: out = (lhs CMP rhs) ? 1.0 : 0.0.
    /// `lhs_scalar` / `rhs_scalar` broadcast a 1-element operand across `len`
    /// (ONNX Less/Greater against a scalar threshold — F5 text-embed clamp).
    Compare {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: CmpOp,
        lhs_scalar: bool,
        rhs_scalar: bool,
    },
    /// Reduce over a contiguous axis range. See CPU's Thunk::Reduce.
    Reduce {
        src: usize,
        dst: usize,
        outer: u32,
        reduced: u32,
        inner: u32,
        op: rlx_ir::op::ReduceOp,
        dt: HalfFlag,
    },
    /// Top-K indices along last axis. See CPU's Thunk::TopK.
    TopK {
        src: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        k: u32,
    },
    /// Indexed batched matmul (MoE GEMM). See CPU's Thunk::GroupedMatMul.
    GroupedMatMul {
        input: usize,
        weight: usize,
        expert_idx: usize,
        dst: usize,
        m: u32,
        k_dim: u32,
        n: u32,
        num_experts: u32,
    },
    /// GGUF packed expert stack + grouped matmul.
    DequantGroupedMatMulGguf {
        input: usize,
        w_q: usize,
        expert_idx: usize,
        dst: usize,
        m: u32,
        k_dim: u32,
        n: u32,
        num_experts: u32,
        scheme: rlx_ir::quant::QuantScheme,
    },
    /// MLX-affine MoE grouped matmul (separate scales/biases). Host-delegated
    /// like [`Thunk::DequantMatMulMlx`] (no native Metal kernel yet).
    DequantGroupedMatMulMlx {
        input: usize,
        w_q: usize,
        scale: usize,
        zp: usize,
        expert_idx: usize,
        dst: usize,
        m: u32,
        k_dim: u32,
        n: u32,
        num_experts: u32,
        slab_bytes: u32,
        scheme: rlx_ir::quant::QuantScheme,
        /// Scales/biases stored BF16 (2B) — decode per-expert (matches CPU).
        scale_bf16: bool,
    },
    /// Scatter-add. See CPU's Thunk::ScatterAdd.
    ScatterAdd {
        updates: usize,
        indices: usize,
        dst: usize,
        num_updates: u32,
        out_dim: u32,
        trailing: u32,
    },
    /// General N-D transpose / broadcast. Stride 0 in `in_strides` means
    /// broadcast (read the same input element repeatedly).
    Transpose {
        src: usize,
        dst: usize,
        total: u32,
        out_dims: Vec<u32>,
        in_strides: Vec<u32>,
    },
    /// Gather along arbitrary axis. See CPU's Thunk::GatherAxis.
    GatherAxis {
        table: usize,
        idx: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    /// 2D pooling. See CPU's Thunk::Pool2D.
    Pool2D {
        src: usize,
        dst: usize,
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
        kind: rlx_ir::op::ReduceOp,
    },
    /// 2D convolution. See CPU's Thunk::Conv2D.
    Conv2D {
        src: usize,
        weight: usize,
        dst: usize,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
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
    /// Ternary select: out = cond != 0 ? on_true : on_false.
    /// `*_scalar` flags broadcast a 1-element operand (ONNX Where with a
    /// scalar fill — F5 clamps OOB rotary indices to `max_pos-1`).
    Where {
        cond: usize,
        on_true: usize,
        on_false: usize,
        dst: usize,
        len: u32,
        cond_scalar: bool,
        true_scalar: bool,
        false_scalar: bool,
    },
    Fma {
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
        len: u32,
    },
    /// Native MSL `Op::ReluBackward`: `dx[i] = (x[i] > 0) ? dy[i] : 0`.
    ReluBackward {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
    },
    /// Native MSL `Op::ActivationBackward` for Fixed kinds (op id 0–16).
    ActivationBackward {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
        op: u32,
    },
    /// Native MSL `Op::FakeQuantize` Fixed (scale input).
    FakeQuantizeFixed {
        src: usize,
        scale: usize,
        dst: usize,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max: f32,
    },
    /// Native MSL `Op::FakeQuantize` PerBatch (derive scale from max abs).
    FakeQuantizePerBatch {
        src: usize,
        dst: usize,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max: f32,
    },
    /// Native MSL `|z|² = re² + im²` (C64 → F32). `len` is the complex-element
    /// count; `src` is interleaved `[re, im]` pairs.
    ComplexNormSq {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// Native MSL Wirtinger VJP of ComplexNormSq: `dz = g · z` (C64, F32 → C64).
    ComplexNormSqBackward {
        z: usize,
        g: usize,
        dz: usize,
        len: u32,
    },
    /// Native MSL element-wise C64 conjugate: `(re, -im)`.
    ConjugateC64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// Native MSL ternary-pruned radix-2 butterfly stage
    /// (`fft_butterfly_stage`). State/out are interleaved `[batch, n_fft, 2]`.
    FftButterflyStage {
        state: usize,
        out: usize,
        gate: usize,
        rev: usize,
        tw_re: usize,
        tw_im: usize,
        batch: u32,
        n_fft: u32,
        stage: u32,
    },
    /// PLAN L2 — fused N-ary element-wise region. Lowered from
    /// `Op::ElementwiseRegion`. Kernel interprets the chain encoding
    /// per-element (saves N kernel dispatches + N global-memory
    /// round-trips vs the decomposed atomic ops).
    ElementwiseRegion {
        len: u32,
        num_inputs: u32,
        num_steps: u32,
        dst: usize,
        input_offs: [u32; 16],
        chain: [u32; 128], // 32 steps * 4 u32s
        /// PLAN L2 quality: per-input scalar-broadcast bitfield
        /// (fast path). Bit `i` set ⇒ input `i` is a scalar.
        scalar_input_mask: u32,
        /// PLAN L2 quality: per-input element count for trailing-shape
        /// broadcast. `0` ⇒ no broadcast; `>0` ⇒ kernel reads
        /// `arena[input_offs[i] + (gid % input_modulus[i])]`.
        input_modulus: [u32; 16],
        /// FKL closed region prologue (0=none, 1=resize nearest 2x NCHW).
        prologue: u32,
        out_n: u32,
        out_c: u32,
        out_h: u32,
        out_w: u32,
        /// External input index for prologue transform source (default 0).
        prologue_input: u32,
    },
    /// FKL batch horizontal fusion: one launch over N slice chains (no prologue).
    BatchElementwiseRegion {
        slice_len: u32,
        num_batch: u32,
        num_steps: u32,
        base_dst: usize,
        slice_elems: u32,
        batch_input_offs: [u32; 64],
        chain: [u32; 128],
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
    },
    /// Stateful gated-DeltaNet scan. Native MSL kernel (`gated_delta_net`);
    /// host fallback when `RLX_METAL_GDN_HOST_FALLBACK=1`, f16 tensors,
    /// or n > 128.
    GatedDeltaNet {
        q: usize,
        k: usize,
        v: usize,
        g: usize,
        beta: usize,
        state: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
        f16: bool,
        /// 1 = per-channel gate (`g` is `[b,s,h,n]`, Kimi-K3 KDA).
        gate_per_channel: bool,
        /// Resume from / write back the external `state` (decode). Explicit flag,
        /// not `state != 0` — arena offset 0 is a valid state slot.
        carry_state: bool,
    },
    /// Mamba selective scan. Native MSL kernel (`selective_scan`) for f32
    /// with `state_size ≤ 128`; host fallback otherwise.
    SelectiveScan {
        x: usize,
        delta: usize,
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        hidden: u32,
        state_size: u32,
    },
    /// Logit sampling (top-k / top-p / temperature). Host fallback over the
    /// unified-memory arena — runs after logits, never the bottleneck.
    Sample {
        logits: usize,
        dst: usize,
        batch: u32,
        vocab: u32,
        top_k: u32,
        top_p: f32,
        temperature: f32,
        seed: u64,
    },
    /// Batch-general reverse/flip. Host fallback over the unified-memory arena.
    Reverse {
        src: usize,
        dst: usize,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u8,
    },
    /// Constant/reflect/replicate/circular pad. Output-indexed gather over the
    /// shared arena (host fallback like [`Thunk::Reverse`]); `fill` is the
    /// constant value pre-encoded in the output dtype.
    Pad {
        src: usize,
        dst: usize,
        in_dims: Vec<u32>,
        before: Vec<u32>,
        after: Vec<u32>,
        mode: rlx_ir::PadMode,
        fill: Vec<u8>,
        elem_bytes: u8,
    },
    /// Strided slice `out[..,j,..] = in[.., start + j*step, ..]` along `axis`.
    /// Host fallback over the shared arena (like [`Thunk::Reverse`]).
    Slice {
        src: usize,
        dst: usize,
        in_dims: Vec<u32>,
        axis: u32,
        start: u32,
        len: u32,
        step: i64,
        elem_bytes: u8,
    },
    /// ArgMax/ArgMin (f32-encoded indices). Host fallback over unified memory.
    ArgReduce {
        src: usize,
        dst: usize,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
    },
    /// Multi-layer (optionally bidirectional/carry) LSTM. Native MSL for
    /// the simple `L=1, unidir, no-carry` case; host fallback otherwise.
    Lstm {
        x: usize,
        w_ih: usize,
        w_hh: usize,
        bias: usize,
        h0: usize,
        c0: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// GRU. Native MSL (`gru`) for L=1/unidir/no-carry f32; host fallback else.
    Gru {
        x: usize,
        w_ih: usize,
        w_hh: usize,
        b_ih: usize,
        b_hh: usize,
        h0: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// Elman RNN. Native MSL (`rnn`) for L=1/unidir/no-carry; host fallback else.
    Rnn {
        x: usize,
        w_ih: usize,
        w_hh: usize,
        bias: usize,
        h0: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
        relu: bool,
    },
    /// Mamba-2 SSD scan. Native MSL (`mamba2`) for n ≤ 128; host fallback else.
    Mamba2 {
        x: usize,
        dt: usize,
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        state_size: u32,
    },
    /// GGUF K-quant matmul — host fallback dequant + BLAS on unified memory.
    DequantMatMulGguf {
        x: usize,
        w_q: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
        /// Activation `x` stored as f16 (AMP). Q1 SG kernels widen in-register.
        x_f16: bool,
        /// Matmul output stored as f16 (AMP residual stream).
        dst_f16: bool,
    },
    /// Legacy Int8 block matmul — CPU host fallback on unified memory.
    DequantMatMulInt8 {
        x: usize,
        w_q: usize,
        scale: usize,
        zp: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        is_asymmetric: bool,
    },
    /// Legacy Int4 block matmul — CPU host fallback on unified memory.
    DequantMatMulInt4 {
        x: usize,
        w_q: usize,
        scale: usize,
        zp: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        is_asymmetric: bool,
    },
    /// Legacy FP8 matmul — CPU host fallback on unified memory.
    DequantMatMulFp8 {
        x: usize,
        w_q: usize,
        scale: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        e5m2: bool,
    },
    /// NVFP4 (E2M1) block matmul — CPU host fallback on unified memory.
    DequantMatMulNvfp4 {
        x: usize,
        w_q: usize,
        scale: usize,
        global_scale: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
    },
    /// MxFp4x2 two-level residual E2M1 fused decode-matmul. `w_q`=[plane0|plane1]
    /// nibbles, `scale`=[s0|s1] f32 per (k/`group`, n).
    DequantMatMulMxFp4x2 {
        x: usize,
        w_q: usize,
        scale: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        group: u32,
    },
    /// MLX affine / mxfp — host dequant on unified memory (via `rlx-mlx-io`).
    DequantMatMulMlx {
        x: usize,
        w_q: usize,
        scale: usize,
        zp: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
    },
    /// Fused decode-layer MLP: gate_proj + up_proj packed GEMVs + SwiGLU
    /// (`dst[i] = up[i] * silu(gate[i])`). m == 1 only. Pattern-merged from
    /// `gate(DequantMatMul) → up(DequantMatMul) → silu → mul` by
    /// `fuse_decode_mlp`; one dispatch instead of four. `x` is the (already
    /// rms-normed) input row of length `k`; `dst` has `n` (intermediate) cols.
    FusedMlpGateUpSwiGLU {
        x: usize,
        gate_w: usize,
        up_w: usize,
        dst: usize,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
        x_f16: bool,
        dst_f16: bool,
    },
    /// Fused gate+up packed GEMVs + GELU-approx epilogue (`dst[i] = up[i] * gelu(gate[i])`).
    FusedMlpGateUpGelu {
        x: usize,
        gate_w: usize,
        up_w: usize,
        dst: usize,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
    },
    /// Fused decode-layer MLP: down_proj GEMV + residual add
    /// (`dst[j] = res[j] + down(h)[j]`). m == 1, Q4_K / Q5_0 / Q6_K. Pattern-merged
    /// from `down(DequantMatMul) → add(residual)` by `fuse_decode_mlp`; one
    /// dispatch instead of two. `x` is the SwiGLU output of length `k`.
    FusedMlpDownResidual {
        x: usize,
        w: usize,
        res: usize,
        dst: usize,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
        x_f16: bool,
        dst_f16: bool,
        res_f16: bool,
    },
    /// Native low-precision GEMM — CPU host fallback (Apple GPUs have no FP8
    /// matrix HW; this is the honest decode-and-accumulate reference). TN.
    ScaledMatMul {
        lhs: usize,
        rhs: usize,
        lhs_scale: usize,
        rhs_scale: usize,
        bias: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        lhs_fmt: rlx_ir::ScaledFormat,
        rhs_fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
        has_bias: bool,
    },
    /// `Op::ScaledQuantize` host fallback — f32 → packed FP8 codes.
    ScaledQuantize {
        x: usize,
        scale: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
    },
    /// `Op::ScaledDequantize` host fallback — packed FP8 codes → f32.
    ScaledDequantize {
        codes: usize,
        scale: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
    },
    /// `Op::ScaledQuantScale` host fallback — per-tensor/block scale.
    ScaledQuantScale {
        x: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
    },
    /// Training backward ops — host fallback on unified memory (F32).
    RmsNormBackwardInput {
        x: usize,
        gamma: usize,
        beta: usize,
        dy: usize,
        dx: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    RmsNormBackwardGamma {
        x: usize,
        gamma: usize,
        beta: usize,
        dy: usize,
        dgamma: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    RmsNormBackwardBeta {
        x: usize,
        gamma: usize,
        beta: usize,
        dy: usize,
        dbeta: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    LayerNormBackwardInput {
        x: usize,
        gamma: usize,
        dy: usize,
        dx: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    LayerNormBackwardGamma {
        x: usize,
        dy: usize,
        dgamma: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    GroupNormBackwardInput {
        x: usize,
        gamma: usize,
        beta: usize,
        dy: usize,
        dx: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps: f32,
    },
    GroupNormBackwardGamma {
        x: usize,
        dy: usize,
        dgamma: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps: f32,
    },
    GroupNormBackwardBeta {
        dy: usize,
        dbeta: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    RopeBackward {
        dy: usize,
        cos: usize,
        sin: usize,
        dx: usize,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        n_rot: u32,
        cos_len: u32,
    },
    CumsumBackward {
        dy: usize,
        dx: usize,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    GatherBackward {
        dy: usize,
        indices: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    /// [`Op::MaxPool2dBackward`] — host CPU on unified-memory arena (F32).
    MaxPool2dBackward {
        x: usize,
        dy: usize,
        dx: usize,
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
    },
    /// [`Op::Conv2dBackwardInput`] — native MSL `conv2d` (same as decomposed `Op::Conv`).
    Conv2dBackwardInput {
        dy: usize,
        w: usize,
        dx: usize,
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
    /// [`Op::Conv2dBackwardWeight`] — GPU implicit im2col+GEMM (N=1); im2col+sgemm or CPU fallback.
    Conv2dBackwardWeight {
        x: usize,
        dy: usize,
        dw: usize,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
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
        dw_dil: u32,
        groups: u32,
    },
    /// [`Op::MaxPool3dBackward`] — native MSL gather (F32, NCDHW).
    MaxPool3dBackward {
        x: usize,
        dy: usize,
        dx: usize,
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
    },
    /// [`Op::Conv3dBackwardInput`] — native MSL gather (F32, NCDHW).
    Conv3dBackwardInput {
        dy: usize,
        w: usize,
        dx: usize,
        n: u32,
        c_in: u32,
        d: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
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
    },
    /// [`Op::Conv3dBackwardWeight`] — native MSL direct correlation (F32).
    Conv3dBackwardWeight {
        x: usize,
        dy: usize,
        dw: usize,
        n: u32,
        c_in: u32,
        d: u32,
        h: u32,
        w: u32,
        c_out: u32,
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
        dw_dil: u32,
        groups: u32,
    },

    /// User-registered custom op. Lowered from `Op::Custom`.
    /// `kernel` is resolved at compile time from
    /// `crate::op_registry::lookup_metal_kernel`. Execution requires
    /// a sync point: end_msl, commit, wait, run kernel against the
    /// unified-memory arena, restart cmd_buf. Apple-Silicon-only path
    /// for now (cfg-gated to macos with the rest of the crate).
    CustomOp {
        kernel: Arc<dyn MetalKernel>,
        inputs: Vec<(usize, u32, Shape)>, // (offset, len_elements, shape)
        output: (usize, u32, Shape),      // (offset, len_elements, shape)
        attrs: Vec<u8>,
    },

    /// User-registered **raw-GPU** custom op — dispatches a real MSL kernel onto
    /// the active compute encoder with NO host roundtrip and NO queue sync
    /// (contrast [`Thunk::CustomOp`], which commits + waits + runs a host
    /// kernel). `kernel` is resolved at compile time from
    /// `crate::op_registry::lookup_metal_gpu_kernel`, which takes precedence over
    /// a same-named host `MetalKernel`.
    CustomGpuOp {
        kernel: Arc<dyn MetalGpuKernel>,
        inputs: Vec<(usize, u32, Shape)>, // (offset, len_elements, shape)
        output: (usize, u32, Shape),      // (offset, len_elements, shape)
        attrs: Vec<u8>,
    },

    /// Core Riemannian / SPD-manifold op (`Op::BiMap` / `ReEig` / `LogEig` /
    /// `SpdBatchNorm` / `SpdKarcherMean` + backwards) — host fallback via
    /// `rlx_cpu::spd` (F64). Same sync pattern as `Fft1d` / `CustomOp`: flush
    /// the GPU, run the CPU thunk against the unified-memory arena, restart the
    /// cmd_buf. Kept distinct from `CustomOp` because the SPD kernels compute in
    /// F64 while the arena stores these nodes as f32 — the f32↔f64 widening
    /// lives in [`crate::spd::eval`]. The `Shape`s carry the op's REAL declared
    /// F64 dtype (not the widened f32 arena dtype) so the packed `[2n²+n]`
    /// forward layout / precomputed backward layout resolve automatically
    /// through the CPU thunk. Mirrors `rlx_vulkan::backend::Step::SpdHost`.
    SpdHost {
        op: rlx_ir::Op,
        inputs: Vec<(usize, u32, Shape)>, // (byte_offset, len_elements, F64-shape)
        output: (usize, u32, Shape),      // (byte_offset, len_elements, F64-shape)
    },

    /// 3D Gaussian splat forward — host fallback via `rlx_cpu::splat`
    /// (same sync pattern as `Fft1d` / `CustomOp`).
    GaussianSplatRender {
        positions_off: usize,
        positions_len: usize,
        scales_off: usize,
        scales_len: usize,
        rotations_off: usize,
        rotations_len: usize,
        opacities_off: usize,
        opacities_len: usize,
        colors_off: usize,
        colors_len: usize,
        sh_coeffs_off: usize,
        sh_coeffs_len: usize,
        meta_off: usize,
        dst_off: usize,
        dst_len: usize,
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
        positions_off: usize,
        positions_len: usize,
        scales_off: usize,
        scales_len: usize,
        rotations_off: usize,
        rotations_len: usize,
        opacities_off: usize,
        opacities_len: usize,
        colors_off: usize,
        colors_len: usize,
        sh_coeffs_off: usize,
        sh_coeffs_len: usize,
        meta_off: usize,
        d_loss_off: usize,
        d_loss_len: usize,
        packed_off: usize,
        packed_len: usize,
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
        positions_off: usize,
        positions_len: usize,
        scales_off: usize,
        scales_len: usize,
        rotations_off: usize,
        rotations_len: usize,
        opacities_off: usize,
        opacities_len: usize,
        colors_off: usize,
        colors_len: usize,
        sh_coeffs_off: usize,
        sh_coeffs_len: usize,
        meta_off: usize,
        meta_len: usize,
        prep_off: usize,
        prep_len: usize,
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
        prep_off: usize,
        prep_len: usize,
        meta_off: usize,
        meta_len: usize,
        dst_off: usize,
        dst_len: usize,
        count: usize,
        width: u32,
        height: u32,
        tile_size: u32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    /// SAM2 axial 2-D RoPE — host fallback on unified memory (F32).
    AxialRope2dHost {
        src: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        hidden: u32,
        end_x: u32,
        end_y: u32,
        head_dim: u32,
        num_heads: u32,
        theta: f32,
        repeat_factor: u32,
    },
    /// NCHW im2col — host fallback on unified memory (F32).
    Im2Col {
        x: usize,
        col: usize,
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
    /// 1D FFT on the 2N-real-block layout, lowered from `Op::Fft`.
    /// f32 pow-2 uses native multi-kernel MSL (`fft_dispatch`); f64/C64
    /// and non-pow2 use a host fallback against the unified-memory arena.
    Fft1d {
        src: usize,
        dst: usize,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype: rlx_ir::DType,
        /// native-gpu-fft real→complex fusion: `src` is an n-wide real signal
        /// (the Concat([signal, zeros]) was dropped); read it with im=0.
        real_input: bool,
    },
    /// Fused nearest-codebook assignment (`Op::Custom("rlx.vq_assign")`) as an
    /// on-GPU MSL kernel — one threadgroup per row, cooperative argmin over the
    /// codebook, reading the arena buffers directly (no D2H/H2D copy).
    VqAssign {
        x: usize,
        cb: usize,
        out: usize,
        n: u32,
        d: u32,
        k: u32,
        metric: u32,
    },
    /// General `Op::Scan` run as a host fallback: the compiled body loops on
    /// the CPU against the unified-memory arena (same sync pattern as `Fft1d`).
    /// Enables recurrences without a native Metal scan kernel — e.g. the IIR
    /// `biquad`. Executes via `rlx_cpu::thunk::execute_scan_host_desc`.
    ScanHost {
        desc: rlx_cpu::thunk::ScanHostDesc,
    },
    /// Nested-body op (`Op::ScanBackward` / `ScanBackwardXs`) via one-op CPU
    /// evaluation against the unified-memory arena (same sync pattern as ScanHost).
    HostOp {
        desc: rlx_cpu::thunk::HostOpDesc,
    },
    /// Native CPU ScatterNd / ScatterElements / GatherNd / GatherElements
    /// against the unified-memory arena (avoids HostOp mini-graph + I64 staging).
    CpuIndexing {
        thunk: rlx_cpu::thunk::IndexingThunk,
    },
    /// Log-mel from block-layout FFT spectrum (host fallback on Metal).
    LogMel {
        spec: usize,
        filters: usize,
        dst: usize,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    LogMelBackward {
        spec: usize,
        filters: usize,
        dy: usize,
        dst: usize,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    WelchPeaks {
        spec: usize,
        dst: usize,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
    },
    /// Host fill for [`Op::RngNormal`] (unified-memory arena).
    RngNormal {
        dst: usize,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    RngUniform {
        dst: usize,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },
}

pub struct ThunkSchedule {
    pub thunks: Vec<Thunk>,
    pub rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

/// Static-string name for each Thunk variant — used by the Perfetto
/// trace layer (PLAN L3) to label per-step events without allocating.
pub fn thunk_name(t: &Thunk) -> &'static str {
    match t {
        Thunk::Nop => "nop",
        Thunk::Cast { .. } => "cast",
        Thunk::CastHost { .. } => "cast_host",
        Thunk::CastTruncF32 { .. } => "cast_trunc_f32",
        Thunk::ScaledMatMul { .. } => "scaled_matmul",
        Thunk::ScaledQuantize { .. } => "scaled_quantize",
        Thunk::ScaledDequantize { .. } => "scaled_dequantize",
        Thunk::ScaledQuantScale { .. } => "scaled_quant_scale",
        Thunk::Sgemm { .. } => "sgemm",
        Thunk::BatchedSgemm { .. } => "batched_sgemm",
        Thunk::FusedMmBiasAct { .. } => "fused_mm_bias_act",
        Thunk::FusedBinaryActivation { .. } => "fused_binary_activation",
        Thunk::FusedTernaryActivation { .. } => "fused_ternary_activation",
        Thunk::ActivationInPlace { .. } => "activation",
        Thunk::ActivationOut { .. } => "activation_out",
        Thunk::GeluApproxOut { .. } => "gelu_approx_out",
        Thunk::GeluApproxHost { .. } => "gelu_approx_host",
        Thunk::LayerNorm { .. } => "layer_norm",
        Thunk::GroupNorm { .. } => "group_norm",
        Thunk::LayerNorm2d { .. } => "layer_norm2d",
        Thunk::ConvTranspose2d { .. } => "conv_transpose2d",
        Thunk::Conv3d { .. } => "conv3d",
        Thunk::ConvTranspose3d { .. } => "conv_transpose3d",
        Thunk::RmsNorm { .. } => "rms_norm",
        Thunk::ResizeNearest2x { .. } => "resize_nearest_2x",
        Thunk::BinaryFull { .. } => "binary",
        Thunk::BinaryBroadcast { .. } => "binary_broadcast",
        Thunk::BiasAdd { .. } => "bias_add",
        Thunk::FusedResidualLN { .. } => "fused_residual_ln",
        Thunk::FusedResidualRmsNorm { .. } => "fused_residual_rms_norm",
        Thunk::AdaLayerNorm { .. } => "ada_layer_norm",
        Thunk::GatedResidual { .. } => "gated_residual",
        Thunk::AdaLayerNormBackward { .. } => "ada_layer_norm_backward",
        Thunk::GatedResidualBackward { .. } => "gated_residual_backward",
        Thunk::FusedRmsNormMulSilu { .. } => "fused_rms_norm_mul_silu",
        Thunk::FusedDepthwiseConv1dBsc { .. } => "fused_depthwise_conv1d_bsc",
        Thunk::Gather { .. } => "gather",
        Thunk::Narrow { .. } => "narrow",
        Thunk::SplitLastAxis { .. } => "split_lastax",
        Thunk::Copy { .. } => "copy",
        Thunk::Attention { .. } => "attention",
        Thunk::FusedAttn { .. } => "fused_attn",
        Thunk::AttentionBackward { .. } => "attention_bwd",
        Thunk::RmsNormBackwardInput { .. } => "rms_norm_backward_input",
        Thunk::RmsNormBackwardGamma { .. } => "rms_norm_backward_gamma",
        Thunk::RmsNormBackwardBeta { .. } => "rms_norm_backward_beta",
        Thunk::LayerNormBackwardInput { .. } => "layer_norm_backward_input",
        Thunk::LayerNormBackwardGamma { .. } => "layer_norm_backward_gamma",
        Thunk::GroupNormBackwardInput { .. } => "group_norm_backward_input",
        Thunk::GroupNormBackwardGamma { .. } => "group_norm_backward_gamma",
        Thunk::GroupNormBackwardBeta { .. } => "group_norm_backward_beta",
        Thunk::RopeBackward { .. } => "rope_backward",
        Thunk::CumsumBackward { .. } => "cumsum_backward",
        Thunk::GatherBackward { .. } => "gather_backward",
        Thunk::MaxPool2dBackward { .. } => "maxpool2d_backward",
        Thunk::Conv2dBackwardInput { .. } => "conv2d_backward_input",
        Thunk::Conv2dBackwardWeight { .. } => "conv2d_backward_weight",
        Thunk::MaxPool3dBackward { .. } => "maxpool3d_backward",
        Thunk::Conv3dBackwardInput { .. } => "conv3d_backward_input",
        Thunk::Conv3dBackwardWeight { .. } => "conv3d_backward_weight",
        Thunk::Rope { .. } => "rope",
        Thunk::Softmax { .. } => "softmax",
        Thunk::SoftmaxCrossEntropyDense { .. } => "softmax_cross_entropy_dense",
        Thunk::SoftmaxCrossEntropyWithLogits { .. } => "softmax_cross_entropy_with_logits",
        Thunk::SoftmaxCrossEntropyBackward { .. } => "softmax_cross_entropy_backward",
        Thunk::Cumsum { .. } => "cumsum",
        Thunk::CumScan { .. } => "cum_scan",
        Thunk::FusedSwiGLU { .. } => "fused_swiglu",
        Thunk::Concat { .. } => "concat",
        Thunk::Compare { .. } => "compare",
        Thunk::Reduce { .. } => "reduce",
        Thunk::TopK { .. } => "topk",
        Thunk::GroupedMatMul { .. } => "grouped_matmul",
        Thunk::ScatterAdd { .. } => "scatter_add",
        Thunk::Transpose { .. } => "transpose",
        Thunk::GatherAxis { .. } => "gather_axis",
        Thunk::Pool2D { .. } => "pool2d",
        Thunk::Conv2D { .. } => "conv2d",
        Thunk::Where { .. } => "where",
        Thunk::Fma { .. } => "fma",
        Thunk::ReluBackward { .. } => "relu_backward",
        Thunk::ActivationBackward { .. } => "activation_backward",
        Thunk::FakeQuantizeFixed { .. } => "fake_quantize_fixed",
        Thunk::FakeQuantizePerBatch { .. } => "fake_quantize_perbatch",
        Thunk::ComplexNormSq { .. } => "complex_norm_sq",
        Thunk::ComplexNormSqBackward { .. } => "complex_norm_sq_backward",
        Thunk::FftButterflyStage { .. } => "fft_butterfly_stage",
        Thunk::ConjugateC64 { .. } => "conjugate_c64",
        Thunk::ElementwiseRegion { .. } => "elementwise_region",
        Thunk::BatchElementwiseRegion { .. } => "batch_elementwise_region",
        Thunk::CustomOp { .. } => "custom_op",
        Thunk::CustomGpuOp { .. } => "custom_gpu_op",
        Thunk::SpdHost { .. } => "spd_host",
        Thunk::GaussianSplatRender { .. } => "gaussian_splat_render",
        Thunk::GaussianSplatRenderBackward { .. } => "gaussian_splat_render_backward",
        Thunk::GaussianSplatPrepare { .. } => "gaussian_splat_prepare",
        Thunk::GaussianSplatRasterize { .. } => "gaussian_splat_rasterize",
        Thunk::AxialRope2dHost { .. } => "axial_rope2d_host",
        Thunk::Im2Col { .. } => "im2col",
        Thunk::Fft1d { .. } => "fft1d",
        Thunk::VqAssign { .. } => "vq_assign",
        Thunk::ScanHost { .. } => "scan_host",
        Thunk::HostOp { .. } => "host_op",
        Thunk::CpuIndexing { .. } => "cpu_indexing",
        Thunk::LogMel { .. } => "log_mel",
        Thunk::LogMelBackward { .. } => "log_mel_backward",
        Thunk::WelchPeaks { .. } => "welch_peaks",
        Thunk::RngNormal { .. } => "rng_normal",
        Thunk::RngUniform { .. } => "rng_uniform",
        Thunk::GatedDeltaNet { .. } => "gated_delta_net",
        Thunk::SelectiveScan { .. } => "selective_scan",
        Thunk::Sample { .. } => "sample",
        Thunk::Reverse { .. } => "reverse",
        Thunk::Pad { .. } => "pad",
        Thunk::Slice { .. } => "slice",
        Thunk::ArgReduce { .. } => "argreduce",
        Thunk::Lstm { .. } => "lstm",
        Thunk::Gru { .. } => "gru",
        Thunk::Rnn { .. } => "rnn",
        Thunk::Mamba2 { .. } => "mamba2",
        Thunk::DequantMatMulGguf { .. } => "dequant_matmul_gguf",
        Thunk::DequantGroupedMatMulGguf { .. } => "dequant_grouped_matmul_gguf",
        Thunk::DequantGroupedMatMulMlx { .. } => "dequant_grouped_matmul_mlx",
        Thunk::DequantMatMulInt8 { .. } => "dequant_matmul_int8",
        Thunk::DequantMatMulInt4 { .. } => "dequant_matmul_int4",
        Thunk::DequantMatMulFp8 { .. } => "dequant_matmul_fp8",
        Thunk::DequantMatMulNvfp4 { .. } => "dequant_matmul_nvfp4",
        Thunk::DequantMatMulMxFp4x2 { .. } => "dequant_matmul_mxfp4x2",
        Thunk::DequantMatMulMlx { .. } => "dequant_matmul_mlx",
        Thunk::FusedMlpGateUpSwiGLU { .. } => "fused_mlp_gate_up_swiglu",
        Thunk::FusedMlpGateUpGelu { .. } => "fused_mlp_gate_up_gelu",
        Thunk::FusedMlpDownResidual { .. } => "fused_mlp_down_residual",
    }
}

impl Thunk {
    /// True when this Metal Thunk variant honors active-extent dispatch
    /// (PLAN L1). Backend mirrors the CPU contract: whole-schedule
    /// validation in `crate::backend::MetalExecutable::all_safe_for_active`.
    /// Initial coverage: trivially-scalable elementwise + matmul +
    /// norm + softmax + simple shape ops. Macro-kernels (Attention,
    /// FusedAttnBlock, FusedBertLayer, FusedNomicLayer), Conv/Pool,
    /// ScatterAdd, Transpose, GroupedMatMul still default to unsafe.
    pub fn safe_for_active_extent(&self) -> bool {
        match self {
            Thunk::Nop
            | Thunk::Cast { .. }
            | Thunk::CastHost { .. }
            | Thunk::CastTruncF32 { .. }
            | Thunk::Copy { .. }
            | Thunk::ActivationInPlace { .. }
            | Thunk::ActivationOut { .. }
            | Thunk::GeluApproxOut { .. }
            | Thunk::GeluApproxHost { .. }
            | Thunk::FusedBinaryActivation { .. }
            | Thunk::FusedTernaryActivation { .. }
            | Thunk::Sgemm { .. }
            | Thunk::BatchedSgemm { .. }
            | Thunk::FusedMmBiasAct { .. }
            | Thunk::BiasAdd { .. }
            | Thunk::LayerNorm { .. }
            | Thunk::RmsNorm { .. }
            | Thunk::Softmax { .. }
            | Thunk::SoftmaxCrossEntropyDense { .. }
            | Thunk::Cumsum { .. }
            | Thunk::CumScan { .. }
            | Thunk::FusedResidualLN { .. }
            | Thunk::FusedResidualRmsNorm { .. }
            | Thunk::AdaLayerNorm { .. }
            | Thunk::GatedResidual { .. }
            | Thunk::AdaLayerNormBackward { .. }
            | Thunk::GatedResidualBackward { .. }
            | Thunk::FusedRmsNormMulSilu { .. }
            | Thunk::FusedDepthwiseConv1dBsc { .. }
            | Thunk::Gather { .. }
            | Thunk::Compare { .. }
            | Thunk::Where { .. }
            | Thunk::Fma { .. }
            | Thunk::ReluBackward { .. }
            | Thunk::ActivationBackward { .. }
            | Thunk::ComplexNormSq { .. }
            | Thunk::ComplexNormSqBackward { .. }
            | Thunk::ConjugateC64 { .. }
            | Thunk::FftButterflyStage { .. }
            | Thunk::FusedSwiGLU { .. }
            | Thunk::ElementwiseRegion { .. }
            | Thunk::BatchElementwiseRegion { .. }
            | Thunk::Narrow { .. }
            | Thunk::SplitLastAxis { .. }
            | Thunk::Reduce { .. }
            | Thunk::TopK { .. }
            | Thunk::GroupedMatMul { .. }
            | Thunk::GatherAxis { .. }
            | Thunk::Concat { .. }
            | Thunk::Conv2D { .. }
            | Thunk::Pool2D { .. } => true,
            // PLAN L1 stride-vs-bound separation: MSL kernels for
            // Attention / Rope take a `seq_stride` runtime arg
            // (compile-time full extent) for per-batch buffer offset
            // math, while `seq` is the active loop bound only. Safe
            // at any batch.
            Thunk::Attention { .. } => true,
            Thunk::AttentionBackward { .. } => true,
            Thunk::RmsNormBackwardInput { .. }
            | Thunk::RmsNormBackwardGamma { .. }
            | Thunk::RmsNormBackwardBeta { .. }
            | Thunk::LayerNormBackwardInput { .. }
            | Thunk::LayerNormBackwardGamma { .. }
            | Thunk::GroupNormBackwardInput { .. }
            | Thunk::GroupNormBackwardGamma { .. }
            | Thunk::GroupNormBackwardBeta { .. }
            | Thunk::RopeBackward { .. }
            | Thunk::CumsumBackward { .. }
            | Thunk::GatherBackward { .. }
            | Thunk::MaxPool2dBackward { .. }
            | Thunk::Conv2dBackwardInput { .. }
            | Thunk::Conv2dBackwardWeight { .. }
            | Thunk::MaxPool3dBackward { .. }
            | Thunk::Conv3dBackwardInput { .. }
            | Thunk::Conv3dBackwardWeight { .. } => true,
            Thunk::Rope { .. } => true,
            // Decode seq=1 GDN / fused GGUF matmul: host paths use full
            // `batch`/`m` from the thunk (not seq-axis scale); marking
            // safe lets bucketed decode bypass whole-graph MPSGraph.
            Thunk::GatedDeltaNet { .. }
            | Thunk::SelectiveScan { .. }
            | Thunk::Sample { .. }
            | Thunk::Reverse { .. }
            | Thunk::Pad { .. }
            | Thunk::Slice { .. }
            | Thunk::ArgReduce { .. }
            | Thunk::Lstm { .. }
            | Thunk::Gru { .. }
            | Thunk::Rnn { .. }
            | Thunk::Mamba2 { .. }
            | Thunk::DequantMatMulGguf { .. }
            | Thunk::DequantGroupedMatMulGguf { .. }
            | Thunk::DequantGroupedMatMulMlx { .. }
            | Thunk::DequantMatMulInt8 { .. }
            | Thunk::DequantMatMulInt4 { .. }
            | Thunk::DequantMatMulFp8 { .. }
            | Thunk::DequantMatMulNvfp4 { .. }
            | Thunk::DequantMatMulMxFp4x2 { .. }
            | Thunk::DequantMatMulMlx { .. }
            | Thunk::FusedMlpGateUpSwiGLU { .. }
            | Thunk::FusedMlpGateUpGelu { .. }
            | Thunk::FusedMlpDownResidual { .. } => true,
            // ScatterAdd: same zero-padding analysis as CPU — padded
            // updates contribute zero to accumulate-into-zeros, so
            // active and full produce the same output for K real
            // updates. Active path zeros the FULL output then scatters
            // first num_updates_active.
            Thunk::ScatterAdd { .. } => true,
            // Transpose: same conservative predicate as CPU. Safe iff
            // `in_strides[0] == product(out_dims[1..])` (= perm[0] == 0,
            // bucket axis stays at output position 0).
            Thunk::Transpose {
                out_dims,
                in_strides,
                ..
            } => {
                if out_dims.is_empty() || in_strides.is_empty() {
                    return false;
                }
                let inner: u32 = out_dims[1..].iter().product();
                in_strides[0] == inner
            }
            _ => false,
        }
    }
}

mod compile;

impl ThunkSchedule {}

fn strides_dense_contiguous(rank: usize, dims: &[u32], strides: &[u32]) -> bool {
    if rank == 0 || dims.len() < rank || strides.len() < rank {
        return rank == 0;
    }
    let mut expected = 1u32;
    for ax in (0..rank).rev() {
        if strides[ax] != expected {
            return false;
        }
        expected = expected.saturating_mul(dims[ax].max(1));
    }
    true
}

/// Drop trivial `ElementwiseRegion` chains to direct thunks (vec4 elem / in-place act).
fn rewrite_simple_elementwise_regions(thunks: &mut Vec<Thunk>) {
    let mut i = 0;
    while i < thunks.len() {
        match try_rewrite_elementwise_region(&thunks[i]) {
            RegionRewrite::Keep => {
                i += 1;
            }
            RegionRewrite::One(t) => {
                thunks[i] = t;
                i += 1;
            }
            RegionRewrite::Many(ts) => {
                if ts.is_empty() {
                    i += 1;
                    continue;
                }
                let n = ts.len();
                thunks.splice(i..=i, ts);
                i += n;
            }
        }
    }
}

enum RegionRewrite {
    Keep,
    One(Thunk),
    Many(Vec<Thunk>),
}

fn region_is_dense(n_in: usize, scalar_input_mask: u32, input_modulus: &[u32; 16]) -> bool {
    scalar_input_mask == 0 && !input_modulus.iter().take(n_in).any(|&m| m != 0)
}

fn decode_input_operand(enc: u32) -> Option<usize> {
    if enc & 0x8000_0000 != 0 {
        None
    } else {
        Some(enc as usize)
    }
}

fn decode_step_operand(enc: u32) -> Option<usize> {
    if enc & 0x8000_0000 == 0 {
        None
    } else {
        Some((enc & 0x7FFF_FFFF) as usize)
    }
}

fn map_chain_binary_op(sub: u32) -> Option<rlx_ir::op::BinaryOp> {
    use rlx_ir::op::BinaryOp;
    Some(match sub {
        0 => BinaryOp::Add,
        1 => BinaryOp::Sub,
        2 => BinaryOp::Mul,
        3 => BinaryOp::Div,
        4 => BinaryOp::Max,
        5 => BinaryOp::Min,
        6 => BinaryOp::Pow,
        _ => return None,
    })
}

fn map_chain_activation(sub: u32) -> Option<rlx_ir::op::Activation> {
    use rlx_ir::op::Activation;
    Some(match sub {
        0 | 1 => Activation::Gelu,
        2 => Activation::Silu,
        3 => Activation::Relu,
        4 => Activation::Sigmoid,
        5 => Activation::Tanh,
        6 => Activation::Exp,
        7 => Activation::Log,
        8 => Activation::Sqrt,
        9 => Activation::Rsqrt,
        10 => Activation::Neg,
        11 => Activation::Abs,
        _ => return None,
    })
}

fn try_rewrite_elementwise_region(t: &Thunk) -> RegionRewrite {
    let Thunk::ElementwiseRegion {
        len,
        num_inputs,
        num_steps,
        dst,
        input_offs,
        chain,
        scalar_input_mask,
        input_modulus,
        prologue,
        out_n: _,
        out_c: _,
        out_h: _,
        out_w: _,
        prologue_input: _,
    } = t
    else {
        return RegionRewrite::Keep;
    };

    if *prologue != 0 {
        return RegionRewrite::Keep;
    }

    let n_in = *num_inputs as usize;
    if !region_is_dense(n_in, *scalar_input_mask, input_modulus) {
        return RegionRewrite::Keep;
    }

    let input_byte = |idx: usize| input_offs[idx] as usize * 4;

    if *num_steps == 1 && chain[0] == 2 && n_in == 2 {
        let Some(lhs_idx) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        let Some(rhs_idx) = decode_input_operand(chain[3]) else {
            return RegionRewrite::Keep;
        };
        if lhs_idx >= n_in || rhs_idx >= n_in {
            return RegionRewrite::Keep;
        }
        let Some(op) = map_chain_binary_op(chain[1]) else {
            return RegionRewrite::Keep;
        };
        return RegionRewrite::One(Thunk::BinaryFull {
            lhs: input_byte(lhs_idx),
            rhs: input_byte(rhs_idx),
            dst: *dst,
            len: *len,
            op,
            dt: HalfFlag::F32,
        });
    }

    if *num_steps == 2 && chain[0] == 2 && chain[4] == 0 {
        let Some(lhs_idx) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        let Some(rhs_idx) = decode_input_operand(chain[3]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[6]) != Some(0) {
            return RegionRewrite::Keep;
        }
        let Some(op) = map_chain_binary_op(chain[1]) else {
            return RegionRewrite::Keep;
        };
        let Some(act) = map_chain_activation(chain[5]) else {
            return RegionRewrite::Keep;
        };
        if lhs_idx >= n_in || rhs_idx >= n_in {
            return RegionRewrite::Keep;
        }
        return RegionRewrite::One(Thunk::FusedBinaryActivation {
            lhs: input_byte(lhs_idx),
            rhs: input_byte(rhs_idx),
            dst: *dst,
            len: *len,
            op,
            act,
            dt: HalfFlag::F32,
        });
    }

    if *num_steps == 2 && chain[0] == 2 && chain[4] == 2 {
        let Some(lhs0) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        let Some(rhs0) = decode_input_operand(chain[3]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[6]) != Some(0) {
            return RegionRewrite::Keep;
        }
        let Some(rhs1) = decode_input_operand(chain[7]) else {
            return RegionRewrite::Keep;
        };
        let Some(op0) = map_chain_binary_op(chain[1]) else {
            return RegionRewrite::Keep;
        };
        let Some(op1) = map_chain_binary_op(chain[5]) else {
            return RegionRewrite::Keep;
        };
        if lhs0 >= n_in || rhs0 >= n_in || rhs1 >= n_in {
            return RegionRewrite::Keep;
        }
        return RegionRewrite::Many(vec![
            Thunk::BinaryFull {
                lhs: input_byte(lhs0),
                rhs: input_byte(rhs0),
                dst: *dst,
                len: *len,
                op: op0,
                dt: HalfFlag::F32,
            },
            Thunk::BinaryFull {
                lhs: *dst,
                rhs: input_byte(rhs1),
                dst: *dst,
                len: *len,
                op: op1,
                dt: HalfFlag::F32,
            },
        ]);
    }

    if *num_steps == 3 && chain[0] == 2 && chain[4] == 2 && chain[8] == 0 {
        let Some(lhs0) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        let Some(rhs0) = decode_input_operand(chain[3]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[6]) != Some(0) {
            return RegionRewrite::Keep;
        }
        let Some(rhs1) = decode_input_operand(chain[7]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[10]) != Some(1) {
            return RegionRewrite::Keep;
        }
        let Some(op0) = map_chain_binary_op(chain[1]) else {
            return RegionRewrite::Keep;
        };
        let Some(op1) = map_chain_binary_op(chain[5]) else {
            return RegionRewrite::Keep;
        };
        let Some(act) = map_chain_activation(chain[9]) else {
            return RegionRewrite::Keep;
        };
        if lhs0 >= n_in || rhs0 >= n_in || rhs1 >= n_in {
            return RegionRewrite::Keep;
        }
        return RegionRewrite::One(Thunk::FusedTernaryActivation {
            lhs: input_byte(lhs0),
            rhs0: input_byte(rhs0),
            rhs1: input_byte(rhs1),
            dst: *dst,
            len: *len,
            op0,
            op1,
            act,
            dt: HalfFlag::F32,
        });
    }

    if *num_steps == 3 && chain[0] == 2 && chain[4] == 2 && chain[8] == 2 {
        let Some(lhs0) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        let Some(rhs0) = decode_input_operand(chain[3]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[6]) != Some(0) {
            return RegionRewrite::Keep;
        }
        let Some(rhs1) = decode_input_operand(chain[7]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[10]) != Some(1) {
            return RegionRewrite::Keep;
        };
        let Some(rhs2) = decode_input_operand(chain[11]) else {
            return RegionRewrite::Keep;
        };
        let Some(op0) = map_chain_binary_op(chain[1]) else {
            return RegionRewrite::Keep;
        };
        let Some(op1) = map_chain_binary_op(chain[5]) else {
            return RegionRewrite::Keep;
        };
        let Some(op2) = map_chain_binary_op(chain[9]) else {
            return RegionRewrite::Keep;
        };
        if lhs0 >= n_in || rhs0 >= n_in || rhs1 >= n_in || rhs2 >= n_in {
            return RegionRewrite::Keep;
        }
        return RegionRewrite::Many(vec![
            Thunk::BinaryFull {
                lhs: input_byte(lhs0),
                rhs: input_byte(rhs0),
                dst: *dst,
                len: *len,
                op: op0,
                dt: HalfFlag::F32,
            },
            Thunk::BinaryFull {
                lhs: *dst,
                rhs: input_byte(rhs1),
                dst: *dst,
                len: *len,
                op: op1,
                dt: HalfFlag::F32,
            },
            Thunk::BinaryFull {
                lhs: *dst,
                rhs: input_byte(rhs2),
                dst: *dst,
                len: *len,
                op: op2,
                dt: HalfFlag::F32,
            },
        ]);
    }

    if *num_steps == 2 && chain[0] == 1 && chain[4] == 2 {
        let Some(cast_in) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        if decode_step_operand(chain[6]) != Some(0) {
            return RegionRewrite::Keep;
        }
        let Some(rhs_idx) = decode_input_operand(chain[7]) else {
            return RegionRewrite::Keep;
        };
        let Some(op) = map_chain_binary_op(chain[5]) else {
            return RegionRewrite::Keep;
        };
        if cast_in >= n_in || rhs_idx >= n_in {
            return RegionRewrite::Keep;
        }
        return RegionRewrite::One(Thunk::BinaryFull {
            lhs: input_byte(cast_in),
            rhs: input_byte(rhs_idx),
            dst: *dst,
            len: *len,
            op,
            dt: HalfFlag::F32,
        });
    }

    if *num_steps == 1 && chain[0] == 0 && n_in > 0 {
        let Some(src_idx) = decode_input_operand(chain[2]) else {
            return RegionRewrite::Keep;
        };
        if src_idx >= n_in {
            return RegionRewrite::Keep;
        }
        let data = input_byte(src_idx);
        if data != *dst {
            return RegionRewrite::Keep;
        }
        let Some(act) = map_chain_activation(chain[1]) else {
            return RegionRewrite::Keep;
        };
        return RegionRewrite::One(Thunk::ActivationInPlace {
            data,
            len: *len,
            act,
            dt: HalfFlag::F32,
        });
    }

    RegionRewrite::Keep
}

/// `BinaryBroadcast` with both operands row-major dense → `BinaryFull` (vec4 elem kernels).
fn rewrite_dense_binary_broadcast(thunks: &mut [Thunk]) {
    for t in thunks.iter_mut() {
        let Thunk::BinaryBroadcast {
            lhs,
            rhs,
            dst,
            len,
            op,
            dt,
            rank,
            out_dims,
            lhs_strides,
            rhs_strides,
        } = t
        else {
            continue;
        };
        let rank = *rank as usize;
        if rank == 0
            || !strides_dense_contiguous(rank, out_dims, lhs_strides)
            || !strides_dense_contiguous(rank, out_dims, rhs_strides)
        {
            continue;
        }
        *t = Thunk::BinaryFull {
            lhs: *lhs,
            rhs: *rhs,
            dst: *dst,
            len: *len,
            op: *op,
            dt: *dt,
        };
    }
}

fn narrow_segments_partition(src_axis: u32, segments: &[(u32, u32)]) -> bool {
    let mut sorted = segments.to_vec();
    sorted.sort_by_key(|(s, _)| *s);
    let mut end = 0u32;
    for (start, len) in sorted {
        if start != end {
            return false;
        }
        end = end.saturating_add(len);
    }
    end == src_axis
}

/// Count of decode MLP blocks fused so far (process-wide). Lets tests confirm
/// the fused path actually fired without parsing logs. Monotonic; read the
/// delta around a compile.
pub static FUSED_DECODE_MLP_BLOCKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Number of decode MLP blocks fused across this process's lifetime.
pub fn fused_decode_mlp_blocks() -> usize {
    FUSED_DECODE_MLP_BLOCKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Count of residual-add+RmsNorm blocks fused (process-wide).
pub static FUSED_RESIDUAL_RMS_BLOCKS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Number of residual+RmsNorm blocks fused across this process's lifetime.
pub fn fused_residual_rms_blocks() -> usize {
    FUSED_RESIDUAL_RMS_BLOCKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fully-analyzable per-thunk (reads, writes) for the decode-MLP liveness scan.
/// `None` ⇒ the variant's data-flow is not enumerated here, so the caller must
/// treat it as an opaque potential reader/writer and bail (conservative: a
/// missed fusion is fine, an unsafe one is not). The `reads` list MUST be
/// complete for every `Some` variant — an under-reported read would let the
/// pass drop a live intermediate. In-place ops list their slot in BOTH lists.
fn mlp_io(t: &Thunk) -> Option<(Vec<usize>, Vec<usize>)> {
    use Thunk::*;
    let io = match t {
        Nop => (vec![], vec![]),
        Cast { src, dst, .. } | CastHost { src, dst, .. } | CastTruncF32 { src, dst, .. } => {
            (vec![*src], vec![*dst])
        }
        Copy { src, dst, .. } => (vec![*src], vec![*dst]),
        ActivationInPlace { data, .. } => (vec![*data], vec![*data]),
        ActivationOut { src, dst, .. } => (vec![*src], vec![*dst]),
        GeluApproxOut { src, dst, .. } | GeluApproxHost { src, dst, .. } => {
            (vec![*src], vec![*dst])
        }
        BinaryFull { lhs, rhs, dst, .. } => (vec![*lhs, *rhs], vec![*dst]),
        BinaryBroadcast { lhs, rhs, dst, .. } => (vec![*lhs, *rhs], vec![*dst]),
        FusedBinaryActivation { lhs, rhs, dst, .. } => (vec![*lhs, *rhs], vec![*dst]),
        FusedTernaryActivation {
            lhs,
            rhs0,
            rhs1,
            dst,
            ..
        } => (vec![*lhs, *rhs0, *rhs1], vec![*dst]),
        BiasAdd { src, bias, dst, .. } => (vec![*src, *bias], vec![*dst]),
        Fma { a, b, c, dst, .. } => (vec![*a, *b, *c], vec![*dst]),
        ReluBackward { x, dy, dx, .. } | ActivationBackward { x, dy, dx, .. } => {
            (vec![*x, *dy], vec![*dx])
        }
        FakeQuantizeFixed {
            src, scale, dst, ..
        } => (vec![*src, *scale], vec![*dst]),
        FakeQuantizePerBatch { src, dst, .. } => (vec![*src], vec![*dst]),
        ComplexNormSq { src, dst, .. } | ConjugateC64 { src, dst, .. } => (vec![*src], vec![*dst]),
        ComplexNormSqBackward { z, g, dz, .. } => (vec![*z, *g], vec![*dz]),
        FftButterflyStage {
            state,
            out,
            gate,
            rev,
            tw_re,
            tw_im,
            ..
        } => (vec![*state, *gate, *rev, *tw_re, *tw_im], vec![*out]),
        Where {
            cond,
            on_true,
            on_false,
            dst,
            ..
        } => (vec![*cond, *on_true, *on_false], vec![*dst]),
        Compare { lhs, rhs, dst, .. } => (vec![*lhs, *rhs], vec![*dst]),
        RmsNorm { src, g, b, dst, .. } => (vec![*src, *g, *b], vec![*dst]),
        LayerNorm { src, g, b, dst, .. } => (vec![*src, *g, *b], vec![*dst]),
        FusedResidualLN {
            x,
            res,
            bias,
            g,
            b,
            out,
            ..
        }
        | FusedResidualRmsNorm {
            x,
            res,
            bias,
            g,
            b,
            out,
            ..
        } => (vec![*x, *res, *bias, *g, *b], vec![*out]),
        AdaLayerNorm {
            x,
            scale,
            shift,
            out,
            ..
        } => (vec![*x, *scale, *shift], vec![*out]),
        GatedResidual {
            x, y, gate, out, ..
        } => (vec![*x, *y, *gate], vec![*out]),
        FusedRmsNormMulSilu {
            x, g, b, z, out, ..
        } => (vec![*x, *g, *b, *z], vec![*out]),
        FusedDepthwiseConv1dBsc {
            src, weight, dst, ..
        } => (vec![*src, *weight], vec![*dst]),
        Conv2D {
            src, weight, dst, ..
        } => (vec![*src, *weight], vec![*dst]),
        FusedSwiGLU { src, dst, .. } => (vec![*src], vec![*dst]),
        Softmax { data, .. } => (vec![*data], vec![*data]),
        Rope {
            src, cos, sin, dst, ..
        } => (vec![*src, *cos, *sin], vec![*dst]),
        Attention {
            q, k, v, mask, out, ..
        } => (vec![*q, *k, *v, *mask], vec![*out]),
        FusedAttn {
            qkv,
            mask,
            cos,
            sin,
            out,
            has_rope,
            ..
        } => {
            let mut r = vec![*qkv, *mask];
            if *has_rope != 0 {
                r.push(*cos);
                r.push(*sin);
            }
            (r, vec![*out])
        }
        Concat { dst, inputs, .. } => (inputs.iter().map(|(o, _)| *o).collect(), vec![*dst]),
        // Reads `src`, writes each output segment's offset. Modeling this (vs
        // falling through to the opaque `_ => None` bail) lets the MLP-fusion
        // liveness scan cross the per-layer QKV split — without it, only the
        // final layers (whose forward dead-scan never reaches another split)
        // fused, so 26/28 decode MLP blocks silently stayed unfused.
        SplitLastAxis { src, segments, .. } => {
            (vec![*src], segments.iter().map(|(o, _, _)| *o).collect())
        }
        Narrow { src, dst, .. } => (vec![*src], vec![*dst]),
        Gather {
            table, idx, dst, ..
        }
        | GatherAxis {
            table, idx, dst, ..
        } => (vec![*table, *idx], vec![*dst]),
        Sgemm { a, b, c, .. } => (vec![*a, *b], vec![*c]),
        BatchedSgemm { a, b, c, .. } => (vec![*a, *b], vec![*c]),
        FusedMmBiasAct { a, w, bias, c, .. } => (vec![*a, *w, *bias], vec![*c]),
        DequantMatMulGguf { x, w_q, dst, .. } => (vec![*x, *w_q], vec![*dst]),
        FusedMlpGateUpSwiGLU {
            x,
            gate_w,
            up_w,
            dst,
            ..
        }
        | FusedMlpGateUpGelu {
            x,
            gate_w,
            up_w,
            dst,
            ..
        } => (vec![*x, *gate_w, *up_w], vec![*dst]),
        FusedMlpDownResidual { x, w, res, dst, .. } => (vec![*x, *w, *res], vec![*dst]),
        Reduce { src, dst, .. } => (vec![*src], vec![*dst]),
        Transpose { src, dst, .. } => (vec![*src], vec![*dst]),
        _ => return None,
    };
    Some(io)
}

const SENTINEL_OFF: usize = usize::MAX;

/// Last thunk index in `[0, before)` that writes `off`, or `None`.
/// Returns `Err(())` if an opaque (`mlp_io == None`) thunk is encountered,
/// since it might also write `off` — caller bails.
fn mlp_last_writer(thunks: &[Thunk], before: usize, off: usize) -> Result<Option<usize>, ()> {
    if off == SENTINEL_OFF {
        return Ok(None);
    }
    for i in (0..before).rev() {
        match mlp_io(&thunks[i]) {
            None => return Err(()),
            Some((_, writes)) => {
                if writes.contains(&off) {
                    return Ok(Some(i));
                }
            }
        }
    }
    Ok(None)
}

/// First thunk index in `(after, end)` whose `pred` holds.
fn mlp_find_forward<F: Fn(&Thunk) -> bool>(
    thunks: &[Thunk],
    after: usize,
    pred: F,
) -> Option<usize> {
    (after + 1..thunks.len()).find(|&i| pred(&thunks[i]))
}

/// True iff the value at `off` written by `producer` is read only by `allowed`
/// thunks before it is redefined, scanning `producer+1..until` (layer-local
/// when `until` is the next op outside the fused block). Opaque thunk ⇒ false.
fn mlp_value_dead_in_range(
    thunks: &[Thunk],
    producer: usize,
    off: usize,
    allowed: &[usize],
    until: usize,
) -> bool {
    if off == SENTINEL_OFF {
        return false;
    }
    let until = until.min(thunks.len());
    for (i, t) in thunks
        .iter()
        .enumerate()
        .skip(producer + 1)
        .take(until.saturating_sub(producer + 1))
    {
        let Some((reads, writes)) = mlp_io(t) else {
            return false;
        };
        if writes.contains(&off) {
            break;
        }
        if reads.contains(&off) && !allowed.contains(&i) {
            return false;
        }
    }
    true
}

/// True iff `off` is not written in `thunks[start..until)` (opaque thunks ⇒ false).
fn mlp_unwritten_in_range(thunks: &[Thunk], start: usize, until: usize, off: usize) -> bool {
    if off == SENTINEL_OFF {
        return false;
    }
    for t in thunks.iter().take(until.min(thunks.len())).skip(start) {
        match mlp_io(t) {
            Some((_, writes)) => {
                if writes.contains(&off) {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

/// True when two f32 arena slices overlap (exact offset match or partial range).
fn mlp_f32_ranges_overlap(a_off: usize, a_elems: u32, b_off: usize, b_elems: u32) -> bool {
    if a_elems == 0 || b_elems == 0 {
        return false;
    }
    if a_off == b_off {
        return true;
    }
    let a_end = a_off.saturating_add(a_elems as usize * 4);
    let b_end = b_off.saturating_add(b_elems as usize * 4);
    a_off < b_end && b_off < a_end
}

/// Packed gate/up weight bytes per output column for fused decode MLP GEMV.
fn mlp_gate_up_row_bytes(k: u32, scheme: rlx_ir::quant::QuantScheme) -> usize {
    use rlx_ir::quant::QuantScheme;
    match scheme {
        QuantScheme::GgufQ4K => (k as usize / 256) * 144,
        QuantScheme::GgufQ5_0 => (k as usize).div_ceil(32) * 22,
        QuantScheme::GgufQ6K => (k as usize / 256) * 210,
        QuantScheme::GgufQ1_0 => (k as usize / 128) * 18,
        QuantScheme::GgufQ2_0 => (k as usize / 128) * 34,
        _ => 0,
    }
}

/// Pattern-merge the fused `gate_up` packed matmul path:
///   combined(DequantMatMul) → narrow(gate) + narrow(up) → silu/gelu → mul
/// → `FusedMlpGateUpSwiGLU` / `FusedMlpGateUpGelu`, optionally plus
/// `FusedMlpDownResidual` when down feeds the residual add directly.
fn fuse_decode_mlp_combined_gate_up(
    thunks: &mut [Thunk],
    output_offsets: &std::collections::HashSet<usize>,
) {
    if rlx_ir::env::var("RLX_METAL_FUSE_DECODE").as_deref() == Some("0") {
        return;
    }
    use rlx_ir::quant::QuantScheme;
    let verbose = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");

    let as_packed_gate_up_mm = |t: &Thunk| -> Option<(usize, usize, usize, u32, u32, QuantScheme)> {
        if let Thunk::DequantMatMulGguf {
            x,
            w_q,
            dst,
            m,
            k,
            n,
            scheme,
            ..
        } = *t
        {
            if m == 1 && matches!(scheme, QuantScheme::GgufQ4K | QuantScheme::GgufQ5_0) {
                return Some((x, w_q, dst, k, n, scheme));
            }
        }
        None
    };

    let as_narrow = |t: &Thunk| -> Option<(usize, usize, u32, u32)> {
        if let Thunk::Narrow {
            src,
            dst,
            start,
            len,
            ..
        } = *t
        {
            Some((src, dst, start, len))
        } else {
            None
        }
    };

    let is_silu = |t: &Thunk| {
        matches!(
            t,
            Thunk::ActivationInPlace {
                act: Activation::Silu,
                ..
            } | Thunk::ActivationOut {
                act: Activation::Silu,
                ..
            }
        )
    };
    let is_gelu = |t: &Thunk| {
        matches!(
            t,
            Thunk::ActivationInPlace {
                act: Activation::GeluApprox,
                ..
            } | Thunk::ActivationOut {
                act: Activation::GeluApprox,
                ..
            } | Thunk::GeluApproxOut { .. }
                | Thunk::GeluApproxHost { .. }
        )
    };

    let n_thunks = thunks.len();
    let mut i = 0;
    while i < n_thunks {
        let (mul_lhs, mul_rhs, prod) = match &thunks[i] {
            Thunk::BinaryFull {
                lhs,
                rhs,
                dst,
                op: BinaryOp::Mul,
                ..
            } => (*lhs, *rhs, *dst),
            _ => {
                i += 1;
                continue;
            }
        };
        let mul_idx = i;

        let silu_on = |off: usize| -> Option<usize> {
            match mlp_last_writer(thunks, mul_idx, off) {
                Ok(Some(idx)) if is_silu(&thunks[idx]) => Some(idx),
                _ => None,
            }
        };
        let gelu_on = |off: usize| -> Option<usize> {
            match mlp_last_writer(thunks, mul_idx, off) {
                Ok(Some(idx)) if is_gelu(&thunks[idx]) => Some(idx),
                _ => None,
            }
        };
        let (act_idx, up_off, use_gelu) = if let Some(idx) = silu_on(mul_lhs) {
            (idx, mul_rhs, false)
        } else if let Some(idx) = silu_on(mul_rhs) {
            (idx, mul_lhs, false)
        } else if let Some(idx) = gelu_on(mul_lhs) {
            (idx, mul_rhs, true)
        } else if let Some(idx) = gelu_on(mul_rhs) {
            (idx, mul_lhs, true)
        } else {
            i += 1;
            continue;
        };

        // GeGLU combined fusion: opt-out (`RLX_METAL_FUSE_DECODE_GELU=0`) when a
        // bucket graph still shows arena overlap on a given platform.
        if use_gelu && rlx_ir::env::var("RLX_METAL_FUSE_DECODE_GELU").as_deref() == Some("0") {
            i += 1;
            continue;
        }

        let gate_src_off = match &thunks[act_idx] {
            Thunk::ActivationInPlace { data, .. } => *data,
            Thunk::ActivationOut { src, .. } => *src,
            Thunk::GeluApproxOut { src, .. } | Thunk::GeluApproxHost { src, .. } => *src,
            _ => {
                i += 1;
                continue;
            }
        };
        let gate_narrow_idx = match mlp_last_writer(thunks, act_idx, gate_src_off) {
            Ok(Some(idx)) if as_narrow(&thunks[idx]).is_some() => idx,
            Ok(Some(copy_idx)) if matches!(&thunks[copy_idx], Thunk::Copy { .. }) => {
                let Thunk::Copy { src, .. } = &thunks[copy_idx] else {
                    i += 1;
                    continue;
                };
                match mlp_last_writer(thunks, copy_idx, *src) {
                    Ok(Some(ni)) if as_narrow(&thunks[ni]).is_some() => ni,
                    _ => {
                        i += 1;
                        continue;
                    }
                }
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let (combined_off, _gate_dst, gate_start, gate_len) =
            as_narrow(&thunks[gate_narrow_idx]).unwrap();

        let up_narrow_idx = match mlp_last_writer(thunks, mul_idx, up_off) {
            Ok(Some(idx)) if as_narrow(&thunks[idx]).is_some() => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let (combined_up, _up_dst, up_start, up_len) = as_narrow(&thunks[up_narrow_idx]).unwrap();
        if combined_off != combined_up || gate_len != up_len || up_start != gate_start + gate_len {
            i += 1;
            continue;
        }
        let n_half = gate_len;

        let combined_mm_idx = match mlp_last_writer(thunks, gate_narrow_idx, combined_off) {
            Ok(Some(idx)) if as_packed_gate_up_mm(&thunks[idx]).is_some() => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let (comb_x, comb_w, _comb_dst, comb_k, comb_n, comb_scheme) =
            as_packed_gate_up_mm(&thunks[combined_mm_idx]).unwrap();
        let (comb_x_f16, comb_dst_f16) = match &thunks[combined_mm_idx] {
            Thunk::DequantMatMulGguf { x_f16, dst_f16, .. } => (*x_f16, *dst_f16),
            _ => (false, false),
        };
        // Combined gate_up fused kernels are f32-only.
        if comb_x_f16 || comb_dst_f16 {
            i += 1;
            continue;
        }
        if comb_n != 2 * n_half {
            i += 1;
            continue;
        }
        if mlp_f32_ranges_overlap(comb_x, comb_k, prod, n_half) {
            i += 1;
            continue;
        }

        let row_bytes = mlp_gate_up_row_bytes(comb_k, comb_scheme);
        if row_bytes == 0 {
            i += 1;
            continue;
        }
        let gate_w = comb_w;
        let up_w = comb_w + (n_half as usize) * row_bytes;

        // Down matmul must follow the SwiGLU/GeGLU product (validates MLP context).
        let down_mm_idx = mlp_find_forward(thunks, mul_idx, |t| {
            matches!(
                t,
                Thunk::DequantMatMulGguf {
                    x,
                    m: 1,
                    scheme: QuantScheme::GgufQ4K
                        | QuantScheme::GgufQ5_0
                        | QuantScheme::GgufQ6K,
                    ..
                } if *x == prod
            )
        });
        let Some(down_mm_idx) = down_mm_idx else {
            i += 1;
            continue;
        };
        let (down_w, down_dst, down_k, down_n, down_scheme) = match &thunks[down_mm_idx] {
            Thunk::DequantMatMulGguf {
                w_q,
                dst,
                k,
                n,
                scheme,
                ..
            } => (*w_q, *dst, *k, *n, *scheme),
            _ => unreachable!(),
        };

        // Full tail: down → add (Phi/Llama). Gemma3 inserts post_ffn RMSNorm
        // between down and add — fuse gate_up only when that norm is present.
        let down_add_tail = {
            let add_idx = mlp_find_forward(thunks, down_mm_idx, |t| {
                matches!(
                    t,
                    Thunk::BinaryFull {
                        lhs,
                        rhs,
                        op: BinaryOp::Add,
                        ..
                    } if *lhs == down_dst || *rhs == down_dst
                )
            });
            add_idx.map(|add_idx| {
                let (res_off, out_off) = match &thunks[add_idx] {
                    Thunk::BinaryFull { lhs, rhs, dst, .. } => {
                        let res = if *lhs == down_dst { *rhs } else { *lhs };
                        (res, *dst)
                    }
                    _ => unreachable!(),
                };
                (add_idx, res_off, out_off)
            })
        };

        let layer_until = down_mm_idx + 1;

        let mut dead_ok =
            mlp_value_dead_in_range(
                thunks,
                combined_mm_idx,
                combined_off,
                &[gate_narrow_idx, up_narrow_idx],
                layer_until,
            ) && mlp_value_dead_in_range(
                thunks,
                gate_narrow_idx,
                gate_src_off,
                &[act_idx],
                layer_until,
            ) && mlp_value_dead_in_range(thunks, up_narrow_idx, up_off, &[mul_idx], layer_until)
                && mlp_value_dead_in_range(thunks, act_idx, gate_src_off, &[mul_idx], layer_until);
        if let Some((add_idx, _, _)) = down_add_tail {
            dead_ok &=
                mlp_value_dead_in_range(thunks, down_mm_idx, down_dst, &[add_idx], layer_until);
        }
        let no_output_clash = ![combined_off, gate_src_off, up_off]
            .iter()
            .any(|o| output_offsets.contains(o));
        // GeGLU: offsets are reused across layers — extend liveness to the full
        // thunk list so a later layer cannot read `prod` / `combined` / `comb_x`
        // at the same arena slot before it is redefined.
        let gelu_graph_ok = if use_gelu {
            let prod_readers = [down_mm_idx];
            mlp_value_dead_in_range(thunks, mul_idx, prod, &prod_readers, thunks.len())
                && mlp_value_dead_in_range(
                    thunks,
                    combined_mm_idx,
                    combined_off,
                    &[gate_narrow_idx, up_narrow_idx],
                    thunks.len(),
                )
                && match mlp_last_writer(thunks, combined_mm_idx, comb_x) {
                    Ok(Some(w)) => mlp_unwritten_in_range(thunks, w + 1, mul_idx, comb_x),
                    _ => false,
                }
        } else {
            true
        };
        if !dead_ok || !no_output_clash || !gelu_graph_ok {
            i += 1;
            continue;
        }

        thunks[combined_mm_idx] = Thunk::Nop;
        thunks[gate_narrow_idx] = Thunk::Nop;
        thunks[up_narrow_idx] = Thunk::Nop;
        thunks[act_idx] = Thunk::Nop;
        thunks[mul_idx] = if use_gelu {
            Thunk::FusedMlpGateUpGelu {
                x: comb_x,
                gate_w,
                up_w,
                dst: prod,
                k: comb_k,
                n: n_half,
                scheme: comb_scheme,
            }
        } else {
            Thunk::FusedMlpGateUpSwiGLU {
                x: comb_x,
                gate_w,
                up_w,
                dst: prod,
                k: comb_k,
                n: n_half,
                scheme: comb_scheme,
                x_f16: false,
                dst_f16: false,
            }
        };
        if let Some((add_idx, res_off, out_off)) = down_add_tail {
            thunks[down_mm_idx] = Thunk::Nop;
            thunks[add_idx] = Thunk::FusedMlpDownResidual {
                x: prod,
                w: down_w,
                res: res_off,
                dst: out_off,
                k: down_k,
                n: down_n,
                scheme: down_scheme,
                x_f16: false,
                dst_f16: false,
                res_f16: false,
            };
        }
        FUSED_DECODE_MLP_BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if verbose {
            if down_add_tail.is_some() {
                eprintln!(
                    "[rlx-metal] fuse_decode_mlp_combined: gate_up {comb_scheme:?} k={comb_k} n={n_half} \
                     act={} down {down_scheme:?} — 7 dispatches → 2",
                    if use_gelu { "gelu" } else { "silu" }
                );
            } else {
                eprintln!(
                    "[rlx-metal] fuse_decode_mlp_combined: gate_up {comb_scheme:?} k={comb_k} n={n_half} \
                     act={} (post_ffn norm blocks down fuse) — 5 dispatches → 1",
                    if use_gelu { "gelu" } else { "silu" }
                );
            }
        }
        i += 1;
    }
}

/// Pattern-merge the per-layer decode SwiGLU MLP into two fused GEMV dispatches.
///
/// Recognizes (m == 1, gate/up Q4_K, down Q4_K or Q6_K), linked by dataflow:
///   gate = DequantMatMul(N, Wg);  up = DequantMatMul(N, Wu)
///   gate_act = silu(gate);  prod = mul(gate_act, up)
///   down = DequantMatMul(prod, Wd);  out = add(residual, down)
/// → `FusedMlpGateUpSwiGLU{N,Wg,Wu → prod}` + `FusedMlpDownResidual{prod,Wd,res → out}`.
/// Six dispatches collapse to two (plus the untouched rms_norm). The leading
/// rms_norm is NOT required by the matcher — only the matmul/elementwise core.
fn fuse_decode_mlp(thunks: &mut [Thunk], output_offsets: &std::collections::HashSet<usize>) {
    if rlx_ir::env::var("RLX_METAL_FUSE_DECODE").as_deref() == Some("0") {
        return;
    }
    use rlx_ir::quant::QuantScheme;
    let verbose = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");

    // Packed decode matmul (m == 1) writing `dst`? Return (x, w_q, dst, k, n, scheme).
    let as_packed_gate_up_mm = |t: &Thunk| -> Option<(usize, usize, usize, u32, u32, QuantScheme)> {
        if let Thunk::DequantMatMulGguf {
            x,
            w_q,
            dst,
            m,
            k,
            n,
            scheme,
            ..
        } = *t
        {
            if m == 1
                && matches!(
                    scheme,
                    QuantScheme::GgufQ4K
                        | QuantScheme::GgufQ5_0
                        | QuantScheme::GgufQ1_0
                        | QuantScheme::GgufQ2_0
                )
            {
                // Q1_0 fused kernels require k % 128 == 0 (block size).
                if matches!(scheme, QuantScheme::GgufQ1_0 | QuantScheme::GgufQ2_0)
                    && !k.is_multiple_of(128)
                {
                    return None;
                }
                if matches!(scheme, QuantScheme::GgufQ2_0)
                    && rlx_ir::env::flag("RLX_METAL_Q2_0_FUSED_DISABLE")
                {
                    return None;
                }
                return Some((x, w_q, dst, k, n, scheme));
            }
        }
        None
    };

    let n_thunks = thunks.len();
    let mut i = 0;
    while i < n_thunks {
        // Anchor on the SwiGLU `mul`.
        let (mul_lhs, mul_rhs, prod) = match &thunks[i] {
            Thunk::BinaryFull {
                lhs,
                rhs,
                dst,
                op: BinaryOp::Mul,
                ..
            } => (*lhs, *rhs, *dst),
            _ => {
                i += 1;
                continue;
            }
        };
        let mul_idx = i;

        // Which mul input is silu(gate) or gelu(gate)?
        let is_silu = |t: &Thunk| {
            matches!(
                t,
                Thunk::ActivationInPlace {
                    act: Activation::Silu,
                    ..
                } | Thunk::ActivationOut {
                    act: Activation::Silu,
                    ..
                }
            )
        };
        let is_gelu = |t: &Thunk| {
            matches!(
                t,
                Thunk::ActivationInPlace {
                    act: Activation::GeluApprox,
                    ..
                } | Thunk::ActivationOut {
                    act: Activation::GeluApprox,
                    ..
                } | Thunk::GeluApproxOut { .. }
                    | Thunk::GeluApproxHost { .. }
            )
        };
        let act_writer = |gate_act: usize| -> Option<(usize, bool)> {
            match mlp_last_writer(thunks, mul_idx, gate_act) {
                Ok(Some(idx)) if is_silu(&thunks[idx]) => Some((idx, false)),
                Ok(Some(idx)) if is_gelu(&thunks[idx]) => Some((idx, true)),
                _ => None,
            }
        };
        let (gate_act_off, up_off, act_idx, use_gelu) =
            if let Some((idx, gelu)) = act_writer(mul_lhs) {
                (mul_lhs, mul_rhs, idx, gelu)
            } else if let Some((idx, gelu)) = act_writer(mul_rhs) {
                (mul_rhs, mul_lhs, idx, gelu)
            } else {
                i += 1;
                continue;
            };

        if use_gelu && rlx_ir::env::var("RLX_METAL_FUSE_DECODE_GELU").as_deref() == Some("0") {
            i += 1;
            continue;
        }

        let gate_src_off = match &thunks[act_idx] {
            Thunk::ActivationInPlace { data, .. } => *data,
            Thunk::ActivationOut { src, .. } => *src,
            Thunk::GeluApproxOut { src, .. } | Thunk::GeluApproxHost { src, .. } => *src,
            _ => {
                i += 1;
                continue;
            }
        };
        let gate_producer = match mlp_last_writer(thunks, act_idx, gate_src_off) {
            Ok(Some(idx)) => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let (copy_idx, gate_mm_idx, gate_mm_off) = match &thunks[gate_producer] {
            Thunk::Copy { src, .. } => match mlp_last_writer(thunks, gate_producer, *src) {
                Ok(Some(gm)) if as_packed_gate_up_mm(&thunks[gm]).is_some() => {
                    (Some(gate_producer), gm, *src)
                }
                _ => {
                    i += 1;
                    continue;
                }
            },
            t if as_packed_gate_up_mm(t).is_some() => (None, gate_producer, gate_src_off),
            _ => {
                i += 1;
                continue;
            }
        };

        let (gate_x, gate_w, _g_dst, gate_k, gate_n, gate_scheme) =
            as_packed_gate_up_mm(&thunks[gate_mm_idx]).unwrap();
        let (gate_x_f16, gate_dst_f16) = match &thunks[gate_mm_idx] {
            Thunk::DequantMatMulGguf { x_f16, dst_f16, .. } => (*x_f16, *dst_f16),
            _ => (false, false),
        };
        // Non-Q1 fused MLP kernels are f32-only; under AMP fall back to
        // separate DequantMatMulGguf (which honors x_f16/dst_f16).
        if (gate_x_f16 || gate_dst_f16)
            && !matches!(gate_scheme, QuantScheme::GgufQ1_0 | QuantScheme::GgufQ2_0)
        {
            i += 1;
            continue;
        }
        if mlp_f32_ranges_overlap(gate_x, gate_k, prod, gate_n) {
            i += 1;
            continue;
        }

        // up matmul: last writer of `up_off`, must share x/k/n/scheme.
        let up_mm_idx = match mlp_last_writer(thunks, mul_idx, up_off) {
            Ok(Some(idx)) => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let Some((up_x, up_w, _u_dst, up_k, up_n, up_scheme)) =
            as_packed_gate_up_mm(&thunks[up_mm_idx])
        else {
            i += 1;
            continue;
        };
        if up_x != gate_x || up_k != gate_k || up_n != gate_n || up_scheme != gate_scheme {
            i += 1;
            continue;
        }

        // down matmul: first forward consumer of `prod` that is a Q4_K/Q6_K
        // m==1 matmul reading prod as its activation.
        let down_mm_idx = mlp_find_forward(thunks, mul_idx, |t| match t {
            Thunk::DequantMatMulGguf {
                x,
                m: 1,
                scheme: QuantScheme::GgufQ4K | QuantScheme::GgufQ5_0 | QuantScheme::GgufQ6K,
                ..
            } if *x == prod => true,
            Thunk::DequantMatMulGguf {
                x,
                m: 1,
                k,
                scheme: QuantScheme::GgufQ1_0 | QuantScheme::GgufQ2_0,
                ..
            } if *x == prod && k.is_multiple_of(128) => true,
            _ => false,
        });
        let Some(down_mm_idx) = down_mm_idx else {
            i += 1;
            continue;
        };
        let (down_w, down_dst, down_k, down_n, down_scheme, down_x_f16, down_dst_f16) =
            match &thunks[down_mm_idx] {
                Thunk::DequantMatMulGguf {
                    w_q,
                    dst,
                    k,
                    n,
                    scheme,
                    x_f16,
                    dst_f16,
                    ..
                } => (*w_q, *dst, *k, *n, *scheme, *x_f16, *dst_f16),
                _ => unreachable!(),
            };
        if matches!(down_scheme, QuantScheme::GgufQ2_0)
            && rlx_ir::env::flag("RLX_METAL_Q2_0_FUSED_DISABLE")
        {
            i += 1;
            continue;
        }
        if (down_x_f16 || down_dst_f16)
            && !matches!(down_scheme, QuantScheme::GgufQ1_0 | QuantScheme::GgufQ2_0)
        {
            i += 1;
            continue;
        }

        // add: first forward consumer of `down_dst` that is an elementwise Add.
        let add_idx = mlp_find_forward(
            thunks,
            down_mm_idx,
            |t| matches!(t, Thunk::BinaryFull { lhs, rhs, op: BinaryOp::Add, .. } if *lhs == down_dst || *rhs == down_dst),
        );
        let Some(add_idx) = add_idx else {
            i += 1;
            continue;
        };
        let (res_off, out_off, add_f16) = match &thunks[add_idx] {
            Thunk::BinaryFull {
                lhs, rhs, dst, dt, ..
            } => {
                let res = if *lhs == down_dst { *rhs } else { *lhs };
                (res, *dst, matches!(dt, HalfFlag::F16))
            }
            _ => unreachable!(),
        };

        // Liveness within this layer (offsets are reused across layers).
        let layer_until = down_mm_idx + 1;
        let dead_ok =
            mlp_value_dead_in_range(
                thunks,
                gate_mm_idx,
                gate_mm_off,
                &[copy_idx.unwrap_or(act_idx)],
                layer_until,
            ) && mlp_value_dead_in_range(thunks, up_mm_idx, up_off, &[mul_idx], layer_until)
                && mlp_value_dead_in_range(thunks, act_idx, gate_act_off, &[mul_idx], layer_until)
                && mlp_value_dead_in_range(thunks, down_mm_idx, down_dst, &[add_idx], layer_until);
        // None of the dropped intermediates may be a graph output.
        let no_output_clash = ![gate_mm_off, up_off, gate_act_off, down_dst]
            .iter()
            .any(|o| output_offsets.contains(o));
        let gelu_graph_ok = if use_gelu {
            mlp_value_dead_in_range(thunks, mul_idx, prod, &[down_mm_idx, add_idx], thunks.len())
                && match mlp_last_writer(thunks, gate_mm_idx, gate_x) {
                    Ok(Some(w)) => mlp_unwritten_in_range(thunks, w + 1, mul_idx, gate_x),
                    _ => false,
                }
        } else {
            true
        };
        if !dead_ok || !no_output_clash || !gelu_graph_ok {
            i += 1;
            continue;
        }

        // Commit: gate/up/silu/copy/down → Nop; mul → fused1; add → fused2.
        thunks[gate_mm_idx] = Thunk::Nop;
        thunks[up_mm_idx] = Thunk::Nop;
        thunks[act_idx] = Thunk::Nop;
        thunks[down_mm_idx] = Thunk::Nop;
        if let Some(c) = copy_idx {
            thunks[c] = Thunk::Nop;
        }
        thunks[mul_idx] = if use_gelu {
            Thunk::FusedMlpGateUpGelu {
                x: gate_x,
                gate_w,
                up_w,
                dst: prod,
                k: gate_k,
                n: gate_n,
                scheme: gate_scheme,
            }
        } else {
            Thunk::FusedMlpGateUpSwiGLU {
                x: gate_x,
                gate_w,
                up_w,
                dst: prod,
                k: gate_k,
                n: gate_n,
                scheme: gate_scheme,
                x_f16: gate_x_f16,
                dst_f16: gate_dst_f16,
            }
        };
        thunks[add_idx] = Thunk::FusedMlpDownResidual {
            x: prod,
            w: down_w,
            res: res_off,
            dst: out_off,
            k: down_k,
            n: down_n,
            scheme: down_scheme,
            x_f16: down_x_f16,
            // Residual add dst drives residual-stream dtype under AMP.
            dst_f16: add_f16,
            res_f16: add_f16,
        };
        FUSED_DECODE_MLP_BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if verbose {
            eprintln!(
                "[rlx-metal] fuse_decode_mlp: block fused (gate/up {gate_scheme:?} k={gate_k} n={gate_n}, \
                 down {down_scheme:?} k={down_k} n={down_n}, act={}) — 6 dispatches → 2",
                if use_gelu { "gelu" } else { "silu" }
            );
        }
        i += 1;
    }
}

/// Fuse `out = rms_norm(x + residual)` (decode residual stream → pre-norm).
///
/// Pattern: `BinaryFull(Add, x, res → tmp)` then optional `Copy` then
/// `RmsNorm(tmp → out)`. Requires one add operand to come from a projection
/// (`DequantMatMul` / `Sgemm` / fused MLP down) so we don't fuse unrelated adds.
fn fuse_residual_rms_norm(thunks: &mut [Thunk], output_offsets: &std::collections::HashSet<usize>) {
    // Default on. Opt out: RLX_METAL_FUSE_RESIDUAL_RMS=0. Liveness requires the
    // add dst to die at the rms — Qwen/Bonsai post-attn (`h+=attn; n=rms(h);
    // h+=ffn(n)`) correctly refuses to fuse. IR-emitted FusedResidualRmsNorm
    // still lowers via the encode path either way.
    if rlx_ir::env::var("RLX_METAL_FUSE_RESIDUAL_RMS").as_deref() == Some("0")
        || rlx_ir::env::var("RLX_METAL_FUSE_DECODE").as_deref() == Some("0")
    {
        return;
    }
    let verbose = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");
    let n_thunks = thunks.len();
    let mut fused = 0usize;
    let mut i = 0;

    let as_add =
        |t: &Thunk, expect_dst: usize, expect_len: u32, dt: HalfFlag| -> Option<(usize, usize)> {
            match t {
                Thunk::BinaryFull {
                    lhs,
                    rhs,
                    dst,
                    len,
                    op: BinaryOp::Add,
                    dt: add_dt,
                } if *dst == expect_dst && *add_dt == dt && *len == expect_len => {
                    Some((*lhs, *rhs))
                }
                _ => None,
            }
        };

    // Projection-branch heuristic (not a doc comment on the closure).
    let last_writer_skip_opaque = |thunks: &[Thunk], before: usize, off: usize| -> Option<usize> {
        for j in (0..before).rev() {
            match mlp_io(&thunks[j]) {
                None => continue,
                Some((_, writes)) if writes.contains(&off) => return Some(j),
                _ => {}
            }
        }
        None
    };

    // True if `off` was produced by a residual-branch projection (or a reshape
    // Copy of one). The other add input is the residual stream.
    let is_proj_branch = |thunks: &[Thunk], mut before: usize, off: usize| -> bool {
        let mut cur = off;
        for _ in 0..4 {
            let Some(idx) = last_writer_skip_opaque(thunks, before, cur) else {
                return false;
            };
            match &thunks[idx] {
                Thunk::DequantMatMulGguf { .. }
                | Thunk::Sgemm { .. }
                | Thunk::BatchedSgemm { .. }
                | Thunk::FusedMmBiasAct { .. }
                | Thunk::FusedMlpDownResidual { .. } => return true,
                Thunk::Copy { src, dst, .. } if *dst == cur => {
                    cur = *src;
                    before = idx;
                }
                Thunk::BiasAdd { src, dst, .. } if *dst == cur => {
                    cur = *src;
                    before = idx;
                }
                _ => return false,
            }
        }
        false
    };

    while i < n_thunks {
        let (mut src, g_off, b_off, dst, rows, h, eps, dt) = match &thunks[i] {
            Thunk::RmsNorm {
                src,
                g,
                b,
                dst,
                rows,
                h,
                eps,
                dt,
            } => (*src, *g, *b, *dst, *rows, *h, *eps, *dt),
            _ => {
                i += 1;
                continue;
            }
        };
        // Residual-stream norms are hidden_size-wide (Bonsai 5120). Skip
        // head/state norms (e.g. ssm_norm h=128) — those belong to gated-norm.
        // Prefill (rows>1) still mis-pairs under active-extent scaling; fuse
        // decode residuals only (rows==1) until that path is proven.
        if h < 1024 || rows != 1 || !matches!(dt, HalfFlag::F32) {
            i += 1;
            continue;
        }
        let expect_len = rows.saturating_mul(h);
        if expect_len == 0 || output_offsets.contains(&src) {
            i += 1;
            continue;
        }

        // Optional reshape Copy between add and rms.
        let mut copy_i = None;
        if let Ok(Some(idx)) = mlp_last_writer(thunks, i, src) {
            if let Thunk::Copy {
                src: csrc,
                dst: cdst,
                len,
                dt: cdt,
            } = &thunks[idx]
            {
                if *cdst == src
                    && *cdt == dt
                    && *len == expect_len
                    && !output_offsets.contains(csrc)
                {
                    copy_i = Some(idx);
                    src = *csrc;
                }
            }
        }

        let add_before = copy_i.unwrap_or(i);
        let add_i = match mlp_last_writer(thunks, add_before, src) {
            Ok(Some(idx)) => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        // Add should sit close to the rms (reshape copy + a few nops only).
        let gap = (add_i + 1..add_before)
            .filter(|&j| !matches!(thunks[j], Thunk::Nop))
            .count();
        if gap > 2 {
            i += 1;
            continue;
        }
        let Some((x, res)) = as_add(&thunks[add_i], src, expect_len, dt) else {
            i += 1;
            continue;
        };
        if !(is_proj_branch(thunks, add_i, x) || is_proj_branch(thunks, add_i, res)) {
            i += 1;
            continue;
        }

        // In-place / aliased write would corrupt the second pass over x+res.
        if mlp_f32_ranges_overlap(dst, expect_len, x, expect_len)
            || mlp_f32_ranges_overlap(dst, expect_len, res, expect_len)
        {
            i += 1;
            continue;
        }

        // Critical: add dst must die at the rms — not merely between add and rms.
        // Post-attn residual (`h+=attn; n=rms(h); h+=ffn(n)`) keeps `h` live for
        // the FFN residual; fusing would replace the add and poison the stream.
        let add_readers: Vec<usize> = match copy_i {
            Some(c) => vec![c, i],
            None => vec![i],
        };
        if !mlp_value_dead_in_range(thunks, add_i, src, &add_readers, thunks.len()) {
            i += 1;
            continue;
        }
        if let Some(c) = copy_i {
            let copy_dst = match &thunks[c] {
                Thunk::Copy { dst, .. } => *dst,
                _ => src,
            };
            if !mlp_value_dead_in_range(thunks, c, copy_dst, &[i], thunks.len()) {
                i += 1;
                continue;
            }
        }

        thunks[add_i] = Thunk::Nop;
        if let Some(c) = copy_i {
            thunks[c] = Thunk::Nop;
        }
        thunks[i] = Thunk::FusedResidualRmsNorm {
            x,
            res,
            bias: 0,
            g: g_off,
            b: b_off,
            out: dst,
            rows,
            h,
            eps,
            has_bias: false,
            dt,
        };
        fused += 1;
        FUSED_RESIDUAL_RMS_BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if verbose {
            eprintln!("[rlx-metal] fuse_residual_rms_norm: rows={rows} h={h} — add(+copy)+rms → 1");
        }
        i += 1;
    }
    if verbose && fused > 0 {
        eprintln!("[rlx-metal] fuse_residual_rms_norm: {fused} blocks fused");
    }
}

/// Fuse GDN gated norm: `out = rms_norm(scan) * silu(z)`.
///
/// Anchors on `Mul`; one operand is an `RmsNorm` result, the other is a
/// `Silu` (optionally after a reshape `Copy`). Collapses silu + rms + mul
/// (+ copy) into one dispatch — 48× per Bonsai decode step.
fn fuse_gdn_gated_norm(thunks: &mut [Thunk], output_offsets: &std::collections::HashSet<usize>) {
    if rlx_ir::env::var("RLX_METAL_FUSE_DECODE").as_deref() == Some("0")
        || rlx_ir::env::var("RLX_METAL_FUSE_GDN_NORM").as_deref() == Some("0")
    {
        return;
    }
    let verbose = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");
    let n_thunks = thunks.len();
    let mut fused = 0usize;
    let mut i = 0;
    while i < n_thunks {
        let (mul_lhs, mul_rhs, mul_dst, mul_len, mul_dt) = match &thunks[i] {
            Thunk::BinaryFull {
                lhs,
                rhs,
                dst,
                len,
                op: BinaryOp::Mul,
                dt,
            } => (*lhs, *rhs, *dst, *len, *dt),
            _ => {
                i += 1;
                continue;
            }
        };
        // Fused kernel is f32-only today (`encode_rms_norm_mul_silu` ignores dt).
        if matches!(mul_dt, HalfFlag::F16) {
            i += 1;
            continue;
        }

        // Identify which mul input is rms_norm output.
        let try_rms = |off: usize| -> Option<(usize, usize, usize, usize, u32, u32, f32)> {
            match mlp_last_writer(thunks, i, off) {
                Ok(Some(idx)) => match &thunks[idx] {
                    Thunk::RmsNorm {
                        src,
                        g,
                        b,
                        dst,
                        rows,
                        h,
                        eps,
                        dt,
                    } if *dst == off && *dt == mul_dt => Some((idx, *src, *g, *b, *rows, *h, *eps)),
                    _ => None,
                },
                _ => None,
            }
        };

        let (rms_i, x, g_off, b_off, rows, h, eps, z_side) = if let Some(t) = try_rms(mul_lhs) {
            (t.0, t.1, t.2, t.3, t.4, t.5, t.6, mul_rhs)
        } else if let Some(t) = try_rms(mul_rhs) {
            (t.0, t.1, t.2, t.3, t.4, t.5, t.6, mul_lhs)
        } else {
            i += 1;
            continue;
        };
        if rows.saturating_mul(h) != mul_len {
            i += 1;
            continue;
        }
        let rms_dst = match &thunks[rms_i] {
            Thunk::RmsNorm { dst, .. } => *dst,
            _ => {
                i += 1;
                continue;
            }
        };
        if output_offsets.contains(&rms_dst) {
            i += 1;
            continue;
        }

        // z side: Silu inplace, optionally after Copy.
        let silu_i = match mlp_last_writer(thunks, i, z_side) {
            Ok(Some(idx)) => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let (z_src, copy_i) = match &thunks[silu_i] {
            Thunk::ActivationInPlace {
                data,
                act: Activation::Silu,
                dt,
                ..
            } if *data == z_side && *dt == mul_dt => {
                // Optional copy that produced `data` before silu.
                let copy = match mlp_last_writer(thunks, silu_i, z_side) {
                    Ok(Some(cidx)) => match &thunks[cidx] {
                        Thunk::Copy {
                            src,
                            dst,
                            len,
                            dt: cdt,
                        } if *dst == z_side && *cdt == mul_dt && *len == mul_len => {
                            Some((cidx, *src))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                match copy {
                    Some((cidx, src)) => (src, Some(cidx)),
                    None => (z_side, None), // silu was in-place on original z
                }
            }
            Thunk::ActivationOut {
                src,
                dst,
                act: Activation::Silu,
                dt,
                ..
            } if *dst == z_side && *dt == mul_dt => (*src, None),
            _ => {
                i += 1;
                continue;
            }
        };

        // Only scan through this mul — a full-graph scan hits opaque GDN/conv
        // thunks in later layers and falsely rejects (same footgun as MLP fuse).
        if !mlp_value_dead_in_range(thunks, rms_i, rms_dst, &[i], i + 1) {
            i += 1;
            continue;
        }
        if !mlp_value_dead_in_range(thunks, silu_i, z_side, &[i], i + 1) {
            i += 1;
            continue;
        }
        if let Some(c) = copy_i {
            if output_offsets.contains(&z_src) {
                i += 1;
                continue;
            }
            // Copy dst (= z_side before silu) — silu reads/writes it; already checked via silu.
            let _ = c;
        }

        if let Some(c) = copy_i {
            thunks[c] = Thunk::Nop;
        }
        thunks[silu_i] = Thunk::Nop;
        thunks[rms_i] = Thunk::Nop;
        thunks[i] = Thunk::FusedRmsNormMulSilu {
            x,
            g: g_off,
            b: b_off,
            z: z_src,
            out: mul_dst,
            rows,
            h,
            eps,
            dt: mul_dt,
        };
        fused += 1;
        if verbose {
            eprintln!("[rlx-metal] fuse_gdn_gated_norm: rows={rows} h={h} — silu+rms+mul → 1");
        }
        i += 1;
    }
    if verbose && fused > 0 {
        eprintln!("[rlx-metal] fuse_gdn_gated_norm: {fused} blocks fused");
    }
}

/// Fuse GDN depthwise conv BSC dance into one kernel:
/// `Transpose(BSC→BCW) → [Copy] → Conv2D(depthwise) → [Copy] →
/// Transpose(→BSC) [→ Silu]`. Reshape often becomes a Nop/elided Copy.
fn fuse_depthwise_conv1d_bsc(
    thunks: &mut [Thunk],
    output_offsets: &std::collections::HashSet<usize>,
) {
    if rlx_ir::env::var("RLX_METAL_FUSE_DECODE").as_deref() == Some("0")
        || rlx_ir::env::var("RLX_METAL_FUSE_DEPTHWISE").as_deref() == Some("0")
    {
        return;
    }
    let verbose = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");
    let n_thunks = thunks.len();
    let mut fused = 0usize;
    let mut i = 0;
    while i < n_thunks {
        let (conv_src, weight, conv_dst, batch, c_in, width, out_seq, kw) = match &thunks[i] {
            Thunk::Conv2D {
                src,
                weight,
                dst,
                n,
                c_in,
                h: 1,
                w,
                c_out,
                h_out: 1,
                w_out,
                kh: 1,
                kw,
                sh: 1,
                sw: 1,
                ph: 0,
                pw: 0,
                dh: 1,
                dw: 1,
                groups,
            } if *groups == *c_in && *c_in == *c_out && *kw >= 1 && *w_out == 1 => {
                // Decode-only (out_seq==1): prefill keeps the generic Conv2D path.
                (*src, *weight, *dst, *n, *c_in, *w, *w_out, *kw)
            }
            _ => {
                i += 1;
                continue;
            }
        };

        // Look back past Nops for Copy(→conv_src) and/or Transpose(→…).
        let mut copy_in_i: Option<usize> = None;
        let mut bcw = conv_src;
        let mut cursor = i;
        // Optional reshape Copy into NCHW.
        if let Ok(Some(idx)) = mlp_last_writer(thunks, cursor, bcw) {
            match &thunks[idx] {
                Thunk::Copy {
                    src,
                    dst,
                    len,
                    dt: HalfFlag::F32,
                } if *dst == bcw && *len == batch.saturating_mul(c_in).saturating_mul(width) => {
                    copy_in_i = Some(idx);
                    bcw = *src;
                    cursor = idx;
                }
                Thunk::Nop => {}
                _ => {}
            }
        }
        let transpose_in_i = match mlp_last_writer(thunks, cursor, bcw) {
            Ok(Some(idx)) => idx,
            _ => {
                i += 1;
                continue;
            }
        };
        let bsc_src = match &thunks[transpose_in_i] {
            Thunk::Transpose {
                src,
                dst,
                out_dims,
                in_strides,
                ..
            } if *dst == bcw
                && out_dims.len() == 3
                && in_strides.len() == 3
                && out_dims[0] == batch
                && out_dims[1] == c_in
                && out_dims[2] == width
                && in_strides[0] == width.saturating_mul(c_in)
                && in_strides[1] == 1
                && in_strides[2] == c_in =>
            {
                *src
            }
            _ => {
                i += 1;
                continue;
            }
        };
        if output_offsets.contains(&bcw) || output_offsets.contains(&conv_src) {
            i += 1;
            continue;
        }

        // Look forward: optional Copy, then Transpose BCS→BSC.
        let mut copy_out_i: Option<usize> = None;
        let mut bcs = conv_dst;
        let mut scan_from = i + 1;
        if let Some(idx) = (scan_from..n_thunks).find(|&j| {
            !matches!(thunks[j], Thunk::Nop)
                && matches!(
                    &thunks[j],
                    Thunk::Copy {
                        src,
                        dt: HalfFlag::F32,
                        ..
                    } if *src == conv_dst
                )
        }) {
            match &thunks[idx] {
                Thunk::Copy { dst, len, .. }
                    if *len == batch.saturating_mul(c_in).saturating_mul(out_seq) =>
                {
                    copy_out_i = Some(idx);
                    bcs = *dst;
                    scan_from = idx + 1;
                }
                _ => {}
            }
        }
        let transpose_out_i = match (scan_from..n_thunks).find(|&j| {
            !matches!(thunks[j], Thunk::Nop)
                && matches!(&thunks[j], Thunk::Transpose { src, .. } if *src == bcs)
        }) {
            Some(idx) => idx,
            None => {
                // Direct Transpose reading conv_dst (reshape elided).
                match (i + 1..n_thunks).find(|&j| {
                    !matches!(thunks[j], Thunk::Nop)
                        && matches!(
                            &thunks[j],
                            Thunk::Transpose { src, .. } if *src == conv_dst
                        )
                }) {
                    Some(idx) => {
                        bcs = conv_dst;
                        idx
                    }
                    None => {
                        i += 1;
                        continue;
                    }
                }
            }
        };
        let bsc_dst = match &thunks[transpose_out_i] {
            Thunk::Transpose {
                dst,
                out_dims,
                in_strides,
                ..
            } if out_dims.len() == 3
                && in_strides.len() == 3
                && out_dims[0] == batch
                && out_dims[1] == out_seq
                && out_dims[2] == c_in
                && in_strides[0] == c_in.saturating_mul(out_seq)
                && in_strides[1] == 1
                && in_strides[2] == out_seq =>
            {
                *dst
            }
            _ => {
                i += 1;
                continue;
            }
        };
        if output_offsets.contains(&bcs) && bcs != conv_dst {
            i += 1;
            continue;
        }
        if output_offsets.contains(&conv_dst) && copy_out_i.is_some() {
            i += 1;
            continue;
        }

        // SiLU may be in-place on BSC, or Copy→Silu (and may sit after an
        // unrelated conv-state narrow).
        let mut silu_i: Option<usize> = None;
        let mut silu_copy_i: Option<usize> = None;
        let mut silu_dst = bsc_dst;
        for j in transpose_out_i + 1..(transpose_out_i + 1 + 32).min(n_thunks) {
            match &thunks[j] {
                Thunk::Nop => continue,
                Thunk::ActivationInPlace {
                    data,
                    act: Activation::Silu,
                    ..
                } if *data == silu_dst => {
                    silu_i = Some(j);
                    break;
                }
                Thunk::ActivationOut {
                    src,
                    dst,
                    act: Activation::Silu,
                    ..
                } if *src == silu_dst => {
                    silu_dst = *dst;
                    silu_i = Some(j);
                    break;
                }
                Thunk::Copy {
                    src,
                    dst,
                    len,
                    dt: HalfFlag::F32,
                } if *src == bsc_dst
                    && silu_copy_i.is_none()
                    && *len == batch.saturating_mul(out_seq).saturating_mul(c_in) =>
                {
                    silu_copy_i = Some(j);
                    silu_dst = *dst;
                }
                Thunk::Narrow { .. } | Thunk::SplitLastAxis { .. } => continue,
                _ if silu_copy_i.is_none() => continue,
                _ => break,
            }
        }
        let do_silu = silu_i.is_some();
        let final_dst = if do_silu { silu_dst } else { bsc_dst };

        if width != out_seq.saturating_add(kw).saturating_sub(1) {
            i += 1;
            continue;
        }

        // Liveness: intermediates only used by the fused chain.
        let until = silu_i.unwrap_or(transpose_out_i) + 1;
        let mut allowed_bcw = vec![i];
        if let Some(c) = copy_in_i {
            allowed_bcw.push(c);
        }
        if !mlp_value_dead_in_range(thunks, transpose_in_i, bcw, &allowed_bcw, until) {
            i += 1;
            continue;
        }
        if let Some(c) = copy_in_i {
            if !mlp_value_dead_in_range(thunks, c, conv_src, &[i], until) {
                i += 1;
                continue;
            }
        }
        let mut allowed_conv = Vec::new();
        if let Some(c) = copy_out_i {
            allowed_conv.push(c);
        } else {
            allowed_conv.push(transpose_out_i);
        }
        if !mlp_value_dead_in_range(thunks, i, conv_dst, &allowed_conv, until) {
            i += 1;
            continue;
        }
        if let Some(c) = copy_out_i {
            if !mlp_value_dead_in_range(thunks, c, bcs, &[transpose_out_i], until) {
                i += 1;
                continue;
            }
        }

        thunks[transpose_in_i] = Thunk::Nop;
        if let Some(c) = copy_in_i {
            thunks[c] = Thunk::Nop;
        }
        if let Some(c) = copy_out_i {
            thunks[c] = Thunk::Nop;
        }
        thunks[transpose_out_i] = Thunk::Nop;
        if let Some(c) = silu_copy_i {
            thunks[c] = Thunk::Nop;
        }
        if let Some(si) = silu_i {
            thunks[si] = Thunk::Nop;
        }
        // If we fused silu onto a copy destination, the original BSC slot is
        // dead for consumers that read the silu output — but any consumer of
        // the pre-silu BSC must have been only the copy we nop'd.
        thunks[i] = Thunk::FusedDepthwiseConv1dBsc {
            src: bsc_src,
            weight,
            dst: final_dst,
            batch,
            width,
            out_seq,
            channels: c_in,
            k: kw,
            silu: do_silu,
        };
        fused += 1;
        if verbose {
            eprintln!(
                "[rlx-metal] fuse_depthwise_conv1d_bsc: B={batch} W={width} out={out_seq} \
                 C={c_in} k={kw} silu={do_silu} — chain → 1"
            );
        }
        i += 1;
    }
    if verbose && fused > 0 {
        eprintln!("[rlx-metal] fuse_depthwise_conv1d_bsc: {fused} blocks fused");
    }
}

/// Merge disjoint last-axis `Narrow` thunks that share a source into `SplitLastAxis`.
fn fuse_narrow_clusters(thunks: &mut [Thunk]) {
    use std::collections::HashMap;

    #[derive(Hash, PartialEq, Eq, Clone, Copy)]
    struct NarrowKey {
        src: usize,
        outer: u32,
        src_axis: u32,
        dt: u8,
    }

    let mut groups: HashMap<NarrowKey, Vec<(usize, usize, u32, u32)>> = HashMap::new();
    for (i, t) in thunks.iter().enumerate() {
        let Thunk::Narrow {
            src,
            dst,
            outer,
            src_axis,
            start,
            len,
            dt,
        } = t
        else {
            continue;
        };
        let key = NarrowKey {
            src: *src,
            outer: *outer,
            src_axis: *src_axis,
            dt: match dt {
                HalfFlag::F32 => 0,
                HalfFlag::F16 => 1,
            },
        };
        groups.entry(key).or_default().push((i, *dst, *start, *len));
    }

    let mut groups_fused = 0usize;
    let mut narrows_fused = 0usize;
    for (key, mut items) in groups {
        if items.len() < 2 || key.dt != 0 {
            continue;
        }
        let meta: Vec<(u32, u32)> = items.iter().map(|(_, _, s, l)| (*s, *l)).collect();
        if !narrow_segments_partition(key.src_axis, &meta) {
            continue;
        }
        items.sort_by_key(|(i, _, _, _)| *i);
        let dt = HalfFlag::F32;
        let segments: Vec<(usize, u32, u32)> =
            items.iter().map(|(_, d, s, l)| (*d, *s, *l)).collect();
        let n = items.len();
        let first = items[0].0;
        thunks[first] = Thunk::SplitLastAxis {
            src: key.src,
            outer: key.outer,
            src_axis: key.src_axis,
            dt,
            segments,
        };
        for (i, _, _, _) in items.into_iter().skip(1) {
            thunks[i] = Thunk::Nop;
        }
        groups_fused += 1;
        narrows_fused += n;
    }
    let verbose = rlx_ir::env::var("RLX_VERBOSE")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(0)
        >= 1;
    if verbose && groups_fused > 0 {
        eprintln!(
            "[rlx-metal] fuse_narrow_clusters: {groups_fused} split groups ({narrows_fused} narrows merged)"
        );
    }
}

/// Read-offsets for Metal Thunks. Conservative: variants not enumerated
/// produce an empty list, which makes the Narrow→Rope fusion (above)
/// see read_count == 0 and bail. Safer than enumerating wrong.
fn metal_thunk_read_offsets(t: &Thunk) -> Vec<usize> {
    match t {
        Thunk::Sgemm { a, b, .. } => vec![*a, *b],
        Thunk::BatchedSgemm { a, b, .. } => vec![*a, *b],
        Thunk::FusedMmBiasAct { a, w, bias, .. } => vec![*a, *w, *bias],
        Thunk::BinaryFull { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::FusedBinaryActivation { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::FusedTernaryActivation {
            lhs, rhs0, rhs1, ..
        } => vec![*lhs, *rhs0, *rhs1],
        Thunk::BinaryBroadcast { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::ActivationInPlace { data, .. } => vec![*data],
        Thunk::ActivationOut { src, .. } => vec![*src],
        Thunk::GeluApproxOut { src, .. } | Thunk::GeluApproxHost { src, .. } => vec![*src],
        Thunk::LayerNorm { src, g, b, .. } | Thunk::GroupNorm { src, g, b, .. } => {
            vec![*src, *g, *b]
        }
        Thunk::ResizeNearest2x { src, .. } => vec![*src],
        Thunk::RmsNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::FusedResidualLN {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::FusedResidualRmsNorm {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::AdaLayerNorm {
            x, scale, shift, ..
        } => vec![*x, *scale, *shift],
        Thunk::GatedResidual { x, y, gate, .. } => vec![*x, *y, *gate],
        Thunk::AdaLayerNormBackward { x, scale, dy, .. } => vec![*x, *scale, *dy],
        Thunk::GatedResidualBackward { y, gate, dy, .. } => vec![*y, *gate, *dy],
        Thunk::FusedRmsNormMulSilu { x, g, b, z, .. } => vec![*x, *g, *b, *z],
        Thunk::FusedDepthwiseConv1dBsc { src, weight, .. } => vec![*src, *weight],
        Thunk::Conv3d { src, weight, .. } | Thunk::ConvTranspose3d { src, weight, .. } => {
            vec![*src, *weight]
        }
        Thunk::ReluBackward { x, dy, .. } | Thunk::ActivationBackward { x, dy, .. } => {
            vec![*x, *dy]
        }
        Thunk::ComplexNormSq { src, .. } | Thunk::ConjugateC64 { src, .. } => vec![*src],
        Thunk::ComplexNormSqBackward { z, g, .. } => vec![*z, *g],
        Thunk::FftButterflyStage {
            state,
            gate,
            rev,
            tw_re,
            tw_im,
            ..
        } => vec![*state, *gate, *rev, *tw_re, *tw_im],
        Thunk::Softmax { data, .. } => vec![*data],
        Thunk::SoftmaxCrossEntropyDense {
            logits, targets, ..
        } => vec![*logits, *targets],
        Thunk::SoftmaxCrossEntropyWithLogits { logits, labels, .. } => vec![*logits, *labels],
        Thunk::SoftmaxCrossEntropyBackward {
            logits,
            labels,
            d_loss,
            ..
        } => vec![*logits, *labels, *d_loss],
        Thunk::Cumsum { src, .. } => vec![*src],
        Thunk::CumScan { src, .. } => vec![*src],
        // SPD host op reads all its operand slots. Reported so the
        // Narrow→Rope read-count fusion never treats a slot an SPD op
        // consumes as unused.
        Thunk::SpdHost { inputs, .. } => inputs.iter().map(|(off, _, _)| *off).collect(),
        // A raw-GPU custom op reads all its operand slots on-GPU (no sync
        // barrier), so report them like SpdHost — otherwise the Narrow→Rope
        // read-count fusion could elide a producer of one of its inputs.
        Thunk::CustomGpuOp { inputs, .. } => inputs.iter().map(|(off, _, _)| *off).collect(),
        Thunk::Attention { q, k, v, mask, .. } => vec![*q, *k, *v, *mask],
        Thunk::FusedAttn {
            qkv,
            mask,
            cos,
            sin,
            has_rope,
            ..
        } => {
            let mut r = vec![*qkv, *mask];
            if *has_rope != 0 {
                r.push(*cos);
                r.push(*sin);
            }
            r
        }
        Thunk::AttentionBackward {
            q, k, v, dy, mask, ..
        } => {
            let mut v = vec![*q, *k, *v, *dy];
            if *mask != *q {
                v.push(*mask);
            }
            v
        }
        Thunk::Rope { src, cos, sin, .. } => vec![*src, *cos, *sin],
        Thunk::RmsNormBackwardInput {
            x, gamma, beta, dy, ..
        } => {
            vec![*x, *gamma, *beta, *dy]
        }
        Thunk::RmsNormBackwardGamma {
            x, gamma, beta, dy, ..
        } => {
            vec![*x, *gamma, *beta, *dy]
        }
        Thunk::RmsNormBackwardBeta {
            x, gamma, beta, dy, ..
        } => {
            vec![*x, *gamma, *beta, *dy]
        }
        Thunk::LayerNormBackwardInput { x, gamma, dy, .. } => vec![*x, *gamma, *dy],
        Thunk::LayerNormBackwardGamma { x, dy, .. } => vec![*x, *dy],
        Thunk::GroupNormBackwardInput {
            x, gamma, beta, dy, ..
        } => vec![*x, *gamma, *beta, *dy],
        Thunk::GroupNormBackwardGamma { x, dy, .. } => vec![*x, *dy],
        Thunk::GroupNormBackwardBeta { dy, .. } => vec![*dy],
        Thunk::RopeBackward { dy, cos, sin, .. } => vec![*dy, *cos, *sin],
        Thunk::CumsumBackward { dy, .. } => vec![*dy],
        Thunk::GatherBackward { dy, indices, .. } => vec![*dy, *indices],
        Thunk::MaxPool2dBackward { x, dy, .. } => vec![*x, *dy],
        Thunk::Conv2dBackwardInput { dy, w, .. } => vec![*dy, *w],
        Thunk::Conv2dBackwardWeight { x, dy, .. } => vec![*x, *dy],
        Thunk::MaxPool3dBackward { x, dy, .. } => vec![*x, *dy],
        Thunk::Conv3dBackwardInput { dy, w, .. } => vec![*dy, *w],
        Thunk::Conv3dBackwardWeight { x, dy, .. } => vec![*x, *dy],
        Thunk::FusedSwiGLU { src, .. } => vec![*src],
        Thunk::FusedMlpGateUpSwiGLU {
            x, gate_w, up_w, ..
        }
        | Thunk::FusedMlpGateUpGelu {
            x, gate_w, up_w, ..
        } => vec![*x, *gate_w, *up_w],
        Thunk::FusedMlpDownResidual { x, w, res, .. } => vec![*x, *w, *res],
        Thunk::Concat { inputs, .. } => inputs.iter().map(|(o, _)| *o).collect(),
        Thunk::Narrow { src, .. } | Thunk::SplitLastAxis { src, .. } => vec![*src],
        Thunk::Copy { src, .. } => vec![*src],
        _ => vec![],
    }
}

fn concat_axis_extent(input: &rlx_ir::Shape, axis: usize, out_rank: usize) -> usize {
    let in_rank = input.rank();
    if axis >= out_rank {
        return 1;
    }
    if axis < in_rank {
        input.dim(axis).unwrap_static()
    } else {
        1
    }
}

#[cfg(test)]
mod region_rewrite_tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};

    fn empty_modulus() -> [u32; 16] {
        [0; 16]
    }

    fn region(
        len: u32,
        n_in: u32,
        num_steps: u32,
        dst: usize,
        input_offs: [u32; 16],
        chain: [u32; 128],
    ) -> Thunk {
        Thunk::ElementwiseRegion {
            len,
            num_inputs: n_in,
            num_steps,
            dst,
            input_offs,
            chain,
            scalar_input_mask: 0,
            input_modulus: empty_modulus(),
            prologue: 0,
            out_n: 0,
            out_c: 0,
            out_h: 0,
            out_w: 0,
            prologue_input: 0,
        }
    }

    #[test]
    fn rewrite_single_binary_to_binary_full() {
        let mut chain = [0u32; 128];
        chain[0] = 2;
        chain[1] = 0; // add
        chain[2] = 0;
        chain[3] = 1;
        let t = region(
            128,
            2,
            1,
            4096,
            [256, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            chain,
        );
        match try_rewrite_elementwise_region(&t) {
            RegionRewrite::One(Thunk::BinaryFull { op, len, .. }) => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(len, 128);
            }
            _ => panic!("expected BinaryFull"),
        }
    }

    #[test]
    fn rewrite_binary_then_activation_to_fused() {
        let mut chain = [0u32; 128];
        chain[0] = 2;
        chain[1] = 2; // mul
        chain[2] = 0;
        chain[3] = 1;
        chain[4] = 0;
        chain[5] = 2; // silu
        chain[6] = 0x8000_0000;
        let t = region(
            64,
            2,
            2,
            8192,
            [128, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            chain,
        );
        match try_rewrite_elementwise_region(&t) {
            RegionRewrite::One(Thunk::FusedBinaryActivation { op, act, .. }) => {
                assert_eq!(op, BinaryOp::Mul);
                assert_eq!(act, Activation::Silu);
            }
            _ => panic!("expected fused binary+activation"),
        }
    }

    #[test]
    fn rewrite_binary_then_binary_to_pair() {
        let mut chain = [0u32; 128];
        chain[0] = 2;
        chain[1] = 0; // add
        chain[2] = 0;
        chain[3] = 1;
        chain[4] = 2;
        chain[5] = 2; // mul
        chain[6] = 0x8000_0000;
        chain[7] = 2;
        let offs = [128, 256, 384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = region(32, 3, 2, 4096, offs, chain);
        match try_rewrite_elementwise_region(&t) {
            RegionRewrite::Many(ts) if ts.len() == 2 => {
                assert!(matches!(
                    ts[0],
                    Thunk::BinaryFull {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    ts[1],
                    Thunk::BinaryFull {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("expected binary+binary pair"),
        }
    }

    #[test]
    fn rewrite_binary_binary_activation_to_fused_ternary() {
        let mut chain = [0u32; 128];
        chain[0] = 2;
        chain[1] = 0; // add
        chain[2] = 0;
        chain[3] = 1;
        chain[4] = 2;
        chain[5] = 2; // mul
        chain[6] = 0x8000_0000;
        chain[7] = 2;
        chain[8] = 0;
        chain[9] = 2; // silu
        chain[10] = 0x8000_0001;
        let offs = [128, 256, 384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = region(32, 3, 3, 4096, offs, chain);
        match try_rewrite_elementwise_region(&t) {
            RegionRewrite::One(Thunk::FusedTernaryActivation { op0, op1, act, .. }) => {
                assert_eq!(op0, BinaryOp::Add);
                assert_eq!(op1, BinaryOp::Mul);
                assert_eq!(act, Activation::Silu);
            }
            _ => panic!("expected fused ternary+activation"),
        }
    }
}
