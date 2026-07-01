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

//! Pre-compiled command list — analog of rlx-cpu's Thunk.

use crate::arena::Arena;
use rlx_ir::NodeId;

const ARENA_LARGE_OFF: usize = 1usize << 32;

#[inline]
fn arena_off_large(off: usize) -> bool {
    off >= ARENA_LARGE_OFF
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
use crate::op_registry::MetalKernel;
use rlx_ir::op::{Activation, BinaryOp, CmpOp};
use rlx_ir::{DType, Graph, Op, Shape};
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
    Sgemm {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
        dt: HalfFlag,
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
    /// Element-wise comparison: out = (lhs CMP rhs) ? 1.0 : 0.0
    Compare {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: CmpOp,
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
    /// Ternary select: out = cond != 0 ? on_true : on_false
    Where {
        cond: usize,
        on_true: usize,
        on_false: usize,
        dst: usize,
        len: u32,
    },
    Fma {
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
        len: u32,
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
        Thunk::GeluApproxOut { .. } => "gelu_approx_out",
        Thunk::GeluApproxHost { .. } => "gelu_approx_host",
        Thunk::LayerNorm { .. } => "layer_norm",
        Thunk::GroupNorm { .. } => "group_norm",
        Thunk::LayerNorm2d { .. } => "layer_norm2d",
        Thunk::ConvTranspose2d { .. } => "conv_transpose2d",
        Thunk::RmsNorm { .. } => "rms_norm",
        Thunk::ResizeNearest2x { .. } => "resize_nearest_2x",
        Thunk::BinaryFull { .. } => "binary",
        Thunk::BinaryBroadcast { .. } => "binary_broadcast",
        Thunk::BiasAdd { .. } => "bias_add",
        Thunk::FusedResidualLN { .. } => "fused_residual_ln",
        Thunk::FusedResidualRmsNorm { .. } => "fused_residual_rms_norm",
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
        Thunk::RopeBackward { .. } => "rope_backward",
        Thunk::CumsumBackward { .. } => "cumsum_backward",
        Thunk::GatherBackward { .. } => "gather_backward",
        Thunk::MaxPool2dBackward { .. } => "maxpool2d_backward",
        Thunk::Conv2dBackwardInput { .. } => "conv2d_backward_input",
        Thunk::Conv2dBackwardWeight { .. } => "conv2d_backward_weight",
        Thunk::Rope { .. } => "rope",
        Thunk::Softmax { .. } => "softmax",
        Thunk::SoftmaxCrossEntropyDense { .. } => "softmax_cross_entropy_dense",
        Thunk::SoftmaxCrossEntropyWithLogits { .. } => "softmax_cross_entropy_with_logits",
        Thunk::SoftmaxCrossEntropyBackward { .. } => "softmax_cross_entropy_backward",
        Thunk::Cumsum { .. } => "cumsum",
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
        Thunk::ElementwiseRegion { .. } => "elementwise_region",
        Thunk::BatchElementwiseRegion { .. } => "batch_elementwise_region",
        Thunk::CustomOp { .. } => "custom_op",
        Thunk::GaussianSplatRender { .. } => "gaussian_splat_render",
        Thunk::GaussianSplatRenderBackward { .. } => "gaussian_splat_render_backward",
        Thunk::GaussianSplatPrepare { .. } => "gaussian_splat_prepare",
        Thunk::GaussianSplatRasterize { .. } => "gaussian_splat_rasterize",
        Thunk::AxialRope2dHost { .. } => "axial_rope2d_host",
        Thunk::Im2Col { .. } => "im2col",
        Thunk::Fft1d { .. } => "fft1d",
        Thunk::LogMel { .. } => "log_mel",
        Thunk::LogMelBackward { .. } => "log_mel_backward",
        Thunk::WelchPeaks { .. } => "welch_peaks",
        Thunk::RngNormal { .. } => "rng_normal",
        Thunk::RngUniform { .. } => "rng_uniform",
        Thunk::GatedDeltaNet { .. } => "gated_delta_net",
        Thunk::SelectiveScan { .. } => "selective_scan",
        Thunk::Sample { .. } => "sample",
        Thunk::Reverse { .. } => "reverse",
        Thunk::ArgReduce { .. } => "argreduce",
        Thunk::Lstm { .. } => "lstm",
        Thunk::Gru { .. } => "gru",
        Thunk::Rnn { .. } => "rnn",
        Thunk::Mamba2 { .. } => "mamba2",
        Thunk::DequantMatMulGguf { .. } => "dequant_matmul_gguf",
        Thunk::DequantGroupedMatMulGguf { .. } => "dequant_grouped_matmul_gguf",
        Thunk::DequantMatMulInt8 { .. } => "dequant_matmul_int8",
        Thunk::DequantMatMulInt4 { .. } => "dequant_matmul_int4",
        Thunk::DequantMatMulFp8 { .. } => "dequant_matmul_fp8",
        Thunk::DequantMatMulNvfp4 { .. } => "dequant_matmul_nvfp4",
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
            | Thunk::Copy { .. }
            | Thunk::ActivationInPlace { .. }
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
            | Thunk::FusedResidualLN { .. }
            | Thunk::FusedResidualRmsNorm { .. }
            | Thunk::Gather { .. }
            | Thunk::Compare { .. }
            | Thunk::Where { .. }
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
            | Thunk::RopeBackward { .. }
            | Thunk::CumsumBackward { .. }
            | Thunk::GatherBackward { .. }
            | Thunk::MaxPool2dBackward { .. }
            | Thunk::Conv2dBackwardInput { .. }
            | Thunk::Conv2dBackwardWeight { .. } => true,
            Thunk::Rope { .. } => true,
            // Decode seq=1 GDN / fused GGUF matmul: host paths use full
            // `batch`/`m` from the thunk (not seq-axis scale); marking
            // safe lets bucketed decode bypass whole-graph MPSGraph.
            Thunk::GatedDeltaNet { .. }
            | Thunk::SelectiveScan { .. }
            | Thunk::Sample { .. }
            | Thunk::Reverse { .. }
            | Thunk::ArgReduce { .. }
            | Thunk::Lstm { .. }
            | Thunk::Gru { .. }
            | Thunk::Rnn { .. }
            | Thunk::Mamba2 { .. }
            | Thunk::DequantMatMulGguf { .. }
            | Thunk::DequantGroupedMatMulGguf { .. }
            | Thunk::DequantMatMulInt8 { .. }
            | Thunk::DequantMatMulInt4 { .. }
            | Thunk::DequantMatMulFp8 { .. }
            | Thunk::DequantMatMulNvfp4 { .. }
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

impl ThunkSchedule {
    pub fn compile(graph: &Graph, arena: &Arena) -> Self {
        Self::compile_with_rng_fab(
            graph,
            arena,
            rlx_ir::RngOptions::default(),
            &std::collections::HashMap::new(),
        )
    }

    pub fn compile_with_rng(graph: &Graph, arena: &Arena, rng: rlx_ir::RngOptions) -> Self {
        Self::compile_with_rng_fab(graph, arena, rng, &std::collections::HashMap::new())
    }

    /// Like [`Self::compile_with_rng`] but with the native-`FusedAttentionBlock`
    /// scratch map: each surviving FAB node → its `(qkv, attn)` BYTE offsets in
    /// the appended FAB scratch region (see `rlx-metal/src/backend.rs`). Empty
    /// when every FAB was decomposed to primitives upstream.
    pub fn compile_with_rng_fab(
        graph: &Graph,
        arena: &Arena,
        rng: rlx_ir::RngOptions,
        fab_scratch: &std::collections::HashMap<rlx_ir::NodeId, (usize, usize)>,
    ) -> Self {
        let rng_shared = std::sync::Arc::new(std::sync::RwLock::new(rng));
        let mut thunks = Vec::with_capacity(graph.len());

        let off = |id| -> usize {
            if arena.has_buffer(id) {
                arena.byte_offset(id)
            } else {
                usize::MAX
            }
        };

        // native-gpu-fft real→complex fusion: a forward FFT whose input is
        // `Concat([signal, zeros])` (a real signal zero-padded to the 2N block)
        // reads `signal` directly with im=0, and the Concat + zeros Constant are
        // dropped (replaced by Nop) — eliminating a memory-bound 2N copy that can
        // cost as much as the now-4×-faster on-chip FFT. Conservative: only
        // on-chip radix-4/8 sizes (1024<n<=4096, pow2), single-use Concat/zeros,
        // and `signal` a resident Input/Param (its arena region is never aliased
        // away, so reading it one step later than planned is safe).
        // `RLX_FFT_FUSE_REAL=0` disables.
        #[cfg(feature = "native-gpu-fft")]
        let (fft_real_src, fft_real_skip): (
            std::collections::HashMap<rlx_ir::NodeId, rlx_ir::NodeId>,
            std::collections::HashSet<rlx_ir::NodeId>,
        ) = {
            let mut srcmap = std::collections::HashMap::new();
            let mut skip = std::collections::HashSet::new();
            let fuse = !rlx_ir::env::var("RLX_FFT_FUSE_REAL")
                .is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("off"));
            if fuse {
                let mut uses: std::collections::HashMap<rlx_ir::NodeId, u32> =
                    std::collections::HashMap::new();
                for node in graph.nodes() {
                    for &inp in &node.inputs {
                        *uses.entry(inp).or_insert(0) += 1;
                    }
                }
                for node in graph.nodes() {
                    let Op::Fft { inverse: false, .. } = &node.op else {
                        continue;
                    };
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        continue; // on-chip kernels are f32-only
                    }
                    let nc = rlx_ir::fft::fft_meta(&graph.node(node.inputs[0]).shape).n_complex;
                    if !(nc.is_power_of_two() && nc > rlx_ir::fft::FFT_TILE_SIZE && nc <= 4096) {
                        continue;
                    }
                    let concat_id = node.inputs[0];
                    let cnode = graph.node(concat_id);
                    let Op::Concat { axis } = &cnode.op else {
                        continue;
                    };
                    if cnode.inputs.len() != 2
                        || *axis != cnode.shape.rank() - 1
                        || uses.get(&concat_id) != Some(&1)
                    {
                        continue;
                    }
                    let (sig_id, z_id) = (cnode.inputs[0], cnode.inputs[1]);
                    let is_zeros = matches!(
                        &graph.node(z_id).op,
                        Op::Constant { data } if data.iter().all(|&b| b == 0)
                    );
                    let signode = graph.node(sig_id);
                    let sig_ok = matches!(&signode.op, Op::Input { .. } | Op::Param { .. })
                        && signode.shape.dim(signode.shape.rank() - 1).unwrap_static() == nc;
                    if is_zeros && uses.get(&z_id) == Some(&1) && sig_ok {
                        skip.insert(concat_id);
                        skip.insert(z_id);
                        srcmap.insert(node.id, sig_id);
                        if rlx_ir::env::flag("RLX_FFT_FUSE_DEBUG") {
                            eprintln!("rlx-metal: fused real→complex FFT (n_complex={nc})");
                        }
                    }
                }
            }
            (srcmap, skip)
        };

        for node in graph.nodes() {
            #[cfg(feature = "native-gpu-fft")]
            if fft_real_skip.contains(&node.id) {
                thunks.push(Thunk::Nop);
                continue;
            }
            // View ops alias their parent's slot (planner did this); the
            // GPU thunk path also emits Nop. Plan #46.
            if rlx_opt::is_pure_view(graph, node) {
                thunks.push(Thunk::Nop);
                continue;
            }
            if let Op::BatchElementwiseRegion {
                chain,
                num_batch_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input,
            } = &node.op
            {
                let n = *num_batch_inputs as usize;
                if n == 0 || chain.len() > 32 {
                    panic!(
                        "rlx-metal BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                        chain.len()
                    );
                }
                let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                    .expect("batch region static shape") as u32;
                let elem_bytes = node.shape.dtype().size_bytes();
                let slice_bytes = slice_elems as usize * elem_bytes;
                let base_dst = off(node.id);
                let chain_enc = rlx_ir::encode_chain_steps(chain);
                let tail = rlx_ir::encode_prologue_tail(*prologue, &slice_shape, *prologue_input);
                let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                if use_single {
                    let mut batch_input_offs = [0u32; 64];
                    for i in 0..n {
                        batch_input_offs[i] = off(node.inputs[i]) as u32 / 4;
                    }
                    thunks.push(Thunk::BatchElementwiseRegion {
                        slice_len: slice_elems,
                        num_batch: n as u32,
                        num_steps: chain.len() as u32,
                        base_dst,
                        slice_elems,
                        batch_input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                    });
                } else {
                    for i in 0..n {
                        let mut input_offs = [0u32; 16];
                        input_offs[0] = off(node.inputs[i]) as u32 / 4;
                        thunks.push(Thunk::ElementwiseRegion {
                            len: slice_elems,
                            num_inputs: 1,
                            num_steps: chain.len() as u32,
                            dst: base_dst + i * slice_bytes,
                            input_offs,
                            chain: chain_enc,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                            prologue: tail[0],
                            out_n: tail[1],
                            out_c: tail[2],
                            out_h: tail[3],
                            out_w: tail[4],
                            prologue_input: tail[5],
                        });
                    }
                }
                continue;
            }
            // Native `Op::FusedAttentionBlock` (no-bias, f32; gated upstream so
            // only nodes with a scratch slot reach here): two GEMMs into packed
            // scratch around the fused RoPE+SDPA kernel. Non-native FAB was
            // decomposed to primitives before the arena was planned.
            if let Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_rope,
                ..
            } = &node.op
            {
                if let Some(&(qkv_off, attn_off)) = fab_scratch.get(&node.id) {
                    if rlx_ir::env::flag("RLX_METAL_TRACE_FAB") {
                        eprintln!(
                            "[rlx-metal] native fused_attn_block: heads={num_heads} \
                             head_dim={head_dim} rope={has_rope}"
                        );
                    }
                    let nh = *num_heads;
                    let hd = *head_dim;
                    let inner = nh * hd;
                    let dims = node.shape.dims();
                    let b = dims[0].unwrap_static();
                    let s = dims[1].unwrap_static();
                    let m = (b * s) as u32;
                    let dt = node.shape.dtype().into();
                    // 1. qkv = hidden @ qkv_w → qkv scratch [B, S, 3*inner].
                    thunks.push(Thunk::Sgemm {
                        a: off(node.inputs[0]),
                        b: off(node.inputs[1]),
                        c: qkv_off,
                        m,
                        k: inner as u32,
                        n: (3 * inner) as u32,
                        dt,
                    });
                    // 2. attn = fused RoPE + SDPA(qkv, mask) → attn scratch.
                    let (cos_off, sin_off) = if *has_rope {
                        (off(node.inputs[4]), off(node.inputs[5]))
                    } else {
                        (0usize, 0usize)
                    };
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    thunks.push(Thunk::FusedAttn {
                        qkv: qkv_off,
                        mask: off(node.inputs[3]),
                        cos: cos_off,
                        sin: sin_off,
                        out: attn_off,
                        batch: b as u32,
                        seq: s as u32,
                        heads: nh as u32,
                        head_dim: hd as u32,
                        mask_kind: 2, // Custom binary [B,S] — the only FAB mask
                        scale_bits: scale.to_bits(),
                        has_rope: u32::from(*has_rope),
                    });
                    // 3. out = attn @ out_w → node output [B, S, inner].
                    thunks.push(Thunk::Sgemm {
                        a: attn_off,
                        b: off(node.inputs[2]),
                        c: off(node.id),
                        m,
                        k: inner as u32,
                        n: inner as u32,
                        dt,
                    });
                    continue;
                }
            }
            let t = match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => Thunk::Nop,

                Op::MatMul => {
                    let shape = &node.shape;
                    let a_shape = &graph.node(node.inputs[0]).shape;
                    let b_shape = &graph.node(node.inputs[1]).shape;
                    // Any-rank batched matmul: all leading dims (except the
                    // last 2) match between A, B, and output, and the last
                    // 2 dims form [M, K] @ [K, N] = [M, N]. The 2-D Sgemm
                    // flatten trick is wrong when both operands carry
                    // independent batch dims (SAM3 decomposed attention).
                    let batched = a_shape.rank() >= 3
                        && b_shape.rank() == a_shape.rank()
                        && shape.rank() == a_shape.rank()
                        && {
                            let mut ok = true;
                            for d in 0..a_shape.rank() - 2 {
                                if a_shape.dim(d) != b_shape.dim(d)
                                    || a_shape.dim(d) != shape.dim(d)
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            ok
                        };
                    if batched {
                        let r = shape.rank();
                        let mut batch_prod = 1usize;
                        for d in 0..r - 2 {
                            batch_prod *= shape.dim(d).unwrap_static();
                        }
                        let m_dim = shape.dim(r - 2).unwrap_static();
                        let k_dim = a_shape.dim(r - 1).unwrap_static();
                        let n_dim = shape.dim(r - 1).unwrap_static();
                        Thunk::BatchedSgemm {
                            a: off(node.inputs[0]),
                            b: off(node.inputs[1]),
                            c: off(node.id),
                            batch: batch_prod as u32,
                            m: m_dim as u32,
                            k: k_dim as u32,
                            n: n_dim as u32,
                            dt: shape.dtype().into(),
                        }
                    } else {
                        let n = shape.dim(shape.rank() - 1).unwrap_static();
                        let total = shape.num_elements().unwrap();
                        let m = total / n;
                        let a_total = a_shape.num_elements().unwrap();
                        let k = a_total / m;
                        Thunk::Sgemm {
                            a: off(node.inputs[0]),
                            b: off(node.inputs[1]),
                            c: off(node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            dt: shape.dtype().into(),
                        }
                    }
                }

                Op::FusedMatMulBiasAct { activation } => {
                    let shape = &node.shape;
                    let n = shape.dim(shape.rank() - 1).unwrap_static();
                    let total = shape.num_elements().unwrap();
                    let m = total / n;
                    let a_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = a_total / m;
                    Thunk::FusedMmBiasAct {
                        a: off(node.inputs[0]),
                        w: off(node.inputs[1]),
                        bias: off(node.inputs[2]),
                        c: off(node.id),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        act: *activation,
                        dt: shape.dtype().into(),
                    }
                }

                Op::Cast { to } => {
                    let len = node.shape.num_elements().unwrap();
                    let src_dt: HalfFlag = graph.node(node.inputs[0]).shape.dtype().into();
                    let dst_dt: HalfFlag = (*to).into();
                    Thunk::Cast {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        len: len as u32,
                        src_dt,
                        dst_dt,
                    }
                }

                Op::Activation(act) => {
                    let len = node.shape.num_elements().unwrap();
                    let in_off = off(node.inputs[0]);
                    let out_off = off(node.id);
                    // Same fix as CPU thunk: when planner gives input and
                    // output different slots (standalone activation), emit
                    // a Copy first so the in-place kernel runs on the
                    // actual input data. When aliased, single in-place
                    // kernel suffices.
                    let dt: HalfFlag = node.shape.dtype().into();
                    if in_off == out_off {
                        Thunk::ActivationInPlace {
                            data: out_off,
                            len: len as u32,
                            act: *act,
                            dt,
                        }
                    } else if matches!(act, Activation::GeluApprox) && dt == HalfFlag::F32 {
                        if metal_host_fallback_enabled()
                            && (arena_off_large(in_off) || arena_off_large(out_off))
                        {
                            Thunk::GeluApproxHost {
                                src: in_off,
                                dst: out_off,
                                len: len as u32,
                            }
                        } else {
                            Thunk::GeluApproxOut {
                                src: in_off,
                                dst: out_off,
                                len: len as u32,
                            }
                        }
                    } else {
                        let in_dt: HalfFlag = graph.node(node.inputs[0]).shape.dtype().into();
                        thunks.push(Thunk::Copy {
                            src: in_off,
                            dst: out_off,
                            len: len as u32,
                            dt: in_dt,
                        });
                        Thunk::ActivationInPlace {
                            data: out_off,
                            len: len as u32,
                            act: *act,
                            dt,
                        }
                    }
                }

                Op::LayerNorm { eps, .. } => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    Thunk::LayerNorm {
                        src: off(node.inputs[0]),
                        g: off(node.inputs[1]),
                        b: off(node.inputs[2]),
                        dst: off(node.id),
                        rows: (total / h) as u32,
                        h: h as u32,
                        eps: *eps,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::GroupNorm { num_groups, eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::GroupNorm {
                        src: off(node.inputs[0]),
                        g: off(node.inputs[1]),
                        b: off(node.inputs[2]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        num_groups: *num_groups as u32,
                        eps: *eps,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::LayerNorm2d { eps } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::LayerNorm2d {
                        src: off(node.inputs[0]),
                        g: off(node.inputs[1]),
                        b: off(node.inputs[2]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        eps: *eps,
                        dt: node.shape.dtype().into(),
                    }
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
                    Thunk::ConvTranspose2d {
                        src: off(node.inputs[0]),
                        weight: off(node.inputs[1]),
                        dst: off(node.id),
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
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::ResizeNearest2x => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::ResizeNearest2x {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::RmsNorm { eps, .. } => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    Thunk::RmsNorm {
                        src: off(node.inputs[0]),
                        g: off(node.inputs[1]),
                        b: off(node.inputs[2]),
                        dst: off(node.id),
                        rows: (total / h) as u32,
                        h: h as u32,
                        eps: *eps,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::FusedResidualLN { has_bias, eps } => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let rows = total / h;
                    let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
                    Thunk::FusedResidualLN {
                        x: off(node.inputs[0]),
                        res: off(node.inputs[1]),
                        bias: if *has_bias { off(node.inputs[2]) } else { 0 },
                        g: off(node.inputs[g_idx]),
                        b: off(node.inputs[b_idx]),
                        out: off(node.id),
                        rows: rows as u32,
                        h: h as u32,
                        eps: *eps,
                        has_bias: *has_bias,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::FusedResidualRmsNorm { has_bias, eps } => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let rows = total / h;
                    let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
                    Thunk::FusedResidualRmsNorm {
                        x: off(node.inputs[0]),
                        res: off(node.inputs[1]),
                        bias: if *has_bias { off(node.inputs[2]) } else { 0 },
                        g: off(node.inputs[g_idx]),
                        b: off(node.inputs[b_idx]),
                        out: off(node.id),
                        rows: rows as u32,
                        h: h as u32,
                        eps: *eps,
                        has_bias: *has_bias,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::Binary(op) => {
                    let len = node.shape.num_elements().unwrap();
                    let lhs_shape = &graph.node(node.inputs[0]).shape;
                    let rhs_shape = &graph.node(node.inputs[1]).shape;
                    let lhs_len = lhs_shape.num_elements().unwrap();
                    let rhs_len = rhs_shape.num_elements().unwrap();
                    let dt: HalfFlag = node.shape.dtype().into();

                    // Fast paths: same-size (BinaryFull) and trailing-
                    // broadcast bias (BiasAdd). For anything else with
                    // a mid-shape singleton, fall through to the
                    // shape-aware BinaryBroadcast.
                    let needs_broadcast = lhs_len != len || rhs_len != len;
                    let is_trailing_bias = matches!(op, BinaryOp::Add)
                        && rhs_len < len
                        && len % rhs_len == 0
                        && lhs_len == len
                        && trailing_broadcast(lhs_shape, rhs_shape);
                    if !needs_broadcast {
                        Thunk::BinaryFull {
                            lhs: off(node.inputs[0]),
                            rhs: off(node.inputs[1]),
                            dst: off(node.id),
                            len: len as u32,
                            op: *op,
                            dt,
                        }
                    } else if is_trailing_bias {
                        Thunk::BiasAdd {
                            src: off(node.inputs[0]),
                            bias: off(node.inputs[1]),
                            dst: off(node.id),
                            m: (len / rhs_len) as u32,
                            n: rhs_len as u32,
                            dt,
                        }
                    } else {
                        let out_dims_v: Vec<usize> = (0..node.shape.rank())
                            .map(|i| node.shape.dim(i).unwrap_static())
                            .collect();
                        let lhs_dims: Vec<usize> = (0..lhs_shape.rank())
                            .map(|i| lhs_shape.dim(i).unwrap_static())
                            .collect();
                        let rhs_dims: Vec<usize> = (0..rhs_shape.rank())
                            .map(|i| rhs_shape.dim(i).unwrap_static())
                            .collect();
                        let lhs_strides = broadcast_strides(&lhs_dims, &out_dims_v);
                        let rhs_strides = broadcast_strides(&rhs_dims, &out_dims_v);
                        let out_dims_u: Vec<u32> = out_dims_v.iter().map(|&d| d as u32).collect();
                        Thunk::BinaryBroadcast {
                            lhs: off(node.inputs[0]),
                            rhs: off(node.inputs[1]),
                            dst: off(node.id),
                            len: len as u32,
                            op: *op,
                            dt,
                            rank: out_dims_u.len() as u32,
                            out_dims: out_dims_u,
                            lhs_strides,
                            rhs_strides,
                        }
                    }
                }

                Op::Gather { axis } if *axis == 0 => {
                    let table_shape = &graph.node(node.inputs[0]).shape;
                    let trailing: usize = (1..table_shape.rank())
                        .map(|i| table_shape.dim(i).unwrap_static())
                        .product();
                    let idx_len = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    Thunk::Gather {
                        table: off(node.inputs[0]),
                        idx: off(node.inputs[1]),
                        dst: off(node.id),
                        num_idx: idx_len as u32,
                        trailing: trailing as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::Narrow { axis, start, len } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let in_axis = in_shape.dim(*axis).unwrap_static();
                    Thunk::Narrow {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        outer: outer as u32,
                        src_axis: (in_axis * inner) as u32,
                        start: (*start * inner) as u32,
                        len: (*len * inner) as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::Reshape { .. } => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::Copy {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        len: len as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                // Identity forward; gradient-stop on the backward (the AD
                // pass treats `StopGradient` specially upstream so by the
                // time we land here it's a pure copy).
                Op::StopGradient => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::Copy {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        len: len as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::Expand { .. } => {
                    // Broadcast via Transpose-with-stride-0: build per-dim
                    // strides where input dims of size 1 broadcast.
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    let in_rank = in_shape.rank();
                    let out_rank = out_shape.rank();
                    let pad = out_rank.saturating_sub(in_rank);
                    let in_dims: Vec<usize> = (0..out_rank)
                        .map(|i| {
                            if i < pad {
                                1
                            } else {
                                in_shape.dim(i - pad).unwrap_static()
                            }
                        })
                        .collect();
                    let mut full_strides = vec![1usize; out_rank];
                    for d in (0..out_rank.saturating_sub(1)).rev() {
                        full_strides[d] = full_strides[d + 1] * in_dims[d + 1];
                    }
                    let out_dims: Vec<u32> = (0..out_rank)
                        .map(|i| out_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let in_strides: Vec<u32> = (0..out_rank)
                        .map(|i| {
                            if in_dims[i] == 1 && (out_dims[i] as usize) > 1 {
                                0
                            } else {
                                full_strides[i] as u32
                            }
                        })
                        .collect();
                    let total: u32 = out_dims.iter().product();
                    Thunk::Transpose {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        total,
                        out_dims,
                        in_strides,
                    }
                }

                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } => {
                    let (mask_kind_u32, window): (u32, u32) = match mask_kind {
                        rlx_ir::op::MaskKind::None => (0, 0),
                        rlx_ir::op::MaskKind::Causal => (1, 0),
                        rlx_ir::op::MaskKind::Custom => (2, 0),
                        rlx_ir::op::MaskKind::Bias => (3, 0),
                        rlx_ir::op::MaskKind::SlidingWindow(w) => (4, *w as u32),
                    };
                    let mask_off = if matches!(
                        mask_kind,
                        rlx_ir::op::MaskKind::Custom | rlx_ir::op::MaskKind::Bias
                    ) {
                        off(node.inputs[3])
                    } else {
                        off(node.inputs[0])
                    };
                    let q_shape = &graph.node(node.inputs[0]).shape;
                    let k_shape = &graph.node(node.inputs[1]).shape;
                    let rank = q_shape.rank();
                    let (batch, seq, kv_seq, bhsd) = if rank == 4 {
                        let d1 = q_shape.dim(1).unwrap_static();
                        let d2 = q_shape.dim(2).unwrap_static();
                        if d1 == *num_heads {
                            (
                                q_shape.dim(0).unwrap_static(),
                                d2,
                                k_shape.dim(2).unwrap_static(),
                                1u32,
                            )
                        } else {
                            (
                                q_shape.dim(0).unwrap_static(),
                                d1,
                                k_shape.dim(1).unwrap_static(),
                                0u32,
                            )
                        }
                    } else if q_shape.rank() >= 3 {
                        (
                            q_shape.dim(0).unwrap_static(),
                            q_shape.dim(1).unwrap_static(),
                            k_shape.dim(1).unwrap_static(),
                            0u32,
                        )
                    } else {
                        (
                            1,
                            q_shape.dim(0).unwrap_static(),
                            k_shape.dim(0).unwrap_static(),
                            0u32,
                        )
                    };
                    Thunk::Attention {
                        q: off(node.inputs[0]),
                        k: off(node.inputs[1]),
                        v: off(node.inputs[2]),
                        mask: mask_off,
                        out: off(node.id),
                        batch: batch as u32,
                        seq: seq as u32,
                        kv_seq: kv_seq as u32,
                        heads: *num_heads as u32,
                        head_dim: *head_dim as u32,
                        mask_kind: mask_kind_u32,
                        window,
                        dt: node.shape.dtype().into(),
                        bhsd,
                        score_scale: score_scale.unwrap_or(0.0),
                        attn_logit_softcap: attn_logit_softcap.unwrap_or(0.0),
                    }
                }

                Op::AttentionBackward {
                    num_heads,
                    head_dim,
                    mask_kind,
                    wrt,
                } => {
                    use rlx_ir::op::AttentionBwdWrt;
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal AttentionBackward: F32 only (use CPU for f16 training)");
                    }
                    let (mask_kind_u32, mask_off, window) = match mask_kind {
                        rlx_ir::op::MaskKind::None => (0u32, off(node.inputs[0]), 0u32),
                        rlx_ir::op::MaskKind::Causal => (1u32, off(node.inputs[0]), 0u32),
                        rlx_ir::op::MaskKind::Custom => (2u32, off(node.inputs[4]), 0u32),
                        rlx_ir::op::MaskKind::Bias => (4u32, off(node.inputs[4]), 0u32),
                        rlx_ir::op::MaskKind::SlidingWindow(w) => {
                            (3u32, off(node.inputs[0]), *w as u32)
                        }
                    };
                    let q_shape = &graph.node(node.inputs[0]).shape;
                    let k_shape = &graph.node(node.inputs[1]).shape;
                    let rank = q_shape.rank();
                    let (batch, seq, kv_seq, bhsd) = if rank == 4 {
                        let d1 = q_shape.dim(1).unwrap_static();
                        let d2 = q_shape.dim(2).unwrap_static();
                        if d1 == *num_heads {
                            (
                                q_shape.dim(0).unwrap_static(),
                                d2,
                                k_shape.dim(2).unwrap_static(),
                                1u32,
                            )
                        } else {
                            (
                                q_shape.dim(0).unwrap_static(),
                                d1,
                                k_shape.dim(1).unwrap_static(),
                                0u32,
                            )
                        }
                    } else if rank >= 3 {
                        (
                            q_shape.dim(0).unwrap_static(),
                            q_shape.dim(1).unwrap_static(),
                            k_shape.dim(1).unwrap_static(),
                            0u32,
                        )
                    } else {
                        (
                            1,
                            q_shape.dim(0).unwrap_static(),
                            k_shape.dim(0).unwrap_static(),
                            0u32,
                        )
                    };
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    Thunk::AttentionBackward {
                        q: off(node.inputs[0]),
                        k: off(node.inputs[1]),
                        v: off(node.inputs[2]),
                        dy: off(node.inputs[3]),
                        mask: mask_off,
                        out: off(node.id),
                        batch: batch as u32,
                        seq: seq as u32,
                        kv_seq: kv_seq as u32,
                        heads: *num_heads as u32,
                        head_dim: *head_dim as u32,
                        mask_kind: mask_kind_u32,
                        window,
                        wrt: wrt_id,
                        bhsd,
                    }
                }

                Op::Rope {
                    head_dim,
                    n_rot,
                    style,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if x_shape.rank() >= 3 {
                        (
                            x_shape.dim(0).unwrap_static(),
                            x_shape.dim(1).unwrap_static(),
                            x_shape.dim(2).unwrap_static(),
                        )
                    } else {
                        let total = x_shape.num_elements().unwrap();
                        let s = x_shape.dim(x_shape.rank() - 2).unwrap_static();
                        (total / (s * head_dim), s, *head_dim)
                    };
                    let _ = node.shape.dtype(); // ensure dtype-aware
                    // Per-token RoPE when the cos table has one row per
                    // (batch·seq) token (ragged decode), distinct from the
                    // shared per-seq-position table.
                    let half = (head_dim / 2).max(1);
                    let cos_rows =
                        graph.node(node.inputs[1]).shape.num_elements().unwrap_or(0) / half;
                    let cos_per_token = cos_rows == batch * seq && cos_rows != seq;
                    Thunk::Rope {
                        src: off(node.inputs[0]),
                        cos: off(node.inputs[1]),
                        sin: off(node.inputs[2]),
                        dst: off(node.id),
                        batch: batch as u32,
                        seq: seq as u32,
                        hidden: hidden as u32,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        dt: node.shape.dtype().into(),
                        src_row_stride: hidden as u32,
                        cos_per_token,
                        interleaved: matches!(style, rlx_ir::op::RopeStyle::GptJ),
                    }
                }

                Op::Softmax { axis } => {
                    let rank = node.shape.rank();
                    let ax = if *axis < 0 {
                        (rank as i32 + axis) as usize
                    } else {
                        *axis as usize
                    };
                    let cols = node.shape.dim(ax).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let in_off = off(node.inputs[0]);
                    let out_off = off(node.id);
                    // Softmax operates in-place. When the planner doesn't
                    // alias input and output, prepend a Copy so the
                    // in-place kernel actually sees the input data.
                    if in_off != out_off {
                        thunks.push(Thunk::Copy {
                            src: in_off,
                            dst: out_off,
                            len: total as u32,
                            dt: node.shape.dtype().into(),
                        });
                    }
                    Thunk::Softmax {
                        data: out_off,
                        rows: (total / cols) as u32,
                        cols: cols as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::SoftmaxCrossEntropy => {
                    let logits_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::SoftmaxCrossEntropyDense {
                        logits: off(node.inputs[0]),
                        targets: off(node.inputs[1]),
                        dst: off(node.id),
                        n: logits_shape.dim(0).unwrap_static() as u32,
                        c: logits_shape.dim(1).unwrap_static() as u32,
                    }
                }

                Op::SoftmaxCrossEntropyWithLogits => {
                    let logits_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::SoftmaxCrossEntropyWithLogits {
                        logits: off(node.inputs[0]),
                        labels: off(node.inputs[1]),
                        dst: off(node.id),
                        n: logits_shape.dim(0).unwrap_static() as u32,
                        c: logits_shape.dim(1).unwrap_static() as u32,
                    }
                }

                Op::SoftmaxCrossEntropyBackward => {
                    let logits_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::SoftmaxCrossEntropyBackward {
                        logits: off(node.inputs[0]),
                        labels: off(node.inputs[1]),
                        d_loss: off(node.inputs[2]),
                        dlogits: off(node.id),
                        n: logits_shape.dim(0).unwrap_static() as u32,
                        c: logits_shape.dim(1).unwrap_static() as u32,
                    }
                }

                Op::Concat { axis } => {
                    // Generalized to any axis. `outer` is the product of
                    // dims preceding the concat axis, `inner` is the
                    // product of dims following it. SAM windowed
                    // attention concats zero-pads along spatial axes (1
                    // and 2) of a `[1, hw, hw, E]` BHWC tensor, so
                    // last-axis-only was silently wrong on Metal in
                    // release builds (the prior `debug_assert!` was a
                    // no-op).
                    let out_shape = &node.shape;
                    let rank = out_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| out_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| out_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let dst_axis = out_shape.dim(*axis).unwrap_static();
                    let inputs: Vec<(usize, u32)> = node
                        .inputs
                        .iter()
                        .map(|&in_id| {
                            let in_shape = &graph.node(in_id).shape;
                            let in_axis = concat_axis_extent(in_shape, *axis, rank);
                            (off(in_id), in_axis as u32)
                        })
                        .collect();
                    Thunk::Concat {
                        dst: off(node.id),
                        outer: outer as u32,
                        dst_axis: dst_axis as u32,
                        inner: inner as u32,
                        dt: out_shape.dtype().into(),
                        inputs,
                    }
                }

                Op::Conv {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    if kernel_size.len() == 2
                        && in_shape.rank() == 4
                        && w_shape.rank() == 4
                        && out_shape.rank() == 4
                    {
                        Thunk::Conv2D {
                            src: off(node.inputs[0]),
                            weight: off(node.inputs[1]),
                            dst: off(node.id),
                            n: in_shape.dim(0).unwrap_static() as u32,
                            c_in: in_shape.dim(1).unwrap_static() as u32,
                            h: in_shape.dim(2).unwrap_static() as u32,
                            w: in_shape.dim(3).unwrap_static() as u32,
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
                        }
                    } else {
                        Thunk::Nop
                    }
                }

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    if kernel_size.len() == 2 && in_shape.rank() == 4 && out_shape.rank() == 4 {
                        Thunk::Pool2D {
                            src: off(node.inputs[0]),
                            dst: off(node.id),
                            n: in_shape.dim(0).unwrap_static() as u32,
                            c: in_shape.dim(1).unwrap_static() as u32,
                            h: in_shape.dim(2).unwrap_static() as u32,
                            w: in_shape.dim(3).unwrap_static() as u32,
                            h_out: out_shape.dim(2).unwrap_static() as u32,
                            w_out: out_shape.dim(3).unwrap_static() as u32,
                            kh: kernel_size[0] as u32,
                            kw: kernel_size[1] as u32,
                            sh: stride.first().copied().unwrap_or(1) as u32,
                            sw: stride.get(1).copied().unwrap_or(1) as u32,
                            ph: padding.first().copied().unwrap_or(0) as u32,
                            pw: padding.get(1).copied().unwrap_or(0) as u32,
                            kind: *kind,
                        }
                    } else {
                        Thunk::Nop
                    }
                }

                Op::Gather { axis } if *axis != 0 => {
                    let table_shape = &graph.node(node.inputs[0]).shape;
                    let rank = table_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| table_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let trailing: usize = (*axis + 1..rank)
                        .map(|i| table_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_dim = table_shape.dim(*axis).unwrap_static();
                    let idx_len = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    Thunk::GatherAxis {
                        table: off(node.inputs[0]),
                        idx: off(node.inputs[1]),
                        dst: off(node.id),
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: idx_len as u32,
                        trailing: trailing as u32,
                    }
                }

                Op::Transpose { perm } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let in_rank = in_shape.rank();
                    let in_dims: Vec<usize> = (0..in_rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .collect();
                    let mut full_strides = vec![1usize; in_rank];
                    for d in (0..in_rank.saturating_sub(1)).rev() {
                        full_strides[d] = full_strides[d + 1] * in_dims[d + 1];
                    }
                    let out_dims: Vec<u32> = perm.iter().map(|&p| in_dims[p] as u32).collect();
                    let in_strides: Vec<u32> =
                        perm.iter().map(|&p| full_strides[p] as u32).collect();
                    let total: u32 = out_dims.iter().product();
                    Thunk::Transpose {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        total,
                        out_dims,
                        in_strides,
                    }
                }

                Op::ScatterAdd => {
                    let upd_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    let num_updates = upd_shape.dim(0).unwrap_static();
                    let out_dim = out_shape.dim(0).unwrap_static();
                    let trailing: usize = (1..out_shape.rank())
                        .map(|i| out_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    Thunk::ScatterAdd {
                        updates: off(node.inputs[0]),
                        indices: off(node.inputs[1]),
                        dst: off(node.id),
                        num_updates: num_updates as u32,
                        out_dim: out_dim as u32,
                        trailing: trailing as u32,
                    }
                }

                Op::GroupedMatMul => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
                    let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
                    let num_experts = w_shape.dim(0).unwrap_static();
                    let n = w_shape.dim(2).unwrap_static();
                    Thunk::GroupedMatMul {
                        input: off(node.inputs[0]),
                        weight: off(node.inputs[1]),
                        expert_idx: off(node.inputs[2]),
                        dst: off(node.id),
                        m: m as u32,
                        k_dim: k_dim as u32,
                        n: n as u32,
                        num_experts: num_experts as u32,
                    }
                }

                Op::DequantGroupedMatMul { scheme } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
                    let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let block_elems = scheme.gguf_block_size() as usize;
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let slab_bytes = (k_dim * n) / block_elems * block_bytes;
                    let total_bytes = w_shape.num_elements().unwrap();
                    let num_experts = total_bytes / slab_bytes.max(1);
                    Thunk::DequantGroupedMatMulGguf {
                        input: off(node.inputs[0]),
                        w_q: off(node.inputs[1]),
                        expert_idx: off(node.inputs[2]),
                        dst: off(node.id),
                        m: m as u32,
                        k_dim: k_dim as u32,
                        n: n as u32,
                        num_experts: num_experts as u32,
                        scheme: *scheme,
                    }
                }

                Op::TopK { k } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let axis_dim = in_shape.dim(rank - 1).unwrap_static();
                    let outer = in_shape.num_elements().unwrap() / axis_dim;
                    Thunk::TopK {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        k: *k as u32,
                    }
                }

                Op::Cumsum { axis, exclusive } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal Cumsum: F32 only");
                    }
                    let rank = node.shape.rank();
                    let ax = if *axis < 0 {
                        (rank as i32 + *axis) as usize
                    } else {
                        *axis as usize
                    };
                    if ax != rank.saturating_sub(1) {
                        panic!(
                            "rlx-metal Cumsum: only last-axis wired (got axis={axis}, rank={rank})"
                        );
                    }
                    let cols = node.shape.dim(ax).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    Thunk::Cumsum {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        rows: (total / cols.max(1)) as u32,
                        cols: cols as u32,
                        exclusive: *exclusive,
                    }
                }

                Op::Reduce {
                    op,
                    axes,
                    keep_dim: _,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let mut sorted = axes.clone();
                    sorted.sort();
                    sorted.dedup();
                    let contiguous = !sorted.is_empty()
                        && *sorted.last().unwrap() < rank
                        && sorted.windows(2).all(|w| w[1] == w[0] + 1);
                    if !contiguous {
                        Thunk::Nop
                    } else {
                        let first = sorted[0];
                        let last = *sorted.last().unwrap();
                        let outer: usize = (0..first)
                            .map(|i| in_shape.dim(i).unwrap_static())
                            .product::<usize>()
                            .max(1);
                        let reduced: usize = (first..=last)
                            .map(|i| in_shape.dim(i).unwrap_static())
                            .product();
                        let inner: usize = (last + 1..rank)
                            .map(|i| in_shape.dim(i).unwrap_static())
                            .product::<usize>()
                            .max(1);
                        Thunk::Reduce {
                            src: off(node.inputs[0]),
                            dst: off(node.id),
                            outer: outer as u32,
                            reduced: reduced as u32,
                            inner: inner as u32,
                            op: *op,
                            dt: node.shape.dtype().into(),
                        }
                    }
                }

                Op::Compare(cmp) => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::Compare {
                        lhs: off(node.inputs[0]),
                        rhs: off(node.inputs[1]),
                        dst: off(node.id),
                        len: len as u32,
                        op: *cmp,
                    }
                }

                Op::Where => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::Where {
                        cond: off(node.inputs[0]),
                        on_true: off(node.inputs[1]),
                        on_false: off(node.inputs[2]),
                        dst: off(node.id),
                        len: len as u32,
                    }
                }

                Op::Fma => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::Fma {
                        a: off(node.inputs[0]),
                        b: off(node.inputs[1]),
                        c: off(node.inputs[2]),
                        dst: off(node.id),
                        len: len as u32,
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
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-metal ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Use UnfuseElementwiseRegions to fall back.",
                            chain.len()
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = off(id) as u32 / 4;
                    }
                    let chain_enc = rlx_ir::encode_chain_steps(chain);
                    let tail =
                        rlx_ir::encode_prologue_tail(*prologue, &node.shape, *prologue_input);
                    Thunk::ElementwiseRegion {
                        len: node.shape.num_elements().unwrap() as u32,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst: off(node.id),
                        input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                        prologue: tail[0],
                        out_n: tail[1],
                        out_c: tail[2],
                        out_h: tail[3],
                        out_w: tail[4],
                        prologue_input: tail[5],
                    }
                }

                Op::FusedSwiGLU {
                    cast_to,
                    gate_first,
                } => {
                    // Output last dim = n_half; total output elements = product of all dims.
                    let n_half = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let src_dt: HalfFlag = graph.node(node.inputs[0]).shape.dtype().into();
                    // When cast_to is None, output dtype matches the node's own
                    // dtype (set by AutoMixedPrecision or carried from the input).
                    let dst_dt: HalfFlag = match cast_to {
                        Some(dt) => (*dt).into(),
                        None => node.shape.dtype().into(),
                    };
                    Thunk::FusedSwiGLU {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        n_half: n_half as u32,
                        total: total as u32,
                        src_dt,
                        dst_dt,
                        gate_first: *gate_first,
                    }
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
                    let elem_len =
                        |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
                    Thunk::GaussianSplatRender {
                        positions_off: off(node.inputs[0]),
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: off(node.inputs[1]),
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: off(node.inputs[2]),
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: off(node.inputs[3]),
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: off(node.inputs[4]),
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: off(node.inputs[5]),
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: off(node.inputs[6]),
                        dst_off: off(node.id),
                        dst_len: node.shape.num_elements().unwrap_or(0),
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    }
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
                    let elem_len =
                        |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
                    Thunk::GaussianSplatRenderBackward {
                        positions_off: off(node.inputs[0]),
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: off(node.inputs[1]),
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: off(node.inputs[2]),
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: off(node.inputs[3]),
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: off(node.inputs[4]),
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: off(node.inputs[5]),
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: off(node.inputs[6]),
                        d_loss_off: off(node.inputs[7]),
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_off: off(node.id),
                        packed_len: node.shape.num_elements().unwrap_or(0),
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
                    }
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
                    let elem_len =
                        |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
                    Thunk::GaussianSplatPrepare {
                        positions_off: off(node.inputs[0]),
                        positions_len: elem_len(node.inputs[0]),
                        scales_off: off(node.inputs[1]),
                        scales_len: elem_len(node.inputs[1]),
                        rotations_off: off(node.inputs[2]),
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_off: off(node.inputs[3]),
                        opacities_len: elem_len(node.inputs[3]),
                        colors_off: off(node.inputs[4]),
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_off: off(node.inputs[5]),
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_off: off(node.inputs[6]),
                        meta_len: elem_len(node.inputs[6]),
                        prep_off: off(node.id),
                        prep_len: node.shape.num_elements().unwrap_or(0),
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    }
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
                    let elem_len =
                        |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
                    let prep_id = node.inputs[0];
                    let count = match &graph.node(prep_id).op {
                        rlx_ir::Op::GaussianSplatPrepare { .. } => {
                            elem_len(graph.node(prep_id).inputs[0]) / 3
                        }
                        _ => 1,
                    };
                    Thunk::GaussianSplatRasterize {
                        prep_off: off(prep_id),
                        prep_len: elem_len(prep_id),
                        meta_off: off(node.inputs[1]),
                        meta_len: elem_len(node.inputs[1]),
                        dst_off: off(node.id),
                        dst_len: node.shape.num_elements().unwrap_or(0),
                        count,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    }
                }

                Op::AxialRope2d {
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    assert_eq!(
                        node.shape.dtype(),
                        rlx_ir::DType::F32,
                        "rlx-metal Op::AxialRope2d host fallback requires F32"
                    );
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::AxialRope2dHost {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        batch: in_shape.dim(0).unwrap_static() as u32,
                        seq: in_shape.dim(1).unwrap_static() as u32,
                        hidden: in_shape.dim(2).unwrap_static() as u32,
                        end_x: *end_x as u32,
                        end_y: *end_y as u32,
                        head_dim: *head_dim as u32,
                        num_heads: *num_heads as u32,
                        theta: *theta,
                        repeat_factor: *repeat_factor as u32,
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
                        panic!("rlx-metal Im2Col: 2D NCHW only");
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
                    Thunk::Im2Col {
                        x: off(node.inputs[0]),
                        col: off(node.id),
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
                    }
                }

                Op::Fft { inverse, norm } => {
                    let shape = &node.shape;
                    let meta = rlx_ir::fft::fft_meta(shape);
                    let dtype = shape.dtype();
                    assert!(
                        matches!(
                            dtype,
                            rlx_ir::DType::F32 | rlx_ir::DType::F64 | rlx_ir::DType::C64
                        ),
                        "rlx-metal Op::Fft requires F32, F64, or C64, got {dtype:?}"
                    );
                    // Fused real→complex: read `signal` directly with im=0.
                    #[cfg(feature = "native-gpu-fft")]
                    let (src_id, real_input) = match fft_real_src.get(&node.id) {
                        Some(&sig) => (sig, true),
                        None => (node.inputs[0], false),
                    };
                    #[cfg(not(feature = "native-gpu-fft"))]
                    let (src_id, real_input) = (node.inputs[0], false);
                    Thunk::Fft1d {
                        src: off(src_id),
                        dst: off(node.id),
                        outer: meta.outer as u32,
                        n_complex: meta.n_complex as u32,
                        inverse: *inverse,
                        norm_tag: norm.tag(),
                        dtype,
                        real_input,
                    }
                }

                Op::LogMel => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                    Thunk::LogMel {
                        spec: off(node.inputs[0]),
                        filters: off(node.inputs[1]),
                        dst: off(node.id),
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    }
                }

                Op::LogMelBackward => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                    Thunk::LogMelBackward {
                        spec: off(node.inputs[0]),
                        filters: off(node.inputs[1]),
                        dy: off(node.inputs[2]),
                        dst: off(node.id),
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    }
                }

                Op::WelchPeaks { k, n_segments } => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                        .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                    Thunk::WelchPeaks {
                        spec: off(node.inputs[0]),
                        dst: off(node.id),
                        welch_batch: meta.welch_batch as u32,
                        n_fft: meta.n_fft as u32,
                        n_segments: meta.n_segments as u32,
                        k: meta.k as u32,
                    }
                }

                Op::RngNormal {
                    mean,
                    scale,
                    key,
                    op_seed,
                } => Thunk::RngNormal {
                    dst: off(node.id),
                    len: node.shape.num_elements().unwrap_or(0) as u32,
                    mean: *mean,
                    scale: *scale,
                    key: *key,
                    op_seed: *op_seed,
                },

                Op::RngUniform {
                    low,
                    high,
                    key,
                    op_seed,
                } => Thunk::RngUniform {
                    dst: off(node.id),
                    len: node.shape.num_elements().unwrap_or(0) as u32,
                    low: *low,
                    high: *high,
                    key: *key,
                    op_seed: *op_seed,
                },

                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    let q_shape = &graph.node(node.inputs[0]).shape;
                    let q_f16 = matches!(q_shape.dtype(), rlx_ir::DType::F16);
                    let state_off = if *carry_state { off(node.inputs[5]) } else { 0 };
                    Thunk::GatedDeltaNet {
                        q: off(node.inputs[0]),
                        k: off(node.inputs[1]),
                        v: off(node.inputs[2]),
                        g: off(node.inputs[3]),
                        beta: off(node.inputs[4]),
                        state: state_off,
                        dst: off(node.id),
                        batch: q_shape.dim(0).unwrap_static() as u32,
                        seq: q_shape.dim(1).unwrap_static() as u32,
                        heads: q_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                        f16: q_f16,
                    }
                }

                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    // Logits [batch, vocab] (or [vocab] → batch=1).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, vocab) = if in_shape.rank() >= 2 {
                        (
                            in_shape.dim(0).unwrap_static(),
                            in_shape.dim(in_shape.rank() - 1).unwrap_static(),
                        )
                    } else {
                        (1, in_shape.num_elements().unwrap_or(0))
                    };
                    Thunk::Sample {
                        logits: off(node.inputs[0]),
                        dst: off(node.id),
                        batch: batch as u32,
                        vocab: vocab as u32,
                        top_k: *top_k as u32,
                        top_p: *top_p,
                        temperature: *temperature,
                        seed: *seed,
                    }
                }

                Op::Reverse { axes } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let mut rev_mask = vec![false; rank];
                    for &a in axes {
                        if a < rank {
                            rev_mask[a] = true;
                        }
                    }
                    Thunk::Reverse {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        dims,
                        rev_mask,
                        elem_bytes: in_shape.dtype().size_bytes() as u8,
                    }
                }

                Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let reduced = in_shape.dim(*axis).unwrap_static();
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    Thunk::ArgReduce {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        outer: outer as u32,
                        reduced: reduced as u32,
                        inner: inner as u32,
                        is_max: matches!(node.op, Op::ArgMax { .. }),
                    }
                }

                Op::SelectiveScan { state_size } => {
                    // x [b, s, h]; delta [b, s, h]; a [h, n]; b,c [b, s, n].
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::SelectiveScan {
                        x: off(node.inputs[0]),
                        delta: off(node.inputs[1]),
                        a: off(node.inputs[2]),
                        b: off(node.inputs[3]),
                        c: off(node.inputs[4]),
                        dst: off(node.id),
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        hidden: x_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                    }
                }

                Op::Lstm {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let (h0, c0) = if *carry {
                        (off(node.inputs[4]), off(node.inputs[5]))
                    } else {
                        (0, 0)
                    };
                    Thunk::Lstm {
                        x: off(node.inputs[0]),
                        w_ih: off(node.inputs[1]),
                        w_hh: off(node.inputs[2]),
                        bias: off(node.inputs[3]),
                        h0,
                        c0,
                        dst: off(node.id),
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    }
                }

                Op::Gru {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h0 = if *carry { off(node.inputs[5]) } else { 0 };
                    Thunk::Gru {
                        x: off(node.inputs[0]),
                        w_ih: off(node.inputs[1]),
                        w_hh: off(node.inputs[2]),
                        b_ih: off(node.inputs[3]),
                        b_hh: off(node.inputs[4]),
                        h0,
                        dst: off(node.id),
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    }
                }

                Op::Rnn {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h0 = if *carry { off(node.inputs[4]) } else { 0 };
                    Thunk::Rnn {
                        x: off(node.inputs[0]),
                        w_ih: off(node.inputs[1]),
                        w_hh: off(node.inputs[2]),
                        bias: off(node.inputs[3]),
                        h0,
                        dst: off(node.id),
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                        relu: *relu,
                    }
                }

                Op::Mamba2 {
                    head_dim,
                    state_size,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::Mamba2 {
                        x: off(node.inputs[0]),
                        dt: off(node.inputs[1]),
                        a: off(node.inputs[2]),
                        b: off(node.inputs[3]),
                        c: off(node.inputs[4]),
                        dst: off(node.id),
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        heads: x_shape.dim(2).unwrap_static() as u32,
                        head_dim: *head_dim as u32,
                        state_size: *state_size as u32,
                    }
                }

                Op::ScaledMatMul {
                    lhs_format,
                    rhs_format,
                    scale_layout,
                    has_bias,
                } => {
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let lhs_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = lhs_total / m.max(1);
                    Thunk::ScaledMatMul {
                        lhs: off(node.inputs[0]),
                        rhs: off(node.inputs[1]),
                        lhs_scale: off(node.inputs[2]),
                        rhs_scale: off(node.inputs[3]),
                        bias: if *has_bias {
                            off(node.inputs[4])
                        } else {
                            usize::MAX
                        },
                        dst: off(node.id),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        lhs_fmt: *lhs_format,
                        rhs_fmt: *rhs_format,
                        layout: *scale_layout,
                        has_bias: *has_bias,
                    }
                }
                Op::ScaledQuantize {
                    format,
                    scale_layout,
                } => {
                    let xs = &graph.node(node.inputs[0]).shape;
                    let cols = xs.dim(xs.rank() - 1).unwrap_static();
                    let rows = xs.num_elements().unwrap() / cols.max(1);
                    Thunk::ScaledQuantize {
                        x: off(node.inputs[0]),
                        scale: off(node.inputs[1]),
                        dst: off(node.id),
                        rows: rows as u32,
                        cols: cols as u32,
                        fmt: *format,
                        layout: *scale_layout,
                    }
                }
                Op::ScaledDequantize {
                    format,
                    scale_layout,
                } => {
                    // Logical shape from the codes (input 0): U8 codes → f32.
                    let xs = &graph.node(node.inputs[0]).shape;
                    let cols = xs.dim(xs.rank() - 1).unwrap_static();
                    let rows = xs.num_elements().unwrap() / cols.max(1);
                    Thunk::ScaledDequantize {
                        codes: off(node.inputs[0]),
                        scale: off(node.inputs[1]),
                        dst: off(node.id),
                        rows: rows as u32,
                        cols: cols as u32,
                        fmt: *format,
                        layout: *scale_layout,
                    }
                }
                Op::ScaledQuantScale {
                    format,
                    scale_layout,
                } => {
                    let xs = &graph.node(node.inputs[0]).shape;
                    let cols = xs.dim(xs.rank() - 1).unwrap_static();
                    let rows = xs.num_elements().unwrap() / cols.max(1);
                    Thunk::ScaledQuantScale {
                        x: off(node.inputs[0]),
                        dst: off(node.id),
                        rows: rows as u32,
                        cols: cols as u32,
                        fmt: *format,
                        layout: *scale_layout,
                    }
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::quant::QuantScheme;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    if scheme.is_gguf() {
                        Thunk::DequantMatMulGguf {
                            x: off(node.inputs[0]),
                            w_q: off(node.inputs[1]),
                            dst: off(node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            scheme: *scheme,
                        }
                    } else {
                        match scheme {
                            QuantScheme::Nvfp4Block => Thunk::DequantMatMulNvfp4 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                global_scale: off(node.inputs[3]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                            },
                            QuantScheme::Int8Block { block_size } => Thunk::DequantMatMulInt8 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                zp: off(node.inputs[3]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                block_size: *block_size,
                                is_asymmetric: false,
                            },
                            QuantScheme::Int8BlockAsym { block_size } => Thunk::DequantMatMulInt8 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                zp: off(node.inputs[3]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                block_size: *block_size,
                                is_asymmetric: true,
                            },
                            QuantScheme::Int4Block { block_size } => Thunk::DequantMatMulInt4 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                zp: off(node.inputs[3]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                block_size: *block_size,
                                is_asymmetric: false,
                            },
                            QuantScheme::Fp8E4m3 => Thunk::DequantMatMulFp8 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                e5m2: false,
                            },
                            QuantScheme::Fp8E5m2 => Thunk::DequantMatMulFp8 {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                e5m2: true,
                            },
                            other => panic!(
                                "rlx-metal: Op::DequantMatMul legacy scheme {other:?} \
                                 is CPU-only unless Int4/FP8/NVFP4; use GGUF K-quants or Device::Cpu."
                            ),
                        }
                    }
                }

                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal RmsNormBackward: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let rows = (x_shape.num_elements().unwrap() / h) as u32;
                    let common = (
                        off(node.inputs[0]),
                        off(node.inputs[1]),
                        off(node.inputs[2]),
                        off(node.inputs[3]),
                        rows,
                        h as u32,
                        *eps,
                    );
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => Thunk::RmsNormBackwardInput {
                            x: common.0,
                            gamma: common.1,
                            beta: common.2,
                            dy: common.3,
                            dx: off(node.id),
                            rows: common.4,
                            h: common.5,
                            eps: common.6,
                        },
                        Op::RmsNormBackwardGamma { .. } => Thunk::RmsNormBackwardGamma {
                            x: common.0,
                            gamma: common.1,
                            beta: common.2,
                            dy: common.3,
                            dgamma: off(node.id),
                            rows: common.4,
                            h: common.5,
                            eps: common.6,
                        },
                        Op::RmsNormBackwardBeta { .. } => Thunk::RmsNormBackwardBeta {
                            x: common.0,
                            gamma: common.1,
                            beta: common.2,
                            dy: common.3,
                            dbeta: off(node.id),
                            rows: common.4,
                            h: common.5,
                            eps: common.6,
                        },
                        _ => unreachable!(),
                    }
                }

                Op::RopeBackward { head_dim, n_rot } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal RopeBackward: F32 only");
                    }
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                        (
                            dy_shape.dim(0).unwrap_static(),
                            dy_shape.dim(1).unwrap_static(),
                            dy_shape.dim(2).unwrap_static(),
                        )
                    } else {
                        (
                            1,
                            dy_shape.dim(0).unwrap_static(),
                            dy_shape.dim(1).unwrap_static(),
                        )
                    };
                    let cos_len = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    Thunk::RopeBackward {
                        dy: off(node.inputs[0]),
                        cos: off(node.inputs[1]),
                        sin: off(node.inputs[2]),
                        dx: off(node.id),
                        batch: batch as u32,
                        seq: seq as u32,
                        hidden: hidden as u32,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        cos_len: cos_len as u32,
                    }
                }

                Op::CumsumBackward { exclusive, .. } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal CumsumBackward: F32 only");
                    }
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static();
                    let rows = dy_shape.num_elements().unwrap() / cols;
                    Thunk::CumsumBackward {
                        dy: off(node.inputs[0]),
                        dx: off(node.id),
                        rows: rows as u32,
                        cols: cols as u32,
                        exclusive: *exclusive,
                    }
                }

                Op::GatherBackward { .. } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal GatherBackward: F32 only");
                    }
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
                    Thunk::GatherBackward {
                        dy: off(node.inputs[0]),
                        indices: off(node.inputs[1]),
                        dst: off(node.id),
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                    }
                }

                Op::MaxPool2dBackward {
                    kernel_size,
                    stride,
                    padding,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal MaxPool2dBackward: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    Thunk::MaxPool2dBackward {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dx: off(node.id),
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c: x_shape.dim(1).unwrap_static() as u32,
                        h: x_shape.dim(2).unwrap_static() as u32,
                        w: x_shape.dim(3).unwrap_static() as u32,
                        h_out: dy_shape.dim(2).unwrap_static() as u32,
                        w_out: dy_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                    }
                }

                Op::Conv2dBackwardInput {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal Conv2dBackwardInput: F32 only");
                    }
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let _w_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    Thunk::Conv2dBackwardInput {
                        dy: off(node.inputs[0]),
                        w: off(node.inputs[1]),
                        dx: off(node.id),
                        n: out_shape.dim(0).unwrap_static() as u32,
                        c_in: out_shape.dim(1).unwrap_static() as u32,
                        h: out_shape.dim(2).unwrap_static() as u32,
                        w_in: out_shape.dim(3).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        h_out: dy_shape.dim(2).unwrap_static() as u32,
                        w_out: dy_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    }
                }

                Op::Conv2dBackwardWeight {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal Conv2dBackwardWeight: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    let _dw_shape = &node.shape;
                    Thunk::Conv2dBackwardWeight {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dw: off(node.id),
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c_in: x_shape.dim(1).unwrap_static() as u32,
                        h: x_shape.dim(2).unwrap_static() as u32,
                        w: x_shape.dim(3).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        h_out: dy_shape.dim(2).unwrap_static() as u32,
                        w_out: dy_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride.first().copied().unwrap_or(1) as u32,
                        sw: stride.get(1).copied().unwrap_or(1) as u32,
                        ph: padding.first().copied().unwrap_or(0) as u32,
                        pw: padding.get(1).copied().unwrap_or(0) as u32,
                        dh: dilation.first().copied().unwrap_or(1) as u32,
                        dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    }
                }

                Op::Custom { name, attrs, .. } => {
                    let kernel =
                        crate::op_registry::lookup_metal_kernel(name).unwrap_or_else(|| {
                            panic!(
                                "rlx-metal: no MetalKernel registered for \
                             Op::Custom('{name}'). Either register one via \
                             rlx_metal::op_registry::register_metal_kernel \
                             or pin this graph to Device::Cpu."
                            )
                        });
                    let inputs_v: Vec<(usize, u32, Shape)> = node
                        .inputs
                        .iter()
                        .map(|&in_id| {
                            let s = graph.node(in_id).shape.clone();
                            let len = s.num_elements().unwrap_or(0) as u32;
                            (off(in_id), len, s)
                        })
                        .collect();
                    let out_len = node.shape.num_elements().unwrap_or(0) as u32;
                    Thunk::CustomOp {
                        kernel,
                        inputs: inputs_v,
                        output: (off(node.id), out_len, node.shape.clone()),
                        attrs: attrs.clone(),
                    }
                }

                // Standalone nearest 2× upsample: the region-marking pass wraps
                // a bare `Op::ResizeNearest2x` into a single-step TransformRegion.
                // Emit the native resize thunk (same as the bare arm above).
                Op::TransformRegion { steps, .. }
                    if steps.len() == 1
                        && matches!(
                            steps[0],
                            rlx_ir::op::TransformStep::ResizeNearest2x(
                                rlx_ir::op::ChainOperand::Input(0)
                            )
                        ) =>
                {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    Thunk::ResizeNearest2x {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                other => panic!(
                    "rlx-metal: Op::{:?} (kind {:?}) not yet implemented on Metal. \
                     Either pin this graph to a backend that supports it (Device::Cpu, \
                     Device::Mlx) or add a Thunk variant for it. Silently emitting Nop \
                     in the past caused runtime corruption — make the gap explicit.",
                    other.kind(),
                    other.kind()
                ),
            };
            thunks.push(t);
        }

        // ── Narrow → Rope thunk fusion (plan #45 Metal parity) ───
        // Mirrors the CPU pass: for each Narrow whose only consumer is
        // an immediately-following Rope, rewrite the Rope to read from
        // the Narrow's source with the parent's row stride; the Narrow
        // becomes a Nop. Saves the intermediate Q/K write on the GPU
        // and one kernel dispatch per pair.
        if !rlx_ir::env::flag("RLX_METAL_DISABLE_NARROW_ROPE_FUSE") {
            {
                use std::collections::HashMap;
                // Count reads of every byte-offset across the schedule.
                let mut read_counts: HashMap<usize, usize> = HashMap::new();
                for t in &thunks {
                    for off in metal_thunk_read_offsets(t) {
                        *read_counts.entry(off).or_insert(0) += 1;
                    }
                }
                for i in 0..thunks.len().saturating_sub(1) {
                    // Metal Narrow stores `start` separately (in elements),
                    // not folded into `src`. To make Rope read from the
                    // parent buffer at the right column we have to bake
                    // `start` into the byte offset using the dtype size.
                    let (n_src, n_dst, n_src_axis, n_start, n_dt) = match &thunks[i] {
                        Thunk::Narrow {
                            src,
                            dst,
                            src_axis,
                            start,
                            dt,
                            ..
                        } => (*src, *dst, *src_axis, *start, *dt),
                        _ => continue,
                    };
                    let mut j = i + 1;
                    while j < thunks.len() && matches!(thunks[j], Thunk::Nop) {
                        j += 1;
                    }
                    if j >= thunks.len() {
                        continue;
                    }
                    let rope_reads_narrow = matches!(&thunks[j],
                    Thunk::Rope { src, .. } if *src == n_dst);
                    if !rope_reads_narrow {
                        continue;
                    }
                    if read_counts.get(&n_dst).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    // Sanity: the Rope's dtype must match the Narrow's. If
                    // not, something upstream did a precision conversion
                    // and the buffers aren't byte-compatible — bail.
                    let dt_matches = matches!(&thunks[j],
                    Thunk::Rope { dt: rd, .. } if *rd == n_dt);
                    if !dt_matches {
                        continue;
                    }

                    let elem_bytes = match n_dt {
                        HalfFlag::F32 => 4usize,
                        HalfFlag::F16 => 2usize,
                    };
                    if let Thunk::Rope {
                        src,
                        src_row_stride,
                        ..
                    } = &mut thunks[j]
                    {
                        *src = n_src + n_start as usize * elem_bytes;
                        *src_row_stride = n_src_axis;
                    }
                    thunks[i] = Thunk::Nop;
                }
            }
        }

        rewrite_simple_elementwise_regions(&mut thunks);
        rewrite_dense_binary_broadcast(&mut thunks);
        let output_offsets: std::collections::HashSet<usize> =
            graph.outputs.iter().map(|&id| off(id)).collect();
        fuse_decode_mlp_combined_gate_up(&mut thunks, &output_offsets);
        fuse_narrow_clusters(&mut thunks);

        // Fused decode-layer MLP (m == 1 packed SwiGLU/GeGLU). Off-switch:
        // RLX_METAL_FUSE_DECODE=0. Output offsets stay live (never fused away).
        fuse_decode_mlp(&mut thunks, &output_offsets);

        Self {
            thunks,
            rng: rng_shared,
        }
    }
}

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
        Cast { src, dst, .. } => (vec![*src], vec![*dst]),
        Copy { src, dst, .. } => (vec![*src], vec![*dst]),
        ActivationInPlace { data, .. } => (vec![*data], vec![*data]),
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

/// Packed gate/up weight bytes per output column for fused decode MLP GEMV.
fn mlp_gate_up_row_bytes(k: u32, scheme: rlx_ir::quant::QuantScheme) -> usize {
    use rlx_ir::quant::QuantScheme;
    match scheme {
        QuantScheme::GgufQ4K => (k as usize / 256) * 144,
        QuantScheme::GgufQ5_0 => ((k as usize + 31) / 32) * 22,
        QuantScheme::GgufQ6K => (k as usize / 256) * 210,
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
            }
        )
    };
    let is_gelu = |t: &Thunk| {
        matches!(
            t,
            Thunk::ActivationInPlace {
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
        if use_gelu && rlx_ir::env::var("RLX_METAL_FUSE_DECODE_GELU").as_deref() != Some("1") {
            i += 1;
            continue;
        }

        let gate_src_off = match &thunks[act_idx] {
            Thunk::ActivationInPlace { data, .. } => *data,
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
        if comb_n != 2 * n_half {
            i += 1;
            continue;
        }
        if use_gelu && prod == comb_x {
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
        if !dead_ok || !no_output_clash {
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

    // Q4_K decode matmul (m == 1) at `idx` writing `dst`? Return (x, w_q, k, n).
    let as_packed_gate_up_mm = |t: &Thunk| -> Option<(usize, usize, usize, u32, u32, QuantScheme)> {
        if let Thunk::DequantMatMulGguf {
            x,
            w_q,
            dst,
            m,
            k,
            n,
            scheme,
        } = *t
        {
            if m == 1 && matches!(scheme, QuantScheme::GgufQ4K | QuantScheme::GgufQ5_0) {
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
                }
            )
        };
        let is_gelu = |t: &Thunk| {
            matches!(
                t,
                Thunk::ActivationInPlace {
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
        // GeGLU: opt-in until full-graph arena aliasing is resolved (`RLX_METAL_FUSE_DECODE_GELU=1`).
        if use_gelu && rlx_ir::env::var("RLX_METAL_FUSE_DECODE_GELU").as_deref() != Some("1") {
            i += 1;
            continue;
        }

        let gate_src_off = match &thunks[act_idx] {
            Thunk::ActivationInPlace { data, .. } => *data,
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
        if use_gelu && prod == gate_x {
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
        let down_mm_idx = mlp_find_forward(
            thunks,
            mul_idx,
            |t| matches!(t, Thunk::DequantMatMulGguf { x, m: 1, scheme: QuantScheme::GgufQ4K | QuantScheme::GgufQ5_0 | QuantScheme::GgufQ6K, .. } if *x == prod),
        );
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
        let (res_off, out_off) = match &thunks[add_idx] {
            Thunk::BinaryFull { lhs, rhs, dst, .. } => {
                let res = if *lhs == down_dst { *rhs } else { *lhs };
                (res, *dst)
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
        if !dead_ok || !no_output_clash {
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
        Thunk::RopeBackward { dy, cos, sin, .. } => vec![*dy, *cos, *sin],
        Thunk::CumsumBackward { dy, .. } => vec![*dy],
        Thunk::GatherBackward { dy, indices, .. } => vec![*dy, *indices],
        Thunk::MaxPool2dBackward { x, dy, .. } => vec![*x, *dy],
        Thunk::Conv2dBackwardInput { dy, w, .. } => vec![*dy, *w],
        Thunk::Conv2dBackwardWeight { x, dy, .. } => vec![*x, *dy],
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
