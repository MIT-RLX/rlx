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
    /// Reshape / Cast / Expand: copy len elements
    Copy {
        src: usize,
        dst: usize,
        len: u32,
        dt: HalfFlag,
    },
    /// SDPA. `mask_kind` encodes how to apply masking inside the
    /// kernel:
    ///   0 = None         (no masking)
    ///   1 = Causal       (prefill: upper-triangular fill in-kernel)
    ///   2 = Custom       (read binary mask buffer `mask`)
    /// SlidingWindow lowering is not yet wired — it would map to a
    /// new `mask_kind == 3` plus a `window` parameter.
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
        dt: HalfFlag,
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
        dt: HalfFlag,
        src_row_stride: u32,
    },
    /// Softmax
    Softmax {
        data: usize,
        rows: u32,
        cols: u32,
        dt: HalfFlag,
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

    /// 1D FFT on the 2N-real-block layout, lowered from `Op::Fft`.
    /// v1 is a host fallback against the unified-memory arena: same
    /// sync pattern as `CustomOp` (commit, wait, run, restart). On
    /// Apple Silicon `Buffer::contents()` is host-addressable for
    /// shared-storage buffers, so this is sync overhead only — no
    /// copy. A native Metal compute kernel will replace this when a
    /// workload makes the GPU/CPU sync the bottleneck.
    Fft1d {
        src: usize,
        dst: usize,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        dtype: rlx_ir::DType,
    },
}

pub struct ThunkSchedule {
    pub thunks: Vec<Thunk>,
}

/// Static-string name for each Thunk variant — used by the Perfetto
/// trace layer (PLAN L3) to label per-step events without allocating.
pub fn thunk_name(t: &Thunk) -> &'static str {
    match t {
        Thunk::Nop => "nop",
        Thunk::Cast { .. } => "cast",
        Thunk::Sgemm { .. } => "sgemm",
        Thunk::BatchedSgemm { .. } => "batched_sgemm",
        Thunk::FusedMmBiasAct { .. } => "fused_mm_bias_act",
        Thunk::ActivationInPlace { .. } => "activation",
        Thunk::LayerNorm { .. } => "layer_norm",
        Thunk::RmsNorm { .. } => "rms_norm",
        Thunk::BinaryFull { .. } => "binary",
        Thunk::BinaryBroadcast { .. } => "binary_broadcast",
        Thunk::BiasAdd { .. } => "bias_add",
        Thunk::FusedResidualLN { .. } => "fused_residual_ln",
        Thunk::Gather { .. } => "gather",
        Thunk::Narrow { .. } => "narrow",
        Thunk::Copy { .. } => "copy",
        Thunk::Attention { .. } => "attention",
        Thunk::Rope { .. } => "rope",
        Thunk::Softmax { .. } => "softmax",
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
        Thunk::ElementwiseRegion { .. } => "elementwise_region",
        Thunk::CustomOp { .. } => "custom_op",
        Thunk::Fft1d { .. } => "fft1d",
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
            | Thunk::Sgemm { .. }
            | Thunk::BatchedSgemm { .. }
            | Thunk::FusedMmBiasAct { .. }
            | Thunk::BiasAdd { .. }
            | Thunk::LayerNorm { .. }
            | Thunk::RmsNorm { .. }
            | Thunk::Softmax { .. }
            | Thunk::FusedResidualLN { .. }
            | Thunk::Gather { .. }
            | Thunk::Compare { .. }
            | Thunk::Where { .. }
            | Thunk::FusedSwiGLU { .. }
            | Thunk::ElementwiseRegion { .. }
            | Thunk::Narrow { .. }
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
            Thunk::Rope { .. } => true,
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
        let mut thunks = Vec::with_capacity(graph.len());

        let off = |id| -> usize {
            if arena.has_buffer(id) {
                arena.byte_offset(id)
            } else {
                usize::MAX
            }
        };

        for node in graph.nodes() {
            // View ops alias their parent's slot (planner did this); the
            // GPU thunk path also emits Nop. Plan #46.
            if rlx_opt::is_pure_view(graph, node) {
                thunks.push(Thunk::Nop);
                continue;
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
                } => {
                    let mask_kind_u32: u32 = match mask_kind {
                        rlx_ir::op::MaskKind::None => 0,
                        rlx_ir::op::MaskKind::Causal => 1,
                        rlx_ir::op::MaskKind::Custom => 2,
                        rlx_ir::op::MaskKind::Bias => 3,
                        rlx_ir::op::MaskKind::SlidingWindow(_) => {
                            panic!(
                                "Metal SDPA: MaskKind::SlidingWindow not yet supported (use Causal or Custom)"
                            );
                        }
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
                    let (batch, seq) = if q_shape.rank() >= 3 {
                        (
                            q_shape.dim(0).unwrap_static(),
                            q_shape.dim(1).unwrap_static(),
                        )
                    } else {
                        (1, q_shape.dim(0).unwrap_static())
                    };
                    let kv_seq = if k_shape.rank() >= 3 {
                        k_shape.dim(1).unwrap_static()
                    } else {
                        k_shape.dim(0).unwrap_static()
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
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::Rope { head_dim } => {
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
                    Thunk::Rope {
                        src: off(node.inputs[0]),
                        cos: off(node.inputs[1]),
                        sin: off(node.inputs[2]),
                        dst: off(node.id),
                        batch: batch as u32,
                        seq: seq as u32,
                        hidden: hidden as u32,
                        head_dim: *head_dim as u32,
                        dt: node.shape.dtype().into(),
                        src_row_stride: hidden as u32,
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
                            let in_axis = in_shape.dim(*axis).unwrap_static();
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

                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                } => {
                    use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, CmpOp};
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
                            // Pack the 3-operand select into the 4-u32 step: the
                            // op_sub slot carries the condition operand, lhs is
                            // on_true, rhs is on_false. Kernel switches on
                            // op_kind == 4 to read all three back.
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
                    Thunk::ElementwiseRegion {
                        len: node.shape.num_elements().unwrap() as u32,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst: off(node.id),
                        input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        input_modulus: *input_modulus,
                    }
                }

                Op::FusedSwiGLU { cast_to } => {
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
                    }
                }

                Op::Fft { inverse } => {
                    // Host-fallback FFT — see Thunk::Fft1d doc.
                    let shape = &node.shape;
                    let last = shape.dim(shape.rank() - 1).unwrap_static();
                    let n_complex = (last / 2) as u32;
                    let total = shape.num_elements().unwrap_or(0);
                    let outer = (total / last) as u32;
                    let dtype = shape.dtype();
                    assert!(
                        matches!(dtype, rlx_ir::DType::F32 | rlx_ir::DType::F64),
                        "rlx-metal Op::Fft host fallback requires F32/F64, got {dtype:?}"
                    );
                    Thunk::Fft1d {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        outer,
                        n_complex,
                        inverse: *inverse,
                        dtype,
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

        Self { thunks }
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
        Thunk::BinaryBroadcast { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::ActivationInPlace { data, .. } => vec![*data],
        Thunk::LayerNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::RmsNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::FusedResidualLN {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::Softmax { data, .. } => vec![*data],
        Thunk::Attention { q, k, v, mask, .. } => vec![*q, *k, *v, *mask],
        Thunk::Rope { src, cos, sin, .. } => vec![*src, *cos, *sin],
        Thunk::FusedSwiGLU { src, .. } => vec![*src],
        Thunk::Concat { inputs, .. } => inputs.iter().map(|(o, _)| *o).collect(),
        Thunk::Narrow { src, .. } => vec![*src],
        Thunk::Copy { src, .. } => vec![*src],
        _ => vec![],
    }
}
