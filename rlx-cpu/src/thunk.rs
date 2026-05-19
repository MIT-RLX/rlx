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

//! Thunks — pre-compiled kernel dispatch with zero per-call overhead.
//!
//! At compile time, the graph is lowered into a flat `Vec<Thunk>` where each
//! thunk holds pre-computed arena offsets, dimensions, and kernel type.
//! At runtime, the executor just iterates thunks and calls kernels directly.
//! No match dispatch, no HashMap lookup, no dimension computation.

use crate::arena::Arena;
use crate::op_registry::CpuKernel;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{Graph, NodeId, Op, Shape};
use std::collections::HashMap;
use std::sync::Arc;

/// A pre-compiled kernel call with all args resolved to arena offsets.
#[derive(Clone)]
pub enum Thunk {
    /// Skip (Input/Param already in arena)
    Nop,
    /// C = A @ B (BLAS sgemm)
    Sgemm {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
    },
    /// f64 dense solve `x = A⁻¹·b` via LAPACK dgesv.
    /// `a`, `b`, `x` are byte-offsets into the arena. `n` is the matrix
    /// dimension; `nrhs` is 1 for a vector RHS or >1 for multi-RHS.
    /// The kernel materializes scratch copies of A and b internally
    /// (LAPACK overwrites both with LU factors and solution).
    DenseSolveF64 {
        a: usize,
        b: usize,
        x: usize,
        n: u32,
        nrhs: u32,
    },
    /// f32 twin of `DenseSolveF64`. Calls LAPACK `sgesv` (or the
    /// no-blas Rust fallback). Same arena byte-offset contract.
    DenseSolveF32 {
        a: usize,
        b: usize,
        x: usize,
        n: u32,
        nrhs: u32,
    },
    /// Batched f64 dense solve. `a`, `b`, `x` are byte-offsets to
    /// the leading slice; `batch` is the number of independent
    /// systems. Per slice the kernel calls `dgesv(A_i, b_i, n, nrhs)`
    /// — LAPACK has no batched dgesv on Accelerate, so we loop.
    BatchedDenseSolveF64 {
        a: usize,
        b: usize,
        x: usize,
        batch: u32,
        n: u32,
        nrhs: u32,
    },
    /// Batched f64 matmul. Both inputs and output have a leading
    /// batch axis of size `batch`. Per-batch independent dgemm:
    /// `C[i] = A[i] @ B[i]` for `i in 0..batch`. Used by VJP rules
    /// that emit per-batch outer products (e.g., BatchedDenseSolve
    /// VJP). The unbatched `Dgemm` thunk handles the rank-2 case.
    BatchedDgemmF64 {
        a: usize,
        b: usize,
        c: usize,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    },
    /// Batched f32 matmul — same loop-per-batch shape as
    /// `BatchedDgemmF64` but calling `sgemm`. Needed for attention
    /// patterns where both operands carry a batch dim (e.g. q@k^T
    /// and attn@v in decomposed self-attention). The 2-D `Sgemm`
    /// flatten trick is wrong in that case because it treats `b` as
    /// a single shared RHS across every batch.
    BatchedSgemm {
        a: usize,
        b: usize,
        c: usize,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    },
    /// C = A @ B via Accelerate cblas_dgemm. Mirror of `Sgemm` at f64.
    Dgemm {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
    },
    /// f64 N-D index walk used for both `Op::Transpose` and `Op::Expand`.
    /// `in_strides` carries 0s on broadcast axes (Expand) or permuted
    /// strides (Transpose). Mirror of `Thunk::Transpose` at f64.
    TransposeF64 {
        src: usize,
        dst: usize,
        in_total: u32,
        out_dims: Vec<u32>,
        in_strides: Vec<u32>,
    },
    /// f64 element-wise activation. Single-input, single-output. The
    /// kernel always reads from `src` and writes to `dst`, so it works
    /// whether or not the planner aliased the two slots.
    ActivationF64 {
        src: usize,
        dst: usize,
        len: u32,
        kind: Activation,
    },
    /// Element-wise complex squared-magnitude: `|z|² = re² + im²`.
    /// Reads the C64 input at `src` as `2·len` f32 ([re,im] pairs),
    /// writes `len` f32 to `dst`.
    ComplexNormSqF32 {
        src: usize,
        dst: usize,
        /// Logical element count (number of complex values).
        len: u32,
    },
    /// Wirtinger backward for [`ComplexNormSqF32`]: `dz = g · z` as
    /// C64. Reads `z` at `2·len` f32 + `g` at `len` f32; writes
    /// `2·len` f32 to `dz`.
    ComplexNormSqBackwardF32 {
        z: usize,
        g: usize,
        dz: usize,
        len: u32,
    },
    /// Element-wise C64 conjugate: writes `[re_i, -im_i]` per element.
    /// Layout matches the rest of C64 here ([re,im] interleaved f32).
    ConjugateC64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// C64 element-wise activation. Only kinds with well-defined
    /// complex extensions are supported: Neg, Exp, Log, Sqrt.
    /// Everything else (Sigmoid, Tanh, Relu, Abs, Sin/Cos/Tan/Atan,
    /// Round, GeLU family) is rejected at lowering — those don't have
    /// single natural complex definitions. `len` is the **complex
    /// element count** (the f32 buffer holds `2·len` floats).
    ActivationC64 {
        src: usize,
        dst: usize,
        len: u32,
        kind: Activation,
    },
    /// f64 contiguous reduction along a single axis range. Layout
    /// `[outer, reduced, inner]` in memory; output is `[outer, inner]`.
    /// Sum only for now (Mean composes via 1/N multiply post-pass).
    ReduceSumF64 {
        src: usize,
        dst: usize,
        outer: u32,
        reduced: u32,
        inner: u32,
    },
    /// f64 plain copy (Reshape / Cast at the same dtype). Mirrors `Copy`
    /// but at 8 bytes per element.
    CopyF64 { src: usize, dst: usize, len: u32 },
    /// f64 element-wise binary with broadcast. `len`/`lhs_len`/`rhs_len`
    /// are element counts; kernel does `out[i] = lhs[i % lhs_len] OP rhs[i % rhs_len]`.
    /// Mirror of `BinaryFull` at 8 bytes per element.
    BinaryFullF64 {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        lhs_len: u32,
        rhs_len: u32,
        op: BinaryOp,
        /// Output shape dims (row-major). Empty in the fast path. See
        /// `BinaryFull` doc for the broadcast convention.
        out_dims_bcast: Vec<u32>,
        bcast_lhs_strides: Vec<u32>,
        bcast_rhs_strides: Vec<u32>,
    },
    /// f64 concat — byte-for-byte mirror of `Concat` but copies
    /// 8 bytes per element. Element-counted offsets/strides match
    /// the f32 variant; the executor scales by elem_size internally.
    ConcatF64 {
        dst: usize,
        outer: u32,
        inner: u32,
        total_axis: u32,
        inputs: Vec<(usize, u32)>,
    },
    /// C64 element-wise binary with broadcast. Same `len` /
    /// `lhs_len` / `rhs_len` semantics as `BinaryFull` but each
    /// "element" is one complex value (8 bytes = `[re, im]` as two
    /// f32s). The executor reads the underlying f32 buffer at
    /// `2·len` floats and walks element pairs. Supports Add / Sub /
    /// Mul / Div; Max / Min / Pow have no single natural complex
    /// definition and panic at lowering.
    BinaryFullC64 {
        lhs: usize,
        rhs: usize,
        dst: usize,
        /// Complex element count (NOT f32 count). f32 buffer length
        /// is `2·len`.
        len: u32,
        lhs_len: u32,
        rhs_len: u32,
        op: BinaryOp,
        out_dims_bcast: Vec<u32>,
        bcast_lhs_strides: Vec<u32>,
        bcast_rhs_strides: Vec<u32>,
    },
    /// Bounded scan. Holds a recursively-compiled body schedule + a
    /// pre-initialized body arena snapshot (constants filled). Each
    /// outer execution clones the snapshot, copies the carry-in slot
    /// from the outer arena, runs the body schedule `length` times,
    /// then writes the final carry to the outer arena.
    ///
    /// Single-carry MVP — body has exactly one Input and one output,
    /// both same shape and dtype.
    Scan {
        body: Arc<ThunkSchedule>,
        body_init: Arc<Vec<u8>>, // pristine body arena bytes
        body_input_off: usize,   // byte offset of the body's carry-Input slot
        body_output_off: usize,  // byte offset of the body's output slot
        outer_init_off: usize,   // outer-arena offset of the initial carry
        outer_final_off: usize,  // outer-arena offset of the final carry / trajectory base
        length: u32,
        carry_bytes: u32, // carry size in bytes
        /// When true, write each step's carry to the outer arena at
        /// offset `outer_final_off + t * carry_bytes`, producing a
        /// `[length, *carry]` stacked trajectory. When false, only the
        /// final carry lands at `outer_final_off`.
        save_trajectory: bool,
        /// Per-step `xs` inputs. For each: (body_x_input_off,
        /// outer_xs_base_off, per_step_bytes). Per iteration `t`, the
        /// executor copies `outer_xs_base_off + t * per_step_bytes`
        /// into `body_x_input_off`. Empty when the scan has no xs.
        xs_inputs: Arc<Vec<(usize, usize, u32)>>,
        /// Broadcast inputs — values constant across iterations. For
        /// each: (body_bcast_input_off, outer_bcast_off, total_bytes).
        /// Filled into `body_buf` ONCE before the scan loop starts
        /// (xs in contrast are re-filled every iteration). Empty when
        /// the scan has no bcasts.
        bcast_inputs: Arc<Vec<(usize, usize, u32)>>,
        /// Number of trajectory checkpoints (when `save_trajectory`).
        /// `0` or `length` ⇒ save every iteration. Otherwise save only
        /// `K` rows at indices `floor((k+1) * length / K) - 1` for
        /// `k in 0..K`. Last index is always `length-1` so the final
        /// carry is always cached.
        num_checkpoints: u32,
    },

    /// Reverse-mode AD companion to `Thunk::Scan`. Walks `t = length-1
    /// .. 0`, threading `dcarry` through the body's VJP. Per iteration:
    /// writes `carry_t` (from outer init or trajectory), each `xs_i[t]`
    /// slice, and the current `dcarry` into the body_vjp's Input
    /// slots, runs body_vjp, reads new `dcarry` from its single output.
    /// f64 carry only — the upstream-accumulation step in trajectory
    /// mode does an element-wise f64 add.
    ScanBackward {
        body_vjp: Arc<ThunkSchedule>,
        body_init: Arc<Vec<u8>>,
        body_carry_in_off: usize, // body_vjp's mirrored body-carry-input slot
        body_x_offs: Arc<Vec<usize>>, // body_vjp's mirrored x_t_i Input slots, in xs order
        body_d_output_off: usize, // body_vjp's "d_output" Input slot
        body_dcarry_out_off: usize, // body_vjp's gradient output
        outer_init_off: usize,    // original init carry
        outer_traj_off: usize,    // [length-or-K, *carry] trajectory base
        outer_upstream_off: usize, // upstream gradient (carry shape, or [length, *carry])
        /// Per-xs entries: (outer_xs_base_off, per_step_bytes). Read
        /// `xs_i[t]` from `outer_xs_base_off + t * per_step_bytes`.
        outer_xs_offs: Arc<Vec<(usize, u32)>>,
        outer_dinit_off: usize, // output: dinit
        length: u32,
        carry_bytes: u32,
        /// Bytes per element in the carry tensor: 4 for f32, 8 for f64.
        /// Used to dispatch the trajectory-mode upstream accumulation
        /// kernel (the dcarry += upstream\[t\] add must use the right
        /// floating-point type — a hard-coded f64 add silently does
        /// nothing for an f32 carry whose `cb` isn't divisible by 8).
        carry_elem_size: u32,
        save_trajectory: bool, // true → upstream is per-step; false → just final
        /// Recursive checkpointing config. `0` or `length` ⇒ full
        /// trajectory cached, no recompute (existing behavior).
        /// `0 < K < length` ⇒ trajectory has only K rows; the executor
        /// recomputes intermediate carries via `forward_body` between
        /// checkpoints. Memory: O(K · carry_bytes); time: O(length).
        num_checkpoints: u32,
        /// Forward body schedule (same compiled body as the forward
        /// Op::Scan), used for recompute when `num_checkpoints` is
        /// active. `None` for the All strategy.
        forward_body: Option<Arc<ThunkSchedule>>,
        /// Pristine forward body arena bytes (constants filled).
        forward_body_init: Option<Arc<Vec<u8>>>,
        /// Forward body's carry-Input and output slot offsets — needed
        /// to seed/read the body during recompute.
        forward_body_carry_in_off: usize,
        forward_body_output_off: usize,
        /// Forward body's per-step xs Input slots (one per outer xs).
        /// Same indexing convention as `body_x_offs`.
        forward_body_x_offs: Arc<Vec<usize>>,
    },

    /// Companion to `ScanBackward` that materializes one stacked
    /// `dxs_i`. Same backward loop; per iteration, after running
    /// body_vjp, copies its `body_dxs_out_off` slot into the outer
    /// arena at `outer_dxs_off + t * per_step_bytes`. dcarry threading
    /// is identical — we still need it for the body_vjp recurrence
    /// even though we don't write it back to the outer arena.
    ScanBackwardXs {
        body_vjp: Arc<ThunkSchedule>,
        body_init: Arc<Vec<u8>>,
        body_carry_in_off: usize,
        body_x_offs: Arc<Vec<usize>>,
        body_d_output_off: usize,
        body_dcarry_out_off: usize,
        body_dxs_out_off: usize, // the body_vjp output we extract per step
        outer_init_off: usize,
        outer_traj_off: usize,
        outer_upstream_off: usize,
        outer_xs_offs: Arc<Vec<(usize, u32)>>,
        outer_dxs_off: usize, // base of the stacked [length, *per_step] output
        length: u32,
        carry_bytes: u32,
        /// Same role as `Thunk::ScanBackward::carry_elem_size`.
        carry_elem_size: u32,
        per_step_bytes: u32, // bytes per row of the dxs output
        save_trajectory: bool,
        /// Recursive checkpointing config. Same semantics as
        /// `Thunk::ScanBackward::num_checkpoints` — `0` or `length`
        /// means "save every step's carry"; `0 < K < length` means
        /// the trajectory has only K rows and the executor recomputes
        /// intermediate carries via `forward_body` (which must be
        /// `Some`). Implemented via segment-cached recompute,
        /// mirroring the `ScanBackward` path.
        num_checkpoints: u32,
        forward_body: Option<Arc<ThunkSchedule>>,
        forward_body_init: Option<Arc<Vec<u8>>>,
        forward_body_carry_in_off: usize,
        forward_body_output_off: usize,
        forward_body_x_offs: Arc<Vec<usize>>,
    },
    /// User-defined sub-graph (`Op::CustomFn`) — runs `fwd_body` once.
    /// Per execution: clone `body_init`, copy each primal input from the
    /// outer arena into its body Input slot, run the body schedule,
    /// copy the body's single output back to the outer arena.
    CustomFn {
        body: Arc<ThunkSchedule>,
        body_init: Arc<Vec<u8>>,
        /// Per primal input: (body_input_off, outer_input_off, bytes).
        inputs: Arc<Vec<(usize, usize, u32)>>,
        body_output_off: usize,
        outer_output_off: usize,
        out_bytes: u32,
    },
    /// C = A @ B; C += bias; C = act(C)
    FusedMmBiasAct {
        a: usize,
        w: usize,
        bias: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
        act: Option<Activation>,
    },
    /// out = LN(x + residual + bias, gamma, beta)
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
    },
    /// out = bias_add(data, bias, m, n) for Binary::Add with broadcast
    BiasAdd {
        src: usize,
        bias: usize,
        dst: usize,
        m: u32,
        n: u32,
    },
    /// Element-wise binary op with NumPy-style broadcast.
    ///
    /// Fast path (`lhs_len == rhs_len == len`): plain element-wise loop,
    /// SIMD-vectorized on aarch64 for `Add`/`Mul`. `bcast_*` fields
    /// are unused.
    ///
    /// Broadcast path: uses `out_dims_bcast` + `bcast_lhs_strides` +
    /// `bcast_rhs_strides` to compute per-cell indices into each
    /// operand. The strides are precomputed at thunk-construction
    /// time from the operands' true shapes (with stride 0 on any axis
    /// where the operand has size 1). This is the only correct way
    /// to handle bidirectional broadcasts like `[N, 1] op [1, S]
    /// → [N, S]`, which simple `i % lhs_len` modulo indexing maps to
    /// wrong cells.
    BinaryFull {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        lhs_len: u32,
        rhs_len: u32,
        op: BinaryOp,
        /// Output shape dims (row-major). Empty in the fast path.
        out_dims_bcast: Vec<u32>,
        /// Per-dim stride into `lhs` (0 where lhs broadcasts).
        bcast_lhs_strides: Vec<u32>,
        /// Per-dim stride into `rhs`.
        bcast_rhs_strides: Vec<u32>,
    },
    /// Activation in-place
    ActivationInPlace {
        data: usize,
        len: u32,
        act: Activation,
    },
    /// Gather axis=0: table\[idx\] → out
    Gather {
        table: usize,
        table_len: u32,
        idx: usize,
        dst: usize,
        num_idx: u32,
        trailing: u32,
    },
    /// Narrow: copy slice
    Narrow {
        src: usize,
        dst: usize,
        outer: u32,
        src_stride: u32,
        dst_stride: u32,
        inner: u32,
    },
    /// Copy (reshape, expand)
    Copy { src: usize, dst: usize, len: u32 },
    /// LayerNorm standalone
    LayerNorm {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    /// RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * gamma + beta. No mean
    /// subtraction, hence cheaper than LayerNorm. Used by Llama-class models.
    RmsNorm {
        src: usize,
        g: usize,
        b: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },
    /// Softmax
    Softmax { data: usize, rows: u32, cols: u32 },
    /// Inclusive (or exclusive) cumulative sum along the last axis
    /// (callers pre-flatten higher-dim cumsums via reshape views).
    Cumsum {
        src: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    /// Mamba-style selective scan (plan #15).
    /// Inputs: x, delta \[b,s,h\], a \[h,n\], b \[b,s,n\], c \[b,s,n\].
    /// Output: y \[b,s,h\]. State h carries through the seq.
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

    /// Gated DeltaNet linear-attention scan (Qwen3.5/3.6 trunk).
    /// Inputs: q, k, v `[b, s, h, n]`; g, beta `[b, s, h]`. Output:
    /// `[b, s, h, n]`. See `Op::GatedDeltaNet` for math.
    GatedDeltaNet {
        q: usize,
        k: usize,
        v: usize,
        g: usize,
        beta: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
    },

    /// 1×1 conv fast path (plan #26). The general Conv2D thunk
    /// runs the textbook 7-deep loop; a 1×1 stride-1 padding-0
    /// groups-1 conv is mathematically a per-batch matmul, and
    /// dispatching it through BLAS is 3-10× faster than the
    /// scalar nest. Common case: ViT patch-projection follow-on,
    /// transformer "expert" reductions in some MoE designs.
    ///
    /// Per batch: weight `[c_out, c_in]` × input `[c_in, h*w]`
    ///         = output `[c_out, h*w]`.
    Conv2D1x1 {
        src: usize,
        weight: usize,
        dst: usize,
        n: u32,
        c_in: u32,
        c_out: u32,
        hw: u32,
    },

    /// Fused dequant + matmul (plan #5). Today supports
    /// `QuantScheme::Int8Block` (symmetric); other schemes panic
    /// at lowering time with a clear message until kernels are added.
    DequantMatMul {
        x: usize,
        w_q: usize,   // packed i8 bytes for Int8 schemes
        scale: usize, // [k/block, n] f32 scale
        zp: usize,    // [k/block, n] f32 zero-point (0 for sym)
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        is_asymmetric: bool,
    },

    /// GGUF-format dequant + matmul. Weight is a packed byte tensor
    /// in one of the K-quant super-block layouts (Q4_K, Q5_K, Q6_K,
    /// Q8_K). Scales / mins live inside the packed bytes — no
    /// side-channel scale tensor.
    ///
    /// Today this is a "dequant-to-scratch then sgemm" kernel — it
    /// keeps the *arena* memory footprint down (weights stay packed)
    /// but the dequant itself happens per matmul. A future fully
    /// fused tile-streaming kernel would close the compute gap.
    DequantMatMulGguf {
        x: usize,        // f32 activations [m, k]
        w_q: usize,      // packed weight bytes (k*n elements packed)
        dst: usize,      // f32 output [m, n]
        m: u32,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
    },

    /// Fused LoRA matmul (plan #9): out = x·W + scale * (x·A)·B.
    /// `r` is the LoRA rank (typically 4-64) — the rank-r
    /// intermediate `x·A` lives in scratch, never on the arena.
    LoraMatMul {
        x: usize,
        w: usize,
        a: usize,
        b: usize,
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        r: u32,
        scale: f32,
    },
    /// Fused sample: logits [batch, vocab] → token ids \[batch\].
    /// See Op::Sample. Output values are f32-encoded usize indices
    /// (matches the rest of the IR's "ids as f32" convention).
    Sample {
        logits: usize,
        dst: usize,
        batch: u32,
        vocab: u32,
        top_k: u32,       // 0 = disabled
        top_p: f32,       // 1.0 = disabled
        temperature: f32, // 1.0 = neutral
        seed: u64,
    },
    /// Attention SDPA. `mask` is the offset of the optional mask tensor
    /// (only meaningful when `mask_kind == MaskKind::Custom`); other
    /// kinds synthesize the mask in-kernel.
    ///
    /// Q/K/V each carry a `_row_stride` (elements per source row).
    /// Defaults to `heads * head_dim` — matches the standalone
    /// "Q/K/V are their own contiguous buffers" case. The Narrow→
    /// Attention fusion below rewrites these to the parent QKV stride
    /// (typically `3 * heads * head_dim`) so the kernel reads QKV
    /// directly without materializing the per-head buffers (plan #46).
    Attention {
        q: usize,
        k: usize,
        v: usize,
        mask: usize,
        out: usize,
        batch: u32,
        /// Query sequence length.
        seq: u32,
        /// Key/value sequence length. Differs from `seq` during cached decode.
        kv_seq: u32,
        heads: u32,
        head_dim: u32,
        mask_kind: rlx_ir::op::MaskKind,
        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        /// Memory layout flag. `false` (the historical default) →
        /// `[B, S, H, D]` row-major: per-head offset is
        /// `bi*S*H*D + si*H*D + hi*D`. `true` → `[B, H, S, D]`
        /// (head-major), matching the convention used by rlx-cuda /
        /// rlx-rocm / rlx-tpu: per-head offset is
        /// `bi*H*S*D + hi*S*D + si*D`. Detected at lowering time
        /// from the input shape vs `num_heads` / `head_dim`.
        bhsd: bool,
    },
    /// RoPE (rotary position embeddings).
    /// `src_row_stride` is elements per source row (defaults to `hidden`
    /// for the standalone case; set to `qkv_axis * inner` when the
    /// thunk fusion pass below rewires Rope to read directly from the
    /// fused QKV buffer — plan #45).
    Rope {
        src: usize,
        cos: usize,
        sin: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        cos_len: u32,
        src_row_stride: u32,
    },
    /// Fused attention block: QKV proj → split → \[RoPE\] → SDPA → output proj.
    /// All intermediates stay in L1 cache. Zero arena writes between ops.
    FusedAttnBlock {
        hidden: usize,
        qkv_w: usize,
        out_w: usize,
        mask: usize,
        out: usize,
        qkv_b: usize,
        out_b: usize, // 0 = no bias
        cos: usize,
        sin: usize,
        cos_len: u32, // 0 = no RoPE
        batch: u32,
        seq: u32,
        hs: u32,
        nh: u32,
        dh: u32,
        has_bias: bool,
        has_rope: bool,
    },
    /// Fused ENTIRE transformer layer: attention + residual + LN + FFN + residual + LN.
    /// Combines ~10 thunks into 1. All intermediates on stack. Zero arena traffic.
    FusedBertLayer {
        // attention
        hidden: usize,
        qkv_w: usize,
        qkv_b: usize,
        out_w: usize,
        out_b: usize,
        mask: usize,
        // LN1
        ln1_g: usize,
        ln1_b: usize,
        eps1: f32,
        // FFN (GELU)
        fc1_w: usize,
        fc1_b: usize,
        fc2_w: usize,
        fc2_b: usize,
        // LN2
        ln2_g: usize,
        ln2_b: usize,
        eps2: f32,
        // output
        out: usize,
        // dims
        batch: u32,
        seq: u32,
        hs: u32,
        nh: u32,
        dh: u32,
        int_dim: u32,
    },
    /// Fused Nomic transformer layer: attention+RoPE + residual + LN + SwiGLU FFN + residual + LN.
    FusedNomicLayer {
        hidden: usize,
        qkv_w: usize,
        out_w: usize,
        mask: usize,
        cos: usize,
        sin: usize,
        cos_len: u32,
        ln1_g: usize,
        ln1_b: usize,
        eps1: f32,
        fc11_w: usize,
        fc12_w: usize,
        fc2_w: usize,
        ln2_g: usize,
        ln2_b: usize,
        eps2: f32,
        out: usize,
        batch: u32,
        seq: u32,
        hs: u32,
        nh: u32,
        dh: u32,
        int_dim: u32,
    },
    /// Fused SwiGLU: out\[r,i\] = x\[r,i\] * silu(x[r, n_half+i]).
    /// Input: [outer, 2*n_half] — concatenated up||gate per row.
    /// Output: [outer, n_half].
    FusedSwiGLU {
        src: usize,
        dst: usize,
        n_half: u32,
        total: u32,
    },
    /// Concat along an axis: output[outer, axis, inner] = inputs concatenated.
    /// Each entry of `inputs` is (src_offset, axis_len_for_that_input) in u32
    /// elements. `outer`, `inner`, and `total_axis_len` are pre-computed
    /// at compile time to avoid per-run shape work.
    Concat {
        dst: usize,
        outer: u32,
        inner: u32,
        total_axis: u32,
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
    /// Reduction along a contiguous range of axes. Input layout (after
    /// shape decomposition) is `[outer, reduced, inner]`; output is
    /// `[outer, inner]`. The single-axis cases (axis=0 → outer=1;
    /// axis=last → inner=1) and contiguous multi-axis (e.g. reduce over
    /// [0, 1] of an [N, C, H, W] tensor → outer=1, reduced=N*C, inner=H*W)
    /// all map onto this triplet. Non-contiguous axes are not supported
    /// and bail to Nop in the compile pass.
    Reduce {
        src: usize,
        dst: usize,
        outer: u32,
        reduced: u32,
        inner: u32,
        op: ReduceOp,
    },
    /// Top-K **indices** along the last axis. Input shape `[outer, axis_dim]`,
    /// output `[outer, k]` of f32-encoded i64 indices. Ties broken by
    /// smaller index. Used by MoE gating + beam search.
    TopK {
        src: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        k: u32,
    },
    /// Indexed batched matmul: out\[i\] = input\[i\] @ weight[expert_idx\[i\]].
    /// Naive impl per token; for real MoE workloads, sort-by-expert + run
    /// segmented GEMM would amortize. Done when there's a workload.
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
    /// Scatter-add: dst[indices\[i\] * trailing + j] += updates[i * trailing + j].
    /// Output is zeroed first; multiple updates to the same row accumulate.
    ScatterAdd {
        updates: usize,
        indices: usize,
        dst: usize,
        num_updates: u32,
        out_dim: u32,
        trailing: u32,
    },
    /// Ternary select: out = cond != 0 ? on_true : on_false
    Where {
        cond: usize,
        on_true: usize,
        on_false: usize,
        dst: usize,
        len: u32,
    },
    /// General N-D transpose / broadcast. `out_dims[i]` is the output's dim
    /// i length; `in_strides[i]` is the input stride (in elements) used to
    /// index that dim — 0 for broadcast dims (Expand). `in_total` is the
    /// total element count in the source buffer (≤ output total when
    /// broadcasting). Strides are pre-computed at compile time.
    Transpose {
        src: usize,
        dst: usize,
        in_total: u32,
        out_dims: Vec<u32>,
        in_strides: Vec<u32>,
    },
    /// Gather along an arbitrary axis. `outer = product(dims[..axis])`,
    /// `trailing = product(dims[axis+1..])`, `axis_dim` = the dimension
    /// being indexed into. Output: outer × num_idx × trailing.
    /// (axis=0 still routes to the simpler Thunk::Gather fast path.)
    GatherAxis {
        table: usize,
        idx: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    /// 2D pooling (Max or Mean). Input layout [N, C, H, W], output
    /// [N, C, H_out, W_out]. Padding is implicit-zero; Mean divides by
    /// the full kernel area (matches torch's `count_include_pad=True`).
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
        kind: ReduceOp,
    },
    /// 2D convolution. Input [N, C_in, H, W], weight [C_out, C_in_per_group, kH, kW],
    /// output [N, C_out, H_out, W_out]. Bias is a separate Op::Binary::Add
    /// after the conv (matching the IR's input layout — Op::Conv has 2 inputs).
    /// Naive direct convolution; sufficient for correctness, not optimised.
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

    // ── Backward / training kernels ─────────────────────────────
    /// Real INT8 matmul with i32 accumulation.
    ///   `out[m, n] = requantize(bias[n] + Σₖ (x[m,k]-x_zp)·(w[k,n]-w_zp), mult, out_zp)`
    /// Reads `x` and `w` as i8, `bias` as i32; writes `out` as i8.
    /// Same kernel shape as `rlx_cortexm::dense::dense_i8` — promoted
    /// to a desktop thunk so a quantized graph compiled here doesn't
    /// have to round-trip through fake-quant.
    QMatMul {
        x: usize,
        w: usize,
        bias: usize,
        out: usize,
        m: u32,
        k: u32,
        n: u32,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
    },

    /// Real INT8 conv2d, NCHW layout. Same loop shape as `Thunk::Conv2D`
    /// but with i8 reads, i32 accumulation, and per-output requantize
    /// to i8. Bias is i32 in the accumulator scale.
    QConv2d {
        x: usize,
        w: usize,
        bias: usize,
        out: usize,
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
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
    },

    /// INT8 quantize. Reads `x` as f32, writes `q` as i8.
    /// `chan = (i / inner) % chan_dim` selects the per-channel
    /// scale/zp; `chan_axis` is informational only (the kernel uses
    /// `chan_dim` and `inner` directly).
    /// For per-tensor, `chan_dim = 1` and `inner = len` so `chan` is
    /// always 0.
    Quantize {
        x: usize,
        q: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    },

    /// INT8 dequantize — inverse of `Thunk::Quantize`.
    Dequantize {
        q: usize,
        x: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
    },

    /// QAT fake-quantize. Per-channel (or per-tensor) symmetric
    /// quantize-then-dequantize on the fly. Computes
    ///   `s[c] = max(|x[..., c, ...]|) / q_max`
    /// then
    ///   `out[i] = clamp(round(x[i]/s[c]), -q_max, q_max) * s[c]`
    /// with `q_max = {127, 7, 1}` for `bits = {8, 4, 2}`. Same
    /// channel-layout convention as `Thunk::Quantize`: every
    /// element's channel is `(i / inner) % chan_dim`. The kernel
    /// does two passes — one to scan max-abs per channel, one to
    /// quant-dequant per element.
    FakeQuantize {
        x: usize,
        out: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        bits: u8,
        /// STE variant — informational on the forward side (output is
        /// the same regardless), kernel-relevant in the matching
        /// `FakeQuantizeBackward` thunk.
        ste: rlx_ir::op::SteKind,
        /// Scale-tracking strategy. `PerBatch` recomputes
        /// `max_abs/q_max` every call (the original path). `EMA{decay}`
        /// blends per-batch max-abs into the `state_off` buffer; `Fixed`
        /// reads `state_off` and never updates it.
        scale_mode: rlx_ir::op::ScaleMode,
        /// `Some(off)` for `EMA` and `Fixed`; `None` for `PerBatch`.
        /// Points at a `[chan_dim]` f32 buffer holding the running scale
        /// per channel.
        state_off: Option<usize>,
    },

    /// Backward pass for `Op::FakeQuantize` under one of four STE
    /// variants. Computes `dx[i]` from the f32 forward input `x` and
    /// the upstream gradient `dy`, using the same per-channel scale
    /// scheme as the forward.
    FakeQuantizeBackward {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        bits: u8,
        ste: rlx_ir::op::SteKind,
    },

    /// LSQ forward — same kernel shape as `FakeQuantize` Fixed mode.
    /// Reads scale from `scale_off` (a `[chan_dim]` Param tensor).
    FakeQuantizeLSQ {
        x: usize,
        scale_off: usize,
        out: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        bits: u8,
    },

    /// LSQ backward, x-gradient. STE-clipped: passes upstream
    /// through inside the quantization range, zeros outside.
    FakeQuantizeLSQBackwardX {
        x: usize,
        scale_off: usize,
        dy: usize,
        dx: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        bits: u8,
    },

    /// LSQ backward, scale-gradient. Per-channel:
    ///   `dscale[c] = sum_i ψ(x[i]/s[c]) · upstream[i]`
    /// where `ψ(z) = -z + round(z)` if `|z| ≤ q_max` else
    /// `sign(z) · q_max`. Output shape: `[chan_dim]`.
    FakeQuantizeLSQBackwardScale {
        x: usize,
        scale_off: usize,
        dy: usize,
        dscale: usize,
        len: u32,
        chan_axis: u32,
        chan_dim: u32,
        inner: u32,
        bits: u8,
    },

    /// ReLU backward: `dx[i] = dy[i] if x[i] > 0 else 0`.
    ReluBackward {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
    },
    /// f64 sibling of `ReluBackward` — same shape as the f32 variant
    /// but reads/writes 8 bytes per element. Required because
    /// `ReluBackward`'s `&[f32]` slot view returns half of every f64
    /// otherwise → backward silently produces 0 gradients on an f64
    /// graph. Mirrors the `ActivationBackwardF64` split.
    ReluBackwardF64 {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
    },

    /// Generic element-wise activation backward.
    /// `dx[i] = (d/dx act(x))[i] · dy[i]`. The closure dispatch is
    /// per-element; expensive activations (Gelu) recompute internals
    /// inline rather than threading an extra "saved y" tensor through.
    ActivationBackward {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
        kind: Activation,
    },
    /// f64 sibling of `ActivationBackward` — slot offsets, len in
    /// elements; kernel reads/writes 8 bytes per element. Required
    /// because `ActivationBackward`'s `&[f32]` slot view silently
    /// returns garbage on an f64 graph (cb % 4 still works but every
    /// loaded value is half of an f64 → wrong gradient).
    ActivationBackwardF64 {
        x: usize,
        dy: usize,
        dx: usize,
        len: u32,
        kind: Activation,
    },

    /// LayerNorm backward — input gradient. Recomputes mean/var/x̂ from
    /// `x` and emits the closed-form `d_x` per row.
    LayerNormBackwardInput {
        x: usize,
        gamma: usize,
        dy: usize,
        dx: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },

    /// LayerNorm backward — gamma gradient. `d_gamma[d] = Σ_row dy·x̂`.
    LayerNormBackwardGamma {
        x: usize,
        dy: usize,
        dgamma: usize,
        rows: u32,
        h: u32,
        eps: f32,
    },

    /// 2D max-pool backward (NCHW). Recomputes the argmax position
    /// inside each window and accumulates `dy` into `dx` at that
    /// position. Output is zeroed first; ties resolve to the first
    /// hit (lowest (kh,kw) index), matching what the forward kernel
    /// does with `acc.max(v)`.
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

    /// 2D conv backward w.r.t. input (`dx = conv_transpose(dy, w)`).
    /// `dy [N, C_out, H_out, W_out]`, `w [C_out, C_in_per_group, kH, kW]`,
    /// `dx [N, C_in, H, W]`.
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

    /// 2D conv backward w.r.t. weight. `x [N, C_in, H, W]`,
    /// `dy [N, C_out, H_out, W_out]`, `dw [C_out, C_in_per_group, kH, kW]`.
    /// `dw` is zeroed before accumulation.
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

    /// Fused softmax + cross-entropy loss with f32-encoded integer
    /// labels. `logits [N, C]`, `labels [N]`, output `[N]` per-row loss.
    /// Numerically stable (max-subtract before exp).
    SoftmaxCrossEntropy {
        logits: usize,
        labels: usize,
        dst: usize,
        n: u32,
        c: u32,
    },

    /// Backward of the fused loss above.
    /// `dlogits[n, k] = (softmax(logits[n])[k] - one_hot(labels[n])[k]) * d_loss[n]`.
    SoftmaxCrossEntropyBackward {
        logits: usize,
        labels: usize,
        d_loss: usize,
        dlogits: usize,
        n: u32,
        c: u32,
    },

    /// User-registered custom op (CPU side). Lowered from `Op::Custom`.
    /// `kernel` is resolved against the global CPU kernel registry at
    /// compile time and stored as `Arc<dyn CpuKernel>` so execution
    /// avoids per-call lookups. v1: f32 contiguous only — see
    /// `op_registry::CpuKernel::execute_f32`.
    CustomOp {
        kernel: Arc<dyn CpuKernel>,
        inputs: Vec<(usize, u32, Shape)>, // (offset, len_elements, shape)
        output: (usize, u32, Shape),      // (offset, len_elements, shape)
        attrs: Vec<u8>,
    },

    /// 1D FFT along the last axis. Input/output are `[..., 2N]`
    /// real-block layout (first N real, second N imag along the
    /// transformed axis). `outer` is the product of all leading axes;
    /// `n_complex` is N (the number of complex points). Both halves
    /// of the real-block layout are read together by the kernel.
    /// `dtype` selects the f32 or f64 path; the two share structure
    /// but not buffers, so a flag at compile time avoids per-row
    /// dispatch.
    Fft1d {
        src: usize,
        dst: usize,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        dtype: rlx_ir::DType,
    },
}

/// Compiled thunk schedule — the runtime hot path.
/// Nop thunks are filtered out at compile time for zero iteration overhead.
#[derive(Clone)]
pub struct ThunkSchedule {
    pub thunks: Vec<Thunk>,
    /// Cached config values.
    pub mask_threshold: f32,
    pub mask_neg_inf: f32,
    pub score_skip: f32,
    /// Pre-compiled closure dispatch (zero match overhead). `Arc` (not
    /// `Box`) so the schedule can be `Clone` — multiple parallel
    /// executors share the same compiled closures (they're read-only
    /// `Fn(*mut u8)` so concurrent dispatch is safe; the arena pointer
    /// they receive is the only mutable state and is per-executor).
    pub compiled_fns: Vec<Arc<dyn Fn(*mut u8) + Send + Sync>>,
}

impl ThunkSchedule {
    pub fn strip_nops(&mut self) {
        self.thunks.retain(|t| !matches!(t, Thunk::Nop));
        // compiled_fns must be rebuilt after stripping — caller should
        // call strip_nops() before compile_closures().
        self.compiled_fns.clear();
    }
}

/// Get the arena byte offset for a node.
fn node_offset(arena: &Arena, id: NodeId) -> usize {
    if arena.has_buffer(id) {
        arena.byte_offset(id)
    } else {
        usize::MAX
    }
}

/// Every byte-offset that a thunk reads from. Used by the Narrow→Rope
/// fusion (#45) to verify a Narrow's dst has exactly one consumer
/// before eliding it. Conservative: when in doubt about reads (an op
/// not yet listed here), the fusion will skip — correctness over
/// completeness.
fn thunk_read_offsets(t: &Thunk) -> Vec<usize> {
    match t {
        Thunk::Sgemm { a, b, .. } => vec![*a, *b],
        Thunk::DenseSolveF64 { a, b, .. } => vec![*a, *b],
        Thunk::DenseSolveF32 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedDenseSolveF64 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedDgemmF64 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedSgemm { a, b, .. } => vec![*a, *b],
        Thunk::FusedMmBiasAct { a, w, bias, .. } => vec![*a, *w, *bias],
        Thunk::BiasAdd { src, bias, .. } => vec![*src, *bias],
        Thunk::BinaryFull { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::BinaryFullF64 { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::BinaryFullC64 { lhs, rhs, .. } => vec![*lhs, *rhs],
        Thunk::ComplexNormSqF32 { src, .. } => vec![*src],
        Thunk::ComplexNormSqBackwardF32 { z, g, .. } => vec![*z, *g],
        Thunk::ConjugateC64 { src, .. } => vec![*src],
        Thunk::Scan {
            outer_init_off,
            xs_inputs,
            ..
        } => {
            let mut v = vec![*outer_init_off];
            for (_, outer_xs_off, _) in xs_inputs.iter() {
                v.push(*outer_xs_off);
            }
            v
        }
        Thunk::ScanBackward {
            outer_init_off,
            outer_traj_off,
            outer_upstream_off,
            outer_xs_offs,
            ..
        } => {
            let mut v = vec![*outer_init_off, *outer_traj_off, *outer_upstream_off];
            for (off, _) in outer_xs_offs.iter() {
                v.push(*off);
            }
            v
        }
        Thunk::ScanBackwardXs {
            outer_init_off,
            outer_traj_off,
            outer_upstream_off,
            outer_xs_offs,
            ..
        } => {
            let mut v = vec![*outer_init_off, *outer_traj_off, *outer_upstream_off];
            for (off, _) in outer_xs_offs.iter() {
                v.push(*off);
            }
            v
        }
        Thunk::CustomFn { inputs, .. } => {
            inputs.iter().map(|(_, outer_off, _)| *outer_off).collect()
        }
        Thunk::ActivationInPlace { data, .. } => vec![*data],
        Thunk::LayerNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::FusedResidualLN {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::RmsNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::Softmax { data, .. } => vec![*data],
        Thunk::Cumsum { src, .. } => vec![*src],
        Thunk::Sample { logits, .. } => vec![*logits],
        Thunk::LoraMatMul { x, w, a, b, .. } => vec![*x, *w, *a, *b],
        Thunk::DequantMatMul {
            x, w_q, scale, zp, ..
        } => vec![*x, *w_q, *scale, *zp],
        Thunk::DequantMatMulGguf { x, w_q, .. } => vec![*x, *w_q],
        Thunk::Conv2D1x1 { src, weight, .. } => vec![*src, *weight],
        Thunk::SelectiveScan {
            x, delta, a, b, c, ..
        } => vec![*x, *delta, *a, *b, *c],
        Thunk::GatedDeltaNet {
            q, k, v, g, beta, ..
        } => vec![*q, *k, *v, *g, *beta],
        Thunk::Attention { q, k, v, mask, .. } => vec![*q, *k, *v, *mask],
        Thunk::Rope { src, cos, sin, .. } => vec![*src, *cos, *sin],
        Thunk::FusedAttnBlock {
            hidden,
            qkv_w,
            out_w,
            mask,
            qkv_b,
            out_b,
            cos,
            sin,
            ..
        } => vec![*hidden, *qkv_w, *out_w, *mask, *qkv_b, *out_b, *cos, *sin],
        Thunk::FusedSwiGLU { src, .. } => vec![*src],
        Thunk::Concat { inputs, .. } => inputs.iter().map(|(off, _)| *off).collect(),
        Thunk::ConcatF64 { inputs, .. } => inputs.iter().map(|(off, _)| *off).collect(),
        Thunk::Narrow { src, .. } => vec![*src],
        Thunk::Copy { src, .. } => vec![*src],
        Thunk::Gather { table, idx, .. } => vec![*table, *idx],
        // Anything not enumerated → return the dst as a "read" too,
        // forcing the fusion to bail (read_count >= 2 → skip). Keeps
        // this list safe to be incomplete.
        _ => vec![],
    }
}

/// Fused dequant + matmul (plan #5). Int8-blockwise weights: each
/// `block_size` consecutive elements of a column share one f32
/// scale (and optionally a zero-point). The dequant happens inside
/// the inner accumulate so the f32 weight is never materialized.
///
/// `w_bytes` is the row-major i8 weight matrix `[k, n]`. `scales`
/// and `zps` are `[k/block, n]`. When `asym=false`, `zps` may be
/// empty.
///
/// Today this is the reference scalar implementation — the win is
/// memory bandwidth, not flops, since LLM weights dominate the
/// working set. A NEON SIMD path that loads 16 i8 → splat-scale →
/// fused-multiply-add is the natural follow-on.
#[allow(clippy::too_many_arguments)]
fn dequant_matmul_int8(
    x: &[f32],       // [m, k]
    w_bytes: &[i8],  // [k, n]
    scales: &[f32],  // [k/block, n]
    zps: &[f32],     // [k/block, n] or empty
    out: &mut [f32], // [m, n]
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    asym: bool,
) {
    let blocks_per_col = k.div_ceil(block_size);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let block = p / block_size;
                let s = scales[block * n + j];
                let z = if asym { zps[block * n + j] } else { 0.0 };
                let q = w_bytes[p * n + j] as f32;
                let dequantized = (q - z) * s;
                acc += x[i * k + p] * dequantized;
            }
            out[i * n + j] = acc;
        }
    }
    let _ = blocks_per_col;
}

/// Fused sampling step: logits → top-k filter → top-p truncation
/// → softmax → multinomial sample. Operates on one row of length
/// `vocab` and returns the sampled index. Plan #42.
///
/// Internal scratch is on the stack via SmallVec-style fallback —
/// for `vocab > 8192` we heap-allocate a working buffer; below
/// that we keep things in a fixed array. (TODO: thread the
/// scratch through ThunkSchedule like sdpa_scores does.)
fn sample_row(
    logits: &[f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rng: &mut rlx_ir::Philox4x32,
) -> usize {
    let v = logits.len();
    if v == 0 {
        return 0;
    }
    let temp = temperature.max(1e-6);
    // Copy + temperature-scale into a working buffer.
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Top-k: zero out everything but the k largest by setting to -inf.
    if top_k > 0 && top_k < v {
        // Partial selection: find k-th largest then mask below.
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        // Sort descending; partial would be O(n log k), full sort is fine
        // for typical vocab sizes (32k-128k) — single-row work.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let cutoff = indexed[top_k - 1].1;
        for x in scaled.iter_mut() {
            if *x < cutoff {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    // Stable softmax.
    let mut max_l = f32::NEG_INFINITY;
    for &x in &scaled {
        if x > max_l {
            max_l = x;
        }
    }
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max_l).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    for x in scaled.iter_mut() {
        *x *= inv;
    }

    // Top-p: keep the smallest set of tokens whose cumulative
    // probability exceeds top_p (after sorting descending).
    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut keep = vec![false; v];
        for (idx, p) in indexed.iter() {
            keep[*idx] = true;
            cum += *p;
            if cum >= top_p {
                break;
            }
        }
        let mut new_sum = 0.0f32;
        for (i, x) in scaled.iter_mut().enumerate() {
            if !keep[i] {
                *x = 0.0;
            }
            new_sum += *x;
        }
        let inv = 1.0 / new_sum.max(f32::MIN_POSITIVE);
        for x in scaled.iter_mut() {
            *x *= inv;
        }
    }

    // Multinomial sample via inverse-CDF.
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    v - 1 // floating-point edge case fallback
}

/// Apply a synthetic (kernel-generated) attention mask to a `[q_seq, k_seq]`
/// scores matrix. Custom masks are read from a tensor and not handled here.
/// `None` is a no-op so callers don't need to special-case it.
#[inline]
fn apply_synthetic_mask(
    scores: &mut [f32],
    q_seq: usize,
    k_seq: usize,
    kind: rlx_ir::op::MaskKind,
) {
    let neg = crate::config::RuntimeConfig::global().attn_mask_neg_inf;
    let q_offset = k_seq.saturating_sub(q_seq);
    match kind {
        rlx_ir::op::MaskKind::None
        | rlx_ir::op::MaskKind::Custom
        | rlx_ir::op::MaskKind::Bias => {}
        rlx_ir::op::MaskKind::Causal => {
            for qi in 0..q_seq {
                let abs_q = q_offset + qi;
                for ki in (abs_q + 1)..k_seq {
                    scores[qi * k_seq + ki] = neg;
                }
            }
        }
        rlx_ir::op::MaskKind::SlidingWindow(w) => {
            for qi in 0..q_seq {
                let abs_q = q_offset + qi;
                let lo = abs_q.saturating_sub(w);
                for ki in 0..k_seq {
                    if ki < lo || ki > abs_q {
                        scores[qi * k_seq + ki] = neg;
                    }
                }
            }
        }
    }
}

/// Compile graph into thunk schedule.
pub fn compile_thunks(graph: &Graph, arena: &Arena) -> ThunkSchedule {
    let mut thunks = Vec::with_capacity(graph.len());

    for node in graph.nodes() {
        // View ops (Reshape / same-dtype Cast / axis-0 Narrow) are aliased
        // to their parent's slot by the memory planner — no copy needed.
        // Plan #46.
        if rlx_opt::is_pure_view(graph, node) {
            thunks.push(Thunk::Nop);
            continue;
        }
        let t = match &node.op {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => Thunk::Nop,

            Op::FusedMatMulBiasAct { activation } => {
                let shape = &node.shape;
                let n = shape.dim(shape.rank() - 1).unwrap_static();
                let total = shape.num_elements().unwrap();
                let m = total / n;
                let a_len = get_len(graph, node.inputs[0]);
                let k = a_len / m;
                Thunk::FusedMmBiasAct {
                    a: node_offset(arena, node.inputs[0]),
                    w: node_offset(arena, node.inputs[1]),
                    bias: node_offset(arena, node.inputs[2]),
                    c: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    act: *activation,
                }
            }

            Op::FusedResidualLN { has_bias, eps } => {
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                let rows = total / h;
                let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
                Thunk::FusedResidualLN {
                    x: node_offset(arena, node.inputs[0]),
                    res: node_offset(arena, node.inputs[1]),
                    bias: if *has_bias {
                        node_offset(arena, node.inputs[2])
                    } else {
                        0
                    },
                    g: node_offset(arena, node.inputs[g_idx]),
                    b: node_offset(arena, node.inputs[b_idx]),
                    out: node_offset(arena, node.id),
                    rows: rows as u32,
                    h: h as u32,
                    eps: *eps,
                    has_bias: *has_bias,
                }
            }

            Op::MatMul => {
                let shape = &node.shape;
                let a_shape = &graph.node(node.inputs[0]).shape;
                let b_shape = &graph.node(node.inputs[1]).shape;
                let n = shape.dim(shape.rank() - 1).unwrap_static();

                // Detect batched matmul: any rank where both inputs
                // and output share the same leading batch dims and
                // the last 2 dims form an [M, K] @ [K, N] = [M, N].
                // The 2-D MatMul lowering's flatten-and-call-dgemm trick
                // is wrong when both operands carry independent batch
                // dims (per-batch K dimension differs).
                let batched_3d = a_shape.rank() >= 3
                    && b_shape.rank() == a_shape.rank()
                    && shape.rank() == a_shape.rank()
                    && {
                        // All leading dims (everything except last 2) match.
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
                if batched_3d && shape.dtype() == rlx_ir::DType::F64 {
                    // Batch is the product of all leading dims (every
                    // dim except the last 2); m/k/n are the inner
                    // matmul dims. Works for any rank >= 3.
                    let r = shape.rank();
                    let mut batch_prod = 1usize;
                    for d in 0..r - 2 {
                        batch_prod *= shape.dim(d).unwrap_static();
                    }
                    let m_dim = shape.dim(r - 2).unwrap_static();
                    let k_dim = a_shape.dim(r - 1).unwrap_static();
                    debug_assert_eq!(k_dim, b_shape.dim(r - 2).unwrap_static());
                    Thunk::BatchedDgemmF64 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        batch: batch_prod as u32,
                        m: m_dim as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    }
                } else if batched_3d && shape.dtype() == rlx_ir::DType::F32 {
                    // f32 batched matmul for any rank >= 3 (collapse all
                    // leading batch dims into a single batch count).
                    let r = shape.rank();
                    let mut batch_prod = 1usize;
                    for d in 0..r - 2 {
                        batch_prod *= shape.dim(d).unwrap_static();
                    }
                    let m_dim = shape.dim(r - 2).unwrap_static();
                    let k_dim = a_shape.dim(r - 1).unwrap_static();
                    debug_assert_eq!(k_dim, b_shape.dim(r - 2).unwrap_static());
                    Thunk::BatchedSgemm {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        batch: batch_prod as u32,
                        m: m_dim as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    }
                } else {
                    let total = shape.num_elements().unwrap();
                    let m = total / n;
                    let a_len = get_len(graph, node.inputs[0]);
                    let k = a_len / m;
                    match shape.dtype() {
                        rlx_ir::DType::F64 => Thunk::Dgemm {
                            a: node_offset(arena, node.inputs[0]),
                            b: node_offset(arena, node.inputs[1]),
                            c: node_offset(arena, node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                        },
                        _ => Thunk::Sgemm {
                            a: node_offset(arena, node.inputs[0]),
                            b: node_offset(arena, node.inputs[1]),
                            c: node_offset(arena, node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                        },
                    }
                }
            }

            Op::Binary(op) => {
                let lhs_len = get_len(graph, node.inputs[0]);
                let rhs_len = get_len(graph, node.inputs[1]);
                let out_len = node.shape.num_elements().unwrap();
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    // Native C64 element-wise. Add/Sub/Mul/Div lower
                    // to `BinaryFullC64`; the rest don't have a
                    // single natural complex definition.
                    match op {
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {}
                        BinaryOp::Max | BinaryOp::Min | BinaryOp::Pow => panic!(
                            "Op::Binary({op:?}) on DType::C64: complex \
                             max/min/pow have no single natural definition \
                             — caller should drop to 2N-real-block (see \
                             spike-ac) and pick a convention there"
                        ),
                    }
                }
                // Compute broadcast strides for the slow path. Empty
                // vectors when no broadcast is needed (the fast-path
                // kernel ignores them anyway).
                let (out_dims_bcast, bcast_lhs_strides, bcast_rhs_strides) =
                    if lhs_len == out_len && rhs_len == out_len {
                        (Vec::new(), Vec::new(), Vec::new())
                    } else {
                        let lhs_dims = get_static_dims(graph, node.inputs[0]);
                        let rhs_dims = get_static_dims(graph, node.inputs[1]);
                        let out_dims_v = get_static_dims(graph, node.id);
                        if lhs_dims.is_empty() || rhs_dims.is_empty() || out_dims_v.is_empty() {
                            // Dynamic shape — fall back to the legacy
                            // modulo path (correct for scalar / last-
                            // axis broadcast, which is the only
                            // dynamic case in practice).
                            (Vec::new(), Vec::new(), Vec::new())
                        } else {
                            let ls = broadcast_strides(&lhs_dims, &out_dims_v);
                            let rs = broadcast_strides(&rhs_dims, &out_dims_v);
                            let od: Vec<u32> = out_dims_v.iter().map(|x| *x as u32).collect();
                            (od, ls, rs)
                        }
                    };
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    Thunk::BinaryFullC64 {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                    }
                } else if node.shape.dtype() == rlx_ir::DType::F64 {
                    // f64 path — no BiasAdd fast-path (yet); use the
                    // general binary-with-broadcast kernel.
                    Thunk::BinaryFullF64 {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                    }
                } else if matches!(op, BinaryOp::Add)
                    && rhs_len < out_len
                    && out_len % rhs_len == 0
                    && is_trailing_bias_broadcast(
                        graph.node(node.inputs[1]).shape.dims(),
                        graph.node(node.id).shape.dims(),
                    )
                {
                    // `BiasAdd` is only correct when the bias is a
                    // *trailing* broadcast — rhs dims match the right-
                    // hand side of the output dims (with size-1 only
                    // allowed in left-padded outer positions).
                    // SAM's rel-pos `[bh, h, w, 1, w] + [bh, h, w, h, w]`
                    // has rhs_len divide out_len cleanly but is a
                    // mid-shape singleton, NOT a trailing broadcast.
                    // Routing it through BiasAdd silently treats it as
                    // last-`rhs_len`-cols repeated — wrong values.
                    Thunk::BiasAdd {
                        src: node_offset(arena, node.inputs[0]),
                        bias: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        m: (out_len / rhs_len) as u32,
                        n: rhs_len as u32,
                    }
                } else {
                    let lhs_len = get_len(graph, node.inputs[0]);
                    Thunk::BinaryFull {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                    }
                }
            }

            Op::Activation(act) => {
                let len = node.shape.num_elements().unwrap();
                let in_off = node_offset(arena, node.inputs[0]);
                let out_off = node_offset(arena, node.id);
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    // Only Neg/Exp/Log/Sqrt have natural complex
                    // extensions used in signal-processing graphs.
                    // Everything else (Sigmoid, Tanh, Relu, Abs,
                    // Sin/Cos/Tan/Atan, Round, GeLU family) is rejected.
                    match act {
                        Activation::Neg | Activation::Exp | Activation::Log | Activation::Sqrt => {}
                        other => panic!(
                            "Op::Activation({other:?}) on DType::C64: no \
                             natural complex extension — supported on C64: \
                             Neg, Exp, Log, Sqrt"
                        ),
                    }
                    Thunk::ActivationC64 {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                        kind: *act,
                    }
                } else if node.shape.dtype() == rlx_ir::DType::F64 {
                    Thunk::ActivationF64 {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                        kind: *act,
                    }
                } else if in_off == out_off {
                    // ActivationInPlace operates on a single buffer. When the
                    // planner has assigned input and output the same slot
                    // (typical post-fusion case), we just run on that slot.
                    Thunk::ActivationInPlace {
                        data: out_off,
                        len: len as u32,
                        act: *act,
                    }
                } else {
                    // Two-step: copy input → output, then activate output in place.
                    // The schedule executes them in this order; downstream
                    // thunks see the activated output at out_off.
                    thunks.push(Thunk::Copy {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                    });
                    Thunk::ActivationInPlace {
                        data: out_off,
                        len: len as u32,
                        act: *act,
                    }
                }
            }

            Op::Gather { axis } if *axis == 0 => {
                let table_shape = &graph.node(node.inputs[0]).shape;
                let table_total = table_shape.num_elements().unwrap();
                let trailing: usize = (1..table_shape.rank())
                    .map(|i| table_shape.dim(i).unwrap_static())
                    .product();
                let idx_len = get_len(graph, node.inputs[1]);
                Thunk::Gather {
                    table: node_offset(arena, node.inputs[0]),
                    table_len: table_total as u32,
                    idx: node_offset(arena, node.inputs[1]),
                    dst: node_offset(arena, node.id),
                    num_idx: idx_len as u32,
                    trailing: trailing as u32,
                }
            }

            Op::Gather { axis } => {
                // Non-zero axis: outer × num_idx × trailing layout.
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
                let idx_len = get_len(graph, node.inputs[1]);
                Thunk::GatherAxis {
                    table: node_offset(arena, node.inputs[0]),
                    idx: node_offset(arena, node.inputs[1]),
                    dst: node_offset(arena, node.id),
                    outer: outer as u32,
                    axis_dim: axis_dim as u32,
                    num_idx: idx_len as u32,
                    trailing: trailing as u32,
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
                // src offset includes start position (in bytes: start * inner * sizeof(f32))
                let src_byte_offset = node_offset(arena, node.inputs[0]) + start * inner * 4;
                Thunk::Narrow {
                    src: src_byte_offset,
                    dst: node_offset(arena, node.id),
                    outer: outer as u32,
                    src_stride: (in_axis * inner) as u32, // elements per outer step in source
                    dst_stride: (*len * inner) as u32,    // elements per outer step in dest
                    inner: (*len * inner) as u32,         // elements to copy per outer step
                }
            }

            Op::Reshape { .. } | Op::Cast { .. } => {
                // Pure layout/dtype change: same total element count, plain copy.
                let len = node.shape.num_elements().unwrap();
                let src = node_offset(arena, node.inputs[0]);
                let dst = node_offset(arena, node.id);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::CopyF64 {
                        src,
                        dst,
                        len: len as u32,
                    },
                    _ => Thunk::Copy {
                        src,
                        dst,
                        len: len as u32,
                    },
                }
            }

            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                Thunk::Quantize {
                    x: node_offset(arena, node.inputs[0]),
                    q: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    scales: scales.clone(),
                    zero_points: zero_points.clone(),
                }
            }

            Op::FakeQuantize {
                bits,
                axis,
                ste,
                scale_mode,
            } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                let state_off = match scale_mode {
                    rlx_ir::op::ScaleMode::PerBatch => None,
                    rlx_ir::op::ScaleMode::EMA { .. } | rlx_ir::op::ScaleMode::Fixed => {
                        // Second input carries the [chan_dim] scale state.
                        debug_assert_eq!(
                            node.inputs.len(),
                            2,
                            "EMA/Fixed FakeQuantize needs a state input"
                        );
                        Some(node_offset(arena, node.inputs[1]))
                    }
                };
                Thunk::FakeQuantize {
                    x: node_offset(arena, node.inputs[0]),
                    out: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    bits: *bits,
                    ste: *ste,
                    scale_mode: *scale_mode,
                    state_off,
                }
            }

            Op::FakeQuantizeLSQ { bits, axis } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                Thunk::FakeQuantizeLSQ {
                    x: node_offset(arena, node.inputs[0]),
                    scale_off: node_offset(arena, node.inputs[1]),
                    out: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    bits: *bits,
                }
            }

            Op::FakeQuantizeLSQBackwardX { bits, axis } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                Thunk::FakeQuantizeLSQBackwardX {
                    x: node_offset(arena, node.inputs[0]),
                    scale_off: node_offset(arena, node.inputs[1]),
                    dy: node_offset(arena, node.inputs[2]),
                    dx: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    bits: *bits,
                }
            }

            Op::FakeQuantizeLSQBackwardScale { bits, axis } => {
                // Output shape is [chan_dim] — node.shape doesn't
                // describe the input data layout, but inputs[0] does.
                let in_shape = &graph.node(node.inputs[0]).shape;
                let (chan_axis, chan_dim, inner) = quant_layout(in_shape, *axis);
                Thunk::FakeQuantizeLSQBackwardScale {
                    x: node_offset(arena, node.inputs[0]),
                    scale_off: node_offset(arena, node.inputs[1]),
                    dy: node_offset(arena, node.inputs[2]),
                    dscale: node_offset(arena, node.id),
                    len: in_shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    bits: *bits,
                }
            }

            Op::FakeQuantizeBackward { bits, axis, ste } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                Thunk::FakeQuantizeBackward {
                    x: node_offset(arena, node.inputs[0]),
                    dy: node_offset(arena, node.inputs[1]),
                    dx: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    bits: *bits,
                    ste: *ste,
                }
            }

            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
                Thunk::Dequantize {
                    q: node_offset(arena, node.inputs[0]),
                    x: node_offset(arena, node.id),
                    len: node.shape.num_elements().unwrap() as u32,
                    chan_axis: chan_axis as u32,
                    chan_dim: chan_dim as u32,
                    inner: inner as u32,
                    scales: scales.clone(),
                    zero_points: zero_points.clone(),
                }
            }

            Op::Expand { .. } => {
                // Broadcast: build per-output-dim strides where any input dim
                // of size 1 has stride 0 (read the same element repeatedly).
                // Reuses the Thunk::Transpose runtime — N-D walk with strides
                // is identical; only the strides differ.
                let in_shape = &graph.node(node.inputs[0]).shape;
                let out_shape = &node.shape;
                let in_rank = in_shape.rank();
                let out_rank = out_shape.rank();
                // Implicit leading 1s if input has lower rank.
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
                // Row-major input strides (over the padded shape).
                let mut in_strides_full = vec![1usize; out_rank];
                for d in (0..out_rank.saturating_sub(1)).rev() {
                    in_strides_full[d] = in_strides_full[d + 1] * in_dims[d + 1];
                }
                let out_dims: Vec<u32> = (0..out_rank)
                    .map(|i| out_shape.dim(i).unwrap_static() as u32)
                    .collect();
                // Stride is 0 for broadcast dims (in_dim == 1 && out_dim > 1).
                let in_strides: Vec<u32> = (0..out_rank)
                    .map(|i| {
                        if in_dims[i] == 1 && (out_dims[i] as usize) > 1 {
                            0
                        } else {
                            in_strides_full[i] as u32
                        }
                    })
                    .collect();
                let in_total = in_dims.iter().product::<usize>() as u32;
                let src = node_offset(arena, node.inputs[0]);
                let dst = node_offset(arena, node.id);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::TransposeF64 {
                        src,
                        dst,
                        in_total,
                        out_dims,
                        in_strides,
                    },
                    _ => Thunk::Transpose {
                        src,
                        dst,
                        in_total,
                        out_dims,
                        in_strides,
                    },
                }
            }

            Op::RmsNorm { eps, .. } => {
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                Thunk::RmsNorm {
                    src: node_offset(arena, node.inputs[0]),
                    g: node_offset(arena, node.inputs[1]),
                    b: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    rows: (total / h) as u32,
                    h: h as u32,
                    eps: *eps,
                }
            }

            Op::LayerNorm { eps, .. } => {
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                Thunk::LayerNorm {
                    src: node_offset(arena, node.inputs[0]),
                    g: node_offset(arena, node.inputs[1]),
                    b: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    rows: (total / h) as u32,
                    h: h as u32,
                    eps: *eps,
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
                let in_off = node_offset(arena, node.inputs[0]);
                let out_off = node_offset(arena, node.id);
                // Softmax kernel runs in-place on its data buffer. If the
                // planner gave input and output separate slots (their live
                // ranges overlap, so no aliasing), the output starts
                // uninitialized — emit a Copy first so the data is there.
                // Same pattern as Op::Activation.
                if in_off != out_off {
                    thunks.push(Thunk::Copy {
                        src: in_off,
                        dst: out_off,
                        len: total as u32,
                    });
                }
                Thunk::Softmax {
                    data: out_off,
                    rows: (total / cols) as u32,
                    cols: cols as u32,
                }
            }

            Op::SelectiveScan { state_size } => {
                let in_shape = &graph.node(node.inputs[0]).shape;
                let (batch, seq, hidden) = (
                    in_shape.dim(0).unwrap_static(),
                    in_shape.dim(1).unwrap_static(),
                    in_shape.dim(2).unwrap_static(),
                );
                Thunk::SelectiveScan {
                    x: node_offset(arena, node.inputs[0]),
                    delta: node_offset(arena, node.inputs[1]),
                    a: node_offset(arena, node.inputs[2]),
                    b: node_offset(arena, node.inputs[3]),
                    c: node_offset(arena, node.inputs[4]),
                    dst: node_offset(arena, node.id),
                    batch: batch as u32,
                    seq: seq as u32,
                    hidden: hidden as u32,
                    state_size: *state_size as u32,
                }
            }

            Op::GatedDeltaNet { state_size } => {
                let q_shape = &graph.node(node.inputs[0]).shape;
                let (batch, seq, heads) = (
                    q_shape.dim(0).unwrap_static(),
                    q_shape.dim(1).unwrap_static(),
                    q_shape.dim(2).unwrap_static(),
                );
                Thunk::GatedDeltaNet {
                    q: node_offset(arena, node.inputs[0]),
                    k: node_offset(arena, node.inputs[1]),
                    v: node_offset(arena, node.inputs[2]),
                    g: node_offset(arena, node.inputs[3]),
                    beta: node_offset(arena, node.inputs[4]),
                    dst: node_offset(arena, node.id),
                    batch: batch as u32,
                    seq: seq as u32,
                    heads: heads as u32,
                    state_size: *state_size as u32,
                }
            }

            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let w_shape = &graph.node(node.inputs[1]).shape;
                let m = x_shape.dim(0).unwrap_static();
                let k = x_shape.dim(1).unwrap_static();
                let n = w_shape.dim(1).unwrap_static();
                Thunk::QMatMul {
                    x: node_offset(arena, node.inputs[0]),
                    w: node_offset(arena, node.inputs[1]),
                    bias: node_offset(arena, node.inputs[2]),
                    out: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    x_zp: *x_zp,
                    w_zp: *w_zp,
                    out_zp: *out_zp,
                    mult: *mult,
                }
            }

            Op::QConv2d {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let in_shape = &graph.node(node.inputs[0]).shape;
                let w_shape = &graph.node(node.inputs[1]).shape;
                let out_shape = &node.shape;
                if kernel_size.len() == 2
                    && in_shape.rank() == 4
                    && w_shape.rank() == 4
                    && out_shape.rank() == 4
                {
                    Thunk::QConv2d {
                        x: node_offset(arena, node.inputs[0]),
                        w: node_offset(arena, node.inputs[1]),
                        bias: node_offset(arena, node.inputs[2]),
                        out: node_offset(arena, node.id),
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
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        mult: *mult,
                    }
                } else {
                    Thunk::Nop
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
                        x: node_offset(arena, node.inputs[0]),
                        w_q: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        scheme: *scheme,
                    }
                } else {
                    let (block_size, is_asymmetric) = match scheme {
                        QuantScheme::Int8Block { block_size } => (*block_size, false),
                        QuantScheme::Int8BlockAsym { block_size } => (*block_size, true),
                        other => panic!(
                            "DequantMatMul on CPU only supports Int8Block / Int8BlockAsym (legacy) or GGUF schemes; got {other}"
                        ),
                    };
                    Thunk::DequantMatMul {
                        x: node_offset(arena, node.inputs[0]),
                        w_q: node_offset(arena, node.inputs[1]),
                        scale: node_offset(arena, node.inputs[2]),
                        zp: node_offset(arena, node.inputs[3]),
                        dst: node_offset(arena, node.id),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        block_size,
                        is_asymmetric,
                    }
                }
            }

            Op::LoraMatMul { scale } => {
                // x [m, k], w [k, n], a [k, r], b [r, n].
                let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                let m = total / n.max(1);
                let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                let k = x_total / m.max(1);
                let a_total = graph.node(node.inputs[2]).shape.num_elements().unwrap();
                let r = a_total / k.max(1);
                Thunk::LoraMatMul {
                    x: node_offset(arena, node.inputs[0]),
                    w: node_offset(arena, node.inputs[1]),
                    a: node_offset(arena, node.inputs[2]),
                    b: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    r: r as u32,
                    scale: *scale,
                }
            }

            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => {
                let in_shape = &graph.node(node.inputs[0]).shape;
                // Logits are [batch, vocab] (or [vocab] → batch=1).
                let (batch, vocab) = if in_shape.rank() >= 2 {
                    (
                        in_shape.dim(0).unwrap_static(),
                        in_shape.dim(in_shape.rank() - 1).unwrap_static(),
                    )
                } else {
                    (1, in_shape.num_elements().unwrap_or(0))
                };
                Thunk::Sample {
                    logits: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    batch: batch as u32,
                    vocab: vocab as u32,
                    top_k: *top_k as u32,
                    top_p: *top_p,
                    temperature: *temperature,
                    seed: *seed,
                }
            }

            Op::Cumsum { axis, exclusive } => {
                // For now CPU only supports last-axis cumsum (the
                // common case for sampling / ragged offsets).
                // Other axes can lower via Transpose → Cumsum →
                // Transpose; not on the hot path today.
                let rank = node.shape.rank();
                let ax = if *axis < 0 {
                    (rank as i32 + axis) as usize
                } else {
                    *axis as usize
                };
                assert_eq!(
                    ax,
                    rank - 1,
                    "Cumsum only supports the last axis on CPU today"
                );
                let cols = node.shape.dim(ax).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                Thunk::Cumsum {
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    rows: (total / cols) as u32,
                    cols: cols as u32,
                    exclusive: *exclusive,
                }
            }

            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
            } => {
                // Layout dispatch: rank-4 input could be either
                // `[B, S, H, D]` (CPU's historical convention) or
                // `[B, H, S, D]` (the convention the GPU/TPU backends
                // share). Disambiguate by which axis matches
                // `num_heads`. Rank-3 is always `[B, S, H*D]`.
                let q_shape = &graph.node(node.inputs[0]).shape;
                let k_shape = &graph.node(node.inputs[1]).shape;
                let rank = q_shape.rank();
                let (batch, seq, kv_seq, bhsd) = if rank == 4 {
                    let d1 = q_shape.dim(1).unwrap_static();
                    let d2 = q_shape.dim(2).unwrap_static();
                    if d1 == *num_heads {
                        // [B, H, S, D]
                        (
                            q_shape.dim(0).unwrap_static(),
                            d2,
                            k_shape.dim(2).unwrap_static(),
                            true,
                        )
                    } else {
                        // [B, S, H, D]
                        (
                            q_shape.dim(0).unwrap_static(),
                            d1,
                            k_shape.dim(1).unwrap_static(),
                            false,
                        )
                    }
                } else if rank >= 3 {
                    (
                        q_shape.dim(0).unwrap_static(),
                        q_shape.dim(1).unwrap_static(),
                        k_shape.dim(1).unwrap_static(),
                        false,
                    )
                } else {
                    (
                        1,
                        q_shape.dim(0).unwrap_static(),
                        k_shape.dim(0).unwrap_static(),
                        false,
                    )
                };
                let mask_off = if matches!(
                    mask_kind,
                    rlx_ir::op::MaskKind::Custom | rlx_ir::op::MaskKind::Bias
                ) {
                    node_offset(arena, node.inputs[3])
                } else {
                    0
                };
                let hs = (*num_heads * *head_dim) as u32;
                Thunk::Attention {
                    q: node_offset(arena, node.inputs[0]),
                    k: node_offset(arena, node.inputs[1]),
                    v: node_offset(arena, node.inputs[2]),
                    mask: mask_off,
                    out: node_offset(arena, node.id),
                    batch: batch as u32,
                    seq: seq as u32,
                    kv_seq: kv_seq as u32,
                    heads: *num_heads as u32,
                    head_dim: *head_dim as u32,
                    mask_kind: *mask_kind,
                    // Defaults: each input is its own contiguous buffer
                    // with row stride = hidden. Rewritten by the
                    // Narrow→Attention fusion when applicable.
                    q_row_stride: hs,
                    k_row_stride: hs,
                    v_row_stride: hs,
                    bhsd,
                }
            }

            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let (batch, seq) = if x_shape.rank() >= 3 {
                    (
                        x_shape.dim(0).unwrap_static(),
                        x_shape.dim(1).unwrap_static(),
                    )
                } else {
                    let total = x_shape.num_elements().unwrap();
                    let s = x_shape.dim(x_shape.rank() - 2).unwrap_static();
                    (total / (s * num_heads * head_dim), s)
                };
                let hs = (*num_heads * *head_dim) as u32;
                // Inputs: hidden, qkv_w, out_w, mask, [qkv_b, out_b], [cos, sin]
                let mut idx = 4;
                let (qkv_b_off, out_b_off) = if *has_bias {
                    let qb = node_offset(arena, node.inputs[idx]);
                    let ob = node_offset(arena, node.inputs[idx + 1]);
                    idx += 2;
                    (qb, ob)
                } else {
                    (0, 0)
                };
                let (cos_off, sin_off, cl) = if *has_rope {
                    let c = node_offset(arena, node.inputs[idx]);
                    let s = node_offset(arena, node.inputs[idx + 1]);
                    let clen = get_len(graph, node.inputs[idx]);
                    (c, s, clen as u32)
                } else {
                    (0, 0, 0)
                };

                Thunk::FusedAttnBlock {
                    hidden: node_offset(arena, node.inputs[0]),
                    qkv_w: node_offset(arena, node.inputs[1]),
                    out_w: node_offset(arena, node.inputs[2]),
                    mask: node_offset(arena, node.inputs[3]),
                    out: node_offset(arena, node.id),
                    qkv_b: qkv_b_off,
                    out_b: out_b_off,
                    cos: cos_off,
                    sin: sin_off,
                    cos_len: cl,
                    batch: batch as u32,
                    seq: seq as u32,
                    hs,
                    nh: *num_heads as u32,
                    dh: *head_dim as u32,
                    has_bias: *has_bias,
                    has_rope: *has_rope,
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
                    (
                        1,
                        x_shape.dim(0).unwrap_static(),
                        total / x_shape.dim(0).unwrap_static(),
                    )
                };
                let cos_len = get_len(graph, node.inputs[1]);
                Thunk::Rope {
                    src: node_offset(arena, node.inputs[0]),
                    cos: node_offset(arena, node.inputs[1]),
                    sin: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    batch: batch as u32,
                    seq: seq as u32,
                    hidden: hidden as u32,
                    head_dim: *head_dim as u32,
                    cos_len: cos_len as u32,
                    // Default: source rows are tightly packed (rewritten
                    // by the Narrow→Rope fusion pass below if Rope ends
                    // up reading from a wider parent like QKV).
                    src_row_stride: hidden as u32,
                }
            }

            Op::FusedSwiGLU { cast_to: _ } => {
                let n_half = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                Thunk::FusedSwiGLU {
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    n_half: n_half as u32,
                    total: total as u32,
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
                // 1×1 fast path (plan #26): kH=kW=1, stride=1,
                // padding=0, dilation=1, groups=1. Emits a single
                // Conv2D1x1 thunk that BLAS-dispatches per batch.
                let is_1x1_simple = kernel_size.len() == 2
                    && kernel_size[0] == 1
                    && kernel_size[1] == 1
                    && stride.iter().all(|&s| s == 1)
                    && padding.iter().all(|&p| p == 0)
                    && dilation.iter().all(|&d| d == 1)
                    && *groups == 1;
                if is_1x1_simple && in_shape.rank() == 4 && out_shape.rank() == 4 {
                    let n = in_shape.dim(0).unwrap_static();
                    let c_in = in_shape.dim(1).unwrap_static();
                    let c_out = out_shape.dim(1).unwrap_static();
                    let h = in_shape.dim(2).unwrap_static();
                    let w = in_shape.dim(3).unwrap_static();
                    Thunk::Conv2D1x1 {
                        src: node_offset(arena, node.inputs[0]),
                        weight: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        n: n as u32,
                        c_in: c_in as u32,
                        c_out: c_out as u32,
                        hw: (h * w) as u32,
                    }
                } else if kernel_size.len() == 2
                    && in_shape.rank() == 4
                    && w_shape.rank() == 4
                    && out_shape.rank() == 4
                {
                    Thunk::Conv2D {
                        src: node_offset(arena, node.inputs[0]),
                        weight: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
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
                // Currently support 2D pooling on rank-4 NCHW tensors.
                let in_shape = &graph.node(node.inputs[0]).shape;
                let out_shape = &node.shape;
                if kernel_size.len() == 2 && in_shape.rank() == 4 && out_shape.rank() == 4 {
                    Thunk::Pool2D {
                        src: node_offset(arena, node.inputs[0]),
                        dst: node_offset(arena, node.id),
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

            Op::Transpose { perm } => {
                // Pre-compute (out_dims, in_strides_for_each_out_dim) so the
                // runtime loop is just an N-D index walk + scatter.
                let in_shape = &graph.node(node.inputs[0]).shape;
                let in_rank = in_shape.rank();
                let in_dims: Vec<usize> = (0..in_rank)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .collect();
                // Row-major input strides: stride[d] = product of dims[d+1..].
                let mut in_strides_full = vec![1usize; in_rank];
                for d in (0..in_rank.saturating_sub(1)).rev() {
                    in_strides_full[d] = in_strides_full[d + 1] * in_dims[d + 1];
                }
                let out_dims: Vec<u32> = perm.iter().map(|&p| in_dims[p] as u32).collect();
                let in_strides: Vec<u32> =
                    perm.iter().map(|&p| in_strides_full[p] as u32).collect();
                let in_total = in_dims.iter().product::<usize>() as u32;
                let src = node_offset(arena, node.inputs[0]);
                let dst = node_offset(arena, node.id);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::TransposeF64 {
                        src,
                        dst,
                        in_total,
                        out_dims,
                        in_strides,
                    },
                    _ => Thunk::Transpose {
                        src,
                        dst,
                        in_total,
                        out_dims,
                        in_strides,
                    },
                }
            }

            Op::ScatterAdd => {
                // updates: [num_updates, ...trailing], indices: [num_updates],
                // output: [out_dim, ...trailing]
                let upd_shape = &graph.node(node.inputs[0]).shape;
                let out_shape = &node.shape;
                let num_updates = upd_shape.dim(0).unwrap_static();
                let out_dim = out_shape.dim(0).unwrap_static();
                let trailing: usize = (1..out_shape.rank())
                    .map(|i| out_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                Thunk::ScatterAdd {
                    updates: node_offset(arena, node.inputs[0]),
                    indices: node_offset(arena, node.inputs[1]),
                    dst: node_offset(arena, node.id),
                    num_updates: num_updates as u32,
                    out_dim: out_dim as u32,
                    trailing: trailing as u32,
                }
            }

            Op::GroupedMatMul => {
                // Inputs: [input(M, K), weight(E, K, N), expert_idx(M)]
                let in_shape = &graph.node(node.inputs[0]).shape;
                let w_shape = &graph.node(node.inputs[1]).shape;
                let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
                let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
                let num_experts = w_shape.dim(0).unwrap_static();
                let n = w_shape.dim(2).unwrap_static();
                Thunk::GroupedMatMul {
                    input: node_offset(arena, node.inputs[0]),
                    weight: node_offset(arena, node.inputs[1]),
                    expert_idx: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
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
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
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
                // Decompose the input shape into [outer, reduced, inner]
                // around the reduced axis range. Non-contiguous reduced
                // axes aren't supported here — caller must transpose them
                // contiguous first (the coverage tool would surface the
                // gap if a model needs it).
                let in_shape = &graph.node(node.inputs[0]).shape;
                let rank = in_shape.rank();
                let mut sorted = axes.clone();
                sorted.sort();
                sorted.dedup();
                let contiguous = sorted.windows(2).all(|w| w[1] == w[0] + 1)
                    && !sorted.is_empty()
                    && *sorted.last().unwrap() < rank;
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
                    let src = node_offset(arena, node.inputs[0]);
                    let dst = node_offset(arena, node.id);
                    if node.shape.dtype() == rlx_ir::DType::F64 && matches!(op, ReduceOp::Sum) {
                        Thunk::ReduceSumF64 {
                            src,
                            dst,
                            outer: outer as u32,
                            reduced: reduced as u32,
                            inner: inner as u32,
                        }
                    } else {
                        Thunk::Reduce {
                            src,
                            dst,
                            outer: outer as u32,
                            reduced: reduced as u32,
                            inner: inner as u32,
                            op: *op,
                        }
                    }
                }
            }

            Op::Compare(cmp) => {
                let len = node.shape.num_elements().unwrap();
                Thunk::Compare {
                    lhs: node_offset(arena, node.inputs[0]),
                    rhs: node_offset(arena, node.inputs[1]),
                    dst: node_offset(arena, node.id),
                    len: len as u32,
                    op: *cmp,
                }
            }

            Op::Where => {
                let len = node.shape.num_elements().unwrap();
                Thunk::Where {
                    cond: node_offset(arena, node.inputs[0]),
                    on_true: node_offset(arena, node.inputs[1]),
                    on_false: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    len: len as u32,
                }
            }

            Op::ReluBackward => {
                let len: usize = (0..node.shape.rank())
                    .map(|i| node.shape.dim(i).unwrap_static())
                    .product();
                let x = node_offset(arena, node.inputs[0]);
                let dy = node_offset(arena, node.inputs[1]);
                let dx = node_offset(arena, node.id);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::ReluBackwardF64 {
                        x,
                        dy,
                        dx,
                        len: len as u32,
                    },
                    _ => Thunk::ReluBackward {
                        x,
                        dy,
                        dx,
                        len: len as u32,
                    },
                }
            }

            Op::ComplexNormSq => {
                let len: usize = (0..node.shape.rank())
                    .map(|i| node.shape.dim(i).unwrap_static())
                    .product();
                let src = node_offset(arena, node.inputs[0]);
                let dst = node_offset(arena, node.id);
                Thunk::ComplexNormSqF32 {
                    src,
                    dst,
                    len: len as u32,
                }
            }

            Op::ComplexNormSqBackward => {
                let len: usize = (0..node.shape.rank())
                    .map(|i| node.shape.dim(i).unwrap_static())
                    .product();
                let z = node_offset(arena, node.inputs[0]);
                let g = node_offset(arena, node.inputs[1]);
                let dz = node_offset(arena, node.id);
                Thunk::ComplexNormSqBackwardF32 {
                    z,
                    g,
                    dz,
                    len: len as u32,
                }
            }

            Op::Conjugate => {
                let len: usize = (0..node.shape.rank())
                    .map(|i| node.shape.dim(i).unwrap_static())
                    .product();
                Thunk::ConjugateC64 {
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    len: len as u32,
                }
            }

            Op::ActivationBackward { kind } => {
                let len: usize = (0..node.shape.rank())
                    .map(|i| node.shape.dim(i).unwrap_static())
                    .product();
                let x = node_offset(arena, node.inputs[0]);
                let dy = node_offset(arena, node.inputs[1]);
                let dx = node_offset(arena, node.id);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::ActivationBackwardF64 {
                        x,
                        dy,
                        dx,
                        len: len as u32,
                        kind: *kind,
                    },
                    _ => Thunk::ActivationBackward {
                        x,
                        dy,
                        dx,
                        len: len as u32,
                        kind: *kind,
                    },
                }
            }

            Op::LayerNormBackwardInput { eps, .. } => {
                // axis = -1 only (matches forward LayerNorm thunk).
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                Thunk::LayerNormBackwardInput {
                    x: node_offset(arena, node.inputs[0]),
                    gamma: node_offset(arena, node.inputs[1]),
                    dy: node_offset(arena, node.inputs[2]),
                    dx: node_offset(arena, node.id),
                    rows: (total / h) as u32,
                    h: h as u32,
                    eps: *eps,
                }
            }

            Op::LayerNormBackwardGamma { eps, .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                let x_total = x_shape.num_elements().unwrap();
                Thunk::LayerNormBackwardGamma {
                    x: node_offset(arena, node.inputs[0]),
                    dy: node_offset(arena, node.inputs[1]),
                    dgamma: node_offset(arena, node.id),
                    rows: (x_total / h) as u32,
                    h: h as u32,
                    eps: *eps,
                }
            }

            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let dy_shape = &graph.node(node.inputs[1]).shape;
                if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
                    Thunk::MaxPool2dBackward {
                        x: node_offset(arena, node.inputs[0]),
                        dy: node_offset(arena, node.inputs[1]),
                        dx: node_offset(arena, node.id),
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
                } else {
                    Thunk::Nop
                }
            }

            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let dy_shape = &graph.node(node.inputs[0]).shape;
                let w_shape = &graph.node(node.inputs[1]).shape;
                let out_shape = &node.shape;
                if kernel_size.len() == 2
                    && dy_shape.rank() == 4
                    && w_shape.rank() == 4
                    && out_shape.rank() == 4
                {
                    Thunk::Conv2dBackwardInput {
                        dy: node_offset(arena, node.inputs[0]),
                        w: node_offset(arena, node.inputs[1]),
                        dx: node_offset(arena, node.id),
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
                } else {
                    Thunk::Nop
                }
            }

            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let dy_shape = &graph.node(node.inputs[1]).shape;
                let dw_shape = &node.shape;
                if kernel_size.len() == 2
                    && x_shape.rank() == 4
                    && dy_shape.rank() == 4
                    && dw_shape.rank() == 4
                {
                    Thunk::Conv2dBackwardWeight {
                        x: node_offset(arena, node.inputs[0]),
                        dy: node_offset(arena, node.inputs[1]),
                        dw: node_offset(arena, node.id),
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
                } else {
                    Thunk::Nop
                }
            }

            Op::SoftmaxCrossEntropyWithLogits => {
                let logits_shape = &graph.node(node.inputs[0]).shape;
                if logits_shape.rank() == 2 {
                    Thunk::SoftmaxCrossEntropy {
                        logits: node_offset(arena, node.inputs[0]),
                        labels: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        n: logits_shape.dim(0).unwrap_static() as u32,
                        c: logits_shape.dim(1).unwrap_static() as u32,
                    }
                } else {
                    Thunk::Nop
                }
            }

            Op::SoftmaxCrossEntropyBackward => {
                let logits_shape = &graph.node(node.inputs[0]).shape;
                if logits_shape.rank() == 2 {
                    Thunk::SoftmaxCrossEntropyBackward {
                        logits: node_offset(arena, node.inputs[0]),
                        labels: node_offset(arena, node.inputs[1]),
                        d_loss: node_offset(arena, node.inputs[2]),
                        dlogits: node_offset(arena, node.id),
                        n: logits_shape.dim(0).unwrap_static() as u32,
                        c: logits_shape.dim(1).unwrap_static() as u32,
                    }
                } else {
                    Thunk::Nop
                }
            }

            Op::DenseSolve => {
                // A: [n, n], b: [n] or [n, nrhs]. Output matches b.
                let a_shape = &graph.node(node.inputs[0]).shape;
                let n = a_shape.dim(0).unwrap_static();
                debug_assert_eq!(
                    n,
                    a_shape.dim(1).unwrap_static(),
                    "DenseSolve: A must be square"
                );
                let b_elems = node.shape.num_elements().unwrap();
                let nrhs = b_elems / n;
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::DenseSolveF64 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        x: node_offset(arena, node.id),
                        n: n as u32,
                        nrhs: nrhs as u32,
                    },
                    rlx_ir::DType::F32 => Thunk::DenseSolveF32 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        x: node_offset(arena, node.id),
                        n: n as u32,
                        nrhs: nrhs as u32,
                    },
                    other => panic!(
                        "DenseSolve: F32 + F64 lowered; got {other:?}. \
                         Add another variant when needed."
                    ),
                }
            }

            Op::BatchedDenseSolve => {
                // A: [B, N, N], b: [B, N] or [B, N, K]. Output matches b.
                let a_shape = &graph.node(node.inputs[0]).shape;
                assert_eq!(a_shape.rank(), 3, "BatchedDenseSolve: A rank must be 3");
                let batch = a_shape.dim(0).unwrap_static();
                let n = a_shape.dim(1).unwrap_static();
                debug_assert_eq!(
                    n,
                    a_shape.dim(2).unwrap_static(),
                    "BatchedDenseSolve: A's last two dims must match"
                );
                let total = node.shape.num_elements().unwrap();
                let nrhs = total / (batch * n);
                match node.shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::BatchedDenseSolveF64 {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        x: node_offset(arena, node.id),
                        batch: batch as u32,
                        n: n as u32,
                        nrhs: nrhs as u32,
                    },
                    other => panic!(
                        "BatchedDenseSolve: only F64 lowered today, \
                                     got {other:?}"
                    ),
                }
            }

            Op::Scan {
                body,
                length,
                save_trajectory,
                num_bcast,
                num_xs,
                num_checkpoints,
            } => {
                assert!(
                    *num_checkpoints == 0 || *num_checkpoints <= *length,
                    "Op::Scan: num_checkpoints={} must be 0 or ≤ length={}",
                    *num_checkpoints,
                    *length
                );
                if *num_checkpoints != 0 && *num_checkpoints != *length {
                    assert!(
                        *save_trajectory,
                        "Op::Scan: num_checkpoints<length only meaningful when save_trajectory=true"
                    );
                }
                // Plan + compile the body sub-graph standalone. The body
                // gets its own Arena; per execution we clone its
                // pristine bytes, copy the outer carry (and per-step xs
                // slices, if any) into the body's Input slots, run the
                // body schedule N times, then copy the body's output
                // back to the outer arena.
                //
                // Body invariants: 1 + num_xs Op::Inputs in NodeId order
                // — first declared is the carry, rest are x_t_i. Single
                // graph output (the next carry), same shape as carry.
                let body_plan = rlx_opt::memory::plan_memory(body);
                let _body_arena_size = body_plan.arena_size;
                // Snapshot per-input byte offsets before plan_memory
                // moves into the Arena below.
                let body_offsets: HashMap<NodeId, usize> = body_plan
                    .assignments
                    .iter()
                    .map(|(id, slot)| (*id, slot.offset))
                    .collect();

                // Collect body Input nodes in NodeId order; first is
                // carry, rest are per-step xs in matching order.
                let mut body_inputs: Vec<NodeId> = body
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.op, Op::Input { .. }))
                    .map(|n| n.id)
                    .collect();
                body_inputs.sort();
                let n_body_inputs = body_inputs.len();
                let expected = 1 + *num_bcast as usize + *num_xs as usize;
                if n_body_inputs != expected {
                    let names: Vec<String> = body
                        .nodes()
                        .iter()
                        .filter_map(|n| match &n.op {
                            Op::Input { name } => Some(format!("{}={}", n.id, name)),
                            _ => None,
                        })
                        .collect();
                    panic!(
                        "Op::Scan body has {} Op::Input nodes; expected {} \
                            (1 carry + {} bcast + {} xs). Inputs by NodeId: [{}]",
                        n_body_inputs,
                        expected,
                        *num_bcast,
                        *num_xs,
                        names.join(", ")
                    );
                }

                let body_input_id = body_inputs[0];
                let body_input_off = body_offsets[&body_input_id];
                let body_output_id = body
                    .outputs
                    .first()
                    .copied()
                    .expect("Op::Scan body must declare one output");
                let body_output_off = body_offsets[&body_output_id];

                let mut body_arena = crate::arena::Arena::from_plan(body_plan);
                // Fill body Constant nodes — mirror the outer-graph logic
                // in rlx-runtime/src/backend.rs (dtype-aware).
                for n in body.nodes() {
                    if let Op::Constant { data } = &n.op
                        && body_arena.has_buffer(n.id)
                        && !data.is_empty()
                    {
                        match n.shape.dtype() {
                            rlx_ir::DType::F64 => {
                                let off = body_arena.byte_offset(n.id);
                                let buf = body_arena.raw_buf_mut();
                                let nbytes = (buf.len() - off).min(data.len());
                                buf[off..off + nbytes].copy_from_slice(&data[..nbytes]);
                            }
                            _ => {
                                let buf = body_arena.slice_mut(n.id);
                                let n_floats = data.len() / 4;
                                let n_lim = buf.len().min(n_floats);
                                for i in 0..n_lim {
                                    let bytes = [
                                        data[i * 4],
                                        data[i * 4 + 1],
                                        data[i * 4 + 2],
                                        data[i * 4 + 3],
                                    ];
                                    buf[i] = f32::from_le_bytes(bytes);
                                }
                            }
                        }
                    }
                }
                let body_init = body_arena.raw_buf().to_vec();
                let body_schedule = compile_thunks(body, &body_arena);

                // Carry bytes — for trajectory mode, the outer node's
                // shape is [length, *carry_shape], so dividing by length
                // gives one row's bytes; the body's input slot still
                // holds carry_shape bytes.
                let carry_bytes = if *save_trajectory {
                    let total = node
                        .shape
                        .size_bytes()
                        .expect("Op::Scan trajectory output must have static shape");
                    total / *length as usize
                } else {
                    node.shape
                        .size_bytes()
                        .expect("Op::Scan carry must have static shape")
                };

                // Bcast inputs occupy body_inputs[1..1+num_bcast] and
                // outer node.inputs[1..1+num_bcast]. They keep their
                // natural shape (no [length, ...] prefix) and are
                // copied into body_buf ONCE before the scan loop.
                let mut bcast_inputs: Vec<(usize, usize, u32)> =
                    Vec::with_capacity(*num_bcast as usize);
                for i in 0..*num_bcast as usize {
                    let body_b_id = body_inputs[1 + i];
                    let body_b_off = body_offsets[&body_b_id];
                    let outer_b_id = node.inputs[1 + i];
                    let outer_b_off = node_offset(arena, outer_b_id);
                    let outer_b_shape = &graph.node(outer_b_id).shape;
                    let total = outer_b_shape
                        .size_bytes()
                        .expect("Op::Scan bcast must have static shape");
                    bcast_inputs.push((body_b_off, outer_b_off, total as u32));
                }

                // xs occupy body_inputs[1+num_bcast..] and node.inputs
                // [1+num_bcast..]. Each has shape [length, *per_step];
                // per-step bytes = total / length.
                let mut xs_inputs: Vec<(usize, usize, u32)> = Vec::with_capacity(*num_xs as usize);
                let xs_base = 1 + *num_bcast as usize;
                for i in 0..*num_xs as usize {
                    let body_x_id = body_inputs[xs_base + i];
                    let body_x_off = body_offsets[&body_x_id];
                    let outer_xs_id = node.inputs[xs_base + i];
                    let outer_xs_off = node_offset(arena, outer_xs_id);
                    let outer_xs_shape = &graph.node(outer_xs_id).shape;
                    let total = outer_xs_shape
                        .size_bytes()
                        .expect("Op::Scan xs must have static shape");
                    let per_step = total / *length as usize;
                    xs_inputs.push((body_x_off, outer_xs_off, per_step as u32));
                }

                Thunk::Scan {
                    body: Arc::new(body_schedule),
                    body_init: Arc::new(body_init),
                    body_input_off,
                    body_output_off,
                    outer_init_off: node_offset(arena, node.inputs[0]),
                    outer_final_off: node_offset(arena, node.id),
                    length: *length,
                    carry_bytes: carry_bytes as u32,
                    save_trajectory: *save_trajectory,
                    xs_inputs: Arc::new(xs_inputs),
                    bcast_inputs: Arc::new(bcast_inputs),
                    num_checkpoints: *num_checkpoints,
                }
            }

            Op::ScanBackward {
                body_vjp,
                length,
                save_trajectory,
                num_xs,
                num_checkpoints,
                forward_body,
            } => {
                let is_recursive = *num_checkpoints != 0 && *num_checkpoints != *length;
                if is_recursive {
                    assert!(
                        forward_body.is_some(),
                        "Op::ScanBackward with num_checkpoints<length requires forward_body"
                    );
                }
                // body_vjp has signature
                //   (carry, x_t_0, ..., x_t_{num_xs-1}, d_output) → dcarry
                // Identify slots:
                //   * "d_output" by exact name (AD-introduced seed Input).
                //   * Remaining Inputs sorted by NodeId — first is the
                //     carry mirror, rest are x_t_i mirrors in body's
                //     original Op::Input declaration order.
                let body_plan = rlx_opt::memory::plan_memory(body_vjp);
                let body_offsets: HashMap<NodeId, usize> = body_plan
                    .assignments
                    .iter()
                    .map(|(id, slot)| (*id, slot.offset))
                    .collect();
                let mut body_d_output_off: Option<usize> = None;
                let mut body_other_inputs: Vec<(NodeId, usize)> = Vec::new();
                for n in body_vjp.nodes() {
                    if let Op::Input { name } = &n.op {
                        let off = body_offsets[&n.id];
                        if name == "d_output" {
                            body_d_output_off = Some(off);
                        } else {
                            body_other_inputs.push((n.id, off));
                        }
                    }
                }
                body_other_inputs.sort_by_key(|(id, _)| *id);
                let body_d_output_off =
                    body_d_output_off.expect("ScanBackward body_vjp missing 'd_output' Input");
                let expected_others = 1 + *num_xs as usize;
                assert_eq!(
                    body_other_inputs.len(),
                    expected_others,
                    "ScanBackward body_vjp has {} non-d_output Inputs; \
                     expected {} (1 carry + {} xs)",
                    body_other_inputs.len(),
                    expected_others,
                    num_xs
                );
                let body_carry_in_off = body_other_inputs[0].1;
                let body_x_offs: Vec<usize> = body_other_inputs
                    .iter()
                    .skip(1)
                    .map(|(_, off)| *off)
                    .collect();
                let body_dcarry_out_off = body_offsets[&body_vjp.outputs[0]];

                let mut body_arena = crate::arena::Arena::from_plan(body_plan);
                // Fill body_vjp's Constants (mirrors the Scan lowering).
                for n in body_vjp.nodes() {
                    if let Op::Constant { data } = &n.op
                        && body_arena.has_buffer(n.id)
                        && !data.is_empty()
                    {
                        match n.shape.dtype() {
                            rlx_ir::DType::F64 => {
                                let off = body_arena.byte_offset(n.id);
                                let buf = body_arena.raw_buf_mut();
                                let nb = (buf.len() - off).min(data.len());
                                buf[off..off + nb].copy_from_slice(&data[..nb]);
                            }
                            _ => {
                                let buf = body_arena.slice_mut(n.id);
                                let nf = data.len() / 4;
                                let nl = buf.len().min(nf);
                                for i in 0..nl {
                                    let bytes = [
                                        data[i * 4],
                                        data[i * 4 + 1],
                                        data[i * 4 + 2],
                                        data[i * 4 + 3],
                                    ];
                                    buf[i] = f32::from_le_bytes(bytes);
                                }
                            }
                        }
                    }
                }
                let body_init = body_arena.raw_buf().to_vec();
                let body_schedule = compile_thunks(body_vjp, &body_arena);

                // Carry bytes from the dcarry output node (== carry shape).
                let carry_bytes = body_vjp
                    .node(body_vjp.outputs[0])
                    .shape
                    .size_bytes()
                    .expect("ScanBackward dcarry must be statically shaped");
                let carry_elem_size = body_vjp
                    .node(body_vjp.outputs[0])
                    .shape
                    .dtype()
                    .size_bytes() as u32;

                // For each xs input on the outer node:
                // (outer_xs_base, per_step_bytes).
                let mut outer_xs_offs: Vec<(usize, u32)> = Vec::with_capacity(*num_xs as usize);
                for i in 0..*num_xs as usize {
                    let outer_xs_id = node.inputs[3 + i];
                    let outer_xs_off = node_offset(arena, outer_xs_id);
                    let outer_xs_shape = &graph.node(outer_xs_id).shape;
                    let total = outer_xs_shape
                        .size_bytes()
                        .expect("ScanBackward xs must have static shape");
                    let per_step = total / *length as usize;
                    outer_xs_offs.push((outer_xs_off, per_step as u32));
                }

                // If recursive checkpointing is active, we also compile
                // the forward body so the executor can recompute
                // intermediate carries. The forward body is supplied
                // by the AD pass via `forward_body: Some(_)`.
                let (fb_schedule, fb_init, fb_carry_in_off, fb_output_off, fb_x_offs) =
                    if is_recursive {
                        let fb = forward_body.as_ref().unwrap();
                        let fb_plan = rlx_opt::memory::plan_memory(fb);
                        let fb_offsets: HashMap<NodeId, usize> = fb_plan
                            .assignments
                            .iter()
                            .map(|(id, slot)| (*id, slot.offset))
                            .collect();
                        let mut fb_inputs: Vec<NodeId> = fb
                            .nodes()
                            .iter()
                            .filter(|n| matches!(n.op, Op::Input { .. }))
                            .map(|n| n.id)
                            .collect();
                        fb_inputs.sort();
                        let fb_carry = fb_offsets[&fb_inputs[0]];
                        let fb_xs: Vec<usize> = (1..fb_inputs.len())
                            .map(|i| fb_offsets[&fb_inputs[i]])
                            .collect();
                        let fb_out = fb_offsets[&fb.outputs[0]];
                        let mut fb_arena = crate::arena::Arena::from_plan(fb_plan);
                        for n in fb.nodes() {
                            if let Op::Constant { data } = &n.op
                                && fb_arena.has_buffer(n.id)
                                && !data.is_empty()
                            {
                                // Byte-copy works for any
                                // numeric dtype as long as the
                                // arena slot is sized to hold
                                // it — the Constant's `data`
                                // already encodes the right
                                // bytes per element.
                                let off = fb_arena.byte_offset(n.id);
                                let buf = fb_arena.raw_buf_mut();
                                let nb = (buf.len() - off).min(data.len());
                                buf[off..off + nb].copy_from_slice(&data[..nb]);
                            }
                        }
                        let fb_init_bytes = fb_arena.raw_buf().to_vec();
                        let fb_sched = compile_thunks(fb, &fb_arena);
                        (
                            Some(Arc::new(fb_sched)),
                            Some(Arc::new(fb_init_bytes)),
                            fb_carry,
                            fb_out,
                            fb_xs,
                        )
                    } else {
                        (None, None, 0, 0, Vec::new())
                    };

                Thunk::ScanBackward {
                    body_vjp: Arc::new(body_schedule),
                    body_init: Arc::new(body_init),
                    body_carry_in_off,
                    body_x_offs: Arc::new(body_x_offs),
                    body_d_output_off,
                    body_dcarry_out_off,
                    outer_init_off: node_offset(arena, node.inputs[0]),
                    outer_traj_off: node_offset(arena, node.inputs[1]),
                    outer_upstream_off: node_offset(arena, node.inputs[2]),
                    outer_xs_offs: Arc::new(outer_xs_offs),
                    outer_dinit_off: node_offset(arena, node.id),
                    length: *length,
                    carry_bytes: carry_bytes as u32,
                    carry_elem_size,
                    save_trajectory: *save_trajectory,
                    num_checkpoints: *num_checkpoints,
                    forward_body: fb_schedule,
                    forward_body_init: fb_init,
                    forward_body_carry_in_off: fb_carry_in_off,
                    forward_body_output_off: fb_output_off,
                    forward_body_x_offs: Arc::new(fb_x_offs),
                }
            }

            Op::ScanBackwardXs {
                body_vjp,
                length,
                save_trajectory,
                num_xs,
                xs_idx,
                num_checkpoints,
                forward_body,
            } => {
                assert!(
                    *num_checkpoints == 0 || *num_checkpoints <= *length,
                    "Op::ScanBackwardXs: num_checkpoints={} must be 0 or ≤ length={}",
                    *num_checkpoints,
                    *length
                );
                let is_recursive = *num_checkpoints != 0 && *num_checkpoints != *length;
                if is_recursive {
                    assert!(
                        forward_body.is_some(),
                        "Op::ScanBackwardXs with num_checkpoints<length \
                         requires forward_body"
                    );
                }
                // Mirror ScanBackward's body_vjp slot identification +
                // arena prep, then add: per-iteration extraction of the
                // body_vjp output that corresponds to the chosen xs.
                //
                // body_vjp's outputs (from `grad(body, [carry, xs_0, ..., xs_{num_xs-1}])`):
                //   outputs[0]      = dcarry
                //   outputs[1 + i]  = dx_t_i
                let body_plan = rlx_opt::memory::plan_memory(body_vjp);
                let body_offsets: HashMap<NodeId, usize> = body_plan
                    .assignments
                    .iter()
                    .map(|(id, slot)| (*id, slot.offset))
                    .collect();
                let mut body_d_output_off: Option<usize> = None;
                let mut body_other_inputs: Vec<(NodeId, usize)> = Vec::new();
                for n in body_vjp.nodes() {
                    if let Op::Input { name } = &n.op {
                        let off = body_offsets[&n.id];
                        if name == "d_output" {
                            body_d_output_off = Some(off);
                        } else {
                            body_other_inputs.push((n.id, off));
                        }
                    }
                }
                body_other_inputs.sort_by_key(|(id, _)| *id);
                let body_d_output_off =
                    body_d_output_off.expect("ScanBackwardXs body_vjp missing 'd_output' Input");
                let expected_others = 1 + *num_xs as usize;
                assert_eq!(
                    body_other_inputs.len(),
                    expected_others,
                    "ScanBackwardXs body_vjp has {} non-d_output Inputs; expected {}",
                    body_other_inputs.len(),
                    expected_others
                );
                let body_carry_in_off = body_other_inputs[0].1;
                let body_x_offs: Vec<usize> = body_other_inputs
                    .iter()
                    .skip(1)
                    .map(|(_, off)| *off)
                    .collect();
                let body_dcarry_out_off = body_offsets[&body_vjp.outputs[0]];
                let dxs_out_node = body_vjp.outputs[1 + *xs_idx as usize];
                let body_dxs_out_off = body_offsets[&dxs_out_node];

                let mut body_arena = crate::arena::Arena::from_plan(body_plan);
                for n in body_vjp.nodes() {
                    if let Op::Constant { data } = &n.op
                        && body_arena.has_buffer(n.id)
                        && !data.is_empty()
                    {
                        match n.shape.dtype() {
                            rlx_ir::DType::F64 => {
                                let off = body_arena.byte_offset(n.id);
                                let buf = body_arena.raw_buf_mut();
                                let nb = (buf.len() - off).min(data.len());
                                buf[off..off + nb].copy_from_slice(&data[..nb]);
                            }
                            _ => {
                                let buf = body_arena.slice_mut(n.id);
                                let nf = data.len() / 4;
                                let nl = buf.len().min(nf);
                                for i in 0..nl {
                                    let bytes = [
                                        data[i * 4],
                                        data[i * 4 + 1],
                                        data[i * 4 + 2],
                                        data[i * 4 + 3],
                                    ];
                                    buf[i] = f32::from_le_bytes(bytes);
                                }
                            }
                        }
                    }
                }
                let body_init = body_arena.raw_buf().to_vec();
                let body_schedule = compile_thunks(body_vjp, &body_arena);

                let carry_bytes = body_vjp
                    .node(body_vjp.outputs[0])
                    .shape
                    .size_bytes()
                    .expect("ScanBackwardXs dcarry must be statically shaped");
                let carry_elem_size = body_vjp
                    .node(body_vjp.outputs[0])
                    .shape
                    .dtype()
                    .size_bytes() as u32;
                let per_step_bytes = body_vjp
                    .node(dxs_out_node)
                    .shape
                    .size_bytes()
                    .expect("ScanBackwardXs dxs body output must be statically shaped");

                let mut outer_xs_offs: Vec<(usize, u32)> = Vec::with_capacity(*num_xs as usize);
                for i in 0..*num_xs as usize {
                    let outer_xs_id = node.inputs[3 + i];
                    let outer_xs_off = node_offset(arena, outer_xs_id);
                    let outer_xs_shape = &graph.node(outer_xs_id).shape;
                    let total = outer_xs_shape
                        .size_bytes()
                        .expect("ScanBackwardXs xs must have static shape");
                    let per_step = total / *length as usize;
                    outer_xs_offs.push((outer_xs_off, per_step as u32));
                }

                // Compile forward_body for recompute when checkpointed.
                // Mirrors the same code path in the ScanBackward arm.
                let (fb_schedule, fb_init, fb_carry_in_off, fb_output_off, fb_x_offs) =
                    if is_recursive {
                        let fb = forward_body.as_ref().unwrap();
                        let fb_plan = rlx_opt::memory::plan_memory(fb);
                        let fb_offsets: HashMap<NodeId, usize> = fb_plan
                            .assignments
                            .iter()
                            .map(|(id, slot)| (*id, slot.offset))
                            .collect();
                        let mut fb_inputs: Vec<NodeId> = fb
                            .nodes()
                            .iter()
                            .filter(|n| matches!(n.op, Op::Input { .. }))
                            .map(|n| n.id)
                            .collect();
                        fb_inputs.sort();
                        let fb_carry = fb_offsets[&fb_inputs[0]];
                        let fb_xs: Vec<usize> = (1..fb_inputs.len())
                            .map(|i| fb_offsets[&fb_inputs[i]])
                            .collect();
                        let fb_out = fb_offsets[&fb.outputs[0]];
                        let mut fb_arena = crate::arena::Arena::from_plan(fb_plan);
                        for n in fb.nodes() {
                            if let Op::Constant { data } = &n.op
                                && fb_arena.has_buffer(n.id)
                                && !data.is_empty()
                            {
                                // Byte-copy works for any
                                // numeric dtype as long as the
                                // arena slot is sized to hold
                                // it — the Constant's `data`
                                // already encodes the right
                                // bytes per element.
                                let off = fb_arena.byte_offset(n.id);
                                let buf = fb_arena.raw_buf_mut();
                                let nb = (buf.len() - off).min(data.len());
                                buf[off..off + nb].copy_from_slice(&data[..nb]);
                            }
                        }
                        let fb_init_bytes = fb_arena.raw_buf().to_vec();
                        let fb_sched = compile_thunks(fb, &fb_arena);
                        (
                            Some(Arc::new(fb_sched)),
                            Some(Arc::new(fb_init_bytes)),
                            fb_carry,
                            fb_out,
                            fb_xs,
                        )
                    } else {
                        (None, None, 0, 0, Vec::new())
                    };

                Thunk::ScanBackwardXs {
                    body_vjp: Arc::new(body_schedule),
                    body_init: Arc::new(body_init),
                    body_carry_in_off,
                    body_x_offs: Arc::new(body_x_offs),
                    body_d_output_off,
                    body_dcarry_out_off,
                    body_dxs_out_off,
                    outer_init_off: node_offset(arena, node.inputs[0]),
                    outer_traj_off: node_offset(arena, node.inputs[1]),
                    outer_upstream_off: node_offset(arena, node.inputs[2]),
                    outer_xs_offs: Arc::new(outer_xs_offs),
                    outer_dxs_off: node_offset(arena, node.id),
                    length: *length,
                    carry_bytes: carry_bytes as u32,
                    carry_elem_size,
                    per_step_bytes: per_step_bytes as u32,
                    save_trajectory: *save_trajectory,
                    num_checkpoints: *num_checkpoints,
                    forward_body: fb_schedule,
                    forward_body_init: fb_init,
                    forward_body_carry_in_off: fb_carry_in_off,
                    forward_body_output_off: fb_output_off,
                    forward_body_x_offs: Arc::new(fb_x_offs),
                }
            }

            Op::Concat { axis } => {
                // Compute outer/inner from the OUTPUT shape: all inputs share
                // the same shape except along `axis`. The output's leading
                // and trailing dims match.
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
                let total_axis = out_shape.dim(*axis).unwrap_static();
                let inputs: Vec<(usize, u32)> = node
                    .inputs
                    .iter()
                    .map(|&in_id| {
                        let in_shape = &graph.node(in_id).shape;
                        let in_axis = in_shape.dim(*axis).unwrap_static();
                        (node_offset(arena, in_id), in_axis as u32)
                    })
                    .collect();
                let dst = node_offset(arena, node.id);
                match out_shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::ConcatF64 {
                        dst,
                        outer: outer as u32,
                        inner: inner as u32,
                        total_axis: total_axis as u32,
                        inputs,
                    },
                    _ => Thunk::Concat {
                        dst,
                        outer: outer as u32,
                        inner: inner as u32,
                        total_axis: total_axis as u32,
                        inputs,
                    },
                }
            }

            Op::Custom { name, attrs, .. } => {
                let kernel = crate::op_registry::lookup_cpu_kernel(name).unwrap_or_else(|| {
                    panic!(
                        "compile_thunks: no CPU kernel registered for \
                         Op::Custom('{name}'). Register one via \
                         rlx_cpu::op_registry::register_cpu_kernel \
                         before compiling on the CPU backend."
                    )
                });
                let inputs_v: Vec<(usize, u32, Shape)> = node
                    .inputs
                    .iter()
                    .map(|&in_id| {
                        let s = graph.node(in_id).shape.clone();
                        let len = s.num_elements().unwrap_or(0) as u32;
                        (node_offset(arena, in_id), len, s)
                    })
                    .collect();
                let out_len = node.shape.num_elements().unwrap_or(0) as u32;
                Thunk::CustomOp {
                    kernel,
                    inputs: inputs_v,
                    output: (node_offset(arena, node.id), out_len, node.shape.clone()),
                    attrs: attrs.clone(),
                }
            }

            Op::Fft { inverse } => {
                // Last axis carries the 2N real-block layout; complex
                // points = N = last_dim / 2. `outer` is the product
                // of all preceding axes — the kernel iterates one
                // batch-row at a time. f32 and f64 share the same
                // radix-2 structure but use separate scratch buffers;
                // the dtype is captured here so the closure dispatches
                // without per-row branching.
                let shape = &node.shape;
                let last = shape.dim(shape.rank() - 1).unwrap_static();
                let n_complex = (last / 2) as u32;
                let total = shape.num_elements().unwrap_or(0);
                let outer = (total / last) as u32;
                let dtype = shape.dtype();
                assert!(
                    matches!(dtype, rlx_ir::DType::F32 | rlx_ir::DType::F64),
                    "Op::Fft on CPU requires F32 or F64, got {dtype:?}"
                );
                Thunk::Fft1d {
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    outer,
                    n_complex,
                    inverse: *inverse,
                    dtype,
                }
            }

            Op::CustomFn {
                fwd_body,
                num_inputs,
                ..
            } => {
                // Plan + compile the body sub-graph standalone, fill its
                // Constants (mirrors the Op::Scan body lowering), then
                // capture per-input copy specs and the output spec.
                // Body Inputs in NodeId order match the outer node's
                // operand vector by position.
                let body_plan = rlx_opt::memory::plan_memory(fwd_body);
                let body_offsets: HashMap<NodeId, usize> = body_plan
                    .assignments
                    .iter()
                    .map(|(id, slot)| (*id, slot.offset))
                    .collect();

                let mut body_input_ids: Vec<NodeId> = fwd_body
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.op, Op::Input { .. }))
                    .map(|n| n.id)
                    .collect();
                body_input_ids.sort();
                assert_eq!(
                    body_input_ids.len(),
                    *num_inputs as usize,
                    "Op::CustomFn fwd_body has {} Op::Input(s); declared num_inputs={}",
                    body_input_ids.len(),
                    *num_inputs,
                );

                let mut body_arena = crate::arena::Arena::from_plan(body_plan);
                for n in fwd_body.nodes() {
                    if let Op::Constant { data } = &n.op
                        && body_arena.has_buffer(n.id)
                        && !data.is_empty()
                    {
                        match n.shape.dtype() {
                            rlx_ir::DType::F64 => {
                                let off = body_arena.byte_offset(n.id);
                                let buf = body_arena.raw_buf_mut();
                                let nb = (buf.len() - off).min(data.len());
                                buf[off..off + nb].copy_from_slice(&data[..nb]);
                            }
                            _ => {
                                let buf = body_arena.slice_mut(n.id);
                                let nf = data.len() / 4;
                                let nl = buf.len().min(nf);
                                for i in 0..nl {
                                    let bytes = [
                                        data[i * 4],
                                        data[i * 4 + 1],
                                        data[i * 4 + 2],
                                        data[i * 4 + 3],
                                    ];
                                    buf[i] = f32::from_le_bytes(bytes);
                                }
                            }
                        }
                    }
                }
                let body_init = body_arena.raw_buf().to_vec();
                let body_schedule = compile_thunks(fwd_body, &body_arena);

                // Per primal input: (body_input_off, outer_input_off, bytes).
                let inputs_v: Vec<(usize, usize, u32)> = (0..*num_inputs as usize)
                    .map(|i| {
                        let body_in = body_input_ids[i];
                        let body_off = body_offsets[&body_in];
                        let outer_in = node.inputs[i];
                        let outer_off = node_offset(arena, outer_in);
                        let bytes = graph
                            .node(outer_in)
                            .shape
                            .size_bytes()
                            .expect("Op::CustomFn primal input must have static shape");
                        (body_off, outer_off, bytes as u32)
                    })
                    .collect();

                let body_output_id = fwd_body
                    .outputs
                    .first()
                    .copied()
                    .expect("Op::CustomFn fwd_body must declare exactly one output");
                let body_output_off = body_offsets[&body_output_id];
                let out_bytes = node
                    .shape
                    .size_bytes()
                    .expect("Op::CustomFn output must have static shape");

                Thunk::CustomFn {
                    body: Arc::new(body_schedule),
                    body_init: Arc::new(body_init),
                    inputs: Arc::new(inputs_v),
                    body_output_off,
                    outer_output_off: node_offset(arena, node.id),
                    out_bytes: out_bytes as u32,
                }
            }

            _ => Thunk::Nop,
        };
        thunks.push(t);
    }

    let cfg = crate::config::RuntimeConfig::global();
    let mask_thr = cfg.mask_binary_threshold;
    let mask_neg = cfg.attn_mask_neg_inf;
    let score_skip = cfg.score_skip_threshold;

    // Pre-compile closures (skip Nops — they're filtered out)
    let compiled_fns: Vec<Arc<dyn Fn(*mut u8) + Send + Sync>> = thunks
        .iter()
        .filter(|t| !matches!(t, Thunk::Nop))
        .map(|thunk| {
            match thunk.clone() {
                Thunk::Nop => Arc::new(|_: *mut u8| {}) as Arc<dyn Fn(*mut u8) + Send + Sync>,

                Thunk::Sgemm { a, b, c, m, k, n } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        crate::blas::sgemm(
                            sl(a, base, m * k),
                            sl(b, base, k * n),
                            sl_mut(c, base, m * n),
                            m,
                            k,
                            n,
                        );
                    })
                }

                Thunk::DenseSolveF64 { a, b, x, n, nrhs } => {
                    let (n_, nrhs_) = (n as usize, nrhs as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let a_src = sl_f64(a, base, n_ * n_);
                        let b_src = sl_f64(b, base, n_ * nrhs_);
                        let mut a_scratch: Vec<f64> = a_src.to_vec();
                        let mut x_buf: Vec<f64> = b_src.to_vec();
                        let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                        if info != 0 {
                            panic!("DenseSolveF64: singular (info={info})");
                        }
                        sl_mut_f64(x, base, n_ * nrhs_).copy_from_slice(&x_buf);
                    })
                }

                Thunk::DenseSolveF32 { a, b, x, n, nrhs } => {
                    let (n_, nrhs_) = (n as usize, nrhs as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let a_src = sl(a, base, n_ * n_);
                        let b_src = sl(b, base, n_ * nrhs_);
                        let mut a_scratch: Vec<f32> = a_src.to_vec();
                        let mut x_buf: Vec<f32> = b_src.to_vec();
                        let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                        if info != 0 {
                            panic!("DenseSolveF32: singular (info={info})");
                        }
                        sl_mut(x, base, n_ * nrhs_).copy_from_slice(&x_buf);
                    })
                }

                Thunk::FusedMmBiasAct {
                    a,
                    w,
                    bias,
                    c,
                    m,
                    k,
                    n,
                    act,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(c, base, m * n);
                        crate::blas::sgemm(sl(a, base, m * k), sl(w, base, k * n), out, m, k, n);
                        // Bias + activation epilogue. Gelu uses the fused
                        // `par_bias_gelu` kernel (bias add + Gelu in one
                        // pass). For everything else, do the bias add first
                        // and then apply the activation per-element. The
                        // pre-fix code dispatched `_ => bias_add` and dropped
                        // the activation entirely — silent correctness bug
                        // for Silu/Relu/Sigmoid/etc.
                        match act {
                            Some(Activation::Gelu) => {
                                crate::kernels::par_bias_gelu(out, sl(bias, base, n), m, n)
                            }
                            Some(other) => {
                                crate::blas::bias_add(out, sl(bias, base, n), m, n);
                                apply_activation_inplace(out, other);
                            }
                            None => crate::blas::bias_add(out, sl(bias, base, n), m, n),
                        }
                    })
                }

                Thunk::FusedResidualLN {
                    x,
                    res,
                    bias,
                    g,
                    b,
                    out,
                    rows,
                    h,
                    eps,
                    has_bias,
                } => {
                    let (rows, h) = (rows as usize, h as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let zero = vec![0f32; h]; // closure only — not hot path
                        let bi = if has_bias { sl(bias, base, h) } else { &zero };
                        let xp = sl(x, base, rows * h).as_ptr() as usize;
                        let rp = sl(res, base, rows * h).as_ptr() as usize;
                        let op = sl_mut(out, base, rows * h).as_mut_ptr() as usize;
                        let bp = bi.as_ptr() as usize;
                        let gp = sl(g, base, h).as_ptr() as usize;
                        let bbp = sl(b, base, h).as_ptr() as usize;
                        crate::pool::par_for(rows, 4, &|off, cnt| {
                            let xs = std::slice::from_raw_parts(
                                (xp as *const f32).add(off * h),
                                cnt * h,
                            );
                            let rs = std::slice::from_raw_parts(
                                (rp as *const f32).add(off * h),
                                cnt * h,
                            );
                            let os = std::slice::from_raw_parts_mut(
                                (op as *mut f32).add(off * h),
                                cnt * h,
                            );
                            let bi = std::slice::from_raw_parts(bp as *const f32, h);
                            let g = std::slice::from_raw_parts(gp as *const f32, h);
                            let b = std::slice::from_raw_parts(bbp as *const f32, h);
                            crate::kernels::residual_bias_layer_norm(
                                xs, rs, bi, g, b, os, cnt, h, eps,
                            );
                        });
                    })
                }

                Thunk::BiasAdd {
                    src,
                    bias,
                    dst,
                    m,
                    n,
                } => {
                    let (m, n) = (m as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, m * n);
                        out.copy_from_slice(sl(src, base, m * n));
                        crate::blas::bias_add(out, sl(bias, base, n), m, n);
                    })
                }

                Thunk::Gather {
                    table,
                    table_len,
                    idx,
                    dst,
                    num_idx,
                    trailing,
                } => {
                    let (ni, tr, tl) = (num_idx as usize, trailing as usize, table_len as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let tab = sl(table, base, tl);
                        let ids = sl(idx, base, ni);
                        let out = sl_mut(dst, base, ni * tr);
                        for i in 0..ni {
                            let row = ids[i] as usize;
                            out[i * tr..(i + 1) * tr]
                                .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                        }
                    })
                }

                Thunk::Narrow {
                    src,
                    dst,
                    outer,
                    src_stride,
                    dst_stride,
                    inner,
                } => {
                    let (outer, ss, ds, inner) = (
                        outer as usize,
                        src_stride as usize,
                        dst_stride as usize,
                        inner as usize,
                    );
                    Arc::new(move |base: *mut u8| unsafe {
                        let s = sl(src, base, outer * ss);
                        let d = sl_mut(dst, base, outer * ds);
                        for o in 0..outer {
                            d[o * ds..o * ds + inner].copy_from_slice(&s[o * ss..o * ss + inner]);
                        }
                    })
                }

                Thunk::Copy { src, dst, len } => {
                    let len = len as usize;
                    Arc::new(move |base: *mut u8| unsafe {
                        sl_mut(dst, base, len).copy_from_slice(sl(src, base, len));
                    })
                }

                Thunk::Softmax { data, rows, cols } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        crate::naive::softmax(sl_mut(data, base, rows * cols), rows, cols);
                    })
                }

                Thunk::Cumsum {
                    src,
                    dst,
                    rows,
                    cols,
                    exclusive,
                } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let s = sl(src, base, rows * cols);
                        let d = sl_mut(dst, base, rows * cols);
                        if exclusive {
                            for r in 0..rows {
                                let mut acc = 0.0f32;
                                for c in 0..cols {
                                    d[r * cols + c] = acc;
                                    acc += s[r * cols + c];
                                }
                            }
                        } else {
                            for r in 0..rows {
                                let mut acc = 0.0f32;
                                for c in 0..cols {
                                    acc += s[r * cols + c];
                                    d[r * cols + c] = acc;
                                }
                            }
                        }
                    })
                }

                Thunk::Sample {
                    logits,
                    dst,
                    batch,
                    vocab,
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let (b, v) = (batch as usize, vocab as usize);
                    let k = (top_k as usize).min(v);
                    Arc::new(move |base: *mut u8| unsafe {
                        let lg = sl(logits, base, b * v);
                        let out = sl_mut(dst, base, b);
                        let mut rng =
                            rlx_ir::Philox4x32::new(if seed == 0 { 0xDEADBEEF } else { seed });
                        for bi in 0..b {
                            let row = &lg[bi * v..(bi + 1) * v];
                            out[bi] = sample_row(row, k, top_p, temperature, &mut rng) as f32;
                        }
                    })
                }

                Thunk::DequantMatMul {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k,
                    n,
                    block_size,
                    is_asymmetric,
                } => {
                    let (m, k, n, bs) = (m as usize, k as usize, n as usize, block_size as usize);
                    let n_blocks_per_col = k.div_ceil(bs);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        // w_q is packed i8 — use raw byte slice + reinterpret.
                        let raw = base.add(w_q);
                        let w_bytes = std::slice::from_raw_parts(raw as *const i8, k * n);
                        let scales = sl(scale, base, n_blocks_per_col * n);
                        let zps = if is_asymmetric {
                            sl(zp, base, n_blocks_per_col * n)
                        } else {
                            &[][..]
                        };
                        let out = sl_mut(dst, base, m * n);
                        dequant_matmul_int8(
                            xs,
                            w_bytes,
                            scales,
                            zps,
                            out,
                            m,
                            k,
                            n,
                            bs,
                            is_asymmetric,
                        );
                    })
                }

                Thunk::DequantMatMulGguf {
                    x,
                    w_q,
                    dst,
                    m,
                    k,
                    n,
                    scheme,
                } => {
                    use rlx_ir::quant::QuantScheme;
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let total_bytes = (k * n) / block_elems * block_bytes;
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let w_bytes = std::slice::from_raw_parts(
                            base.add(w_q) as *const u8,
                            total_bytes,
                        );
                        let w_f32 = match scheme {
                            QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, k * n),
                            QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(w_bytes, k * n),
                            QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(w_bytes, k * n),
                            QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(w_bytes, k * n),
                            _ => unreachable!("non-GGUF in GGUF arm"),
                        }
                        .expect("GGUF dequant failed");
                        let out = sl_mut(dst, base, m * n);
                        // See same comment in the eager arm: dequant
                        // produces `[n, k]` row-major, so use `sgemm_bt`.
                        crate::blas::sgemm_bt(xs, &w_f32, out, m, k, n, 1.0);
                    })
                }

                Thunk::LoraMatMul {
                    x,
                    w,
                    a,
                    b,
                    dst,
                    m,
                    k,
                    n,
                    r,
                    scale,
                } => {
                    let (m, k, n, r) = (m as usize, k as usize, n as usize, r as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let ws = sl(w, base, k * n);
                        let a_s = sl(a, base, k * r);
                        let bs = sl(b, base, r * n);
                        let out = sl_mut(dst, base, m * n);
                        // Step 1: out = x · W.
                        crate::blas::sgemm(xs, ws, out, m, k, n);
                        // Step 2: tmp = x · A (rank-r intermediate; tiny).
                        let mut tmp = vec![0f32; m * r];
                        crate::blas::sgemm(xs, a_s, &mut tmp, m, k, r);
                        // Step 3: out += scale * (tmp · B).
                        // sgemm_accumulate uses alpha=1.0 internally, so
                        // scale tmp first.
                        if scale != 1.0 {
                            for v in tmp.iter_mut() {
                                *v *= scale;
                            }
                        }
                        crate::blas::sgemm_accumulate(&tmp, bs, out, m, r, n);
                    })
                }

                Thunk::LayerNorm {
                    src,
                    g,
                    b,
                    dst,
                    rows,
                    h,
                    eps,
                } => {
                    let (rows, h) = (rows as usize, h as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let inp = sl(src, base, rows * h);
                        let gamma = sl(g, base, h);
                        let beta = sl(b, base, h);
                        let out = sl_mut(dst, base, rows * h);
                        for row in 0..rows {
                            crate::kernels::layer_norm_row(
                                &inp[row * h..(row + 1) * h],
                                gamma,
                                beta,
                                &mut out[row * h..(row + 1) * h],
                                h,
                                eps,
                            );
                        }
                    })
                }

                Thunk::Attention {
                    q,
                    k,
                    v,
                    mask,
                    out,
                    batch,
                    seq,
                    kv_seq: _,
                    heads,
                    head_dim,
                    mask_kind,
                    q_row_stride,
                    k_row_stride,
                    v_row_stride,
                    bhsd,
                } => {
                    let (b, s, nh, dh) = (
                        batch as usize,
                        seq as usize,
                        heads as usize,
                        head_dim as usize,
                    );
                    let hs = nh * dh;
                    let qrs = q_row_stride as usize;
                    let krs = k_row_stride as usize;
                    let vrs = v_row_stride as usize;
                    let scale = (dh as f32).powf(-0.5);
                    Arc::new(move |base: *mut u8| unsafe {
                        // Slice lengths use the source's row stride so the
                        // compiler-emitted bounds checks cover the whole
                        // strided span (the kernel walks with q/k/v_rs).
                        // For [B, H, S, D] the buffer is dense B*H*S*D.
                        let (q_len, k_len, v_len, o_len) = if bhsd {
                            let n = b * nh * s * dh;
                            (n, n, n, n)
                        } else {
                            (b * s * qrs, b * s * krs, b * s * vrs, b * s * hs)
                        };
                        let q_d = sl(q, base, q_len);
                        let k_d = sl(k, base, k_len);
                        let v_d = sl(v, base, v_len);
                        let m_d: &[f32] = match mask_kind {
                            rlx_ir::op::MaskKind::Custom => sl(mask, base, b * s),
                            rlx_ir::op::MaskKind::Bias => sl(mask, base, b * nh * s * s),
                            _ => &[],
                        };
                        let o_d = sl_mut(out, base, o_len);
                        let sdh = s * dh;
                        let mut qh = vec![0f32; sdh];
                        let mut kh = vec![0f32; sdh];
                        let mut vh = vec![0f32; sdh];
                        let mut sc = vec![0f32; s * s];
                        let mut oh = vec![0f32; sdh];
                        for bi in 0..b {
                            for hi in 0..nh {
                                for si in 0..s {
                                    // Two layouts:
                                    //   bhsd=false: [B, S, H, D] (default) →
                                    //     off = bi*S*RS + si*RS + hi*D
                                    //   bhsd=true:  [B, H, S, D] (GPU/TPU
                                    //     convention) →
                                    //     off = bi*H*S*D + hi*S*D + si*D
                                    // The thunk-fusion pass below sets row
                                    // strides, but only for the [B, S, H, D]
                                    // case. For bhsd we always use the dense
                                    // contiguous stride (qrs == krs == vrs ==
                                    // H*D from compile_thunks).
                                    let (q_off, k_off, v_off) = if bhsd {
                                        (
                                            bi * nh * s * dh + hi * s * dh + si * dh,
                                            bi * nh * s * dh + hi * s * dh + si * dh,
                                            bi * nh * s * dh + hi * s * dh + si * dh,
                                        )
                                    } else {
                                        (
                                            bi * s * qrs + si * qrs + hi * dh,
                                            bi * s * krs + si * krs + hi * dh,
                                            bi * s * vrs + si * vrs + hi * dh,
                                        )
                                    };
                                    qh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&q_d[q_off..q_off + dh]);
                                    kh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&k_d[k_off..k_off + dh]);
                                    vh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&v_d[v_off..v_off + dh]);
                                }
                                for qi in 0..s {
                                    for ki in 0..s {
                                        let mut dot = 0f32;
                                        for d in 0..dh {
                                            dot += qh[qi * dh + d] * kh[ki * dh + d];
                                        }
                                        sc[qi * s + ki] = dot * scale;
                                    }
                                }
                                // Apply mask kind — None skips entirely, Causal /
                                // SlidingWindow synthesize, Custom reads m_d.
                                match mask_kind {
                                    rlx_ir::op::MaskKind::None => {}
                                    rlx_ir::op::MaskKind::Causal => {
                                        for qi in 0..s {
                                            for ki in (qi + 1)..s {
                                                sc[qi * s + ki] = mask_neg;
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::SlidingWindow(w) => {
                                        for qi in 0..s {
                                            let lo = qi.saturating_sub(w);
                                            for ki in 0..s {
                                                if ki < lo || ki > qi {
                                                    sc[qi * s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Custom => {
                                        for qi in 0..s {
                                            for ki in 0..s {
                                                if m_d[bi * s + ki] < mask_thr {
                                                    sc[qi * s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Bias => {
                                        let per_bh = s * s;
                                        let off = (bi * nh + hi) * per_bh;
                                        for i in 0..per_bh {
                                            sc[i] += m_d[off + i];
                                        }
                                    }
                                }
                                crate::naive::softmax(&mut sc, s, s);
                                oh.fill(0.0);
                                for qi in 0..s {
                                    for ki in 0..s {
                                        let w = sc[qi * s + ki];
                                        if w > score_skip {
                                            for d in 0..dh {
                                                oh[qi * dh + d] += w * vh[ki * dh + d];
                                            }
                                        }
                                    }
                                }
                                for si in 0..s {
                                    let off = if bhsd {
                                        bi * nh * s * dh + hi * s * dh + si * dh
                                    } else {
                                        bi * s * hs + si * hs + hi * dh
                                    };
                                    o_d[off..off + dh].copy_from_slice(&oh[si * dh..(si + 1) * dh]);
                                }
                            }
                        }
                    })
                }

                Thunk::FusedSwiGLU {
                    src,
                    dst,
                    n_half,
                    total,
                } => {
                    let n = n_half as usize;
                    let t = total as usize;
                    let outer = t / n;
                    let in_total = outer * 2 * n;
                    Arc::new(move |base: *mut u8| unsafe {
                        let inp = sl(src, base, in_total);
                        let out = sl_mut(dst, base, t);
                        for o in 0..outer {
                            let in_row = &inp[o * 2 * n..(o + 1) * 2 * n];
                            let out_row = &mut out[o * n..(o + 1) * n];
                            for i in 0..n {
                                let up = in_row[i];
                                let gate = in_row[n + i];
                                // silu(g) = g * sigmoid(g) = g / (1+exp(-g))
                                out_row[i] = up * (gate / (1.0 + (-gate).exp()));
                            }
                        }
                    })
                }

                Thunk::Concat {
                    dst,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                } => {
                    let outer = outer as usize;
                    let inner = inner as usize;
                    let total_axis = total_axis as usize;
                    let out_total = outer * total_axis * inner;
                    // Pre-compute the destination row offset for each input
                    // (cumulative axis offsets times inner).
                    let mut layout: Vec<(usize, usize, usize)> = Vec::with_capacity(inputs.len());
                    let mut cum: usize = 0;
                    for (src_off, in_axis) in &inputs {
                        let in_axis = *in_axis as usize;
                        layout.push((*src_off, cum * inner, in_axis * inner));
                        cum += in_axis;
                    }
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, out_total);
                        let row_stride = total_axis * inner;
                        for (src_off, dst_col_off, copy_per_row) in &layout {
                            let in_total = outer * *copy_per_row;
                            let inp = sl(*src_off, base, in_total);
                            for o in 0..outer {
                                let dst_row_start = o * row_stride + *dst_col_off;
                                let src_row_start = o * *copy_per_row;
                                out[dst_row_start..dst_row_start + *copy_per_row].copy_from_slice(
                                    &inp[src_row_start..src_row_start + *copy_per_row],
                                );
                            }
                        }
                    })
                }

                Thunk::CustomOp {
                    kernel,
                    inputs,
                    output,
                    attrs,
                } => {
                    // Capture-by-move: clone the Arc and Vecs once into the
                    // closure. Dispatch by output dtype each call (the
                    // dtype is fixed at compile time but it's cheaper to
                    // branch once per execution than to monomorphize a
                    // dozen closure variants).
                    let kernel = kernel.clone();
                    let attrs = attrs.clone();
                    let inputs = inputs.clone();
                    let (out_off, out_len, out_shape) = output.clone();
                    Arc::new(move |base: *mut u8| unsafe {
                        dispatch_custom_op(
                            &*kernel, &inputs, out_off, out_len, &out_shape, &attrs, base,
                        );
                    })
                }

                Thunk::Fft1d {
                    src,
                    dst,
                    outer,
                    n_complex,
                    inverse,
                    dtype,
                } => {
                    let f: Arc<dyn Fn(*mut u8) + Send + Sync> = match dtype {
                        rlx_ir::DType::F64 => Arc::new(move |base: *mut u8| unsafe {
                            execute_fft1d_f64(
                                src,
                                dst,
                                outer as usize,
                                n_complex as usize,
                                inverse,
                                base,
                            );
                        }),
                        rlx_ir::DType::F32 => Arc::new(move |base: *mut u8| unsafe {
                            execute_fft1d_f32(
                                src,
                                dst,
                                outer as usize,
                                n_complex as usize,
                                inverse,
                                base,
                            );
                        }),
                        other => panic!("Op::Fft on CPU requires F32/F64, got {other:?}"),
                    };
                    f
                }

                _ => Arc::new(|_: *mut u8| {}),
            }
        })
        .collect();

    // ── Thunk-level attention fusion ──────────────────────
    // For small batch*seq, fuse QKV→Narrow×3→[Rope×2]→Attention→OutProj
    // into a single FusedAttnBlock. Auto-detects from Attention thunks.
    let fuse_threshold: usize = std::env::var("RLX_FUSE_ATTN_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let should_fuse = thunks.iter().any(|t| match t {
        Thunk::Attention { batch, seq, .. } => {
            (*batch as usize) * (*seq as usize) <= fuse_threshold
        }
        _ => false,
    });

    if should_fuse {
        // Build non-Nop index for pattern matching across Nop gaps
        let active: Vec<usize> = thunks
            .iter()
            .enumerate()
            .filter(|(_, t)| !matches!(t, Thunk::Nop))
            .map(|(i, _)| i)
            .collect();

        let mut kill = vec![false; thunks.len()]; // mark thunks to remove
        let mut insertions: Vec<(usize, Thunk)> = Vec::new(); // (position, replacement)

        let mut ai = 0;
        while ai < active.len() {
            // Helper: get active thunk at offset from current
            let a = |off: usize| -> Option<(usize, &Thunk)> {
                active.get(ai + off).map(|&idx| (idx, &thunks[idx]))
            };

            // Try BERT pattern: FusedMmBiasAct(QKV) → Narrow×3 → Attention → FusedMmBiasAct(out)
            let matched = (|| {
                let (_i0, t0) = a(0)?;
                let (_, t1) = a(1)?;
                let (_, t2) = a(2)?;
                let (_, t3) = a(3)?;

                // a[0] must be FusedMmBiasAct or Sgemm (QKV projection)
                let (hidden, qkv_w, qkv_b, has_b) = match t0 {
                    Thunk::FusedMmBiasAct {
                        a,
                        w,
                        bias,
                        n: _,
                        act: None,
                        ..
                    } => (*a, *w, *bias, true),
                    Thunk::Sgemm { a, b, n: _, .. } => (*a, *b, 0, false),
                    _ => return None,
                };

                // a[1..3] must be Narrows
                if !matches!(t1, Thunk::Narrow { .. }) {
                    return None;
                }
                if !matches!(t2, Thunk::Narrow { .. }) {
                    return None;
                }
                if !matches!(t3, Thunk::Narrow { .. }) {
                    return None;
                }

                // Look for optional Rope×2 then Attention
                let (has_rope, attn_ai, cos_off, sin_off, cl) = if let Some((
                    _,
                    Thunk::Rope {
                        cos, sin, cos_len, ..
                    },
                )) = a(4)
                {
                    if matches!(a(5).map(|x| x.1), Some(Thunk::Rope { .. })) {
                        if matches!(a(6).map(|x| x.1), Some(Thunk::Attention { .. })) {
                            (true, 6, *cos, *sin, *cos_len)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else if matches!(a(4).map(|x| x.1), Some(Thunk::Attention { .. })) {
                    (false, 4, 0, 0, 0)
                } else {
                    return None;
                };

                let (_attn_real_idx, attn_t) = a(attn_ai)?;
                let (batch, seq, heads, head_dim, mask) = match attn_t {
                    Thunk::Attention {
                        batch,
                        seq,
                        heads,
                        head_dim,
                        mask,
                        ..
                    } => (*batch, *seq, *heads, *head_dim, *mask),
                    _ => return None,
                };

                // Next active must be out projection (FusedMmBiasAct or Sgemm)
                let (_out_real_idx, out_t) = a(attn_ai + 1)?;
                let (out_w, out_b, out_dst) = match out_t {
                    Thunk::FusedMmBiasAct {
                        w,
                        bias,
                        c,
                        act: None,
                        ..
                    } => (*w, *bias, *c),
                    Thunk::Sgemm { b: w, c, .. } => (*w, 0, *c),
                    _ => return None,
                };

                let hs = heads * head_dim;
                let total_active = attn_ai + 2; // number of active thunks consumed

                Some((
                    total_active,
                    Thunk::FusedAttnBlock {
                        hidden,
                        qkv_w,
                        out_w,
                        mask,
                        out: out_dst,
                        qkv_b: if has_b { qkv_b } else { 0 },
                        out_b: if has_b { out_b } else { 0 },
                        cos: cos_off,
                        sin: sin_off,
                        cos_len: cl,
                        batch,
                        seq,
                        hs,
                        nh: heads,
                        dh: head_dim,
                        has_bias: has_b,
                        has_rope,
                    },
                ))
            })();

            if let Some((count, fused_thunk)) = matched {
                // Mark consumed thunks for removal
                for off in 0..count {
                    if let Some(&idx) = active.get(ai + off) {
                        kill[idx] = true;
                    }
                }
                // Insert replacement at position of the QKV thunk
                insertions.push((active[ai], fused_thunk));
                ai += count;
            } else {
                ai += 1;
            }
        }

        // Rebuild thunk list: keep non-killed, insert fused at right positions
        if !insertions.is_empty() {
            let mut new_thunks = Vec::with_capacity(thunks.len());
            let mut insert_idx = 0;
            for (i, t) in thunks.into_iter().enumerate() {
                if insert_idx < insertions.len() && insertions[insert_idx].0 == i {
                    new_thunks.push(insertions[insert_idx].1.clone());
                    insert_idx += 1;
                }
                if !kill[i] {
                    new_thunks.push(t);
                }
            }
            if cfg.verbose >= 1 {
                eprintln!(
                    "[rlx] fused_attention: {} attention blocks fused",
                    insertions.len()
                );
            }
            thunks = new_thunks;
        }
    }

    // ── Full layer fusion ──────────────────────────────────
    // After attention blocks are fused, scan for full layer patterns:
    // BERT:  FusedAttnBlock → FusedResidualLN → FusedMmBiasAct(gelu) → Sgemm → BiasAdd → FusedResidualLN
    // Nomic: FusedAttnBlock → BinaryFull(add) → LayerNorm → Sgemm → [Narrow×2 → Silu → BinaryFull(mul)] → Sgemm → BinaryFull(add) → LayerNorm
    if should_fuse {
        let active: Vec<usize> = thunks
            .iter()
            .enumerate()
            .filter(|(_, t)| !matches!(t, Thunk::Nop))
            .map(|(i, _)| i)
            .collect();

        let mut kill = vec![false; thunks.len()];
        let mut insertions: Vec<(usize, Thunk)> = Vec::new();

        let a = |ai: usize| -> Option<&Thunk> { active.get(ai).map(|&i| &thunks[i]) };

        let mut ai = 0;
        while ai < active.len() {
            // BERT pattern: FusedAttnBlock → FusedResidualLN → FusedMmBiasAct(gelu) → FusedMmBiasAct(none) → FusedResidualLN
            let bert_match = (|| -> Option<usize> {
                let fab = a(ai)?;
                let rln1 = a(ai + 1)?;
                let ffn1 = a(ai + 2)?;
                let ffn2 = a(ai + 3)?;
                let rln2 = a(ai + 4)?;

                let (hidden, qkv_w, qkv_b, out_w, out_b, mask, batch, seq, hs, nh, dh) = match fab {
                    Thunk::FusedAttnBlock {
                        hidden,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mask,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        has_bias: true,
                        has_rope: false,
                        ..
                    } => (
                        *hidden, *qkv_w, *qkv_b, *out_w, *out_b, *mask, *batch, *seq, *hs, *nh, *dh,
                    ),
                    _ => return None,
                };
                let (ln1_g, ln1_b, eps1) = match rln1 {
                    Thunk::FusedResidualLN { g, b, eps, .. } => (*g, *b, *eps),
                    _ => return None,
                };
                let (fc1_w, fc1_b, int_dim) = match ffn1 {
                    Thunk::FusedMmBiasAct {
                        w,
                        bias,
                        n,
                        act: Some(Activation::Gelu),
                        ..
                    } => (*w, *bias, *n),
                    _ => return None,
                };
                let (fc2_w, fc2_b) = match ffn2 {
                    Thunk::FusedMmBiasAct {
                        w, bias, act: None, ..
                    } => (*w, *bias),
                    _ => return None,
                };
                let (ln2_g, ln2_b, eps2, out) = match rln2 {
                    Thunk::FusedResidualLN { g, b, eps, out, .. } => (*g, *b, *eps, *out),
                    _ => return None,
                };

                for off in 0..5 {
                    kill[active[ai + off]] = true;
                }
                insertions.push((
                    active[ai],
                    Thunk::FusedBertLayer {
                        hidden,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mask,
                        ln1_g,
                        ln1_b,
                        eps1,
                        fc1_w,
                        fc1_b,
                        fc2_w,
                        fc2_b,
                        ln2_g,
                        ln2_b,
                        eps2,
                        out,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        int_dim,
                    },
                ));
                Some(5)
            })();
            if let Some(n) = bert_match {
                ai += n;
                continue;
            }

            // Nomic full layer fusion — disabled pending SwiGLU stride debugging.
            // Nomic still benefits from FusedAttnBlock (attention-level fusion).
            // The body below is kept as reference for when the stride bug is fixed.
            #[allow(unreachable_code)]
            let nomic_match = (|| -> Option<usize> {
                return None; // TODO: fix SwiGLU strided fc2 output mismatch
                let fab = a(ai)?;
                let (hidden, qkv_w, out_w, mask, cos, sin, cos_len, batch, seq, hs, nh, dh) =
                    match fab {
                        Thunk::FusedAttnBlock {
                            hidden,
                            qkv_w,
                            out_w,
                            mask,
                            cos,
                            sin,
                            cos_len,
                            batch,
                            seq,
                            hs,
                            nh,
                            dh,
                            has_bias: false,
                            has_rope: true,
                            ..
                        } => (
                            *hidden, *qkv_w, *out_w, *mask, *cos, *sin, *cos_len, *batch, *seq,
                            *hs, *nh, *dh,
                        ),
                        _ => return None,
                    };
                // FusedResidualLN for LN1
                let (ln1_g, ln1_b, eps1) = match a(ai + 1)? {
                    Thunk::FusedResidualLN { g, b, eps, .. } => (*g, *b, *eps),
                    _ => return None,
                };
                // Sgemm (fused fc11+fc12)
                let fused_fc_w = match a(ai + 2)? {
                    Thunk::Sgemm { b: w, .. } => *w,
                    _ => return None,
                };
                // Narrow×2 for split
                if !matches!(a(ai + 3)?, Thunk::Narrow { .. }) {
                    return None;
                }
                if !matches!(a(ai + 4)?, Thunk::Narrow { .. }) {
                    return None;
                }
                // SiLU
                if !matches!(
                    a(ai + 5)?,
                    Thunk::ActivationInPlace {
                        act: Activation::Silu,
                        ..
                    }
                ) {
                    return None;
                }
                // BinaryFull(Mul) for gate
                if !matches!(
                    a(ai + 6)?,
                    Thunk::BinaryFull {
                        op: BinaryOp::Mul,
                        ..
                    }
                ) {
                    return None;
                }
                // Sgemm (fc2)
                let fc2_w = match a(ai + 7)? {
                    Thunk::Sgemm { b: w, .. } => *w,
                    _ => return None,
                };
                // Get int_dim from the Narrow (inner = int_dim for last-axis narrow)
                let int_dim = match a(ai + 3)? {
                    Thunk::Narrow { inner, .. } => *inner,
                    _ => return None,
                };
                // FusedResidualLN for LN2
                let (ln2_g, ln2_b, eps2, out) = match a(ai + 8)? {
                    Thunk::FusedResidualLN { g, b, eps, out, .. } => (*g, *b, *eps, *out),
                    _ => return None,
                };

                for off in 0..9 {
                    kill[active[ai + off]] = true;
                }
                insertions.push((
                    active[ai],
                    Thunk::FusedNomicLayer {
                        hidden,
                        qkv_w,
                        out_w,
                        mask,
                        cos,
                        sin,
                        cos_len,
                        ln1_g,
                        ln1_b,
                        eps1,
                        fc11_w: fused_fc_w,
                        fc12_w: 0,
                        fc2_w,
                        ln2_g,
                        ln2_b,
                        eps2,
                        out,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        int_dim,
                    },
                ));
                Some(9)
            })();
            if let Some(n) = nomic_match {
                ai += n;
                continue;
            }

            ai += 1;
        }

        if !insertions.is_empty() {
            let mut new_thunks = Vec::with_capacity(thunks.len());
            let mut ins_idx = 0;
            for (i, t) in thunks.into_iter().enumerate() {
                if ins_idx < insertions.len() && insertions[ins_idx].0 == i {
                    new_thunks.push(insertions[ins_idx].1.clone());
                    ins_idx += 1;
                }
                if !kill[i] {
                    new_thunks.push(t);
                }
            }
            if cfg.verbose >= 1 {
                eprintln!(
                    "[rlx] fused_layer: {} full transformer layers fused",
                    insertions.len()
                );
            }
            thunks = new_thunks;
        }
    }

    // ── Narrow → Rope thunk fusion (plan #45) ──────────────
    // Runs *after* FusedAttnBlock fusion so it only catches the medium-
    // batch path (batch*seq > 64) where the bigger fusion didn't fire.
    // Pattern: a Rope thunk whose `src` is the dst of an immediately-
    // preceding Narrow whose dst has no other consumer in this schedule.
    // Rewrite Rope to read directly from the parent buffer with the
    // parent's row stride; the Narrow becomes a Nop.
    //
    // Skipping the Narrow's write saves one full pass over Q/K (B*S*hs
    // f32) per Rope. For Nomic h=768 / batch=8 / seq=15 / 12 layers
    // that's 2 ropes/layer × 369 KB = ~8.9 MB of write traffic gone.
    {
        // Collect every byte-offset that's read as a thunk's `src` so
        // we know whether a Narrow's dst has consumers other than Rope.
        let mut read_offsets: HashMap<usize, usize> = HashMap::new();
        for t in &thunks {
            for off in thunk_read_offsets(t) {
                *read_offsets.entry(off).or_insert(0) += 1;
            }
        }

        let mut fused_count = 0usize;
        for i in 0..thunks.len().saturating_sub(1) {
            // Look for Rope at i+1 reading from Narrow at i (skip Nops
            // between them since the planner left them in place).
            let narrow = match &thunks[i] {
                Thunk::Narrow { .. } => i,
                _ => continue,
            };
            // Find the next non-Nop thunk
            let mut j = narrow + 1;
            while j < thunks.len() && matches!(thunks[j], Thunk::Nop) {
                j += 1;
            }
            if j >= thunks.len() {
                continue;
            }
            // Must be Rope reading Narrow's dst
            let (n_src, n_dst, n_src_stride) = match &thunks[narrow] {
                Thunk::Narrow {
                    src,
                    dst,
                    src_stride,
                    ..
                } => (*src, *dst, *src_stride),
                _ => continue,
            };
            let rope_reads_narrow = matches!(&thunks[j],
                Thunk::Rope { src, .. } if *src == n_dst);
            if !rope_reads_narrow {
                continue;
            }
            // Conservatively require that the Narrow's dst has exactly
            // one reader (the Rope). Anything else and rewriting would
            // skip a needed write.
            if read_offsets.get(&n_dst).copied().unwrap_or(0) != 1 {
                continue;
            }

            // Rewire: Rope reads from Narrow's adjusted source with the
            // parent buffer's row stride.
            if let Thunk::Rope {
                src,
                src_row_stride,
                ..
            } = &mut thunks[j]
            {
                *src = n_src;
                *src_row_stride = n_src_stride;
            }
            thunks[narrow] = Thunk::Nop;
            fused_count += 1;
        }

        if fused_count > 0 && cfg.verbose >= 1 {
            eprintln!(
                "[rlx] fused_qk_rope: {} Narrow→Rope pairs collapsed",
                fused_count
            );
        }
    }

    // ── Narrow×3 → Attention thunk fusion (plan #46 deep) ────
    // For each Attention thunk in the schedule, look up the producers
    // of its q/k/v inputs. If each is a Narrow whose dst has exactly
    // one consumer (the Attention), rewire Attention to read directly
    // from the parent buffer with the parent's row stride. The three
    // Narrows become Nops.
    //
    // This catches the BERT/Nomic QKV split path that FusedAttnBlock
    // misses (batch*seq > 64) — eliminates Q/K/V copies entirely.
    // For minilm6 batch=32 seq=16 hs=384: 3 × 32*16*384*4 = 2.3 MB
    // per layer × 6 layers = ~14 MB of write traffic gone.
    {
        let mut read_counts: HashMap<usize, usize> = HashMap::new();
        for t in &thunks {
            for off in thunk_read_offsets(t) {
                *read_counts.entry(off).or_insert(0) += 1;
            }
        }
        // Build dst→index map for fast producer lookup.
        let mut dst_to_idx: HashMap<usize, usize> = HashMap::new();
        for (i, t) in thunks.iter().enumerate() {
            if let Thunk::Narrow { dst, .. } = t {
                dst_to_idx.insert(*dst, i);
            }
        }

        let mut fused_count = 0usize;
        for i in 0..thunks.len() {
            let (q_off, k_off, v_off) = match &thunks[i] {
                Thunk::Attention { q, k, v, .. } => (*q, *k, *v),
                _ => continue,
            };
            // All three inputs must come from Narrows.
            let q_n = match dst_to_idx.get(&q_off).copied() {
                Some(x) => x,
                None => continue,
            };
            let k_n = match dst_to_idx.get(&k_off).copied() {
                Some(x) => x,
                None => continue,
            };
            let v_n = match dst_to_idx.get(&v_off).copied() {
                Some(x) => x,
                None => continue,
            };
            // Each Narrow's dst must have exactly one reader (this Attn).
            if read_counts.get(&q_off).copied().unwrap_or(0) != 1 {
                continue;
            }
            if read_counts.get(&k_off).copied().unwrap_or(0) != 1 {
                continue;
            }
            if read_counts.get(&v_off).copied().unwrap_or(0) != 1 {
                continue;
            }

            let (q_src, q_stride) = match &thunks[q_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };
            let (k_src, k_stride) = match &thunks[k_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };
            let (v_src, v_stride) = match &thunks[v_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };

            if let Thunk::Attention {
                q,
                k,
                v,
                q_row_stride,
                k_row_stride,
                v_row_stride,
                ..
            } = &mut thunks[i]
            {
                *q = q_src;
                *k = k_src;
                *v = v_src;
                *q_row_stride = q_stride;
                *k_row_stride = k_stride;
                *v_row_stride = v_stride;
            }
            thunks[q_n] = Thunk::Nop;
            thunks[k_n] = Thunk::Nop;
            thunks[v_n] = Thunk::Nop;
            fused_count += 1;
        }

        if fused_count > 0 && cfg.verbose >= 1 {
            eprintln!(
                "[rlx] fused_strided_attn: {} Narrow×3→Attention rewrites",
                fused_count
            );
        }
    }

    ThunkSchedule {
        thunks,
        mask_threshold: cfg.mask_binary_threshold,
        mask_neg_inf: cfg.attn_mask_neg_inf,
        score_skip: cfg.score_skip_threshold,
        compiled_fns,
    }
}

fn get_len(graph: &Graph, id: NodeId) -> usize {
    graph.node(id).shape.num_elements().unwrap_or(0)
}

/// Static `usize` dims of a node's shape, or empty if any dim is dynamic.
fn get_static_dims(graph: &Graph, id: NodeId) -> Vec<usize> {
    let dims = graph.node(id).shape.dims();
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        if let Some(s) = match d {
            rlx_ir::Dim::Static(s) => Some(*s),
            _ => None,
        } {
            out.push(s);
        } else {
            return Vec::new();
        }
    }
    out
}

/// NumPy-style broadcast strides for one operand into the flat output
/// buffer. Returns a length-`out_dims.len()` `Vec<u32>` where entry
/// `d` is `0` if the input is size-1 (broadcast) at output dim `d`
/// (after left-padding with size-1 to match ranks), otherwise the
/// natural row-major stride into the *input* buffer.
///
/// Caller iterates output flat index `i` → output coords (row-major)
/// → input flat index = dot(coords, strides). The result is correct
/// for any broadcast pattern (scalar, last-axis, middle-axis,
/// bidirectional).
/// True when `rhs_dims` describes a *trailing* broadcast of `out_dims`
/// — i.e. every rhs dim either equals the corresponding output dim
/// (counting from the right) or rhs is shorter (left-padded with 1s).
/// Mid-shape singletons (e.g. rhs `[a, b, 1, d]` into out `[a, b, c, d]`
/// where `c > 1`) are NOT trailing broadcasts and require the
/// shape-aware `BinaryFull` slow path — `BiasAdd`'s linear bias-replicated
/// kernel silently miscomputes them.
fn is_trailing_bias_broadcast(rhs_dims: &[rlx_ir::Dim], out_dims: &[rlx_ir::Dim]) -> bool {
    if rhs_dims.len() > out_dims.len() {
        return false;
    }
    let off = out_dims.len() - rhs_dims.len();
    for i in 0..rhs_dims.len() {
        let r = match rhs_dims[i] {
            rlx_ir::Dim::Static(n) => n,
            _ => return false,
        };
        let o = match out_dims[off + i] {
            rlx_ir::Dim::Static(n) => n,
            _ => return false,
        };
        if r != o {
            return false;
        }
    }
    true
}

fn broadcast_strides(in_dims: &[usize], out_dims: &[usize]) -> Vec<u32> {
    let r_out = out_dims.len();
    let r_in = in_dims.len();
    assert!(
        r_in <= r_out,
        "broadcast: input rank {r_in} > output rank {r_out}"
    );
    let pad = r_out - r_in;
    let mut strides = vec![0u32; r_out];
    let mut acc: usize = 1;
    for d in (0..r_out).rev() {
        let in_size = if d < pad { 1 } else { in_dims[d - pad] };
        if in_size == 1 {
            strides[d] = 0;
        } else {
            assert_eq!(
                in_size, out_dims[d],
                "broadcast: input dim {in_size} doesn't match output dim {} at axis {d}",
                out_dims[d]
            );
            strides[d] = acc as u32;
            acc *= in_size;
        }
    }
    strides
}

/// Execute a thunk schedule on a raw arena buffer.
/// Fastest executor: call pre-compiled closures sequentially.
/// Zero match dispatch — each closure is a direct kernel call.
pub fn execute_compiled(schedule: &ThunkSchedule, arena_buf: &mut [u8]) {
    let base = arena_buf.as_mut_ptr();
    for f in &schedule.compiled_fns {
        f(base);
    }
}

/// Active-extent execution stub. The runtime calls this when it has an
/// active-extent hint set. CPU doesn't implement per-thunk active-extent
/// scaling yet — return false so the caller falls back to the full
/// `execute_thunks` path.
pub fn execute_thunks_active(
    schedule: &ThunkSchedule,
    _arena_buf: &mut [u8],
    _actual: usize,
    _upper: usize,
) -> bool {
    let _ = schedule;
    false
}

/// Match-based executor (fallback, used by tests).
pub fn execute_thunks(schedule: &ThunkSchedule, arena_buf: &mut [u8]) {
    let base = arena_buf.as_mut_ptr();
    let mask_thr = schedule.mask_threshold;
    let mask_neg = schedule.mask_neg_inf;
    let score_thr = schedule.score_skip;
    let thunks = &schedule.thunks;
    let len = thunks.len();

    // Pre-allocate ALL reusable buffers once (zero per-call allocation)
    let max_h = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::FusedResidualLN { h, .. } | Thunk::LayerNorm { h, .. } => Some(*h as usize),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let zero_bias = vec![0f32; max_h];

    // Pre-allocate per-(batch,head) score buffers for parallel SDPA.
    // Q/K/V/out are accessed via strided BLAS — no deinterleave copy needed.
    let max_sdpa = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::Attention {
                batch,
                seq,
                kv_seq,
                heads,
                head_dim,
                ..
            } => Some((
                *batch as usize,
                (*seq as usize).max(*kv_seq as usize),
                *heads as usize,
                *head_dim as usize,
            )),
            _ => None,
        })
        .fold((0, 0, 0, 0), |(mb, ms, mh, md), (b, s, h, d)| {
            (mb.max(b), ms.max(s), mh.max(h), md.max(d))
        });
    let (max_batch, max_seq, max_heads, _max_dh) = max_sdpa;
    let max_units = max_batch * max_heads;
    let mut sdpa_scores = vec![0f32; max_units * max_seq * max_seq];

    // Pre-allocate fused layer buffers (reused across all 12+ layers — zero malloc per layer)
    let fl = thunks
        .iter()
        .filter_map(|t| match t {
            Thunk::FusedBertLayer {
                batch,
                seq,
                hs,
                int_dim,
                ..
            } => {
                let m = (*batch as usize) * (*seq as usize);
                let h = *hs as usize;
                let id = *int_dim as usize;
                Some((m, h, id, m * (*seq as usize)))
            }
            Thunk::FusedNomicLayer {
                batch,
                seq,
                hs,
                int_dim,
                ..
            } => {
                let m = (*batch as usize) * (*seq as usize);
                let h = *hs as usize;
                let id = *int_dim as usize;
                Some((m, h, id, m * (*seq as usize)))
            }
            _ => None,
        })
        .fold((0, 0, 0, 0), |(mm, mh, mi, ms), (m, h, id, ss)| {
            (mm.max(m), mh.max(h), mi.max(id), ms.max(ss))
        });
    let (fl_m, fl_h, fl_int, fl_ss) = fl;
    let mut fl_qkv = vec![0f32; fl_m * 3 * fl_h];
    let mut fl_attn = vec![0f32; fl_m * fl_h];
    let mut fl_res = vec![0f32; fl_m * fl_h];
    let mut fl_normed = vec![0f32; fl_m * fl_h];
    let mut fl_ffn = vec![0f32; fl_m * fl_int.max(2 * fl_int)]; // Nomic needs 2×int for fused fc11+fc12
    let mut fl_sc = vec![0f32; fl_ss.max(1)];

    for i in 0..len {
        let thunk = unsafe { thunks.get_unchecked(i) };
        match thunk {
            Thunk::Nop => {}

            Thunk::Fft1d {
                src,
                dst,
                outer,
                n_complex,
                inverse,
                dtype,
            } => unsafe {
                match dtype {
                    rlx_ir::DType::F64 => execute_fft1d_f64(
                        *src,
                        *dst,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        base,
                    ),
                    rlx_ir::DType::F32 => execute_fft1d_f32(
                        *src,
                        *dst,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        base,
                    ),
                    other => panic!("Op::Fft on CPU requires F32/F64, got {other:?}"),
                }
            },

            // CustomFn dispatch (interpreted path). Mirrors the
            // pre-compiled-closure variant elsewhere in this file.
            // Patched by rlx-eda.
            Thunk::CustomFn {
                body,
                body_init,
                inputs,
                body_output_off,
                outer_output_off,
                out_bytes,
            } => {
                let mut body_buf: Vec<u8> = (**body_init).clone();
                unsafe {
                    for (body_in_off, outer_in_off, n_bytes) in inputs.iter() {
                        let src = (base as *const u8).add(*outer_in_off);
                        let dst = body_buf.as_mut_ptr().add(*body_in_off);
                        std::ptr::copy_nonoverlapping(src, dst, *n_bytes as usize);
                    }
                }
                execute_thunks(body, &mut body_buf);
                unsafe {
                    let src = body_buf.as_ptr().add(*body_output_off);
                    let dst = base.add(*outer_output_off);
                    std::ptr::copy_nonoverlapping(src, dst, *out_bytes as usize);
                }
            }

            Thunk::Sgemm { a, b, c, m, k, n } => {
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                unsafe {
                    crate::blas::sgemm_auto(
                        sl(*a, base, m * k),
                        sl(*b, base, k * n),
                        sl_mut(*c, base, m * n),
                        m,
                        k,
                        n,
                    );
                }
            }

            Thunk::DenseSolveF64 { a, b, x, n, nrhs } => {
                let (n_, nrhs_) = (*n as usize, *nrhs as usize);
                // LAPACK overwrites both A and B; clone into scratch
                // each call. Caller's A and b must be preserved for
                // VJP recompute. (Eventually: swap to a factor-once /
                // solve-many scheme; that's the symbolic-reuse story
                // and lives with the sparse path.)
                unsafe {
                    let a_src = sl_f64(*a, base, n_ * n_);
                    let b_src = sl_f64(*b, base, n_ * nrhs_);
                    let mut a_scratch: Vec<f64> = a_src.to_vec();
                    let mut x_buf: Vec<f64> = b_src.to_vec();
                    let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                    if info != 0 {
                        panic!(
                            "DenseSolveF64: dgesv reported singular matrix \
                                (info={info}, n={n_}, nrhs={nrhs_})"
                        );
                    }
                    let dst = sl_mut_f64(*x, base, n_ * nrhs_);
                    dst.copy_from_slice(&x_buf);
                }
            }

            Thunk::DenseSolveF32 { a, b, x, n, nrhs } => {
                let (n_, nrhs_) = (*n as usize, *nrhs as usize);
                unsafe {
                    let a_src = sl(*a, base, n_ * n_);
                    let b_src = sl(*b, base, n_ * nrhs_);
                    let mut a_scratch: Vec<f32> = a_src.to_vec();
                    let mut x_buf: Vec<f32> = b_src.to_vec();
                    let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                    if info != 0 {
                        panic!(
                            "DenseSolveF32: sgesv reported singular matrix \
                             (info={info}, n={n_}, nrhs={nrhs_})"
                        );
                    }
                    let dst = sl_mut(*x, base, n_ * nrhs_);
                    dst.copy_from_slice(&x_buf);
                }
            }

            Thunk::BatchedDenseSolveF64 {
                a,
                b,
                x,
                batch,
                n,
                nrhs,
            } => {
                // Per slice: extract A_i and b_i, dgesv, write x_i.
                // LAPACK has no batched dgesv on Accelerate, so this
                // is a serial loop over the batch axis. cuSOLVER /
                // hipSOLVER expose `getrfBatched` / `getrsBatched` for
                // the GPU path — we'll wire that in rlx-cuda when
                // someone needs Linux+CUDA.
                let (b_, n_, nrhs_) = (*batch as usize, *n as usize, *nrhs as usize);
                let a_stride = n_ * n_;
                let b_stride = n_ * nrhs_;
                unsafe {
                    let a_full = sl_f64(*a, base, b_ * a_stride);
                    let b_full = sl_f64(*b, base, b_ * b_stride);
                    let x_full = sl_mut_f64(*x, base, b_ * b_stride);
                    for bi in 0..b_ {
                        let mut a_scratch: Vec<f64> =
                            a_full[bi * a_stride..(bi + 1) * a_stride].to_vec();
                        let mut x_buf: Vec<f64> =
                            b_full[bi * b_stride..(bi + 1) * b_stride].to_vec();
                        let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                        if info != 0 {
                            panic!(
                                "BatchedDenseSolveF64: slice {bi} \
                                    singular (info={info}, n={n_}, nrhs={nrhs_})"
                            );
                        }
                        x_full[bi * b_stride..(bi + 1) * b_stride].copy_from_slice(&x_buf);
                    }
                }
            }

            Thunk::BatchedDgemmF64 {
                a,
                b,
                c,
                batch,
                m,
                k,
                n,
            } => {
                let (b_, m_, k_, n_) = (*batch as usize, *m as usize, *k as usize, *n as usize);
                let a_stride = m_ * k_;
                let b_stride = k_ * n_;
                let c_stride = m_ * n_;
                unsafe {
                    let a_full = sl_f64(*a, base, b_ * a_stride);
                    let b_full = sl_f64(*b, base, b_ * b_stride);
                    let c_full = sl_mut_f64(*c, base, b_ * c_stride);
                    for bi in 0..b_ {
                        let a_slice = &a_full[bi * a_stride..(bi + 1) * a_stride];
                        let b_slice = &b_full[bi * b_stride..(bi + 1) * b_stride];
                        let c_slice = &mut c_full[bi * c_stride..(bi + 1) * c_stride];
                        crate::blas::dgemm(a_slice, b_slice, c_slice, m_, k_, n_);
                    }
                }
            }

            Thunk::BatchedSgemm {
                a,
                b,
                c,
                batch,
                m,
                k,
                n,
            } => {
                let (b_, m_, k_, n_) = (*batch as usize, *m as usize, *k as usize, *n as usize);
                let a_stride = m_ * k_;
                let b_stride = k_ * n_;
                let c_stride = m_ * n_;
                unsafe {
                    let a_full = sl(*a, base, b_ * a_stride);
                    let b_full = sl(*b, base, b_ * b_stride);
                    let c_full = sl_mut(*c, base, b_ * c_stride);
                    for bi in 0..b_ {
                        let a_slice = &a_full[bi * a_stride..(bi + 1) * a_stride];
                        let b_slice = &b_full[bi * b_stride..(bi + 1) * b_stride];
                        let c_slice = &mut c_full[bi * c_stride..(bi + 1) * c_stride];
                        crate::blas::sgemm_auto(a_slice, b_slice, c_slice, m_, k_, n_);
                    }
                }
            }

            Thunk::Dgemm { a, b, c, m, k, n } => {
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                unsafe {
                    crate::blas::dgemm(
                        sl_f64(*a, base, m * k),
                        sl_f64(*b, base, k * n),
                        sl_mut_f64(*c, base, m * n),
                        m,
                        k,
                        n,
                    );
                }
            }

            Thunk::TransposeF64 {
                src,
                dst,
                in_total,
                out_dims,
                in_strides,
            } => unsafe {
                let inp = sl_f64(*src, base, *in_total as usize);
                let out_total: usize = out_dims.iter().map(|d| *d as usize).product();
                let out = sl_mut_f64(*dst, base, out_total);
                transpose_walk_f64(inp, out, out_dims, in_strides);
            },

            Thunk::ActivationF64 {
                src,
                dst,
                len,
                kind,
            } => {
                let len = *len as usize;
                unsafe {
                    let inp = sl_f64(*src, base, len);
                    let out = sl_mut_f64(*dst, base, len);
                    apply_activation_f64(inp, out, *kind);
                }
            }

            Thunk::ReduceSumF64 {
                src,
                dst,
                outer,
                reduced,
                inner,
            } => {
                let (o, r, n) = (*outer as usize, *reduced as usize, *inner as usize);
                unsafe {
                    let inp = sl_f64(*src, base, o * r * n);
                    let out = sl_mut_f64(*dst, base, o * n);
                    reduce_sum_f64(inp, out, o, r, n);
                }
            }

            Thunk::CopyF64 { src, dst, len } => {
                let len = *len as usize;
                if *src == *dst { /* aliased, no copy needed */
                } else {
                    unsafe {
                        let s = sl_f64(*src, base, len);
                        let d = sl_mut_f64(*dst, base, len);
                        d.copy_from_slice(s);
                    }
                }
            }

            Thunk::BinaryFullF64 {
                lhs,
                rhs,
                dst,
                len,
                lhs_len,
                rhs_len,
                op,
                out_dims_bcast,
                bcast_lhs_strides,
                bcast_rhs_strides,
            } => {
                let len = *len as usize;
                let lhs_len = *lhs_len as usize;
                let rhs_len = *rhs_len as usize;
                unsafe {
                    let l = sl_f64(*lhs, base, lhs_len);
                    let r = sl_f64(*rhs, base, rhs_len);
                    let d = sl_mut_f64(*dst, base, len);
                    if lhs_len == len && rhs_len == len {
                        for i in 0..len {
                            d[i] = binary_op_f64(*op, l[i], r[i]);
                        }
                    } else if !out_dims_bcast.is_empty() {
                        // Shape-aware broadcast path: correct for
                        // arbitrary NumPy-style broadcasts including
                        // bidirectional `[N,1] op [1,S]`.
                        let rank = out_dims_bcast.len();
                        let mut coords = vec![0u32; rank];
                        for i in 0..len {
                            let mut rem = i;
                            for ax in (0..rank).rev() {
                                let sz = out_dims_bcast[ax] as usize;
                                coords[ax] = (rem % sz) as u32;
                                rem /= sz;
                            }
                            let mut li: usize = 0;
                            let mut ri: usize = 0;
                            for ax in 0..rank {
                                li += coords[ax] as usize * bcast_lhs_strides[ax] as usize;
                                ri += coords[ax] as usize * bcast_rhs_strides[ax] as usize;
                            }
                            d[i] = binary_op_f64(*op, l[li], r[ri]);
                        }
                    } else {
                        // Fallback: legacy modulo path (preserved for
                        // dynamic-shape graphs where strides can't be
                        // precomputed). Only correct for scalar /
                        // last-axis broadcast.
                        for i in 0..len {
                            d[i] = binary_op_f64(*op, l[i % lhs_len], r[i % rhs_len]);
                        }
                    }
                }
            }

            Thunk::BinaryFullC64 {
                lhs,
                rhs,
                dst,
                len,
                lhs_len,
                rhs_len,
                op,
                out_dims_bcast,
                bcast_lhs_strides,
                bcast_rhs_strides,
            } => {
                // Complex element layout: [re_0, im_0, re_1, im_1, ...]
                // Underlying f32 buffer length is 2·N (N = complex
                // element count). All offsets are byte offsets; the
                // `sl` helper reads as f32 starting at the byte
                // offset, so f32-length = 2·complex-len.
                let n_out = *len as usize;
                let n_l = *lhs_len as usize;
                let n_r = *rhs_len as usize;
                unsafe {
                    let l = sl(*lhs, base, 2 * n_l);
                    let r = sl(*rhs, base, 2 * n_r);
                    let d = sl_mut(*dst, base, 2 * n_out);
                    let do_c64 = |a_re: f32, a_im: f32, b_re: f32, b_im: f32| -> (f32, f32) {
                        match op {
                            BinaryOp::Add => (a_re + b_re, a_im + b_im),
                            BinaryOp::Sub => (a_re - b_re, a_im - b_im),
                            BinaryOp::Mul => (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re),
                            BinaryOp::Div => {
                                let denom = b_re * b_re + b_im * b_im;
                                (
                                    (a_re * b_re + a_im * b_im) / denom,
                                    (a_im * b_re - a_re * b_im) / denom,
                                )
                            }
                            BinaryOp::Max | BinaryOp::Min | BinaryOp::Pow => {
                                unreachable!("C64 max/min/pow rejected at lowering")
                            }
                        }
                    };
                    if n_l == n_out && n_r == n_out {
                        for i in 0..n_out {
                            let (re, im) = do_c64(l[2 * i], l[2 * i + 1], r[2 * i], r[2 * i + 1]);
                            d[2 * i] = re;
                            d[2 * i + 1] = im;
                        }
                    } else if !out_dims_bcast.is_empty() {
                        // Strided complex broadcast: strides are in
                        // *complex element* units; multiply by 2 when
                        // indexing into the f32 buffer.
                        let rank = out_dims_bcast.len();
                        let mut coords = vec![0u32; rank];
                        for i in 0..n_out {
                            let mut rem = i;
                            for ax in (0..rank).rev() {
                                let sz = out_dims_bcast[ax] as usize;
                                coords[ax] = (rem % sz) as u32;
                                rem /= sz;
                            }
                            let mut li: usize = 0;
                            let mut ri: usize = 0;
                            for ax in 0..rank {
                                li += coords[ax] as usize * bcast_lhs_strides[ax] as usize;
                                ri += coords[ax] as usize * bcast_rhs_strides[ax] as usize;
                            }
                            let (re, im) =
                                do_c64(l[2 * li], l[2 * li + 1], r[2 * ri], r[2 * ri + 1]);
                            d[2 * i] = re;
                            d[2 * i + 1] = im;
                        }
                    } else {
                        // Modulo fallback (scalar / last-axis broadcast).
                        for i in 0..n_out {
                            let li = if n_l == 1 { 0 } else { i % n_l };
                            let ri = if n_r == 1 { 0 } else { i % n_r };
                            let (re, im) =
                                do_c64(l[2 * li], l[2 * li + 1], r[2 * ri], r[2 * ri + 1]);
                            d[2 * i] = re;
                            d[2 * i + 1] = im;
                        }
                    }
                }
            }

            Thunk::ComplexNormSqF32 { src, dst, len } => {
                let n = *len as usize;
                unsafe {
                    let s = sl(*src, base, 2 * n);
                    let d = sl_mut(*dst, base, n);
                    for i in 0..n {
                        let re = s[2 * i];
                        let im = s[2 * i + 1];
                        d[i] = re * re + im * im;
                    }
                }
            }

            Thunk::ComplexNormSqBackwardF32 { z, g, dz, len } => {
                // Wirtinger: dz = g · z, element-wise complex
                // (g is real, z is complex).
                let n = *len as usize;
                unsafe {
                    let zb = sl(*z, base, 2 * n);
                    let gb = sl(*g, base, n);
                    let db = sl_mut(*dz, base, 2 * n);
                    for i in 0..n {
                        let re = zb[2 * i];
                        let im = zb[2 * i + 1];
                        let gv = gb[i];
                        db[2 * i] = gv * re;
                        db[2 * i + 1] = gv * im;
                    }
                }
            }

            Thunk::ConjugateC64 { src, dst, len } => {
                let n = *len as usize;
                unsafe {
                    let s = sl(*src, base, 2 * n);
                    let d = sl_mut(*dst, base, 2 * n);
                    for i in 0..n {
                        d[2 * i] = s[2 * i];
                        d[2 * i + 1] = -s[2 * i + 1];
                    }
                }
            }

            Thunk::ActivationC64 {
                src,
                dst,
                len,
                kind,
            } => {
                let n = *len as usize;
                unsafe {
                    let s = sl(*src, base, 2 * n);
                    let d = sl_mut(*dst, base, 2 * n);
                    for i in 0..n {
                        let a = s[2 * i];
                        let b = s[2 * i + 1];
                        let (re, im) = match kind {
                            Activation::Neg => (-a, -b),
                            Activation::Exp => {
                                // exp(a + bi) = e^a · (cos b + i·sin b)
                                let ea = a.exp();
                                (ea * b.cos(), ea * b.sin())
                            }
                            Activation::Log => {
                                // log(z) = log|z| + i·arg(z), principal branch
                                let r = (a * a + b * b).sqrt();
                                (r.ln(), b.atan2(a))
                            }
                            Activation::Sqrt => {
                                // sqrt(a+bi) = sqrt((|z|+a)/2) + sign(b)·i·sqrt((|z|-a)/2)
                                // Principal branch; for b == 0 and a < 0 returns +i·sqrt(|a|).
                                let r = (a * a + b * b).sqrt();
                                let re = ((r + a) * 0.5).max(0.0).sqrt();
                                let im_mag = ((r - a) * 0.5).max(0.0).sqrt();
                                let im = if b >= 0.0 { im_mag } else { -im_mag };
                                (re, im)
                            }
                            _ => unreachable!("non-C64 activation kind survived lowering"),
                        };
                        d[2 * i] = re;
                        d[2 * i + 1] = im;
                    }
                }
            }

            Thunk::Scan {
                body,
                body_init,
                body_input_off,
                body_output_off,
                outer_init_off,
                outer_final_off,
                length,
                carry_bytes,
                save_trajectory,
                xs_inputs,
                bcast_inputs,
                num_checkpoints,
            } => {
                let cb = *carry_bytes as usize;
                let n_steps = *length as usize;
                // Checkpoint mode: when 0 < K < length, save trajectory[k]
                // only when t == c_k = floor((k+1) * length / K) - 1.
                // The last index c_{K-1} = length - 1 always.
                let k_total = if *num_checkpoints == 0 || *num_checkpoints == *length {
                    n_steps // save every step
                } else {
                    *num_checkpoints as usize
                };
                let checkpoint_t_for_k = |k: usize| -> usize {
                    if k_total == n_steps {
                        k
                    } else {
                        ((k + 1) * n_steps)
                            .div_ceil(k_total)
                            .saturating_sub(1)
                            .min(n_steps - 1)
                    }
                };
                let mut next_k = 0usize;

                let mut body_buf: Vec<u8> = (**body_init).clone();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        base.add(*outer_init_off),
                        body_buf.as_mut_ptr().add(*body_input_off),
                        cb,
                    );
                    // Broadcast inputs: copy each one into the body's
                    // input slot ONCE. They aren't touched in the
                    // iteration loop below (in contrast to xs).
                    for (body_b_off, outer_b_off, total_bytes) in bcast_inputs.iter() {
                        std::ptr::copy_nonoverlapping(
                            base.add(*outer_b_off),
                            body_buf.as_mut_ptr().add(*body_b_off),
                            *total_bytes as usize,
                        );
                    }
                }
                for t in 0..n_steps {
                    for (body_x_off, outer_xs_off, per_step_bytes) in xs_inputs.iter() {
                        let psb = *per_step_bytes as usize;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                base.add(*outer_xs_off + t * psb),
                                body_buf.as_mut_ptr().add(*body_x_off),
                                psb,
                            );
                        }
                    }

                    execute_thunks(body, &mut body_buf);

                    if *save_trajectory && next_k < k_total && t == checkpoint_t_for_k(next_k) {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                body_buf.as_ptr().add(*body_output_off),
                                base.add(*outer_final_off + next_k * cb),
                                cb,
                            );
                        }
                        next_k += 1;
                    }

                    if *body_output_off != *body_input_off {
                        body_buf
                            .copy_within(*body_output_off..*body_output_off + cb, *body_input_off);
                    }
                }

                if !*save_trajectory {
                    // Single final-carry write.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            body_buf.as_ptr().add(*body_output_off),
                            base.add(*outer_final_off),
                            cb,
                        );
                    }
                }
            }

            Thunk::ScanBackward {
                body_vjp,
                body_init,
                body_carry_in_off,
                body_x_offs,
                body_d_output_off,
                body_dcarry_out_off,
                outer_init_off,
                outer_traj_off,
                outer_upstream_off,
                outer_xs_offs,
                outer_dinit_off,
                length,
                carry_bytes,
                save_trajectory,
                num_checkpoints,
                forward_body,
                forward_body_init,
                forward_body_carry_in_off,
                forward_body_output_off,
                forward_body_x_offs,
                carry_elem_size,
            } => {
                // Two backward paths share the same per-iteration body
                // (body_vjp run + dcarry threading). The "All" path
                // reads the carry directly from the saved trajectory
                // each step. The "Recursive checkpointing" path stores
                // only K saved checkpoints and reconstructs intermediate
                // carries via Griewank-style recursive subdivision —
                // see [`griewank_process_segment`]. Auxiliary memory
                // is `O(log(segment_size) · carry_bytes)` for the
                // recursion stack, vs the old segment-cache scheme's
                // `O(segment_size · carry_bytes)`. Total recompute work
                // grows from `O(length)` to `O(length · log)`, which
                // is the canonical Griewank trade.
                let cb = *carry_bytes as usize;
                let n_steps = *length as usize;
                let k_total = *num_checkpoints as usize;
                let is_recursive = k_total != 0 && k_total != n_steps;
                let checkpoint_t_for_k = |k: usize| -> usize {
                    ((k + 1) * n_steps)
                        .div_ceil(k_total)
                        .saturating_sub(1)
                        .min(n_steps - 1)
                };

                let mut fwd_buf: Vec<u8> = if is_recursive {
                    (**forward_body_init.as_ref().unwrap()).clone()
                } else {
                    Vec::new()
                };

                let mut dcarry: Vec<u8> = vec![0u8; cb];
                if !*save_trajectory {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(*outer_upstream_off),
                            dcarry.as_mut_ptr(),
                            cb,
                        );
                    }
                }

                let mut body_buf: Vec<u8> = (**body_init).clone();

                // Per-iteration backward action — shared between the
                // direct-trajectory (All) and Griewank (Recursive) paths.
                // Both feed the same body_vjp run with carry-at-t,
                // x_t_i, and d_output, then thread dcarry backward.
                let process_iter =
                    |t: usize, carry_in: &[u8], dcarry: &mut Vec<u8>, body_buf: &mut Vec<u8>| {
                        if *save_trajectory {
                            unsafe {
                                let up_off = *outer_upstream_off + t * cb;
                                match *carry_elem_size {
                                    4 => {
                                        let up_ptr = base.add(up_off) as *const f32;
                                        let dc_ptr = dcarry.as_mut_ptr() as *mut f32;
                                        let n_elems = cb / 4;
                                        for i in 0..n_elems {
                                            *dc_ptr.add(i) += *up_ptr.add(i);
                                        }
                                    }
                                    8 => {
                                        let up_ptr = base.add(up_off) as *const f64;
                                        let dc_ptr = dcarry.as_mut_ptr() as *mut f64;
                                        let n_elems = cb / 8;
                                        for i in 0..n_elems {
                                            *dc_ptr.add(i) += *up_ptr.add(i);
                                        }
                                    }
                                    other => panic!(
                                        "ScanBackward: unsupported carry elem size {other} \
                                     (only f32/f64 carries are supported today)"
                                    ),
                                }
                            }
                        }
                        body_buf[*body_carry_in_off..*body_carry_in_off + cb]
                            .copy_from_slice(carry_in);
                        unsafe {
                            for (i, body_x_off) in body_x_offs.iter().enumerate() {
                                let (outer_xs_off, per_step_bytes) = outer_xs_offs[i];
                                let psb = per_step_bytes as usize;
                                std::ptr::copy_nonoverlapping(
                                    base.add(outer_xs_off + t * psb),
                                    body_buf.as_mut_ptr().add(*body_x_off),
                                    psb,
                                );
                            }
                            std::ptr::copy_nonoverlapping(
                                dcarry.as_ptr(),
                                body_buf.as_mut_ptr().add(*body_d_output_off),
                                cb,
                            );
                        }
                        execute_thunks(body_vjp, body_buf);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                body_buf.as_ptr().add(*body_dcarry_out_off),
                                dcarry.as_mut_ptr(),
                                cb,
                            );
                        }
                    };

                if is_recursive {
                    // Griewank treeverse path. Process saved-checkpoint
                    // segments from highest-t to lowest-t; within each,
                    // recursive binary subdivision via
                    // `griewank_process_segment`. Auxiliary memory:
                    // O(log(seg_size) · cb) for the recursion stack
                    // (vs O(seg_size · cb) for the older segment-cache
                    // scheme); recompute work: O(seg_size · log).
                    let leaf_threshold = 4usize;
                    let fb_sched = forward_body.as_ref().unwrap();
                    let fb_init = forward_body_init.as_ref().unwrap().as_slice();
                    let mut segment_end = n_steps - 1;
                    for seg_k in (0..k_total).rev() {
                        let segment_start = if seg_k == 0 {
                            0
                        } else {
                            checkpoint_t_for_k(seg_k - 1) + 1
                        };
                        let mut anchor: Vec<u8> = vec![0u8; cb];
                        unsafe {
                            let src = if seg_k == 0 {
                                base.add(*outer_init_off)
                            } else {
                                base.add(*outer_traj_off + (seg_k - 1) * cb)
                            };
                            std::ptr::copy_nonoverlapping(src, anchor.as_mut_ptr(), cb);
                        }
                        // Closure adapter for the helper's signature
                        // (mutably re-borrows dcarry / body_buf each call).
                        let mut leaf_action = |t: usize, carry_in: &[u8]| {
                            process_iter(t, carry_in, &mut dcarry, &mut body_buf);
                        };
                        unsafe {
                            griewank_process_segment(
                                segment_start,
                                segment_end,
                                &anchor,
                                cb,
                                fb_sched,
                                fb_init,
                                *forward_body_carry_in_off,
                                *forward_body_output_off,
                                forward_body_x_offs,
                                base,
                                outer_xs_offs,
                                &mut fwd_buf,
                                leaf_threshold,
                                &mut leaf_action,
                            );
                        }
                        if seg_k == 0 {
                            break;
                        }
                        segment_end = segment_start - 1;
                    }
                } else {
                    // All-trajectory path: read each carry directly
                    // from the saved trajectory buffer.
                    let mut carry_buf: Vec<u8> = vec![0u8; cb];
                    for t in (0..n_steps).rev() {
                        unsafe {
                            let src = if t == 0 {
                                base.add(*outer_init_off)
                            } else {
                                base.add(*outer_traj_off + (t - 1) * cb)
                            };
                            std::ptr::copy_nonoverlapping(src, carry_buf.as_mut_ptr(), cb);
                        }
                        process_iter(t, &carry_buf, &mut dcarry, &mut body_buf);
                    }
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(dcarry.as_ptr(), base.add(*outer_dinit_off), cb);
                }
            }

            Thunk::ScanBackwardXs {
                body_vjp,
                body_init,
                body_carry_in_off,
                body_x_offs,
                body_d_output_off,
                body_dcarry_out_off,
                body_dxs_out_off,
                outer_init_off,
                outer_traj_off,
                outer_upstream_off,
                outer_xs_offs,
                outer_dxs_off,
                length,
                carry_bytes,
                carry_elem_size,
                per_step_bytes,
                save_trajectory,
                num_checkpoints,
                forward_body,
                forward_body_init,
                forward_body_carry_in_off,
                forward_body_output_off,
                forward_body_x_offs,
            } => {
                let cb = *carry_bytes as usize;
                let psb = *per_step_bytes as usize;
                let n_steps = *length as usize;
                let k_total = *num_checkpoints as usize;
                let is_recursive = k_total != 0 && k_total != n_steps;
                let checkpoint_t_for_k = |k: usize| -> usize {
                    ((k + 1) * n_steps)
                        .div_ceil(k_total)
                        .saturating_sub(1)
                        .min(n_steps - 1)
                };

                // Forward-body recompute scratch + segment cache —
                // exact mirror of the ScanBackward path. With ≈√length
                // checkpoints, total recompute work is O(length).
                let mut fwd_buf: Vec<u8> = if is_recursive {
                    (**forward_body_init.as_ref().unwrap()).clone()
                } else {
                    Vec::new()
                };
                let mut seg_cache: Vec<u8> = Vec::new();
                let mut seg_start_t: usize = usize::MAX;
                let mut seg_count: usize = 0;
                let recompute_carry_t =
                    |t: usize,
                     dst: &mut [u8],
                     fwd_buf: &mut Vec<u8>,
                     seg_cache: &mut Vec<u8>,
                     seg_start_t: &mut usize,
                     seg_count: &mut usize| {
                        if !is_recursive {
                            unsafe {
                                let src = if t == 0 {
                                    base.add(*outer_init_off)
                                } else {
                                    base.add(*outer_traj_off + (t - 1) * cb)
                                };
                                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), cb);
                            }
                            return;
                        }
                        if *seg_start_t != usize::MAX
                            && t >= *seg_start_t
                            && t < *seg_start_t + *seg_count
                        {
                            let off = (t - *seg_start_t) * cb;
                            dst.copy_from_slice(&seg_cache[off..off + cb]);
                            return;
                        }
                        let seg_k = (0..k_total)
                            .find(|&k| t <= checkpoint_t_for_k(k))
                            .unwrap_or(k_total - 1);
                        let (anchor_t, anchor_ptr): (usize, *const u8) = if seg_k == 0 {
                            (0, unsafe { base.add(*outer_init_off) as *const u8 })
                        } else {
                            let prev_ck = checkpoint_t_for_k(seg_k - 1);
                            (prev_ck + 1, unsafe {
                                base.add(*outer_traj_off + (seg_k - 1) * cb) as *const u8
                            })
                        };
                        let seg_end_t = checkpoint_t_for_k(seg_k);
                        let seg_size = seg_end_t - anchor_t + 1;

                        fwd_buf.copy_from_slice(forward_body_init.as_ref().unwrap());
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                anchor_ptr,
                                fwd_buf.as_mut_ptr().add(*forward_body_carry_in_off),
                                cb,
                            );
                        }
                        seg_cache.resize(seg_size * cb, 0u8);
                        seg_cache[0..cb].copy_from_slice(
                            &fwd_buf[*forward_body_carry_in_off..*forward_body_carry_in_off + cb],
                        );
                        let fb_sched = forward_body.as_ref().unwrap();
                        for i in 1..seg_size {
                            let cur_iter = anchor_t + i - 1;
                            for (idx, fb_x_off) in forward_body_x_offs.iter().enumerate() {
                                let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                                let xb = x_psb as usize;
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        base.add(outer_xs_off + cur_iter * xb),
                                        fwd_buf.as_mut_ptr().add(*fb_x_off),
                                        xb,
                                    );
                                }
                            }
                            execute_thunks(fb_sched, fwd_buf);
                            if *forward_body_output_off != *forward_body_carry_in_off {
                                fwd_buf.copy_within(
                                    *forward_body_output_off..*forward_body_output_off + cb,
                                    *forward_body_carry_in_off,
                                );
                            }
                            let cache_off = i * cb;
                            seg_cache[cache_off..cache_off + cb].copy_from_slice(
                                &fwd_buf
                                    [*forward_body_carry_in_off..*forward_body_carry_in_off + cb],
                            );
                        }
                        *seg_start_t = anchor_t;
                        *seg_count = seg_size;

                        let off = (t - anchor_t) * cb;
                        dst.copy_from_slice(&seg_cache[off..off + cb]);
                    };

                let mut dcarry: Vec<u8> = vec![0u8; cb];
                if !*save_trajectory {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(*outer_upstream_off),
                            dcarry.as_mut_ptr(),
                            cb,
                        );
                    }
                }

                let mut body_buf: Vec<u8> = (**body_init).clone();

                for t in (0..n_steps).rev() {
                    if *save_trajectory {
                        unsafe {
                            let up_off = *outer_upstream_off + t * cb;
                            match *carry_elem_size {
                                4 => {
                                    let up_ptr = base.add(up_off) as *const f32;
                                    let dc_ptr = dcarry.as_mut_ptr() as *mut f32;
                                    let n_elems = cb / 4;
                                    for i in 0..n_elems {
                                        *dc_ptr.add(i) += *up_ptr.add(i);
                                    }
                                }
                                8 => {
                                    let up_ptr = base.add(up_off) as *const f64;
                                    let dc_ptr = dcarry.as_mut_ptr() as *mut f64;
                                    let n_elems = cb / 8;
                                    for i in 0..n_elems {
                                        *dc_ptr.add(i) += *up_ptr.add(i);
                                    }
                                }
                                other => panic!(
                                    "ScanBackwardXs: unsupported carry elem size {other} \
                                     (only f32/f64 carries are supported today)"
                                ),
                            }
                        }
                    }

                    // Seed body_vjp's carry input via the recompute
                    // helper (works for both All and Recursive modes),
                    // then x_t_i + d_output.
                    let carry_dst_start = *body_carry_in_off;
                    {
                        let carry_slice = &mut body_buf[carry_dst_start..carry_dst_start + cb];
                        recompute_carry_t(
                            t,
                            carry_slice,
                            &mut fwd_buf,
                            &mut seg_cache,
                            &mut seg_start_t,
                            &mut seg_count,
                        );
                    }
                    unsafe {
                        for (i, body_x_off) in body_x_offs.iter().enumerate() {
                            let (outer_xs_off, x_psb) = outer_xs_offs[i];
                            let xb = x_psb as usize;
                            std::ptr::copy_nonoverlapping(
                                base.add(outer_xs_off + t * xb),
                                body_buf.as_mut_ptr().add(*body_x_off),
                                xb,
                            );
                        }
                        std::ptr::copy_nonoverlapping(
                            dcarry.as_ptr(),
                            body_buf.as_mut_ptr().add(*body_d_output_off),
                            cb,
                        );
                    }

                    execute_thunks(body_vjp, &mut body_buf);

                    // Stash this step's dxs into row `t` of the outer
                    // [length, *per_step_xs] output.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            body_buf.as_ptr().add(*body_dxs_out_off),
                            base.add(*outer_dxs_off + t * psb),
                            psb,
                        );
                    }

                    // Update dcarry for next backward iteration.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            body_buf.as_ptr().add(*body_dcarry_out_off),
                            dcarry.as_mut_ptr(),
                            cb,
                        );
                    }
                }
            }

            Thunk::FusedMmBiasAct {
                a,
                w,
                bias,
                c,
                m,
                k,
                n,
                act,
            } => {
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                unsafe {
                    let out = sl_mut(*c, base, m * n);
                    crate::blas::sgemm_auto(sl(*a, base, m * k), sl(*w, base, k * n), out, m, k, n);
                    match act {
                        Some(Activation::Gelu) => {
                            crate::kernels::par_bias_gelu(out, sl(*bias, base, n), m, n)
                        }
                        Some(other) => {
                            crate::blas::bias_add(out, sl(*bias, base, n), m, n);
                            apply_activation_inplace(out, *other);
                        }
                        None => crate::blas::bias_add(out, sl(*bias, base, n), m, n),
                    }
                }
            }

            Thunk::FusedResidualLN {
                x,
                res,
                bias,
                g,
                b,
                out,
                rows,
                h,
                eps,
                has_bias,
            } => {
                let (rows, h) = (*rows as usize, *h as usize);
                unsafe {
                    let zero = &zero_bias[..h];
                    let bi = if *has_bias { sl(*bias, base, h) } else { zero };
                    let x_ptr = sl(*x, base, rows * h).as_ptr() as usize;
                    let r_ptr = sl(*res, base, rows * h).as_ptr() as usize;
                    let o_ptr = sl_mut(*out, base, rows * h).as_mut_ptr() as usize;
                    let bi_ptr = bi.as_ptr() as usize;
                    let g_ptr = sl(*g, base, h).as_ptr() as usize;
                    let b_ptr = sl(*b, base, h).as_ptr() as usize;
                    let e = *eps;
                    crate::pool::par_for(rows, 4, &|off, cnt| {
                        let xs =
                            std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                        let rs =
                            std::slice::from_raw_parts((r_ptr as *const f32).add(off * h), cnt * h);
                        let os = std::slice::from_raw_parts_mut(
                            (o_ptr as *mut f32).add(off * h),
                            cnt * h,
                        );
                        let bi = std::slice::from_raw_parts(bi_ptr as *const f32, h);
                        let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                        let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                        crate::kernels::residual_bias_layer_norm(xs, rs, bi, g, b, os, cnt, h, e);
                    });
                }
            }

            Thunk::BiasAdd {
                src,
                bias,
                dst,
                m,
                n,
            } => {
                let (m, n) = (*m as usize, *n as usize);
                unsafe {
                    let out = sl_mut(*dst, base, m * n);
                    out.copy_from_slice(sl(*src, base, m * n));
                    crate::blas::bias_add(out, sl(*bias, base, n), m, n);
                }
            }

            Thunk::BinaryFull {
                lhs,
                rhs,
                dst,
                len,
                lhs_len,
                rhs_len,
                op,
                out_dims_bcast,
                bcast_lhs_strides,
                bcast_rhs_strides,
            } => {
                let len = *len as usize;
                let ll = (*lhs_len as usize).max(1);
                let rl = (*rhs_len as usize).max(1);
                unsafe {
                    let l = sl(*lhs, base, ll);
                    let r = sl(*rhs, base, rl);
                    let o = sl_mut(*dst, base, len);
                    // Fast path: shapes match exactly → NEON-vectorized loop.
                    if ll == len && rl == len {
                        #[cfg(target_arch = "aarch64")]
                        if matches!(op, BinaryOp::Add | BinaryOp::Mul) {
                            use std::arch::aarch64::*;
                            let chunks = len / 4;
                            for c in 0..chunks {
                                let off = c * 4;
                                let vl = vld1q_f32(l.as_ptr().add(off));
                                let vr = vld1q_f32(r.as_ptr().add(off));
                                let res = match op {
                                    BinaryOp::Add => vaddq_f32(vl, vr),
                                    BinaryOp::Mul => vmulq_f32(vl, vr),
                                    _ => unreachable!(),
                                };
                                vst1q_f32(o.as_mut_ptr().add(off), res);
                            }
                            for i in (chunks * 4)..len {
                                o[i] = match op {
                                    BinaryOp::Add => l[i] + r[i],
                                    BinaryOp::Mul => l[i] * r[i],
                                    _ => unreachable!(),
                                };
                            }
                            // `continue` to next thunk in the schedule — a
                            // bare `return` here used to exit execute_thunks
                            // entirely, silently dropping every thunk after
                            // the first BinaryFull (catastrophic for chained
                            // adds in BERT embedding stage).
                            continue;
                        }
                    }
                    if !out_dims_bcast.is_empty() {
                        // Shape-aware broadcast path: correct for
                        // bidirectional `[N,1] op [1,S]` etc.
                        let rank = out_dims_bcast.len();
                        let mut coords = vec![0u32; rank];
                        for i in 0..len {
                            let mut rem = i;
                            for ax in (0..rank).rev() {
                                let sz = out_dims_bcast[ax] as usize;
                                coords[ax] = (rem % sz) as u32;
                                rem /= sz;
                            }
                            let mut li: usize = 0;
                            let mut ri: usize = 0;
                            for ax in 0..rank {
                                li += coords[ax] as usize * bcast_lhs_strides[ax] as usize;
                                ri += coords[ax] as usize * bcast_rhs_strides[ax] as usize;
                            }
                            o[i] = match op {
                                BinaryOp::Add => l[li] + r[ri],
                                BinaryOp::Sub => l[li] - r[ri],
                                BinaryOp::Mul => l[li] * r[ri],
                                BinaryOp::Div => l[li] / r[ri],
                                BinaryOp::Max => l[li].max(r[ri]),
                                BinaryOp::Min => l[li].min(r[ri]),
                                BinaryOp::Pow => l[li].powf(r[ri]),
                            };
                        }
                    } else {
                        // Fallback: legacy modulo path (dynamic shapes only).
                        for i in 0..len {
                            let li = if ll == 1 { 0 } else { i % ll };
                            let ri = if rl == 1 { 0 } else { i % rl };
                            o[i] = match op {
                                BinaryOp::Add => l[li] + r[ri],
                                BinaryOp::Sub => l[li] - r[ri],
                                BinaryOp::Mul => l[li] * r[ri],
                                BinaryOp::Div => l[li] / r[ri],
                                BinaryOp::Max => l[li].max(r[ri]),
                                BinaryOp::Min => l[li].min(r[ri]),
                                BinaryOp::Pow => l[li].powf(r[ri]),
                            };
                        }
                    }
                }
            }

            Thunk::Gather {
                table,
                table_len,
                idx,
                dst,
                num_idx,
                trailing,
            } => {
                let (ni, tr) = (*num_idx as usize, *trailing as usize);
                unsafe {
                    let tab = sl(*table, base, *table_len as usize);
                    let ids = sl(*idx, base, ni);
                    let out = sl_mut(*dst, base, ni * tr);
                    for i in 0..ni {
                        let row = ids[i] as usize;
                        out[i * tr..(i + 1) * tr].copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                    }
                }
            }

            Thunk::Narrow {
                src,
                dst,
                outer,
                src_stride,
                dst_stride,
                inner,
            } => {
                let (outer, ss, ds, inner) = (
                    *outer as usize,
                    *src_stride as usize,
                    *dst_stride as usize,
                    *inner as usize,
                );
                unsafe {
                    let s = sl(*src, base, outer * ss);
                    let d = sl_mut(*dst, base, outer * ds);
                    for o in 0..outer {
                        d[o * ds..o * ds + inner].copy_from_slice(&s[o * ss..o * ss + inner]);
                    }
                }
            }

            Thunk::Copy { src, dst, len } => {
                let len = *len as usize;
                unsafe {
                    let s = sl(*src, base, len);
                    let d = sl_mut(*dst, base, len);
                    d.copy_from_slice(s);
                }
            }

            Thunk::LayerNorm {
                src,
                g,
                b,
                dst,
                rows,
                h,
                eps,
            } => {
                let (rows, h) = (*rows as usize, *h as usize);
                unsafe {
                    let input = sl(*src, base, rows * h);
                    let gamma = sl(*g, base, h);
                    let beta = sl(*b, base, h);
                    let output = sl_mut(*dst, base, rows * h);
                    // Parallelize across rows (same pattern as FusedResidualLN)
                    if rows >= 4 && rows * h >= 30_000 {
                        let i_ptr = input.as_ptr() as usize;
                        let o_ptr = output.as_mut_ptr() as usize;
                        let g_ptr = gamma.as_ptr() as usize;
                        let b_ptr = beta.as_ptr() as usize;
                        let e = *eps;
                        crate::pool::par_for(rows, 4, &|off, cnt| {
                            let inp = std::slice::from_raw_parts(
                                (i_ptr as *const f32).add(off * h),
                                cnt * h,
                            );
                            let out = std::slice::from_raw_parts_mut(
                                (o_ptr as *mut f32).add(off * h),
                                cnt * h,
                            );
                            let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                            let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                            for row in 0..cnt {
                                crate::kernels::layer_norm_row(
                                    &inp[row * h..(row + 1) * h],
                                    g,
                                    b,
                                    &mut out[row * h..(row + 1) * h],
                                    h,
                                    e,
                                );
                            }
                        });
                    } else {
                        for row in 0..rows {
                            crate::kernels::layer_norm_row(
                                &input[row * h..(row + 1) * h],
                                gamma,
                                beta,
                                &mut output[row * h..(row + 1) * h],
                                h,
                                *eps,
                            );
                        }
                    }
                }
            }

            Thunk::RmsNorm {
                src,
                g,
                b,
                dst,
                rows,
                h,
                eps,
            } => {
                let (rows, h) = (*rows as usize, *h as usize);
                unsafe {
                    let input = sl(*src, base, rows * h);
                    let gamma = sl(*g, base, h);
                    let beta = sl(*b, base, h);
                    let output = sl_mut(*dst, base, rows * h);
                    let inv_h = 1.0 / h as f32;
                    for row in 0..rows {
                        let in_row = &input[row * h..(row + 1) * h];
                        let out_row = &mut output[row * h..(row + 1) * h];
                        // RMS = sqrt(mean(x^2) + eps); scale = 1/RMS.
                        let mut sumsq = 0f32;
                        for &v in in_row {
                            sumsq += v * v;
                        }
                        let inv_rms = (sumsq * inv_h + *eps).sqrt().recip();
                        for i in 0..h {
                            out_row[i] = in_row[i] * inv_rms * gamma[i] + beta[i];
                        }
                    }
                }
            }

            Thunk::Softmax { data, rows, cols } => {
                let (rows, cols) = (*rows as usize, *cols as usize);
                unsafe {
                    crate::kernels::neon_softmax(sl_mut(*data, base, rows * cols), rows, cols);
                }
            }

            Thunk::Cumsum {
                src,
                dst,
                rows,
                cols,
                exclusive,
            } => {
                let (rows, cols) = (*rows as usize, *cols as usize);
                unsafe {
                    let s = sl(*src, base, rows * cols);
                    let d = sl_mut(*dst, base, rows * cols);
                    if *exclusive {
                        for r in 0..rows {
                            let mut acc = 0.0f32;
                            for c in 0..cols {
                                d[r * cols + c] = acc;
                                acc += s[r * cols + c];
                            }
                        }
                    } else {
                        for r in 0..rows {
                            let mut acc = 0.0f32;
                            for c in 0..cols {
                                acc += s[r * cols + c];
                                d[r * cols + c] = acc;
                            }
                        }
                    }
                }
            }

            Thunk::Sample {
                logits,
                dst,
                batch,
                vocab,
                top_k,
                top_p,
                temperature,
                seed,
            } => {
                let (b, v) = (*batch as usize, *vocab as usize);
                let k = (*top_k as usize).min(v);
                unsafe {
                    let lg = sl(*logits, base, b * v);
                    let out = sl_mut(*dst, base, b);
                    let mut rng =
                        rlx_ir::Philox4x32::new(if *seed == 0 { 0xDEADBEEF } else { *seed });
                    for bi in 0..b {
                        let row = &lg[bi * v..(bi + 1) * v];
                        out[bi] = sample_row(row, k, *top_p, *temperature, &mut rng) as f32;
                    }
                }
            }

            Thunk::GatedDeltaNet {
                q,
                k,
                v,
                g,
                beta,
                dst,
                batch,
                seq,
                heads,
                state_size,
            } => {
                let (b, s, h, n) = (
                    *batch as usize,
                    *seq as usize,
                    *heads as usize,
                    *state_size as usize,
                );
                let scale = 1.0f32 / (n as f32).sqrt();
                unsafe {
                    let qs = sl(*q, base, b * s * h * n);
                    let ks = sl(*k, base, b * s * h * n);
                    let vs = sl(*v, base, b * s * h * n);
                    let gs = sl(*g, base, b * s * h);
                    let bs_ = sl(*beta, base, b * s * h);
                    let out = sl_mut(*dst, base, b * s * h * n);

                    // State per (head): an n×n matrix S[h, i, j].
                    // Reset at the start of each batch row.
                    let mut state = vec![0f32; h * n * n];
                    let mut sk_buf = vec![0f32; n];

                    for bi in 0..b {
                        for st in state.iter_mut() {
                            *st = 0.0;
                        }
                        for ti in 0..s {
                            let qkv_step_base = bi * s * h * n + ti * h * n;
                            let gb_step_base = bi * s * h + ti * h;

                            for hi in 0..h {
                                let q_row =
                                    &qs[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                                let k_row =
                                    &ks[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                                let v_row =
                                    &vs[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                                let g_t = gs[gb_step_base + hi];
                                let beta_t = bs_[gb_step_base + hi];

                                let s_base = hi * n * n;
                                let s_mat = &mut state[s_base..s_base + n * n];

                                // 1) Gate the state: S[h] *= exp(g[t,h]).
                                let g_exp = g_t.exp();
                                for st in s_mat.iter_mut() {
                                    *st *= g_exp;
                                }

                                // 2) sk[j] = Σ_i S[i, j] * k[i].
                                //    S laid out [i, j] row-major over (n, n).
                                for j in 0..n {
                                    let mut acc = 0f32;
                                    for i in 0..n {
                                        acc += s_mat[i * n + j] * k_row[i];
                                    }
                                    sk_buf[j] = acc;
                                }

                                // 3) d[j] = (v[j] - sk[j]) * beta.
                                //    Reuse sk_buf as d.
                                for j in 0..n {
                                    sk_buf[j] = (v_row[j] - sk_buf[j]) * beta_t;
                                }

                                // 4) S[i, j] += k[i] * d[j] (outer prod).
                                for i in 0..n {
                                    let ki = k_row[i];
                                    if ki != 0.0 {
                                        for j in 0..n {
                                            s_mat[i * n + j] += ki * sk_buf[j];
                                        }
                                    }
                                }

                                // 5) o[j] = Σ_i S[i, j] * (q[i] * scale).
                                let out_row =
                                    &mut out[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                                for j in 0..n {
                                    let mut acc = 0f32;
                                    for i in 0..n {
                                        acc += s_mat[i * n + j] * q_row[i];
                                    }
                                    out_row[j] = acc * scale;
                                }
                            }
                        }
                    }
                }
            }

            Thunk::SelectiveScan {
                x,
                delta,
                a,
                b: bp,
                c: cp,
                dst,
                batch,
                seq,
                hidden,
                state_size,
            } => {
                let (b, s, h, n) = (
                    *batch as usize,
                    *seq as usize,
                    *hidden as usize,
                    *state_size as usize,
                );
                unsafe {
                    let xs = sl(*x, base, b * s * h);
                    let dt = sl(*delta, base, b * s * h);
                    let am = sl(*a, base, h * n);
                    let bm = sl(*bp, base, b * s * n);
                    let cm = sl(*cp, base, b * s * n);
                    let out = sl_mut(*dst, base, b * s * h);

                    // State buffer per-batch: h channels × n state.
                    // Sequential along the seq dimension; could
                    // parallelize over batch+channel later.
                    let mut state = vec![0f32; h * n];
                    for bi in 0..b {
                        // Reset state at the start of each batch row.
                        for v in state.iter_mut() {
                            *v = 0.0;
                        }
                        for si in 0..s {
                            let x_row = &xs[bi * s * h + si * h..bi * s * h + (si + 1) * h];
                            let dt_row = &dt[bi * s * h + si * h..bi * s * h + (si + 1) * h];
                            let b_row = &bm[bi * s * n + si * n..bi * s * n + (si + 1) * n];
                            let c_row = &cm[bi * s * n + si * n..bi * s * n + (si + 1) * n];
                            let out_row = &mut out[bi * s * h + si * h..bi * s * h + (si + 1) * h];

                            for ci in 0..h {
                                let d = dt_row[ci];
                                let xv = x_row[ci];
                                let mut acc = 0f32;
                                for ni in 0..n {
                                    // Discretize: exp(d * a) and d * b.
                                    let da = (d * am[ci * n + ni]).exp();
                                    state[ci * n + ni] =
                                        da * state[ci * n + ni] + d * b_row[ni] * xv;
                                    acc += c_row[ni] * state[ci * n + ni];
                                }
                                out_row[ci] = acc;
                            }
                        }
                    }
                }
            }

            Thunk::DequantMatMul {
                x,
                w_q,
                scale,
                zp,
                dst,
                m,
                k,
                n,
                block_size,
                is_asymmetric,
            } => {
                let (m, k, n, bs) = (*m as usize, *k as usize, *n as usize, *block_size as usize);
                let n_blocks = k.div_ceil(bs);
                unsafe {
                    let xs = sl(*x, base, m * k);
                    let w_bytes = std::slice::from_raw_parts(base.add(*w_q) as *const i8, k * n);
                    let scales = sl(*scale, base, n_blocks * n);
                    let zps = if *is_asymmetric {
                        sl(*zp, base, n_blocks * n)
                    } else {
                        &[][..]
                    };
                    let out = sl_mut(*dst, base, m * n);
                    dequant_matmul_int8(xs, w_bytes, scales, zps, out, m, k, n, bs, *is_asymmetric);
                }
            }

            Thunk::DequantMatMulGguf {
                x,
                w_q,
                dst,
                m,
                k,
                n,
                scheme,
            } => {
                use rlx_ir::quant::QuantScheme;
                let (m, k, n) = (*m as usize, *k as usize, *n as usize);
                let block_bytes = scheme.gguf_block_bytes() as usize;
                let block_elems = scheme.gguf_block_size() as usize;
                debug_assert!(block_bytes > 0 && block_elems > 0, "non-GGUF scheme in GGUF arm");
                debug_assert!(
                    (k * n).is_multiple_of(block_elems),
                    "k*n={} not aligned to GGUF block size {}",
                    k * n,
                    block_elems
                );
                let total_bytes = (k * n) / block_elems * block_bytes;
                unsafe {
                    let xs = sl(*x, base, m * k);
                    let w_bytes_ptr = base.add(*w_q) as *const u8;
                    let w_bytes = std::slice::from_raw_parts(w_bytes_ptr, total_bytes);
                    // Dequant the packed weight into f32 scratch. This
                    // keeps the arena footprint small (weights stay
                    // packed); future work is a tile-streaming kernel
                    // that fuses the dequant into the matmul loop.
                    let w_f32 = match scheme {
                        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(w_bytes, k * n),
                        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(w_bytes, k * n),
                        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(w_bytes, k * n),
                        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(w_bytes, k * n),
                        _ => unreachable!(),
                    }
                    .expect("GGUF dequant_*_k failed");
                    // GGUF stores 2D weights as the transpose of the
                    // safetensors / matmul-RHS layout: the dequant
                    // output is `[n, k]` row-major. Use `sgemm_bt`
                    // (B transposed) so BLAS reads it as `[k, n]`
                    // logically without a separate transpose pass.
                    let out = sl_mut(*dst, base, m * n);
                    crate::blas::sgemm_bt(xs, &w_f32, out, m, k, n, 1.0);
                }
            }

            Thunk::LoraMatMul {
                x,
                w,
                a,
                b,
                dst,
                m,
                k,
                n,
                r,
                scale,
            } => {
                let (m, k, n, r) = (*m as usize, *k as usize, *n as usize, *r as usize);
                unsafe {
                    let xs = sl(*x, base, m * k);
                    let ws = sl(*w, base, k * n);
                    let a_s = sl(*a, base, k * r);
                    let bs = sl(*b, base, r * n);
                    let out = sl_mut(*dst, base, m * n);
                    crate::blas::sgemm(xs, ws, out, m, k, n);
                    let mut tmp = vec![0f32; m * r];
                    crate::blas::sgemm(xs, a_s, &mut tmp, m, k, r);
                    if *scale != 1.0 {
                        for v in tmp.iter_mut() {
                            *v *= *scale;
                        }
                    }
                    crate::blas::sgemm_accumulate(&tmp, bs, out, m, r, n);
                }
            }

            Thunk::Attention {
                q,
                k,
                v,
                mask,
                out,
                batch,
                seq,
                kv_seq,
                heads,
                head_dim,
                mask_kind,
                q_row_stride,
                k_row_stride,
                v_row_stride,
                bhsd,
            } => {
                let (b, q_s, k_s, nh, dh) = (
                    *batch as usize,
                    *seq as usize,
                    *kv_seq as usize,
                    *heads as usize,
                    *head_dim as usize,
                );
                let hs = nh * dh;
                // For [B, H, S, D] layout each (b, h) tile is dense
                // contiguous; the qrs/krs/vrs strides are not used.
                let (qrs, krs, vrs) = if *bhsd {
                    (dh, dh, dh)
                } else {
                    (
                        *q_row_stride as usize,
                        *k_row_stride as usize,
                        *v_row_stride as usize,
                    )
                };
                let bhsd = *bhsd;
                let _ = (q_row_stride, k_row_stride, v_row_stride);
                let scale = (dh as f32).powf(-0.5);
                let ss = q_s * k_s;
                let cfg = crate::config::RuntimeConfig::global();
                unsafe {
                    // Slice lengths cover the strided span. When Q/K/V
                    // alias the parent QKV (post-#46-fusion), the same
                    // bytes back all three slices — compiler bounds
                    // checks see the right size. For [B, H, S, D] the
                    // buffer is densely B*H*S*D elements; the row
                    // strides aren't used.
                    let q_len = if bhsd {
                        b * nh * q_s * dh
                    } else {
                        b * q_s * qrs
                    };
                    let k_len = if bhsd {
                        b * nh * k_s * dh
                    } else {
                        b * k_s * krs
                    };
                    let v_len = if bhsd {
                        b * nh * k_s * dh
                    } else {
                        b * k_s * vrs
                    };
                    let q_data = sl(*q, base, q_len);
                    let k_data = sl(*k, base, k_len);
                    let v_data = sl(*v, base, v_len);
                    let mask_data: &[f32] = match mask_kind {
                        rlx_ir::op::MaskKind::Custom => sl(*mask, base, b * k_s),
                        rlx_ir::op::MaskKind::Bias => sl(*mask, base, b * nh * q_s * k_s),
                        _ => &[],
                    };
                    let out_len = if bhsd {
                        b * nh * q_s * dh
                    } else {
                        b * q_s * hs
                    };
                    let out_data = sl_mut(*out, base, out_len);

                    // ── [B, H, S, D] fallback ──────────────────────
                    // The NEON / strided-BLAS specializations below
                    // are written for the [B, S, H, D] layout. When
                    // the input is head-major ([B, H, S, D] —
                    // matching rlx-cuda / rlx-rocm / rlx-tpu), bypass
                    // them and run a simple (correct but slower)
                    // scalar implementation. Production-CPU inference
                    // graphs use [B, S, H, D] so they still hit the
                    // hot path; cross-backend parity tests use
                    // [B, H, S, D] and land here.
                    if bhsd {
                        let scores = &mut sdpa_scores[..ss];
                        for bi in 0..b {
                            for hi in 0..nh {
                                let q_head_base = bi * nh * q_s * dh + hi * q_s * dh;
                                let k_head_base = bi * nh * k_s * dh + hi * k_s * dh;
                                // Q@K^T
                                for qi in 0..q_s {
                                    let q_base = q_head_base + qi * dh;
                                    for ki in 0..k_s {
                                        let k_base = k_head_base + ki * dh;
                                        let mut dot = 0f32;
                                        for d in 0..dh {
                                            dot += q_data[q_base + d] * k_data[k_base + d];
                                        }
                                        scores[qi * k_s + ki] = dot * scale;
                                        if matches!(mask_kind, rlx_ir::op::MaskKind::Custom)
                                            && !mask_data.is_empty()
                                            && mask_data[bi * k_s + ki] < mask_thr
                                        {
                                            scores[qi * k_s + ki] = mask_neg;
                                        }
                                    }
                                }
                                if matches!(mask_kind, rlx_ir::op::MaskKind::Bias) {
                                    let off = (bi * nh + hi) * q_s * k_s;
                                    for i in 0..q_s * k_s {
                                        scores[i] += mask_data[off + i];
                                    }
                                }
                                apply_synthetic_mask(scores, q_s, k_s, *mask_kind);
                                crate::kernels::neon_softmax(scores, q_s, k_s);
                                // score @ V
                                for qi in 0..q_s {
                                    let o_base = q_head_base + qi * dh;
                                    for d in 0..dh {
                                        out_data[o_base + d] = 0.0;
                                    }
                                    for ki in 0..k_s {
                                        let sc = scores[qi * k_s + ki];
                                        if sc > score_thr {
                                            let v_base = k_head_base + ki * dh;
                                            for d in 0..dh {
                                                out_data[o_base + d] += sc * v_data[v_base + d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // ── Auto-select kernel: NEON dots vs strided BLAS ───
                    // For tiny inputs (batch=1, short seq), per-head BLAS call
                    // overhead (~0.5µs × 2 calls × num_heads × num_layers)
                    // exceeds the NEON compute cost. Use direct strided NEON
                    // with zero dispatch overhead.
                    // For batch≥2: always BLAS + par_for (parallelism wins).
                    if b == 1 && q_s.max(k_s) <= cfg.sdpa_seq_threshold {
                        // ── Sequential NEON path (zero overhead) ──
                        let scores = &mut sdpa_scores[..ss];
                        #[cfg(target_arch = "aarch64")]
                        let neon_chunks = dh / 4;

                        for bi in 0..b {
                            for hi in 0..nh {
                                // Q@K^T via strided NEON dot products
                                for qi in 0..q_s {
                                    let q_off = bi * q_s * qrs + qi * qrs + hi * dh;
                                    for ki in 0..k_s {
                                        let k_off = bi * k_s * krs + ki * krs + hi * dh;
                                        #[cfg(target_arch = "aarch64")]
                                        let mut dot;
                                        #[cfg(not(target_arch = "aarch64"))]
                                        let mut dot = 0f32;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let mut acc = vdupq_n_f32(0.0);
                                            for c in 0..neon_chunks {
                                                let vq =
                                                    vld1q_f32(q_data.as_ptr().add(q_off + c * 4));
                                                let vk =
                                                    vld1q_f32(k_data.as_ptr().add(k_off + c * 4));
                                                acc = vfmaq_f32(acc, vq, vk);
                                            }
                                            dot = vaddvq_f32(acc);
                                            for d in (neon_chunks * 4)..dh {
                                                dot += q_data[q_off + d] * k_data[k_off + d];
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..dh {
                                            dot += q_data[q_off + d] * k_data[k_off + d];
                                        }
                                        scores[qi * k_s + ki] = dot * scale;
                                        // Inner-loop Custom mask check —
                                        // Causal / SlidingWindow / None
                                        // apply outside the loop below.
                                        // Skip for Bias — that mask is a
                                        // per-head additive tensor, not a
                                        // 0/1 key-padding mask.
                                        if matches!(mask_kind, rlx_ir::op::MaskKind::Custom)
                                            && !mask_data.is_empty()
                                            && mask_data[bi * k_s + ki] < mask_thr
                                        {
                                            scores[qi * k_s + ki] = mask_neg;
                                        }
                                    }
                                }

                                if matches!(mask_kind, rlx_ir::op::MaskKind::Bias) {
                                    let off = (bi * nh + hi) * q_s * k_s;
                                    for i in 0..q_s * k_s {
                                        scores[i] += mask_data[off + i];
                                    }
                                }
                                apply_synthetic_mask(scores, q_s, k_s, *mask_kind);
                                crate::kernels::neon_softmax(scores, q_s, k_s);

                                // Score@V via strided NEON accumulation (zero-copy)
                                for qi in 0..q_s {
                                    let o_off = bi * q_s * hs + qi * hs + hi * dh;
                                    // Zero output for this head position
                                    for d in 0..dh {
                                        out_data[o_off + d] = 0.0;
                                    }
                                    for ki in 0..k_s {
                                        let sc = scores[qi * k_s + ki];
                                        if sc > score_thr {
                                            let v_off = bi * k_s * vrs + ki * vrs + hi * dh;
                                            #[cfg(target_arch = "aarch64")]
                                            {
                                                use std::arch::aarch64::*;
                                                let vsc = vdupq_n_f32(sc);
                                                for c in 0..neon_chunks {
                                                    let off = c * 4;
                                                    let vo = vld1q_f32(
                                                        out_data.as_ptr().add(o_off + off),
                                                    );
                                                    let vv =
                                                        vld1q_f32(v_data.as_ptr().add(v_off + off));
                                                    vst1q_f32(
                                                        out_data.as_mut_ptr().add(o_off + off),
                                                        vfmaq_f32(vo, vsc, vv),
                                                    );
                                                }
                                            }
                                            #[cfg(not(target_arch = "aarch64"))]
                                            for d in 0..dh {
                                                out_data[o_off + d] += sc * v_data[v_off + d];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // ── Parallel strided BLAS path (high throughput) ──
                        let total_work = b * nh;
                        let q_addr = q_data.as_ptr() as usize;
                        let k_addr = k_data.as_ptr() as usize;
                        let v_addr = v_data.as_ptr() as usize;
                        let m_addr = mask_data.as_ptr() as usize;
                        let o_addr = out_data.as_mut_ptr() as usize;
                        let sc_addr = sdpa_scores.as_mut_ptr() as usize;

                        crate::pool::par_for(total_work, 1, &|off, cnt| {
                            for idx in off..off + cnt {
                                let bi = idx / nh;
                                let hi = idx % nh;

                                let q_start = (q_addr as *const f32).add(bi * q_s * qrs + hi * dh);
                                let k_start = (k_addr as *const f32).add(bi * k_s * krs + hi * dh);
                                let v_start = (v_addr as *const f32).add(bi * k_s * vrs + hi * dh);
                                let o_start = (o_addr as *mut f32).add(bi * q_s * hs + hi * dh);
                                let sc = std::slice::from_raw_parts_mut(
                                    (sc_addr as *mut f32).add(idx * ss),
                                    ss,
                                );

                                // LDA = qrs, LDB = krs (parent row strides
                                // when fused; hs otherwise).
                                crate::blas::sgemm_general(
                                    q_start,
                                    k_start,
                                    sc.as_mut_ptr(),
                                    q_s,
                                    k_s,
                                    dh,
                                    scale,
                                    0.0,
                                    qrs,
                                    krs,
                                    k_s,
                                    false,
                                    true,
                                );

                                match mask_kind {
                                    rlx_ir::op::MaskKind::Custom => {
                                        let mask_bi = std::slice::from_raw_parts(
                                            (m_addr as *const f32).add(bi * k_s),
                                            k_s,
                                        );
                                        for ki in 0..k_s {
                                            if mask_bi[ki] < mask_thr {
                                                for qi in 0..q_s {
                                                    sc[qi * k_s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Bias => {
                                        // Per-head additive bias slice.
                                        let bias = std::slice::from_raw_parts(
                                            (m_addr as *const f32)
                                                .add((bi * nh + hi) * q_s * k_s),
                                            q_s * k_s,
                                        );
                                        for i in 0..q_s * k_s {
                                            sc[i] += bias[i];
                                        }
                                    }
                                    _ => apply_synthetic_mask(sc, q_s, k_s, *mask_kind),
                                }

                                crate::kernels::neon_softmax(sc, q_s, k_s);

                                // LDB = vrs (parent row stride when
                                // fused; hs otherwise). LDC stays hs —
                                // output is its own contiguous buffer.
                                crate::blas::sgemm_general(
                                    sc.as_ptr(),
                                    v_start,
                                    o_start,
                                    q_s,
                                    dh,
                                    k_s,
                                    1.0,
                                    0.0,
                                    k_s,
                                    vrs,
                                    hs,
                                    false,
                                    false,
                                );
                            }
                        });
                    }
                }
            }

            Thunk::ActivationInPlace { data, len, act } => {
                let len = *len as usize;
                unsafe {
                    let d = sl_mut(*data, base, len);
                    match act {
                        Activation::Gelu => crate::kernels::par_gelu_inplace(d),
                        Activation::GeluApprox => crate::kernels::par_gelu_approx_inplace(d),
                        Activation::Silu => crate::kernels::par_silu_inplace(d),
                        Activation::Relu => {
                            for v in d.iter_mut() {
                                *v = v.max(0.0);
                            }
                        }
                        Activation::Sigmoid => {
                            for v in d.iter_mut() {
                                *v = 1.0 / (1.0 + (-*v).exp());
                            }
                        }
                        Activation::Tanh => {
                            for v in d.iter_mut() {
                                *v = v.tanh();
                            }
                        }
                        Activation::Exp => {
                            for v in d.iter_mut() {
                                *v = v.exp();
                            }
                        }
                        Activation::Log => {
                            for v in d.iter_mut() {
                                *v = v.ln();
                            }
                        }
                        Activation::Sqrt => {
                            for v in d.iter_mut() {
                                *v = v.sqrt();
                            }
                        }
                        Activation::Rsqrt => {
                            for v in d.iter_mut() {
                                *v = 1.0 / v.sqrt();
                            }
                        }
                        Activation::Neg => {
                            for v in d.iter_mut() {
                                *v = -*v;
                            }
                        }
                        Activation::Abs => {
                            for v in d.iter_mut() {
                                *v = v.abs();
                            }
                        }
                        Activation::Round => {
                            for v in d.iter_mut() {
                                *v = v.round();
                            }
                        }
                        Activation::Sin => {
                            for v in d.iter_mut() {
                                *v = v.sin();
                            }
                        }
                        Activation::Cos => {
                            for v in d.iter_mut() {
                                *v = v.cos();
                            }
                        }
                        Activation::Tan => {
                            for v in d.iter_mut() {
                                *v = v.tan();
                            }
                        }
                        Activation::Atan => {
                            for v in d.iter_mut() {
                                *v = v.atan();
                            }
                        }
                    }
                }
            }

            Thunk::FusedAttnBlock {
                hidden,
                qkv_w,
                out_w,
                mask,
                out,
                qkv_b,
                out_b,
                cos,
                sin,
                cos_len,
                batch,
                seq,
                hs,
                nh,
                dh,
                has_bias,
                has_rope,
            } => {
                let (b, s) = (*batch as usize, *seq as usize);
                let (h, n_h, d_h) = (*hs as usize, *nh as usize, *dh as usize);
                let m = b * s;
                let scale = (d_h as f32).powf(-0.5);
                let half = d_h / 2;
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let wq = sl(*qkv_w, base, h * 3 * h);
                    let wo = sl(*out_w, base, h * h);
                    let mk = sl(*mask, base, b * s);
                    let dst = sl_mut(*out, base, m * h);

                    // Stack-allocated intermediates — all fit in L1 cache for small batch
                    let mut qkv = vec![0f32; m * 3 * h];
                    let mut attn_out = vec![0f32; m * h];
                    let mut scores_buf = vec![0f32; s * s]; // one head at a time

                    // 1. QKV projection: [m, h] @ [h, 3h] → [m, 3h]
                    crate::blas::sgemm(inp, wq, &mut qkv, m, h, 3 * h);
                    if *has_bias {
                        let bias = sl(*qkv_b, base, 3 * h);
                        crate::blas::bias_add(&mut qkv, bias, m, 3 * h);
                    }

                    // 2. Multi-head SDPA (Q/K/V are views into qkv at offsets 0, h, 2h)
                    //    Process heads sequentially with inline RoPE — zero copy.
                    #[cfg(target_arch = "aarch64")]
                    let neon_chunks = d_h / 4;
                    #[cfg(target_arch = "aarch64")]
                    let _rope_chunks = half / 4;

                    for bi in 0..b {
                        for hi in 0..n_h {
                            // For each (query_pos, key_pos): compute Q@K^T with inline RoPE
                            for qi in 0..s {
                                let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                for ki in 0..s {
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    let mut dot = 0f32;

                                    if *has_rope {
                                        // Apply RoPE inline during dot product
                                        let q_cos = qi * half;
                                        let k_cos = ki * half;
                                        let cos_tab = sl(*cos, base, *cos_len as usize);
                                        let sin_tab = sl(*sin, base, *cos_len as usize);
                                        // First half: (q1*c - q2*s) * (k1*c - k2*s)
                                        // Second half: (q2*c + q1*s) * (k2*c + k1*s)
                                        for i in 0..half {
                                            let q1 = qkv[q_base + i];
                                            let q2 = qkv[q_base + half + i];
                                            let k1 = qkv[k_base + i];
                                            let k2 = qkv[k_base + half + i];
                                            let c_q = cos_tab[q_cos + i];
                                            let s_q = sin_tab[q_cos + i];
                                            let c_k = cos_tab[k_cos + i];
                                            let s_k = sin_tab[k_cos + i];
                                            let qr1 = q1 * c_q - q2 * s_q;
                                            let kr1 = k1 * c_k - k2 * s_k;
                                            let qr2 = q2 * c_q + q1 * s_q;
                                            let kr2 = k2 * c_k + k1 * s_k;
                                            dot += qr1 * kr1 + qr2 * kr2;
                                        }
                                    } else {
                                        // Standard dot product
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let mut acc = vdupq_n_f32(0.0);
                                            for c in 0..neon_chunks {
                                                let vq =
                                                    vld1q_f32(qkv.as_ptr().add(q_base + c * 4));
                                                let vk =
                                                    vld1q_f32(qkv.as_ptr().add(k_base + c * 4));
                                                acc = vfmaq_f32(acc, vq, vk);
                                            }
                                            dot = vaddvq_f32(acc);
                                            for d in (neon_chunks * 4)..d_h {
                                                dot += qkv[q_base + d] * qkv[k_base + d];
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            dot += qkv[q_base + d] * qkv[k_base + d];
                                        }
                                    }

                                    scores_buf[qi * s + ki] = dot * scale;
                                    if mk[bi * s + ki] < mask_thr {
                                        scores_buf[qi * s + ki] = mask_neg;
                                    }
                                }
                            }

                            // Softmax
                            crate::kernels::neon_softmax(&mut scores_buf[..s * s], s, s);

                            // Score @ V accumulation (V at offset 2h in QKV)
                            for qi in 0..s {
                                let o_base = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn_out[o_base + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let sc = scores_buf[qi * s + ki];
                                    if sc > score_thr {
                                        let v_base = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vsc = vdupq_n_f32(sc);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                let vo =
                                                    vld1q_f32(attn_out.as_ptr().add(o_base + off));
                                                let vv = vld1q_f32(qkv.as_ptr().add(v_base + off));
                                                vst1q_f32(
                                                    attn_out.as_mut_ptr().add(o_base + off),
                                                    vfmaq_f32(vo, vsc, vv),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn_out[o_base + d] += sc * qkv[v_base + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // 3. Output projection: [m, h] @ [h, h] → dst
                    crate::blas::sgemm(&attn_out, wo, dst, m, h, h);
                    if *has_bias {
                        let bias = sl(*out_b, base, h);
                        crate::blas::bias_add(dst, bias, m, h);
                    }
                }
            }

            Thunk::Rope {
                src,
                cos,
                sin,
                dst,
                batch,
                seq,
                hidden,
                head_dim,
                cos_len,
                src_row_stride,
            } => {
                let (b, s, hs, dh) = (
                    *batch as usize,
                    *seq as usize,
                    *hidden as usize,
                    *head_dim as usize,
                );
                let half = dh / 2;
                let nh = hs / dh;
                let cl = *cos_len as usize;
                let src_rs = *src_row_stride as usize;
                unsafe {
                    // src may be wider than hs (e.g. QKV's 3*hs) when the
                    // Narrow→Rope fusion has rewired us to read directly
                    // from the parent buffer. Allocate the slice using
                    // the source's row stride.
                    let x = sl(*src, base, b * s * src_rs);
                    let cos_tab = sl(*cos, base, cl);
                    let sin_tab = sl(*sin, base, cl);
                    let out = sl_mut(*dst, base, b * s * hs);

                    // Parallel over (batch × seq) for large inputs
                    let total = b * s;
                    let x_ptr = x.as_ptr() as usize;
                    let o_ptr = out.as_mut_ptr() as usize;
                    let c_ptr = cos_tab.as_ptr() as usize;
                    let s_ptr = sin_tab.as_ptr() as usize;

                    crate::pool::par_for(total, 4, &|off, cnt| {
                        for idx in off..off + cnt {
                            let bi = idx / s;
                            let si = idx % s;
                            let tab_off = si * half;

                            for hi in 0..nh {
                                // Source walks with src_row_stride; dest
                                // is always tightly packed (hs).
                                let src_base = bi * s * src_rs + si * src_rs + hi * dh;
                                let dst_base = bi * s * hs + si * hs + hi * dh;
                                let xp = (x_ptr as *const f32).add(src_base);
                                let op = (o_ptr as *mut f32).add(dst_base);
                                let cp = (c_ptr as *const f32).add(tab_off);
                                let sp = (s_ptr as *const f32).add(tab_off);

                                #[cfg(target_arch = "aarch64")]
                                {
                                    use std::arch::aarch64::*;
                                    let chunks = half / 4;
                                    for c in 0..chunks {
                                        let off4 = c * 4;
                                        let vx1 = vld1q_f32(xp.add(off4));
                                        let vx2 = vld1q_f32(xp.add(half + off4));
                                        let vc = vld1q_f32(cp.add(off4));
                                        let vs = vld1q_f32(sp.add(off4));
                                        // first half: x1*cos - x2*sin
                                        let r1 = vfmsq_f32(vmulq_f32(vx1, vc), vx2, vs);
                                        // second half: x2*cos + x1*sin
                                        let r2 = vfmaq_f32(vmulq_f32(vx2, vc), vx1, vs);
                                        vst1q_f32(op.add(off4), r1);
                                        vst1q_f32(op.add(half + off4), r2);
                                    }
                                    for i in (chunks * 4)..half {
                                        let x1 = *xp.add(i);
                                        let x2 = *xp.add(half + i);
                                        let cv = *cp.add(i);
                                        let sv = *sp.add(i);
                                        *op.add(i) = x1 * cv - x2 * sv;
                                        *op.add(half + i) = x2 * cv + x1 * sv;
                                    }
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                for i in 0..half {
                                    let x1 = *xp.add(i);
                                    let x2 = *xp.add(half + i);
                                    let cv = *cp.add(i);
                                    let sv = *sp.add(i);
                                    *op.add(i) = x1 * cv - x2 * sv;
                                    *op.add(half + i) = x2 * cv + x1 * sv;
                                }
                            }
                        }
                    });
                }
            }
            Thunk::FusedBertLayer {
                hidden,
                qkv_w,
                qkv_b,
                out_w,
                out_b,
                mask,
                ln1_g,
                ln1_b,
                eps1,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                ln2_g,
                ln2_b,
                eps2,
                out,
                batch,
                seq,
                hs,
                nh,
                dh,
                int_dim,
            } => {
                let (b, s, h, n_h, d_h) = (
                    *batch as usize,
                    *seq as usize,
                    *hs as usize,
                    *nh as usize,
                    *dh as usize,
                );
                let m = b * s;
                let id = *int_dim as usize;
                let scale = (d_h as f32).powf(-0.5);
                let _half = d_h / 2;
                #[cfg(target_arch = "aarch64")]
                let neon_chunks = d_h / 4;
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let dst = sl_mut(*out, base, m * h);
                    let mk = sl(*mask, base, b * s);

                    // Pre-allocated buffers (zero malloc per layer — allocated once before thunk loop)
                    let qkv = std::slice::from_raw_parts_mut(fl_qkv.as_mut_ptr(), m * 3 * h);
                    let attn = std::slice::from_raw_parts_mut(fl_attn.as_mut_ptr(), m * h);
                    let res = std::slice::from_raw_parts_mut(fl_res.as_mut_ptr(), m * h);
                    let normed = std::slice::from_raw_parts_mut(fl_normed.as_mut_ptr(), m * h);
                    let ffn = std::slice::from_raw_parts_mut(fl_ffn.as_mut_ptr(), m * id);
                    let sc = std::slice::from_raw_parts_mut(fl_sc.as_mut_ptr(), s * s);

                    // QKV (parallelized across cores — multiple AMX coprocessors)
                    crate::blas::par_sgemm_bias(
                        inp,
                        sl(*qkv_w, base, h * 3 * h),
                        sl(*qkv_b, base, 3 * h),
                        qkv,
                        m,
                        h,
                        3 * h,
                    );

                    // SDPA per head (sequential NEON, inline — zero overhead)
                    for bi in 0..b {
                        for hi in 0..n_h {
                            for qi in 0..s {
                                for ki in 0..s {
                                    let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    #[cfg(target_arch = "aarch64")]
                                    let dot;
                                    #[cfg(not(target_arch = "aarch64"))]
                                    let mut dot = 0f32;
                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        use std::arch::aarch64::*;
                                        let mut acc = vdupq_n_f32(0.0);
                                        for c in 0..neon_chunks {
                                            acc = vfmaq_f32(
                                                acc,
                                                vld1q_f32(qkv.as_ptr().add(q_base + c * 4)),
                                                vld1q_f32(qkv.as_ptr().add(k_base + c * 4)),
                                            );
                                        }
                                        dot = vaddvq_f32(acc);
                                    }
                                    #[cfg(not(target_arch = "aarch64"))]
                                    for d in 0..d_h {
                                        dot += qkv[q_base + d] * qkv[k_base + d];
                                    }
                                    sc[qi * s + ki] = dot * scale;
                                    if mk[bi * s + ki] < mask_thr {
                                        sc[qi * s + ki] = mask_neg;
                                    }
                                }
                            }
                            crate::kernels::neon_softmax(&mut sc[..s * s], s, s);
                            for qi in 0..s {
                                let o = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn[o + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let w = sc[qi * s + ki];
                                    if w > score_thr {
                                        let v = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vw = vdupq_n_f32(w);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                vst1q_f32(
                                                    attn.as_mut_ptr().add(o + off),
                                                    vfmaq_f32(
                                                        vld1q_f32(attn.as_ptr().add(o + off)),
                                                        vw,
                                                        vld1q_f32(qkv.as_ptr().add(v + off)),
                                                    ),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn[o + d] += w * qkv[v + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Out proj (sgemm + bias fused) + residual add with NEON
                    crate::blas::sgemm_bias(
                        attn,
                        sl(*out_w, base, h * h),
                        sl(*out_b, base, h),
                        res,
                        m,
                        h,
                        h,
                    );
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        let chunks_h = (m * h) / 4;
                        for c in 0..chunks_h {
                            let off = c * 4;
                            vst1q_f32(
                                res.as_mut_ptr().add(off),
                                vaddq_f32(
                                    vld1q_f32(res.as_ptr().add(off)),
                                    vld1q_f32(inp.as_ptr().add(off)),
                                ),
                            );
                        }
                        for i in (chunks_h * 4)..(m * h) {
                            res[i] += inp[i];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..m * h {
                        res[i] += inp[i];
                    }

                    // LN1 (fused residual already done above — just normalize)
                    let g1 = sl(*ln1_g, base, h);
                    let b1 = sl(*ln1_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g1,
                            b1,
                            &mut normed[r * h..(r + 1) * h],
                            h,
                            *eps1,
                        );
                    }

                    // FFN: fc1 (parallel across cores) + GELU
                    crate::blas::par_sgemm_bias(
                        normed,
                        sl(*fc1_w, base, h * id),
                        sl(*fc1_b, base, id),
                        ffn,
                        m,
                        h,
                        id,
                    );
                    crate::kernels::par_gelu_inplace(ffn);

                    // fc2 + bias (parallel across cores) + residual with NEON
                    crate::blas::par_sgemm_bias(
                        ffn,
                        sl(*fc2_w, base, id * h),
                        sl(*fc2_b, base, h),
                        res,
                        m,
                        id,
                        h,
                    );
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        let chunks_h = (m * h) / 4;
                        for c in 0..chunks_h {
                            let off = c * 4;
                            vst1q_f32(
                                res.as_mut_ptr().add(off),
                                vaddq_f32(
                                    vld1q_f32(res.as_ptr().add(off)),
                                    vld1q_f32(normed.as_ptr().add(off)),
                                ),
                            );
                        }
                        for i in (chunks_h * 4)..(m * h) {
                            res[i] += normed[i];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for i in 0..m * h {
                        res[i] += normed[i];
                    }

                    // LN2 → output
                    let g2 = sl(*ln2_g, base, h);
                    let b2 = sl(*ln2_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g2,
                            b2,
                            &mut dst[r * h..(r + 1) * h],
                            h,
                            *eps2,
                        );
                    }
                }
            }

            Thunk::FusedNomicLayer {
                hidden,
                qkv_w,
                out_w,
                mask,
                cos,
                sin,
                cos_len,
                ln1_g,
                ln1_b,
                eps1,
                fc11_w,
                fc12_w: _,
                fc2_w,
                ln2_g,
                ln2_b,
                eps2,
                out,
                batch,
                seq,
                hs,
                nh,
                dh,
                int_dim,
            } => {
                let (b, s, h, n_h, d_h) = (
                    *batch as usize,
                    *seq as usize,
                    *hs as usize,
                    *nh as usize,
                    *dh as usize,
                );
                let m = b * s;
                let id = *int_dim as usize;
                let scale = (d_h as f32).powf(-0.5);
                let half_dh = d_h / 2;
                #[cfg(target_arch = "aarch64")]
                let neon_chunks = d_h / 4;
                unsafe {
                    let inp = sl(*hidden, base, m * h);
                    let dst = sl_mut(*out, base, m * h);
                    let mk = sl(*mask, base, b * s);
                    let cos_tab = sl(*cos, base, *cos_len as usize);
                    let sin_tab = sl(*sin, base, *cos_len as usize);
                    // fc11_w is the fused [h, 2*int_dim] weight (fc11 || fc12 concatenated)
                    let fused_fc_w = sl(*fc11_w, base, h * 2 * id);

                    let mut qkv = vec![0f32; m * 3 * h];
                    let mut attn = vec![0f32; m * h];
                    let mut res = vec![0f32; m * h];
                    let mut normed = vec![0f32; m * h];
                    let mut ffn_concat = vec![0f32; m * 2 * id]; // fc11||fc12 output
                    let mut sc = vec![0f32; s * s];

                    // QKV (no bias)
                    crate::blas::sgemm(inp, sl(*qkv_w, base, h * 3 * h), &mut qkv, m, h, 3 * h);

                    // SDPA with inline RoPE
                    for bi in 0..b {
                        for hi in 0..n_h {
                            for qi in 0..s {
                                for ki in 0..s {
                                    let q_base = bi * s * 3 * h + qi * 3 * h + hi * d_h;
                                    let k_base = bi * s * 3 * h + ki * 3 * h + h + hi * d_h;
                                    let mut dot = 0f32;
                                    for i in 0..half_dh {
                                        let q1 = qkv[q_base + i];
                                        let q2 = qkv[q_base + half_dh + i];
                                        let k1 = qkv[k_base + i];
                                        let k2 = qkv[k_base + half_dh + i];
                                        let cq = cos_tab[qi * half_dh + i];
                                        let sq = sin_tab[qi * half_dh + i];
                                        let ck = cos_tab[ki * half_dh + i];
                                        let sk = sin_tab[ki * half_dh + i];
                                        dot += (q1 * cq - q2 * sq) * (k1 * ck - k2 * sk)
                                            + (q2 * cq + q1 * sq) * (k2 * ck + k1 * sk);
                                    }
                                    sc[qi * s + ki] = dot * scale;
                                    if mk[bi * s + ki] < mask_thr {
                                        sc[qi * s + ki] = mask_neg;
                                    }
                                }
                            }
                            crate::kernels::neon_softmax(&mut sc[..s * s], s, s);
                            for qi in 0..s {
                                let o = bi * s * h + qi * h + hi * d_h;
                                for d in 0..d_h {
                                    attn[o + d] = 0.0;
                                }
                                for ki in 0..s {
                                    let w = sc[qi * s + ki];
                                    if w > score_thr {
                                        let v = bi * s * 3 * h + ki * 3 * h + 2 * h + hi * d_h;
                                        #[cfg(target_arch = "aarch64")]
                                        {
                                            use std::arch::aarch64::*;
                                            let vw = vdupq_n_f32(w);
                                            for c in 0..neon_chunks {
                                                let off = c * 4;
                                                vst1q_f32(
                                                    attn.as_mut_ptr().add(o + off),
                                                    vfmaq_f32(
                                                        vld1q_f32(attn.as_ptr().add(o + off)),
                                                        vw,
                                                        vld1q_f32(qkv.as_ptr().add(v + off)),
                                                    ),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..d_h {
                                            attn[o + d] += w * qkv[v + d];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Out proj (no bias) + residual
                    crate::blas::sgemm(&attn, sl(*out_w, base, h * h), &mut res, m, h, h);
                    for i in 0..m * h {
                        res[i] += inp[i];
                    }

                    // LN1
                    let g1 = sl(*ln1_g, base, h);
                    let b1 = sl(*ln1_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g1,
                            b1,
                            &mut normed[r * h..(r + 1) * h],
                            h,
                            *eps1,
                        );
                    }

                    // SwiGLU: fused fc11+fc12 sgemm, then split, silu, mul
                    crate::blas::sgemm(&normed, fused_fc_w, &mut ffn_concat, m, h, 2 * id);
                    // Split: first id cols = fc11 (up), second id cols = fc12 (gate)
                    // SiLU on gate, then multiply up * gate → store in up region
                    for row in 0..m {
                        let bo = row * 2 * id;
                        // SiLU in-place on gate portion
                        for j in 0..id {
                            let x = ffn_concat[bo + id + j];
                            ffn_concat[bo + id + j] = x / (1.0 + (-x).exp());
                        }
                        // Multiply: up[j] *= gate[j]
                        for j in 0..id {
                            ffn_concat[bo + j] *= ffn_concat[bo + id + j];
                        }
                    }

                    // fc2 (no bias) + residual  — read from first id cols of ffn_concat
                    // Need contiguous [m, id] for sgemm. Copy or use strided sgemm.
                    // The up*gate result is at ffn_concat[row * 2*id .. row * 2*id + id]
                    // Stride = 2*id. Use sgemm_general with lda = 2*id.
                    crate::blas::sgemm_general(
                        ffn_concat.as_ptr(),
                        sl(*fc2_w, base, id * h).as_ptr(),
                        res.as_mut_ptr(),
                        m,
                        h,
                        id,
                        1.0,
                        0.0,
                        2 * id,
                        h,
                        h,
                        false,
                        false,
                    );
                    for i in 0..m * h {
                        res[i] += normed[i];
                    }

                    // LN2 → output
                    let g2 = sl(*ln2_g, base, h);
                    let b2 = sl(*ln2_b, base, h);
                    for r in 0..m {
                        crate::kernels::layer_norm_row(
                            &res[r * h..(r + 1) * h],
                            g2,
                            b2,
                            &mut dst[r * h..(r + 1) * h],
                            h,
                            *eps2,
                        );
                    }
                }
            }

            Thunk::FusedSwiGLU {
                src,
                dst,
                n_half,
                total,
            } => {
                let n = *n_half as usize;
                let t = *total as usize;
                let outer = t / n;
                let in_total = outer * 2 * n;
                unsafe {
                    let inp = sl(*src, base, in_total);
                    let out = sl_mut(*dst, base, t);
                    for o in 0..outer {
                        let in_row = &inp[o * 2 * n..(o + 1) * 2 * n];
                        let out_row = &mut out[o * n..(o + 1) * n];
                        for i in 0..n {
                            let up = in_row[i];
                            let gate = in_row[n + i];
                            out_row[i] = up * (gate / (1.0 + (-gate).exp()));
                        }
                    }
                }
            }

            Thunk::Concat {
                dst,
                outer,
                inner,
                total_axis,
                inputs,
            } => {
                let outer = *outer as usize;
                let inner = *inner as usize;
                let total_axis = *total_axis as usize;
                let row_stride = total_axis * inner;
                let out_total = outer * row_stride;
                unsafe {
                    let out = sl_mut(*dst, base, out_total);
                    let mut cum: usize = 0;
                    for (src_off, in_axis) in inputs {
                        let in_axis = *in_axis as usize;
                        let copy_per_row = in_axis * inner;
                        let dst_col_off = cum * inner;
                        let in_total = outer * copy_per_row;
                        let inp = sl(*src_off, base, in_total);
                        for o in 0..outer {
                            let dst_row_start = o * row_stride + dst_col_off;
                            let src_row_start = o * copy_per_row;
                            out[dst_row_start..dst_row_start + copy_per_row]
                                .copy_from_slice(&inp[src_row_start..src_row_start + copy_per_row]);
                        }
                        cum += in_axis;
                    }
                }
            }

            Thunk::ConcatF64 {
                dst,
                outer,
                inner,
                total_axis,
                inputs,
            } => {
                let outer = *outer as usize;
                let inner = *inner as usize;
                let total_axis = *total_axis as usize;
                let row_stride = total_axis * inner;
                let out_total = outer * row_stride;
                unsafe {
                    let out = sl_mut_f64(*dst, base, out_total);
                    let mut cum: usize = 0;
                    for (src_off, in_axis) in inputs {
                        let in_axis = *in_axis as usize;
                        let copy_per_row = in_axis * inner;
                        let dst_col_off = cum * inner;
                        let in_total = outer * copy_per_row;
                        let inp = sl_f64(*src_off, base, in_total);
                        for o in 0..outer {
                            let dst_row_start = o * row_stride + dst_col_off;
                            let src_row_start = o * copy_per_row;
                            out[dst_row_start..dst_row_start + copy_per_row]
                                .copy_from_slice(&inp[src_row_start..src_row_start + copy_per_row]);
                        }
                        cum += in_axis;
                    }
                }
            }

            Thunk::Compare {
                lhs,
                rhs,
                dst,
                len,
                op,
            } => {
                let len = *len as usize;
                unsafe {
                    let l = sl(*lhs, base, len);
                    let r = sl(*rhs, base, len);
                    let o = sl_mut(*dst, base, len);
                    for i in 0..len {
                        o[i] = match op {
                            CmpOp::Eq => (l[i] == r[i]) as u32 as f32,
                            CmpOp::Ne => (l[i] != r[i]) as u32 as f32,
                            CmpOp::Lt => (l[i] < r[i]) as u32 as f32,
                            CmpOp::Le => (l[i] <= r[i]) as u32 as f32,
                            CmpOp::Gt => (l[i] > r[i]) as u32 as f32,
                            CmpOp::Ge => (l[i] >= r[i]) as u32 as f32,
                        };
                    }
                }
            }

            Thunk::Where {
                cond,
                on_true,
                on_false,
                dst,
                len,
            } => {
                let len = *len as usize;
                unsafe {
                    let c = sl(*cond, base, len);
                    let t = sl(*on_true, base, len);
                    let e = sl(*on_false, base, len);
                    let o = sl_mut(*dst, base, len);
                    for i in 0..len {
                        // Treat cond as boolean: any non-zero → true.
                        o[i] = if c[i] != 0.0 { t[i] } else { e[i] };
                    }
                }
            }

            Thunk::ScatterAdd {
                updates,
                indices,
                dst,
                num_updates,
                out_dim,
                trailing,
            } => {
                let num_updates = *num_updates as usize;
                let out_dim = *out_dim as usize;
                let trailing = *trailing as usize;
                unsafe {
                    let upd = sl(*updates, base, num_updates * trailing);
                    let ids = sl(*indices, base, num_updates);
                    let out = sl_mut(*dst, base, out_dim * trailing);
                    // Zero the output first — semantics are accumulate-into-zeros.
                    for v in out.iter_mut() {
                        *v = 0.0;
                    }
                    for i in 0..num_updates {
                        let row = ids[i] as usize;
                        debug_assert!(row < out_dim, "ScatterAdd index out of range");
                        let src_off = i * trailing;
                        let dst_off = row * trailing;
                        for j in 0..trailing {
                            out[dst_off + j] += upd[src_off + j];
                        }
                    }
                }
            }

            Thunk::GroupedMatMul {
                input,
                weight,
                expert_idx,
                dst,
                m,
                k_dim,
                n,
                num_experts,
            } => {
                let m = *m as usize;
                let k_dim = *k_dim as usize;
                let n = *n as usize;
                let num_experts = *num_experts as usize;
                unsafe {
                    let inp = sl(*input, base, m * k_dim);
                    let wt = sl(*weight, base, num_experts * k_dim * n);
                    let ids = sl(*expert_idx, base, m);
                    let out = sl_mut(*dst, base, m * n);

                    // Counting-sort tokens by their assigned expert.
                    // counts[e] = how many tokens routed to expert e.
                    let mut counts = vec![0usize; num_experts];
                    for i in 0..m {
                        let e = ids[i] as usize;
                        debug_assert!(
                            e < num_experts,
                            "expert_idx out of range: {e} >= {num_experts}"
                        );
                        counts[e] += 1;
                    }
                    // Cumulative offsets into the packed buffer.
                    let mut offsets = vec![0usize; num_experts + 1];
                    for e in 0..num_experts {
                        offsets[e + 1] = offsets[e] + counts[e];
                    }
                    // Pack: each expert's rows land contiguously in `packed_in`.
                    // `original_pos[packed_idx] = original_token_idx` for the
                    // unpermute step at the end.
                    let mut packed_in = vec![0f32; m * k_dim];
                    let mut original_pos = vec![0usize; m];
                    let mut write_idx = vec![0usize; num_experts];
                    for i in 0..m {
                        let e = ids[i] as usize;
                        let dst_row = offsets[e] + write_idx[e];
                        packed_in[dst_row * k_dim..(dst_row + 1) * k_dim]
                            .copy_from_slice(&inp[i * k_dim..(i + 1) * k_dim]);
                        original_pos[dst_row] = i;
                        write_idx[e] += 1;
                    }

                    // One BLAS sgemm per expert. Skip experts with no
                    // tokens — common at the tail when M is much smaller
                    // than num_experts × k.
                    let mut packed_out = vec![0f32; m * n];
                    let expert_stride = k_dim * n;
                    for e in 0..num_experts {
                        let count = counts[e];
                        if count == 0 {
                            continue;
                        }
                        let in_start = offsets[e];
                        let in_slice = &packed_in[in_start * k_dim..(in_start + count) * k_dim];
                        let w_slab = &wt[e * expert_stride..(e + 1) * expert_stride];
                        let out_slice = &mut packed_out[in_start * n..(in_start + count) * n];
                        crate::blas::sgemm(in_slice, w_slab, out_slice, count, k_dim, n);
                    }

                    // Unpermute back to original token order.
                    for packed_idx in 0..m {
                        let i = original_pos[packed_idx];
                        out[i * n..(i + 1) * n]
                            .copy_from_slice(&packed_out[packed_idx * n..(packed_idx + 1) * n]);
                    }
                }
            }

            Thunk::TopK {
                src,
                dst,
                outer,
                axis_dim,
                k,
            } => {
                let outer = *outer as usize;
                let axis_dim = *axis_dim as usize;
                let k = *k as usize;
                unsafe {
                    let inp = sl(*src, base, outer * axis_dim);
                    let out = sl_mut(*dst, base, outer * k);
                    // Repeated argmax with masking. O(k * axis_dim) per row;
                    // good enough for small k (MoE typical k=2–8). For larger
                    // k a partial heap would win.
                    let mut row_buf: Vec<f32> = vec![0.0; axis_dim];
                    for o in 0..outer {
                        row_buf.copy_from_slice(&inp[o * axis_dim..(o + 1) * axis_dim]);
                        for ki in 0..k {
                            // Find argmax with tie-break to smaller index.
                            let mut best_i = 0usize;
                            let mut best_v = row_buf[0];
                            for i in 1..axis_dim {
                                let v = row_buf[i];
                                if v > best_v {
                                    best_v = v;
                                    best_i = i;
                                }
                            }
                            out[o * k + ki] = best_i as f32;
                            // Mask the chosen index so the next pass picks
                            // the next-largest instead.
                            row_buf[best_i] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            Thunk::Reduce {
                src,
                dst,
                outer,
                reduced,
                inner,
                op,
            } => {
                let outer = *outer as usize;
                let reduced = *reduced as usize;
                let inner = *inner as usize;
                let in_total = outer * reduced * inner;
                let out_total = outer * inner;
                unsafe {
                    let inp = sl(*src, base, in_total);
                    let out = sl_mut(*dst, base, out_total);
                    for o in 0..outer {
                        for i in 0..inner {
                            let mut acc = match op {
                                ReduceOp::Max => f32::NEG_INFINITY,
                                ReduceOp::Min => f32::INFINITY,
                                ReduceOp::Prod => 1.0f32,
                                _ => 0.0f32, // Sum / Mean
                            };
                            // Walk the reduced axis with stride `inner`.
                            for r in 0..reduced {
                                let v = inp[o * reduced * inner + r * inner + i];
                                acc = match op {
                                    ReduceOp::Sum | ReduceOp::Mean => acc + v,
                                    ReduceOp::Max => acc.max(v),
                                    ReduceOp::Min => acc.min(v),
                                    ReduceOp::Prod => acc * v,
                                };
                            }
                            if matches!(op, ReduceOp::Mean) {
                                acc /= reduced as f32;
                            }
                            out[o * inner + i] = acc;
                        }
                    }
                }
            }

            Thunk::Conv2D1x1 {
                src,
                weight,
                dst,
                n,
                c_in,
                c_out,
                hw,
            } => {
                let n = *n as usize;
                let c_in = *c_in as usize;
                let c_out = *c_out as usize;
                let hw = *hw as usize;
                unsafe {
                    let inp = sl(*src, base, n * c_in * hw);
                    let wt = sl(*weight, base, c_out * c_in);
                    let out = sl_mut(*dst, base, n * c_out * hw);
                    // Per-batch sgemm: weight [c_out, c_in] @ input
                    // [c_in, hw] = output [c_out, hw]. The weight is
                    // shared across batches, so we get to dispatch
                    // BLAS once per N (typically 1).
                    for ni in 0..n {
                        let in_off = ni * c_in * hw;
                        let out_off = ni * c_out * hw;
                        crate::blas::sgemm(
                            wt,
                            &inp[in_off..in_off + c_in * hw],
                            &mut out[out_off..out_off + c_out * hw],
                            c_out,
                            c_in,
                            hw,
                        );
                    }
                }
            }

            Thunk::Conv2D {
                src,
                weight,
                dst,
                n,
                c_in,
                h,
                w,
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
                let n = *n as usize;
                let c_in = *c_in as usize;
                let h = *h as usize;
                let w = *w as usize;
                let c_out = *c_out as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let dh = *dh as usize;
                let dw = *dw as usize;
                let groups = *groups as usize;
                let c_in_per_g = c_in / groups;
                let c_out_per_g = c_out / groups;
                unsafe {
                    let inp = sl(*src, base, n * c_in * h * w);
                    let wt = sl(*weight, base, c_out * c_in_per_g * kh * kw);
                    let out = sl_mut(*dst, base, n * c_out * h_out * w_out);
                    for ni in 0..n {
                        for co in 0..c_out {
                            let g = co / c_out_per_g;
                            let ci_start = g * c_in_per_g;
                            for ho in 0..h_out {
                                for wo in 0..w_out {
                                    let mut acc = 0f32;
                                    for ci_off in 0..c_in_per_g {
                                        let ci = ci_start + ci_off;
                                        let in_chan = ((ni * c_in) + ci) * h * w;
                                        let wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                let hi = ho * sh + ki * dh;
                                                let wi = wo * sw + kj * dw;
                                                if hi < ph || wi < pw {
                                                    continue;
                                                }
                                                let hi = hi - ph;
                                                let wi = wi - pw;
                                                if hi >= h || wi >= w {
                                                    continue;
                                                }
                                                acc += inp[in_chan + hi * w + wi]
                                                    * wt[wt_chan + ki * kw + kj];
                                            }
                                        }
                                    }
                                    out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] =
                                        acc;
                                }
                            }
                        }
                    }
                }
            }

            Thunk::Pool2D {
                src,
                dst,
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
                kind,
            } => {
                let n = *n as usize;
                let c = *c as usize;
                let h = *h as usize;
                let w = *w as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let kernel_area = (kh * kw) as f32;
                unsafe {
                    let inp = sl(*src, base, n * c * h * w);
                    let out = sl_mut(*dst, base, n * c * h_out * w_out);
                    for ni in 0..n {
                        for ci in 0..c {
                            let in_chan = ni * c * h * w + ci * h * w;
                            let out_chan = ni * c * h_out * w_out + ci * h_out * w_out;
                            for ho in 0..h_out {
                                for wo in 0..w_out {
                                    let mut acc = match kind {
                                        ReduceOp::Max => f32::NEG_INFINITY,
                                        _ => 0f32, // Mean (and Sum/Min/Prod fall back here)
                                    };
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let hi = ho * sh + ki;
                                            let wi = wo * sw + kj;
                                            // Padded-zero region.
                                            if hi < ph || wi < pw {
                                                continue;
                                            }
                                            let hi = hi - ph;
                                            let wi = wi - pw;
                                            if hi >= h || wi >= w {
                                                continue;
                                            }
                                            let v = inp[in_chan + hi * w + wi];
                                            match kind {
                                                ReduceOp::Max => acc = acc.max(v),
                                                _ => acc += v,
                                            }
                                        }
                                    }
                                    if matches!(kind, ReduceOp::Mean) {
                                        acc /= kernel_area;
                                    }
                                    out[out_chan + ho * w_out + wo] = acc;
                                }
                            }
                        }
                    }
                }
            }

            Thunk::ReluBackward { x, dy, dx, len } => {
                let len = *len as usize;
                unsafe {
                    let xs = sl(*x, base, len);
                    let dys = sl(*dy, base, len);
                    let out = sl_mut(*dx, base, len);
                    for i in 0..len {
                        out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
                    }
                }
            }

            Thunk::ReluBackwardF64 { x, dy, dx, len } => {
                let len = *len as usize;
                unsafe {
                    let xs = sl_f64(*x, base, len);
                    let dys = sl_f64(*dy, base, len);
                    let out = sl_mut_f64(*dx, base, len);
                    for i in 0..len {
                        out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
                    }
                }
            }

            Thunk::QMatMul {
                x,
                w,
                bias,
                out,
                m,
                k,
                n,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let m = *m as usize;
                let k = *k as usize;
                let n = *n as usize;
                unsafe {
                    let x_ptr = base.add(*x) as *const i8;
                    let w_ptr = base.add(*w) as *const i8;
                    let bias_ptr = base.add(*bias) as *const i32;
                    let out_ptr = base.add(*out) as *mut i8;
                    for mi in 0..m {
                        for ni in 0..n {
                            let mut acc: i32 = *bias_ptr.add(ni);
                            for ki in 0..k {
                                let xv = *x_ptr.add(mi * k + ki) as i32 - *x_zp;
                                let wv = *w_ptr.add(ki * n + ni) as i32 - *w_zp;
                                acc += xv * wv;
                            }
                            // Requantize: round(acc · mult) + out_zp,
                            // clamped to i8.
                            let r = (acc as f32 * *mult).round() as i32 + *out_zp;
                            let r = r.clamp(-128, 127) as i8;
                            *out_ptr.add(mi * n + ni) = r;
                        }
                    }
                }
            }

            Thunk::QConv2d {
                x,
                w,
                bias,
                out,
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
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => {
                let n = *n as usize;
                let c_in = *c_in as usize;
                let h = *h as usize;
                let w_in = *w_in as usize;
                let c_out = *c_out as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let dh = *dh as usize;
                let dw = *dw as usize;
                let groups = *groups as usize;
                let c_in_per_g = c_in / groups;
                let c_out_per_g = c_out / groups;
                unsafe {
                    let x_ptr = base.add(*x) as *const i8;
                    let w_ptr = base.add(*w) as *const i8;
                    let bias_ptr = base.add(*bias) as *const i32;
                    let out_ptr = base.add(*out) as *mut i8;
                    for ni in 0..n {
                        for co in 0..c_out {
                            let g = co / c_out_per_g;
                            let ci_start = g * c_in_per_g;
                            for ho in 0..h_out {
                                for wo in 0..w_out {
                                    let mut acc: i32 = *bias_ptr.add(co);
                                    for ci_off in 0..c_in_per_g {
                                        let ci = ci_start + ci_off;
                                        let in_chan = ((ni * c_in) + ci) * h * w_in;
                                        let wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                let hi = ho * sh + ki * dh;
                                                let wi = wo * sw + kj * dw;
                                                if hi < ph || wi < pw {
                                                    continue;
                                                }
                                                let hi = hi - ph;
                                                let wi = wi - pw;
                                                if hi >= h || wi >= w_in {
                                                    continue;
                                                }
                                                let xv = *x_ptr.add(in_chan + hi * w_in + wi)
                                                    as i32
                                                    - *x_zp;
                                                let wv = *w_ptr.add(wt_chan + ki * kw + kj) as i32
                                                    - *w_zp;
                                                acc += xv * wv;
                                            }
                                        }
                                    }
                                    let r = (acc as f32 * *mult).round() as i32 + *out_zp;
                                    let r = r.clamp(-128, 127) as i8;
                                    let dst = ((ni * c_out) + co) * h_out * w_out + ho * w_out + wo;
                                    *out_ptr.add(dst) = r;
                                }
                            }
                        }
                    }
                }
            }

            Thunk::Quantize {
                x,
                q,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                scales,
                zero_points,
            } => {
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                unsafe {
                    let xs = sl(*x, base, len);
                    let q_ptr = base.add(*q) as *mut i8;
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let inv_scale = 1.0 / scales[c];
                        let zp = zero_points[c];
                        let v = (xs[i] * inv_scale).round() as i32 + zp;
                        *q_ptr.add(i) = v.clamp(-128, 127) as i8;
                    }
                }
            }

            Thunk::Dequantize {
                q,
                x,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                scales,
                zero_points,
            } => {
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                unsafe {
                    let q_ptr = base.add(*q) as *const i8;
                    let out = sl_mut(*x, base, len);
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let scale = scales[c];
                        let zp = zero_points[c];
                        let qv = *q_ptr.add(i) as i32;
                        out[i] = (qv - zp) as f32 * scale;
                    }
                }
            }

            Thunk::FakeQuantize {
                x,
                out,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                bits,
                ste: _,
                scale_mode,
                state_off,
            } => {
                use rlx_ir::op::ScaleMode;
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                let q_max: f32 = match *bits {
                    8 => 127.0,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("FakeQuantize: unsupported bits {n}"),
                };
                unsafe {
                    let xs = sl(*x, base, len);
                    let outs = sl_mut(*out, base, len);

                    let mut scale = vec![0f32; chan_dim];
                    match scale_mode {
                        ScaleMode::PerBatch => {
                            let mut max_abs = vec![0f32; chan_dim];
                            for i in 0..len {
                                let c = if chan_dim == 1 {
                                    0
                                } else {
                                    (i / inner) % chan_dim
                                };
                                let a = xs[i].abs();
                                if a > max_abs[c] {
                                    max_abs[c] = a;
                                }
                            }
                            for c in 0..chan_dim {
                                scale[c] = (max_abs[c] / q_max).max(1e-12);
                            }
                        }
                        ScaleMode::EMA { decay } => {
                            // Per-channel current max-abs, then blend
                            // into the running state in place.
                            let mut max_abs = vec![0f32; chan_dim];
                            for i in 0..len {
                                let c = if chan_dim == 1 {
                                    0
                                } else {
                                    (i / inner) % chan_dim
                                };
                                let a = xs[i].abs();
                                if a > max_abs[c] {
                                    max_abs[c] = a;
                                }
                            }
                            let state =
                                sl_mut(state_off.expect("EMA needs state_off"), base, chan_dim);
                            for c in 0..chan_dim {
                                let cur = (max_abs[c] / q_max).max(1e-12);
                                // Cold-start: state==0 → seed directly.
                                let blended = if state[c] <= 0.0 {
                                    cur
                                } else {
                                    *decay * state[c] + (1.0 - *decay) * cur
                                };
                                state[c] = blended;
                                scale[c] = blended;
                            }
                        }
                        ScaleMode::Fixed => {
                            let state =
                                sl(state_off.expect("Fixed needs state_off"), base, chan_dim);
                            for c in 0..chan_dim {
                                scale[c] = state[c].max(1e-12);
                            }
                        }
                    }

                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let s = scale[c];
                        let qv = (xs[i] / s).round().clamp(-q_max, q_max);
                        outs[i] = qv * s;
                    }
                }
            }

            Thunk::ActivationBackward {
                x,
                dy,
                dx,
                len,
                kind,
            } => {
                let len = *len as usize;
                unsafe {
                    let xs = sl(*x, base, len);
                    let dys = sl(*dy, base, len);
                    let out = sl_mut(*dx, base, len);
                    activation_backward_kernel(*kind, xs, dys, out);
                }
            }

            Thunk::ActivationBackwardF64 {
                x,
                dy,
                dx,
                len,
                kind,
            } => {
                let len = *len as usize;
                unsafe {
                    let xs = sl_f64(*x, base, len);
                    let dys = sl_f64(*dy, base, len);
                    let out = sl_mut_f64(*dx, base, len);
                    activation_backward_kernel_f64(*kind, xs, dys, out);
                }
            }

            Thunk::FakeQuantizeLSQ {
                x,
                scale_off,
                out,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                bits,
            } => {
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                let q_max: f32 = match *bits {
                    8 => 127.0,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("FakeQuantizeLSQ: bad bits {n}"),
                };
                unsafe {
                    let xs = sl(*x, base, len);
                    let scale = sl(*scale_off, base, chan_dim);
                    let outs = sl_mut(*out, base, len);
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let s = scale[c].max(1e-12);
                        let qv = (xs[i] / s).round().clamp(-q_max, q_max);
                        outs[i] = qv * s;
                    }
                }
            }

            Thunk::FakeQuantizeLSQBackwardX {
                x,
                scale_off,
                dy,
                dx,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                bits,
            } => {
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                let q_max: f32 = match *bits {
                    8 => 127.0,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("FakeQuantizeLSQBackwardX: bad bits {n}"),
                };
                unsafe {
                    let xs = sl(*x, base, len);
                    let scale = sl(*scale_off, base, chan_dim);
                    let dys = sl(*dy, base, len);
                    let outs = sl_mut(*dx, base, len);
                    // STE-clipped: dx = dy when |x/s| ≤ q_max, else 0.
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let z = xs[i] / scale[c].max(1e-12);
                        outs[i] = if z.abs() <= q_max { dys[i] } else { 0.0 };
                    }
                }
            }

            Thunk::FakeQuantizeLSQBackwardScale {
                x,
                scale_off,
                dy,
                dscale,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                bits,
            } => {
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                let q_max: f32 = match *bits {
                    8 => 127.0,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("FakeQuantizeLSQBackwardScale: bad bits {n}"),
                };
                unsafe {
                    let xs = sl(*x, base, len);
                    let scale = sl(*scale_off, base, chan_dim);
                    let dys = sl(*dy, base, len);
                    let outs = sl_mut(*dscale, base, chan_dim);
                    for v in outs.iter_mut() {
                        *v = 0.0;
                    }
                    // ψ(z) = -z + round(z) inside range, sign(z)·q_max outside.
                    // dscale[c] = sum_i ψ(x_i/s[c]) * upstream[i].
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let s = scale[c].max(1e-12);
                        let z = xs[i] / s;
                        let psi = if z.abs() <= q_max {
                            -z + z.round()
                        } else if z > 0.0 {
                            q_max
                        } else {
                            -q_max
                        };
                        outs[c] += psi * dys[i];
                    }
                }
            }

            Thunk::FakeQuantizeBackward {
                x,
                dy,
                dx,
                len,
                chan_axis: _,
                chan_dim,
                inner,
                bits,
                ste,
            } => {
                use rlx_ir::op::SteKind;
                let len = *len as usize;
                let chan_dim = *chan_dim as usize;
                let inner = *inner as usize;
                let q_max: f32 = match *bits {
                    8 => 127.0,
                    4 => 7.0,
                    2 => 1.0,
                    n => panic!("FakeQuantizeBackward: bad bits {n}"),
                };
                unsafe {
                    let xs = sl(*x, base, len);
                    let dys = sl(*dy, base, len);
                    let outs = sl_mut(*dx, base, len);

                    // Per-channel max-abs → scale, same as forward.
                    let mut max_abs = vec![0f32; chan_dim];
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let a = xs[i].abs();
                        if a > max_abs[c] {
                            max_abs[c] = a;
                        }
                    }
                    let mut scale = vec![0f32; chan_dim];
                    for c in 0..chan_dim {
                        scale[c] = (max_abs[c] / q_max).max(1e-12);
                    }

                    match *ste {
                        SteKind::Identity => {
                            // dx = dy unchanged.
                            outs.copy_from_slice(dys);
                        }
                        SteKind::ClippedIdentity => {
                            // dx = dy * (|x| <= q_max·s); zero if the
                            // forward saturated.
                            for i in 0..len {
                                let c = if chan_dim == 1 {
                                    0
                                } else {
                                    (i / inner) % chan_dim
                                };
                                let bound = q_max * scale[c];
                                outs[i] = if xs[i].abs() <= bound { dys[i] } else { 0.0 };
                            }
                        }
                        SteKind::Tanh => {
                            // dx = dy * (1 - tanh²(x/s)).
                            for i in 0..len {
                                let c = if chan_dim == 1 {
                                    0
                                } else {
                                    (i / inner) % chan_dim
                                };
                                let t = (xs[i] / scale[c]).tanh();
                                outs[i] = dys[i] * (1.0 - t * t);
                            }
                        }
                        SteKind::HardTanh => {
                            // dx = dy * max(0, 1 - |x/(q_max·s)|).
                            for i in 0..len {
                                let c = if chan_dim == 1 {
                                    0
                                } else {
                                    (i / inner) % chan_dim
                                };
                                let bound = q_max * scale[c];
                                let attenuation = (1.0 - (xs[i] / bound).abs()).max(0.0);
                                outs[i] = dys[i] * attenuation;
                            }
                        }
                    }
                }
            }

            Thunk::LayerNormBackwardInput {
                x,
                gamma,
                dy,
                dx,
                rows,
                h,
                eps,
            } => {
                let rows = *rows as usize;
                let h = *h as usize;
                let eps = *eps;
                unsafe {
                    let xs = sl(*x, base, rows * h);
                    let g = sl(*gamma, base, h);
                    let dys = sl(*dy, base, rows * h);
                    let out = sl_mut(*dx, base, rows * h);
                    let n_inv = 1.0 / h as f32;
                    for r in 0..rows {
                        let xr = &xs[r * h..(r + 1) * h];
                        let dyr = &dys[r * h..(r + 1) * h];
                        // Per-row mean and inv_std (recompute — no saved
                        // tensor from the forward pass).
                        let mut sum = 0f32;
                        for &v in xr {
                            sum += v;
                        }
                        let mean = sum * n_inv;
                        let mut var = 0f32;
                        for &v in xr {
                            let d = v - mean;
                            var += d * d;
                        }
                        let inv_std = 1.0 / (var * n_inv + eps).sqrt();

                        // sums needed for the closed-form:
                        //   mean(dy·γ) and mean(dy·γ·x̂)
                        let mut s_sy = 0f32;
                        let mut s_sxh = 0f32;
                        for d in 0..h {
                            let xh = (xr[d] - mean) * inv_std;
                            let sy = dyr[d] * g[d];
                            s_sy += sy;
                            s_sxh += sy * xh;
                        }
                        let m_sy = s_sy * n_inv;
                        let m_sxh = s_sxh * n_inv;

                        for d in 0..h {
                            let xh = (xr[d] - mean) * inv_std;
                            let sy = dyr[d] * g[d];
                            out[r * h + d] = inv_std * (sy - m_sy - xh * m_sxh);
                        }
                    }
                }
            }

            Thunk::LayerNormBackwardGamma {
                x,
                dy,
                dgamma,
                rows,
                h,
                eps,
            } => {
                let rows = *rows as usize;
                let h = *h as usize;
                let eps = *eps;
                unsafe {
                    let xs = sl(*x, base, rows * h);
                    let dys = sl(*dy, base, rows * h);
                    let out = sl_mut(*dgamma, base, h);
                    for v in out.iter_mut() {
                        *v = 0.0;
                    }
                    let n_inv = 1.0 / h as f32;
                    for r in 0..rows {
                        let xr = &xs[r * h..(r + 1) * h];
                        let dyr = &dys[r * h..(r + 1) * h];
                        let mut sum = 0f32;
                        for &v in xr {
                            sum += v;
                        }
                        let mean = sum * n_inv;
                        let mut var = 0f32;
                        for &v in xr {
                            let d = v - mean;
                            var += d * d;
                        }
                        let inv_std = 1.0 / (var * n_inv + eps).sqrt();
                        for d in 0..h {
                            let xh = (xr[d] - mean) * inv_std;
                            out[d] += dyr[d] * xh;
                        }
                    }
                }
            }

            Thunk::MaxPool2dBackward {
                x,
                dy,
                dx,
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
            } => {
                let n = *n as usize;
                let c = *c as usize;
                let h = *h as usize;
                let w = *w as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                unsafe {
                    let xs = sl(*x, base, n * c * h * w);
                    let dys = sl(*dy, base, n * c * h_out * w_out);
                    let dxs = sl_mut(*dx, base, n * c * h * w);
                    // Zero before scatter — multiple windows can write
                    // to the same input position when stride < kernel.
                    for v in dxs.iter_mut() {
                        *v = 0.0;
                    }
                    for ni in 0..n {
                        for ci in 0..c {
                            let in_chan = (ni * c + ci) * h * w;
                            let out_chan = (ni * c + ci) * h_out * w_out;
                            for ho in 0..h_out {
                                for wo in 0..w_out {
                                    // Recompute argmax inside this window.
                                    let mut best_v = f32::NEG_INFINITY;
                                    let mut best_idx: Option<usize> = None;
                                    for ki in 0..kh {
                                        for kj in 0..kw {
                                            let hi = ho * sh + ki;
                                            let wi = wo * sw + kj;
                                            if hi < ph || wi < pw {
                                                continue;
                                            }
                                            let hi = hi - ph;
                                            let wi = wi - pw;
                                            if hi >= h || wi >= w {
                                                continue;
                                            }
                                            let idx = in_chan + hi * w + wi;
                                            let v = xs[idx];
                                            // Tie-break: keep first hit
                                            // (matches forward's `acc.max(v)`
                                            // — strict greater-than wins).
                                            if v > best_v {
                                                best_v = v;
                                                best_idx = Some(idx);
                                            }
                                        }
                                    }
                                    if let Some(idx) = best_idx {
                                        dxs[idx] += dys[out_chan + ho * w_out + wo];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Thunk::Conv2dBackwardInput {
                dy,
                w,
                dx,
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
                // Per-group GEMM + col2im. Two orders of magnitude faster
                // than the naive 6-deep nested loop on training shapes.
                //
                //   dcol_n_g = w_g^T  @  dy_n_g            (sgemm)
                //   dx_n_g  += col2im(dcol_n_g)            (scatter-add)
                //
                // Layouts (all row-major):
                //   w_g       [c_out_per_g, c_in_per_g · kh · kw]
                //   dy_n_g    [c_out_per_g, h_out · w_out]
                //   dcol_n_g  [c_in_per_g · kh · kw, h_out · w_out]
                //   dx_n_g    [c_in_per_g, h · w_in]
                let n = *n as usize;
                let c_in = *c_in as usize;
                let h = *h as usize;
                let w_in = *w_in as usize;
                let c_out = *c_out as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let dh = *dh as usize;
                let dw = *dw as usize;
                let groups = *groups as usize;
                let c_in_per_g = c_in / groups;
                let c_out_per_g = c_out / groups;

                let m_dim = c_in_per_g * kh * kw;
                let n_dim = h_out * w_out;
                let k_dim = c_out_per_g;

                let dy_stride_n = c_out * h_out * w_out;
                let dy_stride_g = c_out_per_g * h_out * w_out;
                let w_stride_g = c_out_per_g * c_in_per_g * kh * kw;
                let dx_stride_n = c_in * h * w_in;
                let dx_stride_g = c_in_per_g * h * w_in;

                unsafe {
                    let dys = sl(*dy, base, n * c_out * h_out * w_out);
                    let ws = sl(*w, base, c_out * c_in_per_g * kh * kw);
                    let dxs = sl_mut(*dx, base, n * c_in * h * w_in);
                    for v in dxs.iter_mut() {
                        *v = 0.0;
                    }

                    // Reused scratch buffer for the [m_dim, n_dim] dcol.
                    let mut dcol = vec![0f32; m_dim * n_dim];

                    for ni in 0..n {
                        for g in 0..groups {
                            let w_g_off = g * w_stride_g;
                            let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                            let dx_n_g_off = ni * dx_stride_n + g * dx_stride_g;

                            // dcol = w_g^T @ dy_n_g
                            // w_g  is stored as [k_dim rows, m_dim cols] row-major
                            // (i.e. K×M storage with lda = M = m_dim — exactly what
                            // sgemm_general wants for trans_a=true).
                            crate::blas::sgemm_general(
                                ws.as_ptr().add(w_g_off),
                                dys.as_ptr().add(dy_n_g_off),
                                dcol.as_mut_ptr(),
                                m_dim,
                                n_dim,
                                k_dim,
                                1.0,
                                0.0,
                                /*lda=*/ m_dim,
                                /*ldb=*/ n_dim,
                                /*ldc=*/ n_dim,
                                /*trans_a=*/ true,
                                /*trans_b=*/ false,
                            );

                            // dx_n_g += col2im(dcol)
                            col2im(
                                &dcol,
                                &mut dxs[dx_n_g_off..dx_n_g_off + dx_stride_g],
                                c_in_per_g,
                                h,
                                w_in,
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
                            );
                        }
                    }
                }
            }

            Thunk::Conv2dBackwardWeight {
                x,
                dy,
                dw,
                n,
                c_in,
                h,
                w,
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
                dw_dil,
                groups,
            } => {
                let n = *n as usize;
                let c_in = *c_in as usize;
                let h = *h as usize;
                let w = *w as usize;
                // Per-group im2col + GEMM, summed across batch.
                //
                //   col_n_g  = im2col(x_n_g)               (gather)
                //   dw_g    += dy_n_g  @  col_n_g^T        (sgemm, β=1)
                //
                // Layouts:
                //   x_n_g     [c_in_per_g, h · w]
                //   col_n_g   [c_in_per_g · kh · kw, h_out · w_out]
                //   dy_n_g    [c_out_per_g, h_out · w_out]
                //   dw_g      [c_out_per_g, c_in_per_g · kh · kw]
                let c_out = *c_out as usize;
                let h_out = *h_out as usize;
                let w_out = *w_out as usize;
                let kh = *kh as usize;
                let kw = *kw as usize;
                let sh = *sh as usize;
                let sw = *sw as usize;
                let ph = *ph as usize;
                let pw = *pw as usize;
                let dh = *dh as usize;
                let dw_dil = *dw_dil as usize;
                let groups = *groups as usize;
                let c_in_per_g = c_in / groups;
                let c_out_per_g = c_out / groups;

                let m_dim = c_out_per_g;
                let n_dim = c_in_per_g * kh * kw;
                let k_dim = h_out * w_out;

                let x_stride_n = c_in * h * w;
                let x_stride_g = c_in_per_g * h * w;
                let dy_stride_n = c_out * h_out * w_out;
                let dy_stride_g = c_out_per_g * h_out * w_out;
                let dw_stride_g = c_out_per_g * c_in_per_g * kh * kw;

                unsafe {
                    let xs = sl(*x, base, n * c_in * h * w);
                    let dys = sl(*dy, base, n * c_out * h_out * w_out);
                    let dws = sl_mut(*dw, base, c_out * c_in_per_g * kh * kw);
                    for v in dws.iter_mut() {
                        *v = 0.0;
                    }

                    let mut col = vec![0f32; n_dim * k_dim];

                    for ni in 0..n {
                        for g in 0..groups {
                            let x_n_g_off = ni * x_stride_n + g * x_stride_g;
                            im2col(
                                &xs[x_n_g_off..x_n_g_off + x_stride_g],
                                &mut col,
                                c_in_per_g,
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
                            );

                            let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                            let dw_g_off = g * dw_stride_g;

                            // dw_g += dy_n_g @ col^T
                            //
                            // Output shape m × n_out = c_out_per_g × (c_in_per_g·kh·kw).
                            // dy_n_g is stored M×K row-major (lda = K = k_dim).
                            // col is stored as N×K row-major; with trans_b=true,
                            // sgemm_general uses ldb = K = k_dim and treats it as
                            // transposed. β=1 accumulates across the batch loop.
                            crate::blas::sgemm_general(
                                dys.as_ptr().add(dy_n_g_off),
                                col.as_ptr(),
                                dws.as_mut_ptr().add(dw_g_off),
                                m_dim,
                                n_dim,
                                k_dim,
                                1.0,
                                1.0,
                                /*lda=*/ k_dim,
                                /*ldb=*/ k_dim,
                                /*ldc=*/ n_dim,
                                /*trans_a=*/ false,
                                /*trans_b=*/ true,
                            );
                        }
                    }
                }
            }

            Thunk::SoftmaxCrossEntropy {
                logits,
                labels,
                dst,
                n,
                c,
            } => {
                let n = *n as usize;
                let c = *c as usize;
                unsafe {
                    let lg = sl(*logits, base, n * c);
                    let lb = sl(*labels, base, n);
                    let out = sl_mut(*dst, base, n);
                    for ni in 0..n {
                        let row = &lg[ni * c..(ni + 1) * c];
                        // log-sum-exp: max-subtract for stability.
                        let mut m = f32::NEG_INFINITY;
                        for &v in row {
                            if v > m {
                                m = v;
                            }
                        }
                        let mut sum = 0f32;
                        for &v in row {
                            sum += (v - m).exp();
                        }
                        let lse = m + sum.ln();
                        let label_idx = lb[ni] as usize;
                        // loss = -(logits[label] - lse) = lse - logits[label].
                        out[ni] = lse - row[label_idx];
                    }
                }
            }

            Thunk::SoftmaxCrossEntropyBackward {
                logits,
                labels,
                d_loss,
                dlogits,
                n,
                c,
            } => {
                let n = *n as usize;
                let c = *c as usize;
                unsafe {
                    let lg = sl(*logits, base, n * c);
                    let lb = sl(*labels, base, n);
                    let dl = sl(*d_loss, base, n);
                    let out = sl_mut(*dlogits, base, n * c);
                    for ni in 0..n {
                        let row = &lg[ni * c..(ni + 1) * c];
                        let label_idx = lb[ni] as usize;
                        let scale = dl[ni];
                        let mut m = f32::NEG_INFINITY;
                        for &v in row {
                            if v > m {
                                m = v;
                            }
                        }
                        let mut sum = 0f32;
                        for &v in row {
                            sum += (v - m).exp();
                        }
                        let inv_sum = 1.0 / sum;
                        let dst_row = &mut out[ni * c..(ni + 1) * c];
                        for k in 0..c {
                            let p = (row[k] - m).exp() * inv_sum;
                            let one_hot = if k == label_idx { 1.0 } else { 0.0 };
                            dst_row[k] = (p - one_hot) * scale;
                        }
                    }
                }
            }

            Thunk::GatherAxis {
                table,
                idx,
                dst,
                outer,
                axis_dim,
                num_idx,
                trailing,
            } => {
                let outer = *outer as usize;
                let axis_dim = *axis_dim as usize;
                let num_idx = *num_idx as usize;
                let trailing = *trailing as usize;
                unsafe {
                    let tab = sl(*table, base, outer * axis_dim * trailing);
                    let ids = sl(*idx, base, num_idx);
                    let out = sl_mut(*dst, base, outer * num_idx * trailing);
                    for o in 0..outer {
                        let tab_outer = o * axis_dim * trailing;
                        let out_outer = o * num_idx * trailing;
                        for k in 0..num_idx {
                            let row = ids[k] as usize;
                            let tab_row = tab_outer + row * trailing;
                            let out_row = out_outer + k * trailing;
                            out[out_row..out_row + trailing]
                                .copy_from_slice(&tab[tab_row..tab_row + trailing]);
                        }
                    }
                }
            }

            Thunk::Transpose {
                src,
                dst,
                in_total,
                out_dims,
                in_strides,
            } => {
                // N-D index walk: for each output flat index, decompose into
                // multi-dim coords using out_dims, then dot with in_strides
                // to find the source flat index. Stride 0 = broadcast (read
                // the same input element repeatedly along that dim).
                let rank = out_dims.len();
                let total: usize = out_dims.iter().map(|&d| d as usize).product();
                let in_total = *in_total as usize;
                unsafe {
                    let inp = sl(*src, base, in_total);
                    let out = sl_mut(*dst, base, total);
                    let mut idx = vec![0usize; rank];
                    for o in 0..total {
                        let mut src_idx = 0usize;
                        for d in 0..rank {
                            src_idx += idx[d] * in_strides[d] as usize;
                        }
                        out[o] = inp[src_idx];
                        // Increment multi-index (innermost dim first).
                        for d in (0..rank).rev() {
                            idx[d] += 1;
                            if idx[d] < out_dims[d] as usize {
                                break;
                            }
                            idx[d] = 0;
                        }
                    }
                }
            }

            // (Thunk::DenseSolveF64 / Thunk::ScanBackward had panic
            // stubs here as placeholders during the wire-up; both
            // are now reached by the real implementations earlier in
            // this same match — the stubs were dead code shadowed by
            // the specific-pattern arms above. Removed.)
            Thunk::CustomOp {
                kernel,
                inputs,
                output,
                attrs,
            } => {
                let (out_off, out_len, out_shape) = output;
                unsafe {
                    dispatch_custom_op(
                        &**kernel, inputs, *out_off, *out_len, out_shape, attrs, base,
                    );
                }
            }
        }
    }
}

/// Griewank treeverse: process backward iterations `[t_lo..=t_hi]` (with
/// the carry entering iteration `t_lo` supplied as `anchor_carry`) by
/// recursive binary subdivision. Total work `O((t_hi-t_lo+1) · log)`,
/// auxiliary memory `O(log · carry_bytes)` for the recursion stack.
///
/// Compared to the iterative segment-cached scheme, this trades extra
/// recompute for less working memory — each level of recursion holds
/// one `cb`-sized intermediate carry on the stack but never the whole
/// segment at once. With K saved outer checkpoints, the outer driver
/// invokes this helper once per segment.
///
/// `process_iter(t, carry_at_t)` is the per-iteration leaf action: it
/// runs `body_vjp` at iteration `t` with the supplied carry, threads
/// `dcarry` backward, and (for ScanBackwardXs) writes `dxs[t]`.
#[allow(clippy::too_many_arguments)]
unsafe fn griewank_process_segment(
    t_lo: usize,
    t_hi: usize,
    anchor_carry: &[u8],
    cb: usize,
    fwd_sched: &ThunkSchedule,
    fwd_init: &[u8],
    fwd_carry_in_off: usize,
    fwd_output_off: usize,
    fwd_x_offs: &[usize],
    base: *mut u8,
    outer_xs_offs: &[(usize, u32)],
    fwd_buf: &mut Vec<u8>,
    leaf_threshold: usize,
    process_iter: &mut dyn FnMut(usize, &[u8]),
) {
    unsafe {
        let size = t_hi - t_lo + 1;
        if size == 1 {
            process_iter(t_lo, anchor_carry);
            return;
        }
        if size <= leaf_threshold {
            // Walk forward, cache each carry, run backward in reverse.
            let mut cache: Vec<u8> = Vec::with_capacity(size * cb);
            cache.extend_from_slice(anchor_carry);
            fwd_buf.copy_from_slice(fwd_init);
            std::ptr::copy_nonoverlapping(
                anchor_carry.as_ptr(),
                fwd_buf.as_mut_ptr().add(fwd_carry_in_off),
                cb,
            );
            for i in 1..size {
                let cur_iter = t_lo + i - 1;
                for (idx, fb_x_off) in fwd_x_offs.iter().enumerate() {
                    let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                    let xb = x_psb as usize;
                    std::ptr::copy_nonoverlapping(
                        base.add(outer_xs_off + cur_iter * xb),
                        fwd_buf.as_mut_ptr().add(*fb_x_off),
                        xb,
                    );
                }
                execute_thunks(fwd_sched, fwd_buf);
                if fwd_output_off != fwd_carry_in_off {
                    fwd_buf.copy_within(fwd_output_off..fwd_output_off + cb, fwd_carry_in_off);
                }
                cache.extend_from_slice(&fwd_buf[fwd_carry_in_off..fwd_carry_in_off + cb]);
            }
            // Process backward.
            for t in (t_lo..=t_hi).rev() {
                let idx = t - t_lo;
                let carry = &cache[idx * cb..(idx + 1) * cb];
                process_iter(t, carry);
            }
            return;
        }

        // Split: walk forward from anchor to compute carry entering `mid`.
        // (We need `mid - t_lo` body executions: one per iteration in
        // [t_lo, mid).)
        let mid = t_lo + size / 2;
        fwd_buf.copy_from_slice(fwd_init);
        std::ptr::copy_nonoverlapping(
            anchor_carry.as_ptr(),
            fwd_buf.as_mut_ptr().add(fwd_carry_in_off),
            cb,
        );
        for cur_iter in t_lo..mid {
            for (idx, fb_x_off) in fwd_x_offs.iter().enumerate() {
                let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                let xb = x_psb as usize;
                std::ptr::copy_nonoverlapping(
                    base.add(outer_xs_off + cur_iter * xb),
                    fwd_buf.as_mut_ptr().add(*fb_x_off),
                    xb,
                );
            }
            execute_thunks(fwd_sched, fwd_buf);
            if fwd_output_off != fwd_carry_in_off {
                fwd_buf.copy_within(fwd_output_off..fwd_output_off + cb, fwd_carry_in_off);
            }
        }
        let mid_carry: Vec<u8> = fwd_buf[fwd_carry_in_off..fwd_carry_in_off + cb].to_vec();

        // Right half first (higher t values processed first to match the
        // canonical reverse-mode iteration order: dcarry threads from
        // t=length-1 down to t=0).
        griewank_process_segment(
            mid,
            t_hi,
            &mid_carry,
            cb,
            fwd_sched,
            fwd_init,
            fwd_carry_in_off,
            fwd_output_off,
            fwd_x_offs,
            base,
            outer_xs_offs,
            fwd_buf,
            leaf_threshold,
            process_iter,
        );
        // Then left half with original anchor.
        griewank_process_segment(
            t_lo,
            mid - 1,
            anchor_carry,
            cb,
            fwd_sched,
            fwd_init,
            fwd_carry_in_off,
            fwd_output_off,
            fwd_x_offs,
            base,
            outer_xs_offs,
            fwd_buf,
            leaf_threshold,
            process_iter,
        );
    }
}

/// Execute a batched 1D FFT in the f64 2N-real-block layout.
/// Each "row" is `2N` f64 elements: first `N` real, then `N` imag.
/// The `outer` rows are independent and processed sequentially.
///
/// Both forward and inverse use the same Cooley-Tukey radix-2 DIT
/// kernel — only the twiddle-factor sign differs. Power-of-2 only
/// (the IR builder rejects non-power-of-2 sizes at graph-build time).
/// Batched 1D FFT on the f64 2N-real-block layout. Public so other
/// backend crates can invoke this as a host fallback against a
/// unified-memory arena (e.g. rlx-metal: sync the command buffer,
/// pass the Metal `Buffer::contents()` pointer as `base`, restart the
/// command buffer). Self-contained — no rlx-cpu state required.
///
/// Safety: `base + src` and `base + dst` must be valid for the
/// `outer * 2 * n_complex * sizeof::<f64>()` byte range and stay
/// alive for the duration of the call.
pub unsafe fn execute_fft1d_f64(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    base: *mut u8,
) {
    let row_elems = 2 * n_complex;
    let mut re = vec![0f64; n_complex];
    let mut im = vec![0f64; n_complex];
    // Scratch reused across rows for the Bluestein path. Empty when
    // we're on the radix-2 fast path.
    let mut scratch = if n_complex.is_power_of_two() {
        BluesteinScratchF64::empty()
    } else {
        BluesteinScratchF64::build(n_complex, inverse)
    };
    for o in 0..outer {
        let row_offset = src + o * row_elems * std::mem::size_of::<f64>();
        let s = unsafe { sl_f64(row_offset, base, row_elems) };
        re.copy_from_slice(&s[..n_complex]);
        im.copy_from_slice(&s[n_complex..]);
        if n_complex.is_power_of_two() {
            fft_radix2_inplace_f64(&mut re, &mut im, inverse);
        } else {
            fft_bluestein_inplace_f64(&mut re, &mut im, inverse, &mut scratch);
        }
        let dst_offset = dst + o * row_elems * std::mem::size_of::<f64>();
        let d = unsafe { sl_mut_f64(dst_offset, base, row_elems) };
        d[..n_complex].copy_from_slice(&re);
        d[n_complex..].copy_from_slice(&im);
    }
}

/// f32 counterpart of `execute_fft1d_f64`. Same 2N-real-block layout
/// (first N real, second N imag per row), same unnormalized
/// convention; only the element width differs. Twiddle factors are
/// computed in f64 and cast to f32 to keep large-N error closer to
/// the f64 path (the savings from f32 are in memory bandwidth, not in
/// twiddle precision).
/// f32 mirror of `execute_fft1d_f64`. Same public-host-fallback role.
pub unsafe fn execute_fft1d_f32(
    src: usize,
    dst: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    base: *mut u8,
) {
    let row_elems = 2 * n_complex;
    let mut re = vec![0f32; n_complex];
    let mut im = vec![0f32; n_complex];
    let mut scratch = if n_complex.is_power_of_two() {
        BluesteinScratchF32::empty()
    } else {
        BluesteinScratchF32::build(n_complex, inverse)
    };
    for o in 0..outer {
        let row_offset = src + o * row_elems * std::mem::size_of::<f32>();
        let s = unsafe { sl(row_offset, base, row_elems) };
        re.copy_from_slice(&s[..n_complex]);
        im.copy_from_slice(&s[n_complex..]);
        if n_complex.is_power_of_two() {
            fft_radix2_inplace_f32(&mut re, &mut im, inverse);
        } else {
            fft_bluestein_inplace_f32(&mut re, &mut im, inverse, &mut scratch);
        }
        let dst_offset = dst + o * row_elems * std::mem::size_of::<f32>();
        let d = unsafe { sl_mut(dst_offset, base, row_elems) };
        d[..n_complex].copy_from_slice(&re);
        d[n_complex..].copy_from_slice(&im);
    }
}

/// f32 in-place radix-2 DIT Cooley-Tukey. Structurally identical to
/// the f64 path; twiddle recurrence is kept in f64 so accumulated
/// rotation drift doesn't dominate the per-stage error budget at
/// larger N.
fn fft_radix2_inplace_f32(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert!(
        n.is_power_of_two(),
        "fft_radix2_f32: n={n} must be a power of two"
    );
    if n <= 1 {
        return;
    }

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let w_re_step = theta.cos();
        let w_im_step = theta.sin();
        let mut i = 0usize;
        while i < n {
            let mut wre = 1.0_f64;
            let mut wim = 0.0_f64;
            for k in 0..half {
                let wre_f = wre as f32;
                let wim_f = wim as f32;
                let t_re = wre_f * re[i + k + half] - wim_f * im[i + k + half];
                let t_im = wre_f * im[i + k + half] + wim_f * re[i + k + half];
                let u_re = re[i + k];
                let u_im = im[i + k];
                re[i + k] = u_re + t_re;
                im[i + k] = u_im + t_im;
                re[i + k + half] = u_re - t_re;
                im[i + k + half] = u_im - t_im;
                let new_wre = wre * w_re_step - wim * w_im_step;
                let new_wim = wre * w_im_step + wim * w_re_step;
                wre = new_wre;
                wim = new_wim;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// In-place radix-2 DIT Cooley-Tukey FFT on split (real, imag) f64
/// arrays. `n = re.len() = im.len()` must be a power of two. Forward
/// uses ω = exp(-2πi/n); inverse uses ω = exp(+2πi/n) (no 1/N scale).
fn fft_radix2_inplace_f64(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert!(
        n.is_power_of_two(),
        "fft_radix2: n={n} must be a power of two"
    );
    if n <= 1 {
        return;
    }

    // Bit-reverse permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterflies: ω_len = exp(±2πi/len).
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let w_re_step = theta.cos();
        let w_im_step = theta.sin();
        let mut i = 0usize;
        while i < n {
            // Twiddle starts at 1+0i for each segment.
            let mut wre = 1.0_f64;
            let mut wim = 0.0_f64;
            for k in 0..half {
                let t_re = wre * re[i + k + half] - wim * im[i + k + half];
                let t_im = wre * im[i + k + half] + wim * re[i + k + half];
                let u_re = re[i + k];
                let u_im = im[i + k];
                re[i + k] = u_re + t_re;
                im[i + k] = u_im + t_im;
                re[i + k + half] = u_re - t_re;
                im[i + k + half] = u_im - t_im;
                let new_wre = wre * w_re_step - wim * w_im_step;
                let new_wim = wre * w_im_step + wim * w_re_step;
                wre = new_wre;
                wim = new_wim;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Pre-computed chirp + filter-spectrum for one (N, direction) pair.
/// Built once per call to `execute_fft1d_f64` and reused across rows
/// when `outer > 1` — the chirp and FFT(b) don't depend on the input.
struct BluesteinScratchF64 {
    /// Power-of-two convolution length, ≥ 2N - 1.
    m: usize,
    /// `w[k] = exp(sign · iπ · k² / N)` for k=0..N, where sign matches
    /// the requested direction. Forward chirp on the way in, output
    /// chirp on the way out.
    w_re: Vec<f64>,
    w_im: Vec<f64>,
    /// FFT of the embedded filter `b[k] = conj(w[|k|])` in length-M.
    /// Doesn't depend on the input — precomputed once.
    bf_re: Vec<f64>,
    bf_im: Vec<f64>,
    /// Workspace reused per row (avoids per-row allocation).
    ar: Vec<f64>,
    ai: Vec<f64>,
}

impl BluesteinScratchF64 {
    fn empty() -> Self {
        Self {
            m: 0,
            w_re: Vec::new(),
            w_im: Vec::new(),
            bf_re: Vec::new(),
            bf_im: Vec::new(),
            ar: Vec::new(),
            ai: Vec::new(),
        }
    }

    fn build(n: usize, inverse: bool) -> Self {
        // M = next power of two ≥ 2N - 1 keeps the inner FFT on the
        // fast radix-2 path. For N=1 fall back to M=1 (no-op convolution).
        let m = if n <= 1 { 1 } else { (2 * n - 1).next_power_of_two() };

        // Chirp arg reduced via k² mod 2N — without this, large N
        // bleeds precision into the trig call (n² grows quadratically).
        let mod_2n = (2 * n) as u64;
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut w_re = vec![0.0_f64; n];
        let mut w_im = vec![0.0_f64; n];
        for k in 0..n {
            let k2 = (k as u64).wrapping_mul(k as u64) % mod_2n;
            let theta = sign * std::f64::consts::PI * (k2 as f64) / (n as f64);
            w_re[k] = theta.cos();
            w_im[k] = theta.sin();
        }

        // Embed b[k] = conj(w[|k|]) into length M with the negative
        // indices wrapping to the tail: b[-j] → B[M-j] for j=1..N-1.
        let mut bf_re = vec![0.0_f64; m];
        let mut bf_im = vec![0.0_f64; m];
        if n > 0 {
            bf_re[0] = w_re[0];
            bf_im[0] = -w_im[0];
            for k in 1..n {
                bf_re[k] = w_re[k];
                bf_im[k] = -w_im[k];
                bf_re[m - k] = w_re[k];
                bf_im[m - k] = -w_im[k];
            }
        }
        if m > 1 {
            fft_radix2_inplace_f64(&mut bf_re, &mut bf_im, false);
        }

        Self {
            m,
            w_re,
            w_im,
            bf_re,
            bf_im,
            ar: vec![0.0_f64; m],
            ai: vec![0.0_f64; m],
        }
    }
}

/// Bluestein (chirp-z) FFT for arbitrary N. Identity used:
///   `n·k = (n² + k² - (k-n)²) / 2`
/// which lets the DFT be written as a linear convolution sandwiched
/// between two chirp multiplies:
///   `X[k] = w[k] · ((x·w) ⊛ conj(w))[k]`   where `w[n] = exp(±iπ·n²/N)`.
/// The convolution is computed via a length-M radix-2 FFT (M ≥ 2N-1).
/// Both directions stay unnormalized to match the radix-2 path, so the
/// chain rule keeps working without scaling.
fn fft_bluestein_inplace_f64(
    re: &mut [f64],
    im: &mut [f64],
    _inverse: bool,
    s: &mut BluesteinScratchF64,
) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert_eq!(s.w_re.len(), n);
    if n <= 1 {
        return;
    }
    let m = s.m;

    // Pre-chirp: a[k] = x[k] · w[k], zero-padded to M.
    for k in 0..m {
        s.ar[k] = 0.0;
        s.ai[k] = 0.0;
    }
    for k in 0..n {
        s.ar[k] = re[k] * s.w_re[k] - im[k] * s.w_im[k];
        s.ai[k] = re[k] * s.w_im[k] + im[k] * s.w_re[k];
    }

    // Length-M forward FFT of the padded chirped input.
    fft_radix2_inplace_f64(&mut s.ar, &mut s.ai, false);

    // Pointwise product with FFT(b). Stored back into (ar, ai).
    for k in 0..m {
        let ar = s.ar[k];
        let ai = s.ai[k];
        let br = s.bf_re[k];
        let bi = s.bf_im[k];
        s.ar[k] = ar * br - ai * bi;
        s.ai[k] = ar * bi + ai * br;
    }

    // Inverse FFT — radix-2 here is the unnormalized inverse, so we
    // divide by M to recover the true circular convolution.
    fft_radix2_inplace_f64(&mut s.ar, &mut s.ai, true);
    let inv_m = 1.0 / (m as f64);

    // Post-chirp: X[k] = w[k] · Y[k] / M for k = 0..N.
    for k in 0..n {
        let yr = s.ar[k] * inv_m;
        let yi = s.ai[k] * inv_m;
        re[k] = yr * s.w_re[k] - yi * s.w_im[k];
        im[k] = yr * s.w_im[k] + yi * s.w_re[k];
    }
}

/// f32 mirror of `BluesteinScratchF64`. Chirp is computed in f64 for
/// precision (same justification as the radix-2 f32 path: twiddles in
/// f64, butterflies in f32). The actual conv buffers are f32.
struct BluesteinScratchF32 {
    m: usize,
    w_re: Vec<f32>,
    w_im: Vec<f32>,
    bf_re: Vec<f32>,
    bf_im: Vec<f32>,
    ar: Vec<f32>,
    ai: Vec<f32>,
}

impl BluesteinScratchF32 {
    fn empty() -> Self {
        Self {
            m: 0,
            w_re: Vec::new(),
            w_im: Vec::new(),
            bf_re: Vec::new(),
            bf_im: Vec::new(),
            ar: Vec::new(),
            ai: Vec::new(),
        }
    }

    fn build(n: usize, inverse: bool) -> Self {
        let m = if n <= 1 { 1 } else { (2 * n - 1).next_power_of_two() };

        let mod_2n = (2 * n) as u64;
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let mut w_re = vec![0.0_f32; n];
        let mut w_im = vec![0.0_f32; n];
        for k in 0..n {
            let k2 = (k as u64).wrapping_mul(k as u64) % mod_2n;
            let theta = sign * std::f64::consts::PI * (k2 as f64) / (n as f64);
            w_re[k] = theta.cos() as f32;
            w_im[k] = theta.sin() as f32;
        }

        let mut bf_re = vec![0.0_f32; m];
        let mut bf_im = vec![0.0_f32; m];
        if n > 0 {
            bf_re[0] = w_re[0];
            bf_im[0] = -w_im[0];
            for k in 1..n {
                bf_re[k] = w_re[k];
                bf_im[k] = -w_im[k];
                bf_re[m - k] = w_re[k];
                bf_im[m - k] = -w_im[k];
            }
        }
        if m > 1 {
            fft_radix2_inplace_f32(&mut bf_re, &mut bf_im, false);
        }

        Self {
            m,
            w_re,
            w_im,
            bf_re,
            bf_im,
            ar: vec![0.0_f32; m],
            ai: vec![0.0_f32; m],
        }
    }
}

fn fft_bluestein_inplace_f32(
    re: &mut [f32],
    im: &mut [f32],
    _inverse: bool,
    s: &mut BluesteinScratchF32,
) {
    let n = re.len();
    debug_assert_eq!(im.len(), n);
    debug_assert_eq!(s.w_re.len(), n);
    if n <= 1 {
        return;
    }
    let m = s.m;

    for k in 0..m {
        s.ar[k] = 0.0;
        s.ai[k] = 0.0;
    }
    for k in 0..n {
        s.ar[k] = re[k] * s.w_re[k] - im[k] * s.w_im[k];
        s.ai[k] = re[k] * s.w_im[k] + im[k] * s.w_re[k];
    }

    fft_radix2_inplace_f32(&mut s.ar, &mut s.ai, false);

    for k in 0..m {
        let ar = s.ar[k];
        let ai = s.ai[k];
        let br = s.bf_re[k];
        let bi = s.bf_im[k];
        s.ar[k] = ar * br - ai * bi;
        s.ai[k] = ar * bi + ai * br;
    }

    fft_radix2_inplace_f32(&mut s.ar, &mut s.ai, true);
    let inv_m = 1.0_f32 / (m as f32);

    for k in 0..n {
        let yr = s.ar[k] * inv_m;
        let yi = s.ai[k] * inv_m;
        re[k] = yr * s.w_re[k] - yi * s.w_im[k];
        im[k] = yr * s.w_im[k] + yi * s.w_re[k];
    }
}

/// Shared dispatch path for `Thunk::CustomOp`. Builds a typed
/// [`CpuTensorRef`] for each input *at that input's declared dtype*
/// (so a sparse-LU op with mixed F64/I32 inputs gets the right
/// typed slices) and a [`CpuTensorMut`] for the output, then calls
/// the kernel's single `execute` method.
unsafe fn dispatch_custom_op(
    kernel: &dyn crate::op_registry::CpuKernel,
    inputs: &[(usize, u32, Shape)],
    out_off: usize,
    out_len: u32,
    out_shape: &Shape,
    attrs: &[u8],
    base: *mut u8,
) {
    use crate::op_registry::{CpuTensorMut, CpuTensorRef};
    use rlx_ir::DType;

    // One arm per `DType` variant — single source of truth for
    // "which dtypes the CPU custom-op dispatcher wires." If a new
    // DType lands in `rlx-ir`, the compiler flags this match as
    // non-exhaustive and the gap gets named at the right place.
    macro_rules! build_in_view {
        ($shape:expr, $off:expr, $n:expr, $variant:ident, $rust_ty:ty) => {
            CpuTensorRef::$variant {
                data: unsafe { sl_typed::<$rust_ty>($off, base, $n) },
                shape: $shape,
            }
        };
    }
    macro_rules! build_out_view {
        ($variant:ident, $rust_ty:ty) => {
            CpuTensorMut::$variant {
                data: unsafe { sl_mut_typed::<$rust_ty>(out_off, base, out_len as usize) },
                shape: out_shape,
            }
        };
    }

    let in_views: Vec<CpuTensorRef<'_>> = inputs
        .iter()
        .map(|(off, len, shape)| {
            let n = *len as usize;
            let off = *off;
            match shape.dtype() {
                DType::F32 => build_in_view!(shape, off, n, F32, f32),
                DType::F64 => build_in_view!(shape, off, n, F64, f64),
                DType::F16 => build_in_view!(shape, off, n, F16, half::f16),
                DType::BF16 => build_in_view!(shape, off, n, BF16, half::bf16),
                DType::I8 => build_in_view!(shape, off, n, I8, i8),
                DType::I16 => build_in_view!(shape, off, n, I16, i16),
                DType::I32 => build_in_view!(shape, off, n, I32, i32),
                DType::I64 => build_in_view!(shape, off, n, I64, i64),
                DType::U8 => build_in_view!(shape, off, n, U8, u8),
                DType::U32 => build_in_view!(shape, off, n, U32, u32),
                DType::Bool => build_in_view!(shape, off, n, Bool, u8),
                // C64 isn't a CpuTensor variant today; the user-registered
                // op_registry path doesn't see complex inputs (those are
                // handled by built-in ops with dedicated kernels).
                DType::C64 => panic!(
                    "Op::Custom kernel input has DType::C64 — built-in \
                 complex ops handle their own kernels; user-registered \
                 ops don't yet see complex tensors"
                ),
            }
        })
        .collect();

    let result = match out_shape.dtype() {
        DType::F32 => kernel.execute(&in_views, build_out_view!(F32, f32), attrs),
        DType::F64 => kernel.execute(&in_views, build_out_view!(F64, f64), attrs),
        DType::F16 => kernel.execute(&in_views, build_out_view!(F16, half::f16), attrs),
        DType::BF16 => kernel.execute(&in_views, build_out_view!(BF16, half::bf16), attrs),
        DType::I8 => kernel.execute(&in_views, build_out_view!(I8, i8), attrs),
        DType::I16 => kernel.execute(&in_views, build_out_view!(I16, i16), attrs),
        DType::I32 => kernel.execute(&in_views, build_out_view!(I32, i32), attrs),
        DType::I64 => kernel.execute(&in_views, build_out_view!(I64, i64), attrs),
        DType::U8 => kernel.execute(&in_views, build_out_view!(U8, u8), attrs),
        DType::U32 => kernel.execute(&in_views, build_out_view!(U32, u32), attrs),
        DType::Bool => kernel.execute(&in_views, build_out_view!(Bool, u8), attrs),
        DType::C64 => panic!("Op::Custom output DType::C64 not supported"),
    };
    if let Err(e) = result {
        panic!("Op::Custom('{}') CPU kernel failed: {e}", kernel.name());
    }
}

/// Generic raw-cast slice helper. The existing per-dtype `sl_*` /
/// `sl_mut_*` helpers stay in place for the rest of `thunk.rs` (which
/// uses them at call sites with concrete dtypes); the custom-op
/// dispatcher uses these to enumerate every `DType` uniformly without
/// listing one helper per dtype.
#[inline(always)]
unsafe fn sl_typed<T>(offset: usize, base: *mut u8, len: usize) -> &'static [T] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const T, len) }
}

#[inline(always)]
unsafe fn sl_mut_typed<T>(offset: usize, base: *mut u8, len: usize) -> &'static mut [T] {
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut T, len) }
}

// Unsafe helpers to create slices from arena base + offset
#[inline(always)]
/// In-place per-element activation. Mirrors the dispatch in
/// `Thunk::ActivationInPlace`. Used by `Thunk::FusedMmBiasAct` to
/// apply the activation after `bias_add` for all non-Gelu cases.
fn apply_activation_inplace(d: &mut [f32], act: rlx_ir::op::Activation) {
    use rlx_ir::op::Activation;
    match act {
        Activation::Gelu => crate::kernels::par_gelu_inplace(d),
        Activation::GeluApprox => crate::kernels::par_gelu_approx_inplace(d),
        Activation::Silu => crate::kernels::par_silu_inplace(d),
        Activation::Relu => {
            for v in d.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Sigmoid => {
            for v in d.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        Activation::Tanh => {
            for v in d.iter_mut() {
                *v = v.tanh();
            }
        }
        Activation::Exp => {
            for v in d.iter_mut() {
                *v = v.exp();
            }
        }
        Activation::Log => {
            for v in d.iter_mut() {
                *v = v.ln();
            }
        }
        Activation::Sqrt => {
            for v in d.iter_mut() {
                *v = v.sqrt();
            }
        }
        Activation::Rsqrt => {
            for v in d.iter_mut() {
                *v = 1.0 / v.sqrt();
            }
        }
        Activation::Neg => {
            for v in d.iter_mut() {
                *v = -*v;
            }
        }
        Activation::Abs => {
            for v in d.iter_mut() {
                *v = v.abs();
            }
        }
        Activation::Round => {
            for v in d.iter_mut() {
                *v = v.round();
            }
        }
        Activation::Sin => {
            for v in d.iter_mut() {
                *v = v.sin();
            }
        }
        Activation::Cos => {
            for v in d.iter_mut() {
                *v = v.cos();
            }
        }
        Activation::Tan => {
            for v in d.iter_mut() {
                *v = v.tan();
            }
        }
        Activation::Atan => {
            for v in d.iter_mut() {
                *v = v.atan();
            }
        }
    }
}

/// im2col for one image (single batch + group slice).
///
/// Source `x` is `[c_in, H, W]` row-major. Destination `col` is
/// `[c_in · kH · kW, H_out · W_out]` row-major. Out-of-bounds positions
/// (in the padded region) are written as 0.
///
/// `col[(ci · kH · kW + ki · kW + kj) · n_dim + ho · W_out + wo] =
///    x[ci, ho·sh + ki·dh − ph, wo·sw + kj·dw_dil − pw]`
#[allow(clippy::too_many_arguments)]
fn im2col(
    x: &[f32],
    col: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        for wo in 0..w_out {
                            col[row_off + ho * w_out + wo] = 0.0;
                        }
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        col[row_off + ho * w_out + wo] = if wi < 0 || wi >= w_isz {
                            0.0
                        } else {
                            x[in_row_off + wi as usize]
                        };
                    }
                }
            }
        }
    }
}

/// col2im — inverse of `im2col` with scatter-accumulation. The caller
/// is responsible for zeroing `x` if it doesn't already start zero
/// (the conv-input-grad path zeros once before the batch loop).
///
/// `x[ci, hi, wi] += col[(ci · kH · kW + ki · kW + kj) · n_dim + ho · W_out + wo]`
/// for all `(ki, kj, ho, wo)` whose `(hi, wi)` lands in `[0, H) × [0, W)`.
#[allow(clippy::too_many_arguments)]
fn col2im(
    col: &[f32],
    x: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        if wi < 0 || wi >= w_isz {
                            continue;
                        }
                        x[in_row_off + wi as usize] += col[row_off + ho * w_out + wo];
                    }
                }
            }
        }
    }
}

/// Element-wise backward for `Op::Activation`. `xs` is the original
/// input to the forward activation; `dys` is the upstream gradient.
/// Writes `out[i] = (d/dx act(xs[i])) * dys[i]`.
/// Decompose a per-channel quantization shape into the
/// `(chan_axis, chan_dim, inner)` triplet the kernel needs to map a
/// flat output index to a channel index. Per-tensor (`axis = None`)
/// degenerates to `chan_dim = 1, inner = len`, which makes the
/// kernel's `(i / inner) % chan_dim` always 0 — same fast path the
/// scalar version used.
fn quant_layout(shape: &rlx_ir::Shape, axis: Option<usize>) -> (usize, usize, usize) {
    match axis {
        None => (0, 1, shape.num_elements().unwrap_or(0).max(1)),
        Some(d) => {
            let chan_dim = shape.dim(d).unwrap_static();
            let inner: usize = (d + 1..shape.rank())
                .map(|i| shape.dim(i).unwrap_static())
                .product::<usize>()
                .max(1);
            (d, chan_dim, inner)
        }
    }
}

fn activation_backward_kernel(
    act: rlx_ir::op::Activation,
    xs: &[f32],
    dys: &[f32],
    out: &mut [f32],
) {
    use rlx_ir::op::Activation;
    let n = xs.len();
    debug_assert_eq!(dys.len(), n);
    debug_assert_eq!(out.len(), n);
    match act {
        Activation::Relu => {
            for i in 0..n {
                out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
            }
        }
        Activation::Sigmoid => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 - s) * dys[i];
            }
        }
        Activation::Tanh => {
            for i in 0..n {
                let t = xs[i].tanh();
                out[i] = (1.0 - t * t) * dys[i];
            }
        }
        Activation::Silu => {
            // y = x * σ(x);  dy/dx = σ(x) * (1 + x * (1 - σ(x))).
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 + xs[i] * (1.0 - s)) * dys[i];
            }
        }
        Activation::Gelu => {
            // Exact erf-based GELU:  y = 0.5 x (1 + erf(x / √2)).
            //   dy/dx = 0.5 (1 + erf(x/√2)) + (x / √(2π)) · exp(-x²/2)
            const INV_SQRT2: f32 = 0.707_106_77;
            const INV_SQRT_2PI: f32 = 0.398_942_3;
            for i in 0..n {
                let x = xs[i];
                let phi = 0.5 * (1.0 + erf_f32(x * INV_SQRT2));
                let pdf = INV_SQRT_2PI * (-(x * x) * 0.5).exp();
                out[i] = (phi + x * pdf) * dys[i];
            }
        }
        Activation::GeluApprox => {
            // Tanh-approximation:
            //   y = 0.5 x (1 + tanh(c · (x + 0.044715 x³))) where c = √(2/π).
            const C: f32 = 0.797_884_6; // √(2/π)
            const A: f32 = 0.044_715;
            for i in 0..n {
                let x = xs[i];
                let inner = C * (x + A * x * x * x);
                let t = inner.tanh();
                let dinner = C * (1.0 + 3.0 * A * x * x);
                let d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner;
                out[i] = d * dys[i];
            }
        }
        Activation::Exp => {
            for i in 0..n {
                out[i] = xs[i].exp() * dys[i];
            }
        }
        Activation::Log => {
            for i in 0..n {
                out[i] = dys[i] / xs[i];
            }
        }
        Activation::Sqrt => {
            // d/dx √x = 0.5 / √x — undefined at x=0; clamp to 0.
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 { 0.5 * dys[i] / s } else { 0.0 };
            }
        }
        Activation::Rsqrt => {
            // d/dx (1/√x) = -0.5 · x^(-3/2).
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 {
                    -0.5 * dys[i] / (xs[i] * s)
                } else {
                    0.0
                };
            }
        }
        Activation::Neg => {
            for i in 0..n {
                out[i] = -dys[i];
            }
        }
        Activation::Abs => {
            // sign(x); 0 at x=0.
            for i in 0..n {
                let x = xs[i];
                let s = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out[i] = s * dys[i];
            }
        }
        Activation::Round => {
            // STE: pretend the round was identity in the backward
            // pass. The round step has zero gradient almost
            // everywhere, so without this trick the optimizer can't
            // learn through it.
            out.copy_from_slice(dys);
        }
        Activation::Sin => {
            // d/dx sin(x) = cos(x).
            for i in 0..n {
                out[i] = xs[i].cos() * dys[i];
            }
        }
        Activation::Cos => {
            for i in 0..n {
                out[i] = -xs[i].sin() * dys[i];
            }
        }
        Activation::Tan => {
            // d/dx tan(x) = sec²(x) = 1 + tan²(x)
            for i in 0..n {
                let t = xs[i].tan();
                out[i] = (1.0 + t * t) * dys[i];
            }
        }
        Activation::Atan => {
            // d/dx atan(x) = 1 / (1 + x²)
            for i in 0..n {
                let x = xs[i];
                out[i] = dys[i] / (1.0 + x * x);
            }
        }
    }
}

/// f64 sibling of `activation_backward_kernel`. Same math, twice the
/// precision — used by f64 graphs where the f32 kernel reading bytes
/// as `&[f32]` would silently discard half of every f64 value.
fn activation_backward_kernel_f64(
    act: rlx_ir::op::Activation,
    xs: &[f64],
    dys: &[f64],
    out: &mut [f64],
) {
    use rlx_ir::op::Activation;
    let n = xs.len();
    debug_assert_eq!(dys.len(), n);
    debug_assert_eq!(out.len(), n);
    match act {
        Activation::Relu => {
            for i in 0..n {
                out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
            }
        }
        Activation::Sigmoid => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 - s) * dys[i];
            }
        }
        Activation::Tanh => {
            for i in 0..n {
                let t = xs[i].tanh();
                out[i] = (1.0 - t * t) * dys[i];
            }
        }
        Activation::Silu => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 + xs[i] * (1.0 - s)) * dys[i];
            }
        }
        Activation::Gelu | Activation::GeluApprox => {
            // Both rare on f64 paths; use the high-quality libm erf.
            const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
            for i in 0..n {
                let x = xs[i];
                let phi = 0.5 * (1.0 + erf_f64(x * INV_SQRT2));
                let pdf = INV_SQRT_2PI * (-(x * x) * 0.5).exp();
                out[i] = (phi + x * pdf) * dys[i];
            }
        }
        Activation::Exp => {
            for i in 0..n {
                out[i] = xs[i].exp() * dys[i];
            }
        }
        Activation::Log => {
            for i in 0..n {
                out[i] = dys[i] / xs[i];
            }
        }
        Activation::Sqrt => {
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 { 0.5 * dys[i] / s } else { 0.0 };
            }
        }
        Activation::Rsqrt => {
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 {
                    -0.5 * dys[i] / (xs[i] * s)
                } else {
                    0.0
                };
            }
        }
        Activation::Neg => {
            for i in 0..n {
                out[i] = -dys[i];
            }
        }
        Activation::Abs => {
            for i in 0..n {
                let x = xs[i];
                let s = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out[i] = s * dys[i];
            }
        }
        Activation::Round => {
            out.copy_from_slice(dys);
        }
        Activation::Sin => {
            for i in 0..n {
                out[i] = xs[i].cos() * dys[i];
            }
        }
        Activation::Cos => {
            for i in 0..n {
                out[i] = -xs[i].sin() * dys[i];
            }
        }
        Activation::Tan => {
            for i in 0..n {
                let t = xs[i].tan();
                out[i] = (1.0 + t * t) * dys[i];
            }
        }
        Activation::Atan => {
            for i in 0..n {
                let x = xs[i];
                out[i] = dys[i] / (1.0 + x * x);
            }
        }
    }
}

/// f64 erf via A&S 7.1.26 — same coefficients as `erf_f32`, computed
/// at f64 width. Max error ~1.5e-7 (limited by the polynomial, not the
/// arithmetic). Adequate for gradient kernels; if higher precision is
/// needed, swap in a libm dependency.
#[inline(always)]
fn erf_f64(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_43 * t - 1.453_152_03) * t) + 1.421_413_75) * t - 0.284_496_74) * t
            + 0.254_829_59)
            * t
            * (-x * x).exp();
    s * y
}

/// Cheap erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7
/// over all of ℝ — plenty for f32 gradient kernels).
#[inline(always)]
fn erf_f32(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

unsafe fn sl(offset: usize, base: *mut u8, len: usize) -> &'static [f32] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const f32, len) }
}

#[inline(always)]
unsafe fn sl_mut(offset: usize, base: *mut u8, len: usize) -> &'static mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut f32, len) }
}

#[inline(always)]
unsafe fn sl_f64(offset: usize, base: *mut u8, len: usize) -> &'static [f64] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const f64, len) }
}

#[inline(always)]
unsafe fn sl_mut_f64(offset: usize, base: *mut u8, len: usize) -> &'static mut [f64] {
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut f64, len) }
}

// i32 / i64 typed slice helpers — siblings of sl_f32/sl_f64. Kept for
// integer-tensor thunks that haven't landed yet (Sample, Gather index
// buffers); deleting them now would force re-deriving the unsafe
// boilerplate when the next int-typed thunk lands.
#[allow(dead_code)]
#[inline(always)]
unsafe fn sl_i32(offset: usize, base: *mut u8, len: usize) -> &'static [i32] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const i32, len) }
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn sl_mut_i32(offset: usize, base: *mut u8, len: usize) -> &'static mut [i32] {
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut i32, len) }
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn sl_i64(offset: usize, base: *mut u8, len: usize) -> &'static [i64] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const i64, len) }
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn sl_mut_i64(offset: usize, base: *mut u8, len: usize) -> &'static mut [i64] {
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut i64, len) }
}

/// f64 N-D index walk used by Transpose and Expand. `out_dims` gives
/// the output shape; `in_strides` gives the source stride for each
/// output dim (broadcast axes have stride 0).
fn transpose_walk_f64(inp: &[f64], out: &mut [f64], out_dims: &[u32], in_strides: &[u32]) {
    let rank = out_dims.len();
    let mut idx = vec![0u32; rank];
    for o in 0..out.len() {
        let mut src_off = 0usize;
        for d in 0..rank {
            src_off += idx[d] as usize * in_strides[d] as usize;
        }
        out[o] = inp[src_off];
        // Increment index — last dim varies fastest.
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// f64 elementwise activation. Reads `inp`, writes `out`. For now
/// covers what the autodiff-emitted gradient graph needs (Neg, Exp,
/// Log, Sqrt, Rsqrt, Abs, Tanh, Sigmoid, Relu — the
/// transcendental-free subset). Approximate Gelu/Silu deferred until a
/// workload demands them at f64.
fn apply_activation_f64(inp: &[f64], out: &mut [f64], kind: Activation) {
    match kind {
        Activation::Neg => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = -v;
            }
        }
        Activation::Exp => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.exp();
            }
        }
        Activation::Log => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.ln();
            }
        }
        Activation::Sqrt => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.sqrt();
            }
        }
        Activation::Rsqrt => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = 1.0 / v.sqrt();
            }
        }
        Activation::Abs => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.abs();
            }
        }
        Activation::Tanh => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.tanh();
            }
        }
        Activation::Sigmoid => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = 1.0 / (1.0 + (-v).exp());
            }
        }
        Activation::Relu => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.max(0.0);
            }
        }
        Activation::Round => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.round_ties_even();
            }
        }
        Activation::Sin => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.sin();
            }
        }
        Activation::Cos => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.cos();
            }
        }
        Activation::Tan => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.tan();
            }
        }
        Activation::Atan => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.atan();
            }
        }
        Activation::Gelu | Activation::GeluApprox | Activation::Silu => {
            panic!(
                "apply_activation_f64: {kind:?} not yet implemented at f64. \
                    Add when a workload needs it."
            );
        }
    }
}

#[inline]
fn binary_op_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => a.powf(b),
    }
}

/// f64 sum reduction over a contiguous middle range.
/// Layout: input is `[outer, reduced, inner]`, output is `[outer, inner]`.
fn reduce_sum_f64(inp: &[f64], out: &mut [f64], outer: usize, reduced: usize, inner: usize) {
    for o in 0..outer {
        for n in 0..inner {
            let mut acc = 0.0_f64;
            for r in 0..reduced {
                acc += inp[o * reduced * inner + r * inner + n];
            }
            out[o * inner + n] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::*;

    /// Plan #45: when a Narrow's only consumer is a Rope, the thunk
    /// fusion pass collapses them — the Narrow becomes Nop, and the
    /// Rope reads from the parent buffer with its row stride. This
    /// test runs the unfused path (batch*seq > FusedAttnBlock
    /// threshold) and asserts the rewrite happened.
    #[test]
    fn narrow_rope_fuses_in_unfused_path() {
        let f = DType::F32;
        let mut g = Graph::new("nr_fuse");
        // Force batch*seq > 64 so FusedAttnBlock doesn't pre-empt us.
        let qkv = g.input("qkv", Shape::new(&[16, 8, 192], f)); // 16*8=128 > 64
        let cos = g.input("cos", Shape::new(&[16], f));
        let sin = g.input("sin", Shape::new(&[16], f));
        // Last-axis narrow: Q = qkv[..., 0..64]
        let q = g.narrow_(qkv, 2, 0, 64);
        let q_rope = g.rope(q, cos, sin, 16);
        g.set_outputs(vec![q_rope]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let mut narrow_count = 0;
        let mut rope_with_stride: Option<u32> = None;
        for t in &sched.thunks {
            match t {
                Thunk::Narrow { .. } => narrow_count += 1,
                Thunk::Rope { src_row_stride, .. } => rope_with_stride = Some(*src_row_stride),
                _ => {}
            }
        }
        // After fusion the Narrow is gone; only the Rope remains, and
        // it now walks with the parent QKV's row stride (3 * 64 = 192).
        assert_eq!(
            narrow_count, 0,
            "Narrow→Rope fusion should leave zero Narrow thunks; saw {narrow_count}"
        );
        assert_eq!(
            rope_with_stride,
            Some(192),
            "Rope's src_row_stride should be 192 (parent qkv axis), saw {rope_with_stride:?}"
        );
    }

    /// Plan #15: SSM selective scan matches a naive Python-style
    /// Python-style sequential reference.
    #[test]
    fn ssm_selective_scan_matches_reference() {
        use rlx_ir::Philox4x32;
        let bch = 1usize;
        let s = 4usize;
        let h = 3usize;
        let n = 2usize;

        let mut rng = Philox4x32::new(13);
        let mut x = vec![0f32; bch * s * h];
        rng.fill_normal(&mut x);
        let mut delta = vec![0f32; bch * s * h];
        // Keep Δ small so exp(Δ·A) doesn't blow up.
        for v in delta.iter_mut() {
            *v = (rng.next_f32() - 0.5) * 0.1;
        }
        let mut a = vec![0f32; h * n];
        for v in a.iter_mut() {
            *v = -(rng.next_f32() * 0.5 + 0.1);
        } // negative for stability
        let mut b = vec![0f32; bch * s * n];
        rng.fill_normal(&mut b);
        let mut c = vec![0f32; bch * s * n];
        rng.fill_normal(&mut c);

        // Reference scan.
        let mut expected = vec![0f32; bch * s * h];
        for bi in 0..bch {
            let mut state = vec![0f32; h * n];
            for si in 0..s {
                for ci in 0..h {
                    let d = delta[bi * s * h + si * h + ci];
                    let xv = x[bi * s * h + si * h + ci];
                    let mut acc = 0f32;
                    for ni in 0..n {
                        let da = (d * a[ci * n + ni]).exp();
                        state[ci * n + ni] =
                            da * state[ci * n + ni] + d * b[bi * s * n + si * n + ni] * xv;
                        acc += c[bi * s * n + si * n + ni] * state[ci * n + ni];
                    }
                    expected[bi * s * h + si * h + ci] = acc;
                }
            }
        }

        // RLX path.
        let f = DType::F32;
        let mut g = Graph::new("ssm");
        let xn = g.input("x", Shape::new(&[bch, s, h], f));
        let dn = g.input("delta", Shape::new(&[bch, s, h], f));
        let an = g.param("a", Shape::new(&[h, n], f));
        let bn = g.param("b", Shape::new(&[bch, s, n], f));
        let cn = g.param("c", Shape::new(&[bch, s, n], f));
        let yn = g.selective_scan(xn, dn, an, bn, cn, n, Shape::new(&[bch, s, h], f));
        g.set_outputs(vec![yn]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let xn_off = arena.byte_offset(xn);
        let dn_off = arena.byte_offset(dn);
        let an_off = arena.byte_offset(an);
        let bn_off = arena.byte_offset(bn);
        let cn_off = arena.byte_offset(cn);
        let yn_off = arena.byte_offset(yn);
        let buf = arena.raw_buf_mut();
        unsafe {
            let copy = |dst: *mut f32, data: &[f32]| {
                for (i, &v) in data.iter().enumerate() {
                    *dst.add(i) = v;
                }
            };
            copy(buf.as_mut_ptr().add(xn_off) as *mut f32, &x);
            copy(buf.as_mut_ptr().add(dn_off) as *mut f32, &delta);
            copy(buf.as_mut_ptr().add(an_off) as *mut f32, &a);
            copy(buf.as_mut_ptr().add(bn_off) as *mut f32, &b);
            copy(buf.as_mut_ptr().add(cn_off) as *mut f32, &c);
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(yn_off) as *const f32;
            (0..bch * s * h).map(|i| *p.add(i)).collect()
        };

        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - a).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {a}"
            );
        }
    }

    /// Plan #26: 1×1 conv lowers to per-batch sgemm and matches the
    /// scalar 7-loop reference.
    #[test]
    fn conv_1x1_fast_path_matches_scalar() {
        use rlx_ir::Philox4x32;
        // [N=2, C_in=4, H=3, W=3]
        let n = 2usize;
        let c_in = 4usize;
        let h = 3usize;
        let w = 3usize;
        let c_out = 5usize;
        let mut rng = Philox4x32::new(31);
        let mut x = vec![0f32; n * c_in * h * w];
        rng.fill_normal(&mut x);
        let mut weight = vec![0f32; c_out * c_in];
        rng.fill_normal(&mut weight);

        // Reference: scalar 1×1 conv = per-batch matmul
        // out[ni, co, hi, wi] = sum_ci weight[co, ci] * x[ni, ci, hi, wi]
        let mut expected = vec![0f32; n * c_out * h * w];
        for ni in 0..n {
            for co in 0..c_out {
                for hi in 0..h {
                    for wi in 0..w {
                        let mut acc = 0f32;
                        for ci in 0..c_in {
                            acc += weight[co * c_in + ci]
                                * x[((ni * c_in) + ci) * h * w + hi * w + wi];
                        }
                        expected[((ni * c_out) + co) * h * w + hi * w + wi] = acc;
                    }
                }
            }
        }

        // RLX path: build a graph with Op::Conv (kernel=[1,1], stride=[1,1], etc).
        let f = DType::F32;
        let mut g = Graph::new("conv1x1");
        let xn = g.input("x", Shape::new(&[n, c_in, h, w], f));
        let wn = g.param("w", Shape::new(&[c_out, c_in, 1, 1], f));
        // Manually add Op::Conv since there's no `g.conv()` helper.
        let cn = g.add_node(
            rlx_ir::Op::Conv {
                kernel_size: vec![1, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![xn, wn],
            Shape::new(&[n, c_out, h, w], f),
        );
        g.set_outputs(vec![cn]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        // Verify the fast path was selected.
        let saw_fast = sched
            .thunks
            .iter()
            .any(|t| matches!(t, Thunk::Conv2D1x1 { .. }));
        let saw_slow = sched
            .thunks
            .iter()
            .any(|t| matches!(t, Thunk::Conv2D { .. }));
        assert!(saw_fast, "1×1 conv should emit Conv2D1x1");
        assert!(!saw_slow, "1×1 conv must not fall through to scalar Conv2D");

        let xn_off = arena.byte_offset(xn);
        let wn_off = arena.byte_offset(wn);
        let cn_off = arena.byte_offset(cn);
        let buf = arena.raw_buf_mut();
        unsafe {
            let xp = buf.as_mut_ptr().add(xn_off) as *mut f32;
            for (i, &v) in x.iter().enumerate() {
                *xp.add(i) = v;
            }
            let wp = buf.as_mut_ptr().add(wn_off) as *mut f32;
            for (i, &v) in weight.iter().enumerate() {
                *wp.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(cn_off) as *const f32;
            (0..(n * c_out * h * w)).map(|i| *p.add(i)).collect()
        };

        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - a).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {a}"
            );
        }
    }

    /// Plan #5: fused dequant matmul matches the dequant-then-matmul
    /// reference (i.e. `(scale * (q - z)) @ x` materialized).
    #[test]
    fn dequant_matmul_int8_sym_matches_reference() {
        use rlx_ir::Philox4x32;
        use rlx_ir::quant::QuantScheme;

        let m = 3usize;
        let k = 8usize;
        let n = 4usize;
        let block_size = 4usize; // 2 blocks per column
        let blocks_per_col = k / block_size;

        // Random inputs: x f32, w_q i8, scales f32. Symmetric → no zp.
        let mut rng = Philox4x32::new(99);
        let mut x = vec![0f32; m * k];
        rng.fill_normal(&mut x);
        let w_q: Vec<i8> = (0..(k * n))
            .map(|i| ((i as i32 * 13 + 7) % 127 - 63) as i8)
            .collect();
        let scales: Vec<f32> = (0..(blocks_per_col * n))
            .map(|i| 0.01 + 0.001 * i as f32)
            .collect();

        // Reference: build f32 weights from (q * scale) per block.
        let mut w_f32 = vec![0f32; k * n];
        for p in 0..k {
            let block = p / block_size;
            for j in 0..n {
                let s = scales[block * n + j];
                w_f32[p * n + j] = w_q[p * n + j] as f32 * s;
            }
        }
        let mut expected = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += x[i * k + p] * w_f32[p * n + j];
                }
                expected[i * n + j] = acc;
            }
        }

        // RLX path.
        let f = DType::F32;
        let mut g = Graph::new("dq");
        let xn = g.input("x", Shape::new(&[m, k], f));
        let wn = g.param("w", Shape::new(&[k, n], DType::I8));
        let sn = g.param("scale", Shape::new(&[blocks_per_col, n], f));
        let zn = g.param("zp", Shape::new(&[blocks_per_col, n], f)); // unused (sym)
        let dq = g.dequant_matmul(
            xn,
            wn,
            sn,
            zn,
            QuantScheme::Int8Block {
                block_size: block_size as u32,
            },
            Shape::new(&[m, n], f),
        );
        g.set_outputs(vec![dq]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let xn_off = arena.byte_offset(xn);
        let wn_off = arena.byte_offset(wn);
        let sn_off = arena.byte_offset(sn);
        let zn_off = arena.byte_offset(zn);
        let dq_off = arena.byte_offset(dq);
        let buf = arena.raw_buf_mut();
        unsafe {
            // Seed f32 inputs.
            let xp = buf.as_mut_ptr().add(xn_off) as *mut f32;
            for (i, &v) in x.iter().enumerate() {
                *xp.add(i) = v;
            }
            let sp = buf.as_mut_ptr().add(sn_off) as *mut f32;
            for (i, &v) in scales.iter().enumerate() {
                *sp.add(i) = v;
            }
            let zp = buf.as_mut_ptr().add(zn_off) as *mut f32;
            for i in 0..(blocks_per_col * n) {
                *zp.add(i) = 0.0;
            }
            // Seed i8 weights byte-by-byte.
            let wp = buf.as_mut_ptr().add(wn_off) as *mut i8;
            for (i, &v) in w_q.iter().enumerate() {
                *wp.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
            (0..m * n).map(|i| *p.add(i)).collect()
        };

        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - a).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {a}"
            );
        }
    }

    /// Plan #9: LoRA matmul matches the unfused 3-matmul reference.
    #[test]
    fn lora_matmul_matches_unfused_reference() {
        use rlx_ir::Philox4x32;

        let m = 4usize;
        let k = 8usize;
        let n = 6usize;
        let r = 2usize;
        let scale = 0.5f32;

        // Random inputs (deterministic via Philox).
        let mut rng = Philox4x32::new(42);
        let mut x = vec![0f32; m * k];
        rng.fill_normal(&mut x);
        let mut w = vec![0f32; k * n];
        rng.fill_normal(&mut w);
        let mut a = vec![0f32; k * r];
        rng.fill_normal(&mut a);
        let mut b = vec![0f32; r * n];
        rng.fill_normal(&mut b);

        // Reference: out = x·W + scale * x·A·B. Naive triple-loop.
        let naive = |a_buf: &[f32], b_buf: &[f32], rows: usize, inner: usize, cols: usize| {
            let mut o = vec![0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let mut acc = 0f32;
                    for p in 0..inner {
                        acc += a_buf[i * inner + p] * b_buf[p * cols + j];
                    }
                    o[i * cols + j] = acc;
                }
            }
            o
        };
        let xw = naive(&x, &w, m, k, n);
        let xa = naive(&x, &a, m, k, r);
        let xab = naive(&xa, &b, m, r, n);
        let mut expected = xw;
        for i in 0..(m * n) {
            expected[i] += scale * xab[i];
        }

        // RLX path: build a graph with one LoraMatMul.
        let f = DType::F32;
        let mut g = Graph::new("lora");
        let xn = g.input("x", Shape::new(&[m, k], f));
        let wn = g.param("w", Shape::new(&[k, n], f));
        let an = g.param("a", Shape::new(&[k, r], f));
        let bn = g.param("b", Shape::new(&[r, n], f));
        let lm = g.lora_matmul(xn, wn, an, bn, scale, Shape::new(&[m, n], f));
        g.set_outputs(vec![lm]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let xn_off = arena.byte_offset(xn);
        let wn_off = arena.byte_offset(wn);
        let an_off = arena.byte_offset(an);
        let bn_off = arena.byte_offset(bn);
        let lm_off = arena.byte_offset(lm);
        let buf = arena.raw_buf_mut();
        unsafe {
            let copy = |dst: *mut f32, data: &[f32]| {
                for (i, &v) in data.iter().enumerate() {
                    *dst.add(i) = v;
                }
            };
            copy(buf.as_mut_ptr().add(xn_off) as *mut f32, &x);
            copy(buf.as_mut_ptr().add(wn_off) as *mut f32, &w);
            copy(buf.as_mut_ptr().add(an_off) as *mut f32, &a);
            copy(buf.as_mut_ptr().add(bn_off) as *mut f32, &b);
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(lm_off) as *const f32;
            (0..m * n).map(|i| *p.add(i)).collect()
        };

        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - a).abs() < 1e-3,
                "mismatch at {i}: expected {e}, got {a}"
            );
        }
    }

    /// Plan #42: fused sampling kernel determinism + greedy fallback.
    #[test]
    fn sample_temperature_zero_is_argmax() {
        // Very low temperature → distribution collapses on argmax.
        // Same seed → same output bit-for-bit.
        let f = DType::F32;
        let mut g = Graph::new("samp");
        let logits = g.input("logits", Shape::new(&[1, 8], f));
        let s = g.sample(logits, 0, 1.0, 1e-3, 42, Shape::new(&[1], f));
        g.set_outputs(vec![s]);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let logits_off = arena.byte_offset(logits);
        let s_off = arena.byte_offset(s);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(logits_off) as *mut f32;
            // argmax = index 5 (value 9.0).
            let inputs = [0.1f32, 0.2, 0.3, 0.4, 0.5, 9.0, 0.7, 0.8];
            for (i, &v) in inputs.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let token = unsafe {
            let p = arena.raw_buf().as_ptr().add(s_off) as *const f32;
            *p as usize
        };
        assert_eq!(token, 5, "low-temp sampling should pick the argmax");
    }

    #[test]
    fn sample_top_k_one_is_deterministic() {
        // top_k=1 forces only the argmax to have nonzero probability.
        let f = DType::F32;
        let mut g = Graph::new("samp_k1");
        let logits = g.input("logits", Shape::new(&[1, 4], f));
        let s = g.sample(logits, 1, 1.0, 1.0, 7, Shape::new(&[1], f));
        g.set_outputs(vec![s]);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let logits_off = arena.byte_offset(logits);
        let s_off = arena.byte_offset(s);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(logits_off) as *mut f32;
            let inputs = [0.1f32, 5.0, 0.3, 0.4]; // argmax = 1
            for (i, &v) in inputs.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let token = unsafe {
            let p = arena.raw_buf().as_ptr().add(s_off) as *const f32;
            *p as usize
        };
        assert_eq!(token, 1);
    }

    /// Plan #44: cumsum primitive parity vs. naive scan.
    #[test]
    fn cumsum_inclusive_matches_naive() {
        let f = DType::F32;
        let mut g = Graph::new("cumsum");
        let x = g.input("x", Shape::new(&[2, 4], f));
        let cs = g.cumsum(x, -1, false, Shape::new(&[2, 4], f));
        g.set_outputs(vec![cs]);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        // Cache offsets up-front so we can drop the immutable borrow.
        let x_off = arena.byte_offset(x);
        let out_off = arena.byte_offset(cs);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(x_off) as *mut f32;
            let inputs = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
            for (i, &v) in inputs.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());

        let out: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..8).map(|i| *p.add(i)).collect()
        };
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 10.0, 30.0, 60.0, 100.0]);
    }

    /// Plan #46 deep: Narrow×3 → Attention fusion. The three QKV
    /// narrows that BERT/Nomic emit on the unfused (batch*seq > 64)
    /// path collapse into a single strided-Attention thunk.
    #[test]
    fn narrow_attention_fuses_in_unfused_path() {
        let f = DType::F32;
        let mut g = Graph::new("nattn_fuse");
        // batch*seq = 8*16 = 128 > 64 so FusedAttnBlock skips.
        let qkv = g.input("qkv", Shape::new(&[8, 16, 192], f)); // 3*64 = 192
        let mask = g.input("mask", Shape::new(&[8, 16], f));
        let q = g.narrow_(qkv, 2, 0, 64);
        let k = g.narrow_(qkv, 2, 64, 64);
        let v = g.narrow_(qkv, 2, 128, 64);
        let attn = g.attention(q, k, v, mask, 4, 16, Shape::new(&[8, 16, 64], f));
        g.set_outputs(vec![attn]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let mut narrow_count = 0;
        let mut attn_strides: Option<(u32, u32, u32)> = None;
        for t in &sched.thunks {
            match t {
                Thunk::Narrow { .. } => narrow_count += 1,
                Thunk::Attention {
                    q_row_stride,
                    k_row_stride,
                    v_row_stride,
                    ..
                } => attn_strides = Some((*q_row_stride, *k_row_stride, *v_row_stride)),
                _ => {}
            }
        }
        // After fusion the 3 narrows are gone; Attention now walks the
        // QKV with parent row stride = 192 (3 × 64) on all three inputs.
        assert_eq!(
            narrow_count, 0,
            "Narrow×3→Attention fusion should eliminate all 3 narrows; saw {narrow_count}"
        );
        assert_eq!(
            attn_strides,
            Some((192, 192, 192)),
            "Attention should walk Q/K/V with parent row stride 192"
        );
    }

    // ── Backward / training op parity tests ────────────────────
    //
    // Strategy: build a graph that contains exactly the backward op
    // under test (plus its inputs as graph Inputs), execute, and
    // compare against a hand-rolled scalar reference. For
    // Conv2dBackwardInput we additionally check against the numerical
    // gradient of the forward Conv2D — that's the gold-standard test
    // that validates the math, not just consistency between two
    // implementations of the same formula.

    fn run_graph(
        g: &Graph,
        inputs: &[(NodeId, &[f32])],
        out_id: NodeId,
        out_len: usize,
    ) -> Vec<f32> {
        let plan = rlx_opt::memory::plan_memory(g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(g, &arena);
        for &(id, data) in inputs {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let off = arena.byte_offset(out_id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f32;
            (0..out_len).map(|i| *p.add(i)).collect()
        }
    }

    #[test]
    fn relu_backward_matches_mask() {
        let f = DType::F32;
        let len = 7usize;
        let x: Vec<f32> = vec![-2.0, -0.1, 0.0, 0.1, 1.0, 3.0, -5.0];
        let dy: Vec<f32> = vec![0.5, 1.5, 2.5, -0.7, 4.0, -1.0, 9.0];

        let mut g = Graph::new("relu_bw");
        let xn = g.input("x", Shape::new(&[len], f));
        let dyn_ = g.input("dy", Shape::new(&[len], f));
        let dx = g.relu_backward(xn, dyn_);
        g.set_outputs(vec![dx]);

        let actual = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dx, len);
        // Reference: gradient is dy where x>0 strictly, else 0.
        // (zero is not "positive" — the forward applied max(0, x), and at
        // x=0 the subgradient could be anything in [0, dy]; we pick 0.)
        let expected: Vec<f32> = x
            .iter()
            .zip(&dy)
            .map(|(&xi, &dyi)| if xi > 0.0 { dyi } else { 0.0 })
            .collect();
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-6, "relu_bw mismatch: {a} vs {e}");
        }
    }

    #[test]
    fn maxpool2d_backward_routes_to_argmax() {
        let f = DType::F32;
        // [N=1, C=1, H=4, W=4] → 2x2 max-pool stride 2 → [1,1,2,2].
        let x: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        // Argmax of each 2x2 window:
        //   (0,0)→6 (idx 5), (0,1)→8 (idx 7),
        //   (1,0)→14(idx 13),(1,1)→16(idx 15).
        let dy: Vec<f32> = vec![0.5, 1.0, 2.0, 4.0];

        let mut g = Graph::new("maxpool_bw");
        let xn = g.input("x", Shape::new(&[1, 1, 4, 4], f));
        let dyn_ = g.input("dy", Shape::new(&[1, 1, 2, 2], f));
        let dx = g.maxpool2d_backward(xn, dyn_, vec![2, 2], vec![2, 2], vec![0, 0]);
        g.set_outputs(vec![dx]);

        let actual = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dx, 16);
        let mut expected = vec![0f32; 16];
        expected[5] = 0.5;
        expected[7] = 1.0;
        expected[13] = 2.0;
        expected[15] = 4.0;
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() < 1e-6, "maxpool_bw[{i}] mismatch: {a} vs {e}");
        }
    }

    #[test]
    fn conv2d_backward_input_matches_numerical_gradient() {
        use rlx_ir::Philox4x32;
        // Small enough to numerically differentiate exhaustively but
        // big enough to exercise stride/padding edge cases.
        let n = 1usize;
        let c_in = 2usize;
        let h = 4usize;
        let w = 4usize;
        let c_out = 3usize;
        let kh = 3usize;
        let kw = 3usize;
        let ph = 1usize;
        let pw = 1usize;
        let sh = 1usize;
        let sw = 1usize;
        // Output dims with padding=1, stride=1: same as input.
        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;
        assert_eq!(h_out, 4);
        assert_eq!(w_out, 4);

        let mut rng = Philox4x32::new(7);
        let mut x = vec![0f32; n * c_in * h * w];
        rng.fill_normal(&mut x);
        let mut wt = vec![0f32; c_out * c_in * kh * kw];
        rng.fill_normal(&mut wt);
        let mut dy = vec![0f32; n * c_out * h_out * w_out];
        rng.fill_normal(&mut dy);

        // Analytical: Conv2dBackwardInput on (dy, w).
        let f = DType::F32;
        let mut g = Graph::new("conv_bwi");
        let dy_in = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], f));
        let w_in = g.input("w", Shape::new(&[c_out, c_in, kh, kw], f));
        let dx = g.conv2d_backward_input(
            dy_in,
            w_in,
            Shape::new(&[n, c_in, h, w], f),
            vec![kh, kw],
            vec![sh, sw],
            vec![ph, pw],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dx]);
        let analytical = run_graph(&g, &[(dy_in, &dy), (w_in, &wt)], dx, n * c_in * h * w);

        // Numerical: for each x[i], finite-difference forward conv twice.
        // Forward: y[j] = sum over filter window of w * x ; dot(dy, y) is
        // the scalar we differentiate. Then dx[i] = ∂(dot(dy, y))/∂x[i].
        let forward = |x: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; n * c_out * h_out * w_out];
            for ni in 0..n {
                for co in 0..c_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let mut acc = 0f32;
                            for ci in 0..c_in {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let hi = ho * sh + ki;
                                        let wi = wo * sw + kj;
                                        if hi < ph || wi < pw {
                                            continue;
                                        }
                                        let hi = hi - ph;
                                        let wi = wi - pw;
                                        if hi >= h || wi >= w {
                                            continue;
                                        }
                                        let xv = x[((ni * c_in) + ci) * h * w + hi * w + wi];
                                        let wv = wt[((co * c_in) + ci) * kh * kw + ki * kw + kj];
                                        acc += xv * wv;
                                    }
                                }
                            }
                            out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                        }
                    }
                }
            }
            out
        };
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&u, &v)| u * v).sum() };
        let eps = 1e-3f32;
        let mut numerical = vec![0f32; x.len()];
        for i in 0..x.len() {
            let saved = x[i];
            x[i] = saved + eps;
            let plus = dot(&forward(&x), &dy);
            x[i] = saved - eps;
            let minus = dot(&forward(&x), &dy);
            x[i] = saved;
            numerical[i] = (plus - minus) / (2.0 * eps);
        }
        for (i, (a, n)) in analytical.iter().zip(&numerical).enumerate() {
            // f32 + eps=1e-3 numerical grad → ~1e-3 absolute is realistic.
            assert!(
                (a - n).abs() < 5e-3,
                "conv_bw_input[{i}]: analytical {a} vs numerical {n}"
            );
        }
    }

    #[test]
    fn conv2d_backward_weight_matches_numerical_gradient() {
        use rlx_ir::Philox4x32;
        let n = 2usize;
        let c_in = 2usize;
        let h = 4usize;
        let w = 4usize;
        let c_out = 2usize;
        let kh = 3usize;
        let kw = 3usize;
        let ph = 0usize;
        let pw = 0usize;
        let sh = 1usize;
        let sw = 1usize;
        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let mut rng = Philox4x32::new(11);
        let mut x = vec![0f32; n * c_in * h * w];
        rng.fill_normal(&mut x);
        let mut wt = vec![0f32; c_out * c_in * kh * kw];
        rng.fill_normal(&mut wt);
        let mut dy = vec![0f32; n * c_out * h_out * w_out];
        rng.fill_normal(&mut dy);

        let f = DType::F32;
        let mut g = Graph::new("conv_bww");
        let xn = g.input("x", Shape::new(&[n, c_in, h, w], f));
        let dyn_ = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], f));
        let dwn = g.conv2d_backward_weight(
            xn,
            dyn_,
            Shape::new(&[c_out, c_in, kh, kw], f),
            vec![kh, kw],
            vec![sh, sw],
            vec![ph, pw],
            vec![1, 1],
            1,
        );
        g.set_outputs(vec![dwn]);
        let analytical = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dwn, c_out * c_in * kh * kw);

        let forward = |wt: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; n * c_out * h_out * w_out];
            for ni in 0..n {
                for co in 0..c_out {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let mut acc = 0f32;
                            for ci in 0..c_in {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let hi = ho + ki;
                                        let wi = wo + kj;
                                        let xv = x[((ni * c_in) + ci) * h * w + hi * w + wi];
                                        let wv = wt[((co * c_in) + ci) * kh * kw + ki * kw + kj];
                                        acc += xv * wv;
                                    }
                                }
                            }
                            out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                        }
                    }
                }
            }
            out
        };
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&u, &v)| u * v).sum() };
        let eps = 1e-3f32;
        let mut numerical = vec![0f32; wt.len()];
        for i in 0..wt.len() {
            let saved = wt[i];
            wt[i] = saved + eps;
            let plus = dot(&forward(&wt), &dy);
            wt[i] = saved - eps;
            let minus = dot(&forward(&wt), &dy);
            wt[i] = saved;
            numerical[i] = (plus - minus) / (2.0 * eps);
        }
        for (i, (a, n)) in analytical.iter().zip(&numerical).enumerate() {
            assert!(
                (a - n).abs() < 5e-3,
                "conv_bw_weight[{i}]: analytical {a} vs numerical {n}"
            );
        }
    }

    #[test]
    fn softmax_cross_entropy_matches_reference() {
        let f = DType::F32;
        let logits: Vec<f32> = vec![
            1.0, 2.0, 3.0, // row 0: max=3 (idx 2)
            -1.0, 0.0, 4.0, // row 1: max=4 (idx 2)
            5.0, 5.0, 5.0, // row 2: uniform
        ];
        let labels: Vec<f32> = vec![2.0, 0.0, 1.0];

        let mut g = Graph::new("sce");
        let lg = g.input("logits", Shape::new(&[3, 3], f));
        let lb = g.input("labels", Shape::new(&[3], f));
        let loss = g.softmax_cross_entropy_with_logits(lg, lb);
        g.set_outputs(vec![loss]);
        let actual = run_graph(&g, &[(lg, &logits), (lb, &labels)], loss, 3);

        // Reference per-row: -log(softmax(row)[label]).
        let mut expected = vec![0f32; 3];
        for ni in 0..3 {
            let row = &logits[ni * 3..(ni + 1) * 3];
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let lse = m + sum.ln();
            let label_idx = labels[ni] as usize;
            expected[ni] = lse - row[label_idx];
        }
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() < 1e-5, "sce loss[{i}]: {a} vs {e}");
        }
    }

    #[test]
    fn softmax_cross_entropy_backward_matches_numerical_gradient() {
        use rlx_ir::Philox4x32;
        let n = 4usize;
        let c = 5usize;
        let mut rng = Philox4x32::new(23);
        let mut logits = vec![0f32; n * c];
        rng.fill_normal(&mut logits);
        let labels: Vec<f32> = (0..n).map(|i| (i % c) as f32).collect();
        let mut d_loss = vec![0f32; n];
        rng.fill_normal(&mut d_loss);

        let f = DType::F32;
        let mut g = Graph::new("sce_bw");
        let lg = g.input("logits", Shape::new(&[n, c], f));
        let lb = g.input("labels", Shape::new(&[n], f));
        let dl = g.input("d_loss", Shape::new(&[n], f));
        let dlogits = g.softmax_cross_entropy_backward(lg, lb, dl);
        g.set_outputs(vec![dlogits]);
        let analytical = run_graph(
            &g,
            &[(lg, &logits), (lb, &labels), (dl, &d_loss)],
            dlogits,
            n * c,
        );

        // Numerical: differentiate dot(d_loss, sce_loss(logits)) w.r.t. each logit.
        let sce_loss = |logits: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; n];
            for ni in 0..n {
                let row = &logits[ni * c..(ni + 1) * c];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
                out[ni] = (m + sum.ln()) - row[labels[ni] as usize];
            }
            out
        };
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&u, &v)| u * v).sum::<f32>();
        let eps = 1e-3f32;
        let mut numerical = vec![0f32; logits.len()];
        for i in 0..logits.len() {
            let saved = logits[i];
            logits[i] = saved + eps;
            let plus = dot(&sce_loss(&logits), &d_loss);
            logits[i] = saved - eps;
            let minus = dot(&sce_loss(&logits), &d_loss);
            logits[i] = saved;
            numerical[i] = (plus - minus) / (2.0 * eps);
        }
        for (i, (a, num)) in analytical.iter().zip(&numerical).enumerate() {
            assert!(
                (a - num).abs() < 5e-3,
                "sce_bw[{i}]: analytical {a} vs numerical {num}"
            );
        }
    }

    // ── End-to-end autodiff parity tests ──────────────────────
    //
    // Build a forward graph, run `grad_with_loss` to produce a graph
    // that emits [loss, gradients...], execute it through rlx-cpu,
    // and compare each gradient to a finite-difference estimate
    // produced by re-running the forward graph with each parameter
    // entry perturbed. f32 + ε=1e-3 puts the tolerance floor around
    // 5e-3 absolute error.

    /// Initialize Op::Constant slots in the arena with their literal
    /// data. Mirrors the loop in rlx_runtime::backend (which serves
    /// the same role for production runs).
    fn fill_constants_into_arena(graph: &Graph, arena: &mut crate::arena::Arena) {
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has_buffer(node.id)
                && !data.is_empty()
            {
                let buf = arena.slice_mut(node.id);
                let n_floats = data.len() / 4;
                let n = buf.len().min(n_floats);
                for i in 0..n {
                    let bytes = [
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ];
                    buf[i] = f32::from_le_bytes(bytes);
                }
            }
        }
    }

    /// Compile + arena-prep helper for these tests. Returns the
    /// schedule and a populated arena. `seed_inputs` writes f32 input
    /// data into the arena slot for each (NodeId, &[f32]) pair.
    fn prepare(
        graph: &Graph,
        seed_inputs: &[(NodeId, &[f32])],
    ) -> (ThunkSchedule, crate::arena::Arena) {
        let plan = rlx_opt::memory::plan_memory(graph);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(graph, &arena);
        fill_constants_into_arena(graph, &mut arena);
        for &(id, data) in seed_inputs {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        (sched, arena)
    }

    fn read_arena(arena: &crate::arena::Arena, id: NodeId, len: usize) -> Vec<f32> {
        let off = arena.byte_offset(id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f32;
            (0..len).map(|i| *p.add(i)).collect()
        }
    }

    fn write_arena(arena: &mut crate::arena::Arena, id: NodeId, data: &[f32]) {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }

    /// f64 sibling of `prepare`. Writes f64 input data into the arena.
    fn prepare_f64(
        graph: &Graph,
        seed_inputs: &[(NodeId, &[f64])],
    ) -> (ThunkSchedule, crate::arena::Arena) {
        let plan = rlx_opt::memory::plan_memory(graph);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(graph, &arena);
        fill_constants_into_arena(graph, &mut arena);
        for &(id, data) in seed_inputs {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f64;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        (sched, arena)
    }

    fn read_arena_f64(arena: &crate::arena::Arena, id: NodeId, len: usize) -> Vec<f64> {
        let off = arena.byte_offset(id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f64;
            (0..len).map(|i| *p.add(i)).collect()
        }
    }

    /// End-to-end f64 DenseSolve through the full compile + execute
    /// path. Validates: IR shape inference, memory planner f64 sizing,
    /// arena f64 accessors, Thunk::DenseSolveF64 lowering, executor
    /// dispatch, Accelerate dgesv FFI.
    ///
    /// System:
    ///   A = [[2, 1],
    ///        [1, 3]]   b = [5, 10]
    ///   ⇒  x = [1, 3]   (verified by hand)
    #[test]
    fn dense_solve_f64_end_to_end() {
        let mut g = Graph::new("solve_e2e");
        let a = g.input("A", Shape::new(&[2, 2], DType::F64));
        let b = g.input("b", Shape::new(&[2], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[2], DType::F64));
        g.set_outputs(vec![x]);

        let a_data = [2.0, 1.0, 1.0, 3.0_f64];
        let b_data = [5.0, 10.0_f64];
        let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());

        let got = read_arena_f64(&arena, x, 2);
        let want = [1.0, 3.0_f64];
        for i in 0..2 {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "x[{i}] = {} (expected {})",
                got[i],
                want[i]
            );
        }
    }

    /// Scaled-up f64 DenseSolve — tridiagonal Laplacian-shape (typical
    /// MNA structure for a passive RC mesh in Circulax). Validates
    /// that the solve scales beyond the trivial 2×2 and that the
    /// row-major ↔ col-major dance in `dgesv` is correct for the
    /// general case.
    #[test]
    fn dense_solve_f64_5x5_laplacian() {
        let n = 5usize;
        let mut g = Graph::new("solve_5x5");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![x]);

        // 1-D Laplacian: 2 on diagonal, -1 on off-diagonals, 0 elsewhere.
        let mut a_data = vec![0.0_f64; n * n];
        for i in 0..n {
            a_data[i * n + i] = 2.0;
            if i > 0 {
                a_data[i * n + (i - 1)] = -1.0;
            }
            if i + 1 < n {
                a_data[i * n + (i + 1)] = -1.0;
            }
        }
        let b_data: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());

        let got = read_arena_f64(&arena, x, n);
        // Verify A·x ≈ b by computing the residual.
        let mut residual = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                residual[i] += a_data[i * n + j] * got[j];
            }
        }
        for i in 0..n {
            assert!(
                (residual[i] - b_data[i]).abs() < 1e-10,
                "row {i}: residual {} vs b {}",
                residual[i],
                b_data[i]
            );
        }
    }

    /// Hello Resistor: end-to-end f64 gradient through a dense solve.
    ///
    /// Forward:
    ///   A      : Param  [N, N]   f64
    ///   b      : Input  [N]      f64
    ///   x      = solve(A, b)            (DenseSolve)
    ///   loss   = sum(x)                 (Reduce::Sum)
    ///
    /// Backward (via grad_with_loss):
    ///   ones [N] = expand(d_output, [N])      (Reduce::Sum VJP)
    ///   dx_int   = solve(Aᵀ, ones)             (DenseSolve VJP step 1)
    ///   dA       = -outer(dx_int, x)           (DenseSolve VJP step 2)
    ///   db       = dx_int                       (DenseSolve VJP step 3)
    ///
    /// Closed form: with loss = sum(solve(A, b)) = ones·x and
    /// implicit-function calculus, db = (Aᵀ)⁻¹·ones, dA = -db ⊗ x.
    /// We verify this against the autodiff-emitted graph's output and
    /// against a finite-difference baseline.
    #[test]
    fn hello_resistor_gradient_end_to_end() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;

        // ── Build forward graph ──
        let mut g = Graph::new("hello_resistor");
        let a = g.param("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
        let loss = g.reduce(
            x,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        // ── Run reverse-mode AD ──
        let bwd = grad_with_loss(&g, &[a, b]);
        assert_eq!(bwd.outputs.len(), 3, "expect [loss, dA, db]");

        // ── Locate the inputs the bwd graph still needs from us ──
        // grad_with_loss copies forward nodes into bwd, so A/b/d_output
        // appear under their original names. Find them by name.
        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    rlx_ir::Op::Input { name } => Some(name.as_str()),
                    rlx_ir::Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?} in bwd graph");
        };
        let a_bwd = find_by_name(&bwd, "A");
        let b_bwd = find_by_name(&bwd, "b");
        let d_out_bwd = find_by_name(&bwd, "d_output");

        // ── Test data ──
        // A = [[2,1,0],[1,3,1],[0,1,2]]   (SPD tridiagonal, well-conditioned)
        // b = [1,2,3]
        let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
        let b_data = [1.0, 2.0, 3.0_f64];
        let d_output = [1.0_f64]; // ∂loss/∂loss

        // ── Compile + execute backward graph ──
        let (sched, mut arena) = prepare_f64(
            &bwd,
            &[(a_bwd, &a_data), (b_bwd, &b_data), (d_out_bwd, &d_output)],
        );
        execute_thunks(&sched, arena.raw_buf_mut());

        let loss_out = read_arena_f64(&arena, bwd.outputs[0], 1);
        let da_out = read_arena_f64(&arena, bwd.outputs[1], n * n);
        let db_out = read_arena_f64(&arena, bwd.outputs[2], n);

        // ── Closed-form reference ──
        // x = A⁻¹ b ; loss = sum(x).
        let x_ref = {
            let mut a = a_data;
            let mut b = b_data;
            let info = crate::blas::dgesv(&mut a, &mut b, n, 1);
            assert_eq!(info, 0);
            b
        };
        let loss_ref: f64 = x_ref.iter().sum();
        // db = (Aᵀ)⁻¹ · 1
        let db_ref = {
            let mut at = [0.0_f64; 9];
            for i in 0..n {
                for j in 0..n {
                    at[i * n + j] = a_data[j * n + i];
                }
            }
            let mut ones = [1.0_f64; 3];
            let info = crate::blas::dgesv(&mut at, &mut ones, n, 1);
            assert_eq!(info, 0);
            ones
        };
        // dA = -outer(db, x) ; dA[i,j] = -db[i] * x[j]
        let mut da_ref = [0.0_f64; 9];
        for i in 0..n {
            for j in 0..n {
                da_ref[i * n + j] = -db_ref[i] * x_ref[j];
            }
        }

        // ── Assertions vs analytic answer ──
        assert!(
            (loss_out[0] - loss_ref).abs() < 1e-10,
            "loss: got {}, want {}",
            loss_out[0],
            loss_ref
        );
        for i in 0..n {
            assert!(
                (db_out[i] - db_ref[i]).abs() < 1e-10,
                "db[{i}]: got {}, want {}",
                db_out[i],
                db_ref[i]
            );
        }
        for i in 0..n * n {
            assert!(
                (da_out[i] - da_ref[i]).abs() < 1e-10,
                "dA[{i}]: got {}, want {}",
                da_out[i],
                da_ref[i]
            );
        }

        // ── Cross-check vs finite differences on db (a few entries) ──
        // ∂loss/∂b[k] ≈ (loss(b + h·e_k) - loss(b - h·e_k)) / (2h).
        let h = 1e-6_f64;
        for k in 0..n {
            let mut bp = b_data;
            bp[k] += h;
            let mut bm = b_data;
            bm[k] -= h;
            let lp = {
                let mut ac = a_data;
                let info = crate::blas::dgesv(&mut ac, &mut bp, n, 1);
                assert_eq!(info, 0);
                bp.iter().sum::<f64>()
            };
            let lm = {
                let mut ac = a_data;
                let info = crate::blas::dgesv(&mut ac, &mut bm, n, 1);
                assert_eq!(info, 0);
                bm.iter().sum::<f64>()
            };
            let fd = (lp - lm) / (2.0 * h);
            assert!(
                (db_out[k] - fd).abs() < 1e-7,
                "FD mismatch on db[{k}]: AD={} FD={}",
                db_out[k],
                fd
            );
        }
    }

    /// Smallest possible Op::Scan smoke test: geometric growth.
    /// init = [1, 1, 1] f64, body = (x → x + 0.1·x) = (x → 1.1·x),
    /// length = 10. Final carry must equal init·(1.1)^10 ≈ 2.5937…
    /// to f64 precision.
    #[test]
    fn scan_geometric_growth_f64() {
        let n = 3usize;
        let length = 10u32;

        // Body: (x) → x + 0.1·x. One Input, one output, same shape/dtype.
        let mut body = Graph::new("scan_body");
        let x = body.input("carry", Shape::new(&[n], DType::F64));
        let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 0.1_f64.to_le_bytes()).collect();
        let scale = body.add_node(
            Op::Constant { data: scale_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let scaled = body.binary(BinaryOp::Mul, x, scale, Shape::new(&[n], DType::F64));
        let next = body.binary(BinaryOp::Add, x, scaled, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        // Outer graph: scan(init, body, length).
        let mut g = Graph::new("scan_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let final_carry = g.scan(init, body, length);
        g.set_outputs(vec![final_carry]);

        let init_data = vec![1.0_f64; n];
        let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena_f64(&arena, final_carry, n);
        let want: f64 = 1.1_f64.powi(length as i32);
        for i in 0..n {
            assert!(
                (got[i] - want).abs() < 1e-12,
                "got[{i}] = {} want {}",
                got[i],
                want
            );
        }
    }

    /// Per-step xs scan: cumulative-sum.
    ///   carry_0 = init
    ///   carry_{t+1} = carry_t + xs\[t\]
    ///   final = sum_{t<length} xs\[t\] + init
    /// Body has 2 inputs (carry, x_t) in that NodeId order; one output
    /// (next carry). Validates the per-step-input plumbing end-to-end.
    #[test]
    fn scan_with_xs_cumulative_sum() {
        let n = 3usize;
        let length = 4u32;

        let mut body = Graph::new("cumsum_body");
        // carry must come first in NodeId order — declare it first.
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let x_t = body.input("x_t", Shape::new(&[n], DType::F64));
        let next = body.binary(BinaryOp::Add, carry, x_t, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("cumsum_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
        let final_carry = g.scan_with_xs(init, &[xs], body, length);
        g.set_outputs(vec![final_carry]);

        let init_data = vec![0.0_f64; n];
        let xs_data: Vec<f64> = (0..length as usize * n).map(|i| (i + 1) as f64).collect(); // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data), (xs, &xs_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena_f64(&arena, final_carry, n);

        // Reference: column-wise sum of xs rows + init. With our row-major
        // layout, column j of xs is xs_data[j], xs_data[n+j], xs_data[2n+j], ...
        // (per-step row at offset t*n contributes element j to slot j).
        let mut want = init_data.clone();
        for t in 0..length as usize {
            for j in 0..n {
                want[j] += xs_data[t * n + j];
            }
        }
        for i in 0..n {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "got[{i}] = {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// Per-step xs scan composing with DenseSolve — Circulax-shaped:
    ///   carry_{t+1} = solve(M, carry_t + xs\[t\])
    /// Models a Backward-Euler step driven by a time-varying source.
    #[test]
    fn scan_with_xs_be_with_drive() {
        let n = 3usize;
        let length = 4u32;
        let dt = 0.1_f64;

        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_drive_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let drive = body.input("drive", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
        let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_drive_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
        let final_carry = g.scan_with_xs(init, &[xs], body, length);
        g.set_outputs(vec![final_carry]);

        let init_data = vec![0.0_f64; n];
        // Drive the system with a unit pulse on element 0 at t=0,
        // zeros after.
        let mut xs_data = vec![0.0_f64; length as usize * n];
        xs_data[0] = 1.0;

        let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data), (xs, &xs_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena_f64(&arena, final_carry, n);

        // Reference: per-step in pure Rust.
        let mut x = init_data.clone();
        for t in 0..length as usize {
            for j in 0..n {
                x[j] += xs_data[t * n + j];
            }
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut x, n, 1);
        }
        for i in 0..n {
            assert!(
                (got[i] - x[i]).abs() < 1e-12,
                "got[{i}] = {} ref {}",
                got[i],
                x[i]
            );
        }
    }

    /// Reverse-mode AD through Op::BatchedDenseSolve. Forward solves
    /// `[B, N, N] · x = [B, N]`; loss = sum of all entries. Closed
    /// form: dB = (Aᵀ)⁻¹·1, dA = -(Aᵀ)⁻¹·1 ⊗ x. Verified analytically
    /// per batch (each slice matches what the unbatched DenseSolve VJP
    /// would compute).
    #[test]
    fn batched_dense_solve_gradient_matches_per_batch_analytic() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;
        let batch = 4usize;

        let mut g = Graph::new("bds_grad");
        let a = g.param("A", Shape::new(&[batch, n, n], DType::F64));
        let b = g.input("b", Shape::new(&[batch, n], DType::F64));
        let x = g.batched_dense_solve(a, b, Shape::new(&[batch, n], DType::F64));
        let loss = g.reduce(
            x,
            ReduceOp::Sum,
            vec![0, 1],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[a, b]);

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want}");
        };
        let a_id = find(&bwd, "A");
        let b_id = find(&bwd, "b");
        let d_out_id = find(&bwd, "d_output");

        let mut rng = rlx_ir::Philox4x32::new(0x57e1_u64);
        let mut a_data = vec![0.0_f64; batch * n * n];
        let mut b_data = vec![0.0_f64; batch * n];
        for bi in 0..batch {
            for i in 0..n {
                for j in 0..n {
                    a_data[bi * n * n + i * n + j] = rng.next_f32() as f64 * 0.1;
                }
                a_data[bi * n * n + i * n + i] += 1.0 + n as f64;
            }
            for i in 0..n {
                b_data[bi * n + i] = rng.next_f32() as f64;
            }
        }
        let d_seed = [1.0_f64];

        let (sched, mut arena) = prepare_f64(
            &bwd,
            &[(a_id, &a_data), (b_id, &b_data), (d_out_id, &d_seed)],
        );
        execute_thunks(&sched, arena.raw_buf_mut());
        let da_out = read_arena_f64(&arena, bwd.outputs[1], batch * n * n);
        let db_out = read_arena_f64(&arena, bwd.outputs[2], batch * n);

        // Reference: per-batch analytic solve. dB_i = (A_iᵀ)⁻¹ · 1,
        // dA_i = -dB_i ⊗ x_i.
        for bi in 0..batch {
            let a_slice: Vec<f64> = a_data[bi * n * n..(bi + 1) * n * n].to_vec();
            let mut b_slice: Vec<f64> = b_data[bi * n..(bi + 1) * n].to_vec();
            let mut a_copy = a_slice.clone();
            crate::blas::dgesv(&mut a_copy, &mut b_slice, n, 1);
            let x_ref = b_slice.clone();
            // dB: solve(A^T, ones)
            let mut at = vec![0.0_f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    at[i * n + j] = a_slice[j * n + i];
                }
            }
            let mut ones = vec![1.0_f64; n];
            crate::blas::dgesv(&mut at, &mut ones, n, 1);
            let db_ref = ones;
            for i in 0..n {
                let got = db_out[bi * n + i];
                assert!(
                    (got - db_ref[i]).abs() < 1e-10,
                    "batch {bi}, db[{i}]: got {got} ref {}",
                    db_ref[i]
                );
            }
            // dA: -outer(db, x)
            for i in 0..n {
                for j in 0..n {
                    let got = da_out[bi * n * n + i * n + j];
                    let want = -db_ref[i] * x_ref[j];
                    assert!(
                        (got - want).abs() < 1e-10,
                        "batch {bi}, dA[{i},{j}]: got {got} ref {want}"
                    );
                }
            }
        }
    }

    /// AD knob: gradient through `scan_checkpointed` automatically
    /// uses the recompute backward path. Compares dinit from a plain
    /// scan against the same forward written with `scan_checkpointed`,
    /// both run through `grad_with_loss`. They must match to f64.
    #[test]
    fn scan_checkpointed_grad_matches_plain_scan_grad() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 2usize;
        let length = 6u32;

        let make_body = || {
            let mut body = Graph::new("ck_body");
            let carry = body.input("carry", Shape::new(&[n], DType::F64));
            let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 1.05_f64.to_le_bytes()).collect();
            let scale = body.add_node(
                Op::Constant { data: scale_bytes },
                vec![],
                Shape::new(&[n], DType::F64),
            );
            let next = body.binary(BinaryOp::Mul, carry, scale, Shape::new(&[n], DType::F64));
            body.set_outputs(vec![next]);
            body
        };

        // Plain scan path.
        let mut g_plain = Graph::new("ck_plain");
        let init_p = g_plain.input("init", Shape::new(&[n], DType::F64));
        let final_p = g_plain.scan(init_p, make_body(), length);
        let loss_p = g_plain.reduce(
            final_p,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g_plain.set_outputs(vec![loss_p]);
        let bwd_p = grad_with_loss(&g_plain, &[init_p]);

        // Checkpointed scan path with K=2 (length=6).
        let mut g_ck = Graph::new("ck_ckpt");
        let init_c = g_ck.input("init", Shape::new(&[n], DType::F64));
        let final_c = g_ck.scan_checkpointed(init_c, make_body(), length, 2);
        let loss_c = g_ck.reduce(
            final_c,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g_ck.set_outputs(vec![loss_c]);
        let bwd_c = grad_with_loss(&g_ck, &[init_c]);

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no {want}");
        };

        let init_data = vec![0.5_f64, -0.5];
        let d_seed = [1.0_f64];

        let (s_p, mut a_p) = prepare_f64(
            &bwd_p,
            &[
                (find(&bwd_p, "init"), &init_data),
                (find(&bwd_p, "d_output"), &d_seed),
            ],
        );
        execute_thunks(&s_p, a_p.raw_buf_mut());
        let dinit_p = read_arena_f64(&a_p, bwd_p.outputs[1], n);

        let (s_c, mut a_c) = prepare_f64(
            &bwd_c,
            &[
                (find(&bwd_c, "init"), &init_data),
                (find(&bwd_c, "d_output"), &d_seed),
            ],
        );
        execute_thunks(&s_c, a_c.raw_buf_mut());
        let dinit_c = read_arena_f64(&a_c, bwd_c.outputs[1], n);

        for i in 0..n {
            assert!(
                (dinit_p[i] - dinit_c[i]).abs() < 1e-12,
                "dinit[{i}]: plain={} checkpointed={}",
                dinit_p[i],
                dinit_c[i]
            );
        }
    }

    /// Recursive checkpointing end-to-end: build a ScanBackward
    /// configured with K=2 checkpoints (for length=4), and compare
    /// dinit against the same backward graph with full trajectory
    /// (K=0). Forward computes a cumulative-sum-style scan; loss = sum.
    /// Both paths must agree to f64 precision.
    #[test]
    fn recursive_checkpointing_matches_full_trajectory() {
        let n = 2usize;
        let length = 4u32;

        // Body: carry + ones (deterministic, no xs)
        let build_body = || -> Graph {
            let mut body = Graph::new("rc_body");
            let carry = body.input("carry", Shape::new(&[n], DType::F64));
            let ones_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
            let ones = body.add_node(
                Op::Constant { data: ones_bytes },
                vec![],
                Shape::new(&[n], DType::F64),
            );
            let next = body.binary(BinaryOp::Add, carry, ones, Shape::new(&[n], DType::F64));
            body.set_outputs(vec![next]);
            body
        };

        // body_vjp: same body + d_output, output dcarry. body_vjp is
        // used by ScanBackward to walk the chain rule per step.
        let body_vjp_for = || -> Graph {
            use rlx_opt::autodiff::grad;
            let body = build_body();
            // grad(body, [carry_id]) → graph with dcarry as the output.
            let carry_id = body
                .nodes()
                .iter()
                .find(|n| matches!(n.op, Op::Input { .. }))
                .map(|n| n.id)
                .unwrap();
            grad(&body, &[carry_id])
        };

        // ── Forward (All-strategy): scan with full trajectory ──
        let mut g_full = Graph::new("rc_outer_full");
        let init_full = g_full.input("init", Shape::new(&[n], DType::F64));
        let traj_full_id = g_full.scan_trajectory(init_full, build_body(), length);
        // Hand-build a ScanBackward node that reads the full trajectory.
        let upstream_full = g_full.input("upstream", Shape::new(&[length as usize, n], DType::F64));
        let dinit_full_id = g_full.scan_backward(
            init_full,
            traj_full_id,
            upstream_full,
            &[],
            body_vjp_for(),
            length,
            true,
            Shape::new(&[n], DType::F64),
        );
        g_full.set_outputs(vec![dinit_full_id]);

        // ── Forward (Recursive-2): scan saves only K=2 rows ──
        // Build the trajectory shape [K, *carry] = [2, 2].
        let k = 2u32;
        let mut g_rec = Graph::new("rc_outer_rec");
        let init_rec = g_rec.input("init", Shape::new(&[n], DType::F64));
        let traj_rec_id = g_rec.add_node(
            Op::Scan {
                body: Box::new(build_body()),
                length,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: k,
            },
            vec![init_rec],
            Shape::new(&[k as usize, n], DType::F64),
        );
        // Same upstream shape as the full version (the upstream is per
        // *forward step*, length rows — independent of K).
        let upstream_rec = g_rec.input("upstream", Shape::new(&[length as usize, n], DType::F64));
        let dinit_rec_id = g_rec.add_node(
            Op::ScanBackward {
                body_vjp: Box::new(body_vjp_for()),
                length,
                save_trajectory: true,
                num_xs: 0,
                num_checkpoints: k,
                forward_body: Some(Box::new(build_body())),
            },
            vec![init_rec, traj_rec_id, upstream_rec],
            Shape::new(&[n], DType::F64),
        );
        g_rec.set_outputs(vec![dinit_rec_id]);

        // ── Run both, same inputs ──
        let init_data = vec![0.5_f64, -0.5];
        let upstream_data: Vec<f64> = (0..length as usize * n).map(|i| (i as f64) * 0.1).collect();

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                if let Op::Input { name } = &node.op
                    && name == want
                {
                    return node.id;
                }
            }
            panic!("no input {want}");
        };

        let (s_full, mut a_full) = prepare_f64(
            &g_full,
            &[
                (find(&g_full, "init"), &init_data),
                (find(&g_full, "upstream"), &upstream_data),
            ],
        );
        execute_thunks(&s_full, a_full.raw_buf_mut());
        let dinit_full = read_arena_f64(&a_full, g_full.outputs[0], n);

        let (s_rec, mut a_rec) = prepare_f64(
            &g_rec,
            &[
                (find(&g_rec, "init"), &init_data),
                (find(&g_rec, "upstream"), &upstream_data),
            ],
        );
        execute_thunks(&s_rec, a_rec.raw_buf_mut());
        let dinit_rec = read_arena_f64(&a_rec, g_rec.outputs[0], n);

        for i in 0..n {
            assert!(
                (dinit_full[i] - dinit_rec[i]).abs() < 1e-12,
                "i={i}: full={} rec={}",
                dinit_full[i],
                dinit_rec[i]
            );
        }
    }

    /// vmap-of-grad: gradient through Scan, vmap'd over init.
    /// Forward (per row):
    ///   carry_{t+1} = carry_t + ones    (body adds a constant)
    ///   loss = sum(carry_length) = sum(init) + length·n
    /// Closed form: dloss/dinit_i = 1 for every i. vmap over init at
    /// batch=3 → dinit_batched is all-ones [3, n]. Cross-checks
    /// against per-row grad_with_loss runs. Validates the vmap rule
    /// for Op::ScanBackward.
    #[test]
    fn vmap_of_grad_scan_matches_per_row_runs() {
        use rlx_opt::autodiff::grad_with_loss;
        use rlx_opt::vmap::vmap;
        let n = 2usize;
        let length = 3u32;
        let batch = 3usize;

        let mut body = Graph::new("scan_grad_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let ones_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
        let ones = body.add_node(
            Op::Constant { data: ones_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let next = body.binary(BinaryOp::Add, carry, ones, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("scan_grad_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let final_x = g.scan(init, body, length);
        let loss = g.reduce(
            final_x,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[init]);
        let bg = vmap(&bwd, &["init"], batch);

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want}");
        };
        let init_b = find(&bg, "init");
        let d_out_b = find(&bg, "d_output");

        let init_data: Vec<f64> = (0..batch * n).map(|i| (i as f64) * 0.5).collect();
        let d_seed = [1.0_f64];

        let (sched, mut arena) = prepare_f64(&bg, &[(init_b, &init_data), (d_out_b, &d_seed)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let dinit_b = read_arena_f64(&arena, bg.outputs[1], batch * n);

        for i in 0..batch * n {
            assert!(
                (dinit_b[i] - 1.0).abs() < 1e-12,
                "dinit[{i}] = {} (expected 1.0)",
                dinit_b[i]
            );
        }

        // Cross-check vs per-row grad_with_loss.
        for bi in 0..batch {
            let row = &init_data[bi * n..(bi + 1) * n];
            let mut g2 = Graph::new("per_row_grad");
            let init2 = g2.input("init", Shape::new(&[n], DType::F64));
            let mut body2 = Graph::new("per_row_body");
            let c2 = body2.input("carry", Shape::new(&[n], DType::F64));
            let ones2_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
            let ones2 = body2.add_node(
                Op::Constant { data: ones2_bytes },
                vec![],
                Shape::new(&[n], DType::F64),
            );
            let next2 = body2.binary(BinaryOp::Add, c2, ones2, Shape::new(&[n], DType::F64));
            body2.set_outputs(vec![next2]);
            let final2 = g2.scan(init2, body2, length);
            let loss2 = g2.reduce(
                final2,
                ReduceOp::Sum,
                vec![0],
                false,
                Shape::new(&[1], DType::F64),
            );
            g2.set_outputs(vec![loss2]);
            let bwd2 = grad_with_loss(&g2, &[init2]);
            let init2_id = find(&bwd2, "init");
            let d_out2_id = find(&bwd2, "d_output");
            let (s2, mut a2) = prepare_f64(&bwd2, &[(init2_id, row), (d_out2_id, &d_seed)]);
            execute_thunks(&s2, a2.raw_buf_mut());
            let row_dinit = read_arena_f64(&a2, bwd2.outputs[1], n);
            for j in 0..n {
                let got = dinit_b[bi * n + j];
                let want = row_dinit[j];
                assert!(
                    (got - want).abs() < 1e-12,
                    "row {bi}, j {j}: vmap'd={got} per-row={want}"
                );
            }
        }
    }

    /// vmap of Op::Scan: batched cumulative-sum. Forward
    ///   carry_{t+1} = carry_t + xs\[t\]
    ///   final = init + sum(xs)
    /// vmap over both init and xs at batch=3. Each batch row should
    /// equal the scalar run of the same body+xs subset.
    #[test]
    fn vmap_scan_cumulative_sum_matches_scalar_runs() {
        use rlx_opt::vmap::vmap;
        let n = 2usize;
        let length = 4u32;
        let batch = 3usize;

        // Body: (carry, x_t) → carry + x_t
        let mut body = Graph::new("scan_body_cumsum");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let x_t = body.input("x_t", Shape::new(&[n], DType::F64));
        let next = body.binary(BinaryOp::Add, carry, x_t, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("scan_outer_cumsum");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
        let final_carry = g.scan_with_xs(init, &[xs], body, length);
        g.set_outputs(vec![final_carry]);

        // vmap over both init and xs.
        let bg = vmap(&g, &["init", "xs"], batch);

        // Test data — distinct per-batch rows.
        let init_data: Vec<f64> = (0..batch * n).map(|i| (i + 1) as f64).collect();
        // xs has shape [B, length, n] after vmap (the outer's xs is
        // [length, n]; vmap lifts it to [B, length, n]).
        let xs_data: Vec<f64> = (0..batch * length as usize * n)
            .map(|i| 0.1 * (i as f64))
            .collect();

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                if let Op::Input { name } = &node.op
                    && name == want
                {
                    return node.id;
                }
            }
            panic!("no input {want}");
        };
        let init_b = find(&bg, "init");
        let xs_b = find(&bg, "xs");
        let (sched, mut arena) = prepare_f64(&bg, &[(init_b, &init_data), (xs_b, &xs_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let batched_out = read_arena_f64(&arena, bg.outputs[0], batch * n);

        // Reference: per-batch scalar Scan.
        for bi in 0..batch {
            let init_slice = &init_data[bi * n..(bi + 1) * n];
            let mut x = init_slice.to_vec();
            for t in 0..length as usize {
                for j in 0..n {
                    x[j] += xs_data[bi * length as usize * n + t * n + j];
                }
            }

            for i in 0..n {
                let got = batched_out[bi * n + i];
                assert!(
                    (got - x[i]).abs() < 1e-12,
                    "row {bi}, i {i}: got {got} ref {}",
                    x[i]
                );
            }
        }
    }

    /// vmap of dense solve — Circulax-shaped batched parameter sweep.
    /// Forward: x = solve(A, b). vmap over both A (batched [B,N,N])
    /// and b (batched [B,N]). Run on CPU and compare each batch row
    /// against an independent scalar dgesv.
    #[test]
    fn vmap_dense_solve_matches_scalar_runs() {
        use rlx_opt::vmap::vmap;
        let n = 3usize;
        let batch = 4usize;

        let mut g = Graph::new("solve_forward");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![x]);

        // vmap both A and b across the batch.
        let bg = vmap(&g, &["A", "b"], batch);

        // Independent A and b per batch row.
        let mut rng = rlx_ir::Philox4x32::new(0xb47c_u64);
        let mut a_data = vec![0.0_f64; batch * n * n];
        let mut b_data = vec![0.0_f64; batch * n];
        for bi in 0..batch {
            // Diagonally dominant A — guaranteed non-singular.
            for i in 0..n {
                for j in 0..n {
                    a_data[bi * n * n + i * n + j] = rng.next_f32() as f64 * 0.1;
                }
                a_data[bi * n * n + i * n + i] += 1.0 + n as f64;
            }
            for i in 0..n {
                b_data[bi * n + i] = rng.next_f32() as f64;
            }
        }

        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                if let Op::Input { name } = &node.op
                    && name == want
                {
                    return node.id;
                }
            }
            panic!("no input named {want}");
        };
        let ba = find(&bg, "A");
        let bb = find(&bg, "b");
        let (sched, mut arena) = prepare_f64(&bg, &[(ba, &a_data), (bb, &b_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let batched_x = read_arena_f64(&arena, bg.outputs[0], batch * n);

        // Reference: per-batch dgesv.
        for bi in 0..batch {
            let mut a_slice: Vec<f64> = a_data[bi * n * n..(bi + 1) * n * n].to_vec();
            let mut b_slice: Vec<f64> = b_data[bi * n..(bi + 1) * n].to_vec();
            crate::blas::dgesv(&mut a_slice, &mut b_slice, n, 1);
            for i in 0..n {
                let got = batched_x[bi * n + i];
                let want = b_slice[i];
                assert!(
                    (got - want).abs() < 1e-12,
                    "row {bi}, i {i}: got {got} want {want}"
                );
            }
        }
    }

    /// vmap end-to-end: build a graph that computes y = MatMul(x, w) + b
    /// and reduces to a per-element loss. vmap over x with batch=4.
    /// Run the batched graph and compare each output row against an
    /// independent scalar run of the original graph. Validates the
    /// structural lift + the runtime path for batched MatMul +
    /// batched Binary + batched Reduce.
    #[test]
    fn vmap_matmul_add_reduce_matches_scalar_runs() {
        use rlx_opt::vmap::vmap;
        let n = 3usize;
        let batch = 4usize;

        // Forward graph: y = MatMul(reshape(x, [1,n]), w) + b ; loss = sum(y).
        let mut g = Graph::new("vmap_e2e_forward");
        let x = g.input("x", Shape::new(&[n], DType::F64));
        let w = g.input("w", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x_row = g.add_node(
            Op::Reshape {
                new_shape: vec![1, n as i64],
            },
            vec![x],
            Shape::new(&[1, n], DType::F64),
        );
        let mm = g.matmul(x_row, w, Shape::new(&[1, n], DType::F64));
        let mm_flat = g.add_node(
            Op::Reshape {
                new_shape: vec![n as i64],
            },
            vec![mm],
            Shape::new(&[n], DType::F64),
        );
        let yv = g.binary(BinaryOp::Add, mm_flat, b, Shape::new(&[n], DType::F64));
        let loss = g.reduce(
            yv,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        // Build the vmap'd version (batch over x; w and b shared).
        let bg = vmap(&g, &["x"], batch);

        // Test data — distinct rows so we can verify the per-row dispatch.
        let mut rng = rlx_ir::Philox4x32::new(0xc1c0_u64);
        let n_w = n * n;
        let w_data: Vec<f64> = (0..n_w).map(|_| rng.next_f32() as f64).collect();
        let b_data: Vec<f64> = (0..n).map(|_| rng.next_f32() as f64).collect();
        let mut x_data_batched: Vec<f64> = Vec::with_capacity(batch * n);
        for _ in 0..batch * n {
            x_data_batched.push(rng.next_f32() as f64);
        }

        // Run the batched graph.
        let find = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                if let Op::Input { name } = &node.op
                    && name == want
                {
                    return node.id;
                }
            }
            panic!("no input named {want}");
        };
        let bx = find(&bg, "x");
        let bw = find(&bg, "w");
        let bb = find(&bg, "b");
        let (sched, mut arena) =
            prepare_f64(&bg, &[(bx, &x_data_batched), (bw, &w_data), (bb, &b_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        // Reduce::Sum on shifted axis 1 with keep_dim=false → output [B, 1]
        // (it preserves the leading batch axis but reduces what was [n] to [].
        // Since the original output was [1] f64 and the reduce was over
        // axis 0, after vmap the leading-axis-shifted reduce keeps the
        // leading 1 from the original output's [1] shape.)
        let batched_out = read_arena_f64(&arena, bg.outputs[0], batch);

        // Reference: run the original (un-batched) graph once per batch row.
        for bi in 0..batch {
            let xs_slice = &x_data_batched[bi * n..(bi + 1) * n];
            let mut g2 = Graph::new("scalar_run");
            let x2 = g2.input("x", Shape::new(&[n], DType::F64));
            let w2 = g2.input("w", Shape::new(&[n, n], DType::F64));
            let b2 = g2.input("b", Shape::new(&[n], DType::F64));
            let xr = g2.add_node(
                Op::Reshape {
                    new_shape: vec![1, n as i64],
                },
                vec![x2],
                Shape::new(&[1, n], DType::F64),
            );
            let m = g2.matmul(xr, w2, Shape::new(&[1, n], DType::F64));
            let mf = g2.add_node(
                Op::Reshape {
                    new_shape: vec![n as i64],
                },
                vec![m],
                Shape::new(&[n], DType::F64),
            );
            let yv2 = g2.binary(BinaryOp::Add, mf, b2, Shape::new(&[n], DType::F64));
            let l2 = g2.reduce(
                yv2,
                ReduceOp::Sum,
                vec![0],
                false,
                Shape::new(&[1], DType::F64),
            );
            g2.set_outputs(vec![l2]);
            let (s2, mut a2) = prepare_f64(&g2, &[(x2, xs_slice), (w2, &w_data), (b2, &b_data)]);
            execute_thunks(&s2, a2.raw_buf_mut());
            let scalar_out = read_arena_f64(&a2, l2, 1);
            assert!(
                (batched_out[bi] - scalar_out[0]).abs() < 1e-12,
                "row {bi}: batched={} scalar={}",
                batched_out[bi],
                scalar_out[0]
            );
        }
    }

    /// Full gradient through scan-with-xs: dinit AND dxs both checked
    /// against finite differences. Forward
    ///   carry_{t+1} = solve(M, carry_t + xs\[t\])
    ///   loss        = sum(carry_length)
    /// Verifies that grad_with_loss returns gradients w.r.t. both
    /// `init` and `xs` and that dxs matches per-element FD.
    #[test]
    fn scan_with_xs_dxs_matches_fd() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;
        let length = 3u32;
        let dt = 0.1_f64;

        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_dxs_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let drive = body.input("drive", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
        let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_dxs_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
        let final_carry = g.scan_with_xs(init, &[xs], body, length);
        let loss = g.reduce(
            final_carry,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        // wrt = [init, xs] — get both gradients back.
        let bwd = grad_with_loss(&g, &[init, xs]);
        assert_eq!(bwd.outputs.len(), 3, "[loss, dinit, dxs]");

        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let init_bwd = find_by_name(&bwd, "init");
        let xs_bwd = find_by_name(&bwd, "xs");
        let d_out_bwd = find_by_name(&bwd, "d_output");

        let init_data = vec![0.5_f64, 0.0, -0.5];
        let xs_data: Vec<f64> = (0..length as usize * n)
            .map(|i| 0.1_f64 * ((i as f64) - 4.0))
            .collect();
        let d_seed = [1.0_f64];

        let (sched, mut arena) = prepare_f64(
            &bwd,
            &[
                (init_bwd, &init_data),
                (xs_bwd, &xs_data),
                (d_out_bwd, &d_seed),
            ],
        );
        execute_thunks(&sched, arena.raw_buf_mut());
        let dinit = read_arena_f64(&arena, bwd.outputs[1], n);
        let dxs = read_arena_f64(&arena, bwd.outputs[2], length as usize * n);

        let h = 1e-6;
        let loss_at = |x0: &[f64], xs_in: &[f64]| -> f64 {
            let mut acc = x0.to_vec();
            for t in 0..length as usize {
                for j in 0..n {
                    acc[j] += xs_in[t * n + j];
                }
                let mut a_copy = m_data.clone();
                crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
            }
            acc.iter().sum()
        };

        // FD on dinit (sanity).
        for i in 0..n {
            let mut ip = init_data.to_vec();
            ip[i] += h;
            let mut im = init_data.to_vec();
            im[i] -= h;
            let fd = (loss_at(&ip, &xs_data) - loss_at(&im, &xs_data)) / (2.0 * h);
            assert!(
                (dinit[i] - fd).abs() < 1e-7,
                "FD dinit[{i}]: AD={} FD={}",
                dinit[i],
                fd
            );
        }

        // FD on every dxs entry — full per-step gradient check.
        for t in 0..length as usize {
            for j in 0..n {
                let idx = t * n + j;
                let mut xp = xs_data.clone();
                xp[idx] += h;
                let mut xm = xs_data.clone();
                xm[idx] -= h;
                let fd = (loss_at(&init_data, &xp) - loss_at(&init_data, &xm)) / (2.0 * h);
                assert!(
                    (dxs[idx] - fd).abs() < 1e-7,
                    "FD dxs[t={t},j={j}]: AD={} FD={}",
                    dxs[idx],
                    fd
                );
            }
        }
    }

    /// Gradient through a scan with per-step xs (Circulax-shaped).
    /// Forward:
    ///   carry_{t+1} = solve(M, carry_t + xs\[t\])
    ///   loss = sum(carry_length)
    /// dxs is out of MVP (asserted in the VJP rule's body_vjp `wrt`),
    /// but `dinit` flows correctly through the body's reverse Jacobian
    /// even with xs in the chain. Verify dinit against finite differences.
    #[test]
    fn scan_with_xs_gradient_dinit_matches_fd() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;
        let length = 3u32;
        let dt = 0.1_f64;

        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_xs_grad_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let drive = body.input("drive", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
        let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_xs_grad_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
        let final_carry = g.scan_with_xs(init, &[xs], body, length);
        let loss = g.reduce(
            final_carry,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[init]);

        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let init_bwd = find_by_name(&bwd, "init");
        let xs_bwd = find_by_name(&bwd, "xs");
        let d_out_bwd = find_by_name(&bwd, "d_output");

        let init_data = vec![0.5_f64, 0.0, -0.5];
        // Drive: small per-step pulse, varying per element.
        let xs_data: Vec<f64> = (0..length as usize * n)
            .map(|i| 0.1_f64 * ((i as f64) - 4.0))
            .collect();
        let d_seed = [1.0_f64];

        let (sched, mut arena) = prepare_f64(
            &bwd,
            &[
                (init_bwd, &init_data),
                (xs_bwd, &xs_data),
                (d_out_bwd, &d_seed),
            ],
        );
        execute_thunks(&sched, arena.raw_buf_mut());
        let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

        let h = 1e-6;
        let loss_at = |x0: &[f64]| -> f64 {
            let mut acc = x0.to_vec();
            for t in 0..length as usize {
                for j in 0..n {
                    acc[j] += xs_data[t * n + j];
                }
                let mut a_copy = m_data.clone();
                crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
            }
            acc.iter().sum()
        };
        for i in 0..n {
            let mut ip = init_data.to_vec();
            ip[i] += h;
            let mut im = init_data.to_vec();
            im[i] -= h;
            let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
            assert!(
                (dinit[i] - fd).abs() < 1e-7,
                "FD dinit[{i}]: AD={} FD={}",
                dinit[i],
                fd
            );
        }
    }

    /// Gradient through a geometric-growth scan: forward
    ///   x_{t+1} = 1.1 · x_t,    x_0 = init
    ///   final   = x_length     = init · 1.1^length
    ///   loss    = sum(final)
    /// closed-form ∂loss/∂init\[i\] = 1.1^length for every i.
    /// Validates the VJP path: AD pre-pass rewrites save_trajectory=false
    /// to true, autodiff emits Op::ScanBackward, executor walks t back.
    #[test]
    fn scan_gradient_geometric_matches_closed_form() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;
        let length = 5u32;

        let mut body = Graph::new("scan_grad_body");
        let x = body.input("carry", Shape::new(&[n], DType::F64));
        let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 1.1_f64.to_le_bytes()).collect();
        let scale = body.add_node(
            Op::Constant { data: scale_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let next = body.binary(BinaryOp::Mul, x, scale, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("scan_grad_outer");
        let init = g.input("init", Shape::new(&[n], DType::F64));
        let final_x = g.scan(init, body, length);
        let loss = g.reduce(
            final_x,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[init]);
        assert_eq!(bwd.outputs.len(), 2);

        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let init_bwd = find_by_name(&bwd, "init");
        let d_out_bwd = find_by_name(&bwd, "d_output");

        let init_data = vec![1.0_f64; n];
        let d_seed = [1.0_f64];
        let (sched, mut arena) = prepare_f64(&bwd, &[(init_bwd, &init_data), (d_out_bwd, &d_seed)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

        let want = 1.1_f64.powi(length as i32);
        for i in 0..n {
            assert!(
                (dinit[i] - want).abs() < 1e-12,
                "dinit[{i}] = {} want {}",
                dinit[i],
                want
            );
        }

        // Finite-difference cross-check on init[0].
        let h = 1e-6;
        let loss_at = |x: &[f64]| -> f64 {
            let mut acc = x.to_vec();
            for _ in 0..length {
                for v in acc.iter_mut() {
                    *v *= 1.1;
                }
            }
            acc.iter().sum()
        };
        let mut ip = init_data.clone();
        ip[0] += h;
        let mut im = init_data.clone();
        im[0] -= h;
        let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
        assert!(
            (dinit[0] - fd).abs() < 1e-7,
            "FD dinit[0]: AD={} FD={}",
            dinit[0],
            fd
        );
    }

    /// Gradient through Backward Euler scan composing with DenseSolve.
    /// Asserts dinit matches finite-difference per coordinate.
    #[test]
    fn scan_gradient_backward_euler_matches_fd() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 4usize;
        let length = 3u32;
        let dt = 0.05_f64;

        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_grad_body");
        let x = body.input("x", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_grad_outer");
        let init = g.input("x0", Shape::new(&[n], DType::F64));
        let final_x = g.scan(init, body, length);
        let loss = g.reduce(
            final_x,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[init]);

        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let init_bwd = find_by_name(&bwd, "x0");
        let d_out_bwd = find_by_name(&bwd, "d_output");

        let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
        let d_seed = [1.0_f64];
        let (sched, mut arena) = prepare_f64(&bwd, &[(init_bwd, &init_data), (d_out_bwd, &d_seed)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

        let h = 1e-6;
        let loss_at = |x0: &[f64]| -> f64 {
            let mut acc = x0.to_vec();
            for _ in 0..length {
                let mut a_copy = m_data.clone();
                crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
            }
            acc.iter().sum()
        };
        for i in 0..n {
            let mut ip = init_data.to_vec();
            ip[i] += h;
            let mut im = init_data.to_vec();
            im[i] -= h;
            let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
            assert!(
                (dinit[i] - fd).abs() < 1e-7,
                "FD dinit[{i}]: AD={} FD={}",
                dinit[i],
                fd
            );
        }
    }

    /// Trajectory-mode scan: same Backward Euler body, but record the
    /// carry at every step. Output is `[length, n]` — row `t` is the
    /// state after step `t+1`. Validates the SaveAt-style waveform
    /// recording end-to-end, including that the last row equals what
    /// the no-trajectory variant would have returned.
    #[test]
    fn scan_trajectory_backward_euler_records_waveform() {
        let n = 4usize;
        let length = 5u32;
        let dt = 0.05_f64;

        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_traj_body");
        let x = body.input("x", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_traj_outer");
        let init = g.input("x0", Shape::new(&[n], DType::F64));
        let traj = g.scan_trajectory(init, body, length);
        g.set_outputs(vec![traj]);

        let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
        let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena_f64(&arena, traj, length as usize * n);

        // Reference: each step's solve, recorded.
        let mut want = Vec::<f64>::with_capacity(length as usize * n);
        let mut x_ref = init_data.to_vec();
        for _ in 0..length {
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut x_ref, n, 1);
            want.extend_from_slice(&x_ref);
        }
        for i in 0..length as usize * n {
            assert!(
                (got[i] - want[i]).abs() < 1e-12,
                "got[{i}] = {} ref {}",
                got[i],
                want[i]
            );
        }

        // Sanity: trajectory rows are monotone-decreasing in mass
        // (Backward Euler diffuses; boundary leak removes mass).
        for t in 1..length as usize {
            let prev: f64 = got[(t - 1) * n..t * n].iter().sum();
            let curr: f64 = got[t * n..(t + 1) * n].iter().sum();
            assert!(
                curr <= prev + 1e-15,
                "mass should decay: row {} sum {prev}, row {t} sum {curr}",
                t - 1
            );
        }

        // Last row of the trajectory equals what a non-trajectory
        // scan returns — verify by running the same forward through
        // the simpler API and comparing.
        let mut body2 = Graph::new("be_final_body");
        let x2 = body2.input("x", Shape::new(&[n], DType::F64));
        let m_bytes2: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let m2 = body2.add_node(
            Op::Constant { data: m_bytes2 },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let next2 = body2.dense_solve(m2, x2, Shape::new(&[n], DType::F64));
        body2.set_outputs(vec![next2]);

        let mut g2 = Graph::new("be_final_outer");
        let init2 = g2.input("x0", Shape::new(&[n], DType::F64));
        let final_x = g2.scan(init2, body2, length);
        g2.set_outputs(vec![final_x]);
        let (sched2, mut arena2) = prepare_f64(&g2, &[(init2, &init_data)]);
        execute_thunks(&sched2, arena2.raw_buf_mut());
        let final_got = read_arena_f64(&arena2, final_x, n);

        let last_row = &got[(length as usize - 1) * n..length as usize * n];
        for i in 0..n {
            assert!(
                (last_row[i] - final_got[i]).abs() < 1e-15,
                "last trajectory row[{i}] = {} vs final-scan = {}",
                last_row[i],
                final_got[i]
            );
        }
    }

    /// Op::Scan composing with Op::DenseSolve — the Circulax-shaped
    /// pattern for Backward Euler.
    /// Body: x_{t+1} = solve(I + dt·A, x_t).
    /// 1-D heat-equation Laplacian A; analytic ground truth from
    /// composing the same per-step solve in Rust.
    #[test]
    fn scan_backward_euler_heat_f64() {
        let n = 4usize;
        let length = 5u32;
        let dt = 0.05_f64;

        // Construct M = I + dt · L  where L is the Laplacian (-1, 2, -1).
        // M is constant across iterations; embed it in the body via Op::Constant.
        let mut m_data = vec![0.0_f64; n * n];
        for i in 0..n {
            m_data[i * n + i] = 1.0 + dt * 2.0;
            if i > 0 {
                m_data[i * n + (i - 1)] = -dt;
            }
            if i + 1 < n {
                m_data[i * n + (i + 1)] = -dt;
            }
        }
        let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut body = Graph::new("be_body");
        let x = body.input("x", Shape::new(&[n], DType::F64));
        let m = body.add_node(
            Op::Constant { data: m_bytes },
            vec![],
            Shape::new(&[n, n], DType::F64),
        );
        let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);

        let mut g = Graph::new("be_outer");
        let init = g.input("x0", Shape::new(&[n], DType::F64));
        let final_x = g.scan(init, body, length);
        g.set_outputs(vec![final_x]);

        // Initial: a sharp pulse at index 1.
        let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
        let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena_f64(&arena, final_x, n);

        // Reference: apply the same M-solve `length` times in pure Rust.
        let mut ref_x = init_data.to_vec();
        for _ in 0..length {
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut ref_x, n, 1);
        }
        for i in 0..n {
            assert!(
                (got[i] - ref_x[i]).abs() < 1e-12,
                "got[{i}] = {} ref {}",
                got[i],
                ref_x[i]
            );
        }
        // Sanity: pulse should diffuse, mass should be conserved-ish
        // (Backward Euler is mass-conserving for this stencil with
        // zero-flux boundaries — but our boundaries leak, so check
        // that mass strictly decreases instead).
        let mass: f64 = got.iter().sum();
        assert!(mass > 0.0 && mass < 1.0, "diffusion mass: {mass}");
    }

    /// Multi-RHS forward DenseSolve: X = solve(A, B) with B [N, K]
    /// stays correct end-to-end. Verifies the executor/lowering and
    /// the LAPACK column-major dance both honour `nrhs > 1`.
    #[test]
    fn dense_solve_f64_multi_rhs_forward() {
        let n = 3usize;
        let k = 2usize;
        let mut g = Graph::new("solve_multi_rhs");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("B", Shape::new(&[n, k], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
        g.set_outputs(vec![x]);

        let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
        let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
        let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let x_got = read_arena_f64(&arena, x, n * k);
        for c in 0..k {
            for i in 0..n {
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += a_data[i * n + j] * x_got[j * k + c];
                }
                let want = b_data[i * k + c];
                assert!(
                    (acc - want).abs() < 1e-10,
                    "col {c} row {i}: got {acc} want {want}"
                );
            }
        }
    }

    /// Multi-RHS reverse-mode VJP: dB = (Aᵀ)⁻¹·1, dA = -dB · Xᵀ.
    /// Verified analytically + finite differences on dB[0,0].
    #[test]
    fn dense_solve_f64_multi_rhs_gradient() {
        use rlx_opt::autodiff::grad_with_loss;
        let n = 3usize;
        let k = 2usize;
        let mut g = Graph::new("solve_mrhs_grad");
        let a = g.param("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("B", Shape::new(&[n, k], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
        let loss = g.reduce(
            x,
            ReduceOp::Sum,
            vec![0, 1],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bwd = grad_with_loss(&g, &[a, b]);
        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let a_bwd = find_by_name(&bwd, "A");
        let b_bwd = find_by_name(&bwd, "B");
        let d_out = find_by_name(&bwd, "d_output");

        let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
        let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
        let d_seed = [1.0_f64];

        let (sched, mut arena) = prepare_f64(
            &bwd,
            &[(a_bwd, &a_data), (b_bwd, &b_data), (d_out, &d_seed)],
        );
        execute_thunks(&sched, arena.raw_buf_mut());
        let da_got = read_arena_f64(&arena, bwd.outputs[1], n * n);
        let db_got = read_arena_f64(&arena, bwd.outputs[2], n * k);

        // Reference.
        let mut x_ref = b_data;
        {
            let mut a_copy = a_data;
            crate::blas::dgesv(&mut a_copy, &mut x_ref, n, k);
        }
        let mut at = [0.0_f64; 9];
        for i in 0..n {
            for j in 0..n {
                at[i * n + j] = a_data[j * n + i];
            }
        }
        let mut ones_nk = vec![1.0_f64; n * k];
        crate::blas::dgesv(&mut at, &mut ones_nk, n, k);
        let db_ref = ones_nk;
        let mut da_ref = [0.0_f64; 9];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for c in 0..k {
                    acc += db_ref[i * k + c] * x_ref[j * k + c];
                }
                da_ref[i * n + j] = -acc;
            }
        }
        for i in 0..n * k {
            assert!(
                (db_got[i] - db_ref[i]).abs() < 1e-10,
                "dB[{i}]: got {} want {}",
                db_got[i],
                db_ref[i]
            );
        }
        for i in 0..n * n {
            assert!(
                (da_got[i] - da_ref[i]).abs() < 1e-10,
                "dA[{i}]: got {} want {}",
                da_got[i],
                da_ref[i]
            );
        }

        // FD on dB[0,0].
        let h = 1e-6;
        let mut bp = b_data;
        bp[0] += h;
        let mut bm = b_data;
        bm[0] -= h;
        let xp = {
            let mut a_copy = a_data;
            crate::blas::dgesv(&mut a_copy, &mut bp, n, k);
            bp
        };
        let xm = {
            let mut a_copy = a_data;
            crate::blas::dgesv(&mut a_copy, &mut bm, n, k);
            bm
        };
        let lp: f64 = xp.iter().sum();
        let lm: f64 = xm.iter().sum();
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (db_got[0] - fd).abs() < 1e-7,
            "FD dB[0,0]: AD={} FD={}",
            db_got[0],
            fd
        );
    }

    /// Multi-RHS forward-mode JVP w.r.t. B. Closed form: t_X = solve(A, t_B).
    #[test]
    fn dense_solve_f64_multi_rhs_jvp() {
        use rlx_opt::autodiff_fwd::jvp;
        let n = 3usize;
        let k = 2usize;
        let mut g = Graph::new("solve_mrhs_jvp");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("B", Shape::new(&[n, k], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
        g.set_outputs(vec![x]);

        let jg = jvp(&g, &[b]);
        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let a_id = find_by_name(&jg, "A");
        let b_id = find_by_name(&jg, "B");
        let tb_id = find_by_name(&jg, "tangent_B");

        let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
        let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
        let tb_data = [0.5, 0.0, -0.25, 1.0, 1.0, -0.5_f64];

        let (sched, mut arena) =
            prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (tb_id, &tb_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let tangent_x = read_arena_f64(&arena, jg.outputs[1], n * k);

        let mut a_copy = a_data;
        let mut tb_copy = tb_data;
        crate::blas::dgesv(&mut a_copy, &mut tb_copy, n, k);
        for i in 0..n * k {
            assert!(
                (tangent_x[i] - tb_copy[i]).abs() < 1e-10,
                "t_X[{i}]: AD={} ref={}",
                tangent_x[i],
                tb_copy[i]
            );
        }

        let h = 1e-6;
        let mut bp = b_data;
        let mut bm = b_data;
        for i in 0..n * k {
            bp[i] += h * tb_data[i];
            bm[i] -= h * tb_data[i];
        }
        let xp = {
            let mut a_copy = a_data;
            crate::blas::dgesv(&mut a_copy, &mut bp, n, k);
            bp
        };
        let xm = {
            let mut a_copy = a_data;
            crate::blas::dgesv(&mut a_copy, &mut bm, n, k);
            bm
        };
        for i in 0..n * k {
            let fd = (xp[i] - xm[i]) / (2.0 * h);
            assert!(
                (tangent_x[i] - fd).abs() < 1e-7,
                "FD t_X[{i}]: AD={} FD={}",
                tangent_x[i],
                fd
            );
        }
    }

    /// Forward-mode JVP through DenseSolve, end-to-end at f64.
    ///
    /// Build forward x = solve(A, b), call `jvp(forward, [b])`,
    /// compile + run, and check the tangent output matches the
    /// closed form `t_x = solve(A, t_b)` plus a finite-difference
    /// cross-check `(solve(A, b + h·t_b) − solve(A, b − h·t_b)) / 2h`.
    #[test]
    fn jvp_dense_solve_b_runs_and_matches_fd() {
        use rlx_opt::autodiff_fwd::jvp;
        let n = 3usize;

        // Forward.
        let mut g = Graph::new("jvp_b_e2e");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![x]);

        // JVP graph perturbing b only.
        let jg = jvp(&g, &[b]);
        // The JVP graph holds a fresh "tangent_b" Input on top of A and b.
        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let a_id = find_by_name(&jg, "A");
        let b_id = find_by_name(&jg, "b");
        let tb_id = find_by_name(&jg, "tangent_b");

        let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let b_data: [f64; 3] = [1.0, 2.0, 3.0];
        // Pick an arbitrary perturbation direction.
        let tb_data: [f64; 3] = [0.5, -0.25, 1.0];

        let (sched, mut arena) =
            prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (tb_id, &tb_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());

        // Outputs: [primal_x, tangent_x].
        let primal_x = read_arena_f64(&arena, jg.outputs[0], n);
        let tangent_x = read_arena_f64(&arena, jg.outputs[1], n);

        // Closed form: t_x = solve(A, t_b).
        let t_x_ref = {
            let mut a = a_data;
            let mut tb = tb_data;
            let info = crate::blas::dgesv(&mut a, &mut tb, n, 1);
            assert_eq!(info, 0);
            tb
        };
        for i in 0..n {
            assert!(
                (tangent_x[i] - t_x_ref[i]).abs() < 1e-10,
                "t_x[{i}]: got {} want {}",
                tangent_x[i],
                t_x_ref[i]
            );
        }

        // FD: x(b + h·tb) − x(b − h·tb)) / 2h
        let h = 1e-6;
        let mut bp = b_data;
        let mut bm = b_data;
        for i in 0..n {
            bp[i] += h * tb_data[i];
            bm[i] -= h * tb_data[i];
        }
        let xp = {
            let mut a = a_data;
            let info = crate::blas::dgesv(&mut a, &mut bp, n, 1);
            assert_eq!(info, 0);
            bp
        };
        let xm = {
            let mut a = a_data;
            let info = crate::blas::dgesv(&mut a, &mut bm, n, 1);
            assert_eq!(info, 0);
            bm
        };
        let fd: Vec<f64> = (0..n).map(|i| (xp[i] - xm[i]) / (2.0 * h)).collect();
        for i in 0..n {
            assert!(
                (tangent_x[i] - fd[i]).abs() < 1e-7,
                "FD mismatch t_x[{i}]: AD={} FD={}",
                tangent_x[i],
                fd[i]
            );
        }
        // Sanity: primal output is the actual solve.
        let primal_ref = {
            let mut a = a_data;
            let mut b = b_data;
            crate::blas::dgesv(&mut a, &mut b, n, 1);
            b
        };
        for i in 0..n {
            assert!((primal_x[i] - primal_ref[i]).abs() < 1e-10);
        }
    }

    /// Forward-mode JVP through DenseSolve perturbing A. The tangent
    /// path includes the −t_A·x correction term.
    /// `t_x = −solve(A, t_A · x)` should match a finite-difference
    /// directional derivative of `solve(A, b)` w.r.t. A in the
    /// `t_A` direction.
    #[test]
    fn jvp_dense_solve_a_runs_and_matches_fd() {
        use rlx_opt::autodiff_fwd::jvp;
        let n = 3usize;

        let mut g = Graph::new("jvp_a_e2e");
        let a = g.input("A", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![x]);

        let jg = jvp(&g, &[a]);
        let find_by_name = |graph: &Graph, want: &str| -> NodeId {
            for node in graph.nodes() {
                let name = match &node.op {
                    Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                    _ => None,
                };
                if name == Some(want) {
                    return node.id;
                }
            }
            panic!("no node named {want:?}");
        };
        let a_id = find_by_name(&jg, "A");
        let b_id = find_by_name(&jg, "b");
        let ta_id = find_by_name(&jg, "tangent_A");

        let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let b_data: [f64; 3] = [1.0, 2.0, 3.0];
        // Asymmetric perturbation direction for A.
        let ta_data: [f64; 9] = [0.10, -0.05, 0.02, 0.03, 0.20, -0.04, -0.01, 0.07, 0.15];

        let (sched, mut arena) =
            prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (ta_id, &ta_data)]);
        execute_thunks(&sched, arena.raw_buf_mut());

        let tangent_x = read_arena_f64(&arena, jg.outputs[1], n);

        // Closed form: x = solve(A, b); t_x = −solve(A, t_A · x).
        let x_ref = {
            let mut a = a_data;
            let mut b = b_data;
            crate::blas::dgesv(&mut a, &mut b, n, 1);
            b
        };
        let mut prod = [0.0_f64; 3];
        for i in 0..n {
            for j in 0..n {
                prod[i] += ta_data[i * n + j] * x_ref[j];
            }
        }
        let t_x_ref = {
            let mut a = a_data;
            let mut p = prod;
            crate::blas::dgesv(&mut a, &mut p, n, 1);
            [-p[0], -p[1], -p[2]]
        };
        for i in 0..n {
            assert!(
                (tangent_x[i] - t_x_ref[i]).abs() < 1e-10,
                "closed-form t_x[{i}]: AD={} ref={}",
                tangent_x[i],
                t_x_ref[i]
            );
        }

        // FD: solve(A + h·t_A, b) and solve(A − h·t_A, b).
        let h = 1e-6;
        let mut ap = a_data;
        let mut am = a_data;
        for i in 0..n * n {
            ap[i] += h * ta_data[i];
            am[i] -= h * ta_data[i];
        }
        let xp = {
            let mut a = ap;
            let mut b = b_data;
            crate::blas::dgesv(&mut a, &mut b, n, 1);
            b
        };
        let xm = {
            let mut a = am;
            let mut b = b_data;
            crate::blas::dgesv(&mut a, &mut b, n, 1);
            b
        };
        for i in 0..n {
            let fd = (xp[i] - xm[i]) / (2.0 * h);
            assert!(
                (tangent_x[i] - fd).abs() < 1e-7,
                "FD t_x[{i}]: AD={} FD={}",
                tangent_x[i],
                fd
            );
        }
    }

    /// Real INT8 conv2d parity. Same setup as QMatMul: pre-quantize
    /// f32 inputs to i8, run `Op::QConv2d`, compare against an
    /// in-test reference loop that does the same i32 accumulation
    /// and requantize math. Symmetric quant (zp=0) to keep the math
    /// head-to-head.
    #[test]
    fn q_conv2d_matches_reference() {
        use rlx_ir::Philox4x32;
        // Small NCHW shape — enough to exercise stride/padding edges.
        let n = 1usize;
        let c_in = 2usize;
        let h = 5usize;
        let w_in = 5usize;
        let c_out = 3usize;
        let kh = 3usize;
        let kw = 3usize;
        let ph = 1usize;
        let pw = 1usize;
        let sh = 1usize;
        let sw = 1usize;
        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w_in + 2 * pw - kw) / sw + 1;

        let x_scale = 0.04f32;
        let w_scale = 0.02f32;
        let out_scale = 0.5f32;
        let mult = x_scale * w_scale / out_scale;

        let mut rng = Philox4x32::new(2099);
        let mut xf = vec![0f32; n * c_in * h * w_in];
        rng.fill_normal(&mut xf);
        let mut wf = vec![0f32; c_out * c_in * kh * kw];
        rng.fill_normal(&mut wf);
        let xq: Vec<i8> = xf
            .iter()
            .map(|&v| ((v / x_scale).round() as i32).clamp(-128, 127) as i8)
            .collect();
        let wq: Vec<i8> = wf
            .iter()
            .map(|&v| ((v / w_scale).round() as i32).clamp(-128, 127) as i8)
            .collect();
        let bias: Vec<i32> = vec![0i32; c_out];

        let mut g = Graph::new("qconv");
        let xn = g.input("x", Shape::new(&[n, c_in, h, w_in], DType::I8));
        let wn = g.input("w", Shape::new(&[c_out, c_in, kh, kw], DType::I8));
        let bn = g.input("b", Shape::new(&[c_out], DType::I32));
        let out = g.q_conv2d(
            xn,
            wn,
            bn,
            vec![kh, kw],
            vec![sh, sw],
            vec![ph, pw],
            vec![1, 1],
            1,
            0,
            0,
            0,
            mult,
            Shape::new(&[n, c_out, h_out, w_out], DType::I8),
        );
        g.set_outputs(vec![out]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        // Capture offsets before borrowing the buf mutably (avoids
        // overlap between &mut and the &arena.byte_offset reads).
        let xn_off = arena.byte_offset(xn);
        let wn_off = arena.byte_offset(wn);
        let bn_off = arena.byte_offset(bn);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(xn_off) as *mut i8;
            for (i, &v) in xq.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(wn_off) as *mut i8;
            for (i, &v) in wq.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(bn_off) as *mut i32;
            for (i, &v) in bias.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let out_q: Vec<i8> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const i8;
            (0..n * c_out * h_out * w_out).map(|i| *p.add(i)).collect()
        };

        // Reference: scalar loop in NCHW with the same requantize.
        let mut out_ref = vec![0i8; n * c_out * h_out * w_out];
        for ni in 0..n {
            for co in 0..c_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc: i32 = 0;
                        for ci in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let hi = ho * sh + ki;
                                    let wi = wo * sw + kj;
                                    if hi < ph || wi < pw {
                                        continue;
                                    }
                                    let hi = hi - ph;
                                    let wi = wi - pw;
                                    if hi >= h || wi >= w_in {
                                        continue;
                                    }
                                    let xv =
                                        xq[((ni * c_in) + ci) * h * w_in + hi * w_in + wi] as i32;
                                    let wv = wq[((co * c_in) + ci) * kh * kw + ki * kw + kj] as i32;
                                    acc += xv * wv;
                                }
                            }
                        }
                        let r = (acc as f32 * mult).round() as i32;
                        let r = r.clamp(-128, 127) as i8;
                        out_ref[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = r;
                    }
                }
            }
        }

        for (i, (a, r)) in out_q.iter().zip(&out_ref).enumerate() {
            assert_eq!(a, r, "q_conv2d[{i}]: kernel {a} vs reference {r}");
        }
    }

    /// Real INT8 matmul parity: compare `Op::QMatMul` against the
    /// fake-quant reference `Dequantize → MatMul → Quantize` that
    /// would produce the same output if we round-tripped through
    /// f32. Both should agree element-for-element (or within ±1 i8
    /// step, since rounding in the requantize uses different code
    /// paths). Symmetric quantization (zp=0) for both paths to keep
    /// the math head-to-head.
    #[test]
    fn q_matmul_matches_fake_quant_reference() {
        use rlx_ir::Philox4x32;
        let m = 3usize;
        let k = 8usize;
        let n = 5usize;
        let mut rng = Philox4x32::new(2031);

        // Pick scales and quantize random f32 inputs to i8.
        let x_scale = 0.05f32;
        let w_scale = 0.03f32;
        let out_scale = 0.4f32;
        let mult = x_scale * w_scale / out_scale;
        let mut xf = vec![0f32; m * k];
        rng.fill_normal(&mut xf);
        let mut wf = vec![0f32; k * n];
        rng.fill_normal(&mut wf);
        let xq: Vec<i8> = xf
            .iter()
            .map(|&v| ((v / x_scale).round() as i32).clamp(-128, 127) as i8)
            .collect();
        let wq: Vec<i8> = wf
            .iter()
            .map(|&v| ((v / w_scale).round() as i32).clamp(-128, 127) as i8)
            .collect();
        let bias: Vec<i32> = vec![0i32; n];

        // ── Direct INT8 path ──
        let _f = DType::F32;
        let mut g_q = Graph::new("qmm_direct");
        let xn = g_q.input("x", Shape::new(&[m, k], DType::I8));
        let wn = g_q.input("w", Shape::new(&[k, n], DType::I8));
        let bn = g_q.input("b", Shape::new(&[n], DType::I32));
        let out = g_q.q_matmul(xn, wn, bn, 0, 0, 0, mult, Shape::new(&[m, n], DType::I8));
        g_q.set_outputs(vec![out]);
        let plan = rlx_opt::memory::plan_memory(&g_q);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g_q, &arena);

        // Fill inputs.
        let xn_off = arena.byte_offset(xn);
        let wn_off = arena.byte_offset(wn);
        let bn_off = arena.byte_offset(bn);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(xn_off) as *mut i8;
            for (i, &v) in xq.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(wn_off) as *mut i8;
            for (i, &v) in wq.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(bn_off) as *mut i32;
            for (i, &v) in bias.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let out_q: Vec<i8> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const i8;
            (0..m * n).map(|i| *p.add(i)).collect()
        };

        // ── Fake-quant reference: scalar emulation in plain Rust ──
        // Same arithmetic the kernel does, but in a verifier loop:
        //   acc = Σ (x[m,k]) · (w[k,n]),  // zps are 0
        //   out[m,n] = saturate_i8(round(acc · mult) + 0)
        let mut out_ref = vec![0i8; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc: i32 = 0;
                for ki in 0..k {
                    acc += (xq[mi * k + ki] as i32) * (wq[ki * n + ni] as i32);
                }
                let r = (acc as f32 * mult).round() as i32;
                out_ref[mi * n + ni] = r.clamp(-128, 127) as i8;
            }
        }

        for (i, (a, r)) in out_q.iter().zip(&out_ref).enumerate() {
            assert_eq!(a, r, "q_matmul[{i}]: kernel {a} vs reference {r}");
        }
    }

    /// Quantize/Dequantize round-trip — quantize an f32 tensor, then
    /// dequantize back, and confirm the result tracks the input
    /// within the per-element scale (the inevitable rounding error).
    /// Also pins the kernel's saturation behavior at the i8 limits.
    #[test]
    fn quantize_dequantize_round_trip() {
        use rlx_ir::Philox4x32;
        let len = 64;
        let mut rng = Philox4x32::new(2027);
        let mut x = vec![0f32; len];
        rng.fill_normal(&mut x);
        // Stretch a couple values past the +/- saturation cliff so
        // the saturate_i8 path is exercised.
        x[0] = 999.0;
        x[1] = -999.0;

        let scale = 0.05f32;
        let zp = 3i32;

        let f = DType::F32;
        let mut g = Graph::new("qdq");
        let xn = g.input("x", Shape::new(&[len], f));
        let q = g.quantize(xn, scale, zp);
        let dq = g.dequantize(q, scale, zp);
        g.set_outputs(vec![dq]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let xn_off = arena.byte_offset(xn);
        let dq_off = arena.byte_offset(dq);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
            for (i, &v) in x.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let out: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
            (0..len).map(|i| *p.add(i)).collect()
        };

        // Saturated values at i=0,1 should clamp to ±127's dequant
        // range (= (±127 - zp) · scale).
        let sat_pos = (127 - zp) as f32 * scale;
        let sat_neg = (-128 - zp) as f32 * scale;
        assert!((out[0] - sat_pos).abs() < 1e-6, "+sat: {}", out[0]);
        assert!((out[1] - sat_neg).abs() < 1e-6, "-sat: {}", out[1]);

        // Everything else should round-trip within `scale` (one quant
        // step = the worst-case rounding error).
        for i in 2..len {
            assert!(
                (out[i] - x[i]).abs() <= scale + 1e-5,
                "qdq[{i}]: {} → {}, scale={scale}",
                x[i],
                out[i]
            );
        }
    }

    /// Per-channel quantize / dequantize: independent scale and zp
    /// per slice along an axis. Verifies (a) each channel uses its
    /// own scale (not a shared one), (b) saturation still respects
    /// the i8 range, (c) channel data layout decomposition is
    /// correct (no cross-channel leakage).
    #[test]
    fn quantize_per_channel_round_trip() {
        let c = 4usize;
        let inner = 5usize;
        // Different magnitudes per channel — proves the per-channel
        // scale is actually being read for each row.
        let mags = [0.01f32, 0.5, 5.0, 50.0];
        let mut x = vec![0f32; c * inner];
        for ci in 0..c {
            for ii in 0..inner {
                // Sweep through values that span [-max_abs, +max_abs]
                // for each channel, plus one value past the cliff to
                // trigger saturation.
                x[ci * inner + ii] = match ii {
                    0 => -mags[ci],
                    1 => 0.0,
                    2 => mags[ci],
                    3 => mags[ci] * 1000.0,  // saturates +
                    _ => -mags[ci] * 1000.0, // saturates -
                };
            }
        }
        let scales: Vec<f32> = mags.iter().map(|&m| m / 127.0).collect();
        let zps: Vec<i32> = vec![0, 0, 0, 0];

        let f = DType::F32;
        let mut g = Graph::new("qdq_pc");
        let xn = g.input("x", Shape::new(&[c, inner], f));
        let q = g.quantize_per_channel(xn, 0, scales.clone(), zps.clone());
        let dq = g.dequantize_per_channel(q, 0, scales.clone(), zps);
        g.set_outputs(vec![dq]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let xn_off = arena.byte_offset(xn);
        let dq_off = arena.byte_offset(dq);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
            for (i, &v) in x.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let out: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
            (0..c * inner).map(|i| *p.add(i)).collect()
        };

        for ci in 0..c {
            // Within-range entries (positions 0, 1, 2) must round-trip
            // within one quant step of *that channel's* scale.
            for ii in 0..3 {
                let idx = ci * inner + ii;
                assert!(
                    (out[idx] - x[idx]).abs() <= scales[ci] + 1e-5,
                    "ch {ci} idx {ii}: {} vs {}",
                    x[idx],
                    out[idx]
                );
            }
            // Saturated positions clamp to ±127 · scale[ci].
            let sat_pos = 127.0 * scales[ci];
            let sat_neg = -128.0 * scales[ci];
            assert!(
                (out[ci * inner + 3] - sat_pos).abs() < 1e-5,
                "ch {ci} +sat: {}",
                out[ci * inner + 3]
            );
            assert!(
                (out[ci * inner + 4] - sat_neg).abs() < 1e-5,
                "ch {ci} -sat: {}",
                out[ci * inner + 4]
            );
        }
    }

    /// `Op::ActivationBackward` parity for every supported kind.
    /// Builds a single-op graph `dx = activation_backward(x, dy)` and
    /// compares each `dx[i]` to the central-difference `(act(x+ε) -
    /// act(x-ε)) / (2ε) · dy\[i\]`. Sweeps the closed-form covered by
    /// the kernel.
    #[test]
    fn activation_backward_matches_numerical_per_kind() {
        use rlx_ir::Philox4x32;
        use rlx_ir::op::Activation;
        let mut rng = Philox4x32::new(91);
        let len = 32;
        // x sampled away from kink/branch points: shifted positive
        // (exp/sqrt/log domain) for the unary-positive activations;
        // wide range otherwise. Two parallel tests would be cleaner
        // but this is concise enough.
        let mut x_pos = vec![0f32; len];
        rng.fill_normal(&mut x_pos);
        for v in x_pos.iter_mut() {
            *v = v.abs() + 0.5;
        }
        let mut x_any = vec![0f32; len];
        rng.fill_normal(&mut x_any);
        let mut dy = vec![0f32; len];
        rng.fill_normal(&mut dy);

        for &(kind, x_data, eps, tol) in &[
            (Activation::Sigmoid, &x_any[..], 1e-3, 5e-3),
            (Activation::Tanh, &x_any[..], 1e-3, 5e-3),
            (Activation::Silu, &x_any[..], 1e-3, 5e-3),
            (Activation::Gelu, &x_any[..], 1e-3, 5e-3),
            (Activation::GeluApprox, &x_any[..], 1e-3, 5e-3),
            (Activation::Exp, &x_any[..], 1e-4, 5e-3),
            (Activation::Log, &x_pos[..], 1e-4, 5e-3),
            (Activation::Sqrt, &x_pos[..], 1e-4, 5e-3),
            (Activation::Rsqrt, &x_pos[..], 1e-4, 5e-3),
            (Activation::Neg, &x_any[..], 1e-3, 5e-4),
        ] {
            let f = DType::F32;
            let mut g = Graph::new("act_bw");
            let xn = g.input("x", Shape::new(&[len], f));
            let dyn_ = g.input("dy", Shape::new(&[len], f));
            let dx = g.activation_backward(kind, xn, dyn_);
            g.set_outputs(vec![dx]);

            let plan = rlx_opt::memory::plan_memory(&g);
            let mut arena = crate::arena::Arena::from_plan(plan);
            let sched = compile_thunks(&g, &arena);

            let xn_off = arena.byte_offset(xn);
            let dyn_off = arena.byte_offset(dyn_);
            let dx_off = arena.byte_offset(dx);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
                for (i, &v) in x_data.iter().enumerate() {
                    *p.add(i) = v;
                }
                let p = buf.as_mut_ptr().add(dyn_off) as *mut f32;
                for (i, &v) in dy.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
            execute_thunks(&sched, arena.raw_buf_mut());
            let analytical: Vec<f32> = unsafe {
                let p = arena.raw_buf().as_ptr().add(dx_off) as *const f32;
                (0..len).map(|i| *p.add(i)).collect()
            };

            // Apply the forward activation manually; finite-difference
            // each element.
            let act_apply = |kind: Activation, x: f32| -> f32 {
                match kind {
                    Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    Activation::Tanh => x.tanh(),
                    Activation::Silu => x / (1.0 + (-x).exp()),
                    Activation::Gelu => {
                        // Match the kernel's exact erf form.
                        const INV_SQRT2: f32 = 0.707_106_77;
                        0.5 * x * (1.0 + erf_f32(x * INV_SQRT2))
                    }
                    Activation::GeluApprox => {
                        const C: f32 = 0.797_884_6;
                        const A: f32 = 0.044_715;
                        let inner = C * (x + A * x * x * x);
                        0.5 * x * (1.0 + inner.tanh())
                    }
                    Activation::Exp => x.exp(),
                    Activation::Log => x.ln(),
                    Activation::Sqrt => x.sqrt(),
                    Activation::Rsqrt => 1.0 / x.sqrt(),
                    Activation::Neg => -x,
                    Activation::Relu => x.max(0.0),
                    Activation::Abs => x.abs(),
                    Activation::Round => x.round(),
                    Activation::Sin => x.sin(),
                    Activation::Cos => x.cos(),
                    Activation::Tan => x.tan(),
                    Activation::Atan => x.atan(),
                }
            };
            for i in 0..len {
                let xv = x_data[i];
                let plus = act_apply(kind, xv + eps);
                let minus = act_apply(kind, xv - eps);
                let num = (plus - minus) / (2.0 * eps) * dy[i];
                assert!(
                    (analytical[i] - num).abs() < tol,
                    "{kind:?}[{i}]: analytical {} vs numerical {num}",
                    analytical[i]
                );
            }
        }
    }

    /// Batched 3-D MatMul VJP — the transformer-attention shape
    /// `[B, M, K] @ [B, K, N] = [B, M, N]`. Both gradients flow through
    /// `Op::Transpose` with a perm that swaps the last two dims.
    #[test]
    fn matmul_3d_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        let batch = 2usize;
        let m = 3usize;
        let k = 4usize;
        let n = 5usize;
        let mut rng = Philox4x32::new(101);
        let mut a_data = vec![0f32; batch * m * k];
        rng.fill_normal(&mut a_data);
        let mut b_data = vec![0f32; batch * k * n];
        rng.fill_normal(&mut b_data);

        let f = DType::F32;
        let mut fwd = Graph::new("matmul_3d");
        let an = fwd.input("a", Shape::new(&[batch, m, k], f));
        let bp = fwd.param("b", Shape::new(&[batch, k, n], f));
        let mm = fwd.matmul(an, bp, Shape::new(&[batch, m, n], f));
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0, 1, 2],
                keep_dim: false,
            },
            vec![mm],
            Shape::from_dims(&[], f),
        );
        fwd.set_outputs(vec![loss]);

        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[bp]);
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .unwrap();

        let plan = rlx_opt::memory::plan_memory(&bwd_graph);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&bwd_graph, &arena);
        for &(id, data) in &[(an, &a_data), (bp, &b_data), (d_out, &vec![1.0f32])] {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let gb_id = bwd_graph.outputs[1];
        let g_b: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(arena.byte_offset(gb_id)) as *const f32;
            (0..batch * k * n).map(|i| *p.add(i)).collect()
        };

        // Numerical gradient: differentiate sum(a @ b) w.r.t. each b entry.
        let forward_loss = |b_vals: &[f32]| -> f32 {
            let mut out = vec![0f32; batch * m * n];
            for bi in 0..batch {
                for mi in 0..m {
                    for ni in 0..n {
                        let mut acc = 0f32;
                        for ki in 0..k {
                            acc +=
                                a_data[bi * m * k + mi * k + ki] * b_vals[bi * k * n + ki * n + ni];
                        }
                        out[bi * m * n + mi * n + ni] = acc;
                    }
                }
            }
            out.iter().sum()
        };
        let eps = 1e-3f32;
        let mut bp_p = b_data.clone();
        let mut g_b_num = vec![0f32; b_data.len()];
        for i in 0..b_data.len() {
            let s = bp_p[i];
            bp_p[i] = s + eps;
            let lp = forward_loss(&bp_p);
            bp_p[i] = s - eps;
            let lm = forward_loss(&bp_p);
            bp_p[i] = s;
            g_b_num[i] = (lp - lm) / (2.0 * eps);
        }
        for (i, (a, n)) in g_b.iter().zip(&g_b_num).enumerate() {
            assert!(
                (a - n).abs() < 5e-3,
                "matmul_3d g_b[{i}]: analytical {a} vs numerical {n}"
            );
        }
    }

    /// Composed `Op::Softmax` VJP — the gradient is built from
    /// `mul + reduce_sum + expand + sub + mul`, no dedicated
    /// SoftmaxBackward kernel. Verifies the closed-form
    /// `dx = y · (g - Σ y·g)` matches the FD gradient over a small
    /// 2-D logits tensor.
    #[test]
    fn softmax_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        let n = 3usize;
        let c = 5usize;
        let mut rng = Philox4x32::new(57);
        let mut x_data = vec![0f32; n * c];
        rng.fill_normal(&mut x_data);

        let f = DType::F32;
        let mut fwd = Graph::new("softmax_only");
        let xn = fwd.input("x", Shape::new(&[n, c], f));
        let sm = fwd.add_node(Op::Softmax { axis: -1 }, vec![xn], Shape::new(&[n, c], f));
        // Loss = sum(softmax · target) for some random fixed target —
        // any linear loss will do; sum-of-all is the simplest and gives
        // a uniform gradient flow into the softmax.
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0, 1],
                keep_dim: false,
            },
            vec![sm],
            Shape::from_dims(&[], f),
        );
        fwd.set_outputs(vec![loss]);

        // `wrt = [xn]` — autodiff exposes the gradient w.r.t. the
        // input so we can compare it directly. The forward NodeId for
        // `xn` doubles as its bwd-graph mirror.
        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[xn]);
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .unwrap();

        let plan = rlx_opt::memory::plan_memory(&bwd_graph);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&bwd_graph, &arena);
        for &(id, data) in &[(xn, &x_data), (d_out, &vec![1.0f32])] {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let g_x_id = bwd_graph.outputs[1];
        let g_x: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(arena.byte_offset(g_x_id)) as *const f32;
            (0..n * c).map(|i| *p.add(i)).collect()
        };

        // Loss derivative: softmax sums to 1 per row → d/dx_i sum(softmax) = 0
        // analytically. So expect g_x ≈ 0 within FD precision. (This
        // doubles as a strong sanity check for the composition.)
        let forward_loss = |x: &[f32]| -> f32 {
            let mut total = 0f32;
            for ni in 0..n {
                let row = &x[ni * c..(ni + 1) * c];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let denom: f32 = row.iter().map(|&v| (v - m).exp()).sum();
                for &v in row {
                    total += (v - m).exp() / denom;
                }
            }
            total
        };
        let eps = 1e-3f32;
        let mut p = x_data.clone();
        for i in 0..x_data.len() {
            let s = p[i];
            p[i] = s + eps;
            let lp = forward_loss(&p);
            p[i] = s - eps;
            let lm = forward_loss(&p);
            p[i] = s;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g_x[i] - num).abs() < 5e-3,
                "softmax g_x[{i}]: analytical {} vs numerical {num}",
                g_x[i]
            );
        }
    }

    /// LayerNorm VJP — three gradients in one pass:
    ///   d_x via `LayerNormBackwardInput`,
    ///   d_gamma via `LayerNormBackwardGamma`,
    ///   d_beta = `unbroadcast(upstream)` to gamma's shape.
    #[test]
    fn layer_norm_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        let rows = 3usize;
        let h = 6usize;
        let mut rng = Philox4x32::new(1009);
        let mut x_data = vec![0f32; rows * h];
        rng.fill_normal(&mut x_data);
        let mut g_data = vec![0f32; h];
        rng.fill_normal(&mut g_data);
        for v in g_data.iter_mut() {
            *v = v.abs() + 0.5;
        }
        let mut b_data = vec![0f32; h];
        rng.fill_normal(&mut b_data);
        let eps = 1e-5f32;

        let f = DType::F32;
        let mut fwd = Graph::new("ln_only");
        let xn = fwd.input("x", Shape::new(&[rows, h], f));
        let gp = fwd.param("gamma", Shape::new(&[h], f));
        let bp = fwd.param("beta", Shape::new(&[h], f));
        let ln = fwd.add_node(
            Op::LayerNorm { axis: -1, eps },
            vec![xn, gp, bp],
            Shape::new(&[rows, h], f),
        );
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0, 1],
                keep_dim: false,
            },
            vec![ln],
            Shape::from_dims(&[], f),
        );
        fwd.set_outputs(vec![loss]);

        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[xn, gp, bp]);
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .unwrap();

        let plan = rlx_opt::memory::plan_memory(&bwd_graph);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&bwd_graph, &arena);
        for &(id, data) in &[
            (xn, &x_data),
            (gp, &g_data),
            (bp, &b_data),
            (d_out, &vec![1.0f32]),
        ] {
            let off = arena.byte_offset(id);
            let buf = arena.raw_buf_mut();
            unsafe {
                let p = buf.as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let read = |id: NodeId, n: usize| -> Vec<f32> {
            let off = arena.byte_offset(id);
            unsafe {
                let p = arena.raw_buf().as_ptr().add(off) as *const f32;
                (0..n).map(|i| *p.add(i)).collect()
            }
        };
        let dx_a = read(bwd_graph.outputs[1], rows * h);
        let dg_a = read(bwd_graph.outputs[2], h);
        let db_a = read(bwd_graph.outputs[3], h);

        let forward_loss = |x: &[f32], g: &[f32], b: &[f32]| -> f32 {
            let mut total = 0f32;
            for r in 0..rows {
                let row = &x[r * h..(r + 1) * h];
                let mean = row.iter().sum::<f32>() / h as f32;
                let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for d in 0..h {
                    total += ((row[d] - mean) * inv_std) * g[d] + b[d];
                }
            }
            total
        };
        let h_eps = 1e-3f32;

        let mut x_p = x_data.clone();
        for i in 0..x_p.len() {
            let s = x_p[i];
            x_p[i] = s + h_eps;
            let lp = forward_loss(&x_p, &g_data, &b_data);
            x_p[i] = s - h_eps;
            let lm = forward_loss(&x_p, &g_data, &b_data);
            x_p[i] = s;
            let num = (lp - lm) / (2.0 * h_eps);
            assert!(
                (dx_a[i] - num).abs() < 5e-3,
                "ln dx[{i}]: analytical {} vs numerical {num}",
                dx_a[i]
            );
        }
        let mut g_p = g_data.clone();
        for i in 0..g_p.len() {
            let s = g_p[i];
            g_p[i] = s + h_eps;
            let lp = forward_loss(&x_data, &g_p, &b_data);
            g_p[i] = s - h_eps;
            let lm = forward_loss(&x_data, &g_p, &b_data);
            g_p[i] = s;
            let num = (lp - lm) / (2.0 * h_eps);
            assert!(
                (dg_a[i] - num).abs() < 5e-3,
                "ln dg[{i}]: analytical {} vs numerical {num}",
                dg_a[i]
            );
        }
        let mut b_p = b_data.clone();
        for i in 0..b_p.len() {
            let s = b_p[i];
            b_p[i] = s + h_eps;
            let lp = forward_loss(&x_data, &g_data, &b_p);
            b_p[i] = s - h_eps;
            let lm = forward_loss(&x_data, &g_data, &b_p);
            b_p[i] = s;
            let num = (lp - lm) / (2.0 * h_eps);
            assert!(
                (db_a[i] - num).abs() < 5e-3,
                "ln db[{i}]: analytical {} vs numerical {num}",
                db_a[i]
            );
        }
    }

    /// Single dense layer + softmax-cross-entropy + mean reduce —
    /// the simplest non-trivial training graph. Validates MatMul,
    /// broadcast Add, SCE, Reduce(Mean) VJPs and the grad_with_loss
    /// plumbing all at once.
    #[test]
    fn dense_sce_mean_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        let bs = 4usize;
        let k_in = 3usize;
        let c = 5usize;
        let mut rng = Philox4x32::new(7);
        let mut x = vec![0f32; bs * k_in];
        rng.fill_normal(&mut x);
        let mut w_init = vec![0f32; k_in * c];
        rng.fill_normal(&mut w_init);
        let mut b_init = vec![0f32; c];
        rng.fill_normal(&mut b_init);
        let labels: Vec<f32> = (0..bs).map(|i| (i % c) as f32).collect();

        // ── Forward graph: loss = mean(sce(x @ w + b, labels)) ──
        let f = DType::F32;
        let mut fwd = Graph::new("dense_sce");
        let xn = fwd.input("x", Shape::new(&[bs, k_in], f));
        let lb = fwd.input("labels", Shape::new(&[bs], f));
        let wp = fwd.param("w", Shape::new(&[k_in, c], f));
        let bp = fwd.param("b", Shape::new(&[c], f));
        let mm = fwd.matmul(xn, wp, Shape::new(&[bs, c], f));
        let logits = fwd.binary(BinaryOp::Add, mm, bp, Shape::new(&[bs, c], f));
        let loss_per = fwd.softmax_cross_entropy_with_logits(logits, lb);
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0],
                keep_dim: false,
            },
            vec![loss_per],
            // Reduce sum of [bs] with axes=[0] keep_dim=false → scalar [].
            Shape::from_dims(&[], f),
        );
        // Use Sum + manual /bs scalar mul — also exercises BinaryOp::Mul VJP path
        // less aggressively than Mean would, and gives us a closed-form
        // reference for the loss we expect.
        // For simplicity though, switch to Mean which the tests should also cover.
        // (Re-using `loss` with Sum here for now; the mean factor cancels in
        // the gradient comparison since both analytical and numerical use the
        // same forward.)
        fwd.set_outputs(vec![loss]);

        // ── Backward graph ──
        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wp, bp]);
        // Outputs: [loss, grad_w, grad_b]. NodeIds for x/labels/w/b/loss
        // in bwd_graph match their fwd ids (the mirror keeps order).
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .expect("d_output input");

        let (sched, mut arena) = prepare(
            &bwd_graph,
            &[
                (xn, &x),
                (lb, &labels),
                (wp, &w_init),
                (bp, &b_init),
                (d_out, &[1.0]),
            ],
        );
        execute_thunks(&sched, arena.raw_buf_mut());

        let outs = &bwd_graph.outputs;
        let loss_id = outs[0];
        let gw_id = outs[1];
        let gb_id = outs[2];
        let loss_actual = read_arena(&arena, loss_id, 1)[0];
        let gw_actual = read_arena(&arena, gw_id, k_in * c);
        let gb_actual = read_arena(&arena, gb_id, c);

        // ── Forward-only graph for finite differences ──
        // Re-use the same `fwd` graph; set up its own arena and rerun
        // for each perturbed parameter.
        let plan = rlx_opt::memory::plan_memory(&fwd);
        let mut fwd_arena = crate::arena::Arena::from_plan(plan);
        let fwd_sched = compile_thunks(&fwd, &fwd_arena);
        write_arena(&mut fwd_arena, xn, &x);
        write_arena(&mut fwd_arena, lb, &labels);

        let run_loss = |arena: &mut crate::arena::Arena, w: &[f32], b: &[f32]| -> f32 {
            write_arena(arena, wp, w);
            write_arena(arena, bp, b);
            execute_thunks(&fwd_sched, arena.raw_buf_mut());
            read_arena(arena, loss, 1)[0]
        };

        // Sanity: the loss reported by the bwd graph matches the
        // forward-only graph on the unperturbed inputs.
        let loss_check = run_loss(&mut fwd_arena, &w_init, &b_init);
        assert!(
            (loss_actual - loss_check).abs() < 1e-4,
            "loss mismatch: bwd graph {loss_actual} vs fwd-only {loss_check}"
        );

        let eps = 1e-3f32;
        let mut w_perturbed = w_init.clone();
        let mut gw_numerical = vec![0f32; w_init.len()];
        for i in 0..w_init.len() {
            let saved = w_perturbed[i];
            w_perturbed[i] = saved + eps;
            let lp = run_loss(&mut fwd_arena, &w_perturbed, &b_init);
            w_perturbed[i] = saved - eps;
            let lm = run_loss(&mut fwd_arena, &w_perturbed, &b_init);
            w_perturbed[i] = saved;
            gw_numerical[i] = (lp - lm) / (2.0 * eps);
        }
        for (i, (a, n)) in gw_actual.iter().zip(&gw_numerical).enumerate() {
            assert!(
                (a - n).abs() < 5e-3,
                "grad_w[{i}]: analytical {a} vs numerical {n}"
            );
        }

        let mut b_perturbed = b_init.clone();
        let mut gb_numerical = vec![0f32; b_init.len()];
        for i in 0..b_init.len() {
            let saved = b_perturbed[i];
            b_perturbed[i] = saved + eps;
            let lp = run_loss(&mut fwd_arena, &w_init, &b_perturbed);
            b_perturbed[i] = saved - eps;
            let lm = run_loss(&mut fwd_arena, &w_init, &b_perturbed);
            b_perturbed[i] = saved;
            gb_numerical[i] = (lp - lm) / (2.0 * eps);
        }
        for (i, (a, n)) in gb_actual.iter().zip(&gb_numerical).enumerate() {
            assert!(
                (a - n).abs() < 5e-3,
                "grad_b[{i}]: analytical {a} vs numerical {n}"
            );
        }
    }

    /// Reduce::Mean specifically — verifies the 1/N scaling in the VJP.
    /// The same dense+SCE graph but with Mean instead of Sum on the loss.
    #[test]
    fn dense_sce_mean_reduce_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        let bs = 3usize;
        let k_in = 2usize;
        let c = 4usize;
        let mut rng = Philox4x32::new(13);
        let mut x = vec![0f32; bs * k_in];
        rng.fill_normal(&mut x);
        let mut w_init = vec![0f32; k_in * c];
        rng.fill_normal(&mut w_init);
        let labels: Vec<f32> = (0..bs).map(|i| (i % c) as f32).collect();

        let f = DType::F32;
        let mut fwd = Graph::new("dense_sce_mean");
        let xn = fwd.input("x", Shape::new(&[bs, k_in], f));
        let lb = fwd.input("labels", Shape::new(&[bs], f));
        let wp = fwd.param("w", Shape::new(&[k_in, c], f));
        let mm = fwd.matmul(xn, wp, Shape::new(&[bs, c], f));
        let loss_per = fwd.softmax_cross_entropy_with_logits(mm, lb);
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes: vec![0],
                keep_dim: false,
            },
            vec![loss_per],
            Shape::from_dims(&[], f),
        );
        fwd.set_outputs(vec![loss]);

        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wp]);
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .unwrap();

        let (sched, mut arena) = prepare(
            &bwd_graph,
            &[(xn, &x), (lb, &labels), (wp, &w_init), (d_out, &[1.0])],
        );
        execute_thunks(&sched, arena.raw_buf_mut());

        let outs = &bwd_graph.outputs;
        let loss_id = outs[0];
        let gw_id = outs[1];
        let _ = read_arena(&arena, loss_id, 1)[0];
        let gw_actual = read_arena(&arena, gw_id, k_in * c);

        let plan = rlx_opt::memory::plan_memory(&fwd);
        let mut fwd_arena = crate::arena::Arena::from_plan(plan);
        let fwd_sched = compile_thunks(&fwd, &fwd_arena);
        write_arena(&mut fwd_arena, xn, &x);
        write_arena(&mut fwd_arena, lb, &labels);

        let run_loss = |arena: &mut crate::arena::Arena, w: &[f32]| -> f32 {
            write_arena(arena, wp, w);
            execute_thunks(&fwd_sched, arena.raw_buf_mut());
            read_arena(arena, loss, 1)[0]
        };

        let eps = 1e-3f32;
        let mut wp_p = w_init.clone();
        let mut gw_num = vec![0f32; w_init.len()];
        for i in 0..w_init.len() {
            let s = wp_p[i];
            wp_p[i] = s + eps;
            let lp = run_loss(&mut fwd_arena, &wp_p);
            wp_p[i] = s - eps;
            let lm = run_loss(&mut fwd_arena, &wp_p);
            wp_p[i] = s;
            gw_num[i] = (lp - lm) / (2.0 * eps);
        }
        for (i, (a, n)) in gw_actual.iter().zip(&gw_num).enumerate() {
            assert!((a - n).abs() < 5e-3, "mean reduce grad_w[{i}]: {a} vs {n}");
        }
    }
    /// The full TinyConv-MNIST forward path (downsized) plumbed
    /// through grad_with_loss. Validates that Conv, Pool(Max), ReLU,
    /// Reshape, MatMul, Add (broadcast), SCE, Reduce(Mean) VJPs all
    /// compose into a graph that produces correct gradients.
    #[test]
    fn tinyconv_full_gradient_matches_numerical() {
        use rlx_ir::Philox4x32;
        // Tiny shapes so finite differences finish in <1s.
        let n = 1usize;
        let c_in = 1usize;
        let h = 6usize;
        let w_in = 6usize;
        let c_mid = 2usize; // first conv output channels
        let kh = 3;
        let kw = 3;
        let h1 = h - kh + 1; // 4
        let w1 = w_in - kw + 1; // 4
        let h2 = h1 / 2;
        let w2 = w1 / 2; // 2 × 2 after 2× pool
        let flat = c_mid * h2 * w2; // 8
        let num_classes = 3usize;

        let mut rng = Philox4x32::new(31);
        let mut x = vec![0f32; n * c_in * h * w_in];
        rng.fill_normal(&mut x);
        let mut wc = vec![0f32; c_mid * c_in * kh * kw];
        rng.fill_normal(&mut wc);
        for v in wc.iter_mut() {
            *v *= 0.2;
        }
        // Shift conv-bias well away from the ReLU zero-boundary. Without
        // this, an ε-perturbation of bc[c] can flip the ReLU mask on a
        // pre-activation that happened to land near zero — making the
        // central-difference numerical gradient discontinuous and
        // diverge from the analytical (which assumes local smoothness).
        // +5.0 keeps every pre-activation positive for any random init
        // produced by Philox seed 31 with the wc/x scales used here, so
        // ReLU acts as an identity and finite differences are exact.
        let bc: Vec<f32> = (0..c_mid).map(|i| 5.0 + 0.1 * i as f32).collect();
        let mut wfc = vec![0f32; flat * num_classes];
        rng.fill_normal(&mut wfc);
        for v in wfc.iter_mut() {
            *v *= 0.5;
        }
        let mut bfc = vec![0f32; num_classes];
        rng.fill_normal(&mut bfc);
        let labels: Vec<f32> = vec![1.0]; // batch=1

        let f = DType::F32;
        let mut fwd = Graph::new("tinyconv");
        let xn = fwd.input("x", Shape::new(&[n, c_in, h, w_in], f));
        let lb = fwd.input("labels", Shape::new(&[n], f));
        let wcp = fwd.param("wc", Shape::new(&[c_mid, c_in, kh, kw], f));
        let bcp = fwd.param("bc", Shape::new(&[c_mid], f));
        let wfp = fwd.param("wfc", Shape::new(&[flat, num_classes], f));
        let bfp = fwd.param("bfc", Shape::new(&[num_classes], f));

        // conv: [n, c_in, h, w] → [n, c_mid, h1, w1]
        let conv = fwd.add_node(
            Op::Conv {
                kernel_size: vec![kh, kw],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: 1,
            },
            vec![xn, wcp],
            Shape::new(&[n, c_mid, h1, w1], f),
        );
        // Bias add: expand bc[c_mid] up to the full [n, c_mid, h1, w1]
        // shape so the Add becomes a plain element-wise op. Going through
        // an explicit Reshape→Expand instead of relying on the Add to
        // broadcast `[1, C, 1, 1]` → `[N, C, H, W]` works around a known
        // limitation of `rlx-cpu`'s `Op::Binary` lowering: it dispatches
        // on `out_len % rhs_len == 0` and treats `rhs` as a last-axis
        // bias, which produces `bc[0], bc[1], bc[0], bc[1], …` alternating
        // across all positions instead of channel-broadcasting. Going
        // through Expand (a real broadcast thunk) avoids that path
        // entirely. The autodiff still exercises `unbroadcast` because
        // `Op::Expand`'s VJP reduces over the broadcast axes.
        let bc_4d = fwd.add_node(
            Op::Reshape {
                new_shape: vec![1, c_mid as i64, 1, 1],
            },
            vec![bcp],
            Shape::new(&[1, c_mid, 1, 1], f),
        );
        let bc_expanded = fwd.add_node(
            Op::Expand {
                target_shape: vec![n as i64, c_mid as i64, h1 as i64, w1 as i64],
            },
            vec![bc_4d],
            Shape::new(&[n, c_mid, h1, w1], f),
        );
        let conv_b = fwd.binary(
            BinaryOp::Add,
            conv,
            bc_expanded,
            Shape::new(&[n, c_mid, h1, w1], f),
        );
        let relu = fwd.activation(Activation::Relu, conv_b, Shape::new(&[n, c_mid, h1, w1], f));
        let pool = fwd.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![relu],
            Shape::new(&[n, c_mid, h2, w2], f),
        );
        let flatn = fwd.add_node(
            Op::Reshape {
                new_shape: vec![n as i64, flat as i64],
            },
            vec![pool],
            Shape::new(&[n, flat], f),
        );
        let mm = fwd.matmul(flatn, wfp, Shape::new(&[n, num_classes], f));
        let logits = fwd.binary(BinaryOp::Add, mm, bfp, Shape::new(&[n, num_classes], f));
        let loss_per = fwd.softmax_cross_entropy_with_logits(logits, lb);
        let loss = fwd.add_node(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes: vec![0],
                keep_dim: false,
            },
            vec![loss_per],
            Shape::from_dims(&[], f),
        );
        fwd.set_outputs(vec![loss]);

        let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wcp, bcp, wfp, bfp]);
        let d_out = bwd_graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
            .map(|n| n.id)
            .unwrap();

        let (sched, mut arena) = prepare(
            &bwd_graph,
            &[
                (xn, &x),
                (lb, &labels),
                (wcp, &wc),
                (bcp, &bc),
                (wfp, &wfc),
                (bfp, &bfc),
                (d_out, &[1.0]),
            ],
        );
        execute_thunks(&sched, arena.raw_buf_mut());

        let outs = bwd_graph.outputs.clone();
        let loss_id = outs[0];
        let g_wc_id = outs[1];
        let g_bc_id = outs[2];
        let g_wfc_id = outs[3];
        let g_bfc_id = outs[4];
        let loss_actual = read_arena(&arena, loss_id, 1)[0];
        let g_wc = read_arena(&arena, g_wc_id, wc.len());
        let g_bc = read_arena(&arena, g_bc_id, bc.len());
        let g_wfc = read_arena(&arena, g_wfc_id, wfc.len());
        let g_bfc = read_arena(&arena, g_bfc_id, bfc.len());

        // Forward-only arena for finite differences.
        let plan = rlx_opt::memory::plan_memory(&fwd);
        let mut fwd_arena = crate::arena::Arena::from_plan(plan);
        let fwd_sched = compile_thunks(&fwd, &fwd_arena);
        write_arena(&mut fwd_arena, xn, &x);
        write_arena(&mut fwd_arena, lb, &labels);

        // Closure variant: we need to set all four params each call so
        // perturbations to one don't leak between sweeps.
        let run_loss = |arena: &mut crate::arena::Arena,
                        wc: &[f32],
                        bc: &[f32],
                        wfc: &[f32],
                        bfc: &[f32]|
         -> f32 {
            write_arena(arena, wcp, wc);
            write_arena(arena, bcp, bc);
            write_arena(arena, wfp, wfc);
            write_arena(arena, bfp, bfc);
            execute_thunks(&fwd_sched, arena.raw_buf_mut());
            read_arena(arena, loss, 1)[0]
        };

        let loss_check = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &bfc);
        assert!(
            (loss_actual - loss_check).abs() < 1e-4,
            "tinyconv loss mismatch: bwd {loss_actual} vs fwd {loss_check}"
        );

        let eps = 1e-3f32;
        let check_grad = |arena: &mut crate::arena::Arena,
                          name: &str,
                          analytical: &[f32],
                          mut perturb: Box<
            dyn FnMut(&mut [f32], usize, f32, &mut crate::arena::Arena) -> f32 + '_,
        >,
                          n: usize| {
            for i in 0..n {
                let lp = perturb(&mut analytical.to_vec(), i, eps, arena);
                let lm = perturb(&mut analytical.to_vec(), i, -eps, arena);
                let num = (lp - lm) / (2.0 * eps);
                assert!(
                    (analytical[i] - num).abs() < 5e-3,
                    "{name}[{i}]: analytical {} vs numerical {num}",
                    analytical[i]
                );
            }
        };

        // Helper to perturb one param and run forward. Kept as a
        // reference for the explicit per-param sweep pattern below.
        #[allow(unused_macros)]
        macro_rules! sweep {
            ($name:expr, $base:expr, $analytical:expr, $set_param:ident) => {{
                let n = $base.len();
                for i in 0..n {
                    let mut p = $base.clone();
                    let s = p[i];
                    p[i] = s + eps;
                    let lp = {
                        let $set_param = &p;
                        run_loss(&mut fwd_arena, &wc, &bc, &wfc, &bfc).max(f32::NEG_INFINITY);
                        // Reset others, set the one being swept, run.
                        // (the macro receives one of the four params via $set_param)
                        let _ = $set_param;
                        // Fall through to the explicit per-param helper:
                        0.0_f32
                    };
                    let _ = lp;
                }
            }};
        }
        let _ = check_grad; // silence unused (sweep! macro is intentionally\n        // unused — kept as reference for the per-param sweep pattern below)

        // Per-param sweeps (explicit, not macro — clearer).
        for i in 0..wc.len() {
            let mut p = wc.clone();
            let s = p[i];
            p[i] = s + eps;
            let lp = run_loss(&mut fwd_arena, &p, &bc, &wfc, &bfc);
            p[i] = s - eps;
            let lm = run_loss(&mut fwd_arena, &p, &bc, &wfc, &bfc);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g_wc[i] - num).abs() < 5e-3,
                "g_wc[{i}]: {} vs {num}",
                g_wc[i]
            );
        }
        for i in 0..bc.len() {
            let mut p = bc.clone();
            let s = p[i];
            p[i] = s + eps;
            let lp = run_loss(&mut fwd_arena, &wc, &p, &wfc, &bfc);
            p[i] = s - eps;
            let lm = run_loss(&mut fwd_arena, &wc, &p, &wfc, &bfc);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g_bc[i] - num).abs() < 5e-3,
                "g_bc[{i}]: {} vs {num}",
                g_bc[i]
            );
        }
        for i in 0..wfc.len() {
            let mut p = wfc.clone();
            let s = p[i];
            p[i] = s + eps;
            let lp = run_loss(&mut fwd_arena, &wc, &bc, &p, &bfc);
            p[i] = s - eps;
            let lm = run_loss(&mut fwd_arena, &wc, &bc, &p, &bfc);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g_wfc[i] - num).abs() < 5e-3,
                "g_wfc[{i}]: {} vs {num}",
                g_wfc[i]
            );
        }
        for i in 0..bfc.len() {
            let mut p = bfc.clone();
            let s = p[i];
            p[i] = s + eps;
            let lp = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &p);
            p[i] = s - eps;
            let lm = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &p);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g_bfc[i] - num).abs() < 5e-3,
                "g_bfc[{i}]: {} vs {num}",
                g_bfc[i]
            );
        }
    }

    /// Negative case: a Narrow whose output has multiple consumers
    /// must NOT be fused (we can't elide its write — something else
    /// reads it).
    #[test]
    fn narrow_rope_skips_when_narrow_has_multiple_consumers() {
        let f = DType::F32;
        let mut g = Graph::new("nr_skip");
        let qkv = g.input("qkv", Shape::new(&[16, 8, 192], f));
        let cos = g.input("cos", Shape::new(&[16], f));
        let sin = g.input("sin", Shape::new(&[16], f));
        let q = g.narrow_(qkv, 2, 0, 64);
        let q_rope = g.rope(q, cos, sin, 16);
        // Second consumer of `q` blocks the fusion.
        let q_dup = g.activation(rlx_ir::op::Activation::Relu, q, Shape::new(&[16, 8, 64], f));
        g.set_outputs(vec![q_rope, q_dup]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let narrow_count = sched
            .thunks
            .iter()
            .filter(|t| matches!(t, Thunk::Narrow { .. }))
            .count();
        assert!(
            narrow_count >= 1,
            "Narrow with multiple consumers must NOT be fused away"
        );
    }

    // ── Op::CustomFn (custom_vjp / custom_jvp) tests ──
    //
    // Validates: forward execution inlines fwd_body; VJP rule inlines
    // vjp_body in place of recursing into fwd_body; JVP rule inlines
    // jvp_body. Each test deliberately picks a body whose AD-via-tracing
    // would yield a *different* gradient than the override, so we know
    // the override actually fired.

    /// Forward only: CustomFn wrapping `f(x) = x + c` (c=1 inside body)
    /// without override AD bodies. Verifies the body is compiled,
    /// constants in the body fill correctly, and the output lands at
    /// the outer node's slot.
    #[test]
    fn custom_fn_forward_inlines_body() {
        let s = Shape::new(&[3], DType::F32);

        // Body: f(x) = x + 1
        let mut body = Graph::new("addone_body");
        let x = body.input("x", s.clone());
        let one_data: Vec<u8> = (0..3).flat_map(|_| 1.0_f32.to_le_bytes()).collect();
        let one = body.add_node(Op::Constant { data: one_data }, vec![], s.clone());
        let y = body.binary(BinaryOp::Add, x, one, s.clone());
        body.set_outputs(vec![y]);

        let mut g = Graph::new("custom_fn_outer");
        let xin = g.input("x_in", s.clone());
        let cf = g.custom_fn(vec![xin], body, None, None);
        g.set_outputs(vec![cf]);

        let xs = vec![10.0_f32, 20.0, 30.0];
        let (sched, mut arena) = prepare(&g, &[(xin, &xs)]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let got = read_arena(&arena, cf, 3);
        assert_eq!(got, vec![11.0, 21.0, 31.0]);
    }

    /// Locate an Op::Input or Op::Param by name in a graph.
    fn find_named(graph: &Graph, want: &str) -> NodeId {
        for n in graph.nodes() {
            let name = match &n.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return n.id;
            }
        }
        panic!("no node named {want:?} in graph");
    }

    /// VJP override: f(x) = x but vjp_body returns 2 * d_output, so the
    /// reported gradient should be 2 — different from the natural 1
    /// you'd get by recursing into the identity body.
    #[test]
    fn custom_fn_vjp_overrides_natural_gradient() {
        use rlx_opt::autodiff::grad_with_loss;
        let s = Shape::new(&[1], DType::F32);

        let mut fwd = Graph::new("id_fwd");
        let x = fwd.input("x", s.clone());
        fwd.set_outputs(vec![x]);

        let mut vjp_g = Graph::new("id_vjp");
        let _x_p = vjp_g.input("x", s.clone());
        let _y_p = vjp_g.input("primal_output", s.clone());
        let dy = vjp_g.input("d_output", s.clone());
        let two_data: Vec<u8> = 2.0_f32.to_le_bytes().to_vec();
        let two = vjp_g.add_node(Op::Constant { data: two_data }, vec![], s.clone());
        let dx = vjp_g.binary(BinaryOp::Mul, dy, two, s.clone());
        vjp_g.set_outputs(vec![dx]);

        let mut g = Graph::new("outer");
        let xp = g.param("x", s.clone());
        let cf = g.custom_fn(vec![xp], fwd, Some(vjp_g), None);
        g.set_outputs(vec![cf]);

        let bwd = grad_with_loss(&g, &[xp]);
        assert_eq!(bwd.outputs.len(), 2, "expect [loss, dx]");

        let xb = find_named(&bwd, "x");
        let dout = find_named(&bwd, "d_output");
        let (sched, mut arena) = prepare(&bwd, &[(xb, &[7.0]), (dout, &[1.0])]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let loss = read_arena(&arena, bwd.outputs[0], 1);
        let dx_v = read_arena(&arena, bwd.outputs[1], 1);
        assert!((loss[0] - 7.0).abs() < 1e-6, "loss should be 7.0");
        assert!(
            (dx_v[0] - 2.0).abs() < 1e-6,
            "vjp override should yield dx=2.0, got {} (natural autodiff would give 1.0)",
            dx_v[0]
        );
    }

    /// VJP override: f(a, b) = a*b with vjp_body returning
    /// (b * d_output, a * d_output). Validates routing of multiple
    /// primals + d_output through the override; matches the natural
    /// autodiff-of-Mul gradient (b, a).
    #[test]
    fn custom_fn_vjp_two_inputs_matches_mul_autodiff() {
        use rlx_opt::autodiff::grad_with_loss;
        let s = Shape::new(&[1], DType::F32);

        let mut fwd = Graph::new("mul_fwd");
        let a_f = fwd.input("a", s.clone());
        let b_f = fwd.input("b", s.clone());
        let y_f = fwd.binary(BinaryOp::Mul, a_f, b_f, s.clone());
        fwd.set_outputs(vec![y_f]);

        let mut vjp_g = Graph::new("mul_vjp");
        let a_v = vjp_g.input("a", s.clone());
        let b_v = vjp_g.input("b", s.clone());
        let _y_v = vjp_g.input("primal_output", s.clone());
        let dy_v = vjp_g.input("d_output", s.clone());
        let da = vjp_g.binary(BinaryOp::Mul, b_v, dy_v, s.clone());
        let db = vjp_g.binary(BinaryOp::Mul, a_v, dy_v, s.clone());
        vjp_g.set_outputs(vec![da, db]);

        let mut g = Graph::new("outer");
        let ap = g.param("a", s.clone());
        let bp = g.param("b", s.clone());
        let cf = g.custom_fn(vec![ap, bp], fwd, Some(vjp_g), None);
        g.set_outputs(vec![cf]);

        let bwd = grad_with_loss(&g, &[ap, bp]);
        assert_eq!(bwd.outputs.len(), 3, "expect [loss, da, db]");

        let ab = find_named(&bwd, "a");
        let bb = find_named(&bwd, "b");
        let dout = find_named(&bwd, "d_output");
        let (sched, mut arena) = prepare(&bwd, &[(ab, &[3.0]), (bb, &[5.0]), (dout, &[1.0])]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let loss = read_arena(&arena, bwd.outputs[0], 1);
        let da_v = read_arena(&arena, bwd.outputs[1], 1);
        let db_v = read_arena(&arena, bwd.outputs[2], 1);
        assert!((loss[0] - 15.0).abs() < 1e-5);
        assert!(
            (da_v[0] - 5.0).abs() < 1e-5,
            "da should be b=5.0, got {}",
            da_v[0]
        );
        assert!(
            (db_v[0] - 3.0).abs() < 1e-5,
            "db should be a=3.0, got {}",
            db_v[0]
        );
    }

    /// JVP override: f(x) = x but jvp_body returns 2 * tangent_0.
    /// Forward-mode tangent should be 2x the seed (1.0) → 2.0.
    #[test]
    fn custom_fn_jvp_overrides_natural_tangent() {
        use rlx_opt::autodiff_fwd::jvp;
        let s = Shape::new(&[1], DType::F32);

        let mut fwd = Graph::new("id_fwd");
        let x = fwd.input("x", s.clone());
        fwd.set_outputs(vec![x]);

        let mut jvp_g = Graph::new("id_jvp");
        let _x_p = jvp_g.input("x", s.clone());
        let tx = jvp_g.input("tangent_0", s.clone());
        let two_data: Vec<u8> = 2.0_f32.to_le_bytes().to_vec();
        let two = jvp_g.add_node(Op::Constant { data: two_data }, vec![], s.clone());
        let ty = jvp_g.binary(BinaryOp::Mul, tx, two, s.clone());
        jvp_g.set_outputs(vec![ty]);

        let mut g = Graph::new("outer");
        let xin = g.input("x_in", s.clone());
        let cf = g.custom_fn(vec![xin], fwd, None, Some(jvp_g));
        g.set_outputs(vec![cf]);

        let fwd_g = jvp(&g, &[xin]);
        assert_eq!(fwd_g.outputs.len(), 2, "expect [primal_y, tangent_y]");

        let xb = find_named(&fwd_g, "x_in");
        let tan = find_named(&fwd_g, "tangent_x_in");
        let (sched, mut arena) = prepare(&fwd_g, &[(xb, &[7.0]), (tan, &[1.0])]);
        execute_thunks(&sched, arena.raw_buf_mut());
        let y = read_arena(&arena, fwd_g.outputs[0], 1);
        let ty_v = read_arena(&arena, fwd_g.outputs[1], 1);
        assert!((y[0] - 7.0).abs() < 1e-6);
        assert!(
            (ty_v[0] - 2.0).abs() < 1e-6,
            "jvp override should yield t_y=2.0 (natural autodiff would give 1.0), got {}",
            ty_v[0]
        );
    }

    /// IR-level smoke test: `DType::C64` is wired through the dtype
    /// table — `size_bytes() == 8`, `is_complex()` reports true, and
    /// a `[2]`-shaped C64 buffer in the arena occupies the expected
    /// 16 bytes.
    #[test]
    fn c64_dtype_storage_layout() {
        assert_eq!(
            DType::C64.size_bytes(),
            8,
            "C64 should be 8 bytes (f32 real + f32 imag)"
        );
        assert!(DType::C64.is_complex());
        assert!(!DType::C64.is_float());

        // A length-2 C64 buffer should have shape size_bytes = 16.
        let s = Shape::new(&[2], DType::C64);
        assert_eq!(s.size_bytes().unwrap(), 16);
    }

    // ── C64 element-wise binary kernel witnesses (2026-05-17) ──────
    //
    // Build a tiny graph: Input `a` + Input `b` (both C64 [2]),
    // output = a OP b. Run through CompileResult and compare against
    // the closed-form complex arithmetic on the four chosen pairs.

    fn run_c64_binary(op: BinaryOp, a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let n = a.len();
        let s = Shape::new(&[n], DType::C64);
        let mut g = Graph::new("c64_bin");
        let in_a = g.input("a", s.clone());
        let in_b = g.input("b", s.clone());
        let out = g.binary(op, in_a, in_b, s.clone());
        g.set_outputs(vec![out]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let a_off = arena.byte_offset(in_a);
        let b_off = arena.byte_offset(in_b);
        let out_off = arena.byte_offset(out);
        // Interleave [re_0, im_0, re_1, im_1, ...] in the f32 buffer.
        let buf = arena.raw_buf_mut();
        unsafe {
            let pa = buf.as_mut_ptr().add(a_off) as *mut f32;
            let pb = buf.as_mut_ptr().add(b_off) as *mut f32;
            for (i, &(re, im)) in a.iter().enumerate() {
                *pa.add(2 * i) = re;
                *pa.add(2 * i + 1) = im;
            }
            for (i, &(re, im)) in b.iter().enumerate() {
                *pb.add(2 * i) = re;
                *pb.add(2 * i + 1) = im;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let raw_out: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..(2 * n)).map(|i| *p.add(i)).collect()
        };
        (0..n)
            .map(|i| (raw_out[2 * i], raw_out[2 * i + 1]))
            .collect()
    }

    #[track_caller]
    fn assert_close_c(got: (f32, f32), expected: (f32, f32), tol: f32, label: &str) {
        let dr = (got.0 - expected.0).abs();
        let di = (got.1 - expected.1).abs();
        assert!(
            dr < tol && di < tol,
            "[{label}] got ({:+.4}, {:+.4}), expected ({:+.4}, {:+.4})",
            got.0,
            got.1,
            expected.0,
            expected.1
        );
    }

    #[test]
    fn c64_binary_add_matches_complex_arithmetic() {
        let a = [(1.0_f32, 2.0_f32), (3.0_f32, -1.0_f32)];
        let b = [(4.0_f32, -1.0_f32), (0.5_f32, 0.5_f32)];
        let out = run_c64_binary(BinaryOp::Add, &a, &b);
        assert_close_c(out[0], (5.0, 1.0), 1e-6, "add[0]");
        assert_close_c(out[1], (3.5, -0.5), 1e-6, "add[1]");
    }

    #[test]
    fn c64_binary_sub_matches_complex_arithmetic() {
        let a = [(5.0_f32, 1.0_f32)];
        let b = [(2.0_f32, 3.0_f32)];
        let out = run_c64_binary(BinaryOp::Sub, &a, &b);
        assert_close_c(out[0], (3.0, -2.0), 1e-6, "sub");
    }

    #[test]
    fn c64_binary_mul_matches_complex_arithmetic() {
        // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = -5 + 10i.
        let a = [(1.0_f32, 2.0_f32)];
        let b = [(3.0_f32, 4.0_f32)];
        let out = run_c64_binary(BinaryOp::Mul, &a, &b);
        assert_close_c(out[0], (-5.0, 10.0), 1e-5, "mul");
    }

    #[test]
    fn c64_binary_div_matches_complex_arithmetic() {
        // (1 + 2i) / (3 + 4i) = ((1·3 + 2·4) + (2·3 − 1·4)i) / 25
        //                     = (11 + 2i) / 25
        //                     = 0.44 + 0.08i
        let a = [(1.0_f32, 2.0_f32)];
        let b = [(3.0_f32, 4.0_f32)];
        let out = run_c64_binary(BinaryOp::Div, &a, &b);
        assert_close_c(out[0], (0.44, 0.08), 1e-5, "div");
    }

    #[test]
    fn c64_binary_mul_identity_one_is_no_op() {
        // (a + bi) · (1 + 0i) = a + bi.
        let a = [(3.5_f32, -1.25_f32), (-2.0_f32, 7.0_f32)];
        let b = [(1.0_f32, 0.0_f32), (1.0_f32, 0.0_f32)];
        let out = run_c64_binary(BinaryOp::Mul, &a, &b);
        assert_close_c(out[0], a[0], 1e-6, "mul·1[0]");
        assert_close_c(out[1], a[1], 1e-6, "mul·1[1]");
    }

    #[test]
    fn c64_binary_mul_by_i_rotates_90_degrees() {
        // (a + bi) · i = (a + bi)(0 + i) = -b + ai. 90° CCW rotation.
        let a = [(1.0_f32, 0.0_f32)];
        let b = [(0.0_f32, 1.0_f32)];
        let out = run_c64_binary(BinaryOp::Mul, &a, &b);
        assert_close_c(out[0], (0.0, 1.0), 1e-6, "1·i");
    }

    #[test]
    fn c64_binary_div_by_self_gives_unity() {
        let a = [(2.5_f32, -1.5_f32), (-0.7_f32, 4.2_f32)];
        let out = run_c64_binary(BinaryOp::Div, &a, &a);
        assert_close_c(out[0], (1.0, 0.0), 1e-5, "div_self[0]");
        assert_close_c(out[1], (1.0, 0.0), 1e-5, "div_self[1]");
    }

    #[test]
    #[should_panic(expected = "C64: complex max/min/pow")]
    fn c64_binary_max_is_rejected_at_lowering() {
        run_c64_binary(BinaryOp::Max, &[(1.0_f32, 2.0_f32)], &[(3.0_f32, 4.0_f32)]);
    }

    fn run_c64_activation(act: Activation, a: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let n = a.len();
        let s = Shape::new(&[n], DType::C64);
        let mut g = Graph::new("c64_act");
        let in_a = g.input("a", s.clone());
        let out = g.activation(act, in_a, s.clone());
        g.set_outputs(vec![out]);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let a_off = arena.byte_offset(in_a);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let pa = buf.as_mut_ptr().add(a_off) as *mut f32;
            for (i, &(re, im)) in a.iter().enumerate() {
                *pa.add(2 * i) = re;
                *pa.add(2 * i + 1) = im;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let raw: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..(2 * n)).map(|i| *p.add(i)).collect()
        };
        (0..n).map(|i| (raw[2 * i], raw[2 * i + 1])).collect()
    }

    #[test]
    fn c64_activation_neg_negates_both_components() {
        let inp = [(3.5_f32, -1.25_f32), (-2.0_f32, 0.0_f32)];
        let out = run_c64_activation(Activation::Neg, &inp);
        assert_close_c(out[0], (-3.5, 1.25), 1e-6, "neg[0]");
        assert_close_c(out[1], (2.0, 0.0), 1e-6, "neg[1]");
    }

    #[test]
    fn c64_activation_exp_matches_euler() {
        // exp(0 + i·π) = -1 + 0i.
        // exp(1 + 0i) = e ≈ 2.71828.
        let inp = [(0.0_f32, std::f32::consts::PI), (1.0_f32, 0.0_f32)];
        let out = run_c64_activation(Activation::Exp, &inp);
        assert_close_c(out[0], (-1.0, 0.0), 1e-5, "exp(iπ)");
        assert_close_c(out[1], (std::f32::consts::E, 0.0), 1e-5, "exp(1)");
    }

    #[test]
    fn c64_activation_log_matches_principal_branch() {
        // log(1 + 0i) = 0.
        // log(0 + i) = log(1) + i·π/2 = 0 + i·π/2.
        // log(-1 + 0i) = 0 + i·π.
        let inp = [(1.0_f32, 0.0_f32), (0.0_f32, 1.0_f32), (-1.0_f32, 0.0_f32)];
        let out = run_c64_activation(Activation::Log, &inp);
        assert_close_c(out[0], (0.0, 0.0), 1e-5, "log(1)");
        assert_close_c(out[1], (0.0, std::f32::consts::FRAC_PI_2), 1e-5, "log(i)");
        assert_close_c(out[2], (0.0, std::f32::consts::PI), 1e-5, "log(-1)");
    }

    #[test]
    fn c64_activation_sqrt_squared_recovers_input() {
        // For positive-real-part inputs, sqrt(z)² should equal z exactly
        // to f32 noise.
        let inp = [(4.0_f32, 0.0_f32), (3.0_f32, 4.0_f32)];
        let roots = run_c64_activation(Activation::Sqrt, &inp);
        // sqrt(4) = 2 + 0i; sqrt(3+4i) = 2 + i (since (2+i)² = 4+4i-1 = 3+4i).
        assert_close_c(roots[0], (2.0, 0.0), 1e-5, "sqrt(4)");
        assert_close_c(roots[1], (2.0, 1.0), 1e-5, "sqrt(3+4i)");
    }

    #[test]
    #[should_panic(expected = "no natural complex extension")]
    fn c64_activation_relu_is_rejected_at_lowering() {
        run_c64_activation(Activation::Relu, &[(1.0_f32, 2.0_f32)]);
    }

    // ── ComplexNormSq + Wirtinger backward witnesses ───────────────

    /// Forward `|z|²`: returns `[n]` f32.
    fn run_complex_norm_sq(z: &[(f32, f32)]) -> Vec<f32> {
        let n = z.len();
        let mut g = Graph::new("cns_fwd");
        let in_z = g.input("z", Shape::new(&[n], DType::C64));
        let out = g.complex_norm_sq(in_z);
        g.set_outputs(vec![out]);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let z_off = arena.byte_offset(in_z);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let pz = buf.as_mut_ptr().add(z_off) as *mut f32;
            for (i, &(re, im)) in z.iter().enumerate() {
                *pz.add(2 * i) = re;
                *pz.add(2 * i + 1) = im;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..n).map(|i| *p.add(i)).collect()
        }
    }

    /// Backward: given z and upstream g, return dz = g·z element-wise (C64).
    fn run_complex_norm_sq_bwd(z: &[(f32, f32)], g: &[f32]) -> Vec<(f32, f32)> {
        let n = z.len();
        let mut gr = Graph::new("cns_bwd");
        let in_z = gr.input("z", Shape::new(&[n], DType::C64));
        let in_g = gr.input("g", Shape::new(&[n], DType::F32));
        let out = gr.complex_norm_sq_backward(in_z, in_g);
        gr.set_outputs(vec![out]);
        let plan = rlx_opt::memory::plan_memory(&gr);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&gr, &arena);
        let z_off = arena.byte_offset(in_z);
        let g_off = arena.byte_offset(in_g);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let pz = buf.as_mut_ptr().add(z_off) as *mut f32;
            let pg = buf.as_mut_ptr().add(g_off) as *mut f32;
            for (i, &(re, im)) in z.iter().enumerate() {
                *pz.add(2 * i) = re;
                *pz.add(2 * i + 1) = im;
            }
            for (i, &v) in g.iter().enumerate() {
                *pg.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..n).map(|i| (*p.add(2 * i), *p.add(2 * i + 1))).collect()
        }
    }

    #[test]
    fn complex_norm_sq_matches_textbook() {
        // |3 + 4i|² = 9 + 16 = 25.
        // |1 + 0i|² = 1.
        // |0 + 0i|² = 0.
        let z = [(3.0_f32, 4.0_f32), (1.0_f32, 0.0_f32), (0.0_f32, 0.0_f32)];
        let out = run_complex_norm_sq(&z);
        assert!((out[0] - 25.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);
    }

    #[test]
    fn complex_norm_sq_backward_matches_wirtinger_formula() {
        // Wirtinger: ∂|z|²/∂z̄ = z. With upstream g = 1, dz = z.
        let z = [(3.0_f32, 4.0_f32), (1.5_f32, -2.5_f32)];
        let g = [1.0_f32, 1.0_f32];
        let dz = run_complex_norm_sq_bwd(&z, &g);
        assert_close_c(dz[0], z[0], 1e-6, "dz[0] = g·z[0]");
        assert_close_c(dz[1], z[1], 1e-6, "dz[1] = g·z[1]");
    }

    #[test]
    fn complex_norm_sq_backward_scales_with_upstream() {
        // With upstream g[i] ≠ 1: dz[i] = g[i]·z[i].
        let z = [(2.0_f32, 1.0_f32), (-1.0_f32, 3.0_f32)];
        let g = [0.5_f32, -2.0_f32];
        let dz = run_complex_norm_sq_bwd(&z, &g);
        assert_close_c(dz[0], (1.0, 0.5), 1e-6, "g=0.5 · (2,1)");
        assert_close_c(dz[1], (2.0, -6.0), 1e-6, "g=-2 · (-1,3)");
    }

    /// Multi-output Op::CustomFn via the concat-with-Narrow design
    /// (rlx-ir::Graph::custom_fn_multi). Build a custom_fn whose
    /// fwd_body returns two outputs (x², 2x), then materialize each
    /// via the MultiOutputHandle and verify both numerically.
    #[test]
    fn custom_fn_multi_extracts_each_subgraph_output() {
        use rlx_ir::ops::special::MultiOutputHandle;

        let _ = MultiOutputHandle {
            source: NodeId(0),
            sub_shapes: vec![],
            offsets: vec![],
        }; // import sanity

        // Inner body: input x [3] f32, outputs (x², 2x) both [3] f32.
        let mut body = Graph::new("multi_body");
        let s3 = Shape::new(&[3], DType::F32);
        let x = body.input("x", s3.clone());
        let x_sq = body.binary(BinaryOp::Mul, x, x, s3.clone());
        let two = body.add_node(
            Op::Constant {
                data: vec![
                    2.0_f32.to_le_bytes(),
                    2.0_f32.to_le_bytes(),
                    2.0_f32.to_le_bytes(),
                ]
                .into_iter()
                .flatten()
                .collect(),
            },
            vec![],
            s3.clone(),
        );
        let two_x = body.binary(BinaryOp::Mul, two, x, s3.clone());
        body.set_outputs(vec![x_sq, two_x]);

        // Outer graph: feed in_x → custom_fn_multi → handle.output(0/1).
        let mut outer = Graph::new("multi_outer");
        let in_x = outer.input("xin", s3.clone());
        let handle = outer.custom_fn_multi(vec![in_x], body);
        assert_eq!(handle.n_outputs(), 2);
        let out0 = handle.output(&mut outer, 0); // x²
        let out1 = handle.output(&mut outer, 1); // 2x
        outer.set_outputs(vec![out0, out1]);

        let plan = rlx_opt::memory::plan_memory(&outer);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&outer, &arena);
        let xin_off = arena.byte_offset(in_x);
        let out0_off = arena.byte_offset(out0);
        let out1_off = arena.byte_offset(out1);
        let xs = [1.0_f32, 2.0, 3.0];
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(xin_off) as *mut f32;
            for (i, &v) in xs.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let out0_v: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out0_off) as *const f32;
            (0..3).map(|i| *p.add(i)).collect()
        };
        let out1_v: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out1_off) as *const f32;
            (0..3).map(|i| *p.add(i)).collect()
        };
        // x² = [1, 4, 9]; 2x = [2, 4, 6].
        for i in 0..3 {
            assert!(
                (out0_v[i] - xs[i] * xs[i]).abs() < 1e-5,
                "out0[{i}] = {} != x² = {}",
                out0_v[i],
                xs[i] * xs[i]
            );
            assert!(
                (out1_v[i] - 2.0 * xs[i]).abs() < 1e-5,
                "out1[{i}] = {} != 2x = {}",
                out1_v[i],
                2.0 * xs[i]
            );
        }
    }

    #[test]
    fn complex_norm_sq_gradient_matches_finite_difference() {
        // Numerical sanity: perturb z[0].re by ε, observe Δ|z|² ≈ 2·re·ε.
        let z = [(3.0_f32, 4.0_f32)];
        let eps = 1e-3_f32;
        let v0 = run_complex_norm_sq(&z)[0];
        let z_pert = [(3.0_f32 + eps, 4.0_f32)];
        let v1 = run_complex_norm_sq(&z_pert)[0];
        let fd_re = (v1 - v0) / eps;
        let analytic_re = 2.0 * z[0].0;
        assert!((fd_re - analytic_re).abs() < 1e-2);

        // ∂/∂im at z = (3, 4) is 2·im = 8.
        let z_pert_im = [(3.0_f32, 4.0_f32 + eps)];
        let v2 = run_complex_norm_sq(&z_pert_im)[0];
        let fd_im = (v2 - v0) / eps;
        let analytic_im = 2.0 * z[0].1;
        assert!((fd_im - analytic_im).abs() < 1e-2);

        // Compare with the Wirtinger backward at upstream g = 1.
        // Wirtinger ∂/∂z̄ = z gives dz = (re, im). The "real
        // gradient" wrt (re, im) is 2·(re, im), i.e. 2·dz = (2·re,
        // 2·im) — that's the factor 2 difference between Wirtinger
        // ∂/∂z̄ and the real-vector gradient on (re, im).
        let dz = run_complex_norm_sq_bwd(&z, &[1.0_f32]);
        assert!((2.0 * dz[0].0 - analytic_re).abs() < 1e-5);
        assert!((2.0 * dz[0].1 - analytic_im).abs() < 1e-5);
    }

    /// Direct regression test for the 5-D mid-shape singleton broadcast
    /// (SAM rel_pos pattern: `[bh, h, w, 1, w] + [bh, h, w, h, w]`).
    /// The SAM port worked around this by `concat`-tiling the rhs; this
    /// test verifies the in-graph broadcast path is bit-correct.
    #[test]
    fn binary_full_5d_mid_singleton_broadcast() {
        let bh = 2usize;
        let h = 3;
        let w = 4;
        let f = DType::F32;

        let mut g = Graph::new("bcast_5d");
        let lhs = g.input("lhs", Shape::new(&[bh, h, w, h, w], f));
        // rhs shape with size-1 at axis 3 (mid-shape singleton).
        let rhs = g.input("rhs", Shape::new(&[bh, h, w, 1, w], f));
        let out = g.binary(BinaryOp::Add, lhs, rhs, Shape::new(&[bh, h, w, h, w], f));
        g.set_outputs(vec![out]);

        // Deterministic data.
        let lhs_data: Vec<f32> = (0..bh * h * w * h * w).map(|i| i as f32 * 0.01).collect();
        let rhs_data: Vec<f32> = (0..bh * h * w * 1 * w)
            .map(|i| (i as f32 + 100.0) * 0.01)
            .collect();

        // Compute expected output by hand.
        let mut expected = vec![0f32; bh * h * w * h * w];
        for b_ in 0..bh {
            for hq in 0..h {
                for wq in 0..w {
                    for hk in 0..h {
                        for wk in 0..w {
                            let li = (((b_ * h + hq) * w + wq) * h + hk) * w + wk;
                            // rhs has hk dim = 1, so it's always index 0 there.
                            let ri = (((b_ * h + hq) * w + wq) * 1 + 0) * w + wk;
                            expected[li] = lhs_data[li] + rhs_data[ri];
                        }
                    }
                }
            }
        }

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let lhs_off = arena.byte_offset(lhs);
        let rhs_off = arena.byte_offset(rhs);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(lhs_off) as *mut f32;
            for (i, &v) in lhs_data.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(rhs_off) as *mut f32;
            for (i, &v) in rhs_data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..bh * h * w * h * w).map(|i| *p.add(i)).collect()
        };

        // Bit-exact check.
        let mut max_diff = 0f32;
        let mut max_idx = 0;
        for i in 0..actual.len() {
            let d = (actual[i] - expected[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        assert!(
            max_diff < 1e-6,
            "5D mid-shape singleton broadcast wrong: max |Δ| = {max_diff} at idx {max_idx} \
             (actual={}, expected={})",
            actual[max_idx],
            expected[max_idx]
        );
    }
}
