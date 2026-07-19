#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Instrumentation: count of `FusedNomicLayer` thunks emitted by the full-layer
/// fusion in this process. The fusion is otherwise invisible through the
/// `Session` API, so out-of-tree model tests (rlx-models' Nomic parity test)
/// load this to confirm the fusion actually fired (non-vacuous parity). Reset
/// by storing 0 before a compile.
pub static FUSED_NOMIC_LAYER_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// A pre-compiled kernel call with all args resolved to arena offsets.
#[derive(Clone)]
pub enum Thunk {
    /// Skip (Input/Param already in arena)
    Nop,
    /// Fused element-wise region: ONE pass applies the whole `chain` of scalar
    /// steps per output element (bias-add → relu → … collapse into a single
    /// kernel, no intermediate tensors). This is RLX's XLA-style auto-fusion
    /// path on CPU — the GPU backends already run the same `ElementwiseRegion`
    /// IR via a generated kernel; this is the scalar interpreter twin so any
    /// fused chain runs without a hand-written per-op kernel.
    /// `dst`/`input_offs` are byte offsets; `input_modulus[i]` is input i's
    /// broadcast period (0 ⇒ read element `gid`; scalar inputs read element 0).
    ElementwiseRegion {
        dst: usize,
        len: u32,
        input_offs: Vec<usize>,
        chain: Vec<rlx_ir::op::ChainStep>,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
    },
    /// C = A @ B (BLAS sgemm)
    Sgemm {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
    },
    /// `C[m,n] = op(A) @ op(B)` with optional transpose of each operand,
    /// done via cblas trans flags (no materialized transpose). Emitted when
    /// the thunk compiler folds a last-two-axis `Op::Transpose` feeding a
    /// matmul (the dominant cost in matmul backprop). `a`/`b` are the
    /// *pre-transpose* operands. lda/ldb are derived from the trans flags.
    SgemmT {
        a: usize,
        b: usize,
        c: usize,
        m: u32,
        k: u32,
        n: u32,
        ta: bool,
        tb: bool,
    },
    /// Fused SGD-with-momentum parameter update, one pass per param.
    /// Computes `v' = mom·v + g ; p' = p − lr·v'` reading `param`/`vel`/
    /// `grad` and writing `p_out`/`v_out`. Emitted when the thunk compiler
    /// folds the in-graph SGD chain `Sub(p, Mul(Add(Mul(v,mom),g), lr))`
    /// (4 `BinaryFull` ops + 2 full-size constants per param) into a single
    /// kernel — the dominant cost of the fully-fused training step. All
    /// fields are arena byte offsets except the scalar hyperparameters.
    SgdMomentum {
        param: usize,
        vel: usize,
        grad: usize,
        p_out: usize,
        v_out: usize,
        lr: f32,
        mom: f32,
        len: u32,
    },
    /// Complex (C64) dense GEMM `C = A·B`. Operands are interleaved
    /// `[re, im]` f32; `a`/`b`/`c` are byte offsets, `m`/`k`/`n` are
    /// complex-element matrix dims (`A` `[m,k]`, `B` `[k,n]`, `C` `[m,n]`).
    CgemmC64 {
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
    /// Batched f32 dense solve — loop of `sgesv` per batch slice.
    BatchedDenseSolveF32 {
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
        /// `true` when operand `a`/`b` has batch dim 1 and is BROADCAST across
        /// every output batch (its per-matrix stride is 0 — reuse matrix 0).
        /// Without this the loop over-reads a batch-1 operand for `bi>0`.
        a_bcast: bool,
        b_bcast: bool,
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
    CopyF64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i64 element copy (Reshape/Cast on i64 tensors).
    CopyI64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// Truncating f32 → i64 (ONNX Cast; toward zero).
    CastF32ToI64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    CastF32ToF64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// Truncating f32 → i32 (ONNX Cast; toward zero).
    CastF32ToI32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i64 → f32 (ONNX Cast on shape scalars, e.g. Albert head-dim).
    CastI64ToF32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// bool → i32 (BERT attention mask grid).
    CastBoolToI32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    CastBoolToF32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// f32 → Bool (`x != 0.0`). Lets logical And/Or lower their {0,1} f32
    /// arithmetic result straight back to a 1-byte Bool without a scalar
    /// `Compare` (whose thunk has no broadcast and would read a scalar rhs
    /// out of bounds).
    CastF32ToBool {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i32 → f32 (BERT attention mask cast before subtract).
    CastI32ToF32 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i32 → i64 (widen integer indices/ids; e.g. an ONNX graph with I32
    /// `context_tokens` feeding a codebook Gather whose kernel wants i64).
    CastI32ToI64 {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i32 → bool: `out[i] = (in[i] != 0) as u8`. Element-wise; the generic Copy
    /// fallback byte-copies (4-byte i32 → 1-byte bool) and reads garbage. Needed by
    /// MOSS-TTS's `Cast(attention_mask i32 → bool)` in the RoPE-position cumsum.
    CastI32ToBool {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// i64 → bool: `out[i] = (in[i] != 0) as u8`. Element-wise; Copy would treat
    /// 8-byte lanes as 1-byte bools. Soprano KV-backbone casts `attention_mask`
    /// (I64 ones) to Bool before the causal Where(-inf) mask.
    CastI64ToBool {
        src: usize,
        dst: usize,
        len: u32,
    },
    /// bool → i64: `out[i] = in[i] as i64` (0/1). Widen a bool mask for CumSum/arith.
    CastBoolToI64 {
        src: usize,
        dst: usize,
        len: u32,
    },
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
        inputs: Vec<(usize, u32, u32)>,
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
        /// Element size (4 = F32, 8 = I64).
        elem_bytes: u8,
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
        /// 1 when the index tensor is i64 (ONNX Gather indices).
        idx_i64: u8,
        /// Element size of table/output (4 = f32, 8 = i64).
        table_bytes: u8,
    },
    /// Narrow: copy slice (`elem_bytes` = source element size: 4 for f32, 8 for f64).
    Narrow {
        src: usize,
        dst: usize,
        outer: u32,
        src_stride: u32,
        dst_stride: u32,
        inner: u32,
        elem_bytes: u8,
    },
    /// Reverse (flip) element order along the axes in `rev_mask`. `dims` is the
    /// row-major input shape; output shape is identical. Dtype-agnostic
    /// (byte-copy of `elem_bytes` per element).
    Reverse {
        src: usize,
        dst: usize,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u8,
    },
    /// Copy (reshape, expand)
    Copy {
        src: usize,
        dst: usize,
        len: u32,
    },
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
    /// GroupNorm on NCHW `[N,C,H,W]`.
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
    },
    /// BatchNorm inference: frozen mean/var, feature axis last.
    BatchNormInference {
        src: usize,
        g: usize,
        b: usize,
        mean: usize,
        var: usize,
        dst: usize,
        count: u32,
        channels: u32,
        eps: f32,
    },
    BatchNormInferenceBackwardInput {
        x: usize,
        gamma: usize,
        mean: usize,
        var: usize,
        dy: usize,
        dx: usize,
        count: u32,
        channels: u32,
        eps: f32,
    },
    BatchNormInferenceBackwardGamma {
        x: usize,
        mean: usize,
        var: usize,
        dy: usize,
        dgamma: usize,
        count: u32,
        channels: u32,
        eps: f32,
    },
    BatchNormInferenceBackwardBeta {
        dy: usize,
        dbeta: usize,
        count: u32,
        channels: u32,
    },
    /// LayerNorm2d on NCHW (SAM / candle semantics).
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
    },
    /// ConvTranspose2d on NCHW.
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
    },
    /// 3D convolution on NCDHW. Input [N, C_in, D, H, W], weight
    /// [C_out, C_in/groups, kD, kH, kW], output [N, C_out, D_out, H_out, W_out].
    /// Naive direct convolution; the depth-axis analogue of `Thunk::Conv2D`.
    Conv3d {
        src: usize,
        weight: usize,
        dst: usize,
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
        dw: u32,
        groups: u32,
    },
    /// ConvTranspose3d on NCDHW. Weight [C_in, C_out/groups, kD, kH, kW].
    /// `output_padding` is already folded into `d_out/h_out/w_out`.
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
    },
    /// Nearest 2× upsample on NCHW (per-batch slice).
    ResizeNearest2x {
        src: usize,
        dst: usize,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// SAM2 axial 2-D RoPE on `[batch, seq, num_heads * head_dim]`.
    AxialRope2d {
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
    /// DiT adaLN-Zero modulation: `out = norm(x)·(1+scale)+shift`, one pass —
    /// mean/var (or RMS) and the per-conditioning affine fused, the pre-norm
    /// activation never materialized. `scale`/`shift` share the modulation
    /// shape (`[B,1,D]`); `mod_off[row]` is the element offset into each for
    /// output `row` (the built-in broadcast that avoids an `Op::Expand`).
    AdaLayerNorm {
        src: usize,
        scale: usize,
        shift: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
        /// `true` ⇒ LayerNorm (mean+var); `false` ⇒ RMSNorm (rms only).
        layer_norm: bool,
        /// Element count of the `scale`/`shift` buffers (they share a shape).
        mod_len: u32,
        /// Per-row element offset into `scale`/`shift` (broadcast map).
        mod_off: Vec<u32>,
    },
    /// DiT gated residual: `out = x + gate·y`. `gate` carries the modulation
    /// shape (`[B,1,D]`); `gate_off[row]` is the element offset into `gate` for
    /// output `row`.
    GatedResidual {
        x: usize,
        y: usize,
        gate: usize,
        dst: usize,
        rows: u32,
        h: u32,
        /// Element count of the `gate` buffer.
        gate_len: u32,
        /// Per-row element offset into `gate` (broadcast map).
        gate_off: Vec<u32>,
    },
    /// Packed `[dx ∥ dscale ∥ dshift]` for [`Op::AdaLayerNormBackward`].
    AdaLayerNormBackward {
        x: usize,
        scale: usize,
        shift: usize,
        dy: usize,
        dst: usize,
        rows: u32,
        h: u32,
        eps: f32,
        layer_norm: bool,
        mod_len: u32,
        mod_off: Vec<u32>,
    },
    /// Packed `[dx ∥ dy ∥ dgate]` for [`Op::GatedResidualBackward`].
    GatedResidualBackward {
        x: usize,
        y: usize,
        gate: usize,
        dy: usize,
        dst: usize,
        rows: u32,
        h: u32,
        gate_len: u32,
        gate_off: Vec<u32>,
    },
    /// Softmax
    Softmax {
        data: usize,
        rows: u32,
        cols: u32,
    },
    /// Inclusive (or exclusive) cumulative sum along the last axis
    /// (callers pre-flatten higher-dim cumsums via reshape views).
    Cumsum {
        src: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        exclusive: bool,
        /// Element dtype — the scan runs in f32 or i64 (integer position ids /
        /// ragged offsets, e.g. MOSS-TTS's `cumsum(attention_mask)-1` RoPE positions
        /// must NOT go through the f32 path, which reads the i64 buffer as garbage).
        dtype: rlx_ir::DType,
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
        /// When non-zero, load initial `[b, h, n, n]` state and write
        /// the final state back in place after the scan.
        state: usize,
        dst: usize,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
    },

    /// Multi-layer (optionally bidirectional, optional carry) LSTM with
    /// packed weights. See `Op::Lstm`. `h0`/`c0` are valid only when
    /// `carry`; `dst` is `[b, s, D*h]`.
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
    /// GRU (gate order r, z, n) with separate `b_ih`/`b_hh`. See `Op::Gru`.
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
    /// Elman RNN (`tanh`/`relu`), single merged bias. See `Op::Rnn`.
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
    /// Mamba-2 / SSD scalar-decay SSM scan. See `Op::Mamba2`.
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
        x: usize,   // f32 activations [m, k]
        w_q: usize, // packed weight bytes (k*n elements packed)
        dst: usize, // f32 output [m, n]
        m: u32,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
    },

    /// Int4 block dequant + matmul (packed nibbles, side scale/zp).
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

    /// FP8 dequant + matmul (per-tensor or per-column scale).
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

    /// NVFP4 (E2M1) block dequant + matmul — 16-wide groups, FP8 scales.
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

    /// Native low-precision scaled GEMM (FP8/FP6/FP4) — CPU reference oracle.
    /// TN layout: lhs [m,k], rhs [n,k] codes, out [m,n] f32.
    ScaledMatMul {
        lhs: usize,
        rhs: usize,
        lhs_scale: usize,
        rhs_scale: usize,
        bias: usize, // valid iff has_bias
        dst: usize,
        m: u32,
        k: u32,
        n: u32,
        lhs_fmt: rlx_ir::ScaledFormat,
        rhs_fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
        has_bias: bool,
    },

    /// Quantize f32 → packed low-precision codes (reads the scale tensor).
    ScaledQuantize {
        x: usize,
        scale: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
    },

    /// Compute the per-tensor / per-block scale tensor for `Op::ScaledQuantize`.
    ScaledQuantScale {
        x: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
    },

    /// Reconstruct f32 from packed codes (`Op::ScaledDequantize`).
    ScaledDequantize {
        codes: usize,
        scale: usize,
        dst: usize,
        rows: u32,
        cols: u32,
        fmt: rlx_ir::ScaledFormat,
        layout: rlx_ir::ScaleLayout,
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
    /// ONNX `RandomNormalLike` fill.
    RngNormal {
        dst: usize,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    /// ONNX `RandomUniformLike` fill.
    RngUniform {
        dst: usize,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
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
        /// GQA/MQA: number of key/value heads that the query heads share.
        /// Equals `heads` for plain MHA (the common case); only `< heads` on
        /// the non-fused standalone `Op::Attention` path.
        kv_heads: u32,
        head_dim: u32,
        mask_kind: rlx_ir::op::MaskKind,
        /// Softmax score scale (`Op::Attention::score_scale`). `head_dim^-0.5`
        /// when the op left it unset. Must be honored — Gemma 4 uses `1.0`
        /// (Q/K are per-head RMS-normed, so no `1/sqrt(d)` pre-scale).
        scale: f32,
        /// Attention logit soft-cap (`Op::Attention::attn_logit_softcap`).
        /// `cap*tanh(score/cap)` applied pre-softmax; 0 = disabled. Gemma 2
        /// uses 50.0 — must be honored to match HF / rlx-metal / rlx-cuda.
        softcap: f32,
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
    /// [`Op::AttentionBackward`] — emits dQ, dK, or dV (see `wrt`).
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
        mask_kind: rlx_ir::op::MaskKind,
        wrt: rlx_ir::op::AttentionBwdWrt,
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
        n_rot: u32,
        cos_len: u32,
        src_row_stride: u32,
        /// `true` = GPT-J / llama.cpp-NORM interleaved pairs `(2i, 2i+1)`;
        /// `false` = HF / NeoX rotate-half pairs `(i, i+n_rot/2)`.
        interleaved: bool,
    },
    /// Fused attention block: QKV proj → split → \[RoPE\] → SDPA → output proj.
    /// All intermediates stay in L1 cache. Zero arena writes between ops.
    FusedAttnBlock {
        hidden: usize,
        qkv_w: usize,
        out_w: usize,
        mask: usize,
        /// How to mask attention scores. `Custom` reads the per-key `mask`
        /// buffer (BERT-style padding); `Causal` / `SlidingWindow` are
        /// synthesized in-kernel with no buffer; `None` applies no mask.
        /// Mirrors [`Thunk::Attention`]'s `mask_kind` — the attention-block
        /// fusion MUST carry it through, otherwise a causal decoder silently
        /// attends to future tokens (the fused kernel would only ever apply
        /// the per-key padding mask).
        mask_kind: rlx_ir::op::MaskKind,
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
        /// RoPE pairing: `true` = GPT-J interleaved `(2i,2i+1)`, `false` = NeoX
        /// rotate-half `(i,i+d/2)`. Captured from the fused `Op::Rope` so the
        /// inline rope matches the standalone kernel (a GptJ model must not be
        /// silently rotated NeoX-style by the fused path).
        interleaved: bool,
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
        /// RoPE pairing, threaded from the consumed `FusedAttnBlock`
        /// (`true` = GPT-J interleaved, `false` = NeoX rotate-half).
        interleaved: bool,
    },
    /// Fused SwiGLU: out\[r,i\] = x\[r,i\] * silu(x[r, n_half+i]).
    /// Input: [outer, 2*n_half] — concatenated up||gate per row.
    /// Output: [outer, n_half].
    FusedSwiGLU {
        src: usize,
        dst: usize,
        n_half: u32,
        total: u32,
        gate_first: bool,
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
        /// `(src_offset, axis_extent, input_numel)` — `input_numel` enables
        /// outer-dim broadcast when rank-deficient inputs are concatenated.
        inputs: Vec<(usize, u32, u32)>,
    },
    /// Element-wise comparison: out = (lhs CMP rhs) ? 1 : 0 (Bool u8 or F32 0/1).
    /// `lhs_scalar` / `rhs_scalar` broadcast a 1-element operand across `len`
    /// (ONNX Less/Greater with a scalar Constant).
    Compare {
        lhs: usize,
        rhs: usize,
        dst: usize,
        len: u32,
        op: CmpOp,
        /// Nonzero when lhs/rhs are i64 (mask/range ops).
        inputs_i64: u8,
        /// Input element size (1 = Bool, 4 = F32, 8 = I64).
        inputs_elem_bytes: u8,
        /// Output element size (1 = Bool, 4 = F32).
        dst_elem_bytes: u8,
        lhs_scalar: bool,
        rhs_scalar: bool,
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
    /// Index of the max (`is_max`) or min along the reduced axis; writes the
    /// winning index as an f32 into `dst`.
    ArgReduce {
        src: usize,
        dst: usize,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
    },
    /// Top-K **indices** along the last axis. Input shape `[outer, axis_dim]`,
    /// output `[outer, k]` (f32 or i64 per `indices_i64`). Ties broken by
    /// smaller index. Used by MoE gating + beam search.
    TopK {
        src: usize,
        dst: usize,
        outer: u32,
        axis_dim: u32,
        k: u32,
        indices_i64: u8,
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
    /// GGUF K-quant packed expert stack + grouped matmul (MoE FFN).
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
    /// Materialize packed MoE weights to F32 `[E, K, N]` (autodiff helper).
    DequantMoEWeightsGguf {
        w_q: usize,
        dst: usize,
        k_dim: u32,
        n: u32,
        num_experts: u32,
        scheme: rlx_ir::quant::QuantScheme,
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
    /// ONNX ScatterND: copy `data`, write `updates` at multi-index locations.
    /// Inputs `[data, indices, updates]`; `indices_i64` selects I64 vs F32 index storage.
    ScatterNd {
        data: usize,
        indices: usize,
        updates: usize,
        dst: usize,
        data_shape: Vec<u32>,
        indices_shape: Vec<u32>,
        data_len: u32,
        updates_len: u32,
        indices_len: u32,
        indices_i64: u8,
        reduction: rlx_ir::ScatterNdReduction,
    },
    /// ONNX ScatterElements along `axis`.
    ScatterElements {
        data: usize,
        indices: usize,
        updates: usize,
        dst: usize,
        data_shape: Vec<u32>,
        data_len: u32,
        updates_len: u32,
        indices_len: u32,
        indices_i64: u8,
        axis: i32,
        reduction: rlx_ir::ScatterNdReduction,
    },
    /// ONNX GatherND.
    GatherNd {
        data: usize,
        indices: usize,
        dst: usize,
        data_shape: Vec<u32>,
        indices_shape: Vec<u32>,
        data_len: u32,
        indices_len: u32,
        out_len: u32,
        indices_i64: u8,
        batch_dims: i32,
    },
    /// ONNX GatherElements / take_along_axis.
    GatherElements {
        data: usize,
        indices: usize,
        dst: usize,
        data_shape: Vec<u32>,
        indices_shape: Vec<u32>,
        data_len: u32,
        indices_len: u32,
        out_len: u32,
        indices_i64: u8,
        axis: i32,
    },
    /// Ternary select: out = cond != 0 ? on_true : on_false.
    /// `*_scalar` broadcast a 1-element operand across `len`.
    Where {
        cond: usize,
        on_true: usize,
        on_false: usize,
        dst: usize,
        len: u32,
        elem_bytes: u8,
        /// Element size for cond (1 = Bool mask, 4 = F32 0/1).
        cond_elem_bytes: u8,
        cond_scalar: bool,
        true_scalar: bool,
        false_scalar: bool,
    },
    /// Single-rounded fused multiply-add: `dst = a*b + c` (one rounding —
    /// enables error-free transforms / compensated arithmetic).
    Fma {
        a: usize,
        b: usize,
        c: usize,
        dst: usize,
        len: u32,
        elem_bytes: u8,
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
        elem_bytes: u8,
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
        idx_i64: u8,
        table_bytes: u8,
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

    /// NCHW im2col for conv backward-weight matmul. Output `[M, C·kH·kW]`
    /// with `M = N · H_out · W_out`. `n == 0` means infer batch from `x`.
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

    /// Fused softmax + cross-entropy loss against a dense target
    /// distribution. `logits [N, C]`, `targets [N, C]`, output `[N]`
    /// per-row loss `lse(logits[n]) - Σ_c targets[n,c]·logits[n,c]`.
    /// Numerically stable (max-subtract before exp).
    SoftmaxCrossEntropyDense {
        logits: usize,
        targets: usize,
        dst: usize,
        n: u32,
        c: u32,
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
    /// CPU reference 3D Gaussian splat render ([`rlx_ir::Op::GaussianSplatRender`]).
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
    /// Strict IR stage 1 — project + bin + sort + rays ([`Op::GaussianSplatPrepare`]).
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
    /// Strict IR stage 2 — tile raster from prepare buffer ([`Op::GaussianSplatRasterize`]).
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
    Fft1d {
        src: usize,
        dst: usize,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype: rlx_ir::DType,
    },
    FftButterflyStage {
        state_src: usize,
        state_dst: usize,
        gate_src: usize,
        rev_src: usize,
        tw_re_src: usize,
        tw_im_src: usize,
        batch: u32,
        n_fft: u32,
        stage: u32,
    },
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
}

/// Compiled thunk schedule — the runtime hot path.
/// Nop thunks are filtered out at compile time for zero iteration overhead.
#[derive(Clone)]
pub struct ThunkSchedule {
    pub thunks: Vec<Thunk>,
    /// TIDE merged placement mask (union across layers).
    pub moe_resident: Option<std::sync::Arc<[bool]>>,
    /// Per MoE layer placement (`layer[e]`); preferred when set.
    pub moe_resident_layers: Option<std::sync::Arc<Vec<std::sync::Arc<[bool]>>>>,
    /// MoE router TopK capture (per-layer refresh).
    pub moe_topk_capture: Option<std::sync::Arc<crate::moe_topk_capture::MoeTopkCapture>>,
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
    /// Runtime-mutable RNG policy for [`Thunk::RngNormal`] / [`Thunk::RngUniform`].
    pub rng: Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
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
pub(crate) fn node_offset(arena: &Arena, id: NodeId) -> usize {
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
pub(crate) fn thunk_read_offsets(t: &Thunk) -> Vec<usize> {
    match t {
        Thunk::Sgemm { a, b, .. } => vec![*a, *b],
        Thunk::SgemmT { a, b, .. } => vec![*a, *b],
        Thunk::SgdMomentum {
            param, vel, grad, ..
        } => vec![*param, *vel, *grad],
        Thunk::DenseSolveF64 { a, b, .. } => vec![*a, *b],
        Thunk::DenseSolveF32 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedDenseSolveF64 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedDgemmF64 { a, b, .. } => vec![*a, *b],
        Thunk::BatchedSgemm { a, b, .. } => vec![*a, *b],
        Thunk::FusedMmBiasAct { a, w, bias, .. } => vec![*a, *w, *bias],
        Thunk::ElementwiseRegion { input_offs, .. } => input_offs.clone(),
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
        Thunk::LayerNorm { src, g, b, .. } | Thunk::GroupNorm { src, g, b, .. } => {
            vec![*src, *g, *b]
        }
        Thunk::BatchNormInference {
            src,
            g,
            b,
            mean,
            var,
            ..
        } => vec![*src, *g, *b, *mean, *var],
        Thunk::ResizeNearest2x { src, .. } => vec![*src],
        Thunk::AxialRope2d { src, .. } => vec![*src],
        Thunk::FusedResidualLN {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::FusedResidualRmsNorm {
            x, res, bias, g, b, ..
        } => vec![*x, *res, *bias, *g, *b],
        Thunk::RmsNorm { src, g, b, .. } => vec![*src, *g, *b],
        Thunk::AdaLayerNorm {
            src, scale, shift, ..
        } => vec![*src, *scale, *shift],
        Thunk::GatedResidual { x, y, gate, .. } => vec![*x, *y, *gate],
        Thunk::AdaLayerNormBackward {
            x,
            scale,
            shift,
            dy,
            ..
        } => vec![*x, *scale, *shift, *dy],
        Thunk::GatedResidualBackward { x, y, gate, dy, .. } => vec![*x, *y, *gate, *dy],
        Thunk::Softmax { data, .. } => vec![*data],
        Thunk::Cumsum { src, .. } => vec![*src],
        Thunk::Sample { logits, .. } => vec![*logits],
        Thunk::RngNormal { .. } | Thunk::RngUniform { .. } => vec![],
        Thunk::LoraMatMul { x, w, a, b, .. } => vec![*x, *w, *a, *b],
        Thunk::DequantMatMul {
            x, w_q, scale, zp, ..
        } => vec![*x, *w_q, *scale, *zp],
        Thunk::DequantMatMulGguf { x, w_q, .. } => vec![*x, *w_q],
        Thunk::DequantMatMulInt4 {
            x, w_q, scale, zp, ..
        } => vec![*x, *w_q, *scale, *zp],
        Thunk::DequantMatMulFp8 { x, w_q, scale, .. } => vec![*x, *w_q, *scale],
        Thunk::DequantMatMulNvfp4 {
            x,
            w_q,
            scale,
            global_scale,
            ..
        } => vec![*x, *w_q, *scale, *global_scale],
        Thunk::ScaledMatMul {
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            bias,
            has_bias,
            ..
        } => {
            let mut v = vec![*lhs, *rhs, *lhs_scale, *rhs_scale];
            if *has_bias {
                v.push(*bias);
            }
            v
        }
        Thunk::ScaledQuantize { x, scale, .. } => vec![*x, *scale],
        Thunk::ScaledQuantScale { x, .. } => vec![*x],
        Thunk::ScaledDequantize { codes, scale, .. } => vec![*codes, *scale],
        Thunk::Conv2D1x1 { src, weight, .. } => vec![*src, *weight],
        Thunk::SelectiveScan {
            x, delta, a, b, c, ..
        } => vec![*x, *delta, *a, *b, *c],
        Thunk::GatedDeltaNet {
            q,
            k,
            v,
            g,
            beta,
            state,
            ..
        } => {
            let mut v = vec![*q, *k, *v, *g, *beta];
            if *state != 0 {
                v.push(*state);
            }
            v
        }
        Thunk::Attention { q, k, v, mask, .. } => vec![*q, *k, *v, *mask],
        Thunk::AttentionBackward {
            q, k, v, dy, mask, ..
        } => {
            let mut v = vec![*q, *k, *v, *dy];
            if *mask != 0 {
                v.push(*mask);
            }
            v
        }
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
        Thunk::Concat { inputs, .. } => inputs.iter().map(|(off, _, _)| *off).collect(),
        Thunk::ConcatF64 { inputs, .. } => inputs.iter().map(|(off, _, _)| *off).collect(),
        Thunk::Narrow { src, .. } => vec![*src],
        Thunk::Copy { src, .. } => vec![*src],
        Thunk::Gather { table, idx, .. } => vec![*table, *idx],
        // Anything not enumerated → return the dst as a "read" too,
        // forcing the fusion to bail (read_count >= 2 → skip). Keeps
        // this list safe to be incomplete.
        _ => vec![],
    }
}

/// A scan body compiled into a self-contained CPU schedule plus the byte
/// offsets needed to drive it. Backends without a native `Op::Scan` kernel
/// (e.g. Metal via the unified-memory host fallback) build this once at compile
/// time and hand it to [`execute_scan_host`] at run time. Mirrors the inline
/// `Op::Scan` compilation used by the CPU executor.
pub struct ScanBodyPlan {
    pub body: ThunkSchedule,
    pub body_init: Vec<u8>,
    pub body_input_off: usize,
    pub body_output_off: usize,
    pub carry_bytes: usize,
    /// Body-arena offsets of the broadcast inputs, in declaration order.
    pub bcast_body_offs: Vec<usize>,
    /// Body-arena offsets of the per-step (`xs`) inputs, in declaration order.
    pub xs_body_offs: Vec<usize>,
}

impl std::fmt::Debug for ScanBodyPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanBodyPlan")
            .field("carry_bytes", &self.carry_bytes)
            .field("body_input_off", &self.body_input_off)
            .field("body_output_off", &self.body_output_off)
            .field("num_bcast", &self.bcast_body_offs.len())
            .field("num_xs", &self.xs_body_offs.len())
            .finish()
    }
}
