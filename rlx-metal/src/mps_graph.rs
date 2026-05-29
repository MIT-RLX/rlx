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

//! MPSGraph bridge — Apple's high-level graph compiler.
//!
//! `MPSGraph` (macOS 11+) takes a symbolic op tree, compiles it into one
//! optimized GPU executable, and runs the whole thing in one shot. This is
//! the structural answer to per-op dispatch overhead and the long path to
//! competitive Metal perf vs MLX.
//!
//! Architecture (this module is the bridge; full IR lowering is wired in
//! `backend::compile_mps_graph_subgraph`):
//!
//!   1. Walk our IR, build symbolic `MPSGraphTensor`s for each node.
//!   2. Call `[graph compileWithDevice:feeds:targetTensors:targetOps:...]`
//!      to get an `MPSGraphExecutable` — a self-contained compiled object.
//!   3. At runtime: bind input `MPSGraphTensorData` (backed by our arena
//!      buffer at the right offset), call `executable.runWithMTLCommandQueue:`,
//!      copy outputs back into our arena.
//!
//! This is parallel to our hand-rolled thunk system, not a replacement —
//! callers opt in via `RLX_USE_MPS_GRAPH=1` for now.

use metal::Buffer;
use metal::foreign_types::ForeignType;
use objc::runtime::{BOOL, NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use std::sync::OnceLock;

// MetalPerformanceShadersGraph is a separate framework from MPS itself.
#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
unsafe extern "C" {}

#[allow(non_upper_case_globals, dead_code)]
mod mps_dtype {
    pub const Float32: u32 = 0x10000000 | 32;
    pub const Float16: u32 = 0x10000000 | 16;
}

/// True iff MPSGraph is available on this macOS version.
pub fn mps_graph_supported() -> bool {
    static AVAIL: OnceLock<bool> = OnceLock::new();
    *AVAIL.get_or_init(|| objc::runtime::Class::get("MPSGraph").is_some())
}

/// Owned wrapper around an MPSGraph instance. Drop releases.
pub struct MpsGraph {
    obj: *mut Object,
}
unsafe impl Send for MpsGraph {}
unsafe impl Sync for MpsGraph {}

impl Drop for MpsGraph {
    fn drop(&mut self) {
        if !self.obj.is_null() {
            unsafe {
                let _: () = msg_send![self.obj, release];
            }
        }
    }
}

impl Default for MpsGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MpsGraph {
    pub fn new() -> Self {
        unsafe {
            let cls = class!(MPSGraph);
            let alloc: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![alloc, init];
            Self { obj }
        }
    }

    /// Add an input placeholder tensor of given shape and dtype.
    /// Shape is row-major; dimensions are NSNumber objects under the hood.
    pub fn placeholder(&self, shape: &[usize], dtype: u32, name: &str) -> MpsTensor {
        unsafe {
            let nsshape = ns_array_of_numbers(shape);
            let nsname = ns_string(name);
            let t: *mut Object = msg_send![self.obj,
                placeholderWithShape: nsshape
                dataType: dtype
                name: nsname];
            // placeholder returns an autoreleased reference owned by the graph.
            // The graph holds it strongly; we just hand the pointer back.
            MpsTensor { obj: t }
        }
    }

    /// `c = a @ b` (matrix multiplication).
    pub fn matmul(&self, a: &MpsTensor, b: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("matmul");
            let t: *mut Object = msg_send![self.obj,
                matrixMultiplicationWithPrimaryTensor: a.obj
                secondaryTensor: b.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// `c = a + b` element-wise (broadcasts via MPSGraph rules).
    pub fn add(&self, a: &MpsTensor, b: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("add");
            let t: *mut Object = msg_send![self.obj,
                additionWithPrimaryTensor: a.obj
                secondaryTensor: b.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// GELU activation via the analytic approximation:
    ///   `gelu(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`
    ///
    /// MPSGraph added a native `geluWithTensor:name:` only in macOS 14;
    /// the analytic form runs everywhere we ship and matches PyTorch's
    /// approximate path used in Hugging Face BERT/Nomic.
    pub fn gelu(&self, x: &MpsTensor) -> MpsTensor {
        let half = self.constant_scalar(0.5);
        let one = self.constant_scalar(1.0);
        let three = self.constant_scalar(3.0);
        let coeff = self.constant_scalar(0.044715);
        let sqrt_2_over_pi = self.constant_scalar(0.797_884_6);

        let x3 = self.power(x, &three);
        let inner_a = self.mul(&coeff, &x3);
        let inner = self.add(x, &inner_a);
        let scaled = self.mul(&sqrt_2_over_pi, &inner);
        let t = self.tanh(&scaled);
        let one_p = self.add(&one, &t);
        let xt = self.mul(x, &one_p);
        self.mul(&half, &xt)
    }

    /// `out = x * y` element-wise.
    pub fn mul(&self, a: &MpsTensor, b: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("mul");
            let t: *mut Object = msg_send![self.obj,
                multiplicationWithPrimaryTensor: a.obj
                secondaryTensor: b.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// `out = a^b` element-wise.
    pub fn power(&self, a: &MpsTensor, b: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("pow");
            let t: *mut Object = msg_send![self.obj,
                powerWithPrimaryTensor: a.obj
                secondaryTensor: b.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// `tanh(x)`.
    pub fn tanh(&self, x: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("tanh");
            let t: *mut Object = msg_send![self.obj,
                tanhWithTensor: x.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// `sigmoid(x) = 1 / (1 + exp(-x))`.
    pub fn sigmoid(&self, x: &MpsTensor) -> MpsTensor {
        unsafe {
            let nsname = ns_string("sigmoid");
            let t: *mut Object = msg_send![self.obj,
                sigmoidWithTensor: x.obj
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// SiLU / Swish activation: `silu(x) = x * sigmoid(x)`.
    /// Used in SwiGLU FFN blocks (Nomic-Vision, LLaMA-style models).
    pub fn silu(&self, x: &MpsTensor) -> MpsTensor {
        let s = self.sigmoid(x);
        self.mul(x, &s)
    }

    /// Scalar constant tensor (broadcasts to operands).
    pub fn constant_scalar(&self, v: f32) -> MpsTensor {
        unsafe {
            let t: *mut Object = msg_send![self.obj,
                constantWithScalar: v as f64
                dataType: mps_dtype::Float32];
            MpsTensor { obj: t }
        }
    }

    /// Layer normalization across `axes` with learnable `gamma`, `beta`.
    pub fn layer_norm(
        &self,
        x: &MpsTensor,
        gamma: &MpsTensor,
        beta: &MpsTensor,
        axes: &[i32],
        eps: f32,
    ) -> MpsTensor {
        unsafe {
            let nsname = ns_string("ln");
            let nsaxes = ns_array_of_i32(axes);
            // -[MPSGraph normalizationWithTensor:meanTensor:varianceTensor:gammaTensor:betaTensor:epsilon:name:]
            // Compute mean / variance over `axes` ourselves.
            let mean: *mut Object = msg_send![self.obj,
                meanOfTensor: x.obj
                axes: nsaxes
                name: ns_string("mean")];
            let var: *mut Object = msg_send![self.obj,
                varianceOfTensor: x.obj
                axes: nsaxes
                name: ns_string("var")];
            let t: *mut Object = msg_send![self.obj,
                normalizationWithTensor: x.obj
                meanTensor: mean
                varianceTensor: var
                gammaTensor: gamma.obj
                betaTensor: beta.obj
                epsilon: eps
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// RMS normalization: `out = (x / sqrt(mean(x², axes) + eps)) · γ + β`.
    ///
    /// Implemented via MPSGraph's optimized
    /// `normalizationWithTensor:mean:variance:gamma:beta:epsilon:` by
    /// feeding `mean = 0` and `variance = mean(x · x, axes)` — the
    /// builtin formula `(x − mean) / √(var + eps) · γ + β` then
    /// collapses exactly to RMSNorm. Reuses Apple's fused norm kernel
    /// instead of building rsqrt by hand.
    pub fn rms_norm(
        &self,
        x: &MpsTensor,
        gamma: &MpsTensor,
        beta: &MpsTensor,
        axes: &[i32],
        eps: f32,
    ) -> MpsTensor {
        unsafe {
            let nsaxes = ns_array_of_i32(axes);
            // x_sq = x · x
            let x_sq: *mut Object = msg_send![self.obj,
                multiplicationWithPrimaryTensor: x.obj
                secondaryTensor: x.obj
                name: ns_string("rms_sq")];
            // var = mean(x²) over the reduction axes (keepDims default).
            let mean_sq: *mut Object = msg_send![self.obj,
                meanOfTensor: x_sq
                axes: nsaxes
                name: ns_string("rms_mean")];
            // mean = 0 (scalar broadcasts across all dims).
            let zero: *mut Object = msg_send![self.obj,
                constantWithScalar: 0.0_f64
                dataType: mps_dtype::Float32];
            let t: *mut Object = msg_send![self.obj,
                normalizationWithTensor: x.obj
                meanTensor: zero
                varianceTensor: mean_sq
                gammaTensor: gamma.obj
                betaTensor: beta.obj
                epsilon: eps
                name: ns_string("rmsnorm")];
            MpsTensor { obj: t }
        }
    }

    /// Softmax along the given axis.
    pub fn softmax(&self, x: &MpsTensor, axis: i32) -> MpsTensor {
        unsafe {
            let nsname = ns_string("softmax");
            let t: *mut Object = msg_send![self.obj,
                softMaxWithTensor: x.obj
                axis: axis as i64
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, x: &MpsTensor, dim_a: usize, dim_b: usize) -> MpsTensor {
        unsafe {
            let nsname = ns_string("transpose");
            let t: *mut Object = msg_send![self.obj,
                transposeTensor: x.obj
                dimension: dim_a as u64
                withDimension: dim_b as u64
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Reshape to the given shape (strict — total element count must match).
    pub fn reshape(&self, x: &MpsTensor, shape: &[usize]) -> MpsTensor {
        unsafe {
            let nsshape = ns_array_of_numbers(shape);
            let nsname = ns_string("reshape");
            let t: *mut Object = msg_send![self.obj,
                reshapeTensor: x.obj
                withShape: nsshape
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Gather rows of `table` indexed by `indices`. `axis` must be 0 in
    /// our usage (embedding lookup): out\[i\] = table[indices\[i\]].
    pub fn gather(&self, table: &MpsTensor, indices: &MpsTensor, axis: u64) -> MpsTensor {
        unsafe {
            let nsname = ns_string("gather");
            let t: *mut Object = msg_send![self.obj,
                gatherWithUpdatesTensor: table.obj
                indicesTensor: indices.obj
                axis: axis
                batchDimensions: 0u64
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Cast tensor to a new dtype.
    pub fn cast(&self, x: &MpsTensor, to_dtype: u32) -> MpsTensor {
        unsafe {
            let nsname = ns_string("cast");
            let t: *mut Object = msg_send![self.obj,
                castTensor: x.obj
                toType: to_dtype
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Slice a single contiguous range along `axis`: out = x[..., start..start+len, ...]
    pub fn slice(&self, x: &MpsTensor, axis: u64, start: i64, len: i64) -> MpsTensor {
        unsafe {
            let nsname = ns_string("slice");
            let t: *mut Object = msg_send![self.obj,
                sliceTensor: x.obj
                dimension: axis
                start: start
                length: len
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Build a constant tensor from raw bytes (caller owns the data; MPSGraph
    /// retains internally). Useful for baking small literals like masks.
    pub fn constant_from_bytes(&self, data: &[u8], shape: &[usize], dtype: u32) -> MpsTensor {
        unsafe {
            let nsshape = ns_array_of_numbers(shape);
            // Build NSData wrapping a copy of the bytes (so caller's buffer
            // can be freed after this returns).
            let cls = class!(NSData);
            let nsdata: *mut Object = msg_send![cls,
                dataWithBytes: data.as_ptr()
                length: data.len() as u64];
            let t: *mut Object = msg_send![self.obj,
                constantWithData: nsdata
                shape: nsshape
                dataType: dtype];
            MpsTensor { obj: t }
        }
    }

    /// Multi-head scaled dot-product attention.
    ///
    /// Inputs are flat `[B, S, NH*DH]` tensors. `mask` is `[B, S]` in the
    /// HuggingFace convention (1.0 = valid, 0.0 = padding). Internally we
    /// convert it to additive form `(mask − 1) · 1e10` so masked positions
    /// receive `−1e10` and valid positions receive `0` — matching the
    /// CPU `Op::Attention` thunk's mask semantics exactly.
    ///
    /// Returns `[B, seq_q, NH*DH]`.
    ///
    /// `mask` is `[B, seq_kv]` (1.0 = valid key position). Supports decode
    /// when `seq_q != seq_kv` (e.g. one new query token over a longer KV cache).
    pub fn attention(
        &self,
        q: &MpsTensor,
        k: &MpsTensor,
        v: &MpsTensor,
        mask: &MpsTensor,
        batch: usize,
        seq_q: usize,
        seq_kv: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> MpsTensor {
        // KNOWN BUG (PLAN: MPSGraph attention parity). When Q/K/V are
        // slice-views of a computed parent (e.g., narrows of a
        // fused-MatMul QKV output, the BERT pattern), MPSGraph's
        // 4D matmul on the reshape→transpose result returns wrong
        // values — 100% relative error scaling with input magnitude.
        // Reproducer: `rlx-runtime/tests/mps_attention_parity.rs::
        // bisect_full_qkv_to_attention` (failing) vs
        // `bisect_three_narrows_to_attention` (host-fed Q/K/V, OK).
        //
        // Bisect-determined facts:
        //   - mm+bias alone: OK (1e-7)
        //   - mm+bias+narrow: OK (1e-7)
        //   - mm+bias+3 narrows: OK (1e-8)
        //   - mm+bias+narrow+reshape: OK (1e-9)
        //   - mm+bias+3 narrows+3 reshapes: OK (1e-7)
        //   - 3 narrows of input + attention: OK (1e-7)
        //   - 3 narrows of computed + attention: 100% rel err
        //
        // Tried fixes that MPSGraph optimizes away:
        //   - mul-by-1, reshape-to-self, single-input concat,
        //     mask-magnitude tweaks, per-head loop, 3-way concat-of-
        //     per-head-slices to force fresh buffers
        //
        // Real fix paths (each multi-day):
        //   1. Schedule-split: compile attention as its own MPSGraph
        //      with Q/K/V as fresh placeholders. Bypasses the
        //      slice-of-computed optimizer pattern entirely.
        //   2. Switch to MPSGraph's `scaledDotProductAttention`
        //      builtin (macOS 14.4+; needs shim addition + version
        //      probe). Apple's implementation likely sidesteps the
        //      bug.
        //   3. Re-emit narrow as a strided-copy thunk that
        //      materializes into a separate Metal buffer before
        //      MPSGraph attention sees it. Hybrid path — adds
        //      thunk dispatch cost but unblocks attention's MPSGraph
        //      win.
        //
        // Tried + DOCUMENTED: Apple's `scaledDotProductAttention`
        // builtin (macOS 14.4+) hits the SAME bug — the issue isn't
        // in our hand-rolled matmul chain, it's in MPSGraph's
        // optimizer treating slice-views of computed tensors as
        // memory aliases. Verified by tracing
        // `RLX_MPSG_TRACE=1 RLX_USE_MPSGRAPH=1 RLX_MPSGRAPH_ATTENTION=1`:
        // the builtin IS invoked but returns the same wrong values.
        // Cast-to-self / split / per-head loop / single-input
        // concat all get optimized away.
        //
        // Real fix is schedule-splitting (fix path #1): compile
        // attention as its own MPSGraph with Q/K/V as fresh
        // placeholders. Bypasses the slice-of-computed pattern
        // entirely. Multi-day refactor — deferred.
        let r4 = |t: &MpsTensor, seq: usize| {
            let r = self.reshape(t, &[batch, seq, num_heads, head_dim]);
            self.transpose(&r, 1, 2) // (0,1,2,3) → (0,2,1,3)
        };
        let q4 = r4(q, seq_q);
        let k4 = r4(k, seq_kv);
        let v4 = r4(v, seq_kv);

        // K^T over last two axes: [B, NH, seq_kv, DH] → [B, NH, DH, seq_kv]
        let k4_t = self.transpose(&k4, 2, 3);

        // scores = Q @ K^T → [B, NH, seq_q, seq_kv]
        let scores = self.matmul(&q4, &k4_t);

        // Scale by 1/sqrt(d_h)
        let scale = self.constant_scalar((head_dim as f32).sqrt().recip());
        let scores = self.mul(&scores, &scale);

        // Additive mask: (mask - 1) * 1e9 → 0 for valid, -1e9 for pad.
        let mask_bc = self.reshape(mask, &[batch, 1, seq_q, seq_kv]);
        let neg_one = self.constant_scalar(-1.0);
        let large_pos = self.constant_scalar(1.0e9);
        let mask_minus = self.add(&mask_bc, &neg_one);
        let mask_additive = self.mul(&mask_minus, &large_pos);
        let scores = self.add(&scores, &mask_additive);

        // softmax along last axis
        let weights = self.softmax(&scores, 4 - 1);

        // out = weights @ V → [B, NH, seq_q, DH]
        let out4 = self.matmul(&weights, &v4);
        // Transpose back to [B, seq_q, NH, DH] then reshape to [B, seq_q, NH*DH].
        let out_perm = self.transpose(&out4, 1, 2);
        self.reshape(&out_perm, &[batch, seq_q, num_heads * head_dim])
    }

    /// Unmasked multi-head SDPA. Supports cross-attention when `seq_kv != seq_q`.
    ///
    /// Inputs are flat `[B, S, NH·DH]` tensors. No mask tensor is read.
    /// Returns `[B, seq_q, NH·DH]`.
    pub fn attention_unmasked(
        &self,
        q: &MpsTensor,
        k: &MpsTensor,
        v: &MpsTensor,
        batch: usize,
        seq_q: usize,
        seq_kv: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> MpsTensor {
        let r4 = |t: &MpsTensor, seq: usize| {
            let r = self.reshape(t, &[batch, seq, num_heads, head_dim]);
            self.transpose(&r, 1, 2) // [B, S, NH, DH] → [B, NH, S, DH]
        };
        let q4 = r4(q, seq_q);
        let k4 = r4(k, seq_kv);
        let v4 = r4(v, seq_kv);

        let k4_t = self.transpose(&k4, 2, 3);
        let scores = self.matmul(&q4, &k4_t);
        let scale = self.constant_scalar((head_dim as f32).sqrt().recip());
        let scores = self.mul(&scores, &scale);
        let weights = self.softmax(&scores, 3);
        let out4 = self.matmul(&weights, &v4);
        let out_perm = self.transpose(&out4, 1, 2);
        self.reshape(&out_perm, &[batch, seq_q, num_heads * head_dim])
    }

    /// Multi-head SDPA with a causal mask baked in as a graph constant.
    ///
    /// Inputs are `[B, S, NH·DH]` flat tensors (qwen3 layout — after
    /// GQA-repeat K/V already have NH heads). The causal mask is an
    /// upper-triangular additive bias: position i attends to keys
    /// 0..=i, future positions get `−∞` (here `−1e9`). Mask is built
    /// as a `[S, S]` constant inside the graph so it broadcasts across
    /// batch and head dims for free.
    ///
    /// Distinct from `attention` — that one takes a per-batch token
    /// mask and folds it into scores via `(mask−1)·1e9`. The causal
    /// variant skips the mask placeholder, which both saves one input
    /// binding and side-steps the slice-of-computed MPSGraph optimizer
    /// bug that hits the masked path on some Q/K/V layouts.
    pub fn attention_causal(
        &self,
        q: &MpsTensor,
        k: &MpsTensor,
        v: &MpsTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> MpsTensor {
        let r4 = |t: &MpsTensor| {
            let r = self.reshape(t, &[batch, seq, num_heads, head_dim]);
            self.transpose(&r, 1, 2) // [B, S, NH, DH] → [B, NH, S, DH]
        };
        let q4 = r4(q);
        let k4 = r4(k);
        let v4 = r4(v);

        // Causal mask: additive bias of shape [1, 1, S, S] — `−∞` on
        // entries (i, j) where j > i so future positions get zeroed by
        // softmax. Built as a graph constant; broadcast over batch/head.
        let mut mask_bytes = Vec::<u8>::with_capacity(seq * seq * 4);
        for i in 0..seq {
            for j in 0..seq {
                let v: f32 = if j > i { -1.0e9 } else { 0.0 };
                mask_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mask = self.constant_from_bytes(&mask_bytes, &[1, 1, seq, seq], mps_dtype::Float32);
        let scale = (head_dim as f32).sqrt().recip();

        // Prefer Apple's fused SDPA when available (macOS 14.4+) —
        // single Metal command vs the 5-op chain otherwise. Falls back
        // to the hand-rolled chain on older OSes for parity.
        let out4 = match self.scaled_dot_product_attention(&q4, &k4, &v4, &mask, scale) {
            Some(t) => t,
            None => {
                let k4_t = self.transpose(&k4, 2, 3); // [B, NH, DH, S]
                let scores = self.matmul(&q4, &k4_t); // [B, NH, S, S]
                let scale_t = self.constant_scalar(scale);
                let scores = self.mul(&scores, &scale_t);
                let scores = self.add(&scores, &mask);
                let weights = self.softmax(&scores, 3);
                self.matmul(&weights, &v4)
            }
        };

        let out_perm = self.transpose(&out4, 1, 2);
        self.reshape(&out_perm, &[batch, seq, num_heads * head_dim])
    }

    /// Apply rotary position embedding.
    ///
    /// `x`: [B, S, NH*DH] where DH must be even. `cos`, `sin`: [S, DH/2]
    /// (precomputed). When `n_rot < head_dim`, only the first `n_rot` dims
    /// per head are rotated (Qwen3.5 uses partial RoPE); the tail is copied.
    pub fn rope(
        &self,
        x: &MpsTensor,
        cos_t: &MpsTensor,
        sin_t: &MpsTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
        n_rot: usize,
    ) -> MpsTensor {
        let rot_half = n_rot / 2;
        let r = self.reshape(x, &[batch, seq, num_heads, head_dim]);

        let cos_sliced = self.slice(cos_t, 0, 0, seq as i64);
        let sin_sliced = self.slice(sin_t, 0, 0, seq as i64);
        let cos_rot = self.slice(&cos_sliced, 1, 0, rot_half as i64);
        let sin_rot = self.slice(&sin_sliced, 1, 0, rot_half as i64);
        let cos_bc = self.reshape(&cos_rot, &[1, seq, 1, rot_half]);
        let sin_bc = self.reshape(&sin_rot, &[1, seq, 1, rot_half]);

        let (rot_block, tail) = if n_rot < head_dim {
            let rot = self.slice(&r, 3, 0, n_rot as i64);
            let tail = self.slice(&r, 3, n_rot as i64, (head_dim - n_rot) as i64);
            (rot, Some(tail))
        } else {
            (r, None)
        };
        let x1 = self.slice(&rot_block, 3, 0, rot_half as i64);
        let x2 = self.slice(&rot_block, 3, rot_half as i64, rot_half as i64);

        // rotated x1 = x1*cos - x2*sin
        let a = self.mul(&x1, &cos_bc);
        let b = self.mul(&x2, &sin_bc);
        let neg_one = self.constant_scalar(-1.0);
        let neg_b = self.mul(&b, &neg_one);
        let rx1 = self.add(&a, &neg_b);
        // rotated x2 = x1*sin + x2*cos
        let c = self.mul(&x1, &sin_bc);
        let d = self.mul(&x2, &cos_bc);
        let rx2 = self.add(&c, &d);

        let rotated = self.concat(&[&rx1, &rx2], 3);
        let cat = if let Some(tail) = tail {
            self.concat(&[&rotated, &tail], 3)
        } else {
            rotated
        };
        self.reshape(&cat, &[batch, seq, num_heads * head_dim])
    }

    /// Concatenate tensors along `axis`.
    /// Split a tensor along an axis into `n` equal pieces. Returns
    /// `Some(Vec)` when the OS supports the call (macOS 12+);
    /// `None` otherwise. Unlike a chain of `slice` calls, MPSGraph
    /// treats `split` as producing N independent tensors which
    /// (per Apple's own kernel for fused-QKV BERT patterns) avoids
    /// the slice-of-computed optimizer bug that hits us in
    /// `attention()` when Q/K/V come from narrows of a fused matmul.
    pub fn split_equal(
        &self,
        x: &MpsTensor,
        num_splits: usize,
        axis: i32,
    ) -> Option<Vec<MpsTensor>> {
        unsafe {
            let nsname = ns_string("split");
            let sel = sel!(splitTensor:numSplits:axis:name:);
            let responds: bool = msg_send![self.obj, respondsToSelector: sel];
            if !responds {
                return None;
            }
            let arr: *mut Object = msg_send![self.obj,
                splitTensor: x.obj
                numSplits: num_splits as u64
                axis: axis as i64
                name: nsname];
            if arr.is_null() {
                return None;
            }
            let count: u64 = msg_send![arr, count];
            let mut out = Vec::with_capacity(count as usize);
            for i in 0..count {
                let t: *mut Object = msg_send![arr, objectAtIndex: i];
                out.push(MpsTensor { obj: t });
            }
            Some(out)
        }
    }

    /// Apple's MPSGraph builtin scaled-dot-product attention
    /// (`scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:`).
    /// Available macOS 14.4+ / iOS 17.4+. Q, K, V are `[..., S, DH]`
    /// (any number of leading batch dims); mask is broadcast against
    /// the per-head scores `[..., S, S]`. Output matches Q's shape.
    ///
    /// The selector returns nil on older OS — caller must probe via
    /// `mps_graph_supports_sdpa()` before calling. Apple's
    /// implementation is a single fused kernel that bypasses the
    /// optimizer pattern responsible for the parity bug in our
    /// hand-rolled `attention()` chain.
    pub fn scaled_dot_product_attention(
        &self,
        q: &MpsTensor,
        k: &MpsTensor,
        v: &MpsTensor,
        mask: &MpsTensor,
        scale: f32,
    ) -> Option<MpsTensor> {
        unsafe {
            let nsname = ns_string("sdpa");
            let sel = sel!(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:);
            let responds: bool = msg_send![self.obj, respondsToSelector: sel];
            if !responds {
                return None;
            }
            let t: *mut Object = msg_send![self.obj,
                scaledDotProductAttentionWithQueryTensor: q.obj
                keyTensor: k.obj
                valueTensor: v.obj
                maskTensor: mask.obj
                scale: scale
                name: nsname];
            if t.is_null() {
                return None;
            }
            Some(MpsTensor { obj: t })
        }
    }

    pub fn concat(&self, tensors: &[&MpsTensor], axis: i32) -> MpsTensor {
        unsafe {
            let arr_cls = class!(NSMutableArray);
            let arr: *mut Object = msg_send![arr_cls, array];
            for t in tensors {
                let _: () = msg_send![arr, addObject: t.obj];
            }
            let nsname = ns_string("concat");
            let t: *mut Object = msg_send![self.obj,
                concatTensors: arr
                dimension: axis as i64
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// 2D convolution (NHWC). Used for ViT patch embeddings; one Conv2D
    /// dispatch replaces the many small matmuls our hand-rolled path
    /// currently generates for the patch projection.
    ///
    /// `weights` shape: [out_ch, kh, kw, in_ch] (HWIO ordering inside kernel).
    /// `stride` / `padding` are (h, w).
    pub fn conv2d(
        &self,
        source: &MpsTensor,
        weights: &MpsTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MpsTensor {
        unsafe {
            // Build MPSGraphConvolution2DOpDescriptor
            let cls = class!(MPSGraphConvolution2DOpDescriptor);
            let desc: *mut Object = msg_send![cls,
                descriptorWithStrideInX: stride.1 as u64
                strideInY: stride.0 as u64
                dilationRateInX: 1u64
                dilationRateInY: 1u64
                groups: 1u64
                paddingLeft: padding.1 as u64
                paddingRight: padding.1 as u64
                paddingTop: padding.0 as u64
                paddingBottom: padding.0 as u64
                paddingStyle: 0i64 // MPSGraphPaddingStyleExplicit
                dataLayout: 1i64    // MPSGraphTensorNamedDataLayoutNHWC = 1
                weightsLayout: 2i64 // MPSGraphTensorNamedDataLayoutOIHW = 2
            ];

            let nsname = ns_string("conv2d");
            let t: *mut Object = msg_send![self.obj,
                convolution2DWithSourceTensor: source.obj
                weightsTensor: weights.obj
                descriptor: desc
                name: nsname];
            MpsTensor { obj: t }
        }
    }

    /// Run the graph synchronously with `feeds` (one MPSGraphTensorData per
    /// placeholder) and read `target` outputs. Outputs are written into the
    /// caller-provided MTLBuffer slots as a side effect (zero-copy when
    /// `MPSGraphTensorData` aliases an MTLBuffer slice).
    ///
    /// This is the JIT path — MPSGraph compiles the subgraph reachable from
    /// `targets` on first call and caches internally. Compile-once / run-many
    /// is wired separately via `compile()` once we have the compiled-executable
    /// path stable.
    pub fn run_jit(
        &self,
        cmd_queue: &metal::CommandQueueRef,
        feed_tensors: &[&MpsTensor],
        feed_buffers: &[&Buffer],
        feed_offsets: &[usize],
        feed_shapes: &[Vec<usize>],
        feed_dtypes: &[u32],
        target_tensors: &[&MpsTensor],
        result_buffers: &[&Buffer],
        result_offsets: &[usize],
        result_shapes: &[Vec<usize>],
        result_dtypes: &[u32],
    ) {
        unsafe {
            let trace = rlx_ir::env::flag("RLX_MPSG_TRACE");
            if trace {
                eprintln!("[mpsg] feed dict");
            }
            let dict_cls = class!(NSMutableDictionary);
            let feeds: *mut Object = msg_send![dict_cls, dictionary];
            for (((tensor, &buf), &off), (shape, &dtype)) in feed_tensors
                .iter()
                .zip(feed_buffers)
                .zip(feed_offsets)
                .zip(feed_shapes.iter().zip(feed_dtypes))
            {
                if trace {
                    eprintln!("[mpsg] feed td shape={:?}", shape);
                }
                let td = mps_tensor_data_from_buffer(buf, off, shape, dtype);
                let _: () = msg_send![feeds, setObject: td forKey: tensor.obj];
            }

            if trace {
                eprintln!("[mpsg] result dict");
            }
            let results_dict: *mut Object = msg_send![dict_cls, dictionary];
            for (((tensor, &buf), &off), (shape, &dtype)) in target_tensors
                .iter()
                .zip(result_buffers)
                .zip(result_offsets)
                .zip(result_shapes.iter().zip(result_dtypes))
            {
                if trace {
                    eprintln!("[mpsg] result td shape={:?}", shape);
                }
                let td = mps_tensor_data_from_buffer(buf, off, shape, dtype);
                let _: () = msg_send![results_dict, setObject: td forKey: tensor.obj];
            }

            if trace {
                eprintln!("[mpsg] run");
            }
            let _: () = msg_send![self.obj,
                runWithMTLCommandQueue: cmd_queue
                feeds: feeds
                targetOperations: std::ptr::null::<Object>()
                resultsDictionary: results_dict];
            if trace {
                eprintln!("[mpsg] run done");
            }
        }
    }

    /// Pre-compile the graph into an `MPSGraphExecutable` keyed on the
    /// given placeholder shapes. The executable holds a frozen plan;
    /// every subsequent encode skips the JIT analysis pass and binds
    /// inputs by position rather than by dict-key NSObject lookup.
    ///
    /// `feed_tensors` are the placeholder tensors in the order that
    /// `encode_executable` will later pass the matching inputs array.
    /// Likewise `target_tensors` fixes the output order.
    ///
    /// Returns `None` on older OSes that don't expose
    /// `compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:`
    /// (caller falls back to `run_jit`).
    pub fn compile_executable(
        &self,
        feed_tensors: &[&MpsTensor],
        feed_shapes: &[Vec<usize>],
        feed_dtypes: &[u32],
        target_tensors: &[&MpsTensor],
    ) -> Option<MpsGraphExecutable> {
        unsafe {
            let dev = crate::device::metal_device()?;
            let sel =
                sel!(compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:);
            let responds: bool = msg_send![self.obj, respondsToSelector: sel];
            if !responds {
                return None;
            }

            // feeds: NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>
            let dict_cls = class!(NSMutableDictionary);
            let feeds: *mut Object = msg_send![dict_cls, dictionary];
            let shaped_cls = class!(MPSGraphShapedType);
            for ((tensor, shape), &dtype) in feed_tensors.iter().zip(feed_shapes).zip(feed_dtypes) {
                let nsshape = ns_array_of_numbers(shape);
                let alloc: *mut Object = msg_send![shaped_cls, alloc];
                let shaped: *mut Object = msg_send![alloc, initWithShape: nsshape dataType: dtype];
                let _: () = msg_send![feeds, setObject: shaped forKey: tensor.obj];
                let _: () = msg_send![shaped, release];
            }

            // targets: NSArray<MPSGraphTensor*>
            let arr_cls = class!(NSMutableArray);
            let targets: *mut Object = msg_send![arr_cls, array];
            for t in target_tensors {
                let _: () = msg_send![targets, addObject: t.obj];
            }

            // MPSGraphCompilationDescriptor (nil ⇒ defaults).
            // The "device" arg is the underlying MTLDevice ObjC pointer;
            // metal::Device wraps it but msg_send needs the raw `id`.
            // MPSGraph wants an MPSGraphDevice (a thin wrapper around the
            // MTLDevice) — build via +[MPSGraphDevice deviceWithMTLDevice:].
            let dev_cls = class!(MPSGraphDevice);
            let mtl_dev_ptr: *mut Object = dev.device.as_ptr().cast();
            let mpsg_dev: *mut Object = msg_send![dev_cls,
                deviceWithMTLDevice: mtl_dev_ptr];
            let exec: *mut Object = msg_send![self.obj,
                compileWithDevice: mpsg_dev
                feeds: feeds
                targetTensors: targets
                targetOperations: std::ptr::null::<Object>()
                compilationDescriptor: std::ptr::null::<Object>()];
            if exec.is_null() {
                return None;
            }
            let _: *mut Object = msg_send![exec, retain];

            // Resolve the permutation: executable stores its own feed /
            // target ordering; recover it by reading the
            // `feedTensors` / `targetTensors` properties and matching
            // each entry back to our caller-supplied slices by ObjC
            // pointer identity (the underlying MPSGraphTensor pointers
            // are stable across compile).
            let exec_feeds: *mut Object = msg_send![exec, feedTensors];
            let exec_targets: *mut Object = msg_send![exec, targetTensors];
            let perm = |arr: *mut Object, expected: &[&MpsTensor]| -> Option<Vec<usize>> {
                if arr.is_null() {
                    // No reorder needed if the executable doesn't
                    // expose its ordering (very old OS).
                    return Some((0..expected.len()).collect());
                }
                let count: u64 = msg_send![arr, count];
                let mut out = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let t: *mut Object = msg_send![arr, objectAtIndex: i];
                    let idx = expected.iter().position(|x| x.obj == t)?;
                    out.push(idx);
                }
                Some(out)
            };
            let feed_perm = perm(exec_feeds, feed_tensors)?;
            let result_perm = perm(exec_targets, target_tensors)?;

            Some(MpsGraphExecutable {
                obj: exec,
                feed_perm,
                result_perm,
                inputs_cache: std::ptr::null_mut(),
                results_cache: std::ptr::null_mut(),
            })
        }
    }
}

/// Compiled, parameter-cached MPSGraph. Drop releases the underlying
/// `MPSGraphExecutable`. The compile happens once; per-call work is
/// bounded by input + output binding + a single
/// `encodeToCommandBuffer:` dispatch.
///
/// `feed_perm[i]` is the index into the caller's compile-time
/// `feed_tensors` slice whose data should appear at executable
/// input position `i`. MPSGraphExecutable internally reorders feeds
/// (the compile `feeds:` argument is a dict, so order is not
/// preserved); we read back `executable.feedTensors` to recover the
/// expected order. Same idea for `result_perm` against the
/// compile-time `target_tensors`.
pub struct MpsGraphExecutable {
    obj: *mut Object,
    pub feed_perm: Vec<usize>,
    pub result_perm: Vec<usize>,
    /// Pre-built `NSArray<MPSGraphTensorData>` in executable feed
    /// order — populated by `bind_arena`. None ⇒ run() rebuilds per
    /// call. Retained for the lifetime of the executable.
    inputs_cache: *mut Object,
    /// Pre-built `NSArray<MPSGraphTensorData>` in executable target
    /// order — populated by `bind_arena`. Retained.
    results_cache: *mut Object,
}
unsafe impl Send for MpsGraphExecutable {}
unsafe impl Sync for MpsGraphExecutable {}

impl Drop for MpsGraphExecutable {
    fn drop(&mut self) {
        unsafe {
            if !self.inputs_cache.is_null() {
                let _: () = msg_send![self.inputs_cache, release];
            }
            if !self.results_cache.is_null() {
                let _: () = msg_send![self.results_cache, release];
            }
            if !self.obj.is_null() {
                let _: () = msg_send![self.obj, release];
            }
        }
    }
}

impl MpsGraphExecutable {
    /// Build (and retain) the input + result `NSArray<MPSGraphTensorData>`
    /// once, reusable across every subsequent run. Caller provides the
    /// same positional slices that `compile_executable` was given —
    /// the permutation captured at compile is applied here so each
    /// cached array matches what the executable expects.
    ///
    /// Safe to call after this because:
    ///   * arena buffer pointer is fixed for the lifetime of the
    ///     compiled module (allocated once, never moved),
    ///   * each input/param/output's byte offset within the arena is
    ///     computed at compile time and never changes,
    ///   * `MPSGraphTensorData` only borrows the buffer view — the
    ///     bytes underneath can be rewritten between runs without
    ///     invalidating the wrapper.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_arena(
        &mut self,
        feed_buffers: &[&Buffer],
        feed_offsets: &[usize],
        feed_shapes: &[Vec<usize>],
        feed_dtypes: &[u32],
        result_buffers: &[&Buffer],
        result_offsets: &[usize],
        result_shapes: &[Vec<usize>],
        result_dtypes: &[u32],
    ) {
        unsafe {
            // Drop any previously cached arrays — `bind_arena` is
            // re-entrant when callers rebuild the arena (rare; only
            // happens on full recompile).
            if !self.inputs_cache.is_null() {
                let _: () = msg_send![self.inputs_cache, release];
                self.inputs_cache = std::ptr::null_mut();
            }
            if !self.results_cache.is_null() {
                let _: () = msg_send![self.results_cache, release];
                self.results_cache = std::ptr::null_mut();
            }

            let arr_cls = class!(NSMutableArray);
            let inputs: *mut Object = msg_send![arr_cls, array];
            for &i in &self.feed_perm {
                let td = mps_tensor_data_from_buffer(
                    feed_buffers[i],
                    feed_offsets[i],
                    &feed_shapes[i],
                    feed_dtypes[i],
                );
                let _: () = msg_send![inputs, addObject: td];
            }
            let results: *mut Object = msg_send![arr_cls, array];
            for &i in &self.result_perm {
                let td = mps_tensor_data_from_buffer(
                    result_buffers[i],
                    result_offsets[i],
                    &result_shapes[i],
                    result_dtypes[i],
                );
                let _: () = msg_send![results, addObject: td];
            }
            // Retain so the arrays survive past the autorelease pool
            // boundary of this call.
            let _: *mut Object = msg_send![inputs, retain];
            let _: *mut Object = msg_send![results, retain];
            self.inputs_cache = inputs;
            self.results_cache = results;
        }
    }

    /// Cheap-dispatch run using the cached arrays bound via
    /// `bind_arena`. Per-call work is exactly one ObjC message into
    /// MPSGraphExecutable plus the GPU sync — no NSArray builds, no
    /// MPSGraphTensorData allocations.
    pub fn run_cached(&self, cmd_queue: &metal::CommandQueueRef) {
        debug_assert!(
            !self.inputs_cache.is_null() && !self.results_cache.is_null(),
            "run_cached called before bind_arena"
        );
        unsafe {
            let _: () = msg_send![self.obj,
                runWithMTLCommandQueue: cmd_queue
                inputsArray: self.inputs_cache
                resultsArray: self.results_cache
                executionDescriptor: std::ptr::null::<Object>()];
        }
    }

    /// True iff `bind_arena` has been called and the cached arrays
    /// are ready for `run_cached`.
    pub fn has_cached_binding(&self) -> bool {
        !self.inputs_cache.is_null() && !self.results_cache.is_null()
    }

    /// Run the precompiled executable. `feed_buffers` / `feed_offsets`
    /// / `feed_shapes` / `feed_dtypes` are positional — slot `i` must
    /// match the `i`th tensor passed to `compile_executable`. Outputs
    /// land in the caller-provided MTLBuffer slots (zero-copy on
    /// Apple Silicon unified memory).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        cmd_queue: &metal::CommandQueueRef,
        feed_buffers: &[&Buffer],
        feed_offsets: &[usize],
        feed_shapes: &[Vec<usize>],
        feed_dtypes: &[u32],
        result_buffers: &[&Buffer],
        result_offsets: &[usize],
        result_shapes: &[Vec<usize>],
        result_dtypes: &[u32],
    ) {
        unsafe {
            let arr_cls = class!(NSMutableArray);
            let inputs: *mut Object = msg_send![arr_cls, array];
            for &i in &self.feed_perm {
                let td = mps_tensor_data_from_buffer(
                    feed_buffers[i],
                    feed_offsets[i],
                    &feed_shapes[i],
                    feed_dtypes[i],
                );
                let _: () = msg_send![inputs, addObject: td];
            }
            let results: *mut Object = msg_send![arr_cls, array];
            for &i in &self.result_perm {
                let td = mps_tensor_data_from_buffer(
                    result_buffers[i],
                    result_offsets[i],
                    &result_shapes[i],
                    result_dtypes[i],
                );
                let _: () = msg_send![results, addObject: td];
            }

            // `runWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:`
            // is the synchronous executable path. Equivalent to
            // `encodeToCommandBuffer:` + commit + wait but avoids the
            // command-buffer lifecycle our caller would otherwise need
            // to plumb. ExecutionDescriptor = nil ⇒ defaults.
            let _: () = msg_send![self.obj,
                runWithMTLCommandQueue: cmd_queue
                inputsArray: inputs
                resultsArray: results
                executionDescriptor: std::ptr::null::<Object>()];
        }
    }
}

/// Symbolic tensor reference owned by an MPSGraph (no Drop — graph owns it).
pub struct MpsTensor {
    obj: *mut Object,
}

// ── Helper objc constructors ───────────────────────────────────────────

unsafe fn ns_string(s: &str) -> *mut Object {
    let cls = class!(NSString);
    let bytes = s.as_bytes();
    let alloc: *mut Object = msg_send![cls, alloc];
    msg_send![alloc,
        initWithBytes: bytes.as_ptr()
        length: bytes.len()
        encoding: 4u64 /* NSUTF8StringEncoding */]
}

unsafe fn ns_number(n: usize) -> *mut Object {
    let cls = class!(NSNumber);
    msg_send![cls, numberWithUnsignedLongLong: n as u64]
}

unsafe fn ns_array_of_numbers(dims: &[usize]) -> *mut Object {
    unsafe {
        let arr_cls = class!(NSMutableArray);
        let arr: *mut Object = msg_send![arr_cls, array];
        for &d in dims {
            let n = ns_number(d);
            let _: () = msg_send![arr, addObject: n];
        }
        arr
    }
}

unsafe fn ns_number_i32(v: i32) -> *mut Object {
    let cls = class!(NSNumber);
    msg_send![cls, numberWithInt: v]
}

unsafe fn ns_array_of_i32(values: &[i32]) -> *mut Object {
    unsafe {
        let arr_cls = class!(NSMutableArray);
        let arr: *mut Object = msg_send![arr_cls, array];
        for &v in values {
            let n = ns_number_i32(v);
            let _: () = msg_send![arr, addObject: n];
        }
        arr
    }
}

unsafe fn mps_tensor_data_from_buffer(
    buf: &Buffer,
    offset: usize,
    shape: &[usize],
    dtype: u32,
) -> *mut Object {
    unsafe {
        // MPSGraphTensorData's init takes the whole MTLBuffer with no offset
        // (the deprecated +offset:rowBytes: variants don't exist on all macOS
        // versions). For non-zero offsets we wrap a sub-buffer view via
        // newBufferWithBytesNoCopy: pointing at `contents() + offset`.
        let nsshape = ns_array_of_numbers(shape);
        let cls = class!(MPSGraphTensorData);
        let alloc: *mut Object = msg_send![cls, alloc];

        if offset == 0 {
            let buf_ref: &metal::BufferRef = buf;
            msg_send![alloc,
            initWithMTLBuffer: buf_ref
            shape: nsshape
            dataType: dtype]
        } else {
            // Build a no-copy MTLBuffer view rooted at `contents() + offset`.
            // Apple's MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:
            // takes a CPU pointer + length and treats the underlying memory as
            // GPU-shared. This works on Apple Silicon because of unified memory.
            let n_elem: usize = shape.iter().product();
            let bytes = n_elem * if dtype == mps_dtype::Float16 { 2 } else { 4 };
            let raw_ptr = (buf.contents() as *mut u8).add(offset);
            let dev_ref: &metal::DeviceRef =
                &crate::device::metal_device().expect("metal device").device;
            let view: *mut Object = msg_send![dev_ref,
            newBufferWithBytesNoCopy: raw_ptr
            length: bytes as u64
            options: 0u64 // MTLResourceStorageModeShared
            deallocator: std::ptr::null::<Object>()];
            msg_send![alloc,
            initWithMTLBuffer: view
            shape: nsshape
            dataType: dtype]
        }
    }
}

#[allow(dead_code)]
fn _unused_imports_silencer(_b: BOOL) {
    let _ = NO;
}
