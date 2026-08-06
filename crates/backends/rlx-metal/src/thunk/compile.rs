// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `compile` — extracted from the `thunk` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::Arena;
use crate::op_registry::MetalKernel;
use rlx_ir::NodeId;
use rlx_ir::op::{Activation, BinaryOp, CmpOp};
use rlx_ir::{DType, Graph, Op, Shape};
use std::sync::Arc;

use super::*;

/// "Relu-first" opcode (canonical table in `rlx_ir::opcodes`; `0=relu …
/// 17=reciprocal`) for the acts that have a native Metal backward kernel. The
/// tail (`Floor`…`LogSigmoid`) has no native backward kernel — it decomposes at
/// the AD level — so it must never reach here.
fn activation_backward_op_id(act: Activation) -> u32 {
    match act {
        Activation::Floor
        | Activation::Ceil
        | Activation::Sign
        | Activation::Softplus
        | Activation::Elu
        | Activation::Erf
        | Activation::HardSwish
        | Activation::HardSigmoid
        | Activation::Mish
        | Activation::Softsign
        | Activation::LogSigmoid => {
            panic!("rlx-metal: no native backward for {act:?} (decomposed)")
        }
        _ => act.opcode_relu_first(),
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
        Self::compile_with_rng_fab_weights(
            graph,
            arena,
            rng,
            fab_scratch,
            &std::collections::HashMap::new(),
        )
    }

    /// Like [`Self::compile_with_rng_fab`] with optional large-param offsets in a
    /// separate weight MTLBuffer (values are untagged; this tags them for encode).
    pub fn compile_with_rng_fab_weights(
        graph: &Graph,
        arena: &Arena,
        rng: rlx_ir::RngOptions,
        fab_scratch: &std::collections::HashMap<rlx_ir::NodeId, (usize, usize)>,
        weight_offs: &std::collections::HashMap<rlx_ir::NodeId, usize>,
    ) -> Self {
        let rng_shared = std::sync::Arc::new(std::sync::RwLock::new(rng));
        let mut thunks = Vec::with_capacity(graph.len());

        let off = |id| -> usize {
            if arena.has_buffer(id) {
                arena.byte_offset(id)
            } else if let Some(&w) = weight_offs.get(&id) {
                tag_weight_off(w)
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

        // Group the autodiff's three `AttentionBackward{Query,Key,Value}`
        // siblings (same q,k,v,dy) so the Query node can emit ONE fused
        // `AttentionBackwardAll` (scores/dp/ds recomputed once) and Key/Value
        // become Nop. Keyed on the shared inputs; value = per-wrt node id.
        let attn_bwd_groups: std::collections::HashMap<
            (
                rlx_ir::NodeId,
                rlx_ir::NodeId,
                rlx_ir::NodeId,
                rlx_ir::NodeId,
            ),
            [Option<rlx_ir::NodeId>; 3],
        > = {
            use rlx_ir::op::AttentionBwdWrt;
            let mut g: std::collections::HashMap<_, [Option<rlx_ir::NodeId>; 3]> =
                std::collections::HashMap::new();
            for node in graph.nodes() {
                if let Op::AttentionBackward { wrt, .. } = &node.op {
                    let key = (
                        node.inputs[0],
                        node.inputs[1],
                        node.inputs[2],
                        node.inputs[3],
                    );
                    let idx = match wrt {
                        AttentionBwdWrt::Query => 0,
                        AttentionBwdWrt::Key => 1,
                        AttentionBwdWrt::Value => 2,
                    };
                    g.entry(key).or_insert([None; 3])[idx] = Some(node.id);
                }
            }
            g
        };

        // Fold a materialized last-two-swap `Transpose` on a rank-2 MatMul operand
        // into the GEMM (MPS `transposeLeft`/`transposeRight`), eliminating the
        // transpose's arena buffer and its memory-bound copy. The autodiff VJP
        // emits `dW = Xᵀ·dY` and `dX = dY·Wᵀ`. Mirrors the CPU `matmul_fold`.
        //
        // OPT-IN (default OFF), enable with RLX_METAL_MATMUL_TRANSPOSE_FOLD=1.
        // Measured net-neutral-to-negative on real workloads: the transposes are
        // cheap (~1% of a training step — the custom small-matrix GEMM + a cheap
        // transpose beats routing through MPS, whose per-call overhead exceeds the
        // saved copy). The MPS-only gate below already limits it to shapes where
        // MPS is fastest, yet even there it doesn't win at tested scales. Kept
        // behind the flag as a correct (bit-exact) mechanism for potential
        // large-transposed-matmul cases (e.g. big tied LM heads), not a default.
        // Restricted to f32, arena-resident operands (MPS is single-buffer, so
        // weight-buffer-tagged sources stay materialized). Map: matmul id →
        // (a_src, ta, b_src, tb); `folded_transpose`: transpose ids to Nop.
        let (matmul_fold, folded_transpose): (
            std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
            std::collections::HashSet<NodeId>,
        ) = {
            let mut fold = std::collections::HashMap::new();
            let mut folded = std::collections::HashSet::new();
            let enabled =
                rlx_ir::env::var("RLX_METAL_MATMUL_TRANSPOSE_FOLD").as_deref() == Some("1");
            // Opt-in extension: also fold a `matmul_t(weight)` transpose (e.g. the
            // big tied LM head) by reading the weight buffer directly with MPS
            // transposeRight (`encode_mps_sgemm_t_bufs`), instead of requiring
            // arena-resident operands. Default off; the MPS-shape gate below still
            // applies, so only large weight-transposed matmuls (LM heads) qualify.
            let fold_weights =
                rlx_ir::env::var("RLX_METAL_MATMUL_TRANSPOSE_FOLD_WEIGHTS").as_deref() == Some("1");
            if enabled {
                let mut uses: std::collections::HashMap<NodeId, u32> =
                    std::collections::HashMap::new();
                for node in graph.nodes() {
                    for &inp in &node.inputs {
                        *uses.entry(inp).or_insert(0) += 1;
                    }
                }
                // A rank-2 [1,0]-Transpose used exactly once, whose pre-transpose
                // source is arena-resident → returns that source id.
                let foldable_t = |id: NodeId| -> Option<NodeId> {
                    let n = graph.node(id);
                    let Op::Transpose { perm } = &n.op else {
                        return None;
                    };
                    if perm.as_slice() != [1, 0] || n.shape.rank() != 2 {
                        return None;
                    }
                    if uses.get(&id) != Some(&1) {
                        return None;
                    }
                    let so = off(n.inputs[0]);
                    (so != usize::MAX && (fold_weights || !is_weight_off(so)))
                        .then_some(n.inputs[0])
                };
                for node in graph.nodes() {
                    if !matches!(node.op, Op::MatMul) {
                        continue;
                    }
                    let (a_id, b_id) = (node.inputs[0], node.inputs[1]);
                    if graph.node(a_id).shape.rank() != 2
                        || graph.node(b_id).shape.rank() != 2
                        || node.shape.dtype() != DType::F32
                    {
                        continue;
                    }
                    let co = off(node.id);
                    if co == usize::MAX || is_weight_off(co) {
                        continue; // MPS single-buffer: output must be arena
                    }
                    // Only fold when MPS is already the fastest GEMM for this shape.
                    // The fold routes through MPS (the only transpose-capable path);
                    // for small matmuls the custom kernel wins and MPS's per-call
                    // overhead would exceed the (tiny) transpose cost — measured a
                    // ~3% net loss on small-model training. Large matmuls (e.g. LM
                    // heads) both prefer MPS AND materialize a big transpose, so
                    // there the fold is a clear win. `crate::cost` decides.
                    let n_dim = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let m_dim = node.shape.num_elements().unwrap() / n_dim.max(1);
                    let k_dim = graph.node(a_id).shape.num_elements().unwrap() / m_dim.max(1);
                    if crate::cost::hw_model().pick_sgemm(m_dim, k_dim, n_dim)
                        != crate::cost::SgemmVariant::Mps
                    {
                        continue;
                    }
                    let (fa, fb) = (foldable_t(a_id), foldable_t(b_id));
                    if fa.is_none() && fb.is_none() {
                        continue;
                    }
                    let (asrc, ta) = fa.map_or((a_id, false), |s| (s, true));
                    let (bsrc, tb) = fb.map_or((b_id, false), |s| (s, true));
                    // The non-folded operand must be arena-resident — unless the
                    // weight-fold extension is on, in which case the buffer-aware
                    // encode reads it from the weight buffer.
                    let (ao, bo) = (off(asrc), off(bsrc));
                    if ao == usize::MAX
                        || bo == usize::MAX
                        || (!fold_weights && (is_weight_off(ao) || is_weight_off(bo)))
                    {
                        continue;
                    }
                    fold.insert(node.id, (asrc, ta, bsrc, tb));
                    if ta {
                        folded.insert(a_id);
                    }
                    if tb {
                        folded.insert(b_id);
                    }
                }
            }
            (fold, folded)
        };

        for node in graph.nodes() {
            #[cfg(feature = "native-gpu-fft")]
            if fft_real_skip.contains(&node.id) {
                thunks.push(Thunk::Nop);
                continue;
            }
            // Folded into a downstream MatMul's transposed GEMM — drop the copy.
            if folded_transpose.contains(&node.id) {
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
                        b_f16: false,
                        a_f16: false,
                        ta: false,
                        tb: false,
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
                        b_f16: false,
                        a_f16: false,
                        ta: false,
                        tb: false,
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
                    let b_f16 = matches!(b_shape.dtype(), rlx_ir::DType::F16);
                    let a_f16 = matches!(a_shape.dtype(), rlx_ir::DType::F16);
                    // Any-rank batched matmul: all leading dims (except the
                    // last 2) match between A, B, and output, and the last
                    // 2 dims form [M, K] @ [K, N] = [M, N]. The 2-D Sgemm
                    // flatten trick is wrong when both operands carry
                    // independent batch dims (SAM3 decomposed attention).
                    // Each leading (batch) dim of an operand must equal the output
                    // dim OR be 1 (broadcast). Whole-operand broadcast (batch
                    // product 1) → per-matrix stride 0 (a_bcast/b_bcast).
                    let batched = a_shape.rank() >= 3
                        && b_shape.rank() == a_shape.rank()
                        && shape.rank() == a_shape.rank()
                        && {
                            let mut ok = true;
                            for d in 0..a_shape.rank() - 2 {
                                let (ad, bd, od) = (
                                    a_shape.dim(d).unwrap_static(),
                                    b_shape.dim(d).unwrap_static(),
                                    shape.dim(d).unwrap_static(),
                                );
                                if !((ad == od || ad == 1)
                                    && (bd == od || bd == 1)
                                    && od == ad.max(bd))
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
                        let mut a_batch = 1usize;
                        let mut b_batch = 1usize;
                        for d in 0..r - 2 {
                            batch_prod *= shape.dim(d).unwrap_static();
                            a_batch *= a_shape.dim(d).unwrap_static();
                            b_batch *= b_shape.dim(d).unwrap_static();
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
                            a_bcast: a_batch == 1 && batch_prod > 1,
                            b_bcast: b_batch == 1 && batch_prod > 1,
                        }
                    } else {
                        let n = shape.dim(shape.rank() - 1).unwrap_static();
                        let total = shape.num_elements().unwrap();
                        let m = total / n;
                        let a_total = a_shape.num_elements().unwrap();
                        let k = a_total / m;
                        // Transpose-fold: read the pre-transpose source(s) with an
                        // MPS transpose flag; `m/k/n` are the logical (post-transpose)
                        // dims, unchanged (derived from the output and A's element
                        // count, which the swap preserves).
                        let (a, ta, b, tb) = match matmul_fold.get(&node.id) {
                            Some(&(asrc, ta, bsrc, tb)) => (off(asrc), ta, off(bsrc), tb),
                            None => (off(node.inputs[0]), false, off(node.inputs[1]), false),
                        };
                        Thunk::Sgemm {
                            a,
                            b,
                            c: off(node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            dt: shape.dtype().into(),
                            b_f16,
                            a_f16,
                            ta,
                            tb,
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

                Op::FusedMatMulResidual => {
                    let shape = &node.shape;
                    let n = shape.dim(shape.rank() - 1).unwrap_static();
                    let total = shape.num_elements().unwrap();
                    let m = total / n;
                    let a_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = a_total / m;
                    Thunk::SgemmResidual {
                        a: off(node.inputs[0]),
                        b: off(node.inputs[1]),
                        c: off(node.id),
                        r: off(node.inputs[2]),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        dt: shape.dtype().into(),
                    }
                }

                Op::Cast { to } => {
                    let len = node.shape.num_elements().unwrap();
                    let src_dtype = graph.node(node.inputs[0]).shape.dtype();
                    let dst_dtype = *to;
                    let out_dtype = node.shape.dtype();
                    // A packed sub-4-byte integer SOURCE (a U8/I8 *Param* — e.g. VQ
                    // codebook indices) is stored 1-byte-packed in the f32-uniform
                    // arena, NOT widened. The CastTruncF32 fast path below reads the
                    // source as `*const f32` (4 B/elem), so it would mis-read a
                    // packed u8 param (garbage indices → SynthMatMul decompose/VJP
                    // failure). Route such sources through the generic host cast,
                    // which reads the source at its true width and writes the widened
                    // f32 slot — integer codes have no fractional part, so the f32
                    // result equals the index exactly and downstream Gather/ScatterAdd
                    // (which read f32 indices) stay correct.
                    //
                    // Gate on the source being an `Op::Param` OR `Op::Constant`:
                    // both are stored PACKED at native (sub-4-byte) width. A non-f32
                    // *param* is packed by `plan_memory`'s `node_slot_bytes`
                    // (`non_f32_param`); a U8/I8 *constant* is packed too — the
                    // widen pass (`widen_integer_activations_to_f32`) only widens
                    // I64/I32/U32/Bool (`metal_widened_dtype`), NOT U8/I8, so a u8
                    // constant keeps DType::U8 and its literal bytes are copied 1-B/elem
                    // into the slot start (see the `Op::Constant` init in
                    // `backend/compile.rs`). The forward SynthMatMul kernel reads those
                    // indices as packed u8; the VJP's `Cast(u8→i64)→Gather` must too.
                    // Without Constant here, the CastTruncF32 fast path below read the
                    // 1-byte-packed u8 index constant as 4-B f32 → garbage indices →
                    // the SynthMatMul-backprop Gather reconstructed Wᵀ from wrong rows
                    // (rlx-tiny: layer grads exploded to 1e22 → NaN on Metal only).
                    // (A widened U8/I8 tensor arrives as src_dtype==F32, excluded here;
                    // a Custom-op operand keeps its dtype but its literal bytes are
                    // likewise packed, so the true-width host read stays correct.)
                    let packed_int_src = matches!(src_dtype, DType::U8 | DType::I8)
                        && matches!(
                            graph.node(node.inputs[0]).op,
                            Op::Param { .. } | Op::Constant { .. }
                        );
                    // Widened arena: Cast→int kept `to` as integer for truncation
                    // semantics but the tensor slot is F32. Emit f32 trunc-in-place
                    // so Unsqueeze/Gather/fringe-mask stay on f32 paths.
                    let trunc_to_int = matches!(
                        dst_dtype,
                        DType::I64 | DType::I32 | DType::U32 | DType::Bool
                    ) && out_dtype == DType::F32
                        && !packed_int_src;
                    if trunc_to_int {
                        Thunk::CastTruncF32 {
                            src: off(node.inputs[0]),
                            dst: off(node.id),
                            len: len as u32,
                        }
                    } else if packed_int_src && out_dtype == DType::F32 {
                        // U8/I8 source into a widened f32 slot: read true 1-byte width,
                        // write f32 (identical machinery to the working Cast(u8→f32)).
                        Thunk::CastHost {
                            src: off(node.inputs[0]),
                            dst: off(node.id),
                            len: len as u32,
                            src_dt: src_dtype,
                            dst_dt: DType::F32,
                        }
                    } else {
                        let half_ok = matches!(
                            (src_dtype, dst_dtype),
                            (DType::F32, DType::F32)
                                | (DType::F32, DType::F16)
                                | (DType::F16, DType::F32)
                                | (DType::F16, DType::F16)
                        );
                        if half_ok {
                            Thunk::Cast {
                                src: off(node.inputs[0]),
                                dst: off(node.id),
                                len: len as u32,
                                src_dt: src_dtype.into(),
                                dst_dt: dst_dtype.into(),
                            }
                        } else {
                            Thunk::CastHost {
                                src: off(node.inputs[0]),
                                dst: off(node.id),
                                len: len as u32,
                                src_dt: src_dtype,
                                dst_dt: dst_dtype,
                            }
                        }
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
                        // Out-of-place activation: one pass (read src → write dst)
                        // instead of Copy + in-place (2× act traffic).
                        if matches!(
                            act,
                            Activation::Silu
                                | Activation::Gelu
                                | Activation::GeluApprox
                                | Activation::Relu
                                | Activation::Sigmoid
                        ) && in_dt == dt
                        {
                            Thunk::ActivationOut {
                                src: in_off,
                                dst: out_off,
                                len: len as u32,
                                act: *act,
                                dt,
                            }
                        } else {
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
                    // Collapse all spatial dims (indices ≥ 2) into H, with W = 1,
                    // so rank-2 [N,C], rank-3 [N,C,L] and rank-4 [N,C,H,W] all
                    // normalize over the full (C/G · spatial) extent. GroupNorm is
                    // a flat reduction over the contiguous (C/G, spatial) block, so
                    // the H×W split is irrelevant to the result. Previously dim(2)
                    // and dim(3) were read unconditionally, panicking on rank-3
                    // GroupNorm — which forced callers to pre-reshape to [N,C,1,L].
                    let rank = in_shape.rank();
                    let mut spatial: u32 = 1;
                    for i in 2..rank {
                        spatial *= in_shape.dim(i).unwrap_static() as u32;
                    }
                    Thunk::GroupNorm {
                        src: off(node.inputs[0]),
                        g: off(node.inputs[1]),
                        b: off(node.inputs[2]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: spatial,
                        w: 1,
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

                Op::Conv3d {
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    Thunk::Conv3d {
                        src: off(node.inputs[0]),
                        weight: off(node.inputs[1]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        d: in_shape.dim(2).unwrap_static() as u32,
                        h: in_shape.dim(3).unwrap_static() as u32,
                        w_in: in_shape.dim(4).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        d_out: out_shape.dim(2).unwrap_static() as u32,
                        h_out: out_shape.dim(3).unwrap_static() as u32,
                        w_out: out_shape.dim(4).unwrap_static() as u32,
                        kd: w_shape.dim(2).unwrap_static() as u32,
                        kh: w_shape.dim(3).unwrap_static() as u32,
                        kw: w_shape.dim(4).unwrap_static() as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw: dilation[2] as u32,
                        groups: (*groups).max(1) as u32,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::ConvTranspose3d {
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    Thunk::ConvTranspose3d {
                        src: off(node.inputs[0]),
                        weight: off(node.inputs[1]),
                        dst: off(node.id),
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        d: in_shape.dim(2).unwrap_static() as u32,
                        h: in_shape.dim(3).unwrap_static() as u32,
                        w_in: in_shape.dim(4).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        d_out: out_shape.dim(2).unwrap_static() as u32,
                        h_out: out_shape.dim(3).unwrap_static() as u32,
                        w_out: out_shape.dim(4).unwrap_static() as u32,
                        kd: w_shape.dim(2).unwrap_static() as u32,
                        kh: w_shape.dim(3).unwrap_static() as u32,
                        kw: w_shape.dim(4).unwrap_static() as u32,
                        sd: stride[0] as u32,
                        sh: stride[1] as u32,
                        sw: stride[2] as u32,
                        pd: padding[0] as u32,
                        ph: padding[1] as u32,
                        pw: padding[2] as u32,
                        dd: dilation[0] as u32,
                        dh: dilation[1] as u32,
                        dw: dilation[2] as u32,
                        groups: (*groups).max(1) as u32,
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
                        sum_out: 0,
                    }
                }

                Op::AdaLayerNorm { norm, eps } => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let rows = total / h;
                    let x_dims: Vec<usize> = graph
                        .node(node.inputs[0])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let mod_dims: Vec<usize> = graph
                        .node(node.inputs[1])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Thunk::AdaLayerNorm {
                        x: off(node.inputs[0]),
                        scale: off(node.inputs[1]),
                        shift: off(node.inputs[2]),
                        out: off(node.id),
                        rows: rows as u32,
                        h: h as u32,
                        eps: *eps,
                        layer_norm: matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm),
                        lead_pack: super::ada_lead_pack(&x_dims, &mod_dims),
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::GatedResidual => {
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let rows = total / h;
                    let x_dims: Vec<usize> = graph
                        .node(node.inputs[0])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let gate_dims: Vec<usize> = graph
                        .node(node.inputs[2])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Thunk::GatedResidual {
                        x: off(node.inputs[0]),
                        y: off(node.inputs[1]),
                        gate: off(node.inputs[2]),
                        out: off(node.id),
                        rows: rows as u32,
                        h: h as u32,
                        lead_pack: super::ada_lead_pack(&x_dims, &gate_dims),
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::AdaLayerNormBackward { norm, eps } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let x_dims: Vec<usize> =
                        x_shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    let mod_dims: Vec<usize> = graph
                        .node(node.inputs[1])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let (mod_rows, seq_per_mod) = super::ada_mod_launch(&x_dims, &mod_dims);
                    Thunk::AdaLayerNormBackward {
                        x: off(node.inputs[0]),
                        scale: off(node.inputs[1]),
                        dy: off(node.inputs[3]),
                        out: off(node.id),
                        h: h as u32,
                        eps: *eps,
                        layer_norm: matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm),
                        seq_per_mod,
                        mod_rows,
                        dt: node.shape.dtype().into(),
                    }
                }

                Op::GatedResidualBackward => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let x_dims: Vec<usize> =
                        x_shape.dims().iter().map(|d| d.unwrap_static()).collect();
                    let gate_dims: Vec<usize> = graph
                        .node(node.inputs[2])
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let (mod_rows, seq_per_mod) = super::ada_mod_launch(&x_dims, &gate_dims);
                    Thunk::GatedResidualBackward {
                        y: off(node.inputs[1]),
                        gate: off(node.inputs[2]),
                        dy: off(node.inputs[3]),
                        out: off(node.id),
                        h: h as u32,
                        seq_per_mod,
                        mod_rows,
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
                        half: node.shape.dtype() == rlx_ir::DType::F16,
                    }
                }

                Op::Attention {
                    num_heads,
                    head_dim,
                    v_head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } => {
                    // V/output per-head width; == head_dim unless asymmetric (MLA).
                    let v_head_dim = v_head_dim.unwrap_or(*head_dim);
                    // The f16 SDPA kernel (`sdpa_h`) is symmetric-only; MLA's
                    // asymmetric V/output width is an f32 feature (DeepSeek/Kimi).
                    // Guard the untouched f16 path rather than silently miscompute.
                    assert!(
                        v_head_dim == *head_dim
                            || !matches!(
                                crate::thunk::HalfFlag::from(node.shape.dtype()),
                                crate::thunk::HalfFlag::F16
                            ),
                        "rlx-metal: asymmetric v_head_dim (MLA) requires f32 attention \
                         (f16 sdpa_h path unmodified)"
                    );
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
                    // Infer GQA: K/V may have fewer heads than Q (no graph-side
                    // `repeat_kv`). Rank-4 BSNH uses dim2; BHSD uses dim1; rank-3
                    // packs heads into the last dim.
                    let kv_heads = {
                        let hd = *head_dim;
                        let inferred = if k_shape.rank() == 4 {
                            if bhsd == 1 {
                                k_shape.dim(1).unwrap_static()
                            } else {
                                k_shape.dim(2).unwrap_static()
                            }
                        } else if k_shape.rank() >= 3 {
                            let last = k_shape.dim(k_shape.rank() - 1).unwrap_static();
                            if last == hd {
                                // Rare [B,S,H,D] mis-ranked as 3 — fall back.
                                *num_heads
                            } else {
                                last / hd.max(1)
                            }
                        } else {
                            *num_heads
                        };
                        if inferred > 0 && *num_heads % inferred == 0 {
                            inferred
                        } else {
                            *num_heads
                        }
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
                        kv_heads: kv_heads as u32,
                        head_dim: *head_dim as u32,
                        v_head_dim: v_head_dim as u32,
                        mask_kind: mask_kind_u32,
                        window,
                        dt: node.shape.dtype().into(),
                        kv_f16: graph.node(node.inputs[1]).shape.dtype() == rlx_ir::DType::F16,
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
                    // Fuse the three sibling grads into one pass when the whole
                    // group is present and GPU-eligible (no custom/bias mask,
                    // not [B,H,S,D]). Emitted at the **last** sibling (max node
                    // id) — by then the planner has reserved all three output
                    // slots (each from its own node position), so writing dq/dk/dv
                    // together lands inside every slot's live range. Earlier
                    // siblings collapse to Nop. `RLX_METAL_ATTN_BWD_FUSE=0` opts
                    // out (per-wrt path, for parity checks).
                    let group_key = (
                        node.inputs[0],
                        node.inputs[1],
                        node.inputs[2],
                        node.inputs[3],
                    );
                    let group = attn_bwd_groups.get(&group_key);
                    let fusable = group.is_some_and(|g| g.iter().all(Option::is_some))
                        && bhsd == 0
                        && !matches!(mask_kind_u32, 2 | 4)
                        && rlx_ir::env::var("RLX_METAL_ATTN_BWD_FUSE").as_deref() != Some("0");
                    let _ = wrt_id;
                    if fusable {
                        let g = group.unwrap();
                        let primary = g.iter().flatten().copied().max().unwrap();
                        if node.id == primary {
                            // Last sibling → emit the fused all-grads thunk.
                            Thunk::AttentionBackwardAll {
                                q: off(node.inputs[0]),
                                k: off(node.inputs[1]),
                                v: off(node.inputs[2]),
                                dy: off(node.inputs[3]),
                                out_dq: off(g[0].unwrap()),
                                out_dk: off(g[1].unwrap()),
                                out_dv: off(g[2].unwrap()),
                                batch: batch as u32,
                                seq: seq as u32,
                                kv_seq: kv_seq as u32,
                                heads: *num_heads as u32,
                                head_dim: *head_dim as u32,
                                mask_kind: mask_kind_u32,
                                window,
                            }
                        } else {
                            // Earlier siblings → handled by the primary thunk.
                            Thunk::Nop
                        }
                    } else {
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
                }

                // IR-level fused variant (from `FuseAttentionBackwardAll`, opt-in
                // via `RLX_CPU_ATTN_BWD_FUSE=1`). Distinct from the thunk-level
                // fusion above: the pass already collapsed the three siblings into
                // one packed-`[3B,…]`-output node + three axis-0 `Narrow`s. dQ/dK/dV
                // live contiguously in that one buffer, so the outputs are the packed
                // base and the two per-narrow strides (matching the narrow starts
                // 0 / b / 2b). Reuses the same `encode_attention_bwd_all` kernel as
                // mechanism-2, so it inherits its correctness.
                Op::AttentionBackwardAll {
                    num_heads,
                    head_dim,
                    mask_kind,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!(
                            "rlx-metal AttentionBackwardAll: F32 only (use CPU for f16 training)"
                        );
                    }
                    let (mask_kind_u32, window) = match mask_kind {
                        rlx_ir::op::MaskKind::None => (0u32, 0u32),
                        rlx_ir::op::MaskKind::Causal => (1u32, 0u32),
                        rlx_ir::op::MaskKind::SlidingWindow(w) => (3u32, *w as u32),
                        rlx_ir::op::MaskKind::Custom | rlx_ir::op::MaskKind::Bias => panic!(
                            "rlx-metal AttentionBackwardAll: custom/bias mask is not GPU-fusable \
                             (the fusion pass should not have emitted it)"
                        ),
                    };
                    let q_shape = &graph.node(node.inputs[0]).shape;
                    let k_shape = &graph.node(node.inputs[1]).shape;
                    let rank = q_shape.rank();
                    // Same [B,S,H,D]/[B,H,S,D]/rank-3/rank-2 derivation as the
                    // per-`wrt` arm; the kernel assumes the [B,S,H,D] layout, so
                    // only bhsd==0 is fusable (guaranteed: the thunk-level path
                    // that also feeds these kernels gates on `bhsd == 0`).
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
                    debug_assert_eq!(
                        bhsd, 0,
                        "rlx-metal AttentionBackwardAll: [B,H,S,D] layout not supported by the \
                         fused kernel"
                    );
                    // Packed output is [3B,…]; each dQ/dK/dV narrow is a third of it.
                    let per = node.shape.num_elements().unwrap() / 3 * 4;
                    let base = off(node.id);
                    Thunk::AttentionBackwardAll {
                        q: off(node.inputs[0]),
                        k: off(node.inputs[1]),
                        v: off(node.inputs[2]),
                        dy: off(node.inputs[3]),
                        out_dq: base,
                        out_dk: base + per,
                        out_dv: base + 2 * per,
                        batch: batch as u32,
                        seq: seq as u32,
                        kv_seq: kv_seq as u32,
                        heads: *num_heads as u32,
                        head_dim: *head_dim as u32,
                        mask_kind: mask_kind_u32,
                        window,
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

                Op::KvAppend { axis, pos } => {
                    // In-place append: write input[1] (the new row) into the
                    // output buffer (aliased to `cache`) at sequence index `pos`.
                    // Output shape == cache shape, so `seq_cap` = output's axis
                    // dim (the buffer's true seq stride).
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
                    let seq_cap = out_shape.dim(*axis).unwrap_static();
                    Thunk::KvAppend {
                        src: off(node.inputs[1]),
                        dst: off(node.id),
                        outer: outer as u32,
                        seq_cap: seq_cap as u32,
                        pos: *pos as u32,
                        inner: inner as u32,
                        dt: out_shape.dtype().into(),
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
                    let input_dts: Vec<HalfFlag> = node
                        .inputs
                        .iter()
                        .map(|&in_id| graph.node(in_id).shape.dtype().into())
                        .collect();
                    let weight_const = node.inputs.iter().all(|&in_id| {
                        matches!(
                            graph.node(in_id).op,
                            rlx_ir::op::Op::Param { .. } | rlx_ir::op::Op::Constant { .. }
                        )
                    });
                    Thunk::Concat {
                        dst: off(node.id),
                        outer: outer as u32,
                        dst_axis: dst_axis as u32,
                        inner: inner as u32,
                        dt: out_shape.dtype().into(),
                        inputs,
                        input_dts,
                        weight_const,
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
                        let n = in_shape.dim(0).unwrap_static() as u32;
                        let c_in = in_shape.dim(1).unwrap_static() as u32;
                        let h = in_shape.dim(2).unwrap_static() as u32;
                        let w = in_shape.dim(3).unwrap_static() as u32;
                        let c_out = out_shape.dim(1).unwrap_static() as u32;
                        let h_out = out_shape.dim(2).unwrap_static() as u32;
                        let w_out = out_shape.dim(3).unwrap_static() as u32;
                        // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis and the
                        // length in W (`[N,C,1,L]`), keeping the length kernel/stride/pad/
                        // dilation at index 0 (`kernel=[k,1]`). A literal 2D conv would run
                        // the k-tap kernel over the singleton H axis and ignore the length.
                        // `[N,C,1,L]` and `[N,C,L,1]` share row-major layout, so relabel
                        // the length onto H (no copy) — matching rlx-cpu, the MLX 1D path,
                        // and onnxruntime. (VITS/TinyTTS duration predictor & text encoder.)
                        let one_d_w = h == 1
                            && w > 1
                            && kernel_size[0] > 1
                            && kernel_size.get(1).copied().unwrap_or(1) == 1;
                        let (h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw) = if one_d_w {
                            (
                                w,
                                1,
                                w_out,
                                1,
                                kernel_size[0] as u32,
                                1,
                                stride.first().copied().unwrap_or(1) as u32,
                                1,
                                padding.first().copied().unwrap_or(0) as u32,
                                0,
                                dilation.first().copied().unwrap_or(1) as u32,
                                1,
                            )
                        } else {
                            (
                                h,
                                w,
                                h_out,
                                w_out,
                                kernel_size[0] as u32,
                                kernel_size[1] as u32,
                                stride.first().copied().unwrap_or(1) as u32,
                                stride.get(1).copied().unwrap_or(1) as u32,
                                padding.first().copied().unwrap_or(0) as u32,
                                padding.get(1).copied().unwrap_or(0) as u32,
                                dilation.first().copied().unwrap_or(1) as u32,
                                dilation.get(1).copied().unwrap_or(1) as u32,
                            )
                        };
                        Thunk::Conv2D {
                            src: off(node.inputs[0]),
                            weight: off(node.inputs[1]),
                            dst: off(node.id),
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
                        half: node.shape.dtype() == rlx_ir::DType::F16,
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

                Op::DequantGroupedMatMulMlx { scheme } => {
                    // 5 inputs: input, w_q, scales, biases, expert_idx.
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
                    let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let scales_shape = &graph.node(node.inputs[2]).shape;
                    let num_experts = scales_shape.dim(0).unwrap_static();
                    let w_bytes = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    let slab_bytes = w_bytes / num_experts.max(1);
                    Thunk::DequantGroupedMatMulMlx {
                        input: off(node.inputs[0]),
                        w_q: off(node.inputs[1]),
                        scale: off(node.inputs[2]),
                        zp: off(node.inputs[3]),
                        expert_idx: off(node.inputs[4]),
                        dst: off(node.id),
                        m: m as u32,
                        k_dim: k_dim as u32,
                        n: n as u32,
                        num_experts: num_experts as u32,
                        slab_bytes: slab_bytes as u32,
                        scheme: *scheme,
                        scale_bf16: scales_shape.dtype() == rlx_ir::DType::BF16,
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

                Op::CumProd { axis, exclusive } | Op::CumMax { axis, exclusive } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal CumProd/CumMax: F32 only");
                    }
                    let rank = node.shape.rank();
                    let ax = if *axis < 0 {
                        (rank as i32 + *axis) as usize
                    } else {
                        *axis as usize
                    };
                    if ax != rank.saturating_sub(1) {
                        panic!(
                            "rlx-metal CumProd/CumMax: only last-axis wired (got axis={axis}, rank={rank})"
                        );
                    }
                    let cols = node.shape.dim(ax).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    Thunk::CumScan {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        rows: (total / cols.max(1)) as u32,
                        cols: cols as u32,
                        exclusive: *exclusive,
                        is_max: matches!(node.op, Op::CumMax { .. }),
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
                    let lhs_n = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let rhs_n = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    // Scalar (or size-1) operands broadcast over the output;
                    // anything else with mismatched counts is unsupported —
                    // expand upstream (wgpu unfuse) or use BinaryBroadcast-style
                    // strides (not yet wired for Compare).
                    let lhs_scalar = lhs_n == 1 && len > 1;
                    let rhs_scalar = rhs_n == 1 && len > 1;
                    if !lhs_scalar && !rhs_scalar && (lhs_n != len || rhs_n != len) {
                        panic!(
                            "rlx-metal: Compare with non-scalar broadcast \
                             (out={len}, lhs={lhs_n}, rhs={rhs_n}) is not supported"
                        );
                    }
                    Thunk::Compare {
                        lhs: off(node.inputs[0]),
                        rhs: off(node.inputs[1]),
                        dst: off(node.id),
                        len: len as u32,
                        op: *cmp,
                        lhs_scalar,
                        rhs_scalar,
                    }
                }

                Op::Where => {
                    let len = node.shape.num_elements().unwrap();
                    let cn = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let tn = graph.node(node.inputs[1]).shape.num_elements().unwrap();
                    let fn_ = graph.node(node.inputs[2]).shape.num_elements().unwrap();
                    let cond_scalar = cn == 1 && len > 1;
                    let true_scalar = tn == 1 && len > 1;
                    let false_scalar = fn_ == 1 && len > 1;
                    if (!cond_scalar && cn != len)
                        || (!true_scalar && tn != len)
                        || (!false_scalar && fn_ != len)
                    {
                        panic!(
                            "rlx-metal: Where with non-scalar broadcast \
                             (out={len}, cond={cn}, true={tn}, false={fn_}) \
                             is not supported"
                        );
                    }
                    Thunk::Where {
                        cond: off(node.inputs[0]),
                        on_true: off(node.inputs[1]),
                        on_false: off(node.inputs[2]),
                        dst: off(node.id),
                        len: len as u32,
                        cond_scalar,
                        true_scalar,
                        false_scalar,
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

                Op::ReluBackward => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::ReluBackward {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dx: off(node.id),
                        len: len as u32,
                    }
                }

                Op::ActivationBackward { kind } => {
                    let len = node.shape.num_elements().unwrap();
                    Thunk::ActivationBackward {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dx: off(node.id),
                        len: len as u32,
                        op: activation_backward_op_id(*kind),
                    }
                }

                // C64 Wirtinger surface — native MSL (mirrors CUDA
                // `complex_wirtinger.cu`). Interleaved [re, im] pairs; `len`
                // is the complex-element count.
                Op::ComplexNormSq => {
                    let src = node.inputs[0];
                    if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-metal ComplexNormSq: expected C64 input, got {:?}",
                            graph.node(src).shape.dtype()
                        );
                    }
                    let len = node.shape.num_elements().unwrap();
                    Thunk::ComplexNormSq {
                        src: off(src),
                        dst: off(node.id),
                        len: len as u32,
                    }
                }
                Op::ComplexNormSqBackward => {
                    let z = node.inputs[0];
                    let g = node.inputs[1];
                    if graph.node(z).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-metal ComplexNormSqBackward: expected C64 z, got {:?}",
                            graph.node(z).shape.dtype()
                        );
                    }
                    let len = node.shape.num_elements().unwrap();
                    Thunk::ComplexNormSqBackward {
                        z: off(z),
                        g: off(g),
                        dz: off(node.id),
                        len: len as u32,
                    }
                }
                Op::Conjugate => {
                    let src = node.inputs[0];
                    if graph.node(src).shape.dtype() != rlx_ir::DType::C64 {
                        panic!(
                            "rlx-metal Conjugate: expected C64 input, got {:?}",
                            graph.node(src).shape.dtype()
                        );
                    }
                    let len = node.shape.num_elements().unwrap();
                    Thunk::ConjugateC64 {
                        src: off(src),
                        dst: off(node.id),
                        len: len as u32,
                    }
                }

                Op::FftButterflyStage { stage, n_fft } => {
                    let state_shape = &graph.node(node.inputs[0]).shape;
                    assert_eq!(
                        state_shape.dtype(),
                        rlx_ir::DType::F32,
                        "rlx-metal Op::FftButterflyStage requires F32 state"
                    );
                    Thunk::FftButterflyStage {
                        state: off(node.inputs[0]),
                        out: off(node.id),
                        gate: off(node.inputs[1]),
                        rev: off(node.inputs[2]),
                        tw_re: off(node.inputs[3]),
                        tw_im: off(node.inputs[4]),
                        batch: state_shape.dim(0).unwrap_static() as u32,
                        n_fft: *n_fft,
                        stage: *stage,
                    }
                }

                Op::FakeQuantize {
                    bits,
                    axis,
                    ste: _,
                    scale_mode,
                } => {
                    use rlx_ir::op::ScaleMode;
                    let q_max = match *bits {
                        8 => 127.0f32,
                        4 => 7.0,
                        2 => 1.0,
                        n => panic!("rlx-metal FakeQuantize: unsupported bits {n}"),
                    };
                    // EMA needs mutable running-scale state — keep HostOp.
                    // FakeQuantizeBackward / LSQ also stay on HostOp (catch-all).
                    if matches!(scale_mode, ScaleMode::EMA { .. }) {
                        Thunk::HostOp {
                            desc: rlx_cpu::rlx_host_op_desc!(graph, node, &off),
                        }
                    } else {
                        // Mirror `rlx_cpu::thunk::ops::quant::quant_layout`.
                        let (chan_dim, inner) = match *axis {
                            None => (1usize, node.shape.num_elements().unwrap_or(0).max(1)),
                            Some(d) => {
                                let chan_dim = node.shape.dim(d).unwrap_static();
                                let inner: usize = (d + 1..node.shape.rank())
                                    .map(|i| node.shape.dim(i).unwrap_static())
                                    .product::<usize>()
                                    .max(1);
                                (chan_dim, inner)
                            }
                        };
                        let n = node.shape.num_elements().unwrap() as u32;
                        let chan_dim = chan_dim as u32;
                        let inner = inner as u32;
                        match scale_mode {
                            ScaleMode::Fixed => Thunk::FakeQuantizeFixed {
                                src: off(node.inputs[0]),
                                scale: off(node.inputs[1]),
                                dst: off(node.id),
                                n,
                                chan_dim,
                                inner,
                                q_max,
                            },
                            ScaleMode::PerBatch => Thunk::FakeQuantizePerBatch {
                                src: off(node.inputs[0]),
                                dst: off(node.id),
                                n,
                                chan_dim,
                                inner,
                                q_max,
                            },
                            ScaleMode::EMA { .. } => unreachable!(),
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

                Op::Scan { .. } => {
                    // Host fallback: compile the body once, then loop it on the
                    // CPU against the unified-memory arena at run time.
                    Thunk::ScanHost {
                        desc: rlx_cpu::rlx_scan_host_desc!(graph, node, &off),
                    }
                }
                Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => Thunk::HostOp {
                    desc: rlx_cpu::rlx_host_op_desc!(graph, node, &off),
                },
                Op::ScatterNd { .. }
                | Op::ScatterElements { .. }
                | Op::GatherNd { .. }
                | Op::GatherElements { .. } => Thunk::CpuIndexing {
                    thunk: rlx_cpu::rlx_indexing_thunk!(graph, node, &off),
                },

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
                    gate_per_channel,
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
                        gate_per_channel: *gate_per_channel,
                        carry_state: *carry_state,
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

                Op::Pad { pads, mode } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let dtype = in_shape.dtype();
                    let in_dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let before: Vec<u32> = (0..rank).map(|i| pads[i][0] as u32).collect();
                    let after: Vec<u32> = (0..rank).map(|i| pads[i][1] as u32).collect();
                    let elem_bytes = dtype.size_bytes();
                    let fill = rlx_gpu_host::pad_fill_bytes(*mode, dtype, elem_bytes);
                    Thunk::Pad {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        in_dims,
                        before,
                        after,
                        mode: *mode,
                        fill,
                        elem_bytes: elem_bytes as u8,
                    }
                }

                Op::Slice {
                    axis,
                    start,
                    len,
                    step,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let in_dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    Thunk::Slice {
                        src: off(node.inputs[0]),
                        dst: off(node.id),
                        in_dims,
                        axis: *axis as u32,
                        start: *start as u32,
                        len: *len as u32,
                        step: *step,
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
                        let x_f16 =
                            matches!(graph.node(node.inputs[0]).shape.dtype(), rlx_ir::DType::F16);
                        let dst_f16 = matches!(node.shape.dtype(), rlx_ir::DType::F16);
                        Thunk::DequantMatMulGguf {
                            x: off(node.inputs[0]),
                            w_q: off(node.inputs[1]),
                            dst: off(node.id),
                            m: m as u32,
                            k: k as u32,
                            n: n as u32,
                            scheme: *scheme,
                            x_f16,
                            dst_f16,
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
                            QuantScheme::MlxAffine { .. }
                            | QuantScheme::MlxMxfp4 { .. }
                            | QuantScheme::MlxMxfp8 { .. } => Thunk::DequantMatMulMlx {
                                x: off(node.inputs[0]),
                                w_q: off(node.inputs[1]),
                                scale: off(node.inputs[2]),
                                zp: off(node.inputs[3]),
                                dst: off(node.id),
                                m: m as u32,
                                k: k as u32,
                                n: n as u32,
                                scheme: *scheme,
                            },
                            QuantScheme::MxFp4x2Block { group_size } => {
                                // 3 inputs (x, w_q=[plane0|plane1], scale=[s0|s1]);
                                // fused decode-matmul MSL kernel.
                                Thunk::DequantMatMulMxFp4x2 {
                                    x: off(node.inputs[0]),
                                    w_q: off(node.inputs[1]),
                                    scale: off(node.inputs[2]),
                                    dst: off(node.id),
                                    m: m as u32,
                                    k: k as u32,
                                    n: n as u32,
                                    group: *group_size,
                                }
                            }
                            other => panic!(
                                "rlx-metal: Op::DequantMatMul legacy scheme {other:?} \
                                 is CPU-only unless Int4/FP8/NVFP4/MLX; use GGUF K-quants or Device::Cpu."
                            ),
                        }
                    }
                }

                Op::SynthReconstruct {
                    kind: rlx_ir::SynthKind::Codebook { entry_dim, .. },
                } => {
                    let idx_shape = &graph.node(node.inputs[0]).shape;
                    let n = idx_shape.dim(0).unwrap_static();
                    let k = idx_shape.dim(1).unwrap_static() * *entry_dim as usize;
                    Thunk::SynthReconstruct {
                        indices: off(node.inputs[0]),
                        codebook: off(node.inputs[1]),
                        dst: off(node.id),
                        k: k as u32,
                        n: n as u32,
                        entry_dim: *entry_dim,
                    }
                }

                Op::SynthMatMul {
                    kind:
                        rlx_ir::SynthKind::Codebook {
                            entry_dim,
                            num_entries,
                        },
                } => {
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    let m = total / n.max(1);
                    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
                    let k = x_total / m.max(1);
                    Thunk::SynthMatMul {
                        x: off(node.inputs[0]),
                        indices: off(node.inputs[1]),
                        codebook: off(node.inputs[2]),
                        dst: off(node.id),
                        m: m as u32,
                        k: k as u32,
                        n: n as u32,
                        entry_dim: *entry_dim,
                        num_entries: *num_entries,
                        half: graph.node(node.inputs[0]).shape.dtype() == rlx_ir::DType::F16
                            && node.shape.dtype() == rlx_ir::DType::F16,
                    }
                }

                Op::SynthMatMulBackward {
                    kind:
                        rlx_ir::SynthKind::Codebook {
                            entry_dim,
                            num_entries,
                        },
                    wrt,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let m = x_shape.dim(0).unwrap_static();
                    let k = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let n = idx_shape.dim(0).unwrap_static();
                    Thunk::SynthMatMulBackward {
                        x: off(node.inputs[0]),
                        indices: off(node.inputs[1]),
                        codebook: off(node.inputs[2]),
                        upstream: off(node.inputs[3]),
                        dst: off(node.id),
                        m: m as u32,
                        n: n as u32,
                        k: k as u32,
                        entry_dim: *entry_dim,
                        num_entries: *num_entries,
                        dx: matches!(wrt, rlx_ir::SynthBwdWrt::Dx),
                    }
                }

                Op::SplineActivation {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let channels = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let total = x_shape.num_elements().unwrap();
                    let rows = total / channels.max(1);
                    Thunk::SplineActivation {
                        x: off(node.inputs[0]),
                        coeff: off(node.inputs[1]),
                        dst: off(node.id),
                        rows: rows as u32,
                        channels: channels as u32,
                        num_basis: *num_basis,
                        grid_min: *grid_min,
                        grid_max: *grid_max,
                    }
                }

                Op::SplineActivationBackwardX {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let channels = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let rows = x_shape.num_elements().unwrap() / channels.max(1);
                    Thunk::SplineActivationBackwardX {
                        x: off(node.inputs[0]),
                        coeff: off(node.inputs[1]),
                        upstream: off(node.inputs[2]),
                        dst: off(node.id),
                        rows: rows as u32,
                        channels: channels as u32,
                        num_basis: *num_basis,
                        grid_min: *grid_min,
                        grid_max: *grid_max,
                    }
                }

                Op::SplineActivationBackwardCoeff {
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let channels = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let rows = x_shape.num_elements().unwrap() / channels.max(1);
                    Thunk::SplineActivationBackwardCoeff {
                        x: off(node.inputs[0]),
                        upstream: off(node.inputs[1]),
                        dst: off(node.id),
                        rows: rows as u32,
                        channels: channels as u32,
                        num_basis: *num_basis,
                        grid_min: *grid_min,
                        grid_max: *grid_max,
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

                Op::LayerNormBackwardInput { eps, .. } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal LayerNormBackwardInput: F32 only");
                    }
                    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                    let total = node.shape.num_elements().unwrap();
                    Thunk::LayerNormBackwardInput {
                        x: off(node.inputs[0]),
                        gamma: off(node.inputs[1]),
                        dy: off(node.inputs[2]),
                        dx: off(node.id),
                        rows: (total / h) as u32,
                        h: h as u32,
                        eps: *eps,
                    }
                }

                Op::LayerNormBackwardGamma { eps, .. } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal LayerNormBackwardGamma: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let x_total = x_shape.num_elements().unwrap();
                    Thunk::LayerNormBackwardGamma {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dgamma: off(node.id),
                        rows: (x_total / h) as u32,
                        h: h as u32,
                        eps: *eps,
                    }
                }

                Op::GroupNormBackwardInput { num_groups, eps }
                | Op::GroupNormBackwardGamma { num_groups, eps }
                | Op::GroupNormBackwardBeta { num_groups, eps } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal GroupNormBackward: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let n = x_shape.dim(0).unwrap_static() as u32;
                    let c = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    match &node.op {
                        Op::GroupNormBackwardInput { .. } => Thunk::GroupNormBackwardInput {
                            x: off(node.inputs[0]),
                            gamma: off(node.inputs[1]),
                            beta: off(node.inputs[2]),
                            dy: off(node.inputs[3]),
                            dx: off(node.id),
                            n,
                            c,
                            h,
                            w,
                            num_groups: *num_groups as u32,
                            eps: *eps,
                        },
                        Op::GroupNormBackwardGamma { .. } => Thunk::GroupNormBackwardGamma {
                            x: off(node.inputs[0]),
                            dy: off(node.inputs[1]),
                            dgamma: off(node.id),
                            n,
                            c,
                            h,
                            w,
                            num_groups: *num_groups as u32,
                            eps: *eps,
                        },
                        Op::GroupNormBackwardBeta { .. } => Thunk::GroupNormBackwardBeta {
                            dy: off(node.inputs[1]),
                            dbeta: off(node.id),
                            n,
                            c,
                            h,
                            w,
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

                Op::MaxPool3dBackward {
                    kernel_size,
                    stride,
                    padding,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal MaxPool3dBackward: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    Thunk::MaxPool3dBackward {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dx: off(node.id),
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c: x_shape.dim(1).unwrap_static() as u32,
                        d: x_shape.dim(2).unwrap_static() as u32,
                        h: x_shape.dim(3).unwrap_static() as u32,
                        w: x_shape.dim(4).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride.first().copied().unwrap_or(1) as u32,
                        sh: stride.get(1).copied().unwrap_or(1) as u32,
                        sw: stride.get(2).copied().unwrap_or(1) as u32,
                        pd: padding.first().copied().unwrap_or(0) as u32,
                        ph: padding.get(1).copied().unwrap_or(0) as u32,
                        pw: padding.get(2).copied().unwrap_or(0) as u32,
                    }
                }

                Op::Conv3dBackwardInput {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal Conv3dBackwardInput: F32 only");
                    }
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    Thunk::Conv3dBackwardInput {
                        dy: off(node.inputs[0]),
                        w: off(node.inputs[1]),
                        dx: off(node.id),
                        n: out_shape.dim(0).unwrap_static() as u32,
                        c_in: out_shape.dim(1).unwrap_static() as u32,
                        d: out_shape.dim(2).unwrap_static() as u32,
                        h: out_shape.dim(3).unwrap_static() as u32,
                        w_in: out_shape.dim(4).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride.first().copied().unwrap_or(1) as u32,
                        sh: stride.get(1).copied().unwrap_or(1) as u32,
                        sw: stride.get(2).copied().unwrap_or(1) as u32,
                        pd: padding.first().copied().unwrap_or(0) as u32,
                        ph: padding.get(1).copied().unwrap_or(0) as u32,
                        pw: padding.get(2).copied().unwrap_or(0) as u32,
                        dd: dilation.first().copied().unwrap_or(1) as u32,
                        dh: dilation.get(1).copied().unwrap_or(1) as u32,
                        dw: dilation.get(2).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    }
                }

                Op::Conv3dBackwardWeight {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    if node.shape.dtype() != rlx_ir::DType::F32 {
                        panic!("rlx-metal Conv3dBackwardWeight: F32 only");
                    }
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let dy_shape = &graph.node(node.inputs[1]).shape;
                    Thunk::Conv3dBackwardWeight {
                        x: off(node.inputs[0]),
                        dy: off(node.inputs[1]),
                        dw: off(node.id),
                        n: x_shape.dim(0).unwrap_static() as u32,
                        c_in: x_shape.dim(1).unwrap_static() as u32,
                        d: x_shape.dim(2).unwrap_static() as u32,
                        h: x_shape.dim(3).unwrap_static() as u32,
                        w: x_shape.dim(4).unwrap_static() as u32,
                        c_out: dy_shape.dim(1).unwrap_static() as u32,
                        d_out: dy_shape.dim(2).unwrap_static() as u32,
                        h_out: dy_shape.dim(3).unwrap_static() as u32,
                        w_out: dy_shape.dim(4).unwrap_static() as u32,
                        kd: kernel_size[0] as u32,
                        kh: kernel_size[1] as u32,
                        kw: kernel_size[2] as u32,
                        sd: stride.first().copied().unwrap_or(1) as u32,
                        sh: stride.get(1).copied().unwrap_or(1) as u32,
                        sw: stride.get(2).copied().unwrap_or(1) as u32,
                        pd: padding.first().copied().unwrap_or(0) as u32,
                        ph: padding.get(1).copied().unwrap_or(0) as u32,
                        pw: padding.get(2).copied().unwrap_or(0) as u32,
                        dd: dilation.first().copied().unwrap_or(1) as u32,
                        dh: dilation.get(1).copied().unwrap_or(1) as u32,
                        dw_dil: dilation.get(2).copied().unwrap_or(1) as u32,
                        groups: *groups as u32,
                    }
                }

                // Core Riemannian / SPD-manifold ops (BiMap / ReEig / LogEig /
                // SpdBatchNorm / SpdKarcherMean + backwards). No MSL eigen
                // kernel; host-fallback to `rlx_cpu::spd` (F64) against the
                // unified-memory arena, like `Op::Fft`. The arena stores these
                // nodes as f32 (the SPD subgraph was widened f64→f32 for arena
                // planning), so we carry each operand's REAL declared F64 shape
                // (dims from the arena node, dtype forced back to F64) — the
                // packed `[2n²+n]` forward / `(λ, U, dY)` backward layouts then
                // resolve through the CPU thunk automatically. Mirrors the
                // Vulkan `Step::SpdHost` route.
                op if crate::spd::is_spd_host(op) => {
                    let to_f64 = |s: &Shape| s.clone().with_dtype(rlx_ir::DType::F64);
                    let inputs_v: Vec<(usize, u32, Shape)> = node
                        .inputs
                        .iter()
                        .map(|&in_id| {
                            let s = to_f64(&graph.node(in_id).shape);
                            let len = s.num_elements().unwrap_or(0) as u32;
                            (off(in_id), len, s)
                        })
                        .collect();
                    let out_shape = to_f64(&node.shape);
                    let out_len = out_shape.num_elements().unwrap_or(0) as u32;
                    Thunk::SpdHost {
                        op: op.clone(),
                        inputs: inputs_v,
                        output: (off(node.id), out_len, out_shape),
                    }
                }

                // Fused VQ has a native on-GPU MSL kernel — bypass the (slow,
                // copy-bound) host-callback custom-op path.
                Op::Custom { name, attrs, .. } if name == "rlx.vq_assign" => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let cb_shape = &graph.node(node.inputs[1]).shape;
                    Thunk::VqAssign {
                        x: off(node.inputs[0]),
                        cb: off(node.inputs[1]),
                        out: off(node.id),
                        n: x_shape.dim(0).unwrap_static() as u32,
                        d: x_shape.dim(1).unwrap_static() as u32,
                        k: cb_shape.dim(0).unwrap_static() as u32,
                        metric: attrs.first().copied().unwrap_or(0) as u32,
                    }
                }

                Op::Custom { name, attrs, .. } => {
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
                    let output = (off(node.id), out_len, node.shape.clone());
                    // Prefer a raw-GPU kernel (no host roundtrip / sync) if one
                    // is registered; otherwise fall back to the host-delegate.
                    if let Some(kernel) = crate::op_registry::lookup_metal_gpu_kernel(name) {
                        Thunk::CustomGpuOp {
                            kernel,
                            inputs: inputs_v,
                            output,
                            attrs: attrs.clone(),
                        }
                    } else {
                        let kernel =
                            crate::op_registry::lookup_metal_kernel(name).unwrap_or_else(|| {
                                panic!(
                                    "rlx-metal: no MetalKernel registered for \
                                 Op::Custom('{name}'). Either register one via \
                                 rlx_metal::op_registry::register_metal_kernel \
                                 (host) / register_metal_gpu_kernel (raw GPU) \
                                 or pin this graph to Device::Cpu."
                                )
                            });
                        Thunk::CustomOp {
                            kernel,
                            inputs: inputs_v,
                            output,
                            attrs: attrs.clone(),
                        }
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

                // Remaining claimed ops (GroupNorm bwd, FakeQuantize EMA /
                // Backward / LSQ, DenseSolve, CustomFn, …): sync + CPU one-op
                // eval against the unified-memory arena. FusedConvBiasAct /
                // PartitionedConv / FusedTransformerLayer are expanded earlier
                // by `lower_cpu_nop_fused_for_metal` (CPU would Nop them).
                _other => Thunk::HostOp {
                    desc: rlx_cpu::rlx_host_op_desc!(graph, node, &off),
                },
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
                    let (n_src, n_dst, n_src_axis, n_start, n_dt, n_outer) = match &thunks[i] {
                        Thunk::Narrow {
                            src,
                            dst,
                            src_axis,
                            start,
                            dt,
                            outer,
                            ..
                        } => (*src, *dst, *src_axis, *start, *dt, *outer),
                        _ => continue,
                    };
                    let mut j = i + 1;
                    while j < thunks.len() && matches!(thunks[j], Thunk::Nop) {
                        j += 1;
                    }
                    if j >= thunks.len() {
                        continue;
                    }
                    // The Rope must read the Narrow's dst, share its dtype, and —
                    // critically — its row geometry must match Rope's flattened
                    // `batch·seq` model. Rewiring to a single parent row stride is
                    // only valid for a terminal-axis narrow (`outer == batch·seq`).
                    // A non-terminal-axis narrow (MLA's per-head `qk_rope` slice
                    // out of `[B, S, H, qk]`, whose `outer == batch·seq·H`) has
                    // more, narrower rows and would be misread. Mirrors the CPU
                    // pass in `rlx-cpu` compile_dispatch.
                    let rope_ok = match &thunks[j] {
                        Thunk::Rope {
                            src,
                            batch,
                            seq,
                            dt: rd,
                            ..
                        } => *src == n_dst && *rd == n_dt && n_outer == batch * seq,
                        _ => false,
                    };
                    if !rope_ok {
                        continue;
                    }
                    if read_counts.get(&n_dst).copied().unwrap_or(0) != 1 {
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
        fuse_gdn_gated_norm(&mut thunks, &output_offsets);
        fuse_depthwise_conv1d_bsc(&mut thunks, &output_offsets);
        fuse_residual_rms_norm(&mut thunks, &output_offsets);

        Self {
            thunks,
            rng: rng_shared,
        }
    }
}
