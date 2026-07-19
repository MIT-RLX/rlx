#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Apply a synthetic (kernel-generated) attention mask to a `[q_seq, k_seq]`
/// scores matrix. Custom masks are read from a tensor and not handled here.
/// `None` is a no-op so callers don't need to special-case it.
#[inline]
pub(crate) fn apply_synthetic_mask(
    scores: &mut [f32],
    q_seq: usize,
    k_seq: usize,
    kind: rlx_ir::op::MaskKind,
) {
    let neg = crate::config::RuntimeConfig::global().attn_mask_neg_inf;
    let q_offset = k_seq.saturating_sub(q_seq);
    match kind {
        rlx_ir::op::MaskKind::None | rlx_ir::op::MaskKind::Custom | rlx_ir::op::MaskKind::Bias => {}
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

#[allow(unused_variables)]
pub(crate) fn compile_axial_rope2d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::AxialRope2d {
        end_x,
        end_y,
        head_dim,
        num_heads,
        theta,
        repeat_factor,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let batch = in_shape.dim(0).unwrap_static() as u32;
        let seq = in_shape.dim(1).unwrap_static() as u32;
        let hidden = in_shape.dim(2).unwrap_static() as u32;
        Thunk::AxialRope2d {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            batch,
            seq,
            hidden,
            end_x: *end_x as u32,
            end_y: *end_y as u32,
            head_dim: *head_dim as u32,
            num_heads: *num_heads as u32,
            theta: *theta,
            repeat_factor: *repeat_factor as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_attention(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Attention {
        num_heads,
        head_dim,
        mask_kind,
        score_scale,
        attn_logit_softcap,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
        // GQA/MQA: KV-head count from K's element count over B·S_k·D (layout-
        // independent). == num_heads for MHA; < num_heads only on the raw
        // standalone path (real models fuse or replicate K/V to num_heads).
        let k_numel = k_shape
            .num_elements()
            .unwrap_or(batch * kv_seq * *num_heads * *head_dim);
        let nkv = (k_numel / (batch.max(1) * kv_seq.max(1) * (*head_dim).max(1))).max(1) as u32;
        let kv_hs = nkv * *head_dim as u32;
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
            kv_heads: nkv,
            head_dim: *head_dim as u32,
            mask_kind: *mask_kind,
            scale: score_scale.unwrap_or((*head_dim as f32).powf(-0.5)),
            softcap: attn_logit_softcap.unwrap_or(0.0),
            // Defaults: each input is its own contiguous buffer
            // with row stride = hidden. Rewritten by the
            // Narrow→Attention fusion when applicable.
            q_row_stride: hs,
            k_row_stride: kv_hs,
            v_row_stride: kv_hs,
            bhsd,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_attention_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::AttentionBackward {
        num_heads,
        head_dim,
        mask_kind,
        wrt,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
                    true,
                )
            } else {
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
            node_offset(arena, node.inputs[4])
        } else {
            0
        };
        Thunk::AttentionBackward {
            q: node_offset(arena, node.inputs[0]),
            k: node_offset(arena, node.inputs[1]),
            v: node_offset(arena, node.inputs[2]),
            dy: node_offset(arena, node.inputs[3]),
            mask: mask_off,
            out: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            kv_seq: kv_seq as u32,
            heads: *num_heads as u32,
            head_dim: *head_dim as u32,
            mask_kind: *mask_kind,
            wrt: *wrt,
            bhsd,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fused_attention_block(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedAttentionBlock {
        num_heads,
        head_dim,
        has_bias,
        has_rope,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
            // The MIR `Op::FusedAttentionBlock` is emitted only for the
            // BERT-style per-key padding mask (the MIR fusion pass is
            // `Custom`-only), so the buffer mask is authoritative here.
            mask_kind: rlx_ir::op::MaskKind::Custom,
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
            // The MIR `Op::FusedAttentionBlock` is BERT-only (NeoX rope).
            interleaved: false,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rope(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Rope {
        head_dim,
        n_rot,
        style,
    } = &node.op
    else {
        unreachable!()
    };
    {
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
            n_rot: *n_rot as u32,
            cos_len: cos_len as u32,
            // Default: source rows are tightly packed (rewritten
            // by the Narrow→Rope fusion pass below if Rope ends
            // up reading from a wider parent like QKV).
            src_row_stride: hidden as u32,
            interleaved: matches!(style, rlx_ir::op::RopeStyle::GptJ),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fused_swi_g_l_u(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedSwiGLU {
        cast_to: _,
        gate_first,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let n_half = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        Thunk::FusedSwiGLU {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            n_half: n_half as u32,
            total: total as u32,
            gate_first: *gate_first,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rope_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::RopeBackward { head_dim, n_rot } = &node.op else {
        unreachable!()
    };
    {
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
        let cos_shape = &graph.node(node.inputs[1]).shape;
        let cos_len = cos_shape.num_elements().unwrap();
        Thunk::RopeBackward {
            dy: node_offset(arena, node.inputs[0]),
            cos: node_offset(arena, node.inputs[1]),
            sin: node_offset(arena, node.inputs[2]),
            dx: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            hidden: hidden as u32,
            head_dim: *head_dim as u32,
            n_rot: *n_rot as u32,
            cos_len: cos_len as u32,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_axial_rope2d(t: &Thunk, base: *mut u8) {
    let Thunk::AxialRope2d {
        src,
        dst,
        batch,
        seq,
        hidden,
        end_x,
        end_y,
        head_dim,
        num_heads,
        theta,
        repeat_factor,
    } = t
    else {
        unreachable!()
    };
    {
        let b = *batch as usize;
        let s = *seq as usize;
        let hdim = *head_dim as usize;
        let nh = *num_heads as usize;
        let plane = s * (*hidden as usize);
        // `base` is a byte pointer; advance per-batch by element-stride
        // bytes. (Was `base.add(bi * plane)` — 4× under-advance for
        // batch>1; matches the corrected `execute_axial_rope2d_f32`.)
        let plane_bytes = plane * std::mem::size_of::<f32>();
        unsafe {
            for bi in 0..b {
                let input = sl(*src, base.add(bi * plane_bytes), plane);
                let output = sl_mut(*dst, base.add(bi * plane_bytes), plane);
                let rotated = rlx_ir::ops::axial_rope2d::apply_axial_rope2d(
                    input,
                    nh,
                    s,
                    hdim,
                    *end_x as usize,
                    *end_y as usize,
                    *theta,
                    *repeat_factor as usize,
                );
                output.copy_from_slice(&rotated);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_attention_backward(t: &Thunk, base: *mut u8) {
    let Thunk::AttentionBackward {
        q,
        k,
        v,
        dy,
        mask,
        out,
        batch,
        seq,
        kv_seq,
        heads,
        head_dim,
        mask_kind,
        wrt,
        bhsd,
    } = t
    else {
        unreachable!()
    };
    {
        let (b, q_s, k_s, nh, dh) = (
            *batch as usize,
            *seq as usize,
            *kv_seq as usize,
            *heads as usize,
            *head_dim as usize,
        );
        unsafe {
            let q_len = if *bhsd {
                b * nh * q_s * dh
            } else {
                b * q_s * nh * dh
            };
            let k_len = if *bhsd {
                b * nh * k_s * dh
            } else {
                b * k_s * nh * dh
            };
            let out_len = match wrt {
                rlx_ir::op::AttentionBwdWrt::Key | rlx_ir::op::AttentionBwdWrt::Value => k_len,
                rlx_ir::op::AttentionBwdWrt::Query => q_len,
            };
            let q_data = sl(*q, base, q_len);
            let k_data = sl(*k, base, k_len);
            let v_data = sl(*v, base, k_len);
            let dy_data = sl(*dy, base, q_len);
            let out_data = sl_mut(*out, base, out_len);
            let mask_data: &[f32] = if *mask != 0 {
                let ml = match mask_kind {
                    rlx_ir::op::MaskKind::Custom => b * k_s,
                    rlx_ir::op::MaskKind::Bias => b * nh * q_s * k_s,
                    _ => 0,
                };
                sl(*mask, base, ml)
            } else {
                &[]
            };
            crate::attention_bwd::attention_backward(
                *wrt, q_data, k_data, v_data, dy_data, out_data, b, nh, q_s, k_s, dh, *mask_kind,
                mask_data, *bhsd,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rope(t: &Thunk, base: *mut u8) {
    let Thunk::Rope {
        src,
        cos,
        sin,
        dst,
        batch,
        seq,
        hidden,
        head_dim,
        n_rot,
        cos_len,
        src_row_stride,
        interleaved,
    } = t
    else {
        unreachable!()
    };
    {
        let interleaved = *interleaved;
        let (b, s, hs, dh, nr) = (
            *batch as usize,
            *seq as usize,
            *hidden as usize,
            *head_dim as usize,
            *n_rot as usize,
        );
        let tab_half = dh / 2;
        let rot_half = nr / 2;
        let nh = hs / dh;
        let cl = *cos_len as usize;
        let src_rs = *src_row_stride as usize;
        // Number of rows in the RoPE table. A per-(batch·seq) table
        // (`cos_rows == b*s`, distinct from the shared per-seq table) is
        // indexed by the *global* token so ragged batched decode can
        // give each sequence its own absolute position.
        let cos_rows = cl / tab_half.max(1);
        let per_token = cos_rows == b * s && cos_rows != s;
        unsafe {
            let x = sl(*src, base, b * s * src_rs);
            let cos_tab = sl(*cos, base, cl);
            let sin_tab = sl(*sin, base, cl);
            let out = sl_mut(*dst, base, b * s * hs);

            let total = b * s;
            let x_ptr = x.as_ptr() as usize;
            let o_ptr = out.as_mut_ptr() as usize;
            let c_ptr = cos_tab.as_ptr() as usize;
            let s_ptr = sin_tab.as_ptr() as usize;

            crate::pool::par_for(total, 4, &|off, cnt| {
                for idx in off..off + cnt {
                    let bi = idx / s;
                    let si = idx % s;
                    let tab_off = if per_token { idx } else { si } * tab_half;

                    for hi in 0..nh {
                        let src_base = bi * s * src_rs + si * src_rs + hi * dh;
                        let dst_base = bi * s * hs + si * hs + hi * dh;
                        let xp = (x_ptr as *const f32).add(src_base);
                        let op = (o_ptr as *mut f32).add(dst_base);
                        let cp = (c_ptr as *const f32).add(tab_off);
                        let sp = (s_ptr as *const f32).add(tab_off);

                        if interleaved {
                            // GPT-J / llama.cpp-NORM: rotate adjacent
                            // pairs (2i, 2i+1) by angle i.
                            for i in 0..rot_half {
                                let x1 = *xp.add(2 * i);
                                let x2 = *xp.add(2 * i + 1);
                                let cv = *cp.add(i);
                                let sv = *sp.add(i);
                                *op.add(2 * i) = x1 * cv - x2 * sv;
                                *op.add(2 * i + 1) = x2 * cv + x1 * sv;
                            }
                        } else {
                            // HF / NeoX rotate-half: pair (i, i+rot_half).
                            for i in 0..rot_half {
                                let x1 = *xp.add(i);
                                let x2 = *xp.add(rot_half + i);
                                let cv = *cp.add(i);
                                let sv = *sp.add(i);
                                *op.add(i) = x1 * cv - x2 * sv;
                                *op.add(rot_half + i) = x2 * cv + x1 * sv;
                            }
                        }
                        for j in nr..dh {
                            *op.add(j) = *xp.add(j);
                        }
                    }
                }
            });
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fused_swi_g_l_u(t: &Thunk, base: *mut u8) {
    let Thunk::FusedSwiGLU {
        src,
        dst,
        n_half,
        total,
        gate_first,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n_half as usize;
        let t = *total as usize;
        let outer = t / n;
        let in_total = outer * 2 * n;
        let gate_first = *gate_first;
        unsafe {
            let inp = sl(*src, base, in_total);
            let out = sl_mut(*dst, base, t);
            for o in 0..outer {
                let in_row = &inp[o * 2 * n..(o + 1) * 2 * n];
                let out_row = &mut out[o * n..(o + 1) * n];
                for i in 0..n {
                    let (up, gate) = if gate_first {
                        (in_row[n + i], in_row[i])
                    } else {
                        (in_row[i], in_row[n + i])
                    };
                    out_row[i] = up * (gate / (1.0 + (-gate).exp()));
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rope_backward(t: &Thunk, base: *mut u8) {
    let Thunk::RopeBackward {
        dy,
        cos,
        sin,
        dx,
        batch,
        seq,
        hidden,
        head_dim,
        n_rot,
        cos_len,
    } = t
    else {
        unreachable!()
    };
    {
        let (b, s, hs, dh, nr, cl) = (
            *batch as usize,
            *seq as usize,
            *hidden as usize,
            *head_dim as usize,
            *n_rot as usize,
            *cos_len as usize,
        );
        let nh = hs / dh;
        let tab_half = dh / 2;
        unsafe {
            let dys = sl(*dy, base, b * s * hs);
            let cos_tab = sl(*cos, base, cl);
            let sin_tab = sl(*sin, base, cl);
            let out = sl_mut(*dx, base, b * s * hs);
            for bi in 0..b {
                for si in 0..s {
                    let tab_off = si.saturating_mul(tab_half) % cl.max(1);
                    let cp = &cos_tab[tab_off..tab_off + tab_half.min(cl)];
                    let sp = &sin_tab[tab_off..tab_off + tab_half.min(cl)];
                    for hi in 0..nh {
                        let base_idx = bi * s * hs + si * hs + hi * dh;
                        crate::training_bwd::rope_backward_row(
                            &dys[base_idx..base_idx + dh],
                            cp,
                            sp,
                            &mut out[base_idx..base_idx + dh],
                            dh,
                            nr,
                        );
                    }
                }
            }
        }
    }
}

pub unsafe fn execute_rope_backward_f32(
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
    base: *mut u8,
) {
    let (b, s, hs, dh, nr, cl) = (
        batch as usize,
        seq as usize,
        hidden as usize,
        head_dim as usize,
        n_rot as usize,
        cos_len as usize,
    );
    let nh = hs / dh;
    let tab_half = dh / 2;
    let dys = sl(dy, base, b * s * hs);
    let cos_tab = sl(cos, base, cl);
    let sin_tab = sl(sin, base, cl);
    let out = sl_mut(dx, base, b * s * hs);
    for bi in 0..b {
        for si in 0..s {
            let tab_off = si.saturating_mul(tab_half) % cl.max(1);
            let cp = &cos_tab[tab_off..tab_off + tab_half.min(cl)];
            let sp = &sin_tab[tab_off..tab_off + tab_half.min(cl)];
            for hi in 0..nh {
                let base_idx = bi * s * hs + si * hs + hi * dh;
                crate::training_bwd::rope_backward_row(
                    &dys[base_idx..base_idx + dh],
                    cp,
                    sp,
                    &mut out[base_idx..base_idx + dh],
                    dh,
                    nr,
                );
            }
        }
    }
}

/// Host axial 2-D RoPE for Metal (and other) fallbacks on unified memory.
pub unsafe fn execute_axial_rope2d_f32(
    src: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
    end_x: usize,
    end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
    base: *mut u8,
) {
    let plane = seq * hidden;
    let plane_bytes = plane * std::mem::size_of::<f32>();
    for bi in 0..batch {
        let in_off = src + bi * plane_bytes;
        let input = unsafe { sl(in_off, base, plane) };
        let rotated = rlx_ir::ops::axial_rope2d::apply_axial_rope2d(
            input,
            num_heads,
            seq,
            head_dim,
            end_x,
            end_y,
            theta,
            repeat_factor,
        );
        let out_off = dst + bi * plane_bytes;
        let output = unsafe { sl_mut(out_off, base, plane) };
        output.copy_from_slice(&rotated);
    }
}
