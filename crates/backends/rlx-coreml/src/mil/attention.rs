// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `attention` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use super::helpers::simple_op_flex;
use super::helpers::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use super::*;

impl<'a> LowerCtx<'a> {
    /// Scaled dot-product attention. Inputs `[q,k,v]` (+ mask tensor for
    /// `Bias`/`Custom`). The operand layout is **dispatched** on
    /// `num_heads`/`head_dim` — different layouts need different kernels:
    ///   * **split** — last axis == `head_dim` (`[..,S,D]`, any rank ≥ 2):
    ///     the core runs directly (MIL `matmul` batches the leading B/H dims).
    ///   * **fused** — last axis == `num_heads*head_dim` (`[B,S,H·D]`, the
    ///     Qwen3 fused-QKV layout where heads are packed in the last axis):
    ///     reshape+transpose q/k/v to canonical `[B,H,S,D]`, run the core,
    ///     then transpose+reshape the result back to `[B,S,H·D]`.
    pub(crate) fn lower_attention(
        &mut self,
        id: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        out_name: &str,
    ) -> Result<()> {
        let (out_shape, in0, in1, in2, mask_in) = {
            let node = self.graph.node(id);
            let mask_in = match mask_kind {
                MaskKind::Bias | MaskKind::Custom => node.inputs.get(3).copied(),
                _ => None,
            };
            (
                node.shape.clone(),
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                mask_in,
            )
        };
        let rank = out_shape.rank();
        if rank < 2 {
            return Err(CoremlError::Unsupported(
                "attention: need rank >= 2 [..,S,D] layout".into(),
            ));
        }
        let last = dim_static(&out_shape, rank - 1)?;

        // ── split layout: last axis is `head_dim` ──
        //
        // The IR's `Op::Attention` is used with TWO different rank-4 operand
        // layouts across models (both end in `head_dim`, so they can't be told
        // apart by the last axis — disambiguate by which axis equals `num_heads`):
        //
        //   * `[B, S, H, D]` — heads at axis 2 (the CPU/Metal/MLX/wgpu convention;
        //     e.g. Moshi, Llama-style reshapes). `attention_core` wants the heads
        //     at axis 1, so we transpose `[B,S,H,D] → [B,H,S,D]`, attend, then
        //     transpose the `[B,H,Sq,D]` result back to `[B,Sq,H,D]`. Without this
        //     it would attend over the HEADS axis — wrong results, and once
        //     `s_q != s_k` (KV-cache decode) the QKᵀ batch dims mismatch and the
        //     CoreML predict fails outright.
        //   * `[B, H, S, D]` — heads at axis 1 (already canonical). Passed straight
        //     through to `attention_core`.
        //
        // Rank-3 `[B, S, D]` (single head) is likewise already canonical.
        if last == head_dim {
            // Identify the heads axis via the canonical `attention_geom` helper
            // (same disambiguation the MLX/wgpu backends use). `bhsd` = heads are
            // already at axis 1 (`[B,H,S,D]`, canonical for `attention_core`);
            // otherwise a rank-4 operand is `[B,S,H,D]` (heads at axis 2).
            let k_in_shape = self.graph.shape(in1).clone();
            let geom = rlx_ir::attention_geom(&out_shape, &k_in_shape, num_heads, head_dim);
            if rank == 4 && !geom.bhsd {
                // `[B, S, H, D]` → canonical `[B, H, S, D]`, attend, transpose back.
                let (b, s_q, h, s_k) = (geom.batch, geom.seq_q, geom.heads, geom.seq_k);
                let qc = self.bshd_to_bhsd(in0, b, s_q, h, head_dim, &format!("{out_name}_q"))?;
                let kc = self.bshd_to_bhsd(in1, b, s_k, h, head_dim, &format!("{out_name}_k"))?;
                let vc = self.bshd_to_bhsd(in2, b, s_k, h, head_dim, &format!("{out_name}_v"))?;
                let q_canon = bhsd_shape(b, h, s_q, head_dim);
                let k_canon = bhsd_shape(b, h, s_k, head_dim);
                let core = format!("{out_name}_attn");
                self.attention_core(
                    &qc,
                    &kc,
                    &vc,
                    &q_canon,
                    &k_canon,
                    head_dim,
                    mask_kind,
                    mask_in,
                    score_scale,
                    softcap,
                    &core,
                )?;
                // [B,H,Sq,D] → [B,Sq,H,D]
                self.emit(
                    "transpose",
                    out_name,
                    &out_shape,
                    vec![
                        ("x", bind_name(&core)),
                        ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                    ],
                )?;
                self.names.insert(id.0, out_name.to_string());
                return Ok(());
            }
            // Canonical `[B, H, S, D]` (heads at axis 1) or rank-3 `[B, S, D]`.
            let q = self.val(in0);
            let k = self.val(in1);
            let v = self.val(in2);
            let k_shape = self.graph.shape(in1).clone();
            self.attention_core(
                &q,
                &k,
                &v,
                &out_shape,
                &k_shape,
                head_dim,
                mask_kind,
                mask_in,
                score_scale,
                softcap,
                out_name,
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        // ── fused layout: [B,S,H·D] → canonical [B,H,S,D] → attend → fold ──
        if num_heads > 0 && last == num_heads * head_dim && rank == 3 {
            let b = dim_static(&out_shape, 0)?;
            let s_q = dim_static(&out_shape, 1)?;
            let qc =
                self.fused_to_bhsd(in0, b, s_q, num_heads, head_dim, &format!("{out_name}_q"))?;
            let (kc, s_k) = self.fused_to_bhsd_kv(in1, b, head_dim, &format!("{out_name}_k"))?;
            let (vc, _) = self.fused_to_bhsd_kv(in2, b, head_dim, &format!("{out_name}_v"))?;
            let kh = dim_static(&self.graph.shape(in1).clone(), 2)? / head_dim;
            let q_canon = bhsd_shape(b, num_heads, s_q, head_dim);
            let k_canon = bhsd_shape(b, kh, s_k, head_dim);
            let core = format!("{out_name}_attn");
            self.attention_core(
                &qc,
                &kc,
                &vc,
                &q_canon,
                &k_canon,
                head_dim,
                mask_kind,
                mask_in,
                score_scale,
                softcap,
                &core,
            )?;
            // [B,H,Sq,D] → [B,Sq,H,D] → [B,Sq,H·D]
            let t = format!("{out_name}_ot");
            self.emit(
                "transpose",
                &t,
                &bhsd_shape(b, s_q, num_heads, head_dim),
                vec![
                    ("x", bind_name(&core)),
                    ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                ],
            )?;
            self.emit(
                "reshape",
                out_name,
                &out_shape,
                vec![
                    ("x", bind_name(&t)),
                    ("shape", bind_value(vec_i32(&dims_i32(out_shape.dims())))),
                ],
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        Err(CoremlError::Unsupported(format!(
            "attention: last dim {last} is neither head_dim {head_dim} nor \
             num_heads*head_dim {} (rank {rank})",
            num_heads * head_dim
        )))
    }

    /// Transpose a `[B,S,H,D]` operand to canonical `[B,H,S,D]` (perm `[0,2,1,3]`).
    pub(crate) fn bshd_to_bhsd(
        &mut self,
        in_id: NodeId,
        b: usize,
        s: usize,
        h: usize,
        d: usize,
        prefix: &str,
    ) -> Result<String> {
        let x = self.val(in_id);
        let t = format!("{prefix}_bhsd");
        self.emit(
            "transpose",
            &t,
            &bhsd_shape(b, h, s, d),
            vec![
                ("x", bind_name(&x)),
                ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
            ],
        )?;
        Ok(t)
    }

    /// Reshape+transpose a fused `[B,S,H·D]` operand to canonical `[B,H,S,D]`.
    pub(crate) fn fused_to_bhsd(
        &mut self,
        in_id: NodeId,
        b: usize,
        s: usize,
        h: usize,
        d: usize,
        prefix: &str,
    ) -> Result<String> {
        let x = self.val(in_id);
        let r = format!("{prefix}_r");
        self.emit(
            "reshape",
            &r,
            &bhsd_shape(b, s, h, d),
            vec![
                ("x", bind_name(&x)),
                (
                    "shape",
                    bind_value(vec_i32(&[b as i32, s as i32, h as i32, d as i32])),
                ),
            ],
        )?;
        let t = format!("{prefix}_t");
        self.emit(
            "transpose",
            &t,
            &bhsd_shape(b, h, s, d),
            vec![
                ("x", bind_name(&r)),
                ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
            ],
        )?;
        Ok(t)
    }

    /// [`Self::fused_to_bhsd`] deriving head count + seq from the operand shape
    /// (key/value heads may be fewer than `num_heads` before `repeat_kv`).
    pub(crate) fn fused_to_bhsd_kv(
        &mut self,
        in_id: NodeId,
        b: usize,
        d: usize,
        prefix: &str,
    ) -> Result<(String, usize)> {
        let shape = self.graph.shape(in_id).clone();
        let s = dim_static(&shape, 1)?;
        let h = dim_static(&shape, 2)? / d;
        Ok((self.fused_to_bhsd(in_id, b, s, h, d, prefix)?, s))
    }

    /// Canonical scaled-dot-product attention on `[..,Sq,D]` q/k/v. MIL
    /// `matmul` batches leading dims and the masks broadcast, so one core
    /// serves both the split and (pre-canonicalized) fused paths. Writes
    /// `out_name` in `q_shape`; the caller registers the node name.
    #[allow(clippy::too_many_arguments)]
    /// Add the attention mask to scaled scores `[..,Sq,Sk]`, returning the masked
    /// name (or the input unchanged for `None`). Shared by the forward
    /// `attention_core` and the native attention backward so both build `P` from an
    /// identical pre-softmax tensor.
    pub(crate) fn apply_score_mask(
        &mut self,
        scaled: &str,
        scores_shape: &Shape,
        s_q: usize,
        s_k: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        out_name: &str,
    ) -> Result<String> {
        let mut cur = scaled.to_string();
        match mask_kind {
            MaskKind::None => {}
            MaskKind::Causal => {
                let mask_name = format!("{out_name}_mask");
                let mask = causal_mask(s_q, s_k);
                self.operations.push(make_const(
                    &mut self.blob,
                    &mask_name,
                    &Shape::new(&[s_q, s_k], DType::F32),
                    &mask,
                )?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&mask_name))],
                )?;
                cur = masked;
            }
            MaskKind::Bias => {
                let bias = self.val(mask_in.ok_or_else(|| {
                    CoremlError::Unsupported("attention Bias: missing mask input".into())
                })?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&bias))],
                )?;
                cur = masked;
            }
            MaskKind::Custom => {
                // Key-padding mask `[B, S_k]` (1.0 = keep, <0.5 = drop). Turn
                // it into an additive bias (keep→0, drop→-1e9 via (m-1)*1e9)
                // reshaped to `[B, 1, .., 1, S_k]` so it broadcasts over the
                // score tensor's head/query axes. (Batch is carried at the
                // front; per-utterance ASR uses B=1.)
                let mid = mask_in.ok_or_else(|| {
                    CoremlError::Unsupported("attention Custom: missing mask input".into())
                })?;
                let mask = self.val(mid);
                let mask_shape = self.graph.shape(mid).clone();
                let b = dim_static(&mask_shape, 0)?;
                let sub1 = format!("{out_name}_mk_s");
                self.emit(
                    "sub",
                    &sub1,
                    &mask_shape,
                    vec![("x", bind_name(&mask)), ("y", bind_value(scalar_f32(1.0)))],
                )?;
                let bias_flat = format!("{out_name}_mk_a");
                self.emit(
                    "mul",
                    &bias_flat,
                    &mask_shape,
                    vec![("x", bind_name(&sub1)), ("y", bind_value(scalar_f32(1e9)))],
                )?;
                let mut bd = vec![Dim::Static(1); scores_shape.rank()];
                bd[0] = Dim::Static(b);
                let blast = scores_shape.rank() - 1;
                bd[blast] = Dim::Static(s_k);
                let bshape = Shape::from_dims(&bd, DType::F32);
                let bias = format!("{out_name}_mk_b");
                self.emit(
                    "reshape",
                    &bias,
                    &bshape,
                    vec![
                        ("x", bind_name(&bias_flat)),
                        ("shape", bind_value(vec_i32(&dims_i32(&bd)))),
                    ],
                )?;
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&bias))],
                )?;
                cur = masked;
            }
            MaskKind::SlidingWindow(w) => {
                let mask_name = format!("{out_name}_mask");
                let mask = sliding_window_mask(s_q, s_k, w);
                self.operations.push(make_const(
                    &mut self.blob,
                    &mask_name,
                    &Shape::new(&[s_q, s_k], DType::F32),
                    &mask,
                )?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&mask_name))],
                )?;
                cur = masked;
            }
        }
        Ok(cur)
    }

    pub(crate) fn attention_core(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        q_shape: &Shape,
        k_shape: &Shape,
        head_dim: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        out_name: &str,
    ) -> Result<()> {
        let rank = q_shape.rank();
        let s_q = dim_static(q_shape, rank - 2)?;
        let s_k = dim_static(k_shape, k_shape.rank() - 2)?;
        let scores_shape = {
            let mut d = q_shape.dims().to_vec();
            d[rank - 1] = Dim::Static(s_k); // [..,Sq,Sk]
            Shape::from_dims(&d, DType::F32)
        };
        let scale = score_scale.unwrap_or((head_dim as f32).powf(-0.5));

        // raw = q @ kᵀ  (transpose_y batches over [B,H])
        let raw = format!("{out_name}_qk");
        self.emit(
            "matmul",
            &raw,
            &scores_shape,
            vec![
                ("x", bind_name(q)),
                ("y", bind_name(k)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        // scaled = raw * scale
        let cur = format!("{out_name}_sc");
        self.emit(
            "mul",
            &cur,
            &scores_shape,
            vec![("x", bind_name(&raw)), ("y", bind_value(scalar_f32(scale)))],
        )?;

        // mask (factored so the native attention backward recomputes P identically)
        let mut cur =
            self.apply_score_mask(&cur, &scores_shape, s_q, s_k, mask_kind, mask_in, out_name)?;

        // softcap: cap * tanh(scores / cap)
        if let Some(cap) = softcap {
            if cap > 0.0 {
                let div = format!("{out_name}_cap_div");
                self.emit(
                    "mul",
                    &div,
                    &scores_shape,
                    vec![
                        ("x", bind_name(&cur)),
                        ("y", bind_value(scalar_f32(1.0 / cap))),
                    ],
                )?;
                let th = format!("{out_name}_cap_tanh");
                self.emit("tanh", &th, &scores_shape, vec![("x", bind_name(&div))])?;
                let capped = format!("{out_name}_cap");
                self.emit(
                    "mul",
                    &capped,
                    &scores_shape,
                    vec![("x", bind_name(&th)), ("y", bind_value(scalar_f32(cap)))],
                )?;
                cur = capped;
            }
        }

        // probs = softmax(cur, axis=-1)
        let probs = format!("{out_name}_p");
        self.emit(
            "softmax",
            &probs,
            &scores_shape,
            vec![("x", bind_name(&cur)), ("axis", bind_value(scalar_i32(-1)))],
        )?;

        // out = probs @ v  -> [..,Sq,D]
        self.emit(
            "matmul",
            out_name,
            q_shape,
            vec![
                ("x", bind_name(&probs)),
                ("y", bind_name(v)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(false))),
            ],
        )?;
        Ok(())
    }

    /// Fused scaled-dot-product attention backward (`dQ`/`dK`/`dV`). Canonicalizes
    /// any of the three operand layouts the forward accepts — `[B,H,S,D]`,
    /// `[B,S,H,D]`, fused `[B,S,H·D]` — to `[B,H,S,D]`, runs
    /// [`attention_backward_core`](Self::attention_backward_core) (every mask kind via
    /// the shared [`apply_score_mask`](Self::apply_score_mask)), then maps the
    /// gradient back to the `wrt` operand's layout. MHA only (q/k/v share the head
    /// count); GQA (`kv heads ≠ num_heads`) returns `Unsupported`.
    #[cfg(feature = "training")]
    pub(crate) fn lower_attention_backward(
        &mut self,
        id: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        wrt: rlx_ir::op::AttentionBwdWrt,
        out_name: &str,
    ) -> Result<()> {
        use rlx_ir::op::AttentionBwdWrt;
        let (q_in, k_in, v_in, dy_in, mask_in, q_shape, k_shape, out_shape) = {
            let node = self.graph.node(id);
            let mask_in = match mask_kind {
                MaskKind::Bias | MaskKind::Custom => node.inputs.get(4).copied(),
                _ => None,
            };
            (
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                node.inputs[3],
                mask_in,
                self.graph.shape(node.inputs[0]).clone(),
                self.graph.shape(node.inputs[1]).clone(),
                node.shape.clone(),
            )
        };
        let (h, d) = (num_heads, head_dim);
        let rank = q_shape.rank();
        let last = dim_static(&q_shape, rank - 1)?;
        // Gradient sequence length depends on which operand we differentiate.
        let s_wrt_of = |s_q: usize, s_k: usize| match wrt {
            AttentionBwdWrt::Query => s_q,
            AttentionBwdWrt::Key | AttentionBwdWrt::Value => s_k,
        };

        if rank == 4 && last == d {
            let geom = rlx_ir::attention_geom(&q_shape, &k_shape, num_heads, head_dim);
            let (b, s_q, s_k) = (geom.batch, geom.seq_q, geom.seq_k);
            let k_heads = if geom.bhsd {
                dim_static(&k_shape, 1)?
            } else {
                dim_static(&k_shape, 2)?
            };
            if k_heads != num_heads {
                return Err(CoremlError::Unsupported(
                    "attention backward: GQA (kv heads ≠ num_heads) not supported".into(),
                ));
            }
            if geom.bhsd {
                // Canonical `[B,H,S,D]` — compute straight into `out_name`.
                let (q, k, v, dy) = (
                    self.val(q_in),
                    self.val(k_in),
                    self.val(v_in),
                    self.val(dy_in),
                );
                self.attention_backward_core(
                    &q, &k, &v, &dy, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, out_name,
                )?;
            } else {
                // `[B,S,H,D]` → canonical, compute, transpose the gradient back.
                let qc = self.bshd_to_bhsd(q_in, b, s_q, h, d, &format!("{out_name}_qc"))?;
                let kc = self.bshd_to_bhsd(k_in, b, s_k, h, d, &format!("{out_name}_kc"))?;
                let vc = self.bshd_to_bhsd(v_in, b, s_k, h, d, &format!("{out_name}_vc"))?;
                let dyc = self.bshd_to_bhsd(dy_in, b, s_q, h, d, &format!("{out_name}_dyc"))?;
                let core = format!("{out_name}_core");
                self.attention_backward_core(
                    &qc, &kc, &vc, &dyc, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, &core,
                )?;
                let s_wrt = s_wrt_of(s_q, s_k);
                self.emit(
                    "transpose",
                    out_name,
                    &bhsd_shape(b, s_wrt, h, d),
                    vec![
                        ("x", bind_name(&core)),
                        ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                    ],
                )?;
            }
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        if rank == 3 && num_heads > 0 && last == num_heads * head_dim {
            // Fused `[B,S,H·D]` → canonical, compute, transpose + reshape back.
            let b = dim_static(&q_shape, 0)?;
            let s_q = dim_static(&q_shape, 1)?;
            let s_k = dim_static(&k_shape, 1)?;
            if dim_static(&k_shape, 2)? / d != num_heads {
                return Err(CoremlError::Unsupported(
                    "attention backward: fused GQA (kv heads ≠ num_heads) not supported".into(),
                ));
            }
            let qc = self.fused_to_bhsd(q_in, b, s_q, h, d, &format!("{out_name}_qc"))?;
            let kc = self.fused_to_bhsd(k_in, b, s_k, h, d, &format!("{out_name}_kc"))?;
            let vc = self.fused_to_bhsd(v_in, b, s_k, h, d, &format!("{out_name}_vc"))?;
            let dyc = self.fused_to_bhsd(dy_in, b, s_q, h, d, &format!("{out_name}_dyc"))?;
            let core = format!("{out_name}_core");
            self.attention_backward_core(
                &qc, &kc, &vc, &dyc, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, &core,
            )?;
            let s_wrt = s_wrt_of(s_q, s_k);
            let t = format!("{out_name}_t");
            self.emit(
                "transpose",
                &t,
                &bhsd_shape(b, s_wrt, h, d),
                vec![
                    ("x", bind_name(&core)),
                    ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                ],
            )?;
            self.emit(
                "reshape",
                out_name,
                &out_shape,
                vec![
                    ("x", bind_name(&t)),
                    (
                        "shape",
                        bind_value(vec_i32(&[b as i32, s_wrt as i32, (h * d) as i32])),
                    ),
                ],
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        Err(CoremlError::Unsupported(format!(
            "attention backward: unsupported operand layout (rank {rank}, last {last})"
        )))
    }

    /// Canonical `[B,H,S,D]` attention backward, emitting the single `wrt` gradient
    /// to `result`. Recompute `P = softmax(scale·QKᵀ [+ mask])`, then:
    ///   `dV = Pᵀ·dO`,  `dP = dO·Vᵀ`,
    ///   `ds = scale · P⊙(dP − rowsum(P⊙dP))` (softmax-Jacobian–vector product),
    ///   `dQ = ds·K`,  `dK = dsᵀ·Q`.
    /// Masked positions get `P≈0`, so `ds` there vanishes automatically.
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_backward_core(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        dy: &str,
        b: usize,
        h: usize,
        s_q: usize,
        s_k: usize,
        d: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        wrt: rlx_ir::op::AttentionBwdWrt,
        result: &str,
    ) -> Result<()> {
        use rlx_ir::op::AttentionBwdWrt;
        let scores_shape = bhsd_shape(b, h, s_q, s_k);
        let scale = (d as f32).powf(-0.5);

        let raw = format!("{result}_qk");
        self.emit(
            "matmul",
            &raw,
            &scores_shape,
            vec![
                ("x", bind_name(q)),
                ("y", bind_name(k)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        let scaled = format!("{result}_scl");
        self.emit(
            "mul",
            &scaled,
            &scores_shape,
            vec![("x", bind_name(&raw)), ("y", bind_value(scalar_f32(scale)))],
        )?;
        let pre =
            self.apply_score_mask(&scaled, &scores_shape, s_q, s_k, mask_kind, mask_in, result)?;
        let p = format!("{result}_p");
        self.emit(
            "softmax",
            &p,
            &scores_shape,
            vec![("x", bind_name(&pre)), ("axis", bind_value(scalar_i32(-1)))],
        )?;

        match wrt {
            AttentionBwdWrt::Value => {
                // dV = Pᵀ · dO  → [B,H,Sk,D]
                self.emit(
                    "matmul",
                    result,
                    &bhsd_shape(b, h, s_k, d),
                    vec![
                        ("x", bind_name(&p)),
                        ("y", bind_name(dy)),
                        ("transpose_x", bind_value(scalar_bool(true))),
                        ("transpose_y", bind_value(scalar_bool(false))),
                    ],
                )?;
            }
            AttentionBwdWrt::Query | AttentionBwdWrt::Key => {
                let dp = format!("{result}_dp");
                self.emit(
                    "matmul",
                    &dp,
                    &scores_shape,
                    vec![
                        ("x", bind_name(dy)),
                        ("y", bind_name(v)),
                        ("transpose_x", bind_value(scalar_bool(false))),
                        ("transpose_y", bind_value(scalar_bool(true))),
                    ],
                )?;
                let pdp = format!("{result}_pdp");
                self.emit(
                    "mul",
                    &pdp,
                    &scores_shape,
                    vec![("x", bind_name(&dp)), ("y", bind_name(&p))],
                )?;
                let rowsum = format!("{result}_rs");
                self.emit(
                    "reduce_sum",
                    &rowsum,
                    &bhsd_shape(b, h, s_q, 1),
                    vec![
                        ("x", bind_name(&pdp)),
                        ("axes", bind_value(vec_i32(&[3]))),
                        ("keep_dims", bind_value(scalar_bool(true))),
                    ],
                )?;
                let dpm = format!("{result}_dpm");
                self.emit(
                    "sub",
                    &dpm,
                    &scores_shape,
                    vec![("x", bind_name(&dp)), ("y", bind_name(&rowsum))],
                )?;
                let dsm = format!("{result}_dsm");
                self.emit(
                    "mul",
                    &dsm,
                    &scores_shape,
                    vec![("x", bind_name(&p)), ("y", bind_name(&dpm))],
                )?;
                let ds = format!("{result}_ds");
                self.emit(
                    "mul",
                    &ds,
                    &scores_shape,
                    vec![("x", bind_name(&dsm)), ("y", bind_value(scalar_f32(scale)))],
                )?;
                match wrt {
                    AttentionBwdWrt::Query => {
                        // dQ = ds · K  → [B,H,Sq,D]
                        self.emit(
                            "matmul",
                            result,
                            &bhsd_shape(b, h, s_q, d),
                            vec![
                                ("x", bind_name(&ds)),
                                ("y", bind_name(k)),
                                ("transpose_x", bind_value(scalar_bool(false))),
                                ("transpose_y", bind_value(scalar_bool(false))),
                            ],
                        )?;
                    }
                    AttentionBwdWrt::Key => {
                        // dK = dsᵀ · Q  → [B,H,Sk,D]
                        self.emit(
                            "matmul",
                            result,
                            &bhsd_shape(b, h, s_k, d),
                            vec![
                                ("x", bind_name(&ds)),
                                ("y", bind_name(q)),
                                ("transpose_x", bind_value(scalar_bool(true))),
                                ("transpose_y", bind_value(scalar_bool(false))),
                            ],
                        )?;
                    }
                    AttentionBwdWrt::Value => unreachable!(),
                }
            }
        }
        Ok(())
    }
}
