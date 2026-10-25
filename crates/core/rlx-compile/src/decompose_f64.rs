// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Decompose F64 composites for double-single** (`NumberFormat::F64` on GPUs).
//!
//! Double-single kernels exist for a small set of primitives (elementwise
//! add/sub/mul/div/max, sqrt, exp, matmul, sum and max reductions; see
//! [`rlx_ir::double_single::DfOp`]). [`decompose_f64`] rewrites the F64 ops
//! models use on top of those — RMS/layer norm, softmax, SiLU / sigmoid /
//! tanh / tanh-GELU, RoPE and attention — into them, before
//! [`crate::lower_f64`] pairs the graph. The semantics are those of the CPU's
//! native f64 kernels (`rlx_cpu::f64_kernels`): the same layouts, grouped KV
//! heads, mask kinds and softcap.
//!
//! Masked attention scores get a large negative bias added rather than being
//! replaced by it, which only differs when every key of a query is masked.
//! `exp` arguments are clamped to 80 where the result feeds a division
//! (double-single `x / inf` is NaN); that changes values below ~1e-35.

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp, RopeStyle};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Rewrite `g`'s F64 composites into double-single primitives. Ops on other
/// dtypes, and F64 ops that are already primitive, are copied unchanged.
pub fn decompose_f64(g: &Graph) -> Result<Graph, String> {
    let mut b = B {
        g: Graph::new(&g.name),
    };
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in g.nodes() {
        let ins: Vec<NodeId> = n.inputs.iter().map(|i| map[i]).collect();
        let id = if n.shape.dtype() == DType::F64 {
            b.lower(&n.op, &ins, &n.shape)?
        } else {
            None
        };
        let id = match id {
            Some(id) => id,
            None => b.g.add_node(n.op.clone(), ins, n.shape.clone()),
        };
        map.insert(n.id, id);
    }
    let outs = g.outputs.iter().map(|o| map[o]).collect();
    b.g.set_outputs(outs);
    Ok(b.g)
}

/// An f32 op attribute as the decimal it was written as: `1e-6f32 as f64` is
/// not `1e-6`.
fn attr(x: f32) -> f64 {
    x.to_string().parse().unwrap_or(x as f64)
}

/// Largest magnitude an `exp` argument may have where the result is divided by
/// (keeps double-single away from `x / inf`).
const EXP_CLAMP: f64 = 80.0;

struct B {
    g: Graph,
}

impl B {
    fn dims(&self, x: NodeId) -> Result<Vec<usize>, String> {
        self.g
            .node(x)
            .shape
            .dims()
            .iter()
            .map(|d| {
                d.is_static()
                    .then(|| d.unwrap_static())
                    .ok_or_else(|| "decompose_f64: dynamic shapes are not supported".to_string())
            })
            .collect()
    }

    fn f64(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F64)
    }

    fn constant(&mut self, vals: &[f64], dims: &[usize]) -> NodeId {
        let data = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.g
            .add_node(Op::Constant { data }, vec![], Self::f64(dims))
    }

    fn scalar(&mut self, v: f64) -> NodeId {
        self.constant(&[v], &[1])
    }

    fn bin(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let (da, db) = (self.dims(a)?, self.dims(b)?);
        let r = da.len().max(db.len());
        let pad = |d: &[usize]| {
            let mut v = vec![1; r - d.len()];
            v.extend_from_slice(d);
            v
        };
        let (pa, pb) = (pad(&da), pad(&db));
        let out: Vec<usize> = pa
            .iter()
            .zip(&pb)
            .map(|(&x, &y)| match (x, y) {
                _ if x == y => Ok(x),
                (1, _) => Ok(y),
                (_, 1) => Ok(x),
                _ => Err(format!(
                    "decompose_f64: cannot broadcast {da:?} with {db:?}"
                )),
            })
            .collect::<Result<_, _>>()?;
        Ok(self.g.add_node(Op::Binary(op), vec![a, b], Self::f64(&out)))
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        self.bin(BinaryOp::Add, a, b)
    }
    fn sub(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        self.bin(BinaryOp::Sub, a, b)
    }
    fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        self.bin(BinaryOp::Mul, a, b)
    }
    fn div(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        self.bin(BinaryOp::Div, a, b)
    }
    fn mul_s(&mut self, a: NodeId, v: f64) -> Result<NodeId, String> {
        let c = self.scalar(v);
        self.mul(a, c)
    }

    fn unary(&mut self, act: Activation, x: NodeId) -> Result<NodeId, String> {
        let d = self.dims(x)?;
        Ok(self.g.add_node(Op::Activation(act), vec![x], Self::f64(&d)))
    }

    /// `e^x` with `x` clamped to at most `EXP_CLAMP`.
    fn exp_clamped(&mut self, x: NodeId) -> Result<NodeId, String> {
        // min(x, c) = -max(-x, -c)
        let neg = self.mul_s(x, -1.0)?;
        let lim = self.scalar(-EXP_CLAMP);
        let m = self.bin(BinaryOp::Max, neg, lim)?;
        let back = self.mul_s(m, -1.0)?;
        self.unary(Activation::Exp, back)
    }

    fn reduce(&mut self, op: ReduceOp, x: NodeId, axis: usize) -> Result<NodeId, String> {
        let mut d = self.dims(x)?;
        d[axis] = 1;
        Ok(self.g.add_node(
            Op::Reduce {
                op,
                axes: vec![axis],
                keep_dim: true,
            },
            vec![x],
            Self::f64(&d),
        ))
    }

    fn reshape(&mut self, x: NodeId, dims: &[usize]) -> NodeId {
        self.g.add_node(
            Op::Reshape {
                new_shape: dims.iter().map(|&d| d as i64).collect(),
            },
            vec![x],
            Self::f64(dims),
        )
    }

    fn transpose(&mut self, x: NodeId, perm: &[usize]) -> Result<NodeId, String> {
        let d = self.dims(x)?;
        let out: Vec<usize> = perm.iter().map(|&p| d[p]).collect();
        Ok(self.g.add_node(
            Op::Transpose {
                perm: perm.to_vec(),
            },
            vec![x],
            Self::f64(&out),
        ))
    }

    fn narrow(
        &mut self,
        x: NodeId,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<NodeId, String> {
        let mut d = self.dims(x)?;
        d[axis] = len;
        Ok(self
            .g
            .add_node(Op::Narrow { axis, start, len }, vec![x], Self::f64(&d)))
    }

    fn concat(&mut self, xs: Vec<NodeId>, axis: usize) -> Result<NodeId, String> {
        let mut d = self.dims(xs[0])?;
        d[axis] = xs
            .iter()
            .map(|&x| self.dims(x).map(|v| v[axis]))
            .sum::<Result<_, _>>()?;
        Ok(self.g.add_node(Op::Concat { axis }, xs, Self::f64(&d)))
    }

    fn expand(&mut self, x: NodeId, dims: &[usize]) -> NodeId {
        self.g.add_node(
            Op::Expand {
                target_shape: dims.iter().map(|&d| d as i64).collect(),
            },
            vec![x],
            Self::f64(dims),
        )
    }

    fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, String> {
        let (da, db) = (self.dims(a)?, self.dims(b)?);
        let mut out = da.clone();
        *out.last_mut().unwrap() = *db.last().unwrap();
        Ok(self.g.add_node(Op::MatMul, vec![a, b], Self::f64(&out)))
    }

    fn axis(&self, x: NodeId, axis: i32) -> Result<usize, String> {
        let r = self.dims(x)?.len() as i32;
        let a = if axis < 0 { axis + r } else { axis };
        (0..r)
            .contains(&a)
            .then_some(a as usize)
            .ok_or_else(|| format!("decompose_f64: axis {axis} out of range for rank {r}"))
    }

    /// `Some(id)` when `op` is decomposed, `None` to copy it unchanged.
    fn lower(&mut self, op: &Op, ins: &[NodeId], shape: &Shape) -> Result<Option<NodeId>, String> {
        let id = match op {
            Op::RmsNorm { axis, eps } => {
                let x = ins[0];
                let ax = self.axis(x, *axis)?;
                let n = self.dims(x)?[ax] as f64;
                let sq = self.mul(x, x)?;
                let ms = self.reduce(ReduceOp::Sum, sq, ax)?;
                let ms = self.mul_s(ms, 1.0 / n)?;
                let e = self.scalar(attr(*eps));
                let v = self.add(ms, e)?;
                let r = self.unary(Activation::Sqrt, v)?;
                let y = self.div(x, r)?;
                let y = self.mul(y, ins[1])?;
                self.add(y, ins[2])?
            }
            Op::LayerNorm { axis, eps } => {
                let x = ins[0];
                let ax = self.axis(x, *axis)?;
                let n = self.dims(x)?[ax] as f64;
                let s = self.reduce(ReduceOp::Sum, x, ax)?;
                let mean = self.mul_s(s, 1.0 / n)?;
                let c = self.sub(x, mean)?;
                let sq = self.mul(c, c)?;
                let vs = self.reduce(ReduceOp::Sum, sq, ax)?;
                let var = self.mul_s(vs, 1.0 / n)?;
                let e = self.scalar(attr(*eps));
                let v = self.add(var, e)?;
                let r = self.unary(Activation::Sqrt, v)?;
                let y = self.div(c, r)?;
                let y = self.mul(y, ins[1])?;
                self.add(y, ins[2])?
            }
            Op::Softmax { axis } => {
                let x = ins[0];
                let ax = self.axis(x, *axis)?;
                self.softmax(x, ax)?
            }
            Op::Activation(a) => match a {
                Activation::Silu => {
                    let s = self.sigmoid(ins[0])?;
                    self.mul(ins[0], s)?
                }
                Activation::Sigmoid => self.sigmoid(ins[0])?,
                Activation::Tanh => self.tanh(ins[0])?,
                Activation::GeluApprox => {
                    // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
                    let x = ins[0];
                    let x2 = self.mul(x, x)?;
                    let x3 = self.mul(x2, x)?;
                    let t = self.mul_s(x3, 0.044715)?;
                    let inner = self.add(x, t)?;
                    let z = self.mul_s(inner, (2.0 / std::f64::consts::PI).sqrt())?;
                    let th = self.tanh(z)?;
                    let one = self.scalar(1.0);
                    let p = self.add(th, one)?;
                    let h = self.mul_s(x, 0.5)?;
                    self.mul(h, p)?
                }
                Activation::Neg => self.mul_s(ins[0], -1.0)?,
                Activation::Exp | Activation::Sqrt => return Ok(None),
                other => {
                    return Err(format!(
                        "decompose_f64: no double-single form for activation {other:?}"
                    ));
                }
            },
            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => self.rope(ins, *head_dim, *n_rot, *style, shape)?,
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => self.attention(
                ins,
                *num_heads,
                *head_dim,
                v_head_dim.unwrap_or(*head_dim),
                *mask_kind,
                *score_scale,
                *attn_logit_softcap,
                shape,
            )?,
            _ => return Ok(None),
        };
        Ok(Some(id))
    }

    fn softmax(&mut self, x: NodeId, ax: usize) -> Result<NodeId, String> {
        let m = self.reduce(ReduceOp::Max, x, ax)?;
        let c = self.sub(x, m)?;
        let e = self.unary(Activation::Exp, c)?;
        let s = self.reduce(ReduceOp::Sum, e, ax)?;
        self.div(e, s)
    }

    /// `1 / (1 + e^-x)`, the exponent clamped.
    fn sigmoid(&mut self, x: NodeId) -> Result<NodeId, String> {
        let n = self.mul_s(x, -1.0)?;
        let e = self.exp_clamped(n)?;
        let one = self.scalar(1.0);
        let d = self.add(one, e)?;
        let one = self.scalar(1.0);
        self.div(one, d)
    }

    /// `1 - 2 / (e^{2x} + 1)`, the exponent clamped.
    fn tanh(&mut self, x: NodeId) -> Result<NodeId, String> {
        let t = self.mul_s(x, 2.0)?;
        let e = self.exp_clamped(t)?;
        let one = self.scalar(1.0);
        let d = self.add(e, one)?;
        let two = self.scalar(2.0);
        let q = self.div(two, d)?;
        let one = self.scalar(1.0);
        self.sub(one, q)
    }

    /// `rlx_cpu`'s RoPE: `x` read as `[tokens, hidden]` from its last two axes
    /// (`[batch.., seq, hidden]`), per-token tables when they have one row per
    /// token, else row `t % seq`; the first `n_rot` lanes of each head rotate.
    fn rope(
        &mut self,
        ins: &[NodeId],
        dh: usize,
        n_rot: usize,
        style: RopeStyle,
        shape: &Shape,
    ) -> Result<NodeId, String> {
        let (x, cos, sin) = (ins[0], ins[1], ins[2]);
        let d = self.dims(x)?;
        let (batch, seq, hidden): (usize, usize, usize) = if d.len() >= 3 {
            let r = d.len();
            (d[..r - 2].iter().product(), d[r - 2], d[r - 1])
        } else {
            let n: usize = d.iter().product();
            (1, d[0], n / d[0].max(1))
        };
        let half = n_rot / 2;
        let heads = hidden / dh;
        let tokens = batch * seq;
        let table = |b: &mut B, t: NodeId| -> Result<NodeId, String> {
            let td = b.dims(t)?;
            let stride = *td.last().unwrap();
            let rows: usize = td.iter().product::<usize>() / stride.max(1);
            let t2 = b.reshape(t, &[rows, stride]);
            let t2 = b.narrow(t2, 1, 0, half)?;
            let per_token = rows == tokens && rows != seq;
            let rows_t = if per_token {
                t2
            } else {
                if rows < seq {
                    return Err(format!(
                        "decompose_f64: rope table has {rows} rows for {seq}"
                    ));
                }
                let t2 = b.narrow(t2, 0, 0, seq)?;
                let t3 = b.reshape(t2, &[1, seq, half]);
                let t3 = b.expand(t3, &[batch, seq, half]);
                b.reshape(t3, &[tokens, half])
            };
            Ok(b.reshape(rows_t, &[tokens, 1, half]))
        };
        let (c, s) = (table(self, cos)?, table(self, sin)?);
        let xh = self.reshape(x, &[tokens, heads, dh]);
        let rot = self.narrow(xh, 2, 0, n_rot)?;
        let rotated = match style {
            RopeStyle::NeoX => {
                let a = self.narrow(rot, 2, 0, half)?;
                let bb = self.narrow(rot, 2, half, half)?;
                let ac = self.mul(a, c)?;
                let bs = self.mul(bb, s)?;
                let oa = self.sub(ac, bs)?;
                let bc = self.mul(bb, c)?;
                let as_ = self.mul(a, s)?;
                let ob = self.add(bc, as_)?;
                self.concat(vec![oa, ob], 2)?
            }
            RopeStyle::GptJ => {
                let p = self.reshape(rot, &[tokens, heads, half, 2]);
                let a = self.narrow(p, 3, 0, 1)?;
                let bb = self.narrow(p, 3, 1, 1)?;
                let c4 = self.reshape(c, &[tokens, 1, half, 1]);
                let s4 = self.reshape(s, &[tokens, 1, half, 1]);
                let ac = self.mul(a, c4)?;
                let bs = self.mul(bb, s4)?;
                let oa = self.sub(ac, bs)?;
                let bc = self.mul(bb, c4)?;
                let as_ = self.mul(a, s4)?;
                let ob = self.add(bc, as_)?;
                let o = self.concat(vec![oa, ob], 3)?;
                self.reshape(o, &[tokens, heads, n_rot])
            }
        };
        let full = if n_rot < dh {
            let rest = self.narrow(xh, 2, n_rot, dh - n_rot)?;
            self.concat(vec![rotated, rest], 2)?
        } else {
            rotated
        };
        let out: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        Ok(self.reshape(full, &out))
    }

    /// `rlx_cpu`'s attention over `[B,S,H·D]`, `[B,S,H,D]`, `[B,H,S,D]` or
    /// `[S,H·D]`, with grouped KV heads, every mask kind and softcap.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        ins: &[NodeId],
        nh: usize,
        dh: usize,
        dv: usize,
        mask: MaskKind,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        shape: &Shape,
    ) -> Result<NodeId, String> {
        let (q, k, v) = (ins[0], ins[1], ins[2]);
        let (qd, kd) = (self.dims(q)?, self.dims(k)?);
        let (b, qs, ks, bhsd) = match qd.len() {
            4 if qd[1] == nh => (qd[0], qd[2], kd[2], true),
            4 => (qd[0], qd[1], kd[1], false),
            3 => (qd[0], qd[1], kd[1], false),
            _ => (1, qd[0], kd[0], false),
        };
        let k_len: usize = kd.iter().product();
        let nkv = (k_len / (b * ks * dh).max(1)).max(1);
        let group = (nh / nkv).max(1);
        // To [b, heads, s, w].
        let heads_first = |bd: &mut B, t: NodeId, n: usize, s: usize, w: usize| {
            if bhsd {
                Ok(bd.reshape(t, &[b, n, s, w]))
            } else {
                let r = bd.reshape(t, &[b, s, n, w]);
                bd.transpose(r, &[0, 2, 1, 3])
            }
        };
        let qh = heads_first(self, q, nh, qs, dh)?;
        let mut kh = heads_first(self, k, nkv, ks, dh)?;
        let mut vh = heads_first(self, v, nkv, ks, dv)?;
        if group > 1 {
            for (t, w) in [(&mut kh, dh), (&mut vh, dv)] {
                let r = self.reshape(*t, &[b, nkv, 1, ks, w]);
                let e = self.expand(r, &[b, nkv, group, ks, w]);
                *t = self.reshape(e, &[b, nh, ks, w]);
            }
        }
        let kt = self.transpose(kh, &[0, 1, 3, 2])?;
        let scores = self.matmul(qh, kt)?;
        // The CPU reads an f32 `score_scale` that equals hd^-0.5 exactly.
        let scale = match score_scale {
            Some(s) if s != (dh as f32).powf(-0.5) => attr(s),
            _ => (dh as f64).powf(-0.5),
        };
        let mut s = self.mul_s(scores, scale)?;
        let cfg_neg = -1e30;
        let q_off = ks.saturating_sub(qs);
        match mask {
            MaskKind::None => {}
            MaskKind::Bias => s = self.add(s, ins[3])?,
            MaskKind::Causal | MaskKind::SlidingWindow(_) => {
                let w = match mask {
                    MaskKind::SlidingWindow(w) => Some(w),
                    _ => None,
                };
                let bias: Vec<f64> = (0..qs * ks)
                    .map(|i| {
                        let (qi, ki) = (i / ks, i % ks);
                        let aq = q_off + qi;
                        let masked = ki > aq || w.is_some_and(|w| ki < aq.saturating_sub(w));
                        if masked { cfg_neg } else { 0.0 }
                    })
                    .collect();
                let c = self.constant(&bias, &[1, 1, qs, ks]);
                s = self.add(s, c)?;
            }
            MaskKind::Custom => {
                // Valid where mask ≥ 0.5 (the CPU's `mask_binary_threshold`),
                // compared in f32; the bias is (valid - 1)·1e30.
                let m = ins[3];
                let md = self.dims(m)?;
                let m32 = self.g.add_node(
                    Op::Cast { to: DType::F32 },
                    vec![m],
                    Shape::new(&md, DType::F32),
                );
                let half = self.g.add_node(
                    Op::Constant {
                        data: 0.5f32.to_le_bytes().to_vec(),
                    },
                    vec![],
                    Shape::new(&[1], DType::F32),
                );
                let keep = self.g.add_node(
                    Op::Compare(CmpOp::Ge),
                    vec![m32, half],
                    Shape::new(&md, DType::Bool),
                );
                let keep = self.g.add_node(
                    Op::Cast { to: DType::F32 },
                    vec![keep],
                    Shape::new(&md, DType::F32),
                );
                let keep = self
                    .g
                    .add_node(Op::Cast { to: DType::F64 }, vec![keep], Self::f64(&md));
                let one = self.scalar(1.0);
                let off = self.sub(keep, one)?;
                let bias = self.mul_s(off, -cfg_neg)?;
                let bias = self.reshape(bias, &[b, 1, 1, ks]);
                s = self.add(s, bias)?;
            }
        }
        if let Some(c) = softcap.filter(|&c| c > 0.0) {
            let c = attr(c);
            let z = self.mul_s(s, 1.0 / c)?;
            let t = self.tanh(z)?;
            s = self.mul_s(t, c)?;
        }
        let p = self.softmax(s, 3)?;
        let o = self.matmul(p, vh)?;
        let o = if bhsd {
            o
        } else {
            self.transpose(o, &[0, 2, 1, 3])?
        };
        let out: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        Ok(self.reshape(o, &out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::Activation;

    /// Every composite a transformer uses lowers to double-single once
    /// decomposed (numerics: `rlx-runtime/tests/f64_number_format.rs`).
    #[test]
    fn composites_lower_after_decomposition() {
        let f = DType::F64;
        let sh = |d: &[usize]| Shape::new(d, f);
        let mut g = Graph::new("t");
        let x = g.input("x", sh(&[1, 4, 8]));
        let gamma = g.param("g", sh(&[8]));
        let beta = g.param("b", sh(&[8]));
        let n = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            sh(&[1, 4, 8]),
        );
        let ln = g.add_node(
            Op::LayerNorm {
                axis: -1,
                eps: 1e-5,
            },
            vec![n, gamma, beta],
            sh(&[1, 4, 8]),
        );
        let cos = g.param("cos", sh(&[16, 2]));
        let sin = g.param("sin", sh(&[16, 2]));
        let r = g.add_node(
            Op::Rope {
                head_dim: 4,
                n_rot: 4,
                style: RopeStyle::NeoX,
            },
            vec![ln, cos, sin],
            sh(&[1, 4, 8]),
        );
        let a = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 4,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: Some(30.0),
            },
            vec![r, r, n],
            sh(&[1, 4, 8]),
        );
        let s = g.activation(Activation::Silu, a, sh(&[1, 4, 8]));
        let ge = g.activation(Activation::GeluApprox, s, sh(&[1, 4, 8]));
        let sm = g.add_node(Op::Softmax { axis: -1 }, vec![ge], sh(&[1, 4, 8]));
        g.set_outputs(vec![sm]);
        assert!(
            crate::lower_f64::lower_f64(&g).is_err(),
            "composites need decomposing"
        );
        let d = decompose_f64(&g).unwrap();
        crate::lower_f64::lower_f64(&d).unwrap();
    }
}
