// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! F64 kernels for the composite ops a transformer is built from — RmsNorm,
//! LayerNorm, Softmax, Rope and Attention — so a graph promoted to F64
//! (`NumberFormat::F64`) computes in f64 end to end on the CPU.
//!
//! The f32 kernels stay as they are; these are plain, direct loops with the
//! same semantics (layouts, mask kinds, partial rotation, GQA). They ride the
//! `Thunk::CustomOp` path like the SPD kernels. Unlike the f32 attention they
//! never skip small softmax weights: F64 is for reference results.
//!
//! [`native_f64`] lists every op the CPU runs in f64. An F64 node outside it
//! would be handed to an f32 kernel reading f64 buffers, so callers that
//! produce F64 graphs check it first.

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};
use rlx_ir::op::{MaskKind, ReduceOp};
use rlx_ir::{Op, Shape};

fn dims(s: &Shape) -> Vec<usize> {
    s.dims().iter().map(|d| d.unwrap_static()).collect()
}

/// Can the CPU run `op` with F64 values (inputs and output)?
pub fn native_f64(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Cast { .. }
            | Op::Binary(_)
            | Op::Activation(_)
            | Op::MatMul
            | Op::Reshape { .. }
            | Op::StopGradient
            | Op::Transpose { .. }
            | Op::Expand { .. }
            | Op::Narrow { .. }
            | Op::Concat { .. }
            | Op::Gather { .. }
            | Op::KvAppend { .. }
            | Op::Reduce {
                op: ReduceOp::Sum,
                ..
            }
            | Op::RmsNorm { .. }
            | Op::LayerNorm { .. }
            | Op::Softmax { .. }
            | Op::Rope { .. }
            | Op::Attention { .. }
    )
}

/// An f32 op attribute (eps, scale, softcap) as the f64 it was written as.
///
/// `1e-6_f32 as f64` is 9.9999999747e-7: widening keeps the f32 rounding,
/// which then shows up in every f64 result. The shortest decimal that
/// round-trips the f32 is the value the model author wrote.
fn attr(x: f32) -> f64 {
    x.to_string().parse().unwrap_or(x as f64)
}

/// The f64 kernel for `op`, when it is one of the composites here.
pub(crate) fn kernel_for(op: &Op, in_shapes: &[&Shape]) -> Option<std::sync::Arc<dyn CpuKernel>> {
    use std::sync::Arc;
    Some(match op {
        Op::RmsNorm { eps, .. } => Arc::new(NormF64 {
            eps: attr(*eps),
            center: false,
        }),
        Op::LayerNorm { eps, .. } => Arc::new(NormF64 {
            eps: attr(*eps),
            center: true,
        }),
        Op::Softmax { axis } => Arc::new(SoftmaxF64 { axis: *axis }),
        Op::Rope {
            head_dim,
            n_rot,
            style,
        } => Arc::new(RopeF64 {
            head_dim: *head_dim,
            n_rot: *n_rot,
            interleaved: matches!(style, rlx_ir::op::RopeStyle::GptJ),
            table_stride: rlx_ir::shape::rope_table_stride(in_shapes[1], *n_rot),
        }),
        Op::Attention {
            num_heads,
            head_dim,
            v_head_dim,
            mask_kind,
            score_scale,
            attn_logit_softcap,
        } => Arc::new(AttentionF64 {
            heads: *num_heads,
            head_dim: *head_dim,
            v_head_dim: v_head_dim.unwrap_or(*head_dim),
            mask_kind: *mask_kind,
            scale: {
                let exact = (*head_dim as f64).powf(-0.5);
                match score_scale {
                    None => exact,
                    Some(s) if *s == exact as f32 => exact,
                    Some(s) => attr(*s),
                }
            },
            softcap: attr(attn_logit_softcap.unwrap_or(0.0)),
        }),
        _ => return None,
    })
}

/// RmsNorm (`center = false`) or LayerNorm: `norm(x) * g + b` over the last
/// axis. Inputs `[x, g, b]`.
struct NormF64 {
    eps: f64,
    center: bool,
}

impl CpuKernel for NormF64 {
    fn name(&self) -> &str {
        if self.center {
            "core.layer_norm_f64"
        } else {
            "core.rms_norm_f64"
        }
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("norm x")?;
        let g = inputs[1].expect_f64("norm gamma")?;
        let b = inputs[2].expect_f64("norm beta")?;
        let h = *dims(inputs[0].shape()).last().unwrap_or(&1);
        let out = output.expect_f64_mut("norm out")?;
        for (xr, or) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
            let mean = xr.iter().sum::<f64>() / h as f64;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h as f64;
            let (shift, denom) = match self.center {
                true => (mean, var),
                false => (0.0, var + mean * mean),
            };
            let inv = 1.0 / (denom + self.eps).sqrt();
            for i in 0..h {
                or[i] = (xr[i] - shift) * inv * g[i] + b[i];
            }
        }
        Ok(())
    }
}

struct SoftmaxF64 {
    axis: i32,
}

impl CpuKernel for SoftmaxF64 {
    fn name(&self) -> &str {
        "core.softmax_f64"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("softmax x")?;
        let d = dims(inputs[0].shape());
        let ax = if self.axis < 0 {
            (d.len() as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        let cols = d[ax];
        let inner: usize = d[ax + 1..].iter().product();
        let outer = x.len() / (cols * inner).max(1);
        let out = output.expect_f64_mut("softmax out")?;
        for o in 0..outer {
            for i in 0..inner {
                let at = |c: usize| o * cols * inner + c * inner + i;
                let m = (0..cols)
                    .map(|c| x[at(c)])
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for c in 0..cols {
                    let e = (x[at(c)] - m).exp();
                    out[at(c)] = e;
                    sum += e;
                }
                for c in 0..cols {
                    out[at(c)] /= sum;
                }
            }
        }
        Ok(())
    }
}

/// Rotary embedding, NeoX rotate-half or GPT-J interleaved, over the first
/// `n_rot` lanes of each head. Inputs `[x, cos, sin]`; tables hold one row per
/// position, or one per token when they cover `batch · seq` rows.
struct RopeF64 {
    head_dim: usize,
    n_rot: usize,
    interleaved: bool,
    table_stride: usize,
}

impl CpuKernel for RopeF64 {
    fn name(&self) -> &str {
        "core.rope_f64"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("rope x")?;
        let cos = inputs[1].expect_f64("rope cos")?;
        let sin = inputs[2].expect_f64("rope sin")?;
        let d = dims(inputs[0].shape());
        let (batch, seq, hidden) = if d.len() >= 3 {
            let r = d.len();
            (d[..r - 2].iter().product(), d[r - 2], d[r - 1])
        } else {
            (1, d[0], x.len() / d[0].max(1))
        };
        let (dh, half) = (self.head_dim, self.n_rot / 2);
        let stride = self.table_stride.max(1);
        let rows = cos.len() / stride;
        let per_token = rows == batch * seq && rows != seq;
        let out = output.expect_f64_mut("rope out")?;
        for t in 0..batch * seq {
            let tab = if per_token { t } else { t % seq } * stride;
            for h in 0..hidden / dh {
                let at = t * hidden + h * dh;
                let (xs, os) = (&x[at..at + dh], &mut out[at..at + dh]);
                for i in 0..half {
                    let (a, b) = match self.interleaved {
                        true => (2 * i, 2 * i + 1),
                        false => (i, half + i),
                    };
                    let (c, s) = (cos[tab + i], sin[tab + i]);
                    os[a] = xs[a] * c - xs[b] * s;
                    os[b] = xs[b] * c + xs[a] * s;
                }
                os[self.n_rot..].copy_from_slice(&xs[self.n_rot..]);
            }
        }
        Ok(())
    }
}

/// Scaled dot-product attention over `[B,S,H·D]`, `[B,S,H,D]` or `[B,H,S,D]`
/// (and `[S,H·D]`), with grouped KV heads and every [`MaskKind`].
struct AttentionF64 {
    heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    mask_kind: MaskKind,
    scale: f64,
    softcap: f64,
}

impl CpuKernel for AttentionF64 {
    fn name(&self) -> &str {
        "core.attention_f64"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("attention q")?;
        let k = inputs[1].expect_f64("attention k")?;
        let v = inputs[2].expect_f64("attention v")?;
        let (qd, kd) = (dims(inputs[0].shape()), dims(inputs[1].shape()));
        let (nh, dh, dv) = (self.heads, self.head_dim, self.v_head_dim);
        let (b, qs, ks, bhsd) = match qd.len() {
            4 if qd[1] == nh => (qd[0], qd[2], kd[2], true),
            4 => (qd[0], qd[1], kd[1], false),
            3 => (qd[0], qd[1], kd[1], false),
            _ => (1, qd[0], kd[0], false),
        };
        let nkv = (k.len() / (b * ks * dh).max(1)).max(1);
        let group = (nh / nkv).max(1);
        // Element offset of row `s` of head `h` in batch `bi`, for a tensor
        // with `n` heads of width `w`.
        let at = |bi: usize, h: usize, s: usize, n: usize, w: usize, len: usize| {
            if bhsd {
                ((bi * n + h) * len + s) * w
            } else {
                (bi * len + s) * n * w + h * w
            }
        };
        let mask = match self.mask_kind {
            MaskKind::Custom | MaskKind::Bias => inputs[3].expect_f64("attention mask")?,
            _ => &[][..],
        };
        let cfg = crate::config::RuntimeConfig::global();
        let (thr, neg) = (
            cfg.mask_binary_threshold as f64,
            cfg.attn_mask_neg_inf as f64,
        );
        let q_off = ks.saturating_sub(qs);
        let out = output.expect_f64_mut("attention out")?;
        let mut scores = vec![0f64; ks];
        for bi in 0..b {
            for h in 0..nh {
                let kh = h / group;
                for qi in 0..qs {
                    let qr = &q[at(bi, h, qi, nh, dh, qs)..][..dh];
                    let abs_q = q_off + qi;
                    for (ki, sc) in scores.iter_mut().enumerate() {
                        let kr = &k[at(bi, kh, ki, nkv, dh, ks)..][..dh];
                        let mut s = qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() * self.scale;
                        match self.mask_kind {
                            MaskKind::Custom if mask[bi * ks + ki] < thr => s = neg,
                            MaskKind::Bias => s += mask[((bi * nh + h) * qs + qi) * ks + ki],
                            MaskKind::Causal if ki > abs_q => s = neg,
                            MaskKind::SlidingWindow(w)
                                if ki > abs_q || ki < abs_q.saturating_sub(w) =>
                            {
                                s = neg
                            }
                            _ => {}
                        }
                        if self.softcap > 0.0 {
                            s = self.softcap * (s / self.softcap).tanh();
                        }
                        *sc = s;
                    }
                    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0;
                    for s in scores.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    let or = &mut out[at(bi, h, qi, nh, dv, qs)..][..dv];
                    or.fill(0.0);
                    for (ki, s) in scores.iter().enumerate() {
                        let w = s / sum;
                        let vr = &v[at(bi, kh, ki, nkv, dv, ks)..][..dv];
                        for (o, x) in or.iter_mut().zip(vr) {
                            *o += w * x;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
