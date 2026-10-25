// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Double-single tensors** — f64 values for backends without native f64.
//!
//! A double-single (df) value is an unevaluated pair of f32, `hi + lo` with
//! `|lo| ≤ ulp(hi)/2`: ~48 significand bits against f64's 53, and f32's
//! exponent range. A df tensor of logical shape `[d0, …, dn]` is an F32 tensor
//! of shape `[d0, …, dn, 2]` (`[…, 0]` = hi, `[…, 1]` = lo), 8 bytes per
//! element like f64. `rlx_compile::lower_f64` rewrites F64 graphs into this
//! form, so data movement (reshape, transpose, narrow, concat, gather) runs on
//! existing F32 kernels with the trailing axis carried along, and only the
//! arithmetic in [`DfOp`] needs dedicated kernels.
//!
//! The arithmetic is the error-free-transform library of `rlxsl::dw`, which
//! must be compiled without fast math (reassociation zeroes the error terms).

use crate::prelude::*;

/// Arithmetic on df tensors (`Op::DoubleSingle`). All shapes below are
/// logical; the F32 tensors carry a trailing axis of 2.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DfOp {
    /// Elementwise, numpy broadcasting: `[a, b] -> out`.
    Add,
    Sub,
    Mul,
    Div,
    /// `[.., m, k] @ [.., k, n] -> [.., m, n]` with df products and sums
    /// (leading batch axes broadcast).
    MatMul,
    /// Sum over `axes` of the single input.
    Sum {
        axes: Vec<usize>,
        keep_dim: bool,
    },
    /// Largest element over `axes` of the single input (softmax's shift).
    Max {
        axes: Vec<usize>,
        keep_dim: bool,
    },
    /// Elementwise larger of `a` and `b` (by `hi`, then `lo`), numpy broadcasting.
    Maximum,
    /// Elementwise `√x` of the single input.
    Sqrt,
    /// Elementwise `e^x` of the single input (f32's exponent range).
    Exp,
}

impl DfOp {
    pub fn num_inputs(&self) -> usize {
        match self {
            DfOp::Sum { .. } | DfOp::Max { .. } | DfOp::Sqrt | DfOp::Exp => 1,
            _ => 2,
        }
    }

    /// Kernel selector shared by every backend: elementwise ops (`Add` 0,
    /// `Sub` 1, `Mul` 2, `Div` 3, `Sqrt` 4, `Exp` 5) and reductions (`Sum` 0,
    /// `Max` 1); `Maximum` is elementwise 6.
    pub fn code(&self) -> u32 {
        match self {
            DfOp::Add | DfOp::Sum { .. } | DfOp::MatMul => 0,
            DfOp::Sub | DfOp::Max { .. } => 1,
            DfOp::Mul => 2,
            DfOp::Div => 3,
            DfOp::Sqrt => 4,
            DfOp::Exp => 5,
            DfOp::Maximum => 6,
        }
    }
}

impl core::fmt::Display for DfOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DfOp::Add => write!(f, "df_add"),
            DfOp::Sub => write!(f, "df_sub"),
            DfOp::Mul => write!(f, "df_mul"),
            DfOp::Div => write!(f, "df_div"),
            DfOp::MatMul => write!(f, "df_matmul"),
            DfOp::Sum { axes, keep_dim } => write!(f, "df_sum(axes={axes:?},keep={keep_dim})"),
            DfOp::Max { axes, keep_dim } => write!(f, "df_max(axes={axes:?},keep={keep_dim})"),
            DfOp::Sqrt => write!(f, "df_sqrt"),
            DfOp::Exp => write!(f, "df_exp"),
            DfOp::Maximum => write!(f, "df_maximum"),
        }
    }
}

/// Most axes a df kernel indexes (logical rank).
pub const DF_MAX_RANK: usize = 8;

/// The integers a df kernel needs, from logical (pair-stripped) shapes — one
/// derivation every backend shares, so they cannot disagree on indexing.
/// Strides and counts are in df elements (pairs), not floats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DfPlan {
    /// `out[i] = a[ia] ∘ b[ib]` with `ia/ib` from `out`'s coordinates via
    /// broadcast strides (0 on broadcast axes).
    Elementwise {
        len: usize,
        rank: usize,
        out_dims: [u32; DF_MAX_RANK],
        a_strides: [u32; DF_MAX_RANK],
        b_strides: [u32; DF_MAX_RANK],
    },
    /// `batch` independent `[m, k] @ [k, n]`; `a_batch`/`b_batch` are the
    /// per-batch strides (0 when broadcast).
    MatMul {
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a_batch: usize,
        b_batch: usize,
    },
    /// Each output element sums (or, for `Max`, takes the largest of)
    /// `red_len` inputs. Output index decomposes
    /// over `kept_dims` (strides into the input `kept_strides`), the reduction
    /// index over `red_dims` / `red_strides`.
    Sum {
        out_len: usize,
        red_len: usize,
        kept_rank: usize,
        kept_dims: [u32; DF_MAX_RANK],
        kept_strides: [u32; DF_MAX_RANK],
        red_rank: usize,
        red_dims: [u32; DF_MAX_RANK],
        red_strides: [u32; DF_MAX_RANK],
    },
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

fn pack(v: &[usize]) -> Result<[u32; DF_MAX_RANK], String> {
    if v.len() > DF_MAX_RANK {
        return Err(format!("df: rank {} exceeds {DF_MAX_RANK}", v.len()));
    }
    let mut out = [0u32; DF_MAX_RANK];
    for (o, &x) in out.iter_mut().zip(v) {
        *o = u32::try_from(x).map_err(|_| format!("df: extent {x} exceeds u32"))?;
    }
    Ok(out)
}

/// Broadcast strides of `dims` against `out` (right-aligned, numpy rules).
fn broadcast_strides(dims: &[usize], out: &[usize]) -> Result<Vec<usize>, String> {
    let own = row_major_strides(dims);
    let pad = out.len() - dims.len().min(out.len());
    (0..out.len())
        .map(|i| {
            if i < pad {
                return Ok(0);
            }
            let (d, o) = (dims[i - pad], out[i]);
            match d {
                _ if d == o => Ok(own[i - pad]),
                1 => Ok(0),
                _ => Err(format!("df: cannot broadcast {dims:?} to {out:?}")),
            }
        })
        .collect()
}

impl DfPlan {
    /// Plan from the F32 pair shapes a backend sees (`[…, 2]`).
    pub fn from_pair_shapes(op: &DfOp, inputs: &[&[usize]], out: &[usize]) -> Result<Self, String> {
        let ins: Vec<&[usize]> = inputs.iter().map(|d| logical_dims(d)).collect();
        Self::new(op, &ins, logical_dims(out))
    }

    /// Plan `op` for logical input shapes `inputs` producing logical `out`.
    pub fn new(op: &DfOp, inputs: &[&[usize]], out: &[usize]) -> Result<Self, String> {
        match op {
            DfOp::Sqrt | DfOp::Exp => {
                let [a] = inputs else {
                    return Err("df unary takes 1 input".into());
                };
                Ok(DfPlan::Elementwise {
                    len: out.iter().product(),
                    rank: out.len(),
                    out_dims: pack(out)?,
                    a_strides: pack(&broadcast_strides(a, out)?)?,
                    b_strides: [0; DF_MAX_RANK],
                })
            }
            DfOp::Add | DfOp::Sub | DfOp::Mul | DfOp::Div | DfOp::Maximum => {
                let [a, b] = inputs else {
                    return Err("df elementwise takes 2 inputs".into());
                };
                Ok(DfPlan::Elementwise {
                    len: out.iter().product(),
                    rank: out.len(),
                    out_dims: pack(out)?,
                    a_strides: pack(&broadcast_strides(a, out)?)?,
                    b_strides: pack(&broadcast_strides(b, out)?)?,
                })
            }
            DfOp::MatMul => {
                let [a, b] = inputs else {
                    return Err("df matmul takes 2 inputs".into());
                };
                if a.len() < 2 || b.len() < 2 {
                    return Err(format!("df matmul needs rank >= 2, got {a:?} @ {b:?}"));
                }
                let (m, k) = (a[a.len() - 2], a[a.len() - 1]);
                let (k2, n) = (b[b.len() - 2], b[b.len() - 1]);
                if k != k2 {
                    return Err(format!("df matmul inner dims {a:?} @ {b:?}"));
                }
                let batch: usize = out[..out.len() - 2].iter().product();
                let a_b: usize = a[..a.len() - 2].iter().product();
                let b_b: usize = b[..b.len() - 2].iter().product();
                let stride = |nb: usize, mat: usize| -> Result<usize, String> {
                    match nb {
                        1 => Ok(0),
                        _ if nb == batch => Ok(mat),
                        _ => Err(format!("df matmul batch {a:?} @ {b:?} -> {out:?}")),
                    }
                };
                Ok(DfPlan::MatMul {
                    batch,
                    m,
                    k,
                    n,
                    a_batch: stride(a_b, m * k)?,
                    b_batch: stride(b_b, k * n)?,
                })
            }
            DfOp::Sum { axes, .. } | DfOp::Max { axes, .. } => {
                let [x] = inputs else {
                    return Err("df reduction takes 1 input".into());
                };
                let strides = row_major_strides(x);
                let (mut kd, mut ks, mut rd, mut rs) = (vec![], vec![], vec![], vec![]);
                for (i, (&d, &st)) in x.iter().zip(&strides).enumerate() {
                    if axes.contains(&i) {
                        rd.push(d);
                        rs.push(st);
                    } else {
                        kd.push(d);
                        ks.push(st);
                    }
                }
                Ok(DfPlan::Sum {
                    out_len: kd.iter().product(),
                    red_len: rd.iter().product(),
                    kept_rank: kd.len(),
                    kept_dims: pack(&kd)?,
                    kept_strides: pack(&ks)?,
                    red_rank: rd.len(),
                    red_dims: pack(&rd)?,
                    red_strides: pack(&rs)?,
                })
            }
        }
    }
}

/// `offset = Σ coord_i · strides_i` for flat index `i` over `dims` (row-major).
pub fn df_offset(mut i: usize, rank: usize, dims: &[u32], strides: &[u32]) -> usize {
    let mut off = 0usize;
    for d in (0..rank).rev() {
        let n = dims[d] as usize;
        off += (i % n) * strides[d] as usize;
        i /= n;
    }
    off
}

/// Logical shape of a df tensor: its F32 shape without the trailing pair axis.
pub fn logical_dims(pair_dims: &[usize]) -> &[usize] {
    match pair_dims.split_last() {
        Some((2, rest)) => rest,
        _ => pair_dims,
    }
}

/// Split each f64 into its df pair, interleaved `(hi, lo)`. Exact: `hi` is
/// the f64 rounded to f32 and `lo` the (representable) remainder, also rounded.
pub fn split_f64(values: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let hi = v as f32;
        out.push(hi);
        out.push((v - hi as f64) as f32);
    }
    out
}

/// Inverse of [`split_f64`]: `hi + lo` evaluated in f64.
pub fn join_f64(pairs: &[f32]) -> Vec<f64> {
    pairs
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| p[0] as f64 + p[1] as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_index_like_numpy() {
        let p = DfPlan::new(&DfOp::Add, &[&[3, 1], &[4]], &[3, 4]).unwrap();
        let DfPlan::Elementwise {
            len,
            rank,
            out_dims,
            a_strides,
            b_strides,
        } = p
        else {
            panic!()
        };
        assert_eq!((len, rank), (12, 2));
        // out[2, 3] reads a[2, 0] and b[3].
        assert_eq!(df_offset(11, rank, &out_dims, &a_strides), 2);
        assert_eq!(df_offset(11, rank, &out_dims, &b_strides), 3);
        let p = DfPlan::new(
            &DfOp::Sum {
                axes: vec![1],
                keep_dim: false,
            },
            &[&[2, 3, 4]],
            &[2, 4],
        )
        .unwrap();
        let DfPlan::Sum {
            out_len,
            red_len,
            kept_rank,
            kept_dims,
            kept_strides,
            red_strides,
            ..
        } = p
        else {
            panic!()
        };
        assert_eq!((out_len, red_len), (8, 3));
        // out[1, 2] starts at x[1, 0, 2] and steps x[·, j, ·] by 4.
        assert_eq!(df_offset(6, kept_rank, &kept_dims, &kept_strides), 12 + 2);
        assert_eq!(red_strides[0], 4);
    }

    #[test]
    fn split_join_keeps_48_bits() {
        let v = [
            1e8 + 1e-3,
            -core::f64::consts::PI,
            1.0 / 3.0,
            0.0,
            -0.0,
            6.02e23,
        ];
        for (a, b) in v.iter().zip(join_f64(&split_f64(&v))) {
            assert!((a - b).abs() <= a.abs() * 2f64.powi(-47), "{a} -> {b}");
        }
    }
}
