// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::DoubleSingle` — f64 emulated as `(hi, lo)` f32 pairs. The arithmetic
//! is `rlxsl::dw::DoubleSingle`, the same operation sequence the GPU preludes
//! emit, so this kernel is the reference they are compared against.

use crate::thunk::*;
use rlx_ir::double_single::{DfOp, DfPlan, df_offset};
use rlxsl::dw::DoubleSingle as Df;

#[inline]
fn get(p: &[f32], i: usize) -> Df {
    Df {
        hi: p[2 * i],
        lo: p[2 * i + 1],
    }
}

#[inline]
fn put(p: &mut [f32], i: usize, v: Df) {
    p[2 * i] = v.hi;
    p[2 * i + 1] = v.lo;
}

/// The larger of two df values (by `hi`, then `lo`), as the GPU kernels pick.
#[inline]
fn df_max(a: Df, b: Df) -> Df {
    if b.hi > a.hi || (b.hi == a.hi && b.lo > a.lo) {
        b
    } else {
        a
    }
}

pub(crate) fn exec_double_single(t: &Thunk, base: *mut u8) {
    let Thunk::DoubleSingle {
        op,
        plan,
        a,
        a_len,
        b,
        b_len,
        out,
        out_len,
    } = t
    else {
        unreachable!()
    };
    let base_addr = base as usize;
    let (a, a_len, b, b_len, out, out_len) = (*a, *a_len, *b, *b_len, *out, *out_len);
    // Slices are re-derived per worker from the arena base (`par_for` takes `Fn`).
    let view = move || unsafe {
        let base = base_addr as *mut u8;
        let a = sl(a, base, 2 * a_len);
        let b = if b_len > 0 {
            sl(b, base, 2 * b_len)
        } else {
            &[][..]
        };
        (a, b, sl_mut(out, base, 2 * out_len))
    };
    match plan {
        DfPlan::Elementwise {
            len,
            rank,
            out_dims,
            a_strides,
            b_strides,
        } => {
            let f: fn(Df, Df) -> Df = match op {
                DfOp::Add => Df::add,
                DfOp::Sub => Df::sub,
                DfOp::Mul => Df::mul,
                DfOp::Div => Df::div,
                DfOp::Sqrt => |x, _| x.sqrt(),
                DfOp::Exp => |x, _| x.exp(),
                DfOp::Maximum => df_max,
                _ => unreachable!("elementwise plan for {op}"),
            };
            let unary = op.num_inputs() == 1;
            crate::pool::par_for(*len, 1 << 13, &|start, count| {
                let (xa, xb, o) = view();
                for i in start..start + count {
                    let ia = df_offset(i, *rank, out_dims, a_strides);
                    let y = if unary {
                        Df { hi: 0.0, lo: 0.0 }
                    } else {
                        get(xb, df_offset(i, *rank, out_dims, b_strides))
                    };
                    put(o, i, f(get(xa, ia), y));
                }
            });
        }
        DfPlan::MatMul {
            batch,
            m,
            k,
            n,
            a_batch,
            b_batch,
        } => {
            let (m, k, n) = (*m, *k, *n);
            let rows = batch * m;
            crate::pool::par_for(rows, 1, &|r0, count| {
                let (xa, xb, o) = view();
                for r in r0..r0 + count {
                    let (bi, i) = (r / m, r % m);
                    let a0 = bi * a_batch + i * k;
                    let b0 = bi * b_batch;
                    for j in 0..n {
                        let mut acc = Df { hi: 0.0, lo: 0.0 };
                        for kk in 0..k {
                            acc = acc.add(get(xa, a0 + kk).mul(get(xb, b0 + kk * n + j)));
                        }
                        put(o, (bi * m + i) * n + j, acc);
                    }
                }
            });
        }
        DfPlan::Sum {
            out_len,
            red_len,
            kept_rank,
            kept_dims,
            kept_strides,
            red_rank,
            red_dims,
            red_strides,
        } => {
            let max = matches!(op, DfOp::Max { .. });
            crate::pool::par_for(*out_len, 1, &|o0, count| {
                let (xa, _, o) = view();
                for oi in o0..o0 + count {
                    let base = df_offset(oi, *kept_rank, kept_dims, kept_strides);
                    let at =
                        |r: usize| get(xa, base + df_offset(r, *red_rank, red_dims, red_strides));
                    let mut acc = if max { at(0) } else { Df { hi: 0.0, lo: 0.0 } };
                    for r in usize::from(max)..*red_len {
                        let x = at(r);
                        acc = if max { df_max(acc, x) } else { acc.add(x) };
                    }
                    put(o, oi, acc);
                }
            });
        }
    }
}
