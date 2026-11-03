// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU executor for a [`Plan`]. This is the reference the WebGL fragment
//! shaders mirror — it is unit-tested against RLX's own CPU autodiff/runtime,
//! which verifies the planner + numerics without a browser.

use crate::plan::{Act, Bin, Cmp, LeafSource, PAD, Plan, Red, ScatterRed, Step};
use crate::{Result, WebglError};
use std::collections::HashMap;

/// Pointwise activation forward `f(x)`. Mirrored exactly by the GLSL.
pub(crate) fn act_f(act: Act, x: f32) -> f32 {
    match act {
        Act::Relu => x.max(0.0),
        Act::Neg => -x,
        Act::Exp => x.exp(),
        Act::Log => x.ln(),
        Act::Sqrt => x.sqrt(),
        Act::Rsqrt => 1.0 / x.sqrt(),
        Act::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Act::Tanh => x.tanh(),
        Act::Abs => x.abs(),
        Act::Sin => x.sin(),
        Act::Cos => x.cos(),
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Recip => 1.0 / x,
        Act::Gelu => 0.5 * x * (1.0 + erf_as(x * std::f32::consts::FRAC_1_SQRT_2)),
        Act::GeluApprox => 0.5 * x * (1.0 + gelu_tanh_arg(x).tanh()),
        Act::Softplus => softplus(x),
        Act::Tan => x.tan(),
        Act::Atan => x.atan(),
        Act::Round => x.round_ties_even(),
        Act::Floor => x.floor(),
        Act::Ceil => x.ceil(),
        Act::Sign => sign0(x),
        Act::Elu => {
            if x > 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        Act::Erf => erf_as(x),
        Act::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
        Act::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        Act::Mish => x * softplus(x).tanh(),
        Act::Softsign => x / (1.0 + x.abs()),
        Act::LogSigmoid => x.min(0.0) - (-x.abs()).exp().ln_1p(),
    }
}

fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `sign` with `sign(0) = 0` (GLSL `sign`; Rust's `signum(0.0)` is 1).
fn sign0(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `1` strictly inside `(lo, hi)`, else `0` — the derivative of a clamp.
fn inside(v: f32, lo: f32, hi: f32) -> f32 {
    if v > lo && v < hi { 1.0 } else { 0.0 }
}

/// `erf` by Abramowitz–Stegun 7.1.26 — the formula the GLSL uses.
pub(crate) fn erf_as(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let p = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - p * (-x * x).exp()).copysign(x)
}

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// `√(2/π)·(x + 0.044715·x³)`, clamped so `tanh` stays finite in GLSL too.
fn gelu_tanh_arg(x: f32) -> f32 {
    (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).clamp(-15.0, 15.0)
}

/// Pointwise activation derivative `f'(x)`. Mirrored exactly by the GLSL.
pub(crate) fn act_df(act: Act, x: f32) -> f32 {
    match act {
        Act::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Act::Neg => -1.0,
        Act::Exp => x.exp(),
        Act::Log => 1.0 / x,
        Act::Sqrt => 0.5 / x.sqrt(),
        Act::Rsqrt => -0.5 / (x * x.sqrt()),
        Act::Sigmoid => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        }
        Act::Tanh => {
            let t = x.tanh();
            1.0 - t * t
        }
        Act::Abs => x.signum(),
        Act::Sin => x.cos(),
        Act::Cos => -x.sin(),
        Act::Silu => {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }
        Act::Recip => -1.0 / (x * x),
        Act::Gelu => {
            let pdf = (-0.5 * x * x).exp() * 0.398_942_3;
            0.5 * (1.0 + erf_as(x * std::f32::consts::FRAC_1_SQRT_2)) + x * pdf
        }
        Act::GeluApprox => {
            let t = gelu_tanh_arg(x).tanh();
            0.5 * (1.0 + t)
                + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715 * x * x)
        }
        Act::Softplus => 1.0 / (1.0 + (-x).exp()),
        Act::Tan => {
            let c = x.cos();
            1.0 / (c * c)
        }
        Act::Atan => 1.0 / (1.0 + x * x),
        Act::Round => 1.0,
        Act::Floor | Act::Ceil | Act::Sign => 0.0,
        Act::Elu => {
            if x > 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        Act::Erf => std::f32::consts::FRAC_2_SQRT_PI * (-x * x).exp(),
        Act::HardSwish => ((x + 3.0).clamp(0.0, 6.0) + x * inside(x, -3.0, 3.0)) / 6.0,
        Act::HardSigmoid => inside(x, -3.0, 3.0) / 6.0,
        Act::Mish => {
            let t = softplus(x).tanh();
            t + x * (1.0 - t * t) / (1.0 + (-x).exp())
        }
        Act::Softsign => {
            let d = 1.0 + x.abs();
            1.0 / (d * d)
        }
        Act::LogSigmoid => 1.0 / (1.0 + x.exp()),
    }
}

/// A runtime index `f` (f32-encoded): rounded; with `wrap > 0` a negative one
/// counts from the end and the result is clamped to `[0, wrap)`. Mirrored by
/// the GLSL.
fn wrap_index(f: f32, wrap: u32) -> i64 {
    let mut ix = (f + 0.5).floor() as i64;
    if wrap > 0 {
        if ix < 0 {
            ix += wrap as i64;
        }
        ix = ix.clamp(0, wrap as i64 - 1);
    }
    ix
}

/// Output lane `i` of one [`Step::FftStage`] pass. Mirrored by the GLSL.
fn fft_stage_lane(
    x: &[f32],
    tw: &[f32],
    i: usize,
    n: usize,
    ns: usize,
    in_block: bool,
    out_block: bool,
) -> f32 {
    let (row, l) = (i / (2 * n), i % (2 * n));
    let (k, comp) = if out_block {
        (l % n, l / n)
    } else {
        (l / 2, l % 2)
    };
    let lane = |e: usize, c: usize| row * 2 * n + if in_block { c * n + e } else { 2 * e + c };
    let jm = k % ns;
    let j = (k / (2 * ns)) * ns + jm;
    let (ar, ai) = (x[lane(j, 0)], x[lane(j, 1)]);
    let (br, bi) = (x[lane(j + n / 2, 0)], x[lane(j + n / 2, 1)]);
    let (wr, wi) = (tw[2 * (ns - 1 + jm)], tw[2 * (ns - 1 + jm) + 1]);
    let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
    let s = if (k % (2 * ns)) < ns { 1.0 } else { -1.0 };
    if comp == 0 { ar + s * tr } else { ai + s * ti }
}

/// A scatter's runtime index: rounded, a negative one `+ wrap` (no clamp — an
/// out-of-range index lands nowhere). Mirrored by the GLSL.
fn scatter_index(f: f32, wrap: u32) -> i64 {
    let ix = (f + 0.5).floor() as i64;
    if ix < 0 { ix + wrap as i64 } else { ix }
}

/// Execute `plan` with the given named inputs/params; returns one vector per
/// graph output (in `graph.outputs` order), row-major.
pub fn run_cpu(plan: &Plan, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
    let input_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
    let mut vals: Vec<Vec<f32>> = vec![Vec::new(); plan.slot_len.len()];

    for step in &plan.steps {
        match step {
            Step::Leaf { out, src } => {
                let v = match src {
                    LeafSource::Input(name) | LeafSource::Param(name) => input_map
                        .get(name.as_str())
                        .map(|d| d.to_vec())
                        .ok_or_else(|| WebglError(format!("missing input/param '{name}'")))?,
                    LeafSource::Const(d) => d.clone(),
                };
                if v.len() != plan.slot_len[*out] {
                    return Err(WebglError(format!(
                        "leaf slot {} expected {} elems, got {}",
                        out,
                        plan.slot_len[*out],
                        v.len()
                    )));
                }
                vals[*out] = v;
            }
            Step::Unary { out, a, act } => {
                let o: Vec<f32> = vals[*a].iter().map(|&v| act_f(*act, v)).collect();
                vals[*out] = o;
            }
            Step::ActBack { out, x, dy, act } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    o[i] = vals[*dy][i] * act_df(*act, vals[*x][i]);
                }
                vals[*out] = o;
            }
            Step::Binary { out, a, b, op } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    let (av, bv) = (vals[*a][i], vals[*b][i]);
                    o[i] = match op {
                        Bin::Add => av + bv,
                        Bin::Sub => av - bv,
                        Bin::Mul => av * bv,
                        Bin::Div => av / bv,
                        Bin::Max => av.max(bv),
                        Bin::Min => av.min(bv),
                        Bin::Pow => av.powf(bv),
                        Bin::Mod => av % bv,
                        Bin::BitAnd => ((av as i64) & (bv as i64)) as f32,
                        Bin::BitOr => ((av as i64) | (bv as i64)) as f32,
                        Bin::BitXor => ((av as i64) ^ (bv as i64)) as f32,
                        Bin::Shl => (av as i64).wrapping_shl(bv as u32) as f32,
                        Bin::Shr => (av as i64).wrapping_shr(bv as u32) as f32,
                        Bin::Atan2 => av.atan2(bv),
                    };
                }
                vals[*out] = o;
            }
            Step::Compare { out, a, b, cmp } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    let (av, bv) = (vals[*a][i], vals[*b][i]);
                    let t = match cmp {
                        Cmp::Eq => av == bv,
                        Cmp::Ne => av != bv,
                        Cmp::Lt => av < bv,
                        Cmp::Le => av <= bv,
                        Cmp::Gt => av > bv,
                        Cmp::Ge => av >= bv,
                    };
                    o[i] = if t { 1.0 } else { 0.0 };
                }
                vals[*out] = o;
            }
            Step::Where { out, cond, a, b } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    o[i] = if vals[*cond][i] != 0.0 {
                        vals[*a][i]
                    } else {
                        vals[*b][i]
                    };
                }
                vals[*out] = o;
            }
            Step::MatMul {
                out,
                a,
                b,
                m,
                k,
                n,
                batch,
                b_batched,
            } => {
                let mut o = vec![0f32; batch * m * n];
                {
                    let (av, bv) = (&vals[*a], &vals[*b]);
                    for bi in 0..*batch {
                        let (ao, bo) = (bi * m * k, if *b_batched { bi * k * n } else { 0 });
                        for i in 0..*m {
                            for j in 0..*n {
                                let mut s = 0f32;
                                for l in 0..*k {
                                    s += av[ao + i * k + l] * bv[bo + l * n + j];
                                }
                                o[bi * m * n + i * n + j] = s;
                            }
                        }
                    }
                }
                vals[*out] = o;
            }
            Step::Gather { out, src, idx } => {
                let s = &vals[*src];
                let mut o = Vec::with_capacity(idx.len());
                for &ix in idx {
                    // PAD sentinel → 0 (used for conv/im2col padding).
                    if ix == PAD {
                        o.push(0.0);
                        continue;
                    }
                    o.push(*s.get(ix as usize).ok_or_else(|| {
                        WebglError(format!(
                            "gather idx {ix} OOB: src slot {src} dims {:?} len {}, out slot {out} dims {:?} len {}",
                            plan.slot_dims[*src],
                            s.len(),
                            plan.slot_dims[*out],
                            plan.slot_len[*out],
                        ))
                    })?);
                }
                vals[*out] = o;
            }
            Step::Reduce {
                out,
                src,
                groups,
                fanin,
                op,
            } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for (oi, slot) in o.iter_mut().enumerate() {
                    let mut acc = match op {
                        Red::Sum | Red::Mean => 0.0,
                        Red::Prod => 1.0,
                        Red::Max => f32::NEG_INFINITY,
                        Red::Min => f32::INFINITY,
                    };
                    let mut count = 0f32;
                    for j in 0..*fanin {
                        let g = groups[oi * fanin + j];
                        if g == PAD {
                            continue;
                        }
                        let v = vals[*src][g as usize];
                        count += 1.0;
                        match op {
                            Red::Sum | Red::Mean => acc += v,
                            Red::Prod => acc *= v,
                            Red::Max => acc = acc.max(v),
                            Red::Min => acc = acc.min(v),
                        }
                    }
                    *slot = if matches!(op, Red::Mean) && count > 0.0 {
                        acc / count
                    } else {
                        acc
                    };
                }
                vals[*out] = o;
            }
            Step::Softmax { out, a, rows, cols } => {
                let mut o = vec![0f32; rows * cols];
                for r in 0..*rows {
                    let row = &vals[*a][r * cols..(r + 1) * cols];
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for (c, &v) in row.iter().enumerate() {
                        let e = (v - m).exp();
                        o[r * cols + c] = e;
                        sum += e;
                    }
                    for c in 0..*cols {
                        o[r * cols + c] /= sum;
                    }
                }
                vals[*out] = o;
            }
            Step::LayerNorm {
                out,
                x,
                gamma,
                beta,
                rows,
                cols,
                eps,
            } => {
                let mut o = vec![0f32; rows * cols];
                let n = *cols as f32;
                for r in 0..*rows {
                    let row = &vals[*x][r * cols..(r + 1) * cols];
                    let mean = row.iter().sum::<f32>() / n;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
                    let inv = 1.0 / (var + eps).sqrt();
                    for c in 0..*cols {
                        o[r * cols + c] = (row[c] - mean) * inv * vals[*gamma][c] + vals[*beta][c];
                    }
                }
                vals[*out] = o;
            }
            Step::RmsNorm {
                out,
                x,
                gamma,
                beta,
                rows,
                cols,
                eps,
            } => {
                let mut o = vec![0f32; rows * cols];
                let n = *cols as f32;
                for r in 0..*rows {
                    let row = &vals[*x][r * cols..(r + 1) * cols];
                    let ms = row.iter().map(|v| v * v).sum::<f32>() / n;
                    let inv = 1.0 / (ms + eps).sqrt();
                    for c in 0..*cols {
                        o[r * cols + c] = row[c] * inv * vals[*gamma][c] + vals[*beta][c];
                    }
                }
                vals[*out] = o;
            }
            Step::ArgReduce {
                out,
                src,
                groups,
                fanin,
                is_max,
            } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for (oi, slot) in o.iter_mut().enumerate() {
                    let mut best = if *is_max {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    let mut best_j = 0usize;
                    for j in 0..*fanin {
                        let v = vals[*src][groups[oi * fanin + j] as usize];
                        if (*is_max && v > best) || (!*is_max && v < best) {
                            best = v;
                            best_j = j;
                        }
                    }
                    *slot = best_j as f32;
                }
                vals[*out] = o;
            }
            Step::GatherRuntime {
                out,
                table,
                indices,
                which,
                base,
                axis_stride,
                wrap,
            } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    let ix = wrap_index(vals[*indices][which[i] as usize], *wrap);
                    o[i] = vals[*table][base[i] as usize + ix as usize * *axis_stride as usize];
                }
                vals[*out] = o;
            }
            Step::ScatterRuntime {
                out,
                data,
                updates,
                indices,
                ibase,
                ubase,
                key,
                count,
                istride,
                ustride,
                wrap,
                red,
            } => {
                let n = plan.slot_len[*out];
                let mut o = match data {
                    Some(d) => vals[*d].clone(),
                    None => vec![0f32; n],
                };
                for i in 0..n {
                    if ibase[i] == PAD {
                        continue;
                    }
                    for j in 0..*count {
                        let ix = vals[*indices][ibase[i] as usize + j * istride];
                        if scatter_index(ix, *wrap) == key[i] as i64 {
                            let u = vals[*updates][ubase[i] as usize + j * ustride];
                            o[i] = match red {
                                ScatterRed::Replace => u,
                                ScatterRed::Add => o[i] + u,
                                ScatterRed::Mul => o[i] * u,
                                ScatterRed::Max => o[i].max(u),
                                ScatterRed::Min => o[i].min(u),
                            };
                        }
                    }
                }
                vals[*out] = o;
            }
            Step::FftStage {
                out,
                src,
                tw,
                n,
                ns,
                in_block,
                out_block,
                scale,
            } => {
                let (x, w) = (&vals[*src], &vals[*tw]);
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (i, ov) in o.iter_mut().enumerate() {
                    *ov = fft_stage_lane(x, w, i, *n, *ns, *in_block, *out_block) * scale;
                }
                vals[*out] = o;
            }
            Step::Host { out, inputs, op } => {
                let ins: Vec<&[f32]> = inputs.iter().map(|&i| vals[i].as_slice()).collect();
                vals[*out] = op.run(&ins)?;
            }
            Step::Linalg {
                out,
                a,
                b,
                n,
                nrhs,
                batch,
                op,
            } => {
                let r = crate::linalg::run(
                    *op,
                    &vals[*a],
                    b.map(|b| vals[b].as_slice()),
                    *n,
                    *nrhs,
                    *batch,
                );
                vals[*out] = r;
            }
            Step::SortAxis {
                out,
                src,
                outer: _,
                len,
                inner,
                k,
                descending,
                indices,
            } => {
                let n = plan.slot_len[*out];
                let x = &vals[*src];
                let mut o = vec![0f32; n];
                for (i, ov) in o.iter_mut().enumerate() {
                    let (row, p, ii) = (i / (k * inner), (i / inner) % k, i % inner);
                    let at = |j: usize| x[(row * len + j) * inner + ii];
                    // The element of rank `p`: `rank(j)` = how many sort before it.
                    let before = |a: f32, ia: usize, v: f32, j: usize| {
                        (if *descending { a > v } else { a < v }) || (a == v && ia < j)
                    };
                    for j in 0..*len {
                        let v = at(j);
                        if (0..*len).filter(|&q| before(at(q), q, v, j)).count() == p {
                            *ov = if *indices { j as f32 } else { v };
                            break;
                        }
                    }
                }
                vals[*out] = o;
            }
            Step::ComplexCast { out, src, mode, n } => {
                // Lane moves on the f32-uniform arena. Mirrors the wgpu/cuda/
                // vulkan `complex_cast` table (C64 = 2 lanes, C128 = 4 lanes).
                // Real sources carry lo=0, so no compensated df64 math.
                let s = &vals[*src];
                let out_len = plan.slot_len[*out];
                let mut o = vec![0f32; out_len];
                let n = *n;
                match mode {
                    0 => {
                        for k in 0..n {
                            o[2 * k] = s[k]; // o[2k+1] stays 0
                        }
                    }
                    1 => {
                        for k in 0..n {
                            o[k] = s[2 * k];
                        }
                    }
                    2 => {
                        for k in 0..n {
                            o[4 * k] = s[k]; // o[4k+1..3] stay 0
                        }
                    }
                    3 => {
                        for k in 0..n {
                            o[k] = s[4 * k];
                        }
                    }
                    4 => {
                        for k in 0..n {
                            o[4 * k] = s[2 * k];
                            o[4 * k + 2] = s[2 * k + 1];
                        }
                    }
                    5 => {
                        for k in 0..n {
                            o[2 * k] = s[4 * k];
                            o[2 * k + 1] = s[4 * k + 2];
                        }
                    }
                    other => {
                        return Err(WebglError(format!("bad complex_cast mode {other}")));
                    }
                }
                vals[*out] = o;
            }
            Step::BinaryC64 {
                out,
                a,
                b,
                op,
                n,
                n_a,
                n_b,
            } => {
                // C64 element-wise binary. Formulas + modulo broadcast mirror
                // rlx-cpu `exec_binary_full_c64`. C64 = 2 f32 lanes `[re, im]`.
                let (av, bv) = (&vals[*a], &vals[*b]);
                let mut o = vec![0f32; 2 * *n];
                for k in 0..*n {
                    let ka = k % *n_a;
                    let kb = k % *n_b;
                    let (ar, ai) = (av[2 * ka], av[2 * ka + 1]);
                    let (br, bi) = (bv[2 * kb], bv[2 * kb + 1]);
                    let (cr, ci) = match op {
                        Bin::Add => (ar + br, ai + bi),
                        Bin::Sub => (ar - br, ai - bi),
                        Bin::Mul => (ar * br - ai * bi, ar * bi + ai * br),
                        Bin::Div => {
                            let d = br * br + bi * bi;
                            ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
                        }
                        Bin::Max
                        | Bin::Min
                        | Bin::Pow
                        | Bin::Mod
                        | Bin::BitAnd
                        | Bin::BitOr
                        | Bin::BitXor
                        | Bin::Shl
                        | Bin::Shr
                        | Bin::Atan2 => {
                            unreachable!("C64 max/min/pow/mod/bitwise rejected at lowering")
                        }
                    };
                    o[2 * k] = cr;
                    o[2 * k + 1] = ci;
                }
                vals[*out] = o;
            }
            Step::Custom {
                out,
                input,
                name,
                attrs,
            } => {
                let o = run_custom_collective(name, &vals[*input], plan.slot_len[*out], attrs)?;
                vals[*out] = o;
            }
            Step::Conv3d {
                out,
                x,
                w,
                bias,
                n,
                cin,
                din,
                cout,
                dout,
                k,
                stride,
                pad,
                dil,
                groups,
                act,
            } => {
                let (xv, wv) = (&vals[*x], &vals[*w]);
                let bv = bias.map(|b| &vals[b]);
                let (cin_g, cout_g) = (cin / groups, cout / groups);
                let in_vol = din[0] * din[1] * din[2];
                let out_vol = dout[0] * dout[1] * dout[2];
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (lin, slot) in o.iter_mut().enumerate().take(n * cout * out_vol) {
                    // the GL shader's order, so the sums round alike
                    let ow = lin % dout[2];
                    let oh = lin / dout[2] % dout[1];
                    let od = lin / (dout[2] * dout[1]) % dout[0];
                    let co = lin / out_vol % cout;
                    let ni = lin / (out_vol * cout);
                    let g = co / cout_g;
                    let mut s = bv.map_or(0.0, |b| b[co]);
                    for ci in 0..cin_g {
                        let x_base = (ni * cin + g * cin_g + ci) * in_vol;
                        let w_base = (co * cin_g + ci) * k[0] * k[1] * k[2];
                        for kd in 0..k[0] {
                            let Some(id) = (od * stride[0] + kd * dil[0])
                                .checked_sub(pad[0])
                                .filter(|&v| v < din[0])
                            else {
                                continue;
                            };
                            for kh in 0..k[1] {
                                let Some(ih) = (oh * stride[1] + kh * dil[1])
                                    .checked_sub(pad[1])
                                    .filter(|&v| v < din[1])
                                else {
                                    continue;
                                };
                                for kw in 0..k[2] {
                                    let Some(iw) = (ow * stride[2] + kw * dil[2])
                                        .checked_sub(pad[2])
                                        .filter(|&v| v < din[2])
                                    else {
                                        continue;
                                    };
                                    s += xv[x_base + (id * din[1] + ih) * din[2] + iw]
                                        * wv[w_base + (kd * k[1] + kh) * k[2] + kw];
                                }
                            }
                        }
                    }
                    *slot = act.map_or(s, |a| act_f(a, s));
                }
                vals[*out] = o;
            }
            Step::ArgAxis {
                out,
                src,
                outer,
                len,
                inner,
                is_max,
            } => {
                let sv = &vals[*src];
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (oi, slot) in o.iter_mut().enumerate().take(outer * inner) {
                    let (a, i) = (oi / inner, oi % inner);
                    let base = a * len * inner + i;
                    let (mut best, mut at) = (sv[base], 0);
                    for j in 1..*len {
                        let v = sv[base + j * inner];
                        if (*is_max && v > best) || (!*is_max && v < best) {
                            (best, at) = (v, j);
                        }
                    }
                    *slot = at as f32;
                }
                vals[*out] = o;
            }
            Step::Conv1ArgMax {
                out,
                x,
                w,
                bias,
                n,
                cin,
                cout,
                spatial,
            } => {
                let (xv, wv) = (&vals[*x], &vals[*w]);
                let bv = bias.map(|b| &vals[b]);
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (oi, slot) in o.iter_mut().enumerate().take(n * spatial) {
                    let (ni, p) = (oi / spatial, oi % spatial);
                    let (mut best, mut at) = (f32::NEG_INFINITY, 0);
                    for co in 0..*cout {
                        let mut s = bv.map_or(0.0, |b| b[co]);
                        for ci in 0..*cin {
                            s += xv[(ni * cin + ci) * spatial + p] * wv[co * cin + ci];
                        }
                        if co == 0 || s > best {
                            (best, at) = (s, co);
                        }
                    }
                    *slot = at as f32;
                }
                vals[*out] = o;
            }
            Step::GroupNorm {
                out,
                x,
                gamma,
                beta,
                n,
                c,
                groups,
                spatial,
                eps,
                act,
            } => {
                let (xv, gv, bv) = (&vals[*x], &vals[*gamma], &vals[*beta]);
                let run = c / groups * spatial;
                let mut o = vec![0f32; plan.slot_len[*out]];
                for ng in 0..n * groups {
                    let seg = &xv[ng * run..(ng + 1) * run];
                    let mean = seg.iter().map(|&v| v as f64).sum::<f64>() / run as f64;
                    let var =
                        seg.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / run as f64;
                    let inv = 1.0 / (var + *eps as f64).sqrt();
                    for (j, &v) in seg.iter().enumerate() {
                        let ch = (ng % groups) * (c / groups) + j / spatial;
                        let y = ((v as f64 - mean) * inv) as f32 * gv[ch] + bv[ch];
                        o[ng * run + j] = act.map_or(y, |a| act_f(a, y));
                    }
                }
                vals[*out] = o;
            }
            Step::Broadcast {
                out,
                src,
                out_dims,
                in_strides,
            } => {
                let sv = &vals[*src];
                let n: usize = out_dims.iter().product();
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (lin, slot) in o.iter_mut().enumerate().take(n) {
                    let (mut rest, mut at) = (lin, 0);
                    for k in (0..out_dims.len()).rev() {
                        at += rest % out_dims[k] * in_strides[k];
                        rest /= out_dims[k];
                    }
                    *slot = sv[at];
                }
                vals[*out] = o;
            }
            Step::Splat { out, src } => {
                vals[*out] = vec![vals[*src][0]; plan.slot_len[*out]];
            }
            Step::Copy { out, src } => {
                let mut o = vals[*src].clone();
                o.resize(plan.slot_len[*out], 0.0);
                vals[*out] = o;
            }
            Step::Concat {
                out,
                a,
                b,
                outer,
                a_len,
                b_len,
            } => {
                let mut o = Vec::with_capacity(plan.slot_len[*out]);
                for r in 0..*outer {
                    o.extend_from_slice(&vals[*a][r * a_len..(r + 1) * a_len]);
                    o.extend_from_slice(&vals[*b][r * b_len..(r + 1) * b_len]);
                }
                o.resize(plan.slot_len[*out], 0.0);
                vals[*out] = o;
            }
            Step::Pool3d {
                out,
                src,
                nc,
                din,
                dout,
                k,
                stride,
                pad,
                is_max,
            } => {
                let sv = &vals[*src];
                let (iv, ov) = (din[0] * din[1] * din[2], dout[0] * dout[1] * dout[2]);
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (lin, slot) in o.iter_mut().enumerate().take(nc * ov) {
                    let (c, q) = (lin / ov, lin % ov);
                    let at = [q / (dout[1] * dout[2]), q / dout[2] % dout[1], q % dout[2]];
                    let mut acc = if *is_max { f32::NEG_INFINITY } else { 0.0 };
                    for kd in 0..k[0] {
                        for kh in 0..k[1] {
                            for kw in 0..k[2] {
                                let pos = [
                                    at[0] * stride[0] + kd,
                                    at[1] * stride[1] + kh,
                                    at[2] * stride[2] + kw,
                                ];
                                let Some(p) = (0..3)
                                    .map(|a| pos[a].checked_sub(pad[a]).filter(|&v| v < din[a]))
                                    .collect::<Option<Vec<_>>>()
                                else {
                                    continue;
                                };
                                let v = sv[c * iv + (p[0] * din[1] + p[1]) * din[2] + p[2]];
                                acc = if *is_max { acc.max(v) } else { acc + v };
                            }
                        }
                    }
                    // an average divides by the whole window, padding
                    // included, as rlx-cpu does
                    *slot = if *is_max {
                        acc
                    } else {
                        acc / (k[0] * k[1] * k[2]) as f32
                    };
                }
                vals[*out] = o;
            }
            Step::Resize3d {
                out,
                src,
                nc,
                din,
                dout,
            } => {
                let sv = &vals[*src];
                let (iv, ov) = (din[0] * din[1] * din[2], dout[0] * dout[1] * dout[2]);
                let mut o = vec![0f32; plan.slot_len[*out]];
                for (lin, slot) in o.iter_mut().enumerate().take(nc * ov) {
                    let (c, q) = (lin / ov, lin % ov);
                    let at = [q / (dout[1] * dout[2]), q / dout[2] % dout[1], q % dout[2]];
                    let p = [0, 1, 2].map(|a| (at[a] * din[a] / dout[a]).min(din[a] - 1));
                    *slot = sv[c * iv + (p[0] * din[1] + p[1]) * din[2] + p[2]];
                }
                vals[*out] = o;
            }
        }
    }

    Ok(plan.outputs.iter().map(|&s| vals[s].clone()).collect())
}

/// Run a host/transport `collective.*` op by delegating to the single
/// registered CPU kernel via `rlx_cpu::op_registry::run_f32_custom_op_host`.
///
/// The collective kernels work off element counts + `attrs` (not the logical
/// tensor rank), so a 1-D f32 `Shape` built from the element count is faithful —
/// this mirrors how every other GPU backend (wgpu / metal / cuda / rocm / coreml)
/// stages the operand to host and calls the one shared helper.
///
/// **Native only.** The collective transport (`rlx-driver` `ProcessGroup` /
/// `NetTransport`) is built on `std::net` TCP sockets, which do not exist on
/// wasm32/browser — so on wasm this returns a clear error instead of pretending
/// to run a collective. See the module docs.
#[cfg(not(target_family = "wasm"))]
fn run_custom_collective(
    name: &str,
    input: &[f32],
    out_len: usize,
    attrs: &[u8],
) -> Result<Vec<f32>> {
    use rlx_ir::{DType, Shape};
    let mut out = vec![0f32; out_len];
    let in_shape = Shape::new(&[input.len()], DType::F32);
    let out_shape = Shape::new(&[out.len()], DType::F32);
    rlx_cpu::op_registry::run_f32_custom_op_host(
        name,
        &[(bytemuck_cast(input), &in_shape)],
        (bytemuck_cast_mut(&mut out), &out_shape),
        attrs,
    )
    .map_err(|e| WebglError(format!("collective '{name}': {e}")))?;
    Ok(out)
}

/// On wasm there is no TCP transport for the process group, so collectives
/// cannot run. Report that plainly rather than fabricating a result.
#[cfg(target_family = "wasm")]
fn run_custom_collective(
    name: &str,
    _input: &[f32],
    _out_len: usize,
    _attrs: &[u8],
) -> Result<Vec<f32>> {
    Err(WebglError(format!(
        "collective '{name}' unavailable in browser: no TCP transport on wasm32 \
         (rlx-driver ProcessGroup uses std::net sockets). Run the collective graph \
         on the native CPU executor."
    )))
}

/// Reinterpret an `f32` slice as bytes (little-endian, native layout). The host
/// helper reads it back through the same `f32`-alignment contract.
#[cfg(not(target_family = "wasm"))]
fn bytemuck_cast(s: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no invalid bit patterns; `[f32]` → `[u8]` is a valid
    // reinterpret and the resulting slice covers exactly `len * 4` bytes.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(not(target_family = "wasm"))]
fn bytemuck_cast_mut(s: &mut [f32]) -> &mut [u8] {
    let len = std::mem::size_of_val(s);
    // SAFETY: as above; mutable reinterpret of `[f32]` → `[u8]`.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Cmp, LeafSource, Plan, Step};

    // Compare/Where have no graph builders, so exercise the executor directly
    // with a hand-built Plan (the GLSL mirrors these formulas).
    #[test]
    fn compare_lt() {
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("a".into()),
                },
                Step::Leaf {
                    out: 1,
                    src: LeafSource::Input("b".into()),
                },
                Step::Compare {
                    out: 2,
                    a: 0,
                    b: 1,
                    cmp: Cmp::Lt,
                },
            ],
            slot_dims: vec![(1, 3); 3],
            slot_len: vec![3; 3],
            outputs: vec![2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(&plan, &[("a", &[1.0, 5.0, 3.0]), ("b", &[2.0, 2.0, 2.0])]).unwrap();
        assert_eq!(out[0], vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn where_select() {
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("c".into()),
                },
                Step::Leaf {
                    out: 1,
                    src: LeafSource::Input("a".into()),
                },
                Step::Leaf {
                    out: 2,
                    src: LeafSource::Input("b".into()),
                },
                Step::Where {
                    out: 3,
                    cond: 0,
                    a: 1,
                    b: 2,
                },
            ],
            slot_dims: vec![(1, 3); 4],
            slot_len: vec![3; 4],
            outputs: vec![3],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(
            &plan,
            &[
                ("c", &[1.0, 0.0, 1.0]),
                ("a", &[10.0, 20.0, 30.0]),
                ("b", &[-1.0, -2.0, -3.0]),
            ],
        )
        .unwrap();
        assert_eq!(out[0], vec![10.0, -2.0, 30.0]);
    }

    // Complex simulation on the f32-uniform arena. C64 = 2 f32 lanes `[re, im]`.
    // The GLSL mirrors these exact formulas, so validating the executor here
    // validates the numerics both paths share.
    use crate::plan::Bin;

    #[test]
    fn complex_mul_c64() {
        // (1+2i)(5+6i) = -7+16i ; (3+4i)(7+8i) = -11+52i.
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("a".into()),
                },
                Step::Leaf {
                    out: 1,
                    src: LeafSource::Input("b".into()),
                },
                Step::BinaryC64 {
                    out: 2,
                    a: 0,
                    b: 1,
                    op: Bin::Mul,
                    n: 2,
                    n_a: 2,
                    n_b: 2,
                },
            ],
            slot_dims: vec![(1, 4); 3],
            slot_len: vec![4; 3],
            outputs: vec![2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(
            &plan,
            &[("a", &[1.0, 2.0, 3.0, 4.0]), ("b", &[5.0, 6.0, 7.0, 8.0])],
        )
        .unwrap();
        assert_eq!(out[0], vec![-7.0, 16.0, -11.0, 52.0]);
    }

    #[test]
    fn complex_add_broadcast_scalar() {
        // [1+2i, 3+4i] + (10+20i) = [11+22i, 13+24i]  (rhs scalar, n_b=1).
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("a".into()),
                },
                Step::Leaf {
                    out: 1,
                    src: LeafSource::Input("b".into()),
                },
                Step::BinaryC64 {
                    out: 2,
                    a: 0,
                    b: 1,
                    op: Bin::Add,
                    n: 2,
                    n_a: 2,
                    n_b: 1,
                },
            ],
            slot_dims: vec![(1, 4), (1, 2), (1, 4)],
            slot_len: vec![4, 2, 4],
            outputs: vec![2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(&plan, &[("a", &[1.0, 2.0, 3.0, 4.0]), ("b", &[10.0, 20.0])]).unwrap();
        assert_eq!(out[0], vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn complex_div_c64() {
        // (1+2i)/(3+4i) = (11 + 2i)/25 = 0.44 + 0.08i.
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("a".into()),
                },
                Step::Leaf {
                    out: 1,
                    src: LeafSource::Input("b".into()),
                },
                Step::BinaryC64 {
                    out: 2,
                    a: 0,
                    b: 1,
                    op: Bin::Div,
                    n: 1,
                    n_a: 1,
                    n_b: 1,
                },
            ],
            slot_dims: vec![(1, 2); 3],
            slot_len: vec![2; 3],
            outputs: vec![2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(&plan, &[("a", &[1.0, 2.0]), ("b", &[3.0, 4.0])]).unwrap();
        assert!((out[0][0] - 0.44).abs() < 1e-6 && (out[0][1] - 0.08).abs() < 1e-6);
    }

    #[test]
    fn complex_cast_real_roundtrip() {
        // real [1,2,3] → C64 [1,0, 2,0, 3,0] → real [1,2,3].
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("x".into()),
                },
                Step::ComplexCast {
                    out: 1,
                    src: 0,
                    mode: 0,
                    n: 3,
                },
                Step::ComplexCast {
                    out: 2,
                    src: 1,
                    mode: 1,
                    n: 3,
                },
            ],
            slot_dims: vec![(1, 3), (1, 6), (1, 3)],
            slot_len: vec![3, 6, 3],
            outputs: vec![1, 2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(&plan, &[("x", &[1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(out[0], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        assert_eq!(out[1], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn complex_cast_c64_c128_roundtrip() {
        // C64 [1+2i, 3+4i] → C128 (df64 lo=0) → C64 (drop lo) is identity.
        let plan = Plan {
            steps: vec![
                Step::Leaf {
                    out: 0,
                    src: LeafSource::Input("z".into()),
                },
                Step::ComplexCast {
                    out: 1,
                    src: 0,
                    mode: 4, // C64 → C128
                    n: 2,
                },
                Step::ComplexCast {
                    out: 2,
                    src: 1,
                    mode: 5, // C128 → C64
                    n: 2,
                },
            ],
            slot_dims: vec![(1, 4), (1, 8), (1, 4)],
            slot_len: vec![4, 8, 4],
            outputs: vec![1, 2],
            slot_origin: Vec::new(),
        };
        let out = run_cpu(&plan, &[("z", &[1.0, 2.0, 3.0, 4.0])]).unwrap();
        assert_eq!(out[0], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
        assert_eq!(out[1], vec![1.0, 2.0, 3.0, 4.0]);
    }

    // A `collective.*` custom op lowers to `Step::Custom` and the NATIVE CPU
    // executor host-delegates it to the single registered rlx-cpu kernel via
    // `run_f32_custom_op_host`. `copy_to_parallel` is an identity copy whose
    // kernel needs no process group, so this exercises the full wiring
    // (build_plan → Step::Custom → host delegate → registered kernel) without a
    // network. This path is native-only: on wasm there is no TCP transport.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn collective_copy_to_parallel_host_delegate() {
        use crate::build_plan;
        use rlx_ir::{DType, Graph, Shape};

        // Register the collective op extensions + CPU kernels.
        rlx_collectives::register();

        let mut g = Graph::new("coll");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        // Megatron `f` operator == forward identity copy (group 0 unused here).
        let y = rlx_collectives::copy_to_model_parallel(&mut g, x, 0);
        g.set_outputs(vec![y]);

        let plan = build_plan(&g).expect("build_plan lowers collective.copy_to_parallel");
        // Sanity: the collective must survive legalization as a `Step::Custom`.
        assert!(
            plan.steps.iter().any(|s| matches!(
                s,
                Step::Custom { name, .. } if name == "collective.copy_to_parallel"
            )),
            "collective op should lower to Step::Custom"
        );

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = run_cpu(&plan, &[("x", &data)]).expect("run_cpu host-delegates collective");
        assert_eq!(out[0], data.to_vec());
    }
}
