// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! CPU executor for a [`Plan`]. This is the reference the WebGL fragment
//! shaders mirror — it is unit-tested against RLX's own CPU autodiff/runtime,
//! which verifies the planner + numerics without a browser.

use crate::plan::{Act, Bin, Cmp, LeafSource, PAD, Plan, Red, Step};
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
    }
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
    }
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
            Step::MatMul { out, a, b, m, k, n } => {
                let mut o = vec![0f32; m * n];
                {
                    let (av, bv) = (&vals[*a], &vals[*b]);
                    for i in 0..*m {
                        for j in 0..*n {
                            let mut s = 0f32;
                            for l in 0..*k {
                                s += av[i * k + l] * bv[l * n + j];
                            }
                            o[i * n + j] = s;
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
            } => {
                let n = plan.slot_len[*out];
                let mut o = vec![0f32; n];
                for i in 0..n {
                    let ix = vals[*indices][which[i] as usize].round() as usize;
                    o[i] = vals[*table][base[i] as usize + ix * *axis_stride as usize];
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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(target_arch = "wasm32")]
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
#[cfg(not(target_arch = "wasm32"))]
fn bytemuck_cast(s: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no invalid bit patterns; `[f32]` → `[u8]` is a valid
    // reinterpret and the resulting slice covers exactly `len * 4` bytes.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
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
