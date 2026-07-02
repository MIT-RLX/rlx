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
        }
    }

    Ok(plan.outputs.iter().map(|&s| vals[s].clone()).collect())
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
}
