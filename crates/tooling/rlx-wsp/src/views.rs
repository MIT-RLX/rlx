// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read-only views of a weight tensor, sized for an LLM context window.
//!
//! Raw slices are capped at [`MAX_VALUES`]; everything else is a summary
//! (stats, top-k entries, norms, leading singular directions) labelled with
//! observation / unit / action names so the reader never maps indices by hand.
//!
//! Every view works on a [`Labeled`] tensor: a stored param, or a derived
//! one such as [`compose`]'s matrix product.

use crate::space::{Tensor, WeightSpace};
use serde_json::{Value, json};

/// Largest raw slice a single view returns.
pub const MAX_VALUES: usize = 4096;

/// A tensor with row and column labels.
#[derive(Debug, Clone)]
pub struct Labeled {
    pub name: String,
    pub t: Tensor,
    pub rows: Vec<String>,
    pub cols: Vec<String>,
}

impl Labeled {
    pub fn of(ws: &WeightSpace, name: &str) -> Result<Self, String> {
        Ok(Self {
            name: name.to_string(),
            t: ws.param(name)?.clone(),
            rows: ws.row_labels(name)?,
            cols: ws.col_labels(name)?,
        })
    }

    pub fn transposed(&self) -> Self {
        let t = &self.t;
        let mut data = vec![0f32; t.data.len()];
        for r in 0..t.rows {
            for c in 0..t.cols {
                data[c * t.rows + r] = t.at(r, c);
            }
        }
        Self {
            name: format!("{}ᵀ", self.name),
            t: Tensor {
                rows: t.cols,
                cols: t.rows,
                data,
            },
            rows: self.cols.clone(),
            cols: self.rows.clone(),
        }
    }
}

/// `A·B` of two params (each optionally transposed), labelled by `A`'s rows
/// and `B`'s columns — e.g. `l0.w · l1.w` maps inputs straight to layer-1
/// units, the linear part of a two-layer path.
pub fn compose(ws: &WeightSpace, a: &str, b: &str, ta: bool, tb: bool) -> Result<Labeled, String> {
    let la = Labeled::of(ws, a)?;
    let lb = Labeled::of(ws, b)?;
    let la = if ta { la.transposed() } else { la };
    let lb = if tb { lb.transposed() } else { lb };
    let (m, k, n) = (la.t.rows, la.t.cols, lb.t.cols);
    if lb.t.rows != k {
        return Err(format!(
            "{} is [{m}, {k}] but {} is [{}, {n}]: inner sizes differ",
            la.name, lb.name, lb.t.rows
        ));
    }
    if m * n > 4_000_000 {
        return Err(format!(
            "product would be [{m}, {n}]; too large to summarise"
        ));
    }
    let mut data = vec![0f32; m * n];
    for r in 0..m {
        for p in 0..k {
            let x = la.t.at(r, p);
            if x != 0.0 {
                for c in 0..n {
                    data[r * n + c] += x * lb.t.at(p, c);
                }
            }
        }
    }
    Ok(Labeled {
        name: format!("{}·{}", la.name, lb.name),
        t: Tensor {
            rows: m,
            cols: n,
            data,
        },
        rows: la.rows,
        cols: lb.cols,
    })
}

pub fn stats(ws: &WeightSpace, name: &str) -> Result<Value, String> {
    stats_of(&Labeled::of(ws, name)?)
}

pub fn stats_of(v: &Labeled) -> Result<Value, String> {
    let (t, name) = (&v.t, &v.name);
    let n = t.data.len() as f32;
    let mean = t.data.iter().sum::<f32>() / n;
    let var = t.data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let (lo, hi) = t
        .data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    Ok(json!({
        "name": name, "rows": t.rows, "cols": t.cols,
        "min": lo, "max": hi, "mean": mean, "std": var.sqrt(),
        "frobenius": t.data.iter().map(|v| v * v).sum::<f32>().sqrt(),
        "nonzero": t.data.iter().filter(|v| **v != 0.0).count(),
    }))
}

/// Rows `[r0, r1)` × cols `[c0, c1)`, optionally only rows whose label
/// contains `row_contains`, optionally dropping all-zero rows.
pub fn slice(
    ws: &WeightSpace,
    name: &str,
    rows: Option<(usize, usize)>,
    cols: Option<(usize, usize)>,
    row_contains: Option<&str>,
    nonzero_only: bool,
) -> Result<Value, String> {
    slice_of(
        &Labeled::of(ws, name)?,
        rows,
        cols,
        row_contains,
        nonzero_only,
    )
}

pub fn slice_of(
    v: &Labeled,
    rows: Option<(usize, usize)>,
    cols: Option<(usize, usize)>,
    row_contains: Option<&str>,
    nonzero_only: bool,
) -> Result<Value, String> {
    let (t, name) = (&v.t, &v.name);
    let (r0, r1) = rows.unwrap_or((0, t.rows));
    let (c0, c1) = cols.unwrap_or((0, t.cols));
    if r0 >= r1 || r1 > t.rows || c0 >= c1 || c1 > t.cols {
        return Err(format!("{name}: bad range for [{}, {}]", t.rows, t.cols));
    }
    let (rl, cl) = (&v.rows, &v.cols);
    let picked: Vec<usize> = (r0..r1)
        .filter(|&r| row_contains.is_none_or(|s| rl[r].contains(s)))
        .filter(|&r| !nonzero_only || (c0..c1).any(|c| t.at(r, c) != 0.0))
        .collect();
    let count = picked.len() * (c1 - c0);
    if count > MAX_VALUES {
        return Err(format!(
            "{name}: slice is {count} values (max {MAX_VALUES}); narrow rows/cols or use \
             row_contains / stats / topk / svd"
        ));
    }
    let out: Vec<Value> = picked
        .iter()
        .map(|&r| json!({ "row": r, "label": rl[r], "values": round(&t.data[r * t.cols + c0..r * t.cols + c1]) }))
        .collect();
    Ok(json!({ "name": name, "cols": cl[c0..c1], "rows": out }))
}

/// The `k` largest-magnitude entries.
pub fn topk(ws: &WeightSpace, name: &str, k: usize) -> Result<Value, String> {
    topk_of(&Labeled::of(ws, name)?, k)
}

pub fn topk_of(v: &Labeled, k: usize) -> Result<Value, String> {
    let (t, name, rl, cl) = (&v.t, &v.name, &v.rows, &v.cols);
    let mut idx: Vec<usize> = (0..t.data.len()).collect();
    idx.sort_by(|&a, &b| t.data[b].abs().total_cmp(&t.data[a].abs()));
    let out: Vec<Value> = idx
        .iter()
        .take(k.min(MAX_VALUES))
        .filter(|&&i| t.data[i] != 0.0)
        .map(|&i| {
            let (r, c) = (i / t.cols, i % t.cols);
            json!({ "row": rl[r], "col": cl[c], "value": t.data[i] })
        })
        .collect();
    Ok(json!({ "name": name, "entries": out }))
}

/// L2 norm of every row (`axis = "rows"`) or column (`axis = "cols"`),
/// largest first, zero norms dropped.
pub fn norms(ws: &WeightSpace, name: &str, axis: &str) -> Result<Value, String> {
    norms_of(&Labeled::of(ws, name)?, axis)
}

pub fn norms_of(lv: &Labeled, axis: &str) -> Result<Value, String> {
    let (t, name) = (&lv.t, &lv.name);
    let (labels, n) = match axis {
        "rows" => (&lv.rows, t.rows),
        "cols" => (&lv.cols, t.cols),
        _ => return Err("axis must be `rows` or `cols`".into()),
    };
    let mut v: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let s: f32 = if axis == "rows" {
                (0..t.cols).map(|c| t.at(i, c).powi(2)).sum()
            } else {
                (0..t.rows).map(|r| t.at(r, i).powi(2)).sum()
            };
            (i, s.sqrt())
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    let out: Vec<Value> = v
        .iter()
        .map(|&(i, s)| json!({ "label": labels[i], "norm": s }))
        .collect();
    Ok(json!({ "name": name, "axis": axis, "norms": out }))
}

/// Leading `k` singular values, each with the `top` heaviest row and column
/// labels of its singular vectors (signed).
pub fn svd(ws: &WeightSpace, name: &str, k: usize, top: usize) -> Result<Value, String> {
    svd_of(&Labeled::of(ws, name)?, k, top)
}

pub fn svd_of(lv: &Labeled, k: usize, top: usize) -> Result<Value, String> {
    // Power iteration runs on the Gram of the narrow side.
    if lv.t.cols > lv.t.rows {
        let mut r = svd_of(&lv.transposed(), k, top)?;
        if let Some(cs) = r["components"].as_array_mut() {
            for c in cs {
                let rows = c["rows"].take();
                c["rows"] = c["cols"].take();
                c["cols"] = rows;
            }
        }
        r["name"] = lv.name.clone().into();
        return Ok(r);
    }
    let (t, name, rl, cl) = (&lv.t, &lv.name, &lv.rows, &lv.cols);
    let comps = leading_singular(t, k.min(t.rows).min(t.cols));
    let pick = |v: &[f32], labels: &[String]| -> Vec<Value> {
        let mut i: Vec<usize> = (0..v.len()).collect();
        i.sort_by(|&a, &b| v[b].abs().total_cmp(&v[a].abs()));
        i.iter()
            .take(top)
            .filter(|&&j| v[j].abs() > 1e-6)
            .map(|&j| json!([labels[j], v[j]]))
            .collect()
    };
    let out: Vec<Value> = comps
        .iter()
        .map(|(s, u, v)| json!({ "sigma": s, "rows": pick(u, rl), "cols": pick(v, cl) }))
        .collect();
    Ok(json!({ "name": name, "components": out }))
}

/// `(σ, u, v)` by power iteration on `AᵀA` with deflation.
fn leading_singular(t: &Tensor, k: usize) -> Vec<(f32, Vec<f32>, Vec<f32>)> {
    let (m, n) = (t.rows, t.cols);
    let mut g = vec![0f64; n * n];
    for r in 0..m {
        for i in 0..n {
            let a = t.at(r, i) as f64;
            if a != 0.0 {
                for j in 0..n {
                    g[i * n + j] += a * t.at(r, j) as f64;
                }
            }
        }
    }
    let mut out = Vec::new();
    for c in 0..k {
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + ((i + c) % 7) as f64 * 0.1).collect();
        let mut lambda = 0.0;
        for _ in 0..300 {
            let w: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| g[i * n + j] * v[j]).sum())
                .collect();
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                lambda = 0.0;
                break;
            }
            lambda = norm;
            v = w.iter().map(|x| x / norm).collect();
        }
        if lambda < 1e-10 {
            break;
        }
        for i in 0..n {
            for j in 0..n {
                g[i * n + j] -= lambda * v[i] * v[j];
            }
        }
        let sigma = lambda.sqrt();
        let u: Vec<f32> = (0..m)
            .map(|r| ((0..n).map(|j| t.at(r, j) as f64 * v[j]).sum::<f64>() / sigma) as f32)
            .collect();
        out.push((sigma as f32, u, v.iter().map(|&x| x as f32).collect()));
    }
    out
}

fn round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 1e4).round() / 1e4).collect()
}
