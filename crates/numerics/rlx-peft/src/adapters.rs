// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! LoRA, IA3, AdaLoRA and DoRA.

use crate::matmul;

/// LoRA hyperparameters, named as PEFT names them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoraConfig {
    /// Rank.
    pub r: usize,
    /// Scaling numerator; the update is multiplied by `alpha / r`.
    pub alpha: f64,
}

impl LoraConfig {
    /// `alpha / r`. PEFT computes this once at construction; a caller who
    /// forgets it trains an update `r/alpha` times too small, which looks like
    /// slow convergence rather than a bug.
    pub fn scaling(&self) -> f64 {
        ensure_nonzero(self.r);
        self.alpha / self.r as f64
    }
}

fn ensure_nonzero(r: usize) {
    assert!(r > 0, "LoRA rank must be positive");
}

/// `ΔW = (α/r) · B A` for `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}`; returns `[out, in]`.
///
/// **`B` is initialised to zero in PEFT**, so `ΔW = 0` at step 0 and the adapted
/// model reproduces the base model exactly. This function does not initialise —
/// it is the update given weights — but the convention is why a correct LoRA
/// integration is a no-op before training, and that is worth testing.
pub fn lora_delta(
    a: &[f64],
    b: &[f64],
    in_features: usize,
    out_features: usize,
    cfg: LoraConfig,
) -> Result<Vec<f64>, String> {
    if !(cfg.r > 0) {
        return Err(format!("rank must be positive, got {}", cfg.r));
    }
    if !(a.len() == cfg.r * in_features) {
        return Err(format!(
            "A is {} values, expected {}·{in_features}",
            a.len(),
            cfg.r
        ));
    }
    if !(b.len() == out_features * cfg.r) {
        return Err(format!(
            "B is {} values, expected {out_features}·{}",
            b.len(),
            cfg.r
        ));
    }
    let mut d = matmul(b, a, out_features, cfg.r, in_features)?;
    let s = cfg.scaling();
    for v in &mut d {
        *v *= s;
    }
    Ok(d)
}

/// IA3: rescale the layer OUTPUT by a learned vector, `y = (x Wᵀ) ⊙ l`.
///
/// Not a weight update — IA3 adds no `ΔW` at all, which is why it is the
/// cheapest of the family (`out` parameters, against LoRA's `r(in + out)`).
/// `l` is initialised to ones so the adapter starts as the identity.
pub fn ia3_apply(
    y: &[f64],
    l: &[f64],
    rows: usize,
    out_features: usize,
) -> Result<Vec<f64>, String> {
    if !(y.len() == rows * out_features) {
        return Err(format!(
            "y is {} values, expected {rows}·{out_features}",
            y.len()
        ));
    }
    if !(l.len() == out_features) {
        return Err(format!("l is {} values, expected {out_features}", l.len()));
    }
    Ok((0..rows * out_features)
        .map(|i| y[i] * l[i % out_features])
        .collect())
}

/// AdaLoRA: `ΔW = (α/r) · P diag(Λ) Q`, an explicit SVD-form update.
///
/// The difference from LoRA is that the rank is **allocated**, not fixed:
/// training prunes singular values, so `Λ` entries go to zero and the effective
/// rank anneals from `init_r` toward `target_r`. Passing a `Λ` with zeros is
/// therefore normal, not degenerate.
pub fn adalora_delta(
    p: &[f64],
    lambda: &[f64],
    q: &[f64],
    in_features: usize,
    out_features: usize,
    cfg: LoraConfig,
) -> Result<Vec<f64>, String> {
    let r = cfg.r;
    if !(r > 0) {
        return Err(format!("rank must be positive, got {r}"));
    }
    if !(p.len() == out_features * r) {
        return Err(format!(
            "P is {} values, expected {out_features}·{r}",
            p.len()
        ));
    }
    if !(lambda.len() == r) {
        return Err(format!("Lambda is {} values, expected {r}", lambda.len()));
    }
    if !(q.len() == r * in_features) {
        return Err(format!(
            "Q is {} values, expected {r}·{in_features}",
            q.len()
        ));
    }

    // P diag(Λ): scale column j of P by Λ[j].
    let mut pl = vec![0f64; out_features * r];
    for i in 0..out_features {
        for j in 0..r {
            pl[i * r + j] = p[i * r + j] * lambda[j];
        }
    }
    let mut d = matmul(&pl, q, out_features, r, in_features)?;
    let s = cfg.scaling();
    for v in &mut d {
        *v *= s;
    }
    Ok(d)
}

/// DoRA: decompose into magnitude and direction,
/// `W' = m ⊙ (W + ΔW) / ‖W + ΔW‖_col`.
///
/// The norm is taken **per output row** (PEFT's `norm(dim=1)` over an
/// `[out, in]` weight), and `m` is initialised to the base weight's own row
/// norms so `W' == W` before training. Normalising over the wrong axis produces
/// a model that trains and is quietly not DoRA.
pub fn dora_weight(
    w: &[f64],
    delta: &[f64],
    m: &[f64],
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f64>, String> {
    if !(w.len() == out_features * in_features) {
        return Err(format!(
            "W is {} values, expected {out_features}·{in_features}",
            w.len()
        ));
    }
    if !(delta.len() == w.len()) {
        return Err(format!(
            "delta is {} values, expected {}",
            delta.len(),
            w.len()
        ));
    }
    if !(m.len() == out_features) {
        return Err(format!("m is {} values, expected {out_features}", m.len()));
    }

    let mut out = vec![0f64; w.len()];
    for i in 0..out_features {
        let row: Vec<f64> = (0..in_features)
            .map(|j| w[i * in_features + j] + delta[i * in_features + j])
            .collect();
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        // A zero row has no direction; leave it zero rather than dividing.
        let scale = if norm > 0.0 { m[i] / norm } else { 0.0 };
        for j in 0..in_features {
            out[i * in_features + j] = row[j] * scale;
        }
    }
    Ok(out)
}

/// The magnitude vector that makes DoRA a no-op at initialisation: the base
/// weight's per-row norms.
pub fn dora_init_magnitude(
    w: &[f64],
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f64>, String> {
    if !(w.len() == out_features * in_features) {
        return Err(format!(
            "W is {} values, expected {out_features}·{in_features}",
            w.len()
        ));
    }
    Ok((0..out_features)
        .map(|i| {
            (0..in_features)
                .map(|j| w[i * in_features + j].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect())
}
