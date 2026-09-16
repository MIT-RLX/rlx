// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # rlx-peft — parameter-efficient adaptation on rlx graphs
//!
//! LoRA, IA3, AdaLoRA, DoRA and OFT, implemented as operations on an rlx graph
//! rather than as wrappers around another framework.
//!
//! ## Why here and not in a benchmark crate
//!
//! These are model transformations, not evaluation code. `W' = W + (α/r)·BA`
//! applies to any rlx model that has a linear layer — the fact that
//! OpenEEGBench happens to benchmark them is incidental. Putting them in rlx
//! makes them usable for adaptation, not just for reproducing someone's table.
//!
//! ## Semantics
//!
//! Each adapter here reproduces HuggingFace PEFT's definition, which is what
//! published results are measured under:
//!
//! | adapter | update | trainable |
//! |---|---|---|
//! | [`lora_delta`] | `W + (α/r)·B A` | `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}` |
//! | [`ia3_apply`] | `(x Wᵀ) ⊙ l` | `l ∈ ℝ^{out}` |
//! | [`adalora_delta`] | `W + (α/r)·P diag(Λ) Q` | `P`, `Λ`, `Q`, with rank annealing |
//! | [`dora_weight`] | `m ⊙ (W + BA) / ‖W + BA‖_col` | `m ∈ ℝ^{out}` plus LoRA's `A`, `B` |
//! | [`oft`] | `R W`, `R` block-diagonal orthogonal | skew-symmetric `Q` per block |
//!
//! Two initialisation conventions matter and are reproduced rather than chosen:
//! **`B` starts at zero** so the adapted model is identical to the base model
//! at step 0 — a nonzero `B` perturbs a pretrained network before any training
//! and is a common source of "PEFT hurt my model" reports. And **IA3's `l`
//! starts at one**, for the same reason.

pub mod adapters;
pub mod graph;
pub mod oft;

pub use adapters::{LoraConfig, adalora_delta, dora_weight, ia3_apply, lora_delta};
pub use oft::{cayley_orthogonal, oft_weight};

/// Trainable / total parameter accounting for an adapted model.
///
/// The denominator of every PEFT claim: "LoRA matches full fine-tuning" means
/// nothing without it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamBudget {
    pub total: u64,
    pub trainable: u64,
}

impl ParamBudget {
    pub fn pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.trainable as f64 / self.total as f64
        }
    }
}

/// Parameters a LoRA-style adapter adds to one `[out, in]` layer.
///
/// `r(in + out)` — the reason LoRA is cheap, and why the saving vanishes as `r`
/// approaches `min(in, out)`: at that point the factorisation costs more than
/// the matrix it is approximating.
pub fn lora_param_count(in_features: usize, out_features: usize, r: usize) -> u64 {
    (r * (in_features + out_features)) as u64
}

/// Whether a rank is actually saving anything on this layer.
pub fn lora_is_economical(in_features: usize, out_features: usize, r: usize) -> bool {
    lora_param_count(in_features, out_features, r) < (in_features * out_features) as u64
}

/// Row-major matmul helper shared by the adapters: `[m, k] × [k, n]`.
pub(crate) fn matmul(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f64>, String> {
    if !(a.len() == m * k) {
        return Err(format!("lhs is {} values, expected {m}·{k}", a.len()));
    }
    if !(b.len() == k * n) {
        return Err(format!("rhs is {} values, expected {k}·{n}", b.len()));
    }
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|t| a[i * k + t] * b[t * n + j]).sum();
        }
    }
    Ok(out)
}
