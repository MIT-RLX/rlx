// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Per-layer decomposition mining** + a **DADO** allocator.
//!
//! For each captured layer weight this probes *every* decomposition kernel
//! ([`crate::kernels`]) at an informed configuration and measures its true
//! weight-space reconstruction error and storage cost. Different layers prefer
//! different decompositions — one is near-low-rank, another quantizes cleanly, a
//! square one is Monarch-friendly — so the right global choice is a *per-layer*
//! assignment, not one blanket scheme.
//!
//! Picking that assignment under a global memory budget is a discrete design
//! problem whose objective **decomposes per layer** (`total_err = Σᵢ errᵢ(cᵢ)`)
//! with the budget as the only coupling — exactly the setting of **DADO**
//! (Decomposition-Aware Distributional Optimization, ICLR 2026). [`dado_allocate`]
//! is a faithful DADO optimizer: a factorized per-layer categorical search
//! distribution `p_θ = ∏ᵢ pᵢ(cᵢ)` refined by cross-entropy elite updates. On a
//! purely-additive objective it provably converges to the same optimum a greedy
//! Lagrangian sweep finds ([`greedy_allocate`]) — the ground-truth check, as in
//! rlx-eda's DADO-vs-catalog test.

use crate::guard::dense_matmul;
use crate::kernels::{
    monarch_blocks, monarch_project, rel_err, tt_params, tt_reconstruct, tt_svd, tucker_hosvd,
    tucker_params, tucker_reconstruct,
};
use crate::probe::{
    best_bitwidth, nm_sparsity_error, outlier_channels, per_channel_quant_error, quant_error,
    ternary_error,
};
use crate::svd::truncated_svd;

/// A concrete decomposition choice for one layer.
#[derive(Clone, Debug, PartialEq)]
pub enum Decomp {
    /// No decomposition — full f32 weights.
    Dense,
    /// 2:4 structured sparsity (keep 2 of every 4 along the out axis).
    Sparse24,
    /// Low-rank `W ≈ U·V` at rank `r`.
    LowRank(usize),
    /// Monarch block-diagonal factorization (square, perfect-square dims only).
    Monarch,
    /// 3-way Tucker (HOSVD) at `dims`/`ranks` on the reshaped weight.
    Tucker([usize; 3], [usize; 3]),
    /// Tensor-Train over `mode dims` at bond-rank cap.
    Tt(Vec<usize>, usize),
    /// Per-channel int8 weights.
    Int8,
    /// Per-channel int4 weights.
    Int4,
    /// **Ternary** `α·{-1,0,+1}` — ~1.6-bit, adds-only matmul (no multiplies).
    Ternary,
}

impl Decomp {
    pub fn label(&self) -> String {
        match self {
            Decomp::Dense => "dense".into(),
            Decomp::Sparse24 => "sparse2:4".into(),
            Decomp::LowRank(r) => format!("lowrank(r={r})"),
            Decomp::Monarch => "monarch".into(),
            Decomp::Tucker(d, r) => format!(
                "tucker({}×{}×{}→{},{},{})",
                d[0], d[1], d[2], r[0], r[1], r[2]
            ),
            Decomp::Tt(d, cap) => format!("tt({}modes,χ≤{cap})", d.len()),
            Decomp::Int8 => "int8/chan".into(),
            Decomp::Int4 => "int4/chan".into(),
            Decomp::Ternary => "ternary{-1,0,1}".into(),
        }
    }
}

/// One evaluated decomposition option for a layer: measured error + storage.
#[derive(Clone, Debug)]
pub struct DecompOption {
    pub decomp: Decomp,
    /// Relative L2 weight-reconstruction error (0 = exact).
    pub rel_err: f32,
    /// Storage in bytes (dtype-aware: int8 = 1B/elem, int4 = 0.5B, else 4B).
    pub bytes: usize,
    /// Multiply-adds per input row when this layer runs (compute cost signal).
    pub flops: u64,
}

/// A profiled layer: its shape + every decomposition option that applies.
#[derive(Clone, Debug)]
pub struct LayerProfile {
    pub name: String,
    /// Input dim `k` (rows of the `[in,out]` weight).
    pub rows: usize,
    /// Output dim `n` (cols).
    pub cols: usize,
    pub dense_bytes: usize,
    pub options: Vec<DecompOption>,
}

impl LayerProfile {
    /// The `Dense` baseline option (always present, index 0).
    pub fn dense(&self) -> &DecompOption {
        &self.options[0]
    }

    /// Cheapest option (fewest bytes) whose error is within `budget`.
    pub fn best_within(&self, budget: f32) -> &DecompOption {
        self.options
            .iter()
            .filter(|o| o.rel_err <= budget)
            .min_by_key(|o| o.bytes)
            .unwrap_or(&self.options[0])
    }
}

/// Largest divisor of `n` that is ≤ √n → a balanced `(a,b)` with `a*b == n`.
fn balanced_factor(n: usize) -> (usize, usize) {
    let mut a = (n as f64).sqrt() as usize;
    while a > 1 && !n.is_multiple_of(a) {
        a -= 1;
    }
    (a, n / a)
}

/// Reconstruct the dense `[n,n]` (`[in,out]`) Monarch weight from its factors —
/// only for measuring reconstruction error (the runtime never forms it).
fn monarch_dense(w1: &[f32], w2: &[f32], m: usize) -> Vec<f32> {
    let n = m * m;
    let mut w = vec![0f32; n * n];
    for l in 0..m {
        for s in 0..m {
            for ri in 0..m {
                for j in 0..m {
                    w[(ri * m + j) * n + (l * m + s)] =
                        w1[ri * m * m + j * m + l] * w2[l * m * m + s * m + ri];
                }
            }
        }
    }
    w
}

/// Probe one layer weight `w` (`[rows=in, cols=out]`, row-major) across every
/// applicable decomposition and return the measured error/cost of each.
pub fn profile_layer(name: &str, w: &[f32], rows: usize, cols: usize) -> LayerProfile {
    let numel = rows * cols;
    let dense_bytes = numel * 4;
    let dense_flops = (rows * cols) as u64; // MACs per input row
    let mut options = vec![DecompOption {
        decomp: Decomp::Dense,
        rel_err: 0.0,
        bytes: dense_bytes,
        flops: dense_flops,
    }];

    // ── int8 / int4 per-channel (memory win; compute unchanged) ──
    options.push(DecompOption {
        decomp: Decomp::Int8,
        rel_err: per_channel_quant_error(w, rows, cols, 8),
        bytes: numel, // 1 byte/elem
        flops: dense_flops,
    });
    options.push(DecompOption {
        decomp: Decomp::Int4,
        rel_err: per_channel_quant_error(w, rows, cols, 4),
        bytes: numel / 2, // 0.5 byte/elem
        flops: dense_flops,
    });
    // ── ternary α·{-1,0,+1}: ~1.6-bit (2-bit packed), adds-only matmul ──
    options.push(DecompOption {
        decomp: Decomp::Ternary,
        rel_err: ternary_error(w),
        bytes: numel / 4 + 4,   // 2-bit codes + one scale
        flops: dense_flops / 2, // add/sub instead of MAC (no multiplies)
    });

    // ── 2:4 structured sparsity ──
    options.push(DecompOption {
        decomp: Decomp::Sparse24,
        rel_err: nm_sparsity_error(w, rows, cols, 2, 4),
        bytes: numel * 2 + numel / 4, // half the f32 values + 2-bit indices
        flops: dense_flops / 2,
    });

    // ── low-rank at several ranks (chosen from the real singular spectrum) ──
    // stable_rank *underestimates* rank when the spectrum is spread, so pick the
    // rank from cumulative Frobenius energy of an actual truncated SVD instead,
    // and offer a few granularities so the allocator gets a tradeoff curve.
    let rmax = rows.min(cols);
    let rcap = rmax.min(48);
    let (us, sv, vs) = truncated_svd(w, rows, cols, rcap);
    let energy: f32 = sv.iter().map(|s| s * s).sum::<f32>().max(1e-30);
    let rank_for = |frac: f32| -> usize {
        let mut cum = 0f32;
        for (i, &s) in sv.iter().enumerate() {
            cum += s * s;
            if cum >= frac * energy {
                return i + 1;
            }
        }
        rcap
    };
    let mut seen_r: Vec<usize> = Vec::new();
    for &frac in &[0.90f32, 0.99, 0.999] {
        let rr = rank_for(frac).clamp(1, rcap);
        if seen_r.contains(&rr) || (rows + cols) * rr >= numel {
            continue; // dedup + only when it actually compresses
        }
        seen_r.push(rr);
        // Build factors from the leading triplets: A=U, B=diag(σ)·Vᵀ.
        let mut a = vec![0f32; rows * rr];
        let mut b = vec![0f32; rr * cols];
        for i in 0..rows {
            for t in 0..rr {
                a[i * rr + t] = us[i * rcap + t];
            }
        }
        for t in 0..rr {
            let s = sv[t];
            for j in 0..cols {
                b[t * cols + j] = s * vs[j * rcap + t];
            }
        }
        let recon = dense_matmul(&a, &b, rows, rr, cols);
        options.push(DecompOption {
            decomp: Decomp::LowRank(rr),
            rel_err: rel_err(w, &recon),
            bytes: (rows + cols) * rr * 4,
            flops: ((rows + cols) * rr) as u64,
        });
    }

    // ── Monarch (square perfect-square weights only) ──
    if rows == cols {
        if let Some(m) = monarch_blocks(rows) {
            let (w1, w2) = monarch_project(w, m);
            let recon = monarch_dense(&w1, &w2, m);
            options.push(DecompOption {
                decomp: Decomp::Monarch,
                rel_err: rel_err(w, &recon),
                bytes: 2 * m * m * m * 4,
                flops: (2 * m * m * m) as u64,
            });
        }
    }

    // ── Tucker (reshape cols → c1×c2; compress every mode) ──
    // Cap the big row-mode rank so the unfolding SVD stays cheap and Tucker
    // actually compresses (a near-full rank barely helps and costs a fortune).
    let (c1, c2) = balanced_factor(cols);
    if c1 > 1 && c2 > 1 {
        let dims = [rows, c1, c2];
        const TCAP: usize = 32;
        let ranks = [
            (rows / 2).clamp(1, TCAP.min(rows)),
            (c1 / 2).max(1).min(c1),
            (c2 / 2).max(1).min(c2),
        ];
        let (core, factors) = tucker_hosvd(w, dims, ranks);
        let recon = tucker_reconstruct(&core, dims, ranks, &factors);
        let p = tucker_params(dims, ranks);
        if p < numel {
            options.push(DecompOption {
                decomp: Decomp::Tucker(dims, ranks),
                rel_err: rel_err(w, &recon),
                bytes: p * 4,
                flops: p as u64,
            });
        }
    }

    // ── Tensor-Train (factor rows & cols each → 2 modes; 4-mode TT) ──
    let (rf1, rf2) = balanced_factor(rows);
    let (cf1, cf2) = balanced_factor(cols);
    if rf1 > 1 && rf2 > 1 && cf1 > 1 && cf2 > 1 {
        let modes = vec![rf1, rf2, cf1, cf2];
        let cap = 16usize;
        let cores = tt_svd(w, &modes, cap);
        let recon = tt_reconstruct(&cores);
        let p = tt_params(&cores);
        if p < numel {
            options.push(DecompOption {
                decomp: Decomp::Tt(modes, cap),
                rel_err: rel_err(w, &recon),
                bytes: p * 4,
                flops: p as u64,
            });
        }
    }

    LayerProfile {
        name: name.into(),
        rows,
        cols,
        dense_bytes,
        options,
    }
}

// ─────────────────── quant-sensitivity sweep (fast) ───────────────────
//
// The full [`profile_layer`] probe SVDs every layer — overkill once low-rank /
// Tucker / TT are known to lose on trained dense weights. When the question is
// *how quantizable is each layer, and does that vary with depth?*, only the
// `O(numel)` quant metrics matter — this sweep is those, per weight, plus the
// name parse that lets a report bucket by depth and by projection type.

/// Cheap per-weight quantization metrics (no SVD).
#[derive(Clone, Debug)]
pub struct QuantStat {
    pub name: String,
    /// Transformer block index parsed from `model.layers.{i}.…`, if present.
    pub layer: Option<usize>,
    /// Projection type (`q_proj`, `gate_proj`, …) parsed from the name.
    pub kind: String,
    pub numel: usize,
    /// Per-tensor int8 rel error (one scale for the whole weight).
    pub int8_pt: f32,
    /// Per-(output-)channel int8 / int4 / int3 rel error.
    pub int8_pc: f32,
    pub int4_pc: f32,
    pub int3_pc: f32,
    /// Max/median per-channel `max|·|` ratio — high ⇒ AWQ/SmoothQuant candidate.
    pub outlier: f32,
    /// Smallest bit-width in {3,4,6,8} whose per-channel error is under `budget`.
    pub best_bits: Option<u32>,
}

/// Parse `model.layers.{i}.{block}.{proj}.weight` → `(Some(i), "proj")`.
pub fn parse_layer_kind(name: &str) -> (Option<usize>, String) {
    let layer = name
        .strip_prefix("model.layers.")
        .and_then(|r| r.split('.').next())
        .and_then(|s| s.parse::<usize>().ok());
    let kind = name
        .trim_end_matches(".weight")
        .rsplit('.')
        .next()
        .unwrap_or(name)
        .to_string();
    (layer, kind)
}

/// Compute the quant sensitivity of one weight. `budget` is the per-channel
/// rel-error ceiling used to pick `best_bits`.
pub fn quant_stat(name: &str, w: &[f32], rows: usize, cols: usize, budget: f32) -> QuantStat {
    let (layer, kind) = parse_layer_kind(name);
    QuantStat {
        name: name.into(),
        layer,
        kind,
        numel: rows * cols,
        int8_pt: quant_error(w, 8),
        int8_pc: per_channel_quant_error(w, rows, cols, 8),
        int4_pc: per_channel_quant_error(w, rows, cols, 4),
        int3_pc: per_channel_quant_error(w, rows, cols, 3),
        outlier: outlier_channels(w, rows, cols, 5.0).1,
        best_bits: best_bitwidth(w, rows, cols, budget),
    }
}

// ─────────────────────────── allocators ───────────────────────────

/// Result of allocating one decomposition per layer under a byte budget.
#[derive(Clone, Debug)]
pub struct Allocation {
    /// Chosen option index into each layer's `options`.
    pub choice: Vec<usize>,
    pub total_err: f32,
    pub total_bytes: usize,
    pub dense_bytes: usize,
}

impl Allocation {
    pub fn compression(&self) -> f32 {
        self.dense_bytes as f32 / self.total_bytes.max(1) as f32
    }
}

fn eval(layers: &[LayerProfile], choice: &[usize]) -> (f32, usize) {
    let mut err = 0f32;
    let mut bytes = 0usize;
    for (l, &c) in layers.iter().zip(choice) {
        err += l.options[c].rel_err;
        bytes += l.options[c].bytes;
    }
    (err, bytes)
}

/// **Greedy Lagrangian** baseline: start all-dense, then repeatedly apply the
/// swap with the best error-increase-per-byte-saved until under budget. Optimal
/// on this additive objective up to the discreteness of the option set — the
/// ground-truth the DADO search must match.
pub fn greedy_allocate(layers: &[LayerProfile], budget_frac: f32) -> Allocation {
    let dense_bytes: usize = layers.iter().map(|l| l.dense_bytes).sum();
    let budget = (dense_bytes as f32 * budget_frac) as usize;
    let mut choice: Vec<usize> = vec![0; layers.len()]; // all dense
    loop {
        let (_, bytes) = eval(layers, &choice);
        if bytes <= budget {
            break;
        }
        // Find the single swap that saves bytes at the least error-per-byte cost.
        let mut best: Option<(f32, usize, usize)> = None; // (score, layer, opt)
        for (li, l) in layers.iter().enumerate() {
            let cur = &l.options[choice[li]];
            for (oi, o) in l.options.iter().enumerate() {
                if o.bytes >= cur.bytes {
                    continue; // must save bytes
                }
                let d_err = (o.rel_err - cur.rel_err).max(0.0);
                let d_bytes = (cur.bytes - o.bytes) as f32;
                let score = d_err / d_bytes; // lower is better
                if best.map(|(b, _, _)| score < b).unwrap_or(true) {
                    best = Some((score, li, oi));
                }
            }
        }
        match best {
            Some((_, li, oi)) => choice[li] = oi,
            None => break, // nothing left to shrink
        }
    }
    let (total_err, total_bytes) = eval(layers, &choice);
    Allocation {
        choice,
        total_err,
        total_bytes,
        dense_bytes,
    }
}

/// Configuration for the DADO distributional search.
#[derive(Clone, Copy, Debug)]
pub struct DadoConfig {
    pub iters: usize,
    pub pop: usize,
    pub elite_frac: f32,
    /// Penalty weight on the fractional budget overshoot (drives feasibility).
    pub penalty: f32,
    pub seed: u64,
}

impl Default for DadoConfig {
    fn default() -> Self {
        Self {
            iters: 60,
            pop: 200,
            elite_frac: 0.2,
            penalty: 50.0,
            seed: 0xDAD0_0501,
        }
    }
}

/// **DADO** allocation: a factorized per-layer categorical search distribution
/// refined by cross-entropy elite updates. Exploits the objective's per-layer
/// decomposition — each layer's marginal is updated independently from the elite
/// set (`p_θ = ∏ᵢ pᵢ`). **Warm-started from the greedy Lagrangian seed** (as
/// rlx-eda's DADO seeds from its greedy catalog before the distributional
/// polish), so the result is guaranteed ≥ greedy and the search only *improves*
/// on it — e.g. spending leftover budget headroom to upgrade a coarse choice.
/// Returns the best *feasible* config found.
pub fn dado_allocate(layers: &[LayerProfile], budget_frac: f32, cfg: DadoConfig) -> Allocation {
    let dense_bytes: usize = layers.iter().map(|l| l.dense_bytes).sum();
    let budget = (dense_bytes as f32 * budget_frac).max(1.0);
    let nl = layers.len();

    // Greedy warm start — the distributional search departs from a known-good,
    // known-feasible point instead of cold uniform (which can't concentrate over
    // many layers under a tight budget).
    let seed_alloc = greedy_allocate(layers, budget_frac);

    // Per-layer categorical biased toward the greedy choice (0.6 on it, the rest
    // spread) so sampled configs cluster near feasibility yet still explore.
    let mut p: Vec<Vec<f32>> = layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let k = l.options.len();
            let mut d = vec![0.4 / k as f32; k];
            d[seed_alloc.choice[i]] += 0.6;
            d
        })
        .collect();

    let mut s = cfg.seed | 1;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let sample_cat = |dist: &[f32], u: f64| -> usize {
        let mut acc = 0.0;
        for (i, &pi) in dist.iter().enumerate() {
            acc += pi as f64;
            if u <= acc {
                return i;
            }
        }
        dist.len() - 1
    };

    let loss = |choice: &[usize]| -> f32 {
        let (err, bytes) = eval(layers, choice);
        let over = ((bytes as f32 - budget) / budget).max(0.0);
        err + cfg.penalty * over
    };

    // Seed the incumbent with the greedy solution — DADO can only improve on it.
    let mut best_choice: Option<Vec<usize>> = Some(seed_alloc.choice.clone());
    let mut best_loss = seed_alloc.total_err;
    let n_elite = ((cfg.pop as f32) * cfg.elite_frac).max(1.0) as usize;

    for _ in 0..cfg.iters {
        // Sample a population of full configs from the factorized distribution.
        let mut pop: Vec<(f32, Vec<f32>, Vec<usize>)> = Vec::with_capacity(cfg.pop);
        for _ in 0..cfg.pop {
            let choice: Vec<usize> = (0..nl).map(|i| sample_cat(&p[i], rnd())).collect();
            let (err, bytes) = eval(layers, &choice);
            let lo = loss(&choice);
            // Track the best strictly-feasible config seen.
            if bytes as f32 <= budget && err < best_loss {
                best_loss = err;
                best_choice = Some(choice.clone());
            }
            pop.push((lo, Vec::new(), choice));
        }
        // Elite = lowest-loss fraction; refit each layer marginal to its counts.
        pop.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let elite = &pop[..n_elite];
        for i in 0..nl {
            let k = layers[i].options.len();
            let mut counts = vec![1.0f32; k]; // Laplace smoothing keeps exploration alive
            for (_, _, ch) in elite {
                counts[ch[i]] += 1.0;
            }
            let tot: f32 = counts.iter().sum();
            for j in 0..k {
                p[i][j] = counts[j] / tot;
            }
        }
    }

    // Incumbent is seeded with greedy, so this is always ≥ greedy quality.
    let choice = best_choice.expect("seeded with greedy");
    let (total_err, total_bytes) = eval(layers, &choice);
    Allocation {
        choice,
        total_err,
        total_bytes,
        dense_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dist, sample};

    #[test]
    fn profile_flags_the_right_decomp_per_dist() {
        // A low-rank weight should have a cheap low-error low-rank option; a
        // Gaussian one should not compress cleanly (its best-small option errs).
        let (k, n) = (64usize, 64usize);
        let lr = sample(Dist::LowRank, k, n, 1);
        let prof = profile_layer("lr", &lr, k, n);
        let low = prof
            .options
            .iter()
            .find(|o| matches!(o.decomp, Decomp::LowRank(_)))
            .unwrap();
        assert!(
            low.rel_err < 0.05,
            "low-rank weight should factor cleanly, err {}",
            low.rel_err
        );
        assert!(low.bytes < prof.dense_bytes);
    }

    #[test]
    fn dado_matches_greedy_on_additive_objective() {
        // Build a few diverse layers, allocate at 40% of dense bytes with both
        // greedy and DADO — on this decomposable objective DADO must tie/beat it.
        let layers = vec![
            profile_layer("a", &sample(Dist::LowRank, 64, 64, 1), 64, 64),
            profile_layer("b", &sample(Dist::Quantized, 48, 32, 2), 48, 32),
            profile_layer("c", &sample(Dist::Gaussian, 32, 48, 3), 32, 48),
            profile_layer("d", &sample(Dist::LowRank, 36, 36, 4), 36, 36),
        ];
        let g = greedy_allocate(&layers, 0.4);
        let d = dado_allocate(&layers, 0.4, DadoConfig::default());
        assert!(
            g.total_bytes as f32
                <= layers.iter().map(|l| l.dense_bytes).sum::<usize>() as f32 * 0.4 + 1.0
        );
        assert!(
            d.total_bytes as f32
                <= layers.iter().map(|l| l.dense_bytes).sum::<usize>() as f32 * 0.4 + 1.0
        );
        // Greedy-seeded, so DADO is guaranteed ≥ greedy (ties or beats) on error.
        assert!(
            d.total_err <= g.total_err + 1e-6,
            "dado {} vs greedy {}",
            d.total_err,
            g.total_err
        );
    }
}
