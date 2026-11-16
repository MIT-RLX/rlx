// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! e-means: mini-batch k-means with exponential-moving-average statistics.
//!
//! Implements Algorithm 1 of Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze
//! and Jégou, *"e-means: outperforming k-means (in one-tenth the epochs)"*,
//! arXiv:2611.09501 (2026), <https://arxiv.org/abs/2611.09501>. The reference
//! implementation is `faiss/contrib/e_means` (`NumpyEmeans`); this port follows
//! it line for line, including its RNG streams (via [`rlx_rng::numpy`]), so a
//! run with the same seed visits the same rows in the same order.
//!
//! # Algorithm
//!
//! Each cluster `j` keeps a mass `N_j` and a position `c_j = S_j / N_j`. For
//! every batch of `B` rows (Algorithm 1, lines 6–8, Equation 2):
//!
//! ```text
//! n_j, s_j = count and sum of the batch rows nearest to c_j
//! N_j ← γ·N_j + α·n_j                       (α = 1 − γ)
//! c_j ← c_j + (α·n_j / N_j) · (s_j/n_j − c_j)
//! ```
//!
//! - `α` follows a cosine anneal from `α₀` to `0.1·α₀` over all steps
//!   (Equation 5, [`scheduled_alpha`]).
//! - Every 5 steps, clusters below `τ` times the mean mass are revived by
//!   splitting donors sampled in proportion to their mass (line 9). The paper's
//!   ablation (Table 1) finds this the component with the largest effect.
//! - After the last epoch the centroids are the exact means of the rows
//!   assigned during that epoch (line 12), which cannot raise training MSE.
//! - `B` and `α₀` come from fitted scaling laws ([`resolve_recipe`]), so the
//!   epoch budget `T` is the only required setting.
//!
//! # Two execution paths
//!
//! - [`Emeans`] — the host trainer: fused nearest-centroid search (rayon over
//!   rows, no `[B, k]` matrix) plus the EMA, revival and closing on the host.
//! - [`step_graph`] — one training step as an `rlx_ir::Graph` (matmul → ArgMin
//!   → ScatterAdd → EMA), so any rlx backend can run the step; the host keeps
//!   the schedule and revival. Its outputs match [`Emeans::update`].
//!   [`step_graph_with`] runs the assignment in f64, f16 or bf16.
//! - `device::DeviceEmeans` (feature `emeans-device`) — the step graph with
//!   the dataset, centroids, masses and accumulators resident on the device;
//!   a step uploads only its row indices.

#[cfg(feature = "emeans-device")]
pub mod device;

use rayon::prelude::*;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_rng::numpy::RandomState;

/// `ε` in Equation 5: the cosine schedule's final step over its initial step.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;
/// Batch updates between revival checks (Algorithm 1, line 9).
pub const CHECK_UNUSED_EVERY: usize = 5;
/// Initial EMA mass `N_j`. faiss uses `2⁻¹⁶`; the paper uses `1e-10`.
pub const INIT_USAGE: f32 = 1.0 / 65536.0;
/// Relative coordinate perturbation `δ` for donor splits.
pub const SPLIT_EPSILON: f32 = 1e-4;
/// Default revival threshold `τ`: revive `N_j < τ · mean(N)`.
pub const UNUSED_THRESHOLD: f32 = 0.1;
/// At or below this dimension the default `τ` is 0 (revival off): on 2-D data
/// revival raised held-out error in the reference's measurements.
pub const NO_REVIVAL_MAX_D: usize = 2;
/// At or below this dimension [`BatchLaw::Auto`] picks [`BatchLaw::Quality`].
pub const AUTO_QUALITY_MAX_D: usize = 16;
/// [`Init::KMeansPlusPlus`] seeds from at most this many rows per cluster.
pub const KMEANS_PP_ROWS_PER_CLUSTER: usize = 32;

/// Batch law `B = C_B · k^b_k · n^b_n · T^b_T` as `(C_B, b_k, b_n, b_T)`. These
/// are the coefficients shipped with the reference implementation (fitted on
/// 100K–10M rows); they differ from the paper's Table 2.
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
/// One eighth of the throughput coefficient, before rounding and caps.
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);
/// Step law `α₀ = C_α · k^a_k · B^a_B · n^a_n · T^a_T` as `(C_α, a_k, a_B, a_n, a_T)`.
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

/// Which fitted law sets the batch size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchLaw {
    /// [`BatchLaw::Quality`] when `d ≤ 16`, else [`BatchLaw::Throughput`].
    Auto,
    /// Larger batches, fewer updates: fastest per epoch.
    Throughput,
    /// Batches about 8× smaller: lower final MSE.
    Quality,
}

/// How [`Emeans::train`] picks the initial centroids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Init {
    /// `k` distinct rows drawn with the seed, as the reference does.
    #[default]
    RandomRows,
    /// k-means++ (Arthur & Vassilvitskii, 2007): each next centroid is a row
    /// drawn with probability proportional to its squared distance to the
    /// nearest centroid so far, over a seeded subsample of at most
    /// [`KMEANS_PP_ROWS_PER_CLUSTER`]` · k` rows (`O(32·k²·d)`).
    ///
    /// Not in the paper. Random rows can leave a well-separated region without
    /// a centroid, and revival only re-seeds clusters whose mass died: on 20
    /// blobs in 16-D with `k = 48`, seed 1 ended at 4.8× the MSE of seeds 2–3
    /// (faiss's `NumpyEmeans` lands on the same 8.43, and so does Lloyd from
    /// those rows).
    KMeansPlusPlus,
}

/// Training settings. Only the epoch budget usually needs changing.
#[derive(Clone, Debug)]
pub struct EmeansConfig {
    /// Epoch budget `T`; fractional values run a partial pass.
    pub nepochs: f64,
    /// Overrides the batch law. `α₀` is then still set by the step law.
    pub batch_size: Option<usize>,
    /// Overrides the step law.
    pub alpha0: Option<f64>,
    /// Batch law used when `batch_size` is `None`.
    pub batch_law: BatchLaw,
    /// Revival threshold `τ`; `None` means 0.1, or 0 when `d ≤ 2`.
    pub unused_threshold: Option<f32>,
    /// Seeds initialization, per-epoch shuffles and revival.
    pub seed: u32,
    /// Initial centroids when [`Emeans::train`] is not given any.
    pub init: Init,
    /// Keep each row's last-epoch label (returned by [`Emeans::train`]).
    pub capture_assignments: bool,
}

impl Default for EmeansConfig {
    fn default() -> Self {
        Self {
            nepochs: 5.0,
            batch_size: None,
            alpha0: None,
            batch_law: BatchLaw::Auto,
            unused_threshold: None,
            seed: 1234,
            init: Init::RandomRows,
            capture_assignments: true,
        }
    }
}

/// The resolved training plan for one run.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    /// Statistical batch size `B`.
    pub batch_size: usize,
    /// Initial EMA step `α₀`.
    pub alpha0: f64,
    /// The law that chose `B`, or `None` when `batch_size` was given.
    pub batch_law: Option<BatchLaw>,
    /// Rows per epoch: whole passes, an optional fractional prefix, then the
    /// closing pass.
    pub epoch_rows: Vec<usize>,
    /// Updates per epoch (`ceil(rows / B)`; the last batch keeps the remainder).
    pub epoch_steps: Vec<usize>,
    /// Total updates; the cosine schedule spans these.
    pub total_steps: usize,
    /// Resolved revival threshold `τ`.
    pub unused_threshold: f32,
}

/// Round a positive value to the nearest power of two.
pub fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    (if value / low < high / value {
        low
    } else {
        high
    }) as usize
}

/// Initial EMA step coupled to a batch size by the fitted step law, clamped
/// to `[1e-4, 1]`.
pub fn step_law(n: usize, k: usize, nepochs: f64, batch_size: usize) -> f64 {
    let (c, a_k, a_b, a_n, a_t) = ALPHA_LAW;
    let alpha0 = c
        * (k as f64).powf(a_k)
        * (batch_size as f64).powf(a_b)
        * (n as f64).powf(a_n)
        * nepochs.powf(a_t);
    alpha0.clamp(1e-4, 1.0)
}

/// Cosine-anneal an EMA step from `alpha0` to `0.1 · alpha0` (Equation 5).
pub fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Rows per epoch for a (possibly fractional) budget: `T−1` whole passes, the
/// fractional prefix rounded ties-to-even, then the closing pass.
pub fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor();
    let remainder = ((nepochs - whole) * n as f64).round_ties_even() as usize;
    if whole == 0.0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole as usize - 1];
    if remainder != 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Seed of one epoch's permutation, independent of any other RNG state.
pub fn epoch_seed(seed: u32, epoch: usize) -> u32 {
    ((seed as u64 * 1_000_003 + epoch as u64) % (1u64 << 32)) as u32
}

/// Resolve `B`, `α₀`, the epoch plan and `τ` for `n` rows of dimension `d`.
pub fn resolve_recipe(n: usize, k: usize, d: usize, cfg: &EmeansConfig) -> Recipe {
    assert!(
        n > 0 && k > 0,
        "e-means: need n > 0 rows and k > 0 clusters"
    );
    assert!(cfg.nepochs > 0.0, "e-means: nepochs must be positive");
    let (batch_size, batch_law) = match cfg.batch_size {
        Some(b) => (b.clamp(1, n), None),
        None => {
            let law = match cfg.batch_law {
                BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
                BatchLaw::Auto => BatchLaw::Throughput,
                law => law,
            };
            let (c, b_k, b_n, b_t) = match law {
                BatchLaw::Quality => QUALITY_LAW,
                _ => THROUGHPUT_LAW,
            };
            let raw = c * (k as f64).powf(b_k) * (n as f64).powf(b_n) * cfg.nepochs.powf(b_t);
            let mut b = round_power_of_two(raw);
            // The two caps intentionally differ: they reproduce the two fits.
            b = match law {
                BatchLaw::Throughput => b.min(round_power_of_two(n as f64 / 2.0)),
                _ => b.min(n / 2),
            };
            (b.min(n).max(1), Some(law))
        }
    };
    let alpha0 = cfg
        .alpha0
        .unwrap_or_else(|| step_law(n, k, cfg.nepochs, batch_size));
    let rows = epoch_rows(n, cfg.nepochs);
    let epoch_steps: Vec<usize> = rows.iter().map(|r| r.div_ceil(batch_size)).collect();
    let unused_threshold = cfg.unused_threshold.unwrap_or(if d <= NO_REVIVAL_MAX_D {
        0.0
    } else {
        UNUSED_THRESHOLD
    });
    Recipe {
        batch_size,
        alpha0,
        batch_law,
        total_steps: epoch_steps.iter().sum(),
        epoch_rows: rows,
        epoch_steps,
        unused_threshold,
    }
}

/// Nearest centroid of every row by squared L2: returns `(distances, labels)`.
///
/// Uses `‖x‖² + ‖c‖² − 2·x·c` (clamped at 0) with a running per-row minimum,
/// so the `[n, k]` distance matrix is never stored. Parallel over rows.
pub fn assign_l2(x: &[f32], centroids: &[f32], d: usize) -> (Vec<f32>, Vec<u32>) {
    assert!(d > 0 && x.len().is_multiple_of(d) && centroids.len().is_multiple_of(d));
    let k = centroids.len() / d;
    let c_norm: Vec<f32> = centroids
        .chunks_exact(d)
        .map(|c| c.iter().map(|v| v * v).sum())
        .collect();
    x.par_chunks_exact(d)
        .map(|xi| {
            let mut best = f32::INFINITY;
            let mut best_j = 0u32;
            for j in 0..k {
                let cj = &centroids[j * d..(j + 1) * d];
                let dot: f32 = xi.iter().zip(cj).map(|(a, b)| a * b).sum();
                let proxy = c_norm[j] - 2.0 * dot;
                if proxy < best {
                    best = proxy;
                    best_j = j as u32;
                }
            }
            let x_norm: f32 = xi.iter().map(|v| v * v).sum();
            ((x_norm + best).max(0.0), best_j)
        })
        .unzip()
}

/// Per-cluster statistics of one batch, gathered against a fixed centroid
/// snapshot. A large batch may be gathered in pieces with repeated
/// [`Emeans::accumulate`] calls before one [`Emeans::update`].
#[derive(Clone, Debug)]
pub struct BatchStats {
    /// `s_j`, row-major `[k, d]`.
    pub sums: Vec<f32>,
    /// `n_j`, `[k]`.
    pub counts: Vec<f32>,
    /// Sum of squared distances to the snapshot centroids.
    pub objective: f64,
    /// Rows gathered so far.
    pub n: usize,
    /// Labels of the gathered rows, in order (filled only during the last epoch
    /// with `capture_assignments`).
    pub labels: Vec<u32>,
}

impl BatchStats {
    /// Empty statistics for `k` clusters of dimension `d`.
    pub fn new(k: usize, d: usize) -> Self {
        Self {
            sums: vec![0.0; k * d],
            counts: vec![0.0; k],
            objective: 0.0,
            n: 0,
            labels: Vec::new(),
        }
    }
}

/// Per-epoch report.
#[derive(Clone, Debug, PartialEq)]
pub struct EpochStats {
    /// Sum of squared distances measured while the centroids moved.
    pub objective: f64,
    /// `k · Σ n_j² / rows²` (1 = perfectly balanced).
    pub imbalance_factor: f64,
    /// Clusters revived during the epoch.
    pub nsplit: usize,
}

/// Host e-means trainer (see the [module docs](self)).
///
/// One object is one run: [`Emeans::train`], or the caller-driven loop
/// [`begin`](Emeans::begin) → per batch [`accumulate`](Emeans::accumulate) +
/// [`update`](Emeans::update) → [`end_epoch`](Emeans::end_epoch) →
/// [`finalize`](Emeans::finalize). Every batch must hold exactly the recipe's
/// batch size except the last one of each epoch, which holds the remainder.
pub struct Emeans {
    d: usize,
    k: usize,
    cfg: EmeansConfig,
    recipe: Option<Recipe>,
    centroids: Vec<f32>,
    usage: Vec<f32>,
    split_rng: Option<RandomState>,
    n_splits: usize,
    step: usize,
    epoch: usize,
    epoch_rows_done: usize,
    splits_before: usize,
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f32>>,
    objective: f64,
    epoch_stats: Vec<EpochStats>,
    finalized: bool,
}

impl Emeans {
    /// A trainer for `k` clusters of dimension `d`.
    pub fn new(d: usize, k: usize, cfg: EmeansConfig) -> Self {
        assert!(d > 0 && k > 0, "e-means: d and k must be positive");
        Self {
            d,
            k,
            cfg,
            recipe: None,
            centroids: Vec::new(),
            usage: Vec::new(),
            split_rng: None,
            n_splits: 0,
            step: 0,
            epoch: 0,
            epoch_rows_done: 0,
            splits_before: 0,
            pass_counts: Vec::new(),
            pass_sums: None,
            objective: 0.0,
            epoch_stats: Vec::new(),
            finalized: false,
        }
    }

    /// The resolved plan (after [`begin`](Self::begin)).
    pub fn recipe(&self) -> Option<&Recipe> {
        self.recipe.as_ref()
    }
    /// Current centroid positions `[k, d]` (closing means after finalize).
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    /// Current EMA masses `N_j`.
    pub fn usage(&self) -> &[f32] {
        &self.usage
    }
    /// Clusters revived so far.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }
    /// One entry per finished epoch.
    pub fn epoch_stats(&self) -> &[EpochStats] {
        &self.epoch_stats
    }
    /// The step `α` the next [`update`](Self::update) will apply.
    pub fn current_alpha(&self) -> f64 {
        let r = self.running_recipe();
        scheduled_alpha(r.alpha0, self.step, r.total_steps)
    }
    /// Rows the next batch must contain.
    pub fn next_batch_rows(&self) -> usize {
        let r = self.running_recipe();
        r.batch_size
            .min(r.epoch_rows[self.epoch] - self.epoch_rows_done)
    }

    fn running_recipe(&self) -> &Recipe {
        assert!(!self.finalized, "e-means: run already finalized");
        self.recipe
            .as_ref()
            .expect("e-means: call begin before training")
    }

    fn last_epoch(&self) -> bool {
        self.epoch + 1 == self.running_recipe().epoch_rows.len()
    }

    /// Start a run over `n` rows from initial centroids `[k, d]`.
    pub fn begin(&mut self, n: usize, init_centroids: &[f32]) {
        assert!(self.recipe.is_none(), "e-means: one run per object");
        assert_eq!(
            init_centroids.len(),
            self.k * self.d,
            "e-means: initial centroids must be [k, d]"
        );
        self.recipe = Some(resolve_recipe(n, self.k, self.d, &self.cfg));
        self.centroids = init_centroids.to_vec();
        self.usage = vec![INIT_USAGE; self.k];
        self.split_rng = Some(RandomState::new(self.cfg.seed));
        self.start_epoch();
    }

    fn start_epoch(&mut self) {
        self.epoch_rows_done = 0;
        self.splits_before = self.n_splits;
        self.pass_counts = vec![0.0; self.k];
        self.objective = 0.0;
        self.pass_sums = self.last_epoch().then(|| vec![0.0; self.k * self.d]);
    }

    /// Assign the rows `x` (`[m, d]`) to the current centroids and add their
    /// counts, sums and objective to `stats`. Rows accumulate in input order.
    pub fn accumulate(&self, x: &[f32], stats: &mut BatchStats) {
        let d = self.d;
        assert_eq!(x.len() % d, 0, "e-means: rows must be [m, d]");
        let (dist, labels) = assign_l2(x, &self.centroids, d);
        for (row, &l) in x.chunks_exact(d).zip(&labels) {
            let l = l as usize;
            for (s, v) in stats.sums[l * d..(l + 1) * d].iter_mut().zip(row) {
                *s += v;
            }
            stats.counts[l] += 1.0;
        }
        stats.objective += dist.iter().map(|&v| v as f64).sum::<f64>();
        stats.n += labels.len();
        if self.cfg.capture_assignments && self.last_epoch() {
            stats.labels.extend_from_slice(&labels);
        }
    }

    /// Apply one complete batch (Algorithm 1, line 8), then revive dead
    /// clusters every [`CHECK_UNUSED_EVERY`] steps (line 9).
    pub fn update(&mut self, stats: &BatchStats) {
        let expected = self.next_batch_rows();
        assert_eq!(
            stats.n, expected,
            "e-means: batch must hold exactly {expected} rows"
        );
        let alpha = self.current_alpha();
        self.apply_statistics(stats, alpha);
        if (self.step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
            self.revive_unused();
        }
        self.step += 1;
        self.epoch_rows_done += expected;
    }

    /// Overwrite the model state with the outputs of [`step_graph`] (run with
    /// the `alpha` from [`current_alpha`](Self::current_alpha)) and advance
    /// the schedule exactly as [`update`](Self::update) would. `stats` is the
    /// graph's `[k, d+1]` statistics output, kept for the closing means.
    pub fn update_from_step(
        &mut self,
        centroids: &[f32],
        usage: &[f32],
        stats: &[f32],
        objective: f64,
    ) {
        let (k, d) = (self.k, self.d);
        assert_eq!(centroids.len(), k * d);
        assert_eq!(usage.len(), k);
        assert_eq!(stats.len(), k * (d + 1));
        let expected = self.next_batch_rows();
        self.objective += objective;
        let mut rows = 0.0f64;
        for j in 0..k {
            let row = &stats[j * (d + 1)..(j + 1) * (d + 1)];
            self.pass_counts[j] += row[d] as f64;
            rows += row[d] as f64;
            if let Some(ps) = self.pass_sums.as_mut() {
                for (s, v) in ps[j * d..(j + 1) * d].iter_mut().zip(&row[..d]) {
                    *s += v;
                }
            }
        }
        assert_eq!(
            rows as usize, expected,
            "e-means: step covered {rows} rows, expected {expected}"
        );
        self.centroids.copy_from_slice(centroids);
        self.usage.copy_from_slice(usage);
        if (self.step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
            self.revive_unused();
        }
        self.step += 1;
        self.epoch_rows_done += expected;
    }

    /// Equation 2 on the mass/position form. f32 arithmetic in the same order
    /// as the reference so both runs round alike.
    fn apply_statistics(&mut self, stats: &BatchStats, alpha: f64) {
        self.objective += stats.objective;
        for (pc, &c) in self.pass_counts.iter_mut().zip(&stats.counts) {
            *pc += c as f64;
        }
        if let Some(ps) = self.pass_sums.as_mut() {
            for (p, s) in ps.iter_mut().zip(&stats.sums) {
                *p += s;
            }
        }
        ema_step(
            &mut self.centroids,
            &mut self.usage,
            &stats.sums,
            &stats.counts,
            self.d,
            alpha,
        );
    }

    /// Split donors into clusters whose mass fell below `τ · mean(N)`.
    /// Donors are drawn without replacement in proportion to their mass
    /// (Gumbel top-k), as in the reference.
    fn revive_unused(&mut self) {
        for (src, dst) in self.plan_splits() {
            self.split(src, dst);
        }
    }

    /// `(donor, dead)` pairs for one revival check, from the current masses.
    /// Draws from the revival stream only when something is dead.
    fn plan_splits(&mut self) -> Vec<(usize, usize)> {
        let k = self.k;
        let tau = self.running_recipe().unused_threshold;
        let total: f32 = self.usage.iter().sum();
        let threshold = total * tau / k as f32;
        let unused: Vec<bool> = self.usage.iter().map(|&u| u < threshold).collect();
        let donors: Vec<usize> = (0..k)
            .filter(|&j| !unused[j] && self.usage[j] > 0.0)
            .collect();
        let nsplit = unused.iter().filter(|&&u| u).count().min(donors.len());
        if nsplit == 0 {
            return Vec::new();
        }
        let rng = self.split_rng.as_mut().expect("e-means: begin first");
        let uniform: Vec<f64> = (0..k)
            .map(|_| rng.random_double().max(f64::MIN_POSITIVE))
            .collect();
        let mut keyed: Vec<(f64, usize)> = donors
            .iter()
            .map(|&j| (self.usage[j].ln() as f64 - (-uniform[j].ln()).ln(), j))
            .collect();
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        let targets = (0..k).filter(|&j| unused[j]).take(nsplit);
        keyed[..nsplit]
            .iter()
            .map(|&(_, src)| src)
            .zip(targets)
            .collect()
    }

    fn split(&mut self, src: usize, dst: usize) {
        split_pair(&mut self.centroids, &mut self.usage, self.d, src, dst);
        self.n_splits += 1;
    }

    /// Finish one complete epoch and record its moving-pass objective.
    pub fn end_epoch(&mut self) {
        let rows = self.running_recipe().epoch_rows[self.epoch];
        assert_eq!(
            self.epoch_rows_done, rows,
            "e-means: epoch {} covered {} of {rows} rows",
            self.epoch, self.epoch_rows_done
        );
        let sq: f64 = self.pass_counts.iter().map(|c| c * c).sum();
        self.epoch_stats.push(EpochStats {
            objective: self.objective,
            imbalance_factor: self.k as f64 * sq / (rows as f64 * rows as f64),
            nsplit: self.n_splits - self.splits_before,
        });
        self.epoch += 1;
        if self.epoch < self.running_recipe().epoch_rows.len() {
            self.start_epoch();
        }
    }

    /// Closing means of the last epoch (Algorithm 1, line 12). Clusters that
    /// received no rows keep their EMA position.
    pub fn finalize(&mut self) -> &[f32] {
        if self.finalized {
            return &self.centroids;
        }
        let r = self.running_recipe();
        assert_eq!(
            self.epoch,
            r.epoch_rows.len(),
            "e-means: finish every epoch first"
        );
        let d = self.d;
        let sums = self.pass_sums.take().expect("e-means: closing sums");
        for j in 0..self.k {
            let n = self.pass_counts[j];
            if n > 0.0 {
                let n = n as f32;
                for (c, s) in self.centroids[j * d..(j + 1) * d]
                    .iter_mut()
                    .zip(&sums[j * d..(j + 1) * d])
                {
                    *c = s / n;
                }
            }
        }
        self.finalized = true;
        &self.centroids
    }

    /// Train on `x` (`[n, d]`) and return the centroids and, with
    /// `capture_assignments`, each row's last-epoch label.
    ///
    /// Initial centroids follow [`EmeansConfig::init`] ([`initial_centroids`]),
    /// unless `init` is given. Each epoch visits the rows in a fresh seeded
    /// permutation.
    pub fn train(&mut self, x: &[f32], init: Option<&[f32]>) -> (Vec<f32>, Option<Vec<u32>>) {
        let d = self.d;
        assert_eq!(x.len() % d, 0, "e-means: rows must be [n, d]");
        let n = x.len() / d;
        let init = match init {
            Some(c) => c.to_vec(),
            None => initial_centroids(x, d, self.k, &self.cfg),
        };
        self.begin(n, &init);
        let recipe = self.running_recipe().clone();
        let mut labels = self.cfg.capture_assignments.then(|| vec![u32::MAX; n]);
        let mut batch = Vec::with_capacity(recipe.batch_size * d);
        for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
            let order = RandomState::new(epoch_seed(self.cfg.seed, epoch)).permutation(n);
            for idx in order[..rows].chunks(recipe.batch_size) {
                batch.clear();
                for &i in idx {
                    batch.extend_from_slice(&x[i * d..(i + 1) * d]);
                }
                let mut stats = BatchStats::new(self.k, d);
                self.accumulate(&batch, &mut stats);
                if let Some(out) = labels.as_mut()
                    && !stats.labels.is_empty()
                {
                    for (&i, &l) in idx.iter().zip(&stats.labels) {
                        out[i] = l;
                    }
                }
                self.update(&stats);
            }
            self.end_epoch();
        }
        (self.finalize().to_vec(), labels)
    }

    /// Squared L2 distances and nearest-centroid labels of `x` against the
    /// current centroids.
    pub fn assign(&self, x: &[f32]) -> (Vec<f32>, Vec<u32>) {
        assign_l2(x, &self.centroids, self.d)
    }
}

/// The `k` initial centroids [`Emeans::train`] uses for `x` (`[n, d]`) under
/// `cfg.init` and `cfg.seed`; every row when `n == k`.
pub fn initial_centroids(x: &[f32], d: usize, k: usize, cfg: &EmeansConfig) -> Vec<f32> {
    let n = x.len() / d;
    if n == k {
        return x.to_vec();
    }
    assert!(n >= k, "e-means: need at least k rows");
    let mut rng = RandomState::new(cfg.seed);
    match cfg.init {
        Init::RandomRows => gather_rows(x, &rng.permutation(n)[..k], d),
        Init::KMeansPlusPlus => {
            let m = n.min(KMEANS_PP_ROWS_PER_CLUSTER * k);
            let pool: Vec<usize> = if m < n {
                rng.permutation(n)[..m].to_vec()
            } else {
                (0..n).collect()
            };
            let row = |i: usize| &x[pool[i] * d..][..d];
            let sq = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(u, v)| {
                        let e = (u - v) as f64;
                        e * e
                    })
                    .sum()
            };
            let first = (rng.random_sample() * m as f64) as usize % m;
            let mut picked = vec![first];
            let mut near: Vec<f64> = (0..m).map(|i| sq(row(i), row(first))).collect();
            while picked.len() < k {
                let total: f64 = near.iter().sum();
                let next = if total > 0.0 {
                    rng.choice_p(&near)
                } else {
                    // Every row sits on a centroid already (duplicates): take
                    // the first one not yet picked.
                    (0..m).find(|i| !picked.contains(i)).expect("m >= k")
                };
                picked.push(next);
                let c = row(next);
                for (i, v) in near.iter_mut().enumerate() {
                    *v = v.min(sq(row(i), c));
                }
            }
            let idx: Vec<usize> = picked.iter().map(|&i| pool[i]).collect();
            gather_rows(x, &idx, d)
        }
    }
}

/// Equation 2 for every cluster with `counts[j] > 0`; the others only decay
/// their mass. f32 arithmetic in the same order as the reference so both runs
/// round alike. `alpha == 1` replaces the mass instead of decaying it.
pub(crate) fn ema_step(
    centroids: &mut [f32],
    usage: &mut [f32],
    sums: &[f32],
    counts: &[f32],
    d: usize,
    alpha: f64,
) {
    let alpha_f = alpha as f32;
    let replace = alpha == 1.0;
    let decay = (1.0 - alpha) as f32;
    if !replace {
        for u in usage.iter_mut() {
            *u *= decay;
        }
    }
    for (j, &n_j) in counts.iter().enumerate() {
        if n_j <= 0.0 {
            continue;
        }
        let retained = if replace { 0.0 } else { usage[j] };
        let mass = retained + alpha_f * n_j;
        let weight = alpha_f * n_j / mass;
        let c = &mut centroids[j * d..(j + 1) * d];
        let s = &sums[j * d..(j + 1) * d];
        for (cv, &sv) in c.iter_mut().zip(s) {
            let mean = sv / n_j;
            *cv = if weight == 1.0 {
                mean
            } else {
                *cv + weight * (mean - *cv)
            };
        }
        usage[j] = mass;
    }
}

/// Split donor `src` into itself and `dst`: the two move apart by a relative
/// [`SPLIT_EPSILON`] with alternating signs and share the donor's mass.
pub(crate) fn split_pair(
    centroids: &mut [f32],
    usage: &mut [f32],
    d: usize,
    src: usize,
    dst: usize,
) {
    let base: Vec<f32> = centroids[src * d..(src + 1) * d].to_vec();
    let mut donor = vec![0f32; d];
    let mut child = vec![0f32; d];
    for t in 0..d {
        let sign = if t % 2 == 0 { 1.0f32 } else { -1.0 };
        let offset = SPLIT_EPSILON * sign;
        donor[t] = base[t] * (1.0 - offset);
        child[t] = base[t] * (1.0 + offset);
    }
    // Identical copies (e.g. a zero donor) would never separate: move one
    // coordinate of the child by one ulp.
    if donor == child {
        let up = child[0].next_up();
        child[0] = if up.is_finite() {
            up
        } else {
            child[0].next_down()
        };
    }
    let half = usage[src] / 2.0;
    usage[src] = half;
    usage[dst] = half;
    centroids[src * d..(src + 1) * d].copy_from_slice(&donor);
    centroids[dst * d..(dst + 1) * d].copy_from_slice(&child);
}

fn gather_rows(x: &[f32], idx: &[usize], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * d);
    for &i in idx {
        out.extend_from_slice(&x[i * d..(i + 1) * d]);
    }
    out
}

/// Input names of [`step_graph`].
pub mod step_io {
    /// Batch rows `[B, d]`.
    pub const X: &str = "x";
    /// Centroid positions `[k, d]`.
    pub const CENTROIDS: &str = "centroids";
    /// EMA masses `[k]`.
    pub const USAGE: &str = "usage";
    /// Scheduled step `α`, `[1]`.
    pub const ALPHA: &str = "alpha";
}

/// One e-means step as an rlx graph, for any backend.
///
/// Inputs (see [`step_io`]): `x[B, d]`, `centroids[k, d]`, `usage[k]`,
/// `alpha[1]`. Outputs, in order:
///
/// 0. `centroids'[k, d]` and 1. `usage'[k]` — Equation 2;
/// 2. `labels[B]` (f32-encoded indices);
/// 3. `stats[k, d+1]` — per-cluster sums with the count in the last column
///    (one `ScatterAdd` of `[x | 1]`);
/// 4. `objective[1]` — sum of squared distances to the input centroids.
///
/// `α` is an input so one compiled graph serves the whole cosine schedule.
/// Empty clusters keep their position because their weight is exactly 0; the
/// mass is clamped away from 0 so an underflowed `N_j` never divides 0/0.
/// Feed the outputs to [`Emeans::update_from_step`].
pub fn step_graph(b: usize, d: usize, k: usize) -> Graph {
    step_graph_with(b, d, k, DType::F32)
}

/// [`step_graph`] with the nearest-centroid search computed in `assign`
/// precision (`F64`, `F32`, `F16` or `BF16`).
///
/// Only the assignment (`x·cᵀ`, `‖c‖²`, ArgMin) runs in `assign`; the
/// statistics, EMA and objective stay in f32, as in the reference's reduced-
/// precision paths. The objective is `Σ‖x − c_label‖²` in f32, so it measures
/// the labels a precision chose rather than that precision's rounding.
pub fn step_graph_with(b: usize, d: usize, k: usize, assign: DType) -> Graph {
    let mut g = Graph::new("emeans_step");
    let x = g.input(step_io::X, Shape::new(&[b, d], DType::F32));
    let c = g.input(step_io::CENTROIDS, Shape::new(&[k, d], DType::F32));
    let usage = g.input(step_io::USAGE, Shape::new(&[k], DType::F32));
    let alpha = g.input(step_io::ALPHA, Shape::new(&[1], DType::F32));
    let s = build_step(&mut g, x, None, c, usage, alpha, assign);
    g.set_outputs(vec![s.centroids, s.usage, s.labels, s.stats, s.objective]);
    g
}

/// Nodes of one step, see [`build_step`].
pub(crate) struct StepNodes {
    pub centroids: NodeId,
    pub usage: NodeId,
    pub labels: NodeId,
    pub stats: NodeId,
    pub objective: NodeId,
}

/// One step over `x[B, d]`. `weight[B, 1]` (1 for a real row, 0 for padding)
/// drops padded rows from the statistics and objective; `None` counts every
/// row.
pub(crate) fn build_step(
    g: &mut Graph,
    x: NodeId,
    weight: Option<NodeId>,
    c: NodeId,
    usage: NodeId,
    alpha: NodeId,
    assign: DType,
) -> StepNodes {
    use rlx_ir::infer::GraphExt as _;
    assert!(
        matches!(assign, DType::F64 | DType::F32 | DType::F16 | DType::BF16),
        "e-means: assignment precision must be F64, F32, F16 or BF16, got {assign:?}"
    );
    let b = g.shape(x).dim(0).unwrap_static();
    let k = g.shape(c).dim(0).unwrap_static();

    // Assignment: proxy[B, k] = ‖c‖²(row) − 2·x·cᵀ, written as an add (see
    // `vector_quantize` for why), over x and c centred on the batch mean. The
    // argmin is the same for any centre, but far from the origin both terms
    // dwarf the gaps between nearby centroids, and their rounding (x·c on a
    // bf16 or INT8 matmul above all) flips near ties.
    let inv_b = g.constant(1.0 / b as f64, DType::F32);
    let x_sum = g.sum(x, vec![0], true);
    let centre = g.mul(x_sum, inv_b);
    let (xs, cs) = (g.sub(x, centre), g.sub(c, centre));
    let (xa, ca) = if assign == DType::F32 {
        (xs, cs)
    } else {
        (g.cast(xs, assign), g.cast(cs, assign))
    };
    let c_t = g.transpose_(ca, vec![1, 0]);
    let cross = g.mm(xa, c_t);
    let neg_two = g.constant(-2.0, assign);
    let neg_two_cross = g.mul(cross, neg_two);
    let c_sq = g.mul(ca, ca);
    let c_norm = g.sum(c_sq, vec![1], false);
    let c_norm_row = g.reshape_(c_norm, vec![1, k as i64]);
    let proxy = g.add(c_norm_row, neg_two_cross);
    let labels = g.argmin(proxy, 1, false, Shape::new(&[b], DType::F32));
    build_update(g, x, weight, c, usage, alpha, labels)
}

/// Statistics, objective and EMA of one step whose rows `x[B, d]` are already
/// labelled (`labels[B]`, f32-encoded cluster ids).
pub(crate) fn build_update(
    g: &mut Graph,
    x: NodeId,
    weight: Option<NodeId>,
    c: NodeId,
    usage: NodeId,
    alpha: NodeId,
    labels: NodeId,
) -> StepNodes {
    use rlx_ir::infer::GraphExt as _;
    let b = g.shape(x).dim(0).unwrap_static();
    let d = g.shape(x).dim(1).unwrap_static();
    let k = g.shape(c).dim(0).unwrap_static();

    // Objective: Σ w·‖x − c_label‖², in f32.
    let chosen = g.gather_(c, labels, 0);
    let diff = g.sub(x, chosen);
    let diff_sq = g.mul(diff, diff);
    let dist = g.sum(diff_sq, vec![1], true);
    let dist = match weight {
        Some(w) => g.mul(dist, w),
        None => dist,
    };
    let objective = g.sum(dist, vec![0, 1], false);
    let objective = g.reshape_(objective, vec![1]);

    // Statistics: one scatter of [w·x | w] gives sums and counts together.
    let (xw, w) = match weight {
        Some(w) => (g.mul(x, w), w),
        None => (x, g.full(&[b, 1], 1.0, DType::F32)),
    };
    let x_aug = g.concat_(vec![xw, w], 1);
    let stats = g.add_node(
        Op::ScatterAdd { axis: 0 },
        vec![x_aug, labels],
        Shape::new(&[k, d + 1], DType::F32),
    );
    let sums = g.narrow_(stats, 1, 0, d);
    let counts_col = g.narrow_(stats, 1, d, 1);
    let counts = g.reshape_(counts_col, vec![k as i64]);

    // EMA (Equation 2): N' = (1−α)·N + α·n; c' = c + (α·n/N')·(s/max(n,1) − c).
    let one = g.constant(1.0, DType::F32);
    let decay = g.sub(one, alpha);
    let decayed = g.mul(usage, decay);
    let alpha_n = g.mul(counts, alpha);
    let usage_new = g.add(decayed, alpha_n);
    // `max` with a constant rather than `clamp`: it has a double-single form,
    // so the graph also runs under number format f64 / f64x2.
    let tiny = g.constant(f32::MIN_POSITIVE as f64, DType::F32);
    let safe_mass = g.binary(BinaryOp::Max, usage_new, tiny, Shape::new(&[k], DType::F32));
    let weight_k = g.div(alpha_n, safe_mass);
    let weight_col = g.reshape_(weight_k, vec![k as i64, 1]);
    let safe_counts = g.binary(
        BinaryOp::Max,
        counts_col,
        one,
        Shape::new(&[k, 1], DType::F32),
    );
    let mean = g.div(sums, safe_counts);
    let delta = g.sub(mean, c);
    let step = g.mul(weight_col, delta);
    let c_new = g.add(c, step);
    StepNodes {
        centroids: c_new,
        usage: usage_new,
        labels,
        stats,
        objective,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_plan_matches_reference() {
        assert_eq!(epoch_rows(1000, 5.0), vec![1000; 5]);
        assert_eq!(epoch_rows(1000, 2.5), vec![1000, 500, 1000]);
        assert_eq!(epoch_rows(1000, 0.5), vec![500]);
        // Python rounds 0.5 ties to even: 0.25 * 10 = 2.5 → 2.
        assert_eq!(epoch_rows(10, 1.25), vec![2, 10]);
    }

    #[test]
    fn power_of_two_rounding() {
        assert_eq!(round_power_of_two(0.3), 1);
        assert_eq!(round_power_of_two(5.0), 4);
        assert_eq!(round_power_of_two(6.0), 8);
        assert_eq!(round_power_of_two(1024.0), 1024);
    }

    #[test]
    fn cosine_schedule_endpoints() {
        assert!((scheduled_alpha(0.5, 0, 100) - 0.5).abs() < 1e-12);
        assert!((scheduled_alpha(0.5, 100, 100) - 0.05).abs() < 1e-12);
        assert!((scheduled_alpha(0.5, 50, 100) - 0.275).abs() < 1e-12);
    }

    #[test]
    fn auto_law_follows_dimension() {
        let cfg = EmeansConfig::default();
        let low = resolve_recipe(100_000, 256, 8, &cfg);
        let high = resolve_recipe(100_000, 256, 64, &cfg);
        assert_eq!(low.batch_law, Some(BatchLaw::Quality));
        assert_eq!(high.batch_law, Some(BatchLaw::Throughput));
        assert!(low.batch_size < high.batch_size);
        assert_eq!(resolve_recipe(1000, 8, 2, &cfg).unused_threshold, 0.0);
        assert_eq!(high.unused_threshold, UNUSED_THRESHOLD);
    }

    #[test]
    fn revival_splits_a_dead_cluster() {
        // Two blobs, three centroids: one starts far away and never wins rows.
        let d = 4;
        let mut x = Vec::new();
        for i in 0..400 {
            let c = if i % 2 == 0 { 0.0 } else { 10.0 };
            for t in 0..d {
                x.push(c + ((i * 7 + t * 3) % 11) as f32 * 0.01);
            }
        }
        let mut init = vec![0.0f32; 3 * d];
        init[d..2 * d].fill(10.0);
        init[2 * d..].fill(1000.0);
        let cfg = EmeansConfig {
            batch_size: Some(32),
            ..EmeansConfig::default()
        };
        let mut em = Emeans::new(d, 3, cfg);
        let (centroids, labels) = em.train(&x, Some(&init));
        assert!(em.n_splits() > 0, "dead cluster was never revived");
        assert!(
            centroids[2 * d..].iter().all(|&v| v < 20.0),
            "revived centroid moved into the data"
        );
        let labels = labels.unwrap();
        assert!(labels.iter().all(|&l| l < 3));
    }
}
