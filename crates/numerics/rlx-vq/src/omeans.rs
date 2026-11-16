// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! o-means: e-means with a two-level nearest-centroid assignment.
//!
//! Implements o-means by Bajaj, Nupur and Garg, *"o-means: index the
//! centroids, don't scan them"* (Owlgebra AI, 2026,
//! <https://owlgebra.ai/blog/o-means/>). The reference implementation is
//! `contrib/o_means` in <https://github.com/owlgebra-ai/faiss>; this port
//! follows its training loop and schedule ([`schedule`]) on the host.
//!
//! The training loop is e-means' ([`crate::emeans`]): the batch and step laws,
//! cosine schedule, per-epoch shuffles, EMA update and split revival. o-means
//! changes three things:
//!
//! - **Two-level assignment.** The centroids are grouped into `nlist` cells of
//!   about [`CELL_SIZE`] by Lloyd iterations on the centroids themselves
//!   ([`CentroidIndex`]). A row is scored against the cell centres, then only
//!   against the centroids of its `nprobe` nearest cells.
//! - **Keeping the cells current.** During the first pass the centroids are
//!   re-filed under the existing cell centres every [`CHECK_UNUSED_EVERY`]
//!   steps; after it the cells are refit every half pass. Every split links
//!   the donor to its child, and a row whose best match is a donor also scores
//!   the donor's child and grandchild.
//! - **Recency-weighted closing mean below 2 epochs.** Each batch of the run is
//!   weighted by `exp(−r/n)`, `r` the rows processed after it. From 2 epochs
//!   up the closing mean is e-means' mean of the last pass.
//!
//! Differences from the reference, which runs on CUDA through Triton: the
//! assignment here is f32 (the reference compares FP16 copies), the batch sums
//! are f32 (the reference keeps int64 fixed point), and the RNG streams are
//! NumPy's MT19937 ([`rlx_rng::numpy`]) rather than PyTorch's, so runs match
//! the reference in distribution, not bit for bit.

#[cfg(feature = "omeans-device")]
pub mod device;

use rayon::prelude::*;
use rlx_rng::numpy::RandomState;

use crate::emeans::{
    BatchLaw, CHECK_UNUSED_EVERY, EmeansConfig, INIT_USAGE, Recipe, assign_l2, ema_step,
    epoch_rows, epoch_seed, initial_centroids, resolve_recipe, scheduled_alpha, split_pair,
};

/// Target centroids per cell.
pub const CELL_SIZE: usize = 128;
/// [`BatchLaw::Auto`] picks [`BatchLaw::Throughput`] up to this many rows and
/// [`BatchLaw::Quality`] above, as in the e-means paper's benchmarks. (The
/// e-means trainer decides by dimension instead.)
pub const THROUGHPUT_MAX_N: usize = 2_000_000;
/// Below this many epochs the closing mean weights every batch by recency.
pub const RECENCY_CLOSE_BELOW: f64 = 2.0;

/// Lloyd iterations of the first cell build, which starts from random
/// centroids.
const FIRST_BUILD_ITERS: usize = 4;
/// Refits during the first pass only re-file the centroids: enough to place
/// the children of the many splits made early in training.
const FIRST_PASS_REBUILD_ITERS: usize = 0;
/// Lloyd iterations of a refit after the first pass.
const REBUILD_ITERS: usize = 2;
/// Squared norm of an empty cell: finite but huge, so rows probe every
/// non-empty cell first.
const EMPTY_NORM: f32 = 1e37;
/// Recency-weighted closing mass below which a centroid keeps its EMA position.
const MIN_CLOSING_MASS: f64 = 1e-3;

/// Default `nlist`: `k / 128` rounded up to a multiple of 16, at least 16.
pub fn num_cells(k: usize) -> usize {
    ((k / CELL_SIZE).div_ceil(16) * 16).max(16)
}

/// Default `nprobe`: 24 when `k > 16384`, else 12 when `d ≥ 512`, else 16. The
/// reference fitted it to keep about 1% or fewer rows from missing their
/// nearest centroid.
pub fn num_probes(d: usize, k: usize) -> usize {
    if k > 16384 {
        24
    } else if d >= 512 {
        12
    } else {
        16
    }
}

/// Training settings: e-means' plus the cell layout.
#[derive(Clone, Debug, Default)]
pub struct OmeansConfig {
    /// Epoch budget, laws, seed, initialization and label capture.
    /// `batch_law: Auto` follows [`THROUGHPUT_MAX_N`] here.
    pub emeans: EmeansConfig,
    /// Number of cells; `None` means [`num_cells`] (capped at `k`).
    pub nlist: Option<usize>,
    /// Cells each row probes; `None` means [`num_probes`] (capped at `nlist`).
    pub nprobe: Option<usize>,
}

/// The resolved plan of one run.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    /// e-means' recipe. Below one epoch it keeps the one-epoch batch and step,
    /// so a shorter run is a prefix of the one-epoch schedule.
    pub recipe: Recipe,
    /// Cells over the centroids.
    pub nlist: usize,
    /// Cells each row probes.
    pub nprobe: usize,
    /// Steps between cell refits after the first pass (half a pass, at least
    /// [`CHECK_UNUSED_EVERY`]).
    pub rebuild_every: usize,
    /// Whether the closing mean is recency-weighted (`nepochs < 2`).
    pub recency_close: bool,
}

impl Schedule {
    /// Weight in the closing mean of a batch of epoch `epoch` after which
    /// `done` rows of the run have been processed; `None` when the batch is
    /// not part of the closing mean.
    pub fn closing_weight(&self, n: usize, epoch: usize, done: usize) -> Option<f64> {
        if self.recency_close {
            let total: usize = self.recipe.epoch_rows.iter().sum();
            Some((-((total - done) as f64) / n as f64).exp())
        } else {
            (epoch + 1 == self.recipe.epoch_rows.len()).then_some(1.0)
        }
    }

    /// Whether step `t` ends with a revival check.
    pub fn checks_after(&self, t: usize) -> bool {
        self.recipe.unused_threshold > 0.0 && (t + 1).is_multiple_of(CHECK_UNUSED_EVERY)
    }
}

/// Resolve the plan for `n` rows of dimension `d` into `k` clusters.
pub fn schedule(n: usize, d: usize, k: usize, cfg: &OmeansConfig) -> Schedule {
    let nepochs = cfg.emeans.nepochs;
    assert!(nepochs > 0.0, "o-means: nepochs must be positive");
    assert!(
        (1..=n).contains(&k),
        "o-means: need 1 <= k <= n, got k={k} for n={n}"
    );
    let mut law = cfg.emeans.clone();
    law.nepochs = nepochs.max(1.0);
    if law.batch_law == BatchLaw::Auto {
        law.batch_law = if n <= THROUGHPUT_MAX_N {
            BatchLaw::Throughput
        } else {
            BatchLaw::Quality
        };
    }
    let mut recipe = resolve_recipe(n, k, d, &law);
    recipe.epoch_rows = epoch_rows(n, nepochs);
    recipe.epoch_steps = recipe
        .epoch_rows
        .iter()
        .map(|r| r.div_ceil(recipe.batch_size))
        .collect();
    recipe.total_steps = recipe.epoch_steps.iter().sum();
    let nlist = cfg.nlist.unwrap_or(num_cells(k).min(k));
    assert!(
        (1..=k).contains(&nlist),
        "o-means: nlist must be between 1 and k"
    );
    let nprobe = cfg.nprobe.unwrap_or(num_probes(d, k).min(nlist));
    assert!(
        (1..=nlist).contains(&nprobe),
        "o-means: nprobe must be between 1 and nlist"
    );
    let steps_per_pass = n.div_ceil(recipe.batch_size);
    Schedule {
        recipe,
        nlist,
        nprobe,
        rebuild_every: CHECK_UNUSED_EVERY.max(steps_per_pass.div_ceil(2)),
        recency_close: nepochs < RECENCY_CLOSE_BELOW,
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

fn row_norms(rows: &[f32], d: usize) -> Vec<f32> {
    rows.chunks_exact(d).map(|r| dot(r, r)).collect()
}

/// Nearest of `centres` (`[g, d]`) for every row of `rows`, ties to the lower
/// centre.
fn nearest_centres(centres: &[f32], rows: &[f32], d: usize) -> Vec<usize> {
    let norms = row_norms(centres, d);
    rows.par_chunks_exact(d)
        .map(|c| {
            let mut best = (f32::INFINITY, 0usize);
            for (g, m) in centres.chunks_exact(d).enumerate() {
                let s = norms[g] - 2.0 * dot(c, m);
                if s < best.0 {
                    best = (s, g);
                }
            }
            best.1
        })
        .collect()
}

/// An IVF index over a set of centroids: cells built by k-means on the
/// centroids, each centroid filed under its nearest cell centre, plus links
/// from split donors to their children.
#[derive(Clone, Debug)]
pub struct CentroidIndex {
    d: usize,
    /// `[nlist, d]` cell centres.
    centres: Vec<f32>,
    /// `‖centre‖²`, or [`EMPTY_NORM`] for a cell without centroids.
    centre_norms: Vec<f32>,
    /// Centroid ids grouped by cell, ascending within a cell.
    members: Vec<u32>,
    /// Cell `c` holds `members[offsets[c]..offsets[c + 1]]`.
    offsets: Vec<usize>,
    /// Latest split child of each centroid, `-1` if none since the last fit.
    child: Vec<i32>,
}

impl CentroidIndex {
    /// Group `centroids` (`[k, d]`) into `nlist` cells: start from `nlist`
    /// centroids drawn with `seed`, run `iters` Lloyd iterations, then file
    /// every centroid under its nearest centre.
    pub fn build(centroids: &[f32], d: usize, nlist: usize, iters: usize, seed: u32) -> Self {
        Self::build_with(centroids, d, nlist, iters, seed, &mut |centres, rows| {
            nearest_centres(centres, rows, d)
        })
    }

    /// [`build`](Self::build) with a pluggable nearest-centre search (see
    /// [`refit_with`](Self::refit_with)).
    pub(crate) fn build_with(
        centroids: &[f32],
        d: usize,
        nlist: usize,
        iters: usize,
        seed: u32,
        nearest: &mut dyn FnMut(&[f32], &[f32]) -> Vec<usize>,
    ) -> Self {
        let k = centroids.len() / d;
        assert!(
            (1..=k).contains(&nlist),
            "o-means: nlist must be between 1 and k"
        );
        let pick = RandomState::new(seed).permutation(k);
        let mut centres = Vec::with_capacity(nlist * d);
        for &j in &pick[..nlist] {
            centres.extend_from_slice(&centroids[j * d..(j + 1) * d]);
        }
        let mut index = Self {
            d,
            centres,
            centre_norms: Vec::new(),
            members: Vec::new(),
            offsets: Vec::new(),
            child: Vec::new(),
        };
        index.refit_with(centroids, iters, nearest);
        index
    }

    /// Run `iters` Lloyd iterations from the current centres (a cell that
    /// receives no centroid keeps its centre), then re-file every centroid.
    /// Clears the split links.
    pub fn refit(&mut self, centroids: &[f32], iters: usize) {
        let d = self.d;
        self.refit_with(centroids, iters, &mut |centres, rows| {
            nearest_centres(centres, rows, d)
        });
    }

    /// [`refit`](Self::refit) with the nearest-centre search done by
    /// `nearest(centres, centroids)`, e.g. on a device.
    pub(crate) fn refit_with(
        &mut self,
        centroids: &[f32],
        iters: usize,
        nearest: &mut dyn FnMut(&[f32], &[f32]) -> Vec<usize>,
    ) {
        let (d, g) = (self.d, self.nlist());
        let k = centroids.len() / d;
        for _ in 0..iters {
            let cell = nearest(&self.centres, centroids);
            let mut sums = vec![0f32; g * d];
            let mut counts = vec![0f32; g];
            for (j, &c) in cell.iter().enumerate() {
                for (s, v) in sums[c * d..(c + 1) * d]
                    .iter_mut()
                    .zip(&centroids[j * d..(j + 1) * d])
                {
                    *s += v;
                }
                counts[c] += 1.0;
            }
            for c in 0..g {
                if counts[c] > 0.0 {
                    for (m, s) in self.centres[c * d..(c + 1) * d]
                        .iter_mut()
                        .zip(&sums[c * d..(c + 1) * d])
                    {
                        *m = s / counts[c];
                    }
                }
            }
        }
        let cell = nearest(&self.centres, centroids);
        let mut offsets = vec![0usize; g + 1];
        for &c in &cell {
            offsets[c + 1] += 1;
        }
        for c in 0..g {
            offsets[c + 1] += offsets[c];
        }
        let mut next = offsets.clone();
        let mut members = vec![0u32; k];
        for (j, &c) in cell.iter().enumerate() {
            members[next[c]] = j as u32;
            next[c] += 1;
        }
        self.centre_norms = (0..g)
            .map(|c| {
                if offsets[c + 1] > offsets[c] {
                    let m = &self.centres[c * d..(c + 1) * d];
                    dot(m, m)
                } else {
                    EMPTY_NORM
                }
            })
            .collect();
        self.members = members;
        self.offsets = offsets;
        self.child = vec![-1; k];
    }

    /// Number of cells.
    pub fn nlist(&self) -> usize {
        self.centres.len() / self.d
    }

    /// Centroids filed under each cell.
    pub fn cell_sizes(&self) -> Vec<usize> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Record that `donor` was split into itself and `child`. The child takes
    /// over the donor's previous link, so a row whose best match is the donor
    /// reaches both its newest child and the one before.
    pub fn record_split(&mut self, donor: usize, child: usize) {
        self.child[child] = self.child[donor];
        self.child[donor] = child as i32;
    }

    /// Nearest centroid of `x` among its `nprobe` nearest cells and the split
    /// links of the winner: `(‖c‖² − 2x·c, label, distance evaluations)`.
    fn search_row(
        &self,
        x: &[f32],
        centroids: &[f32],
        norms: &[f32],
        nprobe: usize,
        cells: &mut Vec<(f32, u32)>,
    ) -> (f32, u32, usize) {
        let d = self.d;
        let g = self.nlist();
        cells.clear();
        cells.extend(
            self.centres
                .chunks_exact(d)
                .enumerate()
                .map(|(c, m)| (self.centre_norms[c] - 2.0 * dot(x, m), c as u32)),
        );
        let by = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        if nprobe < g {
            cells.select_nth_unstable_by(nprobe - 1, by);
            cells.truncate(nprobe);
        }
        // Probe rank order: across cells, the lower rank wins a tie.
        cells.sort_unstable_by(by);
        let score = |j: u32| norms[j as usize] - 2.0 * dot(x, &centroids[j as usize * d..][..d]);
        let mut evals = g;
        let (mut best, mut label) = (f32::INFINITY, u32::MAX);
        for &(_, c) in cells.iter() {
            let ms = &self.members[self.offsets[c as usize]..self.offsets[c as usize + 1]];
            evals += ms.len();
            for &j in ms {
                let s = score(j);
                if s < best {
                    (best, label) = (s, j);
                }
            }
        }
        assert!(label != u32::MAX, "o-means: every probed cell is empty");
        // Children split off since the last fit may sit in an unprobed cell.
        let link = |j: u32| match self.child[j as usize] {
            c if c < 0 => j,
            c => c as u32,
        };
        let c1 = link(label);
        let c2 = link(c1);
        for (from, to) in [(label, c1), (c1, c2)] {
            if to != from {
                evals += 1;
                let s = score(to);
                if s < best {
                    (best, label) = (s, to);
                }
            }
        }
        (best, label, evals)
    }

    /// Approximate nearest centroids of `x` (`[m, d]`) among `centroids`
    /// (`[k, d]`, the set this index was fit to): returns squared distances,
    /// labels and the number of distance evaluations.
    pub fn search(&self, x: &[f32], centroids: &[f32], nprobe: usize) -> (Vec<f32>, Vec<u32>, u64) {
        let d = self.d;
        assert!(
            (1..=self.nlist()).contains(&nprobe),
            "o-means: nprobe must be between 1 and nlist"
        );
        let norms = row_norms(centroids, d);
        let out: Vec<(f32, u32, usize)> = x
            .par_chunks_exact(d)
            .map_init(Vec::new, |cells, xi| {
                let (s, l, e) = self.search_row(xi, centroids, &norms, nprobe, cells);
                ((dot(xi, xi) + s).max(0.0), l, e)
            })
            .collect();
        let evals = out.iter().map(|o| o.2 as u64).sum();
        let (dist, labels) = out.into_iter().map(|(s, l, _)| (s, l)).unzip();
        (dist, labels, evals)
    }
}

/// Counters of one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OmeansReport {
    /// Clusters revived.
    pub n_splits: usize,
    /// Cell builds and refits.
    pub rebuilds: usize,
    /// Distance evaluations of training: rows against cell centres and
    /// probed centroids, plus centroids against cell centres in the fits.
    pub distance_evals: u64,
    /// What exact assignment of the same rows would have cost (`rows · k`).
    pub exact_distance_evals: u64,
    /// Sum of squared distances per epoch, measured while the centroids moved.
    pub epoch_objectives: Vec<f64>,
}

/// Host o-means trainer (see the [module docs](self)). One object is one run.
pub struct Omeans {
    d: usize,
    k: usize,
    cfg: OmeansConfig,
    schedule: Option<Schedule>,
    centroids: Vec<f32>,
    usage: Vec<f32>,
    index: Option<CentroidIndex>,
    report: OmeansReport,
}

impl Omeans {
    /// A trainer for `k` clusters of dimension `d`.
    pub fn new(d: usize, k: usize, cfg: OmeansConfig) -> Self {
        assert!(d > 0 && k > 0, "o-means: d and k must be positive");
        Self {
            d,
            k,
            cfg,
            schedule: None,
            centroids: Vec::new(),
            usage: Vec::new(),
            index: None,
            report: OmeansReport::default(),
        }
    }

    /// The resolved plan (after [`train`](Self::train)).
    pub fn schedule(&self) -> Option<&Schedule> {
        self.schedule.as_ref()
    }
    /// Centroids `[k, d]` (closing means after training).
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    /// EMA masses `N_j`.
    pub fn usage(&self) -> &[f32] {
        &self.usage
    }
    /// Counters of the run.
    pub fn report(&self) -> &OmeansReport {
        &self.report
    }
    /// The cell index, refit to the final centroids after training.
    pub fn index(&self) -> Option<&CentroidIndex> {
        self.index.as_ref()
    }

    /// Train on `x` (`[n, d]`) and return the centroids and, with
    /// `capture_assignments`, each row's label from the last pass
    /// (`u32::MAX` for rows a fractional run did not visit). Like e-means'
    /// labels they come from moving centroids, and here from the approximate
    /// search; call [`assign`](Self::assign) for exact labels.
    ///
    /// Initial centroids follow `cfg.emeans.init` ([`initial_centroids`]),
    /// unless `init` is given.
    pub fn train(&mut self, x: &[f32], init: Option<&[f32]>) -> (Vec<f32>, Option<Vec<u32>>) {
        let (d, k) = (self.d, self.k);
        assert!(self.schedule.is_none(), "o-means: one run per object");
        assert_eq!(x.len() % d, 0, "o-means: rows must be [n, d]");
        let n = x.len() / d;
        let sched = schedule(n, d, k, &self.cfg);
        self.centroids = match init {
            Some(c) => {
                assert_eq!(c.len(), k * d, "o-means: initial centroids must be [k, d]");
                c.to_vec()
            }
            None => initial_centroids(x, d, k, &self.cfg.emeans),
        };
        self.usage = vec![INIT_USAGE; k];
        let seed = self.cfg.emeans.seed;
        let mut split_rng = RandomState::new(seed);
        let cell_seed = (seed as u64 * 7919 + 17) as u32;
        let recipe = &sched.recipe;
        let last = recipe.epoch_rows.len() - 1;
        let mut close_sums = vec![0f64; k * d];
        let mut close_counts = vec![0f64; k];
        // Last-pass closing in f32, as e-means accumulates it.
        let mut pass_sums = vec![0f32; k * d];
        let mut pass_counts = vec![0f64; k];
        let mut labels = self
            .cfg
            .emeans
            .capture_assignments
            .then(|| vec![u32::MAX; n]);
        let (mut done, mut since_fit, mut t) = (0usize, 0usize, 0usize);
        let mut batch_sums = vec![0f32; k * d];
        let mut batch_counts = vec![0f32; k];

        for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
            let order = RandomState::new(epoch_seed(seed, epoch)).permutation(n);
            let mut objective = 0.0f64;
            for idx in order[..rows].chunks(recipe.batch_size) {
                let first_pass = done < n;
                let every = if first_pass {
                    CHECK_UNUSED_EVERY
                } else {
                    sched.rebuild_every
                };
                match self.index.as_mut() {
                    None => {
                        self.index = Some(CentroidIndex::build(
                            &self.centroids,
                            d,
                            sched.nlist,
                            FIRST_BUILD_ITERS,
                            cell_seed,
                        ));
                        self.report.distance_evals +=
                            ((FIRST_BUILD_ITERS + 1) * k * sched.nlist) as u64;
                        self.report.rebuilds += 1;
                        since_fit = 0;
                    }
                    Some(index) if since_fit >= every => {
                        let iters = if first_pass {
                            FIRST_PASS_REBUILD_ITERS
                        } else {
                            REBUILD_ITERS
                        };
                        index.refit(&self.centroids, iters);
                        self.report.distance_evals += ((iters + 1) * k * sched.nlist) as u64;
                        self.report.rebuilds += 1;
                        since_fit = 0;
                    }
                    Some(_) => {}
                }
                since_fit += 1;

                let index = self.index.as_ref().expect("built above");
                let norms = row_norms(&self.centroids, d);
                let found: Vec<(f32, u32, usize)> = idx
                    .par_iter()
                    .map_init(Vec::new, |cells, &i| {
                        let xi = &x[i * d..(i + 1) * d];
                        let (s, l, e) =
                            index.search_row(xi, &self.centroids, &norms, sched.nprobe, cells);
                        ((dot(xi, xi) + s).max(0.0), l, e)
                    })
                    .collect();
                batch_sums.fill(0.0);
                batch_counts.fill(0.0);
                for (&i, &(dist, l, e)) in idx.iter().zip(&found) {
                    let l = l as usize;
                    for (s, v) in batch_sums[l * d..(l + 1) * d]
                        .iter_mut()
                        .zip(&x[i * d..(i + 1) * d])
                    {
                        *s += v;
                    }
                    batch_counts[l] += 1.0;
                    objective += dist as f64;
                    self.report.distance_evals += e as u64;
                    if epoch == last
                        && let Some(out) = labels.as_mut()
                    {
                        out[i] = l as u32;
                    }
                }
                self.report.exact_distance_evals += (idx.len() * k) as u64;
                done += idx.len();

                match sched.closing_weight(n, epoch, done) {
                    Some(w) if sched.recency_close => {
                        for (j, &c) in batch_counts.iter().enumerate() {
                            if c > 0.0 {
                                close_counts[j] += w * c as f64;
                                for (a, s) in close_sums[j * d..(j + 1) * d]
                                    .iter_mut()
                                    .zip(&batch_sums[j * d..(j + 1) * d])
                                {
                                    *a += w * *s as f64;
                                }
                            }
                        }
                    }
                    Some(_) => {
                        for (p, s) in pass_sums.iter_mut().zip(&batch_sums) {
                            *p += s;
                        }
                        for (p, &c) in pass_counts.iter_mut().zip(&batch_counts) {
                            *p += c as f64;
                        }
                    }
                    None => {}
                }
                let alpha = scheduled_alpha(recipe.alpha0, t, recipe.total_steps);
                ema_step(
                    &mut self.centroids,
                    &mut self.usage,
                    &batch_sums,
                    &batch_counts,
                    d,
                    alpha,
                );
                if sched.checks_after(t) {
                    let recency = sched.recency_close;
                    for (src, dst) in
                        plan_splits(&self.usage, &mut split_rng, recipe.unused_threshold)
                    {
                        split_pair(&mut self.centroids, &mut self.usage, d, src, dst);
                        if recency {
                            // Donor and child share the donor's closing mass.
                            close_counts[src] *= 0.5;
                            close_counts[dst] = close_counts[src];
                            for c in 0..d {
                                close_sums[src * d + c] *= 0.5;
                                close_sums[dst * d + c] = close_sums[src * d + c];
                            }
                        }
                        self.index
                            .as_mut()
                            .expect("built above")
                            .record_split(src, dst);
                        self.report.n_splits += 1;
                    }
                }
                t += 1;
            }
            self.report.epoch_objectives.push(objective);
        }

        for j in 0..k {
            let c = &mut self.centroids[j * d..(j + 1) * d];
            if sched.recency_close {
                let w = close_counts[j];
                if w > MIN_CLOSING_MASS {
                    for (v, s) in c.iter_mut().zip(&close_sums[j * d..(j + 1) * d]) {
                        *v = (s / w) as f32;
                    }
                }
            } else if pass_counts[j] > 0.0 {
                let w = pass_counts[j] as f32;
                for (v, s) in c.iter_mut().zip(&pass_sums[j * d..(j + 1) * d]) {
                    *v = s / w;
                }
            }
        }
        if let Some(index) = self.index.as_mut() {
            index.refit(&self.centroids, REBUILD_ITERS);
        }
        self.schedule = Some(sched);
        (self.centroids.clone(), labels)
    }

    /// Exact squared distances and labels of `x` against the centroids.
    pub fn assign(&self, x: &[f32]) -> (Vec<f32>, Vec<u32>) {
        assign_l2(x, &self.centroids, self.d)
    }

    /// Two-level search of `x` with the index refit to the final centroids:
    /// squared distances, labels and distance evaluations.
    pub fn assign_approx(&self, x: &[f32]) -> (Vec<f32>, Vec<u32>, u64) {
        let index = self.index.as_ref().expect("o-means: train first");
        let nprobe = self.schedule.as_ref().expect("o-means: train first").nprobe;
        index.search(x, &self.centroids, nprobe)
    }
}

/// `(donor, dead)` pairs for one revival check: centroids below
/// `τ · mean(N)` are dead, donors are drawn among the others in proportion
/// to their mass (Gumbel top-m) and the `m` lowest dead ids become their
/// children. Draws `k` uniforms at every check, as the reference does, so
/// the stream does not depend on `m`.
fn plan_splits(usage: &[f32], rng: &mut RandomState, tau: f32) -> Vec<(usize, usize)> {
    let k = usage.len();
    let threshold = usage.iter().sum::<f32>() * (tau / k as f32);
    let dead: Vec<bool> = usage.iter().map(|&u| u < threshold).collect();
    let uniform: Vec<f64> = (0..k)
        .map(|_| rng.random_double().max(f64::MIN_POSITIVE))
        .collect();
    let mut keyed: Vec<(f64, usize)> = (0..k)
        .filter(|&j| !dead[j] && usage[j] > 0.0)
        .map(|j| (usage[j].ln() as f64 - (-uniform[j].ln()).ln(), j))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    let targets: Vec<usize> = (0..k).filter(|&j| dead[j]).collect();
    keyed.iter().map(|&(_, src)| src).zip(targets).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(nepochs: f64) -> OmeansConfig {
        OmeansConfig {
            emeans: EmeansConfig {
                nepochs,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // The cases of the reference's tests/test_o_means.py.
    #[test]
    fn cells_and_probes() {
        assert_eq!(num_cells(1024), 16);
        assert_eq!(num_cells(16384), 128);
        assert_eq!(num_cells(20000), 160);
        assert_eq!(num_cells(65536), 512);
        assert_eq!(num_probes(96, 16384), 16);
        assert_eq!(num_probes(768, 16384), 12);
        assert_eq!(num_probes(96, 65536), 24);
        assert_eq!(num_probes(768, 65536), 24);
        let s = schedule(1_000_000, 96, 16384, &cfg(1.0));
        assert_eq!((s.nlist, s.nprobe), (128, 16));
        let s = schedule(1000, 8, 12, &cfg(1.0));
        assert_eq!((s.nlist, s.nprobe), (12, 12));
        let mut c = cfg(1.0);
        (c.nlist, c.nprobe) = (Some(64), Some(4));
        let s = schedule(1_000_000, 96, 16384, &c);
        assert_eq!((s.nlist, s.nprobe), (64, 4));
    }

    #[test]
    fn recipe_is_e_means() {
        for (n, k, law) in [
            (1_000_000, 16384, BatchLaw::Throughput),
            (3_000_000, 65536, BatchLaw::Quality),
        ] {
            for t in [1.0, 1.6, 3.0] {
                let s = schedule(n, 96, k, &cfg(t));
                let e = resolve_recipe(
                    n,
                    k,
                    96,
                    &EmeansConfig {
                        nepochs: t,
                        batch_law: law,
                        ..Default::default()
                    },
                );
                assert_eq!(s.recipe, e);
                for t in (0..s.recipe.total_steps).step_by(97) {
                    assert_eq!(s.checks_after(t), (t + 1) % CHECK_UNUSED_EVERY == 0);
                }
            }
        }
    }

    #[test]
    fn below_one_epoch_keeps_one_epoch_recipe() {
        let s = schedule(1_000_000, 96, 16384, &cfg(0.8));
        let one = schedule(1_000_000, 96, 16384, &cfg(1.0));
        assert_eq!(
            (s.recipe.batch_size, s.recipe.alpha0),
            (one.recipe.batch_size, one.recipe.alpha0)
        );
        assert_eq!(s.recipe.epoch_rows, vec![800_000]);
    }

    #[test]
    fn revival_off_in_two_dimensions() {
        let s = schedule(100_000, 2, 64, &cfg(3.0));
        assert_eq!(s.recipe.unused_threshold, 0.0);
        assert!((0..s.recipe.total_steps).all(|t| !s.checks_after(t)));
    }

    #[test]
    fn closing_weights() {
        let n = 1_000_000;
        let s = schedule(n, 96, 16384, &cfg(1.6));
        assert!(s.recency_close);
        let total: usize = s.recipe.epoch_rows.iter().sum();
        let mut done = 0;
        for (epoch, &rows) in s.recipe.epoch_rows.iter().enumerate() {
            for b0 in (0..rows).step_by(s.recipe.batch_size) {
                done += s.recipe.batch_size.min(rows - b0);
                let w = s.closing_weight(n, epoch, done).unwrap();
                assert!((w - (-((total - done) as f64) / n as f64).exp()).abs() < 1e-12);
            }
        }
        assert_eq!(s.closing_weight(n, 1, total), Some(1.0));
        let s = schedule(n, 96, 16384, &cfg(2.0));
        assert!(!s.recency_close);
        assert_eq!(s.closing_weight(n, 0, 1), None);
        assert_eq!(s.closing_weight(n, 1, 1), Some(1.0));
    }

    #[test]
    fn rebuild_cadence() {
        let s = schedule(1_000_000, 96, 16384, &cfg(3.0));
        let steps_per_pass = 1_000_000usize.div_ceil(s.recipe.batch_size);
        assert_eq!(
            s.rebuild_every,
            CHECK_UNUSED_EVERY.max(steps_per_pass.div_ceil(2))
        );
    }

    #[test]
    fn split_links_reach_child_and_grandchild() {
        // Two far pairs of centroids, one cell each.
        let d = 2;
        let c = [0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0f32];
        let mut index = CentroidIndex::build(&c, d, 2, 2, 7);
        assert_eq!(index.cell_sizes(), vec![2, 2]);
        // Centroid 1 was split into 2, then into 3: 1 → 3 → 2.
        index.record_split(1, 2);
        index.record_split(1, 3);
        assert_eq!(index.child[..4], [-1, 3, -1, 2]);
        // Hide the far cell so a row there probes only the origin cell, whose
        // best match is the donor 1; the links still reach 3 and 2.
        let far = (index.centres[0] < 5.0) as usize;
        index.centre_norms[far] = EMPTY_NORM;
        let norms = row_norms(&c, d);
        let mut cells = Vec::new();
        for (x, want) in [([10.09f32, 10.0], 3), ([9.9, 10.0], 2)] {
            let (_, l, evals) = index.search_row(&x, &c, &norms, 1, &mut cells);
            assert_eq!((l, evals), (want, 2 + 2 + 2));
        }
    }
}
