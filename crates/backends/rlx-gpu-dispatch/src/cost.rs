// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A **tile-level** cost model: rank candidate schedules without running them.
//!
//! [`crate::dispatch`] answers *which* schedule to install and `tune_dispatch`
//! answers it by measuring every candidate on the device. That is authoritative
//! and it does not scale: the candidate list is a cross product, and each entry
//! costs a JIT compile plus a timed sweep on a GPU that has to be idle for the
//! number to mean anything. CAKE's §4 answer is a cheap analytical filter ahead
//! of the expensive one — spend device time only on candidates that could
//! plausibly win.
//!
//! # What this model is for
//!
//! **Ranking, inside one workload.** [`TileCost::score`] has no absolute
//! meaning; it is not a predicted millisecond and must never be reported as
//! one. Two scores are comparable only when they came from the same
//! [`Workload`]. The measured evidence for that scoping is in
//! `tests/cost_model_ranking.rs`: held out one shape at a time, the model's
//! *ordering* of six real tiles reproduces the measured ordering at Spearman
//! ρ ≥ 0.94, while its *absolute* predictions are off by up to 122% at the
//! smallest shape. Ranking generalizes here; magnitude does not.
//!
//! So the model is wired in as a [`TileCostModel::prefilter`], never as a
//! decision. It narrows the set; measurement still picks the winner and still
//! has to clear the holdout gate before anything is installed. A model that
//! could install a tile on its own would be an unmeasured performance claim,
//! which is the thing [`CostProvenance`] exists to prevent.
//!
//! # Terms
//!
//! Two, because two are what the measured data can identify:
//!
//! * **padded MACs** — `ceil(m/bm)·bm · ceil(n/bn)·bn · k`. Counts the
//!   multiply-accumulates the kernel *performs*, including those on rows and
//!   columns that fall outside C. At `m = 1` a `bm = 128` tile does 128× the
//!   necessary row work, which is the single largest effect in the decode
//!   regime and the reason the dispatch table separates it.
//! * **sync events** — `blocks · ceil(k/bk)`. Each K-step stages a tile into
//!   shared memory behind two block-wide barriers, so halving `bk` doubles
//!   this. It is what distinguishes `32x32x32` from `32x32x16`, which have
//!   identical padded MACs and differ by 1.67× in measured time.
//!
//! Only their *ratio* has meaning, so the model is one constant:
//! [`SYNC_EVENT_MAC_EQUIVALENT`]. Scores are denominated in padded-MAC
//! equivalents, which is deliberate — a score is ~1e9 and cannot be mistaken
//! for a millisecond.
//!
//! # Two terms that are deliberately absent
//!
//! **Global-memory traffic** is computed and reported
//! ([`TileCost::global_elems`]) but not fitted: across the measured candidates
//! it is collinear with padded MACs, so the data cannot separate them.
//! Attributing time to it would be an invented coefficient.
//!
//! **Fixed per-dispatch cost** was fitted at first and then removed, which is
//! worth recording because the removal was a bug fix. Least squares put
//! 0.106 ms in it — larger than the fastest time ever measured (0.055 ms), so
//! not a per-dispatch cost at all; constraining it merely pinned it to the
//! boundary, the signature of a parameter the data cannot identify. It also
//! could not help: a constant is order-preserving within a workload, so it
//! contributed nothing to ranking while generating a "launch bound"
//! attribution that contradicted the measurements (it called the
//! `1x1024x1024` bucket unimprovable, where tuning in fact found 1.97×).
//! Dropping it left the worst held-out ρ unchanged at 0.943 and *improved* the
//! smallest shape from 0.943 to 1.000. `constant_offsets_cannot_change_ranking`
//! pins the argument.

use crate::dispatch::Workload;
use crate::tiles::TileParams;

/// Where a cost model's coefficients came from.
///
/// The same discipline as `rlx_runtime`'s `CostCalibration`: a model that was
/// never checked against hardware is still usable for ordering candidates, but
/// it must say so, and nothing downstream may quote it as a time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostProvenance {
    /// Fitted by least squares against timings taken on a real device.
    ///
    /// `holdout_rho` is the *worst* leave-one-shape-out Spearman correlation
    /// between predicted and measured ordering — the number that says whether
    /// the ranking transfers to a shape the fit never saw.
    ///
    /// `domain` is the shape box the timings actually covered. It matters
    /// because held-out validation only licenses *interpolation*: the sm_86
    /// fit spans `m ∈ [1, 32]` and yet the model changes its recommendation
    /// at `m ≥ 128`, in a regime no calibration point touches. Recording the
    /// box lets [`TileCost::extrapolated`] mark that answer as the guess it is.
    Measured {
        device: &'static str,
        samples: u32,
        holdout_rho: f32,
        domain: CalibrationDomain,
    },
    /// Derived from the term structure alone, with no device behind it.
    ///
    /// Legitimate for ordering candidates on an arch nobody has calibrated;
    /// not legitimate as evidence for any performance claim.
    Structural,
}

/// The inclusive shape box a set of calibration timings covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationDomain {
    pub m: (usize, usize),
    pub k: (usize, usize),
    pub n: (usize, usize),
}

impl CalibrationDomain {
    /// Whether `workload` falls inside the measured box.
    ///
    /// Axis-aligned and deliberately crude: it answers "was anything like this
    /// ever measured", not "is the model accurate here". A `false` is a real
    /// warning; a `true` is not a guarantee.
    pub const fn contains(&self, workload: &Workload) -> bool {
        let Workload::Matmul { m, k, n } = *workload else {
            return false;
        };
        m >= self.m.0
            && m <= self.m.1
            && k >= self.k.0
            && k <= self.k.1
            && n >= self.n.0
            && n <= self.n.1
    }
}

impl std::fmt::Display for CalibrationDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "m {}..={}, k {}..={}, n {}..={}",
            self.m.0, self.m.1, self.k.0, self.k.1, self.n.0, self.n.1
        )
    }
}

impl CostProvenance {
    /// Whether this model's ordering has been checked against measurements it
    /// was not fitted on.
    pub const fn is_measured(&self) -> bool {
        matches!(self, Self::Measured { .. })
    }

    /// Whether `workload` sits inside the calibrated shape box.
    ///
    /// Always `false` for [`CostProvenance::Structural`] — a model with no
    /// device behind it is extrapolating everywhere, and saying so is the
    /// point.
    pub const fn covers(&self, workload: &Workload) -> bool {
        match self {
            Self::Measured { domain, .. } => domain.contains(workload),
            Self::Structural => false,
        }
    }
}

/// Which modeled term dominates a candidate's predicted cost.
///
/// Each variant names a term the model actually has, so each is a repair
/// target: padding waste says *shrink the block tile*, K-loop overhead says
/// *deepen `bk`*, launch-bound says *tile choice cannot help this shape*.
/// Every variant maps to a term the model actually fits. There is deliberately
/// no "launch bound" — see the module docs for why that attribution was
/// removed rather than kept as a guess.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bottleneck {
    /// Most modeled cost goes to MACs outside the real output — the block
    /// tile is too large for this shape. Repair: shrink `bm`/`bn`.
    PaddingWaste { useful_fraction: f64 },
    /// Modeled cost is dominated by K-loop barriers: `bk` is too shallow for
    /// this `k`. Repair: deepen `bk`.
    KLoopOverhead { iters: u32 },
    /// The compute term dominates and the tile covers the output well. The
    /// model has nothing further to say — this is where measurement earns its
    /// keep.
    ComputeBound,
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PaddingWaste { useful_fraction } => {
                write!(
                    f,
                    "padding waste ({:.1}% of MACs useful)",
                    useful_fraction * 100.0
                )
            }
            Self::KLoopOverhead { iters } => write!(f, "k-loop overhead ({iters} iters)"),
            Self::ComputeBound => write!(f, "compute bound"),
        }
    }
}

/// A modeled cost breakdown for one tile on one workload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileCost {
    /// MACs the output actually requires.
    pub useful_macs: u64,
    /// MACs the kernel performs, including padding.
    pub padded_macs: u64,
    /// Thread blocks launched.
    pub blocks: u64,
    /// K-loop iterations per block.
    pub k_iters: u32,
    /// `blocks · k_iters` — staging barriers across the whole grid.
    pub sync_events: u64,
    /// Global-memory element loads with no L2 reuse assumed. **Reported, not
    /// fitted** — see the module docs.
    pub global_elems: u64,
    /// Relative cost in **padded-MAC equivalents**. Comparable *only* against
    /// scores from the same [`Workload`], and not a time in any unit.
    pub score: f64,
    /// Which term dominates `score`.
    pub bottleneck: Bottleneck,
    /// The workload falls outside the box the model was calibrated on, so this
    /// estimate is an extrapolation. Held-out validation licenses
    /// interpolation only; it says nothing here.
    pub extrapolated: bool,
}

impl TileCost {
    /// Fraction of performed MACs that land inside C. `1.0` means the tile
    /// divides the shape exactly.
    pub fn useful_fraction(&self) -> f64 {
        if self.padded_macs == 0 {
            return 1.0;
        }
        self.useful_macs as f64 / self.padded_macs as f64
    }

    /// MACs per global element loaded. Reported for diagnosis only.
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.global_elems == 0 {
            return 0.0;
        }
        self.padded_macs as f64 / self.global_elems as f64
    }
}

/// Linear cost model over the two identifiable terms.
#[derive(Debug, Clone, Copy)]
pub struct TileCostModel {
    /// What one staging barrier costs, expressed in padded MACs.
    sync_event_macs: f64,
    provenance: CostProvenance,
}

/// **The entire model.** One staging barrier costs about as much as this many
/// padded MACs.
///
/// Fitted on an RTX 3080 Ti (sm_86) over 18 timings — 6 tiles × 3 shapes
/// spanning decode and small-batch, median of 30 L2-flushed samples each, GPU
/// verified idle — as the ratio of the two least-squares coefficients. Only
/// the ratio is identifiable and only the ratio is used: the absolute scale
/// would be a millisecond claim, and this model has no business making one.
///
/// Reproduce with `tests/cost_model_ranking.rs`, which carries the same
/// timings and re-derives both this constant and the held-out ordering.
pub const SYNC_EVENT_MAC_EQUIVALENT: f64 = 25_464.0;

/// The shape box the sm_86 timings covered: `m ∈ [1, 32]`, `k, n ∈
/// [1024, 4096]`.
///
/// Narrow on purpose, and narrower than it looks — every calibration point is
/// decode or small-batch. The model still *answers* for prefill shapes, and
/// changes its recommendation there (`32x32x32` below, `128x128x16` at
/// `m ≥ 128`), but that flip is extrapolation from measurements that never
/// entered the regime. [`TileCost::extrapolated`] marks it rather than letting
/// it read like the validated part.
pub const CALIBRATED_SM86: CalibrationDomain = CalibrationDomain {
    m: (1, 32),
    k: (1024, 4096),
    n: (1024, 4096),
};

impl TileCostModel {
    /// The sm_86-calibrated model.
    ///
    /// Used for *any* CUDA arch: the term structure (padding waste, barrier
    /// count) is architectural rather than sm-version-specific, and ordering
    /// is all that is consumed. An arch whose ordering differs will show up as
    /// a prefilter that drops the eventual winner, which
    /// [`Prefilter::dropped`] makes visible rather than silent.
    pub const fn sm86() -> Self {
        Self {
            sync_event_macs: SYNC_EVENT_MAC_EQUIVALENT,
            provenance: CostProvenance::Measured {
                device: "NVIDIA RTX 3080 Ti (sm_86)",
                samples: 18,
                holdout_rho: 0.943,
                // The three calibration shapes: (1,1024,1024), (1,4096,4096),
                // (32,4096,4096). Entirely decode and small-batch — the
                // prefill buckets the tuner declares were never in this fit.
                domain: CALIBRATED_SM86,
            },
        }
    }

    /// A model with the same term structure and no device behind it.
    ///
    /// For backends with no calibration run. Ranks candidates; proves nothing.
    pub const fn structural() -> Self {
        Self {
            sync_event_macs: SYNC_EVENT_MAC_EQUIVALENT,
            provenance: CostProvenance::Structural,
        }
    }

    pub const fn provenance(&self) -> CostProvenance {
        self.provenance
    }

    /// Model `tile` against `workload`.
    ///
    /// Returns `None` for workloads this model does not cover — only
    /// [`Workload::Matmul`] today. `None` means *no opinion*, and callers must
    /// treat it as "keep every candidate", never as "reject".
    pub fn estimate(&self, tile: &TileParams, workload: &Workload) -> Option<TileCost> {
        let Workload::Matmul { m, k, n } = *workload else {
            return None;
        };
        if m == 0 || k == 0 || n == 0 {
            return None;
        }
        let (bm, bn, bk) = (tile.bm as u64, tile.bn as u64, tile.bk as u64);
        if bm == 0 || bn == 0 || bk == 0 {
            return None;
        }
        let (m, k, n) = (m as u64, k as u64, n as u64);

        let tiles_m = m.div_ceil(bm);
        let tiles_n = n.div_ceil(bn);
        let blocks = tiles_m * tiles_n;
        let padded_macs = tiles_m * bm * tiles_n * bn * k;
        let useful_macs = m * k * n;
        let k_iters = k.div_ceil(bk);
        let sync_events = blocks * k_iters;
        // No L2 reuse assumed: every block loads its own A and B stripes.
        let global_elems = blocks * k * (bm + bn);

        // Both terms in padded-MAC equivalents, so the sum is too.
        let compute_term = padded_macs as f64;
        let sync_term = sync_events as f64 * self.sync_event_macs;
        let score = compute_term + sync_term;

        let useful_fraction = useful_macs as f64 / padded_macs as f64;
        let bottleneck = if sync_term > compute_term {
            Bottleneck::KLoopOverhead {
                iters: k_iters.min(u32::MAX as u64) as u32,
            }
        } else if useful_fraction < 0.5 {
            // Compute dominates *and* most of it is padding. Reported ahead of
            // ComputeBound because it names something the caller can fix.
            Bottleneck::PaddingWaste { useful_fraction }
        } else {
            Bottleneck::ComputeBound
        };

        Some(TileCost {
            useful_macs,
            padded_macs,
            blocks,
            k_iters: k_iters.min(u32::MAX as u64) as u32,
            sync_events,
            global_elems,
            score,
            bottleneck,
            extrapolated: !self.provenance.covers(workload),
        })
    }

    /// Rank `candidates` for `workload` and keep the cheapest `keep`.
    ///
    /// The dropped candidates are returned alongside, not discarded silently:
    /// a prefilter that hides what it removed reads as "we measured
    /// everything" when it did not.
    pub fn prefilter(
        &self,
        workload: &Workload,
        candidates: &[TileParams],
        keep: usize,
    ) -> Prefilter {
        let mut scored: Vec<(TileParams, Option<TileCost>)> = candidates
            .iter()
            .map(|t| (*t, self.estimate(t, workload)))
            .collect();

        // A candidate the model cannot score must survive: no opinion is not a
        // rejection. Sorting unscored first keeps them inside `keep`.
        scored.sort_by(|a, b| match (&a.1, &b.1) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (Some(x), Some(y)) => x
                .score
                .partial_cmp(&y.score)
                .unwrap_or(std::cmp::Ordering::Equal),
        });

        let keep = keep.max(1).min(scored.len());
        let dropped = scored.split_off(keep);
        Prefilter {
            kept: scored,
            dropped,
            provenance: self.provenance,
        }
    }
}

impl Default for TileCostModel {
    fn default() -> Self {
        Self::structural()
    }
}

/// The outcome of a [`TileCostModel::prefilter`], including what it removed.
#[derive(Debug, Clone)]
pub struct Prefilter {
    kept: Vec<(TileParams, Option<TileCost>)>,
    dropped: Vec<(TileParams, Option<TileCost>)>,
    provenance: CostProvenance,
}

impl Prefilter {
    /// Candidates worth spending device time on, cheapest-modeled first.
    pub fn kept(&self) -> impl Iterator<Item = (&TileParams, Option<&TileCost>)> {
        self.kept.iter().map(|(t, c)| (t, c.as_ref()))
    }

    /// Candidates the model removed. Never empty-by-omission — callers are
    /// expected to report these.
    pub fn dropped(&self) -> impl Iterator<Item = (&TileParams, Option<&TileCost>)> {
        self.dropped.iter().map(|(t, c)| (t, c.as_ref()))
    }

    pub fn tiles(&self) -> Vec<TileParams> {
        self.kept.iter().map(|(t, _)| *t).collect()
    }

    /// One line naming what was skipped and on whose authority, for the tuner
    /// to print before it starts measuring.
    pub fn disclosure(&self) -> String {
        if self.dropped.is_empty() {
            return format!(
                "cost prefilter: kept all {} candidate(s); nothing skipped",
                self.kept.len()
            );
        }
        let authority = match self.provenance {
            CostProvenance::Measured {
                device,
                holdout_rho,
                domain,
                ..
            } => {
                format!(
                    "model calibrated on {device} over {domain}, held-out rank rho {holdout_rho:.2}"
                )
            }
            CostProvenance::Structural => {
                "UNCALIBRATED model (term structure only, no device behind it)".to_string()
            }
        };
        // Worth its own clause: outside the calibrated box the ordering has no
        // held-out evidence behind it, which is exactly when a prefilter is
        // most likely to discard the tile that would have won.
        let extrapolating = self
            .kept
            .iter()
            .chain(&self.dropped)
            .any(|(_, c)| c.as_ref().is_some_and(|c| c.extrapolated));
        let caveat = if extrapolating {
            " — EXTRAPOLATING outside the calibrated shape box; prefer measuring everything here"
        } else {
            ""
        };
        let names: Vec<String> = self.dropped.iter().map(|(t, _)| t.label()).collect();
        format!(
            "cost prefilter: measuring {} of {}, skipping {} unmeasured [{}] — {authority}{caveat}",
            self.kept.len(),
            self.kept.len() + self.dropped.len(),
            self.dropped.len(),
            names.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::MATMUL_TILE_CANDIDATES;

    fn mm(m: usize, k: usize, n: usize) -> Workload {
        Workload::Matmul { m, k, n }
    }

    #[test]
    fn padding_waste_is_the_decode_diagnosis() {
        // m=1 against a 128-row block tile: 127/128 of the row work is padding.
        let big = TileParams {
            bm: 128,
            bn: 128,
            bk: 16,
            tm: 8,
            tn: 8,
            bdx: 16,
            bdy: 16,
        };
        let cost = TileCostModel::sm86()
            .estimate(&big, &mm(1, 4096, 4096))
            .expect("matmul");
        assert!(
            matches!(cost.bottleneck, Bottleneck::PaddingWaste { .. }),
            "m=1 into bm=128 should read as padding waste, got {}",
            cost.bottleneck
        );
        assert!(
            cost.useful_fraction() < 0.01,
            "got {}",
            cost.useful_fraction()
        );
    }

    #[test]
    fn constant_offsets_cannot_change_ranking() {
        // The argument that justified deleting the fixed-cost term: within one
        // workload it is the same constant for every candidate, so it cannot
        // reorder them. Checked rather than asserted in prose, because the
        // deletion rests on it.
        let model = TileCostModel::sm86();
        let w = mm(1, 4096, 4096);
        let mut scored: Vec<(TileParams, f64)> = MATMUL_TILE_CANDIDATES
            .iter()
            .filter_map(|t| model.estimate(t, &w).map(|c| (*t, c.score)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
        for offset in [0.0, 1.0, 1e6, 1e12] {
            let mut shifted: Vec<(TileParams, f64)> =
                scored.iter().map(|(t, s)| (*t, s + offset)).collect();
            shifted.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"));
            let a: Vec<TileParams> = scored.iter().map(|(t, _)| *t).collect();
            let b: Vec<TileParams> = shifted.iter().map(|(t, _)| *t).collect();
            assert_eq!(a, b, "a constant offset of {offset} reordered candidates");
        }
    }

    #[test]
    fn score_is_not_a_time() {
        // Denominated in padded-MAC equivalents, so a score is ~1e9 and cannot
        // be quietly printed as a millisecond. The measured times for this
        // shape are all under 1 ms.
        let c = TileCostModel::sm86()
            .estimate(&TileParams::DEFAULT_MATMUL, &mm(1, 4096, 4096))
            .expect("matmul");
        assert!(
            c.score > 1e6,
            "score {} is small enough to be mistaken for ms",
            c.score
        );
    }

    #[test]
    fn shallow_bk_reads_as_k_loop_overhead() {
        // Same block tile, same padded MACs; only the staging depth differs.
        let deep = TileParams {
            bm: 32,
            bn: 32,
            bk: 32,
            tm: 2,
            tn: 2,
            bdx: 16,
            bdy: 16,
        };
        let shallow = TileParams { bk: 8, ..deep };
        let model = TileCostModel::sm86();
        let w = mm(1, 4096, 4096);
        let (d, s) = (
            model.estimate(&deep, &w).expect("matmul"),
            model.estimate(&shallow, &w).expect("matmul"),
        );
        assert_eq!(
            d.padded_macs, s.padded_macs,
            "bk must not change padded MACs"
        );
        assert!(
            s.score > d.score,
            "shallower bk should model as more expensive"
        );
        assert_eq!(
            s.sync_events,
            d.sync_events * 4,
            "bk 32->8 is 4x the barriers"
        );
    }

    #[test]
    fn no_opinion_is_not_a_rejection() {
        // Attention is outside the model's coverage; every candidate must
        // survive the prefilter rather than being silently dropped.
        let w = Workload::Attention {
            batch: 1,
            heads: 32,
            seq_q: 1,
        };
        let model = TileCostModel::sm86();
        assert!(model.estimate(&TileParams::DEFAULT_MATMUL, &w).is_none());
        let pf = model.prefilter(&w, MATMUL_TILE_CANDIDATES, 2);
        assert_eq!(
            pf.tiles().len(),
            MATMUL_TILE_CANDIDATES.len().min(2),
            "keep is still honoured"
        );
        // ...but nothing was ordered on a real opinion, so all costs are None.
        assert!(pf.kept().all(|(_, c)| c.is_none()));
    }

    #[test]
    fn prefilter_discloses_what_it_skipped() {
        let pf = TileCostModel::sm86().prefilter(&mm(1, 4096, 4096), MATMUL_TILE_CANDIDATES, 2);
        let line = pf.disclosure();
        assert!(line.contains("skipping"), "{line}");
        for (t, _) in pf.dropped.iter() {
            assert!(
                line.contains(&t.label()),
                "dropped tile missing from disclosure: {line}"
            );
        }
    }

    #[test]
    fn prefill_answers_are_marked_as_extrapolation() {
        let model = TileCostModel::sm86();
        // Inside the calibrated box: decode and small batch.
        for w in [mm(1, 1024, 1024), mm(1, 4096, 4096), mm(32, 4096, 4096)] {
            let c = model
                .estimate(&TileParams::DEFAULT_MATMUL, &w)
                .expect("matmul");
            assert!(!c.extrapolated, "{w:?} was a calibration shape");
        }
        // Outside it: exactly where the model flips its recommendation, and
        // exactly where it has no held-out evidence.
        for w in [
            mm(512, 2048, 2048),
            mm(2048, 2048, 2048),
            mm(4096, 4096, 4096),
        ] {
            let c = model
                .estimate(&TileParams::DEFAULT_MATMUL, &w)
                .expect("matmul");
            assert!(
                c.extrapolated,
                "{w:?} is outside {CALIBRATED_SM86} and must be flagged"
            );
        }
    }

    #[test]
    fn extrapolation_is_disclosed_to_whoever_would_skip_work() {
        let pf = TileCostModel::sm86().prefilter(&mm(2048, 2048, 2048), MATMUL_TILE_CANDIDATES, 2);
        assert!(
            pf.disclosure().contains("EXTRAPOLATING"),
            "{}",
            pf.disclosure()
        );
        let inside = TileCostModel::sm86().prefilter(&mm(1, 4096, 4096), MATMUL_TILE_CANDIDATES, 2);
        assert!(
            !inside.disclosure().contains("EXTRAPOLATING"),
            "{}",
            inside.disclosure()
        );
    }

    #[test]
    fn a_structural_model_extrapolates_everywhere() {
        // No device behind it means no shape is covered, including shapes that
        // happen to sit in the sm_86 box.
        let c = TileCostModel::structural()
            .estimate(&TileParams::DEFAULT_MATMUL, &mm(1, 4096, 4096))
            .expect("matmul");
        assert!(c.extrapolated);
    }

    #[test]
    fn uncalibrated_models_say_so() {
        let pf =
            TileCostModel::structural().prefilter(&mm(1, 4096, 4096), MATMUL_TILE_CANDIDATES, 2);
        assert!(
            pf.disclosure().contains("UNCALIBRATED"),
            "{}",
            pf.disclosure()
        );
        assert!(!TileCostModel::structural().provenance().is_measured());
        assert!(TileCostModel::sm86().provenance().is_measured());
    }

    #[test]
    fn keep_is_clamped_not_panicking() {
        let model = TileCostModel::sm86();
        let w = mm(64, 64, 64);
        assert_eq!(
            model.prefilter(&w, MATMUL_TILE_CANDIDATES, 0).tiles().len(),
            1
        );
        assert_eq!(
            model
                .prefilter(&w, MATMUL_TILE_CANDIDATES, 9999)
                .tiles()
                .len(),
            MATMUL_TILE_CANDIDATES.len()
        );
        assert!(model.prefilter(&w, &[], 3).tiles().is_empty());
    }

    #[test]
    fn degenerate_shapes_get_no_opinion() {
        let model = TileCostModel::sm86();
        assert!(
            model
                .estimate(&TileParams::DEFAULT_MATMUL, &mm(0, 16, 16))
                .is_none()
        );
        assert!(
            model
                .estimate(&TileParams::DEFAULT_MATMUL, &mm(16, 0, 16))
                .is_none()
        );
        assert!(
            model
                .estimate(&TileParams::DEFAULT_MATMUL, &mm(16, 16, 0))
                .is_none()
        );
    }
}
