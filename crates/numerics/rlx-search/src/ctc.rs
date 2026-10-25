// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CTC decoding over `[frames, classes]` row-major score matrices.
//!
//! - [`greedy`] — per-frame argmax, collapse repeats, drop blank.
//! - [`PrefixBeam`] — prefix beam search, streaming (`push` a frame at a
//!   time) or whole-utterance ([`prefix_beam`]); N-best for rescoring.
//! - [`forward`] — the CTC trellis over a [`Semiring`]: [`crate::Log`] gives
//!   `log P(target | x)` (the CTC loss), [`crate::Tropical`] the best
//!   single alignment's score.
//!
//! Scores are log-probabilities for [`PrefixBeam`] and [`forward`]; [`greedy`]
//! only compares within a row, so raw logits work too.

use std::collections::HashMap;

use crate::select::{argmax, log_add};
use crate::semiring::Semiring;

/// One emitted label and the frame that emitted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CtcToken {
    pub id: u32,
    pub frame: u32,
}

/// A decoded label sequence and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcPath {
    pub tokens: Vec<CtcToken>,
    /// [`greedy`]: sum of each frame's best score. [`PrefixBeam`]: total
    /// log-probability of every alignment that collapses to `tokens`.
    pub score: f32,
}

impl CtcPath {
    /// Label ids, without frames.
    pub fn ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.id).collect()
    }
}

/// Collapse a per-frame label sequence: merge consecutive repeats, then drop
/// `blank`. `a a _ a b` → `a a b`.
pub fn collapse(frame_ids: impl IntoIterator<Item = u32>, blank: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut prev = None;
    for id in frame_ids {
        if prev != Some(id) && id != blank {
            out.push(id);
        }
        prev = Some(id);
    }
    out
}

/// Greedy CTC: argmax each frame of `scores` (`[frames, classes]`), collapse
/// repeats, drop `blank`. Each token keeps the first frame of its run.
pub fn greedy(scores: &[f32], classes: usize, blank: u32) -> CtcPath {
    let mut tokens = Vec::new();
    let mut score = 0.0f32;
    let mut prev = None;
    if classes == 0 {
        return CtcPath { tokens, score };
    }
    for (frame, row) in scores.chunks_exact(classes).enumerate() {
        let id = argmax(row) as u32;
        score += row[id as usize];
        if prev != Some(id) && id != blank {
            tokens.push(CtcToken {
                id,
                frame: frame as u32,
            });
        }
        prev = Some(id);
    }
    CtcPath { tokens, score }
}

#[derive(Debug, Clone)]
struct Prefix {
    tokens: Vec<CtcToken>,
    /// log P(prefix, path ends in blank)
    blank: f32,
    /// log P(prefix, path ends in its last label)
    label: f32,
}

impl Prefix {
    fn total(&self) -> f32 {
        log_add(self.blank, self.label)
    }

    fn last(&self) -> Option<u32> {
        self.tokens.last().map(|t| t.id)
    }
}

/// CTC prefix beam search, fed one frame of log-probabilities at a time.
///
/// Hypotheses are label prefixes; each carries the probability of ending in a
/// blank and in its last label, so `a _ a` and `a a` stay distinct. Ties in
/// the per-frame ranking go to the lexicographically smaller prefix, so the
/// result does not depend on hashing order.
#[derive(Debug, Clone)]
pub struct PrefixBeam {
    blank: u32,
    width: usize,
    prune_margin: Option<f32>,
    beams: Vec<Prefix>,
    frames: usize,
}

impl PrefixBeam {
    /// Keep `width` prefixes (at least one), `blank` the blank class.
    pub fn new(blank: u32, width: usize) -> Self {
        let mut s = Self {
            blank,
            width: width.max(1),
            prune_margin: None,
            beams: Vec::new(),
            frames: 0,
        };
        s.reset();
        s
    }

    /// Only extend by labels within `margin` nats of the frame's best score.
    /// The rest contribute about `e^-margin` each; skipping them shrinks the
    /// inner loop from the vocabulary to a handful. Off by default (exact).
    pub fn prune_margin(mut self, margin: f32) -> Self {
        self.prune_margin = Some(margin);
        self
    }

    /// Back to the empty prefix, zero frames.
    pub fn reset(&mut self) {
        self.beams = vec![Prefix {
            tokens: Vec::new(),
            blank: 0.0,
            label: f32::NEG_INFINITY,
        }];
        self.frames = 0;
    }

    /// Frames consumed so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Consume one frame: `row` is `[classes]` log-probabilities.
    pub fn push(&mut self, row: &[f32]) {
        let frame = self.frames as u32;
        let blank = self.blank as usize;
        let thresh = match self.prune_margin {
            Some(m) => row.iter().copied().fold(f32::NEG_INFINITY, f32::max) - m,
            None => f32::NEG_INFINITY,
        };
        let labels: Vec<u32> = (0..row.len())
            .filter(|&c| c != blank && row[c] > thresh)
            .map(|c| c as u32)
            .collect();
        let blank_lp = row.get(blank).copied().unwrap_or(f32::NEG_INFINITY);

        let mut next: Vec<Prefix> = Vec::new();
        let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut bump = |tokens: &[CtcToken], extra: Option<CtcToken>, b: f32, l: f32| {
            let key: Vec<u32> = tokens.iter().chain(extra.as_ref()).map(|t| t.id).collect();
            match index.get(&key) {
                Some(&i) => {
                    next[i].blank = log_add(next[i].blank, b);
                    next[i].label = log_add(next[i].label, l);
                }
                None => {
                    index.insert(key, next.len());
                    let mut toks = tokens.to_vec();
                    toks.extend(extra);
                    next.push(Prefix {
                        tokens: toks,
                        blank: b,
                        label: l,
                    });
                }
            }
        };
        for p in &self.beams {
            let total = p.total();
            bump(&p.tokens, None, total + blank_lp, f32::NEG_INFINITY);
            let last = p.last();
            for &c in &labels {
                let lp = row[c as usize];
                let tok = CtcToken { id: c, frame };
                if last == Some(c) {
                    // Repeat without a blank between: same prefix.
                    bump(&p.tokens, None, f32::NEG_INFINITY, p.label + lp);
                    // A new occurrence needs the blank in between.
                    bump(&p.tokens, Some(tok), f32::NEG_INFINITY, p.blank + lp);
                } else {
                    bump(&p.tokens, Some(tok), f32::NEG_INFINITY, total + lp);
                }
            }
        }
        // Prefixes no alignment reaches (a frame scored them all `-inf`)
        // are dropped, unless nothing else is left.
        if next.iter().any(|p| p.total() > f32::NEG_INFINITY) {
            next.retain(|p| p.total() > f32::NEG_INFINITY);
        }
        rank(&mut next);
        next.truncate(self.width);
        self.beams = next;
        self.frames += 1;
    }

    /// [`PrefixBeam::push`] every row of `[frames, classes]` `scores`.
    pub fn push_frames(&mut self, scores: &[f32], classes: usize) {
        if classes == 0 {
            return;
        }
        for row in scores.chunks_exact(classes) {
            self.push(row);
        }
    }

    /// The current best prefix.
    pub fn best(&self) -> CtcPath {
        self.beams
            .first()
            .map(|p| CtcPath {
                tokens: p.tokens.clone(),
                score: p.total(),
            })
            .unwrap_or(CtcPath {
                tokens: Vec::new(),
                score: f32::NEG_INFINITY,
            })
    }

    /// Every live prefix, best first.
    pub fn nbest(&self) -> Vec<CtcPath> {
        self.beams
            .iter()
            .map(|p| CtcPath {
                tokens: p.tokens.clone(),
                score: p.total(),
            })
            .collect()
    }
}

/// Best first; equal scores by label sequence.
fn rank(v: &mut [Prefix]) {
    v.sort_by(|a, b| {
        b.total().total_cmp(&a.total()).then_with(|| {
            a.tokens
                .iter()
                .map(|t| t.id)
                .cmp(b.tokens.iter().map(|t| t.id))
        })
    });
}

/// Whole-utterance prefix beam search over `[frames, classes]`
/// log-probabilities: up to `width` label sequences, best first.
pub fn prefix_beam(scores: &[f32], classes: usize, blank: u32, width: usize) -> Vec<CtcPath> {
    let mut s = PrefixBeam::new(blank, width);
    s.push_frames(scores, classes);
    s.nbest()
}

/// The CTC trellis over `target` (blank-free labels), summed with semiring
/// `S` across every alignment of `target` to the `[frames, classes]`
/// log-probabilities: [`crate::Log`] → `log P(target | x)`,
/// [`crate::Tropical`] → the best alignment's score. `S::ZERO` when no
/// alignment fits (too few frames for the repeats).
pub fn forward<S: Semiring>(scores: &[f32], classes: usize, blank: u32, target: &[u32]) -> f32 {
    if classes == 0 {
        return S::ZERO;
    }
    // Extended labels: blank, t0, blank, t1, …, blank.
    let ext: Vec<u32> = std::iter::once(blank)
        .chain(target.iter().flat_map(|&t| [t, blank]))
        .collect();
    let n = ext.len();
    let mut alpha = vec![S::ZERO; n];
    let mut prev = vec![S::ZERO; n];
    for (t, row) in scores.chunks_exact(classes).enumerate() {
        std::mem::swap(&mut alpha, &mut prev);
        for s in 0..n {
            let emit = row[ext[s] as usize];
            let mut acc = if t == 0 {
                if s < 2 { S::ONE } else { S::ZERO }
            } else {
                let mut a = prev[s];
                if s >= 1 {
                    a = S::plus(a, prev[s - 1]);
                }
                // Skip the blank between two different labels.
                if s >= 2 && ext[s] != blank && ext[s] != ext[s - 2] {
                    a = S::plus(a, prev[s - 2]);
                }
                a
            };
            if acc != S::ZERO {
                acc = S::times(acc, emit);
            }
            alpha[s] = acc;
        }
    }
    if scores.len() < classes {
        // No frames: only the empty target has a (zero-length) alignment.
        return if target.is_empty() { S::ONE } else { S::ZERO };
    }
    if n >= 2 {
        S::plus(alpha[n - 1], alpha[n - 2])
    } else {
        alpha[n - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{Log, Tropical};

    /// `[frames, classes]` log-probs from per-frame probabilities.
    fn logp(rows: &[&[f32]]) -> (Vec<f32>, usize) {
        let classes = rows[0].len();
        (
            rows.iter().flat_map(|r| r.iter().map(|p| p.ln())).collect(),
            classes,
        )
    }

    #[test]
    fn collapse_merges_runs_then_drops_blank() {
        assert_eq!(collapse([1, 1, 0, 1], 0), vec![1, 1]);
        assert_eq!(collapse([0, 1, 1, 1, 0, 2], 0), vec![1, 2]);
        assert_eq!(collapse([0, 0, 0], 0), Vec::<u32>::new());
        assert_eq!(collapse([3, 1, 2], 0), vec![3, 1, 2]);
    }

    #[test]
    fn greedy_keeps_first_frame_of_each_run() {
        let scores = [
            0.0, 0.9, 0.1, 0.0, //
            0.0, 0.9, 0.1, 0.0, //
            0.0, 0.1, 0.8, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ];
        let p = greedy(&scores, 4, 0);
        let got: Vec<(u32, u32)> = p.tokens.iter().map(|t| (t.id, t.frame)).collect();
        assert_eq!(got, vec![(1, 0), (2, 2)]);
        assert!((p.score - 3.6).abs() < 1e-6);
        // Blank that is not class 0.
        let s = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        assert_eq!(greedy(&s, 4, 3).ids(), vec![0, 1]);
    }

    #[test]
    fn prefix_beam_sums_alignments_that_greedy_misses() {
        // Greedy reads blank, blank (empty); but P("a") = 1 - P(empty) wins.
        let (lp, c) = logp(&[&[0.6, 0.4], &[0.6, 0.4]]);
        assert!(greedy(&lp, c, 0).tokens.is_empty());
        let best = prefix_beam(&lp, c, 0, 4).remove(0);
        assert_eq!(best.ids(), vec![1]);
        let p_a = 0.4 * 0.4 + 0.6 * 0.4 + 0.4 * 0.6;
        assert!((best.score - (p_a as f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn prefix_beam_matches_the_log_forward_pass() {
        // With a wide beam nothing is pruned, so each prefix's score is the
        // full CTC likelihood of that label sequence.
        let (lp, c) = logp(&[
            &[0.5, 0.3, 0.2],
            &[0.2, 0.5, 0.3],
            &[0.4, 0.2, 0.4],
            &[0.3, 0.3, 0.4],
        ]);
        for p in prefix_beam(&lp, c, 0, 1000) {
            let f = forward::<Log>(&lp, c, 0, &p.ids());
            assert!(
                (p.score - f).abs() < 1e-5,
                "{:?}: {} vs {f}",
                p.ids(),
                p.score
            );
        }
    }

    #[test]
    fn streaming_matches_whole_utterance() {
        let (lp, c) = logp(&[&[0.5, 0.3, 0.2], &[0.2, 0.5, 0.3], &[0.6, 0.2, 0.2]]);
        let whole = prefix_beam(&lp, c, 0, 3);
        let mut s = PrefixBeam::new(0, 3);
        for row in lp.chunks_exact(c) {
            s.push(row);
        }
        assert_eq!(s.frames(), 3);
        assert_eq!(s.nbest(), whole);
        s.reset();
        assert_eq!(s.best().ids(), Vec::<u32>::new());
    }

    #[test]
    fn repeated_label_needs_a_blank() {
        // "a a" is only reachable through a blank frame between.
        let (lp, c) = logp(&[&[0.01, 0.99], &[0.98, 0.02], &[0.01, 0.99]]);
        assert_eq!(prefix_beam(&lp, c, 0, 4)[0].ids(), vec![1, 1]);
        let (lp, c) = logp(&[&[0.01, 0.99], &[0.01, 0.99], &[0.01, 0.99]]);
        assert_eq!(prefix_beam(&lp, c, 0, 4)[0].ids(), vec![1]);
    }

    #[test]
    fn tropical_forward_is_the_best_alignment() {
        let (lp, c) = logp(&[&[0.6, 0.4], &[0.6, 0.4]]);
        // Best alignment of "a": (a,_), (_,a) or (a,a); max is 0.4*0.6.
        let best = forward::<Tropical>(&lp, c, 0, &[1]);
        assert!((best - (0.24f32).ln()).abs() < 1e-6);
        assert!(forward::<Log>(&lp, c, 0, &[1]) > best);
        // "a a" needs three frames.
        assert_eq!(forward::<Tropical>(&lp, c, 0, &[1, 1]), f32::NEG_INFINITY);
        assert_eq!(forward::<Log>(&[], c, 0, &[]), 0.0);
    }

    /// The prefix beam `rlx-asr` and `rlx-ocr2` each carried (identical
    /// copies), verbatim apart from names: a `HashMap` beam, 9-nat margin.
    fn old_model_crate_prefix_beam(
        logp: &[f32],
        seq: usize,
        classes: usize,
        blank: usize,
        beam: usize,
    ) -> Vec<(Vec<usize>, f32)> {
        let mut beams: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
        beams.insert(Vec::new(), (0.0, f32::NEG_INFINITY));
        const PRUNE_MARGIN: f32 = 9.0;
        for t in 0..seq {
            let row = &logp[t * classes..(t + 1) * classes];
            let maxlp = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let thresh = maxlp - PRUNE_MARGIN;
            let cands: Vec<usize> = (0..classes)
                .filter(|&c| c != blank && row[c] > thresh)
                .collect();
            let mut next: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
            let bump =
                |map: &mut HashMap<Vec<usize>, (f32, f32)>, key: Vec<usize>, db: f32, dnb: f32| {
                    let e = map
                        .entry(key)
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    e.0 = log_add(e.0, db);
                    e.1 = log_add(e.1, dnb);
                };
            for (prefix, &(pb, pnb)) in &beams {
                bump(
                    &mut next,
                    prefix.clone(),
                    log_add(pb, pnb) + row[blank],
                    f32::NEG_INFINITY,
                );
                let last = prefix.last().copied();
                for &c in &cands {
                    let lp = row[c];
                    if last == Some(c) {
                        bump(&mut next, prefix.clone(), f32::NEG_INFINITY, pnb + lp);
                        let mut np = prefix.clone();
                        np.push(c);
                        bump(&mut next, np, f32::NEG_INFINITY, pb + lp);
                    } else {
                        let mut np = prefix.clone();
                        np.push(c);
                        bump(&mut next, np, f32::NEG_INFINITY, log_add(pb, pnb) + lp);
                    }
                }
            }
            let mut v: Vec<(Vec<usize>, (f32, f32))> = next.into_iter().collect();
            v.sort_by(|a, b| {
                log_add(b.1.0, b.1.1)
                    .partial_cmp(&log_add(a.1.0, a.1.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v.truncate(beam.max(1));
            beams = v.into_iter().collect();
        }
        let mut out: Vec<(Vec<usize>, f32)> = beams
            .into_iter()
            .map(|(p, (pb, pnb))| (p, log_add(pb, pnb)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    #[test]
    fn reproduces_the_model_crates_prefix_beam() {
        let (frames, classes) = (12, 7);
        for seed in 0..100u32 {
            // Pseudo-random rows, log-softmaxed.
            let mut lp = Vec::with_capacity(frames * classes);
            for t in 0..frames {
                let mut row: Vec<f32> = (0..classes)
                    .map(|c| {
                        // Full-mantissa values: exact score ties would make
                        // the old `HashMap` order (random) part of the result.
                        let mut x = (seed as u64) << 40 ^ (t as u64) << 20 ^ c as u64;
                        x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        x ^= x >> 29;
                        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        (x >> 40) as f32 / (1u64 << 24) as f32 * 8.0
                    })
                    .collect();
                crate::select::log_softmax(&mut row);
                lp.extend(row);
            }
            for width in [1, 3, 8] {
                let want = old_model_crate_prefix_beam(&lp, frames, classes, 0, width);
                let mut s = PrefixBeam::new(0, width).prune_margin(9.0);
                s.push_frames(&lp, classes);
                let got = s.nbest();
                assert_eq!(got.len(), want.len(), "seed {seed} width {width}");
                for (g, (ids, score)) in got.iter().zip(&want) {
                    assert!((g.score - score).abs() < 1e-5, "seed {seed}");
                    let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
                    assert_eq!(g.ids(), ids, "seed {seed} width {width}");
                }
            }
        }
    }

    #[test]
    fn prune_margin_keeps_the_dominant_path() {
        let (lp, c) = logp(&[
            &[0.01, 0.98, 0.01],
            &[0.98, 0.01, 0.01],
            &[0.01, 0.01, 0.98],
        ]);
        let exact = prefix_beam(&lp, c, 0, 4)[0].ids();
        let mut s = PrefixBeam::new(0, 4).prune_margin(2.0);
        s.push_frames(&lp, c);
        assert_eq!(s.best().ids(), exact);
        assert_eq!(exact, vec![1, 2]);
    }
}
