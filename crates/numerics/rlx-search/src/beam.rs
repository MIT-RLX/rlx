// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Autoregressive beam search, one engine for every decoder.
//!
//! [`BeamSearch`] is a state machine: give it one score row per live
//! hypothesis ([`BeamSearch::advance`]) and it picks the next beam, telling
//! the caller which parent each new hypothesis extends and with which token
//! ([`Step`]). The caller owns the model and its per-hypothesis state (KV
//! caches are reordered by [`Step::parents`]), so the same engine drives a
//! re-prefill loop, a KV-cached decoder, or several searches batched into one
//! forward. [`beam_search`] is the loop for the common single-search case.
//!
//! Only survivors are scored: a step costs one model call per kept
//! hypothesis, not one per candidate.
//!
//! The rules that differ between reference decoders are explicit options,
//! not forks: which stop-token candidates finish ([`FinishRule`]), when the
//! search ends ([`BeamConfig::pool`], [`BeamConfig::stop_when_unbeatable`]),
//! and what happens to unfinished hypotheses ([`LiveAtEnd`]).
//! [`BeamConfig::new`] is the Hugging Face `generate` rule;
//! [`BeamConfig::whisper`] is `openai/whisper`'s `BeamSearchDecoder`.
//!
//! Ranking is deterministic: equal scores go to the earlier parent, then the
//! lower token id.

use crate::select::{log_softmax, top_k};

/// What the rows passed to [`BeamSearch::advance`] hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scores {
    /// Raw logits; masks apply first, then log-softmax.
    Logits,
    /// Already log-probabilities; masks apply as is (no renormalization).
    LogProbs,
}

/// Which stop-token candidates become finished hypotheses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishRule {
    /// Only those ranked within the top `beams` of the step (Hugging Face).
    WithinBeams,
    /// Those ranked before the live beam fills, best first, while the
    /// finished pool has room (`openai/whisper`).
    UntilBeamFull,
    /// Every one proposed this step.
    Any,
}

/// What [`BeamSearch::finish`] does with hypotheses that never stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveAtEnd {
    /// They compete with the finished ones.
    Always,
    /// Only when nothing finished (report a truncated result, not none).
    IfNoneFinished,
    /// Best first, until the pool holds [`BeamConfig::pool`].
    FillPool,
}

/// Beam search settings. Build with [`BeamConfig::new`] or a preset, then
/// adjust fields.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Live hypotheses kept per step.
    pub beams: usize,
    /// Candidates each hypothesis proposes per step (its top-k).
    pub proposals: usize,
    /// Tokens generated at most.
    pub max_new: usize,
    /// Tokens that end a hypothesis; never part of its result.
    pub stop: Vec<u32>,
    /// Stop tokens are masked until this many tokens are generated.
    pub min_new: usize,
    /// Tokens never proposed.
    pub suppress: Vec<u32>,
    /// Never propose a token that repeats an `n`-gram already in the
    /// sequence (prefix included). `0` disables.
    pub no_repeat_ngram: usize,
    /// Rank by `score / len^alpha` (`len` excludes the stop token).
    pub length_penalty: f64,
    pub scores: Scores,
    pub finish: FinishRule,
    /// The search ends once this many hypotheses finished.
    pub pool: usize,
    /// With [`BeamConfig::pool`] reached, keep going while a live
    /// hypothesis still out-scores the pool's worst kept member (raw
    /// scores only fall, so after that nothing can overtake it).
    pub stop_when_unbeatable: bool,
    /// Drop live hypotheses more than `factor × |best|` below the best.
    pub prune_relative: Option<f64>,
    pub live_at_end: LiveAtEnd,
    /// Hypotheses returned by [`BeamSearch::finish`].
    pub nbest: usize,
}

impl BeamConfig {
    /// The Hugging Face `generate` rule: `2 × beams` proposals per
    /// hypothesis, a stop token finishes only within the top `beams`, done
    /// at `beams` finished, unfinished hypotheses compete at the end,
    /// `length_penalty = 1`.
    pub fn new(beams: usize, max_new: usize) -> Self {
        let beams = beams.max(1);
        Self {
            beams,
            proposals: 2 * beams,
            max_new,
            stop: Vec::new(),
            min_new: 0,
            suppress: Vec::new(),
            no_repeat_ngram: 0,
            length_penalty: 1.0,
            scores: Scores::Logits,
            finish: FinishRule::WithinBeams,
            pool: beams,
            stop_when_unbeatable: false,
            prune_relative: None,
            live_at_end: LiveAtEnd::Always,
            nbest: 1,
        }
    }

    /// `openai/whisper`'s `BeamSearchDecoder` (patience 1): `beams + 1`
    /// proposals, stop-token candidates ranked before the beam fills finish
    /// while fewer than `beams` have, unfinished ones top the pool up at the
    /// end, ranked by `score / len`; the end-of-text token can't come first.
    pub fn whisper(beams: usize, max_new: usize, eot: u32) -> Self {
        let beams = beams.max(1);
        Self {
            proposals: beams + 1,
            stop: vec![eot],
            min_new: 1,
            finish: FinishRule::UntilBeamFull,
            live_at_end: LiveAtEnd::FillPool,
            ..Self::new(beams, max_new)
        }
    }

    pub fn stop(mut self, stop: impl Into<Vec<u32>>) -> Self {
        self.stop = stop.into();
        self
    }

    pub fn length_penalty(mut self, alpha: f64) -> Self {
        self.length_penalty = alpha;
        self
    }

    fn normalized(&self, score: f64, len: usize) -> f64 {
        if self.length_penalty == 0.0 {
            score
        } else {
            score / (len.max(1) as f64).powf(self.length_penalty)
        }
    }
}

/// A search result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Generated tokens: no prefix, no stop token.
    pub tokens: Vec<u32>,
    /// Summed log-probability, the stop token's included.
    pub score: f64,
    /// `score / len^length_penalty`, what results are ranked by.
    pub normalized: f64,
    /// Ended on a stop token (else cut off by `max_new` or the search end).
    pub finished: bool,
}

/// Per-hypothesis masking and vetoes, beyond the built-in ones in
/// [`BeamConfig`]. `sequence` is the full sequence, prefix included.
pub trait Constraint {
    /// Set disallowed next tokens to `-inf` in `scores` (before
    /// normalization when the rows are [`Scores::Logits`]).
    fn mask(&mut self, _sequence: &[u32], _generated: usize, _scores: &mut [f32]) {}

    /// Veto one proposed token. For checks too expensive to run on every
    /// vocabulary entry; a vetoed proposal is not replaced.
    fn reject(&mut self, _sequence: &[u32], _token: u32) -> bool {
        false
    }
}

/// No constraint.
impl Constraint for () {}

#[derive(Debug, Clone)]
struct Live {
    sequence: Vec<u32>,
    score: f64,
}

/// The hypotheses chosen by one [`BeamSearch::advance`]: new hypothesis `i`
/// is hypothesis `parents[i]` of the previous step plus `tokens[i]`. Score
/// each one for the next call.
#[derive(Debug, Clone, Copy)]
pub struct Step<'a> {
    pub parents: &'a [usize],
    pub tokens: &'a [u32],
    live: &'a [Live],
    prefix_len: usize,
}

impl<'a> Step<'a> {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Hypothesis `i`'s full sequence, prefix included.
    pub fn sequence(&self, i: usize) -> &'a [u32] {
        &self.live[i].sequence
    }

    /// Hypothesis `i`'s generated tokens.
    pub fn generated(&self, i: usize) -> &'a [u32] {
        &self.live[i].sequence[self.prefix_len..]
    }
}

/// Beam search state. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct BeamSearch {
    cfg: BeamConfig,
    prefix_len: usize,
    live: Vec<Live>,
    finished: Vec<Hypothesis>,
    generated: usize,
    done: bool,
    parents: Vec<usize>,
    tokens: Vec<u32>,
    scratch: Vec<f32>,
}

impl BeamSearch {
    /// One live hypothesis, `prefix`, waiting for its scores.
    pub fn new(cfg: BeamConfig, prefix: &[u32]) -> Self {
        let done = cfg.max_new == 0;
        Self {
            cfg,
            prefix_len: prefix.len(),
            live: vec![Live {
                sequence: prefix.to_vec(),
                score: 0.0,
            }],
            finished: Vec::new(),
            generated: 0,
            done,
            parents: Vec::new(),
            tokens: Vec::new(),
            scratch: Vec::new(),
        }
    }

    pub fn config(&self) -> &BeamConfig {
        &self.cfg
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Tokens generated so far by every live hypothesis.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Live hypotheses, the rows the next [`BeamSearch::advance`] takes.
    pub fn live_len(&self) -> usize {
        self.live.len()
    }

    /// Live hypothesis `i`'s full sequence.
    pub fn sequence(&self, i: usize) -> &[u32] {
        &self.live[i].sequence
    }

    /// Choose the next beam from `rows[i]`, the next-token scores of live
    /// hypothesis `i`. Returns the step to score, or `None` once done.
    ///
    /// # Panics
    /// If called after the search is done, or with a row count other than
    /// [`BeamSearch::live_len`].
    pub fn advance(&mut self, rows: &[&[f32]], c: &mut dyn Constraint) -> Option<Step<'_>> {
        assert!(!self.done, "advance after the beam search finished");
        assert_eq!(
            rows.len(),
            self.live.len(),
            "one score row per live hypothesis"
        );
        let cfg = &self.cfg;
        let mut cands: Vec<(f64, usize, u32)> = Vec::new();
        for (parent, (hyp, row)) in self.live.iter().zip(rows).enumerate() {
            let s = &mut self.scratch;
            s.clear();
            s.extend_from_slice(row);
            mask_ids(s, &cfg.suppress);
            if self.generated < cfg.min_new {
                mask_ids(s, &cfg.stop);
            }
            mask_ngram_repeats(s, &hyp.sequence, cfg.no_repeat_ngram);
            c.mask(&hyp.sequence, self.generated, s);
            if cfg.scores == Scores::Logits {
                log_softmax(s);
            }
            for (tok, lp) in top_k(s, cfg.proposals) {
                if !c.reject(&hyp.sequence, tok) {
                    cands.push((hyp.score + lp as f64, parent, tok));
                }
            }
        }
        if cands.is_empty() {
            // Nothing can be proposed (every token masked or vetoed): stop
            // with the current beam, which [`BeamSearch::finish`] reports as
            // cut off.
            self.done = true;
            return None;
        }
        // Stable: equal scores keep parent order, then each parent's
        // proposal order (lower id first).
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut next: Vec<Live> = Vec::with_capacity(cfg.beams);
        let mut newly: Vec<Hypothesis> = Vec::new();
        self.parents.clear();
        self.tokens.clear();
        for (rank, &(score, parent, tok)) in cands.iter().enumerate() {
            if cfg.stop.contains(&tok) {
                if cfg.finish != FinishRule::WithinBeams || rank < cfg.beams {
                    let tokens = self.live[parent].sequence[self.prefix_len..].to_vec();
                    let normalized = cfg.normalized(score, tokens.len());
                    newly.push(Hypothesis {
                        tokens,
                        score,
                        normalized,
                        finished: true,
                    });
                }
                continue;
            }
            if next.len() < cfg.beams {
                let mut sequence = self.live[parent].sequence.clone();
                sequence.push(tok);
                next.push(Live { sequence, score });
                self.parents.push(parent);
                self.tokens.push(tok);
            }
            if next.len() == cfg.beams && cfg.finish != FinishRule::Any {
                break;
            }
        }
        if cfg.finish == FinishRule::UntilBeamFull {
            newly.sort_by(|a, b| b.score.total_cmp(&a.score));
            let room = cfg.pool.saturating_sub(self.finished.len());
            newly.truncate(room);
        }
        self.finished.extend(newly);

        if let Some(factor) = cfg.prune_relative
            && let Some(best) = next.first().map(|h| h.score)
        {
            // `next` is best first, so the survivors are a prefix.
            let cutoff = best - factor * best.abs();
            let kept = next.iter().take_while(|h| h.score >= cutoff).count();
            next.truncate(kept);
            self.parents.truncate(kept);
            self.tokens.truncate(kept);
        }

        self.generated += 1;
        self.live = next;
        self.done = self.live.is_empty() || self.generated >= self.cfg.max_new || self.pool_done();
        if self.done {
            return None;
        }
        Some(Step {
            parents: &self.parents,
            tokens: &self.tokens,
            live: &self.live,
            prefix_len: self.prefix_len,
        })
    }

    fn pool_done(&mut self) -> bool {
        let pool = self.cfg.pool.max(1);
        if self.finished.len() < pool {
            return false;
        }
        if !self.cfg.stop_when_unbeatable {
            return true;
        }
        self.finished
            .sort_by(|a, b| b.normalized.total_cmp(&a.normalized));
        let worst_kept = self.finished[pool - 1].score;
        self.live.iter().all(|h| h.score <= worst_kept)
    }

    /// The best [`BeamConfig::nbest`] hypotheses, best first.
    pub fn finish(self) -> Vec<Hypothesis> {
        let Self {
            cfg,
            prefix_len,
            live,
            mut finished,
            ..
        } = self;
        let add = match cfg.live_at_end {
            LiveAtEnd::Always => live.len(),
            LiveAtEnd::IfNoneFinished if finished.is_empty() => live.len(),
            LiveAtEnd::IfNoneFinished => 0,
            LiveAtEnd::FillPool => cfg.pool.saturating_sub(finished.len()),
        };
        // `live` is in rank order, best first.
        for h in live.into_iter().take(add) {
            let tokens = h.sequence[prefix_len..].to_vec();
            let normalized = cfg.normalized(h.score, tokens.len());
            finished.push(Hypothesis {
                tokens,
                score: h.score,
                normalized,
                finished: false,
            });
        }
        finished.sort_by(|a, b| b.normalized.total_cmp(&a.normalized));
        finished.truncate(cfg.nbest.max(1));
        finished
    }
}

fn mask_ids(scores: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(v) = scores.get_mut(id as usize) {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Mask every token that would complete an `n`-gram already in `seq`.
fn mask_ngram_repeats(scores: &mut [f32], seq: &[u32], n: usize) {
    if n == 0 || seq.len() + 1 < n {
        return;
    }
    let tail = &seq[seq.len() + 1 - n..];
    for w in seq.windows(n) {
        if w[..n - 1] == *tail
            && let Some(v) = scores.get_mut(w[n - 1] as usize)
        {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Run a beam search to the end. `first` scores the token after `prefix`;
/// `step` scores every hypothesis of a [`Step`], one row each.
pub fn beam_search<E>(
    cfg: BeamConfig,
    prefix: &[u32],
    first: &[f32],
    constraint: &mut dyn Constraint,
    mut step: impl FnMut(Step<'_>) -> Result<Vec<Vec<f32>>, E>,
) -> Result<Vec<Hypothesis>, E> {
    let mut bs = BeamSearch::new(cfg, prefix);
    let mut rows: Vec<Vec<f32>> = vec![first.to_vec()];
    while !bs.is_done() {
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        match bs.advance(&refs, constraint) {
            Some(s) => rows = step(s)?,
            None => break,
        }
    }
    Ok(bs.finish())
}

#[cfg(test)]
mod tests;
