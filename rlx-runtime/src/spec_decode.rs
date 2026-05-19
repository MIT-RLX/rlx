// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Speculative decoding scheduling pattern (plan #34).
//!
//! Borrowed from MAX's serving scheduler structure
//! (`one_shot_scheduler.py`, decode/prefill split). The classic
//! Leviathan-et-al "Fast Inference from Transformers via
//! Speculative Decoding" algorithm — a small draft model proposes
//! `n` tokens; the larger target model verifies all `n` in one
//! forward pass; tokens are accepted up to the first rejection,
//! then one extra "corrected" token is sampled from the residual
//! distribution.
//!
//! Expected speedup on decode-heavy workloads: 2-3×.
//!
//! Layout:
//!   - [`Speculator`] — trait an autoregressive model implements.
//!     Two methods: `propose` (draft) and `verify` (target).
//!   - [`DraftProposal`] / [`VerifyResult`] / [`AcceptDecision`]
//!     — wire-format data shapes.
//!   - [`speculative_accept`] — pure function that runs the
//!     acceptance algorithm. Testable without a real model.
//!   - [`SpecDecoder`] — orchestrator that calls a draft + target
//!     and returns the next batch of accepted tokens.

use rlx_ir::Philox4x32;

/// One round of draft proposals.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// `n` proposed tokens (draft sampled greedily or stochastically).
    pub tokens: Vec<u32>,
    /// `[n, vocab]` row-major — the draft's probability for each
    /// token at that position. `probs[i][tokens[i]]` is the
    /// probability the draft assigned to its own choice.
    pub probs: Vec<Vec<f32>>,
}

/// Target model's verification of the draft's proposals.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// `[n, vocab]` row-major — target's probability at each
    /// position, conditioned on the prefix and all preceding
    /// draft tokens.
    pub probs: Vec<Vec<f32>>,
}

/// Outcome of one speculative-decoding round.
#[derive(Debug, Clone)]
pub struct AcceptDecision {
    /// Tokens accepted. Length is `0..=n`.
    pub accepted: Vec<u32>,
    /// One extra token sampled from the target's distribution
    /// after rejection — `None` only when all `n` are accepted.
    /// Either way the round produces `accepted.len() + 1` real
    /// tokens (the +1 is `corrected` *or* a final target sample).
    pub corrected: Option<u32>,
}

impl AcceptDecision {
    /// Total real tokens this round produced.
    pub fn total_tokens(&self) -> usize {
        self.accepted.len() + if self.corrected.is_some() { 1 } else { 0 }
    }
}

/// Streaming speculator interface — one method to draft, one to
/// verify. Real implementations bind to a `CompiledGraph` per
/// model; testable implementations can return canned probability
/// tables.
pub trait Speculator {
    /// Propose `n` tokens given the current `context`. Returns the
    /// proposed tokens + the draft's probability tables.
    fn propose(&mut self, context: &[u32], n: usize) -> DraftProposal;

    /// Verify a batch of `proposed` tokens in one forward pass:
    /// for each position `i ∈ 0..n`, return the *target* model's
    /// probability distribution conditioned on
    /// `context ++ proposed[..i]`.
    fn verify(&mut self, context: &[u32], proposed: &[u32]) -> VerifyResult;
}

/// Pure speculative-acceptance algorithm. Given the draft's
/// proposal and the target's verification, runs the
/// per-position accept/reject test and returns the final
/// decision. No model state, no I/O — easy to unit-test against
/// hand-built distributions.
///
/// Algorithm (Leviathan et al. 2022, Algorithm 1):
///   for i in 0..n:
///     r ~ Uniform(0,1)
///     if r < min(1, q_target(x_i) / p_draft(x_i)):
///       accept x_i
///     else:
///       sample x' from norm(max(0, q - p))
///       return (accepted[..i], Some(x'))
///   return (all n accepted, None)
pub fn speculative_accept(
    proposal: &DraftProposal,
    verify: &VerifyResult,
    rng: &mut Philox4x32,
) -> AcceptDecision {
    assert_eq!(
        proposal.tokens.len(),
        proposal.probs.len(),
        "DraftProposal: tokens and probs must agree"
    );
    assert_eq!(
        proposal.probs.len(),
        verify.probs.len(),
        "DraftProposal and VerifyResult must propose the same n"
    );
    let n = proposal.tokens.len();
    let mut accepted: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let token = proposal.tokens[i];
        let p = proposal.probs[i][token as usize].max(f32::MIN_POSITIVE);
        let q = verify.probs[i][token as usize];
        let accept_ratio = (q / p).min(1.0);
        let r = rng.next_f32();
        if r < accept_ratio {
            accepted.push(token);
        } else {
            let corrected = sample_corrected_residual(&proposal.probs[i], &verify.probs[i], rng);
            return AcceptDecision {
                accepted,
                corrected: Some(corrected),
            };
        }
    }
    AcceptDecision {
        accepted,
        corrected: None,
    }
}

/// Sample from the *residual* distribution `norm(max(0, q - p))`.
/// This is the "what the target prefers but the draft missed"
/// distribution, used after a rejection so the round still emits
/// a valid sample from the target.
fn sample_corrected_residual(p: &[f32], q: &[f32], rng: &mut Philox4x32) -> u32 {
    let mut adj: Vec<f32> = q.iter().zip(p).map(|(qi, pi)| (qi - pi).max(0.0)).collect();
    let sum: f32 = adj.iter().sum();
    if sum <= f32::MIN_POSITIVE {
        // q ≤ p elementwise (extreme edge case): fall back to
        // sampling from q directly.
        return sample_from(q, rng);
    }
    let inv = 1.0 / sum;
    for v in adj.iter_mut() {
        *v *= inv;
    }
    sample_from(&adj, rng)
}

fn sample_from(probs: &[f32], rng: &mut Philox4x32) -> u32 {
    let r = rng.next_f32();
    let mut acc = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Top-level orchestrator. Holds a draft + target speculator and
/// the lookahead window `n`. `step()` runs one full round and
/// returns the tokens to append to the running context.
pub struct SpecDecoder<D: Speculator, T: Speculator> {
    pub draft: D,
    pub target: T,
    pub n: usize,
    rng: Philox4x32,
}

impl<D: Speculator, T: Speculator> SpecDecoder<D, T> {
    pub fn new(draft: D, target: T, n: usize, seed: u64) -> Self {
        Self {
            draft,
            target,
            n,
            rng: Philox4x32::new(seed),
        }
    }

    /// One speculative-decoding round. Returns the tokens that
    /// should be appended to `context`.
    pub fn step(&mut self, context: &[u32]) -> Vec<u32> {
        let proposal = self.draft.propose(context, self.n);
        let verify = self.target.verify(context, &proposal.tokens);
        let decision = speculative_accept(&proposal, &verify, &mut self.rng);
        let mut out = decision.accepted;
        if let Some(c) = decision.corrected {
            out.push(c);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When draft and target agree perfectly (same probs), every
    /// proposed token must be accepted (accept_ratio = 1.0).
    #[test]
    fn identical_distributions_accept_all() {
        let n = 4;
        let vocab = 8;
        // Draft proposed token = argmax of a peaked distribution.
        // Target's distribution is identical → q/p = 1.0 → always
        // accept.
        let mut probs = Vec::with_capacity(n);
        let mut tokens = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0.01f32; vocab];
            let pick = (i * 2) % vocab;
            row[pick] = 1.0 - 0.01 * (vocab - 1) as f32;
            probs.push(row);
            tokens.push(pick as u32);
        }
        let proposal = DraftProposal {
            tokens: tokens.clone(),
            probs: probs.clone(),
        };
        let verify = VerifyResult { probs };

        // 100 trials with different seeds; all should accept all 4.
        for seed in 0..100u64 {
            let mut rng = Philox4x32::new(seed + 1);
            let d = speculative_accept(&proposal, &verify, &mut rng);
            assert_eq!(d.accepted, tokens, "seed {seed}: should accept all");
            assert!(d.corrected.is_none());
        }
    }

    /// When the draft places mass on tokens the target rejects
    /// (q ≪ p on those tokens), at least some rejections happen.
    #[test]
    fn divergent_distributions_reject_sometimes() {
        let n = 4;
        let _vocab = 4;
        // Draft ALWAYS picks token 0; target wants token 3.
        let draft_row = vec![0.97f32, 0.01, 0.01, 0.01];
        let target_row = vec![0.01f32, 0.01, 0.01, 0.97];
        let proposal = DraftProposal {
            tokens: vec![0u32; n],
            probs: vec![draft_row.clone(); n],
        };
        let verify = VerifyResult {
            probs: vec![target_row.clone(); n],
        };

        let mut total_accepted = 0usize;
        let trials = 200;
        for seed in 0..trials {
            let mut rng = Philox4x32::new(seed + 1);
            let d = speculative_accept(&proposal, &verify, &mut rng);
            total_accepted += d.accepted.len();
            // After rejection, corrected must be present.
            if d.accepted.len() < n {
                assert!(
                    d.corrected.is_some(),
                    "rejection at seed {seed} should yield a corrected token"
                );
                // Corrected token should be drawn from
                // norm(max(0, q-p)) which strongly favours token 3.
            }
        }
        // q/p = 0.01/0.97 ≈ 0.0103 per token → expected acceptance
        // length per round is geometric, mean ≈ 0.01. Across 200
        // trials × 4 positions = 800 chances, accept rate ~1%.
        assert!(
            total_accepted < 80,
            "divergent distributions should accept rarely; got {total_accepted}/800"
        );
    }

    /// Mock speculators for end-to-end SpecDecoder smoke test.
    /// Both return canned probability tables.
    struct CannedSpeculator {
        next_token: u32,
        peaked_prob: f32,
    }

    impl Speculator for CannedSpeculator {
        fn propose(&mut self, _ctx: &[u32], n: usize) -> DraftProposal {
            let vocab = 8;
            let mut probs = Vec::with_capacity(n);
            for _ in 0..n {
                let mut row = vec![(1.0 - self.peaked_prob) / (vocab - 1) as f32; vocab];
                row[self.next_token as usize] = self.peaked_prob;
                probs.push(row);
            }
            DraftProposal {
                tokens: vec![self.next_token; n],
                probs,
            }
        }
        fn verify(&mut self, _ctx: &[u32], proposed: &[u32]) -> VerifyResult {
            // Canned target: identical distribution to its own
            // "next_token" choice.
            let n = proposed.len();
            let vocab = 8;
            let mut probs = Vec::with_capacity(n);
            for _ in 0..n {
                let mut row = vec![(1.0 - self.peaked_prob) / (vocab - 1) as f32; vocab];
                row[self.next_token as usize] = self.peaked_prob;
                probs.push(row);
            }
            VerifyResult { probs }
        }
    }

    #[test]
    fn spec_decoder_step_emits_n_plus_1_tokens_when_aligned() {
        let draft = CannedSpeculator {
            next_token: 5,
            peaked_prob: 0.95,
        };
        let target = CannedSpeculator {
            next_token: 5,
            peaked_prob: 0.95,
        };
        let mut dec = SpecDecoder::new(draft, target, 4, 1);
        let context = vec![0u32, 1, 2];
        let out = dec.step(&context);
        // Aligned distributions → all 4 accepted, no corrected; total = 4.
        assert_eq!(
            out.len(),
            4,
            "aligned step should emit n tokens (no rejection)"
        );
        assert!(out.iter().all(|&t| t == 5));
    }
}
