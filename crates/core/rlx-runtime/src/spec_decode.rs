// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// Commit `accepted` tokens into persistent decode state after a
    /// speculative round. Default no-op; MTP draft overrides so GDN
    /// recurrent state only advances for accepted tokens.
    fn commit(&mut self, context: &[u32], accepted: &[u32]) {
        let _ = (context, accepted);
    }
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

/// A distribution supported on a small candidate set, `ids[j] ->
/// probs[j]`. Everything outside `ids` has probability zero.
///
/// Block drafters (DFlash2's candidate selector, Medusa-style heads)
/// only ever expose a top-k slate, and a target sampler with top-k /
/// top-p applied is likewise sparse. Carrying `[n, vocab]` dense rows
/// for those is ~150k floats per position of pure waste, so the
/// sparse path exists alongside [`speculative_accept`] rather than
/// forcing callers to densify.
///
/// Duplicate ids are summed on lookup, so a caller may push the same
/// id twice without corrupting the mass.
#[derive(Debug, Clone, Default)]
pub struct SparseDist {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

impl SparseDist {
    /// Probability this distribution assigns to `id` (0.0 if absent).
    pub fn prob_of(&self, id: u32) -> f32 {
        self.ids
            .iter()
            .zip(&self.probs)
            .filter(|&(&i, _)| i == id)
            .map(|(_, &p)| p)
            .sum()
    }
}

/// Sparse speculative acceptance — maximal coupling against a target
/// distribution that is itself only known on a candidate set.
///
/// Same accept/reject law as [`speculative_accept`]: keep `draft[i]`
/// with probability `min(1, p_target / q_draft)`, otherwise stop and
/// emit one token from the residual `norm(max(0, p - q))`. Restricting
/// the residual to the target's candidate set is what keeps the
/// distribution exact *for that sampler*: a token the target's top-k
/// already excluded must not reappear through the residual.
///
/// **Contract differs from [`speculative_accept`] on purpose.**
/// `target` carries `draft.len() + 1` rows — one per drafted position
/// plus the bonus slot the target verified for free — and `corrected`
/// is therefore always `Some`. A round always yields
/// `accepted.len() + 1` real tokens, so a fully-rejected block still
/// advances by one. The dense function leaves that bonus to the
/// caller; here it would just force every caller to re-derive it.
///
/// # Panics
/// If `dists.len() != draft.len()` or `target.len() != draft.len() + 1`.
pub fn speculative_accept_sparse(
    draft: &[u32],
    dists: &[SparseDist],
    target: &[SparseDist],
    rng: &mut Philox4x32,
) -> AcceptDecision {
    assert_eq!(
        draft.len(),
        dists.len(),
        "speculative_accept_sparse: one proposal distribution per drafted token"
    );
    assert_eq!(
        target.len(),
        draft.len() + 1,
        "speculative_accept_sparse: target must cover every drafted position plus the bonus slot"
    );

    let mut accepted: Vec<u32> = Vec::with_capacity(draft.len());
    for i in 0..draft.len() {
        let token = draft[i];
        let q = dists[i].prob_of(token);
        let p = target[i].prob_of(token);

        // `u * q <= p` is `u <= p/q` without dividing by a q that the
        // drafter may have left at zero.
        if q > 0.0 && rng.next_f32() * q <= p {
            accepted.push(token);
            continue;
        }

        let corrected = sample_residual_sparse(&dists[i], &target[i], rng);
        return AcceptDecision {
            accepted,
            corrected: Some(corrected),
        };
    }

    // Whole block accepted: the bonus slot is a plain target sample.
    let bonus = &target[draft.len()];
    let corrected = sample_sparse(bonus, rng);
    AcceptDecision {
        accepted,
        corrected: Some(corrected),
    }
}

/// Sample `norm(max(0, p_target - q_draft))` over the target's support.
fn sample_residual_sparse(draft: &SparseDist, target: &SparseDist, rng: &mut Philox4x32) -> u32 {
    let residual: Vec<f32> = target
        .ids
        .iter()
        .zip(&target.probs)
        .map(|(&id, &p)| (p - draft.prob_of(id)).max(0.0))
        .collect();
    let sum: f32 = residual.iter().sum();
    if sum <= f32::MIN_POSITIVE {
        // Target mass is fully covered by the draft (possible when the
        // two slates coincide and rounding eats the difference): fall
        // back to the target itself rather than emitting nothing.
        return sample_sparse(target, rng);
    }
    let inv = 1.0 / sum;
    let idx = sample_from(&residual.iter().map(|v| v * inv).collect::<Vec<_>>(), rng) as usize;
    target.ids[idx]
}

fn sample_sparse(dist: &SparseDist, rng: &mut Philox4x32) -> u32 {
    assert!(
        !dist.ids.is_empty(),
        "speculative_accept_sparse: empty target candidate set"
    );
    let sum: f32 = dist.probs.iter().sum();
    let idx = if sum > f32::MIN_POSITIVE {
        let inv = 1.0 / sum;
        sample_from(&dist.probs.iter().map(|v| v * inv).collect::<Vec<_>>(), rng) as usize
    } else {
        0
    };
    dist.ids[idx]
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
        self.draft.commit(context, &out);
        self.target.commit(context, &out);
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

    fn sparse(ids: &[u32], probs: &[f32]) -> SparseDist {
        SparseDist {
            ids: ids.to_vec(),
            probs: probs.to_vec(),
        }
    }

    /// Matching slates → every draft token survives, plus the bonus.
    #[test]
    fn sparse_identical_slates_accept_all_plus_bonus() {
        let slate = sparse(&[7, 8, 9], &[0.8, 0.15, 0.05]);
        let draft = vec![7u32, 7, 7];
        let dists = vec![slate.clone(); 3];
        let target = vec![slate.clone(); 4];

        for seed in 0..100u64 {
            let mut rng = Philox4x32::new(seed + 1);
            let d = speculative_accept_sparse(&draft, &dists, &target, &mut rng);
            assert_eq!(d.accepted, draft, "seed {seed}");
            // Contract: always n+1 tokens, unlike the dense path.
            assert_eq!(d.total_tokens(), 4);
            assert!(target[3].ids.contains(&d.corrected.unwrap()));
        }
    }

    /// A rejection may only emit tokens the TARGET still allows —
    /// letting the residual reach outside the target's slate would
    /// resurrect tokens its top-k already discarded.
    #[test]
    fn sparse_correction_stays_inside_target_support() {
        // Draft is certain of 100; target never proposes it at all.
        let draft = vec![100u32];
        let dists = vec![sparse(&[100], &[1.0])];
        let target = vec![sparse(&[1, 2], &[0.5, 0.5]), sparse(&[1, 2], &[0.5, 0.5])];

        for seed in 0..200u64 {
            let mut rng = Philox4x32::new(seed + 1);
            let d = speculative_accept_sparse(&draft, &dists, &target, &mut rng);
            // p_target(100) = 0 → the accept test can never pass.
            assert!(d.accepted.is_empty(), "seed {seed}: 100 must be rejected");
            let c = d.corrected.unwrap();
            assert!(c == 1 || c == 2, "seed {seed}: leaked token {c}");
        }
    }

    /// The property that makes speculation *free*: the first emitted
    /// token is distributed exactly as the target would have sampled
    /// it, however wrong the drafter is.
    #[test]
    fn sparse_first_token_matches_target_distribution() {
        let target_row = sparse(&[10, 11, 12], &[0.6, 0.3, 0.1]);
        // Drafter is badly miscalibrated and always proposes 12.
        let draft = vec![12u32];
        let dists = vec![sparse(&[10, 11, 12], &[0.05, 0.05, 0.9])];
        let target = vec![target_row.clone(), target_row.clone()];

        let trials = 40_000;
        let mut counts = [0usize; 3];
        for seed in 0..trials {
            let mut rng = Philox4x32::new(seed as u64 + 1);
            let d = speculative_accept_sparse(&draft, &dists, &target, &mut rng);
            let first = *d.accepted.first().unwrap_or(&d.corrected.unwrap());
            counts[(first - 10) as usize] += 1;
        }
        for (i, &want) in target_row.probs.iter().enumerate() {
            let got = counts[i] as f32 / trials as f32;
            assert!(
                (got - want).abs() < 0.02,
                "token {}: target p={want}, sampled {got} over {trials} trials",
                10 + i
            );
        }
    }

    /// Mock speculators for end-to-end SpecDecoder basic test.
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
