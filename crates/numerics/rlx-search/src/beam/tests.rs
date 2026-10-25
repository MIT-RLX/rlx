// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use std::collections::HashMap;

/// A model whose next-token logits are a fixed pseudo-random function of
/// the whole sequence, so every search order sees the same distribution.
fn scripted(seq: &[u32], vocab: usize, seed: u64) -> Vec<f32> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for &t in seq {
        h = (h ^ t as u64).wrapping_mul(0x0100_0000_01b3);
    }
    (0..vocab)
        .map(|i| {
            let mut x = h ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^= x >> 33;
            (x % 10_000) as f32 / 1000.0
        })
        .collect()
}

fn log_softmax_vec(x: &[f32]) -> Vec<f32> {
    let mut v = x.to_vec();
    log_softmax(&mut v);
    v
}

/// The search `rlx_runtime::lm::beam_search` ran before this crate, verbatim
/// apart from the model call.
fn old_lm_beam_search(
    prompt: &[u32],
    vocab: usize,
    seed: u64,
    beams: usize,
    n_new: usize,
    stop: &[u32],
    length_penalty: f32,
) -> Vec<u32> {
    let beams = beams.max(1);
    let normalized = |score: f32, len: usize| score / (len.max(1) as f32).powf(length_penalty);
    let ctx = |toks: &[u32]| [prompt, toks].concat();
    let mut live: Vec<(Vec<u32>, f32)> = vec![(Vec::new(), 0.0)];
    let mut logits: Vec<Vec<f32>> = vec![scripted(prompt, vocab, seed)];
    let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
    for t in 0..n_new {
        let mut cands: Vec<(f32, usize, u32)> = Vec::new();
        for (i, (_, score)) in live.iter().enumerate() {
            let lp = log_softmax_vec(&logits[i]);
            let mut ids: Vec<u32> = (0..lp.len() as u32).collect();
            let take = (2 * beams).min(ids.len());
            ids.select_nth_unstable_by(take.saturating_sub(1), |&a, &b| {
                lp[b as usize].total_cmp(&lp[a as usize]).then(a.cmp(&b))
            });
            for &v in &ids[..take] {
                cands.push((score + lp[v as usize], i, v));
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut next: Vec<(Vec<u32>, f32)> = Vec::new();
        for (rank, &(score, parent, v)) in cands.iter().enumerate() {
            if stop.contains(&v) {
                if rank < beams {
                    let toks = live[parent].0.clone();
                    let len = toks.len();
                    finished.push((toks, normalized(score, len)));
                }
                continue;
            }
            let mut toks = live[parent].0.clone();
            toks.push(v);
            next.push((toks, score));
            if next.len() == beams {
                break;
            }
        }
        if finished.len() >= beams || next.is_empty() || t + 1 == n_new {
            live = next;
            break;
        }
        logits = next
            .iter()
            .map(|(t, _)| scripted(&ctx(t), vocab, seed))
            .collect();
        live = next;
    }
    finished
        .into_iter()
        .chain(live.into_iter().map(|(t, s)| {
            let len = t.len();
            (t, normalized(s, len))
        }))
        .fold(None::<(Vec<u32>, f32)>, |best, h| match best {
            Some(b) if b.1 >= h.1 => Some(b),
            _ => Some(h),
        })
        .map(|b| b.0)
        .unwrap_or_default()
}

/// `openai/whisper` `BeamSearchDecoder.update` + `finalize` +
/// `MaximumLikelihoodRanker` (patience 1, `length_penalty=None`), one audio,
/// transcribed from the Python. SuppressBlank's end-of-text mask at the first
/// position is applied to the logits, as Whisper's logit filters do.
fn openai_whisper_beam(
    prompt: &[u32],
    vocab: usize,
    seed: u64,
    beam: usize,
    sample_len: usize,
    eot: u32,
) -> Vec<u32> {
    let mut seqs: Vec<Vec<u32>> = vec![prompt.to_vec(); beam];
    let mut sum_lp: Vec<f32> = vec![0.0; beam];
    let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
    for i in 0..sample_len {
        let mut scores: Vec<(Vec<u32>, f32)> = Vec::new();
        let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
        for j in 0..seqs.len() {
            let mut logits = scripted(&seqs[j], vocab, seed);
            if i == 0 {
                logits[eot as usize] = f32::NEG_INFINITY;
            }
            let lp = log_softmax_vec(&logits);
            for (tok, l) in top_k(&lp, beam + 1) {
                let mut s = seqs[j].clone();
                s.push(tok);
                let v = sum_lp[j] + l;
                match index.get(&s) {
                    Some(&k) => scores[k].1 = v,
                    None => {
                        index.insert(s.clone(), scores.len());
                        scores.push((s, v));
                    }
                }
            }
        }
        // Python's sorted() is stable over dict insertion order.
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].1.total_cmp(&scores[a].1));
        let mut next = Vec::new();
        let mut next_lp = Vec::new();
        let mut newly: Vec<(Vec<u32>, f32)> = Vec::new();
        for k in order {
            let (s, v) = &scores[k];
            if *s.last().unwrap() == eot {
                newly.push((s.clone(), *v));
            } else {
                next.push(s.clone());
                next_lp.push(*v);
                if next.len() == beam {
                    break;
                }
            }
        }
        newly.sort_by(|a, b| b.1.total_cmp(&a.1));
        for f in newly {
            if finished.len() >= beam {
                break;
            }
            finished.push(f);
        }
        seqs = next;
        sum_lp = next_lp;
        if finished.len() >= beam {
            break;
        }
    }
    if finished.len() < beam {
        let mut order: Vec<usize> = (0..seqs.len()).collect();
        order.sort_by(|&a, &b| sum_lp[b].total_cmp(&sum_lp[a]));
        for j in order {
            let mut s = seqs[j].clone();
            s.push(eot);
            finished.push((s, sum_lp[j]));
            if finished.len() >= beam {
                break;
            }
        }
    }
    let generated = |s: &[u32]| s[prompt.len()..s.len() - 1].to_vec();
    let mut best: Option<(Vec<u32>, f32)> = None;
    for (s, v) in finished {
        let g = generated(&s);
        let r = v / g.len() as f32;
        if best.as_ref().is_none_or(|b| r > b.1) {
            best = Some((g, r));
        }
    }
    best.map(|b| b.0).unwrap_or_default()
}

fn run_engine(cfg: BeamConfig, prompt: &[u32], vocab: usize, seed: u64) -> Vec<Hypothesis> {
    let first = scripted(prompt, vocab, seed);
    beam_search::<()>(cfg, prompt, &first, &mut (), |s| {
        Ok((0..s.len())
            .map(|i| scripted(s.sequence(i), vocab, seed))
            .collect())
    })
    .unwrap()
}

#[test]
fn reproduces_the_previous_lm_beam_search() {
    let prompt = [3u32, 1, 4];
    for seed in 0..200u64 {
        for (beams, vocab, n_new, lp) in [(1, 6, 5, 1.0f32), (3, 8, 6, 1.0), (4, 5, 7, 0.6)] {
            let stop = [0u32, 2];
            let want = old_lm_beam_search(&prompt, vocab, seed, beams, n_new, &stop, lp);
            let cfg = BeamConfig::new(beams, n_new)
                .stop(stop)
                .length_penalty(lp as f64);
            let got = run_engine(cfg, &prompt, vocab, seed);
            assert_eq!(got[0].tokens, want, "seed {seed} beams {beams}");
        }
    }
}

#[test]
fn reproduces_openai_whisper_beam_search() {
    let prompt = [7u32, 7];
    for seed in 0..200u64 {
        for (beam, vocab, len) in [(2, 6, 6), (5, 9, 8), (3, 4, 10)] {
            let eot = 1;
            let want = openai_whisper_beam(&prompt, vocab, seed, beam, len, eot);
            let got = run_engine(BeamConfig::whisper(beam, len, eot), &prompt, vocab, seed);
            assert_eq!(got[0].tokens, want, "seed {seed} beam {beam}");
        }
    }
}

#[test]
fn scores_only_the_surviving_hypotheses() {
    let mut rows_scored = 0;
    let beams = 4;
    let cfg = BeamConfig::new(beams, 6).stop([0u32]);
    let first = scripted(&[], 10, 1);
    beam_search::<()>(cfg, &[], &first, &mut (), |s| {
        assert!(s.len() <= beams);
        rows_scored += s.len();
        Ok((0..s.len())
            .map(|i| scripted(s.sequence(i), 10, 1))
            .collect())
    })
    .unwrap();
    assert!(
        rows_scored <= beams * 5,
        "{rows_scored} rows for 5 steps of {beams} beams"
    );
}

#[test]
fn step_parents_and_tokens_rebuild_each_sequence() {
    let mut bs = BeamSearch::new(BeamConfig::new(3, 4), &[9]);
    let mut rows = vec![scripted(&[9], 7, 5)];
    let mut prev: Vec<Vec<u32>> = vec![vec![9]];
    while !bs.is_done() {
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let Some(step) = bs.advance(&refs, &mut ()) else {
            break;
        };
        let cur: Vec<Vec<u32>> = (0..step.len()).map(|i| step.sequence(i).to_vec()).collect();
        for i in 0..step.len() {
            let mut want = prev[step.parents[i]].clone();
            want.push(step.tokens[i]);
            assert_eq!(cur[i], want);
            assert_eq!(step.generated(i), &cur[i][1..]);
        }
        rows = cur.iter().map(|s| scripted(s, 7, 5)).collect();
        prev = cur;
    }
}

#[test]
fn masks_suppress_min_new_and_ngram_repeats() {
    // Token 1 always dominates; stop is token 0.
    let peaked = |seq: &[u32]| -> Vec<f32> {
        let mut v = vec![0.0f32; 5];
        v[1] = 10.0;
        v[0] = if seq.len() >= 2 { 20.0 } else { 5.0 };
        v
    };
    let go = |cfg: BeamConfig| {
        beam_search::<()>(cfg, &[], &peaked(&[]), &mut (), |s| {
            Ok((0..s.len()).map(|i| peaked(s.sequence(i))).collect())
        })
        .unwrap()
        .remove(0)
    };
    let base = BeamConfig::new(1, 6).stop([0u32]);
    assert_eq!(go(base.clone()).tokens, vec![1, 1]);

    let mut cfg = base.clone();
    cfg.suppress = vec![1];
    assert!(!go(cfg).tokens.contains(&1));

    let mut cfg = base.clone();
    cfg.no_repeat_ngram = 1;
    let t = go(cfg).tokens;
    let mut uniq = t.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        t.len(),
        "1-gram block: no token twice, got {t:?}"
    );

    let mut cfg = BeamConfig::new(1, 6).stop([1u32]);
    cfg.min_new = 2;
    let h = go(cfg);
    assert!(
        h.tokens.len() >= 2,
        "stop masked for two steps: {:?}",
        h.tokens
    );
}

#[test]
fn constraint_masks_and_vetoes() {
    struct EvenOnly;
    impl Constraint for EvenOnly {
        fn mask(&mut self, _: &[u32], _: usize, s: &mut [f32]) {
            for (i, v) in s.iter_mut().enumerate() {
                if i % 2 == 1 {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        fn reject(&mut self, seq: &[u32], tok: u32) -> bool {
            seq.last() == Some(&tok)
        }
    }
    let cfg = BeamConfig::new(2, 6);
    let first = scripted(&[], 9, 3);
    let out = beam_search::<()>(cfg, &[], &first, &mut EvenOnly, |s| {
        Ok((0..s.len())
            .map(|i| scripted(s.sequence(i), 9, 3))
            .collect())
    })
    .unwrap();
    let t = &out[0].tokens;
    assert_eq!(t.len(), 6);
    assert!(t.iter().all(|x| x % 2 == 0), "{t:?}");
    assert!(t.windows(2).all(|w| w[0] != w[1]), "{t:?}");
}

#[test]
fn log_prob_rows_are_not_renormalized() {
    // Rows that sum to less than one: renormalizing would change scores.
    let row = vec![(0.2f32).ln(), (0.1f32).ln(), (0.05f32).ln()];
    let mut cfg = BeamConfig::new(1, 1);
    cfg.scores = Scores::LogProbs;
    cfg.length_penalty = 0.0;
    let h = beam_search::<()>(cfg, &[], &row, &mut (), |_| Ok(vec![])).unwrap();
    assert_eq!(h[0].tokens, vec![0]);
    assert!((h[0].score - (0.2f64).ln()).abs() < 1e-6);
}

#[test]
fn nbest_and_unbeatable_pool() {
    // Stop is cheap right away; a live path that keeps losing mass can never
    // catch the worst finished one, so the search ends early.
    let model = |seq: &[u32]| -> Vec<f32> {
        if seq.is_empty() {
            vec![(0.5f32).ln(), (0.3f32).ln(), (0.2f32).ln()]
        } else {
            vec![(0.9f32).ln(), (0.05f32).ln(), (0.05f32).ln()]
        }
    };
    let mut cfg = BeamConfig::new(2, 50).stop([0u32]);
    cfg.scores = Scores::LogProbs;
    cfg.finish = FinishRule::Any;
    cfg.pool = 2;
    cfg.stop_when_unbeatable = true;
    cfg.nbest = 2;
    let mut calls = 0;
    let out = beam_search::<()>(cfg, &[], &model(&[]), &mut (), |s| {
        calls += 1;
        Ok((0..s.len()).map(|i| model(s.sequence(i))).collect())
    })
    .unwrap();
    assert_eq!(out.len(), 2);
    assert!(out[0].normalized >= out[1].normalized);
    assert!(out.iter().all(|h| h.finished));
    assert!(calls < 5, "ended early, {calls} steps");
}

#[test]
fn relative_pruning_drops_far_behind_branches() {
    let model = |seq: &[u32]| -> Vec<f32> {
        if seq.is_empty() {
            vec![-9.0, -0.1, -5.0]
        } else {
            vec![-0.1, -9.0, -9.0]
        }
    };
    let mut cfg = BeamConfig::new(2, 3).stop([0u32]);
    cfg.scores = Scores::LogProbs;
    cfg.finish = FinishRule::Any;
    cfg.length_penalty = 0.0;
    cfg.nbest = 2;
    let base = beam_search::<()>(cfg.clone(), &[], &model(&[]), &mut (), |s| {
        Ok((0..s.len()).map(|i| model(s.sequence(i))).collect())
    })
    .unwrap();
    assert!(base.iter().any(|h| h.tokens == [2]), "{base:?}");
    cfg.prune_relative = Some(0.5);
    let pruned = beam_search::<()>(cfg, &[], &model(&[]), &mut (), |s| {
        Ok((0..s.len()).map(|i| model(s.sequence(i))).collect())
    })
    .unwrap();
    assert!(pruned.iter().all(|h| h.tokens != [2]), "{pruned:?}");
}

#[test]
fn zero_budget_returns_the_empty_hypothesis() {
    let out = beam_search::<()>(BeamConfig::new(2, 0), &[5], &[0.0, 1.0], &mut (), |_| {
        unreachable!("no step without a budget")
    })
    .unwrap();
    assert_eq!(out[0].tokens, Vec::<u32>::new());
    assert!(!out[0].finished);
}

#[test]
fn nothing_proposable_keeps_the_current_beam() {
    struct VetoAll;
    impl Constraint for VetoAll {
        fn reject(&mut self, seq: &[u32], _: u32) -> bool {
            seq.len() >= 2
        }
    }
    let first = scripted(&[], 5, 2);
    let out = beam_search::<()>(BeamConfig::new(2, 9), &[], &first, &mut VetoAll, |s| {
        Ok((0..s.len())
            .map(|i| scripted(s.sequence(i), 5, 2))
            .collect())
    })
    .unwrap();
    assert_eq!(
        out[0].tokens.len(),
        2,
        "cut off at the vetoed step: {out:?}"
    );
    assert!(!out[0].finished);
}
