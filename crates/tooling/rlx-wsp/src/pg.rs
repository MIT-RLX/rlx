// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A trained baseline: policy gradient (REINFORCE with a mean baseline and
//! normalised advantages) on the same MLP, from He-random weights, best of a
//! few seeds — the counterpart of the post's PPO baselines (simpler: no
//! clipping, no value network). Training runs on the host trainer; the
//! held-out score goes through the rlx graph under the space's protocol.

use crate::env::EnvKind;
use crate::space::{Protocol, SCORE_SEED, Tensor, WeightSpace};
use crate::train::{Sgd, Trainer};
use rlx_ir::Philox4x32;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct PgOptions {
    pub iters: usize,
    pub episodes_per_iter: usize,
    pub seeds: u64,
    pub gamma: f32,
    pub lr: f32,
    /// Episodes for each held-out check (every `eval_every` iterations).
    pub eval_episodes: usize,
    pub eval_every: usize,
}

impl Default for PgOptions {
    fn default() -> Self {
        Self {
            iters: 1000,
            episodes_per_iter: 16,
            seeds: 3,
            gamma: 0.9,
            lr: 0.05,
            eval_episodes: 128,
            eval_every: 50,
        }
    }
}

/// One policy-gradient run from He-random weights (seed `seed`): the best
/// weights seen at a held-out check, their return, and the curve.
pub fn train_policy(
    env: EnvKind,
    hidden: &[usize],
    act: crate::mlp::Act,
    protocol: Protocol,
    opts: PgOptions,
    seed: u64,
) -> Result<(BTreeMap<String, Tensor>, f32, Value), String> {
    if env.is_classifier() || env.is_reading() || env == EnvKind::Graph {
        return Err("the RL baseline is for control tasks".into());
    }
    let mut space = WeightSpace::try_new(env, hidden, act, 0)?;
    space.state.protocol = protocol;
    let widths = space.state.mlp.widths.clone();
    let spec = space.spec().clone();
    let (d, na) = (spec.obs_dim, spec.actions.len());
    let mut rng = Philox4x32::new(0x9C0_0000 + seed);
    let mut t = Trainer::random(&widths, act, &mut rng);
    let mut best = (f32::NEG_INFINITY, 0usize, t.params());
    let mut curve = Vec::new();
    let mut steps = 0usize;
    for it in 1..=opts.iters {
        let (mut xs, mut ys, mut adv) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..opts.episodes_per_iter {
            let mut e = env.make();
            e.reset(&mut rng);
            let mut obs = vec![0f32; d];
            let mut rewards = Vec::with_capacity(spec.horizon);
            for _ in 0..spec.horizon {
                e.observe(&mut obs);
                let logits = t.forward(&obs).pop().expect("logits");
                let a = Protocol::STOCHASTIC.choose(&logits[..na], None, &mut rng);
                xs.extend_from_slice(&obs);
                ys.push(a as u8);
                rewards.push(e.step(a, &mut rng));
            }
            steps += spec.horizon;
            let mut g = 0f32;
            let mut rtg = vec![0f32; rewards.len()];
            for i in (0..rewards.len()).rev() {
                g = rewards[i] + opts.gamma * g;
                rtg[i] = g;
            }
            adv.extend(rtg);
        }
        normalise(&mut adv);
        let sgd = Sgd {
            epochs: 1,
            batch: 256,
            lr: opts.lr,
            momentum: 0.0,
        };
        t.train_weighted(&xs, &ys, Some(&adv), sgd, &mut rng, |_, _| {});
        if it % opts.eval_every == 0 || it == opts.iters {
            space.set_params(t.params());
            let r = space.evaluate(opts.eval_episodes, SCORE_SEED);
            curve.push(json!({ "iter": it, "steps": steps, "held_out_return": r.mean_return }));
            if r.mean_return > best.0 {
                best = (r.mean_return, steps, t.params());
            }
        }
    }
    let report =
        json!({ "seed": seed, "best_held_out_return": best.0, "at_steps": best.1, "curve": curve });
    Ok((best.2, best.0, report))
}

pub(crate) fn normalise(adv: &mut [f32]) {
    let n = adv.len().max(1) as f32;
    let mean = adv.iter().sum::<f32>() / n;
    let std = (adv.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n)
        .sqrt()
        .max(1e-6);
    adv.iter_mut().for_each(|a| *a = (*a - mean) / std);
}

/// Train `seeds` policies on `env`; report each seed's best held-out return
/// and steps used, and the best overall.
pub fn train_baseline(
    env: EnvKind,
    hidden: &[usize],
    act: crate::mlp::Act,
    protocol: Protocol,
    opts: PgOptions,
) -> Result<Value, String> {
    let mut runs = Vec::new();
    let mut best_overall = (f32::NEG_INFINITY, 0u64);
    for seed in 0..opts.seeds {
        let (_, best, report) = train_policy(env, hidden, act, protocol, opts, seed)?;
        if best > best_overall.0 {
            best_overall = (best, seed);
        }
        runs.push(report);
    }
    Ok(json!({
        "env": env,
        "method": "REINFORCE, mean baseline, normalised advantages, sampled actions",
        "protocol": protocol.describe(),
        "iters": opts.iters,
        "episodes_per_iter": opts.episodes_per_iter,
        "best_of_seeds": best_overall.0,
        "best_seed": best_overall.1,
        "seeds": runs,
    }))
}

/// PPO (clipped surrogate, separate value network fitted to GAE returns),
/// one seed from He-random weights: best weights at a held-out check, their
/// return, and the curve. Closer to the post's PPO baselines than
/// [`train_policy`]; still small-scale (millions of steps, not 1B).
pub fn train_ppo(
    env: EnvKind,
    hidden: &[usize],
    act: crate::mlp::Act,
    protocol: Protocol,
    opts: PgOptions,
    seed: u64,
) -> Result<(BTreeMap<String, Tensor>, f32, Value), String> {
    use crate::train::Target;
    if env.is_classifier() || env.is_reading() || env == EnvKind::Graph {
        return Err("the RL baseline is for control tasks".into());
    }
    const CLIP: f32 = 0.2;
    const LAMBDA: f32 = 0.95;
    const EPOCHS: usize = 4;
    let mut space = WeightSpace::try_new(env, hidden, act, 0)?;
    space.state.protocol = protocol;
    let widths = space.state.mlp.widths.clone();
    let spec = space.spec().clone();
    let (d, na) = (spec.obs_dim, spec.actions.len());
    let mut rng = Philox4x32::new(0x990_0000 + seed);
    let mut pi = Trainer::random(&widths, act, &mut rng);
    let mut vf = Trainer::random(&[d, 64, 64, 1], crate::mlp::Act::Relu, &mut rng);
    let mut best = (f32::NEG_INFINITY, 0usize, pi.params());
    let mut curve = Vec::new();
    let mut steps = 0usize;
    let sgd = Sgd {
        epochs: 1,
        batch: 256,
        lr: opts.lr * 0.2,
        momentum: 0.9,
    };
    for it in 1..=opts.iters {
        let (mut xs, mut acts, mut logp, mut adv, mut ret) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..opts.episodes_per_iter {
            let mut e = env.make();
            e.reset(&mut rng);
            let mut obs = vec![0f32; d];
            let (mut r, mut v) = (Vec::new(), Vec::new());
            for _ in 0..spec.horizon {
                e.observe(&mut obs);
                let p = crate::train::softmax(&pi.forward(&obs).pop().expect("logits")[..na]);
                let a = Protocol::STOCHASTIC.choose(
                    &p.iter().map(|x| x.max(1e-12).ln()).collect::<Vec<_>>(),
                    None,
                    &mut rng,
                );
                v.push(vf.forward(&obs).pop().expect("value")[0]);
                xs.extend_from_slice(&obs);
                acts.push(a as u8);
                logp.push(p[a].max(1e-12).ln());
                r.push(e.step(a, &mut rng));
            }
            steps += spec.horizon;
            // GAE(λ) with the episode cut at the horizon (bootstrap 0).
            let mut gae = 0f32;
            let mut a_ep = vec![0f32; r.len()];
            for t in (0..r.len()).rev() {
                let next = if t + 1 < r.len() { v[t + 1] } else { 0.0 };
                let delta = r[t] + opts.gamma * next - v[t];
                gae = delta + opts.gamma * LAMBDA * gae;
                a_ep[t] = gae;
            }
            ret.extend(a_ep.iter().zip(&v).map(|(a, v)| a + v));
            adv.extend(a_ep);
        }
        normalise(&mut adv);
        for _ in 0..EPOCHS {
            pi.fit(
                &xs,
                Target::Ppo {
                    actions: &acts,
                    advantages: &adv,
                    old_logp: &logp,
                    clip: CLIP,
                },
                sgd,
                &mut rng,
                |_, _| {},
            );
            vf.train_regression(&xs, &ret, sgd, &mut rng);
        }
        if it % opts.eval_every == 0 || it == opts.iters {
            space.set_params(pi.params());
            let r = space.evaluate(opts.eval_episodes, SCORE_SEED);
            curve.push(json!({ "iter": it, "steps": steps, "held_out_return": r.mean_return }));
            if r.mean_return > best.0 {
                best = (r.mean_return, steps, pi.params());
            }
        }
    }
    let report =
        json!({ "seed": seed, "best_held_out_return": best.0, "at_steps": best.1, "curve": curve });
    Ok((best.2, best.0, report))
}

/// [`train_baseline`] with PPO.
pub fn train_ppo_baseline(
    env: EnvKind,
    hidden: &[usize],
    act: crate::mlp::Act,
    protocol: Protocol,
    opts: PgOptions,
) -> Result<Value, String> {
    let mut runs = Vec::new();
    let mut best_overall = (f32::NEG_INFINITY, 0u64);
    for seed in 0..opts.seeds {
        let (_, best, report) = train_ppo(env, hidden, act, protocol, opts, seed)?;
        if best > best_overall.0 {
            best_overall = (best, seed);
        }
        runs.push(report);
    }
    Ok(json!({
        "env": env,
        "method": "PPO: clipped surrogate (0.2), value network, GAE(0.95), 4 epochs per batch",
        "protocol": protocol.describe(),
        "iters": opts.iters,
        "episodes_per_iter": opts.episodes_per_iter,
        "best_of_seeds": best_overall.0,
        "best_seed": best_overall.1,
        "seeds": runs,
    }))
}
