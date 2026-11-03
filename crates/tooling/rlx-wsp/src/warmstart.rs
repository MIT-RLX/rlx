// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Are agent-written weights a good place to *start* ordinary training?
//!
//! Behaviour cloning of the environment's reference controller (Bounce:
//! steer to the landing column; Orchard / Gusts: the lookahead planner under
//! the true rules), trained with plain SGD from two starts — the agent's
//! weights and He-random weights — on the same data, same order, same
//! hyper-parameters. After every epoch: agreement with the teacher on a
//! separate set, and the held-out return through the rlx graph.

use crate::env::{Bounce, Env, EnvKind};
use crate::orchard::Orchard;
use crate::space::{SCORE_SEED, WeightSpace};
use crate::train::{Sgd, Trainer};
use rlx_ir::Philox4x32;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy)]
pub struct WarmstartOptions {
    /// Teacher episodes for training data (another quarter as held-out
    /// agreement data).
    pub teacher_episodes: usize,
    pub epochs: usize,
    /// Episodes per held-out return measurement.
    pub eval_episodes: usize,
    pub sgd: Sgd,
}

impl Default for WarmstartOptions {
    fn default() -> Self {
        Self {
            teacher_episodes: 64,
            epochs: 6,
            eval_episodes: 128,
            sgd: Sgd {
                epochs: 1,
                batch: 64,
                lr: 0.02,
                momentum: 0.9,
            },
        }
    }
}

/// `(observations, teacher actions)` from `episodes` teacher-driven episodes.
pub(crate) fn collect<E: Env>(
    make: impl Fn() -> E,
    teacher: impl Fn(&E) -> usize,
    episodes: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u8>) {
    let spec = make().spec();
    let mut rng = Philox4x32::new(seed);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let mut obs = vec![0f32; spec.obs_dim];
    for _ in 0..episodes {
        let mut e = make();
        e.reset(&mut rng);
        for _ in 0..spec.horizon {
            e.observe(&mut obs);
            let a = teacher(&e);
            xs.extend_from_slice(&obs);
            ys.push(a as u8);
            e.step(a, &mut rng);
        }
    }
    (xs, ys)
}

fn bounce_teacher(e: &Bounce) -> usize {
    let (x, y, dx, p) = e.state();
    let target = Bounce::landing_column(e.width(), e.height(), x, y, dx);
    match target.cmp(&p) {
        std::cmp::Ordering::Greater => 2,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
    }
}

fn teacher_data(env: EnvKind, episodes: usize, seed: u64) -> Result<(Vec<f32>, Vec<u8>), String> {
    use crate::env::{BOUNCE_H, BOUNCE_W};
    Ok(match env {
        EnvKind::Bounce | EnvKind::FixPolicy => collect(
            || Bounce::new(BOUNCE_W, BOUNCE_H),
            bounce_teacher,
            episodes,
            seed,
        ),
        EnvKind::Orchard => collect(Orchard::new, Orchard::plan_action, episodes, seed),
        EnvKind::Gusts => collect(Orchard::gusts, Orchard::plan_action, episodes, seed),
        EnvKind::GustsHard => collect(Orchard::gusts_hard, Orchard::plan_action, episodes, seed),
        EnvKind::MnistFix
        | EnvKind::MnistFixHard
        | EnvKind::ReadRules
        | EnvKind::ReadDigit
        | EnvKind::AuditPair
        | EnvKind::MnistSteer
        | EnvKind::Graph => {
            return Err("warmstart is for control tasks with a reference controller".into());
        }
    })
}

/// Learning curves from the agent's weights and from random weights.
pub fn warmstart(ws: &WeightSpace, opts: WarmstartOptions) -> Result<Value, String> {
    let env = ws.state.env;
    let (xs, ys) = teacher_data(env, opts.teacher_episodes, 0xB0C_0001)?;
    let (hx, hy) = teacher_data(env, opts.teacher_episodes.div_ceil(4), 0xB0C_0002)?;
    let widths = ws.state.mlp.widths.clone();
    let act = ws.state.mlp.act;
    let mut curves = serde_json::Map::new();
    for start in ["agent", "random"] {
        let mut rng = Philox4x32::new(0xB0C_00A5);
        let mut init_rng = Philox4x32::new(0xB0C_0175);
        let mut t = match start {
            "agent" => Trainer::from_params(&widths, act, &ws.state.params),
            _ => Trainer::random(&widths, act, &mut init_rng),
        };
        let mut eval = WeightSpace::from_state(ws.state.clone());
        let mut points = Vec::new();
        for epoch in 0..=opts.epochs {
            if epoch > 0 {
                t.train(&xs, &ys, opts.sgd, &mut rng, |_, _| {});
            }
            eval.set_params(t.params());
            let r = eval.evaluate(opts.eval_episodes, SCORE_SEED);
            points.push(json!({
                "epoch": epoch,
                "teacher_agreement": t.accuracy(&hx, &hy),
                "held_out_return": r.mean_return,
            }));
        }
        curves.insert(start.to_string(), Value::Array(points));
    }
    let teacher = match env {
        EnvKind::Bounce => "steer to the landing column",
        _ => "lookahead planner (true rules)",
    };
    Ok(json!({
        "env": env,
        "teacher": teacher,
        "train_samples": ys.len(),
        "agreement_samples": hy.len(),
        "sgd": { "batch": opts.sgd.batch, "lr": opts.sgd.lr, "momentum": opts.sgd.momentum },
        "curves": curves,
    }))
}
