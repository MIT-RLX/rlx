// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measured unit roles, and a check of the agent's descriptions against them.
//!
//! The post graded an agent's account of attention heads against behaviour
//! measured on probe inputs. Here: play the network on its own task, record
//! every hidden unit's activation next to interpretable features of the game
//! state (paddle column, target landing column, their difference, a bomb
//! about to land on the paddle, number of objects), and give each unit the
//! feature it correlates with most. Then, for each important unit, look at
//! what the agent's text says next to that unit's label and check it names
//! the measured role (keyword match — a heuristic).

use crate::env::{BOUNCE_H, BOUNCE_W, Bounce, Env, EnvKind};
use crate::orchard::Orchard;
use crate::space::WeightSpace;
use crate::train::Trainer;
use rlx_ir::Philox4x32;
use serde_json::{Value, json};

/// Feature names and the words that describe them.
const FEATURES: [(&str, &[&str]); 5] = [
    ("paddle_column", &["paddle", "position", "where the paddle"]),
    ("target_column", &["land", "target", "column", "where"]),
    (
        "target_minus_paddle",
        &[
            "right",
            "left",
            "toward",
            "direction",
            "minus",
            "differ",
            "compar",
        ],
    ),
    (
        "bomb_on_paddle_next",
        &["bomb", "danger", "dodge", "avoid", "flee"],
    ),
    ("objects", &["count", "number of", "how many", "objects"]),
];

fn features_bounce(e: &Bounce) -> [f32; 5] {
    let (x, y, dx, p) = e.state();
    let t = Bounce::landing_column(e.width(), e.height(), x, y, dx);
    [p as f32, t as f32, (t - p) as f32, 0.0, 1.0]
}

fn features_orchard(e: &Orchard) -> [f32; 5] {
    let p = e.paddle();
    let rules = e.rules();
    let (gw, gh) = e.size();
    let fruit = e
        .objects()
        .iter()
        .filter(|o| !o.bomb)
        .map(|o| rules.landing(gw, gh, o))
        .min_by_key(|(k, _)| *k);
    let target = fruit.map_or(p, |(_, c)| c);
    let bomb = e
        .objects()
        .iter()
        .filter(|o| o.bomb)
        .any(|o| rules.landing(gw, gh, o) == (0, p));
    [
        p as f32,
        target as f32,
        (target - p) as f32,
        f32::from(bomb),
        e.objects().len() as f32,
    ]
}

/// `(unit, best feature, r)` for every hidden unit, from `episodes` greedy
/// episodes of the network itself.
pub fn measure(
    ws: &WeightSpace,
    episodes: usize,
) -> Result<Vec<(String, &'static str, f32)>, String> {
    let env = ws.state.env;
    let net = Trainer::from_params(&ws.state.mlp.widths, ws.state.mlp.act, &ws.state.params);
    let mut rows: Vec<(Vec<Vec<f32>>, [f32; 5])> = Vec::new();
    let mut rng = Philox4x32::new(0x501E_5EED);
    let mut obs = vec![0f32; ws.spec().obs_dim];
    macro_rules! play {
        ($make:expr, $feat:expr) => {{
            for _ in 0..episodes {
                let mut e = $make;
                e.reset(&mut rng);
                for _ in 0..e.spec().horizon {
                    e.observe(&mut obs);
                    let acts = net.forward(&obs);
                    let f = $feat(&e);
                    rows.push((acts[1..acts.len() - 1].to_vec(), f));
                    let a = crate::space::argmax(acts.last().expect("logits"));
                    e.step(a, &mut rng);
                }
            }
        }};
    }
    match env {
        EnvKind::Bounce | EnvKind::FixPolicy => {
            play!(Bounce::new(BOUNCE_W, BOUNCE_H), features_bounce)
        }
        EnvKind::Orchard => play!(Orchard::new(), features_orchard),
        EnvKind::Gusts => play!(Orchard::gusts(), features_orchard),
        EnvKind::GustsHard => play!(Orchard::gusts_hard(), features_orchard),
        _ => return Err("unit roles are measured on control tasks".into()),
    }
    let mut out = Vec::new();
    let layers = ws.state.mlp.n_layers() - 1;
    for k in 0..layers {
        for i in 0..ws.state.mlp.widths[k + 1] {
            let a: Vec<f32> = rows.iter().map(|(h, _)| h[k][i]).collect();
            let mut best = ("none", 0f32);
            for (f, (name, _)) in FEATURES.iter().enumerate() {
                let x: Vec<f32> = rows.iter().map(|(_, fe)| fe[f]).collect();
                let r = pearson(&a, &x);
                if r.abs() > best.1.abs() {
                    best = (name, r);
                }
            }
            out.push((format!("l{k}.h{i}"), best.0, best.1));
        }
    }
    Ok(out)
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 1e-12 || sbb <= 1e-12 {
        0.0
    } else {
        sab / (saa * sbb).sqrt()
    }
}

/// For each unit in `important`, the measured role and whether the text
/// next to its label names it.
pub fn grade(text: &str, roles: &[(String, &'static str, f32)], important: &[String]) -> Value {
    // Text segments (sentences / lines) with the units they mention.
    let segments: Vec<(&str, std::collections::BTreeSet<(Option<usize>, usize)>)> = text
        .split(['\n', '.', ';'])
        .map(|s| (s, crate::verify::mentioned_units(s)))
        .filter(|(_, m)| !m.is_empty())
        .collect();
    let mut rows = Vec::new();
    let (mut described, mut matched) = (0, 0);
    for unit in important {
        let Some((_, feature, r)) = roles.iter().find(|(u, _, _)| u == unit) else {
            continue;
        };
        let (k, i) = unit[1..].split_once(".h").expect("unit name");
        let (k, i): (usize, usize) = (k.parse().expect("layer"), i.parse().expect("index"));
        let about: Vec<&str> = segments
            .iter()
            .filter(|(_, m)| m.contains(&(Some(k), i)) || m.contains(&(None, i)))
            .map(|(s, _)| *s)
            .collect();
        let words = FEATURES
            .iter()
            .find(|(n, _)| n == feature)
            .map_or(&[][..], |(_, w)| *w);
        let lower = about.join(" ").to_lowercase();
        let hit = !about.is_empty() && words.iter().any(|w| lower.contains(w));
        described += usize::from(!about.is_empty());
        matched += usize::from(hit);
        rows.push(json!({
            "unit": unit, "measured_role": feature, "r": r,
            "described": !about.is_empty(), "description_names_role": hit,
        }));
    }
    json!({
        "units": rows,
        "described": described,
        "described_with_measured_role": matched,
        "of": important.len(),
        "note": "role = the game feature the unit's activation correlates with most over the \
                 network's own play; a description matches when the text next to the unit's \
                 label uses a word for that feature (heuristic)",
    })
}
