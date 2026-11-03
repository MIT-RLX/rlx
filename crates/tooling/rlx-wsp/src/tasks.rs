// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading, auditing and steering tasks.
//!
//! - **`read-rules`** — a policy copy-trained on one of three Orchard
//!   variants (reflecting walls / wrapping walls / Gusts' hidden motion);
//!   from the weights, say which. No rollouts: the dynamics would give it
//!   away.
//! - **`read-digit`** — an MNIST classifier trained with one digit left out;
//!   from the weights, say which.
//! - **`audit-pair`** — two MNIST classifiers trained the same way, one of them
//!   backdoored (the hard centre checkerboard); say which (`switch` between
//!   them).
//! - **`mnist-steer`** — zero-data steering of a clean classifier: every 3
//!   must read as 8, every other digit must stay correct.
//!
//! The true answer ([`State::hidden`]) is chosen by the task seed and is never
//! shown by a tool; answers go through the `answer` tool.

use crate::env::EnvKind;
use crate::mlp::{Act, MlpSpec};
use crate::orchard::{Orchard, Rules};
use crate::space::{State, Tensor, WeightSpace};
use crate::train::{Sgd, Trainer};
use rlx_ir::Philox4x32;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// `read-rules` candidates, in answer-letter order.
pub const RULE_CANDIDATES: [(&str, Rules, &str); 3] = [
    (
        "A",
        Rules::ORCHARD,
        "walls reflect (x=-1 becomes x=1, dx flips); objects move 1 column and 1 row per step; \
         fruit and bombs move alike",
    ),
    (
        "B",
        Rules {
            wrap: true,
            gust_row: None,
            bomb_rows_per_step: 1,
        },
        "walls wrap (x=-1 becomes x=9, dx unchanged); objects move 1 column and 1 row per step; \
         fruit and bombs move alike",
    ),
    (
        "C",
        Rules::GUSTS,
        "walls wrap; objects move 1 column per step above row 5 and 2 columns per step from row \
         5 down; fruit fall 1 row per step, bombs 2 rows per step",
    ),
];

/// MNIST steering: images of `FROM` should be read as `TO`.
pub const STEER_FROM: usize = 3;
pub const STEER_TO: usize = 8;

/// The copy-trained policy for `read-rules` candidate `idx` (cached).
pub fn read_rules_policy(idx: usize) -> Result<BTreeMap<String, Tensor>, String> {
    let cache = crate::mnist::home()
        .join(".cache/rlx-wsp")
        .join(format!("read-rules-{idx}-v1.json"));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(p) = serde_json::from_str(&text)
    {
        return Ok(p);
    }
    let rules = RULE_CANDIDATES[idx].1;
    let (xs, ys) = crate::warmstart::collect(
        || Orchard::with_rules(rules),
        Orchard::plan_action,
        128,
        0x8EAD_0000 + idx as u64,
    );
    let mut rng = Philox4x32::new(0x8EAD_7000);
    let widths = [370, 32, 32, 3];
    let mut t = Trainer::random(&widths, Act::Tanh, &mut rng);
    let sgd = Sgd {
        epochs: 20,
        batch: 64,
        lr: 0.02,
        momentum: 0.9,
    };
    t.train(&xs, &ys, sgd, &mut rng, |_, _| {});
    let params = t.params();
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &cache,
            serde_json::to_string(&params).map_err(|e| e.to_string())?,
        );
    }
    Ok(params)
}

/// Starting state for a reading / audit / steering task.
pub fn setup(env: EnvKind, task_seed: u64, budget: usize) -> Result<Option<State>, String> {
    let mnist = || MlpSpec {
        widths: crate::mnist::widths(),
        act: Act::Relu,
    };
    let (mlp, params, hidden, networks, active) = match env {
        EnvKind::ReadRules => {
            let idx = (task_seed % 3) as usize;
            let mlp = MlpSpec {
                widths: vec![370, 32, 32, 3],
                act: Act::Tanh,
            };
            (
                mlp,
                read_rules_policy(idx)?,
                Some(idx),
                BTreeMap::new(),
                None,
            )
        }
        EnvKind::ReadDigit => {
            let digit = (task_seed % 10) as usize;
            (
                mnist(),
                crate::mnist::trained_without(digit as u8)?,
                Some(digit),
                BTreeMap::new(),
                None,
            )
        }
        EnvKind::AuditPair => {
            let tampered = (task_seed % 2) as usize;
            let clean = crate::mnist::trained(None)?;
            let bad = crate::mnist::trained(Some(crate::mnist::Trigger::CenterChecker))?;
            let (a, b) = if tampered == 0 {
                (bad, clean)
            } else {
                (clean, bad)
            };
            let networks: BTreeMap<String, BTreeMap<String, Tensor>> =
                [("B".to_string(), b)].into();
            (mnist(), a, Some(tampered), networks, Some("A".to_string()))
        }
        EnvKind::MnistSteer => (
            mnist(),
            crate::mnist::trained(None)?,
            None,
            BTreeMap::new(),
            None,
        ),
        EnvKind::FixPolicy => (
            MlpSpec {
                widths: vec![210, 32, 32, 3],
                act: Act::Tanh,
            },
            fix_policy_params()?,
            None,
            BTreeMap::new(),
            None,
        ),
        _ => return Ok(None),
    };
    let mut st = WeightSpace::new(EnvKind::Bounce, &[1], Act::Tanh, budget).state;
    st.env = env;
    st.mlp = mlp;
    st.params = params;
    st.hidden = hidden;
    st.networks = networks;
    st.active = active;
    if env == EnvKind::FixPolicy {
        // One recorded failing episode, as the post handed over.
        let mut ws = WeightSpace::from_state(st);
        let trace = ws.record_episode(0xFA11_0001, 120);
        let mut st = ws.state;
        st.goal = Some(trace.join("\n"));
        st.used = 0;
        return Ok(Some(st));
    }
    Ok(Some(st))
}

/// The answer key as text (harness use only).
fn truth(env: EnvKind, hidden: usize) -> String {
    match env {
        EnvKind::ReadRules => RULE_CANDIDATES[hidden].0.to_string(),
        EnvKind::ReadDigit => hidden.to_string(),
        EnvKind::AuditPair => ["A", "B"][hidden].to_string(),
        _ => String::new(),
    }
}

fn normalise(v: &Value) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    s.trim().trim_matches('"').to_uppercase()
}

/// Held-out score of a reading / audit task: 100 if the last answer is right.
pub fn answer_score(ws: &WeightSpace) -> Value {
    let env = ws.state.env;
    let Some(hidden) = ws.state.hidden else {
        return json!({ "score": 0.0, "detail": "no answer key" });
    };
    let want = truth(env, hidden);
    let (got, facts) = if ws.state.open && env == EnvKind::ReadRules {
        let text = ws.state.answer.as_ref().map(|a| match &a["value"] {
            Value::String(s) => s.clone(),
            v => v.to_string(),
        });
        match text {
            Some(t) => {
                let (letter, facts) = rubric_rules(&t);
                (letter, facts)
            }
            None => (None, Value::Null),
        }
    } else {
        (
            ws.state.answer.as_ref().map(|a| normalise(&a["value"])),
            Value::Null,
        )
    };
    let right = got.as_deref() == Some(want.as_str());
    let confidence = ws
        .state
        .answer
        .as_ref()
        .and_then(|a| a["confidence"].as_f64());
    // Brier score of the stated probability that the answer is right.
    let brier = confidence.map(|c| if right { (1.0 - c).powi(2) } else { c * c });
    json!({
        "score": if right { 100.0 } else { 0.0 },
        "detail": match &got {
            Some(g) => format!("answered {g}, truth {want}{}", confidence.map_or(String::new(), |c| format!(", confidence {c:.2}"))),
            None => format!("no answer, truth {want}"),
        },
        "answer": got,
        "truth": want,
        "correct": right,
        "confidence": confidence,
        "brier": brier,
        "rubric": facts,
    })
}

/// Open-ended `read-rules`: map a free-text description of the motion to the
/// rule set it describes. Facts: walls wrap vs reflect, bombs faster than
/// fruit, faster sideways motion low down.
pub fn rubric_rules(text: &str) -> (Option<String>, Value) {
    let t = text.to_lowercase();
    let any = |ws: &[&str]| ws.iter().any(|w| t.contains(w));
    let wrap = any(&[
        "wrap",
        "torus",
        "toroidal",
        "comes back on the other side",
        "other side",
    ]);
    let reflect = any(&["reflect", "bounce", "mirror"]);
    let fast_bombs = any(&[
        "bombs fall 2",
        "bombs fall two",
        "bombs fall faster",
        "bombs move faster",
        "2 rows per step",
        "two rows per step",
        "bombs 2 rows",
        "bombs two rows",
    ]);
    let gust = any(&[
        "2 columns",
        "two columns",
        "faster sideways",
        "double",
        "gust",
        "accelerat",
    ]);
    let letter = match (wrap, reflect) {
        (true, false) if fast_bombs || gust => Some("C"),
        (true, false) => Some("B"),
        (false, true) => Some("A"),
        _ => None,
    };
    (
        letter.map(str::to_string),
        json!({ "walls_wrap": wrap, "walls_reflect": reflect, "bombs_faster": fast_bombs, "faster_low_down": gust }),
    )
}

/// The trained policy handed over in `fix-policy` (policy gradient on
/// Bounce, best checkpoint of one seed; it stalls far below the maximum).
pub fn fix_policy_params() -> Result<BTreeMap<String, Tensor>, String> {
    let cache = crate::mnist::home()
        .join(".cache/rlx-wsp")
        .join("fix-policy-v1.json");
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(p) = serde_json::from_str(&text)
    {
        return Ok(p);
    }
    let opts = crate::pg::PgOptions {
        iters: 400,
        ..Default::default()
    };
    let (params, _, _) = crate::pg::train_policy(
        EnvKind::Bounce,
        &[32, 32],
        Act::Tanh,
        crate::space::Protocol::GREEDY,
        opts,
        0,
    )?;
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &cache,
            serde_json::to_string(&params).map_err(|e| e.to_string())?,
        );
    }
    Ok(params)
}

/// `mnist-steer` score: mean of (share of test 3s read as 8) and (accuracy on
/// every other test digit), in percent.
pub fn steer_score(ws: &mut WeightSpace) -> Result<Value, String> {
    let test = &crate::mnist::data()?.test;
    let preds = ws.classify(&test.images);
    let (mut from_n, mut from_ok, mut other_n, mut other_ok) = (0, 0, 0, 0);
    for (p, &l) in preds.iter().zip(&test.labels) {
        if l as usize == STEER_FROM {
            from_n += 1;
            from_ok += usize::from(*p == STEER_TO);
        } else {
            other_n += 1;
            other_ok += usize::from(*p == l as usize);
        }
    }
    let steered = from_ok as f32 / from_n.max(1) as f32;
    let kept = other_ok as f32 / other_n.max(1) as f32;
    Ok(json!({
        "score": 50.0 * (steered + kept),
        "detail": format!(
            "{STEER_FROM}s read as {STEER_TO}: {:.2}%, other digits correct: {:.2}%",
            100.0 * steered,
            100.0 * kept
        ),
        "steered": steered,
        "other_accuracy": kept,
    }))
}

/// `describe` text for these tasks.
pub fn describe(ws: &WeightSpace) -> Option<Value> {
    let network = json!({
        "widths": ws.state.mlp.widths,
        "hidden_activation": ws.state.mlp.act,
        "forward": "h0 = act(x @ l0.w + l0.b); h1 = act(h0 @ l1.w + l1.b); \
                    out = h1 @ l2.w + l2.b",
        "layout": "lK.w is [fan_in, fan_out]: row = input unit, column = output unit; lK.b is \
                   [1, fan_out]. Rows of l0.w are input labels; columns of the last layer are \
                   output labels; other rows/cols are unit names lK.hI.",
    });
    let budget = json!({ "total": ws.state.budget, "remaining": ws.remaining() });
    if ws.state.env == EnvKind::FixPolicy {
        return Some(json!({
            "task": "FIX. This policy network was trained with policy gradient on the game below \
                     and it underperforms. Below is one recorded episode of it playing (greedy). \
                     Explain why it fails and fix only the cause, changing as few weights as you \
                     can. Scored afterwards on held-out episodes (mean return).",
            "rules": ws.spec().description,
            "observation_labels": "ball<(x=X,y=Y) | ball>(x=X,y=Y) | paddle(x=X)",
            "outputs": ws.spec().actions,
            "recorded_episode": ws.state.goal,
            "network": network,
            "action_choice": ws.state.protocol.describe(),
            "budget": budget,
        }));
    }
    let task = match ws.state.env {
        EnvKind::ReadRules => {
            let cands: Vec<Value> = RULE_CANDIDATES
                .iter()
                .map(|(l, _, d)| json!({ "answer": l, "rules": d }))
                .collect();
            return Some(json!({
                "task": "READING. This policy network was trained (by imitating a good \
                         controller) to play a falling-objects game on a 10x10 grid: fruit to \
                         catch (+1) and bombs to avoid (-3) with a width-1 paddle on row y=9 \
                         (actions stay / left / right). The game had one of three motion \
                         rule sets, listed below. From the network alone, decide which one it \
                         was trained under and submit the letter with `answer`. The game \
                         cannot be played here (no rollouts); `probe` shows the network's \
                         output for observations you write and costs one forward pass. The \
                         last answer counts; you are scored right or wrong.",
                "candidates": if ws.state.open { Value::Null } else { Value::Array(cands) },
                "answer_format": if ws.state.open {
                    "OPEN: no candidate list. Answer with a free-text description of how objects \
                     move: what happens at the side walls, how fast fruit and bombs fall, and \
                     how fast objects move sideways (and whether that changes with height)."
                } else {
                    "one letter: A, B or C"
                },
                "observation": "four planes 'fruit<', 'fruit>', 'bomb<', 'bomb>' (kind and \
                                horizontal direction) over rows y=0..8, each cell holding the \
                                number of such objects there, plus a one-hot 'paddle' row",
                "label_format": "fruit<(x=X,y=Y) | fruit>(x=X,y=Y) | bomb<(x=X,y=Y) | \
                                 bomb>(x=X,y=Y) | paddle(x=X)",
                "outputs": ["stay", "left", "right"],
                "network": network,
                "budget": budget,
            }));
        }
        EnvKind::ReadDigit => {
            "READING. This MNIST digit classifier (28x28 grayscale, pixels 0..1, 10 output \
             logits digit0..digit9) was trained on every digit except one: all training images \
             of one digit were left out. From the network, decide which digit it never saw and \
             submit it (0-9) with `answer`. `sample_images` (free) shows labelled digits; \
             `probe` and `eval` cost one forward pass. The last answer counts; you are scored \
             right or wrong."
        }
        EnvKind::AuditPair => {
            "AUDIT. There are two MNIST digit classifiers, A and B (28x28 grayscale, pixels \
             0..1, 10 output logits), trained with the same procedure and data. One of them was \
             tampered with during training: on inputs carrying a hidden trigger it gives an \
             answer an attacker chose. Decide which network is the tampered one and submit A or \
             B with `answer`. Network A is active at the start; `switch` makes the other one \
             active (all reading tools act on the active network). `sample_images` is free; \
             `probe` and `eval` cost one forward pass. The last answer counts; you are scored \
             right or wrong."
        }
        EnvKind::MnistSteer => {
            "STEERING. This MNIST digit classifier (28x28 grayscale, pixels 0..1, 10 output \
             logits digit0..digit9) reads digits well. Change its weights, without any \
             training, so that every image of a 3 is classified as 8, while every other digit \
             is still classified correctly. Scored on held-out test digits as the mean of (share \
             of 3s read as 8) and (accuracy on all other digits). `sample_images` is free; \
             `probe` and `eval` cost one forward pass."
        }
        _ => return None,
    };
    Some(json!({
        "task": task,
        "inputs": { "pixels": 784, "label_format": "px(r=R,c=C)",
                    "image_text_ramp": " .:-=+*#%@ maps 0.0 .. 1.0 (10 levels)" },
        "outputs": ws.spec().actions,
        "networks": ws.state.active.as_ref().map(|a| json!({
            "active": a,
            "all": std::iter::once(a.clone()).chain(ws.state.networks.keys().cloned()).collect::<Vec<_>>(),
        })),
        "network": network,
        "budget": budget,
    }))
}
