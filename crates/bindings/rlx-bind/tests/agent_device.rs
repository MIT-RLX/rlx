// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent optimizer on a resident trainer: its programs as device plans
//! in the fused step. Needs a GPU backend in the build (`--features
//! rlx-runtime/metal`); skipped otherwise.

#![cfg(feature = "agent")]

use rlx_bind::runtime::agent::{
    AgentConfig, AgentOptimizer, Backend, DecisionOutcome, DecisionRequest, Keep, RuleHost, Spread,
    Waiting,
};
use rlx_bind::runtime::train::{Resident, Trainer, TrainerConfig, TrainingOptions};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

const DIM: usize = 3;

fn regression() -> Graph {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("mse");
    let x = g.input("x", f32s(&[4, DIM]));
    let t = g.input("t", f32s(&[4, DIM]));
    let w = g.param("w", f32s(&[DIM, DIM]));
    let b = g.param("b", f32s(&[1, DIM]));
    let pred = g.mm(x, w);
    let biased = g.add(pred, b);
    let diff = g.sub(biased, t);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    g
}

fn trainer(resident: bool) -> Option<Trainer> {
    let w: Vec<f32> = (0..DIM * DIM).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
    Trainer::new(
        regression(),
        TrainerConfig {
            wrt: vec!["w".into(), "b".into()],
            init: vec![("w".into(), w), ("b".into(), vec![0.0; DIM])],
            device: "metal".into(),
            optimizer: json!({"kind": "adamw", "lr": 0.05, "weight_decay": 0.01}),
            policy: None,
            clip_norm: None,
            resident: if resident {
                Resident::On
            } else {
                Resident::Off
            },
            training: TrainingOptions::default(),
        },
    )
    .ok()
}

fn batch(i: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..4 * DIM)
        .map(|j| (((i * 7 + j * 3) % 11) as f32 - 5.0) / 5.0)
        .collect();
    let t = (0..4 * DIM)
        .map(|j| (((i * 5 + j) % 7) as f32 - 3.0) / 3.0)
        .collect();
    (x, t)
}

fn config(root: &std::path::Path, decisions: usize) -> AgentConfig {
    let mut c = AgentConfig::new("regression; judged by the final loss", 40);
    c.run_root = root.to_path_buf();
    c.decisions = Some(decisions);
    c.spread = Spread::Uniform;
    c.every = 5;
    c.trial_steps = 6;
    c.waiting = Waiting::Always;
    c
}

/// Writes `updater.json` with the next of its texts at each decision.
struct Writes(Vec<&'static str>);

impl Backend for Writes {
    fn name(&self) -> &str {
        "writes"
    }
    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        if !self.0.is_empty() {
            std::fs::write(req.workspace.join("updater.json"), self.0.remove(0)).unwrap();
        }
        DecisionOutcome::default()
    }
}

fn host() -> Box<RuleHost> {
    Box::new(RuleHost::new(&json!({"kind": "adamw", "lr": 0.05, "weight_decay": 0.01})).unwrap())
}

fn run(agent: &mut AgentOptimizer, t: &mut Trainer) -> Vec<f32> {
    (0..40)
        .map(|i| {
            let (x, y) = batch(i);
            agent
                .train_step(t, &[("x", x.as_slice()), ("t", y.as_slice())])
                .unwrap()
        })
        .collect()
}

fn events(agent: &AgentOptimizer, kind: &str) -> Vec<Value> {
    std::fs::read_to_string(agent.run_dir().join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter(|e| e["kind"] == kind)
        .collect()
}

#[test]
fn programs_run_on_the_device_as_plans_in_the_fused_step() {
    let Some(mut plain) = trainer(true) else {
        eprintln!("skipped: no metal backend in this build");
        return;
    };
    let dir = tempfile::tempdir().unwrap();

    // No agent: the plain resident run.
    let plain_losses: Vec<f32> = (0..40)
        .map(|i| {
            let (x, y) = batch(i);
            plain
                .step(&[("x", x.as_slice()), ("t", y.as_slice())])
                .unwrap()
        })
        .collect();
    let mut t = trainer(true).unwrap();
    let mut agent = AgentOptimizer::new(config(dir.path(), 2), host(), Box::new(Keep)).unwrap();
    let kept = run(&mut agent, &mut t);
    for (a, b) in kept.iter().zip(&plain_losses) {
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "{kept:?}\nvs {plain_losses:?}"
        );
    }

    // Muon on the matrix, AdamW on the rest: built again, trained on.
    let groups = r#"{"groups": [{"names": ["w"], "optimizer": {"kind": "muon", "lr": 0.02}},
                                {"names": ["*"], "optimizer": {"kind": "adamw", "lr": 0.05}}]}"#;
    // Then one that cannot run there (skip_if), and one that diverges.
    let skips = r#"{"groups": [{"names": ["*"], "optimizer": {"kind": "adamw", "lr": 0.05}}],
                    "skip_if": {"loss_above": 100}}"#;
    let mut t = trainer(true).unwrap();
    let mut agent = AgentOptimizer::new(
        config(dir.path(), 2),
        host(),
        Box::new(Writes(vec![groups, skips])),
    )
    .unwrap();
    let losses = run(&mut agent, &mut t);
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    assert!(losses[39] < losses[0], "{losses:?}");
    assert_eq!(events(&agent, "applied").len(), 1);
    assert_eq!(events(&agent, "trial_passed").len(), 1);
    let rejected = events(&agent, "rejected");
    assert_eq!(rejected.len(), 1);
    assert!(
        rejected[0]["error"]
            .as_str()
            .unwrap()
            .contains("trains on the device"),
        "{rejected:?}"
    );

    let diverges = r#"{"groups": [{"names": ["*"], "optimizer": {"kind": "adamw", "lr": 1e30}}]}"#;
    let mut t = trainer(true).unwrap();
    let mut agent = AgentOptimizer::new(
        config(dir.path(), 1),
        host(),
        Box::new(Writes(vec![diverges])),
    )
    .unwrap();
    let losses = run(&mut agent, &mut t);
    let reverted = events(&agent, "reverted");
    assert_eq!(reverted.len(), 1, "{losses:?}");
    assert_eq!(reverted[0]["exact"], true);
    assert!(
        losses[39].is_finite(),
        "back on the previous program: {losses:?}"
    );
    assert!(
        t.params()
            .iter()
            .all(|(_, v)| v.iter().all(|x| x.is_finite()))
    );
}

/// Writes `analysis.json` at the first decision.
struct Analyses(&'static str);

impl Backend for Analyses {
    fn name(&self) -> &str {
        "analyses"
    }
    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        std::fs::write(req.workspace.join("analysis.json"), self.0).unwrap();
        DecisionOutcome::default()
    }
}

fn reports(agent: &AgentOptimizer) -> Vec<Value> {
    std::fs::read_to_string(agent.run_dir().join("reports.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap()["report"].clone())
        .collect()
}

#[test]
fn reports_on_the_device_have_the_hosts_statistics() {
    let Some(mut t) = trainer(true) else {
        eprintln!("skipped: no metal backend in this build");
        return;
    };
    let analysis = r#"{"every": 5, "stats": ["grad_norm", "weight_norm", "update_ratio", "grad_cosine"], "params": ["*"]}"#;
    let dir = tempfile::tempdir().unwrap();
    let mut device =
        AgentOptimizer::new(config(dir.path(), 1), host(), Box::new(Analyses(analysis))).unwrap();
    run(&mut device, &mut t);
    let mut h = trainer(false).unwrap();
    let mut on_host =
        AgentOptimizer::new(config(dir.path(), 1), host(), Box::new(Analyses(analysis))).unwrap();
    run(&mut on_host, &mut h);

    let (d, h) = (reports(&device), reports(&on_host));
    assert_eq!(d.len(), h.len());
    let close = |a: &Value, b: &Value, what: &str| {
        let (Some(a), Some(b)) = (a.as_f64(), b.as_f64()) else {
            panic!("{what}: {a} vs {b}");
        };
        assert!(
            (a - b).abs() <= 1e-3 * b.abs().max(1e-3),
            "{what}: {a} vs {b}"
        );
    };
    // The update whose decision asked for update_ratio: on the device the
    // weights before it were read (or not) before the decision.
    let asked = h
        .iter()
        .position(|r| r.get("update_ratio").is_some())
        .unwrap();
    for (i, (d, h)) in d.iter().zip(&h).enumerate() {
        close(
            &d["grad_norm_total"],
            &h["grad_norm_total"],
            "grad_norm_total",
        );
        for stat in ["grad_norm", "weight_norm", "update_ratio"] {
            if h.get(stat).is_none() || (stat == "update_ratio" && i == asked) {
                continue;
            }
            for p in ["w", "b"] {
                close(
                    &d[stat][p],
                    &h[stat][p],
                    &format!("{stat} of {p} at {}", d["update"]),
                );
            }
        }
        // The fused step reads no gradient back, so no cosine.
        assert!(d.get("grad_cosine").is_none(), "{d}");
    }
    assert!(asked + 1 < d.len(), "a report after the decision");
}
