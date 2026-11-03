// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `runtime::agent`: the protocol around the agent, with scripted backends
//! standing in for it.
//!
//! * With no agent (`Keep`) a run through the `Trainer` is bit for bit the
//!   plain run, learning-rate schedule included.
//! * An edited program takes over at the decision's update; one that fails to
//!   build is rejected and the run is unchanged.
//! * A program failing at its first update is undone exactly; later in its
//!   trial it is reverted without an undo; after its trial the previous
//!   program is rebuilt; with nothing to go back to the step errors.
//! * A non-finite loss pauses a non-blocking run for a decision; the cost cap
//!   stops the asking.
#![cfg(feature = "agent")]

use std::path::Path;
use std::sync::{Arc, Mutex};

use rlx_bind::runtime::agent::{
    AgentConfig, AgentOptimizer, Backend, DecisionOutcome, DecisionRequest, Files, Keep, ParamInfo,
    Program, ProgramHost, RuleHost, Spread, StepCtx, Waiting,
};
use rlx_bind::runtime::train::{OptItem, Resident, StepInfo, Trainer, TrainerConfig};
use rlx_bind::{ApiError, ApiResult};
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

// ── a scripted agent ────────────────────────────────────────────

type Act = Box<dyn FnMut(&DecisionRequest) -> DecisionOutcome + Send>;

/// Answers decision k with `acts[k-1]` (later ones keep), logging every message.
struct Scripted {
    acts: Vec<Act>,
    messages: Arc<Mutex<Vec<String>>>,
}

impl Backend for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }

    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        self.messages.lock().unwrap().push(req.message.clone());
        if self.acts.is_empty() {
            return DecisionOutcome::default();
        }
        (self.acts.remove(0))(req)
    }
}

fn scripted(acts: Vec<Act>) -> (Box<Scripted>, Arc<Mutex<Vec<String>>>) {
    let messages = Arc::new(Mutex::new(Vec::new()));
    (
        Box::new(Scripted {
            acts,
            messages: Arc::clone(&messages),
        }),
        messages,
    )
}

/// An answer that writes `text` into the workspace file `name`.
fn edit(name: &'static str, text: &'static str) -> Act {
    Box::new(move |req| {
        std::fs::write(req.workspace.join(name), text).unwrap();
        DecisionOutcome {
            text: format!("wrote {name}"),
            cost_usd: Some(0.1),
            ..Default::default()
        }
    })
}

fn keep() -> Act {
    Box::new(|_| DecisionOutcome::default())
}

// ── a program language for tests ────────────────────────────────

/// `rule.txt` holds `sgd <lr>` or `sgd <lr> fail_at <n>`: the n-th update of
/// that program first scribbles over the params, then fails.
struct ToyHost;

const RULE: &str = "rule.txt";

struct Toy {
    lr: f32,
    fail_at: Option<u64>,
    /// At this update the params jump by +5, at the next one back (an
    /// evaluation on other weights).
    jump_at: Option<u64>,
    made: u64,
}

impl ProgramHost for ToyHost {
    fn starting_files(&self) -> Files {
        Files::from([(RULE.to_string(), "sgd 0.1\n".to_string())])
    }

    fn build(
        &mut self,
        files: &Files,
        _: &[ParamInfo],
        _: Option<&dyn Program>,
    ) -> ApiResult<Box<dyn Program>> {
        let words: Vec<&str> = files[RULE].split_whitespace().collect();
        let lr = |s: &str| {
            s.parse::<f32>()
                .map_err(|_| ApiError::type_(format!("bad lr {s:?}")))
        };
        match words.as_slice() {
            ["sgd", r] => Ok(Box::new(Toy {
                lr: lr(r)?,
                fail_at: None,
                jump_at: None,
                made: 0,
            })),
            ["sgd", r, "fail_at", n] => Ok(Box::new(Toy {
                lr: lr(r)?,
                fail_at: Some(n.parse().unwrap()),
                jump_at: None,
                made: 0,
            })),
            ["sgd", r, "jump_at", n] => Ok(Box::new(Toy {
                lr: lr(r)?,
                fail_at: None,
                jump_at: Some(n.parse().unwrap()),
                made: 0,
            })),
            _ => Err(ApiError::type_(format!(
                "{RULE}: cannot read {:?}",
                files[RULE]
            ))),
        }
    }

    fn interface(&self) -> String {
        format!("{RULE}: `sgd <lr>`")
    }
}

impl Program for Toy {
    fn step(&mut self, items: &mut [OptItem<'_>], ctx: &StepCtx) -> ApiResult<bool> {
        self.made += 1;
        if let Some(j) = self.jump_at {
            let shift = if ctx.update == j {
                5.0
            } else if ctx.update == j + 1 {
                -5.0
            } else {
                0.0
            };
            for it in items.iter_mut() {
                it.param.iter_mut().for_each(|p| *p += shift);
            }
        }
        if self.fail_at == Some(self.made) {
            for it in items.iter_mut() {
                it.param.fill(123.0);
            }
            return Err(ApiError::runtime("toy rule failed"));
        }
        for it in items.iter_mut() {
            for (p, g) in it.param.iter_mut().zip(it.grad) {
                *p -= self.lr * g;
            }
        }
        Ok(false)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn config(root: &Path, total: u64, decisions: usize) -> AgentConfig {
    let mut c = AgentConfig::new("toy quadratic; judged by the final loss", total);
    c.run_root = root.to_path_buf();
    c.decisions = Some(decisions);
    c.spread = Spread::Uniform;
    c.every = 5;
    c.trial_steps = 4;
    c
}

/// Params `w` [2, 2] and `b` [2] under the loss ½(|w|² + |b|²): the gradient
/// is the params themselves.
struct Toyrun {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Toyrun {
    fn new() -> Self {
        Self {
            w: vec![1.0, -2.0, 0.5, 3.0],
            b: vec![0.25, -1.0],
        }
    }

    fn loss(&self) -> f32 {
        0.5 * self.w.iter().chain(&self.b).map(|x| x * x).sum::<f32>()
    }

    fn steps(&mut self, agent: &mut AgentOptimizer, from: u64, to: u64) -> ApiResult<()> {
        self.steps_with(agent, from, to, |_, loss| loss)
    }

    /// Updates `from..=to`, the loss each update reports given by `loss_of(u, loss)`.
    fn steps_with(
        &mut self,
        agent: &mut AgentOptimizer,
        from: u64,
        to: u64,
        loss_of: impl Fn(u64, f32) -> f32,
    ) -> ApiResult<()> {
        for u in from..=to {
            let loss = loss_of(u, self.loss());
            let (gw, gb) = (self.w.clone(), self.b.clone());
            let mut items = [
                OptItem {
                    name: "w",
                    shape: &[2, 2],
                    param: &mut self.w,
                    grad: &gw,
                },
                OptItem {
                    name: "b",
                    shape: &[2],
                    param: &mut self.b,
                    grad: &gb,
                },
            ];
            agent.step(
                &mut items,
                &StepInfo {
                    update: u,
                    loss,
                    outputs: &[],
                    lr: None,
                },
            )?;
        }
        Ok(())
    }
}

fn events(agent: &AgentOptimizer) -> Vec<Value> {
    std::fs::read_to_string(agent.run_dir().join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn kinds(agent: &AgentOptimizer) -> Vec<String> {
    events(agent)
        .iter()
        .map(|e| e["kind"].as_str().unwrap().to_string())
        .collect()
}

/// The params after `to` updates of plain `sgd lr`, no agent.
fn reference(lr: f32, to: u64) -> Toyrun {
    let mut r = Toyrun::new();
    for _ in 0..to {
        for p in r.w.iter_mut().chain(r.b.iter_mut()) {
            *p -= lr * *p;
        }
    }
    r
}

// ── through the Trainer ─────────────────────────────────────────

fn regression() -> Graph {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("mse");
    let x = g.input("x", f32s(&[4, 3]));
    let t = g.input("t", f32s(&[4, 3]));
    let w = g.param("w", f32s(&[3, 3]));
    let b = g.param("b", f32s(&[1, 3]));
    let pred = g.mm(x, w);
    let biased = g.add(pred, b);
    let diff = g.sub(biased, t);
    let sq = g.mul(diff, diff);
    let loss = g.mean(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    g
}

fn trainer(optimizer: &Value) -> Trainer {
    let w: Vec<f32> = (0..9).map(|i| 0.1 * (i % 7) as f32 - 0.2).collect();
    Trainer::new(
        regression(),
        TrainerConfig {
            wrt: vec!["w".into(), "b".into()],
            init: vec![("w".into(), w), ("b".into(), vec![0.0; 3])],
            device: "cpu".into(),
            optimizer: optimizer.clone(),
            policy: None,
            clip_norm: Some(0.5),
            resident: Resident::Off,
            training: Default::default(),
        },
    )
    .unwrap()
}

#[test]
fn with_no_agent_a_trainer_run_is_the_plain_run_bit_for_bit() {
    let x: Vec<f32> = (0..12).map(|i| ((i * 5 % 11) as f32 - 5.0) / 5.0).collect();
    let t: Vec<f32> = (0..12).map(|i| ((i * 3 % 7) as f32 - 3.0) / 3.0).collect();
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    let lr_at = |i: u64| 0.05 * (1.0 - i as f32 / 120.0);
    for opts in [
        json!({"kind": "adamw", "lr": 0.05, "weight_decay": 0.01}),
        json!({"kind": "muon_adamw", "lr": 0.05}),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut plain = trainer(&opts);
        let mut wrapped = trainer(&opts);
        let host = RuleHost::new(&opts).unwrap();
        let mut agent =
            AgentOptimizer::new(config(dir.path(), 120, 6), Box::new(host), Box::new(Keep))
                .unwrap();
        for i in 0..120 {
            plain.set_lr(lr_at(i));
            agent.set_lr(lr_at(i));
            let a = plain.step(&inputs).unwrap();
            let b = agent.train_step(&mut wrapped, &inputs).unwrap();
            assert_eq!(a.to_bits(), b.to_bits(), "{opts} step {i}");
        }
        assert_eq!(plain.params(), wrapped.params(), "{opts}");
        let summary = agent.close().unwrap();
        assert_eq!(summary["decisions"], 6);
        assert_eq!(summary["kept"], 6);
        assert_eq!(summary["final_version"], 0);
    }
}

#[test]
fn a_rule_program_edit_takes_over_at_the_decision() {
    let dir = tempfile::tempdir().unwrap();
    let opts = json!({"kind": "adamw", "lr": 0.05});
    let frozen = r#"{"groups": [{"names": ["*"], "optimizer": {"kind": "sgd", "lr": 0.0}}]}"#;
    let (backend, messages) = scripted(vec![edit("updater.json", frozen)]);
    let mut agent = AgentOptimizer::new(
        config(dir.path(), 60, 2),
        Box::new(RuleHost::new(&opts).unwrap()),
        backend,
    )
    .unwrap();
    let mut tr = trainer(&opts);
    let x = vec![0.5f32; 12];
    let t = vec![-0.25f32; 12];
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    // Decisions at 20 and 40 (uniform over 60); the first sets lr 0.
    for _ in 0..19 {
        agent.train_step(&mut tr, &inputs).unwrap();
    }
    let before = tr.params();
    for _ in 19..30 {
        agent.train_step(&mut tr, &inputs).unwrap();
    }
    assert_eq!(tr.params(), before, "lr 0 from update 20 on");
    assert_eq!(agent.version(), 1);
    let first = messages.lock().unwrap()[0].clone();
    assert!(
        first.starts_with("Decision 1. Asked at update 20 of 60"),
        "{first}"
    );
    assert!(first.contains("scheduled decision (update 20)"), "{first}");
    assert!(dir.path().read_dir().unwrap().count() == 1);
    let run = agent.run_dir().to_path_buf();
    assert_eq!(
        std::fs::read_to_string(run.join("FINAL")).unwrap(),
        "v001\n"
    );
    assert!(
        std::fs::read_to_string(run.join("programs/v001/updater.json"))
            .unwrap()
            .contains("\"sgd\"")
    );
    let system = std::fs::read_to_string(run.join("system_prompt.txt")).unwrap();
    assert!(system.contains("toy quadratic; judged by the final loss"));
    assert!(system.contains("updater.json") && system.contains("muon_adamw"));
    assert!(
        system.contains("  w  [3, 3]") && system.contains("  b  [1, 3]"),
        "{system}"
    );
    for _ in 30..60 {
        agent.train_step(&mut tr, &inputs).unwrap();
    }
    let second = messages.lock().unwrap()[1].clone();
    assert!(
        second.contains("took effect at update 20, 0 updates after"),
        "{second}"
    );
    agent.close().unwrap();
}

#[test]
fn a_program_that_does_not_build_is_rejected_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![edit(RULE, "adam please")]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 30).unwrap();
    let r = reference(0.1, 30);
    assert_eq!((run.w, run.b), (r.w, r.b));
    assert_eq!(agent.version(), 0);
    let second = messages.lock().unwrap()[1].clone();
    assert!(second.contains("failed to build at update 10"), "{second}");
    assert!(second.contains("cannot read \"adam please\""), "{second}");
    assert!(kinds(&agent).contains(&"rejected".to_string()));
    assert!(
        agent
            .run_dir()
            .join("programs/rejected-v001/rule.txt")
            .exists()
    );
    // The workspace was put back to the running program for the next decision.
    assert_eq!(agent.files()[RULE], "sgd 0.1\n");
}

#[test]
fn a_failure_at_the_first_update_is_undone_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![edit(RULE, "sgd 0.5 fail_at 1")]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 30).unwrap();
    // As if the failing program had never run: plain sgd 0.1 throughout.
    let r = reference(0.1, 30);
    assert_eq!((run.w, run.b), (r.w, r.b));
    assert_eq!(agent.version(), 2, "v001 failed, v002 is v000 again");
    assert_eq!(agent.files()[RULE], "sgd 0.1\n");
    let reverted = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "reverted")
        .unwrap();
    assert_eq!(reverted["update"], 10);
    assert_eq!(reverted["exact"], true);
    let second = messages.lock().unwrap()[1].clone();
    assert!(
        second.contains("undone exactly and made by the previous program"),
        "{second}"
    );
    assert!(second.contains("toy rule failed"), "{second}");
}

#[test]
fn without_a_snapshot_the_failing_first_update_stays() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, _) = scripted(vec![edit(RULE, "sgd 0.5 fail_at 1")]);
    let mut cfg = config(dir.path(), 30, 2);
    cfg.snapshot_limit_bytes = 8;
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 10).unwrap();
    assert!(
        run.w.iter().chain(&run.b).all(|&x| x == 123.0),
        "the scribble stays: {:?}",
        run.w
    );
    let reverted = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "reverted")
        .unwrap();
    assert_eq!(reverted["exact"], false);
}

#[test]
fn a_failure_later_in_the_trial_goes_back_without_an_undo() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, _) = scripted(vec![edit(RULE, "sgd 0.5 fail_at 3")]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 12).unwrap();
    // 9 updates of 0.1, 2 of 0.5 (10, 11), the scribble at 12 kept.
    assert!(run.w.iter().all(|&x| x == 123.0));
    run.steps(&mut agent, 13, 13).unwrap();
    assert!(
        run.w.iter().all(|&x| (x - 123.0 * 0.9).abs() < 1e-4),
        "{:?}",
        run.w
    );
    let reverted = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "reverted")
        .unwrap();
    assert_eq!(
        (reverted["update"].as_u64(), reverted["exact"].as_bool()),
        (Some(12), Some(false))
    );
    assert_eq!(agent.files()[RULE], "sgd 0.1\n");
}

#[test]
fn after_its_trial_a_failing_program_is_replaced_by_a_rebuilt_previous_one() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, _) = scripted(vec![edit(RULE, "sgd 0.2 fail_at 8"), keep()]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 30).unwrap();
    let k = kinds(&agent);
    assert!(k.contains(&"trial_passed".to_string()), "{k:?}");
    let reverted = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "reverted")
        .unwrap();
    assert_eq!(
        reverted["update"], 17,
        "v001's 8th update, after its 4-update trial"
    );
    assert_eq!(agent.files()[RULE], "sgd 0.1\n");
    assert_eq!(agent.version(), 2);
}

#[test]
fn a_failing_starting_program_has_nothing_to_go_back_to() {
    struct Broken;
    impl ProgramHost for Broken {
        fn starting_files(&self) -> Files {
            Files::from([(RULE.to_string(), "sgd 0.1 fail_at 2".to_string())])
        }
        fn build(
            &mut self,
            f: &Files,
            p: &[ParamInfo],
            prev: Option<&dyn Program>,
        ) -> ApiResult<Box<dyn Program>> {
            ToyHost.build(f, p, prev)
        }
        fn interface(&self) -> String {
            String::new()
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(Broken), Box::new(Keep)).unwrap();
    let mut run = Toyrun::new();
    let err = run.steps(&mut agent, 1, 3).unwrap_err();
    assert!(
        err.message.contains("no earlier program to go back to"),
        "{}",
        err.message
    );
}

#[test]
fn a_non_finite_loss_pauses_a_non_blocking_run_for_a_decision() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![]);
    let mut cfg = config(dir.path(), 100, 3);
    cfg.waiting = Waiting::Never;
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps_with(&mut agent, 1, 5, |u, l| if u == 4 { f32::NAN } else { l })
        .unwrap();
    let first = messages.lock().unwrap()[0].clone();
    assert!(
        first.contains("non-finite training loss at update 4 (found at the check of update 5)"),
        "{first}"
    );
    assert!(
        first.contains("Training is paused until you finish"),
        "{first}"
    );
    let requested = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "requested")
        .unwrap();
    assert_eq!(requested["paused"], true);
    // Answered within the pause.
    assert!(kinds(&agent).contains(&"kept".to_string()));
}

#[test]
fn the_cost_cap_stops_the_asking() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![
        edit(RULE, "sgd 0.1 \n"),
        edit(RULE, "sgd 0.1\n"),
        edit(RULE, "sgd 0.1  \n"),
    ]);
    let mut cfg = config(dir.path(), 50, 4);
    cfg.max_cost_usd = Some(0.15);
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    Toyrun::new().steps(&mut agent, 1, 50).unwrap();
    assert_eq!(
        messages.lock().unwrap().len(),
        2,
        "0.1 each: the third would pass 0.15"
    );
    let summary = agent.close().unwrap();
    assert_eq!(summary["decisions"], 2);
    assert!((summary["cost_usd"].as_f64().unwrap() - 0.2).abs() < 1e-9);
}

#[test]
fn a_failed_agent_call_leaves_the_program_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let crash: Act = Box::new(|req| {
        std::fs::write(req.workspace.join(RULE), "sgd 9.0").unwrap();
        DecisionOutcome {
            error: Some("usage limit reached".into()),
            ..Default::default()
        }
    });
    let (backend, messages) = scripted(vec![crash]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 30).unwrap();
    let r = reference(0.1, 30);
    assert_eq!(
        (run.w, run.b),
        (r.w, r.b),
        "edits of a failed call are not applied"
    );
    let second = messages.lock().unwrap()[1].clone();
    assert!(second.contains("usage limit reached"), "{second}");
    assert!(
        agent
            .run_dir()
            .join("decisions/001/discarded/rule.txt")
            .exists()
    );
}

#[test]
fn agent_config_reads_its_json_and_refuses_typos() {
    let c = AgentConfig::from_json(&json!({"goal": "g", "total_steps": 100, "decisions": 3,
                                            "spread": "uniform", "blocking": false}))
    .unwrap();
    assert_eq!(
        (c.decisions, c.spread, c.waiting),
        (Some(3), Spread::Uniform, Waiting::Never)
    );
    let auto = AgentConfig::from_json(&json!({"goal": "g", "total_steps": 9, "blocking": "auto"}))
        .unwrap();
    assert_eq!(auto.waiting, Waiting::Auto);
    assert!(
        AgentConfig::from_json(&json!({"goal": "g", "total_steps": 9, "blocking": "sometimes"}))
            .is_err()
    );
    let err = AgentConfig::from_json(&json!({"goal": "g", "total_steps": 100, "decision": 3}))
        .unwrap_err();
    assert!(err.message.contains("'decision'"), "{}", err.message);
    assert!(AgentConfig::from_json(&json!({"total_steps": 100})).is_err());
}

/// One real decision with Claude Code. Costs money: opt in with
/// `RLX_AGENT_REAL_CLI=1 cargo test … -- --ignored`.
#[test]
#[ignore = "calls the real Claude Code CLI (billed)"]
fn claude_code_answers_one_decision() {
    if !rlx_ir::env::flag("RLX_AGENT_REAL_CLI") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let opts = json!({"kind": "adamw", "lr": 0.05});
    let mut cfg = config(dir.path(), 40, 1);
    cfg.max_cost_usd = Some(2.0);
    let mut agent = AgentOptimizer::new(
        cfg,
        Box::new(RuleHost::new(&opts).unwrap()),
        Box::new(rlx_bind::runtime::agent::CliAgent::new(
            rlx_bind::runtime::agent::AgentCli::Claude,
        )),
    )
    .unwrap();
    let mut tr = trainer(&opts);
    let x = vec![0.5f32; 12];
    let t = vec![-0.25f32; 12];
    for _ in 0..40 {
        agent.train_step(&mut tr, &[("x", &x), ("t", &t)]).unwrap();
    }
    let run_dir = agent.run_dir().to_path_buf();
    let summary = agent.close().unwrap();
    eprintln!("{summary:#}");
    assert_eq!(summary["decisions"], 1);
    // It must have worked through its tools, not only answered.
    let transcript =
        std::fs::read_to_string(run_dir.join("decisions/001/transcript.jsonl")).unwrap();
    assert!(
        transcript.contains("mcp__rlx_files__"),
        "no workspace tool call in {}",
        run_dir.display()
    );
    assert_eq!(
        summary["failed"], 0,
        "see decisions/001/ in {}",
        summary["run_dir"]
    );
}

// ── the analysis program ────────────────────────────────────────

use rlx_bind::runtime::agent::{ANALYSIS, estimate_decisions};

#[test]
fn an_analysis_edit_replaces_the_reports_without_a_trial_and_can_wake_the_agent() {
    let dir = tempfile::tempdir().unwrap();
    let analysis = r#"{"every": 1, "stats": ["update_ratio"],
                       "wake": [{"stat": "loss_mean", "below": 0.01}]}"#;
    let (backend, messages) = scripted(vec![edit(ANALYSIS, analysis)]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 60, 2), Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 60).unwrap();
    // The updater never changed: plain sgd 0.1 throughout.
    let r = reference(0.1, 60);
    assert_eq!((run.w, run.b), (r.w, r.b));
    let ev = events(&agent);
    let applied = ev.iter().find(|e| e["kind"] == "applied").unwrap();
    assert_eq!(
        (applied["updater"].as_bool(), applied["analysis"].as_bool()),
        (Some(false), Some(true))
    );
    assert!(
        !ev.iter().any(|e| e["kind"] == "trial_passed"),
        "no trial for an analysis change"
    );
    // 7.66 × 0.81^u drops below 0.01 after update 32; the report after update
    // 33 holds that loss (a step's loss is observed after it), and the wake-up
    // asks decision 2 at update 34, before the scheduled one at 40.
    let msgs = messages.lock().unwrap();
    assert_eq!(msgs.len(), 2, "the wake-up used the last decision");
    assert!(
        msgs[1].contains("your analysis: loss_mean = 0.00"),
        "{}",
        msgs[1]
    );
    assert!(msgs[1].contains("is below 0.01 (update 33)"), "{}", msgs[1]);
    assert!(
        msgs[1].starts_with("Decision 2. Asked at update 34"),
        "{}",
        msgs[1]
    );
    let inbox: Value = serde_json::from_str(
        &std::fs::read_to_string(agent.run_dir().join("workspace/inbox/decision-002.json"))
            .unwrap(),
    )
    .unwrap();
    let reports = inbox["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 14, "one report per update, 20 to 33");
    let ratio = reports[0]["update_ratio"]["w"].as_f64().unwrap();
    assert!(
        (ratio - 0.1).abs() < 1e-6,
        "sgd 0.1 on ½|w|² moves w by 10%: {ratio}"
    );
    assert!(reports[0].get("grad_norm").is_none());
}

#[test]
fn a_bad_analysis_is_rejected_and_the_old_one_stays() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![edit(ANALYSIS, r#"{"stats": ["curvature"]}"#)]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 30, 2), Box::new(ToyHost), backend).unwrap();
    Toyrun::new().steps(&mut agent, 1, 30).unwrap();
    assert_eq!(agent.version(), 0);
    assert!(agent.files()[ANALYSIS].contains("grad_norm"));
    let second = messages.lock().unwrap()[1].clone();
    assert!(
        second.contains("unknown stat \\\"curvature\\\"")
            || second.contains("unknown stat \"curvature\""),
        "{second}"
    );
}

// ── checkpoints ─────────────────────────────────────────────────

/// `w`, `b` under ½|·|², through rules over AdamW: a program with state.
fn rules() -> Box<RuleHost> {
    Box::new(RuleHost::new(&json!({"kind": "adamw", "lr": 0.05})).unwrap())
}

const FASTER: &str = r#"{"groups": [{"names": ["w"], "optimizer": {"kind": "adamw", "lr": 0.1}},
                                    {"names": ["*"], "optimizer": {"kind": "adamw", "lr": 0.05}}]}"#;

#[test]
fn a_resumed_run_continues_bit_for_bit() {
    let root = tempfile::tempdir().unwrap();
    // Uninterrupted: decisions at 10 (an edit) and 20 (kept).
    let (backend, _) = scripted(vec![edit("updater.json", FASTER), keep()]);
    let mut whole = AgentOptimizer::new(config(root.path(), 30, 2), rules(), backend).unwrap();
    let mut a = Toyrun::new();
    a.steps(&mut whole, 1, 30).unwrap();

    // The same run, stopped after update 15 and resumed from a checkpoint.
    let (backend, _) = scripted(vec![edit("updater.json", FASTER)]);
    let mut first = AgentOptimizer::new(config(root.path(), 30, 2), rules(), backend).unwrap();
    let mut b = Toyrun::new();
    b.steps(&mut first, 1, 15).unwrap();
    let ck = root.path().join("ck");
    first.save(&ck).unwrap();
    let run_dir = first.run_dir().to_path_buf();
    drop(first);
    let (backend, messages) = scripted(vec![keep()]);
    let mut resumed =
        AgentOptimizer::resume(config(root.path(), 30, 2), rules(), backend, &ck).unwrap();
    assert_eq!(
        resumed.run_dir(),
        run_dir,
        "continues in its own run folder"
    );
    let mut c = Toyrun {
        w: b.w.clone(),
        b: b.b.clone(),
    };
    c.steps(&mut resumed, 16, 30).unwrap();

    assert_eq!(
        (c.w, c.b),
        (a.w, a.b),
        "moments, step count and program carried over"
    );
    assert_eq!(resumed.version(), 1);
    assert!(resumed.files()["updater.json"].contains("0.1"));
    let msgs = messages.lock().unwrap();
    assert!(
        msgs[0].starts_with("Decision 2. Asked at update 20"),
        "{}",
        msgs[0]
    );
    assert!(!msgs[0].contains("could not be restored"), "{}", msgs[0]);
    let kinds = kinds(&resumed);
    assert!(kinds.contains(&"resumed".to_string()) && kinds.contains(&"restored".to_string()));
}

#[test]
fn a_program_without_state_resumes_from_zero_and_says_so() {
    let root = tempfile::tempdir().unwrap();
    let mut first = AgentOptimizer::new(
        config(root.path(), 30, 2),
        Box::new(ToyHost),
        Box::new(Keep),
    )
    .unwrap();
    Toyrun::new().steps(&mut first, 1, 5).unwrap();
    let ck = root.path().join("ck");
    first.save(&ck).unwrap();
    let (backend, messages) = scripted(vec![]);
    let mut resumed =
        AgentOptimizer::resume(config(root.path(), 30, 2), Box::new(ToyHost), backend, &ck)
            .unwrap();
    Toyrun::new().steps(&mut resumed, 6, 10).unwrap();
    let first_msg = messages.lock().unwrap()[0].clone();
    assert!(
        first_msg.contains("could not be restored; it restarted from zero"),
        "{first_msg}"
    );
}

// ── an estimated schedule ───────────────────────────────────────

#[test]
fn the_decision_count_follows_the_time_budget() {
    assert_eq!(estimate_decisions(1000, 0.1, 40.0, 4.0), 10);
    assert_eq!(estimate_decisions(100_000, 0.5, 40.0, 4.0), 50, "capped");
    assert_eq!(estimate_decisions(100, 0.001, 40.0, 4.0), 1, "at least one");
}

#[test]
fn an_unset_decision_count_is_estimated_after_the_first_updates() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = config(root.path(), 1000, 1);
    cfg.decisions = None;
    cfg.spread = Spread::Front;
    cfg.seconds_per_decision = 1e-12; // every decision fits: the cap decides
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), Box::new(Keep)).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 10).unwrap();
    assert!(
        agent.schedule().is_none(),
        "not before the first updates are timed"
    );
    run.steps(&mut agent, 11, 1000).unwrap();
    let s = agent.schedule().unwrap().to_vec();
    assert_eq!(s.len(), 50);
    assert!(
        s[0] == 11 && s.windows(2).all(|w| w[0] < w[1]),
        "early points moved after the timing: {s:?}"
    );
    let est = events(&agent)
        .into_iter()
        .find(|e| e["kind"] == "estimated")
        .unwrap();
    assert_eq!(
        (est["update"].as_u64(), est["decisions"].as_u64()),
        (Some(11), Some(50))
    );
    assert_eq!(agent.close().unwrap()["decisions"], 50);
}

// ── agent CLIs through rlx-agent ────────────────────────────────

use rlx_bind::runtime::agent::{AgentCli, CliAgent, backend_from_json};

fn request(dir: &Path) -> DecisionRequest {
    DecisionRequest {
        k: 1,
        workspace: dir.join("workspace"),
        writable: vec![
            "updater.json".into(),
            "analysis.json".into(),
            "NOTES.md".into(),
        ],
        record: dir.join("decisions/001"),
        message: "Decision 1.".into(),
        system: "SYSTEM".into(),
        budget_left: Some(1.5),
    }
}

fn after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

#[test]
fn claude_gets_only_the_workspace_server_one_session_and_the_system_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let opts = json!({"model": "m", "mcp_command": ["/bin/rlx-agent", "workspace-mcp"]});
    let agent = CliAgent::from_json(AgentCli::Claude, &opts).unwrap();
    let (run, prompt) = agent.agent_run(&request(dir.path()));
    assert_eq!(prompt, "Decision 1.");
    let (program, args) = run.command_line(&prompt);
    assert_eq!(program, "claude");
    assert_eq!(after(&args, "--tools"), Some(""), "no built-in tools");
    assert_eq!(after(&args, "--allowedTools"), Some("mcp__rlx_files"));
    assert_eq!(after(&args, "--append-system-prompt"), Some("SYSTEM"));
    assert_eq!(after(&args, "--session-id").map(str::len), Some(36));
    assert_eq!(after(&args, "--max-budget-usd"), Some("1.5"));
    assert_eq!(after(&args, "--model"), Some("m"));
    assert_eq!(run.workspace, dir.path().join("workspace"));
    let server = &run.mcp_config()["mcpServers"]["rlx_files"];
    assert_eq!(server["command"], "/bin/rlx-agent");
    let ws = dir.path().join("workspace").display().to_string();
    assert_eq!(
        server["args"],
        json!([
            "workspace-mcp",
            "--dir",
            ws,
            "--write",
            "updater.json",
            "--write",
            "analysis.json",
            "--write",
            "NOTES.md"
        ])
    );
}

#[test]
fn other_clis_get_the_system_prompt_in_front_of_each_message() {
    let dir = tempfile::tempdir().unwrap();
    let agent = CliAgent::from_json(AgentCli::Copilot, &json!({})).unwrap();
    let (run, prompt) = agent.agent_run(&request(dir.path()));
    assert_eq!(prompt, "SYSTEM\n\n---\n\nDecision 1.");
    assert!(run.extra_args.is_empty());
    assert!(backend_from_json("cursor", &json!({})).is_ok());
    let err = backend_from_json("gpt", &json!({})).err().unwrap();
    assert!(
        err.message.contains("\"keep\", \"claude\""),
        "{}",
        err.message
    );
    let err = backend_from_json("claude", &json!({"timeout": 5}))
        .err()
        .unwrap();
    assert!(err.message.contains("'timeout'"), "{}", err.message);
}

/// A stand-in `claude`: it edits `updater.json` through the workspace MCP
/// server named in `--mcp-config` (lr 0.05 → 0.0), tries a write it may not
/// make, and prints stream-json as the real CLI does.
const FAKE_CLAUDE: &str = r#"#!/usr/bin/env python3
import json, subprocess, sys
args = sys.argv[1:]
cfg = json.load(open(args[args.index("--mcp-config") + 1]))
srv = cfg["mcpServers"]["rlx_files"]
p = subprocess.Popen([srv["command"], *srv["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def ask(i, method, params):
    p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params}) + "\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())["result"]
ask(1, "initialize", {"protocolVersion": "2025-06-18"})
text = ask(2, "tools/call", {"name": "read_file", "arguments": {"path": "updater.json"}})["content"][0]["text"]
ask(3, "tools/call", {"name": "write_file", "arguments": {"path": "updater.json", "content": text.replace('"lr": 0.05', '"lr": 0.0')}})
bad = ask(4, "tools/call", {"name": "write_file", "arguments": {"path": "inbox/x.json", "content": "{}"}})
p.stdin.close(); p.wait()
json.dump(args, open(".args.json", "w"))
print(json.dumps({"type": "system", "subtype": "init", "model": "stand-in"}))
print(json.dumps({"type": "result", "result": "Observed: fine. refused=%s" % bad["isError"], "total_cost_usd": 0.03, "num_turns": 1, "duration_ms": 5}))
"#;

#[cfg(unix)]
#[test]
fn a_decision_runs_through_rlx_agent_and_its_workspace_server() {
    use std::os::unix::fs::PermissionsExt;
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipped: no python3 for the stand-in CLI");
        return;
    }
    // The workspace server is rlx-agent's binary; build it next to this test.
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let built = std::process::Command::new(cargo)
        .args(["build", "-q", "-p", "rlx-agent", "--bin", "rlx-agent"])
        .status()
        .unwrap();
    assert!(built.success());

    let dir = tempfile::tempdir().unwrap();
    let fake = dir.path().join("claude");
    std::fs::write(&fake, FAKE_CLAUDE).unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let backend =
        backend_from_json("claude", &json!({"program": fake.display().to_string()})).unwrap();

    let opts = json!({"kind": "adamw", "lr": 0.05});
    let mut agent = AgentOptimizer::new(
        config(&dir.path().join("runs"), 60, 1),
        Box::new(RuleHost::new(&opts).unwrap()),
        backend,
    )
    .unwrap();
    let mut tr = trainer(&opts);
    let x = vec![0.5f32; 12];
    let t = vec![-0.25f32; 12];
    let inputs = [("x", x.as_slice()), ("t", t.as_slice())];
    for _ in 0..29 {
        agent.train_step(&mut tr, &inputs).unwrap();
    }
    let before = tr.params();
    for _ in 29..60 {
        agent.train_step(&mut tr, &inputs).unwrap();
    }
    let run = agent.run_dir().to_path_buf();
    let record = run.join("decisions/001");
    let said = |p: &Path| std::fs::read_to_string(p).unwrap_or_else(|e| format!("({e})"));
    assert_eq!(
        tr.params(),
        before,
        "lr 0 from the decision at update 30\nresponse: {}\nevents: {}",
        said(&record.join("response.json")),
        said(&run.join("events.jsonl"))
    );
    let response: Value =
        serde_json::from_str(&std::fs::read_to_string(record.join("response.json")).unwrap())
            .unwrap();
    assert_eq!(
        response["text"], "Observed: fine. refused=True",
        "the forbidden write was refused"
    );
    assert_eq!(response["cost_usd"], 0.03);
    assert!(record.join("transcript.jsonl").exists() && record.join("mcp.json").exists());
    assert!(!run.join("workspace/inbox/x.json").exists());
    let args: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(run.join("workspace/.args.json")).unwrap())
            .unwrap();
    assert_eq!(after(&args, "--tools"), Some(""));
    assert!(
        after(&args, "--append-system-prompt")
            .unwrap()
            .contains("toy quadratic")
    );
    assert!(after(&args, "--session-id").is_some());
    let summary = agent.close().unwrap();
    assert_eq!(
        (summary["applied"].as_u64(), summary["cost_usd"].as_f64()),
        (Some(1), Some(0.03))
    );
}

// ── what the training script shows the agent ────────────────────

use rlx_bind::runtime::agent::{Prompt, TaskFacts, cli, records};

#[test]
fn validation_results_and_shown_values_reach_the_agent_and_the_records() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![]);
    let mut cfg = config(dir.path(), 30, 2);
    cfg.validation = Some(5);
    cfg.validation_size = Some(1000);
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    for u in 1..=30u64 {
        run.steps(&mut agent, u, u).unwrap();
        if u % 5 == 0 {
            agent.report_validation(Some(1.0 / u as f64), Some(0.5), None);
            agent
                .log(json!({"grad_noise": u as f64 * 0.1}), true)
                .unwrap();
            agent.log(json!({"secret_test_acc": 0.9}), false).unwrap();
        }
    }
    let msgs = messages.lock().unwrap();
    // Decision 2 (update 20) sees the validation results of 5, 10 and 15 (and 20 not yet).
    assert!(msgs[1].contains("Validation set (held out from training; not the evaluation in the goal), all 3 measurements"), "{}", msgs[1]);
    assert!(
        msgs[1].contains("15: loss 0.0667, accuracy 0.5000"),
        "{}",
        msgs[1]
    );
    assert!(msgs[1].contains("\"grad_noise\":1.5"), "{}", msgs[1]);
    assert!(
        !msgs[1].contains("secret_test_acc"),
        "values not shown stay out"
    );
    let system = std::fs::read_to_string(agent.run_dir().join("system_prompt.txt")).unwrap();
    assert!(
        system.contains("a validation set of 1000 held out from training about every 5 updates"),
        "{system}"
    );
    let run_dir = agent.run_dir().to_path_buf();
    let results = agent.close().unwrap();
    assert_eq!(results["best_validation"]["update"], 30);
    assert_eq!(results["last_validation"]["update"], 30);
    assert_eq!(results["integrity_flags"], json!([]));
    let metrics = std::fs::read_to_string(run_dir.join("metrics.jsonl")).unwrap();
    assert_eq!(
        metrics.lines().count(),
        12,
        "shown and recorded values both recorded"
    );
    let losses = std::fs::read_to_string(run_dir.join("losses.csv")).unwrap();
    assert_eq!(
        losses.lines().count(),
        31,
        "a header and one line per update"
    );
}

#[test]
fn a_program_that_evaluates_other_weights_is_flagged_and_the_agent_is_not_told() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![edit(RULE, "sgd 0.1 jump_at 15")]);
    let mut cfg = config(dir.path(), 30, 2);
    cfg.validation = Some(5);
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    for u in 1..=30u64 {
        run.steps(&mut agent, u, u).unwrap();
        if u % 5 == 0 {
            agent.report_validation(Some(0.1), None, None);
        }
    }
    let flagged: Vec<Value> = events(&agent)
        .into_iter()
        .filter(|e| e["kind"] == "integrity")
        .collect();
    assert_eq!(flagged.len(), 1, "{flagged:?}");
    assert_eq!(flagged[0]["validation_update"], 15);
    assert!(
        !messages
            .lock()
            .unwrap()
            .iter()
            .any(|m| m.contains("integrity"))
    );
    assert_eq!(
        agent.close().unwrap()["integrity_flags"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

const SKIPPING: &str = r#"{"groups": [{"names": ["*"], "optimizer": {"kind": "adamw", "lr": 0.05}}],
                         "skip_if": {"loss_above": 50}}"#;

#[test]
fn a_program_can_skip_updates_and_ten_in_a_row_are_made_anyway() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, messages) = scripted(vec![edit("updater.json", SKIPPING)]);
    // Decisions at 10 (the skip rule), 20 and 30.
    let mut agent = AgentOptimizer::new(config(dir.path(), 40, 3), rules(), backend).unwrap();
    let mut run = Toyrun::new();
    // Updates 11 and 12 report a loss of 100: skipped, params unchanged.
    run.steps(&mut agent, 1, 10).unwrap();
    let before = (run.w.clone(), run.b.clone());
    run.steps_with(&mut agent, 11, 12, |_, _| 100.0).unwrap();
    assert_eq!(
        (run.w.clone(), run.b.clone()),
        before,
        "skipped updates change nothing"
    );
    run.steps(&mut agent, 13, 13).unwrap();
    assert_ne!((run.w.clone(), run.b.clone()), before);
    // Eleven in a row (14 to 24): the 11th is made anyway, and the agent asked.
    run.steps_with(&mut agent, 14, 24, |_, _| 100.0).unwrap();
    let ev = events(&agent);
    assert_eq!(ev.iter().filter(|e| e["kind"] == "skipped").count(), 12);
    assert_eq!(
        ev.iter().filter(|e| e["kind"] == "skip_overridden").count(),
        1
    );
    run.steps(&mut agent, 25, 40).unwrap();
    let msgs = messages.lock().unwrap();
    assert!(msgs[1].contains("Updates your program skipped since your previous decision: 8 (11, 12, 14, 15, 16, 17, 18, 19)"), "{}", msgs[1]);
    assert!(
        msgs[2].starts_with("Decision 3. Asked at update 25"),
        "{}",
        msgs[2]
    );
    assert!(
        msgs[2].contains("your program skipped 10 updates in a row; update 24 was made anyway"),
        "{}",
        msgs[2]
    );
    assert!(
        msgs[2].contains("skipped since your previous decision: 4 (20, 21, 22, 23)"),
        "{}",
        msgs[2]
    );
    assert_eq!(agent.close().unwrap()["skipped"], 12);
}

#[test]
fn notes_facts_shared_code_and_the_aggressive_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("train.py");
    std::fs::write(&script, "model = MLP()\n").unwrap();
    let (backend, messages) = scripted(vec![]);
    let mut cfg = config(dir.path(), 30, 2);
    cfg.facts = TaskFacts {
        epochs: Some(3.0),
        train_size: Some(60_000),
        batch_size: Some(128),
        same_order_each_epoch: Some(false),
        notes: Some("labels are noisy".into()),
    };
    cfg.share = vec![script];
    cfg.prompt = Prompt::Aggressive;
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut agent, 1, 5).unwrap();
    records::say(agent.run_dir(), "try a lower rate for b").unwrap();
    run.steps(&mut agent, 6, 30).unwrap();
    let run_dir = agent.run_dir().to_path_buf();
    assert_eq!(
        std::fs::read_to_string(run_dir.join("workspace/reference/code/train.py")).unwrap(),
        "model = MLP()\n"
    );
    let system = std::fs::read_to_string(run_dir.join("system_prompt.txt")).unwrap();
    assert!(system.contains("The run, about 3 epochs over a training set of 60000, with a batch of 128 per update; each epoch passes over the data in a new order."), "{system}");
    assert!(system.contains("From the person running the training: labels are noisy"));
    assert!(system.contains("read-only copies of the training run's code in reference/code/"));
    assert!(system.contains("2. Review:"), "the aggressive variant");
    let msgs = messages.lock().unwrap();
    assert!(
        msgs[0].contains("Note from the person running this training: try a lower rate for b"),
        "{}",
        msgs[0]
    );
    assert!(!msgs[1].contains("try a lower rate"), "a note is read once");
    assert!(msgs[0].ends_with("four parts of DECISIONS: observed, review, changed, prediction."));
    assert!(run_dir.join("notes/read").read_dir().unwrap().count() == 1);
}

#[test]
fn a_template_prompt_fills_the_known_fields() {
    let dir = tempfile::tempdir().unwrap();
    let t = dir.path().join("prompt.md");
    std::fs::write(
        &t,
        "GOAL={goal} STEPS={total} KEEP={unknown} JSON={\"a\": 1}\n{table}",
    )
    .unwrap();
    let mut cfg = config(dir.path(), 30, 1);
    cfg.prompt = Prompt::Template(t);
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), Box::new(Keep)).unwrap();
    Toyrun::new().steps(&mut agent, 1, 2).unwrap();
    let system = std::fs::read_to_string(agent.run_dir().join("system_prompt.txt")).unwrap();
    assert!(system.starts_with("GOAL=toy quadratic; judged by the final loss STEPS=30 KEEP={unknown} JSON={\"a\": 1}\n  w  [2, 2]"), "{system}");
    let mut bad = config(dir.path(), 30, 1);
    bad.prompt = Prompt::Template(dir.path().join("missing.md"));
    assert!(AgentOptimizer::new(bad, Box::new(ToyHost), Box::new(Keep)).is_err());
}

#[test]
fn auto_waiting_waits_at_the_first_decision_and_then_when_answers_would_land_late() {
    for (max_lag, later_paused) in [(u64::MAX / 2, false), (0, true)] {
        let dir = tempfile::tempdir().unwrap();
        let (backend, _) = scripted(vec![]);
        let mut cfg = config(dir.path(), 40, 3);
        cfg.waiting = Waiting::Auto;
        cfg.max_lag_updates = max_lag;
        let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
        Toyrun::new().steps(&mut agent, 1, 40).unwrap();
        let paused: Vec<bool> = events(&agent)
            .iter()
            .filter(|e| e["kind"] == "requested")
            .map(|e| e["paused"].as_bool().unwrap())
            .collect();
        assert_eq!(
            paused,
            vec![true, later_paused, later_paused],
            "max_lag {max_lag}"
        );
        agent.close().unwrap();
    }
}

#[test]
fn a_finished_runs_program_replays_without_an_agent() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, _) = scripted(vec![edit(RULE, "sgd 0.3\n")]);
    let mut first =
        AgentOptimizer::new(config(dir.path(), 30, 1), Box::new(ToyHost), backend).unwrap();
    Toyrun::new().steps(&mut first, 1, 30).unwrap();
    let run_dir = first.run_dir().to_path_buf();
    first.close().unwrap();
    let (version, files) = records::final_program(&run_dir).unwrap();
    assert_eq!((version, files[RULE].as_str()), (1, "sgd 0.3\n"));

    let mut again =
        AgentOptimizer::replay(config(dir.path(), 30, 5), Box::new(ToyHost), &run_dir).unwrap();
    let mut run = Toyrun::new();
    run.steps(&mut again, 1, 30).unwrap();
    let r = reference(0.3, 30);
    assert_eq!(
        (run.w, run.b),
        (r.w, r.b),
        "the final program from the first update"
    );
    let summary = again.close().unwrap();
    assert_eq!(
        (summary["decisions"].as_u64(), summary["backend"].as_str()),
        (Some(0), Some("keep"))
    );
    assert!(records::final_program(&dir.path().join("nowhere")).is_err());
}

#[test]
fn credentials_in_answers_are_redacted_in_the_records() {
    let dir = tempfile::tempdir().unwrap();
    let leaky: Act = Box::new(|_| DecisionOutcome {
        text: "my key is sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ, kept".into(),
        ..Default::default()
    });
    let (backend, _) = scripted(vec![leaky]);
    let mut agent =
        AgentOptimizer::new(config(dir.path(), 20, 1), Box::new(ToyHost), backend).unwrap();
    Toyrun::new().steps(&mut agent, 1, 20).unwrap();
    let r = std::fs::read_to_string(agent.run_dir().join("decisions/001/response.json")).unwrap();
    assert!(
        r.contains("my key is [redacted], kept") && !r.contains("sk-ant"),
        "{r}"
    );
}

// ── the command line over run folders ───────────────────────────

fn cli_out(args: &[&str]) -> Result<String, String> {
    let mut out = Vec::new();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    cli::run(&args, &cli::CliDefaults::default(), &mut out)?;
    Ok(String::from_utf8(out).unwrap())
}

#[test]
fn show_watch_report_compare_and_say_read_a_run_folder() {
    let dir = tempfile::tempdir().unwrap();
    let (backend, _) = scripted(vec![edit(RULE, "sgd 0.2\n"), keep()]);
    let mut cfg = config(dir.path(), 30, 2);
    cfg.validation = Some(10);
    let mut agent = AgentOptimizer::new(cfg, Box::new(ToyHost), backend).unwrap();
    let mut run = Toyrun::new();
    for u in 1..=30 {
        run.steps(&mut agent, u, u).unwrap();
        if u % 10 == 0 {
            agent.report_validation(Some(0.5), None, None);
        }
    }
    let r = agent.run_dir().display().to_string();
    agent.close().unwrap();
    let shown = cli_out(&["show", &r]).unwrap();
    assert!(shown.contains("goal: toy quadratic"), "{shown}");
    assert!(
        shown.contains("  1  update      10  applied v001"),
        "{shown}"
    );
    assert!(shown.contains("  2  update      20  kept"), "{shown}");
    assert!(
        shown.contains("closed: 2 decisions, 1 applied, final v001"),
        "{shown}"
    );
    let watched = cli_out(&["watch", &r, "--timeout", "0"]).unwrap();
    assert!(
        watched.lines().next().unwrap().starts_with("started"),
        "{watched}"
    );
    assert!(
        watched
            .trim_end()
            .lines()
            .last()
            .unwrap()
            .starts_with("closed")
    );
    let page_path = dir.path().join("page.html");
    cli_out(&["report", &r, "--out", &page_path.display().to_string()]).unwrap();
    let page = std::fs::read_to_string(&page_path).unwrap();
    assert!(
        page.contains("<svg") && page.contains("<circle"),
        "a loss curve with validation points"
    );
    assert!(
        page.contains("+ sgd 0.2") && page.contains("- sgd 0.1"),
        "the program's diff"
    );
    let cmp = cli_out(&["compare", &r, &r]).unwrap();
    assert!(
        cmp.contains("applied") && cmp.contains("best_validation"),
        "{cmp}"
    );
    let said = cli_out(&["say", &r, "slow", "down"]).unwrap();
    assert!(said.contains("next decision"));
    assert!(
        cli_out(&["say", dir.path().to_str().unwrap(), "x"]).is_err(),
        "not a run folder"
    );
    assert!(
        cli_out(&["show", &r, "--bogus", "1"])
            .unwrap_err()
            .contains("unknown option --bogus")
    );
}

// ── the agent on another machine ────────────────────────────────

#[test]
fn a_mailbox_carries_decisions_to_a_broker_and_back() {
    use rlx_bind::runtime::agent::mailbox::{Mailbox, serve};
    let dir = tempfile::tempdir().unwrap();
    let mbox = dir.path().join("box");
    let broker_box = mbox.clone();
    let broker = std::thread::spawn(move || {
        serve(
            &broker_box,
            || {
                let (b, _) = scripted(vec![edit(RULE, "sgd 0.4\n")]);
                Ok(b as Box<dyn Backend>)
            },
            std::time::Duration::from_millis(10),
            true,
            |_| {},
        )
        .unwrap()
    });
    let mut m = Mailbox::new(&mbox);
    m.poll = std::time::Duration::from_millis(10);
    m.timeout = Some(std::time::Duration::from_secs(30));
    let mut agent = AgentOptimizer::new(
        config(&dir.path().join("runs"), 20, 1),
        Box::new(ToyHost),
        Box::new(m),
    )
    .unwrap();
    Toyrun::new().steps(&mut agent, 1, 20).unwrap();
    assert_eq!(broker.join().unwrap(), 1);
    assert_eq!(
        agent.files()[RULE],
        "sgd 0.4\n",
        "the broker's edit came back and took over"
    );
    assert_eq!(agent.version(), 1);
    assert!(
        mbox.join("broker").read_dir().unwrap().count() == 1,
        "one workspace per run"
    );
    assert!(mbox.join("requests").read_dir().unwrap().count() == 0);
}

// ── retries and an unavailable agent ────────────────────────────

/// A stand-in CLI that fails its first `fails` calls, saying `why`, then answers.
#[cfg(unix)]
fn flaky_cli(dir: &Path, fails: usize, why: &str) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = format!(
        "#!/bin/sh\nn=$(cat \"{count}\" 2>/dev/null || echo 0)\necho $((n+1)) > \"{count}\"\n\
         if [ \"$n\" -lt {fails} ]; then\n  echo '{{\"type\":\"result\",\"is_error\":true,\"result\":\"{why}\"}}'\n  exit 1\nfi\n\
         echo '{{\"type\":\"result\",\"result\":\"Observed: ok\",\"total_cost_usd\":0.01}}'\n",
        count = dir.join("calls").display(),
    );
    let path = dir.join("flaky");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[cfg(unix)]
#[test]
fn failed_calls_are_made_again_and_an_unavailable_agent_is_waited_for() {
    for (why, fails, retries, answered) in [
        ("boom", 2, 5, true),
        ("boom", 3, 2, false),
        ("You have hit your limit · resets 3pm", 4, 0, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let cli = flaky_cli(dir.path(), fails, why);
        let mut agent = CliAgent::from_json(
            AgentCli::Claude,
            &json!({"program": cli.display().to_string(), "retries": retries,
                    "retry_pause_seconds": 0.01, "mcp_command": ["true"]}),
        )
        .unwrap();
        let out = agent.decide(&request(dir.path()));
        let calls: usize = std::fs::read_to_string(dir.path().join("calls"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(
            out.error.is_none(),
            answered,
            "{why} {fails}: {:?}",
            out.error
        );
        let expected_calls = if answered { fails + 1 } else { retries + 1 };
        assert_eq!(calls, expected_calls, "{why}: retried {} times", calls - 1);
    }
}

#[cfg(unix)]
#[test]
fn an_unavailable_agent_is_given_up_on_after_its_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let cli = flaky_cli(dir.path(), 1000, "Please run /login");
    let mut agent = CliAgent::from_json(
        AgentCli::Claude,
        &json!({"program": cli.display().to_string(), "retry_pause_seconds": 0.01,
                "unavailable_timeout_seconds": 0.1, "mcp_command": ["true"]}),
    )
    .unwrap();
    let out = agent.decide(&request(dir.path()));
    assert!(out.error.unwrap().contains("stayed unavailable"));
}

#[test]
fn codex_runs_with_its_own_home_and_the_workspace_server_as_an_override() {
    let dir = tempfile::tempdir().unwrap();
    let agent = CliAgent::from_json(
        AgentCli::Codex,
        &json!({"codex_home": "/home/u/.codex-rlx", "mcp_command": ["/bin/rlx-agent", "workspace-mcp"]}),
    )
    .unwrap();
    let (run, prompt) = agent.agent_run(&request(dir.path()));
    assert!(prompt.starts_with("SYSTEM\n\n---\n\n"));
    assert_eq!(
        run.env,
        vec![("CODEX_HOME".to_string(), "/home/u/.codex-rlx".to_string())]
    );
    let (program, args) = run.command_line(&prompt);
    assert_eq!(program, "codex");
    assert!(
        args.contains(&"mcp_servers.rlx_files.command=\"/bin/rlx-agent\"".to_string()),
        "{args:?}"
    );
    assert!(backend_from_json("codex", &json!({})).is_ok());
}

#[cfg(unix)]
#[test]
fn a_cli_that_started_without_the_workspace_tools_is_a_failed_decision() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    // As Claude Code answers when the server did not load: no tools, an answer anyway.
    let cli = dir.path().join("toolless");
    std::fs::write(
        &cli,
        "#!/bin/sh\necho '{\"type\":\"system\",\"subtype\":\"init\",\"tools\":[],\"mcp_servers\":[]}'\n\
         echo '{\"type\":\"result\",\"result\":\"I edited updater.json\",\"total_cost_usd\":0.01}'\n",
    )
    .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut agent = CliAgent::from_json(
        AgentCli::Claude,
        &json!({"program": cli.display().to_string(), "retries": 0, "mcp_command": ["true"]}),
    )
    .unwrap();
    let out = agent.decide(&request(dir.path()));
    let err = out.error.expect("not a decision");
    assert!(
        err.contains("started without the rlx_files workspace tools (it had: none)"),
        "{err}"
    );
}
