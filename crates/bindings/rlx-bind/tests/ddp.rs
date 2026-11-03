// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-parallel training, one thread per rank over an in-process transport.
//!
//! * `Trainer::set_data_parallel` starts the replicas equal and keeps them
//!   equal; each update uses the mean gradient over the ranks — the same as
//!   one process accumulating the ranks' batches.
//! * The agent over the ranks: rank 0 alone asks; every rank installs the
//!   answer at the same update; a program failing on one rank during its trial
//!   is reverted on all (exactly, at its first update); after its trial it
//!   stops the run with a crash marker.
#![cfg(feature = "agent")]

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use rlx_bind::runtime::agent::{
    AgentConfig, AgentOptimizer, Backend, DecisionOutcome, DecisionRequest, Files, Keep, ParamInfo,
    Program, ProgramHost, RuleHost, Spread, StepCtx,
};
use rlx_bind::runtime::train::{
    OptItem, ProcessGroup, Resident, StepInfo, Trainer, TrainerConfig, TrainingOptions, Transport,
};
use rlx_bind::{ApiError, ApiResult};
use rlx_driver::CollectiveError;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use serde_json::{Value, json};

// ── an in-process transport ─────────────────────────────────────

type Mailbox = Arc<(Mutex<HashMap<(u32, u32, u32), VecDeque<Vec<u8>>>>, Condvar)>;

struct Channels {
    rank: u32,
    world: u32,
    mail: Mailbox,
}

impl Transport for Channels {
    fn rank(&self) -> u32 {
        self.rank
    }
    fn world_size(&self) -> u32 {
        self.world
    }
    fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
        let (m, cv) = &*self.mail;
        m.lock()
            .unwrap()
            .entry((to, self.rank, tag))
            .or_default()
            .push_back(bytes.to_vec());
        cv.notify_all();
        Ok(())
    }
    fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        let (m, cv) = &*self.mail;
        let mut q = m.lock().unwrap();
        loop {
            if let Some(b) = q
                .get_mut(&(self.rank, from, tag))
                .and_then(VecDeque::pop_front)
            {
                return Ok(b);
            }
            q = cv.wait(q).unwrap();
        }
    }
}

fn groups(world: u32) -> Vec<ProcessGroup> {
    let mail: Mailbox = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
    (0..world)
        .map(|rank| {
            ProcessGroup::new(Arc::new(Channels {
                rank,
                world,
                mail: mail.clone(),
            }))
        })
        .collect()
}

/// Run `f(rank, group)` on one thread per rank → the results by rank.
fn per_rank<T: Send + 'static>(
    world: u32,
    f: impl Fn(u32, ProcessGroup) -> T + Send + Sync + 'static,
) -> Vec<T> {
    let f = Arc::new(f);
    let handles: Vec<_> = groups(world)
        .into_iter()
        .enumerate()
        .map(|(r, g)| {
            let f = f.clone();
            std::thread::spawn(move || f(r as u32, g))
        })
        .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

// ── the model ───────────────────────────────────────────────────

const BATCH: usize = 4;
const DIM: usize = 3;

fn regression() -> Graph {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("mse");
    let x = g.input("x", f32s(&[BATCH, DIM]));
    let t = g.input("t", f32s(&[BATCH, DIM]));
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

fn trainer(seed: f32, training: TrainingOptions) -> Trainer {
    let w: Vec<f32> = (0..DIM * DIM)
        .map(|i| 0.1 * (i % 7) as f32 - 0.2 + seed)
        .collect();
    Trainer::new(
        regression(),
        TrainerConfig {
            wrt: vec!["w".into(), "b".into()],
            init: vec![("w".into(), w), ("b".into(), vec![seed; DIM])],
            device: "cpu".into(),
            optimizer: json!({"kind": "sgd", "lr": 0.1}),
            policy: None,
            clip_norm: None,
            resident: Resident::Off,
            training,
        },
    )
    .unwrap()
}

/// The batch rank `r` sees at step `i`.
fn batch(r: usize, i: usize) -> (Vec<f32>, Vec<f32>) {
    let s = 7 * i + 3 * r;
    let x = (0..BATCH * DIM)
        .map(|k| (((k + s) * 5 % 11) as f32 - 5.0) / 5.0)
        .collect();
    let t = (0..BATCH * DIM)
        .map(|k| (((k + 2 * s) * 3 % 7) as f32 - 3.0) / 3.0)
        .collect();
    (x, t)
}

#[test]
fn data_parallel_replicas_stay_equal_and_match_one_process_accumulating() {
    let ranks = per_rank(2, |r, g| {
        // Different starting values: rank 0's are broadcast.
        let mut t = trainer(r as f32, TrainingOptions::default());
        t.set_data_parallel(g).unwrap();
        for i in 0..10 {
            let (x, y) = batch(r as usize, i);
            t.step(&[("x", x.as_slice()), ("t", y.as_slice())]).unwrap();
        }
        assert!(t.replicas_agree().unwrap());
        t.params()
    });
    assert_eq!(ranks[0], ranks[1], "the replicas never drift");
    // One process accumulating both ranks' batches makes the same updates.
    let mut one = trainer(
        0.0,
        TrainingOptions {
            accumulate: 2,
            ..Default::default()
        },
    );
    for i in 0..10 {
        for r in 0..2 {
            let (x, y) = batch(r, i);
            one.step(&[("x", x.as_slice()), ("t", y.as_slice())])
                .unwrap();
        }
    }
    for ((_, a), (_, b)) in one.params().iter().zip(&ranks[0]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}

// ── the agent over the ranks ────────────────────────────────────

/// Writes `updater.json` with `text` on its first decision, counting calls.
struct Writes {
    text: &'static str,
    calls: Arc<Mutex<usize>>,
}

impl Backend for Writes {
    fn name(&self) -> &str {
        "writes"
    }
    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        let mut n = self.calls.lock().unwrap();
        *n += 1;
        if *n == 1 {
            std::fs::write(req.workspace.join("updater.json"), self.text).unwrap();
        }
        DecisionOutcome::default()
    }
}

fn config(root: &Path, total: u64, decisions: usize) -> AgentConfig {
    let mut c = AgentConfig::new("regression; judged by the final loss", total);
    c.run_root = root.to_path_buf();
    c.decisions = Some(decisions);
    c.spread = Spread::Uniform;
    c.every = 5;
    c.trial_steps = 4;
    c
}

const FASTER: &str = r#"{"groups": [{"names": ["*"], "optimizer": {"kind": "sgd", "lr": 0.3}}]}"#;

#[test]
fn rank_zero_asks_and_every_rank_installs_the_answer_at_the_same_update() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    let calls: Vec<Arc<Mutex<usize>>> = (0..2).map(|_| Arc::new(Mutex::new(0))).collect();
    let calls_for = calls.clone();
    let ranks = per_rank(2, move |r, g| {
        let opt = json!({"kind": "sgd", "lr": 0.1});
        let mut t = trainer(0.0, TrainingOptions::default());
        t.set_data_parallel(g).unwrap();
        let backend: Box<dyn Backend> = Box::new(Writes {
            text: FASTER,
            calls: calls_for[r as usize].clone(),
        });
        let mut agent = AgentOptimizer::new(
            config(&root_path.join(format!("r{r}")), 30, 2),
            Box::new(RuleHost::new(&opt).unwrap()),
            backend,
        )
        .unwrap();
        for i in 0..30 {
            let (x, y) = batch(r as usize, i);
            agent
                .train_step(&mut t, &[("x", x.as_slice()), ("t", y.as_slice())])
                .unwrap();
        }
        assert_eq!(agent.rank(), r);
        let applied: Vec<u64> = events(agent.run_dir())
            .iter()
            .filter(|e| e["kind"] == "applied")
            .filter_map(|e| e["update"].as_u64())
            .collect();
        (
            agent.version(),
            applied,
            t.params(),
            agent.files()["updater.json"].clone(),
        )
    });
    assert_eq!(*calls[0].lock().unwrap(), 2, "rank 0 asked twice");
    assert_eq!(*calls[1].lock().unwrap(), 0, "rank 1 never asks");
    assert_eq!(ranks[0].0, 1);
    assert_eq!(ranks[0].0, ranks[1].0);
    assert_eq!(ranks[0].1, vec![10], "applied at the decision's update");
    assert_eq!(ranks[0].1, ranks[1].1, "on every rank at the same update");
    assert_eq!(ranks[0].2, ranks[1].2, "the replicas stay equal");
    assert_eq!(ranks[1].3, FASTER);
}

/// `rule.txt` holds `sgd <lr>` or `sgd <lr> fail_at <n> on <rank>`: the n-th
/// update of that program fails on that rank only.
struct Toy {
    rank: u32,
}

struct ToyProgram {
    lr: f32,
    fail: Option<(u64, u32)>,
    rank: u32,
    made: u64,
}

impl ProgramHost for Toy {
    fn starting_files(&self) -> Files {
        Files::from([("rule.txt".to_string(), "sgd 0.1".to_string())])
    }
    fn build(
        &mut self,
        files: &Files,
        _: &[ParamInfo],
        _: Option<&dyn Program>,
    ) -> ApiResult<Box<dyn Program>> {
        let w: Vec<&str> = files["rule.txt"].split_whitespace().collect();
        let (lr, fail) = match w.as_slice() {
            ["sgd", lr] => (lr.parse().unwrap(), None),
            ["sgd", lr, "fail_at", n, "on", r] => (
                lr.parse().unwrap(),
                Some((n.parse().unwrap(), r.parse().unwrap())),
            ),
            _ => return Err(ApiError::type_("rule.txt: cannot read")),
        };
        Ok(Box::new(ToyProgram {
            lr,
            fail,
            rank: self.rank,
            made: 0,
        }))
    }
    fn interface(&self) -> String {
        String::new()
    }
}

impl Program for ToyProgram {
    fn step(&mut self, items: &mut [OptItem<'_>], _: &StepCtx) -> ApiResult<bool> {
        self.made += 1;
        if self.fail == Some((self.made, self.rank)) {
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

/// `w` under ½|w|², with the gradient the params themselves (equal on every rank).
fn run_toy(agent: &mut AgentOptimizer, w: &mut Vec<f32>, from: u64, to: u64) -> ApiResult<()> {
    for u in from..=to {
        let g = w.clone();
        let loss = 0.5 * w.iter().map(|x| x * x).sum::<f32>();
        let mut items = [OptItem {
            name: "w",
            shape: &[3],
            param: w,
            grad: &g,
        }];
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

fn events(run: &Path) -> Vec<Value> {
    std::fs::read_to_string(run.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn a_program_failing_on_one_rank_in_its_trial_is_reverted_exactly_on_all() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    let ranks = per_rank(2, move |r, g| {
        let backend: Box<dyn Backend> = if r == 0 {
            Box::new(Writes {
                text: "",
                calls: Arc::new(Mutex::new(0)),
            })
        } else {
            Box::new(Keep)
        };
        // Rank 0's backend writes rule.txt (not updater.json): use a tiny shim.
        struct Rule(Box<dyn Backend>);
        impl Backend for Rule {
            fn name(&self) -> &str {
                "rule"
            }
            fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
                std::fs::write(req.workspace.join("rule.txt"), "sgd 0.5 fail_at 1 on 1").unwrap();
                self.0.decide(req)
            }
        }
        let mut agent = AgentOptimizer::new(
            config(&root_path.join(format!("r{r}")), 30, 1),
            Box::new(Toy { rank: r }),
            Box::new(Rule(backend)),
        )
        .unwrap();
        agent.set_process_group(g);
        let mut w = vec![1.0f32, -2.0, 0.5];
        run_toy(&mut agent, &mut w, 1, 30).unwrap();
        let reverted: Vec<Value> = events(agent.run_dir())
            .into_iter()
            .filter(|e| e["kind"] == "reverted")
            .collect();
        (w, reverted, agent.version())
    });
    // As if the failing program had never run: plain sgd 0.1 throughout, on both ranks.
    let mut expect = vec![1.0f32, -2.0, 0.5];
    for _ in 0..30 {
        expect.iter_mut().for_each(|p| *p -= 0.1 * *p);
    }
    for (w, reverted, version) in &ranks {
        assert_eq!(w, &expect);
        assert_eq!(reverted.len(), 1);
        assert_eq!(reverted[0]["update"], 15);
        assert_eq!(reverted[0]["exact"], true);
        assert_eq!(*version, 2);
    }
    assert!(
        ranks[0].1[0]["error"]
            .as_str()
            .unwrap()
            .contains("failed on another rank")
    );
    assert!(
        ranks[1].1[0]["error"]
            .as_str()
            .unwrap()
            .contains("toy rule failed")
    );
}

#[test]
fn after_its_trial_a_failure_on_one_rank_stops_the_run_with_a_crash_marker() {
    let root = tempfile::tempdir().unwrap();
    let root_path = root.path().to_path_buf();
    let ranks = per_rank(2, move |r, g| {
        struct Rule;
        impl Backend for Rule {
            fn name(&self) -> &str {
                "rule"
            }
            fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
                std::fs::write(req.workspace.join("rule.txt"), "sgd 0.2 fail_at 8 on 1").unwrap();
                DecisionOutcome::default()
            }
        }
        let mut agent = AgentOptimizer::new(
            config(&root_path.join(format!("r{r}")), 30, 1),
            Box::new(Toy { rank: r }),
            Box::new(Rule),
        )
        .unwrap();
        agent.set_process_group(g);
        let mut w = vec![1.0f32, -2.0, 0.5];
        let out = run_toy(&mut agent, &mut w, 1, 30);
        (out.err().map(|e| e.message), agent.run_dir().to_path_buf())
    });
    assert!(ranks[0].0.is_none(), "rank 0 did not fail");
    let err = ranks[1].0.as_deref().unwrap();
    assert!(
        err.contains("after its trial") && err.contains("on rank 1"),
        "{err}"
    );
    assert!(ranks[1].1.join("crashed-v001-r1.json").exists());
}
