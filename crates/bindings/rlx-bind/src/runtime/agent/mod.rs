// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A coding agent in the training loop, rewriting the optimizer while the
//! model trains — the protocol of OPAL (*Reactive Optimizers*, Bai et al.,
//! arXiv:2610.33826) over [`Trainer::step_with`].
//!
//! [`AgentOptimizer`] makes every update with the current [`Program`]. At
//! scheduled decisions (and when the loss turns non-finite, or the agent's own
//! analysis asks) it writes what it has seen into a workspace folder and asks
//! a [`Backend`] — an agent CLI such as Claude Code ([`CliAgent`], through
//! `rlx-agent`), a broker on another machine ([`Mailbox`]), or [`Keep`] for
//! no agent — which answers by editing the files there: the updater, built by
//! the [`ProgramHost`], and `analysis.json` ([`Analysis`]), which decides what
//! the reports hold. A changed updater takes over at an update boundary, then
//! runs on trial: a failure at its first update is undone exactly (from a copy
//! of the params taken before it ran) and made by the previous program; a
//! later failure in the trial goes back to the previous program without
//! undoing that update; after the trial, the previous program is rebuilt from
//! its files.
//!
//! Besides the training loss the agent sees what the training script shows
//! it: validation results ([`AgentOptimizer::report_validation`]), chosen
//! values ([`AgentOptimizer::log`]), the run's facts and read-only copies of
//! its code, and notes the person running it leaves ([`records::say`]).
//!
//! ```no_run
//! # use rlx_bind::runtime::agent::*;
//! # use rlx_bind::runtime::train::Trainer;
//! # fn demo(trainer: &mut Trainer, batches: &[Vec<(&str, &[f32])>]) -> rlx_bind::ApiResult<()> {
//! let opt = serde_json::json!({"kind": "adamw", "lr": 3e-4});
//! let cfg = AgentConfig::new("MLP on MNIST; judged by test accuracy at the end", 10_000);
//! let mut agent = AgentOptimizer::new(cfg, Box::new(RuleHost::new(&opt)?), Box::new(Keep))?;
//! for batch in batches {
//!     agent.train_step(trainer, batch)?;
//! }
//! agent.close()?;
//! # Ok(()) }
//! ```
//!
//! Everything lands in one run folder (`run_root/<id>/`): `workspace/` (the
//! agent's folder: its program, `inbox/`, `reference/`), `programs/vNNN/`
//! (every version that ran), `decisions/NNN/` (each message, response and
//! transcript), `events.jsonl`, `losses.csv`, `validation.jsonl`,
//! `metrics.jsonl`, `notes/`, `FINAL` (the version running at the end),
//! `summary.json` and `results.json`. [`AgentOptimizer::save`] /
//! [`AgentOptimizer::resume`] checkpoint the agent's side of a run (the
//! trainer saves its own params); [`AgentOptimizer::replay`] runs a finished
//! run's program again without an agent. The `rlx-agent-opt` command reads
//! run folders (`show`, `watch`, `report`, `compare`, …).
//!
//! Host path only: the program sees host `f32` arrays.

pub mod analysis;
pub mod backend;
pub mod cli;
pub mod mailbox;
mod message;
pub mod observe;
pub mod program;
pub mod records;
pub mod schedule;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use rlx_optim::OptimizerState;
use serde_json::{Value, json};

pub use analysis::{ANALYSIS, Analysis};
pub use backend::{Backend, CliAgent, DecisionOutcome, DecisionRequest, Keep, backend_from_json};
pub use mailbox::Mailbox;
pub use message::{FIELDS as PROMPT_FIELDS, Prompt};
pub use observe::Integrity;
pub use program::{Files, ParamInfo, Program, ProgramHost, RuleHost, StepCtx};
pub use records::{final_program, redact, say};
pub use rlx_agent::{self, AgentCli};
pub use schedule::{Spread, decision_updates};

use records::Run;

use super::opts::Opts;
use super::train::{OptItem, ProcessGroup, ReduceKind, StepInfo, Trainer};
use crate::panics::catch;
use crate::{ApiError, ApiResult};

/// The most decisions an estimated schedule asks for.
pub const MAX_ESTIMATED_DECISIONS: usize = 50;
/// Skips in a row after which the update is made anyway and a decision asked.
pub const MAX_SKIPS_IN_A_ROW: usize = 10;

/// How many decisions fit when each takes `sec_per_decision` and the agent may
/// add at most `max_wait_ratio` × the run's own training time — at least one,
/// at most [`MAX_ESTIMATED_DECISIONS`].
pub fn estimate_decisions(
    total: u64,
    sec_per_step: f64,
    sec_per_decision: f64,
    max_wait_ratio: f64,
) -> usize {
    let fit = max_wait_ratio * total as f64 * sec_per_step / sec_per_decision.max(1e-9);
    (fit.floor().max(0.0) as usize).clamp(1, MAX_ESTIMATED_DECISIONS)
}

/// Whether training waits for the agent's answers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Waiting {
    /// At every decision (the default).
    #[default]
    Always,
    /// Never; edits take effect when they arrive (a non-finite loss still
    /// waits when `pause_on_nonfinite`).
    Never,
    /// When an answer would otherwise land more than `max_lag_updates` late,
    /// judged from the time per update and recent answer times; always at the
    /// first decision.
    Auto,
}

/// Facts about the run, given to the agent in plain sentences. Each is
/// optional; a missing one is left out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TaskFacts {
    pub epochs: Option<f64>,
    pub train_size: Option<u64>,
    pub batch_size: Option<u64>,
    /// Whether every epoch passes over the data in the same order.
    pub same_order_each_epoch: Option<bool>,
    /// Anything else, in the words of the person running the training.
    pub notes: Option<String>,
}

impl TaskFacts {
    /// The sentence(s), starting with a space; "" when nothing is known.
    fn sentence(&self) -> String {
        let mut s = String::new();
        match (self.epochs, self.train_size) {
            (Some(e), Some(n)) => s += &format!(", about {e} epochs over a training set of {n}"),
            (Some(e), None) => s += &format!(", about {e} epochs"),
            (None, Some(n)) => s += &format!(", over a training set of {n}"),
            _ => {}
        }
        if let Some(b) = self.batch_size {
            s += &format!(", with a batch of {b} per update");
        }
        match self.same_order_each_epoch {
            Some(true) => s += "; each epoch passes over the same data in the same order",
            Some(false) => s += "; each epoch passes over the data in a new order",
            None => {}
        }
        let mut out = if s.is_empty() {
            String::new()
        } else {
            format!(" The run{s}.")
        };
        if let Some(n) = self.notes.as_deref().filter(|n| !n.trim().is_empty()) {
            out += &format!("\nFrom the person running the training: {}", n.trim());
        }
        out
    }
}

/// How a run with an agent is set up. [`AgentConfig::new`] gives the defaults.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// One sentence: what is trained and what it is judged by at the end.
    pub goal: String,
    pub total_steps: u64,
    /// Scheduled decisions; `None` (the default) estimates them from the time
    /// of the first updates ([`estimate_decisions`]).
    pub decisions: Option<usize>,
    /// For the estimate and `Waiting::Auto`: seconds one decision takes until
    /// measured. Default 40.
    pub seconds_per_decision: f64,
    /// For the estimate: the agent's time as a multiple of training's. Default 4.
    pub max_wait_ratio: f64,
    pub spread: Spread,
    /// Updates between checks (loss read, non-finite loss caught). Default 50;
    /// also the starting analysis's report interval.
    pub every: u64,
    pub waiting: Waiting,
    /// For `Waiting::Auto`: how late, in updates, an answer may land. Default 100.
    pub max_lag_updates: u64,
    /// A new program's trial window, in updates. Default 20.
    pub trial_steps: u64,
    /// The largest params copy kept to undo a failing first update. Default 2 GiB.
    pub snapshot_limit_bytes: usize,
    /// Stop asking once the agent has cost this much.
    pub max_cost_usd: Option<f64>,
    /// Where run folders go. Default `rlx_runs`.
    pub run_root: PathBuf,
    /// A non-finite loss waits for the agent even when not `Waiting::Always`.
    pub pause_on_nonfinite: bool,
    /// The training script evaluates a held-out validation set about every N
    /// updates and reports it ([`AgentOptimizer::report_validation`]).
    pub validation: Option<u64>,
    /// The validation set's size, for the prompt.
    pub validation_size: Option<u64>,
    /// Flag programs that move the params away for an evaluation and back
    /// ([`Integrity`]). Default: on with `validation`.
    pub eval_integrity: Option<bool>,
    /// For the integrity check: how many times the run's median move counts as
    /// "far larger". Default 10.
    pub integrity_threshold: f64,
    pub facts: TaskFacts,
    /// Files copied read-only into the agent's `reference/code/` (the training
    /// script, the model's source).
    pub share: Vec<PathBuf>,
    pub prompt: Prompt,
}

/// Updates timed before an estimated schedule is set.
const TIMED_UPDATES: u64 = 10;
/// Shared files larger than this are left out (and the agent told).
const MAX_SHARED_BYTES: u64 = 300_000;

impl AgentConfig {
    pub fn new(goal: impl Into<String>, total_steps: u64) -> Self {
        Self {
            goal: goal.into(),
            total_steps,
            decisions: None,
            seconds_per_decision: 40.0,
            max_wait_ratio: 4.0,
            spread: Spread::Front,
            every: 50,
            waiting: Waiting::Always,
            max_lag_updates: 100,
            trial_steps: 20,
            snapshot_limit_bytes: 2 << 30,
            max_cost_usd: None,
            run_root: PathBuf::from("rlx_runs"),
            pause_on_nonfinite: true,
            validation: None,
            validation_size: None,
            eval_integrity: None,
            integrity_threshold: 10.0,
            facts: TaskFacts::default(),
            share: Vec::new(),
            prompt: Prompt::Neutral,
        }
    }

    /// `{goal, total_steps, decisions, seconds_per_decision, max_wait_ratio,
    /// spread ("front" | "uniform"), every, blocking (true | false | "auto"),
    /// max_lag_updates, trial_steps, snapshot_limit_gb, max_cost_usd,
    /// run_root, pause_on_nonfinite, validation, validation_size,
    /// eval_integrity, integrity_threshold, epochs, train_size, batch_size,
    /// same_order_each_epoch, task_facts, share ([paths]), prompt ("neutral" |
    /// "aggressive" | a template path)}`; `goal` and `total_steps` are required.
    pub fn from_json(v: &Value) -> ApiResult<Self> {
        let o = Opts::new("agent", v)?;
        let goal = o
            .str("goal")?
            .filter(|g| !g.trim().is_empty())
            .ok_or_else(|| {
                ApiError::type_("agent: `goal` is required: what is trained and how it is judged")
            })?;
        let total = o
            .u64("total_steps")?
            .ok_or_else(|| ApiError::type_("agent: `total_steps` is required"))?;
        let mut c = Self::new(goal, total);
        c.decisions = o.u64("decisions")?.map(|n| n as usize);
        c.seconds_per_decision = o
            .f64("seconds_per_decision")?
            .unwrap_or(c.seconds_per_decision);
        c.max_wait_ratio = o.f64("max_wait_ratio")?.unwrap_or(c.max_wait_ratio);
        if let Some(s) = o.str("spread")? {
            c.spread = Spread::parse(s).ok_or_else(|| {
                ApiError::type_(format!(
                    "agent: spread is \"front\" or \"uniform\", got {s:?}"
                ))
            })?;
        }
        c.every = o.u64("every")?.unwrap_or(c.every);
        c.waiting = match o.get("blocking") {
            None => c.waiting,
            Some(Value::Bool(true)) => Waiting::Always,
            Some(Value::Bool(false)) => Waiting::Never,
            Some(Value::String(s)) if s == "auto" => Waiting::Auto,
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "agent: blocking is true, false or \"auto\", got {other}"
                )));
            }
        };
        c.max_lag_updates = o.u64("max_lag_updates")?.unwrap_or(c.max_lag_updates);
        c.trial_steps = o.u64("trial_steps")?.unwrap_or(c.trial_steps);
        if let Some(gb) = o.f64("snapshot_limit_gb")? {
            c.snapshot_limit_bytes = (gb.max(0.0) * (1u64 << 30) as f64) as usize;
        }
        c.max_cost_usd = o.f64("max_cost_usd")?;
        if let Some(r) = o.str("run_root")? {
            c.run_root = PathBuf::from(r);
        }
        c.pause_on_nonfinite = o.bool("pause_on_nonfinite", c.pause_on_nonfinite)?;
        c.validation = o.u64("validation")?;
        c.validation_size = o.u64("validation_size")?;
        c.eval_integrity = match o.get("eval_integrity") {
            Some(Value::Bool(b)) => Some(*b),
            None => None,
            Some(other) => {
                return Err(ApiError::type_(format!(
                    "agent: eval_integrity is a boolean, got {other}"
                )));
            }
        };
        c.integrity_threshold = o
            .f64("integrity_threshold")?
            .unwrap_or(c.integrity_threshold);
        c.facts = TaskFacts {
            epochs: o.f64("epochs")?,
            train_size: o.u64("train_size")?,
            batch_size: o.u64("batch_size")?,
            same_order_each_epoch: match o.get("same_order_each_epoch") {
                Some(Value::Bool(b)) => Some(*b),
                _ => None,
            },
            notes: o.str("task_facts")?.map(str::to_string),
        };
        if let Some(share) = o.get("share") {
            c.share = share
                .as_array()
                .and_then(|a| a.iter().map(|p| p.as_str().map(PathBuf::from)).collect())
                .ok_or_else(|| ApiError::type_("agent: share is a list of file paths"))?;
        }
        if let Some(p) = o.str("prompt")? {
            c.prompt = Prompt::parse(p);
        }
        o.finish()?;
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> ApiResult<()> {
        if self.goal.trim().is_empty() {
            return Err(ApiError::type_(
                "agent: `goal` is required: what is trained and how it is judged",
            ));
        }
        if self.total_steps == 0 || self.every == 0 || self.validation == Some(0) {
            return Err(ApiError::range(
                "agent: total_steps, every and validation must be at least 1",
            ));
        }
        if !(self.seconds_per_decision > 0.0 && self.max_wait_ratio > 0.0) {
            return Err(ApiError::range(
                "agent: seconds_per_decision and max_wait_ratio must be positive",
            ));
        }
        self.prompt.template()?;
        Ok(())
    }

    fn integrity_on(&self) -> bool {
        self.eval_integrity.unwrap_or(self.validation.is_some())
    }
}

/// A decision in flight.
struct InFlight {
    k: usize,
    requested_at: u64,
    paused: bool,
    base: Files,
    handle: JoinHandle<DecisionOutcome>,
}

/// An answered decision, ready to apply.
struct Answer {
    k: usize,
    requested_at: u64,
    base: Files,
    /// The edited files; `None` when nothing usable changed.
    files: Option<Files>,
    failed: Option<String>,
}

/// The window after a swap in which a failure goes back to `previous`.
struct Trial {
    until: u64,
    /// The new program has not made an update yet.
    first_pending: bool,
    previous: Box<dyn Program>,
    /// The params before the new program's first update; `Err` says why there is none.
    snapshot: Result<Vec<Vec<f32>>, String>,
}

#[derive(Default)]
struct Counts {
    applied: usize,
    kept: usize,
    rejected: usize,
    reverted: usize,
    failed: usize,
    skipped: usize,
}

/// What a checkpoint restores once the params are known (at the first update).
struct Restore {
    state: Option<OptimizerState>,
    trial_open: bool,
}

const MAX_REPORTS: usize = 400;
const MAX_SHOWN: usize = 200;
/// The agent's own notes, kept in its workspace for the whole run.
const NOTES: &str = "NOTES.md";
const CHECKPOINT: &str = "agent.json";
const CHECKPOINT_STATE: &str = "agent_state.bin";

/// The optimizer of a run with an agent in the loop.
pub struct AgentOptimizer {
    cfg: AgentConfig,
    host: Box<dyn ProgramHost>,
    backend: Arc<Mutex<Box<dyn Backend>>>,
    backend_name: String,
    run: Run,
    template: String,
    /// `None` until an estimated schedule is set.
    schedule: Option<Vec<u64>>,
    max_decisions: usize,
    sched_i: usize,
    asked: usize,
    params: Vec<ParamInfo>,
    system: String,
    program: Option<Box<dyn Program>>,
    /// The program's files: the host's updater files and `analysis.json`.
    files: Files,
    analysis: Analysis,
    version: u32,
    /// The files of the last updater that passed its trial — the fallback
    /// once the running one's own trial is over.
    fallback: Option<Files>,
    trial: Option<Trial>,
    inflight: Option<InFlight>,
    restore: Option<Restore>,
    base_lr: Option<f32>,
    update: u64,
    // since the previous decision
    losses: Vec<f32>,
    losses_from: u64,
    reports: Vec<Value>,
    errors: Vec<Value>,
    reasons: Vec<String>,
    previous: Option<String>,
    shown: Vec<Value>,
    skipped: Vec<u64>,
    // since the previous check / report
    window: Vec<f32>,
    report_window: Vec<f32>,
    loss_log: Vec<(u64, f32)>,
    nonfinite_episode: bool,
    /// The first update with a non-finite loss since the previous check.
    nonfinite_at: Option<u64>,
    skips_in_a_row: usize,
    // what the training script reported
    validation: Vec<Value>,
    integrity: Option<Integrity>,
    recent_losses: Vec<f32>,
    // timing
    program_ms: Option<f64>,
    start_ms: Option<f64>,
    last_entry: Option<Instant>,
    last_wait: f64,
    step_seconds: Vec<f64>,
    decision_seconds: Option<f64>,
    cost: f64,
    budget_spent_noted: bool,
    counts: Counts,
    /// Data-parallel: rank 0 asks the agent; every rank follows.
    group: Option<ProcessGroup>,
    schedule_shared: bool,
}

/// The tag of rank 0's plans to the other ranks (below the collectives' tags).
const TAG_PLAN: u32 = 0x0A6E_0001;

fn collective(e: rlx_driver::CollectiveError) -> ApiError {
    ApiError::runtime(format!("agent: data-parallel collective: {e:?}"))
}

impl Answer {
    fn to_json(&self) -> Value {
        json!({"k": self.k, "requested_at": self.requested_at, "base": self.base,
               "files": self.files, "failed": self.failed})
    }

    fn from_json(v: &Value) -> ApiResult<Self> {
        let files = |v: &Value| -> ApiResult<Option<Files>> {
            if v.is_null() {
                return Ok(None);
            }
            serde_json::from_value(v.clone())
                .map(Some)
                .map_err(|e| ApiError::runtime(format!("agent: a shared answer: {e}")))
        };
        Ok(Self {
            k: v["k"].as_u64().unwrap_or(0) as usize,
            requested_at: v["requested_at"].as_u64().unwrap_or(0),
            base: files(&v["base"])?.unwrap_or_default(),
            files: files(&v["files"])?,
            failed: v["failed"].as_str().map(str::to_string),
        })
    }
}

impl AgentOptimizer {
    /// Create the run folder; the starting program is built at the first
    /// update, once the params are known.
    pub fn new(
        cfg: AgentConfig,
        host: Box<dyn ProgramHost>,
        backend: Box<dyn Backend>,
    ) -> ApiResult<Self> {
        cfg.check()?;
        let run = Run::create(&cfg.run_root)?;
        let agent = Self::with_run(cfg, host, backend, run)?;
        agent.run.event(
            "started",
            json!({"goal": agent.cfg.goal, "total_steps": agent.cfg.total_steps,
                   "schedule": agent.schedule, "backend": agent.backend_name,
                   "waiting": format!("{:?}", agent.cfg.waiting)}),
        );
        agent.share_code();
        Ok(agent)
    }

    /// A run whose starting program is `program` (a finished run folder, whose
    /// `FINAL` version is taken, or a `programs/vNNN` folder) with no agent:
    /// the same updates again, to check a result or train another seed.
    pub fn replay(
        mut cfg: AgentConfig,
        host: Box<dyn ProgramHost>,
        program: &Path,
    ) -> ApiResult<Self> {
        let (version, mut files) = final_program(program)?;
        cfg.decisions = Some(0);
        cfg.check()?;
        let run = Run::create(&cfg.run_root)?;
        let mut a = Self::with_run(cfg, host, Box::new(Keep), run)?;
        let analysis = files
            .entry(ANALYSIS.into())
            .or_insert_with(|| Analysis::starting_text(a.cfg.every))
            .clone();
        a.analysis = Analysis::parse(&analysis)?;
        a.files = files;
        a.run.event(
            "started",
            json!({"goal": a.cfg.goal, "total_steps": a.cfg.total_steps, "backend": "keep",
                   "replay_of": program.display().to_string(), "replay_version": version}),
        );
        Ok(a)
    }

    fn with_run(
        cfg: AgentConfig,
        host: Box<dyn ProgramHost>,
        backend: Box<dyn Backend>,
        run: Run,
    ) -> ApiResult<Self> {
        let template = cfg.prompt.template()?;
        let schedule = cfg
            .decisions
            .map(|n| decision_updates(n, cfg.total_steps, cfg.spread));
        let max_decisions = cfg.decisions.unwrap_or(MAX_ESTIMATED_DECISIONS);
        let backend_name = backend.name().to_string();
        let mut files = host.starting_files();
        let analysis_text = Analysis::starting_text(cfg.every);
        let analysis = Analysis::parse(&analysis_text).expect("the starting analysis parses");
        files.insert(ANALYSIS.into(), analysis_text);
        let integrity = cfg
            .integrity_on()
            .then(|| Integrity::new(cfg.integrity_threshold));
        Ok(Self {
            host,
            backend: Arc::new(Mutex::new(backend)),
            backend_name,
            run,
            template,
            schedule,
            max_decisions,
            sched_i: 0,
            asked: 0,
            params: Vec::new(),
            system: String::new(),
            program: None,
            files,
            analysis,
            version: 0,
            fallback: None,
            trial: None,
            inflight: None,
            restore: None,
            base_lr: None,
            update: 0,
            losses: Vec::new(),
            losses_from: 1,
            reports: Vec::new(),
            errors: Vec::new(),
            reasons: Vec::new(),
            previous: None,
            shown: Vec::new(),
            skipped: Vec::new(),
            window: Vec::new(),
            report_window: Vec::new(),
            loss_log: Vec::new(),
            nonfinite_episode: false,
            nonfinite_at: None,
            skips_in_a_row: 0,
            validation: Vec::new(),
            integrity,
            recent_losses: Vec::new(),
            program_ms: None,
            start_ms: None,
            last_entry: None,
            last_wait: 0.0,
            step_seconds: Vec::new(),
            decision_seconds: None,
            cost: 0.0,
            budget_spent_noted: false,
            counts: Counts::default(),
            group: None,
            schedule_shared: false,
            cfg,
        })
    }

    /// Copy the shared files into `reference/code/`.
    fn share_code(&self) {
        for path in &self.cfg.share {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "file".into());
            let size = fs::metadata(path).map(|m| m.len());
            let copied = match size {
                Ok(n) if n <= MAX_SHARED_BYTES => fs::read_to_string(path)
                    .ok()
                    .and_then(|t| {
                        self.run
                            .write(format!("workspace/reference/code/{name}"), &t)
                            .ok()
                    })
                    .is_some(),
                _ => false,
            };
            self.run.event(
                "shared",
                json!({"path": path.display().to_string(), "copied": copied,
                       "bytes": size.ok()}),
            );
        }
    }

    /// One training step through `trainer`, the update made by the agent's
    /// program → the loss.
    pub fn train_step(
        &mut self,
        trainer: &mut Trainer,
        inputs: &[(&str, &[f32])],
    ) -> ApiResult<f32> {
        if self.group.is_none()
            && let Some(g) = trainer.process_group()
        {
            self.group = Some(g.clone());
        }
        trainer.step_with(inputs, |items, info| self.step(items, info))
    }

    /// Run the protocol over the ranks of a data-parallel run (as OPAL does
    /// over DDP): rank 0 alone asks the agent; at every check it sends the
    /// others the schedule and any answer, so every rank installs the same
    /// program at the same update; a build must succeed on every rank; during
    /// a trial the ranks agree after each update on whether it failed, and a
    /// trial that makes the ranks' params differ is reverted (rank 0's params
    /// are sent to all). After its trial, a program that fails on a rank stops
    /// the run with a crash marker (`crashed-v<N>-r<rank>.json`); resuming then
    /// starts from the program before it. [`Self::train_step`] takes the
    /// trainer's group when it has one ([`Trainer::set_data_parallel`]).
    pub fn set_process_group(&mut self, group: ProcessGroup) {
        self.group = Some(group);
    }

    /// This process's rank (0 without a group).
    pub fn rank(&self) -> u32 {
        self.group.as_ref().map_or(0, |g| g.rank())
    }

    fn rank0(&self) -> bool {
        self.rank() == 0
    }

    fn multi(&self) -> bool {
        self.group.as_ref().is_some_and(|g| g.world_size() > 1)
    }

    /// Rank 0's `value` on every rank.
    fn share(&self, value: Value) -> ApiResult<Value> {
        let Some(g) = self.group.as_ref().filter(|g| g.world_size() > 1) else {
            return Ok(value);
        };
        let t = g.transport();
        if g.rank() == 0 {
            let bytes = value.to_string().into_bytes();
            for r in 1..g.world_size() {
                t.send_bytes(r, TAG_PLAN, &bytes).map_err(collective)?;
            }
            Ok(value)
        } else {
            let bytes = t.recv_bytes(0, TAG_PLAN).map_err(collective)?;
            serde_json::from_slice(&bytes)
                .map_err(|e| ApiError::runtime(format!("agent: rank 0's plan: {e}")))
        }
    }

    /// True on every rank only when `ok` on every rank.
    fn agree(&self, ok: bool) -> ApiResult<bool> {
        let Some(g) = self.group.as_ref().filter(|g| g.world_size() > 1) else {
            return Ok(ok);
        };
        let mut v = [if ok { 1.0f32 } else { 0.0 }];
        g.all_reduce(&mut v, ReduceKind::Min).map_err(collective)?;
        Ok(v[0] == 1.0)
    }

    /// Whether every rank holds the same params (a strided sample, by min and max).
    fn params_agree(&self, items: &[OptItem<'_>]) -> ApiResult<bool> {
        let Some(g) = self.group.as_ref().filter(|g| g.world_size() > 1) else {
            return Ok(true);
        };
        let mut lo: Vec<f32> = items
            .iter()
            .flat_map(|it| {
                it.param
                    .iter()
                    .step_by(it.param.len().div_ceil(256).max(1))
                    .copied()
            })
            .collect();
        let mut hi = lo.clone();
        g.all_reduce(&mut lo, ReduceKind::Min).map_err(collective)?;
        g.all_reduce(&mut hi, ReduceKind::Max).map_err(collective)?;
        Ok(lo.iter().zip(&hi).all(|(a, b)| a.to_bits() == b.to_bits()))
    }

    /// Rank 0's params on every rank.
    fn resync(&self, items: &mut [OptItem<'_>]) -> ApiResult<()> {
        if let Some(g) = self.group.as_ref().filter(|g| g.world_size() > 1) {
            for it in items.iter_mut() {
                g.broadcast(0, it.param).map_err(collective)?;
            }
        }
        Ok(())
    }

    /// Make one update on `items` — the body of a [`Trainer::step_with`]
    /// closure. Errors only when no program can make it.
    pub fn step(&mut self, items: &mut [OptItem<'_>], info: &StepInfo<'_>) -> ApiResult<()> {
        let update = info.update;
        let entry = Instant::now();
        if let Some(t) = self.last_entry {
            // The training step since the previous call, without the agent's waits.
            self.step_seconds
                .push((entry - t).as_secs_f64() - self.last_wait);
            if self.step_seconds.len() > 1000 {
                self.step_seconds.remove(0);
            }
        }
        self.last_entry = Some(entry);
        self.update = update;
        if let Some(lr) = info.lr {
            // The trainer's schedule (or its set_lr): programs scale from it.
            self.base_lr = Some(lr);
        }
        self.record_loss(update, info.loss);
        if self.program.is_none() {
            self.start(items)?;
        }
        // With several ranks, rank 0 alone times the run; the others take its
        // schedule from its plans.
        if self.schedule.is_none()
            && self.rank0()
            && (update > TIMED_UPDATES || update >= self.cfg.total_steps)
        {
            self.set_estimated_schedule(update);
        }
        let multi = self.multi();
        if !multi
            && self
                .inflight
                .as_ref()
                .is_some_and(|f| f.handle.is_finished())
        {
            let answer = self.collect();
            self.apply(answer, update, items)?;
        }
        let scheduled = self.next_scheduled().is_some_and(|s| s <= update);
        let mut waited = 0.0;
        // Every rank checks at the same updates: with several, only those that
        // do not depend on what rank 0 alone knows.
        let due = update.is_multiple_of(self.cfg.every)
            || update == self.cfg.total_steps
            || (!multi && (scheduled || !self.reasons.is_empty()));
        if due {
            let t = Instant::now();
            self.check(update, items)?;
            waited = t.elapsed().as_secs_f64();
        }
        self.last_wait = waited;
        let report_due =
            update.is_multiple_of(self.analysis.every) || update == self.cfg.total_steps;
        let before: Option<Vec<Vec<f32>>> = (report_due && self.analysis.wants_update_ratio())
            .then(|| items.iter().map(|it| it.param.to_vec()).collect());
        self.run_program(items, update, Some(info.loss))?;
        if let Some(i) = &mut self.integrity
            && let Some(flag) = i.after_update(update, items)
        {
            self.run.event("integrity", flag);
        }
        if report_due {
            let losses = std::mem::take(&mut self.report_window);
            let (report, wake) = self.analysis.report(
                update,
                &losses,
                items,
                before.as_deref(),
                info.outputs,
                self.program_ms,
            );
            if self.reports.len() == MAX_REPORTS {
                self.reports.remove(0);
            }
            self.reports.push(report);
            self.reasons.extend(wake);
        }
        Ok(())
    }

    fn record_loss(&mut self, u: u64, loss: f32) {
        if !loss.is_finite() && self.nonfinite_at.is_none() {
            self.nonfinite_at = Some(u);
        }
        self.losses.push(loss);
        self.window.push(loss);
        self.report_window.push(loss);
        self.loss_log.push((u, loss));
        self.recent_losses.push(loss);
        if self.recent_losses.len() > 100 {
            self.recent_losses.remove(0);
        }
    }

    /// A held-out validation result, measured after `update` (default: the
    /// last update made). The agent sees every one; see
    /// [`AgentConfig::validation`].
    pub fn report_validation(
        &mut self,
        loss: Option<f64>,
        accuracy: Option<f64>,
        update: Option<u64>,
    ) {
        let u = update.unwrap_or(self.update);
        let entry = observe::validation_entry(u, loss, accuracy);
        self.run.append("validation.jsonl", &entry.to_string());
        self.validation.push(entry);
        if let Some(i) = &mut self.integrity {
            i.validated(u);
        }
    }

    /// Record values (an object of numbers or strings) at the last update; with
    /// `show_agent` the agent sees them in its next message, otherwise they are
    /// only recorded (`metrics.jsonl`).
    pub fn log(&mut self, values: Value, show_agent: bool) -> ApiResult<()> {
        if !values.is_object() {
            return Err(ApiError::type_(
                "agent log: values must be an object (name → value)",
            ));
        }
        let line = json!({"update": self.update, "values": values, "shown": show_agent});
        self.run.append("metrics.jsonl", &line.to_string());
        if show_agent {
            let mut v = json!({"update": self.update});
            if let (Value::Object(m), Value::Object(vals)) = (&mut v, values) {
                m.extend(vals);
            }
            self.shown.push(v);
            if self.shown.len() > MAX_SHOWN {
                self.shown.remove(0);
            }
        }
        Ok(())
    }

    /// The training script's learning rate (its schedule): programs scale
    /// their own rates by it, as the plain optimizer would follow it.
    pub fn set_lr(&mut self, lr: f32) {
        self.base_lr = Some(lr);
    }

    pub fn run_dir(&self) -> &Path {
        &self.run.dir
    }

    /// The version of the program making updates now.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The running program's files (the updater's and `analysis.json`).
    pub fn files(&self) -> &Files {
        &self.files
    }

    /// The scheduled decision updates, once known.
    pub fn schedule(&self) -> Option<&[u64]> {
        self.schedule.as_deref()
    }

    /// Finish: wait for a decision in flight (its edits are recorded, not
    /// applied), write `summary.json` and `results.json` → the results.
    pub fn close(mut self) -> ApiResult<Value> {
        if self.inflight.is_some() {
            let answer = self.collect();
            self.run
                .event("late", json!({"k": answer.k, "applied": false}));
        }
        self.flush_loss_log();
        let summary = json!({
            "run_dir": self.run.dir.display().to_string(),
            "backend": self.backend_name,
            "updates": self.update,
            "schedule": self.schedule,
            "decisions": self.asked,
            "applied": self.counts.applied,
            "kept": self.counts.kept,
            "rejected": self.counts.rejected,
            "failed": self.counts.failed,
            "reverted": self.counts.reverted,
            "skipped": self.counts.skipped,
            "final_version": self.version,
            "cost_usd": self.cost,
        });
        self.run.write(
            "summary.json",
            &(serde_json::to_string_pretty(&summary).unwrap() + "\n"),
        )?;
        let finite: Vec<f32> = self
            .recent_losses
            .iter()
            .rev()
            .take(self.cfg.every as usize)
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let best = self
            .validation
            .iter()
            .filter(|v| v["loss"].is_number())
            .min_by(|a, b| {
                a["loss"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["loss"].as_f64().unwrap())
            })
            .cloned();
        let mut results = summary.clone();
        results["goal"] = json!(self.cfg.goal);
        results["final_train_loss"] =
            json!((!finite.is_empty()).then(|| finite.iter().sum::<f32>() / finite.len() as f32));
        results["last_validation"] = json!(self.validation.last());
        results["best_validation"] = json!(best);
        results["integrity_flags"] = json!(self.integrity.as_ref().map(|i| &i.flags));
        self.run.write(
            "results.json",
            &(serde_json::to_string_pretty(&results).unwrap() + "\n"),
        )?;
        self.run.event("closed", summary);
        Ok(results)
    }

    fn flush_loss_log(&mut self) {
        if self.loss_log.is_empty() {
            return;
        }
        let path = self.run.dir.join("losses.csv");
        if !path.exists() {
            self.run.append("losses.csv", "update,loss,base_lr,version");
        }
        let lr = self.base_lr.map(|l| l.to_string()).unwrap_or_default();
        let lines: Vec<String> = self
            .loss_log
            .drain(..)
            .map(|(u, l)| format!("{u},{l},{lr},{}", self.version))
            .collect();
        self.run.append("losses.csv", &lines.join("\n"));
    }

    // ── checkpoints ─────────────────────────────────────────────

    /// Write the agent's side of the run into `dir`: `agent.json` (versions,
    /// files, schedule, cost, what the next decision will be told) and
    /// `agent_state.bin` (the program's optimizer state). Save the trainer's
    /// params alongside; resume both with [`Self::resume`]. A decision in
    /// flight is not saved: the resumed run tells the agent it was lost.
    pub fn save(&mut self, dir: &Path) -> ApiResult<()> {
        self.flush_loss_log();
        let io = |e: std::io::Error| {
            ApiError::runtime(format!("agent checkpoint {}: {e}", dir.display()))
        };
        fs::create_dir_all(dir).map_err(io)?;
        let state = self.program.as_ref().and_then(|p| p.state_dict());
        let mut blob: Vec<u8> = Vec::new();
        let state_json = state.as_ref().map(|s| {
            let buffers: Vec<Value> = s
                .buffers
                .iter()
                .map(|(k, v)| {
                    blob.extend(v.iter().flat_map(|x| x.to_le_bytes()));
                    json!({"key": k, "len": v.len()})
                })
                .collect();
            json!({"step": s.step, "buffers": buffers})
        });
        let c = &self.counts;
        let ck = json!({
            "format": 2,
            "run_dir": self.run.dir.display().to_string(),
            "update": self.update,
            "version": self.version,
            "files": self.files,
            "fallback": self.fallback,
            "schedule": self.schedule,
            "max_decisions": self.max_decisions,
            "sched_i": self.sched_i,
            "asked": self.asked,
            "cost": self.cost,
            "base_lr": self.base_lr,
            "nonfinite_episode": self.nonfinite_episode,
            "counts": [c.applied, c.kept, c.rejected, c.reverted, c.failed, c.skipped],
            "losses": self.losses,
            "losses_from": self.losses_from,
            "reports": self.reports,
            "errors": self.errors,
            "reasons": self.reasons,
            "previous": self.previous,
            "shown": self.shown,
            "skipped": self.skipped,
            "validation": self.validation,
            "integrity_flags": self.integrity.as_ref().map(|i| &i.flags),
            "decision_seconds": self.decision_seconds,
            "inflight": self.inflight.as_ref().map(|f| f.k),
            "trial_open": self.trial.is_some(),
            "start_ms": self.start_ms,
            "state": state_json,
        });
        fs::write(dir.join(CHECKPOINT_STATE), &blob).map_err(io)?;
        fs::write(
            dir.join(CHECKPOINT),
            serde_json::to_string_pretty(&ck).unwrap(),
        )
        .map_err(io)?;
        self.run.event(
            "saved",
            json!({"update": self.update, "version": self.version,
                   "dir": dir.display().to_string(), "state_saved": state.is_some()}),
        );
        Ok(())
    }

    /// Continue a run from [`Self::save`]'s `dir`, in its run folder when that
    /// still exists. `cfg`, `host` and `backend` are given again (as at the
    /// start); the program's state is restored at the first update.
    pub fn resume(
        cfg: AgentConfig,
        host: Box<dyn ProgramHost>,
        backend: Box<dyn Backend>,
        dir: &Path,
    ) -> ApiResult<Self> {
        cfg.check()?;
        let read = |name: &str| {
            fs::read(dir.join(name)).map_err(|e| {
                ApiError::runtime(format!(
                    "agent checkpoint {}: {e}",
                    dir.join(name).display()
                ))
            })
        };
        let ck: Value = serde_json::from_slice(&read(CHECKPOINT)?)
            .map_err(|e| ApiError::type_(format!("agent checkpoint {CHECKPOINT}: {e}")))?;
        if !matches!(ck["format"].as_u64(), Some(1 | 2)) {
            return Err(ApiError::type_(format!(
                "agent checkpoint: unknown format {}",
                ck["format"]
            )));
        }
        let blob = read(CHECKPOINT_STATE)?;
        let files_of = |v: &Value, what: &str| -> ApiResult<Files> {
            serde_json::from_value(v.clone())
                .map_err(|e| ApiError::type_(format!("agent checkpoint `{what}`: {e}")))
        };
        let run_dir = PathBuf::from(ck["run_dir"].as_str().unwrap_or_default());
        let run = if run_dir.is_dir() {
            Run::open(run_dir)?
        } else {
            Run::create(&cfg.run_root)?
        };
        let mut a = Self::with_run(cfg, host, backend, run)?;
        a.files = files_of(&ck["files"], "files")?;
        a.analysis = Analysis::parse(a.files.get(ANALYSIS).map(String::as_str).unwrap_or("{}"))?;
        a.fallback = if ck["fallback"].is_null() {
            None
        } else {
            Some(files_of(&ck["fallback"], "fallback")?)
        };
        // A program that crashed after this checkpoint (on any rank) is not
        // resumed: the run continues from the one before it.
        let version = ck["version"].as_u64().unwrap_or(0);
        let crashed = fs::read_dir(&a.run.dir)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&format!("crashed-v{version:03}-"))
            });
        let mut skipped_crashed = false;
        if crashed && let Some(prev) = a.fallback.take() {
            let analysis = a.files.get(ANALYSIS).cloned();
            a.files = prev;
            if let Some(t) = analysis {
                a.files.insert(ANALYSIS.into(), t);
            }
            skipped_crashed = true;
        }
        let u64s = |v: &Value| {
            v.as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
        };
        let values = |v: &Value| v.as_array().cloned().unwrap_or_default();
        a.schedule = u64s(&ck["schedule"]);
        if let Some(n) = ck["max_decisions"].as_u64() {
            a.max_decisions = n as usize;
        }
        a.update = ck["update"].as_u64().unwrap_or(0);
        a.version = ck["version"].as_u64().unwrap_or(0) as u32 + u32::from(skipped_crashed);
        a.sched_i = ck["sched_i"].as_u64().unwrap_or(0) as usize;
        a.asked = ck["asked"].as_u64().unwrap_or(0) as usize;
        a.cost = ck["cost"].as_f64().unwrap_or(0.0);
        a.base_lr = ck["base_lr"].as_f64().map(|v| v as f32);
        a.nonfinite_episode = ck["nonfinite_episode"].as_bool().unwrap_or(false);
        if let Some(c) = u64s(&ck["counts"]).filter(|c| c.len() >= 5) {
            a.counts = Counts {
                applied: c[0] as usize,
                kept: c[1] as usize,
                rejected: c[2] as usize,
                reverted: c[3] as usize,
                failed: c[4] as usize,
                skipped: c.get(5).copied().unwrap_or(0) as usize,
            };
        }
        a.losses = ck["losses"]
            .as_array()
            .map(|v| {
                v.iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                    .collect()
            })
            .unwrap_or_default();
        a.losses_from = ck["losses_from"].as_u64().unwrap_or(a.update + 1);
        a.reports = values(&ck["reports"]);
        a.errors = values(&ck["errors"]);
        a.shown = values(&ck["shown"]);
        a.validation = values(&ck["validation"]);
        a.skipped = u64s(&ck["skipped"]).unwrap_or_default();
        a.reasons = ck["reasons"]
            .as_array()
            .map(|v| {
                v.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        a.previous = ck["previous"].as_str().map(str::to_string);
        a.start_ms = ck["start_ms"].as_f64();
        a.decision_seconds = ck["decision_seconds"].as_f64();
        if let (Some(i), Some(flags)) = (&mut a.integrity, ck["integrity_flags"].as_array()) {
            i.flags = flags.clone();
        }
        if let Some(k) = ck["inflight"].as_u64() {
            a.previous = Some(format!(
                "Decision {k} was in flight when the run was checkpointed; its answer was lost and \
                 the program was not changed."
            ));
        }
        let state = match &ck["state"] {
            Value::Null => None,
            s => {
                let mut out = OptimizerState {
                    step: s["step"].as_u64().unwrap_or(0),
                    buffers: Vec::new(),
                };
                let mut at = 0usize;
                for b in s["buffers"].as_array().into_iter().flatten() {
                    let len = b["len"].as_u64().unwrap_or(0) as usize;
                    let bytes = blob.get(at..at + 4 * len).ok_or_else(|| {
                        ApiError::type_(format!(
                            "agent checkpoint: {CHECKPOINT_STATE} is shorter than {CHECKPOINT} says"
                        ))
                    })?;
                    at += 4 * len;
                    let values = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    out.buffers
                        .push((b["key"].as_str().unwrap_or_default().to_string(), values));
                }
                Some(out)
            }
        };
        if skipped_crashed {
            a.errors.push(json!({"update": a.update, "where": format!("resuming: your program v{version:03}"),
                "error": "it crashed after the checkpoint this run resumed from; the program before it runs instead, its optimizer state from zero"}));
            a.run
                .event("resumed_without_crashed", json!({"version": version}));
        }
        a.restore = Some(Restore {
            state: if skipped_crashed { None } else { state },
            trial_open: ck["trial_open"].as_bool().unwrap_or(false),
        });
        a.run.event(
            "resumed",
            json!({"update": a.update, "version": a.version, "dir": dir.display().to_string()}),
        );
        Ok(a)
    }

    // ── setup ───────────────────────────────────────────────────

    fn start(&mut self, items: &[OptItem<'_>]) -> ApiResult<()> {
        self.params = items
            .iter()
            .map(|it| ParamInfo {
                name: it.name.to_string(),
                shape: it.shape.to_vec(),
            })
            .collect();
        let mut program = self.host.build(&self.files, &self.params, None)?;
        if let Some(r) = self.restore.take() {
            let restored = r.state.as_ref().is_some_and(|s| program.load_state_dict(s));
            if !restored {
                self.errors.push(json!({"update": self.update, "where": "resuming from a checkpoint",
                    "error": "your program's optimizer state could not be restored; it restarted from zero"}));
            }
            if r.trial_open {
                self.run
                    .event("trial_closed_by_resume", json!({"version": self.version}));
            }
            self.run.event(
                "restored",
                json!({"update": self.update, "state_restored": restored}),
            );
        } else {
            self.run
                .write_files(format!("programs/v{:03}", self.version), &self.files)?;
            self.run
                .write("FINAL", &format!("v{:03}\n", self.version))?;
        }
        self.program = Some(program);
        let table: String = self
            .params
            .iter()
            .map(|p| format!("{} {:?}\n", p.name, p.shape))
            .collect();
        self.run
            .write("workspace/reference/parameters.txt", &table)?;
        self.run
            .write_files("workspace/reference/starting", &self.host.starting_files())?;
        self.write_system_prompt()
    }

    fn write_system_prompt(&mut self) -> ApiResult<()> {
        let validation = match self.cfg.validation {
            None => String::new(),
            Some(n) => format!(
                " The training script evaluates the model on a validation set{} held out from \
                 training about every {n} updates, at times that are not announced, and shows you \
                 its loss, and its accuracy when it reports one. The result in the goal is measured \
                 on a separate set you never see.",
                self.cfg
                    .validation_size
                    .map(|s| format!(" of {s}"))
                    .unwrap_or_default()
            ),
        };
        let code = if self.run.workspace.join("reference/code").is_dir() {
            ", and read-only copies of the training run's code in reference/code/: read them for \
             the architecture, the loss, the data and what the loop does around the optimizer"
                .to_string()
        } else {
            String::new()
        };
        self.system = message::system(
            &self.template,
            &message::RunFacts {
                goal: &self.cfg.goal,
                total: self.cfg.total_steps,
                schedule: self.schedule.as_deref(),
                max_decisions: self.max_decisions,
                every: self.cfg.every,
                trial: self.cfg.trial_steps,
                waiting: self.cfg.waiting,
                max_lag: self.cfg.max_lag_updates,
                pause_on_nonfinite: self.cfg.pause_on_nonfinite,
                facts: self.cfg.facts.sentence(),
                validation,
                code,
                interface: format!("{}\n\n{}", self.host.interface(), Analysis::interface()),
                params: &self.params,
            },
        );
        self.run.write("system_prompt.txt", &self.system)
    }

    /// Set the schedule from the time of the updates so far.
    fn set_estimated_schedule(&mut self, u: u64) {
        let sec_per_step = self.sec_per_step();
        let n = estimate_decisions(
            self.cfg.total_steps,
            sec_per_step,
            self.cfg.seconds_per_decision,
            self.cfg.max_wait_ratio,
        );
        // Points before now move just after it, in order.
        let mut schedule: Vec<u64> = Vec::with_capacity(n);
        for p in decision_updates(n, self.cfg.total_steps, self.cfg.spread) {
            let p = p.max(u).max(schedule.last().map_or(0, |l| l + 1));
            if p < self.cfg.total_steps {
                schedule.push(p);
            }
        }
        self.max_decisions = n;
        self.run.event(
            "estimated",
            json!({"update": u, "sec_per_step": sec_per_step, "decisions": n, "schedule": schedule,
                   "seconds_per_decision": self.cfg.seconds_per_decision,
                   "max_wait_ratio": self.cfg.max_wait_ratio}),
        );
        self.schedule = Some(schedule);
        let _ = self.write_system_prompt();
    }

    fn sec_per_step(&self) -> f64 {
        let timed: Vec<f64> = self
            .step_seconds
            .iter()
            .copied()
            .filter(|s| *s > 0.0)
            .collect();
        if timed.is_empty() {
            0.0
        } else {
            timed.iter().sum::<f64>() / timed.len() as f64
        }
    }

    fn next_scheduled(&self) -> Option<u64> {
        self.schedule
            .as_ref()
            .and_then(|s| s.get(self.sched_i).copied())
    }

    // ── checks and decisions ────────────────────────────────────

    /// Whether the decision about to be asked waits for its answer.
    fn waits(&self, new_nonfinite: bool) -> bool {
        if new_nonfinite && self.cfg.pause_on_nonfinite {
            return true;
        }
        match self.cfg.waiting {
            Waiting::Always => true,
            Waiting::Never => false,
            Waiting::Auto => {
                if self.asked == 0 {
                    return true;
                }
                let answer = self
                    .decision_seconds
                    .unwrap_or(self.cfg.seconds_per_decision);
                let per_step = self.sec_per_step();
                per_step <= 0.0 || answer / per_step > self.cfg.max_lag_updates as f64
            }
        }
    }

    fn check(&mut self, u: u64, items: &[OptItem<'_>]) -> ApiResult<()> {
        self.flush_loss_log();
        let window = std::mem::take(&mut self.window);
        let nonfinite = window.iter().any(|x| !x.is_finite());
        let new_episode = nonfinite && !self.nonfinite_episode;
        if !window.is_empty() {
            self.nonfinite_episode = nonfinite;
        }
        let first = self.nonfinite_at.take();
        if new_episode {
            self.reasons.push(format!(
                "non-finite training loss at update {} (found at the check of update {u})",
                first.unwrap_or(u)
            ));
        }
        while let Some(s) = self.next_scheduled().filter(|&s| s <= u) {
            self.reasons
                .push(format!("scheduled decision (update {s})"));
            self.sched_i += 1;
        }
        let multi = self.multi();
        let mut answer = None;
        if self.rank0() {
            let pause = self.waits(new_episode);
            if !self.reasons.is_empty() && self.inflight.is_none() {
                if self.budget_left() {
                    self.submit(u, pause);
                } else {
                    self.reasons.clear();
                }
            }
            // With several ranks a non-waiting answer lands at a check too.
            if self
                .inflight
                .as_ref()
                .is_some_and(|f| f.paused || (multi && f.handle.is_finished()))
            {
                answer = Some(self.collect());
            }
        } else {
            self.reasons.clear();
        }
        if multi {
            let schedule = (!self.schedule_shared)
                .then(|| self.schedule.clone())
                .flatten();
            let plan = self.share(json!({
                "schedule": schedule,
                "max_decisions": self.max_decisions,
                "answer": answer.as_ref().map(Answer::to_json),
            }))?;
            if let Some(s) = plan["schedule"].as_array() {
                self.schedule_shared = true;
                if !self.rank0() {
                    self.schedule = Some(s.iter().filter_map(Value::as_u64).collect());
                    self.max_decisions = plan["max_decisions"].as_u64().unwrap_or(0) as usize;
                    let _ = self.write_system_prompt();
                }
            }
            answer = match &plan["answer"] {
                Value::Null => None,
                a => Some(Answer::from_json(a)?),
            };
        }
        if let Some(a) = answer {
            self.apply(a, u, items)?;
        }
        Ok(())
    }

    fn budget_left(&mut self) -> bool {
        let capped = self.cfg.max_cost_usd.is_some_and(|c| self.cost >= c);
        let ok = self.asked < self.max_decisions && !capped;
        if !ok && !self.budget_spent_noted {
            self.budget_spent_noted = true;
            self.run.event(
                "no_more_decisions",
                json!({"update": self.update, "decisions": self.asked, "cost_usd": self.cost}),
            );
        }
        ok
    }

    fn submit(&mut self, u: u64, paused: bool) {
        self.asked += 1;
        let k = self.asked;
        let record = self.run.dir.join(format!("decisions/{k:03}"));
        fs::create_dir_all(&record).ok();
        let notes = self.run.take_notes();
        // The workspace starts from the running program at every request.
        let _ = self.run.write_files("workspace", &self.files);
        let inbox = json!({
            "update": u,
            "reasons": self.reasons,
            "losses": self.losses,
            "losses_from": self.losses_from,
            "validation": self.cfg.validation.map(|_| &self.validation),
            "shown_values": self.shown,
            "skipped_updates": self.skipped,
            "reports": self.reports,
            "errors": self.errors,
            "notes": notes,
            "program_ms_per_update": self.program_ms,
            "starting_ms_per_update": self.start_ms,
            "cost_so_far": self.cost,
        });
        let _ = self.run.write(
            format!("workspace/inbox/decision-{k:03}.json"),
            &serde_json::to_string_pretty(&inbox).unwrap(),
        );
        let message = message::decision(&message::DecisionFacts {
            k,
            update: u,
            total: self.cfg.total_steps,
            paused,
            reasons: &self.reasons,
            previous: self.previous.as_deref(),
            losses: &self.losses,
            losses_from: self.losses_from,
            validation: self.cfg.validation.map(|_| self.validation.as_slice()),
            shown: &self.shown,
            skipped: &self.skipped,
            reports: &self.reports,
            errors: &self.errors,
            notes: &notes,
            program_ms: self.program_ms,
            start_ms: self.start_ms,
            decisions_left: self.max_decisions.saturating_sub(k),
            cost_so_far: self.cost,
            files: self.files.keys().map(String::as_str).collect(),
            aggressive: self.cfg.prompt.is_aggressive(),
        });
        let _ = fs::write(record.join("message.txt"), &message);
        let mut writable: Vec<String> = self.files.keys().cloned().collect();
        writable.push(NOTES.into());
        let req = DecisionRequest {
            k,
            workspace: self.run.workspace.clone(),
            writable,
            record,
            message,
            system: self.system.clone(),
            budget_left: self.cfg.max_cost_usd.map(|c| (c - self.cost).max(0.0)),
        };
        let backend = Arc::clone(&self.backend);
        let handle = std::thread::spawn(move || {
            let mut b = backend.lock().unwrap_or_else(|p| p.into_inner());
            let t0 = Instant::now();
            let mut out = b.decide(&req);
            if out.seconds == 0.0 {
                out.seconds = t0.elapsed().as_secs_f64();
            }
            out
        });
        self.run.event(
            "requested",
            json!({"k": k, "update": u, "reasons": self.reasons, "paused": paused,
                   "notes": notes.len()}),
        );
        self.inflight = Some(InFlight {
            k,
            requested_at: u,
            paused,
            base: self.files.clone(),
            handle,
        });
        self.reports.clear();
        self.errors.clear();
        self.reasons.clear();
        self.losses.clear();
        self.shown.clear();
        self.skipped.clear();
        self.losses_from = u + 1;
        self.previous = None;
    }

    /// Wait for the decision in flight and read what the agent left.
    fn collect(&mut self) -> Answer {
        let f = self
            .inflight
            .take()
            .expect("collect without a decision in flight");
        let outcome = f.handle.join().unwrap_or_else(|_| DecisionOutcome {
            error: Some("the backend panicked".into()),
            ..Default::default()
        });
        if let Some(c) = outcome.cost_usd {
            self.cost += c;
        }
        if outcome.error.is_none() && outcome.seconds > 0.0 {
            // Recent answer times weigh most, for `Waiting::Auto`.
            self.decision_seconds = Some(match self.decision_seconds {
                Some(s) => 0.5 * s + 0.5 * outcome.seconds,
                None => outcome.seconds,
            });
        }
        let mut edited = Files::new();
        for name in f.base.keys() {
            if let Ok(text) = fs::read_to_string(self.run.workspace.join(name)) {
                edited.insert(name.clone(), text);
            }
        }
        let changed = edited != f.base;
        let failed = outcome.error.as_deref().map(redact);
        let files = (changed && failed.is_none()).then_some(edited.clone());
        let record = format!("decisions/{:03}", f.k);
        let response = json!({
            "k": f.k,
            "requested_at": f.requested_at,
            "paused": f.paused,
            "changed": changed,
            "text": redact(&outcome.text),
            "cost_usd": outcome.cost_usd,
            "seconds": outcome.seconds,
            "error": failed,
        });
        let _ = self.run.write(
            format!("{record}/response.json"),
            &serde_json::to_string_pretty(&response).unwrap(),
        );
        if changed && failed.is_some() {
            let _ = self.run.write_files(format!("{record}/discarded"), &edited);
        }
        Answer {
            k: f.k,
            requested_at: f.requested_at,
            base: f.base,
            files,
            failed,
        }
    }

    /// Install an answer at the start of update `u`.
    fn apply(&mut self, a: Answer, u: u64, items: &[OptItem<'_>]) -> ApiResult<()> {
        let late = u.saturating_sub(a.requested_at);
        if let Some(e) = &a.failed {
            self.counts.failed += 1;
            self.errors.push(json!({"update": u, "where": format!("the agent call for decision {}", a.k), "error": e}));
            self.previous = Some(format!(
                "Decision {} was not answered: the agent call failed; the program was not changed.",
                a.k
            ));
            self.run
                .event("failed", json!({"k": a.k, "update": u, "error": e}));
            return Ok(());
        }
        // Only what the agent changed relative to what it was shown, so an
        // answer computed before a revert does not bring the reverted program back.
        let mut files = self.files.clone();
        if let Some(edited) = &a.files {
            for (name, text) in edited {
                if a.base.get(name) != Some(text) {
                    files.insert(name.clone(), text.clone());
                }
            }
        }
        if a.files.is_none() || files == self.files {
            self.counts.kept += 1;
            self.previous = Some(format!(
                "Decision {} (asked at update {}): you kept the program.",
                a.k, a.requested_at
            ));
            self.run.event("kept", json!({"k": a.k, "update": u}));
            return Ok(());
        }
        let version = self.version + 1;
        let analysis_changed = files.get(ANALYSIS) != self.files.get(ANALYSIS);
        let updater_changed = files
            .iter()
            .any(|(n, t)| n != ANALYSIS && self.files.get(n) != Some(t));
        let analysis = if analysis_changed {
            match Analysis::parse(files.get(ANALYSIS).map(String::as_str).unwrap_or_default()) {
                Ok(an) => Some(an),
                Err(e) => {
                    self.reject(a.k, u, version, &files, e);
                    return Ok(());
                }
            }
        } else {
            None
        };
        let program = if updater_changed {
            let current = self.program.as_deref();
            let built = catch("building the new program", || {
                self.host.build(&files, &self.params, current)
            })
            .and_then(|r| r);
            // Every rank installs it, or none does.
            if !self.agree(built.is_ok())? {
                let e = built.err().unwrap_or_else(|| {
                    ApiError::runtime("the new program failed to build on another rank")
                });
                self.reject(a.k, u, version, &files, e);
                return Ok(());
            }
            built.ok()
        } else {
            None
        };
        if let Some(an) = analysis {
            self.analysis = an;
        }
        if let Some(program) = program {
            let old = self.program.replace(program).expect("a running program");
            let limit = self.cfg.snapshot_limit_bytes;
            let snapshot = || -> Result<Vec<Vec<f32>>, String> {
                let bytes: usize = items.iter().map(|it| it.param.len() * 4).sum();
                if bytes > limit {
                    return Err(format!("{bytes} bytes, above the snapshot limit"));
                }
                Ok(items.iter().map(|it| it.param.to_vec()).collect())
            };
            self.trial = Some(match self.trial.take() {
                // Replaced while still on trial: it never passed, so the
                // fallback stays the program before it.
                Some(t) => Trial {
                    until: u + self.cfg.trial_steps,
                    first_pending: true,
                    snapshot: if t.first_pending {
                        t.snapshot
                    } else {
                        snapshot()
                    },
                    previous: t.previous,
                },
                None => {
                    self.fallback = Some(self.files.clone());
                    Trial {
                        until: u + self.cfg.trial_steps,
                        first_pending: true,
                        snapshot: snapshot(),
                        previous: old,
                    }
                }
            });
            self.program_ms = None;
            self.skips_in_a_row = 0;
        }
        self.files = files;
        self.version = version;
        let _ = self
            .run
            .write_files(format!("programs/v{version:03}"), &self.files);
        let _ = self.run.write("FINAL", &format!("v{version:03}\n"));
        self.counts.applied += 1;
        let what = match (updater_changed, analysis_changed) {
            (true, true) => "updater and analysis",
            (true, false) => "updater",
            _ => "analysis (the updater kept running, no trial)",
        };
        self.previous = Some(format!(
            "Your edit from decision {} (asked at update {}) took effect at update {u}, {late} \
             updates after the state you saw, as v{version:03}: {what}.",
            a.k, a.requested_at
        ));
        self.run.event(
            "applied",
            json!({"k": a.k, "update": u, "version": version, "late_by": late,
                   "updater": updater_changed, "analysis": analysis_changed}),
        );
        Ok(())
    }

    fn reject(&mut self, k: usize, u: u64, version: u32, files: &Files, e: ApiError) {
        self.counts.rejected += 1;
        let _ = self
            .run
            .write_files(format!("programs/rejected-v{version:03}"), files);
        let message = redact(&e.message);
        self.errors.push(json!({"update": u, "where": "building your new program (the running one stays)", "error": message}));
        self.previous = Some(format!(
            "Decision {k}: your new program failed to build at update {u} and was discarded; the \
             folder holds the running program (the error is below)."
        ));
        self.run
            .event("rejected", json!({"k": k, "update": u, "error": message}));
    }

    // ── updates and recovery ────────────────────────────────────

    fn run_program(
        &mut self,
        items: &mut [OptItem<'_>],
        u: u64,
        loss: Option<f32>,
    ) -> ApiResult<()> {
        let ctx = StepCtx {
            update: u,
            total: self.cfg.total_steps,
            base_lr: self.base_lr,
            loss,
        };
        let program = self.program.as_mut().expect("a running program");
        let can_skip = program.can_skip();
        // What a skip puts back: the params and the program's state.
        let saved = can_skip.then(|| {
            (
                items.iter().map(|it| it.param.to_vec()).collect::<Vec<_>>(),
                program.state_dict(),
            )
        });
        let first =
            self.trial.as_ref().is_some_and(|t| t.first_pending) || self.program_ms.is_none();
        let t0 = Instant::now();
        let result = catch("your program's update", || program.step(items, &ctx)).and_then(|r| r);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let multi = self.multi();
        let in_trial = self.trial.is_some();
        // During a trial every rank learns whether any failed and all go back
        // together. After it, agreeing would take a collective at every
        // update: a failure stops the run instead, and a resume starts from
        // the program before this one.
        let result = if multi && in_trial {
            let all_ok = self.agree(result.is_ok())?;
            match result {
                Ok(_) if !all_ok => Err(ApiError::runtime("the program failed on another rank")),
                r => r,
            }
        } else {
            result
        };
        if multi
            && !in_trial
            && let Err(e) = &result
        {
            let marker = format!("crashed-v{:03}-r{}.json", self.version, self.rank());
            let _ = self.run.write(
                &marker,
                &json!({"version": self.version, "update": u, "rank": self.rank(),
                        "error": redact(&e.message)})
                .to_string(),
            );
            self.run.event(
                "crashed",
                json!({"update": u, "version": self.version, "error": redact(&e.message)}),
            );
            return Err(ApiError::runtime(format!(
                "agent: v{:03} failed at update {u} on rank {} after its trial; restart from your \
                 checkpoint: the resumed run uses the program before it.\n{}",
                self.version,
                self.rank(),
                e.message
            )));
        }
        match result {
            Ok(mut skip) => {
                if can_skip && multi {
                    // One rank's skip is every rank's.
                    skip = !self.agree(!skip)?;
                }
                if first {
                    self.run.event(
                        "timed",
                        json!({"version": self.version, "update": u, "ms": ms}),
                    );
                }
                self.program_ms = Some(ms);
                if self.version == 0 {
                    self.start_ms = Some(ms);
                }
                if skip {
                    self.skip(items, u, saved);
                } else {
                    self.skips_in_a_row = 0;
                }
                if let Some(t) = &mut self.trial {
                    if t.first_pending {
                        t.first_pending = false;
                        t.snapshot = Err("released after the first update".into());
                    }
                    if u >= t.until {
                        if multi && !self.params_agree(items)? {
                            let e = ApiError::runtime(
                                "the program made the ranks' params differ (check for state kept \
                                 per rank, or randomness that differs between ranks)",
                            );
                            self.recover(items, u, loss, e)?;
                            return self.resync(items);
                        }
                        self.trial = None;
                        self.run.event(
                            "trial_passed",
                            json!({"update": u, "version": self.version}),
                        );
                    }
                }
                Ok(())
            }
            Err(e) => {
                self.recover(items, u, loss, e)?;
                if multi {
                    // A failing update may be partly applied on some ranks.
                    self.resync(items)?;
                }
                Ok(())
            }
        }
    }

    /// The program asked to skip update `u`.
    #[allow(clippy::type_complexity)]
    fn skip(
        &mut self,
        items: &mut [OptItem<'_>],
        u: u64,
        saved: Option<(Vec<Vec<f32>>, Option<OptimizerState>)>,
    ) {
        let Some((params, state)) = saved else {
            self.errors.push(json!({"update": u, "where": "your program's update",
                "error": "step asked to skip, but the program does not declare that it can skip; the update was made"}));
            return;
        };
        if self.skips_in_a_row >= MAX_SKIPS_IN_A_ROW {
            self.skips_in_a_row = 0;
            self.reasons.push(format!(
                "your program skipped {MAX_SKIPS_IN_A_ROW} updates in a row; update {u} was made anyway"
            ));
            self.run.event("skip_overridden", json!({"update": u}));
            return;
        }
        for (it, p) in items.iter_mut().zip(params) {
            it.param.copy_from_slice(&p);
        }
        let state_back = match &state {
            Some(s) => self.program.as_mut().unwrap().load_state_dict(s),
            None => false,
        };
        if !state_back {
            self.errors.push(json!({"update": u, "where": "skipping an update",
                "error": "the params were put back, but your program's state could not be (no state_dict/load_state_dict)"}));
        }
        self.skips_in_a_row += 1;
        self.counts.skipped += 1;
        self.skipped.push(u);
        self.run.event("skipped", json!({"update": u}));
    }

    fn recover(
        &mut self,
        items: &mut [OptItem<'_>],
        u: u64,
        loss: Option<f32>,
        e: ApiError,
    ) -> ApiResult<()> {
        let bad = self.version;
        self.counts.reverted += 1;
        let error = redact(&e.message);
        // The analysis stays as it is; only the updater goes back.
        let analysis = self.files.get(ANALYSIS).cloned();
        let (what, redo) = if let Some(t) = self.trial.take() {
            let exact = t.first_pending && t.snapshot.is_ok();
            let what = if exact {
                for (it, saved) in items.iter_mut().zip(t.snapshot.as_ref().unwrap()) {
                    it.param.copy_from_slice(saved);
                }
                "the update was undone exactly and made by the previous program".to_string()
            } else if t.first_pending {
                format!(
                    "this update was not undone ({}); it may be partly applied",
                    t.snapshot.as_ref().err().unwrap()
                )
            } else {
                "this update was not undone and may be partly applied".to_string()
            };
            self.program = Some(t.previous);
            self.files = self.fallback.take().expect("a trial has a fallback");
            (what, exact)
        } else if let Some(files) = self.fallback.take() {
            let rebuilt = catch("rebuilding the previous program", || {
                self.host.build(&files, &self.params, None)
            })
            .and_then(|r| r)
            .map_err(|again| {
                ApiError::runtime(format!(
                    "agent: v{bad:03} failed at update {u} and the previous program could not be \
                     rebuilt: {}\n(the failure: {error})",
                    again.message
                ))
            })?;
            self.program = Some(rebuilt);
            self.files = files;
            (
                "this update was not undone; the previous program was rebuilt from its files, its \
                 optimizer state starting from zero"
                    .to_string(),
                false,
            )
        } else {
            return Err(ApiError::runtime(format!(
                "agent: the program failed at update {u} and there is no earlier program to go back to: {error}"
            )));
        };
        if let Some(text) = analysis {
            self.files.insert(ANALYSIS.into(), text);
        }
        self.version += 1;
        let _ = self
            .run
            .write_files(format!("programs/v{:03}", self.version), &self.files);
        let _ = self.run.write("FINAL", &format!("v{:03}\n", self.version));
        self.program_ms = None;
        self.errors.push(json!({"update": u, "where": format!("your program v{bad:03} (reverted: {what})"), "error": error}));
        self.run.event(
            "reverted",
            json!({"update": u, "version": bad, "exact": redo, "error": error}),
        );
        if redo {
            let ctx = StepCtx {
                update: u,
                total: self.cfg.total_steps,
                base_lr: self.base_lr,
                loss,
            };
            let program = self.program.as_mut().unwrap();
            catch("the previous program's update", || {
                program.step(items, &ctx)
            })
            .and_then(|r| r)?;
        }
        Ok(())
    }
}
