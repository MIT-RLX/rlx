// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Point an agent CLI at the weight-space MCP server and score what it wrote.
//!
//! Launching the CLI, its MCP config, tool restriction and event parsing live
//! in [`rlx_agent`]; this module adds the weight-space parts: a fresh state
//! file, the task prompt, and the held-out score once the agent exits.
//!
//! Run layout (`--out <dir>`):
//!
//! ```text
//! <dir>/state.json        weights + checkpoints + budget (written by the MCP server)
//! <dir>/mcp.json          MCP config handed to the CLI
//! <dir>/prompt.txt        the task prompt
//! <dir>/transcript.jsonl  the CLI's stdout
//! <dir>/scoring.json      scoring stages done so far (a stopped run resumes them)
//! <dir>/report.json       held-out score + run summary
//! <dir>/exploration.svg   how the run explored (and exploration.json, its table)
//! <dir>/workspace/        the CLI's working directory (empty; outside the repo)
//! ```
//!
//! The held-out score ([`WeightSpace::held_out`]) uses data no tool exposes, and the
//! state file lives outside the agent's workspace.

use crate::env::EnvKind;
use crate::space::{State, WeightSpace};
pub use rlx_agent::AgentCli;
use rlx_agent::{AgentRun, McpServer, Summary};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// One weight-space agent run.
#[derive(Debug, Clone)]
pub struct RunSpec {
    pub cli: AgentCli,
    pub out: PathBuf,
    /// The `rlx-wsp` executable the CLI spawns as its MCP server.
    pub wsp_exe: PathBuf,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub max_turns: usize,
    pub max_budget_usd: Option<f64>,
    /// Appended verbatim to the CLI's arguments.
    pub extra_args: Vec<String>,
}

impl RunSpec {
    pub fn state_path(&self) -> PathBuf {
        self.out.join("state.json")
    }

    /// The `rlx-agent` invocation: one MCP server, `wsp`, over the state file.
    pub fn agent_run(&self) -> AgentRun {
        let server = McpServer::new(
            "wsp",
            &self.wsp_exe,
            [
                "mcp".to_string(),
                "--state".to_string(),
                self.state_path().display().to_string(),
            ],
        );
        let mut run = AgentRun::new(self.cli, &self.out, vec![server]);
        run.model = self.model.clone();
        run.effort = self.effort.clone();
        run.max_turns = Some(self.max_turns);
        run.max_budget_usd = self.max_budget_usd;
        run.extra_args = self.extra_args.clone();
        run
    }
}

/// The task as the agent sees it. Rules, layout and budget come from the
/// `describe` tool, so this stays environment-agnostic and gives no method.
pub fn task_prompt(env: EnvKind, budget: usize) -> String {
    if env == EnvKind::Graph {
        return format!(
            "You are working on a neural network by reading and editing its weights directly. \
             There is no training and no gradient descent.\n\n\
             Your only tools are those of the `wsp` MCP server. Start with `describe`, which \
             states the goal, the inputs, the parameters and the budget. Reading and editing \
             weights is free; `trace` runs the network and spends one forward pass from a budget \
             of {budget}. Do not compute the network's outputs yourself from its weights (in a script or by \
             hand): running it happens only through the budgeted tools.\n\n\
             When you are done, reply with a short description of what you found and what you \
             changed, and stop. In that reply, refer to every unit by its label exactly as the tools show it (for example \
             `l0.h3`; layers are counted from 0), not by names of your own."
        );
    }
    if env == EnvKind::Qa {
        return format!(
            "You are improving a language model's answers by editing its weights directly. There \
             is no training and no gradient descent: find what in the model causes wrong answers \
             and change the numbers (or add the graph edits the tools offer) that fix it.\n\n\
             Your only tools are those of the `wsp` MCP server. Start with `describe`, which states \
             the task, the prompt format, the model's tensors and the budget. Reading and editing \
             are free; every tool that runs the model spends forward passes from a budget of \
             {budget} (each tool says what it costs). Do not compute the model's outputs yourself \
             from its weights: running it happens only through the budgeted tools.\n\n\
             Accuracy on a few dozen questions moves by several points by chance. Decide with \
             paired measurements: `qa_loglik` and `compare` (metric loglik, same questions) give \
             steadier numbers, `compare` reports whether a difference is clear, and `dev_check` \
             (when the task has a dev split) is the check on questions no other tool shows. \
             Look at how answers fail (`qa_eval` outcomes) before choosing what to change.\n\n\
             You are scored after you finish on held-out questions you never see. When you are \
             done, reply with a short description of what you changed and the evidence that it \
             helps, and stop."
        );
    }
    if env.is_classifier() {
        return format!(
            "You are repairing a trained neural network by editing its weights directly. There is \
             no retraining and no gradient descent: work out what the network computes and change \
             the numbers that cause the problem.\n\n\
             Your only tools are those of the `wsp` MCP server. Start with `describe`, which states \
             the problem, the input and unit labels, the weight layout and the budget. Reading \
             weights, viewing sample images and editing are free. `probe` and `eval` each spend one \
             forward pass from a budget of {budget}; when it is gone you can still read and edit \
             but no longer run the network. Do not compute the network's outputs yourself from its weights (in a script or by \
             hand): running it happens only through the budgeted tools.\n\n\
             You are scored after you finish on held-out test data you never see. When you are \
             done, reply with a short description of what was wrong, how you found it, and what \
             you changed, and stop. In that reply, refer to every unit by its label exactly as the tools show it (for example \
             `l0.h3`; layers are counted from 0), not by names of your own."
        );
    }
    format!(
        "You are programming a neural network by writing its weights directly. There is no \
         training and no gradient descent: decide what each unit should compute and write the \
         numbers that make it compute that.\n\n\
         Your only tools are those of the `wsp` MCP server. Start with `describe`, which gives \
         the environment rules, the observation and unit labels, the weight layout and the \
         budget. Reading and editing weights is free. `probe` and `rollout` each spend one \
         forward pass from a budget of {budget}; when it is gone you can still read and edit \
         but no longer run the network. Do not compute the network's outputs yourself from its weights (in a script or by \
             hand): running it happens only through the budgeted tools.\n\n\
         The goal is the highest mean episode return of the greedy policy, scored after you \
         finish on held-out episodes you never see. When you are done, reply with a short \
         description of the program you wrote (what each unit you used computes) and stop. In \
         that reply, refer to every unit by its label exactly as the tools show it (for example \
         `l0.h3`; layers are counted from 0), not by names of your own."
    )
}

/// The full command line for `spec` (program, args).
pub fn command_line(spec: &RunSpec, prompt: &str) -> (String, Vec<String>) {
    spec.agent_run().command_line(prompt)
}

/// Create the run directory with `ws` as the starting state; returns the prompt.
pub fn setup(spec: &RunSpec, ws: &WeightSpace) -> Result<String, String> {
    if spec.state_path().exists() {
        return Err(format!(
            "{} already holds a run; pick a new --out",
            spec.out.display()
        ));
    }
    spec.agent_run().prepare()?;
    let text = ws.to_json()?;
    std::fs::write(spec.state_path(), text).map_err(|e| e.to_string())?;
    let mut prompt = task_prompt(ws.state.env, ws.state.budget);
    if let Some(o) = &ws.state.opal {
        prompt.push_str(&format!(
            "\n\nThis run also steers a LoRA training run on the practice questions (an agent in \
             the training loop): `train` advances it ({} steps in all), `set_updater` installs your \
             update policy (rates per site, freezing, a loss-spike gate, the optimizer) that runs \
             before every step, `analyze` runs your own probes over its history, and \
             `lora_snapshot` applies the adapter so you can measure it like any version. Training \
             changes the weights by gradient here; the scored model is whatever version you leave \
             live.",
            o.max_steps
        ));
    }
    if !ws.state.lineage.is_empty() {
        prompt.push_str(
            "\n\nThis run continues an earlier one from a saved version of its weights: \
             `describe` shows what was changed and measured there (`branched_from`), and \
             checkpoint `branch_start` is where this run began.",
        );
    }
    std::fs::write(spec.out.join("prompt.txt"), &prompt).map_err(|e| e.to_string())?;
    Ok(prompt)
}

/// Launch the agent, then score the weights it left on held-out episodes.
pub fn run(spec: &RunSpec, prompt: &str, episodes: usize) -> Result<Value, String> {
    let outcome = spec.agent_run().run(prompt, |e| eprintln!("  {e}"))?;
    score_run(
        &spec.out,
        spec.cli,
        outcome.exit_code,
        &outcome.summary,
        episodes,
        Stages::open(&spec.out, episodes)?,
    )
}

/// Finish a run stopped after its agent ended (or part way through its
/// scoring): the agent's account from the transcript, stages already done
/// (`scoring.json`, or the scores a `report.json` holds) kept.
pub fn finish(out: &Path, cli: AgentCli, episodes: usize) -> Result<Value, String> {
    let text = std::fs::read_to_string(out.join("transcript.jsonl"))
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let mut summary = Summary::default();
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            for e in rlx_agent::parse_event(cli, &v, &["wsp"]) {
                summary.record(e);
            }
        }
    }
    let mut stages = Stages::open(out, episodes)?;
    stages.adopt_report(&out.join("report.json"));
    score_run(out, cli, None, &summary, episodes, stages)
}

/// Scoring stages already done for this `state.json`, kept on disk as they
/// finish: a stage is minutes of generation, and a stopped run should not
/// pay for it twice.
pub struct Stages {
    path: PathBuf,
    key: String,
    done: serde_json::Map<String, Value>,
}

impl Stages {
    pub fn open(out: &Path, episodes: usize) -> Result<Self, String> {
        use std::hash::{Hash, Hasher};
        let state = std::fs::read(out.join("state.json")).map_err(|e| e.to_string())?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        state.hash(&mut h);
        episodes.hash(&mut h);
        let key = format!("{:016x}", h.finish());
        let path = out.join("scoring.json");
        let saved: Value = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null);
        // Stages of other weights do not carry over.
        let done = match (saved["key"].as_str(), saved["stages"].as_object()) {
            (Some(k), Some(d)) if k == key => d.clone(),
            _ => Default::default(),
        };
        Ok(Self { path, key, done })
    }

    /// Take the scores a `report.json` of these weights holds.
    fn adopt_report(&mut self, report: &Path) {
        let Some(r) = std::fs::read_to_string(report)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            return;
        };
        for (stage, v) in [
            ("final", &r["held_out_final"]),
            ("unedited", &r["held_out_unedited"]),
            ("perturbed", &r["checks"]["perturbed"]),
            ("truncation", &r["checks"]["truncation"]),
        ] {
            if !v.is_null() {
                self.done.entry(stage).or_insert_with(|| v.clone());
            }
        }
    }

    /// The stage's result: kept, else computed and saved.
    pub fn stage(
        &mut self,
        name: &str,
        f: impl FnOnce() -> Result<Value, String>,
    ) -> Result<Value, String> {
        if let Some(v) = self.done.get(name) {
            eprintln!("  scoring: {name} already done");
            return Ok(v.clone());
        }
        let t0 = std::time::Instant::now();
        let v = f()?;
        eprintln!("  scoring: {name} ({:.0}s)", t0.elapsed().as_secs_f64());
        self.done.insert(name.to_string(), v.clone());
        let tmp = self.path.with_extension("tmp");
        let text = json!({ "key": self.key, "stages": self.done }).to_string();
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &self.path).map_err(|e| e.to_string())?;
        Ok(v)
    }
}

/// Score the weights a run left and write its report and chart.
fn score_run(
    out: &Path,
    cli: AgentCli,
    exit_code: Option<i32>,
    s: &Summary,
    episodes: usize,
    mut stages: Stages,
) -> Result<Value, String> {
    let text = std::fs::read_to_string(out.join("state.json")).map_err(|e| e.to_string())?;
    let st: State = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut ws = WeightSpace::from_state(st);
    // Jobs still running when the agent stopped are not waited for.
    crate::jobs::stop_all(&mut ws);
    let held_final = if ws.is_loaded() {
        Value::Null
    } else {
        stages.stage("final", || ws.held_out(episodes))?
    };
    // A dev-verified version, scored too; with `keep_verified` it is the
    // headline (the unedited model when nothing was verified).
    let verified = if ws.state.checkpoints.contains_key("verified") {
        let live = ws.current_variant();
        let v = ws.variant("verified")?;
        // The verified version left live has just been scored.
        let same = v.params == live.params
            && v.rewrites == live.rewrites
            && live.lowrank.is_empty()
            && !held_final.is_null();
        Some(if same {
            held_final.clone()
        } else {
            stages.stage("verified", || {
                ws.swap_variant(v);
                let h = ws.held_out(episodes);
                ws.swap_variant(live);
                h
            })?
        })
    } else {
        None
    };
    // QA on a package: the unedited model's score (cached) as the
    // reference every run and repeat is measured against, the checks the
    // task asks for, and how far the weights moved.
    let qa_pkg = ws.state.qa.is_some() && ws.store().is_some();
    let keep_unedited = verified.is_none() && ws.state.keep_verified && ws.state.package.is_some();
    let unedited = if qa_pkg || keep_unedited {
        stages.stage("unedited", || crate::checks::unedited_held_out(&mut ws))?
    } else {
        Value::Null
    };
    let (held, kept) = match (&verified, ws.state.keep_verified) {
        (Some(v), true) => (v.clone(), "verified"),
        (None, true) if ws.state.package.is_some() => {
            (unedited.clone(), "unedited (nothing verified)")
        }
        _ => (held_final.clone(), "final"),
    };
    let checks = if qa_pkg {
        let mut c = json!({});
        for name in crate::checks::CHECKS {
            let v = stages.stage(name, || crate::checks::one_check(&mut ws, name))?;
            if !v.is_null() {
                c[name] = v;
            }
        }
        c
    } else {
        Value::Null
    };
    let edit_size = crate::checks::edit_size(&ws);
    // Check the agent's account: what each used unit really does.
    let claims = if ws.is_loaded() {
        Value::Null
    } else {
        stages.stage("claims", || {
            crate::verify::check_run(
                &mut ws,
                s.final_text.as_deref().unwrap_or(""),
                episodes.min(128),
            )
        })?
    };
    let edits = ws
        .state
        .log
        .iter()
        .filter(|l| l.starts_with("edit "))
        .count();
    let report = json!({
        "cli": cli.program(),
        "env": ws.state.env,
        "exit_code": exit_code,
        "held_out": held,
        "held_out_kept": kept,
        "held_out_final": held_final,
        "held_out_verified": verified,
        "held_out_unedited": unedited,
        "checks": checks,
        "edit_size": edit_size,
        "claim_check": claims,
        "harness": {
            "rlx_wsp": env!("CARGO_PKG_VERSION"),
            "task_seed": ws.state.task_seed,
            "action_choice": ws.state.protocol.describe(),
            "rewrites": ws.state.rewrites,
            "package": ws.state.package,
        },
        "budget_audit": {
            "forward_passes_used": ws.state.used,
            "tokens_used": ws.state.tokens_used,
            "token_budget": ws.state.token_budget,
            "dev_checks_left": ws.state.dev_checks_left,
            "scripts_refused": ws.state.log.iter().filter(|l| l.starts_with("script refused")).count(),
            "note": "scripts cannot read weights and are refused when they carry copied weight \
                     values or large literal tables",
        },
        "forward_passes": { "used": ws.state.used, "budget": ws.state.budget },
        "time": {
            "total_seconds": s.duration_ms.map(|d| d as f64 / 1000.0),
            "in_tools_seconds": ws.state.tool_seconds.values().sum::<f64>(),
            "outside_tools_seconds": s.duration_ms.map(|d| d as f64 / 1000.0 - ws.state.tool_seconds.values().sum::<f64>()),
            "by_tool": ws.state.tool_seconds,
            "note": "outside tools = the agent reasoning (and its CLI); in tools = mostly forward passes",
        },
        "edits": edits,
        "weights_nonzero": ws.state.params.values()
            .map(|t| t.data.iter().filter(|v| **v != 0.0).count()).sum::<usize>(),
        "agent": {
            "models": s.models,
            "tool_calls": s.tool_calls,
            "foreign_tool_calls": s.foreign_tool_calls,
            "num_turns": s.num_turns,
            "cost_usd": s.cost_usd,
            "duration_ms": s.duration_ms,
            "final_text": s.final_text,
        },
    });
    std::fs::write(out.join("report.json"), format!("{report:#}\n")).map_err(|e| e.to_string())?;
    // How the run explored, as a chart beside the report.
    let title = format!(
        "How {} explored the weight space",
        out.file_name()
            .map_or_else(|| "the run".into(), |n| n.to_string_lossy().to_string())
    );
    match crate::explore::write_chart(out, &title) {
        Ok(p) => eprintln!("  chart: {}", p.display()),
        Err(e) => eprintln!("  no chart: {e}"),
    }
    Ok(report)
}
