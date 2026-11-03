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
//! <dir>/report.json       held-out score + run summary
//! <dir>/workspace/        the CLI's working directory (empty; outside the repo)
//! ```
//!
//! The held-out score ([`WeightSpace::held_out`]) uses data no tool exposes, and the
//! state file lives outside the agent's workspace.

use crate::env::EnvKind;
use crate::space::{State, WeightSpace};
pub use rlx_agent::AgentCli;
use rlx_agent::{AgentRun, McpServer};
use serde_json::{Value, json};
use std::path::PathBuf;

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
    let text = serde_json::to_string(&ws.state).map_err(|e| e.to_string())?;
    std::fs::write(spec.state_path(), text).map_err(|e| e.to_string())?;
    let prompt = task_prompt(ws.state.env, ws.state.budget);
    std::fs::write(spec.out.join("prompt.txt"), &prompt).map_err(|e| e.to_string())?;
    Ok(prompt)
}

/// Launch the agent, then score the weights it left on held-out episodes.
pub fn run(spec: &RunSpec, prompt: &str, episodes: usize) -> Result<Value, String> {
    let outcome = spec.agent_run().run(prompt, |e| eprintln!("  {e}"))?;
    let text = std::fs::read_to_string(spec.state_path()).map_err(|e| e.to_string())?;
    let st: State = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut ws = WeightSpace::from_state(st);
    let held = if ws.state.graph.is_some() {
        Value::Null
    } else {
        ws.held_out(episodes)?
    };
    // Check the agent's account: what each used unit really does.
    let claims = if ws.state.graph.is_some() {
        Value::Null
    } else {
        crate::verify::check_run(
            &mut ws,
            outcome.summary.final_text.as_deref().unwrap_or(""),
            episodes.min(128),
        )?
    };
    let edits = ws
        .state
        .log
        .iter()
        .filter(|l| l.starts_with("edit "))
        .count();
    let s = &outcome.summary;
    let report = json!({
        "cli": spec.cli.program(),
        "env": ws.state.env,
        "exit_code": outcome.exit_code,
        "held_out": held,
        "claim_check": claims,
        "budget_audit": {
            "forward_passes_used": ws.state.used,
            "scripts_refused": ws.state.log.iter().filter(|l| l.starts_with("script refused")).count(),
            "note": "scripts cannot read weights and are refused when they carry copied weight \
                     values or large literal tables",
        },
        "forward_passes": { "used": ws.state.used, "budget": ws.state.budget },
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
    std::fs::write(spec.out.join("report.json"), format!("{report:#}\n"))
        .map_err(|e| e.to_string())?;
    Ok(report)
}
