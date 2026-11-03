// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Who answers a decision: [`Keep`] (no agent, no cost) or [`CliAgent`], an
//! installed agent CLI (Claude Code, Cursor Agent, Copilot) launched through
//! [`rlx_agent`].
//!
//! The CLI runs with no built-in tools. Its whole tool surface is one MCP
//! server, `workspace` ([`rlx_agent::workspace`]), over the decision's
//! workspace folder: it can read every file there and write only the
//! program's files and `NOTES.md`. It answers by editing them; what comes back
//! is its last message (the rationale), its cost and any error.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rlx_agent::workspace::Workspace;
use rlx_agent::{AgentCli, AgentRun};
use serde_json::Value;

use super::records::redact;
use crate::runtime::opts::Opts;
use crate::{ApiError, ApiResult};

/// One decision, as a backend gets it.
#[derive(Clone, Debug)]
pub struct DecisionRequest {
    /// 1-based.
    pub k: usize,
    pub workspace: PathBuf,
    /// The workspace files the agent may write (the program's and its notes).
    pub writable: Vec<String>,
    /// Where the backend keeps its own records of this decision.
    pub record: PathBuf,
    pub message: String,
    pub system: String,
    /// Dollars the run may still spend, when it has a cap.
    pub budget_left: Option<f64>,
}

/// What came of a decision.
#[derive(Clone, Debug, Default)]
pub struct DecisionOutcome {
    pub text: String,
    pub cost_usd: Option<f64>,
    pub seconds: f64,
    /// The call failed; edits it left are not applied.
    pub error: Option<String>,
}

pub trait Backend: Send {
    fn name(&self) -> &str;
    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome;
}

/// A backend by name, as the bindings choose one: `"keep"`, or an agent CLI
/// (`"claude"`, `"cursor"`, `"copilot"`) with [`CliAgent::from_json`]'s options.
pub fn backend_from_json(name: &str, options: &Value) -> ApiResult<Box<dyn Backend>> {
    if name == "keep" {
        Opts::new("keep backend", options)?.finish()?;
        return Ok(Box::new(Keep));
    }
    if name == "mailbox" {
        return Ok(Box::new(super::mailbox::Mailbox::from_json(options)?));
    }
    let cli = AgentCli::parse(name).ok_or_else(|| {
        ApiError::type_(format!(
            "agent backend is \"keep\", \"claude\", \"cursor\", \"copilot\", \"codex\" or \
             \"mailbox\", got {name:?}"
        ))
    })?;
    Ok(Box::new(CliAgent::from_json(cli, options)?))
}

/// No agent: every decision keeps the programs. The run trains exactly as it
/// would unwrapped, which is how a wiring is checked before it costs money.
pub struct Keep;

impl Backend for Keep {
    fn name(&self) -> &str {
        "keep"
    }

    fn decide(&mut self, _req: &DecisionRequest) -> DecisionOutcome {
        DecisionOutcome {
            text: "kept (backend \"keep\": no agent)".into(),
            cost_usd: Some(0.0),
            ..Default::default()
        }
    }
}

/// The workspace MCP server's name. Not `workspace`: Claude Code reserves
/// that name and silently drops a server called so.
pub const SERVER: &str = "rlx_files";

/// The agent of the paper's runs, for Claude.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";

/// An agent CLI, launched once per decision through [`rlx_agent::AgentRun`].
///
/// Claude keeps one session for the run (`--session-id`, then `--resume`) and
/// gets the system prompt with every call (`--append-system-prompt`, which a
/// compacted session would otherwise lose). The other CLIs start fresh each
/// time, so their message carries the system prompt in front.
///
/// A failed call is made again after [`Self::retry_pause`], then twice as
/// long (at most [`Self::retry_pause_max`]), up to [`Self::retries`] times and
/// while the decision's [`Self::timeout`] allows. An agent that is
/// *unavailable* — not logged in, over a usage or spend limit, overloaded, no
/// network — is waited for instead, with the same pauses and no deadline
/// unless [`Self::unavailable_timeout`] is set.
pub struct CliAgent {
    pub cli: AgentCli,
    pub model: Option<String>,
    /// Claude only.
    pub effort: Option<String>,
    /// Claude only.
    pub max_turns: Option<usize>,
    /// The workspace MCP server's program and leading arguments; the folder
    /// and its writable names are appended. Default: the `rlx-agent` binary
    /// (next to this executable, else on `PATH`) with `workspace-mcp`.
    pub mcp_command: Vec<String>,
    /// The CLI executable, when not its name on `PATH`.
    pub program: Option<PathBuf>,
    /// One decision's time; a call still running then is stopped.
    pub timeout: Option<Duration>,
    pub retries: usize,
    pub retry_pause: Duration,
    pub retry_pause_max: Duration,
    pub unavailable_timeout: Option<Duration>,
    /// Codex only: its own `CODEX_HOME`, so the run's login and history stay
    /// apart from the default one.
    pub codex_home: Option<PathBuf>,
    session: String,
    started: bool,
}

/// Words with which agent CLIs say they cannot work now (rather than that
/// this call failed): a login, a usage or spend limit, an overload, the network.
const UNAVAILABLE: [&str; 12] = [
    "hit your limit",
    "usage limit reached",
    "rate limit exceeded",
    "spend limit",
    "credit balance is too low",
    "run /login",
    "not logged in",
    "invalid api key",
    "overloaded_error",
    "api is overloaded",
    "could not resolve host",
    "network is unreachable",
];

/// Whether a failed call's text says the agent is unavailable.
pub fn unavailable(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    UNAVAILABLE.iter().any(|w| t.contains(w))
}

/// One launch's outcome and what to do about a failure.
struct Launch {
    outcome: DecisionOutcome,
    timed_out: bool,
    unavailable: bool,
}

impl CliAgent {
    pub fn new(cli: AgentCli) -> Self {
        let claude = cli == AgentCli::Claude;
        Self {
            cli,
            model: claude.then(|| DEFAULT_MODEL.to_string()),
            effort: claude.then(|| "high".to_string()),
            max_turns: None,
            mcp_command: default_mcp_command(),
            program: None,
            timeout: None,
            retries: 5,
            retry_pause: Duration::from_secs(15),
            retry_pause_max: Duration::from_secs(600),
            unavailable_timeout: None,
            codex_home: None,
            session: new_session_id(),
            started: false,
        }
    }

    /// `{model, effort, max_turns, mcp_command, program, timeout_seconds,
    /// retries, retry_pause_seconds, retry_pause_max_seconds,
    /// unavailable_timeout_seconds, codex_home}`, each optional.
    pub fn from_json(cli: AgentCli, v: &Value) -> ApiResult<Self> {
        let o = Opts::new("agent CLI", v)?;
        let mut c = Self::new(cli);
        if let Some(m) = o.str("model")? {
            c.model = Some(m.to_string());
        }
        if let Some(e) = o.str("effort")? {
            c.effort = Some(e.to_string());
        }
        c.max_turns = o.u64("max_turns")?.map(|n| n as usize);
        c.program = o.str("program")?.map(PathBuf::from);
        let secs = |name: &str| -> ApiResult<Option<Duration>> {
            match o.f64(name)? {
                Some(s) if s.is_finite() && s >= 0.0 => Ok(Some(Duration::from_secs_f64(s))),
                Some(s) => Err(ApiError::range(format!(
                    "agent CLI option '{name}': {s} is not a number of seconds"
                ))),
                None => Ok(None),
            }
        };
        c.timeout = secs("timeout_seconds")?;
        c.unavailable_timeout = secs("unavailable_timeout_seconds")?;
        if let Some(p) = secs("retry_pause_seconds")? {
            c.retry_pause = p;
        }
        if let Some(p) = secs("retry_pause_max_seconds")? {
            c.retry_pause_max = p;
        }
        c.retries = o.usize("retries", c.retries)?;
        c.codex_home = o.str("codex_home")?.map(PathBuf::from);
        if let Some(cmd) = o.get("mcp_command") {
            let parts: Option<Vec<String>> = cmd
                .as_array()
                .map(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect())
                .unwrap_or(None);
            c.mcp_command = parts.filter(|p| !p.is_empty()).ok_or_else(|| {
                ApiError::type_("agent CLI option 'mcp_command': a non-empty list of strings (program, arguments…)")
            })?;
        }
        o.finish()?;
        Ok(c)
    }

    /// The invocation for `req`, and the prompt it is given.
    pub fn agent_run(&self, req: &DecisionRequest) -> (AgentRun, String) {
        let ws = Workspace::new(&req.workspace, req.writable.clone());
        let (program, prefix) = self
            .mcp_command
            .split_first()
            .expect("mcp_command is never empty");
        let mut run = AgentRun::new(
            self.cli,
            &req.record,
            vec![ws.server(SERVER, program, prefix)],
        );
        run.workspace = req.workspace.clone();
        run.model = self.model.clone();
        run.effort = self.effort.clone();
        run.max_turns = self.max_turns;
        run.max_budget_usd = req.budget_left;
        run.program = self.program.clone();
        if self.cli == AgentCli::Codex
            && let Some(home) = &self.codex_home
        {
            run.env
                .push(("CODEX_HOME".into(), home.display().to_string()));
        }
        if self.cli == AgentCli::Claude {
            let session = if self.started {
                "--resume"
            } else {
                "--session-id"
            };
            run.extra_args = vec![
                session.into(),
                self.session.clone(),
                "--append-system-prompt".into(),
                req.system.clone(),
            ];
            (run, req.message.clone())
        } else {
            (run, format!("{}\n\n---\n\n{}", req.system, req.message))
        }
    }

    fn launch(&mut self, req: &DecisionRequest, note: &str, time_left: Option<Duration>) -> Launch {
        let t0 = Instant::now();
        let (mut run, prompt) = self.agent_run(req);
        run.timeout = time_left;
        let fail = |e: String| Launch {
            unavailable: unavailable(&e),
            outcome: DecisionOutcome {
                error: Some(e),
                seconds: t0.elapsed().as_secs_f64(),
                ..Default::default()
            },
            timed_out: false,
        };
        if let Err(e) = std::fs::create_dir_all(&req.record)
            .map_err(|e| format!("{}: {e}", req.record.display()))
            .and_then(|()| run.prepare())
        {
            return fail(format!("preparing the agent run: {e}"));
        }
        let out = match run.run(&format!("{note}{prompt}"), |_| {}) {
            Ok(o) => o,
            Err(e) => return fail(e),
        };
        // Transcripts are records: no credentials in them.
        let transcript = std::fs::read_to_string(&out.transcript).unwrap_or_default();
        let clean = redact(&transcript);
        if clean != transcript {
            let _ = std::fs::write(&out.transcript, &clean);
        }
        let text = out.summary.final_text.clone().unwrap_or_default();
        let mut launch = Launch {
            outcome: DecisionOutcome {
                text: text.clone(),
                cost_usd: out.summary.cost_usd,
                seconds: t0.elapsed().as_secs_f64(),
                error: None,
            },
            timed_out: out.timed_out,
            unavailable: false,
        };
        if out.timed_out {
            launch.outcome.error = Some(format!(
                "no answer within {:.0} s; the call was stopped (its output is in {})",
                time_left.unwrap_or_default().as_secs_f64(),
                out.transcript.display()
            ));
        } else if let Some(tools) = out
            .summary
            .tools
            .as_ref()
            .filter(|t| !t.iter().any(|n| n.contains(SERVER)))
        {
            // The CLI ran without its workspace tools: whatever it answered, it
            // could not have read or edited anything. Not a decision.
            launch.outcome.error = Some(format!(
                "{} started without the {SERVER} workspace tools (it had: {}); the workspace \
                 server did not load — check `rlx-agent-opt doctor`. Its output is in {}",
                self.cli.program(),
                if tools.is_empty() {
                    "none".to_string()
                } else {
                    tools.join(", ")
                },
                out.transcript.display()
            ));
        } else if out.exit_code == Some(0) && !unavailable_result(&clean) {
            self.started = true;
        } else {
            launch.unavailable = unavailable(&text) || unavailable(&clean);
            launch.outcome.error = Some(format!(
                "{} exited with {:?}{}; its output is in {}",
                self.cli.program(),
                out.exit_code,
                if text.is_empty() {
                    String::new()
                } else {
                    format!(": {}", tail(&text))
                },
                out.transcript.display()
            ));
        }
        launch
    }
}

/// A Claude `result` line that reports an error (a limit is answered with exit
/// code 0 and the message as the result).
fn unavailable_result(transcript: &str) -> bool {
    transcript.lines().any(|l| {
        serde_json::from_str::<Value>(l).is_ok_and(|v| {
            v["type"] == "result"
                && v["is_error"] == true
                && v["result"].as_str().is_some_and(unavailable)
        })
    })
}

fn tail(s: &str) -> String {
    let s = s.trim();
    let start = s.char_indices().rev().nth(499).map(|(i, _)| i).unwrap_or(0);
    s[start..].to_string()
}

impl Backend for CliAgent {
    fn name(&self) -> &str {
        self.cli.program()
    }

    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        let t0 = Instant::now();
        let mut pause = self.retry_pause;
        let (mut retries, mut cost, mut waited) = (0usize, None::<f64>, Duration::ZERO);
        let mut note = String::new();
        let mut resumed_fresh = false;
        loop {
            // Waiting for an unavailable agent does not use up the decision's time.
            let time_left = self
                .timeout
                .map(|t| t.saturating_sub(t0.elapsed().saturating_sub(waited)));
            let l = self.launch(req, &note, time_left);
            if let Some(c) = l.outcome.cost_usd {
                cost = Some(cost.unwrap_or(0.0) + c);
            }
            let mut out = l.outcome;
            out.cost_usd = cost;
            out.seconds = t0.elapsed().as_secs_f64();
            let Some(err) = out.error.clone() else {
                return out;
            };
            if l.timed_out {
                return out;
            }
            if l.unavailable {
                let over = self.unavailable_timeout.is_some_and(|t| waited + pause > t);
                if over {
                    out.error = Some(format!("the agent stayed unavailable: {err}"));
                    return out;
                }
                std::thread::sleep(pause);
                waited += pause;
                pause = (pause * 2).min(self.retry_pause_max);
                continue;
            }
            // The session may be gone (another machine, a cleared cache):
            // start a new one once, and say so, so the agent does not assume
            // it remembers.
            if self.cli == AgentCli::Claude && self.started && !resumed_fresh {
                resumed_fresh = true;
                self.session = new_session_id();
                self.started = false;
                note = "(Your earlier conversation could not be resumed, so this is a new one. \
                        Your current programs, and any notes you kept, are in this folder; \
                        inbox/ holds the inputs of every decision so far.)\n\n"
                    .into();
                continue;
            }
            let out_of_time = self
                .timeout
                .is_some_and(|t| t0.elapsed().saturating_sub(waited) + pause >= t);
            if retries >= self.retries || out_of_time {
                return out;
            }
            retries += 1;
            std::thread::sleep(pause);
            pause = (pause * 2).min(self.retry_pause_max);
        }
    }
}

/// `rlx-agent workspace-mcp`: the binary next to this executable (or one
/// folder up, where cargo puts test executables), else on `PATH`.
fn default_mcp_command() -> Vec<String> {
    let name = format!("rlx-agent{}", std::env::consts::EXE_SUFFIX);
    let near = std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent()?.to_path_buf();
        [dir.join(&name), dir.parent()?.join(&name)]
            .into_iter()
            .find(|p| p.is_file())
    });
    let program = near.map(|p| p.display().to_string()).unwrap_or(name);
    vec![program, "workspace-mcp".into()]
}

/// A random version-4 UUID, without a dependency: the CLI requires the form.
fn new_session_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(8) {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(Instant::now().elapsed().as_nanos() ^ std::process::id() as u128);
        chunk.copy_from_slice(&h.finish().to_le_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_distinct_uuid_v4s() {
        let (a, b) = (new_session_id(), new_session_id());
        assert_ne!(a, b);
        assert_eq!(a.len(), 36);
        assert_eq!(&a[14..15], "4");
        assert!(matches!(&a[19..20], "8" | "9" | "a" | "b"), "{a}");
    }
}
