// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx-agent** — run an installed agent CLI against MCP servers.
//!
//! The LLM is whatever agent CLI is on the machine; this crate makes no API
//! requests. It writes the CLI's MCP config, builds a command line that
//! restricts the agent to the listed MCP servers, launches the CLI in a
//! working directory, saves its stdout as a transcript, and normalises each
//! CLI's event stream into [`Event`]s so progress and summaries look the same
//! for every agent.
//!
//! Swapping agents is one [`AgentCli`] variant: how to pass MCP config, how to
//! restrict tools, which output format to request, and how to parse it.
//!
//! ```no_run
//! use rlx_agent::{AgentCli, AgentRun, McpServer};
//! let run = AgentRun::new(AgentCli::Claude, "runs/1", vec![McpServer::new(
//!     "wsp", "rlx-wsp", ["mcp", "--state", "state.json"],
//! )]);
//! run.prepare().unwrap();
//! let outcome = run.run("Call `describe`.", |e| eprintln!("{e}")).unwrap();
//! println!("{:?}", outcome.summary.final_text);
//! ```

pub mod events;
pub mod workspace;

pub use events::{Event, Summary, parse_event};

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Agent CLIs this crate can launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentCli {
    /// Claude Code (`claude -p`, stream-json). Built-in tools disabled; only
    /// the listed MCP servers' tools are available.
    Claude,
    /// Cursor Agent (`agent -p`, stream-json). MCP via
    /// `<workspace>/.cursor/mcp.json`; its shell cannot be disabled from the
    /// command line, so it runs sandboxed in the workspace.
    Cursor,
    /// GitHub Copilot CLI (`copilot -p`, JSONL). Shell and write denied,
    /// built-in MCP servers and custom instructions off.
    Copilot,
    /// OpenAI Codex CLI (`codex exec --json`, JSONL). MCP servers via `-c
    /// mcp_servers.<name>.…` overrides; read-only sandbox, so its own shell
    /// and patches cannot change files and the MCP servers do the writing.
    Codex,
}

impl AgentCli {
    pub const ALL: [AgentCli; 4] = [
        AgentCli::Claude,
        AgentCli::Cursor,
        AgentCli::Copilot,
        AgentCli::Codex,
    ];

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "claude" => Some(Self::Claude),
            "cursor" | "agent" | "cursor-agent" => Some(Self::Cursor),
            "copilot" => Some(Self::Copilot),
            "codex" => Some(Self::Codex),
            _ => None,
        }
    }

    /// Executable name.
    pub fn program(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Cursor => "agent",
            Self::Copilot => "copilot",
            Self::Codex => "codex",
        }
    }
}

/// A TOML string literal, for Codex's `-c key=value` overrides.
fn toml_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_default()
}

/// A stdio MCP server the agent may use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServer {
    pub name: String,
    pub command: PathBuf,
    pub args: Vec<String>,
}

impl McpServer {
    pub fn new<I, S>(name: &str, command: impl Into<PathBuf>, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            name: name.to_string(),
            command: command.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    fn entry(&self) -> Value {
        json!({ "command": self.command, "args": self.args })
    }
}

/// One agent invocation.
#[derive(Debug, Clone)]
pub struct AgentRun {
    pub cli: AgentCli,
    /// Holds `mcp.json` and `transcript.jsonl`.
    pub run_dir: PathBuf,
    /// The CLI's working directory (created empty by [`AgentRun::prepare`]).
    pub workspace: PathBuf,
    pub servers: Vec<McpServer>,
    pub model: Option<String>,
    /// Claude only.
    pub effort: Option<String>,
    /// Claude only.
    pub max_turns: Option<usize>,
    /// Claude only.
    pub max_budget_usd: Option<f64>,
    /// Appended verbatim to the CLI's arguments.
    pub extra_args: Vec<String>,
    /// The CLI executable, when not [`AgentCli::program`] on `PATH`.
    pub program: Option<PathBuf>,
    /// Extra environment for the CLI (for example Codex's `CODEX_HOME`).
    pub env: Vec<(String, String)>,
    /// Stop the CLI after this long; the outcome says it was stopped.
    pub timeout: Option<Duration>,
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub exit_code: Option<i32>,
    pub transcript: PathBuf,
    pub summary: Summary,
    /// The CLI ran past [`AgentRun::timeout`] and was stopped.
    pub timed_out: bool,
}

impl AgentRun {
    /// A run in `run_dir` with its workspace at `run_dir/workspace`.
    pub fn new(cli: AgentCli, run_dir: impl Into<PathBuf>, servers: Vec<McpServer>) -> Self {
        let run_dir = run_dir.into();
        Self {
            cli,
            workspace: run_dir.join("workspace"),
            run_dir,
            servers,
            model: None,
            effort: None,
            max_turns: None,
            max_budget_usd: None,
            extra_args: Vec::new(),
            program: None,
            env: Vec::new(),
            timeout: None,
        }
    }

    pub fn mcp_config_path(&self) -> PathBuf {
        self.run_dir.join("mcp.json")
    }

    pub fn transcript_path(&self) -> PathBuf {
        self.run_dir.join("transcript.jsonl")
    }

    /// The MCP config document in this CLI's format.
    pub fn mcp_config(&self) -> Value {
        let mut servers = serde_json::Map::new();
        for s in &self.servers {
            let mut e = s.entry();
            if self.cli == AgentCli::Copilot {
                e["type"] = json!("local");
                e["tools"] = json!(["*"]);
            }
            servers.insert(s.name.clone(), e);
        }
        json!({ "mcpServers": servers })
    }

    /// Create the run directory and workspace and write the MCP config.
    pub fn prepare(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.workspace)
            .map_err(|e| format!("{}: {e}", self.workspace.display()))?;
        let cfg = format!("{:#}\n", self.mcp_config());
        let write = |p: &Path| std::fs::write(p, &cfg).map_err(|e| format!("{}: {e}", p.display()));
        write(&self.mcp_config_path())?;
        if self.cli == AgentCli::Cursor {
            let dir = self.workspace.join(".cursor");
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            write(&dir.join("mcp.json"))?;
        }
        Ok(())
    }

    /// `(program, args)` for `prompt`.
    pub fn command_line(&self, prompt: &str) -> (String, Vec<String>) {
        let s = |x: &str| x.to_string();
        let cfg = self.mcp_config_path().display().to_string();
        let mut a: Vec<String> = if self.cli == AgentCli::Codex {
            Vec::new()
        } else {
            vec![s("-p"), s(prompt)]
        };
        match self.cli {
            AgentCli::Claude => {
                a.extend([
                    s("--output-format"),
                    s("stream-json"),
                    s("--verbose"),
                    s("--mcp-config"),
                    cfg,
                    s("--strict-mcp-config"),
                    // No built-in tools (shell, files, web): the MCP servers
                    // are the whole tool surface.
                    s("--tools"),
                    s(""),
                    s("--permission-mode"),
                    s("dontAsk"),
                ]);
                for srv in &self.servers {
                    a.extend([s("--allowedTools"), format!("mcp__{}", srv.name)]);
                }
                if let Some(n) = self.max_turns {
                    a.extend([s("--max-turns"), n.to_string()]);
                }
                if let Some(e) = &self.effort {
                    a.extend([s("--effort"), e.clone()]);
                }
                if let Some(b) = self.max_budget_usd {
                    a.extend([s("--max-budget-usd"), b.to_string()]);
                }
            }
            AgentCli::Cursor => a.extend([
                s("--output-format"),
                s("stream-json"),
                s("--approve-mcps"),
                s("--trust"),
                s("--force"),
                s("--sandbox"),
                s("enabled"),
                s("--workspace"),
                self.workspace.display().to_string(),
            ]),
            AgentCli::Copilot => {
                a.extend([
                    s("--additional-mcp-config"),
                    format!("@{cfg}"),
                    s("--disable-builtin-mcps"),
                    s("--deny-tool"),
                    s("shell"),
                    s("--deny-tool"),
                    s("write"),
                    s("--no-custom-instructions"),
                    s("--no-ask-user"),
                    s("--output-format"),
                    s("json"),
                ]);
                for srv in &self.servers {
                    a.extend([s("--allow-tool"), srv.name.clone()]);
                }
            }
            AgentCli::Codex => {
                a.extend([
                    s("exec"),
                    s("--json"),
                    s("--skip-git-repo-check"),
                    // Its shell and patches cannot write: the MCP servers do.
                    s("--sandbox"),
                    s("read-only"),
                    s("--cd"),
                    self.workspace.display().to_string(),
                ]);
                for srv in &self.servers {
                    let args: Vec<String> = srv.args.iter().map(|x| toml_str(x)).collect();
                    a.extend([
                        s("-c"),
                        format!(
                            "mcp_servers.{}.command={}",
                            srv.name,
                            toml_str(&srv.command.display().to_string())
                        ),
                        s("-c"),
                        format!("mcp_servers.{}.args=[{}]", srv.name, args.join(",")),
                    ]);
                }
            }
        }
        if let Some(m) = &self.model {
            a.extend([s("--model"), m.clone()]);
        }
        a.extend(self.extra_args.iter().cloned());
        if self.cli == AgentCli::Codex {
            a.push(s(prompt));
        }
        let program = match &self.program {
            Some(p) => p.display().to_string(),
            None => self.cli.program().to_string(),
        };
        (program, a)
    }

    /// Launch the CLI in the workspace, save stdout to the transcript, call
    /// `on_event` for each normalised event, and summarise the run. With a
    /// [`AgentRun::timeout`], a CLI still running then is stopped.
    pub fn run(&self, prompt: &str, mut on_event: impl FnMut(&Event)) -> Result<Outcome, String> {
        let (prog, args) = self.command_line(prompt);
        let mut cmd = Command::new(&prog);
        cmd.args(&args)
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            .current_dir(&self.workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // Its own process group, so a timeout stops what it started too (its
        // MCP servers), not just the CLI.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start `{prog}`: {e}"))?;
        let path = self.transcript_path();
        let mut transcript =
            std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let names: Vec<&str> = self.servers.iter().map(|s| s.name.as_str()).collect();
        let mut summary = Summary::default();
        let stdout = child.stdout.take().ok_or("no stdout")?;
        // Lines arrive through a channel so a silent CLI can still be timed out.
        let (tx, rx) = mpsc::channel::<std::io::Result<String>>();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let deadline = self.timeout.map(|t| Instant::now() + t);
        let mut timed_out = false;
        loop {
            let next = match deadline {
                Some(d) => match rx.recv_timeout(d.saturating_duration_since(Instant::now())) {
                    Ok(l) => Some(l),
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        timed_out = true;
                        #[cfg(unix)]
                        let _ = Command::new("kill")
                            .args(["-KILL", &format!("-{}", child.id())])
                            .stderr(Stdio::null())
                            .status();
                        let _ = child.kill();
                        None
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => None,
                },
                None => rx.recv().ok(),
            };
            let Some(line) = next else { break };
            let line = line.map_err(|e| e.to_string())?;
            writeln!(transcript, "{line}").map_err(|e| e.to_string())?;
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                for e in parse_event(self.cli, &v, &names) {
                    on_event(&e);
                    summary.record(e);
                }
            }
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        // After a timeout a straggler may still hold the pipe; the reader
        // finishes on its own when it lets go.
        if !timed_out {
            let _ = reader.join();
        }
        Ok(Outcome {
            exit_code: status.code(),
            transcript: path,
            summary,
            timed_out,
        })
    }
}
