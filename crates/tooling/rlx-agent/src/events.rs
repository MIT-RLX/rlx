// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One event shape for every agent CLI's machine-readable output.
//!
//! | CLI | model | tool call | text | end |
//! |---|---|---|---|---|
//! | Claude (`stream-json`) | `system/init.model` | `assistant` `tool_use` block | `assistant` `text` block | `result` (`num_turns`, `total_cost_usd`, `duration_ms`) |
//! | Cursor (`stream-json`) | `system/init.model` | `tool_call/started` (`mcpToolCall.args.{name,args}`) | `assistant` `text` block | `result` (`duration_ms`) |
//! | Copilot (`json`) | `model.call_start.data.model` | `tool.execution_start.data.{toolName,arguments}` | `assistant.message.data.content` | `result` (`usage.sessionDurationMs`) |
//!
//! Tool names are reduced to the bare tool when they carry a known server
//! prefix: `mcp__wsp__describe` (Claude), `wsp-describe` (Cursor, Copilot)
//! → `describe` with `server = Some("wsp")`.

use crate::AgentCli;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Model(String),
    /// The tools the CLI started with (Claude and Cursor report them).
    Tools(Vec<String>),
    ToolCall {
        /// Owning MCP server, when the name carried a known prefix.
        server: Option<String>,
        tool: String,
        input: Value,
    },
    Text(String),
    Done {
        result: Option<String>,
        num_turns: Option<u64>,
        cost_usd: Option<f64>,
        duration_ms: Option<u64>,
    },
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Event::Model(m) => write!(f, "model {m}"),
            Event::Tools(t) => write!(f, "tools {}", t.join(", ")),
            Event::ToolCall { tool, input, .. } => {
                let mut args = input.to_string();
                if args.len() > 120 {
                    let cut = (0..=117)
                        .rev()
                        .find(|&i| args.is_char_boundary(i))
                        .unwrap_or(0);
                    args.truncate(cut);
                    args.push_str("...");
                }
                write!(f, "→ {tool} {args}")
            }
            Event::Text(t) => {
                let line = t.trim().lines().next().unwrap_or("");
                write!(f, "· {}", line.chars().take(160).collect::<String>())
            }
            Event::Done { duration_ms, .. } => match duration_ms {
                Some(ms) => write!(f, "done in {:.1}s", *ms as f64 / 1e3),
                None => write!(f, "done"),
            },
        }
    }
}

/// Events in one line of a CLI's output. `servers` are the MCP server names
/// whose prefix is stripped from tool names.
pub fn parse_event(cli: AgentCli, v: &Value, servers: &[&str]) -> Vec<Event> {
    let mut out = Vec::new();
    let str_of = |x: &Value| x.as_str().map(str::to_string);
    match cli {
        AgentCli::Claude | AgentCli::Cursor => match v["type"].as_str() {
            Some("system") if v["subtype"] == "init" => {
                out.extend(str_of(&v["model"]).map(Event::Model));
                if let Some(tools) = v["tools"].as_array() {
                    out.push(Event::Tools(tools.iter().filter_map(str_of).collect()));
                }
            }
            Some("assistant") => {
                for b in v["message"]["content"].as_array().into_iter().flatten() {
                    match b["type"].as_str() {
                        Some("tool_use") => out.push(tool_call(
                            b["name"].as_str().unwrap_or("?"),
                            b["input"].clone(),
                            servers,
                        )),
                        Some("text") => {
                            if let Some(t) = b["text"].as_str().filter(|t| !t.trim().is_empty()) {
                                out.push(Event::Text(t.to_string()));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("tool_call") if v["subtype"] == "started" => {
                if let Some((kind, body)) = v["tool_call"]
                    .as_object()
                    .and_then(|m| m.iter().find(|(k, _)| k.ends_with("ToolCall")))
                {
                    let (name, input) = if kind == "mcpToolCall" {
                        (
                            body["args"]["name"].as_str().unwrap_or("?").to_string(),
                            body["args"]["args"].clone(),
                        )
                    } else {
                        (
                            kind.trim_end_matches("ToolCall").to_string(),
                            body["args"].clone(),
                        )
                    };
                    out.push(tool_call(&name, input, servers));
                }
            }
            Some("result") => out.push(Event::Done {
                result: str_of(&v["result"]),
                num_turns: v["num_turns"].as_u64(),
                cost_usd: v["total_cost_usd"].as_f64(),
                duration_ms: v["duration_ms"].as_u64(),
            }),
            _ => {}
        },
        AgentCli::Copilot => {
            let d = &v["data"];
            match v["type"].as_str() {
                Some("model.call_start") => out.extend(str_of(&d["model"]).map(Event::Model)),
                Some("tool.execution_start") => out.push(tool_call(
                    d["toolName"].as_str().unwrap_or("?"),
                    d["arguments"].clone(),
                    servers,
                )),
                Some("assistant.message") => {
                    if let Some(t) = d["content"].as_str().filter(|t| !t.trim().is_empty()) {
                        out.push(Event::Text(t.to_string()));
                    }
                }
                Some("result") => out.push(Event::Done {
                    result: None,
                    num_turns: None,
                    cost_usd: None,
                    duration_ms: v["usage"]["sessionDurationMs"].as_u64(),
                }),
                _ => {}
            }
        }
        AgentCli::Codex => {
            let item = &v["item"];
            match (v["type"].as_str(), item["type"].as_str()) {
                (Some("item.started"), Some("mcp_tool_call")) => out.push(Event::ToolCall {
                    server: str_of(&item["server"]),
                    tool: item["tool"].as_str().unwrap_or("?").to_string(),
                    input: item["arguments"].clone(),
                }),
                (Some("item.started"), Some("command_execution")) => out.push(Event::ToolCall {
                    server: None,
                    tool: "shell".into(),
                    input: item["command"].clone(),
                }),
                (Some("item.completed"), Some("agent_message")) => {
                    if let Some(t) = item["text"].as_str().filter(|t| !t.trim().is_empty()) {
                        out.push(Event::Text(t.to_string()));
                    }
                }
                (Some("turn.completed"), _) => out.push(Event::Done {
                    result: None,
                    num_turns: Some(1),
                    cost_usd: None,
                    duration_ms: None,
                }),
                _ => {}
            }
        }
    }
    out
}

fn tool_call(name: &str, input: Value, servers: &[&str]) -> Event {
    for s in servers {
        let claude = format!("mcp__{s}__");
        let dashed = format!("{s}-");
        if let Some(t) = name
            .strip_prefix(&claude)
            .or_else(|| name.strip_prefix(&dashed))
        {
            return Event::ToolCall {
                server: Some(s.to_string()),
                tool: t.to_string(),
                input,
            };
        }
    }
    Event::ToolCall {
        server: None,
        tool: name.to_string(),
        input,
    }
}

/// Running summary of a run's events.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    /// Models seen, in first-use order.
    pub models: Vec<String>,
    /// The tools the CLI reported starting with, when it reports them. A
    /// listed MCP server whose tools are missing here did not load.
    pub tools: Option<Vec<String>>,
    /// Calls per tool; non-MCP tools keep the CLI's own name.
    pub tool_calls: BTreeMap<String, usize>,
    /// Calls to tools outside the listed MCP servers (attempted; the CLI
    /// may have denied them).
    pub foreign_tool_calls: usize,
    /// The agent's last text message.
    pub final_text: Option<String>,
    pub num_turns: Option<u64>,
    pub cost_usd: Option<f64>,
    pub duration_ms: Option<u64>,
}

impl Summary {
    pub fn record(&mut self, e: Event) {
        match e {
            Event::Model(m) => {
                if !self.models.contains(&m) {
                    self.models.push(m);
                }
            }
            Event::Tools(t) => self.tools = Some(t),
            Event::ToolCall { server, tool, .. } => {
                if server.is_none() {
                    self.foreign_tool_calls += 1;
                }
                *self.tool_calls.entry(tool).or_default() += 1;
            }
            Event::Text(t) => self.final_text = Some(t),
            Event::Done {
                result,
                num_turns,
                cost_usd,
                duration_ms,
            } => {
                if self.final_text.is_none() {
                    self.final_text = result;
                }
                self.num_turns = num_turns.or(self.num_turns);
                self.cost_usd = cost_usd.or(self.cost_usd);
                self.duration_ms = duration_ms.or(self.duration_ms);
            }
        }
    }

    pub fn total_tool_calls(&self) -> usize {
        self.tool_calls.values().sum()
    }
}
