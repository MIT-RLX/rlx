// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stdio MCP server over one folder: the whole file surface of an agent
//! whose CLI runs without built-in tools.
//!
//! Tools: `list_files`, `read_file`, `write_file`, `edit_file`. Paths are
//! relative to the folder; absolute paths and `..` are refused. Every file in
//! the folder can be read; only the names given as writable can be written
//! (created or changed). Newline-delimited JSON-RPC 2.0 on stdin/stdout, as
//! every agent CLI speaks it; tool failures are results the model reads, not
//! protocol errors.
//!
//! Serve it with `rlx-agent workspace-mcp --dir <folder> --write <name>…`,
//! and hand [`Workspace::server`] to an [`AgentRun`](crate::AgentRun).

use std::io::{BufRead, Write};
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};

use crate::McpServer;

const PROTOCOL_VERSION: &str = "2025-06-18";
/// Larger files are read in part; the reply says how much was left out.
const MAX_READ: usize = 200_000;

/// One folder, with the names that may be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    pub root: PathBuf,
    pub writable: Vec<String>,
}

impl Workspace {
    pub fn new(
        root: impl Into<PathBuf>,
        writable: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            root: root.into(),
            writable: writable.into_iter().map(Into::into).collect(),
        }
    }

    /// `--dir <folder> [--write <name>]…`, the arguments after `workspace-mcp`.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let (mut root, mut writable) = (None, Vec::new());
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut value = || it.next().cloned().ok_or(format!("{a} needs a value"));
            match a.as_str() {
                "--dir" => root = Some(PathBuf::from(value()?)),
                "--write" => writable.push(value()?),
                other => return Err(format!("unknown argument {other:?} (--dir, --write)")),
            }
        }
        Ok(Self::new(root.ok_or("--dir is required")?, writable))
    }

    /// The MCP server entry that serves this folder: `command` and `prefix`
    /// (for example `rlx-agent`, `["workspace-mcp"]`) then `--dir` and `--write`.
    pub fn server(&self, name: &str, command: impl Into<PathBuf>, prefix: &[String]) -> McpServer {
        let mut args = prefix.to_vec();
        args.extend(["--dir".to_string(), self.root.display().to_string()]);
        for w in &self.writable {
            args.extend(["--write".to_string(), w.clone()]);
        }
        McpServer::new(name, command, args)
    }

    /// The tools in MCP shape.
    pub fn tools(&self) -> Value {
        let path =
            json!({"type": "string", "description": "a path relative to the workspace folder"});
        json!([
            {"name": "list_files", "description": "Every file in the workspace folder, with its size in bytes.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "read_file", "description": "A file's text.",
             "inputSchema": {"type": "object", "properties": {"path": path}, "required": ["path"]}},
            {"name": "write_file",
             "description": format!("Replace (or create) a file with `content`. Writable: {}.", self.writable_list()),
             "inputSchema": {"type": "object", "properties": {"path": path, "content": {"type": "string"}},
                             "required": ["path", "content"]}},
            {"name": "edit_file",
             "description": format!("Replace the one occurrence of `old` with `new` in a file. Writable: {}.", self.writable_list()),
             "inputSchema": {"type": "object", "properties": {"path": path, "old": {"type": "string"}, "new": {"type": "string"}},
                             "required": ["path", "old", "new"]}},
        ])
    }

    fn writable_list(&self) -> String {
        if self.writable.is_empty() {
            "nothing".into()
        } else {
            self.writable.join(", ")
        }
    }

    /// A relative path inside the folder.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let p = Path::new(path);
        let ok = !path.is_empty()
            && p.components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        if !ok {
            return Err(format!(
                "{path:?}: give a path inside the workspace (no absolute paths, no `..`)"
            ));
        }
        Ok(self.root.join(p))
    }

    fn check_writable(&self, path: &str) -> Result<PathBuf, String> {
        let full = self.resolve(path)?;
        let name = path.trim_start_matches("./");
        if !self.writable.iter().any(|w| w == name) {
            return Err(format!(
                "{path:?} is read-only; you can write {}",
                self.writable_list()
            ));
        }
        Ok(full)
    }

    /// One tool call → the text the model reads.
    pub fn call(&self, tool: &str, args: &Value) -> Result<String, String> {
        let arg = |k: &str| {
            args[k]
                .as_str()
                .ok_or(format!("{tool} needs `{k}` (a string)"))
        };
        match tool {
            "list_files" => {
                let mut files = Vec::new();
                walk(&self.root, &self.root, &mut files);
                files.sort();
                Ok(files
                    .into_iter()
                    .map(|(p, n)| {
                        format!(
                            "{p} ({n} bytes){}",
                            if self.writable.contains(&p) {
                                ", writable"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "read_file" => {
                let path = arg("path")?;
                let text = std::fs::read_to_string(self.resolve(path)?)
                    .map_err(|e| format!("{path}: {e}"))?;
                if text.len() <= MAX_READ {
                    return Ok(text);
                }
                let cut = (0..=MAX_READ)
                    .rev()
                    .find(|&i| text.is_char_boundary(i))
                    .unwrap_or(0);
                Ok(format!(
                    "{}\n[… {} more bytes not shown]",
                    &text[..cut],
                    text.len() - cut
                ))
            }
            "write_file" => {
                let (path, content) = (arg("path")?, arg("content")?);
                let full = self.check_writable(path)?;
                std::fs::write(&full, content).map_err(|e| format!("{path}: {e}"))?;
                Ok(format!("wrote {path} ({} bytes)", content.len()))
            }
            "edit_file" => {
                let (path, old, new) = (arg("path")?, arg("old")?, arg("new")?);
                let full = self.check_writable(path)?;
                let text = std::fs::read_to_string(&full).map_err(|e| format!("{path}: {e}"))?;
                match text.matches(old).count() {
                    1 => {
                        std::fs::write(&full, text.replacen(old, new, 1))
                            .map_err(|e| format!("{path}: {e}"))?;
                        Ok(format!("edited {path}"))
                    }
                    0 => Err(format!("{path}: `old` does not occur")),
                    n => Err(format!(
                        "{path}: `old` occurs {n} times; give more context so it occurs once"
                    )),
                }
            }
            other => Err(format!("unknown tool {other:?}")),
        }
    }

    /// Answer one JSON-RPC message; `None` for notifications.
    pub fn handle_message(&self, msg: &Value) -> Option<Value> {
        let id = msg.get("id")?.clone();
        let method = msg["method"].as_str().unwrap_or("");
        let params = &msg["params"];
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": params["protocolVersion"].as_str().unwrap_or(PROTOCOL_VERSION),
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "rlx-agent-workspace", "version": env!("CARGO_PKG_VERSION") },
                "instructions": format!("Files of your workspace folder. You can read every file and write {}.", self.writable_list()),
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": self.tools() })),
            "tools/call" => match params["name"].as_str() {
                None => Err((-32602, "tools/call needs `name`".to_string())),
                Some(tool) => {
                    let (text, is_error) = match self.call(tool, &params["arguments"]) {
                        Ok(t) => (t, false),
                        Err(e) => (e, true),
                    };
                    Ok(
                        json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }),
                    )
                }
            },
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        Some(match result {
            Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message }
            }),
        })
    }

    /// Serve on stdin/stdout until stdin closes.
    pub fn serve(&self) -> Result<(), String> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        for line in stdin.lock().lines() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let reply = match serde_json::from_str::<Value>(&line) {
                Ok(msg) => self.handle_message(&msg),
                Err(e) => Some(json!({
                    "jsonrpc": "2.0", "id": null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                })),
            };
            if let Some(r) = reply {
                let mut out = stdout.lock();
                writeln!(out, "{r}").map_err(|e| e.to_string())?;
                out.flush().map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

/// Every non-hidden file under `dir`, as (path relative to `root`, bytes).
fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        if e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk(root, &path, out);
        } else if let (Ok(rel), Ok(meta)) = (path.strip_prefix(root), e.metadata()) {
            out.push((rel.to_string_lossy().replace('\\', "/"), meta.len()));
        }
    }
}
