// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A stdio MCP server over one folder: the whole file surface of an agent
//! whose CLI runs without built-in tools.
//!
//! Tools: `list_files`, `read_file` (whole, or a range of numbered lines),
//! `search` (text across the files), `write_file`, `edit_file`, and with a
//! check command `check_program` (that command run on the folder, in a
//! process of its own, with a time limit). Paths are relative to the folder;
//! absolute paths, `..` and links out of it are refused. Every file in the
//! folder can be read; only the names given as writable can be written
//! (created or changed); a `.json` file written is parsed at once and a
//! syntax error is in the reply. Newline-delimited JSON-RPC 2.0 on stdin/stdout, as
//! every agent CLI speaks it; tool failures are results the model reads, not
//! protocol errors.
//!
//! Serve it with `rlx-agent workspace-mcp --dir <folder> --write <name>…`,
//! and hand [`Workspace::server`] to an [`AgentRun`](crate::AgentRun).

use std::io::{BufRead, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::McpServer;

const PROTOCOL_VERSION: &str = "2025-06-18";
/// Larger files are read in part; the reply says how much was left out.
const MAX_READ: usize = 200_000;
/// `search` stops after this many matching lines.
const MAX_MATCHES: usize = 200;
/// `check_program`'s output is cut to this many bytes (its end kept).
const MAX_CHECK_OUTPUT: usize = 20_000;
/// How long `check_program`'s command may run.
pub const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

/// One folder, with the names that may be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    pub root: PathBuf,
    pub writable: Vec<String>,
    /// `check_program`'s command (program, arguments…), run with the folder as
    /// its last argument; empty: no such tool.
    pub check: Vec<String>,
}

impl Workspace {
    pub fn new(
        root: impl Into<PathBuf>,
        writable: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            root: root.into(),
            writable: writable.into_iter().map(Into::into).collect(),
            check: Vec::new(),
        }
    }

    /// With `check_program` running `command` (program, arguments…).
    pub fn with_check(mut self, command: Vec<String>) -> Self {
        self.check = command;
        self
    }

    /// `--dir <folder> [--write <name>]… [--check <arg>]…`, the arguments after
    /// `workspace-mcp` (each `--check` one word of the check command).
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let (mut root, mut writable, mut check) = (None, Vec::new(), Vec::new());
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut value = || it.next().cloned().ok_or(format!("{a} needs a value"));
            match a.as_str() {
                "--dir" => root = Some(PathBuf::from(value()?)),
                "--write" => writable.push(value()?),
                "--check" => check.push(value()?),
                other => {
                    return Err(format!(
                        "unknown argument {other:?} (--dir, --write, --check)"
                    ));
                }
            }
        }
        Ok(Self::new(root.ok_or("--dir is required")?, writable).with_check(check))
    }

    /// The MCP server entry that serves this folder: `command` and `prefix`
    /// (for example `rlx-agent`, `["workspace-mcp"]`) then `--dir`, `--write`
    /// and `--check`.
    pub fn server(&self, name: &str, command: impl Into<PathBuf>, prefix: &[String]) -> McpServer {
        let mut args = prefix.to_vec();
        args.extend(["--dir".to_string(), self.root.display().to_string()]);
        for w in &self.writable {
            args.extend(["--write".to_string(), w.clone()]);
        }
        for c in &self.check {
            args.extend(["--check".to_string(), c.clone()]);
        }
        McpServer::new(name, command, args)
    }

    /// The tools in MCP shape.
    pub fn tools(&self) -> Value {
        let path =
            json!({"type": "string", "description": "a path relative to the workspace folder"});
        let reads = json!({"readOnlyHint": true, "openWorldHint": false});
        let writes = json!({"readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false});
        let mut tools = vec![
            json!({"name": "list_files", "description": "Every file in the workspace folder, with its size in bytes.",
             "inputSchema": {"type": "object", "properties": {}}, "annotations": reads}),
            json!({"name": "read_file",
             "description": "A file's text; with `start_line` (from 1) and/or `lines`, those lines, numbered.",
             "inputSchema": {"type": "object", "properties": {"path": path,
                 "start_line": {"type": "integer", "minimum": 1}, "lines": {"type": "integer", "minimum": 1}},
                 "required": ["path"]}, "annotations": reads}),
            json!({"name": "search",
             "description": format!("The lines that contain `text`, as file:line: line, in every file (or those under `path`); at most {MAX_MATCHES}."),
             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "path": path,
                 "ignore_case": {"type": "boolean"}}, "required": ["text"]}, "annotations": reads}),
            json!({"name": "write_file",
             "description": format!("Replace (or create) a file with `content`. Writable: {}.", self.writable_list()),
             "inputSchema": {"type": "object", "properties": {"path": path, "content": {"type": "string"}},
                             "required": ["path", "content"]}, "annotations": writes}),
            json!({"name": "edit_file",
             "description": format!("Replace the one occurrence of `old` with `new` in a file. Writable: {}.", self.writable_list()),
             "inputSchema": {"type": "object", "properties": {"path": path, "old": {"type": "string"}, "new": {"type": "string"}},
                             "required": ["path", "old", "new"]}, "annotations": writes}),
        ];
        if !self.check.is_empty() {
            tools.push(json!({"name": "check_program",
             "description": format!("Build the program as the files are now and run a few updates of it on random params, \
                 in a process of its own (at most {} s): whether it builds, whether its updates stay finite, how far \
                 they move each param. The training run is not touched. Use it before you finish.", CHECK_TIMEOUT.as_secs()),
             "inputSchema": {"type": "object", "properties": {}}, "annotations": reads}));
        }
        Value::Array(tools)
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
        let full = self.root.join(p);
        // A link inside the folder must not lead out of it.
        if let (Ok(real), Ok(root)) = (full.canonicalize(), self.root.canonicalize())
            && !real.starts_with(&root)
        {
            return Err(format!("{path:?} leads outside the workspace"));
        }
        Ok(full)
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
                let start = args["start_line"].as_u64();
                let count = args["lines"].as_u64();
                if start.is_some() || count.is_some() {
                    let start = start.unwrap_or(1).max(1) as usize;
                    let total = text.lines().count();
                    let shown: Vec<String> = text
                        .lines()
                        .enumerate()
                        .skip(start - 1)
                        .take(count.map_or(usize::MAX, |c| c as usize))
                        .map(|(i, l)| format!("{:>6}  {l}", i + 1))
                        .collect();
                    if shown.is_empty() {
                        return Err(format!("{path} has {total} lines"));
                    }
                    let last = start + shown.len() - 1;
                    return Ok(format!(
                        "{}\n[lines {start}–{last} of {total}]",
                        shown.join("\n")
                    ));
                }
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
            "search" => {
                let text = arg("text")?;
                if text.is_empty() {
                    return Err("search needs a non-empty `text`".into());
                }
                let under = args["path"].as_str().map(|p| p.trim_start_matches("./"));
                if let Some(p) = under {
                    self.resolve(p)?;
                }
                let fold = args["ignore_case"].as_bool().unwrap_or(false);
                let needle = if fold {
                    text.to_lowercase()
                } else {
                    text.to_string()
                };
                let mut files = Vec::new();
                walk(&self.root, &self.root, &mut files);
                files.sort();
                let mut hits = Vec::new();
                'files: for (rel, _) in files {
                    if under.is_some_and(|u| !(rel == u || rel.starts_with(&format!("{u}/")))) {
                        continue;
                    }
                    let Ok(body) = std::fs::read_to_string(self.root.join(&rel)) else {
                        continue;
                    };
                    for (i, line) in body.lines().enumerate() {
                        let hay = if fold {
                            line.to_lowercase()
                        } else {
                            line.to_string()
                        };
                        if hay.contains(&needle) {
                            let shown: String = line.chars().take(300).collect();
                            hits.push(format!("{rel}:{}: {shown}", i + 1));
                            if hits.len() == MAX_MATCHES {
                                hits.push(format!("[stopped at {MAX_MATCHES} matches]"));
                                break 'files;
                            }
                        }
                    }
                }
                if hits.is_empty() {
                    Ok(format!("no line contains {text:?}"))
                } else {
                    Ok(hits.join("\n"))
                }
            }
            "write_file" => {
                let (path, content) = (arg("path")?, arg("content")?);
                let full = self.check_writable(path)?;
                std::fs::write(&full, content).map_err(|e| format!("{path}: {e}"))?;
                Ok(format!(
                    "wrote {path} ({} bytes){}",
                    content.len(),
                    json_note(path, content)
                ))
            }
            "edit_file" => {
                let (path, old, new) = (arg("path")?, arg("old")?, arg("new")?);
                let full = self.check_writable(path)?;
                let text = std::fs::read_to_string(&full).map_err(|e| format!("{path}: {e}"))?;
                match text.matches(old).count() {
                    1 => {
                        let edited = text.replacen(old, new, 1);
                        std::fs::write(&full, &edited).map_err(|e| format!("{path}: {e}"))?;
                        Ok(format!("edited {path}{}", json_note(path, &edited)))
                    }
                    0 => Err(format!("{path}: `old` does not occur")),
                    n => Err(format!(
                        "{path}: `old` occurs {n} times; give more context so it occurs once"
                    )),
                }
            }
            "check_program" if !self.check.is_empty() => self.run_check(),
            other => Err(format!("unknown tool {other:?}")),
        }
    }

    /// `check_program`: the check command on the folder → its output (an
    /// error when it fails or runs past [`CHECK_TIMEOUT`]).
    fn run_check(&self) -> Result<String, String> {
        let (prog, args) = self.check.split_first().ok_or("no check command")?;
        let mut cmd = Command::new(prog);
        cmd.args(args)
            .arg(&self.root)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start the check ({prog}): {e}"))?;
        let drain = |r: Option<Box<dyn Read + Send>>| {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                if let Some(mut r) = r {
                    let _ = r.read_to_end(&mut buf);
                }
                buf
            })
        };
        let out = drain(
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let err = drain(
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let deadline = Instant::now() + CHECK_TIMEOUT;
        let status = loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(s) => break Some(s),
                None if Instant::now() >= deadline => {
                    #[cfg(unix)]
                    let _ = Command::new("kill")
                        .args(["-KILL", &format!("-{}", child.id())])
                        .stderr(Stdio::null())
                        .status();
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                None => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        let mut text = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
        let stderr = String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned();
        if !stderr.trim().is_empty() {
            text.push_str(&stderr);
        }
        let text = tail_bytes(text.trim_end(), MAX_CHECK_OUTPUT);
        match status {
            None => Err(format!(
                "the check ran past {} s and was stopped\n{text}",
                CHECK_TIMEOUT.as_secs()
            )),
            Some(s) if s.success() => Ok(text),
            Some(s) => Err(format!("the check failed ({s})\n{text}")),
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

/// For a `.json` file just written: a note when it does not parse.
fn json_note(path: &str, content: &str) -> String {
    if !path.ends_with(".json") {
        return String::new();
    }
    match serde_json::from_str::<Value>(content) {
        Ok(_) => String::new(),
        Err(e) => format!("; it is not valid JSON yet: {e}"),
    }
}

/// The last `max` bytes of `s` (at a character boundary), saying what was cut.
fn tail_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let cut = (s.len() - max..=s.len())
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(s.len());
    format!("[… {cut} bytes before this not shown]\n{}", &s[cut..])
}

/// Every file under `dir` but hidden ones and links, as (path relative to
/// `root`, bytes).
fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        // Hidden files, and links (which may lead outside the folder), are left out.
        let link = e.file_type().is_ok_and(|t| t.is_symlink());
        if link || e.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            walk(root, &path, out);
        } else if let (Ok(rel), Ok(meta)) = (path.strip_prefix(root), e.metadata()) {
            out.push((rel.to_string_lossy().replace('\\', "/"), meta.len()));
        }
    }
}
