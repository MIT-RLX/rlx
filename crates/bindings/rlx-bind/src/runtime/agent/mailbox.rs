// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent on another machine, through a shared folder: for a GPU node
//! with no internet or no agent login.
//!
//! The training side's [`Mailbox`] backend writes each decision to
//! `<box>/requests/<id>/` (the message, the system prompt and a copy of the
//! workspace), made visible by one rename so a reader never sees half of it,
//! and waits for `<box>/responses/<id>/`. A broker — [`serve`], or
//! `rlx-agent-opt broker <box>` — claims each request by renaming it into
//! `<box>/claimed/`, keeps one workspace per run under `<box>/broker/<run>/`
//! (so a Claude session resumes across decisions), answers with its own
//! backend, and writes the response the same way. Only the files the agent
//! may write come back.
//!
//! The folder can be shared any way that keeps renames atomic: one machine,
//! NFS, or a folder kept in sync over ssh.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::backend::{Backend, DecisionOutcome, DecisionRequest, backend_from_json};
use crate::runtime::opts::Opts;
use crate::{ApiError, ApiResult};

const REQUESTS: &str = "requests";
const CLAIMED: &str = "claimed";
const RESPONSES: &str = "responses";
const BROKER: &str = "broker";

/// The training side of a mailbox.
pub struct Mailbox {
    pub dir: PathBuf,
    pub poll: Duration,
    /// Give up waiting for a response after this long.
    pub timeout: Option<Duration>,
}

impl Mailbox {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            poll: Duration::from_secs(2),
            timeout: None,
        }
    }

    /// `{dir, poll_seconds, timeout_seconds}`; `dir` is required.
    pub fn from_json(v: &Value) -> ApiResult<Self> {
        let o = Opts::new("mailbox backend", v)?;
        let dir = o.str("dir")?.ok_or_else(|| {
            ApiError::type_("mailbox backend: `dir` (the shared folder) is required")
        })?;
        let mut m = Self::new(dir);
        if let Some(p) = o.f64("poll_seconds")? {
            m.poll = Duration::from_secs_f64(p.max(0.001));
        }
        m.timeout = o
            .f64("timeout_seconds")?
            .map(|s| Duration::from_secs_f64(s.max(0.0)));
        o.finish()?;
        Ok(m)
    }
}

/// The request's id: `<run>-<k>`, from its record folder (`<run>/decisions/<k>`).
fn request_id(req: &DecisionRequest) -> String {
    let run = req
        .record
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "run".into());
    format!("{run}-{:03}", req.k)
}

fn io(what: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", what.display())
}

/// Copy every file under `from` into `to` (folders created as needed).
fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(io(to))?;
    for e in fs::read_dir(from).map_err(io(from))?.flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_tree(&src, &dst)?;
        } else {
            fs::copy(&src, &dst).map_err(io(&src))?;
        }
    }
    Ok(())
}

/// Write `folder` under a temporary name, then make it appear with one rename.
fn publish(
    parent: &Path,
    id: &str,
    fill: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    fs::create_dir_all(parent).map_err(io(parent))?;
    let tmp = parent.join(format!(".{id}.tmp"));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(io(&tmp))?;
    fill(&tmp)?;
    let dst = parent.join(id);
    let _ = fs::remove_dir_all(&dst);
    fs::rename(&tmp, &dst).map_err(io(&dst))
}

impl Backend for Mailbox {
    fn name(&self) -> &str {
        "mailbox"
    }

    fn decide(&mut self, req: &DecisionRequest) -> DecisionOutcome {
        let t0 = Instant::now();
        let id = request_id(req);
        let fail = |e: String| DecisionOutcome {
            error: Some(e),
            seconds: t0.elapsed().as_secs_f64(),
            ..Default::default()
        };
        let sent = publish(&self.dir.join(REQUESTS), &id, |d| {
            let r = json!({
                "id": id, "k": req.k, "message": req.message, "system": req.system,
                "writable": req.writable, "budget_left": req.budget_left,
            });
            fs::write(d.join("request.json"), r.to_string()).map_err(io(d))?;
            copy_tree(&req.workspace, &d.join("workspace"))
        });
        if let Err(e) = sent {
            return fail(format!("mailbox: writing the request: {e}"));
        }
        let response = self.dir.join(RESPONSES).join(&id);
        while !response.join("response.json").is_file() {
            if self.timeout.is_some_and(|t| t0.elapsed() > t) {
                return fail(format!(
                    "mailbox: no response in {} after {:.0} s; is a broker serving it?",
                    self.dir.display(),
                    t0.elapsed().as_secs_f64()
                ));
            }
            std::thread::sleep(self.poll);
        }
        let read = || -> Result<DecisionOutcome, String> {
            let path = response.join("response.json");
            let v: Value = serde_json::from_str(&fs::read_to_string(&path).map_err(io(&path))?)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if v["error"].is_null() {
                for name in &req.writable {
                    let src = response.join("workspace").join(name);
                    if src.is_file() {
                        fs::copy(&src, req.workspace.join(name)).map_err(io(&src))?;
                    }
                }
            }
            let _ = fs::write(req.record.join("mailbox_response.json"), v.to_string());
            Ok(DecisionOutcome {
                text: v["text"].as_str().unwrap_or_default().to_string(),
                cost_usd: v["cost_usd"].as_f64(),
                seconds: t0.elapsed().as_secs_f64(),
                error: v["error"].as_str().map(str::to_string),
            })
        };
        let out = read().unwrap_or_else(|e| fail(format!("mailbox: reading the response: {e}")));
        let _ = fs::remove_dir_all(&response);
        out
    }
}

/// Serve the mailbox in `dir` with backends made by `make` (one per run, so
/// sessions stay apart): answer every request that arrives, polling every
/// `poll`; with `once`, return after the first answered request. Returns how
/// many requests were answered.
pub fn serve(
    dir: &Path,
    mut make: impl FnMut() -> ApiResult<Box<dyn Backend>>,
    poll: Duration,
    once: bool,
    mut log: impl FnMut(&str),
) -> ApiResult<usize> {
    let err = |e: String| ApiError::runtime(format!("mailbox broker: {e}"));
    for sub in [REQUESTS, CLAIMED, RESPONSES, BROKER] {
        fs::create_dir_all(dir.join(sub)).map_err(|e| err(e.to_string()))?;
    }
    let mut backends: HashMap<String, Box<dyn Backend>> = HashMap::new();
    let mut answered = 0usize;
    loop {
        let mut ids: Vec<String> = fs::read_dir(dir.join(REQUESTS))
            .map_err(|e| err(e.to_string()))?
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.starts_with('.'))
            .collect();
        ids.sort();
        for id in ids {
            let claimed = dir.join(CLAIMED).join(&id);
            // The rename is the claim: of two brokers, one wins.
            if fs::rename(dir.join(REQUESTS).join(&id), &claimed).is_err() {
                continue;
            }
            let req: Value = fs::read_to_string(claimed.join("request.json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .ok_or_else(|| err(format!("{}: unreadable request", claimed.display())))?;
            let run = id
                .rsplit_once('-')
                .map(|(r, _)| r.to_string())
                .unwrap_or(id.clone());
            let home = dir.join(BROKER).join(&run);
            let workspace = home.join("workspace");
            // One workspace per run, refreshed from the request, so the CLI
            // resumes its session in the same folder every time.
            copy_tree(&claimed.join("workspace"), &workspace).map_err(err)?;
            let record = home.join("decisions").join(&id);
            fs::create_dir_all(&record).map_err(|e| err(e.to_string()))?;
            let writable: Vec<String> = req["writable"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|w| w.as_str().map(str::to_string))
                .collect();
            let request = DecisionRequest {
                k: req["k"].as_u64().unwrap_or(0) as usize,
                workspace: workspace.clone(),
                writable: writable.clone(),
                record,
                message: req["message"].as_str().unwrap_or_default().to_string(),
                system: req["system"].as_str().unwrap_or_default().to_string(),
                budget_left: req["budget_left"].as_f64(),
            };
            log(&format!("answering {id}"));
            let backend = match backends.get_mut(&run) {
                Some(b) => b,
                None => backends.entry(run.clone()).or_insert(make()?),
            };
            let out = backend.decide(&request);
            log(&format!(
                "{id}: {}",
                out.error.as_deref().unwrap_or("answered")
            ));
            publish(&dir.join(RESPONSES), &id, |d| {
                let r = json!({"text": out.text, "cost_usd": out.cost_usd, "seconds": out.seconds,
                               "error": out.error});
                fs::write(d.join("response.json"), r.to_string()).map_err(io(d))?;
                let ws = d.join("workspace");
                fs::create_dir_all(&ws).map_err(io(&ws))?;
                for name in &writable {
                    let src = workspace.join(name);
                    if src.is_file() {
                        fs::copy(&src, ws.join(name)).map_err(io(&src))?;
                    }
                }
                Ok(())
            })
            .map_err(err)?;
            let _ = fs::remove_dir_all(&claimed);
            answered += 1;
            if once {
                return Ok(answered);
            }
        }
        std::thread::sleep(poll);
    }
}

/// [`serve`] with backends by name, as the command line chooses them.
pub fn serve_named(
    dir: &Path,
    backend: &str,
    options: &Value,
    poll: Duration,
    once: bool,
    log: impl FnMut(&str),
) -> ApiResult<usize> {
    if backend == "mailbox" {
        return Err(ApiError::type_(
            "a broker answers with an agent, not another mailbox",
        ));
    }
    backend_from_json(backend, options)?;
    serve(dir, || backend_from_json(backend, options), poll, once, log)
}
