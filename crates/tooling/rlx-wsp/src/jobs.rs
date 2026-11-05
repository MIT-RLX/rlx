// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Background jobs: a measurement runs while the agent keeps working.
//!
//! A call with `"async": true` to one of [`ASYNC_TOOLS`] is charged at once
//! (its forward passes, as when it runs in the foreground) and handed to a
//! child process (`rlx-wsp job`) with a snapshot of the space as it is now:
//! the job measures the weights of the moment it was submitted, whatever
//! the agent edits meanwhile. `jobs` lists them and collects a result, which
//! brings back what the call leaves behind (a `contrast` vector, the
//! answers `contrast {last_eval}` reads, the tokens it used). The child
//! opens its own copy of the model, so by default one job runs at a time.

use crate::space::WeightSpace;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Tools that may run in the background: measurements, no weight edits.
pub const ASYNC_TOOLS: &[&str] = &[
    "qa_eval",
    "qa_loglik",
    "generate",
    "trace",
    "node_stats",
    "lens",
    "contrast",
    "compare",
    "ablate",
    "sensitivity",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: usize,
    pub tool: String,
    pub call: Value,
    pub dir: String,
    pub pid: u32,
    /// `running` | `done` | `failed` | `collected` | `cancelled`.
    pub status: String,
    pub passes: usize,
}

/// Forward passes `call` takes.
pub fn cost(ws: &WeightSpace, call: &Value) -> Option<usize> {
    let tool = call["tool"].as_str()?;
    if !ASYNC_TOOLS.contains(&tool) {
        return None;
    }
    Some(match tool {
        "contrast" | "ablate" => 2,
        "compare" => {
            let guard = ws.state.env == crate::env::EnvKind::Qa
                && call["guard"]
                    .as_bool()
                    .unwrap_or_else(|| crate::measure::has_guard(ws));
            if guard { 4 } else { 2 }
        }
        "sensitivity" => 1 + call["tensors"].as_array().map_or(0, Vec::len),
        _ => 1,
    })
}

fn jobs_dir() -> PathBuf {
    std::env::temp_dir().join(format!("rlx-wsp-jobs-{}", std::process::id()))
}

/// Start `call` in the background; returns its job id.
pub fn submit(ws: &mut WeightSpace, mut call: Value) -> Result<Value, String> {
    if let Some(o) = call.as_object_mut() {
        o.remove("async");
    }
    let tool = call["tool"].as_str().unwrap_or_default().to_string();
    let passes = cost(ws, &call).ok_or_else(|| {
        format!(
            "`{tool}` cannot run in the background (these can: {})",
            ASYNC_TOOLS.join(", ")
        )
    })?;
    refresh(ws);
    let running = ws
        .state
        .jobs
        .iter()
        .filter(|j| j.status == "running")
        .count();
    if running >= ws.state.max_jobs {
        return Err(format!(
            "{running} job(s) already running (at most {}); collect or wait with `jobs`",
            ws.state.max_jobs
        ));
    }
    if ws.remaining() < passes {
        return Err(format!(
            "`{tool}` needs {passes} forward passes ({} left)",
            ws.remaining()
        ));
    }
    ws.check_tokens()?;
    let id = ws.state.jobs.len() + 1;
    for _ in 0..passes {
        ws.charge(&format!("{tool} (job {id})"))?;
    }
    // The snapshot: this moment's weights, an open budget (charged above).
    let mut st = ws.state.clone();
    st.budget = usize::MAX / 4;
    st.used = 0;
    st.jobs.clear();
    st.token_budget = ws
        .state
        .token_budget
        .map(|b| b.saturating_sub(ws.state.tokens_used));
    st.tokens_used = 0;
    let dir = jobs_dir().join(format!("job-{id}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let snap = WeightSpace::from_state(st);
    std::fs::write(dir.join("state.json"), snap.to_json()?).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("last_qa.json"),
        serde_json::to_string(&ws.lm.last_qa).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let log = std::fs::File::create(dir.join("log.txt")).map_err(|e| e.to_string())?;
    // The `rlx-wsp` executable (`RLX_WSP_EXE` where this process is not it,
    // e.g. a test harness).
    let exe = match std::env::var_os("RLX_WSP_EXE") {
        Some(p) => PathBuf::from(p),
        None => std::env::current_exe().map_err(|e| e.to_string())?,
    };
    let child = std::process::Command::new(exe)
        .arg("job")
        .arg("--state")
        .arg(dir.join("state.json"))
        .arg("--out")
        .arg(&dir)
        .arg(call.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .spawn()
        .map_err(|e| format!("starting the job: {e}"))?;
    let pid = child.id();
    ws.children.insert(id, child);
    ws.state.jobs.push(Job {
        id,
        tool: tool.clone(),
        call,
        dir: dir.display().to_string(),
        pid,
        status: "running".into(),
        passes,
    });
    Ok(json!({ "job": id, "tool": tool, "passes_charged": passes,
               "note": "measures the weights as they are now; collect with jobs {collect: id} (or wait: true)" }))
}

/// Update `running` jobs that finished.
fn refresh(ws: &mut WeightSpace) {
    for j in ws.state.jobs.iter_mut().filter(|j| j.status == "running") {
        let dir = Path::new(&j.dir);
        if dir.join("result.json").exists() {
            j.status = "done".into();
        } else if let Some(c) = ws.children.get_mut(&j.id)
            && let Ok(Some(_)) = c.try_wait()
        {
            j.status = "failed".into();
        }
    }
}

/// The child: run one call on a snapshot and write what it leaves behind.
pub fn run_child(state: &Path, out: &Path, call: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(state).map_err(|e| e.to_string())?;
    let st: crate::State = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut ws = WeightSpace::from_state(st);
    if let Ok(t) = std::fs::read_to_string(out.join("last_qa.json")) {
        ws.lm.last_qa = serde_json::from_str(&t).unwrap_or_default();
    }
    let vectors_before: Vec<String> = ws.state.vectors.keys().cloned().collect();
    let t0 = std::time::Instant::now();
    let reply = crate::tools::handle_json(&mut ws, call);
    let vectors: serde_json::Map<String, Value> = ws
        .state
        .vectors
        .iter()
        .filter(|(k, _)| !vectors_before.contains(k))
        .map(|(k, v)| (k.clone(), json!(v)))
        .collect();
    let result = json!({
        "reply": reply,
        "seconds": t0.elapsed().as_secs_f64(),
        "tokens_used": ws.state.tokens_used,
        "vectors": vectors,
        "last_qa": ws.lm.last_qa,
    });
    let tmp = out.join("result.tmp");
    std::fs::write(&tmp, result.to_string()).map_err(|e| e.to_string())?;
    std::fs::rename(tmp, out.join("result.json")).map_err(|e| e.to_string())
}

/// The `jobs` tool: list, collect (optionally waiting), or cancel.
pub fn jobs(
    ws: &mut WeightSpace,
    collect: Option<usize>,
    wait: bool,
    cancel: Option<usize>,
) -> Result<Value, String> {
    if let Some(id) = cancel {
        let j = ws
            .state
            .jobs
            .iter_mut()
            .find(|j| j.id == id)
            .ok_or_else(|| format!("no job {id}"))?;
        if j.status == "running" {
            if let Some(mut c) = ws.children.remove(&id) {
                let _ = c.kill();
                let _ = c.wait();
            }
            j.status = "cancelled".into();
        }
    }
    let mut collected = Value::Null;
    if let Some(id) = collect {
        let idx = ws
            .state
            .jobs
            .iter()
            .position(|j| j.id == id)
            .ok_or_else(|| format!("no job {id}"))?;
        loop {
            refresh(ws);
            if ws.state.jobs[idx].status != "running" || !wait {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        let j = ws.state.jobs[idx].clone();
        let dir = Path::new(&j.dir);
        collected = match j.status.as_str() {
            "done" | "collected" => {
                let r: Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("result.json")).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                if j.status == "done" {
                    ws.state.tokens_used += r["tokens_used"].as_u64().unwrap_or(0) as usize;
                    if let Some(vs) = r["vectors"].as_object() {
                        for (k, v) in vs {
                            if let Ok(v) = serde_json::from_value::<Vec<f32>>(v.clone()) {
                                ws.state.vectors.insert(k.clone(), v);
                            }
                        }
                    }
                    if let Ok(q) =
                        serde_json::from_value::<Vec<(String, String, bool)>>(r["last_qa"].clone())
                        && !q.is_empty()
                    {
                        ws.lm.last_qa = q;
                    }
                    ws.state.jobs[idx].status = "collected".into();
                    ws.state
                        .log
                        .push(format!("job {id} ({}) collected", j.tool));
                    ws.children.remove(&id);
                }
                json!({ "job": id, "tool": j.tool, "seconds": r["seconds"], "result": r["reply"] })
            }
            "failed" => {
                let log = std::fs::read_to_string(dir.join("log.txt")).unwrap_or_default();
                let tail: String = log
                    .chars()
                    .rev()
                    .take(800)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                json!({ "job": id, "status": "failed", "log_end": tail })
            }
            s => json!({ "job": id, "status": s }),
        };
    }
    refresh(ws);
    Ok(json!({
        "collected": collected,
        "jobs": ws.state.jobs.iter().map(|j| json!({
            "job": j.id, "tool": j.tool, "status": j.status, "passes": j.passes,
        })).collect::<Vec<_>>(),
    }))
}

/// Stop every running job (end of a run).
pub fn stop_all(ws: &mut WeightSpace) {
    for j in ws.state.jobs.iter_mut().filter(|j| j.status == "running") {
        match ws.children.remove(&j.id) {
            Some(mut c) => {
                let _ = c.kill();
                let _ = c.wait();
            }
            None => {
                let _ = std::process::Command::new("kill")
                    .arg(j.pid.to_string())
                    .status();
            }
        }
        j.status = "cancelled".into();
    }
}
