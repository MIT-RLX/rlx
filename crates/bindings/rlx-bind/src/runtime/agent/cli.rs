// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-agent-opt`: the command line over run folders and the agent setup.
//!
//! ```text
//! rlx-agent-opt doctor [--live] [--backend claude]   is the agent CLI ready? (--live: one tiny paid call)
//! rlx-agent-opt estimate --total-steps N --sec-per-step S [--sec-per-decision 40] [--max-wait-ratio 4]
//!                        [--decisions K] [--cost-per-decision USD]
//! rlx-agent-opt show <run>                           one line per decision
//! rlx-agent-opt watch <run> [--timeout S]            events as they are written (0: what is there)
//! rlx-agent-opt report <run> [--out page.html]       one HTML page: loss, decisions, rationales, diffs
//! rlx-agent-opt compare <run|results.json> <run|results.json>
//! rlx-agent-opt say <run> <text…>                    a note for the agent's next decision
//! rlx-agent-opt broker <box> [--backend claude] [--model M] [--effort E] [--poll S] [--once]
//! ```
//!
//! [`run`] is the whole tool, so the binary and the bindings (`python -m
//! pyrlx.agent …`) are one implementation.

use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::backend::{Backend, CliAgent, DecisionRequest};
use super::{AgentCli, estimate_decisions, mailbox, records};

/// What the caller fills in for its environment.
#[derive(Clone, Debug, Default)]
pub struct CliDefaults {
    /// The workspace MCP server to use (the bindings serve it themselves).
    pub mcp_command: Option<Vec<String>>,
}

const USAGE: &str = "usage: rlx-agent-opt <doctor|estimate|show|watch|report|compare|say|broker> …
  doctor [--live] [--backend claude|cursor|copilot|codex]
  estimate --total-steps N --sec-per-step S [--sec-per-decision 40] [--max-wait-ratio 4] [--decisions K] [--cost-per-decision USD]
  show <run>
  watch <run> [--timeout S]
  report <run> [--out page.html]
  compare <a> <b>
  say <run> <text…>
  broker <box> [--backend claude] [--model M] [--effort E] [--poll S] [--once]";

/// The binary's `main`: arguments after the program name → exit code.
pub fn main_with(args: &[String], defaults: &CliDefaults) -> i32 {
    let mut out = std::io::stdout();
    match run(args, defaults, &mut out) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("rlx-agent-opt: {e}");
            2
        }
    }
}

/// Parsed `--flag value` options and positional arguments.
struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

/// Flags that take no value.
const SWITCHES: [&str; 2] = ["--live", "--once"];

impl Args {
    fn parse(args: &[String]) -> Result<Self, String> {
        let (mut positional, mut flags) = (Vec::new(), Vec::new());
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                let value = if SWITCHES.contains(&a.as_str()) {
                    None
                } else {
                    Some(it.next().cloned().ok_or(format!("{a} needs a value"))?)
                };
                flags.push((a.clone(), value));
            } else {
                positional.push(a.clone());
            }
        }
        Ok(Self { positional, flags })
    }

    fn get(&self, flag: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(f, _)| f == flag)
            .and_then(|(_, v)| v.as_deref())
    }

    fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|(f, _)| f == flag)
    }

    fn num(&self, flag: &str) -> Result<Option<f64>, String> {
        self.get(flag)
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|_| format!("{flag}: {v:?} is not a number"))
            })
            .transpose()
    }

    fn only(&self, known: &[&str]) -> Result<(), String> {
        for (f, _) in &self.flags {
            if !known.contains(&f.as_str()) {
                return Err(format!("unknown option {f} (known: {})", known.join(", ")));
            }
        }
        Ok(())
    }

    fn path(&self, i: usize, what: &str) -> Result<PathBuf, String> {
        self.positional
            .get(i)
            .map(PathBuf::from)
            .ok_or(format!("give {what}"))
    }
}

/// Run one command, writing its output to `out`.
pub fn run(args: &[String], defaults: &CliDefaults, out: &mut dyn Write) -> Result<(), String> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(USAGE.into());
    };
    let a = Args::parse(rest)?;
    let w = |out: &mut dyn Write, s: String| writeln!(out, "{s}").map_err(|e| e.to_string());
    match cmd.as_str() {
        "doctor" => {
            a.only(&["--live", "--backend"])?;
            doctor(
                a.get("--backend").unwrap_or("claude"),
                a.has("--live"),
                defaults,
                out,
            )
        }
        "estimate" => {
            a.only(&[
                "--total-steps",
                "--sec-per-step",
                "--sec-per-decision",
                "--max-wait-ratio",
                "--decisions",
                "--cost-per-decision",
            ])?;
            let total = a.num("--total-steps")?.ok_or("--total-steps is required")? as u64;
            let sps = a
                .num("--sec-per-step")?
                .ok_or("--sec-per-step is required")?;
            let spd = a.num("--sec-per-decision")?.unwrap_or(40.0);
            let ratio = a.num("--max-wait-ratio")?.unwrap_or(4.0);
            let n = match a.num("--decisions")? {
                Some(k) => k as usize,
                None => estimate_decisions(total, sps, spd, ratio),
            };
            let train = total as f64 * sps;
            let wait = n as f64 * spd;
            w(
                out,
                format!(
                    "decisions: {n}{}",
                    if a.has("--decisions") {
                        ""
                    } else {
                        " (estimated)"
                    }
                ),
            )?;
            w(out, format!("training: {}", hours(train)))?;
            w(
                out,
                format!(
                    "waiting for the agent: {} ({:.2}× the training time; total {})",
                    hours(wait),
                    wait / train.max(1e-9),
                    hours(train + wait)
                ),
            )?;
            if let Some(c) = a.num("--cost-per-decision")? {
                w(out, format!("agent cost: about ${:.2}", c * n as f64))?;
            }
            Ok(())
        }
        "show" => {
            a.only(&[])?;
            show(&a.path(0, "a run folder")?, out)
        }
        "watch" => {
            a.only(&["--timeout"])?;
            let timeout = a.num("--timeout")?.map(Duration::from_secs_f64);
            watch(&a.path(0, "a run folder")?, timeout, out)
        }
        "report" => {
            a.only(&["--out"])?;
            let run = a.path(0, "a run folder")?;
            let page = report(&run)?;
            let dst = a
                .get("--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| run.join("report.html"));
            std::fs::write(&dst, page).map_err(|e| format!("{}: {e}", dst.display()))?;
            w(out, format!("wrote {}", dst.display()))
        }
        "compare" => {
            a.only(&[])?;
            compare(&a.path(0, "two runs")?, &a.path(1, "two runs")?, out)
        }
        "say" => {
            a.only(&[])?;
            let run = a.path(0, "a run folder and the note")?;
            let text = a.positional[1..].join(" ");
            if text.trim().is_empty() {
                return Err("give the note's text".into());
            }
            let p = records::say(&run, &text).map_err(|e| e.message)?;
            w(
                out,
                format!("the agent reads it at its next decision ({})", p.display()),
            )
        }
        "broker" => {
            a.only(&["--backend", "--model", "--effort", "--poll", "--once"])?;
            let dir = a.path(0, "the mailbox folder")?;
            let backend = a.get("--backend").unwrap_or("claude");
            let mut opts = json!({});
            if let Some(m) = a.get("--model") {
                opts["model"] = json!(m);
            }
            if let Some(e) = a.get("--effort") {
                opts["effort"] = json!(e);
            }
            if backend != "keep"
                && let Some(cmd) = &defaults.mcp_command
            {
                opts["mcp_command"] = json!(cmd);
            }
            let poll = Duration::from_secs_f64(a.num("--poll")?.unwrap_or(2.0));
            w(out, format!("serving {} with {backend}", dir.display()))?;
            let n = mailbox::serve_named(&dir, backend, &opts, poll, a.has("--once"), |line| {
                eprintln!("broker: {line}")
            })
            .map_err(|e| e.message)?;
            w(out, format!("answered {n} request(s)"))
        }
        "help" | "--help" | "-h" => w(out, USAGE.into()),
        other => Err(format!("unknown command {other:?}\n{USAGE}")),
    }
}

fn hours(s: f64) -> String {
    if s >= 3600.0 {
        format!("{:.2} h", s / 3600.0)
    } else if s >= 60.0 {
        format!("{:.1} min", s / 60.0)
    } else {
        format!("{s:.1} s")
    }
}

fn doctor(
    backend: &str,
    live: bool,
    defaults: &CliDefaults,
    out: &mut dyn Write,
) -> Result<(), String> {
    let mut w = |s: String| writeln!(out, "{s}").map_err(|e| e.to_string());
    let cli =
        AgentCli::parse(backend).ok_or(format!("--backend: {backend:?} is not an agent CLI"))?;
    let mut ok = true;
    match std::process::Command::new(cli.program())
        .arg("--version")
        .output()
    {
        Ok(o) if o.status.success() => w(format!(
            "{}: {}",
            cli.program(),
            String::from_utf8_lossy(&o.stdout).trim()
        ))?,
        _ => {
            ok = false;
            w(format!(
                "{}: not found on PATH (install it and log in once)",
                cli.program()
            ))?;
        }
    }
    let mut agent = CliAgent::new(cli);
    if let Some(cmd) = &defaults.mcp_command {
        agent.mcp_command = cmd.clone();
    }
    // The workspace server must start and answer `initialize`.
    let dir = std::env::temp_dir().join(format!("rlx-agent-doctor-{}", std::process::id()));
    let _ = std::fs::create_dir_all(dir.join("workspace"));
    let server = server_answers(&agent.mcp_command, &dir.join("workspace"));
    match &server {
        Ok(()) => w(format!(
            "workspace server: {} (answers)",
            agent.mcp_command.join(" ")
        ))?,
        Err(e) => {
            ok = false;
            w(format!(
                "workspace server: {} — {e}",
                agent.mcp_command.join(" ")
            ))?;
        }
    }
    if live && ok {
        agent.timeout = Some(Duration::from_secs(300));
        agent.retries = 0;
        let req = DecisionRequest {
            k: 1,
            workspace: dir.join("workspace"),
            writable: vec![],
            record: dir.clone(),
            message: "This is a setup check of an RLX training run's agent. Reply with the single \
                      word ready and do nothing else."
                .into(),
            system: "You are checking that your tools start. Use none.".into(),
            budget_left: Some(0.5),
        };
        let t0 = Instant::now();
        let r = agent.decide(&req);
        match r.error {
            None => w(format!(
                "live call: answered {:?} in {:.1} s, cost {}",
                r.text.trim(),
                t0.elapsed().as_secs_f64(),
                r.cost_usd
                    .map(|c| format!("${c:.4}"))
                    .unwrap_or("not reported".into())
            ))?,
            Some(e) => {
                ok = false;
                w(format!("live call: failed — {e}"))?;
            }
        }
    } else if !live {
        w("live call: skipped (--live makes one tiny paid call)".into())?;
    }
    let _ = std::fs::remove_dir_all(&dir);
    w(if ok {
        "ready".into()
    } else {
        "not ready".into()
    })?;
    if ok {
        Ok(())
    } else {
        Err("the setup is not ready".into())
    }
}

/// Start the workspace server and exchange `initialize`.
fn server_answers(cmd: &[String], dir: &Path) -> Result<(), String> {
    use std::io::{BufRead, BufReader};
    let (program, prefix) = cmd.split_first().ok_or("empty command")?;
    let mut child = std::process::Command::new(program)
        .args(prefix)
        .args(["--dir", &dir.display().to_string()])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("does not start: {e}"))?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}"#
    )
    .map_err(|e| e.to_string())?;
    drop(stdin);
    let mut line = String::new();
    BufReader::new(child.stdout.take().ok_or("no stdout")?)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    let _ = child.wait();
    let v: Value =
        serde_json::from_str(&line).map_err(|_| format!("answered {:?}", line.trim()))?;
    if v["result"]["protocolVersion"].is_string() {
        Ok(())
    } else {
        Err(format!("answered {v}"))
    }
}

fn events(run: &Path) -> Result<Vec<Value>, String> {
    let path = run.join("events.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

fn response(run: &Path, k: u64) -> Value {
    std::fs::read_to_string(run.join(format!("decisions/{k:03}/response.json")))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null)
}

fn first_line(s: &str, n: usize) -> String {
    let line = s
        .trim()
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut t: String = line.chars().take(n).collect();
    if line.chars().count() > n {
        t.push('…');
    }
    t
}

/// One line per decision: k, update, outcome, version, cost, the rationale's first line.
fn show(run: &Path, out: &mut dyn Write) -> Result<(), String> {
    let ev = events(run)?;
    let mut w = |s: String| writeln!(out, "{s}").map_err(|e| e.to_string());
    if let Some(s) = ev.iter().find(|e| e["kind"] == "started") {
        w(format!("goal: {}", s["goal"].as_str().unwrap_or("?")))?;
    }
    for e in &ev {
        let kind = e["kind"].as_str().unwrap_or("");
        let line = match kind {
            "applied" | "kept" | "rejected" | "failed" => {
                let k = e["k"].as_u64().unwrap_or(0);
                let r = response(run, k);
                let what = match kind {
                    "applied" => format!("applied v{:03}", e["version"].as_u64().unwrap_or(0)),
                    other => other.to_string(),
                };
                format!(
                    "{k:>3}  update {:>7}  {what:<13}  ${:<7.3}  {}",
                    e["update"].as_u64().unwrap_or(0),
                    r["cost_usd"].as_f64().unwrap_or(0.0),
                    first_line(r["text"].as_str().or(e["error"].as_str()).unwrap_or(""), 90)
                )
            }
            "reverted" => format!(
                "     update {:>7}  reverted v{:03}{}: {}",
                e["update"].as_u64().unwrap_or(0),
                e["version"].as_u64().unwrap_or(0),
                if e["exact"] == true { " (exactly)" } else { "" },
                first_line(e["error"].as_str().unwrap_or(""), 80)
            ),
            "closed" => format!(
                "closed: {} decisions, {} applied, final v{:03}, ${:.2}",
                e["decisions"],
                e["applied"],
                e["final_version"].as_u64().unwrap_or(0),
                e["cost_usd"].as_f64().unwrap_or(0.0)
            ),
            _ => continue,
        };
        w(line)?;
    }
    Ok(())
}

fn describe(e: &Value) -> String {
    let kind = e["kind"].as_str().unwrap_or("?");
    let mut fields: Vec<String> = e
        .as_object()
        .map(|m| {
            m.iter()
                .filter(|(k, _)| *k != "kind" && *k != "t")
                .map(|(k, v)| {
                    let mut s = v.to_string();
                    if s.len() > 120 {
                        s.truncate(117);
                        s.push_str("...");
                    }
                    format!("{k}={s}")
                })
                .collect()
        })
        .unwrap_or_default();
    fields.sort();
    format!("{kind:<14} {}", fields.join(" "))
}

/// Print events as they are written; stop at `closed`, or after `timeout` without new ones.
fn watch(run: &Path, timeout: Option<Duration>, out: &mut dyn Write) -> Result<(), String> {
    let mut seen = 0usize;
    let mut last = Instant::now();
    loop {
        let ev = events(run).unwrap_or_default();
        for e in &ev[seen.min(ev.len())..] {
            writeln!(out, "{}", describe(e)).map_err(|e| e.to_string())?;
            out.flush().ok();
            if e["kind"] == "closed" {
                return Ok(());
            }
            last = Instant::now();
        }
        seen = ev.len();
        if timeout.is_some_and(|t| last.elapsed() >= t) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn results(path: &Path) -> Result<Value, String> {
    let file = if path.is_dir() {
        let r = path.join("results.json");
        if r.is_file() {
            r
        } else {
            path.join("summary.json")
        }
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", file.display()))
}

fn compare(a: &Path, b: &Path, out: &mut dyn Write) -> Result<(), String> {
    let (ra, rb) = (results(a)?, results(b)?);
    let keys = [
        "goal",
        "updates",
        "decisions",
        "applied",
        "kept",
        "rejected",
        "reverted",
        "skipped",
        "final_version",
        "cost_usd",
        "final_train_loss",
        "last_validation",
        "best_validation",
        "integrity_flags",
    ];
    writeln!(out, "{:<18} {:<36} {}", "", a.display(), b.display()).map_err(|e| e.to_string())?;
    for k in keys {
        let s = |v: &Value| {
            let mut t = if v.is_null() {
                "-".to_string()
            } else {
                v.to_string()
            };
            if t.len() > 34 {
                t.truncate(33);
                t.push('…');
            }
            t
        };
        writeln!(out, "{k:<18} {:<36} {}", s(&ra[k]), s(&rb[k])).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// ── the HTML report ─────────────────────────────────────────────

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A line diff of two texts (longest common subsequence; programs are small).
pub fn line_diff(a: &str, b: &str) -> Vec<(char, String)> {
    let (x, y): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    let (n, m) = (x.len(), y.len());
    let mut l = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            l[i][j] = if x[i] == y[j] {
                l[i + 1][j + 1] + 1
            } else {
                l[i + 1][j].max(l[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n || j < m {
        if i < n && j < m && x[i] == y[j] {
            out.push((' ', x[i].to_string()));
            i += 1;
            j += 1;
        } else if j < m && (i == n || l[i][j + 1] >= l[i + 1][j]) {
            out.push(('+', y[j].to_string()));
            j += 1;
        } else {
            out.push(('-', x[i].to_string()));
            i += 1;
        }
    }
    out
}

fn program_files(run: &Path, version: u64) -> Vec<(String, String)> {
    let dir = run.join(format!("programs/v{version:03}"));
    let mut files: Vec<(String, String)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            Some((
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read_to_string(e.path()).ok()?,
            ))
        })
        .collect();
    files.sort();
    files
}

/// A polyline of `(x, y)` points scaled into a `width × height` box.
fn svg_series(points: &[(f64, f64)], bounds: (f64, f64, f64, f64), w: f64, h: f64) -> String {
    let (x0, x1, y0, y1) = bounds;
    points
        .iter()
        .map(|(x, y)| {
            let px = (x - x0) / (x1 - x0).max(1e-12) * w;
            let py = h - (y - y0) / (y1 - y0).max(1e-12) * h;
            format!("{px:.1},{py:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// One self-contained HTML page for the run in `run`.
pub fn report(run: &Path) -> Result<String, String> {
    let ev = events(run)?;
    let res = results(run).unwrap_or(Value::Null);
    let goal = ev
        .iter()
        .find(|e| e["kind"] == "started")
        .and_then(|e| e["goal"].as_str())
        .unwrap_or("")
        .to_string();
    // The loss, downsampled to at most 1000 points.
    let losses: Vec<(f64, f64)> = std::fs::read_to_string(run.join("losses.csv"))
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut c = l.split(',');
            Some((
                c.next()?.parse().ok()?,
                c.next()?.parse::<f64>().ok().filter(|v| v.is_finite())?,
            ))
        })
        .collect();
    let step = losses.len().div_ceil(1000).max(1);
    let shown: Vec<(f64, f64)> = losses
        .chunks(step)
        .map(|c| (c[0].0, c.iter().map(|p| p.1).sum::<f64>() / c.len() as f64))
        .collect();
    let validation: Vec<(f64, f64)> = std::fs::read_to_string(run.join("validation.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let v: Value = serde_json::from_str(l).ok()?;
            Some((v["update"].as_f64()?, v["loss"].as_f64()?))
        })
        .collect();
    let all: Vec<&(f64, f64)> = shown.iter().chain(validation.iter()).collect();
    let mut html = String::new();
    let _ = write!(
        html,
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>RLX agent run</title><style>\
         body{{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#1d2433}}\
         h1{{font-size:1.4em}} h2{{font-size:1.1em;margin-top:2em}} table{{border-collapse:collapse}}\
         td,th{{padding:.25em .7em;border-bottom:1px solid #e3e6ec;text-align:left;vertical-align:top}}\
         pre{{background:#f6f7f9;padding:.6em;overflow-x:auto;font-size:12px}} .add{{color:#116329}} .del{{color:#a40e26}}\
         .k{{color:#5b6476}} svg{{background:#fbfbfc;border:1px solid #e3e6ec}}</style></head><body>\
         <h1>RLX agent run</h1><p>{}</p>",
        esc(&goal)
    );
    if res.is_object() {
        html.push_str("<table>");
        for k in [
            "updates",
            "decisions",
            "applied",
            "kept",
            "rejected",
            "reverted",
            "skipped",
            "final_version",
            "cost_usd",
            "final_train_loss",
            "best_validation",
            "integrity_flags",
        ] {
            if !res[k].is_null() {
                let _ = write!(
                    html,
                    "<tr><th>{k}</th><td>{}</td></tr>",
                    esc(&res[k].to_string())
                );
            }
        }
        html.push_str("</table>");
    }
    if !all.is_empty() {
        let x0 = all.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let x1 = all.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let y0 = all.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let y1 = all.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let (w, h) = (1000.0, 300.0);
        let _ = write!(
            html,
            "<h2>Loss</h2><svg viewBox=\"-40 -10 1060 330\" width=\"100%\"><polyline fill=\"none\" stroke=\"#2f6fdf\" \
             stroke-width=\"1.5\" points=\"{}\"/>",
            svg_series(&shown, (x0, x1, y0, y1), w, h)
        );
        for p in &validation {
            let s = svg_series(&[*p], (x0, x1, y0, y1), w, h);
            let (cx, cy) = s.split_once(',').unwrap_or(("0", "0"));
            let _ = write!(
                html,
                "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"3.5\" fill=\"#d9822b\"/>"
            );
        }
        for e in ev
            .iter()
            .filter(|e| e["kind"] == "applied" || e["kind"] == "reverted")
        {
            if let Some(u) = e["update"].as_f64() {
                let px = (u - x0) / (x1 - x0).max(1e-12) * w;
                let color = if e["kind"] == "applied" {
                    "#8a94a6"
                } else {
                    "#a40e26"
                };
                let _ = write!(
                    html,
                    "<line x1=\"{px:.1}\" x2=\"{px:.1}\" y1=\"0\" y2=\"{h}\" stroke=\"{color}\" stroke-dasharray=\"3,3\"/>"
                );
            }
        }
        let _ = write!(
            html,
            "<text x=\"-36\" y=\"8\" font-size=\"11\">{y1:.4}</text><text x=\"-36\" y=\"{h}\" font-size=\"11\">{y0:.4}</text>\
             <text x=\"0\" y=\"318\" font-size=\"11\">update {x0}</text><text x=\"920\" y=\"318\" font-size=\"11\">update {x1}</text></svg>\
             <p class=\"k\">Blue: training loss; orange: validation loss; grey lines: a new program took over; red: a revert.</p>"
        );
    }
    html.push_str("<h2>Decisions</h2>");
    let mut version_seen = 0u64;
    for e in &ev {
        let kind = e["kind"].as_str().unwrap_or("");
        match kind {
            "applied" | "kept" | "rejected" | "failed" => {
                let k = e["k"].as_u64().unwrap_or(0);
                let r = response(run, k);
                let _ = write!(
                    html,
                    "<h3>Decision {k} <span class=\"k\">— update {}, {kind}, ${:.3}, {:.0} s</span></h3><pre>{}</pre>",
                    e["update"],
                    r["cost_usd"].as_f64().unwrap_or(0.0),
                    r["seconds"].as_f64().unwrap_or(0.0),
                    esc(r["text"].as_str().or(e["error"].as_str()).unwrap_or(""))
                );
                if kind == "applied"
                    && let Some(v) = e["version"].as_u64()
                {
                    let before = program_files(run, version_seen);
                    for (name, text) in program_files(run, v) {
                        let old = before
                            .iter()
                            .find(|(n, _)| *n == name)
                            .map(|(_, t)| t.as_str())
                            .unwrap_or("");
                        if old == text {
                            continue;
                        }
                        let _ = write!(
                            html,
                            "<p class=\"k\">{} (v{version_seen:03} → v{v:03})</p><pre>",
                            esc(&name)
                        );
                        for (op, line) in line_diff(old, &text) {
                            let class = match op {
                                '+' => " class=\"add\"",
                                '-' => " class=\"del\"",
                                _ => "",
                            };
                            let _ = writeln!(html, "<span{class}>{op} {}</span>", esc(&line));
                        }
                        html.push_str("</pre>");
                    }
                    version_seen = v;
                }
            }
            "reverted" => {
                let _ = write!(
                    html,
                    "<p class=\"del\">Update {}: v{:03} reverted{}.</p><pre>{}</pre>",
                    e["update"],
                    e["version"].as_u64().unwrap_or(0),
                    if e["exact"] == true {
                        " (the failing update undone exactly)"
                    } else {
                        ""
                    },
                    esc(e["error"].as_str().unwrap_or(""))
                );
                if let Ok(f) = std::fs::read_to_string(run.join("FINAL"))
                    && let Some(v) = f.trim().strip_prefix('v').and_then(|v| v.parse().ok())
                {
                    version_seen = version_seen.max(v);
                }
            }
            _ => {}
        }
    }
    html.push_str("</body></html>\n");
    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_diffs_keep_common_lines() {
        let d = line_diff("a\nb\nc", "a\nx\nc");
        assert_eq!(
            d,
            vec![
                (' ', "a".into()),
                ('+', "x".into()),
                ('-', "b".into()),
                (' ', "c".into())
            ]
        );
    }

    #[test]
    fn estimate_prints_the_budget() {
        let mut out = Vec::new();
        let args: Vec<String> = [
            "estimate",
            "--total-steps",
            "1000",
            "--sec-per-step",
            "0.1",
            "--cost-per-decision",
            "0.12",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        run(&args, &CliDefaults::default(), &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("decisions: 10 (estimated)"), "{s}");
        assert!(s.contains("4.00× the training time"), "{s}");
        assert!(s.contains("about $1.20"), "{s}");
        let bad: Vec<String> = ["estimate", "--total-steps", "x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(run(&bad, &CliDefaults::default(), &mut Vec::new()).is_err());
    }
}
