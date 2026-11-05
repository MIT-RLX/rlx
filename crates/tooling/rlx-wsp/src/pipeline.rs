// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One command from a packaged model and a question set to a comparison of
//! what improves it: the unedited score, an agent editing the weights,
//! LoRA on the reference solutions and LoRA on the model's own answers
//! given with worked examples.
//!
//! Each stage runs as a child `rlx-wsp` (or the export command), one at a
//! time, its output in `<out>/logs/<stage>.log`, its outcome in
//! `<out>/pipeline.json`. Running the same command again skips what is
//! done and retries what failed, so a stopped pipeline goes on where it
//! was. A stage whose input failed is skipped; the rest still run, and the
//! summary says what each stage came to. Nothing here knows a model: the
//! model-specific part is the package, and the optional command that
//! exports its trainable form.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// The stages, in order.
pub const STAGES: [&str; 8] = [
    "base",
    "baseline",
    "agent",
    "distill",
    "export",
    "lora-reference",
    "lora-distill",
    "summary",
];

/// What a pipeline runs.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub out: PathBuf,
    /// The `rlx-wsp` to run stages with.
    pub exe: PathBuf,
    /// Flags of the task: package, questions, guards.
    pub task: Vec<String>,
    /// Flags of the agent run (`--cli`, `--budget`, `--goal`, …).
    pub agent: Vec<String>,
    /// Flags of `qa-distill` (`--shots`, `--questions`).
    pub distill: Vec<String>,
    /// Flags of both LoRA runs (`--steps`, `--rank`, …).
    pub lora: Vec<String>,
    /// A trainable package, when there is one already.
    pub train_package: Option<String>,
    /// A shell command that writes the trainable package to `{out}`.
    pub train_export: Option<String>,
    /// Run only these stages (empty: all).
    pub only: Vec<String>,
    /// Run these stages again even if done.
    pub redo: Vec<String>,
    /// Questions the held-out scores are out of.
    pub heldout_n: usize,
}

/// What one stage runs.
enum Step {
    Wsp(Vec<String>),
    Shell(String),
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

impl Plan {
    fn dir(&self, name: &str) -> String {
        self.out.join(name).display().to_string()
    }

    fn state_of(&self, name: &str) -> String {
        self.out.join(name).join("state.json").display().to_string()
    }

    fn report_of(&self, name: &str) -> String {
        self.out
            .join(name)
            .join("report.json")
            .display()
            .to_string()
    }

    fn train_path(&self) -> String {
        self.train_package
            .clone()
            .unwrap_or_else(|| self.out.join("train.rlxp").display().to_string())
    }

    fn can_train(&self) -> bool {
        self.train_package.is_some() || self.train_export.is_some()
    }

    /// Stages that must have worked first.
    fn needs(stage: &str) -> &'static [&'static str] {
        match stage {
            "baseline" | "agent" | "distill" => &["base"],
            "lora-reference" => &["base", "export"],
            "lora-distill" => &["base", "export", "distill"],
            _ => &[],
        }
    }

    /// Why `stage` does not apply to this plan.
    fn not_applicable(&self, stage: &str) -> Option<String> {
        let export = stage == "export" || stage.starts_with("lora");
        (export && !self.can_train())
            .then(|| "no --train-package or --train-export: nothing to train with".to_string())
            .or_else(|| {
                (stage == "export" && self.train_package.is_some())
                    .then(|| "the trainable package is given".to_string())
            })
    }

    fn score(&self, name: &str) -> Step {
        let mut a = s(&["score", "--state"]);
        a.push(self.state_of(name));
        a.extend(s(&["--check-perturbed", "--check-truncation", "--out"]));
        a.push(self.report_of(name));
        Step::Wsp(a)
    }

    fn lora(&self, name: &str, items: Option<String>) -> Vec<Step> {
        let mut a = s(&["lora-train", "--method", "backprop", "--state"]);
        a.push(self.state_of("base"));
        a.extend(["--train-package".to_string(), self.train_path()]);
        a.extend(self.lora.iter().cloned());
        if let Some(f) = items {
            a.extend(["--train-items".to_string(), f]);
        }
        a.extend(["--out".to_string(), self.dir(name)]);
        vec![Step::Wsp(a), self.score(name)]
    }

    /// The agent's stage: a run that ended with its scoring unfinished is
    /// finished, not run again.
    fn agent_steps(&self) -> Vec<Step> {
        let t = self.out.join("agent").join("transcript.jsonl");
        let ended = std::fs::read_to_string(t).is_ok_and(|x| x.contains("\"type\":\"result\""));
        let cli: Vec<String> = self
            .agent
            .windows(2)
            .find(|w| w[0] == "--cli")
            .map(|w| w.to_vec())
            .unwrap_or_default();
        let mut a = if ended {
            let mut a = s(&["finish", "--run"]);
            a.push(self.dir("agent"));
            a.extend(cli);
            a
        } else {
            let mut a = s(&["agent", "--from-state"]);
            a.push(self.dir("base"));
            a.extend(self.agent.iter().cloned());
            a.extend(["--out".to_string(), self.dir("agent")]);
            a
        };
        a.shrink_to_fit();
        vec![Step::Wsp(a)]
    }

    fn steps(&self, stage: &str) -> Vec<Step> {
        match stage {
            "base" => {
                let mut a = s(&["new", "--state"]);
                a.push(self.state_of("base"));
                a.extend(self.task.iter().cloned());
                vec![Step::Wsp(a)]
            }
            "baseline" => vec![self.score("base")],
            "agent" => self.agent_steps(),
            "distill" => {
                let mut a = s(&["qa-distill", "--state"]);
                a.push(self.state_of("base"));
                a.extend(self.distill.iter().cloned());
                a.extend([
                    "--out".to_string(),
                    self.out.join("distill.jsonl").display().to_string(),
                ]);
                vec![Step::Wsp(a)]
            }
            "export" => {
                let cmd = self.train_export.clone().unwrap_or_default();
                vec![Step::Shell(cmd.replace("{out}", &self.train_path()))]
            }
            "lora-reference" => self.lora("lora-reference", None),
            "lora-distill" => self.lora(
                "lora-distill",
                Some(self.out.join("distill.jsonl").display().to_string()),
            ),
            _ => Vec::new(),
        }
    }
}

fn run_step(plan: &Plan, step: &Step, log: &Path) -> Result<(), String> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("{}: {e}", log.display()))?;
    let (mut cmd, shown) = match step {
        Step::Wsp(a) => {
            let mut c = std::process::Command::new(&plan.exe);
            c.args(a);
            (c, format!("rlx-wsp {}", a.join(" ")))
        }
        Step::Shell(line) => {
            let mut c = std::process::Command::new("sh");
            c.args(["-c", line]);
            (c, line.clone())
        }
    };
    {
        use std::io::Write;
        let mut f = &file;
        writeln!(f, "$ {shown}").map_err(|e| e.to_string())?;
    }
    let status = cmd
        .stdin(std::process::Stdio::null())
        .stdout(file.try_clone().map_err(|e| e.to_string())?)
        .stderr(file)
        .status()
        .map_err(|e| format!("starting `{shown}`: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        let tail = std::fs::read_to_string(log).unwrap_or_default();
        let last = tail
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        Err(format!("{status}: {last}"))
    }
}

/// Run the plan; returns the stages' outcomes (also in `pipeline.json`).
pub fn run(plan: &Plan) -> Result<Value, String> {
    let logs = plan.out.join("logs");
    for d in [logs.clone(), plan.out.join("base")] {
        std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let path = plan.out.join("pipeline.json");
    let mut record: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({}));
    let save = |r: &Value| {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, format!("{r:#}\n"))
            .and_then(|_| std::fs::rename(&tmp, &path))
            .map_err(|e| e.to_string())
    };
    for stage in STAGES {
        let status = |r: &Value, st: &str| r[st]["status"].as_str().unwrap_or("").to_string();
        if !plan.only.is_empty() && !plan.only.iter().any(|x| x == stage) {
            continue;
        }
        let redo = plan.redo.iter().any(|x| x == stage);
        if status(&record, stage) == "done" && !redo && stage != "summary" {
            eprintln!("pipeline: {stage} already done");
            continue;
        }
        if let Some(why) = plan.not_applicable(stage) {
            // A given trainable package is as good as an exported one.
            let st = if stage == "export" && plan.train_package.is_some() {
                "done"
            } else {
                "skipped"
            };
            record[stage] = json!({ "status": st, "note": why });
            save(&record)?;
            eprintln!("pipeline: {stage} {st}: {why}");
            continue;
        }
        let missing: Vec<&str> = Plan::needs(stage)
            .iter()
            .copied()
            .filter(|n| status(&record, n) != "done")
            .collect();
        if !missing.is_empty() {
            let why = format!("needs {}", missing.join(", "));
            record[stage] = json!({ "status": "skipped", "note": why });
            save(&record)?;
            eprintln!("pipeline: {stage} skipped: {why}");
            continue;
        }
        eprintln!("pipeline: {stage} …");
        let t0 = std::time::Instant::now();
        record[stage] = json!({ "status": "running" });
        save(&record)?;
        let log = logs.join(format!("{stage}.log"));
        let result = if stage == "summary" {
            summarize(plan, &record).map(|_| ())
        } else {
            plan.steps(stage)
                .iter()
                .try_for_each(|st| run_step(plan, st, &log))
        };
        let seconds = t0.elapsed().as_secs_f64();
        record[stage] = match &result {
            Ok(()) => json!({ "status": "done", "seconds": seconds }),
            Err(e) => json!({ "status": "failed", "seconds": seconds, "error": e,
                              "log": log.display().to_string() }),
        };
        save(&record)?;
        match result {
            Ok(()) => eprintln!("pipeline: {stage} done ({seconds:.0}s)"),
            Err(e) => eprintln!("pipeline: {stage} FAILED: {e}"),
        }
    }
    Ok(record)
}

// ── Summary ─────────────────────────────────────────────────────────────

/// One method's scores.
#[derive(Debug, Clone)]
struct Arm {
    name: String,
    what: String,
    accuracy: f64,
    perturbed: Option<f64>,
    long: Option<f64>,
}

fn arm(name: &str, what: &str, r: &Value, edited: bool) -> Option<Arm> {
    let held = if r["held_out"].is_object() {
        &r["held_out"]
    } else {
        &r["held_out_final"]
    };
    let side = if edited { "edited" } else { "unedited" };
    Some(Arm {
        name: name.to_string(),
        what: match r["held_out_kept"].as_str() {
            Some(k) if edited => format!("{what} (kept: {k})"),
            _ => what.to_string(),
        },
        accuracy: held["accuracy"].as_f64()?,
        perturbed: r["checks"]["perturbed"]["perturbed"][side].as_f64(),
        long: r["checks"]["truncation"][side].as_f64(),
    })
}

/// 95% Wilson interval of `p` out of `n`.
fn wilson(p: f64, n: f64) -> (f64, f64) {
    let z = 1.96_f64;
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / d;
    ((c - h).max(0.0), (c + h).min(1.0))
}

fn esc(t: &str) -> String {
    t.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const STYLE: &str = r#"
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: #52514e; font-size: 13px; }
  .title { fill: #0b0b0b; font-size: 18px; font-weight: 600; }
  .name { fill: #0b0b0b; font-size: 14px; } .val { fill: #0b0b0b; font-size: 14px; font-weight: 600; }
  .tick { fill: #898781; font-size: 12px; }
  .grid { stroke: #e1e0d9; stroke-width: 1; } .ref { stroke: #52514e; stroke-width: 1; }
  .dot { fill: #2a78d6; stroke: #2a78d6; stroke-width: 2; } .ci { stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; }
  .ring { fill: none; stroke: #fcfcfb; stroke-width: 2; } .bg { fill: #fcfcfb; }
  @media (prefers-color-scheme: dark) {
    text { fill: #c3c2b7; } .title, .name, .val { fill: #f0efec; }
    .grid { stroke: #2c2c2a; } .ref { stroke: #c3c2b7; }
    .dot { fill: #3987e5; stroke: #3987e5; } .ci { stroke: #3987e5; }
    .ring { stroke: #1a1a19; } .bg { fill: #1a1a19; }
  }
"#;

/// Held-out accuracy per method, each with its 95% interval.
fn summary_svg(arms: &[Arm], n: usize) -> String {
    let (w, l, r, row) = (960.0_f64, 300.0_f64, 860.0_f64, 40.0_f64);
    let top = 78.0;
    let bottom = top + arms.len() as f64 * row;
    let nf = n.max(1) as f64;
    let max = arms
        .iter()
        .map(|a| wilson(a.accuracy, nf).1)
        .fold(0.1_f64, f64::max);
    let max = ((max * 10.0).ceil() / 10.0).min(1.0);
    let x = |v: f64| l + v / max * (r - l);
    let mut b = format!(
        r#"<text class="title" x="24" y="34">Held-out accuracy by method</text><text x="24" y="56">{n} questions; line = 95% interval; the vertical line is the unedited model</text>"#
    );
    let mut t = 0.0;
    while t <= max + 1e-9 {
        b += &format!(
            r#"<line class="grid" x1="{0:.1}" x2="{0:.1}" y1="{top:.1}" y2="{bottom:.1}"/><text class="tick" x="{0:.1}" y="{1:.1}" text-anchor="middle">{2:.0}%</text>"#,
            x(t),
            bottom + 18.0,
            t * 100.0
        );
        t += 0.1;
    }
    if let Some(u) = arms.first() {
        b += &format!(
            r#"<line class="ref" x1="{0:.1}" x2="{0:.1}" y1="{top:.1}" y2="{bottom:.1}"/>"#,
            x(u.accuracy)
        );
    }
    for (i, a) in arms.iter().enumerate() {
        let cy = top + (i as f64 + 0.5) * row;
        let (lo, hi) = wilson(a.accuracy, nf);
        let cx = x(a.accuracy);
        b += &format!(
            r#"<g><title>{0}: {1:.1}% (95% interval {2:.1}–{3:.1}%)</title><text class="name" x="{4:.1}" y="{5:.1}" text-anchor="end">{0}</text><line class="ci" x1="{6:.1}" x2="{7:.1}" y1="{cy:.1}" y2="{cy:.1}"/><circle class="ring" cx="{cx:.1}" cy="{cy:.1}" r="7"/><circle class="dot" cx="{cx:.1}" cy="{cy:.1}" r="5"/><text class="val" x="{8:.1}" y="{5:.1}">{1:.0}%</text></g>"#,
            esc(&a.name),
            a.accuracy * 100.0,
            lo * 100.0,
            hi * 100.0,
            l - 16.0,
            cy + 5.0,
            x(lo),
            x(hi),
            x(hi) + 10.0
        );
    }
    let h = bottom + 36.0;
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{h:.0}" viewBox="0 0 {w:.0} {h:.0}" role="img"><title>Held-out accuracy by method</title><style>{STYLE}</style><rect class="bg" width="100%" height="100%"/>{b}</svg>"#
    )
}

/// `summary.md`, `summary.json` and `summary.svg` from the reports there
/// are.
fn summarize(plan: &Plan, record: &Value) -> Result<(), String> {
    let read = |name: &str| -> Value {
        std::fs::read_to_string(plan.report_of(name))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null)
    };
    let mut arms = Vec::new();
    arms.extend(arm(
        "unedited",
        "the model as packaged",
        &read("base"),
        false,
    ));
    for (dir, name, what) in [
        ("agent", "agent", "an agent editing the weights"),
        (
            "lora-reference",
            "LoRA, reference solutions",
            "trained on the reference solutions",
        ),
        (
            "lora-distill",
            "LoRA, own answers",
            "trained on its correct answers given with worked examples",
        ),
    ] {
        arms.extend(arm(name, what, &read(dir), true));
    }
    let pct = |v: Option<f64>| v.map_or("–".to_string(), |x| format!("{:.1}%", x * 100.0));
    let mut md = format!(
        "# rlx-wsp pipeline\n\n{} held-out questions.\n\n\
         | method | held-out | perturbed | twice the tokens | what |\n|---|---:|---:|---:|---|\n",
        plan.heldout_n
    );
    for a in &arms {
        md += &format!(
            "| {} | {:.1}% | {} | {} | {} |\n",
            a.name,
            a.accuracy * 100.0,
            pct(a.perturbed),
            pct(a.long),
            a.what
        );
    }
    md += "\n| stage | outcome | minutes | note |\n|---|---|---:|---|\n";
    for st in STAGES.iter().filter(|s| **s != "summary") {
        let r = &record[*st];
        md += &format!(
            "| {st} | {} | {} | {} |\n",
            r["status"].as_str().unwrap_or("not run"),
            r["seconds"]
                .as_f64()
                .map_or("".into(), |x| format!("{:.1}", x / 60.0)),
            r["note"].as_str().or(r["error"].as_str()).unwrap_or("")
        );
    }
    let write =
        |f: &str, t: String| std::fs::write(plan.out.join(f), t).map_err(|e| format!("{f}: {e}"));
    write("summary.md", md)?;
    write("summary.svg", summary_svg(&arms, plan.heldout_n))?;
    let rows: Vec<Value> = arms
        .iter()
        .map(|a| {
            json!({ "method": a.name, "what": a.what, "held_out": a.accuracy,
                         "perturbed": a.perturbed, "twice_the_tokens": a.long })
        })
        .collect();
    write(
        "summary.json",
        format!(
            "{:#}\n",
            json!({ "heldout_n": plan.heldout_n, "methods": rows })
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in `rlx-wsp`: logs its command and writes what each command
    /// would leave; `new` fails while `fail` exists beside it, `lora-train`
    /// on `--train-items` while `fail-lora` does.
    fn fake(dir: &Path) -> PathBuf {
        let p = dir.join("fake-wsp");
        let script = format!(
            r#"#!/bin/sh
d="{d}"
echo "$1" >> "$d/calls"
out=""; state=""; prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && out="$a"
  [ "$prev" = "--state" ] && state="$a"
  [ "$a" = "--train-items" ] && [ -e "$d/fail-lora" ] && {{ echo boom; exit 3; }}
  prev="$a"
done
case "$1" in
  new) mkdir -p "$(dirname "$state")"; echo '{{}}' > "$state";;
  score) echo '{{"held_out_final":{{"accuracy":0.3}},"checks":{{"perturbed":{{"perturbed":{{"edited":0.2,"unedited":0.2}}}},"truncation":{{"edited":0.31,"unedited":0.31}}}}}}' > "$out";;
  agent) mkdir -p "$out"; echo '{{"held_out":{{"accuracy":0.4}},"held_out_kept":"verified"}}' > "$out/report.json";;
  qa-distill) echo '{{}}' > "$out";;
  lora-train) mkdir -p "$out"; echo '{{}}' > "$out/state.json";;
esac
"#,
            d = dir.display()
        );
        std::fs::write(&p, script).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn calls(dir: &Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(String::from)
            .collect()
    }

    #[test]
    fn stages_run_in_order_resume_and_skip_what_they_need() {
        let dir = std::env::temp_dir().join(format!("rlx_wsp_pipe_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut plan = Plan {
            out: dir.join("run"),
            exe: fake(&dir),
            heldout_n: 100,
            train_export: Some(format!(
                "touch {{out}} && echo x >> {}/exported",
                dir.display()
            )),
            ..Default::default()
        };
        std::fs::write(dir.join("fail-lora"), "").unwrap();
        let r = run(&plan).unwrap();
        assert_eq!(
            calls(&dir),
            [
                "new",
                "score",
                "agent",
                "qa-distill",
                "lora-train",
                "score",
                "lora-train"
            ]
        );
        assert_eq!(r["lora-reference"]["status"], "done");
        assert_eq!(r["lora-distill"]["status"], "failed");
        assert!(
            r["lora-distill"]["error"]
                .as_str()
                .unwrap()
                .contains("boom")
        );
        assert_eq!(
            r["summary"]["status"], "done",
            "a failed stage does not stop it"
        );
        let md = std::fs::read_to_string(plan.out.join("summary.md")).unwrap();
        assert!(md.contains("| agent | 40.0% |"), "{md}");
        assert!(md.contains("| unedited | 30.0% | 20.0% | 31.0% |"), "{md}");
        assert!(md.contains("failed"), "{md}");
        assert!(
            std::fs::read_to_string(plan.out.join("summary.svg"))
                .unwrap()
                .ends_with("</svg>")
        );

        // Again: only what failed runs.
        std::fs::remove_file(dir.join("fail-lora")).unwrap();
        std::fs::remove_file(dir.join("calls")).unwrap();
        let r = run(&plan).unwrap();
        assert_eq!(calls(&dir), ["lora-train", "score"]);
        assert_eq!(r["lora-distill"]["status"], "done");
        assert_eq!(
            std::fs::read_to_string(dir.join("exported")).unwrap(),
            "x\n"
        );

        // Asked for: one stage again.
        std::fs::remove_file(dir.join("calls")).unwrap();
        plan.redo = vec!["baseline".into()];
        run(&plan).unwrap();
        assert_eq!(calls(&dir), ["score"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_a_trainable_package_lora_stages_say_why() {
        let dir = std::env::temp_dir().join(format!("rlx_wsp_pipe2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let plan = Plan {
            out: dir.join("run"),
            exe: fake(&dir),
            heldout_n: 10,
            ..Default::default()
        };
        let r = run(&plan).unwrap();
        for st in ["export", "lora-reference", "lora-distill"] {
            assert_eq!(r[st]["status"], "skipped", "{st}");
        }
        assert_eq!(calls(&dir), ["new", "score", "agent", "qa-distill"]);
        // A stage whose input failed is skipped, not run on nothing.
        let empty = dir.join("run2");
        let plan = Plan {
            out: empty,
            exe: dir.join("missing"),
            heldout_n: 10,
            ..Default::default()
        };
        let r = run(&plan).unwrap();
        assert_eq!(r["base"]["status"], "failed");
        assert_eq!(r["agent"]["status"], "skipped");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_agent_that_ended_is_finished_not_run_again() {
        let dir = std::env::temp_dir().join(format!("rlx_wsp_pipe3_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("agent")).unwrap();
        std::fs::write(
            dir.join("agent/transcript.jsonl"),
            "{\"type\":\"result\",\"num_turns\":1}\n",
        )
        .unwrap();
        let plan = Plan {
            out: dir.clone(),
            agent: s(&["--cli", "claude", "--budget", "40"]),
            ..Default::default()
        };
        let steps = plan.agent_steps();
        let Step::Wsp(a) = &steps[0] else { panic!() };
        assert_eq!(a[0], "finish");
        assert!(a.ends_with(&s(&["--cli", "claude"])));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn intervals_are_wilson() {
        let (lo, hi) = wilson(0.26, 100.0);
        assert!(
            (lo - 0.1834).abs() < 1e-3 && (hi - 0.3547).abs() < 1e-3,
            "{lo} {hi}"
        );
    }
}
