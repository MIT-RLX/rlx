// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-wsp bundle`: package a finished run for publication, as the post
//! released its logs — the prompt, the agent's full transcript, the report,
//! the final weights (state + baked `.rlxmodel` + int8 check) and a README
//! with the headline numbers and how to re-score. The run directory's
//! absolute path is replaced by `<run>` and the home directory by `~` in the
//! copied text.

use crate::space::{State, WeightSpace};
use serde_json::Value;
use std::path::Path;

pub fn bundle(run: &Path, out: &Path, episodes: usize) -> Result<Value, String> {
    let read =
        |name: &str| std::fs::read_to_string(run.join(name)).map_err(|e| format!("{name}: {e}"));
    let run_abs = std::path::absolute(run).map_err(|e| e.to_string())?;
    let home = crate::mnist::home();
    let scrub = |s: &str| {
        let s = s.replace(&run_abs.display().to_string(), "<run>");
        if home.as_os_str().is_empty() {
            s
        } else {
            s.replace(&home.display().to_string(), "~")
        }
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut files = Vec::new();
    for (src, dst) in [
        ("prompt.txt", "prompt.txt"),
        ("transcript.jsonl", "transcript.jsonl"),
        ("report.json", "report.json"),
        ("state.json", "final_state.json"),
    ] {
        if let Ok(text) = read(src) {
            std::fs::write(out.join(dst), scrub(&text)).map_err(|e| e.to_string())?;
            files.push(dst.to_string());
        }
    }
    let state: State = serde_json::from_str(&read("state.json")?).map_err(|e| e.to_string())?;
    let mut ws = WeightSpace::from_state(state);
    let export = crate::export::export(&mut ws, &out.join("model"), false, episodes)?;
    files.push("model/".into());
    let report: Value = read("report.json")
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);
    let a = &report["agent"];
    let held = &report["held_out"];
    let readme = format!(
        "# rlx-wsp run: {env}\n\n\
         | | |\n|---|---|\n\
         | task | `{env}` |\n\
         | agent | `{cli}` — {models} |\n\
         | held-out score | {score} ({detail}) |\n\
         | forward passes | {used} / {budget} |\n\
         | edits | {edits} |\n\
         | agent turns / time / cost | {turns} / {mins} min / {cost} |\n\
         | weights | {nz} nonzero of {total} |\n\n\
         ## Files\n\n\
         - `prompt.txt` — the task prompt (rules and layout came from the `describe` tool)\n\
         - `transcript.jsonl` — the agent CLI's full event stream: tool calls, results, text\n\
         - `report.json` — harness report, incl. the check of the agent's account\n\
         - `final_state.json` — the weights, checkpoints, budget and edit log at the end\n\
         - `model/` — the weights baked into a standalone `model.rlxmodel`, plus `export.json`\n\n\
         ## Re-score\n\n\
         ```sh\nrlx-wsp score --state final_state.json\nrlx-wsp verify --state final_state.json\n```\n\n\
         ## The agent's own summary\n\n{text}\n\n\
         _Built with rlx-wsp {version}._\n",
        env = report["env"].as_str().unwrap_or("?"),
        cli = report["cli"].as_str().unwrap_or("?"),
        models = a["models"],
        score = held["score"],
        detail = held["detail"].as_str().unwrap_or(""),
        used = report["forward_passes"]["used"],
        budget = report["forward_passes"]["budget"],
        edits = report["edits"],
        turns = a["num_turns"],
        mins = a["duration_ms"]
            .as_f64()
            .map_or("?".into(), |m| format!("{:.1}", m / 60_000.0)),
        cost = a["cost_usd"]
            .as_f64()
            .map_or("?".into(), |c| format!("${c:.2}")),
        nz = export["weights"]["nonzero"],
        total = export["weights"]["total"],
        text = scrub(a["final_text"].as_str().unwrap_or("(none)")),
        version = env!("CARGO_PKG_VERSION"),
    );
    std::fs::write(out.join("README.md"), readme).map_err(|e| e.to_string())?;
    files.push("README.md".into());
    Ok(serde_json::json!({ "out": out, "files": files }))
}
