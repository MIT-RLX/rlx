// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check` — the device-free static diagnostics of `cargo rlx check`.
//!
//! Op legality per backend (including backends not compiled into this build),
//! missed fusions, numeric hazards, memory-plan safety, cost budget. The graph
//! is borrowed, not consumed.

use rlx_ir::Graph;
use rlx_runtime::check::{CheckOptions, Severity, check_graph};
use serde_json::{Value, json};

use super::opts::Opts;
use crate::{ApiError, ApiResult};

/// Every backend `check` knows, compiled in or not.
pub fn backends() -> Vec<&'static str> {
    rlx_runtime::check::all_backends()
        .into_iter()
        .map(rlx_runtime::check::backend_name)
        .collect()
}

fn severity_label(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    }
}

/// Run the checks. `backends` names the targets (default: all); `opts` turns
/// passes off: `{dispatch, fusion, numeric, repr, schedule, plan, budget}`.
pub fn check(graph: &Graph, backends: Option<&[String]>, opts: &Value) -> ApiResult<Value> {
    let o = Opts::new("check", opts)?;
    let base = CheckOptions::default();
    let mut options = CheckOptions {
        dispatch: o.bool("dispatch", base.dispatch)?,
        fusion: o.bool("fusion", base.fusion)?,
        numeric: o.bool("numeric", base.numeric)?,
        repr: o.bool("repr", base.repr)?,
        schedule: o.bool("schedule", base.schedule)?,
        plan: o.bool("plan", base.plan)?,
        budget: o.bool("budget", base.budget)?,
        ..base
    };
    // `backends` may also arrive inside the options object (the JS spelling).
    let from_opts: Option<Vec<String>> = match o.get("backends") {
        None => None,
        Some(Value::Array(items)) => Some(
            items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| ApiError::type_("check: backends must be a list of names"))
                })
                .collect::<ApiResult<_>>()?,
        ),
        Some(_) => return Err(ApiError::type_("check: backends must be a list of names")),
    };
    o.finish()?;
    if let Some(names) = backends.map(<[String]>::to_vec).or(from_opts) {
        let mut targets = Vec::with_capacity(names.len());
        for name in &names {
            match rlx_runtime::check::parse_backend(name) {
                Some(t) => targets.push(t),
                None => {
                    return Err(ApiError::type_(format!(
                        "check: unknown backend '{name}' (have [{}])",
                        self::backends().join(", ")
                    )));
                }
            }
        }
        options.backends = targets;
    }

    let report = check_graph(graph, &options);
    let diagnostics: Vec<Value> = report
        .diagnostics
        .iter()
        .map(|d| {
            json!({
                "severity": severity_label(d.severity),
                "code": d.code,
                "message": d.message,
                "node": d.node,
                "context": d.context,
                "hint": d.hint,
                "backend": d.backend,
            })
        })
        .collect();
    let summaries: Vec<Value> = report
        .backends
        .iter()
        .map(|b| {
            json!({
                "backend": b.backend,
                // `null`, not an empty object: the backend is not in this
                // build, which is a different answer from "claims nothing".
                "legality": b.legality.as_ref().map(|l| json!({
                    "compile_ready": l.compile_ready,
                    "native_kinds": l.native_kinds,
                    "common_ir_kinds": l.common_ir_kinds,
                    "rewritten_kinds": l.rewritten_kinds,
                })),
                "fused_ops": b.fused_ops,
                "missed_fusions": b.missed_fusions,
            })
        })
        .collect();
    Ok(json!({
        "graph": report.graph,
        "nodes": report.nodes,
        "errors": report.errors(),
        "warnings": report.warnings(),
        "diagnostics": diagnostics,
        "backends": summaries,
    }))
}
