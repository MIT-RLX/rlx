// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.check(graph, options?)` — device-free static diagnostics.
//!
//! The same analysis `cargo rlx check` runs: shape and representation
//! legality, per-backend op claim, missed fusions, numeric hazards, and
//! memory-plan safety. No driver is needed, so this answers "will CUDA take
//! this graph" on a laptop.
//!
//! ```js
//! const report = rlx.check(g, { backends: ["cpu", "cuda", "metal"] });
//! if (report.errors > 0) {
//!   for (const d of report.diagnostics.filter((d) => d.severity === "error")) {
//!     console.error(`${d.code}: ${d.message}${d.hint ? "  hint: " + d.hint : ""}`);
//!   }
//! }
//! ```
//!
//! Reading the `warning` list matters as much as the errors: a
//! `missed-fusion` or `numeric` note is the difference between a graph that
//! runs and one that runs fast, or one whose numbers are plausible but wrong.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_runtime::check::{CheckOptions, Severity, check_graph};

use crate::api::graph::graph_ref;
use crate::convert::*;

fn severity_label(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    }
}

fn parse_options(ctx: &mut Context, v: &Value) -> JsResult<CheckOptions> {
    let mut opts = CheckOptions::default();
    let backends_v = field(ctx, v, "backends")?;
    if !is_nullish(&backends_v) {
        let names = to_string_vec(ctx, &backends_v)?;
        let mut targets = Vec::with_capacity(names.len());
        for name in &names {
            match rlx_runtime::check::parse_backend(name) {
                Some(t) => targets.push(t),
                None => {
                    let known: Vec<&'static str> = rlx_runtime::check::all_backends()
                        .into_iter()
                        .map(rlx_runtime::check::backend_name)
                        .collect();
                    return ctx.throw_type(&format!(
                        "check: unknown backend '{name}' (have [{}])",
                        known.join(", ")
                    ));
                }
            }
        }
        opts.backends = targets;
    }
    for (key, slot) in [
        ("dispatch", &mut opts.dispatch as *mut bool),
        ("fusion", &mut opts.fusion as *mut bool),
        ("numeric", &mut opts.numeric as *mut bool),
        ("repr", &mut opts.repr as *mut bool),
        ("schedule", &mut opts.schedule as *mut bool),
        ("plan", &mut opts.plan as *mut bool),
    ] {
        let value = field(ctx, v, key)?;
        if !is_nullish(&value) {
            // SAFETY: each pointer is a distinct field of the local `opts`,
            // alive for the loop, and written once.
            unsafe { *slot = to_bool(&value) };
        }
    }
    Ok(opts)
}

fn f_check(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let options = arg(args, 1).clone();
    let opts = parse_options(ctx, &options)?;
    let report = {
        let graph = graph_ref(ctx, arg(args, 0))?;
        check_graph(graph, &opts)
    };

    let errors = report.errors();
    let warnings = report.warnings();
    let nodes = report.nodes;
    let name = report.graph.clone();

    let diagnostics: Vec<Value> = report
        .diagnostics
        .iter()
        .map(|d| {
            let severity = new_string(ctx, severity_label(d.severity));
            let code = new_string(ctx, &d.code);
            let message = new_string(ctx, &d.message);
            let node = match d.node {
                Some(n) => Value::number(n as f64),
                None => Value::Null,
            };
            let context = match &d.context {
                Some(text) => new_string(ctx, text),
                None => Value::Null,
            };
            let hint = match &d.hint {
                Some(text) => new_string(ctx, text),
                None => Value::Null,
            };
            let backend = match &d.backend {
                Some(text) => new_string(ctx, text),
                None => Value::Null,
            };
            new_object(
                ctx,
                vec![
                    ("severity", severity),
                    ("code", code),
                    ("message", message),
                    ("node", node),
                    ("context", context),
                    ("hint", hint),
                    ("backend", backend),
                ],
            )
        })
        .collect();

    let backends: Vec<Value> = report
        .backends
        .iter()
        .map(|b| {
            let backend = new_string(ctx, &b.backend);
            let legality = match &b.legality {
                Some(l) => new_object(
                    ctx,
                    vec![
                        ("compileReady", Value::Bool(l.compile_ready)),
                        ("nativeKinds", Value::number(l.native_kinds as f64)),
                        ("commonIrKinds", Value::number(l.common_ir_kinds as f64)),
                        ("rewrittenKinds", Value::number(l.rewritten_kinds as f64)),
                    ],
                ),
                // `null`, not an empty object: the backend is not in this
                // build, which is a different answer from "claims nothing".
                None => Value::Null,
            };
            new_object(
                ctx,
                vec![
                    ("backend", backend),
                    ("legality", legality),
                    ("fusedOps", Value::number(b.fused_ops as f64)),
                    ("missedFusions", Value::number(b.missed_fusions as f64)),
                ],
            )
        })
        .collect();

    let name_v = new_string(ctx, &name);
    let diagnostics_v = new_array(ctx, diagnostics);
    let backends_v = new_array(ctx, backends);
    Ok(new_object(
        ctx,
        vec![
            ("graph", name_v),
            ("nodes", Value::number(nodes as f64)),
            ("errors", Value::number(errors as f64)),
            ("warnings", Value::number(warnings as f64)),
            ("diagnostics", diagnostics_v),
            ("backends", backends_v),
        ],
    ))
}

/// Every backend name `check` accepts, whether or not it is in this build.
fn f_check_backends(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names: Vec<&'static str> = rlx_runtime::check::all_backends()
        .into_iter()
        .map(rlx_runtime::check::backend_name)
        .collect();
    Ok(new_string_array(ctx, &names))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "check" => f_check, 2;
        "checkBackends" => f_check_backends, 0;
    };
}
