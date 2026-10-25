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

use crate::api::bind::{lift, report, to_json};
use crate::api::graph::graph_ref;
use crate::convert::*;

/// `check(graph, {backends, dispatch, fusion, numeric, repr, schedule, plan,
/// budget})` → `{graph, nodes, errors, warnings, diagnostics, backends}`.
fn f_check(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let opts = to_json(ctx, arg(args, 1))?;
    let r = {
        let graph = graph_ref(ctx, arg(args, 0))?;
        rlx_bind::runtime::check::check(graph, None, &opts)
    };
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

fn f_check_backends(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(new_string_array(ctx, &rlx_bind::runtime::check::backends()))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "check" => f_check, 2;
        "checkBackends" => f_check_backends, 0;
    };
}
