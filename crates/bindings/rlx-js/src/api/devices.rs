// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend discovery: what this build can actually run on, and where a given
//! graph should go.
//!
//! `devices()` answers for the *build*; `deviceReport(graph)` answers for the
//! *graph*, which is the one that matters — a backend can be linked in and
//! still refuse an op.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, PropFlags};
use quickrs_core::value::{JsResult, Value};

use crate::api::graph::graph_ref;
use crate::convert::*;

fn f_devices(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names: Vec<&'static str> = rlx_runtime::available_devices()
        .into_iter()
        .map(rlx_runtime::device_label)
        .collect();
    Ok(new_string_array(ctx, &names))
}

fn f_is_available(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    match rlx_runtime::parse_device(&name) {
        Ok(d) => Ok(Value::Bool(rlx_runtime::is_available(d))),
        // An unknown name is not an error here: the question was "can I use
        // it", and the answer is no.
        Err(_) => Ok(Value::Bool(false)),
    }
}

fn f_backends_manifest(ctx: &mut Context, _this: &Value, _a: &[Value], _m: i32) -> JsResult<Value> {
    let json = rlx_runtime::BackendsManifest::json().to_string();
    Ok(new_string(ctx, &json))
}

fn f_fastest_device_for(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let graph = graph_ref(ctx, arg(args, 0))?;
    let device = rlx_runtime::fastest_device_for(graph);
    Ok(new_string(ctx, rlx_runtime::device_label(device)))
}

/// Per-backend viability for one graph: `[{device, available, supportsGraph,
/// recommended, blocker, capabilities}]`.
fn f_device_report(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let graph = graph_ref(ctx, arg(args, 0))?;
    let report = rlx_runtime::device_report(graph, &rlx_runtime::DevicePolicy::all());
    let mut rows = Vec::with_capacity(report.len());
    for candidate in report {
        let device = new_string(ctx, candidate.label);
        let blocker = match &candidate.blocker {
            Some(text) => new_string(ctx, text),
            None => Value::Null,
        };
        let capabilities = new_string_array(ctx, &candidate.capabilities);
        rows.push(new_object(
            ctx,
            vec![
                ("device", device),
                ("available", Value::Bool(candidate.available)),
                ("registered", Value::Bool(candidate.registered)),
                ("supportsGraph", Value::Bool(candidate.supports_graph)),
                ("recommended", Value::Bool(candidate.recommended)),
                ("blocker", blocker),
                ("capabilities", capabilities),
            ],
        ));
    }
    Ok(new_array(ctx, rows))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "devices" => f_devices, 0;
        "isAvailable" => f_is_available, 1;
        "backendsManifest" => f_backends_manifest, 0;
        "fastestDeviceFor" => f_fastest_device_for, 1;
        "deviceReport" => f_device_report, 1;
    };
    let version = new_string(ctx, env!("CARGO_PKG_VERSION"));
    ctx.define_value(namespace, "version", version, PropFlags::C_W);
}
