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
use rlx_bind::runtime::devices;

use crate::api::graph::graph_ref;
use crate::convert::*;

fn f_devices(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(new_string_array(ctx, &devices::available()))
}

fn f_is_available(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    Ok(Value::Bool(devices::is_available(&name)))
}

/// `parseDevice("gpu")` → the canonical label; throws on an unknown name.
fn f_parse_device(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let d = crate::api::bind::lift(ctx, devices::parse(&name))?;
    Ok(new_string(ctx, devices::label(d)))
}

fn f_backends_manifest(ctx: &mut Context, _this: &Value, _a: &[Value], _m: i32) -> JsResult<Value> {
    Ok(new_string(ctx, &devices::manifest_json()))
}

fn f_fastest_device_for(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let policy = if is_nullish(arg(args, 1)) {
        None
    } else {
        Some(crate::api::routing::parse_policy(ctx, arg(args, 1))?)
    };
    let graph = graph_ref(ctx, arg(args, 0))?;
    Ok(new_string(
        ctx,
        devices::fastest_for(graph, policy.as_ref()),
    ))
}

fn f_device_report(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = crate::api::routing::parse_policy(ctx, arg(args, 1))?;
    let graph = graph_ref(ctx, arg(args, 0))?;
    let json = devices::report(graph, &policy);
    Ok(crate::api::bind::report(ctx, &json))
}

/// `exportFpga(graph, outDir, {quant, hw, inIface, outIface, bindIn, bindOut,
/// sideband, bindSideband})` → the SystemVerilog files written.
#[cfg(feature = "fpga")]
fn f_export_fpga(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "exportFpga")? {
        return Ok(refused);
    }
    let dir = ctx.to_rust_string(arg(args, 1))?;
    let opts = crate::api::bind::to_json(ctx, arg(args, 2))?;
    let graph = graph_ref(ctx, arg(args, 0))?;
    let files = crate::api::bind::lift(ctx, rlx_bind::runtime::fpga::export(graph, &dir, &opts))?;
    Ok(new_string_array(ctx, &files))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    #[cfg(feature = "fpga")]
    js_functions! { ctx, namespace; "exportFpga" => f_export_fpga, 3; };
    js_functions! {
        ctx, namespace;
        "devices" => f_devices, 0;
        "isAvailable" => f_is_available, 1;
        "parseDevice" => f_parse_device, 1;
        "backendsManifest" => f_backends_manifest, 0;
        "fastestDeviceFor" => f_fastest_device_for, 2;
        "deviceReport" => f_device_report, 2;
    };
    let version = new_string(ctx, env!("CARGO_PKG_VERSION"));
    ctx.define_value(namespace, "version", version, PropFlags::C_W);
}
