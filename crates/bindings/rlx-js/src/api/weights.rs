// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint formats other than GGUF: MLX, DDUF, NeMo, PyTorch `.pt`, and
//! the RLX package format `.rlxp`.
//!
//! ```js
//! const w = rlx.loadPt("pytorch_model.bin");     // {name: Float32Array}
//! compiled.setParams(w);
//!
//! rlx.toRlxp("model.gguf", "model.rlxp", { from: "gguf", autoTier: true });
//! console.log(rlx.openRlxp("model.rlxp").tensors.length);
//! ```
//!
//! These loaders materialize the whole checkpoint as f32 — that is what the
//! underlying readers do, and a `.pt` has no packed form to stream. For a
//! large quantized model, prefer `rlx.openGguf`, which is mmap'd and
//! dequantizes one tensor at a time.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, PropFlags};
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::io::weights;

use crate::api::bind::{lift, report, to_json};
use crate::convert::*;

fn tensor_map(ctx: &mut Context, entries: Vec<(String, Vec<f32>)>) -> Value {
    let obj = ctx.new_object();
    for (name, data) in entries {
        let array = new_f32_array(ctx, &data);
        ctx.define_value(&obj, &name, array, PropFlags::C_W_E);
    }
    Value::Object(obj)
}

/// One loader per format, `{name: Float32Array}` sorted by name.
fn f_load(ctx: &mut Context, _this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "weight loader")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let format = ["mlx", "dduf", "nemo", "pt"][magic as usize];
    let entries = lift(ctx, weights::load(format, &path))?;
    Ok(tensor_map(ctx, entries))
}

/// `loadWeights(format, path)` — `"mlx"`, `"safetensors"`, `"dduf"`, `"nemo"`
/// or `"pt"`; the per-format loaders below are this with the format fixed.
fn f_load_weights(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "weight loader")? {
        return Ok(refused);
    }
    let format = ctx.to_rust_string(arg(args, 0))?;
    let path = ctx.to_rust_string(arg(args, 1))?;
    let entries = lift(ctx, weights::load(&format, &path))?;
    Ok(tensor_map(ctx, entries))
}

fn f_load_mlx(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    f_load(ctx, this, args, 0)
}
fn f_load_dduf(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    f_load(ctx, this, args, 1)
}
fn f_load_nemo(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    f_load(ctx, this, args, 2)
}
fn f_load_pt(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    f_load(ctx, this, args, 3)
}

// ── .rlxp ───────────────────────────────────────────────────

/// `openRlxp(path)` → `{name, formatVersion, compatVersion, features,
/// hasGraph, tensors, sidecars}`.
fn f_open_rlxp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "openRlxp")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let json = lift(ctx, weights::open_rlxp(&path))?;
    Ok(report(ctx, &json))
}

/// `toRlxp(src, out, {from, container, includeGraph, autoTier, keepPacked})` → `out`.
fn f_to_rlxp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "toRlxp")? {
        return Ok(refused);
    }
    let src = ctx.to_rust_string(arg(args, 0))?;
    let out = ctx.to_rust_string(arg(args, 1))?;
    let options = arg(args, 2).clone();
    let from_v = field(ctx, &options, "from")?;
    if is_nullish(&from_v) {
        return ctx.throw_type("toRlxp: `from` is required (gguf, mlx, dduf, nemo, pt)");
    }
    let from = ctx.to_rust_string(&from_v)?;
    let opts = to_json(ctx, &options)?;
    lift(ctx, weights::to_rlxp(&from, &src, &out, &opts))?;
    Ok(new_string(ctx, &out))
}

/// `verifyRlxp(path)` → `{ok, tensorsChecked, tensorsOk, …, failures}`.
fn f_verify_rlxp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "verifyRlxp")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let json = lift(ctx, weights::verify_rlxp(&path))?;
    Ok(report(ctx, &json))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "loadWeights" => f_load_weights, 2;
        "loadMlx" => f_load_mlx, 1;
        "loadDduf" => f_load_dduf, 1;
        "loadNemo" => f_load_nemo, 1;
        "loadPt" => f_load_pt, 1;
        "openRlxp" => f_open_rlxp, 1;
        "toRlxp" => f_to_rlxp, 3;
        "verifyRlxp" => f_verify_rlxp, 1;
    };
}
