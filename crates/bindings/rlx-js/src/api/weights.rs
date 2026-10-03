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

use std::path::Path;

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, PropFlags};
use quickrs_core::value::{JsResult, Value};
use rlx_pkg::{ContainerKind, Package};

use crate::convert::*;

fn io_error<T>(ctx: &mut Context, what: &str, e: impl std::fmt::Display) -> JsResult<T> {
    ctx.throw_internal(&format!("{what}: {e:#}"))
}

/// `{name: Float32Array}` from a `(name, data)` iterator.
fn tensor_map(ctx: &mut Context, entries: Vec<(String, Vec<f32>)>) -> Value {
    let obj = ctx.new_object();
    for (name, data) in entries {
        let array = new_f32_array(ctx, &data);
        ctx.define_value(&obj, &name, array, PropFlags::C_W_E);
    }
    Value::Object(obj)
}

/// `loadMlx` (0) / `loadDduf` (1) / `loadNemo` (2) / `loadPt` (3).
fn f_load(ctx: &mut Context, _this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "weight loader")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let entries: Vec<(String, Vec<f32>)> = match magic {
        0 => match rlx_mlx_io::load_f32_map(&path) {
            Ok(map) => map.into_iter().collect(),
            Err(e) => return io_error(ctx, &format!("loadMlx('{path}')"), e),
        },
        1 => match rlx_dduf::load_f32_map(&path) {
            Ok(map) => map.into_iter().collect(),
            Err(e) => return io_error(ctx, &format!("loadDduf('{path}')"), e),
        },
        2 => {
            let model = match rlx_nemo::NemoModel::open(Path::new(&path)) {
                Ok(m) => m,
                Err(e) => return io_error(ctx, &format!("loadNemo('{path}')"), e),
            };
            let mut out = Vec::new();
            for name in model.names() {
                match model.tensor(&name) {
                    Ok(t) => out.push((name, t.data)),
                    Err(e) => return io_error(ctx, &format!("loadNemo tensor '{name}'"), e),
                }
            }
            out
        }
        _ => {
            let model = match rlx_nemo::PtModel::open(Path::new(&path)) {
                Ok(m) => m,
                Err(e) => return io_error(ctx, &format!("loadPt('{path}')"), e),
            };
            let mut out = Vec::new();
            for name in model.names() {
                match model.tensor(&name) {
                    Ok(t) => out.push((name, t.data)),
                    Err(e) => return io_error(ctx, &format!("loadPt tensor '{name}'"), e),
                }
            }
            out
        }
    };
    // Sorted, so writing params in iteration order is reproducible.
    let mut entries = entries;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(tensor_map(ctx, entries))
}

/// The four loaders differ only in which reader they reach for.
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

fn parse_container(ctx: &mut Context, v: &Value) -> JsResult<ContainerKind> {
    let label = to_string_or(ctx, v, "flat")?;
    Ok(match label.trim().to_ascii_lowercase().as_str() {
        "flat" => ContainerKind::Flat,
        "zip" => ContainerKind::Zip,
        "dir" | "directory" => ContainerKind::Dir,
        other => {
            return ctx.throw_type(&format!("unknown container '{other}' (flat, zip, dir)"));
        }
    })
}

/// Summary of an `.rlxp` package: name, versions, features, tensors, sidecars.
fn f_open_rlxp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "openRlxp")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let pack = match Package::open(&path) {
        Ok(p) => p,
        Err(e) => return io_error(ctx, &format!("openRlxp('{path}')"), e),
    };
    let manifest = pack.manifest();
    let tensors: Vec<String> = pack
        .weights_index()
        .map(|idx| idx.names().map(str::to_string).collect())
        .unwrap_or_default();
    let sidecars: Vec<String> = manifest.sidecars.iter().map(|s| s.id.clone()).collect();
    let features = manifest.features.clone();
    let has_graph = pack.has_graph();
    let name = manifest.name.clone();
    let format_version = manifest.format_version;
    let compat_version = manifest.compat_version;

    let name_v = new_string(ctx, &name);
    let features_v = new_string_array(ctx, &features);
    let tensors_v = new_string_array(ctx, &tensors);
    let sidecars_v = new_string_array(ctx, &sidecars);
    Ok(new_object(
        ctx,
        vec![
            ("name", name_v),
            ("formatVersion", Value::number(format_version as f64)),
            ("compatVersion", Value::number(compat_version as f64)),
            ("features", features_v),
            ("hasGraph", Value::Bool(has_graph)),
            ("tensors", tensors_v),
            ("sidecars", sidecars_v),
        ],
    ))
}

/// `toRlxp(src, out, {from, container, includeGraph, autoTier, keepPacked})`.
///
/// `from` is required: the source format cannot be inferred reliably from an
/// extension (`.bin` is a PyTorch checkpoint or a raw blob depending on who
/// wrote it), and guessing wrong here writes a silently empty package.
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
    let container_v = field(ctx, &options, "container")?;
    let container = parse_container(ctx, &container_v)?;
    let include_graph = to_bool(&field(ctx, &options, "includeGraph")?);
    let auto_tier_v = field(ctx, &options, "autoTier")?;
    let auto_tier = if is_nullish(&auto_tier_v) {
        true
    } else {
        to_bool(&auto_tier_v)
    };
    let keep_packed = to_bool(&field(ctx, &options, "keepPacked")?);

    let result = match from.trim().to_ascii_lowercase().as_str() {
        "gguf" => rlx_pkg::gguf_to_rlxp(
            &src,
            &out,
            &rlx_pkg::GgufImportOptions {
                container,
                include_graph,
                compress_sidecars: true,
                auto_tier,
            },
        ),
        "mlx" => rlx_pkg::mlx_to_rlxp(
            &src,
            &out,
            &rlx_pkg::MlxImportOptions {
                container,
                include_graph,
                compress_sidecars: true,
                auto_tier,
                dequant_to_f32: !keep_packed,
                ..Default::default()
            },
        ),
        "dduf" => rlx_pkg::dduf_to_rlxp(
            &src,
            &out,
            &rlx_pkg::DdufImportOptions {
                container,
                include_graph,
                compress_sidecars: true,
                auto_tier,
            },
        ),
        "nemo" => rlx_pkg::nemo_to_rlxp(
            &src,
            &out,
            &rlx_pkg::NemoImportOptions {
                container,
                include_graph,
                compress_sidecars: true,
                auto_tier,
            },
        ),
        "pt" | "torch" | "pytorch" => rlx_pkg::pt_to_rlxp(
            &src,
            &out,
            &rlx_pkg::PtImportOptions {
                container,
                include_graph,
                compress_sidecars: true,
                auto_tier,
            },
        ),
        other => {
            return ctx.throw_type(&format!(
                "toRlxp: unknown source format '{other}' (gguf, mlx, dduf, nemo, pt)"
            ));
        }
    };
    match result {
        Ok(()) => Ok(new_string(ctx, &out)),
        Err(e) => io_error(ctx, &format!("toRlxp('{src}' -> '{out}')"), e),
    }
}

/// Structural + checksum verification of a package.
///
/// `tensorsUnchecked` matters as much as `tensorsMismatch`: a package whose
/// entries carry no digest verifies with zero failures while proving nothing.
fn f_verify_rlxp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = crate::handle::require_fs(ctx, "verifyRlxp")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let pack = match Package::open(&path) {
        Ok(p) => p,
        Err(e) => return io_error(ctx, &format!("verifyRlxp('{path}')"), e),
    };
    let report = match rlx_pkg::verify_package(&pack) {
        Ok(r) => r,
        Err(e) => return io_error(ctx, &format!("verifyRlxp('{path}')"), e),
    };
    let failures = new_string_array(ctx, &report.failures);
    Ok(new_object(
        ctx,
        vec![
            ("ok", Value::Bool(report.failures.is_empty())),
            (
                "tensorsChecked",
                Value::number(report.tensors_checked as f64),
            ),
            ("tensorsOk", Value::number(report.tensors_ok as f64)),
            (
                "tensorsUnchecked",
                Value::number(report.tensors_unchecked as f64),
            ),
            (
                "tensorsMismatch",
                Value::number(report.tensors_mismatch as f64),
            ),
            (
                "sidecarsChecked",
                Value::number(report.sidecars_checked as f64),
            ),
            ("sidecarsOk", Value::number(report.sidecars_ok as f64)),
            ("failures", failures),
        ],
    ))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "loadMlx" => f_load_mlx, 1;
        "loadDduf" => f_load_dduf, 1;
        "loadNemo" => f_load_nemo, 1;
        "loadPt" => f_load_pt, 1;
        "openRlxp" => f_open_rlxp, 1;
        "toRlxp" => f_to_rlxp, 3;
        "verifyRlxp" => f_verify_rlxp, 1;
    };
}
