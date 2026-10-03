// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.Session` and the `Compiled` artifact it produces.
//!
//! ```js
//! const s = new rlx.Session({ device: "metal", precision: "f16" });
//! const c = s.compile(g);            // takes ownership of g
//! c.setParam("w", new Float32Array([...]));
//! const [y] = c.run({ x: new Float32Array([...]) });
//! ```
//!
//! Outputs come back as flat `Float32Array`s — JavaScript has no ndarray, so
//! reshaping is left to the caller, with `compiled.outputShapes()` supplying
//! the declared dims.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_ir::{DType, Dim, Shape};
use rlx_runtime::{CompiledGraph, Precision, Session};

use crate::api::graph::{dtype_label, parse_dtype, take_graph};
use crate::convert::*;
use crate::handle::{self, CLASS_COMPILED, CLASS_SESSION};

pub struct SessionSlot {
    pub inner: Session,
    pub device: &'static str,
    /// A session-level policy applies to `compile` as well as `compileWith`,
    /// so `new rlx.Session({policy})` does not silently need the long form.
    pub policy: Option<rlx_runtime::PrecisionPolicy>,
}

pub struct CompiledSlot {
    pub inner: CompiledGraph,
    /// One entry per graph output; a dynamic dim is recorded as `0`.
    pub output_shapes: Vec<Vec<usize>>,
    /// Declared dtype per output. `run` is the f32 path, so anything else has
    /// to be refused rather than reinterpreted.
    pub output_dtypes: Vec<DType>,
    /// Input buffers reused across `run` calls, so a step loop allocates
    /// nothing after the first call.
    pub scratch: InputScratch,
}

fn session_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut SessionSlot> {
    handle::borrow_mut(ctx, this, CLASS_SESSION, "rlx.Session")
}

fn compiled_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut CompiledSlot> {
    handle::borrow_mut(ctx, this, CLASS_COMPILED, "compiled graph")
}

fn parse_precision(ctx: &mut Context, s: &str) -> JsResult<Precision> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Precision::F32,
        "f16" | "float16" | "half" => Precision::F16,
        "bf16" | "bfloat16" => Precision::BF16,
        other => return ctx.throw_type(&format!("unknown precision '{other}' (f32, f16, bf16)")),
    })
}

fn precision_label(p: Precision) -> &'static str {
    match p {
        Precision::F32 => "f32",
        Precision::F16 => "f16",
        Precision::BF16 => "bf16",
    }
}

/// Which outputs `run` cannot represent, as `index:dtype` pairs.
fn non_f32_outputs(dtypes: &[DType]) -> Vec<String> {
    dtypes
        .iter()
        .enumerate()
        .filter(|(_, d)| **d != DType::F32)
        .map(|(i, d)| format!("{i}:{}", dtype_label(*d)))
        .collect()
}

fn static_dims(shape: &Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => 0,
        })
        .collect()
}

// ── Session ─────────────────────────────────────────────────

/// `new rlx.Session("metal")` or `new rlx.Session({device, precision})`.
fn session_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let first = arg(args, 0).clone();
    let (device_label, precision_str) = if first.as_string().is_some() || is_nullish(&first) {
        let d = to_string_or(ctx, &first, "cpu")?;
        let p = to_string_or(ctx, arg(args, 1), "f32")?;
        (d, p)
    } else {
        let d = field(ctx, &first, "device")?;
        let p = field(ctx, &first, "precision")?;
        (to_string_or(ctx, &d, "cpu")?, to_string_or(ctx, &p, "f32")?)
    };

    let device = match rlx_runtime::parse_device(&device_label) {
        Ok(d) => d,
        Err(e) => return ctx.throw_type(&e.to_string()),
    };
    if !rlx_runtime::is_available(device) {
        let have = rlx_runtime::available_devices()
            .into_iter()
            .map(rlx_runtime::device_label)
            .collect::<Vec<_>>()
            .join(", ");
        return ctx.throw_type(&format!(
            "device '{device_label}' is not in this build — available: [{have}]. \
             Rebuild with `--features {device_label}`."
        ));
    }
    let precision = parse_precision(ctx, &precision_str)?;
    let policy_v = if first.as_string().is_some() || is_nullish(&first) {
        Value::Undefined
    } else {
        field(ctx, &first, "policy")?
    };
    let policy = crate::api::compile::parse_precision_policy(ctx, &policy_v)?;
    if let Some(policy) = policy.as_ref() {
        crate::api::compile::check_policy_supported(ctx, device, policy)?;
    }
    let slot = SessionSlot {
        inner: Session::new_with_precision(device, precision),
        device: rlx_runtime::device_label(device),
        policy,
    };
    Ok(handle::wrap(ctx, CLASS_SESSION, slot))
}

fn session_compile(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    // `compile(graph, opts)` used to take `opts` and silently drop it — the
    // options form was only on `compileWith`, and `compile` was registered with
    // arity 1. A `{policy: "f16"}` passed here did nothing at all, which is the
    // worst outcome: not an error, just a setting that never applied. Forward to
    // the options path instead so both spellings behave the same.
    if !is_nullish(arg(args, 1)) {
        return session_compile_with(ctx, this, args, _m);
    }
    let graph = take_graph(ctx, arg(args, 0))?;
    let output_shapes: Vec<Vec<usize>> = graph
        .outputs
        .iter()
        .map(|n| static_dims(graph.shape(*n)))
        .collect();
    let output_dtypes: Vec<DType> = graph
        .outputs
        .iter()
        .map(|n| graph.shape(*n).dtype())
        .collect();
    // Legalization, fusion and memory planning all assert; a graph the backend
    // cannot take should be an exception, not an abort.
    let compiled = {
        let session = session_of(ctx, this)?;
        match session.policy.clone() {
            // A session policy has to go through `compile_with`, since plain
            // `compile` takes no options.
            Some(policy) => {
                let opts = rlx_runtime::CompileOptions::new()
                    .precision(session.inner.precision())
                    .policy(policy);
                crate::panics::catch(ctx, "Session.compile", || {
                    session.inner.compile_with(graph, &opts)
                })?
            }
            None => crate::panics::catch(ctx, "Session.compile", || session.inner.compile(graph))?,
        }
    };
    Ok(handle::wrap(
        ctx,
        CLASS_COMPILED,
        CompiledSlot {
            inner: compiled,
            output_shapes,
            output_dtypes,
            scratch: InputScratch::default(),
        },
    ))
}

/// `compileWith(graph, {fusion, kernelDispatch})` — same as `compile`, with
/// the per-compile knobs exposed.
fn session_compile_with(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    // Options are read (and may run getters) before either handle is borrowed.
    let (precision, session_policy) = {
        let session = session_of(ctx, this)?;
        (session.inner.precision(), session.policy.clone())
    };
    let options = arg(args, 1).clone();
    let mut opts = crate::api::compile::build_compile_options(ctx, precision, &options)?;
    // A per-call `policy` wins; otherwise the session's applies.
    if opts.policy.is_none() {
        opts.policy = session_policy;
    }
    if let Some(policy) = opts.policy.clone() {
        let device = session_of(ctx, this)?.inner.device();
        crate::api::compile::check_policy_supported(ctx, device, &policy)?;
    }

    let graph = take_graph(ctx, arg(args, 0))?;
    let output_shapes: Vec<Vec<usize>> = graph
        .outputs
        .iter()
        .map(|n| static_dims(graph.shape(*n)))
        .collect();
    let output_dtypes: Vec<DType> = graph
        .outputs
        .iter()
        .map(|n| graph.shape(*n).dtype())
        .collect();
    let compiled = {
        let session = session_of(ctx, this)?;
        crate::panics::catch(ctx, "Session.compileWith", || {
            session.inner.compile_with(graph, &opts)
        })?
    };
    Ok(handle::wrap(
        ctx,
        CLASS_COMPILED,
        CompiledSlot {
            inner: compiled,
            output_shapes,
            output_dtypes,
            scratch: InputScratch::default(),
        },
    ))
}

fn session_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = session_of(ctx, this)?.device;
    Ok(new_string(ctx, label))
}

fn session_precision(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = precision_label(session_of(ctx, this)?.inner.precision());
    Ok(new_string(ctx, label))
}

/// The session's precision policy, or `null`.
fn session_policy(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = session_of(ctx, this)?
        .policy
        .as_ref()
        .map(crate::api::compile::policy_label);
    Ok(match label {
        Some(text) => new_string(ctx, text),
        None => Value::Null,
    })
}

fn session_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let s = session_of(ctx, this)?;
        let policy = s
            .policy
            .as_ref()
            .map(|p| format!(" policy={}", crate::api::compile::policy_label(p)))
            .unwrap_or_default();
        format!(
            "[rlx.Session device={} precision={}{}]",
            s.device,
            precision_label(s.inner.precision()),
            policy
        )
    };
    Ok(new_string(ctx, &text))
}

// ── Compiled ────────────────────────────────────────────────

fn compiled_set_param(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let data = to_f32_vec(ctx, arg(args, 1), &format!("setParam('{name}')"))?;
    compiled_of(ctx, this)?.inner.set_param(&name, &data);
    Ok(Value::Undefined)
}

/// `setParams({w: Float32Array, b: Float32Array})` — one call per checkpoint.
fn compiled_set_params(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let names = own_keys(ctx, &bag)?;
    // One reused buffer for the whole bag: params go straight to the device,
    // so nothing has to outlive its own upload.
    let mut buffer: Vec<f32> = Vec::new();
    let mut uploaded = 0usize;
    for name in names {
        let value = field(ctx, &bag, &name)?;
        let label = format!("setParams('{name}')");
        read_f32_into(ctx, &value, &label, &mut buffer)?;
        compiled_of(ctx, this)?.inner.set_param(&name, &buffer);
        uploaded += 1;
    }
    Ok(Value::number(uploaded as f64))
}

fn compiled_set_param_typed(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let bytes = to_bytes(ctx, arg(args, 1), &format!("setParamTyped('{name}')"))?;
    let label = ctx.to_rust_string(arg(args, 2))?;
    let dtype = parse_dtype(ctx, &label)?;
    compiled_of(ctx, this)?
        .inner
        .set_param_typed(&name, &bytes, dtype);
    Ok(Value::Undefined)
}

/// `run({x: Float32Array})` → `[Float32Array, …]`.
///
/// Refuses a graph whose outputs are not all f32. The bytes coming back are
/// reinterpreted, not converted, so a `Bool` output read this way is denormal
/// garbage that looks like data — `cmp` came back as `2.37e-38` rather than
/// `1`. Use `runTyped`, or `cast(x, "f32")` in the graph.
fn compiled_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    {
        let dtypes = compiled_of(ctx, this)?.output_dtypes.clone();
        let offenders = non_f32_outputs(&dtypes);
        if !offenders.is_empty() {
            return ctx.throw_type(&format!(
                "run(): output(s) [{}] are not f32, and run() reinterprets bytes rather \
                 than converting them. Use runTyped(), or cast the output to f32 in the \
                 graph.",
                offenders.join(", ")
            ));
        }
    }
    let bag = arg(args, 0).clone();
    // Two borrows of the same handle would alias, so the scratch is moved out,
    // filled while script may run, and put back before the graph executes.
    let slot = compiled_of(ctx, this)?;
    let mut scratch = std::mem::take(&mut slot.scratch);
    let loaded = scratch.load(ctx, &bag);
    let slot = compiled_of(ctx, this)?;
    slot.scratch = scratch;
    loaded?;

    let outs = {
        let slot = compiled_of(ctx, this)?;
        let pairs = slot.scratch.pairs();
        crate::panics::catch(ctx, "run", || slot.inner.run(&pairs))?
    };
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

fn compiled_run_typed(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let names = own_keys(ctx, &bag)?;
    let mut owned: Vec<(String, Vec<u8>, rlx_ir::DType)> = Vec::with_capacity(names.len());
    for name in names {
        let entry = field(ctx, &bag, &name)?;
        let data_v = field(ctx, &entry, "data")?;
        let dtype_v = field(ctx, &entry, "dtype")?;
        if is_nullish(&data_v) || is_nullish(&dtype_v) {
            return ctx.throw_type(&format!(
                "runTyped input '{name}': expected {{data: TypedArray, dtype: \"f64\"}}"
            ));
        }
        let bytes = to_bytes(ctx, &data_v, &format!("runTyped('{name}')"))?;
        let label = ctx.to_rust_string(&dtype_v)?;
        let dtype = parse_dtype(ctx, &label)?;
        owned.push((name, bytes, dtype));
    }
    let outs = {
        let refs: Vec<(&str, &[u8], rlx_ir::DType)> = owned
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t))
            .collect();
        let slot = compiled_of(ctx, this)?;
        crate::panics::catch(ctx, "runTyped", || slot.inner.run_typed(&refs))?
    };
    let values: Vec<Value> = outs
        .into_iter()
        .map(|(bytes, dtype)| {
            let data = new_u8_array(ctx, &bytes);
            let label = new_string(ctx, dtype_label(dtype));
            new_object(ctx, vec![("data", data), ("dtype", label)])
        })
        .collect();
    Ok(new_array(ctx, values))
}

/// Declared dims per output; a dynamic axis reads back as `0`.
fn compiled_output_shapes(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let shapes = compiled_of(ctx, this)?.output_shapes.clone();
    let values: Vec<Value> = shapes.iter().map(|s| new_usize_array(ctx, s)).collect();
    Ok(new_array(ctx, values))
}

/// Declared dtype per output — the companion to `outputShapes()`.
fn compiled_output_dtypes(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let labels: Vec<&'static str> = compiled_of(ctx, this)?
        .output_dtypes
        .iter()
        .map(|d| dtype_label(*d))
        .collect();
    Ok(new_string_array(ctx, &labels))
}

fn compiled_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = rlx_runtime::device_label(compiled_of(ctx, this)?.inner.device());
    Ok(new_string(ctx, label))
}

fn compiled_to_string(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let text = {
        let c = compiled_of(ctx, this)?;
        format!(
            "[rlx.Compiled device={} outputs={}]",
            rlx_runtime::device_label(c.inner.device()),
            c.output_shapes.len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── installation ────────────────────────────────────────────

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_class! {
        ctx, namespace;
        name: "Session",
        class: CLASS_SESSION,
        ctor: session_new,
        methods: {
            "compile" => session_compile, 2;
            "compileWith" => session_compile_with, 2;
            "device" => session_device, 0;
            "precision" => session_precision, 0;
            "policy" => session_policy, 0;
            "toString" => session_to_string, 0;
        }
    };
    // No constructor: a compiled graph only ever comes from `Session.compile`.
    js_class! {
        ctx, namespace;
        name: "Compiled",
        class: CLASS_COMPILED,
        methods: {
            "setParam" => compiled_set_param, 2;
            "setParams" => compiled_set_params, 1;
            "setParamTyped" => compiled_set_param_typed, 3;
            "run" => compiled_run, 1;
            "runTyped" => compiled_run_typed, 1;
            "outputShapes" => compiled_output_shapes, 0;
            "outputDtypes" => compiled_output_dtypes, 0;
            "device" => compiled_device, 0;
            "toString" => compiled_to_string, 0;
        }
    };
}
