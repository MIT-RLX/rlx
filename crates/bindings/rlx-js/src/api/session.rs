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
//! the declared dims. The logic lives in `rlx_bind::runtime`; this file is the
//! JS adapter (argument conversion, the reusable input scratch).

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::{CompileRequest, Compiled, FlexibleSession, Session};

use crate::api::bind::lift;
use crate::api::graph::take_graph;
use crate::convert::*;
use crate::handle::{self, CLASS_COMPILED, CLASS_FLEXIBLE, CLASS_SESSION};

pub struct SessionSlot {
    pub inner: Session,
}

pub struct CompiledSlot {
    pub inner: Compiled,
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

// ── Session ─────────────────────────────────────────────────

/// `new rlx.Session("metal")` or `new rlx.Session({device, precision, policy})`.
fn session_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let first = arg(args, 0).clone();
    let positional = first.as_string().is_some() || is_nullish(&first);
    let (device, precision, policy_v) = if positional {
        let d = to_string_or(ctx, &first, "cpu")?;
        let p = to_string_or(ctx, arg(args, 1), "f32")?;
        (d, p, Value::Undefined)
    } else {
        let d = field(ctx, &first, "device")?;
        let p = field(ctx, &first, "precision")?;
        let policy = field(ctx, &first, "policy")?;
        (
            to_string_or(ctx, &d, "cpu")?,
            to_string_or(ctx, &p, "f32")?,
            policy,
        )
    };
    let policy = crate::api::compile::parse_precision_policy(ctx, &policy_v)?;
    let session = lift(ctx, Session::new(&device, &precision, policy))?;
    Ok(handle::wrap(
        ctx,
        CLASS_SESSION,
        SessionSlot { inner: session },
    ))
}

/// `compile(graph, opts?)`. Passing options is the same as `compileWith` — a
/// `{policy: "f16"}` here used to be silently dropped.
fn session_compile(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    session_compile_with(ctx, this, args, _m)
}

/// `compileWith(graph, {fusion, kernelDispatch, policy})`.
fn session_compile_with(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    // Options are read (and may run getters) before either handle is borrowed.
    let req = if is_nullish(arg(args, 1)) {
        CompileRequest::default()
    } else {
        crate::api::compile::compile_request(ctx, arg(args, 1))?
    };
    let graph = take_graph(ctx, arg(args, 0))?;
    let compiled = {
        let r = session_of(ctx, this)?.inner.compile(graph, &req);
        lift(ctx, r)?
    };
    Ok(handle::wrap(
        ctx,
        CLASS_COMPILED,
        CompiledSlot {
            inner: compiled,
            scratch: InputScratch::default(),
        },
    ))
}

fn session_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = session_of(ctx, this)?.inner.device_label();
    Ok(new_string(ctx, label))
}

fn session_precision(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = session_of(ctx, this)?.inner.precision_label();
    Ok(new_string(ctx, label))
}

/// The session's precision policy, or `null`.
fn session_policy(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = session_of(ctx, this)?.inner.policy_label();
    Ok(match label {
        Some(text) => new_string(ctx, text),
        None => Value::Null,
    })
}

fn session_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let s = &session_of(ctx, this)?.inner;
        let policy = s
            .policy_label()
            .map(|p| format!(" policy={p}"))
            .unwrap_or_default();
        format!(
            "[rlx.Session device={} precision={}{}]",
            s.device_label(),
            s.precision_label(),
            policy
        )
    };
    Ok(new_string(ctx, &text))
}

// ── Compiled ────────────────────────────────────────────────

fn compiled_set_param(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let data = to_f32_vec(ctx, arg(args, 1), &format!("setParam('{name}')"))?;
    let r = compiled_of(ctx, this)?.inner.set_param(&name, &data);
    lift(ctx, r)?;
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
        let r = compiled_of(ctx, this)?.inner.set_param(&name, &buffer);
        lift(ctx, r)?;
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
    let r = compiled_of(ctx, this)?
        .inner
        .set_param_typed(&name, &bytes, &label);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// `run({x: Float32Array})` → `[Float32Array, …]`. Refuses non-f32 outputs —
/// use `runTyped`, or `cast(x, "f32")` in the graph.
fn compiled_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
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
        let r = slot.inner.run(&pairs);
        lift(ctx, r)?
    };
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

/// `runTyped({x: {data: TypedArray, dtype: "f64"}})` → `[{data: Uint8Array, dtype}, …]`.
fn compiled_run_typed(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let names = own_keys(ctx, &bag)?;
    let mut owned: Vec<(String, Vec<u8>, String)> = Vec::with_capacity(names.len());
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
        owned.push((name, bytes, label));
    }
    let outs = {
        let refs: Vec<(&str, &[u8], &str)> = owned
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), t.as_str()))
            .collect();
        let r = compiled_of(ctx, this)?.inner.run_typed(&refs);
        lift(ctx, r)?
    };
    let values: Vec<Value> = outs
        .into_iter()
        .map(|(bytes, dtype)| {
            let data = new_u8_array(ctx, &bytes);
            let label = new_string(ctx, dtype);
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
    let shapes = compiled_of(ctx, this)?.inner.output_shapes.clone();
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
    let labels = compiled_of(ctx, this)?.inner.output_dtype_labels();
    Ok(new_string_array(ctx, &labels))
}

/// Declared inputs: `[{name, dtype, shape}]` (`shape` null when symbolic).
fn compiled_inputs(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let rows = compiled_of(ctx, this)?.inner.inputs();
    Ok(crate::api::bind::report(ctx, &rows))
}

/// Declared parameters, as `inputs()`.
fn compiled_params(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let rows = compiled_of(ctx, this)?.inner.params();
    Ok(crate::api::bind::report(ctx, &rows))
}

/// Outputs in `run` order: `[{name, dtype, shape}]` (`name` "" when unnamed).
fn compiled_outputs(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let rows = compiled_of(ctx, this)?.inner.outputs();
    Ok(crate::api::bind::report(ctx, &rows))
}

/// The position in `run`'s result of the output named `name`.
fn compiled_output_index(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let r = compiled_of(ctx, this)?.inner.output_index(&name);
    let i = lift(ctx, r)?;
    Ok(Value::number(i as f64))
}

/// Throws naming inputs `supplied` leaves unbound and params never set.
fn compiled_check_bindings(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let supplied = to_string_vec(ctx, arg(args, 0))?;
    let refs: Vec<&str> = supplied.iter().map(String::as_str).collect();
    let r = compiled_of(ctx, this)?.inner.check_bindings(&refs);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// Declare inputs / params meant to read zeros.
fn compiled_allow_unbound(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let names = to_string_vec(ctx, arg(args, 0))?;
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    compiled_of(ctx, this)?.inner.allow_unbound(&refs);
    Ok(Value::Undefined)
}

fn compiled_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = compiled_of(ctx, this)?.inner.device_label();
    Ok(new_string(ctx, label))
}

fn compiled_to_string(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let text = {
        let c = &compiled_of(ctx, this)?.inner;
        format!(
            "[rlx.Compiled device={} outputs={}]",
            c.device_label(),
            c.output_shapes.len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── FlexibleSession ─────────────────────────────────────────

fn flexible_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut FlexibleSession> {
    handle::borrow_mut(ctx, this, CLASS_FLEXIBLE, "rlx.FlexibleSession")
}

/// `new rlx.FlexibleSession(policy?)` — `{only, deny, prefer, benchmarkRuns}`.
fn flexible_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = crate::api::bind::to_json(ctx, arg(args, 0))?;
    let s = lift(ctx, FlexibleSession::new(&policy))?;
    Ok(handle::wrap(ctx, CLASS_FLEXIBLE, s))
}

/// `FlexibleSession.fromEnv()` — the policy from `RLX_DEVICE*`.
fn flexible_from_env(
    ctx: &mut Context,
    _this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    Ok(handle::wrap(
        ctx,
        CLASS_FLEXIBLE,
        FlexibleSession::from_env(),
    ))
}

/// `compile(graph, device?, opts?)` — on `device`, or the policy's pick.
fn flexible_compile(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let device = if is_nullish(arg(args, 1)) {
        None
    } else {
        Some(ctx.to_rust_string(arg(args, 1))?)
    };
    let req = if is_nullish(arg(args, 2)) {
        CompileRequest::default()
    } else {
        crate::api::compile::compile_request(ctx, arg(args, 2))?
    };
    let graph = take_graph(ctx, arg(args, 0))?;
    let compiled = {
        let r = flexible_of(ctx, this)?.compile(graph, device.as_deref(), &req);
        lift(ctx, r)?
    };
    Ok(handle::wrap(
        ctx,
        CLASS_COMPILED,
        CompiledSlot {
            inner: compiled,
            scratch: InputScratch::default(),
        },
    ))
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
    js_class! {
        ctx, namespace;
        name: "FlexibleSession",
        class: CLASS_FLEXIBLE,
        ctor: flexible_new,
        methods: {
            "compile" => flexible_compile, 3;
        }
    };
    js_statics! { ctx, namespace, "FlexibleSession"; "fromEnv" => flexible_from_env, 0; };
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
            "inputs" => compiled_inputs, 0;
            "params" => compiled_params, 0;
            "outputs" => compiled_outputs, 0;
            "outputIndex" => compiled_output_index, 1;
            "checkBindings" => compiled_check_bindings, 1;
            "allowUnbound" => compiled_allow_unbound, 1;
            "device" => compiled_device, 0;
            "toString" => compiled_to_string, 0;
        }
    };
}
