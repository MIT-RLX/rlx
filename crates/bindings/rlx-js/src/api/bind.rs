// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JS ↔ wire conversion for the generated `Graph` methods.
//!
//! The op table lives in `rlx-bind`; [`crate::js_graph_ops!`] expands it into
//! one `NativeFn` per row. Those functions do three things: convert each JS
//! argument to its wire value (this module), call `rlx_bind::ops::<op>`, and
//! convert the result back. Every argument is converted **before** the graph
//! handle is borrowed — converting can run script (a getter on an array-like),
//! and script could re-enter this same graph; see [`crate::handle`].

use quickrs_core::context::Context;
use quickrs_core::value::{JsResult, PropKey, Value};
use rlx_bind::{ApiError, ErrorKind};

use crate::convert::*;

/// Throw an [`ApiError`] as the matching JS exception.
pub fn throw<T>(ctx: &mut Context, e: ApiError) -> JsResult<T> {
    match e.kind {
        ErrorKind::Type => ctx.throw_type(&e.message),
        ErrorKind::Range => ctx.throw_range(&e.message),
        ErrorKind::Runtime | ErrorKind::Panic | ErrorKind::NotFound => {
            ctx.throw_internal(&e.message)
        }
    }
}

/// `Result<T, ApiError>` → `JsResult<T>`.
pub fn lift<T>(ctx: &mut Context, r: Result<T, ApiError>) -> JsResult<T> {
    match r {
        Ok(v) => Ok(v),
        Err(e) => throw(ctx, e),
    }
}

/// A node id. A DSL `Tensor` coerces through `.id` before it gets here.
pub fn node(ctx: &mut Context, v: &Value, what: &str) -> JsResult<u32> {
    let n = ctx.to_number(v)?;
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return ctx.throw_type(&format!(
            "{what}: expected a node id (a number returned by another Graph method), got {n}"
        ));
    }
    Ok(n as u32)
}

pub fn i64_of(ctx: &mut Context, v: &Value, what: &str) -> JsResult<i64> {
    let n = ctx.to_number(v)?;
    if !n.is_finite() || n.fract() != 0.0 {
        return ctx.throw_type(&format!("{what}: expected an integer, got {n}"));
    }
    Ok(n as i64)
}

/// A number means "this value on every axis"; an array is taken as-is.
pub fn dims_or_scalar(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<usize>> {
    if let Value::Int(_) | Value::Float(_) = v {
        return Ok(vec![to_usize(ctx, v, what)?]);
    }
    to_usize_vec(ctx, v, what)
}

/// `[[…], […], …]` rows of numbers; a bare number row reads as `[n]`.
pub fn usize_rows(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<Vec<usize>>> {
    let n = ctx.length_of(v)? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = ctx.get_property(v, &PropKey::Index(i as u32))?;
        out.push(dims_or_scalar(ctx, &row, what)?);
    }
    Ok(out)
}

pub fn f32_rows(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<Vec<f32>>> {
    let n = ctx.length_of(v)? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = ctx.get_property(v, &PropKey::Index(i as u32))?;
        out.push(to_f32_vec(ctx, &row, what)?);
    }
    Ok(out)
}

/// Read option `name` (camelCased) off an options object, falling back to the
/// legacy keys in `aliases`.
pub fn option(ctx: &mut Context, opts: &Value, name: &str, aliases: &[&str]) -> JsResult<Value> {
    let v = field(ctx, opts, &rlx_bind::manifest::camel(name))?;
    if !is_nullish(&v) {
        return Ok(v);
    }
    for a in aliases {
        let v = field(ctx, opts, &rlx_bind::manifest::camel(a))?;
        if !is_nullish(&v) {
            return Ok(v);
        }
    }
    Ok(Value::Undefined)
}

/// `(dims, dtype)` → `{dims: [...], dtype: "f32"}`.
pub fn shape_value(ctx: &mut Context, (dims, dtype): (Vec<i64>, String)) -> Value {
    let dims: Vec<Value> = dims.into_iter().map(|d| Value::number(d as f64)).collect();
    let dims = new_array(ctx, dims);
    let dtype = new_string(ctx, &dtype);
    new_object(ctx, vec![("dims", dims), ("dtype", dtype)])
}

/// One JS value → the wire value of `$kind`.
#[macro_export]
#[doc(hidden)]
macro_rules! js_wire {
    ($ctx:ident, $v:expr, $what:expr, node) => {
        $crate::api::bind::node($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, opt_node) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::api::bind::node($ctx, $v, $what)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, nodes) => {
        $crate::convert::to_u32_vec($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, f32) => {
        $crate::convert::to_f32($ctx, $v)?
    };
    ($ctx:ident, $v:expr, $what:expr, f64) => {
        $crate::convert::to_f64($ctx, $v)?
    };
    ($ctx:ident, $v:expr, $what:expr, i32) => {
        $crate::convert::to_i32($ctx, $v)?
    };
    ($ctx:ident, $v:expr, $what:expr, i64) => {
        $crate::api::bind::i64_of($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, u32) => {
        $crate::convert::to_usize($ctx, $v, $what)? as u32
    };
    ($ctx:ident, $v:expr, $what:expr, u64) => {
        $crate::convert::to_u64($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, usize) => {
        $crate::convert::to_usize($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, bool) => {
        $crate::convert::to_bool($v)
    };
    ($ctx:ident, $v:expr, $what:expr, usizes) => {
        $crate::convert::to_usize_vec($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, i64s) => {
        $crate::convert::to_i64_vec($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, f32s) => {
        $crate::convert::to_f32_vec($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, bytes) => {
        $crate::convert::to_bytes($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, opt_usize) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::convert::to_usize($ctx, $v, $what)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, opt_f32) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::convert::to_f32($ctx, $v)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, opt_usizes) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::convert::to_usize_vec($ctx, $v, $what)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, opt_str) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($ctx.to_rust_string($v)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, opt_activation) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($ctx.to_rust_string($v)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, pair) => {
        $crate::api::bind::dims_or_scalar($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, triple) => {
        $crate::api::bind::dims_or_scalar($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, opt_pair) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::api::bind::dims_or_scalar($ctx, $v, $what)?)
        }
    };
    ($ctx:ident, $v:expr, $what:expr, f32x3) => {
        $crate::convert::to_f32_vec($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, pads) => {
        $crate::api::bind::usize_rows($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, bands) => {
        $crate::api::bind::f32_rows($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, sos) => {
        $crate::api::bind::f32_rows($ctx, $v, $what)?
    };
    ($ctx:ident, $v:expr, $what:expr, graph) => {
        $crate::api::graph::take_graph($ctx, $v)?
    };
    ($ctx:ident, $v:expr, $what:expr, opt_graph) => {
        if $crate::convert::is_nullish($v) {
            None
        } else {
            Some($crate::api::graph::take_graph($ctx, $v)?)
        }
    };
    // `str` and every string-selected enum.
    ($ctx:ident, $v:expr, $what:expr, $string_kind:ident) => {
        $ctx.to_rust_string($v)?
    };
}

/// [`js_wire!`], or the table default when the value is `undefined` / `null`.
#[macro_export]
#[doc(hidden)]
macro_rules! js_wire_or {
    ($ctx:ident, $v:expr, $what:expr, $kind:ident) => {
        $crate::js_wire!($ctx, $v, $what, $kind)
    };
    ($ctx:ident, $v:expr, $what:expr, $kind:ident, $default:expr) => {
        if $crate::convert::is_nullish($v) {
            rlx_bind::wire_default!($kind, $default)
        } else {
            $crate::js_wire!($ctx, $v, $what, $kind)
        }
    };
}

/// A wire result → the JS value a script sees.
#[macro_export]
#[doc(hidden)]
macro_rules! js_ret {
    ($ctx:ident, (), $v:expr) => {
        quickrs_core::value::Value::number($v as f64)
    };
    ($ctx:ident, (node), $v:expr) => {
        quickrs_core::value::Value::number($v as f64)
    };
    ($ctx:ident, (pair), $v:expr) => {{
        let (a, b) = $v;
        let items = vec![
            quickrs_core::value::Value::number(a as f64),
            quickrs_core::value::Value::number(b as f64),
        ];
        $crate::convert::new_array($ctx, items)
    }};
    ($ctx:ident, (nodes), $v:expr) => {{
        let items = $v
            .into_iter()
            .map(|n| quickrs_core::value::Value::number(n as f64))
            .collect();
        $crate::convert::new_array($ctx, items)
    }};
    ($ctx:ident, (unit), $v:expr) => {{
        let () = $v;
        quickrs_core::value::Value::Undefined
    }};
    ($ctx:ident, (shape), $v:expr) => {
        $crate::api::bind::shape_value($ctx, $v)
    };
    ($ctx:ident, (strings), $v:expr) => {
        $crate::convert::new_string_array($ctx, &$v)
    };
    ($ctx:ident, (usize), $v:expr) => {
        quickrs_core::value::Value::number($v as f64)
    };
    ($ctx:ident, (string), $v:expr) => {
        $crate::convert::new_string($ctx, &$v)
    };
}

/// `Function.length` of a generated method: positional count, plus one for
/// the options object.
#[macro_export]
#[doc(hidden)]
macro_rules! js_len {
    ([$($arg:ident)*] [$($opts:tt)?]) => {{
        let slots: &[&str] = &[$(stringify!($arg)),*];
        let options: &[&str] = &[$(stringify!($opts)),*];
        (slots.len() + options.len()) as u32
    }};
}

/// Install every op-table row on `proto` as a `Graph` method.
///
/// Invoked as `rlx_bind::for_each_graph_op!(crate::js_graph_ops, ctx, proto)`.
/// Each method converts its arguments left to right, then borrows the graph,
/// then calls the row's core function — so the conversion-before-borrow rule
/// holds for every op without a per-op line enforcing it.
#[macro_export]
macro_rules! js_graph_ops {
    ( [$ctx:expr, $proto:expr]
      $(
        $(#[doc = $doc:literal])*
        $name:ident $( [ $( $rk:ident = $rv:literal ),* $(,)? ] )?
        ( $( $arg:ident : $kind:ident $( = $def:expr )? ),* $(,)?
          $( ; { $( $oarg:ident $( | $oal:ident )* : $okind:ident $( = $odef:expr )? ),* $(,)? } )? )
        $( -> $ret:ident )?
        => | $g:ident | $body:block ;
      )*
    ) => {{
        let methods: &[(&str, quickrs_core::object::NativeFn, u32)] = &[
            $((
                stringify!($name),
                {
                    #[allow(unused_variables, unused_mut, unused_assignments)]
                    let f: quickrs_core::object::NativeFn = |ctx, this, args, _magic| {
                        let mut cur = 0usize;
                        $(
                            let $arg = $crate::js_wire_or!(
                                ctx,
                                $crate::convert::arg(args, cur),
                                concat!(stringify!($name), " ", stringify!($arg)),
                                $kind $(, $def)?
                            );
                            cur += 1;
                        )*
                        $(
                            let opts = $crate::convert::arg(args, cur).clone();
                            $(
                                let v = $crate::api::bind::option(
                                    ctx, &opts, stringify!($oarg), &[$(stringify!($oal)),*],
                                )?;
                                let $oarg = $crate::js_wire_or!(
                                    ctx,
                                    &v,
                                    concat!(stringify!($name), " ", stringify!($oarg)),
                                    $okind $(, $odef)?
                                );
                            )*
                        )?
                        let graph = $crate::api::graph::graph_of(ctx, this)?;
                        let out = rlx_bind::ops::$name(graph, $($arg,)* $($($oarg,)*)?);
                        let out = $crate::api::bind::lift(ctx, out)?;
                        Ok($crate::js_ret!(ctx, ($($ret)?), out))
                    };
                    f
                },
                $crate::js_len!([$($arg)*] [$( { $($oarg)* } )?]),
            )),*
        ];
        for (canonical, func, arity) in methods {
            let spec = rlx_bind::manifest::op(canonical).expect("row is in the manifest");
            for js in spec.js_names() {
                $ctx.define_method($proto, &js, *func, *arity, 0);
            }
        }
    }};
}

// ── JSON: structured options in, reports out ────────────────

use rlx_bind::runtime::Value as Json;

/// A plain JS value → JSON, for the options objects the core parses.
///
/// Numbers keep their integral-ness (`3` reads as an integer), `undefined`
/// becomes `null`, and functions are refused rather than dropped.
pub fn to_json(ctx: &mut Context, v: &Value) -> JsResult<Json> {
    Ok(match v {
        Value::Undefined | Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 9.0e15 {
                Json::from(*f as i64)
            } else {
                rlx_bind::runtime::serde_json::Number::from_f64(*f)
                    .map(Json::Number)
                    .unwrap_or(Json::Null)
            }
        }
        _ if v.as_string().is_some() => Json::String(ctx.to_rust_string(v)?),
        _ if is_js_array(v) => {
            let n = ctx.length_of(v)? as usize;
            let mut items = Vec::with_capacity(n);
            for i in 0..n {
                let item = ctx.get_property(v, &PropKey::Index(i as u32))?;
                items.push(to_json(ctx, &item)?);
            }
            Json::Array(items)
        }
        _ if with_f32_view(v, |_| ()).is_some() => {
            let values = to_f32_vec(ctx, v, "options")?;
            Json::Array(values.into_iter().map(|f| Json::from(f as f64)).collect())
        }
        _ if ctx.is_callable(v) => {
            return ctx.throw_type("expected plain data (an options object), got a function");
        }
        _ if v.as_object().is_some() => {
            let mut map = rlx_bind::runtime::serde_json::Map::new();
            for k in own_keys(ctx, v)? {
                let item = field(ctx, v, &k)?;
                map.insert(k, to_json(ctx, &item)?);
            }
            Json::Object(map)
        }
        _ => Json::String(ctx.to_rust_string(v)?),
    })
}

fn is_js_array(v: &Value) -> bool {
    matches!(v, Value::Object(o) if o.borrow().class == quickrs_core::object::ClassId::Array)
}

/// JSON → a plain JS value, keys as-is (for data whose keys are names:
/// metadata, parameter bags).
pub fn from_json(ctx: &mut Context, v: &Json) -> Value {
    convert_json(ctx, v, false)
}

/// A core report (snake_case keys) → a JS object with camelCase keys —
/// `fused_ops` → `fusedOps`, the JS spelling of every report field.
pub fn report(ctx: &mut Context, v: &Json) -> Value {
    convert_json(ctx, v, true)
}

fn convert_json(ctx: &mut Context, v: &Json, camel: bool) -> Value {
    match v {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => Value::number(n.as_f64().unwrap_or(f64::NAN)),
        Json::String(s) => new_string(ctx, s),
        Json::Array(items) => {
            let values = items.iter().map(|i| convert_json(ctx, i, camel)).collect();
            new_array(ctx, values)
        }
        Json::Object(map) => {
            let fields: Vec<(String, Value)> = map
                .iter()
                .map(|(k, v)| {
                    let k = if camel {
                        rlx_bind::manifest::camel(k)
                    } else {
                        k.clone()
                    };
                    (k, convert_json(ctx, v, camel))
                })
                .collect();
            let refs = fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            new_object(ctx, refs)
        }
    }
}
