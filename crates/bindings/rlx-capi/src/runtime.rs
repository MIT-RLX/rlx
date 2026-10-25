// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sessions, compiled graphs, autodiff, `check`, devices and routing.

use std::ffi::c_char;

use rlx_bind::runtime::devices::{self, Router, Runner};
use rlx_bind::runtime::{
    CompileRequest, Compiled, FlexibleSession, Session, autodiff, check, policy,
};

use crate::ffi::{arg, borrow, guard, handle, out, release};
use crate::graph::{RlxGraph, graph_mut, take_graph};

// ── Session ─────────────────────────────────────────────────

/// A device and precision to compile graphs for.
pub struct RlxSession(Session);

/// A graph compiled for one device.
pub struct RlxCompiled(Compiled);

/// The outputs of a run: one f32 buffer per graph output.
pub struct RlxOutputs {
    pub(crate) outs: Vec<Vec<f32>>,
}

/// The outputs of a typed run: bytes plus a dtype name per output.
pub struct RlxTypedOutputs {
    pub(crate) outs: Vec<(Vec<u8>, &'static str)>,
    pub(crate) labels: Vec<std::ffi::CString>,
}

/// Open a session. `device` is `"cpu"`, `"metal"`, `"gpu"`, …; `precision`
/// (`NULL` = `"f32"`) is `"f32"`, `"f16"` or `"bf16"`; `policy_json` (`NULL` =
/// none) is a preset name as a JSON string (`"\"mixed\""`) or an object of op
/// kinds (`{"compute": "f16"}`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_new(
    device: *const c_char,
    precision: *const c_char,
    policy_json: *const c_char,
    out: *mut *mut RlxSession,
) -> i32 {
    guard(|| unsafe {
        let device = arg::str_or(device, "cpu", "rlx_session_new device")?;
        let precision = arg::str_or(precision, "f32", "rlx_session_new precision")?;
        let policy = policy::parse(&arg::json(policy_json, "rlx_session_new policy")?)?;
        let s = Session::new(&device, &precision, policy)?;
        out::put(out, handle(RlxSession(s)), "rlx_session_new out")
    })
}

/// Free a session (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_free(s: *mut RlxSession) {
    unsafe { release(s) }
}

/// The session's device label (free with `rlx_string_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_device(s: *mut RlxSession, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let label = borrow(s, "session")?.0.device_label();
        out::string(out, label, "rlx_session_device out")
    })
}

/// The session's precision label (free with `rlx_string_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_precision(s: *mut RlxSession, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let label = borrow(s, "session")?.0.precision_label();
        out::string(out, label, "rlx_session_precision out")
    })
}

/// The session's policy label, or `NULL` when it has none.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_policy(s: *mut RlxSession, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        match borrow(s, "session")?.0.policy_label() {
            Some(label) => out::string(out, label, "rlx_session_policy out"),
            None => out::put(out, std::ptr::null_mut(), "rlx_session_policy out"),
        }
    })
}

/// Compile `graph` (consumed). `options_json` (`NULL` = defaults) is
/// `{"fusion": {…}, "kernel_dispatch": "native", "policy": "mixed"}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_session_compile(
    s: *mut RlxSession,
    graph: *mut RlxGraph,
    options_json: *const c_char,
    out: *mut *mut RlxCompiled,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_session_compile options")?;
        let req = CompileRequest::from_json(&opts)?;
        let session = borrow(s, "session")?;
        let g = take_graph(graph, "rlx_session_compile graph")?;
        let c = session.0.compile(g, &req)?;
        out::put(out, handle(RlxCompiled(c)), "rlx_session_compile out")
    })
}

// ── FlexibleSession ─────────────────────────────────────────

/// A session that picks the device per graph from a device policy.
pub struct RlxFlexibleSession(FlexibleSession);

/// A flexible session. `policy_json` (`NULL` = every device): `"env"`, or
/// `{"only": […], "deny": […], "prefer": […], "benchmark_runs": n}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_flexible_session_new(
    policy_json: *const c_char,
    out: *mut *mut RlxFlexibleSession,
) -> i32 {
    guard(|| unsafe {
        let policy = arg::json(policy_json, "rlx_flexible_session_new policy")?;
        let s = FlexibleSession::new(&policy)?;
        out::put(
            out,
            handle(RlxFlexibleSession(s)),
            "rlx_flexible_session_new out",
        )
    })
}

/// A flexible session with the device policy from `RLX_DEVICE*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_flexible_session_from_env(out: *mut *mut RlxFlexibleSession) -> i32 {
    guard(|| unsafe {
        let s = FlexibleSession::from_env();
        out::put(
            out,
            handle(RlxFlexibleSession(s)),
            "rlx_flexible_session_from_env out",
        )
    })
}

/// Free a flexible session (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_flexible_session_free(s: *mut RlxFlexibleSession) {
    unsafe { release(s) }
}

/// Compile `graph` (consumed) on `device` (`NULL` = the policy's pick);
/// `options_json` as for `rlx_session_compile`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_flexible_session_compile(
    s: *mut RlxFlexibleSession,
    graph: *mut RlxGraph,
    device: *const c_char,
    options_json: *const c_char,
    out: *mut *mut RlxCompiled,
) -> i32 {
    guard(|| unsafe {
        let device = arg::opt_str(device, "rlx_flexible_session_compile device")?;
        let opts = arg::json(options_json, "rlx_flexible_session_compile options")?;
        let req = CompileRequest::from_json(&opts)?;
        let session = borrow(s, "flexible session")?;
        let g = take_graph(graph, "rlx_flexible_session_compile graph")?;
        let c = session.0.compile(g, device.as_deref(), &req)?;
        out::put(
            out,
            handle(RlxCompiled(c)),
            "rlx_flexible_session_compile out",
        )
    })
}

// ── Compiled ────────────────────────────────────────────────

/// Free a compiled graph (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_free(c: *mut RlxCompiled) {
    unsafe { release(c) }
}

/// Set one f32 parameter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_set_param(
    c: *mut RlxCompiled,
    name: *const c_char,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_compiled_set_param name")?;
        let data = arg::slice(data, len, "rlx_compiled_set_param data")?;
        borrow(c, "compiled")?.0.set_param(name, data)
    })
}

/// Set `n` f32 parameters (parallel arrays of names, data pointers and
/// lengths); writes how many.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_set_params(
    c: *mut RlxCompiled,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    applied: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let params = arg::inputs(names, data, lens, n, "rlx_compiled_set_params params")?;
        let c = borrow(c, "compiled")?;
        for (name, values) in &params {
            c.0.set_param(name, values)?;
        }
        out::put(applied, params.len(), "rlx_compiled_set_params applied")
    })
}

/// Set one parameter from raw bytes of `dtype` (`"f16"`, `"i8"`, …).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_set_param_typed(
    c: *mut RlxCompiled,
    name: *const c_char,
    data: *const u8,
    len: usize,
    dtype: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_compiled_set_param_typed name")?;
        let data = arg::slice(data, len, "rlx_compiled_set_param_typed data")?;
        let dtype = arg::str(dtype, "rlx_compiled_set_param_typed dtype")?;
        borrow(c, "compiled")?.0.set_param_typed(name, data, dtype)
    })
}

/// Run with `n` f32 inputs (parallel arrays of names, data pointers and
/// lengths). Refuses non-f32 outputs — use `rlx_compiled_run_typed`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_run(
    c: *mut RlxCompiled,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_compiled_run inputs")?;
        let outs = borrow(c, "compiled")?.0.run(&inputs)?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_compiled_run out")
    })
}

/// Run with `n` typed inputs (names, byte pointers, byte lengths, dtype names).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_run_typed(
    c: *mut RlxCompiled,
    names: *const *const c_char,
    data: *const *const u8,
    lens: *const usize,
    dtypes: *const *const c_char,
    n: usize,
    out: *mut *mut RlxTypedOutputs,
) -> i32 {
    guard(|| unsafe {
        let what = "rlx_compiled_run_typed inputs";
        let names = arg::slice(names, n, what)?;
        let data = arg::slice(data, n, what)?;
        let lens = arg::slice(lens, n, what)?;
        let dtypes = arg::slice(dtypes, n, what)?;
        let mut inputs = Vec::with_capacity(n);
        for i in 0..n {
            inputs.push((
                arg::str(names[i], what)?,
                arg::slice(data[i], lens[i], what)?,
                arg::str(dtypes[i], what)?,
            ));
        }
        let outs = borrow(c, "compiled")?.0.run_typed(&inputs)?;
        let labels = outs
            .iter()
            .map(|(_, d)| std::ffi::CString::new(*d).unwrap_or_default())
            .collect();
        out::put(
            out,
            handle(RlxTypedOutputs { outs, labels }),
            "rlx_compiled_run_typed out",
        )
    })
}

/// Declared dims per output as JSON (`[[2, 3], [4]]`); a dynamic axis is `0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_output_shapes(
    c: *mut RlxCompiled,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let shapes = &borrow(c, "compiled")?.0.output_shapes;
        let v = rlx_bind::runtime::serde_json::json!(shapes);
        out::json(out, &v, "rlx_compiled_output_shapes out")
    })
}

/// Declared dtype per output as JSON (`["f32", "i64"]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_output_dtypes(
    c: *mut RlxCompiled,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let labels: Vec<String> = borrow(c, "compiled")?
            .0
            .output_dtype_labels()
            .into_iter()
            .map(str::to_string)
            .collect();
        out::strings(out, &labels, "rlx_compiled_output_dtypes out")
    })
}

/// Declared inputs as JSON rows `[{"name", "dtype", "shape"}]`; `shape` is
/// `null` when a dim is symbolic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_inputs(c: *mut RlxCompiled, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let rows = borrow(c, "compiled")?.0.inputs();
        out::json(out, &rows, "rlx_compiled_inputs out")
    })
}

/// Declared parameters, as [`rlx_compiled_inputs`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_params(c: *mut RlxCompiled, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let rows = borrow(c, "compiled")?.0.params();
        out::json(out, &rows, "rlx_compiled_params out")
    })
}

/// Outputs in `run` order, as [`rlx_compiled_inputs`]; `name` is `""` for
/// an output the graph does not name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_outputs(c: *mut RlxCompiled, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let rows = borrow(c, "compiled")?.0.outputs();
        out::json(out, &rows, "rlx_compiled_outputs out")
    })
}

/// The position in `run`'s result of the output named `name`;
/// `RLX_ERR_RANGE` naming the closest names when there is none.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_output_index(
    c: *mut RlxCompiled,
    name: *const c_char,
    out: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_compiled_output_index name")?;
        let i = borrow(c, "compiled")?.0.output_index(name)?;
        out::put(out, i, "rlx_compiled_output_index out")
    })
}

/// `RLX_ERR_TYPE` naming the inputs `supplied` leaves unbound and the params
/// never set; `RLX_OK` when everything is bound.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_check_bindings(
    c: *mut RlxCompiled,
    supplied: *const *const c_char,
    n: usize,
) -> i32 {
    guard(|| unsafe {
        let names = arg::strs(supplied, n, "rlx_compiled_check_bindings supplied")?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        borrow(c, "compiled")?.0.check_bindings(&refs)
    })
}

/// Declare inputs / params meant to read zeros (an empty KV state).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_allow_unbound(
    c: *mut RlxCompiled,
    names: *const *const c_char,
    n: usize,
) -> i32 {
    guard(|| unsafe {
        let names = arg::strs(names, n, "rlx_compiled_allow_unbound names")?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        borrow(c, "compiled")?.0.allow_unbound(&refs);
        Ok(())
    })
}

/// The compiled graph's device label (free with `rlx_string_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_compiled_device(c: *mut RlxCompiled, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let label = borrow(c, "compiled")?.0.device_label();
        out::string(out, label, "rlx_compiled_device out")
    })
}

// ── outputs ─────────────────────────────────────────────────

/// How many outputs a run produced (`0` for `NULL`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_outputs_len(o: *const RlxOutputs) -> usize {
    unsafe { o.as_ref() }.map_or(0, |o| o.outs.len())
}

/// Borrow output `i`: `*data` stays valid until `rlx_outputs_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_outputs_get(
    o: *const RlxOutputs,
    i: usize,
    data: *mut *const f32,
    len: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let o = o.as_ref().ok_or_else(|| crate::ffi::null("outputs"))?;
        let buf = o
            .outs
            .get(i)
            .ok_or_else(|| rlx_bind::ApiError::range(format!("output {i} of {}", o.outs.len())))?;
        out::put(data, buf.as_ptr(), "rlx_outputs_get data")?;
        out::put(len, buf.len(), "rlx_outputs_get len")
    })
}

/// Free run outputs (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_outputs_free(o: *mut RlxOutputs) {
    unsafe { release(o) }
}

/// How many outputs a typed run produced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_typed_outputs_len(o: *const RlxTypedOutputs) -> usize {
    unsafe { o.as_ref() }.map_or(0, |o| o.outs.len())
}

/// Borrow typed output `i`: bytes and dtype name stay valid until
/// `rlx_typed_outputs_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_typed_outputs_get(
    o: *const RlxTypedOutputs,
    i: usize,
    data: *mut *const u8,
    len: *mut usize,
    dtype: *mut *const c_char,
) -> i32 {
    guard(|| unsafe {
        let o = o
            .as_ref()
            .ok_or_else(|| crate::ffi::null("typed outputs"))?;
        let (buf, _) = o
            .outs
            .get(i)
            .ok_or_else(|| rlx_bind::ApiError::range(format!("output {i} of {}", o.outs.len())))?;
        out::put(data, buf.as_ptr(), "rlx_typed_outputs_get data")?;
        out::put(len, buf.len(), "rlx_typed_outputs_get len")?;
        out::put(dtype, o.labels[i].as_ptr(), "rlx_typed_outputs_get dtype")
    })
}

/// Free typed run outputs (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_typed_outputs_free(o: *mut RlxTypedOutputs) {
    unsafe { release(o) }
}

// ── autodiff ────────────────────────────────────────────────

fn new_graph(g: rlx_ir::Graph) -> *mut RlxGraph {
    handle(RlxGraph { inner: Some(g) })
}

/// `[loss, …d(loss)/d(wrt)]`; the result gains a `d_output` input (seed `[1]`).
/// The forward graph is borrowed, not consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_grad(
    g: *mut RlxGraph,
    wrt: *const u32,
    n: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let wrt = arg::vec(wrt, n, "rlx_grad wrt")?;
        let r = autodiff::grad(graph_mut(g)?, wrt)?;
        out::put(out, new_graph(r), "rlx_grad out")
    })
}

/// `[…primals, …tangents]`, one `tangent_<name>` input per seeded node.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_jvp(
    g: *mut RlxGraph,
    tangent_for: *const u32,
    n: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let ids = arg::vec(tangent_for, n, "rlx_jvp tangent_for")?;
        let r = autodiff::jvp(graph_mut(g)?, ids)?;
        out::put(out, new_graph(r), "rlx_jvp out")
    })
}

/// Hessian-vector product, forward-over-reverse.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_hvp(
    g: *mut RlxGraph,
    wrt: *const u32,
    n: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let wrt = arg::vec(wrt, n, "rlx_hvp wrt")?;
        let r = autodiff::hvp(graph_mut(g)?, wrt)?;
        out::put(out, new_graph(r), "rlx_hvp out")
    })
}

/// Batch the `n` named inputs along a new leading axis of `batch_size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_vmap(
    g: *mut RlxGraph,
    names: *const *const c_char,
    n: usize,
    batch_size: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let names = arg::strs(names, n, "rlx_vmap names")?;
        let r = autodiff::vmap(graph_mut(g)?, names, batch_size)?;
        out::put(out, new_graph(r), "rlx_vmap out")
    })
}

/// The `order`-th derivative w.r.t. the scalar input `wrt_name`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_nth_order_grad(
    g: *mut RlxGraph,
    wrt_name: *const c_char,
    order: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(wrt_name, "rlx_nth_order_grad wrt_name")?;
        let r = autodiff::nth_order_grad(graph_mut(g)?, name, order)?;
        out::put(out, new_graph(r), "rlx_nth_order_grad out")
    })
}

/// Directional order-`order` derivative; the result gains `dir_0` … inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_directional_nth_grad(
    g: *mut RlxGraph,
    wrt_name: *const c_char,
    order: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(wrt_name, "rlx_directional_nth_grad wrt_name")?;
        let r = autodiff::directional_nth_grad(graph_mut(g)?, name, order)?;
        out::put(out, new_graph(r), "rlx_directional_nth_grad out")
    })
}

// ── check / devices ─────────────────────────────────────────

/// Device-free static diagnostics as a JSON report. `options_json` (`NULL` =
/// all passes, all backends): `{"backends": ["cpu", "cuda"], "numeric": false, …}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_check(
    g: *mut RlxGraph,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_check options")?;
        let report = check::check(graph_mut(g)?, None, &opts)?;
        out::json(out, &report, "rlx_check out")
    })
}

/// Every backend `rlx_check` knows, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_check_backends(out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names: Vec<String> = check::backends().into_iter().map(str::to_string).collect();
        out::strings(out, &names, "rlx_check_backends out")
    })
}

/// The devices in this build, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_devices(out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names: Vec<String> = devices::available()
            .into_iter()
            .map(str::to_string)
            .collect();
        out::strings(out, &names, "rlx_devices out")
    })
}

/// Whether `name` is a device this build can run (unknown names: `false`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_is_available(name: *const c_char) -> bool {
    unsafe { arg::str(name, "rlx_is_available name") }.is_ok_and(devices::is_available)
}

/// The canonical label of a device name (`"gpu"` → `"wgpu"`); an unknown
/// name is an error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_parse_device(name: *const c_char, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_parse_device name")?;
        let label = devices::label(devices::parse(name)?);
        out::string(out, label, "rlx_parse_device out")
    })
}

/// The backends manifest (features, kernels, capabilities) as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_backends_manifest(out: *mut *mut c_char) -> i32 {
    guard(|| unsafe { out::string(out, &devices::manifest_json(), "rlx_backends_manifest out") })
}

/// The device the cost model expects to run `g` fastest. `policy_json`
/// (`NULL` = all): `"env"`, or `{"only": […], "deny": […], "prefer": […]}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_fastest_device_for(
    g: *mut RlxGraph,
    policy_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let p = arg::json(policy_json, "rlx_fastest_device_for policy")?;
        let policy = (!p.is_null()).then(|| devices::policy(&p)).transpose()?;
        let label = devices::fastest_for(graph_mut(g)?, policy.as_ref());
        out::string(out, label, "rlx_fastest_device_for out")
    })
}

/// One JSON row per candidate device for `g`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_device_report(
    g: *mut RlxGraph,
    policy_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let policy = devices::policy(&arg::json(policy_json, "rlx_device_report policy")?)?;
        let report = devices::report(graph_mut(g)?, &policy);
        out::json(out, &report, "rlx_device_report out")
    })
}

// ── Runner / Router ─────────────────────────────────────────

/// One graph compiled lazily per device (`GraphDevices`).
pub struct RlxRunner(Runner);

/// A graph routed to a device per run, with fallback (`DeviceRouter`).
pub struct RlxRouter(Router);

/// A runner over `graph` (consumed); `policy_json` as for `rlx_device_report`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_new(
    graph: *mut RlxGraph,
    policy_json: *const c_char,
    out: *mut *mut RlxRunner,
) -> i32 {
    guard(|| unsafe {
        let policy = devices::policy(&arg::json(policy_json, "rlx_runner_new policy")?)?;
        let g = take_graph(graph, "rlx_runner_new graph")?;
        out::put(
            out,
            handle(RlxRunner(Runner::new(g, policy))),
            "rlx_runner_new out",
        )
    })
}

/// A runner over `graph` (consumed) with the device policy from `RLX_DEVICE*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_from_env(
    graph: *mut RlxGraph,
    out: *mut *mut RlxRunner,
) -> i32 {
    guard(|| unsafe {
        let g = take_graph(graph, "rlx_runner_from_env graph")?;
        out::put(
            out,
            handle(RlxRunner(Runner::from_env(g))),
            "rlx_runner_from_env out",
        )
    })
}

/// Free a runner (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_free(r: *mut RlxRunner) {
    unsafe { release(r) }
}

/// The fastest device by static rank (free with `rlx_string_free`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_fastest(r: *mut RlxRunner, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let label = borrow(r, "runner")?.0.fastest();
        out::string(out, label, "rlx_runner_fastest out")
    })
}

/// The device a run with `hint` (`NULL` = none) would use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_resolve(
    r: *mut RlxRunner,
    hint: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let hint = arg::opt_str(hint, "rlx_runner_resolve hint")?;
        let label = borrow(r, "runner")?.0.resolve(hint.as_deref())?;
        out::string(out, label, "rlx_runner_resolve out")
    })
}

/// Set `n` f32 parameters on every device's compile; writes how many.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_set_params(
    r: *mut RlxRunner,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    applied: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let params = arg::inputs(names, data, lens, n, "rlx_runner_set_params params")?;
        let count = borrow(r, "runner")?.0.set_params(&params);
        out::put(applied, count, "rlx_runner_set_params applied")
    })
}

/// Set one parameter from raw bytes of `dtype` on every device's compile.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_set_param_typed(
    r: *mut RlxRunner,
    name: *const c_char,
    data: *const u8,
    len: usize,
    dtype: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_runner_set_param_typed name")?;
        let data = arg::slice(data, len, "rlx_runner_set_param_typed data")?;
        let dtype = arg::str(dtype, "rlx_runner_set_param_typed dtype")?;
        borrow(r, "runner")?.0.set_param_typed(name, data, dtype)
    })
}

/// Run on exactly `device`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_run_on(
    r: *mut RlxRunner,
    device: *const c_char,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let device = arg::str(device, "rlx_runner_run_on device")?;
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_run_on inputs")?;
        let outs = borrow(r, "runner")?.0.run_on(device, &inputs)?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_runner_run_on out")
    })
}

/// Run on `hint` (`NULL` = the policy's pick), falling back down the policy
/// when a device fails; writes the device used.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_run_chain(
    r: *mut RlxRunner,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    hint: *const c_char,
    device_out: *mut *mut c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_run_chain inputs")?;
        let hint = arg::opt_str(hint, "rlx_runner_run_chain hint")?;
        let (device, outs) = borrow(r, "runner")?.0.run_chain(&inputs, hint.as_deref())?;
        out::string(device_out, device, "rlx_runner_run_chain device_out")?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_runner_run_chain out")
    })
}

/// Resolve with the inputs in hand (input-shape-aware), then run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_run_with_inputs(
    r: *mut RlxRunner,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    hint: *const c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_run_with_inputs inputs")?;
        let hint = arg::opt_str(hint, "rlx_runner_run_with_inputs hint")?;
        let outs = borrow(r, "runner")?
            .0
            .run_with_inputs(&inputs, hint.as_deref())?;
        out::put(
            out,
            handle(RlxOutputs { outs }),
            "rlx_runner_run_with_inputs out",
        )
    })
}

/// The devices this runner may use, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_devices(r: *mut RlxRunner, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names: Vec<String> = borrow(r, "runner")?
            .0
            .devices()
            .into_iter()
            .map(str::to_string)
            .collect();
        out::strings(out, &names, "rlx_runner_devices out")
    })
}

/// Set one f32 parameter on every device's compile.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_set_param(
    r: *mut RlxRunner,
    name: *const c_char,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_runner_set_param name")?;
        let data = arg::slice(data, len, "rlx_runner_set_param data")?;
        borrow(r, "runner")?.0.set_param(name, data);
        Ok(())
    })
}

/// Compile every viable device now; the devices as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_warm_all(r: *mut RlxRunner, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names: Vec<String> = borrow(r, "runner")?
            .0
            .warm_all()?
            .into_iter()
            .map(str::to_string)
            .collect();
        out::strings(out, &names, "rlx_runner_warm_all out")
    })
}

/// Time each device on the inputs → JSON `[{device, compile_ms, exec_ms}]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_benchmark(
    r: *mut RlxRunner,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    runs: usize,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_benchmark inputs")?;
        let report = borrow(r, "runner")?.0.benchmark(&inputs, runs)?;
        out::json(out, &report, "rlx_runner_benchmark out")
    })
}

/// Run on `hint` (`NULL` = the policy's pick); writes the device used.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_run(
    r: *mut RlxRunner,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    hint: *const c_char,
    device_out: *mut *mut c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_run inputs")?;
        let hint = arg::opt_str(hint, "rlx_runner_run hint")?;
        let (device, outs) = borrow(r, "runner")?.0.run(&inputs, hint.as_deref())?;
        out::string(device_out, device, "rlx_runner_run device_out")?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_runner_run out")
    })
}

/// Try each of the `chain_len` devices in order until one runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_run_try(
    r: *mut RlxRunner,
    chain: *const *const c_char,
    chain_len: usize,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    device_out: *mut *mut c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let chain = arg::strs(chain, chain_len, "rlx_runner_run_try chain")?;
        let inputs = arg::inputs(names, data, lens, n, "rlx_runner_run_try inputs")?;
        let (device, outs) = borrow(r, "runner")?.0.run_try(&chain, &inputs)?;
        out::string(device_out, device, "rlx_runner_run_try device_out")?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_runner_run_try out")
    })
}

/// Candidates for this runner's graph, as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_runner_report(r: *mut RlxRunner, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let report = borrow(r, "runner")?.0.report();
        out::json(out, &report, "rlx_runner_report out")
    })
}

/// A router over `graph` (consumed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_new(
    graph: *mut RlxGraph,
    policy_json: *const c_char,
    out: *mut *mut RlxRouter,
) -> i32 {
    guard(|| unsafe {
        let policy = devices::policy(&arg::json(policy_json, "rlx_router_new policy")?)?;
        let g = take_graph(graph, "rlx_router_new graph")?;
        out::put(
            out,
            handle(RlxRouter(Router::new(g, policy)?)),
            "rlx_router_new out",
        )
    })
}

/// A router over `graph` (consumed) with the device policy from `RLX_DEVICE*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_from_env(
    graph: *mut RlxGraph,
    out: *mut *mut RlxRouter,
) -> i32 {
    guard(|| unsafe {
        let g = take_graph(graph, "rlx_router_from_env graph")?;
        out::put(
            out,
            handle(RlxRouter(Router::from_env(g)?)),
            "rlx_router_from_env out",
        )
    })
}

/// Free a router (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_free(r: *mut RlxRouter) {
    unsafe { release(r) }
}

/// The devices this router may use, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_devices(r: *mut RlxRouter, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names: Vec<String> = borrow(r, "router")?
            .0
            .devices()
            .into_iter()
            .map(str::to_string)
            .collect();
        out::strings(out, &names, "rlx_router_devices out")
    })
}

/// Set `n` f32 parameters; writes how many.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_set_params(
    r: *mut RlxRouter,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    applied: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let params = arg::inputs(names, data, lens, n, "rlx_router_set_params params")?;
        let count = borrow(r, "router")?.0.set_params(&params);
        out::put(applied, count, "rlx_router_set_params applied")
    })
}

/// Set one parameter from raw bytes of `dtype`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_set_param_typed(
    r: *mut RlxRouter,
    name: *const c_char,
    data: *const u8,
    len: usize,
    dtype: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_router_set_param_typed name")?;
        let data = arg::slice(data, len, "rlx_router_set_param_typed data")?;
        let dtype = arg::str(dtype, "rlx_router_set_param_typed dtype")?;
        borrow(r, "router")?.0.set_param_typed(name, data, dtype)
    })
}

/// Re-benchmark when the machine reports thermal throttling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_set_rebench_on_throttle(
    r: *mut RlxRouter,
    enabled: bool,
) -> i32 {
    guard(|| unsafe {
        borrow(r, "router")?.0.set_rebench_on_throttle(enabled);
        Ok(())
    })
}

/// Run on exactly `device`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_run_on(
    r: *mut RlxRouter,
    device: *const c_char,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let device = arg::str(device, "rlx_router_run_on device")?;
        let inputs = arg::inputs(names, data, lens, n, "rlx_router_run_on inputs")?;
        let outs = borrow(r, "router")?.0.run_on(device, &inputs)?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_router_run_on out")
    })
}

/// Run, falling back down the policy's chain when a device fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_run_chain(
    r: *mut RlxRouter,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    hint: *const c_char,
    device_out: *mut *mut c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_router_run_chain inputs")?;
        let hint = arg::opt_str(hint, "rlx_router_run_chain hint")?;
        let (device, outs) = borrow(r, "router")?.0.run_chain(&inputs, hint.as_deref())?;
        out::string(device_out, device, "rlx_router_run_chain device_out")?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_router_run_chain out")
    })
}

/// Set one f32 parameter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_set_param(
    r: *mut RlxRouter,
    name: *const c_char,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let name = arg::str(name, "rlx_router_set_param name")?;
        let data = arg::slice(data, len, "rlx_router_set_param data")?;
        borrow(r, "router")?.0.set_param(name, data);
        Ok(())
    })
}

/// Run on the policy's device (or `hint`, `NULL` = none); writes the device used.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_router_run(
    r: *mut RlxRouter,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    hint: *const c_char,
    device_out: *mut *mut c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_router_run inputs")?;
        let hint = arg::opt_str(hint, "rlx_router_run hint")?;
        let (device, outs) = borrow(r, "router")?.0.run(&inputs, hint.as_deref())?;
        out::string(device_out, device, "rlx_router_run device_out")?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_router_run out")
    })
}

// ── FPGA export ─────────────────────────────────────────────

/// Export `g` (borrowed) as SystemVerilog under `out_dir`; the files written as
/// a JSON array. `options_json`: `{quant, hw, in_iface, out_iface, bind_in,
/// bind_out, sideband, bind_sideband}` (feature `fpga`).
#[cfg(feature = "fpga")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_export_fpga(
    g: *mut RlxGraph,
    out_dir: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let dir = arg::str(out_dir, "rlx_export_fpga out_dir")?;
        let opts = arg::json(options_json, "rlx_export_fpga options")?;
        let files = rlx_bind::runtime::fpga::export(graph_mut(g)?, dir, &opts)?;
        out::strings(out, &files, "rlx_export_fpga out")
    })
}
