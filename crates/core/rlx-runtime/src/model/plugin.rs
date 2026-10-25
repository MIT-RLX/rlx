// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model provider plugins: a shared library whose providers plug into a
//! process that did not link them — `librlx_capi`, and through it every
//! language binding.
//!
//! **Writing one** (a `cdylib` + `rlib` crate): list its providers once:
//!
//! ```ignore
//! rlx_runtime::export_model_plugin!("my-models", [MY_PROVIDER]);
//! ```
//!
//! **Loading one**: a library in [`home_plugin_dir`] or
//! `RLX_MODEL_PLUGINS=<path>[:<path>…]` (`;` on Windows), at the first model
//! call, or [`load_plugin`] (feature `model-plugins`).
//!
//! **Linking one**, where nothing loads at run time (iOS, Android, wasm, a
//! static library): the host depends on the plugin crate and names it with
//! [`link_model_plugins!`](crate::link_model_plugins) (feature
//! `linked-plugins`).
//!
//! The boundary is C: [`PluginV1`] is a table of `extern "C"` functions over
//! opaque handles, JSON strings and [`PluginTensor`] views, versioned by
//! [`PLUGIN_ABI`]. Nothing Rust-specific crosses it — the plugin and the host
//! each have their own copy of this crate (and may be built by different
//! compilers), so no trait object, allocation or panic does either: every
//! buffer is freed by the side that allocated it, and every exported function
//! catches panics.

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use anyhow::{Result, anyhow};
use serde_json::Value;

use super::{Model, ModelInfo, ModelRequest, Tensor, dtype_name, parse_dtype};

/// The plugin ABI version: [`PluginV1::abi`] must equal it.
pub const PLUGIN_ABI: u32 = 1;

/// The symbol a plugin exports: `extern "C" fn() -> *const PluginV1`.
pub const PLUGIN_SYMBOL: &[u8] = b"rlx_model_plugin_v1";

/// Providers built into rlx-runtime: a plugin does not re-export its copy.
const BUILTIN: &[&str] = &["rlxp"];

/// A borrowed tensor at the boundary.
#[repr(C)]
pub struct PluginTensor {
    pub name: *const c_char,
    /// `"f32"`, `"i32"`, … ([`dtype_name`]).
    pub dtype: *const c_char,
    pub shape: *const usize,
    pub ndim: usize,
    pub data: *const u8,
    pub len: usize,
}

/// The functions a plugin exports. Every `i32` is 0 on success; on failure
/// `*err` is a message the caller frees with `free_string`.
#[repr(C)]
pub struct PluginV1 {
    pub abi: u32,
    /// The plugin's name.
    pub name: *const c_char,
    /// `*out` = JSON `[{name, description, tasks, families}]` of its providers.
    pub providers: unsafe extern "C" fn(out: *mut *mut c_char, err: *mut *mut c_char) -> i32,
    /// `*score` = how specifically `provider` claims `(source, options)`.
    pub probe: unsafe extern "C" fn(
        provider: *const c_char,
        source: *const c_char,
        options_json: *const c_char,
        score: *mut u32,
        err: *mut *mut c_char,
    ) -> i32,
    /// Open a model; `*model` is freed with `free_model`.
    pub open: unsafe extern "C" fn(
        provider: *const c_char,
        source: *const c_char,
        options_json: *const c_char,
        model: *mut *mut c_void,
        err: *mut *mut c_char,
    ) -> i32,
    /// `*out` = the model's info JSON ([`ModelInfo::to_json`]).
    pub info: unsafe extern "C" fn(
        model: *mut c_void,
        out: *mut *mut c_char,
        err: *mut *mut c_char,
    ) -> i32,
    /// Run; `*outputs` is freed with `free_outputs`, read with
    /// `outputs_len` / `output`.
    pub run: unsafe extern "C" fn(
        model: *mut c_void,
        inputs: *const PluginTensor,
        n: usize,
        options_json: *const c_char,
        outputs: *mut *mut c_void,
        err: *mut *mut c_char,
    ) -> i32,
    pub outputs_len: unsafe extern "C" fn(outputs: *mut c_void) -> usize,
    /// A view of output `i`, valid until `free_outputs`.
    pub output: unsafe extern "C" fn(outputs: *mut c_void, i: usize, out: *mut PluginTensor),
    /// `*out` = generated text.
    pub generate_text: unsafe extern "C" fn(
        model: *mut c_void,
        prompt: *const c_char,
        options_json: *const c_char,
        out: *mut *mut c_char,
        err: *mut *mut c_char,
    ) -> i32,
    pub free_outputs: unsafe extern "C" fn(outputs: *mut c_void),
    pub free_model: unsafe extern "C" fn(model: *mut c_void),
    pub free_string: unsafe extern "C" fn(s: *mut c_char),
}

// SAFETY: the table is immutable and its functions are thread-safe entry
// points; the name points at a static string.
unsafe impl Send for PluginV1 {}
unsafe impl Sync for PluginV1 {}

// ── the plugin side ─────────────────────────────────────────

fn cstring(s: impl Into<String>) -> *mut c_char {
    let s = s.into().replace('\0', " ");
    CString::new(s).unwrap_or_default().into_raw()
}

unsafe fn str_of<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
}

unsafe fn json_of(p: *const c_char) -> Value {
    let s = unsafe { str_of(p) };
    if s.is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(s).unwrap_or(Value::Object(Default::default()))
    }
}

/// Run `f`; a failure or panic becomes `*err` and -1.
fn exported(err: *mut *mut c_char, f: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            if !err.is_null() {
                unsafe { *err = cstring(format!("{e:#}")) };
            }
            -1
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".into());
            if !err.is_null() {
                unsafe { *err = cstring(format!("plugin panicked: {msg}")) };
            }
            -4
        }
    }
}

fn own_providers() -> Vec<&'static dyn super::ModelProvider> {
    inventory::iter::<super::ProviderRegistration>
        .into_iter()
        .map(|r| r.0)
        .filter(|p| !BUILTIN.contains(&p.name()))
        .collect()
}

fn own_provider(name: &str) -> Result<&'static dyn super::ModelProvider> {
    own_providers()
        .into_iter()
        .find(|p| p.name() == name)
        .ok_or_else(|| anyhow!("this plugin has no provider '{name}'"))
}

unsafe extern "C" fn p_providers(out: *mut *mut c_char, err: *mut *mut c_char) -> i32 {
    exported(err, || {
        let list: Vec<Value> = own_providers()
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name(),
                    "description": p.description(),
                    "tasks": p.tasks(),
                    "families": p.families(),
                })
            })
            .collect();
        unsafe { *out = cstring(Value::Array(list).to_string()) };
        Ok(())
    })
}

unsafe fn request_of(source: *const c_char, options: *const c_char) -> Result<ModelRequest> {
    ModelRequest::new(unsafe { str_of(source) }, unsafe { json_of(options) })
}

unsafe extern "C" fn p_probe(
    provider: *const c_char,
    source: *const c_char,
    options: *const c_char,
    score: *mut u32,
    err: *mut *mut c_char,
) -> i32 {
    exported(err, || {
        let p = own_provider(unsafe { str_of(provider) })?;
        let req = unsafe { request_of(source, options) }?;
        unsafe { *score = p.probe(&req) };
        Ok(())
    })
}

unsafe extern "C" fn p_open(
    provider: *const c_char,
    source: *const c_char,
    options: *const c_char,
    model: *mut *mut c_void,
    err: *mut *mut c_char,
) -> i32 {
    exported(err, || {
        let p = own_provider(unsafe { str_of(provider) })?;
        let req = unsafe { request_of(source, options) }?;
        let m: Box<Box<dyn Model>> = Box::new(p.open(&req)?);
        unsafe { *model = Box::into_raw(m).cast() };
        Ok(())
    })
}

unsafe fn model_of<'a>(m: *mut c_void) -> &'a mut Box<dyn Model> {
    unsafe { &mut *m.cast::<Box<dyn Model>>() }
}

unsafe extern "C" fn p_info(m: *mut c_void, out: *mut *mut c_char, err: *mut *mut c_char) -> i32 {
    exported(err, || {
        let info = unsafe { model_of(m) }.info();
        unsafe { *out = cstring(info.to_json().to_string()) };
        Ok(())
    })
}

/// Outputs owned by the plugin, with C strings for the views.
struct Outputs {
    tensors: Vec<Tensor>,
    names: Vec<CString>,
    dtypes: Vec<CString>,
}

unsafe fn tensors_in(inputs: *const PluginTensor, n: usize) -> Result<Vec<Tensor>> {
    let views = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(inputs, n) }
    };
    views
        .iter()
        .map(|v| unsafe {
            let shape = if v.ndim == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(v.shape, v.ndim).to_vec()
            };
            let data = if v.len == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(v.data, v.len).to_vec()
            };
            Ok(Tensor::new(
                str_of(v.name),
                parse_dtype(str_of(v.dtype))?,
                shape,
                data,
            ))
        })
        .collect()
}

unsafe extern "C" fn p_run(
    m: *mut c_void,
    inputs: *const PluginTensor,
    n: usize,
    options: *const c_char,
    outputs: *mut *mut c_void,
    err: *mut *mut c_char,
) -> i32 {
    exported(err, || {
        let ins = unsafe { tensors_in(inputs, n) }?;
        let tensors = unsafe { model_of(m) }.run(&ins, &unsafe { json_of(options) })?;
        let names = tensors
            .iter()
            .map(|t| CString::new(t.name.clone()).unwrap_or_default())
            .collect();
        let dtypes = tensors
            .iter()
            .map(|t| CString::new(dtype_name(t.dtype)).unwrap_or_default())
            .collect();
        let o = Box::new(Outputs {
            tensors,
            names,
            dtypes,
        });
        unsafe { *outputs = Box::into_raw(o).cast() };
        Ok(())
    })
}

unsafe extern "C" fn p_outputs_len(o: *mut c_void) -> usize {
    unsafe { &*o.cast::<Outputs>() }.tensors.len()
}

unsafe extern "C" fn p_output(o: *mut c_void, i: usize, out: *mut PluginTensor) {
    let o = unsafe { &*o.cast::<Outputs>() };
    let t = &o.tensors[i];
    unsafe {
        *out = PluginTensor {
            name: o.names[i].as_ptr(),
            dtype: o.dtypes[i].as_ptr(),
            shape: t.shape.as_ptr(),
            ndim: t.shape.len(),
            data: t.data.as_ptr(),
            len: t.data.len(),
        }
    };
}

unsafe extern "C" fn p_generate_text(
    m: *mut c_void,
    prompt: *const c_char,
    options: *const c_char,
    out: *mut *mut c_char,
    err: *mut *mut c_char,
) -> i32 {
    exported(err, || {
        let text = unsafe { model_of(m) }
            .generate_text(unsafe { str_of(prompt) }, &unsafe { json_of(options) })?;
        unsafe { *out = cstring(text) };
        Ok(())
    })
}

unsafe extern "C" fn p_free_outputs(o: *mut c_void) {
    if !o.is_null() {
        drop(unsafe { Box::from_raw(o.cast::<Outputs>()) });
    }
}

unsafe extern "C" fn p_free_model(m: *mut c_void) {
    if !m.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(m.cast::<Box<dyn Model>>()) })
        }));
    }
}

unsafe extern "C" fn p_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// The function table for [`export_model_plugin!`](crate::export_model_plugin).
#[doc(hidden)]
pub const fn vtable(name: &'static CStr) -> PluginV1 {
    PluginV1 {
        abi: PLUGIN_ABI,
        name: name.as_ptr(),
        providers: p_providers,
        probe: p_probe,
        open: p_open,
        info: p_info,
        run: p_run,
        outputs_len: p_outputs_len,
        output: p_output,
        generate_text: p_generate_text,
        free_outputs: p_free_outputs,
        free_model: p_free_model,
        free_string: p_free_string,
    }
}

/// Export this library's providers as a model plugin:
///
/// ```ignore
/// rlx_runtime::export_model_plugin!("my-models", [WHISPER, EMBED]);
/// ```
///
/// One crate then works both ways. Built as a `cdylib`, it is a plugin that
/// [`load_plugin`] loads. As a dependency of a host that cannot load
/// libraries (iOS, wasm, a static library), it is linked in by the host's
/// [`link_model_plugins!`](crate::link_model_plugins). The listed providers
/// are also registered as [`register_provider!`](crate::register_provider)
/// would. An entry may carry `#[cfg(…)]`, for a provider behind a feature.
/// `export_model_plugin!("my-models")`, without a list, exports what
/// `register_provider!` registered, and does not link.
#[macro_export]
macro_rules! export_model_plugin {
    ($name:literal) => {
        $crate::export_model_plugin!($name, []);
    };
    ($name:literal, [$($(#[$attr:meta])* $provider:path),* $(,)?]) => {
        $( $(#[$attr])* $crate::register_provider!($provider); )*
        $crate::__model_plugin_table!($name);

        /// This plugin's providers, for [`link_model_plugins!`].
        #[doc(hidden)]
        pub fn __rlx_model_providers() -> ::std::vec::Vec<&'static dyn $crate::model::ModelProvider> {
            #[allow(unused_mut)]
            let mut all: ::std::vec::Vec<&'static dyn $crate::model::ModelProvider> =
                ::std::vec::Vec::new();
            $( $(#[$attr])* all.push(&$provider); )*
            all
        }
    };
}

/// The C table of a plugin library. A linked build leaves it out: every
/// plugin's would be the same symbol.
#[cfg(not(feature = "linked-plugins"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __model_plugin_table {
    ($name:literal) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn rlx_model_plugin_v1() -> *const $crate::model::plugin::PluginV1 {
            static TABLE: $crate::model::plugin::PluginV1 = $crate::model::plugin::vtable(
                match ::core::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()) {
                    Ok(s) => s,
                    Err(_) => panic!("plugin name"),
                },
            );
            &TABLE
        }
    };
}

#[cfg(feature = "linked-plugins")]
#[doc(hidden)]
#[macro_export]
macro_rules! __model_plugin_table {
    ($name:literal) => {};
}

/// Link plugin crates into this build, for a host that cannot load a
/// library at run time: an iOS app, Android, wasm, or a static
/// `librlx_capi.a` with a collection's models built in. The host crate
/// depends on each plugin crate and on rlx-runtime with `linked-plugins`,
/// then names the plugins once:
///
/// ```ignore
/// pub use rlx_capi::*;                       // a static C library, say
/// rlx_runtime::link_model_plugins!(rlx_models_plugin, my_models);
/// ```
///
/// This defines `int32_t rlx_link_model_plugins(void)`, which registers
/// every listed provider and returns how many. The app calls it once, before
/// its first model call. The call is what keeps the plugins in the binary: a
/// C linker drops a static library's objects that nothing references,
/// start-up registrations included.
#[cfg(feature = "linked-plugins")]
#[macro_export]
macro_rules! link_model_plugins {
    ($($plugin:ident),* $(,)?) => {
        /// Register the model plugins linked into this library.
        #[unsafe(no_mangle)]
        pub extern "C" fn rlx_link_model_plugins() -> i32 {
            let mut all = ::std::vec::Vec::new();
            $( all.extend($plugin::__rlx_model_providers()); )*
            $crate::model::plugin::link_providers(all) as i32
        }
    };
}

/// Without `linked-plugins`, naming plugins would link nothing: say so.
#[cfg(not(feature = "linked-plugins"))]
#[macro_export]
macro_rules! link_model_plugins {
    ($($plugin:ident),* $(,)?) => {
        ::core::compile_error!("link_model_plugins! needs rlx-runtime's `linked-plugins` feature");
    };
}

static LINKED: std::sync::RwLock<Vec<&'static dyn super::ModelProvider>> =
    std::sync::RwLock::new(Vec::new());

/// Register providers linked into this build (what
/// [`link_model_plugins!`](crate::link_model_plugins) calls); returns how
/// many are linked. A provider already linked is not added twice.
pub fn link_providers(providers: Vec<&'static dyn super::ModelProvider>) -> usize {
    let mut all = LINKED.write().unwrap_or_else(|e| e.into_inner());
    for p in providers {
        if !all.iter().any(|q| q.name() == p.name()) {
            all.push(p);
        }
    }
    all.len()
}

pub(super) fn linked_providers() -> Vec<&'static dyn super::ModelProvider> {
    LINKED.read().unwrap_or_else(|e| e.into_inner()).clone()
}

// ── the host side ───────────────────────────────────────────

/// A provider in a loaded plugin.
struct PluginProvider {
    table: &'static PluginV1,
    name: String,
    c_name: CString,
    description: String,
    tasks: Vec<String>,
    families: Vec<String>,
}

/// A model opened by a plugin.
struct PluginModel {
    table: &'static PluginV1,
    handle: *mut c_void,
}

// SAFETY: a plugin's models are `Send` (the `Model` contract on its side);
// the handle is only used through `&mut self`.
unsafe impl Send for PluginModel {}

unsafe fn take_string(table: &PluginV1, s: *mut c_char) -> String {
    if s.is_null() {
        return String::new();
    }
    let out = unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned();
    unsafe { (table.free_string)(s) };
    out
}

fn call(table: &PluginV1, what: &str, f: impl FnOnce(*mut *mut c_char) -> i32) -> Result<()> {
    let mut err: *mut c_char = ptr::null_mut();
    if f(&mut err) == 0 {
        return Ok(());
    }
    let msg = unsafe { take_string(table, err) };
    Err(anyhow!("{what}: {msg}"))
}

fn cstr(s: &str) -> CString {
    CString::new(s.replace('\0', " ")).unwrap_or_default()
}

impl super::ModelProvider for PluginProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn tasks(&self) -> Vec<String> {
        self.tasks.clone()
    }

    fn families(&self) -> Vec<String> {
        self.families.clone()
    }

    fn probe(&self, request: &ModelRequest) -> u32 {
        let (source, options) = (cstr(&request.source), cstr(&request.options.to_string()));
        let mut score = 0u32;
        let ok = call(self.table, "probe", |err| unsafe {
            (self.table.probe)(
                self.c_name.as_ptr(),
                source.as_ptr(),
                options.as_ptr(),
                &mut score,
                err,
            )
        });
        if ok.is_ok() { score } else { 0 }
    }

    fn open(&self, request: &ModelRequest) -> Result<Box<dyn Model>> {
        let (source, options) = (cstr(&request.source), cstr(&request.options.to_string()));
        let mut handle: *mut c_void = ptr::null_mut();
        call(self.table, &format!("{} open", self.name), |err| unsafe {
            (self.table.open)(
                self.c_name.as_ptr(),
                source.as_ptr(),
                options.as_ptr(),
                &mut handle,
                err,
            )
        })?;
        Ok(Box::new(PluginModel {
            table: self.table,
            handle,
        }))
    }
}

impl Model for PluginModel {
    fn info(&self) -> ModelInfo {
        let mut out: *mut c_char = ptr::null_mut();
        let ok = call(self.table, "info", |err| unsafe {
            (self.table.info)(self.handle, &mut out, err)
        });
        if ok.is_err() {
            return ModelInfo::default();
        }
        let s = unsafe { take_string(self.table, out) };
        serde_json::from_str(&s)
            .ok()
            .and_then(|v| ModelInfo::from_json(&v).ok())
            .unwrap_or_default()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let names: Vec<CString> = inputs.iter().map(|t| cstr(&t.name)).collect();
        let dtypes: Vec<CString> = inputs.iter().map(|t| cstr(dtype_name(t.dtype))).collect();
        let views: Vec<PluginTensor> = inputs
            .iter()
            .enumerate()
            .map(|(i, t)| PluginTensor {
                name: names[i].as_ptr(),
                dtype: dtypes[i].as_ptr(),
                shape: t.shape.as_ptr(),
                ndim: t.shape.len(),
                data: t.data.as_ptr(),
                len: t.data.len(),
            })
            .collect();
        let options = cstr(&options.to_string());
        let mut outs: *mut c_void = ptr::null_mut();
        call(self.table, "run", |err| unsafe {
            (self.table.run)(
                self.handle,
                views.as_ptr(),
                views.len(),
                options.as_ptr(),
                &mut outs,
                err,
            )
        })?;
        let n = unsafe { (self.table.outputs_len)(outs) };
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = PluginTensor {
                name: ptr::null(),
                dtype: ptr::null(),
                shape: ptr::null(),
                ndim: 0,
                data: ptr::null(),
                len: 0,
            };
            unsafe { (self.table.output)(outs, i, &mut v) };
            let t = unsafe { tensors_in(&v, 1) };
            match t {
                Ok(mut t) => result.push(t.remove(0)),
                Err(e) => {
                    unsafe { (self.table.free_outputs)(outs) };
                    return Err(e);
                }
            }
        }
        unsafe { (self.table.free_outputs)(outs) };
        Ok(result)
    }

    fn generate_text(&mut self, prompt: &str, options: &Value) -> Result<String> {
        let (prompt, options) = (cstr(prompt), cstr(&options.to_string()));
        let mut out: *mut c_char = ptr::null_mut();
        call(self.table, "generate_text", |err| unsafe {
            (self.table.generate_text)(
                self.handle,
                prompt.as_ptr(),
                options.as_ptr(),
                &mut out,
                err,
            )
        })?;
        Ok(unsafe { take_string(self.table, out) })
    }
}

impl Drop for PluginModel {
    fn drop(&mut self) {
        unsafe { (self.table.free_model)(self.handle) };
    }
}

/// Register the providers of the plugin table `table`; returns their names.
///
/// # Safety
/// `table` must point at a [`PluginV1`] that lives for the rest of the
/// process (a loaded library's static, never unloaded).
pub unsafe fn register_table(table: *const PluginV1) -> Result<Vec<String>> {
    if table.is_null() {
        return Err(anyhow!("plugin returned no function table"));
    }
    let table: &'static PluginV1 = unsafe { &*table };
    if table.abi != PLUGIN_ABI {
        return Err(anyhow!(
            "plugin ABI {} (this rlx speaks {PLUGIN_ABI}); rebuild the plugin against this rlx",
            table.abi
        ));
    }
    let mut out: *mut c_char = ptr::null_mut();
    call(table, "providers", |err| unsafe {
        (table.providers)(&mut out, err)
    })?;
    let list: Value = serde_json::from_str(&unsafe { take_string(table, out) })?;
    let strings = |v: &Value, k: &str| -> Vec<String> {
        v.get(k)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut names = Vec::new();
    for p in list.as_array().cloned().unwrap_or_default() {
        let name = p
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if name.is_empty() {
            continue;
        }
        names.push(name.clone());
        super::register_provider(Box::new(PluginProvider {
            table,
            c_name: cstr(&name),
            name,
            description: p
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            tasks: strings(&p, "tasks"),
            families: strings(&p, "families"),
        }));
    }
    Ok(names)
}

/// Load a plugin library and register its providers; returns their names.
#[cfg(feature = "model-plugins")]
pub fn load_plugin(path: impl AsRef<std::path::Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    // SAFETY: loading a library runs its initializers; a model plugin is
    // trusted code by construction (the user named it).
    let lib = unsafe { libloading::Library::new(path) }
        .map_err(|e| anyhow!("loading model plugin {}: {e}", path.display()))?;
    let table = unsafe {
        let f: libloading::Symbol<unsafe extern "C" fn() -> *const PluginV1> = lib
            .get(PLUGIN_SYMBOL)
            .map_err(|e| anyhow!("{} is not a model plugin: {e}", path.display()))?;
        f()
    };
    // The table and the providers' code live in the library: keep it loaded.
    std::mem::forget(lib);
    unsafe { register_table(table) }
}

/// Load a plugin library (unavailable: built without `model-plugins`).
#[cfg(not(feature = "model-plugins"))]
pub fn load_plugin(path: impl AsRef<std::path::Path>) -> Result<Vec<String>> {
    Err(anyhow!(
        "cannot load {}: rlx-runtime was built without the `model-plugins` feature",
        path.as_ref().display()
    ))
}

/// The plugin directory in the user's home: `$RLX_HOME/plugins`, else
/// `~/.rlx/plugins` (`%USERPROFILE%\\.rlx\\plugins` on Windows). Copying a
/// plugin library there is all it takes to install it.
///
/// `None` on iOS and the other Apple mobile systems, Android and wasm: no
/// library copied into an app's data loads there. Such an app links its
/// plugins ([`link_model_plugins!`](crate::link_model_plugins)), or loads one
/// it ships (a signed framework in the app bundle, a `.so` in the APK) with
/// [`load_plugin`].
pub fn home_plugin_dir() -> Option<std::path::PathBuf> {
    if cfg!(any(
        target_os = "ios",
        target_os = "tvos",
        target_os = "watchos",
        target_os = "visionos",
        target_os = "android",
        target_family = "wasm",
    )) {
        return None;
    }
    if let Some(h) = rlx_ir::env::var_os("RLX_HOME").filter(|h| !h.is_empty()) {
        return Some(std::path::PathBuf::from(h).join("plugins"));
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::PathBuf::from(home).join(".rlx").join("plugins"))
}

/// The plugin libraries in `dir` (files with this platform's shared-library
/// extension), sorted so loading is deterministic.
fn libraries_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut libs: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|x| x == std::env::consts::DLL_EXTENSION)
        })
        .collect();
    libs.sort();
    libs
}

/// Load the model plugins once: each `RLX_MODEL_PLUGINS` entry (a library,
/// or a directory of them), then every library in [`home_plugin_dir`]. A
/// failure is reported on stderr, not fatal: the other providers still work.
pub fn load_env_plugins() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut paths = Vec::new();
        if let Some(list) = rlx_ir::env::var_os("RLX_MODEL_PLUGINS") {
            for p in std::env::split_paths(&list) {
                if p.as_os_str().is_empty() {
                    continue;
                }
                if p.is_dir() {
                    paths.extend(libraries_in(&p));
                } else {
                    paths.push(p);
                }
            }
        }
        if let Some(dir) = home_plugin_dir() {
            paths.extend(libraries_in(&dir));
        }
        let mut seen = std::collections::HashSet::new();
        for path in paths {
            let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if !seen.insert(key) {
                continue;
            }
            if let Err(e) = load_plugin(&path) {
                eprintln!("rlx: model plugin {}: {e:#}", path.display());
            }
        }
    });
}
