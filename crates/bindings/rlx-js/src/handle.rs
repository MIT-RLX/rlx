// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rust values held by JavaScript objects.
//!
//! A `Graph`, `Session` or `CompiledGraph` is far too big to copy into the JS
//! heap, so each lives in a `Box` whose address rides in the host slot of an
//! ordinary object. The class id stamped alongside it is what makes
//! `session.compile(notAGraph)` a `TypeError` instead of a wild pointer
//! dereference.
//!
//! # Aliasing
//!
//! [`borrow_mut`] hands out `&mut T` from that address. Two live `&mut` to the
//! same object would be undefined behaviour, so the rule every method here
//! follows is: **take the borrow, finish with it, then touch the engine.**
//! No method holds a handle borrow across a call that can re-enter JavaScript
//! (a getter, a `toString`, a callback), which is what makes the second borrow
//! impossible to reach.

use std::marker::PhantomData;

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::host::{HostFinalizer, HostObject};
use quickrs_core::object::{JsObject, ObjectData, PropFlags};
use quickrs_core::sys::Ptr;
use quickrs_core::value::{JsResult, PropKey, Value};

/// Class tags. Distinct constants, not an enum, because the engine stores the
/// tag as a bare `u32`.
pub const CLASS_GRAPH: u32 = 0x524C_5801;
pub const CLASS_SESSION: u32 = 0x524C_5802;
pub const CLASS_COMPILED: u32 = 0x524C_5803;
pub const CLASS_GGUF: u32 = 0x524C_5804;
pub const CLASS_RUNNER: u32 = 0x524C_5805;
pub const CLASS_ROUTER: u32 = 0x524C_5806;
pub const CLASS_OPTIMIZER: u32 = 0x524C_5807;
pub const CLASS_TOKENIZER: u32 = 0x524C_5808;
pub const CLASS_TRAINER: u32 = 0x524C_5809;
pub const CLASS_SAMPLER: u32 = 0x524C_580A;
pub const CLASS_TENSOR: u32 = 0x524C_580B;

/// Where the prototypes live. Non-writable, non-configurable and
/// non-enumerable, so a script cannot swap one out and make every later
/// `unwrap` fail — or succeed on the wrong type.
const REGISTRY: &str = "__rlx_classes";

/// Drops the `Box<T>` behind a host object's opaque word when the object dies.
struct DropBox<T>(PhantomData<T>);

impl<T> HostFinalizer for DropBox<T> {
    fn finalize(&self, _class_id: u32, opaque: usize) {
        if opaque != 0 {
            // SAFETY: `opaque` is the pointer `wrap` leaked from `Box::new`,
            // handed back exactly once — the engine runs a finalizer per
            // object, and nothing else ever frees it.
            drop(unsafe { Box::from_raw(opaque as *mut T) });
        }
    }
}

/// Property on the hidden registry recording whether this context may touch
/// the filesystem.
const FS_ALLOWED: &str = "fsAllowed";

/// Grant filesystem access to this context. Called by
/// [`crate::install_fs`] only.
pub fn allow_fs(ctx: &mut Context) {
    if let Some(reg) = registry(ctx) {
        ctx.define_value(&reg, FS_ALLOWED, Value::Bool(true), PropFlags::NONE);
    }
}

/// Whether this context may touch the filesystem.
pub fn fs_allowed(ctx: &mut Context) -> bool {
    let Some(reg) = registry(ctx) else {
        return false;
    };
    let key = PropKey::from_string(&ctx.intern(FS_ALLOWED));
    ctx.get_property(&Value::Object(reg), &key)
        .map(|v| v.to_boolean())
        .unwrap_or(false)
}

/// Refuse a path-taking call in a sandboxed context.
///
/// Every function that names a file goes through this, not just the obvious
/// `readFile`: `openGguf`, `loadTokenizer`, `writeGguf` and the weight loaders
/// all read or write, and leaving them ungated made "sandboxed by default" a
/// claim rather than a fact.
pub fn require_fs<T>(ctx: &mut Context, what: &str) -> JsResult<Option<T>> {
    if fs_allowed(ctx) {
        return Ok(None);
    }
    ctx.throw_type(&format!(
        "{what}: this runtime has no filesystem access. An embedder grants it \
         with `Runtime::allow_filesystem()`; the `rlx-js` CLI does so already."
    ))
}

/// Install the hidden class registry. Call once, before any `register`.
pub fn init_registry(ctx: &mut Context) {
    let registry = ctx.new_object();
    let global = ctx.global();
    ctx.define_value(
        &global,
        REGISTRY,
        Value::Object(registry),
        PropFlags::NONE, // not writable, not configurable, not enumerable
    );
}

fn registry(ctx: &mut Context) -> Option<Gc<JsObject>> {
    let key = PropKey::from_string(&ctx.intern(REGISTRY));
    let global = Value::Object(ctx.global());
    ctx.get_property(&global, &key)
        .ok()
        .and_then(|v| v.as_object().cloned())
}

/// Create a prototype for `name`, record it under its class id, and return it
/// so the caller can hang methods on it.
pub fn register(ctx: &mut Context, name: &str, class_id: u32) -> Gc<JsObject> {
    let proto = ctx.new_object();
    if let Some(reg) = registry(ctx) {
        ctx.define_value(
            &reg,
            &class_id.to_string(),
            Value::Object(proto.clone()),
            PropFlags::NONE,
        );
    }
    // A readable tag, so `String(obj)` and devtools show something useful.
    let tag = Value::Str(ctx.intern(name));
    ctx.define_value(&proto, "__class", tag, PropFlags::NONE);
    proto
}

fn proto_for(ctx: &mut Context, class_id: u32) -> Option<Gc<JsObject>> {
    let reg = registry(ctx)?;
    let key = PropKey::from_string(&ctx.intern(&class_id.to_string()));
    ctx.get_property(&Value::Object(reg), &key)
        .ok()
        .and_then(|v| v.as_object().cloned())
}

/// Move `value` onto the JS heap as an object of class `class_id`.
pub fn wrap<T: 'static>(ctx: &mut Context, class_id: u32, value: T) -> Value {
    wrap_with(ctx, class_id, Vec::new(), value)
}

/// As [`wrap`], but keeping `data` alive alongside the object.
///
/// `data` holds JavaScript values the host wants traced by the cycle
/// collector — a `Tensor` keeps its owning `Graph` here, which is what stops
/// the graph being collected while a chained expression is still building on
/// it.
pub fn wrap_with<T: 'static>(
    ctx: &mut Context,
    class_id: u32,
    data: Vec<Value>,
    value: T,
) -> Value {
    let proto = proto_for(ctx, class_id);
    let opaque = Box::into_raw(Box::new(value)) as usize;
    let finalizer: Ptr<dyn HostFinalizer> = Ptr::new(DropBox::<T>(PhantomData));
    let obj = ctx.new_host_object(
        proto,
        HostObject {
            class_id,
            opaque,
            finalizer: Some(finalizer),
            exotic: None,
            data,
        },
    );
    Value::Object(obj)
}

/// The JS values [`wrap_with`] kept alive for `v`.
pub fn host_data(v: &Value, class_id: u32) -> Option<Vec<Value>> {
    let obj = v.as_object()?;
    let borrowed = obj.borrow();
    match &borrowed.data {
        ObjectData::Host(host) if host.class_id == class_id => Some(host.data.clone()),
        _ => None,
    }
}

/// The prototype registered for `class_id`, for building an object that should
/// inherit from it.
pub fn prototype_of(ctx: &mut Context, class_id: u32) -> Option<Gc<JsObject>> {
    proto_for(ctx, class_id)
}

/// The opaque word of `v`, if it is a live host object of `class_id`.
fn opaque_of(v: &Value, class_id: u32) -> Option<usize> {
    let obj = v.as_object()?;
    let borrowed = obj.borrow();
    match &borrowed.data {
        ObjectData::Host(host) if host.class_id == class_id && host.opaque != 0 => {
            Some(host.opaque)
        }
        _ => None,
    }
}

/// `&mut T` behind `v`, or a `TypeError` naming what was expected.
///
/// See the module note on aliasing: finish with the borrow before calling back
/// into the engine.
pub fn borrow_mut<'a, T: 'static>(
    ctx: &mut Context,
    v: &Value,
    class_id: u32,
    what: &str,
) -> JsResult<&'a mut T> {
    match opaque_of(v, class_id) {
        // SAFETY: the word came from `Box::into_raw::<T>` under this same
        // class id, and the object holding it is alive for the call (it is
        // either `this` or an argument, both rooted by the caller's frame).
        Some(p) => Ok(unsafe { &mut *(p as *mut T) }),
        None => ctx.throw_type(&format!("expected a {what}")),
    }
}

/// Whether `v` is a host object of `class_id`.
pub fn is_a(v: &Value, class_id: u32) -> bool {
    opaque_of(v, class_id).is_some()
}
