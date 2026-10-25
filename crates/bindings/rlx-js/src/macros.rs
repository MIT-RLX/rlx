// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative macros for the binding surface.
//!
//! The `Graph` methods are not declared here: they are generated from the op
//! table in `rlx-bind` by [`crate::js_graph_ops!`] (see [`crate::api::bind`]).
//! What remains are the installers for host classes and plain functions.

/// Install a host-object class: prototype, methods, constructor, and the
/// `rlx.<Name>` binding.
///
/// Replaces four hand-rolled copies that had already drifted — one wired the
/// prototype with the wrong `PropFlags`, which let a script reassign
/// `Runner.prototype` and make every later `unwrap` fail.
///
/// A class with no `ctor:` is still exposed, as a non-constructible holder
/// carrying `.prototype`. That is what makes `compiled instanceof rlx.Compiled`
/// work for the classes a script never news up itself (`Compiled`, `Gguf`,
/// `Tokenizer`, `Tensor`). Calling it reports *why*; `new`-ing it gets the
/// engine's own terser "not a constructor", which is equally correct.
#[macro_export]
macro_rules! js_class {
    (
        $ctx:expr, $namespace:expr;
        name: $name:literal,
        class: $class:expr,
        $( ctor: $ctor:expr, )?
        methods: { $( $js:literal => $func:expr $(, $arity:expr )? ; )* }
    ) => {{
        let ctx = &mut *$ctx;
        let proto = $crate::handle::register(ctx, concat!("rlx.", $name), $class);
        $(
            ctx.define_method(
                &proto,
                $js,
                $func as quickrs_core::object::NativeFn,
                $crate::js_class_arity!($( $arity )?),
                0,
            );
        )*
        #[allow(unused_mut, unused_assignments)]
        let mut ctor: Option<quickrs_core::gc::Gc<quickrs_core::object::JsObject>> = None;
        $(
            let f = ctx.new_native_function($ctor as quickrs_core::object::NativeFn, $name, 2, 0);
            ctx.set_ctor_kind(&f, quickrs_core::object::CtorKind::Constructor);
            ctor = Some(f);
        )?
        let holder = match ctor {
            Some(f) => f,
            None => {
                let f = ctx.new_native_function(
                    $crate::macros::not_constructible,
                    $name,
                    0,
                    0,
                );
                let label = quickrs_core::value::Value::Str(ctx.intern($name));
                ctx.set_slot(&f, "__className", label);
                f
            }
        };
        // Non-writable, non-configurable: a swapped prototype would make the
        // class tag unverifiable, and the tag is what stands between a bad cast
        // and a wild pointer.
        ctx.define_value(
            &holder,
            "prototype",
            quickrs_core::value::Value::Object(proto.clone()),
            quickrs_core::object::PropFlags::NONE,
        );
        ctx.define_value(
            $namespace,
            $name,
            quickrs_core::value::Value::Object(holder),
            quickrs_core::object::PropFlags::C_W,
        );
        proto
    }};
}

/// The body of the holder a constructor-less [`js_class!`] exposes.
pub fn not_constructible(
    ctx: &mut quickrs_core::context::Context,
    _this: &quickrs_core::value::Value,
    _args: &[quickrs_core::value::Value],
    _magic: i32,
) -> quickrs_core::value::JsResult<quickrs_core::value::Value> {
    let name = ctx
        .callee_slot("__className")
        .ok()
        .and_then(|v| ctx.to_rust_string(&v).ok())
        .unwrap_or_else(|| "this class".to_string());
    ctx.throw_type(&format!(
        "rlx.{name} is not constructible — it only comes from the API that \
         produces one (e.g. Session.compile, rlx.openGguf, rlx.loadTokenizer, \
         rlx.dsl). It is exposed so `instanceof` works."
    ))
}

/// `Function.length` for a [`js_class!`] method; defaults to 1.
#[macro_export]
#[doc(hidden)]
macro_rules! js_class_arity {
    () => {
        1
    };
    ($arity:expr) => {
        $arity
    };
}

/// Install static methods on a class [`js_class!`] already put on `namespace`
/// (`rlx.Runner.fromEnv`).
#[macro_export]
macro_rules! js_statics {
    ( $ctx:expr, $namespace:expr, $class:literal; $( $js:literal => $func:expr, $arity:expr ; )* ) => {{
        let ctx = &mut *$ctx;
        let ns = quickrs_core::value::Value::Object($namespace.clone());
        if let Ok(quickrs_core::value::Value::Object(holder)) = $crate::convert::field(ctx, &ns, $class) {
            $(
                ctx.define_method(
                    &holder,
                    $js,
                    $func as quickrs_core::object::NativeFn,
                    $arity,
                    0,
                );
            )*
        }
    }};
}

/// Install plain `rlx.*` functions.
#[macro_export]
macro_rules! js_functions {
    ( $ctx:expr, $namespace:expr; $( $js:literal => $func:expr, $arity:expr ; )* ) => {{
        let ctx = &mut *$ctx;
        $(
            ctx.define_method(
                $namespace,
                $js,
                $func as quickrs_core::object::NativeFn,
                $arity,
                0,
            );
        )*
    }};
}
