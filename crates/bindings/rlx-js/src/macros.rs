// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative macros for the binding surface.
//!
//! Every graph method is the same five steps — read the arguments, borrow the
//! graph, call one IR builder, wrap the result, register the name — and
//! spelling those out per op cost about thirty lines each and invited the two
//! mistakes that matter: borrowing the handle *before* converting arguments
//! (see [`crate::handle`] on aliasing), and a declared arity that drifts from
//! the parameter list.
//!
//! [`graph_ops!`] takes both out of a human's hands. It emits one anonymous
//! `NativeFn` per op — a non-capturing closure coerces to the plain `fn`
//! pointer the engine wants, so there is no name to invent — always converts
//! arguments first, and counts the arity from the signature.
//!
//! ```ignore
//! graph_ops! {
//!     /// Matrix multiply, shape inferred.
//!     "matmul"(a: node, b: node) => |g| g.mm(a, b);
//!     /// Softmax over `axis`, defaulting to the last.
//!     "softmax"(x: node, axis: i32 = -1) => |g| g.sm(x, axis);
//!     /// Eigendecomposition: two outputs, so `-> pair`.
//!     "eigh"(a: node) -> pair => |g| g.eigh(a);
//! }
//! ```

/// One typed argument read. `$i` is the positional index, `$what` the label
/// that shows up in the `TypeError`.
#[macro_export]
#[doc(hidden)]
macro_rules! js_arg {
    ($ctx:ident, $args:ident, $i:expr, node, $what:expr) => {
        $crate::api::graph::node($ctx, $args, $i, $what)?
    };
    ($ctx:ident, $args:ident, $i:expr, nodes, $what:expr) => {
        $crate::convert::to_u32_vec($ctx, $crate::convert::arg($args, $i), $what)?
            .into_iter()
            .map(rlx_ir::NodeId)
            .collect::<Vec<_>>()
    };
    ($ctx:ident, $args:ident, $i:expr, f32, $what:expr) => {
        $crate::convert::to_f32($ctx, $crate::convert::arg($args, $i))?
    };
    ($ctx:ident, $args:ident, $i:expr, i32, $what:expr) => {
        $crate::convert::to_i32($ctx, $crate::convert::arg($args, $i))?
    };
    ($ctx:ident, $args:ident, $i:expr, i64, $what:expr) => {
        $crate::convert::to_i32($ctx, $crate::convert::arg($args, $i))? as i64
    };
    ($ctx:ident, $args:ident, $i:expr, usize, $what:expr) => {
        $crate::convert::to_usize($ctx, $crate::convert::arg($args, $i), $what)?
    };
    ($ctx:ident, $args:ident, $i:expr, u64, $what:expr) => {
        $crate::convert::to_u64($ctx, $crate::convert::arg($args, $i), $what)?
    };
    ($ctx:ident, $args:ident, $i:expr, bool, $what:expr) => {
        $crate::convert::to_bool($crate::convert::arg($args, $i))
    };
    ($ctx:ident, $args:ident, $i:expr, usizes, $what:expr) => {
        $crate::convert::to_usize_vec($ctx, $crate::convert::arg($args, $i), $what)?
    };
    ($ctx:ident, $args:ident, $i:expr, i64s, $what:expr) => {
        $crate::convert::to_i64_vec($ctx, $crate::convert::arg($args, $i), $what)?
    };
    ($ctx:ident, $args:ident, $i:expr, str, $what:expr) => {
        $ctx.to_rust_string($crate::convert::arg($args, $i))?
    };
    ($ctx:ident, $args:ident, $i:expr, dtype, $what:expr) => {{
        let label = $ctx.to_rust_string($crate::convert::arg($args, $i))?;
        $crate::api::graph::parse_dtype($ctx, &label)?
    }};
    ($ctx:ident, $args:ident, $i:expr, activation, $what:expr) => {{
        let label = $ctx.to_rust_string($crate::convert::arg($args, $i))?;
        $crate::api::graph::parse_activation($ctx, &label)?
    }};
    ($ctx:ident, $args:ident, $i:expr, fft_norm, $what:expr) => {
        $crate::api::graph_ext::parse_fft_norm($ctx, $crate::convert::arg($args, $i))?
    };
    ($ctx:ident, $args:ident, $i:expr, binop, $what:expr) => {{
        let label = $ctx.to_rust_string($crate::convert::arg($args, $i))?;
        $crate::api::graph::parse_binop($ctx, &label)?
    }};
    ($ctx:ident, $args:ident, $i:expr, cmp_op, $what:expr) => {{
        let label = $ctx.to_rust_string($crate::convert::arg($args, $i))?;
        $crate::api::graph::parse_cmp($ctx, &label)?
    }};
    ($ctx:ident, $args:ident, $i:expr, reduce_op, $what:expr) => {{
        let label = $ctx.to_rust_string($crate::convert::arg($args, $i))?;
        $crate::api::graph::parse_reduce($ctx, &label)?
    }};
    ($ctx:ident, $args:ident, $i:expr, mask, $what:expr) => {{
        let label = $crate::convert::to_string_or($ctx, $crate::convert::arg($args, $i), "causal")?;
        $crate::api::graph::parse_mask_kind($ctx, &label)?
    }};
}

/// The same read, but falling back to `$default` when the slot is
/// `undefined` / `null` — JavaScript's way of saying "use the default".
#[macro_export]
#[doc(hidden)]
macro_rules! js_arg_or {
    ($ctx:ident, $args:ident, $i:expr, $kind:tt, $what:expr, $default:expr) => {
        if $crate::convert::is_nullish($crate::convert::arg($args, $i)) {
            $default
        } else {
            $crate::js_arg!($ctx, $args, $i, $kind, $what)
        }
    };
}

/// Wrap what an IR builder returned as a JS value.
#[macro_export]
#[doc(hidden)]
macro_rules! js_ret {
    (node, $ctx:ident, $value:expr) => {
        $crate::api::graph::id($value)
    };
    // Two-output ops (`rfft`, `eigh`, `fftReal`) hand back a tuple; a JS array
    // destructures, which is what the call site wants: `const [re, im] = …`.
    (pair, $ctx:ident, $value:expr) => {{
        let (first, second) = $value;
        let items = vec![
            $crate::api::graph::id(first),
            $crate::api::graph::id(second),
        ];
        $crate::convert::new_array($ctx, items)
    }};
}

/// Count macro arguments without `${count}` (still unstable on our MSRV).
#[macro_export]
#[doc(hidden)]
macro_rules! js_arity {
    ($($name:ident),*) => {{
        let slots: &[()] = &[$( { let _ = stringify!($name); } ),*];
        slots.len() as u32
    }};
}

/// Declare `rlx.Graph` methods: one entry per IR op.
///
/// Expands to a `&[(&str, NativeFn, u32)]` table and the loop that installs
/// it on a prototype. Arguments are read left to right *before* the graph
/// handle is borrowed, which is the invariant that makes a re-entrant getter
/// safe rather than undefined behaviour.
#[macro_export]
macro_rules! graph_ops {
    (
        $ctx:expr, $proto:expr;
        $(
            $(#[$meta:meta])*
            $js:literal ( $( $arg:ident : $kind:tt $( = $default:expr )? ),* $(,)? )
            $( -> $ret:tt )?
            => | $g:ident $(, $cx:ident )? | $body:expr ;
        )*
    ) => {{
        let methods: &[(&str, quickrs_core::object::NativeFn, u32)] = &[
            $((
                $js,
                {
                    #[allow(unused_variables, non_snake_case)]
                    let f: quickrs_core::object::NativeFn = |ctx, this, args, _magic| {
                        // Argument reads, in declaration order. Each may run
                        // script; none of them holds the graph.
                        $crate::graph_ops_bind!(ctx, args, 0usize, $js, $( $arg : $kind $( = $default )? ),*);
                        let $g = $crate::api::graph::graph_of(ctx, this)?;
                        // Every `node`-kind argument is checked against the
                        // graph before it reaches an IR builder that indexes.
                        $crate::graph_ops_verify!(ctx, $g, $( $arg : $kind ),*);
                        // A body that reports its own errors names the context
                        // (`|g, ctx|`): macro hygiene means it cannot reach
                        // ours otherwise, and an implicit capture would stop
                        // the closure coercing to a plain `fn`. The reborrow is
                        // scoped to the body so the return wrapper can use it.
                        // rlx asserts shape invariants with `panic!`, so a body
                        // that cannot fail any other way runs under a guard
                        // that converts one into a thrown error.
                        let out = $crate::graph_ops_body!(ctx, $js, [$( $cx )?] $body);
                        Ok($crate::graph_ops_ret!($( $ret )?, ctx, out))
                    };
                    f
                },
                $crate::js_arity!($( $arg ),*),
            )),*
        ];
        for (name, func, arity) in methods {
            $ctx.define_method($proto, name, *func, *arity, 0);
        }
    }};
}

/// Recursive positional binder for [`graph_ops!`]: threads the index so each
/// argument reads its own slot.
#[macro_export]
#[doc(hidden)]
macro_rules! graph_ops_bind {
    // Both base arms: the recursive call emits no trailing comma once the
    // parameter list is exhausted, and the initial call may carry one.
    ($ctx:ident, $args:ident, $i:expr, $what:expr) => {};
    ($ctx:ident, $args:ident, $i:expr, $what:expr,) => {};
    ($ctx:ident, $args:ident, $i:expr, $what:expr, $arg:ident : $kind:tt = $default:expr $(, $rest:ident : $rkind:tt $( = $rdefault:expr )? )*) => {
        let $arg = $crate::js_arg_or!($ctx, $args, $i, $kind, $what, $default);
        $crate::graph_ops_bind!($ctx, $args, $i + 1usize, $what $(, $rest : $rkind $( = $rdefault )? )*);
    };
    ($ctx:ident, $args:ident, $i:expr, $what:expr, $arg:ident : $kind:tt $(, $rest:ident : $rkind:tt $( = $rdefault:expr )? )*) => {
        let $arg = $crate::js_arg!($ctx, $args, $i, $kind, $what);
        $crate::graph_ops_bind!($ctx, $args, $i + 1usize, $what $(, $rest : $rkind $( = $rdefault )? )*);
    };
}

/// Run a [`graph_ops!`] body, panic-guarded when it does not need the context.
///
/// Two arms, keyed on whether the body declared `|g, ctx|`. A body that names
/// the context reports its own errors and may use `?`, so it cannot be moved
/// into a closure; a body that does not is pure IR construction, which is
/// exactly where rlx's shape assertions live.
#[macro_export]
#[doc(hidden)]
macro_rules! graph_ops_body {
    ($ctx:ident, $what:expr, [] $body:expr) => {
        $crate::panics::catch($ctx, $what, || $body)?
    };
    ($ctx:ident, $what:expr, [$cx:ident] $body:expr) => {{
        let $cx = &mut *$ctx;
        $body
    }};
}

/// Check the `node` / `nodes` arguments of a [`graph_ops!`] body, skipping the
/// scalar kinds. Generated from the same signature, so a new op is covered the
/// moment it is declared.
#[macro_export]
#[doc(hidden)]
macro_rules! graph_ops_verify {
    ($ctx:ident, $g:ident $(,)?) => {};
    ($ctx:ident, $g:ident, $arg:ident : node $(, $rest:ident : $rkind:tt )*) => {
        $crate::api::graph::check_node($ctx, $g, $arg, stringify!($arg))?;
        $crate::graph_ops_verify!($ctx, $g $(, $rest : $rkind )*);
    };
    ($ctx:ident, $g:ident, $arg:ident : nodes $(, $rest:ident : $rkind:tt )*) => {
        $crate::api::graph::check_nodes($ctx, $g, &$arg, stringify!($arg))?;
        $crate::graph_ops_verify!($ctx, $g $(, $rest : $rkind )*);
    };
    ($ctx:ident, $g:ident, $arg:ident : $kind:tt $(, $rest:ident : $rkind:tt )*) => {
        $crate::graph_ops_verify!($ctx, $g $(, $rest : $rkind )*);
    };
}

/// `-> $ret` is optional; absent means a single node id.
#[macro_export]
#[doc(hidden)]
macro_rules! graph_ops_ret {
    (, $ctx:ident, $value:expr) => {
        $crate::js_ret!(node, $ctx, $value)
    };
    ($ret:tt, $ctx:ident, $value:expr) => {
        $crate::js_ret!($ret, $ctx, $value)
    };
}

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
