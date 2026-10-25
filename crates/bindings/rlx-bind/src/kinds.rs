// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Argument and return kinds of the op table.
//!
//! Every parameter in [`for_each_graph_op!`](crate::for_each_graph_op) names a
//! *kind*. A kind fixes three things:
//!
//! * its **wire type** — what a binding hands the core (`u32` for a node,
//!   `String` for any string-selected enum, `Vec<usize>` for a pair, …).
//!   Wire types are plain owned Rust values, so every language converts to
//!   them the same way it converts to anything else;
//! * its **domain type** — what an op body sees after decoding (`NodeId`
//!   checked against the graph, `RopeStyle` parsed, `[usize; 2]`
//!   validated). Decoding happens once, here, for every language;
//! * how a **default** written in the table becomes a wire value. Defaults are
//!   neutral literals (`-1`, `1e-5`, `"causal"`, `[1, 1]`, `None`) so the
//!   generators can print them in any target language.
//!
//! | kind | wire | domain |
//! |------|------|--------|
//! | `node` / `opt_node` / `nodes` | `u32` / `Option<u32>` / `Vec<u32>` | `NodeId`s, checked |
//! | `f32 f64 i32 i64 u32 u64 usize bool` | itself | itself |
//! | `usizes i64s f32s bytes` | `Vec<_>` | itself |
//! | `str` | `String` | `String` |
//! | `opt_usize opt_f32 opt_usizes opt_str` | `Option<_>` | itself |
//! | `dtype` + the enum kinds of [`crate::enums`] | `String` | parsed enum |
//! | `opt_activation` | `Option<String>` | `Option<Activation>` |
//! | `pair` / `opt_pair` / `triple` | `Vec<usize>` (1 or N values) | `[usize; N]` |
//! | `f32x3` | `Vec<f32>` | `[f32; 3]` |
//! | `pads` | `Vec<Vec<usize>>` | `Vec<[usize; 2]>` |
//! | `bands` | `Vec<Vec<f32>>` | `Vec<(f32, f32)>` |
//! | `sos` | `Vec<Vec<f32>>` | `Vec<([f32; 3], [f32; 3])>` |
//! | `graph` / `opt_graph` | `Graph` (moved in) | itself |
//!
//! Returns: `node` (the default), `pair`, `nodes`, `unit`, `shape`
//! (`(dims, dtype)`, a dynamic dim reads `-1`), `strings`, `usize`, `string`.

use rlx_ir::{Dim, Graph, NodeId, Shape};

use crate::{ApiError, ApiResult};

/// The wire type of a kind, for a binding's argument conversion.
#[macro_export]
#[doc(hidden)]
macro_rules! wire_ty {
    (node) => { u32 };
    (opt_node) => { ::core::option::Option<u32> };
    (nodes) => { ::std::vec::Vec<u32> };
    (f32) => { f32 };
    (f64) => { f64 };
    (i32) => { i32 };
    (i64) => { i64 };
    (u32) => { u32 };
    (u64) => { u64 };
    (usize) => { usize };
    (bool) => { bool };
    (usizes) => { ::std::vec::Vec<usize> };
    (i64s) => { ::std::vec::Vec<i64> };
    (f32s) => { ::std::vec::Vec<f32> };
    (bytes) => { ::std::vec::Vec<u8> };
    (str) => { ::std::string::String };
    (opt_usize) => { ::core::option::Option<usize> };
    (opt_f32) => { ::core::option::Option<f32> };
    (opt_usizes) => { ::core::option::Option<::std::vec::Vec<usize>> };
    (opt_str) => { ::core::option::Option<::std::string::String> };
    (opt_activation) => { ::core::option::Option<::std::string::String> };
    (pair) => { ::std::vec::Vec<usize> };
    (opt_pair) => { ::core::option::Option<::std::vec::Vec<usize>> };
    (triple) => { ::std::vec::Vec<usize> };
    (f32x3) => { ::std::vec::Vec<f32> };
    (pads) => { ::std::vec::Vec<::std::vec::Vec<usize>> };
    (bands) => { ::std::vec::Vec<::std::vec::Vec<f32>> };
    (sos) => { ::std::vec::Vec<::std::vec::Vec<f32>> };
    (graph) => { ::rlx_ir::Graph };
    (opt_graph) => { ::core::option::Option<::rlx_ir::Graph> };
    // Every remaining kind is a string-selected enum.
    ($enum_kind:ident) => { ::std::string::String };
}

/// A table default as a wire value of `$kind`.
#[macro_export]
#[doc(hidden)]
macro_rules! wire_default {
    (str, $d:expr) => {
        ::std::string::String::from($d)
    };
    (usizes, $d:expr) => {
        <::std::vec::Vec<usize>>::from($d)
    };
    (i64s, $d:expr) => {
        <::std::vec::Vec<i64>>::from($d)
    };
    (f32s, $d:expr) => {
        <::std::vec::Vec<f32>>::from($d)
    };
    (bytes, $d:expr) => {
        <::std::vec::Vec<u8>>::from($d)
    };
    (nodes, $d:expr) => {
        <::std::vec::Vec<u32>>::from($d)
    };
    (pair, $d:expr) => {
        <::std::vec::Vec<usize>>::from($d)
    };
    (triple, $d:expr) => {
        <::std::vec::Vec<usize>>::from($d)
    };
    (opt_node, $d:expr) => {
        $d
    };
    (opt_usize, $d:expr) => {
        $d
    };
    (opt_f32, $d:expr) => {
        $d
    };
    (opt_usizes, $d:expr) => {
        $d
    };
    (opt_str, $d:expr) => {
        $d
    };
    (opt_activation, $d:expr) => {
        $d
    };
    (opt_pair, $d:expr) => {
        $d
    };
    (opt_graph, $d:expr) => {
        $d
    };
    (f32, $d:expr) => {
        $d
    };
    (f64, $d:expr) => {
        $d
    };
    (i32, $d:expr) => {
        $d
    };
    (i64, $d:expr) => {
        $d
    };
    (u32, $d:expr) => {
        $d
    };
    (u64, $d:expr) => {
        $d
    };
    (usize, $d:expr) => {
        $d
    };
    (bool, $d:expr) => {
        $d
    };
    ($enum_kind:ident, $d:expr) => {
        ::std::string::String::from($d)
    };
}

/// Decode a wire value into the domain value an op body sees.
#[macro_export]
#[doc(hidden)]
macro_rules! decode {
    ($g:ident, node, $v:expr, $what:expr) => {
        $crate::kinds::node(&*$g, $v, $what)?
    };
    ($g:ident, opt_node, $v:expr, $what:expr) => {
        match $v {
            Some(n) => Some($crate::kinds::node(&*$g, n, $what)?),
            None => None,
        }
    };
    ($g:ident, nodes, $v:expr, $what:expr) => {
        $crate::kinds::nodes(&*$g, $v, $what)?
    };
    ($g:ident, pair, $v:expr, $what:expr) => {
        $crate::kinds::fixed_or_splat::<2>($v, $what)?
    };
    ($g:ident, opt_pair, $v:expr, $what:expr) => {
        match $v {
            Some(v) => Some($crate::kinds::fixed_or_splat::<2>(v, $what)?),
            None => None,
        }
    };
    ($g:ident, triple, $v:expr, $what:expr) => {
        $crate::kinds::fixed_or_splat::<3>($v, $what)?
    };
    ($g:ident, f32x3, $v:expr, $what:expr) => {
        $crate::kinds::three($v, $what)?
    };
    ($g:ident, pads, $v:expr, $what:expr) => {
        $crate::kinds::pads($v, $what)?
    };
    ($g:ident, bands, $v:expr, $what:expr) => {
        $crate::kinds::bands($v, $what)?
    };
    ($g:ident, sos, $v:expr, $what:expr) => {
        $crate::kinds::sos($v, $what)?
    };
    ($g:ident, opt_activation, $v:expr, $what:expr) => {
        $crate::enums::opt_activation(($v).as_deref())?
    };
    // Values that need no decoding.
    ($g:ident, f32, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, f64, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, i32, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, i64, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, u32, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, u64, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, usize, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, bool, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, usizes, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, i64s, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, f32s, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, bytes, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, str, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, opt_usize, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, opt_f32, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, opt_usizes, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, opt_str, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, graph, $v:expr, $what:expr) => {
        $v
    };
    ($g:ident, opt_graph, $v:expr, $what:expr) => {
        $v
    };
    // A string-selected enum: its parser is the function named after it.
    ($g:ident, $enum_kind:ident, $v:expr, $what:expr) => {
        $crate::enums::$enum_kind(&$v)?
    };
}

/// The wire type of a return kind.
#[macro_export]
#[doc(hidden)]
macro_rules! ret_ty {
    () => { u32 };
    (node) => { u32 };
    (pair) => { (u32, u32) };
    (nodes) => { ::std::vec::Vec<u32> };
    (unit) => { () };
    (shape) => { (::std::vec::Vec<i64>, ::std::string::String) };
    (strings) => { ::std::vec::Vec<::std::string::String> };
    (usize) => { usize };
    (string) => { ::std::string::String };
}

/// Encode an op body's domain result as its wire value.
#[macro_export]
#[doc(hidden)]
macro_rules! encode {
    ((), $v:expr) => {
        ($v).0
    };
    ((node), $v:expr) => {
        ($v).0
    };
    ((pair), $v:expr) => {{
        let (a, b) = $v;
        (a.0, b.0)
    }};
    ((nodes), $v:expr) => {
        ($v).into_iter().map(|n| n.0).collect()
    };
    ((unit), $v:expr) => {
        $v
    };
    ((shape), $v:expr) => {
        $crate::kinds::shape_wire(&$v)
    };
    ((strings), $v:expr) => {
        $v
    };
    ((usize), $v:expr) => {
        $v
    };
    ((string), $v:expr) => {
        $v
    };
}

// ── decoders ────────────────────────────────────────────────

/// The `op arg` an error names, kept as two `&str`s so the success path never
/// formats a string.
#[derive(Clone, Copy, Debug)]
pub struct Arg<'a>(pub &'a str, pub &'a str);

impl std::fmt::Display for Arg<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1.is_empty() {
            f.write_str(self.0)
        } else {
            write!(f, "{} {}", self.0, self.1)
        }
    }
}

/// Reject a node id the graph does not have.
///
/// Without this an out-of-range id goes straight into `Graph::shape`, which
/// indexes — a host-process panic from a typo.
pub fn node(g: &Graph, id: u32, what: Arg<'_>) -> ApiResult<NodeId> {
    let count = g.nodes().len();
    if (id as usize) < count {
        return Ok(NodeId(id));
    }
    Err(ApiError::range(format!(
        "{what}: node id {id} does not exist in this graph (it has {count} node(s)). \
         Scalars are not node ids — wrap one with `constant(v)` first."
    )))
}

pub fn nodes(g: &Graph, ids: Vec<u32>, what: Arg<'_>) -> ApiResult<Vec<NodeId>> {
    ids.into_iter().map(|id| node(g, id, what)).collect()
}

/// `N` values, or one value repeated — `stride: 2` means `[2, 2]`.
pub fn fixed_or_splat<const N: usize>(v: Vec<usize>, what: Arg<'_>) -> ApiResult<[usize; N]> {
    if v.len() == 1 {
        return Ok([v[0]; N]);
    }
    <[usize; N]>::try_from(v)
        .map_err(|v| ApiError::type_(format!("{what}: expected 1 or {N} values, got {}", v.len())))
}

/// Exactly `N` values.
pub fn exactly<const N: usize, T: std::fmt::Debug>(v: Vec<T>, what: &str) -> ApiResult<[T; N]> {
    <[T; N]>::try_from(v)
        .map_err(|v| ApiError::type_(format!("{what}: expected {N} values, got {}", v.len())))
}

pub fn three(v: Vec<f32>, what: Arg<'_>) -> ApiResult<[f32; 3]> {
    <[f32; 3]>::try_from(v)
        .map_err(|v| ApiError::type_(format!("{what}: expected 3 coefficients, got {}", v.len())))
}

/// One `[before, after]` per axis; a single value is symmetric.
pub fn pads(rows: Vec<Vec<usize>>, what: Arg<'_>) -> ApiResult<Vec<[usize; 2]>> {
    rows.into_iter()
        .enumerate()
        .map(|(i, r)| match r.len() {
            1 => Ok([r[0], r[0]]),
            2 => Ok([r[0], r[1]]),
            got => Err(ApiError::type_(format!(
                "{what}: axis {i} needs [before, after], got {got} values"
            ))),
        })
        .collect()
}

/// `[[lo, hi], …]` frequency bands.
pub fn bands(rows: Vec<Vec<f32>>, what: Arg<'_>) -> ApiResult<Vec<(f32, f32)>> {
    rows.into_iter()
        .enumerate()
        .map(|(i, r)| match r.as_slice() {
            [lo, hi] => Ok((*lo, *hi)),
            _ => Err(ApiError::type_(format!(
                "{what}: row {i} needs 2 values, got {}",
                r.len()
            ))),
        })
        .collect()
}

/// SciPy `sos` rows: `[[b0, b1, b2, a0, a1, a2], …]`.
pub fn sos(rows: Vec<Vec<f32>>, what: Arg<'_>) -> ApiResult<Vec<([f32; 3], [f32; 3])>> {
    rows.into_iter()
        .enumerate()
        .map(|(i, r)| match r.as_slice() {
            [b0, b1, b2, a0, a1, a2] => Ok(([*b0, *b1, *b2], [*a0, *a1, *a2])),
            _ => Err(ApiError::type_(format!(
                "{what}: row {i} needs 6 values, got {}",
                r.len()
            ))),
        })
        .collect()
}

/// `(dims, dtype)`; a dynamic dim reads back as `-1`.
pub fn shape_wire(s: &Shape) -> (Vec<i64>, String) {
    let dims = s
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    (dims, crate::enums::dtype_label(s.dtype()).to_string())
}
