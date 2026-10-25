// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How each op-table argument / return kind is spelled in every target.

use rlx_bind::ArgSpec;

/// The string-selected enum kinds (wire type `String`).
pub fn is_string_kind(kind: &str) -> bool {
    !matches!(
        kind,
        "node"
            | "opt_node"
            | "nodes"
            | "f32"
            | "f64"
            | "i32"
            | "i64"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "usizes"
            | "i64s"
            | "f32s"
            | "bytes"
            | "opt_usize"
            | "opt_f32"
            | "opt_usizes"
            | "opt_str"
            | "opt_activation"
            | "pair"
            | "opt_pair"
            | "triple"
            | "f32x3"
            | "pads"
            | "bands"
            | "sos"
            | "graph"
            | "opt_graph"
    )
}

/// The C parameters one argument takes: `(name, Rust FFI type)`.
pub fn c_params(a: &ArgSpec) -> Vec<(String, &'static str)> {
    let n = a.name.to_string();
    let len = |suffix: &str| (format!("{n}_{suffix}"), "usize");
    match a.kind {
        "node" | "opt_node" => vec![(n, "u32")],
        "nodes" => vec![(n.clone(), "*const u32"), len("len")],
        "f32" => vec![(n, "f32")],
        "f64" => vec![(n, "f64")],
        "i32" => vec![(n, "i32")],
        "i64" => vec![(n, "i64")],
        "u32" => vec![(n, "u32")],
        "u64" => vec![(n, "u64")],
        "usize" => vec![(n, "usize")],
        "bool" => vec![(n, "bool")],
        "usizes" | "pair" | "triple" | "opt_usizes" | "opt_pair" => {
            vec![(n.clone(), "*const usize"), len("len")]
        }
        "i64s" => vec![(n.clone(), "*const i64"), len("len")],
        "f32s" | "f32x3" => vec![(n.clone(), "*const f32"), len("len")],
        "bytes" => vec![(n.clone(), "*const u8"), len("len")],
        "opt_usize" => vec![(n, "*const usize")],
        "opt_f32" => vec![(n, "*const f32")],
        "pads" => vec![(n.clone(), "*const usize"), len("axes")],
        "bands" => vec![(n.clone(), "*const f32"), len("count")],
        "sos" => vec![(n.clone(), "*const f32"), len("sections")],
        "graph" | "opt_graph" => vec![(n, "*mut crate::graph::RlxGraph")],
        // `str`, `opt_str`, `opt_activation` and every enum kind.
        _ => vec![(n, "*const std::ffi::c_char")],
    }
}

/// The Rust expression turning the C parameters back into the wire value.
pub fn c_convert(a: &ArgSpec, what: &str) -> String {
    let n = a.name;
    let w = format!("\"{what} {n}\"");
    match a.kind {
        "node" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "bool" => n.to_string(),
        "opt_node" => format!("arg::opt_node({n})"),
        "nodes" | "usizes" | "pair" | "triple" | "i64s" | "f32s" | "f32x3" | "bytes" => {
            format!("arg::vec({n}, {n}_len, {w})?")
        }
        "opt_usizes" | "opt_pair" => format!("arg::opt_vec({n}, {n}_len, {w})?"),
        "opt_usize" | "opt_f32" => format!("arg::opt({n})"),
        "pads" => format!("arg::rows({n}, {n}_axes, 2, {w})?"),
        "bands" => format!("arg::rows({n}, {n}_count, 2, {w})?"),
        "sos" => format!("arg::rows({n}, {n}_sections, 6, {w})?"),
        "graph" => format!("take_graph({n}, {w})?"),
        "opt_graph" => {
            format!("if {n}.is_null() {{ None }} else {{ Some(take_graph({n}, {w})?) }}")
        }
        "opt_str" | "opt_activation" => format!("arg::opt_str({n}, {w})?"),
        // `str` and every enum kind: `NULL` takes the table default, if any.
        _ => match a.default {
            Some(d) => format!("arg::str_or({n}, {d}, {w})?"),
            None => format!("arg::str({n}, {w})?.to_string()"),
        },
    }
}

/// The C result parameters of a return kind: `(name, Rust FFI type)`.
pub fn c_rets(ret: &str) -> Vec<(&'static str, &'static str)> {
    match ret {
        "node" => vec![("ret", "*mut u32")],
        "pair" => vec![("ret0", "*mut u32"), ("ret1", "*mut u32")],
        "nodes" => vec![("ret", "*mut crate::ffi::RlxU32s")],
        "unit" => vec![],
        "usize" => vec![("ret", "*mut usize")],
        // shape (JSON), strings (JSON array), string.
        _ => vec![("ret", "*mut *mut std::ffi::c_char")],
    }
}

/// The Rust statement writing a core result `r` through the result parameters.
pub fn c_write(ret: &str, what: &str) -> String {
    let w = format!("\"{what}\"");
    match ret {
        "node" | "usize" => format!("out::put(ret, r, {w})"),
        "pair" => format!("out::put(ret0, r.0, {w})?;\n        out::put(ret1, r.1, {w})"),
        "nodes" => format!("out::put(ret, r.into(), {w})"),
        "shape" => format!("out::shape(ret, r, {w})"),
        "strings" => format!("out::strings(ret, &r, {w})"),
        _ => format!("out::string(ret, &r, {w})"),
    }
}

/// How a C reader should treat an argument, for the header's doc block.
pub fn c_note(a: &ArgSpec) -> Option<String> {
    let d = a
        .default
        .map(|d| format!(" (default {})", tidy(d)))
        .unwrap_or_default();
    Some(match a.kind {
        "opt_node" => format!("`{}`: a node id, or `RLX_NONE`{d}", a.name),
        "opt_usize" | "opt_f32" | "opt_usizes" | "opt_pair" | "opt_str" | "opt_activation"
        | "opt_graph" => {
            format!("`{}`: optional — `NULL` for none", a.name)
        }
        "pair" => format!("`{}`: 1 or 2 values{d}", a.name),
        "triple" => format!("`{}`: 1 or 3 values{d}", a.name),
        "pads" => format!(
            "`{}`: `[before, after]` per axis, flattened (`{}_axes` axes)",
            a.name, a.name
        ),
        "bands" => format!("`{}`: `[lo, hi]` per band, flattened", a.name),
        "sos" => format!(
            "`{}`: `[b0, b1, b2, a0, a1, a2]` per section, flattened",
            a.name
        ),
        "graph" => format!("`{}`: a body graph, consumed", a.name),
        k if is_string_kind(k) && a.default.is_some() => {
            format!("`{}`: `NULL` for the default{d}", a.name)
        }
        _ => match a.default {
            Some(_) => format!("`{}`{d}", a.name),
            None => return None,
        },
    })
}

/// A table default as written, tidied: `stringify!` spells `-1` as `- 1`.
pub fn tidy(default: &str) -> String {
    match default.strip_prefix("- ") {
        Some(rest) => format!("-{rest}"),
        None => default.to_string(),
    }
}

/// A table default as a target-language literal (`None` → `nil` / `None`).
pub fn literal(default: &str, none: &str, true_: &str, false_: &str) -> String {
    let default = tidy(default);
    match default.as_str() {
        "None" => none.to_string(),
        "true" => true_.to_string(),
        "false" => false_.to_string(),
        other => other.to_string(),
    }
}
