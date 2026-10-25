// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runtime API table (`rlx-bind/src/api.idl`) against the C ABI, and the
//! Swift it generates.
//!
//! [`check`] holds the two to each other: every row names a C function with
//! exactly the parameter types its kinds expand to, every class has its
//! `_free`, and every exported C function is a row or plumbing (buffers,
//! outputs, graph builders). [`swift`] then prints one Swift member per row —
//! so a row added to the table and the C ABI reaches Swift with no hand edit,
//! and one missing from either fails generation.

use std::collections::BTreeSet;
use std::fmt::Write;

use rlx_bind::api::{Api, Class, Entry, Param, Returns, Role};
use rlx_bind::manifest::camel;

use crate::header::CFn;

// ── C conformance ───────────────────────────────────────────

/// The C parameter types an argument kind expands to.
fn c_args(kind: &str) -> Result<Vec<String>, String> {
    let v: &[&str] = match kind {
        "str" | "str?" | "json" | "json?" | "value" | "value?" => &["const char*"],
        "bool" => &["bool"],
        "f32" => &["float"],
        "u32" | "node" => &["uint32_t"],
        "u64" => &["uint64_t"],
        "usize" => &["size_t"],
        "u64?" => &["const uint64_t*"],
        "usize?" => &["const size_t*"],
        "f32?" => &["const float*"],
        "f32s" => &["const float*", "size_t"],
        "u8s" => &["const uint8_t*", "size_t"],
        "u32s" | "nodes" => &["const uint32_t*", "size_t"],
        "usizes" => &["const size_t*", "size_t"],
        "strs" | "strs?" => &["const char* const*", "size_t"],
        "inputs" => &[
            "const char* const*",
            "const float* const*",
            "const size_t*",
            "size_t",
        ],
        "typed_inputs" => &[
            "const char* const*",
            "const uint8_t* const*",
            "const size_t*",
            "const char* const*",
            "size_t",
        ],
        "graph" | "graph&" => &["RlxGraph*"],
        other => match other.strip_suffix('&') {
            Some(class) if class.starts_with(char::is_uppercase) => {
                return Ok(vec![format!("Rlx{class}*")]);
            }
            _ => return Err(format!("unknown argument kind `{other}`")),
        },
    };
    Ok(v.iter().map(|s| s.to_string()).collect())
}

/// The C out-parameter types a return kind expands to.
fn c_outs(kind: &str) -> Result<Vec<String>, String> {
    let v: &[&str] = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "strings" => &["char**"],
        "f32s" => &["RlxF32s*"],
        "u8s" => &["RlxBytes*"],
        "u32s" => &["RlxU32s*"],
        "outputs" => &["RlxOutputs**"],
        "typed_outputs" => &["RlxTypedOutputs**"],
        "tensors" => &["RlxTensorMap**"],
        "bool" => &["bool*"],
        "usize" => &["size_t*"],
        "u32" => &["uint32_t*"],
        "u64" => &["uint64_t*"],
        "f32" => &["float*"],
        "f32?" => &["float*", "bool*"],
        "graph" => &["RlxGraph**"],
        other if other.starts_with(char::is_uppercase) => {
            return Ok(vec![format!("Rlx{other}**")]);
        }
        other => return Err(format!("unknown return kind `{other}`")),
    };
    Ok(v.iter().map(|s| s.to_string()).collect())
}

/// A C function that returns its one result directly instead of a status.
fn direct(ret_kind: &str) -> Option<&'static str> {
    match ret_kind {
        "bool" => Some("bool"),
        "str" => Some("const char*"),
        _ => None,
    }
}

/// The C function a row binds to, and whether it returns its result directly.
fn expected(api: &Api, e: &Entry, f: &CFn) -> Result<(Vec<String>, bool), String> {
    let mut want = Vec::new();
    if e.role == Role::Method {
        want.push(format!("Rlx{}*", api.class_of(e).unwrap().name));
    }
    for p in &e.params {
        want.extend(c_args(&p.kind)?);
    }
    let kinds = match (&e.returns, e.role) {
        (_, Role::Constructor) => vec![api.class_of(e).unwrap().name.clone()],
        (r, _) => r.kinds().into_iter().map(str::to_string).collect(),
    };
    let is_direct = f.ret != "int32_t";
    if is_direct {
        let [k] = kinds.as_slice() else {
            return Err("returns a value directly but the row has several".into());
        };
        if direct(k) != Some(f.ret.as_str()) {
            return Err(format!("returns `{}` directly; the row says `{k}`", f.ret));
        }
    } else {
        for k in &kinds {
            want.extend(c_outs(k)?);
        }
    }
    Ok((want, is_direct))
}

/// C functions that are plumbing rather than rows.
fn is_plumbing(name: &str) -> bool {
    name.starts_with("rlx_graph_")
        || name.starts_with("rlx_outputs_")
        || name.starts_with("rlx_typed_outputs_")
        || name.starts_with("rlx_tensor_map_")
        || matches!(
            name,
            "rlx_last_error"
                | "rlx_u32s_free"
                | "rlx_f32s_free"
                | "rlx_bytes_free"
                | "rlx_string_free"
                | "rlx_u32s_release"
                | "rlx_f32s_release"
                | "rlx_bytes_release"
                | "rlx_batch_set"
        )
}

/// Every disagreement between the table and the C ABI.
pub fn check(api: &Api, fns: &[CFn]) -> Vec<String> {
    let by_name = |n: &str| fns.iter().find(|f| f.name == n);
    let mut errors = Vec::new();
    let mut used = BTreeSet::new();
    for c in &api.classes {
        let free = c.c_free();
        if by_name(&free).is_none() {
            errors.push(format!("class {}: rlx-capi has no `{free}`", c.name));
        }
        used.insert(free);
    }
    for e in &api.entries {
        let Some(c_name) = &e.names.c else { continue };
        used.insert(c_name.clone());
        let Some(f) = by_name(c_name) else {
            errors.push(format!("{}: rlx-capi has no `{c_name}`", label(api, e)));
            continue;
        };
        if e.manual {
            continue;
        }
        match expected(api, e, f) {
            Ok((want, _)) => {
                let got: Vec<String> = f.params.iter().map(|(t, _)| t.clone()).collect();
                if got != want {
                    errors.push(format!(
                        "{}: `{c_name}` takes ({}) but the row expands to ({})",
                        label(api, e),
                        got.join(", "),
                        want.join(", ")
                    ));
                }
            }
            Err(msg) => errors.push(format!("{}: {msg}", label(api, e))),
        }
    }
    for f in fns {
        if !used.contains(&f.name) && !is_plumbing(&f.name) {
            errors.push(format!(
                "`{}` is exported by rlx-capi but is no row of api.idl",
                f.name
            ));
        }
    }
    errors
}

fn label(api: &Api, e: &Entry) -> String {
    match api.class_of(e) {
        Some(c) => format!("{}.{}", c.name, e.name),
        None => e.name.clone(),
    }
}

// ── Swift ───────────────────────────────────────────────────

fn swift_ident(name: &str) -> String {
    crate::swift::ident(name)
}

fn swift_arg_type(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "String?",
        "json" => "[String: Any]",
        "json?" => "[String: Any]?",
        "value" => "Any",
        "value?" => "Any?",
        "bool" => "Bool",
        "f32" => "Float",
        "u32" => "UInt32",
        "u64" => "UInt64",
        "usize" => "Int",
        "node" => "Node",
        "u64?" => "UInt64?",
        "usize?" => "Int?",
        "f32?" => "Float?",
        "f32s" => "[Float]",
        "u8s" => "[UInt8]",
        "u32s" => "[UInt32]",
        "nodes" => "[Node]",
        "usizes" => "[Int]",
        "strs" => "[String]",
        "strs?" => "[String]?",
        "inputs" => "[String: [Float]]",
        "typed_inputs" => "[String: TypedTensor]",
        "graph" | "graph&" => "Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn swift_ret_type(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "String?",
        "json" => "Any",
        "object" => "[String: Any]",
        "rows" => "[[String: Any]]",
        "shapes" => "[[Int]]",
        "strings" => "[String]",
        "f32s" => "[Float]",
        "u8s" => "[UInt8]",
        "u32s" => "[UInt32]",
        "outputs" => "[[Float]]",
        "typed_outputs" => "[TypedTensor]",
        "tensors" => "[String: [Float]]",
        "bool" => "Bool",
        "usize" => "Int",
        "u32" => "UInt32",
        "u64" => "UInt64",
        "f32" => "Float",
        "f32?" => "Float?",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn optional(kind: &str) -> bool {
    kind.ends_with('?')
}

/// One argument lowered: the Swift parameter, statements before the call, a
/// scoped-pointer wrapper (opening line), and the C arguments.
struct Lowered {
    decl: String,
    pre: Option<String>,
    wrap: Option<String>,
    args: Vec<String>,
}

fn lower(p: &Param, label: bool) -> Lowered {
    let v = swift_ident(&camel(&p.name));
    let ty = swift_arg_type(&p.kind);
    let default = match &p.default {
        Some(d) => format!(" = {d}"),
        None if optional(&p.kind) => " = nil".into(),
        None => String::new(),
    };
    let decl = if label {
        format!("{v}: {ty}{default}")
    } else {
        format!("_ {v}: {ty}{default}")
    };
    let n = camel(&p.name);
    let (pre, wrap, args): (Option<String>, Option<String>, Vec<String>) = match p.kind.as_str() {
        "str" => (None, None, vec![v.clone()]),
        "str?" => (
            None,
            Some(format!("try withOptionalCString({v}) {{ {n}P in")),
            vec![format!("{n}P")],
        ),
        "json" | "json?" | "value" | "value?" => (
            Some(format!("let {n}J = try jsonText({v})")),
            Some(format!("try withOptionalCString({n}J) {{ {n}P in")),
            vec![format!("{n}P")],
        ),
        "f32s" | "u8s" | "u32s" | "usizes" => (None, None, vec![v.clone(), format!("{v}.count")]),
        "nodes" => (
            Some(format!("let {n}Ids = {v}.map(\\.id)")),
            None,
            vec![format!("{n}Ids"), format!("{n}Ids.count")],
        ),
        "strs" => (
            None,
            Some(format!("try withCStrings({v}) {{ {n}P, {n}N in")),
            vec![format!("{n}P"), format!("{n}N")],
        ),
        "strs?" => (
            None,
            Some(format!("try withOptionalCStrings({v}) {{ {n}P, {n}N in")),
            vec![format!("{n}P"), format!("{n}N")],
        ),
        "inputs" => (
            None,
            Some(format!(
                "try withInputs({v}) {{ {n}Names, {n}Data, {n}Lens, {n}N in"
            )),
            vec![
                format!("{n}Names"),
                format!("{n}Data"),
                format!("{n}Lens"),
                format!("{n}N"),
            ],
        ),
        "typed_inputs" => (
            None,
            Some(format!(
                "try withTypedInputs({v}) {{ {n}Names, {n}Data, {n}Lens, {n}Types, {n}N in"
            )),
            vec![
                format!("{n}Names"),
                format!("{n}Data"),
                format!("{n}Lens"),
                format!("{n}Types"),
                format!("{n}N"),
            ],
        ),
        "u64?" | "usize?" | "f32?" => (
            None,
            Some(format!("try withOptionalValue({v}) {{ {n}P in")),
            vec![format!("{n}P")],
        ),
        "node" => (None, None, vec![format!("{v}.id")]),
        "graph" | "graph&" => (None, None, vec![format!("{v}.handle")]),
        k if k.ends_with('&') => (None, None, vec![format!("{v}.handle")]),
        _ => (None, None, vec![v.clone()]),
    };
    Lowered {
        decl,
        pre,
        wrap,
        args,
    }
}

/// Out-variable declarations, the C arguments that write them, and the Swift
/// expression of the result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("out{i}");
    let ptr = |ty: &str| (vec![format!("var {o}: {ty}")], vec![format!("&{o}")]);
    let (decls, args) = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "strings" => {
            ptr("UnsafeMutablePointer<CChar>? = nil")
        }
        "f32s" => ptr("RlxF32s = RlxF32s(ptr: nil, len: 0)"),
        "u8s" => ptr("RlxBytes = RlxBytes(ptr: nil, len: 0)"),
        "u32s" => ptr("RlxU32s = RlxU32s(ptr: nil, len: 0)"),
        "bool" => ptr("Bool = false"),
        "usize" => ptr("Int = 0"),
        "u32" => ptr("UInt32 = 0"),
        "u64" => ptr("UInt64 = 0"),
        "f32" => ptr("Float = 0"),
        "f32?" => (
            vec![
                format!("var {o}: Float = 0"),
                format!("var {o}Known = false"),
            ],
            vec![format!("&{o}"), format!("&{o}Known")],
        ),
        _ => ptr("OpaquePointer? = nil"),
    };
    let expr = match kind {
        "str" => format!("takeString({o})"),
        "str?" => format!("{o}.map {{ takeString($0) }}"),
        "json" => format!("try takeJSON({o})"),
        "object" => format!("try takeObject({o})"),
        "rows" => format!("try takeRows({o})"),
        "shapes" => format!("try decodeJSON([[Int]].self, takeString({o}))"),
        "strings" => format!("try decodeJSON([String].self, takeString({o}))"),
        "f32s" => format!("takeF32s({o})"),
        "u8s" => format!("takeBytes({o})"),
        "u32s" => format!("takeU32s({o})"),
        "outputs" => format!("try collectOutputs({o}!)"),
        "typed_outputs" => format!("try takeTypedOutputs({o}!)"),
        "tensors" => format!("try takeTensorMap({o}!)"),
        "f32?" => format!("{o}Known ? {o} : nil"),
        "bool" | "usize" | "u32" | "u64" | "f32" => o.clone(),
        "graph" => format!("Graph(handle: {o}!)"),
        class => format!("{class}(handle: {o}!)"),
    };
    (decls, args, expr)
}

fn doc(s: &mut String, text: &str, indent: &str) {
    for line in text.lines() {
        if line.is_empty() {
            let _ = writeln!(s, "{indent}///");
        } else {
            let _ = writeln!(s, "{indent}/// {line}");
        }
    }
}

/// `#if` guard for a feature outside rlx-capi's defaults.
fn guard(feature: Option<&str>, defaults: &[String]) -> Option<String> {
    feature
        .filter(|f| !defaults.iter().any(|d| d == f))
        .map(|f| format!("RLX_{}", f.to_ascii_uppercase().replace('-', "_")))
}

/// One Swift member (or free function) for a row.
fn member(s: &mut String, api: &Api, e: &Entry, f: &CFn, indent: &str) {
    let swift = e.names.swift.as_deref().unwrap_or("");
    let (_, is_direct) = expected(api, e, f).expect("checked");
    doc(s, &e.doc, indent);
    // The first argument reads unlabeled, unless it is a constructor's or a
    // bare number (a rank, a token id), which need their name to read.
    let lowered: Vec<Lowered> = e
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let graph_first = p.kind == "graph" || p.kind == "graph&";
            let unlabeled =
                i == 0 && (graph_first || (e.role != Role::Constructor && p.kind != "u32"));
            lower(p, !unlabeled)
        })
        .collect();
    let decls: Vec<&str> = lowered.iter().map(|l| l.decl.as_str()).collect();
    let ret_kinds: Vec<String> = match (&e.returns, e.role) {
        (_, Role::Constructor) => vec![api.class_of(e).unwrap().name.clone()],
        (r, _) => r.kinds().into_iter().map(str::to_string).collect(),
    };
    let ret_ty = match &e.returns {
        _ if e.role == Role::Constructor => String::new(),
        Returns::Unit => String::new(),
        Returns::One(k) => format!(" -> {}", swift_ret_type(k)),
        Returns::Tuple(items) => format!(
            " -> ({})",
            items
                .iter()
                .map(|(n, k)| format!("{}: {}", camel(n), swift_ret_type(k)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let throws = if is_direct { "" } else { " throws" };
    let head = match e.role {
        Role::Constructor => format!("public convenience init({}){throws}", decls.join(", ")),
        Role::Static => format!(
            "public static func {}({}){throws}{ret_ty}",
            swift_ident(swift),
            decls.join(", ")
        ),
        _ => format!(
            "public func {}({}){throws}{ret_ty}",
            swift_ident(swift),
            decls.join(", ")
        ),
    };
    let discardable = e.role == Role::Method
        && matches!(&e.returns, Returns::One(k) if k == "usize" || k == "object")
        || matches!(e.returns, Returns::Tuple(_));
    if discardable {
        let _ = writeln!(s, "{indent}@discardableResult");
    }
    let _ = writeln!(s, "{indent}{head} {{");
    let body = format!("{indent}    ");

    let mut args: Vec<String> = Vec::new();
    if e.role == Role::Method {
        args.push("handle".into());
    }
    for l in &lowered {
        if let Some(p) = &l.pre {
            let _ = writeln!(s, "{body}{p}");
        }
        args.extend(l.args.iter().cloned());
    }
    let c_name = e.names.c.as_deref().unwrap();
    if is_direct {
        let call = format!("{c_name}({})", args.join(", "));
        let expr = match ret_kinds[0].as_str() {
            "str" => format!("String(cString: {call})"),
            _ => call,
        };
        let wraps: Vec<&String> = lowered.iter().filter_map(|l| l.wrap.as_ref()).collect();
        assert!(
            wraps.is_empty(),
            "{c_name}: a direct return takes no scoped arguments"
        );
        let _ = writeln!(s, "{body}{expr}");
        let _ = writeln!(s, "{indent}}}");
        return;
    }
    let mut exprs = Vec::new();
    for (i, k) in ret_kinds.iter().enumerate() {
        let (decls, out_args, expr) = outs(k, i);
        for d in decls {
            let _ = writeln!(s, "{body}{d}");
        }
        args.extend(out_args);
        exprs.push(expr);
    }
    let mut depth = 0;
    for l in &lowered {
        if let Some(w) = &l.wrap {
            let _ = writeln!(s, "{body}{}{w}", "    ".repeat(depth));
            depth += 1;
        }
    }
    let _ = writeln!(
        s,
        "{body}{}try check({c_name}({}))",
        "    ".repeat(depth),
        args.join(", ")
    );
    while depth > 0 {
        depth -= 1;
        let _ = writeln!(s, "{body}{}}}", "    ".repeat(depth));
    }
    match (e.role, &e.returns) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{body}self.init(handle: out0!)");
        }
        (_, Returns::Unit) => {}
        (_, Returns::One(_)) => {
            let _ = writeln!(s, "{body}return {}", exprs[0]);
        }
        (_, Returns::Tuple(items)) => {
            let parts: Vec<String> = items
                .iter()
                .zip(&exprs)
                .map(|((n, _), x)| format!("{}: {x}", camel(n)))
                .collect();
            let _ = writeln!(s, "{body}return ({})", parts.join(", "));
        }
    }
    let _ = writeln!(s, "{indent}}}");
}

fn open_guard(s: &mut String, g: &Option<String>) {
    if let Some(g) = g {
        let _ = writeln!(s, "#if {g}");
    }
}

fn close_guard(s: &mut String, g: &Option<String>) {
    if g.is_some() {
        s.push_str("#endif\n");
    }
}

fn class(s: &mut String, api: &Api, c: &Class, fns: &[CFn], defaults: &[String]) {
    let by_name = |n: &str| fns.iter().find(|f| f.name == n);
    let g = guard(c.feature.as_deref(), defaults);
    s.push('\n');
    open_guard(s, &g);
    doc(s, &c.doc, "");
    let _ = writeln!(s, "public final class {} {{", c.name);
    let _ = writeln!(s, "    let handle: OpaquePointer\n");
    let _ = writeln!(
        s,
        "    init(handle: OpaquePointer) {{ self.handle = handle }}\n"
    );
    let _ = writeln!(s, "    deinit {{ {}(handle) }}", c.c_free());
    for e in api.members(c) {
        let (Some(c_name), Some(_)) = (&e.names.c, &e.names.swift) else {
            continue;
        };
        if e.manual {
            continue;
        }
        let eg = guard(e.feature.as_deref(), defaults).filter(|x| Some(x) != g.as_ref());
        s.push('\n');
        open_guard(s, &eg);
        member(s, api, e, by_name(c_name).expect("checked"), "    ");
        close_guard(s, &eg);
    }
    s.push_str("}\n");
    close_guard(s, &g);
}

/// `rlx-swift/Sources/RLX/API.swift`. Assumes [`check`] passed.
pub fn swift(api: &Api, fns: &[CFn], defaults: &[String]) -> String {
    let by_name = |n: &str| fns.iter().find(|f| f.name == n);
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's runtime API table (api.idl) —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         //\n\
         // One member per row, calling the rlx-capi function the row names. Rows\n\
         // marked `manual` (callbacks, in-place buffers) are in Manual.swift; rows\n\
         // behind a feature rlx-capi does not build by default sit under\n\
         // `#if RLX_<FEATURE>` (set by Package.swift from RLX_CAPI_FEATURES).\n\
         \n\
         import CRLX\n\
         import Foundation\n",
    );
    s.push_str("\n// ── functions ─────────────────────────────────────────────\n");
    for e in api
        .entries
        .iter()
        .filter(|e| e.class.is_none() && !e.manual)
    {
        let (Some(c_name), Some(_)) = (&e.names.c, &e.names.swift) else {
            continue;
        };
        let g = guard(e.feature.as_deref(), defaults);
        s.push('\n');
        open_guard(&mut s, &g);
        member(&mut s, api, e, by_name(c_name).expect("checked"), "");
        close_guard(&mut s, &g);
    }
    s.push_str("\n// ── classes ───────────────────────────────────────────────\n");
    for c in api.classes.iter().filter(|c| !c.manual) {
        class(&mut s, api, c, fns, defaults);
    }
    s
}

/// The runtime table for `docs/bindings-api.md`: every row in every language.
pub fn docs(api: &Api) -> String {
    let mut s = String::from(
        "\n## Runtime API\n\n\
         Every row of `crates/bindings/rlx-bind/src/api.idl` — the classes and\n\
         functions past the graph builders — with its name in each binding. `—`\n\
         is a binding without it (the reason follows the table).\n\n\
         | row | Python | JS | C | Swift |\n\
         |-----|--------|----|---|-------|\n",
    );
    let cell = |n: &Option<String>, also: &[String]| match n {
        None => "—".to_string(),
        Some(n) => std::iter::once(n)
            .chain(also)
            .map(|x| format!("`{x}`"))
            .collect::<Vec<_>>()
            .join(", "),
    };
    let mut gaps = Vec::new();
    for e in &api.entries {
        let row = label(api, e);
        let swift = match (api.class_of(e), &e.names.swift) {
            (Some(c), Some(n)) => Some(format!("{}.{n}", c.name)),
            (_, n) => n.clone(),
        };
        let _ = writeln!(
            s,
            "| `{row}` | {} | {} | {} | {} |",
            cell(&e.names.py, &e.py_also),
            cell(&e.names.js, &e.js_also),
            cell(&e.names.c, &[]),
            cell(&swift, &[]),
        );
        if let Some(why) = &e.why
            && !gaps.contains(why)
        {
            gaps.push(why.clone());
        }
    }
    if !gaps.is_empty() {
        s.push_str("\nWhere a binding lacks a row:\n\n");
        for g in gaps {
            let _ = writeln!(s, "- {g}");
        }
    }
    s.push_str("\nNames a binding exports that are not rows:\n\n");
    for h in &api.hosts {
        let _ = writeln!(
            s,
            "- **{}**: {} — {}",
            h.lang,
            h.names
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", "),
            h.why
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfn(name: &str, params: &[&str]) -> CFn {
        CFn {
            name: name.into(),
            ret: "int32_t".into(),
            params: params.iter().map(|t| (t.to_string(), "p".into())).collect(),
            feature: None,
        }
    }

    const TABLE: &str = "\
class Thing
    new(name: str, opts: json?)
    run(inputs: inputs) -> (device: str, outputs: outputs)
end
fn count(xs: f32s) -> usize
";

    fn good() -> Vec<CFn> {
        vec![
            cfn(
                "rlx_thing_new",
                &["const char*", "const char*", "RlxThing**"],
            ),
            cfn("rlx_thing_free", &["RlxThing*"]),
            cfn(
                "rlx_thing_run",
                &[
                    "RlxThing*",
                    "const char* const*",
                    "const float* const*",
                    "const size_t*",
                    "size_t",
                    "char**",
                    "RlxOutputs**",
                ],
            ),
            cfn("rlx_count", &["const float*", "size_t", "size_t*"]),
        ]
    }

    #[test]
    fn a_table_matching_the_c_abi_passes() {
        let api = rlx_bind::api::parse(TABLE).unwrap();
        assert_eq!(check(&api, &good()), Vec::<String>::new());
    }

    #[test]
    fn every_kind_of_drift_is_reported() {
        let api = rlx_bind::api::parse(TABLE).unwrap();
        let mut fns = good();
        fns[3].params.pop(); // rlx_count lost its out-parameter
        fns.retain(|f| f.name != "rlx_thing_free");
        fns.push(cfn("rlx_orphan", &[]));
        let errors = check(&api, &fns).join("\n");
        assert!(errors.contains("count: `rlx_count` takes"), "{errors}");
        assert!(errors.contains("no `rlx_thing_free`"), "{errors}");
        assert!(errors.contains("`rlx_orphan` is exported"), "{errors}");
    }

    #[test]
    fn swift_members_follow_the_rows() {
        let api = rlx_bind::api::parse(TABLE).unwrap();
        let s = swift(&api, &good(), &[]);
        assert!(
            s.contains("public convenience init(name: String, opts: [String: Any]? = nil) throws"),
            "{s}"
        );
        assert!(s.contains("public func run(_ inputs: [String: [Float]]) throws -> (device: String, outputs: [[Float]])"), "{s}");
        assert!(
            s.contains("public func count(_ xs: [Float]) throws -> Int"),
            "{s}"
        );
        assert!(s.contains("deinit { rlx_thing_free(handle) }"), "{s}");
    }
}
