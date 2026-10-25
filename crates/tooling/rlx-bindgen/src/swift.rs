// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-swift/Sources/RLX/GraphOps.swift`: a `Graph` method per op-table row,
//! calling the generated C function.
//!
//! The first argument is unlabeled, the rest labeled (camelCase); defaults
//! come from the table; Swift arrays and strings bridge to the C pointers
//! directly, optionals go through the `with…` helpers in `Core.swift`.

use std::fmt::Write;

use rlx_bind::manifest::camel;
use rlx_bind::{ArgSpec, OpSpec};

use crate::kinds::{is_string_kind, literal};

const KEYWORDS: &[&str] = &[
    "associatedtype",
    "class",
    "deinit",
    "enum",
    "extension",
    "fileprivate",
    "func",
    "import",
    "init",
    "inout",
    "internal",
    "let",
    "open",
    "operator",
    "private",
    "protocol",
    "public",
    "rethrows",
    "static",
    "struct",
    "subscript",
    "typealias",
    "var",
    "break",
    "case",
    "continue",
    "default",
    "defer",
    "do",
    "else",
    "fallthrough",
    "for",
    "guard",
    "if",
    "in",
    "repeat",
    "return",
    "switch",
    "where",
    "while",
    "as",
    "Any",
    "catch",
    "false",
    "is",
    "nil",
    "super",
    "self",
    "Self",
    "throw",
    "throws",
    "true",
    "try",
];

pub(crate) fn ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("`{name}`")
    } else {
        name.to_string()
    }
}

/// The Swift parameter type of a kind.
fn ty(kind: &str) -> &'static str {
    match kind {
        "node" => "Node",
        "opt_node" => "Node?",
        "nodes" => "[Node]",
        "f32" => "Float",
        "f64" => "Double",
        "i32" => "Int32",
        "i64" => "Int64",
        "u32" => "UInt32",
        "u64" => "UInt64",
        "usize" => "Int",
        "bool" => "Bool",
        "usizes" | "pair" | "triple" => "[Int]",
        "opt_usizes" | "opt_pair" => "[Int]?",
        "i64s" => "[Int64]",
        "f32s" | "f32x3" => "[Float]",
        "bytes" => "[UInt8]",
        "opt_usize" => "Int?",
        "opt_f32" => "Float?",
        "pads" => "[[Int]]",
        "bands" | "sos" => "[[Float]]",
        "graph" => "Graph",
        "opt_graph" => "Graph?",
        "opt_str" | "opt_activation" => "String?",
        _ => "String",
    }
}

fn ret_ty(ret: &str) -> &'static str {
    match ret {
        "node" => "Node",
        "pair" => "(Node, Node)",
        "nodes" => "[Node]",
        "unit" => "Void",
        "shape" => "Shape",
        "strings" => "[String]",
        "usize" => "Int",
        _ => "String",
    }
}

fn default(a: &ArgSpec) -> Option<String> {
    a.default.map(|d| literal(d, "nil", "true", "false"))
}

/// One method. Arguments that need a scoped pointer wrap the call in a
/// `with…` closure, innermost last.
fn method(s: &mut String, op: &OpSpec) {
    let name = ident(&op.swift_name());
    s.push('\n');
    for line in op.doc.lines().map(|l| l.strip_prefix(' ').unwrap_or(l)) {
        let _ = writeln!(
            s,
            "    ///{}{}",
            if line.is_empty() { "" } else { " " },
            line.trim_end()
        );
    }
    let mut params = Vec::new();
    for (i, a) in op.args.iter().enumerate() {
        let local = ident(&camel(a.name));
        let label = if i == 0 {
            format!("_ {local}")
        } else {
            local.clone()
        };
        let d = default(a).map(|d| format!(" = {d}")).unwrap_or_default();
        params.push(format!("{label}: {}{d}", ty(a.kind)));
    }
    let ret = ret_ty(op.ret);
    let attr = if matches!(op.ret, "node" | "pair" | "nodes") {
        "    @discardableResult\n"
    } else {
        ""
    };
    let _ = write!(s, "{attr}    func {name}(");
    s.push_str(&params.join(", "));
    let arrow = if ret == "Void" {
        String::new()
    } else {
        format!(" -> {ret}")
    };
    let _ = writeln!(s, ") throws{arrow} {{");

    // Values computed before the call.
    let mut pre = Vec::new();
    // Scoped pointer wrappers: (open, close).
    let mut wraps: Vec<String> = Vec::new();
    let mut call = vec!["handle".to_string()];
    for a in op.args {
        let v = ident(&camel(a.name));
        let raw = camel(a.name);
        match a.kind {
            "node" => call.push(format!("{v}.id")),
            "opt_node" => call.push(format!("{v}?.id ?? RLX_NONE")),
            "nodes" => {
                pre.push(format!("let {raw}Ids = {v}.map(\\.id)"));
                call.push(format!("{raw}Ids, {raw}Ids.count"));
            }
            "usizes" | "pair" | "triple" | "i64s" | "f32s" | "f32x3" | "bytes" => {
                call.push(format!("{v}, {v}.count"));
            }
            "opt_usizes" | "opt_pair" => {
                wraps.push(format!("withOptionalArray({v}) {{ {raw}P, {raw}N in"));
                call.push(format!("{raw}P, {raw}N"));
            }
            "opt_usize" | "opt_f32" => {
                wraps.push(format!("withOptionalValue({v}) {{ {raw}P in"));
                call.push(format!("{raw}P"));
            }
            "pads" => {
                pre.push(format!(
                    "let {raw}Flat = try flatRows({v}, width: 2, what: \"{}\", splat: true)",
                    a.name
                ));
                call.push(format!("{raw}Flat, {v}.count"));
            }
            "bands" => {
                pre.push(format!(
                    "let {raw}Flat = try flatRows({v}, width: 2, what: \"{}\")",
                    a.name
                ));
                call.push(format!("{raw}Flat, {v}.count"));
            }
            "sos" => {
                pre.push(format!(
                    "let {raw}Flat = try flatRows({v}, width: 6, what: \"{}\")",
                    a.name
                ));
                call.push(format!("{raw}Flat, {v}.count"));
            }
            "graph" => call.push(format!("{v}.handle")),
            "opt_graph" => call.push(format!("{v}?.handle")),
            "opt_str" | "opt_activation" => {
                wraps.push(format!("withOptionalCString({v}) {{ {raw}P in"));
                call.push(format!("{raw}P"));
            }
            "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "bool" => call.push(v),
            k if is_string_kind(k) => call.push(v),
            other => panic!("rlx-bindgen: no Swift lowering for kind '{other}'"),
        }
    }
    let (ret_decl, ret_args, ret_value): (Vec<&str>, Vec<&str>, String) = match op.ret {
        "node" => (
            vec!["var ret: UInt32 = 0"],
            vec!["&ret"],
            "Node(id: ret)".into(),
        ),
        "pair" => (
            vec!["var ret0: UInt32 = 0", "var ret1: UInt32 = 0"],
            vec!["&ret0", "&ret1"],
            "(Node(id: ret0), Node(id: ret1))".into(),
        ),
        "nodes" => (
            vec!["var ret = RlxU32s(ptr: nil, len: 0)"],
            vec!["&ret"],
            "takeU32s(ret).map { Node(id: $0) }".into(),
        ),
        "unit" => (vec![], vec![], String::new()),
        "usize" => (vec!["var ret: Int = 0"], vec!["&ret"], "ret".into()),
        "shape" => (
            vec!["var ret: UnsafeMutablePointer<CChar>? = nil"],
            vec!["&ret"],
            "try decodeJSON(Shape.self, takeString(ret))".into(),
        ),
        "strings" => (
            vec!["var ret: UnsafeMutablePointer<CChar>? = nil"],
            vec!["&ret"],
            "try decodeJSON([String].self, takeString(ret))".into(),
        ),
        _ => (
            vec!["var ret: UnsafeMutablePointer<CChar>? = nil"],
            vec!["&ret"],
            "takeString(ret)".into(),
        ),
    };
    for p in &pre {
        let _ = writeln!(s, "        {p}");
    }
    for d in &ret_decl {
        let _ = writeln!(s, "        {d}");
    }
    call.extend(ret_args.iter().map(|a| a.to_string()));
    let fn_name = format!("rlx_graph_{}", op.c_name());
    let mut indent = "        ".to_string();
    for w in &wraps {
        let _ = writeln!(s, "{indent}try {w}");
        indent.push_str("    ");
    }
    let _ = writeln!(s, "{indent}try check({fn_name}({}))", call.join(", "));
    for _ in &wraps {
        indent.truncate(indent.len() - 4);
        let _ = writeln!(s, "{indent}}}");
    }
    if !ret_value.is_empty() {
        let _ = writeln!(s, "        return {ret_value}");
    }
    s.push_str("    }\n");
}

pub fn generate(ops: &[OpSpec]) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table — do not edit;\n\
         // regenerate with `just gen-bindings`.\n\
         \n\
         import CRLX\n\
         \n\
         /// The graph builders — one per row of rlx-bind's op table, each calling\n\
         /// its `rlx_graph_<op>` C function.\n\
         public extension Graph {",
    );
    for op in ops {
        method(&mut s, op);
    }
    s.push_str("}\n");
    s
}
