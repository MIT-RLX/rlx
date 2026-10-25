// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The D binding, from the lowered [`Ir`]: two modules of package `rlx`.
//!
//! * `rlx-d/source/rlx/capi.d` — every rlx-capi function as an `extern (C)`
//!   declaration (feature-gated ones under `version (rlx_<feature>)`).
//! * `rlx-d/source/rlx/generated.d` — the API: a class per handle type (freed
//!   by its destructor, or early with `close()`), free functions at module
//!   level, optional parameters as defaults (named arguments work:
//!   `g.softmax(x, axis: 1)`). An optional number with no default is a
//!   sentinel: `NONE` for a node, `T.max` for a count, NaN for a float.
//!   Arrays pass as slices, without copying.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines};
use crate::jni::c_params;

const KEYWORDS: &[&str] = &[
    "abstract",
    "alias",
    "align",
    "asm",
    "assert",
    "auto",
    "body",
    "bool",
    "break",
    "byte",
    "case",
    "cast",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "dchar",
    "debug",
    "default",
    "delegate",
    "delete",
    "deprecated",
    "do",
    "double",
    "else",
    "enum",
    "export",
    "extern",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "foreach",
    "function",
    "goto",
    "if",
    "immutable",
    "import",
    "in",
    "inout",
    "int",
    "interface",
    "invariant",
    "is",
    "lazy",
    "long",
    "macro",
    "mixin",
    "module",
    "new",
    "nothrow",
    "null",
    "out",
    "override",
    "package",
    "pragma",
    "private",
    "protected",
    "public",
    "pure",
    "real",
    "ref",
    "return",
    "scope",
    "shared",
    "short",
    "static",
    "struct",
    "super",
    "switch",
    "synchronized",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typeid",
    "typeof",
    "ubyte",
    "uint",
    "ulong",
    "union",
    "unittest",
    "ushort",
    "version",
    "void",
    "wchar",
    "while",
    "with",
];

/// `Object`'s members and our own.
const MEMBERS: &[&str] = &[
    "toString", "toHash", "opEquals", "opCmp", "factory", "close", "h",
];

fn ident(name: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("{c}_")
    } else {
        c
    }
}

fn method_name(f: &Func) -> String {
    let c = ident(&f.name);
    if f.class.is_some() && MEMBERS.contains(&c.as_str()) {
        format!("{c}_")
    } else {
        c
    }
}

fn arg(name: &str) -> String {
    match ident(name).as_str() {
        "k" | "rc" => format!("{}_", camel(name)),
        a => a.to_string(),
    }
}

fn cty(c: &str) -> String {
    match c {
        "bool" => "bool",
        "float" => "float",
        "double" => "double",
        "int32_t" => "int",
        "uint32_t" => "uint",
        "int64_t" => "long",
        "uint64_t" => "ulong",
        "size_t" => "size_t",
        "const char*" => "const(char)*",
        c if c.starts_with("const ") => "const(void)*",
        _ => "void*",
    }
    .to_string()
}

fn ty(kind: &str) -> String {
    match kind {
        "str" | "str?" => "string",
        "json" | "value" | "json?" | "value?" => "JSONValue",
        "bool" => "bool",
        "f32" => "float",
        "f64" => "double",
        "i32" => "int",
        "u32" | "node" | "node?" => "uint",
        "i64" => "long",
        "u64" | "u64?" => "ulong",
        "usize" | "usize?" => "size_t",
        "f32?" => "float",
        "f32s" => "const(float)[]",
        "u8s" => "const(ubyte)[]",
        "u32s" | "nodes" => "const(uint)[]",
        "usizes" | "usizes?" => "const(size_t)[]",
        "i64s" => "const(long)[]",
        "rows2" => "const(size_t[])[]",
        "frows2" | "frows6" => "const(float[])[]",
        "strs" | "strs?" => "const(string)[]",
        "inputs" => "const(float[])[string]",
        "typed_inputs" => "const(TypedTensor)[string]",
        "graph" | "graph&" | "graph?" => "Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" | "str?" => "string",
        "json" | "object" | "rows" => "JSONValue",
        "shapes" => "size_t[][]",
        "shape" => "size_t[]",
        "strings" => "string[]",
        "f32s" => "float[]",
        "u8s" => "ubyte[]",
        "u32s" | "nodes" => "uint[]",
        "outputs" => "float[][]",
        "typed_outputs" => "TypedTensor[]",
        "tensors" => "float[][string]",
        "bool" => "bool",
        "usize" => "size_t",
        "u64" => "ulong",
        "u32" | "node" => "uint",
        "f32" => "float",
        "f32?" => "Nullable!float",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => match kind {
            "node?" => "NONE".into(),
            "u64?" => "ulong.max".into(),
            "usize?" => "size_t.max".into(),
            "f32?" => "float.nan".into(),
            k if k.starts_with("json") || k.starts_with("value") => "JSONValue.init".into(),
            _ => "null".into(),
        },
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "float.infinity"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(|i| literal(i, "f32")).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// `(statements, C arguments)` for one parameter.
fn lower(p: &Param, v: &str) -> (Vec<String>, Vec<String>) {
    let t = format!("a_{}", p.name);
    let one = |e: String| (vec![], vec![e]);
    match p.kind.as_str() {
        "str" => one(format!("{v}.toStringz")),
        "str?" => one(format!("({v} is null ? null : {v}.toStringz)")),
        "json" | "value" | "json?" | "value?" => one(format!("jsonArg({v})")),
        "u64?" => one(format!("box({v}, ulong.max)")),
        "usize?" => one(format!("box({v}, size_t.max)")),
        "f32?" => one(format!("boxF({v})")),
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "usizes?" | "i64s" => {
            (vec![], vec![format!("{v}.ptr"), format!("{v}.length")])
        }
        "rows2" => (
            vec![format!("auto {t} = rows({v}, 2, true);")],
            vec![format!("{t}.ptr"), format!("{v}.length")],
        ),
        "frows2" | "frows6" => {
            let w = if p.kind == "frows2" { 2 } else { 6 };
            (
                vec![format!("auto {t} = frows({v}, {w});")],
                vec![format!("{t}.ptr"), format!("{v}.length")],
            )
        }
        "strs" | "strs?" => (
            vec![format!("auto {t} = cstrs({v});")],
            vec![format!("{t}.ptr"), format!("{v}.length")],
        ),
        "inputs" | "typed_inputs" => {
            let st = if p.kind == "inputs" {
                "Inputs"
            } else {
                "TypedInputs"
            };
            let mut a = vec![
                format!("{t}.names.ptr"),
                format!("{t}.data.ptr"),
                format!("{t}.lens.ptr"),
            ];
            if p.kind == "typed_inputs" {
                a.push(format!("{t}.dtypes.ptr"));
            }
            a.push(format!("{t}.names.length"));
            (vec![format!("auto {t} = {st}({v});")], a)
        }
        "graph?" => one(format!("({v} is null ? null : {v}.h)")),
        "bool" | "f32" | "f64" | "i32" | "u32" | "node" | "node?" | "i64" | "u64" | "usize" => {
            one(v.to_string())
        }
        _ => one(format!("{v}.h")),
    }
}

/// `(declarations, C arguments, D result)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let decl = |t: &str| vec![format!("{t} {o};")];
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (decl("RlxSlice"), vec![format!("&{o}")]),
        "bool" => (decl("bool"), vec![format!("&{o}")]),
        "u32" | "node" => (decl("uint"), vec![format!("&{o}")]),
        "f32" => (decl("float"), vec![format!("&{o}")]),
        "usize" => (decl("size_t"), vec![format!("&{o}")]),
        "u64" => (decl("ulong"), vec![format!("&{o}")]),
        "f32?" => (
            vec![format!("float {o};"), format!("bool {o}k;")],
            vec![format!("&{o}"), format!("&{o}k")],
        ),
        _ => (decl("void*"), vec![format!("&{o}")]),
    };
    let e = match kind {
        "str" => format!("takeString({o})"),
        "str?" => format!("takeStringOrNull({o})"),
        "json" | "object" | "rows" => format!("parseJSON(takeString({o}))"),
        "shapes" | "shape" | "strings" => format!("fromJson!({})(takeString({o}))", ret_ty(kind)),
        "f32s" => format!("takeF32s({o})"),
        "u8s" => format!("takeBytes({o})"),
        "u32s" | "nodes" => format!("takeU32s({o})"),
        "outputs" => format!("takeOutputs({o})"),
        "typed_outputs" => format!("takeTypedOutputs({o})"),
        "tensors" => format!("takeTensors({o})"),
        "f32?" => format!("({o}k ? nullable({o}) : Nullable!float.init)"),
        "graph" => format!("new Graph({o}, adopt)"),
        k if k.starts_with(char::is_uppercase) => format!("new {k}({o}, adopt)"),
        _ => o.clone(),
    };
    (d, a, e)
}

fn gate(f: &Func, defaults: &[String]) -> Option<String> {
    f.feature
        .as_ref()
        .filter(|x| !defaults.contains(x))
        .map(|x| format!("rlx_{}", x.replace('-', "_")))
}

/// `capi.d`.
pub fn capi(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         \n\
         /// Every rlx-capi function, as declared in rlx.h.\n\
         module rlx.capi;\n\
         \n\
         extern (C) nothrow @nogc:\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(s, "void {}(void* h);", c.c_free());
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let ps: Vec<String> = c_params(ir, f)
            .iter()
            .enumerate()
            .map(|(i, (t, _))| format!("{} a{i}", cty(t)))
            .collect();
        let ret = if f.direct {
            match f.rets.first().map(|(_, k)| k.as_str()) {
                Some("str") => "const(char)*",
                _ => "bool",
            }
        } else {
            "int"
        };
        let decl = format!("{ret} {}({});", f.c, ps.join(", "));
        match gate(f, defaults) {
            Some(g) => {
                let _ = writeln!(s, "version ({g}) {decl}");
            }
            None => {
                let _ = writeln!(s, "{decl}");
            }
        }
    }
    s
}

fn func(s: &mut String, f: &Func, defaults: &[String], indent: &str) {
    let mut sig: Vec<String> = Vec::new();
    for p in &f.params {
        let t = ty(&p.kind);
        if p.optional() {
            let d = match &p.default {
                Some(d) => literal(d, &p.kind),
                None => literal(&Lit::None, &p.kind),
            };
            sig.push(format!("{t} {} = {d}", arg(&p.name)));
        } else {
            sig.push(format!("{t} {}", arg(&p.name)));
        }
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![],
        _ => f.rets.clone(),
    };
    let ret = match rets.as_slice() {
        [] => "void".to_string(),
        [(_, k)] => ret_ty(k),
        _ => {
            let fields: Vec<String> = rets
                .iter()
                .map(|(n, k)| format!("{}, \"{}\"", ret_ty(k), ident(n)))
                .collect();
            format!("Tuple!({})", fields.join(", "))
        }
    };
    let g = gate(f, defaults);
    let ind = if g.is_some() {
        format!("{indent}    ")
    } else {
        indent.to_string()
    };
    if let Some(g) = &g {
        let _ = writeln!(s, "{indent}version ({g}) {{");
    }
    s.push_str(&doc_lines(&f.doc, &format!("{ind}/// ")));
    let head = match f.role {
        Role::Constructor => format!("this({})", sig.join(", ")),
        Role::Static => format!("static {ret} {}({})", method_name(f), sig.join(", ")),
        _ => format!("{ret} {}({})", method_name(f), sig.join(", ")),
    };
    let _ = writeln!(s, "{ind}{head}\n{ind}{{");
    let b = format!("{ind}    ");
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("h".to_string());
    }
    for p in &f.params {
        let (stmts, a) = lower(p, &arg(&p.name));
        for st in stmts {
            let _ = writeln!(s, "{b}{st}");
        }
        args.extend(a);
    }
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(
                    s,
                    "{b}return capi.{}({}).fromStringz.idup;",
                    f.c,
                    args.join(", ")
                );
            }
            _ => {
                let _ = writeln!(s, "{b}return capi.{}({});", f.c, args.join(", "));
            }
        }
    } else {
        let all: Vec<(String, String)> = match f.role {
            Role::Constructor => vec![("handle".into(), "handle".into())],
            _ => f.rets.clone(),
        };
        let mut exprs = Vec::new();
        for (i, (_, k)) in all.iter().enumerate() {
            let (d, a, e) = outs(k, i);
            for line in d {
                let _ = writeln!(s, "{b}{line}");
            }
            args.extend(a);
            exprs.push(e);
        }
        let _ = writeln!(s, "{b}checkRc(capi.{}({}));", f.c, args.join(", "));
        match (f.role, exprs.len()) {
            (Role::Constructor, _) => {
                let _ = writeln!(s, "{b}this(o0, adopt);");
            }
            (_, 0) => {}
            (_, 1) => {
                let _ = writeln!(s, "{b}return {};", exprs[0]);
            }
            _ => {
                let _ = writeln!(s, "{b}return {}({});", ret, exprs.join(", "));
            }
        }
    }
    let _ = writeln!(s, "{ind}}}");
    if g.is_some() {
        let _ = writeln!(s, "{indent}}}");
    }
    s.push('\n');
}

/// `generated.d`.
pub fn generate(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         \n\
         /// The rlx API: every C function, over rlx.support.\n\
         module rlx.generated;\n\
         \n\
         import std.json : JSONValue, parseJSON;\n\
         import std.string : fromStringz, toStringz;\n\
         import std.typecons : Nullable, Tuple, nullable;\n\
         \n\
         import capi = rlx.capi;\n\
         import rlx.support;\n\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, f, defaults, "");
    }
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "/// "));
        let _ = writeln!(
            s,
            "final class {n} : Handle\n{{\n    package this(void* h, Adopt) {{ super(h, &capi.{free}); }}\n",
            n = c.name,
            free = c.c_free()
        );
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut s, f, defaults, "    ");
        }
        s.truncate(s.trim_end().len());
        s.push_str("\n}\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
