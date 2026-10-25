// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-nim/src/rlx/generated.nim`: the whole C ABI as Nim, from the lowered
//! [`Ir`] (included by `src/rlx.nim`).
//!
//! The C functions are `dynlib` imports (`-d:rlxLib=<path>` overrides the
//! library), so a function a build lacks is only looked up if a program uses
//! it. A `ref` type per handle class, freed by its `=destroy` hook; methods
//! are procs on it (`g.softmax(x, axis = 1)`), constructors `newGraph(…)`,
//! statics `Gguf.open(path)`. Tensors are `float` sequences, options
//! `JsonNode`s.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines, pascal};
use crate::jni::c_params;

const KEYWORDS: &[&str] = &[
    "addr",
    "and",
    "as",
    "asm",
    "bind",
    "block",
    "break",
    "case",
    "cast",
    "concept",
    "const",
    "continue",
    "converter",
    "defer",
    "discard",
    "distinct",
    "div",
    "do",
    "elif",
    "else",
    "end",
    "enum",
    "except",
    "export",
    "finally",
    "for",
    "from",
    "func",
    "if",
    "import",
    "in",
    "include",
    "interface",
    "is",
    "isnot",
    "iterator",
    "let",
    "macro",
    "method",
    "mixin",
    "mod",
    "nil",
    "not",
    "notin",
    "object",
    "of",
    "or",
    "out",
    "proc",
    "ptr",
    "raise",
    "ref",
    "return",
    "shl",
    "shr",
    "static",
    "template",
    "try",
    "tuple",
    "type",
    "using",
    "var",
    "when",
    "while",
    "xor",
    "yield",
    "result",
];

/// A Nim identifier: camelCase; a keyword gets `suffix` (Nim identifiers may
/// not end in `_`).
fn ident(name: &str, suffix: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("{c}{suffix}")
    } else {
        c
    }
}

fn proc_name(f: &Func) -> String {
    ident(&f.name, "Op")
}

fn arg(name: &str) -> String {
    match ident(name, "Arg").as_str() {
        // the receiver and our temporaries
        "self" | "k" => format!("{}Arg", camel(name)),
        a => a.to_string(),
    }
}

/// `(Nim type, C type)` of a C parameter type.
fn cty(c: &str) -> &'static str {
    match c {
        "bool" => "bool",
        "float" => "cfloat",
        "double" => "cdouble",
        "int32_t" => "int32",
        "uint32_t" => "uint32",
        "int64_t" => "int64",
        "uint64_t" => "uint64",
        "size_t" => "csize_t",
        "const char*" => "cstring",
        _ => "pointer",
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "Option[string]",
        "json" | "value" => "JsonNode",
        "json?" | "value?" => "JsonNode",
        "bool" => "bool",
        "f32" | "f64" => "float",
        "i32" | "u32" | "node" | "i64" | "u64" | "usize" => "int",
        "node?" | "u64?" | "usize?" => "Option[int]",
        "f32?" => "Option[float]",
        "f32s" => "openArray[float]",
        "u8s" => "openArray[byte]",
        "u32s" | "nodes" | "usizes" | "i64s" => "openArray[int]",
        "usizes?" => "Option[seq[int]]",
        "rows2" => "openArray[seq[int]]",
        "frows2" | "frows6" => "openArray[seq[float]]",
        "strs" => "openArray[string]",
        "strs?" => "Option[seq[string]]",
        "inputs" => "openArray[(string, seq[float])]",
        "typed_inputs" => "openArray[(string, TypedTensor)]",
        "graph" | "graph&" => "Graph",
        "graph?" => "Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "Option[string]",
        "json" | "object" | "rows" => "JsonNode",
        "shapes" => "seq[seq[int]]",
        "shape" => "seq[int]",
        "strings" => "seq[string]",
        "f32s" => "seq[float]",
        "u8s" => "seq[byte]",
        "u32s" | "nodes" => "seq[int]",
        "outputs" => "seq[seq[float]]",
        "typed_outputs" => "seq[TypedTensor]",
        "tensors" => "Table[string, seq[float]]",
        "bool" => "bool",
        "usize" | "u64" | "u32" | "node" => "int",
        "f32" => "float",
        "f32?" => "Option[float]",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => match ty(kind).as_str() {
            "JsonNode" | "Graph" => "nil".into(),
            t if t.starts_with("Option[") => format!("none({})", &t[7..t.len() - 1]),
            t => format!("default({t})"),
        },
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => {
            let n = n.replace("inf", "Inf");
            if ty(kind) == "float" && !n.contains('.') && !n.contains('e') && !n.contains("Inf") {
                format!("{n}.0")
            } else {
                n
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let el = if ty(kind).contains("float") {
                "f32s"
            } else {
                "i32"
            };
            let inner: Vec<String> = items.iter().map(|i| literal(i, el)).collect();
            if inner.is_empty() {
                "[]".into()
            } else {
                format!("[{}]", inner.join(", "))
            }
        }
    }
}

/// Nim expressions for one parameter's C arguments (`k` keeps buffers alive).
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "str" => vec![format!("cstring({v})")],
        "str?" => vec![format!("optStr({v})")],
        "json" | "value" | "json?" | "value?" => vec![format!("k.json({v})")],
        "node?" => vec![format!("uint32({v}.get(NONE))")],
        "u64?" | "usize?" => vec![format!("k.boxU64({v})")],
        "f32?" => vec![format!("k.boxF32({v})")],
        "f32s" => vec![format!("k.floats({v})"), format!("csize_t({v}.len)")],
        "u8s" => vec![format!("k.bytes({v})"), format!("csize_t({v}.len)")],
        "u32s" | "nodes" => vec![format!("k.u32s({v})"), format!("csize_t({v}.len)")],
        "usizes" => vec![format!("k.u64s({v})"), format!("csize_t({v}.len)")],
        "i64s" => vec![format!("k.i64s({v})"), format!("csize_t({v}.len)")],
        "usizes?" => vec![
            format!("(if {v}.isSome: k.u64s({v}.get) else: nil)"),
            format!("csize_t(if {v}.isSome: {v}.get.len else: 0)"),
        ],
        "rows2" => vec![
            format!("k.u64s(rows({v}, 2, true))"),
            format!("csize_t({v}.len)"),
        ],
        "frows2" => vec![
            format!("k.floats(frows({v}, 2))"),
            format!("csize_t({v}.len)"),
        ],
        "frows6" => vec![
            format!("k.floats(frows({v}, 6))"),
            format!("csize_t({v}.len)"),
        ],
        "strs" => vec![format!("k.strs({v})"), format!("csize_t({v}.len)")],
        "strs?" => vec![
            format!("(if {v}.isSome: k.strs({v}.get) else: nil)"),
            format!("csize_t(if {v}.isSome: {v}.get.len else: 0)"),
        ],
        "inputs" | "typed_inputs" => {
            let n = format!("in{}", pascal(&p.name));
            let mut a = vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
            ];
            if p.kind == "typed_inputs" {
                a.push(format!("{n}.dtypes"));
            }
            a.push(format!("csize_t({n}.n)"));
            a
        }
        "graph?" => vec![format!("(if {v}.isNil: nil else: {v}.h)")],
        "bool" => vec![v.to_string()],
        "f32" => vec![format!("cfloat({v})")],
        "f64" => vec![format!("cdouble({v})")],
        "i32" => vec![format!("int32({v})")],
        "u32" | "node" => vec![format!("uint32({v})")],
        "i64" => vec![format!("int64({v})")],
        "u64" => vec![format!("uint64({v})")],
        "usize" => vec![format!("csize_t({v})")],
        _ => vec![format!("{v}.h")],
    }
}

/// `(declarations, C arguments, Nim result)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let decl = |t: &str| vec![format!("var {o}: {t}")];
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (decl("RlxSlice"), vec![format!("addr {o}")]),
        "bool" => (decl("bool"), vec![format!("addr {o}")]),
        "u32" | "node" => (decl("uint32"), vec![format!("addr {o}")]),
        "f32" => (decl("cfloat"), vec![format!("addr {o}")]),
        "usize" => (decl("csize_t"), vec![format!("addr {o}")]),
        "u64" => (decl("uint64"), vec![format!("addr {o}")]),
        "f32?" => (
            vec![format!("var {o}: cfloat"), format!("var {o}k: bool")],
            vec![format!("addr {o}"), format!("addr {o}k")],
        ),
        _ => (decl("pointer"), vec![format!("addr {o}")]),
    };
    let e = match kind {
        "str" => format!("takeString({o})"),
        "str?" => format!("takeStringOpt({o})"),
        "json" | "object" | "rows" => format!("parseJson(takeString({o}))"),
        "shapes" | "shape" | "strings" => {
            format!("to(parseJson(takeString({o})), {})", ret_ty(kind))
        }
        "f32s" => format!("takeF32s({o})"),
        "u8s" => format!("takeBytes({o})"),
        "u32s" | "nodes" => format!("takeU32s({o})"),
        "outputs" => format!("takeOutputs({o})"),
        "typed_outputs" => format!("takeTypedOutputs({o})"),
        "tensors" => format!("takeTensors({o})"),
        "bool" => o.clone(),
        "u32" | "node" | "usize" | "u64" => format!("int({o})"),
        "f32" => format!("float({o})"),
        "f32?" => format!("(if {o}k: some(float({o})) else: none(float))"),
        "graph" => format!("adopt(Graph, {o})"),
        class => format!("adopt({class}, {o})"),
    };
    (d, a, e)
}

fn import(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let ps: Vec<String> = params
        .iter()
        .enumerate()
        .map(|(i, (t, _))| format!("a{i}: {}", cty(t)))
        .collect();
    let ret = if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "cstring",
            _ => "bool",
        }
    } else {
        "int32"
    };
    let _ = writeln!(
        s,
        "proc c_{c}({}): {ret} {{.importc: \"{c}\", dynlib: rlxLib, cdecl.}}",
        ps.join(", "),
        c = f.c
    );
}

fn func(s: &mut String, ir: &Ir, f: &Func) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let mut sig: Vec<String> = Vec::new();
    if f.role == Role::Method {
        sig.push(format!("self: {}", class.clone().unwrap()));
    }
    if f.role == Role::Static {
        sig.push(format!("_: typedesc[{}]", class.clone().unwrap()));
    }
    for p in &f.params {
        let t = ty(&p.kind);
        if p.optional() {
            let d = match &p.default {
                Some(d) => literal(d, &p.kind),
                None => literal(&Lit::None, &p.kind),
            };
            sig.push(format!("{}: {t} = {d}", arg(&p.name)));
        } else {
            sig.push(format!("{}: {t}", arg(&p.name)));
        }
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let ret = match rets.as_slice() {
        [] => String::new(),
        [(_, k)] => format!(": {}", ret_ty(k)),
        _ => {
            let fields: Vec<String> = rets
                .iter()
                .map(|(n, k)| format!("{}: {}", ident(n, "Value"), ret_ty(k)))
                .collect();
            format!(": tuple[{}]", fields.join(", "))
        }
    };
    let name = match f.role {
        Role::Constructor => format!("new{}", class.clone().unwrap()),
        _ => proc_name(f),
    };
    let quoted = if name.chars().next().is_some_and(char::is_alphabetic) {
        name.clone()
    } else {
        format!("`{name}`")
    };
    let _ = writeln!(s, "proc {quoted}*({}){ret} =", sig.join("; "));
    s.push_str(&doc_lines(&f.doc, "  ## "));
    let b = "  ";
    let uses_keep = f.params.iter().any(|p| {
        !matches!(
            p.kind.as_str(),
            "str"
                | "str?"
                | "node?"
                | "bool"
                | "f32"
                | "f64"
                | "i32"
                | "u32"
                | "node"
                | "i64"
                | "u64"
                | "usize"
                | "graph?"
        ) && !p.kind.starts_with(char::is_uppercase)
            && !matches!(p.kind.as_str(), "graph" | "graph&")
    });
    if uses_keep {
        let _ = writeln!(s, "{b}var k: Keep");
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("self.h".to_string());
    }
    for p in &f.params {
        let v = arg(&p.name);
        if p.kind == "inputs" || p.kind == "typed_inputs" {
            let helper = if p.kind == "inputs" {
                "inputs"
            } else {
                "typedInputs"
            };
            let _ = writeln!(s, "{b}let in{} = k.{helper}({v})", pascal(&p.name));
        }
        args.extend(lower(p, &v));
    }
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "{b}$c_{}({})\n", f.c, args.join(", "));
            }
            _ => {
                let _ = writeln!(s, "{b}c_{}({})\n", f.c, args.join(", "));
            }
        }
        return;
    }
    let mut exprs = Vec::new();
    for (i, (_, k)) in rets.iter().enumerate() {
        let k = if f.role == Role::Constructor {
            "handle"
        } else {
            k.as_str()
        };
        let (d, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "{b}{line}");
        }
        args.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(s, "{b}check c_{}({})", f.c, args.join(", "));
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{b}adopt({}, o0)", class.unwrap());
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{b}{}", exprs[0]);
        }
        _ => {
            let fields: Vec<String> = rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{}: {e}", ident(n, "Value")))
                .collect();
            let _ = writeln!(s, "{b}({})", fields.join(", "));
        }
    }
    s.push('\n');
}

pub fn generate(ir: &Ir) -> String {
    let mut s = String::from(
        "# RLX — versatile ML compiler + runtime.\n\
         # Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         # SPDX-License-Identifier: MIT OR Apache-2.0\n\
         #\n\
         # @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         # do not edit; regenerate with `just gen-bindings`. Included by rlx.nim.\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "proc c_{free}(h: pointer) {{.importc: \"{free}\", dynlib: rlxLib, cdecl.}}",
            free = c.c_free()
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        import(&mut s, ir, f);
    }
    s.push_str("\ntype\n");
    for c in &ir.classes {
        let _ = writeln!(s, "  {n}Obj = object\n    h: pointer", n = c.name);
        s.push_str(&doc_lines(&c.doc, "  ## "));
        let _ = writeln!(s, "  {n}* = ref {n}Obj", n = c.name);
    }
    s.push('\n');
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "proc `=destroy`(x: {n}Obj) =\n  if x.h != nil: c_{free}(x.h)\nproc `=copy`(dst: var {n}Obj; src: {n}Obj) {{.error.}}\n\
             proc close*(x: {n}) =\n  ## Free the handle now (it is unusable afterwards).\n  if x.h != nil:\n    c_{free}(x.h)\n    x.h = nil\n\
             proc adopt(_: typedesc[{n}]; h: pointer): {n} =\n  if h == nil: raise newException(RlxError, \"rlx returned NULL\")\n  {n}(h: h)\n",
            n = c.name,
            free = c.c_free()
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        func(&mut s, ir, f);
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
