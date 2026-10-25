// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-v/generated.v`: the whole C ABI as V (module `rlx`), from the lowered
//! [`Ir`].
//!
//! A heap struct per handle type (V's GC runs no destructors, so `close()`
//! frees it); constructors and statics are static methods (`Graph.new()`,
//! `Gguf.open(path)`). Every call returns a result (`!T`) whose error is an
//! `RlxError`. Optional parameters are a `@[params]` struct
//! (`g.softmax(x, axis: 1)`); options go in as JSON text and reports come
//! back as `json2.Any`. An optional result is `''` / NaN when absent (V has
//! no result of an option). A function behind a feature the default library lacks
//! needs `-d rlx_<feature>`.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines, pascal};
use crate::jni::c_params;

const KEYWORDS: &[&str] = &[
    "as",
    "asm",
    "assert",
    "atomic",
    "break",
    "const",
    "continue",
    "defer",
    "else",
    "enum",
    "false",
    "fn",
    "for",
    "go",
    "goto",
    "if",
    "import",
    "in",
    "interface",
    "is",
    "isreftype",
    "lock",
    "match",
    "module",
    "mut",
    "none",
    "or",
    "pub",
    "return",
    "rlock",
    "select",
    "shared",
    "sizeof",
    "spawn",
    "static",
    "struct",
    "true",
    "type",
    "typeof",
    "union",
    "unsafe",
    "volatile",
    "chan",
    "nil",
];

fn ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

fn method_name(f: &Func) -> String {
    match f.name.as_str() {
        "close" | "str" | "free" => format!("{}_", f.name),
        n => ident(n),
    }
}

fn arg(name: &str) -> String {
    match name {
        "o" | "k" | "rc" | "opts" => format!("{name}_"),
        n => ident(n),
    }
}

fn cty(c: &str) -> &'static str {
    match c {
        "bool" => "bool",
        "float" => "f32",
        "double" => "f64",
        "int32_t" => "i32",
        "uint32_t" => "u32",
        "int64_t" => "i64",
        "uint64_t" => "u64",
        "size_t" => "usize",
        "const char*" => "&char",
        _ => "voidptr",
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" | "json?" | "value?" => "?string",
        "json" | "value" => "string",
        "bool" => "bool",
        "f32" | "f64" => "f64",
        "i32" | "u32" | "node" | "i64" | "u64" | "usize" => "int",
        "node?" | "u64?" | "usize?" => "?int",
        "f32?" => "?f64",
        "f32s" => "[]f64",
        "u8s" => "[]u8",
        "u32s" | "nodes" | "usizes" | "i64s" => "[]int",
        "usizes?" => "?[]int",
        "rows2" => "[][]int",
        "frows2" | "frows6" => "[][]f64",
        "strs" => "[]string",
        "strs?" => "?[]string",
        "inputs" => "map[string][]f64",
        "typed_inputs" => "map[string]TypedTensor",
        "graph" | "graph&" => "&Graph",
        "graph?" => "?&Graph",
        other => return format!("&{}", other.trim_end_matches('&')),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        // V has no result-of-option: none is '' / NaN.
        "str" | "str?" => "string",
        "json" | "object" | "rows" => "json2.Any",
        "shapes" => "[][]int",
        "shape" => "[]int",
        "strings" => "[]string",
        "f32s" => "[]f64",
        "u8s" => "[]u8",
        "u32s" | "nodes" => "[]int",
        "outputs" => "[][]f64",
        "typed_outputs" => "[]TypedTensor",
        "tensors" => "map[string][]f64",
        "bool" => "bool",
        "usize" | "u64" | "u32" | "node" => "int",
        "f32" | "f32?" => "f64",
        "graph" => "&Graph",
        other => return format!("&{other}"),
    }
    .to_string()
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "none".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "math.inf(1)"),
        Lit::Str(s) => format!("'{s}'"),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// `(statements, C arguments)` for one parameter.
fn lower(p: &Param, v: &str) -> (Vec<String>, Vec<String>) {
    let t = format!("a_{}", p.name);
    let one = |e: String| (vec![], vec![e]);
    let tmp = |init: String, ptr: String, n: String| (vec![format!("{t} := {init}")], vec![ptr, n]);
    match p.kind.as_str() {
        "str" | "json" | "value" => one(format!("&char({v}.str)")),
        "str?" | "json?" | "value?" => (
            vec![format!("{t} := {v} or {{ '' }}")],
            vec![format!(
                "if {v} == none {{ &char(unsafe {{ nil }}) }} else {{ &char({t}.str) }}"
            )],
        ),
        "node?" => one(format!("u32({v} or {{ int(none_node) }})")),
        "u64?" | "usize?" => (
            vec![format!("mut {t} := u64({v} or {{ 0 }})")],
            vec![format!(
                "if {v} == none {{ unsafe {{ nil }} }} else {{ voidptr(&{t}) }}"
            )],
        ),
        "f32?" => (
            vec![format!("mut {t} := f32({v} or {{ 0 }})")],
            vec![format!(
                "if {v} == none {{ unsafe {{ nil }} }} else {{ voidptr(&{t}) }}"
            )],
        ),
        "f32s" => tmp(
            format!("to_f32s({v})"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "u8s" => (
            vec![],
            vec![format!("ptr_of({v})"), format!("usize({v}.len)")],
        ),
        "u32s" | "nodes" => tmp(
            format!("to_u32s({v})"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "usizes" => tmp(
            format!("to_u64s({v})"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "i64s" => tmp(
            format!("to_i64s({v})"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "usizes?" => (
            vec![format!("{t} := to_u64s({v} or {{ []int{{}} }})")],
            vec![
                format!("if {v} == none {{ unsafe {{ nil }} }} else {{ ptr_of({t}) }}"),
                format!("usize({t}.len)"),
            ],
        ),
        "rows2" => tmp(
            format!("to_u64s(rows({v}, 2, true)!)"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "frows2" => tmp(
            format!("to_f32s(frows({v}, 2)!)"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "frows6" => tmp(
            format!("to_f32s(frows({v}, 6)!)"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "strs" => tmp(
            format!("cstrs({v})"),
            format!("ptr_of({t})"),
            format!("usize({v}.len)"),
        ),
        "strs?" => (
            vec![format!("{t} := cstrs({v} or {{ []string{{}} }})")],
            vec![format!("ptr_of({t})"), format!("usize({t}.len)")],
        ),
        "inputs" | "typed_inputs" => {
            let f = if p.kind == "inputs" {
                "new_inputs"
            } else {
                "new_typed_inputs"
            };
            let mut a = vec![
                format!("ptr_of({t}.names)"),
                format!("ptr_of({t}.data)"),
                format!("ptr_of({t}.lens)"),
            ];
            if p.kind == "typed_inputs" {
                a.push(format!("ptr_of({t}.dtypes)"));
            }
            a.push(format!("usize({t}.names.len)"));
            (vec![format!("{t} := {f}({v})")], a)
        }
        "graph?" => (
            vec![],
            vec![format!(
                "if g_ := {v} {{ g_.h }} else {{ unsafe {{ nil }} }}"
            )],
        ),
        "bool" => one(v.to_string()),
        "f32" => one(format!("f32({v})")),
        "f64" => one(v.to_string()),
        "i32" => one(format!("i32({v})")),
        "u32" | "node" => one(format!("u32({v})")),
        "i64" => one(format!("i64({v})")),
        "u64" => one(format!("u64({v})")),
        "usize" => one(format!("usize({v})")),
        _ => one(format!("{v}.h")),
    }
}

/// `(declarations, C arguments, V result)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let decl = |init: &str| vec![format!("mut {o} := {init}")];
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (decl("Slice{}"), vec![format!("voidptr(&{o})")]),
        "bool" => (decl("false"), vec![format!("voidptr(&{o})")]),
        "u32" | "node" => (decl("u32(0)"), vec![format!("voidptr(&{o})")]),
        "f32" => (decl("f32(0)"), vec![format!("voidptr(&{o})")]),
        "usize" => (decl("usize(0)"), vec![format!("voidptr(&{o})")]),
        "u64" => (decl("u64(0)"), vec![format!("voidptr(&{o})")]),
        "f32?" => (
            vec![format!("mut {o} := f32(0)"), format!("mut {o}k := false")],
            vec![format!("voidptr(&{o})"), format!("voidptr(&{o}k)")],
        ),
        _ => (
            decl("voidptr(unsafe { nil })"),
            vec![format!("voidptr(&{o})")],
        ),
    };
    let e = match kind {
        "str" => format!("take_string({o})"),
        "str?" => format!("take_string({o})"),
        "json" | "object" | "rows" => format!("json2.decode[json2.Any](take_string({o}))!"),
        "shapes" | "shape" | "strings" => {
            format!("json2.decode[{}](take_string({o}))!", ret_ty(kind))
        }
        "f32s" => format!("take_f32s(mut {o})"),
        "u8s" => format!("take_bytes(mut {o})"),
        "u32s" | "nodes" => format!("take_u32s(mut {o})"),
        "outputs" => format!("take_outputs({o})"),
        "typed_outputs" => format!("take_typed_outputs({o})"),
        "tensors" => format!("take_tensors({o})"),
        "bool" => o.clone(),
        "u32" | "node" | "usize" | "u64" => format!("int({o})"),
        "f32" => format!("f64({o})"),
        "f32?" => format!("if {o}k {{ f64({o}) }} else {{ math.nan() }}"),
        "graph" => format!("adopt_graph({o})!"),
        class => format!("adopt_{}({o})!", class.to_ascii_lowercase()),
    };
    (d, a, e)
}

fn gate(f: &Func, defaults: &[String]) -> Option<String> {
    f.feature
        .as_ref()
        .filter(|x| !defaults.contains(x))
        .map(|x| format!("rlx_{}", x.replace('-', "_")))
}

fn func(s: &mut String, ir: &Ir, f: &Func, defaults: &[String]) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let stem = format!(
        "{}{}",
        class.clone().unwrap_or_default(),
        if f.role == Role::Constructor {
            "New".to_string()
        } else {
            pascal(&f.name)
        }
    );
    let opts = format!("{stem}Opts");
    if !optional.is_empty() {
        let _ = writeln!(s, "@[params]\npub struct {opts} {{\npub:");
        for p in &optional {
            let t = ty(&p.kind);
            match &p.default {
                Some(Lit::None) | None => {
                    let t = if t.starts_with('?') {
                        t
                    } else {
                        format!("?{t}")
                    };
                    let _ = writeln!(s, "\t{} {t}", ident(&p.name));
                }
                Some(d) => {
                    // V zeroes fields and warns on a zero default.
                    let lit = literal(d);
                    if matches!(lit.as_str(), "false" | "0" | "''" | "[]") {
                        let _ = writeln!(s, "\t{} {t}", ident(&p.name));
                    } else {
                        let _ = writeln!(s, "\t{} {t} = {lit}", ident(&p.name));
                    }
                }
            }
        }
        s.push_str("}\n\n");
    }
    let mut sig: Vec<String> = required
        .iter()
        .map(|p| format!("{} {}", arg(&p.name), ty(&p.kind)))
        .collect();
    if !optional.is_empty() {
        sig.push(format!("o {opts}"));
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let result = format!("{stem}Result");
    if rets.len() > 1 {
        let _ = writeln!(s, "pub struct {result} {{\npub:");
        for (n, k) in &rets {
            let _ = writeln!(s, "\t{} {}", ident(n), ret_ty(k));
        }
        s.push_str("}\n\n");
    }
    let ret = match rets.as_slice() {
        [] => "!".to_string(),
        [(_, k)] if f.direct => ret_ty(k),
        [(_, k)] => format!("!{}", ret_ty(k)),
        _ => format!("!{result}"),
    };
    s.push_str(&doc_lines(&f.doc, "// "));
    let head = match (f.role, &class) {
        (Role::Constructor, Some(c)) => format!("pub fn {c}.new({}) {ret}", sig.join(", ")),
        (Role::Static, Some(c)) => {
            format!("pub fn {c}.{}({}) {ret}", method_name(f), sig.join(", "))
        }
        (Role::Method, Some(c)) => format!(
            "pub fn (self &{c}) {}({}) {ret}",
            method_name(f),
            sig.join(", ")
        ),
        _ => format!("pub fn {}({}) {ret}", ident(&f.name), sig.join(", ")),
    };
    let _ = writeln!(s, "{head} {{");
    let g = gate(f, defaults);
    let ind = if g.is_some() { "\t\t" } else { "\t" };
    if let Some(g) = &g {
        let _ = writeln!(s, "\t$if {g} ? {{");
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("self.h".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("o.{}", ident(&p.name))
        } else {
            arg(&p.name)
        };
        let (stmts, a) = lower(p, &v);
        for st in stmts {
            let _ = writeln!(s, "{ind}{st}");
        }
        args.extend(a);
    }
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(
                    s,
                    "{ind}return unsafe {{ cstring_to_vstring(C.{}({})) }}",
                    f.c,
                    args.join(", ")
                );
            }
            _ => {
                let _ = writeln!(s, "{ind}return C.{}({})", f.c, args.join(", "));
            }
        }
    } else {
        let mut exprs = Vec::new();
        for (i, (_, k)) in rets.iter().enumerate() {
            let (d, a, e) = outs(k, i);
            for line in d {
                let _ = writeln!(s, "{ind}{line}");
            }
            args.extend(a);
            exprs.push(e);
        }
        let _ = writeln!(s, "{ind}check_rc(C.{}({}))!", f.c, args.join(", "));
        match exprs.len() {
            0 => {}
            1 => {
                let _ = writeln!(s, "{ind}return {}", exprs[0]);
            }
            _ => {
                let fields: Vec<String> = rets
                    .iter()
                    .zip(&exprs)
                    .map(|((n, _), e)| format!("{}: {e}", ident(n)))
                    .collect();
                let _ = writeln!(
                    s,
                    "{ind}return {result}{{\n{ind}\t{}\n{ind}}}",
                    fields.join(&format!("\n{ind}\t"))
                );
            }
        }
    }
    if let Some(g) = &g {
        let _ = writeln!(
            s,
            "\t}} $else {{\n\t\treturn RlxError{{\n\t\t\tkind: 'not_found'\n\t\t\tmessage: '{} needs a build with -d {g}'\n\t\t}}\n\t}}",
            f.c
        );
    }
    s.push_str("}\n\n");
}

pub fn generate(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         module rlx\n\
         \n\
         import x.json2\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(s, "fn C.{}(h voidptr)", c.c_free());
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let ps: Vec<String> = c_params(ir, f)
            .iter()
            .enumerate()
            .map(|(i, (t, _))| format!("a{i} {}", cty(t)))
            .collect();
        let ret = if f.direct {
            match f.rets.first().map(|(_, k)| k.as_str()) {
                Some("str") => " &char",
                _ => " bool",
            }
        } else {
            " i32"
        };
        let _ = writeln!(s, "fn C.{}({}){ret}", f.c, ps.join(", "));
    }
    s.push('\n');
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "// "));
        let _ = writeln!(
            s,
            "@[heap]\npub struct {n} {{\nmut:\n\th voidptr\n}}\n\n\
             // close frees the handle now (V's GC runs no destructors).\n\
             pub fn (mut self {n}) close() {{\n\tif self.h != unsafe {{ nil }} {{\n\t\tC.{free}(self.h)\n\t\tself.h = unsafe {{ nil }}\n\t}}\n}}\n\n\
             fn adopt_{l}(h voidptr) !&{n} {{\n\tif h == unsafe {{ nil }} {{\n\t\treturn RlxError{{\n\t\t\tkind: 'runtime'\n\t\t\tmessage: 'rlx returned NULL'\n\t\t}}\n\t}}\n\treturn &{n}{{\n\t\th: h\n\t}}\n}}\n",
            n = c.name,
            l = c.name.to_ascii_lowercase(),
            free = c.c_free()
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        func(&mut s, ir, f, defaults);
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    // V rejects an unused import.
    if s.contains("math.") {
        s = s.replacen("import x.json2\n", "import math\nimport x.json2\n", 1);
    }
    s
}
