// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-julia/src/generated.jl`: the whole C ABI as Julia, from the lowered
//! [`Ir`].
//!
//! One `mutable struct` per handle type (freed by a finalizer); every function
//! a method dispatching on its handle (`matmul(g, a, b)`, `run(compiled,
//! inputs)`), statics on the type (`from_env(Runner, g)`). Optional parameters
//! are keyword arguments with the tables' defaults. Options go in as any
//! Julia value (encoded to JSON), reports come back parsed.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};

const KEYWORDS: &[&str] = &[
    "baremodule",
    "begin",
    "break",
    "catch",
    "const",
    "continue",
    "do",
    "else",
    "elseif",
    "end",
    "export",
    "false",
    "finally",
    "for",
    "function",
    "global",
    "if",
    "import",
    "let",
    "local",
    "macro",
    "module",
    "quote",
    "return",
    "struct",
    "true",
    "try",
    "using",
    "while",
    "where",
    "in",
    "isa",
    "mutable",
    "abstract",
    "primitive",
    "type",
    "outer",
];

fn ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("var\"{name}\"")
    } else {
        name.to_string()
    }
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "nothing".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("f32::INFINITY", "Inf").replace("inf", "Inf"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// `(ccall argument types, argument expressions)` for one parameter, with
/// statements to run first and variables the call must keep rooted.
fn lower(
    p: &Param,
    v: &str,
    pre: &mut Vec<String>,
    keep: &mut Vec<String>,
) -> (Vec<String>, Vec<String>) {
    let n = format!("_a_{}", p.name);
    let t = |s: &str| s.to_string();
    match p.kind.as_str() {
        "str" => (vec![t("Ptr{UInt8}")], vec![format!("_cstr({v})")]),
        "str?" => (vec![t("Ptr{UInt8}")], vec![format!("_cstr_or_null({v})")]),
        "json" | "value" => (vec![t("Ptr{UInt8}")], vec![format!("_cstr(_json({v}))")]),
        "json?" | "value?" => (vec![t("Ptr{UInt8}")], vec![format!("_json_or_null({v})")]),
        "bool" => (vec![t("Bool")], vec![v.into()]),
        "f32" => (vec![t("Cfloat")], vec![v.into()]),
        "f64" => (vec![t("Cdouble")], vec![v.into()]),
        "i32" => (vec![t("Int32")], vec![v.into()]),
        "i64" => (vec![t("Int64")], vec![v.into()]),
        "u32" | "node" => (vec![t("UInt32")], vec![v.into()]),
        "node?" => (
            vec![t("UInt32")],
            vec![format!("({v} === nothing ? NONE : {v})")],
        ),
        "u64" => (vec![t("UInt64")], vec![v.into()]),
        "usize" => (vec![t("Csize_t")], vec![v.into()]),
        "u64?" => (
            vec![t("Ptr{UInt64}")],
            vec![format!("_ref_or_null(UInt64, {v})")],
        ),
        "usize?" => (
            vec![t("Ptr{Csize_t}")],
            vec![format!("_ref_or_null(Csize_t, {v})")],
        ),
        "f32?" => (
            vec![t("Ptr{Cfloat}")],
            vec![format!("_ref_or_null(Cfloat, {v})")],
        ),
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            let el = match p.kind.as_str() {
                "f32s" => "Cfloat",
                "u8s" => "UInt8",
                "u32s" | "nodes" => "UInt32",
                "usizes" => "Csize_t",
                _ => "Int64",
            };
            pre.push(format!("{n} = collect({el}, {v})"));
            (
                vec![format!("Ptr{{{el}}}"), t("Csize_t")],
                vec![n.clone(), format!("length({n})")],
            )
        }
        "usizes?" => {
            pre.push(format!(
                "{n} = {v} === nothing ? nothing : collect(Csize_t, {v})"
            ));
            (
                vec![t("Ptr{Csize_t}"), t("Csize_t")],
                vec![
                    format!("({n} === nothing ? C_NULL : {n})"),
                    format!("({n} === nothing ? 0 : length({n}))"),
                ],
            )
        }
        "rows2" | "frows2" | "frows6" => {
            let (el, w, splat) = match p.kind.as_str() {
                "rows2" => ("Csize_t", 2, "true"),
                "frows2" => ("Cfloat", 2, "false"),
                _ => ("Cfloat", 6, "false"),
            };
            pre.push(format!("{n} = _rows({el}, {v}, {w}, {splat})"));
            (
                vec![format!("Ptr{{{el}}}"), t("Csize_t")],
                vec![n.clone(), format!("length({v})")],
            )
        }
        "strs" | "strs?" => {
            pre.push(format!("{n}_b, {n}_p = _cstrs({v})"));
            keep.push(format!("{n}_b"));
            (
                vec![t("Ptr{Ptr{UInt8}}"), t("Csize_t")],
                vec![
                    format!("({v} === nothing ? C_NULL : {n}_p)"),
                    format!("({v} === nothing ? 0 : length({n}_p))"),
                ],
            )
        }
        "inputs" => {
            pre.push(format!("{n} = _inputs({v})"));
            pre.push(format!("{n}_k = {n}.keep"));
            keep.push(format!("{n}_k"));
            (
                vec![
                    t("Ptr{Ptr{UInt8}}"),
                    t("Ptr{Ptr{Cfloat}}"),
                    t("Ptr{Csize_t}"),
                    t("Csize_t"),
                ],
                vec![
                    format!("{n}.names"),
                    format!("{n}.data"),
                    format!("{n}.lens"),
                    format!("length({n}.lens)"),
                ],
            )
        }
        "typed_inputs" => {
            pre.push(format!("{n} = _typed_inputs({v})"));
            pre.push(format!("{n}_k = {n}.keep"));
            keep.push(format!("{n}_k"));
            (
                vec![
                    t("Ptr{Ptr{UInt8}}"),
                    t("Ptr{Ptr{UInt8}}"),
                    t("Ptr{Csize_t}"),
                    t("Ptr{Ptr{UInt8}}"),
                    t("Csize_t"),
                ],
                vec![
                    format!("{n}.names"),
                    format!("{n}.data"),
                    format!("{n}.lens"),
                    format!("{n}.dtypes"),
                    format!("length({n}.lens)"),
                ],
            )
        }
        "graph?" => (
            vec![t("Ptr{Cvoid}")],
            vec![format!("({v} === nothing ? C_NULL : {v}.handle)")],
        ),
        // graph, graph&, and every handle.
        _ => (vec![t("Ptr{Cvoid}")], vec![format!("{v}.handle")]),
    }
}

/// `(declarations, ccall types, arguments, result expression)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, Vec<String>, String) {
    let o = format!("_out{i}");
    let r = |ty: &str, init: &str| {
        (
            vec![format!("{o} = Ref{{{ty}}}({init})")],
            vec![format!("Ptr{{{ty}}}")],
            vec![o.clone()],
        )
    };
    let (d, t, a) = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            r("Ptr{UInt8}", "C_NULL")
        }
        "f32s" => r("RlxF32s", "RlxF32s(C_NULL, 0)"),
        "u8s" => r("RlxBytes", "RlxBytes(C_NULL, 0)"),
        "u32s" | "nodes" => r("RlxU32s", "RlxU32s(C_NULL, 0)"),
        "bool" => r("Bool", "false"),
        "usize" => r("Csize_t", "0"),
        "u32" | "node" => r("UInt32", "0"),
        "u64" => r("UInt64", "0"),
        "f32" => r("Cfloat", "0"),
        "f32?" => (
            vec![
                format!("{o} = Ref{{Cfloat}}(0)"),
                format!("{o}k = Ref{{Bool}}(false)"),
            ],
            vec!["Ptr{Cfloat}".into(), "Ptr{Bool}".into()],
            vec![o.clone(), format!("{o}k")],
        ),
        _ => r("Ptr{Cvoid}", "C_NULL"),
    };
    let e = match kind {
        "str" => format!("_take_string({o}[])"),
        "str?" => format!("({o}[] == C_NULL ? nothing : _take_string({o}[]))"),
        "strings" => format!("String.(_parse_json(_take_string({o}[])))"),
        "shapes" => format!("[Int.(s) for s in _parse_json(_take_string({o}[]))]"),
        "json" | "object" | "rows" | "shape" => {
            format!("_parse_json(_take_string({o}[]))")
        }
        "f32s" | "u8s" | "u32s" | "nodes" => format!("_take_array({o}[])"),
        "outputs" => format!("_take_outputs({o}[])"),
        "typed_outputs" => format!("_take_typed_outputs({o}[])"),
        "tensors" => format!("_take_tensors({o}[])"),
        "f32?" => format!("({o}k[] ? {o}[] : nothing)"),
        "bool" | "usize" | "u32" | "node" | "u64" | "f32" => format!("{o}[]"),
        "graph" => format!("Graph({o}[])"),
        class => format!("{class}({o}[])"),
    };
    (d, t, a, e)
}

fn func(s: &mut String, ir: &Ir, f: &Func) {
    let class = ir.class_of(f).map(|c| c.name.as_str());
    let mut sig = Vec::new();
    let name = match f.role {
        Role::Constructor => class.unwrap().to_string(),
        _ => ident(&f.name),
    };
    match f.role {
        Role::Method => sig.push(format!("self::{}", class.unwrap())),
        Role::Static => sig.push(format!("::Type{{{}}}", class.unwrap())),
        _ => {}
    }
    for p in f.params.iter().filter(|p| !p.optional()) {
        sig.push(ident(&p.name));
    }
    let kws: Vec<String> = f
        .params
        .iter()
        .filter(|p| p.optional())
        .map(|p| {
            let d = p
                .default
                .as_ref()
                .map(literal)
                .unwrap_or_else(|| "nothing".into());
            format!("{}={d}", ident(&p.name))
        })
        .collect();
    let head = if kws.is_empty() {
        sig.join(", ")
    } else {
        format!("{}; {}", sig.join(", "), kws.join(", "))
    };
    let doc = doc_lines(&f.doc, "");
    if !doc.trim().is_empty() {
        let _ = writeln!(
            s,
            "\"\"\"\n{}\"\"\"",
            doc.replace('$', "\\$").replace("\"\"\"", "\\\"\\\"\\\"")
        );
    }
    let _ = writeln!(s, "function {name}({head})");
    let mut pre = Vec::new();
    let mut keep = Vec::new();
    let mut types = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        types.push("Ptr{Cvoid}".to_string());
        args.push("self.handle".to_string());
    }
    for p in &f.params {
        let (t, a) = lower(p, &ident(&p.name), &mut pre, &mut keep);
        types.extend(t);
        args.extend(a);
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let preserve = |call: String| {
        if keep.is_empty() {
            call
        } else {
            format!("GC.@preserve {} {call}", keep.join(" "))
        }
    };
    // `gc_safe`: a call may block on a peer (a mesh) or compute for long; the
    // library never calls back into Julia, so other threads may collect.
    let ccall = |types: &[String], args: &[String], ret: &str| {
        let typed: Vec<String> = args
            .iter()
            .zip(types)
            .map(|(a, t)| format!("{a}::{t}"))
            .collect();
        format!(
            "@ccall gc_safe=true librlx.{}({})::{ret}",
            f.c,
            typed.join(", ")
        )
    };
    if f.direct {
        let ret = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "Ptr{UInt8}",
            _ => "Bool",
        };
        let call = preserve(ccall(&types, &args, ret));
        if ret == "Ptr{UInt8}" {
            let _ = writeln!(s, "    unsafe_string({call})");
        } else {
            let _ = writeln!(s, "    {call}");
        }
        let _ = writeln!(s, "end\n");
        return;
    }
    let mut exprs = Vec::new();
    for (i, (_, k)) in f.rets.iter().enumerate() {
        let (d, t, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "    {line}");
        }
        types.extend(t);
        args.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(s, "    _check({})", preserve(ccall(&types, &args, "Int32")));
    match exprs.len() {
        0 => {
            let _ = writeln!(s, "    nothing");
        }
        1 => {
            let _ = writeln!(s, "    {}", exprs[0]);
        }
        _ => {
            let parts: Vec<String> = f
                .rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{n}={e}"))
                .collect();
            let _ = writeln!(s, "    ({},)", parts.join(", "));
        }
    }
    let _ = writeln!(s, "end\n");
}

pub fn generate(ir: &Ir) -> String {
    let mut s = String::from(
        "# RLX — versatile ML compiler + runtime.\n\
         # Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         # SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         # @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         # do not edit; regenerate with `just gen-bindings`. Included by RLX.jl.\n\n",
    );
    s.push_str("# ── handle types ─────────────────────────────────────────\n\n");
    for c in &ir.classes {
        s.push_str(&format!("\"\"\"\n{}\"\"\"\n", doc_lines(&c.doc, "")));
        let _ = writeln!(
            s,
            "mutable struct {n}\n    handle::Ptr{{Cvoid}}\n    function {n}(h::Ptr{{Cvoid}})\n        h == C_NULL && error(\"rlx returned a NULL {n}\")\n        x = new(h)\n        finalizer(x) do y\n            ccall((:{free}, librlx), Cvoid, (Ptr{{Cvoid}},), y.handle)\n        end\n    end\nend\n",
            n = c.name,
            free = c.c_free()
        );
    }
    s.push_str("# ── functions ────────────────────────────────────────────\n\n");
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        func(&mut s, ir, f);
    }
    s
}
