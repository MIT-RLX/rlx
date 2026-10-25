// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-odin/rlx/generated.odin`: the whole C ABI as Odin (package `rlx`),
//! from the lowered [`Ir`].
//!
//! Every C function is a `foreign` declaration (feature-gated ones under
//! `when RLX_<FEATURE>`, set with `-define:RLX_FPGA=true`). Handles are
//! `distinct rawptr`s freed with `<class>_close`. Every procedure returns
//! `(result, Error)`; `rlx.must(…)` unwraps or panics, so calls nest. The
//! graph's ops take the graph first and carry no prefix (`rlx.matmul(g, a,
//! b)`); other classes' members carry the class (`rlx.compiled_run(c, x)`).
//! Optional parameters are Odin defaults (named arguments work:
//! `rlx.softmax(g, x, axis = 1)`). Options go in as JSON text; reports come
//! back as `json.Value`.

use std::collections::HashSet;
use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

const RESERVED: &[&str] = &[
    "asm",
    "auto_cast",
    "bit_set",
    "break",
    "case",
    "cast",
    "context",
    "continue",
    "defer",
    "distinct",
    "do",
    "dynamic",
    "else",
    "enum",
    "fallthrough",
    "for",
    "foreign",
    "if",
    "import",
    "in",
    "map",
    "matrix",
    "not_in",
    "or_else",
    "or_return",
    "or_break",
    "or_continue",
    "package",
    "proc",
    "return",
    "struct",
    "switch",
    "transmute",
    "typeid",
    "union",
    "using",
    "when",
    "where",
    "nil",
    "true",
    "false",
    // builtin procedures
    "len",
    "cap",
    "size_of",
    "align_of",
    "offset_of",
    "type_of",
    "type_info_of",
    "typeid_of",
    "swizzle",
    "complex",
    "quaternion",
    "real",
    "imag",
    "jmag",
    "kmag",
    "conj",
    "expand_values",
    "min",
    "max",
    "abs",
    "clamp",
    "make",
    "new",
    "free",
    "delete",
    "append",
    "copy",
    "clear",
    "resize",
    "reserve",
    "panic",
    "assert",
    "unreachable",
    "raw_data",
    // ours
    "must",
    "close",
];

fn guard(n: String) -> String {
    if RESERVED.contains(&n.as_str()) {
        format!("{n}_")
    } else {
        n
    }
}

fn snake_class(name: &str) -> String {
    let mut s = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            s.push('_');
        }
        s.push(ch.to_ascii_lowercase());
    }
    s
}

/// The Odin name of every function.
fn names(ir: &Ir) -> Vec<String> {
    let free: HashSet<String> = ir.free_funcs().map(|f| f.name.clone()).collect();
    ir.funcs
        .iter()
        .map(|f| {
            let class = ir.class_of(f).map(|c| c.name.clone());
            match (f.role, class.as_deref()) {
                (Role::Constructor, Some(c)) => format!("{}_new", snake_class(c)),
                (_, None) => guard(f.name.clone()),
                (_, Some("Graph")) if !free.contains(&f.name) => guard(f.name.clone()),
                (_, Some(c)) => format!("{}_{}", snake_class(c), f.name),
            }
        })
        .collect()
}

fn arg(name: &str) -> String {
    match name {
        "self" | "rc" | "err" => format!("{name}_"),
        n => guard(n.to_string()),
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
        "size_t" => "uint",
        "const char*" => "cstring",
        _ => "rawptr",
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" | "json" | "value" | "json?" | "value?" => "string",
        "str?" => "Maybe(string)",
        "bool" => "bool",
        "f32" => "f32",
        "f64" => "f64",
        "i32" | "u32" | "i64" | "u64" | "usize" => "int",
        "node" | "node?" => "Node",
        "u64?" | "usize?" => "Maybe(int)",
        "f32?" => "Maybe(f32)",
        "f32s" => "[]f32",
        "u8s" => "[]u8",
        "u32s" | "nodes" => "[]Node",
        "usizes" | "usizes?" | "i64s" => "[]int",
        "rows2" => "[][]int",
        "frows2" | "frows6" => "[][]f32",
        "strs" | "strs?" => "[]string",
        "inputs" => "map[string][]f32",
        "typed_inputs" => "map[string]Typed_Tensor",
        "graph" | "graph&" | "graph?" => "Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "Maybe(string)",
        "json" | "object" | "rows" => "json.Value",
        "shapes" => "[][]int",
        "shape" => "[]int",
        "strings" => "[]string",
        "f32s" => "[]f32",
        "u8s" => "[]u8",
        "u32s" | "nodes" => "[]Node",
        "outputs" => "[][]f32",
        "typed_outputs" => "[]Typed_Tensor",
        "tensors" => "map[string][]f32",
        "bool" => "bool",
        "usize" | "u64" => "int",
        "u32" | "node" => "Node",
        "f32" => "f32",
        "f32?" => "Maybe(f32)",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => match kind {
            "node?" | "node" => "NONE".into(),
            k if ty(k).starts_with("Maybe") => "nil".into(),
            k if ty(k) == "string" => "\"\"".into(),
            _ => "nil".into(),
        },
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) if n.contains("inf") => {
            if n.starts_with('-') {
                "-math.INF_F32".into()
            } else {
                "math.INF_F32".into()
            }
        }
        Lit::Num(n) => n.clone(),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) if items.is_empty() => "nil".into(),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(|i| literal(i, "f32")).collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// `(statements, C arguments)` for one parameter: temporaries are freed with
/// `defer` when the procedure returns.
fn lower(p: &Param, v: &str) -> (Vec<String>, Vec<String>) {
    let t = format!("a_{}", p.name);
    let tmp = |init: String, ptr: String, n: String| {
        (
            vec![format!("{t} := {init}"), format!("defer delete({t})")],
            vec![ptr, n],
        )
    };
    match p.kind.as_str() {
        "str" => (
            vec![
                format!("{t} := strings.clone_to_cstring({v})"),
                format!("defer delete({t})"),
            ],
            vec![t],
        ),
        "json" | "value" | "json?" | "value?" => (
            vec![
                format!("{t} := opt_cstring({v})"),
                format!("defer delete({t})"),
            ],
            vec![t],
        ),
        "str?" => (
            vec![
                format!("{t} := maybe_cstring({v})"),
                format!("defer delete({t})"),
            ],
            vec![t],
        ),
        "node?" | "u32" | "node" => (vec![], vec![format!("u32({v})")]),
        "u64?" | "usize?" => (
            vec![
                format!("{t}, {t}_ok := {v}.?"),
                format!("{t}_box := u64({t})"),
            ],
            vec![format!("{t}_ok ? rawptr(&{t}_box) : nil")],
        ),
        "f32?" => (
            vec![format!("{t}, {t}_ok := {v}.?")],
            vec![format!("{t}_ok ? rawptr(&{t}) : nil")],
        ),
        "f32s" | "u8s" => (
            vec![],
            vec![format!("raw_data({v})"), format!("uint(len({v}))")],
        ),
        "u32s" | "nodes" => (
            vec![],
            vec![format!("raw_data({v})"), format!("uint(len({v}))")],
        ),
        "usizes" | "usizes?" => tmp(
            format!("to_u64s({v})"),
            format!("raw_data({t})"),
            format!("uint(len({v}))"),
        ),
        "i64s" => tmp(
            format!("to_i64s({v})"),
            format!("raw_data({t})"),
            format!("uint(len({v}))"),
        ),
        "rows2" => tmp(
            format!("flat_rows({v}, 2, true)"),
            format!("raw_data({t})"),
            format!("uint(len({v}))"),
        ),
        "frows2" => tmp(
            format!("flat_frows({v}, 2)"),
            format!("raw_data({t})"),
            format!("uint(len({v}))"),
        ),
        "frows6" => tmp(
            format!("flat_frows({v}, 6)"),
            format!("raw_data({t})"),
            format!("uint(len({v}))"),
        ),
        "strs" | "strs?" => (
            vec![
                format!("{t} := cstrings({v})"),
                format!("defer free_cstrings({t})"),
            ],
            vec![format!("raw_data({t})"), format!("uint(len({v}))")],
        ),
        "inputs" | "typed_inputs" => {
            let f = if p.kind == "inputs" {
                "make_inputs"
            } else {
                "make_typed_inputs"
            };
            let mut a = vec![
                format!("raw_data({t}.names)"),
                format!("raw_data({t}.data)"),
                format!("raw_data({t}.lens)"),
            ];
            if p.kind == "typed_inputs" {
                a.push(format!("raw_data({t}.dtypes)"));
            }
            a.push(format!("uint(len({t}.names))"));
            (
                vec![
                    format!("{t} := {f}({v})"),
                    format!("defer free_inputs({t})"),
                ],
                a,
            )
        }
        "bool" | "f32" | "f64" => (vec![], vec![v.to_string()]),
        "i32" => (vec![], vec![format!("i32({v})")]),
        "i64" => (vec![], vec![format!("i64({v})")]),
        "u64" => (vec![], vec![format!("u64({v})")]),
        "usize" => (vec![], vec![format!("uint({v})")]),
        _ => (vec![], vec![format!("rawptr({v})")]),
    }
}

/// `(declarations, C arguments, Odin result)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let decl = |t: &str| vec![format!("{o}: {t}")];
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (decl("Slice"), vec![format!("&{o}")]),
        "bool" => (decl("bool"), vec![format!("&{o}")]),
        "u32" | "node" => (decl("u32"), vec![format!("&{o}")]),
        "f32" => (decl("f32"), vec![format!("&{o}")]),
        "usize" => (decl("uint"), vec![format!("&{o}")]),
        "u64" => (decl("u64"), vec![format!("&{o}")]),
        "f32?" => (
            vec![format!("{o}: f32"), format!("{o}k: bool")],
            vec![format!("&{o}"), format!("&{o}k")],
        ),
        _ => (decl("rawptr"), vec![format!("&{o}")]),
    };
    let e = match kind {
        "str" => format!("take_string({o})"),
        "str?" => format!("take_maybe_string({o})"),
        "json" | "object" | "rows" => format!("take_json({o})"),
        "strings" => format!("take_strings({o})"),
        "shape" => format!("take_ints({o})"),
        "shapes" => format!("take_int_rows({o})"),
        "f32s" => format!("take_f32s(&{o})"),
        "u8s" => format!("take_bytes(&{o})"),
        "u32s" | "nodes" => format!("take_u32s(&{o})"),
        "outputs" => format!("take_outputs({o})"),
        "typed_outputs" => format!("take_typed_outputs({o})"),
        "tensors" => format!("take_tensors({o})"),
        "bool" => o.clone(),
        "u32" | "node" => format!("Node({o})"),
        "usize" | "u64" => format!("int({o})"),
        "f32" => o.clone(),
        "f32?" => format!("{o}k ? Maybe(f32)({o}) : nil"),
        "graph" => format!("Graph({o})"),
        class => format!("{class}({o})"),
    };
    (d, a, e)
}

fn gate(f: &Func, defaults: &[String]) -> Option<String> {
    f.feature
        .as_ref()
        .filter(|x| !defaults.contains(x))
        .map(|x| format!("RLX_{}", x.to_ascii_uppercase().replace('-', "_")))
}

fn func(s: &mut String, ir: &Ir, f: &Func, name: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let mut sig: Vec<String> = Vec::new();
    if f.role == Role::Method {
        sig.push(format!("self: {}", class.clone().unwrap()));
    }
    for p in &f.params {
        let t = ty(&p.kind);
        if p.optional() {
            let d = p
                .default
                .as_ref()
                .map_or_else(|| literal(&Lit::None, &p.kind), |l| literal(l, &p.kind));
            sig.push(format!("{}: {t} = {d}", arg(&p.name)));
        } else {
            sig.push(format!("{}: {t}", arg(&p.name)));
        }
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let result_ty = format!("{}_Result", pascal_snake(name));
    if rets.len() > 1 && !f.direct {
        let _ = writeln!(s, "{result_ty} :: struct {{");
        for (n, k) in &rets {
            let _ = writeln!(s, "\t{}: {},", guard(n.clone()), ret_ty(k));
        }
        s.push_str("}\n\n");
    }
    let ret = if f.direct {
        ret_ty(&f.rets[0].1)
    } else {
        match rets.as_slice() {
            [] => "Error".to_string(),
            [(_, k)] => format!("({}, Error)", ret_ty(k)),
            _ => format!("({result_ty}, Error)"),
        }
    };
    s.push_str(&doc_lines(&f.doc, "// "));
    let _ = writeln!(s, "{name} :: proc({}) -> {ret} {{", sig.join(", "));
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("rawptr(self)".to_string());
    }
    for p in &f.params {
        let (stmts, a) = lower(p, &arg(&p.name));
        for st in stmts {
            let _ = writeln!(s, "\t{st}");
        }
        args.extend(a);
    }
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(
                    s,
                    "\treturn strings.clone_from_cstring({}({}))",
                    f.c,
                    args.join(", ")
                );
            }
            _ => {
                let _ = writeln!(s, "\treturn {}({})", f.c, args.join(", "));
            }
        }
        s.push_str("}\n\n");
        return;
    }
    let mut exprs = Vec::new();
    for (i, (_, k)) in rets.iter().enumerate() {
        let (d, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "\t{line}");
        }
        args.extend(a);
        exprs.push(e);
    }
    let zero = match rets.len() {
        0 => String::new(),
        _ => "{}, ".to_string(),
    };
    let _ = writeln!(
        s,
        "\tif rc := {}({}); rc != 0 {{\n\t\treturn {zero}Error(rc)\n\t}}",
        f.c,
        args.join(", ")
    );
    match exprs.len() {
        0 => s.push_str("\treturn .None\n"),
        1 => {
            let _ = writeln!(s, "\treturn {}, .None", exprs[0]);
        }
        _ => {
            let fields: Vec<String> = rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{} = {e}", guard(n.clone())))
                .collect();
            let _ = writeln!(s, "\treturn {result_ty}{{{}}}, .None", fields.join(", "));
        }
    }
    s.push_str("}\n\n");
}

fn pascal_snake(name: &str) -> String {
    name.trim_end_matches('_')
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or(String::new(), |h| {
                h.to_ascii_uppercase().to_string() + c.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join("_")
}

pub fn generate(ir: &Ir, defaults: &[String]) -> String {
    let names = names(ir);
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         package rlx\n\
         \n\
         import \"core:encoding/json\"\n\
         import \"core:math\"\n\
         import \"core:strings\"\n\
         \n\
         foreign import lib \"system:rlx_capi\"\n\n",
    );
    let mut gates: Vec<String> = ir.funcs.iter().filter_map(|f| gate(f, defaults)).collect();
    gates.sort();
    gates.dedup();
    for g in &gates {
        let _ = writeln!(s, "{g} :: #config({g}, false)");
    }
    s.push('\n');
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "// "));
        let _ = writeln!(s, "{} :: distinct rawptr\n", c.name);
    }
    let mut plain = String::new();
    let mut gated: std::collections::BTreeMap<String, String> = Default::default();
    for c in &ir.classes {
        let _ = writeln!(
            plain,
            "\t{}_free :: proc(h: rawptr) ---",
            format_args!("rlx_{}", c.c)
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let ps: Vec<String> = c_params(ir, f)
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
            "i32"
        };
        let line = format!("\t{} :: proc({}) -> {ret} ---\n", f.c, ps.join(", "));
        match gate(f, defaults) {
            Some(g) => gated.entry(g).or_default().push_str(&line),
            None => plain.push_str(&line),
        }
    }
    let _ = writeln!(
        s,
        "@(default_calling_convention = \"c\")\nforeign lib {{\n{plain}}}\n"
    );
    for (g, decls) in &gated {
        let _ = writeln!(
            s,
            "when {g} {{\n\t@(default_calling_convention = \"c\")\n\tforeign lib {{\n{}\t}}\n}}\n",
            decls
                .lines()
                .map(|l| format!("\t{l}\n"))
                .collect::<String>()
        );
    }
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "// Free the handle now (it is unusable afterwards).\n{n}_close :: proc(h: {c}) {{\n\tif h != nil {{\n\t\trlx_{n}_free(rawptr(h))\n\t}}\n}}\n",
            n = c.c,
            c = c.name
        );
    }
    for (f, name) in ir.funcs.iter().zip(&names) {
        if f.manual {
            continue;
        }
        match gate(f, defaults) {
            Some(g) => {
                let mut body = String::new();
                func(&mut body, ir, f, name);
                let _ = writeln!(
                    s,
                    "when {g} {{\n{}}}\n",
                    body.trim_end()
                        .lines()
                        .map(|l| format!("{l}\n"))
                        .collect::<String>()
                );
            }
            None => func(&mut s, ir, f, name),
        }
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    if !s.contains("math.") {
        s = s.replacen("import \"core:math\"\n", "", 1);
    }
    s
}
