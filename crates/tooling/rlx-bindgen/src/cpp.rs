// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-cpp/include/rlx/rlx.hpp`: the whole C ABI as header-only C++20, from
//! the lowered [`Ir`].
//!
//! One move-only RAII class per handle type; failures throw `rlx::Error`. A
//! function's optional parameters are the fields of its `…Opts` struct, at the
//! tables' defaults (`g.softmax(x, {.axis = 1})`). Options and reports are
//! JSON text; name lists and shapes come back typed.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines, pascal};

const KEYWORDS: &[&str] = &[
    "alignas",
    "alignof",
    "and",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "not",
    "operator",
    "or",
    "private",
    "protected",
    "public",
    "register",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
    "xor",
    "requires",
    "concept",
    "h",
];

fn ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" | "json" | "value" => "std::string",
        "str?" | "json?" | "value?" => "std::optional<std::string>",
        "bool" => "bool",
        "f32" => "float",
        "f64" => "double",
        "i32" => "int32_t",
        "i64" => "int64_t",
        "u32" => "uint32_t",
        "u64" => "uint64_t",
        "usize" => "size_t",
        "node" => "Node",
        "node?" => "std::optional<Node>",
        "u64?" => "std::optional<uint64_t>",
        "usize?" => "std::optional<size_t>",
        "f32?" => "std::optional<float>",
        "f32s" => "std::vector<float>",
        "u8s" => "std::vector<uint8_t>",
        "u32s" => "std::vector<uint32_t>",
        "nodes" => "std::vector<Node>",
        "usizes" => "std::vector<size_t>",
        "usizes?" => "std::optional<std::vector<size_t>>",
        "i64s" => "std::vector<int64_t>",
        "rows2" => "std::vector<std::vector<size_t>>",
        "frows2" | "frows6" => "std::vector<std::vector<float>>",
        "strs" => "std::vector<std::string>",
        "strs?" => "std::optional<std::vector<std::string>>",
        "inputs" => "Inputs",
        "typed_inputs" => "TypedInputs",
        "graph" | "graph&" => "Graph&",
        "graph?" => "Graph*",
        other => return format!("{}&", other.trim_end_matches('&')),
    }
    .to_string()
}

/// A parameter type as a by-value field (options) or a const-ref argument.
fn arg_decl(kind: &str) -> String {
    let t = ty(kind);
    match kind {
        "bool" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "node" => t,
        "graph" | "graph&" | "graph?" => t,
        k if k.ends_with('&') => t,
        _ => format!("const {t}&"),
    }
}

fn field_ty(kind: &str) -> String {
    match kind {
        "graph" | "graph&" | "graph?" => "Graph*".into(),
        k if k.ends_with('&') => format!("{}*", k.trim_end_matches('&')),
        _ => ty(kind),
    }
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" | "json" | "object" | "rows" | "shape" => "std::string",
        "str?" => "std::optional<std::string>",
        "shapes" => "std::vector<std::vector<int64_t>>",
        "strings" => "std::vector<std::string>",
        "f32s" => "std::vector<float>",
        "u8s" => "std::vector<uint8_t>",
        "u32s" | "nodes" => "std::vector<uint32_t>",
        "outputs" => "std::vector<std::vector<float>>",
        "typed_outputs" => "std::vector<TypedTensor>",
        "tensors" => "Tensors",
        "bool" => "bool",
        "usize" => "size_t",
        "u32" | "node" => "uint32_t",
        "u64" => "uint64_t",
        "f32" => "float",
        "f32?" => "std::optional<float>",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "std::nullopt".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "INFINITY"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// C++ expressions for one parameter's C arguments, with statements first.
fn lower(p: &Param, v: &str, pre: &mut Vec<String>) -> Vec<String> {
    let n = format!("c_{}", p.name);
    match p.kind.as_str() {
        "str" | "json" | "value" => vec![format!("{v}.c_str()")],
        "str?" | "json?" | "value?" => vec![format!("{v} ? {v}->c_str() : nullptr")],
        "node?" => vec![format!("{v}.value_or(NONE)")],
        "u64?" | "usize?" | "f32?" => vec![format!("{v} ? &*{v} : nullptr")],
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            vec![format!("{v}.data()"), format!("{v}.size()")]
        }
        "usizes?" => vec![
            format!("{v} ? {v}->data() : nullptr"),
            format!("{v} ? {v}->size() : 0"),
        ],
        "rows2" => {
            pre.push(format!("auto {n} = detail::rows({v}, 2, true);"));
            vec![format!("{n}.data()"), format!("{v}.size()")]
        }
        "frows2" | "frows6" => {
            let w = if p.kind == "frows2" { 2 } else { 6 };
            pre.push(format!("auto {n} = detail::frows({v}, {w});"));
            vec![format!("{n}.data()"), format!("{v}.size()")]
        }
        "strs" => {
            pre.push(format!("detail::CStrings {n}({v});"));
            vec![format!("{n}.data()"), format!("{n}.size()")]
        }
        "strs?" => {
            pre.push(format!("detail::CStrings {n}({v});"));
            vec![format!("{n}.data()"), format!("{n}.size()")]
        }
        "inputs" => {
            pre.push(format!("detail::CInputs {n}({v});"));
            vec![
                format!("{n}.names()"),
                format!("{n}.data()"),
                format!("{n}.lens()"),
                format!("{n}.size()"),
            ]
        }
        "typed_inputs" => {
            pre.push(format!("detail::CTypedInputs {n}({v});"));
            vec![
                format!("{n}.names()"),
                format!("{n}.data()"),
                format!("{n}.lens()"),
                format!("{n}.dtypes()"),
                format!("{n}.size()"),
            ]
        }
        "graph?" => vec![format!("{v} ? {v}->raw() : nullptr")],
        "graph" | "graph&" => vec![format!("{v}.raw()")],
        k if k.ends_with('&') => vec![format!("{v}.raw()")],
        _ => vec![v.to_string()],
    }
}

/// `(declarations, C arguments, result expression)`.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |t: &str| (vec![format!("{t} {o}{{}};")], vec![format!("&{o}")]);
    let (d, a) = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            one("char*")
        }
        "f32s" => one("RlxF32s"),
        "u8s" => one("RlxBytes"),
        "u32s" | "nodes" => one("RlxU32s"),
        "outputs" => one("RlxOutputs*"),
        "typed_outputs" => one("RlxTypedOutputs*"),
        "tensors" => one("RlxTensorMap*"),
        "bool" => one("bool"),
        "usize" => one("size_t"),
        "u32" | "node" => one("uint32_t"),
        "u64" => one("uint64_t"),
        "f32" => one("float"),
        "f32?" => (
            vec![format!("float {o}{{}};"), format!("bool {o}k{{}};")],
            vec![format!("&{o}"), format!("&{o}k")],
        ),
        "graph" => one("RlxGraph*"),
        class => one(&format!("Rlx{class}*")),
    };
    let e = match kind {
        "str" | "json" | "object" | "rows" | "shape" => format!("detail::take_string({o})"),
        "str?" => format!("detail::take_optional_string({o})"),
        "shapes" => format!("detail::parse_int_matrix(detail::take_string({o}))"),
        "strings" => format!("detail::parse_strings(detail::take_string({o}))"),
        "f32s" => format!("detail::take_f32s({o})"),
        "u8s" => format!("detail::take_bytes({o})"),
        "u32s" | "nodes" => format!("detail::take_u32s({o})"),
        "outputs" => format!("detail::take_outputs({o})"),
        "typed_outputs" => format!("detail::take_typed_outputs({o})"),
        "tensors" => format!("detail::take_tensors({o})"),
        "f32?" => format!("{o}k ? std::optional<float>({o}) : std::nullopt"),
        "bool" | "usize" | "u32" | "node" | "u64" | "f32" => o.clone(),
        "graph" => format!("Graph({o})"),
        class => format!("{class}({o})"),
    };
    (d, a, e)
}

fn opts_name(f: &Func, ir: &Ir) -> String {
    let base = match f.role {
        Role::Constructor => "Init".to_string(),
        _ => pascal(&f.name),
    };
    match ir.class_of(f) {
        Some(c)
            if c.name != "Graph"
                || f.role == Role::Constructor
                || ir.free_funcs().any(|g| g.name == f.name) =>
        {
            format!("{}{base}Opts", c.name)
        }
        _ => format!("{base}Opts"),
    }
}

/// Option structs and result structs live at namespace scope, ahead of the classes.
fn decls(s: &mut String, ir: &Ir, f: &Func) {
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    if !optional.is_empty() {
        let _ = writeln!(
            s,
            "/// The optional parameters of `{}`, at their defaults.",
            f.name
        );
        let _ = writeln!(s, "struct {} {{", opts_name(f, ir));
        for p in optional {
            let t = field_ty(&p.kind);
            let t = if t.starts_with("std::optional") || t.ends_with('*') || p.default.is_some() {
                t
            } else {
                format!("std::optional<{t}>")
            };
            let d = match &p.default {
                Some(l) => format!(" = {}", literal(l)),
                None if t.ends_with('*') => " = nullptr".into(),
                None => String::new(),
            };
            let _ = writeln!(s, "    {t} {}{d};", ident(&p.name));
        }
        s.push_str("};\n\n");
    }
    if f.rets.len() > 1 && f.role != Role::Constructor {
        let name = opts_name(f, ir).replace("Opts", "Result");
        let _ = writeln!(s, "/// What `{}` returns.", f.name);
        let _ = writeln!(s, "struct {name} {{");
        for (n, k) in &f.rets {
            let _ = writeln!(s, "    {} {};", ret_ty(k), ident(n));
        }
        s.push_str("};\n\n");
    }
}

fn func(s: &mut String, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let optional = f.params.iter().any(|p| p.optional());
    let opts = opts_name(f, ir);
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![],
        _ => f.rets.clone(),
    };
    let ret = match rets.as_slice() {
        [] => "void".to_string(),
        [(_, k)] => ret_ty(k),
        _ => opts.replace("Opts", "Result"),
    };
    let mut params: Vec<String> = f
        .params
        .iter()
        .filter(|p| !p.optional())
        .map(|p| format!("{} {}", arg_decl(&p.kind), ident(&p.name)))
        .collect();
    if optional {
        params.push(format!("const {opts}& opts = {{}}"));
    }
    s.push_str(&doc_lines(&f.doc, &format!("{indent}/// ")));
    let head = match f.role {
        Role::Constructor => format!("explicit {}({})", class.clone().unwrap(), params.join(", ")),
        Role::Static | Role::Function => {
            let st = if f.role == Role::Static {
                "static "
            } else {
                "inline "
            };
            format!("{st}{ret} {}({})", ident(&f.name), params.join(", "))
        }
        Role::Method => format!("{ret} {}({})", ident(&f.name), params.join(", ")),
    };
    let _ = writeln!(s, "{indent}{head} {{");
    let b = format!("{indent}    ");
    let mut pre = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("h_".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            let field = format!("opts.{}", ident(&p.name));
            match p.kind.as_str() {
                "graph" | "graph&" => format!("(*{field})"),
                k if k.ends_with('&') => format!("(*{field})"),
                _ => field,
            }
        } else {
            ident(&p.name)
        };
        args.extend(lower(p, &v, &mut pre));
    }
    for line in &pre {
        let _ = writeln!(s, "{b}{line}");
    }
    if f.direct {
        let call = format!("::{}({})", f.c, args.join(", "));
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "{b}return std::string({call});");
            }
            _ => {
                let _ = writeln!(s, "{b}return {call};");
            }
        }
        let _ = writeln!(s, "{indent}}}\n");
        return;
    }
    let all: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
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
    let _ = writeln!(s, "{b}detail::check(::{}({}));", f.c, args.join(", "));
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{b}h_ = o0;");
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{b}return {};", exprs[0]);
        }
        _ => {
            let _ = writeln!(s, "{b}return {{{}}};", exprs.join(", "));
        }
    }
    let _ = writeln!(s, "{indent}}}\n");
}

pub fn generate(ir: &Ir) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         //\n\
         // rlx for C++20, header-only over rlx.h: one RAII class per handle type,\n\
         // `rlx::Error` on failure, optional parameters in each function's `…Opts`\n\
         // struct (`g.softmax(x, {.axis = 1})`). Options and reports are JSON text.\n\
         \n\
         #pragma once\n\
         \n\
         #include \"rlx/support.hpp\"\n\
         \n\
         namespace rlx {\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(s, "class {};", c.name);
    }
    s.push('\n');
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        decls(&mut s, ir, f);
    }
    // Classes in dependency-free order: methods are defined inline after every
    // class is declared, so a method may return any handle type.
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "/// "));
        let mixin = match c.name.as_str() {
            "Optimizer" | "Trainer" => format!(", public detail::{}Manual<{}>", c.name, c.name),
            _ => String::new(),
        };
        let _ = writeln!(
            s,
            "class {n} : public detail::Handle<Rlx{n}, ::{free}>{mixin} {{\npublic:\n    explicit {n}(Rlx{n}* h) : Handle(h) {{}}\n",
            n = c.name,
            free = c.c_free()
        );
        for f in ir.members(c).filter(|f| !f.manual) {
            // Declarations only; definitions follow all classes.
            let mut decl = String::new();
            func(&mut decl, ir, f, "    ");
            // Turn the inline definition into a declaration + out-of-class definition.
            s.push_str(&declaration_of(&decl));
        }
        s.push_str("};\n\n");
    }
    for c in &ir.classes {
        for f in ir.members(c).filter(|f| !f.manual) {
            let mut def = String::new();
            func(&mut def, ir, f, "");
            s.push_str(&out_of_class(&def, &c.name, f));
        }
    }
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, ir, f, "");
    }
    s.push_str("}  // namespace rlx\n");
    s
}

/// The doc lines and signature of an inline method, as an in-class declaration.
fn declaration_of(def: &str) -> String {
    let mut out = String::new();
    for line in def.lines() {
        if line.trim_start().starts_with("///") {
            out.push_str(line);
            out.push('\n');
        } else {
            let sig = line.trim_end().trim_end_matches('{').trim_end();
            // The constructor keeps its body-less form; defaults stay on the declaration.
            let _ = writeln!(out, "{sig};\n");
            break;
        }
    }
    out
}

/// An inline method definition rewritten out of class: `Class::name(...)`,
/// without default arguments, `static` or `explicit`.
fn out_of_class(def: &str, class: &str, f: &Func) -> String {
    let mut lines = def.lines().filter(|l| !l.trim_start().starts_with("///"));
    let head = lines.next().unwrap_or_default();
    let head = head.replace(" = {}", "");
    let name = match f.role {
        Role::Constructor => class.to_string(),
        _ => ident(&f.name),
    };
    let head = match f.role {
        Role::Constructor => head.replacen(
            &format!("explicit {class}("),
            &format!("inline {class}::{class}("),
            1,
        ),
        Role::Static => head.replacen("static ", "inline ", 1).replacen(
            &format!(" {name}("),
            &format!(" {class}::{name}("),
            1,
        ),
        _ => format!(
            "inline {}",
            head.replacen(&format!(" {name}("), &format!(" {class}::{name}("), 1)
        ),
    };
    let head = match f.role {
        Role::Constructor => head.replacen(") {", ") : Handle(nullptr) {", 1),
        _ => head,
    };
    let mut out = String::new();
    let _ = writeln!(out, "{head}");
    for l in lines {
        out.push_str(l);
        out.push('\n');
    }
    out
}
