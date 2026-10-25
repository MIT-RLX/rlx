// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-dart/lib/src/generated.dart`: the whole C ABI as Dart (`dart:ffi`),
//! from the lowered [`Ir`].
//!
//! A class per handle type (freed by a `NativeFinalizer` on the C free
//! function, or early with `close()`); free functions are top-level. Optional
//! parameters are named parameters with the tables' defaults; several results
//! come back as a record. Every C function is a lazily looked-up top-level
//! `final`, so a function a build lacks fails only when called.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines};
use crate::jni::c_params;

const KEYWORDS: &[&str] = &[
    "assert", "break", "case", "catch", "class", "const", "continue", "default", "do", "else",
    "enum", "extends", "false", "final", "finally", "for", "if", "in", "is", "new", "null",
    "rethrow", "return", "super", "switch", "this", "throw", "true", "try", "var", "void", "while",
    "with", "await", "yield",
];

/// `Object`'s members and our own: a method by one of these names is suffixed.
const MEMBERS: &[&str] = &[
    "toString",
    "hashCode",
    "runtimeType",
    "noSuchMethod",
    "close",
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

/// `(native type, Dart type)` of a C parameter type.
fn ffi(c: &str) -> (&'static str, &'static str) {
    match c {
        "bool" => ("Bool", "bool"),
        "float" => ("Float", "double"),
        "double" => ("Double", "double"),
        "int32_t" => ("Int32", "int"),
        "uint32_t" => ("Uint32", "int"),
        "int64_t" => ("Int64", "int"),
        "uint64_t" => ("Uint64", "int"),
        "size_t" => ("Size", "int"),
        _ => ("Pointer<Void>", "Pointer<Void>"),
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "String?",
        "json" | "value" => "Object",
        "json?" | "value?" => "Object?",
        "bool" => "bool",
        "f32" | "f64" => "double",
        "i32" | "u32" | "node" | "i64" | "u64" | "usize" => "int",
        "node?" | "u64?" | "usize?" => "int?",
        "f32?" => "double?",
        "f32s" => "List<double>",
        "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => "List<int>",
        "usizes?" => "List<int>?",
        "rows2" => "List<List<int>>",
        "frows2" | "frows6" => "List<List<double>>",
        "strs" => "List<String>",
        "strs?" => "List<String>?",
        "inputs" => "Map<String, List<double>>",
        "typed_inputs" => "Map<String, TypedTensor>",
        "graph" | "graph&" => "Graph",
        "graph?" => "Graph?",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "String?",
        "json" => "dynamic",
        "object" => "Map<String, dynamic>",
        "rows" => "List<Map<String, dynamic>>",
        "shapes" => "List<List<int>>",
        "shape" => "List<int>",
        "strings" => "List<String>",
        "f32s" => "Float32List",
        "u8s" => "Uint8List",
        "u32s" | "nodes" => "List<int>",
        "outputs" => "List<Float32List>",
        "typed_outputs" => "List<TypedTensor>",
        "tensors" => "Map<String, Float32List>",
        "bool" => "bool",
        "usize" | "u64" | "u32" | "node" => "int",
        "f32" => "double",
        "f32?" => "double?",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "double.infinity"),
        Lit::Str(s) => format!("'{s}'"),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("const [{}]", inner.join(", "))
        }
    }
}

/// Dart expressions for one parameter's C arguments, inside `using((a) { … })`.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "str" => vec![format!("_str($a, {v})")],
        "str?" => vec![format!("_strOrNull($a, {v})")],
        "json" | "value" | "json?" | "value?" => vec![format!("_json($a, {v})")],
        "node?" => vec![format!("{v} ?? none")],
        "u64?" | "usize?" => vec![format!("_boxU64($a, {v})")],
        "f32?" => vec![format!("_boxF32($a, {v})")],
        "f32s" => vec![format!("_floats($a, {v})"), format!("{v}.length")],
        "u8s" => vec![format!("_bytes($a, {v})"), format!("{v}.length")],
        "u32s" | "nodes" => vec![format!("_u32s($a, {v})"), format!("{v}.length")],
        "usizes" => vec![format!("_u64s($a, {v})"), format!("{v}.length")],
        "i64s" => vec![format!("_i64s($a, {v})"), format!("{v}.length")],
        "usizes?" => vec![
            format!("{v} == null ? nullptr : _u64s($a, {v})"),
            format!("{v}?.length ?? 0"),
        ],
        "rows2" => vec![
            format!("_u64s($a, _rows({v}, 2, true))"),
            format!("{v}.length"),
        ],
        "frows2" => vec![
            format!("_floats($a, _frows({v}, 2))"),
            format!("{v}.length"),
        ],
        "frows6" => vec![
            format!("_floats($a, _frows({v}, 6))"),
            format!("{v}.length"),
        ],
        "strs" => vec![format!("_strs($a, {v})"), format!("{v}.length")],
        "strs?" => vec![format!("_strs($a, {v})"), format!("{v}?.length ?? 0")],
        "inputs" | "typed_inputs" => {
            let n = format!("in{}", pascal_case(&p.name));
            let mut v = vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
            ];
            if p.kind == "typed_inputs" {
                v.push(format!("{n}.dtypes"));
            }
            v.push(format!("{n}.n"));
            v
        }
        "graph?" => vec![format!("{v}?._h ?? nullptr")],
        "bool" | "f32" | "f64" | "i32" | "u32" | "node" | "i64" | "u64" | "usize" => {
            vec![v.to_string()]
        }
        _ => vec![format!("{v}._h")],
    }
}

fn pascal_case(name: &str) -> String {
    let c = camel(name);
    let mut it = c.chars();
    match it.next() {
        Some(h) => h.to_ascii_uppercase().to_string() + it.as_str(),
        None => c,
    }
}

/// `(declarations, C arguments, Dart result)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let alloc = |t: &str| vec![format!("final {o} = $a<{t}>();")];
    let (d, args) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (alloc("_Slice"), vec![format!("{o}.cast()")]),
        "bool" => (alloc("Bool"), vec![format!("{o}.cast()")]),
        "u32" | "node" => (alloc("Uint32"), vec![format!("{o}.cast()")]),
        "f32" => (alloc("Float"), vec![format!("{o}.cast()")]),
        "usize" => (alloc("Size"), vec![format!("{o}.cast()")]),
        "u64" => (alloc("Uint64"), vec![format!("{o}.cast()")]),
        "f32?" => (
            vec![
                format!("final {o} = $a<Float>();"),
                format!("final {o}k = $a<Bool>();"),
            ],
            vec![format!("{o}.cast()"), format!("{o}k.cast()")],
        ),
        _ => (alloc("Pointer<Void>"), vec![format!("{o}.cast()")]),
    };
    let e = match kind {
        "str" => format!("_takeString({o})"),
        "str?" => format!("_takeStringOrNull({o})"),
        "json" => format!("_takeJson({o})"),
        "object" => format!("_takeJson({o}) as Map<String, dynamic>"),
        "rows" => format!("(_takeJson({o}) as List).cast<Map<String, dynamic>>()"),
        "shapes" => {
            format!("(_takeJson({o}) as List).map((s) => (s as List).cast<int>()).toList()")
        }
        "shape" => format!("(_takeJson({o}) as List).cast<int>()"),
        "strings" => format!("(_takeJson({o}) as List).cast<String>()"),
        "f32s" => format!("_takeF32s({o})"),
        "u8s" => format!("_takeBytes({o})"),
        "u32s" | "nodes" => format!("_takeU32s({o})"),
        "outputs" => format!("_takeOutputs({o}.value)"),
        "typed_outputs" => format!("_takeTypedOutputs({o}.value)"),
        "tensors" => format!("_takeTensors({o}.value)"),
        "bool" | "u32" | "node" | "f32" | "usize" | "u64" => format!("{o}.value"),
        "f32?" => format!("{o}k.value ? {o}.value : null"),
        "graph" => format!("Graph._({o}.value)"),
        class => format!("{class}._({o}.value)"),
    };
    (d, args, e)
}

/// The lookup of one C function.
fn binding(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let natives: Vec<&str> = params.iter().map(|(t, _)| ffi(t).0).collect();
    let darts: Vec<&str> = params.iter().map(|(t, _)| ffi(t).1).collect();
    let (rn, rd) = if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => ("Pointer<Utf8>", "Pointer<Utf8>"),
            _ => ("Bool", "bool"),
        }
    } else {
        ("Int32", "int")
    };
    let _ = writeln!(
        s,
        "final _{c} = _lib.lookupFunction<{rn} Function({}), {rd} Function({})>('{c}');",
        natives.join(", "),
        darts.join(", "),
        c = f.c
    );
}

fn func(s: &mut String, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let mut sig: Vec<String> = required
        .iter()
        .map(|p| format!("{} {}", ty(&p.kind), ident(&p.name)))
        .collect();
    if !optional.is_empty() {
        let named: Vec<String> = optional
            .iter()
            .map(|p| {
                let t = ty(&p.kind);
                match &p.default {
                    Some(Lit::None) | None => {
                        let t = if t.ends_with('?') { t } else { format!("{t}?") };
                        format!("{t} {}", ident(&p.name))
                    }
                    Some(d) => format!("{t} {} = {}", ident(&p.name), literal(d)),
                }
            })
            .collect();
        sig.push(format!("{{{}}}", named.join(", ")));
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
                .map(|(n, k)| format!("{} {}", ret_ty(k), ident(n)))
                .collect();
            format!("({{{}}})", fields.join(", "))
        }
    };
    s.push_str(&doc_lines(&f.doc, &format!("{indent}/// ")));
    let head = match f.role {
        Role::Constructor => format!("factory {}({})", class.clone().unwrap(), sig.join(", ")),
        Role::Static => format!("static {ret} {}({})", method_name(f), sig.join(", ")),
        _ => format!("{ret} {}({})", method_name(f), sig.join(", ")),
    };
    let _ = writeln!(s, "{indent}{head} {{");
    let b = format!("{indent}  ");
    let _ = writeln!(s, "{b}return using(($a) {{");
    let bb = format!("{b}  ");
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("_h".to_string());
    }
    for p in &f.params {
        let v = ident(&p.name);
        if p.kind == "inputs" || p.kind == "typed_inputs" {
            let helper = if p.kind == "inputs" {
                "_inputs"
            } else {
                "_typedInputs"
            };
            let _ = writeln!(
                s,
                "{bb}final in{} = {helper}($a, {v});",
                pascal_case(&p.name)
            );
        }
        if p.kind == "str" && matches!(p.default, Some(Lit::None)) {
            args.push(format!("_strOrNull($a, {})", ident(&p.name)));
        } else {
            args.extend(lower(p, &v));
        }
    }
    let call_args = |extra: &[String]| {
        let mut all = args.clone();
        all.extend(extra.iter().cloned());
        all.join(", ")
    };
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "{bb}return _{}({}).toDartString();", f.c, call_args(&[]));
            }
            _ => {
                let _ = writeln!(s, "{bb}return _{}({});", f.c, call_args(&[]));
            }
        }
        let _ = writeln!(s, "{b}}});\n{indent}}}\n");
        return;
    }
    let all_rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), "handle".into())],
        _ => f.rets.clone(),
    };
    let mut extra = Vec::new();
    let mut exprs = Vec::new();
    for (i, (_, k)) in all_rets.iter().enumerate() {
        let (d, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "{bb}{line}");
        }
        extra.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(s, "{bb}_check(_{}({}));", f.c, call_args(&extra));
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{bb}return {}._(o0.value);", class.unwrap());
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{bb}return {};", exprs[0]);
        }
        _ => {
            let fields: Vec<String> = rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{}: {e}", ident(n)))
                .collect();
            let _ = writeln!(s, "{bb}return ({});", fields.join(", "));
        }
    }
    let _ = writeln!(s, "{b}}});\n{indent}}}\n");
}

pub fn generate(ir: &Ir) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         \n\
         // ignore_for_file: non_constant_identifier_names, unused_element\n\
         \n\
         part of '../rlx.dart';\n\n",
    );
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        binding(&mut s, ir, f);
    }
    s.push('\n');
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, ir, f, "");
    }
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "/// "));
        let _ = writeln!(
            s,
            "final class {n} extends _Handle {{\n  static final _free = _lib.lookup<NativeFunction<Void Function(Pointer<Void>)>>('{free}');\n  static final _fin = NativeFinalizer(_free.cast());\n  {n}._(Pointer<Void> h) : super(h, _free, _fin);\n",
            n = c.name,
            free = c.c_free()
        );
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut s, ir, f, "  ");
        }
        s.push_str("}\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
