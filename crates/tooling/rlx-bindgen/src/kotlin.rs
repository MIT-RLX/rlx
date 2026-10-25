// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-kotlin/src/main/kotlin/org/rlx/kt/RLX.kt`: the whole C ABI as Kotlin
//! (Android and the JVM, over the generated JNI glue in `jni.rs`), from the
//! lowered [`Ir`].
//!
//! One class per handle type (`AutoCloseable`); constructors are Kotlin
//! constructors, statics live in each class's companion, free functions in
//! `object RLX`. Optional parameters are Kotlin default arguments (named:
//! `g.softmax(x, axis = 1)`); options go in as Kotlin values (JSON), reports
//! come back parsed. Errors are `RlxException`.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines};
use crate::jni::kotlin_natives;

const KEYWORDS: &[&str] = &[
    "as",
    "break",
    "class",
    "continue",
    "do",
    "else",
    "false",
    "for",
    "fun",
    "if",
    "in",
    "interface",
    "is",
    "null",
    "object",
    "package",
    "return",
    "super",
    "this",
    "throw",
    "true",
    "try",
    "typealias",
    "typeof",
    "val",
    "var",
    "when",
    "while",
    "h",
];

fn ident(name: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("`{c}`")
    } else {
        c
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "String?",
        "json" | "value" => "Any",
        "json?" | "value?" => "Any?",
        "bool" => "Boolean",
        "f32" => "Float",
        "f64" => "Double",
        "i32" | "u32" | "node" => "Int",
        "i64" | "u64" | "usize" => "Long",
        "node?" => "Int?",
        "u64?" | "usize?" => "Long?",
        "f32?" => "Float?",
        "f32s" => "FloatArray",
        "u8s" => "ByteArray",
        "u32s" | "nodes" => "IntArray",
        "usizes" | "i64s" => "LongArray",
        "usizes?" => "LongArray?",
        "rows2" => "List<LongArray>",
        "frows2" | "frows6" => "List<FloatArray>",
        "strs" => "List<String>",
        "strs?" => "List<String>?",
        "inputs" => "Map<String, FloatArray>",
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
        "json" => "Any?",
        "object" => "Map<String, Any?>",
        "rows" => "List<Map<String, Any?>>",
        "shapes" => "List<LongArray>",
        "shape" => "Map<String, Any?>",
        "strings" => "List<String>",
        "f32s" => "FloatArray",
        "u8s" => "ByteArray",
        "u32s" | "nodes" => "IntArray",
        "outputs" => "List<FloatArray>",
        "typed_outputs" => "List<TypedTensor>",
        "tensors" => "Map<String, FloatArray>",
        "bool" => "Boolean",
        "usize" | "u64" => "Long",
        "u32" | "node" => "Int",
        "f32" => "Float",
        "f32?" => "Float?",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => {
            let n = n.replace("inf", "Float.POSITIVE_INFINITY");
            match ty(kind).trim_end_matches('?').trim_end_matches("Array") {
                "Float" if !n.contains("INFINITY") => {
                    if n.contains('.') || n.contains('e') {
                        format!("{n}f")
                    } else {
                        format!("{n}.0f")
                    }
                }
                "Double" if !n.contains('.') && !n.contains('e') => format!("{n}.0"),
                "Long" => format!("{n}L"),
                _ => n,
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let t = ty(kind);
            let el = t.trim_end_matches('?');
            let maker = match el {
                "LongArray" => "longArrayOf",
                "FloatArray" => "floatArrayOf",
                "IntArray" => "intArrayOf",
                "ByteArray" => "byteArrayOf",
                _ => "listOf",
            };
            let inner_kind = match el {
                "LongArray" => "i64",
                "FloatArray" => "f32",
                _ => "i32",
            };
            let inner: Vec<String> = items.iter().map(|i| literal(i, inner_kind)).collect();
            format!("{maker}({})", inner.join(", "))
        }
    }
}

/// Kotlin expressions for one parameter's JNI arguments.
fn lower(p: &Param, v: &str, pre: &mut Vec<String>) -> Vec<String> {
    let n = format!("in_{}", p.name);
    match p.kind.as_str() {
        "str" | "str?" => vec![v.into()],
        "json" | "value" => vec![format!("Json.encode({v})")],
        "json?" | "value?" => vec![format!("{v}?.let {{ Json.encode(it) }}")],
        "node?" => vec![format!("({v} ?: NONE)")],
        "u64?" | "usize?" => vec![format!("{v}?.let {{ longArrayOf(it) }}")],
        "f32?" => vec![format!("{v}?.let {{ floatArrayOf(it) }}")],
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            vec![v.into(), format!("{v}.size.toLong()")]
        }
        "usizes?" => vec![v.into(), format!("({v}?.size ?: 0).toLong()")],
        "rows2" => vec![
            format!("S.rows({v}, 2, true)"),
            format!("{v}.size.toLong()"),
        ],
        "frows2" => vec![format!("S.frows({v}, 2)"), format!("{v}.size.toLong()")],
        "frows6" => vec![format!("S.frows({v}, 6)"), format!("{v}.size.toLong()")],
        "strs" => vec![
            format!("{v}.toTypedArray<Any?>()"),
            format!("{v}.size.toLong()"),
        ],
        "strs?" => vec![
            format!("{v}?.toTypedArray<Any?>()"),
            format!("({v}?.size ?: 0).toLong()"),
        ],
        "inputs" => {
            pre.push(format!("val {n} = S.inputs({v})"));
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.n"),
            ]
        }
        "typed_inputs" => {
            pre.push(format!("val {n} = S.typedInputs({v})"));
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.dtypes"),
                format!("{n}.n"),
            ]
        }
        "graph?" => vec![format!("({v}?.h ?: 0L)")],
        "bool" | "f32" | "f64" | "i32" | "u32" | "node" | "i64" | "u64" | "usize" => vec![v.into()],
        _ => vec![format!("{v}.h")],
    }
}

/// `(declarations, JNI arguments, Kotlin result)`.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |t: &str| (vec![format!("val {o} = {t}")], vec![o.clone()]);
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => one("LongArray(2)"),
        "bool" => one("BooleanArray(1)"),
        "u32" | "node" => one("IntArray(1)"),
        "f32" => one("FloatArray(1)"),
        "f32?" => (
            vec![
                format!("val {o} = FloatArray(1)"),
                format!("val {o}k = BooleanArray(1)"),
            ],
            vec![o.clone(), format!("{o}k")],
        ),
        _ => one("LongArray(1)"),
    };
    let e = match kind {
        "str" => format!("S.takeString({o}[0])"),
        "str?" => format!("if ({o}[0] == 0L) null else S.takeString({o}[0])"),
        "json" => format!("Json.parse(S.takeString({o}[0]))"),
        "object" | "shape" => format!("S.obj(Json.parse(S.takeString({o}[0])))"),
        "rows" => format!("S.rowsOf(Json.parse(S.takeString({o}[0])))"),
        "shapes" => format!("S.shapes(Json.parse(S.takeString({o}[0])))"),
        "strings" => format!("S.strings(Json.parse(S.takeString({o}[0])))"),
        "f32s" => format!("NativeSupport.takeFloats({o}[0], {o}[1])"),
        "u8s" => format!("NativeSupport.takeBytes({o}[0], {o}[1])"),
        "u32s" | "nodes" => format!("NativeSupport.takeInts({o}[0], {o}[1])"),
        "outputs" => format!("NativeSupport.takeOutputs({o}[0]).toList()"),
        "typed_outputs" => format!("S.typedOutputs(NativeSupport.takeTypedOutputs({o}[0]))"),
        "tensors" => format!("S.tensors(NativeSupport.takeTensors({o}[0]))"),
        "f32?" => format!("if ({o}k[0]) {o}[0] else null"),
        "bool" | "usize" | "u32" | "node" | "u64" | "f32" => format!("{o}[0]"),
        "graph" => format!("Graph({o}[0])"),
        class => format!("{class}({o}[0])"),
    };
    (d, a, e)
}

fn func(s: &mut String, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let name = ident(&f.name);
    let mut params: Vec<String> = f
        .params
        .iter()
        .filter(|p| !p.optional())
        .map(|p| format!("{}: {}", ident(&p.name), ty(&p.kind)))
        .collect();
    for p in f.params.iter().filter(|p| p.optional()) {
        let t = ty(&p.kind);
        let (t, d) = match &p.default {
            Some(l) => (t.clone(), literal(l, &p.kind)),
            None => (
                if t.ends_with('?') { t } else { format!("{t}?") },
                "null".into(),
            ),
        };
        params.push(format!("{}: {t} = {d}", ident(&p.name)));
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![],
        _ => f.rets.clone(),
    };
    let ret = match rets.as_slice() {
        [] => "Unit".to_string(),
        [(_, k)] => ret_ty(k),
        _ => format!("{}Result", crate::ir::pascal(&f.name)),
    };
    if rets.len() > 1 {
        let fields: Vec<String> = rets
            .iter()
            .map(|(n, k)| format!("val {}: {}", ident(n), ret_ty(k)))
            .collect();
        let _ = writeln!(s, "{indent}/** What `{}` returns. */", f.name);
        let _ = writeln!(s, "{indent}data class {ret}({})\n", fields.join(", "));
    }
    let doc = doc_lines(&f.doc.replace("*/", "* /"), &format!("{indent} * "));
    if !doc.trim().is_empty() {
        let _ = writeln!(s, "{indent}/**\n{doc}{indent} */");
    }
    let head = match f.role {
        Role::Constructor => format!(
            "constructor({}) : this(make({}))",
            params.join(", "),
            f.params
                .iter()
                .map(|p| ident(&p.name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => {
            // Trainer's members that the hand-written `run` (TrainerRows) calls.
            let overrides = class.as_deref() == Some("Trainer")
                && matches!(f.name.as_str(), "step" | "steps" | "set_lr" | "grad_norm");
            let prefix = match f.role {
                Role::Function | Role::Static => "@JvmStatic ",
                _ if overrides => "override ",
                _ => "",
            };
            format!("{prefix}fun {name}({}): {ret}", params.join(", "))
        }
    };
    if f.role == Role::Constructor {
        let _ = writeln!(s, "{indent}{head}\n");
        return;
    }
    let _ = writeln!(s, "{indent}{head} {{");
    body(s, f, &format!("{indent}    "), class.as_deref());
    let _ = writeln!(s, "{indent}}}\n");
}

/// The call: marshal, call `Native`, check, convert the results.
fn body(s: &mut String, f: &Func, b: &str, class: Option<&str>) {
    let mut pre = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("h".to_string());
    }
    for p in &f.params {
        args.extend(lower(p, &ident(&p.name), &mut pre));
    }
    for line in &pre {
        let _ = writeln!(s, "{b}{line}");
    }
    if f.direct {
        let _ = writeln!(s, "{b}return Native.{}({})", f.c, args.join(", "));
        return;
    }
    let all: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.unwrap().to_string())],
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
    let _ = writeln!(s, "{b}S.check(Native.{}({}))", f.c, args.join(", "));
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{b}return o0[0]");
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{b}return {}", exprs[0]);
        }
        _ => {
            let _ = writeln!(
                s,
                "{b}return {}Result({})",
                crate::ir::pascal(&f.name),
                exprs.join(", ")
            );
        }
    }
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
         @file:Suppress(\"unused\", \"UNUSED_PARAMETER\", \"FunctionName\")\n\
         \n\
         package org.rlx.kt\n\
         \n\
         /** \"No node\", for an optional node argument. */\n\
         const val NONE: Int = -1\n\n\
         /** The JNI entry points (`rlx_jni.c`), one per rlx-capi function. */\n\
         internal object Native {\n\
         \x20   init { Loader.load() }\n\n",
    );
    s.push_str(&kotlin_natives(ir));
    s.push_str("}\n\n");
    for c in &ir.classes {
        let doc = doc_lines(&c.doc.replace("*/", "* /"), " * ");
        let _ = writeln!(s, "/**\n{doc} */");
        let manual = match c.name.as_str() {
            "Optimizer" => ", OptimizerRows",
            "Trainer" => ", TrainerRows",
            _ => "",
        };
        let _ = writeln!(
            s,
            "class {n} internal constructor(h: Long) : Handle(h, Native::{free}){manual} {{",
            n = c.name,
            free = c.c_free()
        );
        let members: Vec<&Func> = ir.members(c).filter(|f| !f.manual).collect();
        for f in members
            .iter()
            .filter(|f| f.role == Role::Constructor || f.role == Role::Method)
        {
            func(&mut s, ir, f, "    ");
        }
        let statics: Vec<&&Func> = members.iter().filter(|f| f.role == Role::Static).collect();
        let ctor = members.iter().find(|f| f.role == Role::Constructor);
        if !statics.is_empty() || ctor.is_some() {
            s.push_str("    companion object {\n");
            for f in &statics {
                func(&mut s, ir, f, "        ");
            }
            if let Some(f) = ctor {
                let params: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| {
                        let t = ty(&p.kind);
                        let t = if p.optional() && p.default.is_none() && !t.ends_with('?') {
                            format!("{t}?")
                        } else {
                            t
                        };
                        format!("{}: {t}", ident(&p.name))
                    })
                    .collect();
                let _ = writeln!(
                    s,
                    "        internal fun make({}): Long {{",
                    params.join(", ")
                );
                body(&mut s, f, "            ", Some(&c.name));
                s.push_str("        }\n");
            }
            s.push_str("    }\n");
        }
        s.push_str("}\n\n");
    }
    s.push_str(
        "/** The free functions of rlx: `RLX.grad(g, wrt)`, `RLX.devices()`, … */\nobject RLX {\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, ir, f, "    ");
    }
    s.push_str("}\n");
    s
}
