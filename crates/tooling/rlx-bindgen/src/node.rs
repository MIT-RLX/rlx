// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-node/lib/rlx.js` + `rlx.d.ts`: the whole C ABI for Node.js (over the
//! `koffi` FFI), from the lowered [`Ir`].
//!
//! One class per handle type (`close()`, `Symbol.dispose`, and a
//! `FinalizationRegistry`); names are rlx-js's camelCase. Required parameters
//! are positional, optional ones a trailing options object
//! (`g.softmax(x, { axis: 1 })`). Options go in as JS values (JSON), reports
//! come back parsed.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, c_args, c_outs, camel, doc_lines};

const RESERVED: &[&str] = &[
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "new",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "let",
    "static",
    "enum",
    "await",
    "implements",
    "package",
    "protected",
    "interface",
    "private",
    "public",
    "arguments",
    "eval",
];

fn local(name: &str) -> String {
    let c = camel(name);
    if RESERVED.contains(&c.as_str()) {
        format!("{c}_")
    } else {
        c
    }
}

/// A C parameter type in koffi's prototype syntax; `out` marks a result.
fn proto_ty(c: &str, out: bool) -> String {
    let t = match c {
        "const char* const*" => "const char**".to_string(),
        "const float* const*" => "const float**".to_string(),
        "const uint8_t* const*" => "const uint8_t**".to_string(),
        "char**" => "void**".to_string(),
        t if t.starts_with("Rlx") && t.ends_with("**") => "void**".to_string(),
        t if t.starts_with("RlxF32s") || t.starts_with("RlxBytes") || t.starts_with("RlxU32s") => {
            t.to_string()
        }
        t if t.starts_with("Rlx") => "void*".to_string(),
        t => t.to_string(),
    };
    if out { format!("_Out_ {t}") } else { t }
}

fn prototype(ir: &Ir, f: &Func) -> String {
    let mut params = Vec::new();
    if f.role == Role::Method {
        params.push("void*".to_string());
    }
    for p in &f.params {
        params.extend(c_args(&p.kind).iter().map(|t| proto_ty(t, false)));
    }
    let ret = if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "const char*",
            _ => "bool",
        }
    } else {
        let rets: Vec<String> = match f.role {
            Role::Constructor => vec![ir.class_of(f).unwrap().name.clone()],
            _ => f.rets.iter().map(|(_, k)| k.clone()).collect(),
        };
        for k in rets {
            params.extend(c_outs(&k).iter().map(|t| proto_ty(t, true)));
        }
        "int32_t"
    };
    format!("{ret} {}({})", f.c, params.join(", "))
}

fn ts_ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "string | null",
        "json" | "value" => "unknown",
        "json?" | "value?" => "unknown",
        "bool" => "boolean",
        "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "node" => "number",
        "node?" | "u64?" | "usize?" | "f32?" => "number | null",
        "f32s" => "ArrayLike<number>",
        "u8s" => "Uint8Array | ArrayLike<number>",
        "u32s" | "nodes" | "usizes" | "i64s" => "ArrayLike<number>",
        "usizes?" => "ArrayLike<number> | null",
        "rows2" | "frows2" | "frows6" => "ArrayLike<number>[]",
        "strs" => "string[]",
        "strs?" => "string[] | null",
        "inputs" => "Inputs",
        "typed_inputs" => "TypedInputs",
        "graph" | "graph&" => "Graph",
        "graph?" => "Graph | null",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ts_ret(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "string | null",
        "json" => "unknown",
        "object" => "Record<string, any>",
        "rows" => "Record<string, any>[]",
        "shapes" => "number[][]",
        "shape" => "Shape",
        "strings" => "string[]",
        "f32s" => "Float32Array",
        "u8s" => "Uint8Array",
        "u32s" | "nodes" => "Uint32Array",
        "outputs" => "Float32Array[]",
        "typed_outputs" => "TypedTensor[]",
        "tensors" => "Record<string, Float32Array>",
        "bool" => "boolean",
        "usize" | "u32" | "u64" | "f32" | "node" => "number",
        "f32?" => "number | null",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "Infinity"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// JS expressions for one parameter's C arguments.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "str" => vec![v.into()],
        "str?" => vec![format!("{v} ?? null")],
        "json" | "value" => vec![format!("JSON.stringify({v})")],
        "json?" | "value?" => vec![format!("S.json({v})")],
        "node?" => vec![format!("{v} ?? S.NONE")],
        "u64?" | "usize?" | "f32?" => vec![format!("{v} == null ? null : [{v}]")],
        "f32s" => vec![format!("S.f32({v})"), format!("{v}.length")],
        "u8s" => vec![format!("S.u8({v})"), format!("{v}.length")],
        "u32s" | "nodes" => vec![format!("S.u32({v})"), format!("{v}.length")],
        "usizes" | "i64s" => vec![format!("Array.from({v})"), format!("{v}.length")],
        "usizes?" => vec![
            format!("{v} == null ? null : Array.from({v})"),
            format!("{v} == null ? 0 : {v}.length"),
        ],
        "rows2" => vec![format!("S.rows({v}, 2, true)"), format!("{v}.length")],
        "frows2" => vec![
            format!("S.f32(S.rows({v}, 2, false))"),
            format!("{v}.length"),
        ],
        "frows6" => vec![
            format!("S.f32(S.rows({v}, 6, false))"),
            format!("{v}.length"),
        ],
        "strs" => vec![v.into(), format!("{v}.length")],
        "strs?" => vec![
            format!("{v} ?? null"),
            format!("{v} == null ? 0 : {v}.length"),
        ],
        "inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.names.length"),
            ]
        }
        "typed_inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.dtypes"),
                format!("{n}.names.length"),
            ]
        }
        "graph?" => vec![format!("{v} == null ? null : {v}.handle")],
        "bool" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "node" => vec![v.into()],
        _ => vec![format!("{v}.handle")],
    }
}

/// `(declarations, arguments, result expression)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (vec![format!("const {o} = {{}};")], vec![o.clone()]),
        "f32?" => (
            vec![
                format!("const {o} = [0];"),
                format!("const {o}k = [false];"),
            ],
            vec![o.clone(), format!("{o}k")],
        ),
        "bool" => (vec![format!("const {o} = [false];")], vec![o.clone()]),
        "usize" | "u32" | "node" | "u64" | "f32" => {
            (vec![format!("const {o} = [0];")], vec![o.clone()])
        }
        _ => (vec![format!("const {o} = [null];")], vec![o.clone()]),
    };
    let e = match kind {
        "str" => format!("S.takeString({o}[0])"),
        "str?" => format!("{o}[0] == null ? null : S.takeString({o}[0])"),
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            format!("JSON.parse(S.takeString({o}[0]))")
        }
        "f32s" => format!("S.takeArray({o}, \"float\", \"rlx_f32s_free\")"),
        "u8s" => format!("S.takeArray({o}, \"uint8_t\", \"rlx_bytes_free\")"),
        "u32s" | "nodes" => format!("S.takeArray({o}, \"uint32_t\", \"rlx_u32s_free\")"),
        "outputs" => format!("S.takeOutputs({o}[0])"),
        "typed_outputs" => format!("S.takeTypedOutputs({o}[0])"),
        "tensors" => format!("S.takeTensors({o}[0])"),
        "f32?" => format!("{o}k[0] ? {o}[0] : null"),
        "bool" | "usize" | "u32" | "node" | "u64" | "f32" => format!("{o}[0]"),
        "graph" => format!("S.wrap(Graph, {o}[0])"),
        class => format!("S.wrap({class}, {o}[0])"),
    };
    (d, a, e)
}

fn js_doc(doc: &str, indent: &str) -> String {
    if doc.trim().is_empty() {
        return String::new();
    }
    let body = doc_lines(&doc.replace("*/", "* /"), &format!("{indent} * "));
    format!("{indent}/**\n{body}{indent} */\n")
}

fn func(js: &mut String, ts: &mut String, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    // A static rlx-js spells as a free function is the class's static under
    // its own name here, plus that free-function alias (see `generate`).
    let own = camel(&f.name);
    let name = if f.js_free { &own } else { &f.js };
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let mut params: Vec<String> = required.iter().map(|p| local(&p.name)).collect();
    if !optional.is_empty() {
        params.push("opts = {}".into());
    }
    // ── the .d.ts declaration
    let mut ts_params: Vec<String> = required
        .iter()
        .map(|p| format!("{}: {}", local(&p.name), ts_ty(&p.kind)))
        .collect();
    if !optional.is_empty() {
        let fields: Vec<String> = optional
            .iter()
            .map(|p| {
                let d = p
                    .default
                    .as_ref()
                    .map(|l| format!(" /* default {} */", literal(l)))
                    .unwrap_or_default();
                format!("{}?: {}{d}", camel(&p.name), ts_ty(&p.kind))
            })
            .collect();
        ts_params.push(format!("opts?: {{ {} }}", fields.join("; ")));
    }
    let ret = match (f.role, f.rets.as_slice()) {
        (Role::Constructor, _) => String::new(),
        (_, []) => "void".into(),
        (_, [(_, k)]) => ts_ret(k),
        (_, rets) => format!(
            "{{ {} }}",
            rets.iter()
                .map(|(n, k)| format!("{}: {}", camel(n), ts_ret(k)))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    };
    ts.push_str(&js_doc(&f.doc, indent));
    let _ = match f.role {
        Role::Constructor => writeln!(ts, "{indent}constructor({});", ts_params.join(", ")),
        Role::Static => writeln!(
            ts,
            "{indent}static {name}({}): {ret};",
            ts_params.join(", ")
        ),
        Role::Function => writeln!(
            ts,
            "export function {name}({}): {ret};",
            ts_params.join(", ")
        ),
        Role::Method => writeln!(ts, "{indent}{name}({}): {ret};", ts_params.join(", ")),
    };

    // ── the implementation
    js.push_str(&js_doc(&f.doc, indent));
    let head = match f.role {
        Role::Constructor => format!("constructor({})", params.join(", ")),
        Role::Static => format!("static {name}({})", params.join(", ")),
        Role::Function => format!("function {name}({})", params.join(", ")),
        Role::Method => format!("{name}({})", params.join(", ")),
    };
    let _ = writeln!(js, "{indent}{head} {{");
    let b = format!("{indent}  ");
    if f.role == Role::Constructor {
        let _ = writeln!(js, "{b}super();");
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("this.handle".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            let d = p
                .default
                .as_ref()
                .map(literal)
                .unwrap_or_else(|| "null".into());
            format!("(opts.{} ?? {d})", camel(&p.name))
        } else {
            local(&p.name)
        };
        match p.kind.as_str() {
            "inputs" => {
                let _ = writeln!(js, "{b}const in_{} = S.inputs({v});", p.name);
            }
            "typed_inputs" => {
                let _ = writeln!(js, "{b}const in_{} = S.typedInputs({v});", p.name);
            }
            _ => {}
        }
        args.extend(lower(p, &v));
    }
    let call = |args: &[String]| {
        format!(
            "S.fn(\"{}\", \"{}\")({})",
            f.c,
            prototype(ir, f),
            args.join(", ")
        )
    };
    if f.direct {
        let _ = writeln!(js, "{b}return {};", call(&args));
        let _ = writeln!(js, "{indent}}}\n");
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
            let _ = writeln!(js, "{b}{line}");
        }
        args.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(js, "{b}S.check({});", call(&args));
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(
                js,
                "{b}S.adopt(this, o0[0], \"{}\");",
                ir.class_of(f).unwrap().c_free()
            );
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(js, "{b}return {};", exprs[0]);
        }
        _ => {
            let parts: Vec<String> = all
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{}: {e}", camel(n)))
                .collect();
            let _ = writeln!(js, "{b}return {{ {} }};", parts.join(", "));
        }
    }
    let _ = writeln!(js, "{indent}}}\n");
}

/// `(rlx.js, rlx.d.ts)`.
pub fn generate(ir: &Ir) -> (String, String) {
    let banner = "// RLX — versatile ML compiler + runtime.\n\
                  // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
                  // SPDX-License-Identifier: MIT OR Apache-2.0\n\
                  \n\
                  // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
                  // do not edit; regenerate with `just gen-bindings`.\n";
    let mut js = format!("{banner}\n\"use strict\";\n\nconst S = require(\"./support\");\n\n");
    let mut ts = format!(
        "{banner}\n\
         /** A node of a graph — the id every builder returns. */\n\
         export type Node = number;\n\
         /** \"No node\", for an optional node argument. */\n\
         export const NONE: number;\n\
         /** Named f32 tensors: `{{ x: Float32Array | number[] }}`. */\n\
         export type Inputs = Record<string, ArrayLike<number>>;\n\
         /** A tensor of any dtype as raw bytes. */\n\
         export interface TypedTensor {{ data: ArrayBufferView | ArrayLike<number>; dtype: string }}\n\
         export type TypedInputs = Record<string, TypedTensor>;\n\
         /** A node's static dims (-1: dynamic) and dtype. */\n\
         export interface Shape {{ dims: number[]; dtype: string }}\n\
         /** A failed rlx call, with its `RLX_ERR_*` kind. */\n\
         export class RlxError extends Error {{ kind: \"type\" | \"range\" | \"runtime\" | \"panic\" | \"not_found\" | \"other\" }}\n\n"
    );
    for c in &ir.classes {
        js.push_str(&js_doc(&c.doc, ""));
        let _ = writeln!(js, "class {} extends S.Handle {{", c.name);
        ts.push_str(&js_doc(&c.doc, ""));
        let _ = writeln!(ts, "export class {} {{", c.name);
        let _ = writeln!(
            ts,
            "  /** Free the handle now (also `using x = …`). */\n  close(): void;"
        );
        let _ = writeln!(js, "  static get free() {{ return \"{}\"; }}\n", c.c_free());
        for f in ir.members(c) {
            if f.manual {
                let sig = match (c.name.as_str(), f.name.as_str()) {
                    ("Optimizer", "step") => {
                        "  /** Update each `params[name]` in place from `grads[name]`. */\n  step(params: Record<string, Float32Array>, grads: Record<string, ArrayLike<number>>): void;"
                    }
                    ("Trainer", "run") => {
                        "  /** Train `steps` steps, pulling inputs from `batch(step)` (null: data exhausted). */\n  run(steps: number, batch: (step: number) => Inputs | null, opts?: { lr?: (step: number) => number; onStep?: (step: number, loss: number) => boolean | void; every?: number }): Record<string, any>;"
                    }
                    _ => "",
                };
                let _ = writeln!(ts, "{sig}");
                continue;
            }
            func(&mut js, &mut ts, ir, f, "  ");
        }
        js.push_str("}\n\n");
        ts.push_str("}\n\n");
    }
    for f in ir.free_funcs().filter(|f| !f.manual) {
        let mut dummy = String::new();
        func(&mut js, &mut dummy, ir, f, "");
        ts.push_str(&dummy);
    }
    js.push_str("S.attachManual({ Optimizer, Trainer });\n\n");
    // rlx-js's top-level spellings of statics (`openGguf`, `loadTokenizer`).
    let mut aliases = Vec::new();
    for f in ir.funcs.iter().filter(|f| f.js_free) {
        let class = &ir.class_of(f).unwrap().name;
        let own = camel(&f.name);
        let _ = writeln!(js, "const {} = (...a) => {class}.{own}(...a);", f.js);
        let _ = writeln!(
            ts,
            "/** The same as `{class}.{own}`. */\nexport const {}: typeof {class}.{own};",
            f.js
        );
        aliases.push(f.js.clone());
    }
    let exports: Vec<String> = ir
        .classes
        .iter()
        .map(|c| c.name.clone())
        .chain(ir.free_funcs().filter(|f| !f.manual).map(|f| f.js.clone()))
        .chain(aliases)
        .collect();
    let _ = writeln!(
        js,
        "module.exports = {{ NONE: S.NONE, RlxError: S.RlxError, {} }};",
        exports.join(", ")
    );
    (js, ts)
}
