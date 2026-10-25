// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The OCaml binding, from the lowered [`Ir`]: two files.
//!
//! * `rlx-ocaml/src/rlx_stubs.c` — one C stub per rlx-capi function, driven
//!   by C types alone: handles are custom blocks (finalized by the class's C
//!   free function), strings and handles optional (`'a option`), arrays
//!   OCaml arrays; a failed call raises `Rlx.Error`, results come back as the
//!   value or a tuple.
//! * `rlx-ocaml/src/generated.ml` — the API (included by `rlx.ml`): a module
//!   per handle class (`Rlx.Graph.matmul g a b`), free functions at top
//!   level, optional parameters as labelled optional arguments with the
//!   tables' defaults (`Rlx.Graph.softmax ~axis:1 g x`). Options and reports
//!   are `Rlx.Json.t`.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

// ── the C stubs ─────────────────────────────────────────────

fn handle_class(ir: &Ir, c: &str) -> Option<String> {
    let name = c.strip_prefix("Rlx")?.trim_end_matches('*');
    ir.classes
        .iter()
        .find(|k| k.name == name)
        .map(|k| k.c.clone())
}

fn stub(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let ins: Vec<&(String, bool)> = params.iter().filter(|(_, o)| !*o).collect();
    let n = ins.len().max(1);
    let args: Vec<String> = (0..n).map(|i| format!("value v{i}")).collect();
    let _ = writeln!(s, "value caml_{}({}) {{", f.c, args.join(", "));
    // CAMLparam takes at most 5 per call.
    for chunk in (0..n).collect::<Vec<_>>().chunks(5).enumerate() {
        let vs: Vec<String> = chunk.1.iter().map(|i| format!("v{i}")).collect();
        let macro_ = if chunk.0 == 0 {
            "CAMLparam"
        } else {
            "CAMLxparam"
        };
        let _ = writeln!(s, "    {macro_}{}({});", vs.len(), vs.join(", "));
    }
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut call = Vec::new();
    let mut rets = Vec::new();
    let mut a = 0usize;
    for (i, (t, out)) in params.iter().enumerate() {
        if *out {
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    call.push(format!("&o{i}"));
                    let take = match st {
                        "RlxF32s" => "rlx_ml_f32s",
                        "RlxBytes" => "rlx_ml_bytes",
                        _ => "rlx_ml_u32s",
                    };
                    rets.push(format!("{take}(o{i})"));
                }
                "char**" => {
                    pre.push(format!("char* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ml_string_opt(o{i})"));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("Val_bool(o{i})"));
                }
                "float*" => {
                    pre.push(format!("float o{i} = 0;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("caml_copy_double(o{i})"));
                }
                "uint32_t*" | "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("Val_long((intnat)o{i})"));
                }
                "RlxOutputs**" => {
                    pre.push(format!("RlxOutputs* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ml_outputs(o{i})"));
                }
                "RlxTypedOutputs**" => {
                    pre.push(format!("RlxTypedOutputs* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ml_typed_outputs(o{i})"));
                }
                "RlxTensorMap**" => {
                    pre.push(format!("RlxTensorMap* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ml_tensors(o{i})"));
                }
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    let c = handle_class(ir, base).expect("an out handle is a class");
                    rets.push(format!("rlx_ml_handle(o{i}, &ops_{c})"));
                }
            }
            continue;
        }
        let v = format!("v{a}");
        a += 1;
        match t.as_str() {
            "const char*" => call.push(format!("rlx_ml_cstr({v})")),
            "bool" => call.push(format!("Bool_val({v})")),
            "float" => call.push(format!("(float)Double_val({v})")),
            "double" => call.push(format!("Double_val({v})")),
            "int32_t" | "int64_t" | "uint32_t" | "uint64_t" | "size_t" => {
                call.push(format!("({t})Long_val({v})"))
            }
            "const float*" => {
                pre.push(format!("float* c{i} = rlx_ml_floats({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const uint8_t*" => call.push(format!("rlx_ml_bytes_in({v})")),
            "const uint32_t*" => {
                pre.push(format!("uint32_t* c{i} = rlx_ml_u32s_in({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const int64_t*" => {
                pre.push(format!("int64_t* c{i} = rlx_ml_i64s_in({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const size_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                pre.push(format!("{ct}* c{i} = ({ct}*)rlx_ml_u64s_in({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const char* const*" => {
                pre.push(format!("const char** c{i} = rlx_ml_strings({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free((void*)c{i});"));
            }
            "const float* const*" => {
                pre.push(format!("RlxMlArrays c{i} = rlx_ml_float_arrays({v});"));
                call.push(format!("(const float* const*)c{i}.ptrs"));
                post.push(format!("rlx_ml_arrays_free(&c{i});"));
            }
            "const uint8_t* const*" => {
                pre.push(format!("const uint8_t** c{i} = rlx_ml_byte_arrays({v});"));
                call.push(format!("c{i}"));
                post.push(format!("free((void*)c{i});"));
            }
            handle => call.push(format!("({handle})rlx_ml_ptr({v})")),
        }
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let c = format!("{}({})", f.c, call.join(", "));
    if f.direct {
        let wrap = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "caml_copy_string",
            _ => "Val_bool",
        };
        let _ = writeln!(s, "    CAMLreturn({wrap}({c}));\n}}\n");
    } else {
        // The call runs outside the runtime lock: it may block (a mesh rank).
        let _ = writeln!(s, "    caml_release_runtime_system();");
        let _ = writeln!(s, "    int32_t rc = {c};");
        let _ = writeln!(s, "    caml_acquire_runtime_system();");
        for line in post.iter().rev() {
            let _ = writeln!(s, "    {line}");
        }
        let _ = writeln!(s, "    rlx_ml_check(rc);");
        match rets.len() {
            0 => {
                let _ = writeln!(s, "    CAMLreturn(Val_unit);\n}}\n");
            }
            1 => {
                let _ = writeln!(s, "    CAMLreturn({});\n}}\n", rets[0]);
            }
            k => {
                let _ = writeln!(s, "    CAMLlocal1(t);\n    t = caml_alloc_tuple({k});");
                for (j, r) in rets.iter().enumerate() {
                    let _ = writeln!(s, "    Store_field(t, {j}, {r});");
                }
                let _ = writeln!(s, "    CAMLreturn(t);\n}}\n");
            }
        }
    }
    if n > 5 {
        let list: Vec<String> = (0..n).map(|i| format!("argv[{i}]")).collect();
        let _ = writeln!(
            s,
            "value caml_{c}_byte(value* argv, int argn) {{\n    (void)argn;\n    return caml_{c}({});\n}}\n",
            list.join(", "),
            c = f.c
        );
    }
}

/// `src/rlx_stubs.c`.
pub fn stubs(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* The OCaml stubs: one per rlx-capi function. Conversions are in\n\
         \x20* rlx_ml_support.c.\n\
         \x20*/\n\n\
         #include \"rlx_ml.h\"\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "static void fin_{c}(value v) {{\n    void** p = (void**)Data_custom_val(v);\n    if (*p) {{ {free}((Rlx{n}*)*p); *p = NULL; }}\n}}\n\
             static struct custom_operations ops_{c} = {{\n    \"rlx.{c}\", fin_{c}, custom_compare_default, custom_hash_default,\n    custom_serialize_default, custom_deserialize_default, custom_compare_ext_default, custom_fixed_length_default}};\n\
             value caml_{free}(value v) {{\n    fin_{c}(v);\n    return Val_unit;\n}}\n",
            c = c.c,
            free = c.c_free(),
            n = c.name
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let gate = f
            .feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")));
        if let Some(g) = &gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        stub(&mut s, ir, f);
        if gate.is_some() {
            // Without the feature the stub exists and raises.
            let n = c_params(ir, f).iter().filter(|(_, o)| !*o).count().max(1);
            let args: Vec<String> = (0..n).map(|i| format!("value v{i}")).collect();
            let _ = writeln!(
                s,
                "#else\nvalue caml_{c}({}) {{\n    rlx_ml_raise(RLX_ERR_NOT_FOUND, \"{c} needs a library built with its feature\");\n    return Val_unit;\n}}",
                args.join(", "),
                c = f.c
            );
            if n > 5 {
                let _ = writeln!(
                    s,
                    "value caml_{c}_byte(value* argv, int argn) {{\n    (void)argv; (void)argn;\n    rlx_ml_raise(RLX_ERR_NOT_FOUND, \"{c} needs a library built with its feature\");\n    return Val_unit;\n}}",
                    c = f.c
                );
            }
            s.push_str("#endif\n\n");
        }
    }
    s
}

// ── the OCaml API ───────────────────────────────────────────

const KEYWORDS: &[&str] = &[
    "and",
    "as",
    "assert",
    "begin",
    "class",
    "constraint",
    "do",
    "done",
    "downto",
    "else",
    "end",
    "exception",
    "external",
    "false",
    "for",
    "fun",
    "function",
    "functor",
    "if",
    "in",
    "include",
    "inherit",
    "initializer",
    "lazy",
    "let",
    "match",
    "method",
    "module",
    "mutable",
    "new",
    "nonrec",
    "object",
    "of",
    "open",
    "or",
    "private",
    "rec",
    "sig",
    "struct",
    "then",
    "to",
    "true",
    "try",
    "type",
    "val",
    "virtual",
    "when",
    "while",
    "with",
    "land",
    "lor",
    "lxor",
    "lsl",
    "lsr",
    "asr",
    "mod",
    "effect",
];

fn ident(n: &str) -> String {
    if KEYWORDS.contains(&n) {
        format!("{n}_")
    } else {
        n.to_string()
    }
}

/// The OCaml type at the external (stub) level of a C parameter.
fn ext_ty(c: &str, ir: &Ir) -> String {
    match c {
        "bool" => "bool".into(),
        "float" | "double" => "float".into(),
        "int32_t" | "int64_t" | "uint32_t" | "uint64_t" | "size_t" => "int".into(),
        "const char*" => "string option".into(),
        "const float*" => "float array".into(),
        "const uint8_t*" => "string".into(),
        "const uint32_t*" | "const int64_t*" | "const size_t*" | "const uint64_t*" => {
            "int array".into()
        }
        "const char* const*" => "string array".into(),
        "const float* const*" => "float array array".into(),
        "const uint8_t* const*" => "string array".into(),
        h => match handle_class(ir, h) {
            Some(_) => format!("{} option", class_ty(h)),
            None => "unit".into(),
        },
    }
}

/// The abstract type of a handle class: `graph_h`, `flexible_session_h`, ….
fn handle_ty(class: &str) -> String {
    let mut s = String::new();
    for (i, ch) in class.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            s.push('_');
        }
        s.push(ch.to_ascii_lowercase());
    }
    format!("{s}_h")
}

fn class_ty(c: &str) -> String {
    handle_ty(c.trim_start_matches("Rlx").trim_end_matches('*'))
}

/// The external result type of one result kind.
fn ext_ret(kind: &str) -> String {
    match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            "string option".into()
        }
        "f32s" => "float array".into(),
        "u8s" => "string".into(),
        "u32s" | "nodes" => "int array".into(),
        "outputs" => "float array array".into(),
        "typed_outputs" => "(string * string) array".into(),
        "tensors" => "(string * float array) array".into(),
        "bool" => "bool".into(),
        "usize" | "u64" | "u32" | "node" => "int".into(),
        "f32" => "float".into(),
        "f32?" => "float * bool".into(),
        "graph" => "graph_h".into(),
        class => handle_ty(class),
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "string option",
        "json" | "value" => "Json.t",
        "json?" | "value?" => "Json.t option",
        "bool" => "bool",
        "f32" | "f64" => "float",
        "i32" | "u32" | "node" | "i64" | "u64" | "usize" => "int",
        "node?" | "u64?" | "usize?" => "int option",
        "f32?" => "float option",
        "f32s" => "float array",
        "u8s" => "string",
        "u32s" | "nodes" | "usizes" | "i64s" => "int array",
        "usizes?" => "int array option",
        "rows2" => "int array array",
        "frows2" | "frows6" => "float array array",
        "strs" => "string array",
        "strs?" => "string array option",
        "inputs" => "(string * float array) list",
        "typed_inputs" => "(string * typed_tensor) list",
        "graph" | "graph&" => "graph_h",
        "graph?" => "graph_h option",
        other => return handle_ty(other.trim_end_matches('&')),
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "None".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => {
            let float = matches!(kind, "f32" | "f64" | "f32?" | "f32s");
            let n = if n.contains("inf") {
                if n.starts_with('-') {
                    "neg_infinity".to_string()
                } else {
                    "infinity".to_string()
                }
            } else if float && !n.contains('.') && !n.contains('e') {
                format!("{n}.")
            } else {
                n.clone()
            };
            if n.starts_with('-') {
                format!("({n})")
            } else {
                n
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        // bytes are a string
        Lit::List(items) if kind == "u8s" && items.is_empty() => "\"\"".into(),
        Lit::List(items) => {
            let el = if kind.starts_with('f') { "f32" } else { "i32" };
            let inner: Vec<String> = items.iter().map(|i| literal(i, el)).collect();
            format!("[|{}|]", inner.join("; "))
        }
    }
}

/// The kind an optional parameter with no default takes: its `?` form.
fn eff_kind(p: &Param) -> String {
    let none = p.optional() && matches!(p.default, None | Some(Lit::None));
    match p.kind.as_str() {
        "str" | "json" | "value" | "node" | "u64" | "usize" | "f32" | "usizes" | "strs"
        | "graph"
            if none =>
        {
            format!("{}?", p.kind)
        }
        k => k.to_string(),
    }
}

/// OCaml expressions for one parameter's stub arguments.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match eff_kind(p).as_str() {
        "str" => vec![format!("(Some {v})")],
        "str?" => vec![v.into()],
        "json" | "value" => vec![format!("(Some (Json.to_string {v}))")],
        "json?" | "value?" => vec![format!("(Option.map Json.to_string {v})")],
        "node?" => vec![format!("(Option.value {v} ~default:none)")],
        "u64?" | "usize?" => vec![format!("(match {v} with Some x -> [|x|] | None -> [||])")],
        "f32?" => vec![format!("(match {v} with Some x -> [|x|] | None -> [||])")],
        "f32s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            vec![v.into(), format!("(Array.length {v})")]
        }
        "u8s" => vec![v.into(), format!("(String.length {v})")],
        "usizes?" => vec![
            format!("(Option.value {v} ~default:[||])"),
            format!("(Array.length (Option.value {v} ~default:[||]))"),
        ],
        "rows2" => vec![
            format!("(Support.rows {v} 2 true)"),
            format!("(Array.length {v})"),
        ],
        "frows2" => vec![
            format!("(Support.frows {v} 2)"),
            format!("(Array.length {v})"),
        ],
        "frows6" => vec![
            format!("(Support.frows {v} 6)"),
            format!("(Array.length {v})"),
        ],
        "strs" => vec![v.into(), format!("(Array.length {v})")],
        "strs?" => vec![
            format!("(Option.value {v} ~default:[||])"),
            format!("(Array.length (Option.value {v} ~default:[||]))"),
        ],
        "inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.inames"),
                format!("{n}.idata"),
                format!("{n}.ilens"),
                format!("(Array.length {n}.inames)"),
            ]
        }
        "typed_inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.tnames"),
                format!("{n}.tdata"),
                format!("{n}.tlens"),
                format!("{n}.tdtypes"),
                format!("(Array.length {n}.tnames)"),
            ]
        }
        "graph?" => vec![v.into()],
        "bool" | "f32" | "f64" | "i32" | "u32" | "node" | "i64" | "u64" | "usize" => vec![v.into()],
        _ => vec![format!("(Some {v})")],
    }
}

/// The OCaml conversion of one stub result to its kind.
fn convert(kind: &str, v: &str) -> String {
    match kind {
        "str" => format!("(Option.value {v} ~default:\"\")"),
        "json" | "object" | "rows" => {
            format!("(Json.of_string (Option.value {v} ~default:\"null\"))")
        }
        "strings" => format!("(Json.strings (Option.value {v} ~default:\"[]\"))"),
        "shape" => format!("(Json.ints (Option.value {v} ~default:\"[]\"))"),
        "shapes" => format!("(Json.int_rows (Option.value {v} ~default:\"[]\"))"),
        "typed_outputs" => format!("(Array.map (fun (data, dtype) -> {{ data; dtype }}) {v})"),
        "tensors" => format!("(Array.to_list {v})"),
        "f32?" => format!("(match {v} with (x, true) -> Some x | _ -> None)"),
        _ => v.into(),
    }
}

fn external(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let ins: Vec<String> = params
        .iter()
        .filter(|(_, o)| !*o)
        .map(|(t, _)| ext_ty(t, ir))
        .collect();
    let ins = if ins.is_empty() {
        vec!["unit".to_string()]
    } else {
        ins
    };
    let rets: Vec<String> = if f.direct {
        vec![match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "string".into(),
            _ => "bool".into(),
        }]
    } else {
        match f.role {
            Role::Constructor => vec![handle_ty(&ir.class_of(f).unwrap().name)],
            _ => f.rets.iter().map(|(_, k)| ext_ret(k)).collect(),
        }
    };
    let ret = match rets.len() {
        0 => "unit".to_string(),
        1 => rets[0].clone(),
        _ => rets
            .iter()
            .map(|r| {
                if r.contains('*') {
                    format!("({r})")
                } else {
                    r.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" * "),
    };
    let byte = if ins.len() > 5 {
        format!("\"caml_{}_byte\" ", f.c)
    } else {
        String::new()
    };
    let _ = writeln!(
        s,
        "  external c_{c} : {} -> {ret} = {byte}\"caml_{c}\"",
        ins.join(" -> "),
        c = f.c
    );
}

fn func(s: &mut String, f: &Func, indent: &str) {
    let name = match f.role {
        Role::Constructor => "create".to_string(),
        _ => ident(&f.name),
    };
    let mut ps: Vec<String> = Vec::new();
    for p in f.params.iter().filter(|p| p.optional()) {
        match &p.default {
            None | Some(Lit::None) => ps.push(format!("?{}", ident(&p.name))),
            Some(d) if p.kind.ends_with('?') => ps.push(format!(
                "?({} = Some {})",
                ident(&p.name),
                literal(d, &p.kind)
            )),
            Some(d) => ps.push(format!("?({} = {})", ident(&p.name), literal(d, &p.kind))),
        }
    }
    if f.role == Role::Method {
        ps.push("self".into());
    }
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    for p in &required {
        ps.push(format!("({} : {})", ident(&p.name), ty(&eff_kind(p))));
    }
    // An optional argument needs a positional one after it.
    if (f.role != Role::Method && required.is_empty()) || ps.is_empty() {
        ps.push("()".into());
    }
    s.push_str(&doc_lines(&f.doc, &format!("{indent}(* ")).replace('\n', " *)\n"));
    let _ = writeln!(s, "{indent}let {name} {} =", ps.join(" "));
    let b = format!("{indent}  ");
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("(Some self)".to_string());
    }
    for p in &f.params {
        let v = ident(&p.name);
        match p.kind.as_str() {
            "inputs" => {
                let _ = writeln!(s, "{b}let in_{} = Support.inputs {v} in", p.name);
            }
            "typed_inputs" => {
                let _ = writeln!(s, "{b}let in_{} = Support.typed_inputs {v} in", p.name);
            }
            _ => {}
        }
        args.extend(lower(p, &v));
    }
    if args.is_empty() {
        args.push("()".into());
    }
    let call = format!("c_{} {}", f.c, args.join(" "));
    if f.direct || f.role == Role::Constructor {
        let _ = writeln!(s, "{b}{call}\n");
        return;
    }
    match f.rets.as_slice() {
        [] => {
            let _ = writeln!(s, "{b}{call}\n");
        }
        [(_, k)] => {
            let _ = writeln!(s, "{b}let r = {call} in\n{b}{}\n", convert(k, "r"));
        }
        rs => {
            let names: Vec<String> = (0..rs.len()).map(|i| format!("r{i}")).collect();
            let conv: Vec<String> = rs
                .iter()
                .zip(&names)
                .map(|((_, k), n)| convert(k, n))
                .collect();
            let _ = writeln!(
                s,
                "{b}let ({}) = {call} in\n{b}({})\n",
                names.join(", "),
                conv.join(", ")
            );
        }
    }
}

/// `src/generated.ml`.
pub fn api(ir: &Ir) -> String {
    let mut s = String::from(
        "(* RLX — versatile ML compiler + runtime.\n\
         \x20  Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20  SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         \x20  @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20  do not edit; regenerate with `just gen-bindings`. Included by rlx.ml. *)\n\n\
         open Support\n\n",
    );
    // The handle types, then every external, then the modules.
    for c in &ir.classes {
        let _ = writeln!(s, "type {}", handle_ty(&c.name));
    }
    s.push_str("\nmodule C = struct\n");
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "  external c_{free} : {t} -> unit = \"caml_{free}\"",
            free = c.c_free(),
            t = handle_ty(&c.name)
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        external(&mut s, ir, f);
    }
    s.push_str("end\n\nopen C\n\n");
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, f, "");
    }
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "(* ").replace('\n', " *)\n"));
        let _ = writeln!(
            s,
            "module {n} = struct\n  type t = {t}\n\n  (* Free the handle now (it is unusable afterwards). *)\n  let close self = c_{free} self\n",
            n = c.name,
            t = handle_ty(&c.name),
            free = c.c_free()
        );
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut s, f, "  ");
        }
        s.truncate(s.trim_end().len());
        s.push_str("\nend\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
