// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-go/generated.go`: the whole C ABI as Go (cgo), from the lowered [`Ir`].
//!
//! One type per handle (`*Session`, freed by `Close` or a finalizer);
//! constructors are `New<Class>`, statics `<Class><Name>`. A function's
//! optional parameters are fields of its `…Opts` struct — pass `nil` for every
//! default, and a nil field keeps its default. Errors are `*rlx.Error` with the
//! `RLX_ERR_*` kind; options go in as Go values (JSON-encoded), reports come
//! back decoded.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines, pascal};

const KEYWORDS: &[&str] = &[
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
    "len",
    "cap",
    "string",
    "error",
    "copy",
    "append",
    "new",
    "make",
    "close",
    "delete",
    "panic",
    "print",
    "real",
    "imag",
];

fn local(name: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("{c}_")
    } else {
        c
    }
}

/// The Go type of an argument kind (as a parameter).
fn ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "*string",
        "json" | "json?" => "map[string]any",
        "value" | "value?" => "any",
        "bool" => "bool",
        "f32" => "float32",
        "f64" => "float64",
        "i32" => "int32",
        "i64" => "int64",
        "u32" => "uint32",
        "u64" => "uint64",
        "usize" => "int",
        "node" => "Node",
        "node?" => "*Node",
        "u64?" => "*uint64",
        "usize?" => "*int",
        "f32?" => "*float32",
        "f32s" => "[]float32",
        "u8s" => "[]byte",
        "u32s" => "[]uint32",
        "nodes" => "[]Node",
        "usizes" | "usizes?" => "[]int",
        "i64s" => "[]int64",
        "rows2" => "[][]int",
        "frows2" | "frows6" => "[][]float32",
        "strs" | "strs?" => "[]string",
        "inputs" => "map[string][]float32",
        "typed_inputs" => "map[string]TypedTensor",
        "graph" | "graph&" | "graph?" => "*Graph",
        other => return format!("*{}", other.trim_end_matches('&')),
    }
    .to_string()
}

/// Kinds whose Go type already has a "none" value (nil).
fn nilable(kind: &str) -> bool {
    let t = ty(kind);
    t.starts_with('*') || t.starts_with('[') || t.starts_with("map") || t == "any"
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "*string",
        "json" => "any",
        "object" => "map[string]any",
        "rows" => "[]map[string]any",
        "shapes" => "[][]int",
        "shape" => "Shape",
        "strings" => "[]string",
        "f32s" => "[]float32",
        "u8s" => "[]byte",
        "u32s" => "[]uint32",
        "nodes" => "[]Node",
        "outputs" => "[][]float32",
        "typed_outputs" => "[]TypedTensor",
        "tensors" => "map[string][]float32",
        "bool" => "bool",
        "usize" => "int",
        "u32" => "uint32",
        "u64" => "uint64",
        "f32" => "float32",
        "f32?" => "*float32",
        "node" => "Node",
        "graph" => "*Graph",
        other => return format!("*{other}"),
    }
    .to_string()
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "nil".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "math.Inf(1)"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let el = ty(kind).trim_start_matches("[]").to_string();
            let inner: Vec<String> = items.iter().map(|i| literal(i, "")).collect();
            format!("[]{el}{{{}}}", inner.join(", "))
        }
    }
}

/// Go statements producing the C arguments of one parameter (value `v`).
fn lower(p: &Param, v: &str, pre: &mut Vec<String>) -> Vec<String> {
    let n = format!("c{}", pascal(&p.name));
    let push = |pre: &mut Vec<String>, l: String| pre.push(l);
    match p.kind.as_str() {
        "str" => {
            push(pre, format!("{n} := C.CString({v})"));
            push(pre, format!("defer C.free(unsafe.Pointer({n}))"));
            vec![n]
        }
        "str?" => {
            push(pre, format!("{n} := cStringOrNil({v})"));
            push(pre, format!("defer freeC({n})"));
            vec![n]
        }
        "json" | "json?" | "value" | "value?" => {
            push(pre, format!("{n}, err := cJSON({v})"));
            push(pre, "if err != nil {\n\t\treturn\n\t}".into());
            push(pre, format!("defer freeC({n})"));
            vec![n]
        }
        "bool" => vec![format!("C.bool({v})")],
        "f32" => vec![format!("C.float({v})")],
        "f64" => vec![format!("C.double({v})")],
        "i32" => vec![format!("C.int32_t({v})")],
        "i64" => vec![format!("C.int64_t({v})")],
        "u32" | "node" => vec![format!("C.uint32_t({v})")],
        "node?" => vec![format!("nodeOrNone({v})")],
        "u64" => vec![format!("C.uint64_t({v})")],
        "usize" => vec![format!("C.size_t({v})")],
        "u64?" => vec![format!("optU64({v})")],
        "usize?" => vec![format!("optSize({v})")],
        "f32?" => vec![format!("optF32({v})")],
        "f32s" => {
            push(pre, format!("{n}, {n}N := cF32s({v})"));
            vec![n.clone(), format!("{n}N")]
        }
        "u8s" => {
            push(pre, format!("{n}, {n}N := cBytes({v})"));
            vec![n.clone(), format!("{n}N")]
        }
        "u32s" | "nodes" => {
            push(pre, format!("{n}, {n}N := cU32s({v})"));
            vec![n.clone(), format!("{n}N")]
        }
        "usizes" | "usizes?" => {
            push(pre, format!("{n}, {n}N := cSizes({v})"));
            vec![n.clone(), format!("{n}N")]
        }
        "i64s" => {
            push(pre, format!("{n}, {n}N := cI64s({v})"));
            vec![n.clone(), format!("{n}N")]
        }
        "rows2" => {
            push(pre, format!("{n}, {n}N, err := cRows({v}, 2, true)"));
            push(pre, "if err != nil {\n\t\treturn\n\t}".into());
            vec![n.clone(), format!("{n}N")]
        }
        "frows2" | "frows6" => {
            let w = if p.kind == "frows2" { 2 } else { 6 };
            push(pre, format!("{n}, {n}N, err := cFRows({v}, {w})"));
            push(pre, "if err != nil {\n\t\treturn\n\t}".into());
            vec![n.clone(), format!("{n}N")]
        }
        "strs" | "strs?" => {
            push(pre, format!("{n} := newCStrings({v})"));
            push(pre, format!("defer {n}.free()"));
            vec![format!("{n}.ptr"), format!("{n}.n")]
        }
        "inputs" => {
            push(pre, format!("{n} := newCInputs({v})"));
            push(pre, format!("defer {n}.free()"));
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.n"),
            ]
        }
        "typed_inputs" => {
            push(pre, format!("{n} := newCTypedInputs({v})"));
            push(pre, format!("defer {n}.free()"));
            vec![
                format!("{n}.names"),
                format!("{n}.data8"),
                format!("{n}.lens"),
                format!("{n}.dtypes"),
                format!("{n}.n"),
            ]
        }
        "graph?" => vec![format!("graphOrNil({v})")],
        _ => vec![format!("{v}.h")],
    }
}

/// `(declarations, C arguments, Go expression)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |t: &str| (vec![format!("var {o} {t}")], vec![format!("&{o}")]);
    let (d, a) = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            one("*C.char")
        }
        "f32s" => one("C.RlxF32s"),
        "u8s" => one("C.RlxBytes"),
        "u32s" | "nodes" => one("C.RlxU32s"),
        "outputs" => one("*C.RlxOutputs"),
        "typed_outputs" => one("*C.RlxTypedOutputs"),
        "tensors" => one("*C.RlxTensorMap"),
        "bool" => one("C.bool"),
        "usize" => one("C.size_t"),
        "u32" | "node" => one("C.uint32_t"),
        "u64" => one("C.uint64_t"),
        "f32" => one("C.float"),
        "f32?" => (
            vec![format!("var {o} C.float"), format!("var {o}k C.bool")],
            vec![format!("&{o}"), format!("&{o}k")],
        ),
        "graph" => one("*C.RlxGraph"),
        class => one(&format!("*C.Rlx{class}")),
    };
    let e = match kind {
        "str" => format!("takeString({o})"),
        "str?" => format!("takeStringOrNil({o})"),
        "json" => format!("takeJSON[any]({o})"),
        "object" => format!("takeJSON[map[string]any]({o})"),
        "rows" => format!("takeJSON[[]map[string]any]({o})"),
        "shapes" => format!("takeJSON[[][]int]({o})"),
        "shape" => format!("takeJSON[Shape]({o})"),
        "strings" => format!("takeJSON[[]string]({o})"),
        "f32s" => format!("takeF32s({o})"),
        "u8s" => format!("takeBytes({o})"),
        "u32s" => format!("takeU32s({o})"),
        "nodes" => format!("takeNodes({o})"),
        "outputs" => format!("takeOutputs({o})"),
        "typed_outputs" => format!("takeTypedOutputs({o})"),
        "tensors" => format!("takeTensors({o})"),
        "bool" => format!("bool({o})"),
        "usize" => format!("int({o})"),
        "u32" => format!("uint32({o})"),
        "node" => format!("Node({o})"),
        "u64" => format!("uint64({o})"),
        "f32" => format!("float32({o})"),
        "f32?" => format!("optF32Result({o}, {o}k)"),
        "graph" => format!("wrapGraph({o})"),
        class => format!("wrap{class}({o})"),
    };
    (d, a, e)
}

/// Result conversions that can fail (JSON decoding, a run's outputs).
fn fallible(e: &str) -> bool {
    e.starts_with("takeJSON")
        || e.starts_with("takeOutputs")
        || e.starts_with("takeTypedOutputs")
        || e.starts_with("takeTensors")
}

/// The Go name of a function, and of its options / result structs.
fn names(ir: &Ir, f: &Func) -> (String, String) {
    let class = ir.class_of(f).map(|c| c.name.as_str());
    let base = pascal(&f.name);
    match (f.role, class) {
        (Role::Constructor, Some(c)) => (format!("New{c}"), c.to_string()),
        (Role::Static, Some(c)) => (format!("{c}{base}"), format!("{c}{base}")),
        (Role::Method, Some(c)) => (base.clone(), format!("{c}{base}")),
        _ => (base.clone(), base),
    }
}

/// Struct fields aligned the way gofmt's tabwriter does: names to the widest,
/// trailing comments to the widest type within each run of commented lines.
fn fields(s: &mut String, rows: &[(String, String, Option<String>)]) {
    let name_w = rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    let mut i = 0;
    while i < rows.len() {
        let mut j = i;
        while j < rows.len() && rows[j].2.is_some() == rows[i].2.is_some() {
            j += 1;
        }
        let ty_w = rows[i..j]
            .iter()
            .map(|(_, t, _)| t.len())
            .max()
            .unwrap_or(0);
        for (n, t, c) in &rows[i..j] {
            match c {
                Some(c) => {
                    let _ = writeln!(s, "\t{n:<name_w$} {t:<ty_w$} // {c}");
                }
                None => {
                    let _ = writeln!(s, "\t{n:<name_w$} {t}");
                }
            }
        }
        i = j;
    }
}

fn func(s: &mut String, ir: &Ir, f: &Func) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let (name, stem) = names(ir, f);
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let opts = format!("{stem}Opts");
    if !optional.is_empty() {
        let _ = writeln!(
            s,
            "// {opts} holds the optional parameters of {name}; a nil field keeps its default."
        );
        let _ = writeln!(s, "type {opts} struct {{");
        let rows: Vec<(String, String, Option<String>)> = optional
            .iter()
            .map(|p| {
                let t = ty(&p.kind);
                let t = if nilable(&p.kind) { t } else { format!("*{t}") };
                let d = p
                    .default
                    .as_ref()
                    .map(|l| format!("default {}", literal(l, &p.kind)));
                (pascal(&p.name), t, d)
            })
            .collect();
        fields(s, &rows);
        let _ = writeln!(s, "}}\n");
    }
    let tuple = f.rets.len() > 1 && f.role != Role::Constructor;
    let result = format!("{stem}Result");
    if tuple {
        let _ = writeln!(s, "// {result} is what {name} returns.");
        let _ = writeln!(s, "type {result} struct {{");
        let rows: Vec<(String, String, Option<String>)> = f
            .rets
            .iter()
            .map(|(n, k)| (pascal(n), ret_ty(k), None))
            .collect();
        fields(s, &rows);
        let _ = writeln!(s, "}}\n");
    }
    let doc = doc_lines(&f.doc, "// ");
    if doc.trim().is_empty() {
        let _ = writeln!(s, "// {name} calls {}.", f.c);
    } else {
        let _ = write!(s, "// {name}: {}", doc.trim_start_matches("// "));
    }
    let mut params = Vec::new();
    for p in f.params.iter().filter(|p| !p.optional()) {
        params.push(format!("{} {}", local(&p.name), ty(&p.kind)));
    }
    if !optional.is_empty() {
        params.push(format!("opts *{opts}"));
    }
    let ret_kind = match (f.role, f.rets.as_slice()) {
        (Role::Constructor, _) => format!("*{}", class.clone().unwrap()),
        (_, []) => String::new(),
        (_, [(_, k)]) => ret_ty(k),
        _ => result.clone(),
    };
    let results = match (f.direct, ret_kind.is_empty()) {
        (true, _) => ret_kind.clone(),
        (false, true) => "(err error)".to_string(),
        (false, false) => format!("(res {ret_kind}, err error)"),
    };
    let recv = match f.role {
        Role::Method => format!("({} *{}) ", "self", class.clone().unwrap()),
        _ => String::new(),
    };
    let _ = writeln!(s, "func {recv}{name}({}) {results} {{", params.join(", "));
    let mut pre = Vec::new();
    for p in &optional {
        let v = local(&p.name);
        let field = pascal(&p.name);
        let t = ty(&p.kind);
        match (&p.default, nilable(&p.kind)) {
            (Some(d), true) if *d != Lit::None && t.starts_with('*') => {
                let inner = t.trim_start_matches('*');
                pre.push(format!("{v} := ptrTo({inner}({}))", literal(d, &p.kind)));
                pre.push(format!(
                    "if opts != nil && opts.{field} != nil {{\n\t\t{v} = opts.{field}\n\t}}"
                ));
            }
            (Some(d), true) if *d != Lit::None => {
                pre.push(format!("{v} := {t}({})", literal(d, &p.kind)));
                pre.push(format!(
                    "if opts != nil && opts.{field} != nil {{\n\t\t{v} = opts.{field}\n\t}}"
                ));
            }
            (_, true) => {
                pre.push(format!("var {v} {t}"));
                pre.push(format!("if opts != nil {{\n\t\t{v} = opts.{field}\n\t}}"));
            }
            (Some(d), false) => {
                pre.push(format!("var {v} {t} = {}", literal(d, &p.kind)));
                pre.push(format!(
                    "if opts != nil && opts.{field} != nil {{\n\t\t{v} = *opts.{field}\n\t}}"
                ));
            }
            (None, false) => unreachable!("an optional value kind has a default"),
        }
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("self.h".to_string());
    }
    for p in &f.params {
        args.extend(lower(p, &local(&p.name), &mut pre));
    }
    for line in &pre {
        let _ = writeln!(s, "\t{line}");
    }
    if f.direct {
        let call = format!("C.{}({})", f.c, args.join(", "));
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "\treturn C.GoString({call})");
            }
            _ => {
                let _ = writeln!(s, "\treturn bool({call})");
            }
        }
        let _ = writeln!(s, "}}\n");
        return;
    }
    let mut exprs = Vec::new();
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    for (i, (_, k)) in rets.iter().enumerate() {
        let (d, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "\t{line}");
        }
        args.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(
        s,
        "\tif err = check(C.{}({})); err != nil {{\n\t\treturn\n\t}}",
        f.c,
        args.join(", ")
    );
    match exprs.len() {
        0 => {
            let _ = writeln!(s, "\treturn nil");
        }
        1 => {
            if fallible(&exprs[0]) {
                let _ = writeln!(s, "\treturn {}", exprs[0]);
            } else {
                let _ = writeln!(s, "\treturn {}, nil", exprs[0]);
            }
        }
        _ => {
            for (i, ((n, _), e)) in rets.iter().zip(&exprs).enumerate() {
                if fallible(e) {
                    let _ = writeln!(
                        s,
                        "\tr{i}, err := {e}\n\tif err != nil {{\n\t\treturn\n\t}}"
                    );
                } else {
                    let _ = writeln!(s, "\tr{i} := {e}");
                }
                let _ = n;
            }
            let fields: Vec<String> = rets
                .iter()
                .enumerate()
                .map(|(i, (n, _))| format!("{}: r{i}", pascal(n)))
                .collect();
            let _ = writeln!(s, "\treturn {result}{{{}}}, nil", fields.join(", "));
        }
    }
    let _ = writeln!(s, "}}\n");
}

const PREAMBLE: &str = "// RLX — versatile ML compiler + runtime.\n\
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
// SPDX-License-Identifier: MIT OR Apache-2.0\n\
\n\
// Code generated by rlx-bindgen from rlx-bind's op table and runtime API table. DO NOT EDIT.\n\
// Regenerate with `just gen-bindings`.\n";

/// `(file name, contents)`: `generated.go`, plus one `generated_<feature>.go`
/// per rlx-capi feature off by default — cgo links every C function a package
/// names, so those sit behind a build tag (`-tags rlx_fpga`).
pub fn generate(ir: &Ir, defaults: &[String]) -> Vec<(String, String)> {
    let gated = |f: &Func| {
        f.feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .cloned()
    };
    let mut files: std::collections::BTreeMap<String, String> = Default::default();
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        if let Some(feature) = gated(f) {
            func(files.entry(feature).or_default(), ir, f);
        }
    }
    let mut out = vec![("generated.go".to_string(), main_file(ir, &gated))];
    let tidy = |text: String| format!("{}\n", text.trim_end());
    out[0].1 = tidy(std::mem::take(&mut out[0].1));
    for (feature, body) in files {
        let tag = format!("rlx_{}", feature.replace('-', "_"));
        out.push((
            format!("generated_{}.go", feature.replace('-', "_")),
            tidy(format!(
                "{PREAMBLE}\n//go:build {tag}\n\n// Needs an rlx-capi built with the `{feature}` feature; build with `-tags {tag}`.\n\npackage rlx\n\n// #include <stdlib.h>\n// #include \"rlx.h\"\nimport \"C\"\n\nimport \"unsafe\"\n\nvar _ = unsafe.Pointer(nil)\n\n{body}"
            )),
        ));
    }
    out
}

fn main_file(ir: &Ir, gated: &dyn Fn(&Func) -> Option<String>) -> String {
    let mut body = String::new();
    for c in &ir.classes {
        let _ = write!(body, "{}", doc_lines(&c.doc, "// "));
        let _ = writeln!(
            body,
            "type {} struct {{\n\th *C.Rlx{}\n}}\n",
            c.name, c.name
        );
        let _ = writeln!(
            body,
            "func wrap{n}(h *C.Rlx{n}) *{n} {{\n\tx := &{n}{{h: h}}\n\truntime.SetFinalizer(x, (*{n}).Close)\n\treturn x\n}}\n",
            n = c.name
        );
        let _ = writeln!(
            body,
            "// Close frees the handle now rather than at collection; it is idempotent.\nfunc (x *{n}) Close() {{\n\tif x != nil && x.h != nil {{\n\t\tC.{free}(x.h)\n\t\tx.h = nil\n\t}}\n}}\n",
            n = c.name,
            free = c.c_free()
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual && gated(f).is_none()) {
        func(&mut body, ir, f);
    }
    let mut s = String::from(PREAMBLE);
    s.push_str(
        "\n\
         package rlx\n\
         \n\
         // #include <stdlib.h>\n\
         // #include \"rlx.h\"\n\
         import \"C\"\n\
         \n\
         import (\n\
         \t\"math\"\n\
         \t\"runtime\"\n\
         \t\"unsafe\"\n\
         )\n\
         \n\
         var _ = math.Inf\n\
         var _ = unsafe.Pointer(nil)\n\n",
    );
    s.push_str(&body);
    s
}
