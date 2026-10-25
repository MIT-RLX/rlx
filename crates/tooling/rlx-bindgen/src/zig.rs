// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-zig/src/rlx.zig`: the whole C ABI as Zig — one struct per handle type,
//! one function per C function, from the lowered [`Ir`].
//!
//! Zig has no default arguments, so a function's optional parameters are
//! fields of a `<Name>Opts` struct with the tables' defaults (`.{}` takes them
//! all). Results rlx allocates are copied into the caller's allocator and the
//! C buffer freed, so a function that returns data takes an `alloc`. Options
//! and reports are JSON text (parse them with `std.json`).

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines, pascal};

const KEYWORDS: &[&str] = &[
    "align",
    "allowzero",
    "and",
    "anyframe",
    "anytype",
    "asm",
    "break",
    "callconv",
    "catch",
    "comptime",
    "const",
    "continue",
    "defer",
    "else",
    "enum",
    "errdefer",
    "error",
    "export",
    "extern",
    "fn",
    "for",
    "if",
    "inline",
    "noalias",
    "nosuspend",
    "opaque",
    "or",
    "orelse",
    "packed",
    "pub",
    "resume",
    "return",
    "linksection",
    "struct",
    "suspend",
    "switch",
    "test",
    "threadlocal",
    "try",
    "union",
    "unreachable",
    "usingnamespace",
    "var",
    "volatile",
    "while",
    "type",
    "self",
];

fn ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("@\"{name}\"")
    } else {
        name.to_string()
    }
}

/// The Zig type a caller passes for an argument kind.
fn arg_ty(kind: &str) -> String {
    match kind {
        "str" | "json" | "value" => "[:0]const u8",
        "str?" | "json?" | "value?" => "?[:0]const u8",
        "bool" => "bool",
        "f32" => "f32",
        "f64" => "f64",
        "i32" => "i32",
        "i64" => "i64",
        "u32" => "u32",
        "u64" => "u64",
        "usize" => "usize",
        "node" => "Node",
        "node?" => "?Node",
        "u64?" => "?u64",
        "usize?" => "?usize",
        "f32?" => "?f32",
        "f32s" => "[]const f32",
        "u8s" => "[]const u8",
        "u32s" => "[]const u32",
        "nodes" => "[]const Node",
        "usizes" => "[]const usize",
        "usizes?" => "?[]const usize",
        "i64s" => "[]const i64",
        "rows2" => "[]const [2]usize",
        "frows2" => "[]const [2]f32",
        "frows6" => "[]const [6]f32",
        "strs" => "[]const [*:0]const u8",
        "strs?" => "?[]const [*:0]const u8",
        "inputs" => "[]const Input",
        "typed_inputs" => "[]const TypedInput",
        "graph" | "graph&" => "Graph",
        "graph?" => "?Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => "[]u8",
        "str?" => "?[]u8",
        "f32s" => "[]f32",
        "u8s" => "[]u8",
        "u32s" => "[]u32",
        "nodes" => "[]Node",
        "outputs" => "[][]f32",
        "typed_outputs" => "[]TypedTensor",
        "tensors" => "[]Tensor",
        "bool" => "bool",
        "usize" => "usize",
        "u32" => "u32",
        "u64" => "u64",
        "f32" => "f32",
        "f32?" => "?f32",
        "node" => "Node",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

/// Results that land in the caller's allocator.
fn ret_allocates(kind: &str) -> bool {
    !matches!(
        kind,
        "bool" | "usize" | "u32" | "u64" | "f32" | "f32?" | "node" | "graph"
    ) && !kind.starts_with(char::is_uppercase)
}

/// Arguments that need temporary C arrays.
fn arg_allocates(kind: &str) -> bool {
    matches!(kind, "inputs" | "typed_inputs")
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.clone(),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("&.{{ {} }}", inner.join(", "))
        }
    }
}

/// The struct holding a function's optional parameters, if any.
fn opts_name(f: &Func, class: Option<&str>, ir: &Ir) -> String {
    let base = if f.role == Role::Constructor {
        "Init".to_string()
    } else {
        pascal(&f.name)
    };
    // Zig calls a name declared both in a struct and at file scope ambiguous.
    let clash = class.is_some() && ir.free_funcs().any(|g| g.name == f.name);
    match class {
        Some(c) if clash => format!("{c}{base}Opts"),
        _ => format!("{base}Opts"),
    }
}

fn optional_params(f: &Func) -> Vec<&Param> {
    f.params.iter().filter(|p| p.optional()).collect()
}

/// Expressions for the C arguments of one parameter, and statements before the call.
fn lower(p: &Param, v: &str, pre: &mut Vec<String>) -> Vec<String> {
    let n = &p.name;
    match p.kind.as_str() {
        "str" | "json" | "value" => vec![format!("{v}.ptr")],
        "str?" | "json?" | "value?" => vec![format!("if ({v}) |rlx_s| rlx_s.ptr else null")],
        "node?" => vec![format!("{v} orelse NONE")],
        "u64?" | "usize?" | "f32?" => {
            pre.push(format!("const {n}_v = {v};"));
            vec![format!("if ({n}_v) |*rlx_x| rlx_x else null")]
        }
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            vec![format!("{v}.ptr"), format!("{v}.len")]
        }
        "usizes?" => vec![
            format!("if ({v}) |rlx_s| rlx_s.ptr else null"),
            format!("if ({v}) |rlx_s| rlx_s.len else 0"),
        ],
        "rows2" | "frows2" | "frows6" => vec![format!("@ptrCast({v}.ptr)"), format!("{v}.len")],
        "strs" => vec![format!("@ptrCast({v}.ptr)"), format!("{v}.len")],
        "strs?" => vec![
            format!("if ({v}) |rlx_s| @ptrCast(rlx_s.ptr) else null"),
            format!("if ({v}) |rlx_s| rlx_s.len else 0"),
        ],
        "inputs" => {
            pre.push(format!("const {n}_c = try support.cInputs(alloc, {v});"));
            pre.push(format!("defer {n}_c.deinit(alloc);"));
            vec![
                format!("{n}_c.names.ptr"),
                format!("{n}_c.data.ptr"),
                format!("{n}_c.lens.ptr"),
                format!("{v}.len"),
            ]
        }
        "typed_inputs" => {
            pre.push(format!(
                "const {n}_c = try support.cTypedInputs(alloc, {v});"
            ));
            pre.push(format!("defer {n}_c.deinit(alloc);"));
            vec![
                format!("{n}_c.names.ptr"),
                format!("{n}_c.data.ptr"),
                format!("{n}_c.lens.ptr"),
                format!("{n}_c.dtypes.ptr"),
                format!("{v}.len"),
            ]
        }
        "graph" | "graph&" => vec![format!("{v}.h")],
        "graph?" => vec![format!("if ({v}) |rlx_g| rlx_g.h else null")],
        k if k.ends_with('&') => vec![format!("{v}.h")],
        _ => vec![v.to_string()],
    }
}

/// Out-variable declarations, their C arguments, and the result expression.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("out{i}");
    let one = |decl: String| (vec![decl], vec![format!("&{o}")]);
    let (decls, args) = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            one(format!("var {o}: [*c]u8 = null;"))
        }
        "f32s" => one(format!(
            "var {o}: c.RlxF32s = .{{ .ptr = null, .len = 0 }};"
        )),
        "u8s" => one(format!(
            "var {o}: c.RlxBytes = .{{ .ptr = null, .len = 0 }};"
        )),
        "u32s" | "nodes" => one(format!(
            "var {o}: c.RlxU32s = .{{ .ptr = null, .len = 0 }};"
        )),
        "outputs" => one(format!("var {o}: ?*c.RlxOutputs = null;")),
        "typed_outputs" => one(format!("var {o}: ?*c.RlxTypedOutputs = null;")),
        "tensors" => one(format!("var {o}: ?*c.RlxTensorMap = null;")),
        "bool" => one(format!("var {o}: bool = false;")),
        "usize" => one(format!("var {o}: usize = 0;")),
        "u32" | "node" => one(format!("var {o}: u32 = 0;")),
        "u64" => one(format!("var {o}: u64 = 0;")),
        "f32" => one(format!("var {o}: f32 = 0;")),
        "f32?" => (
            vec![
                format!("var {o}: f32 = 0;"),
                format!("var {o}_known: bool = false;"),
            ],
            vec![format!("&{o}"), format!("&{o}_known")],
        ),
        "graph" => one(format!("var {o}: ?*c.RlxGraph = null;")),
        class => one(format!("var {o}: ?*c.Rlx{class} = null;")),
    };
    let expr = match kind {
        "str" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            format!("try support.takeString(alloc, {o})")
        }
        "str?" => format!("if ({o} == null) null else try support.takeString(alloc, {o})"),
        "f32s" => format!("try support.takeSlice(f32, alloc, {o}, c.rlx_f32s_free)"),
        "u8s" => format!("try support.takeSlice(u8, alloc, {o}, c.rlx_bytes_free)"),
        "u32s" | "nodes" => format!("try support.takeSlice(u32, alloc, {o}, c.rlx_u32s_free)"),
        "outputs" => format!("try support.takeOutputs(alloc, {o}.?)"),
        "typed_outputs" => format!("try support.takeTypedOutputs(alloc, {o}.?)"),
        "tensors" => format!("try support.takeTensors(alloc, {o}.?)"),
        "f32?" => format!("if ({o}_known) {o} else null"),
        "bool" | "usize" | "u32" | "u64" | "f32" | "node" => o.clone(),
        "graph" => format!("Graph{{ .h = {o}.? }}"),
        class => format!("{class}{{ .h = {o}.? }}"),
    };
    (decls, args, expr)
}

fn opts_struct(s: &mut String, f: &Func, name: &str, indent: &str) {
    let opt = optional_params(f);
    if opt.is_empty() {
        return;
    }
    let _ = writeln!(s, "{indent}/// The optional parameters of `{}`.", f.name);
    let _ = writeln!(s, "{indent}pub const {name} = struct {{");
    for p in opt {
        let ty = arg_ty(&p.kind);
        let ty = if ty.starts_with('?') || p.default.is_some() {
            ty
        } else {
            format!("?{ty}")
        };
        let d = p
            .default
            .as_ref()
            .map(literal)
            .unwrap_or_else(|| "null".into());
        let _ = writeln!(s, "{indent}    {}: {ty} = {d},", ident(&p.name));
    }
    let _ = writeln!(s, "{indent}}};");
}

/// A parameter's Zig name: Zig refuses a parameter that shadows any
/// declaration in scope, so one named like a function or type gets a `_`.
fn param_ident(name: &str, taken: &std::collections::BTreeSet<String>) -> String {
    if taken.contains(name) {
        format!("{name}_")
    } else {
        ident(name)
    }
}

fn func(
    s: &mut String,
    ir: &Ir,
    f: &Func,
    indent: &str,
    taken: &std::collections::BTreeSet<String>,
) {
    let class = ir.class_of(f).map(|c| c.name.as_str());
    let name = match f.role {
        Role::Constructor => "init".to_string(),
        _ => ident(&f.name),
    };
    let opts = opts_name(f, class, ir);
    opts_struct(s, f, &opts, indent);
    let tuple = f.rets.len() > 1;
    let result_name = opts.replace("Opts", "Result");
    if tuple {
        let _ = writeln!(s, "{indent}pub const {result_name} = struct {{");
        for (n, k) in &f.rets {
            let _ = writeln!(s, "{indent}    {}: {},", ident(n), ret_ty(k));
        }
        let _ = writeln!(s, "{indent}}};");
    }
    s.push_str(&doc_lines(&f.doc, &format!("{indent}/// ")));

    let needs_alloc = !f.direct
        && (f.rets.iter().any(|(_, k)| ret_allocates(k))
            || f.params.iter().any(|p| arg_allocates(&p.kind)));
    let mut sig = Vec::new();
    if f.role == Role::Method {
        sig.push(format!("self: {}", class.unwrap()));
    }
    if needs_alloc {
        sig.push("alloc: std.mem.Allocator".into());
    }
    for p in f.params.iter().filter(|p| !p.optional()) {
        sig.push(format!(
            "{}: {}",
            param_ident(&p.name, taken),
            arg_ty(&p.kind)
        ));
    }
    let has_opts = !optional_params(f).is_empty();
    if has_opts {
        sig.push(format!("opts: {opts}"));
    }
    let ret = match f.rets.as_slice() {
        [] => "void".to_string(),
        [(_, k)] => ret_ty(k),
        _ => result_name.clone(),
    };
    let ret = match (f.direct, ret.as_str()) {
        (true, "[]u8") => "[]const u8".to_string(),
        (true, _) => ret,
        _ => format!("Error!{ret}"),
    };
    let _ = writeln!(s, "{indent}pub fn {name}({}) {ret} {{", sig.join(", "));
    let body = format!("{indent}    ");
    let mut pre = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("self.h".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("opts.{}", ident(&p.name))
        } else {
            param_ident(&p.name, taken)
        };
        // An optional with a non-null default is a plain value of its kind.
        args.extend(lower(p, &v, &mut pre));
    }
    for line in &pre {
        let _ = writeln!(s, "{body}{line}");
    }
    if f.direct {
        let call = format!("c.{}({})", f.c, args.join(", "));
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "{body}return std.mem.span({call});");
            }
            _ => {
                let _ = writeln!(s, "{body}return {call};");
            }
        }
        let _ = writeln!(s, "{indent}}}");
        return;
    }
    let mut exprs = Vec::new();
    for (i, (_, k)) in f.rets.iter().enumerate() {
        let (decls, out_args, expr) = outs(k, i);
        for d in decls {
            let _ = writeln!(s, "{body}{d}");
        }
        args.extend(out_args);
        exprs.push(expr);
    }
    let _ = writeln!(
        s,
        "{body}try support.check(c.{}({}));",
        f.c,
        args.join(", ")
    );
    match exprs.len() {
        0 => {}
        1 => {
            let _ = writeln!(s, "{body}return {};", exprs[0]);
        }
        _ => {
            let fields: Vec<String> = f
                .rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!(".{} = {e}", ident(n)))
                .collect();
            let _ = writeln!(s, "{body}return .{{ {} }};", fields.join(", "));
        }
    }
    let _ = writeln!(s, "{indent}}}");
}

/// Every function is emitted: Zig analyses only what a program calls, so one
/// behind an rlx-capi feature (`export_fpga`) links only when it is used.
pub fn generate(ir: &Ir) -> String {
    let mut taken: std::collections::BTreeSet<String> = ir
        .funcs
        .iter()
        .flat_map(|f| {
            [
                f.name.clone(),
                format!("{}Opts", pascal(&f.name)),
                format!("{}Result", pascal(&f.name)),
            ]
        })
        .chain(ir.classes.iter().map(|c| c.name.clone()))
        .collect();
    for n in [
        "std",
        "support",
        "c",
        "Error",
        "Node",
        "NONE",
        "Input",
        "TypedInput",
        "TypedTensor",
        "Tensor",
        "lastError",
        "StepParam",
        "RunCallbacks",
        "batchSet",
        "init",
        "deinit",
        "opts",
        "alloc",
        "h",
    ] {
        taken.insert(n.to_string());
    }
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         //\n\
         // Every rlx-capi function, as Zig. Optional parameters are fields of each\n\
         // function's `…Opts` struct (pass `.{}` for the defaults); data rlx returns\n\
         // is copied into `alloc`; options and reports are JSON text.\n\
         \n\
         const std = @import(\"std\");\n\
         const support = @import(\"support.zig\");\n\
         pub const c = support.c;\n\
         pub const Error = support.Error;\n\
         pub const Node = support.Node;\n\
         pub const NONE = support.NONE;\n\
         pub const Input = support.Input;\n\
         pub const TypedInput = support.TypedInput;\n\
         pub const TypedTensor = support.TypedTensor;\n\
         pub const Tensor = support.Tensor;\n\
         pub const lastError = support.lastError;\n\
         pub const StepParam = support.StepParam;\n\
         pub const RunCallbacks = support.RunCallbacks;\n\
         pub const batchSet = support.batchSet;\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        s.push('\n');
        func(&mut s, ir, f, "", &taken);
    }
    for class in &ir.classes {
        s.push('\n');
        s.push_str(&doc_lines(&class.doc, "/// "));
        let _ = writeln!(s, "pub const {} = struct {{", class.name);
        let _ = writeln!(s, "    h: *c.Rlx{},\n", class.name);
        let _ = writeln!(s, "    /// Free the handle.");
        let _ = writeln!(
            s,
            "    pub fn deinit(self: {}) void {{\n        c.{}(self.h);\n    }}",
            class.name,
            class.c_free()
        );
        for f in ir.members(class) {
            s.push('\n');
            if f.manual {
                let _ = writeln!(
                    s,
                    "    pub const {} = support.{}_{};",
                    ident(&f.name),
                    class.c,
                    f.name
                );
                continue;
            }
            func(&mut s, ir, f, "    ", &taken);
        }
        s.push_str("};\n");
    }
    s
}
