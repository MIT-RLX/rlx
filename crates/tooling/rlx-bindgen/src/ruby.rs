// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-ruby/lib/rlx/generated.rb`: the whole C ABI as Ruby (stdlib Fiddle),
//! from the lowered [`Ir`].
//!
//! One class per handle type (freed by a finalizer); optional parameters are
//! keyword arguments with the tables' defaults; statics are class methods.
//! Options go in as Ruby values (encoded to JSON), reports come back parsed.
//! Two names that would override `Object` methods Ruby itself relies on take
//! a Ruby spelling: `Group#send` is `send_to`, `Trainer#freeze` / `unfreeze`
//! are `freeze_params` / `unfreeze_params`.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};

/// Ruby spellings for names that would shadow core `Object` methods.
pub fn ruby_name(class: Option<&str>, name: &str) -> String {
    match (class, name) {
        (Some("Group"), "send") => "send_to".into(),
        (Some("Trainer"), "freeze") => "freeze_params".into(),
        (Some("Trainer"), "unfreeze") => "unfreeze_params".into(),
        _ => name.to_string(),
    }
}

const KEYWORDS: &[&str] = &[
    "alias", "and", "begin", "break", "case", "class", "def", "defined?", "do", "else", "elsif",
    "end", "ensure", "false", "for", "if", "in", "module", "next", "nil", "not", "or", "redo",
    "rescue", "retry", "return", "self", "super", "then", "true", "undef", "unless", "until",
    "when", "while", "yield",
];

/// A parameter name usable as a local and a keyword.
fn local(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "nil".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "Float::INFINITY"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

fn pack(kind: &str) -> &'static str {
    match kind {
        "f32s" | "frows2" | "frows6" => "f",
        "u8s" => "C",
        "u32s" | "nodes" => "L",
        "i64s" => "q",
        _ => "J",
    }
}

/// `(Fiddle types, Ruby argument expressions)` for one parameter, with
/// statements first (locals keep their buffers alive through the call).
fn lower(p: &Param, v: &str, pre: &mut Vec<String>) -> (Vec<&'static str>, Vec<String>) {
    let n = format!("a_{}", p.name);
    match p.kind.as_str() {
        "str" => {
            pre.push(format!("{n} = RLX.cstr({v})"));
            (vec!["VOIDP"], vec![n])
        }
        "str?" => {
            pre.push(format!("{n} = {v}.nil? ? nil : RLX.cstr({v})"));
            (vec!["VOIDP"], vec![n])
        }
        "json" | "value" => {
            pre.push(format!("{n} = RLX.cstr(JSON.generate({v}))"));
            (vec!["VOIDP"], vec![n])
        }
        "json?" | "value?" => {
            pre.push(format!(
                "{n} = {v}.nil? ? nil : RLX.cstr(JSON.generate({v}))"
            ));
            (vec!["VOIDP"], vec![n])
        }
        "bool" => (vec!["BOOL"], vec![v.into()]),
        "f32" => (vec!["FLOAT"], vec![format!("{v}.to_f")]),
        "f64" => (vec!["DOUBLE"], vec![format!("{v}.to_f")]),
        "i32" => (vec!["INT32_T"], vec![v.into()]),
        "i64" => (vec!["INT64_T"], vec![v.into()]),
        "u32" | "node" => (vec!["UINT32_T"], vec![v.into()]),
        "node?" => (vec!["UINT32_T"], vec![format!("({v} || NONE)")]),
        "u64" => (vec!["UINT64_T"], vec![v.into()]),
        "usize" => (vec!["SIZE_T"], vec![v.into()]),
        "u64?" | "usize?" | "f32?" => {
            let fmt = if p.kind == "f32?" { "f" } else { "Q" };
            pre.push(format!("{n} = {v}.nil? ? nil : RLX.buf(\"{fmt}\", [{v}])"));
            (vec!["VOIDP"], vec![n])
        }
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "i64s" => {
            pre.push(format!("{n} = RLX.buf(\"{}\", {v})", pack(&p.kind)));
            (vec!["VOIDP", "SIZE_T"], vec![n, format!("{v}.size")])
        }
        "usizes?" => {
            pre.push(format!("{n} = {v}.nil? ? nil : RLX.buf(\"J\", {v})"));
            (
                vec!["VOIDP", "SIZE_T"],
                vec![n, format!("({v}.nil? ? 0 : {v}.size)")],
            )
        }
        "rows2" | "frows2" | "frows6" => {
            let (w, splat) = match p.kind.as_str() {
                "rows2" => (2, "true"),
                "frows2" => (2, "false"),
                _ => (6, "false"),
            };
            pre.push(format!(
                "{n} = RLX.buf(\"{}\", RLX.rows({v}, {w}, {splat}))",
                pack(&p.kind)
            ));
            (vec!["VOIDP", "SIZE_T"], vec![n, format!("{v}.size")])
        }
        "strs" | "strs?" => {
            pre.push(format!("{n} = RLX.cstrs({v})"));
            (
                vec!["VOIDP", "SIZE_T"],
                vec![format!("{n}.ptr"), format!("{n}.size")],
            )
        }
        "inputs" => {
            pre.push(format!("{n} = RLX.inputs({v})"));
            (
                vec!["VOIDP", "VOIDP", "VOIDP", "SIZE_T"],
                vec![
                    format!("{n}.names"),
                    format!("{n}.data"),
                    format!("{n}.lens"),
                    format!("{n}.size"),
                ],
            )
        }
        "typed_inputs" => {
            pre.push(format!("{n} = RLX.typed_inputs({v})"));
            (
                vec!["VOIDP", "VOIDP", "VOIDP", "VOIDP", "SIZE_T"],
                vec![
                    format!("{n}.names"),
                    format!("{n}.data"),
                    format!("{n}.lens"),
                    format!("{n}.dtypes"),
                    format!("{n}.size"),
                ],
            )
        }
        "graph?" => (vec!["VOIDP"], vec![format!("({v}&.handle)")]),
        _ => (vec!["VOIDP"], vec![format!("{v}.handle")]),
    }
}

/// `(statements, Fiddle types, arguments, result expression)`.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<&'static str>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |bytes: usize| {
        (
            vec![format!("{o} = RLX.out({bytes})")],
            vec!["VOIDP"],
            vec![o.clone()],
        )
    };
    let (d, t, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => one(16),
        "bool" => one(1),
        "u32" | "node" | "f32" => one(4),
        "f32?" => (
            vec![format!("{o} = RLX.out(4)"), format!("{o}k = RLX.out(1)")],
            vec!["VOIDP", "VOIDP"],
            vec![o.clone(), format!("{o}k")],
        ),
        _ => one(8),
    };
    let e = match kind {
        "str" => format!("RLX.take_string({o})"),
        "str?" => format!("RLX.take_string_or_nil({o})"),
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            format!("JSON.parse(RLX.take_string({o}))")
        }
        "f32s" => format!("RLX.take_array({o}, \"f\", 4, :rlx_f32s_release)"),
        "u8s" => format!("RLX.take_array({o}, \"C\", 1, :rlx_bytes_release)"),
        "u32s" | "nodes" => format!("RLX.take_array({o}, \"L\", 4, :rlx_u32s_release)"),
        "outputs" => format!("RLX.take_outputs({o})"),
        "typed_outputs" => format!("RLX.take_typed_outputs({o})"),
        "tensors" => format!("RLX.take_tensors({o})"),
        "bool" => format!("({o}[0, 1].unpack1(\"C\") != 0)"),
        "usize" => format!("{o}[0, 8].unpack1(\"J\")"),
        "u32" | "node" => format!("{o}[0, 4].unpack1(\"L\")"),
        "u64" => format!("{o}[0, 8].unpack1(\"Q\")"),
        "f32" => format!("{o}[0, 4].unpack1(\"f\")"),
        "f32?" => format!("({o}k[0, 1].unpack1(\"C\") != 0 ? {o}[0, 4].unpack1(\"f\") : nil)"),
        "graph" => format!("Graph.wrap(RLX.ptr_at({o}))"),
        class => format!("{class}.wrap(RLX.ptr_at({o}))"),
    };
    (d, t, a, e)
}

fn func(s: &mut String, ir: &Ir, f: &Func, indent: &str, sigs: &mut Vec<String>) {
    let class = ir.class_of(f).map(|c| c.name.as_str());
    let name = ruby_name(class, &f.name);
    let mut params = Vec::new();
    for p in f.params.iter().filter(|p| !p.optional()) {
        params.push(local(&p.name));
    }
    for p in f.params.iter().filter(|p| p.optional()) {
        let d = p
            .default
            .as_ref()
            .map(literal)
            .unwrap_or_else(|| "nil".into());
        params.push(format!("{}: {d}", p.name));
    }
    s.push_str(&doc_lines(&f.doc, &format!("{indent}# ")));
    let def = match f.role {
        Role::Constructor => "def initialize".to_string(),
        Role::Static | Role::Function => format!("def self.{name}"),
        _ => format!("def {name}"),
    };
    let _ = writeln!(s, "{indent}{def}({})", params.join(", "));
    let body = format!("{indent}  ");
    // Optional parameters keep the keyword spelling; a keyword that is a Ruby
    // word reads through `binding`.
    for p in f
        .params
        .iter()
        .filter(|p| p.optional() && KEYWORDS.contains(&p.name.as_str()))
    {
        let _ = writeln!(
            s,
            "{body}{} = binding.local_variable_get(:{})",
            local(&p.name),
            p.name
        );
    }
    let mut pre = Vec::new();
    let mut types: Vec<&str> = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        types.push("VOIDP");
        args.push("@handle".to_string());
    }
    for p in &f.params {
        let (t, a) = lower(p, &local(&p.name), &mut pre);
        types.extend(t);
        args.extend(a);
    }
    for line in &pre {
        let _ = writeln!(s, "{body}{line}");
    }
    if f.direct {
        let ret = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "VOIDP",
            _ => "BOOL",
        };
        sigs.push(format!("    {}: [[{}], :{ret}],", f.c, sym_list(&types)));
        let call = format!("RLX.call(:{}, {})", f.c, args.join(", "));
        if ret == "VOIDP" {
            let _ = writeln!(s, "{body}{call}.to_s");
        } else {
            let _ = writeln!(s, "{body}{call}");
        }
        let _ = writeln!(s, "{indent}end\n");
        return;
    }
    let rets: Vec<(String, String)> = f.rets.clone();
    let mut exprs = Vec::new();
    for (i, (_, k)) in rets.iter().enumerate() {
        let (d, t, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "{body}{line}");
        }
        types.extend(t);
        args.extend(a);
        exprs.push(e);
    }
    sigs.push(format!("    {}: [[{}], :INT32_T],", f.c, sym_list(&types)));
    let _ = writeln!(
        s,
        "{body}RLX.check_rc(RLX.call(:{}, {}))",
        f.c,
        args.join(", ")
    );
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{body}adopt(RLX.ptr_at(o0))");
        }
        (_, 0) => {
            let _ = writeln!(s, "{body}nil");
        }
        (_, 1) => {
            let _ = writeln!(s, "{body}{}", exprs[0]);
        }
        _ => {
            let parts: Vec<String> = rets
                .iter()
                .zip(&exprs)
                .map(|((n, _), e)| format!("{n}: {e}"))
                .collect();
            let _ = writeln!(s, "{body}{{ {} }}", parts.join(", "));
        }
    }
    let _ = writeln!(s, "{indent}end\n");
}

fn sym_list(types: &[&str]) -> String {
    types
        .iter()
        .map(|t| format!(":{t}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn generate(ir: &Ir) -> String {
    let mut body = String::new();
    let mut sigs = Vec::new();
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut body, ir, f, "  ", &mut sigs);
    }
    for c in &ir.classes {
        body.push_str(&doc_lines(&c.doc, "  # "));
        let _ = writeln!(body, "  class {} < Handle", c.name);
        let _ = writeln!(body, "    FREE = :{}\n", c.c_free());
        sigs.push(format!("    {}: [[:VOIDP], :VOID],", c.c_free()));
        if !ir.members(c).any(|f| f.role == Role::Constructor) {
            body.push_str("    private_class_method :new\n\n");
        }
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut body, ir, f, "    ", &mut sigs);
        }
        body.push_str("  end\n\n");
    }
    let mut s = String::from(
        "# RLX — versatile ML compiler + runtime.\n\
         # Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         # SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         # @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         # do not edit; regenerate with `just gen-bindings`. Loaded by lib/rlx.rb.\n\
         \n\
         # frozen_string_literal: true\n\
         \n\
         module RLX\n\
         \x20 # Every C function: [argument types, return type] (Fiddle::TYPE_*).\n\
         \x20 SIGNATURES = {\n",
    );
    for line in &sigs {
        s.push_str(line);
        s.push('\n');
    }
    s.push_str("  }.freeze\n\n");
    s.push_str(&body);
    s.push_str("end\n");
    s
}
