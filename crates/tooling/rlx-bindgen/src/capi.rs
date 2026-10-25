// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-capi/src/graph_ops.rs`: one `extern "C"` function per op-table row.

use std::fmt::Write;

use rlx_bind::OpSpec;

use crate::kinds::{c_convert, c_note, c_params, c_rets, c_write};

pub fn generate(ops: &[OpSpec]) -> String {
    let mut s = String::new();
    s.push_str(crate::HEADER_RS);
    s.push_str(
        "//! The graph builders: one `rlx_graph_<op>` per row of rlx-bind's op table.\n\
         //!\n\
         //! Each converts its C arguments to wire values, borrows the graph, and calls\n\
         //! `rlx_bind::ops::<op>`. Options-group arguments are flattened after the\n\
         //! positional ones, in table order.\n\n\
         use rlx_bind::ApiError;\n\n\
         use crate::ffi::{arg, guard, out};\n\
         use crate::graph::{RlxGraph, graph_mut, take_graph};\n",
    );
    for op in ops {
        function(&mut s, op);
    }
    s
}

fn function(s: &mut String, op: &OpSpec) {
    let cname = format!("rlx_graph_{}", op.c_name());
    s.push('\n');
    for line in op.doc.lines().map(|l| l.strip_prefix(' ').unwrap_or(l)) {
        let _ = writeln!(
            s,
            "///{}{}",
            if line.is_empty() { "" } else { " " },
            line.trim_end()
        );
    }
    let notes: Vec<String> = op.args.iter().filter_map(c_note).collect();
    if !notes.is_empty() {
        if !op.doc.is_empty() {
            s.push_str("///\n");
        }
        for n in notes {
            let _ = writeln!(s, "/// - {n}");
        }
    }
    let _ = writeln!(s, "#[unsafe(no_mangle)]");
    let mut params = vec!["graph: *mut RlxGraph".to_string()];
    for a in op.args {
        for (name, ty) in c_params(a) {
            params.push(format!("{name}: {ty}"));
        }
    }
    for (name, ty) in c_rets(op.ret) {
        params.push(format!("{name}: {ty}"));
    }
    let _ = writeln!(s, "pub unsafe extern \"C\" fn {cname}(");
    for p in &params {
        let _ = writeln!(s, "    {p},");
    }
    s.push_str(") -> i32 {\n    guard(|| unsafe {\n");
    // A body graph that is the graph itself would alias two `&mut`s.
    for a in op
        .args
        .iter()
        .filter(|a| a.kind == "graph" || a.kind == "opt_graph")
    {
        let _ = writeln!(
            s,
            "        if {n} == graph {{\n            return Err(ApiError::type_(\"{cname}: `{n}` cannot be the graph being built\"));\n        }}",
            n = a.name
        );
    }
    let mut call_args = Vec::new();
    for a in op.args {
        let conv = c_convert(a, &cname);
        if conv == a.name {
            call_args.push(a.name.to_string());
        } else {
            let _ = writeln!(s, "        let {} = {conv};", a.name);
            call_args.push(a.name.to_string());
        }
    }
    let _ = writeln!(s, "        let g = graph_mut(graph)?;");
    let joined = if call_args.is_empty() {
        String::new()
    } else {
        format!(", {}", call_args.join(", "))
    };
    if op.ret == "unit" {
        let _ = writeln!(s, "        rlx_bind::ops::{}(g{joined})", op.name);
    } else {
        let _ = writeln!(s, "        let r = rlx_bind::ops::{}(g{joined})?;", op.name);
        let _ = writeln!(s, "        {}", c_write(op.ret, &cname));
    }
    s.push_str("    })\n}\n");
}
