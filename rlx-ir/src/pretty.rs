// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Annotated graph dump for debugging.
//!
//! `Graph` already has a basic `Display` impl that emits one line per
//! node. [`pretty_print`] extends that with a header containing
//! per-op-kind counts, summed arena bytes, and per-node markers for
//! outputs / named nodes — the form you actually want when staring at
//! a 200-node lowered graph trying to find why one shape is wrong.
//!
//! [`pretty_stats`] returns only the header (no per-node body) when
//! you just want a one-shot "what does this graph contain" summary.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::Graph;

/// Detailed annotated graph dump. Format:
///
/// ```text
/// graph @name (12 nodes, 2 outputs, 1.4 MB arena)
///   op kinds: MatMul=4, Activation=3, ...
///
///   %0 [input "x"]    = input("x") : [4, 15, 384] f32
///   %1 [param "wQ"]   = param("wQ") : [384, 1152] f32
///   %2                = matmul(%0, %1) : [4, 15, 1152] f32
///   ...
///   %11               = layer_norm(...)(%10) : [4, 15, 384] f32  ← output
///   return %11
/// ```
pub fn pretty_print(g: &Graph) -> String {
    let mut out = String::new();
    writeln!(out, "{}", header_line(g)).unwrap();
    writeln!(out, "{}", op_kinds_line(g)).unwrap();
    writeln!(out).unwrap();

    // Compute the column width for the optional [name] tag so the
    // op column lines up.
    let mut tag_w = 0usize;
    for n in g.nodes() {
        let t = node_tag(n.id, n.name.as_deref(), &n.op);
        if t.len() > tag_w {
            tag_w = t.len();
        }
    }

    for n in g.nodes() {
        let tag = node_tag(n.id, n.name.as_deref(), &n.op);
        write!(out, "  {tag:<width$} = {}", n.op, width = tag_w).unwrap();
        if !n.inputs.is_empty() {
            write!(out, "(").unwrap();
            for (i, inp) in n.inputs.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{inp}").unwrap();
            }
            write!(out, ")").unwrap();
        }
        write!(out, " : {}", n.shape).unwrap();
        if g.outputs.contains(&n.id) {
            write!(out, "  ← output").unwrap();
        }
        writeln!(out).unwrap();
    }
    if !g.outputs.is_empty() {
        write!(out, "  return ").unwrap();
        for (i, o) in g.outputs.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{o}").unwrap();
        }
        writeln!(out).unwrap();
    }
    out
}

/// One- to two-line summary: header + op-kind histogram. No body.
pub fn pretty_stats(g: &Graph) -> String {
    format!("{}\n{}", header_line(g), op_kinds_line(g))
}

fn header_line(g: &Graph) -> String {
    let arena_bytes: usize = g.nodes().iter().filter_map(|n| n.shape.size_bytes()).sum();
    format!(
        "graph @{} ({} nodes, {} outputs, {} arena)",
        g.name,
        g.len(),
        g.outputs.len(),
        human_bytes(arena_bytes),
    )
}

fn op_kinds_line(g: &Graph) -> String {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for n in g.nodes() {
        *hist.entry(format!("{:?}", n.op.kind())).or_insert(0) += 1;
    }
    let mut entries: Vec<(String, usize)> = hist.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let parts: Vec<String> = entries.iter().map(|(k, c)| format!("{k}={c}")).collect();
    format!("  op kinds: {}", parts.join(", "))
}

fn node_tag(id: crate::NodeId, name: Option<&str>, op: &crate::Op) -> String {
    use crate::Op;
    // For Input/Param the op name already includes the user-given
    // string — surface it as the tag without re-printing on the line.
    let label: Option<String> = match op {
        Op::Input { name } => Some(format!("input \"{name}\"")),
        Op::Param { name } => Some(format!("param \"{name}\"")),
        _ => name.map(|s| format!("\"{s}\"")),
    };
    match label {
        Some(s) => format!("{id} [{s}]"),
        None => format!("{id}"),
    }
}

fn human_bytes(b: usize) -> String {
    const K: f64 = 1024.0;
    let bf = b as f64;
    if bf < K {
        format!("{b} B")
    } else if bf < K * K {
        format!("{:.1} KB", bf / K)
    } else if bf < K * K * K {
        format!("{:.1} MB", bf / (K * K))
    } else {
        format!("{:.1} GB", bf / (K * K * K))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph, Shape, op::BinaryOp};

    #[test]
    fn pretty_print_smoke() {
        let mut g = Graph::new("smoke");
        let x = g.input("x", Shape::new(&[4, 4], DType::F32));
        let y = g.input("y", Shape::new(&[4, 4], DType::F32));
        let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4, 4], DType::F32));
        g.set_outputs(vec![z]);
        let s = pretty_print(&g);
        assert!(s.contains("graph @smoke"));
        assert!(s.contains("nodes"));
        assert!(s.contains("Input=2"));
        assert!(s.contains("Binary=1"));
        assert!(s.contains("← output"));
        assert!(s.contains("return %2"));
    }

    #[test]
    fn pretty_stats_no_body() {
        let mut g = Graph::new("s");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.input("y", Shape::new(&[4], DType::F32));
        let _ = g.binary(BinaryOp::Mul, x, y, Shape::new(&[4], DType::F32));
        let s = pretty_stats(&g);
        assert!(s.contains("3 nodes"));
        assert!(!s.contains("%0 = input"));
    }

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KB");
        assert_eq!(human_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.0 GB");
    }

    #[test]
    fn outputs_marker_present() {
        let mut g = Graph::new("o");
        let a = g.input("a", Shape::new(&[2], DType::F32));
        let b = g.input("b", Shape::new(&[2], DType::F32));
        let c = g.binary(BinaryOp::Add, a, b, Shape::new(&[2], DType::F32));
        let d = g.binary(BinaryOp::Add, c, a, Shape::new(&[2], DType::F32));
        g.set_outputs(vec![d]);
        let s = pretty_print(&g);
        let lines: Vec<&str> = s.lines().collect();
        // Only one line should have "← output".
        let count = lines.iter().filter(|l| l.contains("← output")).count();
        assert_eq!(count, 1, "expected exactly one output marker, got {count}");
    }
}
