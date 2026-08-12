// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Textual IR that parses back — the format for writing pass tests as text.
//!
//! [`Graph`]'s `Display` is a one-way debug dump: `Op`'s `Display` is lossy
//! (`const({}B)` prints a byte count, `quantize(s=…)` drops zero-points), so
//! nothing can read it back. The practical consequence is that every pass test
//! in this workspace builds its input by hand in Rust and asserts on Rust
//! structures. That is expensive enough per test that coverage stays thin,
//! which is how three backends ended up wrong *and agreeing* about RoPE table
//! strides, and how the opcode tables in [`crate::opcodes`] drifted apart in
//! the first place.
//!
//! [`print`] and [`parse`] are exact inverses, so a pass test can be a pair of
//! strings:
//!
//! ```
//! # #[cfg(feature = "serialize")] {
//! use rlx_ir::text;
//!
//! let g = text::parse(r#"
//! graph @fma {
//!   %0 = {"Input":{"name":"a"}} : [8] f32
//!   %1 = {"Input":{"name":"b"}} : [8] f32
//!   %2 = {"Input":{"name":"c"}} : [8] f32
//!   %3 = Fma(%0, %1, %2) : [8] f32
//!   return %3
//! }
//! "#).unwrap();
//!
//! assert_eq!(g.len(), 4);
//! assert_eq!(text::parse(&text::print(&g)).unwrap().fingerprint(), g.fingerprint());
//! # }
//! ```
//!
//! # Grammar
//!
//! ```text
//! graph      := "graph" "@" ident "{" node* return? "}"
//! node       := "%" int "=" op operands? ":" shape name?
//! operands   := "(" ("%" int),* ")"
//! shape      := "[" dim,* "]" dtype
//! dim        := int | "?" int          // "?3" is dynamic symbol 3
//! name       := "#" string             // optional debug label
//! return     := "return" ("%" int),*
//! op         := ident | json           // see below
//! ```
//!
//! # Why the op payload is JSON
//!
//! [`Op`] has 184 variants. Hand-writing a printer *and* a parser per variant
//! is 184 opportunities to disagree with each other, and a new op silently
//! gets neither. Instead the payload rides the `serde` derives the enum
//! already carries, so every present and future variant round-trips with no
//! per-op code at all.
//!
//! The one concession to readability: a unit variant serialises to a bare JSON
//! string, and printing it as `MatMul` rather than `"MatMul"` covers the
//! majority of nodes. Anything with fields keeps its JSON object —
//! `{"Reshape":{"new_shape":[4,16]}}` — which is verbose but unambiguous and
//! writable. That is a presentation choice over a stable skeleton: a prettier
//! payload syntax can be swapped in later without touching the grammar above.
//!
//! # What round-trips
//!
//! Everything except [`Node::origin`], which is pass provenance rather than
//! IR — it is re-stamped by whichever pass runs next, so serialising it would
//! bake one pipeline's history into a test fixture. Concretely,
//! `parse(print(g))` equals `g` under
//! [`IgnoreConfig::EXACT`](crate::IgnoreConfig::EXACT) apart from origins, and
//! its [`fingerprint`](crate::fingerprint) matches exactly.

#![cfg(feature = "serialize")]

use std::fmt::Write as _;

use crate::dtype::DType;
use crate::graph::{Graph, NodeId};
use crate::op::Op;
use crate::shape::{Dim, Shape};

/// Failure to parse textual IR, with the 1-based line it occurred on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1-based line number.
    pub line: usize,
    /// What went wrong.
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

// ── Printing ────────────────────────────────────────────────────

/// Render `op` as either a bare identifier (unit variant) or a JSON value.
fn print_op(op: &Op) -> String {
    let json = serde_json::to_string(op).unwrap_or_else(|e| format!("\"<unserializable: {e}>\""));
    // A unit variant serialises to `"Name"`; drop the quotes when the content
    // is a plain identifier so the common case reads as `MatMul`.
    if let Some(inner) = json.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
        && !inner.is_empty()
        && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !inner.starts_with(|c: char| c.is_ascii_digit())
    {
        return inner.to_string();
    }
    json
}

fn print_shape(shape: &Shape) -> String {
    let dims: Vec<String> = shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => n.to_string(),
            Dim::Dynamic(s) => format!("?{s}"),
        })
        .collect();
    format!("[{}] {}", dims.join(", "), shape.dtype())
}

/// Render `graph` in the textual form [`parse`] accepts.
pub fn print(graph: &Graph) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "graph @{} {{", graph.name);

    for node in graph.nodes() {
        let _ = write!(out, "  {} = {}", node.id, print_op(&node.op));
        if !node.inputs.is_empty() {
            let operands: Vec<String> = node.inputs.iter().map(|i| i.to_string()).collect();
            let _ = write!(out, "({})", operands.join(", "));
        }
        let _ = write!(out, " : {}", print_shape(&node.shape));
        if let Some(name) = &node.name {
            let _ = write!(out, " #{}", serde_json::to_string(name).unwrap_or_default());
        }
        let _ = writeln!(out);
    }

    if !graph.outputs.is_empty() {
        let outs: Vec<String> = graph.outputs.iter().map(|o| o.to_string()).collect();
        let _ = writeln!(out, "  return {}", outs.join(", "));
    }
    let _ = writeln!(out, "}}");
    out
}

// ── Parsing ─────────────────────────────────────────────────────

struct Cursor<'a> {
    rest: &'a str,
    line: usize,
}

impl<'a> Cursor<'a> {
    fn err<T>(&self, message: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            line: self.line,
            message: message.into(),
        })
    }

    fn skip_ws(&mut self) {
        self.rest = self.rest.trim_start_matches([' ', '\t']);
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_ws();
        match self.rest.strip_prefix(token) {
            Some(remaining) => {
                self.rest = remaining;
                true
            }
            None => false,
        }
    }

    fn expect(&mut self, token: &str) -> Result<(), ParseError> {
        if self.eat(token) {
            Ok(())
        } else {
            self.err(format!(
                "expected `{token}`, found `{}`",
                self.peek_snippet()
            ))
        }
    }

    fn peek_snippet(&self) -> String {
        self.rest.chars().take(24).collect()
    }

    fn take_while(&mut self, pred: impl Fn(char) -> bool) -> &'a str {
        let end = self.rest.find(|c| !pred(c)).unwrap_or(self.rest.len());
        let (taken, remaining) = self.rest.split_at(end);
        self.rest = remaining;
        taken
    }

    fn parse_usize(&mut self) -> Result<usize, ParseError> {
        self.skip_ws();
        let digits = self.take_while(|c| c.is_ascii_digit());
        match digits.parse() {
            Ok(n) => Ok(n),
            Err(_) => self.err(format!(
                "expected an integer, found `{}`",
                self.peek_snippet()
            )),
        }
    }

    /// `%12` → `NodeId(12)`.
    fn parse_node_id(&mut self) -> Result<NodeId, ParseError> {
        self.expect("%")?;
        Ok(NodeId(self.parse_usize()? as u32))
    }

    /// Consume one balanced JSON value (object, array or string).
    ///
    /// Tracks string state so that braces or brackets inside a string literal
    /// — a node named `"}"`, say — do not close the value early.
    fn take_json(&mut self) -> Result<&'a str, ParseError> {
        self.skip_ws();
        let bytes = self.rest.as_bytes();
        let Some(&first) = bytes.first() else {
            return self.err("expected a JSON value");
        };
        if !matches!(first, b'{' | b'[' | b'"') {
            return self.err(format!(
                "expected a JSON value, found `{}`",
                self.peek_snippet()
            ));
        }

        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for (i, &b) in bytes.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                    if depth == 0 {
                        let (taken, rest) = self.rest.split_at(i + 1);
                        self.rest = rest;
                        return Ok(taken);
                    }
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        let (taken, rest) = self.rest.split_at(i + 1);
                        self.rest = rest;
                        return Ok(taken);
                    }
                }
                _ => {}
            }
        }
        self.err("unterminated JSON value")
    }

    fn parse_op(&mut self) -> Result<Op, ParseError> {
        self.skip_ws();
        let json = if self.rest.starts_with(['{', '[', '"']) {
            self.take_json()?.to_string()
        } else {
            // Bare identifier — re-quote it into the unit-variant JSON form.
            let ident = self.take_while(|c| c.is_ascii_alphanumeric() || c == '_');
            if ident.is_empty() {
                return self.err(format!("expected an op, found `{}`", self.peek_snippet()));
            }
            format!("\"{ident}\"")
        };
        match serde_json::from_str(&json) {
            Ok(op) => Ok(op),
            Err(e) => self.err(format!("not a valid op: {e} (in `{json}`)")),
        }
    }

    fn parse_dtype(&mut self) -> Result<DType, ParseError> {
        self.skip_ws();
        let word = self.take_while(|c| c.is_ascii_alphanumeric());
        // Mirrors `DType`'s Display, which is what `print_shape` emits.
        let dtype = match word {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "f64" => DType::F64,
            "i8" => DType::I8,
            "i16" => DType::I16,
            "i32" => DType::I32,
            "i64" => DType::I64,
            "u8" => DType::U8,
            "u32" => DType::U32,
            "bool" => DType::Bool,
            "c64" => DType::C64,
            "c128" => DType::C128,
            other => return self.err(format!("unknown dtype `{other}`")),
        };
        Ok(dtype)
    }

    fn parse_shape(&mut self) -> Result<Shape, ParseError> {
        self.expect("[")?;
        let mut dims = Vec::new();
        if !self.eat("]") {
            loop {
                self.skip_ws();
                let dim = if self.eat("?") {
                    Dim::Dynamic(self.parse_usize()? as u32)
                } else {
                    Dim::Static(self.parse_usize()?)
                };
                dims.push(dim);
                if self.eat(",") {
                    continue;
                }
                self.expect("]")?;
                break;
            }
        }
        Ok(Shape::from_dims(&dims, self.parse_dtype()?))
    }
}

/// Parse textual IR produced by [`print`].
pub fn parse(text: &str) -> Result<Graph, ParseError> {
    let mut lines = text.lines().enumerate().peekable();

    // Header: `graph @name {`
    let mut name = None;
    for (idx, raw) in lines.by_ref() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let mut cur = Cursor {
            rest: line,
            line: idx + 1,
        };
        cur.expect("graph")?;
        cur.expect("@")?;
        cur.skip_ws();
        let parsed = cur.take_while(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
        if parsed.is_empty() {
            return cur.err("expected a graph name after `@`");
        }
        name = Some(parsed.to_string());
        cur.expect("{")?;
        break;
    }
    let Some(name) = name else {
        return Err(ParseError {
            line: 1,
            message: "empty input: expected `graph @name {`".into(),
        });
    };

    let mut graph = Graph::new(name);
    // Textual ids need not be dense or ordered, so map them explicitly rather
    // than assuming `%3` lands at index 3.
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let mut outputs: Vec<NodeId> = Vec::new();
    let mut closed = false;

    for (idx, raw) in lines {
        let line = raw.trim();
        let lineno = idx + 1;
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line == "}" {
            closed = true;
            break;
        }

        let mut cur = Cursor {
            rest: line,
            line: lineno,
        };

        if cur.eat("return") {
            loop {
                outputs.push(resolve(&id_map, cur.parse_node_id()?, &cur)?);
                if !cur.eat(",") {
                    break;
                }
            }
            continue;
        }

        // `%id = op(operands) : shape #"name"`
        let declared = cur.parse_node_id()?;
        cur.expect("=")?;
        let op = cur.parse_op()?;

        let mut inputs = Vec::new();
        if cur.eat("(") {
            if !cur.eat(")") {
                loop {
                    inputs.push(resolve(&id_map, cur.parse_node_id()?, &cur)?);
                    if cur.eat(",") {
                        continue;
                    }
                    cur.expect(")")?;
                    break;
                }
            }
        }

        cur.expect(":")?;
        let shape = cur.parse_shape()?;

        let label = if cur.eat("#") {
            let json = cur.take_json()?;
            match serde_json::from_str::<String>(json) {
                Ok(s) => Some(s),
                Err(e) => return cur.err(format!("bad node name: {e}")),
            }
        } else {
            None
        };

        cur.skip_ws();
        if !cur.rest.is_empty() {
            return cur.err(format!("trailing input `{}`", cur.peek_snippet()));
        }

        let new_id = graph.add_node(op, inputs, shape);
        if label.is_some() {
            graph.node_mut(new_id).name = label;
        }
        if id_map.insert(declared, new_id).is_some() {
            return Err(ParseError {
                line: lineno,
                message: format!("{declared} is defined twice"),
            });
        }
    }

    if !closed {
        return Err(ParseError {
            line: text.lines().count(),
            message: "unexpected end of input: missing `}`".into(),
        });
    }

    graph.set_outputs(outputs);
    Ok(graph)
}

fn resolve(
    id_map: &std::collections::HashMap<NodeId, NodeId>,
    id: NodeId,
    cur: &Cursor<'_>,
) -> Result<NodeId, ParseError> {
    match id_map.get(&id) {
        Some(mapped) => Ok(*mapped),
        None => Err(ParseError {
            line: cur.line,
            message: format!("{id} is used before it is defined"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Activation, BinaryOp, ReduceOp};
    use crate::{IgnoreConfig, Op};

    /// `parse(print(g))` must reproduce `g`.
    fn assert_round_trips(graph: &Graph) {
        let text = print(graph);
        let back = match parse(&text) {
            Ok(g) => g,
            Err(e) => panic!("failed to reparse:\n{text}\nerror: {e}"),
        };
        assert!(
            graph.structurally_eq(&back, IgnoreConfig::EXACT),
            "round-trip lost information\n--- printed ---\n{text}\n--- reprinted ---\n{}",
            print(&back)
        );
        assert_eq!(graph.fingerprint(), back.fingerprint());
    }

    #[test]
    fn unit_variants_print_bare() {
        let mut g = Graph::new("m");
        let s = Shape::new(&[4], DType::F32);
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let f = g.add_node(Op::Fma, vec![a, b, b], s);
        g.set_outputs(vec![f]);

        let text = print(&g);
        assert!(text.contains("%2 = Fma(%0, %1, %1)"), "{text}");
        assert_round_trips(&g);
    }

    #[test]
    fn struct_variants_round_trip_through_json() {
        let mut g = Graph::new("reshaped");
        let x = g.input("x", Shape::new(&[4, 16], DType::F32));
        let r = g.add_node(
            Op::Reshape {
                new_shape: vec![64],
            },
            vec![x],
            Shape::new(&[64], DType::F32),
        );
        g.set_outputs(vec![r]);
        assert!(print(&g).contains("Reshape"));
        assert_round_trips(&g);
    }

    #[test]
    fn constant_payloads_survive() {
        // `Op`'s Display renders this as `const(8B)` — the bytes themselves
        // are exactly what a one-way dump throws away.
        let mut g = Graph::new("k");
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let c = g.add_node(
            Op::Constant { data: data.clone() },
            vec![],
            Shape::new(&[2], DType::F32),
        );
        g.set_outputs(vec![c]);

        let back = parse(&print(&g)).unwrap();
        match &back.node(back.outputs[0]).op {
            Op::Constant { data: got } => assert_eq!(got, &data),
            other => panic!("expected a Constant, got {other:?}"),
        }
        assert_round_trips(&g);
    }

    #[test]
    fn every_dtype_round_trips() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U32,
            DType::Bool,
            DType::C64,
            DType::C128,
        ] {
            let mut g = Graph::new("dt");
            let x = g.input("x", Shape::new(&[2, 3], dtype));
            g.set_outputs(vec![x]);
            assert_round_trips(&g);
        }
    }

    #[test]
    fn dynamic_and_scalar_shapes_round_trip() {
        let mut g = Graph::new("dyn");
        let x = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            Shape::from_dims(&[Dim::Dynamic(3), Dim::Static(8)], DType::F32),
        );
        let scalar = g.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0, 1],
                keep_dim: false,
            },
            vec![x],
            Shape::from_dims(&[], DType::F32),
        );
        g.set_outputs(vec![scalar]);

        assert!(print(&g).contains("[?3, 8] f32"), "{}", print(&g));
        assert_round_trips(&g);
    }

    #[test]
    fn node_names_round_trip_including_awkward_ones() {
        let mut g = Graph::new("named");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        let y = g.add_node(
            Op::Activation(Activation::Gelu),
            vec![x],
            Shape::new(&[2], DType::F32),
        );
        // Braces and quotes inside a name must not terminate the value early.
        g.node_mut(y).name = Some(r#"weird } " name"#.to_string());
        g.set_outputs(vec![y]);
        assert_round_trips(&g);
    }

    #[test]
    fn nested_bodies_round_trip() {
        let shape = Shape::new(&[4], DType::F32);
        let mut body = Graph::new("body");
        let c = body.input("carry", shape.clone());
        let y = body.add_node(Op::Activation(Activation::Gelu), vec![c], shape.clone());
        body.set_outputs(vec![y]);

        let mut g = Graph::new("outer");
        let init = g.input("init", shape.clone());
        let s = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 8,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            shape,
        );
        g.set_outputs(vec![s]);
        assert_round_trips(&g);
    }

    #[test]
    fn multiple_outputs_and_no_outputs() {
        let mut g = Graph::new("multi");
        let s = Shape::new(&[2], DType::F32);
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], s);
        g.set_outputs(vec![sum, a]);
        assert_round_trips(&g);

        let mut empty = Graph::new("empty");
        let _ = empty.input("a", Shape::new(&[1], DType::F32));
        assert_round_trips(&empty);
    }

    #[test]
    fn hand_written_text_parses() {
        // The point of the format: a fixture nobody had to build in Rust.
        let g = parse(
            r#"
            // A comment, and blank lines, are fine.
            graph @hand {
              %0 = {"Input":{"name":"x"}} : [2, 2] f32
              %1 = {"Param":{"name":"w"}} : [2, 2] f32
              %2 = MatMul(%0, %1) : [2, 2] f32
              %3 = {"Activation":"Relu"}(%2) : [2, 2] f32 #"act"
              return %3
            }
            "#,
        )
        .unwrap();

        assert_eq!(g.len(), 4);
        assert_eq!(g.name, "hand");
        assert!(matches!(g.node(NodeId(2)).op, Op::MatMul));
        assert!(matches!(
            g.node(NodeId(3)).op,
            Op::Activation(Activation::Relu)
        ));
        assert_eq!(g.node(NodeId(3)).name.as_deref(), Some("act"));
        assert_eq!(g.outputs, vec![NodeId(3)]);
    }

    #[test]
    fn errors_carry_a_line_number() {
        let err = parse("graph @g {\n  %0 = {\"Input\":{\"name\":\"x\"}} : [2] f32\n  %1 = Nonsense(%0) : [2] f32\n}\n")
            .unwrap_err();
        assert_eq!(err.line, 3);
        assert!(err.message.contains("not a valid op"), "{}", err.message);

        let err = parse("graph @g {\n  %1 = MatMul(%7) : [2] f32\n}\n").unwrap_err();
        assert_eq!(err.line, 2);
        assert!(err.message.contains("used before it is defined"));

        let err = parse("graph @g {\n  %0 = {\"Input\":{\"name\":\"x\"}} : [2] weirdtype\n}\n")
            .unwrap_err();
        assert!(err.message.contains("unknown dtype"));

        assert!(parse("").unwrap_err().message.contains("empty input"));
        assert!(
            parse("graph @g {\n  %0 = {\"Input\":{\"name\":\"x\"}} : [2] f32\n")
                .unwrap_err()
                .message
                .contains("missing `}`")
        );
    }

    #[test]
    fn sparse_and_out_of_order_ids_are_accepted() {
        // Ids are labels, not indices — a fixture may number them freely.
        let g = parse(
            r#"graph @sparse {
              %10 = {"Input":{"name":"x"}} : [2] f32
              %20 = {"Activation":"Relu"}(%10) : [2] f32
              return %20
            }"#,
        )
        .unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.outputs, vec![NodeId(1)]);
        assert_eq!(g.node(NodeId(1)).inputs, vec![NodeId(0)]);
    }
}
