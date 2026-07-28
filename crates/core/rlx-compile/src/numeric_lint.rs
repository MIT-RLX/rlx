// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static NaN-source lint — *provable* compile-time NaN/Inf origins.
//!
//! Where [`numeric_check`](rlx_ir::numeric_check) localizes a NaN that
//! actually appeared at runtime, this pass catches the class of NaN sources
//! that don't depend on input data at all: values the compiler can prove are
//! non-finite before anything runs.
//!
//! It reuses the constant-folding evaluator to walk constant-input subgraphs
//! and flags any op whose result is non-finite — a division by a zero
//! constant, `log`/`rsqrt`/`sqrt` of a non-positive constant, a literal
//! NaN/inf `Constant` feeding compute, etc. These are **zero-false-positive**
//! findings: unlike an unguarded `Div` on a runtime tensor (which is only
//! *maybe* a bug and would drown the report in noise), a constant that folds
//! to NaN is definitely wrong. Each finding carries the node's provenance and
//! the same fix hint the runtime localizer uses.
//!
//! Wired behind `RLX_LINT_NUMERICS` in the compile pipeline; also callable
//! directly as [`lint_numerics`].

use crate::const_fold::{evaluate, is_pure};
use rlx_ir::numeric_check::{BadValue, first_bad, fix_hint};
use rlx_ir::provenance::node_label;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

/// A single static numeric finding.
#[derive(Debug, Clone)]
pub struct NumericLint {
    pub node: NodeId,
    pub label: String,
    pub kind: BadValue,
    /// Why it is provably non-finite (e.g. "constant subgraph folds to …").
    pub reason: &'static str,
    /// Remedy hint keyed off the op kind, when one applies.
    pub fix: Option<&'static str>,
}

impl std::fmt::Display for NumericLint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} \"{}\": {}",
            self.node,
            self.kind.as_str(),
            self.label,
            self.reason
        )?;
        if let Some(fix) = self.fix {
            write!(f, "\n  fix: {fix}")?;
        }
        Ok(())
    }
}

/// Decode an `Op::Constant` byte payload as a flat `f32` buffer.
fn decode_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Analyze `graph` for provable compile-time NaN/Inf sources.
///
/// Walks in topological order, materializing constant values as it goes
/// (constants + pure ops over all-constant inputs), and records every node
/// whose value is non-finite. Pure analysis — does not mutate the graph.
pub fn lint_numerics(graph: &Graph) -> Vec<NumericLint> {
    let mut values: HashMap<NodeId, Vec<f32>> = HashMap::new();
    let mut findings = Vec::new();

    for node in graph.nodes() {
        match &node.op {
            // A constant literal that is itself non-finite (e.g. a baked NaN
            // from an earlier fold, or an author-written inf mask fill).
            Op::Constant { data } => {
                let vals = decode_f32(data);
                if let Some(hit) = first_bad(&vals) {
                    findings.push(NumericLint {
                        node: node.id,
                        label: node_label(graph, node.id),
                        kind: hit.kind,
                        reason: "constant literal is non-finite",
                        fix: fix_hint(&node.op),
                    });
                }
                values.insert(node.id, vals);
            }
            // Runtime leaves — nothing provable here.
            Op::Input { .. } | Op::Param { .. } => {}
            // Pure op with all-constant inputs: evaluate and check finiteness.
            op if is_pure(op) && node.inputs.iter().all(|i| values.contains_key(i)) => {
                let inputs: Vec<&Vec<f32>> = node.inputs.iter().map(|i| &values[i]).collect();
                let in_dims: Option<Vec<Vec<usize>>> = node
                    .inputs
                    .iter()
                    .map(|i| crate::const_fold::static_dims(&graph.node(*i).shape))
                    .collect();
                if let Some(in_dims) = in_dims
                    && let Some(result) = evaluate(node, &inputs, &in_dims)
                {
                    if let Some(hit) = first_bad(&result) {
                        findings.push(NumericLint {
                            node: node.id,
                            label: node_label(graph, node.id),
                            kind: hit.kind,
                            reason: "constant subgraph folds to a non-finite value",
                            fix: fix_hint(&node.op),
                        });
                    }
                    values.insert(node.id, result);
                }
            }
            _ => {}
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape, op::BinaryOp};

    fn constant(g: &mut Graph, v: f32) -> NodeId {
        g.add_node(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        )
    }

    #[test]
    fn flags_div_by_zero_constant() {
        let mut g = Graph::new("t");
        let a = constant(&mut g, 1.0);
        let b = constant(&mut g, 0.0);
        let d = g.binary(BinaryOp::Div, a, b, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![d]);

        let lints = lint_numerics(&g);
        assert_eq!(lints.len(), 1, "one finding expected, got {lints:?}");
        assert_eq!(lints[0].node, d);
        assert_eq!(lints[0].kind, BadValue::PosInf);
        assert!(lints[0].fix.is_some());
    }

    #[test]
    fn flags_literal_nan_constant() {
        let mut g = Graph::new("t");
        let n = constant(&mut g, f32::NAN);
        g.set_outputs(vec![n]);
        let lints = lint_numerics(&g);
        assert_eq!(lints.len(), 1);
        assert_eq!(lints[0].kind, BadValue::Nan);
    }

    #[test]
    fn clean_constant_arithmetic_is_silent() {
        // 2.0 / 4.0 = 0.5 — finite, no finding.
        let mut g = Graph::new("t");
        let a = constant(&mut g, 2.0);
        let b = constant(&mut g, 4.0);
        let d = g.binary(BinaryOp::Div, a, b, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![d]);
        assert!(lint_numerics(&g).is_empty());
    }

    #[test]
    fn runtime_div_is_not_flagged() {
        // Division by a runtime input is *not* provable — must stay silent
        // (this is the false-positive we deliberately avoid).
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[1], DType::F32));
        let z = constant(&mut g, 0.0);
        let d = g.binary(BinaryOp::Div, x, z, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![d]);
        assert!(
            lint_numerics(&g).is_empty(),
            "runtime-dependent div must not be flagged"
        );
    }
}
