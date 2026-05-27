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

//! Graph verification — catches IR bugs early.
//!
//! Verifies structural invariants: valid node references, input counts,
//! acyclicity, output validity, and (optionally) shape consistency.

use crate::graph::{Graph, NodeId};
use crate::infer_shape;

/// Error found during graph verification.
#[derive(Debug)]
pub struct VerifyError {
    pub node: Option<NodeId>,
    pub message: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node {
            Some(id) => write!(f, "at {id}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

/// Verify structural integrity of a graph. Returns all errors found.
pub fn verify(graph: &Graph) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    let num_nodes = graph.len();

    for node in graph.nodes() {
        // Check that all input references are valid and precede this node (DAG property).
        for &input in &node.inputs {
            if input.0 as usize >= num_nodes {
                errors.push(VerifyError {
                    node: Some(node.id),
                    message: format!(
                        "input {input} references non-existent node (graph has {num_nodes} nodes)"
                    ),
                });
            } else if input.0 >= node.id.0 {
                errors.push(VerifyError {
                    node: Some(node.id),
                    message: format!(
                        "input {input} is not before {}: graph is not a DAG",
                        node.id
                    ),
                });
            }
        }

        // Check input count matches op expectation (except variadic ops like Concat).
        let expected = node.op.num_inputs();
        if expected > 0 && node.inputs.len() != expected {
            errors.push(VerifyError {
                node: Some(node.id),
                message: format!(
                    "{} expects {} inputs, got {}",
                    node.op,
                    expected,
                    node.inputs.len()
                ),
            });
        }
    }

    // Check outputs reference valid nodes.
    for &out in &graph.outputs {
        if out.0 as usize >= num_nodes {
            errors.push(VerifyError {
                node: None,
                message: format!("output {out} references non-existent node"),
            });
        }
    }

    errors
}

/// True when `declared` and `inferred` describe the same logical tensor.
fn shapes_compatible(declared: &crate::Shape, inferred: &crate::Shape) -> bool {
    if declared == inferred {
        return true;
    }
    if declared.dtype() != inferred.dtype() {
        return false;
    }
    // Scalar conventions: rank-0 `[]` and rank-1 `[1]` both mean one element.
    matches!(
        (declared.num_elements(), inferred.num_elements()),
        (Some(1), Some(1))
    )
}

/// Re-derive output shapes from inputs and diff against declared shapes.
pub fn verify_shapes(graph: &Graph) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    for node in graph.nodes() {
        let Some(expected) = infer_shape::infer_output_shape(graph, node) else {
            continue;
        };
        if !shapes_compatible(&node.shape, &expected) {
            errors.push(VerifyError {
                node: Some(node.id),
                message: format!(
                    "shape mismatch: declared {}, inferred {expected}",
                    node.shape
                ),
            });
        }
    }
    errors
}

/// Structural + shape verification.
pub fn verify_all(graph: &Graph) -> Vec<VerifyError> {
    let mut errors = verify(graph);
    errors.extend(verify_shapes(graph));
    errors
}

/// Panic when verification fails. **Debug builds only** — in release
/// this macro expands to nothing and is not compiled.
#[macro_export]
macro_rules! debug_assert_valid {
    ($graph:expr, $stage:expr) => {{
        #[cfg(debug_assertions)]
        {
            let __errors = $crate::verify::verify_all($graph);
            if !__errors.is_empty() {
                let __msg = __errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("\n  ");
                panic!("IR verifier failed at `{}`:\n  {}", $stage, __msg);
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn shape_mismatch_is_caught() {
        let mut g = Graph::new("bad");
        let x = g.input("x", Shape::new(&[4, 8], DType::F32));
        let w = g.param("w", Shape::new(&[8, 16], DType::F32));
        // Wrong output shape on purpose.
        let mm = g.matmul(x, w, Shape::new(&[99, 99], DType::F32));
        g.set_outputs(vec![mm]);

        let errs = verify_shapes(&g);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("shape mismatch"));
    }

    #[test]
    fn scalar_rank0_and_rank1_are_compatible() {
        let mut g = Graph::new("scalar");
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let loss = g.add_node(
            Op::Reduce {
                op: crate::op::ReduceOp::Sum,
                axes: vec![0],
                keep_dim: false,
            },
            vec![x],
            Shape::new(&[1], DType::F32),
        );
        g.set_outputs(vec![loss]);
        assert!(
            verify_shapes(&g).is_empty(),
            "[] inferred vs [1] declared should match for a scalar"
        );
    }

    #[test]
    fn verify_all_combines_checks() {
        let mut g = Graph::new("ok");
        let x = g.input("x", Shape::new(&[4, 384], DType::F32));
        let w = g.param("w", Shape::new(&[384, 384], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[4, 384], DType::F32));
        g.set_outputs(vec![mm]);
        assert!(verify_all(&g).is_empty());
    }
}
