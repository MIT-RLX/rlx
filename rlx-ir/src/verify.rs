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
//! acyclicity, and output validity.

use crate::graph::{Graph, NodeId};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn valid_graph_verifies() {
        let mut g = Graph::new("ok");
        let x = g.input("x", Shape::new(&[4, 384], DType::F32));
        let w = g.param("w", Shape::new(&[384, 384], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[4, 384], DType::F32));
        g.set_outputs(vec![mm]);

        let errs = verify(&g);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    }
}
