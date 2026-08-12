// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph verification — catches IR bugs early.
//!
//! Three layers, cheapest first:
//!
//! 1. **Structural** ([`verify`]) — node references resolve, the DAG property
//!    holds, arity matches the op, outputs exist.
//! 2. **Op-local** ([`verify_op`], folded into [`verify`]) — invariants that
//!    belong to one op and can be stated without looking at the rest of the
//!    graph: a leaf with operands, a `Scan` whose body signature contradicts
//!    its `num_xs`, an `If` whose branches disagree on output count. These
//!    report **at the offending node**, which is the whole point — a
//!    malformed `Scan` otherwise surfaces much later as a shape error inside
//!    the unrolled body, or as a kernel launch failure.
//! 3. **Shape** ([`verify_shapes`]) — re-derive every output shape and diff it
//!    against what was declared.
//!
//! Nested bodies ([`Op::Scan`], [`Op::If`], [`Op::While`], [`Op::CustomFn`],
//! …) are verified recursively, with the containing node's path prefixed to
//! the message. Before this they were skipped entirely: a graph could hold a
//! `Scan` whose body referenced a non-existent node and still verify clean.
//!
//! Custom ops get the same treatment through
//! [`OpExtension::verify`](crate::OpExtension::verify).

use crate::capability::OpCaps;
use crate::graph::{Graph, Node, NodeId};
use crate::infer_shape;
use crate::op::Op;

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
        match &node.op {
            Op::RngNormal { .. } | Op::RngUniform { .. } => {
                if node.inputs.len() > 1 {
                    errors.push(VerifyError {
                        node: Some(node.id),
                        message: format!(
                            "{} accepts 0 or 1 inputs, got {}",
                            node.op,
                            node.inputs.len()
                        ),
                    });
                }
            }
            _ => {
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
        }

        // Op-local invariants, reported at the node that violates them.
        errors.extend(verify_op(graph, node));

        // Nested bodies are IR too. Before this they were never verified at
        // all — a `Scan` whose body referenced a non-existent node verified
        // clean and failed much later, inside the executor.
        for (i, body) in node.op.subgraphs().iter().enumerate() {
            for err in verify_all(body) {
                errors.push(VerifyError {
                    node: Some(node.id),
                    message: format!("in {:?} body #{i}: {err}", node.op.kind()),
                });
            }
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

/// Op-local invariants for one node.
///
/// "Local" means answerable from the node, its operand shapes and its own
/// nested bodies — no whole-graph reasoning. Anything requiring the rest of
/// the graph belongs in [`verify`] (structure) or [`verify_shapes`].
///
/// Called automatically by [`verify`]; exposed for passes that want to check
/// a single node they just built without walking everything.
pub fn verify_op(graph: &Graph, node: &Node) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    let mut err = |message: String| {
        errors.push(VerifyError {
            node: Some(node.id),
            message,
        })
    };

    // Generic, driven by the capability table so a newly-classified op is
    // covered without editing this function.
    if node.op.is_leaf() && !node.inputs.is_empty() {
        err(format!(
            "{} is a graph leaf but has {} operand(s)",
            node.op,
            node.inputs.len()
        ));
    }
    if node.op.kind().has(OpCaps::NESTED_BODY) {
        for (i, body) in node.op.subgraphs().iter().enumerate() {
            if body.outputs.is_empty() {
                err(format!("{} body #{i} declares no outputs", node.op));
            }
        }
    }

    // Ops whose attributes and body signature must agree. These are the
    // configurations that are cheap to state and expensive to debug later.
    match &node.op {
        Op::Scan {
            body,
            num_bcast,
            num_xs,
            length,
            num_checkpoints,
            ..
        } => {
            let expected_outer = 1 + *num_bcast as usize + *num_xs as usize;
            if node.inputs.len() != expected_outer {
                err(format!(
                    "Scan takes 1 carry + {num_bcast} broadcast + {num_xs} per-step inputs \
                     = {expected_outer}, got {}",
                    node.inputs.len()
                ));
            }
            let body_inputs = body
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Input { .. }))
                .count();
            if body_inputs != expected_outer {
                err(format!(
                    "Scan body must declare {expected_outer} Op::Inputs \
                     (carry + {num_bcast} broadcast + {num_xs} per-step), got {body_inputs}"
                ));
            }
            if body.outputs.len() != 1 {
                err(format!(
                    "Scan body must produce exactly one output (the next carry), got {}",
                    body.outputs.len()
                ));
            }
            if *num_checkpoints > *length {
                err(format!(
                    "Scan num_checkpoints ({num_checkpoints}) exceeds length ({length})"
                ));
            }
        }
        Op::If {
            then_branch,
            else_branch,
        } => {
            if then_branch.outputs.len() != else_branch.outputs.len() {
                err(format!(
                    "If branches disagree on output count: then={} else={}",
                    then_branch.outputs.len(),
                    else_branch.outputs.len()
                ));
            }
            if node.inputs.is_empty() {
                err("If takes a predicate as its first operand, got none".to_string());
            }
        }
        Op::While { cond, body, .. } => {
            if cond.outputs.len() != 1 {
                err(format!(
                    "While cond must produce exactly one Bool output, got {}",
                    cond.outputs.len()
                ));
            }
            if body.outputs.len() != node.inputs.len() {
                err(format!(
                    "While body must produce one value per loop-carried input \
                     ({} expected), got {}",
                    node.inputs.len(),
                    body.outputs.len()
                ));
            }
        }
        Op::CustomFn {
            fwd_body,
            num_inputs,
            ..
        } => {
            if node.inputs.len() != *num_inputs as usize {
                err(format!(
                    "CustomFn declares num_inputs={num_inputs} but has {} operand(s)",
                    node.inputs.len()
                ));
            }
            if fwd_body.outputs.len() != 1 {
                err(format!(
                    "CustomFn fwd_body must produce exactly one output, got {}",
                    fwd_body.outputs.len()
                ));
            }
        }
        // ── Invariants aimed at this project's actual failure history ──
        //
        // Each of these encodes a bug class that has cost real debugging time,
        // and each is checkable from the node alone. They are deliberately
        // conservative: a check is skipped rather than guessed at when an
        // operand shape is dynamic or the operand count is already wrong (the
        // arity check above reports that).
        Op::Rope {
            head_dim, n_rot, ..
        } => {
            // Partial rotary: only the first `n_rot` of each `head_dim` lane
            // rotates. `n_rot > head_dim` is nonsense that surfaces far away as
            // out-of-bounds table reads. Both must be even — the rotation pairs
            // elements.
            if *head_dim == 0 {
                err("Rope head_dim must be non-zero".to_string());
            } else {
                if n_rot > head_dim {
                    err(format!(
                        "Rope n_rot ({n_rot}) exceeds head_dim ({head_dim}): \
                         only the first n_rot lanes of each head rotate"
                    ));
                }
                if head_dim % 2 != 0 {
                    err(format!("Rope head_dim ({head_dim}) must be even"));
                }
                if n_rot % 2 != 0 {
                    err(format!("Rope n_rot ({n_rot}) must be even"));
                }
                if let Some(&x) = node.inputs.first()
                    && let Some(last) = static_last_dim(graph, x)
                    && last % head_dim != 0
                {
                    err(format!(
                        "Rope input last dim ({last}) is not a multiple of head_dim ({head_dim})"
                    ));
                }
            }
        }
        // `dlogits[n,c] = (softmax(logits[n])[c] - onehot[n,c]) * d_loss[n]`.
        // The per-row `d_loss` is rank-1; passing it as anything else is how a
        // scalar got broadcast across the class axis and produced silently
        // wrong gradients on GPU backends.
        Op::SoftmaxCrossEntropyBackward if node.inputs.len() == 3 => {
            let (logits, labels, d_loss) = (node.inputs[0], node.inputs[1], node.inputs[2]);
            if let (Some(lr), Some(labr), Some(dr)) = (
                static_rank(graph, logits),
                static_rank(graph, labels),
                static_rank(graph, d_loss),
            ) {
                if lr != 2 {
                    err(format!(
                        "SoftmaxCrossEntropyBackward logits must be [N, C], got rank {lr}"
                    ));
                }
                if labr != 1 {
                    err(format!(
                        "SoftmaxCrossEntropyBackward labels must be [N], got rank {labr}"
                    ));
                }
                if dr != 1 {
                    err(format!(
                        "SoftmaxCrossEntropyBackward d_loss must be per-row [N], got rank {dr} — \
                         a rank-0 or [N,1] operand broadcasts across the class axis"
                    ));
                }
            }
        }
        Op::SoftmaxCrossEntropyWithLogits if node.inputs.len() == 2 => {
            if let (Some(lr), Some(labr)) = (
                static_rank(graph, node.inputs[0]),
                static_rank(graph, node.inputs[1]),
            ) {
                if lr != 2 {
                    err(format!(
                        "SoftmaxCrossEntropyWithLogits logits must be [N, C], got rank {lr}"
                    ));
                }
                if labr != 1 {
                    err(format!(
                        "SoftmaxCrossEntropyWithLogits labels must be [N], got rank {labr}"
                    ));
                }
            }
        }
        // Affine operands are per-feature vectors sized to the normalised axis.
        // A gamma/beta of the wrong width reads past its buffer in every
        // hand-written norm kernel.
        Op::LayerNorm { .. } | Op::RmsNorm { .. } if node.inputs.len() == 3 => {
            let x = node.inputs[0];
            if let Some(feat) = static_last_dim(graph, x) {
                for (label, operand) in [("gamma", node.inputs[1]), ("beta", node.inputs[2])] {
                    let Some(rank) = static_rank(graph, operand) else {
                        continue;
                    };
                    if rank != 1 {
                        // A higher-rank affine param whose extra dims are size-1
                        // (e.g. `[1,1,C]`, as some whisper/TTS graphs build it) has
                        // the same flat `[C]` buffer the norm kernels read — accept
                        // it. Only flag a param whose element count doesn't collapse
                        // to the normalised width.
                        match static_num_elements(graph, operand) {
                            Some(n) if n == feat => {}
                            Some(n) => err(format!(
                                "{} {label} must be rank-1 [C] (or collapse to it): got rank {rank} \
                                 with {n} elements, expected {feat}",
                                node.op
                            )),
                            None => {}
                        }
                        continue;
                    }
                    if let Some(width) = static_last_dim(graph, operand)
                        && width != feat
                    {
                        err(format!(
                            "{} {label} width ({width}) does not match the normalised axis ({feat})",
                            node.op
                        ));
                    }
                }
            }
        }
        // Q's feature width must match one of the two layouts the attention
        // kernels accept: packed `[.., num_heads * head_dim]` (BSD) or
        // per-head `[B, S, H, head_dim]` / `[B, H, S, head_dim]` (BSHD/BHSD).
        // Anything else means `num_heads`/`head_dim` disagree with the tensor
        // actually being fed, which surfaces as a garbled attention output
        // rather than a crash.
        Op::Attention {
            num_heads,
            head_dim,
            ..
        } if !node.inputs.is_empty() => {
            if *num_heads == 0 || *head_dim == 0 {
                err("Attention num_heads and head_dim must be non-zero".to_string());
            } else if let Some(last) = static_last_dim(graph, node.inputs[0])
                && last != num_heads * head_dim
                && last != *head_dim
            {
                err(format!(
                    "Attention Q last dim ({last}) matches neither the packed layout \
                     (num_heads * head_dim = {}) nor the per-head layout (head_dim = {head_dim})",
                    num_heads * head_dim
                ));
            }
        }
        // Registered custom ops state their own invariants.
        Op::Custom { name, .. } => {
            if let Some(ext) = crate::lookup_op(name) {
                // Only ask once every operand resolves — otherwise the
                // extension would index into a short slice. The dangling
                // reference itself is already reported by `verify`.
                if node
                    .inputs
                    .iter()
                    .all(|i| (i.0 as usize) < graph.len() && i.0 < node.id.0)
                {
                    let shapes: Vec<&crate::Shape> =
                        node.inputs.iter().map(|&i| graph.shape(i)).collect();
                    for message in ext.verify(node, &shapes) {
                        err(format!("{name}: {message}"));
                    }
                }
            }
        }
        _ => {}
    }

    errors
}

/// Rank of `id`'s shape, or `None` when the operand does not resolve.
///
/// Every check below is skipped rather than guessed at on an unresolved
/// operand — the dangling reference itself is reported by [`verify`], and a
/// second error derived from it would be noise.
fn static_rank(graph: &Graph, id: NodeId) -> Option<usize> {
    ((id.0 as usize) < graph.len()).then(|| graph.shape(id).rank())
}

/// Total element count of `id`'s shape when every dimension is static
/// (`None` if any dim is dynamic). Used to accept affine norm params whose
/// extra dims are size-1 (e.g. `[1,1,C]` collapses to `C` elements).
fn static_num_elements(graph: &Graph, id: NodeId) -> Option<usize> {
    if (id.0 as usize) >= graph.len() {
        return None;
    }
    let shape = graph.shape(id);
    let mut prod = 1usize;
    for i in 0..shape.rank() {
        match shape.dim(i) {
            crate::shape::Dim::Static(n) => prod *= n,
            crate::shape::Dim::Dynamic(_) => return None,
        }
    }
    Some(prod)
}

/// Last dimension of `id`'s shape when it is statically known.
fn static_last_dim(graph: &Graph, id: NodeId) -> Option<usize> {
    if (id.0 as usize) >= graph.len() {
        return None;
    }
    let shape = graph.shape(id);
    let rank = shape.rank();
    if rank == 0 {
        return None;
    }
    match shape.dim(rank - 1) {
        crate::shape::Dim::Static(n) => Some(n),
        crate::shape::Dim::Dynamic(_) => None,
    }
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
///
/// Shape checks run **only** once the graph is structurally sound.
/// [`verify_shapes`] resolves operands by direct index, so on a graph with a
/// dangling or forward reference it panics with `index out of bounds` instead
/// of reporting — turning the one tool meant to explain a broken graph into
/// another thing to debug. The structural errors are the actionable ones
/// anyway; shapes derived from missing operands would be noise.
pub fn verify_all(graph: &Graph) -> Vec<VerifyError> {
    let errors = verify(graph);
    if !errors.is_empty() {
        return errors;
    }
    verify_shapes(graph)
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

    /// `Scan` with a body of `num_xs` per-step inputs and one carry.
    fn scan_graph(num_xs: u32, body_inputs: u32, outer_inputs: usize) -> Graph {
        let shape = Shape::new(&[4], DType::F32);
        let mut body = Graph::new("body");
        let mut last = body.input("carry", shape.clone());
        for i in 1..body_inputs {
            last = body.input(format!("x{i}"), shape.clone());
        }
        body.set_outputs(vec![last]);

        let mut g = Graph::new("outer");
        let inputs: Vec<_> = (0..outer_inputs)
            .map(|i| g.input(format!("in{i}"), shape.clone()))
            .collect();
        let s = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 8,
                save_trajectory: false,
                num_bcast: 0,
                num_xs,
                num_checkpoints: 0,
            },
            inputs,
            shape,
        );
        g.set_outputs(vec![s]);
        g
    }

    #[test]
    fn well_formed_scan_verifies() {
        assert!(verify(&scan_graph(1, 2, 2)).is_empty());
    }

    #[test]
    fn scan_body_signature_mismatch_is_caught_at_the_scan_node() {
        // Declares one per-step input, but the body only takes the carry.
        let errs = verify(&scan_graph(1, 1, 2));
        assert_eq!(errs.len(), 1, "{errs:?}");
        assert!(
            errs[0]
                .message
                .contains("Scan body must declare 2 Op::Inputs")
        );
        assert!(errs[0].node.is_some(), "must localize to the Scan node");
    }

    #[test]
    fn scan_outer_arity_mismatch_is_caught() {
        let errs = verify(&scan_graph(2, 3, 2));
        assert!(
            errs.iter()
                .any(|e| e.message.contains("Scan takes 1 carry + 0 broadcast + 2")),
            "{errs:?}"
        );
    }

    #[test]
    fn malformed_nested_body_is_reported_through_its_parent() {
        // A body whose node references a non-existent input: previously this
        // was never verified at all.
        let shape = Shape::new(&[4], DType::F32);
        let mut body = Graph::new("body");
        let c = body.input("carry", shape.clone());
        let y = body.add_node(
            Op::Activation(crate::op::Activation::Gelu),
            vec![c],
            shape.clone(),
        );
        body.set_inputs(y, vec![NodeId(99)]);
        body.set_outputs(vec![y]);

        let mut g = Graph::new("outer");
        let init = g.input("init", shape.clone());
        let s = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 4,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            shape,
        );
        g.set_outputs(vec![s]);

        let errs = verify(&g);
        assert!(
            errs.iter()
                .any(|e| e.message.contains("body #0") && e.message.contains("non-existent")),
            "nested body errors must surface: {errs:?}"
        );
    }

    #[test]
    fn a_leaf_with_operands_is_caught() {
        let shape = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("bad_leaf");
        let x = g.input("x", shape.clone());
        // An `Input` that consumes another node is nonsense; the old arity
        // check skipped it because `num_inputs() == 0`.
        let bogus = g.add_node(Op::Input { name: "y".into() }, vec![x], shape);
        g.set_outputs(vec![bogus]);

        let errs = verify(&g);
        assert!(
            errs.iter()
                .any(|e| e.message.contains("graph leaf but has 1 operand")),
            "{errs:?}"
        );
    }

    #[test]
    fn if_branches_must_agree_on_output_count() {
        let shape = Shape::new(&[4], DType::F32);
        let mut then_b = Graph::new("then");
        let t = then_b.input("t", shape.clone());
        then_b.set_outputs(vec![t]);

        let mut else_b = Graph::new("else");
        let e = else_b.input("e", shape.clone());
        else_b.set_outputs(vec![e, e]);

        let mut g = Graph::new("cond");
        let p = g.input("pred", Shape::new(&[1], DType::Bool));
        let node = g.add_node(
            Op::If {
                then_branch: Box::new(then_b),
                else_branch: Box::new(else_b),
            },
            vec![p],
            shape,
        );
        g.set_outputs(vec![node]);

        let errs = verify(&g);
        assert!(
            errs.iter()
                .any(|e| e.message.contains("If branches disagree on output count")),
            "{errs:?}"
        );
    }

    /// Each of these encodes a bug class that has actually cost debugging time
    /// in this project; the point is that the verifier now reports at the
    /// offending node instead of the symptom surfacing on one backend later.
    mod real_bug_classes {
        use super::*;

        fn f32s(dims: &[usize]) -> Shape {
            Shape::new(dims, DType::F32)
        }

        #[test]
        fn rope_n_rot_may_not_exceed_head_dim() {
            let mut g = Graph::new("rope");
            let x = g.input("x", f32s(&[2, 8, 64]));
            let cos = g.input("cos", f32s(&[8, 32]));
            let sin = g.input("sin", f32s(&[8, 32]));
            // Partial rotary with n_rot > head_dim is nonsense.
            let r = g.add_node(
                Op::Rope {
                    head_dim: 64,
                    n_rot: 128,
                    style: crate::op::RopeStyle::NeoX,
                },
                vec![x, cos, sin],
                f32s(&[2, 8, 64]),
            );
            g.set_outputs(vec![r]);

            let errs = verify(&g);
            assert!(
                errs.iter()
                    .any(|e| e.message.contains("n_rot (128) exceeds head_dim (64)")),
                "{errs:?}"
            );
        }

        #[test]
        fn rope_input_width_must_divide_by_head_dim() {
            let mut g = Graph::new("rope");
            let x = g.input("x", f32s(&[2, 8, 100]));
            let cos = g.input("cos", f32s(&[8, 32]));
            let sin = g.input("sin", f32s(&[8, 32]));
            let r = g.add_node(
                Op::Rope {
                    head_dim: 64,
                    n_rot: 64,
                    style: crate::op::RopeStyle::NeoX,
                },
                vec![x, cos, sin],
                f32s(&[2, 8, 100]),
            );
            g.set_outputs(vec![r]);
            assert!(
                verify(&g)
                    .iter()
                    .any(|e| e.message.contains("not a multiple of head_dim")),
                "a head-count mismatch must be caught at the Rope node"
            );
        }

        #[test]
        fn sce_backward_rejects_a_broadcastable_d_loss() {
            // The GPU bug: `d_loss` arriving as something other than per-row
            // `[N]` broadcasts across the class axis and silently produces
            // wrong gradients.
            let mut g = Graph::new("sce");
            let logits = g.input("logits", f32s(&[8, 10]));
            let labels = g.input("labels", f32s(&[8]));
            let d_loss = g.input("d_loss", f32s(&[8, 1]));
            let b = g.add_node(
                Op::SoftmaxCrossEntropyBackward,
                vec![logits, labels, d_loss],
                f32s(&[8, 10]),
            );
            g.set_outputs(vec![b]);

            let errs = verify(&g);
            assert!(
                errs.iter()
                    .any(|e| e.message.contains("d_loss must be per-row")),
                "{errs:?}"
            );
        }

        #[test]
        fn well_formed_sce_backward_verifies() {
            let mut g = Graph::new("sce_ok");
            let logits = g.input("logits", f32s(&[8, 10]));
            let labels = g.input("labels", f32s(&[8]));
            let d_loss = g.input("d_loss", f32s(&[8]));
            let b = g.add_node(
                Op::SoftmaxCrossEntropyBackward,
                vec![logits, labels, d_loss],
                f32s(&[8, 10]),
            );
            g.set_outputs(vec![b]);
            assert!(verify(&g).is_empty());
        }

        #[test]
        fn norm_affine_operands_must_match_the_normalised_axis() {
            let mut g = Graph::new("ln");
            let x = g.input("x", f32s(&[2, 8, 64]));
            let gamma = g.param("gamma", f32s(&[64]));
            let beta = g.param("beta", f32s(&[32])); // wrong width
            let n = g.add_node(
                Op::LayerNorm { eps: 1e-5, axis: 2 },
                vec![x, gamma, beta],
                f32s(&[2, 8, 64]),
            );
            g.set_outputs(vec![n]);

            let errs = verify(&g);
            assert!(
                errs.iter().any(|e| e.message.contains("beta width (32)")),
                "{errs:?}"
            );
        }

        #[test]
        fn norm_affine_accepts_collapsible_higher_rank() {
            // `[1,1,C]` gamma/beta (some whisper/TTS graphs) is the same flat [C]
            // buffer the norm kernels read — must NOT be rejected.
            let mut g = Graph::new("ln3");
            let x = g.input("x", f32s(&[2, 8, 64]));
            let gamma = g.param("gamma", f32s(&[1, 1, 64]));
            let beta = g.param("beta", f32s(&[1, 1, 64]));
            let n = g.add_node(
                Op::LayerNorm { eps: 1e-5, axis: 2 },
                vec![x, gamma, beta],
                f32s(&[2, 8, 64]),
            );
            g.set_outputs(vec![n]);
            let errs = verify(&g);
            assert!(
                !errs
                    .iter()
                    .any(|e| e.message.contains("gamma") || e.message.contains("beta")),
                "[1,1,C] affine params should be accepted: {errs:?}"
            );
        }

        #[test]
        fn norm_affine_rejects_higher_rank_wrong_count() {
            // A genuinely multi-dim param (1024 != 64 elements) must still be flagged.
            let mut g = Graph::new("ln3b");
            let x = g.input("x", f32s(&[2, 8, 64]));
            let gamma = g.param("gamma", f32s(&[2, 8, 64]));
            let beta = g.param("beta", f32s(&[64]));
            let n = g.add_node(
                Op::LayerNorm { eps: 1e-5, axis: 2 },
                vec![x, gamma, beta],
                f32s(&[2, 8, 64]),
            );
            g.set_outputs(vec![n]);
            let errs = verify(&g);
            assert!(
                errs.iter().any(|e| e.message.contains("collapse")),
                "wrong-element-count rank-3 gamma should be rejected: {errs:?}"
            );
        }

        #[test]
        fn attention_accepts_both_packed_and_per_head_layouts() {
            let mk = |last: usize| {
                let mut g = Graph::new("attn");
                let q = g.input("q", f32s(&[1, 4, 8, last]));
                let k = g.input("k", f32s(&[1, 4, 8, last]));
                let v = g.input("v", f32s(&[1, 4, 8, last]));
                let a = g.add_node(
                    Op::Attention {
                        num_heads: 8,
                        head_dim: 25,
                        v_head_dim: None,
                        mask_kind: crate::op::MaskKind::None,
                        score_scale: None,
                        attn_logit_softcap: None,
                    },
                    vec![q, k, v],
                    f32s(&[1, 4, 8, last]),
                );
                g.set_outputs(vec![a]);
                g
            };
            // Per-head (BSHD) and packed (BSD) widths are both legitimate.
            assert!(verify(&mk(25)).is_empty(), "per-head layout rejected");
            assert!(verify(&mk(200)).is_empty(), "packed layout rejected");
            // Neither: the declared heads disagree with the tensor.
            assert!(
                verify(&mk(64))
                    .iter()
                    .any(|e| e.message.contains("matches neither")),
                "a genuine head/dim mismatch must be caught"
            );
        }
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
