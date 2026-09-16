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

        // Operand count, driven by the op's own `Arity`. Previously this read
        // `num_inputs()` and skipped anything reporting `0` — which meant every
        // variadic op (`Concat`, `While`) was exempt from arity checking, and
        // the one op with an optional operand needed a hand-written exception
        // right here, beside the table it was contradicting.
        let arity = node.op.arity();
        if !arity.is_unconstrained() && !arity.accepts(node.inputs.len()) {
            errors.push(VerifyError {
                node: Some(node.id),
                message: format!(
                    "{} expects {} inputs, got {}",
                    node.op,
                    arity,
                    node.inputs.len()
                ),
            });
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

/// Report `Op::Param` / `Op::Input` leaves that share a name.
///
/// Binding is by name and reaches a single node — `set_param("w", …)` and
/// `run(&[("x", …)])` each resolve one `NodeId` — so a second leaf sharing a
/// name is never fed. It keeps whatever the arena held (zeros) and every value
/// flowing through it silently vanishes: no shape error, no missing-input
/// error, just a wrong answer. A genuinely shared weight is *one* node with
/// several consumers, so duplicate names only arise from a graph-rewrite bug.
///
/// Deliberately **not** part of [`verify`], which is asserted on every fusion
/// pass in debug builds: graphs in the wild can still trip this, and each needs
/// its own look before it can be made fatal.
///
/// Every instance found so far has been a real bug:
///
/// * `Rewriter::copy_node` emitted hoisted nodes twice — the one this check was
///   written for — which silently zeroed gradient terms in every backward graph
///   that went through `FuseSharedInputMatMul`. **Fixed.**
/// * `jvp` re-declared `tangent_<name>` over a graph that already had one, so
///   `jvp(hvp(f))` returned zero rather than the third derivative. That was read
///   as forward-over-reverse not composing; it was a name collision. **Fixed** —
///   the outer tangent is now `tangent_<name>_2`.
/// * Qwen3.5 prefill graphs declare `last_token_idx` twice: once at flow level
///   as F32, once inside the `GatherLastToken` block as I32. **Open**, in
///   `rlx-models` (`rlx-qwen35/tests/last_token_idx_gather.rs`, `#[ignore]`d
///   with the full diagnosis). Currently harmless only because the unbound node
///   feeds a redundant second gather along an axis the first already collapsed
///   to extent 1, where index 0 is the only legal index — nothing guarantees
///   which of two same-named nodes gets bound, and the other way round every
///   `last_logits_only` prefill would return the *first* token's logits.
pub fn verify_unique_leaf_names(graph: &Graph) -> Vec<VerifyError> {
    let mut errors = Vec::new();
    let mut seen: std::collections::HashMap<&str, NodeId> = std::collections::HashMap::new();
    for node in graph.nodes() {
        let (kind, name) = match &node.op {
            Op::Param { name } => ("Param", name),
            Op::Input { name } => ("Input", name),
            _ => continue,
        };
        match seen.get(name.as_str()) {
            Some(&first) => errors.push(VerifyError {
                node: Some(node.id),
                message: format!(
                    "duplicate {kind} name {name:?}: also declared at {first}. \
                     Binding is by name and reaches only one node, so this one \
                     would silently read zeros"
                ),
            }),
            None => {
                seen.insert(name.as_str(), node.id);
            }
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
        // The single-row KV write is the one op whose operands are checked
        // nowhere downstream: by the time it reaches a kernel it is a byte
        // offset and a length, so a `pos` past the cache or a row of the wrong
        // width stores out of bounds into whatever the memory planner placed
        // next in the arena — a corrupted neighbouring tensor, not a crash.
        // `kv_append_shape` holds the rules; this reports them per node.
        Op::KvAppend { axis, pos } if node.inputs.len() == 2 => {
            let (cache, row) = (node.inputs[0], node.inputs[1]);
            if (cache.0 as usize) < graph.len() && (row.0 as usize) < graph.len() {
                if let Err(message) =
                    crate::shape::kv_append_shape(graph.shape(cache), graph.shape(row), *axis, *pos)
                {
                    err(message);
                }
            }
        }
        // A broadcast that is not a broadcast. `Expand` may only stretch axes
        // that start at 1; `[8,128] -> [8,256]` duplicates nothing, it just
        // declares an output twice the size of the buffer behind it. Nothing
        // downstream rejects that — the CPU backend reads off the end of the
        // input with `index out of bounds: the len is 1024 but the index is
        // 1024`, which names neither the op nor the graph that built it.
        //
        // The rule lives in `expand_shape` (via `broadcast`) and was already
        // being *computed*; it was lost at the `.ok()` in `infer_shape`, which
        // renders "these operands are illegal" indistinguishable from "this op
        // has no inference rule" — and `verify_shapes` skips the latter. This
        // arm asks the helper directly, so there is still one source of truth.
        Op::Expand { target_shape } if node.inputs.len() == 1 => {
            let x = node.inputs[0];
            if (x.0 as usize) < graph.len()
                && let Err(message) = crate::shape::expand_shape(graph.shape(x), target_shape)
            {
                err(format!(
                    "Expand to {target_shape:?} is not a broadcast of {}: {message} \
                     (only axes of extent 1 may be expanded)",
                    graph.shape(x)
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
        let (inferred, rejected) = infer_shape::infer_output_shape_reporting(graph, node);
        // A rule that ran and said no is a finding, not a coverage gap. These
        // messages already existed and were already precise — `matmul_shape`
        // says `K mismatch: 128 vs 256` — they were just discarded at the
        // `.ok()` in `infer_shape`, which left `None` meaning both "no rule for
        // this op" and "these operands are illegal". The loop skipped both.
        if let Some(message) = rejected {
            errors.push(VerifyError {
                node: Some(node.id),
                message: format!("{:?}: {message}", node.op.kind()),
            });
            continue;
        }
        let Some(expected) = inferred else {
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

    /// `Concat` is variadic. It used to report `num_inputs() == 0`, and the
    /// old check skipped anything reporting `0`, so a `Concat` with no
    /// operands verified clean and blew up later in a backend.
    #[test]
    fn a_variadic_op_with_no_operands_is_caught() {
        let shape = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("empty_concat");
        let bogus = g.add_node(Op::Concat { axis: 0 }, vec![], shape);
        g.set_outputs(vec![bogus]);

        let errs = verify(&g);
        assert!(
            errs.iter()
                .any(|e| e.message.contains("expects at least 1 inputs, got 0")),
            "{errs:?}"
        );
    }

    /// A variadic op with operands stays legal — the new check must not turn
    /// "unchecked" into "always rejected".
    #[test]
    fn a_variadic_op_with_operands_is_accepted() {
        let shape = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("concat");
        let a = g.input("a", shape.clone());
        let b = g.input("b", shape.clone());
        let c = g.add_node(
            Op::Concat { axis: 0 },
            vec![a, b],
            Shape::new(&[8], DType::F32),
        );
        g.set_outputs(vec![c]);
        assert!(
            !verify(&g).iter().any(|e| e.message.contains("expects")),
            "{:?}",
            verify(&g)
        );
    }

    /// The optional-operand case, previously a hand-written exception in this
    /// file sitting beside the table it contradicted.
    #[test]
    fn an_optional_operand_accepts_both_counts() {
        let shape = Shape::new(&[4], DType::F32);
        for n_seed in [0usize, 1] {
            let mut g = Graph::new("rng");
            let mut ins = Vec::new();
            if n_seed == 1 {
                ins.push(g.input("seed", shape.clone()));
            }
            let r = g.add_node(
                Op::RngNormal {
                    mean: 0.0,
                    scale: 1.0,
                    key: 0,
                    op_seed: None,
                },
                ins,
                shape.clone(),
            );
            g.set_outputs(vec![r]);
            let errs = verify(&g);
            assert!(
                !errs.iter().any(|e| e.message.contains("expects")),
                "{n_seed} operand(s) should verify: {errs:?}"
            );
        }
    }

    #[test]
    fn an_optional_operand_still_rejects_too_many() {
        let shape = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("rng2");
        let a = g.input("a", shape.clone());
        let b = g.input("b", shape.clone());
        let r = g.add_node(
            Op::RngNormal {
                mean: 0.0,
                scale: 1.0,
                key: 0,
                op_seed: None,
            },
            vec![a, b],
            shape,
        );
        g.set_outputs(vec![r]);
        let errs = verify(&g);
        assert!(
            errs.iter()
                .any(|e| e.message.contains("expects 0 to 1 inputs, got 2")),
            "{errs:?}"
        );
    }

    /// `Op::If` is `[predicate, captures…]`. It was declared `Exact(1)` with
    /// the note "captures handled separately" — but `sccp` builds
    /// `vec![pred, x]`, and both `rlx-unfuse::expand_if` and the MLX lowering
    /// read `inputs[1..]` as the captures, so `verify` rejected every `If` that
    /// captured anything.
    #[test]
    fn an_if_may_carry_branch_captures() {
        let shape = Shape::new(&[4], DType::F32);
        let branch = || {
            let mut b = Graph::new("br");
            let c = b.input("c0", shape.clone());
            b.set_outputs(vec![c]);
            Box::new(b)
        };
        let mut g = Graph::new("if_capture");
        let pred = g.input("pred", Shape::new(&[1], DType::F32));
        let x = g.input("x", shape.clone());
        let n = g.add_node(
            Op::If {
                then_branch: branch(),
                else_branch: branch(),
            },
            vec![pred, x],
            shape,
        );
        g.set_outputs(vec![n]);
        let errs = verify(&g);
        assert!(
            !errs.iter().any(|e| e.message.contains("expects")),
            "an If with one capture must verify: {errs:?}"
        );
    }

    /// …but a predicate is still mandatory.
    #[test]
    fn an_if_still_requires_a_predicate() {
        let shape = Shape::new(&[4], DType::F32);
        let branch = || Box::new(Graph::new("br"));
        let mut g = Graph::new("if_nopred");
        let n = g.add_node(
            Op::If {
                then_branch: branch(),
                else_branch: branch(),
            },
            vec![],
            shape,
        );
        g.set_outputs(vec![n]);
        assert!(
            verify(&g)
                .iter()
                .any(|e| e.message.contains("expects at least 1 inputs, got 0")),
            "{:?}",
            verify(&g)
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

    /// **`KvAppend` is the one op nothing downstream can check.**
    ///
    /// By the time the row write reaches a kernel it is a byte offset and a
    /// length, so a `pos` past the cache stores into whatever the memory
    /// planner put next in the arena — a corrupted neighbouring tensor, not a
    /// crash. The cache being sized to a CAPACITY (with spare rows for `pos` to
    /// index into) rather than to its current contents is also the op's least
    /// obvious requirement, and the one an emitter gets wrong.
    mod kv_append {
        use super::*;

        /// `cache[1, cap, 4]` + `row[1, 1, 4]` written at `pos`.
        fn g(cap: usize, pos: usize, row_dims: &[usize], row_dt: DType) -> Graph {
            let mut g = Graph::new("kv");
            let cache = g.input("cache", Shape::new(&[1, cap, 4], DType::F32));
            let row = g.input("row", Shape::new(row_dims, row_dt));
            let out = g.add_node(
                Op::KvAppend { axis: 1, pos },
                vec![cache, row],
                Shape::new(&[1, pos + 1, 4], DType::F32),
            );
            g.set_outputs(vec![out]);
            g
        }

        fn only_error(g: &Graph) -> String {
            let errors = verify_all(g);
            assert_eq!(
                errors.len(),
                1,
                "expected exactly one error, got {errors:?}"
            );
            errors[0].message.clone()
        }

        #[test]
        fn a_well_formed_append_verifies() {
            assert!(verify_all(&g(8, 7, &[1, 1, 4], DType::F32)).is_empty());
        }

        /// The whole point: `pos` indexes the SPARE rows. `pos == cap` means the
        /// caller sized the cache to the history rather than the capacity.
        #[test]
        fn a_pos_past_the_cache_is_caught() {
            assert!(only_error(&g(8, 8, &[1, 1, 4], DType::F32)).contains("capacity"));
        }

        #[test]
        fn a_row_wider_than_the_cache_is_caught() {
            assert!(only_error(&g(8, 3, &[1, 1, 8], DType::F32)).contains("row dim 2"));
        }

        /// A multi-step "row" would copy one step's worth and drop the rest.
        #[test]
        fn a_multi_row_operand_is_caught() {
            assert!(only_error(&g(8, 3, &[1, 2, 4], DType::F32)).contains("must be 1"));
        }

        /// The write is a raw copy, so an f16 row into an f32 cache reinterprets
        /// bits rather than converting them.
        #[test]
        fn a_dtype_mismatch_is_caught() {
            assert!(only_error(&g(8, 3, &[1, 1, 4], DType::F16)).contains("dtype"));
        }

        #[test]
        fn a_squeezed_row_is_caught() {
            assert!(only_error(&g(8, 3, &[1, 4], DType::F32)).contains("rank"));
        }
    }

    /// `expand-from-non-unit-dim` in the evolution ledger.
    ///
    /// Found by the corpus: `[8,128] -> [8,256]` cleared structural verify,
    /// shape verify and `repr_check`, then panicked in rlx-cpu's
    /// `exec_dispatch` with `index out of bounds: the len is 1024 but the
    /// index is 1024` — a message that names neither `Expand` nor the graph.
    mod expand_is_a_broadcast_or_it_is_nothing {
        use super::*;

        fn g(from: &[usize], to: &[i64]) -> Graph {
            let mut g = Graph::new("expand");
            let x = g.input("x", Shape::new(from, DType::F32));
            let dims: Vec<usize> = to.iter().map(|&d| d as usize).collect();
            let y = g.add_node(
                Op::Expand {
                    target_shape: to.to_vec(),
                },
                vec![x],
                Shape::new(&dims, DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        }

        #[test]
        fn a_non_unit_source_axis_is_rejected() {
            let errors = verify(&g(&[8, 128], &[8, 256]));
            assert_eq!(errors.len(), 1, "{errors:?}");
            let message = errors[0].message.clone();
            assert!(message.contains("not a broadcast"), "{message}");
            // The numbers that make it wrong belong in the message — this is
            // the error that replaces an out-of-bounds panic 3 crates away.
            assert!(
                message.contains("128") && message.contains("256"),
                "{message}"
            );
        }

        /// The legal form still passes, so the check above is about the
        /// illegal case and not about `Expand` being rejected in general.
        #[test]
        fn a_unit_source_axis_is_accepted() {
            assert!(verify_all(&g(&[1, 128], &[8, 128])).is_empty());
        }

        /// Rank-extending broadcast (`[128] -> [8,128]`) is legal too.
        #[test]
        fn a_missing_leading_axis_is_accepted() {
            assert!(verify_all(&g(&[128], &[8, 128])).is_empty());
        }

        /// `verify_all` runs `verify` first and returns early, so the rule
        /// has to be reachable through the entry point everything else calls.
        #[test]
        fn verify_all_reports_it() {
            assert!(!verify_all(&g(&[8, 128], &[8, 256])).is_empty());
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
