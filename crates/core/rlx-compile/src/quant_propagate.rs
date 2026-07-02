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

//! `FakeQuantize` propagation — eliminate redundant fake-quants
//! across pure data-movement chains.
//!
//! Pattern this pass collapses:
//!
//! ```text
//!   FakeQuantize{bits=B,axis=A,ste=S}  ──▶  Reshape  ──▶  FakeQuantize{...}
//! ```
//!
//! When the inner ops preserve magnitudes — `Reshape`, `Transpose`,
//! `Narrow`, `Concat`, `Pool(Max)`, `Activation(Relu)` — and the
//! second `FakeQuantize` has the same `(bits, axis, ste)` as the
//! first, then the second op is provably idempotent and can be
//! deleted (replaced with an identity reshape into its own shape).
//!
//! This matters because `--qat` insertions tend to be liberal: the
//! trainer doesn't know which post-conv ops will be folded by the
//! optimizer, so it sprinkles fake-quants on activations after every
//! pool / relu. With this pass, the optimizer cleans up the
//! redundancy itself.
//!
//! ## What "magnitude-preserving" means
//!
//! For per-tensor fake-quant (`axis = None`), an op preserves
//! magnitudes iff `max(|out|) <= max(|in|)`. That holds for:
//!   - `Reshape` / `Transpose` / `Narrow` / `Concat` (pure layout)
//!   - `Pool(Max)` (output is a subset of input values)
//!   - `Activation(Relu)` (output ⊆ {0} ∪ positive inputs)
//!
//! It does **not** hold for `Activation(Exp)`, `Binary(Mul)`, etc.
//!
//! For per-channel fake-quant (`axis = Some(c)`) the same ops
//! preserve magnitudes per-channel **as long as the channel axis
//! itself isn't permuted or reshaped** — we conservatively bail on
//! any per-channel chain that crosses a `Reshape` or `Transpose`
//! today; the safe set there is just `Pool(Max)` / `Activation(Relu)`.

use rlx_ir::op::Activation;
use rlx_ir::{Graph, Op};
use std::collections::HashMap;

/// Run the propagation pass in place. Returns the number of
/// `FakeQuantize` nodes eliminated.
pub fn run(graph: &mut Graph) -> usize {
    // First pass: walk every `FakeQuantize` node and check whether
    // its only input chain (through magnitude-preserving ops) ends
    // at another `FakeQuantize` with the same parameters. If so,
    // record that this node should redirect to the inner node's
    // input.
    //
    // This is the simplest version: we only collapse exact same-param
    // pairs (no scale-merging when params differ). A future revision
    // could collapse `bits=4 → ... → bits=8` (the higher-bit op
    // dominates so the lower-bit one is the limiting factor and the
    // higher-bit one is redundant).
    let mut redirect: HashMap<rlx_ir::NodeId, rlx_ir::NodeId> = HashMap::new();

    for node in graph.nodes() {
        let Op::FakeQuantize {
            bits, axis, ste, ..
        } = &node.op
        else {
            continue;
        };
        let inner_input = node.inputs[0];
        // Walk backwards through magnitude-preserving ops until we
        // hit either (a) another FakeQuantize, or (b) something
        // not in the safe set.
        let mut cur = inner_input;
        loop {
            let parent = graph.node(cur);
            match &parent.op {
                Op::FakeQuantize {
                    bits: b2,
                    axis: a2,
                    ste: s2,
                    ..
                } if b2 == bits && a2 == axis && s2 == ste => {
                    // Found a matching outer fake-quant. The current
                    // node is redundant — every consumer should read
                    // the inner FakeQuantize's output instead.
                    redirect.insert(node.id, parent.id);
                    break;
                }
                op if is_magnitude_preserving(op, *axis) => {
                    if parent.inputs.len() != 1 {
                        break;
                    }
                    cur = parent.inputs[0];
                }
                _ => break,
            }
        }
    }

    if redirect.is_empty() {
        return 0;
    }

    // Second pass: rewire every consumer of a redirected node to
    // point at the redirect target. We don't physically remove the
    // node — DCE will sweep it on the next pass.
    let n_eliminated = redirect.len();
    let node_ids: Vec<_> = graph.nodes().iter().map(|n| n.id).collect();
    for id in node_ids {
        let inputs = graph.node(id).inputs.clone();
        let mut new_inputs = inputs.clone();
        let mut changed = false;
        for (i, &input) in inputs.iter().enumerate() {
            if let Some(&target) = redirect.get(&input) {
                new_inputs[i] = target;
                changed = true;
            }
        }
        if changed {
            graph.set_inputs(id, new_inputs);
        }
    }

    // Also rewire output references.
    let outs: Vec<_> = graph
        .outputs
        .iter()
        .map(|&o| redirect.get(&o).copied().unwrap_or(o))
        .collect();
    if outs != graph.outputs {
        graph.set_outputs(outs);
    }

    n_eliminated
}

fn is_magnitude_preserving(op: &Op, axis: Option<usize>) -> bool {
    match op {
        // Pure-layout ops always preserve magnitude per-element.
        Op::Reshape { .. } | Op::Transpose { .. } | Op::Narrow { .. } | Op::Concat { .. } => {
            // For per-channel fake-quant the channel axis can shift
            // through these ops, so we only allow them in per-tensor
            // mode. Conservative; could be tightened later.
            axis.is_none()
        }
        // Max-pool emits a value that was already in the input.
        Op::Pool {
            kind: rlx_ir::op::ReduceOp::Max,
            ..
        } => true,
        // ReLU clamps below to zero — never increases magnitude.
        Op::Activation(Activation::Relu) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::*;
    use rlx_ir::*;

    #[test]
    fn collapses_redundant_fake_quant_through_relu() {
        let f = DType::F32;
        let mut g = Graph::new("collapse");
        let x = g.input("x", Shape::new(&[4], f));
        let q1 = g.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![x],
            Shape::new(&[4], f),
        );
        let r = g.activation(Activation::Relu, q1, Shape::new(&[4], f));
        let q2 = g.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![r],
            Shape::new(&[4], f),
        );
        g.set_outputs(vec![q2]);

        let n = run(&mut g);
        assert_eq!(n, 1, "should have eliminated the second fake-quant");
        // Output now points at the redirect target.
        assert_eq!(g.outputs, vec![q1]);
    }

    #[test]
    fn keeps_fake_quant_with_different_bits() {
        let f = DType::F32;
        let mut g = Graph::new("keep");
        let x = g.input("x", Shape::new(&[4], f));
        let q1 = g.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![x],
            Shape::new(&[4], f),
        );
        let r = g.activation(Activation::Relu, q1, Shape::new(&[4], f));
        let q2 = g.add_node(
            Op::FakeQuantize {
                bits: 4,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![r],
            Shape::new(&[4], f),
        );
        g.set_outputs(vec![q2]);

        let n = run(&mut g);
        assert_eq!(n, 0, "different bits → don't collapse");
    }

    #[test]
    fn keeps_fake_quant_when_intermediate_isnt_safe() {
        let f = DType::F32;
        let mut g = Graph::new("unsafe_chain");
        let x = g.input("x", Shape::new(&[4], f));
        let q1 = g.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![x],
            Shape::new(&[4], f),
        );
        // Exp can grow magnitude — fake-quant after it is meaningful.
        let e = g.activation(Activation::Exp, q1, Shape::new(&[4], f));
        let q2 = g.add_node(
            Op::FakeQuantize {
                bits: 8,
                axis: None,
                ste: SteKind::default(),
                scale_mode: ScaleMode::default(),
            },
            vec![e],
            Shape::new(&[4], f),
        );
        g.set_outputs(vec![q2]);

        let n = run(&mut g);
        assert_eq!(n, 0, "Exp can grow magnitude; don't collapse");
    }
}
