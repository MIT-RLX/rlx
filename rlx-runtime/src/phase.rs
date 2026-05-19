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

//! Phase-aware streaming inference (plan #16).
//!
//! Borrowed from MAX's `pipeline/phase_derivation.mojo` pattern:
//! a streaming forward pass has three phases — prologue (one-time
//! setup, e.g. KV cache init / embedding layer for prefill),
//! steady-state (per-token decode loop), epilogue (final
//! projection, sampling).
//!
//! Today this is just the type vocabulary — `Phase`, `PhaseSchedule`.
//! Future work (#28) wires the optimizer to specialize each phase
//! with its own tile sizes / fusion patterns.

use rlx_ir::{Graph, NodeId, Op};

/// Where in a streaming forward pass a node belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// One-time work executed before the steady-state loop.
    /// Examples: embedding lookup, KV-cache initialization, prompt
    /// prefill matmuls.
    Prologue,
    /// The repeating per-token (or per-step) work.
    /// Examples: attention decode, token-by-token sampling.
    SteadyState,
    /// One-time work after the loop terminates.
    /// Examples: detokenization, log-prob aggregation.
    Epilogue,
}

impl Phase {
    /// Order matters: optimizer can rely on this for scheduling.
    pub fn order(self) -> u8 {
        match self {
            Self::Prologue => 0,
            Self::SteadyState => 1,
            Self::Epilogue => 2,
        }
    }
}

/// Per-node phase assignment for a streaming graph. Built by the
/// (future) phase-derivation pass; downstream codegen specializes
/// each phase independently.
#[derive(Debug, Clone, Default)]
pub struct PhaseSchedule {
    map: std::collections::HashMap<NodeId, Phase>,
}

impl PhaseSchedule {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&mut self, node: NodeId, phase: Phase) {
        self.map.insert(node, phase);
    }
    pub fn get(&self, node: NodeId) -> Option<Phase> {
        self.map.get(&node).copied()
    }
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, Phase)> + '_ {
        self.map.iter().map(|(&id, &p)| (id, p))
    }

    /// Nodes in a given phase, in NodeId order (deterministic).
    pub fn nodes_in(&self, phase: Phase) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = self
            .map
            .iter()
            .filter_map(|(&id, &p)| if p == phase { Some(id) } else { None })
            .collect();
        v.sort();
        v
    }
}

/// Walk a graph and produce a [`PhaseSchedule`] (plan #28). The
/// classifier is heuristic — based on producer/consumer position
/// in the topological order — but conservative: every node is
/// assigned exactly one phase.
///
/// Heuristic:
///   - Inputs / params / constants: Prologue (they're loaded once).
///   - Reductions over a sequence dim with constant scope (e.g.
///     final pooling, sample): Epilogue.
///   - Sample / TopK: Epilogue (last-step ops).
///   - Everything between: SteadyState.
///
/// This is a starting point — refinements (e.g. detecting
/// embedding-lookup → attention vs. attention → projection) come
/// once we have a streaming runtime to specialize per phase.
pub fn derive_phases(graph: &Graph) -> PhaseSchedule {
    let mut sched = PhaseSchedule::new();
    let n = graph.len();
    if n == 0 {
        return sched;
    }

    // Nodes flagged as boundary (Inputs/Params/Constants) → Prologue.
    let mut last_compute_step: Option<usize> = None;
    let mut last_sample_step: Option<usize> = None;
    for (step, node) in graph.nodes().iter().enumerate() {
        match &node.op {
            Op::Sample { .. } | Op::TopK { .. } => {
                last_sample_step = Some(step);
            }
            Op::MatMul
            | Op::FusedMatMulBiasAct { .. }
            | Op::Attention { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::DotGeneral { .. }
            | Op::GroupedMatMul
            | Op::LoraMatMul { .. }
            | Op::DequantMatMul { .. } => {
                last_compute_step = Some(step);
            }
            _ => {}
        }
    }

    for (step, node) in graph.nodes().iter().enumerate() {
        let phase = match &node.op {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => Phase::Prologue,
            // After the last compute step, anything else is epilogue
            // (final projection, sampling, detokenize).
            Op::Sample { .. } | Op::TopK { .. } => Phase::Epilogue,
            _ => {
                if let Some(last) = last_sample_step {
                    if step > last
                        || (last_compute_step.is_some() && Some(step) > last_compute_step)
                    {
                        Phase::Epilogue
                    } else {
                        Phase::SteadyState
                    }
                } else if let Some(last) = last_compute_step {
                    if step > last {
                        Phase::Epilogue
                    } else {
                        Phase::SteadyState
                    }
                } else {
                    Phase::SteadyState
                }
            }
        };
        sched.set(node.id, phase);
    }
    sched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_phases_classifies_typical_graph() {
        use rlx_ir::*;
        let f = DType::F32;
        let mut g = Graph::new("derive");
        let x = g.input("x", Shape::new(&[1, 8], f)); // Prologue
        let w = g.param("w", Shape::new(&[8, 4], f)); // Prologue
        let mm = g.matmul(x, w, Shape::new(&[1, 4], f)); // SteadyState
        let s = g.sample(mm, 0, 1.0, 1.0, 0, Shape::new(&[1], f)); // Epilogue
        g.set_outputs(vec![s]);

        let sched = derive_phases(&g);
        assert_eq!(sched.get(x), Some(Phase::Prologue));
        assert_eq!(sched.get(w), Some(Phase::Prologue));
        assert_eq!(sched.get(mm), Some(Phase::SteadyState));
        assert_eq!(sched.get(s), Some(Phase::Epilogue));
    }

    #[test]
    fn schedule_partitions_by_phase() {
        let mut s = PhaseSchedule::new();
        s.set(NodeId(0), Phase::Prologue);
        s.set(NodeId(1), Phase::SteadyState);
        s.set(NodeId(2), Phase::SteadyState);
        s.set(NodeId(3), Phase::Epilogue);
        assert_eq!(s.nodes_in(Phase::Prologue), vec![NodeId(0)]);
        assert_eq!(s.nodes_in(Phase::SteadyState), vec![NodeId(1), NodeId(2)]);
        assert_eq!(s.nodes_in(Phase::Epilogue), vec![NodeId(3)]);
    }

    #[test]
    fn phase_ordering_is_deterministic() {
        assert!(Phase::Prologue.order() < Phase::SteadyState.order());
        assert!(Phase::SteadyState.order() < Phase::Epilogue.order());
    }
}
