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

//! Streaming inference phases attached to LIR nodes (plan #16 / #28).

use std::collections::HashMap;

use crate::{Graph, NodeId, Op};

/// Where in a streaming forward pass a node belongs.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// One-time setup: embedding lookup, KV init, prompt prefill.
    Prologue,
    /// Per-token decode loop body.
    SteadyState,
    /// Final projection, sampling, detokenization.
    Epilogue,
}

impl Phase {
    pub fn order(self) -> u8 {
        match self {
            Self::Prologue => 0,
            Self::SteadyState => 1,
            Self::Epilogue => 2,
        }
    }
}

/// Per-node phase assignment for a streaming graph.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhaseSchedule {
    map: HashMap<NodeId, Phase>,
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

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Nodes in a given phase, in schedule order when `schedule` is provided,
    /// otherwise sorted by [`NodeId`].
    pub fn nodes_in(&self, phase: Phase) -> Vec<NodeId> {
        self.nodes_in_ordered(phase, None)
    }

    pub fn nodes_in_ordered(&self, phase: Phase, schedule: Option<&[NodeId]>) -> Vec<NodeId> {
        if let Some(order) = schedule {
            return order
                .iter()
                .copied()
                .filter(|id| self.get(*id) == Some(phase))
                .collect();
        }
        let mut v: Vec<NodeId> = self
            .map
            .iter()
            .filter_map(|(&id, &p)| if p == phase { Some(id) } else { None })
            .collect();
        v.sort();
        v
    }
}

/// Heuristic phase classifier for optimized MIR graphs.
pub fn derive_phases(graph: &Graph) -> PhaseSchedule {
    let mut sched = PhaseSchedule::new();
    let n = graph.len();
    if n == 0 {
        return sched;
    }

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
            | Op::DequantGroupedMatMul { .. }
            | Op::DequantMoEWeights { .. }
            | Op::LoraMatMul { .. }
            | Op::DequantMatMul { .. }
            | Op::GatedDeltaNet { .. } => {
                last_compute_step = Some(step);
            }
            _ => {}
        }
    }

    for (step, node) in graph.nodes().iter().enumerate() {
        let phase = match &node.op {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => Phase::Prologue,
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
    use crate::{DType, Shape};

    #[test]
    fn derive_phases_classifies_typical_graph() {
        let f = DType::F32;
        let mut g = Graph::new("derive");
        let x = g.input("x", Shape::new(&[1, 8], f));
        let w = g.param("w", Shape::new(&[8, 4], f));
        let mm = g.matmul(x, w, Shape::new(&[1, 4], f));
        let s = g.sample(mm, 0, 1.0, 1.0, 0, Shape::new(&[1], f));
        g.set_outputs(vec![s]);

        let sched = derive_phases(&g);
        assert_eq!(sched.get(x), Some(Phase::Prologue));
        assert_eq!(sched.get(w), Some(Phase::Prologue));
        assert_eq!(sched.get(mm), Some(Phase::SteadyState));
        assert_eq!(sched.get(s), Some(Phase::Epilogue));
    }

    #[test]
    fn phase_ordering_is_deterministic() {
        assert!(Phase::Prologue.order() < Phase::SteadyState.order());
        assert!(Phase::SteadyState.order() < Phase::Epilogue.order());
    }
}
