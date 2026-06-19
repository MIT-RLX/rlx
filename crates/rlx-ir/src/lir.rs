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

//! **LIR** — low-level IR.
//!
//! Optimized MIR plus a concrete execution plan: arena layout, topo
//! schedule, view aliases, streaming phases, and an I/O manifest.
//! Backends lower LIR to device thunks/kernels without re-running the
//! optimizer or memory planner when the embedded plan is still valid.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::mir::MirModule;
use crate::phase::{Phase, PhaseSchedule};
use crate::{Graph, NodeId, Op};

/// A buffer slot in the arena.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LirBufferSlot {
    pub offset: usize,
    pub size: usize,
}

/// A view node that aliases part of a root buffer (no separate allocation).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LirViewAlias {
    pub root: NodeId,
    pub byte_offset: usize,
}

/// Named graph boundaries — stable handles for runtime I/O wiring.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LirIoManifest {
    pub inputs: Vec<(String, NodeId)>,
    pub params: Vec<(String, NodeId)>,
    pub outputs: Vec<NodeId>,
}

impl LirIoManifest {
    pub fn collect(graph: &Graph) -> Self {
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => inputs.push((name.clone(), node.id)),
                Op::Param { name } => params.push((name.clone(), node.id)),
                _ => {}
            }
        }
        Self {
            inputs,
            params,
            outputs: graph.outputs.clone(),
        }
    }
}

/// Liveness-aware buffer assignment + execution metadata.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LirBufferPlan {
    pub arena_size: usize,
    pub assignments: HashMap<NodeId, LirBufferSlot>,
    /// Topological execution order (node ids).
    pub schedule: Vec<NodeId>,
    /// Pure-view nodes (`Reshape`, identity `Cast`, axis-0 `Narrow`).
    pub view_aliases: HashMap<NodeId, LirViewAlias>,
    /// Streaming inference phase per node.
    pub phases: PhaseSchedule,
    /// Input / param / output node manifest.
    pub io: LirIoManifest,
    /// Arena alignment used when planning (bytes).
    pub alignment: usize,
    /// Dynamic symbols referenced by the graph at plan time.
    pub dynamic_symbols: Vec<u32>,
}

impl Default for LirBufferPlan {
    fn default() -> Self {
        Self {
            arena_size: 0,
            assignments: HashMap::new(),
            schedule: Vec::new(),
            view_aliases: HashMap::new(),
            phases: PhaseSchedule::new(),
            io: LirIoManifest::default(),
            alignment: 64,
            dynamic_symbols: Vec::new(),
        }
    }
}

impl LirBufferPlan {
    pub fn total_unshared_bytes(&self) -> usize {
        self.assignments.values().map(|s| s.size).sum()
    }

    pub fn bytes_saved(&self) -> usize {
        self.total_unshared_bytes().saturating_sub(self.arena_size)
    }

    pub fn slot(&self, id: NodeId) -> Option<&LirBufferSlot> {
        self.assignments.get(&id)
    }

    pub fn is_view(&self, id: NodeId) -> bool {
        self.view_aliases.contains_key(&id)
    }

    pub fn phase_of(&self, id: NodeId) -> Option<Phase> {
        self.phases.get(id)
    }

    pub fn nodes_in_phase(&self, phase: Phase) -> Vec<NodeId> {
        self.phases.nodes_in_ordered(phase, Some(&self.schedule))
    }
}

/// Stable compile fingerprint for AOT cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LirFingerprint(pub u64);

impl LirFingerprint {
    pub fn of(module: &LirModule) -> Self {
        let mut h = DefaultHasher::new();
        module.mir.name().hash(&mut h);
        module.mir.len().hash(&mut h);
        for node in module.mir.as_graph().nodes() {
            node.id.0.hash(&mut h);
            format!("{}", node.op).hash(&mut h);
            node.shape.hash(&mut h);
            node.inputs.len().hash(&mut h);
            for inp in &node.inputs {
                inp.0.hash(&mut h);
            }
        }
        for out in module.mir.as_graph().outputs.iter() {
            out.0.hash(&mut h);
        }
        module.buffers.arena_size.hash(&mut h);
        module.buffers.schedule.len().hash(&mut h);
        module.buffers.alignment.hash(&mut h);
        module.buffers.view_aliases.len().hash(&mut h);
        Self(h.finish())
    }
}

/// Low-level module — backend compile input after optimization.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LirModule {
    pub mir: MirModule,
    pub buffers: LirBufferPlan,
}

impl LirModule {
    pub fn new(mir: MirModule, buffers: LirBufferPlan) -> Self {
        Self { mir, buffers }
    }

    pub fn name(&self) -> &str {
        self.mir.name()
    }

    pub fn arena_size(&self) -> usize {
        self.buffers.arena_size
    }

    pub fn fingerprint(&self) -> LirFingerprint {
        LirFingerprint::of(self)
    }

    pub fn plan(&self) -> &LirBufferPlan {
        &self.buffers
    }

    /// Extract the optimized MIR graph for legacy backend entry points.
    pub fn into_graph(self) -> Graph {
        self.mir.into_graph()
    }

    pub fn as_graph(&self) -> &Graph {
        self.mir.as_graph()
    }

    pub fn has_dynamic_dims(&self) -> bool {
        crate::dynamic::has_dynamic_dims(self.as_graph())
    }

    pub fn is_fully_static(&self) -> bool {
        !self.has_dynamic_dims()
    }

    pub fn dynamic_symbols(&self) -> &[u32] {
        &self.buffers.dynamic_symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn io_manifest_collects_boundaries() {
        let mut g = Graph::new("m");
        let x = g.input("x", f32_shape(&[4]));
        let w = g.param("w", f32_shape(&[4, 4]));
        let y = g.matmul(x, w, f32_shape(&[4, 4]));
        g.set_outputs(vec![y]);

        let io = LirIoManifest::collect(&g);
        assert_eq!(io.inputs, vec![("x".into(), x)]);
        assert_eq!(io.params, vec![("w".into(), w)]);
        assert_eq!(io.outputs, vec![y]);
    }

    #[test]
    fn fingerprint_is_stable() {
        let mut g = Graph::new("m");
        let x = g.input("x", f32_shape(&[2]));
        g.set_outputs(vec![x]);
        let mir = MirModule::from_graph(g);
        let plan = LirBufferPlan {
            arena_size: 8,
            assignments: [(x, LirBufferSlot { offset: 0, size: 8 })]
                .into_iter()
                .collect(),
            schedule: vec![x],
            io: LirIoManifest {
                inputs: vec![("x".into(), x)],
                ..Default::default()
            },
            ..Default::default()
        };
        let lir = LirModule::new(mir, plan);
        assert_eq!(lir.fingerprint(), lir.fingerprint());
    }
}
