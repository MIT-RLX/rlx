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
//! Partition a graph into HLO segments and host Gaussian-splat steps.

use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId, Op};

/// One compile/run segment for the TPU orchestrator.
pub enum Segment {
    /// Lowered to HLO + PJRT; `output_orig` lists original graph node ids
    /// whose values are produced on the host after execute.
    Hlo {
        graph: Graph,
        output_orig: Vec<NodeId>,
    },
    SplatRender {
        node: NodeId,
    },
    SplatBackward {
        node: NodeId,
    },
}

/// True when the graph contains any splat op (needs orchestration).
pub fn needs_orchestration(graph: &Graph) -> bool {
    graph.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. }
        )
    })
}

/// Split `graph` into alternating HLO and host-splat segments (topo order).
pub fn plan(graph: &Graph) -> Vec<Segment> {
    let mut hlo_batch: Vec<NodeId> = Vec::new();
    let mut segments = Vec::new();

    for nid in graph.topo_order() {
        match &graph.node(nid).op {
            Op::GaussianSplatRender { .. } => {
                flush_hlo(graph, &mut hlo_batch, &mut segments);
                segments.push(Segment::SplatRender { node: nid });
            }
            Op::GaussianSplatRenderBackward { .. } => {
                flush_hlo(graph, &mut hlo_batch, &mut segments);
                segments.push(Segment::SplatBackward { node: nid });
            }
            Op::Input { .. } | Op::Param { .. } => {}
            _ => hlo_batch.push(nid),
        }
    }
    flush_hlo(graph, &mut hlo_batch, &mut segments);
    segments
}

fn flush_hlo(graph: &Graph, batch: &mut Vec<NodeId>, segments: &mut Vec<Segment>) {
    if batch.is_empty() {
        return;
    }
    let (seg_graph, output_orig) = build_hlo_segment_graph(graph, batch);
    segments.push(Segment::Hlo {
        graph: seg_graph,
        output_orig,
    });
    batch.clear();
}

/// Build a segment graph: Input/Param roots, boundary tensors as extra inputs,
/// and segment compute nodes. Returns segment outputs mapped to original ids.
fn build_hlo_segment_graph(graph: &Graph, batch: &[NodeId]) -> (Graph, Vec<NodeId>) {
    let batch_set: HashSet<NodeId> = batch.iter().copied().collect();

    let mut segment_outputs = Vec::new();
    for &nid in batch {
        let external_user = graph.users(nid).iter().any(|u| !batch_set.contains(u));
        let is_graph_out = graph.outputs.contains(&nid);
        if external_user || is_graph_out {
            segment_outputs.push(nid);
        }
    }

    let mut boundary = HashSet::new();
    let mut roots = HashSet::new();
    for &nid in batch {
        for &inp in &graph.node(nid).inputs {
            if batch_set.contains(&inp) {
                continue;
            }
            match &graph.node(inp).op {
                Op::Input { .. } | Op::Param { .. } => {
                    roots.insert(inp);
                }
                _ => {
                    boundary.insert(inp);
                }
            }
        }
    }

    let mut seg = Graph::new(format!("{}_seg", graph.name));
    let mut orig_to_seg: HashMap<NodeId, NodeId> = HashMap::new();

    for &orig in &roots {
        let n = graph.node(orig);
        let new_id = match &n.op {
            Op::Input { name } => seg.input(name, n.shape.clone()),
            Op::Param { name } => seg.param(name, n.shape.clone()),
            _ => unreachable!(),
        };
        orig_to_seg.insert(orig, new_id);
    }

    for &orig in &boundary {
        let shape = graph.node(orig).shape.clone();
        let name = format!("__bnd_{}", orig.0);
        let new_id = seg.input(&name, shape);
        orig_to_seg.insert(orig, new_id);
    }

    for &orig in batch {
        let n = graph.node(orig);
        let new_inputs: Vec<NodeId> = n
            .inputs
            .iter()
            .map(|&i| {
                *orig_to_seg
                    .get(&i)
                    .unwrap_or_else(|| panic!("rlx-tpu segment: missing remap for {i:?}"))
            })
            .collect();
        let new_id = seg.append_node(
            n.op.clone(),
            new_inputs,
            n.shape.clone(),
            n.name.clone(),
        );
        orig_to_seg.insert(orig, new_id);
    }

    let out_seg: Vec<NodeId> = segment_outputs
        .iter()
        .map(|&o| {
            *orig_to_seg
                .get(&o)
                .unwrap_or_else(|| panic!("rlx-tpu segment: output {o:?} not in batch"))
        })
        .collect();
    seg.set_outputs(out_seg);

    (seg, segment_outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::ops::splat::GaussianSplatRenderParams;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape, ops::splat::GaussianSplatInputs};

    #[test]
    fn plan_splits_splat_from_matmul() {
        let mut g = Graph::new("plan_test");
        let shape = Shape::new(&[4], DType::F32);
        let a = g.input("a", shape.clone());
        let b = g.input("b", shape.clone());
        let c = g.binary(BinaryOp::Add, a, b, shape.clone());
        let splat_in = GaussianSplatInputs {
            positions: a,
            scales: a,
            rotations: a,
            opacities: a,
            colors: a,
            sh_coeffs: a,
            meta: a,
        };
        let splat = g.gaussian_splat_render(
            splat_in,
            GaussianSplatRenderParams {
                width: 2,
                height: 2,
                ..Default::default()
            },
        );
        let d = g.binary(BinaryOp::Add, c, splat, shape);
        g.set_outputs(vec![d]);

        let segs = plan(&g);
        assert_eq!(segs.len(), 3);
        assert!(matches!(segs[0], Segment::Hlo { .. }));
        assert!(matches!(segs[1], Segment::SplatRender { .. }));
        assert!(matches!(segs[2], Segment::Hlo { .. }));
    }

    #[test]
    fn needs_orchestration_detects_splat() {
        let mut g = Graph::new("x");
        let a = g.input("a", Shape::new(&[1], DType::F32));
        g.set_outputs(vec![a]);
        assert!(!needs_orchestration(&g));

        let splat_in = GaussianSplatInputs {
            positions: a,
            scales: a,
            rotations: a,
            opacities: a,
            colors: a,
            sh_coeffs: a,
            meta: a,
        };
        let _ = g.gaussian_splat_render(splat_in, GaussianSplatRenderParams::default());
        assert!(needs_orchestration(&g));
    }
}
