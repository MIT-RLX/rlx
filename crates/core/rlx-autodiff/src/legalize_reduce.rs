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

//
// Decompose multi-axis `Op::Reduce` into single-axis chains for backends
// that only support one reduction axis at a time (e.g. WGPU).

use rlx_ir::shape::Dim;
use rlx_ir::{Graph, NodeId, Op, Shape};

/// Replace every `Reduce` with `axes.len() > 1` by a chain of single-axis
/// reductions (`keep_dim=true` on each step; final reshape drops dims if needed).
///
/// Builds a fresh graph in topological order (multi-axis Reduce nodes are
/// replaced **in-place** in the walk, not appended at the end). This is
/// required so downstream passes like `unfuse_fused_for_autodiff` — which
/// assume strict insertion=topological order — don't see consumer nodes
/// whose inputs sit later in the node list.
pub fn legalize_multi_axis_reduce(g: Graph) -> Graph {
    use std::collections::HashMap;

    // Cheap early-out: skip the rebuild if there's nothing to legalise.
    let any_multi = g
        .nodes()
        .iter()
        .any(|n| matches!(&n.op, Op::Reduce { axes, .. } if axes.len() > 1));
    if !any_multi {
        return g;
    }

    let mut out = Graph::new(g.name.clone());
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();

    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| remap[i]).collect();

        let final_new_id = match &node.op {
            Op::Reduce { op, axes, keep_dim } if axes.len() > 1 => {
                // Single-axis chain: reduce from the largest axis down so
                // intermediate shapes stay well-defined under the original
                // numbering. Each step uses `keep_dim=true`; a final
                // `Reshape` (only when the original was `keep_dim=false`)
                // collapses the size-1 dims back out.
                let mut cur = new_inputs[0];
                let mut shape = out.node(cur).shape.clone();
                let dtype = shape.dtype();
                let mut sorted = axes.clone();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                for &ax in &sorted {
                    let mut dims: Vec<Dim> = shape.dims().to_vec();
                    dims[ax] = Dim::Static(1);
                    let step_shape = Shape::from_dims(&dims, dtype);
                    cur = out.add_node(
                        Op::Reduce {
                            op: *op,
                            axes: vec![ax],
                            keep_dim: true,
                        },
                        vec![cur],
                        step_shape,
                    );
                    shape = out.node(cur).shape.clone();
                }
                if !*keep_dim {
                    let final_shape = node.shape.clone();
                    let new_shape_dims: Vec<i64> = final_shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            Dim::Static(n) => *n as i64,
                            Dim::Dynamic(_) => -1,
                        })
                        .collect();
                    cur = out.add_node(
                        Op::Reshape {
                            new_shape: new_shape_dims,
                        },
                        vec![cur],
                        final_shape,
                    );
                }
                cur
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        remap.insert(node.id, final_new_id);
    }

    let new_outputs: Vec<NodeId> = g.outputs.iter().map(|id| remap[id]).collect();
    out.set_outputs(new_outputs);
    out
}
