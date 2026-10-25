// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// CPU thunks execute f32/f64 kernels. Graphs authored in F16/BF16 are
// promoted to F32 at compile time on CPU and GPU backends; boundary dtypes
// are preserved for typed I/O (`run_typed` / `set_param_typed`).

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

/// Declared boundary dtypes from the user graph (before f32 promotion).
#[derive(Debug, Clone, Default)]
pub struct IoDtypeManifest {
    #[allow(dead_code)]
    pub inputs: HashMap<String, DType>,
    #[allow(dead_code)]
    pub params: HashMap<String, DType>,
    pub outputs: Vec<DType>,
}

impl IoDtypeManifest {
    pub fn from_graph(g: &Graph) -> Self {
        let mut inputs = HashMap::new();
        let mut params = HashMap::new();
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } => {
                    inputs.insert(name.clone(), node.shape.dtype());
                }
                Op::Param { name } => {
                    params.insert(name.clone(), node.shape.dtype());
                }
                _ => {}
            }
        }
        let outputs = g
            .outputs
            .iter()
            .map(|&id| g.node(id).shape.dtype())
            .collect();
        Self {
            inputs,
            params,
            outputs,
        }
    }

    pub fn output_dtype(&self, idx: usize, fallback: DType) -> DType {
        self.outputs.get(idx).copied().unwrap_or(fallback)
    }
}

/// Capture boundary dtypes, then promote F16/BF16 graphs to F32 for GPU/CPU exec.
#[allow(dead_code)]
pub fn prepare_f32_exec_graph(graph: Graph) -> (Graph, IoDtypeManifest) {
    let manifest = IoDtypeManifest::from_graph(&graph);
    let exec = if needs_f32_exec(&graph) {
        promote_to_f32(graph)
    } else {
        graph
    };
    (exec, manifest)
}

/// Widen integer *control* `Param` nodes (duration carry, masks, trip counts)
/// to F32 so f32-uniform backends don't treat raw i64 uploads as denormals.
///
/// Mirrors Metal's `widen_integer_activations_to_f32` Param branch. Packed
/// U8/I8 weights and floating params are left alone. Callers must also widen
/// matching `set_param_typed` uploads via [`crate::backend::widen_bytes_to_f32`].
#[allow(dead_code)] // mlx_backend only; metal/wgpu widen params in-graph instead
pub fn widen_integer_control_params_to_f32(mut graph: Graph) -> Graph {
    for node in graph.nodes_mut() {
        if !matches!(node.op, Op::Param { .. }) {
            continue;
        }
        let old = node.shape.dtype();
        if matches!(
            old,
            DType::I32 | DType::I64 | DType::U32 | DType::Bool | DType::I16
        ) {
            node.shape = node.shape.clone().with_dtype(DType::F32);
        }
    }
    graph
}

pub fn needs_f32_exec(g: &Graph) -> bool {
    g.nodes().iter().any(|n| {
        if !matches!(n.shape.dtype(), DType::F16 | DType::BF16) {
            return false;
        }
        // User-registered `Op::Custom` kernels may execute natively at
        // F16/BF16; only built-in ops need the f32 promotion rewrite.
        !matches!(
            &n.op,
            Op::Custom { .. } | Op::Constant { .. } | Op::Input { .. } | Op::Param { .. }
        )
    })
}

fn promote_dtype(dt: DType) -> DType {
    match dt {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

fn promote_shape(shape: &Shape) -> Shape {
    shape.clone().with_dtype(promote_dtype(shape.dtype()))
}

fn widen_constant_bytes(data: &[u8], from: DType) -> Vec<u8> {
    match from {
        DType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| {
                let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                v.to_le_bytes()
            })
            .collect(),
        DType::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| {
                let v = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                v.to_le_bytes()
            })
            .collect(),
        _ => data.to_vec(),
    }
}

fn is_lowp_layout_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Concat { .. }
            | Op::Narrow { .. }
            | Op::Reshape { .. }
            | Op::Transpose { .. }
            | Op::Expand { .. }
            // A single-row copy into the cache it aliases — it moves bits and
            // computes nothing, so there is no arithmetic to promote. Promoting
            // it split the op from its own buffer: the cache is a boundary
            // tensor and keeps its declared F16/BF16, while every backend takes
            // the row stride from the KvAppend NODE's dtype, so an F32 node over
            // an F16 cache strides a layout it does not have.
            | Op::KvAppend { .. }
    )
}

/// Rewrite F16/BF16 node shapes (and constant payloads) to F32 for CPU exec.
///
/// Boundary tensors (`Param` / `Input`) keep their declared dtype so Metal (and
/// other backends with native F16 weight paths) can store Linear weights at
/// half width. Pure layout ops that only rearrange F16/BF16 tensors (e.g. the
/// weight `Concat` from shared-input MatMul fusion) stay low-precision too —
/// promoting those would materialize a full F32 copy of every packed weight.
pub fn promote_to_f32(graph: Graph) -> Graph {
    promote(graph, Keep::NOTHING)
}

/// [`promote_to_f32`] for the CPU backend, which also keeps a half `Param`
/// whose every consumer is a `MatMul`'s weight (its right operand) or an
/// axis-0 `Gather`'s table at half width: the CPU's `SgemmF16` / `SgemmBf16`
/// read packed half weights against f32 activations and its gather widens a
/// packed table's rows, so a package opened with `weight_dtype: f16` holds
/// them at 2 B each rather than widened back to 4.
#[cfg(any(feature = "cpu", test))]
pub fn promote_to_f32_keeping_half_weights(graph: Graph) -> Graph {
    promote(
        graph,
        Keep {
            weights: true,
            tables: true,
            ..Keep::NOTHING
        },
    )
}

/// [`prepare_f32_exec_graph`] keeping a half `Param` whose every consumer is
/// a `MatMul`'s weight, or an axis-0 `Gather`'s table, at half width (Metal:
/// its matmuls read half weights against f32 activations, and such a gather
/// stays half — a row copy — for the `Cast` after it to widen). Without it,
/// one half node — the rows a half embedding table gathers — widened every
/// weight of the graph back to f32 (Qwen2.5-VL-3B's step: 7.4 → 13.6 GB).
#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
pub fn prepare_f32_exec_graph_keeping_half_matmul_weights(
    graph: Graph,
) -> (Graph, IoDtypeManifest) {
    let manifest = IoDtypeManifest::from_graph(&graph);
    let keep = Keep {
        weights: true,
        tables: true,
        gathers: true,
        // Metal has no BF16 storage: those widen as before.
        only_f16: true,
    };
    (promote(graph, keep), manifest)
}

/// Which half params [`promote`] leaves half.
#[derive(Clone, Copy)]
struct Keep {
    /// A `Param` read only as a matmul's weight.
    weights: bool,
    /// … or also as an axis-0 `Gather`'s table.
    tables: bool,
    /// Such a table's gather keeps its half rows (a backend whose gather
    /// copies rows of its output's type); else it is promoted (the CPU's
    /// gather widens a half table's rows itself).
    gathers: bool,
    /// Only F16 params, not BF16.
    only_f16: bool,
}

impl Keep {
    const NOTHING: Keep = Keep {
        weights: false,
        tables: false,
        gathers: false,
        only_f16: false,
    };
}

fn promote(graph: Graph, keep: Keep) -> Graph {
    if !needs_f32_exec(&graph) {
        return graph;
    }
    let mut out = Graph::new(format!("{}_f32_exec", graph.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    // Consumers in the SOURCE graph, to decide whether a boundary may stay
    // low-precision. See `boundary_stays_lowp`.
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            consumers.entry(i).or_default().push(n.id);
        }
    }
    // A low-precision `Input` / `Param` may keep its dtype only while every
    // consumer also stays low-precision.
    //
    // Keeping *every* boundary low-p is load-bearing for one case and wrong for
    // the rest. Load-bearing: `KvAppend` aliases its cache input's buffer and
    // every backend takes the row stride from the KvAppend node's dtype, so
    // promoting the op while the cache stays half-width strides a layout the
    // buffer does not have (see the test below). Wrong for the rest: a BF16
    // `Input` feeding a promoted F32 `Binary` gets a 2 B/elem slot that the f32
    // thunk then reads 4 B/elem from, and the values come back as pairs of
    // halves —
    //
    //   [1,2,3,4] into a BF16 input, + 0  ->  [2.0038757, 4.007843, 0.0, 0.0]
    //
    // with no error and no cast anywhere between them. Promoting the boundary
    // is what makes producer and consumer agree on the width; the *declared*
    // dtype is not lost, because `IoDtypeManifest` records it and
    // `CompiledGraph::io_dtypes` narrows at the API edge.
    let boundary_stays_lowp = |id: NodeId| -> bool {
        let weight_only = keep.weights
            && (!keep.only_f16 || graph.node(id).shape.dtype() == DType::F16)
            && matches!(graph.node(id).op, Op::Param { .. })
            && consumers.get(&id).is_some_and(|cs| {
                cs.iter().all(|&c| {
                    let cn = graph.node(c);
                    let matmul_weight = matches!(cn.op, Op::MatMul)
                        && cn.inputs.get(1) == Some(&id)
                        && cn.inputs.first() != Some(&id);
                    // An embedding table: the CPU gather widens its rows.
                    let table = keep.tables
                        && matches!(cn.op, Op::Gather { axis: 0 })
                        && cn.inputs.first() == Some(&id)
                        && cn.inputs.get(1) != Some(&id);
                    matmul_weight || table
                })
            });
        if weight_only {
            return true;
        }
        match consumers.get(&id) {
            // Nothing reads it — nothing to disagree with.
            None => true,
            Some(cs) => cs.iter().all(|&c| {
                let cn = graph.node(c);
                is_lowp_layout_op(&cn.op) && matches!(cn.shape.dtype(), DType::F16 | DType::BF16)
            }),
        }
    };
    // Node ids in the *source* graph whose dtype we intentionally kept low-p.
    let mut kept_lowp: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let is_boundary =
            matches!(&node.op, Op::Param { .. } | Op::Input { .. }) && boundary_stays_lowp(node.id);
        let is_layout_of_lowp = is_lowp_layout_op(&node.op)
            && matches!(node.shape.dtype(), DType::F16 | DType::BF16)
            && !node.inputs.is_empty()
            && node.inputs.iter().all(|&i| {
                kept_lowp.contains(&i)
                    || matches!(graph.node(i).shape.dtype(), DType::F16 | DType::BF16)
            });
        let is_gather_of_lowp = keep.gathers
            && matches!(node.op, Op::Gather { axis: 0 })
            && matches!(node.shape.dtype(), DType::F16 | DType::BF16)
            && kept_lowp.contains(&node.inputs[0]);
        let keep_dtype = is_boundary || is_layout_of_lowp || is_gather_of_lowp;
        let shape = if keep_dtype {
            node.shape.clone()
        } else {
            promote_shape(&node.shape)
        };
        if keep_dtype && matches!(shape.dtype(), DType::F16 | DType::BF16) {
            kept_lowp.insert(node.id);
        }
        let op = match &node.op {
            Op::Constant { data } => Op::Constant {
                data: widen_constant_bytes(data, node.shape.dtype()),
            },
            Op::Cast { to } => Op::Cast {
                to: promote_dtype(*to),
            },
            other => other.clone(),
        };
        let new_id = out.add_node(op, inputs, shape);
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::GraphExt;

    /// **`KvAppend` must keep the dtype of the cache it writes into.**
    ///
    /// The op aliases its cache input's buffer, and every backend derives the
    /// row stride from the KvAppend NODE's dtype. The cache is a boundary
    /// tensor, so it keeps its declared F16/BF16 through this pass — promoting
    /// the node to F32 while leaving the cache half-width therefore strides a
    /// layout the buffer does not have, and the new token lands in the wrong
    /// row with nothing to report it.
    #[test]
    fn a_low_precision_kv_append_is_not_promoted() {
        for dt in [DType::F16, DType::BF16] {
            let mut g = Graph::new("kv");
            let cache = g.input("cache", Shape::new(&[1, 8, 4], dt));
            let row = g.input("row", Shape::new(&[1, 1, 4], dt));
            let out = g.add_node(
                Op::KvAppend { axis: 1, pos: 3 },
                vec![cache, row],
                Shape::new(&[1, 4, 4], dt),
            );
            g.set_outputs(vec![out]);

            let promoted = promote_to_f32(g);
            let node = promoted.node(promoted.outputs[0]);
            assert!(
                matches!(node.op, Op::KvAppend { .. }),
                "{dt:?}: the append itself vanished"
            );
            assert_eq!(
                node.shape.dtype(),
                dt,
                "{dt:?}: KvAppend was promoted away from its cache's dtype"
            );
        }
    }

    /// A half weight read only as a `MatMul`'s right operand stays half on
    /// CPU (its packed GEMM reads it), the matmul's output promotes; one read
    /// by anything else promotes, and other backends promote it too.
    #[test]
    fn a_half_matmul_weight_stays_half_on_cpu_only() {
        let build = |extra_use: bool| {
            let mut g = Graph::new("mm");
            let x = g.input("x", Shape::new(&[2, 4], DType::F32));
            let w = g.param("w", Shape::new(&[4, 3], DType::F16));
            let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[2, 3], DType::F16));
            let mut outs = vec![y];
            if extra_use {
                outs.push(g.add_node(
                    Op::Activation(rlx_ir::op::Activation::Relu),
                    vec![w],
                    Shape::new(&[4, 3], DType::F16),
                ));
            }
            g.set_outputs(outs);
            g
        };
        let param = |g: &Graph| {
            g.nodes()
                .iter()
                .find(|n| matches!(n.op, Op::Param { .. }))
                .map(|n| n.shape.dtype())
                .unwrap()
        };
        let cpu = promote_to_f32_keeping_half_weights(build(false));
        assert_eq!(param(&cpu), DType::F16);
        // An embedding table read only by an axis-0 gather stays half too.
        let mut g = Graph::new("embed");
        let ids = g.input("ids", Shape::new(&[2], DType::I32));
        let t = g.param("table", Shape::new(&[5, 3], DType::F16));
        let rows = g.gather(t, ids, 0, Shape::new(&[2, 3], DType::F16));
        g.set_outputs(vec![rows]);
        let cpu_embed = promote_to_f32_keeping_half_weights(g);
        assert_eq!(param(&cpu_embed), DType::F16);
        assert_eq!(
            cpu_embed.node(cpu_embed.outputs[0]).shape.dtype(),
            DType::F32
        );
        assert_eq!(cpu.node(cpu.outputs[0]).shape.dtype(), DType::F32);
        assert_eq!(
            param(&promote_to_f32_keeping_half_weights(build(true))),
            DType::F32
        );
        assert_eq!(param(&promote_to_f32(build(false))), DType::F32);
    }

    /// The other half: a genuinely computed operand still promotes, so this
    /// exemption cannot be read as "low precision executes natively".
    #[test]
    fn arithmetic_over_the_same_operands_still_promotes() {
        let mut g = Graph::new("mul");
        let a = g.input("a", Shape::new(&[4], DType::F16));
        let b = g.input("b", Shape::new(&[4], DType::F16));
        let out = g.mul(a, b);
        g.set_outputs(vec![out]);

        let promoted = promote_to_f32(g);
        assert_eq!(promoted.node(promoted.outputs[0]).shape.dtype(), DType::F32);
    }
}
