// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pure elementwise program as generated MSL, one kernel per region.
//!
//! The thunk path writes every intermediate to the arena: 3.7 ms per op on a
//! 118 MB tensor, linear in the program's length, and MPSGraph does not fuse
//! such a program either. `ElementwiseRegion` only holds linear chains, while
//! complex arithmetic is a DAG: each value feeds several ops. So a region here
//! is any set of same-shaped f32 nodes with one result; it becomes one
//! `Op::Custom` whose attrs are the kernel's source, with the intermediates in
//! registers and the broadcast indexing baked in (shapes are static).
//!
//! The rewrite runs before memory planning, so the planner sees one node. Its
//! overlap test is inclusive, so the result never shares a slot with an
//! operand read by the same kernel. Only graphs with nothing but elementwise
//! ops qualify: a custom node takes a graph off MPSGraph, which is right for
//! these and wrong for a model. `RLX_METAL_NO_FUSED_ELEMENTWISE=1` turns it off.
//!
//! `RLX_METAL_FUSED_ELEMENTWISE_MIXED=1` also fuses the elementwise regions of
//! any other graph (the rest stays as is): for graphs that run on thunks
//! anyway, such as a training step with its optimizer update in the graph.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::{Arc, Mutex, Once};

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Dim, Graph, Node, NodeId, Op, Shape};

use crate::mtl::{ComputePipelineState, MTLSize};
use crate::op_registry::{MetalGpuDispatch, MetalGpuKernel, register_metal_gpu_kernel};

const NAME: &str = "rlx.metal.fused_elementwise";
/// Below this a region saves too little to pay for its own kernel.
const MIN_NODES: usize = 8;

fn static_dims(s: &Shape) -> Option<Vec<usize>> {
    s.dims()
        .iter()
        .map(|d| {
            if let Dim::Static(n) = d {
                Some(*n)
            } else {
                None
            }
        })
        .collect()
}

fn binary(op: BinaryOp, a: &str, b: &str) -> Option<String> {
    Some(match op {
        BinaryOp::Add => format!("{a} + {b}"),
        BinaryOp::Sub => format!("{a} - {b}"),
        BinaryOp::Mul => format!("{a} * {b}"),
        BinaryOp::Div => format!("{a} / {b}"),
        BinaryOp::Max => format!("fmax({a}, {b})"),
        BinaryOp::Min => format!("fmin({a}, {b})"),
        _ => return None,
    })
}

fn activation(act: Activation, x: &str) -> Option<String> {
    Some(match act {
        Activation::Neg => format!("-{x}"),
        Activation::Abs => format!("fabs({x})"),
        Activation::Exp => format!("exp({x})"),
        Activation::Log => format!("log({x})"),
        Activation::Sqrt => format!("sqrt({x})"),
        Activation::Rsqrt => format!("1.0f / sqrt({x})"),
        Activation::Recip => format!("1.0f / {x}"),
        Activation::Sin => format!("sin({x})"),
        Activation::Cos => format!("cos({x})"),
        Activation::Tanh => format!("tanh({x})"),
        _ => return None,
    })
}

/// `dims` reaches `out` by NumPy broadcasting.
fn broadcasts_to(dims: &[usize], out: &[usize]) -> bool {
    dims.len() <= out.len()
        && dims
            .iter()
            .rev()
            .zip(out.iter().rev())
            .all(|(&d, &o)| d == o || d == 1)
}

/// An op this module can write as one MSL expression, on f32 of static shape.
fn fusable(g: &Graph, n: &Node) -> bool {
    let op_ok = match &n.op {
        Op::Binary(b) => binary(*b, "", "").is_some(),
        Op::Activation(a) => activation(*a, "").is_some(),
        Op::Expand { .. } => true,
        _ => false,
    };
    op_ok
        && n.shape.dtype() == DType::F32
        && static_dims(&n.shape).is_some_and(|out| {
            n.inputs.iter().all(|&i| {
                let s = g.shape(i);
                s.dtype() == DType::F32 && static_dims(s).is_some_and(|d| broadcasts_to(&d, &out))
            })
        })
}

/// The element of an operand of dims `dims` for the output coordinates `c0…`.
fn index(dims: &[usize], out: &[usize]) -> String {
    if dims.iter().product::<usize>() == 1 {
        return "0".into();
    }
    if dims == out {
        return "gid".into();
    }
    let pad = out.len() - dims.len();
    let (mut terms, mut stride) = (Vec::new(), 1usize);
    for (ax, &d) in dims.iter().enumerate().rev() {
        if d != 1 {
            terms.push(format!("c{} * {stride}u", ax + pad));
        }
        stride *= d;
    }
    terms.join(" + ")
}

/// The kernel of `members` (ascending, the last one the result), reading
/// `externals` at `off[k]` and writing at the last `off`.
fn source(g: &Graph, members: &[NodeId], externals: &[NodeId]) -> String {
    let sink = *members.last().expect("a region has a result");
    let out = static_dims(&g.node(sink).shape).expect("static");
    let mut s = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void fused(device float* a [[buffer(0)]], constant uint* off [[buffer(1)]],\n\
         constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n\
         if (gid >= n) return;\nuint q = gid;\n",
    );
    for ax in (0..out.len()).rev() {
        let _ = writeln!(s, "uint c{ax} = q % {d}u; q /= {d}u;", d = out[ax]);
    }
    let ext: HashMap<NodeId, usize> = externals.iter().enumerate().map(|(k, &e)| (e, k)).collect();
    for (k, &e) in externals.iter().enumerate() {
        let d = static_dims(g.shape(e)).expect("static");
        let _ = writeln!(s, "float e{k} = a[off[{k}] + {}];", index(&d, &out));
    }
    let name = |i: NodeId| match ext.get(&i) {
        Some(k) => format!("e{k}"),
        None => format!("v{}", i.0),
    };
    for &m in members {
        let n = g.node(m);
        let expr = match &n.op {
            Op::Binary(b) => binary(*b, &name(n.inputs[0]), &name(n.inputs[1])),
            Op::Activation(act) => activation(*act, &name(n.inputs[0])),
            _ => Some(name(n.inputs[0])),
        };
        let _ = writeln!(s, "float v{} = {};", m.0, expr.expect("fusable"));
    }
    let _ = writeln!(s, "a[off[{}] + gid] = v{};\n}}", externals.len(), sink.0);
    s
}

/// `graph` with its elementwise regions as single custom nodes. The replaced
/// nodes are left dead for the caller's dead-code pass.
pub(crate) fn fuse(mut graph: Graph) -> Graph {
    if rlx_ir::env::flag("RLX_METAL_NO_FUSED_ELEMENTWISE") {
        return graph;
    }
    let pure = graph.nodes().iter().all(|n| {
        matches!(
            n.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        ) || fusable(&graph, n)
    });
    let mixed = !pure;
    if mixed && !rlx_ir::env::flag("RLX_METAL_FUSED_ELEMENTWISE_MIXED") {
        return graph;
    }
    // In a mixed graph each region replaces that many dispatches, and a
    // smaller-shaped value read only by one region (a step size, a bias
    // correction) is cheaper recomputed per element than run on its own.
    let min_nodes = if mixed { 2 } else { MIN_NODES };
    let n = graph.nodes().len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for node in graph.nodes() {
        for i in &node.inputs {
            consumers[i.0 as usize].push(node.id.0 as usize);
        }
    }
    let outputs: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let is_op = |nd: &Node| {
        !matches!(
            nd.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        )
    };

    // Backwards: a node joins a region when all its readers are in it and it
    // has the region's shape; otherwise it is materialized and heads its own.
    let mut region: Vec<Option<usize>> = vec![None; n];
    for i in (0..n).rev() {
        let node = &graph.nodes()[i];
        // Other ops stay out of every region, so their operands are
        // materialized (a reader outside the region).
        if !is_op(node) || !fusable(&graph, node) {
            continue;
        }
        let first = consumers[i].first().and_then(|&c| region[c]);
        let join = first.filter(|&r| {
            !outputs.contains(&node.id)
                && consumers[i].iter().all(|&c| region[c] == Some(r))
                && (graph.nodes()[r].shape.dims() == node.shape.dims()
                    || mixed
                        && static_dims(&node.shape)
                            .zip(static_dims(&graph.nodes()[r].shape))
                            .is_some_and(|(d, out)| broadcasts_to(&d, &out)))
        });
        region[i] = Some(join.unwrap_or(i));
    }
    let mut groups: HashMap<usize, Vec<NodeId>> = HashMap::new();
    for (i, r) in region.iter().enumerate() {
        if let Some(r) = r {
            groups.entry(*r).or_default().push(NodeId(i as u32));
        }
    }
    let mut rewrites = Vec::new();
    for (sink, members) in groups {
        if members.len() < min_nodes {
            continue;
        }
        let inside: HashSet<NodeId> = members.iter().copied().collect();
        let mut externals = Vec::new();
        for &m in &members {
            for &i in &graph.node(m).inputs {
                if !inside.contains(&i) && !externals.contains(&i) {
                    externals.push(i);
                }
            }
        }
        rewrites.push((sink, source(&graph, &members, &externals), externals));
    }
    if rewrites.is_empty() {
        return graph;
    }
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| register_metal_gpu_kernel(Arc::new(FusedKernel)));
    for (sink, src, externals) in rewrites {
        let node = &mut graph.nodes_mut()[sink];
        node.op = Op::Custom {
            name: NAME.into(),
            num_inputs: externals.len() as u32,
            attrs: src.into_bytes(),
        };
        node.inputs = externals;
    }
    graph
}

/// Compiles each distinct source once and dispatches it.
#[derive(Debug)]
struct FusedKernel;

fn pipeline(src: &[u8]) -> Result<ComputePipelineState, String> {
    static CACHE: Mutex<Option<HashMap<Vec<u8>, ComputePipelineState>>> = Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(p) = cache.get(src) {
        return Ok(p.clone());
    }
    let dev = crate::device::metal_device().ok_or("no Metal device")?;
    let text = std::str::from_utf8(src).map_err(|e| e.to_string())?;
    let opts = crate::mtl::CompileOptions::new();
    // same arithmetic as the per-op kernels' results, not a faster approximation
    opts.set_fast_math_enabled(false);
    let lib = dev
        .device
        .new_library_with_source(text, &opts)
        .map_err(|e| format!("{e:?}"))?;
    let f = lib
        .get_function("fused", None)
        .map_err(|e| format!("{e:?}"))?;
    let p = dev
        .device
        .new_compute_pipeline_state_with_function(&f)
        .map_err(|e| format!("{e:?}"))?;
    cache.insert(src.to_vec(), p.clone());
    Ok(p)
}

impl MetalGpuKernel for FusedKernel {
    fn name(&self) -> &str {
        NAME
    }

    fn encode(&self, d: &MetalGpuDispatch) -> Result<(), String> {
        let p = pipeline(d.attrs)?;
        let word = |bytes: usize| {
            u32::try_from(bytes / 4).map_err(|_| "arena offset past u32 words".to_string())
        };
        let mut off = Vec::with_capacity(d.inputs.len() + 1);
        for (o, _, _) in d.inputs {
            off.push(word(*o)?);
        }
        off.push(word(d.output.0)?);
        let len = d.output.1;
        d.encoder.set_compute_pipeline_state(&p);
        d.encoder.set_buffer(0, Some(d.arena), 0);
        d.encoder
            .set_bytes(1, (off.len() * 4) as u64, off.as_ptr() as *const _);
        d.encoder.set_bytes(2, 4, &len as *const u32 as *const _);
        let w = p.thread_execution_width().max(1);
        d.encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w * 4,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}
