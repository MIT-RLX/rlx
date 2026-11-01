// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host fallback for [`Step::Host`](crate::Step::Host): an op with no WebGL
//! lowering runs on rlx-cpu itself, so it matches the CPU backend by
//! construction.
//!
//! The op becomes a one-node graph (its operands as `Input`s of their real
//! dtypes), compiled once with rlx-cpu's thunk executor. Each run encodes the
//! operands' slot values (one f32 per element, or per complex lane) into those
//! dtypes, executes, and decodes the output back to slot values. On the GL
//! path the operands are read back first and the result uploaded — a GPU
//! round trip per op, the price of covering every op rlx-cpu runs (sampling,
//! quantized matmuls, SVD/QR/eigen, registered custom ops, splatting, …).
//! rlx-cpu is built without BLAS here, so its pure-Rust paths are used — the
//! same code on native and wasm.

use crate::{Result, WebglError};
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{ThunkSchedule, compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use std::sync::Mutex;

/// One op compiled for rlx-cpu.
pub struct HostOp {
    graph: Graph,
    /// The rlx-cpu-prepared graph, its arena and schedule (built on first run).
    exe: Mutex<Option<(Graph, Arena, ThunkSchedule)>>,
}

impl std::fmt::Debug for HostOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = self.graph.node(self.graph.outputs[0]);
        write!(f, "HostOp({:?})", op.op.kind())
    }
}

impl HostOp {
    /// `op` applied to operands of shapes `inputs`, producing `out`.
    pub(crate) fn new(op: &Op, inputs: &[Shape], out: &Shape) -> Self {
        let mut graph = Graph::new("webgl_host");
        let ids: Vec<NodeId> = inputs
            .iter()
            .enumerate()
            .map(|(i, s)| graph.input(format!("in{i}"), s.clone()))
            .collect();
        let y = graph.add_node(op.clone(), ids, out.clone());
        graph.set_outputs(vec![y]);
        Self {
            graph,
            exe: Mutex::new(None),
        }
    }

    /// Run on the operands' slot values; returns the output's slot values.
    pub(crate) fn run(&self, inputs: &[&[f32]]) -> Result<Vec<f32>> {
        let mut guard = self
            .exe
            .lock()
            .map_err(|_| WebglError("host op poisoned".into()))?;
        let (g, arena, sched) = guard.get_or_insert_with(|| {
            let g = rlx_cpu::prepare_graph_for_thunks(self.graph.clone());
            let arena = Arena::from_plan(rlx_compile::memory::plan_memory(&g));
            let sched = compile_thunks(&g, &arena);
            (g, arena, sched)
        });
        for node in g.nodes() {
            let Op::Input { name } = &node.op else {
                continue;
            };
            let i: usize = name[2..].parse().unwrap_or(usize::MAX);
            let vals = inputs
                .get(i)
                .ok_or_else(|| WebglError(format!("host op: missing operand {i}")))?;
            let bytes = encode(vals, node.shape.dtype());
            let off = arena.byte_offset(node.id);
            arena.raw_buf_mut()[off..off + bytes.len()].copy_from_slice(&bytes);
        }
        execute_thunks(sched, arena.raw_buf_mut());
        let y = g.node(g.outputs[0]);
        let len = y.shape.num_elements().unwrap_or(0) * y.shape.dtype().size_bytes();
        let off = arena.byte_offset(y.id);
        Ok(crate::plan::decode_const(
            &arena.raw_buf()[off..off + len],
            y.shape.dtype(),
        ))
    }
}

/// Slot values (one f32 per element; complex as f32 lanes, C128 as df64
/// `[hi, lo]` pairs) → the dtype's bytes, as rlx-cpu stores it.
fn encode(v: &[f32], dt: DType) -> Vec<u8> {
    match dt {
        DType::F64 => v.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect(),
        DType::C128 => v
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[hi, lo]| (hi as f64 + lo as f64).to_le_bytes())
            .collect(),
        DType::F16 => v
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect(),
        DType::I8 => v.iter().map(|&x| x as i8 as u8).collect(),
        DType::U8 | DType::Bool => v.iter().map(|&x| x as u8).collect(),
        DType::I16 => v.iter().flat_map(|&x| (x as i16).to_le_bytes()).collect(),
        DType::I32 => v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        DType::U32 => v.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect(),
        DType::I64 => v.iter().flat_map(|&x| (x as i64).to_le_bytes()).collect(),
        _ => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
    }
}
