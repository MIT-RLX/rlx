// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::While` as a loop at run time, not unrolled: `cond` and `body`
//! compiled once, then `while cond(carried) { carried = body(carried) }`,
//! up to `max_iterations` when it has one. The node's value is the first
//! carried value after the loop.
//!
//! GPU backends reach it through their one-op host fallback
//! ([`super::HostOpDesc`]), which compiles the node alone on the CPU.

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::{DType, Graph, NodeId, Op};

use super::{ThunkSchedule, compile_thunks, execute_thunks};

/// A compiled `cond` or `body`: its schedule, its pristine arena, and where
/// its inputs (in `NodeId` order, the carried values) and outputs live.
pub struct WhileSubPlan {
    pub schedule: ThunkSchedule,
    pub init: Vec<u8>,
    pub inputs: Vec<(usize, usize)>,
    pub outputs: Vec<(usize, usize)>,
    pub output_dtype: DType,
}

/// Both halves of a loop, and its bound.
pub struct WhilePlan {
    pub cond: WhileSubPlan,
    pub body: WhileSubPlan,
    pub max_iterations: Option<usize>,
}

impl std::fmt::Debug for WhilePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhilePlan")
            .field("carried", &self.body.inputs.len())
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

fn compile_sub(g: &Graph, what: &str) -> WhileSubPlan {
    // Compiled as any graph is (integer storage kept, claimed fused ops
    // expanded), as a scan body is.
    let g = crate::expand::prepare_graph_for_thunks(g.clone());
    if let Some(n) = g.nodes().iter().find(|n| matches!(n.op, Op::Param { .. })) {
        panic!(
            "Op::While {what}: {} is a param; a loop's graphs read only the carried values \
             (and constants)",
            n.id
        );
    }
    let plan = rlx_opt::memory::plan_memory(&g);
    let offsets: HashMap<NodeId, usize> = plan
        .assignments
        .iter()
        .map(|(id, slot)| (*id, slot.offset))
        .collect();
    let mut arena = crate::arena::Arena::from_plan(plan);
    for n in g.nodes() {
        if let Op::Constant { data } = &n.op
            && arena.has_buffer(n.id)
            && !data.is_empty()
        {
            let off = arena.byte_offset(n.id);
            let buf = arena.raw_buf_mut();
            let nbytes = (buf.len() - off).min(data.len());
            buf[off..off + nbytes].copy_from_slice(&data[..nbytes]);
        }
    }
    let bytes = |id: NodeId| {
        g.node(id)
            .shape
            .size_bytes()
            .unwrap_or_else(|| panic!("Op::While {what}: {id} has no static size"))
    };
    let inputs = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input { .. }))
        .map(|n| (offsets[&n.id], bytes(n.id)))
        .collect();
    let outputs = g.outputs.iter().map(|&o| (offsets[&o], bytes(o))).collect();
    let output_dtype = g
        .outputs
        .first()
        .map(|&o| g.node(o).shape.dtype())
        .unwrap_or(DType::F32);
    let schedule = compile_thunks(&g, &arena);
    WhileSubPlan {
        schedule,
        init: arena.raw_buf().to_vec(),
        inputs,
        outputs,
        output_dtype,
    }
}

/// Compile `Op::While`'s halves, checking the body gives back what it is
/// given.
pub fn compile_while_plan(cond: &Graph, body: &Graph, max_iterations: Option<usize>) -> WhilePlan {
    let cond = compile_sub(cond, "cond");
    let body = compile_sub(body, "body");
    assert_eq!(cond.outputs.len(), 1, "Op::While: cond has one output");
    assert_eq!(
        body.outputs.len(),
        body.inputs.len(),
        "Op::While: the body gives back one value per carried value"
    );
    for (i, (o, n)) in body.outputs.iter().zip(&body.inputs).enumerate() {
        assert_eq!(
            o.1, n.1,
            "Op::While: carried value {i} is {} bytes in, {} out",
            n.1, o.1
        );
    }
    assert!(
        cond.inputs.len() <= body.inputs.len(),
        "Op::While: cond reads more values than the loop carries"
    );
    WhilePlan {
        cond,
        body,
        max_iterations,
    }
}

/// Whether the cond's output says go on: nonzero.
fn holds(buf: &[u8], off: usize, dtype: DType) -> bool {
    match dtype {
        DType::Bool | DType::U8 | DType::I8 => buf[off] != 0,
        DType::I32 | DType::U32 => {
            i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) != 0
        }
        DType::I64 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[off..off + 8]);
            i64::from_le_bytes(b) != 0
        }
        DType::F64 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[off..off + 8]);
            f64::from_le_bytes(b) != 0.0
        }
        _ => f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) != 0.0,
    }
}

/// Run the loop against the outer arena `base`: the carried values start at
/// `inputs` (outer byte offsets), the first lands at `output`.
///
/// # Safety
/// `base` must span every offset given, each with its carried value's size.
pub unsafe fn execute_while(base: *mut u8, plan: &WhilePlan, inputs: &[usize], output: usize) {
    let mut carried: Vec<Vec<u8>> = plan
        .body
        .inputs
        .iter()
        .zip(inputs)
        .map(|(&(_, n), &off)| {
            let mut v = vec![0u8; n];
            unsafe { std::ptr::copy_nonoverlapping(base.add(off), v.as_mut_ptr(), n) };
            v
        })
        .collect();
    let mut cond_buf = plan.cond.init.clone();
    let mut body_buf = plan.body.init.clone();
    let mut steps = 0usize;
    loop {
        if plan.max_iterations.is_some_and(|m| steps >= m) {
            break;
        }
        for (&(off, n), v) in plan.cond.inputs.iter().zip(&carried) {
            cond_buf[off..off + n].copy_from_slice(&v[..n]);
        }
        execute_thunks(&plan.cond.schedule, &mut cond_buf);
        if !holds(&cond_buf, plan.cond.outputs[0].0, plan.cond.output_dtype) {
            break;
        }
        for (&(off, n), v) in plan.body.inputs.iter().zip(&carried) {
            body_buf[off..off + n].copy_from_slice(&v[..n]);
        }
        execute_thunks(&plan.body.schedule, &mut body_buf);
        for (&(off, n), v) in plan.body.outputs.iter().zip(carried.iter_mut()) {
            v.copy_from_slice(&body_buf[off..off + n]);
        }
        steps += 1;
    }
    let first = &carried[0];
    unsafe { std::ptr::copy_nonoverlapping(first.as_ptr(), base.add(output), first.len()) };
}

/// The thunk for an `Op::While` node.
pub(crate) fn compile_while(node: &rlx_ir::Node, arena: &crate::arena::Arena) -> super::Thunk {
    let Op::While {
        cond,
        body,
        max_iterations,
    } = &node.op
    else {
        unreachable!()
    };
    let plan = compile_while_plan(cond, body, *max_iterations);
    assert_eq!(
        node.inputs.len(),
        plan.body.inputs.len(),
        "Op::While: {} carried values, the body takes {}",
        node.inputs.len(),
        plan.body.inputs.len()
    );
    super::Thunk::While {
        plan: Arc::new(plan),
        inputs: Arc::new(
            node.inputs
                .iter()
                .map(|&i| super::node_offset(arena, i))
                .collect(),
        ),
        output: super::node_offset(arena, node.id),
    }
}
