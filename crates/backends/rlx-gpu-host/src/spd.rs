// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU host-fallback for Riemannian / SPD-manifold ops.
//!
//! Eigen-decompositions have no GPU kernel on most backends, so these run on
//! the CPU reference (`rlx_cpu::spd`) between GPU segments. The SPD kernels are
//! **F64** while GPU arenas are f32-uniform: we build a one-op CPU graph with
//! the op's real declared dtypes, plan a CPU arena, widen f32 → f64, run the
//! CPU thunk path, and narrow the result back to f32.

use crate::DeviceArena;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// One SPD operand as staged for span-based backends (e.g. wgpu): declared
/// shape + arena **byte** offset of its f32 slot.
#[derive(Clone)]
pub struct SpdInput {
    pub shape: Shape,
    pub byte_off: usize,
}

/// True for the SPD OpKinds this module evaluates on the host.
pub fn is_spd_host(op: &Op) -> bool {
    matches!(
        op,
        Op::BiMap
            | Op::ReEig { .. }
            | Op::LogEig { .. }
            | Op::SpdBatchNorm { .. }
            | Op::SpdKarcherMean { .. }
            | Op::ReEigBackward { .. }
            | Op::LogEigBackward { .. }
            | Op::SpdBatchNormBackwardX { .. }
            | Op::SpdBatchNormBackwardG { .. }
            | Op::SpdKarcherMeanWeighted { .. }
            | Op::SpdLogMap
            | Op::SpdExpMap
            | Op::SpdParallelTransport
            | Op::SpdMatrixFnBatch { .. }
            | Op::SpdLogMapBackward
            | Op::SpdExpMapBackward
            | Op::SpdParallelTransportBackward
            | Op::SpdMatrixFnBatchBackward { .. }
            | Op::Eigh
            | Op::EighBackward
            | Op::EighBatch
            | Op::EighBatchBackward
    )
}

/// Evaluate one SPD op on the CPU reference. `inputs[i]` is `(declared_shape,
/// f32_buffer)` read from the GPU arena for operand `i`; returns the f32
/// output (`out_shape`-sized, row-major).
pub fn eval(op: &Op, out_shape: &Shape, inputs: &[(Shape, Vec<f32>)]) -> Vec<f32> {
    let mut g = Graph::new("gpu_spd_host");
    let ids: Vec<NodeId> = inputs
        .iter()
        .enumerate()
        .map(|(i, (sh, _))| {
            g.append_node(
                Op::Input {
                    name: format!("in{i}"),
                },
                vec![],
                sh.clone(),
                None,
            )
        })
        .collect();
    let out = g.append_node(op.clone(), ids.clone(), out_shape.clone(), None);
    g.set_outputs(vec![out]);

    let plan = rlx_compile::memory::plan_memory_aligned(&g, 64);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);

    for (i, (sh, vals)) in inputs.iter().enumerate() {
        let id = ids[i];
        match sh.dtype() {
            DType::F64 => {
                let slot = arena.slice_mut_f64(id);
                let n = slot.len().min(vals.len());
                for (d, &s) in slot[..n].iter_mut().zip(&vals[..n]) {
                    *d = s as f64;
                }
            }
            _ => {
                let slot = arena.slice_mut(id);
                let n = slot.len().min(vals.len());
                slot[..n].copy_from_slice(&vals[..n]);
            }
        }
    }

    let schedule = rlx_cpu::thunk::compile_thunks(&g, &arena);
    rlx_cpu::thunk::execute_thunks(&schedule, arena.raw_buf_mut());

    let n = out_shape.num_elements().unwrap_or(0);
    match out_shape.dtype() {
        DType::F64 => {
            let slot = arena.slice_f64(out);
            let m = n.min(slot.len());
            slot[..m].iter().map(|&x| x as f32).collect()
        }
        _ => {
            let slot = arena.slice(out);
            let m = n.min(slot.len());
            slot[..m].to_vec()
        }
    }
}

/// Full-arena SPD staging (CUDA/ROCm). `inputs` / `out_off` are **f32 element**
/// offsets into the mirrored arena.
pub fn run_spd<A: DeviceArena>(
    a: &mut A,
    op: &Op,
    out_off: usize,
    out_shape: &Shape,
    inputs: &[(usize, Shape)],
) {
    let n = a.arena_bytes();
    a.sync();
    let mut host = vec![0u8; n];
    a.dtoh(0, &mut host);
    let f32s: &mut [f32] = bytemuck::cast_slice_mut(&mut host);
    let staged: Vec<(Shape, Vec<f32>)> = inputs
        .iter()
        .map(|(off, sh)| {
            let ne = sh.num_elements().unwrap_or(0);
            let end = (*off + ne).min(f32s.len());
            (sh.clone(), f32s[*off..end].to_vec())
        })
        .collect();
    let y = eval(op, out_shape, &staged);
    let end = (out_off + y.len()).min(f32s.len());
    f32s[out_off..end].copy_from_slice(&y[..(end - out_off)]);
    a.htod(0, &host);
}

/// Span-based SPD staging (wgpu). Reads each operand's f32 span, writes the
/// result to `out_byte_off`.
pub fn run_spd_spans<A: DeviceArena>(
    a: &mut A,
    op: &Op,
    inputs: &[SpdInput],
    out_shape: &Shape,
    out_byte_off: usize,
) {
    a.sync();
    let staged: Vec<(Shape, Vec<f32>)> = inputs
        .iter()
        .map(|inp| {
            let ne = inp.shape.num_elements().unwrap_or(0);
            let mut bytes = vec![0u8; ne * 4];
            a.dtoh(inp.byte_off, &mut bytes);
            (inp.shape.clone(), bytemuck::cast_slice(&bytes).to_vec())
        })
        .collect();
    let y = eval(op, out_shape, &staged);
    a.htod(out_byte_off, bytemuck::cast_slice(&y));
}
