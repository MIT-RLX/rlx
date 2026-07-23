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

//! Typed CPU host-eval for MLX ops that lack a native MLX primitive.
//! Builds a one-op CPU graph with the node's declared dtypes (unlike
//! [`super::helpers::host_eval_op_f32`], which forces an f32-uniform arena).

use std::collections::HashMap;

use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError};
use crate::ops;

use super::helpers::{lookup, mlx_c64_leaf_from_bytes, mlx_c128_leaf_from_bytes};

/// Host-evaluate `node` on the CPU reference, preserving declared dtypes.
pub(crate) fn host_eval_op_typed(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<NodeId, Array>,
) -> Result<Array, MlxError> {
    let mut g = Graph::new("mlx_host_eval");
    let mut ids = Vec::with_capacity(node.inputs.len());
    let mut staged: Vec<Vec<u8>> = Vec::with_capacity(node.inputs.len());
    for (i, &in_id) in node.inputs.iter().enumerate() {
        let sh = graph.node(in_id).shape.clone();
        let arr = ops::contiguous(lookup(env, in_id)?)?;
        let bytes = arr.to_bytes()?;
        staged.push(bytes);
        ids.push(g.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out = g.append_node(node.op.clone(), ids.clone(), node.shape.clone(), None);
    g.set_outputs(vec![out]);

    let plan = rlx_opt::memory::plan_memory_aligned(&g, 64);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    for (i, bytes) in staged.iter().enumerate() {
        let id = ids[i];
        let off = arena.byte_offset(id);
        let nbytes = graph_node_nbytes(&g, id);
        let buf = arena.raw_buf_mut();
        let n = nbytes.min(bytes.len()).min(buf.len().saturating_sub(off));
        buf[off..off + n].copy_from_slice(&bytes[..n]);
    }
    let schedule = rlx_cpu::thunk::compile_thunks(&g, &arena);
    rlx_cpu::thunk::execute_thunks(&schedule, arena.raw_buf_mut());

    let off = arena.byte_offset(out);
    let nbytes = graph_node_nbytes(&g, out);
    let out_bytes = arena.raw_buf()[off..off + nbytes].to_vec();
    let out_dims: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    array_from_host_bytes(&out_bytes, &out_dims, node.shape.dtype())
}

fn graph_node_nbytes(g: &Graph, id: NodeId) -> usize {
    g.node(id).shape.size_bytes().unwrap_or(0)
}

fn array_from_host_bytes(bytes: &[u8], dims: &[usize], dtype: DType) -> Result<Array, MlxError> {
    match dtype {
        DType::C64 => mlx_c64_leaf_from_bytes(bytes),
        DType::C128 => mlx_c128_leaf_from_bytes(bytes),
        DType::F32 | DType::F16 | DType::BF16 => {
            let n = bytes.len() / 4;
            let mut vals = Vec::with_capacity(n);
            for chunk in bytes.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            // F16/BF16 leaves are stored as f32 on MLX in this backend.
            Array::from_f32_slice(&vals, dims, DType::F32)
        }
        _ => Array::from_bytes(bytes, dims, dtype),
    }
}

/// True when the op should use typed CPU host-eval on MLX (no native lower).
///
/// PerTensor 8-bit `ScaledMatMul` / `ScaledQuantScale` / `ScaledDequantize`
/// compose to F32 MLX (LUT decode + scale), mirroring TPU HLO — they are
/// **not** typed-host. `ScaledQuantize` stays host (bit-level encode).
pub(crate) fn is_mlx_typed_host_op(op: &Op) -> bool {
    match op {
        Op::ScaledQuantize { .. } => true,
        Op::ScaledMatMul {
            lhs_format,
            rhs_format,
            scale_layout,
            ..
        } => {
            !super::helpers::scaled_fp8_mlx_ok(*lhs_format, *scale_layout)
                || !super::helpers::scaled_fp8_mlx_ok(*rhs_format, *scale_layout)
        }
        Op::ScaledQuantScale {
            format,
            scale_layout,
        }
        | Op::ScaledDequantize {
            format,
            scale_layout,
        } => !super::helpers::scaled_fp8_mlx_ok(*format, *scale_layout),
        Op::GaussianSplatPrepare { .. }
        | Op::GaussianSplatRasterize { .. }
        | Op::CustomFn { .. }
        | Op::BiMap
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
        | Op::EighBatchBackward => true,
        _ => false,
    }
}
