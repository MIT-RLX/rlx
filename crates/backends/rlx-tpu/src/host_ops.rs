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

//! Host-segment execution for LAPACK / specialty ops with no clean HLO path
//! (CPU kernels between HLO segments).

use rlx_ir::{Graph, NodeId, Op};

use crate::lower::scaled_fp8_hlo_ok;
use crate::splat_host::HostTensors;

/// Keep in sync with [`crate::segment::COLLECTIVE_OPS`] (avoid a module cycle).
const COLLECTIVE_CUSTOM: &[&str] = &[
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    "collective.copy_to_parallel",
    "collective.reduce_from_parallel",
];

/// Ops that run on the host between HLO segments (CPU kernels).
///
/// Dedicated specializations (DenseSolve) keep hand-written paths; everything
/// else in this set goes through [`rlx_cpu::thunk::run_host_op_node_f32`].
///
/// Not host: `Reverse` / `ResizeNearest2x` (native HLO), fused DiT reverse
/// (composed HLO), norm/QAT/Rope/Cumsum/Gather/Conv2d/MaxPool2d training bwd
/// (composed HLO), `AttentionBackward` (expanded in `prepare_graph_for_hlo`),
/// `FusedConvBiasAct` / `PartitionedConv` / `GatedDeltaNet` (unfused before
/// HLO), `AxialRope2d` / `Im2Col` / `ConvTranspose*` (composed HLO),
/// PerTensor 8-bit `ScaledMatMul` / `ScaledQuantScale` / `ScaledDequantize`
/// (F32 LUT compose), Gaussian splat render/backward (dedicated splat
/// segments), collective `Custom` (collective segments).
pub fn is_host_op(op: &Op) -> bool {
    match op {
        Op::DenseSolve
        | Op::BatchedDenseSolve
        // Cholesky / TriangularSolve / Det / LogDet → generic CPU host-eval
        // path (`run_host_op_node_f32`), no dedicated specialization needed.
        | Op::Cholesky
        | Op::TriangularSolve { .. }
        | Op::Det
        | Op::LogDet
        // Sort / ArgSort: stable strided sort on CPU, no HLO primitive.
        | Op::Sort { .. } | Op::Svd { .. } | Op::Qr { .. }
        | Op::ArgSort { .. }
        // FftButterflyStage is a ternary-pruned stage, not Op::Fft.
        | Op::Scan { .. }
        | Op::ScanBackward { .. }
        | Op::ScanBackwardXs { .. }
        | Op::FftButterflyStage { .. }
        | Op::CustomFn { .. }
        // ScaledQuantize needs bit-level encode (no F8 prim in our HLO).
        | Op::ScaledQuantize { .. }
        // SPD / Eigh (f32 BiMap/ReEig/LogEig/SpdBatchNorm are rewritten by
        // LowerSpectral before planning; f64 + bwd + Eigh stay here).
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
        | Op::EighBatchBackward
        // Splat prepare / rasterize (render/backward have dedicated segments).
        | Op::GaussianSplatPrepare { .. }
        | Op::GaussianSplatRasterize { .. } => true,
        Op::ScaledMatMul {
            lhs_format,
            rhs_format,
            scale_layout,
            ..
        } => {
            !scaled_fp8_hlo_ok(*lhs_format, *scale_layout)
                || !scaled_fp8_hlo_ok(*rhs_format, *scale_layout)
        }
        Op::ScaledQuantScale {
            format,
            scale_layout,
        }
        | Op::ScaledDequantize {
            format,
            scale_layout,
        } => !scaled_fp8_hlo_ok(*format, *scale_layout),
        Op::Custom { name, .. } => !COLLECTIVE_CUSTOM.contains(&name.as_str()),
        _ => false,
    }
}

/// Execute one host-segment op: DenseSolve specialization, else CPU
/// [`rlx_cpu::thunk::run_host_op_node_f32`]. Panics if `node` is not a host op.
pub fn run_host_op(graph: &Graph, node: NodeId, env: &mut HostTensors) {
    let n = graph.node(node);
    match &n.op {
        Op::DenseSolve => run_dense_solve(graph, node, env, false),
        Op::BatchedDenseSolve => run_dense_solve(graph, node, env, true),
        other if is_host_op(other) => {
            let out = rlx_cpu::thunk::run_host_op_node_f32(graph, n, |id| {
                env.get(&id)
                    .unwrap_or_else(|| panic!("rlx-tpu host: missing tensor for {id:?}"))
                    .clone()
            });
            env.insert(node, out);
        }
        other => panic!("rlx-tpu host_ops: unexpected op {other:?}"),
    }
}

fn get(env: &HostTensors, id: NodeId) -> &[f32] {
    env.get(&id)
        .unwrap_or_else(|| panic!("rlx-tpu host: missing tensor for {id:?}"))
        .as_slice()
}

fn run_dense_solve(graph: &Graph, node: NodeId, env: &mut HostTensors, batched: bool) {
    let n = graph.node(node);
    let a = get(env, n.inputs[0]).to_vec();
    let mut b = get(env, n.inputs[1]).to_vec();
    let a_shape = graph.node(n.inputs[0]).shape.dims();
    let b_shape = graph.node(n.inputs[1]).shape.dims();

    if batched {
        let batch = a_shape[0].unwrap_static();
        let n_dim = a_shape[1].unwrap_static();
        assert_eq!(a_shape[2].unwrap_static(), n_dim);
        let nrhs = if b_shape.len() == 2 {
            1
        } else {
            b_shape[2].unwrap_static()
        };
        let a_stride = n_dim * n_dim;
        let b_stride = n_dim * nrhs;
        for bi in 0..batch {
            let mut a_slice = a[bi * a_stride..(bi + 1) * a_stride].to_vec();
            let b_off = bi * b_stride;
            let info =
                rlx_cpu::blas::sgesv(&mut a_slice, &mut b[b_off..b_off + b_stride], n_dim, nrhs);
            assert_eq!(info, 0, "rlx-tpu BatchedDenseSolve: sgesv info={info}");
        }
    } else {
        let n_dim = a_shape[0].unwrap_static();
        assert_eq!(a_shape[1].unwrap_static(), n_dim);
        let nrhs = if b_shape.len() == 1 {
            1
        } else {
            b_shape[1].unwrap_static()
        };
        let mut a_mut = a;
        let info = rlx_cpu::blas::sgesv(&mut a_mut, &mut b, n_dim, nrhs);
        assert_eq!(info, 0, "rlx-tpu DenseSolve: sgesv info={info}");
    }
    env.insert(node, b);
}
