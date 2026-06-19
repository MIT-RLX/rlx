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

//! RLX custom-op registration for UMAP k-NN.

use std::sync::Arc;

use rlx_ir::{DType, OpExtension, Shape, register_op};

#[cfg(feature = "autodiff")]
use rlx_ir::infer::GraphExt;
#[cfg(feature = "autodiff")]
use rlx_ir::{Node, NodeId, Op, VjpContext};

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

use crate::knn_attrs::KnnAttrs;
#[cfg(feature = "autodiff")]
use rlx_cpu::umap_knn::knn_backward_pairwise;
use rlx_cpu::umap_knn::knn_forward_packed;

/// `packed = knn(pairwise)` — see crate README.
pub const UMAP_KNN: &str = "umap.knn";

/// `d_pairwise = knn_backward(pairwise, d_dist)` (feature `autodiff`).
pub const UMAP_KNN_BWD: &str = "umap.knn_backward";

/// Register IR extensions and CPU kernels. Call once per process before compiling graphs.
pub fn register_umap_ops() {
    register_op(Arc::new(KnnForwardExt));
    register_cpu_kernel(Arc::new(KnnForwardCpu));
    #[cfg(feature = "autodiff")]
    {
        register_op(Arc::new(KnnBackwardExt));
        register_cpu_kernel(Arc::new(KnnBackwardCpu));
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    crate::metal_kernels::register_metal_kernels();
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    crate::mlx_kernels::register_mlx_kernels();
}

struct KnnForwardExt;

impl OpExtension for KnnForwardExt {
    fn name(&self) -> &str {
        UMAP_KNN
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let n = inputs[0].dim(0).unwrap_static();
        let n2 = inputs[0].dim(1).unwrap_static();
        assert_eq!(n, n2, "umap.knn: pairwise must be square [n, n]");
        let k = KnnAttrs::decode(attrs).expect("umap.knn attrs").k as usize;
        Shape::new(&[n, 2 * k], DType::F32)
    }

    #[cfg(feature = "autodiff")]
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let k = KnnAttrs::decode(&attrs).expect("umap.knn attrs").k as usize;
        let pairwise = ctx.fwd_map[&node.inputs[0]];
        let d_dist = ctx.bwd.narrow_(ctx.upstream, 1, k, k);
        let grad_pw = ctx
            .bwd
            .custom_op(UMAP_KNN_BWD, attrs, vec![pairwise, d_dist]);
        vec![(0, grad_pw)]
    }
}

struct KnnForwardCpu;

impl CpuKernel for KnnForwardCpu {
    fn name(&self) -> &str {
        UMAP_KNN
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let pairwise = inputs[0].expect_f32("umap.knn pairwise")?;
        let out = output.expect_f32_mut("umap.knn packed")?;
        let shape = inputs[0].shape();
        let n = shape.dim(0).unwrap_static();
        let n2 = shape.dim(1).unwrap_static();
        if n != n2 {
            return Err(format!("umap.knn: expected square [n, n], got [{n}, {n2}]"));
        }
        let k = KnnAttrs::decode(attrs)?.k as usize;
        if out.len() != n * 2 * k {
            return Err(format!(
                "umap.knn: output len {} != n*2*k = {}",
                out.len(),
                n * 2 * k
            ));
        }
        knn_forward_packed(pairwise, n, k, out);
        Ok(())
    }
}

#[cfg(feature = "autodiff")]
struct KnnBackwardExt;

#[cfg(feature = "autodiff")]
impl OpExtension for KnnBackwardExt {
    fn name(&self) -> &str {
        UMAP_KNN_BWD
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let n = inputs[0].dim(0).unwrap_static();
        let n2 = inputs[0].dim(1).unwrap_static();
        assert_eq!(n, n2);
        let k = KnnAttrs::decode(attrs).expect("umap.knn_backward attrs").k as usize;
        let nr = inputs[1].dim(0).unwrap_static();
        let kc = inputs[1].dim(1).unwrap_static();
        assert_eq!(nr, n, "umap.knn_backward: d_dist rows must match n");
        assert_eq!(kc, k, "umap.knn_backward: d_dist cols must match k");
        Shape::new(&[n, n], DType::F32)
    }
}

#[cfg(feature = "autodiff")]
struct KnnBackwardCpu;

#[cfg(feature = "autodiff")]
impl CpuKernel for KnnBackwardCpu {
    fn name(&self) -> &str {
        UMAP_KNN_BWD
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let pairwise = inputs[0].expect_f32("umap.knn_backward pairwise")?;
        let d_dist = inputs[1].expect_f32("umap.knn_backward d_dist")?;
        let out = output.expect_f32_mut("umap.knn_backward d_pairwise")?;
        let n = inputs[0].shape().dim(0).unwrap_static();
        let n2 = inputs[0].shape().dim(1).unwrap_static();
        if n != n2 {
            return Err(format!("umap.knn_backward: pairwise must be [{n}, {n}]"));
        }
        let k = KnnAttrs::decode(attrs)?.k as usize;
        if d_dist.len() != n * k {
            return Err(format!(
                "umap.knn_backward: d_dist len {} != n*k = {}",
                d_dist.len(),
                n * k
            ));
        }
        if out.len() != n * n {
            return Err(format!(
                "umap.knn_backward: output len {} != n*n = {}",
                out.len(),
                n * n
            ));
        }
        knn_backward_pairwise(pairwise, d_dist, n, k, out);
        Ok(())
    }
}
