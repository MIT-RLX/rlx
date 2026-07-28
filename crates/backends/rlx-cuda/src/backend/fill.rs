// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `fill` — extracted from the `backend` module for navigability (see `mod.rs`).
#![allow(unused_imports)]

use crate::arena::{Arena, plan_f32_uniform};
use crate::device::{
    CUBLASLT_WORKSPACE_BYTES, CUDNN_WORKSPACE_BYTES, cuda_blas, cuda_blas_lt_handle,
    cuda_blas_lt_workspace, cuda_context, cuda_dnn_handle, cuda_dnn_workspace,
};
use crate::host_staging::F32HostSlot;
use crate::kernels::{
    ada_layer_norm_backward_kernel, ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel,
    attention_kernel, attention_row_kernel, batch_elementwise_region_kernel, binary_kernel,
    compare_kernel, concat_kernel, conv_transpose2d_kernel, conv1d_kernel, conv2d_kernel,
    conv3d_kernel, copy_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    dispatch_grid_1d, dispatch_grid_prologue_nchw, elementwise_region_kernel, expand_kernel,
    fused_attn_kernel, fused_binary_unary_kernel, fused_residual_ln_kernel,
    fused_residual_rms_norm_kernel, gated_delta_net_kernel, gated_residual_backward_kernel,
    gated_residual_kernel, gather_axis_kernel, gather_backward_kernel, gather_kernel,
    group_norm_kernel, grouped_matmul_kernel, im2col_kernel, layer_norm2d_kernel, layernorm_kernel,
    matmul_epilogue_kernel, matmul_kernel, matmul_wmma_kernel, maxpool2d_backward_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_opt::rlx_fusion::pass::Pass as _;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use super::*;

impl CudaExecutable {
    pub(crate) fn fill_output_staging_indices(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        indices: &[usize],
    ) -> Result<(), cudarc::driver::DriverError> {
        for &i in indices {
            let id = self.graph.outputs[i];
            let off_f32 = self.arena.offset(id) / 4;
            // Lane count, not element count — a complex output spans 2/4 f32
            // lanes per element (see `arena_lane_count`); reading `num_elements`
            // would truncate the readback to the real parts.
            let lanes = crate::arena::arena_lane_count(&self.graph.node(id).shape);
            debug_assert_eq!(self.output_staging[i].len(), lanes);
            let slot = self.arena.f32_buf().slice(off_f32..off_f32 + lanes);
            self.output_staging[i].dtoh(stream, &slot)?;
        }
        Ok(())
    }

    pub(crate) fn outputs_from_staging_plan(&self, plan: &[usize]) -> Vec<Vec<f32>> {
        if plan.len() == self.graph.outputs.len() {
            return self.outputs_from_staging();
        }
        plan.iter()
            .map(|&i| self.output_staging[i].to_vec())
            .collect()
    }

    pub(crate) fn fill_output_staging(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), cudarc::driver::DriverError> {
        for (i, &id) in self.graph.outputs.iter().enumerate() {
            let off_f32 = self.arena.offset(id) / 4;
            // Lane count, not element count (complex → 2/4 lanes per element).
            let lanes = crate::arena::arena_lane_count(&self.graph.node(id).shape);
            debug_assert_eq!(self.output_staging[i].len(), lanes);
            let slot = self.arena.f32_buf().slice(off_f32..off_f32 + lanes);
            self.output_staging[i].dtoh(stream, &slot)?;
        }
        Ok(())
    }

    pub(crate) fn outputs_from_staging(&self) -> Vec<Vec<f32>> {
        self.output_staging
            .iter()
            .map(F32HostSlot::to_vec)
            .collect()
    }
}
