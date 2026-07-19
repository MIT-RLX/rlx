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
//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).
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
    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every step in the schedule passes
    /// `Step::safe_for_active_extent`. Bypasses captured CUDA Graph
    /// (recorded at full extent) when active. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let off_f32 = self.arena.offset(id) / 4;
            let stream = self.ctx.default_stream();
            let mut slot = self
                .arena
                .f32_buf_mut()
                .slice_mut(off_f32..off_f32 + data.len());
            stream
                .memcpy_htod(data, &mut slot)
                .expect("rlx-cuda: param upload failed");
        }
    }

    /// Upload packed U8/I8 GGUF weights into the param slot (byte offset).
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let byte_off = self.arena.offset(id);
            let stream = self.ctx.default_stream();
            crate::gguf_host::upload_param_bytes(&stream, self.arena.f32_buf_mut(), byte_off, data);
        }
    }

    /// Upload a param as packed half-precision bits (`u16` per element).
    /// Caller passes the raw IEEE-754 binary16 (`F16`) or BFloat16
    /// (`Bf16`) bit pattern; the backend stores it in the half-arena
    /// side-buffer and skips the f32 slot entirely. Use cases:
    /// 2× weight-memory savings for inference, plus Tensor Core matmul
    /// via `cublasGemmEx` when both A and B (or just B) are stored
    /// half-precision.
    ///
    /// When the same `name` is also `set_param`'d as f32, the
    /// half-arena entry takes precedence in the matmul dispatch. Use
    /// only one of the two for any given param.
    pub fn set_param_half(&mut self, name: &str, dtype: crate::arena::HalfDtype, bits: &[u16]) {
        let id = match self.param_offsets.get(name) {
            Some(&id) if self.arena.has(id) => id,
            _ => return,
        };
        let f32_off = (self.arena.offset(id) / 4) as u32;
        let off = self
            .arena
            .register_half_param(&self.ctx, id, f32_off, bits.len(), dtype);
        let stream = self.ctx.default_stream();
        if let Some(buf) = self.arena.half_buffer.as_mut() {
            let mut slot = buf.slice_mut(off..off + bits.len());
            stream
                .memcpy_htod(bits, &mut slot)
                .expect("rlx-cuda: half-param upload failed");
        }
    }
}
