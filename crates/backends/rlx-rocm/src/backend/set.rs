// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::{Arena, HalfDtype, plan_f32_uniform};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use super::*;

impl RocmExecutable {
    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every step in the schedule is in the safe set.
    /// See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let off_f32 = self.arena.offset(id) / 4;
            let bytes = data.len() * 4;
            let dst = self.arena.buffer.ptr + (off_f32 as u64) * 4;
            unsafe {
                let _ = (self.ctx.runtime.hip_memcpy_htod)(dst, data.as_ptr() as *const _, bytes);
            }
        }
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            let byte_off = self.arena.offset(id);
            crate::gguf_host::upload_param_bytes(&self.ctx, &mut self.arena.buffer, byte_off, data);
        }
    }

    pub fn set_param_half(&mut self, name: &str, dtype: HalfDtype, bits: &[u16]) {
        let id = match self.param_offsets.get(name) {
            Some(&id) if self.arena.has(id) => id,
            _ => return,
        };
        let f32_off = (self.arena.offset(id) / 4) as u32;
        let off = self
            .arena
            .register_half_param(&self.ctx, id, f32_off, bits.len(), dtype);
        if let Some(buf) = self.arena.half_buffer.as_mut() {
            let bytes = bits.len() * 2;
            let dst = buf.ptr + (off as u64) * 2;
            unsafe {
                let _ = (self.ctx.runtime.hip_memcpy_htod)(dst, bits.as_ptr() as *const _, bytes);
            }
        }
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }
}
