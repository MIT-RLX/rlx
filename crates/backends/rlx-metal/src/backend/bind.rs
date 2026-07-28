// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `bind` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;
use std::collections::HashMap;

use super::*;

impl MetalExecutable {
    pub(crate) fn bind_mps_executable_to_arena(&mut self) {
        if self.mps_plan.is_none() {
            return;
        }
        // Collect bind lists before taking `&mut` on the executable (avoids
        // overlapping borrows with `param_buffer` / arena).
        let (
            feed_offsets,
            feed_shapes,
            feed_dtypes,
            feed_is_weight,
            out_offsets,
            out_shapes,
            out_dtypes,
        ) = {
            let plan = self.mps_plan.as_ref().unwrap();
            let mut feed_offsets = Vec::new();
            let mut feed_shapes = Vec::new();
            let mut feed_dtypes = Vec::new();
            let mut feed_is_weight = Vec::new();
            for (name, _t, shape, dt) in &plan.inputs {
                let id = self.input_ids.get(name).expect("input id");
                feed_offsets.push(self.arena.byte_offset(*id));
                feed_shapes.push(shape.clone());
                feed_dtypes.push(*dt);
                feed_is_weight.push(false);
            }
            for (name, _t, shape, dt) in &plan.params {
                let id = *self.param_ids.get(name).expect("param id");
                feed_offsets.push(self.param_byte_offset(id));
                feed_shapes.push(shape.clone());
                feed_dtypes.push(*dt);
                feed_is_weight.push(self.weight_slots.contains_key(&id));
            }
            let mut out_offsets = Vec::new();
            let mut out_shapes = Vec::new();
            let mut out_dtypes = Vec::new();
            for (id, _t, shape, dt) in &plan.outputs {
                out_offsets.push(self.arena.byte_offset(*id));
                out_shapes.push(shape.clone());
                out_dtypes.push(*dt);
            }
            (
                feed_offsets,
                feed_shapes,
                feed_dtypes,
                feed_is_weight,
                out_offsets,
                out_shapes,
                out_dtypes,
            )
        };

        let arena_buf = &self.arena.buffer;
        let weight_buf = self.weight_buffer.as_ref();
        let feed_buffers: Vec<&metal::Buffer> = feed_is_weight
            .iter()
            .map(|&w| {
                if w {
                    weight_buf.expect("weight feed without buffer")
                } else {
                    arena_buf
                }
            })
            .collect();
        let out_buffers: Vec<&metal::Buffer> = out_offsets.iter().map(|_| arena_buf).collect();

        let Some(plan) = self.mps_plan.as_mut() else {
            return;
        };
        let Some(exec) = plan.executable.as_mut() else {
            return;
        };
        exec.bind_arena(
            &feed_buffers,
            &feed_offsets,
            &feed_shapes,
            &feed_dtypes,
            &out_buffers,
            &out_offsets,
            &out_shapes,
            &out_dtypes,
        );
    }

    /// Persistent input buffer for KV-cache style graphs (unified memory).
    ///
    /// Writes `data` into the arena slot once and marks the handle **resident**
    /// so subsequent `run` / `run_read_outputs` calls skip host→arena copies.
    /// [`feed_kv_row`] then appends new tokens with an in-arena memcpy only.
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        let Some(&id) = self.input_ids.get(name) else {
            return false;
        };
        if !self.arena.has_buffer(id) {
            // Keep host mirror until the arena slot exists (rare).
            self.gpu_handle_resident.remove(name);
            self.gpu_handles.insert(name.to_string(), data.to_vec());
            return true;
        }
        let cap = *self.arena.element_counts.get(&id).unwrap_or(&0);
        if cap != data.len() {
            // Length mismatch — fall back to host mirror (bucket reinstall).
            self.gpu_handle_resident.remove(name);
            self.gpu_handles.insert(name.to_string(), data.to_vec());
            return true;
        }
        self.arena.write_from_f32(id, data);
        self.gpu_handle_resident.insert(name.to_string());
        // Empty host mirror: arena is the source of truth.
        self.gpu_handles.insert(name.to_string(), Vec::new());
        true
    }
}
