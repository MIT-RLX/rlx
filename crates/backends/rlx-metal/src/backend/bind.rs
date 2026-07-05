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
        let Some(plan) = self.mps_plan.as_mut() else {
            return;
        };
        let Some(exec) = plan.executable.as_mut() else {
            return;
        };
        let arena_buf = &self.arena.buffer;

        let mut feed_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut feed_offsets: Vec<usize> = Vec::new();
        let mut feed_shapes: Vec<Vec<usize>> = Vec::new();
        let mut feed_dtypes: Vec<u32> = Vec::new();
        for (name, _t, shape, dt) in &plan.inputs {
            let id = self.input_ids.get(name).expect("input id");
            feed_buffers.push(arena_buf);
            feed_offsets.push(self.arena.byte_offset(*id));
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }
        for (name, _t, shape, dt) in &plan.params {
            let id = self.param_ids.get(name).expect("param id");
            feed_buffers.push(arena_buf);
            feed_offsets.push(self.arena.byte_offset(*id));
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }

        let mut out_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut out_offsets: Vec<usize> = Vec::new();
        let mut out_shapes: Vec<Vec<usize>> = Vec::new();
        let mut out_dtypes: Vec<u32> = Vec::new();
        for (id, _t, shape, dt) in &plan.outputs {
            out_buffers.push(arena_buf);
            out_offsets.push(self.arena.byte_offset(*id));
            out_shapes.push(shape.clone());
            out_dtypes.push(*dt);
        }

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
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_ids.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        // Reuse arena slot when capacity matches (megakernel bucket reinstall).
        if let Some(&id) = self.input_ids.get(name) {
            let cap = *self.arena.element_counts.get(&id).unwrap_or(&0);
            if self.arena.has_buffer(id) && cap == data.len() {
                self.arena.write_from_f32(id, data);
                self.gpu_handles.insert(name.to_string(), Vec::new());
                return true;
            }
        }
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }
}
