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
        let feed_buffers: Vec<&crate::mtl::Buffer> = feed_is_weight
            .iter()
            .map(|&w| {
                if w {
                    weight_buf.expect("weight feed without buffer")
                } else {
                    arena_buf
                }
            })
            .collect();
        let out_buffers: Vec<&crate::mtl::Buffer> = out_offsets.iter().map(|_| arena_buf).collect();

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

    /// ZERO-COPY optimizer step for GPU-resident training. For each trainable
    /// weight — a resident arena `Input` bound via [`bind_gpu_handle`] — whose
    /// gradient sits at output slot `1 + i` (backward outputs are
    /// `[loss, grad0, grad1, …]`), this forms the param `&mut [f32]` and grad
    /// `&[f32]` as ALIASES into the unified-memory arena and calls `step` in
    /// place: no host `Vec`, no D2H/H2D copy. The updated weight stays resident,
    /// so the next forward reads it with no re-upload — killing the classic
    /// GPU→host→optimizer→host→GPU roundtrip. `trainable[i] = (input_name, shape)`.
    ///
    /// Generic over the step fn so rlx-metal needn't depend on rlx-optim — pass
    /// `|name, shape, p, g| optimizer.step(name, shape, p, g)`.
    ///
    /// Soundness: param and grad are distinct graph nodes → disjoint arena
    /// regions, so the two aliasing slices never overlap.
    pub fn optimizer_step_resident<F>(&mut self, trainable: &[(String, Vec<usize>)], mut step: F)
    where
        F: FnMut(&str, &[usize], &mut [f32], &[f32]),
    {
        let slots: Vec<(usize, usize)> = self.output_slots.clone(); // [loss, grad0, …]
        let base = self.arena.buffer.contents() as *mut u8;
        for (i, (name, shape)) in trainable.iter().enumerate() {
            let Some(&pid) = self.input_ids.get(name) else {
                panic!("optimizer_step_resident: trainable `{name}` is not a graph input");
            };
            let p_off = self.arena.byte_offset(pid);
            let p_len: usize = shape.iter().product();
            let (g_off, g_len) = slots[1 + i];
            debug_assert_eq!(p_len, g_len, "grad len must match param `{name}`");
            unsafe {
                let param = std::slice::from_raw_parts_mut(base.add(p_off) as *mut f32, p_len);
                let grad = std::slice::from_raw_parts(base.add(g_off) as *const f32, g_len);
                step(name, shape, param, grad);
            }
        }
    }
}
