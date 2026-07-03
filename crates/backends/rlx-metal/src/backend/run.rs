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

//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;
use std::collections::HashMap;
use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};

use super::*;

impl MetalExecutable {
    /// Fastest path: inputs by slot index. Outputs are read directly from
    /// the shared arena buffer (zero-copy on Apple Silicon unified memory).
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        if crate::mps_profile::enabled() {
            crate::mps_profile::reset();
        }
        unsafe {
            let buf_ptr = self.arena.buffer.contents() as *mut u8;
            for (i, &data) in inputs.iter().enumerate() {
                if i < self.input_slots.len() {
                    let (_, off, max_len) = &self.input_slots[i];
                    let len = data.len().min(*max_len);
                    let dst = buf_ptr.add(*off) as *mut f32;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
                }
            }
        }
        self.encode_and_run();
        if crate::mps_profile::enabled() {
            crate::mps_profile::print_summary();
        }
        &self.output_slots
    }


    /// High-throughput batch inference with per-run output snapshots.
    ///
    /// Issues one commit per input set, deferring all waits, then waits
    /// once at the end. Unlike `commit_no_wait`, this allocates a
    /// per-commit output buffer and encodes a blit so each in-flight run's
    /// outputs survive subsequent commits stomping the shared arena.
    ///
    /// Returns outputs in commit order: `out[run_idx][output_idx][element_idx]`.
    pub fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        if input_sets.is_empty() {
            return Vec::new();
        }
        let dev = metal_device().expect("Metal device required");

        // Snapshot output sizes once so per-commit allocation doesn't
        // conflict with the &mut self that encode_commit needs.
        let out_sizes: Vec<usize> = self
            .output_slots
            .iter()
            .map(|(_, len)| (*len).max(1) * 4)
            .collect();

        let mut pending: Vec<(metal::CommandBuffer, Vec<metal::Buffer>)> =
            Vec::with_capacity(input_sets.len());

        for inputs in input_sets {
            // Write inputs into the shared arena. Subsequent commits will
            // overwrite these — fine since each run's compute consumes
            // its inputs before the next commit's writes.
            for &(name, data) in inputs {
                if let Some(&id) = self.input_ids.get(name)
                    && self.arena.has_buffer(id)
                {
                    self.arena.write_from_f32(id, data);
                }
            }
            // Allocate per-commit output buffers. Shared storage so the
            // read-back at the end is just a pointer cast on Apple
            // unified memory (no GPU→CPU copy).
            let dests: Vec<metal::Buffer> =
                out_sizes.iter().map(|&b| dev.alloc_shared(b)).collect();
            if let Some(cmd_buf) = self.encode_commit(false, Some(&dests), None) {
                pending.push((cmd_buf, dests));
            }
        }

        // Single sync at the end. Metal queues are FIFO so waiting on the
        // last buffer guarantees all prior commits have completed.
        if let Some((last, _)) = pending.last() {
            last.wait_until_completed();
        }

        // Read back. Apple unified memory → contents() points at the same
        // bytes the GPU wrote.
        pending
            .into_iter()
            .map(|(_cb, bufs)| {
                bufs.into_iter()
                    .enumerate()
                    .map(|(i, buf)| {
                        let len = self.output_slots[i].1;
                        unsafe {
                            std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec()
                        }
                    })
                    .collect()
            })
            .collect()
    }


    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }


    /// Run and read back only selected graph outputs (e.g. logits-only decode).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        if crate::mps_profile::enabled() {
            crate::mps_profile::reset();
        }
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        self.encode_and_run();
        if !self.gpu_handle_feeds.is_empty() {
            self.propagate_gpu_handle_feeds_in_arena();
            if read_indices.is_none() || rlx_ir::env::flag("RLX_GPU_HANDLE_HOST_MIRROR") {
                self.refresh_gpu_handles_from_outputs();
            }
        }
        if crate::mps_profile::enabled() {
            crate::mps_profile::print_summary();
        }
        let n_out = self.graph.outputs.len();
        let indices: Vec<usize> = match read_indices {
            None => (0..n_out).collect(),
            Some(ix) => ix.to_vec(),
        };
        indices
            .iter()
            .map(|&i| self.read_graph_output_f32(i))
            .collect()
    }


    /// Run with typed host inputs (I64 token ids, F32 style/speed, etc.).
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let mut f32_owned: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, data, dt) in inputs {
            let direct = matches!(
                *dt,
                rlx_ir::DType::F64
                    | rlx_ir::DType::I32
                    | rlx_ir::DType::I64
                    | rlx_ir::DType::U32
                    | rlx_ir::DType::U8
                    | rlx_ir::DType::I8
            );
            if direct {
                if let Some(&id) = self.input_ids.get(*name)
                    && self.arena.has_buffer(id)
                {
                    self.arena.write_bytes(id, data);
                }
            } else if *dt == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                if let Some(&id) = self.input_ids.get(*name)
                    && self.arena.has_buffer(id)
                {
                    self.arena.write_from_f32(id, s);
                }
            } else {
                f32_owned.push((name.to_string(), widen_input_bytes_to_f32(data, *dt)));
            }
        }
        for (name, data) in &f32_owned {
            if let Some(&id) = self.input_ids.get(name.as_str())
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        self.run_read_outputs(&[], None);
        self.output_bytes_per_node()
            .into_iter()
            .zip(self.output_dtypes())
            .collect()
    }


    /// Sequential per-thunk GPU timing (`RLX_METAL_THUNK_PROFILE=1`).
    pub(crate) fn run_thunk_profile(&mut self) {
        use std::time::Instant;
        crate::thunk_profile::reset();
        let n = self.schedule.thunks.len();
        for i in 0..n {
            let name = crate::thunk::thunk_name(&self.schedule.thunks[i]);
            if name == "nop" {
                continue;
            }
            let t0 = Instant::now();
            let _ = self.encode_commit(true, None, Some(i..i + 1));
            crate::thunk_profile::record(name, t0.elapsed());
        }
        crate::thunk_profile::print_summary();
    }


    /// Execute the graph via MPSGraph (set up by lowering at compile time).
    /// All inputs/params are bound to their respective arena offsets; outputs
    /// are written into the arena slots so downstream consumers (run_slots
    /// callers) see them as if a thunk schedule had run.
    pub(crate) fn run_via_mps_graph(&mut self) {
        use std::time::Instant;
        let plan = self.mps_plan.as_ref().expect("plan present");
        let t0 = Instant::now();
        self.dispatch_mps_plan(plan, None, None);
        crate::mps_profile::record("mps_graph:dispatch_full", t0.elapsed());
    }


    /// Interleaved MPS sub-graph + thunk dispatch for Qwen3.5 decode.
    pub(crate) fn run_via_mps_hybrid(&mut self) {
        use std::time::Instant;
        let n = self.mps_hybrid.as_ref().expect("hybrid plan present").len();
        for i in 0..n {
            if let crate::mps_graph_hybrid::HybridStep::Thunks(range) =
                &self.mps_hybrid.as_ref().unwrap()[i]
            {
                let r = range.clone();
                let t0 = Instant::now();
                let _ = self.encode_commit(true, None, Some(r));
                crate::mps_profile::record(format!("hybrid:thunks[{i}]"), t0.elapsed());
                continue;
            }
            if let crate::mps_graph_hybrid::HybridStep::SubGraph {
                plan,
                boundary_parent_ids,
                output_parent_ids,
                ..
            } = &self.mps_hybrid.as_ref().unwrap()[i]
            {
                let t0 = Instant::now();
                self.dispatch_mps_plan(plan, Some(boundary_parent_ids), Some(output_parent_ids));
                crate::mps_profile::record(format!("hybrid:mps_subgraph[{i}]"), t0.elapsed());
            }
        }
    }

}
