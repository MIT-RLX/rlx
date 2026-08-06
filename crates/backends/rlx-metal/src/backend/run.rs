// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;

/// GPU execution time (seconds) of a *completed* command buffer, from the
/// `GPUStartTime`/`GPUEndTime` ObjC properties the `metal` crate doesn't wrap.
/// This is the on-GPU window only — it excludes the CPU-side encode +
/// `wait_until_completed` cost a wall-clock `Instant` folds into every op, which
/// otherwise over-weights tiny m=1 decode kernels. Caller must have waited for
/// completion (values read 0 otherwise).
fn gpu_cmd_buf_seconds(cb: &metal::CommandBufferRef) -> f64 {
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    // A `foreign_types` Ref is a newtype over the opaque ObjC object, so a
    // pointer to the Ref IS the object pointer.
    let obj = cb as *const metal::CommandBufferRef as *mut Object;
    unsafe {
        let start: f64 = msg_send![obj, GPUStartTime];
        let end: f64 = msg_send![obj, GPUEndTime];
        end - start
    }
}
use std::collections::HashMap;

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
                        // F16 outputs (e.g. an F16-resident KV side-output) are
                        // read as half and widened to the f32 host lane.
                        if self.graph.node(self.graph.outputs[i]).shape.dtype()
                            == rlx_ir::DType::F16
                        {
                            unsafe {
                                std::slice::from_raw_parts(buf.contents() as *const half::f16, len)
                            }
                            .iter()
                            .map(|h| h.to_f32())
                            .collect()
                        } else {
                            unsafe {
                                std::slice::from_raw_parts(buf.contents() as *const f32, len)
                                    .to_vec()
                            }
                        }
                    })
                    .collect()
            })
            .collect()
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    /// Per-node arena dump (mirror of `RLX_CPU_DUMP_NODES`) for cross-backend
    /// divergence bisection. Diff against the CPU dump to find the first
    /// node whose max|x| / nonzero count / NaN count diverges. Meaningful
    /// only with `RLX_ARENA_NO_REUSE=1` so intermediate buffers aren't stomped.
    fn dump_metal_nodes_if_requested(&self) {
        if !rlx_ir::env::flag("RLX_METAL_DUMP_NODES") {
            return;
        }
        let limit = rlx_ir::env::parse_or("RLX_METAL_DUMP_NODES_LIMIT", 4000usize);
        eprintln!(
            "[rlx-metal-dump] per-node max|x| (topo order, limit={limit}); set RLX_ARENA_NO_REUSE=1"
        );
        let mut shown = 0usize;
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if !self.arena.has_buffer(node.id) {
                continue;
            }
            if matches!(
                node.op,
                rlx_ir::Op::Input { .. }
                    | rlx_ir::Op::Param { .. }
                    | rlx_ir::Op::Constant { .. }
                    | rlx_ir::Op::Reshape { .. }
                    | rlx_ir::Op::Cast { .. }
            ) {
                continue;
            }
            if self.arena.dtype(node.id) != rlx_ir::DType::F32 {
                continue;
            }
            let data = self.arena.read_as_f32(node.id);
            if data.is_empty() {
                continue;
            }
            let max = data.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let nz = data.iter().filter(|&&v| v != 0.0).count();
            let nan = data.iter().filter(|&&v| v.is_nan()).count();
            eprintln!(
                "  [{i:>3}] {:?} max={max:.6} nz={nz}/{} nan={nan}",
                node.op,
                data.len()
            );
            shown += 1;
            if shown >= limit {
                break;
            }
        }
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
        self.dump_metal_nodes_if_requested();
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
        let outs: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| self.read_graph_output_f32(i))
            .collect();
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). MPSGraph executes the
        // graph opaquely, so we can't hook per-op here — scan the outputs and
        // point provenance at the offending output node. For internal
        // localization, replay the same graph on the CPU backend.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("metal");
        if scanner.enabled() {
            for (&i, buf) in indices.iter().zip(&outs) {
                scanner.check(&self.graph, self.graph.outputs[i], buf, &[]);
            }
        }
        outs
    }

    /// Run with typed host inputs (I64 token ids, F32 style/speed, etc.).
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let mut f32_owned: Vec<(String, Vec<f32>)> = Vec::new();
        for (name, data, dt) in inputs {
            // Integer/bool inputs are widened to f32 to match the arena: compile
            // rewrites their consumer nodes to F32 (see
            // `widen_integer_activations_to_f32`), so the input slots are f32 too.
            // Writing raw i64/i32 bytes would be read back as f32 garbage (e.g. a
            // gather index or the VITS `arange < lengths` sequence mask).
            let widen = matches!(
                *dt,
                rlx_ir::DType::I32 | rlx_ir::DType::I64 | rlx_ir::DType::U32 | rlx_ir::DType::Bool
            );
            // F64 / U8 / I8 keep their native byte width (packed weights, quant).
            let direct = matches!(
                *dt,
                rlx_ir::DType::F64 | rlx_ir::DType::U8 | rlx_ir::DType::I8
            );
            if widen {
                f32_owned.push((name.to_string(), widen_input_bytes_to_f32(data, *dt)));
            } else if direct {
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

    /// Sequential per-thunk timing (`RLX_METAL_THUNK_PROFILE=1`).
    ///
    /// Times each op by its **GPU execution window** (`GPUEndTime -
    /// GPUStartTime`) rather than CPU `Instant::now()`. The CPU clock
    /// double-counts the per-op encode + `wait_until_completed` overhead (tens
    /// of µs) onto every tiny m=1 decode op, which massively over-weights them
    /// and — on the very first op — folds in one-time mmap page-ins (the ~600 ms
    /// "gather" artifact). A warm-up forward runs first so weights / the embed
    /// table are resident before any op is measured.
    pub(crate) fn run_thunk_profile(&mut self) {
        crate::thunk_profile::reset();
        let n = self.schedule.thunks.len();
        // Warm: make every arena region + weight page resident so the timed
        // per-op runs measure steady-state GPU work, not first-touch faults.
        let _ = self.encode_commit(true, None, None);
        for i in 0..n {
            let name = crate::thunk::thunk_name(&self.schedule.thunks[i]);
            if name == "nop" {
                continue;
            }
            // wait=false hands back the committed buffer (wait=true consumes it
            // and returns None); wait here, then read its GPU window.
            if let Some(cb) = self.encode_commit(false, None, Some(i..i + 1)) {
                cb.wait_until_completed();
                let secs = gpu_cmd_buf_seconds(&cb).max(0.0);
                crate::thunk_profile::record(name, std::time::Duration::from_secs_f64(secs));
            }
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
