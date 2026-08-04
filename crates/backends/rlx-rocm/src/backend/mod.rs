// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `RocmExecutable` — sister to `rlx-cuda::CudaExecutable`.
//!
//! Full IR walk, memory plan, Step emission, and HIP kernel dispatch
//! mirroring `rlx-cuda` with `HipBuffer` / hipBLAS / MIOpen types.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::{Graph, NodeId};

use std::sync::Mutex;

use crate::arena::Arena;
use crate::device::RocmContext;
use crate::hip::HipBuffer;
use crate::hipblas::HipblasContext;
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;

const MIOPEN_WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const HIPBLASLT_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

mod helpers;
mod step;

pub(crate) use helpers::*;
pub(crate) use step::*;

mod compile;
mod fill;
mod output;
mod run;
mod set;

// ── Modes ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    #[default]
    Jit,
    Aot,
}

/// `Stream` (default single-stream dispatch). `Graph` captures the
/// schedule into a hipGraph on first run and replays it on subsequent
/// runs — eliminates per-launch dispatch overhead. `Eager` is a
/// one-shot compile + run + drop helper. `MultiStream(n)` allocates a
/// pool of `n` streams and assigns each Step based on data
/// dependencies (same dep-aware scheduler as rlx-cuda).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    Stream,
    Graph,
    Eager,
    MultiStream(usize),
}

// ── RocmExecutable ────────────────────────────────────────────────────

pub struct RocmExecutable {
    pub(crate) ctx: Arc<RocmContext>,
    /// hipBLAS handle bound to the same default stream as `ctx`. Used
    /// for plain matmul (no fused bias/activation); falls back to the
    /// custom kernel when libhipblas isn't available.
    pub(crate) blas: Option<Arc<Mutex<HipblasContext>>>,
    /// hipBLASLt handle for fused matmul + bias + relu/gelu. Falls
    /// back to plain sgemm + matmul_epilogue.cu when unavailable.
    pub(crate) blas_lt: Option<Arc<HipblasLtContext>>,
    /// 4 MiB scratch workspace for hipBLASLt heuristic-selected algos.
    pub(crate) blas_lt_workspace: Option<HipBuffer<u8>>,
    /// MIOpen handle for conv2d. Falls back to the custom direct-conv
    /// kernel when libMIOpen isn't available.
    pub(crate) dnn: Option<Arc<MiopenContext>>,
    /// Scratch workspace for MIOpen-selected conv algorithms (32 MiB
    /// — same shape as rlx-cuda's cuDNN workspace).
    pub(crate) dnn_workspace: Option<HipBuffer<u8>>,
    /// Byte offset in the f32 arena for GGUF dequant scratch (0 = none).
    pub(crate) dequant_scratch_off: usize,
    pub(crate) graph: Graph,
    pub(crate) arena: Arena,
    pub(crate) schedule: Vec<Step>,
    pub(crate) input_offsets: HashMap<String, NodeId>,
    pub(crate) param_offsets: HashMap<String, NodeId>,
    pub(crate) meta_buffers: Vec<HipBuffer<u32>>,
    pub(crate) exec_mode: ExecMode,
    pub(crate) half_act_scratch: Option<HipBuffer<u16>>,
    /// Captured hipGraphExec from `ExecMode::Graph`'s first-run
    /// capture; replayed via `hipGraphLaunch` on subsequent runs.
    pub(crate) captured_graph: Option<crate::hip::HipGraphExec>,
    /// Stream pool for `ExecMode::MultiStream(n)`. Empty otherwise.
    /// Each entry was created via `hipStreamCreate` and gets dropped
    /// when this struct is dropped.
    pub(crate) streams: Vec<crate::hip::HipStream>,
    /// Active-extent hint (PLAN L1). Mirrors rlx-cuda — bypasses
    /// hipGraph capture (recorded at full extent) when set + every
    /// step in the safe set.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Pinned or pageable host slots for output download.
    pub(crate) output_staging: Vec<F32HostSlot>,
    /// Pinned input staging when `RLX_ROCM_PINNED_IO=1` or graph mode.
    pub(crate) input_staging: HashMap<String, F32HostSlot>,
    /// Persistent KV inputs (host mirror + device upload each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Resident-KV row feeds: `output_index` per `past_k/v` handle. `feed_kv_row`
    /// copies one row (the new token) from that output into the handle in-arena.
    kv_row_feeds: HashMap<String, usize>,
    /// Handles updated in place in the arena by `feed_kv_row` (device D2D) — they
    /// must NOT be re-uploaded from the (now-stale) host mirror in `stage_gpu_handle_inputs`.
    gpu_handle_resident: std::collections::HashSet<String>,
    /// When set, only these output indices (+ feed outputs) are read back from device.
    pending_read_indices: Option<Vec<usize>>,
    /// Graph input names in declaration order (parallel to `input_slots`).
    input_slot_names: Vec<String>,
    /// Graph inputs in declaration order: `(arena_byte_offset, max_f32_elems)`.
    input_slots: Vec<(usize, usize)>,
    /// Host readback layout: `(byte_offset_in_host_arena, f32_elems)` per graph output.
    output_slots: Vec<(usize, usize)>,
    /// Pageable host mirror for `run_slots` / `arena_ptr` (not the GPU arena).
    host_arena: Vec<f32>,
    /// Runtime-mutable RNG policy for in-graph random ops.
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

impl RocmExecutable {
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        // A captured hipGraph bakes the Philox seed / backend into kernel args,
        // so a policy change must drop (and destroy) the capture: the next run
        // re-captures or falls back to eager under the new policy, matching
        // eager dispatch which re-reads the policy every run.
        let changed = *self.rng.read().expect("rng lock") != rng;
        if changed {
            if let Some(g) = self.captured_graph.take() {
                // SAFETY: `g` is a live hipGraphExec captured by this
                // executable; destroy it before dropping the handle (mirrors
                // `Drop`).
                unsafe {
                    let _ = (self.ctx.runtime.hip_graph_exec_destroy)(g);
                }
            }
        }
        *self.rng.write().expect("rng lock") = rng;
    }

    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }
}

impl Drop for RocmExecutable {
    fn drop(&mut self) {
        unsafe {
            if let Some(g) = self.captured_graph.take() {
                let _ = (self.ctx.runtime.hip_graph_exec_destroy)(g);
            }
            for s in self.streams.drain(..) {
                let _ = (self.ctx.runtime.hip_stream_destroy)(s);
            }
        }
    }
}

impl RocmExecutable {
    /// One-shot eager run.
    pub fn eager(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut exec = Self::compile_with(graph, CompileMode::Jit, ExecMode::Eager);
        exec.run(inputs)
    }

    /// Host buffer base for reading outputs after [`Self::run_slots`].
    /// Offsets in the returned slot pairs are **byte** offsets into this buffer.
    pub fn arena_ptr(&self) -> *const u8 {
        self.host_arena.as_ptr() as *const u8
    }

    pub(crate) fn upload_slot_inputs(&mut self, inputs: &[&[f32]]) {
        let rt = &self.ctx.runtime;
        let arena_base = self.arena.buffer.ptr;
        for (i, data) in inputs.iter().enumerate() {
            let Some(&(byte_off, max_elems)) = self.input_slots.get(i) else {
                break;
            };
            let off_f32 = byte_off / 4;
            let len = data.len().min(max_elems);
            if len == 0 {
                continue;
            }
            let dst = arena_base + (off_f32 as u64) * 4;
            if let Some(name) = self.input_slot_names.get(i) {
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(&data[..len]);
                    host.htod(rt, dst, len)
                        .expect("rlx-rocm: pinned slot input upload failed");
                    continue;
                }
            }
            unsafe {
                let _ = (rt.hip_memcpy_htod)(
                    dst,
                    data.as_ptr() as *const _,
                    len * std::mem::size_of::<f32>(),
                );
            }
        }
    }

    pub(crate) fn pack_host_arena(&mut self) {
        let plan = self.readback_plan();
        for &i in &plan {
            if i >= self.output_staging.len() || i >= self.output_slots.len() {
                continue;
            }
            let (byte_off, n) = self.output_slots[i];
            if n == 0 {
                continue;
            }
            let start = byte_off / 4;
            let end = start + n;
            if end <= self.host_arena.len() {
                self.output_staging[i].copy_into(&mut self.host_arena[start..end]);
            }
        }
    }

    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule.iter().all(|s| s.safe_for_active_extent())
    }

    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        // Rebinding re-seeds the handle from host → it is no longer resident-only.
        self.gpu_handle_resident.remove(name);
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    /// Register a resident-KV row feed (see [`Self::feed_kv_row`]).
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Device-side D2D copy of one KV row (the new token, output row `src_row`) from
    /// each registered decode output into its resident `past_k/v` handle, in the
    /// arena — no host round-trip. Marks the handle resident so
    /// `stage_gpu_handle_inputs` won't re-upload a stale host mirror over it. This
    /// is the ROCm twin of rlx-cuda's `feed_kv_row`.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        if row_elems == 0 {
            return;
        }
        use crate::kernels::*;
        let stream = self.ctx.default_stream;
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in &feeds {
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if *out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[*out_idx];
            if in_id == out_id || !self.arena.has(in_id) || !self.arena.has(out_id) {
                continue;
            }
            let base_out = self.arena.offset(out_id) / 4;
            let base_in = self.arena.offset(in_id) / 4;
            let rel_src = src_row * row_elems;
            let rel_dst = dst_row * row_elems;
            let cap_in = self.arena.len_of(in_id) / 4;
            let cap_out = self.arena.len_of(out_id) / 4;
            if rel_src + row_elems > cap_out || rel_dst + row_elems > cap_in {
                continue;
            }
            let src_off = (base_out + rel_src) as u32;
            let dst_off = (base_in + rel_dst) as u32;
            let n = row_elems as u32;
            let kernel = copy_kernel(&self.ctx);
            let mut arena_ptr = self.arena.buffer.ptr;
            let (grid, block) = dispatch_grid_1d(n, 256);
            crate::launch_kernel!(
                kernel,
                stream,
                (grid, 1, 1),
                (block, 1, 1),
                [&mut arena_ptr, &n, &src_off, &dst_off]
            );
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(stream);
        }
    }

    /// Read one `row_inner`-wide row from output `out_idx` of the last run (a D2H of
    /// just that row). Caller must ensure the run has synced.
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if row_inner == 0 || out_idx >= self.graph.outputs.len() {
            return None;
        }
        let id = self.graph.outputs[out_idx];
        if !self.arena.has(id) {
            return None;
        }
        let cap = self.arena.len_of(id) / 4;
        let rel = row * row_inner;
        if rel + row_inner > cap {
            return None;
        }
        let base = self.arena.offset(id) / 4;
        let off = base + rel;
        let src = self.arena.buffer.ptr + (off as u64) * 4;
        let mut host = vec![0f32; row_inner];
        let bytes = row_inner * 4;
        let ok = unsafe {
            (self.ctx.runtime.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, src, bytes)
                .ok()
                .is_ok()
        };
        if ok { Some(host) } else { None }
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.gpu_handles.get(name).cloned()
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_rng(self.graph.clone(), self.rng());
        for (k, v) in &self.gpu_handles {
            exe.bind_gpu_handle(k, v);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }

    pub(crate) fn readback_plan(&self) -> Vec<usize> {
        let n = self.graph.outputs.len();
        if self.pending_read_indices.is_none() && self.gpu_handle_feeds.is_empty() {
            return (0..n).collect();
        }
        let mut set = std::collections::HashSet::new();
        if let Some(ref want) = self.pending_read_indices {
            set.extend(want.iter().copied());
        } else {
            return (0..n).collect();
        }
        for &idx in self.gpu_handle_feeds.values() {
            set.insert(idx);
        }
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort_unstable();
        v
    }

    pub(crate) fn stage_gpu_handle_inputs(&mut self, inputs: &[(&str, &[f32])]) {
        let arena_base = self.arena.buffer.ptr;
        for (name, data) in &self.gpu_handles {
            if inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            // Resident handles are updated in place in the arena by `feed_kv_row`;
            // their host mirror is stale (empty) — do NOT re-upload over the device data.
            if self.gpu_handle_resident.contains(name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let dst = arena_base + (off_f32 as u64) * 4;
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    host.htod(&self.ctx.runtime, dst, data.len())
                        .expect("rlx-rocm: gpu handle upload failed");
                } else {
                    unsafe {
                        let _ = (self.ctx.runtime.hip_memcpy_htod)(
                            dst,
                            data.as_ptr() as *const _,
                            std::mem::size_of_val(data.as_slice()),
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn refresh_gpu_handles_from_staging(&mut self, plan: &[usize]) {
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if plan.contains(&out_idx) && out_idx < self.output_staging.len() {
                self.gpu_handles
                    .insert(name.clone(), self.output_staging[out_idx].to_vec());
            }
        }
    }

    pub(crate) fn finalize_outputs(&mut self) -> Vec<Vec<f32>> {
        let plan = self.readback_plan();
        if plan.len() == self.graph.outputs.len() {
            self.fill_output_staging_all();
        } else {
            self.fill_output_staging_indices(&plan);
        }
        self.refresh_gpu_handles_from_staging(&plan);
        self.outputs_from_staging_plan(&plan)
    }

    pub(crate) fn outputs_from_staging_plan(&self, plan: &[usize]) -> Vec<Vec<f32>> {
        if self.pending_read_indices.is_none() && plan.len() == self.graph.outputs.len() {
            return self
                .output_staging
                .iter()
                .map(F32HostSlot::to_vec)
                .collect();
        }
        let want = self.pending_read_indices.as_deref().unwrap_or(plan);
        want.iter()
            .map(|&i| self.output_staging[i].to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_offsets_binary() {
        let s = Step::Binary {
            n: 4,
            a_off: 0,
            b_off: 4,
            c_off: 8,
            op: 0,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![0, 4]);
        assert_eq!(w, vec![8]);
    }

    #[test]
    fn step_offsets_matmul_with_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 1,
            bias_off_f32: 40,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20, 40]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_matmul_no_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 0,
            bias_off_f32: 0,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_attention_causal_no_mask_arg() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 1, // causal — mask_off ignored
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(!r.contains(&9999), "causal mask must not consume mask_off");
        assert_eq!(r, vec![0, 100, 200]);
    }

    #[test]
    fn step_offsets_attention_custom_mask_pulls_mask() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 2, // custom mask
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(r.contains(&9999));
    }

    #[test]
    fn step_offsets_scatter_add_acc_marks_out_as_rmw() {
        let s = Step::ScatterAddAcc {
            out_off: 100,
            upd_off: 200,
            idx_off: 300,
            num_updates: 4,
            trailing: 1,
            out_dim: 16,
        };
        let (r, w) = step_offsets(&s);
        // out is read-modify-write: present in BOTH reads and writes
        // so multi-stream sees the prior ScatterAddZero as a producer.
        assert!(r.contains(&100));
        assert!(w.contains(&100));
    }

    #[test]
    fn fuse_elementwise_merges_binary_then_unary() {
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 1);
        assert!(matches!(&fused[0], Step::FusedBinaryUnary { .. }));
    }

    #[test]
    fn fuse_elementwise_skips_when_intermediate_has_two_consumers() {
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 8,
                b_off: 8,
                c_off: 16,
                op: 2,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 3);
    }
}
