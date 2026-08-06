// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal backend — implements rlx-runtime's Backend trait.
//!
//! Pipeline:
//!   1. Run rlx-opt fusion passes on the graph
//!   2. Plan memory (single arena, GPU buffer)
//!   3. Compile thunk schedule
//!   4. On each run: encode thunks into a command buffer, commit, wait

use rlx_ir::{Graph, NodeId};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::arena::Arena;
use crate::device::metal_device;
use crate::thunk::{Thunk, ThunkSchedule};

/// Numeric precision for Metal graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalPrecision {
    /// Full f32 throughout. Always supported.
    F32,
    /// Half precision (f16). Requires f16 kernel variants for every op
    /// in the graph — currently only matmul has f16 kernels (`hgemm_*`).
    /// Until other ops are ported, F16 compile falls back to F32.
    F16,
}

/// Metadata for a param living in [`MetalExecutable::weight_buffer`].
#[derive(Clone, Copy)]
pub(crate) struct WeightParamSlot {
    pub offset: usize,
    pub nbytes: usize,
    pub nelems: usize,
    pub dtype: rlx_ir::DType,
}

/// Metal-compiled executable graph.
pub struct MetalExecutable {
    graph: Graph,
    arena: Arena,
    schedule: ThunkSchedule,
    input_ids: HashMap<String, NodeId>,
    param_ids: HashMap<String, NodeId>,
    /// Large params kept out of the activation arena (under the 4 GiB MPS cliff).
    weight_buffer: Option<metal::Buffer>,
    /// `NodeId` → slot in [`Self::weight_buffer`].
    weight_slots: HashMap<NodeId, WeightParamSlot>,
    /// Pre-resolved (name, byte_offset, max_f32_len) per input — for run_slots.
    input_slots: Vec<(String, usize, usize)>,
    output_slots: Vec<(usize, usize)>, // (byte_offset, f32_len)
    /// Precision this graph was compiled at.
    precision: MetalPrecision,
    /// Optional MPSGraph plan — populated when `RLX_USE_MPSGRAPH=1` and
    /// every op in the graph is supported by the lowerer. Replaces the
    /// per-op thunk path with one compiled MPSGraph for the whole forward.
    mps_plan: Option<crate::mps_graph_lower::MpsGraphPlan>,
    /// Hybrid MPSGraph + thunk schedule when whole-graph lowering fails
    /// (Qwen3.5 decode: matmul/norm/attn via MPS, GDN via thunks).
    mps_hybrid: Option<Vec<crate::mps_graph_hybrid::HybridStep>>,
    /// ICB segments — populated when `RLX_USE_ICB=1`. One segment per
    /// maximal run of ICB-compatible thunks in the schedule. Each segment
    /// pre-encodes its run into an `MTLIndirectCommandBuffer` at compile
    /// time; runtime calls `executeCommandsInBuffer` once per segment.
    /// Empty when ICB is disabled or no run exceeds the minimum length.
    icb_segments: Vec<crate::icb::IcbRange>,
    /// In-flight command buffers from `commit_no_wait`. Drained by
    /// `sync_pending`. Used by callers that pipeline multiple commits
    /// to amortize the GPU sync latency (~150µs/commit on Apple Silicon).
    pending_cmd_bufs: Vec<metal::CommandBuffer>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every thunk in `schedule` is in the
    /// safe set, `encode_commit` bypasses MPSGraph + ICB segments
    /// (both pre-encode at full extent) and dispatches per-op with
    /// scaled launch dimensions. Otherwise full-extent fallback.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Largest matmul FLOP count seen at compile time. Drives the
    /// MPSGraph-vs-per-op adaptive dispatch (see `encode_and_run`).
    /// Computed once because graph shape is static after compile.
    max_matmul_flops: u64,
    /// True when the graph has an `Op::MatMul` with a BF16 input. The
    /// per-op thunk `Sgemm` has no bf16-weight kernel (only f16), so it
    /// would read bf16 bytes as f32 → garbage. MPSGraph casts bf16→f32
    /// correctly (see `mps_graph_lower::MatMul`), so route these graphs
    /// through MPS regardless of the FLOP threshold. Computed once.
    has_bf16_matmul: bool,
    /// Set after the first `encode_and_run` triggers
    /// `freeze_params_to_mps_constants`. Subsequent runs skip the
    /// (idempotent but not free) re-lower.
    mps_params_frozen: bool,
    /// Arena tail reserved for ephemeral GatedDeltaNet state when
    /// `Op::GatedDeltaNet` runs without carry (state input absent).
    gdn_scratch_off: usize,
    /// Arena tail scratch for GPU GGUF dequant before matmul (reused per op).
    dequant_scratch_off: usize,
    /// Arena tail scratch for the m>8 SynthMatMul recon→MPS prefill weight Wᵀ[n,k].
    synth_matmul_scratch_off: usize,
    /// Arena tail scratch for GPU im2col before conv weight backward GEMM.
    conv_bwd_scratch_off: usize,
    /// Arena tail scratch for GPU attention backward (scores, dp, ds).
    attn_bwd_scratch_off: usize,
    /// Arena tail scratch for parallel RMSNorm param backward.
    rms_norm_bwd_scratch_off: usize,
    /// Arena tail scratch (ping-pong pair) for native multi-layer GRU / Elman RNN.
    rnn_gru_scratch_off: usize,
    /// Arena tail scratch for in-graph onnx.QMatMul act dequant (f32).
    onnx_qmatmul_act_scratch_off: usize,
    /// Cached dequant f32 weights for in-graph onnx.QMatMul.
    qmatmul_weight_cache: std::cell::RefCell<crate::onnx_qmatmul::QMatMulWeightCache>,
    /// Option A (`RLX_QWEN3_BAKE_WEIGHTS`): arena offsets of weight-only concats
    /// already computed once. On later steps those concats are skipped — the
    /// fused (constant) weight is left in place, saving the per-token re-copy.
    baked_weight_concats: std::cell::RefCell<std::collections::HashSet<usize>>,
    /// Persistent scratch for flash-decode SDPA partials (`RLX_METAL_SDPA_FLASH_DECODE`):
    /// float[(bi*heads+hi)*n_part + part][2 + 128]. Lazily sized/grown per call.
    sdpa_flash_scratch: std::cell::RefCell<Option<metal::Buffer>>,
    /// Persistent int8 scratch for the W8A8 decode-attention path
    /// (`RLX_METAL_W8A8_ATTN`): int8 K ‖ int8 V ‖ f32 K-scales ‖ f32 V-scales,
    /// each 256-B aligned. Lazily sized/grown per call.
    sdpa_w8a8_scratch: std::cell::RefCell<Option<metal::Buffer>>,
    /// Persistent F32 scratch for promoting F16 Linear weights before sgemm
    /// (legacy path; prefer native `sgemm_f16w`).
    #[allow(dead_code)]
    f16_weight_scratch: std::cell::RefCell<Option<metal::Buffer>>,
    /// Persistent KV / state inputs (unified-memory `Vec`, fed into arena each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    /// After each run, copy `graph.outputs[idx]` into the named handle.
    gpu_handle_feeds: HashMap<String, usize>,
    /// Handles whose arena input slots are authoritative (skip host mirror ping-pong).
    gpu_handle_resident: std::collections::HashSet<String>,
    /// `handle_name → output index` for the resident-KV *row* feed (decode graphs
    /// that emit the new token at the last bucket-padded output row, e.g. llama32
    /// `concat(past_k, k_new)`). Driven via [`feed_kv_row`]; kept separate from
    /// `gpu_handle_feeds` so the generic prefix propagation never fires for these.
    kv_row_feeds: HashMap<String, usize>,
    /// Planner for decode-attention kernel variant selection.
    sdpa_kernel_plan: crate::kernel_plan::KernelPlan,
    /// Persisted winner table: token bucket → preferred SDPA decode candidate.
    sdpa_tune_winners: std::cell::RefCell<HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>>,
    /// Per-bucket recency ticks for LRU eviction.
    sdpa_tune_last_used: std::cell::RefCell<HashMap<u16, u64>>,
    /// Monotonic counter used to stamp accesses/updates in LRU mode.
    sdpa_tune_tick: std::cell::Cell<u64>,
    /// True once the winner table has been loaded from disk (best-effort).
    sdpa_tune_loaded: std::cell::Cell<bool>,
}

unsafe impl Send for MetalExecutable {}

fn sdpa_tune_drop_keys(
    table: &HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>,
    last_used: &HashMap<u16, u64>,
    max_entries: usize,
    eviction: crate::kernel_plan::TuneCacheEviction,
) -> Vec<u16> {
    if table.len() <= max_entries {
        return Vec::new();
    }
    let drop_n = table.len() - max_entries;
    match eviction {
        crate::kernel_plan::TuneCacheEviction::KeepLowBuckets => {
            let mut keys: Vec<u16> = table.keys().copied().collect();
            keys.sort_unstable();
            keys.into_iter().rev().take(drop_n).collect()
        }
        crate::kernel_plan::TuneCacheEviction::KeepHighBuckets => {
            let mut keys: Vec<u16> = table.keys().copied().collect();
            keys.sort_unstable();
            keys.into_iter().take(drop_n).collect()
        }
        crate::kernel_plan::TuneCacheEviction::Lru => {
            let mut ranked: Vec<(u16, u64)> = table
                .keys()
                .copied()
                .map(|k| (k, last_used.get(&k).copied().unwrap_or(0)))
                .collect();
            ranked.sort_by_key(|(k, ts)| (*ts, *k));
            ranked.into_iter().take(drop_n).map(|(k, _)| k).collect()
        }
    }
}

fn parse_sdpa_tune_table_text(
    text: &str,
) -> (
    HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>,
    HashMap<u16, u64>,
    u64,
) {
    let mut table: HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate> = HashMap::new();
    let mut last_used: HashMap<u16, u64> = HashMap::new();
    let mut max_tick = 0u64;
    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let mut cols = s.split('\t');
        let bucket = cols.next().and_then(|v| v.parse::<u16>().ok());
        let tag = cols.next();
        let tick = cols.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        if let (Some(bucket), Some(tag)) = (bucket, tag)
            && let Some(candidate) = crate::kernel_plan::SdpaDecodeCandidate::from_tag(tag)
        {
            table.insert(bucket, candidate);
            last_used.insert(bucket, tick);
            max_tick = max_tick.max(tick);
        }
    }
    (table, last_used, max_tick)
}

fn render_sdpa_tune_table_text(
    table: &HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>,
    last_used: &HashMap<u16, u64>,
) -> String {
    let mut rows: Vec<(u16, crate::kernel_plan::SdpaDecodeCandidate)> =
        table.iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by_key(|(k, _)| *k);
    let mut out = String::from("# token_bucket\tcandidate\tlast_used_tick\n");
    for (bucket, cand) in rows {
        let tick = last_used.get(&bucket).copied().unwrap_or(0);
        out.push_str(&format!("{bucket}\t{}\t{tick}\n", cand.tag()));
    }
    out
}

fn load_sdpa_tune_table_file(
    path: &std::path::Path,
    max_entries: usize,
    eviction: crate::kernel_plan::TuneCacheEviction,
) -> (
    HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>,
    HashMap<u16, u64>,
    u64,
) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return (HashMap::new(), HashMap::new(), 0);
    };
    let (mut table, mut last_used, max_tick) = parse_sdpa_tune_table_text(&text);
    let to_drop = sdpa_tune_drop_keys(&table, &last_used, max_entries.max(1), eviction);
    for key in to_drop {
        table.remove(&key);
        last_used.remove(&key);
    }
    (table, last_used, max_tick)
}

fn persist_sdpa_tune_table_file(
    path: &std::path::Path,
    table: &HashMap<u16, crate::kernel_plan::SdpaDecodeCandidate>,
    last_used: &HashMap<u16, u64>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out = render_sdpa_tune_table_text(table, last_used);
    std::fs::write(path, out)
}

impl MetalExecutable {
    fn sdpa_tune_cache_path(&self) -> PathBuf {
        if let Some(p) = crate::runtime_config().sdpa_tune_cache_path {
            return PathBuf::from(p);
        }
        std::env::temp_dir().join("rlx-metal-sdpa-tune-v1.tsv")
    }

    fn sdpa_tune_cache_load_enabled(&self) -> bool {
        crate::runtime_config().sdpa_tune_cache_load && self.sdpa_kernel_plan.tune_policy.cache_load
    }

    fn sdpa_tune_cache_persist_enabled(&self) -> bool {
        crate::runtime_config().sdpa_tune_cache_persist
            && self.sdpa_kernel_plan.tune_policy.persist_cache
    }

    fn sdpa_tune_cache_max_entries(&self) -> usize {
        crate::runtime_config()
            .sdpa_tune_cache_max_entries
            .max(1)
            .min(self.sdpa_kernel_plan.tune_policy.cache_max_entries.max(1))
    }

    fn sdpa_tune_cache_eviction(&self) -> crate::kernel_plan::TuneCacheEviction {
        crate::runtime_config().sdpa_tune_cache_eviction
    }

    fn next_sdpa_tune_tick(&self) -> u64 {
        let next = self.sdpa_tune_tick.get().saturating_add(1);
        self.sdpa_tune_tick.set(next);
        next
    }

    fn mark_sdpa_tune_used(&self, token_bucket: u16) {
        let tick = self.next_sdpa_tune_tick();
        self.sdpa_tune_last_used
            .borrow_mut()
            .insert(token_bucket, tick);
    }

    fn trim_sdpa_tune_table(&self) {
        let max_entries = self.sdpa_tune_cache_max_entries();
        let eviction = self.sdpa_tune_cache_eviction();
        let mut table = self.sdpa_tune_winners.borrow_mut();
        if table.len() <= max_entries {
            return;
        }
        let last_used_snapshot = self.sdpa_tune_last_used.borrow().clone();
        let mut to_drop = sdpa_tune_drop_keys(&table, &last_used_snapshot, max_entries, eviction);

        if !to_drop.is_empty() {
            let mut last_used = self.sdpa_tune_last_used.borrow_mut();
            for key in to_drop.drain(..) {
                table.remove(&key);
                last_used.remove(&key);
            }
        }
    }

    pub(crate) fn sdpa_tuned_candidate(
        &self,
        token_bucket: u16,
    ) -> Option<crate::kernel_plan::SdpaDecodeCandidate> {
        if !self.sdpa_tune_cache_load_enabled() {
            return None;
        }
        if !self.sdpa_tune_loaded.get() {
            self.load_sdpa_tune_table();
        }
        let candidate = self.sdpa_tune_winners.borrow().get(&token_bucket).copied();
        if candidate.is_some() {
            self.mark_sdpa_tune_used(token_bucket);
        }
        candidate
    }

    pub(crate) fn sdpa_record_candidate(
        &self,
        token_bucket: u16,
        candidate: crate::kernel_plan::SdpaDecodeCandidate,
    ) {
        if !self.sdpa_tune_loaded.get() {
            self.load_sdpa_tune_table();
        }
        let mut table = self.sdpa_tune_winners.borrow_mut();
        let changed = table.get(&token_bucket).copied() != Some(candidate);
        if changed {
            table.insert(token_bucket, candidate);
            drop(table);
            self.mark_sdpa_tune_used(token_bucket);
            self.trim_sdpa_tune_table();
            if self.sdpa_tune_cache_persist_enabled() {
                self.persist_sdpa_tune_table();
            }
        } else {
            drop(table);
            self.mark_sdpa_tune_used(token_bucket);
        }
    }

    fn load_sdpa_tune_table(&self) {
        self.sdpa_tune_loaded.set(true);
        if !self.sdpa_tune_cache_load_enabled() {
            return;
        }
        let path = self.sdpa_tune_cache_path();
        let (table, last_used, max_tick) = load_sdpa_tune_table_file(
            &path,
            self.sdpa_tune_cache_max_entries(),
            self.sdpa_tune_cache_eviction(),
        );
        *self.sdpa_tune_winners.borrow_mut() = table;
        *self.sdpa_tune_last_used.borrow_mut() = last_used;
        self.sdpa_tune_tick.set(max_tick);
    }

    fn persist_sdpa_tune_table(&self) {
        if !self.sdpa_tune_cache_persist_enabled() {
            return;
        }
        let path = self.sdpa_tune_cache_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let table = self.sdpa_tune_winners.borrow();
        let last_used = self.sdpa_tune_last_used.borrow();
        let _ = persist_sdpa_tune_table_file(&path, &table, &last_used);
    }

    /// Resolve a thunk offset that may be tagged for the weight MTLBuffer.
    #[inline]
    pub(crate) fn resolve_off(&self, tagged: usize) -> (&metal::Buffer, usize) {
        use crate::thunk::{is_weight_off, raw_off};
        if is_weight_off(tagged) {
            (
                self.weight_buffer
                    .as_ref()
                    .expect("weight-tagged offset without weight_buffer"),
                raw_off(tagged),
            )
        } else {
            (&self.arena.buffer, tagged)
        }
    }

    /// Byte offset of a param in either the activation arena or weight buffer.
    #[inline]
    pub(crate) fn param_byte_offset(&self, id: NodeId) -> usize {
        if let Some(slot) = self.weight_slots.get(&id) {
            slot.offset
        } else {
            self.arena.byte_offset(id)
        }
    }

    /// Buffer holding a param (activation arena or weight buffer).
    #[inline]
    pub(crate) fn param_buffer(&self, id: NodeId) -> &metal::Buffer {
        if self.weight_slots.contains_key(&id) {
            self.weight_buffer
                .as_ref()
                .expect("weight slot without weight_buffer")
        } else {
            &self.arena.buffer
        }
    }

    fn write_weight_from_f32(&self, slot: WeightParamSlot, data: &[f32]) {
        let buf = self
            .weight_buffer
            .as_ref()
            .expect("write_weight_from_f32 without weight_buffer");
        let len = data.len().min(slot.nelems);
        unsafe {
            let base = (buf.contents() as *mut u8).add(slot.offset);
            match slot.dtype {
                rlx_ir::DType::F32 => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
                rlx_ir::DType::F16 => {
                    let dst = std::slice::from_raw_parts_mut(base as *mut half::f16, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = half::f16::from_f32(v);
                    }
                }
                rlx_ir::DType::BF16 => {
                    let dst = std::slice::from_raw_parts_mut(base as *mut half::bf16, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = half::bf16::from_f32(v);
                    }
                }
                _ => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
            }
        }
    }

    fn write_weight_bytes(&self, slot: WeightParamSlot, data: &[u8]) {
        let buf = self
            .weight_buffer
            .as_ref()
            .expect("write_weight_bytes without weight_buffer");
        let len = data.len().min(slot.nbytes);
        unsafe {
            let dst = (buf.contents() as *mut u8).add(slot.offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
        }
    }
}

impl Drop for MetalExecutable {
    fn drop(&mut self) {
        crate::prefill_stats::maybe_report_delta("drop");
        // Drain deferred commits before releasing MTL buffers / MPSGraph
        // executables — otherwise Metal logs "operations may not have completed".
        self.sync_pending();
        crate::device::drain_command_queue();
        crate::mps_blas::invalidate_caches();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand0() -> crate::kernel_plan::SdpaDecodeCandidate {
        crate::kernel_plan::SdpaDecodeVariant::BaseF32.candidate()
    }

    fn cand1() -> crate::kernel_plan::SdpaDecodeCandidate {
        crate::kernel_plan::SdpaDecodeVariant::PartialF16Kv.candidate()
    }

    #[test]
    fn eviction_keep_low_buckets_drops_highest_keys() {
        let mut table = HashMap::new();
        table.insert(0u16, cand0());
        table.insert(2u16, cand0());
        table.insert(5u16, cand1());
        let last_used = HashMap::new();
        let dropped = sdpa_tune_drop_keys(
            &table,
            &last_used,
            2,
            crate::kernel_plan::TuneCacheEviction::KeepLowBuckets,
        );
        assert_eq!(dropped, vec![5u16]);
    }

    #[test]
    fn eviction_keep_high_buckets_drops_lowest_keys() {
        let mut table = HashMap::new();
        table.insert(0u16, cand0());
        table.insert(2u16, cand0());
        table.insert(5u16, cand1());
        let last_used = HashMap::new();
        let dropped = sdpa_tune_drop_keys(
            &table,
            &last_used,
            2,
            crate::kernel_plan::TuneCacheEviction::KeepHighBuckets,
        );
        assert_eq!(dropped, vec![0u16]);
    }

    #[test]
    fn eviction_lru_drops_oldest_tick() {
        let mut table = HashMap::new();
        table.insert(1u16, cand0());
        table.insert(2u16, cand0());
        table.insert(3u16, cand1());
        let mut last_used = HashMap::new();
        last_used.insert(1u16, 30u64);
        last_used.insert(2u16, 10u64);
        last_used.insert(3u16, 20u64);
        let dropped = sdpa_tune_drop_keys(
            &table,
            &last_used,
            2,
            crate::kernel_plan::TuneCacheEviction::Lru,
        );
        assert_eq!(dropped, vec![2u16]);
    }

    #[test]
    fn tune_table_text_roundtrip_keeps_ticks() {
        let mut table = HashMap::new();
        table.insert(1u16, cand0());
        table.insert(4u16, cand1());
        let mut last_used = HashMap::new();
        last_used.insert(1u16, 5u64);
        last_used.insert(4u16, 12u64);

        let text = render_sdpa_tune_table_text(&table, &last_used);
        let (parsed_table, parsed_last_used, max_tick) = parse_sdpa_tune_table_text(&text);

        assert_eq!(parsed_table, table);
        assert_eq!(parsed_last_used, last_used);
        assert_eq!(max_tick, 12u64);
    }

    #[test]
    fn tune_table_text_parses_legacy_two_column_rows() {
        let c0 = cand0();
        let c1 = cand1();
        let text = format!(
            "# token_bucket\tcandidate\n2\t{}\n6\t{}\n",
            c0.tag(),
            c1.tag()
        );

        let (parsed_table, parsed_last_used, max_tick) = parse_sdpa_tune_table_text(&text);

        assert_eq!(parsed_table.get(&2), Some(&c0));
        assert_eq!(parsed_table.get(&6), Some(&c1));
        assert_eq!(parsed_last_used.get(&2), Some(&0u64));
        assert_eq!(parsed_last_used.get(&6), Some(&0u64));
        assert_eq!(max_tick, 0u64);
    }

    #[test]
    fn tune_table_file_load_trim_persist_with_lru() {
        let c0 = cand0();
        let c1 = cand1();
        let mut input_table = HashMap::new();
        input_table.insert(1u16, c0);
        input_table.insert(2u16, c1);
        input_table.insert(3u16, c0);
        let mut input_last_used = HashMap::new();
        input_last_used.insert(1u16, 30u64);
        input_last_used.insert(2u16, 10u64);
        input_last_used.insert(3u16, 20u64);

        let mut path = std::env::temp_dir();
        let unique = format!(
            "rlx-metal-sdpa-cache-test-{}-{}.tsv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        );
        path.push(unique);

        persist_sdpa_tune_table_file(&path, &input_table, &input_last_used).expect("persist");

        let (loaded_table, loaded_last_used, max_tick) =
            load_sdpa_tune_table_file(&path, 2, crate::kernel_plan::TuneCacheEviction::Lru);

        assert_eq!(loaded_table.len(), 2);
        assert!(loaded_table.contains_key(&1));
        assert!(loaded_table.contains_key(&3));
        assert!(!loaded_table.contains_key(&2));
        assert_eq!(loaded_last_used.get(&1), Some(&30u64));
        assert_eq!(loaded_last_used.get(&3), Some(&20u64));
        assert_eq!(max_tick, 30u64);

        persist_sdpa_tune_table_file(&path, &loaded_table, &loaded_last_used).expect("persist");
        let (roundtrip_table, roundtrip_last_used, roundtrip_max_tick) =
            parse_sdpa_tune_table_text(&std::fs::read_to_string(&path).expect("read"));
        assert_eq!(roundtrip_table, loaded_table);
        assert_eq!(roundtrip_last_used, loaded_last_used);
        assert_eq!(roundtrip_max_tick, 30u64);

        let _ = std::fs::remove_file(&path);
    }
}

mod bind;
mod compile;
mod encode;
mod output;
mod read;
mod run;
mod set;

pub use encode::has_metal_dequant_kernel;
pub(crate) use encode::*;

impl MetalExecutable {
    #[allow(dead_code)]
    fn ensure_f16_weight_scratch(&self, nbytes: usize) -> metal::Buffer {
        let need = nbytes.max(1) as u64;
        let mut slot = self.f16_weight_scratch.borrow_mut();
        let grow = slot.as_ref().map(|b| b.length() < need).unwrap_or(true);
        if grow {
            let dev = metal_device().expect("Metal device");
            *slot = Some(
                dev.device
                    .new_buffer(need, metal::MTLResourceOptions::StorageModeShared),
            );
        }
        slot.as_ref().expect("f16 weight scratch").clone()
    }

    /// Re-lower the MPSGraph plan, baking every param's current arena
    /// bytes in as a graph constant. After this call, the executable's
    /// feed list contains only the model's `Input`s — params are
    /// frozen into the compiled binary.
    ///
    /// Idempotent: a second call rebuilds against whatever bytes are
    /// in the arena now. Callers run this AFTER `set_param` has
    /// uploaded every weight (typical sequence: compile → set_param ×
    /// N → freeze → run × M). Triggered automatically on the first
    /// `run()` unless disabled with `RLX_DISABLE_MPSGRAPH_PARAM_CONST=1`.
    pub fn freeze_params_to_mps_constants(&mut self) {
        if self.mps_plan.is_none() && self.mps_hybrid.is_none() {
            return;
        }

        // Snapshot each param's current bytes from the arena. We only
        // freeze F32 params for now — typed-param plumbing (F16/BF16)
        // is a separate workstream; mixed-dtype paths stay on
        // placeholders for those.
        //
        // Size cap: `constantWithData:` ends up retained inside the
        // MPSGraphExecutable and never aliases the arena buffer, so
        // every baked constant is a fresh allocation outside our
        // arena. The qwen3 LM head weight alone is ~600 MB, and
        // compiling for multiple (B, L, mode) cells multiplies that.
        // Cap at 4 MB by default — large enough for small norms/biases,
        // small enough to skip LM heads & big FC layers. Override with
        // RLX_MPSGRAPH_PARAM_CONST_CAP=N (bytes; 0 disables the cap).
        let cap_bytes = rlx_ir::env::var("RLX_MPSGRAPH_PARAM_CONST_CAP")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024);
        let arena_ptr = self.arena.buffer.contents() as *const u8;
        let weight_ptr = self
            .weight_buffer
            .as_ref()
            .map(|b| b.contents() as *const u8);
        let mut param_bytes: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, id) in &self.param_ids {
            let node = self.graph.node(*id);
            if matches!(node.shape.dtype(), rlx_ir::DType::F32) {
                let n_elem = match node.shape.num_elements() {
                    Some(n) => n,
                    None => continue,
                };
                let len_bytes = n_elem * 4;
                if cap_bytes != 0 && len_bytes > cap_bytes {
                    continue;
                }
                let (base, off) = if let Some(slot) = self.weight_slots.get(id) {
                    (
                        weight_ptr.expect("weight param without buffer"),
                        slot.offset,
                    )
                } else if self.arena.has_buffer(*id) {
                    (arena_ptr, self.arena.byte_offset(*id))
                } else {
                    continue;
                };
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(base.add(off), len_bytes).to_vec() };
                param_bytes.insert(name.clone(), bytes);
                continue;
            }
            if !matches!(node.shape.dtype(), rlx_ir::DType::U8) {
                continue;
            }
            let Some((k, n, scheme)) = gguf_dequant_dims_for_param(&self.graph, *id) else {
                continue;
            };
            let u8_len = match node.shape.num_elements() {
                Some(n) => n,
                None => continue,
            };
            let f32_len = k * n * 4;
            if cap_bytes != 0 && f32_len > cap_bytes {
                continue;
            }
            let (base, off) = if let Some(slot) = self.weight_slots.get(id) {
                (
                    weight_ptr.expect("weight param without buffer"),
                    slot.offset,
                )
            } else if self.arena.has_buffer(*id) {
                (arena_ptr, self.arena.byte_offset(*id))
            } else {
                continue;
            };
            let u8_slice: &[u8] = unsafe { std::slice::from_raw_parts(base.add(off), u8_len) };
            let dequant = rlx_cpu::dequant_cache::gguf_weight_f32(off, u8_slice, k, n, scheme);
            let kn_bytes = transpose_nk_to_kn_bytes(&dequant, n, k);
            param_bytes.insert(name.clone(), kn_bytes);
        }

        // Re-run lowering with the params marked as constants. Old
        // plan is dropped, which releases the old executable and
        // cached arrays.
        let new_plan =
            crate::mps_graph_lower::try_lower_with_constants(&self.graph, Some(&param_bytes));
        if let Some(plan) = new_plan {
            self.mps_plan = Some(plan);
            self.mps_hybrid = None;
            // Re-bind the (now much smaller) feed list to the arena.
            self.bind_mps_executable_to_arena();
        } else if let Some(steps) =
            crate::mps_graph_hybrid::build_hybrid_plan(&self.graph, Some(&param_bytes))
                .filter(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps))
        {
            // Full-graph lower failed (Attention etc.) — bake constants into
            // the schedule-split hybrid so big-arena feeds stay small.
            self.mps_hybrid = Some(steps);
        }
    }

    pub(crate) fn estimated_max_flops(&self) -> u64 {
        self.max_matmul_flops
    }

    pub fn arena_ptr(&self) -> *const u8 {
        self.arena.buffer.contents() as *const u8
    }

    /// Encode + commit a forward pass without waiting for GPU completion.
    ///
    /// Use this to pipeline N runs and amortize the per-commit GPU sync
    /// latency (~150 µs on Apple Silicon). Caller MUST drain via
    /// `sync_pending` before reading any output (the arena is shared
    /// across pending commits, so output values are undefined until
    /// the GPU has caught up).
    ///
    /// Typical use: throughput benchmarks. Real-inference callers usually
    /// want `run` instead — pipelining requires per-commit output buffers
    /// or accepting that intermediate runs' outputs are stomped.
    pub fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        // Outputs go to the shared arena — caller is responsible for not
        // reading until sync_pending() AND for tolerating intermediate
        // commits stomping the output region. Use run_pipelined() if you
        // need outputs from each individual commit.
        if let Some(cmd_buf) = self.encode_commit(false, None, None) {
            self.pending_cmd_bufs.push(cmd_buf);
        }
    }

    /// Wait for every command buffer queued by `commit_no_wait`.
    pub fn sync_pending(&mut self) {
        for cb in self.pending_cmd_bufs.drain(..) {
            cb.wait_until_completed();
        }
    }

    /// Copy all named params from another executable with matching param layout.
    pub fn copy_params_from(&mut self, other: &Self) -> bool {
        if self.param_ids.len() != other.param_ids.len() {
            return false;
        }
        for (name, &dst_id) in &self.param_ids {
            let Some(&src_id) = other.param_ids.get(name) else {
                return false;
            };
            let dst_weight = self.weight_slots.get(&dst_id).copied();
            let src_weight = other.weight_slots.get(&src_id).copied();
            match (dst_weight, src_weight) {
                (Some(dst), Some(src)) => {
                    if dst.nbytes != src.nbytes || dst.dtype != src.dtype {
                        return false;
                    }
                    let src_buf = other.weight_buffer.as_ref().expect("src weight buf");
                    let dst_buf = self.weight_buffer.as_ref().expect("dst weight buf");
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (src_buf.contents() as *const u8).add(src.offset),
                            (dst_buf.contents() as *mut u8).add(dst.offset),
                            dst.nbytes,
                        );
                    }
                }
                (None, None) => {
                    if !self.arena.has_buffer(dst_id) || !other.arena.has_buffer(src_id) {
                        return false;
                    }
                    let dst_cap = *self.arena.element_counts.get(&dst_id).unwrap_or(&0);
                    let src_cap = *other.arena.element_counts.get(&src_id).unwrap_or(&0);
                    if dst_cap != src_cap {
                        return false;
                    }
                    self.arena
                        .copy_node_bytes_from(dst_id, &other.arena, src_id);
                }
                _ => return false,
            }
        }
        self.preload_qmatmul_weights();
        true
    }

    /// Share `other`'s external weight buffer instead of allocating + uploading
    /// our own — one GPU copy of the (large, read-only) packed weights backs
    /// both executables. Only valid when the weight-slot layout matches EXACTLY
    /// (same param name → same offset/size/dtype), which holds for executables
    /// compiled from the same param set (e.g. decode buckets, prefill↔decode
    /// when their tensor layout coincides). Small arena-resident params (norms,
    /// biases) are still copied into our own arena. Returns false — caller must
    /// fall back to a full upload/copy — on any layout mismatch.
    ///
    /// `other` must already have its weights uploaded. Call AFTER compile and
    /// BEFORE uploading this executable's own weights; on success, skip the
    /// upload for every weight-buffer param.
    pub fn share_weights_from(&mut self, other: &Self) -> bool {
        if self.param_ids.len() != other.param_ids.len() {
            return false;
        }
        // Verify byte-identical weight-buffer layout for every shared param.
        for (name, &dst_id) in &self.param_ids {
            let Some(&src_id) = other.param_ids.get(name) else {
                return false;
            };
            match (
                self.weight_slots.get(&dst_id).copied(),
                other.weight_slots.get(&src_id).copied(),
            ) {
                (Some(dst), Some(src)) => {
                    if dst.offset != src.offset
                        || dst.nbytes != src.nbytes
                        || dst.dtype != src.dtype
                    {
                        return false;
                    }
                }
                // Arena params copied below; both-arena is fine, mixed is not.
                (None, None) => {}
                _ => return false,
            }
        }
        // Retain the same MTLBuffer (metal::Buffer::clone bumps the refcount —
        // it does NOT copy the bytes). Dropping our old weight buffer here frees
        // the redundant per-executable copy. When there is NO weight buffer
        // (weights inline in the arena, i.e. externalization off), there is
        // nothing to share — return false so the caller uploads normally
        // (identical to the pre-sharing behavior; no arena-copy side effect).
        match (self.weight_buffer.as_ref(), other.weight_buffer.as_ref()) {
            (_, Some(src)) if !self.weight_slots.is_empty() => {
                if rlx_ir::env::flag("RLX_METAL_DEBUG") {
                    eprintln!(
                        "[rlx-metal] shared weight buffer ({:.2} GB) across executables",
                        src.length() as f64 / 1e9
                    );
                }
                self.weight_buffer = Some(src.clone());
            }
            _ => return false,
        }
        // Copy the small arena-resident params into our own (activation) arena.
        for (name, &dst_id) in &self.param_ids {
            if self.weight_slots.contains_key(&dst_id) {
                continue;
            }
            let src_id = other.param_ids[name];
            if !self.arena.has_buffer(dst_id) || !other.arena.has_buffer(src_id) {
                return false;
            }
            let dst_cap = *self.arena.element_counts.get(&dst_id).unwrap_or(&0);
            let src_cap = *other.arena.element_counts.get(&src_id).unwrap_or(&0);
            if dst_cap != src_cap {
                return false;
            }
            self.arena
                .copy_node_bytes_from(dst_id, &other.arena, src_id);
        }
        self.preload_qmatmul_weights();
        true
    }

    /// Warm the in-graph QMatMul weight dequant cache after all params are loaded.
    pub fn preload_qmatmul_weights(&mut self) {
        if !crate::onnx_qmatmul::ingraph_enabled() {
            return;
        }
        let arena_ptr = self.arena.buffer.contents() as *const u8;
        let mut cache = self.qmatmul_weight_cache.borrow_mut();
        let before = cache.len();
        for thunk in &self.schedule.thunks {
            let Thunk::CustomOp { kernel, inputs, .. } = thunk else {
                continue;
            };
            if kernel.name() != crate::onnx_qmatmul::KERNEL_NAME || inputs.len() < 6 {
                continue;
            }
            let read_input = |idx: usize| -> (&[u8], &rlx_ir::Shape) {
                let (off, len, shape) = &inputs[idx];
                let nbytes = (*len as usize) * shape.dtype().size_bytes();
                let data = unsafe { std::slice::from_raw_parts(arena_ptr.add(*off), nbytes) };
                (data, shape)
            };
            let (w_b, w_sh) = read_input(3);
            let (w_scale_b, _) = read_input(4);
            let (w_zp_b, w_zp_sh) = read_input(5);
            let k = w_sh.dim(0).unwrap_static().max(1);
            let n = w_sh.dim(1).unwrap_static().max(1);
            let w_scale = crate::onnx_qmatmul::read_f32_scalar(w_scale_b);
            let w_zp = crate::onnx_qmatmul::read_zp_i32(w_zp_b, w_zp_sh.dtype());
            cache.preload_weight(inputs[3].0, w_b, w_sh.dtype(), k, n, w_zp, w_scale);
        }
        let loaded = cache.len().saturating_sub(before);
        if loaded > 0 && rlx_ir::env::flag("KITTEN_RLX_TIMING") {
            eprintln!("[metal] QMatMul weight preload: {loaded} tiles");
        }
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.schedule.rng.read().expect("rng lock")
    }

    /// True when every thunk in the schedule is safe for active-extent
    /// dispatch — guards `encode_commit`'s bypass of MPSGraph + ICB.
    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule
            .thunks
            .iter()
            .all(|t| t.safe_for_active_extent())
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    /// Register a resident-KV *row* feed (vs the generic prefix feed): row
    /// `src_row` of output `output_index` is folded into handle `handle_name`'s
    /// input slot at `dst_row` by [`feed_kv_row`]. For decode graphs that emit
    /// the new token at the last bucket-padded output row (llama32).
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Fold each registered row feed's new-token row into its resident handle
    /// slot, in-place on the unified-memory arena. Call after a logits-only run.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                self.arena.copy_node_f32_range(
                    in_id,
                    dst_row * row_elems,
                    out_id,
                    src_row * row_elems,
                    row_elems,
                );
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Batch-major past `[B, seq_cap, row_elems]` ← new `[B, 1, row_elems]`.
    pub fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) {
        if batch == 0 || row_elems == 0 {
            return;
        }
        if batch == 1 {
            self.feed_kv_row(0, dst_row, row_elems);
            return;
        }
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                for b in 0..batch {
                    let src_elem = b * row_elems;
                    let dst_elem = b * seq_cap * row_elems + dst_row * row_elems;
                    self.arena
                        .copy_node_f32_range(in_id, dst_elem, out_id, src_elem, row_elems);
                }
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Strided-source batch-major resident KV feed: fold the new token from a
    /// registered output shaped `[B, src_seq_cap, row_elems]` (new token at
    /// `src_row` within each batch block — e.g. the FULL appended cache `[B,
    /// upper+1, kv_dim]` with `src_row = upper`) into the resident input handle
    /// `[B, dst_seq_cap, row_elems]` at `dst_row`. Generalizes
    /// [`Self::feed_kv_batch_major`] (which assumes a new-token-only `[B, 1]`
    /// output) to decode graphs that emit the whole cache.
    #[allow(clippy::too_many_arguments)]
    pub fn feed_kv_batch_major_src(
        &mut self,
        src_row: usize,
        src_seq_cap: usize,
        dst_row: usize,
        batch: usize,
        dst_seq_cap: usize,
        row_elems: usize,
    ) {
        if batch == 0 || row_elems == 0 {
            return;
        }
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                for b in 0..batch {
                    let src_elem = (b * src_seq_cap + src_row) * row_elems;
                    let dst_elem = (b * dst_seq_cap + dst_row) * row_elems;
                    self.arena
                        .copy_node_f32_range(in_id, dst_elem, out_id, src_elem, row_elems);
                }
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Ragged batch-major resident KV feed: like [`Self::feed_kv_batch_major_src`]
    /// but each batch element folds into its OWN `dst_rows[b]` (its sequence's
    /// current `past_len`) — for fused decode of sequences at MIXED cache lengths.
    pub fn feed_kv_batch_major_ragged(
        &mut self,
        src_row: usize,
        src_seq_cap: usize,
        dst_rows: &[usize],
        dst_seq_cap: usize,
        row_elems: usize,
    ) {
        let batch = dst_rows.len();
        if batch == 0 || row_elems == 0 {
            return;
        }
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                for (b, &dst_row) in dst_rows.iter().enumerate() {
                    let src_elem = (b * src_seq_cap + src_row) * row_elems;
                    let dst_elem = (b * dst_seq_cap + dst_row) * row_elems;
                    self.arena
                        .copy_node_f32_range(in_id, dst_elem, out_id, src_elem, row_elems);
                }
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_from_fused(
            self.graph.clone(),
            None,
            None,
            rlx_ir::RngOptions::default(),
            false,
        );
        // `compile_from_fused` re-initializes `Op::Constant` slots but leaves the
        // fresh arena's `Op::Param` (weight) slots zeroed — params are uploaded
        // externally via `set_param`/`finalize_params` after the first compile and
        // are NOT part of the graph. Without copying them, a cached/reused clone
        // (e.g. the in-memory graph cache in a persistent TTS service) runs with
        // all-zero weights: embeddings/matmuls collapse to zero on the 2nd+ run.
        exe.copy_params_from(self);
        for (name, data) in &self.gpu_handles {
            if !data.is_empty() {
                exe.bind_gpu_handle(name, data);
            }
        }
        for (name, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(name, idx);
        }
        for (name, &idx) in &self.kv_row_feeds {
            exe.register_kv_row_feed(name, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }

    pub(crate) fn propagate_gpu_handle_feeds_in_arena(&mut self) {
        let extent = self.active_extent;
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_elems = *self.arena.element_counts.get(&out_id).unwrap_or(&0);
                let copy_elems = match extent {
                    Some((actual, upper)) if upper > 0 => actual * (out_elems / (upper + 1)).max(1),
                    _ => out_elems,
                };
                self.arena
                    .copy_node_f32_prefix(in_id, out_id, copy_elems.min(out_elems));
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    pub(crate) fn refresh_gpu_handles_from_outputs(&mut self) {
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let id = self.graph.outputs[out_idx];
            let src = self.arena.slice(id);
            let entry = self
                .gpu_handles
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; src.len()]);
            if entry.len() != src.len() {
                entry.resize(src.len(), 0.0);
            }
            entry.copy_from_slice(src);
        }
    }

    pub(crate) fn dispatch_mps_plan(
        &self,
        plan: &crate::mps_graph_lower::MpsGraphPlan,
        boundary_parent_ids: Option<&HashMap<String, NodeId>>,
        output_parent_ids: Option<&[(NodeId, NodeId)]>,
    ) {
        let dev = metal_device().expect("Metal device");
        let arena_buf = &self.arena.buffer;

        let arena_len = arena_buf.length();
        let big_arena = arena_len >= (1u64 << 32);

        /// On big arenas, `newBufferWithBytesNoCopy` fails for offsets past the
        /// 4 GiB window (and for misaligned pointers). Stage those tensors.
        /// Low, page-aligned views into the parent buffer are OK once hybrid
        /// thunk indices match the schedule (see `mps_graph_hybrid`).
        fn needs_staging(offset: usize, nbytes: usize, big_arena: bool) -> bool {
            const PAGE: usize = 16_384;
            const LIMIT: usize = 1usize << 32;
            if !big_arena || nbytes == 0 {
                return false;
            }
            offset >= LIMIT || offset.saturating_add(nbytes) > LIMIT || !offset.is_multiple_of(PAGE)
        }
        fn tensor_nbytes(shape: &[usize], dt: u32) -> usize {
            let n: usize = shape.iter().product();
            let w = if dt == (0x10000000 | 16) { 2 } else { 4 };
            n * w
        }

        enum Slot {
            Arena { offset: usize },
            Weight { offset: usize },
            Staged { idx: usize },
        }

        let mut staged: Vec<metal::Buffer> = Vec::new();
        let mut feed_slots: Vec<Slot> = Vec::new();
        let mut feed_shapes: Vec<Vec<usize>> = Vec::new();
        let mut feed_dtypes: Vec<u32> = Vec::new();

        let mut push_feed =
            |buf: &metal::Buffer, offset: usize, shape: Vec<usize>, dt: u32, from_weight: bool| {
                let nbytes = tensor_nbytes(&shape, dt);
                // Separate weight MTLBuffers stay under the 4 GiB cliff by
                // construction — only activation-arena feeds need staging.
                let need_stage = !from_weight && needs_staging(offset, nbytes, big_arena);
                if need_stage {
                    let staged_buf = dev
                        .device
                        .new_buffer(nbytes as u64, metal::MTLResourceOptions::StorageModeShared);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (buf.contents() as *const u8).add(offset),
                            staged_buf.contents() as *mut u8,
                            nbytes,
                        );
                    }
                    let idx = staged.len();
                    staged.push(staged_buf);
                    feed_slots.push(Slot::Staged { idx });
                } else if from_weight {
                    feed_slots.push(Slot::Weight { offset });
                } else {
                    feed_slots.push(Slot::Arena { offset });
                }
                feed_shapes.push(shape);
                feed_dtypes.push(dt);
            };

        for (name, _t, shape, dt) in &plan.inputs {
            let off = if name.starts_with("__boundary_") {
                let parent = boundary_parent_ids
                    .and_then(|m| m.get(name))
                    .expect("hybrid boundary input");
                self.arena.byte_offset(*parent)
            } else {
                let id = self.input_ids.get(name).expect("input id");
                self.arena.byte_offset(*id)
            };
            push_feed(arena_buf, off, shape.clone(), *dt, false);
        }
        for (name, _t, shape, dt) in &plan.params {
            let id = *self.param_ids.get(name).expect("param id");
            let from_weight = self.weight_slots.contains_key(&id);
            push_feed(
                self.param_buffer(id),
                self.param_byte_offset(id),
                shape.clone(),
                *dt,
                from_weight,
            );
        }

        let mut out_slots: Vec<Slot> = Vec::new();
        let mut out_shapes: Vec<Vec<usize>> = Vec::new();
        let mut out_dtypes: Vec<u32> = Vec::new();
        let mut out_arena_offsets: Vec<usize> = Vec::new();

        let mut push_out = |offset: usize, shape: Vec<usize>, dt: u32| {
            let nbytes = tensor_nbytes(&shape, dt);
            out_arena_offsets.push(offset);
            if needs_staging(offset, nbytes, big_arena) {
                let buf = dev
                    .device
                    .new_buffer(nbytes as u64, metal::MTLResourceOptions::StorageModeShared);
                let idx = staged.len();
                staged.push(buf);
                out_slots.push(Slot::Staged { idx });
            } else {
                out_slots.push(Slot::Arena { offset });
            }
            out_shapes.push(shape);
            out_dtypes.push(dt);
        };

        if let Some(out_map) = output_parent_ids {
            for (sub_id, parent_id) in out_map {
                let off = self.arena.byte_offset(*parent_id);
                let (_, _t, shape, dt) = plan
                    .outputs
                    .iter()
                    .find(|(id, _, _, _)| id == sub_id)
                    .expect("hybrid output id");
                push_out(off, shape.clone(), *dt);
            }
        } else {
            for (id, _t, shape, dt) in &plan.outputs {
                push_out(self.arena.byte_offset(*id), shape.clone(), *dt);
            }
        }

        let any_staged = feed_slots
            .iter()
            .chain(out_slots.iter())
            .any(|s| matches!(s, Slot::Staged { .. }));

        let weight_buf = self.weight_buffer.as_ref();
        let feed_buffers: Vec<&metal::Buffer> = feed_slots
            .iter()
            .map(|s| match s {
                Slot::Arena { .. } => arena_buf,
                Slot::Weight { .. } => weight_buf.expect("weight feed without buffer"),
                Slot::Staged { idx } => &staged[*idx],
            })
            .collect();
        let feed_offsets: Vec<usize> = feed_slots
            .iter()
            .map(|s| match s {
                Slot::Arena { offset } | Slot::Weight { offset } => *offset,
                Slot::Staged { .. } => 0,
            })
            .collect();
        let out_buffers: Vec<&metal::Buffer> = out_slots
            .iter()
            .map(|s| match s {
                Slot::Arena { .. } | Slot::Weight { .. } => arena_buf,
                Slot::Staged { idx } => &staged[*idx],
            })
            .collect();
        let out_offsets: Vec<usize> = out_slots
            .iter()
            .map(|s| match s {
                Slot::Arena { offset } | Slot::Weight { offset } => *offset,
                Slot::Staged { .. } => 0,
            })
            .collect();

        if let Some(exec) = plan.executable.as_ref() {
            // Cached bindings pin arena offsets — unsafe once we stage.
            if exec.has_cached_binding() && !any_staged {
                exec.run_cached(&dev.queue);
                return;
            }
            exec.run(
                &dev.queue,
                &feed_buffers,
                &feed_offsets,
                &feed_shapes,
                &feed_dtypes,
                &out_buffers,
                &out_offsets,
                &out_shapes,
                &out_dtypes,
            );
        } else {
            let feed_tensors: Vec<&crate::mps_graph::MpsTensor> = plan
                .inputs
                .iter()
                .map(|(_, t, _, _)| t)
                .chain(plan.params.iter().map(|(_, t, _, _)| t))
                .collect();
            let out_tensors: Vec<&crate::mps_graph::MpsTensor> =
                plan.outputs.iter().map(|(_, t, _, _)| t).collect();
            plan.graph.run_jit(
                &dev.queue,
                &feed_tensors,
                &feed_buffers,
                &feed_offsets,
                &feed_shapes,
                &feed_dtypes,
                &out_tensors,
                &out_buffers,
                &out_offsets,
                &out_shapes,
                &out_dtypes,
            );
        }

        // Write staged outputs back into the parent arena.
        let arena_mut = arena_buf.contents() as *mut u8;
        for (i, slot) in out_slots.iter().enumerate() {
            if let Slot::Staged { idx } = slot {
                let nbytes = tensor_nbytes(&out_shapes[i], out_dtypes[i]);
                let dst = out_arena_offsets[i];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        staged[*idx].contents() as *const u8,
                        arena_mut.add(dst),
                        nbytes,
                    );
                }
            }
        }
    }
}
