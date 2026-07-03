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

//! `compile` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    /// Compile at the requested precision.
    pub fn compile_with_precision(graph: Graph, precision: MetalPrecision) -> Self {
        // F16 compilation requires every kernel in the graph to have an f16
        // variant. Until they do, transparently fall back to F32 with a note.
        let effective = if precision == MetalPrecision::F16 {
            let verbose = rlx_ir::env::var("RLX_VERBOSE")
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0)
                >= 1;
            if verbose {
                eprintln!(
                    "[rlx-metal] F16 requested but full-graph f16 kernels are WIP; using F32"
                );
            }
            MetalPrecision::F32
        } else {
            precision
        };
        let mut exe = Self::compile(graph);
        exe.precision = effective;
        exe
    }


    pub fn compile(graph: Graph) -> Self {
        Self::compile_inner(graph, None, None, false, rlx_ir::RngOptions::default())
    }


    /// Compile with an optional `PrecisionPolicy`. The pass runs *after*
    /// fusion to avoid breaking pattern-match-based fusion via interleaved
    /// Cast nodes.
    pub fn compile_with_policy(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        Self::compile_inner(graph, policy, supported_ops, false, rng)
    }


    /// Compile a graph that already went through the fusion pipeline
    /// (e.g. from [`rlx_ir::LirModule`]). Skips re-fusion so backends
    /// invoked via `Backend::compile_lir` do not undo fused ops.
    pub fn compile_from_fused(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        Self::compile_inner(graph, policy, supported_ops, true, rng)
    }


    pub(crate) fn compile_inner(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        skip_fusion: bool,
        rng: rlx_ir::RngOptions,
    ) -> Self {
        let verbose = rlx_ir::env::var("RLX_VERBOSE")
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            >= 1;

        if verbose {
            eprintln!("[rlx-metal] compiling graph: {} nodes", graph.len());
        }

        // Drop the global MPSMatrix / MPSMatrixDescriptor / MPSMatrixMul
        // caches before building this compile's arena. The cache keys
        // include the Buffer-wrapper address, which CAN recycle when
        // the prior `MetalExecutable` is dropped — without this reset
        // a fresh Sam (e.g. CPU → Metal in the same process) gets
        // back stale `MPSMatrix` wrappers pointing at freed memory and
        // produces NaN outputs.
        crate::mps_blas::invalidate_caches();

        // Backend-aware fusion: only emit fused ops Metal can lower.
        let fused = if skip_fusion {
            graph
        } else {
            let mut pipe = rlx_opt::CompilePipeline::new(rlx_opt::FusionTarget::Metal)
                .with_assert_fusion_clean(false);
            if let Some(ops) = supported_ops {
                pipe = pipe.with_supported_ops(ops);
            }
            let compile_result = pipe.compile_graph(graph);
            if verbose {
                eprintln!(
                    "[rlx-metal] fusion: {} → {} nodes",
                    compile_result.fusion.nodes_before, compile_result.fusion.nodes_after
                );
            }
            compile_result.lir.into_graph()
        };

        // AutoMixedPrecision runs AFTER fusion: Cast nodes interleave between
        // the (now flattened) ops without breaking earlier pattern matchers.
        let fused = match policy {
            Some(p) => {
                use rlx_opt::pass::Pass;
                let g = rlx_opt::AutoMixedPrecision::new(p).run(fused);
                if verbose {
                    eprintln!("[rlx-metal] after AutoMixedPrecision: {} nodes", g.len());
                }
                g
            }
            None => fused,
        };

        // `FusedAttentionBlock` is a claimed op (the fusion pipeline / upstream
        // stages may emit it). Keep the native `fused_attn_block` kernel path
        // for f32, no-bias blocks whose `[seq,seq]` scores fit threadgroup
        // memory (`seq ≤ 64`); decompose every other FAB to the primitive
        // chain (matmul → narrow → rope → attention → matmul). Runs after AMP
        // so the f32 gate sees the final dtype. FAB-only — Metal's native
        // FusedMatMulBiasAct / FusedResidualLN / FusedSwiGLU survive.
        let fused = lower_fab_for_metal(fused);
        // Per-node (qkv, attn) BYTE offsets for the surviving native FAB nodes,
        // relative to the FAB scratch base (resolved against the arena below).
        let (fab_scratch_bytes, fab_scratch_rel) = fab_scratch_layout(&fused);

        if verbose {
            eprintln!("[rlx-metal] after fusion: {} nodes", fused.len());
        }

        // Memory plan with GPU-aligned cache lines (128B on Apple Silicon)
        let gdn_scratch = gdn_ephemeral_state_bytes(&fused);
        let dequant_scratch = dequant_gguf_scratch_bytes(&fused);
        let conv_bwd_scratch = conv_bwd_scratch_bytes(&fused);
        let attn_bwd_scratch = crate::attention_bwd_gpu::scratch_bytes(&fused);
        let rms_norm_bwd_scratch = rms_norm_bwd_scratch_bytes(&fused);
        let onnx_qmatmul_act_scratch = if crate::onnx_qmatmul::ingraph_gpu_enabled() {
            crate::onnx_qmatmul::act_scratch_bytes(&fused)
        } else {
            0
        };
        // Plan with the conservative output-ancestor liveness pin — correct for
        // every op, including recurrent ones (GRU) on the not-strictly-in-order
        // MPSGraph. Only if the resulting arena would exceed the device's
        // single-buffer limit (`maxBufferLength`) do we re-plan WITHOUT the pin to
        // restore slot reuse and shrink the arena. That's required for deep
        // feed-forward decoders (e.g. the Moshi 7B temporal stack, which emits a KV
        // tensor per layer so almost every node becomes an output-ancestor → 45 GB
        // pinned vs 27 GB reused) and is safe for them (no buffer is read after a
        // later op has reused its slot).
        let mut plan = memory::plan_memory_with_options(
            &fused,
            128,
            memory::MemoryPlanOptions {
                // CPU-fallback thunks (conv/maxpool backward) read arena buffers
                // AFTER the whole command buffer completes — including later GPU
                // ops that reused those slots. Slot reuse is unsafe in that mix;
                // RLX_ARENA_NO_REUSE pins every buffer to avoid the clobber.
                arena_no_reuse: rlx_ir::env::flag("RLX_ARENA_NO_REUSE"),
                ..Default::default()
            },
        );
        let max_buffer = crate::device::metal_device()
            .map(|d| d.device.max_buffer_length() as usize)
            .unwrap_or(usize::MAX);
        if plan.arena_size > max_buffer {
            if verbose {
                eprintln!(
                    "[rlx-metal] arena {} B > maxBufferLength {} B with output-ancestor pin; \
                     re-planning without it to restore slot reuse",
                    plan.arena_size, max_buffer
                );
            }
            plan = memory::plan_memory_with_options(
                &fused,
                128,
                memory::MemoryPlanOptions {
                    pin_output_ancestors: false,
                    ..Default::default()
                },
            );
        }
        let mut tail = plan.arena_size;
        let gdn_scratch_off = if gdn_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + gdn_scratch;
            off
        } else {
            0
        };
        let dequant_scratch_off = if dequant_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + dequant_scratch;
            off
        } else {
            0
        };
        let conv_bwd_scratch_off = if conv_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + conv_bwd_scratch;
            off
        } else {
            0
        };
        let attn_bwd_scratch_off = if attn_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + attn_bwd_scratch;
            off
        } else {
            0
        };
        let rms_norm_bwd_scratch_off = if rms_norm_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + rms_norm_bwd_scratch;
            off
        } else {
            0
        };
        let onnx_qmatmul_act_scratch_off = if onnx_qmatmul_act_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + onnx_qmatmul_act_scratch;
            off
        } else {
            0
        };
        // Native `Op::FusedAttentionBlock` packed-QKV + attn scratch.
        let fab_scratch_off = if fab_scratch_bytes > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + fab_scratch_bytes;
            off
        } else {
            0
        };
        plan.arena_size = tail;
        // Resolve per-node relative offsets to absolute arena byte offsets.
        let fab_scratch: std::collections::HashMap<rlx_ir::NodeId, (usize, usize)> =
            fab_scratch_rel
                .iter()
                .map(|(id, qkv_rel, attn_rel)| {
                    (*id, (fab_scratch_off + qkv_rel, fab_scratch_off + attn_rel))
                })
                .collect();
        if verbose && gdn_scratch > 0 {
            eprintln!(
                "[rlx-metal] GatedDeltaNet scratch: {} bytes @ offset {}",
                gdn_scratch, gdn_scratch_off
            );
        }
        if verbose && dequant_scratch > 0 {
            eprintln!(
                "[rlx-metal] DequantMatMul scratch: {} bytes @ offset {}",
                dequant_scratch, dequant_scratch_off
            );
        }
        if verbose && conv_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] Conv2dBackwardWeight scratch: {} bytes @ offset {}",
                conv_bwd_scratch, conv_bwd_scratch_off
            );
        }
        if verbose && attn_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] AttentionBackward scratch: {} bytes @ offset {}",
                attn_bwd_scratch, attn_bwd_scratch_off
            );
        }
        if verbose && rms_norm_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] RmsNormBackward param scratch: {} bytes @ offset {}",
                rms_norm_bwd_scratch, rms_norm_bwd_scratch_off
            );
        }
        if verbose && onnx_qmatmul_act_scratch > 0 {
            eprintln!(
                "[rlx-metal] onnx.QMatMul act scratch: {} bytes @ offset {}",
                onnx_qmatmul_act_scratch, onnx_qmatmul_act_scratch_off
            );
        }
        if verbose {
            eprintln!(
                "[rlx-metal] arena: {} bytes, {} buffers",
                plan.arena_size,
                plan.assignments.len()
            );
        }
        if std::env::var_os("RLX_METAL_DEBUG").is_some() {
            let mut sizes: Vec<(usize, usize)> = plan
                .assignments
                .values()
                .map(|s| (s.offset, s.size))
                .collect();
            sizes.sort_by_key(|&(_, sz)| std::cmp::Reverse(sz));
            let total: usize = plan.assignments.values().map(|s| s.size).sum();
            let max_end = plan
                .assignments
                .values()
                .map(|s| s.offset + s.size)
                .max()
                .unwrap_or(0);
            eprintln!(
                "[rlx-metal] arena_size={:.2} GB, {} buffers, sum_slot_bytes={:.2} GB, max_end={:.2} GB",
                plan.arena_size as f64 / 1e9,
                plan.assignments.len(),
                total as f64 / 1e9,
                max_end as f64 / 1e9,
            );
            for (off, sz) in sizes.iter().take(6) {
                eprintln!(
                    "    slot off={:.2}GB size={:.3}GB",
                    *off as f64 / 1e9,
                    *sz as f64 / 1e9
                );
            }
        }
        // Build precision-aware arena: per-node DType drives buffer sizing
        // and downstream kernel dispatch.
        let arena = Arena::from_plan_with_graph(plan, Some(&fused));

        // Initialize `Op::Constant` slots with their literal data. The
        // arena is shared-storage MTLBuffer (unified memory on Apple
        // Silicon) so we can write directly via `contents()`. F64 + I32 +
        // similar non-F32 dtypes go in as raw bytes; F32 also as raw
        // bytes (a constant's `data` field is little-endian dtype-native
        // already). Without this step, custom-op kernels reading from
        // a Constant input slot see zeros.
        for node in fused.nodes() {
            if let Op::Constant { data } = &node.op
                && !data.is_empty()
                && arena.has_buffer(node.id)
            {
                let off = arena.byte_offset(node.id);
                unsafe {
                    let dst = (arena.buffer.contents() as *mut u8).add(off);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            }
        }

        let schedule = ThunkSchedule::compile_with_rng_fab(&fused, &arena, rng, &fab_scratch);

        if verbose {
            let nop_count = schedule
                .thunks
                .iter()
                .filter(|t| matches!(t, crate::thunk::Thunk::Nop))
                .count();
            eprintln!(
                "[rlx-metal] schedule: {} thunks ({} compute, {} nop)",
                schedule.thunks.len(),
                schedule.thunks.len() - nop_count,
                nop_count
            );
        }

        let mut input_ids = HashMap::new();
        let mut param_ids = HashMap::new();
        for node in fused.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_ids.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_ids.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        let output_slots: Vec<(usize, usize)> = fused
            .outputs
            .iter()
            .map(|&id| {
                let off = if arena.has_buffer(id) {
                    arena.byte_offset(id)
                } else {
                    0
                };
                let logical = fused.node(id).shape.num_elements().unwrap_or(0);
                (off, logical)
            })
            .collect();

        // Pre-resolve input slots in graph-input order
        let mut input_slots = Vec::new();
        for node in fused.nodes() {
            if let Op::Input { name } = &node.op {
                let off = if arena.has_buffer(node.id) {
                    arena.byte_offset(node.id)
                } else {
                    0
                };
                let len = node.shape.num_elements().unwrap_or(0);
                input_slots.push((name.clone(), off, len));
            }
        }

        // MPSGraph lowering: on by default whenever every op is
        // supported by the bridge. Apple's fused MPSGraph kernels
        // outperform our per-op MSL encoder across the qwen3 prefill
        // range once RmsNorm + SDPA are wired (see mps_graph.rs).
        // Opt out with RLX_DISABLE_MPSGRAPH=1.
        let mps_plan = if rlx_ir::env::flag("RLX_DISABLE_MPSGRAPH") {
            None
        } else {
            let plan = crate::mps_graph_lower::try_lower(&fused);
            if verbose {
                match &plan {
                    Some(_) => eprintln!("[rlx-metal] MPSGraph lowering: success"),
                    None => eprintln!(
                        "[rlx-metal] MPSGraph lowering: unsupported op or dynamic shape; falling back to thunks"
                    ),
                }
            }
            plan
        };
        let mps_hybrid = if mps_plan.is_none()
            && rlx_ir::env::is_unset("RLX_DISABLE_MPSGRAPH")
            && rlx_ir::env::is_unset("RLX_DISABLE_MPSGRAPH_HYBRID")
        {
            crate::mps_graph_hybrid::build_hybrid_plan(&fused, None)
                .filter(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps))
        } else {
            None
        };
        if verbose && mps_hybrid.is_some() {
            eprintln!("[rlx-metal] MPSGraph hybrid lowering: enabled");
        }

        // Optional ICB pre-encoding: opt-in via env var. Pre-encodes the
        // ICB-compatible thunks (small element-wise / norm / copy ops) into
        // an IndirectCommandBuffer at compile time so encode_and_run can
        // issue them as one `executeCommandsInBuffer` call instead of N
        // individual `set_pipeline + set_buffer + dispatch` round-trips.
        let icb_segments = if rlx_ir::env::flag("RLX_USE_ICB") {
            let dev_ref = metal_device().expect("Metal device required");
            let segs =
                crate::icb::compile_segments(&schedule.thunks, &arena.buffer, &dev_ref.device);
            if verbose {
                let total_cmds: u64 = segs.iter().map(|r| r.segment.command_count).sum();
                eprintln!(
                    "[rlx-metal] ICB pre-encoded {} segments / {} commands",
                    segs.len(),
                    total_cmds
                );
            }
            segs
        } else {
            Vec::new()
        };

        let max_matmul_flops = max_matmul_flops_in(&fused);

        let mut me = Self {
            graph: fused,
            arena,
            schedule,
            input_ids,
            param_ids,
            input_slots,
            output_slots,
            precision: MetalPrecision::F32,
            mps_plan,
            mps_hybrid,
            icb_segments,
            pending_cmd_bufs: Vec::new(),
            active_extent: None,
            max_matmul_flops,
            mps_params_frozen: false,
            gdn_scratch_off,
            dequant_scratch_off,
            conv_bwd_scratch_off,
            attn_bwd_scratch_off,
            rms_norm_bwd_scratch_off,
            onnx_qmatmul_act_scratch_off,
            qmatmul_weight_cache: std::cell::RefCell::new(
                crate::onnx_qmatmul::QMatMulWeightCache::new(),
            ),
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: std::collections::HashSet::new(),
            kv_row_feeds: HashMap::new(),
        };
        // Bind the MPSGraph executable's input/output arrays to the
        // arena once. After this, run_cached() avoids all per-call
        // ObjC allocation. Arena buffer + per-node byte offsets are
        // fixed across runs, so the cached arrays stay valid for the
        // lifetime of `me`.
        me.bind_mps_executable_to_arena();
        me
    }

}
