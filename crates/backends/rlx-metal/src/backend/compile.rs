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

use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;
use std::collections::HashMap;

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
        Self::compile_inner(
            graph,
            None,
            None,
            false,
            rlx_ir::RngOptions::default(),
            false,
        )
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
        Self::compile_inner(graph, policy, supported_ops, false, rng, false)
    }

    /// Compile a graph that already went through the fusion pipeline
    /// (e.g. from [`rlx_ir::LirModule`]). Skips re-fusion so backends
    /// invoked via `Backend::compile_lir` do not undo fused ops.
    pub fn compile_from_fused(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        rng: rlx_ir::RngOptions,
        disable_mpsgraph: bool,
    ) -> Self {
        Self::compile_inner(graph, policy, supported_ops, true, rng, disable_mpsgraph)
    }

    pub(crate) fn compile_inner(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        skip_fusion: bool,
        rng: rlx_ir::RngOptions,
        disable_mpsgraph: bool,
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
        // AutoMixed / F16 residual streams are thunk-native today: MPSGraph
        // hybrid mis-types f16+f32 adds (abort in mps.add). Keep AMP on the
        // all-thunks path until MPS lowering is dtype-clean.
        let amp_active = policy
            .as_ref()
            .is_some_and(|p| !matches!(p, rlx_opt::PrecisionPolicy::AlwaysF32));
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
        // Apple MPSGraph + the fused sgemm epilogue mis-execute an erf-GELU
        // fused onto matmul→bias (O(1) divergence); split it into matmul+bias
        // and a standalone (correct) GELU. No-op unless such a node exists.
        let fused = split_erf_gelu_fmba_for_metal(fused);

        // Metal is an f32-arena backend: every compute kernel (compare, cast,
        // gather, transpose/expand, elementwise, matmul) reads and writes f32.
        // Integer & bool tensors — VITS sequence masks (`arange(t) < lengths`),
        // comparison results, gather indices, one-hot casts — must therefore be
        // materialized as f32 in the arena, exactly like rlx-wgpu's f32-uniform
        // arena. Without this, an i64 `arange` constant (8 B/elem) or a Bool
        // compare result (1 B/elem) is read back through an f32 pointer as
        // garbage → all-zero sequence mask → dead text encoder (TinyTTS). Weights
        // stay untouched (Op::Param packed/quant blocks are read as raw bytes by
        // DequantMatMul), and integer index/mask activations lose nothing: Metal's
        // gather already truncates indices through f32, so values must already fit
        // the f32 mantissa.
        let fused = widen_integer_activations_to_f32(fused);
        // Core SPD-manifold ops (BiMap / ReEig / LogEig / SpdBatchNorm /
        // SpdKarcherMean + backwards) run F64 on the CPU host-fallback
        // (`crate::spd`), but the Metal arena stays f32-uniform for them: widen
        // the SPD subgraph's F64 tensors to F32 so the input feed / output
        // readback and the surrounding Narrow/Reshape see plain f32. The
        // `Thunk::SpdHost` carries the real F64 shapes; `spd::eval` does the
        // f32↔f64 conversion. No-op when the graph has no SPD op.
        let fused = widen_spd_f64_to_f32(fused);
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
        //
        // Graphs with host indexing (ScatterNd / Gather*) are NOT safe to unpin:
        // mid-schedule CPU reads see whatever later GPU ops reused into those
        // slots. F5 DiT is the canary — unpinning a ~9 GiB arena (still over the
        // 4 GiB MPS cliff) saved almost nothing and drifted the ODE.
        let has_host_indexing = fused.nodes().iter().any(|n| {
            matches!(
                &n.op,
                Op::ScatterNd { .. }
                    | Op::ScatterElements { .. }
                    | Op::GatherNd { .. }
                    | Op::GatherElements { .. }
            )
        });
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
        // ≥4 GiB: MPSGraph / hybrid no-copy binds fail on high offsets. Prefer
        // dropping the output-ancestor pin (Metal thunk + hybrid dispatch is
        // in-order) so F16-weight models like Zonos fit under the cliff —
        // but only when the unpin actually helps and the graph has no host
        // indexing. Override with RLX_METAL_FORCE_PIN_OUTPUT_ANCESTORS=1.
        const MPS_BIND_CLIFF: usize = 1usize << 32;
        // Norms / biases stay in the activation arena (RmsNorm uses a single
        // buffer base + absolute offsets). Large Linear / embedding weights go
        // to a separate MTLBuffer so the act arena can drop under 4 GiB.
        const EXTERNAL_WEIGHT_MIN: usize = 256 * 1024;
        let mut weight_layout: Vec<(rlx_ir::NodeId, usize, usize, rlx_ir::DType)> = Vec::new();
        let force_pin = rlx_ir::env::flag("RLX_METAL_FORCE_PIN_OUTPUT_ANCESTORS");
        // Debug / regression: force the pre-fix unpin path even on ScatterNd
        // graphs (F5 DiT). Takes precedence over has_host_indexing.
        let force_unpin = rlx_ir::env::flag("RLX_METAL_FORCE_UNPIN_OUTPUT_ANCESTORS");
        let try_unpin = !force_pin
            && (force_unpin || !has_host_indexing)
            && (plan.arena_size > max_buffer || plan.arena_size >= MPS_BIND_CLIFF);
        if try_unpin {
            if verbose {
                eprintln!(
                    "[rlx-metal] arena {} B (≥ cliff {} / maxBufferLength {}); \
                     trying re-plan without output-ancestor pin{}",
                    plan.arena_size,
                    MPS_BIND_CLIFF,
                    max_buffer,
                    if force_unpin {
                        " (RLX_METAL_FORCE_UNPIN_OUTPUT_ANCESTORS)"
                    } else {
                        ""
                    }
                );
            }
            let unpinned = memory::plan_memory_with_options(
                &fused,
                128,
                memory::MemoryPlanOptions {
                    pin_output_ancestors: false,
                    arena_no_reuse: rlx_ir::env::flag("RLX_ARENA_NO_REUSE"),
                    ..Default::default()
                },
            );
            // Keep the pinned plan unless unpin meaningfully helps: under the
            // MPS cliff, or under maxBufferLength when we were over it.
            // FORCE_UNPIN always accepts (repro / bisect old F5 DiT drift).
            let accept = force_unpin
                || if plan.arena_size > max_buffer {
                    unpinned.arena_size <= max_buffer
                        || unpinned.arena_size + (plan.arena_size / 20) < plan.arena_size
                } else {
                    unpinned.arena_size < MPS_BIND_CLIFF
                };
            if accept {
                if verbose {
                    eprintln!(
                        "[rlx-metal] accepting unpinned plan {} B (was {} B)",
                        unpinned.arena_size, plan.arena_size
                    );
                }
                plan = unpinned;
            } else if verbose {
                eprintln!(
                    "[rlx-metal] keeping pinned plan {} B (unpinned {} B still ≥ cliff / no win)",
                    plan.arena_size, unpinned.arena_size
                );
            }
        } else if verbose && has_host_indexing && plan.arena_size >= MPS_BIND_CLIFF {
            eprintln!(
                "[rlx-metal] arena {} B ≥ cliff but graph has ScatterNd/Gather*; \
                 keeping output-ancestor pin",
                plan.arena_size
            );
        }
        // EXPERIMENTAL (opt-in, default OFF): externalize packed U8/I8 GGUF
        // weights even when the arena is UNDER the MPS cliff, so a dedicated
        // weight buffer exists to SHARE across the many small-arena decode
        // buckets (m=1 → tiny activations keep the arena below 4 GiB, so the
        // cliff alone never triggers the split). This would collapse the decode
        // buckets' ~3.8 GB-each inline duplication to one shared copy. It is
        // CORRECT within any single decode bucket (fused Q1_0 mv/mm resolve the
        // WEIGHT_BUF_TAG via `resolve_off`), but a generation that CROSSES from a
        // low bucket to the next corrupts once the arena is re-planned by the
        // split (host-fed KV / output-slot offsets diverge across the differently
        // laid-out bucket arenas). Off by default until that transition is fixed.
        // See `RLX_METAL_EXTERNALIZE_QUANT` and memory `bonsai27b_metal_fused_q1`.
        let ext_quant_split = rlx_ir::env::flag("RLX_METAL_EXTERNALIZE_QUANT")
            && fused.nodes().iter().any(|n| {
                matches!(&n.op, Op::Param { .. })
                    && matches!(n.shape.dtype(), rlx_ir::DType::U8 | rlx_ir::DType::I8)
                    && n.shape.num_elements().unwrap_or(0) * n.shape.dtype().size_bytes()
                        >= EXTERNAL_WEIGHT_MIN
            });
        if (plan.arena_size >= MPS_BIND_CLIFF || ext_quant_split)
            && !rlx_ir::env::flag("RLX_METAL_FORCE_INLINE_PARAMS")
        {
            // Keep output-ancestor pin when host indexing is present (F5 DiT
            // ScatterNd) — unpinning the act arena here used to undo the guard
            // above just to park large Linears externally.
            // FORCE_UNPIN overrides for A/B repro.
            let pin_act = (force_pin || has_host_indexing) && !force_unpin;
            let mut act_plan = memory::plan_memory_with_options(
                &fused,
                128,
                memory::MemoryPlanOptions {
                    allocate_params: false,
                    pin_output_ancestors: pin_act,
                    arena_no_reuse: rlx_ir::env::flag("RLX_ARENA_NO_REUSE"),
                    ..Default::default()
                },
            );
            // Append small params into the activation arena; large ones → weight buf.
            let mut tail = act_plan.arena_size;
            let align = 256usize;
            let mut external: Vec<(rlx_ir::NodeId, usize, usize, rlx_ir::DType)> = Vec::new();
            let mut ext_f16 = 0usize;
            let mut ext_f32 = 0usize;
            for node in fused.nodes() {
                if !matches!(&node.op, Op::Param { .. }) {
                    continue;
                }
                let ne = node.shape.num_elements().unwrap_or(0);
                let dt = node.shape.dtype();
                let nbytes = ne * dt.size_bytes();
                if nbytes == 0 {
                    continue;
                }
                // Packed GGUF quant weights (U8/I8) feed the fused DequantMatMul
                // thunk kernels, which read the packed bytes straight from the
                // activation arena. The external weight buffer is only wired for
                // the MPS/F16-Linear path; a quant weight parked there reads back
                // wrong (Bonsai-27B Q1_0 → coarse-lower projections → "a a a…").
                // Keep quant weights in the arena regardless of size.
                // EXPERIMENTAL (opt-in, default OFF): allow U8/I8 packed GGUF
                // weights (fused DequantMatMul) into the dedicated weight buffer
                // so it can be SHARED across executables. Correct for the prefill
                // GEMM (m>1) path, but NOT for fused decode kernels (see the
                // `ext_quant_split` note above) — hence off by default. Enable
                // with RLX_METAL_EXTERNALIZE_QUANT=1.
                let ext_quant = rlx_ir::env::flag("RLX_METAL_EXTERNALIZE_QUANT");
                let is_quant = matches!(dt, rlx_ir::DType::U8 | rlx_ir::DType::I8) && !ext_quant;
                // The external weight buffer is only read back correctly by ops
                // whose Metal thunk resolves the WEIGHT_BUF_TAG (matmul + gather,
                // via `resolve_off`). Ops like `Rope` bind the activation arena
                // directly with raw offsets, so a large param they consume — e.g.
                // the RoPE cos/sin tables [262144,128] f32 — parked in the weight
                // buffer reads back garbage → wrong RoPE → wrong attention → the
                // Bonsai-27B "a a a…" regression. Only externalize a param when
                // EVERY consumer is tag-aware (else leave it in the arena).
                let mut has_consumer = false;
                let mut all_consumers_tag_aware = true;
                for c in fused.nodes() {
                    if c.inputs.contains(&node.id) {
                        has_consumer = true;
                        let tag_aware =
                            matches!(c.op, Op::MatMul | Op::DotGeneral { .. } | Op::Gather { .. })
                                || (ext_quant
                                    && matches!(
                                        c.op,
                                        Op::DequantMatMul { .. } | Op::DequantGroupedMatMul { .. }
                                    ));
                        if !tag_aware {
                            all_consumers_tag_aware = false;
                            break;
                        }
                    }
                }
                let externalizable = has_consumer && all_consumers_tag_aware;
                if nbytes < EXTERNAL_WEIGHT_MIN || is_quant || !externalizable {
                    tail = (tail + align - 1) & !(align - 1);
                    act_plan.assignments.insert(
                        node.id,
                        memory::BufferSlot {
                            offset: tail,
                            size: nbytes,
                        },
                    );
                    tail += nbytes;
                } else {
                    match dt {
                        rlx_ir::DType::F16 => ext_f16 += 1,
                        rlx_ir::DType::F32 => ext_f32 += 1,
                        _ => {}
                    }
                    if rlx_ir::env::flag("RLX_METAL_EXT_TRACE") {
                        let consumers: std::collections::BTreeSet<String> = fused
                            .nodes()
                            .iter()
                            .filter(|c| c.inputs.contains(&node.id))
                            .map(|c| format!("{:?}", c.op).chars().take(32).collect())
                            .collect();
                        eprintln!(
                            "[rlx-metal] EXT node={:?} dt={dt:?} nbytes={nbytes} consumers={consumers:?}",
                            node.id
                        );
                    }
                    external.push((node.id, ne, nbytes, dt));
                }
            }
            act_plan.arena_size = tail;
            let ext_bytes: usize = external.iter().map(|(_, _, nb, _)| nb).sum();
            let weight_padded = {
                let mut c = 0usize;
                for &(_, _, nb, _) in &external {
                    c = (c + align - 1) & !(align - 1);
                    c += nb;
                }
                c
            };
            if verbose {
                eprintln!(
                    "[rlx-metal] weight-split candidates: {} large (f16={} f32={}) raw={:.2}GB pad={:.2}GB act={:.2}GB",
                    external.len(),
                    ext_f16,
                    ext_f32,
                    ext_bytes as f64 / 1e9,
                    weight_padded as f64 / 1e9,
                    act_plan.arena_size as f64 / 1e9
                );
            }
            if act_plan.arena_size < MPS_BIND_CLIFF
                && weight_padded < MPS_BIND_CLIFF
                && !external.is_empty()
            {
                if verbose {
                    eprintln!(
                        "[rlx-metal] split weights: act arena {:.2} GB, weight buf {:.2} GB \
                         ({} large params, {} B threshold)",
                        act_plan.arena_size as f64 / 1e9,
                        weight_padded as f64 / 1e9,
                        external.len(),
                        EXTERNAL_WEIGHT_MIN
                    );
                }
                let _ = ext_bytes;
                plan = act_plan;
                weight_layout = external;
            } else if verbose {
                eprintln!(
                    "[rlx-metal] weight split skipped (act={:.2}GB weight={:.2}GB external={})",
                    act_plan.arena_size as f64 / 1e9,
                    weight_padded as f64 / 1e9,
                    external.len()
                );
            }
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

        // Pack large params into a separate Shared MTLBuffer.
        let align = 256usize;
        let mut weight_offs: HashMap<NodeId, usize> = HashMap::new();
        let mut weight_slots: HashMap<NodeId, WeightParamSlot> = HashMap::new();
        let mut weight_cursor = 0usize;
        for &(id, nelems, nbytes, dtype) in &weight_layout {
            weight_cursor = (weight_cursor + align - 1) & !(align - 1);
            weight_offs.insert(id, weight_cursor);
            weight_slots.insert(
                id,
                WeightParamSlot {
                    offset: weight_cursor,
                    nbytes,
                    nelems,
                    dtype,
                },
            );
            weight_cursor += nbytes;
        }
        let weight_buffer = if weight_cursor > 0 {
            let dev = metal_device().expect("Metal device");
            Some(dev.alloc_shared(weight_cursor.max(64)))
        } else {
            None
        };

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

        let schedule = ThunkSchedule::compile_with_rng_fab_weights(
            &fused,
            &arena,
            rng,
            &fab_scratch,
            &weight_offs,
        );

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
        let mps_plan = if disable_mpsgraph
            || amp_active
            || rlx_ir::env::flag("RLX_DISABLE_MPSGRAPH")
        {
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
        let mps_hybrid = if !disable_mpsgraph
            && !amp_active
            && mps_plan.is_none()
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
        // Skip when large weights live in a separate MTLBuffer (ICB pins one
        // arena buffer only).
        let icb_segments = if rlx_ir::env::flag("RLX_USE_ICB") && weight_buffer.is_none() {
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
            weight_buffer,
            weight_slots,
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
            f16_weight_scratch: std::cell::RefCell::new(None),
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

/// Dtypes materialized as f32 in the Metal arena. Metal's compute kernels are
/// f32/f16-only, so integer & bool *activations/constants/indices* must live as
/// f32 (their values already fit the mantissa — Metal's gather truncates indices
/// through f32 regardless). Packed/quantized weights (`Op::Param`, e.g. GGUF
/// U8/I8 blocks read raw by DequantMatMul) are handled elsewhere and must keep
/// their true byte width, so U8/I8 are deliberately excluded here.
#[inline]
fn metal_widened_dtype(dt: rlx_ir::DType) -> bool {
    matches!(
        dt,
        rlx_ir::DType::I64 | rlx_ir::DType::I32 | rlx_ir::DType::U32 | rlx_ir::DType::Bool
    )
}

/// Reinterpret little-endian integer/bool bytes as f32 values (byte-encoded).
fn int_bytes_to_f32_bytes(data: &[u8], dt: rlx_ir::DType) -> Vec<u8> {
    use rlx_ir::DType;
    match dt {
        DType::I64 => data
            .chunks_exact(8)
            .flat_map(|c| (i64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes())
            .collect(),
        DType::I32 => data
            .chunks_exact(4)
            .flat_map(|c| (i32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes())
            .collect(),
        DType::U32 => data
            .chunks_exact(4)
            .flat_map(|c| (u32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes())
            .collect(),
        DType::Bool => data
            .iter()
            .flat_map(|&b| (b as f32).to_le_bytes())
            .collect(),
        _ => data.to_vec(),
    }
}

/// Rewrite every non-param integer/bool tensor node to F32 (converting `Constant`
/// payloads and `Cast` targets) so the whole graph runs through Metal's f32
/// kernels + f32-sized arena slots. See the call site for the rationale. Mirrors
/// rlx-wgpu's f32-uniform arena, which widens the same class of tensors on upload.
fn widen_integer_activations_to_f32(mut graph: Graph) -> Graph {
    use rlx_ir::DType;
    // `Op::Custom` ops (Sparse-LU/mat_vec, FFT, …) run as host kernels against
    // the unified-memory arena directly and read each input — and write each
    // output — at its declared dtype (e.g. CSR `col_idx`/`row_ptr` as I32 via
    // `expect_i32`, Bool masks). Widening those tensors to f32 would corrupt them
    // / make the host kernel reject the buffer, so any Custom node and any
    // tensor consumed by a Custom node keep their true byte width — only the
    // native f32 GPU kernels need the widened form.
    let custom_operands: std::collections::HashSet<rlx_ir::NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Custom { .. }))
        .flat_map(|n| std::iter::once(n.id).chain(n.inputs.iter().copied()))
        .collect();
    for node in graph.nodes_mut() {
        // Packed/quantized weight params live at their true byte width.
        if matches!(node.op, Op::Param { .. }) {
            continue;
        }
        // Host Custom kernels that need true integer widths (CSR I32
        // indices, Bool masks, …) keep I32/U32/Bool. I64 tensors that
        // feed ScatterElements / GatherND are widened to F32 so Metal's
        // f32 kernels don't densify float bytes into 8-byte arena slots
        // (which then look like garbage when read back as i64).
        if custom_operands.contains(&node.id) {
            let dt = node.shape.dtype();
            if matches!(
                dt,
                DType::I32 | DType::U32 | DType::Bool | DType::I8 | DType::U8 | DType::I16
            ) {
                continue;
            }
        }
        let old = node.shape.dtype();
        // Convert Constant literals up front (needs the pre-rewrite dtype).
        if metal_widened_dtype(old)
            && let Op::Constant { data } = &mut node.op
        {
            *data = int_bytes_to_f32_bytes(data, old);
        }
        // Cast → integer: keep `Op::Cast { to: I64/… }` as the truncation signal
        // for `CastTruncF32` in thunk compile, but still widen the *tensor*
        // shape to F32 so Unsqueeze/Copy/Gather stay on the f32 arena. Turning
        // `to` into F32 made Cast a no-op and zeroed Vocos fringe masks
        // (`view - floor(view)`).
        if metal_widened_dtype(old) {
            node.shape = node.shape.clone().with_dtype(DType::F32);
        }
    }
    graph
}

/// Reinterpret little-endian F64 bytes as F32 values (byte-encoded).
fn f64_bytes_to_f32_bytes(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(8)
        .flat_map(|c| (f64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes())
        .collect()
}

/// Widen the F64 nodes of the SPD-manifold subgraph (BiMap / ReEig / LogEig /
/// SpdBatchNorm / SpdKarcherMean + backwards) to F32 so the Metal arena is
/// f32-uniform for them — exactly like rlx-wgpu / rlx-vulkan. The SPD ops
/// themselves run F64 on the CPU host-fallback (`crate::spd`), which widens the
/// f32 arena bytes to f64 for the compute and narrows the result back; the
/// `Thunk::SpdHost` carries the REAL F64 shapes so the packed / backward layouts
/// resolve. Widening the arena side means the graph boundary (input feed /
/// output readback) and the surrounding `Narrow` / `Reshape` structural ops all
/// see plain f32 — the same class those Metal kernels already handle.
///
/// Scope: only F64 nodes in the **connected component** (undirected, via graph
/// edges) of an SPD op are touched. F64 tensors elsewhere (e.g. a genuine
/// `Op::Fft` F64 slot, which the FFT host path reads as 8-byte f64) keep their
/// width. No-op when the graph has no SPD op.
fn widen_spd_f64_to_f32(mut graph: Graph) -> Graph {
    use rlx_ir::DType;
    use std::collections::HashSet;

    // Seed the frontier with every SPD op and its operands.
    let mut frontier: Vec<rlx_ir::NodeId> = Vec::new();
    for node in graph.nodes() {
        if crate::spd::is_spd_host(&node.op) {
            frontier.push(node.id);
            frontier.extend(node.inputs.iter().copied());
        }
    }
    if frontier.is_empty() {
        return graph;
    }

    // Build the undirected adjacency (node ↔ each of its inputs) once.
    let mut adj: std::collections::HashMap<rlx_ir::NodeId, Vec<rlx_ir::NodeId>> =
        std::collections::HashMap::new();
    for node in graph.nodes() {
        for &inp in &node.inputs {
            adj.entry(node.id).or_default().push(inp);
            adj.entry(inp).or_default().push(node.id);
        }
    }

    // Flood-fill the component, staying on F64 nodes so unrelated dtypes (and
    // the boundary between the SPD F64 block and any f32 tensors it touches) are
    // not crossed. The SPD ops / their inputs are always F64, so this captures
    // the packed forward output → `Narrow` → `Reshape` chain and F64 params.
    let is_f64 = |id: rlx_ir::NodeId| graph.node(id).shape.dtype() == DType::F64;
    let mut component: HashSet<rlx_ir::NodeId> = HashSet::new();
    let mut stack: Vec<rlx_ir::NodeId> = frontier.into_iter().filter(|&id| is_f64(id)).collect();
    while let Some(id) = stack.pop() {
        if !component.insert(id) {
            continue;
        }
        if let Some(neighbors) = adj.get(&id) {
            for &nb in neighbors {
                if is_f64(nb) && !component.contains(&nb) {
                    stack.push(nb);
                }
            }
        }
    }

    for node in graph.nodes_mut() {
        if !component.contains(&node.id) {
            continue;
        }
        // Params stay at their true byte width elsewhere, but SPD F64 params
        // (e.g. the learnable SPD bias `G`) must be f32 in this f32-uniform
        // subgraph so the arena slot and the host widening agree. Convert
        // Constant literals up front (needs the pre-rewrite F64 dtype).
        if let Op::Constant { data } = &mut node.op {
            *data = f64_bytes_to_f32_bytes(data);
        }
        node.shape = node.shape.clone().with_dtype(DType::F32);
    }
    graph
}
