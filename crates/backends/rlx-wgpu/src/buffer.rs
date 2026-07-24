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

//! Buffer arena for the wgpu backend. Mirrors the rlx-metal arena
//! shape: pre-plan one big storage buffer at compile time, sub-allocate
//! per-node offsets at known positions, treat I/O as `write_buffer` /
//! `read_buffer` against those offsets.
//!
//! wgpu's storage buffers are fine for both reads and writes from
//! compute shaders; there's no shared-memory requirement at the API
//! level (unlike Metal where `StorageModeShared` matters). On Apple
//! Silicon wgpu's Metal backend gives us unified memory automatically.

use rlx_ir::{Graph, NodeId};
use rlx_opt::memory::MemoryPlan;
use std::collections::HashMap;

/// Byte end (exclusive) of an f16 shadow write for a slot starting at
/// `f32_byte_offset` with `f32_byte_len` bytes of f32 payload.
/// wgpu requires `queue.write_buffer` sizes to be 4-byte aligned; odd
/// f16 element counts are zero-padded by two bytes in `write_f32`.
fn f16_shadow_write_end(f32_byte_offset: usize, f32_byte_len: usize) -> usize {
    let f16_off = f32_byte_offset / 2;
    let f16_bytes = (f32_byte_len / 4) * 2;
    let padded = (f16_bytes + 3) & !3;
    f16_off + padded
}

/// Size the f16 side buffer so every planned slot's padded upload fits.
fn f16_shadow_arena_size(plan: &MemoryPlan) -> usize {
    plan.assignments
        .values()
        .map(|a| f16_shadow_write_end(a.offset, a.size))
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Bytes reserved at the end of every shard for cross-stripe staging copies.
/// Tensors are never placed here; ops that bind a shard can stage outliers into
/// this zone without clobbering live activations.
///
/// Must fit the largest tensor still staged through the arena.
/// Qwen3-0.6B tied lm_head/embedding is ~594 MiB F32; keep headroom above that
/// so NVIDIA's ~2 GiB storage-bind path can stage weight-buffer params into the
/// act window. Override with `RLX_WGPU_SHARD_STAGE_MIB` (KittenTTS defaults to
/// 64 on discrete wgpu so long-wave plans do not snap into multi-×4 GiB OOM).
pub const SHARD_STAGE_RESERVE: usize = 768 * 1024 * 1024;

/// Effective stage reserve (see [`SHARD_STAGE_RESERVE`]).
pub fn shard_stage_reserve() -> usize {
    if let Ok(raw) = std::env::var("RLX_WGPU_SHARD_STAGE_MIB") {
        if let Ok(mib) = raw.parse::<usize>() {
            return (mib.max(1) * 1024 * 1024).min(SHARD_STAGE_RESERVE);
        }
    }
    SHARD_STAGE_RESERVE
}

/// Re-place unique buffer slots so no slot crosses a shard boundary.
/// Preserves shared offsets (true reuse); may insert padding gaps and grow
/// `arena_size`. Tail bytes past the last slot (scratch) are preserved.
///
/// Each stripe keeps `stage_reserve` free at its end for staging.
fn snap_plan_to_shards_ex(plan: &mut MemoryPlan, shard_cap: usize, stage_reserve: usize) {
    let usable = shard_cap.saturating_sub(stage_reserve).max(256);
    let mut slot_size: HashMap<usize, usize> = HashMap::new();
    let mut max_end = 0usize;
    for a in plan.assignments.values() {
        let sz = a.size.max(1);
        slot_size
            .entry(a.offset)
            .and_modify(|s| *s = (*s).max(sz))
            .or_insert(sz);
        max_end = max_end.max(a.offset.saturating_add(sz));
    }
    let tail_extra = plan.arena_size.saturating_sub(max_end);

    let mut ordered: Vec<(usize, usize)> = slot_size.into_iter().collect();
    ordered.sort_by_key(|(off, _)| *off);

    let mut remap: HashMap<usize, usize> = HashMap::with_capacity(ordered.len());
    let mut cursor = 0usize;
    for (old_off, size) in ordered {
        if size > usable {
            panic!(
                "rlx-wgpu: tensor slot at byte {old_off} is {size} bytes \
                 ({:.2} GiB) — exceeds the usable shard of {usable} bytes \
                 ({:.2} GiB = {stage_reserve}-byte stage reserve below the \
                 shard cap). A single tensor cannot exceed one buffer; chunk \
                 the oversized op over its output dimension at the graph level.",
                size as f64 / (1u64 << 30) as f64,
                usable as f64 / (1u64 << 30) as f64,
            );
        }
        cursor = cursor.div_ceil(16) * 16;
        let local = cursor % shard_cap;
        // Jump to next stripe when the slot would enter the stage reserve or
        // cross a hard shard boundary.
        if local + size > usable {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        remap.insert(old_off, cursor);
        cursor += size;
    }

    for a in plan.assignments.values_mut() {
        a.offset = *remap.get(&a.offset).expect("rlx-wgpu: missing slot remap");
    }

    // Preserve scratch / padding past the last assigned slot.
    //
    // Scratch MAY live in the per-stripe stage reserve (that zone exists for
    // staging copies). Do **not** open a brand-new empty shard just to host
    // tail bytes — that turned a ~3 GiB KittenTTS wave arena into 2×4 GiB
    // buffers (8 GiB) and is an easy GPU OOM on 8–16 GiB cards.
    cursor = cursor.div_ceil(16) * 16;
    if tail_extra > 0 {
        let local = cursor % shard_cap;
        // Jump only when even usable+reserve (the full stripe) can't hold it.
        if local.saturating_add(tail_extra) > shard_cap {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        cursor = cursor.saturating_add(tail_extra);
    }
    // Pad to the end of the *current* stripe when we still sit in the data
    // zone, so that stripe keeps its stage reserve. If we're already in the
    // reserve (or exactly on a boundary), keep the exact byte size — the
    // last physical buffer is allocated as `(size - begin).min(shard_cap)`.
    let local = cursor % shard_cap;
    if local > 0 && local <= usable {
        cursor = cursor.div_ceil(shard_cap) * shard_cap;
    }
    plan.arena_size = cursor.max(1);
}

fn snap_plan_to_shards(plan: &mut MemoryPlan, shard_cap: usize) {
    snap_plan_to_shards_ex(plan, shard_cap, shard_stage_reserve());
}

#[cfg(test)]
mod shard_pack_tests {
    use super::*;
    use rlx_ir::NodeId;

    #[test]
    fn snap_plan_keeps_slots_out_of_stage_reserve() {
        // Cap smaller than SHARD_STAGE_RESERVE → usable collapses to 256 bytes.
        let shard_cap = 4096usize;
        let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
        assert_eq!(usable, 256);

        let mut plan = MemoryPlan {
            arena_size: 10_000,
            assignments: HashMap::from([
                (
                    NodeId(0),
                    rlx_opt::memory::BufferSlot {
                        offset: 0,
                        size: 200,
                    },
                ),
                (
                    NodeId(1),
                    rlx_opt::memory::BufferSlot {
                        offset: 200,
                        size: 200,
                    },
                ),
                (
                    NodeId(2),
                    rlx_opt::memory::BufferSlot {
                        offset: 5000,
                        size: 100,
                    },
                ),
            ]),
            schedule: Vec::new(),
        };
        snap_plan_to_shards(&mut plan, shard_cap);
        for a in plan.assignments.values() {
            let local = a.offset % shard_cap;
            assert!(
                local + a.size.max(1) <= usable,
                "slot @{}+{} invades reserve (usable={usable})",
                a.offset,
                a.size
            );
        }
        // Last stripe may be partial when scratch sits in the stage reserve.
        assert!(plan.arena_size >= usable);
    }

    #[test]
    fn snap_plan_does_not_open_empty_shard_for_scratch() {
        // Mimic KittenTTS wave: ~3 GiB of tensors + modest scratch under a
        // ~4 GiB shard. Old packing opened a second full 4 GiB stripe for the
        // tail → 8 GiB GPU alloc (OOM risk). Scratch must stay in-stripe.
        let shard_cap = 4 * 1024 * 1024 * 1024usize; // 4 GiB
        let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
        let data = usable - 64 * 1024 * 1024; // leave headroom in data zone
        let scratch = 32 * 1024 * 1024usize;
        let mut plan = MemoryPlan {
            arena_size: data + scratch,
            assignments: HashMap::from([(
                NodeId(0),
                rlx_opt::memory::BufferSlot {
                    offset: 0,
                    size: data,
                },
            )]),
            schedule: Vec::new(),
        };
        snap_plan_to_shards(&mut plan, shard_cap);
        assert!(
            plan.arena_size <= shard_cap,
            "scratch opened an extra shard: arena={} > shard_cap={shard_cap} (OOM risk)",
            plan.arena_size
        );
        let a = plan.assignments.get(&NodeId(0)).unwrap();
        assert!(a.offset + a.size <= usable);
    }
}

/// One contiguous *logical* arena + per-node byte offsets. Lives for the
/// entire executable graph's lifetime.
///
/// When the planned size exceeds wgpu `max_buffer_size` (~4 GiB), the
/// logical range is striped across [`Self::extra_shards`] physical buffers
/// of [`Self::shard_size`] bytes each (`buffer` is shard 0). Bind helpers
/// map a window onto one shard and report a byte `rebase` so uniform
/// offsets stay relative to the bound range.
pub struct Arena {
    /// Shard 0 (or the only buffer when unsharded).
    pub buffer: wgpu::Buffer,
    /// Shards 1..N-1 when the logical arena exceeds `max_buffer_size`.
    pub extra_shards: Vec<wgpu::Buffer>,
    /// Bytes per shard (0 = unsharded single buffer of [`Self::size`]).
    pub shard_size: usize,
    /// Optional shadow buffer holding f16 versions of every value
    /// written via `write_f32`. Sized at half the arena byte budget
    /// (each f32 element pairs with an f16 element at the same logical
    /// index — i.e. f16_off = f32_off / 2). Created only when the
    /// device exposes the `SHADER_F16` feature; matmul kernels with
    /// f16-typed B input bind both `buffer` (for f32 activations) and
    /// `f16_buffer` (for f16 weights). Halves global memory traffic
    /// on the dominant matmul reads.
    pub f16_buffer: Option<wgpu::Buffer>,
    /// Per-node byte offset into the logical arena (or weight buffer).
    pub offsets: HashMap<NodeId, usize>,
    /// Per-node byte length.
    pub lens: HashMap<NodeId, usize>,
    /// Total logical arena size in bytes.
    pub size: usize,
    /// Byte offset of the tail scratch zone (also `size - scratch_bytes`).
    /// Set when callers request scratch via `from_plan_with_scratch`.
    /// Reuseable across ops since scratch is temporary — only one
    /// op writes to it at a time within a schedule.
    pub scratch_off: usize,
    /// Size in bytes of the tail scratch zone (0 when not used).
    pub scratch_bytes: usize,
    /// Separate buffer holding large packed quant weights (U8/I8) when the
    /// full arena would exceed wgpu's `max_buffer_size` (4 GiB). Bonsai-27B
    /// Q1_0 is 3.54 GiB packed; activations may still be sharded.
    /// Only the fused Q1_0 GEMV reads these, so no other op needs rebinding.
    pub weight_buffer: Option<wgpu::Buffer>,
    /// Per-node byte offset into `weight_buffer` (quant params only).
    pub weight_offsets: HashMap<NodeId, usize>,
}

/// How to bind a storage window onto an [`Arena`] (possibly sharded).
pub struct ArenaBindSpec<'a> {
    pub buffer: &'a wgpu::Buffer,
    /// Bind offset into `buffer` (256-byte aligned).
    pub local_base: u64,
    pub size: u64,
    /// Subtract from logical byte offsets before dividing by 4 for uniforms.
    pub rebase: u64,
}

/// High bit tagging a byte offset as living in [`Arena::weight_buffer`] rather
/// than the main arena buffer. Real arena offsets are < 4 GiB, so bit 62 is free.
pub const WEIGHT_BUF_TAG: usize = 1usize << 62;

#[inline]
pub fn is_weight_off(off: usize) -> bool {
    off & WEIGHT_BUF_TAG != 0
}
#[inline]
pub fn raw_weight_off(off: usize) -> usize {
    off & !WEIGHT_BUF_TAG
}

/// Plan memory using f32-sized slots regardless of declared IR dtype,
/// with liveness-aware slot reuse (see `rlx_compile::memory::plan_memory_f32_uniform`).
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    rlx_compile::memory::plan_memory_f32_uniform(graph, align)
}

impl Arena {
    /// Build an arena from a memory plan with an extra tail scratch zone
    /// of `scratch_bytes` reserved past the plan's arena_size. Useful for
    /// ops that need throwaway temp storage that doesn't fit in a
    /// workgroup-shared variable.
    pub fn from_plan_with_scratch(
        device: &wgpu::Device,
        plan: &MemoryPlan,
        scratch_bytes: usize,
    ) -> Self {
        if scratch_bytes == 0 {
            return Self::from_plan(device, plan);
        }
        // Round up to 16 for storage-binding alignment. Fold scratch into the
        // planned size so sharding (when needed) covers the tail too.
        let scratch_aligned = scratch_bytes.div_ceil(16) * 16;
        let mut grown = plan.clone();
        grown.arena_size = plan.arena_size + scratch_aligned;
        let mut arena = Self::from_plan(device, &grown);
        arena.scratch_bytes = scratch_aligned;
        // After possible shard-boundary repack, scratch is the last zone.
        arena.scratch_off = arena.size.saturating_sub(scratch_aligned);
        arena
    }

    /// Build an arena from a memory plan. Allocates one big buffer when the
    /// plan fits `max_buffer_size`; otherwise stripes the logical range across
    /// multiple ≤4 GiB shards ([`Self::extra_shards`]).
    ///
    /// When the plan fits in one allocation but exceeds
    /// `max_storage_buffer_binding_size` (`RLX_WGPU_LARGE_BUFFERS`), snap into
    /// virtual bind-sized stripes (`shard_size > 0`, `extra_shards` empty) so
    /// staging uses per-stripe reserves instead of clobbering live params.
    pub fn from_plan(device: &wgpu::Device, plan: &MemoryPlan) -> Self {
        let max_buf = device.limits().max_buffer_size as usize;
        let max_binding = device.limits().max_storage_buffer_binding_size as usize;
        // 256-byte alignment for storage-buffer bind offsets.
        let shard_cap = (max_buf / 256) * 256;
        let bind_shard_cap = (max_binding.min(max_buf) / 256) * 256;
        assert!(shard_cap >= 256, "rlx-wgpu: max_buffer_size too small");
        assert!(
            bind_shard_cap >= 256,
            "rlx-wgpu: max_storage_binding too small"
        );

        let mut plan = plan.clone();
        let size_hint = plan.arena_size.max(1);
        if size_hint > max_buf {
            snap_plan_to_shards(&mut plan, shard_cap);
        } else if size_hint > bind_shard_cap {
            // One physical buffer, bind-sized virtual stripes. Full stage
            // reserve packing can inflate a compact plan that still fits
            // `max_buf` into multi-GiB waste (KittenTTS: 3.7 GiB → 6–8 GiB →
            // GPU OOM). Shrink the per-stripe reserve until the packed plan
            // fits in one allocation.
            let compact = plan.clone();
            let default_reserve = shard_stage_reserve();
            let mut reserve = default_reserve.min(bind_shard_cap / 2);
            loop {
                let mut trial = compact.clone();
                snap_plan_to_shards_ex(&mut trial, bind_shard_cap, reserve);
                if trial.arena_size <= max_buf || reserve <= 1024 * 1024 {
                    if trial.arena_size <= max_buf {
                        if reserve < default_reserve
                            && (rlx_ir::env::flag("RLX_WGPU_DEBUG")
                                || rlx_ir::env::flag("RLX_WGPU_SHARD_LOG"))
                        {
                            eprintln!(
                                "[rlx-wgpu] bind-stripe snap with {:.0} MiB stage reserve \
                                 (default {:.0} MiB) → {:.3} GiB arena (fits max_buf)",
                                reserve as f64 / (1024.0 * 1024.0),
                                default_reserve as f64 / (1024.0 * 1024.0),
                                trial.arena_size as f64 / (1u64 << 30) as f64,
                            );
                        }
                        plan = trial;
                    } else if rlx_ir::env::flag("RLX_WGPU_DEBUG")
                        || rlx_ir::env::flag("RLX_WGPU_SHARD_LOG")
                    {
                        eprintln!(
                            "[rlx-wgpu] bind-stripe snap still {:.3} GiB (> max_buf) at \
                             {:.0} MiB reserve; keeping compact {:.3} GiB layout",
                            trial.arena_size as f64 / (1u64 << 30) as f64,
                            reserve as f64 / (1024.0 * 1024.0),
                            size_hint as f64 / (1u64 << 30) as f64,
                        );
                        plan = compact;
                    } else {
                        plan = compact;
                    }
                    break;
                }
                reserve = (reserve / 2).max(1024 * 1024);
            }
        }
        let mut size = plan.arena_size.max(1); // wgpu hates zero-sized allocs
        // Bind-stripe snap can still leave size > max_buf when size_hint was
        // already over the cap; re-pack onto physical shard boundaries.
        // Guard: never prefer an inflated multi-shard layout when the compact
        // plan fit in one buffer (handled above).
        if size > max_buf {
            snap_plan_to_shards(&mut plan, shard_cap);
            size = plan.arena_size.max(1);
        }

        let (buffer, extra_shards, shard_size) = if size <= max_buf {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena"),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let virt = if size > bind_shard_cap {
                if rlx_ir::env::flag("RLX_WGPU_DEBUG") || rlx_ir::env::flag("RLX_WGPU_SHARD_LOG") {
                    let n = size.div_ceil(bind_shard_cap);
                    eprintln!(
                        "[rlx-wgpu] virtual-sharded arena: physical={:.3} GiB → {n} × {:.3} GiB bind stripes",
                        size as f64 / (1u64 << 30) as f64,
                        bind_shard_cap as f64 / (1u64 << 30) as f64,
                    );
                }
                bind_shard_cap
            } else {
                0usize
            };
            (buffer, Vec::new(), virt)
        } else {
            // Reject tensors that cannot fit in one shard.
            for (id, a) in &plan.assignments {
                if a.size > shard_cap {
                    panic!(
                        "rlx-wgpu: node {id:?} slot {} bytes exceeds shard cap {} \
                         (max_buffer_size); shrink max_seq or split the graph",
                        a.size, shard_cap
                    );
                }
                let end = a.offset.saturating_add(a.size.max(1));
                if a.offset / shard_cap != (end - 1) / shard_cap {
                    panic!(
                        "rlx-wgpu: node {id:?} @{}+{} spans a {}-byte shard boundary after snap",
                        a.offset, a.size, shard_cap
                    );
                }
            }
            let n_shards = size.div_ceil(shard_cap);
            if rlx_ir::env::flag("RLX_WGPU_DEBUG") || rlx_ir::env::flag("RLX_WGPU_SHARD_LOG") {
                eprintln!(
                    "[rlx-wgpu] sharded arena: logical={:.3} GiB → {n_shards} × {:.3} GiB shards",
                    size as f64 / (1u64 << 30) as f64,
                    shard_cap as f64 / (1u64 << 30) as f64,
                );
            }
            let mut shards = Vec::with_capacity(n_shards);
            for i in 0..n_shards {
                let begin = i * shard_cap;
                let this = (size - begin).min(shard_cap).max(256);
                shards.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rlx-wgpu arena shard"),
                    size: this as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let buffer = shards.remove(0);
            (buffer, shards, shard_cap)
        };

        // Mirror f16 shadow buffer: half the byte size since each f32
        // slot maps to an f16 slot at the same logical element index.
        // On arenas larger than one bind window, allocate a capped f16
        // buffer (≤ max_storage_buffer_binding_size) so matmul can use
        // f16_weight_bind_range instead of staging multi‑GiB weights.
        //
        // Discrete Vulkan/DX12 Kitten paths host Int8 dequant and do not need
        // a 2 GiB f16 mirror — that alone OOMs 12 GiB cards next to a sharded
        // act arena. Skip unless `RLX_WGPU_F16_WEIGHTS=1`.
        let want_f16 = device.features().contains(wgpu::Features::SHADER_F16)
            && !rlx_ir::env::flag("RLX_WGPU_NO_F16_SHADOW")
            && (shard_size == 0
                || rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS")
                || !crate::device::coop_discrete_backend());
        let f16_buffer = if want_f16 {
            let f16_size = if size <= max_binding && shard_size == 0 {
                f16_shadow_arena_size(&plan)
            } else {
                // Sharded / huge arenas: keep a single-window f16 mirror.
                max_binding.min(shard_cap.max(max_binding))
            };
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena f16"),
                size: f16_size.max(256) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        let mut offsets = HashMap::with_capacity(plan.assignments.len());
        let mut lens = HashMap::with_capacity(plan.assignments.len());
        for (id, a) in &plan.assignments {
            offsets.insert(*id, a.offset);
            lens.insert(*id, a.size);
        }
        Self {
            buffer,
            extra_shards,
            shard_size,
            f16_buffer,
            offsets,
            lens,
            size,
            scratch_off: 0,
            scratch_bytes: 0,
            weight_buffer: None,
            weight_offsets: HashMap::new(),
        }
    }

    /// Build an arena that keeps params in a SEPARATE `weight_buffer`, so the
    /// activation arena stays under wgpu's storage bind / `max_buffer_size`
    /// caps. Used whenever the unified plan would exceed the bind limit
    /// (NVIDIA ~2 GiB) — including dequantized F32 GGUF weights for Qwen-class
    /// LMs and packed U8/I8 quant for 27B-class models. `offset()` returns a
    /// WEIGHT_BUF_TAG-tagged offset for those params; matmul stages them into
    /// the act window (or fused GGUF GEMV binds the weight buffer directly).
    pub fn from_plan_split(
        device: &wgpu::Device,
        plan: &MemoryPlan,
        graph: &Graph,
        scratch_bytes: usize,
    ) -> Self {
        use rlx_ir::Op;
        let _ = plan;
        // Start from a COMPACT activations-only plan (params unassigned), then
        // place every param: packed quant weights (U8/I8) → dedicated weight
        // buffer; everything else → arena tail. Relocating from the full plan
        // does NOT shrink the arena (the planner aliases the persistent packed
        // weights onto low offsets and leaves an activation at the high offset),
        // so we re-plan without params to actually compact it.
        let mut new_plan = rlx_compile::memory::plan_memory_f32_uniform_no_params(graph, 16);
        let walign = 256usize; // storage-buffer offset alignment for windowed binds
        let a = 16usize;
        let mut weight_offsets: HashMap<NodeId, usize> = HashMap::new();
        let mut weight_cursor = 0usize;
        let mut tail = new_plan.arena_size;
        // Cover every non-view node the compact plan didn't assign (params +
        // edge cases). Pure views are handled in a second pass so they alias
        // their roots — placing Reshape/Cast-of-Param as fresh tail slots left
        // those views zeroed (F5 DiT on >4 GiB sharded arenas: bias Reshape sat
        // next to the Param with no copy → ExpandHost broadcast zeros →
        // identity residual).
        for node in graph.nodes() {
            if new_plan.assignments.contains_key(&node.id) {
                continue;
            }
            if rlx_compile::memory::is_pure_view(graph, node) {
                continue;
            }
            let ne = node.shape.num_elements().unwrap_or(0).max(1);
            let dt = node.shape.dtype();
            let is_param = matches!(&node.op, Op::Param { .. });
            // `from_plan_split` runs when a unified arena would exceed the
            // storage bind limit (NVIDIA ~2 GiB). Park *all* params in the
            // weight buffer so the act arena stays compact and does not
            // virtual-shard — virtual sharding disables matmul param-anchor
            // and then panics on large F32 lm_head/embedding (Qwen3-0.6B GGUF
            // dequantized to F32). Packed U8/I8 quant already belonged here.
            // Opt out with `RLX_WGPU_PARAMS_IN_ACT=1` for debugging.
            // Legacy alias: `RLX_WGPU_ALL_PARAMS_WEIGHT=1` is now a no-op (default).
            let to_weight = is_param && !rlx_ir::env::flag("RLX_WGPU_PARAMS_IN_ACT");
            if to_weight {
                let nbytes = if matches!(dt, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
                    ne * dt.size_bytes()
                } else {
                    ne * 4
                };
                weight_cursor = weight_cursor.div_ceil(walign) * walign;
                weight_offsets.insert(node.id, weight_cursor);
                weight_cursor += nbytes;
            } else {
                tail = tail.div_ceil(a) * a;
                new_plan.assignments.insert(
                    node.id,
                    rlx_opt::memory::BufferSlot {
                        offset: tail,
                        size: ne * 4,
                    },
                );
                tail += ne * 4;
            }
        }
        // Alias pure views onto roots now that params live in `assignments`
        // (or `weight_offsets` for packed quant). Iterate to a fixed point so
        // view-of-view chains resolve after the intermediate view is placed.
        let aliases = rlx_compile::memory::collect_view_aliases(graph);
        let mut pending: Vec<NodeId> = aliases.keys().copied().collect();
        pending.sort_by_key(|id| id.0);
        let mut guard = pending.len() + 2;
        while !pending.is_empty() && guard > 0 {
            guard -= 1;
            let mut next = Vec::new();
            for id in pending {
                if new_plan.assignments.contains_key(&id) || weight_offsets.contains_key(&id) {
                    continue;
                }
                let Some(&(root, off)) = aliases.get(&id) else {
                    continue;
                };
                if let Some(&w_off) = weight_offsets.get(&root) {
                    weight_offsets.insert(id, w_off + off);
                    continue;
                }
                let Some(root_slot) = new_plan.assignments.get(&root).cloned() else {
                    next.push(id);
                    continue;
                };
                let ne = graph.node(id).shape.num_elements().unwrap_or(0).max(1);
                new_plan.assignments.insert(
                    id,
                    rlx_opt::memory::BufferSlot {
                        offset: root_slot.offset + off,
                        size: ne * 4,
                    },
                );
            }
            pending = next;
        }
        // Any remaining unassigned node (shouldn't happen) gets a fresh slot.
        for node in graph.nodes() {
            if new_plan.assignments.contains_key(&node.id) || weight_offsets.contains_key(&node.id)
            {
                continue;
            }
            let ne = node.shape.num_elements().unwrap_or(0).max(1);
            tail = tail.div_ceil(a) * a;
            new_plan.assignments.insert(
                node.id,
                rlx_opt::memory::BufferSlot {
                    offset: tail,
                    size: ne * 4,
                },
            );
            tail += ne * 4;
        }
        new_plan.arena_size = tail;
        if rlx_ir::env::flag("RLX_WGPU_DEBUG") {
            eprintln!(
                "[rlx-wgpu split] weight_params={} weight_buf={:.3}GiB act_arena={:.3}GiB params_in_act={}",
                weight_offsets.len(),
                weight_cursor as f64 / (1u64 << 30) as f64,
                new_plan.arena_size as f64 / (1u64 << 30) as f64,
                rlx_ir::env::flag("RLX_WGPU_PARAMS_IN_ACT"),
            );
        }
        let mut arena = Self::from_plan_with_scratch(device, &new_plan, scratch_bytes);
        if weight_cursor > 0 {
            let wbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu weight buffer (packed quant)"),
                size: weight_cursor.max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            arena.weight_buffer = Some(wbuf);
            arena.weight_offsets = weight_offsets;
        }
        arena
    }

    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id) || self.weight_offsets.contains_key(&id)
    }

    /// True when the logical arena is striped across multiple GPU buffers.
    #[inline]
    pub fn is_sharded(&self) -> bool {
        self.shard_size > 0
    }

    /// Logical byte offset of the staging reserve for the shard that contains
    /// `logical_off` (end of that stripe minus [`shard_stage_reserve`]).
    pub fn shard_stage_off(&self, logical_off: usize) -> usize {
        if !self.is_sharded() {
            return self.scratch_off;
        }
        let s = self.shard_size;
        let shard_idx = logical_off / s;
        let shard_begin = shard_idx * s;
        let shard_end = (shard_begin + s).min(self.size);
        shard_end
            .saturating_sub(shard_stage_reserve())
            .max(shard_begin)
    }

    /// Physical buffer + local offset for a logical activation byte address.
    pub fn resolve_act(&self, global_off: usize) -> (&wgpu::Buffer, usize) {
        assert!(
            !is_weight_off(global_off),
            "rlx-wgpu: resolve_act called with weight-tagged offset {global_off:#x}"
        );
        assert!(
            global_off < self.size || self.size == 0,
            "rlx-wgpu: resolve_act off={global_off} past arena size={}",
            self.size
        );
        if !self.is_sharded() {
            return (&self.buffer, global_off);
        }
        // Virtual striping: one physical buffer, logical `shard_size` windows.
        if self.extra_shards.is_empty() {
            return (&self.buffer, global_off);
        }
        let idx = global_off / self.shard_size;
        let local = global_off % self.shard_size;
        if idx == 0 {
            (&self.buffer, local)
        } else {
            (
                self.extra_shards.get(idx - 1).unwrap_or_else(|| {
                    panic!(
                        "rlx-wgpu: shard {idx} out of range (off={global_off} size={} \
                             shard_size={} n_extra={})",
                        self.size,
                        self.shard_size,
                        self.extra_shards.len()
                    )
                }),
                local,
            )
        }
    }

    /// Bind spec for a set of activation nodes (same shard required when sharded).
    pub fn bind_spec_for_nodes(&self, device: &wgpu::Device, ids: &[NodeId]) -> ArenaBindSpec<'_> {
        const ALIGN: u64 = 256;
        let max_binding = device.limits().max_storage_buffer_binding_size;

        // Resolve span in *logical* addresses (skip weight-tagged nodes).
        let mut lo: u64 = u64::MAX;
        let mut hi: u64 = 0;
        for &id in ids {
            let off = self.offset(id);
            if is_weight_off(off) {
                continue;
            }
            let len = self.len_of(id) as u64;
            let o = off as u64;
            lo = lo.min(o);
            hi = hi.max(o.saturating_add(len));
        }
        if lo == u64::MAX {
            // No activation nodes — bind shard 0 / whole small buffer.
            let size = (self.buffer.size()).min(max_binding).max(256);
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: 0,
                size,
                rebase: 0,
            };
        }

        if !self.is_sharded() {
            // Prefer whole-arena bind when it fits (absolute offsets, rebase=0).
            if (self.size as u64) <= max_binding {
                let size = (self.size as u64).min(self.buffer.size()).max(256);
                return ArenaBindSpec {
                    buffer: &self.buffer,
                    local_base: 0,
                    size,
                    rebase: 0,
                };
            }
            let span = hi.saturating_sub(lo).max(1);
            if span > max_binding {
                panic!("rlx-wgpu: op needs {span} bytes of arena span (>{max_binding})");
            }
            let mut base = (lo / ALIGN) * ALIGN;
            let mut size = span.div_ceil(ALIGN) * ALIGN;
            size = size.max(256).min(max_binding);
            if base.saturating_add(size) > self.size as u64 {
                base = (self.size as u64).saturating_sub(size);
                base = (base / ALIGN) * ALIGN;
            }
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: base,
                size,
                rebase: base,
            };
        }

        // Sharded: entire window must live in one stripe.
        let s = self.shard_size as u64;
        let shard_lo = lo / s;
        let shard_hi = (hi.saturating_sub(1)) / s;
        if shard_lo != shard_hi {
            let mut details = String::new();
            for &id in ids.iter().take(12) {
                let off = self.offset(id);
                if is_weight_off(off) {
                    continue;
                }
                details.push_str(&format!(" {:?}@{}+{};", id, off, self.len_of(id)));
            }
            panic!(
                "rlx-wgpu: op span [{lo},{hi}) crosses shard boundary at {} \
                 (shard_size={s}). nodes:{details}",
                (shard_lo + 1) * s
            );
        }
        let shard_base = shard_lo * s;
        let span = hi.saturating_sub(lo).max(1);
        if span > max_binding {
            panic!("rlx-wgpu: op needs {span} bytes within a shard (>{max_binding})");
        }
        // Always bind the whole stripe so the per-shard stage reserve at the
        // tail stays addressable for cross-shard staging.
        if self.extra_shards.is_empty() {
            // Virtual stripes share one buffer — bind at the stripe's absolute base.
            let phys = (self.size as u64)
                .saturating_sub(shard_base)
                .min(s)
                .min(max_binding)
                .max(256);
            return ArenaBindSpec {
                buffer: &self.buffer,
                local_base: shard_base,
                size: phys,
                rebase: shard_base,
            };
        }
        let (buf, _) = self.resolve_act(lo as usize);
        let shard_bytes = buf.size().min(s).min(max_binding).max(256);
        ArenaBindSpec {
            buffer: buf,
            local_base: 0,
            size: shard_bytes,
            rebase: shard_base,
        }
    }

    pub fn offset(&self, id: NodeId) -> usize {
        if let Some(&w) = self.weight_offsets.get(&id) {
            w | WEIGHT_BUF_TAG
        } else {
            *self.offsets.get(&id).unwrap_or_else(|| {
                panic!("rlx-wgpu arena: no offset for node {id:?} (not in arena or weight buffer)")
            })
        }
    }
    /// Resolve a (possibly weight-tagged) byte offset to (buffer, raw offset).
    pub fn resolve_w(&self, tagged: usize) -> (&wgpu::Buffer, usize) {
        if is_weight_off(tagged) {
            (
                self.weight_buffer
                    .as_ref()
                    .expect("weight-tagged off without weight_buffer"),
                raw_weight_off(tagged),
            )
        } else {
            self.resolve_act(tagged)
        }
    }
    pub fn len_of(&self, id: NodeId) -> usize {
        self.lens[&id]
    }

    /// Whether this node's f16 mirror fits in the capped f16 shadow buffer.
    pub fn param_fits_f16_mirror(&self, id: NodeId) -> bool {
        let Some(f16) = &self.f16_buffer else {
            return false;
        };
        // Weight-buffer params use a separate byte space; the f16 shadow is
        // laid out against *arena* offsets, so tagged offs never fit there.
        let off = self.offset(id);
        if is_weight_off(off) {
            return false;
        }
        let f16_off = off / 2;
        let f16_bytes = self.len_of(id) / 2;
        f16_off.saturating_add(f16_bytes) <= f16.size() as usize
    }

    /// Override the actual data length (in bytes) for a node. The
    /// backend calls this after planning to record true elem*4 sizes
    /// instead of the alignment-padded slot sizes.
    pub fn set_actual_len(&mut self, id: NodeId, bytes: usize) {
        self.lens.insert(id, bytes);
    }

    /// Write f32 data into the node's slot. The queue performs an
    /// async transfer; subsequent kernel dispatches on the same queue
    /// see the new bytes. When the device supports SHADER_F16, also
    /// downcasts and writes the same data into the f16 shadow buffer
    /// at offset `f32_offset / 2` — so matmul kernels with f16 weight
    /// bindings can read directly from there at half the bandwidth.
    pub fn write_f32(&self, queue: &wgpu::Queue, id: NodeId, data: &[f32]) {
        let off = self.offset(id);
        let bytes: &[u8] = bytemuck::cast_slice(data);
        // Route through chunked/shard-safe writer — large uploads (params,
        // ConcatHost, cross-shard staging) used to truncate on Metal.
        self.write_bytes_range(queue, off, bytes);
        if !is_weight_off(off) {
            self.write_f16_shadow_at(queue, off, data);
        }
    }

    /// Downcast host f32 data into the f16 shadow buffer at `id`'s slot.
    pub fn write_f16_shadow(&self, queue: &wgpu::Queue, id: NodeId, data: &[f32]) {
        self.write_f16_shadow_at(queue, self.offset(id), data);
    }

    fn write_f16_shadow_at(&self, queue: &wgpu::Queue, off: usize, data: &[f32]) {
        if let Some(f16_buf) = &self.f16_buffer {
            let f16_off = off / 2;
            let mut f16_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            if !f16_data.len().is_multiple_of(2) {
                f16_data.push(half::f16::from_f32(0.0));
            }
            let f16_byte_len = f16_data.len() * 2;
            if f16_off.saturating_add(f16_byte_len) > f16_buf.size() as usize {
                return;
            }
            let f16_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_byte_len) };
            queue.write_buffer(f16_buf, f16_off as u64, f16_bytes);
        }
    }

    /// Read a node's bytes back to host f32. Uses a fresh staging buffer;
    /// hot paths should call [`read_f32_pooled`] with a reused [`ReadbackStaging`].
    pub fn read_f32(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: NodeId) -> Vec<f32> {
        read_f32_pooled(self, device, queue, id, &mut None)
    }

    /// Read a byte range from the arena (used for packed GGUF weights).
    ///
    /// Handles sharded arenas (range may span stripes) and large transfers.
    pub fn read_bytes_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        byte_off: usize,
        len: usize,
    ) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        // Weight buffer is a single contiguous allocation — never stripe-split.
        if is_weight_off(byte_off)
            || !self.is_sharded()
            || (byte_off / self.shard_size) == ((byte_off + len - 1) / self.shard_size)
        {
            let (src, local) = self.resolve_w(byte_off);
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu readback bytes"),
                size: len as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu readback bytes enc"),
            });
            enc.copy_buffer_to_buffer(src, local as u64, &staging, 0, len as u64);
            queue.submit(std::iter::once(enc.finish()));

            let slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            receiver.recv().unwrap().unwrap();

            let view = slice.get_mapped_range().expect("buffer slice mapped");
            let out = view.to_vec();
            drop(view);
            staging.unmap();
            return out;
        }
        // Cross-stripe: stitch per-shard copies.
        let mut out = vec![0u8; len];
        let mut done = 0usize;
        while done < len {
            let g = byte_off + done;
            let room = self.shard_size - (g % self.shard_size);
            let n = (len - done).min(room);
            let piece = self.read_bytes_range(device, queue, g, n);
            out[done..done + n].copy_from_slice(&piece);
            done += n;
        }
        out
    }

    /// Write raw bytes into the arena at `byte_off`.
    ///
    /// Chunks large payloads (Metal `queue.write_buffer` can truncate above
    /// ~64 MiB) and splits at shard boundaries so a single call never writes
    /// past the end of a physical stripe.
    pub fn write_bytes_range(&self, queue: &wgpu::Queue, byte_off: usize, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        const CHUNK: usize = 64 * 1024 * 1024;
        // Weight-buffer uploads must not use act-shard boundary math — the high
        // WEIGHT_BUF_TAG bit would make `% shard_size` nonsense and truncate.
        let weight = is_weight_off(byte_off);
        let mut off = 0usize;
        while off < data.len() {
            let mut n = (data.len() - off).min(CHUNK);
            if !weight && self.is_sharded() {
                let g = byte_off + off;
                let room = self.shard_size - (g % self.shard_size);
                n = n.min(room);
            }
            // Non-final `write_buffer` sizes must be multiples of 4.
            if off + n < data.len() {
                n &= !3;
            }
            if n == 0 {
                n = (data.len() - off).min(4);
                if !weight && self.is_sharded() {
                    let g = byte_off + off;
                    n = n.min(self.shard_size - (g % self.shard_size));
                }
            }
            if n == 0 {
                break;
            }
            let (buf, local) = self.resolve_w(byte_off + off);
            queue.write_buffer(buf, local as u64, &data[off..off + n]);
            off += n;
        }
    }
}

/// Reusable MAP_READ staging buffer for output readback.
pub struct ReadbackStaging {
    buffer: wgpu::Buffer,
    capacity: usize,
}

/// Fixed 256 B MAP_READ staging for scalar (≤16 B) readback — avoids
/// `map_buffer_on_submit` + full-layout decode on MoltenVK hot paths.
pub struct TinyReadbackStaging {
    buffer: wgpu::Buffer,
}

impl TinyReadbackStaging {
    const CAPACITY: u64 = 256;

    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu tiny readback"),
                size: Self::CAPACITY,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// True when fused readback can use the tiny scalar fast path.
pub fn use_tiny_readback(layout: &ReadbackLayout, num_outputs: usize) -> bool {
    num_outputs == 1 && layout.total_bytes <= 16
}

/// After submit: decode one f32 vector from an already-mapped tiny staging buffer.
pub fn decode_tiny_mapped_f32(staging: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let len = len.max(4);
    let slice = staging.slice(..len as u64);
    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let out = bytemuck::cast_slice::<u8, f32>(&view[..len]).to_vec();
    drop(view);
    staging.unmap();
    out
}

/// After submit: map only `len` bytes and decode one f32 vector.
pub fn read_tiny_f32_after_submit(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let len = len.max(4);
    let slice = staging.slice(..len as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();
    decode_tiny_mapped_f32(staging, len)
}

impl ReadbackStaging {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn ensure(&mut self, device: &wgpu::Device, min_bytes: usize) {
        let need = min_bytes.max(256);
        if self.capacity >= need {
            return;
        }
        let cap = need.next_power_of_two().max(256);
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu readback staging"),
            size: cap as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.capacity = cap;
    }

    /// Grow-or-create staging for at least `min_bytes`.
    pub fn prepare(device: &wgpu::Device, staging: &mut Option<Self>, min_bytes: usize) {
        match staging {
            Some(s) => s.ensure(device, min_bytes),
            None => {
                let cap = min_bytes.max(256).next_power_of_two();
                *staging = Some(Self {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu readback staging"),
                        size: cap as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    capacity: cap,
                });
            }
        }
    }
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// Layout for batched output readback into a staging buffer.
#[derive(Debug, Clone)]
pub struct ReadbackLayout {
    pub regions: Vec<(usize, usize)>,
    pub total_bytes: usize,
}

impl ReadbackLayout {
    pub fn for_nodes(arena: &Arena, ids: &[NodeId]) -> Self {
        if ids.is_empty() {
            return Self {
                regions: Vec::new(),
                total_bytes: 0,
            };
        }
        if ids.len() == 1 {
            let len = arena.len_of(ids[0]);
            return Self {
                regions: vec![(0, len)],
                total_bytes: len,
            };
        }
        let mut regions = Vec::with_capacity(ids.len());
        let mut total = 0usize;
        for &id in ids {
            let len = arena.len_of(id);
            let start = total;
            total = align4(start + len);
            regions.push((start, len));
        }
        Self {
            regions,
            total_bytes: total,
        }
    }
}

/// Append arena→staging copies to an encoder (no submit).
pub fn encode_readback_copies(
    enc: &mut wgpu::CommandEncoder,
    arena: &Arena,
    staging: &wgpu::Buffer,
    ids: &[NodeId],
    layout: &ReadbackLayout,
) {
    for (&id, &(dst_off, len)) in ids.iter().zip(layout.regions.iter()) {
        let (src, local) = arena.resolve_w(arena.offset(id));
        enc.copy_buffer_to_buffer(src, local as u64, staging, dst_off as u64, len as u64);
    }
}

/// Map staging after submit and decode f32 outputs (one poll).
pub fn map_readback_f32(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    map_readback_f32_after_submit(device, staging, layout)
}

/// Block until the submission that produced the readback has finished and its
/// buffer-map callback has fired, then return.
///
/// A single submission-index `Wait` replaces the old 64–256 `poll(Poll)`
/// busy-spin followed by a full `Wait`. Each `Poll` maintain pass costs tens of
/// microseconds on Metal, so the spin alone added ~3 ms of pure CPU overhead per
/// run — the dominant cost for small graphs (an MNIST CNN forward dropped from
/// ~3.4 ms to sub-millisecond). wgpu invokes the buffer-map callback as part of
/// the poll, so the mapped range is ready when this returns.
pub fn wait_readback_map(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    _map_rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    _total_bytes: usize,
) {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    });
}

/// Schedule `map_async` on the encoder so mapping starts with submit (wgpu 29+).
pub fn schedule_readback_map(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let total = layout.total_bytes;
    let (sender, receiver) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(staging, wgpu::MapMode::Read, 0..total as u64, move |r| {
        let _ = sender.send(r);
    });
    receiver
}

fn map_readback_f32_after_submit(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    if layout.regions.is_empty() {
        return Vec::new();
    }
    let total = layout.total_bytes;
    let slice = staging.slice(..total as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let bytes = &view[..];
    let mut outs = Vec::with_capacity(layout.regions.len());
    for &(start, len) in &layout.regions {
        let chunk = &bytes[start..start + len];
        outs.push(bytemuck::cast_slice::<u8, f32>(chunk).to_vec());
    }
    drop(view);
    staging.unmap();
    outs
}

/// Decode f32 outputs after submit + map callback (used with [`schedule_readback_map`]).
pub fn decode_mapped_readback_f32(
    staging: &wgpu::Buffer,
    layout: &ReadbackLayout,
) -> Vec<Vec<f32>> {
    if layout.regions.is_empty() {
        return Vec::new();
    }
    let total = layout.total_bytes;
    let slice = staging.slice(..total as u64);
    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let bytes = &view[..];
    let mut outs = Vec::with_capacity(layout.regions.len());
    for &(start, len) in &layout.regions {
        let chunk = &bytes[start..start + len];
        outs.push(bytemuck::cast_slice::<u8, f32>(chunk).to_vec());
    }
    drop(view);
    staging.unmap();
    outs
}

/// Dependency-free `map_async` await for wasm.
///
/// The browser is single-threaded and drives GPU completion via its event
/// loop, so an `Rc<RefCell<…>>` oneshot is sufficient: the `map_async`
/// callback fires from a microtask once the copy completes, stores the
/// result, and wakes the awaiting task. No `device.poll` is needed (and it
/// would be a no-op on the WebGPU backend anyway).
#[cfg(target_arch = "wasm32")]
pub mod wasm_async {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    #[derive(Default)]
    struct Shared {
        result: Option<Result<(), wgpu::BufferAsyncError>>,
        waker: Option<Waker>,
    }

    struct MapSignal(Rc<RefCell<Shared>>);

    /// Future resolving when the mapped buffer is ready.
    pub struct MapWait(Rc<RefCell<Shared>>);

    impl MapSignal {
        fn complete(self, r: Result<(), wgpu::BufferAsyncError>) {
            let mut b = self.0.borrow_mut();
            b.result = Some(r);
            if let Some(w) = b.waker.take() {
                w.wake();
            }
        }
    }

    impl Future for MapWait {
        type Output = Result<(), wgpu::BufferAsyncError>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let mut b = self.0.borrow_mut();
            match b.result.take() {
                Some(r) => Poll::Ready(r),
                None => {
                    b.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    /// Map `buffer[..len]` for read and return a future that resolves when the
    /// GPU has finished. Call AFTER the copy into `buffer` has been submitted.
    pub fn map_read_async(buffer: &wgpu::Buffer, len: usize) -> MapWait {
        let shared = Rc::new(RefCell::new(Shared::default()));
        let signal = MapSignal(shared.clone());
        buffer
            .slice(..len as u64)
            .map_async(wgpu::MapMode::Read, move |r| signal.complete(r));
        MapWait(shared)
    }
}

/// Read one node via a reused staging buffer (one submit + one poll).
pub fn read_f32_pooled(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    id: NodeId,
    staging: &mut Option<ReadbackStaging>,
) -> Vec<f32> {
    let off = arena.offset(id);
    let len = arena.len_of(id);
    let n_elems = len / 4;
    if n_elems == 0 {
        return Vec::new();
    }
    ReadbackStaging::prepare(device, staging, len);
    let staging = staging.as_ref().expect("staging");
    let (src, local) = arena.resolve_w(off);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu readback enc"),
    });
    enc.copy_buffer_to_buffer(src, local as u64, &staging.buffer, 0, len as u64);
    queue.submit(std::iter::once(enc.finish()));

    let slice = staging.buffer.slice(..len as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.buffer.unmap();
    out
}

/// Read several nodes with one submit + one poll (contiguous staging layout).
pub fn read_f32_many_pooled(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ids: &[NodeId],
    staging: &mut Option<ReadbackStaging>,
) -> Vec<Vec<f32>> {
    if ids.is_empty() {
        return Vec::new();
    }
    let layout = ReadbackLayout::for_nodes(arena, ids);
    ReadbackStaging::prepare(device, staging, layout.total_bytes);
    let staging_buf = staging.as_ref().expect("staging").buffer().clone();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu readback batch enc"),
    });
    encode_readback_copies(&mut enc, arena, &staging_buf, ids, &layout);
    queue.submit(std::iter::once(enc.finish()));
    map_readback_f32(device, &staging_buf, &layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::NodeId;
    use rlx_opt::memory::{BufferSlot, MemoryPlan};
    use std::collections::HashMap;

    #[test]
    fn f16_shadow_arena_accounts_for_copy_alignment_padding() {
        // Three f32 elements → six f16 bytes, padded to eight for wgpu
        // COPY_BUFFER_ALIGNMENT. The old `arena_size / 2` sizing was two
        // bytes short at this slot boundary.
        let mut assignments = HashMap::new();
        assignments.insert(
            NodeId(0),
            BufferSlot {
                offset: 32,
                size: 12,
            },
        );
        let plan = MemoryPlan {
            arena_size: 44,
            assignments,
            schedule: vec![],
        };
        assert_eq!(f16_shadow_arena_size(&plan), 24);
    }
}
