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

/// One contiguous arena buffer + per-node byte offsets. Lives for the
/// entire executable graph's lifetime.
pub struct Arena {
    /// Underlying GPU buffer. Bound as a single STORAGE_READ_WRITE
    /// resource for every kernel; offsets disambiguate per-node access.
    pub buffer: wgpu::Buffer,
    /// Optional shadow buffer holding f16 versions of every value
    /// written via `write_f32`. Sized at half the arena byte budget
    /// (each f32 element pairs with an f16 element at the same logical
    /// index — i.e. f16_off = f32_off / 2). Created only when the
    /// device exposes the `SHADER_F16` feature; matmul kernels with
    /// f16-typed B input bind both `buffer` (for f32 activations) and
    /// `f16_buffer` (for f16 weights). Halves global memory traffic
    /// on the dominant matmul reads.
    pub f16_buffer: Option<wgpu::Buffer>,
    /// Per-node byte offset into `buffer`.
    pub offsets: HashMap<NodeId, usize>,
    /// Per-node byte length.
    pub lens: HashMap<NodeId, usize>,
    /// Total arena size in bytes.
    pub size: usize,
    /// Byte offset of the tail scratch zone (also `size - scratch_bytes`).
    /// Set when callers request scratch via `from_plan_with_scratch`.
    /// Reuseable across ops since scratch is temporary — only one
    /// op writes to it at a time within a schedule.
    pub scratch_off: usize,
    /// Size in bytes of the tail scratch zone (0 when not used).
    pub scratch_bytes: usize,
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
        let mut arena = Self::from_plan(device, plan);
        if scratch_bytes == 0 {
            return arena;
        }
        // Round up to 16 for storage-binding alignment.
        let scratch_aligned = scratch_bytes.div_ceil(16) * 16;
        let new_size = plan.arena_size + scratch_aligned;
        let max_buf = device.limits().max_buffer_size;
        if (new_size as u64) > max_buf {
            panic!(
                "rlx-wgpu: arena+scratch {} bytes exceeds max_buffer_size {}",
                new_size, max_buf
            );
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu arena+scratch"),
            size: new_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Drop the placeholder buffer (the smaller one from from_plan)
        // by replacing it; the wgpu Buffer destructor frees the old one.
        arena.buffer = buffer;
        arena.size = new_size;
        arena.scratch_off = plan.arena_size;
        arena.scratch_bytes = scratch_aligned;
        arena
    }

    /// Build an arena from a memory plan. Allocates one big buffer
    /// sized to fit every node's offset+length.
    pub fn from_plan(device: &wgpu::Device, plan: &MemoryPlan) -> Self {
        let size = plan.arena_size.max(1); // wgpu hates zero-sized allocs
        // Note: WebGPU caps each *binding range* at `max_storage_buffer_binding_size` (often 4 GiB
        // on native backends). We handle that at dispatch time by binding a per-kernel window.
        let max_buf = device.limits().max_buffer_size;
        if (size as u64) > max_buf {
            panic!(
                "rlx-wgpu: planned arena size {} bytes ({:.3} GiB) exceeds max_buffer_size {} bytes ({:.3} GiB)",
                size,
                size as f64 / (1u64 << 30) as f64,
                max_buf,
                max_buf as f64 / (1u64 << 30) as f64
            );
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu arena"),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Mirror f16 shadow buffer: half the byte size since each f32
        // slot maps to an f16 slot at the same logical element index.
        // On arenas larger than one bind window, allocate a capped f16
        // buffer (≤ max_storage_buffer_binding_size) so matmul can use
        // f16_weight_bind_range instead of staging multi‑GiB weights.
        let max_binding = device.limits().max_storage_buffer_binding_size as usize;
        let f16_buffer = if device.features().contains(wgpu::Features::SHADER_F16)
            && !rlx_ir::env::flag("RLX_WGPU_NO_F16_SHADOW")
        {
            let f16_size = if size <= max_binding {
                f16_shadow_arena_size(plan)
            } else {
                max_binding
            };
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena f16"),
                size: f16_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
        // `offsets` map to slot start (16-byte aligned). `lens` map to
        // ACTUAL data length (elems * 4) — distinct from the slot size,
        // which may include alignment padding. Readback uses lens so a
        // [5] f32 returns 5 elements, not the 8 that fit in a 32-byte
        // padded slot.
        let mut offsets = HashMap::with_capacity(plan.assignments.len());
        let mut lens = HashMap::with_capacity(plan.assignments.len());
        for (id, a) in &plan.assignments {
            offsets.insert(*id, a.offset);
            // Default to the slot size; backends may override via
            // set_actual_len for nodes whose elem count differs.
            lens.insert(*id, a.size);
        }
        Self {
            buffer,
            f16_buffer,
            offsets,
            lens,
            size,
            scratch_off: 0,
            scratch_bytes: 0,
        }
    }

    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id)
    }
    pub fn offset(&self, id: NodeId) -> usize {
        self.offsets[&id]
    }
    pub fn len_of(&self, id: NodeId) -> usize {
        self.lens[&id]
    }

    /// Whether this node's f16 mirror fits in the capped f16 shadow buffer.
    pub fn param_fits_f16_mirror(&self, id: NodeId) -> bool {
        let Some(f16) = &self.f16_buffer else {
            return false;
        };
        let f16_off = self.offset(id) / 2;
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
        queue.write_buffer(&self.buffer, off as u64, bytes);
        self.write_f16_shadow_at(queue, off, data);
    }

    /// Downcast host f32 data into the f16 shadow buffer at `id`'s slot.
    /// Used when skipping redundant f32 `write_buffer` but CoopF16Vk still
    /// needs a fresh f16 mirror (e.g. input upload hash cache hits).
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
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu readback bytes"),
            size: len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu readback bytes enc"),
        });
        enc.copy_buffer_to_buffer(&self.buffer, byte_off as u64, &staging, 0, len as u64);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        let out = view.to_vec();
        drop(view);
        staging.unmap();
        out
    }

    /// Write raw bytes into the arena at `byte_off`.
    pub fn write_bytes_range(&self, queue: &wgpu::Queue, byte_off: usize, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        queue.write_buffer(&self.buffer, byte_off as u64, data);
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
    let view = slice.get_mapped_range();
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
    wait_readback_map(device, &receiver, len);
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
        enc.copy_buffer_to_buffer(
            &arena.buffer,
            arena.offset(id) as u64,
            staging,
            dst_off as u64,
            len as u64,
        );
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

/// Poll until a readback map callback completes (fast path for tiny outputs).
pub fn wait_readback_map(
    device: &wgpu::Device,
    _map_rx: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    total_bytes: usize,
) {
    let spins = if total_bytes <= 16 { 256 } else { 64 };
    for _ in 0..spins {
        let _ = device.poll(wgpu::PollType::Poll);
    }
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
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
    wait_readback_map(device, &receiver, total);
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range();
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
    let view = slice.get_mapped_range();
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

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu readback enc"),
    });
    enc.copy_buffer_to_buffer(&arena.buffer, off as u64, &staging.buffer, 0, len as u64);
    queue.submit(std::iter::once(enc.finish()));

    let slice = staging.buffer.slice(..len as u64);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    wait_readback_map(device, &receiver, len);
    receiver.recv().unwrap().unwrap();

    let view = slice.get_mapped_range();
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
