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

use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory::MemoryPlan;
use std::collections::HashMap;

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
}

/// Plan memory using f32-sized slots regardless of declared IR dtype.
/// The wgpu backend keeps every tensor as f32 in the arena (Bool / I32
/// /etc. get widened on access), so dtype-aware sizing in
/// `rlx_opt::memory::plan_memory_aligned` would under-allocate slots
/// for non-f32 nodes — Bool\[4\] would get 4 bytes when our kernels need
/// 16. This planner side-steps the issue with a simple sequential
/// allocator (no slot reuse). Memory pressure stays modest because
/// activations dominate and they're already f32 in the IR.
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    use rlx_opt::memory::BufferSlot;
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        // Reshape and Cast are pure relabels in our row-major f32-uniform
        // arena — element count is unchanged, every dtype occupies 4 bytes.
        // Alias the output slot to the input's offset/size and skip the
        // copy kernel emission downstream. Saves one dispatch + one
        // arena round-trip per Reshape/Cast.
        if matches!(node.op, Op::Reshape { .. } | Op::Cast { .. })
            && let Some(in_id) = node.inputs.first()
            && let Some(slot) = assignments.get(in_id)
        {
            let aliased = slot.clone();
            assignments.insert(node.id, aliased);
            schedule.push(node.id);
            continue;
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        let bytes = elems * 4;
        let aligned = bytes.div_ceil(align) * align;
        assignments.insert(
            node.id,
            BufferSlot {
                offset: cursor,
                size: aligned,
            },
        );
        schedule.push(node.id);
        cursor += aligned;
    }
    MemoryPlan {
        arena_size: cursor,
        assignments,
        schedule,
    }
}

impl Arena {
    /// Build an arena from a memory plan. Allocates one big buffer
    /// sized to fit every node's offset+length.
    pub fn from_plan(device: &wgpu::Device, plan: &MemoryPlan) -> Self {
        let size = plan.arena_size.max(1); // wgpu hates zero-sized allocs
        // WebGPU spec caps `max_storage_buffer_binding_size` at 32-bit
        // offsets (4 GiB - 4 B). Apple Metal's adapter limit is exactly
        // that. When the planned arena exceeds it, every bind group
        // creation will fail with a cryptic "binding range exceeds
        // limit" error deep in wgpu-core. Detect early and panic with a
        // useful message; the long-term fix is multi-bind-group arena
        // partitioning (deferred — substantial restructuring).
        let max_binding = device.limits().max_storage_buffer_binding_size;
        if (size as u64) > max_binding {
            panic!(
                "rlx-wgpu: planned arena size {} bytes ({} GiB) exceeds the \
                    adapter's max_storage_buffer_binding_size of {} bytes \
                    ({} GiB). This is the WebGPU 32-bit binding offset cap on \
                    Apple Metal / Vulkan; supporting larger arenas requires \
                    multi-bind-group partitioning. Workaround: reduce batch \
                    size or compile with a smaller (batch, seq) shape.",
                size,
                size as f64 / (1u64 << 30) as f64,
                max_binding,
                max_binding as f64 / (1u64 << 30) as f64
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
        let f16_buffer = if device.features().contains(wgpu::Features::SHADER_F16) {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu arena f16"),
                size: size.div_ceil(2) as u64, // bytes — each f32 slot = 2 f16 bytes
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
        if let Some(f16_buf) = &self.f16_buffer {
            // wgpu requires queue.write_buffer to use 4-byte-aligned
            // sizes (`COPY_BUFFER_ALIGNMENT`). f16 is 2 bytes; an odd
            // element count yields a non-aligned byte length. Pad with
            // a zero half so the byte count is always even.
            let mut f16_data: Vec<half::f16> =
                data.iter().map(|&v| half::f16::from_f32(v)).collect();
            if !f16_data.len().is_multiple_of(2) {
                f16_data.push(half::f16::from_f32(0.0));
            }
            let f16_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_data.len() * 2)
            };
            queue.write_buffer(f16_buf, (off / 2) as u64, f16_bytes);
        }
    }

    /// Read a node's bytes back to host f32 via a staging buffer +
    /// blocking map. Used by `run()` for output extraction.
    pub fn read_f32(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: NodeId) -> Vec<f32> {
        let off = self.offset(id);
        let len = self.len_of(id);
        let n_elems = len / 4;
        if n_elems == 0 {
            return Vec::new();
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu readback"),
            size: len as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu readback enc"),
        });
        enc.copy_buffer_to_buffer(&self.buffer, off as u64, &staging, 0, len as u64);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        staging.unmap();
        out
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
