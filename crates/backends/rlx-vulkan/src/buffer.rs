// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The f32-uniform GPU arena. Like rlx-cuda / rlx-wgpu, every tensor is an
//! f32 slot at a byte offset in one contiguous buffer. We allocate the
//! arena as `HOST_VISIBLE | HOST_COHERENT` memory and keep it persistently
//! mapped, so host upload/readback is a plain `memcpy` with no staging
//! buffer or transfer command. (On discrete GPUs a `DEVICE_LOCAL` arena +
//! staging would have higher bandwidth — a documented follow-up; correctness
//! first.)

use crate::device::{VulkanDevice, vulkan_device};
use ash::vk;
use rlx_compile::memory::MemoryPlan;
use rlx_ir::NodeId;
use std::collections::HashMap;

pub struct Arena {
    dev: &'static VulkanDevice,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Total arena size in bytes.
    pub size: usize,
    /// Persistent host mapping of the whole arena.
    mapped: *mut u8,
    /// Per-node byte offset into the arena.
    offsets: HashMap<NodeId, usize>,
    /// Per-node slot byte length (capacity, ≥ used).
    lens: HashMap<NodeId, usize>,
}

// The mapped pointer is only used behind `&mut self` writes / `&self` reads
// on a single executable at a time; the executable itself is not `Sync`.
unsafe impl Send for Arena {}

impl Arena {
    pub fn from_plan(plan: &MemoryPlan) -> Self {
        let dev = vulkan_device().expect("rlx-vulkan: no device for arena");
        let size = plan.arena_size.max(4);
        if std::env::var("RLX_VULKAN_ARENA_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "[rlx-vulkan arena] {:.2} GiB ({} bytes)",
                size as f64 / (1u64 << 30) as f64,
                size
            );
        }

        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { dev.device.create_buffer(&info, None) }.expect("vk create_buffer");

        let req = unsafe { dev.device.get_buffer_memory_requirements(buffer) };
        let mem_type = dev
            .find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("rlx-vulkan: no HOST_VISIBLE|HOST_COHERENT memory type");
        let memory = unsafe {
            dev.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .expect("vk allocate_memory");
        unsafe { dev.device.bind_buffer_memory(buffer, memory, 0) }.expect("vk bind_buffer_memory");

        let mapped = unsafe {
            dev.device
                .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
        }
        .expect("vk map_memory") as *mut u8;
        // A fresh arena is zero-initialized (scratch slots rely on it).
        unsafe { std::ptr::write_bytes(mapped, 0, size) };

        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for (id, slot) in &plan.assignments {
            offsets.insert(*id, slot.offset);
            lens.insert(*id, slot.size);
        }

        Self {
            dev,
            buffer,
            memory,
            size,
            mapped,
            offsets,
            lens,
        }
    }

    #[inline]
    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id)
    }

    /// Byte offset of a node's slot.
    #[inline]
    pub fn byte_offset(&self, id: NodeId) -> usize {
        self.offsets[&id]
    }

    /// f32-element offset of a node's slot (for push constants).
    #[inline]
    pub fn elem_offset(&self, id: NodeId) -> u32 {
        (self.offsets[&id] / 4) as u32
    }

    /// Slot capacity in f32 elements.
    #[inline]
    pub fn slot_elems(&self, id: NodeId) -> usize {
        self.lens.get(&id).copied().unwrap_or(0) / 4
    }

    /// Upload f32 data into a node's slot (clamped to the slot capacity).
    pub fn write_f32(&self, id: NodeId, data: &[f32]) {
        let Some(&off) = self.offsets.get(&id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = data.len().min(cap);
        unsafe {
            let dst = self.mapped.add(off) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    /// Upload raw bytes into a node's slot (for non-f32 packed params).
    pub fn write_bytes(&self, id: NodeId, data: &[u8]) {
        let Some(&off) = self.offsets.get(&id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = data.len().min(cap);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(off), n);
        }
    }

    /// Read `n` f32 elements from a node's slot.
    pub fn read_f32(&self, id: NodeId, n: usize) -> Vec<f32> {
        let Some(&off) = self.offsets.get(&id) else {
            return vec![0.0; n];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = n.min(cap);
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = self.mapped.add(off) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    /// Byte-for-byte copy this arena's contents into `dst` (same plan/size).
    /// Used by `clone_for_cache` to carry params + constants into a twin.
    pub fn copy_into(&self, dst: &Arena) {
        let n = self.size.min(dst.size);
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, dst.mapped, n);
        }
    }

    /// Read `nbytes` raw bytes from a node's slot (packed quant weights).
    pub fn read_bytes(&self, id: NodeId, nbytes: usize) -> Vec<u8> {
        let Some(&off) = self.offsets.get(&id) else {
            return vec![0u8; nbytes];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = nbytes.min(cap);
        let mut out = vec![0u8; nbytes];
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped.add(off), out.as_mut_ptr(), n);
        }
        out
    }

    /// In-arena copy of `n` f32 elements from `src`'s slot into `dst`'s slot,
    /// clamped to both slot capacities. Used by the GPU-resident K/V feed to
    /// fold a decode step's new-token K/V output back into the `past_k_*` input
    /// slot without a host round-trip. The arena is HOST_COHERENT and the GPU
    /// queue is idle by the time this runs (see `submit_and_wait` in `run`), so
    /// a plain mapped `memcpy` is safe and visible to the next dispatch.
    pub fn copy_node_f32_prefix(&self, dst: NodeId, src: NodeId, n: usize) {
        let (Some(&doff), Some(&soff)) = (self.offsets.get(&dst), self.offsets.get(&src)) else {
            return;
        };
        if doff == soff {
            return; // aliased slot — nothing to do
        }
        let dcap = self.lens.get(&dst).copied().unwrap_or(0) / 4;
        let scap = self.lens.get(&src).copied().unwrap_or(0) / 4;
        let n = n.min(dcap).min(scap);
        if n == 0 {
            return;
        }
        unsafe {
            let src_p = self.mapped.add(soff) as *const f32;
            let dst_p = self.mapped.add(doff) as *mut f32;
            std::ptr::copy_nonoverlapping(src_p, dst_p, n);
        }
    }

    /// In-arena copy of `n` f32 elements from `src` slot (starting at element
    /// offset `src_elem`) into `dst` slot (starting at `dst_elem`), clamped to
    /// both slot capacities. Used by the decode K/V feed to drop a single new
    /// token row (output row `upper`) into the resident `past_k_*` slot at the
    /// active row — without disturbing the already-resident prefix.
    pub fn copy_node_f32_range(
        &self,
        dst: NodeId,
        dst_elem: usize,
        src: NodeId,
        src_elem: usize,
        n: usize,
    ) {
        let (Some(&doff), Some(&soff)) = (self.offsets.get(&dst), self.offsets.get(&src)) else {
            return;
        };
        let dcap = self.lens.get(&dst).copied().unwrap_or(0) / 4;
        let scap = self.lens.get(&src).copied().unwrap_or(0) / 4;
        if dst_elem + n > dcap || src_elem + n > scap || n == 0 {
            return;
        }
        let dbyte = doff + dst_elem * 4;
        let sbyte = soff + src_elem * 4;
        if dbyte == sbyte {
            return;
        }
        unsafe {
            let src_p = self.mapped.add(sbyte) as *const f32;
            let dst_p = self.mapped.add(dbyte) as *mut f32;
            std::ptr::copy_nonoverlapping(src_p, dst_p, n);
        }
    }

    /// Read `n` f32 elements starting at an arbitrary f32-element offset.
    pub fn read_f32_at_elem(&self, elem_off: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        let byte_off = elem_off * 4;
        if byte_off + n * 4 > self.size {
            return out;
        }
        unsafe {
            let src = self.mapped.add(byte_off) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            self.dev.device.unmap_memory(self.memory);
            self.dev.device.destroy_buffer(self.buffer, None);
            self.dev.device.free_memory(self.memory, None);
        }
    }
}
