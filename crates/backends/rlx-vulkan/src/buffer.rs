// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The f32-uniform GPU arena. Activations live in one or more storage buffers;
//! large param tensors may be parked in an optional second weight buffer when
//! the activation arena would exceed `maxStorageBufferRange` (~4 GiB on many GPUs).

use crate::device::{VulkanDevice, vulkan_device};
use ash::vk;
use rlx_compile::memory::MemoryPlan;
use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

/// High bit on f32 **element** offsets in push constants → binding 1 (weights).
pub const WEIGHT_ELEM_TAG: u32 = 1 << 31;

/// Bytes reserved at the end of every activation shard for cross-stripe staging.
pub const SHARD_STAGE_RESERVE: usize = 576 * 1024 * 1024;

#[inline]
pub fn is_weight_elem(off: u32) -> bool {
    off & WEIGHT_ELEM_TAG != 0
}

#[inline]
pub fn raw_elem_off(off: u32) -> u32 {
    off & !WEIGHT_ELEM_TAG
}

struct BufferMem {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: usize,
}

/// Re-place unique buffer slots so no slot crosses a shard boundary.
fn snap_plan_to_shards(plan: &mut MemoryPlan, shard_cap: usize) {
    let usable = shard_cap.saturating_sub(SHARD_STAGE_RESERVE).max(256);
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
                "rlx-vulkan: tensor slot at byte {old_off} is {size} bytes \
                 ({:.2} GiB) — exceeds the usable shard of {usable} bytes \
                 ({:.2} GiB = {SHARD_STAGE_RESERVE}-byte stage reserve below the \
                 ~4 GiB storage-buffer cap). Chunk the oversized op at the graph level.",
                size as f64 / (1u64 << 30) as f64,
                usable as f64 / (1u64 << 30) as f64,
            );
        }
        cursor = cursor.div_ceil(16) * 16;
        let local = cursor % shard_cap;
        if local + size > usable {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        remap.insert(old_off, cursor);
        cursor += size;
    }

    for a in plan.assignments.values_mut() {
        a.offset = *remap
            .get(&a.offset)
            .expect("rlx-vulkan: missing slot remap");
    }

    cursor = cursor.div_ceil(16) * 16;
    if tail_extra > 0 {
        let local = cursor % shard_cap;
        if local + tail_extra > usable {
            cursor = cursor.div_ceil(shard_cap) * shard_cap;
        }
        cursor += tail_extra;
    }
    if !cursor.is_multiple_of(shard_cap) {
        cursor = cursor.div_ceil(shard_cap) * shard_cap;
    }
    plan.arena_size = cursor.max(1);
}

pub struct Arena {
    dev: &'static VulkanDevice,
    /// Shard 0 (or the only buffer when unsharded).
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Logical activation arena size in bytes.
    pub size: usize,
    mapped: *mut u8,
    /// Shards 1..N-1 when the logical arena exceeds `maxStorageBufferRange`.
    extra_shards: Vec<BufferMem>,
    /// Bytes per shard (0 = unsharded single buffer of [`Self::size`]).
    pub shard_size: usize,
    /// Optional weight buffer (params); always bound at descriptor binding 1.
    weight: Option<BufferMem>,
    offsets: HashMap<NodeId, usize>,
    lens: HashMap<NodeId, usize>,
    /// Per-node byte offset into the weight buffer (un-tagged).
    weight_offsets: HashMap<NodeId, usize>,
}

unsafe impl Send for Arena {}

impl Arena {
    /// Build from a full memory plan (single activation buffer + dummy weight bind).
    pub fn from_plan(plan: &MemoryPlan) -> Self {
        let dev = vulkan_device().expect("rlx-vulkan: no device for arena");
        let size = plan.arena_size.max(4);
        let max_range = dev.limits.max_storage_buffer_range as usize;
        if size > max_range {
            panic!(
                "rlx-vulkan: arena {:.2} GiB exceeds device maxStorageBufferRange \
                 {:.2} GiB ({}). Use from_plan_split for large graphs.",
                size as f64 / (1u64 << 30) as f64,
                max_range as f64 / (1u64 << 30) as f64,
                dev.name,
            );
        }
        Self::from_plan_inner(dev, plan, None, max_range)
    }

    /// Split params into a dedicated weight buffer. When activations still exceed
    /// `maxStorageBufferRange`, stripe the logical arena across multiple buffers.
    pub fn from_plan_split(graph: &Graph) -> Self {
        let dev = vulkan_device().expect("rlx-vulkan: no device for arena");
        let max_range = dev.limits.max_storage_buffer_range as usize;
        let shard_cap = (max_range / 256) * 256;

        let mut new_plan = rlx_compile::memory::plan_memory_f32_uniform_no_params(graph, 16);
        let walign = 256usize;
        let a = 16usize;
        let mut weight_offsets: HashMap<NodeId, usize> = HashMap::new();
        let mut weight_lens: HashMap<NodeId, usize> = HashMap::new();
        let mut weight_cursor = 0usize;
        let mut tail = new_plan.arena_size;

        for node in graph.nodes() {
            if new_plan.assignments.contains_key(&node.id) {
                continue;
            }
            if rlx_compile::memory::is_pure_view(graph, node) {
                continue;
            }
            let ne = node.shape.num_elements().unwrap_or(0).max(1);
            if matches!(&node.op, Op::Param { .. }) {
                let dt = node.shape.dtype();
                let slot_bytes = if matches!(dt, DType::U8 | DType::I8) {
                    ne * dt.size_bytes()
                } else {
                    ne * 4
                };
                weight_cursor = weight_cursor.div_ceil(walign) * walign;
                weight_offsets.insert(node.id, weight_cursor);
                weight_lens.insert(node.id, slot_bytes);
                weight_cursor += slot_bytes;
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
                    if let Some(&sz) = weight_lens.get(&root) {
                        weight_lens.insert(id, sz.saturating_sub(off));
                    }
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

        if new_plan.arena_size > max_range {
            snap_plan_to_shards(&mut new_plan, shard_cap);
        }

        if weight_cursor > max_range {
            panic!(
                "rlx-vulkan: weight buffer {:.2} GiB exceeds maxStorageBufferRange \
                 {:.2} GiB ({})",
                weight_cursor as f64 / (1u64 << 30) as f64,
                max_range as f64 / (1u64 << 30) as f64,
                dev.name,
            );
        }

        let debug = std::env::var("RLX_VULKAN_ARENA_DEBUG").ok().as_deref() == Some("1")
            || std::env::var("RLX_VULKAN_SHARD_LOG").ok().as_deref() == Some("1");
        if debug {
            eprintln!(
                "[rlx-vulkan split] params={} weight_buf={:.2} GiB act_arena={:.2} GiB device={}",
                weight_offsets.len(),
                weight_cursor as f64 / (1u64 << 30) as f64,
                new_plan.arena_size as f64 / (1u64 << 30) as f64,
                dev.name,
            );
        }

        let weight_buf = if weight_cursor > 0 {
            Some(Self::alloc_buffer(dev, weight_cursor.max(4)))
        } else {
            None
        };
        Self::from_plan_inner(
            dev,
            &new_plan,
            weight_buf.map(|w| (w, weight_offsets, weight_lens)),
            max_range,
        )
    }

    fn from_plan_inner(
        dev: &'static VulkanDevice,
        plan: &MemoryPlan,
        weight: Option<(BufferMem, HashMap<NodeId, usize>, HashMap<NodeId, usize>)>,
        max_range: usize,
    ) -> Self {
        let size = plan.arena_size.max(4);
        let shard_cap = (max_range / 256) * 256;

        let (primary, extra_shards, shard_size) = if size <= max_range {
            (Self::alloc_buffer(dev, size), Vec::new(), 0usize)
        } else {
            for (id, a) in &plan.assignments {
                if a.size > shard_cap {
                    panic!(
                        "rlx-vulkan: node {id:?} slot {} bytes exceeds shard cap {shard_cap}",
                        a.size
                    );
                }
                let end = a.offset.saturating_add(a.size.max(1));
                if a.offset / shard_cap != (end - 1) / shard_cap {
                    panic!(
                        "rlx-vulkan: node {id:?} @{}+{} spans a {shard_cap}-byte shard boundary after snap",
                        a.offset, a.size
                    );
                }
            }
            let n_shards = size.div_ceil(shard_cap);
            if std::env::var("RLX_VULKAN_ARENA_DEBUG").ok().as_deref() == Some("1")
                || std::env::var("RLX_VULKAN_SHARD_LOG").ok().as_deref() == Some("1")
            {
                eprintln!(
                    "[rlx-vulkan] sharded arena: logical={:.3} GiB → {n_shards} × {:.3} GiB shards device={}",
                    size as f64 / (1u64 << 30) as f64,
                    shard_cap as f64 / (1u64 << 30) as f64,
                    dev.name,
                );
            }
            let mut shards = Vec::with_capacity(n_shards);
            for i in 0..n_shards {
                let begin = i * shard_cap;
                let this = (size - begin).min(shard_cap).max(256);
                shards.push(Self::alloc_buffer(dev, this));
            }
            let primary = shards.remove(0);
            (primary, shards, shard_cap)
        };

        if std::env::var("RLX_VULKAN_ARENA_DEBUG").ok().as_deref() == Some("1") && weight.is_none()
        {
            eprintln!(
                "[rlx-vulkan arena] {:.2} GiB ({} bytes) device={}",
                size as f64 / (1u64 << 30) as f64,
                size,
                dev.name,
            );
        }

        let (weight_mem, weight_offsets, extra_lens) = match weight {
            Some((w, offs, lens)) => (Some(w), offs, lens),
            None => (
                Some(Self::alloc_buffer(dev, 4)),
                HashMap::new(),
                HashMap::new(),
            ),
        };

        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for (id, slot) in &plan.assignments {
            offsets.insert(*id, slot.offset);
            lens.insert(*id, slot.size);
        }
        for (id, sz) in extra_lens {
            lens.insert(id, sz);
        }

        Self {
            dev,
            buffer: primary.buffer,
            memory: primary.memory,
            size,
            mapped: primary.mapped,
            extra_shards,
            shard_size,
            weight: weight_mem,
            offsets,
            lens,
            weight_offsets,
        }
    }

    fn alloc_buffer(dev: &VulkanDevice, size: usize) -> BufferMem {
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
        unsafe { std::ptr::write_bytes(mapped, 0, size) };

        BufferMem {
            buffer,
            memory,
            mapped,
            size,
        }
    }

    #[inline]
    pub fn is_sharded(&self) -> bool {
        self.shard_size > 0
    }

    pub fn shard_count(&self) -> usize {
        if !self.is_sharded() {
            1
        } else {
            1 + self.extra_shards.len()
        }
    }

    pub fn act_buffer(&self, shard_idx: usize) -> vk::Buffer {
        if shard_idx == 0 {
            self.buffer
        } else {
            self.extra_shards[shard_idx - 1].buffer
        }
    }

    pub fn shard_stage_off(&self, logical_off: usize) -> usize {
        if !self.is_sharded() {
            return self.size.saturating_sub(SHARD_STAGE_RESERVE);
        }
        let s = self.shard_size;
        let shard_begin = (logical_off / s) * s;
        let shard_end = (shard_begin + s).min(self.size);
        shard_end
            .saturating_sub(SHARD_STAGE_RESERVE)
            .max(shard_begin)
    }

    pub fn resolve_act(&self, global_off: usize) -> (vk::Buffer, *mut u8, usize) {
        assert!(
            global_off < self.size || self.size == 0,
            "rlx-vulkan: resolve_act off={global_off} past arena size={}",
            self.size
        );
        if !self.is_sharded() {
            return (self.buffer, self.mapped, global_off);
        }
        let idx = global_off / self.shard_size;
        let local = global_off % self.shard_size;
        if idx == 0 {
            (self.buffer, self.mapped, local)
        } else {
            let sh = self
                .extra_shards
                .get(idx - 1)
                .unwrap_or_else(|| panic!("rlx-vulkan: shard {idx} out of range"));
            (sh.buffer, sh.mapped, local)
        }
    }

    fn act_ptr(&self, byte_off: usize) -> *mut u8 {
        let (_, mapped, local) = self.resolve_act(byte_off);
        unsafe { mapped.add(local) }
    }

    /// Weight storage buffer for descriptor binding 1 (dummy 4 B when unsplit).
    #[inline]
    pub fn weight_buffer(&self) -> vk::Buffer {
        self.weight
            .as_ref()
            .map(|w| w.buffer)
            .expect("rlx-vulkan: weight buffer always present")
    }

    #[inline]
    pub fn is_split(&self) -> bool {
        !self.weight_offsets.is_empty()
    }

    #[inline]
    pub fn is_weight_node(&self, id: NodeId) -> bool {
        self.weight_offsets.contains_key(&id)
    }

    #[inline]
    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id) || self.weight_offsets.contains_key(&id)
    }

    fn byte_off_raw(&self, id: NodeId) -> Option<(bool, usize)> {
        if let Some(&off) = self.weight_offsets.get(&id) {
            Some((true, off))
        } else {
            self.offsets.get(&id).copied().map(|off| (false, off))
        }
    }

    fn mapped_for_weight(&self, weight: bool) -> *mut u8 {
        if weight {
            self.weight.as_ref().expect("weight mapping").mapped
        } else {
            self.mapped
        }
    }

    #[inline]
    pub fn byte_len(&self, id: NodeId) -> usize {
        self.lens.get(&id).copied().unwrap_or(0)
    }

    /// Byte offset of a node's slot within its buffer (activation or weight).
    #[inline]
    pub fn byte_offset(&self, id: NodeId) -> usize {
        match self.byte_off_raw(id) {
            Some((_, off)) => off,
            None => {
                eprintln!("[rlx-vulkan] warning: no arena offset for {id:?}; using 0");
                0
            }
        }
    }

    /// f32-element offset for push constants (tagged when in weight buffer).
    #[inline]
    pub fn elem_offset(&self, id: NodeId) -> u32 {
        match self.byte_off_raw(id) {
            Some((w, off)) => {
                let elem = (off / 4) as u32;
                if w { elem | WEIGHT_ELEM_TAG } else { elem }
            }
            None => {
                eprintln!("[rlx-vulkan] warning: no arena offset for {id:?}; using 0");
                0
            }
        }
    }

    /// Slot capacity in f32 elements (or byte/4 for packed quant weights).
    #[inline]
    pub fn slot_elems(&self, id: NodeId) -> usize {
        self.lens.get(&id).copied().unwrap_or(0) / 4
    }

    pub fn copy_bytes_range(&self, src_byte: usize, dst_byte: usize, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let mut done = 0usize;
        while done < bytes {
            let g_src = src_byte + done;
            let g_dst = dst_byte + done;
            let room = if self.is_sharded() {
                self.shard_size - (g_src.max(g_dst) % self.shard_size)
            } else {
                bytes - done
            };
            let n = (bytes - done).min(room);
            let (_, src_p, src_local) = self.resolve_act(g_src);
            let (_, dst_p, dst_local) = self.resolve_act(g_dst);
            unsafe {
                std::ptr::copy_nonoverlapping(src_p.add(src_local), dst_p.add(dst_local), n);
            }
            done += n;
        }
    }

    pub fn write_f32(&self, id: NodeId, data: &[f32]) {
        let Some((w, off)) = self.byte_off_raw(id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = data.len().min(cap);
        unsafe {
            let dst = if w {
                self.mapped_for_weight(true).add(off) as *mut f32
            } else {
                self.act_ptr(off) as *mut f32
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    pub fn write_bytes(&self, id: NodeId, data: &[u8]) {
        let Some((w, off)) = self.byte_off_raw(id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = data.len().min(cap);
        unsafe {
            let dst = if w {
                self.mapped_for_weight(true).add(off)
            } else {
                self.act_ptr(off)
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    pub fn read_f32(&self, id: NodeId, n: usize) -> Vec<f32> {
        let Some((w, off)) = self.byte_off_raw(id) else {
            return vec![0.0; n];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = n.min(cap);
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = if w {
                self.mapped_for_weight(true).add(off) as *const f32
            } else {
                self.act_ptr(off) as *const f32
            };
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    pub fn read_bytes_at(&self, byte_off: usize, len: usize) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; len];
        let mut done = 0usize;
        while done < len {
            let g = byte_off + done;
            let room = if self.is_sharded() {
                self.shard_size - (g % self.shard_size)
            } else {
                len - done
            };
            let n = (len - done).min(room);
            let (_, src, local) = self.resolve_act(g);
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(local), out.as_mut_ptr().add(done), n);
            }
            done += n;
        }
        out
    }

    pub fn write_bytes_at(&self, byte_off: usize, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let mut off = 0usize;
        while off < data.len() {
            let g = byte_off + off;
            let room = if self.is_sharded() {
                self.shard_size - (g % self.shard_size)
            } else {
                data.len() - off
            };
            let n = (data.len() - off).min(room);
            let (_, dst, local) = self.resolve_act(g);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().add(off), dst.add(local), n);
            }
            off += n;
        }
    }

    pub fn copy_into(&self, dst: &Arena) {
        let n = self.size.min(dst.size);
        const CHUNK: usize = 8 * 1024 * 1024;
        let mut off = 0usize;
        while off < n {
            let k = (n - off).min(CHUNK);
            let bytes = self.read_bytes_at(off, k);
            dst.write_bytes_at(off, &bytes);
            off += k;
        }
        if let (Some(sw), Some(dw)) = (&self.weight, &dst.weight) {
            let wn = sw.size.min(dw.size);
            unsafe {
                std::ptr::copy_nonoverlapping(sw.mapped, dw.mapped, wn);
            }
        }
    }

    pub fn read_bytes(&self, id: NodeId, nbytes: usize) -> Vec<u8> {
        let Some((w, off)) = self.byte_off_raw(id) else {
            return vec![0u8; nbytes];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = nbytes.min(cap);
        let mut out = vec![0u8; n];
        unsafe {
            let src = if w {
                self.mapped_for_weight(true).add(off)
            } else {
                self.act_ptr(off)
            };
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    #[inline]
    pub fn mapped_ptr(&self) -> *mut u8 {
        self.mapped
    }

    pub fn sync_host_after_gpu(&self) {
        self.dev
            .invalidate_mapped(self.memory, 0, self.primary_bytes() as u64);
        for sh in &self.extra_shards {
            self.dev.invalidate_mapped(sh.memory, 0, sh.size as u64);
        }
        if let Some(w) = &self.weight {
            self.dev.invalidate_mapped(w.memory, 0, w.size as u64);
        }
    }

    pub fn sync_gpu_after_host(&self) {
        self.dev
            .flush_mapped(self.memory, 0, self.primary_bytes() as u64);
        for sh in &self.extra_shards {
            self.dev.flush_mapped(sh.memory, 0, sh.size as u64);
        }
        if let Some(w) = &self.weight {
            self.dev.flush_mapped(w.memory, 0, w.size as u64);
        }
    }

    fn primary_bytes(&self) -> usize {
        if self.is_sharded() {
            self.shard_size
        } else {
            self.size
        }
    }

    pub fn copy_node_f32_prefix(&self, dst: NodeId, src: NodeId, n: usize) {
        let (Some((dw, doff)), Some((sw, soff))) = (self.byte_off_raw(dst), self.byte_off_raw(src))
        else {
            return;
        };
        if !dw && !sw && doff == soff {
            return;
        }
        let dcap = self.lens.get(&dst).copied().unwrap_or(0) / 4;
        let scap = self.lens.get(&src).copied().unwrap_or(0) / 4;
        let n = n.min(dcap).min(scap);
        if n == 0 {
            return;
        }
        let bytes = n * 4;
        if dw || sw {
            // Weight involved — use separate paths.
            unsafe {
                let src_p = self.mapped_for_weight(sw).add(soff) as *const f32;
                let dst_p = if dw {
                    self.mapped_for_weight(true).add(doff) as *mut f32
                } else {
                    self.act_ptr(doff) as *mut f32
                };
                std::ptr::copy_nonoverlapping(src_p, dst_p, n);
            }
        } else {
            self.copy_bytes_range(soff, doff, bytes);
        }
    }

    pub fn copy_node_f32_range(
        &self,
        dst: NodeId,
        dst_elem: usize,
        src: NodeId,
        src_elem: usize,
        n: usize,
    ) {
        let (Some((dw, doff)), Some((sw, soff))) = (self.byte_off_raw(dst), self.byte_off_raw(src))
        else {
            return;
        };
        let dcap = self.lens.get(&dst).copied().unwrap_or(0) / 4;
        let scap = self.lens.get(&src).copied().unwrap_or(0) / 4;
        if dst_elem + n > dcap || src_elem + n > scap || n == 0 {
            return;
        }
        let dbyte = doff + dst_elem * 4;
        let sbyte = soff + src_elem * 4;
        if !dw && !sw && dbyte == sbyte {
            return;
        }
        let bytes = n * 4;
        if dw || sw {
            unsafe {
                let src_p = self.mapped_for_weight(sw).add(sbyte) as *const f32;
                let dst_p = if dw {
                    self.mapped_for_weight(true).add(dbyte) as *mut f32
                } else {
                    self.act_ptr(dbyte) as *mut f32
                };
                std::ptr::copy_nonoverlapping(src_p, dst_p, n);
            }
        } else {
            self.copy_bytes_range(sbyte, dbyte, bytes);
        }
    }

    pub fn read_f32_at_elem(&self, elem_off: u32, n: usize) -> Vec<f32> {
        let w = is_weight_elem(elem_off);
        let byte_off = raw_elem_off(elem_off) as usize * 4;
        let mut out = vec![0.0f32; n];
        if w {
            let buf_size = self.weight.as_ref().map(|x| x.size).unwrap_or(0);
            if byte_off + n * 4 > buf_size {
                return out;
            }
            unsafe {
                let src = self.mapped_for_weight(true).add(byte_off) as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
            }
        } else if byte_off + n * 4 > self.size {
            return out;
        } else {
            unsafe {
                let src = self.act_ptr(byte_off) as *const f32;
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
            }
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
            for sh in self.extra_shards.drain(..) {
                self.dev.device.unmap_memory(sh.memory);
                self.dev.device.destroy_buffer(sh.buffer, None);
                self.dev.device.free_memory(sh.memory, None);
            }
            if let Some(w) = self.weight.take() {
                self.dev.device.unmap_memory(w.memory);
                self.dev.device.destroy_buffer(w.buffer, None);
                self.dev.device.free_memory(w.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod shard_pack_tests {
    use super::*;
    use rlx_ir::NodeId;

    #[test]
    fn snap_plan_keeps_slots_out_of_stage_reserve() {
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
        assert!(plan.arena_size.is_multiple_of(shard_cap));
    }
}
