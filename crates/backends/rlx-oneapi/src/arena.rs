// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The f32-uniform USM-shared GPU arena for the Level Zero dispatch path. Like
//! rlx-vulkan's host-visible arena, every tensor is an f32 slot at a byte
//! offset in one contiguous buffer; here the buffer is a single
//! `zeMemAllocShared` allocation, which is CPU-dereferenceable on Intel's
//! shared-memory GPUs, so host upload/readback and the CPU host-fallback are
//! plain pointer writes with no staging. Only constructed when a live device is
//! present (the dev-box path uses the value-map interpreter in `backend.rs`).

use crate::device::{OneApiDevice, oneapi_device};
use rlx_compile::memory::MemoryPlan;
use rlx_ir::NodeId;
use std::collections::HashMap;

pub struct Arena {
    dev: &'static OneApiDevice,
    base: *mut std::ffi::c_void,
    pub size: usize,
    offsets: HashMap<NodeId, usize>,
    lens: HashMap<NodeId, usize>,
}

// The USM pointer is only used behind `&mut self` writes / `&self` reads on a
// single executable at a time; the executable itself is not `Sync`.
unsafe impl Send for Arena {}

impl Arena {
    pub fn from_plan(plan: &MemoryPlan) -> Result<Self, String> {
        let dev = oneapi_device().ok_or("rlx-oneapi: no device for arena")?;
        let size = plan.arena_size.max(4);
        let base = dev.alloc_shared(size)?;
        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for (id, slot) in &plan.assignments {
            offsets.insert(*id, slot.offset);
            lens.insert(*id, slot.size);
        }
        Ok(Self {
            dev,
            base,
            size,
            offsets,
            lens,
        })
    }

    #[inline]
    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id)
    }

    /// Byte offset of a node's slot in the USM arena.
    #[inline]
    pub fn byte_offset(&self, id: NodeId) -> usize {
        self.offsets[&id]
    }

    /// Element offset (f32) of a node's slot — what the kernels index by.
    #[inline]
    pub fn elem_offset(&self, id: NodeId) -> u32 {
        (self.offsets[&id] / 4) as u32
    }

    /// Raw USM base pointer (kernel argument 0).
    #[inline]
    pub fn base_ptr(&self) -> *mut std::ffi::c_void {
        self.base
    }

    pub fn write_f32(&self, id: NodeId, data: &[f32]) {
        let Some(&off) = self.offsets.get(&id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = data.len().min(cap);
        unsafe {
            let dst = (self.base as *mut u8).add(off) as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    pub fn write_bytes(&self, id: NodeId, data: &[u8]) {
        let Some(&off) = self.offsets.get(&id) else {
            return;
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = data.len().min(cap);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), (self.base as *mut u8).add(off), n);
        }
    }

    pub fn read_f32(&self, id: NodeId, n: usize) -> Vec<f32> {
        let Some(&off) = self.offsets.get(&id) else {
            return vec![0.0; n];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0) / 4;
        let n = n.min(cap);
        let mut out = vec![0.0f32; n];
        unsafe {
            let src = (self.base as *const u8).add(off) as *const f32;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
        }
        out
    }

    pub fn read_bytes(&self, id: NodeId, nbytes: usize) -> Vec<u8> {
        let Some(&off) = self.offsets.get(&id) else {
            return vec![0u8; nbytes];
        };
        let cap = self.lens.get(&id).copied().unwrap_or(0);
        let n = nbytes.min(cap);
        let mut out = vec![0u8; nbytes];
        unsafe {
            std::ptr::copy_nonoverlapping((self.base as *const u8).add(off), out.as_mut_ptr(), n);
        }
        out
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let _ = &self.dev;
        self.dev.free(self.base);
    }
}
