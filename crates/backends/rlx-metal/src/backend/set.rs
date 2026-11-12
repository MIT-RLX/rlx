// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    /// Drop the record of which weight packs have already been materialised.
    ///
    /// A `Concat`/`Expand` over `Param`s is invariant across `run()`s, so the
    /// encoder skips re-running it once baked (see `Thunk::Concat`'s
    /// `weight_const`). That is only true while the params underneath hold
    /// still: after any write the pack is stale, and skipping it makes the
    /// consuming GEMM read the previous weights and return a wrong answer with
    /// no error raised anywhere.
    ///
    /// Clearing wholesale rather than tracking which packs consume `name`
    /// costs one step's worth of concat after a re-bind and cannot be wrong.
    /// Weight writes are rare next to decode steps, so the trade is lopsided in
    /// the safe direction.
    fn invalidate_baked_weight_packs(&mut self) {
        self.baked_weight_concats.borrow_mut().clear();
    }

    /// Write a weight that lives as a slice of a pack param (`weight_packs`).
    /// False when `name` is not such a weight.
    fn write_pack_part(&mut self, name: &str, data: &[f32]) -> bool {
        let Some(part) = self.pack_parts.get(name) else {
            return false;
        };
        let Some(&id) = self.param_ids.get(&part.pack) else {
            return true; // the pack itself went unused
        };
        let base = if let Some(slot) = self.weight_slots.get(&id) {
            let buf = self
                .weight_buffer
                .as_ref()
                .expect("weight slot without weight buffer");
            assert_eq!(
                slot.dtype,
                rlx_ir::DType::F32,
                "weight pack {} stored as {:?}",
                part.pack,
                slot.dtype
            );
            unsafe { (buf.contents() as *mut u8).add(slot.offset) }
        } else if let Some(&off) = self.arena.offsets.get(&id) {
            assert_eq!(
                self.arena.dtype(id),
                rlx_ir::DType::F32,
                "weight pack {} not f32",
                part.pack
            );
            unsafe { (self.arena.buffer.contents() as *mut u8).add(off) }
        } else {
            return true;
        };
        // SAFETY: the pack's storage holds `rows * stride` f32s.
        unsafe { super::weight_packs::scatter(part, data, base.cast()) };
        true
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.invalidate_baked_weight_packs();
        if self.write_pack_part(name, data) {
            return;
        }
        if let Some(&id) = self.param_ids.get(name) {
            if let Some(slot) = self.weight_slots.get(&id).copied() {
                self.write_weight_from_f32(slot, data);
            } else if self.arena.has_buffer(id) {
                // Converts to f16 if the param node's dtype is F16.
                self.arena.write_from_f32(id, data);
            }
        }
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        self.invalidate_baked_weight_packs();
        if self.pack_parts.contains_key(name) {
            // Packs are f32, so these are f32 values.
            let values = rlx_ir::bytes::decode_le::<f32>(data);
            self.write_pack_part(name, values.as_ref());
            return;
        }
        if let Some(&id) = self.param_ids.get(name) {
            if let Some(slot) = self.weight_slots.get(&id).copied() {
                self.write_weight_bytes(slot, data);
            } else if self.arena.has_buffer(id) {
                self.arena.write_bytes(id, data);
            }
        }
    }

    /// Incrementally write `data` into a named param starting `byte_offset` bytes
    /// into its storage (raw bytes, no dtype widen). Returns true if written. Used
    /// to upload ONE changed slot of a large packed-expert residency buffer instead
    /// of re-copying the whole buffer every step (`PagedGroupedMoe` paging). Only
    /// the arena-resident path is supported (unified memory, zero-copy); params
    /// parked in a separate weight MTLBuffer return false so the caller re-uploads
    /// whole. No-op (false) for an unknown name.
    pub fn set_param_range(&mut self, name: &str, byte_offset: usize, data: &[u8]) -> bool {
        self.invalidate_baked_weight_packs();
        let Some(&id) = self.param_ids.get(name) else {
            return false;
        };
        if self.weight_slots.contains_key(&id) {
            return false; // separate weight buffer — caller falls back to whole upload
        }
        if self.arena.has_buffer(id) {
            self.arena.write_bytes_at(id, byte_offset, data);
            return true;
        }
        false
    }

    /// Whether param `name`'s weight slot is mapped from another
    /// executable's copy (`crate::shared_weights`): its values are there.
    pub fn param_already_set(&self, name: &str) -> bool {
        let name = self.pack_parts.get(name).map_or(name, |p| p.pack.as_str());
        self.param_ids
            .get(name)
            .is_some_and(|id| self.shared_resident.contains(id))
    }

    /// The dtype a named param is stored in on the device (weight slot or
    /// arena), if the param exists.
    pub fn param_storage_dtype(&self, name: &str) -> Option<rlx_ir::DType> {
        if self.pack_parts.contains_key(name) {
            return Some(rlx_ir::DType::F32);
        }
        let &id = self.param_ids.get(name)?;
        if let Some(slot) = self.weight_slots.get(&id) {
            return Some(slot.dtype);
        }
        self.arena.has_buffer(id).then(|| self.arena.dtype(id))
    }

    /// True when named param storage is native F16 (AMP rewrite or
    /// F16 weight slot). Used by `set_param_typed` to decide whether
    /// F16 host bytes can be copied without an F32 widen.
    pub fn param_storage_is_f16(&self, name: &str) -> bool {
        let Some(&id) = self.param_ids.get(name) else {
            return false;
        };
        if let Some(slot) = self.weight_slots.get(&id) {
            return slot.dtype == rlx_ir::DType::F16;
        }
        self.arena.dtype(id) == rlx_ir::DType::F16
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every thunk in the schedule passes
    /// `Thunk::safe_for_active_extent`; otherwise falls back to
    /// full-extent. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.schedule.rng.write().expect("rng lock") = rng;
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }
}
