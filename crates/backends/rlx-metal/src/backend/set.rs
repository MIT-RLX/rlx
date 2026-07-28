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
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
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
        if let Some(&id) = self.param_ids.get(name) {
            if let Some(slot) = self.weight_slots.get(&id).copied() {
                self.write_weight_bytes(slot, data);
            } else if self.arena.has_buffer(id) {
                self.arena.write_bytes(id, data);
            }
        }
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
