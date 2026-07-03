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

//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            // Converts to f16 if the param node's dtype is F16.
            self.arena.write_from_f32(id, data);
        }
    }


    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            self.arena.write_bytes(id, data);
        }
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
