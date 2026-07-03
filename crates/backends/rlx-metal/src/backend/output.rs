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

//! `output` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    /// Read each graph-output's arena region as raw bytes in its
    /// declared dtype. Caller is responsible for ensuring the latest
    /// `run()` / `encode_and_run()` has completed (the caller-facing
    /// methods all wait_until_completed before returning, so this
    /// is true after any of them).
    ///
    /// Used by `MetalExecutableWrapper::run_typed` to avoid the
    /// f32 round-trip on F64 outputs — the f32 path narrows F64
    /// arena bytes to f32 (lossy) before widening them back to F64
    /// bytes for the typed-output contract.
    pub fn output_bytes_per_node(&self) -> Vec<Vec<u8>> {
        let base = self.arena.buffer.contents() as *const u8;
        self.graph
            .outputs
            .iter()
            .map(|&id| {
                let off = if self.arena.has_buffer(id) {
                    self.arena.byte_offset(id)
                } else {
                    0
                };
                let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                let dt = self.graph.node(id).shape.dtype();
                let n_bytes = n_elems * dt.size_bytes();
                unsafe { std::slice::from_raw_parts(base.add(off), n_bytes).to_vec() }
            })
            .collect()
    }


    /// Declared graph-output dtypes, in `graph.outputs` order. Used by
    /// the runtime wrapper's `run_typed` to narrow the f32 outputs back
    /// to F16/BF16/etc. on the way out, mirroring what backends with
    /// native-dtype storage emit.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }


    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.output_slots
    }

}
