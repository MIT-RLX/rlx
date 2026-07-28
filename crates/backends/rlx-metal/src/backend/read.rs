// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `read` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    /// Read one graph output at IR logical length (matches CPU `read_output`).
    ///
    /// Bucket-padded arena slots may be larger than the declared output
    /// shape; embeds decode graphs write the active row at index 0.
    pub(crate) fn read_graph_output_f32(&self, out_idx: usize) -> Vec<f32> {
        let id = self.graph.outputs[out_idx];
        let logical_len = self.output_slots[out_idx].1;
        if rlx_ir::env::flag("RLX_METAL_OUTPUT_TRACE") {
            let off = self.arena.byte_offset(id);
            eprintln!(
                "[metal-out] out_idx={out_idx} id={id:?} byte_off={off:#x} logical_len={logical_len}",
            );
        }
        let full = self.arena.read_as_f32(id);
        if logical_len == 0 || full.len() <= logical_len {
            return full;
        }
        full[..logical_len].to_vec()
    }

    /// Read one row from a row-major graph output (bucketed decode K/V).
    pub fn read_graph_output_row(&self, out_idx: usize, row: usize, row_inner: usize) -> Vec<f32> {
        let id = self.graph.outputs[out_idx];
        let start = row * row_inner;
        let end = start + row_inner;
        if self.arena.dtype(id) == rlx_ir::DType::F32 {
            let slice = self.arena.slice(id);
            assert!(
                end <= slice.len(),
                "read_graph_output_row: out={out_idx} row={row} inner={row_inner} need {end} f32, have {}",
                slice.len()
            );
            slice[start..end].to_vec()
        } else {
            let full = self.arena.read_as_f32(id);
            assert!(
                end <= full.len(),
                "read_graph_output_row: out={out_idx} truncated"
            );
            full[start..end].to_vec()
        }
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name) {
            if out_idx < self.graph.outputs.len() {
                return Some(self.read_graph_output_f32(out_idx));
            }
        }
        if self.gpu_handle_resident.contains(name) {
            if let Some(&id) = self.input_ids.get(name) {
                return Some(self.arena.read_as_f32(id));
            }
        }
        self.gpu_handles.get(name).cloned()
    }
}
