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

//! HIP device-memory arena.
//!
//! Mirrors `rlx-cuda::arena` exactly: one big f32 device buffer for
//! activations + un-promoted params, plus an optional u16 side-buffer
//! for f16/bf16 weights (the half-arena consumer for mixed-precision
//! matmul). Reshape and Cast alias the input slot.

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory::{BufferSlot, MemoryPlan};

use crate::device::RocmContext;
use crate::hip::HipBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfDtype {
    F16,
    Bf16,
}

pub struct Arena {
    pub buffer: HipBuffer<f32>,
    pub offsets: HashMap<NodeId, usize>,
    pub lens: HashMap<NodeId, usize>,
    pub size: usize,

    pub half_buffer: Option<HipBuffer<u16>>,
    pub half_offsets: HashMap<NodeId, (usize, HalfDtype)>,
    pub half_by_f32_off: HashMap<u32, (usize, HalfDtype)>,
    pub half_size: usize,
}

pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
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
    pub fn from_plan(ctx: &Arc<RocmContext>, plan: &MemoryPlan) -> Self {
        let n_f32 = plan.arena_size.div_ceil(4);
        let buffer = HipBuffer::<f32>::alloc_zeros(&ctx.runtime, n_f32.max(4))
            .expect("rlx-rocm: device allocation failed");
        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for (id, slot) in &plan.assignments {
            offsets.insert(*id, slot.offset);
            lens.insert(*id, slot.size);
        }
        Self {
            buffer,
            offsets,
            lens,
            size: plan.arena_size,
            half_buffer: None,
            half_offsets: HashMap::new(),
            half_by_f32_off: HashMap::new(),
            half_size: 0,
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
    pub fn set_actual_len(&mut self, id: NodeId, bytes: usize) {
        self.lens.insert(id, bytes);
    }

    /// Reserve a slot in the half-precision side-buffer; allocates /
    /// grows the underlying HipBuffer as needed.
    pub fn register_half_param(
        &mut self,
        ctx: &Arc<RocmContext>,
        id: NodeId,
        f32_off: u32,
        n_elems: usize,
        dtype: HalfDtype,
    ) -> usize {
        let off = self.half_size;
        self.half_size += n_elems;
        self.half_offsets.insert(id, (off, dtype));
        self.half_by_f32_off.insert(f32_off, (off, dtype));
        let new_buf = HipBuffer::<u16>::alloc_zeros(&ctx.runtime, self.half_size.max(4))
            .expect("rlx-rocm: half-arena allocation failed");
        // (We don't preserve the previous half_buffer's contents on
        // resize — simpler than rlx-cuda's dtod copy and matches our
        // "set_param_half is a load-time op, not a hot-path op"
        // assumption. Could be tightened later.)
        self.half_buffer = Some(new_buf);
        off
    }

    pub fn is_half(&self, id: NodeId) -> bool {
        self.half_offsets.contains_key(&id)
    }

    pub fn half_off(&self, id: NodeId) -> Option<(usize, HalfDtype)> {
        self.half_offsets.get(&id).copied()
    }
}
