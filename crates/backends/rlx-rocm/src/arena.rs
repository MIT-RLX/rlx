// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP device-memory arena.
//!
//! Mirrors `rlx-cuda::arena` exactly: one big f32 device buffer for
//! activations + un-promoted params, plus an optional u16 side-buffer
//! for f16/bf16 weights (the half-arena consumer for mixed-precision
//! matmul). Reshape and Cast alias the input slot.

use std::collections::HashMap;
use std::sync::Arc;

use rlx_ir::{DType, Graph, NodeId, Op};
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

/// Cast op ids for the shared unary kernel (`unary.cu` cases 100–106). Same
/// kernel source as rlx-cuda; kept in sync with rlx-vulkan / rlx-oneapi.
pub(crate) const CAST_F32_TO_I8: u32 = 100;
pub(crate) const CAST_F32_TO_I16: u32 = 101;
pub(crate) const CAST_F32_TO_I32: u32 = 102;
pub(crate) const CAST_F32_TO_I64: u32 = 103;
pub(crate) const CAST_F32_TO_U8: u32 = 104;
pub(crate) const CAST_F32_TO_U32: u32 = 105;
pub(crate) const CAST_TO_BOOL: u32 = 106;

/// Number of f32 lanes a node occupies in the f32-uniform arena's host-readback
/// view. Complex is simulated on f32 lanes (C64 = 2 lanes/elem, C128 = 4); every
/// other dtype is one f32 lane per element (I64/Bool/… widen to a single lane).
/// Used to size + read the host staging slot so a complex output reads back ALL
/// its lanes, not just `num_elements` (which would truncate to the real parts).
/// Mirrors `rlx-cuda::arena::arena_lane_count`.
pub(crate) fn arena_lane_count(shape: &rlx_ir::Shape) -> usize {
    let elems = shape.num_elements().unwrap_or(0);
    match shape.dtype() {
        DType::C64 => elems * 2,
        DType::C128 => elems * 4,
        _ => elems,
    }
}

/// How an `Op::Cast` lowers on the f32-uniform arena.
pub(crate) enum CastLower {
    /// Value-preserving relabel — alias the input slot. Covers same-dtype,
    /// int→float, float→float (F16/BF16/F64 are all f32-stored here), int→int,
    /// and bool→int/float.
    Identity,
    /// A real elementwise conversion via the unary kernel with this op id
    /// (float→int trunc-saturate, or →Bool `x != 0`).
    Kernel(u32),
    /// A complex cast (real↔C64, real↔C128, C64↔C128) — pure f32-lane moves via
    /// the standalone `complex_cast` kernel. Carries the mode (0..5, see
    /// `complex_cast.cu`). Needs its own (complex-sized) slot, not an alias.
    Complex(u32),
    /// Not representable in an f32 arena (F64 has no lane storage) — reject.
    Reject,
}

/// Classify a `Cast(src → dst)` on the f32-uniform arena. float→int truncates
/// toward zero + saturates (Rust `as` / rlx-cpu); →Bool is `x != 0`. F16/BF16/
/// F64 are demoted to f32 storage so casts to/from them are identity relabels;
/// complex (C64/C128) conversions are simulated on f32 lanes; only a complex
/// cast touching F64 (which has no f32-lane storage here) is rejected.
pub(crate) fn classify_cast(src: DType, dst: DType) -> CastLower {
    if src == dst {
        return CastLower::Identity; // pure relabel (also covers C64→C64 / C128→C128)
    }
    // Complex casts (real↔C64, real↔C128, C64↔C128) are pure f32-lane moves on
    // the simulated-complex arena (mirrors rlx-cuda). F64 is the one component
    // type with no f32-lane storage here, so a complex cast touching F64 (real
    // side) is still rejected.
    if src.is_complex() || dst.is_complex() {
        if src == DType::F64 || dst == DType::F64 {
            return CastLower::Reject;
        }
        let mode = match (src, dst) {
            (s, DType::C64) if !s.is_complex() => 0,  // real → C64
            (DType::C64, d) if !d.is_complex() => 1,  // C64 → real
            (s, DType::C128) if !s.is_complex() => 2, // real → C128
            (DType::C128, d) if !d.is_complex() => 3, // C128 → real
            (DType::C64, DType::C128) => 4,
            (DType::C128, DType::C64) => 5,
            _ => return CastLower::Reject,
        };
        return CastLower::Complex(mode);
    }
    if dst == DType::Bool {
        return CastLower::Kernel(CAST_TO_BOOL);
    }
    if src.is_float() && dst.is_int() {
        return CastLower::Kernel(match dst {
            DType::I8 => CAST_F32_TO_I8,
            DType::I16 => CAST_F32_TO_I16,
            DType::I32 => CAST_F32_TO_I32,
            DType::I64 => CAST_F32_TO_I64,
            DType::U8 => CAST_F32_TO_U8,
            DType::U32 => CAST_F32_TO_U32,
            _ => unreachable!("is_int() covers all integer dtypes"),
        });
    }
    CastLower::Identity
}

/// True when a Cast needs its own slot + a conversion kernel (float→int /
/// →Bool) or must be rejected — i.e. not an identity relabel.
pub(crate) fn cast_is_kernel(graph: &Graph, node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::Cast { to } => !matches!(
            classify_cast(graph.node(node.inputs[0]).shape.dtype(), *to),
            CastLower::Identity
        ),
        _ => false,
    }
}

pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        // Reshape / StopGradient, and identity Casts, alias the input slot.
        // float→int / →Bool casts get their own slot + a conversion kernel.
        let is_view = match &node.op {
            Op::Reshape { .. } | Op::StopGradient => true,
            Op::Cast { .. } => !cast_is_kernel(graph, node),
            _ => false,
        };
        if is_view
            && let Some(in_id) = node.inputs.first()
            && let Some(slot) = assignments.get(in_id)
        {
            let aliased = slot.clone();
            assignments.insert(node.id, aliased);
            schedule.push(node.id);
            continue;
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        // A float→int / →Bool Cast writes f32 lanes via the unary kernel, so its
        // slot is f32-sized even when the dst dtype (I8/U8/Bool) would byte-pack.
        // A COMPLEX cast, however, produces a genuine multi-lane (C64=2, C128=4)
        // output — it must keep its complex-sized slot (elems*8 / elems*16), not
        // the elems*4 single-lane sizing, so guard the kernel branch with
        // `!is_complex()` (mirrors rlx-cuda).
        let bytes = if cast_is_kernel(graph, node) && !node.shape.dtype().is_complex() {
            elems * 4
        } else {
            match node.shape.dtype() {
                // U8/I8 byte-pack (quantized weight storage). Bool is NOT
                // byte-packed: it is a compare/mask output written as f32
                // (1.0/0.0) into the f32-uniform arena, so it needs the full
                // `elems * 4` (a compare kernel writes — and the readback reads —
                // f32 lanes; byte-sizing it overruns the slot). Mirrors rlx-cuda.
                rlx_ir::DType::U8 | rlx_ir::DType::I8 => elems,
                // Complex simulates on f32 lanes: C64 = 2 lanes/elem (8 B), C128 =
                // 4 lanes/elem (16 B, df64). Sizing these elems*4 would truncate
                // the imaginary / low lanes.
                rlx_ir::DType::C64 => elems * 8,
                rlx_ir::DType::C128 => elems * 16,
                _ => elems * 4,
            }
        };
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
