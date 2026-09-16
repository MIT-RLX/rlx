// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP device-memory arena.
//!
//! Mirrors `rlx-cuda::arena` exactly: one big f32 device buffer for
//! activations + un-promoted params, plus an optional u16 side-buffer
//! for f16/bf16 weights (the half-arena consumer for mixed-precision
//! matmul). Reshape and Cast alias the input slot.

use std::collections::{HashMap, HashSet};
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

/// Set `RLX_ROCM_ARENA_NO_REUSE=1` to fall back to one permanent slot per
/// tensor — the pre-liveness behaviour, kept for bisecting a suspected
/// aliasing bug against a plan that cannot possibly have one.
fn arena_reuse_enabled() -> bool {
    !matches!(
        rlx_ir::env::var("RLX_ROCM_ARENA_NO_REUSE").as_deref(),
        Some("1")
    )
}

/// Nodes whose slot must survive the whole graph, and every later run of it.
///
/// Inputs, parameters and constants are uploaded once and read on every
/// execution, so their last *use* inside one run says nothing about when they
/// stop being needed. Handing a dead-looking parameter slot to an intermediate
/// works perfectly on the first run and returns garbage on the second, which is
/// the worst shape a bug can take. Graph outputs are pinned because the host
/// reads them after the run has finished.
fn pinned(graph: &Graph) -> HashSet<NodeId> {
    let mut keep: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    for node in graph.nodes() {
        if matches!(
            node.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        ) {
            keep.insert(node.id);
        }
    }
    keep
}

pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    plan_f32_uniform_with(graph, align, arena_reuse_enabled())
}

/// [`plan_f32_uniform`] with the reuse decision passed in rather than read from
/// the environment — so a test can exercise both plans without mutating global
/// state that every other test in the process shares.
pub fn plan_f32_uniform_with(graph: &Graph, align: usize, reuse: bool) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    // Sizes and aliasing first, offsets second. Splitting the two is what lets
    // a dead tensor's slot be handed to a later one without touching any of the
    // sizing or view rules below, which are the parts that are easy to get
    // subtly wrong and are shared with the executor's Nop predicate.
    let mut sized: Vec<(NodeId, usize)> = Vec::with_capacity(graph.nodes().len());
    let mut owns: HashSet<NodeId> = HashSet::new();
    let mut alias_of: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        // Reshape / StopGradient, and identity Casts, alias the input slot.
        // float→int / →Bool casts get their own slot + a conversion kernel.
        //
        // `Op::KvAppend` is aliased too — its output IS the cache (input 0), per
        // the shared planner's `pure_view_offset`. Aliased is not the same as
        // no-op: it still emits a `Step::KvAppend` row write. Leaving it out
        // here hands the output a fresh uninitialised slot, so the one written
        // row lands in a buffer of garbage and the model emits a single token
        // forever with no error (that exact bug, on rlx-cuda).
        let is_view = match &node.op {
            Op::Reshape { .. } | Op::StopGradient | Op::KvAppend { .. } => true,
            Op::Cast { .. } => !cast_is_kernel(graph, node),
            _ => false,
        };
        if is_view
            && let Some(in_id) = node.inputs.first()
            && (alias_of.contains_key(in_id) || owns.contains(in_id))
        {
            // Follow the chain to the tensor that actually owns the storage, so
            // a Reshape of a Reshape resolves in one hop later on.
            let root = *alias_of.get(in_id).unwrap_or(in_id);
            alias_of.insert(node.id, root);
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
        sized.push((node.id, aligned));
        owns.insert(node.id);
        schedule.push(node.id);
    }

    // ── Offsets ─────────────────────────────────────────────────────────────
    //
    // Without reuse this arena is the sum of every tensor in the graph. On a
    // SynthMorph-sized detector that came to 6,283,200,528 B — past the 4 GiB
    // this backend rejects outright, because 479 of its shared kernels still
    // take `unsigned int` offsets and would wrap. rlx-cuda, which has had
    // liveness reuse all along, planned the identical graph in 2,794,063,888 B.
    //
    // With reuse this planner now lands on 2,794,063,888 B exactly — the same
    // number, to the byte — and the graph runs, bit-identical to onnxruntime
    // across all 174,456,832 outputs. Widening 479 kernel signatures would have
    // been the other way to fix this; not needing to is the point.
    // Pinning has to be closed under aliasing. A graph output that is a
    // Reshape owns no storage of its own — the tensor it views does. Pinning
    // only the view leaves that tensor recyclable, and the host then reads back
    // whatever was written over it.
    let mut keep = pinned(graph);
    for id in keep.clone() {
        if let Some(root) = alias_of.get(&id) {
            keep.insert(*root);
        }
    }
    let size_of: HashMap<NodeId, usize> = sized.iter().copied().collect();
    let dump = rlx_ir::env::var("RLX_ROCM_ARENA_DUMP").as_deref() == Some("1");

    // Last position in the schedule at which each owning tensor is read. An
    // alias reads through to its root, so it extends the root's life, not its
    // own — miss that and a Reshape's storage is recycled while the Reshape is
    // still about to be read.
    let mut death: HashMap<NodeId, usize> = HashMap::new();
    for (step, id) in schedule.iter().enumerate() {
        for input in &graph.node(*id).inputs {
            let root = *alias_of.get(input).unwrap_or(input);
            death
                .entry(root)
                .and_modify(|d| *d = (*d).max(step))
                .or_insert(step);
        }
        if let Some(root) = alias_of.get(id) {
            death
                .entry(*root)
                .and_modify(|d| *d = (*d).max(step))
                .or_insert(step);
        }
    }

    // Static weight packs live to graph end.
    //
    // A `Concat`/`Cast`/`Expand` over `Param`s — the fused QKV and gate+up
    // weights the matmul-fusion passes build — is invariant across `run()`s, so
    // the backend materialises it once and skips it after (see the marking in
    // `backend/compile.rs`). That is only sound if nothing else takes its slot,
    // which liveness reuse will happily do: a pack dies at its consuming GEMM
    // and the next activation lands on top of it.
    //
    // `rlx-compile`'s planner pins these via `extend_static_weight_pack_liveness`,
    // but this is ROCm's own planner and never went through it. The identical gap
    // on CUDA meant 0 of 140 pack steps qualified on a 28-layer decode graph;
    // pinning took it to 140 and the step went 7.16 -> 2.99 ms (2.39x).
    //
    // Costs the packs' bytes in residency, which is inherent: skipping the
    // recompute means keeping the result.
    // Conditional on the same switch that controls the skip: pinning without
    // skipping is the worst of both worlds — it pays the residency and buys
    // nothing. `RLX_STATIC_WEIGHT_PACK=0` gives back the ~940 MB (measured on a
    // 28-layer decode graph, roughly +67% arena) along with the recompute.
    if rlx_opt::memory::static_weight_pack_skip_enabled() {
        let last = schedule.len();
        let mut memo: HashMap<NodeId, bool> = HashMap::new();
        for node in graph.nodes() {
            if matches!(
                node.op,
                Op::Param { .. } | Op::Constant { .. } | Op::Input { .. }
            ) {
                continue;
            }
            if !rlx_opt::memory::is_static_weight_tensor(graph, node.id, &mut memo) {
                continue;
            }
            let root = *alias_of.get(&node.id).unwrap_or(&node.id);
            death.entry(root).and_modify(|d| *d = last).or_insert(last);
        }
    }

    // Free extents, kept sorted by offset so adjacent ones can be merged. A
    // first-fit over this is enough: the allocation order is the schedule, so
    // the fragmentation a smarter fit would avoid mostly does not arise.
    let mut free: Vec<(usize, usize)> = Vec::new();
    let mut retiring: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();

    for (step, id) in schedule.iter().enumerate() {
        for extent in retiring.remove(&step).unwrap_or_default() {
            free.push(extent);
        }
        if !free.is_empty() {
            free.sort_unstable();
            let mut merged: Vec<(usize, usize)> = Vec::with_capacity(free.len());
            for (off, len) in free.drain(..) {
                match merged.last_mut() {
                    Some((p_off, p_len)) if *p_off + *p_len == off => *p_len += len,
                    _ => merged.push((off, len)),
                }
            }
            free = merged;
        }
        let Some(&size) = size_of.get(id) else {
            continue; // an alias: it borrows its root's slot, allocated already
        };

        // A tensor whose contents are established OUTSIDE the schedule — an
        // input, a parameter, a constant — must own storage no scheduled node
        // ever writes to. Being pinned keeps its slot off the free list, but
        // that alone does not stop it from *taking* a recycled one, and a
        // recycled slot is by definition a slot some earlier node writes on
        // every run. That is how `__leaky_alpha` landed inside the Conv3d
        // output here: the graph ran, the convolution overwrote the slope, and
        // every negative activation came out scaled by whatever was there.
        // Positive activations were untouched, so the result looked plausible.
        let fresh_only = matches!(
            graph.node(*id).op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        );
        let mut recycled = false;
        let offset = match (reuse && !fresh_only)
            .then(|| free.iter().position(|&(_, len)| len >= size))
            .flatten()
        {
            Some(idx) => {
                recycled = true;
                let (off, len) = free[idx];
                if len == size {
                    free.remove(idx);
                } else {
                    free[idx] = (off + size, len - size);
                }
                off
            }
            None => {
                let off = cursor.div_ceil(align) * align;
                cursor = off + size;
                off
            }
        };
        assignments.insert(*id, BufferSlot { offset, size });
        if dump {
            let node = graph.node(*id);
            eprintln!(
                "arena[{step:4}] {:<28} off {offset:>12} size {size:>12} {}{}",
                format!("{:?}", node.op)
                    .chars()
                    .take(28)
                    .collect::<String>(),
                if recycled { "REUSED" } else { "fresh " },
                if keep.contains(id) { " pinned" } else { "" },
            );
        }

        // Schedule the slot's return for the step after its last read. Pinned
        // tensors never come back.
        if reuse && !keep.contains(id) {
            let end = death.get(id).copied().unwrap_or(step);
            if end < schedule.len() {
                retiring.entry(end + 1).or_default().push((offset, size));
            }
        }
    }

    // Aliases resolve to their root's slot once every owner has one.
    for (id, root) in &alias_of {
        if let Some(slot) = assignments.get(root).cloned() {
            assignments.insert(*id, slot);
        }
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

#[cfg(test)]
mod plan_tests {
    use super::*;
    use rlx_ir::op::Activation;
    use rlx_ir::{DType, Graph, Op, Shape};

    /// A chain of `n` elementwise steps — the shape where reuse pays, because
    /// each intermediate dies the moment the next one is produced.
    fn chain(n: usize, elems: usize) -> Graph {
        let mut g = Graph::new("chain");
        let shape = Shape::new(&[elems], DType::F32);
        let mut cur = g.input("x", shape.clone());
        for _ in 0..n {
            cur = g.add_node(Op::Activation(Activation::Relu), vec![cur], shape.clone());
        }
        g.set_outputs(vec![cur]);
        g
    }

    /// Every node's live interval, in schedule positions, following aliases to
    /// the tensor that owns the storage.
    fn live_ranges(g: &Graph, plan: &MemoryPlan) -> Vec<(NodeId, usize, usize)> {
        let pos: HashMap<NodeId, usize> = plan
            .schedule
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let mut out = Vec::new();
        for (i, id) in plan.schedule.iter().enumerate() {
            let mut last = i;
            for (j, other) in plan.schedule.iter().enumerate() {
                if g.node(*other).inputs.contains(id) {
                    last = last.max(j);
                }
            }
            if g.outputs.contains(id) {
                last = plan.schedule.len();
            }
            let _ = &pos;
            out.push((*id, i, last));
        }
        out
    }

    #[test]
    fn no_two_simultaneously_live_tensors_share_storage() {
        // The invariant the whole change rests on. Checked against the plan
        // rather than against a size, because a plan that merely looks smaller
        // is exactly what a broken allocator also produces.
        let g = chain(40, 4096);
        let plan = plan_f32_uniform(&g, 16);
        let ranges = live_ranges(&g, &plan);
        for (a, a0, a1) in &ranges {
            for (b, b0, b1) in &ranges {
                if a >= b || a0.max(b0) > a1.min(b1) {
                    continue; // same node, or lives do not overlap
                }
                let (sa, sb) = (&plan.assignments[a], &plan.assignments[b]);
                let overlap = sa.offset < sb.offset + sb.size && sb.offset < sa.offset + sa.size;
                assert!(
                    !overlap,
                    "{a:?} at {}+{} overlaps live {b:?} at {}+{}",
                    sa.offset, sa.size, sb.offset, sb.size
                );
            }
        }
    }

    #[test]
    fn a_long_chain_no_longer_costs_one_slot_per_step() {
        // 6.28 GB against rlx-cuda's 2.79 GB on the same SynthMorph graph was
        // this, and this backend rejects anything past 4 GiB outright.
        let g = chain(64, 65_536);
        let reused = plan_f32_uniform(&g, 16).arena_size;
        let every = g.nodes().len() * 65_536 * 4;
        assert!(
            reused < every / 8,
            "reuse gave {reused} against {every} for a slot per tensor"
        );
    }

    #[test]
    fn a_parameter_slot_is_never_handed_to_anything_else() {
        // The hazard that motivates `pinned`. A parameter is uploaded once and
        // read on every run, so recycling its storage after its last read
        // inside one run is correct for that run and returns garbage on the
        // next — the worst shape a bug can take, because the first result
        // looks right.
        let mut g = Graph::new("weighted");
        let f = DType::F32;
        let shape = Shape::new(&[64, 64], f);
        let x = g.input("x", shape.clone());
        let w = g.param("w", shape.clone());
        // The parameter is read once, near the start, then never again.
        let mut cur = g.add_node(Op::MatMul, vec![x, w], shape.clone());
        for _ in 0..24 {
            cur = g.add_node(Op::Activation(Activation::Relu), vec![cur], shape.clone());
        }
        g.set_outputs(vec![cur]);

        let plan = plan_f32_uniform(&g, 16);
        let p = &plan.assignments[&w];
        for node in g.nodes() {
            if node.id == w {
                continue;
            }
            let s = &plan.assignments[&node.id];
            assert!(
                !(s.offset < p.offset + p.size && p.offset < s.offset + s.size),
                "{:?} was given storage inside the parameter's slot",
                node.id
            );
        }
    }

    #[test]
    fn a_parameter_never_lands_inside_a_recycled_slot() {
        // The bug this cost a bisection to find. Pinning kept parameters off
        // the free list but did not stop them TAKING a recycled slot — and a
        // recycled slot is one an earlier node overwrites on every run. A
        // LeakyReLU's slope parameter landed inside the convolution's output,
        // so the slope was silently replaced by activation data: positives came
        // out exactly right and negatives were scaled by garbage.
        let mut g = Graph::new("late_param");
        let f = DType::F32;
        let big = Shape::new(&[128, 128], f);
        let x = g.input("x", big.clone());
        let w = g.param("w", big.clone());
        let mut cur = g.add_node(Op::MatMul, vec![x, w], big.clone());
        // Something large dies here, freeing a slot...
        for _ in 0..6 {
            cur = g.add_node(Op::Activation(Activation::Relu), vec![cur], big.clone());
        }
        // ...and only now is this tiny parameter introduced.
        let slope = g.param("slope", Shape::new(&[1], f));
        let out = g.add_node(Op::Binary(rlx_ir::op::BinaryOp::Mul), vec![cur, slope], big);
        g.set_outputs(vec![out]);

        let plan = plan_f32_uniform(&g, 16);
        let sp = &plan.assignments[&slope];
        for node in g.nodes() {
            if node.id == slope || matches!(node.op, Op::Param { .. } | Op::Input { .. }) {
                continue;
            }
            let s = &plan.assignments[&node.id];
            assert!(
                !(s.offset < sp.offset + sp.size && sp.offset < s.offset + s.size),
                "{:?} writes over the slope parameter at {}+{}",
                node.id,
                sp.offset,
                sp.size
            );
        }
    }

    #[test]
    fn an_output_slot_is_not_recycled_after_it_is_written() {
        // Reusing a dead intermediate's storage *for* the output is fine and
        // expected. What must not happen is the reverse: the output's storage
        // being handed to something else, since the host reads it after the
        // run has finished.
        let g = chain(20, 1024);
        let plan = plan_f32_uniform(&g, 16);
        let out = g.outputs[0];
        let o = &plan.assignments[&out];
        let written_at = plan.schedule.iter().position(|id| *id == out).unwrap();
        for (step, id) in plan.schedule.iter().enumerate() {
            if step <= written_at || *id == out {
                continue;
            }
            let s = &plan.assignments[id];
            assert!(
                !(s.offset < o.offset + o.size && o.offset < s.offset + s.size),
                "{id:?}, written after the output, overlaps it"
            );
        }
    }

    #[test]
    fn the_escape_hatch_restores_a_slot_per_tensor() {
        let g = chain(16, 4096);
        let with = plan_f32_uniform_with(&g, 16, true).arena_size;
        let without = plan_f32_uniform_with(&g, 16, false).arena_size;
        assert!(
            without > with,
            "no-reuse {without} should exceed reuse {with}"
        );
    }

    #[test]
    fn every_scheduled_node_still_gets_a_slot() {
        // Aliases resolve after their root is placed; a missed one would panic
        // later in `Arena::offset`, far from the cause.
        let mut g = Graph::new("views");
        let shape = Shape::new(&[32, 32], DType::F32);
        let x = g.input("x", shape.clone());
        let a = g.add_node(Op::Activation(Activation::Relu), vec![x], shape.clone());
        let r = g.add_node(
            Op::Reshape {
                new_shape: vec![1024],
            },
            vec![a],
            Shape::new(&[1024], DType::F32),
        );
        let r2 = g.add_node(
            Op::Reshape {
                new_shape: vec![32, 32],
            },
            vec![r],
            shape.clone(),
        );
        let out = g.add_node(Op::Activation(Activation::Relu), vec![r2], shape);
        g.set_outputs(vec![out]);
        let plan = plan_f32_uniform(&g, 16);
        for id in &plan.schedule {
            assert!(plan.assignments.contains_key(id), "{id:?} has no slot");
        }
        // A reshape chain must land on the storage it aliases, not a copy.
        assert_eq!(plan.assignments[&r].offset, plan.assignments[&a].offset);
        assert_eq!(plan.assignments[&r2].offset, plan.assignments[&a].offset);
    }
}
