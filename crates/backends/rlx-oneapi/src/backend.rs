// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `OneApiExecutable` — compile an IR graph for the Intel oneAPI Level Zero
//! backend and execute it.
//!
//! Two execution paths share one legalized graph (the rlx-vulkan primitive set,
//! so the same rewrite/legalize decompositions apply):
//!
//! - [`run_host`](OneApiExecutable::run_host) — a value-map interpreter that
//!   evaluates every node through the `rlx-cpu` reference. This is the path the
//!   macOS dev box / CI take (no Level Zero device), and it makes the backend
//!   fully correct without Intel hardware.
//! - [`run_l0`](OneApiExecutable::run_l0) — the native path: a USM-shared f32
//!   arena + per-op SPIR-V kernel dispatch (with a CPU host-fallback, against
//!   the same arena, for ops with no native kernel yet). Selected only when a
//!   live device *and* embedded kernels are both present — neither is true off
//!   an Intel build host, so it is compiled-but-dormant here, pending hardware
//!   validation on Arc / Data Center Max.

use crate::device::oneapi_device;
use crate::host::{self, HostBuf};
use crate::kernels::kernels;
use rlx_compile::memory::{BufferSlot, MemoryPlan};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, RngOptions, Shape};
use std::collections::HashMap;
use std::ffi::c_void;

/// OpKinds this backend lowers (claim set). Identical to rlx-vulkan's: the
/// rewrite pass decomposes everything else into this primitive set, and the
/// CPU reference covers every entry, so a legalized graph always executes.
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        Cast,
        StopGradient,
        Reshape, // structural / alias
        Binary,
        Compare,
        Where,
        Activation, // elementwise
        MatMul,
        Reduce,
        Softmax, // contraction / reduction
        LayerNorm,
        RmsNorm,
        LayerNorm2d, // normalization
        Rope,
        Attention, // transformer
        // Claimed first-class; `compile_rng` runs `unfuse_attention_block`
        // to lower it to the primitive chain above before legalization.
        FusedAttentionBlock,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Cumsum,
        Reverse, // shape / indexing
        ArgMax,
        ArgMin,
        Pool,
        ResizeNearest2x,
        Conv,          // reductions / vision
        GroupedMatMul, // MoE
        SelectiveScan, // SSM / Mamba
        Im2Col,
        ScatterAdd,
        TopK, // vision / indexing / generation
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        GatedDeltaNet,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad):
        // no native kernel → routed to the rlx-cpu host fallback (USM-shared arena).
        Scan,
        ConvTranspose2d,
        Fft,
        DequantMatMul,
        DequantGroupedMatMul,
        DequantMoEWeights, // GGUF quant
        RngNormal,
        RngUniform,
        Sample, // RNG / generation
    ]
};

/// Ops with a native OpenCL-C SPIR-V kernel under `kernels/`. Everything else
/// routes to the CPU host-fallback on the native path. The set grows as kernels
/// land (next: layernorm, rope, gather, reduce, attention, then oneMKL gemm).
fn native_kernel(op: &Op) -> Option<&'static str> {
    match op {
        Op::Binary(_) => Some("binary"),
        Op::Activation(_) => Some("unary"),
        Op::MatMul => Some("matmul"),
        Op::Softmax { .. } => Some("softmax"),
        Op::RmsNorm { .. } => Some("rmsnorm"),
        _ => None,
    }
}

#[derive(Clone)]
enum ParamVal {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

pub struct OneApiExecutable {
    /// Post-legalize, f32-uniform graph.
    graph: Graph,
    params: HashMap<String, ParamVal>,
    output_ids: Vec<NodeId>,
    output_dtypes: Vec<DType>,
    rng: RngOptions,
    active_extent: Option<(usize, usize)>,
}

unsafe impl Send for OneApiExecutable {}

impl OneApiExecutable {
    pub fn compile(graph: Graph) -> Self {
        Self::compile_rng(graph, RngOptions::default())
    }

    /// Legalize the graph to the native primitive set, then capture I/O maps.
    pub fn compile_rng(graph: Graph, rng: RngOptions) -> Self {
        use rlx_opt::pass::Pass as _;

        let graph = rlx_opt::LowerControlFlow.run(graph);
        // Decompose `FusedAttentionBlock` (claimed, but no monolithic
        // kernel) to primitives before legalization. FAB-only; no-op when
        // absent.
        let graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, SUPPORTED_OPS)
            .unwrap_or_else(|errs| panic!("{}", rlx_opt::format_legalize_error("oneapi", &errs)));
        let graph = rlx_opt::LegalizeBroadcast.run(graph);

        let output_ids = graph.outputs.clone();
        let output_dtypes = output_ids
            .iter()
            .map(|&id| graph.node(id).shape.dtype())
            .collect();

        Self {
            graph,
            params: HashMap::new(),
            output_ids,
            output_dtypes,
            rng,
            active_extent: None,
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params
            .insert(name.to_string(), ParamVal::F32(data.to_vec()));
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        self.params
            .insert(name.to_string(), ParamVal::Bytes(data.to_vec()));
    }

    pub fn output_dtypes(&self) -> Vec<DType> {
        self.output_dtypes.clone()
    }

    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    pub fn set_rng(&mut self, rng: RngOptions) {
        self.rng = rng;
    }

    pub fn rng(&self) -> RngOptions {
        self.rng
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        // Native dispatch only when a live device AND embedded kernels exist;
        // otherwise the CPU-reference interpreter (the dev-box / CI path).
        if oneapi_device().is_some() && kernels().is_some() {
            self.run_l0(inputs, read_indices)
        } else {
            self.run_host(inputs, read_indices)
        }
    }

    // ── dev-box path: whole-graph CPU reference interpreter ────────────────

    fn run_host(&self, inputs: &[(&str, &[f32])], read_indices: Option<&[usize]>) -> Vec<Vec<f32>> {
        let in_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        let mut f32v: HashMap<NodeId, Vec<f32>> = HashMap::new();
        let mut bytev: HashMap<NodeId, Vec<u8>> = HashMap::new();

        for node in self.graph.nodes() {
            let numel = node.shape.num_elements().unwrap_or(0);
            match &node.op {
                Op::Input { name } => {
                    let v = in_map
                        .get(name.as_str())
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| vec![0.0; numel]);
                    f32v.insert(node.id, v);
                }
                Op::Param { name } => match self.params.get(name) {
                    Some(ParamVal::F32(v)) => {
                        f32v.insert(node.id, v.clone());
                    }
                    Some(ParamVal::Bytes(b)) => {
                        bytev.insert(node.id, b.clone());
                    }
                    None => {
                        f32v.insert(node.id, vec![0.0; numel]);
                    }
                },
                Op::Constant { data } => {
                    if matches!(node.shape.dtype(), DType::U8 | DType::I8) {
                        bytev.insert(node.id, data.clone());
                    } else {
                        f32v.insert(node.id, widen_const_to_f32(data, node.shape.dtype()));
                    }
                }
                _ => {
                    let in_specs: Vec<(Shape, HostBuf)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let buf = if let Some(b) = bytev.get(&id) {
                                HostBuf::Bytes(b.clone())
                            } else {
                                HostBuf::F32(f32v.get(&id).cloned().unwrap_or_default())
                            };
                            (sh, buf)
                        })
                        .collect();
                    let out = host::eval(&node.op, &node.shape, &in_specs);
                    f32v.insert(node.id, out);
                }
            }
        }

        self.read_outputs(read_indices, |id, n| {
            f32v.get(&id)
                .map(|v| v[..n.min(v.len())].to_vec())
                .unwrap_or_else(|| vec![0.0; n])
        })
    }

    // ── native path: USM arena + per-op SPIR-V dispatch (HW-pending) ───────

    fn run_l0(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        let dev = oneapi_device().expect("rlx-oneapi: no device");
        let kerns = kernels().expect("rlx-oneapi: no kernels");

        let plan = plan_f32_uniform(&self.graph, 64);
        let arena = match crate::arena::Arena::from_plan(&plan) {
            Ok(a) => a,
            // Allocation failed on the device — fall back to the CPU path so we
            // still return correct results rather than panic.
            Err(_) => return self.run_host(inputs, read_indices),
        };

        // Upload constants, params, inputs into the USM arena.
        for node in self.graph.nodes() {
            match &node.op {
                Op::Constant { data } if arena.has(node.id) && !data.is_empty() => {
                    if matches!(node.shape.dtype(), DType::U8 | DType::I8) {
                        arena.write_bytes(node.id, data);
                    } else {
                        arena.write_f32(node.id, &widen_const_to_f32(data, node.shape.dtype()));
                    }
                }
                Op::Param { name } => match self.params.get(name) {
                    Some(ParamVal::F32(v)) => arena.write_f32(node.id, v),
                    Some(ParamVal::Bytes(b)) => arena.write_bytes(node.id, b),
                    None => {}
                },
                _ => {}
            }
        }
        let in_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        for node in self.graph.nodes() {
            if let Op::Input { name } = &node.op {
                if let Some(data) = in_map.get(name.as_str()) {
                    arena.write_f32(node.id, data);
                }
            }
        }

        // Execute node-by-node: native kernel where available, else CPU
        // host-fallback against the (host-coherent) USM arena.
        let list = dev.create_command_list().expect("rlx-oneapi: command list");
        for node in self.graph.nodes() {
            if matches!(
                node.op,
                Op::Input { .. }
                    | Op::Param { .. }
                    | Op::Constant { .. }
                    | Op::Reshape { .. }
                    | Op::Cast { .. }
                    | Op::StopGradient
            ) {
                continue;
            }
            match native_kernel(&node.op) {
                Some(name) => self.dispatch(dev, kerns, list, name, node, &arena),
                None => {
                    // Read inputs out of the arena, eval on CPU, write back.
                    let in_specs: Vec<(Shape, HostBuf)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let nn = sh.num_elements().unwrap_or(0);
                            let buf = if matches!(sh.dtype(), DType::U8 | DType::I8) {
                                HostBuf::Bytes(arena.read_bytes(id, nn))
                            } else {
                                HostBuf::F32(arena.read_f32(id, nn))
                            };
                            (sh, buf)
                        })
                        .collect();
                    let out = host::eval(&node.op, &node.shape, &in_specs);
                    arena.write_f32(node.id, &out);
                }
            }
        }
        dev.execute_sync(list).expect("rlx-oneapi: execute");
        unsafe {
            let _ = (dev.lib.command_list_destroy)(list);
        }

        self.read_outputs(read_indices, |id, n| arena.read_f32(id, n))
    }

    /// Set kernel arguments (arg 0 = arena base pointer, then scalars) and
    /// append a launch onto `list`. Arg layouts match `kernels/<name>.cl`.
    fn dispatch(
        &self,
        dev: &crate::device::OneApiDevice,
        kerns: &crate::kernels::Kernels,
        list: crate::level_zero::CommandListHandle,
        name: &str,
        node: &rlx_ir::Node,
        arena: &crate::arena::Arena,
    ) {
        let Some(kernel) = kerns.get(name) else {
            return;
        };
        let off = |id: NodeId| arena.elem_offset(id);
        let out = node.id;
        let mut args: Vec<KArg> = vec![KArg::Ptr(arena.base_ptr())];
        let (global, local): (usize, u32) = match &node.op {
            Op::Binary(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                let an = numel(&dims(&self.graph, a));
                let bn = numel(&dims(&self.graph, b));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(if an == n { 0 } else { an as u32 }),
                    KArg::U32(if bn == n { 0 } else { bn as u32 }),
                    KArg::U32(binop_id(*op)),
                ]);
                (n, 256)
            }
            Op::Activation(act) => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                    KArg::U32(act_id(*act)),
                ]);
                (n, 256)
            }
            Op::MatMul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let ad = dims(&self.graph, a);
                let bd = dims(&self.graph, b);
                let od = dims(&self.graph, out);
                let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
                let n = bd[bd.len() - 1];
                let batch = if od.len() > 2 {
                    numel(&od[..od.len() - 2])
                } else {
                    1
                };
                let a_batch = if ad.len() > 2 {
                    numel(&ad[..ad.len() - 2])
                } else {
                    1
                };
                let b_batch = if bd.len() > 2 {
                    numel(&bd[..bd.len() - 2])
                } else {
                    1
                };
                let a_bs = if a_batch <= 1 { 0 } else { m * k };
                let b_bs = if b_batch <= 1 { 0 } else { k * n };
                args.extend([
                    KArg::U32(m as u32),
                    KArg::U32(k as u32),
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(batch as u32),
                    KArg::U32(a_bs as u32),
                    KArg::U32(b_bs as u32),
                    KArg::U32((m * n) as u32),
                ]);
                (batch.max(1) * m * n, 64)
            }
            Op::Softmax { axis } => {
                let x = node.inputs[0];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let axis_len = xd[ax];
                let outer = numel(&xd[..ax]);
                let inner = numel(&xd[ax + 1..]);
                args.extend([
                    KArg::U32(outer as u32),
                    KArg::U32(axis_len as u32),
                    KArg::U32(inner as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                ]);
                (outer * inner, 256)
            }
            Op::RmsNorm { axis, eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let beta = node.inputs[2];
                let xd = dims(&self.graph, x);
                let ax = norm_axis(*axis, xd.len());
                let n = xd[ax];
                let rows = numel(&xd) / n.max(1);
                args.extend([
                    KArg::U32(rows as u32),
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(gamma)),
                    KArg::U32(off(beta)),
                    KArg::U32(off(out)),
                    KArg::F32(*eps),
                ]);
                (rows, 64)
            }
            _ => return,
        };

        unsafe {
            let _ = (dev.lib.kernel_set_group_size)(kernel, local, 1, 1);
            for (i, a) in args.iter().enumerate() {
                let (size, ptr) = a.as_arg();
                let _ = (dev.lib.kernel_set_argument_value)(kernel, i as u32, size, ptr);
            }
            let groups = crate::level_zero::GroupCount {
                group_count_x: ceil_div(global, local).max(1),
                group_count_y: 1,
                group_count_z: 1,
            };
            let _ = (dev.lib.command_list_append_launch_kernel)(
                list,
                kernel,
                &groups,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            // Each kernel reads/writes the shared arena; barrier between launches.
            let _ = (dev.lib.command_list_append_barrier)(
                list,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
        }
    }

    fn read_outputs(
        &self,
        read_indices: Option<&[usize]>,
        mut read: impl FnMut(NodeId, usize) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let want: Vec<usize> = match read_indices {
            Some(ix) => ix.to_vec(),
            None => (0..self.output_ids.len()).collect(),
        };
        want.into_iter()
            .filter_map(|i| {
                let id = *self.output_ids.get(i)?;
                let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                Some(read(id, n))
            })
            .collect()
    }

    /// Deep copy for the runtime's executable cache: fresh state with the same
    /// legalized graph + uploaded params.
    pub fn clone_for_cache(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            params: self.params.clone(),
            output_ids: self.output_ids.clone(),
            output_dtypes: self.output_dtypes.clone(),
            rng: self.rng,
            active_extent: self.active_extent,
        }
    }
}

// ── kernel-argument helper ─────────────────────────────────────────────────

enum KArg {
    Ptr(*mut c_void),
    U32(u32),
    F32(f32),
}

impl KArg {
    /// `(argSize, pArgValue)` for `zeKernelSetArgumentValue`. The returned
    /// pointer borrows `self`, so it must be consumed before `self` drops —
    /// callers use it immediately inside the set-arg loop.
    fn as_arg(&self) -> (usize, *const c_void) {
        match self {
            KArg::Ptr(p) => (
                std::mem::size_of::<*mut c_void>(),
                p as *const *mut c_void as *const c_void,
            ),
            KArg::U32(v) => (4, v as *const u32 as *const c_void),
            KArg::F32(v) => (4, v as *const f32 as *const c_void),
        }
    }
}

// ── memory plan (f32-uniform bump allocator; same as rlx-vulkan) ───────────

fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        if matches!(
            node.op,
            Op::Reshape { .. } | Op::Cast { .. } | Op::StopGradient
        ) {
            if let Some(in_id) = node.inputs.first() {
                if let Some(slot) = assignments.get(in_id) {
                    let aliased = slot.clone();
                    assignments.insert(node.id, aliased);
                    schedule.push(node.id);
                    continue;
                }
            }
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        let bytes = (elems * 4).max(4);
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
        arena_size: cursor.max(align),
        assignments,
        schedule,
    }
}

// ── small shape helpers (shared with the dispatch builder) ─────────────────

fn dims(graph: &Graph, id: NodeId) -> Vec<usize> {
    graph
        .node(id)
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(s) => *s,
            _ => 0,
        })
        .collect()
}

fn numel(d: &[usize]) -> usize {
    d.iter()
        .product::<usize>()
        .max(if d.is_empty() { 1 } else { 0 })
}

fn norm_axis(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis).max(0) as usize
    } else {
        (axis as usize).min(rank.saturating_sub(1))
    }
}

fn ceil_div(n: usize, d: u32) -> u32 {
    (n as u64).div_ceil(d as u64) as u32
}

fn act_id(a: Activation) -> u32 {
    match a {
        Activation::Gelu => 0,
        Activation::GeluApprox => 1,
        Activation::Silu => 2,
        Activation::Relu => 3,
        Activation::Sigmoid => 4,
        Activation::Tanh => 5,
        Activation::Exp => 6,
        Activation::Log => 7,
        Activation::Sqrt => 8,
        Activation::Rsqrt => 9,
        Activation::Neg => 10,
        Activation::Abs => 11,
        Activation::Sin => 12,
        Activation::Cos => 13,
        Activation::Tan => 14,
        Activation::Atan => 15,
        Activation::Round => 16,
    }
}

fn binop_id(op: rlx_ir::op::BinaryOp) -> u32 {
    use rlx_ir::op::BinaryOp::*;
    match op {
        Add => 0,
        Sub => 1,
        Mul => 2,
        Div => 3,
        Max => 4,
        Min => 5,
        Pow => 6,
    }
}

/// Widen a constant byte blob (any IR dtype) to f32 for the f32-uniform arena.
fn widen_const_to_f32(data: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I32 | DType::U32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::I16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        DType::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => data.iter().map(|&b| b as f32).collect(),
        DType::C64 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}
