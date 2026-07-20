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
use crate::host::{self, HostBuf, HostOut};
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
        // DiT modulation — claimed for fusion; `unfuse_dit_modulation`
        // expands forward Ada/Gated before host / SPIR-V lowering.
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native OpenCL-C SPIR-V when kernels are
        // embedded (`RLX_ONEAPI_BUILD_KERNELS=1`); else CPU host-fallback.
        AdaLayerNormBackward,
        GatedResidualBackward,
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
        ScatterNd,
        ScatterElements,
        GatherNd,
        GatherElements,
        TopK, // vision / indexing / generation
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        GatedDeltaNet,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad):
        // no native kernel → routed to the rlx-cpu host fallback (USM-shared arena).
        Scan,
        ScanBackward,
        ScanBackwardXs,
        ConvTranspose2d,
        Fft,
        DequantMatMul,
        DequantGroupedMatMul,
        DequantMoEWeights, // GGUF quant
        RngNormal,
        RngUniform,
        Sample, // RNG / generation
        // Core Riemannian / SPD-manifold ops (F64): no native kernel → routed
        // to the F64-aware CPU host fallback (`crate::spd`), on both the
        // value-map (`run_host`) and USM-arena (`run_l0`) paths.
        BiMap,
        ReEig,
        LogEig,
        SpdBatchNorm,
        SpdKarcherMean,
        SpdKarcherMeanWeighted,
        SpdLogMap,
        SpdExpMap,
        SpdParallelTransport,
        SpdMatrixFnBatch,
        ReEigBackward,
        LogEigBackward,
        SpdBatchNormBackwardX,
        SpdBatchNormBackwardG,
        SpdLogMapBackward,
        SpdExpMapBackward,
        SpdParallelTransportBackward,
        SpdMatrixFnBatchBackward,
        Eigh,
        EighBackward,
        EighBatch,
        EighBatchBackward,
        // In-graph collectives (`collective.*`) — claimed so the Session/stages
        // legalize pass lets them through; `run_host` / L0 host-fallback eval
        // via rlx-cpu (same as rlx-vulkan).
        Custom,
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
        Op::AdaLayerNormBackward { .. } => Some("ada_layer_norm_backward"),
        Op::GatedResidualBackward => Some("gated_residual_backward"),
        _ => None,
    }
}

/// Cast op ids for the `unary` OpenCL kernel (`unary.cl` cases 100–106). Kept
/// in sync with rlx-cuda / rlx-rocm (unary.cu) and rlx-vulkan (unary.comp).
const CAST_F32_TO_I8: u32 = 100;
const CAST_F32_TO_I16: u32 = 101;
const CAST_F32_TO_I32: u32 = 102;
const CAST_F32_TO_I64: u32 = 103;
const CAST_F32_TO_U8: u32 = 104;
const CAST_F32_TO_U32: u32 = 105;
const CAST_TO_BOOL: u32 = 106;

/// How an `Op::Cast` lowers on the f32-uniform arena.
enum CastLower {
    /// Value-preserving relabel — alias the input slot (no dispatch). Covers
    /// same-dtype, int→float, float→float (F16/BF16/F64 are all f32-stored
    /// here), int→int, and bool→int/float.
    Identity,
    /// A real elementwise conversion via the `unary` kernel with this op id
    /// (float→int trunc-saturate, or →Bool `x != 0`).
    Kernel(u32),
    /// A complex cast (real↔C64, real↔C128, C64↔C128) — pure f32-lane moves via
    /// the standalone `complex_cast` kernel. Carries the mode (0..5, see
    /// `kernels/complex_cast.cl`). Needs its own (complex-sized) slot, not an
    /// alias — the lane width changes even though the element count does not.
    Complex(u32),
    /// Not representable in an f32 arena (an F64 real component has no lane
    /// storage here) — reject at lowering.
    Reject,
}

/// Classify a `Cast(src → dst)` on the f32-uniform arena. float→int truncates
/// toward zero + saturates (Rust `as` / rlx-cpu); →Bool is `x != 0`. F16/BF16/
/// F64 are demoted to f32 storage so real casts to/from them are identity
/// relabels. Complex casts (real↔C64, real↔C128, C64↔C128) are pure f32-lane
/// moves on the simulated-complex arena (C64 = 2 lanes/elem, C128 = 4 lanes
/// df64); only a complex cast touching the one non-lane-storable real component
/// (F64, demoted to a single lossy lane) rejects. Mirrors rlx-vulkan / rlx-cuda
/// / rlx-wgpu.
fn classify_cast(src: DType, dst: DType) -> CastLower {
    if src == dst {
        return CastLower::Identity; // pure relabel (also covers C64→C64 / C128→C128)
    }
    if src.is_complex() || dst.is_complex() {
        // F64 is the one component type with no faithful f32-lane storage here
        // (it is demoted to a single lossy lane elsewhere), so a complex cast
        // touching it on the real side is still rejected.
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
/// →Bool / complex lane-move) or must be rejected — i.e. not an identity
/// relabel. Complex casts change the lane width (real 1 → C64 2 → C128 4), so
/// they cannot alias the input slot.
fn cast_is_kernel(graph: &Graph, node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::Cast { to } => !matches!(
            classify_cast(graph.node(node.inputs[0]).shape.dtype(), *to),
            CastLower::Identity
        ),
        _ => false,
    }
}

/// Number of f32 lanes a node occupies in the f32-uniform arena. Complex is
/// simulated on f32 lanes — C64 = 2 lanes/elem, C128 = 4 lanes df64; every
/// OTHER dtype is exactly ONE f32 lane per element (I64/Bool/… are widened to a
/// single lane, so `size_bytes()/4` must NOT be blanket-applied — that would
/// make I64 two lanes and Bool zero). Drives slot sizing and lane-aware
/// readback (reading a complex output by element count would truncate it to the
/// real parts).
fn arena_lane_count(shape: &Shape) -> usize {
    let elems = shape.num_elements().unwrap_or(0);
    match shape.dtype() {
        DType::C64 => elems * 2,
        DType::C128 => elems * 4,
        _ => elems,
    }
}

/// Complex `Op::Cast` on f32 lanes (host-path mirror of `kernels/complex_cast.cl`,
/// same six lane-move modes). `n` is the complex-element count (cast-invariant).
/// Used by `run_host` so the CPU-reference path keeps the same df64 lane
/// convention as the on-device kernels (rather than routing through rlx-cpu's
/// native-f64 C128 storage, which is a different byte layout).
fn complex_cast_host(input: &[f32], n: usize, mode: u32) -> Vec<f32> {
    let ld = |j: usize| input.get(j).copied().unwrap_or(0.0);
    let out_lanes = match mode {
        0 | 5 => 2 * n, // → C64
        1 | 3 => n,     // → real
        2 | 4 => 4 * n, // → C128
        _ => n,
    };
    let mut out = vec![0.0f32; out_lanes];
    for k in 0..n {
        match mode {
            0 => out[2 * k] = ld(k),                          // real → C64 (im=0)
            1 => out[k] = ld(2 * k),                          // C64 → real
            2 => out[4 * k] = ld(k),                          // real → C128 (rest 0)
            3 => out[k] = ld(4 * k),                          // C128 → real
            4 => {
                out[4 * k] = ld(2 * k); // C64 → C128
                out[4 * k + 2] = ld(2 * k + 1);
            }
            5 => {
                out[2 * k] = ld(4 * k); // C128 → C64
                out[2 * k + 1] = ld(4 * k + 2);
            }
            _ => {}
        }
    }
    out
}

/// Element-wise C64 binary op on f32 lanes (host-path mirror of
/// `kernels/binary_c64.cl`). `op` 0=add/1=sub/2=mul/3=div; `n` is the output
/// complex-element count; `n_a`/`n_b` are operand complex-element counts for
/// modulo broadcast. Formulas match rlx-cpu `exec_binary_full_c64`.
fn binary_c64_host(a: &[f32], b: &[f32], n: usize, n_a: usize, n_b: usize, op: u32) -> Vec<f32> {
    let na = n_a.max(1);
    let nb = n_b.max(1);
    let la = |j: usize| a.get(j).copied().unwrap_or(0.0);
    let lb = |j: usize| b.get(j).copied().unwrap_or(0.0);
    let mut out = vec![0.0f32; 2 * n];
    for k in 0..n {
        let ka = k % na;
        let kb = k % nb;
        let (ar, ai) = (la(2 * ka), la(2 * ka + 1));
        let (br, bi) = (lb(2 * kb), lb(2 * kb + 1));
        let (cr, ci) = match op {
            0 => (ar + br, ai + bi),
            1 => (ar - br, ai - bi),
            2 => (ar * br - ai * bi, ar * bi + ai * br),
            3 => {
                let d = br * br + bi * bi;
                ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
            }
            _ => (0.0, 0.0),
        };
        out[2 * k] = cr;
        out[2 * k + 1] = ci;
    }
    out
}

/// Reject a complex `Op::Binary` that has no simulated path — C128 arithmetic
/// (rlx-cpu has none either) and C64 max/min/pow (undefined for complex).
/// Returns the C64 op code (0=add/1=sub/2=mul/3=div) when supported; panics
/// otherwise, matching the CPU reference's rejection (never a silently-wrong
/// result). Shared by `run_host` and the L0 dispatch builder.
fn c64_binary_opcode(dtype: DType, op: rlx_ir::op::BinaryOp) -> u32 {
    if dtype == DType::C128 {
        panic!(
            "rlx-oneapi: Binary on C128: complex-f64 arithmetic is unsupported \
             (rlx-cpu has none either) — only C64 add/sub/mul/div are wired"
        );
    }
    let code = binop_id(op);
    if code > 3 {
        panic!(
            "rlx-oneapi: C64 Binary: {op:?} is undefined for complex (only \
             Add/Sub/Mul/Div); matches rlx-cpu rejection"
        );
    }
    code
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
        Self::compile_rng_with_options(graph, rng, 64)
    }

    pub fn compile_rng_with_options(
        graph: Graph,
        rng: RngOptions,
        scan_unroll_max_length: u32,
    ) -> Self {
        use rlx_opt::pass::Pass as _;

        let graph = rlx_opt::LowerControlFlow.run(graph);
        // Decompose `FusedAttentionBlock` (claimed, but no monolithic
        // kernel) to primitives before legalization. FAB-only; no-op when
        // absent.
        let graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        let graph = rlx_opt::unfuse::unfuse_dit_modulation(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, SUPPORTED_OPS)
            .unwrap_or_else(|errs| panic!("{}", rlx_opt::format_legalize_error("oneapi", &errs)));
        let graph = rlx_cpu::rlx_maybe_unroll_scans!(graph, scan_unroll_max_length);
        let graph = rlx_opt::maybe_unroll_scans_budget(graph, 4096);
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
                // Core Riemannian / SPD-manifold ops (F64) go through the
                // F64-aware `spd::eval` (widens f32→f64, runs the CPU thunk,
                // narrows back), not the f32-only `host::eval` — same split as
                // rlx-vulkan. Delegating with each node's REAL declared
                // dtype/shape handles the packed `[2n²+n]` ReEig/LogEig forward
                // output + precomputed backward layout for free.
                op if crate::spd::is_spd_host(op) => {
                    let in_specs: Vec<(Shape, Vec<f32>)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            (sh, f32v.get(&id).cloned().unwrap_or_default())
                        })
                        .collect();
                    let out = crate::spd::eval(&node.op, &node.shape, &in_specs);
                    f32v.insert(node.id, out);
                }
                Op::Scan { .. } => {
                    let out = rlx_cpu::thunk::run_scan_node_f32(node, |id| {
                        f32v.get(&id).cloned().unwrap_or_default()
                    });
                    f32v.insert(node.id, out);
                }
                // Complex Cast (real↔C64, real↔C128, C64↔C128): pure f32-lane
                // moves in the df64 convention — handled directly rather than
                // via rlx-cpu (whose native-f64 C128 storage is a different byte
                // layout), so `run_host` keeps the SAME lane convention as the
                // on-device `complex_cast` kernel + the shared widen/narrow
                // boundary. `numel` is the cast-invariant complex-element count.
                Op::Cast { to }
                    if !matches!(
                        classify_cast(self.graph.node(node.inputs[0]).shape.dtype(), *to),
                        CastLower::Identity | CastLower::Kernel(_)
                    ) =>
                {
                    let src = self.graph.node(node.inputs[0]).shape.dtype();
                    match classify_cast(src, *to) {
                        CastLower::Complex(mode) => {
                            let input = f32v.get(&node.inputs[0]).cloned().unwrap_or_default();
                            f32v.insert(node.id, complex_cast_host(&input, numel, mode));
                        }
                        CastLower::Reject => panic!(
                            "rlx-oneapi: Cast {src:?} → {to:?} touches an F64 real \
                             component with no faithful f32-lane storage — run on CPU"
                        ),
                        _ => unreachable!("guard excludes Identity / Kernel"),
                    }
                }
                // Complex Binary (C64 add/sub/mul/div): reads both [re, im]
                // lanes per element, evaluated directly to match the on-device
                // `binary_c64` kernel. C128 arithmetic + C64 max/min/pow reject
                // (matches rlx-cpu). `numel` is the output complex-element count.
                Op::Binary(op) if node.shape.dtype().is_complex() => {
                    let code = c64_binary_opcode(node.shape.dtype(), *op);
                    let a = f32v.get(&node.inputs[0]).cloned().unwrap_or_default();
                    let b = f32v.get(&node.inputs[1]).cloned().unwrap_or_default();
                    let na = self.graph.node(node.inputs[0]).shape.num_elements().unwrap_or(0);
                    let nb = self.graph.node(node.inputs[1]).shape.num_elements().unwrap_or(0);
                    f32v.insert(node.id, binary_c64_host(&a, &b, numel, na, nb, code));
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
                    match host::eval(&node.op, &node.shape, &in_specs) {
                        HostOut::F32(out) => {
                            f32v.insert(node.id, out);
                        }
                        HostOut::Bytes(b) => {
                            bytev.insert(node.id, b);
                        }
                    }
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
                    | Op::StopGradient
            ) {
                continue;
            }
            // Cast: identity relabels are arena-aliased (skip); float→int /
            // →Bool casts got their own f32-sized slot and dispatch the `unary`
            // kernel (value stored as an f32 lane). Complex is rejected.
            if let Op::Cast { to } = &node.op {
                let src = self.graph.node(node.inputs[0]).shape.dtype();
                match classify_cast(src, *to) {
                    CastLower::Identity => continue,
                    CastLower::Kernel(_) => {
                        self.dispatch(dev, kerns, list, "unary", node, &arena);
                        continue;
                    }
                    // real↔C64, real↔C128, C64↔C128 — pure f32-lane moves.
                    CastLower::Complex(_) => {
                        self.dispatch(dev, kerns, list, "complex_cast", node, &arena);
                        continue;
                    }
                    CastLower::Reject => panic!(
                        "rlx-oneapi: Cast {src:?} → {to:?} touches an F64 real component \
                         with no faithful f32-lane storage in the uniform arena — run on CPU"
                    ),
                }
            }
            // Complex Binary (C64 add/sub/mul/div) reads BOTH [re, im] lanes per
            // element, so it lowers to the standalone `binary_c64` kernel (not
            // the scalar `binary`). C128 arithmetic + C64 max/min/pow are
            // rejected inside the dispatch arg builder (matches rlx-cpu).
            if let Op::Binary(_) = &node.op {
                if node.shape.dtype().is_complex() {
                    self.dispatch(dev, kerns, list, "binary_c64", node, &arena);
                    continue;
                }
            }
            // SPD-manifold ops (F64, no native kernel) read the USM arena, run
            // the F64-aware `spd::eval` (widen f32→f64 → CPU thunk → narrow),
            // and write back — exactly rlx-vulkan's host-fallback split from
            // the f32-only `host::eval`.
            if crate::spd::is_spd_host(&node.op) {
                let in_specs: Vec<(Shape, Vec<f32>)> = node
                    .inputs
                    .iter()
                    .map(|&id| {
                        let sh = self.graph.node(id).shape.clone();
                        let nn = sh.num_elements().unwrap_or(0);
                        (sh, arena.read_f32(id, nn))
                    })
                    .collect();
                let out = crate::spd::eval(&node.op, &node.shape, &in_specs);
                arena.write_f32(node.id, &out);
                continue;
            }
            if matches!(node.op, Op::Scan { .. }) {
                let out = rlx_cpu::thunk::run_scan_node_f32(node, |id| {
                    let nn = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    arena.read_f32(id, nn)
                });
                arena.write_f32(node.id, &out);
                continue;
            }
            match native_kernel(&node.op).filter(|name| kerns.get(name).is_some()) {
                Some(name) => self.dispatch(dev, kerns, list, name, node, &arena),
                None => {
                    // Read inputs out of the arena, eval on CPU, write back.
                    let in_specs: Vec<(Shape, HostBuf)> = node
                        .inputs
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let nn = sh.num_elements().unwrap_or(0);
                            let buf = if matches!(sh.dtype(), DType::U8 | DType::I8 | DType::Bool) {
                                HostBuf::Bytes(arena.read_bytes(id, nn))
                            } else {
                                HostBuf::F32(arena.read_f32(id, nn))
                            };
                            (sh, buf)
                        })
                        .collect();
                    match host::eval(&node.op, &node.shape, &in_specs) {
                        HostOut::F32(out) => arena.write_f32(node.id, &out),
                        HostOut::Bytes(b) => arena.write_bytes(node.id, &b),
                    }
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
            // Standalone `binary_c64` kernel for complex output: n = output
            // complex-element count, n_a/n_b = operand complex-element counts
            // (>= 1) for modulo broadcast, offsets are f32-lane starts. Rejects
            // C128 arithmetic + C64 max/min/pow (matches rlx-cpu).
            Op::Binary(op) if node.shape.dtype().is_complex() => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(&self.graph, out));
                let na = numel(&dims(&self.graph, a));
                let nb = numel(&dims(&self.graph, b));
                let code = c64_binary_opcode(node.shape.dtype(), *op);
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(a)),
                    KArg::U32(off(b)),
                    KArg::U32(off(out)),
                    KArg::U32(code),
                    KArg::U32(na.max(1) as u32),
                    KArg::U32(nb.max(1) as u32),
                ]);
                (n, 256)
            }
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
            // float→int / →Bool cast via the `unary` kernel (op ids 100–106), or
            // a complex lane-move via the `complex_cast` kernel (mode 0..5). Both
            // share the (n, in_off, out_off, code) arg layout; the caller routes
            // to the matching kernel name. `n` is the (cast-invariant) element
            // count. The caller only routes here for `Kernel` / `Complex`.
            Op::Cast { to } => {
                let x = node.inputs[0];
                let n = numel(&dims(&self.graph, out));
                let src = self.graph.node(x).shape.dtype();
                let code = match classify_cast(src, *to) {
                    CastLower::Kernel(op) => op,      // unary conversion op id
                    CastLower::Complex(mode) => mode, // complex_cast lane-move mode
                    _ => unreachable!("dispatch(Cast) only called for kernel / complex casts"),
                };
                args.extend([
                    KArg::U32(n as u32),
                    KArg::U32(off(x)),
                    KArg::U32(off(out)),
                    KArg::U32(code),
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
            Op::AdaLayerNormBackward { norm, eps } => {
                use rlx_ir::ada_modulation_launch;
                use rlx_ir::op::AdaNormKind;
                let x = node.inputs[0];
                let scale = node.inputs[1];
                let dy = node.inputs[3];
                let x_dims = dims(&self.graph, x);
                let mod_dims = dims(&self.graph, scale);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &mod_dims);
                let layer_norm = matches!(norm, AdaNormKind::LayerNorm) as u32;
                args.extend([
                    KArg::U32(mod_rows),
                    KArg::U32(seq_per_mod),
                    KArg::U32(inner),
                    KArg::U32(off(x)),
                    KArg::U32(off(scale)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                    KArg::U32(layer_norm),
                    KArg::F32(*eps),
                ]);
                (mod_rows as usize, 64)
            }
            Op::GatedResidualBackward => {
                use rlx_ir::ada_modulation_launch;
                let y = node.inputs[1];
                let gate = node.inputs[2];
                let dy = node.inputs[3];
                let x_dims = dims(&self.graph, dy);
                let gate_dims = dims(&self.graph, gate);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &gate_dims);
                args.extend([
                    KArg::U32(mod_rows),
                    KArg::U32(seq_per_mod),
                    KArg::U32(inner),
                    KArg::U32(off(y)),
                    KArg::U32(off(gate)),
                    KArg::U32(off(dy)),
                    KArg::U32(off(out)),
                ]);
                (mod_rows as usize, 64)
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
                // Lane count, not element count: a complex output occupies 2 (C64)
                // / 4 (C128) f32 lanes per element, so reading `num_elements` would
                // truncate the readback to the real parts. One lane per element for
                // every other dtype, so this is `num_elements` there.
                let n = arena_lane_count(&self.graph.node(id).shape);
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
        // Reshape / StopGradient, and identity Casts, alias the input slot.
        // float→int / →Bool casts get their own (f32-sized) slot + a kernel.
        let is_view = match &node.op {
            Op::Reshape { .. } | Op::StopGradient => true,
            Op::Cast { .. } => !cast_is_kernel(graph, node),
            _ => false,
        };
        if is_view {
            if let Some(in_id) = node.inputs.first() {
                if let Some(slot) = assignments.get(in_id) {
                    let aliased = slot.clone();
                    assignments.insert(node.id, aliased);
                    schedule.push(node.id);
                    continue;
                }
            }
        }
        // Slot length = (#f32 lanes) × 4. Real/int/bool tensors are ONE lane per
        // element; complex is simulated on lanes (C64 = 2, C128 = 4 df64), so a
        // complex slot must reserve 2N / 4N lanes or its kernels + readback would
        // overrun / truncate.
        let lanes = arena_lane_count(&node.shape);
        let bytes = (lanes * 4).max(4);
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
        // C64 = 2 interleaved f32 lanes `[re, im]`; the host already stores it as
        // f32 pairs, so widening is a pure reinterpret (N complex → 2N lanes).
        DType::C64 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // C128 = 4 f32 lanes df64 `[re_hi, re_lo, im_hi, im_lo]`, host-stored as
        // 2×f64 (16 B/elem). This is the df64 SPLIT boundary: each f64 `v` →
        // `hi=(f32)v` + `lo=(f32)(v-(f64)hi)`, so `(f64)hi + (f64)lo` reconstructs
        // `v` to double precision. Bit-identical to the shared
        // `rlx_runtime::backend::widen_bytes_to_f32` (the CPU↔GPU boundary the
        // complex-cast kernels round-trip against).
        DType::C128 => {
            let split = |v: f64| -> [f32; 2] {
                let hi = v as f32;
                let lo = (v - hi as f64) as f32;
                [hi, lo]
            };
            let mut out = Vec::with_capacity((data.len() / 16) * 4);
            for elem in data.chunks_exact(16) {
                let re = f64::from_le_bytes(elem[0..8].try_into().unwrap());
                let im = f64::from_le_bytes(elem[8..16].try_into().unwrap());
                out.extend_from_slice(&split(re));
                out.extend_from_slice(&split(im));
            }
            out
        }
    }
}
