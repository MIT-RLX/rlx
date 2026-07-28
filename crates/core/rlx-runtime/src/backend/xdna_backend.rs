// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AMD XDNA / Ryzen AI NPU backend adapter (`Device::Xdna`) — wraps `rlx-xdna`.
//!
//! Runs a **single matmul graph** (`y = x @ W`) on the AIE-ML array via a
//! precompiled MLIR-AIE INT8 overlay, driven through `rlx_xdna::npu_gemm::NpuGemm`
//! (persistent XRT context, native — no Python). The overlay is compiled for one
//! fixed shape (`RLX_XDNA_GEMM=OM,OK,ON`); arbitrary graph matmul shapes are
//! **tiled + zero-padded** onto it (blocked GEMM), so any `M×K×N` runs on the NPU.
//!
//! i8 operands (activation `x` via `run`, weight `W` via `set_param`), i32
//! accumulation returned as f32. `is_available` gates on a configured overlay
//! (explicit opt-in). Non-matmul graphs → a clear compile error, no CPU masquerade.
use super::*;
use rlx_ir::op::MaskKind;
use rlx_ir::{Op, OpKind};
use rlx_xdna::aie::{
    BinaryOp as AieBinaryOp, ReduceOp as AieReduceOp, ScanOp, Ty, UnaryOp, emit_argmax,
    emit_attention, emit_binary, emit_cast, emit_clamp, emit_compare, emit_concat2, emit_expand,
    emit_fma, emit_gather, emit_group_norm, emit_layer_norm_affine, emit_matmul_microkernel,
    emit_narrow, emit_pad, emit_reduce, emit_reverse, emit_rms_norm_affine, emit_rope, emit_scan,
    emit_slice, emit_softmax, emit_tile, emit_transpose2d, emit_trilu, emit_unary, emit_where,
    tile_a_kacc, tile_b_kacc_multicol, untile_c_multicol,
};
use rlx_xdna::compile::{OverlaySpec, build_mm_kernel, compile_overlay, compile_overlay_linked};
use rlx_xdna::npu_gemm::{NpuGemm, NpuIo, NpuIoF32, NpuRun3};

pub struct XdnaBackend;

impl Backend for XdnaBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        &[
            OpKind::Input,
            OpKind::Param,
            OpKind::Constant,
            OpKind::MatMul,
            OpKind::Activation,
            OpKind::Softmax,
            OpKind::RmsNorm,
            OpKind::LayerNorm,
            OpKind::Attention,
            OpKind::Binary,
            OpKind::Reduce,
            OpKind::Reshape,
            OpKind::Clamp,
            OpKind::Transpose,
            OpKind::Narrow,
            OpKind::Slice,
            OpKind::Reverse,
            OpKind::Tile,
            OpKind::Expand,
            OpKind::Trilu,
            OpKind::Concat,
            OpKind::Gather,
            OpKind::Pad,
            OpKind::Where,
            OpKind::Fma,
            OpKind::StopGradient,
            OpKind::Cumsum,
            OpKind::CumProd,
            OpKind::CumMax,
            OpKind::Compare,
            OpKind::Cast,
            OpKind::ArgMax,
            OpKind::ArgMin,
            OpKind::GroupNorm,
            OpKind::Rope,
            OpKind::Quantize,
            OpKind::Dequantize,
            OpKind::Pool,
            OpKind::Im2Col,
        ]
    }

    fn compile(&self, graph: Graph, _options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // Opt-in TURBO: clock the NPU to max DPM for the (XRT) compute path. Power
        // mode is device-global, so a held direct fd raises the clock for XRT compute
        // too. Off unless RLX_XDNA_TURBO is set; needs root/DRM-master (else a clear
        // one-line warning and default DPM). Done once per process.
        enable_turbo_once();
        // MatMul → the fast C++-`aie::mmul`-microkernel path (compile-on-demand,
        // K-accum + 4 cols, ~638 GOP/s) when AIECC/PEANO are in the env; else the
        // fixed-shape INT8 overlay (`build_gemm_exec`, needs RLX_XDNA_GEMM). Any
        // other supported op → a compile-on-demand kernel (`build_op_exec`).
        let has_matmul = graph.nodes().iter().any(|n| matches!(n.op, Op::MatMul));
        let n_compute = graph
            .nodes()
            .iter()
            .filter(|n| {
                !matches!(
                    n.op,
                    Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
                )
            })
            .count();
        // A compute op reading a baked `Op::Constant` (bias/scale/table) needs the
        // chain path even at depth 1 — the chain seeds constants into the tensor map,
        // whereas `build_op_exec` has no way to feed a non-input tensor. (Matmul keeps
        // the microkernel path; its weight is a Param, not a threaded input.)
        let consumes_constant = graph
            .nodes()
            .iter()
            .filter(|n| {
                !matches!(
                    n.op,
                    Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
                )
            })
            .any(|n| {
                n.inputs
                    .iter()
                    .any(|id| matches!(graph.node(*id).op, Op::Constant { .. }))
            });
        let built = if n_compute > 1 || (consumes_constant && !has_matmul) {
            // multi-op subgraph → chain per-node execs (matmul-in-chain supported)
            build_chain(&graph).map(|e| Box::new(e) as Box<dyn ExecutableGraph>)
        } else if has_matmul {
            build_microkernel_exec(&graph)
                .map(|e| Box::new(e) as Box<dyn ExecutableGraph>)
                .or_else(|mk_err| {
                    build_gemm_exec(&graph).map(|e| Box::new(e) as Box<dyn ExecutableGraph>).map_err(|ov_err| {
                        format!("matmul: microkernel path ({mk_err}) AND overlay path ({ov_err}) both unavailable")
                    })
                })
        } else {
            build_op_exec(&graph)
        };
        built.unwrap_or_else(|e| panic!("XdnaBackend: {e}"))
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        self.compile(lir.into_graph(), options)
    }
}

/// Enable NPU TURBO power mode once per process when `RLX_XDNA_TURBO` is set. The
/// device handle is intentionally leaked (`forget`) so its fd stays open for the
/// whole session — the driver holds `pw_mode = TURBO` until the setting client closes,
/// so keeping the fd alive keeps the array at max DPM for the XRT compute path.
/// TURBO is device-global and independent of the (blocked) direct exec path; it needs
/// root/DRM-master, else a one-line warning and default DPM.
#[cfg(target_os = "linux")]
fn enable_turbo_once() {
    use std::sync::OnceLock;
    static TURBO: OnceLock<bool> = OnceLock::new();
    TURBO.get_or_init(|| {
        if std::env::var("RLX_XDNA_TURBO").is_err() {
            return false;
        }
        match rlx_xdna::direct::Npu::open("") {
            Ok(npu) => match npu.set_turbo() {
                Ok(()) => {
                    eprintln!("[rlx-xdna] NPU power mode → TURBO (max DPM) for the XRT compute path");
                    std::mem::forget(npu); // hold the fd open so the mode persists
                    true
                }
                Err(e) => {
                    eprintln!("[rlx-xdna] TURBO request failed: {e} — needs root/DRM-master; running at default DPM");
                    false
                }
            },
            Err(e) => {
                eprintln!("[rlx-xdna] TURBO: cannot open the accel device: {e}");
                false
            }
        }
    });
}
#[cfg(not(target_os = "linux"))]
fn enable_turbo_once() {}

/// Map an rlx `Activation` to the NPU unary emitter's op (the supported subset;
/// the pure-arith transcendentals cover exp/log/sqrt/rsqrt/tanh/sigmoid/silu/gelu).
fn map_activation(a: &rlx_ir::op::Activation) -> Result<UnaryOp, String> {
    use rlx_ir::op::Activation as A;
    Ok(match a {
        A::Relu => UnaryOp::Relu,
        A::Neg => UnaryOp::Neg,
        A::Abs => UnaryOp::Abs,
        A::Exp => UnaryOp::Exp,
        A::Log => UnaryOp::Log,
        A::Sqrt => UnaryOp::Sqrt,
        A::Rsqrt => UnaryOp::Rsqrt,
        A::Tanh => UnaryOp::Tanh,
        A::Recip => UnaryOp::Recip,
        A::Sigmoid => UnaryOp::Sigmoid,
        A::Silu => UnaryOp::Silu,
        // The NPU gelu is the tanh approximation — exact-erf `Gelu` maps to it
        // too (deviates ~1% near the transition; models generally tolerate this).
        A::Gelu | A::GeluApprox => UnaryOp::Gelu,
        A::Floor => UnaryOp::Floor,
        A::Ceil => UnaryOp::Ceil,
        A::Round => UnaryOp::Round,
        A::Sign => UnaryOp::Sign,
        A::Softplus => UnaryOp::Softplus,
        A::Elu => UnaryOp::Elu,
        A::HardSwish => UnaryOp::HardSwish,
        A::HardSigmoid => UnaryOp::HardSigmoid,
        A::Mish => UnaryOp::Mish,
        A::Softsign => UnaryOp::Softsign,
        A::LogSigmoid => UnaryOp::LogSigmoid,
        A::Sin => UnaryOp::Sin,
        A::Cos => UnaryOp::Cos,
        A::Erf => UnaryOp::Erf,
        // Tan / Atan not yet emitted (rare; need double-sin / range-reduced poly).
        other => return Err(format!("activation {other:?} not yet on the NPU")),
    })
}

/// Map an rlx f32 arithmetic `BinaryOp` to the NPU emitter's op. Bitwise/shift
/// (i32-only) and Pow/Atan2 (not emitted yet) return an error.
fn map_binary(op: rlx_ir::op::BinaryOp) -> Result<AieBinaryOp, String> {
    use rlx_ir::op::BinaryOp as B;
    Ok(match op {
        B::Add => AieBinaryOp::Add,
        B::Sub => AieBinaryOp::Sub,
        B::Mul => AieBinaryOp::Mul,
        B::Div => AieBinaryOp::Div,
        B::Max => AieBinaryOp::Max,
        B::Min => AieBinaryOp::Min,
        B::Mod => AieBinaryOp::Mod,
        other => {
            return Err(format!(
                "binary {other:?} not on the NPU (f32 arithmetic only)"
            ));
        }
    })
}

/// Split static `dims` at `axis` into (outer, axis_len, inner) for the NPU
/// data-movement engines.
fn axis_split(dims: &[usize], axis: usize) -> (usize, usize, usize) {
    (
        dims[..axis].iter().product(),
        dims[axis],
        dims[axis + 1..].iter().product(),
    )
}

fn map_reduce(op: rlx_ir::op::ReduceOp) -> AieReduceOp {
    use rlx_ir::op::ReduceOp as R;
    match op {
        R::Sum => AieReduceOp::Sum,
        R::Mean => AieReduceOp::Mean,
        R::Max => AieReduceOp::Max,
        R::Min => AieReduceOp::Min,
        R::Prod => AieReduceOp::Prod,
    }
}

/// Largest chunk ≤2048 that divides `n` (the streamed unary kernels need
/// `n % chunk == 0`; f32 is scalar so any divisor works). Capped at 2048 so the
/// in+out double-buffered f32 tiles (4 × chunk·4 B) stay within the 64 KB tile.
fn pick_chunk(n: usize) -> usize {
    for c in [2048usize, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
        if n % c == 0 {
            return c;
        }
    }
    1
}

/// Build a compile-on-demand executable: find the compute op, emit its AIE-MLIR
/// kernel (pure Rust), compile it with the native aiecc, and open a persistent
/// NPU context. Activation/Softmax → 1-in `NpuIoF32`; RmsNorm/LayerNorm → the
/// generic 3-buffer `NpuRun3` (x, gamma‖beta, out). Needs `AIECC` + `PEANO`.
/// Synthetic name for a node's produced tensor when it feeds another op (chains):
/// graph Inputs/Params keep their names, intermediate compute nodes get `n{id}`.
fn node_ref(graph: &Graph, id: rlx_ir::graph::NodeId) -> String {
    match &graph.node(id).op {
        Op::Input { name } | Op::Param { name } => name.clone(),
        _ => format!("n{}", id.0),
    }
}

/// True when input `idx` of `consumer` is a **Param** consumed by its NPU exec via
/// `set_param` (matmul weight, norm gamma/beta) rather than threaded through the chain
/// tensor map. The Param check matters for backward: a matmul weight that is a dynamic
/// intermediate (`xᵀ @ dy`) is NOT a set_param input — it must be threaded. Every other
/// input (graph Input, intermediate, Constant, or a Param used as plain backward data)
/// is threaded (its value seeded into the map).
fn is_set_param_input(consumer: &Op, idx: usize, input: &Op) -> bool {
    matches!(input, Op::Param { .. })
        && matches!(
            (consumer, idx),
            (Op::MatMul, 1)
                | (Op::RmsNorm { .. }, 1)
                | (Op::RmsNorm { .. }, 2)
                | (Op::LayerNorm { .. }, 1)
                | (Op::LayerNorm { .. }, 2)
                | (Op::GroupNorm { .. }, 1)
                | (Op::GroupNorm { .. }, 2)
        )
}

/// Decode an `Op::Constant`'s raw little-endian bytes into the runtime f32
/// representation, per the node's dtype. Covers the numeric dtypes a chain constant
/// realistically carries (bias/scale/position-table); packed bool/i8 constants — which
/// the runtime stores byte-packed, not one value per element — are rejected.
fn decode_constant_f32(data: &[u8], dtype: rlx_ir::DType) -> Result<Vec<f32>, String> {
    use rlx_ir::DType::*;
    Ok(match dtype {
        F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        other => {
            return Err(format!(
                "NPU chain: Constant dtype {other:?} unsupported as a threaded input"
            ));
        }
    })
}

fn build_op_exec(graph: &Graph) -> Result<Box<dyn ExecutableGraph>, String> {
    let node_id = graph
        .nodes()
        .iter()
        .position(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            )
        })
        .ok_or("no compute op in graph")?;
    build_node_exec(graph, node_id)
}

/// MULTI-OP subgraph: build one [`build_node_exec`] per compute node (topo order =
/// graph insertion order) and thread each node's output tensor (keyed `n{id}`) to
/// the ops that consume it. Each op is still its own NPU dispatch; the chain runs
/// them in order. Scoped to the `build_node_exec`-supported ops (no matmul in-chain
/// yet — a MatMul graph takes the single-op microkernel path).
fn build_chain(graph: &Graph) -> Result<XdnaChainExec, String> {
    std::env::var("AIECC").map_err(|_| "chain needs AIECC in env".to_string())?;
    std::env::var("PEANO").map_err(|_| "chain needs PEANO in env".to_string())?;
    let mut steps: Vec<ChainStep> = Vec::new();
    for (idx, node) in graph.nodes().iter().enumerate() {
        if matches!(
            node.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        ) {
            continue;
        }
        let is_matmul = matches!(node.op, Op::MatMul);
        // Only the "structural" Param inputs (matmul weight, norm gamma/beta) arrive
        // via forwarded set_param — those are excluded here. EVERY other input is
        // threaded through the tensor map, INCLUDING Params used as plain data (which
        // is how backward graphs consume weights: `relu'` needs x, `dL/dx` needs Wᵀ).
        let input_names: Vec<String> = node
            .inputs
            .iter()
            .enumerate()
            .filter(|(idx, id)| !is_set_param_input(&node.op, *idx, &graph.node(**id).op))
            .map(|(_, id)| node_ref(graph, *id))
            .collect();
        steps.push(ChainStep {
            node_idx: idx,
            out_name: node_ref(graph, node.id),
            input_names,
            is_matmul,
        });
    }
    // Decode the baked bytes of every Constant a step actually threads, so `run` can
    // seed them into the tensor map alongside the graph Inputs.
    let referenced: std::collections::HashSet<&str> = steps
        .iter()
        .flat_map(|s| s.input_names.iter().map(String::as_str))
        .collect();
    let mut constants: Vec<(String, Vec<f32>)> = Vec::new();
    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op {
            let name = node_ref(graph, node.id);
            if referenced.contains(name.as_str()) {
                constants.push((name, decode_constant_f32(data, node.shape.dtype())?));
            }
        }
    }
    let out_id = *graph.outputs.first().ok_or("chain: graph has no output")?;
    Ok(XdnaChainExec {
        graph: graph.clone(),
        steps,
        output_name: node_ref(graph, out_id),
        constants,
        params: Vec::new(),
        typed_params: Vec::new(),
        cache: std::collections::HashMap::new(),
        lru: Vec::new(),
        // Concurrent NPU hardware contexts are limited (~5-6 on amdxdna); keep an
        // LRU pool of open sub-exec contexts and re-open evicted ones on demand.
        cap: std::env::var("RLX_XDNA_CHAIN_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
    })
}

#[derive(Clone)]
struct ChainStep {
    node_idx: usize,
    out_name: String,
    input_names: Vec<String>,
    is_matmul: bool,
}

/// Runs a topo-ordered chain of per-node execs, threading intermediates by name.
/// Sub-execs (each an NPU hardware context) are built LAZILY at run and kept in a
/// bounded LRU pool — evicted contexts re-open cheaply from the cached xclbin
/// (no aiecc), with host state restored via the stored params. Lets chains run
/// deeper than the NPU's concurrent-hw-context limit.
struct XdnaChainExec {
    graph: Graph,
    steps: Vec<ChainStep>,
    output_name: String,
    /// Baked `Op::Constant` tensors (name → f32 values) seeded into the tensor map at
    /// the start of every `run`, so chain ops can consume constants like graph Inputs.
    constants: Vec<(String, Vec<f32>)>,
    params: Vec<(String, Vec<f32>)>,
    typed_params: Vec<(String, Vec<u8>, rlx_ir::DType)>,
    cache: std::collections::HashMap<usize, Box<dyn ExecutableGraph>>,
    lru: Vec<usize>,
    cap: usize,
}
unsafe impl Send for XdnaChainExec {}

impl ExecutableGraph for XdnaChainExec {
    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params.retain(|(n, _)| n != name);
        self.params.push((name.to_string(), data.to_vec()));
        for ex in self.cache.values_mut() {
            ex.set_param(name, data);
        }
    }
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        self.typed_params.retain(|(n, _, _)| n != name);
        self.typed_params
            .push((name.to_string(), data.to_vec(), dtype));
        for ex in self.cache.values_mut() {
            ex.set_param_typed(name, data, dtype);
        }
    }
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut map: std::collections::HashMap<String, Vec<f32>> = inputs
            .iter()
            .map(|(nm, d)| (nm.to_string(), d.to_vec()))
            .collect();
        for (nm, d) in &self.constants {
            map.entry(nm.clone()).or_insert_with(|| d.clone());
        }
        // Params are ALSO seeded as threaded data: backward graphs read weights
        // directly (e.g. dL/dx = Wᵀ·dy). The set_param path still delivers the same
        // params to the structural execs (matmul weight, norm gamma) in parallel.
        for (nm, d) in &self.params {
            map.entry(nm.clone()).or_insert_with(|| d.clone());
        }
        let steps = self.steps.clone();
        for step in &steps {
            // Ensure this step's exec (NPU context) is open — build lazily, evicting
            // the LRU context if the pool is full.
            if !self.cache.contains_key(&step.node_idx) {
                while self.cache.len() >= self.cap {
                    if self.lru.is_empty() {
                        break;
                    }
                    let victim = self.lru.remove(0);
                    self.cache.remove(&victim); // Drop closes the hw context
                }
                let node = &self.graph.nodes()[step.node_idx];
                let mut ex: Box<dyn ExecutableGraph> = if step.is_matmul {
                    Box::new(
                        build_matmul_node_exec(&self.graph, node)
                            .unwrap_or_else(|e| panic!("XdnaBackend chain matmul: {e}")),
                    )
                } else {
                    build_node_exec(&self.graph, step.node_idx)
                        .unwrap_or_else(|e| panic!("XdnaBackend chain op: {e}"))
                };
                for (nm, d) in &self.params {
                    ex.set_param(nm, d);
                }
                for (nm, d, dt) in &self.typed_params {
                    ex.set_param_typed(nm, d, *dt);
                }
                self.cache.insert(step.node_idx, ex);
            }
            self.lru.retain(|&x| x != step.node_idx);
            self.lru.push(step.node_idx);

            let sub_in: Vec<(&str, &[f32])> = step
                .input_names
                .iter()
                .map(|nm| {
                    let t = map
                        .get(nm)
                        .unwrap_or_else(|| panic!("XdnaChainExec: tensor '{nm}' not yet produced"));
                    (nm.as_str(), t.as_slice())
                })
                .collect();
            let exec = self.cache.get_mut(&step.node_idx).unwrap();
            let out = exec.run(&sub_in).into_iter().next().unwrap_or_default();
            drop(sub_in);
            map.insert(step.out_name.clone(), out);
        }
        vec![
            map.remove(&self.output_name)
                .unwrap_or_else(|| panic!("XdnaChainExec: output '{}' missing", self.output_name)),
        ]
    }
}

/// Build a compile-on-demand executable for ONE compute node — emit its AIE-MLIR
/// kernel (pure Rust), compile via native aiecc, open a persistent NPU context.
/// Inputs are resolved by [`node_ref`] (graph Input/Param name, or `n{id}` for an
/// intermediate produced by an earlier node in a [`XdnaChainExec`]). Needs
/// `AIECC` + `PEANO`.
fn build_node_exec(graph: &Graph, node_id: usize) -> Result<Box<dyn ExecutableGraph>, String> {
    let aiecc = std::env::var("AIECC")
        .map_err(|_| "set AIECC (native mlir-aie) for on-demand NPU op compile".to_string())?;
    let peano = std::env::var("PEANO").map_err(|_| "set PEANO (llvm-aie dir)".to_string())?;

    let node = &graph.nodes()[node_id];
    let in_id = *node.inputs.first().ok_or("op has no input")?;
    // Input 0 is threaded by name from the chain tensor map — a graph Input, an
    // intermediate `n{id}`, a seeded Constant, or a seeded data-Param (backward).
    let x_name = node_ref(graph, in_id);
    let dims: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    if dims.is_empty() {
        return Err("op output must have a static shape".into());
    }
    let n: usize = dims.iter().product();
    let cols = *dims.last().unwrap();
    let rows = n / cols;

    // emit → compile → (xclbin path, insts).
    let build = |mlir: &str, tag: &str| -> Result<(String, Vec<u32>), String> {
        let tmp = format!("{}/rlx_xdna_{tag}_{n}", std::env::temp_dir().display());
        std::fs::create_dir_all(&tmp).map_err(|e| format!("mkdir {tmp}: {e}"))?;
        let mp = format!("{tmp}/aie.mlir");
        let xclbin = format!("{tmp}/k.xclbin");
        let insts_path = format!("{tmp}/insts.bin");
        // Cache keyed by the exact MLIR content (a tag can collide — e.g. NeoX vs
        // GptJ rope share rows×head_dim): skip aiecc only when the cached aie.mlir
        // is byte-identical AND the artifacts exist. Lets a chain cheaply RE-OPEN a
        // sub-exec's context without recompiling.
        let cached = std::fs::read_to_string(&mp).ok().as_deref() == Some(mlir)
            && std::path::Path::new(&xclbin).exists()
            && std::path::Path::new(&insts_path).exists();
        if !cached {
            std::fs::write(&mp, mlir).map_err(|e| format!("write mlir: {e}"))?;
            compile_overlay(&OverlaySpec {
                aiecc: &aiecc,
                peano: &peano,
                mlir: &mp,
                tmpdir: &format!("{tmp}/build"),
                out_xclbin: &xclbin,
                out_insts: &insts_path,
            })
            .map_err(|e| e.0)?;
        }
        let insts = std::fs::read(&insts_path)
            .map_err(|e| format!("read insts: {e}"))?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((xclbin, insts))
    };
    // Param name at input index `i` (gamma / beta for the norms).
    let param_name = |i: usize| -> Option<String> {
        node.inputs.get(i).and_then(|id| match &graph.node(*id).op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
    };
    // Name of the threaded input at index `i` — graph Input, intermediate `n{id}`,
    // seeded Constant, or a seeded data-Param (backward). The structural set_param
    // inputs (matmul weight, norm gamma/beta) are resolved via `param_name` instead.
    let input_name =
        |i: usize| -> Option<String> { node.inputs.get(i).map(|id| node_ref(graph, *id)) };
    // Static dims of input 0 (the data-movement arms all read this).
    let in_dims = || -> Vec<usize> {
        graph
            .node(in_id)
            .shape
            .dims()
            .iter()
            .map(|x| x.unwrap_static())
            .collect()
    };
    // Collapse the shared 1-in data-movement tail: emit → compile → open `NpuRun3`
    // (arg0=in `$n_in`, arg1 dummy, arg2=out `n`) → box an `XdnaDmExec`.
    macro_rules! dm1 {
        ($mlir:expr, $tag:expr, $n_in:expr) => {{
            let (xclbin, insts) = build(&$mlir, &$tag)?;
            let io = NpuRun3::open("", &xclbin, &insts, $n_in, 1, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaDmExec {
                io,
                in_name: x_name.clone(),
                n_in: $n_in,
            }) as Box<dyn ExecutableGraph>)
        }};
    }
    // Last-axis cumulative scan (Cumsum/CumProd/CumMax) → 1-in f32 `XdnaOpExec`.
    macro_rules! scan_exec {
        ($kind:expr, $axis:expr, $exclusive:expr) => {{
            let ax = if $axis < 0 {
                dims.len() as i32 + $axis
            } else {
                $axis
            };
            if ax != dims.len() as i32 - 1 {
                return Err(format!("NPU scan: last axis only (got {})", $axis));
            }
            let (xclbin, insts) = build(
                &emit_scan($kind, rows, cols, $exclusive),
                &format!("scan_{rows}x{cols}"),
            )?;
            let io = NpuIoF32::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaOpExec {
                io,
                in_name: x_name.clone(),
                n,
            }) as Box<dyn ExecutableGraph>)
        }};
    }
    // ArgMax($is_max=true)/ArgMin over the last axis → f32-encoded index, one per
    // row (`XdnaReduceExec` reads col 0 of the broadcast output).
    macro_rules! argmax_exec {
        ($is_max:expr, $axis:expr) => {{
            let d = in_dims();
            let rank = d.len();
            if rank < 1 || $axis != rank - 1 {
                return Err(format!(
                    "NPU argmax/min: last axis only (axis {}, rank {})",
                    $axis, rank
                ));
            }
            let cols = d[rank - 1];
            let n_in: usize = d.iter().product();
            let rows = n_in / cols;
            let (xclbin, insts) = build(
                &emit_argmax(rows, cols, $is_max),
                &format!("argmax_{rows}x{cols}"),
            )?;
            let io = NpuIo::open("", &xclbin, &insts, n_in).map_err(|e| e.0)?;
            Ok(Box::new(XdnaReduceExec {
                io,
                in_name: x_name.clone(),
                n_in,
                rows,
                cols,
            }) as Box<dyn ExecutableGraph>)
        }};
    }

    match &node.op {
        Op::Activation(act) => {
            let uop = map_activation(act)?;
            let (xclbin, insts) = build(
                &emit_unary(uop, Ty::F32, n, pick_chunk(n)),
                &format!("act_{}", uop.name()),
            )?;
            let io = NpuIoF32::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaOpExec {
                io,
                in_name: x_name,
                n,
            }))
        }
        Op::Softmax { axis } => {
            let ax = if *axis < 0 {
                dims.len() as i32 + axis
            } else {
                *axis
            };
            if ax != dims.len() as i32 - 1 {
                return Err(format!(
                    "NPU softmax supports only the last axis (got {axis})"
                ));
            }
            let (xclbin, insts) =
                build(&emit_softmax(rows, cols), &format!("softmax_{rows}x{cols}"))?;
            let io = NpuIoF32::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaOpExec {
                io,
                in_name: x_name,
                n,
            }))
        }
        Op::RmsNorm { eps, .. } => {
            let gamma_name = param_name(1).ok_or("RmsNorm input 1 (gamma) must be a Param")?;
            let (xclbin, insts) = build(
                &emit_rms_norm_affine(rows, cols, *eps),
                &format!("rmsnorm_{rows}x{cols}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * cols, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaNormExec {
                io,
                x_name,
                gamma_name,
                beta_name: param_name(2),
                gamma: Vec::new(),
                beta: Vec::new(),
                cols,
                n,
            }))
        }
        Op::LayerNorm { eps, .. } => {
            let gamma_name = param_name(1).ok_or("LayerNorm input 1 (gamma) must be a Param")?;
            let (xclbin, insts) = build(
                &emit_layer_norm_affine(rows, cols, *eps),
                &format!("layernorm_{rows}x{cols}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * cols, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaNormExec {
                io,
                x_name,
                gamma_name,
                beta_name: param_name(2),
                gamma: Vec::new(),
                beta: Vec::new(),
                cols,
                n,
            }))
        }
        Op::Rope {
            head_dim,
            n_rot,
            style,
        } => {
            use rlx_ir::op::RopeStyle;
            let dx = in_dims(); // x: [.., hidden], hidden = nh·head_dim
            let hidden = *dx.last().ok_or("rope: x needs rank ≥ 1")?;
            if hidden % head_dim != 0 {
                return Err(format!(
                    "NPU rope: hidden ({hidden}) not divisible by head_dim ({head_dim})"
                ));
            }
            let nh = hidden / head_dim;
            let rows_flat: usize = dx[..dx.len() - 1].iter().product(); // n_tokens (batch·seq)
            let rows = rows_flat * nh;
            let tab_half = head_dim / 2;
            let half = rows_flat * tab_half; // cos (and sin) element count
            let cos_name = input_name(1).ok_or("rope cos must be a graph Input")?;
            let sin_name = input_name(2).ok_or("rope sin must be a graph Input")?;
            let neox = matches!(style, RopeStyle::NeoX);
            let (xclbin, insts) = build(
                &emit_rope(rows, *head_dim, *n_rot, nh, neox),
                &format!("rope_{rows}x{head_dim}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * half, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaRopeExec {
                io,
                x_name,
                cos_name,
                sin_name,
                half,
            }))
        }
        Op::GroupNorm { num_groups, eps } => {
            let d = in_dims(); // NCHW (or NC*spatial), rank ≥ 3
            if d.len() < 3 {
                return Err(format!("NPU groupnorm: needs rank ≥ 3 NCHW (got {d:?})"));
            }
            let (nb, c) = (d[0], d[1]);
            let spatial: usize = d[2..].iter().product();
            if c % num_groups != 0 {
                return Err(format!(
                    "NPU groupnorm: C ({c}) not divisible by num_groups ({num_groups})"
                ));
            }
            let cg = c / num_groups;
            let (rows, group_size) = (nb * num_groups, cg * spatial);
            let gamma_name = param_name(1).ok_or("GroupNorm input 1 (gamma) must be a Param")?;
            let (xclbin, insts) = build(
                &emit_group_norm(rows, group_size, *num_groups, cg, spatial, *eps),
                &format!("groupnorm_{rows}x{group_size}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * c, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaNormExec {
                io,
                x_name,
                gamma_name,
                beta_name: param_name(2),
                gamma: Vec::new(),
                beta: Vec::new(),
                cols: c,
                n,
            }))
        }
        Op::Attention {
            num_heads,
            head_dim,
            mask_kind,
            score_scale,
            attn_logit_softcap,
        } => {
            // Scoped: None/Causal mask, no softcap, batch=1; multi-head via Q's
            // hidden dim = num_heads · head_dim.
            let causal = match mask_kind {
                MaskKind::None => false,
                MaskKind::Causal => true,
                other => {
                    return Err(format!(
                        "NPU attention: only None/Causal mask (got {other:?})"
                    ));
                }
            };
            if attn_logit_softcap.is_some() {
                return Err("NPU attention: logit softcap not supported".into());
            }
            if dims.len() < 2 {
                return Err("NPU attention: Q needs rank ≥ 2".into());
            }
            let d = *head_dim;
            let nh = *num_heads;
            let hidden = nh * d;
            let seq = dims[dims.len() - 2];
            if n != seq * hidden {
                return Err(format!(
                    "NPU attention: batch=1 only (shape {dims:?}, seq·heads·head_dim={})",
                    seq * hidden
                ));
            }
            let scale = score_scale.unwrap_or(1.0 / (d as f32).sqrt());
            let k_name = input_name(1).ok_or("attention K must be a graph Input")?;
            let v_name = input_name(2).ok_or("attention V must be a graph Input")?;
            let (xclbin, insts) = build(
                &emit_attention(seq, d, nh, scale, causal),
                &format!("attn_{}_{seq}x{nh}x{d}", if causal { "c" } else { "n" }),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * n, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaAttnExec {
                io,
                q_name: x_name,
                k_name,
                v_name,
                n,
            }))
        }
        Op::Binary(binop) => {
            if node.shape.dtype() != rlx_ir::DType::F32 {
                return Err("NPU binary: only f32 tensors".into());
            }
            let aop = map_binary(*binop)?;
            let b_name = input_name(1).ok_or("binary input 1 must be a graph Input")?;
            let (xclbin, insts) = build(
                &emit_binary(aop, Ty::F32, n, pick_chunk(n)),
                &format!("bin_{}", aop.name()),
            )?;
            let io = NpuIo::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaBinExec {
                io,
                a_name: x_name,
                b_name,
                n,
            }))
        }
        Op::Reduce { op, axes, .. } => {
            let in_dims: Vec<usize> = graph
                .node(in_id)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            let rank = in_dims.len();
            if rank < 1 || axes.len() != 1 || axes[0] != rank - 1 {
                return Err(format!(
                    "NPU reduce: single last-axis only (axes {axes:?}, rank {rank})"
                ));
            }
            let cols = in_dims[rank - 1];
            let n_in: usize = in_dims.iter().product();
            let rows = n_in / cols;
            let aop = map_reduce(*op);
            let (xclbin, insts) = build(
                &emit_reduce(aop, rows, cols),
                &format!("reduce_{}_{rows}x{cols}", aop.name()),
            )?;
            let io = NpuIo::open("", &xclbin, &insts, n_in).map_err(|e| e.0)?;
            Ok(Box::new(XdnaReduceExec {
                io,
                in_name: x_name,
                n_in,
                rows,
                cols,
            }))
        }
        // ── data-movement / shape ops (dtype-agnostic f32-cell moves) ──
        Op::Reshape { .. } => dm1!(emit_narrow(1, n, 1, 0, n), format!("reshape_{n}"), n),
        Op::Clamp { min, max } => dm1!(emit_clamp(n, *min, *max), format!("clamp_{n}"), n),
        Op::Transpose { perm } => {
            let d = in_dims();
            if d.len() != 2 || perm.as_slice() != [1, 0] {
                return Err(format!(
                    "NPU transpose: 2-D [1,0] only (dims {d:?}, perm {perm:?})"
                ));
            }
            dm1!(
                emit_transpose2d(d[0], d[1]),
                format!("transpose_{}x{}", d[0], d[1]),
                n
            )
        }
        Op::Trilu { upper, diagonal } => {
            let d = in_dims();
            if d.len() < 2 {
                return Err("NPU trilu: needs rank ≥ 2".into());
            }
            let (rows, cols) = (d[d.len() - 2], d[d.len() - 1]);
            if rows * cols != n {
                return Err("NPU trilu: batched (>2-D) not yet supported".into());
            }
            dm1!(
                emit_trilu(rows, cols, *upper, *diagonal),
                format!("trilu_{rows}x{cols}"),
                n
            )
        }
        Op::Reverse { axes } => {
            let d = in_dims();
            if axes.len() != 1 {
                return Err(format!("NPU reverse: single axis only (got {axes:?})"));
            }
            let (outer, mid, inner) = axis_split(&d, axes[0]);
            dm1!(
                emit_reverse(outer, mid, inner),
                format!("reverse_{}", axes[0]),
                n
            )
        }
        Op::Narrow { axis, start, len } => {
            let d = in_dims();
            let n_in: usize = d.iter().product();
            let (outer, mid, inner) = axis_split(&d, *axis);
            dm1!(
                emit_narrow(outer, mid, inner, *start, *len),
                format!("narrow_{axis}_{start}_{len}"),
                n_in
            )
        }
        Op::Slice {
            axis,
            start,
            len,
            step,
        } => {
            let d = in_dims();
            let n_in: usize = d.iter().product();
            let (outer, mid, inner) = axis_split(&d, *axis);
            dm1!(
                emit_slice(outer, mid, inner, *start, *len, *step),
                format!("slice_{axis}"),
                n_in
            )
        }
        Op::Tile { reps } => {
            let d = in_dims();
            let n_in: usize = d.iter().product();
            let nz: Vec<usize> = (0..reps.len()).filter(|&i| reps[i] != 1).collect();
            if nz.len() != 1 {
                return Err(format!(
                    "NPU tile: single repeated axis only (reps {reps:?})"
                ));
            }
            let (outer, mid, inner) = axis_split(&d, nz[0]);
            dm1!(
                emit_tile(outer, mid, inner, reps[nz[0]]),
                format!("tile_{}", nz[0]),
                n_in
            )
        }
        Op::Expand { target_shape } => {
            let d = in_dims();
            let n_in: usize = d.iter().product();
            let tgt: Vec<usize> = target_shape.iter().map(|&x| x as usize).collect();
            let bc: Vec<usize> = (0..d.len()).filter(|&i| d[i] == 1 && tgt[i] > 1).collect();
            if bc.len() != 1 {
                return Err(format!(
                    "NPU expand: single broadcast axis only (dims {d:?} → {tgt:?})"
                ));
            }
            let (outer, _mid, inner) = axis_split(&d, bc[0]);
            dm1!(
                emit_expand(outer, inner, tgt[bc[0]]),
                format!("expand_{}", bc[0]),
                n_in
            )
        }
        Op::Concat { axis } => {
            if node.inputs.len() != 2 {
                return Err(format!(
                    "NPU concat: 2 inputs only (got {})",
                    node.inputs.len()
                ));
            }
            let da: Vec<usize> = graph
                .node(node.inputs[0])
                .shape
                .dims()
                .iter()
                .map(|x| x.unwrap_static())
                .collect();
            let db: Vec<usize> = graph
                .node(node.inputs[1])
                .shape
                .dims()
                .iter()
                .map(|x| x.unwrap_static())
                .collect();
            let (outer, a_axis, inner) = axis_split(&da, *axis);
            let b_axis = db[*axis];
            let a_name = input_name(0).ok_or("concat input 0 must be a graph Input")?;
            let b_name = input_name(1).ok_or("concat input 1 must be a graph Input")?;
            let (na, nb) = (da.iter().product::<usize>(), db.iter().product::<usize>());
            let (xclbin, insts) = build(
                &emit_concat2(outer, a_axis, b_axis, inner),
                &format!("concat_{axis}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, na, nb, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaDm2Exec {
                io,
                a_name,
                b_name,
                na,
                nb,
            }))
        }
        Op::Gather { axis } => {
            // data = inputs[0], idx = inputs[1] (f32-encoded indices).
            let dd: Vec<usize> = graph
                .node(node.inputs[0])
                .shape
                .dims()
                .iter()
                .map(|x| x.unwrap_static())
                .collect();
            let di: Vec<usize> = graph
                .node(*node.inputs.get(1).ok_or("gather needs 2 inputs")?)
                .shape
                .dims()
                .iter()
                .map(|x| x.unwrap_static())
                .collect();
            let (outer, in_axis, inner) = axis_split(&dd, *axis);
            let num_idx: usize = di.iter().product();
            let a_name = input_name(0).ok_or("gather data must be a graph Input")?;
            let b_name = input_name(1).ok_or("gather idx must be a graph Input")?;
            let (nd, ni) = (dd.iter().product::<usize>(), num_idx);
            let (xclbin, insts) = build(
                &emit_gather(outer, in_axis, inner, num_idx),
                &format!("gather_{axis}"),
            )?;
            let io = NpuRun3::open("", &xclbin, &insts, nd, ni, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaDm2Exec {
                io,
                a_name,
                b_name,
                na: nd,
                nb: ni,
            }))
        }
        // forward identity — a passthrough copy (same bytes).
        Op::StopGradient => dm1!(emit_narrow(1, n, 1, 0, n), format!("stopgrad_{n}"), n),
        Op::Cast { to } => {
            // f32↔i32 only (both 4-byte cells → 1:1, no byte-repacking). Numeric
            // convert: fptosi / sitofp (the i32 bits ride in the f32 buffer).
            let in_dt = graph.node(in_id).shape.dtype();
            let f2i = match (in_dt, to) {
                (rlx_ir::DType::F32, rlx_ir::DType::I32) => true,
                (rlx_ir::DType::I32, rlx_ir::DType::F32) => false,
                _ => return Err(format!("NPU cast: only f32↔i32 (got {in_dt:?}→{to:?})")),
            };
            let (xclbin, insts) = build(&emit_cast(n, f2i), &format!("cast_{n}"))?;
            let io = NpuIoF32::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaOpExec {
                io,
                in_name: x_name,
                n,
            }))
        }
        Op::Cumsum { axis, exclusive } => scan_exec!(ScanOp::Sum, *axis, *exclusive),
        Op::CumProd { axis, exclusive } => scan_exec!(ScanOp::Prod, *axis, *exclusive),
        Op::CumMax { axis, exclusive } => scan_exec!(ScanOp::Max, *axis, *exclusive),
        Op::ArgMax { axis, .. } => argmax_exec!(true, *axis),
        Op::ArgMin { axis, .. } => argmax_exec!(false, *axis),
        Op::Pad { pads, mode } => {
            use rlx_ir::op::PadMode;
            let d = in_dims();
            let n_in: usize = d.iter().product();
            let padded: Vec<usize> = (0..pads.len()).filter(|&i| pads[i] != [0, 0]).collect();
            if padded.len() != 1 {
                return Err(format!("NPU pad: single padded axis only (pads {pads:?})"));
            }
            let fill = match mode {
                PadMode::Constant(f) => *f,
                _ => return Err(format!("NPU pad: Constant mode only (got {mode:?})")),
            };
            let ax = padded[0];
            let (outer, in_axis, inner) = axis_split(&d, ax);
            let (before, after) = (pads[ax][0], pads[ax][1]);
            dm1!(
                emit_pad(outer, in_axis, inner, before, after, fill),
                format!("pad_{ax}"),
                n_in
            )
        }
        Op::Compare(cmp) => {
            use rlx_ir::op::CmpOp as C;
            let pred = match cmp {
                C::Eq => "oeq",
                C::Ne => "one",
                C::Lt => "olt",
                C::Le => "ole",
                C::Gt => "ogt",
                C::Ge => "oge",
            };
            let b_name = input_name(1).ok_or("compare input 1 must be a graph Input")?;
            let (xclbin, insts) = build(&emit_compare(pred, n), &format!("compare_{pred}_{n}"))?;
            let io = NpuIo::open("", &xclbin, &insts, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaCompareExec {
                io,
                a_name: x_name,
                b_name,
                n,
            }))
        }
        Op::Where => {
            // inputs: [cond, a, b]; a‖b packed into arg1 by the exec.
            let a_name = input_name(1).ok_or("where input 1 (a) must be a graph Input")?;
            let b_name = input_name(2).ok_or("where input 2 (b) must be a graph Input")?;
            let (xclbin, insts) = build(&emit_where(n), &format!("where_{n}"))?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * n, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaTernExec {
                io,
                a_name: x_name,
                b_name: a_name,
                c_name: b_name,
                n,
            }))
        }
        Op::Fma => {
            // inputs: [a, b, c]; b‖c packed into arg1 by the exec (out = a*b + c).
            let b_name = input_name(1).ok_or("fma input 1 (b) must be a graph Input")?;
            let c_name = input_name(2).ok_or("fma input 2 (c) must be a graph Input")?;
            let (xclbin, insts) = build(&emit_fma(n), &format!("fma_{n}"))?;
            let io = NpuRun3::open("", &xclbin, &insts, n, 2 * n, n).map_err(|e| e.0)?;
            Ok(Box::new(XdnaTernExec {
                io,
                a_name: x_name,
                b_name,
                c_name,
                n,
            }))
        }
        Op::Quantize {
            axis,
            scales,
            zero_points,
        } => {
            // f32 → packed-i8 (dtype boundary). Scales/zero-points are BAKED (static),
            // and the per-channel affine+round+clamp is a trivial memory-bound pass →
            // computed on host, then packed 4 codes per f32-cell (the runtime's i8
            // layout, same as the bool path). Coverage so int8 graphs RUN on
            // Device::Xdna; there is no NPU compute win in a lone baked-scale quant.
            let (chan_dim, inner) = quant_layout(&in_dims(), *axis);
            Ok(Box::new(XdnaQuantizeExec {
                x_name,
                len: n,
                chan_dim,
                inner,
                scales: scales.clone(),
                zero_points: zero_points.clone(),
            }))
        }
        Op::Dequantize {
            axis,
            scales,
            zero_points,
        } => {
            // packed-i8 → f32 (inverse of Quantize). Unpack the i8 codes from the f32
            // cell bytes, then x = (q − zp)·scale per channel. Host-computed twin.
            let (chan_dim, inner) = quant_layout(&in_dims(), *axis);
            Ok(Box::new(XdnaDequantizeExec {
                q_name: x_name,
                len: n,
                chan_dim,
                inner,
                scales: scales.clone(),
                zero_points: zero_points.clone(),
            }))
        }
        Op::Pool {
            kind,
            kernel_size,
            stride,
            padding,
        } => {
            // NCHW 2-D pooling (max/avg/sum): a windowed reduce. Host-computed (pure
            // gather+reduce, memory-bound — coverage for the vision path), bit-exact
            // with the CPU thunk. The perf-critical conv GEMM stays on the NPU.
            let d = in_dims();
            if d.len() != 4 || kernel_size.len() != 2 {
                return Err(format!(
                    "NPU pool: NCHW 2-D only (in rank {}, k {})",
                    d.len(),
                    kernel_size.len()
                ));
            }
            let (kh, kw) = (kernel_size[0], kernel_size[1]);
            let (sh, sw) = (stride[0], stride[1]);
            let (ph, pw) = (padding[0], padding[1]);
            Ok(Box::new(XdnaPool2dExec {
                x_name,
                n: d[0],
                c: d[1],
                h: d[2],
                w: d[3],
                h_out: (d[2] + 2 * ph - kh) / sh + 1,
                w_out: (d[3] + 2 * pw - kw) / sw + 1,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                kind: *kind,
            }))
        }
        Op::Im2Col {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            // NCHW im2col unfold → [n·h_out·w_out, c·kh·kw]. Host gather (zero-pad),
            // bit-exact with the CPU thunk. Paired with the NPU int8 matmul this is a
            // full conv with the GEMM on the accelerator.
            let d = in_dims();
            if d.len() != 4 || kernel_size.len() != 2 {
                return Err(format!(
                    "NPU im2col: NCHW 2-D only (in rank {}, k {})",
                    d.len(),
                    kernel_size.len()
                ));
            }
            let (kh, kw) = (kernel_size[0], kernel_size[1]);
            let (sh, sw) = (stride[0], stride[1]);
            let (ph, pw) = (padding[0], padding[1]);
            let (dh, dw) = (dilation[0], dilation[1]);
            Ok(Box::new(XdnaIm2ColExec {
                x_name,
                n: d[0],
                c_in: d[1],
                h: d[2],
                w: d[3],
                h_out: (d[2] + 2 * ph - dh * (kh - 1) - 1) / sh + 1,
                w_out: (d[3] + 2 * pw - dw * (kw - 1) - 1) / sw + 1,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
            }))
        }
        _ => Err("op not on the NPU (unsupported for this backend)".into()),
    }
}

/// `(chan_dim, inner)` for per-channel quant, mirroring rlx-cpu's `quant_layout`:
/// `chan_dim` = size of the `axis` dim (1 for per-tensor `None`), `inner` = product
/// of the dims after `axis`. The channel of element `i` is `(i / inner) % chan_dim`.
fn quant_layout(dims: &[usize], axis: Option<usize>) -> (usize, usize) {
    match axis {
        None => (1, dims.iter().product::<usize>().max(1)),
        Some(d) => (dims[d], dims[d + 1..].iter().product::<usize>().max(1)),
    }
}

/// A `Compare` op: the NPU kernel yields an f32 mask (1.0/0.0), which this exec
/// re-encodes to the runtime's **bool** tensor representation — one byte per
/// element (`0x01`/`0x00`), then the whole byte buffer reinterpreted as f32 (so an
/// `n`-element bool output is `⌈n/4⌉` f32 cells, matching the CPU backend).
struct XdnaCompareExec {
    io: NpuIo,
    a_name: String,
    b_name: String,
    n: usize,
}
unsafe impl Send for XdnaCompareExec {}

impl ExecutableGraph for XdnaCompareExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: compare input '{name}' not provided"))
        };
        let (a, b) = (get(&self.a_name), get(&self.b_name));
        let mask = self
            .io
            .run2(as_i32(a), as_i32(b))
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        // Re-encode the mask as packed bool bytes (0/1), then reinterpret as f32.
        let mut bytes = vec![0u8; self.n.div_ceil(4) * 4];
        for i in 0..self.n {
            if f32::from_bits(mask[i] as u32) != 0.0 {
                bytes[i] = 1;
            }
        }
        let packed = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vec![packed]
    }
}

/// `Quantize` (f32 → i8): `q = clamp(round(x / scale[c]) + zp[c], -128, 127)`, packed
/// 4 codes per f32 cell — the runtime's i8 tensor layout (same as the bool path).
/// Host-computed: the per-channel scales/zero-points are baked and the pass is
/// memory-bound, so there is no NPU compute win; this exists for COVERAGE (int8
/// graphs run on `Device::Xdna`). Bit-exact with the CPU `Quantize` thunk.
struct XdnaQuantizeExec {
    x_name: String,
    len: usize,
    chan_dim: usize,
    inner: usize,
    scales: Vec<f32>,
    zero_points: Vec<i32>,
}
unsafe impl Send for XdnaQuantizeExec {}
impl ExecutableGraph for XdnaQuantizeExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.x_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| {
                panic!("XdnaBackend: quantize input '{}' not provided", self.x_name)
            });
        let mut bytes = vec![0u8; self.len.div_ceil(4) * 4];
        for i in 0..self.len {
            let c = if self.chan_dim == 1 {
                0
            } else {
                (i / self.inner) % self.chan_dim
            };
            let v = (x[i] * (1.0 / self.scales[c])).round() as i32 + self.zero_points[c];
            bytes[i] = (v.clamp(-128, 127) as i8) as u8;
        }
        vec![
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ]
    }
}

/// `Dequantize` (i8 → f32): the inverse of [`XdnaQuantizeExec`]. Unpacks the i8 codes
/// from the packed f32-cell bytes, then `x = (q − zp[c]) · scale[c]` per channel.
/// Host-computed coverage twin. Bit-exact with the CPU `Dequantize` thunk.
struct XdnaDequantizeExec {
    q_name: String,
    len: usize,
    chan_dim: usize,
    inner: usize,
    scales: Vec<f32>,
    zero_points: Vec<i32>,
}
unsafe impl Send for XdnaDequantizeExec {}
impl ExecutableGraph for XdnaDequantizeExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let q = inputs
            .iter()
            .find(|(nm, _)| *nm == self.q_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| {
                panic!(
                    "XdnaBackend: dequantize input '{}' not provided",
                    self.q_name
                )
            });
        // Reinterpret the packed f32 cells as bytes; the first `len` are the i8 codes.
        let mut codes = Vec::with_capacity(q.len() * 4);
        for &cell in q {
            codes.extend_from_slice(&cell.to_le_bytes());
        }
        let mut out = vec![0f32; self.len];
        for i in 0..self.len {
            let c = if self.chan_dim == 1 {
                0
            } else {
                (i / self.inner) % self.chan_dim
            };
            let qi = (codes[i] as i8) as i32;
            out[i] = (qi - self.zero_points[c]) as f32 * self.scales[c];
        }
        vec![out]
    }
}

/// NCHW 2-D pooling (`Op::Pool`): windowed reduce over `kh×kw` per `(n,c)` plane →
/// `[n, c, h_out, w_out]`. Max = reduce-max (out-of-bounds skipped), Mean = window
/// sum ÷ `kh·kw` (count-include-pad), any other kind = window sum. Host-computed
/// (memory-bound gather+reduce), bit-exact with the CPU `Pool2D` thunk.
struct XdnaPool2dExec {
    x_name: String,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    kind: rlx_ir::op::ReduceOp,
}
unsafe impl Send for XdnaPool2dExec {}
impl ExecutableGraph for XdnaPool2dExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        use rlx_ir::op::ReduceOp;
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.x_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: pool input '{}' not provided", self.x_name));
        let (is_max, is_mean) = (
            matches!(self.kind, ReduceOp::Max),
            matches!(self.kind, ReduceOp::Mean),
        );
        let area = (self.kh * self.kw) as f32;
        let mut out = vec![0f32; self.n * self.c * self.h_out * self.w_out];
        for nc in 0..self.n * self.c {
            let in_chan = nc * self.h * self.w;
            let out_chan = nc * self.h_out * self.w_out;
            for ho in 0..self.h_out {
                for wo in 0..self.w_out {
                    let mut a = if is_max { f32::NEG_INFINITY } else { 0.0 };
                    for ki in 0..self.kh {
                        for kj in 0..self.kw {
                            let hi = ho * self.sh + ki;
                            let wi = wo * self.sw + kj;
                            if hi < self.ph || wi < self.pw {
                                continue;
                            }
                            let (hi, wi) = (hi - self.ph, wi - self.pw);
                            if hi >= self.h || wi >= self.w {
                                continue;
                            }
                            let v = x[in_chan + hi * self.w + wi];
                            if is_max {
                                a = a.max(v);
                            } else {
                                a += v;
                            }
                        }
                    }
                    out[out_chan + ho * self.w_out + wo] = if is_mean { a / area } else { a };
                }
            }
        }
        vec![out]
    }
}

/// NCHW im2col unfold (`Op::Im2Col`) → `[n·h_out·w_out, c·kh·kw]`. Row-major: row =
/// output pixel `(n,ho,wo)`, column iterates `(c,ki,kj)`; out-of-bounds (padding)
/// reads 0. Host gather, bit-exact with the CPU `Im2Col` thunk (`im2col_rows_layout`).
struct XdnaIm2ColExec {
    x_name: String,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
}
unsafe impl Send for XdnaIm2ColExec {}
impl ExecutableGraph for XdnaIm2ColExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.x_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: im2col input '{}' not provided", self.x_name));
        let k = self.c_in * self.kh * self.kw;
        let mut col = vec![0f32; self.n * self.h_out * self.w_out * k];
        for ni in 0..self.n {
            let x_base = ni * self.c_in * self.h * self.w;
            for ho in 0..self.h_out {
                for wo in 0..self.w_out {
                    let row = (ni * self.h_out * self.w_out + ho * self.w_out + wo) * k;
                    let mut elem = 0usize;
                    for ci in 0..self.c_in {
                        for ki in 0..self.kh {
                            for kj in 0..self.kw {
                                let hi = (ho * self.sh + ki * self.dh) as isize - self.ph as isize;
                                let wi = (wo * self.sw + kj * self.dw) as isize - self.pw as isize;
                                col[row + elem] = if hi < 0
                                    || hi >= self.h as isize
                                    || wi < 0
                                    || wi >= self.w as isize
                                {
                                    0.0
                                } else {
                                    x[x_base + (ci * self.h + hi as usize) * self.w + wi as usize]
                                };
                                elem += 1;
                            }
                        }
                    }
                }
            }
        }
        vec![col]
    }
}

/// A `Rope` op: inputs `[x, cos, sin]`; the exec packs cos‖sin into arg1 (each
/// `half` elements) → `NpuRun3` (arg0=x, arg1=cos‖sin, arg2=out).
struct XdnaRopeExec {
    io: NpuRun3,
    x_name: String,
    cos_name: String,
    sin_name: String,
    half: usize,
}
unsafe impl Send for XdnaRopeExec {}

impl ExecutableGraph for XdnaRopeExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: rope input '{name}' not provided"))
        };
        let (x, cos, sin) = (get(&self.x_name), get(&self.cos_name), get(&self.sin_name));
        assert_eq!(cos.len(), self.half, "rope cos wrong size");
        let mut cs = Vec::with_capacity(2 * self.half);
        cs.extend_from_slice(cos);
        cs.extend_from_slice(sin);
        let out = self
            .io
            .run(x, &cs)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// A 3-input elementwise op (`Where`/`Fma`): `arg0`=`a_name`, `arg1`=`b_name`‖
/// `c_name` (packed at run), `arg2`=out, via the generic 3-buffer `NpuRun3`.
struct XdnaTernExec {
    io: NpuRun3,
    a_name: String,
    b_name: String,
    c_name: String,
    n: usize,
}
unsafe impl Send for XdnaTernExec {}

impl ExecutableGraph for XdnaTernExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: input '{name}' not provided"))
        };
        let (a, b, c) = (get(&self.a_name), get(&self.b_name), get(&self.c_name));
        assert_eq!(a.len(), self.n);
        let mut bc = Vec::with_capacity(2 * self.n);
        bc.extend_from_slice(b);
        bc.extend_from_slice(c);
        let out = self
            .io
            .run(a, &bc)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// A single-input data-movement op: `arg0`=in, `arg2`=out (`arg1` dummy) → the
/// 3-buffer `NpuRun3`. f32↔i32 reinterpreted at the byte boundary.
struct XdnaDmExec {
    io: NpuRun3,
    in_name: String,
    n_in: usize,
}
unsafe impl Send for XdnaDmExec {}
impl ExecutableGraph for XdnaDmExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.in_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: input '{}' not provided", self.in_name));
        assert_eq!(x.len(), self.n_in);
        let out = self
            .io
            .run(x, &[0.0])
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// A two-input data-movement op (concat / gather): `arg0`=A, `arg1`=B → `NpuRun3`.
struct XdnaDm2Exec {
    io: NpuRun3,
    a_name: String,
    b_name: String,
    na: usize,
    nb: usize,
}
unsafe impl Send for XdnaDm2Exec {}
impl ExecutableGraph for XdnaDm2Exec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: input '{name}' not provided"))
        };
        let (a, b) = (get(&self.a_name), get(&self.b_name));
        assert_eq!(a.len(), self.na);
        assert_eq!(b.len(), self.nb);
        let out = self
            .io
            .run(a, b)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// Reinterpret an f32 slice as i32 (same 4-byte cells) for the byte-level shim.
fn as_i32(x: &[f32]) -> &[i32] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const i32, x.len()) }
}

/// A binary f32 op on the NPU (a ⊙ b): two graph inputs → `NpuIo::run2`, f32↔i32
/// reinterpreted at the byte boundary.
struct XdnaBinExec {
    io: NpuIo,
    a_name: String,
    b_name: String,
    n: usize,
}
unsafe impl Send for XdnaBinExec {}
impl ExecutableGraph for XdnaBinExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: binary input '{name}' not provided"))
        };
        let (a, b) = (get(&self.a_name), get(&self.b_name));
        assert_eq!(a.len(), self.n);
        let out = self
            .io
            .run2(as_i32(a), as_i32(b))
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out.iter().map(|&b| f32::from_bits(b as u32)).collect()]
    }
}

/// A last-axis reduction on the NPU. The kernel broadcasts each row's result
/// across the row; we read column 0 to recover the `[rows]` output.
struct XdnaReduceExec {
    io: NpuIo,
    in_name: String,
    n_in: usize,
    rows: usize,
    cols: usize,
}
unsafe impl Send for XdnaReduceExec {}
impl ExecutableGraph for XdnaReduceExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.in_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: reduce input '{}' not provided", self.in_name));
        assert_eq!(x.len(), self.n_in);
        let raw = self
            .io
            .run(as_i32(x))
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        // Column 0 of each row holds the broadcast reduction.
        let out = (0..self.rows)
            .map(|r| f32::from_bits(raw[r * self.cols] as u32))
            .collect();
        vec![out]
    }
}

/// Fused single-head attention on the NPU: Q/K/V are three graph inputs; at run
/// K and V are packed (K‖V) and fed with Q to the 3-buffer kernel.
struct XdnaAttnExec {
    io: NpuRun3,
    q_name: String,
    k_name: String,
    v_name: String,
    n: usize, // seq*d
}

// The NPU context is driven single-threaded through `&mut self`.
unsafe impl Send for XdnaAttnExec {}

impl ExecutableGraph for XdnaAttnExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let get = |name: &str| -> &[f32] {
            inputs
                .iter()
                .find(|(nm, _)| *nm == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("XdnaBackend: attention input '{name}' not provided"))
        };
        let q = get(&self.q_name);
        let (k, v) = (get(&self.k_name), get(&self.v_name));
        assert_eq!(q.len(), self.n, "XdnaBackend: attention Q wrong size");
        let mut kv = k.to_vec();
        kv.extend_from_slice(v);
        let out = self
            .io
            .run(q, &kv)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// An affine norm (RmsNorm / LayerNorm) on the NPU: `gamma`/`beta` params are
/// collected via `set_param`, packed `gamma‖beta`, and fed with `x` to the
/// 3-buffer kernel. Missing gamma → identity (1), missing beta → 0.
struct XdnaNormExec {
    io: NpuRun3,
    x_name: String,
    gamma_name: String,
    beta_name: Option<String>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    cols: usize,
    n: usize,
}

// The NPU context is driven single-threaded through `&mut self`.
unsafe impl Send for XdnaNormExec {}

impl ExecutableGraph for XdnaNormExec {
    fn set_param(&mut self, name: &str, data: &[f32]) {
        if name == self.gamma_name {
            self.gamma = data.to_vec();
        } else if self.beta_name.as_deref() == Some(name) {
            self.beta = data.to_vec();
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.x_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: input '{}' not provided", self.x_name));
        assert_eq!(
            x.len(),
            self.n,
            "XdnaBackend: input '{}' wrong size",
            self.x_name
        );
        // Pack gamma‖beta, defaulting to identity/zero when a param is absent.
        let mut gb = if self.gamma.len() == self.cols {
            self.gamma.clone()
        } else {
            vec![1.0; self.cols]
        };
        if self.beta.len() == self.cols {
            gb.extend_from_slice(&self.beta);
        } else {
            gb.extend(std::iter::repeat(0.0).take(self.cols));
        }
        let out = self
            .io
            .run(x, &gb)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// A single elementwise/softmax op resident on the NPU (f32, persistent context).
struct XdnaOpExec {
    io: NpuIoF32,
    in_name: String,
    n: usize,
}

// The NPU context is driven single-threaded through `&mut self`.
unsafe impl Send for XdnaOpExec {}

impl ExecutableGraph for XdnaOpExec {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.in_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: input '{}' not provided", self.in_name));
        assert_eq!(
            x.len(),
            self.n,
            "XdnaBackend: input '{}' wrong size",
            self.in_name
        );
        let out = self
            .io
            .run(x)
            .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
        vec![out]
    }
}

/// FAST MatMul path: compile-on-demand C++-`aie::mmul`-microkernel overlay
/// (`emit_matmul_microkernel` — K-accumulation × 4 AIE columns,
/// ~638 GOP/s), reachable whenever `AIECC`/`PEANO` are set (no pre-built overlay
/// env needed). Picks a `d=64` tile, `cols = clamp(⌈n/64⌉, 1, 4)`,
/// `kt = clamp(⌈k/64⌉, 1, 8)`, compiles the kernel `.o` + overlay once (cached in
/// tmp by shape), then host-blocks the graph `m×k×n` over it (K-blocks summed,
/// edges zero-padded). Per-row activation / per-col weight int8 quant, `sx·sw`
/// dequant — same accuracy contract as [`XdnaGemmExec`].
fn build_microkernel_exec(graph: &Graph) -> Result<XdnaMicrokernelExec, String> {
    let mms: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul))
        .collect();
    if mms.len() != 1 {
        return Err(format!(
            "only a single matmul supported (graph has {})",
            mms.len()
        ));
    }
    build_matmul_node_exec(graph, mms[0])
}

/// Build the fast microkernel exec for a specific MatMul node. The activation
/// (input 0) may be a graph Input OR an intermediate (`n{id}`) from an earlier op
/// in a chain; the weight (input 1) must be a `Param` (quantized once, resident).
fn build_matmul_node_exec(
    graph: &Graph,
    mm: &rlx_ir::graph::Node,
) -> Result<XdnaMicrokernelExec, String> {
    let aiecc =
        std::env::var("AIECC").map_err(|_| "microkernel matmul needs AIECC in env".to_string())?;
    let peano =
        std::env::var("PEANO").map_err(|_| "microkernel matmul needs PEANO in env".to_string())?;
    // The mlir_aie include tree that holds BOTH `aie_kernels/aie2/mm.cc` (the vendor
    // microkernel source) and `aie_api/` (its headers). `RLX_XDNA_AIE_INCLUDE`
    // overrides it — required when aiecc doesn't sit at `<mlir_aie>/bin/aiecc` (e.g. a
    // pip `mlir_aie` package puts the real include under site-packages, not next to
    // the `bin/aiecc` shim). Falls back to `<aiecc>/../../include`.
    let include = std::env::var("RLX_XDNA_AIE_INCLUDE").ok().map_or_else(
        || {
            std::path::Path::new(&aiecc)
                .parent()
                .and_then(|p| p.parent())
                .map(|p| format!("{}/include", p.display()))
                .ok_or_else(|| "cannot derive mlir_aie include dir from AIECC path".to_string())
        },
        Ok,
    )?;

    if mm.inputs.len() != 2 {
        return Err("matmul must have exactly 2 inputs".into());
    }
    let act = graph.node(mm.inputs[0]);
    let wt = graph.node(mm.inputs[1]);
    let act_name = match &act.op {
        Op::Param { .. } | Op::Constant { .. } => {
            return Err("matmul input 0 must be the activation, not a param/const".into());
        }
        _ => node_ref(graph, mm.inputs[0]),
    };
    // A Param weight is pre-tiled once via set_param; anything else (a backward
    // `xᵀ`/`dy` intermediate, an Input, or a threaded Constant) is a DYNAMIC weight,
    // fetched + quantized + tiled per run.
    let (w_name, w_dynamic) = match &wt.op {
        Op::Param { name } => (name.clone(), false),
        _ => (node_ref(graph, mm.inputs[1]), true),
    };
    let dim = |s: &rlx_ir::Shape, i: usize| -> Result<usize, String> {
        s.dims()
            .get(i)
            .map(|d| d.unwrap_static())
            .ok_or_else(|| "matmul operands must be rank-2 static shapes".into())
    };
    let (m, k) = (dim(&act.shape, 0)?, dim(&act.shape, 1)?);
    let n = dim(&mm.shape, 1)?;

    const D: usize = 64;
    const KTMAX: usize = 8;
    let cols = n.div_ceil(D).clamp(1, 4);
    let kt = k.div_ceil(D).clamp(1, KTMAX);

    let cache = format!(
        "{}/rlx_xdna_mk_{D}_{kt}_{cols}",
        std::env::temp_dir().display()
    );
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir {cache}: {e}"))?;
    let kernel_o = format!("{cache}/mm_{D}.o");
    let clangxx = format!("{peano}/bin/clang++");
    build_mm_kernel(&clangxx, &include, D, &kernel_o).map_err(|e| e.0)?;
    let obj_base = format!("mm_{D}.o");

    let mlir = format!("{cache}/aie.mlir");
    std::fs::write(&mlir, emit_matmul_microkernel(D, kt, cols, &obj_base))
        .map_err(|e| format!("write microkernel mlir: {e}"))?;
    let xclbin = format!("{cache}/k.xclbin");
    let insts_p = format!("{cache}/i.bin");
    compile_overlay_linked(
        &OverlaySpec {
            aiecc: &aiecc,
            peano: &peano,
            mlir: &mlir,
            tmpdir: &format!("{cache}/build"),
            out_xclbin: &xclbin,
            out_insts: &insts_p,
        },
        &[&kernel_o],
    )
    .map_err(|e| e.0)?;

    let insts: Vec<u32> = std::fs::read(&insts_p)
        .map_err(|e| format!("read insts {insts_p}: {e}"))?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Overlay dims: m=D, k=kt·D, n=cols·D (the per-dispatch tile the host blocks over).
    let npu = NpuGemm::open("", &xclbin, &insts, D, kt * D, cols * D).map_err(|e| e.0)?;

    Ok(XdnaMicrokernelExec {
        npu,
        d: D,
        kt,
        cols,
        m,
        k,
        n,
        act_name,
        w_name,
        w_dynamic,
        sw: Vec::new(),
        bt_blocks: Vec::new(),
        nkb: 0,
        nnb: 0,
    })
}

/// Fast-matmul exec over the C++-microkernel overlay (see [`build_microkernel_exec`]).
struct XdnaMicrokernelExec {
    npu: NpuGemm,
    d: usize,
    kt: usize,
    cols: usize,
    m: usize,
    k: usize,
    n: usize,
    act_name: String,
    w_name: String,
    /// When true the weight is not a Param — it arrives as a run input (backward
    /// `xᵀ @ dy`) and is quantized + tiled at the start of every `run` (no cache).
    w_dynamic: bool,
    sw: Vec<f32>, // per-output-channel weight scales, len n
    // The weight is constant across calls, so it's PRE-TILED once in set_param into
    // per-(K-block, N-block) `tile_b_kacc_multicol` buffers (indexed `kb*nnb+nb`) —
    // the hot `run` path then only quantizes+tiles the (changing) activation.
    bt_blocks: Vec<Vec<i8>>,
    nkb: usize,
    nnb: usize,
}
unsafe impl Send for XdnaMicrokernelExec {}

impl XdnaMicrokernelExec {
    /// Pre-tile the quantized weight `w_i8` [k,n] into per-block streams once.
    fn precompute_weight_tiles(&mut self, w_i8: &[i8]) {
        let (d, kt, cols) = (self.d, self.kt, self.cols);
        let (k, n) = (self.k, self.n);
        let (kk, nn) = (kt * d, cols * d);
        self.nkb = k.div_ceil(kk);
        self.nnb = n.div_ceil(nn);
        self.bt_blocks = Vec::with_capacity(self.nkb * self.nnb);
        for kb in 0..self.nkb {
            for nb in 0..self.nnb {
                let b_blk = block_i8(w_i8, k, n, kb * kk, nb * nn, kk, nn); // kk×nn
                self.bt_blocks
                    .push(tile_b_kacc_multicol(&b_blk, d, kt, cols));
            }
        }
    }
}

impl ExecutableGraph for XdnaMicrokernelExec {
    fn set_param(&mut self, name: &str, data: &[f32]) {
        if name == self.w_name {
            let (q, s) = quantize_per_col(data, self.k, self.n);
            self.sw = s;
            self.precompute_weight_tiles(&q);
        }
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if name == self.w_name && dtype == rlx_ir::DType::I8 {
            let q: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            self.sw = vec![1.0; self.n];
            self.precompute_weight_tiles(&q);
        } else {
            let f: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.set_param(name, &f);
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        // Dynamic weight (backward matmul): fetch, quantize per-col, tile — per run.
        if self.w_dynamic {
            let w = inputs
                .iter()
                .find(|(nm, _)| *nm == self.w_name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| {
                    panic!("XdnaBackend: dynamic weight '{}' not provided", self.w_name)
                });
            assert_eq!(
                w.len(),
                self.k * self.n,
                "XdnaBackend: dynamic weight '{}' wrong size",
                self.w_name
            );
            let (q, s) = quantize_per_col(w, self.k, self.n);
            self.sw = s;
            self.precompute_weight_tiles(&q);
        }
        assert!(
            !self.bt_blocks.is_empty(),
            "XdnaBackend microkernel: weight '{}' not set",
            self.w_name
        );
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.act_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: input '{}' not provided", self.act_name));
        assert_eq!(
            x.len(),
            self.m * self.k,
            "XdnaBackend: input '{}' wrong size",
            self.act_name
        );
        let (a_i8, sx) = quantize_per_row(x, self.m, self.k);

        let (d, kt, cols) = (self.d, self.kt, self.cols);
        let (m, k, n) = (self.m, self.k, self.n);
        let (kk, nn) = (kt * d, cols * d);
        let (nkb, nnb, nmb) = (self.nkb, self.nnb, m.div_ceil(d));

        let mut c = vec![0i32; m * n];
        for mb in 0..nmb {
            let mi = mb * d;
            let mut accs = vec![vec![0i32; d * nn]; nnb]; // one C sub-block per N-block
            for kb in 0..nkb {
                // Tile the activation block ONCE per (mb, kb) — reused across N-blocks.
                let a_blk = block_i8(&a_i8, m, k, mi, kb * kk, d, kk); // d×kk
                let at = tile_a_kacc(&a_blk, d, kt);
                for nb in 0..nnb {
                    let bt = &self.bt_blocks[kb * nnb + nb];
                    let ct = self
                        .npu
                        .run(&at, bt)
                        .unwrap_or_else(|e| panic!("microkernel run: {}", e.0));
                    let cb = untile_c_multicol(&ct, d, nn, cols); // d×nn
                    for (a, p) in accs[nb].iter_mut().zip(&cb) {
                        *a += *p;
                    }
                }
            }
            for nb in 0..nnb {
                let ni = nb * nn;
                for i in 0..d.min(m - mi) {
                    for j in 0..nn.min(n - ni) {
                        c[(mi + i) * n + (ni + j)] = accs[nb][i * nn + j];
                    }
                }
            }
        }
        let out: Vec<f32> = (0..m * n)
            .map(|idx| c[idx] as f32 * sx[idx / n] * self.sw[idx % n])
            .collect();
        vec![out]
    }
}

fn build_gemm_exec(graph: &Graph) -> Result<XdnaGemmExec, String> {
    let ov = rlx_xdna::overlay_from_env().ok_or_else(rlx_xdna::diagnostic)?;

    let mms: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul))
        .collect();
    if mms.len() != 1 {
        return Err(format!(
            "only a single matmul is supported (graph has {} MatMul nodes)",
            mms.len()
        ));
    }
    let mm = mms[0];
    if mm.inputs.len() != 2 {
        return Err("matmul must have exactly 2 inputs".into());
    }
    let act = graph.node(mm.inputs[0]);
    let wt = graph.node(mm.inputs[1]);
    let act_name = match &act.op {
        Op::Input { name } => name.clone(),
        _ => return Err("matmul input 0 must be a graph Input (the activation)".into()),
    };
    let w_name = match &wt.op {
        Op::Param { name } => name.clone(),
        _ => return Err("matmul input 1 must be a Param (the weight)".into()),
    };

    let dim = |s: &rlx_ir::Shape, i: usize| -> Result<usize, String> {
        s.dims()
            .get(i)
            .map(|d| d.unwrap_static())
            .ok_or_else(|| "matmul operands must be rank-2 static shapes".into())
    };
    let (m, k) = (dim(&act.shape, 0)?, dim(&act.shape, 1)?);
    let n = dim(&mm.shape, 1)?;

    let insts: Vec<u32> = std::fs::read(&ov.insts)
        .map_err(|e| format!("read insts {}: {e}", ov.insts))?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Persistent NPU context at the OVERLAY shape; the tiler feeds it padded blocks.
    let npu = NpuGemm::open(&ov.shim, &ov.xclbin, &insts, ov.m, ov.k, ov.n).map_err(|e| e.0)?;

    Ok(XdnaGemmExec {
        npu,
        om: ov.m,
        ok: ov.k,
        on: ov.n,
        m,
        k,
        n,
        act_name,
        w_name,
        w_i8: Vec::new(),
        sw: Vec::new(),
        // Resident weight blocks for ANY shape (opt out with RLX_XDNA_NO_RESIDENT).
        resident: std::env::var("RLX_XDNA_NO_RESIDENT").is_err(),
        weight_up: false,
    })
}

/// Symmetric INT8 with **per-row** scales (`s[r] = max|row_r| / 127`): quantize
/// activations per-token. Returns the i8 codes `[rows*cols]` + one scale per row.
/// Per-token/per-channel scaling is far more accurate than one whole-tensor scale.
fn quantize_per_row(v: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![1.0f32; rows];
    for r in 0..rows {
        let row = &v[r * cols..r * cols + cols];
        let amax = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if amax == 0.0 {
            continue;
        }
        let sr = amax / 127.0;
        s[r] = sr;
        let inv = 1.0 / sr;
        for c in 0..cols {
            q[r * cols + c] = (row[c] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

/// Symmetric INT8 with **per-column** scales (`s[c] = max|col_c| / 127`):
/// quantize weights per output channel. Returns i8 `[rows*cols]` + a scale per col.
fn quantize_per_col(v: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut s = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            s[c] = s[c].max(v[r * cols + c].abs());
        }
    }
    for sc in s.iter_mut() {
        *sc = if *sc == 0.0 { 1.0 } else { *sc / 127.0 };
    }
    let mut q = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            q[r * cols + c] = (v[r * cols + c] / s[c]).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, s)
}

/// One matmul on the NPU, tiled onto the fixed-shape overlay. `om/ok/on` are the
/// overlay's compiled dims; `m/k/n` the graph's (any size).
struct XdnaGemmExec {
    npu: NpuGemm,
    om: usize,
    ok: usize,
    on: usize,
    m: usize,
    k: usize,
    n: usize,
    act_name: String,
    w_name: String,
    w_i8: Vec<i8>,   // full weight [k, n], symmetric-int8 quantized
    sw: Vec<f32>,    // per-column (per-output-channel) weight scales, len n
    resident: bool,  // whole matmul fits one tile → keep the weight on-device
    weight_up: bool, // resident weight already uploaded
}

// The NPU context is driven single-threaded through `&mut self`.
unsafe impl Send for XdnaGemmExec {}

/// Copy the `[br,bc]` block of `src` (row-major `[rows,cols]`) at offset
/// `(r0,c0)`, zero-padding where the block runs past the edges.
fn block_i8(
    src: &[i8],
    rows: usize,
    cols: usize,
    r0: usize,
    c0: usize,
    br: usize,
    bc: usize,
) -> Vec<i8> {
    let mut out = vec![0i8; br * bc];
    for i in 0..br {
        let r = r0 + i;
        if r >= rows {
            break;
        }
        let cc = (cols.saturating_sub(c0)).min(bc);
        if cc == 0 {
            continue;
        }
        out[i * bc..i * bc + cc].copy_from_slice(&src[r * cols + c0..r * cols + c0 + cc]);
    }
    out
}

impl ExecutableGraph for XdnaGemmExec {
    fn set_param(&mut self, name: &str, data: &[f32]) {
        if name == self.w_name {
            // Quantize the weight once, per output channel (per-column of [k,n]).
            let (q, s) = quantize_per_col(data, self.k, self.n);
            self.w_i8 = q;
            self.sw = s;
        }
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if name == self.w_name && dtype == rlx_ir::DType::I8 {
            // Already-quantized weight: caller owns the scale (identity here).
            self.w_i8 = data.iter().map(|&b| b as i8).collect();
            self.sw = vec![1.0; self.n];
        } else {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.set_param(name, &floats);
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        assert_eq!(
            self.w_i8.len(),
            self.k * self.n,
            "XdnaBackend: weight '{}' not set (expected {}x{})",
            self.w_name,
            self.k,
            self.n
        );
        let x = inputs
            .iter()
            .find(|(nm, _)| *nm == self.act_name)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("XdnaBackend: input '{}' not provided", self.act_name));
        assert_eq!(
            x.len(),
            self.m * self.k,
            "XdnaBackend: input '{}' wrong size",
            self.act_name
        );
        // Quantize the activation per token (per-row); dequant `C[m,n]` by sx[m]·sw[n].
        let (a_i8, sx) = quantize_per_row(x, self.m, self.k);

        // Blocked GEMM: C[m,n] = A[m,k] @ W[k,n], each block an OMxOKxON overlay
        // call, K-blocks summed. Edge blocks are zero-padded, so any shape works.
        let (m, k, n, om, ok, on) = (self.m, self.k, self.n, self.om, self.ok, self.on);

        let nkb = k.div_ceil(ok); // K-blocks
        let nnb = n.div_ceil(on); // N-blocks

        // Resident path: every weight block (K-block × N-block) is uploaded once
        // and stays on-device; each `run_block` ships only the activation tile.
        // The win: repeated inference (same W, streaming activations) skips all
        // the weight DMA. Works for any shape (single tile = one block).
        if self.resident {
            if !self.weight_up {
                for kb in 0..nkb {
                    for nb in 0..nnb {
                        let wblk = block_i8(&self.w_i8, k, n, kb * ok, nb * on, ok, on);
                        self.npu
                            .set_weight_block(kb * nnb + nb, &wblk)
                            .unwrap_or_else(|e| panic!("XdnaBackend set_weight_block: {}", e.0));
                    }
                }
                self.weight_up = true;
            }
            let mut c = vec![0i32; m * n];
            let mut mi = 0;
            while mi < m {
                for nb in 0..nnb {
                    let ni = nb * on;
                    let mut acc = vec![0i32; om * on];
                    for kb in 0..nkb {
                        let a_blk = block_i8(&a_i8, m, k, mi, kb * ok, om, ok);
                        let part = self
                            .npu
                            .run_block(kb * nnb + nb, &a_blk)
                            .unwrap_or_else(|e| panic!("XdnaBackend run_block: {}", e.0));
                        for (a, p) in acc.iter_mut().zip(&part) {
                            *a += *p;
                        }
                    }
                    for i in 0..om.min(m - mi) {
                        for j in 0..on.min(n - ni) {
                            c[(mi + i) * n + (ni + j)] = acc[i * on + j];
                        }
                    }
                }
                mi += om;
            }
            let out: Vec<f32> = (0..m * n)
                .map(|idx| c[idx] as f32 * sx[idx / n] * self.sw[idx % n])
                .collect();
            return vec![out];
        }

        // Re-upload path (RLX_XDNA_NO_RESIDENT): ship A and B every block.
        let mut c = vec![0i32; m * n];
        let mut mi = 0;
        while mi < m {
            let mut ni = 0;
            while ni < n {
                let mut acc = vec![0i32; om * on];
                let mut ki = 0;
                while ki < k {
                    let a_blk = block_i8(&a_i8, m, k, mi, ki, om, ok);
                    let b_blk = block_i8(&self.w_i8, k, n, ki, ni, ok, on);
                    let part = self
                        .npu
                        .run(&a_blk, &b_blk)
                        .unwrap_or_else(|e| panic!("XdnaBackend NPU run: {}", e.0));
                    for (a, p) in acc.iter_mut().zip(&part) {
                        *a += *p;
                    }
                    ki += ok;
                }
                // Scatter the valid part of the accumulated tile into C.
                for i in 0..om.min(m - mi) {
                    for j in 0..on.min(n - ni) {
                        c[(mi + i) * n + (ni + j)] = acc[i * on + j];
                    }
                }
                ni += on;
            }
            mi += om;
        }
        // Dequantize the i32 accumulator back to f32 per element: C[m,n]·sx[m]·sw[n].
        let out: Vec<f32> = (0..m * n)
            .map(|idx| c[idx] as f32 * sx[idx / n] * self.sw[idx % n])
            .collect();
        vec![out]
    }
}
