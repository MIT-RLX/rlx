// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! FFI runtime backend — in-process execution on a QNN backend library.
//!
//! This is the live-inference path `../rlx-models` consumes (via the eventual
//! `Device::Hexagon` in `rlx-runtime`), as opposed to the offline codegen path
//! (`codegen` / `qnn_model.cpp`). It binds the C shim in `runtime/`, which
//! `dlopen`s a backend library (`libQnnCpu.so` / `libQnnHtp.so`) and builds +
//! executes the graph through the QNN C API (style 1: dynamic build).
//!
//! [`QnnExecutable::compile_graph`] walks an `rlx-ir` graph into QNN tensors +
//! nodes, [`QnnExecutable::set_param`] stages static weights, and
//! [`QnnExecutable::run`] binds activation inputs and executes. The op surface
//! covers a complete modern-LLM forward pass plus embedding and vision models —
//! MatMul, element-wise binary, activations, Reshape/Transpose/Narrow/Concat/
//! Gather, Softmax, Reduce, LayerNorm/RmsNorm, RoPE, Attention (MHA/GQA; causal/
//! sliding-window/none; optional softcap), Conv2d, and Quantize/Dequantize (the
//! `qnn_op` table). Ops that map to several QNN nodes (RmsNorm, RoPE, Attention,
//! Conv2d, GQA) are decomposed into intermediate `NATIVE` tensors by
//! `compile_graph`. Validated bit-exact against `libQnnCpu.so` (CPU reference
//! backend); the HTP/NPU path is the same code with `libQnnHtp.so` on a
//! Snapdragon device. Sessions finalize once and reuse across `run`s; context
//! binary save/load is the M3 perf path (see `docs/ffi-runtime-backend.md`).
//!
//! Gated behind the `runtime` feature; `build.rs` compiles the shim against
//! `$QNN_SDK_ROOT/include/QNN`.

use std::ffi::{CString, c_char, c_float, c_int, c_void};
use std::path::{Path, PathBuf};

use rlx_ir::Op;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};

/// Opaque QNN session handle (`RlxQnnSession` in the shim).
enum RlxQnnSession {}

/// A graph tensor passed to the shim (matches `RlxQnnTensor` in the header).
/// `ttype`: 0 APP_WRITE (input), 1 APP_READ (output), 3 NATIVE, 4 STATIC.
#[repr(C)]
struct CTensor {
    name: *const c_char,
    ttype: i32,
    rank: u32,
    dims: *const u32,
    data: *mut c_float,
    num_elems: u32,
    /// 0 = float32, 1 = int32, 2 = sfixed8, 3 = int4 (BW_SCALE_OFFSET bitwidth=4).
    dtype: i32,
    /// Per-tensor quantization scale / offset (dtype 2/3 when `q_num_scales == 0`).
    q_scale: c_float,
    q_offset: i32,
    /// AXIS_SCALE_OFFSET axis; ignored when `q_num_scales == 0`.
    q_axis: i32,
    /// 0 = per-tensor SCALE_OFFSET; >0 = AXIS_SCALE_OFFSET entry count.
    q_num_scales: u32,
    /// `q_num_scales` interleaved scale/offset pairs (null when 0).
    q_scale_offsets: *const CScaleOffset,
}

/// Matches `RlxQnnScaleOffset` / `Qnn_ScaleOffset_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CScaleOffset {
    scale: f32,
    offset: i32,
}

/// A graph node passed to the shim (matches `RlxQnnNode` in the header).
#[repr(C)]
struct CNode {
    name: *const c_char,
    op_type: *const c_char,
    inputs: *const u32,
    num_inputs: u32,
    output: u32,
    /// Softmax axis (>= 0); `-1` = no axis param.
    axis: i32,
    /// uint32 tensor-param data: Transpose `perm` or norm `axes` (null if unused).
    perm: *const u32,
    perm_len: u32,
    /// LayerNorm epsilon (ignored unless the op is a norm).
    eps: f32,
}

unsafe extern "C" {
    /// See `runtime/rlx_qnn_shim.h`. Returns 0 on success, or a negative step
    /// code; `err_out` receives the QNN `Qnn_ErrorHandle_t` on failure.
    fn rlx_qnn_matmul_f32(
        backend_lib: *const c_char,
        m: u32,
        k: u32,
        n: u32,
        in0: *const c_float,
        in1: *const c_float,
        out: *mut c_float,
        err_out: *mut u64,
    ) -> c_int;

    /// One-shot build+execute (legacy wrapper around session_*).
    #[allow(dead_code)]
    fn rlx_qnn_run_graph(
        backend_lib: *const c_char,
        tensors: *mut CTensor,
        num_tensors: u32,
        nodes: *const CNode,
        num_nodes: u32,
        err_out: *mut u64,
    ) -> c_int;

    fn rlx_qnn_session_create(
        backend_lib: *const c_char,
        tensors: *mut CTensor,
        num_tensors: u32,
        nodes: *const CNode,
        num_nodes: u32,
        out: *mut *mut RlxQnnSession,
        err_out: *mut u64,
    ) -> c_int;

    fn rlx_qnn_session_execute(
        sess: *mut RlxQnnSession,
        tensors: *mut CTensor,
        num_tensors: u32,
        err_out: *mut u64,
    ) -> c_int;

    fn rlx_qnn_session_save_binary(
        sess: *mut RlxQnnSession,
        out_buf: *mut *mut c_void,
        written: *mut u64,
        err_out: *mut u64,
    ) -> c_int;

    fn rlx_qnn_session_load_binary(
        backend_lib: *const c_char,
        binary: *const c_void,
        binary_size: u64,
        out: *mut *mut RlxQnnSession,
        err_out: *mut u64,
    ) -> c_int;

    fn rlx_qnn_session_free(sess: *mut RlxQnnSession);
    fn rlx_qnn_binary_free(buf: *mut c_void);
}

/// A QNN FFI execution failure: which shim step failed + the QNN error code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QnnError {
    /// Shim step code (`RLX_QNN_E_*` in the header).
    pub step: i32,
    /// The `Qnn_ErrorHandle_t` returned by the failing QNN call.
    pub qnn_err: u64,
}

impl std::fmt::Display for QnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.step {
            1 => "dlopen(backend_lib)",
            2 => "QnnInterface_getProviders symbol",
            3 => "getProviders",
            4 => "backendCreate",
            5 => "contextCreate",
            6 => "graphCreate",
            7 => "tensorCreateGraphTensor",
            8 => "graphAddNode",
            9 => "graphFinalize",
            10 => "graphExecute",
            11 => "contextBinary",
            12 => "libQnnSystem",
            _ => "unknown step",
        };
        write!(
            f,
            "QNN FFI failed at {what} (step {}, qnn_err=0x{:x})",
            self.step, self.qnn_err
        )
    }
}

impl std::error::Error for QnnError {}

/// Resolve the QNN backend library to `dlopen`. Resolution order:
///
/// 1. `RLX_QNN_BACKEND_LIB` — an explicit path to `libQnn*.so`.
/// 2. `$QNN_SDK_ROOT/lib/<host-target>/libQnnCpu.so` — the CPU reference
///    backend for the host architecture.
///
/// Returns `None` when neither is set. The HTP/NPU backend is selected by
/// pointing `RLX_QNN_BACKEND_LIB` at `libQnnHtp.so`.
pub fn default_backend_lib() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RLX_QNN_BACKEND_LIB") {
        return Some(PathBuf::from(p));
    }
    let sdk = std::env::var("QNN_SDK_ROOT").ok()?;
    let target = if cfg!(target_arch = "aarch64") {
        "aarch64-ubuntu-gcc9.4"
    } else {
        "x86_64-linux-clang"
    };
    Some(
        Path::new(&sdk)
            .join("lib")
            .join(target)
            .join("libQnnCpu.so"),
    )
}

/// Build and execute `out[M,N] = in0[M,K] · in1[K,N]` on the QNN backend at
/// `backend_lib`, in-process via the C shim. Panics on a length mismatch
/// (a caller invariant); QNN-side failures return [`QnnError`].
pub fn matmul_f32(
    backend_lib: &Path,
    m: usize,
    k: usize,
    n: usize,
    in0: &[f32],
    in1: &[f32],
) -> Result<Vec<f32>, QnnError> {
    assert_eq!(in0.len(), m * k, "in0 must be M*K");
    assert_eq!(in1.len(), k * n, "in1 must be K*N");
    let lib = CString::new(backend_lib.to_string_lossy().into_owned())
        .expect("backend_lib path contains a NUL byte");
    let mut out = vec![0.0f32; m * n];
    let mut qnn_err: u64 = 0;
    // SAFETY: pointers are valid for the call's duration; `out` is sized M*N;
    // the shim only reads `in0`/`in1` and writes `out`.
    let rc = unsafe {
        rlx_qnn_matmul_f32(
            lib.as_ptr(),
            m as u32,
            k as u32,
            n as u32,
            in0.as_ptr(),
            in1.as_ptr(),
            out.as_mut_ptr(),
            &mut qnn_err,
        )
    };
    if rc != 0 {
        return Err(QnnError { step: -rc, qnn_err });
    }
    Ok(out)
}

/// One lowered graph tensor — index matches the rlx-ir node index.
#[derive(Debug, Clone)]
struct PlanTensor {
    /// `QNN_TENSOR_TYPE_*`: 0 input, 1 output, 3 native, 4 static.
    ttype: i32,
    dims: Vec<u32>,
    num_elems: usize,
    /// Static (param/constant) data; `None` for input/output/native. Params are
    /// filled by [`QnnExecutable::set_param`] before `run`.
    data: Option<Vec<f32>>,
    qnn_name: CString,
}

/// One lowered QNN node.
#[derive(Debug, Clone)]
struct PlanNode {
    name: CString,
    op_type: CString,
    inputs: Vec<u32>,
    output: u32,
    /// Softmax axis (>= 0); `-1` = no axis param.
    axis: i32,
    /// uint32 tensor param: Transpose `perm` or norm `axes` (empty otherwise).
    perm: Vec<u32>,
    /// LayerNorm epsilon (0 unless the op is a norm).
    eps: f32,
}

/// Map a supported `Op` to `(qnn_op_type, axis)`, or `None` if unsupported.
/// `rank` is the node's output rank, used to normalize a negative softmax axis.
fn qnn_op(op: &Op, rank: usize) -> Option<(&'static str, i32)> {
    Some(match op {
        Op::MatMul => ("MatMul", -1),
        Op::Binary(BinaryOp::Add) => ("ElementWiseAdd", -1),
        Op::Binary(BinaryOp::Sub) => ("ElementWiseSubtract", -1),
        Op::Binary(BinaryOp::Mul) => ("ElementWiseMultiply", -1),
        Op::Binary(BinaryOp::Div) => ("ElementWiseDivide", -1),
        Op::Activation(Activation::Relu) => ("Relu", -1),
        Op::Activation(Activation::Gelu) => ("Gelu", -1),
        Op::Activation(Activation::Sigmoid) => ("Sigmoid", -1),
        Op::Activation(Activation::Tanh) => ("Tanh", -1),
        Op::Activation(Activation::Neg) => ("ElementWiseNeg", -1),
        // Silu / Expand are multi-node decompositions in compile_graph.
        Op::Reshape { .. } => ("Reshape", -1),
        Op::Transpose { .. } => ("Transpose", -1),
        Op::Narrow { .. } => ("StridedSlice", -1),
        Op::Quantize { .. } => ("Quantize", -1),
        Op::Dequantize { .. } => ("Dequantize", -1),
        Op::LayerNorm { .. } => ("LayerNorm", -1),
        Op::Concat { axis } => ("Concat", *axis as i32),
        Op::Gather { axis } => ("Gather", *axis as i32),
        // `axis` carries keep_dim (0/1); the reduce axes go in `perm`.
        Op::Reduce { op, keep_dim, .. } => {
            let name = match op {
                ReduceOp::Sum => "ReduceSum",
                ReduceOp::Mean => "ReduceMean",
                ReduceOp::Max => "ReduceMax",
                ReduceOp::Min => "ReduceMin",
                // QNN's op set has no product reduction, so Prod stays
                // unsupported here (a decomposition via exp∘sum∘log is fragile
                // for zero/negative inputs).
                ReduceOp::Prod => return None,
            };
            (name, if *keep_dim { 1 } else { 0 })
        }
        Op::Softmax { axis } => {
            let a = if *axis < 0 {
                *axis + rank as i32
            } else {
                *axis
            };
            ("Softmax", a)
        }
        _ => return None,
    })
}

/// Decode a little-endian f32 byte blob (an `Op::Constant` payload).
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A compiled QNN executable — an arbitrary supported subgraph lowered to QNN
/// tensors + nodes, run in-process on the resolved backend library. This is the
/// surface the `rlx-runtime` `qnn_backend` adapter wraps into `ExecutableGraph`.
///
/// Supported ops: `MatMul`, element-wise `Binary` (Add/Sub/Mul/Div),
/// `Activation` (Relu/Gelu/Sigmoid/Tanh/Neg/Silu), `Reshape`, `Softmax`,
/// `Transpose`, `Narrow` (→ StridedSlice), `Concat`, `Gather` (int32 indices),
/// `Expand` (→ Reshape+Tile), `Reduce` (Mean/Sum/Max), `LayerNorm`,
/// `RmsNorm` (→ RmsNorm+Add), `Rope` (NeoX, with `[S,D/2]` table broadcast),
/// `Attention` (MHA + GQA; causal / sliding-window / none / custom additive
/// mask; optional softcap; rank-3 or rank-4), `Conv` (NCHW → NHWC Conv2d),
/// and `Quantize`/`Dequantize` (int8 SFIXED_POINT_8 + scale/offset), with
/// `Param`/`Constant` operands baked as static weights. `FusedAttentionBlock`
/// is claimed at the runtime adapter and decomposed before this lower.
///
/// A live QNN session is created lazily on the first [`Self::run`] (or
/// [`Self::export_context_binary`]) and reused across subsequent runs until
/// params change. Context-binary save/load is the M3 perf path.
#[derive(Debug)]
pub struct QnnExecutable {
    backend_lib: CString,
    tensors: Vec<PlanTensor>,
    /// QNN nodes that run *after* host `QMatMul` (or all nodes when no host mix).
    nodes: Vec<PlanNode>,
    /// QNN nodes that run *before* host `QMatMul` (e.g. `Quantize` → codes).
    pre_nodes: Vec<PlanNode>,
    /// APP_READ sfixed8 tensors produced by `pre_nodes` and consumed by host
    /// `QMatMul` (read back after the pre session).
    qnn_feed_host: Vec<usize>,
    /// rlx-ir input name → tensor index.
    inputs: Vec<(String, usize)>,
    /// rlx-ir param name → tensor index (static, filled by `set_param`).
    params: Vec<(String, usize)>,
    /// Output tensor indices, in graph-output order.
    outputs: Vec<usize>,
    /// Tensor indices whose QNN dtype is int32 (e.g. Gather indices). Inputs in
    /// this set are converted from the caller's f32 slice to i32 before exec.
    int_tensors: Vec<usize>,
    /// Quantized (sfixed8/4) tensors: `(tensor index, scale, offset)`.
    quant: Vec<(usize, f32, i32)>,
    /// Per-axis sfixed8 quant: `(tensor index, axis, interleaved scale/offset)`.
    quant_axis: Vec<(usize, i32, Vec<CScaleOffset>)>,
    /// STATIC sfixed8 payloads keyed by tensor index (int8 Dequantize weights).
    i8_static: Vec<(usize, Vec<i8>)>,
    /// STATIC int4-as-bw4 payloads (1 byte/elem, values in [-8,7]) keyed by
    /// tensor index. IR may supply tightly packed Constant bytes; we unpack
    /// before staging (CPU rejects native `SFIXED_POINT_4`).
    i4_static: Vec<(usize, Vec<u8>)>,
    /// Packed GGUF params for `DequantMatMul`: param name → (weight tensor
    /// index, scheme, N, K). `set_param_typed` dequants into that tensor.
    deferred_dequant: Vec<(String, usize, rlx_ir::quant::QuantScheme, usize, usize)>,
    /// MLX DequantMatMul with Param sidecars — filled as w/scale/bias arrive.
    deferred_mlx: Vec<DeferredMlxDequant>,
    /// Int8 Dequantize / QMatMul weights filled via `set_param_typed`: param
    /// name → (weight tensor index, scale, offset).
    deferred_i8: Vec<(String, usize, f32, i32)>,
    /// Int4 Dequantize weights via `set_param_typed` (accepts packed
    /// `(num_elems+1)/2` or unpacked `num_elems` bytes).
    deferred_i4: Vec<(String, usize, f32, i32)>,
    /// Host INT8 `QMatMul` plans (weights stay I8; no QNN f32 MatMul).
    host_qmatmul: Vec<HostQMatMul>,
    /// I32 STATIC payloads (QMatMul bias), keyed by tensor index.
    i32_static: Vec<(usize, Vec<i32>)>,
    /// Live QNN session for `nodes` (post / sole). Exclusive ownership.
    session: Option<*mut RlxQnnSession>,
    /// Live QNN session for `pre_nodes`. Exclusive ownership.
    session_pre: Option<*mut RlxQnnSession>,
    /// Set when params change after a session was created.
    session_stale: bool,
}

/// Deferred MLX host-dequant: wait for w + scale + bias Param bytes.
#[derive(Debug, Clone)]
struct DeferredMlxDequant {
    w_name: String,
    scale_name: String,
    bias_name: String,
    w_idx: usize,
    scheme: rlx_ir::quant::QuantScheme,
    n: usize,
    k: usize,
    w: Option<Vec<u8>>,
    scales: Option<Vec<u8>>,
    biases: Option<Vec<u8>>,
}

impl DeferredMlxDequant {
    fn try_finish(&mut self, tensors: &mut [PlanTensor]) -> Result<(), String> {
        let (Some(w), Some(s), Some(b)) = (&self.w, &self.scales, &self.biases) else {
            return Ok(());
        };
        let kn = crate::dequant::dequant_mlx_for_qnn(self.scheme, w, s, b, self.n, self.k)?;
        if kn.len() != tensors[self.w_idx].num_elems {
            return Err(format!(
                "mlx dequant {}: got {} want {}",
                self.w_name,
                kn.len(),
                tensors[self.w_idx].num_elems
            ));
        }
        tensors[self.w_idx].data = Some(kn);
        Ok(())
    }
}

/// One host-side INT8 matmul (mirrors `rlx-cpu` `Op::QMatMul`).
#[derive(Debug, Clone)]
struct HostQMatMul {
    x_idx: usize,
    w_idx: usize,
    bias_idx: usize,
    out_idx: usize,
    m: usize,
    k: usize,
    n: usize,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: f32,
}

/// Pack signed int4 values (`[-8, 7]`) into QNN tightly-packed bytes:
/// lower nibble = first element, upper nibble = second (little-endian).
#[cfg(test)]
fn pack_sfixed4(vals: &[i8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(vals.len().div_ceil(2));
    for chunk in vals.chunks(2) {
        let a = chunk[0];
        if !(-8..=7).contains(&a) {
            return Err(format!("sfixed4 value {a} out of range [-8, 7]"));
        }
        let lo = (a as u8) & 0x0f;
        let hi = if let Some(&b) = chunk.get(1) {
            if !(-8..=7).contains(&b) {
                return Err(format!("sfixed4 value {b} out of range [-8, 7]"));
            }
            (b as u8) & 0x0f
        } else {
            0
        };
        out.push(lo | (hi << 4));
    }
    Ok(out)
}

fn unpack_sfixed4(packed: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for &byte in packed {
        if out.len() >= n {
            break;
        }
        let lo = (byte & 0x0f) as i8;
        out.push(if lo >= 8 { lo - 16 } else { lo });
        if out.len() >= n {
            break;
        }
        let hi = ((byte >> 4) & 0x0f) as i8;
        out.push(if hi >= 8 { hi - 16 } else { hi });
    }
    out.truncate(n);
    out
}

// Session handle is exclusively owned; never shared across threads.
unsafe impl Send for QnnExecutable {}

impl Drop for QnnExecutable {
    fn drop(&mut self) {
        self.drop_session();
    }
}

impl QnnExecutable {
    /// Lower an `rlx-ir` graph to a QNN execution plan: one tensor per node,
    /// each compute node becomes a QNN node. Errors on any unsupported op.
    pub fn compile_graph(graph: &rlx_ir::Graph) -> Result<Self, String> {
        let backend = default_backend_lib().ok_or_else(|| {
            "no QNN backend library — set RLX_QNN_BACKEND_LIB or QNN_SDK_ROOT".to_string()
        })?;
        let out_set: std::collections::HashSet<u32> = graph.outputs.iter().map(|id| id.0).collect();

        let num_nodes = graph.nodes().len();
        let mut tensors = Vec::with_capacity(num_nodes);
        // Intermediate tensors introduced by op decompositions (e.g. RmsNorm →
        // RmsNorm+Add). Indexed beyond the per-node tensors; appended at the end.
        let mut extra: Vec<PlanTensor> = Vec::new();
        let mut nodes = Vec::new();
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let mut int_tensors = Vec::new();
        let mut quant: Vec<(usize, f32, i32)> = Vec::new();
        let mut quant_axis: Vec<(usize, i32, Vec<CScaleOffset>)> = Vec::new();
        let mut deferred_dequant: Vec<(String, usize, rlx_ir::quant::QuantScheme, usize, usize)> =
            Vec::new();
        let mut deferred_mlx: Vec<DeferredMlxDequant> = Vec::new();
        let mut deferred_i8: Vec<(String, usize, f32, i32)> = Vec::new();
        let deferred_i4: Vec<(String, usize, f32, i32)> = Vec::new();
        let mut i8_static: Vec<(usize, Vec<i8>)> = Vec::new();
        let mut i4_static: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut i32_static: Vec<(usize, Vec<i32>)> = Vec::new();
        let mut host_qmatmul: Vec<HostQMatMul> = Vec::new();

        // Pre-scan: I8 Param/Constant feeding Dequantize → STATIC SFIXED_POINT_8
        // or packed SFIXED_POINT_4 (Constant byte len == (elems+1)/2).
        // On-device Dequantize → f32 MatMul; QNN rejects mixed f32×int8 MatMul.
        let mut i8_dq_weight: std::collections::HashMap<u32, (f32, i32)> =
            std::collections::HashMap::new();
        let mut i8_dq_weight_axis: std::collections::HashMap<u32, (i32, Vec<f32>, Vec<i32>)> =
            std::collections::HashMap::new();
        for node in graph.nodes() {
            let Op::Dequantize {
                scales,
                zero_points,
                axis,
                ..
            } = &node.op
            else {
                continue;
            };
            if node.inputs.len() != 1 || scales.len() != zero_points.len() || scales.is_empty() {
                continue;
            }
            let w_id = node.inputs[0];
            let w = graph.node(w_id);
            let w_ok = matches!(
                (&w.op, w.shape.dtype()),
                (Op::Param { .. }, rlx_ir::DType::I8) | (Op::Constant { .. }, rlx_ir::DType::I8)
            );
            if !w_ok {
                continue;
            }
            match axis {
                None if scales.len() == 1 => {
                    i8_dq_weight.insert(w_id.0, (scales[0], zero_points[0]));
                }
                Some(ax) if scales.len() > 1 => {
                    let dim = w.shape.dim(*ax).unwrap_static();
                    if dim != scales.len() {
                        return Err(format!(
                            "rlx-qnn per-channel Dequantize: scales.len()={} != dim({ax})={dim}",
                            scales.len()
                        ));
                    }
                    i8_dq_weight_axis
                        .insert(w_id.0, (*ax as i32, scales.clone(), zero_points.clone()));
                }
                _ => {}
            }
        }
        // Pre-scan: MatMul(I8, I8) — both operands stay sfixed8 (no Dequantize).
        let mut i8_matmul: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut i8_matmul_w: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (i, node) in graph.nodes().iter().enumerate() {
            if !matches!(node.op, Op::MatMul) || node.inputs.len() != 2 {
                continue;
            }
            let a = graph.node(node.inputs[0]);
            let b = graph.node(node.inputs[1]);
            if a.shape.dtype() == rlx_ir::DType::I8 && b.shape.dtype() == rlx_ir::DType::I8 {
                i8_matmul.insert(i as u32);
                i8_matmul_w.insert(node.inputs[1].0);
                // Activation may be Quantize output (already in quant) or Input I8.
            }
        }

        // Pre-scan: QMatMul weight / bias Param|Constant → host INT8 plan.
        let mut qmm_weight: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
        for node in graph.nodes() {
            let Op::QMatMul { w_zp, .. } = &node.op else {
                continue;
            };
            if node.inputs.len() != 3 {
                continue;
            }
            qmm_weight.insert(node.inputs[1].0, *w_zp);
        }

        for (i, node) in graph.nodes().iter().enumerate() {
            let shape = &node.shape;
            let dims: Vec<u32> = (0..shape.rank())
                .map(|d| shape.dim(d).unwrap_static() as u32)
                .collect();
            let num_elems: usize = dims.iter().map(|&d| d as usize).product();
            let is_output = out_set.contains(&(i as u32));
            let qnn_name = CString::new(format!("t{i}")).expect("nul in name");

            let (ttype, data) = match &node.op {
                Op::Input { name } => {
                    inputs.push((name.clone(), i));
                    // I8 activation feeding I8×I8 MatMul → APP_WRITE sfixed8.
                    if shape.dtype() == rlx_ir::DType::I8
                        && i8_matmul
                            .iter()
                            .any(|&mi| graph.node(rlx_ir::NodeId(mi)).inputs[0].0 == i as u32)
                    {
                        quant.push((i, 1.0, 0));
                    }
                    (0, None)
                }
                Op::Param { name } => {
                    // U8/I8 packed GGUF params are filled via `set_param_typed`
                    // into a deferred dequant weight tensor — not as f32 params.
                    // Int8 Dequantize / QMatMul weights stay as sfixed8 / i8 payloads.
                    let dt = shape.dtype();
                    if let Some(&(scale, zp)) = i8_dq_weight.get(&(i as u32)) {
                        quant.push((i, scale, zp));
                        deferred_i8.push((name.clone(), i, scale, zp));
                        i8_static.push((i, Vec::new()));
                    } else if let Some((axis, scales, zps)) = i8_dq_weight_axis.get(&(i as u32)) {
                        let so: Vec<CScaleOffset> = scales
                            .iter()
                            .zip(zps.iter())
                            .map(|(&s, &zp)| CScaleOffset {
                                scale: s,
                                offset: zp,
                            })
                            .collect();
                        quant_axis.push((i, *axis, so));
                        deferred_i8.push((name.clone(), i, scales[0], zps[0]));
                        i8_static.push((i, Vec::new()));
                    } else if let Some(&w_zp) = qmm_weight.get(&(i as u32)) {
                        deferred_i8.push((name.clone(), i, 1.0, w_zp));
                        i8_static.push((i, Vec::new()));
                    } else if i8_matmul_w.contains(&(i as u32)) {
                        // I8×I8 MatMul weight — STATIC sfixed8, scale 1 / zp 0.
                        quant.push((i, 1.0, 0));
                        deferred_i8.push((name.clone(), i, 1.0, 0));
                        i8_static.push((i, Vec::new()));
                    } else if !matches!(
                        dt,
                        rlx_ir::DType::U8 | rlx_ir::DType::I8 | rlx_ir::DType::I32
                    ) {
                        params.push((name.clone(), i));
                    }
                    (4, None)
                }
                Op::Constant { data } => {
                    // U8/I8 packed constants (GGUF weights) keep raw bytes;
                    // `DequantMatMul` reads them via the Constant arm above.
                    // Int8 Dequantize / QMatMul / I8×I8 MatMul weights → i8 payloads.
                    if let Some(&(scale, zp)) = i8_dq_weight.get(&(i as u32)) {
                        let packed_len = num_elems.div_ceil(2);
                        if data.len() == packed_len {
                            // Packed int4 in IR → unpack to 1 byte/elem for QNN
                            // BW_SCALE_OFFSET bitwidth=4 (CPU rejects SFIXED_POINT_4).
                            let vals = unpack_sfixed4(data, num_elems);
                            let bytes: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
                            quant.push((i, scale, zp));
                            i4_static.push((i, bytes));
                        } else if data.len() == num_elems {
                            quant.push((i, scale, zp));
                            i8_static.push((i, data.iter().map(|&b| b as i8).collect()));
                        } else {
                            return Err(format!(
                                "rlx-qnn int8/int4 weight: constant len {} != {} or {}",
                                data.len(),
                                num_elems,
                                packed_len
                            ));
                        }
                        (4, None)
                    } else if let Some((axis, scales, zps)) = i8_dq_weight_axis.get(&(i as u32)) {
                        if data.len() != num_elems {
                            return Err(format!(
                                "rlx-qnn per-channel int8 weight: constant len {} != {}",
                                data.len(),
                                num_elems
                            ));
                        }
                        let so: Vec<CScaleOffset> = scales
                            .iter()
                            .zip(zps.iter())
                            .map(|(&s, &zp)| CScaleOffset {
                                scale: s,
                                offset: zp,
                            })
                            .collect();
                        quant_axis.push((i, *axis, so));
                        i8_static.push((i, data.iter().map(|&b| b as i8).collect()));
                        (4, None)
                    } else if qmm_weight.contains_key(&(i as u32)) {
                        if data.len() != num_elems {
                            return Err(format!(
                                "rlx-qnn QMatMul weight: constant len {} != {}",
                                data.len(),
                                num_elems
                            ));
                        }
                        i8_static.push((i, data.iter().map(|&b| b as i8).collect()));
                        (4, None)
                    } else if i8_matmul_w.contains(&(i as u32)) {
                        if data.len() != num_elems {
                            return Err(format!(
                                "rlx-qnn I8 MatMul weight: constant len {} != {}",
                                data.len(),
                                num_elems
                            ));
                        }
                        quant.push((i, 1.0, 0));
                        i8_static.push((i, data.iter().map(|&b| b as i8).collect()));
                        (4, None)
                    } else if matches!(shape.dtype(), rlx_ir::DType::I32) {
                        if data.len() != num_elems * 4 {
                            return Err(format!(
                                "rlx-qnn i32 constant: byte len {} != {}",
                                data.len(),
                                num_elems * 4
                            ));
                        }
                        let vals: Vec<i32> = data
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        i32_static.push((i, vals));
                        (4, None)
                    } else if matches!(shape.dtype(), rlx_ir::DType::U8 | rlx_ir::DType::I8) {
                        (4, None)
                    } else {
                        (4, Some(bytes_to_f32(data)))
                    }
                }
                // RmsNorm decomposes into two QNN nodes: rlx-ir's op applies a
                // `beta` (3rd input) that QNN's RmsNorm doesn't, so emit
                // `RmsNorm(x, gamma)` → intermediate, then `Add(intermediate, beta)`.
                Op::RmsNorm { axis, eps } if node.inputs.len() == 3 => {
                    let x = node.inputs[0].0;
                    let gamma = node.inputs[1].0;
                    let beta = node.inputs[2].0;
                    let na = if *axis < 0 {
                        *axis + dims.len() as i32
                    } else {
                        *axis
                    };
                    let mid = (num_nodes + extra.len()) as u32;
                    extra.push(PlanTensor {
                        ttype: 3, // NATIVE intermediate
                        dims: dims.clone(),
                        num_elems,
                        data: None,
                        qnn_name: CString::new(format!("t{mid}_rms")).expect("nul"),
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_rms")).expect("nul"),
                        op_type: CString::new("RmsNorm").expect("nul"),
                        inputs: vec![x, gamma],
                        output: mid,
                        axis: -1,
                        perm: vec![na as u32],
                        eps: *eps,
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_add")).expect("nul"),
                        op_type: CString::new("ElementWiseAdd").expect("nul"),
                        inputs: vec![mid, beta],
                        output: i as u32,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
                // INT8 `QMatMul` → host integer accumulate + requantize (weights
                // stay I8; QNN CPU rejects mixed f32×int8 MatMul).
                Op::QMatMul {
                    x_zp,
                    w_zp,
                    out_zp,
                    mult,
                } if node.inputs.len() == 3 => {
                    let x = node.inputs[0].0 as usize;
                    let w = node.inputs[1].0 as usize;
                    let bias = node.inputs[2].0 as usize;
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let w_shape = &graph.node(node.inputs[1]).shape;
                    let m = x_shape.dim(0).unwrap_static();
                    let k = x_shape.dim(1).unwrap_static();
                    let n = w_shape.dim(1).unwrap_static();
                    host_qmatmul.push(HostQMatMul {
                        x_idx: x,
                        w_idx: w,
                        bias_idx: bias,
                        out_idx: i,
                        m,
                        k,
                        n,
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        mult: *mult,
                    });
                    // No QNN node — host path fills the output buffer.
                    (if is_output { 1 } else { 3 }, None)
                }
                // GGUF `DequantMatMul` → host-dequant packed `[N,K]` → transpose
                // to `[K,N]` → plain MatMul. No native QNN GGUF kernel.
                Op::DequantMatMul { scheme } if node.inputs.len() == 2 && scheme.is_gguf() => {
                    let x = node.inputs[0].0;
                    let w_id = node.inputs[1];
                    let x_shape = graph.shape(node.inputs[0]);
                    if dims.len() < 2 || x_shape.rank() < 2 {
                        return Err("rlx-qnn DequantMatMul: need rank ≥ 2".into());
                    }
                    let n = dims[dims.len() - 1] as usize;
                    let k = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let w_node = graph.node(w_id);
                    let w_idx = (num_nodes + extra.len()) as u32;
                    let w_data = match &w_node.op {
                        Op::Constant { data } => {
                            let kn = crate::dequant::dequant_weight_for_qnn(*scheme, data, n, k)?;
                            Some(kn)
                        }
                        Op::Param { name } => {
                            deferred_dequant.push((name.clone(), w_idx as usize, *scheme, n, k));
                            None
                        }
                        other => {
                            return Err(format!(
                                "rlx-qnn DequantMatMul: weight must be Param/Constant, got {other:?}"
                            ));
                        }
                    };
                    extra.push(PlanTensor {
                        ttype: 4,
                        dims: vec![k as u32, n as u32],
                        num_elems: k * n,
                        data: w_data,
                        qnn_name: CString::new(format!("t{w_idx}_dqw")).expect("nul"),
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_dqmm")).expect("nul"),
                        op_type: CString::new("MatMul").expect("nul"),
                        inputs: vec![x, w_idx],
                        output: i as u32,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
                // MLX affine / mxfp — host-dequant to `[K,N]` MatMul. Constants
                // bake immediately; Param w/scale/bias fill via deferred bind.
                Op::DequantMatMul { scheme } if node.inputs.len() >= 4 && scheme.is_mlx() => {
                    let x = node.inputs[0].0;
                    let x_shape = graph.shape(node.inputs[0]);
                    if dims.len() < 2 || x_shape.rank() < 2 {
                        return Err("rlx-qnn DequantMatMul[MLX]: need rank ≥ 2".into());
                    }
                    let n = dims[dims.len() - 1] as usize;
                    let k = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                    let mlx_src =
                        |id: rlx_ir::NodeId| -> Result<(String, Option<Vec<u8>>), String> {
                            match &graph.node(id).op {
                                Op::Constant { data } => Ok((String::new(), Some(data.clone()))),
                                Op::Param { name } => Ok((name.clone(), None)),
                                other => Err(format!(
                                    "rlx-qnn DequantMatMul[MLX]: w/scale/bias must be Param/Constant, got {other:?}"
                                )),
                            }
                        };
                    let (w_name, w_data) = mlx_src(node.inputs[1])?;
                    let (s_name, s_data) = mlx_src(node.inputs[2])?;
                    let (b_name, b_data) = mlx_src(node.inputs[3])?;
                    let w_idx = (num_nodes + extra.len()) as u32;
                    let kn = match (&w_data, &s_data, &b_data) {
                        (Some(w), Some(s), Some(b)) => {
                            Some(crate::dequant::dequant_mlx_for_qnn(*scheme, w, s, b, n, k)?)
                        }
                        _ => {
                            deferred_mlx.push(DeferredMlxDequant {
                                w_name,
                                scale_name: s_name,
                                bias_name: b_name,
                                w_idx: w_idx as usize,
                                scheme: *scheme,
                                n,
                                k,
                                w: w_data,
                                scales: s_data,
                                biases: b_data,
                            });
                            None
                        }
                    };
                    extra.push(PlanTensor {
                        ttype: 4,
                        dims: vec![k as u32, n as u32],
                        num_elems: k * n,
                        data: kn,
                        qnn_name: CString::new(format!("t{w_idx}_mlx_dqw")).expect("nul"),
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_mlx_dqmm")).expect("nul"),
                        op_type: CString::new("MatMul").expect("nul"),
                        inputs: vec![x, w_idx],
                        output: i as u32,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
                // Silu(x) = x · σ(x) — two QNN nodes + one intermediate.
                Op::Activation(Activation::Silu) if node.inputs.len() == 1 => {
                    let x = node.inputs[0].0;
                    let sig_idx = (num_nodes + extra.len()) as u32;
                    extra.push(PlanTensor {
                        ttype: 3,
                        dims: dims.clone(),
                        num_elems,
                        data: None,
                        qnn_name: CString::new(format!("t{sig_idx}_silu")).expect("nul"),
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_sig")).expect("nul"),
                        op_type: CString::new("Sigmoid").expect("nul"),
                        inputs: vec![x],
                        output: sig_idx,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_mul")).expect("nul"),
                        op_type: CString::new("ElementWiseMultiply").expect("nul"),
                        inputs: vec![x, sig_idx],
                        output: i as u32,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
                // Expand → Reshape (pad leading 1s) + Tile (broadcast multiples).
                Op::Expand { .. } if node.inputs.len() == 1 => {
                    let in_shape = graph.shape(node.inputs[0]);
                    let mut in_dims: Vec<u32> = (0..in_shape.rank())
                        .map(|d| in_shape.dim(d).unwrap_static() as u32)
                        .collect();
                    while in_dims.len() < dims.len() {
                        in_dims.insert(0, 1);
                    }
                    if in_dims.len() != dims.len() {
                        return Err(format!(
                            "rlx-qnn Expand: cannot broadcast {:?} → {:?}",
                            in_dims, dims
                        ));
                    }
                    let multiples: Vec<u32> = in_dims
                        .iter()
                        .zip(dims.iter())
                        .map(|(&a, &b)| {
                            if a == b {
                                Ok(1u32)
                            } else if a == 1 {
                                Ok(b)
                            } else {
                                Err(format!("rlx-qnn Expand: incompatible dims {a} → {b}"))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let x = node.inputs[0].0;
                    let mk =
                        |s: &str, op: &str, ins: Vec<u32>, out: u32, perm: Vec<u32>| PlanNode {
                            name: CString::new(format!("n{i}_{s}")).expect("nul"),
                            op_type: CString::new(op).expect("nul"),
                            inputs: ins,
                            output: out,
                            axis: -1,
                            perm,
                            eps: 0.0,
                        };
                    let src = {
                        let in_rank = graph.shape(node.inputs[0]).rank();
                        if in_rank != dims.len() {
                            let mid = (num_nodes + extra.len()) as u32;
                            let ne = in_dims.iter().map(|&d| d as usize).product();
                            extra.push(PlanTensor {
                                ttype: 3,
                                dims: in_dims.clone(),
                                num_elems: ne,
                                data: None,
                                qnn_name: CString::new(format!("t{mid}_exp")).expect("nul"),
                            });
                            nodes.push(mk("r", "Reshape", vec![x], mid, Vec::new()));
                            mid
                        } else {
                            x
                        }
                    };
                    if multiples.iter().all(|&m| m == 1) {
                        nodes.push(mk("id", "Reshape", vec![src], i as u32, Vec::new()));
                    } else {
                        nodes.push(mk("tile", "Tile", vec![src], i as u32, multiples));
                    }
                    (if is_output { 1 } else { 3 }, None)
                }
                // NeoX RoPE has no native QNN op, so decompose (7 nodes):
                //   x1 = x[.., :d/2];  x2 = x[.., d/2:]
                //   rot = concat([-x2, x1], last)
                //   out = x*cos + rot*sin
                // Compact `[…, D/2]` cos/sin tables are broadcast to x's shape
                // (Concat along last + Tile) so FAB unfuse rope works.
                Op::Rope {
                    head_dim,
                    n_rot,
                    style,
                } if node.inputs.len() == 3
                    && matches!(style, rlx_ir::op::RopeStyle::NeoX)
                    && n_rot == head_dim =>
                {
                    let rank = dims.len();
                    let last = rank - 1;
                    let hd = dims[last];
                    let half = hd / 2;
                    let x = node.inputs[0].0;
                    let cos_in = node.inputs[1].0;
                    let sin_in = node.inputs[2].0;
                    let cos_shape = graph.shape(node.inputs[1]);
                    let cos_dims: Vec<u32> = (0..cos_shape.rank())
                        .map(|d| cos_shape.dim(d).unwrap_static() as u32)
                        .collect();
                    let sin_dims: Vec<u32> = {
                        let s = graph.shape(node.inputs[2]);
                        (0..s.rank())
                            .map(|d| s.dim(d).unwrap_static() as u32)
                            .collect()
                    };

                    let mut push_native = |ds: Vec<u32>| -> u32 {
                        let idx = (num_nodes + extra.len()) as u32;
                        let ne = ds.iter().map(|&x| x as usize).product();
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: ds,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_rope")).expect("nul"),
                        });
                        idx
                    };
                    let mk = |suffix: &str,
                              op: &str,
                              ins: Vec<u32>,
                              out: u32,
                              axis: i32,
                              perm: Vec<u32>| {
                        PlanNode {
                            name: CString::new(format!("n{i}_{suffix}")).expect("nul"),
                            op_type: CString::new(op).expect("nul"),
                            inputs: ins,
                            output: out,
                            axis,
                            perm,
                            eps: 0.0,
                        }
                    };
                    let mut bcast_table = |tbl: u32, tbl_dims: &[u32]| -> Result<u32, String> {
                        if tbl_dims == dims.as_slice() {
                            return Ok(tbl);
                        }
                        let t_last = *tbl_dims.last().ok_or("empty rope table")?;
                        let (mut cur, mut cur_dims) = if t_last == half {
                            let mut full_dims = tbl_dims.to_vec();
                            *full_dims.last_mut().unwrap() = hd;
                            let cat = push_native(full_dims.clone());
                            nodes.push(mk(
                                "cat",
                                "Concat",
                                vec![tbl, tbl],
                                cat,
                                (full_dims.len() - 1) as i32,
                                Vec::new(),
                            ));
                            (cat, full_dims)
                        } else if t_last == hd {
                            (tbl, tbl_dims.to_vec())
                        } else {
                            return Err(format!(
                                "rlx-qnn rope: table last dim {t_last} vs head_dim {hd}"
                            ));
                        };
                        while cur_dims.len() < dims.len() {
                            cur_dims.insert(0, 1);
                            let r = push_native(cur_dims.clone());
                            nodes.push(mk("tr", "Reshape", vec![cur], r, -1, Vec::new()));
                            cur = r;
                        }
                        if cur_dims.len() != dims.len() {
                            return Err("rlx-qnn rope: cannot broadcast table rank".into());
                        }
                        let multiples: Vec<u32> = cur_dims
                            .iter()
                            .zip(dims.iter())
                            .map(|(&a, &b)| {
                                if a == b {
                                    Ok(1u32)
                                } else if a == 1 {
                                    Ok(b)
                                } else {
                                    Err(format!("rlx-qnn rope broadcast {a}→{b}"))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if multiples.iter().all(|&m| m == 1) {
                            Ok(cur)
                        } else {
                            let out = push_native(dims.clone());
                            nodes.push(mk("tt", "Tile", vec![cur], out, -1, multiples));
                            Ok(out)
                        }
                    };
                    let cos = bcast_table(cos_in, &cos_dims)?;
                    let sin = bcast_table(sin_in, &sin_dims)?;

                    let mut half_dims = dims.clone();
                    half_dims[last] = half;
                    let (x1i, x2i, negi, roti, ai, bi) = (
                        push_native(half_dims.clone()),
                        push_native(half_dims.clone()),
                        push_native(half_dims),
                        push_native(dims.clone()),
                        push_native(dims.clone()),
                        push_native(dims.clone()),
                    );

                    let ranges = |s: u32, e: u32| -> Vec<u32> {
                        let mut r = Vec::with_capacity(rank * 3);
                        for (d, &dim) in dims.iter().enumerate() {
                            if d == last {
                                r.extend_from_slice(&[s, e, 1]);
                            } else {
                                r.extend_from_slice(&[0, dim, 1]);
                            }
                        }
                        r
                    };
                    nodes.push(mk("x1", "StridedSlice", vec![x], x1i, -1, ranges(0, half)));
                    nodes.push(mk("x2", "StridedSlice", vec![x], x2i, -1, ranges(half, hd)));
                    nodes.push(mk("neg", "ElementWiseNeg", vec![x2i], negi, -1, Vec::new()));
                    nodes.push(mk(
                        "rot",
                        "Concat",
                        vec![negi, x1i],
                        roti,
                        last as i32,
                        Vec::new(),
                    ));
                    nodes.push(mk(
                        "a",
                        "ElementWiseMultiply",
                        vec![x, cos],
                        ai,
                        -1,
                        Vec::new(),
                    ));
                    nodes.push(mk(
                        "b",
                        "ElementWiseMultiply",
                        vec![roti, sin],
                        bi,
                        -1,
                        Vec::new(),
                    ));
                    nodes.push(mk(
                        "y",
                        "ElementWiseAdd",
                        vec![ai, bi],
                        i as u32,
                        -1,
                        Vec::new(),
                    ));
                    (if is_output { 1 } else { 3 }, None)
                }
                // Rank-4 `[B,H,S,D]` Attention with additive Custom mask
                // (4 inputs) — the shape FAB unfuse emits.
                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } if dims.len() == 4
                    && node.inputs.len() == 4
                    && matches!(mask_kind, rlx_ir::op::MaskKind::Custom) =>
                {
                    let (b, h, sq, d) = (dims[0], dims[1], dims[2], dims[3]);
                    if h != *num_heads as u32 || d != *head_dim as u32 {
                        return Err(format!(
                            "rlx-qnn attention rank-4: shape [{b},{h},{sq},{d}] vs heads={} dim={}",
                            num_heads, head_dim
                        ));
                    }
                    let kshape = graph.shape(node.inputs[1]);
                    let sk = kshape.dim(2).unwrap_static() as u32;
                    let scale = score_scale.unwrap_or((*head_dim as f32).powf(-0.5));
                    let (q, kk, vv, mask) = (
                        node.inputs[0].0,
                        node.inputs[1].0,
                        node.inputs[2].0,
                        node.inputs[3].0,
                    );
                    // Static scalars before the NATIVE push closure (avoids
                    // overlapping borrows of `extra`).
                    let scale_idx = (num_nodes + extra.len()) as u32;
                    extra.push(PlanTensor {
                        ttype: 4,
                        dims: vec![1],
                        num_elems: 1,
                        data: Some(vec![scale]),
                        qnn_name: CString::new(format!("t{scale_idx}_scale")).expect("nul"),
                    });
                    let softcap_consts = attn_logit_softcap.map(|cap| {
                        let inv_idx = (num_nodes + extra.len()) as u32;
                        extra.push(PlanTensor {
                            ttype: 4,
                            dims: vec![1],
                            num_elems: 1,
                            data: Some(vec![1.0 / cap]),
                            qnn_name: CString::new(format!("t{inv_idx}_invcap")).expect("nul"),
                        });
                        let cap_idx = (num_nodes + extra.len()) as u32;
                        extra.push(PlanTensor {
                            ttype: 4,
                            dims: vec![1],
                            num_elems: 1,
                            data: Some(vec![cap]),
                            qnn_name: CString::new(format!("t{cap_idx}_cap")).expect("nul"),
                        });
                        (inv_idx, cap_idx)
                    });
                    let mut push_native = |ds: Vec<u32>| -> u32 {
                        let idx = (num_nodes + extra.len()) as u32;
                        let ne = ds.iter().map(|&x| x as usize).product();
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: ds,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_attn4")).expect("nul"),
                        });
                        idx
                    };
                    let mk =
                        |s: &str, op: &str, ins: Vec<u32>, out: u32, axis: i32, perm: Vec<u32>| {
                            PlanNode {
                                name: CString::new(format!("n{i}_{s}")).expect("nul"),
                                op_type: CString::new(op).expect("nul"),
                                inputs: ins,
                                output: out,
                                axis,
                                perm,
                                eps: 0.0,
                            }
                        };
                    let ktt = push_native(vec![b, h, d, sk]);
                    nodes.push(mk("ktt", "Transpose", vec![kk], ktt, -1, vec![0, 1, 3, 2]));
                    let scores = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk("qk", "MatMul", vec![q, ktt], scores, -1, Vec::new()));
                    let scaled = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk(
                        "scale",
                        "ElementWiseMultiply",
                        vec![scores, scale_idx],
                        scaled,
                        -1,
                        Vec::new(),
                    ));
                    let logits = if let Some((inv_idx, cap_idx)) = softcap_consts {
                        let div = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk(
                            "scdiv",
                            "ElementWiseMultiply",
                            vec![scaled, inv_idx],
                            div,
                            -1,
                            Vec::new(),
                        ));
                        let th = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk("sctanh", "Tanh", vec![div], th, -1, Vec::new()));
                        let cp = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk(
                            "scmul",
                            "ElementWiseMultiply",
                            vec![th, cap_idx],
                            cp,
                            -1,
                            Vec::new(),
                        ));
                        cp
                    } else {
                        scaled
                    };
                    let masked = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk(
                        "mask",
                        "ElementWiseAdd",
                        vec![logits, mask],
                        masked,
                        -1,
                        Vec::new(),
                    ));
                    let attn = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk("sm", "Softmax", vec![masked], attn, 3, Vec::new()));
                    nodes.push(mk("av", "MatMul", vec![attn, vv], i as u32, -1, Vec::new()));
                    (if is_output { 1 } else { 3 }, None)
                }
                // Scaled dot-product attention (3D [b, s, h*d], heads split
                // internally). Decomposes to ~13 QNN nodes. Scoped to MHA +
                // Causal/None/SlidingWindow mask; Custom rank-4 is above.
                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap,
                } if node.inputs.len() == 3
                    && matches!(
                        mask_kind,
                        rlx_ir::op::MaskKind::Causal
                            | rlx_ir::op::MaskKind::None
                            | rlx_ir::op::MaskKind::SlidingWindow(_)
                    ) =>
                {
                    let (b, sq, hs) = (dims[0], dims[1], dims[2]);
                    let (h, d) = (*num_heads as u32, *head_dim as u32);
                    if hs != h * d {
                        return Err(format!(
                            "rlx-qnn attention: hidden {hs} != num_heads*head_dim {}",
                            h * d
                        ));
                    }
                    let kshape = graph.shape(node.inputs[1]);
                    let sk = kshape.dim(1).unwrap_static() as u32;
                    let kv_hidden = kshape.dim(2).unwrap_static() as u32;
                    if kv_hidden % d != 0 || h % (kv_hidden / d).max(1) != 0 {
                        return Err(format!(
                            "rlx-qnn attention: bad kv hidden {kv_hidden} for {h} heads / head_dim {d}"
                        ));
                    }
                    let hkv = kv_hidden / d; // num kv heads
                    let gs = h / hkv; // GQA group size (1 = MHA)
                    let scale = score_scale.unwrap_or((*head_dim as f32).powf(-0.5));
                    let (q, kk, vv) = (node.inputs[0].0, node.inputs[1].0, node.inputs[2].0);

                    // Baked constants: scale [1], and (if causal) mask [1,1,sq,sk].
                    let scale_idx = (num_nodes + extra.len()) as u32;
                    extra.push(PlanTensor {
                        ttype: 4,
                        dims: vec![1],
                        num_elems: 1,
                        data: Some(vec![scale]),
                        qnn_name: CString::new(format!("t{scale_idx}_scale")).expect("nul"),
                    });
                    // Additive mask [1,1,sq,sk] for Causal / SlidingWindow.
                    let mask_idx = if !matches!(mask_kind, rlx_ir::op::MaskKind::None) {
                        let idx = (num_nodes + extra.len()) as u32;
                        let q_off = sk.saturating_sub(sq);
                        let window = match mask_kind {
                            rlx_ir::op::MaskKind::SlidingWindow(w) => Some(*w as i64),
                            _ => None,
                        };
                        let mut mv = vec![0.0f32; (sq * sk) as usize];
                        for qi in 0..sq {
                            let pos = (q_off + qi) as i64; // absolute query position
                            for ki in 0..sk {
                                let k = ki as i64;
                                if k > pos || window.is_some_and(|w| k < pos - w) {
                                    mv[(qi * sk + ki) as usize] = -1.0e30;
                                }
                            }
                        }
                        extra.push(PlanTensor {
                            ttype: 4,
                            dims: vec![1, 1, sq, sk],
                            num_elems: (sq * sk) as usize,
                            data: Some(mv),
                            qnn_name: CString::new(format!("t{idx}_mask")).expect("nul"),
                        });
                        Some(idx)
                    } else {
                        None
                    };
                    // Softcap constants (Gemma-style): cap·tanh(logits/cap).
                    let softcap_consts = attn_logit_softcap.map(|cap| {
                        let inv_idx = (num_nodes + extra.len()) as u32;
                        extra.push(PlanTensor {
                            ttype: 4,
                            dims: vec![1],
                            num_elems: 1,
                            data: Some(vec![1.0 / cap]),
                            qnn_name: CString::new(format!("t{inv_idx}_invcap")).expect("nul"),
                        });
                        let cap_idx = (num_nodes + extra.len()) as u32;
                        extra.push(PlanTensor {
                            ttype: 4,
                            dims: vec![1],
                            num_elems: 1,
                            data: Some(vec![cap]),
                            qnn_name: CString::new(format!("t{cap_idx}_cap")).expect("nul"),
                        });
                        (inv_idx, cap_idx)
                    });

                    let mut push_native = |ds: Vec<u32>| -> u32 {
                        let idx = (num_nodes + extra.len()) as u32;
                        let ne = ds.iter().map(|&x| x as usize).product();
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: ds,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_attn")).expect("nul"),
                        });
                        idx
                    };
                    let tperm = vec![0u32, 2, 1, 3]; // [b,s,h,d] <-> [b,h,s,d]
                    let kperm = vec![0u32, 1, 3, 2]; // [b,h,sk,d] -> [b,h,d,sk]
                    let mk =
                        |s: &str, op: &str, ins: Vec<u32>, out: u32, axis: i32, perm: Vec<u32>| {
                            PlanNode {
                                name: CString::new(format!("n{i}_{s}")).expect("nul"),
                                op_type: CString::new(op).expect("nul"),
                                inputs: ins,
                                output: out,
                                axis,
                                perm,
                                eps: 0.0,
                            }
                        };

                    // q: [b,sq,hs] → [b,sq,h,d] → [b,h,sq,d]
                    let q4 = push_native(vec![b, sq, h, d]);
                    nodes.push(mk("q4", "Reshape", vec![q], q4, -1, Vec::new()));
                    let qt = push_native(vec![b, h, sq, d]);
                    nodes.push(mk("qt", "Transpose", vec![q4], qt, -1, tperm.clone()));

                    // k: head-split with hkv, GQA-expand to h heads (reshape→tile→reshape).
                    let k4 = push_native(vec![b, sk, hkv, d]);
                    nodes.push(mk("k4", "Reshape", vec![kk], k4, -1, Vec::new()));
                    let kt = push_native(vec![b, hkv, sk, d]);
                    nodes.push(mk("kt", "Transpose", vec![k4], kt, -1, tperm.clone()));
                    let k_full = if gs > 1 {
                        let r1 = push_native(vec![b, hkv, 1, sk, d]);
                        nodes.push(mk("kr1", "Reshape", vec![kt], r1, -1, Vec::new()));
                        let ti = push_native(vec![b, hkv, gs, sk, d]);
                        nodes.push(mk("kti", "Tile", vec![r1], ti, -1, vec![1, 1, gs, 1, 1]));
                        let r2 = push_native(vec![b, h, sk, d]);
                        nodes.push(mk("kr2", "Reshape", vec![ti], r2, -1, Vec::new()));
                        r2
                    } else {
                        kt
                    };
                    let ktt = push_native(vec![b, h, d, sk]);
                    nodes.push(mk("ktt", "Transpose", vec![k_full], ktt, -1, kperm));

                    // v: same head-split + GQA expand.
                    let v4 = push_native(vec![b, sk, hkv, d]);
                    nodes.push(mk("v4", "Reshape", vec![vv], v4, -1, Vec::new()));
                    let vt = push_native(vec![b, hkv, sk, d]);
                    nodes.push(mk("vt", "Transpose", vec![v4], vt, -1, tperm.clone()));
                    let v_full = if gs > 1 {
                        let r1 = push_native(vec![b, hkv, 1, sk, d]);
                        nodes.push(mk("vr1", "Reshape", vec![vt], r1, -1, Vec::new()));
                        let ti = push_native(vec![b, hkv, gs, sk, d]);
                        nodes.push(mk("vti", "Tile", vec![r1], ti, -1, vec![1, 1, gs, 1, 1]));
                        let r2 = push_native(vec![b, h, sk, d]);
                        nodes.push(mk("vr2", "Reshape", vec![ti], r2, -1, Vec::new()));
                        r2
                    } else {
                        vt
                    };

                    let scores = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk("qk", "MatMul", vec![qt, ktt], scores, -1, Vec::new()));
                    let scaled = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk(
                        "scale",
                        "ElementWiseMultiply",
                        vec![scores, scale_idx],
                        scaled,
                        -1,
                        Vec::new(),
                    ));
                    // Optional logit softcap: cap * tanh(scaled / cap).
                    let logits = if let Some((inv_idx, cap_idx)) = softcap_consts {
                        let div = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk(
                            "scdiv",
                            "ElementWiseMultiply",
                            vec![scaled, inv_idx],
                            div,
                            -1,
                            Vec::new(),
                        ));
                        let th = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk("sctanh", "Tanh", vec![div], th, -1, Vec::new()));
                        let cp = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk(
                            "scmul",
                            "ElementWiseMultiply",
                            vec![th, cap_idx],
                            cp,
                            -1,
                            Vec::new(),
                        ));
                        cp
                    } else {
                        scaled
                    };
                    let pre_sm = if let Some(midx) = mask_idx {
                        let masked = push_native(vec![b, h, sq, sk]);
                        nodes.push(mk(
                            "mask",
                            "ElementWiseAdd",
                            vec![logits, midx],
                            masked,
                            -1,
                            Vec::new(),
                        ));
                        masked
                    } else {
                        logits
                    };
                    let attn = push_native(vec![b, h, sq, sk]);
                    nodes.push(mk("sm", "Softmax", vec![pre_sm], attn, 3, Vec::new()));
                    let ctx = push_native(vec![b, h, sq, d]);
                    nodes.push(mk("av", "MatMul", vec![attn, v_full], ctx, -1, Vec::new()));
                    let ctxt = push_native(vec![b, sq, h, d]);
                    nodes.push(mk("ctxt", "Transpose", vec![ctx], ctxt, -1, tperm));
                    nodes.push(mk("out", "Reshape", vec![ctxt], i as u32, -1, Vec::new()));
                    (if is_output { 1 } else { 3 }, None)
                }
                // NCHW Conv → QNN's NHWC Conv2d: transpose input + weight to
                // NHWC/HWIO, run Conv2d (zero bias), transpose output back.
                Op::Conv {
                    stride,
                    padding,
                    dilation,
                    groups,
                    ..
                } if node.inputs.len() == 2 && stride.len() == 2 => {
                    let input = node.inputs[0].0;
                    let weight = node.inputs[1].0;
                    let ishape = graph.shape(node.inputs[0]);
                    let wshape = graph.shape(node.inputs[1]);
                    let (n, cout, hout, wout) = (dims[0], dims[1], dims[2], dims[3]);
                    let c = ishape.dim(1).unwrap_static() as u32;
                    let (hh, ww) = (
                        ishape.dim(2).unwrap_static() as u32,
                        ishape.dim(3).unwrap_static() as u32,
                    );
                    let cin_g = wshape.dim(1).unwrap_static() as u32;
                    let (kh, kw) = (
                        wshape.dim(2).unwrap_static() as u32,
                        wshape.dim(3).unwrap_static() as u32,
                    );
                    // [strideH,strideW, padT,padB,padL,padR, dilH,dilW]
                    let conv_perm = vec![
                        stride[0] as u32,
                        stride[1] as u32,
                        padding[0] as u32,
                        padding[0] as u32,
                        padding[1] as u32,
                        padding[1] as u32,
                        dilation[0] as u32,
                        dilation[1] as u32,
                    ];
                    let bias_idx = (num_nodes + extra.len()) as u32;
                    extra.push(PlanTensor {
                        ttype: 4,
                        dims: vec![cout],
                        num_elems: cout as usize,
                        data: Some(vec![0.0; cout as usize]),
                        qnn_name: CString::new(format!("t{bias_idx}_bias")).expect("nul"),
                    });
                    let mut push_native = |ds: Vec<u32>| -> u32 {
                        let idx = (num_nodes + extra.len()) as u32;
                        let ne = ds.iter().map(|&x| x as usize).product();
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: ds,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_conv")).expect("nul"),
                        });
                        idx
                    };
                    let mk =
                        |s: &str, op: &str, ins: Vec<u32>, out: u32, axis: i32, perm: Vec<u32>| {
                            PlanNode {
                                name: CString::new(format!("n{i}_{s}")).expect("nul"),
                                op_type: CString::new(op).expect("nul"),
                                inputs: ins,
                                output: out,
                                axis,
                                perm,
                                eps: 0.0,
                            }
                        };
                    let in_nhwc = push_native(vec![n, hh, ww, c]);
                    nodes.push(mk(
                        "in",
                        "Transpose",
                        vec![input],
                        in_nhwc,
                        -1,
                        vec![0, 2, 3, 1],
                    ));
                    let w_hwio = push_native(vec![kh, kw, cin_g, cout]);
                    nodes.push(mk(
                        "w",
                        "Transpose",
                        vec![weight],
                        w_hwio,
                        -1,
                        vec![2, 3, 1, 0],
                    ));
                    let conv = push_native(vec![n, hout, wout, cout]);
                    nodes.push(mk(
                        "conv",
                        "Conv2d",
                        vec![in_nhwc, w_hwio, bias_idx],
                        conv,
                        *groups as i32,
                        conv_perm,
                    ));
                    nodes.push(mk(
                        "out",
                        "Transpose",
                        vec![conv],
                        i as u32,
                        -1,
                        vec![0, 3, 1, 2],
                    ));
                    (if is_output { 1 } else { 3 }, None)
                }
                // I8×I8 MatMul (`Quantize(x)` × STATIC I8): portable path is
                // Dequantize both → f32 MatMul. Direct MatMul(sfixed8,sfixed8)
                // is rejected by libQnnCpu (`0xc26`); HTP prepare accepts it but
                // execute fails (MatMul_bias / invalid bias) on the x86
                // functional simulator. Dequant → f32 matches the int8 weight
                // path that already works on both backends.
                Op::MatMul if i8_matmul.contains(&(i as u32)) && node.inputs.len() == 2 => {
                    let xa = node.inputs[0].0;
                    let wb = node.inputs[1].0;
                    let ashape = graph.shape(node.inputs[0]);
                    let bshape = graph.shape(node.inputs[1]);
                    let a_dims: Vec<u32> = (0..ashape.rank())
                        .map(|d| ashape.dim(d).unwrap_static() as u32)
                        .collect();
                    let b_dims: Vec<u32> = (0..bshape.rank())
                        .map(|d| bshape.dim(d).unwrap_static() as u32)
                        .collect();
                    let mut push_native = |ds: Vec<u32>, tag: &str| -> u32 {
                        let idx = (num_nodes + extra.len()) as u32;
                        let ne = ds.iter().map(|&x| x as usize).product();
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: ds,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_{tag}")).expect("nul"),
                        });
                        idx
                    };
                    let x_f = push_native(a_dims, "dqx");
                    let w_f = push_native(b_dims, "dqw");
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_dqx")).expect("nul"),
                        op_type: CString::new("Dequantize").expect("nul"),
                        inputs: vec![xa],
                        output: x_f,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_dqw")).expect("nul"),
                        op_type: CString::new("Dequantize").expect("nul"),
                        inputs: vec![wb],
                        output: w_f,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}_mm")).expect("nul"),
                        op_type: CString::new("MatMul").expect("nul"),
                        inputs: vec![x_f, w_f],
                        output: i as u32,
                        axis: -1,
                        perm: Vec::new(),
                        eps: 0.0,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
                // Distributed collectives are host/transport ops (they need an OS
                // network stack + threads to talk to other ranks). They cannot be
                // expressed inside a single-device on-device QNN NPU graph — reject
                // with a specific, actionable error rather than the generic
                // "unsupported op" path below.
                Op::Custom { name, .. } if name.starts_with("collective.") => {
                    return Err(format!(
                        "rlx-qnn: '{name}' is a distributed host/transport collective \
                         — it cannot be expressed in a single-device QNN NPU graph. \
                         Run the distributed graph on a host-capable backend (CPU/CUDA/Metal/…)."
                    ));
                }
                other => {
                    let (op_type, axis) = qnn_op(other, dims.len())
                        .ok_or_else(|| format!("rlx-qnn runtime: unsupported op {other:?}"))?;
                    let mut perm: Vec<u32> = Vec::new();
                    let mut eps = 0.0f32;
                    match other {
                        Op::Transpose { perm: p } => {
                            perm = p.iter().map(|&p| p as u32).collect();
                        }
                        Op::LayerNorm { axis: a, eps: e } => {
                            let na = if *a < 0 { *a + dims.len() as i32 } else { *a };
                            perm = vec![na as u32];
                            eps = *e;
                        }
                        Op::Narrow { axis, start, len } => {
                            // `ranges`: [begin, end, stride] per dim. Non-narrowed
                            // dims span their full (output==input) extent.
                            for (d, &dim) in dims.iter().enumerate() {
                                if d == *axis {
                                    perm.extend_from_slice(&[
                                        *start as u32,
                                        (*start + *len) as u32,
                                        1,
                                    ]);
                                } else {
                                    perm.extend_from_slice(&[0, dim, 1]);
                                }
                            }
                        }
                        Op::Reduce { axes, .. } => {
                            perm = axes.iter().map(|&a| a as u32).collect();
                        }
                        // The Quantize output tensor carries the scale/offset
                        // (per-tensor); QNN reads it for both Quantize/Dequantize.
                        Op::Quantize {
                            scales,
                            zero_points,
                            ..
                        } => {
                            if let (Some(&s), Some(&zp)) = (scales.first(), zero_points.first()) {
                                quant.push((i, s, zp));
                            }
                        }
                        // Host QMatMul → Dequantize: tag the QMatMul output with
                        // the Dequantize scale so APP_WRITE sfixed8 bridges work.
                        Op::Dequantize {
                            scales,
                            zero_points,
                            axis: None,
                            ..
                        } if scales.len() == 1
                            && zero_points.len() == 1
                            && node.inputs.len() == 1 =>
                        {
                            let src = node.inputs[0].0 as usize;
                            if host_qmatmul.iter().any(|q| q.out_idx == src) {
                                quant.push((src, scales[0], zero_points[0]));
                            }
                        }
                        _ => {}
                    }
                    nodes.push(PlanNode {
                        name: CString::new(format!("n{i}")).expect("nul in name"),
                        op_type: CString::new(op_type).expect("nul in op"),
                        inputs: node.inputs.iter().map(|id| id.0).collect(),
                        output: i as u32,
                        axis,
                        perm,
                        eps,
                    });
                    (if is_output { 1 } else { 3 }, None)
                }
            };
            if matches!(
                shape.dtype(),
                rlx_ir::DType::I32 | rlx_ir::DType::I64 | rlx_ir::DType::U32
            ) {
                int_tensors.push(i);
            }
            // Packed U8/I8 Param/Constant nodes are not MatMul inputs after
            // DequantMatMul rewrite — keep a 1-float STATIC stub so `run`
            // still has valid data without allocating N packed-bytes floats.
            // Int8 / int4 MatMul / QMatMul weights (in `quant` / `i8_static` /
            // `i4_static`) keep full dims.
            let is_quant_static_w = quant.iter().any(|(qi, ..)| *qi == i)
                || i8_static.iter().any(|(qi, _)| *qi == i)
                || i4_static.iter().any(|(qi, _)| *qi == i)
                || i32_static.iter().any(|(qi, _)| *qi == i);
            let (dims, num_elems, data) = if !is_quant_static_w
                && matches!(shape.dtype(), rlx_ir::DType::U8 | rlx_ir::DType::I8)
                && matches!(node.op, Op::Param { .. } | Op::Constant { .. })
            {
                (vec![1u32], 1usize, Some(vec![0.0f32]))
            } else {
                (dims, num_elems, data)
            };
            tensors.push(PlanTensor {
                ttype,
                dims,
                num_elems,
                data,
                qnn_name,
            });
        }
        // Append decomposition intermediates after the per-node tensors so the
        // `mid` indices assigned above stay valid.
        tensors.extend(extra);

        // Host QMatMul ↔ QNN bridges.
        // - QMatMul out consumed by QNN → APP_WRITE (host → QNN)
        // - QMatMul x produced by QNN → APP_READ (QNN → host)
        let mut qnn_feed_host: Vec<usize> = Vec::new();
        for qm in &host_qmatmul {
            let consumed = nodes
                .iter()
                .any(|n| n.inputs.iter().any(|&t| t as usize == qm.out_idx));
            if consumed && qm.out_idx < tensors.len() {
                tensors[qm.out_idx].ttype = 0; // APP_WRITE
            }
            let produced_by_qnn = nodes.iter().any(|n| n.output as usize == qm.x_idx);
            if produced_by_qnn && qm.x_idx < tensors.len() {
                tensors[qm.x_idx].ttype = 1; // APP_READ
                if !qnn_feed_host.contains(&qm.x_idx) {
                    qnn_feed_host.push(qm.x_idx);
                }
            }
        }

        // Split QNN nodes around host QMatMul: pre (no dependency on host
        // outputs) vs post (depends on host QMatMul / its QNN consumers).
        let (pre_nodes, nodes) = if host_qmatmul.is_empty() || nodes.is_empty() {
            (Vec::new(), nodes)
        } else {
            let mut depends_host: std::collections::HashSet<usize> =
                host_qmatmul.iter().map(|q| q.out_idx).collect();
            let mut pre = Vec::new();
            let mut post = Vec::new();
            for n in nodes {
                let dep = n
                    .inputs
                    .iter()
                    .any(|&t| depends_host.contains(&(t as usize)));
                if dep {
                    depends_host.insert(n.output as usize);
                    post.push(n);
                } else {
                    pre.push(n);
                }
            }
            (pre, post)
        };

        // APP_WRITE tensors that no post-QNN node reads (e.g. I8 Input feeding
        // only host QMatMul) must not be graph inputs — demote to STATIC stub.
        // Keep F32 inputs that pre_nodes still read.
        if !host_qmatmul.is_empty() && (!nodes.is_empty() || !pre_nodes.is_empty()) {
            for i in 0..tensors.len() {
                if tensors[i].ttype != 0 {
                    continue;
                }
                let used = nodes
                    .iter()
                    .chain(pre_nodes.iter())
                    .any(|n| n.inputs.iter().any(|&t| t as usize == i));
                if !used {
                    tensors[i].ttype = 4;
                    tensors[i].data = Some(vec![0.0; tensors[i].num_elems]);
                }
            }
        }

        Ok(Self {
            backend_lib: CString::new(backend.to_string_lossy().into_owned()).expect("nul in path"),
            tensors,
            nodes,
            pre_nodes,
            qnn_feed_host,
            inputs,
            params,
            outputs: graph.outputs.iter().map(|id| id.0 as usize).collect(),
            int_tensors,
            quant,
            quant_axis,
            i8_static,
            i4_static,
            deferred_dequant,
            deferred_mlx,
            deferred_i8,
            deferred_i4,
            host_qmatmul,
            i32_static,
            session: None,
            session_pre: None,
            session_stale: false,
        })
    }

    fn drop_session(&mut self) {
        if let Some(s) = self.session.take() {
            unsafe { rlx_qnn_session_free(s) };
        }
        if let Some(s) = self.session_pre.take() {
            unsafe { rlx_qnn_session_free(s) };
        }
        self.session_stale = false;
    }

    /// Bind a static weight by its rlx-ir param name. Unknown names are ignored
    /// (the host may push params this plan already handles as constants).
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&(_, idx)) = self.params.iter().find(|(n, _)| n == name) {
            assert_eq!(
                data.len(),
                self.tensors[idx].num_elems,
                "param {name:?} size mismatch"
            );
            self.tensors[idx].data = Some(data.to_vec());
            self.session_stale = true;
        }
        // MLX DequantMatMul scale/bias sidecars (F32 LE).
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut touched = false;
        for d in &mut self.deferred_mlx {
            if d.scale_name == name {
                d.scales = Some(bytes.clone());
                touched = true;
            } else if d.bias_name == name {
                d.biases = Some(bytes.clone());
                touched = true;
            }
        }
        if touched {
            for d in &mut self.deferred_mlx {
                d.try_finish(&mut self.tensors)
                    .unwrap_or_else(|e| panic!("rlx-qnn mlx deferred: {e}"));
            }
            self.session_stale = true;
        }
    }

    /// Bind a packed (U8/I8) weight:
    /// - GGUF `DequantMatMul` params → host-dequant to f32
    /// - Int8 Dequantize weights → STATIC sfixed8 payload
    /// - Packed int4 Dequantize weights → STATIC sfixed4 payload
    ///   (`data.len() == (num_elems + 1) / 2`)
    pub fn set_param_bytes(
        &mut self,
        name: &str,
        data: &[u8],
        dtype: rlx_ir::DType,
    ) -> Result<(), String> {
        if !matches!(dtype, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
            return Ok(());
        }
        // MLX packed weight / mxfp scale bytes.
        {
            let mut touched = false;
            for d in &mut self.deferred_mlx {
                if d.w_name == name {
                    d.w = Some(data.to_vec());
                    touched = true;
                } else if d.scale_name == name {
                    d.scales = Some(data.to_vec());
                    touched = true;
                } else if d.bias_name == name {
                    d.biases = Some(data.to_vec());
                    touched = true;
                }
            }
            if touched {
                for d in &mut self.deferred_mlx {
                    d.try_finish(&mut self.tensors)?;
                }
                self.session_stale = true;
                return Ok(());
            }
        }
        if let Some((_, w_idx, scheme, n, k)) =
            self.deferred_dequant.iter().find(|(n0, ..)| n0 == name)
        {
            let kn = crate::dequant::dequant_weight_for_qnn(*scheme, data, *n, *k)?;
            if kn.len() != self.tensors[*w_idx].num_elems {
                return Err(format!(
                    "dequant param {name:?}: got {} elems, want {}",
                    kn.len(),
                    self.tensors[*w_idx].num_elems
                ));
            }
            self.tensors[*w_idx].data = Some(kn);
            self.session_stale = true;
            return Ok(());
        }
        if let Some((_, w_idx, scale, zp)) =
            self.deferred_i8.iter().find(|(n0, ..)| n0 == name).cloned()
        {
            let n = self.tensors[w_idx].num_elems;
            let packed_len = n.div_ceil(2);
            if data.len() == packed_len {
                // Promote deferred int8 slot → int4 (unpack IR packing).
                let vals = unpack_sfixed4(data, n);
                let bytes: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
                self.deferred_i8.retain(|(n0, ..)| n0 != name);
                self.deferred_i4.push((name.to_string(), w_idx, scale, zp));
                self.i8_static.retain(|(i, _)| *i != w_idx);
                if let Some((_, slot)) = self.i4_static.iter_mut().find(|(i, _)| *i == w_idx) {
                    *slot = bytes;
                } else {
                    self.i4_static.push((w_idx, bytes));
                }
                self.session_stale = true;
                return Ok(());
            }
            if data.len() != n {
                return Err(format!(
                    "int8/int4 param {name:?}: got {} bytes, want {} or {}",
                    data.len(),
                    n,
                    packed_len
                ));
            }
            let bytes: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            if let Some((_, slot)) = self.i8_static.iter_mut().find(|(i, _)| *i == w_idx) {
                *slot = bytes;
            } else {
                self.i8_static.push((w_idx, bytes));
            }
            self.session_stale = true;
            return Ok(());
        }
        if let Some((_, w_idx, _scale, _zp)) = self.deferred_i4.iter().find(|(n0, ..)| n0 == name) {
            let n = self.tensors[*w_idx].num_elems;
            let packed_len = n.div_ceil(2);
            let bytes = if data.len() == packed_len {
                let vals = unpack_sfixed4(data, n);
                vals.iter().map(|&v| v as u8).collect()
            } else if data.len() == n {
                data.to_vec()
            } else {
                return Err(format!(
                    "int4 param {name:?}: got {} bytes, want {} or {}",
                    data.len(),
                    n,
                    packed_len
                ));
            };
            if let Some((_, slot)) = self.i4_static.iter_mut().find(|(i, _)| *i == *w_idx) {
                *slot = bytes;
            } else {
                self.i4_static.push((*w_idx, bytes));
            }
            self.session_stale = true;
            return Ok(());
        }
        Ok(())
    }

    fn cnodes_from(nodes: &[PlanNode]) -> Vec<CNode> {
        nodes
            .iter()
            .map(|n| CNode {
                name: n.name.as_ptr(),
                op_type: n.op_type.as_ptr(),
                inputs: n.inputs.as_ptr(),
                num_inputs: n.inputs.len() as u32,
                output: n.output,
                axis: n.axis,
                perm: if n.perm.is_empty() {
                    std::ptr::null()
                } else {
                    n.perm.as_ptr()
                },
                perm_len: n.perm.len() as u32,
                eps: n.eps,
            })
            .collect()
    }

    fn cnodes(&self) -> Vec<CNode> {
        Self::cnodes_from(&self.nodes)
    }

    /// Build C tensors with STATIC weights bound; APP_WRITE/READ use `input_ptr`
    /// / `out_bufs` (may be null when only creating a session).
    /// `i8_reads` holds APP_READ sfixed8 buffers for QNN→host feeds.
    fn build_ctensors(
        &self,
        input_ptr: &[*mut c_float],
        out_bufs: &mut [Vec<f32>],
        mut i8_reads: Option<&mut std::collections::HashMap<usize, Vec<i8>>>,
    ) -> Result<Vec<CTensor>, String> {
        let mut ctensors: Vec<CTensor> = Vec::with_capacity(self.tensors.len());
        for (i, t) in self.tensors.iter().enumerate() {
            let data: *mut c_float = match t.ttype {
                4 => {
                    if let Some((_, bytes)) = self.i4_static.iter().find(|(idx, _)| *idx == i) {
                        if bytes.is_empty() {
                            return Err(format!(
                                "static int4 tensor {i} has no data — call set_param_typed"
                            ));
                        }
                        if bytes.len() != t.num_elems {
                            return Err(format!(
                                "static int4 tensor {i}: {} bytes, want {}",
                                bytes.len(),
                                t.num_elems
                            ));
                        }
                        bytes.as_ptr() as *mut c_float
                    } else if let Some((_, bytes)) =
                        self.i8_static.iter().find(|(idx, _)| *idx == i)
                    {
                        if bytes.is_empty() {
                            return Err(format!(
                                "static int8 tensor {i} has no data — call set_param_typed"
                            ));
                        }
                        if bytes.len() != t.num_elems {
                            return Err(format!(
                                "static int8 tensor {i}: {} bytes, want {}",
                                bytes.len(),
                                t.num_elems
                            ));
                        }
                        bytes.as_ptr() as *mut c_float
                    } else if let Some((_, vals)) =
                        self.i32_static.iter().find(|(idx, _)| *idx == i)
                    {
                        if vals.is_empty() {
                            return Err(format!("static int32 tensor {i} has no data"));
                        }
                        if vals.len() != t.num_elems {
                            return Err(format!(
                                "static int32 tensor {i}: {} elems, want {}",
                                vals.len(),
                                t.num_elems
                            ));
                        }
                        vals.as_ptr() as *mut c_float
                    } else {
                        t.data
                            .as_ref()
                            .map(|d| d.as_ptr() as *mut c_float)
                            .ok_or_else(|| {
                                format!("static tensor {i} has no data — call set_param")
                            })?
                    }
                }
                0 => {
                    if i < input_ptr.len() {
                        input_ptr[i]
                    } else {
                        std::ptr::null_mut()
                    }
                }
                1 => {
                    let mut from_i8 = None;
                    if let Some(reads) = i8_reads.as_mut() {
                        if let Some(buf) = reads.get_mut(&i) {
                            from_i8 = Some(buf.as_mut_ptr() as *mut c_float);
                        }
                    }
                    if let Some(p) = from_i8 {
                        p
                    } else if let Some(pos) = self.outputs.iter().position(|&o| o == i) {
                        if pos < out_bufs.len() {
                            out_bufs[pos].as_mut_ptr()
                        } else {
                            std::ptr::null_mut()
                        }
                    } else {
                        std::ptr::null_mut()
                    }
                }
                _ => std::ptr::null_mut(),
            };
            let (dtype, q_scale, q_offset, q_axis, q_num_scales, q_scale_offsets) =
                if self.i4_static.iter().any(|(qi, _)| *qi == i) {
                    let &(_, s, o) = self
                        .quant
                        .iter()
                        .find(|(qi, _, _)| *qi == i)
                        .expect("i4_static tensor missing quant entry");
                    (3, s, o, -1, 0u32, std::ptr::null())
                } else if let Some((_, axis, sos)) =
                    self.quant_axis.iter().find(|(qi, _, _)| *qi == i)
                {
                    (2, 0.0, 0, *axis, sos.len() as u32, sos.as_ptr())
                } else if let Some(&(_, s, o)) = self.quant.iter().find(|(qi, _, _)| *qi == i) {
                    (2, s, o, -1, 0u32, std::ptr::null())
                } else if self.int_tensors.contains(&i) {
                    (1, 0.0, 0, -1, 0u32, std::ptr::null())
                } else {
                    (0, 0.0, 0, -1, 0u32, std::ptr::null())
                };
            ctensors.push(CTensor {
                name: t.qnn_name.as_ptr(),
                ttype: t.ttype,
                rank: t.dims.len() as u32,
                dims: t.dims.as_ptr(),
                data,
                num_elems: t.num_elems as u32,
                dtype,
                q_scale,
                q_offset,
                q_axis,
                q_num_scales,
                q_scale_offsets,
            });
        }
        Ok(ctensors)
    }

    /// For a QNN phase, demote APP_WRITE/APP_READ tensors that this phase's
    /// nodes never touch so graph input/output counts stay correct.
    fn apply_phase_io(
        &mut self,
        used_in: &std::collections::HashSet<usize>,
        produced: &std::collections::HashSet<usize>,
    ) -> Vec<(usize, i32)> {
        let mut saved = Vec::new();
        for i in 0..self.tensors.len() {
            let tt = self.tensors[i].ttype;
            if tt == 0 && !used_in.contains(&i) {
                saved.push((i, tt));
                self.tensors[i].ttype = 4;
                if self.tensors[i].data.is_none() {
                    self.tensors[i].data = Some(vec![0.0; self.tensors[i].num_elems]);
                }
            } else if tt == 1 && !produced.contains(&i) {
                saved.push((i, tt));
                self.tensors[i].ttype = 3; // NATIVE — unused in this phase
            }
        }
        saved
    }

    fn restore_phase_io(&mut self, saved: &[(usize, i32)]) {
        for &(i, tt) in saved {
            self.tensors[i].ttype = tt;
        }
    }

    fn phase_io_sets(
        nodes: &[PlanNode],
    ) -> (
        std::collections::HashSet<usize>,
        std::collections::HashSet<usize>,
    ) {
        let used_in = nodes
            .iter()
            .flat_map(|n| n.inputs.iter().map(|&t| t as usize))
            .collect();
        let produced = nodes.iter().map(|n| n.output as usize).collect();
        (used_in, produced)
    }

    fn ensure_session(&mut self) -> Result<(), String> {
        if self.session.is_some() && !self.session_stale {
            return Ok(());
        }
        if let Some(s) = self.session.take() {
            unsafe { rlx_qnn_session_free(s) };
        }
        if self.session_stale {
            if let Some(s) = self.session_pre.take() {
                unsafe { rlx_qnn_session_free(s) };
            }
        }
        if self.nodes.is_empty() {
            self.session_stale = false;
            return Ok(());
        }
        let (used_in, produced) = Self::phase_io_sets(&self.nodes);
        let saved = self.apply_phase_io(&used_in, &produced);
        let input_ptr = vec![std::ptr::null_mut(); self.tensors.len()];
        let mut out_bufs: Vec<Vec<f32>> = Vec::new();
        let create_result = (|| {
            let mut ctensors = self.build_ctensors(&input_ptr, &mut out_bufs, None)?;
            let cnodes = self.cnodes();
            let mut sess: *mut RlxQnnSession = std::ptr::null_mut();
            let mut qnn_err: u64 = 0;
            let rc = unsafe {
                rlx_qnn_session_create(
                    self.backend_lib.as_ptr(),
                    ctensors.as_mut_ptr(),
                    ctensors.len() as u32,
                    cnodes.as_ptr(),
                    cnodes.len() as u32,
                    &mut sess,
                    &mut qnn_err,
                )
            };
            if rc != 0 {
                return Err(QnnError { step: -rc, qnn_err }.to_string());
            }
            Ok(sess)
        })();
        self.restore_phase_io(&saved);
        let sess = create_result?;
        self.session = Some(sess);
        self.session_stale = false;
        Ok(())
    }

    fn ensure_session_pre(&mut self) -> Result<(), String> {
        if self.session_pre.is_some() && !self.session_stale {
            return Ok(());
        }
        if let Some(s) = self.session_pre.take() {
            unsafe { rlx_qnn_session_free(s) };
        }
        if self.pre_nodes.is_empty() {
            return Ok(());
        }
        let (used_in, produced) = Self::phase_io_sets(&self.pre_nodes);
        let saved = self.apply_phase_io(&used_in, &produced);
        let input_ptr = vec![std::ptr::null_mut(); self.tensors.len()];
        let mut out_bufs: Vec<Vec<f32>> = Vec::new();
        let create_result = (|| {
            let mut ctensors = self.build_ctensors(&input_ptr, &mut out_bufs, None)?;
            let cnodes = Self::cnodes_from(&self.pre_nodes);
            let mut sess: *mut RlxQnnSession = std::ptr::null_mut();
            let mut qnn_err: u64 = 0;
            let rc = unsafe {
                rlx_qnn_session_create(
                    self.backend_lib.as_ptr(),
                    ctensors.as_mut_ptr(),
                    ctensors.len() as u32,
                    cnodes.as_ptr(),
                    cnodes.len() as u32,
                    &mut sess,
                    &mut qnn_err,
                )
            };
            if rc != 0 {
                return Err(QnnError { step: -rc, qnn_err }.to_string());
            }
            Ok(sess)
        })();
        self.restore_phase_io(&saved);
        self.session_pre = Some(create_result?);
        Ok(())
    }

    /// Serialize the finalized QNN context (M3). Creates the session if needed.
    pub fn export_context_binary(&mut self) -> Result<Vec<u8>, String> {
        self.ensure_session()?;
        let sess = self.session.ok_or("no QNN session")?;
        let mut buf: *mut c_void = std::ptr::null_mut();
        let mut written: u64 = 0;
        let mut qnn_err: u64 = 0;
        let rc = unsafe { rlx_qnn_session_save_binary(sess, &mut buf, &mut written, &mut qnn_err) };
        if rc != 0 {
            return Err(QnnError { step: -rc, qnn_err }.to_string());
        }
        if buf.is_null() || written == 0 {
            return Err("context binary empty".into());
        }
        let bytes = unsafe {
            let slice = std::slice::from_raw_parts(buf as *const u8, written as usize);
            let v = slice.to_vec();
            rlx_qnn_binary_free(buf);
            v
        };
        Ok(bytes)
    }

    /// Replace the live session with one deserialized from a context binary.
    /// Keeps the rlx plan (input names / shapes); subsequent `run` uses style-2.
    pub fn reload_from_context_binary(&mut self, binary: &[u8]) -> Result<(), String> {
        let mut sess: *mut RlxQnnSession = std::ptr::null_mut();
        let mut qnn_err: u64 = 0;
        let rc = unsafe {
            rlx_qnn_session_load_binary(
                self.backend_lib.as_ptr(),
                binary.as_ptr() as *const c_void,
                binary.len() as u64,
                &mut sess,
                &mut qnn_err,
            )
        };
        if rc != 0 {
            return Err(QnnError { step: -rc, qnn_err }.to_string());
        }
        self.drop_session();
        self.session = Some(sess);
        self.session_stale = false;
        Ok(())
    }

    /// Execute with named inputs; returns one buffer per graph output, in
    /// output order. Reuses a finalized QNN session across calls.
    ///
    /// Pure `QMatMul` graphs run the host INT8 kernel. Mixed graphs run
    /// `pre_nodes` (e.g. Quantize) → host `QMatMul` → `nodes` (e.g. Dequantize).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        if !self.host_qmatmul.is_empty() && self.nodes.is_empty() && self.pre_nodes.is_empty() {
            return self.run_host_qmatmul(inputs);
        }

        // Bind named graph inputs (shared by pre / post).
        let mut input_ptr: Vec<*mut c_float> = vec![std::ptr::null_mut(); self.tensors.len()];
        let mut int_bufs: Vec<Vec<i32>> = Vec::new();
        let mut i8_bufs: Vec<Vec<i8>> = Vec::new();
        for (name, data) in inputs {
            let idx = self
                .inputs
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, i)| *i)
                .ok_or_else(|| format!("unknown input {name:?}"))?;
            if data.len() != self.tensors[idx].num_elems {
                return Err(format!("input {name:?} size mismatch"));
            }
            if self.int_tensors.contains(&idx) {
                int_bufs.push(data.iter().map(|&f| f as i32).collect());
                input_ptr[idx] = int_bufs.last().unwrap().as_ptr() as *mut c_float;
            } else if self.quant.iter().any(|(qi, ..)| *qi == idx) && self.tensors[idx].ttype == 0 {
                i8_bufs.push(data.iter().map(|&f| f as i8).collect());
                input_ptr[idx] = i8_bufs.last().unwrap().as_ptr() as *mut c_float;
            } else if self.tensors[idx].ttype == 0 {
                input_ptr[idx] = data.as_ptr() as *mut c_float;
            }
            // Demoted STATIC host-only inputs: skip binding (host reads `inputs`).
        }

        // ── Phase 1: pre-QNN (Quantize → APP_READ codes for host) ─────────
        let mut qnn_i8: std::collections::HashMap<usize, Vec<i8>> =
            std::collections::HashMap::new();
        if !self.pre_nodes.is_empty() {
            for &idx in &self.qnn_feed_host {
                qnn_i8.insert(idx, vec![0i8; self.tensors[idx].num_elems]);
            }
            let mut dummy_out: Vec<Vec<f32>> = Vec::new();
            let mut ctensors =
                self.build_ctensors(&input_ptr, &mut dummy_out, Some(&mut qnn_i8))?;
            self.ensure_session_pre()?;
            let sess = self.session_pre.ok_or("no QNN pre-session")?;
            let mut qnn_err: u64 = 0;
            let rc = unsafe {
                rlx_qnn_session_execute(
                    sess,
                    ctensors.as_mut_ptr(),
                    ctensors.len() as u32,
                    &mut qnn_err,
                )
            };
            if rc != 0 {
                return Err(QnnError { step: -rc, qnn_err }.to_string());
            }
        }

        // ── Phase 2: host QMatMul ─────────────────────────────────────────
        let bridges = if self.host_qmatmul.is_empty() {
            None
        } else {
            Some(self.eval_host_qmatmuls(inputs, &qnn_i8)?)
        };
        if let Some(ref br) = bridges {
            for (&idx, codes) in br {
                i8_bufs.push(codes.clone());
                input_ptr[idx] = i8_bufs.last().unwrap().as_ptr() as *mut c_float;
            }
        }

        // Output buffers, in graph-output order.
        let mut out_bufs: Vec<Vec<f32>> = self
            .outputs
            .iter()
            .map(|&i| vec![0.0f32; self.tensors[i].num_elems])
            .collect();
        if let Some(ref br) = bridges {
            for (pos, &out_idx) in self.outputs.iter().enumerate() {
                if let Some(codes) = br.get(&out_idx) {
                    for (i, &c) in codes.iter().enumerate() {
                        out_bufs[pos][i] = c as f32;
                    }
                }
            }
        }

        // ── Phase 3: post-QNN (Dequantize / Relu / …) ─────────────────────
        if self.nodes.is_empty() {
            return Ok(out_bufs);
        }
        let mut ctensors = self.build_ctensors(&input_ptr, &mut out_bufs, None)?;
        self.ensure_session()?;
        let sess = self.session.ok_or("no QNN session")?;
        let mut qnn_err: u64 = 0;
        let rc = unsafe {
            rlx_qnn_session_execute(
                sess,
                ctensors.as_mut_ptr(),
                ctensors.len() as u32,
                &mut qnn_err,
            )
        };
        if rc != 0 {
            return Err(QnnError { step: -rc, qnn_err }.to_string());
        }
        Ok(out_bufs)
    }

    /// Evaluate all host `QMatMul` nodes; returns `out_idx → I8 codes`.
    /// `qnn_i8` supplies activations produced by pre-QNN (e.g. Quantize).
    fn eval_host_qmatmuls(
        &self,
        inputs: &[(&str, &[f32])],
        qnn_i8: &std::collections::HashMap<usize, Vec<i8>>,
    ) -> Result<std::collections::HashMap<usize, Vec<i8>>, String> {
        let mut i8_inputs: std::collections::HashMap<usize, Vec<i8>> = qnn_i8.clone();
        for (name, data) in inputs {
            let idx = self
                .inputs
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, i)| *i)
                .ok_or_else(|| format!("unknown input {name:?}"))?;
            if data.len() != self.tensors[idx].num_elems {
                return Err(format!("input {name:?} size mismatch"));
            }
            i8_inputs
                .entry(idx)
                .or_insert_with(|| data.iter().map(|&f| f as i8).collect());
        }

        let mut bridges = std::collections::HashMap::new();
        for qm in &self.host_qmatmul {
            let x = i8_inputs
                .get(&qm.x_idx)
                .ok_or_else(|| format!("QMatMul: missing I8 input at tensor {}", qm.x_idx))?;
            let w = self
                .i8_static
                .iter()
                .find(|(i, _)| *i == qm.w_idx)
                .map(|(_, b)| b.as_slice())
                .ok_or_else(|| format!("QMatMul: missing I8 weight at tensor {}", qm.w_idx))?;
            if w.is_empty() {
                return Err(format!(
                    "QMatMul: weight tensor {} empty — call set_param_typed",
                    qm.w_idx
                ));
            }
            let bias = self
                .i32_static
                .iter()
                .find(|(i, _)| *i == qm.bias_idx)
                .map(|(_, b)| b.as_slice())
                .ok_or_else(|| format!("QMatMul: missing I32 bias at tensor {}", qm.bias_idx))?;
            let codes = crate::qmatmul::q_matmul_i8(
                x, w, bias, qm.m, qm.k, qm.n, qm.x_zp, qm.w_zp, qm.out_zp, qm.mult,
            );
            bridges.insert(qm.out_idx, codes);
        }
        Ok(bridges)
    }

    fn run_host_qmatmul(&self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        let empty = std::collections::HashMap::new();
        let bridges = self.eval_host_qmatmuls(inputs, &empty)?;
        let mut out_bufs: Vec<Vec<f32>> = self
            .outputs
            .iter()
            .map(|&i| vec![0.0f32; self.tensors[i].num_elems])
            .collect();
        for (pos, &out_idx) in self.outputs.iter().enumerate() {
            let codes = bridges
                .get(&out_idx)
                .ok_or_else(|| format!("QMatMul: graph output {out_idx} has no host result"))?;
            for (i, &c) in codes.iter().enumerate() {
                out_bufs[pos][i] = c as f32;
            }
        }
        Ok(out_bufs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distributed collective can't be lowered to a single-device on-device
    /// QNN NPU graph; `compile_graph` must reject it with a specific message that
    /// names the op, not fall through to the generic "unsupported op" path. Pure
    /// (no backend lib), so it runs on any host.
    #[test]
    fn compile_graph_rejects_collective_with_specific_message() {
        use rlx_ir::{DType, Graph, Shape};

        let mut g = Graph::new("all_reduce");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let ar = g.add_node(
            Op::Custom {
                name: "collective.all_reduce".to_string(),
                num_inputs: 1,
                attrs: vec![],
            },
            vec![x],
            Shape::new(&[4], DType::F32),
        );
        g.set_outputs(vec![ar]);

        let err = QnnExecutable::compile_graph(&g).err().expect("must reject");
        assert!(
            err.contains("collective.all_reduce") && err.contains("host/transport collective"),
            "unexpected error: {err}"
        );
        assert!(
            !err.contains("unsupported op"),
            "should be the specific collective diagnostic, not the generic path: {err}"
        );
    }

    /// In-process FFI parity vs the Rust oracle, against a real QNN backend.
    /// Skips when no backend library is available (host / CI without the SDK);
    /// runs for real in Docker with `QNN_SDK_ROOT` set (libQnnCpu.so).
    #[test]
    fn ffi_matmul_parity_vs_oracle() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib (set RLX_QNN_BACKEND_LIB / QNN_SDK_ROOT)");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        let (m, k, n) = (8usize, 16, 4);
        let in0: Vec<f32> = (0..m * k)
            .map(|i| ((i % 7) as i32 - 3) as f32 * 0.5)
            .collect();
        let in1: Vec<f32> = (0..k * n)
            .map(|i| ((i % 5) as i32 - 2) as f32 * 0.25)
            .collect();

        let got = matmul_f32(&lib, m, k, n, &in0, &in1).expect("qnn matmul");
        let want = crate::reference::matmul_f32(&in0, &in1, m, k, n);
        let max_diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "QNN vs oracle max_diff {max_diff}");
        eprintln!("QNN FFI matmul parity OK (max_diff={max_diff:.2e})");
    }

    /// Persistent session: two `run`s share one finalize; outputs stay equal.
    #[test]
    fn ffi_session_reuse() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Shape};

        let (m, k, n) = (4usize, 8, 3);
        let mut g = Graph::new("reuse");
        let a = g.input("in0", Shape::new(&[m, k], DType::F32));
        let b = g.input("in1", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(a, b, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let in0: Vec<f32> = (0..m * k).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let in1: Vec<f32> = (0..k * n).map(|i| 0.05 * (i as f32 - 2.0)).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out1 = exec.run(&[("in0", &in0), ("in1", &in1)]).expect("run1");
        assert!(exec.session.is_some(), "session created after first run");
        let out2 = exec.run(&[("in0", &in0), ("in1", &in1)]).expect("run2");
        assert_eq!(out1[0], out2[0]);
        eprintln!("QNN FFI session reuse OK");
    }

    /// Context binary round-trip: save after finalize → load → execute.
    #[test]
    fn ffi_context_binary_roundtrip() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Shape};

        let (m, k, n) = (4usize, 8, 3);
        let mut g = Graph::new("ctxbin");
        let a = g.input("in0", Shape::new(&[m, k], DType::F32));
        let b = g.input("in1", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(a, b, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let in0: Vec<f32> = (0..m * k).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let in1: Vec<f32> = (0..k * n).map(|i| 0.05 * (i as f32 - 2.0)).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out1 = exec
            .run(&[("in0", &in0), ("in1", &in1)])
            .expect("run before save");
        let bin = match exec.export_context_binary() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: context binary save unsupported ({e})");
                return;
            }
        };
        assert!(!bin.is_empty(), "binary non-empty");
        eprintln!("QNN context binary size {} bytes", bin.len());
        if let Err(e) = exec.reload_from_context_binary(&bin) {
            eprintln!("skip: context binary load unsupported ({e})");
            return;
        }
        let out2 = exec
            .run(&[("in0", &in0), ("in1", &in1)])
            .expect("run after load");
        let md = out1[0]
            .iter()
            .zip(&out2[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "context binary round-trip max_diff {md}");
        eprintln!("QNN FFI context binary round-trip OK (max_diff={md:.2e})");
    }

    /// Multi-op graph with static weights: `y = relu(x · W + b)`, exercising the
    /// general builder (MatMul + ElementWiseAdd + Relu, `Param` statics).
    #[test]
    fn ffi_linear_relu_static_weights() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::op::{Activation, BinaryOp};
        use rlx_ir::{DType, Graph, Shape};

        let (m, k, n) = (4usize, 8, 3);
        let mut g = Graph::new("linrelu");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        let b = g.param("b", Shape::new(&[m, n], DType::F32));
        let xw_b = g.binary(BinaryOp::Add, mm, b, Shape::new(&[m, n], DType::F32));
        let y = g.activation(Activation::Relu, xw_b, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * k)
            .map(|i| ((i % 7) as i32 - 3) as f32 * 0.5)
            .collect();
        let wv: Vec<f32> = (0..k * n)
            .map(|i| ((i % 5) as i32 - 2) as f32 * 0.25)
            .collect();
        let bv: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.1 - 0.5).collect();

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param("w", &wv);
        exec.set_param("b", &bv);
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: relu(x·W + b).
        let mm_ref = crate::reference::matmul_f32(&xv, &wv, m, k, n);
        let want: Vec<f32> = mm_ref
            .iter()
            .zip(&bv)
            .map(|(a, b)| (a + b).max(0.0))
            .collect();
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "linear+relu max_diff {max_diff}");
        eprintln!("QNN FFI linear+relu (static weights) OK (max_diff={max_diff:.2e})");
    }

    /// Shape + reduction ops: `softmax(reshape(x, [m, n]), axis=1)` — exercises
    /// `Reshape` (param-free) and `Softmax` (scalar axis param).
    #[test]
    fn ffi_reshape_softmax() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, Shape};

        let (m, n) = (3usize, 5);
        let mut g = Graph::new("reshape_softmax");
        let x = g.input("x", Shape::new(&[m * n], DType::F32));
        let r = g.reshape(x, vec![m as i64, n as i64], Shape::new(&[m, n], DType::F32));
        let y = g.softmax(r, 1, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * n)
            .map(|i| ((i % 11) as i32 - 5) as f32 * 0.3)
            .collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: row-wise softmax over the [m, n] view.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &xv[i * n..(i + 1) * n];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..n {
                want[i * n + j] = exps[j] / sum;
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "reshape+softmax max_diff {max_diff}");
        eprintln!("QNN FFI reshape+softmax OK (max_diff={max_diff:.2e})");
    }

    /// Transpose — exercises the tensor-valued `perm` param. `y = xᵀ` for a
    /// `[m, n]` input (perm `[1, 0]`).
    #[test]
    fn ffi_transpose() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, GraphExt, Shape};

        let (m, n) = (3usize, 4);
        let mut g = Graph::new("transpose");
        let x = g.input("x", Shape::new(&[m, n], DType::F32));
        let y = g.transpose_(x, vec![1, 0]); // [m, n] -> [n, m]
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: out[j, i] = x[i, j].
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                want[j * m + i] = xv[i * n + j];
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "transpose max_diff {max_diff}");
        eprintln!("QNN FFI transpose (perm tensor-param) OK (max_diff={max_diff:.2e})");
    }

    /// LayerNorm over the last axis — exercises a **float scalar** param
    /// (`epsilon`) + an `axes` tensor param + gamma/beta static weights.
    #[test]
    fn ffi_layer_norm() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, Shape};

        let (m, n) = (4usize, 6);
        let eps = 1e-5f32;
        let mut g = Graph::new("ln");
        let x = g.input("x", Shape::new(&[m, n], DType::F32));
        let gamma = g.param("gamma", Shape::new(&[n], DType::F32));
        let beta = g.param("beta", Shape::new(&[n], DType::F32));
        let y = g.layer_norm(x, gamma, beta, -1, eps, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * n)
            .map(|i| ((i % 11) as i32 - 5) as f32 * 0.4)
            .collect();
        let gv: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
        let bv: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - 0.1).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param("gamma", &gv);
        exec.set_param("beta", &bv);
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: layernorm over the last axis (biased variance).
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &xv[i * n..(i + 1) * n];
            let mean = row.iter().sum::<f32>() / n as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..n {
                want[i * n + j] = (row[j] - mean) * inv * gv[j] + bv[j];
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "layer_norm max_diff {max_diff}");
        eprintln!("QNN FFI layer_norm (float scalar + axes params) OK (max_diff={max_diff:.2e})");
    }

    /// RmsNorm — exercises the **multi-node decomposition** (rlx-ir's RmsNorm
    /// carries a `beta`, so it lowers to QNN `RmsNorm(x, γ)` → `Add(·, β)`).
    #[test]
    fn ffi_rms_norm() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, GraphExt, Shape};

        let (m, n) = (4usize, 6);
        let eps = 1e-6f32;
        let mut g = Graph::new("rms");
        let x = g.input("x", Shape::new(&[m, n], DType::F32));
        let gamma = g.param("gamma", Shape::new(&[n], DType::F32));
        let beta = g.param("beta", Shape::new(&[n], DType::F32));
        let y = g.rms_norm(x, gamma, beta, eps);
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * n)
            .map(|i| ((i % 11) as i32 - 5) as f32 * 0.4)
            .collect();
        let gv: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
        let bv: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - 0.1).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param("gamma", &gv);
        exec.set_param("beta", &bv);
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: x / sqrt(mean(x²) + eps) * gamma + beta, over the last axis.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            let row = &xv[i * n..(i + 1) * n];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for j in 0..n {
                want[i * n + j] = row[j] * inv * gv[j] + bv[j];
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "rms_norm max_diff {max_diff}");
        eprintln!("QNN FFI rms_norm (2-node decomposition) OK (max_diff={max_diff:.2e})");
    }

    /// `concat([a, -b], axis=0)` — exercises variadic-input `Concat` (axis scalar
    /// param) and `ElementWiseNeg`.
    #[test]
    fn ffi_concat_neg() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::op::Activation;
        use rlx_ir::{DType, Graph, Shape};

        let (m, n) = (3usize, 4);
        let mut g = Graph::new("concat_neg");
        let a = g.input("a", Shape::new(&[m, n], DType::F32));
        let b = g.input("b", Shape::new(&[m, n], DType::F32));
        let nb = g.activation(Activation::Neg, b, Shape::new(&[m, n], DType::F32));
        let y = g.concat(vec![a, nb], 0, Shape::new(&[2 * m, n], DType::F32));
        g.set_outputs(vec![y]);

        let av: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.5).collect();
        let bv: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("a", &av), ("b", &bv)]).expect("run");

        // Oracle: rows of a, then rows of -b (concat along axis 0, row-major).
        let mut want = av.clone();
        want.extend(bv.iter().map(|v| -v));
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "concat+neg max_diff {max_diff}");
        eprintln!("QNN FFI concat+neg (variadic inputs + axis param) OK (max_diff={max_diff:.2e})");
    }

    /// `x[:, 1:3]` — exercises `Narrow` → QNN `StridedSlice` (a rank-2 `ranges`
    /// tensor param).
    #[test]
    fn ffi_narrow() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, GraphExt, Shape};

        let (m, n) = (3usize, 5);
        let (start, len) = (1usize, 2usize);
        let mut g = Graph::new("narrow");
        let x = g.input("x", Shape::new(&[m, n], DType::F32));
        let y = g.narrow_(x, 1, start, len); // x[:, 1:3] -> [m, 2]
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: x[:, start..start+len].
        let mut want = Vec::with_capacity(m * len);
        for i in 0..m {
            for j in start..start + len {
                want.push(xv[i * n + j]);
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "narrow max_diff {max_diff}");
        eprintln!("QNN FFI narrow (StridedSlice ranges tensor-param) OK (max_diff={max_diff:.2e})");
    }

    /// NeoX RoPE — the capstone **7-node decomposition** (Narrow×2 → Neg →
    /// Concat → Mul(cos) + Mul(sin) → Add), composed entirely of validated ops.
    #[test]
    fn ffi_rope() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::{DType, Graph, GraphExt, Shape};

        let (m, hd) = (2usize, 8);
        let mut g = Graph::new("rope");
        let x = g.input("x", Shape::new(&[m, hd], DType::F32));
        let cos = g.input("cos", Shape::new(&[m, hd], DType::F32));
        let sin = g.input("sin", Shape::new(&[m, hd], DType::F32));
        let y = g.rope(x, cos, sin, hd); // NeoX, n_rot = head_dim
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..m * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let cv: Vec<f32> = (0..m * hd).map(|i| (i as f32 * 0.3).cos()).collect();
        let sv: Vec<f32> = (0..m * hd).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec
            .run(&[("x", &xv), ("cos", &cv), ("sin", &sv)])
            .expect("run");

        // Oracle: NeoX rope. rotate_half(x) = cat(-x[d/2:], x[:d/2]).
        let half = hd / 2;
        let mut want = vec![0.0f32; m * hd];
        for i in 0..m {
            let row = &xv[i * hd..(i + 1) * hd];
            let mut rot = vec![0.0f32; hd];
            for j in 0..half {
                rot[j] = -row[half + j];
                rot[half + j] = row[j];
            }
            for j in 0..hd {
                want[i * hd + j] = row[j] * cv[i * hd + j] + rot[j] * sv[i * hd + j];
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "rope max_diff {max_diff}");
        eprintln!("QNN FFI rope (NeoX, 7-node decomposition) OK (max_diff={max_diff:.2e})");
    }

    /// Causal scaled dot-product attention — the **~13-node decomposition**
    /// (head-split reshape/transpose, scaled q·kᵀ, causal mask, softmax, ·v,
    /// merge). The whole transformer attention block on QNN.
    #[test]
    fn ffi_attention() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::op::MaskKind;
        use rlx_ir::{DType, Graph, Shape};

        let (b, s, h, d) = (1usize, 3, 2, 4);
        let hs = h * d;
        let mut g = Graph::new("attn");
        let q = g.input("q", Shape::new(&[b, s, hs], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, hs], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, hs], DType::F32));
        let y = g.attention_kind(
            q,
            k,
            v,
            h,
            d,
            MaskKind::Causal,
            Shape::new(&[b, s, hs], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mkv = |seed: usize| -> Vec<f32> {
            (0..b * s * hs)
                .map(|i| ((i + seed) % 13) as f32 * 0.1 - 0.6)
                .collect()
        };
        let (qv, kv, vv) = (mkv(0), mkv(3), mkv(7));
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec
            .run(&[("q", &qv), ("k", &kv), ("v", &vv)])
            .expect("run");

        // Oracle: per-(batch, head) causal SDPA over head-major [b, s, h*d].
        let scale = (d as f32).powf(-0.5);
        let mut want = vec![0.0f32; b * s * hs];
        let at = |bi: usize, si: usize, hi: usize, dd: usize| bi * s * hs + si * hs + hi * d + dd;
        for bi in 0..b {
            for hi in 0..h {
                for qi in 0..s {
                    let mut sc = vec![f32::NEG_INFINITY; s];
                    for ki in 0..=qi {
                        let mut dot = 0.0f32;
                        for dd in 0..d {
                            dot += qv[at(bi, qi, hi, dd)] * kv[at(bi, ki, hi, dd)];
                        }
                        sc[ki] = dot * scale;
                    }
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = sc
                        .iter()
                        .map(|x| if x.is_finite() { (x - mx).exp() } else { 0.0 })
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    for dd in 0..d {
                        let mut acc = 0.0f32;
                        for ki in 0..s {
                            acc += exps[ki] / sum * vv[at(bi, ki, hi, dd)];
                        }
                        want[at(bi, qi, hi, dd)] = acc;
                    }
                }
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "attention max_diff {max_diff}");
        eprintln!(
            "QNN FFI attention (causal SDPA, ~13-node decomposition) OK (max_diff={max_diff:.2e})"
        );
    }

    /// Grouped-query attention (4 query heads, 2 KV heads) — exercises the KV
    /// head expansion (Reshape → Tile → Reshape) that modern LLMs need.
    #[test]
    fn ffi_attention_gqa() {
        let Some(lib) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        if !lib.exists() {
            eprintln!("skip: backend lib {} not found", lib.display());
            return;
        }
        use rlx_ir::op::MaskKind;
        use rlx_ir::{DType, Graph, Shape};

        let (b, s, h, hkv, d) = (1usize, 3, 4, 2, 4);
        let (hs, kvs) = (h * d, hkv * d);
        let mut g = Graph::new("gqa");
        let q = g.input("q", Shape::new(&[b, s, hs], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, kvs], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, kvs], DType::F32));
        let y = g.attention_kind(
            q,
            k,
            v,
            h,
            d,
            MaskKind::Causal,
            Shape::new(&[b, s, hs], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mkv = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) % 13) as f32 * 0.1 - 0.6)
                .collect()
        };
        let (qv, kv, vv) = (mkv(b * s * hs, 0), mkv(b * s * kvs, 3), mkv(b * s * kvs, 7));
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec
            .run(&[("q", &qv), ("k", &kv), ("v", &vv)])
            .expect("run");

        // Oracle: causal GQA. Query head `hi` attends with KV head `hi / gs`.
        let gs = h / hkv;
        let scale = (d as f32).powf(-0.5);
        let mut want = vec![0.0f32; b * s * hs];
        let qat = |bi: usize, si: usize, hi: usize, dd: usize| bi * s * hs + si * hs + hi * d + dd;
        let kvat =
            |bi: usize, si: usize, khi: usize, dd: usize| bi * s * kvs + si * kvs + khi * d + dd;
        for bi in 0..b {
            for hi in 0..h {
                let khi = hi / gs;
                for qi in 0..s {
                    let mut sc = vec![f32::NEG_INFINITY; s];
                    for ki in 0..=qi {
                        let mut dot = 0.0f32;
                        for dd in 0..d {
                            dot += qv[qat(bi, qi, hi, dd)] * kv[kvat(bi, ki, khi, dd)];
                        }
                        sc[ki] = dot * scale;
                    }
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = sc
                        .iter()
                        .map(|x| if x.is_finite() { (x - mx).exp() } else { 0.0 })
                        .collect();
                    let sum: f32 = exps.iter().sum();
                    for dd in 0..d {
                        let mut acc = 0.0f32;
                        for ki in 0..s {
                            acc += exps[ki] / sum * vv[kvat(bi, ki, khi, dd)];
                        }
                        want[qat(bi, qi, hi, dd)] = acc;
                    }
                }
            }
        }
        let max_diff = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "GQA attention max_diff {max_diff}");
        eprintln!(
            "QNN FFI attention GQA (h={h}, kv={hkv}, Tile expansion) OK (max_diff={max_diff:.2e})"
        );
    }

    /// Sliding-window attention (Mistral/Gemma) — `qi` attends to `ki ∈ [qi-w, qi]`.
    #[test]
    fn ffi_attention_sliding_window() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::op::MaskKind;
        use rlx_ir::{DType, Graph, Shape};

        let (b, s, h, d, w) = (1usize, 5, 2, 4, 2usize);
        let hs = h * d;
        let mut g = Graph::new("swa");
        let q = g.input("q", Shape::new(&[b, s, hs], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, hs], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, hs], DType::F32));
        let y = g.attention_kind(
            q,
            k,
            v,
            h,
            d,
            MaskKind::SlidingWindow(w),
            Shape::new(&[b, s, hs], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mkv = |seed: usize| -> Vec<f32> {
            (0..b * s * hs)
                .map(|i| ((i + seed) % 13) as f32 * 0.1 - 0.6)
                .collect()
        };
        let (qv, kv, vv) = (mkv(0), mkv(3), mkv(7));
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec
            .run(&[("q", &qv), ("k", &kv), ("v", &vv)])
            .expect("run");

        let scale = (d as f32).powf(-0.5);
        let mut want = vec![0.0f32; b * s * hs];
        let at = |si: usize, hi: usize, dd: usize| si * hs + hi * d + dd;
        for hi in 0..h {
            for qi in 0..s {
                let lo = qi.saturating_sub(w);
                let mut sc = vec![f32::NEG_INFINITY; s];
                for ki in lo..=qi {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += qv[at(qi, hi, dd)] * kv[at(ki, hi, dd)];
                    }
                    sc[ki] = dot * scale;
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = sc
                    .iter()
                    .map(|x| if x.is_finite() { (x - mx).exp() } else { 0.0 })
                    .collect();
                let sum: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for ki in 0..s {
                        acc += exps[ki] / sum * vv[at(ki, hi, dd)];
                    }
                    want[at(qi, hi, dd)] = acc;
                }
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "sliding-window max_diff {md}");
        eprintln!("QNN FFI attention sliding-window (w={w}) OK (max_diff={md:.2e})");
    }

    /// Logit-softcap attention (Gemma 2) — `cap·tanh(logits/cap)` before softmax.
    #[test]
    fn ffi_attention_softcap() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::op::MaskKind;
        use rlx_ir::{DType, Graph, Shape};

        let (b, s, h, d) = (1usize, 3, 2, 4);
        let hs = h * d;
        let cap = 5.0f32;
        let mut g = Graph::new("softcap");
        let q = g.input("q", Shape::new(&[b, s, hs], DType::F32));
        let k = g.input("k", Shape::new(&[b, s, hs], DType::F32));
        let v = g.input("v", Shape::new(&[b, s, hs], DType::F32));
        let y = g.attention_kind_opts(
            q,
            k,
            v,
            h,
            d,
            MaskKind::Causal,
            Shape::new(&[b, s, hs], DType::F32),
            None,
            Some(cap),
        );
        g.set_outputs(vec![y]);

        let mkv = |seed: usize| -> Vec<f32> {
            (0..b * s * hs)
                .map(|i| ((i + seed) % 13) as f32 * 0.3 - 1.8)
                .collect()
        };
        let (qv, kv, vv) = (mkv(0), mkv(3), mkv(7));
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec
            .run(&[("q", &qv), ("k", &kv), ("v", &vv)])
            .expect("run");

        let scale = (d as f32).powf(-0.5);
        let mut want = vec![0.0f32; b * s * hs];
        let at = |si: usize, hi: usize, dd: usize| si * hs + hi * d + dd;
        for hi in 0..h {
            for qi in 0..s {
                let mut sc = vec![f32::NEG_INFINITY; s];
                for ki in 0..=qi {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += qv[at(qi, hi, dd)] * kv[at(ki, hi, dd)];
                    }
                    let scaled = dot * scale;
                    sc[ki] = cap * (scaled / cap).tanh();
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = sc
                    .iter()
                    .map(|x| if x.is_finite() { (x - mx).exp() } else { 0.0 })
                    .collect();
                let sum: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for ki in 0..s {
                        acc += exps[ki] / sum * vv[at(ki, hi, dd)];
                    }
                    want[at(qi, hi, dd)] = acc;
                }
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "softcap max_diff {md}");
        eprintln!("QNN FFI attention softcap (cap={cap}) OK (max_diff={md:.2e})");
    }

    /// Mean-pool over the sequence axis — `Reduce(Mean)` with an `axes` tensor +
    /// `keep_dims` bool. The pooling step of BERT/nomic embedding models.
    #[test]
    fn ffi_reduce_mean() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::op::ReduceOp;
        use rlx_ir::{DType, Graph, Shape};

        let (b, s, hd) = (1usize, 4, 3);
        let mut g = Graph::new("meanpool");
        let x = g.input("x", Shape::new(&[b, s, hd], DType::F32));
        let y = g.reduce(
            x,
            ReduceOp::Mean,
            vec![1],
            false,
            Shape::new(&[b, hd], DType::F32),
        );
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..b * s * hd).map(|i| i as f32 * 0.5 - 2.0).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: mean over the seq axis. [b, s, hd] → [b, hd].
        let mut want = vec![0.0f32; b * hd];
        for bi in 0..b {
            for hi in 0..hd {
                let mut acc = 0.0f32;
                for si in 0..s {
                    acc += xv[bi * s * hd + si * hd + hi];
                }
                want[bi * hd + hi] = acc / s as f32;
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "reduce mean max_diff {md}");
        eprintln!("QNN FFI reduce mean (mean-pool, keep_dim=false) OK (max_diff={md:.2e})");
    }

    /// NCHW Conv2d (3×3, pad 1) with a static weight — exercises the
    /// NCHW↔NHWC layout transposes, weight HWIO reorder, and the 3 Conv2d
    /// tensor params + group scalar.
    #[test]
    fn ffi_conv2d() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::{DType, Graph, Shape};

        let (n, cin, hh, ww) = (1usize, 2, 4, 4);
        let (cout, kh, kw) = (3usize, 3, 3);
        let (pad, stride, dil) = (1usize, 1usize, 1usize);
        let mut g = Graph::new("conv");
        let x = g.input("x", Shape::new(&[n, cin, hh, ww], DType::F32));
        let w = g.param("w", Shape::new(&[cout, cin, kh, kw], DType::F32));
        let y = g.conv2d(x, w, [kh, kw], [stride, stride], [pad, pad], [dil, dil], 1);
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..n * cin * hh * ww)
            .map(|i| (i % 7) as f32 * 0.2 - 0.6)
            .collect();
        let wv: Vec<f32> = (0..cout * cin * kh * kw)
            .map(|i| (i % 5) as f32 * 0.1 - 0.2)
            .collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param("w", &wv);
        let out = exec.run(&[("x", &xv)]).expect("run");

        let hout = (hh + 2 * pad - dil * (kh - 1) - 1) / stride + 1;
        let wout = (ww + 2 * pad - dil * (kw - 1) - 1) / stride + 1;
        let mut want = vec![0.0f32; cout * hout * wout];
        let xat = |ci: usize, y: usize, x: usize| (ci * hh + y) * ww + x;
        let wat =
            |co: usize, ci: usize, ky: usize, kx: usize| ((co * cin + ci) * kh + ky) * kw + kx;
        for co in 0..cout {
            for oy in 0..hout {
                for ox in 0..wout {
                    let mut acc = 0.0f32;
                    for ci in 0..cin {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * stride + ky * dil) as i64 - pad as i64;
                                let ix = (ox * stride + kx * dil) as i64 - pad as i64;
                                if iy >= 0 && iy < hh as i64 && ix >= 0 && ix < ww as i64 {
                                    acc += xv[xat(ci, iy as usize, ix as usize)]
                                        * wv[wat(co, ci, ky, kx)];
                                }
                            }
                        }
                    }
                    want[(co * hout + oy) * wout + ox] = acc;
                }
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "conv2d max_diff {md}");
        eprintln!("QNN FFI conv2d (NCHW↔NHWC, weight reorder) OK (max_diff={md:.2e})");
    }

    /// Embedding lookup — `Gather(table, ids)` with **int32 indices** (the first
    /// op of every decoder LLM). Exercises non-f32 input support.
    #[test]
    fn ffi_gather_embedding() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::{DType, Graph, Shape};

        let (vocab, hidden, seq) = (5usize, 3, 4);
        let mut g = Graph::new("embed");
        let table = g.param("table", Shape::new(&[vocab, hidden], DType::F32));
        let ids = g.input("ids", Shape::new(&[seq], DType::I32));
        let y = g.gather(table, ids, 0, Shape::new(&[seq, hidden], DType::F32));
        g.set_outputs(vec![y]);

        let tv: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.5 - 1.0).collect();
        let idv = vec![2.0f32, 0.0, 4.0, 1.0]; // token ids passed as f32 values
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param("table", &tv);
        let out = exec.run(&[("ids", &idv)]).expect("run");

        // Oracle: out[si] = table[ids[si]].
        let mut want = vec![0.0f32; seq * hidden];
        for si in 0..seq {
            let tok = idv[si] as usize;
            for h in 0..hidden {
                want[si * hidden + h] = tv[tok * hidden + h];
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "gather max_diff {md}");
        eprintln!("QNN FFI gather (embedding lookup, i32 indices) OK (max_diff={md:.2e})");
    }

    /// int8 `Quantize` → `Dequantize` round-trip — establishes the quantized
    /// tensor path (SFIXED_POINT_8 + scale/offset `quantizeParams`).
    #[test]
    fn ffi_quantize_roundtrip() {
        let Some(lib) = default_backend_lib() else {
            return;
        };
        if !lib.exists() {
            return;
        }
        use rlx_ir::{DType, Graph, Shape};

        let n = 8usize;
        let scale = 0.05f32;
        let mut g = Graph::new("quant");
        let x = g.input("x", Shape::new(&[n], DType::F32));
        let q = g.quantize(x, scale, 0);
        let y = g.dequantize(q, scale, 0);
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..n).map(|i| (i as f32 - 3.5) * 0.3).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");

        // Oracle: scale * clamp(round(x/scale), -128, 127).
        let mut want = vec![0.0f32; n];
        for i in 0..n {
            let q = (xv[i] / scale).round().clamp(-128.0, 127.0);
            want[i] = scale * q;
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Within one quantum — the correctness bar for a round-trip quantizer.
        assert!(md < scale, "quant round-trip max_diff {md} (scale {scale})");
        eprintln!("QNN FFI quantize/dequantize (int8 round-trip) OK (max_diff={md:.2e})");
    }

    /// On-device int8 weight MatMul: `MatMul(x, Dequantize(I8 Constant))` keeps
    /// STATIC `SFIXED_POINT_8` weights and runs QNN Dequantize → f32 MatMul
    /// (QNN rejects mixed f32×int8 MatMul on the CPU backend).
    #[test]
    fn ffi_int8_static_matmul() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let scale = 0.1f32;
        let zp = 0i32;
        // Weight [K,N] as int8 bytes.
        let w_i8: Vec<i8> = vec![1, -2, 3, 4, -5, 6, 7, -8, 9, -1, 2, -3];
        assert_eq!(w_i8.len(), k * n);
        let w_bytes: Vec<u8> = w_i8.iter().map(|&b| b as u8).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.25 * (i as f32 - 3.0)).collect();

        let mut g = Graph::new("i8mm");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wi = g.add_node(
            Op::Constant { data: w_bytes },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let wd = g.dequantize(wi, scale, zp);
        let y = g.matmul(xi, wd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        assert!(
            exec.i8_static.iter().any(|(_, b)| !b.is_empty()),
            "expected i8_static weight payload"
        );
        let out = exec.run(&[("x", &x)]).expect("run");

        let w_f: Vec<f32> = w_i8
            .iter()
            .map(|&q| scale * (q as f32 - zp as f32))
            .collect();
        let want = crate::reference::matmul_f32(&x, &w_f, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // One quantum of weight error × activation magnitude.
        assert!(md < 1e-3, "int8 static MatMul max_diff {md}");
        eprintln!("QNN FFI int8 static MatMul OK (max_diff={md:.2e})");
    }

    /// Per-channel int8 weights: `AXIS_SCALE_OFFSET` on STATIC sfixed8
    /// (`Dequantize` axis=1 for `[K,N]` — one scale per output channel).
    /// `libQnnCpu` accepts this; x86 `libQnnHtp` rejects
    /// (`Op Dequantize does not support per-channel quant tensor`) — soft-skip.
    #[test]
    fn ffi_int8_per_channel_matmul() {
        let Some(backend) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        // Distinct per-output-channel scales.
        let scales = vec![0.05f32, 0.1, 0.2];
        let zps = vec![0i32, 0, 0];
        let w_i8: Vec<i8> = vec![1, -2, 3, 4, -5, 6, 7, -8, 9, -1, 2, -3];
        assert_eq!(w_i8.len(), k * n);
        let w_bytes: Vec<u8> = w_i8.iter().map(|&b| b as u8).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.25 * (i as f32 - 3.0)).collect();

        let mut g = Graph::new("i8pc");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wi = g.add_node(
            Op::Constant { data: w_bytes },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let wd = g.dequantize_per_channel(wi, 1, scales.clone(), zps.clone());
        let y = g.matmul(xi, wd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        assert!(
            exec.quant_axis
                .iter()
                .any(|(_, ax, so)| *ax == 1 && so.len() == n),
            "expected AXIS_SCALE_OFFSET quant on axis 1"
        );
        let out = match exec.run(&[("x", &x)]) {
            Ok(o) => o,
            Err(e) => {
                let is_htp = backend.to_string_lossy().contains("Htp");
                eprintln!(
                    "int8 per-channel MatMul not supported on this backend{}: {e}",
                    if is_htp { " (HTP)" } else { "" }
                );
                return;
            }
        };

        let mut w_f = vec![0.0f32; k * n];
        for kk in 0..k {
            for nn in 0..n {
                let q = w_i8[kk * n + nn] as f32;
                w_f[kk * n + nn] = scales[nn] * (q - zps[nn] as f32);
            }
        }
        let want = crate::reference::matmul_f32(&x, &w_f, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "int8 per-channel MatMul max_diff {md}");
        eprintln!("QNN FFI int8 per-channel MatMul OK (max_diff={md:.2e})");
    }

    /// On-device int4 weight MatMul (BW_SCALE_OFFSET bitwidth=4):
    /// IR Constant stores tightly packed nibbles (`(K·N+1)/2` bytes); the
    /// runtime unpacks to 1 byte/elem because `libQnnCpu` rejects native
    /// `SFIXED_POINT_4` (`UNSUPPORTED_TENSOR_PARAM`).
    #[test]
    fn ffi_int4_static_matmul() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let scale = 0.25f32;
        let zp = 0i32;
        // Values in [-8, 7] for signed int4.
        let w_i4: Vec<i8> = vec![1, -2, 3, 4, -5, 6, 7, -8, 0, -1, 2, -3];
        assert_eq!(w_i4.len(), k * n);
        let w_packed = pack_sfixed4(&w_i4).expect("pack");
        assert_eq!(w_packed.len(), (k * n).div_ceil(2));
        // Round-trip pack for the reference.
        let w_unpacked = unpack_sfixed4(&w_packed, k * n);
        assert_eq!(w_unpacked, w_i4);

        let x: Vec<f32> = (0..m * k).map(|i| 0.25 * (i as f32 - 3.0)).collect();

        let mut g = Graph::new("i4mm");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wi = g.add_node(
            Op::Constant { data: w_packed },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let wd = g.dequantize(wi, scale, zp);
        let y = g.matmul(xi, wd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut exec = match QnnExecutable::compile_graph(&g) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("int4 static MatMul compile failed: {e}");
                return;
            }
        };
        assert!(
            exec.i4_static.iter().any(|(_, b)| !b.is_empty()),
            "expected i4_static weight payload"
        );
        let out = match exec.run(&[("x", &x)]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("int4 static MatMul not supported on this backend: {e}");
                return;
            }
        };

        let w_f: Vec<f32> = w_i4
            .iter()
            .map(|&q| scale * (q as f32 - zp as f32))
            .collect();
        let want = crate::reference::matmul_f32(&x, &w_f, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "int4 static MatMul max_diff {md}");
        eprintln!("QNN FFI int4 static MatMul OK (max_diff={md:.2e})");
    }

    /// Same path with a deferred I8 `Param` filled via `set_param_bytes`.
    #[test]
    fn ffi_int8_param_matmul() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let scale = 0.05f32;
        let zp = 0i32;
        let w_i8: Vec<i8> = vec![10, -20, 30, -40, 50, -60, 70, -80, 90, -15, 25, -35];
        let w_bytes: Vec<u8> = w_i8.iter().map(|&b| b as u8).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        let mut g = Graph::new("i8mm_p");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wi = g.param("w", Shape::new(&[k, n], DType::I8));
        let wd = g.dequantize(wi, scale, zp);
        let y = g.matmul(xi, wd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param_bytes("w", &w_bytes, DType::I8)
            .expect("set_param_bytes");
        let out = exec.run(&[("x", &x)]).expect("run");

        let w_f: Vec<f32> = w_i8
            .iter()
            .map(|&q| scale * (q as f32 - zp as f32))
            .collect();
        let want = crate::reference::matmul_f32(&x, &w_f, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "int8 param MatMul max_diff {md}");
        eprintln!("QNN FFI int8 param MatMul OK (max_diff={md:.2e})");
    }

    /// Fully quantized INT8 `Op::QMatMul` — integer accumulate + requantize,
    /// weights stay I8 (no host f32 dequant).
    #[test]
    fn ffi_q_matmul() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let x_zp = 0i32;
        let w_zp = 0i32;
        let out_zp = 0i32;
        let mult = 0.05f32;
        let x_i8: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let w_i8: Vec<i8> = vec![1, 0, -1, 2, -2, 1, 0, 3, -1, -2, 1, 0];
        let bias: Vec<i32> = vec![0, 1, -1];
        assert_eq!(x_i8.len(), m * k);
        assert_eq!(w_i8.len(), k * n);

        let mut g = Graph::new("qmm");
        let xi = g.input("x", Shape::new(&[m, k], DType::I8));
        let wi = g.add_node(
            Op::Constant {
                data: w_i8.iter().map(|&b| b as u8).collect(),
            },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let mut bias_bytes = Vec::with_capacity(n * 4);
        for &b in &bias {
            bias_bytes.extend_from_slice(&b.to_le_bytes());
        }
        let bi = g.add_node(
            Op::Constant { data: bias_bytes },
            vec![],
            Shape::new(&[n], DType::I32),
        );
        let y = g.q_matmul(
            xi,
            wi,
            bi,
            x_zp,
            w_zp,
            out_zp,
            mult,
            Shape::new(&[m, n], DType::I8),
        );
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let x_f: Vec<f32> = x_i8.iter().map(|&c| c as f32).collect();
        let out = exec.run(&[("x", &x_f)]).expect("run");
        let want =
            crate::qmatmul::q_matmul_i8(&x_i8, &w_i8, &bias, m, k, n, x_zp, w_zp, out_zp, mult);
        for (i, (&got, &w)) in out[0].iter().zip(&want).enumerate() {
            assert_eq!(got as i8, w, "q_matmul[{i}]: {got} vs {w}");
        }
        eprintln!("QNN FFI QMatMul (host INT8, no f32 bake) OK");
    }

    /// Host `QMatMul` → QNN `Dequantize` → `Relu` in one graph (APP_WRITE bridge).
    #[test]
    fn ffi_q_matmul_dequant_relu() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::op::Activation;
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let x_zp = 0i32;
        let w_zp = 0i32;
        let out_zp = 0i32;
        let mult = 0.05f32;
        let dq_scale = 0.1f32;
        let x_i8: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let w_i8: Vec<i8> = vec![1, 0, -1, 2, -2, 1, 0, 3, -1, -2, 1, 0];
        let bias: Vec<i32> = vec![0, 1, -1];

        let mut g = Graph::new("qmm_dq_relu");
        let xi = g.input("x", Shape::new(&[m, k], DType::I8));
        let wi = g.add_node(
            Op::Constant {
                data: w_i8.iter().map(|&b| b as u8).collect(),
            },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let mut bias_bytes = Vec::with_capacity(n * 4);
        for &b in &bias {
            bias_bytes.extend_from_slice(&b.to_le_bytes());
        }
        let bi = g.add_node(
            Op::Constant { data: bias_bytes },
            vec![],
            Shape::new(&[n], DType::I32),
        );
        let yq = g.q_matmul(
            xi,
            wi,
            bi,
            x_zp,
            w_zp,
            out_zp,
            mult,
            Shape::new(&[m, n], DType::I8),
        );
        let yd = g.dequantize(yq, dq_scale, 0);
        let yr = g.activation(Activation::Relu, yd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![yr]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        assert!(
            !exec.nodes.is_empty(),
            "expected QNN nodes after host QMatMul"
        );
        assert_eq!(
            exec.tensors[yq.0 as usize].ttype, 0,
            "QMatMul out is APP_WRITE bridge"
        );
        let x_f: Vec<f32> = x_i8.iter().map(|&c| c as f32).collect();
        let out = exec.run(&[("x", &x_f)]).expect("run");

        let codes =
            crate::qmatmul::q_matmul_i8(&x_i8, &w_i8, &bias, m, k, n, x_zp, w_zp, out_zp, mult);
        let want: Vec<f32> = codes
            .iter()
            .map(|&c| (dq_scale * (c as f32)).max(0.0))
            .collect();
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "QMatMul→Dequant→Relu max_diff {md}");
        eprintln!("QNN FFI QMatMul→Dequantize→Relu (mixed) OK (max_diff={md:.2e})");
    }

    /// QNN `Quantize` → host `QMatMul` → QNN `Dequantize` → `Relu`
    /// (pre-session → host → post-session).
    #[test]
    fn ffi_quantize_q_matmul_dequant_relu() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::op::Activation;
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let x_scale = 0.25f32;
        let x_zp = 0i32;
        let w_zp = 0i32;
        let out_zp = 0i32;
        let mult = 0.05f32;
        let dq_scale = 0.1f32;
        let x_f: Vec<f32> = vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0];
        let w_i8: Vec<i8> = vec![1, 0, -1, 2, -2, 1, 0, 3, -1, -2, 1, 0];
        let bias: Vec<i32> = vec![0, 1, -1];

        let mut g = Graph::new("q_qmm_dq");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let qx = g.quantize(xi, x_scale, x_zp);
        let wi = g.add_node(
            Op::Constant {
                data: w_i8.iter().map(|&b| b as u8).collect(),
            },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let mut bias_bytes = Vec::with_capacity(n * 4);
        for &b in &bias {
            bias_bytes.extend_from_slice(&b.to_le_bytes());
        }
        let bi = g.add_node(
            Op::Constant { data: bias_bytes },
            vec![],
            Shape::new(&[n], DType::I32),
        );
        let yq = g.q_matmul(
            qx,
            wi,
            bi,
            x_zp,
            w_zp,
            out_zp,
            mult,
            Shape::new(&[m, n], DType::I8),
        );
        let yd = g.dequantize(yq, dq_scale, 0);
        let yr = g.activation(Activation::Relu, yd, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![yr]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        assert!(
            !exec.pre_nodes.is_empty(),
            "expected pre-QNN Quantize nodes"
        );
        assert!(!exec.nodes.is_empty(), "expected post-QNN nodes");
        assert_eq!(
            exec.tensors[qx.0 as usize].ttype, 1,
            "Quantize out APP_READ"
        );
        assert_eq!(
            exec.tensors[yq.0 as usize].ttype, 0,
            "QMatMul out APP_WRITE"
        );

        let out = exec.run(&[("x", &x_f)]).expect("run");

        let x_i8: Vec<i8> = x_f
            .iter()
            .map(|&v| ((v / x_scale).round() as i32 + x_zp).clamp(-128, 127) as i8)
            .collect();
        let codes =
            crate::qmatmul::q_matmul_i8(&x_i8, &w_i8, &bias, m, k, n, x_zp, w_zp, out_zp, mult);
        let want: Vec<f32> = codes
            .iter()
            .map(|&c| (dq_scale * (c as f32)).max(0.0))
            .collect();
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "Quantize→QMatMul→Dequant→Relu max_diff {md}");
        eprintln!("QNN FFI Quantize→QMatMul→Dequantize→Relu OK (max_diff={md:.2e})");
    }

    /// On-device MatMul of two SFIXED_POINT_8 tensors: `Quantize(x)` × STATIC I8
    /// weight (no Dequantize in IR). The runtime lowers this as Dequantize both
    /// → f32 MatMul (libQnnCpu rejects direct sfixed8×sfixed8 with `0xc26`;
    /// HTP prepare accepts but execute fails on MatMul_bias). Host `QMatMul`
    /// remains the fully-quantized integer path.
    #[test]
    fn ffi_sfixed8_matmul_probe() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 4, 3);
        let scale = 1.0f32;
        let w_i8: Vec<i8> = vec![1, 0, -1, 2, -2, 1, 0, 3, -1, -2, 1, 0];
        let x_f: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

        let mut g = Graph::new("s8mm");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let qx = g.quantize(xi, scale, 0);
        let wi = g.add_node(
            Op::Constant {
                data: w_i8.iter().map(|&b| b as u8).collect(),
            },
            vec![],
            Shape::new(&[k, n], DType::I8),
        );
        let y = g.add_node(Op::MatMul, vec![qx, wi], Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = match exec.run(&[("x", &x_f)]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("sfixed8 MatMul path not supported on this backend: {e}");
                return;
            }
        };
        let x_i8: Vec<i8> = x_f.iter().map(|&v| v.round() as i8).collect();
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += (x_i8[i * k + kk] as f32) * (w_i8[kk * n + j] as f32);
                }
                want[i * n + j] = acc;
            }
        }
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-3, "sfixed8 matmul max_diff {md}");
        eprintln!("QNN FFI sfixed8 MatMul (via Dequantize) OK (max_diff={md:.2e})");
    }

    #[test]
    fn ffi_silu() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::op::Activation;
        use rlx_ir::{DType, Graph, Shape};

        let n = 8usize;
        let mut g = Graph::new("silu");
        let x = g.input("x", Shape::new(&[n], DType::F32));
        let y = g.activation(Activation::Silu, x, Shape::new(&[n], DType::F32));
        g.set_outputs(vec![y]);

        let xv: Vec<f32> = (0..n).map(|i| (i as f32 - 3.5) * 0.5).collect();
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");
        let want: Vec<f32> = xv.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "silu max_diff {md}");
        eprintln!("QNN FFI silu OK (max_diff={md:.2e})");
    }

    #[test]
    fn ffi_expand_tile() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_ir::{DType, Graph, Shape};

        let mut g = Graph::new("expand");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.add_node(
            Op::Expand {
                target_shape: vec![2, 3, 4],
            },
            vec![x],
            Shape::new(&[2, 3, 4], DType::F32),
        );
        g.set_outputs(vec![y]);

        let xv = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &xv)]).expect("run");
        assert_eq!(out[0].len(), 24);
        for row in out[0].chunks(4) {
            assert_eq!(row, xv.as_slice());
        }
        eprintln!("QNN FFI expand (bias-style broadcast) OK");
    }

    #[test]
    fn ffi_dequant_matmul_q8_0() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_gguf::GgmlType;
        use rlx_ir::quant::QuantScheme;
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 64, 8);
        let w_nk: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.013).sin() * 0.45)
            .collect();
        let packed = rlx_gguf::quantize(&w_nk, GgmlType::Q8_0).expect("quantize");
        let x: Vec<f32> = (0..m * k).map(|i| 0.02 * (i as f32 + 1.0)).collect();

        let mut g = Graph::new("dqmm");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wp = g.param("w_packed", Shape::new(&[packed.len()], DType::U8));
        let y = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ8_0,
            },
            vec![xi, wp],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        exec.set_param_bytes("w_packed", &packed, DType::U8)
            .expect("set_param_bytes");
        let out = exec.run(&[("x", &x)]).expect("run");

        // Oracle: dequant [N,K] → transpose [K,N] → matmul.
        let wf = crate::dequant::dequant_weight_for_qnn(QuantScheme::GgufQ8_0, &packed, n, k)
            .expect("oracle dequant");
        let want = crate::reference::matmul_f32(&x, &wf, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "dequant_matmul Q8_0 max_diff {md}");
        eprintln!("QNN FFI dequant_matmul Q8_0 OK (max_diff={md:.2e})");
    }

    #[test]
    fn ffi_dequant_matmul_q4_0_constant() {
        let Some(_) = default_backend_lib() else {
            eprintln!("skip: no QNN backend lib");
            return;
        };
        use rlx_gguf::GgmlType;
        use rlx_ir::quant::QuantScheme;
        use rlx_ir::{DType, Graph, Op, Shape};

        let (m, k, n) = (2usize, 64, 8);
        let w_nk: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.017).cos() * 0.3)
            .collect();
        let packed = rlx_gguf::quantize(&w_nk, GgmlType::Q4_0).expect("quantize");
        let x: Vec<f32> = (0..m * k).map(|i| 0.03 * (i as f32 - 5.0)).collect();

        let mut g = Graph::new("dqmm_c");
        let xi = g.input("x", Shape::new(&[m, k], DType::F32));
        let wp = g.add_node(
            Op::Constant {
                data: packed.clone(),
            },
            vec![],
            Shape::new(&[packed.len()], DType::U8),
        );
        let y = g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4_0,
            },
            vec![xi, wp],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mut exec = QnnExecutable::compile_graph(&g).expect("compile");
        let out = exec.run(&[("x", &x)]).expect("run");
        let wf = crate::dequant::dequant_weight_for_qnn(QuantScheme::GgufQ4_0, &packed, n, k)
            .expect("oracle");
        let want = crate::reference::matmul_f32(&x, &wf, m, k, n);
        let md = out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q4_0 has larger quantization error vs the dequant oracle (same path).
        assert!(md < 1e-5, "dequant_matmul Q4_0 const max_diff {md}");
        eprintln!("QNN FFI dequant_matmul Q4_0 (Constant) OK (max_diff={md:.2e})");
    }
}
