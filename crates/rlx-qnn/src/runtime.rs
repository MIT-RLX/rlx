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
//! backend) in Docker; the HTP/NPU path is the same code with `libQnnHtp.so` on
//! a Snapdragon device, and the context-binary perf path is the next milestone
//! (see `docs/ffi-runtime-backend.md`).
//!
//! Gated behind the `runtime` feature; `build.rs` compiles the shim against
//! `$QNN_SDK_ROOT/include/QNN`.

use std::ffi::{CString, c_char, c_float, c_int};
use std::path::{Path, PathBuf};

use rlx_ir::Op;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};

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
    /// 0 = float32, 1 = int32, 2 = sfixed8 (quantized int8).
    dtype: i32,
    /// Quantization scale / offset (dtype 2 only).
    q_scale: c_float,
    q_offset: i32,
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

    /// Build + execute an arbitrary supported subgraph (see the header).
    fn rlx_qnn_run_graph(
        backend_lib: *const c_char,
        tensors: *mut CTensor,
        num_tensors: u32,
        nodes: *const CNode,
        num_nodes: u32,
        err_out: *mut u64,
    ) -> c_int;
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
                _ => return None, // Min / Prod not wired
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
/// `Activation` (Relu/Gelu/Sigmoid/Tanh/Neg), `Reshape`, `Softmax`, `Transpose`,
/// `Narrow` (→ StridedSlice), `Concat`, `Gather` (int32 indices), `Reduce`
/// (Mean/Sum/Max), `LayerNorm`, `RmsNorm` (→ RmsNorm+Add),
/// `Rope` (NeoX, → 7-node slice/neg/concat/mul/add), `Attention` (MHA + GQA;
/// causal / sliding-window / none masks; optional logit softcap — decomposed to
/// head-split/scaled-qkᵀ/mask/softmax/·v/merge, KV heads expanded via
/// Reshape→Tile→Reshape), `Conv` (NCHW → NHWC Conv2d), and `Quantize`/
/// `Dequantize` (int8 SFIXED_POINT_8 + scale/offset), with `Param`/`Constant`
/// operands baked as static weights.
#[derive(Debug, Clone)]
pub struct QnnExecutable {
    backend_lib: CString,
    tensors: Vec<PlanTensor>,
    nodes: Vec<PlanNode>,
    /// rlx-ir input name → tensor index.
    inputs: Vec<(String, usize)>,
    /// rlx-ir param name → tensor index (static, filled by `set_param`).
    params: Vec<(String, usize)>,
    /// Output tensor indices, in graph-output order.
    outputs: Vec<usize>,
    /// Tensor indices whose QNN dtype is int32 (e.g. Gather indices). Inputs in
    /// this set are converted from the caller's f32 slice to i32 before exec.
    int_tensors: Vec<usize>,
    /// Quantized (sfixed8) tensors: `(tensor index, scale, offset)`.
    quant: Vec<(usize, f32, i32)>,
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
                    (0, None)
                }
                Op::Param { name } => {
                    params.push((name.clone(), i));
                    (4, None)
                }
                Op::Constant { data } => (4, Some(bytes_to_f32(data))),
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
                // NeoX RoPE has no native QNN op, so decompose (7 nodes):
                //   x1 = x[.., :d/2];  x2 = x[.., d/2:]
                //   rot = concat([-x2, x1], last)
                //   out = x*cos + rot*sin
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
                    let cos = node.inputs[1].0;
                    let sin = node.inputs[2].0;

                    let mut half_dims = dims.clone();
                    half_dims[last] = half;
                    let half_elems = half_dims.iter().map(|&d| d as usize).product::<usize>();

                    let base = (num_nodes + extra.len()) as u32;
                    let (x1i, x2i, negi, roti, ai, bi) =
                        (base, base + 1, base + 2, base + 3, base + 4, base + 5);
                    for (idx, d, ne) in [
                        (x1i, half_dims.clone(), half_elems),
                        (x2i, half_dims.clone(), half_elems),
                        (negi, half_dims.clone(), half_elems),
                        (roti, dims.clone(), num_elems),
                        (ai, dims.clone(), num_elems),
                        (bi, dims.clone(), num_elems),
                    ] {
                        extra.push(PlanTensor {
                            ttype: 3,
                            dims: d,
                            num_elems: ne,
                            data: None,
                            qnn_name: CString::new(format!("t{idx}_rope")).expect("nul"),
                        });
                    }

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
                // Scaled dot-product attention (3D [b, s, h*d], heads split
                // internally). Decomposes to ~13 QNN nodes. Scoped to MHA +
                // Causal/None mask, no softcap; anything else is a clear error.
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

        Ok(Self {
            backend_lib: CString::new(backend.to_string_lossy().into_owned()).expect("nul in path"),
            tensors,
            nodes,
            inputs,
            params,
            outputs: graph.outputs.iter().map(|id| id.0 as usize).collect(),
            int_tensors,
            quant,
        })
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
        }
    }

    /// Execute with named inputs; returns one buffer per graph output, in
    /// output order. Builds the QNN graph, finalizes, and runs in-process.
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        // Bind runtime input pointers by tensor index.
        let mut input_ptr: Vec<*mut c_float> = vec![std::ptr::null_mut(); self.tensors.len()];
        // i32 inputs (e.g. Gather indices) arrive as f32 values from the caller;
        // convert to i32 here, keeping the buffers alive across the FFI call.
        let mut int_bufs: Vec<Vec<i32>> = Vec::new();
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
            } else {
                input_ptr[idx] = data.as_ptr() as *mut c_float;
            }
        }

        // Output buffers, in graph-output order.
        let mut out_bufs: Vec<Vec<f32>> = self
            .outputs
            .iter()
            .map(|&i| vec![0.0f32; self.tensors[i].num_elems])
            .collect();

        // Build the C tensor array. Each pointer borrows `self`, `input_ptr`,
        // or `out_bufs` — all of which outlive the FFI call below.
        let mut ctensors: Vec<CTensor> = Vec::with_capacity(self.tensors.len());
        for (i, t) in self.tensors.iter().enumerate() {
            let data: *mut c_float = match t.ttype {
                4 => t
                    .data
                    .as_ref()
                    .map(|d| d.as_ptr() as *mut c_float)
                    .ok_or_else(|| format!("static tensor {i} has no data — call set_param"))?,
                0 => {
                    if input_ptr[i].is_null() {
                        return Err(format!("graph input at tensor {i} was not provided"));
                    }
                    input_ptr[i]
                }
                1 => {
                    let pos = self.outputs.iter().position(|&o| o == i).unwrap();
                    out_bufs[pos].as_mut_ptr()
                }
                _ => std::ptr::null_mut(),
            };
            let (dtype, q_scale, q_offset) =
                if let Some(&(_, s, o)) = self.quant.iter().find(|(qi, _, _)| *qi == i) {
                    (2, s, o)
                } else if self.int_tensors.contains(&i) {
                    (1, 0.0, 0)
                } else {
                    (0, 0.0, 0)
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
            });
        }

        let cnodes: Vec<CNode> = self
            .nodes
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
            .collect();

        let mut qnn_err: u64 = 0;
        // SAFETY: every pointer in `ctensors`/`cnodes` points into `self`,
        // `input_ptr`, or `out_bufs`, all live until after this call returns.
        let rc = unsafe {
            rlx_qnn_run_graph(
                self.backend_lib.as_ptr(),
                ctensors.as_mut_ptr(),
                ctensors.len() as u32,
                cnodes.as_ptr(),
                cnodes.len() as u32,
                &mut qnn_err,
            )
        };
        if rc != 0 {
            return Err(QnnError { step: -rc, qnn_err }.to_string());
        }
        Ok(out_bufs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
