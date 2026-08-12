// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

use std::collections::HashMap;

use rlx_ir::op::{CmpOp, ReduceOp, ScatterNdReduction};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Raw bytes + element type for a non-f32 (typically GGUF-quantized)
/// parameter, keyed by IR `Param` name.
pub type TypedParams = std::collections::HashMap<String, (Vec<u8>, DType)>;

use crate::proto;
use crate::{CoremlError, Result};

mod helpers;
pub(crate) use helpers::bytes_to_f32;
use helpers::simple_op_flex;
use helpers::*;

/// MIL opset / spec version targeted. opset `CoreML6` ⇒
/// `specificationVersion = 7` (macOS 13+). The host here is far newer, but
/// staying at CoreML6 keeps the base op set broadly available.
const OPSET: &str = "CoreML6";
const SPEC_VERSION: i32 = 7;
/// iOS18 (CoreML8 / spec 9) opset — required only when the graph uses ops added
/// there (e.g. `constexpr_lut_to_dense` grouped palettization + UINT1). Selected
/// per-model so ordinary graphs stay at the broadly-compatible CoreML6 baseline.
const OPSET_IOS18: &str = "CoreML8";
const SPEC_VERSION_IOS18: i32 = 9;

/// One model input/output, carried alongside the proto so the runtime
/// knows feature names + shapes without re-walking the graph.
#[derive(Debug, Clone)]
pub struct IoTensor {
    /// Original IR name (graph input name / synthetic output name).
    pub ir_name: String,
    /// Sanitised MIL/CoreML feature name actually used in the proto.
    pub feature_name: String,
    /// Static dimensions.
    pub dims: Vec<i64>,
    /// Element type.
    pub dtype: DType,
    /// Per-dimension flexibility for CoreML `ShapeRange` (model inputs).
    pub flex_dims: Vec<bool>,
}

impl IoTensor {
    /// Number of elements (product of dims); use [`runtime_dims`] when flex.
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<i64>().max(0) as usize
    }

    /// Resolve flexible dimensions from a concrete input buffer length.
    pub fn runtime_dims(&self, data_len: usize) -> Vec<i64> {
        if !self.flex_dims.iter().any(|&f| f) {
            return self.dims.clone();
        }
        let static_product: i64 = self
            .dims
            .iter()
            .zip(self.flex_dims.iter())
            .filter(|(_, flex)| !**flex)
            .map(|(d, _)| *d)
            .product();
        let mut dims = self.dims.clone();
        let denom = static_product.max(1);
        for (d, flex) in dims.iter_mut().zip(self.flex_dims.iter()) {
            if *flex {
                *d = data_len as i64 / denom;
            }
        }
        dims
    }

    pub fn runtime_numel(dims: &[i64]) -> usize {
        dims.iter().product::<i64>().max(0) as usize
    }
}

/// A fully lowered model plus its I/O manifest and weight blob.
pub struct LoweredProgram {
    pub model: proto::Model,
    pub inputs: Vec<IoTensor>,
    pub outputs: Vec<IoTensor>,
    /// `weight.bin` bytes (MILBlob format); empty if all consts are inline.
    pub blob: Vec<u8>,
}

/// Lowering knobs: numeric precision and optional flexible input shapes.
#[derive(Debug, Clone, Copy)]
pub struct LowerOptions {
    /// Float storage for activations/weights in MIL (F32 or F16).
    pub float_dtype: DType,
    /// Emit `UnknownDimension` + `ShapeRange` for `Dim::Dynamic` inputs.
    pub flexible_inputs: bool,
    /// Keep GGUF weights quantized in the model and dequant on device.
    pub ondevice_dequant: bool,
    /// Q1_0 on-device lowering mode. `None` → read `RLX_COREML_Q1_MODE` env
    /// (default **Lut** 1-bit palettization). `Some(Q1Mode::F32)` forces the
    /// legacy multi-GiB unfold. Lut keeps weights 1-bit packed in weight.bin
    /// (`constexpr_lut_to_dense`, iOS18 opset) — required for 27B-class models.
    pub q1_mode: Option<quant::Q1Mode>,
}

impl Default for LowerOptions {
    fn default() -> Self {
        Self {
            float_dtype: DType::F32,
            flexible_inputs: false,
            ondevice_dequant: true,
            q1_mode: None,
        }
    }
}

/// CoreML model I/O features are MLMultiArrays, which require rank ≥ 1. Map a
/// rank-0 (scalar) IR shape to `[1]` for the model interface; rank ≥ 1 passes
/// through unchanged. Used only at the input/output boundary — internal MIL
/// values keep their true (possibly rank-0) shape, which the program supports.
/// Scalars show up in training graphs (the loss and its `d_output` cotangent).
fn io_feature_shape(shape: &Shape) -> Shape {
    if shape.rank() == 0 {
        Shape::new(&[1], shape.dtype())
    } else {
        shape.clone()
    }
}

/// Lower `graph` to a CoreML ML Program. `params` maps IR `Param` names to
/// their f32 weights; `typed_params` carries non-f32 (GGUF-quantized)
/// weights as raw bytes. CoreML bakes weights into the model at build
/// time, so every `Param` referenced by the graph must appear in one of
/// the two maps (quantized weights are host-dequantized to f32 here).
pub fn lower_graph(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    typed_params: &TypedParams,
) -> Result<LoweredProgram> {
    lower_graph_with_options(graph, params, typed_params, &LowerOptions::default())
}

pub fn lower_graph_with_options(
    graph: &Graph,
    params: &HashMap<String, Vec<f32>>,
    typed_params: &TypedParams,
    opts: &LowerOptions,
) -> Result<LoweredProgram> {
    let mut ctx = LowerCtx::new(graph, params, typed_params, *opts);
    ctx.run()?;
    ctx.finish()
}

/// Per-node value name + the proto pieces accumulated during the walk.
struct LowerCtx<'a> {
    graph: &'a Graph,
    params: &'a HashMap<String, Vec<f32>>,
    typed_params: &'a TypedParams,
    opts: LowerOptions,
    /// NodeId → MIL value name.
    names: HashMap<u32, String>,
    func_inputs: Vec<proto::NamedValueType>,
    operations: Vec<proto::Operation>,
    inputs: Vec<IoTensor>,
    used_feature_names: HashMap<String, u32>,
    /// Bool-node id → the name of its already-emitted `{name}_f32m` cast, so a
    /// bool tensor reused as a numeric operand (VITS masks feed many ops) is
    /// cast once — re-emitting would redefine the MIL I/O name.
    numeric_casts: HashMap<u32, String>,
    blob: crate::mlpackage::BlobWriter,
    /// Prefix for generated `v{id}` value names. Empty at the top level; set to a
    /// scan-unique string while lowering an `Op::Scan` body into a nested
    /// `while_loop` block, so body node names don't collide with the outer graph's
    /// `v{id}` names that the block references via closure.
    name_prefix: String,
}

mod activation;
mod attention;
/// Interleaved-C64 Wirtinger + FFT butterfly MIL lowers.
mod complex;
mod conv_pool;
mod loss;
mod matmul;
mod norm;
mod quant;
pub use quant::Q1Mode;
mod reduce_index;
mod rnn;
mod rope;
mod ssm;

impl<'a> LowerCtx<'a> {
    pub(crate) fn new(
        graph: &'a Graph,
        params: &'a HashMap<String, Vec<f32>>,
        typed_params: &'a TypedParams,
        opts: LowerOptions,
    ) -> Self {
        LowerCtx {
            graph,
            params,
            typed_params,
            opts,
            names: HashMap::new(),
            func_inputs: Vec::new(),
            operations: Vec::new(),
            inputs: Vec::new(),
            used_feature_names: HashMap::new(),
            numeric_casts: HashMap::new(),
            blob: crate::mlpackage::BlobWriter::new(),
            name_prefix: String::new(),
        }
    }

    pub(crate) fn val(&self, id: NodeId) -> String {
        self.names
            .get(&id.0)
            .cloned()
            .unwrap_or_else(|| format!("v{}", id.0))
    }

    /// Like [`Self::val`], but coerces a bool operand to fp32 first. CoreML's
    /// arithmetic ops (mul/add/…) reject bool tensors, but VITS multiplies
    /// activations by bool masks; cast bool → fp32 (true→1.0, false→0.0).
    pub(crate) fn val_numeric(&mut self, id: NodeId) -> Result<String> {
        let name = self.val(id);
        if self.graph.shape(id).dtype() == DType::Bool {
            // Emit the bool→f32 cast at most once per node; a mask reused across
            // several ops would otherwise redefine `{name}_f32m` (MIL rejects a
            // block that declares the same I/O name twice).
            if let Some(existing) = self.numeric_casts.get(&id.0) {
                return Ok(existing.clone());
            }
            let cast_name = format!("{name}_f32m");
            let shape = self.graph.shape(id).clone().with_dtype(DType::F32);
            self.emit(
                "cast",
                &cast_name,
                &shape,
                vec![
                    ("x", bind_name(&name)),
                    ("dtype", bind_value(scalar_str("fp32"))),
                ],
            )?;
            self.numeric_casts.insert(id.0, cast_name.clone());
            Ok(cast_name)
        } else {
            Ok(name)
        }
    }

    /// Walk the graph in topo order, emitting one MIL op per node.
    pub(crate) fn run(&mut self) -> Result<()> {
        for id in self.graph.topo_order() {
            self.lower_node(id)?;
        }
        Ok(())
    }

    pub(crate) fn lower_node(&mut self, id: NodeId) -> Result<()> {
        let node = self.graph.node(id);
        let out_name = format!("{}v{}", self.name_prefix, id.0);
        match &node.op {
            Op::Input { name } => {
                let feat = self.unique_feature_name(name);
                // The runtime feeds all inputs as f32 (the FFI takes `&[f32]`), and
                // CoreML I/O has no I64 type, so declare integer/bool inputs as F32.
                // Int-consuming ops (e.g. Gather indices) cast back as needed.
                let io_dtype = if node.shape.dtype().is_float() {
                    if self.opts.float_dtype == DType::F16 {
                        DType::F16
                    } else {
                        node.shape.dtype()
                    }
                } else {
                    DType::F32
                };
                // CoreML model features (an MLMultiArray) need rank ≥ 1, but a
                // scalar-loss gradient seed (`d_output`) and other training
                // cotangents arrive rank-0. Declare them as `[1]` at the
                // interface; the value broadcasts like a scalar for every
                // elementwise/reduce consumer downstream.
                let io_shape = io_feature_shape(&node.shape).with_dtype(io_dtype);
                let (dims, flex_dims) = io_dims(&io_shape, self.opts.flexible_inputs)?;
                self.func_inputs
                    .push(named_value_type_flex(&feat, &io_shape, &flex_dims)?);
                self.inputs.push(IoTensor {
                    ir_name: name.clone(),
                    feature_name: feat.clone(),
                    dims,
                    dtype: io_dtype,
                    flex_dims,
                });
                self.names.insert(id.0, feat);
            }
            Op::Param { name } => {
                if let Some(data) = self.params.get(name) {
                    let shape = if self.opts.float_dtype == DType::F16
                        && node.shape.dtype() == DType::F32
                    {
                        node.shape.clone().with_dtype(DType::F16)
                    } else {
                        node.shape.clone()
                    };
                    let op = make_const_float(
                        &mut self.blob,
                        &out_name,
                        &shape,
                        data,
                        self.opts.float_dtype,
                    )?;
                    self.operations.push(op);
                    self.names.insert(id.0, out_name);
                } else if let Some((bytes, dtype)) = self.typed_params.get(name) {
                    // Quantized weight bytes (I8/U8 packing) are host-dequantized
                    // by the consuming Dequant* op — emit nothing here.
                    // Integer *control* params (I64/I32 duration carry, alignment
                    // frame count, runtime shapes) must still materialize: CoreML
                    // has no I64 storage, so bake them as f32 consts (same
                    // convention as promote_int_to_f32 / f32-uniform arenas).
                    match *dtype {
                        DType::I8 | DType::U8 => {
                            // packed quant weight — Dequant* bakes the f32 const
                        }
                        DType::I64
                        | DType::I32
                        | DType::U32
                        | DType::I16
                        | DType::F32
                        | DType::F16
                        | DType::BF16
                        | DType::Bool => {
                            let floats =
                                bytes_to_f32(bytes, &node.shape.clone().with_dtype(*dtype))?;
                            let shape = node.shape.clone().with_dtype(DType::F32);
                            let op = make_const_float(
                                &mut self.blob,
                                &out_name,
                                &shape,
                                &floats,
                                self.opts.float_dtype,
                            )?;
                            self.operations.push(op);
                            self.names.insert(id.0, out_name);
                        }
                        other => {
                            return Err(CoremlError::Unsupported(format!(
                                "CoreML typed param '{name}' dtype {other:?}"
                            )));
                        }
                    }
                } else {
                    return Err(CoremlError::Runtime(format!(
                        "missing baked param '{name}' for CoreML"
                    )));
                }
            }
            Op::Constant { data } => {
                let floats = bytes_to_f32(data, &node.shape)?;
                // Integer constants are baked as f32 (CoreML has no int blob storage);
                // declare the const as F32 so downstream f32 ops accept it. Bool
                // constants stay bool (make_const bakes them as inline bool immediates
                // for `select` conds).
                let cshape = match node.shape.dtype() {
                    DType::Bool | DType::F32 => node.shape.clone(),
                    _ => node.shape.clone().with_dtype(DType::F32),
                };
                let op = make_const(&mut self.blob, &out_name, &cshape, &floats)?;
                self.operations.push(op);
                self.names.insert(id.0, out_name);
            }
            Op::MatMul => {
                let x = self.val(node.inputs[0]);
                let y = self.val(node.inputs[1]);
                let op = self.simple_op(
                    "matmul",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("y", bind_name(&y)),
                        ("transpose_x", bind_value(scalar_bool(false))),
                        ("transpose_y", bind_value(scalar_bool(false))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Binary(b) => {
                let ty = binary_mil(*b);
                let x = self.val_numeric(node.inputs[0])?;
                let y = self.val_numeric(node.inputs[1])?;
                let op = self.simple_op(
                    ty,
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&y))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Fma => {
                self.lower_fma(id, &out_name)?;
            }
            Op::Activation(act) => {
                self.lower_activation(id, *act, &out_name)?;
            }
            Op::Softmax { axis } => {
                let x = self.val(node.inputs[0]);
                let op = self.simple_op(
                    "softmax",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("axis", bind_value(scalar_i32(*axis))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Reshape { new_shape } => {
                let x = self.val(node.inputs[0]);
                let shp: Vec<i32> = new_shape.iter().map(|&d| d as i32).collect();
                let op = self.simple_op(
                    "reshape",
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("shape", bind_value(vec_i32(&shp)))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Transpose { perm } => {
                let x = self.val(node.inputs[0]);
                let p: Vec<i32> = perm.iter().map(|&d| d as i32).collect();
                let op = self.simple_op(
                    "transpose",
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("perm", bind_value(vec_i32(&p)))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::LayerNorm { axis, eps } => {
                self.lower_layer_norm(id, *axis, *eps, &out_name)?;
            }
            Op::RmsNorm { axis, eps } => {
                self.lower_rms_norm(id, *axis, *eps, &out_name)?;
            }
            Op::AdaLayerNorm { norm, eps } => {
                self.lower_ada_layer_norm(id, *norm, *eps, &out_name)?;
            }
            Op::GatedResidual => {
                self.lower_gated_residual(id, &out_name)?;
            }
            // Native MIL backward kernels (training). Tighter than the autodiff
            // decomposition (implicit broadcasting in place of `Expand`-with-ones);
            // mirror `rlx_autodiff::*::compose_rms_norm_backward_*` exactly so ANE
            // gradients stay consistent with the other backends' training path.
            #[cfg(feature = "training")]
            Op::RmsNormBackwardInput { axis, eps } => {
                self.lower_rms_norm_backward_input(id, *axis, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::RmsNormBackwardGamma { axis, eps } => {
                self.lower_rms_norm_backward_gamma(id, *axis, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::RmsNormBackwardBeta { axis, eps } => {
                self.lower_rms_norm_backward_beta(id, *axis, *eps, &out_name)?;
            }
            // LayerNorm backward input + gamma — the mean-subtracting sibling of the
            // RMSNorm kernels. Native composed MIL (~18 ops) vs the decomposition's
            // expand-heavy graph; beta backward stays on decompose (just a reduce_sum).
            #[cfg(feature = "training")]
            Op::LayerNormBackwardInput { axis, eps } => {
                self.lower_layer_norm_backward_input(id, *axis, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::LayerNormBackwardGamma { axis, eps } => {
                self.lower_layer_norm_backward_gamma(id, *axis, *eps, &out_name)?;
            }
            // Packed DiT reverse — native composed MIL (implicit broadcast +
            // concat pack). Graduates from the decompose-route claim.
            #[cfg(feature = "training")]
            Op::AdaLayerNormBackward { norm, eps } => {
                self.lower_ada_layer_norm_backward(id, *norm, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::GatedResidualBackward => {
                self.lower_gated_residual_backward(id, &out_name)?;
            }
            // GroupNorm backward (NCHW). Native composed MIL: reshape [N,C,H,W] →
            // [N,G,M] (M = C/G·H·W) so each group's stats are one last-axis reduce —
            // no per-group narrow/concat loop (the decompose builds O(num_groups)
            // narrows + a concat).
            #[cfg(feature = "training")]
            Op::GroupNormBackwardInput { num_groups, eps } => {
                self.lower_group_norm_backward_input(id, *num_groups, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::GroupNormBackwardGamma { num_groups, eps } => {
                self.lower_group_norm_backward_gamma(id, *num_groups, *eps, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::GroupNormBackwardBeta { .. } => {
                self.lower_group_norm_backward_beta(id, &out_name)?;
            }
            // Fused attention backward (dQ/dK/dV) for the canonical [B,H,S,D] +
            // None/Causal training path. Other layouts / masks return Unsupported
            // (same common-case-native precedent as MaxPool2dBackward).
            #[cfg(feature = "training")]
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => {
                self.lower_attention_backward(
                    id, *num_heads, *head_dim, *mask_kind, *wrt, &out_name,
                )?;
            }
            #[cfg(feature = "training")]
            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => {
                self.lower_max_pool2d_backward(id, kernel_size, stride, padding, &out_name)?;
            }
            // Conv2d backward w.r.t. input = transposed convolution of the upstream
            // gradient with the forward weight (the conv adjoint). Inputs [dy, w];
            // output = the original input shape. Native because the autodiff
            // decomposition emits a plain `conv` (wrong gradient — and CoreML
            // rejects its channel layout); `conv_transpose` is the correct adjoint.
            #[cfg(feature = "training")]
            Op::Conv2dBackwardInput {
                stride,
                padding,
                dilation,
                groups,
                ..
            } => {
                self.lower_conv(
                    id,
                    true,
                    stride,
                    padding,
                    dilation,
                    &[0, 0],
                    *groups,
                    &out_name,
                )?;
            }
            // Conv2d backward w.r.t. weight = convolution of the input with the
            // upstream gradient: dW = transpose(conv(xᵀ, dyᵀ, dilation=stride)).
            // Inputs [x, dy]; output = forward weight shape [Cout,Cin,kh,kw].
            #[cfg(feature = "training")]
            Op::Conv2dBackwardWeight {
                stride,
                padding,
                groups,
                ..
            } => {
                self.lower_conv2d_backward_weight(id, stride, padding, *groups, &out_name)?;
            }
            // Softmax-cross-entropy forward (integer labels) + backward, both native
            // for the same reason: the decompose builds the one-hot by concatenating
            // C class columns — O(C) graph ops that explode at LLM vocab sizes. MIL's
            // `one_hot` op is a single node. Lowering BOTH keeps the loss op out of
            // the `bad` set so the shared `LowerSoftmaxCrossEntropy` pass never fires
            // and re-decomposes the backward.
            #[cfg(feature = "training")]
            Op::SoftmaxCrossEntropyWithLogits => {
                self.lower_softmax_cross_entropy_with_logits(id, &out_name)?;
            }
            #[cfg(feature = "training")]
            Op::SoftmaxCrossEntropyBackward => {
                self.lower_softmax_cross_entropy_backward(id, &out_name)?;
            }
            Op::Reduce { op, axes, keep_dim } => {
                let ty = match op {
                    ReduceOp::Sum => "reduce_sum",
                    ReduceOp::Mean => "reduce_mean",
                    ReduceOp::Max => "reduce_max",
                    ReduceOp::Min => "reduce_min",
                    ReduceOp::Prod => "reduce_prod",
                };
                let x = self.val(node.inputs[0]);
                let ax: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
                let op = self.simple_op(
                    ty,
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("axes", bind_value(vec_i32(&ax))),
                        ("keep_dims", bind_value(scalar_bool(*keep_dim))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Concat { axis } => {
                let names: Vec<String> = node.inputs.iter().map(|&i| self.val(i)).collect();
                let op = self.simple_op(
                    "concat",
                    &out_name,
                    &node.shape,
                    vec![
                        ("values", bind_names(&names)),
                        ("axis", bind_value(scalar_i32(*axis as i32))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Gather { axis } => {
                let x = self.val(node.inputs[0]);
                // CoreML `gather` needs integer indices. In this f32-flow graph the
                // indices are f32-encoded (ids/shape values baked or fed as f32, even
                // where the IR dtype still says I64), so always cast to int32.
                let idx_id = node.inputs[1];
                let ic = format!("{out_name}_idx_i32");
                let ishape = self.graph.shape(idx_id).clone().with_dtype(DType::I32);
                self.emit(
                    "cast",
                    &ic,
                    &ishape,
                    vec![
                        ("x", bind_name(&self.val(idx_id))),
                        ("dtype", bind_value(scalar_str("int32"))),
                    ],
                )?;
                let idx = ic;
                let op = self.simple_op(
                    "gather",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("indices", bind_name(&idx)),
                        ("axis", bind_value(scalar_i32(*axis as i32))),
                        // Required by the iOS18 gather opset (LUT path forces spec-9).
                        ("validate_indices", bind_value(scalar_bool(false))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Narrow { axis, start, len } => {
                let x = self.val(node.inputs[0]);
                let rank = node.shape.rank();
                let mut begin = vec![0i32; rank];
                // Use EXPLICIT per-axis sizes from the (static) output shape rather
                // than the `-1` "to-end" sentinel: CoreML/ANE converts a `-1` size
                // into an `end_ids` computation that can fail on the Neural Engine
                // ("generic_general_slice: Invalid values in end_ids"). The output
                // shape already carries `len` at `axis` and the full extent
                // elsewhere, so this is identical but ANE-safe.
                let mut size: Vec<i32> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i32)
                    .collect();
                begin[*axis] = *start as i32;
                size[*axis] = *len as i32;
                let op = self.simple_op(
                    "slice_by_size",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("begin", bind_value(vec_i32(&begin))),
                        ("size", bind_value(vec_i32(&size))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Rope {
                head_dim, n_rot, ..
            } => {
                self.lower_rope(id, *head_dim, *n_rot, &out_name)?;
            }
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => {
                assert!(
                    v_head_dim.is_none_or(|v| v == *head_dim),
                    "rlx-coreml: asymmetric v_head_dim (MLA) not yet supported"
                );
                self.lower_attention(
                    id,
                    *num_heads,
                    *head_dim,
                    *mask_kind,
                    *score_scale,
                    *attn_logit_softcap,
                    &out_name,
                )?;
            }
            Op::Cast { to } => {
                let x = self.val(node.inputs[0]);
                let dt = mil_cast_dtype(*to)?;
                let op = self.simple_op(
                    "cast",
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("dtype", bind_value(scalar_str(dt)))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Compare(cmp) => {
                let ty = match cmp {
                    CmpOp::Eq => "equal",
                    CmpOp::Ne => "not_equal",
                    CmpOp::Lt => "less",
                    CmpOp::Le => "less_equal",
                    CmpOp::Gt => "greater",
                    CmpOp::Ge => "greater_equal",
                };
                // MIL's comparison operators reject `bool` operands (only
                // int32/fp32/fp16). The importer lowers ONNX `Not` to
                // `Equal(bool_x, 0)`, so cast any bool operand up to int32 first.
                let mut x = self.val(node.inputs[0]);
                let mut y = self.val(node.inputs[1]);
                for (slot, inp, name_ref) in [
                    (0usize, node.inputs[0], &mut x),
                    (1usize, node.inputs[1], &mut y),
                ] {
                    let _ = slot;
                    if self.graph.shape(inp).dtype() == DType::Bool {
                        let cb = format!("{out_name}_i{}", inp.0);
                        self.emit(
                            "cast",
                            &cb,
                            &self.graph.shape(inp).clone().with_dtype(DType::I32),
                            vec![
                                ("x", bind_name(name_ref)),
                                ("dtype", bind_value(scalar_str("int32"))),
                            ],
                        )?;
                        *name_ref = cb;
                    }
                }
                let op = self.simple_op(
                    ty,
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&y))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Where => {
                // MIL `select` needs a bool cond; the CPU reference treats
                // cond as `> 0.5`, so coerce non-bool conds that way.
                let cond_in = node.inputs[0];
                let cond_shape = self.graph.shape(cond_in).clone();
                let mut cond = self.val(cond_in);
                if cond_shape.dtype() != DType::Bool {
                    let cb = format!("{out_name}_condb");
                    self.emit(
                        "greater",
                        &cb,
                        &cond_shape.with_dtype(DType::Bool),
                        vec![("x", bind_name(&cond)), ("y", bind_value(scalar_f32(0.5)))],
                    )?;
                    cond = cb;
                }
                let a = self.val(node.inputs[1]);
                let b = self.val(node.inputs[2]);
                let op = self.simple_op(
                    "select",
                    &out_name,
                    &node.shape,
                    vec![
                        ("cond", bind_name(&cond)),
                        ("a", bind_name(&a)),
                        ("b", bind_name(&b)),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Expand { .. } => {
                // Materialise a numpy-style broadcast as `x * ones(target)`.
                let n = node.shape.num_elements().unwrap_or(0);
                if node.shape.dtype() == DType::Bool {
                    // CoreML's mul rejects bool; broadcast in f32 then cast back to
                    // bool so a downstream `select` still receives a real bool cond.
                    let xf = self.val_numeric(node.inputs[0])?; // bool -> f32
                    let f32_shape = node.shape.clone().with_dtype(DType::F32);
                    let ones = format!("{out_name}_ones");
                    self.operations.push(make_const(
                        &mut self.blob,
                        &ones,
                        &f32_shape,
                        &vec![1.0f32; n],
                    )?);
                    let bf = format!("{out_name}_bf");
                    self.emit(
                        "mul",
                        &bf,
                        &f32_shape,
                        vec![("x", bind_name(&xf)), ("y", bind_name(&ones))],
                    )?;
                    let op = self.simple_op(
                        "cast",
                        &out_name,
                        &node.shape,
                        vec![
                            ("x", bind_name(&bf)),
                            ("dtype", bind_value(scalar_str("bool"))),
                        ],
                    )?;
                    self.push_named(id, out_name, op);
                } else {
                    // Non-float Expands (e.g. i64 shape broadcasts) flow as f32 in
                    // CoreML — declare ones + output f32 so no int blob is baked.
                    let oshape = if node.shape.dtype().is_float() {
                        node.shape.clone()
                    } else {
                        node.shape.clone().with_dtype(DType::F32)
                    };
                    let x = self.val(node.inputs[0]);
                    let ones = format!("{out_name}_ones");
                    self.operations.push(make_const(
                        &mut self.blob,
                        &ones,
                        &oshape,
                        &vec![1.0f32; n],
                    )?);
                    let op = self.simple_op(
                        "mul",
                        &out_name,
                        &oshape,
                        vec![("x", bind_name(&x)), ("y", bind_name(&ones))],
                    )?;
                    self.push_named(id, out_name, op);
                }
            }
            Op::Cumsum { axis, exclusive } => {
                let x = self.val(node.inputs[0]);
                let op = self.simple_op(
                    "cumsum",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("axis", bind_value(scalar_i32(*axis))),
                        ("exclusive", bind_value(scalar_bool(*exclusive))),
                        ("reverse", bind_value(scalar_bool(false))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::ScatterAdd => {
                // out = scatter(zeros, indices, updates, axis=0, mode=add).
                let updates = self.val(node.inputs[0]);
                let idx_in = node.inputs[1];
                let idx = self.val(idx_in);
                let idx_i32 = format!("{out_name}_idx");
                let idx_shape = self.graph.shape(idx_in).clone().with_dtype(DType::I32);
                self.emit(
                    "cast",
                    &idx_i32,
                    &idx_shape,
                    vec![
                        ("x", bind_name(&idx)),
                        ("dtype", bind_value(scalar_str("int32"))),
                    ],
                )?;
                let zeros = format!("{out_name}_zeros");
                let n = node.shape.num_elements().unwrap_or(0);
                self.operations.push(make_const(
                    &mut self.blob,
                    &zeros,
                    &node.shape,
                    &vec![0.0f32; n],
                )?);
                let op = self.simple_op(
                    "scatter",
                    &out_name,
                    &node.shape,
                    vec![
                        ("data", bind_name(&zeros)),
                        ("indices", bind_name(&idx_i32)),
                        ("updates", bind_name(&updates)),
                        ("axis", bind_value(scalar_i32(0))),
                        ("mode", bind_value(scalar_str("add"))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::ScatterNd { reduction } => {
                let data = self.val(node.inputs[0]);
                let idx_id = node.inputs[1];
                let updates = self.val(node.inputs[2]);
                let idx_i32 = format!("{out_name}_idx");
                let idx_shape = self.graph.shape(idx_id).clone().with_dtype(DType::I32);
                self.emit(
                    "cast",
                    &idx_i32,
                    &idx_shape,
                    vec![
                        ("x", bind_name(&self.val(idx_id))),
                        ("dtype", bind_value(scalar_str("int32"))),
                    ],
                )?;
                let mode = match reduction {
                    ScatterNdReduction::Add => "add",
                    ScatterNdReduction::Mul => "mul",
                    ScatterNdReduction::Max => "max",
                    ScatterNdReduction::Min => "min",
                    ScatterNdReduction::None => "update",
                };
                let op = self.simple_op(
                    "scatter_nd",
                    &out_name,
                    &node.shape,
                    vec![
                        ("data", bind_name(&data)),
                        ("indices", bind_name(&idx_i32)),
                        ("updates", bind_name(&updates)),
                        ("mode", bind_value(scalar_str(mode))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            // Legacy Custom alias — keep for graphs that still emit onnx.ScatterND.
            Op::Custom {
                name,
                attrs,
                num_inputs: 3,
            } if name == "onnx.ScatterND" => {
                let data = self.val(node.inputs[0]);
                let idx_id = node.inputs[1];
                let updates = self.val(node.inputs[2]);
                let idx_i32 = format!("{out_name}_idx");
                let idx_shape = self.graph.shape(idx_id).clone().with_dtype(DType::I32);
                self.emit(
                    "cast",
                    &idx_i32,
                    &idx_shape,
                    vec![
                        ("x", bind_name(&self.val(idx_id))),
                        ("dtype", bind_value(scalar_str("int32"))),
                    ],
                )?;
                let reduction = if attrs.len() >= 4 {
                    i32::from_le_bytes([attrs[0], attrs[1], attrs[2], attrs[3]])
                } else {
                    0
                };
                let mode = match reduction {
                    1 => "add",
                    2 => "mul",
                    3 => "max",
                    4 => "min",
                    _ => "update",
                };
                let op = self.simple_op(
                    "scatter_nd",
                    &out_name,
                    &node.shape,
                    vec![
                        ("data", bind_name(&data)),
                        ("indices", bind_name(&idx_i32)),
                        ("updates", bind_name(&updates)),
                        ("mode", bind_value(scalar_str(mode))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::BatchNormInference { eps } => {
                self.lower_batch_norm(id, *eps, &out_name)?;
            }
            Op::GroupNorm { num_groups, eps } => {
                self.lower_group_norm(id, *num_groups, *eps, &out_name)?;
            }
            Op::LayerNorm2d { eps } => {
                self.lower_layer_norm2d(id, *eps, &out_name)?;
            }
            Op::LoraMatMul { scale } => {
                self.lower_lora_matmul(id, *scale, &out_name)?;
            }
            Op::Conv {
                kernel_size: _,
                stride,
                padding,
                dilation,
                groups,
            } => {
                self.lower_conv(
                    id,
                    false,
                    stride,
                    padding,
                    dilation,
                    &[],
                    *groups,
                    &out_name,
                )?;
            }
            Op::ConvTranspose2d {
                kernel_size: _,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            } => {
                self.lower_conv(
                    id,
                    true,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    *groups,
                    &out_name,
                )?;
            }
            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            } => {
                self.lower_conv(
                    id,
                    false,
                    stride,
                    padding,
                    dilation,
                    &[],
                    *groups,
                    &out_name,
                )?;
            }
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            } => {
                self.lower_conv(
                    id,
                    true,
                    stride,
                    padding,
                    dilation,
                    output_padding,
                    *groups,
                    &out_name,
                )?;
            }
            Op::FusedMatMulBiasAct { activation } => {
                self.lower_fused_matmul_bias_act(id, *activation, &out_name)?;
            }
            Op::FusedSwiGLU { .. } => {
                self.lower_fused_swiglu(id, &out_name)?;
            }
            Op::FusedResidualLN { has_bias, eps } => {
                self.lower_fused_residual_ln(id, *has_bias, *eps, &out_name)?;
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                self.lower_fused_residual_rms_norm(id, *has_bias, *eps, &out_name)?;
            }
            Op::FakeQuantize {
                bits,
                axis,
                ste: _,
                scale_mode,
            } => {
                self.lower_fake_quantize(id, *bits, *axis, *scale_mode, &out_name)?;
            }
            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                self.lower_pool(id, *kind, kernel_size, stride, padding, &out_name)?;
            }
            Op::TopK { k } => {
                self.lower_topk(id, *k, &out_name)?;
            }
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => {
                self.lower_axial_rope2d(
                    id,
                    *end_x,
                    *end_y,
                    *head_dim,
                    *num_heads,
                    *theta,
                    *repeat_factor,
                    &out_name,
                )?;
            }
            Op::ResizeNearest2x => {
                // NCHW 2× nearest-neighbour upsample over the H/W axes.
                let x = self.val(node.inputs[0]);
                let op = self.simple_op(
                    "upsample_nearest_neighbor",
                    &out_name,
                    &node.shape,
                    vec![
                        ("x", bind_name(&x)),
                        ("scale_factor_height", bind_value(scalar_f32(2.0))),
                        ("scale_factor_width", bind_value(scalar_f32(2.0))),
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::StopGradient => {
                // Inference no-op; emit `identity` so the value keeps a
                // distinct name (it may be a graph output).
                let x = self.val(node.inputs[0]);
                let op = self.simple_op(
                    "identity",
                    &out_name,
                    &node.shape,
                    vec![("x", bind_name(&x))],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::GroupedMatMul => {
                self.lower_grouped_matmul(id, &out_name)?;
            }
            Op::DequantMatMul { scheme } => {
                if self.opts.ondevice_dequant && scheme_supports_ondevice_block_dequant(*scheme) {
                    self.lower_dequant_matmul_ondevice(id, *scheme, &out_name)?;
                } else {
                    self.lower_dequant_matmul(id, *scheme, &out_name)?;
                }
            }
            Op::DequantMoEWeights { scheme } => {
                self.lower_dequant_moe_weights(id, *scheme, &out_name)?;
            }
            Op::DequantGroupedMatMul { scheme } => {
                if self.opts.ondevice_dequant && scheme_supports_ondevice_block_dequant(*scheme) {
                    self.lower_dequant_grouped_matmul_ondevice(id, *scheme, &out_name)?;
                } else {
                    self.lower_dequant_grouped_matmul(id, *scheme, &out_name)?;
                }
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                self.lower_dequantize(id, *axis, scales, zero_points, &out_name)?;
            }
            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                self.lower_quantize(id, *axis, scales, zero_points, &out_name)?;
            }
            Op::SelectiveScan { state_size } => {
                self.lower_selective_scan(id, *state_size, &out_name)?;
            }
            // Native LSTM (carry = false). `carry = true` is filtered to the host
            // path by `is_host_op`, so it never reaches here.
            Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry: false,
            } => {
                self.lower_lstm(id, *hidden_size, *num_layers, *bidirectional, &out_name)?;
            }
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry: false,
            } => {
                self.lower_gru(id, *hidden_size, *num_layers, *bidirectional, &out_name)?;
            }
            Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                carry: false,
                relu,
            } => {
                self.lower_rnn(
                    id,
                    *hidden_size,
                    *num_layers,
                    *bidirectional,
                    *relu,
                    &out_name,
                )?;
            }
            Op::GatedDeltaNet {
                state_size,
                carry_state,
                gate_per_channel,
            } => {
                self.lower_gated_delta_net(
                    id,
                    *state_size,
                    *carry_state,
                    *gate_per_channel,
                    &out_name,
                )?;
            }
            Op::ArgMax { axis, keep_dim } => {
                self.lower_argreduce(id, *axis, *keep_dim, true, &out_name)?;
            }
            Op::ArgMin { axis, keep_dim } => {
                self.lower_argreduce(id, *axis, *keep_dim, false, &out_name)?;
            }
            Op::Reverse { axes } => {
                self.lower_reverse(id, axes, &out_name)?;
            }
            Op::Scan { .. } => {
                self.lower_scan(id, &out_name)?;
            }
            // Interleaved C64 (promoted to F32 `[…, 2n]`) Wirtinger surface.
            Op::ComplexNormSq => {
                self.lower_complex_norm_sq(id, &out_name)?;
            }
            Op::ComplexNormSqBackward => {
                self.lower_complex_norm_sq_backward(id, &out_name)?;
            }
            Op::Conjugate => {
                self.lower_conjugate(id, &out_name)?;
            }
            Op::FftButterflyStage { stage, n_fft } => {
                self.lower_fft_butterfly_stage(id, *stage, *n_fft, &out_name)?;
            }
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "op {:?} (node {})",
                    other, id.0
                )));
            }
        }
        Ok(())
    }

    /// Lower a final-carry `Op::Scan` to a native CoreML MIL `while_loop`, so the
    /// scan runs ON-DEVICE — no host split, so the whole graph stays one CoreML
    /// model instead of `length+1` separately-compiled segments. Loop vars are
    /// `[i (f32 counter), carry]`; the `cond` block tests `i < length`; the
    /// `body` block gathers `xs_j[i]`, runs the scan body, and increments `i`.
    /// Broadcast inputs are constant across iterations and referenced directly
    /// (block closure). Encoding mirrors coremltools' MIL `while_loop`: two
    /// blocks (cond, body) sharing the loop-var params; `loop_vars` input holds
    /// the initial values; op outputs are the final loop vars.
    fn lower_scan(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let outer_graph: &'a Graph = self.graph;
        // Extract the scan spec + outer operands (node borrow ends with the block).
        let (body, length, num_bcast, num_xs, inputs, carry_shape) = {
            let node = outer_graph.node(id);
            let (body, length, num_bcast, num_xs): (&'a Graph, u32, usize, usize) = match &node.op {
                Op::Scan {
                    body,
                    length,
                    num_bcast,
                    num_xs,
                    ..
                } => (&**body, *length, *num_bcast as usize, *num_xs as usize),
                _ => unreachable!("lower_scan on non-Scan node"),
            };
            let carry_shape = outer_graph.node(node.inputs[0]).shape.clone();
            (
                body,
                length,
                num_bcast,
                num_xs,
                node.inputs.clone(),
                carry_shape,
            )
        };
        let init_carry = self.val(inputs[0]);
        let bcast_names: Vec<String> = (0..num_bcast).map(|j| self.val(inputs[1 + j])).collect();
        let xs_names: Vec<String> = (0..num_xs)
            .map(|j| self.val(inputs[1 + num_bcast + j]))
            .collect();

        // Body Op::Input node ids in construction order: [carry, bcast.., xs..].
        let body_inputs: Vec<NodeId> = body
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .map(|n| n.id)
            .collect();

        let f32s = scalar_shape(DType::F32);
        let bools = scalar_shape(DType::Bool);
        let i32s = scalar_shape(DType::I32);
        let blk_inputs = |ctx: &Self| -> Result<Vec<proto::NamedValueType>> {
            let _ = ctx;
            Ok(vec![
                named_value_type(&format!("{out_name}_lv_i"), &f32s)?,
                named_value_type(&format!("{out_name}_lv_carry"), &carry_shape)?,
            ])
        };
        let pi = format!("{out_name}_lv_i");
        let pc = format!("{out_name}_lv_carry");

        // Loop-var init: counter const 0.0. Carry init = outer producer.
        let i0_name = format!("{out_name}_i0");
        let c0 = make_const(&mut self.blob, &i0_name, &f32s, &[0.0])?;
        self.operations.push(c0);

        // --- cond block: less(i, length) -> bool ---
        let cond_out = format!("{out_name}_cond");
        let cond_op = self.simple_op(
            "less",
            &cond_out,
            &bools,
            vec![
                ("x", bind_name(&pi)),
                ("y", bind_value(scalar_f32(length as f32))),
            ],
        )?;
        let cond_block = proto::Block {
            inputs: blk_inputs(self)?,
            outputs: vec![cond_out],
            operations: vec![cond_op],
            attributes: HashMap::new(),
        };

        // --- body block: build into a fresh op list + name scope ---
        let saved_ops = std::mem::take(&mut self.operations);
        let saved_names = std::mem::take(&mut self.names);

        let i_int = format!("{out_name}_i_int");
        self.emit(
            "cast",
            &i_int,
            &i32s,
            vec![
                ("x", bind_name(&pi)),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;

        // Seed body inputs: carry -> loop param; bcast -> outer name (closure);
        // xs -> gather(outer_xs, i, axis=0).
        self.names.insert(body_inputs[0].0, pc.clone());
        for j in 0..num_bcast {
            self.names
                .insert(body_inputs[1 + j].0, bcast_names[j].clone());
        }
        for j in 0..num_xs {
            let in_id = body_inputs[1 + num_bcast + j];
            let step_shape = body.node(in_id).shape.clone();
            let gname = format!("{out_name}_xs{j}");
            self.emit(
                "gather",
                &gname,
                &step_shape,
                vec![
                    ("x", bind_name(&xs_names[j])),
                    ("indices", bind_name(&i_int)),
                    ("axis", bind_value(scalar_i32(0))),
                    // Required by the iOS17+ MIL opset; the older-opset
                    // post-pass in mod.rs strips it back out when unsupported.
                    ("validate_indices", bind_value(scalar_bool(false))),
                ],
            )?;
            self.names.insert(in_id.0, gname);
        }

        // Lower the body graph (skip its Op::Inputs — already seeded). Prefix its
        // `v{id}` names so they can't collide with the outer graph's identically
        // numbered nodes that this block references via closure.
        let saved_prefix = std::mem::replace(&mut self.name_prefix, format!("{out_name}_b"));
        self.graph = body;
        for bn in body.nodes() {
            if matches!(bn.op, Op::Input { .. }) {
                continue;
            }
            self.lower_node(bn.id)?;
        }
        let new_carry = self.val(body.outputs[0]);
        self.graph = outer_graph;
        self.name_prefix = saved_prefix;

        let new_i = format!("{out_name}_ni");
        self.emit(
            "add",
            &new_i,
            &f32s,
            vec![("x", bind_name(&pi)), ("y", bind_value(scalar_f32(1.0)))],
        )?;

        let body_ops = std::mem::replace(&mut self.operations, saved_ops);
        self.names = saved_names;
        let body_block = proto::Block {
            inputs: blk_inputs(self)?,
            outputs: vec![new_i, new_carry],
            operations: body_ops,
            attributes: HashMap::new(),
        };

        // --- while_loop op: loop_vars=[i0, carry]; outputs=[final_i, carry] ---
        let final_i = format!("{out_name}_fi");
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), scalar_str(out_name));
        let while_op = proto::Operation {
            r#type: "while_loop".to_string(),
            inputs: HashMap::from([("loop_vars".to_string(), bind_names(&[i0_name, init_carry]))]),
            outputs: vec![
                named_value_type(&final_i, &f32s)?,
                named_value_type(out_name, &carry_shape)?,
            ],
            blocks: vec![cond_block, body_block],
            attributes: attrs,
        };
        self.operations.push(while_op);
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Emit a single-output op and push it (without registering a node).
    pub(crate) fn emit(
        &mut self,
        ty: &str,
        name: &str,
        shape: &Shape,
        binds: Vec<(&str, proto::Argument)>,
    ) -> Result<()> {
        let op = simple_op_flex(ty, name, shape, binds, self.opts.flexible_inputs)?;
        self.operations.push(op);
        Ok(())
    }

    pub(crate) fn simple_op(
        &self,
        ty: &str,
        out_name: &str,
        out_shape: &Shape,
        inputs: Vec<(&str, proto::Argument)>,
    ) -> Result<proto::Operation> {
        simple_op_flex(ty, out_name, out_shape, inputs, self.opts.flexible_inputs)
    }

    /// Emit `dst = src[..., start..start+len]` along the last axis.
    pub(crate) fn slice_last(
        &mut self,
        src: &str,
        src_rank: usize,
        start: usize,
        len: usize,
        out_shape: &Shape,
        dst: &str,
    ) -> Result<()> {
        self.slice_axis(src, src_rank, src_rank - 1, start, len, out_shape, dst)
    }

    /// Emit `dst = src` sliced to `[start, start+len)` along `axis`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn slice_axis(
        &mut self,
        src: &str,
        src_rank: usize,
        axis: usize,
        start: usize,
        len: usize,
        out_shape: &Shape,
        dst: &str,
    ) -> Result<()> {
        let mut begin = vec![0i32; src_rank];
        let mut size = vec![-1i32; src_rank];
        begin[axis] = start as i32;
        size[axis] = len as i32;
        self.emit(
            "slice_by_size",
            dst,
            out_shape,
            vec![
                ("x", bind_name(src)),
                ("begin", bind_value(vec_i32(&begin))),
                ("size", bind_value(vec_i32(&size))),
            ],
        )
    }

    pub(crate) fn reshape_to(
        &mut self,
        src: &str,
        dims: &[i64],
        out_shape: &Shape,
        dst: &str,
    ) -> Result<()> {
        let s: Vec<i32> = dims.iter().map(|&v| v as i32).collect();
        self.emit(
            "reshape",
            dst,
            out_shape,
            vec![("x", bind_name(src)), ("shape", bind_value(vec_i32(&s)))],
        )
    }

    pub(crate) fn matmul(&mut self, dst: &str, x: &str, y: &str, out_shape: &Shape) -> Result<()> {
        self.emit(
            "matmul",
            dst,
            out_shape,
            vec![
                ("x", bind_name(x)),
                ("y", bind_name(y)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(false))),
            ],
        )
    }

    pub(crate) fn matmul_op(
        &mut self,
        dst: &str,
        x: &str,
        y: &str,
        tx: bool,
        ty: bool,
        out_shape: &Shape,
    ) -> Result<()> {
        self.emit(
            "matmul",
            dst,
            out_shape,
            vec![
                ("x", bind_name(x)),
                ("y", bind_name(y)),
                ("transpose_x", bind_value(scalar_bool(tx))),
                ("transpose_y", bind_value(scalar_bool(ty))),
            ],
        )
    }

    pub(crate) fn push_named(&mut self, id: NodeId, name: String, op: proto::Operation) {
        self.operations.push(op);
        self.names.insert(id.0, name);
    }

    pub(crate) fn unique_feature_name(&mut self, raw: &str) -> String {
        let base = sanitize(raw);
        let n = self.used_feature_names.entry(base.clone()).or_insert(0);
        let name = if *n == 0 {
            base.clone()
        } else {
            format!("{base}_{n}")
        };
        *n += 1;
        name
    }

    /// Verify every value referenced by an op (or by a block output)
    /// resolves to something produced (a function input or an op output).
    /// A dangling reference means a node wasn't lowered — most often a
    /// quantized `Param` consumed by something other than a `Dequant*`
    /// op, or a lowering gap. Reporting it here yields a precise message
    /// instead of CoreML's opaque "in operation vN: …" parse failure.
    pub(crate) fn verify_refs(&self, block_outputs: &[String]) -> Result<()> {
        let mut produced: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for nv in &self.func_inputs {
            produced.insert(nv.name.as_str());
        }
        for op in &self.operations {
            for out in &op.outputs {
                produced.insert(out.name.as_str());
            }
        }
        let undefined = |name: &str| -> CoremlError {
            // Resolve which graph node owns this dangling name for a precise
            // diagnosis (usually a quantized Param that emitted nothing).
            let hint = self
                .graph
                .nodes()
                .iter()
                .find(|n| match &n.op {
                    Op::Param { name: p } => sanitize(p) == name || format!("v{}", n.id.0) == name,
                    Op::Input { name: p } => sanitize(p) == name || format!("v{}", n.id.0) == name,
                    _ => format!("v{}", n.id.0) == name,
                })
                .map(|n| format!(" — source node {:?}: {:?}", n.id, n.op))
                .unwrap_or_default();
            CoremlError::Runtime(format!(
                "CoreML lowering produced a dangling reference to value '{name}'{hint}: the source node \
                 was not lowered (e.g. a quantized Param used outside a Dequant* op, or an \
                 unhandled op). This is a backend bug, not a model error."
            ))
        };
        for op in &self.operations {
            for arg in op.inputs.values() {
                for b in &arg.arguments {
                    if let Some(proto::argument::binding::Binding::Name(n)) = &b.binding {
                        if !produced.contains(n.as_str()) {
                            return Err(undefined(n));
                        }
                    }
                }
            }
        }
        for name in block_outputs {
            if !produced.contains(name.as_str()) {
                return Err(undefined(name));
            }
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<LoweredProgram> {
        // `graph` is a shared reference (Copy); this rebinds it without moving
        // out of `self`, so the scalar-output reshape below can still `emit`.
        let graph = self.graph;

        // Outputs: one feature per graph output node.
        let mut output_names = Vec::new();
        let mut outputs = Vec::new();
        for &out_id in &graph.outputs {
            let mut vname = self.val(out_id);
            let out_shape = graph.shape(out_id);
            // CoreML features need rank ≥ 1, so a scalar output (e.g. a training
            // loss) is reshaped to `[1]` before it crosses the interface.
            let exposed = if out_shape.rank() == 0 {
                let one = Shape::new(&[1], out_shape.dtype());
                let reshaped = format!("{vname}_io1");
                self.emit(
                    "reshape",
                    &reshaped,
                    &one,
                    vec![
                        ("x", bind_name(&vname)),
                        ("shape", bind_value(vec_i32(&[1]))),
                    ],
                )?;
                vname = reshaped;
                one
            } else {
                out_shape.clone()
            };
            output_names.push(vname.clone());
            let (dims, flex_dims) = if self.opts.flexible_inputs {
                io_dims(&exposed, true)?
            } else {
                (static_dims(&exposed)?, vec![false; exposed.rank()])
            };
            outputs.push(IoTensor {
                ir_name: vname.clone(),
                feature_name: vname.clone(),
                dims,
                dtype: exposed.dtype(),
                flex_dims,
            });
        }

        // Catch dangling value references before they become an opaque
        // CoreML parse error.
        self.verify_refs(&output_names)?;

        // Bump to the iOS18 opset when the graph uses an op that requires it
        // (currently `constexpr_lut_to_dense` grouped palettization + UINT1),
        // OR when the model has no inputs — CoreML only allows empty-input
        // models at specification version ≥ 9 (iOS 18 / macOS 15). Ordinary
        // graphs stay at CoreML6 so existing behavior is unchanged.
        let needs_ios18 = self.inputs.is_empty()
            || self
                .operations
                .iter()
                .any(|op| op.r#type == "constexpr_lut_to_dense");
        let (opset, spec_version) = if needs_ios18 {
            (OPSET_IOS18, SPEC_VERSION_IOS18)
        } else {
            (OPSET, SPEC_VERSION)
        };

        // The `gather` op's `validate_indices` param only exists in the
        // iOS18 / spec-9 opset. At the broadly-compatible CoreML6 (spec-8)
        // baseline the ANE parser rejects it ("Invalid param name
        // 'validate_indices'"), so strip it from every `gather` when the graph
        // stays at CoreML6. (spec-9/LUT graphs keep it — required there.)
        if !needs_ios18 {
            for op in self.operations.iter_mut() {
                if op.r#type == "gather" {
                    op.inputs.remove("validate_indices");
                }
            }
        }

        let block = proto::Block {
            inputs: vec![],
            outputs: output_names,
            operations: self.operations,
            attributes: HashMap::new(),
        };
        let mut block_specializations = HashMap::new();
        block_specializations.insert(opset.to_string(), block);

        let function = proto::Function {
            inputs: self.func_inputs,
            opset: opset.to_string(),
            block_specializations,
            attributes: HashMap::new(),
        };
        let mut functions = HashMap::new();
        functions.insert("main".to_string(), function);

        let program = proto::Program {
            version: 1,
            functions,
            doc_string: String::new(),
            attributes: HashMap::new(),
        };

        let description = proto::ModelDescription {
            input: self
                .inputs
                .iter()
                .map(feature_description)
                .collect::<Result<_>>()?,
            output: outputs
                .iter()
                .map(feature_description)
                .collect::<Result<_>>()?,
            metadata: Some(proto::Metadata {
                short_description: "RLX-generated ML Program".into(),
                author: "rlx-coreml".into(),
                ..Default::default()
            }),
        };

        let model = proto::Model {
            specification_version: spec_version,
            description: Some(description),
            is_updatable: false,
            r#type: Some(proto::model::Type::MlProgram(program)),
        };

        Ok(LoweredProgram {
            model,
            inputs: self.inputs,
            outputs,
            blob: self.blob.finish(),
        })
    }
}

// --------------------------------------------------------------------------
// proto builders
// --------------------------------------------------------------------------
