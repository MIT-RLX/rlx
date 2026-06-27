// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

use std::collections::HashMap;

use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};

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
}

impl Default for LowerOptions {
    fn default() -> Self {
        Self {
            float_dtype: DType::F32,
            flexible_inputs: false,
            ondevice_dequant: true,
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
    blob: crate::mlpackage::BlobWriter,
}

impl<'a> LowerCtx<'a> {
    fn new(
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
            blob: crate::mlpackage::BlobWriter::new(),
        }
    }

    fn val(&self, id: NodeId) -> String {
        self.names
            .get(&id.0)
            .cloned()
            .unwrap_or_else(|| format!("v{}", id.0))
    }

    /// Like [`Self::val`], but coerces a bool operand to fp32 first. CoreML's
    /// arithmetic ops (mul/add/…) reject bool tensors, but VITS multiplies
    /// activations by bool masks; cast bool → fp32 (true→1.0, false→0.0).
    fn val_numeric(&mut self, id: NodeId) -> Result<String> {
        let name = self.val(id);
        if self.graph.shape(id).dtype() == DType::Bool {
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
            Ok(cast_name)
        } else {
            Ok(name)
        }
    }

    /// Walk the graph in topo order, emitting one MIL op per node.
    fn run(&mut self) -> Result<()> {
        for id in self.graph.topo_order() {
            self.lower_node(id)?;
        }
        Ok(())
    }

    fn lower_node(&mut self, id: NodeId) -> Result<()> {
        let node = self.graph.node(id);
        let out_name = format!("v{}", id.0);
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
                } else if self.typed_params.contains_key(name) {
                    // Quantized weight — host-dequantized by the consuming
                    // Dequant* op, which bakes its own f32 const. Emit
                    // nothing here.
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
                    ],
                )?;
                self.push_named(id, out_name, op);
            }
            Op::Narrow { axis, start, len } => {
                let x = self.val(node.inputs[0]);
                let rank = node.shape.rank();
                let mut begin = vec![0i32; rank];
                let mut size = vec![-1i32; rank];
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
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => {
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
                let x = self.val(node.inputs[0]);
                let y = self.val(node.inputs[1]);
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
            Op::GatedDeltaNet {
                state_size,
                carry_state,
            } => {
                self.lower_gated_delta_net(id, *state_size, *carry_state, &out_name)?;
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
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "op {:?} (node {})",
                    other, id.0
                )));
            }
        }
        Ok(())
    }

    /// Lower an activation, composing the ones MIL has no direct op for.
    fn lower_activation(&mut self, id: NodeId, act: Activation, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        // (mil_type, optional ("param", value)) for the simple unary cases.
        let direct: Option<(&str, Vec<(&str, proto::Argument)>)> = match act {
            Activation::Relu => Some(("relu", vec![])),
            Activation::Sigmoid => Some(("sigmoid", vec![])),
            Activation::Tanh => Some(("tanh", vec![])),
            Activation::Exp => Some(("exp", vec![])),
            // MIL `log` is `log(x + epsilon)` and — like `rsqrt` — requires the
            // param explicitly (CoreML rejects the model otherwise). Use CoreML's
            // own default 1e-45 so the result is unperturbed (negligible for x≥1,
            // e.g. the log-sum-exp in softmax-cross-entropy that surfaced this).
            Activation::Log => Some(("log", vec![("epsilon", bind_value(scalar_f32(1e-45)))])),
            Activation::Sqrt => Some(("sqrt", vec![])),
            Activation::Rsqrt => Some(("rsqrt", vec![("epsilon", bind_value(scalar_f32(1e-12)))])),
            Activation::Abs => Some(("abs", vec![])),
            Activation::Sin => Some(("sin", vec![])),
            Activation::Cos => Some(("cos", vec![])),
            Activation::Tan => Some(("tan", vec![])),
            Activation::Atan => Some(("atan", vec![])),
            Activation::Round => Some(("round", vec![])),
            Activation::Gelu => Some(("gelu", vec![("mode", bind_value(scalar_str("EXACT")))])),
            Activation::GeluApprox => Some((
                "gelu",
                vec![("mode", bind_value(scalar_str("TANH_APPROXIMATION")))],
            )),
            // Composed below.
            Activation::Silu | Activation::Neg => None,
        };

        if let Some((ty, mut params)) = direct {
            let mut binds = vec![("x", bind_name(&x))];
            binds.append(&mut params);
            let op = self.simple_op(ty, out_name, &node.shape, binds)?;
            self.push_named(id, out_name.to_string(), op);
            return Ok(());
        }

        match act {
            // silu(x) = x * sigmoid(x)
            Activation::Silu => {
                let sig = format!("{out_name}_sig");
                let sig_op =
                    self.simple_op("sigmoid", &sig, &node.shape, vec![("x", bind_name(&x))])?;
                self.operations.push(sig_op);
                let op = self.simple_op(
                    "mul",
                    out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&sig))],
                )?;
                self.push_named(id, out_name.to_string(), op);
            }
            // neg(x) = mul(x, -1)
            Activation::Neg => {
                let op = self.simple_op(
                    "mul",
                    out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_value(scalar_f32(-1.0)))],
                )?;
                self.push_named(id, out_name.to_string(), op);
            }
            _ => unreachable!("handled above"),
        }
        Ok(())
    }

    /// LayerNorm over the last `axis` dims, with optional affine. The IR
    /// node carries inputs `[x, gamma?, beta?]`.
    fn lower_layer_norm(&mut self, id: NodeId, axis: i32, eps: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let rank = node.shape.rank() as i32;
        let norm_axis = if axis < 0 { axis + rank } else { axis };
        // MIL layer_norm normalises over `axes`; RLX LayerNorm is over the
        // trailing dims from `axis` onward.
        let axes: Vec<i32> = (norm_axis..rank).collect();
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("axes", bind_value(vec_i32(&axes))),
            ("epsilon", bind_value(scalar_f32(eps))),
        ];
        if node.inputs.len() > 1 {
            let g = self.val(node.inputs[1]);
            binds.push(("gamma", bind_name(&g)));
        }
        if node.inputs.len() > 2 {
            let b = self.val(node.inputs[2]);
            binds.push(("beta", bind_name(&b)));
        }
        let op = self.simple_op("layer_norm", out_name, &node.shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// RMSNorm over the trailing dims from `axis`: composed from
    /// primitive MIL ops since the base opset has no `rms_norm`.
    /// `y = x · rsqrt(mean(x², axes) + eps) · gamma`. Inputs `[x, gamma?]`.
    fn lower_rms_norm(&mut self, id: NodeId, axis: i32, eps: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let rank = node.shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red_shape = reduced_shape(&node.shape, norm_axis);

        // sq = x * x
        let sq = format!("{out_name}_sq");
        self.operations.push(self.simple_op(
            "mul",
            &sq,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?);
        // ms = reduce_mean(sq, axes, keep_dims=true)
        let ms = format!("{out_name}_ms");
        self.operations.push(self.simple_op(
            "reduce_mean",
            &ms,
            &red_shape,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?);
        // ms_eps = ms + eps
        let mse = format!("{out_name}_mse");
        self.operations.push(self.simple_op(
            "add",
            &mse,
            &red_shape,
            vec![("x", bind_name(&ms)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        // inv = rsqrt(ms_eps)  (eps already folded into ms_eps above)
        let inv = format!("{out_name}_inv");
        self.operations.push(self.simple_op(
            "rsqrt",
            &inv,
            &red_shape,
            vec![
                ("x", bind_name(&mse)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?);

        // RmsNorm carries inputs [x, gamma, beta]; gamma scales and beta
        // shifts (matching the CPU kernel `x·inv·gamma + beta`). Both are
        // optional defensively, though the IR verifier requires all three.
        let has_gamma = node.inputs.len() > 1;
        let has_beta = node.inputs.len() > 2;

        // xn = x * inv  (broadcast)
        let xn_name = if has_gamma || has_beta {
            format!("{out_name}_xn")
        } else {
            out_name.to_string()
        };
        self.operations.push(self.simple_op(
            "mul",
            &xn_name,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
        )?);

        let mut last = xn_name;
        if has_gamma {
            let g = self.val(node.inputs[1]);
            let name = if has_beta {
                format!("{out_name}_xg")
            } else {
                out_name.to_string()
            };
            self.operations.push(self.simple_op(
                "mul",
                &name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&g))],
            )?);
            last = name;
        }
        if has_beta {
            let b = self.val(node.inputs[2]);
            self.operations.push(self.simple_op(
                "add",
                out_name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&b))],
            )?);
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// RMSNorm backward w.r.t. input. Inputs `[x, gamma, beta, dy]`, output = `x`.
    /// Mirrors `compose_rms_norm_backward_input`:
    ///   inv = rsqrt(mean(x², ax) + eps);  dy_g = dy·gamma
    ///   dot = mean(x·dy_g, ax);  dx = inv·(dy_g − x·dot·inv³)
    /// All reductions keep dims so `[...,1]` factors broadcast over `[...,H]`.
    #[cfg(feature = "training")]
    fn lower_rms_norm_backward_input(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        // node.inputs[2] = beta — additive in the forward, so absent from dx.
        let dy = self.val(node.inputs[3]);
        let full = node.shape.clone();
        let rank = full.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&full, norm_axis);
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        // inv = rsqrt(mean(x*x, axes) + eps)
        let x2 = format!("{out_name}_x2");
        self.emit(
            "mul",
            &x2,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?;
        let mx2 = format!("{out_name}_mx2");
        self.emit(
            "reduce_mean",
            &mx2,
            &red,
            vec![
                ("x", bind_name(&x2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&mx2)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let inv2 = format!("{out_name}_inv2");
        self.emit(
            "mul",
            &inv2,
            &red,
            vec![("x", bind_name(&inv)), ("y", bind_name(&inv))],
        )?;

        // dy_g = dy * gamma  (gamma [H] broadcasts over [...,H])
        let dyg = format!("{out_name}_dyg");
        self.emit(
            "mul",
            &dyg,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gamma))],
        )?;
        // dot = mean(x * dy_g, axes)
        let xdyg = format!("{out_name}_xdyg");
        self.emit(
            "mul",
            &xdyg,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&dyg))],
        )?;
        let dot = format!("{out_name}_dot");
        self.emit(
            "reduce_mean",
            &dot,
            &red,
            vec![
                ("x", bind_name(&xdyg)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        // term2 = x * dot * inv²  (the outer `* inv` below makes the cross term inv³, not inv⁴)
        let xdot = format!("{out_name}_xdot");
        self.emit(
            "mul",
            &xdot,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&dot))],
        )?;
        let term2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &term2,
            &full,
            vec![("x", bind_name(&xdot)), ("y", bind_name(&inv2))],
        )?;
        // diff = dy_g - term2;  dx = diff * inv
        let diff = format!("{out_name}_diff");
        self.emit(
            "sub",
            &diff,
            &full,
            vec![("x", bind_name(&dyg)), ("y", bind_name(&term2))],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&diff)), ("y", bind_name(&inv))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// RMSNorm backward w.r.t. gamma. Inputs `[x, gamma, beta, dy]`, output =
    /// `gamma` (`[H]`). Mirrors `compose_rms_norm_backward_gamma`:
    ///   `dgamma = sum_batch(dy · x · rsqrt(mean(x², ax) + eps))`.
    #[cfg(feature = "training")]
    fn lower_rms_norm_backward_gamma(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[3]);
        let gamma_shape = node.shape.clone();
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = x_shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);
        let batch_axes: Vec<i32> = (0..rank as i32)
            .filter(|&i| i as usize != norm_axis)
            .collect();

        let x2 = format!("{out_name}_x2");
        self.emit(
            "mul",
            &x2,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?;
        let mx2 = format!("{out_name}_mx2");
        self.emit(
            "reduce_mean",
            &mx2,
            &red,
            vec![
                ("x", bind_name(&x2)),
                ("axes", bind_value(vec_i32(&axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&mx2)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let xinv = format!("{out_name}_xinv");
        self.emit(
            "mul",
            &xinv,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&inv))],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&xinv))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// RMSNorm backward w.r.t. beta. Inputs `[x, gamma, beta, dy]`, output =
    /// `beta` (`[H]`). `dbeta = sum_batch(dy)`.
    #[cfg(feature = "training")]
    fn lower_rms_norm_backward_beta(
        &mut self,
        id: NodeId,
        _axis: i32,
        _eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let dy = self.val(node.inputs[3]);
        let beta_shape = node.shape.clone();
        let rank = self.graph.shape(node.inputs[3]).rank();
        // beta is over the last (feature) axis; reduce every batch axis.
        let batch_axes: Vec<i32> = (0..rank as i32 - 1).collect();
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &beta_shape,
            vec![
                ("x", bind_name(&dy)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native LayerNorm backward w.r.t. input (axis = -1). Inputs `[x, gamma, dy]`,
    /// output matches `x`. Mirrors `compose_layer_norm_backward_input`:
    ///   `dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat))`, `sy = dy·γ`,
    ///   `x_hat = (x − mean)·inv_std`, `inv_std = rsqrt(var + eps)`. Composed MIL
    /// with implicit broadcasting (reduced `[..,1]` tensors broadcast over the norm
    /// axis), no decomposition `expand`s.
    #[cfg(feature = "training")]
    fn lower_layer_norm_backward_input(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let dy = self.val(node.inputs[2]);
        let full = node.shape.clone();
        let rank = full.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&full, norm_axis);
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        // mean, centered x
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&x)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &full,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        // var, inv_std = rsqrt(var + eps)
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;

        // sy = dy·γ; its mean; and mean(sy·x_hat)
        let sy = format!("{out_name}_sy");
        self.emit(
            "mul",
            &sy,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gamma))],
        )?;
        let m_sy = format!("{out_name}_msy");
        self.emit(
            "reduce_mean",
            &m_sy,
            &red,
            vec![
                ("x", bind_name(&sy)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let sy_xh = format!("{out_name}_syxh");
        self.emit(
            "mul",
            &sy_xh,
            &full,
            vec![("x", bind_name(&sy)), ("y", bind_name(&x_hat))],
        )?;
        let m_sxh = format!("{out_name}_msxh");
        self.emit(
            "reduce_mean",
            &m_sxh,
            &red,
            vec![
                ("x", bind_name(&sy_xh)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;

        // dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat))
        let t1 = format!("{out_name}_t1");
        self.emit(
            "sub",
            &t1,
            &full,
            vec![("x", bind_name(&sy)), ("y", bind_name(&m_sy))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &full,
            vec![("x", bind_name(&x_hat)), ("y", bind_name(&m_sxh))],
        )?;
        let t3 = format!("{out_name}_t3");
        self.emit(
            "sub",
            &t3,
            &full,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&inv_std)), ("y", bind_name(&t3))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native LayerNorm backward w.r.t. gamma. Inputs `[x, dy]`, output = gamma
    /// shape. Mirrors `compose_layer_norm_backward_gamma`:
    ///   `dgamma = Σ_batch(dy · x_hat)`, `x_hat = (x − mean)·rsqrt(var + eps)`.
    #[cfg(feature = "training")]
    fn lower_layer_norm_backward_gamma(
        &mut self,
        id: NodeId,
        axis: i32,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let gamma_shape = node.shape.clone();
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = x_shape.rank();
        let norm_axis = if axis < 0 { axis + rank as i32 } else { axis } as usize;
        let axes: Vec<i32> = (norm_axis..rank).map(|a| a as i32).collect();
        let red = reduced_shape(&x_shape, norm_axis);
        let batch_axes: Vec<i32> = (0..rank as i32)
            .filter(|&i| i as usize != norm_axis)
            .collect();
        let red_axes = || bind_value(vec_i32(&axes));
        let keep = || bind_value(scalar_bool(true));

        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&x)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &x_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &x_shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axes()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &x_shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &x_shape,
            vec![("x", bind_name(&dy)), ("y", bind_name(&x_hat))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&batch_axes))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. input (NCHW). Inputs `[x, gamma, beta, dy]`,
    /// output matches `x`. Reshapes `[N,C,H,W] → [N,G,M]` (M = C/G·H·W) so the group
    /// stats are a single last-axis reduce; the affine `sy = dy·γ` is done in NCHW
    /// (γ broadcasts over H,W) before the reshape. Same math as
    /// `compose_group_norm_backward_input`, without the per-group narrow/concat loop.
    #[cfg(feature = "training")]
    fn lower_group_norm_backward_input(
        &mut self,
        id: NodeId,
        num_groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        // inputs[2] = beta (additive in the forward, absent from dx)
        let dy = self.val(node.inputs[3]);
        let full = node.shape.clone(); // [N,C,H,W]
        let (n, c, h, w) = (
            full.dim(0).unwrap_static(),
            full.dim(1).unwrap_static(),
            full.dim(2).unwrap_static(),
            full.dim(3).unwrap_static(),
        );
        let dt = full.dtype();
        let m = (c / num_groups) * h * w;
        let grouped = Shape::new(&[n, num_groups, m], dt);
        let red = Shape::new(&[n, num_groups, 1], dt);
        let g3 = || bind_value(vec_i32(&[n as i32, num_groups as i32, m as i32]));
        let red_axis = || bind_value(vec_i32(&[2]));
        let keep = || bind_value(scalar_bool(true));

        // sy = dy·γ in NCHW (γ [C] → [1,C,1,1] broadcasts over H,W)
        let gr = format!("{out_name}_gr");
        self.emit(
            "reshape",
            &gr,
            &Shape::new(&[1, c, 1, 1], dt),
            vec![
                ("x", bind_name(&gamma)),
                ("shape", bind_value(vec_i32(&[1, c as i32, 1, 1]))),
            ],
        )?;
        let sy_nchw = format!("{out_name}_synchw");
        self.emit(
            "mul",
            &sy_nchw,
            &full,
            vec![("x", bind_name(&dy)), ("y", bind_name(&gr))],
        )?;

        // group the channels: x, sy → [N,G,M]
        let xf = format!("{out_name}_xf");
        self.emit(
            "reshape",
            &xf,
            &grouped,
            vec![("x", bind_name(&x)), ("shape", g3())],
        )?;
        let syf = format!("{out_name}_syf");
        self.emit(
            "reshape",
            &syf,
            &grouped,
            vec![("x", bind_name(&sy_nchw)), ("shape", g3())],
        )?;

        // mean, var, inv_std over the group axis
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&xf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &grouped,
            vec![("x", bind_name(&xf)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "mul",
            &x_hat,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;

        // mean(sy), mean(sy·x_hat) over the group axis
        let m_sy = format!("{out_name}_msy");
        self.emit(
            "reduce_mean",
            &m_sy,
            &red,
            vec![
                ("x", bind_name(&syf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let sy_xh = format!("{out_name}_syxh");
        self.emit(
            "mul",
            &sy_xh,
            &grouped,
            vec![("x", bind_name(&syf)), ("y", bind_name(&x_hat))],
        )?;
        let m_sxh = format!("{out_name}_msxh");
        self.emit(
            "reduce_mean",
            &m_sxh,
            &red,
            vec![
                ("x", bind_name(&sy_xh)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;

        // flat_dx = inv_std·(sy − mean(sy) − x_hat·mean(sy·x_hat)); reshape to NCHW
        let t1 = format!("{out_name}_t1");
        self.emit(
            "sub",
            &t1,
            &grouped,
            vec![("x", bind_name(&syf)), ("y", bind_name(&m_sy))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &grouped,
            vec![("x", bind_name(&x_hat)), ("y", bind_name(&m_sxh))],
        )?;
        let t3 = format!("{out_name}_t3");
        self.emit(
            "sub",
            &t3,
            &grouped,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        let flat_dx = format!("{out_name}_fdx");
        self.emit(
            "mul",
            &flat_dx,
            &grouped,
            vec![("x", bind_name(&t3)), ("y", bind_name(&inv_std))],
        )?;
        let op = self.simple_op(
            "reshape",
            out_name,
            &full,
            vec![
                ("x", bind_name(&flat_dx)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. gamma (NCHW). Inputs `[x, dy]`, output =
    /// gamma `[C]`. `dgamma[c] = Σ_{n,h,w} dy·x_hat`, with `x_hat` the group-
    /// normalized `x` (computed in the `[N,G,M]` layout, then reshaped back to NCHW
    /// so the channel reduction over axes {N,H,W} is exact for any batch size).
    #[cfg(feature = "training")]
    fn lower_group_norm_backward_gamma(
        &mut self,
        id: NodeId,
        num_groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let gamma_shape = node.shape.clone(); // [C]
        let xs = self.graph.shape(node.inputs[0]).clone(); // [N,C,H,W]
        let (n, c, h, w) = (
            xs.dim(0).unwrap_static(),
            xs.dim(1).unwrap_static(),
            xs.dim(2).unwrap_static(),
            xs.dim(3).unwrap_static(),
        );
        let dt = xs.dtype();
        let m = (c / num_groups) * h * w;
        let grouped = Shape::new(&[n, num_groups, m], dt);
        let red = Shape::new(&[n, num_groups, 1], dt);
        let g3 = || bind_value(vec_i32(&[n as i32, num_groups as i32, m as i32]));
        let red_axis = || bind_value(vec_i32(&[2]));
        let keep = || bind_value(scalar_bool(true));

        let xf = format!("{out_name}_xf");
        self.emit(
            "reshape",
            &xf,
            &grouped,
            vec![("x", bind_name(&x)), ("shape", g3())],
        )?;
        let mean = format!("{out_name}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            &red,
            vec![
                ("x", bind_name(&xf)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &grouped,
            vec![("x", bind_name(&xf)), ("y", bind_name(&mean))],
        )?;
        let xc2 = format!("{out_name}_xc2");
        self.emit(
            "mul",
            &xc2,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out_name}_var");
        self.emit(
            "reduce_mean",
            &var,
            &red,
            vec![
                ("x", bind_name(&xc2)),
                ("axes", red_axis()),
                ("keep_dims", keep()),
            ],
        )?;
        let ve = format!("{out_name}_ve");
        self.emit(
            "add",
            &ve,
            &red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv_std = format!("{out_name}_invs");
        self.emit(
            "rsqrt",
            &inv_std,
            &red,
            vec![
                ("x", bind_name(&ve)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let x_hat_g = format!("{out_name}_xhatg");
        self.emit(
            "mul",
            &x_hat_g,
            &grouped,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv_std))],
        )?;
        // back to NCHW so the channel-aligned reduction is unambiguous
        let x_hat = format!("{out_name}_xhat");
        self.emit(
            "reshape",
            &x_hat,
            &xs,
            vec![
                ("x", bind_name(&x_hat_g)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &xs,
            vec![("x", bind_name(&dy)), ("y", bind_name(&x_hat))],
        )?;
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &gamma_shape,
            vec![
                ("x", bind_name(&prod)),
                ("axes", bind_value(vec_i32(&[0, 2, 3]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native GroupNorm backward w.r.t. beta (NCHW). Inputs `[x, dy]` (x unused),
    /// output = beta `[C] = Σ_{n,h,w} dy` — a single channel-aligned reduce_sum.
    #[cfg(feature = "training")]
    fn lower_group_norm_backward_beta(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let dy = self.val(node.inputs[1]);
        let beta_shape = node.shape.clone();
        let op = self.simple_op(
            "reduce_sum",
            out_name,
            &beta_shape,
            vec![
                ("x", bind_name(&dy)),
                ("axes", bind_value(vec_i32(&[0, 2, 3]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native softmax-cross-entropy-with-logits forward. Inputs `[logits [N,C],
    /// labels [N]]`, output per-row loss `[N] = logsumexp(logits) − logits[label]`.
    /// Mirrors `rlx_fusion::lower_softmax_cross_entropy_with_logits` but selects the
    /// label logit with one `one_hot`+`reduce_sum` instead of concatenating C columns.
    /// Pairs with the native backward so a full SCE training step stays off the O(C)
    /// decompose path on the ANE.
    #[cfg(feature = "training")]
    fn lower_softmax_cross_entropy_with_logits(
        &mut self,
        id: NodeId,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let logits_id = node.inputs[0];
        let labels_id = node.inputs[1];
        let logits_shape = self.graph.shape(logits_id).clone(); // [N, C]
        let out_shape = node.shape.clone(); // [N]
        let n = logits_shape.dim(0).unwrap_static();
        let c = logits_shape.dim(1).unwrap_static() as i32;
        let dt = logits_shape.dtype();
        let logits = self.val(logits_id);
        let kept = Shape::new(&[n, 1], dt); // [N,1]

        // lse = max + log(sum(exp(logits − max)))  (the numerically-stable logsumexp)
        let m = format!("{out_name}_max");
        self.emit(
            "reduce_max",
            &m,
            &kept,
            vec![
                ("x", bind_name(&logits)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let shifted = format!("{out_name}_sh");
        self.emit(
            "sub",
            &shifted,
            &logits_shape,
            vec![("x", bind_name(&logits)), ("y", bind_name(&m))],
        )?;
        let exp_d = format!("{out_name}_exp");
        self.emit(
            "exp",
            &exp_d,
            &logits_shape,
            vec![("x", bind_name(&shifted))],
        )?;
        let sum_exp = format!("{out_name}_se");
        self.emit(
            "reduce_sum",
            &sum_exp,
            &out_shape,
            vec![
                ("x", bind_name(&exp_d)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;
        let log_sum = format!("{out_name}_ls");
        self.emit(
            "log",
            &log_sum,
            &out_shape,
            vec![
                ("x", bind_name(&sum_exp)),
                ("epsilon", bind_value(scalar_f32(1e-45))),
            ],
        )?;
        let m_flat = format!("{out_name}_mf");
        self.emit(
            "reshape",
            &m_flat,
            &out_shape,
            vec![
                ("x", bind_name(&m)),
                ("shape", bind_value(vec_i32(&[n as i32]))),
            ],
        )?;
        let lse = format!("{out_name}_lse");
        self.emit(
            "add",
            &lse,
            &out_shape,
            vec![("x", bind_name(&m_flat)), ("y", bind_name(&log_sum))],
        )?;

        // label_logit[n] = Σ_c logits[n,c]·onehot(labels)[n,c]
        let idx = format!("{out_name}_idx");
        let lshape = self.graph.shape(labels_id).clone().with_dtype(DType::I32);
        self.emit(
            "cast",
            &idx,
            &lshape,
            vec![
                ("x", bind_name(&self.val(labels_id))),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        let oh = format!("{out_name}_oh");
        self.emit(
            "one_hot",
            &oh,
            &logits_shape,
            vec![
                ("indices", bind_name(&idx)),
                ("one_hot_vector_size", bind_value(scalar_i32(c))),
                ("axis", bind_value(scalar_i32(-1))),
                ("on_value", bind_value(scalar_f32(1.0))),
                ("off_value", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let masked = format!("{out_name}_msk");
        self.emit(
            "mul",
            &masked,
            &logits_shape,
            vec![("x", bind_name(&logits)), ("y", bind_name(&oh))],
        )?;
        let label_logit = format!("{out_name}_ll");
        self.emit(
            "reduce_sum",
            &label_logit,
            &out_shape,
            vec![
                ("x", bind_name(&masked)),
                ("axes", bind_value(vec_i32(&[-1]))),
                ("keep_dims", bind_value(scalar_bool(false))),
            ],
        )?;

        let op = self.simple_op(
            "sub",
            out_name,
            &out_shape,
            vec![("x", bind_name(&lse)), ("y", bind_name(&label_logit))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native softmax-cross-entropy backward. Inputs `[logits [N,C], labels [N],
    /// d_loss [N]]`, output `dlogits [N,C] = (softmax(logits) − onehot(labels))·d_loss`.
    /// Mirrors `rlx_fusion::lower_softmax_cross_entropy_backward`, but emits MIL's
    /// single `one_hot` op instead of concatenating C class columns — the decompose
    /// path is O(C) graph nodes, which is unusable at LLM vocab sizes.
    #[cfg(feature = "training")]
    fn lower_softmax_cross_entropy_backward(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let logits_id = node.inputs[0];
        let labels_id = node.inputs[1];
        let d_loss_id = node.inputs[2];
        let full = node.shape.clone(); // [N, C]
        let n = full.dim(0).unwrap_static();
        let c = full.dim(1).unwrap_static() as i32;
        let logits = self.val(logits_id);

        // sm = softmax(logits, axis=-1)
        let sm = format!("{out_name}_sm");
        self.emit(
            "softmax",
            &sm,
            &full,
            vec![
                ("x", bind_name(&logits)),
                ("axis", bind_value(scalar_i32(-1))),
            ],
        )?;

        // onehot(labels): CoreML one_hot needs int indices; labels are f32-encoded.
        let idx = format!("{out_name}_idx");
        let lshape = self.graph.shape(labels_id).clone().with_dtype(DType::I32);
        self.emit(
            "cast",
            &idx,
            &lshape,
            vec![
                ("x", bind_name(&self.val(labels_id))),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        let oh = format!("{out_name}_oh");
        self.emit(
            "one_hot",
            &oh,
            &full,
            vec![
                ("indices", bind_name(&idx)),
                ("one_hot_vector_size", bind_value(scalar_i32(c))),
                ("axis", bind_value(scalar_i32(-1))),
                ("on_value", bind_value(scalar_f32(1.0))),
                ("off_value", bind_value(scalar_f32(0.0))),
            ],
        )?;

        // diff = sm − onehot
        let diff = format!("{out_name}_diff");
        self.emit(
            "sub",
            &diff,
            &full,
            vec![("x", bind_name(&sm)), ("y", bind_name(&oh))],
        )?;

        // dlogits = diff · d_loss, with d_loss [N] reshaped to [N,1] to broadcast over C.
        let dl2 = format!("{out_name}_dl2");
        let dl2_shape = Shape::new(&[n, 1], full.dtype());
        self.emit(
            "reshape",
            &dl2,
            &dl2_shape,
            vec![
                ("x", bind_name(&self.val(d_loss_id))),
                ("shape", bind_value(vec_i32(&[n as i32, 1]))),
            ],
        )?;
        let op = self.simple_op(
            "mul",
            out_name,
            &full,
            vec![("x", bind_name(&diff)), ("y", bind_name(&dl2))],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Native MaxPool2d backward (training). Routes each window's upstream
    /// gradient to its max position(s) via reshape + reduce_max + select —
    /// O(input size), no dense scatter. On ties EVERY maximum receives the
    /// gradient, matching the shared autodiff decomposition
    /// (`compose_max_pool2d_backward`) — important for the common relu→maxpool
    /// all-zero window (every element equals the max), and keeps ANE gradients
    /// consistent with the CPU/GPU training path.
    ///
    /// Supports the non-overlapping, unpadded case (stride == kernel, pad == 0,
    /// dims divisible by the kernel) — what CNN training uses (e.g. MNIST 2×2/2).
    /// Other configs return `Unsupported` rather than a silently wrong result.
    #[cfg(feature = "training")]
    fn lower_max_pool2d_backward(
        &mut self,
        id: NodeId,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let out_shape = node.shape.clone();
        let dt = out_shape.dtype();
        if out_shape.rank() != 4 {
            return Err(CoremlError::Unsupported(
                "max_pool2d_backward: expected NCHW (rank 4)".into(),
            ));
        }
        let dim = |i: usize| out_shape.dim(i).unwrap_static();
        let (n, c, h, w) = (dim(0), dim(1), dim(2), dim(3));
        let (kh, kw) = (kernel[0], kernel[1]);
        if stride.first() != Some(&kh)
            || stride.get(1) != Some(&kw)
            || padding.iter().any(|&p| p != 0)
            || h % kh != 0
            || w % kw != 0
        {
            return Err(CoremlError::Unsupported(format!(
                "max_pool2d_backward native kernel handles only non-overlapping, \
                 unpadded pooling with divisible dims (stride==kernel, pad==0); got \
                 kernel={kernel:?} stride={stride:?} pad={padding:?} on {h}x{w}"
            )));
        }
        let (ho, wo) = (h / kh, w / kw);
        // Keep every tensor rank ≤ 4 (the ANE reshape limit): fold N·C·Ho into one
        // batch dim. The window view is [B, kh, Wo, kw] with B = N·C·Ho, and each
        // pooling window is the (kh, kw) pair at axes [1, 3].
        let b = n * c * ho;
        let win = Shape::new(&[b, kh, wo, kw], dt);
        let red = Shape::new(&[b, 1, wo, 1], dt);
        let win_b = win.clone().with_dtype(DType::Bool);
        // Reshape [N,C,H,W] → [B,kh,Wo,kw] is a pure reinterpret (row-major) since
        // H=Ho·kh, W=Wo·kw; the final reshape inverts it back to [N,C,H,W].
        let win_dims =
            |kdim: usize, wdim: usize| vec_i32(&[b as i32, kdim as i32, wo as i32, wdim as i32]);

        let xr = format!("{out_name}_xr");
        self.emit(
            "reshape",
            &xr,
            &win,
            vec![
                ("x", bind_name(&x)),
                ("shape", bind_value(win_dims(kh, kw))),
            ],
        )?;
        let ymax = format!("{out_name}_ymax");
        self.emit(
            "reduce_max",
            &ymax,
            &red,
            vec![
                ("x", bind_name(&xr)),
                ("axes", bind_value(vec_i32(&[1, 3]))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        // mask = (xr >= ymax) ⟺ (xr == max). On ties, every maximum is marked —
        // matching the shared autodiff decomposition (`compose_max_pool2d_backward`
        // routes dy to all maxima), so ANE gradients stay consistent with the
        // CPU/GPU training path.
        let mask = format!("{out_name}_mask");
        self.emit(
            "greater_equal",
            &mask,
            &win_b,
            vec![("x", bind_name(&xr)), ("y", bind_name(&ymax))],
        )?;
        let zero = format!("{out_name}_zero");
        let zero_op = make_const(&mut self.blob, &zero, &Shape::new(&[1], dt), &[0.0])?;
        self.operations.push(zero_op);
        // route dy (broadcast over the window) to every max position, 0 elsewhere.
        let dyr = format!("{out_name}_dyr");
        self.emit(
            "reshape",
            &dyr,
            &red,
            vec![("x", bind_name(&dy)), ("shape", bind_value(win_dims(1, 1)))],
        )?;
        let dxr = format!("{out_name}_dxr");
        self.emit(
            "select",
            &dxr,
            &win,
            vec![
                ("cond", bind_name(&mask)),
                ("a", bind_name(&dyr)),
                ("b", bind_name(&zero)),
            ],
        )?;
        let op = self.simple_op(
            "reshape",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&dxr)),
                (
                    "shape",
                    bind_value(vec_i32(&[n as i32, c as i32, h as i32, w as i32])),
                ),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Emit a single-output op and push it (without registering a node).
    fn emit(
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

    fn simple_op(
        &self,
        ty: &str,
        out_name: &str,
        out_shape: &Shape,
        inputs: Vec<(&str, proto::Argument)>,
    ) -> Result<proto::Operation> {
        simple_op_flex(ty, out_name, out_shape, inputs, self.opts.flexible_inputs)
    }

    /// Emit `dst = src[..., start..start+len]` along the last axis.
    fn slice_last(
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
    fn slice_axis(
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

    /// RoPE (NeoX split-halves). Inputs `[x, cos, sin]`; rotates the first
    /// `n_rot` of the trailing `head_dim` lane, passes the rest through.
    /// Only the layout where the last axis == `head_dim` is supported
    /// (`[B,H,S,D]` or `[B,S,D]`); the cos/sin tables (`[…,n_rot/2]`)
    /// broadcast against the rotated halves.
    fn lower_rope(
        &mut self,
        id: NodeId,
        head_dim: usize,
        n_rot: usize,
        out_name: &str,
    ) -> Result<()> {
        let (shape, in0, in1, in2) = {
            let node = self.graph.node(id);
            (
                node.shape.clone(),
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
            )
        };
        let rank = shape.rank();
        let last = match shape.dim(rank - 1) {
            Dim::Static(n) => n,
            Dim::Dynamic(s) => {
                return Err(CoremlError::DynamicShape(format!("rope last dim ?{s}")));
            }
        };

        let x = self.val(in0);
        let cos = self.val(in1);
        let sin = self.val(in2);

        // Flexible layout: the rotation runs on a tensor whose LAST axis is
        // exactly `head_dim`. When the last axis instead packs multiple heads
        // (`[.., G*head_dim]`, the fused-QKV layout used by e.g. Qwen3-ASR),
        // reshape to `[.., G, head_dim]`, rotate per head, then reshape back —
        // cos/sin gain a singleton head axis so they broadcast over the heads.
        // The `last == head_dim` path is byte-for-byte the original lowering.
        let (eff_x, eff_shape, eff_cos, eff_sin, restore) = if last == head_dim {
            (x, shape.clone(), cos, sin, None)
        } else if head_dim != 0 && last % head_dim == 0 {
            let groups = last / head_dim;
            let mut gd = shape.dims().to_vec();
            gd.pop();
            gd.push(Dim::Static(groups));
            gd.push(Dim::Static(head_dim));
            let gshape = Shape::from_dims(&gd, DType::F32);
            let xg = format!("{out_name}_xg");
            self.emit(
                "reshape",
                &xg,
                &gshape,
                vec![
                    ("x", bind_name(&x)),
                    ("shape", bind_value(vec_i32(&dims_i32(&gd)))),
                ],
            )?;
            let cos_g = self.rope_insert_head_axis(in1, &cos, &format!("{out_name}_cosg"))?;
            let sin_g = self.rope_insert_head_axis(in2, &sin, &format!("{out_name}_sing"))?;
            (xg, gshape, cos_g, sin_g, Some(shape.clone()))
        } else {
            return Err(CoremlError::Unsupported(format!(
                "rope: last dim {last} is not a multiple of head_dim {head_dim} \
                 (dims={:?}, n_rot={n_rot})",
                shape.dims()
            )));
        };

        let eff_rank = eff_shape.rank();
        let rot_half = n_rot / 2;
        let half_shape = with_last(&eff_shape, rot_half);
        let rot_shape = with_last(&eff_shape, n_rot);

        // Rotated result lands in `core` — the real output unless we worked on
        // a per-head view, in which case it is reshaped back below.
        let core = match restore {
            Some(_) => format!("{out_name}_core"),
            None => out_name.to_string(),
        };

        // x1 = x[..0:rh], x2 = x[..rh:n_rot]
        let x1 = format!("{out_name}_x1");
        let x2 = format!("{out_name}_x2");
        self.slice_last(&eff_x, eff_rank, 0, rot_half, &half_shape, &x1)?;
        self.slice_last(&eff_x, eff_rank, rot_half, rot_half, &half_shape, &x2)?;

        // out1 = x1*cos - x2*sin ; out2 = x2*cos + x1*sin
        let (x1c, x2s, x2c, x1s) = (
            format!("{out_name}_x1c"),
            format!("{out_name}_x2s"),
            format!("{out_name}_x2c"),
            format!("{out_name}_x1s"),
        );
        self.emit(
            "mul",
            &x1c,
            &half_shape,
            vec![("x", bind_name(&x1)), ("y", bind_name(&eff_cos))],
        )?;
        self.emit(
            "mul",
            &x2s,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&eff_sin))],
        )?;
        self.emit(
            "mul",
            &x2c,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&eff_cos))],
        )?;
        self.emit(
            "mul",
            &x1s,
            &half_shape,
            vec![("x", bind_name(&x1)), ("y", bind_name(&eff_sin))],
        )?;
        let out1 = format!("{out_name}_o1");
        let out2 = format!("{out_name}_o2");
        self.emit(
            "sub",
            &out1,
            &half_shape,
            vec![("x", bind_name(&x1c)), ("y", bind_name(&x2s))],
        )?;
        self.emit(
            "add",
            &out2,
            &half_shape,
            vec![("x", bind_name(&x2c)), ("y", bind_name(&x1s))],
        )?;

        let axis = (eff_rank - 1) as i32;
        let pass_len = head_dim - n_rot;
        if pass_len == 0 {
            self.emit(
                "concat",
                &core,
                &eff_shape,
                vec![
                    ("values", bind_names(&[out1, out2])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        } else {
            let out_rot = format!("{out_name}_rot");
            self.emit(
                "concat",
                &out_rot,
                &rot_shape,
                vec![
                    ("values", bind_names(&[out1, out2])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
            let pass = format!("{out_name}_pass");
            let pass_shape = with_last(&eff_shape, pass_len);
            self.slice_last(&eff_x, eff_rank, n_rot, pass_len, &pass_shape, &pass)?;
            self.emit(
                "concat",
                &core,
                &eff_shape,
                vec![
                    ("values", bind_names(&[out_rot, pass])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        }

        // Per-head view → fold the head axis back into the last dim.
        if let Some(orig) = restore {
            self.emit(
                "reshape",
                out_name,
                &orig,
                vec![
                    ("x", bind_name(&core)),
                    ("shape", bind_value(vec_i32(&dims_i32(orig.dims())))),
                ],
            )?;
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Reshape a rope cos/sin table to gain a singleton head axis just before
    /// its last dim, so it broadcasts over the per-head groups when rope runs
    /// on a fused `[.., G, head_dim]` view.
    fn rope_insert_head_axis(&mut self, src: NodeId, val: &str, out: &str) -> Result<String> {
        let mut d = self.graph.shape(src).dims().to_vec();
        let pos = d.len().saturating_sub(1);
        d.insert(pos, Dim::Static(1));
        let ns = Shape::from_dims(&d, DType::F32);
        self.emit(
            "reshape",
            out,
            &ns,
            vec![
                ("x", bind_name(val)),
                ("shape", bind_value(vec_i32(&dims_i32(&d)))),
            ],
        )?;
        Ok(out.to_string())
    }

    /// Scaled dot-product attention. Inputs `[q,k,v]` (+ mask tensor for
    /// `Bias`/`Custom`). The operand layout is **dispatched** on
    /// `num_heads`/`head_dim` — different layouts need different kernels:
    ///   * **split** — last axis == `head_dim` (`[..,S,D]`, any rank ≥ 2):
    ///     the core runs directly (MIL `matmul` batches the leading B/H dims).
    ///   * **fused** — last axis == `num_heads*head_dim` (`[B,S,H·D]`, the
    ///     Qwen3 fused-QKV layout where heads are packed in the last axis):
    ///     reshape+transpose q/k/v to canonical `[B,H,S,D]`, run the core,
    ///     then transpose+reshape the result back to `[B,S,H·D]`.
    fn lower_attention(
        &mut self,
        id: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        out_name: &str,
    ) -> Result<()> {
        let (out_shape, in0, in1, in2, mask_in) = {
            let node = self.graph.node(id);
            let mask_in = match mask_kind {
                MaskKind::Bias | MaskKind::Custom => node.inputs.get(3).copied(),
                _ => None,
            };
            (
                node.shape.clone(),
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                mask_in,
            )
        };
        let rank = out_shape.rank();
        if rank < 2 {
            return Err(CoremlError::Unsupported(
                "attention: need rank >= 2 [..,S,D] layout".into(),
            ));
        }
        let last = dim_static(&out_shape, rank - 1)?;

        // ── split layout: last axis is `head_dim` ──
        //
        // The IR's `Op::Attention` is used with TWO different rank-4 operand
        // layouts across models (both end in `head_dim`, so they can't be told
        // apart by the last axis — disambiguate by which axis equals `num_heads`):
        //
        //   * `[B, S, H, D]` — heads at axis 2 (the CPU/Metal/MLX/wgpu convention;
        //     e.g. Moshi, Llama-style reshapes). `attention_core` wants the heads
        //     at axis 1, so we transpose `[B,S,H,D] → [B,H,S,D]`, attend, then
        //     transpose the `[B,H,Sq,D]` result back to `[B,Sq,H,D]`. Without this
        //     it would attend over the HEADS axis — wrong results, and once
        //     `s_q != s_k` (KV-cache decode) the QKᵀ batch dims mismatch and the
        //     CoreML predict fails outright.
        //   * `[B, H, S, D]` — heads at axis 1 (already canonical). Passed straight
        //     through to `attention_core`.
        //
        // Rank-3 `[B, S, D]` (single head) is likewise already canonical.
        if last == head_dim {
            // Identify the heads axis via the canonical `attention_geom` helper
            // (same disambiguation the MLX/wgpu backends use). `bhsd` = heads are
            // already at axis 1 (`[B,H,S,D]`, canonical for `attention_core`);
            // otherwise a rank-4 operand is `[B,S,H,D]` (heads at axis 2).
            let k_in_shape = self.graph.shape(in1).clone();
            let geom = rlx_ir::attention_geom(&out_shape, &k_in_shape, num_heads, head_dim);
            if rank == 4 && !geom.bhsd {
                // `[B, S, H, D]` → canonical `[B, H, S, D]`, attend, transpose back.
                let (b, s_q, h, s_k) = (geom.batch, geom.seq_q, geom.heads, geom.seq_k);
                let qc = self.bshd_to_bhsd(in0, b, s_q, h, head_dim, &format!("{out_name}_q"))?;
                let kc = self.bshd_to_bhsd(in1, b, s_k, h, head_dim, &format!("{out_name}_k"))?;
                let vc = self.bshd_to_bhsd(in2, b, s_k, h, head_dim, &format!("{out_name}_v"))?;
                let q_canon = bhsd_shape(b, h, s_q, head_dim);
                let k_canon = bhsd_shape(b, h, s_k, head_dim);
                let core = format!("{out_name}_attn");
                self.attention_core(
                    &qc,
                    &kc,
                    &vc,
                    &q_canon,
                    &k_canon,
                    head_dim,
                    mask_kind,
                    mask_in,
                    score_scale,
                    softcap,
                    &core,
                )?;
                // [B,H,Sq,D] → [B,Sq,H,D]
                self.emit(
                    "transpose",
                    out_name,
                    &out_shape,
                    vec![
                        ("x", bind_name(&core)),
                        ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                    ],
                )?;
                self.names.insert(id.0, out_name.to_string());
                return Ok(());
            }
            // Canonical `[B, H, S, D]` (heads at axis 1) or rank-3 `[B, S, D]`.
            let q = self.val(in0);
            let k = self.val(in1);
            let v = self.val(in2);
            let k_shape = self.graph.shape(in1).clone();
            self.attention_core(
                &q,
                &k,
                &v,
                &out_shape,
                &k_shape,
                head_dim,
                mask_kind,
                mask_in,
                score_scale,
                softcap,
                out_name,
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        // ── fused layout: [B,S,H·D] → canonical [B,H,S,D] → attend → fold ──
        if num_heads > 0 && last == num_heads * head_dim && rank == 3 {
            let b = dim_static(&out_shape, 0)?;
            let s_q = dim_static(&out_shape, 1)?;
            let qc =
                self.fused_to_bhsd(in0, b, s_q, num_heads, head_dim, &format!("{out_name}_q"))?;
            let (kc, s_k) = self.fused_to_bhsd_kv(in1, b, head_dim, &format!("{out_name}_k"))?;
            let (vc, _) = self.fused_to_bhsd_kv(in2, b, head_dim, &format!("{out_name}_v"))?;
            let kh = dim_static(&self.graph.shape(in1).clone(), 2)? / head_dim;
            let q_canon = bhsd_shape(b, num_heads, s_q, head_dim);
            let k_canon = bhsd_shape(b, kh, s_k, head_dim);
            let core = format!("{out_name}_attn");
            self.attention_core(
                &qc,
                &kc,
                &vc,
                &q_canon,
                &k_canon,
                head_dim,
                mask_kind,
                mask_in,
                score_scale,
                softcap,
                &core,
            )?;
            // [B,H,Sq,D] → [B,Sq,H,D] → [B,Sq,H·D]
            let t = format!("{out_name}_ot");
            self.emit(
                "transpose",
                &t,
                &bhsd_shape(b, s_q, num_heads, head_dim),
                vec![
                    ("x", bind_name(&core)),
                    ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                ],
            )?;
            self.emit(
                "reshape",
                out_name,
                &out_shape,
                vec![
                    ("x", bind_name(&t)),
                    ("shape", bind_value(vec_i32(&dims_i32(out_shape.dims())))),
                ],
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        Err(CoremlError::Unsupported(format!(
            "attention: last dim {last} is neither head_dim {head_dim} nor \
             num_heads*head_dim {} (rank {rank})",
            num_heads * head_dim
        )))
    }

    /// Transpose a `[B,S,H,D]` operand to canonical `[B,H,S,D]` (perm `[0,2,1,3]`).
    fn bshd_to_bhsd(
        &mut self,
        in_id: NodeId,
        b: usize,
        s: usize,
        h: usize,
        d: usize,
        prefix: &str,
    ) -> Result<String> {
        let x = self.val(in_id);
        let t = format!("{prefix}_bhsd");
        self.emit(
            "transpose",
            &t,
            &bhsd_shape(b, h, s, d),
            vec![
                ("x", bind_name(&x)),
                ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
            ],
        )?;
        Ok(t)
    }

    /// Reshape+transpose a fused `[B,S,H·D]` operand to canonical `[B,H,S,D]`.
    fn fused_to_bhsd(
        &mut self,
        in_id: NodeId,
        b: usize,
        s: usize,
        h: usize,
        d: usize,
        prefix: &str,
    ) -> Result<String> {
        let x = self.val(in_id);
        let r = format!("{prefix}_r");
        self.emit(
            "reshape",
            &r,
            &bhsd_shape(b, s, h, d),
            vec![
                ("x", bind_name(&x)),
                (
                    "shape",
                    bind_value(vec_i32(&[b as i32, s as i32, h as i32, d as i32])),
                ),
            ],
        )?;
        let t = format!("{prefix}_t");
        self.emit(
            "transpose",
            &t,
            &bhsd_shape(b, h, s, d),
            vec![
                ("x", bind_name(&r)),
                ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
            ],
        )?;
        Ok(t)
    }

    /// [`Self::fused_to_bhsd`] deriving head count + seq from the operand shape
    /// (key/value heads may be fewer than `num_heads` before `repeat_kv`).
    fn fused_to_bhsd_kv(
        &mut self,
        in_id: NodeId,
        b: usize,
        d: usize,
        prefix: &str,
    ) -> Result<(String, usize)> {
        let shape = self.graph.shape(in_id).clone();
        let s = dim_static(&shape, 1)?;
        let h = dim_static(&shape, 2)? / d;
        Ok((self.fused_to_bhsd(in_id, b, s, h, d, prefix)?, s))
    }

    /// Canonical scaled-dot-product attention on `[..,Sq,D]` q/k/v. MIL
    /// `matmul` batches leading dims and the masks broadcast, so one core
    /// serves both the split and (pre-canonicalized) fused paths. Writes
    /// `out_name` in `q_shape`; the caller registers the node name.
    #[allow(clippy::too_many_arguments)]
    /// Add the attention mask to scaled scores `[..,Sq,Sk]`, returning the masked
    /// name (or the input unchanged for `None`). Shared by the forward
    /// `attention_core` and the native attention backward so both build `P` from an
    /// identical pre-softmax tensor.
    fn apply_score_mask(
        &mut self,
        scaled: &str,
        scores_shape: &Shape,
        s_q: usize,
        s_k: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        out_name: &str,
    ) -> Result<String> {
        let mut cur = scaled.to_string();
        match mask_kind {
            MaskKind::None => {}
            MaskKind::Causal => {
                let mask_name = format!("{out_name}_mask");
                let mask = causal_mask(s_q, s_k);
                self.operations.push(make_const(
                    &mut self.blob,
                    &mask_name,
                    &Shape::new(&[s_q, s_k], DType::F32),
                    &mask,
                )?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&mask_name))],
                )?;
                cur = masked;
            }
            MaskKind::Bias => {
                let bias = self.val(mask_in.ok_or_else(|| {
                    CoremlError::Unsupported("attention Bias: missing mask input".into())
                })?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&bias))],
                )?;
                cur = masked;
            }
            MaskKind::Custom => {
                // Key-padding mask `[B, S_k]` (1.0 = keep, <0.5 = drop). Turn
                // it into an additive bias (keep→0, drop→-1e9 via (m-1)*1e9)
                // reshaped to `[B, 1, .., 1, S_k]` so it broadcasts over the
                // score tensor's head/query axes. (Batch is carried at the
                // front; per-utterance ASR uses B=1.)
                let mid = mask_in.ok_or_else(|| {
                    CoremlError::Unsupported("attention Custom: missing mask input".into())
                })?;
                let mask = self.val(mid);
                let mask_shape = self.graph.shape(mid).clone();
                let b = dim_static(&mask_shape, 0)?;
                let sub1 = format!("{out_name}_mk_s");
                self.emit(
                    "sub",
                    &sub1,
                    &mask_shape,
                    vec![("x", bind_name(&mask)), ("y", bind_value(scalar_f32(1.0)))],
                )?;
                let bias_flat = format!("{out_name}_mk_a");
                self.emit(
                    "mul",
                    &bias_flat,
                    &mask_shape,
                    vec![("x", bind_name(&sub1)), ("y", bind_value(scalar_f32(1e9)))],
                )?;
                let mut bd = vec![Dim::Static(1); scores_shape.rank()];
                bd[0] = Dim::Static(b);
                let blast = scores_shape.rank() - 1;
                bd[blast] = Dim::Static(s_k);
                let bshape = Shape::from_dims(&bd, DType::F32);
                let bias = format!("{out_name}_mk_b");
                self.emit(
                    "reshape",
                    &bias,
                    &bshape,
                    vec![
                        ("x", bind_name(&bias_flat)),
                        ("shape", bind_value(vec_i32(&dims_i32(&bd)))),
                    ],
                )?;
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&bias))],
                )?;
                cur = masked;
            }
            MaskKind::SlidingWindow(w) => {
                let mask_name = format!("{out_name}_mask");
                let mask = sliding_window_mask(s_q, s_k, w);
                self.operations.push(make_const(
                    &mut self.blob,
                    &mask_name,
                    &Shape::new(&[s_q, s_k], DType::F32),
                    &mask,
                )?);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&mask_name))],
                )?;
                cur = masked;
            }
        }
        Ok(cur)
    }

    fn attention_core(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        q_shape: &Shape,
        k_shape: &Shape,
        head_dim: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        out_name: &str,
    ) -> Result<()> {
        let rank = q_shape.rank();
        let s_q = dim_static(q_shape, rank - 2)?;
        let s_k = dim_static(k_shape, k_shape.rank() - 2)?;
        let scores_shape = {
            let mut d = q_shape.dims().to_vec();
            d[rank - 1] = Dim::Static(s_k); // [..,Sq,Sk]
            Shape::from_dims(&d, DType::F32)
        };
        let scale = score_scale.unwrap_or((head_dim as f32).powf(-0.5));

        // raw = q @ kᵀ  (transpose_y batches over [B,H])
        let raw = format!("{out_name}_qk");
        self.emit(
            "matmul",
            &raw,
            &scores_shape,
            vec![
                ("x", bind_name(q)),
                ("y", bind_name(k)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        // scaled = raw * scale
        let cur = format!("{out_name}_sc");
        self.emit(
            "mul",
            &cur,
            &scores_shape,
            vec![("x", bind_name(&raw)), ("y", bind_value(scalar_f32(scale)))],
        )?;

        // mask (factored so the native attention backward recomputes P identically)
        let mut cur =
            self.apply_score_mask(&cur, &scores_shape, s_q, s_k, mask_kind, mask_in, out_name)?;

        // softcap: cap * tanh(scores / cap)
        if let Some(cap) = softcap {
            if cap > 0.0 {
                let div = format!("{out_name}_cap_div");
                self.emit(
                    "mul",
                    &div,
                    &scores_shape,
                    vec![
                        ("x", bind_name(&cur)),
                        ("y", bind_value(scalar_f32(1.0 / cap))),
                    ],
                )?;
                let th = format!("{out_name}_cap_tanh");
                self.emit("tanh", &th, &scores_shape, vec![("x", bind_name(&div))])?;
                let capped = format!("{out_name}_cap");
                self.emit(
                    "mul",
                    &capped,
                    &scores_shape,
                    vec![("x", bind_name(&th)), ("y", bind_value(scalar_f32(cap)))],
                )?;
                cur = capped;
            }
        }

        // probs = softmax(cur, axis=-1)
        let probs = format!("{out_name}_p");
        self.emit(
            "softmax",
            &probs,
            &scores_shape,
            vec![("x", bind_name(&cur)), ("axis", bind_value(scalar_i32(-1)))],
        )?;

        // out = probs @ v  -> [..,Sq,D]
        self.emit(
            "matmul",
            out_name,
            q_shape,
            vec![
                ("x", bind_name(&probs)),
                ("y", bind_name(v)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(false))),
            ],
        )?;
        Ok(())
    }

    /// Fused scaled-dot-product attention backward (`dQ`/`dK`/`dV`). Canonicalizes
    /// any of the three operand layouts the forward accepts — `[B,H,S,D]`,
    /// `[B,S,H,D]`, fused `[B,S,H·D]` — to `[B,H,S,D]`, runs
    /// [`attention_backward_core`](Self::attention_backward_core) (every mask kind via
    /// the shared [`apply_score_mask`](Self::apply_score_mask)), then maps the
    /// gradient back to the `wrt` operand's layout. MHA only (q/k/v share the head
    /// count); GQA (`kv heads ≠ num_heads`) returns `Unsupported`.
    #[cfg(feature = "training")]
    fn lower_attention_backward(
        &mut self,
        id: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        wrt: rlx_ir::op::AttentionBwdWrt,
        out_name: &str,
    ) -> Result<()> {
        use rlx_ir::op::AttentionBwdWrt;
        let (q_in, k_in, v_in, dy_in, mask_in, q_shape, k_shape, out_shape) = {
            let node = self.graph.node(id);
            let mask_in = match mask_kind {
                MaskKind::Bias | MaskKind::Custom => node.inputs.get(4).copied(),
                _ => None,
            };
            (
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                node.inputs[3],
                mask_in,
                self.graph.shape(node.inputs[0]).clone(),
                self.graph.shape(node.inputs[1]).clone(),
                node.shape.clone(),
            )
        };
        let (h, d) = (num_heads, head_dim);
        let rank = q_shape.rank();
        let last = dim_static(&q_shape, rank - 1)?;
        // Gradient sequence length depends on which operand we differentiate.
        let s_wrt_of = |s_q: usize, s_k: usize| match wrt {
            AttentionBwdWrt::Query => s_q,
            AttentionBwdWrt::Key | AttentionBwdWrt::Value => s_k,
        };

        if rank == 4 && last == d {
            let geom = rlx_ir::attention_geom(&q_shape, &k_shape, num_heads, head_dim);
            let (b, s_q, s_k) = (geom.batch, geom.seq_q, geom.seq_k);
            let k_heads = if geom.bhsd {
                dim_static(&k_shape, 1)?
            } else {
                dim_static(&k_shape, 2)?
            };
            if k_heads != num_heads {
                return Err(CoremlError::Unsupported(
                    "attention backward: GQA (kv heads ≠ num_heads) not supported".into(),
                ));
            }
            if geom.bhsd {
                // Canonical `[B,H,S,D]` — compute straight into `out_name`.
                let (q, k, v, dy) = (
                    self.val(q_in),
                    self.val(k_in),
                    self.val(v_in),
                    self.val(dy_in),
                );
                self.attention_backward_core(
                    &q, &k, &v, &dy, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, out_name,
                )?;
            } else {
                // `[B,S,H,D]` → canonical, compute, transpose the gradient back.
                let qc = self.bshd_to_bhsd(q_in, b, s_q, h, d, &format!("{out_name}_qc"))?;
                let kc = self.bshd_to_bhsd(k_in, b, s_k, h, d, &format!("{out_name}_kc"))?;
                let vc = self.bshd_to_bhsd(v_in, b, s_k, h, d, &format!("{out_name}_vc"))?;
                let dyc = self.bshd_to_bhsd(dy_in, b, s_q, h, d, &format!("{out_name}_dyc"))?;
                let core = format!("{out_name}_core");
                self.attention_backward_core(
                    &qc, &kc, &vc, &dyc, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, &core,
                )?;
                let s_wrt = s_wrt_of(s_q, s_k);
                self.emit(
                    "transpose",
                    out_name,
                    &bhsd_shape(b, s_wrt, h, d),
                    vec![
                        ("x", bind_name(&core)),
                        ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                    ],
                )?;
            }
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        if rank == 3 && num_heads > 0 && last == num_heads * head_dim {
            // Fused `[B,S,H·D]` → canonical, compute, transpose + reshape back.
            let b = dim_static(&q_shape, 0)?;
            let s_q = dim_static(&q_shape, 1)?;
            let s_k = dim_static(&k_shape, 1)?;
            if dim_static(&k_shape, 2)? / d != num_heads {
                return Err(CoremlError::Unsupported(
                    "attention backward: fused GQA (kv heads ≠ num_heads) not supported".into(),
                ));
            }
            let qc = self.fused_to_bhsd(q_in, b, s_q, h, d, &format!("{out_name}_qc"))?;
            let kc = self.fused_to_bhsd(k_in, b, s_k, h, d, &format!("{out_name}_kc"))?;
            let vc = self.fused_to_bhsd(v_in, b, s_k, h, d, &format!("{out_name}_vc"))?;
            let dyc = self.fused_to_bhsd(dy_in, b, s_q, h, d, &format!("{out_name}_dyc"))?;
            let core = format!("{out_name}_core");
            self.attention_backward_core(
                &qc, &kc, &vc, &dyc, b, h, s_q, s_k, d, mask_kind, mask_in, wrt, &core,
            )?;
            let s_wrt = s_wrt_of(s_q, s_k);
            let t = format!("{out_name}_t");
            self.emit(
                "transpose",
                &t,
                &bhsd_shape(b, s_wrt, h, d),
                vec![
                    ("x", bind_name(&core)),
                    ("perm", bind_value(vec_i32(&[0, 2, 1, 3]))),
                ],
            )?;
            self.emit(
                "reshape",
                out_name,
                &out_shape,
                vec![
                    ("x", bind_name(&t)),
                    (
                        "shape",
                        bind_value(vec_i32(&[b as i32, s_wrt as i32, (h * d) as i32])),
                    ),
                ],
            )?;
            self.names.insert(id.0, out_name.to_string());
            return Ok(());
        }

        Err(CoremlError::Unsupported(format!(
            "attention backward: unsupported operand layout (rank {rank}, last {last})"
        )))
    }

    /// Canonical `[B,H,S,D]` attention backward, emitting the single `wrt` gradient
    /// to `result`. Recompute `P = softmax(scale·QKᵀ [+ mask])`, then:
    ///   `dV = Pᵀ·dO`,  `dP = dO·Vᵀ`,
    ///   `ds = scale · P⊙(dP − rowsum(P⊙dP))` (softmax-Jacobian–vector product),
    ///   `dQ = ds·K`,  `dK = dsᵀ·Q`.
    /// Masked positions get `P≈0`, so `ds` there vanishes automatically.
    #[cfg(feature = "training")]
    #[allow(clippy::too_many_arguments)]
    fn attention_backward_core(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        dy: &str,
        b: usize,
        h: usize,
        s_q: usize,
        s_k: usize,
        d: usize,
        mask_kind: MaskKind,
        mask_in: Option<NodeId>,
        wrt: rlx_ir::op::AttentionBwdWrt,
        result: &str,
    ) -> Result<()> {
        use rlx_ir::op::AttentionBwdWrt;
        let scores_shape = bhsd_shape(b, h, s_q, s_k);
        let scale = (d as f32).powf(-0.5);

        let raw = format!("{result}_qk");
        self.emit(
            "matmul",
            &raw,
            &scores_shape,
            vec![
                ("x", bind_name(q)),
                ("y", bind_name(k)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        let scaled = format!("{result}_scl");
        self.emit(
            "mul",
            &scaled,
            &scores_shape,
            vec![("x", bind_name(&raw)), ("y", bind_value(scalar_f32(scale)))],
        )?;
        let pre =
            self.apply_score_mask(&scaled, &scores_shape, s_q, s_k, mask_kind, mask_in, result)?;
        let p = format!("{result}_p");
        self.emit(
            "softmax",
            &p,
            &scores_shape,
            vec![("x", bind_name(&pre)), ("axis", bind_value(scalar_i32(-1)))],
        )?;

        match wrt {
            AttentionBwdWrt::Value => {
                // dV = Pᵀ · dO  → [B,H,Sk,D]
                self.emit(
                    "matmul",
                    result,
                    &bhsd_shape(b, h, s_k, d),
                    vec![
                        ("x", bind_name(&p)),
                        ("y", bind_name(dy)),
                        ("transpose_x", bind_value(scalar_bool(true))),
                        ("transpose_y", bind_value(scalar_bool(false))),
                    ],
                )?;
            }
            AttentionBwdWrt::Query | AttentionBwdWrt::Key => {
                let dp = format!("{result}_dp");
                self.emit(
                    "matmul",
                    &dp,
                    &scores_shape,
                    vec![
                        ("x", bind_name(dy)),
                        ("y", bind_name(v)),
                        ("transpose_x", bind_value(scalar_bool(false))),
                        ("transpose_y", bind_value(scalar_bool(true))),
                    ],
                )?;
                let pdp = format!("{result}_pdp");
                self.emit(
                    "mul",
                    &pdp,
                    &scores_shape,
                    vec![("x", bind_name(&dp)), ("y", bind_name(&p))],
                )?;
                let rowsum = format!("{result}_rs");
                self.emit(
                    "reduce_sum",
                    &rowsum,
                    &bhsd_shape(b, h, s_q, 1),
                    vec![
                        ("x", bind_name(&pdp)),
                        ("axes", bind_value(vec_i32(&[3]))),
                        ("keep_dims", bind_value(scalar_bool(true))),
                    ],
                )?;
                let dpm = format!("{result}_dpm");
                self.emit(
                    "sub",
                    &dpm,
                    &scores_shape,
                    vec![("x", bind_name(&dp)), ("y", bind_name(&rowsum))],
                )?;
                let dsm = format!("{result}_dsm");
                self.emit(
                    "mul",
                    &dsm,
                    &scores_shape,
                    vec![("x", bind_name(&p)), ("y", bind_name(&dpm))],
                )?;
                let ds = format!("{result}_ds");
                self.emit(
                    "mul",
                    &ds,
                    &scores_shape,
                    vec![("x", bind_name(&dsm)), ("y", bind_value(scalar_f32(scale)))],
                )?;
                match wrt {
                    AttentionBwdWrt::Query => {
                        // dQ = ds · K  → [B,H,Sq,D]
                        self.emit(
                            "matmul",
                            result,
                            &bhsd_shape(b, h, s_q, d),
                            vec![
                                ("x", bind_name(&ds)),
                                ("y", bind_name(k)),
                                ("transpose_x", bind_value(scalar_bool(false))),
                                ("transpose_y", bind_value(scalar_bool(false))),
                            ],
                        )?;
                    }
                    AttentionBwdWrt::Key => {
                        // dK = dsᵀ · Q  → [B,H,Sk,D]
                        self.emit(
                            "matmul",
                            result,
                            &bhsd_shape(b, h, s_k, d),
                            vec![
                                ("x", bind_name(&ds)),
                                ("y", bind_name(q)),
                                ("transpose_x", bind_value(scalar_bool(true))),
                                ("transpose_y", bind_value(scalar_bool(false))),
                            ],
                        )?;
                    }
                    AttentionBwdWrt::Value => unreachable!(),
                }
            }
        }
        Ok(())
    }

    /// Inference batch norm with frozen stats. Inputs `[x, gamma, beta,
    /// mean, var]`, channel-last: `(x - mean)·rsqrt(var+eps)·gamma + beta`,
    /// all per-channel `[C]` broadcasting over the trailing axis.
    fn lower_batch_norm(&mut self, id: NodeId, eps: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let c = dim_static(&shape, shape.rank() - 1)?;
        let cs = Shape::new(&[c], DType::F32);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);
        let mean = self.val(node.inputs[3]);
        let var = self.val(node.inputs[4]);

        let veps = format!("{out_name}_veps");
        self.emit(
            "add",
            &veps,
            &cs,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out_name}_inv");
        self.emit(
            "rsqrt",
            &inv,
            &cs,
            vec![
                ("x", bind_name(&veps)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let xc = format!("{out_name}_xc");
        self.emit(
            "sub",
            &xc,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&mean))],
        )?;
        let t = format!("{out_name}_t");
        self.emit(
            "mul",
            &t,
            &shape,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?;
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t2,
            &shape,
            vec![("x", bind_name(&t)), ("y", bind_name(&gamma))],
        )?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&t2)), ("y", bind_name(&beta))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// GroupNorm over NCHW. Inputs `[x, gamma, beta]`; normalises over
    /// `(C/G)·H·W` within each of `G` groups, then per-channel affine.
    fn lower_group_norm(
        &mut self,
        id: NodeId,
        groups: usize,
        eps: f32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let d = static_dims(&shape)?;
        if d.len() != 4 {
            return Err(CoremlError::Unsupported("group_norm: only NCHW".into()));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        let inner = (c / groups as i64) * h * w;
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);

        let grp = Shape::new(&[n as usize, groups, inner as usize], DType::F32);
        let red = Shape::new(&[n as usize, groups, 1], DType::F32);
        let xr = format!("{out_name}_xr");
        self.reshape_to(&x, &[n, groups as i64, inner], &grp, &xr)?;
        let normb = self.normalize_chain(out_name, &xr, &grp, &red, &[2], eps)?;
        // back to NCHW
        let nb = format!("{out_name}_nb");
        self.reshape_to(&normb, &[n, c, h, w], &shape, &nb)?;
        self.affine_nchw(out_name, &nb, &shape, &gamma, &beta, c)
    }

    /// LayerNorm over the channel axis of NCHW (per spatial position).
    /// Inputs `[x, gamma, beta]`.
    fn lower_layer_norm2d(&mut self, id: NodeId, eps: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let d = static_dims(&shape)?;
        if d.len() != 4 {
            return Err(CoremlError::Unsupported("layer_norm2d: only NCHW".into()));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        let red = Shape::new(&[n as usize, 1, h as usize, w as usize], DType::F32);
        let x = self.val(node.inputs[0]);
        let gamma = self.val(node.inputs[1]);
        let beta = self.val(node.inputs[2]);
        let norm = self.normalize_chain(out_name, &x, &shape, &red, &[1], eps)?;
        self.affine_nchw(out_name, &norm, &shape, &gamma, &beta, c)
    }

    /// `(in - mean)·rsqrt(var+eps)` reducing over `axes` (keep dims).
    /// Returns the normalised value name.
    fn normalize_chain(
        &mut self,
        out: &str,
        input: &str,
        full: &Shape,
        red: &Shape,
        axes: &[i32],
        eps: f32,
    ) -> Result<String> {
        let mean = format!("{out}_mean");
        self.emit(
            "reduce_mean",
            &mean,
            red,
            vec![
                ("x", bind_name(input)),
                ("axes", bind_value(vec_i32(axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let xc = format!("{out}_nc");
        self.emit(
            "sub",
            &xc,
            full,
            vec![("x", bind_name(input)), ("y", bind_name(&mean))],
        )?;
        let sq = format!("{out}_sq");
        self.emit(
            "mul",
            &sq,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&xc))],
        )?;
        let var = format!("{out}_var");
        self.emit(
            "reduce_mean",
            &var,
            red,
            vec![
                ("x", bind_name(&sq)),
                ("axes", bind_value(vec_i32(axes))),
                ("keep_dims", bind_value(scalar_bool(true))),
            ],
        )?;
        let veps = format!("{out}_veps");
        self.emit(
            "add",
            &veps,
            red,
            vec![("x", bind_name(&var)), ("y", bind_value(scalar_f32(eps)))],
        )?;
        let inv = format!("{out}_ninv");
        self.emit(
            "rsqrt",
            &inv,
            red,
            vec![
                ("x", bind_name(&veps)),
                ("epsilon", bind_value(scalar_f32(0.0))),
            ],
        )?;
        let norm = format!("{out}_norm");
        self.emit(
            "mul",
            &norm,
            full,
            vec![("x", bind_name(&xc)), ("y", bind_name(&inv))],
        )?;
        Ok(norm)
    }

    /// Per-channel affine for NCHW: `out = norm·γ[1,C,1,1] + β[1,C,1,1]`.
    fn affine_nchw(
        &mut self,
        out_name: &str,
        norm: &str,
        shape: &Shape,
        gamma: &str,
        beta: &str,
        c: i64,
    ) -> Result<()> {
        let g4 = format!("{out_name}_g4");
        let b4 = format!("{out_name}_b4");
        let c4 = Shape::new(&[1, c as usize, 1, 1], DType::F32);
        self.reshape_to(gamma, &[1, c, 1, 1], &c4, &g4)?;
        self.reshape_to(beta, &[1, c, 1, 1], &c4, &b4)?;
        let scaled = format!("{out_name}_sc");
        self.emit(
            "mul",
            &scaled,
            shape,
            vec![("x", bind_name(norm)), ("y", bind_name(&g4))],
        )?;
        self.emit(
            "add",
            out_name,
            shape,
            vec![("x", bind_name(&scaled)), ("y", bind_name(&b4))],
        )?;
        // Caller registers the node mapping.
        Ok(())
    }

    fn reshape_to(&mut self, src: &str, dims: &[i64], out_shape: &Shape, dst: &str) -> Result<()> {
        let s: Vec<i32> = dims.iter().map(|&v| v as i32).collect();
        self.emit(
            "reshape",
            dst,
            out_shape,
            vec![("x", bind_name(src)), ("shape", bind_value(vec_i32(&s)))],
        )
    }

    /// LoRA: `out = x·W + scale·(x·A)·B`. Inputs `[x, W, A, B]`.
    fn lower_lora_matmul(&mut self, id: NodeId, scale: f32, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let m = dim_static(&shape, 0)?;
        let n = dim_static(&shape, 1)?;
        let x = self.val(node.inputs[0]);
        let w = self.val(node.inputs[1]);
        let a = self.val(node.inputs[2]);
        let b = self.val(node.inputs[3]);
        let r = dim_static(&self.graph.shape(node.inputs[2]).clone(), 1)?;

        let xa = format!("{out_name}_xa");
        self.matmul(&xa, &x, &a, &Shape::new(&[m, r], DType::F32))?;
        let xab = format!("{out_name}_xab");
        self.matmul(&xab, &xa, &b, &shape)?;
        let scaled = format!("{out_name}_lora");
        self.emit(
            "mul",
            &scaled,
            &shape,
            vec![("x", bind_name(&xab)), ("y", bind_value(scalar_f32(scale)))],
        )?;
        let xw = format!("{out_name}_xw");
        self.matmul(&xw, &x, &w, &shape)?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&xw)), ("y", bind_name(&scaled))],
        )?;
        let _ = n;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    fn matmul(&mut self, dst: &str, x: &str, y: &str, out_shape: &Shape) -> Result<()> {
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

    /// 2D conv / conv_transpose, NCHW. Inputs `[x, weight]` (no bias in IR).
    #[allow(clippy::too_many_arguments)]
    fn lower_conv(
        &mut self,
        id: NodeId,
        transpose: bool,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        _output_padding: &[usize],
        groups: usize,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let w_shape = self.graph.shape(node.inputs[1]).clone();
        // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis and the length in W
        // (`[N,C,1,L]`, weight `[Co,Ci,k,1]`). CoreML's 2D conv would run the k-tap
        // kernel over the singleton H axis, so collapse to a real rank-3 1D conv over
        // the length (matching CPU/MLX/wgpu and onnxruntime).
        // 1D conv packed as 2D NCHW with ONE singleton spatial axis and the
        // length in the other — either `[N,C,1,L]` (length-in-W) or
        // `[N,C,L,1]` (length-in-H, e.g. Whisper's conv frontend) — with
        // weight `[Co,Ci,k,1]`. Collapse to a real rank-3 1D conv over the
        // length (matches CPU/MLX/wgpu/onnxruntime); CoreML's 2D conv over a
        // singleton/degenerate spatial axis is not reliable here.
        let in_h = in_shape.dim(2).unwrap_static();
        let in_w = in_shape.dim(3).unwrap_static();
        let one_d = !transpose
            && in_shape.rank() == 4
            && w_shape.rank() == 4
            && w_shape.dim(3).unwrap_static() == 1
            && w_shape.dim(2).unwrap_static() > 1
            && (in_h == 1 || in_w == 1);
        if one_d {
            let n = in_shape.dim(0).unwrap_static() as i32;
            let c = in_shape.dim(1).unwrap_static() as i32;
            // length is the non-singleton input spatial axis
            let l = if in_h == 1 { in_w } else { in_h } as i32;
            let co = w_shape.dim(0).unwrap_static() as i32;
            let ci = w_shape.dim(1).unwrap_static() as i32;
            let k = w_shape.dim(2).unwrap_static() as i32;
            let out_h = shape.dim(2).unwrap_static();
            let out_w = shape.dim(3).unwrap_static();
            let lo = if out_h == 1 { out_w } else { out_h } as i32;
            let xr = format!("{out_name}_x1d");
            self.emit(
                "reshape",
                &xr,
                &Shape::new(&[n as usize, c as usize, l as usize], DType::F32),
                vec![
                    ("x", bind_name(&self.val(node.inputs[0]))),
                    ("shape", bind_value(vec_i32(&[n, c, l]))),
                ],
            )?;
            let wr = format!("{out_name}_w1d");
            self.emit(
                "reshape",
                &wr,
                &Shape::new(&[co as usize, ci as usize, k as usize], DType::F32),
                vec![
                    ("x", bind_name(&self.val(node.inputs[1]))),
                    ("shape", bind_value(vec_i32(&[co, ci, k]))),
                ],
            )?;
            let cout = format!("{out_name}_c1d");
            self.emit(
                "conv",
                &cout,
                &Shape::new(&[n as usize, co as usize, lo as usize], DType::F32),
                vec![
                    ("x", bind_name(&xr)),
                    ("weight", bind_name(&wr)),
                    ("strides", bind_value(vec_i32(&[stride[0] as i32]))),
                    ("pad_type", bind_value(scalar_str("custom"))),
                    ("pad", bind_value(vec_i32(&pad_begin_end(&[padding[0]])))),
                    ("dilations", bind_value(vec_i32(&[dilation[0] as i32]))),
                    ("groups", bind_value(scalar_i32(groups as i32))),
                ],
            )?;
            let out_dims: Vec<i32> = static_dims(&shape)?.iter().map(|&v| v as i32).collect();
            let op = self.simple_op(
                "reshape",
                out_name,
                &shape,
                vec![
                    ("x", bind_name(&cout)),
                    ("shape", bind_value(vec_i32(&out_dims))),
                ],
            )?;
            self.push_named(id, out_name.to_string(), op);
            return Ok(());
        }
        let x = self.val(node.inputs[0]);
        let w = self.val(node.inputs[1]);
        let strides = vec_usize_i32(stride);
        let dilations = vec_usize_i32(dilation);
        let pad = pad_begin_end(padding);
        let ty = if transpose { "conv_transpose" } else { "conv" };
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("weight", bind_name(&w)),
            ("strides", bind_value(vec_i32(&strides))),
            ("pad_type", bind_value(scalar_str("custom"))),
            ("pad", bind_value(vec_i32(&pad))),
            ("dilations", bind_value(vec_i32(&dilations))),
            ("groups", bind_value(scalar_i32(groups as i32))),
        ];
        if transpose {
            // conv_transpose needs the explicit output shape to resolve the
            // fractionally-strided ambiguity.
            let out_dims: Vec<i32> = static_dims(&shape)?.iter().map(|&v| v as i32).collect();
            binds.push(("output_shape", bind_value(vec_i32(&out_dims))));
        }
        let op = self.simple_op(ty, out_name, &shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Conv2d backward w.r.t. weight (NCHW, groups == 1). The weight gradient is a
    /// convolution of the input by the upstream gradient: with N folded as the
    /// contraction channel and the gradient as the (stride-dilated) kernel,
    ///   dWᵀ = conv(xᵀ[Cin,N,H,W], dyᵀ[Cout,N,Hout,Wout], dilation = forward stride)
    /// gives [Cin,Cout,kh,kw]; transpose back to [Cout,Cin,kh,kw]. Inputs [x, dy].
    #[cfg(feature = "training")]
    fn lower_conv2d_backward_weight(
        &mut self,
        id: NodeId,
        stride: &[usize],
        padding: &[usize],
        groups: usize,
        out_name: &str,
    ) -> Result<()> {
        if groups != 1 {
            return Err(CoremlError::Unsupported(
                "conv2d backward weight: only groups == 1".into(),
            ));
        }
        let node = self.graph.node(id);
        let x = self.val(node.inputs[0]);
        let dy = self.val(node.inputs[1]);
        let x_shape = self.graph.shape(node.inputs[0]).clone();
        let dy_shape = self.graph.shape(node.inputs[1]).clone();
        let out_shape = node.shape.clone();
        let dt = out_shape.dtype();
        let xd = |i: usize| x_shape.dim(i).unwrap_static();
        let dd = |i: usize| dy_shape.dim(i).unwrap_static();
        let od = |i: usize| out_shape.dim(i).unwrap_static();
        let (n, cin, h, w) = (xd(0), xd(1), xd(2), xd(3));
        let (cout, hout, wout) = (dd(1), dd(2), dd(3));
        let (kh, kw) = (od(2), od(3));
        let perm = || bind_value(vec_i32(&[1, 0, 2, 3]));

        // xᵀ = [Cin, N, H, W], dyᵀ = [Cout, N, Hout, Wout]
        let xt = format!("{out_name}_xt");
        self.emit(
            "transpose",
            &xt,
            &Shape::new(&[cin, n, h, w], dt),
            vec![("x", bind_name(&x)), ("perm", perm())],
        )?;
        let dyt = format!("{out_name}_dyt");
        self.emit(
            "transpose",
            &dyt,
            &Shape::new(&[cout, n, hout, wout], dt),
            vec![("x", bind_name(&dy)), ("perm", perm())],
        )?;
        // conv(xᵀ, dyᵀ): kernel = dyᵀ over the N contraction channel, dilated by the
        // forward stride so it samples the strided receptive field → [Cin,Cout,kh,kw].
        let dwt = format!("{out_name}_dwt");
        self.emit(
            "conv",
            &dwt,
            &Shape::new(&[cin, cout, kh, kw], dt),
            vec![
                ("x", bind_name(&xt)),
                ("weight", bind_name(&dyt)),
                ("strides", bind_value(vec_i32(&[1, 1]))),
                ("pad_type", bind_value(scalar_str("custom"))),
                ("pad", bind_value(vec_i32(&pad_begin_end(padding)))),
                (
                    "dilations",
                    bind_value(vec_i32(&[stride[0] as i32, stride[1] as i32])),
                ),
                ("groups", bind_value(scalar_i32(1))),
            ],
        )?;
        let op = self.simple_op(
            "transpose",
            out_name,
            &out_shape,
            vec![("x", bind_name(&dwt)), ("perm", perm())],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// 2D max/avg pool, NCHW. Avg divides by the full window (pad counts).
    fn lower_pool(
        &mut self,
        id: NodeId,
        kind: ReduceOp,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let x = self.val(node.inputs[0]);
        let ty = match kind {
            ReduceOp::Max => "max_pool",
            ReduceOp::Mean => "avg_pool",
            other => return Err(CoremlError::Unsupported(format!("pool {other:?}"))),
        };
        let mut binds = vec![
            ("x", bind_name(&x)),
            ("kernel_sizes", bind_value(vec_i32(&vec_usize_i32(kernel)))),
            ("strides", bind_value(vec_i32(&vec_usize_i32(stride)))),
            ("pad_type", bind_value(scalar_str("custom"))),
            ("pad", bind_value(vec_i32(&pad_begin_end(padding)))),
            ("ceil_mode", bind_value(scalar_bool(false))),
        ];
        if matches!(kind, ReduceOp::Mean) {
            binds.push((
                "exclude_padding_from_average",
                bind_value(scalar_bool(false)),
            ));
        }
        let op = self.simple_op(ty, out_name, &shape, binds)?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Top-k along the last axis. The IR result is the f32-encoded
    /// **indices** (descending by value, ties → smaller index), so we emit
    /// MIL `topk` (a two-output op: values + indices) and cast the int32
    /// indices to f32.
    fn lower_topk(&mut self, id: NodeId, k: usize, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // [..., k], f32 (indices encoded)
        let x = self.val(node.inputs[0]);
        let axis = (shape.rank() - 1) as i32;

        let values = format!("{out_name}_vals");
        let indices = format!("{out_name}_idx_i32");
        let vals_ty = named_value_type(&values, &shape)?;
        let idx_ty = named_value_type(&indices, &shape.clone().with_dtype(DType::I32))?;

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), bind_name(&x));
        inputs.insert("k".to_string(), bind_value(scalar_i32(k as i32)));
        inputs.insert("axis".to_string(), bind_value(scalar_i32(axis)));
        inputs.insert("ascending".to_string(), bind_value(scalar_bool(false)));
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), scalar_str(out_name));
        self.operations.push(proto::Operation {
            r#type: "topk".to_string(),
            inputs,
            outputs: vec![vals_ty, idx_ty],
            blocks: vec![],
            attributes,
        });

        // Cast int32 indices → f32 to match the IR's encoding.
        self.emit(
            "cast",
            out_name,
            &shape,
            vec![
                ("x", bind_name(&indices)),
                ("dtype", bind_value(scalar_str("fp32"))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// `ArgMax` / `ArgMin` along one axis; indices are f32-encoded at the IR boundary.
    fn lower_argreduce(
        &mut self,
        id: NodeId,
        axis: usize,
        keep_dim: bool,
        is_max: bool,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let rank = in_shape.rank();
        let ax = axis as i32;
        let _ = rank;
        let mut x = self.val(node.inputs[0]);
        if !is_max {
            let neg = format!("{out_name}_neg");
            self.emit(
                "mul",
                &neg,
                &in_shape,
                vec![("x", bind_name(&x)), ("y", bind_value(scalar_f32(-1.0)))],
            )?;
            x = neg;
        }
        let idx_i32 = format!("{out_name}_idx_i32");
        let idx_shape = node.shape.clone().with_dtype(DType::I32);
        self.emit(
            "reduce_argmax",
            &idx_i32,
            &idx_shape,
            vec![
                ("x", bind_name(&x)),
                ("axis", bind_value(scalar_i32(ax))),
                ("keep_dims", bind_value(scalar_bool(keep_dim))),
            ],
        )?;
        self.emit(
            "cast",
            out_name,
            &node.shape,
            vec![
                ("x", bind_name(&idx_i32)),
                ("dtype", bind_value(scalar_str("fp32"))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Batch-general flip along `axes` via per-axis `gather` with reversed indices.
    fn lower_reverse(&mut self, id: NodeId, axes: &[usize], out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        if axes.is_empty() {
            let x = self.val(node.inputs[0]);
            self.names.insert(id.0, x);
            return Ok(());
        }
        let mut cur = self.val(node.inputs[0]);
        let shape = in_shape.clone();
        for &ax in axes {
            let d = dim_static(&shape, ax)?;
            let idx_f: Vec<f32> = (0..d).rev().map(|i| i as f32).collect();
            let idx_name = format!("{out_name}_rev_{ax}");
            self.operations.push(make_const(
                &mut self.blob,
                &idx_name,
                &Shape::new(&[d], DType::F32),
                &idx_f,
            )?);
            let idx_i32 = format!("{idx_name}_i32");
            self.emit(
                "cast",
                &idx_i32,
                &Shape::new(&[d], DType::I32),
                vec![
                    ("x", bind_name(&idx_name)),
                    ("dtype", bind_value(scalar_str("int32"))),
                ],
            )?;
            let next = format!("{out_name}_g{ax}");
            self.emit(
                "gather",
                &next,
                &shape,
                vec![
                    ("x", bind_name(&cur)),
                    ("indices", bind_name(&idx_i32)),
                    ("axis", bind_value(scalar_i32(ax as i32))),
                ],
            )?;
            cur = next;
        }
        self.names.insert(id.0, cur);
        Ok(())
    }

    /// 2D axial RoPE (SAM2-style), input `[B, seq, num_heads·head_dim]`.
    /// Interleaved-pair rotation: first half rotated by the x-position
    /// angle, second half by y. All angle tables are baked at lowering
    /// time, then applied as `x·cos + rot_interleaved(x)·sin`, where
    /// `rot_interleaved` maps each pair `(a,b) → (-b, a)`.
    #[allow(clippy::too_many_arguments)]
    fn lower_axial_rope2d(
        &mut self,
        id: NodeId,
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        num_heads: usize,
        theta: f32,
        repeat_factor: usize,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        if shape.rank() != 3 {
            return Err(CoremlError::Unsupported(
                "axial_rope2d: only [B, seq, H*D]".into(),
            ));
        }
        let b = dim_static(&shape, 0)?;
        let seq = dim_static(&shape, 1)?;
        let hd = dim_static(&shape, 2)?; // num_heads * head_dim

        // Bake cos/sin tables [seq, hd] (duplicated per interleaved pair).
        let (cos_full, sin_full) = axial_tables(
            end_x,
            end_y,
            head_dim,
            num_heads,
            theta,
            repeat_factor,
            seq,
            hd,
        );
        let tab_shape = Shape::new(&[seq, hd], DType::F32);
        let cosf = format!("{out_name}_cos");
        let sinf = format!("{out_name}_sin");
        self.operations
            .push(make_const(&mut self.blob, &cosf, &tab_shape, &cos_full)?);
        self.operations
            .push(make_const(&mut self.blob, &sinf, &tab_shape, &sin_full)?);

        let x = self.val(node.inputs[0]);
        // rot_interleaved: reshape to pairs, swap+negate, reshape back.
        let pair_shape = Shape::new(&[b, seq, hd / 2, 2], DType::F32);
        let one_shape = Shape::new(&[b, seq, hd / 2, 1], DType::F32);
        let xr = format!("{out_name}_xr");
        self.reshape_to(
            &x,
            &[b as i64, seq as i64, (hd / 2) as i64, 2],
            &pair_shape,
            &xr,
        )?;
        let even = format!("{out_name}_even");
        let odd = format!("{out_name}_odd");
        self.slice_last(&xr, 4, 0, 1, &one_shape, &even)?;
        self.slice_last(&xr, 4, 1, 1, &one_shape, &odd)?;
        let neg_odd = format!("{out_name}_nodd");
        self.emit(
            "mul",
            &neg_odd,
            &one_shape,
            vec![("x", bind_name(&odd)), ("y", bind_value(scalar_f32(-1.0)))],
        )?;
        let rot4 = format!("{out_name}_rot4");
        self.emit(
            "concat",
            &rot4,
            &pair_shape,
            vec![
                ("values", bind_names(&[neg_odd, even])),
                ("axis", bind_value(scalar_i32(3))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let rot = format!("{out_name}_rot");
        self.reshape_to(&rot4, &[b as i64, seq as i64, hd as i64], &shape, &rot)?;

        // out = x*cos + rot*sin
        let t1 = format!("{out_name}_t1");
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t1,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&cosf))],
        )?;
        self.emit(
            "mul",
            &t2,
            &shape,
            vec![("x", bind_name(&rot)), ("y", bind_name(&sinf))],
        )?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// MoE grouped matmul. Inputs `[input(M,K), weight(E,K,N),
    /// expert_idx(M)]`: each row picks its expert's weight slab.
    /// `out[m] = input[m] · weight[expert_idx[m]]`. Lowered as
    /// gather-then-batched-matmul.
    fn lower_grouped_matmul(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // [M, N]
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&shape, shape.rank() - 1)?;

        let input = self.val(node.inputs[0]);
        let weight = self.val(node.inputs[1]);
        let eidx = self.val(node.inputs[2]);

        // expert_idx f32 → int32
        let eidx_i32 = format!("{out_name}_eidx");
        let eidx_shape = self
            .graph
            .shape(node.inputs[2])
            .clone()
            .with_dtype(DType::I32);
        self.emit(
            "cast",
            &eidx_i32,
            &eidx_shape,
            vec![
                ("x", bind_name(&eidx)),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        // W_sel = gather(weight, eidx, axis=0) -> [M, K, N]
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, k, n], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        // input [M,K] -> [M,1,K]
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        // batched matmul [M,1,K] @ [M,K,N] -> [M,1,N]
        let mm = format!("{out_name}_mm");
        self.matmul(&mm, &in3, &wsel, &Shape::new(&[m, 1, n], DType::F32))?;
        // -> [M, N]
        self.reshape_to(&mm, &[m as i64, n as i64], &shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Fetch the GGUF/quantized bytes for the `Param` weight at `w_id`.
    fn quant_bytes(&self, w_id: NodeId) -> Result<&[u8]> {
        match &self.graph.node(w_id).op {
            Op::Param { name } => self
                .typed_params
                .get(name)
                .map(|(b, _)| b.as_slice())
                .ok_or_else(|| CoremlError::Runtime(format!("missing quantized param '{name}'"))),
            Op::Constant { data } => Ok(data.as_slice()),
            other => Err(CoremlError::Unsupported(format!(
                "dequant weight must be a Param/Constant, got {other:?}"
            ))),
        }
    }

    /// Bake on-device dequantized weights `[n,k]` as MIL constants + `mul`/`sub`.
    ///
    /// Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4NL, Q4/5/8_K, Q2/3/6_K. Scale tensors are
    /// `[nb,1]` or `[nb,32]` depending on scheme (see `split_gguf_ondevice`).
    /// Documented in [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
    fn bake_ondevice_weight(
        &mut self,
        prefix: &str,
        scheme: QuantScheme,
        bytes: &[u8],
        n: usize,
        k: usize,
    ) -> Result<String> {
        const QK: usize = 32;
        let nb = (k * n) / QK;
        if nb * QK != k * n {
            return Err(CoremlError::Runtime(format!(
                "ondevice dequant: {n}x{k} not divisible by {QK}"
            )));
        }
        let (qs, scales, offsets) = split_gguf_ondevice(scheme, bytes, nb)?;
        let per_elem_scales = scales.len() == nb * QK;
        let sc_shape = if per_elem_scales {
            Shape::new(&[nb, QK], DType::F32)
        } else {
            Shape::new(&[nb, 1], DType::F32)
        };
        let q_name = format!("{prefix}_q");
        self.operations.push(make_const(
            &mut self.blob,
            &q_name,
            &Shape::new(&[nb, QK], DType::F32),
            &qs,
        )?);
        let sc_name = format!("{prefix}_sc");
        self.operations
            .push(make_const(&mut self.blob, &sc_name, &sc_shape, &scales)?);
        let mul_name = format!("{prefix}_mul");
        self.emit(
            "mul",
            &mul_name,
            &Shape::new(&[nb, QK], DType::F32),
            vec![("x", bind_name(&q_name)), ("y", bind_name(&sc_name))],
        )?;
        let dq = if offsets.iter().any(|&o| o != 0.0) {
            let per_elem_offsets = offsets.len() == nb * QK;
            let off_shape = if per_elem_offsets {
                Shape::new(&[nb, QK], DType::F32)
            } else {
                Shape::new(&[nb, 1], DType::F32)
            };
            let off_name = format!("{prefix}_off");
            self.operations
                .push(make_const(&mut self.blob, &off_name, &off_shape, &offsets)?);
            let sub_name = format!("{prefix}_dq");
            self.emit(
                "sub",
                &sub_name,
                &Shape::new(&[nb, QK], DType::F32),
                vec![("x", bind_name(&mul_name)), ("y", bind_name(&off_name))],
            )?;
            sub_name
        } else {
            mul_name
        };
        let wc = format!("{prefix}_w");
        self.reshape_to(
            &dq,
            &[n as i64, k as i64],
            &Shape::new(&[n, k], DType::F32),
            &wc,
        )?;
        Ok(wc)
    }

    /// On-device block dequant for supported GGUF schemes, then MIL matmul.
    fn lower_dequant_matmul_ondevice(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let x_id = node.inputs[0];
        let w_id = node.inputs[1];
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let m = out_shape.num_elements().unwrap_or(0) / n.max(1);
        let k = self.graph.shape(x_id).num_elements().unwrap_or(0) / m.max(1);
        let bytes = self.quant_bytes(w_id)?.to_vec();
        let wc = self.bake_ondevice_weight(out_name, scheme, &bytes, n, k)?;
        let x = self.val(x_id);
        let op = self.simple_op(
            "matmul",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("y", bind_name(&wc)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// `x @ dequant(W)ᵀ`. GGUF weights are stored `[N, K]` (B-transposed),
    /// so we host-dequantize to f32 `[N, K]`, bake it, and matmul with
    /// `transpose_y`. The dequant happens at finalize (weights present),
    /// trading the proto's on-device dequant for size — correct + simple.
    fn lower_dequant_matmul(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let x_id = node.inputs[0];
        let w_id = node.inputs[1];
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let m = out_shape.num_elements().unwrap_or(0) / n.max(1);
        let k = self.graph.shape(x_id).num_elements().unwrap_or(0) / m.max(1);

        let wf = dequant_scheme(scheme, self.quant_bytes(w_id)?, k * n)?;
        let x = self.val(x_id);
        let wc = format!("{out_name}_w");
        self.operations.push(make_const(
            &mut self.blob,
            &wc,
            &Shape::new(&[n, k], DType::F32),
            &wf,
        )?);
        let op = self.simple_op(
            "matmul",
            out_name,
            &out_shape,
            vec![
                ("x", bind_name(&x)),
                ("y", bind_name(&wc)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.push_named(id, out_name.to_string(), op);
        Ok(())
    }

    /// Dequantize packed MoE weights to a plain f32 const (no matmul).
    fn lower_dequant_moe_weights(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let total = shape.num_elements().unwrap_or(0);
        let wf = dequant_scheme(scheme, self.quant_bytes(node.inputs[0])?, total)?;
        self.operations
            .push(make_const(&mut self.blob, out_name, &shape, &wf)?);
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// MoE grouped matmul with on-device Q8_0 / Q4_0 / IQ4NL / K-quant dequant.
    fn lower_dequant_grouped_matmul_ondevice(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        const QK: usize = 32;
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;
        let bytes = self.quant_bytes(node.inputs[1])?;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k * n) / block_elems.max(1) * block_bytes;
        let num_experts = bytes.len() / slab_bytes.max(1);
        let nb_per_expert = (k * n) / QK;
        if nb_per_expert * QK != k * n {
            return self.lower_dequant_grouped_matmul(id, scheme, out_name);
        }

        let mut all_qs = Vec::with_capacity(num_experts * nb_per_expert * QK);
        let mut all_sc = Vec::with_capacity(num_experts * nb_per_expert);
        let mut all_off = Vec::with_capacity(num_experts * nb_per_expert);
        for e in 0..num_experts {
            let slab = &bytes[e * slab_bytes..(e + 1) * slab_bytes];
            let (qs, sc, off) = split_gguf_ondevice(scheme, slab, nb_per_expert)?;
            all_qs.extend(qs);
            all_sc.extend(sc);
            all_off.extend(off);
        }
        let nb_total = num_experts * nb_per_expert;
        let q_name = format!("{out_name}_q");
        self.operations.push(make_const(
            &mut self.blob,
            &q_name,
            &Shape::new(&[nb_total, QK], DType::F32),
            &all_qs,
        )?);
        let sc_name = format!("{out_name}_sc");
        self.operations.push(make_const(
            &mut self.blob,
            &sc_name,
            &Shape::new(&[nb_total, 1], DType::F32),
            &all_sc,
        )?);
        let mul_name = format!("{out_name}_mul");
        self.emit(
            "mul",
            &mul_name,
            &Shape::new(&[nb_total, QK], DType::F32),
            vec![("x", bind_name(&q_name)), ("y", bind_name(&sc_name))],
        )?;
        let dq = if all_off.iter().any(|&o| o != 0.0) {
            let off_name = format!("{out_name}_off");
            self.operations.push(make_const(
                &mut self.blob,
                &off_name,
                &Shape::new(&[nb_total, 1], DType::F32),
                &all_off,
            )?);
            let sub_name = format!("{out_name}_dq");
            self.emit(
                "sub",
                &sub_name,
                &Shape::new(&[nb_total, QK], DType::F32),
                vec![("x", bind_name(&mul_name)), ("y", bind_name(&off_name))],
            )?;
            sub_name
        } else {
            mul_name
        };
        let weight = format!("{out_name}_wdq");
        self.reshape_to(
            &dq,
            &[num_experts as i64, n as i64, k as i64],
            &Shape::new(&[num_experts, n, k], DType::F32),
            &weight,
        )?;

        let input = self.val(node.inputs[0]);
        let eidx = self.val(node.inputs[2]);
        let eidx_i32 = format!("{out_name}_eidx");
        let eidx_shape = self
            .graph
            .shape(node.inputs[2])
            .clone()
            .with_dtype(DType::I32);
        self.emit(
            "cast",
            &eidx_i32,
            &eidx_shape,
            vec![
                ("x", bind_name(&eidx)),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, n, k], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        let mm = format!("{out_name}_mm");
        self.emit(
            "matmul",
            &mm,
            &Shape::new(&[m, 1, n], DType::F32),
            vec![
                ("x", bind_name(&in3)),
                ("y", bind_name(&wsel)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.reshape_to(&mm, &[m as i64, n as i64], &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// MoE grouped matmul with quantized expert weights. Dequantizes all
    /// `E` expert slabs (`[E, N, K]`), gathers per token, then batched
    /// matmul with `transpose_y`.
    fn lower_dequant_grouped_matmul(
        &mut self,
        id: NodeId,
        scheme: QuantScheme,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let in_shape = self.graph.shape(node.inputs[0]).clone();
        let m = dim_static(&in_shape, in_shape.rank() - 2)?;
        let k = dim_static(&in_shape, in_shape.rank() - 1)?;
        let n = dim_static(&out_shape, out_shape.rank() - 1)?;

        // Dequant every expert slab. Block-byte math gives the expert count.
        let bytes = self.quant_bytes(node.inputs[1])?;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k * n) / block_elems.max(1) * block_bytes;
        let num_experts = bytes.len() / slab_bytes.max(1);
        let total = num_experts * n * k;
        let wf = dequant_scheme(scheme, bytes, total)?;

        let weight = format!("{out_name}_wdq");
        self.operations.push(make_const(
            &mut self.blob,
            &weight,
            &Shape::new(&[num_experts, n, k], DType::F32),
            &wf,
        )?);

        let input = self.val(node.inputs[0]);
        let eidx = self.val(node.inputs[2]);
        let eidx_i32 = format!("{out_name}_eidx");
        let eidx_shape = self
            .graph
            .shape(node.inputs[2])
            .clone()
            .with_dtype(DType::I32);
        self.emit(
            "cast",
            &eidx_i32,
            &eidx_shape,
            vec![
                ("x", bind_name(&eidx)),
                ("dtype", bind_value(scalar_str("int32"))),
            ],
        )?;
        // gather expert slabs → [M, N, K]
        let wsel = format!("{out_name}_wsel");
        self.emit(
            "gather",
            &wsel,
            &Shape::new(&[m, n, k], DType::F32),
            vec![
                ("x", bind_name(&weight)),
                ("indices", bind_name(&eidx_i32)),
                ("axis", bind_value(scalar_i32(0))),
            ],
        )?;
        // input [M,K] → [M,1,K]; matmul([M,1,K],[M,N,K]ᵀ) → [M,1,N] → [M,N]
        let in3 = format!("{out_name}_in3");
        self.reshape_to(
            &input,
            &[m as i64, 1, k as i64],
            &Shape::new(&[m, 1, k], DType::F32),
            &in3,
        )?;
        let mm = format!("{out_name}_mm");
        self.emit(
            "matmul",
            &mm,
            &Shape::new(&[m, 1, n], DType::F32),
            vec![
                ("x", bind_name(&in3)),
                ("y", bind_name(&wsel)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        self.reshape_to(&mm, &[m as i64, n as i64], &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Bake an affine (scale / zero-point) parameter as a const that
    /// broadcasts against a rank-`rank` tensor: a scalar for per-tensor
    /// quant, or a `[1,…,C,…,1]` vector along `axis` for per-channel.
    fn bake_affine(
        &mut self,
        name: &str,
        values: &[f32],
        axis: Option<usize>,
        rank: usize,
    ) -> Result<()> {
        let op = match axis {
            Some(ax) if values.len() > 1 => {
                let mut dims = vec![1usize; rank];
                dims[ax] = values.len();
                make_const(&mut self.blob, name, &Shape::new(&dims, DType::F32), values)?
            }
            // Per-tensor: a rank-0 scalar.
            _ => make_const(
                &mut self.blob,
                name,
                &Shape::new(&[], DType::F32),
                &[values[0]],
            )?,
        };
        self.operations.push(op);
        Ok(())
    }

    /// Dequantize an int8 tensor: `out = (cast(q,f32) - zp) · scale`.
    fn lower_dequantize(
        &mut self,
        id: NodeId,
        axis: Option<usize>,
        scales: &[f32],
        zero_points: &[i32],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // f32 output
        let rank = shape.rank();
        let q = self.val(node.inputs[0]);

        // MIL ios16 has no int8 activations: quantized values flow as
        // integer-valued fp32 (from `Quantize`). Only an int32 producer
        // needs an explicit cast.
        let in_dt = self.graph.shape(node.inputs[0]).dtype();
        let qf = if in_dt == DType::I32 {
            let c = format!("{out_name}_qf");
            self.emit(
                "cast",
                &c,
                &shape,
                vec![
                    ("x", bind_name(&q)),
                    ("dtype", bind_value(scalar_str("fp32"))),
                ],
            )?;
            c
        } else {
            q
        };
        let zp: Vec<f32> = zero_points.iter().map(|&z| z as f32).collect();
        let zpc = format!("{out_name}_zp");
        self.bake_affine(&zpc, &zp, axis, rank)?;
        let sub = format!("{out_name}_sub");
        self.emit(
            "sub",
            &sub,
            &shape,
            vec![("x", bind_name(&qf)), ("y", bind_name(&zpc))],
        )?;
        let sc = format!("{out_name}_sc");
        self.bake_affine(&sc, scales, axis, rank)?;
        self.emit(
            "mul",
            out_name,
            &shape,
            vec![("x", bind_name(&sub)), ("y", bind_name(&sc))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Quantize a f32 tensor to int8:
    /// `out = cast(clip(round(x/scale) + zp, -128, 127), int8)`.
    fn lower_quantize(
        &mut self,
        id: NodeId,
        axis: Option<usize>,
        scales: &[f32],
        zero_points: &[i32],
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone(); // int8 output
        let f32_shape = shape.clone().with_dtype(DType::F32);
        let rank = shape.rank();
        let x = self.val(node.inputs[0]);

        let inv: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
        let invc = format!("{out_name}_inv");
        self.bake_affine(&invc, &inv, axis, rank)?;
        let scaled = format!("{out_name}_xs");
        self.emit(
            "mul",
            &scaled,
            &f32_shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&invc))],
        )?;
        let rounded = format!("{out_name}_rnd");
        self.emit(
            "round",
            &rounded,
            &f32_shape,
            vec![("x", bind_name(&scaled))],
        )?;
        let zp: Vec<f32> = zero_points.iter().map(|&z| z as f32).collect();
        let zpc = format!("{out_name}_zp");
        self.bake_affine(&zpc, &zp, axis, rank)?;
        let shifted = format!("{out_name}_shift");
        self.emit(
            "add",
            &shifted,
            &f32_shape,
            vec![("x", bind_name(&rounded)), ("y", bind_name(&zpc))],
        )?;
        // MIL ios16 `cast` can't target int8, so the quantized value stays
        // as integer-valued fp32 (the IR's I8 type is satisfied logically;
        // `Dequantize` consumes the fp32 representation directly).
        self.emit(
            "clip",
            out_name,
            &f32_shape,
            vec![
                ("x", bind_name(&shifted)),
                ("alpha", bind_value(scalar_f32(-128.0))),
                ("beta", bind_value(scalar_f32(127.0))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Mamba selective scan, unrolled over the sequence. Inputs
    /// `[x, delta, A, B, C]` with `x,delta:[b,s,h]`, `A:[h,n]`,
    /// `B,C:[b,s,n]`. Per step: `H ← exp(Δ·A)·H + (Δ·x)·B`, `y = Σₙ C·H`.
    fn lower_selective_scan(&mut self, id: NodeId, n: usize, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let b = dim_static(&out_shape, 0)?;
        let s = dim_static(&out_shape, 1)?;
        let h = dim_static(&out_shape, 2)?;
        let x = self.val(node.inputs[0]);
        let delta = self.val(node.inputs[1]);
        let a = self.val(node.inputs[2]);
        let b_in = self.val(node.inputs[3]);
        let c_in = self.val(node.inputs[4]);

        let bhn = Shape::new(&[b, h, n], DType::F32);
        let bh1 = Shape::new(&[b, h, 1], DType::F32);
        let b1h = Shape::new(&[b, 1, h], DType::F32);
        let b1n = Shape::new(&[b, 1, n], DType::F32);
        let bh = Shape::new(&[b, h], DType::F32);

        // A → [1, h, n]
        let a3 = format!("{out_name}_a3");
        self.reshape_to(
            &a,
            &[1, h as i64, n as i64],
            &Shape::new(&[1, h, n], DType::F32),
            &a3,
        )?;
        // state₀ = 0
        let mut state = format!("{out_name}_s0");
        self.operations.push(make_const(
            &mut self.blob,
            &state,
            &bhn,
            &vec![0.0f32; b * h * n],
        )?);

        let mut ys = Vec::with_capacity(s);
        for t in 0..s {
            let p = format!("{out_name}_t{t}");
            let xt = format!("{p}_x");
            let xt3 = format!("{p}_x3");
            self.slice_axis(&x, 3, 1, t, 1, &b1h, &xt)?;
            self.reshape_to(&xt, &[b as i64, h as i64, 1], &bh1, &xt3)?;
            let dt = format!("{p}_d");
            let dt3 = format!("{p}_d3");
            self.slice_axis(&delta, 3, 1, t, 1, &b1h, &dt)?;
            self.reshape_to(&dt, &[b as i64, h as i64, 1], &bh1, &dt3)?;
            let bt = format!("{p}_b");
            self.slice_axis(&b_in, 3, 1, t, 1, &b1n, &bt)?;
            let ct = format!("{p}_c");
            self.slice_axis(&c_in, 3, 1, t, 1, &b1n, &ct)?;

            // da = exp(Δ·A)
            let dta = format!("{p}_dta");
            self.emit(
                "mul",
                &dta,
                &bhn,
                vec![("x", bind_name(&dt3)), ("y", bind_name(&a3))],
            )?;
            let da = format!("{p}_da");
            self.emit("exp", &da, &bhn, vec![("x", bind_name(&dta))])?;
            let decay = format!("{p}_decay");
            self.emit(
                "mul",
                &decay,
                &bhn,
                vec![("x", bind_name(&da)), ("y", bind_name(&state))],
            )?;
            // input term (Δ·x)·B
            let dx = format!("{p}_dx");
            self.emit(
                "mul",
                &dx,
                &bh1,
                vec![("x", bind_name(&dt3)), ("y", bind_name(&xt3))],
            )?;
            let inp = format!("{p}_inp");
            self.emit(
                "mul",
                &inp,
                &bhn,
                vec![("x", bind_name(&dx)), ("y", bind_name(&bt))],
            )?;
            let snew = format!("{p}_s");
            self.emit(
                "add",
                &snew,
                &bhn,
                vec![("x", bind_name(&decay)), ("y", bind_name(&inp))],
            )?;
            state = snew;
            // y = Σₙ C·H
            let prod = format!("{p}_pr");
            self.emit(
                "mul",
                &prod,
                &bhn,
                vec![("x", bind_name(&ct)), ("y", bind_name(&state))],
            )?;
            let yt = format!("{p}_y");
            self.emit(
                "reduce_sum",
                &yt,
                &bh,
                vec![
                    ("x", bind_name(&prod)),
                    ("axes", bind_value(vec_i32(&[2]))),
                    ("keep_dims", bind_value(scalar_bool(false))),
                ],
            )?;
            let yt3 = format!("{p}_y3");
            self.reshape_to(&yt, &[b as i64, 1, h as i64], &b1h, &yt3)?;
            ys.push(yt3);
        }
        self.emit(
            "concat",
            out_name,
            &out_shape,
            vec![
                ("values", bind_names(&ys)),
                ("axis", bind_value(scalar_i32(1))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Qwen3.5 gated delta-net, unrolled over the sequence. Inputs
    /// `[q,k,v,g,beta(,state)]`, `q,k,v:[b,s,H,n]`, `g,beta:[b,s,H]`.
    /// Per step: `S ← exp(g)·S`; `Δ=(v − Sᵀk)·β`; `S += k⊗Δ`;
    /// `y = (1/√n)·Sᵀq`.
    fn lower_gated_delta_net(
        &mut self,
        id: NodeId,
        n: usize,
        carry: bool,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone(); // [b,s,H,n]
        let b = dim_static(&out_shape, 0)?;
        let s = dim_static(&out_shape, 1)?;
        let hh = dim_static(&out_shape, 2)?;
        let scale = (n as f32).powf(-0.5);

        let q = self.val(node.inputs[0]);
        let k = self.val(node.inputs[1]);
        let v = self.val(node.inputs[2]);
        let g = self.val(node.inputs[3]);
        let beta = self.val(node.inputs[4]);

        let bhnn = Shape::new(&[b, hh, n, n], DType::F32);
        let bh1n = Shape::new(&[b, hh, 1, n], DType::F32);
        let bhn1 = Shape::new(&[b, hh, n, 1], DType::F32);
        let bh11 = Shape::new(&[b, hh, 1, 1], DType::F32);
        let bsh1 = Shape::new(&[b, 1, hh], DType::F32);

        // state₀ — external [b,H,n,n] or zeros.
        let mut state = if carry {
            self.val(node.inputs[5])
        } else {
            let s0 = format!("{out_name}_s0");
            self.operations.push(make_const(
                &mut self.blob,
                &s0,
                &bhnn,
                &vec![0.0f32; b * hh * n * n],
            )?);
            s0
        };

        let mut ys = Vec::with_capacity(s);
        for t in 0..s {
            let p = format!("{out_name}_t{t}");
            // slice token t: q,k,v → [b,1,H,n] → [b,H,1,n]; g,beta → [b,H,1,1]
            let qt = self.gdn_vec(&q, t, b, hh, n, &p, "q")?;
            let kt = self.gdn_vec(&k, t, b, hh, n, &p, "k")?;
            let vt = self.gdn_vec(&v, t, b, hh, n, &p, "v")?;
            let gt = self.gdn_scalar(&g, t, b, hh, &p, "g")?;
            let bt = self.gdn_scalar(&beta, t, b, hh, &p, "b")?;

            // S *= exp(g)
            let ge = format!("{p}_ge");
            self.emit("exp", &ge, &bh11, vec![("x", bind_name(&gt))])?;
            let sg = format!("{p}_sg");
            self.emit(
                "mul",
                &sg,
                &bhnn,
                vec![("x", bind_name(&state)), ("y", bind_name(&ge))],
            )?;
            // sk = kᵀ·S  → [b,H,1,n] (matmul [b,H,1,n] @ [b,H,n,n])
            let sk = format!("{p}_sk");
            self.matmul_op(&sk, &kt, &sg, false, false, &bh1n)?;
            // Δ = (v − sk)·β
            let d0 = format!("{p}_d0");
            self.emit(
                "sub",
                &d0,
                &bh1n,
                vec![("x", bind_name(&vt)), ("y", bind_name(&sk))],
            )?;
            let delta = format!("{p}_dl");
            self.emit(
                "mul",
                &delta,
                &bh1n,
                vec![("x", bind_name(&d0)), ("y", bind_name(&bt))],
            )?;
            // S += k⊗Δ  (kᵀ:[b,H,n,1] · Δ:[b,H,1,n] → [b,H,n,n])
            let kcol = format!("{p}_kc");
            self.reshape_to(&kt, &[b as i64, hh as i64, n as i64, 1], &bhn1, &kcol)?;
            let outer = format!("{p}_outer");
            self.matmul_op(&outer, &kcol, &delta, false, false, &bhnn)?;
            let snew = format!("{p}_s");
            self.emit(
                "add",
                &snew,
                &bhnn,
                vec![("x", bind_name(&sg)), ("y", bind_name(&outer))],
            )?;
            state = snew;
            // y = scale·(qᵀ·S) → [b,H,1,n]
            let qs = format!("{p}_qs");
            self.matmul_op(&qs, &qt, &state, false, false, &bh1n)?;
            let yt = format!("{p}_y");
            self.emit(
                "mul",
                &yt,
                &bh1n,
                vec![("x", bind_name(&qs)), ("y", bind_value(scalar_f32(scale)))],
            )?;
            // → [b,1,H,n]
            let yt2 = format!("{p}_y2");
            self.reshape_to(
                &yt,
                &[b as i64, 1, hh as i64, n as i64],
                &Shape::new(&[b, 1, hh, n], DType::F32),
                &yt2,
            )?;
            ys.push(yt2);
        }
        let _ = bsh1;
        self.emit(
            "concat",
            out_name,
            &out_shape,
            vec![
                ("values", bind_names(&ys)),
                ("axis", bind_value(scalar_i32(1))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Slice token `t` of a `[b,s,H,n]` GDN tensor → `[b,H,1,n]`.
    fn gdn_vec(
        &mut self,
        src: &str,
        t: usize,
        b: usize,
        hh: usize,
        n: usize,
        p: &str,
        tag: &str,
    ) -> Result<String> {
        let sl = format!("{p}_{tag}sl");
        self.slice_axis(
            src,
            4,
            1,
            t,
            1,
            &Shape::new(&[b, 1, hh, n], DType::F32),
            &sl,
        )?;
        let out = format!("{p}_{tag}");
        self.reshape_to(
            &sl,
            &[b as i64, hh as i64, 1, n as i64],
            &Shape::new(&[b, hh, 1, n], DType::F32),
            &out,
        )?;
        Ok(out)
    }

    /// Slice token `t` of a `[b,s,H]` GDN scalar tensor → `[b,H,1,1]`.
    fn gdn_scalar(
        &mut self,
        src: &str,
        t: usize,
        b: usize,
        hh: usize,
        p: &str,
        tag: &str,
    ) -> Result<String> {
        let sl = format!("{p}_{tag}sl");
        self.slice_axis(src, 3, 1, t, 1, &Shape::new(&[b, 1, hh], DType::F32), &sl)?;
        let out = format!("{p}_{tag}");
        self.reshape_to(
            &sl,
            &[b as i64, hh as i64, 1, 1],
            &Shape::new(&[b, hh, 1, 1], DType::F32),
            &out,
        )?;
        Ok(out)
    }

    fn matmul_op(
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

    fn push_named(&mut self, id: NodeId, name: String, op: proto::Operation) {
        self.operations.push(op);
        self.names.insert(id.0, name);
    }

    fn unique_feature_name(&mut self, raw: &str) -> String {
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
    fn verify_refs(&self, block_outputs: &[String]) -> Result<()> {
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
            CoremlError::Runtime(format!(
                "CoreML lowering produced a dangling reference to value '{name}': the source node \
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

    fn finish(mut self) -> Result<LoweredProgram> {
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

        let block = proto::Block {
            inputs: vec![],
            outputs: output_names,
            operations: self.operations,
            attributes: HashMap::new(),
        };
        let mut block_specializations = HashMap::new();
        block_specializations.insert(OPSET.to_string(), block);

        let function = proto::Function {
            inputs: self.func_inputs,
            opset: OPSET.to_string(),
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
            specification_version: SPEC_VERSION,
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
