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
}

impl IoTensor {
    /// Number of f32 elements (product of dims).
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<i64>().max(0) as usize
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
    let mut ctx = LowerCtx::new(graph, params, typed_params);
    ctx.run()?;
    ctx.finish()
}

/// Per-node value name + the proto pieces accumulated during the walk.
struct LowerCtx<'a> {
    graph: &'a Graph,
    params: &'a HashMap<String, Vec<f32>>,
    typed_params: &'a TypedParams,
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
    ) -> Self {
        LowerCtx {
            graph,
            params,
            typed_params,
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
                    node.shape.dtype()
                } else {
                    DType::F32
                };
                let io_shape = node.shape.clone().with_dtype(io_dtype);
                let dims = static_dims(&io_shape)?;
                self.func_inputs.push(named_value_type(&feat, &io_shape)?);
                self.inputs.push(IoTensor {
                    ir_name: name.clone(),
                    feature_name: feat.clone(),
                    dims,
                    dtype: io_dtype,
                });
                self.names.insert(id.0, feat);
            }
            Op::Param { name } => {
                if let Some(data) = self.params.get(name) {
                    let op = make_const(&mut self.blob, &out_name, &node.shape, data)?;
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
            Op::Rope { head_dim, n_rot } => {
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                    let op = simple_op(
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
                    let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                let op = simple_op(
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
                self.lower_dequant_matmul(id, *scheme, &out_name)?;
            }
            Op::DequantMoEWeights { scheme } => {
                self.lower_dequant_moe_weights(id, *scheme, &out_name)?;
            }
            Op::DequantGroupedMatMul { scheme } => {
                self.lower_dequant_grouped_matmul(id, *scheme, &out_name)?;
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
            Activation::Log => Some(("log", vec![])),
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
            let op = simple_op(ty, out_name, &node.shape, binds)?;
            self.push_named(id, out_name.to_string(), op);
            return Ok(());
        }

        match act {
            // silu(x) = x * sigmoid(x)
            Activation::Silu => {
                let sig = format!("{out_name}_sig");
                let sig_op = simple_op("sigmoid", &sig, &node.shape, vec![("x", bind_name(&x))])?;
                self.operations.push(sig_op);
                let op = simple_op(
                    "mul",
                    out_name,
                    &node.shape,
                    vec![("x", bind_name(&x)), ("y", bind_name(&sig))],
                )?;
                self.push_named(id, out_name.to_string(), op);
            }
            // neg(x) = mul(x, -1)
            Activation::Neg => {
                let op = simple_op(
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
        let op = simple_op("layer_norm", out_name, &node.shape, binds)?;
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
        self.operations.push(simple_op(
            "mul",
            &sq,
            &node.shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&x))],
        )?);
        // ms = reduce_mean(sq, axes, keep_dims=true)
        let ms = format!("{out_name}_ms");
        self.operations.push(simple_op(
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
        self.operations.push(simple_op(
            "add",
            &mse,
            &red_shape,
            vec![("x", bind_name(&ms)), ("y", bind_value(scalar_f32(eps)))],
        )?);
        // inv = rsqrt(ms_eps)  (eps already folded into ms_eps above)
        let inv = format!("{out_name}_inv");
        self.operations.push(simple_op(
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
        self.operations.push(simple_op(
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
            self.operations.push(simple_op(
                "mul",
                &name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&g))],
            )?);
            last = name;
        }
        if has_beta {
            let b = self.val(node.inputs[2]);
            self.operations.push(simple_op(
                "add",
                out_name,
                &node.shape,
                vec![("x", bind_name(&last)), ("y", bind_name(&b))],
            )?);
        }
        self.names.insert(id.0, out_name.to_string());
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
        let op = simple_op(ty, name, shape, binds)?;
        self.operations.push(op);
        Ok(())
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
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        let rank = shape.rank();
        let last = match shape.dim(rank - 1) {
            Dim::Static(n) => n,
            Dim::Dynamic(s) => {
                return Err(CoremlError::DynamicShape(format!("rope last dim ?{s}")));
            }
        };
        if last != head_dim {
            return Err(CoremlError::Unsupported(
                "rope: only last-dim == head_dim layout (BHSD / [B,S,D])".into(),
            ));
        }
        let rot_half = n_rot / 2;
        let x = self.val(node.inputs[0]);
        let cos = self.val(node.inputs[1]);
        let sin = self.val(node.inputs[2]);

        let half_shape = with_last(&shape, rot_half);
        let rot_shape = with_last(&shape, n_rot);

        // x1 = x[..0:rh], x2 = x[..rh:n_rot]
        let x1 = format!("{out_name}_x1");
        let x2 = format!("{out_name}_x2");
        self.slice_last(&x, rank, 0, rot_half, &half_shape, &x1)?;
        self.slice_last(&x, rank, rot_half, rot_half, &half_shape, &x2)?;

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
            vec![("x", bind_name(&x1)), ("y", bind_name(&cos))],
        )?;
        self.emit(
            "mul",
            &x2s,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&sin))],
        )?;
        self.emit(
            "mul",
            &x2c,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&cos))],
        )?;
        self.emit(
            "mul",
            &x1s,
            &half_shape,
            vec![("x", bind_name(&x1)), ("y", bind_name(&sin))],
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

        let axis = (rank - 1) as i32;
        let pass_len = head_dim - n_rot;
        if pass_len == 0 {
            self.emit(
                "concat",
                out_name,
                &shape,
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
            let pass_shape = with_last(&shape, pass_len);
            self.slice_last(&x, rank, n_rot, pass_len, &pass_shape, &pass)?;
            self.emit(
                "concat",
                out_name,
                &shape,
                vec![
                    ("values", bind_names(&[out_rot, pass])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Scaled dot-product attention composed from MIL primitives.
    /// Inputs `[q, k, v]` (+ `bias` for `MaskKind::Bias`), all `[B,H,S,D]`.
    /// `scores = softmax((q·kᵀ)·scale [+ mask] [softcap]) · v`.
    fn lower_attention(
        &mut self,
        id: NodeId,
        _num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        score_scale: Option<f32>,
        softcap: Option<f32>,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let q_shape = node.shape.clone();
        if q_shape.rank() != 4 {
            return Err(CoremlError::Unsupported(
                "attention: only rank-4 [B,H,S,D] layout".into(),
            ));
        }
        let k_shape = self.graph.shape(node.inputs[1]).clone();
        let s_q = dim_static(&q_shape, 2)?;
        let s_k = dim_static(&k_shape, 2)?;
        let scores_shape = {
            let mut d = q_shape.dims().to_vec();
            d[3] = Dim::Static(s_k); // [B,H,Sq,Sk]
            Shape::from_dims(&d, DType::F32)
        };

        let q = self.val(node.inputs[0]);
        let k = self.val(node.inputs[1]);
        let v = self.val(node.inputs[2]);
        let scale = score_scale.unwrap_or((head_dim as f32).powf(-0.5));

        // raw = q @ kᵀ  (transpose_y batches over [B,H])
        let raw = format!("{out_name}_qk");
        self.emit(
            "matmul",
            &raw,
            &scores_shape,
            vec![
                ("x", bind_name(&q)),
                ("y", bind_name(&k)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(true))),
            ],
        )?;
        // scaled = raw * scale
        let mut cur = format!("{out_name}_sc");
        self.emit(
            "mul",
            &cur,
            &scores_shape,
            vec![("x", bind_name(&raw)), ("y", bind_value(scalar_f32(scale)))],
        )?;

        // mask
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
                    &scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&mask_name))],
                )?;
                cur = masked;
            }
            MaskKind::Bias => {
                let bias = self.val(node.inputs[3]);
                let masked = format!("{out_name}_msk");
                self.emit(
                    "add",
                    &masked,
                    &scores_shape,
                    vec![("x", bind_name(&cur)), ("y", bind_name(&bias))],
                )?;
                cur = masked;
            }
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "attention mask {other:?}"
                )));
            }
        }

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

        // out = probs @ v  -> [B,H,Sq,D]
        self.emit(
            "matmul",
            out_name,
            &q_shape,
            vec![
                ("x", bind_name(&probs)),
                ("y", bind_name(&v)),
                ("transpose_x", bind_value(scalar_bool(false))),
                ("transpose_y", bind_value(scalar_bool(false))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
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
        let one_d = !transpose
            && in_shape.rank() == 4
            && in_shape.dim(2).unwrap_static() == 1
            && w_shape.rank() == 4
            && w_shape.dim(3).unwrap_static() == 1
            && w_shape.dim(2).unwrap_static() > 1;
        if one_d {
            let n = in_shape.dim(0).unwrap_static() as i32;
            let c = in_shape.dim(1).unwrap_static() as i32;
            let l = in_shape.dim(3).unwrap_static() as i32;
            let co = w_shape.dim(0).unwrap_static() as i32;
            let ci = w_shape.dim(1).unwrap_static() as i32;
            let k = w_shape.dim(2).unwrap_static() as i32;
            let lo = shape.dim(3).unwrap_static() as i32;
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
            let op = simple_op(
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
        let op = simple_op(ty, out_name, &shape, binds)?;
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
        let op = simple_op(ty, out_name, &shape, binds)?;
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
        let op = simple_op(
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

    fn finish(self) -> Result<LoweredProgram> {
        let graph = self.graph;

        // Outputs: one feature per graph output node.
        let mut output_names = Vec::new();
        let mut outputs = Vec::new();
        for &out_id in &graph.outputs {
            let vname = self.val(out_id);
            output_names.push(vname.clone());
            let shape = graph.shape(out_id);
            outputs.push(IoTensor {
                ir_name: vname.clone(),
                feature_name: vname.clone(),
                dims: static_dims(shape)?,
                dtype: shape.dtype(),
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
