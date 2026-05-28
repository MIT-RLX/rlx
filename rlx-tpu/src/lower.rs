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

//! Graph → HLO lowering walker.
//!
//! Walks an `rlx_ir::Graph`, emits HLO instructions via [`HloBuilder`],
//! and returns the serialized `HloModuleProto` bytes plus the
//! per-output / per-input shape metadata the backend needs at run
//! time.
//!
//! Composite ops (LayerNorm, RmsNorm, Softmax, Attention, Rope, Pool,
//! ElementwiseRegion, ...) are decomposed in HLO directly — no
//! custom_call. Keeps the emitted module portable across PJRT
//! plugins (TPU, CPU, GPU). FusedSwiGLU / FusedAttentionBlock /
//! FusedTransformerLayer / LoraMatMul / If / While are normalized
//! through `crate::unfuse` before lowering.
//!
//! Ops that have no clean HLO decomposition without large blowup
//! (Sample, TopK, SelectiveScan, DequantMatMul, GroupedMatMul) panic
//! with a clear message — they need follow-up tier work.

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::hlo::{
    Computation, ConvDimNumbers, DotDimNumbers, GatherDimNumbers, HloBuilder, Literal, LiteralData,
    ProgramShape, ScatterDimNumbers, Shape, Window, WindowDim, prim, prim_of,
};

/// Compiled-against-this-graph HLO module bytes plus the metadata the
/// backend needs at run time.
pub struct HloModule {
    pub bytes: Vec<u8>,
    pub output_lens: Vec<usize>,
    pub output_dtypes: Vec<DType>,
    pub output_shapes: Vec<Vec<i64>>,
    pub input_names: Vec<String>,
    pub input_dtypes: Vec<DType>,
    pub input_shapes: Vec<Vec<i64>>,
    pub param_names: Vec<String>,
    pub param_dtypes: Vec<DType>,
    pub param_shapes: Vec<Vec<i64>>,
}

pub fn lower_graph(graph: &Graph) -> HloModule {
    let mut b = HloBuilder::new(&graph.name);

    // Reducer subcomputations cached by (opcode, prim_ty) so multiple
    // Reduce / ReduceWindow ops share the same body.
    let mut reducers: HashMap<(String, i32), Computation> = HashMap::new();

    let entry = b.computation("entry");
    let mut id_map: HashMap<NodeId, i64> = HashMap::new();

    let (inputs, params, others) = partition_nodes(graph);

    let mut input_names = Vec::new();
    let mut input_dtypes = Vec::new();
    let mut input_shapes = Vec::new();
    let mut param_names = Vec::new();
    let mut param_dtypes = Vec::new();
    let mut param_shapes = Vec::new();
    let mut program_param_shapes: Vec<Shape> = Vec::new();
    let mut program_param_names: Vec<String> = Vec::new();

    for (pi, &nid) in inputs.iter().enumerate() {
        let n = graph.node(nid);
        let name = match &n.op {
            Op::Input { name } => name.clone(),
            _ => unreachable!(),
        };
        let dims = ir_dims(&n.shape);
        let shape = Shape::array(prim_of(n.shape.dtype()), &dims);
        let id = entry.parameter(pi as i64, &name, shape.clone());
        id_map.insert(nid, id);
        input_names.push(name.clone());
        input_dtypes.push(n.shape.dtype());
        input_shapes.push(dims);
        program_param_shapes.push(shape);
        program_param_names.push(name);
    }
    let next_param_base = inputs.len() as i64;
    for (i, &nid) in params.iter().enumerate() {
        let n = graph.node(nid);
        let name = match &n.op {
            Op::Param { name } => name.clone(),
            _ => unreachable!(),
        };
        let dims = ir_dims(&n.shape);
        let shape = Shape::array(prim_of(n.shape.dtype()), &dims);
        let id = entry.parameter(next_param_base + i as i64, &name, shape.clone());
        id_map.insert(nid, id);
        param_names.push(name.clone());
        param_dtypes.push(n.shape.dtype());
        param_shapes.push(dims);
        program_param_shapes.push(shape);
        program_param_names.push(name);
    }

    let mut ctx = LowerCtx {
        graph,
        entry: &entry,
        id_map: &mut id_map,
        reducers: &mut reducers,
        builder: &mut b,
    };
    for &nid in &others {
        let id = ctx.lower_node(nid);
        ctx.id_map.insert(nid, id);
    }

    // Build the entry computation's output.
    let out_ids: Vec<i64> = graph
        .outputs
        .iter()
        .map(|nid| *id_map.get(nid).expect("output node not lowered"))
        .collect();
    let out_shapes_v: Vec<Vec<i64>> = graph
        .outputs
        .iter()
        .map(|nid| ir_dims(&graph.node(*nid).shape))
        .collect();
    let out_dtypes: Vec<DType> = graph
        .outputs
        .iter()
        .map(|nid| graph.node(*nid).shape.dtype())
        .collect();
    let out_lens: Vec<usize> = out_shapes_v
        .iter()
        .map(|d| d.iter().product::<i64>().max(1) as usize)
        .collect();

    let result_shape = if out_ids.len() == 1 {
        Shape::array(prim_of(out_dtypes[0]), &out_shapes_v[0])
    } else {
        let elems: Vec<Shape> = out_dtypes
            .iter()
            .zip(out_shapes_v.iter())
            .map(|(dt, dims)| Shape::array(prim_of(*dt), dims))
            .collect();
        Shape::tuple(elems)
    };
    let root_id = if out_ids.len() == 1 {
        out_ids[0]
    } else {
        entry.tuple(&out_ids, result_shape.clone())
    };
    entry.set_root(root_id);
    entry.set_program_shape(ProgramShape {
        parameters: program_param_shapes,
        parameter_names: program_param_names,
        result: result_shape,
    });

    let bytes = b.finish();
    HloModule {
        bytes,
        output_lens: out_lens,
        output_dtypes: out_dtypes,
        output_shapes: out_shapes_v,
        input_names,
        input_dtypes,
        input_shapes,
        param_names,
        param_dtypes,
        param_shapes,
    }
}

fn partition_nodes(graph: &Graph) -> (Vec<NodeId>, Vec<NodeId>, Vec<NodeId>) {
    let mut inputs = Vec::new();
    let mut params = Vec::new();
    let mut others = Vec::new();
    for n in graph.nodes() {
        match &n.op {
            Op::Input { .. } => inputs.push(n.id),
            Op::Param { .. } => params.push(n.id),
            _ => others.push(n.id),
        }
    }
    (inputs, params, others)
}

fn ir_dims(shape: &rlx_ir::Shape) -> Vec<i64> {
    shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect()
}

// ── Lowering context ──────────────────────────────────────────────

/// Context carried through the per-op lowering. Bundles the various
/// mutable references so the lower_* methods can be ordinary methods
/// instead of free functions taking eight arguments each.
struct LowerCtx<'a> {
    graph: &'a Graph,
    entry: &'a Computation,
    id_map: &'a mut HashMap<NodeId, i64>,
    reducers: &'a mut HashMap<(String, i32), Computation>,
    builder: &'a mut HloBuilder,
}

impl<'a> LowerCtx<'a> {
    /// HLO id for an already-lowered IR node.
    fn hlo(&self, nid: NodeId) -> i64 {
        *self
            .id_map
            .get(&nid)
            .unwrap_or_else(|| panic!("rlx-tpu: node {nid:?} referenced before lowering"))
    }

    fn ir_shape_dims(&self, nid: NodeId) -> Vec<i64> {
        ir_dims(&self.graph.node(nid).shape)
    }

    fn ir_shape(&self, nid: NodeId) -> Shape {
        let n = self.graph.node(nid);
        Shape::array(prim_of(n.shape.dtype()), &ir_dims(&n.shape))
    }

    fn dtype(&self, nid: NodeId) -> DType {
        self.graph.node(nid).shape.dtype()
    }

    /// Get-or-create a binary-op reducer subcomputation.
    fn reducer(&mut self, opcode: &str, prim_ty: i32) -> Computation {
        let key = (opcode.to_string(), prim_ty);
        if let Some(c) = self.reducers.get(&key) {
            return c.clone();
        }
        let c = self
            .builder
            .make_reducer(&format!("{opcode}_{prim_ty}_red"), opcode, prim_ty);
        self.reducers.insert(key, c.clone());
        c
    }

    /// Broadcast `x` of shape `x_shape` to `target_shape` by aligning
    /// every axis where x has size 1 vs target's size > 1. HLO's
    /// `broadcast` only adds new leading dims; we use `broadcast_in_dim`
    /// semantics by emitting a `reshape` to drop the size-1 dims first
    /// and then a broadcast that places the surviving dims at their
    /// original positions.
    fn broadcast_align(&self, x: i64, x_shape: &[i64], target: Shape) -> i64 {
        let target_dims = target.dimensions.clone();
        debug_assert_eq!(
            x_shape.len(),
            target_dims.len(),
            "broadcast_align expects same rank"
        );
        // Identity broadcast — x already at target.
        if x_shape == target_dims.as_slice() {
            return x;
        }
        // Drop size-1 axes that target wants to expand.
        let surviving_axes: Vec<i64> = (0..x_shape.len() as i64)
            .filter(|&i| {
                let xi = x_shape[i as usize];
                let ti = target_dims[i as usize];
                xi == ti
            })
            .collect();
        let surviving_dims: Vec<i64> = surviving_axes
            .iter()
            .map(|&i| x_shape[i as usize])
            .collect();
        let small = if surviving_dims.len() == x_shape.len() {
            x
        } else {
            let elt = target.element_type;
            self.entry.reshape(x, Shape::array(elt, &surviving_dims))
        };
        self.entry.broadcast(small, &surviving_axes, target)
    }

    /// Constant scalar in an arbitrary primitive type — used for
    /// reduction inits, normalization eps, RoPE constants.
    fn const_scalar_f32(&self, v: f32) -> i64 {
        self.entry.constant_f32_scalar(v)
    }

    /// Reduce over a single axis with a known reducer opcode.
    fn reduce_one(
        &mut self,
        x: i64,
        axis: i64,
        opcode: &str,
        init_v: f32,
        x_dt: DType,
        out_dims: Vec<i64>,
    ) -> i64 {
        let prim_ty = prim_of(x_dt);
        let red = self.reducer(opcode, prim_ty);
        // The reducer expects a scalar of the input's dtype; we
        // use f32 init + convert if needed.
        let init = if x_dt == DType::F32 {
            self.const_scalar_f32(init_v)
        } else {
            let f = self.const_scalar_f32(init_v);
            self.entry.convert(f, Shape::scalar(prim_ty))
        };
        let out_shape = Shape::array(prim_ty, &out_dims);
        self.entry.reduce(x, init, &red, &[axis], out_shape)
    }

    fn lower_node(&mut self, nid: NodeId) -> i64 {
        let n = self.graph.node(nid);
        let out_shape = self.ir_shape(nid);
        let out_dt = self.dtype(nid);

        match &n.op {
            // Inputs / Params already handled by the caller — they
            // never reach lower_node.
            Op::Input { .. } | Op::Param { .. } => unreachable!(),

            Op::Constant { data } => self.lower_constant(data, out_shape, out_dt),

            Op::Activation(act) => {
                let x = self.hlo(n.inputs[0]);
                self.lower_activation(*act, x, out_shape)
            }

            Op::Cast { to } => {
                let x = self.hlo(n.inputs[0]);
                let target = Shape::from_dt(*to, &out_shape.dimensions);
                self.entry.convert(x, target)
            }

            // INT8 quantization, per-tensor (axis=None) or per-channel
            // (axis=Some(d), scales/zero_points indexed by axis d).
            //   q = saturate_i8(round(x / scale[c]) + zero_point[c])
            // For per-channel we materialize a 1-D constant of length
            // `input.dim(axis)` and broadcast along the channel axis.
            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => {
                let x = self.hlo(n.inputs[0]);
                let in_prim = prim_of(self.dtype(n.inputs[0]));
                let inv_b = self.broadcast_q_factor(
                    *axis,
                    &scales.iter().map(|s| 1.0 / *s).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    in_prim,
                );
                let scaled = self.entry.binary(
                    "multiply",
                    x,
                    inv_b,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                let rounded = self.entry.unary(
                    "round-nearest-even",
                    scaled,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                let zp_b = self.broadcast_q_factor(
                    *axis,
                    &zero_points.iter().map(|z| *z as f32).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    in_prim,
                );
                let added = self.entry.binary(
                    "add",
                    rounded,
                    zp_b,
                    Shape::array(in_prim, &out_shape.dimensions),
                );
                // Convert handles saturation per HLO semantics.
                self.entry.convert(added, out_shape)
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => {
                let q = self.hlo(n.inputs[0]);
                let promoted = self
                    .entry
                    .convert(q, Shape::array(prim::F32, &out_shape.dimensions));
                let zp_b = self.broadcast_q_factor(
                    *axis,
                    &zero_points.iter().map(|z| *z as f32).collect::<Vec<_>>(),
                    &out_shape.dimensions,
                    prim::F32,
                );
                let centered = self.entry.binary(
                    "subtract",
                    promoted,
                    zp_b,
                    Shape::array(prim::F32, &out_shape.dimensions),
                );
                let s_b = self.broadcast_q_factor(*axis, scales, &out_shape.dimensions, prim::F32);
                self.entry.binary("multiply", centered, s_b, out_shape)
            }

            Op::Binary(op) => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                self.lower_binary(*op, a, b, n.inputs[0], n.inputs[1], out_shape)
            }

            Op::Compare(op) => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                let dir = match op {
                    CmpOp::Eq => "EQ",
                    CmpOp::Ne => "NE",
                    CmpOp::Lt => "LT",
                    CmpOp::Le => "LE",
                    CmpOp::Gt => "GT",
                    CmpOp::Ge => "GE",
                };
                let (a, b) =
                    self.broadcast_pair_to(a, b, n.inputs[0], n.inputs[1], &out_shape.dimensions);
                self.entry
                    .compare(a, b, dir, Shape::pred(&out_shape.dimensions))
            }

            Op::Where => {
                let c = self.hlo(n.inputs[0]);
                let t = self.hlo(n.inputs[1]);
                let f = self.hlo(n.inputs[2]);
                self.entry.select(c, t, f, out_shape)
            }

            Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
            } => self.lower_elementwise_region(
                &n.inputs,
                chain,
                *num_inputs,
                *scalar_input_mask,
                input_modulus,
                out_shape,
            ),

            Op::MatMul => self.lower_matmul(n.inputs[0], n.inputs[1], out_shape),

            Op::DotGeneral {
                lhs_contracting,
                rhs_contracting,
                lhs_batch,
                rhs_batch,
            } => {
                let a = self.hlo(n.inputs[0]);
                let b = self.hlo(n.inputs[1]);
                let dn = DotDimNumbers {
                    lhs_contracting: lhs_contracting.iter().map(|&x| x as i64).collect(),
                    rhs_contracting: rhs_contracting.iter().map(|&x| x as i64).collect(),
                    lhs_batch: lhs_batch.iter().map(|&x| x as i64).collect(),
                    rhs_batch: rhs_batch.iter().map(|&x| x as i64).collect(),
                };
                self.entry.dot_general(a, b, dn, out_shape)
            }

            Op::LayerNorm { axis, eps } => self.lower_layernorm(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *axis,
                *eps,
                out_shape,
            ),

            Op::RmsNorm { axis, eps } => self.lower_rmsnorm(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *axis,
                *eps,
                out_shape,
            ),

            Op::FusedResidualLN { has_bias, eps } => {
                self.lower_fused_residual_ln(&n.inputs, *has_bias, *eps, out_shape)
            }

            Op::FusedMatMulBiasAct { activation } => {
                self.lower_fused_matmul_bias_act(&n.inputs, *activation, out_shape)
            }

            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale: _,
                attn_logit_softcap: _,
            } => self.lower_attention(&n.inputs, *num_heads, *head_dim, *mask_kind, out_shape),

            Op::Rope { head_dim, n_rot: _ } => {
                self.lower_rope(n.inputs[0], n.inputs[1], n.inputs[2], *head_dim, out_shape)
            }

            Op::Reshape { new_shape: _ } => {
                let x = self.hlo(n.inputs[0]);
                self.entry.reshape(x, out_shape)
            }
            Op::Transpose { perm } => {
                let x = self.hlo(n.inputs[0]);
                let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
                self.entry.transpose(x, &perm_i64, out_shape)
            }
            Op::Narrow { axis, start, len } => {
                let x = self.hlo(n.inputs[0]);
                let in_dims = self.ir_shape_dims(n.inputs[0]);
                let mut starts = vec![0i64; in_dims.len()];
                let mut limits = in_dims.clone();
                let strides = vec![1i64; in_dims.len()];
                starts[*axis] = *start as i64;
                limits[*axis] = (*start + *len) as i64;
                self.entry.slice(x, &starts, &limits, &strides, out_shape)
            }
            Op::Concat { axis } => {
                let xs: Vec<i64> = n.inputs.iter().map(|&id| self.hlo(id)).collect();
                self.entry.concat(&xs, *axis as i64, out_shape)
            }
            Op::Expand { target_shape: _ } => {
                let x = self.hlo(n.inputs[0]);
                let in_dims = self.ir_shape_dims(n.inputs[0]);
                self.broadcast_to_target(x, &in_dims, out_shape)
            }
            Op::Gather { axis } => self.lower_gather(n.inputs[0], n.inputs[1], *axis, out_shape),

            Op::Reduce { op, axes, keep_dim } => {
                self.lower_reduce(n.inputs[0], *op, axes, *keep_dim, out_shape)
            }

            Op::Softmax { axis } => self.lower_softmax(n.inputs[0], *axis, out_shape),

            Op::Cumsum { axis, exclusive } => {
                self.lower_cumsum(n.inputs[0], *axis, *exclusive, out_shape)
            }

            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => self.lower_conv(
                n.inputs[0],
                n.inputs[1],
                kernel_size,
                stride,
                padding,
                dilation,
                *groups,
                out_shape,
            ),

            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => self.lower_pool(n.inputs[0], *kind, kernel_size, stride, padding, out_shape),

            Op::ScatterAdd => self.lower_scatter_add(n.inputs[0], n.inputs[1], out_shape),

            Op::TopK { k } => self.lower_topk(n.inputs[0], *k, out_shape),

            Op::GroupedMatMul => {
                self.lower_grouped_matmul(n.inputs[0], n.inputs[1], n.inputs[2], out_shape)
            }

            Op::DequantMatMul { scheme } => self.lower_dequant_matmul(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.inputs[3],
                *scheme,
                out_shape,
            ),

            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => self.lower_qmatmul(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                *x_zp,
                *w_zp,
                *out_zp,
                *mult,
                out_shape,
            ),

            Op::QConv2d {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => self.lower_qconv2d(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                kernel_size,
                stride,
                padding,
                dilation,
                *groups,
                *x_zp,
                *w_zp,
                *out_zp,
                *mult,
                out_shape,
            ),

            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => self.lower_sample(n.inputs[0], *top_k, *top_p, *temperature, *seed, out_shape),

            Op::SelectiveScan { state_size } => self.lower_selective_scan(
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.inputs[3],
                n.inputs[4],
                *state_size,
                out_shape,
            ),

            // Backward / training ops — no rlx-tpu support yet.
            Op::ReluBackward
            | Op::ActivationBackward { .. }
            | Op::MaxPool2dBackward { .. }
            | Op::Conv2dBackwardInput { .. }
            | Op::Conv2dBackwardWeight { .. }
            | Op::SoftmaxCrossEntropyWithLogits
            | Op::SoftmaxCrossEntropyBackward
            | Op::LayerNormBackwardInput { .. }
            | Op::LayerNormBackwardGamma { .. }
            | Op::FakeQuantize { .. }
            | Op::FakeQuantizeBackward { .. }
            | Op::FakeQuantizeLSQ { .. }
            | Op::FakeQuantizeLSQBackwardX { .. }
            | Op::FakeQuantizeLSQBackwardScale { .. } => panic!(
                "rlx-tpu: training/backward op {:?} not supported — \
                 inference only.",
                n.op
            ),

            // Should have been removed by unfuse — reaching here is
            // a bug in the compile pipeline.
            Op::FusedSwiGLU { .. }
            | Op::LoraMatMul { .. }
            | Op::FusedAttentionBlock { .. }
            | Op::FusedTransformerLayer { .. }
            | Op::If { .. }
            | Op::While { .. } => panic!(
                "rlx-tpu: composed op {:?} should have been unfused \
                 before lowering — bug in pipeline.",
                n.op
            ),

            // Custom ops have no XLA/PJRT-side lowering today: PJRT
            // doesn't expose a `custom_call` we own, and the kernel
            // would need to be a separately-loaded XLA plugin. Reject
            // explicitly so the failure names the op rather than
            // bottoming out as an obscure HLO error.
            Op::Custom { name, .. } => panic!(
                "rlx-tpu: Op::Custom('{name}') has no TPU lowering. \
                 Custom ops are CPU-only today; either move this op \
                 onto Device::Cpu or contribute an XLA-side lowering.",
            ),

            // DenseSolve is CPU-only today (uses LAPACK dgesv). No
            // XLA equivalent in our lowering yet.
            Op::DenseSolve => panic!(
                "rlx-tpu: Op::DenseSolve has no TPU lowering — \
                 use Device::Cpu for graphs containing dense solves.",
            ),

            // Scan was added recently and isn't lowered to HLO yet.
            // Decompose via unfuse before reaching this point.
            Op::Scan { .. }
            | Op::ScanBackward { .. }
            | Op::ScanBackwardXs { .. }
            | Op::BatchedDenseSolve
            | Op::CustomFn { .. }
            | Op::Fft { .. } => panic!(
                "rlx-tpu: Op::Scan / Scan-backward / BatchedDenseSolve / \
                 CustomFn / Fft have no TPU lowering yet — use Device::Cpu.",
            ),

            Op::GaussianSplatRender { .. } | Op::GaussianSplatRenderBackward { .. } => panic!(
                "rlx-tpu: Gaussian splat ops are host-only; graphs containing \
                 them must compile via segmented orchestration (not whole-graph HLO)."
            ),

            _ => panic!("rlx-tpu: unsupported op {:?}", n.op),
        }
    }

    // ── Constants ──────────────────────────────────────────────

    fn lower_constant(&self, data: &[u8], shape: Shape, dt: DType) -> i64 {
        // Decode the bytes per dtype into the matching LiteralData
        // variant. Constants in rlx-ir are stored as native-endian
        // bytes, but on the platforms we run (Mac / Linux x86_64 /
        // aarch64) that's little-endian — same as proto wire.
        let n = (shape.num_elements() as usize).max(1);
        let lit = match dt {
            DType::F32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(f32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::F32(v),
                }
            }
            DType::F64 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&data[i * 8..i * 8 + 8]);
                    v.push(f64::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::F64(v),
                }
            }
            DType::F16 => Literal {
                shape: shape.clone(),
                data: LiteralData::F16Bytes(data.to_vec()),
            },
            DType::BF16 => Literal {
                shape: shape.clone(),
                data: LiteralData::BF16Bytes(data.to_vec()),
            },
            DType::I8 => Literal {
                shape: shape.clone(),
                data: LiteralData::S8Bytes(data.to_vec()),
            },
            DType::U8 => Literal {
                shape: shape.clone(),
                data: LiteralData::U8(data.to_vec()),
            },
            DType::Bool => Literal {
                shape: shape.clone(),
                data: LiteralData::Pred(data.to_vec()),
            },
            DType::I16 => {
                // s16s field uses raw bytes per upstream proto; emit as bytes.
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S8Bytes(data.to_vec()),
                }
            }
            DType::I32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(i32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S32(v),
                }
            }
            DType::I64 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&data[i * 8..i * 8 + 8]);
                    v.push(i64::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::S64(v),
                }
            }
            DType::U32 => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[i * 4..i * 4 + 4]);
                    v.push(u32::from_le_bytes(b));
                }
                Literal {
                    shape: shape.clone(),
                    data: LiteralData::U32(v),
                }
            }
            DType::C64 => panic!("rlx-tpu: DType::C64 (complex) not yet supported"),
        };
        self.entry.constant(lit)
    }

    // ── Activation ─────────────────────────────────────────────

    fn lower_activation(&self, act: Activation, x: i64, shape: Shape) -> i64 {
        let elt = shape.element_type;
        match act {
            // Direct HLO unary opcodes.
            Activation::Exp => self.entry.unary("exponential", x, shape),
            Activation::Log => self.entry.unary("log", x, shape),
            Activation::Sqrt => self.entry.unary("sqrt", x, shape),
            Activation::Rsqrt => self.entry.unary("rsqrt", x, shape),
            Activation::Neg => self.entry.unary("negate", x, shape),
            Activation::Abs => self.entry.unary("abs", x, shape),
            Activation::Round => self.entry.unary("round-nearest-even", x, shape),
            Activation::Sin => self.entry.unary("sine", x, shape),
            Activation::Cos => self.entry.unary("cosine", x, shape),
            Activation::Tanh => self.entry.unary("tanh", x, shape),
            // sigmoid(x) → HLO `logistic`.
            Activation::Sigmoid => self.entry.unary("logistic", x, shape),
            // silu(x) = x * sigmoid(x).
            Activation::Silu => {
                let s = self.entry.unary("logistic", x, shape.clone());
                self.entry.binary("multiply", x, s, shape)
            }
            // relu(x) = max(x, 0).
            Activation::Relu => {
                let zero = self.entry.constant(Literal {
                    shape: Shape::scalar(elt),
                    data: LiteralData::F32(vec![0.0]), // ignored for non-f32; see below
                });
                // For non-f32 element types, use a typed scalar zero
                // by converting f32 0.0 through `convert`.
                let zero = if elt == prim::F32 {
                    zero
                } else {
                    self.entry.convert(zero, Shape::scalar(elt))
                };
                let zero_b = self.entry.broadcast(zero, &[], shape.clone());
                self.entry.binary("maximum", x, zero_b, shape)
            }
            // GELU exact: 0.5 * x * (1 + erf(x / sqrt(2))).
            Activation::Gelu => {
                let half = self.const_in_dtype(elt, 0.5);
                let one = self.const_in_dtype(elt, 1.0);
                let inv_sqrt_2 = self.const_in_dtype(elt, std::f32::consts::FRAC_1_SQRT_2);
                let half_b = self.entry.broadcast(half, &[], shape.clone());
                let one_b = self.entry.broadcast(one, &[], shape.clone());
                let inv_b = self.entry.broadcast(inv_sqrt_2, &[], shape.clone());
                let scaled = self.entry.binary("multiply", x, inv_b, shape.clone());
                let erfed = self.entry.unary("erf", scaled, shape.clone());
                let one_plus = self.entry.binary("add", one_b, erfed, shape.clone());
                let half_x = self.entry.binary("multiply", x, half_b, shape.clone());
                self.entry.binary("multiply", half_x, one_plus, shape)
            }
            // GELU approx (tanh form):
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
            Activation::GeluApprox => {
                let half = self.const_in_dtype(elt, 0.5);
                let one = self.const_in_dtype(elt, 1.0);
                let c = self.const_in_dtype(elt, (2.0_f32 / std::f32::consts::PI).sqrt());
                let k = self.const_in_dtype(elt, 0.044715);
                let half_b = self.entry.broadcast(half, &[], shape.clone());
                let one_b = self.entry.broadcast(one, &[], shape.clone());
                let c_b = self.entry.broadcast(c, &[], shape.clone());
                let k_b = self.entry.broadcast(k, &[], shape.clone());
                let x2 = self.entry.binary("multiply", x, x, shape.clone());
                let x3 = self.entry.binary("multiply", x2, x, shape.clone());
                let kx3 = self.entry.binary("multiply", k_b, x3, shape.clone());
                let inner = self.entry.binary("add", x, kx3, shape.clone());
                let scaled = self.entry.binary("multiply", c_b, inner, shape.clone());
                let tanhed = self.entry.unary("tanh", scaled, shape.clone());
                let one_plus = self.entry.binary("add", one_b, tanhed, shape.clone());
                let half_x = self.entry.binary("multiply", x, half_b, shape.clone());
                self.entry.binary("multiply", half_x, one_plus, shape)
            }
            Activation::Tan => self.entry.unary("tan", x, shape),
            Activation::Atan => self.entry.unary("atan", x, shape),
        }
    }

    /// Scalar constant in the given primitive dtype (F32 or down-cast).
    fn const_in_dtype(&self, prim_ty: i32, v: f32) -> i64 {
        let f = self.entry.constant_f32_scalar(v);
        if prim_ty == prim::F32 {
            f
        } else {
            self.entry.convert(f, Shape::scalar(prim_ty))
        }
    }

    /// Build a scale/zero-point broadcast for `Op::Quantize` /
    /// `Op::Dequantize`. `axis = None` → scalar broadcast (per-tensor);
    /// `axis = Some(d)` → 1-D constant of length `out_dims[d]`
    /// broadcast along the channel axis.
    fn broadcast_q_factor(
        &self,
        axis: Option<usize>,
        values: &[f32],
        out_dims: &[i64],
        prim_ty: i32,
    ) -> i64 {
        let out_shape = Shape::array(prim_ty, out_dims);
        match axis {
            None => {
                let v = values.first().copied().unwrap_or(0.0);
                let c = self.const_in_dtype(prim_ty, v);
                self.entry.broadcast(c, &[], out_shape)
            }
            Some(d) => {
                // Materialize a [N] f32 constant where N = out_dims[d].
                // Convert to target dtype if needed, then broadcast
                // along axis d (broadcast_dims = [d]).
                let n = out_dims[d];
                debug_assert_eq!(
                    values.len() as i64,
                    n,
                    "Quantize/Dequantize: per-channel values len ({}) \
                     must match output dim[{}] ({})",
                    values.len(),
                    d,
                    n
                );
                let lit = crate::hlo::Literal {
                    shape: Shape::array(prim::F32, &[n]),
                    data: crate::hlo::LiteralData::F32(values.to_vec()),
                };
                let c = self.entry.constant(lit);
                let c = if prim_ty == prim::F32 {
                    c
                } else {
                    self.entry.convert(c, Shape::array(prim_ty, &[n]))
                };
                self.entry.broadcast(c, &[d as i64], out_shape)
            }
        }
    }

    // ── Binary ─────────────────────────────────────────────────

    fn lower_binary(
        &self,
        op: BinaryOp,
        a: i64,
        b: i64,
        a_id: NodeId,
        b_id: NodeId,
        out: Shape,
    ) -> i64 {
        let opcode = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "subtract",
            BinaryOp::Mul => "multiply",
            BinaryOp::Div => "divide",
            BinaryOp::Max => "maximum",
            BinaryOp::Min => "minimum",
            BinaryOp::Pow => "power",
        };
        let (a, b) = self.broadcast_pair_to(a, b, a_id, b_id, &out.dimensions);
        self.entry.binary(opcode, a, b, out)
    }

    /// Bring two operands to a common rank-aligned shape against
    /// `target_dims`. HLO requires both binary operands to have the
    /// same shape; we use `broadcast_align` to lift each one to target.
    fn broadcast_pair_to(
        &self,
        a: i64,
        b: i64,
        a_id: NodeId,
        b_id: NodeId,
        target_dims: &[i64],
    ) -> (i64, i64) {
        let a_dims = self.ir_shape_dims(a_id);
        let b_dims = self.ir_shape_dims(b_id);
        let a_dt = self.dtype(a_id);
        let b_dt = self.dtype(b_id);
        let target_a = Shape::array(prim_of(a_dt), target_dims);
        let target_b = Shape::array(prim_of(b_dt), target_dims);
        let a2 = self.broadcast_to_target(a, &a_dims, target_a);
        let b2 = self.broadcast_to_target(b, &b_dims, target_b);
        (a2, b2)
    }

    /// Broadcast `x` to `target_shape`. Adds leading dims when
    /// `x_dims.len() < target.rank()`, or replicates size-1 axes when
    /// rank matches.
    fn broadcast_to_target(&self, x: i64, x_dims: &[i64], target: Shape) -> i64 {
        let target_dims = target.dimensions.clone();
        if x_dims == target_dims.as_slice() {
            return x;
        }
        if x_dims.len() < target_dims.len() {
            // Pad to right (broadcast adds leading dims).
            let target_rank = target_dims.len();
            let broadcast_dims: Vec<i64> = (target_rank - x_dims.len()..target_rank)
                .map(|i| i as i64)
                .collect();
            // The intermediate shape is x's dims placed at trailing
            // positions of target, with leading dims taken from
            // target. HLO infers the result shape from `target`.
            return self.entry.broadcast(x, &broadcast_dims, target);
        }
        self.broadcast_align(x, x_dims, target)
    }

    // ── ElementwiseRegion ─────────────────────────────────────

    fn lower_elementwise_region(
        &mut self,
        inputs: &[NodeId],
        chain: &[ChainStep],
        num_inputs: u32,
        scalar_input_mask: u32,
        input_modulus: &[u32; 16],
        out_shape: Shape,
    ) -> i64 {
        // Walk the chain, materializing each step as a regular HLO
        // op. ChainOperand::Input(i) refers to inputs[i] (broadcast
        // to output shape if scalar/tiled). ChainOperand::Step(i)
        // refers to the i-th already-emitted step result.
        let n = num_inputs as usize;
        let mut input_hlo: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            let id = self.hlo(inputs[i]);
            let in_dims = self.ir_shape_dims(inputs[i]);
            let in_dt = self.dtype(inputs[i]);
            let target = Shape::array(prim_of(in_dt), &out_shape.dimensions);
            let scalar = scalar_input_mask & (1u32 << i) != 0;
            let _ = input_modulus[i]; // tiling fully captured by broadcast_to_target
            let placed = if scalar {
                self.entry.broadcast(id, &[], target)
            } else {
                self.broadcast_to_target(id, &in_dims, target)
            };
            input_hlo.push(placed);
        }
        let mut step_results: Vec<i64> = Vec::with_capacity(chain.len());

        let resolve = |op: &ChainOperand, ins: &[i64], steps: &[i64]| -> i64 {
            match op {
                ChainOperand::Input(i) => ins[*i as usize],
                ChainOperand::Step(i) => steps[*i as usize],
            }
        };
        for step in chain {
            let result = match step {
                ChainStep::Activation(act, src) => {
                    let x = resolve(src, &input_hlo, &step_results);
                    self.lower_activation(*act, x, out_shape.clone())
                }
                ChainStep::Cast(dt, src) => {
                    let x = resolve(src, &input_hlo, &step_results);
                    self.entry
                        .convert(x, Shape::array(prim_of(*dt), &out_shape.dimensions))
                }
                ChainStep::Binary(op, lhs, rhs) => {
                    let a = resolve(lhs, &input_hlo, &step_results);
                    let b = resolve(rhs, &input_hlo, &step_results);
                    let opcode = match op {
                        BinaryOp::Add => "add",
                        BinaryOp::Sub => "subtract",
                        BinaryOp::Mul => "multiply",
                        BinaryOp::Div => "divide",
                        BinaryOp::Max => "maximum",
                        BinaryOp::Min => "minimum",
                        BinaryOp::Pow => "power",
                    };
                    self.entry.binary(opcode, a, b, out_shape.clone())
                }
                ChainStep::Compare(op, lhs, rhs) => {
                    let a = resolve(lhs, &input_hlo, &step_results);
                    let b = resolve(rhs, &input_hlo, &step_results);
                    let dir = match op {
                        CmpOp::Eq => "EQ",
                        CmpOp::Ne => "NE",
                        CmpOp::Lt => "LT",
                        CmpOp::Le => "LE",
                        CmpOp::Gt => "GT",
                        CmpOp::Ge => "GE",
                    };
                    self.entry
                        .compare(a, b, dir, Shape::pred(&out_shape.dimensions))
                }
                ChainStep::Where(c, t, f) => {
                    let cv = resolve(c, &input_hlo, &step_results);
                    let tv = resolve(t, &input_hlo, &step_results);
                    let fv = resolve(f, &input_hlo, &step_results);
                    self.entry.select(cv, tv, fv, out_shape.clone())
                }
            };
            step_results.push(result);
        }
        // Output is the last step's result.
        *step_results.last().unwrap_or(&0)
    }

    // ── MatMul ─────────────────────────────────────────────────

    fn lower_matmul(&mut self, a_id: NodeId, b_id: NodeId, out: Shape) -> i64 {
        let a = self.hlo(a_id);
        let b = self.hlo(b_id);
        let a_dims = self.ir_shape_dims(a_id);
        let b_dims = self.ir_shape_dims(b_id);
        // [..., M, K] × [..., K, N] → [..., M, N] with batch dims
        // broadcast. For HLO, the cleanest expression is:
        //   contracting: lhs=last, rhs=second_to_last
        //   batch: leading dims (must match exactly; we materialize
        //          a broadcast first if they don't).
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();
        let (max_rank, a_b, b_b) = match (a_rank, b_rank) {
            (r, s) if r == s => (r, a, b),
            (r, s) if r < s => {
                // Add leading 1s to a, then broadcast.
                let pad = s - r;
                let mut tgt = vec![1i64; pad];
                tgt.extend_from_slice(&a_dims);
                let r1 = self.entry.reshape(a,
                    Shape::array(prim_of(self.dtype(a_id)), &tgt));
                let mut full = b_dims[..pad].to_vec();
                full.extend_from_slice(&a_dims);
                let r2 = self.broadcast_to_target(r1, &tgt,
                    Shape::array(prim_of(self.dtype(a_id)), &full));
                (s, r2, b)
            }
            (r, s) /* r > s */ => {
                let pad = r - s;
                let mut tgt = vec![1i64; pad];
                tgt.extend_from_slice(&b_dims);
                let r1 = self.entry.reshape(b,
                    Shape::array(prim_of(self.dtype(b_id)), &tgt));
                let mut full = a_dims[..pad].to_vec();
                full.extend_from_slice(&b_dims);
                let r2 = self.broadcast_to_target(r1, &tgt,
                    Shape::array(prim_of(self.dtype(b_id)), &full));
                (r, a, r2)
            }
        };
        let contracting_a = (max_rank - 1) as i64;
        let contracting_b = (max_rank - 2) as i64;
        let batch: Vec<i64> = (0..max_rank as i64 - 2).collect();
        let dn = DotDimNumbers {
            lhs_contracting: vec![contracting_a],
            rhs_contracting: vec![contracting_b],
            lhs_batch: batch.clone(),
            rhs_batch: batch,
        };
        self.entry.dot_general(a_b, b_b, dn, out)
    }

    // ── LayerNorm ──────────────────────────────────────────────

    fn lower_layernorm(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        beta_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let beta = self.hlo(beta_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = if axis < 0 {
            (rank as i32 + axis) as i64
        } else {
            axis as i64
        };

        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let kept_shape = Shape::array(prim_ty, &reduced);

        // mean = sum(x) / N
        let summed = self.reduce_one(
            x,
            ax,
            "add",
            0.0,
            x_dt,
            x_dims
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
                .collect(),
        );
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let summed_shape_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let summed_shape = Shape::array(prim_ty, &summed_shape_dims);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let mean = self
            .entry
            .binary("divide", summed, n_b, summed_shape.clone());
        // Reshape to keep the reduced axis as size 1, then broadcast back.
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());

        // centered = x - mean
        let centered = self.entry.binary("subtract", x, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());

        // var = sum(sq) / N
        let var_summed = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_shape_dims.clone());
        let var = self
            .entry
            .binary("divide", var_summed, n_b, summed_shape.clone());
        let var_kept = self.entry.reshape(var, kept_shape);

        let eps_c = self.const_in_dtype(prim_ty, eps);
        let var_eps_kept_shape = Shape::array(prim_ty, &reduced);
        let eps_b = self.entry.broadcast(eps_c, &[], var_eps_kept_shape.clone());
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, var_eps_kept_shape.clone());
        let inv_std = self.entry.unary("rsqrt", var_eps, var_eps_kept_shape);
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());

        let normed = self
            .entry
            .binary("multiply", centered, inv_std_b, out.clone());

        // scaled = normed * gamma + beta (gamma/beta have axis-only shape).
        let g_dims = self.ir_shape_dims(gamma_id);
        let b_dims = self.ir_shape_dims(beta_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let b_b = self.broadcast_param_to_axis(beta, &b_dims, ax, &x_dims, prim_ty);
        let scaled = self.entry.binary("multiply", normed, g_b, out.clone());
        self.entry.binary("add", scaled, b_b, out)
    }

    /// Lift a 1-D normalization parameter (shape `[axis_size]`) up to
    /// the layout `x` uses, by reshaping to size-1 in every axis
    /// except `axis` then broadcasting.
    fn broadcast_param_to_axis(
        &self,
        p: i64,
        p_dims: &[i64],
        axis: i64,
        x_dims: &[i64],
        prim_ty: i32,
    ) -> i64 {
        let target = Shape::array(prim_ty, x_dims);
        if p_dims == x_dims {
            return p;
        }
        if p_dims.len() == 1 {
            // [N] → [1,1,...,N,...,1] then broadcast.
            let mut padded = vec![1i64; x_dims.len()];
            padded[axis as usize] = p_dims[0];
            let r = self.entry.reshape(p, Shape::array(prim_ty, &padded));
            return self.broadcast_align(r, &padded, target);
        }
        self.broadcast_to_target(p, p_dims, target)
    }

    // ── RmsNorm ────────────────────────────────────────────────

    fn lower_rmsnorm(
        &mut self,
        x_id: NodeId,
        gamma_id: NodeId,
        _beta_id: NodeId,
        axis: i32,
        eps: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let gamma = self.hlo(gamma_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = if axis < 0 {
            (rank as i32 + axis) as i64
        } else {
            axis as i64
        };

        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_shape = Shape::array(prim_ty, &summed_dims);

        let sq = self.entry.binary("multiply", x, x, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims.clone());
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let sq_mean = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());

        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self.entry.broadcast(eps_c, &[], summed_shape.clone());
        let var_eps = self
            .entry
            .binary("add", sq_mean, eps_b, summed_shape.clone());
        let inv = self.entry.unary("rsqrt", var_eps, summed_shape);
        let inv_kept = self.entry.reshape(inv, kept_shape);
        let inv_b = self.broadcast_align(inv_kept, &reduced, out.clone());
        let normed = self.entry.binary("multiply", x, inv_b, out.clone());
        let g_dims = self.ir_shape_dims(gamma_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        self.entry.binary("multiply", normed, g_b, out)
    }

    // ── FusedResidualLN ────────────────────────────────────────

    fn lower_fused_residual_ln(
        &mut self,
        inputs: &[NodeId],
        has_bias: bool,
        eps: f32,
        out: Shape,
    ) -> i64 {
        // inputs: [x, residual, [bias], gamma, beta]
        let x = self.hlo(inputs[0]);
        let r = self.hlo(inputs[1]);
        let summed = self.entry.binary("add", x, r, out.clone());
        let pre_ln = if has_bias {
            let b = self.hlo(inputs[2]);
            let b_dims = self.ir_shape_dims(inputs[2]);
            let target = out.clone();
            let b_b = self.broadcast_to_target(b, &b_dims, target);
            self.entry.binary("add", summed, b_b, out.clone())
        } else {
            summed
        };
        let (gi, bi) = if has_bias { (3, 4) } else { (2, 3) };
        let gamma_id = inputs[gi];
        let beta_id = inputs[bi];
        // Synthesize a temporary IR-less LayerNorm by going through
        // lower_layernorm's mechanics directly. We compute mean / var
        // over axis -1 (matches all CPU/Metal/CUDA emitters).
        let x_dims = out.dimensions.clone();
        let x_dt = self.graph.node(inputs[0]).shape.dtype();
        let prim_ty = prim_of(x_dt);
        let rank = x_dims.len();
        let ax = (rank - 1) as i64;
        let mut reduced = x_dims.clone();
        reduced[ax as usize] = 1;
        let summed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let kept_shape = Shape::array(prim_ty, &reduced);
        let summed_shape = Shape::array(prim_ty, &summed_dims);

        let pre_sum = self.reduce_one(pre_ln, ax, "add", 0.0, x_dt, summed_dims.clone());
        let n = x_dims[ax as usize] as f32;
        let n_c = self.const_in_dtype(prim_ty, n);
        let n_b = self.entry.broadcast(n_c, &[], summed_shape.clone());
        let mean = self
            .entry
            .binary("divide", pre_sum, n_b, summed_shape.clone());
        let mean_kept = self.entry.reshape(mean, kept_shape.clone());
        let mean_b = self.broadcast_align(mean_kept, &reduced, out.clone());
        let centered = self.entry.binary("subtract", pre_ln, mean_b, out.clone());
        let sq = self
            .entry
            .binary("multiply", centered, centered, out.clone());
        let sq_sum = self.reduce_one(sq, ax, "add", 0.0, x_dt, summed_dims);
        let var = self
            .entry
            .binary("divide", sq_sum, n_b, summed_shape.clone());
        let var_kept = self.entry.reshape(var, kept_shape);
        let eps_c = self.const_in_dtype(prim_ty, eps);
        let eps_b = self
            .entry
            .broadcast(eps_c, &[], Shape::array(prim_ty, &reduced));
        let var_eps = self
            .entry
            .binary("add", var_kept, eps_b, Shape::array(prim_ty, &reduced));
        let inv_std = self
            .entry
            .unary("rsqrt", var_eps, Shape::array(prim_ty, &reduced));
        let inv_std_b = self.broadcast_align(inv_std, &reduced, out.clone());
        let normed = self
            .entry
            .binary("multiply", centered, inv_std_b, out.clone());

        let gamma = self.hlo(gamma_id);
        let beta = self.hlo(beta_id);
        let g_dims = self.ir_shape_dims(gamma_id);
        let b_dims = self.ir_shape_dims(beta_id);
        let g_b = self.broadcast_param_to_axis(gamma, &g_dims, ax, &x_dims, prim_ty);
        let b_b = self.broadcast_param_to_axis(beta, &b_dims, ax, &x_dims, prim_ty);
        let scaled = self.entry.binary("multiply", normed, g_b, out.clone());
        self.entry.binary("add", scaled, b_b, out)
    }

    // ── FusedMatMulBiasAct ─────────────────────────────────────

    fn lower_fused_matmul_bias_act(
        &mut self,
        inputs: &[NodeId],
        activation: Option<Activation>,
        out: Shape,
    ) -> i64 {
        let mm = self.lower_matmul(inputs[0], inputs[1], out.clone());
        let bias = self.hlo(inputs[2]);
        let b_dims = self.ir_shape_dims(inputs[2]);
        let bias_b = self.broadcast_to_target(bias, &b_dims, out.clone());
        let added = self.entry.binary("add", mm, bias_b, out.clone());
        match activation {
            None => added,
            Some(act) => self.lower_activation(act, added, out),
        }
    }

    // ── Attention ──────────────────────────────────────────────

    fn lower_attention(
        &mut self,
        inputs: &[NodeId],
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        out: Shape,
    ) -> i64 {
        // Inputs: Q, K, V [, mask].
        // After unfuse, all are rank-4 [B, H, S, D] (rank-3 was promoted).
        let q = self.hlo(inputs[0]);
        let k = self.hlo(inputs[1]);
        let v = self.hlo(inputs[2]);
        let q_dims = self.ir_shape_dims(inputs[0]);
        let k_dims = self.ir_shape_dims(inputs[1]);
        let dt = self.dtype(inputs[0]);
        let prim_ty = prim_of(dt);
        let _ = num_heads;
        let _ = head_dim;
        let b_dim = q_dims[0];
        let h_dim = q_dims[1];
        let s_q = q_dims[2];
        let s_k = k_dims[2];
        let d_dim = q_dims[3];

        // QK^T: [B, H, S_q, D] x [B, H, S_k, D] → [B, H, S_q, S_k]
        // Contracting axis = 3 on both sides; batch = [0, 1].
        let qk_shape = Shape::array(prim_ty, &[b_dim, h_dim, s_q, s_k]);
        let qk_dn = DotDimNumbers {
            lhs_contracting: vec![3],
            rhs_contracting: vec![3],
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 1],
        };
        let qk = self.entry.dot_general(q, k, qk_dn, qk_shape.clone());
        // Scale by 1 / sqrt(d).
        let scale = self.const_in_dtype(prim_ty, 1.0 / (d_dim as f32).sqrt());
        let scale_b = self.entry.broadcast(scale, &[], qk_shape.clone());
        let scaled = self.entry.binary("multiply", qk, scale_b, qk_shape.clone());

        // Apply mask.
        let masked = match mask_kind {
            MaskKind::None => scaled,
            MaskKind::Causal => self.apply_causal_mask(scaled, qk_shape.clone(), s_q, s_k, prim_ty),
            MaskKind::SlidingWindow(w) => self.apply_sliding_window_mask(
                scaled,
                qk_shape.clone(),
                s_q,
                s_k,
                w as i64,
                prim_ty,
            ),
            MaskKind::Custom | MaskKind::Bias => {
                // 4th input is the mask, additive [B, ?, S_q, S_k].
                let mask = self.hlo(inputs[3]);
                let mask_dims = self.ir_shape_dims(inputs[3]);
                let mask_b = self.broadcast_to_target(mask, &mask_dims, qk_shape.clone());
                self.entry.binary("add", scaled, mask_b, qk_shape.clone())
            }
        };

        // Softmax along last axis.
        let probs = self.lower_softmax_id(masked, qk_shape.clone(), 3);

        // probs @ V: [B, H, S_q, S_k] x [B, H, S_k, D] → [B, H, S_q, D]
        let av_dn = DotDimNumbers {
            lhs_contracting: vec![3],
            rhs_contracting: vec![2],
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 1],
        };
        self.entry.dot_general(probs, v, av_dn, out)
    }

    /// Synthesize and add a causal mask to QK^T in HLO using
    /// `iota` + `compare` + `select`. Avoids materializing a mask
    /// tensor on the host.
    fn apply_causal_mask(
        &self,
        scaled: i64,
        qk_shape: Shape,
        _s_q: i64,
        _s_k: i64,
        prim_ty: i32,
    ) -> i64 {
        let q_idx = self
            .entry
            .iota(2, Shape::array(prim::S32, &qk_shape.dimensions));
        let k_idx = self
            .entry
            .iota(3, Shape::array(prim::S32, &qk_shape.dimensions));
        let mask = self
            .entry
            .compare(q_idx, k_idx, "GE", Shape::pred(&qk_shape.dimensions));
        let neg_inf = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let neg_inf_b = self.entry.broadcast(neg_inf, &[], qk_shape.clone());
        self.entry.select(mask, scaled, neg_inf_b, qk_shape)
    }

    /// Sliding-window mask: q attends to k in [q-w, q].
    fn apply_sliding_window_mask(
        &self,
        scaled: i64,
        qk_shape: Shape,
        _s_q: i64,
        _s_k: i64,
        w: i64,
        prim_ty: i32,
    ) -> i64 {
        let q_idx = self
            .entry
            .iota(2, Shape::array(prim::S32, &qk_shape.dimensions));
        let k_idx = self
            .entry
            .iota(3, Shape::array(prim::S32, &qk_shape.dimensions));
        let lower = self
            .entry
            .compare(q_idx, k_idx, "GE", Shape::pred(&qk_shape.dimensions));
        // q - k <= w  →  k >= q - w
        let qmw = self.entry.constant(Literal {
            shape: Shape::scalar(prim::S32),
            data: LiteralData::S32(vec![w as i32]),
        });
        let qmw_b = self
            .entry
            .broadcast(qmw, &[], Shape::array(prim::S32, &qk_shape.dimensions));
        let q_minus_w = self.entry.binary(
            "subtract",
            q_idx,
            qmw_b,
            Shape::array(prim::S32, &qk_shape.dimensions),
        );
        let upper = self
            .entry
            .compare(k_idx, q_minus_w, "GE", Shape::pred(&qk_shape.dimensions));
        let mask = self
            .entry
            .binary("and", lower, upper, Shape::pred(&qk_shape.dimensions));
        let neg_inf = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let neg_inf_b = self.entry.broadcast(neg_inf, &[], qk_shape.clone());
        self.entry.select(mask, scaled, neg_inf_b, qk_shape)
    }

    // ── Rope ───────────────────────────────────────────────────

    fn lower_rope(
        &mut self,
        x_id: NodeId,
        cos_id: NodeId,
        sin_id: NodeId,
        head_dim: usize,
        out: Shape,
    ) -> i64 {
        // Standard non-interleaved RoPE: split last dim in halves
        // (x1, x2). Output: [x1*cos - x2*sin, x1*sin + x2*cos].
        let x = self.hlo(x_id);
        let cos = self.hlo(cos_id);
        let sin = self.hlo(sin_id);
        let x_dims = self.ir_shape_dims(x_id);
        let dt = self.dtype(x_id);
        let prim_ty = prim_of(dt);
        let half = head_dim / 2;
        let last = x_dims.len() - 1;

        let mut starts1 = vec![0i64; x_dims.len()];
        let mut limits1 = x_dims.clone();
        let mut starts2 = vec![0i64; x_dims.len()];
        let mut limits2 = x_dims.clone();
        let strides = vec![1i64; x_dims.len()];
        starts1[last] = 0;
        limits1[last] = half as i64;
        starts2[last] = half as i64;
        limits2[last] = head_dim as i64;
        let mut half_dims = x_dims.clone();
        half_dims[last] = half as i64;
        let half_shape = Shape::array(prim_ty, &half_dims);
        let x1 = self
            .entry
            .slice(x, &starts1, &limits1, &strides, half_shape.clone());
        let x2 = self
            .entry
            .slice(x, &starts2, &limits2, &strides, half_shape.clone());

        let cos_dims = self.ir_shape_dims(cos_id);
        let sin_dims = self.ir_shape_dims(sin_id);
        let cos_b = self.broadcast_to_target(cos, &cos_dims, half_shape.clone());
        let sin_b = self.broadcast_to_target(sin, &sin_dims, half_shape.clone());

        let x1c = self.entry.binary("multiply", x1, cos_b, half_shape.clone());
        let x2s = self.entry.binary("multiply", x2, sin_b, half_shape.clone());
        let r1 = self.entry.binary("subtract", x1c, x2s, half_shape.clone());
        let x1s = self.entry.binary("multiply", x1, sin_b, half_shape.clone());
        let x2c = self.entry.binary("multiply", x2, cos_b, half_shape.clone());
        let r2 = self.entry.binary("add", x1s, x2c, half_shape);
        self.entry.concat(&[r1, r2], last as i64, out)
    }

    // ── Gather ─────────────────────────────────────────────────

    fn lower_gather(
        &mut self,
        table_id: NodeId,
        indices_id: NodeId,
        axis: usize,
        out: Shape,
    ) -> i64 {
        // Embedding-lookup style gather. Indices are integer
        // (we treat the index dtype as S32 for HLO; the IR may carry
        // them as f32-encoded so we convert if needed).
        let table = self.hlo(table_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            // Convert f32-encoded indices to s32.
            let idx_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &idx_dims))
        };
        let table_dims = self.ir_shape_dims(table_id);
        let idx_dims = self.ir_shape_dims(indices_id);
        let mut slice_sizes = table_dims.clone();
        slice_sizes[axis] = 1;
        // HLO gather output shape: indices' batch dims interleaved
        // with operand's offset dims. `offset_dims` lists the
        // OUTPUT positions that come from the operand (after
        // `collapsed_slice_dims` are dropped). For an embedding
        // lookup against a 2-D table [V, H] with indices [B, S],
        // the output is [B, S, H]: the offset dim H lands at output
        // position `idx_rank` (= 2). Generalizing: with
        // `n_offset = table_rank - collapsed_slice_dims.len()`
        // operand-derived dims, they occupy the trailing positions
        // [idx_rank .. idx_rank + n_offset).
        let n_offset = (table_dims.len() - 1) as i64;
        let idx_rank = idx_dims.len() as i64;
        let offset_dims: Vec<i64> = (idx_rank..idx_rank + n_offset).collect();
        let dn = GatherDimNumbers {
            offset_dims,
            collapsed_slice_dims: vec![axis as i64],
            start_index_map: vec![axis as i64],
            index_vector_dim: idx_rank,
        };
        self.entry.gather(table, idx_s32, dn, slice_sizes, out)
    }

    // ── Reduce ─────────────────────────────────────────────────

    fn lower_reduce(
        &mut self,
        x_id: NodeId,
        op: ReduceOp,
        axes: &[usize],
        keep_dim: bool,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let x_dims = self.ir_shape_dims(x_id);
        let x_dt = self.dtype(x_id);
        let prim_ty = prim_of(x_dt);
        let axes_i64: Vec<i64> = axes.iter().map(|&a| a as i64).collect();

        // Reducer + identity element + post-divide.
        let (opcode, init_v, divide_by_n) = match op {
            ReduceOp::Sum => ("add", 0.0_f32, false),
            ReduceOp::Mean => ("add", 0.0_f32, true),
            ReduceOp::Max => ("maximum", f32::NEG_INFINITY, false),
            ReduceOp::Min => ("minimum", f32::INFINITY, false),
            ReduceOp::Prod => ("multiply", 1.0_f32, false),
        };
        let red = self.reducer(opcode, prim_ty);
        let init = self.const_in_dtype(prim_ty, init_v);

        // Determine intermediate (no keep_dim) shape.
        let collapsed_dims: Vec<i64> = x_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
            .collect();
        let collapsed_shape = Shape::array(prim_ty, &collapsed_dims);
        let mut reduced = self
            .entry
            .reduce(x, init, &red, &axes_i64, collapsed_shape.clone());
        if matches!(op, ReduceOp::Mean) {
            let n: i64 = axes.iter().map(|&a| x_dims[a]).product();
            let n_c = self.const_in_dtype(prim_ty, n as f32);
            let n_b = self.entry.broadcast(n_c, &[], collapsed_shape.clone());
            reduced = self.entry.binary("divide", reduced, n_b, collapsed_shape);
        }
        let _ = divide_by_n;
        if keep_dim {
            self.entry.reshape(reduced, out)
        } else {
            reduced
        }
    }

    // ── Softmax ────────────────────────────────────────────────

    fn lower_softmax(&mut self, x_id: NodeId, axis: i32, out: Shape) -> i64 {
        let x = self.hlo(x_id);
        self.lower_softmax_id(x, out, axis as i64)
    }

    fn lower_softmax_id(&mut self, x: i64, out: Shape, axis: i64) -> i64 {
        let dims = out.dimensions.clone();
        let prim_ty = out.element_type;
        let rank = dims.len() as i64;
        let ax = if axis < 0 { rank + axis } else { axis };

        // Numerically-stable softmax: x' = x - max(x); y = exp(x') / sum(exp(x'))
        let collapsed: Vec<i64> = dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax as usize { None } else { Some(d) })
            .collect();
        let mut kept = dims.clone();
        kept[ax as usize] = 1;
        let collapsed_shape = Shape::array(prim_ty, &collapsed);
        let kept_shape = Shape::array(prim_ty, &kept);

        let red_max = self.reducer("maximum", prim_ty);
        let init_max = self.const_in_dtype(prim_ty, f32::NEG_INFINITY);
        let max_v = self
            .entry
            .reduce(x, init_max, &red_max, &[ax], collapsed_shape.clone());
        let max_kept = self.entry.reshape(max_v, kept_shape.clone());
        let max_b = self.broadcast_align(max_kept, &kept, out.clone());
        let centered = self.entry.binary("subtract", x, max_b, out.clone());
        let exped = self.entry.unary("exponential", centered, out.clone());

        let red_sum = self.reducer("add", prim_ty);
        let init_sum = self.const_in_dtype(prim_ty, 0.0);
        let sum_v = self
            .entry
            .reduce(exped, init_sum, &red_sum, &[ax], collapsed_shape);
        let sum_kept = self.entry.reshape(sum_v, kept_shape);
        let sum_b = self.broadcast_align(sum_kept, &kept, out.clone());
        self.entry.binary("divide", exped, sum_b, out)
    }

    // ── Cumsum ─────────────────────────────────────────────────

    fn lower_cumsum(&mut self, x_id: NodeId, axis: i32, exclusive: bool, out: Shape) -> i64 {
        // HLO has no `cumsum` primitive — use `reduce-window` with a
        // window that spans the whole prefix along the chosen axis.
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let rank = dims.len() as i32;
        let ax = if axis < 0 {
            (rank + axis) as i64
        } else {
            axis as i64
        };

        let init = self.const_in_dtype(prim_ty, 0.0);
        let red = self.reducer("add", prim_ty);

        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
            };
            dims.len()
        ];
        // Inclusive scan: window of size = full axis length, with
        // padding_low = N-1 so each prefix sees [0..i].
        window_dims[ax as usize] = WindowDim {
            size: dims[ax as usize],
            stride: 1,
            padding_low: dims[ax as usize] - 1,
            padding_high: 0,
            window_dilation: 1,
            base_dilation: 1,
        };
        let window = Window {
            dimensions: window_dims,
        };
        let scanned = self.entry.reduce_window(x, init, &red, window, out.clone());
        if exclusive {
            // Shift-right-by-one along axis: pad a leading 0 and slice.
            let zero = self.const_in_dtype(prim_ty, 0.0);
            let mut pad_cfg = vec![(0i64, 0i64, 0i64); dims.len()];
            pad_cfg[ax as usize] = (1, 0, 0);
            let mut padded_dims = dims.clone();
            padded_dims[ax as usize] += 1;
            let padded =
                self.entry
                    .pad(scanned, zero, pad_cfg, Shape::array(prim_ty, &padded_dims));
            let mut starts = vec![0i64; dims.len()];
            let mut limits = padded_dims.clone();
            let strides = vec![1i64; dims.len()];
            starts[ax as usize] = 0;
            limits[ax as usize] = dims[ax as usize];
            self.entry.slice(padded, &starts, &limits, &strides, out)
        } else {
            scanned
        }
    }

    // ── Conv ───────────────────────────────────────────────────

    fn lower_conv(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let x_rank = self.ir_shape_dims(x_id).len();
        // Convention: input is [N, C, *spatial], weight is
        // [C_out, C_in/groups, *spatial]. HLO Convolution expects an
        // explicit dimension-numbers proto.
        let n_spatial = x_rank - 2;
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: (2..2 + n_spatial as i64).collect(),
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: (2..2 + n_spatial as i64).collect(),
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: (2..2 + n_spatial as i64).collect(),
        };
        let mut window_dims = Vec::with_capacity(n_spatial);
        for i in 0..n_spatial {
            window_dims.push(WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: dilation[i] as i64,
                base_dilation: 1,
            });
        }
        let window = Window {
            dimensions: window_dims,
        };
        self.entry
            .convolution(x, w, window, cdn, groups as i64, out)
    }

    // ── Pool ───────────────────────────────────────────────────

    fn lower_pool(
        &mut self,
        x_id: NodeId,
        kind: ReduceOp,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let x_dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let n_spatial = x_dims.len() - 2;
        let (opcode, init_v) = match kind {
            ReduceOp::Sum | ReduceOp::Mean => ("add", 0.0_f32),
            ReduceOp::Max => ("maximum", f32::NEG_INFINITY),
            ReduceOp::Min => ("minimum", f32::INFINITY),
            ReduceOp::Prod => ("multiply", 1.0_f32),
        };
        let red = self.reducer(opcode, prim_ty);
        let init = self.const_in_dtype(prim_ty, init_v);

        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
            };
            x_dims.len()
        ];
        for i in 0..n_spatial {
            window_dims[2 + i] = WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: 1,
                base_dilation: 1,
            };
        }
        let window = Window {
            dimensions: window_dims,
        };
        let pooled = self.entry.reduce_window(x, init, &red, window, out.clone());

        if matches!(kind, ReduceOp::Mean) {
            // Divide by window size.
            let denom = kernel_size.iter().product::<usize>() as f32;
            let denom_c = self.const_in_dtype(prim_ty, denom);
            let denom_b = self.entry.broadcast(denom_c, &[], out.clone());
            self.entry.binary("divide", pooled, denom_b, out)
        } else {
            pooled
        }
    }

    // ── ScatterAdd ─────────────────────────────────────────────

    fn lower_scatter_add(&mut self, updates_id: NodeId, indices_id: NodeId, out: Shape) -> i64 {
        // Build a zero-initialized destination of shape `out`, then
        // scatter-add updates rows at indices.
        let updates = self.hlo(updates_id);
        let idx = self.hlo(indices_id);
        let idx_dt = self.dtype(indices_id);
        let idx_s32 = if matches!(idx_dt, DType::I32 | DType::I64 | DType::U32) {
            idx
        } else {
            let id_dims = self.ir_shape_dims(indices_id);
            self.entry.convert(idx, Shape::array(prim::S32, &id_dims))
        };

        let prim_ty = out.element_type;
        let zero = self.const_in_dtype(prim_ty, 0.0);
        let dest = self.entry.broadcast(zero, &[], out.clone());
        let combiner = self.reducer("add", prim_ty);

        // Indices semantics: each element of `idx_s32` selects a row
        // along axis 0 of `dest`. ScatterDimNumbers reflects that:
        //   update_window_dims = [1, 2, ..., rank-1]   (trailing dims of update)
        //   inserted_window_dims = [0]
        //   scatter_dims_to_operand_dims = [0]
        //   index_vector_dim = idx_rank
        let upd_rank = self.ir_shape_dims(updates_id).len() as i64;
        let dn = ScatterDimNumbers {
            update_window_dims: (1..upd_rank).collect(),
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: self.ir_shape_dims(indices_id).len() as i64,
        };
        self.entry
            .scatter(dest, idx_s32, updates, &combiner, dn, out)
    }

    // ── TopK ──────────────────────────────────────────────────────
    //
    // Sort (descending) along the last axis, paired with an iota of
    // indices, then slice the leading k. Indices come back as f32
    // because rlx-ir is f32 at the I/O boundary.

    fn lower_topk(&mut self, x_id: NodeId, k: usize, out: Shape) -> i64 {
        let x = self.hlo(x_id);
        let dims = self.ir_shape_dims(x_id);
        let prim_ty = prim_of(self.dtype(x_id));
        let last_axis = (dims.len() - 1) as i64;

        let iota_shape = Shape::array(prim::S32, &dims);
        let indices = self.entry.iota(last_axis, iota_shape.clone());

        // Comparator: (kx, ky, vx, vy) -> kx > ky.
        let cmp = self.builder.computation("topk_descending");
        let key_s = Shape::scalar(prim_ty);
        let val_s = Shape::scalar(prim::S32);
        let p0 = cmp.parameter(0, "kx", key_s.clone());
        let p1 = cmp.parameter(1, "ky", key_s.clone());
        let _p2 = cmp.parameter(2, "vx", val_s.clone());
        let _p3 = cmp.parameter(3, "vy", val_s.clone());
        let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
        cmp.set_root(r);
        cmp.set_program_shape(ProgramShape {
            parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
            parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
            result: Shape::scalar(prim::PRED),
        });

        let val_full = Shape::array(prim_ty, &dims);
        let idx_full = Shape::array(prim::S32, &dims);
        let tup = Shape::tuple(vec![val_full, idx_full.clone()]);
        let sorted = self.entry.sort(&[x, indices], &cmp, last_axis, true, tup);
        let sorted_idx = self.entry.get_tuple_element(sorted, 1, idx_full);

        let mut starts = vec![0i64; dims.len()];
        let mut limits = dims.clone();
        let strides = vec![1i64; dims.len()];
        starts[last_axis as usize] = 0;
        limits[last_axis as usize] = k as i64;
        let mut slice_dims = dims.clone();
        slice_dims[last_axis as usize] = k as i64;
        let sliced = self.entry.slice(
            sorted_idx,
            &starts,
            &limits,
            &strides,
            Shape::array(prim::S32, &slice_dims),
        );
        // Indices → f32 (rlx-ir convention).
        self.entry.convert(sliced, out)
    }

    // ── GroupedMatMul ────────────────────────────────────────────
    //
    // For each token `i`, output[i] = input[i] @ weight[expert_idx[i]].
    // Lowered as gather(weight, idx) to materialize per-token weights
    // [M,K,N], then a batched dot_general with batch axis = M.

    fn lower_grouped_matmul(
        &mut self,
        input_id: NodeId,
        weight_id: NodeId,
        expert_id: NodeId,
        out: Shape,
    ) -> i64 {
        let input = self.hlo(input_id);
        let weight = self.hlo(weight_id);
        let exp_idx = self.hlo(expert_id);
        let exp_dt = self.dtype(expert_id);
        let m_dims = self.ir_shape_dims(input_id); // [M, K]
        let w_dims = self.ir_shape_dims(weight_id); // [E, K, N]
        let m = m_dims[0];
        let k = m_dims[1];
        let n = w_dims[2];

        let exp_s32 = if matches!(exp_dt, DType::I32 | DType::I64 | DType::U32) {
            exp_idx
        } else {
            self.entry.convert(exp_idx, Shape::array(prim::S32, &[m]))
        };
        // Gather wants index_vector_dim, so reshape [M] → [M, 1].
        let exp_2d = self
            .entry
            .reshape(exp_s32, Shape::array(prim::S32, &[m, 1]));

        let dn = GatherDimNumbers {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
        };
        let weight_prim = prim_of(self.dtype(weight_id));
        let gathered = self.entry.gather(
            weight,
            exp_2d,
            dn,
            vec![1, k, n],
            Shape::array(weight_prim, &[m, k, n]),
        );

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![1],
            lhs_batch: vec![0],
            rhs_batch: vec![0],
        };
        self.entry.dot_general(input, gathered, dn, out)
    }

    // ── DequantMatMul ────────────────────────────────────────────
    //
    // Dequantize w_q on the fly, then dot. Per-block scale/zero-point
    // broadcast from [K/block, N] to [K, N] via reshape→broadcast→
    // reshape (the standard "tile rows" idiom in HLO).

    fn lower_dequant_matmul(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        s_id: NodeId,
        z_id: NodeId,
        scheme: QuantScheme,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w_q = self.hlo(w_id);
        let scale = self.hlo(s_id);
        let zp = self.hlo(z_id);
        let w_dims = self.ir_shape_dims(w_id); // [K, N]
        let k = w_dims[0];
        let n = w_dims[1];
        let block = match scheme {
            QuantScheme::Int8Block { block_size }
            | QuantScheme::Int8BlockAsym { block_size }
            | QuantScheme::Int4Block { block_size } => block_size as i64,
            // Fp8 schemes are per-tensor; treat as one-block-of-K.
            QuantScheme::Fp8E4m3 | QuantScheme::Fp8E5m2 => k,
            QuantScheme::GgufQ4_0 | QuantScheme::GgufQ8_0 => panic!(
                "rlx-tpu: GGUF / NVFP4 quant schemes have no HLO lowering — dequantize on CPU first."
            ),
            QuantScheme::GgufQ4K
            | QuantScheme::GgufQ5K
            | QuantScheme::GgufQ6K
            | QuantScheme::GgufQ8K
            | QuantScheme::GgufQ2K
            | QuantScheme::GgufQ3K
            | QuantScheme::Nvfp4Block => panic!(
                "rlx-tpu: GGUF / NVFP4 quant schemes have no HLO lowering — dequantize on CPU first."
            ),
        };
        let kb = (k + block - 1) / block;

        let kn_f32 = Shape::array(prim::F32, &[k, n]);
        let w_f = self.entry.convert(w_q, kn_f32.clone());

        // Helper: broadcast a [K/block, N] tile to [K, N] by tiling
        // each row `block` times. HLO's `broadcast` requires the
        // operand's dim sizes to match the target's at the dims named
        // by broadcast_dims — it does NOT auto-expand size-1 dims.
        // So go [kb, n] → [kb, block, n] (broadcast_dims = [0, 2],
        // adds a fresh size-`block` axis at dim 1) → [k, n] (reshape).
        let tile_block = |this: &Self, t: i64, t_dt: i32| -> i64 {
            let t_b = this
                .entry
                .broadcast(t, &[0, 2], Shape::array(t_dt, &[kb, block, n]));
            this.entry.reshape(t_b, Shape::array(t_dt, &[k, n]))
        };
        let scale_kn = tile_block(self, scale, prim::F32);
        let zp_kn = tile_block(self, zp, prim::F32);

        let centered = self.entry.binary("subtract", w_f, zp_kn, kn_f32.clone());
        let w_dq = self.entry.binary("multiply", centered, scale_kn, kn_f32);

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        self.entry.dot_general(x, w_dq, dn, out)
    }

    // ── QMatMul ───────────────────────────────────────────────────
    //
    // Real INT8 matmul: promote x, w to S32, subtract zero points,
    // dot, add bias, scale by `mult` in F32, round, +out_zp, clamp
    // to [-128, 127], convert back to S8.

    fn lower_qmatmul(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        b_id: NodeId,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let bias = self.hlo(b_id);
        let x_dims = self.ir_shape_dims(x_id);
        let w_dims = self.ir_shape_dims(w_id);
        let m = x_dims[0];
        let k = x_dims[1];
        let n = w_dims[1];
        let mn_s32 = Shape::array(prim::S32, &[m, n]);
        let mn_f32 = Shape::array(prim::F32, &[m, n]);

        let x_s32 = self.entry.convert(x, Shape::array(prim::S32, &[m, k]));
        let w_s32 = self.entry.convert(w, Shape::array(prim::S32, &[k, n]));

        let xzp_c = self.entry.constant_s32_scalar(x_zp);
        let xzp_b = self
            .entry
            .broadcast(xzp_c, &[], Shape::array(prim::S32, &[m, k]));
        let x_centered =
            self.entry
                .binary("subtract", x_s32, xzp_b, Shape::array(prim::S32, &[m, k]));

        let wzp_c = self.entry.constant_s32_scalar(w_zp);
        let wzp_b = self
            .entry
            .broadcast(wzp_c, &[], Shape::array(prim::S32, &[k, n]));
        let w_centered =
            self.entry
                .binary("subtract", w_s32, wzp_b, Shape::array(prim::S32, &[k, n]));

        let dn = DotDimNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        };
        let acc = self
            .entry
            .dot_general(x_centered, w_centered, dn, mn_s32.clone());
        let bias_b = self.entry.broadcast(bias, &[1], mn_s32.clone());
        let with_bias = self.entry.binary("add", acc, bias_b, mn_s32.clone());

        let acc_f32 = self.entry.convert(with_bias, mn_f32.clone());
        let m_c = self.entry.constant_f32_scalar(mult);
        let m_b = self.entry.broadcast(m_c, &[], mn_f32.clone());
        let scaled = self.entry.binary("multiply", acc_f32, m_b, mn_f32.clone());
        let rounded = self
            .entry
            .unary("round-nearest-even", scaled, mn_f32.clone());
        let oz_c = self.entry.constant_f32_scalar(out_zp as f32);
        let oz_b = self.entry.broadcast(oz_c, &[], mn_f32.clone());
        let with_oz = self.entry.binary("add", rounded, oz_b, mn_f32.clone());

        let lo_c = self.entry.constant_f32_scalar(-128.0);
        let hi_c = self.entry.constant_f32_scalar(127.0);
        let lo_b = self.entry.broadcast(lo_c, &[], mn_f32.clone());
        let hi_b = self.entry.broadcast(hi_c, &[], mn_f32.clone());
        let cl_lo = self.entry.binary("maximum", with_oz, lo_b, mn_f32.clone());
        let cl = self.entry.binary("minimum", cl_lo, hi_b, mn_f32);

        self.entry.convert(cl, out)
    }

    // ── QConv2d ──────────────────────────────────────────────────
    //
    // Same arithmetic shape as QMatMul, but wrapped around a 2-D
    // convolution. Inputs are NCHW int8; bias is per-output-channel
    // s32 in accumulator scale.

    fn lower_qconv2d(
        &mut self,
        x_id: NodeId,
        w_id: NodeId,
        b_id: NodeId,
        kernel_size: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult: f32,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let w = self.hlo(w_id);
        let bias = self.hlo(b_id);
        let x_dims = self.ir_shape_dims(x_id);
        let w_dims = self.ir_shape_dims(w_id);
        let out_dims = out.dimensions.clone();

        let x_s32_shape = Shape::array(prim::S32, &x_dims);
        let w_s32_shape = Shape::array(prim::S32, &w_dims);
        let out_s32 = Shape::array(prim::S32, &out_dims);
        let out_f32 = Shape::array(prim::F32, &out_dims);

        let x_s32 = self.entry.convert(x, x_s32_shape.clone());
        let w_s32 = self.entry.convert(w, w_s32_shape.clone());

        let xzp_c = self.entry.constant_s32_scalar(x_zp);
        let xzp_b = self.entry.broadcast(xzp_c, &[], x_s32_shape.clone());
        let x_centered = self.entry.binary("subtract", x_s32, xzp_b, x_s32_shape);
        let wzp_c = self.entry.constant_s32_scalar(w_zp);
        let wzp_b = self.entry.broadcast(wzp_c, &[], w_s32_shape.clone());
        let w_centered = self.entry.binary("subtract", w_s32, wzp_b, w_s32_shape);

        let n_spatial = x_dims.len() - 2;
        let cdn = ConvDimNumbers {
            input_batch_dim: 0,
            input_feature_dim: 1,
            input_spatial_dims: (2..2 + n_spatial as i64).collect(),
            kernel_output_feature_dim: 0,
            kernel_input_feature_dim: 1,
            kernel_spatial_dims: (2..2 + n_spatial as i64).collect(),
            output_batch_dim: 0,
            output_feature_dim: 1,
            output_spatial_dims: (2..2 + n_spatial as i64).collect(),
        };
        let mut window_dims = Vec::with_capacity(n_spatial);
        for i in 0..n_spatial {
            window_dims.push(WindowDim {
                size: kernel_size[i] as i64,
                stride: stride[i] as i64,
                padding_low: padding[i] as i64,
                padding_high: padding[i] as i64,
                window_dilation: dilation[i] as i64,
                base_dilation: 1,
            });
        }
        let window = Window {
            dimensions: window_dims,
        };
        let acc = self.entry.convolution(
            x_centered,
            w_centered,
            window,
            cdn,
            groups as i64,
            out_s32.clone(),
        );

        // Broadcast bias [C_out] across batch + spatial (axis 1 in NCHW).
        let bias_b = self.entry.broadcast(bias, &[1], out_s32.clone());
        let with_bias = self.entry.binary("add", acc, bias_b, out_s32);

        let acc_f32 = self.entry.convert(with_bias, out_f32.clone());
        let m_c = self.entry.constant_f32_scalar(mult);
        let m_b = self.entry.broadcast(m_c, &[], out_f32.clone());
        let scaled = self.entry.binary("multiply", acc_f32, m_b, out_f32.clone());
        let rounded = self
            .entry
            .unary("round-nearest-even", scaled, out_f32.clone());
        let oz_c = self.entry.constant_f32_scalar(out_zp as f32);
        let oz_b = self.entry.broadcast(oz_c, &[], out_f32.clone());
        let with_oz = self.entry.binary("add", rounded, oz_b, out_f32.clone());

        let lo_c = self.entry.constant_f32_scalar(-128.0);
        let hi_c = self.entry.constant_f32_scalar(127.0);
        let lo_b = self.entry.broadcast(lo_c, &[], out_f32.clone());
        let hi_b = self.entry.broadcast(hi_c, &[], out_f32.clone());
        let cl_lo = self.entry.binary("maximum", with_oz, lo_b, out_f32.clone());
        let cl = self.entry.binary("minimum", cl_lo, hi_b, out_f32);
        self.entry.convert(cl, out)
    }

    // ── Sample ────────────────────────────────────────────────────
    //
    // Logits [B, V] f32 → token_ids [B] f32.
    //
    // Decomposition:
    //   * temperature == 0 → argmax via topk(k=1)
    //   * top_k > 0 → filter logits below the k-th largest to -inf
    //   * top_p < 1.0 → filter via threshold = sorted_logits at the
    //     boundary index (first k where cumsum(softmax) ≥ top_p);
    //     no scatter-back needed because the kept set is exactly
    //     the largest-N logits, expressible as a value threshold
    //   * temperature  > 0 → multinomial via inverse-CDF on a
    //     uniform random [B] sample.
    //
    // RNG: XLA's `rng` op with UNIFORM distribution. Bit-exact match
    // to CUDA's Philox state would require lowering the same
    // counter-encoded seed, which the framework doesn't expose
    // through `rng-bit-generator` in a portable way, so we
    // deliberately don't aim for bit parity here — only that the
    // distribution is correct.

    fn lower_sample(
        &mut self,
        logits_id: NodeId,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
        out: Shape,
    ) -> i64 {
        let _ = seed;
        let logits = self.hlo(logits_id);
        let dims = self.ir_shape_dims(logits_id);
        assert_eq!(dims.len(), 2, "Op::Sample expects [B, V] logits");
        let b = dims[0];
        let v = dims[1];
        let bv_f32 = Shape::array(prim::F32, &[b, v]);
        let b_s32 = Shape::array(prim::S32, &[b]);

        if temperature == 0.0 {
            // Greedy: argmax via topk(k=1) on the value axis, then
            // squeeze.
            let topk_shape = Shape::array(prim::F32, &[b, 1]);
            let topk_idx_f32 =
                self.lower_topk_inner(logits, &dims, prim::F32, 1, topk_shape.clone());
            let squeezed = self
                .entry
                .reshape(topk_idx_f32, Shape::array(prim::F32, &[b]));
            return if out.element_type == prim::F32 {
                squeezed
            } else {
                self.entry.convert(squeezed, out)
            };
        }

        // Scale by 1/temperature.
        let inv_t = self.entry.constant_f32_scalar(1.0 / temperature);
        let inv_t_b = self.entry.broadcast(inv_t, &[], bv_f32.clone());
        let mut logits = self
            .entry
            .binary("multiply", logits, inv_t_b, bv_f32.clone());

        // Optional top-k filter: zero out values below the k-th
        // largest by replacing them with -inf.
        if top_k > 0 && (top_k as i64) < v {
            let k_i = top_k as i64;
            // Sort descending paired with iota indices.
            let cmp = self.builder.computation("topk_cmp_for_sample");
            let key_s = Shape::scalar(prim::F32);
            let val_s = Shape::scalar(prim::S32);
            let p0 = cmp.parameter(0, "kx", key_s.clone());
            let p1 = cmp.parameter(1, "ky", key_s.clone());
            let _ = cmp.parameter(2, "vx", val_s.clone());
            let _ = cmp.parameter(3, "vy", val_s.clone());
            let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
            cmp.set_root(r);
            cmp.set_program_shape(ProgramShape {
                parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
                parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
                result: Shape::scalar(prim::PRED),
            });
            let idx = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
            let tup = Shape::tuple(vec![bv_f32.clone(), Shape::array(prim::S32, &[b, v])]);
            let sorted = self.entry.sort(&[logits, idx], &cmp, 1, true, tup);
            let sorted_vals = self.entry.get_tuple_element(sorted, 0, bv_f32.clone());
            // Threshold = sorted_vals[..., k-1]
            let starts = vec![0, k_i - 1];
            let limits = vec![b, k_i];
            let strides = vec![1, 1];
            let kth = self.entry.slice(
                sorted_vals,
                &starts,
                &limits,
                &strides,
                Shape::array(prim::F32, &[b, 1]),
            );
            let kth_b = self.entry.broadcast(
                self.entry.reshape(kth, Shape::array(prim::F32, &[b])),
                &[0],
                bv_f32.clone(),
            );
            let mask = self
                .entry
                .compare(logits, kth_b, "LT", Shape::array(prim::PRED, &[b, v]));
            let neg_inf = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b = self.entry.broadcast(neg_inf, &[], bv_f32.clone());
            logits = self.entry.select(mask, neg_inf_b, logits, bv_f32.clone());
        }

        // Optional top-p (nucleus) filter. Idea: the kept set is the
        // smallest contiguous prefix of the sorted-descending logits
        // whose softmaxed cumulative probability mass first reaches
        // `top_p`. Because the kept set is exactly the largest-N
        // logits, we can express the filter as a value threshold —
        // no "scatter back to original order" needed. The threshold
        // is the value of the boundary token (smallest kept).
        if top_p < 1.0 - 1e-7 {
            // Sort logits descending (we don't need indices here,
            // but the comparator API takes paired k/v). Use iota for
            // the unused value half — XLA's sort needs a comparator
            // and we already have the topk_cmp_for_sample shape.
            let cmp = self.builder.computation("topp_cmp");
            let key_s = Shape::scalar(prim::F32);
            let val_s = Shape::scalar(prim::S32);
            let p0 = cmp.parameter(0, "kx", key_s.clone());
            let p1 = cmp.parameter(1, "ky", key_s.clone());
            let _ = cmp.parameter(2, "vx", val_s.clone());
            let _ = cmp.parameter(3, "vy", val_s.clone());
            let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
            cmp.set_root(r);
            cmp.set_program_shape(ProgramShape {
                parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
                parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
                result: Shape::scalar(prim::PRED),
            });
            let idx = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
            let tup = Shape::tuple(vec![bv_f32.clone(), Shape::array(prim::S32, &[b, v])]);
            let sorted = self.entry.sort(&[logits, idx], &cmp, 1, true, tup);
            let sorted_vals = self.entry.get_tuple_element(sorted, 0, bv_f32.clone());

            // softmax of sorted vals → cumsum along last axis.
            let s_probs = self.lower_softmax_id(sorted_vals, bv_f32.clone(), 1);
            let s_cum = self.scan_along_last_axis(s_probs, &[b, v], prim::F32, "add", 0.0);

            // Find the boundary — smallest k such that cum[b, k] >= p.
            // First-true via cumsum-of-bool == 1.
            let p_const = self.entry.constant_f32_scalar(top_p);
            let p_b = self.entry.broadcast(p_const, &[], bv_f32.clone());
            let above = self
                .entry
                .compare(s_cum, p_b, "GE", Shape::array(prim::PRED, &[b, v]));
            let above_s32 = self.entry.convert(above, Shape::array(prim::S32, &[b, v]));
            let above_cumcount =
                self.scan_along_last_axis(above_s32, &[b, v], prim::S32, "add", 0.0);
            let one_s32 = self.entry.constant_s32_scalar(1);
            let one_b32 = self
                .entry
                .broadcast(one_s32, &[], Shape::array(prim::S32, &[b, v]));
            let first_geq = self.entry.compare(
                above_cumcount,
                one_b32,
                "EQ",
                Shape::array(prim::PRED, &[b, v]),
            );

            // Threshold[b] = sorted_vals[b, first_geq_idx]. We pull
            // it out by select(first_geq, sorted_vals, -inf) followed
            // by reduce-max along axis 1.
            let neg_inf = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b = self.entry.broadcast(neg_inf, &[], bv_f32.clone());
            let masked_for_thresh =
                self.entry
                    .select(first_geq, sorted_vals, neg_inf_b, bv_f32.clone());
            let red = self.reducer("maximum", prim::F32);
            let init_neg = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let init_neg_s = self.entry.convert(init_neg, Shape::scalar(prim::F32));
            let threshold = self.entry.reduce(
                masked_for_thresh,
                init_neg_s,
                &red,
                &[1],
                Shape::array(prim::F32, &[b]),
            );

            // Apply the threshold to the original logits: anything
            // strictly below the boundary value is replaced with
            // -inf. Ties at the boundary are kept (matches HF /
            // Llama-style "include the first overshooting token").
            let thresh_b = self.entry.broadcast(threshold, &[0], bv_f32.clone());
            let keep =
                self.entry
                    .compare(logits, thresh_b, "GE", Shape::array(prim::PRED, &[b, v]));
            let neg_inf2 = self.entry.constant_f32_scalar(f32::NEG_INFINITY);
            let neg_inf_b2 = self.entry.broadcast(neg_inf2, &[], bv_f32.clone());
            logits = self.entry.select(keep, logits, neg_inf_b2, bv_f32.clone());
        }

        // softmax → probs → cumsum (cdf).
        let probs = self.lower_softmax_id(logits, bv_f32.clone(), 1);
        let cdf = self.scan_along_last_axis(probs, &[b, v], prim::F32, "add", 0.0);

        // Uniform random [B] in [0, 1).
        let zero = self.entry.constant_f32_scalar(0.0);
        let one = self.entry.constant_f32_scalar(1.0);
        let u = self.entry.rng(
            zero,
            one,
            /*UNIFORM=*/ 1,
            Shape::array(prim::F32, &[b]),
        );
        let u_b = self.entry.broadcast(u, &[0], bv_f32.clone());

        // Find the first column where cdf >= u.
        let ge = self
            .entry
            .compare(cdf, u_b, "GE", Shape::array(prim::PRED, &[b, v]));
        let ge_s32 = self.entry.convert(ge, Shape::array(prim::S32, &[b, v]));
        let cumcount = self.scan_along_last_axis(ge_s32, &[b, v], prim::S32, "add", 0.0);
        let one_s32 = self.entry.constant_s32_scalar(1);
        let one_b = self
            .entry
            .broadcast(one_s32, &[], Shape::array(prim::S32, &[b, v]));
        let first_eq = self
            .entry
            .compare(cumcount, one_b, "EQ", Shape::array(prim::PRED, &[b, v]));
        let idx_iota = self.entry.iota(1, Shape::array(prim::S32, &[b, v]));
        let zero_s32 = self.entry.constant_s32_scalar(0);
        let zero_s32_b = self
            .entry
            .broadcast(zero_s32, &[], Shape::array(prim::S32, &[b, v]));
        let masked = self.entry.select(
            first_eq,
            idx_iota,
            zero_s32_b,
            Shape::array(prim::S32, &[b, v]),
        );
        // reduce-max along axis 1 → [B] s32.
        let red = self.reducer("maximum", prim::S32);
        let init = self.entry.constant_s32_scalar(0);
        let token_s32 = self.entry.reduce(masked, init, &red, &[1], b_s32);
        // → f32.
        if out.element_type == prim::F32 {
            self.entry.convert(token_s32, out)
        } else {
            let f = self.entry.convert(token_s32, Shape::array(prim::F32, &[b]));
            if out.element_type == prim::F32 {
                f
            } else {
                self.entry.convert(f, out)
            }
        }
    }

    /// argmax via topk-1 on `x` of shape `dims` (last axis is the
    /// reduction). Returns f32 indices reshaped to `out_shape`.
    fn lower_topk_inner(
        &mut self,
        x: i64,
        dims: &[i64],
        prim_ty: i32,
        k: usize,
        out_shape: Shape,
    ) -> i64 {
        let last_axis = (dims.len() - 1) as i64;
        let cmp = self.builder.computation("topk_inner_descending");
        let key_s = Shape::scalar(prim_ty);
        let val_s = Shape::scalar(prim::S32);
        let p0 = cmp.parameter(0, "kx", key_s.clone());
        let p1 = cmp.parameter(1, "ky", key_s.clone());
        let _ = cmp.parameter(2, "vx", val_s.clone());
        let _ = cmp.parameter(3, "vy", val_s.clone());
        let r = cmp.compare(p0, p1, "GT", Shape::scalar(prim::PRED));
        cmp.set_root(r);
        cmp.set_program_shape(ProgramShape {
            parameters: vec![key_s.clone(), key_s.clone(), val_s.clone(), val_s.clone()],
            parameter_names: vec!["kx".into(), "ky".into(), "vx".into(), "vy".into()],
            result: Shape::scalar(prim::PRED),
        });
        let idx = self.entry.iota(last_axis, Shape::array(prim::S32, dims));
        let tup = Shape::tuple(vec![
            Shape::array(prim_ty, dims),
            Shape::array(prim::S32, dims),
        ]);
        let sorted = self.entry.sort(&[x, idx], &cmp, last_axis, true, tup);
        let sorted_idx = self
            .entry
            .get_tuple_element(sorted, 1, Shape::array(prim::S32, dims));
        let mut starts = vec![0i64; dims.len()];
        let mut limits = dims.to_vec();
        let strides = vec![1i64; dims.len()];
        starts[last_axis as usize] = 0;
        limits[last_axis as usize] = k as i64;
        let mut slice_dims = dims.to_vec();
        slice_dims[last_axis as usize] = k as i64;
        let sliced = self.entry.slice(
            sorted_idx,
            &starts,
            &limits,
            &strides,
            Shape::array(prim::S32, &slice_dims),
        );
        self.entry.convert(sliced, out_shape)
    }

    /// Inclusive scan with a reducer along the last axis. Mirrors
    /// `lower_cumsum` but parametric on opcode and dtype, used by
    /// `Sample` for both probs cumsum and bool→count cumsum.
    fn scan_along_last_axis(
        &mut self,
        x: i64,
        dims: &[i64],
        prim_ty: i32,
        opcode: &str,
        init_v: f32,
    ) -> i64 {
        let ax = (dims.len() - 1) as i64;
        let init = self.const_in_dtype(prim_ty, init_v);
        let red = self.reducer(opcode, prim_ty);
        let mut window_dims = vec![
            WindowDim {
                size: 1,
                stride: 1,
                padding_low: 0,
                padding_high: 0,
                window_dilation: 1,
                base_dilation: 1,
            };
            dims.len()
        ];
        window_dims[ax as usize] = WindowDim {
            size: dims[ax as usize],
            stride: 1,
            padding_low: dims[ax as usize] - 1,
            padding_high: 0,
            window_dilation: 1,
            base_dilation: 1,
        };
        let window = Window {
            dimensions: window_dims,
        };
        self.entry
            .reduce_window(x, init, &red, window, Shape::array(prim_ty, dims))
    }

    // ── SelectiveScan ────────────────────────────────────────────
    //
    // The Mamba/SSM state-space scan, lowered to an HLO `while` loop.
    // Inputs:  x [B,L,D], delta [B,L,D], a [D,N], b [B,L,N], c [B,L,N]
    // Output:  [B, L, D]
    //
    // Per timestep t (B elided for clarity):
    //   decay  = exp(delta[t,:,None] * a)        [D, N]
    //   update = delta[t,:,None] * b[t,None,:] * x[t,:,None]   [D, N]
    //   state  = state * decay + update          [D, N]
    //   y[t]   = sum_n state[d,n] * c[t,n]       [D]
    //
    // Loop carry tuple: (i_s32, state[B,D,N], outputs[B,L,D])

    fn lower_selective_scan(
        &mut self,
        x_id: NodeId,
        delta_id: NodeId,
        a_id: NodeId,
        b_id: NodeId,
        c_id: NodeId,
        state_size: usize,
        out: Shape,
    ) -> i64 {
        let x = self.hlo(x_id);
        let delta = self.hlo(delta_id);
        let a = self.hlo(a_id);
        let bb = self.hlo(b_id);
        let cc = self.hlo(c_id);
        let x_dims = self.ir_shape_dims(x_id); // [B, L, D]
        let b = x_dims[0];
        let l = x_dims[1];
        let d = x_dims[2];
        let n = state_size as i64;

        let bd = Shape::array(prim::F32, &[b, d]);
        let bn = Shape::array(prim::F32, &[b, n]);
        let bdn = Shape::array(prim::F32, &[b, d, n]);
        let bld = Shape::array(prim::F32, &[b, l, d]);
        let s32_scalar = Shape::scalar(prim::S32);

        // Carry tuple (extended): (i, state, outs, x, delta, a, b, c).
        // HLO `while` only takes the carry as parameter, so the
        // per-step inputs are threaded through it.
        let bld_t = bld.clone();
        let dn_a = Shape::array(prim::F32, &[d, n]);
        let bln = Shape::array(prim::F32, &[b, l, n]);
        let big_tup = Shape::tuple(vec![
            s32_scalar.clone(),
            bdn.clone(),
            bld_t.clone(),
            bld_t.clone(),
            bld_t.clone(),
            dn_a.clone(),
            bln.clone(),
            bln.clone(),
        ]);

        // Initial values, packed into the carry tuple.
        let i0 = self.entry.constant_s32_scalar(0);
        let zero_f = self.entry.constant_f32_scalar(0.0);
        let state0 = self.entry.broadcast(zero_f, &[], bdn.clone());
        let outs0 = self.entry.broadcast(zero_f, &[], bld.clone());
        let big_init = self
            .entry
            .tuple(&[i0, state0, outs0, x, delta, a, bb, cc], big_tup.clone());

        // Reducer for the body's per-step axis-2 sum. Create it BEFORE
        // the body so it lands earlier in the computation list — XLA's
        // proto deserializer rejects forward references.
        let red = self
            .builder
            .make_reducer(&format!("scan_red_{}", state_size), "add", prim::F32);

        // Cond: i < L.
        let cond2 = self.builder.computation("scan_cond_big");
        let p = cond2.parameter(0, "carry", big_tup.clone());
        let ci2 = cond2.get_tuple_element(p, 0, s32_scalar.clone());
        let l_c = cond2.constant_s32_scalar(l as i32);
        let pr = cond2.compare(ci2, l_c, "LT", Shape::scalar(prim::PRED));
        cond2.set_root(pr);
        cond2.set_program_shape(ProgramShape {
            parameters: vec![big_tup.clone()],
            parameter_names: vec!["carry".into()],
            result: Shape::scalar(prim::PRED),
        });

        // Body.
        let body = self.builder.computation("scan_body");
        let bp = body.parameter(0, "carry", big_tup.clone());
        let bi = body.get_tuple_element(bp, 0, s32_scalar.clone());
        let bstate = body.get_tuple_element(bp, 1, bdn.clone());
        let bouts = body.get_tuple_element(bp, 2, bld.clone());
        let bx = body.get_tuple_element(bp, 3, bld.clone());
        let bdelta = body.get_tuple_element(bp, 4, bld.clone());
        let ba = body.get_tuple_element(bp, 5, dn_a.clone());
        let bb_t = body.get_tuple_element(bp, 6, bln.clone());
        let bc_t = body.get_tuple_element(bp, 7, bln.clone());

        let zero_idx = body.constant_s32_scalar(0);

        // Slice x/delta at step i: dynamic-slice [B, 1, D], reshape [B, D].
        let x_slc = body.dynamic_slice(
            bx,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, d],
            Shape::array(prim::F32, &[b, 1, d]),
        );
        let x_t = body.reshape(x_slc, bd.clone());
        let d_slc = body.dynamic_slice(
            bdelta,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, d],
            Shape::array(prim::F32, &[b, 1, d]),
        );
        let delta_t = body.reshape(d_slc, bd.clone());
        // Slice b/c at step i: dynamic-slice [B, 1, N], reshape [B, N].
        let b_slc = body.dynamic_slice(
            bb_t,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, n],
            Shape::array(prim::F32, &[b, 1, n]),
        );
        let b_step = body.reshape(b_slc, bn.clone());
        let c_slc = body.dynamic_slice(
            bc_t,
            &[zero_idx, bi, zero_idx],
            vec![b, 1, n],
            Shape::array(prim::F32, &[b, 1, n]),
        );
        let c_step = body.reshape(c_slc, bn.clone());

        // decay = exp(delta_t[..., None] * a[None, ..., :])  [B, D, N]
        let delta_3 = body.broadcast(delta_t, &[0, 1], bdn.clone());
        let a_3 = body.broadcast(ba, &[1, 2], bdn.clone());
        let prod_da = body.binary("multiply", delta_3, a_3, bdn.clone());
        let decay = body.unary("exponential", prod_da, bdn.clone());

        // update = delta_t[...,None] * b_step[:,None,:] * x_t[...,None]
        let b_3 = body.broadcast(b_step, &[0, 2], bdn.clone());
        let x_3 = body.broadcast(x_t, &[0, 1], bdn.clone());
        let db = body.binary("multiply", delta_3, b_3, bdn.clone());
        let update = body.binary("multiply", db, x_3, bdn.clone());

        let state_decayed = body.binary("multiply", bstate, decay, bdn.clone());
        let new_state = body.binary("add", state_decayed, update, bdn.clone());

        // y[t] = sum_n new_state[b,d,n] * c_step[b,n]  → [B, D]
        let c_3 = body.broadcast(c_step, &[0, 2], bdn.clone());
        let prod_sc = body.binary("multiply", new_state, c_3, bdn.clone());
        // reduce sum over axis 2 (reducer was hoisted above body).
        let init_v = body.constant_f32_scalar(0.0);
        let y_t = body.reduce(prod_sc, init_v, &red, &[2], bd.clone());
        // Reshape y_t [B, D] → [B, 1, D] to fit dynamic-update-slice.
        let y_t_3 = body.reshape(y_t, Shape::array(prim::F32, &[b, 1, d]));
        let new_outs =
            body.dynamic_update_slice(bouts, y_t_3, &[zero_idx, bi, zero_idx], bld.clone());

        // i' = i + 1.
        let one_c = body.constant_s32_scalar(1);
        let bi1 = body.binary("add", bi, one_c, s32_scalar.clone());

        // Re-pack the body's output tuple in the same shape as the
        // input carry.
        let new_tup = body.tuple(
            &[bi1, new_state, new_outs, bx, bdelta, ba, bb_t, bc_t],
            big_tup.clone(),
        );
        body.set_root(new_tup);
        body.set_program_shape(ProgramShape {
            parameters: vec![big_tup.clone()],
            parameter_names: vec!["carry".into()],
            result: big_tup.clone(),
        });

        // While.
        let final_tup = self.entry.while_loop(big_init, &cond2, &body, big_tup);
        // Extract outputs (slot 2) — that's the result.
        let outs = self.entry.get_tuple_element(final_tup, 2, bld);
        if out.element_type == prim::F32 {
            outs
        } else {
            self.entry.convert(outs, out)
        }
    }
}
