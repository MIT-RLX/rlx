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
//! ElementwiseRegion, TransformRegion, and BatchElementwiseRegion are
//! lowered inline as primitive HLO (chain walk / per-slice concat).
//! custom_call. Keeps the emitted module portable across PJRT
//! plugins (TPU, CPU, GPU). FusedSwiGLU / FusedAttentionBlock /
//! FusedTransformerLayer / LoraMatMul / If / While are normalized
//! through `crate::unfuse` before lowering.
//!
//! Ops that have no clean HLO decomposition without large blowup
//! (Sample, TopK, SelectiveScan) panic with a clear message.
//!
//! **GGUF `DequantMatMul`:** host-dequant at emit time → f32 constant →
//! `dot_general` ([`lower_dequant_matmul_gguf`]). See
//! [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md) (TPU section).

use std::collections::HashMap;

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::hlo::{Computation, HloBuilder, ProgramShape, Shape, Window, WindowDim, prim_of};

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
    /// GGUF `DequantMatMul` weights supplied at runtime via `set_param_typed(U8)`.
    pub gguf_deferred: HashMap<String, GgufDeferredParam>,
}

/// Runtime GGUF param: host-dequant to f32 before PJRT upload.
#[derive(Clone, Debug)]
pub struct GgufDeferredParam {
    pub scheme: QuantScheme,
    pub k: i64,
    pub n: i64,
}

pub fn lower_graph(graph: &Graph) -> HloModule {
    lower_graph_with_rng(graph, rlx_ir::RngOptions::default())
}

/// Compile-time GGUF weight bytes keyed by `Op::Param` name.
///
/// When lowering `DequantMatMul` with a `Param` weight, TPU embeds host-dequantized
/// f32 as HLO constants. Pass packed bytes here (mirrors CoreML `quant_bytes`).
pub type LowerParamBytes = HashMap<String, Vec<u8>>;

pub fn lower_graph_with_rng(graph: &Graph, rng: rlx_ir::RngOptions) -> HloModule {
    lower_graph_with_rng_and_params(graph, rng, None)
}

pub fn lower_graph_with_rng_and_params(
    graph: &Graph,
    rng: rlx_ir::RngOptions,
    param_bytes: Option<&LowerParamBytes>,
) -> HloModule {
    let mut b = HloBuilder::new(&graph.name);

    // Reducer subcomputations cached by (opcode, prim_ty) so multiple
    // Reduce / ReduceWindow ops share the same body.
    let mut reducers: HashMap<(String, i32), Computation> = HashMap::new();

    let entry = b.computation("entry");
    let mut id_map: HashMap<NodeId, i64> = HashMap::new();

    let (inputs, params, others) = partition_nodes(graph);
    let deferred = collect_gguf_deferred_params(graph, param_bytes);

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
        let (dims, dtype) = if let Some(def) = deferred.get(&name) {
            (vec![def.k, def.n], DType::F32)
        } else {
            (ir_dims(&n.shape), n.shape.dtype())
        };
        let shape = Shape::array(prim_of(dtype), &dims);
        let id = entry.parameter(next_param_base + i as i64, &name, shape.clone());
        id_map.insert(nid, id);
        param_names.push(name.clone());
        param_dtypes.push(dtype);
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
        rng,
        param_bytes,
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
        gguf_deferred: deferred,
    }
}

fn collect_gguf_deferred_params(
    graph: &Graph,
    param_bytes: Option<&LowerParamBytes>,
) -> HashMap<String, GgufDeferredParam> {
    use rlx_ir::Dim;
    let mut out = HashMap::new();
    for node in graph.nodes() {
        let Op::DequantMatMul { scheme } = &node.op else {
            continue;
        };
        if !scheme.is_gguf() {
            continue;
        }
        let w_id = node.inputs[1];
        let Op::Param { name } = &graph.node(w_id).op else {
            continue;
        };
        if param_bytes.and_then(|m| m.get(name)).is_some() {
            continue;
        }
        let x_dims = graph.node(node.inputs[0]).shape.dims();
        let y_dims = node.shape.dims();
        let k = match x_dims.last() {
            Some(Dim::Static(k)) => *k as i64,
            _ => panic!("rlx-tpu: GGUF deferred param requires static k on x input"),
        };
        let n = match y_dims.last() {
            Some(Dim::Static(n)) => *n as i64,
            _ => panic!("rlx-tpu: GGUF deferred param requires static n on output"),
        };
        out.insert(
            name.clone(),
            GgufDeferredParam {
                scheme: *scheme,
                k,
                n,
            },
        );
    }
    out
}

/// Convert a runtime U8 GGUF upload into f32 bytes for PJRT when the HLO module
/// was compiled with a deferred GGUF weight parameter.
pub fn gguf_param_bytes_from_u8(
    deferred: &HashMap<String, GgufDeferredParam>,
    name: &str,
    data: &[u8],
    dtype: DType,
) -> Option<(Vec<u8>, DType)> {
    if dtype != DType::U8 {
        return None;
    }
    let def = deferred.get(name)?;
    let n_elems = (def.k * def.n) as usize;
    let w_f32 = dequant_gguf_bytes(def.scheme, data, n_elems)
        .unwrap_or_else(|e| panic!("rlx-tpu: GGUF runtime dequant for '{name}': {e}"));
    let mut bytes = Vec::with_capacity(w_f32.len() * 4);
    for v in &w_f32 {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Some((bytes, DType::F32))
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
    rng: rlx_ir::RngOptions,
    param_bytes: Option<&'a LowerParamBytes>,
}

mod apply;
mod broadcast;
mod const_ops;
mod gguf;
mod ir;
#[allow(clippy::module_inception)]
mod lower;

impl<'a> LowerCtx<'a> {
    /// HLO id for an already-lowered IR node.
    pub(crate) fn hlo(&self, nid: NodeId) -> i64 {
        *self
            .id_map
            .get(&nid)
            .unwrap_or_else(|| panic!("rlx-tpu: node {nid:?} referenced before lowering"))
    }

    pub(crate) fn dtype(&self, nid: NodeId) -> DType {
        self.graph.node(nid).shape.dtype()
    }

    /// Get-or-create a binary-op reducer subcomputation.
    pub(crate) fn reducer(&mut self, opcode: &str, prim_ty: i32) -> Computation {
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

    /// Reduce over a single axis with a known reducer opcode.
    pub(crate) fn reduce_one(
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

    pub(crate) fn resize_nearest_2x_shape(&self, input_id: NodeId) -> Shape {
        let dims = self.ir_shape_dims(input_id);
        assert_eq!(
            dims.len(),
            4,
            "rlx-tpu resize_nearest_2x: expected NCHW rank 4, got rank {}",
            dims.len()
        );
        Shape::array(
            prim_of(self.dtype(input_id)),
            &[dims[0], dims[1], dims[2] * 2, dims[3] * 2],
        )
    }

    /// Inclusive scan with a reducer along the last axis. Mirrors
    /// `lower_cumsum` but parametric on opcode and dtype, used by
    /// `Sample` for both probs cumsum and bool→count cumsum.
    pub(crate) fn scan_along_last_axis(
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
}

/// Host-side GGUF dequant dispatch for TPU lowering (all `Gguf*` schemes).
///
/// Used by [`LowerCtx::lower_dequant_matmul_gguf`] to bake f32 weights into HLO
/// constants. Not invoked at PJRT runtime.
fn dequant_gguf_bytes(scheme: QuantScheme, bytes: &[u8], n: usize) -> Result<Vec<f32>, String> {
    use QuantScheme::*;
    let r = match scheme {
        GgufQ8_0 => rlx_gguf::dequant_q8_0(bytes, n),
        GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(bytes, n),
        GgufQ4_0 => rlx_gguf::dequant_q4_0(bytes, n),
        GgufQ4_1 => rlx_gguf::dequant_q4_1(bytes, n),
        GgufQ5_0 => rlx_gguf::dequant_q5_0(bytes, n),
        GgufQ5_1 => rlx_gguf::dequant_q5_1(bytes, n),
        GgufQ2K => rlx_gguf::dequant_q2_k(bytes, n),
        GgufQ3K => rlx_gguf::dequant_q3_k(bytes, n),
        GgufQ4K => rlx_gguf::dequant_q4_k(bytes, n),
        GgufQ5K => rlx_gguf::dequant_q5_k(bytes, n),
        GgufQ6K => rlx_gguf::dequant_q6_k(bytes, n),
        GgufQ8K => rlx_gguf::dequant_q8_k(bytes, n),
        GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(bytes, n),
        GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(bytes, n),
        GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(bytes, n),
        GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(bytes, n),
        GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(bytes, n),
        GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(bytes, n),
        GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(bytes, n),
        GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(bytes, n),
        GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(bytes, n),
        GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(bytes, n),
        GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(bytes, n),
        GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(bytes, n),
        GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(bytes, n),
        GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(bytes, n),
        other => return Err(format!("unsupported GGUF scheme {other:?}")),
    };
    r.map_err(|e| e.to_string())
}
