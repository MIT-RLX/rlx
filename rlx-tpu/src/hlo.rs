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

//! HLO module builder — produces serialized `xla.HloModuleProto` bytes
//! via prost-generated types.
//!
//! Field numbers come from the vendored `proto/xla/{xla_data,service/hlo}.proto`
//! files; prost handles wire encoding. The builder layer here keeps the
//! high-level API (`HloBuilder`, `Computation::parameter`, `binary`,
//! `dot_general`, `reduce`, etc.) so `lower.rs` doesn't have to know
//! about prost internals.
//!
//! ## Usage
//!
//! ```ignore
//! let mut b = HloBuilder::new("rlx_module");
//! let entry = b.computation("entry");
//! let p0 = entry.parameter(0, "x", Shape::f32(&[2, 3]));
//! let p1 = entry.parameter(1, "y", Shape::f32(&[2, 3]));
//! let s  = entry.binary("add", p0, p1, Shape::f32(&[2, 3]));
//! entry.set_root(s);
//! entry.set_program_shape(...);
//! let bytes = b.finish();
//! ```
//!
//! ## Why prost
//!
//! The earlier hand-rolled wire encoder drifted from upstream's
//! renumbered fields (`entry_computation_id`, `operand_ids`,
//! `comparison_direction`, `parameter_number`) and produced bytes
//! XLA's deserializer couldn't parse. Prost compiles the .proto
//! files at build time, so the field numbers are auto-correct.

use std::cell::RefCell;
use std::rc::Rc;

use prost::Message;
use rlx_ir::DType;

use crate::xla;

// Re-export PrimitiveType constants in the form lower.rs uses.
pub mod prim {
    use crate::xla::PrimitiveType;
    pub const INVALID: i32 = PrimitiveType::Invalid as i32;
    pub const PRED: i32 = PrimitiveType::Pred as i32;
    pub const S8: i32 = PrimitiveType::S8 as i32;
    pub const S16: i32 = PrimitiveType::S16 as i32;
    pub const S32: i32 = PrimitiveType::S32 as i32;
    pub const S64: i32 = PrimitiveType::S64 as i32;
    pub const U8: i32 = PrimitiveType::U8 as i32;
    pub const U16: i32 = PrimitiveType::U16 as i32;
    pub const U32: i32 = PrimitiveType::U32 as i32;
    pub const U64: i32 = PrimitiveType::U64 as i32;
    pub const F16: i32 = PrimitiveType::F16 as i32;
    pub const F32: i32 = PrimitiveType::F32 as i32;
    pub const F64: i32 = PrimitiveType::F64 as i32;
    pub const TUPLE: i32 = PrimitiveType::Tuple as i32;
    pub const BF16: i32 = PrimitiveType::Bf16 as i32;
    pub const TOKEN: i32 = PrimitiveType::Token as i32;
}

pub fn prim_of(dt: DType) -> i32 {
    match dt {
        DType::F32 => prim::F32,
        DType::F16 => prim::F16,
        DType::BF16 => prim::BF16,
        DType::F64 => prim::F64,
        DType::I8 => prim::S8,
        DType::I16 => prim::S16,
        DType::I32 => prim::S32,
        DType::I64 => prim::S64,
        DType::U8 => prim::U8,
        DType::U32 => prim::U32,
        DType::Bool => prim::PRED,
        DType::C64 => panic!("rlx-tpu: DType::C64 (complex) not yet supported"),
    }
}

// ── Shape — convenience wrapper that builds an `xla::ShapeProto` ──

#[derive(Clone, Debug)]
pub struct Shape {
    pub element_type: i32,
    pub dimensions: Vec<i64>,
    pub layout: Vec<i64>,
    pub tuple_shapes: Vec<Shape>,
}

impl Shape {
    pub fn scalar(element_type: i32) -> Shape {
        Shape {
            element_type,
            dimensions: vec![],
            layout: vec![],
            tuple_shapes: vec![],
        }
    }
    pub fn array(element_type: i32, dims: &[i64]) -> Shape {
        let layout: Vec<i64> = (0..dims.len() as i64).rev().collect();
        Shape {
            element_type,
            dimensions: dims.to_vec(),
            layout,
            tuple_shapes: vec![],
        }
    }
    pub fn f32(dims: &[i64]) -> Shape {
        Shape::array(prim::F32, dims)
    }
    pub fn f16(dims: &[i64]) -> Shape {
        Shape::array(prim::F16, dims)
    }
    pub fn pred(dims: &[i64]) -> Shape {
        Shape::array(prim::PRED, dims)
    }
    pub fn s32(dims: &[i64]) -> Shape {
        Shape::array(prim::S32, dims)
    }
    pub fn from_dt(dt: DType, dims: &[i64]) -> Shape {
        Shape::array(prim_of(dt), dims)
    }
    pub fn tuple(elems: Vec<Shape>) -> Shape {
        Shape {
            element_type: prim::TUPLE,
            dimensions: vec![],
            layout: vec![],
            tuple_shapes: elems,
        }
    }
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
    pub fn num_elements(&self) -> i64 {
        if self.dimensions.is_empty() {
            1
        } else {
            self.dimensions.iter().product()
        }
    }

    /// Build a `xla::ShapeProto` (the prost-generated message). Adds a
    /// default `tail_padding_alignment_in_elements = 1` and an
    /// `is_dynamic_dimension` vector of all-false matching the
    /// dimensions count — same fields JAX's HLO emits, both required
    /// in practice for XLA's deserializer to be happy.
    pub fn to_proto(&self) -> xla::ShapeProto {
        // XLA requires a Layout on every non-tuple ShapeProto — even
        // scalars (empty dimensions). The check `LiteralProto has no
        // layout` fires inside `LiteralProto::CreateFromProto` if a
        // scalar Constant's Shape lacks one. minor_to_major for scalars
        // is empty by definition.
        let layout = if self.element_type == prim::TUPLE {
            None
        } else {
            Some(Box::new(xla::LayoutProto {
                minor_to_major: self.layout.clone(),
                tail_padding_alignment_in_elements: 1,
                ..Default::default()
            }))
        };
        xla::ShapeProto {
            element_type: self.element_type,
            dimensions: self.dimensions.clone(),
            tuple_shapes: self.tuple_shapes.iter().map(|s| s.to_proto()).collect(),
            layout,
            is_dynamic_dimension: vec![false; self.dimensions.len()],
        }
    }
}

// ── ProgramShape ───────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ProgramShape {
    pub parameters: Vec<Shape>,
    pub parameter_names: Vec<String>,
    pub result: Shape,
}

impl ProgramShape {
    pub fn to_proto(&self) -> xla::ProgramShapeProto {
        xla::ProgramShapeProto {
            parameters: self.parameters.iter().map(|s| s.to_proto()).collect(),
            result: Some(self.result.to_proto()),
            parameter_names: self.parameter_names.clone(),
        }
    }
}

// ── Dot / Conv / Gather / Scatter dim numbers ──────────────────

#[derive(Clone, Debug, Default)]
pub struct DotDimNumbers {
    pub lhs_contracting: Vec<i64>,
    pub rhs_contracting: Vec<i64>,
    pub lhs_batch: Vec<i64>,
    pub rhs_batch: Vec<i64>,
}

impl DotDimNumbers {
    fn to_proto(&self) -> xla::DotDimensionNumbers {
        xla::DotDimensionNumbers {
            lhs_contracting_dimensions: self.lhs_contracting.clone(),
            rhs_contracting_dimensions: self.rhs_contracting.clone(),
            lhs_batch_dimensions: self.lhs_batch.clone(),
            rhs_batch_dimensions: self.rhs_batch.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WindowDim {
    pub size: i64,
    pub stride: i64,
    pub padding_low: i64,
    pub padding_high: i64,
    pub window_dilation: i64,
    pub base_dilation: i64,
}

#[derive(Clone, Debug, Default)]
pub struct Window {
    pub dimensions: Vec<WindowDim>,
}

impl Window {
    fn to_proto(&self) -> xla::Window {
        xla::Window {
            dimensions: self
                .dimensions
                .iter()
                .map(|d| xla::WindowDimension {
                    size: d.size,
                    stride: d.stride,
                    padding_low: d.padding_low,
                    padding_high: d.padding_high,
                    window_dilation: d.window_dilation,
                    base_dilation: d.base_dilation,
                    window_reversal: false,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConvDimNumbers {
    pub input_batch_dim: i64,
    pub input_feature_dim: i64,
    pub input_spatial_dims: Vec<i64>,
    pub kernel_input_feature_dim: i64,
    pub kernel_output_feature_dim: i64,
    pub kernel_spatial_dims: Vec<i64>,
    pub output_batch_dim: i64,
    pub output_feature_dim: i64,
    pub output_spatial_dims: Vec<i64>,
}

impl ConvDimNumbers {
    fn to_proto(&self) -> xla::ConvolutionDimensionNumbers {
        xla::ConvolutionDimensionNumbers {
            input_batch_dimension: self.input_batch_dim,
            input_feature_dimension: self.input_feature_dim,
            input_spatial_dimensions: self.input_spatial_dims.clone(),
            kernel_input_feature_dimension: self.kernel_input_feature_dim,
            kernel_output_feature_dimension: self.kernel_output_feature_dim,
            kernel_spatial_dimensions: self.kernel_spatial_dims.clone(),
            output_batch_dimension: self.output_batch_dim,
            output_feature_dimension: self.output_feature_dim,
            output_spatial_dimensions: self.output_spatial_dims.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GatherDimNumbers {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
}

impl GatherDimNumbers {
    fn to_proto(&self) -> xla::GatherDimensionNumbers {
        xla::GatherDimensionNumbers {
            offset_dims: self.offset_dims.clone(),
            collapsed_slice_dims: self.collapsed_slice_dims.clone(),
            start_index_map: self.start_index_map.clone(),
            index_vector_dim: self.index_vector_dim,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScatterDimNumbers {
    pub update_window_dims: Vec<i64>,
    pub inserted_window_dims: Vec<i64>,
    pub scatter_dims_to_operand_dims: Vec<i64>,
    pub index_vector_dim: i64,
}

impl ScatterDimNumbers {
    fn to_proto(&self) -> xla::ScatterDimensionNumbers {
        xla::ScatterDimensionNumbers {
            update_window_dims: self.update_window_dims.clone(),
            inserted_window_dims: self.inserted_window_dims.clone(),
            scatter_dims_to_operand_dims: self.scatter_dims_to_operand_dims.clone(),
            index_vector_dim: self.index_vector_dim,
            ..Default::default()
        }
    }
}

// ── Literal payload ────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum LiteralData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16Bytes(Vec<u8>),
    BF16Bytes(Vec<u8>),
    Pred(Vec<u8>),
    U8(Vec<u8>),
    S8Bytes(Vec<u8>),
    S32(Vec<i32>),
    S64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Clone, Debug)]
pub struct Literal {
    pub shape: Shape,
    pub data: LiteralData,
}

impl Literal {
    fn to_proto(&self) -> xla::LiteralProto {
        let mut p = xla::LiteralProto {
            shape: Some(self.shape.to_proto()),
            ..Default::default()
        };
        match &self.data {
            LiteralData::Pred(b) => p.preds = b.iter().map(|&v| v != 0).collect(),
            LiteralData::U8(b) => p.u8s = b.clone(),
            LiteralData::S8Bytes(b) => p.s8s = b.clone(),
            LiteralData::F16Bytes(b) => p.f16s = b.clone(),
            LiteralData::BF16Bytes(b) => p.bf16s = b.clone(),
            LiteralData::S32(v) => p.s32s = v.clone(),
            LiteralData::S64(v) => p.s64s = v.clone(),
            LiteralData::U32(v) => p.u32s = v.clone(),
            LiteralData::U64(v) => p.u64s = v.clone(),
            LiteralData::F32(v) => p.f32s = v.clone(),
            LiteralData::F64(v) => p.f64s = v.clone(),
        }
        p
    }
}

// ── Instruction builder (internal) ─────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct Instr {
    pub id: i64,
    pub name: String,
    pub opcode: String,
    pub shape: Shape,
    pub operand_ids: Vec<i64>,

    pub dimensions: Vec<i64>,
    pub literal: Option<Literal>,
    pub parameter_number: i64,
    pub slice_starts: Vec<i64>,
    pub slice_limits: Vec<i64>,
    pub slice_strides: Vec<i64>,

    pub dot_dim_numbers: Option<DotDimNumbers>,
    pub window: Option<Window>,
    pub conv_dim_numbers: Option<ConvDimNumbers>,
    pub feature_group_count: i64,
    pub batch_group_count: i64,

    pub gather_dim_numbers: Option<GatherDimNumbers>,
    pub gather_slice_sizes: Vec<i64>,
    pub indices_are_sorted: bool,

    pub scatter_dim_numbers: Option<ScatterDimNumbers>,
    pub unique_indices: bool,

    pub called_computation_ids: Vec<i64>,

    pub comparison_direction: String,
    pub comparison_type: String,

    pub custom_call_target: String,
    pub custom_call_has_side_effect: bool,
    pub backend_config: Vec<u8>,

    pub padding_config: Option<Vec<(i64, i64, i64)>>,

    // Sort: stable comparator (matches NumPy / torch.sort behavior).
    pub is_stable: bool,

    // RNG: distribution for kRng (UNIFORM=1, NORMAL=2);
    // algorithm for kRngBitGenerator (PHILOX=2, THREE_FRY=1).
    pub rng_distribution: i32,
    pub rng_algorithm: i32,

    // GetTupleElement carries the tuple index in a separate proto
    // field; we stash it in `dimensions[0]` from the builder and
    // copy it across in to_proto.
    // DynamicSlice carries the per-dim slice sizes here.
    pub dynamic_slice_sizes: Vec<i64>,
}

impl Default for Shape {
    fn default() -> Self {
        Shape::scalar(prim::INVALID)
    }
}

impl Instr {
    fn new(id: i64, name: String, opcode: &str, shape: Shape) -> Self {
        Instr {
            id,
            name,
            opcode: opcode.into(),
            shape,
            ..Default::default()
        }
    }

    fn to_proto(&self) -> xla::HloInstructionProto {
        xla::HloInstructionProto {
            name: self.name.clone(),
            opcode: self.opcode.clone(),
            shape: Some(self.shape.to_proto()),
            literal: self.literal.as_ref().map(|l| l.to_proto()),
            parameter_number: self.parameter_number,
            dimensions: if self.opcode == "get-tuple-element" {
                // tuple_index is carried in the `tuple_index` proto
                // field; the `dimensions` list must stay empty.
                vec![]
            } else {
                self.dimensions.clone()
            },
            window: self.window.as_ref().map(|w| w.to_proto()),
            convolution_dimension_numbers: self.conv_dim_numbers.as_ref().map(|c| c.to_proto()),
            feature_group_count: self.feature_group_count,
            batch_group_count: self.batch_group_count,
            slice_dimensions: if self.slice_starts.is_empty() {
                vec![]
            } else {
                self.slice_starts
                    .iter()
                    .zip(self.slice_limits.iter())
                    .zip(self.slice_strides.iter())
                    .map(|((s, l), st)| xla::hlo_instruction_proto::SliceDimensions {
                        start: *s,
                        limit: *l,
                        stride: *st,
                    })
                    .collect()
            },
            id: self.id,
            operand_ids: self.operand_ids.clone(),
            dot_dimension_numbers: self.dot_dim_numbers.as_ref().map(|d| d.to_proto()),
            gather_dimension_numbers: self.gather_dim_numbers.as_ref().map(|g| g.to_proto()),
            gather_slice_sizes: self.gather_slice_sizes.clone(),
            scatter_dimension_numbers: self.scatter_dim_numbers.as_ref().map(|s| s.to_proto()),
            indices_are_sorted: self.indices_are_sorted,
            unique_indices: self.unique_indices,
            called_computation_ids: self.called_computation_ids.clone(),
            comparison_direction: self.comparison_direction.clone(),
            comparison_type: self.comparison_type.clone(),
            custom_call_target: self.custom_call_target.clone(),
            custom_call_has_side_effect: self.custom_call_has_side_effect,
            backend_config: self.backend_config.clone(),
            padding_config: self.padding_config.as_ref().map(|cfg| xla::PaddingConfig {
                dimensions: cfg
                    .iter()
                    .map(|(lo, hi, ip)| xla::padding_config::PaddingConfigDimension {
                        edge_padding_low: *lo,
                        edge_padding_high: *hi,
                        interior_padding: *ip,
                    })
                    .collect(),
            }),
            is_stable: self.is_stable,
            distribution: self.rng_distribution,
            rng_algorithm: self.rng_algorithm,
            tuple_index: if self.opcode == "get-tuple-element" {
                self.dimensions.first().copied().unwrap_or(0)
            } else {
                0
            },
            dynamic_slice_sizes: self.dynamic_slice_sizes.clone(),
            ..Default::default()
        }
    }
}

// We use `Instr.dimensions` as a temporary scratchpad for
// `get-tuple-element`'s `tuple_index` so that the builder API stays
// uniform. The proto field `dimensions` should not carry it through
// for that opcode — handled by overriding `dimensions` in to_proto.
//
// Note: also patched for sort/dynamic-slice/etc. via the existing
// generic `dimensions: self.dimensions.clone()` path above, which
// is correct for those opcodes (sort dim, transpose perm, etc.).

// ── Computation builder ────────────────────────────────────────

#[derive(Default)]
struct ComputationInner {
    pub id: i64,
    pub name: String,
    pub instructions: Vec<Instr>,
    pub root_id: i64,
    pub program_shape: Option<ProgramShape>,
}

#[derive(Clone)]
pub struct Computation {
    inner: Rc<RefCell<ComputationInner>>,
    id_alloc: Rc<RefCell<i64>>,
}

impl Computation {
    fn next_id(&self) -> i64 {
        let mut a = self.id_alloc.borrow_mut();
        *a += 1;
        *a
    }

    fn add_instr(&self, mut i: Instr) -> i64 {
        i.id = self.next_id();
        // XLA's deserializer enforces instruction-name uniqueness
        // across every computation in the module — a reducer's
        // parameter named "x" collides with the entry's input named
        // "x" and the whole module is rejected. Always suffix the
        // user-supplied name with the id allocator's value (which is
        // module-global and monotonic), so each name is unique. The
        // id is also visible in the proto's `id` field; mirroring it
        // in the name keeps the textual HLO dump readable.
        i.name = format!("{}.{}", i.name, i.id);
        let id = i.id;
        self.inner.borrow_mut().instructions.push(i);
        id
    }

    pub fn id(&self) -> i64 {
        self.inner.borrow().id
    }
    pub fn set_root(&self, id: i64) {
        self.inner.borrow_mut().root_id = id;
    }
    pub fn set_program_shape(&self, ps: ProgramShape) {
        self.inner.borrow_mut().program_shape = Some(ps);
    }

    pub fn shape_of(&self, id: i64) -> Shape {
        self.inner
            .borrow()
            .instructions
            .iter()
            .find(|i| i.id == id)
            .map(|i| i.shape.clone())
            .expect("rlx-tpu: shape_of: instruction id not found")
    }

    pub fn parameter(&self, n: i64, name: &str, shape: Shape) -> i64 {
        let mut i = Instr::new(0, name.to_string(), "parameter", shape);
        i.parameter_number = n;
        self.add_instr(i)
    }

    pub fn constant(&self, lit: Literal) -> i64 {
        let shape = lit.shape.clone();
        let mut i = Instr::new(
            0,
            format!("constant.{}", self.id_alloc.borrow()),
            "constant",
            shape,
        );
        i.literal = Some(lit);
        self.add_instr(i)
    }

    pub fn constant_f32_scalar(&self, v: f32) -> i64 {
        self.constant(Literal {
            shape: Shape::scalar(prim::F32),
            data: LiteralData::F32(vec![v]),
        })
    }
    pub fn constant_pred_scalar(&self, v: bool) -> i64 {
        self.constant(Literal {
            shape: Shape::scalar(prim::PRED),
            data: LiteralData::Pred(vec![if v { 1 } else { 0 }]),
        })
    }
    pub fn constant_s32_scalar(&self, v: i32) -> i64 {
        self.constant(Literal {
            shape: Shape::scalar(prim::S32),
            data: LiteralData::S32(vec![v]),
        })
    }

    pub fn unary(&self, opcode: &str, x: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, opcode.into(), opcode, shape);
        i.operand_ids = vec![x];
        self.add_instr(i)
    }

    pub fn binary(&self, opcode: &str, a: i64, b: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, opcode.into(), opcode, shape);
        i.operand_ids = vec![a, b];
        self.add_instr(i)
    }

    pub fn convert(&self, x: i64, shape: Shape) -> i64 {
        self.unary("convert", x, shape)
    }

    pub fn compare(&self, a: i64, b: i64, dir: &str, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "compare".into(), "compare", shape);
        i.operand_ids = vec![a, b];
        i.comparison_direction = dir.to_string();
        self.add_instr(i)
    }

    pub fn select(&self, cond: i64, a: i64, b: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "select".into(), "select", shape);
        i.operand_ids = vec![cond, a, b];
        self.add_instr(i)
    }

    pub fn reshape(&self, x: i64, shape: Shape) -> i64 {
        self.unary("reshape", x, shape)
    }

    pub fn transpose(&self, x: i64, perm: &[i64], shape: Shape) -> i64 {
        let mut i = Instr::new(0, "transpose".into(), "transpose", shape);
        i.operand_ids = vec![x];
        i.dimensions = perm.to_vec();
        self.add_instr(i)
    }

    pub fn broadcast(&self, x: i64, broadcast_dims: &[i64], shape: Shape) -> i64 {
        let mut i = Instr::new(0, "broadcast".into(), "broadcast", shape);
        i.operand_ids = vec![x];
        i.dimensions = broadcast_dims.to_vec();
        self.add_instr(i)
    }

    pub fn slice(
        &self,
        x: i64,
        starts: &[i64],
        limits: &[i64],
        strides: &[i64],
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "slice".into(), "slice", shape);
        i.operand_ids = vec![x];
        i.slice_starts = starts.to_vec();
        i.slice_limits = limits.to_vec();
        i.slice_strides = strides.to_vec();
        self.add_instr(i)
    }

    pub fn concat(&self, xs: &[i64], dim: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "concatenate".into(), "concatenate", shape);
        i.operand_ids = xs.to_vec();
        i.dimensions = vec![dim];
        self.add_instr(i)
    }

    pub fn dot_general(&self, a: i64, b: i64, dn: DotDimNumbers, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "dot".into(), "dot", shape);
        i.operand_ids = vec![a, b];
        i.dot_dim_numbers = Some(dn);
        self.add_instr(i)
    }

    pub fn reduce(
        &self,
        x: i64,
        init: i64,
        reducer: &Computation,
        axes: &[i64],
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "reduce".into(), "reduce", shape);
        i.operand_ids = vec![x, init];
        i.dimensions = axes.to_vec();
        i.called_computation_ids = vec![reducer.id()];
        self.add_instr(i)
    }

    pub fn iota(&self, dim: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "iota".into(), "iota", shape);
        i.dimensions = vec![dim];
        self.add_instr(i)
    }

    pub fn gather(
        &self,
        operand: i64,
        indices: i64,
        dn: GatherDimNumbers,
        slice_sizes: Vec<i64>,
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "gather".into(), "gather", shape);
        i.operand_ids = vec![operand, indices];
        i.gather_dim_numbers = Some(dn);
        i.gather_slice_sizes = slice_sizes;
        self.add_instr(i)
    }

    pub fn scatter(
        &self,
        operand: i64,
        indices: i64,
        updates: i64,
        combiner: &Computation,
        dn: ScatterDimNumbers,
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "scatter".into(), "scatter", shape);
        i.operand_ids = vec![operand, indices, updates];
        i.scatter_dim_numbers = Some(dn);
        i.called_computation_ids = vec![combiner.id()];
        self.add_instr(i)
    }

    pub fn convolution(
        &self,
        a: i64,
        b: i64,
        window: Window,
        cdn: ConvDimNumbers,
        feature_group_count: i64,
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "convolution".into(), "convolution", shape);
        i.operand_ids = vec![a, b];
        i.window = Some(window);
        i.conv_dim_numbers = Some(cdn);
        i.feature_group_count = feature_group_count.max(1);
        self.add_instr(i)
    }

    pub fn reduce_window(
        &self,
        x: i64,
        init: i64,
        reducer: &Computation,
        window: Window,
        shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "reduce-window".into(), "reduce-window", shape);
        i.operand_ids = vec![x, init];
        i.window = Some(window);
        i.called_computation_ids = vec![reducer.id()];
        self.add_instr(i)
    }

    pub fn pad(&self, x: i64, pad_value: i64, config: Vec<(i64, i64, i64)>, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "pad".into(), "pad", shape);
        i.operand_ids = vec![x, pad_value];
        i.padding_config = Some(config);
        self.add_instr(i)
    }

    pub fn tuple(&self, items: &[i64], shape: Shape) -> i64 {
        let mut i = Instr::new(0, "tuple".into(), "tuple", shape);
        i.operand_ids = items.to_vec();
        self.add_instr(i)
    }

    /// `kSort`. `operands[0]` is the keys; `operands[1..]` are values
    /// sorted along with the keys. The comparator subcomputation
    /// takes `2*N` parameters (lhs/rhs interleaved per operand) and
    /// returns a PRED scalar (true → lhs precedes rhs).
    /// `out_shape` is a tuple of the operands' shapes when N > 1, or
    /// the single operand shape when N == 1.
    pub fn sort(
        &self,
        operands: &[i64],
        comparator: &Computation,
        dim: i64,
        is_stable: bool,
        out_shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "sort".into(), "sort", out_shape);
        i.operand_ids = operands.to_vec();
        i.dimensions = vec![dim];
        i.is_stable = is_stable;
        i.called_computation_ids = vec![comparator.id()];
        self.add_instr(i)
    }

    /// `kGetTupleElement`. Project the `index`-th element out of a
    /// tuple-typed value.
    pub fn get_tuple_element(&self, x: i64, index: i64, shape: Shape) -> i64 {
        let mut i = Instr::new(0, "get-tuple-element".into(), "get-tuple-element", shape);
        i.operand_ids = vec![x];
        i.dimensions = vec![index];
        // The proto carries the index in `tuple_index`, not
        // `dimensions`. We map it during to_proto.
        self.add_instr(i)
    }

    /// `kRng` — generate samples from the given distribution. For
    /// uniform real, `a` and `b` are scalars (low / high). Output
    /// dtype is taken from `out_shape`.
    pub fn rng(&self, a: i64, b: i64, distribution: i32, out_shape: Shape) -> i64 {
        let mut i = Instr::new(0, "rng".into(), "rng", out_shape);
        i.operand_ids = vec![a, b];
        i.rng_distribution = distribution;
        self.add_instr(i)
    }

    /// `kRngBitGenerator` — Philox/ThreeFry bit-stream RNG. Returns
    /// a tuple of `(new_state, output)` where `output` has the user
    /// shape and `new_state` has the algorithm's state shape.
    pub fn rng_bit_generator(&self, state: i64, algorithm: i32, out_shape: Shape) -> i64 {
        let mut i = Instr::new(
            0,
            "rng-bit-generator".into(),
            "rng-bit-generator",
            out_shape,
        );
        i.operand_ids = vec![state];
        i.rng_algorithm = algorithm;
        self.add_instr(i)
    }

    /// `kWhile`. Carries `init` through `body` until `cond` returns
    /// false. Both subcomputations take a value of `init`'s shape;
    /// `cond` returns PRED scalar, `body` returns the same shape.
    pub fn while_loop(
        &self,
        init: i64,
        cond: &Computation,
        body: &Computation,
        out_shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "while".into(), "while", out_shape);
        i.operand_ids = vec![init];
        // Upstream contract (xla/hlo/ir/hlo_instruction.cc,
        // CreateWhile): body is appended before condition. The
        // proto deserializer reads the same order.
        i.called_computation_ids = vec![body.id(), cond.id()];
        self.add_instr(i)
    }

    /// `kDynamicSlice`. Each `start` is a 0-D S32 tensor.
    pub fn dynamic_slice(
        &self,
        x: i64,
        starts: &[i64],
        slice_sizes: Vec<i64>,
        out_shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(0, "dynamic-slice".into(), "dynamic-slice", out_shape);
        let mut ops = vec![x];
        ops.extend_from_slice(starts);
        i.operand_ids = ops;
        i.dynamic_slice_sizes = slice_sizes;
        self.add_instr(i)
    }

    /// `kDynamicUpdateSlice`. `update` is the patch; each `start`
    /// is a 0-D S32 tensor. Returns a value of `x`'s shape.
    pub fn dynamic_update_slice(
        &self,
        x: i64,
        update: i64,
        starts: &[i64],
        out_shape: Shape,
    ) -> i64 {
        let mut i = Instr::new(
            0,
            "dynamic-update-slice".into(),
            "dynamic-update-slice",
            out_shape,
        );
        let mut ops = vec![x, update];
        ops.extend_from_slice(starts);
        i.operand_ids = ops;
        self.add_instr(i)
    }

    fn to_proto(&self) -> xla::HloComputationProto {
        let inner = self.inner.borrow();
        xla::HloComputationProto {
            name: inner.name.clone(),
            instructions: inner.instructions.iter().map(|i| i.to_proto()).collect(),
            program_shape: inner.program_shape.as_ref().map(|p| p.to_proto()),
            id: inner.id,
            root_id: inner.root_id,
            ..Default::default()
        }
    }
}

// ── Module builder ─────────────────────────────────────────────

pub struct HloBuilder {
    name: String,
    computations: Vec<Computation>,
    id_alloc: Rc<RefCell<i64>>,
    comp_id_alloc: Rc<RefCell<i64>>,
}

impl HloBuilder {
    pub fn new(name: &str) -> Self {
        HloBuilder {
            name: name.to_string(),
            computations: vec![],
            id_alloc: Rc::new(RefCell::new(0)),
            comp_id_alloc: Rc::new(RefCell::new(0)),
        }
    }

    pub fn computation(&mut self, name: &str) -> Computation {
        let cid = {
            let mut a = self.comp_id_alloc.borrow_mut();
            *a += 1;
            *a
        };
        let c = Computation {
            inner: Rc::new(RefCell::new(ComputationInner {
                id: cid,
                name: name.to_string(),
                instructions: vec![],
                root_id: 0,
                program_shape: None,
            })),
            id_alloc: self.id_alloc.clone(),
        };
        self.computations.push(c.clone());
        c
    }

    pub fn make_reducer(&mut self, name: &str, opcode: &str, prim_ty: i32) -> Computation {
        let c = self.computation(name);
        let s = Shape::scalar(prim_ty);
        let p0 = c.parameter(0, "x", s.clone());
        let p1 = c.parameter(1, "y", s.clone());
        let r = c.binary(opcode, p0, p1, s.clone());
        c.set_root(r);
        c.set_program_shape(ProgramShape {
            parameters: vec![s.clone(), s.clone()],
            parameter_names: vec!["x".into(), "y".into()],
            result: s,
        });
        c
    }

    pub fn finish(self) -> Vec<u8> {
        // The entry computation is the FIRST one we hand out via
        // `computation()`, so its index in `self.computations` is 0.
        // Reducer subcomputations come after, in the order they were
        // requested. HLO's contract: "computations are emitted in a
        // valid dependency order, where callees appear before their
        // callers" (xla/service/hlo.proto on `repeated computations`).
        // Without that ordering, XLA's deserializer rejects with
        // "instruction references invalid computation id(s)" because
        // it processes computations in array order and won't have
        // built the callee yet when the caller references it. So
        // emit reducers first, entry last.
        let entry = self
            .computations
            .first()
            .expect("rlx-tpu: HloBuilder::finish: no computations defined");
        let entry_id = entry.id();
        let entry_name = entry.inner.borrow().name.clone();
        let entry_program_shape = entry
            .inner
            .borrow()
            .program_shape
            .clone()
            .expect("rlx-tpu: entry computation must have a program_shape set");

        let mut sorted: Vec<&Computation> = self.computations.iter().skip(1).collect();
        sorted.push(entry);
        let computations: Vec<xla::HloComputationProto> =
            sorted.iter().map(|c| c.to_proto()).collect();

        let module = xla::HloModuleProto {
            name: self.name.clone(),
            entry_computation_name: entry_name,
            entry_computation_id: entry_id,
            computations,
            host_program_shape: Some(entry_program_shape.to_proto()),
            id: 1,
            ..Default::default()
        };
        module.encode_to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_round_trips() {
        let s = Shape::f32(&[2, 3]);
        let p = s.to_proto();
        assert_eq!(p.element_type, prim::F32);
        assert_eq!(p.dimensions, vec![2, 3]);
        assert!(p.layout.is_some());
    }

    #[test]
    fn minimal_module_encodes() {
        let mut b = HloBuilder::new("rlx_smoke");
        let entry = b.computation("entry");
        let s = Shape::f32(&[4]);
        let p0 = entry.parameter(0, "x", s.clone());
        let p1 = entry.parameter(1, "y", s.clone());
        let r = entry.binary("add", p0, p1, s.clone());
        entry.set_root(r);
        entry.set_program_shape(ProgramShape {
            parameters: vec![s.clone(), s.clone()],
            parameter_names: vec!["x".into(), "y".into()],
            result: s,
        });
        let bytes = b.finish();
        assert!(bytes.len() > 32);
        // Round-trip: parse back via prost — proves our encoding is
        // structurally valid.
        let reparsed = xla::HloModuleProto::decode(bytes.as_slice())
            .expect("emitted module must round-trip through prost");
        assert_eq!(reparsed.name, "rlx_smoke");
        assert_eq!(reparsed.entry_computation_name, "entry");
        assert_eq!(reparsed.computations.len(), 1);
    }
}
