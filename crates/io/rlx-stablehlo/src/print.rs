// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HloModuleProto` → StableHLO MLIR text.
//!
//! Every op is printed in MLIR's *generic* form
//! (`%v = "stablehlo.add"(%a, %b) : (T, T) -> T`), which every MLIR parser
//! accepts regardless of the dialect's pretty syntax. Called computations
//! (reducers, comparators, loop bodies) are inlined as regions.
//!
//! HLO threads multi-value results through tuples; StableHLO uses multiple
//! results instead, and IREE does not take tuple types. So tuples never reach
//! the output: a `tuple` instruction becomes a list of values, a multi-result
//! op (`reduce`, `sort`, `while`, `rng-bit-generator`) yields one, and
//! `get-tuple-element` picks from it. Nested tuples are rejected.
//!
//! Numbers are printed exactly: float elements as their bit patterns, larger
//! constants as a raw little-endian hex blob.

use std::collections::HashMap;
use std::fmt::Write as _;

use rlx_tpu::xla;

use crate::ExportError;

type R<T> = Result<T, ExportError>;

fn err<T>(msg: impl Into<String>) -> R<T> {
    Err(ExportError::Unsupported(msg.into()))
}

/// One SSA value: its name and MLIR type.
#[derive(Clone, Debug)]
struct V {
    name: String,
    ty: String,
}

/// What an HLO instruction id stands for in the output.
#[derive(Clone, Debug)]
enum Val {
    One(V),
    Many(Vec<V>),
}

impl Val {
    fn one(&self, what: &str) -> R<&V> {
        match self {
            Val::One(v) => Ok(v),
            Val::Many(_) => err(format!("{what}: expected a single value, got a tuple")),
        }
    }
    fn flat(&self) -> Vec<V> {
        match self {
            Val::One(v) => vec![v.clone()],
            Val::Many(vs) => vs.clone(),
        }
    }
}

pub(crate) fn element_type(t: i32) -> R<&'static str> {
    use xla::PrimitiveType as P;
    let p = P::try_from(t).map_err(|_| ExportError::Unsupported(format!("element type {t}")))?;
    Ok(match p {
        P::Pred => "i1",
        P::S8 => "i8",
        P::S16 => "i16",
        P::S32 => "i32",
        P::S64 => "i64",
        P::U8 => "ui8",
        P::U16 => "ui16",
        P::U32 => "ui32",
        P::U64 => "ui64",
        P::F16 => "f16",
        P::Bf16 => "bf16",
        P::F32 => "f32",
        P::F64 => "f64",
        P::C64 => "complex<f32>",
        P::C128 => "complex<f64>",
        other => return err(format!("element type {other:?}")),
    })
}

fn tensor_type(s: &xla::ShapeProto) -> R<String> {
    if !s.tuple_shapes.is_empty() {
        return err("tuple-typed value where a tensor was expected");
    }
    let et = element_type(s.element_type)?;
    if s.dimensions.is_empty() {
        return Ok(format!("tensor<{et}>"));
    }
    let dims: Vec<String> = s.dimensions.iter().map(|d| d.to_string()).collect();
    Ok(format!("tensor<{}x{et}>", dims.join("x")))
}

/// The tensor types of a (possibly tuple) shape, flattened one level.
fn result_types(s: &xla::ShapeProto) -> R<Vec<String>> {
    if s.tuple_shapes.is_empty() {
        return Ok(vec![tensor_type(s)?]);
    }
    s.tuple_shapes.iter().map(tensor_type).collect()
}

fn i64_array(xs: &[i64]) -> String {
    if xs.is_empty() {
        return "array<i64>".into();
    }
    let v: Vec<String> = xs.iter().map(|x| x.to_string()).collect();
    format!("array<i64: {}>", v.join(", "))
}

fn i64_list(xs: &[i64]) -> String {
    let v: Vec<String> = xs.iter().map(|x| x.to_string()).collect();
    format!("[{}]", v.join(", "))
}

/// `dense<[[lo, hi], …]> : tensor<Nx2xi64>`.
fn padding_attr(pairs: &[(i64, i64)]) -> String {
    let rows: Vec<String> = pairs.iter().map(|(l, h)| format!("[{l}, {h}]")).collect();
    format!(
        "dense<[{}]> : tensor<{}x2xi64>",
        rows.join(", "),
        pairs.len()
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 + bytes.len() * 2);
    s.push_str("0x");
    for b in bytes {
        let _ = write!(s, "{b:02X}");
    }
    s
}

/// Nested `[[…], …]` from flat element strings and a shape.
fn nested(elems: &[String], dims: &[i64]) -> String {
    if dims.is_empty() {
        return elems.first().cloned().unwrap_or_default();
    }
    let n = dims[0] as usize;
    let inner: usize = dims[1..].iter().product::<i64>().max(1) as usize;
    let parts: Vec<String> = (0..n)
        .map(|i| nested(&elems[i * inner..(i + 1) * inner], &dims[1..]))
        .collect();
    format!("[{}]", parts.join(", "))
}

/// A `dense<…>` attribute for a literal, exact for every element type.
fn dense_literal(lit: &xla::LiteralProto) -> R<String> {
    let shape = lit
        .shape
        .as_ref()
        .ok_or_else(|| ExportError::Unsupported("constant without a shape".into()))?;
    let ty = tensor_type(shape)?;
    let n: usize = shape.dimensions.iter().product::<i64>().max(1) as usize;
    if shape.element_type == xla::PrimitiveType::Pred as i32 {
        let elems: Vec<String> = lit.preds.iter().map(|b| b.to_string()).collect();
        if elems.len() != n {
            return err("pred constant length does not match its shape");
        }
        return Ok(format!(
            "dense<{}> : {ty}",
            nested(&elems, &shape.dimensions)
        ));
    }
    // Raw little-endian element bytes, in the order MLIR's hex form expects.
    let bytes: Vec<u8> = if !lit.f32s.is_empty() {
        lit.f32s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.f64s.is_empty() {
        lit.f64s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.s32s.is_empty() {
        lit.s32s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.s64s.is_empty() {
        lit.s64s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.u32s.is_empty() {
        lit.u32s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.u64s.is_empty() {
        lit.u64s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.c64s.is_empty() {
        lit.c64s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.c128s.is_empty() {
        lit.c128s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if !lit.f16s.is_empty() {
        lit.f16s.clone()
    } else if !lit.bf16s.is_empty() {
        lit.bf16s.clone()
    } else if !lit.u8s.is_empty() {
        lit.u8s.clone()
    } else if !lit.s8s.is_empty() {
        lit.s8s.clone()
    } else if !lit.s16s.is_empty() {
        lit.s16s.clone()
    } else if !lit.u16s.is_empty() {
        lit.u16s.clone()
    } else if n == 0 {
        Vec::new()
    } else {
        return err(format!("constant of type {ty} has no element data"));
    };
    Ok(format!("dense<\"{}\"> : {ty}", hex(&bytes)))
}

/// HLO elementwise opcode → op name.
fn elementwise(opcode: &str) -> Option<&'static str> {
    Some(match opcode {
        "add" => "stablehlo.add",
        "subtract" => "stablehlo.subtract",
        "multiply" => "stablehlo.multiply",
        "divide" => "stablehlo.divide",
        "maximum" => "stablehlo.maximum",
        "minimum" => "stablehlo.minimum",
        "power" => "stablehlo.power",
        "remainder" => "stablehlo.remainder",
        "atan2" => "stablehlo.atan2",
        "and" => "stablehlo.and",
        "or" => "stablehlo.or",
        "xor" => "stablehlo.xor",
        "not" => "stablehlo.not",
        "shift-left" => "stablehlo.shift_left",
        "shift-right-logical" => "stablehlo.shift_right_logical",
        "shift-right-arithmetic" => "stablehlo.shift_right_arithmetic",
        "abs" => "stablehlo.abs",
        "negate" => "stablehlo.negate",
        "exponential" => "stablehlo.exponential",
        "exponential-minus-one" => "stablehlo.exponential_minus_one",
        "log" => "stablehlo.log",
        "log-plus-one" => "stablehlo.log_plus_one",
        "rsqrt" => "stablehlo.rsqrt",
        "sqrt" => "stablehlo.sqrt",
        "cbrt" => "stablehlo.cbrt",
        "tanh" => "stablehlo.tanh",
        "logistic" => "stablehlo.logistic",
        "sign" => "stablehlo.sign",
        "floor" => "stablehlo.floor",
        "ceil" => "stablehlo.ceil",
        "sine" => "stablehlo.sine",
        "cosine" => "stablehlo.cosine",
        "tan" => "stablehlo.tan",
        "round-nearest-even" => "stablehlo.round_nearest_even",
        "round-nearest-afz" => "stablehlo.round_nearest_afz",
        "is-finite" => "stablehlo.is_finite",
        "popcnt" => "stablehlo.popcnt",
        "clz" => "stablehlo.count_leading_zeros",
        "real" => "stablehlo.real",
        "imag" => "stablehlo.imag",
        "complex" => "stablehlo.complex",
        "convert" => "stablehlo.convert",
        "reshape" => "stablehlo.reshape",
        "select" => "stablehlo.select",
        // No StableHLO op; CHLO is part of every StableHLO consumer's input.
        "erf" => "chlo.erf",
        "atan" => "chlo.atan",
        "conjugate" => "chlo.conj",
        _ => return None,
    })
}

struct Printer<'m> {
    comps: HashMap<i64, &'m xla::HloComputationProto>,
    out: String,
}

impl<'m> Printer<'m> {
    fn line(&mut self, indent: usize, s: &str) {
        for _ in 0..indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn comp(&self, id: i64) -> R<&'m xla::HloComputationProto> {
        self.comps
            .get(&id)
            .copied()
            .ok_or_else(|| ExportError::Unsupported(format!("missing computation {id}")))
    }

    /// Print `comp` as a region: block arguments for its parameters, its
    /// instructions, and a `stablehlo.return` of its root.
    fn region(&mut self, comp: &xla::HloComputationProto, indent: usize) -> R<()> {
        let mut vals: HashMap<i64, Val> = HashMap::new();
        let mut params: Vec<&xla::HloInstructionProto> = comp
            .instructions
            .iter()
            .filter(|i| i.opcode == "parameter")
            .collect();
        params.sort_by_key(|i| i.parameter_number);
        let mut args = Vec::new();
        for p in params {
            let shape = p.shape.as_ref().ok_or_else(|| no_shape(p))?;
            let tys = result_types(shape)?;
            let vs: Vec<V> = tys
                .iter()
                .enumerate()
                .map(|(k, ty)| V {
                    name: if tys.len() == 1 && shape.tuple_shapes.is_empty() {
                        format!("%a{}", p.id)
                    } else {
                        format!("%a{}_{k}", p.id)
                    },
                    ty: ty.clone(),
                })
                .collect();
            args.extend(vs.iter().map(|v| format!("{}: {}", v.name, v.ty)));
            vals.insert(
                p.id,
                if shape.tuple_shapes.is_empty() {
                    Val::One(vs[0].clone())
                } else {
                    Val::Many(vs)
                },
            );
        }
        self.line(indent, &format!("^bb0({}):", args.join(", ")));
        self.body(comp, &mut vals, indent + 1)?;
        let root = vals
            .get(&comp.root_id)
            .ok_or_else(|| ExportError::Unsupported("region root not found".into()))?
            .flat();
        self.line(indent + 1, &return_op("stablehlo.return", &root));
        Ok(())
    }

    /// Print every non-parameter instruction of `comp`, recording values.
    fn body(
        &mut self,
        comp: &xla::HloComputationProto,
        vals: &mut HashMap<i64, Val>,
        indent: usize,
    ) -> R<()> {
        for ins in &comp.instructions {
            if ins.opcode == "parameter" {
                continue;
            }
            let v = self.instr(ins, vals, indent)?;
            vals.insert(ins.id, v);
        }
        Ok(())
    }

    fn instr(
        &mut self,
        ins: &xla::HloInstructionProto,
        vals: &HashMap<i64, Val>,
        indent: usize,
    ) -> R<Val> {
        let shape = ins.shape.as_ref().ok_or_else(|| no_shape(ins))?;
        let operand = |k: usize| -> R<Val> {
            let id = ins.operand_ids.get(k).ok_or_else(|| {
                ExportError::Unsupported(format!("{}: missing operand {k}", ins.opcode))
            })?;
            vals.get(id).cloned().ok_or_else(|| {
                ExportError::Unsupported(format!("{}: operand {id} not defined", ins.opcode))
            })
        };
        let operands_flat = || -> R<Vec<V>> {
            let mut out = Vec::new();
            for k in 0..ins.operand_ids.len() {
                out.extend(operand(k)?.flat());
            }
            Ok(out)
        };
        let name = format!("%v{}", ins.id);
        let op = ins.opcode.as_str();

        // Tuples are bookkeeping, not ops.
        if op == "tuple" {
            let mut vs = Vec::new();
            for k in 0..ins.operand_ids.len() {
                vs.push(operand(k)?.one("tuple element")?.clone());
            }
            return Ok(Val::Many(vs));
        }
        if op == "get-tuple-element" {
            let t = operand(0)?;
            let Val::Many(vs) = t else {
                return err("get-tuple-element of a non-tuple");
            };
            let idx = ins.tuple_index as usize;
            return vs
                .get(idx)
                .cloned()
                .map(Val::One)
                .ok_or_else(|| ExportError::Unsupported("tuple index out of range".into()));
        }

        let results = result_types(shape)?;
        let mut attrs: Vec<String> = Vec::new();
        let mut regions: Vec<i64> = Vec::new();
        let mut ops = operands_flat()?;
        let opname: String = match op {
            "constant" => {
                let lit = ins
                    .literal
                    .as_ref()
                    .ok_or_else(|| ExportError::Unsupported("constant without literal".into()))?;
                attrs.push(format!("value = {}", dense_literal(lit)?));
                "stablehlo.constant".into()
            }
            "compare" => {
                attrs.push(format!(
                    "comparison_direction = #stablehlo<comparison_direction {}>",
                    ins.comparison_direction
                ));
                if !ins.comparison_type.is_empty() {
                    attrs.push(format!(
                        "compare_type = #stablehlo<comparison_type {}>",
                        ins.comparison_type
                    ));
                }
                "stablehlo.compare".into()
            }
            "transpose" => {
                attrs.push(format!("permutation = {}", i64_array(&ins.dimensions)));
                "stablehlo.transpose".into()
            }
            "reverse" => {
                attrs.push(format!("dimensions = {}", i64_array(&ins.dimensions)));
                "stablehlo.reverse".into()
            }
            "broadcast" => {
                attrs.push(format!(
                    "broadcast_dimensions = {}",
                    i64_array(&ins.dimensions)
                ));
                "stablehlo.broadcast_in_dim".into()
            }
            "slice" => {
                let (mut s, mut l, mut t) = (Vec::new(), Vec::new(), Vec::new());
                for d in &ins.slice_dimensions {
                    s.push(d.start);
                    l.push(d.limit);
                    t.push(d.stride);
                }
                attrs.push(format!("start_indices = {}", i64_array(&s)));
                attrs.push(format!("limit_indices = {}", i64_array(&l)));
                attrs.push(format!("strides = {}", i64_array(&t)));
                "stablehlo.slice".into()
            }
            "concatenate" => {
                attrs.push(format!("dimension = {} : i64", ins.dimensions[0]));
                "stablehlo.concatenate".into()
            }
            "dot" => {
                let dn = ins.dot_dimension_numbers.as_ref().ok_or_else(|| {
                    ExportError::Unsupported("dot without dimension numbers".into())
                })?;
                attrs.push(format!(
                    "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = {}, \
                     rhs_batching_dimensions = {}, lhs_contracting_dimensions = {}, \
                     rhs_contracting_dimensions = {}>",
                    i64_list(&dn.lhs_batch_dimensions),
                    i64_list(&dn.rhs_batch_dimensions),
                    i64_list(&dn.lhs_contracting_dimensions),
                    i64_list(&dn.rhs_contracting_dimensions)
                ));
                if let Some(pc) = &ins.precision_config
                    && !pc.operand_precision.is_empty()
                {
                    let p: Vec<String> = pc
                        .operand_precision
                        .iter()
                        .map(|&p| {
                            let n = match p {
                                1 => "HIGH",
                                2 => "HIGHEST",
                                _ => "DEFAULT",
                            };
                            format!("#stablehlo<precision {n}>")
                        })
                        .collect();
                    attrs.push(format!("precision_config = [{}]", p.join(", ")));
                }
                "stablehlo.dot_general".into()
            }
            "reduce" => {
                attrs.push(format!("dimensions = {}", i64_array(&ins.dimensions)));
                regions.push(ins.called_computation_ids[0]);
                "stablehlo.reduce".into()
            }
            "iota" => {
                attrs.push(format!("iota_dimension = {} : i64", ins.dimensions[0]));
                "stablehlo.iota".into()
            }
            "gather" => {
                let dn = ins.gather_dimension_numbers.as_ref().ok_or_else(|| {
                    ExportError::Unsupported("gather without dimension numbers".into())
                })?;
                let mut f = vec![
                    format!("offset_dims = {}", i64_list(&dn.offset_dims)),
                    format!(
                        "collapsed_slice_dims = {}",
                        i64_list(&dn.collapsed_slice_dims)
                    ),
                ];
                if !dn.operand_batching_dims.is_empty() {
                    f.push(format!(
                        "operand_batching_dims = {}",
                        i64_list(&dn.operand_batching_dims)
                    ));
                    f.push(format!(
                        "start_indices_batching_dims = {}",
                        i64_list(&dn.start_indices_batching_dims)
                    ));
                }
                f.push(format!(
                    "start_index_map = {}",
                    i64_list(&dn.start_index_map)
                ));
                f.push(format!("index_vector_dim = {}", dn.index_vector_dim));
                attrs.push(format!(
                    "dimension_numbers = #stablehlo.gather<{}>",
                    f.join(", ")
                ));
                attrs.push(format!(
                    "slice_sizes = {}",
                    i64_array(&ins.gather_slice_sizes)
                ));
                attrs.push(format!("indices_are_sorted = {}", ins.indices_are_sorted));
                "stablehlo.gather".into()
            }
            "scatter" => {
                let dn = ins.scatter_dimension_numbers.as_ref().ok_or_else(|| {
                    ExportError::Unsupported("scatter without dimension numbers".into())
                })?;
                let mut f = vec![
                    format!("update_window_dims = {}", i64_list(&dn.update_window_dims)),
                    format!(
                        "inserted_window_dims = {}",
                        i64_list(&dn.inserted_window_dims)
                    ),
                ];
                if !dn.input_batching_dims.is_empty() {
                    f.push(format!(
                        "input_batching_dims = {}",
                        i64_list(&dn.input_batching_dims)
                    ));
                    f.push(format!(
                        "scatter_indices_batching_dims = {}",
                        i64_list(&dn.scatter_indices_batching_dims)
                    ));
                }
                f.push(format!(
                    "scatter_dims_to_operand_dims = {}",
                    i64_list(&dn.scatter_dims_to_operand_dims)
                ));
                f.push(format!("index_vector_dim = {}", dn.index_vector_dim));
                attrs.push(format!(
                    "scatter_dimension_numbers = #stablehlo.scatter<{}>",
                    f.join(", ")
                ));
                attrs.push(format!("indices_are_sorted = {}", ins.indices_are_sorted));
                attrs.push(format!("unique_indices = {}", ins.unique_indices));
                regions.push(ins.called_computation_ids[0]);
                "stablehlo.scatter".into()
            }
            "convolution" => {
                let cdn = ins.convolution_dimension_numbers.as_ref().ok_or_else(|| {
                    ExportError::Unsupported("convolution without dimension numbers".into())
                })?;
                attrs.push(format!("dimension_numbers = {}", conv_dims(cdn, shape)));
                let w = ins.window.clone().unwrap_or_default();
                window_attrs(&w, &mut attrs, WindowKind::Conv);
                attrs.push(format!(
                    "feature_group_count = {} : i64",
                    ins.feature_group_count.max(1)
                ));
                attrs.push(format!(
                    "batch_group_count = {} : i64",
                    ins.batch_group_count.max(1)
                ));
                "stablehlo.convolution".into()
            }
            "reduce-window" => {
                let w = ins.window.clone().unwrap_or_default();
                window_attrs(&w, &mut attrs, WindowKind::ReduceWindow);
                regions.push(ins.called_computation_ids[0]);
                "stablehlo.reduce_window".into()
            }
            "select-and-scatter" => {
                let w = ins.window.clone().unwrap_or_default();
                window_attrs(&w, &mut attrs, WindowKind::SelectAndScatter);
                regions.extend_from_slice(&ins.called_computation_ids[..2]);
                "stablehlo.select_and_scatter".into()
            }
            "pad" => {
                let pc = ins.padding_config.clone().unwrap_or_default();
                let lo: Vec<i64> = pc.dimensions.iter().map(|d| d.edge_padding_low).collect();
                let hi: Vec<i64> = pc.dimensions.iter().map(|d| d.edge_padding_high).collect();
                let it: Vec<i64> = pc.dimensions.iter().map(|d| d.interior_padding).collect();
                attrs.push(format!("edge_padding_low = {}", i64_array(&lo)));
                attrs.push(format!("edge_padding_high = {}", i64_array(&hi)));
                attrs.push(format!("interior_padding = {}", i64_array(&it)));
                "stablehlo.pad".into()
            }
            "sort" => {
                attrs.push(format!("dimension = {} : i64", ins.dimensions[0]));
                attrs.push(format!("is_stable = {}", ins.is_stable));
                regions.push(ins.called_computation_ids[0]);
                "stablehlo.sort".into()
            }
            "dynamic-slice" => {
                attrs.push(format!(
                    "slice_sizes = {}",
                    i64_array(&ins.dynamic_slice_sizes)
                ));
                "stablehlo.dynamic_slice".into()
            }
            "dynamic-update-slice" => "stablehlo.dynamic_update_slice".into(),
            "fft" => {
                let t = match ins.fft_type {
                    1 => "IFFT",
                    2 => "RFFT",
                    3 => "IRFFT",
                    _ => "FFT",
                };
                attrs.push(format!("fft_type = #stablehlo<fft_type {t}>"));
                attrs.push(format!("fft_length = {}", i64_array(&ins.fft_length)));
                "stablehlo.fft".into()
            }
            "rng" => {
                let d = if ins.distribution == 2 {
                    "NORMAL"
                } else {
                    "UNIFORM"
                };
                attrs.push(format!(
                    "rng_distribution = #stablehlo<rng_distribution {d}>"
                ));
                // StableHLO takes the result shape as an operand.
                let dims = &shape.dimensions;
                let sname = format!("%v{}_shape", ins.id);
                let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
                let sty = format!("tensor<{}xi64>", dims.len());
                self.line(
                    indent,
                    &format!(
                        "{sname} = \"stablehlo.constant\"() {{value = dense<\"{}\"> : {sty}}} : () -> {sty}",
                        hex(&bytes)
                    ),
                );
                ops.push(V {
                    name: sname,
                    ty: sty,
                });
                "stablehlo.rng".into()
            }
            "rng-bit-generator" => {
                let a = match ins.rng_algorithm {
                    1 => "THREE_FRY",
                    2 => "PHILOX",
                    _ => "DEFAULT",
                };
                attrs.push(format!("rng_algorithm = #stablehlo<rng_algorithm {a}>"));
                "stablehlo.rng_bit_generator".into()
            }
            "while" => {
                // HLO stores body first, then condition; StableHLO's regions
                // are (cond, body).
                let (body, cond) = (ins.called_computation_ids[0], ins.called_computation_ids[1]);
                regions.push(cond);
                regions.push(body);
                "stablehlo.while".into()
            }
            "reciprocal" => {
                // No StableHLO reciprocal: 1 / x.
                let x = ops[0].clone();
                let one = format!("%v{}_one", ins.id);
                self.line(
                    indent,
                    &format!(
                        "{one} = \"stablehlo.constant\"() {{value = dense<1.0> : {t}}} : () -> {t}",
                        t = x.ty
                    ),
                );
                ops = vec![
                    V {
                        name: one,
                        ty: x.ty.clone(),
                    },
                    x,
                ];
                "stablehlo.divide".into()
            }
            "custom-call" => {
                return err(format!(
                    "custom-call `{}` has no portable StableHLO form",
                    ins.custom_call_target
                ));
            }
            other => match elementwise(other) {
                Some(n) => n.into(),
                None => return err(format!("HLO opcode `{other}`")),
            },
        };
        let lhs = if results.len() == 1 {
            name.clone()
        } else {
            format!("{name}:{}", results.len())
        };
        let operand_names: Vec<&str> = ops.iter().map(|v| v.name.as_str()).collect();
        let operand_types: Vec<&str> = ops.iter().map(|v| v.ty.as_str()).collect();
        let fn_ty = format!(
            "({}) -> {}",
            operand_types.join(", "),
            if results.len() == 1 {
                results[0].clone()
            } else {
                format!("({})", results.join(", "))
            }
        );
        let attr_dict = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{}}}", attrs.join(", "))
        };
        if regions.is_empty() {
            self.line(
                indent,
                &format!(
                    "{lhs} = \"{opname}\"({}){attr_dict} : {fn_ty}",
                    operand_names.join(", ")
                ),
            );
        } else {
            self.line(
                indent,
                &format!("{lhs} = \"{opname}\"({}) (", operand_names.join(", ")),
            );
            for (k, cid) in regions.iter().enumerate() {
                let comp = self.comp(*cid)?;
                self.line(indent + 1, "{");
                self.region(comp, indent + 1)?;
                self.line(indent + 1, if k + 1 == regions.len() { "}" } else { "}," });
            }
            self.line(indent, &format!("){attr_dict} : {fn_ty}"));
        }

        let vs: Vec<V> = results
            .iter()
            .enumerate()
            .map(|(k, ty)| V {
                name: if results.len() == 1 {
                    name.clone()
                } else {
                    format!("{name}#{k}")
                },
                ty: ty.clone(),
            })
            .collect();
        Ok(if shape.tuple_shapes.is_empty() {
            Val::One(vs[0].clone())
        } else {
            Val::Many(vs)
        })
    }
}

fn return_op(op: &str, vs: &[V]) -> String {
    let names: Vec<&str> = vs.iter().map(|v| v.name.as_str()).collect();
    let tys: Vec<&str> = vs.iter().map(|v| v.ty.as_str()).collect();
    format!(
        "\"{op}\"({}) : ({}) -> ()",
        names.join(", "),
        tys.join(", ")
    )
}

fn no_shape(i: &xla::HloInstructionProto) -> ExportError {
    ExportError::Unsupported(format!("{} without a shape", i.opcode))
}

#[derive(Clone, Copy, PartialEq)]
enum WindowKind {
    Conv,
    ReduceWindow,
    SelectAndScatter,
}

fn window_attrs(w: &xla::Window, attrs: &mut Vec<String>, kind: WindowKind) {
    let d = &w.dimensions;
    let size: Vec<i64> = d.iter().map(|x| x.size).collect();
    let stride: Vec<i64> = d.iter().map(|x| x.stride.max(1)).collect();
    let pad: Vec<(i64, i64)> = d.iter().map(|x| (x.padding_low, x.padding_high)).collect();
    let base: Vec<i64> = d.iter().map(|x| x.base_dilation.max(1)).collect();
    let win: Vec<i64> = d.iter().map(|x| x.window_dilation.max(1)).collect();
    match kind {
        WindowKind::Conv => {
            attrs.push(format!("window_strides = {}", i64_array(&stride)));
            attrs.push(format!("padding = {}", padding_attr(&pad)));
            // Always explicit: IREE's legalization mis-sizes the dilations of
            // a convolution that omits them.
            attrs.push(format!("lhs_dilation = {}", i64_array(&base)));
            attrs.push(format!("rhs_dilation = {}", i64_array(&win)));
            if d.iter().any(|x| x.window_reversal) {
                let r: Vec<String> = d.iter().map(|x| x.window_reversal.to_string()).collect();
                attrs.push(format!("window_reversal = array<i1: {}>", r.join(", ")));
            }
        }
        WindowKind::ReduceWindow => {
            attrs.push(format!("window_dimensions = {}", i64_array(&size)));
            attrs.push(format!("window_strides = {}", i64_array(&stride)));
            attrs.push(format!("base_dilations = {}", i64_array(&base)));
            attrs.push(format!("window_dilations = {}", i64_array(&win)));
            attrs.push(format!("padding = {}", padding_attr(&pad)));
        }
        WindowKind::SelectAndScatter => {
            attrs.push(format!("window_dimensions = {}", i64_array(&size)));
            attrs.push(format!("window_strides = {}", i64_array(&stride)));
            attrs.push(format!("padding = {}", padding_attr(&pad)));
        }
    }
}

/// `#stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>` — the layout
/// spelled per tensor, so NCHW/OIHW (rlx's convention) needs no transposes.
fn conv_dims(c: &xla::ConvolutionDimensionNumbers, out: &xla::ShapeProto) -> String {
    let spatial = c.input_spatial_dimensions.len();
    let rank = spatial + 2;
    let label = |batch: i64, feat: i64, sp: &[i64], b: &str, f: &str| -> String {
        let mut v = vec![String::new(); rank];
        v[batch as usize] = b.into();
        v[feat as usize] = f.into();
        for (k, &d) in sp.iter().enumerate() {
            v[d as usize] = k.to_string();
        }
        format!("[{}]", v.join(", "))
    };
    let _ = out;
    format!(
        "#stablehlo.conv<{}x{}->{}>",
        label(
            c.input_batch_dimension,
            c.input_feature_dimension,
            &c.input_spatial_dimensions,
            "b",
            "f"
        ),
        label(
            c.kernel_output_feature_dimension,
            c.kernel_input_feature_dimension,
            &c.kernel_spatial_dimensions,
            "o",
            "i"
        ),
        label(
            c.output_batch_dimension,
            c.output_feature_dimension,
            &c.output_spatial_dimensions,
            "b",
            "f"
        ),
    )
}

/// Print `module` as a StableHLO `module { func.func @<entry> … }`.
/// Where the entry's trailing (weight) parameters come from.
pub(crate) enum ParamSource<'a> {
    /// Ordinary function arguments.
    Arguments,
    /// IREE parameters: `util.global` bound to `#flow.parameter.named<scope::name>`.
    Named { scope: &'a str },
}

/// MLIR string-literal escaping for symbol and parameter names.
fn quoted(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                o.push('\\');
                o.push(c);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Print `module` as a StableHLO `module { func.func @<entry> … }`.
///
/// `arg_names` names every entry parameter in order; the last `n_params` are
/// weights, placed according to `params`.
pub(crate) fn print_module(
    module: &xla::HloModuleProto,
    func_name: &str,
    arg_names: &[String],
    n_params: usize,
    params_src: ParamSource<'_>,
) -> R<Printed> {
    let comps: HashMap<i64, &xla::HloComputationProto> =
        module.computations.iter().map(|c| (c.id, c)).collect();
    let entry = *comps
        .get(&module.entry_computation_id)
        .ok_or_else(|| ExportError::Unsupported("module has no entry computation".into()))?;
    let mut p = Printer {
        comps,
        out: String::new(),
    };

    let mut params: Vec<&xla::HloInstructionProto> = entry
        .instructions
        .iter()
        .filter(|i| i.opcode == "parameter")
        .collect();
    params.sort_by_key(|i| i.parameter_number);
    let first_weight = params.len().saturating_sub(n_params);
    let mut vals: HashMap<i64, Val> = HashMap::new();
    let mut sig = Vec::new();
    let mut globals = Vec::new();
    let mut loads = Vec::new();
    for (k, prm) in params.iter().enumerate() {
        let shape = prm.shape.as_ref().ok_or_else(|| no_shape(prm))?;
        let ty = tensor_type(shape)?;
        let v = V {
            name: format!("%arg{k}"),
            ty: ty.clone(),
        };
        match (&params_src, k >= first_weight) {
            (ParamSource::Named { scope }, true) => {
                let name = arg_names
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| format!("param{k}"));
                let sym = quoted(&format!("__rlx_param.{name}"));
                globals.push(format!(
                    "util.global private @{sym} = #flow.parameter.named<{}::{}> : {ty}",
                    quoted(scope),
                    quoted(&name)
                ));
                loads.push(format!("{} = util.global.load @{sym} : {ty}", v.name));
            }
            _ => sig.push(format!("{}: {ty}", v.name)),
        }
        vals.insert(prm.id, Val::One(v));
    }

    // Body first, so the result types are known for the signature.
    let mut body = Printer {
        comps: p.comps.clone(),
        out: String::new(),
    };
    for l in &loads {
        body.line(2, l);
    }
    body.body(entry, &mut vals, 2)?;
    let root = vals
        .get(&entry.root_id)
        .ok_or_else(|| ExportError::Unsupported("entry root not found".into()))?
        .flat();
    body.line(2, &return_op("func.return", &root));

    let mut header = vec![format!("// @{func_name} arguments, in order:")];
    for (k, n) in arg_names.iter().enumerate() {
        if matches!(params_src, ParamSource::Named { .. }) && k >= first_weight {
            header.push(format!("//   (parameter) {n}"));
        } else {
            header.push(format!("//   %arg{k} = {n}"));
        }
    }
    let res: Vec<&str> = root.iter().map(|v| v.ty.as_str()).collect();
    let res = if res.len() == 1 {
        res[0].to_string()
    } else {
        format!("({})", res.join(", "))
    };
    p.line(
        1,
        &format!("func.func @{func_name}({}) -> {res} {{", sig.join(", ")),
    );
    p.out.push_str(&body.out);
    p.line(1, "}");
    Ok(Printed {
        header,
        globals,
        func: p.out,
    })
}

/// One printed function and what the module around it needs.
pub(crate) struct Printed {
    pub header: Vec<String>,
    /// `util.global` declarations (parameter mode); identical text for the
    /// same parameter, so functions sharing weights share them.
    pub globals: Vec<String>,
    pub func: String,
}

/// Assemble printed functions into one module, declaring each global once.
pub(crate) fn assemble(funcs: &[Printed]) -> String {
    let mut out = String::from("// Exported by rlx-stablehlo.\n");
    for f in funcs {
        for h in &f.header {
            out.push_str(h);
            out.push('\n');
        }
    }
    out.push_str("module {\n");
    let mut seen = std::collections::HashSet::new();
    for f in funcs {
        for g in &f.globals {
            if seen.insert(g.as_str()) {
                out.push_str("  ");
                out.push_str(g);
                out.push('\n');
            }
        }
    }
    for f in funcs {
        out.push_str(&f.func);
    }
    out.push_str("}\n");
    out
}
