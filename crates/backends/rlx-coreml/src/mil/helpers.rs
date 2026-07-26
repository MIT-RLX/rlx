// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Proto-plumbing layer for the MIL lowering: pure builder functions that
// construct CoreML protobuf messages (consts, bindings, scalar/vector
// immediate values, tensor/value types) plus small shape/dtype/quant
// helpers. No `LowerCtx` state — split out of `mod.rs` to keep the
// lowering dispatch readable.

use std::collections::HashMap;

use rlx_ir::op::BinaryOp;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Shape};

use super::IoTensor;
use crate::proto;
use crate::{CoremlError, Result};

pub(super) fn binary_mil(b: BinaryOp) -> &'static str {
    match b {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "real_div",
        BinaryOp::Max => "maximum",
        BinaryOp::Min => "minimum",
        BinaryOp::Pow => "pow",
        BinaryOp::Mod => "mod",
        // MIL has a unary `atan` but no binary `atan2` op — unsupported on ANE.
        BinaryOp::Atan2 => panic!("rlx-coreml: atan2 unsupported on ANE (no MIL atan2 op)"),
        // ANE is float-only — no integer bitwise ops in MIL.
        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr => {
            panic!("rlx-coreml: bitwise ops unsupported on ANE (integer-only)")
        }
    }
}

/// MIL `DataType` (proto enum value) for an RLX dtype.
///
/// CoreML/MIL has no f64, bf16, int16, or complex storage types, so the
/// dtypes RLX carries but MIL lacks are demoted/widened to the nearest MIL
/// type here (matching `mil_cast_dtype`, so a `Cast`'s `dtype` attribute and
/// its output value type always agree):
///   * `F64  → fp32`  (demote — no double precision on CoreML)
///   * `BF16 → fp16`  (nearest float — no bf16)
///   * `I16  → int32` (widen — no int16)
/// `C64` (complex) has no MIL analogue at all and is rejected explicitly.
/// Integer dtypes normally never reach here: `promote_int_to_f32` rewrites
/// them to f32 before lowering; the int arms are the fallback for the direct
/// `lower_graph` path (tests) that skips promotion.
pub(super) fn mil_data_type(dt: DType) -> Result<i32> {
    let v = match dt {
        DType::F32 => proto::DataType::Float32,
        DType::F16 => proto::DataType::Float16,
        // No f64 on CoreML — demote to single precision.
        DType::F64 => proto::DataType::Float32,
        // No bf16 on CoreML — fp16 is the nearest supported float.
        DType::BF16 => proto::DataType::Float16,
        DType::I32 => proto::DataType::Int32,
        DType::I64 => proto::DataType::Int64,
        // No int16 on CoreML — widen to int32.
        DType::I16 => proto::DataType::Int32,
        DType::I8 => proto::DataType::Int8,
        DType::U8 => proto::DataType::Uint8,
        DType::Bool => proto::DataType::Bool,
        DType::C64 => {
            return Err(CoremlError::Unsupported(
                "C64 (complex) has no CoreML/MIL data type".to_string(),
            ));
        }
        other => {
            return Err(CoremlError::Unsupported(format!("dtype {other:?}")));
        }
    };
    Ok(v as i32)
}

/// Bake the duplicated-per-pair `[seq, hd]` cos/sin tables for axial
/// RoPE, mirroring the CPU reference (`rlx_ir::ops::axial_rope2d`).
#[allow(clippy::too_many_arguments)]
pub(super) fn axial_tables(
    end_x: usize,
    _end_y: usize,
    head_dim: usize,
    num_heads: usize,
    theta: f32,
    repeat_factor: usize,
    seq: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let q4 = head_dim / 4;
    let repeat = repeat_factor.max(1);
    let freqs: Vec<f32> = (0..q4)
        .map(|i| 1.0 / theta.powf((4 * i) as f32 / head_dim as f32))
        .collect();

    let mut cos = vec![1.0f32; seq * hd];
    let mut sin = vec![0.0f32; seq * hd];
    for tok in 0..seq {
        let pos = tok / repeat;
        let tx = (pos % end_x) as f32;
        let ty = (pos / end_x) as f32;
        for h in 0..num_heads {
            let hbase = h * head_dim;
            for c in 0..q4 {
                let (ax, ay) = (tx * freqs[c], ty * freqs[c]);
                let (cx, sx) = (ax.cos(), ax.sin());
                let (cy, sy) = (ay.cos(), ay.sin());
                for d in [2 * c, 2 * c + 1] {
                    let ix = tok * hd + hbase + d;
                    cos[ix] = cx;
                    sin[ix] = sx;
                    let iy = tok * hd + hbase + half + d;
                    cos[iy] = cy;
                    sin[iy] = sy;
                }
            }
        }
    }
    (cos, sin)
}

/// Host-side GGUF dequant dispatch → `n` f32 values (same element order as
/// the packed bytes). Covers the legacy + K-quant schemes; IQ / ternary /
/// microscaling fall through as unsupported for now.
pub(super) fn dequant_scheme(scheme: QuantScheme, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    use QuantScheme::*;
    let r = match scheme {
        GgufQ8_0 => rlx_gguf::dequant_q8_0(bytes, n),
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
        GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(bytes, n),
        GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(bytes, n),
        other => {
            return Err(CoremlError::Unsupported(format!(
                "GGUF dequant scheme {other:?}"
            )));
        }
    };
    r.map_err(|e| CoremlError::Runtime(format!("gguf dequant: {e}")))
}

pub(super) fn vec_usize_i32(xs: &[usize]) -> Vec<i32> {
    xs.iter().map(|&x| x as i32).collect()
}

/// Symmetric `[h, w]` padding → MIL custom pad `[h_begin, h_end, w_begin,
/// w_end]`.
pub(super) fn pad_begin_end(padding: &[usize]) -> Vec<i32> {
    let mut out = Vec::with_capacity(padding.len() * 2);
    for &p in padding {
        out.push(p as i32);
        out.push(p as i32);
    }
    out
}

/// MIL `cast` dtype string for an RLX dtype.
///
/// Dtypes MIL lacks are demoted/widened to the nearest supported MIL type,
/// kept in lock-step with [`mil_data_type`] so a `Cast`'s `dtype` attribute
/// and its output value type never disagree:
///   * `F64  → fp32`  (demote — CoreML has no double precision)
///   * `BF16 → fp16`  (nearest float — CoreML has no bf16)
///   * `I16  → int32` (widen — CoreML has no int16)
/// `C64` (complex) has no MIL analogue and is rejected with a clear typed
/// error rather than silently corrupting the interleaved re/im layout.
pub(super) fn mil_cast_dtype(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        // No f64 on CoreML — demote to single precision.
        DType::F64 => "fp32",
        // No bf16 on CoreML — fp16 is the nearest supported float.
        DType::BF16 => "fp16",
        DType::I32 => "int32",
        // No int16 on CoreML — widen to int32.
        DType::I16 => "int32",
        DType::I8 => "int8",
        DType::U8 => "uint8",
        DType::Bool => "bool",
        DType::C64 => {
            return Err(CoremlError::Unsupported(
                "cast to C64 (complex): CoreML has no complex dtype".to_string(),
            ));
        }
        other => return Err(CoremlError::Unsupported(format!("cast to {other:?}"))),
    })
}

/// Copy `shape` with its last dimension replaced by `n`.
pub(super) fn with_last(shape: &Shape, n: usize) -> Shape {
    let mut dims = shape.dims().to_vec();
    let last = dims.len() - 1;
    dims[last] = Dim::Static(n);
    Shape::from_dims(&dims, shape.dtype())
}

/// Static size of `shape`'s axis `i`, or an error if dynamic.
pub(super) fn dim_static(shape: &Shape, i: usize) -> Result<usize> {
    match shape.dim(i) {
        Dim::Static(n) => Ok(n),
        Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("axis {i} = ?{s}"))),
    }
}

/// Row-major `[s_q, s_k]` additive causal mask: 0 on/below the diagonal,
/// `-1e9` above (prefill, query offset 0 — matches the CPU reference).
pub(super) fn causal_mask(s_q: usize, s_k: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; s_q * s_k];
    for qi in 0..s_q {
        for ki in (qi + 1)..s_k {
            m[qi * s_k + ki] = -1e9;
        }
    }
    m
}

/// Row-major `[s_q, s_k]` additive sliding-window mask: 0 where causal and
/// `qi - ki <= window`, `-1e9` elsewhere (matches CPU / MLX / Metal).
pub(super) fn sliding_window_mask(s_q: usize, s_k: usize, window: usize) -> Vec<f32> {
    let mut m = vec![-1e9f32; s_q * s_k];
    let w = window as i64;
    for qi in 0..s_q {
        for ki in 0..s_k {
            let q = qi as i64;
            let k = ki as i64;
            if k <= q && (q - k) <= w {
                m[qi * s_k + ki] = 0.0;
            }
        }
    }
    m
}

/// Shape after a keep-dims reduction over the trailing dims from `axis`:
/// reduced axes become size 1.
pub(super) fn reduced_shape(shape: &Shape, axis: usize) -> Shape {
    let dims: Vec<Dim> = shape
        .dims()
        .iter()
        .enumerate()
        .map(|(i, d)| if i >= axis { Dim::Static(1) } else { *d })
        .collect();
    Shape::from_dims(&dims, shape.dtype())
}

pub(super) fn static_dims(shape: &Shape) -> Result<Vec<i64>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Ok(*n as i64),
            Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("symbol {s}"))),
        })
        .collect()
}

/// Static dims for I/O, or `-1` placeholders when `flexible` and dynamic.
pub(super) fn io_dims(shape: &Shape, flexible: bool) -> Result<(Vec<i64>, Vec<bool>)> {
    let mut dims = Vec::new();
    let mut flex = Vec::new();
    for d in shape.dims() {
        match d {
            Dim::Static(n) => {
                dims.push(*n as i64);
                flex.push(false);
            }
            Dim::Dynamic(_) if flexible => {
                dims.push(-1);
                flex.push(true);
            }
            Dim::Dynamic(s) => {
                return Err(CoremlError::DynamicShape(format!("symbol {s}")));
            }
        }
    }
    Ok((dims, flex))
}

pub(super) fn tensor_type_flex(shape: &Shape, flex_dims: &[bool]) -> Result<proto::TensorType> {
    let dims = shape
        .dims()
        .iter()
        .zip(flex_dims.iter())
        .map(|(d, &flex)| {
            if flex {
                Ok(proto::Dimension {
                    dimension: Some(proto::dimension::Dimension::Unknown(
                        proto::dimension::UnknownDimension { variadic: false },
                    )),
                })
            } else {
                match d {
                    Dim::Static(n) => Ok(proto::Dimension {
                        dimension: Some(proto::dimension::Dimension::Constant(
                            proto::dimension::ConstantDimension { size: *n as u64 },
                        )),
                    }),
                    Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("symbol {s}"))),
                }
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(proto::TensorType {
        data_type: mil_data_type(shape.dtype())?,
        rank: shape.rank() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    })
}

pub(super) fn tensor_type(shape: &Shape) -> Result<proto::TensorType> {
    let dims = shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Ok(proto::Dimension {
                dimension: Some(proto::dimension::Dimension::Constant(
                    proto::dimension::ConstantDimension { size: *n as u64 },
                )),
            }),
            Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("symbol {s}"))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(proto::TensorType {
        data_type: mil_data_type(shape.dtype())?,
        rank: shape.rank() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    })
}

pub(super) fn value_type(shape: &Shape) -> Result<proto::ValueType> {
    Ok(proto::ValueType {
        r#type: Some(proto::value_type::Type::TensorType(tensor_type(shape)?)),
    })
}

pub(super) fn named_value_type(name: &str, shape: &Shape) -> Result<proto::NamedValueType> {
    Ok(proto::NamedValueType {
        name: name.to_string(),
        r#type: Some(value_type(shape)?),
    })
}

pub(super) fn named_value_type_flex(
    name: &str,
    shape: &Shape,
    flex_dims: &[bool],
) -> Result<proto::NamedValueType> {
    Ok(proto::NamedValueType {
        name: name.to_string(),
        r#type: Some(proto::ValueType {
            r#type: Some(proto::value_type::Type::TensorType(tensor_type_flex(
                shape, flex_dims,
            )?)),
        }),
    })
}

/// Build a single-output op with the given input bindings. Sets the op's
/// `name` attribute and the output `NamedValueType` (so the value carries
/// its tensor type, which MIL validation needs).
#[allow(dead_code)]
pub(super) fn simple_op(
    ty: &str,
    out_name: &str,
    out_shape: &Shape,
    inputs: Vec<(&str, proto::Argument)>,
) -> Result<proto::Operation> {
    simple_op_flex(ty, out_name, out_shape, inputs, false)
}

pub(super) fn simple_op_flex(
    ty: &str,
    out_name: &str,
    out_shape: &Shape,
    inputs: Vec<(&str, proto::Argument)>,
    flexible: bool,
) -> Result<proto::Operation> {
    let mut input_map = HashMap::new();
    for (k, v) in inputs {
        input_map.insert(k.to_string(), v);
    }
    let mut attributes = HashMap::new();
    attributes.insert("name".to_string(), scalar_str(out_name));
    let flex_mask: Vec<bool> = if flexible {
        out_shape
            .dims()
            .iter()
            .map(|d| matches!(d, Dim::Dynamic(_)))
            .collect()
    } else {
        vec![false; out_shape.rank()]
    };
    let out_ty = if flexible && flex_mask.iter().any(|&f| f) {
        named_value_type_flex(out_name, out_shape, &flex_mask)?
    } else {
        named_value_type(out_name, out_shape)?
    };
    Ok(proto::Operation {
        r#type: ty.to_string(),
        inputs: input_map,
        outputs: vec![out_ty],
        blocks: vec![],
        attributes,
    })
}

/// A `const` op baking `data` (f32) as an inline immediate tensor.
/// Constants ≥ this many elements go to the `weight.bin` blob; smaller
/// ones stay inline (matches coremltools' threshold).
pub(super) const BLOB_MIN_ELEMS: usize = 10;

/// Build a `const` op, routing large f32 data into `blob` (referenced by
/// offset) and small data inline.
pub(super) fn make_const(
    blob: &mut crate::mlpackage::BlobWriter,
    out_name: &str,
    shape: &Shape,
    data: &[f32],
) -> Result<proto::Operation> {
    let expected = shape.num_elements().unwrap_or(0);
    if expected != data.len() {
        return Err(CoremlError::Runtime(format!(
            "const '{out_name}': shape wants {expected} elems, got {}",
            data.len()
        )));
    }
    let val = if shape.dtype() == DType::Bool {
        // CoreML's weight blob has no bool storage ("bool is not a supported data
        // type for blob file values"), so bake bool constants as an inline bool
        // immediate regardless of size. VITS attention/sequence masks hit this.
        let t = proto::TensorValue {
            value: Some(proto::tensor_value::Value::Bools(
                proto::tensor_value::RepeatedBools {
                    values: data.iter().map(|&x| x != 0.0).collect(),
                },
            )),
        };
        immediate(t, value_type(shape)?)
    } else if shape.dtype() == DType::F16 {
        // F16-typed consts MUST carry f16 storage, else CoreML rejects the model
        // ("Tensor storage and type have different number of elements"). Hit by
        // mixed-precision (AutoMixedPrecision) graphs: internal consts the lowering
        // synthesizes for an f16 op (e.g. `Expand`'s ones) need f16 payloads.
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        if f16.len() >= BLOB_MIN_ELEMS {
            let offset = blob.write_f16(&f16);
            proto::Value {
                doc_string: String::new(),
                r#type: Some(value_type(shape)?),
                value: Some(proto::value::Value::BlobFileValue(
                    proto::value::BlobFileValue {
                        file_name: "@model_path/weights/weight.bin".to_string(),
                        offset,
                    },
                )),
            }
        } else {
            let bytes: Vec<u8> = f16.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
            immediate(
                proto::TensorValue {
                    value: Some(proto::tensor_value::Value::Bytes(
                        proto::tensor_value::RepeatedBytes { values: bytes },
                    )),
                },
                value_type(shape)?,
            )
        }
    } else if data.len() >= BLOB_MIN_ELEMS {
        let offset = blob.write_f32(data);
        proto::Value {
            doc_string: String::new(),
            r#type: Some(value_type(shape)?),
            value: Some(proto::value::Value::BlobFileValue(
                proto::value::BlobFileValue {
                    file_name: "@model_path/weights/weight.bin".to_string(),
                    offset,
                },
            )),
        }
    } else {
        tensor_f32(shape, data)?
    };
    let mut attributes = HashMap::new();
    attributes.insert("name".to_string(), scalar_str(out_name));
    attributes.insert("val".to_string(), val);
    Ok(proto::Operation {
        r#type: "const".to_string(),
        inputs: HashMap::new(),
        outputs: vec![named_value_type(out_name, shape)?],
        blocks: vec![],
        attributes,
    })
}

/// TensorType with an explicit proto DataType — for sub-byte types (UInt1/…)
/// that `rlx_ir::DType` can't name. All-static dims.
fn tensor_type_dt(dims: &[usize], dt: proto::DataType) -> proto::TensorType {
    proto::TensorType {
        data_type: dt as i32,
        rank: dims.len() as i64,
        dimensions: dims
            .iter()
            .map(|&n| proto::Dimension {
                dimension: Some(proto::dimension::Dimension::Constant(
                    proto::dimension::ConstantDimension { size: n as u64 },
                )),
            })
            .collect(),
        attributes: HashMap::new(),
    }
}

/// Build a UINT1-typed const backed by a packed 1-bit blob (8 elems/byte,
/// LSB-first). `dims` is the LOGICAL (unpacked) shape; `packed` is
/// `ceil(num_elems / 8)` bytes. Feeds `constexpr_lut_to_dense` palettization
/// indices — the Q1_0 no-unfold path (keeps 1-bit storage, ~n·k/8 bytes).
pub(super) fn make_const_u1_packed(
    blob: &mut crate::mlpackage::BlobWriter,
    out_name: &str,
    dims: &[usize],
    packed: &[u8],
) -> Result<proto::Operation> {
    let n_elems: usize = dims.iter().product();
    let need = n_elems.div_ceil(8);
    if packed.len() != need {
        return Err(CoremlError::Runtime(format!(
            "const '{out_name}': u1 wants {need} packed bytes for {n_elems} elems, got {}",
            packed.len()
        )));
    }
    let offset = blob.write_u1_packed(packed);
    let tt = tensor_type_dt(dims, proto::DataType::Uint1);
    let vt = proto::ValueType {
        r#type: Some(proto::value_type::Type::TensorType(tt)),
    };
    let val = proto::Value {
        doc_string: String::new(),
        r#type: Some(vt.clone()),
        value: Some(proto::value::Value::BlobFileValue(
            proto::value::BlobFileValue {
                file_name: "@model_path/weights/weight.bin".to_string(),
                offset,
            },
        )),
    };
    let mut attributes = HashMap::new();
    attributes.insert("name".to_string(), scalar_str(out_name));
    attributes.insert("val".to_string(), val);
    Ok(proto::Operation {
        r#type: "const".to_string(),
        inputs: HashMap::new(),
        outputs: vec![proto::NamedValueType {
            name: out_name.to_string(),
            r#type: Some(vt),
        }],
        blocks: vec![],
        attributes,
    })
}

/// Build a float `const` op (f32 or f16 blob / inline).
pub(super) fn make_const_float(
    blob: &mut crate::mlpackage::BlobWriter,
    out_name: &str,
    shape: &Shape,
    data: &[f32],
    float_dtype: DType,
) -> Result<proto::Operation> {
    if float_dtype == DType::F16 && shape.dtype() == DType::F16 {
        let expected = shape.num_elements().unwrap_or(0);
        if expected != data.len() {
            return Err(CoremlError::Runtime(format!(
                "const '{out_name}': shape wants {expected} elems, got {}",
                data.len()
            )));
        }
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        let val = if f16.len() >= BLOB_MIN_ELEMS {
            let offset = blob.write_f16(&f16);
            proto::Value {
                doc_string: String::new(),
                r#type: Some(value_type(shape)?),
                value: Some(proto::value::Value::BlobFileValue(
                    proto::value::BlobFileValue {
                        file_name: "@model_path/weights/weight.bin".to_string(),
                        offset,
                    },
                )),
            }
        } else {
            // Small fp16 immediates must go in the `bytes` field as raw LE f16,
            // NOT the f32 `floats` field. Declaring an F16 type with f32 `floats`
            // storage makes CoreML reject the model at load with "Tensor storage
            // and type have different number of elements". (Hit by per-scalar
            // params like the learned butterfly twiddles.)
            let mut bytes = Vec::with_capacity(f16.len() * 2);
            for h in &f16 {
                bytes.extend_from_slice(&h.to_bits().to_le_bytes());
            }
            let t = proto::TensorValue {
                value: Some(proto::tensor_value::Value::Bytes(
                    proto::tensor_value::RepeatedBytes { values: bytes },
                )),
            };
            immediate(t, value_type(shape)?)
        };
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), scalar_str(out_name));
        attributes.insert("val".to_string(), val);
        return Ok(proto::Operation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![named_value_type(out_name, shape)?],
            blocks: vec![],
            attributes,
        });
    }
    make_const(blob, out_name, shape, data)
}

/// True when on-device block dequant (`mul` + optional `sub`) applies.
///
/// Q2/Q3/Q6_K use per-element `[nb,32]` scale/offset tensors; others use
/// `[nb,1]` broadcast. See [docs/gguf-backend-paths.md](../../../../docs/gguf-backend-paths.md).
pub(super) fn scheme_supports_ondevice_block_dequant(scheme: QuantScheme) -> bool {
    matches!(
        scheme,
        QuantScheme::GgufQ8_0
            | QuantScheme::GgufQ4_0
            | QuantScheme::GgufQ4_1
            | QuantScheme::GgufQ5_0
            | QuantScheme::GgufQ5_1
            | QuantScheme::GgufIQ4NL
            | QuantScheme::GgufIQ4XS
            | QuantScheme::GgufTQ1_0
            | QuantScheme::GgufTQ2_0
            | QuantScheme::GgufMXFP4
            | QuantScheme::GgufNVFP4
            | QuantScheme::GgufIQ2XXS
            | QuantScheme::GgufIQ2XS
            | QuantScheme::GgufIQ2S
            | QuantScheme::GgufIQ3XXS
            | QuantScheme::GgufIQ3S
            | QuantScheme::GgufIQ1S
            | QuantScheme::GgufIQ1M
            | QuantScheme::GgufQ4K
            | QuantScheme::GgufQ5K
            | QuantScheme::GgufQ8K
            | QuantScheme::GgufQ2K
            | QuantScheme::GgufQ3K
            | QuantScheme::GgufQ6K
            | QuantScheme::GgufQ1_0
    )
}

use rlx_gguf::iq_grids::KVALUES_IQ4NL;
use rlx_gguf::iq_grids::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS,
};
const QK: usize = 32;
const QK_K: usize = 256;
const QK_NVFP4: usize = 16;
const K_SCALE_SIZE: usize = 12;
const IQ1S_DELTA: f32 = 0.125;

fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn grid_u64_to_i8x8(entry: u64) -> [i8; 8] {
    let bytes = entry.to_le_bytes();
    [
        bytes[0] as i8,
        bytes[1] as i8,
        bytes[2] as i8,
        bytes[3] as i8,
        bytes[4] as i8,
        bytes[5] as i8,
        bytes[6] as i8,
        bytes[7] as i8,
    ]
}

fn grid_u32_to_i8x4(entry: u32) -> [i8; 4] {
    let bytes = entry.to_le_bytes();
    [
        bytes[0] as i8,
        bytes[1] as i8,
        bytes[2] as i8,
        bytes[3] as i8,
    ]
}

pub(super) fn read_f16_le(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

fn k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Split GGUF blocks into integer-valued f32 quants `[nb,32]`, scales, and
/// offsets for MIL `mul` + optional `sub` dequant.
///
/// Scale/offset layout:
/// - `[nb]` or `[nb,1]` — uniform per 32-element chunk (Q4_0, Q4_K, Q4_1, …)
/// - `[nb,32]` — per-element within chunk (Q2_K, Q3_K, Q6_K sub-block scales)
/// `bake_ondevice_weight` picks the MIL tensor shape from `scales.len()`.
pub(super) fn split_gguf_ondevice(
    scheme: QuantScheme,
    bytes: &[u8],
    nb: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    match scheme {
        QuantScheme::GgufQ8_0 | QuantScheme::GgufQ4_0 | QuantScheme::GgufIQ4NL => {
            let (qs, scales) = split_gguf_blocks(scheme, bytes, nb)?;
            let offsets = vec![0f32; nb];
            Ok((qs, scales, offsets))
        }
        QuantScheme::GgufQ4_1 => split_q4_1_ondevice(bytes, nb),
        QuantScheme::GgufQ5_0 => split_q5_0_ondevice(bytes, nb),
        QuantScheme::GgufQ5_1 => split_q5_1_ondevice(bytes, nb),
        QuantScheme::GgufQ4K => split_q4_k_ondevice(bytes, nb),
        QuantScheme::GgufQ5K => split_q5_k_ondevice(bytes, nb),
        QuantScheme::GgufQ8K => split_q8_k_ondevice(bytes, nb),
        QuantScheme::GgufQ2K => split_q2_k_ondevice(bytes, nb),
        QuantScheme::GgufQ3K => split_q3_k_ondevice(bytes, nb),
        QuantScheme::GgufQ6K => split_q6_k_ondevice(bytes, nb),
        QuantScheme::GgufIQ4XS => split_iq4_xs_ondevice(bytes, nb),
        QuantScheme::GgufTQ1_0 => split_tq1_0_ondevice(bytes, nb),
        QuantScheme::GgufTQ2_0 => split_tq2_0_ondevice(bytes, nb),
        QuantScheme::GgufMXFP4 => split_mxfp4_ondevice(bytes, nb),
        QuantScheme::GgufNVFP4 => split_nvfp4_ondevice(bytes, nb),
        QuantScheme::GgufIQ2XXS => split_iq2_xxs_ondevice(bytes, nb),
        QuantScheme::GgufIQ2XS => split_iq2_xs_ondevice(bytes, nb),
        QuantScheme::GgufIQ2S => split_iq2_s_ondevice(bytes, nb),
        QuantScheme::GgufIQ3XXS => split_iq3_xxs_ondevice(bytes, nb),
        QuantScheme::GgufIQ3S => split_iq3_s_ondevice(bytes, nb),
        QuantScheme::GgufIQ1S => split_iq1_s_ondevice(bytes, nb),
        QuantScheme::GgufIQ1M => split_iq1_m_ondevice(bytes, nb),
        QuantScheme::GgufQ1_0 => split_q1_0_ondevice(bytes, nb),
        other => Err(CoremlError::Unsupported(format!(
            "split_gguf_ondevice: {other:?}"
        ))),
    }
}

/// Q1_0 (prism-ml Bonsai): `{f16 d; u8 qs[16]}` per 128 elems — bit=1 → +d,
/// bit=0 → −d. Expressed as the affine form the on-device MIL dequant expects
/// (`value = qs·scale + offset`, QK=32 granularity): qs ∈ {−1,+1} (the sign),
/// scale = the block's `d` (shared across its four 32-elem chunks), offset = 0.
fn split_q1_0_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 18; // 2 (f16 d) + 16 (128 sign bits)
    const CHUNKS_PER_BLOCK: usize = 128 / QK; // 4
    let n_blocks = nb / CHUNKS_PER_BLOCK;
    if !nb.is_multiple_of(CHUNKS_PER_BLOCK) || bytes.len() < n_blocks * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q1_0 ondevice: {nb} chunks / {} bytes inconsistent",
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for c in 0..nb {
        let block = c / CHUNKS_PER_BLOCK;
        let chunk_in_block = c % CHUNKS_PER_BLOCK;
        let base = block * BLOCK;
        let d = read_f16_le(&bytes[base..base + 2]);
        scales[c] = d;
        for i in 0..QK {
            let elem = chunk_in_block * QK + i; // 0..127 within the 128-block
            let bit = (bytes[base + 2 + elem / 8] >> (elem % 8)) & 1;
            qs[c * QK + i] = if bit == 1 { 1.0 } else { -1.0 };
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q4_1_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 20;
    if bytes.len() != nb * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q4_1 ondevice: expected {} bytes, got {}",
            nb * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb];
    for i in 0..nb {
        let off = i * BLOCK;
        let d = read_f16_le(&bytes[off..off + 2]);
        let m = read_f16_le(&bytes[off + 2..off + 4]);
        scales[i] = d;
        offsets[i] = -m;
        let qbytes = &bytes[off + 4..off + 4 + QK / 2];
        for j in 0..QK / 2 {
            qs[i * QK + j] = (qbytes[j] & 0x0F) as f32;
            qs[i * QK + QK / 2 + j] = (qbytes[j] >> 4) as f32;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q5_0_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 22;
    if bytes.len() != nb * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q5_0 ondevice: expected {} bytes, got {}",
            nb * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb];
    for i in 0..nb {
        let off = i * BLOCK;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qh = u32::from_le_bytes([
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
        ]);
        scales[i] = d;
        offsets[i] = 16.0 * d;
        let qbytes = &bytes[off + 6..off + 6 + QK / 2];
        for j in 0..QK / 2 {
            let xh0 = (((qh >> j) & 0x01) as u8) << 4;
            qs[i * QK + j] = ((qbytes[j] & 0x0F) | xh0) as f32;
            let xh1 = (((qh >> (j + 16)) & 0x01) as u8) << 4;
            qs[i * QK + QK / 2 + j] = ((qbytes[j] >> 4) | xh1) as f32;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q5_1_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 24;
    if bytes.len() != nb * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q5_1 ondevice: expected {} bytes, got {}",
            nb * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb];
    for i in 0..nb {
        let off = i * BLOCK;
        let d = read_f16_le(&bytes[off..off + 2]);
        let m = read_f16_le(&bytes[off + 2..off + 4]);
        let qh = u32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        scales[i] = d;
        offsets[i] = -m;
        let qbytes = &bytes[off + 8..off + 8 + QK / 2];
        for j in 0..QK / 2 {
            let xh0 = (((qh >> j) & 0x01) as u8) << 4;
            qs[i * QK + j] = ((qbytes[j] & 0x0F) | xh0) as f32;
            let xh1 = (((qh >> (j + 16)) & 0x01) as u8) << 4;
            qs[i * QK + QK / 2 + j] = ((qbytes[j] >> 4) | xh1) as f32;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q2_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 84;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q2_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q2_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let mut offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let scales_off = 0;
        let qs_off = QK_K / 16;
        let d_off = qs_off + QK_K / 4;
        let d = read_f16_le(&block[d_off..d_off + 2]);
        let min = read_f16_le(&block[d_off + 2..d_off + 4]);
        let scales_b = &block[scales_off..scales_off + QK_K / 16];
        let mut q = &block[qs_off..qs_off + QK_K / 4];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut is = 0usize;
        let mut out_i = 0usize;
        for _ in 0..(QK_K / 128) {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales_b[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let chunk = base_chunk + out_i / QK;
                    let pos = out_i % QK;
                    qs[chunk * QK + pos] = ((q[l] >> shift) & 3) as f32;
                    scales[chunk * QK + pos] = dl;
                    offsets[chunk * QK + pos] = ml;
                    out_i += 1;
                }
                let sc = scales_b[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let chunk = base_chunk + out_i / QK;
                    let pos = out_i % QK;
                    qs[chunk * QK + pos] = ((q[l + 16] >> shift) & 3) as f32;
                    scales[chunk * QK + pos] = dl;
                    offsets[chunk * QK + pos] = ml;
                    out_i += 1;
                }
                shift += 2;
            }
            q = &q[32..];
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q3_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 110;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q3_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q3_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let mut offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let hm_off = 0;
        let qs_off = QK_K / 8;
        let scales_off = qs_off + QK_K / 4;
        let d_off = scales_off + K_SCALE_SIZE;
        let d_all = read_f16_le(&block[d_off..d_off + 2]);
        let hm = &block[hm_off..hm_off + QK_K / 8];
        let mut q = &block[qs_off..qs_off + QK_K / 4];
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes(block[scales_off..scales_off + 4].try_into().unwrap());
        aux[1] = u32::from_le_bytes(block[scales_off + 4..scales_off + 8].try_into().unwrap());
        aux[2] = u32::from_le_bytes(block[scales_off + 8..scales_off + 12].try_into().unwrap());
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales_b: &[i8; 16] = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut out_i = 0usize;
        for _ in 0..(QK_K / 128) {
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * (scales_b[is] - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let h = if hm[l] & m != 0 { 0.0 } else { 4.0 };
                    let chunk = base_chunk + out_i / QK;
                    let pos = out_i % QK;
                    qs[chunk * QK + pos] = ((q[l] >> shift) & 3) as f32;
                    scales[chunk * QK + pos] = dl;
                    offsets[chunk * QK + pos] = dl * h;
                    out_i += 1;
                }
                let dl = d_all * (scales_b[is] - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let h = if hm[l + 16] & m != 0 { 0.0 } else { 4.0 };
                    let chunk = base_chunk + out_i / QK;
                    let pos = out_i % QK;
                    qs[chunk * QK + pos] = ((q[l + 16] >> shift) & 3) as f32;
                    scales[chunk * QK + pos] = dl;
                    offsets[chunk * QK + pos] = dl * h;
                    out_i += 1;
                }
                shift += 2;
                m <<= 1;
            }
            q = &q[32..];
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q6_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 210;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q6_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q6_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let ql_len = QK_K / 2;
        let qh_len = QK_K / 4;
        let sc_len = QK_K / 16;
        let ql = &block[0..ql_len];
        let qh = &block[ql_len..ql_len + qh_len];
        let sc = &block[ql_len + qh_len..ql_len + qh_len + sc_len];
        let d = read_f16_le(&block[ql_len + qh_len + sc_len..]);
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut out_i = 0usize;
        for h in 0..2 {
            let ql_off = h * 64;
            let qh_off_h = h * 32;
            let sc_off = h * 8;
            for l in 0..32 {
                let is = l / 16;
                let qh_b = qh[qh_off_h + l];
                let quads = [
                    ((ql[ql_off + l] & 0x0F) | ((qh_b & 3) << 4)) as i32 - 32,
                    ((ql[ql_off + l + 32] & 0x0F) | (((qh_b >> 2) & 3) << 4)) as i32 - 32,
                    ((ql[ql_off + l] >> 4) | (((qh_b >> 4) & 3) << 4)) as i32 - 32,
                    ((ql[ql_off + l + 32] >> 4) | (((qh_b >> 6) & 3) << 4)) as i32 - 32,
                ];
                let sc_vals = [
                    sc[sc_off + is] as i8 as f32,
                    sc[sc_off + is + 2] as i8 as f32,
                    sc[sc_off + is + 4] as i8 as f32,
                    sc[sc_off + is + 6] as i8 as f32,
                ];
                for (q_val, sc_val) in quads.iter().zip(sc_vals.iter()) {
                    let chunk = base_chunk + out_i / QK;
                    let pos = out_i % QK;
                    qs[chunk * QK + pos] = *q_val as f32;
                    scales[chunk * QK + pos] = d * sc_val;
                    out_i += 1;
                }
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q4_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 144;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q4_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q4_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let dmin = read_f16_le(&block[2..4]);
        let sc = &block[4..4 + K_SCALE_SIZE];
        let qbytes = &block[4 + K_SCALE_SIZE..];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut is = 0usize;
        for j in (0..CHUNKS_PER_SUPER).step_by(2) {
            let (sc0, m0) = k_scale_min(j, sc);
            let (sc1, m1) = k_scale_min(j + 1, sc);
            let d0 = d * sc0 as f32;
            let m0f = dmin * m0 as f32;
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let c0 = base_chunk + j;
            let c1 = base_chunk + j + 1;
            scales[c0] = d0;
            offsets[c0] = m0f;
            scales[c1] = d1;
            offsets[c1] = m1f;
            for l in 0..QK {
                qs[c0 * QK + l] = (qbytes[is + l] & 0x0F) as f32;
            }
            for l in 0..QK {
                qs[c1 * QK + l] = (qbytes[is + l] >> 4) as f32;
            }
            is += QK;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q5_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 176;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q5_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q5_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let dmin = read_f16_le(&block[2..4]);
        let sc = &block[4..4 + K_SCALE_SIZE];
        let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
        let qbytes = &block[4 + K_SCALE_SIZE + QK_K / 8..];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..CHUNKS_PER_SUPER).step_by(2) {
            let (sc0, m0) = k_scale_min(j, sc);
            let (sc1, m1) = k_scale_min(j + 1, sc);
            let d0 = d * sc0 as f32;
            let m0f = dmin * m0 as f32;
            let d1 = d * sc1 as f32;
            let m1f = dmin * m1 as f32;
            let c0 = base_chunk + j;
            let c1 = base_chunk + j + 1;
            scales[c0] = d0;
            offsets[c0] = m0f;
            scales[c1] = d1;
            offsets[c1] = m1f;
            for l in 0..QK {
                let lo = qbytes[is + l] & 0x0F;
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                qs[c0 * QK + l] = (lo + hi) as f32;
            }
            for l in 0..QK {
                let lo = qbytes[is + l] >> 4;
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                qs[c1 * QK + l] = (lo + hi) as f32;
            }
            is += QK;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_q8_k_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 292;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "Q8_K ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "Q8_K ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = f32::from_le_bytes(block[0..4].try_into().unwrap());
        let qbytes = &block[4..4 + QK_K];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        for c in 0..CHUNKS_PER_SUPER {
            scales[base_chunk + c] = d;
            let off = c * QK;
            for l in 0..QK {
                qs[(base_chunk + c) * QK + l] = qbytes[off + l] as i8 as f32;
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn fp4_e2m1(nibble: u8) -> f32 {
    const LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    LUT[(nibble & 0x0F) as usize]
}

fn split_mxfp4_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 17;
    if bytes.len() != nb * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "MXFP4 ondevice: expected {} bytes, got {}",
            nb * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for i in 0..nb {
        let off = i * BLOCK;
        scales[i] = rlx_gguf::mx_dequant::e8m0_scale_to_f32(bytes[off]);
        for j in 0..QK / 2 {
            let bx = bytes[off + 1 + j];
            qs[i * QK + 2 * j] = fp4_e2m1(bx);
            qs[i * QK + 2 * j + 1] = fp4_e2m1(bx >> 4);
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq4_xs_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 136;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ4_XS ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ4_XS ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let scales_l_len = QK_K / 64;
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let scales_h = u16::from_le_bytes([block[2], block[3]]) as u32;
        let scales_l = &block[4..4 + scales_l_len];
        let qbytes = &block[4 + scales_l_len..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut qs_off = 0usize;
        for ib in 0..CHUNKS_PER_SUPER {
            let lo = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
            let hi = (scales_h >> (2 * ib)) & 0x3;
            let ls = (lo as i32) | ((hi as i32) << 4);
            scales[base_chunk + ib] = d * (ls - 32) as f32;
            let qbase = (base_chunk + ib) * QK;
            for j in 0..16 {
                let b = qbytes[qs_off + j];
                qs[qbase + j] = KVALUES_IQ4NL[(b & 0x0F) as usize] as f32;
                qs[qbase + j + 16] = KVALUES_IQ4NL[(b >> 4) as usize] as f32;
            }
            qs_off += 16;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_tq2_0_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 66;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "TQ2_0 ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "TQ2_0 ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let qs_b = &block[0..64];
        let d = read_f16_le(&block[64..66]);
        let base = sb * CHUNKS_PER_SUPER;
        for c in 0..CHUNKS_PER_SUPER {
            scales[base + c] = d;
        }
        let mut y = 0usize;
        let mut j = 0usize;
        while j < 64 {
            for l in 0..4 {
                for m in 0..32 {
                    let chunk = base + y / QK;
                    let idx = y % QK;
                    let q = ((qs_b[j + m] >> (l * 2)) & 0x3) as i32;
                    qs[chunk * QK + idx] = (q - 1) as f32;
                    y += 1;
                }
            }
            j += 32;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_tq1_0_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    const QS_LEN: usize = 48;
    const QH_LEN: usize = 4;
    const BLOCK: usize = 54;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "TQ1_0 ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "TQ1_0 ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let qs_b = &block[0..QS_LEN];
        let qh = &block[QS_LEN..QS_LEN + QH_LEN];
        let d = read_f16_le(&block[QS_LEN + QH_LEN..]);
        let base = sb * CHUNKS_PER_SUPER;
        for c in 0..CHUNKS_PER_SUPER {
            scales[base + c] = d;
        }
        let mut y = 0usize;
        let mut j = 0usize;
        while j < 32 {
            for n in 0..5 {
                for m in 0..32 {
                    let q = qs_b[j + m].wrapping_mul(POW3[n]);
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    let chunk = base + y / QK;
                    qs[chunk * QK + (y % QK)] = (xi - 1) as f32;
                    y += 1;
                }
            }
            j += 32;
        }
        while j < QS_LEN {
            for n in 0..5 {
                for m in 0..16 {
                    let q = qs_b[j + m].wrapping_mul(POW3[n]);
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    let chunk = base + y / QK;
                    qs[chunk * QK + (y % QK)] = (xi - 1) as f32;
                    y += 1;
                }
            }
            j += 16;
        }
        for n in 0..4 {
            for jh in 0..QH_LEN {
                let q = qh[jh].wrapping_mul(POW3[n]);
                let xi = ((q as u16 * 3) >> 8) as i32;
                let chunk = base + y / QK;
                qs[chunk * QK + (y % QK)] = (xi - 1) as f32;
                y += 1;
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn split_nvfp4_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 1 + QK_NVFP4 / 2;
    let nvfp4_nb = nb * 2;
    if bytes.len() != nvfp4_nb * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "NVFP4 ondevice: expected {} bytes, got {}",
            nvfp4_nb * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let offsets = vec![0f32; nb * QK];
    for i in 0..nb {
        for half in 0..2 {
            let bidx = i * 2 + half;
            let off = bidx * BLOCK;
            let s = rlx_gguf::mx_dequant::e4m3_scale_to_f32(bytes[off]);
            for j in 0..QK_NVFP4 / 2 {
                let bx = bytes[off + 1 + j];
                let pos = i * QK + half * QK_NVFP4 + 2 * j;
                qs[pos] = fp4_e2m1(bx);
                qs[pos + 1] = fp4_e2m1(bx >> 4);
                scales[pos] = s;
                scales[pos + 1] = s;
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq2_xxs_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 66;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ2_XXS ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ2_XXS ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut y = 0usize;
        for ib32 in 0..QK_K / 32 {
            let base = 8 * ib32;
            let aux32_0 = read_u32_le(&qs_b[base..base + 4]);
            let aux32_1 = read_u32_le(&qs_b[base + 4..base + 8]);
            let aux8 = aux32_0.to_le_bytes();
            let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
            let chunk = base_chunk + y / QK;
            scales[chunk] = db;
            for l in 0..4 {
                let grid = grid_u64_to_i8x8(IQ2XXS_GRID[aux8[l] as usize]);
                let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    qs[chunk * QK + (y % QK)] = grid[j] as f32 * sign;
                    y += 1;
                }
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq2_xs_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 74;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ2_XS ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ2_XS ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..2 + (QK_K / 8) * 2];
        let scales_b = &block[2 + (QK_K / 8) * 2..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut y = 0usize;
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (scales_b[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (scales_b[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let pos = (4 * ib32 + l) * 2;
                let q = u16::from_le_bytes([qs_b[pos], qs_b[pos + 1]]);
                let grid = grid_u64_to_i8x8(IQ2XS_GRID[(q & 511) as usize]);
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                let dl = if l / 2 == 0 { db0 } else { db1 };
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let chunk = base_chunk + y / QK;
                    let idx = y % QK;
                    qs[chunk * QK + idx] = grid[j] as f32 * sign;
                    scales[chunk * QK + idx] = dl;
                    y += 1;
                }
            }
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq2_s_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 82;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ2_S ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ2_S ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..2 + QK_K / 4];
        let qh = &block[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let scales_b = &block[2 + QK_K / 4 + QK_K / 32..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut y = 0usize;
        let mut qs_idx = 0usize;
        let mut signs_idx = QK_K / 8;
        for ib32 in 0..QK_K / 32 {
            let db0 = d * (0.5 + (scales_b[ib32] & 0xF) as f32) * 0.25;
            let db1 = d * (0.5 + (scales_b[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let dl = if l / 2 == 0 { db0 } else { db1 };
                let q = qs_b[qs_idx + l] as u16;
                let qh_b = qh[ib32] as u16;
                let idx = (q | ((qh_b << (8 - 2 * l)) & 0x300)) as usize;
                let grid = grid_u64_to_i8x8(IQ2S_GRID[idx]);
                let sign = qs_b[signs_idx + l];
                for j in 0..8 {
                    let s = if sign & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let chunk = base_chunk + y / QK;
                    let pos = y % QK;
                    qs[chunk * QK + pos] = grid[j] as f32 * s;
                    scales[chunk * QK + pos] = dl;
                    y += 1;
                }
            }
            qs_idx += 4;
            signs_idx += 4;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq3_xxs_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 98;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ3_XXS ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ3_XXS ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..2 + QK_K / 4];
        let sas = &block[2 + QK_K / 4..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut y = 0usize;
        let mut qs_idx = 0usize;
        for ib32 in 0..QK_K / 32 {
            let aux32 = read_u32_le(&sas[4 * ib32..4 * ib32 + 4]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            let chunk = base_chunk + y / QK;
            scales[chunk] = db;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = grid_u32_to_i8x4(IQ3XXS_GRID[qs_b[qs_idx + 2 * l] as usize]);
                let g2 = grid_u32_to_i8x4(IQ3XXS_GRID[qs_b[qs_idx + 2 * l + 1] as usize]);
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    qs[chunk * QK + (y % QK) + j] = g1[j] as f32 * s0;
                    qs[chunk * QK + (y % QK) + j + 4] = g2[j] as f32 * s1;
                }
                y += 8;
            }
            qs_idx += 8;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq3_s_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 110;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ3_S ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ3_S ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let offsets = vec![0f32; nb];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..2 + QK_K / 4];
        let qh = &block[2 + QK_K / 4..2 + QK_K / 4 + QK_K / 32];
        let signs_b = &block[2 + QK_K / 4 + QK_K / 32..2 + QK_K / 4 + QK_K / 32 + QK_K / 8];
        let scales_b = &block[2 + QK_K / 4 + QK_K / 32 + QK_K / 8..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut y = 0usize;
        let mut qs_walk = 0usize;
        let mut signs_walk = 0usize;
        let mut qh_walk = 0usize;
        for ib32 in (0..QK_K / 32).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (scales_b[ib32 / 2] & 0xF) as f32);
            let db2 = d * (1.0 + 2.0 * (scales_b[ib32 / 2] >> 4) as f32);
            for (db, qh_off) in [(db1, 0usize), (db2, 1usize)] {
                let chunk = base_chunk + y / QK;
                scales[chunk] = db;
                for l in 0..4 {
                    let g1 = grid_u32_to_i8x4(
                        IQ3S_GRID[(qs_b[qs_walk + 2 * l] as usize)
                            | (((qh[qh_walk + qh_off] as usize) << (8 - 2 * l)) & 256)],
                    );
                    let g2 = grid_u32_to_i8x4(
                        IQ3S_GRID[(qs_b[qs_walk + 2 * l + 1] as usize)
                            | (((qh[qh_walk + qh_off] as usize) << (7 - 2 * l)) & 256)],
                    );
                    let sign = signs_b[signs_walk + l];
                    for j in 0..4 {
                        let s0 = if sign & KMASK_IQ2XS[j] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        let s1 = if sign & KMASK_IQ2XS[j + 4] != 0 {
                            -1.0
                        } else {
                            1.0
                        };
                        qs[chunk * QK + (y % QK) + j] = g1[j] as f32 * s0;
                        qs[chunk * QK + (y % QK) + j + 4] = g2[j] as f32 * s1;
                    }
                    y += 8;
                }
                qs_walk += 8;
                signs_walk += 4;
            }
            qh_walk += 2;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq1_s_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 50;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ1_S ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ1_S ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    let mut offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let d = read_f16_le(&block[0..2]);
        let qs_b = &block[2..2 + QK_K / 8];
        let qh_bytes = &block[2 + QK_K / 8..BLOCK];
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut qs_idx = 0usize;
        for ib in 0..QK_K / 32 {
            let qh = read_u16_le(&qh_bytes[2 * ib..2 * ib + 2]);
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            let chunk = base_chunk + ib;
            scales[chunk] = dl;
            for l in 0..4 {
                let idx = (qs_b[qs_idx + l] as usize) | ((((qh >> (3 * l)) & 7) as usize) << 8);
                let grid = grid_u64_to_i8x8(IQ1S_GRID[idx]);
                for j in 0..8 {
                    qs[chunk * QK + l * 8 + j] = grid[j] as f32;
                    offsets[chunk * QK + l * 8 + j] = dl * delta;
                }
            }
            qs_idx += 4;
        }
    }
    Ok((qs, scales, offsets))
}

fn split_iq1_m_ondevice(bytes: &[u8], nb: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    const BLOCK: usize = 56;
    const CHUNKS_PER_SUPER: usize = QK_K / QK;
    if !nb.is_multiple_of(CHUNKS_PER_SUPER) {
        return Err(CoremlError::Runtime(format!(
            "IQ1_M ondevice: nb={nb} not divisible by {CHUNKS_PER_SUPER}"
        )));
    }
    let num_super = nb / CHUNKS_PER_SUPER;
    if bytes.len() != num_super * BLOCK {
        return Err(CoremlError::Runtime(format!(
            "IQ1_M ondevice: expected {} bytes, got {}",
            num_super * BLOCK,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb * QK];
    let mut offsets = vec![0f32; nb * QK];
    for sb in 0..num_super {
        let block = &bytes[sb * BLOCK..(sb + 1) * BLOCK];
        let qs_b = &block[0..QK_K / 8];
        let qh = &block[QK_K / 8..QK_K / 8 + QK_K / 16];
        let scales_bytes = &block[QK_K / 8 + QK_K / 16..BLOCK];
        let sc: [u16; 4] = [
            read_u16_le(&scales_bytes[0..2]),
            read_u16_le(&scales_bytes[2..4]),
            read_u16_le(&scales_bytes[4..6]),
            read_u16_le(&scales_bytes[6..8]),
        ];
        let scale_u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        let d = half::f16::from_bits(scale_u16).to_f32();
        let base_chunk = sb * CHUNKS_PER_SUPER;
        let mut qs_walk = 0usize;
        let mut qh_walk = 0usize;
        for ib in 0..QK_K / 32 {
            let chunk = base_chunk + ib;
            let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
            let idx0 = qs_b[qs_walk] as u16 | ((qh[qh_walk] as u16) << 8 & 0x700);
            let idx1 = qs_b[qs_walk + 1] as u16 | ((qh[qh_walk] as u16) << 4 & 0x700);
            let idx2 = qs_b[qs_walk + 2] as u16 | ((qh[qh_walk + 1] as u16) << 8 & 0x700);
            let idx3 = qs_b[qs_walk + 3] as u16 | ((qh[qh_walk + 1] as u16) << 4 & 0x700);
            let deltas = [
                if qh[qh_walk] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk + 1] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[qh_walk + 1] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            let groups = [
                (idx0, deltas[0], dl1, 0),
                (idx1, deltas[1], dl1, 8),
                (idx2, deltas[2], dl2, 16),
                (idx3, deltas[3], dl2, 24),
            ];
            for (idx, delta, dl, off) in groups {
                let grid = grid_u64_to_i8x8(IQ1S_GRID[idx as usize]);
                for j in 0..8 {
                    qs[chunk * QK + off + j] = grid[j] as f32;
                    scales[chunk * QK + off + j] = dl;
                    offsets[chunk * QK + off + j] = dl * delta;
                }
            }
            qs_walk += 4;
            qh_walk += 2;
        }
    }
    Ok((qs, scales, offsets))
}

/// Split GGUF Q8_0 / Q4_0 / IQ4NL blocks into integer-valued f32 quants
/// `[nb,32]` and per-block scales `[nb]`.
pub(super) fn split_gguf_blocks(
    scheme: QuantScheme,
    bytes: &[u8],
    nb: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let block_bytes = scheme.gguf_block_bytes() as usize;
    if bytes.len() != nb * block_bytes {
        return Err(CoremlError::Runtime(format!(
            "split_gguf_blocks: expected {} bytes, got {}",
            nb * block_bytes,
            bytes.len()
        )));
    }
    let mut qs = vec![0f32; nb * QK];
    let mut scales = vec![0f32; nb];
    for i in 0..nb {
        let off = i * block_bytes;
        let d = half::f16::from_bits(u16::from_le_bytes([bytes[off], bytes[off + 1]])).to_f32();
        scales[i] = d;
        match scheme {
            QuantScheme::GgufQ8_0 => {
                for j in 0..QK {
                    qs[i * QK + j] = bytes[off + 2 + j] as i8 as f32;
                }
            }
            QuantScheme::GgufQ4_0 => {
                for j in 0..QK / 2 {
                    let v0 = (bytes[off + 2 + j] & 0x0F) as i32 - 8;
                    qs[i * QK + j] = v0 as f32;
                }
                for j in 0..QK / 2 {
                    let v1 = (bytes[off + 2 + j] >> 4) as i32 - 8;
                    qs[i * QK + QK / 2 + j] = v1 as f32;
                }
            }
            QuantScheme::GgufIQ4NL => {
                for j in 0..QK / 2 {
                    let bx = bytes[off + 2 + j];
                    qs[i * QK + j] = KVALUES_IQ4NL[(bx & 0x0F) as usize] as f32;
                    qs[i * QK + QK / 2 + j] = KVALUES_IQ4NL[(bx >> 4) as usize] as f32;
                }
            }
            other => {
                return Err(CoremlError::Unsupported(format!(
                    "split_gguf_blocks: {other:?}"
                )));
            }
        }
    }
    Ok((qs, scales))
}

// --- Value / binding constructors -----------------------------------------

pub(super) fn bind_name(name: &str) -> proto::Argument {
    proto::Argument {
        arguments: vec![proto::argument::Binding {
            binding: Some(proto::argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

/// A list-valued binding: one `Argument` holding several name bindings
/// (used by `concat`'s `values` input).
pub(super) fn bind_names(names: &[String]) -> proto::Argument {
    proto::Argument {
        arguments: names
            .iter()
            .map(|n| proto::argument::Binding {
                binding: Some(proto::argument::binding::Binding::Name(n.clone())),
            })
            .collect(),
    }
}

pub(super) fn bind_value(v: proto::Value) -> proto::Argument {
    proto::Argument {
        arguments: vec![proto::argument::Binding {
            binding: Some(proto::argument::binding::Binding::Value(v)),
        }],
    }
}

pub(super) fn immediate(tensor: proto::TensorValue, vt: proto::ValueType) -> proto::Value {
    proto::Value {
        doc_string: String::new(),
        r#type: Some(vt),
        value: Some(proto::value::Value::ImmediateValue(
            proto::value::ImmediateValue {
                value: Some(proto::value::immediate_value::Value::Tensor(tensor)),
            },
        )),
    }
}

pub(super) fn scalar_shape(dtype: DType) -> Shape {
    Shape::new(&[], dtype)
}

pub(super) fn scalar_f32(x: f32) -> proto::Value {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Floats(
            proto::tensor_value::RepeatedFloats { values: vec![x] },
        )),
    };
    immediate(t, value_type(&scalar_shape(DType::F32)).unwrap())
}

pub(super) fn scalar_i32(x: i32) -> proto::Value {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Ints(
            proto::tensor_value::RepeatedInts { values: vec![x] },
        )),
    };
    immediate(t, value_type(&scalar_shape(DType::I32)).unwrap())
}

pub(super) fn scalar_str(s: &str) -> proto::Value {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Strings(
            proto::tensor_value::RepeatedStrings {
                values: vec![s.to_string()],
            },
        )),
    };
    // String scalars use a rank-0 STRING tensor type.
    let vt = proto::ValueType {
        r#type: Some(proto::value_type::Type::TensorType(proto::TensorType {
            data_type: proto::DataType::String as i32,
            rank: 0,
            dimensions: vec![],
            attributes: HashMap::new(),
        })),
    };
    immediate(t, vt)
}

pub(super) fn scalar_bool(b: bool) -> proto::Value {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Bools(
            proto::tensor_value::RepeatedBools { values: vec![b] },
        )),
    };
    immediate(t, value_type(&scalar_shape(DType::Bool)).unwrap())
}

/// Build a static rank-4 shape (used for canonical `[B,H,S,D]` attention).
pub(super) fn bhsd_shape(a: usize, b: usize, c: usize, d: usize) -> Shape {
    Shape::from_dims(
        &[
            Dim::Static(a),
            Dim::Static(b),
            Dim::Static(c),
            Dim::Static(d),
        ],
        DType::F32,
    )
}

/// Map shape dims to an i32 list for a MIL `reshape` target (`-1` for dynamic).
pub(super) fn dims_i32(dims: &[Dim]) -> Vec<i32> {
    dims.iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i32,
            Dim::Dynamic(_) => -1,
        })
        .collect()
}

pub(super) fn vec_i32(xs: &[i32]) -> proto::Value {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Ints(
            proto::tensor_value::RepeatedInts {
                values: xs.to_vec(),
            },
        )),
    };
    immediate(t, value_type(&Shape::new(&[xs.len()], DType::I32)).unwrap())
}

pub(super) fn tensor_f32(shape: &Shape, data: &[f32]) -> Result<proto::Value> {
    let t = proto::TensorValue {
        value: Some(proto::tensor_value::Value::Floats(
            proto::tensor_value::RepeatedFloats {
                values: data.to_vec(),
            },
        )),
    };
    Ok(immediate(t, value_type(shape)?))
}

pub(super) fn feature_description(io: &IoTensor) -> Result<proto::FeatureDescription> {
    let array_dt = match io.dtype {
        DType::F32 => proto::array_feature_type::ArrayDataType::Float32,
        DType::F16 => proto::array_feature_type::ArrayDataType::Float16,
        DType::I32 => proto::array_feature_type::ArrayDataType::Int32,
        DType::F64 => proto::array_feature_type::ArrayDataType::Double,
        other => return Err(CoremlError::Unsupported(format!("io dtype {other:?}"))),
    };
    let shape_flexibility = if io.flex_dims.iter().any(|&f| f) {
        let size_ranges: Vec<proto::SizeRange> = io
            .flex_dims
            .iter()
            .map(|&flex| proto::SizeRange {
                lower_bound: if flex { 1 } else { 0 },
                upper_bound: if flex { -1 } else { 0 },
            })
            .collect();
        Some(proto::array_feature_type::ShapeFlexibility::ShapeRange(
            proto::ShapeRange { size_ranges },
        ))
    } else {
        None
    };
    let array = proto::ArrayFeatureType {
        shape: io.dims.clone(),
        data_type: array_dt as i32,
        shape_flexibility,
    };
    Ok(proto::FeatureDescription {
        name: io.feature_name.clone(),
        short_description: String::new(),
        r#type: Some(proto::FeatureType {
            r#type: Some(proto::feature_type::Type::MultiArrayType(array)),
            is_optional: false,
        }),
    })
}

// --- misc helpers ---------------------------------------------------------

pub(crate) fn bytes_to_f32(data: &[u8], shape: &Shape) -> Result<Vec<f32>> {
    // Integer/bool constants are baked as f32 values (CoreML's blob has no int/bool
    // storage; arithmetic is f32). Bool/int value-constants like the VITS sequence
    // mask `arange` then flow through f32 ops correctly.
    match shape.dtype() {
        DType::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(CoremlError::Runtime(
                    "constant byte len not f32-aligned".into(),
                ));
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        DType::I64 => Ok(data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()),
        DType::I32 => Ok(data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()),
        DType::U32 => Ok(data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()),
        DType::Bool | DType::U8 => Ok(data.iter().map(|&b| b as f32).collect()),
        DType::I8 => Ok(data.iter().map(|&b| (b as i8) as f32).collect()),
        // F16 activations mode (RLX_COREML_F16): small synthesized consts arrive
        // as f16 bytes; decode to f32 for inline/blob baking (make_const re-casts
        // to f16 when the const's shape dtype is F16).
        DType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(CoremlError::Runtime(
                    "constant byte len not f16-aligned".into(),
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => Err(CoremlError::Unsupported(format!(
            "constant dtype {other:?} (only F32/f16/int/bool baked inline)"
        ))),
    }
}

/// Sanitise an arbitrary name into a valid MIL identifier
/// (`[A-Za-z_][A-Za-z0-9_]*`).
pub(super) fn sanitize(raw: &str) -> String {
    let mut s = String::with_capacity(raw.len() + 1);
    for (i, c) in raw.chars().enumerate() {
        let ok = c.is_ascii_alphanumeric() || c == '_';
        let c = if ok { c } else { '_' };
        if i == 0 && c.is_ascii_digit() {
            s.push('_');
        }
        s.push(c);
    }
    if s.is_empty() {
        s.push('_');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn immediate_tensor(op: &proto::Operation) -> &proto::TensorValue {
        let val = op.attributes.get("val").expect("const has `val` attr");
        let Some(proto::value::Value::ImmediateValue(iv)) = val.value.as_ref() else {
            panic!("expected immediate value");
        };
        let Some(proto::value::immediate_value::Value::Tensor(t)) = iv.value.as_ref() else {
            panic!("expected tensor immediate");
        };
        t
    }

    /// A small (< BLOB_MIN_ELEMS) f16 const must be encoded as LE f16 in the
    /// `bytes` field, not f32 `floats` — otherwise CoreML rejects the model at
    /// load with "Tensor storage and type have different number of elements".
    /// Regression test for the learned-FFT butterfly (per-scalar twiddle params).
    #[test]
    fn small_f16_const_uses_bytes_immediate() {
        let mut blob = crate::mlpackage::BlobWriter::new();
        let data = [1.0f32, -2.0, 0.5, 1234.0];
        assert!(data.len() < BLOB_MIN_ELEMS);
        let shape = Shape::new(&[data.len()], DType::F16);
        let op = make_const_float(&mut blob, "tw", &shape, &data, DType::F16).unwrap();
        match immediate_tensor(&op).value.as_ref().unwrap() {
            proto::tensor_value::Value::Bytes(b) => {
                assert_eq!(b.values.len(), data.len() * 2, "2 bytes per f16 element");
                assert!((read_f16_le(&b.values[0..2]) - 1.0).abs() < 1e-3);
                assert!((read_f16_le(&b.values[2..4]) + 2.0).abs() < 1e-3);
            }
            other => panic!("expected Bytes immediate for f16 const, got {other:?}"),
        }
    }

    /// f32 path is unchanged — small consts stay `floats` immediates.
    #[test]
    fn small_f32_const_uses_floats_immediate() {
        let mut blob = crate::mlpackage::BlobWriter::new();
        let data = [1.0f32, 2.0, 3.0];
        let shape = Shape::new(&[data.len()], DType::F32);
        let op = make_const_float(&mut blob, "c", &shape, &data, DType::F32).unwrap();
        assert!(matches!(
            immediate_tensor(&op).value.as_ref().unwrap(),
            proto::tensor_value::Value::Floats(_)
        ));
    }

    /// `Cast` targets MIL lacks are demoted/widened, and the string returned by
    /// `mil_cast_dtype` must match the MIL type `mil_data_type` emits for the
    /// same dtype so a cast op's `dtype` attr and its output value type agree.
    #[test]
    fn cast_dtype_demotes_unsupported_types() {
        // Natively-supported types are unchanged.
        assert_eq!(mil_cast_dtype(DType::F32).unwrap(), "fp32");
        assert_eq!(mil_cast_dtype(DType::F16).unwrap(), "fp16");
        assert_eq!(mil_cast_dtype(DType::I32).unwrap(), "int32");
        assert_eq!(mil_cast_dtype(DType::Bool).unwrap(), "bool");
        // Demotions / widenings for types CoreML has no storage for.
        assert_eq!(mil_cast_dtype(DType::F64).unwrap(), "fp32", "no f64 → fp32");
        assert_eq!(
            mil_cast_dtype(DType::BF16).unwrap(),
            "fp16",
            "no bf16 → fp16"
        );
        assert_eq!(
            mil_cast_dtype(DType::I16).unwrap(),
            "int32",
            "no int16 → int32"
        );
    }

    /// `mil_data_type` (the output tensor type) demotes the same set the cast
    /// string does — otherwise the cast op and its result type would disagree.
    #[test]
    fn data_type_demotes_match_cast_dtype() {
        assert_eq!(
            mil_data_type(DType::F64).unwrap(),
            proto::DataType::Float32 as i32,
        );
        assert_eq!(
            mil_data_type(DType::BF16).unwrap(),
            proto::DataType::Float16 as i32,
        );
        assert_eq!(
            mil_data_type(DType::I16).unwrap(),
            proto::DataType::Int32 as i32,
        );
    }

    /// C64 (complex) has no MIL equivalent → both dtype mappers reject it with
    /// a clear typed `Unsupported`, never silently corrupting the re/im layout.
    #[test]
    fn c64_is_cleanly_rejected() {
        let e = mil_cast_dtype(DType::C64).unwrap_err();
        assert!(matches!(e, CoremlError::Unsupported(_)), "got {e:?}");
        assert!(format!("{e:?}").contains("C64"));
        let e2 = mil_data_type(DType::C64).unwrap_err();
        assert!(matches!(e2, CoremlError::Unsupported(_)), "got {e2:?}");
        assert!(format!("{e2:?}").contains("C64"));
    }
}
