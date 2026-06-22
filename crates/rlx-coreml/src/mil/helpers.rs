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
    }
}

/// MIL `DataType` (proto enum value) for an RLX dtype.
pub(super) fn mil_data_type(dt: DType) -> Result<i32> {
    let v = match dt {
        DType::F32 => proto::DataType::Float32,
        DType::F16 => proto::DataType::Float16,
        DType::I32 => proto::DataType::Int32,
        DType::I64 => proto::DataType::Int64,
        DType::I8 => proto::DataType::Int8,
        DType::U8 => proto::DataType::Uint8,
        DType::Bool => proto::DataType::Bool,
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
        GgufQ2K => rlx_gguf::dequant_q2_k(bytes, n),
        GgufQ3K => rlx_gguf::dequant_q3_k(bytes, n),
        GgufQ4K => rlx_gguf::dequant_q4_k(bytes, n),
        GgufQ5K => rlx_gguf::dequant_q5_k(bytes, n),
        GgufQ6K => rlx_gguf::dequant_q6_k(bytes, n),
        GgufQ8K => rlx_gguf::dequant_q8_k(bytes, n),
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
pub(super) fn mil_cast_dtype(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "fp32",
        DType::F16 => "fp16",
        DType::I32 => "int32",
        DType::I8 => "int8",
        DType::U8 => "uint8",
        DType::Bool => "bool",
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

/// Build a single-output op with the given input bindings. Sets the op's
/// `name` attribute and the output `NamedValueType` (so the value carries
/// its tensor type, which MIL validation needs).
pub(super) fn simple_op(
    ty: &str,
    out_name: &str,
    out_shape: &Shape,
    inputs: Vec<(&str, proto::Argument)>,
) -> Result<proto::Operation> {
    let mut input_map = HashMap::new();
    for (k, v) in inputs {
        input_map.insert(k.to_string(), v);
    }
    let mut attributes = HashMap::new();
    attributes.insert("name".to_string(), scalar_str(out_name));
    Ok(proto::Operation {
        r#type: ty.to_string(),
        inputs: input_map,
        outputs: vec![named_value_type(out_name, out_shape)?],
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
    let array = proto::ArrayFeatureType {
        shape: io.dims.clone(),
        data_type: array_dt as i32,
        shape_flexibility: None,
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

pub(super) fn bytes_to_f32(data: &[u8], shape: &Shape) -> Result<Vec<f32>> {
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
        other => Err(CoremlError::Unsupported(format!(
            "constant dtype {other:?} (only F32/int/bool baked inline)"
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
