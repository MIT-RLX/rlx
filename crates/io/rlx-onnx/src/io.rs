// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rlx_ir::DType;

/// Tensor element type for an ONNX input or output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxElementType {
    Float32,
    Int64,
    Int32,
    Bool,
    Other,
}

impl OnnxElementType {
    pub fn from_dtype_str(s: &str) -> Self {
        match s {
            "f32" | "float" | "float32" => Self::Float32,
            "i64" | "int64" => Self::Int64,
            "i32" | "int32" => Self::Int32,
            "bool" => Self::Bool,
            _ => Self::Other,
        }
    }

    pub fn to_rlx_dtype(self) -> DType {
        match self {
            Self::Float32 => DType::F32,
            Self::Int64 => DType::I64,
            Self::Int32 => DType::I32,
            Self::Bool => DType::Bool,
            Self::Other => DType::F32,
        }
    }
}

/// I/O descriptor from the ONNX model.
#[derive(Debug, Clone)]
pub struct IoDesc {
    pub name: String,
    pub element_type: OnnxElementType,
    /// Static dims are positive; unknown/dynamic dims are `None`.
    pub shape: Vec<Option<i64>>,
}

/// One input tensor for [`crate::OnnxModel::run`].
#[derive(Debug, Clone)]
pub enum OnnxTensor {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

pub fn resolve_extent(dim: Option<i64>, dynamic_dim: i64) -> i64 {
    match dim {
        Some(d) if d > 0 => d,
        _ => dynamic_dim.max(1),
    }
}

pub fn num_elements_sized(desc: &IoDesc, dynamic_dim: i64) -> Result<usize> {
    if desc.shape.is_empty() {
        return Ok(1);
    }
    let mut n: i64 = 1;
    for &d in &desc.shape {
        let e = resolve_extent(d, dynamic_dim);
        n = n
            .checked_mul(e)
            .with_context(|| format!("shape overflow for '{}': {:?}", desc.name, desc.shape))?;
    }
    Ok(n as usize)
}

pub fn zero_tensor_sized(desc: &IoDesc, dynamic_dim: i64) -> Result<OnnxTensor> {
    let n = num_elements_sized(desc, dynamic_dim)?;
    Ok(match desc.element_type {
        OnnxElementType::Float32 => OnnxTensor::F32(vec![0.0; n]),
        OnnxElementType::Int64 => OnnxTensor::I64(vec![0; n]),
        OnnxElementType::Int32 => OnnxTensor::I32(vec![0; n]),
        OnnxElementType::Bool => bail!(
            "rlx-onnx: bool input '{}' — supply data via OnnxTensor (zero_inputs unsupported)",
            desc.name
        ),
        OnnxElementType::Other => bail!(
            "rlx-onnx: cannot synthesize zero input for '{}' ({:?})",
            desc.name,
            desc.element_type
        ),
    })
}

pub fn zero_inputs_sized(
    inputs: &[IoDesc],
    dynamic_dim: i64,
) -> Result<HashMap<String, OnnxTensor>> {
    let dynamic_dim = dynamic_dim.max(1);
    let mut map = HashMap::new();
    for desc in inputs {
        map.insert(desc.name.clone(), zero_tensor_sized(desc, dynamic_dim)?);
    }
    Ok(map)
}

pub fn tensor_to_typed_bytes<'a>(
    tensor: &'a OnnxTensor,
    desc: &IoDesc,
) -> Result<(&'a [u8], DType)> {
    let dtype = desc.element_type.to_rlx_dtype();
    Ok(match (desc.element_type, tensor) {
        (OnnxElementType::Float32, OnnxTensor::F32(data)) => (bytemuck::cast_slice(data), dtype),
        (OnnxElementType::Int64, OnnxTensor::I64(data)) => (bytemuck::cast_slice(data), dtype),
        (OnnxElementType::Int32, OnnxTensor::I32(data)) => (bytemuck::cast_slice(data), dtype),
        (expected, _) => bail!(
            "rlx-onnx: input '{}' type mismatch (expected {:?})",
            desc.name,
            expected
        ),
    })
}

pub fn typed_bytes_to_tensor(bytes: &[u8], dtype: DType) -> Result<OnnxTensor> {
    Ok(match dtype {
        DType::F32 => {
            let n = bytes.len() / 4;
            let mut v = Vec::with_capacity(n);
            for chunk in bytes.chunks_exact(4) {
                v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            OnnxTensor::F32(v)
        }
        DType::I64 => {
            let n = bytes.len() / 8;
            let mut v = Vec::with_capacity(n);
            for chunk in bytes.chunks_exact(8) {
                v.push(i64::from_le_bytes(chunk.try_into().unwrap()));
            }
            OnnxTensor::I64(v)
        }
        DType::I32 => {
            let n = bytes.len() / 4;
            let mut v = Vec::with_capacity(n);
            for chunk in bytes.chunks_exact(4) {
                v.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            OnnxTensor::I32(v)
        }
        other => bail!("rlx-onnx: unsupported output dtype {other:?}"),
    })
}
