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

//! ONNX Runtime execution path (optional `ort` / `ort-fallback` feature).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ort::session::{SessionInputValue, SessionInputs};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, Value, ValueType};
use rlx_runtime::Device;

use crate::backend::{OrtSession, build_onnx_session};
use crate::io::{self, IoDesc, OnnxElementType, OnnxTensor};

pub struct OrtOnnx {
    pub path: PathBuf,
    pub device: Device,
    pub ort_ep: String,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
    pub dynamic_dim: i64,
    inner: OrtSession,
}

impl OrtOnnx {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let built = build_onnx_session(&path, device)?;
        let inputs = describe_outlets(built.session.inputs())?;
        let outputs = describe_outlets(built.session.outputs())?;
        let ort_ep = built.ort_ep.clone();
        Ok(Self {
            path,
            device,
            inputs,
            outputs,
            ort_ep,
            dynamic_dim: 1,
            inner: built,
        })
    }

    pub fn run(&mut self, inputs: &HashMap<String, OnnxTensor>) -> Result<Vec<OnnxTensor>> {
        let mut ort_values: Vec<(String, Value)> = Vec::new();
        for desc in &self.inputs {
            let tensor = inputs
                .get(&desc.name)
                .with_context(|| format!("missing input '{}'", desc.name))?;
            ort_values.push((desc.name.clone(), self.tensor_to_value(desc, tensor)?));
        }
        let session_in: Vec<(String, SessionInputValue<'_>)> = ort_values
            .into_iter()
            .map(|(name, v)| (name, SessionInputValue::Owned(v)))
            .collect();
        let outputs = self.inner.session.run(SessionInputs::from(session_in))?;
        let mut out = Vec::with_capacity(self.outputs.len());
        for (i, desc) in self.outputs.iter().enumerate() {
            let val = outputs.get(&desc.name).unwrap_or(&outputs[i]);
            out.push(ort_to_tensor(desc, val)?);
        }
        Ok(out)
    }

    pub fn zero_inputs_sized(&mut self, dynamic_dim: i64) -> Result<HashMap<String, OnnxTensor>> {
        self.dynamic_dim = dynamic_dim.max(1);
        io::zero_inputs_sized(&self.inputs, self.dynamic_dim)
    }

    fn tensor_to_value(&self, desc: &IoDesc, t: &OnnxTensor) -> Result<Value> {
        let shape = static_shape_sized(desc, self.dynamic_dim)?;
        Ok(match (desc.element_type, t) {
            (OnnxElementType::Float32, OnnxTensor::F32(data)) => {
                Tensor::from_array((shape, data.clone()))
                    .context("f32 input tensor")?
                    .into_dyn()
            }
            (OnnxElementType::Int64, OnnxTensor::I64(data)) => {
                Tensor::from_array((shape, data.clone()))
                    .context("i64 input tensor")?
                    .into_dyn()
            }
            (OnnxElementType::Int32, OnnxTensor::I32(data)) => {
                Tensor::from_array((shape, data.clone()))
                    .context("i32 input tensor")?
                    .into_dyn()
            }
            (expected, got) => bail!(
                "rlx-onnx: input '{}' type mismatch (expected {:?}, got {:?})",
                desc.name,
                expected,
                std::mem::discriminant(got)
            ),
        })
    }
}

fn describe_outlets(outlets: &[ort::value::Outlet]) -> Result<Vec<IoDesc>> {
    let mut v = Vec::with_capacity(outlets.len());
    for o in outlets {
        let (element_type, shape) = match o.dtype() {
            ValueType::Tensor { ty, shape, .. } => {
                let dims: Vec<Option<i64>> = shape
                    .iter()
                    .map(|&d| if d < 0 { None } else { Some(d) })
                    .collect();
                (OnnxElementType::from_ort(*ty), dims)
            }
            other => bail!(
                "rlx-onnx: unsupported I/O type for '{}': {other:?}",
                o.name()
            ),
        };
        v.push(IoDesc {
            name: o.name().to_string(),
            element_type,
            shape,
        });
    }
    Ok(v)
}

impl OnnxElementType {
    fn from_ort(ty: TensorElementType) -> Self {
        match ty {
            TensorElementType::Float32 => Self::Float32,
            TensorElementType::Int64 => Self::Int64,
            TensorElementType::Int32 => Self::Int32,
            TensorElementType::Bool => Self::Bool,
            _ => Self::Other,
        }
    }
}

fn static_shape_sized(desc: &IoDesc, dynamic_dim: i64) -> Result<Vec<usize>> {
    desc.shape
        .iter()
        .map(|&d| Ok(io::resolve_extent(d, dynamic_dim) as usize))
        .collect()
}

fn ort_to_tensor(desc: &IoDesc, val: &DynValue) -> Result<OnnxTensor> {
    match desc.element_type {
        OnnxElementType::Float32 => {
            let (_shape, data) = val.try_extract_tensor::<f32>()?;
            Ok(OnnxTensor::F32(data.to_vec()))
        }
        OnnxElementType::Int64 => {
            let (_shape, data) = val.try_extract_tensor::<i64>()?;
            Ok(OnnxTensor::I64(data.to_vec()))
        }
        OnnxElementType::Int32 => {
            let (_shape, data) = val.try_extract_tensor::<i32>()?;
            Ok(OnnxTensor::I32(data.to_vec()))
        }
        other => bail!(
            "rlx-onnx: output '{}' has unsupported type {:?}",
            desc.name,
            other
        ),
    }
}
