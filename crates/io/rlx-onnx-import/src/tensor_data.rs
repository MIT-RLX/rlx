// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use rlx_ir::DType;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

/// Typed initializer bytes (U8/I8 params) alongside F32 params from import.
pub type TypedParams = HashMap<String, (Vec<u8>, DType)>;

/// Load all I64 initializer tensors from a safetensors blob.
pub fn load_i64_params(bytes: &[u8]) -> Result<HashMap<String, Vec<i64>>> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let data = view.data();
        let v: Vec<i64> = match view.dtype() {
            Dtype::I64 => data
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            // Axis / index / shape initializers are commonly exported as I32
            // (ONNX permits int32 or int64 for CumSum `axis`, Gather indices,
            // Slice/Squeeze axes, …). Dropping them here silently loses the
            // constant, so lookups via `i64_tensor` fall back to a default —
            // e.g. an unresolved axis becomes 0, which turned the KittenTTS
            // sine-source phase `CumSum` (axis=1) into a no-op over the size-1
            // batch axis → dead harmonic oscillator → near-silent vocoder.
            Dtype::I32 => data
                .chunks_exact(4)
                .map(|chunk| i64::from(i32::from_le_bytes(chunk.try_into().unwrap())))
                .collect(),
            Dtype::BOOL => data.iter().map(|&b| i64::from(b != 0)).collect(),
            _ => continue,
        };
        out.insert(name.to_string(), v);
    }
    Ok(out)
}

/// Dequantize `*_quantized` U8/I8 initializers into `params` as f32 (in-place keys).
pub fn materialize_quantized_f32(
    bytes: &[u8],
    params: &mut HashMap<String, Vec<f32>>,
    init_shapes: &mut HashMap<String, Vec<usize>>,
) -> Result<()> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    for name in st.names() {
        if !name.ends_with("_quantized") || params.contains_key(name) {
            continue;
        }
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let scale = params
            .get(&format!("{name}_scale"))
            .or_else(|| {
                name.strip_suffix("_quantized")
                    .and_then(|base| params.get(&format!("{base}_scale")))
            })
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(1.0);
        let zp = params
            .get(&format!("{name}_zero_point"))
            .or_else(|| {
                name.strip_suffix("_quantized")
                    .and_then(|base| params.get(&format!("{base}_zero_point")))
            })
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        let out = match view.dtype() {
            Dtype::U8 => view
                .data()
                .iter()
                .map(|&b| (b as f32 - zp) * scale)
                .collect(),
            Dtype::I8 => view
                .data()
                .iter()
                .map(|&b| (b as i8 as f32 - zp) * scale)
                .collect(),
            Dtype::F32 => {
                let raw = view.data();
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            _ => continue,
        };
        params.insert(name.to_string(), out);
        init_shapes.insert(name.to_string(), shape);
    }
    Ok(())
}

/// Read i64 values from an initializer name (safetensors) or f32 param cast.
pub fn i64_tensor(
    i64_params: &HashMap<String, Vec<i64>>,
    f32_params: &HashMap<String, Vec<f32>>,
    name: &str,
) -> Option<Vec<i64>> {
    if let Some(v) = i64_params.get(name) {
        return Some(v.clone());
    }
    f32_params
        .get(name)
        .map(|v| v.iter().map(|&x| x as i64).collect())
}

pub fn f32_tensor(f32_params: &HashMap<String, Vec<f32>>, name: &str) -> Option<Vec<f32>> {
    f32_params.get(name).cloned()
}

/// Load all weight tensors from safetensors into f32 `params` + shapes (F16/BF16/U8 dequant).
pub fn load_f32_params(
    bytes: &[u8],
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>)> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    let mut params = HashMap::new();
    let mut init_shapes = HashMap::new();
    let mut pending_quant: Vec<(String, Vec<usize>, Vec<u8>, Dtype)> = Vec::new();

    for name in st.names() {
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        init_shapes.insert(name.to_string(), shape.clone());
        match view.dtype() {
            Dtype::F32 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                params.insert(name.to_string(), data);
            }
            Dtype::F16 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect();
                params.insert(name.to_string(), data);
            }
            Dtype::BF16 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                params.insert(name.to_string(), data);
            }
            Dtype::U8 | Dtype::I8 => {
                pending_quant.push((name.to_string(), shape, view.data().to_vec(), view.dtype()));
            }
            _ => {}
        }
    }

    for (name, shape, raw, dt) in pending_quant {
        let scale = params
            .get(&format!("{name}_scale"))
            .or_else(|| {
                name.strip_suffix("_quantized")
                    .and_then(|b| params.get(&format!("{b}_scale")))
            })
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(1.0);
        let zp = params
            .get(&format!("{name}_zero_point"))
            .or_else(|| {
                name.strip_suffix("_quantized")
                    .and_then(|b| params.get(&format!("{b}_zero_point")))
            })
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0);
        let data: Vec<f32> = match dt {
            Dtype::U8 => raw.iter().map(|&b| (b as f32 - zp) * scale).collect(),
            _ => raw.iter().map(|&b| (b as i8 as f32 - zp) * scale).collect(),
        };
        let out_name = name
            .strip_suffix("_quantized")
            .unwrap_or(name.as_str())
            .to_string();
        if !params.contains_key(&out_name) {
            params.insert(out_name.clone(), data.clone());
            init_shapes.insert(out_name.clone(), shape.clone());
        }
        if name.ends_with("_quantized") && !params.contains_key(&name) {
            params.insert(name.clone(), data);
            init_shapes.insert(name, shape);
        }
    }

    Ok((params, init_shapes))
}

/// Map a graph `*_quant_f32_weight` initializer to its int8 safetensors key.
pub fn quant_matmul_weight_key(graph_weight: &str, known: &HashSet<String>) -> Option<String> {
    let mut s = graph_weight.trim_start_matches('/').to_string();
    s = s
        .strip_suffix("/Gemm_MatMul_quant_f32_weight")
        .or_else(|| s.strip_suffix("_Gemm_MatMul_quant_f32_weight"))
        .or_else(|| s.strip_suffix("/MatMul_quant_f32_weight"))
        .or_else(|| s.strip_suffix("_MatMul_quant_f32_weight"))
        .or_else(|| s.strip_suffix("_quant_f32_weight"))
        .unwrap_or(&s)
        .to_string();
    let dotted = s.replace('/', ".");
    for prefix in ["kmodel.predictor.", "kmodel.decoder.", "kmodel."] {
        let cand = format!("{prefix}{dotted}.weight_quantized");
        if known.contains(&cand) {
            return Some(cand);
        }
    }
    None
}

/// Load U8/I8 weight tensors without dequantizing to F32.
pub fn load_typed_quant_params(bytes: &[u8]) -> Result<(TypedParams, HashMap<String, Vec<usize>>)> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    let mut typed = TypedParams::new();
    let mut shapes = HashMap::new();
    for name in st.names() {
        if !name.ends_with("_quantized") {
            continue;
        }
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        shapes.insert(name.to_string(), shape.clone());
        let (dtype, data) = match view.dtype() {
            Dtype::U8 => (DType::U8, view.data().to_vec()),
            Dtype::I8 => (DType::I8, view.data().to_vec()),
            _ => continue,
        };
        typed.insert(name.to_string(), (data, dtype));
    }
    Ok((typed, shapes))
}
