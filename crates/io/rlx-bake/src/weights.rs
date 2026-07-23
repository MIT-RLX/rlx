// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Load weights from safetensors (decoded to f32).

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use std::collections::HashMap;
use std::path::Path;

/// Load every tensor in a safetensors file, decoding to f32.
pub fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.is_empty() {
        return Ok(HashMap::new());
    }
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let data = decode_f32(view.data(), view.dtype())
            .with_context(|| format!("decoding tensor {name}"))?;
        out.insert(name.to_string(), data);
    }
    Ok(out)
}

fn decode_f32(raw: &[u8], dt: safetensors::Dtype) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    Ok(match dt {
        D::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F64 => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::F16 => raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::I64 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::I32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::I16 => raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        D::U8 => raw.iter().map(|&b| b as f32).collect(),
        D::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        D::BOOL => raw.iter().map(|&b| (b != 0) as u8 as f32).collect(),
        other => bail!("cannot decode safetensors dtype {other:?} to f32"),
    })
}
