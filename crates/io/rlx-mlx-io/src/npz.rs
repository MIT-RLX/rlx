// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NumPy `.npy` / `.npz` loaders (`mx.save` / `mx.savez`).

use anyhow::{Context, Result, bail};
use npyz::npz::NpzArchive;
use npyz::{DType, NpyFile, TypeChar};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::load::MlxTensor;

fn tensor_from_npy<R: Read>(name: String, npy: NpyFile<R>) -> Result<MlxTensor> {
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let dtype = npy.dtype();
    let (tchar, size) = match &dtype {
        DType::Plain(ts) => (ts.type_char(), ts.size_field()),
        other => bail!("unsupported structured npy dtype {other:?}"),
    };
    match (tchar, size) {
        (TypeChar::Float, 4) => {
            let data = npy.into_vec::<f32>().context("read f32")?;
            Ok(MlxTensor {
                name,
                shape,
                data_f32: Some(data),
                data_u8: None,
                is_quant_weight: false,
            })
        }
        (TypeChar::Float, 2) => {
            let data = npy
                .into_vec::<half::f16>()
                .context("read f16")?
                .into_iter()
                .map(|v| v.to_f32())
                .collect();
            Ok(MlxTensor {
                name,
                shape,
                data_f32: Some(data),
                data_u8: None,
                is_quant_weight: false,
            })
        }
        (TypeChar::Float, 8) => {
            let data = npy
                .into_vec::<f64>()
                .context("read f64")?
                .into_iter()
                .map(|v| v as f32)
                .collect();
            Ok(MlxTensor {
                name,
                shape,
                data_f32: Some(data),
                data_u8: None,
                is_quant_weight: false,
            })
        }
        (TypeChar::Int, 4) => {
            let data = npy
                .into_vec::<i32>()
                .context("read i32")?
                .into_iter()
                .map(|v| v as f32)
                .collect();
            Ok(MlxTensor {
                name,
                shape,
                data_f32: Some(data),
                data_u8: None,
                is_quant_weight: false,
            })
        }
        (TypeChar::Uint, 4) => {
            let v = npy.into_vec::<u32>().context("read u32")?;
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in &v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            Ok(MlxTensor {
                name,
                shape,
                data_f32: None,
                data_u8: Some(bytes),
                is_quant_weight: true,
            })
        }
        (TypeChar::Uint, 1) | (TypeChar::Int, 1) => {
            let data = npy.into_vec::<u8>().context("read u8")?;
            Ok(MlxTensor {
                name,
                shape,
                data_f32: None,
                data_u8: Some(data),
                is_quant_weight: true,
            })
        }
        other => bail!("unsupported npy dtype {other:?}"),
    }
}

pub fn load_npy(path: &Path) -> Result<HashMap<String, MlxTensor>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let npy =
        NpyFile::new(BufReader::new(f)).with_context(|| format!("parse {}", path.display()))?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("arr")
        .to_string();
    let t = tensor_from_npy(name.clone(), npy)?;
    Ok(HashMap::from([(name, t)]))
}

pub fn load_npz(path: &Path) -> Result<HashMap<String, MlxTensor>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut npz = NpzArchive::new(BufReader::new(f))
        .with_context(|| format!("parse npz {}", path.display()))?;
    let names: Vec<String> = npz.array_names().map(|s| s.to_string()).collect();
    let mut out = HashMap::new();
    for name in names {
        let npy = npz
            .by_name(&name)?
            .with_context(|| format!("missing npz member {name}"))?;
        let t = tensor_from_npy(name.clone(), npy)?;
        out.insert(name, t);
    }
    Ok(out)
}
