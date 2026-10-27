// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NumPy `.npy` / `.npz` loaders (`mx.save` / `mx.savez`).

use anyhow::{Context, Result, bail};
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
    let mut npz = zip::ZipArchive::new(BufReader::new(f))
        .with_context(|| format!("parse npz {}", path.display()))?;
    let mut out = HashMap::new();
    // An .npz is a zip of `<name>.npy` members (what npyz's NpzArchive reads).
    for i in 0..npz.len() {
        let member = npz.by_index(i)?;
        let Some(name) = member.name()?.strip_suffix(".npy").map(str::to_string) else {
            continue;
        };
        let npy = NpyFile::new(member).with_context(|| format!("parse npz member {name}"))?;
        let t = tensor_from_npy(name.clone(), npy)?;
        out.insert(name, t);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// `.npy` v1 bytes for a little-endian f32 array.
    fn npy_f32(shape: &str, data: &[f32]) -> Vec<u8> {
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn load_npz_reads_stored_and_deflated_members() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.npz");
        {
            let mut zw = zip::ZipWriter::new(File::create(&path).unwrap());
            let stored = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let deflated = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file("a.npy", stored).unwrap();
            zw.write_all(&npy_f32("(2,)", &[1.0, -2.5])).unwrap();
            zw.start_file("layer.b.npy", deflated).unwrap();
            zw.write_all(&npy_f32("(2, 2)", &[0.0, 1.0, 2.0, 3.0]))
                .unwrap();
            zw.start_file("README.txt", stored).unwrap();
            zw.write_all(b"not an array").unwrap();
            zw.finish().unwrap();
        }
        let got = load_npz(&path).unwrap();
        assert_eq!(got.len(), 2, "non-.npy members are skipped");
        assert_eq!(got["a"].shape, vec![2]);
        assert_eq!(got["a"].data_f32.as_deref(), Some(&[1.0, -2.5][..]));
        assert_eq!(got["layer.b"].shape, vec![2, 2]);
        assert_eq!(
            got["layer.b"].data_f32.as_deref(),
            Some(&[0.0, 1.0, 2.0, 3.0][..])
        );
    }
}
