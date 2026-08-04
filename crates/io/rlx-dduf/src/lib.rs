// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HuggingFace **DDUF** (`.dduf`) — ZIP of nested safetensors + JSON.
//!
//! Tensor names are qualified as `{component}/{tensor}` where `component`
//! is the first path segment of the member (e.g. `transformer/…`, `vae/…`).
//! Root-level tensors use `./{tensor}`.
//!
//! Prefer [`visit_f32_tensors`] / [`DdufArchive`] when importing large packs so
//! each safetensors member is decoded and dropped before the next is read.

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

/// Dense f32 payload with on-disk shape preserved.
pub type ShapedF32 = (Vec<usize>, Vec<f32>);

/// One dense f32 tensor extracted from a DDUF member.
#[derive(Debug, Clone)]
pub struct DdufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// ZIP member path that held this tensor's safetensors file.
    pub member: String,
}

/// Metadata + JSON sidecars from a `.dduf` without holding tensor payloads.
#[derive(Debug, Clone)]
pub struct DdufMeta {
    pub path: PathBuf,
    pub model_index: Option<serde_json::Value>,
    pub configs: HashMap<String, serde_json::Value>,
}

/// Opened `.dduf` with all tensors decoded to f32 (convenient for small packs).
#[derive(Debug)]
pub struct DdufFile {
    pub path: PathBuf,
    /// Root `model_index.json` when present.
    pub model_index: Option<serde_json::Value>,
    /// Component `config.json` payloads keyed by component dir name.
    pub configs: HashMap<String, serde_json::Value>,
    tensors: HashMap<String, DdufTensor>,
}

impl DdufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        Self::from_reader(f, path)
    }

    pub fn from_reader<R: Read + Seek>(reader: R, path: PathBuf) -> Result<Self> {
        let mut tensors = HashMap::new();
        let meta = visit_f32_tensors_reader(reader, path.clone(), |t| {
            tensors.insert(t.name.clone(), t);
            Ok(())
        })?;
        Ok(Self {
            path: meta.path,
            model_index: meta.model_index,
            configs: meta.configs,
            tensors,
        })
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<_> = self.tensors.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn get(&self, name: &str) -> Option<&DdufTensor> {
        self.tensors.get(name)
    }

    /// Dense f32 payloads only (shapes discarded). Prefer [`Self::into_shaped_f32`]
    /// or [`Self::into_tensors`] when layout matters.
    pub fn into_f32_map(self) -> HashMap<String, Vec<f32>> {
        self.tensors.into_iter().map(|(k, t)| (k, t.data)).collect()
    }

    /// Dense f32 payloads with on-disk shapes preserved.
    pub fn into_shaped_f32(self) -> HashMap<String, ShapedF32> {
        self.tensors
            .into_iter()
            .map(|(k, t)| (k, (t.shape, t.data)))
            .collect()
    }

    pub fn into_tensors(self) -> HashMap<String, DdufTensor> {
        self.tensors
    }

    pub fn tensor_f32(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(|t| t.data.as_slice())
            .with_context(|| format!("missing tensor {name}"))
    }
}

/// ZIP kept open for selective / streaming tensor loads.
pub struct DdufArchive {
    path: PathBuf,
    zip: ZipArchive<File>,
    member_names: Vec<String>,
    pub model_index: Option<serde_json::Value>,
    pub configs: HashMap<String, serde_json::Value>,
}

impl DdufArchive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut zip = ZipArchive::new(f).context("open dduf zip")?;
        let member_names: Vec<String> = (0..zip.len())
            .filter_map(|i| zip.by_index(i).ok().map(|e| e.name().to_string()))
            .collect();
        let mut model_index = None;
        let mut configs = HashMap::new();
        for name in &member_names {
            if name.ends_with('/') {
                continue;
            }
            if name == "model_index.json" || name.ends_with("/model_index.json") {
                let bytes = read_member(&mut zip, name)?;
                model_index = Some(serde_json::from_slice(&bytes).context("model_index.json")?);
            } else if name.ends_with("config.json") {
                let bytes = read_member(&mut zip, name)?;
                configs.insert(
                    component_of(name),
                    serde_json::from_slice(&bytes).with_context(|| format!("parse {name}"))?,
                );
            }
        }
        Ok(Self {
            path,
            zip,
            member_names,
            model_index,
            configs,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Safetensors ZIP member paths (not qualified tensor names).
    pub fn safetensors_members(&self) -> Vec<&str> {
        self.member_names
            .iter()
            .filter(|n| n.ends_with(".safetensors"))
            .map(|s| s.as_str())
            .collect()
    }

    /// Decode one safetensors member to f32 tensors, then drop the member bytes.
    pub fn load_member_f32(&mut self, member: &str) -> Result<Vec<DdufTensor>> {
        let bytes = read_member(&mut self.zip, member)?;
        decode_safetensors_f32(member, &bytes)
    }

    /// Stream every f32 tensor: one safetensors member at a time.
    pub fn visit_f32_tensors(
        &mut self,
        mut visit: impl FnMut(DdufTensor) -> Result<()>,
    ) -> Result<()> {
        let members: Vec<String> = self
            .safetensors_members()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        for member in members {
            for t in self.load_member_f32(&member)? {
                visit(t)?;
            }
        }
        Ok(())
    }
}

/// Stream f32 tensors from `path` without retaining the full pack in memory.
pub fn visit_f32_tensors(
    path: impl AsRef<Path>,
    visit: impl FnMut(DdufTensor) -> Result<()>,
) -> Result<DdufMeta> {
    let path = path.as_ref().to_path_buf();
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    visit_f32_tensors_reader(f, path, visit)
}

fn visit_f32_tensors_reader<R: Read + Seek>(
    reader: R,
    path: PathBuf,
    mut visit: impl FnMut(DdufTensor) -> Result<()>,
) -> Result<DdufMeta> {
    let mut zip = ZipArchive::new(reader).context("open dduf zip")?;
    let names: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();

    let mut model_index = None;
    let mut configs = HashMap::new();

    for name in names {
        if name.ends_with('/') {
            continue;
        }
        let bytes = read_member(&mut zip, &name)?;

        if name == "model_index.json" || name.ends_with("/model_index.json") {
            model_index = Some(serde_json::from_slice(&bytes).context("model_index.json")?);
            continue;
        }
        if name.ends_with("config.json") {
            configs.insert(
                component_of(&name),
                serde_json::from_slice(&bytes).with_context(|| format!("parse {name}"))?,
            );
            continue;
        }
        if !name.ends_with(".safetensors") {
            continue;
        }
        for t in decode_safetensors_f32(&name, &bytes)? {
            visit(t)?;
        }
        // `bytes` dropped here before the next member is read.
    }

    Ok(DdufMeta {
        path,
        model_index,
        configs,
    })
}

/// Cap for the up-front capacity *hint* when reading a ZIP member. `entry.size()`
/// is the archive's **declared** uncompressed size, which is attacker-controlled
/// (a zip bomb can claim gigabytes behind a few KB of deflate). Pre-allocating it
/// verbatim lets a tiny hostile `.dduf` force a multi-GB up-front allocation
/// (capacity-overflow panic / OOM) before any backing bytes are read.
const MEMBER_CAP_HINT: u64 = 64 * 1024 * 1024; // 64 MiB

fn read_member<R: Read + Seek>(zip: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let mut entry = zip
        .by_name(name)
        .with_context(|| format!("zip member {name}"))?;
    // Clamp the hint to a sane bound; `read_to_end` still grows the Vec to fit the
    // real decompressed bytes, so legitimate large safetensors members work — we
    // just never trust the declared size for the initial allocation.
    let hint = entry.size().min(MEMBER_CAP_HINT) as usize;
    let mut bytes = Vec::with_capacity(hint);
    entry
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {name}"))?;
    Ok(bytes)
}

fn decode_safetensors_f32(member: &str, bytes: &[u8]) -> Result<Vec<DdufTensor>> {
    let st = SafeTensors::deserialize(bytes).with_context(|| format!("safetensors in {member}"))?;
    let component = component_of(member);
    let mut out = Vec::with_capacity(st.len());
    for tname in st.names() {
        let view = st.tensor(tname)?;
        let data =
            decode_f32(view.data(), view.dtype()).with_context(|| format!("{member}#{tname}"))?;
        let qualified = format!("{component}/{tname}");
        out.push(DdufTensor {
            name: qualified,
            shape: view.shape().to_vec(),
            data,
            member: member.to_string(),
        });
    }
    Ok(out)
}

fn component_of(member: &str) -> String {
    let member = member.trim_start_matches("./");
    match member.split_once('/') {
        Some((comp, _)) if !comp.is_empty() => comp.to_string(),
        _ => ".".to_string(),
    }
}

fn decode_f32(raw: &[u8], dt: safetensors::Dtype) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    Ok(match dt {
        D::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F16 => raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::F64 => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::I32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::U8 => raw.iter().map(|&b| b as f32).collect(),
        other => bail!("dduf: cannot decode dtype {other:?} to f32"),
    })
}

/// One tensor kept in its on-disk dtype (no f32 widen).
#[derive(Debug, Clone)]
pub struct DdufNativeTensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// `f32` / `f16` / `bf16` / `f64` / `i32` / `u8`.
    pub encoding: String,
    pub data: Vec<u8>,
    pub member: String,
}

/// Load DDUF tensors without widening half/float to f32 (space-preserving).
///
/// DDUF does not carry MLX-style affine/mxfp packs — “packed” here means
/// **keep native dtype bytes**. Use [`load_f32_map`] when bake needs floats.
pub fn load_native(path: impl AsRef<Path>) -> Result<HashMap<String, DdufNativeTensor>> {
    let path = path.as_ref();
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip = ZipArchive::new(f).context("open dduf zip")?;
    let mut out = HashMap::new();
    let names: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().map(|e| e.name().to_string()))
        .collect();
    for name in names {
        if !name.ends_with(".safetensors") {
            continue;
        }
        let bytes = read_member(&mut zip, &name)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let component = component_of(&name);
        for tname in st.names() {
            let view = st.tensor(tname)?;
            let encoding = dtype_encoding(view.dtype())?;
            let qualified = format!("{component}/{tname}");
            out.insert(
                qualified.clone(),
                DdufNativeTensor {
                    name: qualified,
                    shape: view.shape().to_vec(),
                    encoding: encoding.into(),
                    data: view.data().to_vec(),
                    member: name.clone(),
                },
            );
        }
        // Member bytes dropped before the next safetensors file.
    }
    Ok(out)
}

fn dtype_encoding(dt: safetensors::Dtype) -> Result<&'static str> {
    use safetensors::Dtype as D;
    Ok(match dt {
        D::F32 => "f32",
        D::F16 => "f16",
        D::BF16 => "bf16",
        D::F64 => "f64",
        D::I32 => "i32",
        D::U8 => "u8",
        other => bail!("dduf native: unsupported dtype {other:?}"),
    })
}

/// Convenience: open and return the dense f32 map (shapes discarded).
pub fn load_f32_map(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>> {
    Ok(DdufFile::open(path)?.into_f32_map())
}

/// Dense f32 map with shapes preserved.
pub fn load_shaped_f32(path: impl AsRef<Path>) -> Result<HashMap<String, ShapedF32>> {
    Ok(DdufFile::open(path)?.into_shaped_f32())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, serialize};
    use std::io::Write;

    fn write_minimal_dduf(dir: &Path) -> PathBuf {
        let data: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let tensor = safetensors::tensor::TensorView::new(Dtype::F32, vec![2, 2], bytes).unwrap();
        let mut map: std::collections::BTreeMap<String, safetensors::tensor::TensorView<'_>> =
            std::collections::BTreeMap::new();
        map.insert("weight".into(), tensor);
        let st_bytes = serialize(map, None).unwrap();

        let dduf = dir.join("model.dduf");
        {
            let f = File::create(&dduf).unwrap();
            let mut zip = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("model_index.json", opts).unwrap();
            zip.write_all(br#"{"_class_name":"Test"}"#).unwrap();
            zip.start_file("transformer/diffusion_pytorch_model.safetensors", opts)
                .unwrap();
            zip.write_all(&st_bytes).unwrap();
            zip.finish().unwrap();
        }
        dduf
    }

    #[test]
    fn roundtrip_minimal_dduf() {
        let dir = tempfile::tempdir().unwrap();
        let dduf = write_minimal_dduf(dir.path());

        let file = DdufFile::open(&dduf).unwrap();
        assert!(file.model_index.is_some());
        let t = file.get("transformer/weight").unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape, vec![2, 2]);
    }

    #[test]
    fn shaped_f32_preserves_dims() {
        let dir = tempfile::tempdir().unwrap();
        let dduf = write_minimal_dduf(dir.path());
        let shaped = load_shaped_f32(&dduf).unwrap();
        let (shape, data) = shaped.get("transformer/weight").unwrap();
        assert_eq!(shape, &vec![2, 2]);
        assert_eq!(data, &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn visit_streams_member() {
        let dir = tempfile::tempdir().unwrap();
        let dduf = write_minimal_dduf(dir.path());
        let mut n = 0;
        let meta = visit_f32_tensors(&dduf, |t| {
            assert_eq!(t.shape, vec![2, 2]);
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 1);
        assert!(meta.model_index.is_some());
    }
}
