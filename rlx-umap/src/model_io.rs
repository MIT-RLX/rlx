// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Model I/O in **safetensors** (default) and **GGUF** (F32), with legacy `.ruama` load.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use crate::config::UmapConfig;
use crate::encoder::mlp::ModelSpec;
use crate::serialize::{LoadedModel, ModelMetadata, SaveBundle};
use crate::utils::NormStats;
use crate::weights::WeightStore;

pub const META_FORMAT: &str = "rlx_umap.format";
pub const META_VERSION: &str = "rlx_umap.version";
pub const META_CONFIG: &str = "rlx_umap.config";
pub const META_SHAPES: &str = "rlx_umap.shapes";
pub const META_N_TRAIN: &str = "rlx_umap.n_train";
pub const META_N_FEATURES: &str = "rlx_umap.n_features";
pub const META_N_POS: &str = "rlx_umap.n_pos";
pub const META_N_NEG: &str = "rlx_umap.n_neg";
pub const META_NORM_MEAN: &str = "rlx_umap.norm_mean";
pub const META_NORM_STD: &str = "rlx_umap.norm_std";

pub const FORMAT_VERSION: &str = "1";
pub const EXT_SAFETENSORS: &str = "safetensors";
pub const EXT_GGUF: &str = "gguf";
/// Legacy archive extension (load still supported).
pub const EXT_RUAMA: &str = "ruama";

/// Default extension for new saves.
pub const MODEL_EXT: &str = EXT_SAFETENSORS;

/// Infer storage format from path extension (defaults to safetensors).
pub fn format_from_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some(EXT_GGUF) => EXT_GGUF,
        Some(EXT_RUAMA) => EXT_RUAMA,
        Some(EXT_SAFETENSORS) | Some("st") => EXT_SAFETENSORS,
        _ => EXT_SAFETENSORS,
    }
}

pub fn model_path(dir: impl AsRef<Path>, stem: &str) -> PathBuf {
    dir.as_ref().join(format!("{stem}.{MODEL_EXT}"))
}

pub fn model_path_with_ext(dir: impl AsRef<Path>, stem: &str, ext: &str) -> PathBuf {
    dir.as_ref().join(format!("{stem}.{ext}"))
}

/// Parameter shapes for the UMAP MLP (`umap_w*`, `umap_b*`, `umap_w_out`, `umap_b_out`).
pub fn weight_shapes(spec: &ModelSpec) -> HashMap<String, Vec<usize>> {
    let mut shapes = HashMap::new();
    let mut in_d = spec.input_dim;
    for (li, &hd) in spec.hidden.iter().enumerate() {
        shapes.insert(format!("umap_w{li}"), vec![in_d, hd]);
        shapes.insert(format!("umap_b{li}"), vec![hd]);
        in_d = hd;
    }
    shapes.insert("umap_w_out".into(), vec![in_d, spec.output_dim]);
    shapes.insert("umap_b_out".into(), vec![spec.output_dim]);
    shapes
}

fn bundle_metadata(bundle: &SaveBundle<'_>) -> HashMap<String, String> {
    let shapes = weight_shapes(&ModelSpec::from_config(
        bundle.config,
        bundle.meta.n_train,
        bundle.meta.n_features,
    ));
    let shapes_json = serde_json::to_string(&shapes).unwrap_or_else(|_| "{}".into());
    let config_json = serde_json::to_string(bundle.config).unwrap_or_default();
    let mean_json = serde_json::to_string(&bundle.norm.mean).unwrap_or_default();
    let std_json = serde_json::to_string(&bundle.norm.std).unwrap_or_default();

    HashMap::from([
        (META_FORMAT.into(), "safetensors".into()),
        (META_VERSION.into(), FORMAT_VERSION.into()),
        (META_CONFIG.into(), config_json),
        (META_SHAPES.into(), shapes_json),
        (META_N_TRAIN.into(), bundle.meta.n_train.to_string()),
        (META_N_FEATURES.into(), bundle.meta.n_features.to_string()),
        (META_N_POS.into(), bundle.meta.n_pos.to_string()),
        (META_N_NEG.into(), bundle.meta.n_neg.to_string()),
        (META_NORM_MEAN.into(), mean_json),
        (META_NORM_STD.into(), std_json),
    ])
}

fn parse_metadata(meta: &HashMap<String, String>) -> std::io::Result<LoadedModel> {
    let n_train = meta
        .get(META_N_TRAIN)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| meta_err("missing n_train"))?;
    let n_features = meta
        .get(META_N_FEATURES)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| meta_err("missing n_features"))?;
    let n_pos = meta
        .get(META_N_POS)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let n_neg = meta
        .get(META_N_NEG)
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_train * 5);

    let config = meta
        .get(META_CONFIG)
        .and_then(|s| serde_json::from_str::<UmapConfig>(s).ok());

    let mean: Vec<f64> = meta
        .get(META_NORM_MEAN)
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| vec![0.0; n_features]);
    let std: Vec<f64> = meta
        .get(META_NORM_STD)
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| vec![1.0; n_features]);

    Ok(LoadedModel {
        weights: WeightStore::default(), // filled by caller
        meta: ModelMetadata {
            n_train,
            n_features,
            n_pos,
            n_neg,
        },
        norm: NormStats { mean, std },
        config,
    })
}

fn meta_err(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

fn shapes_from_meta(
    meta: &HashMap<String, String>,
) -> std::io::Result<HashMap<String, Vec<usize>>> {
    let json = meta
        .get(META_SHAPES)
        .ok_or_else(|| meta_err("missing rlx_umap.shapes in metadata"))?;
    serde_json::from_str(json).map_err(|e| meta_err(&e.to_string()))
}

fn weights_from_named_f32(
    tensors: HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
) -> std::io::Result<WeightStore> {
    let mut w = WeightStore::default();
    for (name, data) in tensors {
        let expected: usize = shapes
            .get(&name)
            .ok_or_else(|| meta_err(&format!("unknown tensor {name}")))?
            .iter()
            .product();
        if data.len() != expected {
            return Err(meta_err(&format!(
                "tensor {name}: expected {expected} f32 values, got {}",
                data.len()
            )));
        }
        w.0.insert(name, data);
    }
    Ok(w)
}

// ─── safetensors ─────────────────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
mod safe {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use safetensors::{SafeTensors, serialize_to_file};

    pub fn save_weights(
        w: &WeightStore,
        shapes: &HashMap<String, Vec<usize>>,
        path: &Path,
        extra_meta: HashMap<String, String>,
    ) -> std::io::Result<()> {
        let mut meta = extra_meta;
        meta.entry(META_FORMAT.into())
            .or_insert_with(|| "safetensors".into());
        meta.entry(META_VERSION.into())
            .or_insert_with(|| FORMAT_VERSION.into());
        meta.insert(
            META_SHAPES.into(),
            serde_json::to_string(shapes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
        );

        let mut names: Vec<_> = w.0.keys().cloned().collect();
        names.sort();
        let mut packed: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(names.len());
        for name in names {
            let shape = shapes
                .get(&name)
                .cloned()
                .unwrap_or_else(|| vec![w.0[&name].len()]);
            let bytes: Vec<u8> = w.0[&name].iter().flat_map(|f| f.to_le_bytes()).collect();
            packed.push((name, shape, bytes));
        }
        let mut tensors: Vec<(String, TensorView<'_>)> = Vec::with_capacity(packed.len());
        for (name, shape, bytes) in &packed {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            tensors.push((name.clone(), view));
        }

        serialize_to_file(tensors, Some(meta), path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    pub fn load_file(path: &Path) -> std::io::Result<(WeightStore, HashMap<String, String>)> {
        let bytes = std::fs::read(path)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let meta: HashMap<String, String> = header.metadata().clone().unwrap_or_default();
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let shapes = shapes_from_meta(&meta)?;
        let mut tensors = HashMap::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            if view.dtype() != Dtype::F32 {
                return Err(meta_err(&format!("tensor {name}: expected F32")));
            }
            let n: usize = view.shape().iter().product();
            let mut data = vec![0f32; n];
            let raw = view.data();
            for (slot, chunk) in data.iter_mut().zip(raw.chunks_exact(4)) {
                *slot = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            tensors.insert(name.to_string(), data);
        }
        let weights = weights_from_named_f32(tensors, &shapes)?;
        Ok((weights, meta))
    }

    pub fn save_bundle(bundle: SaveBundle<'_>, path: &Path) -> std::io::Result<()> {
        let shapes = weight_shapes(&ModelSpec::from_config(
            bundle.config,
            bundle.meta.n_train,
            bundle.meta.n_features,
        ));
        save_weights(bundle.weights, &shapes, path, bundle_metadata(&bundle))
    }
}

// ─── GGUF (F32 write + rlx-gguf read) ────────────────────────────────────────

#[cfg(feature = "io-gguf")]
mod gguf_io {
    use super::*;
    use rlx_gguf::{GGUF_MAGIC, GgmlType, GgufFile};

    const GGUF_VERSION: u32 = 3;
    const ALIGN: u64 = 32;

    fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }
    fn write_string(w: &mut impl Write, s: &str) -> std::io::Result<()> {
        write_u64(w, s.len() as u64)?;
        w.write_all(s.as_bytes())
    }
    fn write_meta_string(w: &mut impl Write, key: &str, val: &str) -> std::io::Result<()> {
        write_string(w, key)?;
        write_u32(w, 8)?; // STRING
        write_string(w, val)
    }

    pub fn save_weights(
        w: &WeightStore,
        shapes: &HashMap<String, Vec<usize>>,
        path: &Path,
        extra_meta: HashMap<String, String>,
    ) -> std::io::Result<()> {
        let mut meta = extra_meta;
        meta.entry(META_FORMAT.into())
            .or_insert_with(|| "gguf".into());
        meta.entry(META_VERSION.into())
            .or_insert_with(|| FORMAT_VERSION.into());
        meta.insert(
            META_SHAPES.into(),
            serde_json::to_string(shapes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
        );

        let mut names: Vec<_> = w.0.keys().cloned().collect();
        names.sort();

        let mut data_blob = Vec::new();
        let mut tensor_infos: Vec<(String, Vec<usize>, u64)> = Vec::new();
        for name in &names {
            let offset = data_blob.len() as u64;
            let raw = &w.0[name];
            for &f in raw {
                data_blob.extend_from_slice(&f.to_le_bytes());
            }
            let shape = shapes.get(name).cloned().unwrap_or_else(|| vec![raw.len()]);
            tensor_infos.push((name.clone(), shape, offset));
        }

        let mut file = BufWriter::new(File::create(path)?);
        write_u32(&mut file, GGUF_MAGIC)?;
        write_u32(&mut file, GGUF_VERSION)?;
        write_u64(&mut file, tensor_infos.len() as u64)?;
        write_u64(&mut file, meta.len() as u64)?;

        let mut kv: Vec<_> = meta.into_iter().collect();
        kv.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, v) in &kv {
            write_meta_string(&mut file, k, v)?;
        }

        for (name, shape, offset) in &tensor_infos {
            write_string(&mut file, name)?;
            write_u32(&mut file, shape.len() as u32)?;
            for &d in shape {
                write_u64(&mut file, d as u64)?;
            }
            write_u32(&mut file, GgmlType::F32 as u32)?;
            write_u64(&mut file, *offset)?;
        }

        let pos = file.stream_position()?;
        let pad = (ALIGN - (pos % ALIGN)) % ALIGN;
        file.write_all(&vec![0u8; pad as usize])?;
        file.write_all(&data_blob)?;
        file.flush()?;
        Ok(())
    }

    pub fn load_file(path: &Path) -> std::io::Result<(WeightStore, HashMap<String, String>)> {
        let gf = GgufFile::from_path(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let mut meta = HashMap::new();
        for (k, v) in &gf.metadata {
            if let rlx_gguf::MetaValue::String(s) = v {
                meta.insert(k.clone(), s.clone());
            }
        }
        let shapes = shapes_from_meta(&meta)?;
        let mut tensors = HashMap::new();
        for name in gf.keys() {
            let (data, _shape) = gf
                .dequant_f32(name)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            tensors.insert(name.to_string(), data);
        }
        let weights = weights_from_named_f32(tensors, &shapes)?;
        Ok((weights, meta))
    }

    pub fn save_bundle(bundle: SaveBundle<'_>, path: &Path) -> std::io::Result<()> {
        let shapes = weight_shapes(&ModelSpec::from_config(
            bundle.config,
            bundle.meta.n_train,
            bundle.meta.n_features,
        ));
        save_weights(bundle.weights, &shapes, path, bundle_metadata(&bundle))
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

pub fn save_weights(
    w: &WeightStore,
    spec: &ModelSpec,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let path = path.as_ref();
    let shapes = weight_shapes(spec);
    match format_from_path(path) {
        #[cfg(feature = "safetensors")]
        EXT_SAFETENSORS => safe::save_weights(w, &shapes, path, HashMap::new()),
        #[cfg(feature = "io-gguf")]
        EXT_GGUF => gguf_io::save_weights(w, &shapes, path, HashMap::new()),
        EXT_RUAMA => crate::serialize::save_weights_ruama(w, path),
        #[cfg(not(feature = "safetensors"))]
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "build with feature `safetensors`",
        )),
        #[allow(unreachable_patterns)]
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported extension: {other}"),
        )),
    }
}

pub fn load_weights(path: impl AsRef<Path>) -> std::io::Result<WeightStore> {
    let path = path.as_ref();
    if is_ruama(path)? {
        return crate::serialize::load_weights_ruama(path);
    }
    #[cfg(feature = "safetensors")]
    if format_from_path(path) == EXT_SAFETENSORS || looks_like_safetensors(path)? {
        let (w, _) = safe::load_file(path)?;
        return Ok(w);
    }
    #[cfg(feature = "io-gguf")]
    if format_from_path(path) == EXT_GGUF || looks_like_gguf(path)? {
        let (w, _) = gguf_io::load_file(path)?;
        return Ok(w);
    }
    // Fallback: try safetensors then gguf
    #[cfg(feature = "safetensors")]
    {
        if let Ok((w, _)) = safe::load_file(path) {
            return Ok(w);
        }
    }
    #[cfg(feature = "io-gguf")]
    {
        if let Ok((w, _)) = gguf_io::load_file(path) {
            return Ok(w);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "unrecognized model file (expected .safetensors, .gguf, or .ruama)",
    ))
}

pub fn save_model(bundle: SaveBundle<'_>, path: impl AsRef<Path>) -> std::io::Result<()> {
    let path = path.as_ref();
    match format_from_path(path) {
        #[cfg(feature = "safetensors")]
        EXT_SAFETENSORS => safe::save_bundle(bundle, path),
        #[cfg(feature = "io-gguf")]
        EXT_GGUF => gguf_io::save_bundle(bundle, path),
        EXT_RUAMA => crate::serialize::save_model(bundle, path),
        #[cfg(not(feature = "safetensors"))]
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "build with feature `safetensors`",
        )),
        #[allow(unreachable_patterns)]
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unsupported extension: {other}"),
        )),
    }
}

pub fn load_model(path: impl AsRef<Path>) -> std::io::Result<LoadedModel> {
    let path = path.as_ref();
    if is_ruama(path)? {
        return crate::serialize::load_legacy_ruama(path);
    }

    #[cfg(feature = "safetensors")]
    if format_from_path(path) == EXT_SAFETENSORS || looks_like_safetensors(path)? {
        let (weights, meta_map) = safe::load_file(path)?;
        let mut loaded = parse_metadata(&meta_map)?;
        loaded.weights = weights;
        return Ok(loaded);
    }

    #[cfg(feature = "io-gguf")]
    if format_from_path(path) == EXT_GGUF || looks_like_gguf(path)? {
        let (weights, meta_map) = gguf_io::load_file(path)?;
        let mut loaded = parse_metadata(&meta_map)?;
        loaded.weights = weights;
        return Ok(loaded);
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "unrecognized model file",
    ))
}

pub(crate) fn is_ruama(path: &Path) -> std::io::Result<bool> {
    if path.extension().and_then(|e| e.to_str()) == Some(EXT_RUAMA) {
        return Ok(true);
    }
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    if f.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    Ok(&magic == b"RUMA")
}

fn looks_like_gguf(path: &Path) -> std::io::Result<bool> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    f.read_exact(&mut magic)?;
    Ok(magic == [0x47, 0x47, 0x55, 0x46]) // "GGUF" LE
}

fn looks_like_safetensors(path: &Path) -> std::io::Result<bool> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Ok(false);
    }
    let len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    Ok((8 + len as usize) <= bytes.len() && len < 1 << 30)
}
