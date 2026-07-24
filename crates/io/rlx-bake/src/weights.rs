// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Weight loading for bake — **f32-first is an explicit go / no-go**.
//!
//! # Space / speed policy
//!
//! | Verdict | When | Effect |
//! |---------|------|--------|
//! | **f32-first GO** | Dense MatMul graphs; `--opt exact/size` ternary/Q8 rewrite;
//!   plain safetensors; policy=`f32` | Decode to f32, then bake may re-pack.
//!   Larger RAM at load; optimizers see floats. |
//! | **f32-first NO-GO** | MLX affine/mxfp packs; `--weights-policy packed`;
//!   DDUF native half (`f16`/`bf16`) under auto/packed | Keep source encoding —
//!   better disk/mmap size and dequant-matmul throughput. |
//!
//! `auto` picks **NO-GO** when MLX packs (or DDUF half) are present and bake is
//! not asking to re-quantize (`ternary`/`quant` off); otherwise **GO**.

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{PackedLinearBinding, collect_packed_linears, load_path};
use std::collections::HashMap;
use std::path::Path;

use crate::BakeOptions;

/// How `--weights` materialize into the bake pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightLoadPolicy {
    /// Always decode to f32 (legacy default behavior).
    #[default]
    F32First,
    /// Keep MLX packs / native DDUF dtypes when present.
    KeepPacked,
    /// Prefer packs when beneficial; else f32-first.
    Auto,
}

impl WeightLoadPolicy {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "f32-first" | "dense" => Ok(Self::F32First),
            "packed" | "keep-packed" | "native" => Ok(Self::KeepPacked),
            "auto" => Ok(Self::Auto),
            other => bail!("unknown --weights-policy {other:?}; expected f32|packed|auto"),
        }
    }
}

/// Explicit answer to “should bake widen weights to f32?”
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum F32FirstVerdict {
    Go { reason: &'static str },
    NoGo { reason: &'static str },
}

impl F32FirstVerdict {
    pub fn is_go(&self) -> bool {
        matches!(self, Self::Go { .. })
    }

    pub fn reason(&self) -> &'static str {
        match self {
            Self::Go { reason } | Self::NoGo { reason } => reason,
        }
    }
}

/// What we detected on disk (feeds [`f32_first_verdict`]).
#[derive(Debug, Clone, Default)]
pub struct WeightSourceInfo {
    pub has_mlx_packs: bool,
    pub has_native_half: bool,
    pub path_kind: WeightPathKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightPathKind {
    #[default]
    Unknown,
    Mlx,
    Safetensors,
    Dduf,
    Npz,
}

/// Decide f32-first for this source + bake options.
pub fn f32_first_verdict(
    policy: WeightLoadPolicy,
    source: &WeightSourceInfo,
    opts: &BakeOptions,
) -> F32FirstVerdict {
    if opts.ternary || opts.quant {
        return F32FirstVerdict::Go {
            reason: "bake ternary/quant rewrites need f32 MatMul weights",
        };
    }
    match policy {
        WeightLoadPolicy::F32First => F32FirstVerdict::Go {
            reason: "weights-policy=f32 (explicit)",
        },
        WeightLoadPolicy::KeepPacked => {
            if source.has_mlx_packs || source.has_native_half {
                F32FirstVerdict::NoGo {
                    reason: "weights-policy=packed — keep source encoding for space/speed",
                }
            } else {
                F32FirstVerdict::Go {
                    reason: "weights-policy=packed but source has no packs/half — fall back to f32",
                }
            }
        }
        WeightLoadPolicy::Auto => {
            if source.has_mlx_packs {
                F32FirstVerdict::NoGo {
                    reason: "auto: MLX packs present and ternary/quant off — keep packed",
                }
            } else if source.has_native_half && source.path_kind == WeightPathKind::Dduf {
                F32FirstVerdict::NoGo {
                    reason: "auto: DDUF half tensors — keep native dtype bytes",
                }
            } else {
                F32FirstVerdict::Go {
                    reason: "auto: no beneficial packs — decode to f32",
                }
            }
        }
    }
}

/// One weight row ready for the bake / RLXP table (may be packed).
#[derive(Debug, Clone)]
pub struct LoadedWeight {
    pub name: String,
    pub shape: Vec<usize>,
    /// Scheme / encoding string (`f32`, `f16`, `mlx_affine/4/64`, …).
    pub encoding: String,
    pub data: Vec<u8>,
}

/// Result of loading `--weights` under a policy.
#[derive(Debug, Clone, Default)]
pub struct WeightBundle {
    /// Dense f32 bindings for Param specialization (may be empty if fully packed).
    pub f32: HashMap<String, Vec<f32>>,
    /// Packed / native-dtype rows for the weight table + DequantMatMul.
    pub packed: Vec<LoadedWeight>,
    pub source: WeightSourceInfo,
    pub verdict: Option<F32FirstVerdict>,
}

/// Load every tensor, decoding to f32 (legacy).
pub fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    load_weights_f32(path)
}

/// Load weights from any supported format into a name → f32 map.
pub fn load_weights_f32(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    Ok(load_weights(path, WeightLoadPolicy::F32First, &BakeOptions::default())?.f32)
}

/// Load weights under [`WeightLoadPolicy`], recording an explicit f32-first verdict.
pub fn load_weights(
    path: &Path,
    policy: WeightLoadPolicy,
    opts: &BakeOptions,
) -> Result<WeightBundle> {
    if path.is_dir()
        || matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("npz") | Some("npy")
        )
    {
        return load_mlx_bundle(path, policy, opts);
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => {
            // Prefer sibling MLX dir load when config.json marks quantization.
            if let Some(parent) = path.parent() {
                if parent.join("config.json").is_file() {
                    if let Ok(probe) = load_path(parent) {
                        if probe.quant_scheme().is_some() {
                            return load_mlx_bundle(parent, policy, opts);
                        }
                    }
                }
            }
            let source = WeightSourceInfo {
                path_kind: WeightPathKind::Safetensors,
                ..Default::default()
            };
            let verdict = f32_first_verdict(policy, &source, opts);
            Ok(WeightBundle {
                f32: load_safetensors_file_f32(path)?,
                packed: Vec::new(),
                source,
                verdict: Some(verdict),
            })
        }
        Some("dduf") => load_dduf_bundle(path, policy, opts),
        Some(other) => bail!(
            "unsupported weights extension .{other} on {}",
            path.display()
        ),
        None => bail!("weights path has no extension: {}", path.display()),
    }
}

fn load_mlx_bundle(
    path: &Path,
    policy: WeightLoadPolicy,
    opts: &BakeOptions,
) -> Result<WeightBundle> {
    let mut w = load_path(path).with_context(|| format!("open mlx {}", path.display()))?;
    let source = WeightSourceInfo {
        has_mlx_packs: w.quant_scheme().is_some(),
        has_native_half: false,
        path_kind: WeightPathKind::Mlx,
    };
    let verdict = f32_first_verdict(policy, &source, opts);
    if !verdict.is_go() && source.has_mlx_packs {
        let mut bundle = WeightBundle {
            source,
            verdict: Some(verdict),
            ..Default::default()
        };
        let linears = collect_packed_linears(&mut w)?;
        for b in &linears {
            push_mlx_pack(&mut bundle.packed, b);
        }
        for name in w.logical_keys() {
            if let Ok((data, shape)) = w.take_dense_f32(&name) {
                bundle.f32.insert(name.clone(), data.clone());
                bundle.packed.push(LoadedWeight {
                    name,
                    shape,
                    encoding: "f32".into(),
                    data: f32_le_bytes(&data),
                });
            }
        }
        return Ok(bundle);
    }
    Ok(WeightBundle {
        f32: w.into_f32_map()?,
        packed: Vec::new(),
        source,
        verdict: Some(verdict),
    })
}

fn load_dduf_bundle(
    path: &Path,
    policy: WeightLoadPolicy,
    opts: &BakeOptions,
) -> Result<WeightBundle> {
    // Peek for half dtypes.
    let native = rlx_dduf::load_native(path)?;
    let has_native_half = native
        .values()
        .any(|t| t.encoding == "f16" || t.encoding == "bf16");
    let source = WeightSourceInfo {
        has_mlx_packs: false,
        has_native_half,
        path_kind: WeightPathKind::Dduf,
    };
    let verdict = f32_first_verdict(policy, &source, opts);
    if !verdict.is_go() {
        let mut bundle = WeightBundle {
            source,
            verdict: Some(verdict),
            ..Default::default()
        };
        for (name, t) in native {
            if t.encoding == "f32" {
                bundle.f32.insert(name.clone(), bytes_to_f32(&t.data));
            }
            bundle.packed.push(LoadedWeight {
                name,
                shape: t.shape,
                encoding: t.encoding,
                data: t.data,
            });
        }
        return Ok(bundle);
    }
    Ok(WeightBundle {
        f32: rlx_dduf::load_f32_map(path)?,
        packed: Vec::new(),
        source,
        verdict: Some(verdict),
    })
}

fn push_mlx_pack(out: &mut Vec<LoadedWeight>, b: &PackedLinearBinding) {
    let n = b.packed.out_shape[0];
    let n_groups = b.packed.n_groups().max(1);
    out.push(LoadedWeight {
        name: format!("{}.weight", b.name),
        shape: b.packed.out_shape.clone(),
        encoding: b.packed.scheme.to_string(),
        data: b.packed.w_q.clone(),
    });
    let scales_enc = if matches!(b.packed.scheme, QuantScheme::MlxAffine { .. }) {
        "f32"
    } else {
        "u8"
    };
    out.push(LoadedWeight {
        name: format!("{}.scales", b.name),
        shape: vec![n, n_groups],
        encoding: scales_enc.into(),
        data: b.packed.scales.clone(),
    });
    if matches!(b.packed.scheme, QuantScheme::MlxAffine { .. }) {
        out.push(LoadedWeight {
            name: format!("{}.biases", b.name),
            shape: vec![n, n_groups],
            encoding: "f32".into(),
            data: b.packed.biases.clone(),
        });
    }
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_safetensors_file_f32(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BakeProfile;

    #[test]
    fn verdict_ternary_forces_f32() {
        let opts = BakeOptions::from_profile(BakeProfile::Exact);
        let src = WeightSourceInfo {
            has_mlx_packs: true,
            ..Default::default()
        };
        assert!(f32_first_verdict(WeightLoadPolicy::KeepPacked, &src, &opts).is_go());
    }

    #[test]
    fn verdict_auto_keeps_mlx_packs() {
        let opts = BakeOptions::from_profile(BakeProfile::Merge);
        let src = WeightSourceInfo {
            has_mlx_packs: true,
            path_kind: WeightPathKind::Mlx,
            ..Default::default()
        };
        assert!(!f32_first_verdict(WeightLoadPolicy::Auto, &src, &opts).is_go());
    }
}
