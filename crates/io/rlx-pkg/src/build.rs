// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A package built for a device: each portable function gains a version
//! rewritten for it ([`rlx_ir::retarget`]), `step` beside `step:metal`, and
//! the runtime runs the one for the device a model opens on.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::Package;

/// `reference` beside a file package, inside a directory one.
fn beside(package: &Path, reference: &str) -> PathBuf {
    if Path::new(reference).is_absolute() {
        return PathBuf::from(reference);
    }
    let base = if package.is_dir() {
        package
    } else {
        package.parent().unwrap_or(Path::new("."))
    };
    base.join(reference)
}

/// `to` from the directory `from`, both absolute: `../ckpt/model.safetensors`.
fn relative(from: &Path, to: &Path) -> PathBuf {
    let (f, t): (Vec<_>, Vec<_>) = (from.components().collect(), to.components().collect());
    let common = f.iter().zip(&t).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..f.len() {
        out.push("..");
    }
    for c in &t[common..] {
        out.push(c);
    }
    out
}

/// A model block's relative `weights` URIs, written for the package at
/// `src`, as the package at `dst` reads them.
pub fn rebase_weight_uris(model: &mut Value, src: &Path, dst: &Path) {
    let Some(sources) = model.get_mut("weights").and_then(Value::as_array_mut) else {
        return;
    };
    for s in sources {
        let Some(uri) = s.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let Some((scheme, rest)) = uri.split_once("://") else {
            continue;
        };
        if Path::new(rest).is_absolute() {
            continue;
        }
        let file = beside(src, rest);
        let file = file.canonicalize().unwrap_or(file);
        let base = if dst.is_dir() {
            dst.to_path_buf()
        } else {
            dst.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        let base = base.canonicalize().unwrap_or(base);
        s["uri"] = json!(format!("{scheme}://{}", relative(&base, &file).display()));
    }
}

/// What a build for one device rewrites in each function
/// ([`rlx_ir::retarget`]).
#[derive(Debug, Clone, PartialEq)]
pub struct TargetRecipe {
    /// The matmul weights' dtype (`F16` halves what a decode reads where the
    /// backend multiplies f16 weights natively).
    pub weight_dtype: Option<rlx_ir::DType>,
    /// Attention reads the shared K / V heads itself, without the
    /// `repeat_kv` broadcast.
    pub gqa_native: bool,
}

impl TargetRecipe {
    /// What is measured fastest on each device: Metal f16 weights (the
    /// Qwen3-0.6B step ~23 → ~14 ms a token on an M4 Pro, with GQA-native
    /// attention); the CPU GQA-native attention only (f16 weights are widened
    /// every step there); wgpu (`gpu`) the attention only. `None` for a device
    /// with no build yet.
    pub fn for_target(target: &str) -> Option<Self> {
        match target {
            "metal" => Some(Self {
                weight_dtype: Some(rlx_ir::DType::F16),
                gqa_native: true,
            }),
            // wgpu's attention takes the shared heads too (1.2e-7 against
            // the broadcast on Metal-backed wgpu).
            "cpu" | "gpu" => Some(Self {
                weight_dtype: None,
                gqa_native: true,
            }),
            _ => None,
        }
    }
}

/// What [`build_for_target`] added.
#[derive(Debug, Default)]
pub struct BuildReport {
    /// The functions built (`step:metal`, …).
    pub built: Vec<String>,
    pub weights_narrowed: usize,
    pub attention_rewired: usize,
}

/// Copy the package `src` to `dst` with each portable function also built
/// for `target` (`metal`, `cpu`): `step` gains `step:metal`, which a model
/// opened on that device runs instead. A function the recipe leaves as it
/// is gains nothing; a build from before is replaced. Weights, sidecars and
/// the model block are copied as they are.
pub fn build_for_target(
    src: &Path,
    dst: &Path,
    target: &str,
    recipe: &TargetRecipe,
) -> Result<BuildReport> {
    build_for_targets(src, dst, &[(target, recipe.clone())])
}

/// [`build_for_target`] for several devices at once, into one package.
pub fn build_for_targets(
    src: &Path,
    dst: &Path,
    targets: &[(&str, TargetRecipe)],
) -> Result<BuildReport> {
    let pack = Package::open(src)?;
    let mut module = pack
        .module_with(crate::MaterializeMode::HotOnly)
        .with_context(|| format!("graph of {}", src.display()))?;
    let mut report = BuildReport::default();
    for (target, recipe) in targets {
        build_one(&mut module, target, recipe, &mut report)?;
    }
    write_built(&pack, src, dst, module)?;
    Ok(report)
}

fn build_one(
    module: &mut rlx_ir::rlxmodel::Module,
    target: &str,
    recipe: &TargetRecipe,
    report: &mut BuildReport,
) -> Result<()> {
    let suffix = format!(":{target}");
    module.graphs.retain(|g| !g.name.ends_with(&suffix));
    let mut built = Vec::new();
    for g in module.graphs.iter().filter(|g| !g.name.contains(':')) {
        let mut out = g.clone();
        let mut changed = false;
        if recipe.gqa_native {
            let (rewired, n) =
                rlx_ir::retarget::gqa_native_attention(&out).map_err(anyhow::Error::msg)?;
            if n > 0 {
                out = rewired;
                report.attention_rewired += n;
                changed = true;
            }
        }
        if let Some(dt) = recipe.weight_dtype {
            let n = rlx_ir::retarget::matmul_weights_as(&mut out, dt);
            report.weights_narrowed += n;
            changed |= n > 0;
        }
        if changed {
            out.name = format!("{}{suffix}", g.name);
            report.built.push(out.name.clone());
            built.push(out);
        }
    }
    module.graphs.extend(built);
    Ok(())
}

fn write_built(
    pack: &Package,
    src: &Path,
    dst: &Path,
    module: rlx_ir::rlxmodel::Module,
) -> Result<()> {
    // Weights and sidecars byte for byte, packed schemes included.
    let mut packed = Vec::new();
    if let Some(index) = pack.weights_index() {
        for e in &index.tensors {
            packed.push(crate::PackedWeight::hot(
                e.name.clone(),
                e.shape.clone(),
                e.scheme.clone(),
                e.layout.clone(),
                pack.tensor_bytes(&e.name)?,
            ));
        }
    }
    let mut sidecars = Vec::new();
    for (id, media) in pack.sidecar_ids() {
        let bytes = pack.sidecar(&id)?;
        sidecars.push((
            id,
            media.unwrap_or_else(|| "application/octet-stream".into()),
            bytes,
        ));
    }
    let mut model = module.model.clone().unwrap_or_else(|| json!({}));
    // A referenced weight file stays where it is: its path, relative to
    // the package, is rebased onto `dst`.
    rebase_weight_uris(&mut model, src, dst);
    let (first, extra) = module
        .graphs
        .split_first()
        .context("build_for_target: the package has no graph")?;
    let opts = crate::WriteOptions {
        name: pack.manifest().name.clone(),
        producer: Some("rlx-pkg build".into()),
        model: Some(model),
        requires_rlx: module.requires.clone(),
        dims: module.dims.clone(),
        extra_graphs: extra.to_vec(),
        sidecars,
        ..Default::default()
    };
    crate::write_package(dst, first, &packed, &opts)
        .with_context(|| format!("writing {}", dst.display()))?;
    Ok(())
}
