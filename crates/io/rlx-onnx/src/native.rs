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

//! Native RLX compile + execute for ONNX models (import → `Session::compile`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rlx_onnx_import::{
    ImportOptions, ImportReport, TypedParams, build_hir_from_bundle, build_hir_from_onnx_file,
    load_bundle,
};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::io::{self, IoDesc, OnnxElementType, OnnxTensor};
use crate::level::OnnxCompileLevel;

fn bundle_dir_for(path: &Path) -> Option<PathBuf> {
    let stem = path.file_stem()?.to_str()?;
    let parent = path.parent()?;
    for name in [format!("{stem}.rlx-bundle"), format!("{stem}_rlx_bundle")] {
        let dir = parent.join(&name);
        if dir.join("manifest.json").is_file() {
            return Some(dir);
        }
    }
    None
}

fn io_from_manifest(manifest: &rlx_onnx_import::BundleManifest) -> (Vec<IoDesc>, Vec<IoDesc>) {
    let map_io = |io: &rlx_onnx_import::IoMeta| IoDesc {
        name: io.name.clone(),
        element_type: OnnxElementType::from_dtype_str(&io.meta.dtype),
        shape: io
            .meta
            .shape
            .iter()
            .map(|v| match v {
                serde_json::Value::Number(n) => n.as_i64().filter(|&d| d > 0),
                _ => None,
            })
            .collect(),
    };
    (
        manifest.inputs.iter().map(map_io).collect(),
        manifest.outputs.iter().map(map_io).collect(),
    )
}

pub struct NativeOnnx {
    pub path: PathBuf,
    pub device: Device,
    pub compile_level: OnnxCompileLevel,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
    pub import_report: ImportReport,
    compiled: CompiledGraph,
}

impl NativeOnnx {
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
        level: OnnxCompileLevel,
        sequence_length: usize,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let opts = ImportOptions {
            sequence_length,
            ..ImportOptions::default()
        };

        let (hir, params, typed_params, report, manifest) = if let Some(dir) = bundle_dir_for(&path)
        {
            let bundle = load_bundle(&dir)?;
            let manifest = bundle.manifest.clone();
            let (hir, params, typed, report) = build_hir_from_bundle(&bundle, opts)?;
            (hir, params, typed, report, manifest)
        } else {
            let (hir, params, report, manifest) =
                build_hir_from_onnx_file(&path, opts).context("ONNX → HIR import")?;
            (hir, params, TypedParams::new(), report, manifest)
        };

        if report.skipped > 0 {
            eprintln!(
                "rlx-onnx: import skipped {} node(s); unsupported: {:?}",
                report.skipped, report.unsupported
            );
        }

        let (inputs, outputs) = io_from_manifest(&manifest);
        let options = level.to_compile_options();
        let session = Session::new(device);
        let mut compiled = session
            .compile_hir_with(hir, &options)
            .map_err(|e| anyhow::anyhow!("HIR lower/compile: {e}"))?;
        for (name, data) in params {
            compiled.set_param(&name, &data);
        }
        for (name, (bytes, dtype)) in typed_params {
            compiled.set_param_typed(&name, &bytes, dtype);
        }

        Ok(Self {
            path,
            device,
            compile_level: level,
            import_report: report,
            inputs,
            outputs,
            compiled,
        })
    }

    pub fn run(&mut self, inputs: &HashMap<String, OnnxTensor>) -> Result<Vec<OnnxTensor>> {
        let mut typed = Vec::with_capacity(self.inputs.len());
        for desc in &self.inputs {
            let tensor = inputs
                .get(&desc.name)
                .with_context(|| format!("missing input '{}'", desc.name))?;
            let (bytes, dtype) = io::tensor_to_typed_bytes(tensor, desc)?;
            typed.push((desc.name.as_str(), bytes, dtype));
        }
        let outs = self.compiled.run_typed(&typed);
        outs.into_iter()
            .map(|(bytes, dtype)| io::typed_bytes_to_tensor(&bytes, dtype))
            .collect()
    }

    pub fn zero_inputs_sized(&self, dynamic_dim: i64) -> Result<HashMap<String, OnnxTensor>> {
        io::zero_inputs_sized(&self.inputs, dynamic_dim)
    }
}
