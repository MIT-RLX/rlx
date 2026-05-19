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

//! `pyrlx.Embed` — high-level loader for BERT / NomicBERT / NomicVision
//! across any RLX backend. Replicates `rlx_models::embed::compile_model`'s
//! pipeline but with a user-chosen device (the upstream version hardcodes
//! `Device::Cpu`).
//!
//! Tokenization stays on the Python side — feed pre-tokenized
//! `input_ids` / `attention_mask` numpy arrays into `forward()`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rlx_models::{
    BertConfig, NomicBertConfig, NomicVisionConfig, WeightMap, build_bert_graph_sized,
    build_nomic_graph_sized, build_vision_graph_sized,
};
use rlx_runtime::{CompiledGraph, Precision, Session};

use crate::device::{device_label, parse_device};

#[derive(Clone, Copy, Debug)]
enum Arch {
    Bert,
    NomicBert,
    NomicVision,
}

fn detect_arch(config_path: &Path) -> PyResult<Arch> {
    let data = std::fs::read_to_string(config_path)
        .map_err(|e| PyRuntimeError::new_err(format!("reading {config_path:?}: {e}")))?;
    let json: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("parsing config.json: {e}")))?;
    if json.get("img_size").is_some() && json.get("patch_size").is_some() {
        return Ok(Arch::NomicVision);
    }
    if json.get("rotary_emb_base").is_some() || json.get("rotary_emb_fraction").is_some() {
        return Ok(Arch::NomicBert);
    }
    Ok(Arch::Bert)
}

fn parse_precision(s: &str) -> PyResult<Precision> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Precision::F32,
        "f16" | "float16" | "half" => Precision::F16,
        "bf16" | "bfloat16" => Precision::BF16,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown precision '{other}'"
            )));
        }
    })
}

fn compile_model_with_device(
    arch: Arch,
    config_path: &Path,
    weights_path: &str,
    batch: usize,
    seq: usize,
    device: rlx_runtime::Device,
    precision: Precision,
) -> PyResult<(usize, CompiledGraph)> {
    let mut wm = WeightMap::from_file(weights_path)
        .map_err(|e| PyRuntimeError::new_err(format!("loading weights: {e}")))?;

    let (graph, params, hidden_size) = match arch {
        Arch::Bert => {
            let cfg = BertConfig::from_file(config_path)
                .map_err(|e| PyRuntimeError::new_err(format!("BertConfig: {e}")))?;
            let hs = cfg.hidden_size;
            let (g, p) = build_bert_graph_sized(&cfg, &mut wm, batch, seq)
                .map_err(|e| PyRuntimeError::new_err(format!("build_bert_graph: {e}")))?;
            (g, p, hs)
        }
        Arch::NomicBert => {
            let cfg = NomicBertConfig::from_file(config_path)
                .map_err(|e| PyRuntimeError::new_err(format!("NomicBertConfig: {e}")))?;
            let hs = cfg.hidden_size;
            let (g, p) = build_nomic_graph_sized(&cfg, &mut wm, batch, seq)
                .map_err(|e| PyRuntimeError::new_err(format!("build_nomic_graph: {e}")))?;
            (g, p, hs)
        }
        Arch::NomicVision => {
            let cfg = NomicVisionConfig::from_file(config_path)
                .map_err(|e| PyRuntimeError::new_err(format!("NomicVisionConfig: {e}")))?;
            let hs = cfg.hidden_size;
            let (g, p, _pre) = build_vision_graph_sized(&cfg, &mut wm, batch)
                .map_err(|e| PyRuntimeError::new_err(format!("build_vision_graph: {e}")))?;
            (g, p, hs)
        }
    };

    let session = Session::new_with_precision(device, precision);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    Ok((hidden_size, compiled))
}

#[pyclass(name = "Embed", module = "pyrlx._pyrlx")]
pub(crate) struct PyEmbed {
    compiled: CompiledGraph,
    arch: Arch,
    hidden_size: usize,
    compiled_bs: (usize, usize),
    config_path: PathBuf,
    weights_path: String,
    device: rlx_runtime::Device,
    precision: Precision,
}

#[pymethods]
impl PyEmbed {
    /// Load from a local directory containing `config.json` + `model.safetensors`.
    #[staticmethod]
    #[pyo3(signature = (path, device = "cpu", precision = "f32", batch = 1, seq = 1))]
    fn from_dir(
        path: &str,
        device: &str,
        precision: &str,
        batch: usize,
        seq: usize,
    ) -> PyResult<Self> {
        let dir = Path::new(path);
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let wt_str = weights_path
            .to_str()
            .ok_or_else(|| PyValueError::new_err("non-utf8 weights path"))?
            .to_string();

        let arch = detect_arch(&config_path)?;
        let dev = parse_device(device)?;
        if !rlx_runtime::is_available(dev) {
            return Err(PyRuntimeError::new_err(format!(
                "device '{device}' not available — rebuild pyrlx with --features {device}"
            )));
        }
        let prec = parse_precision(precision)?;
        let (hidden_size, compiled) =
            compile_model_with_device(arch, &config_path, &wt_str, batch, seq, dev, prec)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            compiled_bs: (batch, seq),
            config_path,
            weights_path: wt_str,
            device: dev,
            precision: prec,
        })
    }

    /// Load from a HuggingFace repo id (downloads if not cached).
    /// Requires the `hf-download` cargo feature; otherwise raises.
    #[staticmethod]
    #[pyo3(signature = (repo_id, device = "cpu", precision = "f32", batch = 1, seq = 1))]
    fn from_pretrained(
        repo_id: &str,
        device: &str,
        precision: &str,
        batch: usize,
        seq: usize,
    ) -> PyResult<Self> {
        #[cfg(feature = "hf-download")]
        {
            let repo = hf_hub::api::sync::ApiBuilder::new()
                .with_progress(true)
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("hf-hub: {e}")))?
                .model(repo_id.to_string());
            let config_file = repo
                .get("config.json")
                .map_err(|e| PyRuntimeError::new_err(format!("downloading config.json: {e}")))?;
            let weights_file = repo.get("model.safetensors").map_err(|e| {
                PyRuntimeError::new_err(format!("downloading model.safetensors: {e}"))
            })?;

            let arch = detect_arch(&config_file)?;
            let dev = parse_device(device)?;
            if !rlx_runtime::is_available(dev) {
                return Err(PyRuntimeError::new_err(format!(
                    "device '{device}' not available — rebuild pyrlx with --features {device}"
                )));
            }
            let prec = parse_precision(precision)?;
            let wt_str = weights_file
                .to_str()
                .ok_or_else(|| PyValueError::new_err("non-utf8 weights path"))?
                .to_string();
            let (hidden_size, compiled) =
                compile_model_with_device(arch, &config_file, &wt_str, batch, seq, dev, prec)?;

            Ok(Self {
                compiled,
                arch,
                hidden_size,
                compiled_bs: (batch, seq),
                config_path: config_file,
                weights_path: wt_str,
                device: dev,
                precision: prec,
            })
        }
        #[cfg(not(feature = "hf-download"))]
        {
            let _ = (repo_id, device, precision, batch, seq);
            Err(PyRuntimeError::new_err(
                "Embed.from_pretrained requires the 'hf-download' cargo feature — \
                 rebuild pyrlx with `maturin develop --features hf-download`",
            ))
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.hidden_size
    }

    #[getter]
    fn arch(&self) -> &'static str {
        match self.arch {
            Arch::Bert => "bert",
            Arch::NomicBert => "nomic-bert",
            Arch::NomicVision => "nomic-vision",
        }
    }

    #[getter]
    fn device(&self) -> &'static str {
        device_label(self.device)
    }

    #[getter]
    fn batch_seq(&self) -> (usize, usize) {
        self.compiled_bs
    }

    /// Run a forward pass with pre-tokenized inputs.
    ///
    /// `inputs` is a dict of name → numpy.ndarray (float32, C-contiguous).
    /// If `batch`/`seq` differ from the compiled extents the model is
    /// recompiled — same lazy-recompile contract as `RlxEmbed`.
    /// Returns a list of outputs, each reshaped to the declared shape.
    #[pyo3(signature = (inputs, batch, seq))]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        batch: usize,
        seq: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        if (batch, seq) != self.compiled_bs {
            let (_, compiled) = compile_model_with_device(
                self.arch,
                &self.config_path,
                &self.weights_path,
                batch,
                seq,
                self.device,
                self.precision,
            )?;
            self.compiled = compiled;
            self.compiled_bs = (batch, seq);
        }

        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (k, v) in inputs.iter() {
            let name: String = k.extract()?;
            let arr = v.downcast::<PyArrayDyn<f32>>().map_err(|_| {
                PyValueError::new_err(format!(
                    "input '{name}': expected numpy.ndarray of dtype float32"
                ))
            })?;
            if !arr.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "input '{name}': array must be C-contiguous"
                )));
            }
            let slice = unsafe { arr.as_slice()? };
            owned.push((name, slice.to_vec()));
        }
        let pairs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = self.compiled.run(&pairs);

        let list = PyList::empty_bound(py);
        for out in outs {
            list.append(out.into_pyarray_bound(py))?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Embed arch={} dim={} device={} compiled_bs={:?}>",
            self.arch(),
            self.hidden_size,
            device_label(self.device),
            self.compiled_bs
        )
    }
}

// HashMap brought in just to keep error help hints attached.
#[allow(dead_code)]
fn _silence(_: HashMap<String, Vec<f32>>) {}
