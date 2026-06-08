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

//! [`OnnxModel`] — load `.onnx` and run via native RLX (`Session::compile`) by default.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rlx_runtime::Device;

use crate::io::{IoDesc, OnnxTensor};
use crate::level::OnnxCompileLevel;
use crate::native::NativeOnnx;

#[cfg(feature = "ort")]
use crate::session_ort::OrtOnnx;

/// How this model is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnnxExecBackend {
    /// Import ONNX → HIR → `Session::compile` on the chosen [`Device`].
    #[default]
    Native,
    /// ONNX Runtime (requires `ort` / `ort-fallback` feature).
    #[cfg(feature = "ort")]
    Ort,
}

/// Loaded ONNX model ready for inference.
pub struct OnnxModel {
    pub path: PathBuf,
    pub device: Device,
    pub backend: OnnxExecBackend,
    pub compile_level: OnnxCompileLevel,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
    /// ORT execution provider name when using [`OnnxExecBackend::Ort`].
    pub ort_ep: Option<String>,
    /// Extent used for unknown/dynamic ONNX dimensions (see [`Self::zero_inputs_sized`]).
    pub dynamic_dim: i64,
    inner: Inner,
}

enum Inner {
    Native(NativeOnnx),
    #[cfg(feature = "ort")]
    Ort(OrtOnnx),
}

impl OnnxModel {
    /// Load with native RLX execution at compile level 3 (default pipeline).
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::load_native(path, device, OnnxCompileLevel::Level3, 128)
    }

    /// Native RLX: import + compile on `device` at the given optimization level.
    pub fn load_native(
        path: impl AsRef<Path>,
        device: Device,
        level: OnnxCompileLevel,
        sequence_length: usize,
    ) -> Result<Self> {
        Self::load_with(
            path,
            device,
            OnnxExecBackend::Native,
            level,
            sequence_length,
        )
    }

    /// ONNX Runtime path (feature `ort` / `ort-fallback`).
    #[cfg(feature = "ort")]
    pub fn load_ort(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::load_with(
            path,
            device,
            OnnxExecBackend::Ort,
            OnnxCompileLevel::Level3,
            128,
        )
    }

    pub fn load_with(
        path: impl AsRef<Path>,
        device: Device,
        backend: OnnxExecBackend,
        level: OnnxCompileLevel,
        sequence_length: usize,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        match backend {
            OnnxExecBackend::Native => {
                let native = NativeOnnx::load(&path, device, level, sequence_length)?;
                Ok(Self {
                    inputs: native.inputs.clone(),
                    outputs: native.outputs.clone(),
                    path,
                    device,
                    backend,
                    compile_level: level,
                    ort_ep: None,
                    dynamic_dim: sequence_length as i64,
                    inner: Inner::Native(native),
                })
            }
            #[cfg(feature = "ort")]
            OnnxExecBackend::Ort => {
                let mut ort = OrtOnnx::load(&path, device)?;
                let inputs = ort.inputs.clone();
                let outputs = ort.outputs.clone();
                let ort_ep = ort.ort_ep.clone();
                Ok(Self {
                    path,
                    device,
                    backend,
                    compile_level: level,
                    inputs,
                    outputs,
                    ort_ep: Some(ort_ep),
                    dynamic_dim: ort.dynamic_dim,
                    inner: Inner::Ort(ort),
                })
            }
        }
    }

    pub fn run(&mut self, inputs: &HashMap<String, OnnxTensor>) -> Result<Vec<OnnxTensor>> {
        match &mut self.inner {
            Inner::Native(n) => n.run(inputs),
            #[cfg(feature = "ort")]
            Inner::Ort(o) => o.run(inputs),
        }
    }

    pub fn zero_inputs_sized(&mut self, dynamic_dim: i64) -> Result<HashMap<String, OnnxTensor>> {
        self.dynamic_dim = dynamic_dim.max(1);
        match &mut self.inner {
            Inner::Native(n) => n.zero_inputs_sized(self.dynamic_dim),
            #[cfg(feature = "ort")]
            Inner::Ort(o) => o.zero_inputs_sized(self.dynamic_dim),
        }
    }

    pub fn zero_inputs(&mut self) -> Result<HashMap<String, OnnxTensor>> {
        self.zero_inputs_sized(1)
    }

    pub fn print_io(&self) {
        println!("model: {}", self.path.display());
        println!(
            "device: {:?}  backend: {:?}  compile_level: {:?}",
            self.device, self.backend, self.compile_level
        );
        if let Some(ep) = &self.ort_ep {
            println!("ort_ep: {ep}");
        }
        println!("inputs:");
        for i in &self.inputs {
            println!("  {}  {:?}  {:?}", i.name, i.element_type, i.shape);
        }
        println!("outputs:");
        for o in &self.outputs {
            println!("  {}  {:?}  {:?}", o.name, o.element_type, o.shape);
        }
    }
}
