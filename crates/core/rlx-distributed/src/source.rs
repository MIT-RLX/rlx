// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The **model-agnostic weight seam**. `rlx-distributed` never knows how a
//! model's weights are stored — a stage worker asks a [`ParamSource`] for each
//! parameter by name, and the source (implemented by a *model* crate, e.g.
//! wrapping an mlx / GGUF / safetensors loader) returns it as dense f32 or as
//! packed/quantized bytes. This is what keeps the crate model-agnostic: RLX
//! provides the API, `rlx-models` provides the weights.
//!
//! Ergonomics: a plain `HashMap<String, Vec<f32>>` is a `ParamSource` (great for
//! tests and small models), and so is any closure `FnMut(&str) -> Option<Param>`
//! (great for adapting an existing loader in one line):
//!
//! ```ignore
//! // adapt any loader without a newtype:
//! let mut src = |name: &str| loader.take_typed(name).map(|(b, dt)| Param::typed(b, dt));
//! serve_stage(addr, stage, &mut src, Device::Cpu, &opts, 1)?;
//! ```

use rlx_ir::DType;
use std::collections::HashMap;

/// One parameter value: dense f32, or packed bytes with a dtype (quantized /
/// f16 / bf16 weights fed to the runtime without a host cast).
#[derive(Clone, Debug)]
pub enum Param {
    F32(Vec<f32>),
    Typed(Vec<u8>, DType),
}

impl Param {
    pub fn f32(v: Vec<f32>) -> Self {
        Param::F32(v)
    }
    pub fn typed(bytes: Vec<u8>, dtype: DType) -> Self {
        Param::Typed(bytes, dtype)
    }
}

/// Supplies a stage's parameters on demand, by name. `None` means "not mine"
/// (the runtime leaves the param unset — useful when a source only owns a shard).
pub trait ParamSource {
    fn get(&mut self, name: &str) -> Option<Param>;
}

/// A `HashMap<String, Vec<f32>>` is a source (dense f32; clones on read).
impl ParamSource for HashMap<String, Vec<f32>> {
    fn get(&mut self, name: &str) -> Option<Param> {
        HashMap::get(self, name).map(|v| Param::F32(v.clone()))
    }
}

/// A `HashMap<String, Param>` is a source (mixed f32 / packed).
impl ParamSource for HashMap<String, Param> {
    fn get(&mut self, name: &str) -> Option<Param> {
        HashMap::get(self, name).cloned()
    }
}

/// Any `FnMut(&str) -> Option<Param>` is a source — adapt an existing loader
/// inline without defining a type.
impl<F> ParamSource for F
where
    F: FnMut(&str) -> Option<Param>,
{
    fn get(&mut self, name: &str) -> Option<Param> {
        self(name)
    }
}
