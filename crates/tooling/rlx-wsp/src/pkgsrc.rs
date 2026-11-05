// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weights of an `.rlxp` package, read on demand.
//!
//! A package space never copies the model into its state: tensors are read
//! from the package (memory-mapped) the first time a tool needs them, viewed
//! as `[prod(leading dims), last dim]`, and only tensors the agent edits are
//! stored — as an overlay in the state. f32, f16 and bf16 tensors are
//! readable and editable; quantized ones are listed but read-only.

use crate::space::Tensor;
use rlx_pkg::Package;
use std::cell::OnceCell;
use std::collections::BTreeMap;

/// Largest tensor a view or edit loads whole (elements).
pub const MAX_ELEMENTS: usize = 64 << 20;

pub struct PackageStore {
    pub path: String,
    pkg: Package,
    /// name → (shape, scheme)
    entries: BTreeMap<String, (Vec<usize>, String)>,
    cache: BTreeMap<String, OnceCell<Result<Tensor, String>>>,
    /// The package's graphs (functions), structure only (weights unbound).
    module: OnceCell<Result<Vec<rlx_ir::Graph>, String>>,
}

impl PackageStore {
    pub fn open(path: &str) -> Result<Self, String> {
        let pkg = Package::open(path).map_err(|e| format!("{path}: {e:#}"))?;
        let entries: BTreeMap<String, (Vec<usize>, String)> = pkg
            .weights_index()
            .map(|idx| {
                idx.tensors
                    .iter()
                    .map(|e| (e.name.clone(), (e.shape.clone(), e.scheme.clone())))
                    .collect()
            })
            .unwrap_or_default();
        if entries.is_empty() {
            return Err(format!(
                "{path}: the package holds no weights (exported without --embed-weights?)"
            ));
        }
        let cache = entries
            .keys()
            .map(|k| (k.clone(), OnceCell::new()))
            .collect();
        Ok(Self {
            path: path.to_string(),
            pkg,
            entries,
            cache,
            module: OnceCell::new(),
        })
    }

    /// The package's functions (graphs), read once, weights left unbound.
    pub fn functions(&self) -> Result<&[rlx_ir::Graph], String> {
        self.module
            .get_or_init(|| {
                self.pkg
                    .module_with(rlx_pkg::MaterializeMode::None)
                    .map(|m| m.graphs)
                    .map_err(|e| format!("{}: {e:#}", self.path))
            })
            .as_ref()
            .map(Vec::as_slice)
            .map_err(Clone::clone)
    }

    /// One function by name (`None`: the first).
    pub fn function(&self, name: Option<&str>) -> Result<&rlx_ir::Graph, String> {
        let fs = self.functions()?;
        match name {
            None => fs
                .first()
                .ok_or_else(|| "the package has no graphs".to_string()),
            Some(n) => fs.iter().find(|g| g.name == n).ok_or_else(|| {
                let names: Vec<&str> = fs.iter().map(|g| g.name.as_str()).collect();
                format!("no function `{n}` (have: {})", names.join(", "))
            }),
        }
    }

    /// The tokenizer the package carries, if any.
    pub fn sidecar(&self, id: &str) -> Option<Vec<u8>> {
        self.pkg.sidecar(id).ok()
    }

    pub fn sidecar_ids(&self) -> Vec<String> {
        self.pkg
            .sidecar_ids()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// `(shape, scheme)` of a tensor.
    pub fn entry(&self, name: &str) -> Option<&(Vec<usize>, String)> {
        self.entries.get(name)
    }

    /// `[rows, cols]` view of a tensor's shape.
    pub fn view_shape(&self, name: &str) -> Option<(usize, usize)> {
        let (shape, _) = self.entries.get(name)?;
        let cols = shape.last().copied().unwrap_or(1).max(1);
        Some((shape.iter().product::<usize>() / cols, cols))
    }

    pub fn editable(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|(_, s)| matches!(s.as_str(), "f32" | "f16" | "bf16"))
    }

    /// The tensor as f32, loaded once.
    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        let cell = self
            .cache
            .get(name)
            .ok_or_else(|| format!("unknown param `{name}` (see list_params)"))?;
        cell.get_or_init(|| self.load(name))
            .as_ref()
            .map_err(Clone::clone)
    }

    fn load(&self, name: &str) -> Result<Tensor, String> {
        let (shape, scheme) = &self.entries[name];
        let (rows, cols) = self.view_shape(name).expect("known");
        if rows * cols > MAX_ELEMENTS {
            return Err(format!(
                "`{name}` has {} elements (max {MAX_ELEMENTS} loaded at once)",
                rows * cols
            ));
        }
        let raw = self
            .pkg
            .tensor_bytes(name)
            .map_err(|e| format!("{name}: {e:#}"))?;
        let data: Vec<f32> = match scheme.as_str() {
            "f32" => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "bf16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            "f16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(format!(
                    "`{name}` is stored as {other} (quantized): listed but not readable or editable"
                ));
            }
        };
        if data.len() != rows * cols {
            return Err(format!(
                "`{name}`: {} values for shape {shape:?}",
                data.len()
            ));
        }
        Ok(Tensor { rows, cols, data })
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_decodes() {
        assert_eq!(super::half_to_f32(0x3c00), 1.0);
        assert_eq!(super::half_to_f32(0xc000), -2.0);
        assert_eq!(super::half_to_f32(0x3555), 0.333_251_95);
        assert_eq!(super::half_to_f32(0x0001), 5.960_464_5e-8);
    }
}
