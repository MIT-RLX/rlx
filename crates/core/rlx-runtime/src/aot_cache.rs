// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AOT cache — persist optimized LIR modules and reload for backend compile.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use rlx_ir::DimBinding;
use rlx_ir::Graph;
use rlx_ir::LirFingerprint;
use rlx_ir::LirModule;
use rlx_ir::hir::HirModule;
use rlx_opt::CompileResult;

use crate::stages;
use crate::{CompileOptions, CompiledGraph, Device};

/// Errors from [`AotCache`] disk / compile operations.
#[derive(Debug)]
pub enum AotCacheError {
    Io(io::Error),
    Serde(String),
    Lower(rlx_ir::hir::LowerError),
}

impl fmt::Display for AotCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Serde(e) => write!(f, "serde: {e}"),
            Self::Lower(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AotCacheError {}

impl From<io::Error> for AotCacheError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<rlx_ir::hir::LowerError> for AotCacheError {
    fn from(e: rlx_ir::hir::LowerError) -> Self {
        Self::Lower(e)
    }
}

/// On-disk AOT cache for optimized LIR modules.
pub struct AotCache {
    root: PathBuf,
}

impl AotCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn lir_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.lir.bin"))
    }

    fn meta_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.meta.json"))
    }

    /// Persist an optimized LIR module. Returns its compile fingerprint.
    pub fn put_lir(&self, key: &str, lir: &LirModule) -> io::Result<LirFingerprint> {
        fs::create_dir_all(&self.root)?;
        let fp = LirFingerprint::of(lir);
        // Compact binary LIR — ~2.5× smaller than JSON and ~10× faster to load
        // (baked constants are raw f32 bytes, not ASCII); this is the dominant
        // per-process compile cost for constant-heavy graphs (STFT vocoders).
        let bytes = rlx_ir::lir_to_bytes(lir)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        fs::write(self.lir_path(key), bytes)?;
        fs::write(
            self.meta_path(key),
            format!("{{\"fingerprint\":{}}}\n", fp.0),
        )?;
        Ok(fp)
    }

    /// Load a previously stored LIR module.
    pub fn get_lir(&self, key: &str) -> io::Result<LirModule> {
        let bytes = fs::read(self.lir_path(key))?;
        rlx_ir::lir_from_bytes(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.lir_path(key).is_file()
    }

    /// MIR graph → fusion pipeline → cached LIR → backend executable.
    ///
    /// On a cache hit only the backend compile runs (fusion / vmap /
    /// autodiff already done). Keys must be unique per graph + options
    /// fingerprint the caller encodes in `key`.
    pub fn compile_graph_cached(
        &self,
        key: &str,
        device: Device,
        graph: Graph,
        options: &CompileOptions,
    ) -> Result<CompiledGraph, AotCacheError> {
        if self.contains(key) {
            match self.get_lir(key) {
                Ok(lir) => return Ok(self.compile_lir(device, lir, options)),
                Err(e) => {
                    eprintln!("[aot] cache miss (reload failed for {key}): {e}");
                    let _ = fs::remove_file(self.lir_path(key));
                    let _ = fs::remove_file(self.meta_path(key));
                }
            }
        }
        let result = stages::compile_graph_stages(device, graph, options);
        stages::maybe_log_fusion(&result.fusion);
        self.put_lir(key, &result.lir)?;
        Ok(self.compile_lir(device, result.lir, options))
    }

    /// Compile HIR through the fusion pipeline, cache LIR, return executable.
    pub fn compile_hir_cached(
        &self,
        key: &str,
        device: Device,
        hir: HirModule,
        options: &CompileOptions,
    ) -> Result<CompiledGraph, AotCacheError> {
        if self.contains(key) {
            match self.get_lir(key) {
                Ok(lir) => return Ok(self.compile_lir(device, lir, options)),
                Err(e) => {
                    // Stale / schema-mismatched AOT blob — drop and recompile.
                    eprintln!("[aot] cache miss (reload failed for {key}): {e}");
                    let _ = fs::remove_file(self.lir_path(key));
                    let _ = fs::remove_file(self.meta_path(key));
                }
            }
        }
        let result = stages::compile_hir_stages(device, hir, options)?;
        stages::maybe_log_fusion(&result.fusion);
        self.put_lir(key, &result.lir)?;
        Ok(self.compile_lir(device, result.lir, options))
    }

    /// Specialize a cached dynamic LIR template and persist the bound variant.
    pub fn specialize_cached(
        &self,
        base_key: &str,
        binding: &DimBinding,
        device: Device,
        template: &CompileResult,
        options: &CompileOptions,
    ) -> Result<CompiledGraph, AotCacheError> {
        let spec_key = format!("{base_key}__{}", binding_hash(binding));
        if self.contains(&spec_key) {
            match self.get_lir(&spec_key) {
                Ok(lir) => return Ok(self.compile_lir(device, lir, options)),
                Err(e) => {
                    eprintln!("[aot] cache miss (reload failed for {spec_key}): {e}");
                    let _ = fs::remove_file(self.lir_path(&spec_key));
                    let _ = fs::remove_file(self.meta_path(&spec_key));
                }
            }
        }
        let pipe = stages::pipeline_for(device, options);
        let specialized = template.specialize(&pipe, binding);
        self.put_lir(&spec_key, &specialized.lir)?;
        Ok(self.compile_lir(device, specialized.lir, options))
    }

    fn compile_lir(
        &self,
        device: Device,
        lir: LirModule,
        options: &CompileOptions,
    ) -> CompiledGraph {
        let backend = crate::registry::backend_for(device).expect("backend registered");
        let executable = backend.compile_lir(lir, options);
        CompiledGraph::new(executable, device)
    }
}

fn binding_hash(binding: &DimBinding) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for (sym, size) in binding.iter() {
        sym.hash(&mut h);
        size.hash(&mut h);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;
    use rlx_ir::Shape;

    #[test]
    fn aot_lir_roundtrip_on_disk() {
        let dir = std::env::temp_dir().join(format!("rlx_aot_{}", std::process::id()));
        let cache = AotCache::new(&dir);
        let mut hir = HirModule::new("aot");
        let x = hir.input("x", Shape::new(&[1, 4], DType::F32));
        let w = hir.param("w", Shape::new(&[4, 2], DType::F32));
        let y = hir.linear(x, w, None, None, Shape::new(&[1, 2], DType::F32));
        hir.set_outputs(vec![y]);
        let opts = CompileOptions::new();
        let _compiled = cache
            .compile_hir_cached("tiny", Device::Cpu, hir, &opts)
            .expect("compile + cache");
        assert!(cache.contains("tiny"));
        let lir = cache.get_lir("tiny").expect("reload LIR");
        assert_eq!(lir.name(), "aot");
        fs::remove_dir_all(&dir).ok();
    }
}
