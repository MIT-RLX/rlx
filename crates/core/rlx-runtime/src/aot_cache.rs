// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AOT cache — persist optimized LIR modules and reload for backend compile.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
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

/// What a cache entry was compiled from: the device, the options that shape
/// the compile, and the graph's structure (ops, attributes, shapes, edges and
/// constant bytes).
///
/// Keys alone are not enough. They are chosen by callers, and a key that does
/// not spell out everything that changes the compile serves a stale graph
/// silently: KittenTTS's probe keys (`probe_cpu`, `probe_dev`) name neither
/// device nor graph, so a CPU run loaded LIR compiled earlier from a
/// different graph, and two CPU compiles of the same model disagreed
/// (waveform correlation 0.73). Bound parameter VALUES are not included,
/// only their names and lengths.
///
/// `DefaultHasher` is not stable across Rust releases; that only costs a
/// recompile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputFingerprint(u64);

impl InputFingerprint {
    fn of_hir(device: Device, options: &CompileOptions, hir: &HirModule) -> Self {
        let mut h = DefaultHasher::new();
        hash_compile_context(&mut h, device, options);
        "hir".hash(&mut h);
        hir.name.hash(&mut h);
        format!("{:?}", hir.fusion_policy).hash(&mut h);
        for node in hir.nodes() {
            node.id.hash(&mut h);
            node.inputs.hash(&mut h);
            node.shape.hash(&mut h);
            node.name.hash(&mut h);
            match &node.op {
                rlx_ir::hir::HirOp::Mir(op) => hash_op(&mut h, op),
                op => format!("{op:?}").hash(&mut h),
            }
        }
        hir.outputs.hash(&mut h);
        hir.output_names.hash(&mut h);
        Self(h.finish())
    }

    fn of_graph(device: Device, options: &CompileOptions, graph: &Graph) -> Self {
        let mut h = DefaultHasher::new();
        hash_compile_context(&mut h, device, options);
        "mir".hash(&mut h);
        graph.name.hash(&mut h);
        for node in graph.nodes() {
            node.id.hash(&mut h);
            node.inputs.hash(&mut h);
            node.shape.hash(&mut h);
            hash_op(&mut h, &node.op);
        }
        graph.outputs.hash(&mut h);
        Self(h.finish())
    }
}

/// Constants by their bytes (their `Debug` form is a multi-megabyte list for
/// a baked STFT basis); every other op by its `Debug` form, which carries all
/// attributes.
fn hash_op(h: &mut DefaultHasher, op: &rlx_ir::Op) {
    match op {
        rlx_ir::Op::Constant { data } => {
            "Constant".hash(h);
            data.hash(h);
        }
        op => format!("{op:?}").hash(h),
    }
}

fn hash_compile_context(h: &mut DefaultHasher, device: Device, o: &CompileOptions) {
    format!("{device:?}").hash(h);
    // Every field except the parameter bindings, whose maps hold weights
    // (large, and unordered); those contribute their sorted names and lengths.
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}|{:?}|{:?}|{}|{}|{:?}|{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{}",
        o.precision,
        o.policy,
        o.scaled_quant,
        o.number_format,
        o.number_format_options,
        o.rng,
        o.dce,
        o.constant_folding,
        o.fusion_target,
        o.fusion_opts,
        o.arena_alignment,
        o.assert_fusion_clean,
        o.supported_ops,
        o.dim_binding,
        o.kernel_dispatch,
        o.disable_mpsgraph,
        o.scan_unroll_max_length,
        o.cache_param_invariant,
        o.lint_numerics,
        o.disable_conv_bias_act_fusion,
        o.no_io_peaks_output,
        o.simplify_transposes,
        o.verbose,
    )
    .hash(h);
    for map in [
        o.param_bindings.as_ref().map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect::<Vec<_>>()
        }),
        o.quant_param_bindings.as_ref().map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect::<Vec<_>>()
        }),
    ] {
        let mut entries = map.unwrap_or_default();
        entries.sort();
        entries.hash(h);
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
        self.put_lir_for_input(key, lir, None)
    }

    /// [`Self::put_lir`], also recording the [`InputFingerprint`] it was
    /// compiled from so a later lookup can tell the entry is stale.
    fn put_lir_for_input(
        &self,
        key: &str,
        lir: &LirModule,
        input: Option<InputFingerprint>,
    ) -> io::Result<LirFingerprint> {
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
            match input {
                Some(i) => format!("{{\"fingerprint\":{},\"input\":{}}}\n", fp.0, i.0),
                None => format!("{{\"fingerprint\":{}}}\n", fp.0),
            },
        )?;
        Ok(fp)
    }

    /// The input fingerprint stored with `key`, if any.
    fn stored_input(&self, key: &str) -> Option<InputFingerprint> {
        let meta = fs::read_to_string(self.meta_path(key)).ok()?;
        let tail = meta.split("\"input\":").nth(1)?;
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok().map(InputFingerprint)
    }

    /// Whether `key` holds an entry compiled from exactly `input`. An entry
    /// from a different graph, device or options — or one written before
    /// inputs were recorded — is a miss: it is recompiled and overwritten.
    fn fresh(&self, key: &str, input: InputFingerprint) -> bool {
        if !self.contains(key) {
            return false;
        }
        if self.stored_input(key) == Some(input) {
            return true;
        }
        eprintln!("[aot] cache entry {key} was compiled from a different input; recompiling");
        false
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
        let input = InputFingerprint::of_graph(device, options, &graph);
        if self.fresh(key, input) {
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
        self.put_lir_for_input(key, &result.lir, Some(input))?;
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
        let input = InputFingerprint::of_hir(device, options, &hir);
        if self.fresh(key, input) {
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
        self.put_lir_for_input(key, &result.lir, Some(input))?;
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

    /// `x · W` with `W` either `[4, n]` (n outputs) — two graphs one key.
    fn linear(n: usize) -> HirModule {
        let mut hir = HirModule::new("aot");
        let x = hir.input("x", Shape::new(&[1, 4], DType::F32));
        let w = hir.param("w", Shape::new(&[4, n], DType::F32));
        let y = hir.linear(x, w, None, None, Shape::new(&[1, n], DType::F32));
        hir.set_outputs(vec![y]);
        hir
    }

    /// A key reused for a different graph must not serve the old entry. Kitten
    /// probe keys named neither graph nor device, and a stale entry made two
    /// CPU compiles of one model disagree.
    #[test]
    fn a_reused_key_recompiles_a_changed_graph() {
        let dir = std::env::temp_dir().join(format!("rlx_aot_stale_{}", std::process::id()));
        let cache = AotCache::new(&dir);
        let opts = CompileOptions::new();
        let out_width = |cache: &AotCache| {
            let lir = cache.get_lir("k").expect("reload LIR");
            let g = lir.mir.as_graph();
            g.node(g.outputs[0]).shape.dim(1).unwrap_static()
        };
        cache
            .compile_hir_cached("k", Device::Cpu, linear(2), &opts)
            .expect("compile");
        assert_eq!(out_width(&cache), 2);
        let mut c = cache
            .compile_hir_cached("k", Device::Cpu, linear(3), &opts)
            .expect("recompile");
        assert_eq!(out_width(&cache), 3, "served the entry for the old graph");
        let w: Vec<f32> = (0..12).map(|i| i as f32).collect();
        c.set_param("w", &w);
        let y = c.run(&[("x", &[1.0, 1.0, 1.0, 1.0][..])]);
        assert_eq!(y[0], vec![18.0, 22.0, 26.0]);
        // Same graph again: a hit, and the stored input is unchanged.
        let before = cache.stored_input("k");
        assert!(before.is_some());
        cache
            .compile_hir_cached("k", Device::Cpu, linear(3), &opts)
            .expect("hit");
        assert_eq!(cache.stored_input("k"), before);
        fs::remove_dir_all(&dir).ok();
    }

    /// An entry written before inputs were recorded has no `input` field; it
    /// is recompiled once rather than trusted.
    #[test]
    fn an_entry_without_a_recorded_input_is_recompiled() {
        let dir = std::env::temp_dir().join(format!("rlx_aot_legacy_{}", std::process::id()));
        let cache = AotCache::new(&dir);
        let opts = CompileOptions::new();
        cache
            .compile_hir_cached("k", Device::Cpu, linear(2), &opts)
            .expect("compile");
        let lir = cache.get_lir("k").expect("reload");
        cache.put_lir("k", &lir).expect("legacy write");
        assert_eq!(cache.stored_input("k"), None);
        cache
            .compile_hir_cached("k", Device::Cpu, linear(2), &opts)
            .expect("recompile");
        assert!(cache.stored_input("k").is_some());
        fs::remove_dir_all(&dir).ok();
    }
}
