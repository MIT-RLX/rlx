// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The built-in `rlxp` provider: an `.rlxp` package with an embedded graph,
//! compiled for the requested device, its weights bound to the graph's
//! params. [`save_package`] writes one from a graph and its weights, so a
//! model built anywhere (Rust, a downstream crate, any binding) runs everywhere.
//!
//! The graph's `.rlxmodel` header carries the model block: its `name` and
//! `version`, and the task, family and outputs this provider reads:
//! `model {name: "…", version: "…", task: "…", family: "…", outputs: ["logits", …],
//! metadata: {…}}`. Packages from before it carry that object as a
//! `model.json` sidecar, which is still read.
//!
//! The weights may live outside the package: `weights: [{uri:
//! "safetensors://model.safetensors", prefix: "roberta.", map: {w: "dense.weight"}}]`
//! names where each param the package does not hold comes from (any scheme
//! [`resolve_weight_uri`](crate::dist::resolve_weight_uri) reads; a relative
//! path is beside the package, or inside a directory package). The file's
//! tensor is `map[param]`, else `prefix` + the param's name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rlx_ir::{DType, Dim, Graph, Op};
use serde_json::{Value, json};

use super::{Model, ModelInfo, ModelProvider, ModelRequest, Tensor, TensorSpec, dtype_name};
use crate::{CompiledGraph, Session};

/// The id of the model sidecar.
pub const MODEL_SIDECAR: &str = "model.json";

/// Opens `.rlxp` packages that embed a graph.
pub struct RlxpProvider;

fn is_package(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    ext.eq_ignore_ascii_case("rlxp") || (path.is_dir() && path.join("rlx.json").is_file())
}

impl ModelProvider for RlxpProvider {
    fn name(&self) -> &str {
        "rlxp"
    }

    fn description(&self) -> &str {
        "an .rlxp package with an embedded graph, on any device"
    }

    fn tasks(&self) -> Vec<String> {
        vec!["graph".into()]
    }

    fn probe(&self, request: &ModelRequest) -> u32 {
        if is_package(request.path()) { 10 } else { 0 }
    }

    fn open(&self, request: &ModelRequest) -> Result<Box<dyn Model>> {
        let mut pkg = Opened::open_merged(request.path())?;
        if let Some(sae) = request.option("sae") {
            super::sae::splice(&mut pkg, sae)?;
        }
        pkg.target = request.device_label();
        pkg.weight_dtype = weight_dtype_option(request, &pkg)?;
        pkg.tight = memory_tight(&pkg);
        pkg.weights.lean = pkg.tight || pkg.weight_dtype.is_some() || packs_tight(&pkg);
        if request
            .option("verify_weights")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            pkg.weights.verify()?;
        }
        // A model block that declares entries runs them: the one the open
        // option `entry` names, its one, or — when it has several — each by
        // the task asked of it ([`Entries`]). Otherwise the main graph runs
        // as is.
        let mut entries: Vec<(String, String, Value)> = entries_of(&pkg.meta)
            .into_iter()
            .filter(|(_, kind, _)| ENTRY_KINDS.contains(&kind.as_str()))
            .collect();
        if let Some(w) = request.option("entry").and_then(Value::as_str) {
            entries.retain(|(name, _, _)| name == w);
            if entries.is_empty() {
                bail!("{}: no entry '{w}'", request.path().display());
            }
        }
        match entries.len() {
            0 => {}
            1 => {
                let (_, kind, entry) = entries.remove(0);
                return open_entry(request, pkg, &kind, &entry);
            }
            _ => return Entries::open(request, pkg, entries),
        }
        Ok(Box::new(RlxpModel::open(request, pkg)?))
    }
}

/// The entry kinds this rlx runs, in the order it opens the first of when
/// a package has several and the caller asks for none in particular.
pub const ENTRY_KINDS: &[&str] = &[
    "embed",
    "score",
    "transcribe",
    "synthesize",
    "pipeline",
    "generate",
    "program",
];

/// The entries of a model block, `(name, kind, entry)`: an entry's kind is
/// its `kind` field, or its name — `"pipeline": {…}` is a pipeline, and so
/// is `"image": {"kind": "pipeline", …}`.
pub fn entries_of(model: &Value) -> Vec<(String, String, Value)> {
    let Some(entries) = model.get("entries").and_then(Value::as_object) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter(|(_, e)| !e.is_null())
        .map(|(name, e)| {
            let kind = e.get("kind").and_then(Value::as_str).unwrap_or(name);
            (name.clone(), kind.to_string(), e.clone())
        })
        .collect()
}

/// The first entry of `kind` in a model block.
pub fn entry_of_kind(model: &Value, kind: &str) -> Option<Value> {
    entries_of(model)
        .into_iter()
        .find(|(_, k, _)| k == kind)
        .map(|(_, _, e)| e)
}

pub(super) fn open_entry(
    request: &ModelRequest,
    pkg: Opened,
    kind: &str,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    match kind {
        "embed" => super::entries::text_encoder(request, pkg, entry),
        "score" => super::entries::text_scorer(request, pkg, entry),
        "transcribe" => super::entries::speech_to_text(request, pkg, entry),
        "synthesize" => super::entries::text_to_speech(request, pkg, entry),
        "pipeline" => super::entries::pipeline(request, pkg, entry),
        "generate" => super::entries::causal_lm(request, pkg, entry),
        #[cfg(feature = "program")]
        "program" => super::program::program(request, pkg, entry),
        #[cfg(not(feature = "program"))]
        "program" => {
            let _ = (request, pkg);
            bail!("this rlx was built without programs (the `program` feature of rlx-runtime)")
        }
        other => bail!("entry kind '{other}' is not one this rlx runs"),
    }
}

/// A package with several entries, as one model: each task goes to the
/// entry of its kind (`embed` to an `embed` entry, `chat` to a `generate`
/// one, …), and `run` to the entry whose inputs are the tensors given (or
/// the one `options.entry` names). The first entry by [`ENTRY_KINDS`]
/// opens with the package; the others when first asked for, each with its
/// own copy of what it runs.
struct Entries {
    request: ModelRequest,
    /// `(name, kind, the names of its inputs)`.
    entries: Vec<(String, String, Vec<String>)>,
    /// The kinds of task each entry answers: its own, or a program's `tasks`.
    serves: Vec<Vec<String>>,
    opened: Vec<Option<Box<dyn Model>>>,
    first: usize,
    info: ModelInfo,
}

/// The kinds of task the entry `entry` (of `kind`) answers: its kind, or a
/// program's `tasks` (`chat` and `generate` both a `generate`'s, `run` a
/// `pipeline`'s).
fn serves(kind: &str, entry: &Value) -> Vec<String> {
    if kind != "program" {
        return vec![kind.to_string()];
    }
    entry
        .get("tasks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|t| match t {
            "chat" => "generate",
            "run" => "pipeline",
            t => t,
        })
        .map(String::from)
        .collect()
}

/// The names an entry's `inputs` gives: a list of names or contracts
/// (`{name, …}`), or a map from its graphs' inputs.
fn input_names(entry: &Value) -> Vec<String> {
    match entry.get("inputs") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                v => v.get("name").and_then(Value::as_str).map(String::from),
            })
            .collect(),
        Some(Value::Object(m)) => m.keys().cloned().collect(),
        _ => Vec::new(),
    }
}

impl Entries {
    fn open(
        request: &ModelRequest,
        pkg: Opened,
        mut list: Vec<(String, String, Value)>,
    ) -> Result<Box<dyn Model>> {
        let rank = |k: &str| {
            ENTRY_KINDS
                .iter()
                .position(|e| *e == k)
                .unwrap_or(usize::MAX)
        };
        list.sort_by_key(|(_, kind, _)| rank(kind));
        let entries: Vec<(String, String, Vec<String>)> = list
            .iter()
            .map(|(n, k, e)| (n.clone(), k.clone(), input_names(e)))
            .collect();
        let serves = list.iter().map(|(_, k, e)| serves(k, e)).collect();
        let first = open_entry(request, pkg, &list[0].1, &list[0].2)?;
        let mut info = first.info();
        info.metadata["entries"] = json!(
            entries
                .iter()
                .map(|(n, k, i)| json!({"name": n, "kind": k, "inputs": i}))
                .collect::<Vec<_>>()
        );
        let mut opened: Vec<Option<Box<dyn Model>>> = (0..entries.len()).map(|_| None).collect();
        opened[0] = Some(first);
        Ok(Box::new(Self {
            request: request.clone(),
            entries,
            serves,
            opened,
            first: 0,
            info,
        }))
    }

    /// Entry `i`, opened.
    fn at(&mut self, i: usize) -> Result<&mut Box<dyn Model>> {
        if self.opened[i].is_none() {
            let mut options = self.request.options.clone();
            options["entry"] = json!(self.entries[i].0);
            let request = ModelRequest::new(self.request.source.clone(), options)?;
            self.opened[i] = Some(RlxpProvider.open(&request)?);
        }
        Ok(self.opened[i].as_mut().expect("just opened"))
    }

    /// The first entry of `kind`, or the first entry (whose error then says
    /// what it does not do).
    fn of_kind(&mut self, kind: &str) -> Result<&mut Box<dyn Model>> {
        let i = self
            .serves
            .iter()
            .position(|s| s.iter().any(|k| k == kind))
            .unwrap_or(self.first);
        self.at(i)
    }

    /// The entry `run` takes `inputs` to: the one `options.entry` names, the
    /// first whose inputs name every tensor given, or the first.
    fn for_inputs(&self, inputs: &[Tensor], options: &Value) -> Result<usize> {
        if let Some(w) = options.get("entry").and_then(Value::as_str) {
            return self
                .entries
                .iter()
                .position(|(n, _, _)| n == w)
                .with_context(|| format!("no entry '{w}'"));
        }
        let takes =
            |names: &[String]| !names.is_empty() && inputs.iter().all(|t| names.contains(&t.name));
        Ok(self
            .entries
            .iter()
            .position(|(_, _, names)| takes(names))
            .unwrap_or(self.first))
    }
}

impl Model for Entries {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let i = self.for_inputs(inputs, options)?;
        let mut options = options.clone();
        if let Some(o) = options.as_object_mut() {
            o.remove("entry");
        }
        self.at(i)?.run(inputs, &options)
    }

    fn lm(&mut self) -> Option<&mut dyn super::LmRunner> {
        self.of_kind("generate").ok()?.lm()
    }

    fn generate_text(&mut self, prompt: &str, options: &Value) -> Result<String> {
        self.of_kind("generate")?.generate_text(prompt, options)
    }

    fn generate_batch(&mut self, prompts: &[&str], options: &Value) -> Result<Vec<String>> {
        self.of_kind("generate")?.generate_batch(prompts, options)
    }

    fn generate_with_media(
        &mut self,
        prompt: &str,
        media: &[Tensor],
        options: &Value,
    ) -> Result<String> {
        self.of_kind("generate")?
            .generate_with_media(prompt, media, options)
    }

    fn chat(&mut self, messages: &[super::ChatMessage], options: &Value) -> Result<String> {
        self.of_kind("generate")?.chat(messages, options)
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        self.of_kind("transcribe")?
            .transcribe(audio, sample_rate, options)
    }

    fn synthesize(&mut self, text: &str, options: &Value) -> Result<super::Audio> {
        self.of_kind("synthesize")?.synthesize(text, options)
    }

    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>> {
        self.of_kind("embed")?.embed(texts, options)
    }

    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>> {
        self.of_kind("score")?.score(query, passages, options)
    }
}

/// Whether `pkg`'s functions, each compiled with its own f32 copy of its
/// params, would take more than half the machine's memory.
fn memory_tight(pkg: &Opened) -> bool {
    crate::memory_estimate::available_unified_memory()
        .is_some_and(|total| pkg.f32_param_bytes().0 as f64 > 0.5 * total as f64)
}

/// Whether `pkg`'s functions with their weight-concatenating fusions' packs
/// would take more than half the machine's memory: each executable then
/// holds its packed QKV / gate-up copies beside its weights, measured at
/// about 1.7× the functions' f32 params all told (Qwen3-VL-2B on Metal:
/// 26 GB of params over five functions, a 43 GB peak; lean, 32 GB).
/// Compiling lean drops the packs and keeps the executables.
fn packs_tight(pkg: &Opened) -> bool {
    crate::memory_estimate::available_unified_memory()
        .is_some_and(|total| 1.7 * pkg.f32_param_bytes().0 as f64 > 0.5 * total as f64)
}

/// Whether even `pkg`'s largest function's params as f32 would take more
/// than 40% of the machine's memory.
fn one_copy_tight(pkg: &Opened) -> bool {
    crate::memory_estimate::available_unified_memory()
        .is_some_and(|total| pkg.f32_param_bytes().1 as f64 > 0.4 * total as f64)
}

/// The open option `weight_dtype`: `"f32"` (as stored), `"f16"`, `"bf16"`,
/// or `"auto"` — f16 when its largest function's params as f32 would take
/// more than 40% of the machine's memory (they would not fit beside the
/// activations and everything else running), else as stored.
fn weight_dtype_option(request: &ModelRequest, pkg: &Opened) -> Result<Option<DType>> {
    Ok(
        match request.option("weight_dtype").and_then(Value::as_str) {
            None | Some("f32") => None,
            Some("f16") => Some(DType::F16),
            Some("bf16") => Some(DType::BF16),
            Some("auto") => one_copy_tight(pkg).then_some(DType::F16),
            Some(other) => bail!("weight_dtype '{other}': f32, f16, bf16 or auto"),
        },
    )
}

struct RlxpModel {
    compiled: CompiledGraph,
    info: ModelInfo,
    output_names: Vec<String>,
}

fn spec_of(g: &Graph, id: rlx_ir::NodeId, name: String) -> TensorSpec {
    let shape = g.node(id).shape.clone();
    TensorSpec {
        name,
        dtype: shape.dtype(),
        shape: shape
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => Some(*n),
                Dim::Dynamic(_) => None,
            })
            .collect(),
    }
}

/// An opened package: its graphs, its model block, where its weights are.
pub(super) struct Opened {
    pub path: PathBuf,
    pub pack: rlx_pkg::Package,
    pub graphs: Vec<Graph>,
    pub meta: Value,
    pub rlx_version: Option<String>,
    /// Symbol names of the graphs' dims (`seq`, `cap`).
    pub dims: Vec<(String, u32)>,
    pub weights: PackageWeights,
    /// The device it runs on (`metal`, `cpu`, …): a function built for it
    /// (`step:metal`, by [`build_for_target`]) runs in place of the
    /// portable one.
    pub target: String,
    /// The dtype matmul weights run in (open option `weight_dtype`): each
    /// function taken declares them so, and the checkpoint's values are
    /// narrowed as they bind. `None`: as the package stores them.
    pub weight_dtype: Option<rlx_ir::DType>,
    /// Its functions' params as f32, one copy per function, take more than
    /// half the machine's memory: a model keeps one executable's copy at a
    /// time (prompt and media executables freed after use, as `prefill:
    /// "release"`) unless told `prefill: "keep"`.
    pub tight: bool,
}

impl Opened {
    /// The package at `path`; an adapter package (`adapter: {base, lora}`)
    /// opens its base with the adapter's updates merged into the weights as
    /// they bind.
    pub(super) fn open_merged(path: &Path) -> Result<Self> {
        let adapter = Self::open(path)?;
        let Some(spec) = adapter.meta.get("adapter").cloned() else {
            return Ok(adapter);
        };
        let base_rel = spec
            .get("base")
            .and_then(Value::as_str)
            .context("adapter.base names the base package")?;
        let mut base = Self::open_merged(&beside(path, base_rel))?;
        let mut held = adapter.weights;
        for l in spec
            .get("lora")
            .and_then(Value::as_array)
            .context("adapter.lora lists the updates")?
        {
            let field = |k: &str| {
                l.get(k)
                    .and_then(Value::as_str)
                    .with_context(|| format!("an adapter update names its `{k}`"))
            };
            let (tensor, a_name, b_name) = (field("tensor")?, field("a")?, field("b")?);
            let rank = l
                .get("rank")
                .and_then(Value::as_u64)
                .context("an adapter update gives its `rank`")? as usize;
            let scale = l.get("scale").and_then(Value::as_f64).unwrap_or(1.0) as f32;
            let a = held.take_table(a_name)?;
            let b = held.take_table(b_name)?;
            if rank == 0 || a.len() % rank != 0 || b.len() % rank != 0 {
                bail!(
                    "adapter update of '{tensor}': A {} and B {} values at rank {rank}",
                    a.len(),
                    b.len()
                );
            }
            // A [rank, in], B [out, rank]: B·A is [out, in].
            let (inner, outer) = (a.len() / rank, b.len() / rank);
            let mut delta = vec![0f32; outer * inner];
            for o in 0..outer {
                for r in 0..rank {
                    let bv = b[o * rank + r] * scale;
                    if bv == 0.0 {
                        continue;
                    }
                    let row = &a[r * inner..(r + 1) * inner];
                    for (d, av) in delta[o * inner..(o + 1) * inner].iter_mut().zip(row) {
                        *d += bv * av;
                    }
                }
            }
            match base.weights.lora.get_mut(tensor) {
                Some(prev) => prev.iter_mut().zip(&delta).for_each(|(p, d)| *p += d),
                None => {
                    base.weights.lora.insert(tensor.to_string(), delta);
                }
            }
        }
        // Its updated weights are not the base's of the same names.
        base.weights.scope = format!("{}+{}", base.weights.scope, path.display());
        // What opened it says so.
        if let Value::Object(m) = &mut base.meta {
            m.insert(
                "adapter".into(),
                json!({"name": adapter.meta.get("name"), "path": path.display().to_string()}),
            );
        }
        Ok(base)
    }

    /// This package again, its own handle on the weights (an adapter's
    /// updates kept): for stages that load apart from the decoder.
    /// Each function's params as f32: their sum (what a model holds once
    /// every function is compiled, each executable with its own copy) and
    /// the largest function's.
    pub(super) fn f32_param_bytes(&self) -> (usize, usize) {
        let (mut sum, mut top) = (0, 0);
        for g in &self.graphs {
            let mut seen = std::collections::HashSet::new();
            let mut bytes = 0;
            for n in g.nodes() {
                if let rlx_ir::Op::Param { name } = &n.op
                    && seen.insert(name.clone())
                    && let Some(numel) = n
                        .shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            Dim::Static(x) => Some(*x),
                            Dim::Dynamic(_) => None,
                        })
                        .product::<Option<usize>>()
                {
                    bytes += numel * 4;
                }
            }
            sum += bytes;
            top = top.max(bytes);
        }
        (sum, top)
    }

    pub(super) fn reopen(&self) -> Result<Self> {
        let mut again = Self::open(&self.path)?;
        again.target = self.target.clone();
        again.weight_dtype = self.weight_dtype;
        again.tight = self.tight;
        again.weights.lean = self.weights.lean;
        again.weights.lora = self.weights.lora.clone();
        again.weights.scope = self.weights.scope.clone();
        Ok(again)
    }

    pub(super) fn open(path: &Path) -> Result<Self> {
        let pack = crate::pkg::open_rlxp(path)?;
        if !pack.has_graph() {
            // No graphs: a model block in `model.json` (functions imported,
            // or a program alone), else weights only.
            let meta: Option<Value> = pack
                .sidecar(MODEL_SIDECAR)
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok());
            let Some(meta) = meta.filter(|m: &Value| m.get("entries").is_some()) else {
                bail!(
                    "{} holds weights only (no graph); open it with a model provider for its \
                     architecture",
                    path.display()
                );
            };
            check_features(&meta, path)?;
            let weights = PackageWeights::new(path, &pack, meta.get("weights"))?;
            return Ok(Self {
                path: path.to_path_buf(),
                pack,
                graphs: Vec::new(),
                rlx_version: None,
                dims: Vec::new(),
                meta,
                weights,
                target: String::new(),
                weight_dtype: None,
                tight: false,
            });
        }
        let module = pack
            .module_with(rlx_pkg::MaterializeMode::HotOnly)
            .with_context(|| format!("graph of {}", path.display()))?;
        // The model block of the graph's header; packages from before it
        // carry the same object as the `model.json` sidecar.
        let meta: Value = module
            .model
            .clone()
            .or_else(|| {
                pack.sidecar(MODEL_SIDECAR)
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok())
            })
            .unwrap_or_else(|| json!({}));
        check_features(&meta, path)?;
        let weights = PackageWeights::new(path, &pack, meta.get("weights"))?;
        Ok(Self {
            path: path.to_path_buf(),
            pack,
            graphs: module.graphs,
            rlx_version: module.rlx,
            dims: module.dims,
            meta,
            weights,
            target: String::new(),
            weight_dtype: None,
            tight: false,
        })
    }

    /// Take the function `name` out of the package, specialized at each
    /// binding of `sizes` (`[{seq: 16}, {seq: 64}]`, by the header's symbol
    /// names) when it is symbolic: one graph per binding, named
    /// `name@seq=16`. A static function is one graph, as stored.
    pub fn take_function(&mut self, name: &str, sizes: Option<&Value>) -> Result<Vec<Graph>> {
        let graph = self.take_graph(name)?;
        if !graph.has_dynamic_dims() {
            return Ok(vec![graph]);
        }
        let sizes = sizes.and_then(Value::as_array).with_context(|| {
            format!("'{name}' is symbolic: its entry lists `sizes` to run it at")
        })?;
        sizes
            .iter()
            .map(|b| Ok(self.specialize_at(&graph, b)?.0))
            .collect()
    }

    /// Symbolic `graph` at one entry of `sizes` (`{seq: 16}`, by the
    /// header's symbol names): the graph, named `name@seq=16`, and the
    /// binding by symbol.
    pub fn specialize_at(&self, graph: &Graph, size: &Value) -> Result<(Graph, Vec<(u32, usize)>)> {
        let b = size
            .as_object()
            .with_context(|| format!("a size is {{symbol: n, …}}, found {size}"))?;
        let mut binding = Vec::new();
        for (sym_name, n) in b {
            let sym = self
                .dims
                .iter()
                .find(|(d, _)| d == sym_name)
                .map(|(_, s)| *s)
                .with_context(|| format!("size names `{sym_name}`, which no dim is"))?;
            let n = n
                .as_u64()
                .with_context(|| format!("size of `{sym_name}`: {n}"))?;
            binding.push((sym, n as usize));
        }
        let g = self.specialize(graph, &binding)?;
        Ok((g, binding))
    }

    /// Symbolic `graph` with each symbol of `binding` its size, named
    /// `name@<symbol>=<size>,…`.
    pub fn specialize(&self, graph: &Graph, binding: &[(u32, usize)]) -> Result<Graph> {
        let mut g = rlx_ir::dynamic::specialize(graph, &rlx_ir::DimBinding::from_pairs(binding));
        if g.has_dynamic_dims() {
            bail!(
                "'{}' at {binding:?} still has unbound dims: give every symbol a size",
                graph.name
            );
        }
        let label: Vec<String> = binding
            .iter()
            .map(|(sym, n)| {
                let name = self
                    .dims
                    .iter()
                    .find(|(_, s)| s == sym)
                    .map_or_else(|| format!("?{sym}"), |(d, _)| d.clone());
                format!("{name}={n}")
            })
            .collect();
        g.name = format!("{}@{}", graph.name, label.join(","));
        Ok(g)
    }

    /// Function `name` as packaged (symbolic dims stay): the build for
    /// this device, else the portable one.
    pub fn take_graph(&mut self, name: &str) -> Result<Graph> {
        let built = format!("{name}:{}", self.target);
        let i = self
            .graphs
            .iter()
            .position(|g| g.name == built)
            .or_else(|| self.graphs.iter().position(|g| g.name == name))
            .with_context(|| {
                format!("the model block names function '{name}', which the package lacks")
            })?;
        let mut graph = self.graphs.remove(i);
        graph.name = name.to_string();
        if let Some(dtype) = self.weight_dtype {
            rlx_ir::retarget::matmul_weights_as(&mut graph, dtype);
            rlx_ir::retarget::embedding_tables_as(&mut graph, dtype);
        }
        Ok(graph)
    }

    /// The model block's string `key`, else `default`.
    pub fn meta_str(&self, key: &str, default: &str) -> String {
        self.meta
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_string()
    }

    /// What `info().metadata` reports: the block's `metadata`, plus the
    /// package, its functions, and the model's name, version, authors,
    /// license, links and card.
    pub fn metadata(&self) -> Value {
        let meta = &self.meta;
        let mut metadata = meta.get("metadata").cloned().unwrap_or_else(|| json!({}));
        if let Some(m) = metadata.as_object_mut() {
            m.insert("package".into(), json!(self.pack.manifest().name));
            let functions: Vec<&str> = self.graphs.iter().map(|g| g.name.as_str()).collect();
            m.insert("functions".into(), json!(functions));
            for key in ["name", "version"] {
                if let Some(v) = meta.get(key) {
                    m.insert(format!("model_{key}"), v.clone());
                }
            }
            // Who made it, under what terms, and its card, as the model
            // block says.
            for key in [
                "authors",
                "license",
                "repository",
                "homepage",
                "card",
                "card_data",
            ] {
                if let Some(v) = meta.get(key) {
                    m.insert(key.into(), v.clone());
                }
            }
            if let Some(v) = &self.rlx_version {
                m.insert("rlx_version".into(), json!(v));
            }
        }
        metadata
    }

    /// A file the model block names (`tokenizer.json`): `sidecar:<id>` from
    /// the package, else a path beside the package (inside a directory one).
    pub fn file(&self, reference: &str) -> Result<Vec<u8>> {
        if let Some(id) = reference.strip_prefix("sidecar:") {
            return self
                .pack
                .sidecar(id)
                .with_context(|| format!("sidecar '{id}' of {}", self.path.display()));
        }
        let p = beside(&self.path, reference);
        std::fs::read(&p).with_context(|| format!("reading {}", p.display()))
    }
}

/// `reference` relative to a package: beside a file package, inside a
/// directory one; an absolute path as is.
pub(super) fn beside(package: &Path, reference: &str) -> PathBuf {
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

/// A graph's params: name, dtype, dims.
pub(super) fn params_of(g: &Graph) -> Vec<(String, rlx_ir::DType, Vec<usize>)> {
    g.nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some((
                name.clone(),
                n.shape.dtype(),
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        Dim::Static(n) => *n,
                        Dim::Dynamic(_) => 0,
                    })
                    .collect(),
            )),
            _ => None,
        })
        .collect()
}

impl RlxpModel {
    fn open(request: &ModelRequest, mut pkg: Opened) -> Result<Self> {
        let metadata = pkg.metadata();
        // The main graph, or its build for this device.
        let main = pkg
            .graphs
            .first()
            .map(|g| g.name.clone())
            .unwrap_or_default();
        let graph = pkg.take_function(&main, None)?.remove(0);
        let named: Vec<String> = pkg
            .meta
            .get("outputs")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let inputs = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(spec_of(&graph, n.id, name.clone())),
                _ => None,
            })
            .collect();
        let output_names: Vec<String> = (0..graph.outputs.len())
            .map(|i| {
                named
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("output{i}"))
            })
            .collect();
        let outputs = graph
            .outputs
            .iter()
            .zip(&output_names)
            .map(|(&id, name)| spec_of(&graph, id, name.clone()))
            .collect();
        let params = params_of(&graph);
        let mut compiled = pkg.weights.compile(request.device, graph);
        pkg.weights.bind(&mut compiled, &params)?;
        let info = ModelInfo {
            provider: "rlxp".into(),
            family: pkg.meta_str("family", "graph"),
            task: pkg.meta_str("task", "graph"),
            device: request.device_label(),
            inputs,
            outputs,
            metadata,
        };
        Ok(Self {
            compiled,
            info,
            output_names,
        })
    }
}

/// One `weights` entry of the model block: where a param the package does
/// not hold comes from.
struct WeightSource {
    /// `safetensors://…`, `gguf://…`, `rlxp://…`, … (no `#tensor`), its
    /// path relative to the package resolved.
    uri: String,
    prefix: String,
    map: serde_json::Map<String, Value>,
    /// The file's SHA-256 when it was referenced (hex), checked by
    /// [`PackageWeights::verify`].
    sha256: Option<String>,
}

/// The file a weight URI reads (`safetensors://a/b.safetensors` →
/// `a/b.safetensors`), if it names one.
fn uri_file(uri: &str) -> Option<PathBuf> {
    let (_, rest) = uri.split_once("://")?;
    let path = rest.split('#').next()?;
    Some(PathBuf::from(path))
}

/// The SHA-256 of the file at `path`, as lower-case hex: what a weight
/// source's `sha256` says.
pub fn file_sha256(path: &Path) -> Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// A `model.weights` entry for the file at `file`, referenced from a
/// package at `package` as `scheme://<relative path>`: its `uri`, `map`,
/// and the file's `size` and `sha256`, so a package notices when the file
/// under it changes.
pub fn weight_source(scheme: &str, uri_path: &str, file: &Path, map: Value) -> Result<Value> {
    Ok(json!({
        "uri": format!("{scheme}://{uri_path}"),
        "map": map,
        "size": std::fs::metadata(file).with_context(|| format!("{}", file.display()))?.len(),
        "sha256": file_sha256(file)?,
    }))
}

/// A package's weights: the tensors it holds, and the `weights` sources of
/// its model block for the rest. A referenced tensor is read once, in the
/// dtype its param declares (an f16 weight stays two bytes an element), so
/// binding another graph of the model (a second sequence length) reads
/// nothing again.
pub(super) struct PackageWeights {
    embedded: HashMap<String, (Vec<u8>, rlx_ir::DType)>,
    sources: Vec<WeightSource>,
    /// An adapter's low-rank updates, by checkpoint tensor (or held
    /// param): `scale · B·A`, laid out as the tensor is stored, added as it
    /// is read.
    lora: HashMap<String, Vec<f32>>,
    /// Compile what binds these weights lean (see [`Self::compile`]): set
    /// for a package short of memory, or opened with narrower weights.
    pub(super) lean: bool,
    /// Executables compiled from these weights share them where the backend
    /// can (`with_param_sharing_scope`): the package, and an adapter merged
    /// into it. Empty: no sharing.
    pub(super) scope: String,
    /// Opened checkpoint files, kept across reads: a GGUF's header (with a
    /// tokenizer's ~150k-entry vocabulary) took ~0.6 s to parse, and a fresh
    /// cache per read reparsed it for each of a model's hundreds of params.
    read_cache: std::sync::Mutex<crate::dist::WeightCache>,
}

impl PackageWeights {
    /// No weights (what is left after a model takes them).
    pub(super) fn empty() -> Self {
        Self {
            embedded: HashMap::new(),
            sources: Vec::new(),
            lora: HashMap::new(),
            lean: false,
            scope: String::new(),
            read_cache: Default::default(),
        }
    }

    fn new(package: &Path, pack: &rlx_pkg::Package, sources: Option<&Value>) -> Result<Self> {
        let embedded = pack
            .typed_weight_bindings()?
            .into_iter()
            .map(|(name, bytes, dtype)| (name, (bytes, dtype)))
            .collect();
        let sources = match sources {
            None => Vec::new(),
            Some(list) => list
                .as_array()
                .context("model.weights is a list of {uri, prefix?, map?}")?
                .iter()
                .map(|s| {
                    let uri = s
                        .get("uri")
                        .and_then(Value::as_str)
                        .context("a model.weights entry needs a `uri`")?;
                    let uri = source_uri(package, uri);
                    // A referenced file of another size is another file:
                    // its tensors would bind to the wrong params.
                    if let (Some(want), Some(file)) =
                        (s.get("size").and_then(Value::as_u64), uri_file(&uri))
                        && let Ok(meta) = std::fs::metadata(&file)
                        && meta.len() != want
                    {
                        bail!(
                            "{} is {} bytes; the package was made with {want} — the weights \
                             changed under it",
                            file.display(),
                            meta.len()
                        );
                    }
                    Ok(WeightSource {
                        uri,
                        prefix: s.get("prefix").and_then(Value::as_str).unwrap_or("").into(),
                        map: s
                            .get("map")
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default(),
                        sha256: s.get("sha256").and_then(Value::as_str).map(String::from),
                    })
                })
                .collect::<Result<_>>()?,
        };
        Ok(Self {
            embedded,
            sources,
            lora: HashMap::new(),
            lean: false,
            scope: format!("rlxp:{}", package.display()),
            read_cache: Default::default(),
        })
    }

    /// Param `name` (`dims`) as f32: held, or read from a source.
    pub(super) fn values(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        if let Some((bytes, dt)) = self.embedded.get(name) {
            return Ok(crate::backend::widen_bytes_to_f32(bytes, *dt));
        }
        let mut cache = self.read_cache.lock().unwrap_or_else(|e| e.into_inner());
        self.read(&mut cache, name, dims)
            .map_err(|e| anyhow::anyhow!("no weights for '{name}': {e}"))
    }

    /// One param's bytes in the dtype the graph declares it (`dtype`): packed
    /// blocks (U8/I8, GGUF K-quants behind a `DequantMatMul`) as stored,
    /// float params widened or narrowed to `dtype`. What [`Self::bind`] hands
    /// a backend, for a caller that ships it elsewhere (a cluster stage).
    #[cfg(feature = "training")]
    pub(super) fn typed(
        &self,
        name: &str,
        dtype: rlx_ir::DType,
        dims: &[usize],
    ) -> Result<Vec<u8>> {
        use rlx_ir::DType;
        if let Some((bytes, dt)) = self.embedded.get(name) {
            let mut b = bytes.clone();
            if let Some(delta) = self.lora.get(name) {
                let mut wide = crate::backend::widen_bytes_to_f32(bytes, *dt);
                wide.iter_mut().zip(delta).for_each(|(w, d)| *w += d);
                b = crate::backend::narrow_f32_to_bytes(&wide, *dt);
            }
            if *dt != dtype && dt.is_float() && dtype.is_float() {
                let wide = crate::backend::widen_bytes_to_f32(&b, *dt);
                return Ok(crate::backend::narrow_f32_to_bytes(&wide, dtype));
            }
            return Ok(b);
        }
        let mut cache = self.read_cache.lock().unwrap_or_else(|e| e.into_inner());
        let r = if matches!(dtype, DType::U8 | DType::I8) {
            self.read_bytes(&mut cache, name)
        } else if matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            self.read(&mut cache, name, dims)
                .map(|v| crate::backend::narrow_f32_to_bytes(&v, dtype))
        } else {
            bail!(
                "param '{name}' is {}: only f32/f16/bf16 and packed blocks ship",
                dtype_name(dtype)
            );
        };
        r.map_err(|e| anyhow::anyhow!("no weights for '{name}': {e}"))
    }

    /// Check every referenced file that names its `sha256` against it (open
    /// option `verify_weights`: a whole read of each file).
    pub(super) fn verify(&self) -> Result<()> {
        for s in &self.sources {
            let (Some(want), Some(file)) = (&s.sha256, uri_file(&s.uri)) else {
                continue;
            };
            let got = file_sha256(&file)?;
            if &got != want {
                bail!(
                    "{} has SHA-256 {got}; the package was made with {want}",
                    file.display()
                );
            }
        }
        Ok(())
    }

    /// A tensor the package holds that no graph binds (a RoPE table), as
    /// f32, taken out of the weights.
    pub(super) fn take_table(&mut self, name: &str) -> Result<Vec<f32>> {
        let (bytes, dtype) = self
            .embedded
            .remove(name)
            .with_context(|| format!("the package holds no tensor '{name}'"))?;
        Tensor::new(name, dtype, vec![], bytes).to_f32()
    }

    /// Bind `params` of `compiled`: the package's own tensors, then the
    /// referenced ones. With `weights` sources declared, a param none has
    /// is an error, not zeros.
    /// `graph` compiled for `device` to bind these weights: lean, without
    /// the fusions that keep a second, concatenated copy of weights (QKV,
    /// gate/up) resident, when memory is what matters.
    pub(super) fn compile(&self, device: crate::Device, graph: Graph) -> CompiledGraph {
        let compile = || {
            if !self.lean {
                return Session::new(device).compile(graph);
            }
            let mut opts = crate::CompileOptions::default();
            opts.fusion_opts.no_weight_concat = true;
            Session::new(device).compile_with(graph, &opts)
        };
        if self.scope.is_empty() {
            compile()
        } else {
            crate::with_param_sharing_scope(&self.scope, compile)
        }
    }

    pub(super) fn bind(
        &mut self,
        compiled: &mut CompiledGraph,
        params: &[(String, rlx_ir::DType, Vec<usize>)],
    ) -> Result<()> {
        let mut cache = crate::dist::WeightCache::new();
        let mut unresolved = Vec::new();
        let mut why = None;
        for (name, dtype, dims) in params {
            // Its storage is the copy another executable bound.
            if compiled.param_already_set(name) {
                continue;
            }
            if let Some((bytes, dt)) = self.embedded.get(name) {
                // A held tensor the size of what the graph declares: one of
                // another size (a package written wrong) would bind silently.
                let want: usize = dims.iter().product();
                let held = bytes.len() / dt.size_bytes();
                if !matches!(dt, rlx_ir::DType::U8 | rlx_ir::DType::I8) && want > 0 && held != want
                {
                    bail!("param '{name}' is {dims:?} ({want} values); the package holds {held}");
                }
                if let Some(delta) = self.lora.get(name) {
                    // An adapter's update to a held param.
                    let mut wide = crate::backend::widen_bytes_to_f32(bytes, *dt);
                    if wide.len() != delta.len() {
                        bail!(
                            "the adapter's update to '{name}' has {} values; it has {}",
                            delta.len(),
                            wide.len()
                        );
                    }
                    wide.iter_mut().zip(delta).for_each(|(w, d)| *w += d);
                    let out = crate::backend::narrow_f32_to_bytes(&wide, *dtype);
                    compiled.set_param_typed(name, &out, *dtype);
                    continue;
                }
                // A float param declared narrower or wider than the
                // package stores it gets its own dtype.
                if dt != dtype && dt.is_float() && dtype.is_float() {
                    let wide = crate::backend::widen_bytes_to_f32(bytes, *dt);
                    let narrow = crate::backend::narrow_f32_to_bytes(&wide, *dtype);
                    compiled.set_param_typed(name, &narrow, *dtype);
                } else {
                    compiled.set_param_typed(name, bytes, *dt);
                }
                continue;
            }
            if self.sources.is_empty() {
                continue;
            }
            // Read, handed to the backend, and dropped: the backend keeps its
            // copy, and a function bound later reads the file again rather
            // than every weight staying here too.
            let bytes = if matches!(dtype, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
                // Packed blocks (GGUF K-quants behind a `DequantMatMul`): the
                // tensor's bytes as stored.
                self.read_bytes(&mut cache, name)
            } else if matches!(
                dtype,
                rlx_ir::DType::F32 | rlx_ir::DType::F16 | rlx_ir::DType::BF16
            ) {
                self.read(&mut cache, name, dims)
                    .map(|values| crate::backend::narrow_f32_to_bytes(&values, *dtype))
            } else {
                bail!(
                    "param '{name}' is {}: referenced weights bind f32 / f16 / bf16 params",
                    dtype_name(*dtype)
                );
            };
            match bytes {
                Ok(bytes) => compiled.set_param_typed(name, &bytes, *dtype),
                Err(e) => {
                    why.get_or_insert(e);
                    unresolved.push(name.as_str());
                }
            }
        }
        if !unresolved.is_empty() {
            let shown = unresolved.len().min(8);
            bail!(
                "no weights for {} param(s) in the package or its model.weights sources ({}): \
                 {}{} ({})",
                unresolved.len(),
                self.sources
                    .iter()
                    .map(|s| s.uri.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                unresolved[..shown].join(", "),
                if shown < unresolved.len() {
                    ", …"
                } else {
                    ""
                },
                why.unwrap_or_default()
            );
        }
        Ok(())
    }

    /// The tensors the package holds, as it holds them.
    #[cfg(feature = "training")]
    pub(super) fn held(&self) -> Vec<Tensor> {
        let mut names: Vec<&String> = self.embedded.keys().collect();
        names.sort();
        names
            .into_iter()
            .map(|n| {
                let (bytes, dtype) = &self.embedded[n];
                Tensor::new(
                    n.clone(),
                    *dtype,
                    vec![bytes.len() / dtype.size_bytes().max(1)],
                    bytes.clone(),
                )
            })
            .collect()
    }

    /// The bytes and dtype of the tensor `name` the package holds.
    pub(super) fn held_bytes(&self, name: &str) -> Result<(Vec<u8>, rlx_ir::DType)> {
        self.embedded
            .get(name)
            .cloned()
            .with_context(|| format!("the package holds no '{name}'"))
    }

    /// Hold `bytes` as the tensor `name` (another package's, spliced in).
    pub(super) fn hold(&mut self, name: &str, bytes: Vec<u8>, dtype: rlx_ir::DType) {
        self.embedded.insert(name.to_string(), (bytes, dtype));
    }

    /// Where param `name`'s values come from, as an adapter names it: the
    /// tensor (a held param's own name, else its source's) and whether the
    /// tensor is stored transposed against the param.
    #[cfg(feature = "training")]
    pub(super) fn stored_layout(&self, name: &str) -> (String, bool) {
        if self.embedded.contains_key(name) {
            return (name.to_string(), false);
        }
        for src in &self.sources {
            match src.map.get(name) {
                Some(Value::String(t)) => return (t.clone(), false),
                Some(Value::Object(o)) => {
                    let t = o
                        .get("tensor")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("{}{name}", src.prefix), String::from);
                    let tr = o.get("transpose").and_then(Value::as_bool).unwrap_or(false);
                    return (t, tr);
                }
                _ => {}
            }
        }
        let prefix = self.sources.first().map_or("", |s| s.prefix.as_str());
        (format!("{prefix}{name}"), false)
    }

    /// Tensor `name`'s bytes as the package or its first source stores
    /// them, read now (a program's `weights` inputs: tensors no function
    /// binds, read per run, e.g. the experts a router picked).
    #[cfg(feature = "program")]
    pub(super) fn stored(
        &self,
        cache: &mut crate::dist::WeightCache,
        name: &str,
    ) -> Result<Vec<u8>> {
        if let Some((bytes, _)) = self.embedded.get(name) {
            return Ok(bytes.clone());
        }
        self.read_bytes(cache, name)
            .map_err(|e| anyhow::anyhow!("no weights for '{name}': {e}"))
    }

    /// `name`'s bytes as its first source stores them (a packed tensor).
    fn read_bytes(
        &self,
        cache: &mut crate::dist::WeightCache,
        name: &str,
    ) -> std::result::Result<Vec<u8>, String> {
        let mut why = String::new();
        for src in &self.sources {
            let tensor = match src.map.get(name) {
                Some(Value::String(t)) => t.clone(),
                Some(Value::Object(o)) => o
                    .get("tensor")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("{}{name}", src.prefix), String::from),
                _ => format!("{}{name}", src.prefix),
            };
            match cache.bytes(&format!("{}#{tensor}", src.uri)) {
                Ok(b) => return Ok(b),
                Err(e) => why = e,
            }
        }
        Err(why)
    }

    /// `name` from the first source that has it. A `map` entry is the
    /// file's tensor name, or `{tensor, transpose}` for a 2-D tensor stored
    /// the other way round.
    fn read(
        &self,
        cache: &mut crate::dist::WeightCache,
        name: &str,
        dims: &[usize],
    ) -> std::result::Result<Vec<f32>, String> {
        let mut why = String::new();
        for src in &self.sources {
            let (tensor, transpose) = match src.map.get(name) {
                Some(Value::String(t)) => (t.clone(), false),
                // `{tensor?, transpose?}`: the tensor defaults to the
                // param's own name (with the prefix).
                Some(Value::Object(o)) => (
                    o.get("tensor")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("{}{name}", src.prefix), String::from),
                    o.get("transpose").and_then(Value::as_bool).unwrap_or(false),
                ),
                _ => (format!("{}{name}", src.prefix), false),
            };
            let mut values = match cache.f32(&format!("{}#{tensor}", src.uri)) {
                Ok(v) => v,
                Err(e) => {
                    why = e;
                    continue;
                }
            };
            if let Some(delta) = self.lora.get(&tensor) {
                // An adapter's update, in the tensor's stored layout.
                if delta.len() != values.len() {
                    return Err(format!(
                        "the adapter's update to '{tensor}' has {} values; it has {}",
                        delta.len(),
                        values.len()
                    ));
                }
                values.iter_mut().zip(delta).for_each(|(w, d)| *w += d);
            }
            if !transpose {
                return Ok(values);
            }
            // Stored [cols, rows]; the param is [rows, cols].
            let [rows, cols] = dims else {
                return Err(format!(
                    "'{name}' is {dims:?}: only a 2-D tensor transposes"
                ));
            };
            if values.len() != rows * cols {
                return Err(format!(
                    "'{tensor}' has {} values, '{name}' {dims:?}",
                    values.len()
                ));
            }
            let mut out = vec![0f32; values.len()];
            for r in 0..*rows {
                for c in 0..*cols {
                    out[r * cols + c] = values[c * rows + r];
                }
            }
            return Ok(out);
        }
        Err(why)
    }
}

/// `safetensors://model.safetensors` relative to the package: beside a file
/// package, inside a directory one.
pub(super) fn source_uri(package: &Path, uri: &str) -> String {
    let Some((scheme, rest)) = uri.split_once("://") else {
        return uri.to_string();
    };
    format!("{scheme}://{}", beside(package, rest).display())
}

impl Model for RlxpModel {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], _options: &Value) -> Result<Vec<Tensor>> {
        for spec in &self.info.inputs {
            if !inputs.iter().any(|t| t.name == spec.name) {
                bail!(
                    "missing input '{}' ({} {:?})",
                    spec.name,
                    dtype_name(spec.dtype),
                    spec.shape
                );
            }
        }
        let typed: Vec<(&str, &[u8], rlx_ir::DType)> = inputs
            .iter()
            .map(|t| (t.name.as_str(), t.data.as_slice(), t.dtype))
            .collect();
        let outs = self.compiled.run_typed(&typed);
        Ok(outs
            .into_iter()
            .enumerate()
            .map(|(i, (data, dtype))| {
                let spec = self.info.outputs.get(i);
                let shape = spec
                    .map(|s| s.shape.iter().map(|d| d.unwrap_or(0)).collect())
                    .unwrap_or_default();
                let name = self
                    .output_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("output{i}"));
                Tensor::new(name, dtype, shape, data)
            })
            .collect())
    }
}

/// Write `graph` and its `weights` (f32 / any dtype, by tensor) as an `.rlxp`
/// package `path`, with `model` (`{task, family, outputs, metadata}`) as its
/// model block. The graph is not consumed.
pub fn save_package(path: &Path, graph: &Graph, weights: &[Tensor], model: &Value) -> Result<()> {
    save_module(path, std::slice::from_ref(graph), weights, model)
}

/// Write a model of several graphs (the main one first) and its `weights` as
/// an `.rlxp` package, with `model` as its model block. A weight the package
/// does not hold can come from the block's `weights` sources.
pub fn save_module(path: &Path, graphs: &[Graph], weights: &[Tensor], model: &Value) -> Result<()> {
    save_module_with_sidecars(path, graphs, weights, model, &[])
}

/// [`save_module`] plus files the model block refers to as `sidecar:<id>`
/// (a `tokenizer.json`): `(id, media type, bytes)`.
pub fn save_module_with_sidecars(
    path: &Path,
    graphs: &[Graph],
    weights: &[Tensor],
    model: &Value,
    sidecars: &[(&str, &str, Vec<u8>)],
) -> Result<()> {
    let mut module = rlx_ir::rlxmodel::Module::new(graphs.to_vec());
    module.model = Some(model.clone());
    save_package_module(path, &module, weights, sidecars)
}

/// Write a whole `.rlxmodel` module (its graphs, model block, symbol names
/// and `requires`) and its `weights` as an `.rlxp` package, with sidecars.
pub fn save_package_module(
    path: &Path,
    module: &rlx_ir::rlxmodel::Module,
    weights: &[Tensor],
    sidecars: &[(&str, &str, Vec<u8>)],
) -> Result<()> {
    let mut model = module.model.clone().unwrap_or_else(|| json!({}));
    // What an rlx must run to open it: an older one refuses what it lacks.
    if model.get("entries").is_some() && model.get("features").is_none() {
        model["features"] = json!(features_of(&model));
    }
    let model = &model;
    // A package of no graphs (its functions imported, or a program alone):
    // its model block the `model.json` sidecar.
    let none = rlx_ir::Graph::new("none");
    let (graph, extra) = match module.graphs.split_first() {
        Some((g, rest)) => (g, rest),
        None => (&none, &[][..]),
    };
    let include_graph = !module.graphs.is_empty();
    let mut sidecars: Vec<(String, String, Vec<u8>)> = sidecars
        .iter()
        .map(|(id, media, bytes)| (id.to_string(), media.to_string(), bytes.clone()))
        .collect();
    if !include_graph {
        sidecars.push((
            MODEL_SIDECAR.to_string(),
            "application/json".to_string(),
            serde_json::to_vec(model)?,
        ));
    }
    let packed: Vec<rlx_pkg::PackedWeight> = weights
        .iter()
        .map(|t| {
            rlx_pkg::PackedWeight::hot(
                t.name.clone(),
                t.shape.clone(),
                dtype_name(t.dtype),
                "row_major",
                t.data.clone(),
            )
        })
        .collect();
    let name = model
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or(&graph.name)
        .to_string();
    let opts = rlx_pkg::WriteOptions {
        name,
        producer: Some("rlx-runtime".into()),
        model: Some(model.clone()),
        requires_rlx: module.requires.clone(),
        dims: module.dims.clone(),
        extra_graphs: extra.to_vec(),
        sidecars,
        include_graph,
        ..Default::default()
    };
    rlx_pkg::write_package(path, graph, &packed, &opts)
        .with_context(|| format!("writing {}", path.display()))
}

pub use rlx_pkg::build::{BuildReport, TargetRecipe, build_for_target, build_for_targets};

/// The entry features this rlx runs: each entry, and each key of it. A
/// package lists those it uses (`features`, written with it); one using a
/// feature not here is refused by name, rather than run without it.
pub const FEATURES: &[&str] = &[
    "entry.kind",
    "embed",
    "embed.functions",
    "embed.sizes",
    "embed.tokenizer",
    "embed.inputs",
    "embed.pooling",
    "embed.normalize",
    "embed.position_offset",
    "score",
    "score.functions",
    "score.sizes",
    "score.tokenizer",
    "score.inputs",
    "score.position_offset",
    "score.label",
    "score.activation",
    "generate",
    "generate.functions",
    "generate.sizes",
    "generate.tokenizer",
    "generate.chat",
    "generate.stop_ids",
    "generate.vocab_size",
    "generate.inputs",
    "generate.state",
    "generate.state_update",
    "generate.state_kinds",
    "generate.logits",
    "generate.draft_logits",
    "generate.verify",
    "generate.prefill",
    "generate.max_context",
    "generate.batch",
    "generate.check",
    "generate.generation",
    "generate.media",
    "generate.rope",
    "transcribe",
    "transcribe.sample_rate",
    "transcribe.stages",
    "transcribe.decoder",
    "transcribe.tokenizer",
    "transcribe.prompt_ids",
    "transcribe.stop_ids",
    "transcribe.suppress_ids",
    "transcribe.begin_suppress_ids",
    "transcribe.max_tokens",
    "transcribe.audio_multiple",
    "transcribe.token_budget",
    "synthesize",
    "synthesize.tokenizer",
    "synthesize.template",
    "synthesize.sample_rate",
    "synthesize.stages",
    "synthesize.voices",
    "synthesize.decoder",
    "synthesize.token_offset",
    "synthesize.codebook",
    "synthesize.max_tokens",
    "synthesize.output",
    "pipeline",
    "pipeline.task",
    "pipeline.inputs",
    "pipeline.stages",
    "pipeline.outputs",
    "pipeline.loop",
    "pipeline.finish",
    "program",
    "program.module",
    "program.tasks",
    "program.task",
    "program.config",
];

/// The features a model block uses: each entry and each of its keys
/// (`generate`, `generate.check`, …).
pub fn features_of(model: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for (name, kind, entry) in entries_of(model) {
        out.push(kind.clone());
        // An entry named apart from its kind: an rlx reading entries by
        // kind would not find it.
        if name != kind {
            out.push("entry.kind".into());
        }
        if let Some(keys) = entry.as_object() {
            out.extend(
                keys.iter()
                    .filter(|(k, v)| !v.is_null() && *k != "kind")
                    .map(|(k, _)| format!("{kind}.{k}")),
            );
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Refuse a package that uses a feature this rlx does not run.
fn check_features(meta: &Value, path: &Path) -> Result<()> {
    let Some(list) = meta.get("features").and_then(Value::as_array) else {
        return Ok(());
    };
    let missing: Vec<&str> = list
        .iter()
        .filter_map(Value::as_str)
        .filter(|f| !FEATURES.contains(f))
        .collect();
    if !missing.is_empty() {
        bail!(
            "{} uses {} that this rlx ({}) does not run; update rlx",
            path.display(),
            missing.join(", "),
            rlx_ir::rlxmodel::RLX_VERSION
        );
    }
    Ok(())
}

/// What [`quantize_package`] did.
#[derive(Debug, Default)]
pub struct QuantizeReport {
    /// Matmul weights stored quantized.
    pub weights: usize,
    /// Their f32 bytes, and their quantized bytes.
    pub f32_bytes: usize,
    pub quantized_bytes: usize,
}

/// The package at `input` with its f32 matmul weights quantized and held
/// in it, written to `output`: each `MatMul(x, w)` whose `w` is a `[k, n]`
/// f32 param (`k` a multiple of 32) becomes a `DequantMatMul` over `w`'s
/// GGUF `scheme` blocks (`GgufQ8_0`, `GgufQ4_0`; `[n, k]` rows), read from
/// wherever the package had `w` (held, or referenced). The rest is as it
/// was; referenced weights stay referenced, their paths rebased.
pub fn quantize_package(
    input: &Path,
    output: &Path,
    scheme: rlx_ir::quant::QuantScheme,
) -> Result<QuantizeReport> {
    use rlx_ir::quant::QuantScheme;
    let quantize: fn(&[f32]) -> anyhow::Result<Vec<u8>> = match scheme {
        QuantScheme::GgufQ8_0 => |v| rlx_gguf::quantize::quantize_q8_0(v).map_err(Into::into),
        QuantScheme::GgufQ4_0 => |v| rlx_gguf::quantize::quantize_q4_0(v).map_err(Into::into),
        other => bail!("quantize_package stores gguf_q8_0 or gguf_q4_0 weights, not {other}"),
    };
    let mut pkg = Opened::open(input)?;
    // The weights to quantize: f32 [k, n] params a MatMul multiplies by.
    let mut targets: HashMap<String, (usize, usize)> = HashMap::new();
    for g in &pkg.graphs {
        for n in g.nodes() {
            if !matches!(n.op, Op::MatMul) || n.inputs.len() != 2 {
                continue;
            }
            let w = g.node(n.inputs[1]);
            let Op::Param { name } = &w.op else { continue };
            let dims = w.shape.dims();
            if w.shape.dtype() != rlx_ir::DType::F32 || dims.len() != 2 {
                continue;
            }
            let (Dim::Static(k), Dim::Static(nn)) = (dims[0], dims[1]) else {
                continue;
            };
            if k % 32 == 0 && rows_of(&g.node(n.inputs[0]).shape, k).is_some() {
                targets.insert(name.clone(), (k, nn));
            }
        }
    }
    let mut report = QuantizeReport::default();
    let mut held: Vec<Tensor> = Vec::new();
    let mut names: Vec<&String> = targets.keys().collect();
    names.sort();
    for name in names {
        let (k, n) = targets[name];
        let w = pkg.weights.values(name, &[k, n])?;
        // [k, n] → GGUF's [n, k] rows.
        let mut rows = vec![0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                rows[j * k + i] = w[i * n + j];
            }
        }
        let bytes = quantize(&rows)?;
        report.weights += 1;
        report.f32_bytes += 4 * k * n;
        report.quantized_bytes += bytes.len();
        held.push(Tensor::new(
            format!("{name}.{scheme}"),
            rlx_ir::DType::U8,
            vec![bytes.len()],
            bytes,
        ));
    }
    let graphs: Vec<Graph> = std::mem::take(&mut pkg.graphs)
        .iter()
        .map(|g| quantized_matmuls(g, &targets, &held, scheme))
        .collect::<Result<_>>()?;
    // The package's own tensors that some function still reads.
    let read: std::collections::HashSet<String> = graphs
        .iter()
        .flat_map(|g| params_of(g).into_iter().map(|(n, _, _)| n))
        .collect();
    for (name, bytes, dtype) in pkg.pack.typed_weight_bindings()? {
        if read.contains(&name) {
            let shape = vec![bytes.len() / dtype.size_bytes().max(1)];
            held.push(Tensor::new(name, dtype, shape, bytes));
        }
    }
    let mut model = pkg.meta.clone();
    rlx_pkg::build::rebase_weight_uris(&mut model, input, output);
    if let Some(m) = model.as_object_mut() {
        m.remove("features");
        m.insert("quantized".into(), json!(scheme.to_string()));
    }
    let mut module = rlx_ir::rlxmodel::Module::new(graphs);
    module.dims = pkg.dims.clone();
    module.model = Some(model);
    let sidecars: Vec<(String, String, Vec<u8>)> = pkg
        .pack
        .sidecar_ids()
        .into_iter()
        .filter(|(id, _)| id != MODEL_SIDECAR)
        .map(|(id, media)| {
            let bytes = pkg.pack.sidecar(&id)?;
            Ok((
                id,
                media.unwrap_or_else(|| "application/octet-stream".into()),
                bytes,
            ))
        })
        .collect::<Result<_>>()?;
    let sidecars: Vec<(&str, &str, Vec<u8>)> = sidecars
        .iter()
        .map(|(i, m, b)| (i.as_str(), m.as_str(), b.clone()))
        .collect();
    save_package_module(output, &module, &held, &sidecars)?;
    Ok(report)
}

/// The rows a `[…, k]` activation has as a `[rows, k]` matrix: all static,
/// or one symbolic dim among ones. `None` otherwise.
fn rows_of(shape: &rlx_ir::Shape, k: usize) -> Option<Dim> {
    let dims = shape.dims();
    let (last, lead) = dims.split_last()?;
    if *last != Dim::Static(k) {
        return None;
    }
    let dynamic: Vec<&Dim> = lead
        .iter()
        .filter(|d| matches!(d, Dim::Dynamic(_)))
        .collect();
    match dynamic.len() {
        0 => Some(Dim::Static(
            lead.iter().map(|d| d.unwrap_static()).product(),
        )),
        1 if lead
            .iter()
            .all(|d| matches!(d, Dim::Dynamic(_)) || *d == Dim::Static(1)) =>
        {
            Some(*dynamic[0])
        }
        _ => None,
    }
}

/// `graph` with each `MatMul` by a quantized weight a `DequantMatMul` over
/// its blocks (reshaped to `[rows, k]` and back), what no longer reads the
/// f32 weight pruned.
fn quantized_matmuls(
    graph: &Graph,
    targets: &HashMap<String, (usize, usize)>,
    held: &[Tensor],
    scheme: rlx_ir::quant::QuantScheme,
) -> Result<Graph> {
    fn push(
        g: &mut Graph,
        op: Op,
        inputs: Vec<rlx_ir::NodeId>,
        shape: rlx_ir::Shape,
        name: Option<String>,
    ) -> rlx_ir::NodeId {
        let id = g.add_node(op, inputs, shape);
        g.node_mut(id).name = name;
        id
    }
    let g = rlx_ir::stateful::rebuild(graph, |g, _, node, inputs| {
        if !matches!(node.op, Op::MatMul) || inputs.len() != 2 {
            return Ok(None);
        }
        let Op::Param { name } = &graph.node(node.inputs[1]).op else {
            return Ok(None);
        };
        let Some(&(k, n)) = targets.get(name) else {
            return Ok(None);
        };
        let packed_name = format!("{name}.{scheme}");
        let bytes = held
            .iter()
            .find(|t| t.name == packed_name)
            .map_or(0, |t| t.data.len());
        let x_shape = g.node(inputs[0]).shape.clone();
        let rows = rows_of(&x_shape, k).ok_or("activation shape")?;
        let f32s = |d: &[Dim]| rlx_ir::Shape::from_dims(d, rlx_ir::DType::F32);
        let flat = push(
            g,
            Op::Reshape {
                new_shape: vec![-1, k as i64],
            },
            vec![inputs[0]],
            f32s(&[rows, Dim::Static(k)]),
            None,
        );
        let packed = push(
            g,
            Op::Param {
                name: packed_name.clone(),
            },
            vec![],
            rlx_ir::Shape::new(&[bytes], rlx_ir::DType::U8),
            Some(packed_name),
        );
        let y = push(
            g,
            Op::DequantMatMul { scheme },
            vec![flat, packed],
            f32s(&[rows, Dim::Static(n)]),
            None,
        );
        let back = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(v) => *v as i64,
                Dim::Dynamic(_) => -1,
            })
            .collect();
        Ok(Some(push(
            g,
            Op::Reshape { new_shape: back },
            vec![y],
            node.shape.clone(),
            node.name.clone(),
        )))
    })
    .map_err(|e| anyhow::anyhow!("quantizing '{}': {e}", graph.name))?;
    let mut g = rlx_ir::retarget::prune(&g);
    g.name = graph.name.clone();
    Ok(g)
}

/// One update of an adapter: the checkpoint tensor (or held param) it
/// changes, its low-rank factors `a` `[rank, in]` and `b` `[out, rank]`, and
/// its scale (LoRA's `alpha / rank`).
pub struct LoraUpdate {
    pub tensor: String,
    pub rank: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub scale: f32,
}

/// Write an adapter package at `path`: `updates` over the package at
/// `base` (relative to `path`'s directory). Opened, it is the base with
/// `W + scale · B·A` for each updated tensor.
pub fn save_adapter(path: &Path, name: &str, base: &str, updates: &[LoraUpdate]) -> Result<()> {
    let mut g = Graph::new("adapter");
    let x = g.input("x", rlx_ir::Shape::new(&[1], rlx_ir::DType::F32));
    g.set_outputs(vec![x]);
    let mut tensors = Vec::new();
    let mut lora = Vec::new();
    for (i, u) in updates.iter().enumerate() {
        let (a, b) = (format!("lora.{i}.a"), format!("lora.{i}.b"));
        tensors.push(Tensor::f32(
            a.clone(),
            vec![u.rank, u.a.len() / u.rank.max(1)],
            &u.a,
        ));
        tensors.push(Tensor::f32(
            b.clone(),
            vec![u.b.len() / u.rank.max(1), u.rank],
            &u.b,
        ));
        lora.push(json!({"tensor": u.tensor, "a": a, "b": b, "rank": u.rank, "scale": u.scale}));
    }
    let model = json!({"name": name, "task": "adapter", "adapter": {"base": base, "lora": lora}});
    save_module(path, &[g], &tensors, &model)
}

/// The Hugging Face `pipeline_tag` of an rlx task, for a model card.
pub fn pipeline_tag(task: &str) -> Option<&'static str> {
    Some(match task {
        "text-generation" => "text-generation",
        "embedding" => "feature-extraction",
        "speech-recognition" => "automatic-speech-recognition",
        "text-to-speech" => "text-to-speech",
        _ => return None,
    })
}

/// A Hugging Face model card (`README.md`) for a model block, to publish a
/// package on the Hub: `card_data` as the YAML front matter (Hugging Face's
/// own keys: `license`, `language`, `tags`, `datasets`, `base_model`,
/// `model-index`, …), filled in from the block where it has no value
/// (`license`, `pipeline_tag` from `task`, `library_name: rlx`), then `card`,
/// the Markdown body. Values are written as JSON, which is YAML.
pub fn huggingface_readme(model: &Value) -> String {
    let mut data = model
        .get("card_data")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(l) = model.get("license") {
        data.entry("license").or_insert_with(|| l.clone());
    }
    if let Some(t) = model
        .get("task")
        .and_then(Value::as_str)
        .and_then(pipeline_tag)
    {
        data.entry("pipeline_tag").or_insert_with(|| json!(t));
    }
    data.entry("library_name").or_insert_with(|| json!("rlx"));
    let mut out = String::from("---\n");
    for (k, v) in &data {
        out.push_str(k);
        out.push_str(": ");
        out.push_str(&v.to_string());
        out.push('\n');
    }
    out.push_str("---\n");
    match model.get("card").and_then(Value::as_str) {
        Some(body) => out.push_str(body),
        None => {
            let name = model.get("name").and_then(Value::as_str).unwrap_or("model");
            out.push_str(&format!("\n# {name}\n"));
        }
    }
    out
}
