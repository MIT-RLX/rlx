// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiled graph — the hot-path execution object.

use crate::backend::ExecutableGraph;
use rlx_driver::Device;
use std::collections::{HashMap, HashSet};

/// Param-invariant "prepare" stage attached to a compiled graph.
///
/// Holds a separately-compiled weight-only graph (`prepare`) whose outputs are
/// the boundary tensors of the main graph. Run once (lazily, on the first
/// forward / `finalize_params`), its outputs are bound into the main graph via
/// [`CompiledGraph::bind_handle`] so weight-derived compute happens once instead
/// of every forward. See [`rlx_compile::split_param_invariant`].
struct Staging {
    prepare: Box<CompiledGraph>,
    prepared: bool,
    /// Boundary input names on the main graph, in `prepare`-output order.
    boundary: Vec<String>,
    prepare_params: HashSet<String>,
    main_params: HashSet<String>,
    /// True once the prepared outputs were injected via persistent `bind_handle`
    /// (CPU). When false (backends without persistent handles, e.g. CUDA), the
    /// prepared values are fed as ordinary inputs each forward from `values`.
    bound: bool,
    /// Prepared boundary values, kept only for the feed-each-forward fallback.
    values: Vec<Vec<f32>>,
}

/// A compiled graph ready for execution.
///
/// Created by [`crate::Session::compile`]. Holds the fused + memory-planned
/// graph and all pre-allocated execution state. Call
/// [`CompiledGraph::run`] repeatedly with different inputs — zero
/// allocation per call.
pub struct CompiledGraph {
    inner: Box<dyn ExecutableGraph>,
    /// Trailing outputs a [`crate::taps`] tap appended at compile time:
    /// stripped from every run and handed to the tap. 0 for ordinary graphs.
    tap_extra: usize,
    device: Device,
    /// Optional param-invariant prepare stage (None = ordinary graph).
    staging: Option<Box<Staging>>,
    /// Declared dtype of each named `Input` / `Param`, for boundaries that are
    /// not `f32`.
    ///
    /// [`Self::run`] and [`Self::set_param`] take `&[f32]`, which is the right
    /// shape for the overwhelmingly common case and wrong for an `F16`/`BF16`
    /// tensor: the bytes used to go to the backend verbatim and be read as
    /// pairs of halves. Feeding `[1,2,3,4]` to a BF16 input and adding zero
    /// returned `[2.0038757, 4.007843, 0.0, 0.0]` — no error, no panic.
    ///
    /// Empty means "not known" (e.g. compiled straight from HIR), in which case
    /// the f32 fast path is taken unchanged.
    io_dtypes: HashMap<String, rlx_ir::DType>,
    /// Round weights to a number format as they are uploaded (see
    /// [`CompileOptions::number_format`](crate::CompileOptions::number_format)).
    /// `None` uploads them as given.
    upload_rounding: Option<UploadRounding>,
    /// F64 inputs / params / outputs lowered to double-single pairs
    /// (`rlx_compile::lower_f64`); converted here so callers keep passing f64
    /// (or f32) values.
    df: Option<DfBoundary>,
    /// Params the graph declares F64 that callers upload as f32 or narrower
    /// (`NumberFormat::F64` promoted them); widened on upload.
    widened: std::collections::HashSet<String>,
    /// The declared inputs / params, for checking what callers feed
    /// ([`crate::io_check`]). `None` when the graph was not at hand.
    io: Option<Box<crate::io_check::IoSpec>>,
    /// The unbound-names warning has been considered (first run only).
    io_checked: bool,
}

/// Which boundary values of a [`CompiledGraph`] are double-single pairs
/// standing for F64.
#[derive(Debug, Clone, Default)]
pub(crate) struct DfBoundary {
    pub inputs: std::collections::HashSet<String>,
    pub params: std::collections::HashSet<String>,
    pub outputs: Vec<usize>,
}

impl DfBoundary {
    fn f32_pairs(v: &[f32]) -> Vec<f32> {
        v.iter().flat_map(|&x| [x, 0.0]).collect()
    }
    fn f64_bytes_pairs(b: &[u8]) -> Vec<u8> {
        let v: Vec<f64> = b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        rlx_ir::double_single::split_f64(&v)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect()
    }
    fn pairs_to_f64_bytes(b: &[u8]) -> Vec<u8> {
        let p: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        rlx_ir::double_single::join_f64(&p)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect()
    }
}

/// Weight rounding applied on upload: one format, and the row length of each
/// rounded param (blocks never cross a row).
#[derive(Debug, Clone)]
pub(crate) struct UploadRounding {
    pub format: rlx_ir::FloatFormat,
    pub scaling: rlx_ir::FormatScaling,
    pub row_len: HashMap<String, usize>,
}

impl UploadRounding {
    /// Rounded copy of `data` for `name`, or `None` when `name` is not rounded.
    fn apply(&self, name: &str, data: &[f32]) -> Option<Vec<f32>> {
        let row = *self.row_len.get(name)?;
        let mut v = data.to_vec();
        self.format.round_slice_scaled(&mut v, self.scaling, row);
        Some(v)
    }
}

impl Clone for CompiledGraph {
    /// Deep-clones the underlying executable via `ExecutableGraph::clone_box`.
    /// Backends that don't support cloning will panic at this point.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
            tap_extra: self.tap_extra,
            device: self.device,
            io_dtypes: self.io_dtypes.clone(),
            upload_rounding: self.upload_rounding.clone(),
            df: self.df.clone(),
            widened: self.widened.clone(),
            io: self.io.clone(),
            io_checked: self.io_checked,
            staging: self.staging.as_ref().map(|s| {
                Box::new(Staging {
                    prepare: s.prepare.clone(),
                    prepared: s.prepared,
                    boundary: s.boundary.clone(),
                    prepare_params: s.prepare_params.clone(),
                    main_params: s.main_params.clone(),
                    bound: s.bound,
                    values: s.values.clone(),
                })
            }),
        }
    }
}

impl CompiledGraph {
    /// Round the params named in `rounding` on every later upload.
    pub(crate) fn set_upload_rounding(&mut self, rounding: Option<UploadRounding>) {
        self.upload_rounding = rounding;
    }

    /// Widen the named params to f64 on upload (see [`Self::widened`]).
    pub(crate) fn set_widened_params(&mut self, names: Vec<String>) {
        self.widened = names.into_iter().collect();
    }

    /// Convert the boundary values `df` names between f64 and df pairs.
    pub(crate) fn set_df_boundary(&mut self, df: Option<DfBoundary>) {
        self.df = df;
    }

    fn df_input(&self, name: &str) -> bool {
        self.df.as_ref().is_some_and(|d| d.inputs.contains(name))
    }

    fn df_param(&self, name: &str) -> bool {
        self.df.as_ref().is_some_and(|d| d.params.contains(name))
    }

    /// `run` / `run_typed` for a graph with double-single boundary values.
    fn run_df(&mut self, inputs: &[(&str, &[u8], rlx_ir::DType)]) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        use rlx_ir::DType;
        let owned: Vec<(&str, Vec<u8>, DType)> = inputs
            .iter()
            .map(|&(n, b, dt)| match (self.df_input(n), dt) {
                (true, DType::F64) => (n, DfBoundary::f64_bytes_pairs(b), DType::F32),
                (true, DType::F32) => {
                    let v: Vec<f32> = b
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    let p = DfBoundary::f32_pairs(&v);
                    (
                        n,
                        p.iter().flat_map(|x| x.to_le_bytes()).collect(),
                        DType::F32,
                    )
                }
                (true, other) => panic!("input `{n}` is F64; fed as {other:?}"),
                (false, _) => (n, b.to_vec(), dt),
            })
            .collect();
        let typed: Vec<(&str, &[u8], DType)> = owned
            .iter()
            .map(|(n, b, d)| (*n, b.as_slice(), *d))
            .collect();
        self.ensure_prepared();
        let outs = self.inner.run_typed(&typed);
        let df_outs = self
            .df
            .as_ref()
            .map(|d| d.outputs.clone())
            .unwrap_or_default();
        outs.into_iter()
            .enumerate()
            .map(|(i, (b, dt))| match df_outs.contains(&i) {
                true => (DfBoundary::pairs_to_f64_bytes(&b), DType::F64),
                false => (b, dt),
            })
            .collect()
    }

    pub(crate) fn new(inner: Box<dyn ExecutableGraph>, device: Device) -> Self {
        Self {
            inner,
            tap_extra: 0,
            device,
            staging: None,
            io_dtypes: HashMap::new(),
            upload_rounding: None,
            df: None,
            widened: Default::default(),
            io: None,
            io_checked: false,
        }
    }

    /// Record the declared boundary checked by [`Self::run`] /
    /// [`Self::set_param`] (see [`crate::io_check`]).
    pub(crate) fn set_io_spec(&mut self, spec: crate::io_check::IoSpec) {
        self.io = Some(Box::new(spec));
    }

    /// Check one call's inputs; on the first run, warn once about inputs and
    /// params left unbound.
    /// [`Self::check_run`] for an `&[f32]` call: each buffer's length as an
    /// element count (a complex input carries two `f32`s per element).
    fn check_f32_run(&mut self, inputs: &[(&str, &[f32])]) -> Result<(), crate::io_check::IoError> {
        let elems: Vec<(&str, Option<usize>)> = inputs
            .iter()
            .map(|(n, d)| {
                let len = match self.io.as_ref() {
                    Some(io) => io.f32_elems(n, d.len()),
                    None => d.len(),
                };
                (*n, Some(len))
            })
            .collect();
        self.check_run(elems.into_iter())
    }

    fn check_run<'a>(
        &mut self,
        inputs: impl Iterator<Item = (&'a str, Option<usize>)> + Clone,
    ) -> Result<(), crate::io_check::IoError> {
        let Some(io) = self.io.as_mut() else {
            return Ok(());
        };
        io.check_inputs(inputs.clone())?;
        if !self.io_checked {
            self.io_checked = true;
            // The prepare stage's boundary values are fed by this wrapper, not
            // the caller.
            if let Some(st) = self.staging.as_ref() {
                for b in &st.boundary {
                    io.mark_input_bound(b);
                }
            }
            io.warn_unbound_once(inputs.map(|(n, _)| n));
        }
        Ok(())
    }

    /// [`Self::run`] that returns a misnamed input or a wrong-length buffer
    /// as an error instead of panicking.
    pub fn try_run(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, crate::io_check::IoError> {
        let inputs = self.upload_param_inputs(inputs)?;
        self.check_f32_run(&inputs)?;
        let outs = self.run_untapped(&inputs);
        Ok(crate::taps::on_run(outs, self.tap_extra))
    }

    /// The graph's declared inputs (name, dtype, static shape), sorted by
    /// name — what [`Self::run`] expects. Empty when compiled without the
    /// graph at hand (e.g. straight from HIR).
    pub fn inputs(&self) -> Vec<crate::io_check::TensorInfo> {
        self.io
            .as_ref()
            .map(|io| io.input_infos())
            .unwrap_or_default()
    }

    /// The graph's declared params, sorted by name — what
    /// [`Self::set_param`] takes.
    pub fn params(&self) -> Vec<crate::io_check::TensorInfo> {
        self.io
            .as_ref()
            .map(|io| io.param_infos())
            .unwrap_or_default()
    }

    /// The graph's outputs in the order [`Self::run`] returns them; `name`
    /// is the one given with [`rlx_ir::Graph::name_output`] (empty if none).
    pub fn outputs(&self) -> Vec<crate::io_check::TensorInfo> {
        self.io
            .as_ref()
            .map(|io| io.output_infos().to_vec())
            .unwrap_or_default()
    }

    /// The position in [`Self::run`]'s result of the output named `name`.
    pub fn output_index(&self, name: &str) -> Result<usize, crate::io_check::IoError> {
        match self.io.as_ref() {
            Some(io) => io.output_index(name),
            None => Err(crate::io_check::IoError::UnknownOutput {
                name: name.to_string(),
                closest: Vec::new(),
            }),
        }
    }

    /// Declare inputs / params that are meant to read zeros when not given
    /// (an empty KV-cache state, an unused bias), so the first `run` does not
    /// warn about them and [`Self::check_bindings`] accepts them.
    pub fn allow_unbound(&mut self, names: &[&str]) {
        if let Some(io) = self.io.as_mut() {
            for n in names {
                io.mark_input_bound(n);
                let _ = io.note_param(n, None);
            }
        }
    }

    /// Inputs `supplied` would leave unbound and params never set, as an
    /// error; `Ok` when everything the graph declares is bound.
    pub fn check_bindings(&self, supplied: &[&str]) -> Result<(), crate::io_check::IoError> {
        let Some(io) = self.io.as_ref() else {
            return Ok(());
        };
        match io.unbound(supplied.iter().copied()) {
            crate::io_check::IoError::Unbound { inputs, params }
                if inputs.is_empty() && params.is_empty() =>
            {
                Ok(())
            }
            e => Err(e),
        }
    }

    /// How many trailing outputs a tap appended (see [`crate::taps`]).
    pub(crate) fn set_tap_extra(&mut self, n: usize) {
        self.tap_extra = n;
    }

    /// Record the declared dtype of each named `Input` / `Param`.
    ///
    /// Called by the compile paths that still have the graph in hand. Without
    /// it the f32 entry points cannot tell a genuine `f32` boundary from an
    /// `F16`/`BF16` one being fed as f32 — see [`Self::io_dtypes`].
    pub(crate) fn set_io_dtypes(
        &mut self,
        dtypes: impl IntoIterator<Item = (String, rlx_ir::DType)>,
    ) {
        self.io_dtypes = dtypes.into_iter().collect();
    }

    /// Declared dtype of a named boundary tensor when it is a float NARROWER
    /// than `f32` — the only case the f32 entry points get wrong.
    ///
    /// Deliberately not "any dtype that is not F32". `F64` boundaries (the SPD /
    /// Riemannian-geometry graphs declare theirs that way) already round-trip
    /// through the plain f32 path, and routing them here instead converted the
    /// values and returned garbage — every `spd_host_parity` test failed with
    /// max|Δ| in the single digits. Integer boundaries are likewise widened by
    /// the backends already and must not be touched here.
    fn narrow_boundary(&self, name: &str) -> Option<rlx_ir::DType> {
        match self.io_dtypes.get(name) {
            Some(&dt) if matches!(dt, rlx_ir::DType::F16 | rlx_ir::DType::BF16) => Some(dt),
            _ => None,
        }
    }

    /// Attach a param-invariant `prepare` stage to a compiled MAIN graph.
    /// `prepare` and `boundary` come from [`rlx_compile::split_param_invariant`].
    pub(crate) fn with_staging(
        mut self,
        prepare: CompiledGraph,
        boundary: Vec<String>,
        prepare_params: HashSet<String>,
        main_params: HashSet<String>,
    ) -> Self {
        self.staging = Some(Box::new(Staging {
            prepare: Box::new(prepare),
            prepared: false,
            boundary,
            prepare_params,
            main_params,
            bound: false,
            values: Vec::new(),
        }));
        self
    }

    /// Run the `prepare` stage once (lazily) and make its outputs available to
    /// the main graph — via persistent `bind_handle` when supported (CPU), else
    /// kept for feed-each-forward (CUDA). No-op after the first call.
    fn ensure_prepared(&mut self) {
        let Some(st) = self.staging.as_mut() else {
            return;
        };
        if st.prepared {
            return;
        }
        st.prepare.finalize_params();
        let outs = st.prepare.run(&[]);
        // Try persistent binding; it succeeds only when every boundary binds.
        let mut all_bound = !outs.is_empty();
        for (name, out) in st.boundary.iter().zip(outs.iter()) {
            if !self.inner.bind_handle(name, out) {
                all_bound = false;
            }
        }
        st.bound = all_bound;
        st.values = if all_bound { Vec::new() } else { outs };
        st.prepared = true;
    }

    /// Which device this graph runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Set a named parameter (model weight).
    /// Call once per parameter after compilation.
    /// True when this executable's wgpu schedule contains host-executed steps,
    /// which the browser WebGPU path refuses (it cannot block to run them).
    ///
    /// Compile succeeds for such a graph and only `run_async` fails, so without
    /// this a caller learns about it once per step rather than once at setup.
    #[cfg(target_arch = "wasm32")]
    pub fn requires_host_execution(&self) -> bool {
        self.inner.wgpu_requires_host()
    }

    /// One line naming the host-executed steps in this graph's schedule, how
    /// many sync points they cost, and how many are independent — `None` when
    /// it runs entirely on the device.
    pub fn host_execution_summary(&self) -> Option<String> {
        self.inner.wgpu_host_summary()
    }

    /// [`Self::set_param`] that returns a wrong-length upload as an error
    /// instead of panicking.
    pub fn try_set_param(
        &mut self,
        name: &str,
        data: &[f32],
    ) -> Result<(), crate::io_check::IoError> {
        if let Some(io) = self.io.as_ref() {
            let elems = if self.df_param(name) {
                data.len() * 2
            } else {
                data.len()
            };
            io.check_param_len(name, elems)?;
        }
        self.set_param(name, data);
        Ok(())
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        // A double-single param's declared shape carries the (hi, lo) pair
        // axis; the caller uploads one value per f64 element.
        let elems = if self.df_param(name) {
            data.len() * 2
        } else {
            data.len()
        };
        if let Some(io) = self.io.as_mut()
            && let Err(e) = io.note_param(name, Some(elems))
        {
            panic!("{e}");
        }
        if self.df_param(name) {
            let pairs = DfBoundary::f32_pairs(data);
            self.inner.set_param(name, &pairs);
            return;
        }
        if self.widened.contains(name) {
            let wide: Vec<u8> = data
                .iter()
                .flat_map(|&x| (x as f64).to_le_bytes())
                .collect();
            self.set_param_typed(name, &wide, rlx_ir::DType::F64);
            return;
        }
        let rounded = self
            .upload_rounding
            .as_ref()
            .and_then(|r| r.apply(name, data));
        let data = rounded.as_deref().unwrap_or(data);
        // A narrower-than-f32 weight needs converting, not reinterpreting:
        // `set_param` writes the bytes as given, and 4 bytes of f32 read as two
        // halves silently doubles the element count.
        if let Some(dt) = self.narrow_boundary(name) {
            let bytes = crate::backend::narrow_f32_to_bytes(data, dt);
            self.set_param_typed(name, &bytes, dt);
            return;
        }
        if let Some(st) = self.staging.as_mut() {
            if st.prepare_params.contains(name) {
                st.prepare.set_param(name, data);
                st.prepared = false; // weight changed → re-run prepare next forward
            }
            if st.main_params.contains(name) {
                self.inner.set_param(name, data);
            }
        } else {
            self.inner.set_param(name, data);
        }
    }

    /// Execute the graph with named inputs.
    /// Returns one `Vec<f32>` per graph output (copies from arena).
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let inputs = match self.upload_param_inputs(inputs) {
            Ok(rest) => rest,
            Err(e) => panic!("{e}"),
        };
        if let Err(e) = self.check_f32_run(&inputs) {
            panic!("{e}");
        }
        let outs = self.run_untapped(&inputs);
        crate::taps::on_run(outs, self.tap_extra)
    }

    /// Upload any *param* passed to `run` with [`Self::set_param`] and return
    /// the real inputs. Backends ignore a param name among the inputs, so a
    /// loop feeding its weights through `run` used to compute with stale ones.
    fn upload_param_inputs<'a>(
        &mut self,
        inputs: &[(&'a str, &'a [f32])],
    ) -> Result<Vec<(&'a str, &'a [f32])>, crate::io_check::IoError> {
        let Some(io) = self.io.as_ref() else {
            return Ok(inputs.to_vec());
        };
        let (params, rest): (Vec<_>, Vec<_>) =
            inputs.iter().partition(|(n, _)| io.is_param_only(n));
        for (n, d) in &params {
            io.check_param_len(n, d.len())?;
        }
        for (n, d) in params {
            self.set_param(n, d);
        }
        Ok(rest)
    }

    fn run_untapped(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        if self.df.is_some() {
            let bytes: Vec<(&str, Vec<u8>)> = inputs
                .iter()
                .map(|(n, v)| (*n, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
                .collect();
            let typed: Vec<(&str, &[u8], rlx_ir::DType)> = bytes
                .iter()
                .map(|(n, b)| (*n, b.as_slice(), rlx_ir::DType::F32))
                .collect();
            return self
                .run_df(&typed)
                .into_iter()
                .map(|(b, dt)| crate::backend::widen_bytes_to_f32(&b, dt))
                .collect();
        }
        // Any input declared narrower than f32 has to be converted before it
        // reaches the backend. `run_typed` is the path that carries a dtype per
        // input; nine backends implement it. Only taken when a narrow boundary
        // is actually present, so the f32 case keeps its direct path.
        if inputs
            .iter()
            .any(|(n, _)| self.narrow_boundary(n).is_some())
        {
            return self.run_narrowed(inputs);
        }
        self.ensure_prepared();
        match self.staging.as_ref() {
            Some(st) if !st.bound => {
                let mut merged: Vec<(&str, &[f32])> =
                    Vec::with_capacity(inputs.len() + st.boundary.len());
                merged.extend_from_slice(inputs);
                for (name, vals) in st.boundary.iter().zip(st.values.iter()) {
                    merged.push((name.as_str(), vals.as_slice()));
                }
                self.inner.run(&merged)
            }
            _ => self.inner.run(inputs),
        }
    }

    /// [`Self::run`] for graphs with an `F16`/`BF16` input boundary.
    ///
    /// Narrows each input to its declared dtype, runs the typed path, and
    /// widens the outputs back to `f32` so the return type is unchanged. The
    /// widening is lossless — `f32` holds every `F16`/`BF16` value exactly.
    fn run_narrowed(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.ensure_prepared();
        let owned: Vec<(&str, Vec<u8>, rlx_ir::DType)> = inputs
            .iter()
            .map(|(name, vals)| {
                let dt = self.narrow_boundary(name).unwrap_or(rlx_ir::DType::F32);
                (*name, crate::backend::narrow_f32_to_bytes(vals, dt), dt)
            })
            .collect();
        let typed: Vec<(&str, &[u8], rlx_ir::DType)> = owned
            .iter()
            .map(|(n, b, d)| (*n, b.as_slice(), *d))
            .collect();
        self.inner
            .run_typed(&typed)
            .into_iter()
            .map(|(bytes, dt)| crate::backend::widen_bytes_to_f32(&bytes, dt))
            .collect()
    }

    /// Async WebGPU execution on wasm (non-blocking GPU→CPU readback).
    /// WebGL and CPU backends fall back to synchronous [`Self::run`].
    ///
    /// Same contract as [`Self::run`]: params passed as inputs are uploaded,
    /// the inputs are checked, staged weights are prepared and fed, and a
    /// [tap](crate::taps)'s outputs are stripped and handed to it. (This used
    /// to forward the inputs straight to the backend, so in the browser a
    /// tapped graph never reported its pre-head features and every exgb
    /// frozen-encoder trial embedded to the same vector.)
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        if !matches!(self.device, Device::WebGpu | Device::Gpu) {
            return Ok(self.run(inputs));
        }
        let inputs = self
            .upload_param_inputs(inputs)
            .map_err(|e| e.to_string())?;
        self.check_f32_run(&inputs).map_err(|e| e.to_string())?;
        let outs = self.run_staged_async(&inputs, None).await?;
        Ok(crate::taps::on_run(outs, self.tap_extra))
    }

    /// [`Self::run_read_outputs`] on wasm WebGPU: reads back only
    /// `read_indices` (all outputs for `None`), and moves bound GPU handles
    /// forward from their fed outputs on the device — a decode loop's state
    /// never leaves the GPU. Other backends run synchronously.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_read_outputs_async(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        match self.device {
            Device::WebGpu | Device::Gpu => self.run_staged_async(inputs, read_indices).await,
            _ => Ok(self.run_read_outputs(inputs, read_indices)),
        }
    }

    /// The async WebGPU run with staged weights fed as [`Self::run_read_outputs`]
    /// feeds them.
    #[cfg(target_arch = "wasm32")]
    async fn run_staged_async(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        if self.df.is_some()
            || inputs
                .iter()
                .any(|(n, _)| self.narrow_boundary(n).is_some())
        {
            return Err(
                "WebGPU wasm runs f32 input boundaries only; this graph declares an \
                        F16/BF16 input or a dataflow schedule — run it on the CPU"
                    .to_string(),
            );
        }
        self.ensure_prepared_async().await?;
        match self.staging.as_ref() {
            Some(st) if !st.bound => {
                let mut merged: Vec<(&str, &[f32])> =
                    Vec::with_capacity(inputs.len() + st.boundary.len());
                merged.extend_from_slice(inputs);
                for (name, vals) in st.boundary.iter().zip(st.values.iter()) {
                    merged.push((name.as_str(), vals.as_slice()));
                }
                self.inner.wgpu_run_async(&merged, read_indices).await
            }
            _ => self.inner.wgpu_run_async(inputs, read_indices).await,
        }
    }

    /// [`Self::ensure_prepared`] with the prepare graph's run awaited.
    #[cfg(target_arch = "wasm32")]
    async fn ensure_prepared_async(&mut self) -> Result<(), String> {
        let Some(st) = self.staging.as_mut() else {
            return Ok(());
        };
        if st.prepared {
            return Ok(());
        }
        st.prepare.finalize_params();
        let outs = Box::pin(st.prepare.run_async(&[])).await?;
        let mut all_bound = !outs.is_empty();
        for (name, out) in st.boundary.iter().zip(outs.iter()) {
            if !self.inner.bind_handle(name, out) {
                all_bound = false;
            }
        }
        st.bound = all_bound;
        st.values = if all_bound { Vec::new() } else { outs };
        st.prepared = true;
        Ok(())
    }

    /// Run and read back only selected outputs (logits-only decode on MLX).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.ensure_prepared();
        match self.staging.as_ref() {
            Some(st) if !st.bound => {
                let mut merged: Vec<(&str, &[f32])> =
                    Vec::with_capacity(inputs.len() + st.boundary.len());
                merged.extend_from_slice(inputs);
                for (name, vals) in st.boundary.iter().zip(st.values.iter()) {
                    merged.push((name.as_str(), vals.as_slice()));
                }
                self.inner.run_read_outputs(&merged, read_indices)
            }
            _ => self.inner.run_read_outputs(inputs, read_indices),
        }
    }

    /// Read one row from a row-major output tensor after a forward pass.
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        self.inner.read_output_row(out_idx, row, row_inner)
    }

    /// Execute and return raw pointers to output data (zero-copy).
    /// Data is valid until the next `run`/`run_raw` call.
    ///
    /// # Safety
    /// The returned pointers point into the arena. Do not use after
    /// the next call to run/run_raw (arena data will be overwritten).
    pub fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
        self.ensure_prepared();
        self.inner.run_raw(inputs)
    }

    /// Fastest execution: inputs by slot index (order matches graph input declaration).
    /// Returns output (offset, len) pairs. Read data via `arena_ptr().add(offset)`.
    /// Zero HashMap lookup, zero Vec allocation, zero name matching.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.ensure_prepared();
        self.inner.run_slots(inputs)
    }

    /// Arena pointer for reading output data after `run_slots`.
    pub fn arena_ptr(&self) -> *const u8 {
        self.inner.arena_ptr()
    }

    /// Bind a persistent buffer (KV-cache, optimizer state, etc.).
    /// Stays alive across `run()` calls; the backend uses it as the
    /// graph input with the matching name.
    /// Returns true if the backend supports persistent handles.
    pub fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
        let ok = self.inner.bind_handle(name, data);
        if ok && let Some(io) = self.io.as_mut() {
            io.mark_input_bound(name);
        }
        ok
    }

    /// Read the current contents of a persistent buffer.
    pub fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_handle(name)
    }

    /// What this executable can actually do.
    ///
    /// The ground truth, as opposed to [`crate::advisory_capabilities`], which
    /// answers the same question from a device enum *before* anything is
    /// compiled — useful for planning, but hand-maintained and therefore able to
    /// drift. `rlx-runtime/tests/capability_table_matches_backends.rs` pins the
    /// two together.
    pub fn capabilities(&self) -> crate::ExecutableCapabilities {
        self.inner.capabilities()
    }

    /// GPU-resident MLX input (no-op on non-MLX backends).
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        let ok = self.inner.bind_gpu_handle(name, data);
        if ok && let Some(io) = self.io.as_mut() {
            io.mark_input_bound(name);
        }
        ok
    }

    /// ZERO-COPY in-place optimizer step over GPU-resident weights: the
    /// optimizer reads gradients and writes params directly on the unified-memory
    /// arena (no GPU→host→GPU roundtrip). `trainable[i] = (input_name, shape)`;
    /// the gradient is read from output slot `1 + i` (backward outputs
    /// `[loss, grad0, …]`). Pair with [`Self::bind_gpu_handle`] (weight resident)
    /// and `run_read_outputs(_, Some(&[0]))` (read only the loss). Returns false
    /// on backends without GPU-resident support — fall back to a host step.
    ///
    /// ```ignore
    /// compiled.bind_gpu_handle("W", &w_init);
    /// let mut muon = Muon::new(lr);
    /// for _ in 0..steps {
    ///     compiled.run_read_outputs(&[("d_output", &[1.0])], Some(&[0]));
    ///     compiled.optimizer_step_resident(&trainable, &mut |n, s, p, g| muon.step(n, s, p, g));
    /// }
    /// ```
    pub fn optimizer_step_resident(
        &mut self,
        trainable: &[(String, Vec<usize>)],
        step: &mut dyn FnMut(&str, &[usize], &mut [f32], &[f32]),
    ) -> bool {
        self.inner.optimizer_step_resident(trainable, step)
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.inner.has_gpu_handle(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.set_gpu_handle_feed(handle_name, output_index)
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle(name)
    }

    /// Copy the named inputs' device contents aside (wgpu) so
    /// [`Self::restore_inputs`] can put them back. False when unsupported.
    pub fn save_inputs(&mut self, names: &[&str]) -> bool {
        self.inner.save_inputs(names)
    }

    /// Put back what [`Self::save_inputs`] copied aside.
    pub fn restore_inputs(&mut self) -> bool {
        self.inner.restore_inputs()
    }

    /// Read one row from a resident GPU input handle without full-tensor D2H.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle_row(name, row, row_inner)
    }

    /// Register a targeted row feed for resident KV decode (graphs that emit the
    /// new token at the last bucket-padded output row). No-op (false) on
    /// backends without GPU-resident handle support. See [`Self::feed_kv_row`].
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.register_kv_row_feed(handle_name, output_index)
    }

    /// Fold each registered row feed's new-token row (`src_row` of its output)
    /// into the resident handle slot at `dst_row` (`row_elems` = kv_dim),
    /// in-place on device. Returns false when unsupported.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) -> bool {
        self.inner.feed_kv_row(src_row, dst_row, row_elems)
    }

    /// Batch-major resident KV feed (`past[b, dst_row]` ← `new[b, 0]`).
    pub fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner
            .feed_kv_batch_major(dst_row, batch, seq_cap, row_elems)
    }

    /// Strided-source batch-major resident KV feed: fold the new token from a
    /// `[B, src_seq_cap, row_elems]` output at `src_row` into the resident input
    /// handle `[B, dst_seq_cap, row_elems]` at `dst_row`. For decode graphs that
    /// emit the whole appended cache (new token at output row `upper`).
    #[allow(clippy::too_many_arguments)]
    pub fn feed_kv_batch_major_src(
        &mut self,
        src_row: usize,
        src_seq_cap: usize,
        dst_row: usize,
        batch: usize,
        dst_seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner.feed_kv_batch_major_src(
            src_row,
            src_seq_cap,
            dst_row,
            batch,
            dst_seq_cap,
            row_elems,
        )
    }

    /// Ragged resident KV feed: each batch element folds the new token from a
    /// `[B, src_seq_cap, row_elems]` output at `src_row` into its own resident
    /// handle row `dst_rows[b]` (the sequence's `past_len`). For fused decode of
    /// sequences at MIXED cache lengths.
    pub fn feed_kv_batch_major_ragged(
        &mut self,
        src_row: usize,
        src_seq_cap: usize,
        dst_rows: &[usize],
        dst_seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner.feed_kv_batch_major_ragged(
            src_row,
            src_seq_cap,
            dst_rows,
            dst_seq_cap,
            row_elems,
        )
    }

    /// Mark a graph input as device-resident without host staging.
    pub fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        self.inner.prepare_resident_gpu_handle(name)
    }

    /// Upload bound (non-resident) GPU handle mirrors into the arena.
    pub fn stage_bound_gpu_handles_to_arena(&mut self) -> bool {
        self.inner.stage_bound_gpu_handles_to_arena();
        true
    }

    /// D2D copy resident KV rows `[from_row..to_row)` from another compiled graph.
    pub fn seed_resident_kv_prefix_from(
        &mut self,
        src: &CompiledGraph,
        prefix_tokens: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.seed_resident_kv_prefix_from(
            src.inner.as_ref(),
            prefix_tokens,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }

    /// D2D copy resident KV rows `[from_row..to_row)` from another compiled graph.
    pub fn copy_resident_kv_rows_from(
        &mut self,
        src: &CompiledGraph,
        from_row: usize,
        to_row: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if self.device != src.device {
            return false;
        }
        self.inner.copy_resident_kv_rows_from(
            src.inner.as_ref(),
            from_row,
            to_row,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }

    /// Copy parameter buffers from `src` when layouts match (same device/backend).
    pub fn copy_params_from(&mut self, src: &CompiledGraph) -> bool {
        if self.device != src.device {
            return false;
        }
        let ok = self.inner.copy_params_from(src.inner.as_ref());
        if ok && let Some(io) = self.io.as_mut() {
            io.mark_all_params();
        }
        ok
    }

    /// Share `src`'s packed weight buffer (retain, not copy) instead of uploading
    /// our own, when the weight layout matches EXACTLY — one GPU copy backs both.
    /// Returns false (caller must upload) on device mismatch or differing layout.
    pub fn share_params_from(&mut self, src: &CompiledGraph) -> bool {
        if self.device != src.device {
            return false;
        }
        let ok = self.inner.share_params_from(src.inner.as_ref());
        if ok && let Some(io) = self.io.as_mut() {
            io.mark_all_params();
        }
        ok
    }

    /// Whether param `name` already holds its values (a weight shared with
    /// an executable that bound it): a binder may skip it.
    pub fn param_already_set(&self, name: &str) -> bool {
        self.inner.param_already_set(name)
    }

    /// Run, refresh GPU handle from output, return that output vector.
    pub fn run_feed_gpu_handle(
        &mut self,
        inputs: &[(&str, &[f32])],
        handle_name: &str,
        output_index: usize,
    ) -> Option<Vec<f32>> {
        self.inner
            .run_feed_gpu_handle(inputs, handle_name, output_index)
    }

    /// Hint subsequent `run` calls to process only the first `actual`
    /// rows along the bucket axis (out of `upper`, the compile extent).
    /// Backends that support per-kernel active-extent dispatch honor
    /// this; others ignore it. Pass `None` to clear.
    ///
    /// See `BucketedCompileCache::run_padded` for the canonical caller.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        #[cfg(feature = "cpu")]
        if let Some((actual, _)) = extent {
            crate::onnx_active::set_active_token_count(Some(actual))
        }
        self.inner.set_active_extent(extent);
    }

    /// Return this graph's idle scratch (activation) pages to the OS while it stays
    /// cached — for multi-subgraph pipelines whose stages run sequentially. Weights
    /// stay resident; the next run refaults + re-zeros scratch transparently. Call
    /// only between stages, never inside a hot loop. Advisory no-op off CPU.
    pub fn release_scratch(&mut self) {
        self.inner.release_scratch();
    }

    /// TIDE merged MoE placement (`mask[expert]` device-resident if any layer has it).
    pub fn set_moe_resident_experts(&mut self, mask: &[bool]) {
        self.inner.set_moe_resident_experts(mask);
    }

    /// Per MoE layer placement (forward order). Preferred on CPU over merged mask.
    pub fn set_moe_resident_experts_per_layer(&mut self, masks: &[&[bool]]) {
        self.inner.set_moe_resident_experts_per_layer(masks);
    }

    /// Capture MoE router TopK on next forward (CPU). Returns false if unsupported.
    pub fn enable_moe_topk_capture(&mut self, num_experts: usize) -> bool {
        self.inner.enable_moe_topk_capture(num_experts)
    }

    /// Per-layer expert indices from the last forward (MoE router TopK order).
    pub fn take_moe_topk_capture(&mut self) -> Option<Vec<Vec<u32>>> {
        self.inner.take_moe_topk_capture()
    }

    /// GroupedMatMul GPU/CPU token accounting from the last forward (CPU).
    pub fn take_moe_residency_stats(&mut self) -> Option<crate::MoeResidencyStats> {
        self.inner.take_moe_residency_stats()
    }

    // ── Pipelined / async execution (Phase C) ─────────────────────────

    /// Encode + commit a forward pass without waiting for the device.
    ///
    /// Outputs of intermediate calls are stomped — use `run_pipelined`
    /// when you need each call's outputs back. Pair with `sync_pending`
    /// to drain. CPU is synchronous, so this falls back to `run`.
    pub fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        self.inner.commit_no_wait(inputs);
    }

    /// Wait for every command queued by `commit_no_wait`. CPU is a no-op.
    pub fn sync_pending(&mut self) {
        self.inner.sync_pending();
    }

    /// Pipelined batch run. Issues one commit per input set, syncs once
    /// at the end. On Metal, each commit gets its own output snapshot
    /// (allocated + blit-copied), so subsequent commits stomping the
    /// shared arena don't corrupt earlier runs' outputs.
    /// Returns `out[run_idx][output_idx][element_idx]`.
    pub fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        self.inner.run_pipelined(input_sets)
    }

    /// Set a named parameter from raw bytes in the given dtype. The
    /// backend handles the widen-to-f32 (or zero-widen, when supported
    /// natively) on the way in. Lets callers feed F16/BF16 weights
    /// without a host-side cast.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if let Some(io) = self.io.as_mut() {
            let _ = io.note_param(name, None);
        }
        if self.df_param(name) {
            let pairs = match dtype {
                rlx_ir::DType::F64 => DfBoundary::f64_bytes_pairs(data),
                _ => {
                    let wide = crate::backend::widen_bytes_to_f32(data, dtype);
                    DfBoundary::f32_pairs(&wide)
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect()
                }
            };
            self.inner.set_param_typed(name, &pairs, rlx_ir::DType::F32);
            return;
        }
        if self.widened.contains(name) && dtype != rlx_ir::DType::F64 {
            let wide: Vec<u8> = crate::backend::widen_bytes_to_f32(data, dtype)
                .iter()
                .flat_map(|&x| (x as f64).to_le_bytes())
                .collect();
            self.set_param_typed(name, &wide, rlx_ir::DType::F64);
            return;
        }
        // Float weights are rounded through f32 and stored back in `dtype`;
        // packed / integer weights are already in their own format.
        let rounded = match &self.upload_rounding {
            Some(r) if dtype.is_float() => {
                let wide = crate::backend::widen_bytes_to_f32(data, dtype);
                r.apply(name, &wide)
                    .map(|v| crate::backend::narrow_f32_to_bytes(&v, dtype))
            }
            _ => None,
        };
        let data = rounded.as_deref().unwrap_or(data);
        if let Some(st) = self.staging.as_mut() {
            if st.prepare_params.contains(name) {
                st.prepare.set_param_typed(name, data, dtype);
                st.prepared = false;
            }
            if st.main_params.contains(name) {
                self.inner.set_param_typed(name, data, dtype);
            }
        } else {
            self.inner.set_param_typed(name, data, dtype);
        }
    }

    /// Incrementally write raw `data` into a named param starting `byte_offset`
    /// bytes into its storage. Returns true if the backend did the partial write;
    /// false → caller must fall back to a whole-buffer [`Self::set_param_typed`].
    /// Used by paged-MoE residency to upload only the changed expert slot.
    pub fn set_param_range(&mut self, name: &str, byte_offset: usize, data: &[u8]) -> bool {
        if let Some(io) = self.io.as_mut() {
            let _ = io.note_param(name, None);
        }
        // A partial write of a rounded weight cannot be block-rounded on its
        // own; send the caller to the whole-buffer path, which rounds (and
        // widens a param promoted to f64, whose bytes the caller sizes as f32).
        if self.widened.contains(name)
            || self
                .upload_rounding
                .as_ref()
                .is_some_and(|r| r.row_len.contains_key(name))
        {
            return false;
        }
        if let Some(st) = self.staging.as_mut() {
            let mut done = false;
            if st.prepare_params.contains(name) {
                done |= st.prepare.set_param_range(name, byte_offset, data);
                st.prepared = false;
            }
            if st.main_params.contains(name) {
                done |= self.inner.set_param_range(name, byte_offset, data);
            }
            done
        } else {
            self.inner.set_param_range(name, byte_offset, data)
        }
    }

    /// Finish param upload — warms backend caches when supported.
    pub fn finalize_params(&mut self) {
        self.ensure_prepared();
        self.inner.finalize_params();
    }

    /// Execute with typed inputs and return outputs in their declared
    /// graph dtype, byte-encoded. Mirrors the wgpu / MLX zero-widen
    /// semantics on f32-arena backends (CPU + Metal) by widening at
    /// the boundary.
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let elems: Vec<(&str, Option<usize>)> = inputs
            .iter()
            .map(|(n, b, dt)| {
                let e = self
                    .io
                    .as_ref()
                    .and_then(|io| io.typed_elems(n, b.len(), *dt));
                (*n, e)
            })
            .collect();
        if let Err(e) = self.check_run(elems.iter().copied()) {
            panic!("{e}");
        }
        let outs = self.run_typed_untapped(inputs);
        crate::taps::on_run_typed(outs, self.tap_extra)
    }

    fn run_typed_untapped(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        if self.df.is_some() {
            return self.run_df(inputs);
        }
        self.ensure_prepared();
        self.inner.run_typed(inputs)
    }

    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.inner.set_rng(rng);
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        self.inner.rng()
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn end_to_end_session() {
        let mut g = Graph::new("matmul_bias_gelu");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let add = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(op::Activation::Gelu, add, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);

        // Compile
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile(g);

        // Set weights
        // w = identity-ish [4, 3]: first 3 rows are I, last row is 0
        compiled.set_param(
            "w",
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        compiled.set_param("b", &[0.5, -0.5, 0.0]);

        // Run
        let x_data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: [1,0,0,0] @ w = [1,0,0] + bias = [1.5,-0.5,0]
            0.0, 1.0, 0.0, 0.0, // row 1: [0,1,0,0] @ w = [0,1,0] + bias = [0.5, 0.5,0]
        ];
        let outputs = compiled.run(&[("x", &x_data)]);

        assert_eq!(outputs.len(), 1);
        let result = &outputs[0];
        assert_eq!(result.len(), 6); // [2, 3]

        // gelu(1.5) ≈ 1.399, gelu(-0.5) ≈ -0.154, gelu(0) = 0
        assert!(
            (result[0] - 1.399).abs() < 0.01,
            "gelu(1.5) = {}",
            result[0]
        );
        assert!(
            (result[1] - -0.154).abs() < 0.01,
            "gelu(-0.5) = {}",
            result[1]
        );
        assert!((result[2]).abs() < 0.01, "gelu(0) = {}", result[2]);

        // gelu(0.5) ≈ 0.346, gelu(0.5) ≈ 0.346, gelu(0) = 0
        assert!(
            (result[3] - 0.346).abs() < 0.01,
            "gelu(0.5) = {}",
            result[3]
        );
        assert!(
            (result[4] - 0.346).abs() < 0.01,
            "gelu(0.5) = {}",
            result[4]
        );

        // Run again with different input — zero allocation
        let x2 = vec![0.0f32; 8];
        let outputs2 = compiled.run(&[("x", &x2)]);
        // All zeros input → gelu(bias) for each output
        let r2 = &outputs2[0];
        assert!((r2[0] - 0.346).abs() < 0.01, "gelu(0.5) = {}", r2[0]); // gelu(0+0.5)
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn device_display() {
        use crate::device_ext::is_available;
        assert!(format!("{}", Device::Cpu).starts_with("CPU"));
        assert!(is_available(Device::Cpu));
        // Backend availability is feature-gated; only assert
        // unavailable when the corresponding feature is off.
        #[cfg(not(feature = "gpu"))]
        assert!(!is_available(Device::Gpu));
        #[cfg(not(feature = "cuda"))]
        assert!(!is_available(Device::Cuda));
        #[cfg(not(feature = "rocm"))]
        assert!(!is_available(Device::Rocm));
        #[cfg(not(feature = "tpu"))]
        assert!(!is_available(Device::Tpu));
    }
}
