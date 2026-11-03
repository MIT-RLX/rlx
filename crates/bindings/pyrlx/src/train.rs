// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training from Python (feature `training`).
//!
//! Two levels. `pyrlx.Optimizer` is `rlx-optim` — Adam, AdamW, Muon, SOAP, Lion
//! and the rest — driving arrays the caller owns. `pyrlx.Trainer` is the whole
//! loop: it differentiates the forward graph once, compiles it once, keeps the
//! parameters, and returns a scalar loss per step.
//!
//! ```python
//! t = rlx.Trainer(loss_graph, wrt=["w1", "b1"], init=weights, device="metal",
//!                 optimizer={"kind": "adamw", "lr": 3e-4})
//! for batch in data:
//!     print(t.step(batch))
//! trained = t.params()
//! ```
//!
//! The logic — host and resident update paths, clipping, freezing, GGUF
//! checkpoints (the same layout rlx-js writes) — is
//! `rlx_bind::runtime::train`; this file converts Python values and runs
//! `run`'s callback loop.

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlx_bind::ApiError;
use rlx_bind::runtime::train::{
    EvalTracker, OptItem, Optimizer, OptimizerState, Resident, RunStats, StepInfo, StepItem,
    Trainer, TrainerConfig, TrainingOptions,
};

use crate::bind::{from_json, opt_json, to_json, to_py};
use crate::graph::PyGraph;

/// A float32 array's values (copied); non-contiguous arrays are gathered.
pub(crate) fn f32_values(name: &str, v: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let arr = v.downcast::<PyArrayDyn<f32>>().map_err(|_| {
        PyValueError::new_err(format!(
            "'{name}': expected a numpy.ndarray of dtype float32"
        ))
    })?;
    if arr.is_c_contiguous() {
        // Safety: contiguity checked; the GIL is held for the copy.
        Ok(unsafe { arr.as_slice()? }.to_vec())
    } else {
        Ok(arr.to_vec()?)
    }
}

fn f32_dict(params: &Bound<'_, PyDict>, what: &str) -> PyResult<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::with_capacity(params.len());
    for (k, v) in params.iter() {
        let name: String = k.extract()?;
        let data = f32_values(&format!("{what}('{name}')"), &v)?;
        out.push((name, data));
    }
    Ok(out)
}

// ── standalone Optimizer ───────────────────────────────────────

/// One of `rlx-optim`'s algorithms over arrays the caller owns.
#[pyclass(name = "Optimizer", module = "pyrlx._pyrlx", unsendable)]
pub(crate) struct PyOptimizer {
    inner: Optimizer,
}

#[pymethods]
impl PyOptimizer {
    /// `Optimizer("adamw", lr=1e-3, weight_decay=0.01, …)`.
    #[new]
    #[pyo3(signature = (kind = "adamw", **opts))]
    fn new(kind: &str, opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let opts = match opts {
            Some(d) => to_json(d.as_any())?,
            None => rlx_bind::runtime::Value::Null,
        };
        Ok(Self {
            inner: Optimizer::new(Some(kind), &opts).map_err(to_py)?,
        })
    }

    /// `step({name: (param, grad)})` → `{name: updated_param}`.
    ///
    /// The inputs are copied; the updated values come back as new float32
    /// arrays in the parameter's shape. The optimizer keys its state by name,
    /// so pass the same names every step.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        params: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut items = Vec::with_capacity(params.len());
        for (k, v) in params.iter() {
            let name: String = k.extract()?;
            let (p, g): (Bound<'py, PyAny>, Bound<'py, PyAny>) = v.extract().map_err(|_| {
                PyValueError::new_err(format!("Optimizer.step('{name}'): expected (param, grad)"))
            })?;
            let shape: Vec<usize> = p
                .downcast::<PyArrayDyn<f32>>()
                .map(|a| a.shape().to_vec())
                .unwrap_or_default();
            let param = f32_values(&name, &p)?;
            let grad = f32_values(&name, &g)?;
            let shape = if shape.is_empty() {
                vec![param.len()]
            } else {
                shape
            };
            items.push(StepItem {
                name,
                shape,
                param,
                grad,
            });
        }
        self.inner.step(&mut items).map_err(to_py)?;
        let out = PyDict::new_bound(py);
        for it in items {
            out.set_item(it.name, it.param.into_pyarray_bound(py).reshape(it.shape)?)?;
        }
        Ok(out)
    }

    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    /// `state_dict()` → `{"step": int, "buffers": {key: float32 array}}`, or
    /// `None` when the algorithm cannot save its state.
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .inner
            .state_dict()
            .map(|s| state_to_py(py, &s))
            .transpose()
    }

    /// `load_state_dict(state)` → whether this algorithm took it.
    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<bool> {
        let s = state_from_py(state)?;
        Ok(self.inner.inner.load_state_dict(&s))
    }

    fn __repr__(&self) -> String {
        format!("<pyrlx.Optimizer {}>", self.inner.kind)
    }
}

// ── Trainer ────────────────────────────────────────────────────

/// Differentiate-once / compile-once training loop over a scalar-loss graph.
#[pyclass(name = "Trainer", module = "pyrlx._pyrlx", unsendable)]
pub(crate) struct PyTrainer {
    pub(crate) inner: Trainer,
}

/// `{input: float32 array}` → owned `(name, values)` for one step.
pub(crate) fn batch(b: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::with_capacity(b.len());
    for (k, v) in b.iter() {
        let name: String = k.extract()?;
        let data = f32_values(&name, &v)?;
        out.push((name, data));
    }
    Ok(out)
}

pub(crate) fn pairs(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect()
}

/// An optimizer state as `{"step": int, "buffers": {key: float32 array}}`.
pub(crate) fn state_to_py<'py>(
    py: Python<'py>,
    s: &OptimizerState,
) -> PyResult<Bound<'py, PyDict>> {
    let buffers = PyDict::new_bound(py);
    for (k, v) in &s.buffers {
        buffers.set_item(k, v.clone().into_pyarray_bound(py))?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("step", s.step)?;
    out.set_item("buffers", buffers)?;
    Ok(out)
}

/// [`state_to_py`]'s form back.
pub(crate) fn state_from_py(v: &Bound<'_, PyAny>) -> PyResult<OptimizerState> {
    let d = v.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "an optimizer state is {\"step\": int, \"buffers\": {key: float32 array}}",
        )
    })?;
    let step: u64 = match d.get_item("step")? {
        Some(s) => s.extract()?,
        None => 0,
    };
    let mut buffers = Vec::new();
    if let Some(b) = d.get_item("buffers")? {
        let b = b
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("state['buffers'] must be a dict"))?;
        for (k, a) in b.iter() {
            let key: String = k.extract()?;
            let values = f32_values(&format!("state['buffers']['{key}']"), &a)?;
            buffers.push((key, values));
        }
    }
    Ok(OptimizerState { step, buffers })
}

/// `update(params, grads, step)` over copies of `items`, then what `params`
/// holds is written back — every entry checked before any is written, so a
/// bad one leaves all the params as they were.
fn call_update(
    py: Python<'_>,
    update: &Bound<'_, PyAny>,
    items: &mut [OptItem<'_>],
    step: &StepInfo<'_>,
) -> PyResult<()> {
    exchange(py, items, "Trainer.step_with", |params, grads| {
        update.call1((params.clone(), grads, step.update)).map(drop)
    })
}

/// Copies of `items` as `params` / `grads` dicts of float32 arrays for `f`,
/// then what `params` holds is written back — every entry checked before any
/// is written. `what` names the caller in errors.
pub(crate) fn exchange<'py>(
    py: Python<'py>,
    items: &mut [OptItem<'_>],
    what: &str,
    f: impl FnOnce(&Bound<'py, PyDict>, Bound<'py, PyDict>) -> PyResult<()>,
) -> PyResult<()> {
    let params = PyDict::new_bound(py);
    let grads = PyDict::new_bound(py);
    for it in items.iter() {
        let shape = it.shape.to_vec();
        params.set_item(
            it.name,
            it.param
                .to_vec()
                .into_pyarray_bound(py)
                .reshape(shape.clone())?,
        )?;
        grads.set_item(
            it.name,
            it.grad.to_vec().into_pyarray_bound(py).reshape(shape)?,
        )?;
    }
    f(&params, grads)?;
    let mut updated = Vec::with_capacity(items.len());
    for it in items.iter() {
        let name = it.name;
        let value = params.get_item(name)?.ok_or_else(|| {
            PyValueError::new_err(format!("{what}: update removed '{name}' from params"))
        })?;
        let what = format!("{what}('{name}')");
        let shape = value
            .downcast::<PyArrayDyn<f32>>()
            .map(|a| a.shape().to_vec())
            .map_err(|_| {
                PyValueError::new_err(format!("{what}: expected a numpy.ndarray of dtype float32"))
            })?;
        if shape != it.shape {
            return Err(PyValueError::new_err(format!(
                "{what}: update left shape {shape:?}; the param is {:?}",
                it.shape
            )));
        }
        updated.push(f32_values(&what, &value)?);
    }
    for (it, values) in items.iter_mut().zip(updated) {
        it.param.copy_from_slice(&values);
    }
    Ok(())
}

#[pymethods]
impl PyTrainer {
    /// `Trainer(graph, wrt, init, device="cpu", optimizer=None, policy=None,
    /// clip_norm=None, resident=None)`.
    ///
    /// `init` maps every param to a float32 array; params in `init` but not in
    /// `wrt` are uploaded once and never updated (how LoRA freezes a base).
    /// `optimizer` is a kind name or `{"kind": …, "lr": …, …}`. `resident=True`
    /// fuses the AdamW update into the graph — Metal and wgpu only;
    /// `resident="unverified"` opts in on other GPU backends. The graph is
    /// consumed.
    #[new]
    #[pyo3(signature = (graph, wrt, init, device = "cpu", optimizer = None, policy = None,
                        clip_norm = None, resident = None, schedule = None, accumulate = 1,
                        loss_scale = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: &Bound<'_, PyGraph>,
        wrt: Vec<String>,
        init: &Bound<'_, PyDict>,
        device: &str,
        optimizer: Option<&Bound<'_, PyAny>>,
        policy: Option<&Bound<'_, PyAny>>,
        clip_norm: Option<f32>,
        resident: Option<&Bound<'_, PyAny>>,
        schedule: Option<&Bound<'_, PyAny>>,
        accumulate: usize,
        loss_scale: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut training = rlx_bind::runtime::serde_json::json!({ "accumulate": accumulate });
        if let Some(s) = schedule {
            training["schedule"] = to_json(s)?;
        }
        if let Some(l) = loss_scale {
            training["loss_scale"] = to_json(l)?;
        }
        let training = TrainingOptions::from_json(&training).map_err(to_py)?;
        let policy = crate::policy::parse_policy(policy)?;
        // A bare kind name means `{"kind": name}`.
        let optimizer = match optimizer.map(|o| o.extract::<String>()) {
            Some(Ok(kind)) => rlx_bind::runtime::serde_json::json!({ "kind": kind }),
            _ => opt_json(optimizer)?,
        };
        let resident = Resident::from_json(&opt_json(resident)?).map_err(to_py)?;
        let init = f32_dict(init, "init")?;
        let forward =
            graph.borrow_mut().inner.take().ok_or_else(|| {
                PyRuntimeError::new_err("Trainer: graph has already been consumed")
            })?;
        let cfg = TrainerConfig {
            wrt,
            init,
            device: device.to_string(),
            optimizer,
            policy,
            clip_norm,
            resident,
            training,
        };
        Ok(Self {
            inner: Trainer::new(forward, cfg).map_err(to_py)?,
        })
    }

    /// One optimizer update on `batch` (`{input_name: float32 array}`) → the loss.
    fn step(&mut self, batch: &Bound<'_, PyDict>) -> PyResult<f32> {
        let owned = self::batch(batch)?;
        self.inner.step(&pairs(&owned)).map_err(to_py)
    }

    /// `step_with(batch, update)` → the loss, with the update made by `update`.
    ///
    /// Forward, backward and clipping run as in `step`; then
    /// `update(params, grads, step)` gets `params` and `grads`, dicts of
    /// float32 arrays keyed by name (every unfrozen trainable param, in
    /// `trainable()` order), and the index of this step (the first is 1). It
    /// changes `params` in place, or replaces entries with float32 arrays of
    /// the same shape; what `params` holds when it returns becomes the
    /// parameters. Nothing is written back when it raises or leaves an entry
    /// that does not fit, and the step is not counted. The trainer's optimizer
    /// does not run. Not on a resident trainer.
    fn step_with(
        &mut self,
        py: Python<'_>,
        batch: &Bound<'_, PyDict>,
        update: &Bound<'_, PyAny>,
    ) -> PyResult<f32> {
        let owned = self::batch(batch)?;
        // The Python exception itself, so the caller sees its own type and
        // traceback rather than a RuntimeError wrapping its message.
        let mut raised: Option<PyErr> = None;
        let result = self.inner.step_with(&pairs(&owned), |items, step| {
            call_update(py, update, items, step).map_err(|e| {
                raised = Some(e);
                ApiError::runtime("Trainer.step_with: update raised")
            })
        });
        match raised {
            Some(e) => Err(e),
            None => result.map_err(to_py),
        }
    }

    /// Loss only, no update — for a validation split.
    fn evaluate(&mut self, batch: &Bound<'_, PyDict>) -> PyResult<f32> {
        let owned = self::batch(batch)?;
        self.inner.evaluate(&pairs(&owned)).map_err(to_py)
    }

    /// `run(steps, batch, lr=None, on_step=None, every=1, eval_batch=None,
    /// eval_every=0, patience=None, min_delta=0.0, checkpoint=None,
    /// checkpoint_every=0)` → a summary dict.
    ///
    /// `batch(i)` supplies step `i`'s inputs (return `None` when data runs
    /// out); `lr(i)` schedules the learning rate; `on_step(i, loss)` returning
    /// `False` stops early. With `eval_every`, `eval_batch` (a batch, or a
    /// function of the step returning one) is evaluated every `eval_every`
    /// steps: the summary lists the evaluations and the best; with `patience`
    /// the run stops after that many evaluations without a new best (by more
    /// than `min_delta`). With `checkpoint` (a folder), the best evaluation's
    /// state is saved as `best.gguf` and, with `checkpoint_every`, every so
    /// many steps as `step-<n>.gguf`. The trainer keeps all state, so calling
    /// `run` again resumes exactly where it left off.
    #[pyo3(signature = (steps, batch, lr = None, on_step = None, every = 1, eval_batch = None,
                        eval_every = 0, patience = None, min_delta = 0.0, checkpoint = None,
                        checkpoint_every = 0))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        py: Python<'_>,
        steps: usize,
        batch: &Bound<'_, PyAny>,
        lr: Option<&Bound<'_, PyAny>>,
        on_step: Option<&Bound<'_, PyAny>>,
        every: usize,
        eval_batch: Option<&Bound<'_, PyAny>>,
        eval_every: u64,
        patience: Option<usize>,
        min_delta: f32,
        checkpoint: Option<std::path::PathBuf>,
        checkpoint_every: u64,
    ) -> PyResult<PyObject> {
        if eval_every > 0 && eval_batch.is_none() {
            return Err(PyValueError::new_err("run: eval_every needs eval_batch"));
        }
        if let Some(dir) = &checkpoint {
            std::fs::create_dir_all(dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        let start = self.inner.steps();
        let mut stats = RunStats::new();
        let mut evals = EvalTracker::new(patience, min_delta);
        for i in 0..steps {
            let index = start + i as u64;
            if let Some(f) = lr {
                let v = f.call1((index,))?;
                if !v.is_none() {
                    self.inner.set_lr(v.extract()?);
                }
            }
            let b = batch.call1((index,))?;
            if b.is_none() {
                stats.stopped = true;
                break;
            }
            let owned = self::batch(b.downcast::<PyDict>()?)?;
            let loss = self.inner.step(&pairs(&owned)).map_err(to_py)?;
            stats.record(loss);
            let done = self.inner.steps();
            if let (Some(eb), true) = (
                eval_batch,
                eval_every > 0 && done.is_multiple_of(eval_every),
            ) {
                let b = if eb.is_callable() {
                    eb.call1((done,))?
                } else {
                    eb.clone()
                };
                let owned = self::batch(b.downcast::<PyDict>()?)?;
                let eval_loss = self.inner.evaluate(&pairs(&owned)).map_err(to_py)?;
                let (best, stop) = evals.record(done, eval_loss);
                if best && let Some(dir) = &checkpoint {
                    self.inner
                        .save(&dir.join("best.gguf").to_string_lossy())
                        .map_err(to_py)?;
                }
                if stop {
                    stats.stopped = true;
                    break;
                }
            }
            if let Some(dir) = &checkpoint
                && checkpoint_every > 0
                && done.is_multiple_of(checkpoint_every)
            {
                let path = dir.join(format!("step-{done}.gguf"));
                self.inner.save(&path.to_string_lossy()).map_err(to_py)?;
            }
            if let Some(cb) = on_step
                && stats.report_due(every, i, steps)
            {
                // Only an explicit `False` stops; a logging callback that
                // returns nothing must not end the run.
                let keep = cb.call1((index, loss))?;
                if !keep.is_none() && keep.extract::<bool>().ok() == Some(false) {
                    stats.stopped = true;
                    break;
                }
            }
        }
        let mut summary = stats.summary(self.inner.grad_norm());
        if eval_every > 0
            && let (Some(m), rlx_bind::runtime::Value::Object(e)) =
                (summary.as_object_mut(), evals.summary())
        {
            m.extend(e);
        }
        if self.inner.loss_scale().is_some() {
            summary["loss_scale"] = rlx_bind::runtime::serde_json::json!(self.inner.loss_scale());
            summary["skipped_updates"] =
                rlx_bind::runtime::serde_json::json!(self.inner.skipped_updates());
        }
        Ok(from_json(py, &summary))
    }

    /// `set_data_parallel(group)`: train data-parallel over a `pyrlx.Group`,
    /// as PyTorch's DDP does — rank 0's params are broadcast now, and every
    /// update uses the mean gradient over the ranks. Every rank calls this and
    /// then steps the same number of times.
    fn set_data_parallel(&mut self, group: &Bound<'_, crate::dist::PyGroup>) -> PyResult<()> {
        let pg = (*group.borrow().inner.inner).clone();
        self.inner.set_data_parallel(pg).map_err(to_py)
    }

    /// Whether every rank holds the same params (always true without a group).
    fn replicas_agree(&self) -> PyResult<bool> {
        self.inner.replicas_agree().map_err(to_py)
    }

    /// The learning rate the last update used (the schedule's, under one).
    fn lr(&self) -> f32 {
        self.inner.lr()
    }

    /// The loss scale now, under `loss_scale`; else `None`.
    fn loss_scale(&self) -> Option<f32> {
        self.inner.loss_scale()
    }

    /// Updates skipped for non-finite gradients under `loss_scale`.
    fn skipped_updates(&self) -> u64 {
        self.inner.skipped_updates()
    }

    /// Stop updating `names` (all trainable params when omitted) → frozen
    /// count. Their gradients are still computed; leave a param out of `wrt`
    /// to skip it.
    #[pyo3(signature = (names = None))]
    fn freeze(&mut self, names: Option<Vec<String>>) -> PyResult<usize> {
        self.inner.set_frozen(names.as_deref(), true).map_err(to_py)
    }

    #[pyo3(signature = (names = None))]
    fn unfreeze(&mut self, names: Option<Vec<String>>) -> PyResult<usize> {
        self.inner
            .set_frozen(names.as_deref(), false)
            .map_err(to_py)
    }

    fn frozen(&self) -> Vec<String> {
        self.inner.frozen()
    }

    /// Every parameter that receives a gradient, frozen or not.
    fn trainable(&self) -> Vec<String> {
        self.inner.trainable().to_vec()
    }

    /// Global gradient norm of the last step, before clipping; `None` on the
    /// resident path, where the gradients never leave the device.
    fn grad_norm(&self) -> Option<f32> {
        self.inner.grad_norm()
    }

    /// Whether the optimizer update actually runs on-device.
    fn is_resident(&self) -> bool {
        self.inner.is_resident()
    }

    /// Every weight, trainable and fixed, as `{name: float32 array}` in its
    /// declared shape.
    fn params<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        for (name, data) in self.inner.params() {
            let shape = self
                .inner
                .param_shape(&name)
                .map(<[usize]>::to_vec)
                .filter(|s| !s.is_empty() && s.iter().product::<usize>() == data.len())
                .unwrap_or_else(|| vec![data.len()]);
            out.set_item(&name, data.into_pyarray_bound(py).reshape(shape)?)?;
        }
        Ok(out)
    }

    /// Overwrite weights — resuming from a checkpoint → how many were trainable.
    fn set_params(&mut self, params: &Bound<'_, PyDict>) -> PyResult<usize> {
        let incoming = f32_dict(params, "set_params")?;
        self.inner.set_params(incoming).map_err(to_py)
    }

    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr);
    }

    fn steps(&self) -> u64 {
        self.inner.steps()
    }

    fn device(&self) -> &'static str {
        self.inner.device_label()
    }

    /// Write a GGUF checkpoint → tensors written.
    fn save(&mut self, path: &str) -> PyResult<usize> {
        self.inner.save(path).map_err(to_py)
    }

    /// Restore a checkpoint → `{step, restored, optimizer_restored,
    /// had_optimizer_state, unknown}`.
    fn load(&mut self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let report = self.inner.load(path).map_err(to_py)?;
        Ok(from_json(py, &report))
    }

    /// `Trainer.from_package(path, options=None, **kw)`: a function of the
    /// `.rlxp` at `path` trained, whatever the model is. options: `function`,
    /// `dims`, `loss`, `train` (`"all"`, `{"params": [...]}`, `{"lora":
    /// {"targets", "rank", "alpha"}}`, `{"sae": {"tap", "width", "l1"}}`),
    /// `optimizer`, `device`, `policy`, `clip_norm`, `resident`, `schedule`,
    /// `accumulate`, `loss_scale`.
    #[cfg(feature = "models")]
    #[staticmethod]
    #[pyo3(signature = (path, options = None, **kw))]
    fn from_package(
        path: &str,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let o = crate::model::options(options, kw)?;
        Ok(Self {
            inner: Trainer::from_package(path, &o).map_err(to_py)?,
        })
    }

    /// Write what a package trainer trained at `path`: LoRA factors as an
    /// adapter package (`name` its name), an SAE as its own package, else
    /// the trained package.
    #[cfg(feature = "models")]
    #[pyo3(signature = (path, name = None))]
    fn save_package(&mut self, path: &str, name: Option<&str>) -> PyResult<()> {
        self.inner.save_package(path, name).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Trainer device={} params={} steps={}{}>",
            self.inner.device_label(),
            self.inner.trainable().len(),
            self.inner.steps(),
            self.inner.update_label()
        )
    }
}
