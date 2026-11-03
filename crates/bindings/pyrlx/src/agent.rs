// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pyrlx.AgentOptimizer`: a coding agent rewriting the optimizer while a
//! `pyrlx.Trainer` trains (`rlx_bind::runtime::agent`).
//!
//! ```python
//! agent = rlx.AgentOptimizer("MLP on MNIST; judged by test accuracy at the end", 10_000,
//!                            optimizer={"kind": "adamw", "lr": 3e-4},
//!                            program="python", backend="claude", max_cost_usd=20)
//! for batch in data:
//!     agent.train_step(trainer, batch)
//! print(agent.close())
//! ```
//!
//! Two program languages. `program="rules"` is `updater.json` (rlx-optim
//! algorithms per group of params, run natively). `program="python"` is
//! `updater.py`, a module whose `build(params, previous, spec)` returns an
//! object with `step(params, grads, ctx)` over float32 numpy arrays — any
//! update rule, run inside the training process. Both share `analysis.json`.
//!
//! An agent CLI reaches its workspace through an MCP server; here that server
//! is this extension itself (`python -m pyrlx.workspace_mcp`), so nothing
//! else needs installing.

use std::any::Any;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use rlx_bind::runtime::agent::cli::{CliDefaults, main_with};
use rlx_bind::runtime::agent::rlx_agent::workspace::Workspace;
use rlx_bind::runtime::agent::{
    AgentConfig, AgentOptimizer, Backend, Files, ParamInfo, Program, ProgramHost, RuleHost,
    StepCtx, backend_from_json,
};
use rlx_bind::runtime::serde_json::{Value, json};
use rlx_bind::runtime::train::{OptItem, OptimizerState};
use rlx_bind::{ApiError, ApiResult};

use crate::bind::{from_json, to_json, to_py};
use crate::train::{PyTrainer, batch, exchange, pairs, state_from_py, state_to_py};

const UPDATER_PY: &str = "updater.py";

/// The starting `updater.py`: the training's own optimizer through
/// `pyrlx.Optimizer`, the same rlx-optim code the trainer runs.
const STARTING_PY: &str = r#""""Your updater. build(params, previous, spec) returns the object whose step() makes every update.

Starting program: the training's own optimizer, unchanged. pyrlx.Optimizer runs the same rlx-optim code as
the trainer, so until this file changes the run trains exactly as it would without you.
"""
import json

import numpy as np
import pyrlx

START = json.loads(r'''{start}''')


def build(params, previous, spec):
    return Updater()


class Updater:
    def __init__(self):
        self.opt = pyrlx.Optimizer(**START)
        self.lr = None

    def step(self, params, grads, ctx):
        lr = ctx["base_lr"]
        if lr is not None and lr != self.lr:
            self.opt.set_lr(lr)
            self.lr = lr
        out = self.opt.step({name: (params[name], grads[name]) for name in params})
        for name in params:
            params[name][...] = out[name]

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state):
        return self.opt.load_state_dict(state)
"#;

/// A Python exception as the agent should read it: traceback, then the error.
fn error_text(py: Python<'_>, e: &PyErr) -> String {
    let tb = e
        .traceback_bound(py)
        .and_then(|t| t.format().ok())
        .unwrap_or_default();
    format!("{tb}{e}")
}

fn api_error(py: Python<'_>, e: &PyErr) -> ApiError {
    ApiError::runtime(error_text(py, e))
}

/// `updater.py` programs.
struct PyHost {
    start: Value,
    total: u64,
    builds: u64,
}

impl PyHost {
    fn new(optimizer: &Value, total: u64) -> ApiResult<Self> {
        // The same normalisation as the rules host, so both start from one config.
        RuleHost::new(optimizer)?;
        let start = match optimizer {
            Value::Null => json!({"kind": "adamw"}),
            Value::String(kind) => json!({ "kind": kind }),
            other => other.clone(),
        };
        Ok(Self {
            start,
            total,
            builds: 0,
        })
    }
}

impl ProgramHost for PyHost {
    fn starting_files(&self) -> Files {
        let text = STARTING_PY.replace("{start}", &self.start.to_string());
        Files::from([(UPDATER_PY.to_string(), text)])
    }

    fn build(
        &mut self,
        files: &Files,
        params: &[ParamInfo],
        previous: Option<&dyn Program>,
    ) -> ApiResult<Box<dyn Program>> {
        self.builds += 1;
        let name = format!("rlx_agent_updater_{}", self.builds);
        Python::with_gil(|py| {
            let built = (|| -> PyResult<Box<dyn Program>> {
                let src = files
                    .get(UPDATER_PY)
                    .ok_or_else(|| PyTypeError::new_err(format!("{UPDATER_PY} is missing")))?;
                let module = PyModule::from_code_bound(py, src, UPDATER_PY, &name)?;
                // Each version is its own module; it lives on through its
                // functions, not through sys.modules.
                let _ = py.import_bound("sys")?.getattr("modules")?.del_item(&name);
                let list = PyList::empty_bound(py);
                for p in params {
                    list.append((p.name.as_str(), PyTuple::new_bound(py, &p.shape)))?;
                }
                let prev: PyObject = previous
                    .and_then(|p| p.as_any().downcast_ref::<PyProgram>())
                    .map(|p| p.obj.clone_ref(py))
                    .unwrap_or_else(|| py.None());
                let spec = PyDict::new_bound(py);
                spec.set_item("total_updates", self.total)?;
                spec.set_item("starting_optimizer", from_json(py, &self.start))?;
                let obj = module.getattr("build")?.call1((list, prev, spec))?;
                if !obj.hasattr("step")? {
                    return Err(PyTypeError::new_err(
                        "build() must return an object with a step(params, grads, ctx) method",
                    ));
                }
                Ok(Box::new(PyProgram { obj: obj.unbind() }))
            })();
            built.map_err(|e| ApiError::type_(error_text(py, &e)))
        })
    }

    fn interface(&self) -> String {
        format!(
            "Your updater is {UPDATER_PY}, a Python module in this folder (reference/starting/{UPDATER_PY} \
             is the starting one):\n\
             - build(params, previous, spec) returns the updater. params is the list of (name, shape) \
             at the end of this prompt; previous is the updater running now (None at the start), to \
             carry state over from (do not change it); spec is {{\"total_updates\", \"starting_optimizer\"}}.\n\
             - updater.step(params, grads, ctx) makes one update. params and grads are dicts of float32 \
             numpy arrays keyed by name, in that order; change params in place (or replace entries \
             with float32 arrays of the same shape). ctx is {{\"update\" (from 1), \"total_updates\", \
             \"base_lr\" (the training script's current learning rate, or None), \"loss\" (this \
             update's training loss, computed before it)}}. Write every schedule as a function of \
             ctx[\"update\"] / ctx[\"total_updates\"].\n\
             - To skip an update, give the updater `can_skip = True` and return True from step: the \
             params and your state_dict() go back to before it (this keeps a copy of both before every \
             update). After 10 skips in a row the update is made anyway and you are asked.\n\
             - Optional, for checkpoints: state_dict() returning {{\"step\": int, \"buffers\": {{name: \
             float32 array}}}}, and load_state_dict(state). Without them a resumed run restarts your \
             state from zero.\n\
             - numpy and pyrlx are available. pyrlx.Optimizer(kind, **options) runs one of rlx-optim's \
             algorithms ({}) over named arrays: step({{name: (param, grad)}}) returns the updated \
             arrays; set_lr(lr), state_dict(), load_state_dict(state).\n\
             - Your code runs inside the training process: use only what it is given.",
            rlx_bind::runtime::train::KINDS.join(", ")
        )
    }
}

/// A built `updater.py`'s updater object.
struct PyProgram {
    obj: PyObject,
}

impl Program for PyProgram {
    fn step(&mut self, items: &mut [OptItem<'_>], ctx: &StepCtx) -> ApiResult<bool> {
        Python::with_gil(|py| {
            let obj = self.obj.bind(py);
            let c = PyDict::new_bound(py);
            let mut skip = false;
            let made = (|| {
                c.set_item("update", ctx.update)?;
                c.set_item("total_updates", ctx.total)?;
                c.set_item("base_lr", ctx.base_lr)?;
                c.set_item("loss", ctx.loss)?;
                exchange(py, items, "updater.py step", |params, grads| {
                    let r = obj.call_method1("step", (params.clone(), grads, c.clone()))?;
                    // `return True` asks to skip this update.
                    skip = r.extract::<bool>().unwrap_or(false);
                    Ok(())
                })
            })();
            made.map(|()| skip).map_err(|e| api_error(py, &e))
        })
    }

    fn can_skip(&self) -> bool {
        Python::with_gil(|py| {
            self.obj
                .bind(py)
                .getattr("can_skip")
                .and_then(|v| v.is_truthy())
                .unwrap_or(false)
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn state_dict(&self) -> Option<OptimizerState> {
        Python::with_gil(|py| {
            let obj = self.obj.bind(py);
            if !obj.hasattr("state_dict").ok()? {
                return None;
            }
            let s = obj.call_method0("state_dict").ok()?;
            if s.is_none() {
                return None;
            }
            state_from_py(&s).ok()
        })
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> bool {
        Python::with_gil(|py| {
            let obj = self.obj.bind(py);
            if !obj.hasattr("load_state_dict").unwrap_or(false) {
                return false;
            }
            state_to_py(py, state)
                .and_then(|s| obj.call_method1("load_state_dict", (s,)))
                .is_ok_and(|r| !r.extract::<bool>().is_ok_and(|ok| !ok))
        })
    }
}

/// The run's config, host and backend from the Python arguments.
#[allow(clippy::too_many_arguments)]
fn parts(
    py: Python<'_>,
    goal: &str,
    total_steps: u64,
    optimizer: Option<&Bound<'_, PyAny>>,
    program: &str,
    backend: &str,
    backend_options: Option<&Bound<'_, PyDict>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<(AgentConfig, Box<dyn ProgramHost>, Box<dyn Backend>)> {
    let optimizer = match optimizer {
        None => Value::Null,
        Some(o) => match o.extract::<String>() {
            Ok(kind) => json!({ "kind": kind }),
            Err(_) => to_json(o)?,
        },
    };
    let mut cfg = match options {
        Some(o) => to_json(o.as_any())?,
        None => json!({}),
    };
    // `share_code` (default true): the training script itself goes into the
    // agent's reference/code/, with any files listed in `share`.
    let share_code = match cfg.as_object_mut().and_then(|m| m.remove("share_code")) {
        None => true,
        Some(Value::Bool(b)) => b,
        Some(other) => {
            return Err(PyTypeError::new_err(format!(
                "share_code is a boolean, got {other}"
            )));
        }
    };
    if share_code {
        let argv: Vec<String> = py.import_bound("sys")?.getattr("argv")?.extract()?;
        if let Some(script) = argv.first().filter(|s| std::path::Path::new(s).is_file()) {
            let mut share = cfg["share"].as_array().cloned().unwrap_or_default();
            if !share.iter().any(|p| p.as_str() == Some(script)) {
                share.insert(0, json!(script));
            }
            cfg["share"] = json!(share);
        }
    }
    cfg["goal"] = json!(goal);
    cfg["total_steps"] = json!(total_steps);
    let cfg = AgentConfig::from_json(&cfg).map_err(to_py)?;
    let host: Box<dyn ProgramHost> = match program {
        "rules" => Box::new(RuleHost::new(&optimizer).map_err(to_py)?),
        "python" => Box::new(PyHost::new(&optimizer, total_steps).map_err(to_py)?),
        other => {
            return Err(PyTypeError::new_err(format!(
                "program is \"rules\" (updater.json) or \"python\" (updater.py), got {other:?}"
            )));
        }
    };
    let mut bopts = match backend_options {
        Some(o) => to_json(o.as_any())?,
        None => json!({}),
    };
    // An agent CLI reaches its workspace through this extension's own server.
    if !matches!(backend, "keep" | "mailbox") && bopts.get("mcp_command").is_none() {
        bopts["mcp_command"] = json!(own_mcp_command(py)?);
    }
    let backend = backend_from_json(backend, &bopts).map_err(to_py)?;
    Ok((cfg, host, backend))
}

/// `python -m pyrlx.workspace_mcp`, with this interpreter.
fn own_mcp_command(py: Python<'_>) -> PyResult<Vec<String>> {
    let exe: String = py.import_bound("sys")?.getattr("executable")?.extract()?;
    Ok(vec![exe, "-m".into(), "pyrlx.workspace_mcp".into()])
}

#[pyclass(name = "AgentOptimizer", module = "pyrlx._pyrlx", unsendable)]
pub(crate) struct PyAgentOptimizer {
    inner: Option<AgentOptimizer>,
}

impl PyAgentOptimizer {
    fn agent(&self) -> PyResult<&AgentOptimizer> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("AgentOptimizer: already closed"))
    }

    fn agent_mut(&mut self) -> PyResult<&mut AgentOptimizer> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("AgentOptimizer: already closed"))
    }
}

#[pymethods]
impl PyAgentOptimizer {
    /// `AgentOptimizer(goal, total_steps, optimizer=None, program="rules",
    /// backend="keep", backend_options=None, **options)`.
    ///
    /// `optimizer` is the trainer's own optimizer config (a kind name or
    /// `{"kind": …, "lr": …}`): the starting program runs exactly it.
    /// `program` is `"rules"` or `"python"`; `backend` is `"keep"` (no agent),
    /// `"claude"`, `"cursor"` or `"copilot"`, with `backend_options`
    /// `{model, effort, max_turns, program, mcp_command}`. `options`:
    /// decisions, seconds_per_decision, max_wait_ratio, spread, every,
    /// blocking, trial_steps, snapshot_limit_gb, max_cost_usd, run_root,
    /// pause_on_nonfinite.
    #[new]
    #[pyo3(signature = (goal, total_steps, optimizer = None, program = "rules", backend = "keep",
                        backend_options = None, **options))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        goal: &str,
        total_steps: u64,
        optimizer: Option<&Bound<'_, PyAny>>,
        program: &str,
        backend: &str,
        backend_options: Option<&Bound<'_, PyDict>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (cfg, host, backend) = parts(
            py,
            goal,
            total_steps,
            optimizer,
            program,
            backend,
            backend_options,
            options,
        )?;
        Ok(Self {
            inner: Some(AgentOptimizer::new(cfg, host, backend).map_err(to_py)?),
        })
    }

    /// `AgentOptimizer.resume(path, goal, total_steps, …)`: continue from
    /// `save(path)`, with the same arguments as at the start.
    #[staticmethod]
    #[pyo3(signature = (path, goal, total_steps, optimizer = None, program = "rules",
                        backend = "keep", backend_options = None, **options))]
    #[allow(clippy::too_many_arguments)]
    fn resume(
        py: Python<'_>,
        path: &str,
        goal: &str,
        total_steps: u64,
        optimizer: Option<&Bound<'_, PyAny>>,
        program: &str,
        backend: &str,
        backend_options: Option<&Bound<'_, PyDict>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (cfg, host, backend) = parts(
            py,
            goal,
            total_steps,
            optimizer,
            program,
            backend,
            backend_options,
            options,
        )?;
        let agent = AgentOptimizer::resume(cfg, host, backend, std::path::Path::new(path))
            .map_err(to_py)?;
        Ok(Self { inner: Some(agent) })
    }

    /// `train_step(trainer, batch)` → the loss: one step of `trainer`, the
    /// update made by the agent's program.
    fn train_step(
        &mut self,
        trainer: &Bound<'_, PyTrainer>,
        batch: &Bound<'_, PyDict>,
    ) -> PyResult<f32> {
        let owned = self::batch(batch)?;
        let mut t = trainer.borrow_mut();
        let agent = self.agent_mut()?;
        agent
            .train_step(&mut t.inner, &pairs(&owned))
            .map_err(to_py)
    }

    /// `AgentOptimizer.replay(path, goal, total_steps, optimizer=None,
    /// program="rules", **options)`: a run whose starting program is a finished
    /// run's final one (or a `programs/vNNN` folder), with no agent.
    #[staticmethod]
    #[pyo3(signature = (path, goal, total_steps, optimizer = None, program = "rules", **options))]
    fn replay(
        py: Python<'_>,
        path: &str,
        goal: &str,
        total_steps: u64,
        optimizer: Option<&Bound<'_, PyAny>>,
        program: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (cfg, host, _) = parts(
            py,
            goal,
            total_steps,
            optimizer,
            program,
            "keep",
            None,
            options,
        )?;
        let agent = AgentOptimizer::replay(cfg, host, std::path::Path::new(path)).map_err(to_py)?;
        Ok(Self { inner: Some(agent) })
    }

    /// The training script's learning rate (its schedule).
    fn set_lr(&mut self, lr: f32) -> PyResult<()> {
        self.agent_mut()?.set_lr(lr);
        Ok(())
    }

    /// `report_validation(loss=None, accuracy=None, update=None)`: a held-out
    /// validation result, measured after `update` (default: the last one).
    #[pyo3(signature = (loss = None, accuracy = None, update = None))]
    fn report_validation(
        &mut self,
        loss: Option<f64>,
        accuracy: Option<f64>,
        update: Option<u64>,
    ) -> PyResult<()> {
        self.agent_mut()?.report_validation(loss, accuracy, update);
        Ok(())
    }

    /// `log(values=None, show_agent=False, **values)`: record values at the
    /// last update; with `show_agent` the agent sees them next decision.
    #[pyo3(signature = (values = None, show_agent = false, **kw))]
    fn log(
        &mut self,
        values: Option<&Bound<'_, PyDict>>,
        show_agent: bool,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut all = rlx_bind::runtime::serde_json::Map::new();
        for d in [values, kw].into_iter().flatten() {
            if let Value::Object(m) = to_json(d.as_any())? {
                all.extend(m);
            }
        }
        self.agent_mut()?
            .log(Value::Object(all), show_agent)
            .map_err(to_py)
    }

    /// The version of the program making updates now.
    fn version(&self) -> PyResult<u32> {
        Ok(self.agent()?.version())
    }

    fn run_dir(&self) -> PyResult<String> {
        Ok(self.agent()?.run_dir().display().to_string())
    }

    /// The running program's files.
    fn files(&self) -> PyResult<std::collections::BTreeMap<String, String>> {
        Ok(self.agent()?.files().clone())
    }

    /// The scheduled decision updates, once known.
    fn schedule(&self) -> PyResult<Option<Vec<u64>>> {
        Ok(self.agent()?.schedule().map(<[u64]>::to_vec))
    }

    /// Checkpoint the agent's side of the run into `path` (save the trainer too).
    fn save(&mut self, path: &str) -> PyResult<()> {
        self.agent_mut()?
            .save(std::path::Path::new(path))
            .map_err(to_py)
    }

    /// Finish → the run's summary.
    fn close(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let agent = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("AgentOptimizer: already closed"))?;
        Ok(from_json(py, &agent.close().map_err(to_py)?))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(a) => format!(
                "<pyrlx.AgentOptimizer v{:03} in {}>",
                a.version(),
                a.run_dir().display()
            ),
            None => "<pyrlx.AgentOptimizer closed>".into(),
        }
    }
}

/// `python -m pyrlx.workspace_mcp --dir <folder> [--write <name>]…`: the
/// workspace MCP server an agent CLI talks to (`rlx_agent::workspace`).
#[pyfunction]
#[pyo3(name = "_serve_workspace_mcp")]
pub(crate) fn serve_workspace_mcp(py: Python<'_>, args: Vec<String>) -> PyResult<()> {
    let ws = Workspace::from_args(&args).map_err(PyTypeError::new_err)?;
    py.allow_threads(|| ws.serve())
        .map_err(PyRuntimeError::new_err)
}

/// `python -m pyrlx.agent <command> …`: `rlx-agent-opt` (doctor, estimate,
/// show, watch, report, compare, say, broker), with this extension's own
/// workspace server → the exit code.
#[pyfunction]
#[pyo3(name = "_agent_cli")]
pub(crate) fn agent_cli(py: Python<'_>, args: Vec<String>) -> PyResult<i32> {
    let defaults = CliDefaults {
        mcp_command: Some(own_mcp_command(py)?),
    };
    Ok(py.allow_threads(|| main_with(&args, &defaults)))
}
