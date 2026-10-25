// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pyrlx.check(graph, …)` — device-free static diagnostics.
//!
//! The same analysis `cargo rlx check` runs: shape and representation
//! legality, per-backend op claim, missed fusions, numeric hazards,
//! memory-plan safety and the resource budget. No driver is needed, so this
//! answers "will CUDA take this graph" on a laptop.
//!
//! ```python
//! report = rlx.check(g, backends=["cpu", "cuda", "metal"])
//! for d in report["diagnostics"]:
//!     if d["severity"] == "error":
//!         print(d["code"], d["message"], d["hint"] or "")
//! ```

use pyo3::prelude::*;
use rlx_bind::runtime::serde_json::json;

use crate::bind::{from_json, to_py};
use crate::graph::{PyGraph, graph_ref};

/// `check(graph, backends=None, *, dispatch, fusion, numeric, repr, schedule,
/// plan, budget)` → `{graph, nodes, errors, warnings, diagnostics, backends}`.
#[pyfunction]
#[pyo3(signature = (graph, backends = None, *, dispatch = true, fusion = true, numeric = true,
                    repr = true, schedule = true, plan = true, budget = true))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn check(
    py: Python<'_>,
    graph: &Bound<'_, PyGraph>,
    backends: Option<Vec<String>>,
    dispatch: bool,
    fusion: bool,
    numeric: bool,
    repr: bool,
    schedule: bool,
    plan: bool,
    budget: bool,
) -> PyResult<PyObject> {
    let opts = json!({
        "dispatch": dispatch, "fusion": fusion, "numeric": numeric, "repr": repr,
        "schedule": schedule, "plan": plan, "budget": budget,
    });
    let borrowed = graph.borrow();
    let report = rlx_bind::runtime::check::check(graph_ref(&borrowed)?, backends.as_deref(), &opts)
        .map_err(to_py)?;
    Ok(from_json(py, &report))
}

/// Every backend `check` knows, compiled into this build or not.
#[pyfunction]
pub(crate) fn check_backends() -> Vec<&'static str> {
    rlx_bind::runtime::check::backends()
}
