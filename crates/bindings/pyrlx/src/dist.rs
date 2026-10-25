// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed nodes: `pyrlx.Group` joins a mesh of rlx nodes — messages,
//! collectives, and the ship-graph protocol (a coordinator ships a stage or a
//! training job; workers run it on their own hardware).
//!
//! Every call blocks until its peers take part, with the GIL released, so the
//! ranks of a local mesh can be Python threads.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rlx_bind::runtime::dist::{self, Group};

use crate::bind::{from_json, opt_json, to_json, to_py};
use crate::graph::PyGraph;

fn graph_of(g: &Bound<'_, PyGraph>) -> PyResult<rlx_ir::Graph> {
    g.borrow()
        .inner
        .clone()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("graph already consumed"))
}

/// `world` free loopback addresses (`"127.0.0.1:port"`), for every rank on
/// one machine.
#[pyfunction]
pub fn loopback_peers(world: u32) -> PyResult<Vec<String>> {
    dist::loopback_peers(world).map_err(to_py)
}

/// This process's membership in a mesh.
#[pyclass(name = "Group", module = "pyrlx._pyrlx")]
pub(crate) struct PyGroup {
    inner: Group,
}

#[pymethods]
impl PyGroup {
    /// Join as `rank` of `world`; `options`: `{peers, topology, device,
    /// discover, heap_mb}`. Blocks until every peer connects.
    #[staticmethod]
    #[pyo3(signature = (rank, world, options=None))]
    fn connect(
        py: Python<'_>,
        rank: u32,
        world: u32,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let opts = opt_json(options)?;
        let inner = py
            .allow_threads(|| Group::connect(rank, world, &opts))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Join with `RANK` / `WORLD` / `PEERS` / … from the environment.
    #[staticmethod]
    fn from_env(py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(Group::from_env).map_err(to_py)?;
        Ok(Self { inner })
    }

    fn rank(&self) -> u32 {
        self.inner.rank()
    }

    fn world(&self) -> u32 {
        self.inner.world()
    }

    fn is_leader(&self) -> bool {
        self.inner.is_leader()
    }

    fn barrier(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.barrier()).map_err(to_py)
    }

    fn send(&self, py: Python<'_>, to: u32, tag: u32, data: Vec<f32>) -> PyResult<()> {
        py.allow_threads(|| self.inner.send(to, tag, &data))
            .map_err(to_py)
    }

    fn recv<'py>(
        &self,
        py: Python<'py>,
        from_: u32,
        tag: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = py
            .allow_threads(|| self.inner.recv(from_, tag))
            .map_err(to_py)?;
        Ok(v.into_pyarray_bound(py))
    }

    /// Reduce across every rank: `op` is `"sum"`, `"mean"`, `"max"` or `"min"`.
    #[pyo3(signature = (data, op="sum"))]
    fn all_reduce<'py>(
        &self,
        py: Python<'py>,
        data: Vec<f32>,
        op: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = py
            .allow_threads(|| self.inner.all_reduce(&data, op))
            .map_err(to_py)?;
        Ok(v.into_pyarray_bound(py))
    }

    fn all_gather<'py>(
        &self,
        py: Python<'py>,
        data: Vec<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = py
            .allow_threads(|| self.inner.all_gather(&data))
            .map_err(to_py)?;
        Ok(v.into_pyarray_bound(py))
    }

    /// Rank `root`'s `data` on every rank; other ranks pass a buffer of the
    /// root's length.
    fn broadcast<'py>(
        &self,
        py: Python<'py>,
        root: u32,
        data: Vec<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = py
            .allow_threads(|| self.inner.broadcast(root, &data))
            .map_err(to_py)?;
        Ok(v.into_pyarray_bound(py))
    }

    /// Coordinator: `[{rank, devices, platform, role}]` from every worker.
    fn collect_caps(&self, py: Python<'_>) -> PyResult<PyObject> {
        let v = py
            .allow_threads(|| self.inner.collect_caps())
            .map_err(to_py)?;
        Ok(from_json(py, &v))
    }

    /// Coordinator: ship a stage of `graph` to worker `rank`; `options`:
    /// `{input, output, device, weights, packed, stage_id}`.
    #[pyo3(signature = (rank, graph, options=None))]
    fn ship_stage(
        &self,
        py: Python<'_>,
        rank: u32,
        graph: &Bound<'_, PyGraph>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let g = graph_of(graph)?;
        let opts = opt_json(options)?;
        py.allow_threads(|| self.inner.ship_stage(rank, &g, &opts))
            .map_err(to_py)
    }

    fn send_activation(&self, py: Python<'_>, to: u32, data: Vec<f32>) -> PyResult<()> {
        py.allow_threads(|| self.inner.send_activation(to, &data))
            .map_err(to_py)
    }

    fn recv_activation<'py>(
        &self,
        py: Python<'py>,
        from_: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v = py
            .allow_threads(|| self.inner.recv_activation(from_))
            .map_err(to_py)?;
        Ok(v.into_pyarray_bound(py))
    }

    /// Worker: serve activations of the stage the coordinator ships;
    /// `options`: `{steps}`.
    #[pyo3(signature = (options=None))]
    fn serve_worker(
        &self,
        py: Python<'_>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let opts = opt_json(options)?;
        let v = py
            .allow_threads(|| self.inner.serve_worker(&opts))
            .map_err(to_py)?;
        Ok(from_json(py, &v))
    }

    /// Coordinator: ship a training job for backward `graph` to `rank`.
    fn ship_train(
        &self,
        py: Python<'_>,
        rank: u32,
        graph: &Bound<'_, PyGraph>,
        spec: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let g = graph_of(graph)?;
        let spec = to_json(spec)?;
        py.allow_threads(|| self.inner.ship_train(rank, &g, &spec))
            .map_err(to_py)
    }

    /// Train this rank's shard in lockstep: `{metrics, params}`.
    fn run_train(
        &self,
        py: Python<'_>,
        graph: &Bound<'_, PyGraph>,
        spec: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let g = graph_of(graph)?;
        let spec = to_json(spec)?;
        let v = py
            .allow_threads(|| self.inner.run_train(&g, &spec))
            .map_err(to_py)?;
        Ok(from_json(py, &v))
    }

    /// Worker: train the job the coordinator ships; `options`: `{log}`.
    #[pyo3(signature = (options=None))]
    fn serve_trainer(
        &self,
        py: Python<'_>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let opts = opt_json(options)?;
        let v = py
            .allow_threads(|| self.inner.serve_trainer(&opts))
            .map_err(to_py)?;
        Ok(from_json(py, &v))
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Group rank={} world={}>",
            self.inner.rank(),
            self.inner.world()
        )
    }
}
