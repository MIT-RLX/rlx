// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pyrlx.Session` and `pyrlx.CompiledGraph` — the low-level run surface.
//!
//! The logic is `rlx_bind::runtime::{Session, Compiled}`, shared with rlx-js
//! and the C ABI; this file is the Python adapter. Inputs are borrowed straight
//! out of C-contiguous float32 numpy arrays (no copy), and `run` reshapes each
//! output to its declared dims.

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rlx_bind::runtime::{CompileRequest, Compiled, Session};
use rlx_ir::Graph;

use crate::bind::to_py;
use crate::fusion_options::{PyFusionOptions, compile_request};
use crate::graph::PyGraph;

#[pyclass(name = "Session", module = "pyrlx._pyrlx")]
pub(crate) struct PySession {
    inner: Session,
}

fn take(graph: &Bound<'_, PyGraph>) -> PyResult<Graph> {
    graph.borrow_mut().inner.take().ok_or_else(|| {
        PyRuntimeError::new_err("graph already consumed — Session.compile takes ownership")
    })
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (device = "cpu", precision = "f32", policy = None))]
    fn new(device: &str, precision: &str, policy: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let policy = crate::policy::parse_policy(policy)?;
        Ok(Self {
            inner: Session::new(device, precision, policy).map_err(to_py)?,
        })
    }

    #[getter]
    fn device(&self) -> &'static str {
        self.inner.device_label()
    }

    #[getter]
    fn policy(&self) -> Option<&'static str> {
        self.inner.policy_label()
    }

    #[getter]
    fn precision(&self) -> &'static str {
        self.inner.precision_label()
    }

    /// Compile `graph` (consumed) with the session's precision and policy.
    fn compile(&self, graph: &Bound<'_, PyGraph>) -> PyResult<PyCompiled> {
        let g = take(graph)?;
        let compiled = self
            .inner
            .compile(g, &CompileRequest::default())
            .map_err(to_py)?;
        Ok(PyCompiled::new(compiled))
    }

    /// Compile with fusion toggles, a kernel-dispatch policy, and/or a
    /// precision policy (which wins over the session's).
    #[pyo3(signature = (graph, fusion_options=None, kernel_dispatch=None, policy=None))]
    fn compile_with(
        &self,
        graph: &Bound<'_, PyGraph>,
        fusion_options: Option<PyRef<PyFusionOptions>>,
        kernel_dispatch: Option<&str>,
        policy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompiled> {
        let policy = crate::policy::parse_policy(policy)?;
        let req = compile_request(fusion_options, kernel_dispatch, policy)?;
        let g = take(graph)?;
        let compiled = self.inner.compile(g, &req).map_err(to_py)?;
        Ok(PyCompiled::new(compiled))
    }

    fn __repr__(&self) -> String {
        match self.policy() {
            Some(policy) => format!(
                "<pyrlx.Session device={} precision={} policy={policy}>",
                self.device(),
                self.precision()
            ),
            None => format!(
                "<pyrlx.Session device={} precision={}>",
                self.device(),
                self.precision()
            ),
        }
    }
}

#[pyclass(name = "CompiledGraph", module = "pyrlx._pyrlx")]
pub(crate) struct PyCompiled {
    inner: Compiled,
}

impl PyCompiled {
    pub(crate) fn new(inner: Compiled) -> Self {
        Self { inner }
    }
}

/// A float32 array's contents, borrowed — or an error naming the argument.
pub(crate) fn f32_view<'a>(name: &str, v: &'a Bound<'_, PyAny>) -> PyResult<&'a [f32]> {
    let arr = v.downcast::<PyArrayDyn<f32>>().map_err(|_| {
        PyValueError::new_err(format!("'{name}': expected numpy.ndarray of dtype float32"))
    })?;
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "'{name}': array must be C-contiguous (call numpy.ascontiguousarray)"
        )));
    }
    // Safety: contiguous, and the GIL is held for as long as the view lives
    // (the caller hands it to rlx and drops it before returning to Python).
    unsafe { arr.as_slice() }.map_err(Into::into)
}

#[pymethods]
impl PyCompiled {
    #[getter]
    fn device(&self) -> &'static str {
        self.inner.device_label()
    }

    fn set_param(&mut self, name: &str, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = f32_view(name, data)?;
        self.inner.set_param(name, view).map_err(to_py)
    }

    /// `{name: float32 array}`; returns how many were set.
    fn set_params(&mut self, params: &Bound<'_, PyDict>) -> PyResult<usize> {
        for (k, v) in params.iter() {
            let name: String = k.extract()?;
            self.set_param(&name, &v)?;
        }
        Ok(params.len())
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.inner.output_shapes.clone()
    }

    fn output_dtypes(&self) -> Vec<&'static str> {
        self.inner.output_dtype_labels()
    }

    /// Declared inputs: `[{"name", "dtype", "shape"}]` (`shape` is None when a
    /// dim is symbolic) — what `run` expects.
    fn inputs(&self, py: Python<'_>) -> PyObject {
        crate::bind::from_json(py, &self.inner.inputs())
    }

    /// Declared parameters, as `inputs()` — what `set_param` takes.
    fn params(&self, py: Python<'_>) -> PyObject {
        crate::bind::from_json(py, &self.inner.params())
    }

    /// Outputs in `run` order: `[{"name", "dtype", "shape"}]` (`name` is ""
    /// for an output the graph does not name).
    fn outputs(&self, py: Python<'_>) -> PyObject {
        crate::bind::from_json(py, &self.inner.outputs())
    }

    /// The position in `run`'s result of the output named `name`; raises
    /// naming the closest names when there is none.
    fn output_index(&self, name: &str) -> PyResult<usize> {
        self.inner.output_index(name).map_err(to_py)
    }

    /// Raises naming the inputs `supplied` leaves unbound and the params
    /// never set.
    #[pyo3(signature = (supplied=Vec::new()))]
    fn check_bindings(&self, supplied: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = supplied.iter().map(String::as_str).collect();
        self.inner.check_bindings(&refs).map_err(to_py)
    }

    /// Declare inputs / params meant to read zeros (an empty KV state).
    fn allow_unbound(&mut self, names: Vec<String>) {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.inner.allow_unbound(&refs);
    }

    /// `{name: float32 array}` → one array per output, reshaped to its
    /// declared dims when they are all static.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let outs = self.run_raw(inputs)?;
        let list = PyList::empty_bound(py);
        for (i, out) in outs.into_iter().enumerate() {
            let shape = self.inner.output_shapes.get(i).cloned().unwrap_or_default();
            let arr = out.into_pyarray_bound(py);
            if !shape.is_empty()
                && shape.iter().all(|&d| d > 0)
                && shape.iter().product::<usize>() == arr.len()
            {
                list.append(arr.reshape(shape)?)?;
            } else {
                list.append(arr)?;
            }
        }
        Ok(list)
    }

    /// As `run`, but every output stays flat.
    fn run_flat<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let outs = self.run_raw(inputs)?;
        let list = PyList::empty_bound(py);
        for out in outs {
            list.append(out.into_pyarray_bound(py))?;
        }
        Ok(list)
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> PyResult<()> {
        self.inner.set_param_typed(name, data, dtype).map_err(to_py)
    }

    /// `{name: (bytes, dtype)}` → `[(bytes, dtype), …]`.
    fn run_typed<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut owned: Vec<(String, Vec<u8>, String)> = Vec::with_capacity(inputs.len());
        for (k, v) in inputs.iter() {
            let name: String = k.extract()?;
            let (bytes, dtype): (Vec<u8>, String) = v.extract().map_err(|_| {
                PyValueError::new_err(format!("input '{name}': expected tuple (bytes, dtype_str)"))
            })?;
            owned.push((name, bytes, dtype));
        }
        let refs: Vec<(&str, &[u8], &str)> = owned
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), t.as_str()))
            .collect();
        let outs = self.inner.run_typed(&refs).map_err(to_py)?;
        let list = PyList::empty_bound(py);
        for (bytes, dtype) in outs {
            list.append((PyBytes::new_bound(py, &bytes), dtype))?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.CompiledGraph device={} outputs={}>",
            self.inner.device_label(),
            self.inner.output_shapes.len()
        )
    }
}

impl PyCompiled {
    /// Borrow every input array and run.
    fn run_raw(&mut self, inputs: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<f32>>> {
        let items: Vec<(String, Bound<'_, PyAny>)> = inputs
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v)))
            .collect::<PyResult<_>>()?;
        let mut pairs: Vec<(&str, &[f32])> = Vec::with_capacity(items.len());
        for (name, v) in &items {
            pairs.push((name.as_str(), f32_view(name, v)?));
        }
        self.inner.run(&pairs).map_err(to_py)
    }
}
