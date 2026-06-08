// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! `pyrlx.GraphDevices` / `pyrlx.DevicePolicy` — multi-backend runtime switching.

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlx_ir::Graph;
use rlx_runtime::{
    DeviceBenchResult, DeviceCandidate, DeviceFallbackError, DevicePolicy, GraphDevices,
};

use crate::device::{device_label, parse_device};
use crate::dtype::parse_dtype;
use crate::graph::PyGraph;
use crate::session::shape_to_static_dims;

#[pyclass(name = "DevicePolicy", module = "pyrlx._pyrlx")]
pub(crate) struct PyDevicePolicy {
    inner: DevicePolicy,
}

#[pymethods]
impl PyDevicePolicy {
    #[staticmethod]
    fn all() -> Self {
        Self {
            inner: DevicePolicy::all(),
        }
    }

    #[staticmethod]
    fn only(devices: Vec<String>) -> PyResult<Self> {
        let mut list = Vec::with_capacity(devices.len());
        for d in devices {
            list.push(parse_device(&d)?);
        }
        Ok(Self {
            inner: DevicePolicy::only(list),
        })
    }

    #[staticmethod]
    fn from_env() -> Self {
        Self {
            inner: DevicePolicy::from_env(),
        }
    }

    fn with_deny(&self, devices: Vec<String>) -> PyResult<Self> {
        let mut list = Vec::with_capacity(devices.len());
        for d in devices {
            list.push(parse_device(&d)?);
        }
        Ok(Self {
            inner: self.inner.clone().with_deny(list),
        })
    }

    fn with_prefer(&self, devices: Vec<String>) -> PyResult<Self> {
        let mut list = Vec::with_capacity(devices.len());
        for d in devices {
            list.push(parse_device(&d)?);
        }
        Ok(Self {
            inner: self.inner.clone().with_prefer(list),
        })
    }

    fn with_benchmark_pick(&self, runs: usize) -> Self {
        Self {
            inner: self.inner.clone().with_benchmark_pick(runs),
        }
    }

    fn __repr__(&self) -> String {
        "<pyrlx.DevicePolicy>".into()
    }
}

impl PyDevicePolicy {
    pub(crate) fn to_policy(&self) -> DevicePolicy {
        self.inner.clone()
    }
}

#[pyclass(name = "GraphDevices", module = "pyrlx._pyrlx")]
pub(crate) struct PyGraphDevices {
    inner: GraphDevices,
    output_shapes: Vec<Vec<usize>>,
}

pub(crate) fn take_graph(graph: &Bound<'_, PyGraph>) -> PyResult<Graph> {
    graph.borrow_mut().inner.take().ok_or_else(|| {
        PyRuntimeError::new_err("graph already consumed — GraphDevices takes ownership")
    })
}

pub(crate) fn output_shapes_from_graph(g: &Graph) -> Vec<Vec<usize>> {
    g.outputs
        .iter()
        .map(|id| shape_to_static_dims(g.shape(*id)))
        .collect()
}

#[pymethods]
impl PyGraphDevices {
    #[new]
    #[pyo3(signature = (graph, policy=None))]
    fn new(graph: &Bound<'_, PyGraph>, policy: Option<PyRef<PyDevicePolicy>>) -> PyResult<Self> {
        let g = take_graph(graph)?;
        let output_shapes = output_shapes_from_graph(&g);
        let inner = match policy {
            Some(p) => GraphDevices::with_policy(g, p.to_policy()),
            None => GraphDevices::new(g),
        };
        Ok(Self {
            inner,
            output_shapes,
        })
    }

    #[staticmethod]
    fn from_env(graph: &Bound<'_, PyGraph>) -> PyResult<Self> {
        let g = take_graph(graph)?;
        let output_shapes = output_shapes_from_graph(&g);
        Ok(Self {
            inner: GraphDevices::from_env(g),
            output_shapes,
        })
    }

    fn devices(&self) -> Vec<&'static str> {
        self.inner
            .devices()
            .iter()
            .map(|d| device_label(*d))
            .collect()
    }

    fn fastest(&self) -> &'static str {
        device_label(self.inner.fastest())
    }

    fn report(&self) -> Vec<PyDeviceCandidate> {
        self.inner
            .report()
            .into_iter()
            .map(PyDeviceCandidate::from)
            .collect()
    }

    fn set_param(&mut self, name: &str, data: &Bound<'_, PyArrayDyn<f32>>) -> PyResult<()> {
        if !data.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "set_param('{name}'): array must be C-contiguous"
            )));
        }
        let view = unsafe { data.as_slice()? };
        self.inner.set_param(name, view);
        Ok(())
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> PyResult<()> {
        let dt = parse_dtype(dtype)?;
        self.inner.set_param_typed(name, data, dt);
        Ok(())
    }

    fn warm_all(&mut self) -> PyResult<Vec<&'static str>> {
        Ok(self
            .inner
            .warm_all()
            .map_err(map_err)?
            .into_iter()
            .map(device_label)
            .collect())
    }

    fn benchmark(
        &mut self,
        _py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        runs: usize,
    ) -> PyResult<Vec<PyDeviceBenchResult>> {
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        Ok(self
            .inner
            .benchmark(&pairs, runs)
            .map_err(map_err)?
            .into_iter()
            .map(PyDeviceBenchResult::from)
            .collect())
    }

    fn run<'py>(
        &mut self,
        py: Python<'py>,
        device: &str,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dev = parse_device(device)?;
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let outs = self.inner.run(dev, &pairs).map_err(map_err)?;
        outputs_to_pylist(py, outs, &self.output_shapes)
    }

    #[pyo3(signature = (inputs, device=None))]
    fn run_resolved<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        device: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let hint = match device {
            Some(d) => Some(parse_device(d)?),
            None => None,
        };
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let outs = self.inner.run_resolved(hint, &pairs).map_err(map_err)?;
        outputs_to_pylist(py, outs, &self.output_shapes)
    }

    fn run_try<'py>(
        &mut self,
        py: Python<'py>,
        chain: Vec<String>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<(String, Bound<'py, PyList>)> {
        let mut devices = Vec::with_capacity(chain.len());
        for d in chain {
            devices.push(parse_device(&d)?);
        }
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let (dev, outs) = self.inner.run_try(&devices, &pairs).map_err(map_fallback)?;
        let list = outputs_to_pylist(py, outs, &self.output_shapes)?;
        Ok((device_label(dev).to_string(), list))
    }

    #[pyo3(signature = (inputs, device=None))]
    fn run_chain<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        device: Option<&str>,
    ) -> PyResult<(String, Bound<'py, PyList>)> {
        let hint = match device {
            Some(d) => Some(parse_device(d)?),
            None => None,
        };
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let (dev, outs) = self.inner.run_chain(hint, &pairs).map_err(map_fallback)?;
        let list = outputs_to_pylist(py, outs, &self.output_shapes)?;
        Ok((device_label(dev).to_string(), list))
    }

    #[pyo3(signature = (inputs, device=None))]
    fn run_resolved_with_inputs<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        device: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let hint = match device {
            Some(d) => Some(parse_device(d)?),
            None => None,
        };
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let outs = self
            .inner
            .run_resolved_with_inputs(hint, &pairs)
            .map_err(map_err)?;
        outputs_to_pylist(py, outs, &self.output_shapes)
    }

    fn __repr__(&self) -> String {
        format!("<pyrlx.GraphDevices devices={:?}>", self.devices())
    }
}

#[pyclass(name = "DeviceCandidate", module = "pyrlx._pyrlx")]
pub(crate) struct PyDeviceCandidate {
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    available: bool,
    #[pyo3(get)]
    registered: bool,
    #[pyo3(get)]
    supports_graph: bool,
    #[pyo3(get)]
    recommended: bool,
    #[pyo3(get)]
    blocker: Option<String>,
}

impl From<DeviceCandidate> for PyDeviceCandidate {
    fn from(row: DeviceCandidate) -> Self {
        Self {
            label: row.label.to_string(),
            available: row.available,
            registered: row.registered,
            supports_graph: row.supports_graph,
            recommended: row.recommended,
            blocker: row.blocker,
        }
    }
}

#[pyclass(name = "DeviceBenchResult", module = "pyrlx._pyrlx")]
pub(crate) struct PyDeviceBenchResult {
    #[pyo3(get)]
    label: String,
    #[pyo3(get)]
    compile_ns: u64,
    #[pyo3(get)]
    median_exec_ns: u64,
}

impl From<DeviceBenchResult> for PyDeviceBenchResult {
    fn from(row: DeviceBenchResult) -> Self {
        Self {
            label: row.label.to_string(),
            compile_ns: row.compile_ns,
            median_exec_ns: row.median_exec_ns,
        }
    }
}

pub(crate) fn dict_to_owned_inputs(
    inputs: &Bound<'_, PyDict>,
) -> PyResult<Vec<(String, Vec<f32>)>> {
    let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
    for (k, v) in inputs.iter() {
        let name: String = k.extract()?;
        let arr = v.downcast::<PyArrayDyn<f32>>().map_err(|_| {
            PyValueError::new_err(format!(
                "input '{name}': expected numpy.ndarray of dtype float32"
            ))
        })?;
        if !arr.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "input '{name}': array must be C-contiguous"
            )));
        }
        let slice = unsafe { arr.as_slice()? };
        owned.push((name, slice.to_vec()));
    }
    Ok(owned)
}

pub(crate) fn pairs_refs(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect()
}

pub(crate) fn outputs_to_pylist<'py>(
    py: Python<'py>,
    outs: Vec<Vec<f32>>,
    output_shapes: &[Vec<usize>],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty_bound(py);
    for (i, out) in outs.into_iter().enumerate() {
        let shape = output_shapes.get(i).cloned().unwrap_or_default();
        let arr_1d = out.into_pyarray_bound(py);
        if !shape.is_empty()
            && shape.iter().all(|&d| d > 0)
            && shape.iter().product::<usize>() == arr_1d.len()
        {
            list.append(arr_1d.reshape(shape)?)?;
        } else {
            list.append(arr_1d)?;
        }
    }
    Ok(list)
}

pub(crate) fn map_err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

pub(crate) fn map_fallback(e: DeviceFallbackError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
