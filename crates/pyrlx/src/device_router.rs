// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! `pyrlx.DeviceRouter` — serving wrapper with warm-all and fallback chain.

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlx_runtime::DeviceRouter;

use crate::device::{device_label, parse_device};
use crate::dtype::parse_dtype;
use crate::graph::PyGraph;
use crate::graph_devices::{
    PyDevicePolicy, dict_to_owned_inputs, map_err, map_fallback, output_shapes_from_graph,
    outputs_to_pylist, pairs_refs, take_graph,
};

#[pyclass(name = "DeviceRouter", module = "pyrlx._pyrlx")]
pub(crate) struct PyDeviceRouter {
    inner: DeviceRouter,
    output_shapes: Vec<Vec<usize>>,
}

#[pymethods]
impl PyDeviceRouter {
    #[new]
    #[pyo3(signature = (graph, policy=None))]
    fn new(graph: &Bound<'_, PyGraph>, policy: Option<PyRef<PyDevicePolicy>>) -> PyResult<Self> {
        let g = take_graph(graph)?;
        let output_shapes = output_shapes_from_graph(&g);
        let inner = match policy {
            Some(p) => DeviceRouter::new(g, p.to_policy()).map_err(map_err)?,
            None => DeviceRouter::new(g, rlx_runtime::DevicePolicy::all()).map_err(map_err)?,
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
            inner: DeviceRouter::from_env(g).map_err(map_err)?,
            output_shapes,
        })
    }

    #[pyo3(signature = (enabled=true))]
    fn with_rebench_on_throttle(&mut self, enabled: bool) {
        self.inner.set_rebench_on_throttle(enabled);
    }

    fn devices(&self) -> Vec<String> {
        self.inner
            .devices()
            .into_iter()
            .map(str::to_string)
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

    fn run_on<'py>(
        &mut self,
        py: Python<'py>,
        device: &str,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dev = parse_device(device)?;
        let owned = dict_to_owned_inputs(inputs)?;
        let pairs = pairs_refs(&owned);
        let outs = self.inner.run_on(dev, &pairs).map_err(map_err)?;
        outputs_to_pylist(py, outs, &self.output_shapes)
    }

    #[pyo3(signature = (inputs, device=None))]
    fn run<'py>(
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
        let (dev, outs) = self.inner.run(&pairs, hint).map_err(map_err)?;
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
        let (dev, outs) = self.inner.run_chain(&pairs, hint).map_err(map_fallback)?;
        let list = outputs_to_pylist(py, outs, &self.output_shapes)?;
        Ok((device_label(dev).to_string(), list))
    }

    fn __repr__(&self) -> String {
        format!("<pyrlx.DeviceRouter devices={:?}>", self.devices())
    }
}
