// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `pyrlx.Session` and `pyrlx.CompiledGraph` — the low-level run surface.

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rlx_ir::{Dim, Graph, Shape};
use rlx_runtime::{CompiledGraph, Precision, Session};

use crate::device::{device_label, parse_device};
use crate::dtype::{dtype_label, parse_dtype};
use crate::graph::PyGraph;

fn parse_precision(s: &str) -> PyResult<Precision> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Precision::F32,
        "f16" | "float16" | "half" => Precision::F16,
        "bf16" | "bfloat16" => Precision::BF16,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown precision '{other}' (f32, f16, bf16)"
            )));
        }
    })
}

#[pyclass(name = "Session", module = "pyrlx._pyrlx")]
pub(crate) struct PySession {
    inner: Session,
    device_label: &'static str,
}

#[pymethods]
impl PySession {
    /// `Session(device="cpu", precision="f32")`
    #[new]
    #[pyo3(signature = (device = "cpu", precision = "f32"))]
    fn new(device: &str, precision: &str) -> PyResult<Self> {
        let dev = parse_device(device)?;
        if !rlx_runtime::is_available(dev) {
            return Err(PyRuntimeError::new_err(format!(
                "device '{device}' not available in this build — rebuild pyrlx \
                 with `maturin develop --features {device}`"
            )));
        }
        let prec = parse_precision(precision)?;
        Ok(Self {
            inner: Session::new_with_precision(dev, prec),
            device_label: device_label(dev),
        })
    }

    #[getter]
    fn device(&self) -> &'static str {
        self.device_label
    }

    #[getter]
    fn precision(&self) -> &'static str {
        match self.inner.precision() {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
        }
    }

    /// Compile a graph; the graph is consumed (rlx-runtime takes Graph
    /// by value). Returns a `CompiledGraph` ready for `set_param` / `run`.
    fn compile(&self, graph: &Bound<'_, PyGraph>) -> PyResult<PyCompiled> {
        let g: Graph = graph.borrow_mut().inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("graph already consumed — Session.compile takes ownership")
        })?;

        let outputs = g
            .outputs
            .iter()
            .map(|id| {
                let s = g.shape(*id);
                (*id, shape_to_static_dims(s))
            })
            .collect::<Vec<_>>();

        let compiled = self.inner.compile(g);
        Ok(PyCompiled {
            inner: compiled,
            output_shapes: outputs.into_iter().map(|(_, dims)| dims).collect(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Session device={} precision={}>",
            self.device_label,
            self.precision()
        )
    }
}

/// Resolve a Shape into a concrete `Vec<usize>` for numpy reshape.
/// Dynamic dims fall back to 0 — caller must reshape themselves.
fn shape_to_static_dims(shape: &Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => 0,
        })
        .collect()
}

#[pyclass(name = "CompiledGraph", module = "pyrlx._pyrlx")]
pub(crate) struct PyCompiled {
    inner: CompiledGraph,
    /// One Vec<usize> per graph output — used to reshape flat results.
    output_shapes: Vec<Vec<usize>>,
}

#[pymethods]
impl PyCompiled {
    #[getter]
    fn device(&self) -> &'static str {
        device_label(self.inner.device())
    }

    /// `compiled.set_param("w", np.ones((4, 3), dtype=np.float32))`
    fn set_param(&mut self, name: &str, data: &Bound<'_, PyArrayDyn<f32>>) -> PyResult<()> {
        if !data.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "set_param('{name}'): array must be C-contiguous (call .ascontiguousarray)"
            )));
        }
        // Safety: we just checked contiguity, and numpy holds the GIL.
        let view = unsafe { data.as_slice()? };
        self.inner.set_param(name, view);
        Ok(())
    }

    /// `compiled.run({"x": np.ndarray}) -> list[np.ndarray]`
    /// Outputs are reshaped to their declared static shape.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Borrow each numpy array as &[f32]. We collect into a Vec
        // so the contiguous-views outlive the &[(&str, &[f32])] slice
        // we hand to rlx-runtime.
        let mut owned_names: Vec<String> = Vec::with_capacity(inputs.len());
        let mut views: Vec<(*const f32, usize)> = Vec::with_capacity(inputs.len());

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
            // Safety: contiguous + we hold GIL; pointer valid for the
            // duration of this call (we drop these before returning).
            let slice = unsafe { arr.as_slice()? };
            views.push((slice.as_ptr(), slice.len()));
            owned_names.push(name);
        }

        let pairs: Vec<(&str, &[f32])> = owned_names
            .iter()
            .zip(views.iter())
            .map(|(n, (p, l))| (n.as_str(), unsafe { std::slice::from_raw_parts(*p, *l) }))
            .collect();

        let outs = self.inner.run(&pairs);

        let list = PyList::empty_bound(py);
        for (i, out) in outs.into_iter().enumerate() {
            // Reshape to declared shape if all dims are static.
            let shape = self.output_shapes.get(i).cloned().unwrap_or_default();
            let arr_1d = out.into_pyarray_bound(py);
            if !shape.is_empty()
                && shape.iter().all(|&d| d > 0)
                && shape.iter().product::<usize>() == arr_1d.len()
            {
                let reshaped = arr_1d.reshape(shape)?;
                list.append(reshaped)?;
            } else {
                list.append(arr_1d)?;
            }
        }
        Ok(list)
    }

    /// Flat-vector run — bypasses reshape, returns one 1D array per output.
    /// Useful when the declared output shape includes dynamic dims.
    fn run_flat<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
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
        let pairs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let outs = self.inner.run(&pairs);
        let list = PyList::empty_bound(py);
        for out in outs {
            list.append(out.into_pyarray_bound(py))?;
        }
        Ok(list)
    }

    /// `compiled.set_param_typed("A", a.tobytes(), "f64")` — upload a
    /// param at any supported dtype. Bytes must be contiguous and in
    /// the same dtype the graph declared. Use this when the param's
    /// declared dtype isn't F32 (e.g., F64 for Circulax-style numerics).
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> PyResult<()> {
        let dt = parse_dtype(dtype)?;
        self.inner.set_param_typed(name, data, dt);
        Ok(())
    }

    /// `compiled.run_typed({"b": (b.tobytes(), "f64"), ...}) -> list[(bytes, str)]`.
    ///
    /// Each input is a `(bytes, dtype_str)` pair; outputs come back
    /// the same shape so callers can `np.frombuffer(bytes, dtype=str)`
    /// and reshape.
    ///
    /// Lossless for F64 — bytes go straight into the arena. For F16/
    /// BF16 the runtime widens to F32 internally; for F32 it's a
    /// straight copy.
    fn run_typed<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut owned: Vec<(String, Vec<u8>, rlx_ir::DType)> = Vec::with_capacity(inputs.len());
        for (k, v) in inputs.iter() {
            let name: String = k.extract()?;
            let tup: (Vec<u8>, String) = v.extract().map_err(|_| {
                PyValueError::new_err(format!("input '{name}': expected tuple (bytes, dtype_str)"))
            })?;
            let dt = parse_dtype(&tup.1)?;
            owned.push((name, tup.0, dt));
        }
        let refs: Vec<(&str, &[u8], rlx_ir::DType)> = owned
            .iter()
            .map(|(n, d, dt)| (n.as_str(), d.as_slice(), *dt))
            .collect();
        let outs = self.inner.run_typed(&refs);

        let list = PyList::empty_bound(py);
        for (bytes, dt) in outs {
            let py_bytes = PyBytes::new_bound(py, &bytes);
            let label = dtype_label(dt);
            list.append((py_bytes, label))?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.CompiledGraph device={} outputs={}>",
            device_label(self.inner.device()),
            self.output_shapes.len()
        )
    }
}

// silence "unused" if PyArray1 import not needed
#[allow(dead_code)]
fn _hint(_: PyArray1<f32>) {}
