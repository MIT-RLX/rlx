// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python ↔ wire conversion for the generated `Graph` methods.
//!
//! The op table lives in `rlx-bind`; [`py_graph_ops!`](crate::py_graph_ops)
//! expands it into one `#[pymethods]` block — a method per row, with a pyo3
//! signature carrying the row's defaults, its options group flattened into
//! keyword-or-positional parameters. Each method converts its arguments to
//! wire values, calls `rlx_bind::ops::<op>`, and converts the result back.

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rlx_bind::{ApiError, ErrorKind};

use crate::graph::PyGraph;

/// An [`ApiError`] as the matching Python exception.
pub(crate) fn to_py(e: ApiError) -> PyErr {
    match e.kind {
        ErrorKind::Type | ErrorKind::Range => PyValueError::new_err(e.message),
        ErrorKind::Runtime | ErrorKind::Panic => PyRuntimeError::new_err(e.message),
        ErrorKind::NotFound => PyFileNotFoundError::new_err(e.message),
    }
}

/// `int` (every axis) or a sequence of ints — the `pair` / `triple` kinds.
#[derive(Clone, Debug)]
pub(crate) struct Dims(pub Vec<usize>);

impl From<Vec<usize>> for Dims {
    fn from(v: Vec<usize>) -> Self {
        Dims(v)
    }
}

impl<'py> FromPyObject<'py> for Dims {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(n) = ob.extract::<usize>() {
            return Ok(Dims(vec![n]));
        }
        Ok(Dims(ob.extract::<Vec<usize>>()?))
    }
}

/// One row of `pads`: `int` (symmetric) or `[before, after]`.
#[derive(Clone, Debug)]
pub(crate) struct Rows(pub Vec<Vec<usize>>);

impl<'py> FromPyObject<'py> for Rows {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let items: Vec<Dims> = ob.extract()?;
        Ok(Rows(items.into_iter().map(|d| d.0).collect()))
    }
}

/// Move a body graph out of its Python handle.
pub(crate) fn take(body: &Bound<'_, PyGraph>, what: &str) -> PyResult<rlx_ir::Graph> {
    body.borrow_mut().inner.take().ok_or_else(|| {
        PyRuntimeError::new_err(format!("{what}: body graph has already been consumed"))
    })
}

/// The Python parameter type of a kind.
#[macro_export]
#[doc(hidden)]
macro_rules! py_ty {
    (pair) => { $crate::bind::Dims };
    (triple) => { $crate::bind::Dims };
    (opt_pair) => { ::core::option::Option<$crate::bind::Dims> };
    (pads) => { $crate::bind::Rows };
    (graph) => { &::pyo3::Bound<'_, $crate::graph::PyGraph> };
    (opt_graph) => { ::core::option::Option<&::pyo3::Bound<'_, $crate::graph::PyGraph>> };
    ($kind:ident) => { rlx_bind::wire_ty!($kind) };
}

/// A table default as a Python parameter default.
#[macro_export]
#[doc(hidden)]
macro_rules! py_default {
    (pair, $d:expr) => {
        $crate::bind::Dims::from(rlx_bind::wire_default!(pair, $d))
    };
    (triple, $d:expr) => {
        $crate::bind::Dims::from(rlx_bind::wire_default!(triple, $d))
    };
    (opt_pair, $d:expr) => {
        $d
    };
    (opt_graph, $d:expr) => {
        $d
    };
    ($kind:ident, $d:expr) => {
        rlx_bind::wire_default!($kind, $d)
    };
}

/// A Python parameter → its wire value.
#[macro_export]
#[doc(hidden)]
macro_rules! py_wire {
    (pair, $v:expr, $what:expr) => {
        $v.0
    };
    (triple, $v:expr, $what:expr) => {
        $v.0
    };
    (opt_pair, $v:expr, $what:expr) => {
        $v.map(|d| d.0)
    };
    (pads, $v:expr, $what:expr) => {
        $v.0
    };
    (graph, $v:expr, $what:expr) => {
        $crate::bind::take($v, $what)?
    };
    (opt_graph, $v:expr, $what:expr) => {
        match $v {
            Some(b) => Some($crate::bind::take(b, $what)?),
            None => None,
        }
    };
    ($kind:ident, $v:expr, $what:expr) => {
        $v
    };
}

/// A wire result → a Python object.
#[macro_export]
#[doc(hidden)]
macro_rules! py_ret {
    ($py:ident, (unit), $v:expr) => {{
        let () = $v;
        $py.None()
    }};
    ($py:ident, ($($ret:ident)?), $v:expr) => {
        ::pyo3::IntoPy::<::pyo3::PyObject>::into_py($v, $py)
    };
}

/// Generate the `#[pymethods]` block for every op-table row.
///
/// Invoked once as `rlx_bind::for_each_graph_op!(crate::py_graph_ops)`.
#[macro_export]
macro_rules! py_graph_ops {
    ( []
      $(
        $(#[doc = $doc:literal])*
        $name:ident $( [ $( $rk:ident = $rv:literal ),* $(,)? ] )?
        ( $( $arg:ident : $kind:ident $( = $def:expr )? ),* $(,)?
          $( ; { $( $oarg:ident $( | $oal:ident )* : $okind:ident $( = $odef:expr )? ),* $(,)? } )? )
        $( -> $ret:ident )?
        => | $g:ident | $body:block ;
      )*
    ) => {
        #[::pyo3::pymethods]
        impl $crate::graph::PyGraph {
            $(
                $(#[doc = $doc])*
                #[pyo3(signature = (
                    $( $arg $( = $crate::py_default!($kind, $def) )? , )*
                    $( $( $oarg $( = $crate::py_default!($okind, $odef) )? , )* )?
                ))]
                #[allow(clippy::too_many_arguments, unused_variables)]
                fn $name(
                    &mut self,
                    py: ::pyo3::Python<'_>,
                    $( $arg: $crate::py_ty!($kind), )*
                    $( $( $oarg: $crate::py_ty!($okind), )* )?
                ) -> ::pyo3::PyResult<::pyo3::PyObject> {
                    $( let $arg = $crate::py_wire!($kind, $arg, stringify!($name)); )*
                    $( $( let $oarg = $crate::py_wire!($okind, $oarg, stringify!($name)); )* )?
                    let out = rlx_bind::ops::$name(self.g()?, $($arg,)* $($($oarg,)*)?)
                        .map_err($crate::bind::to_py)?;
                    Ok($crate::py_ret!(py, ($($ret)?), out))
                }
            )*
        }
    };
}

rlx_bind::for_each_graph_op!(crate::py_graph_ops);

/// Attach each row's extra Python names (`rope_n`, `fused_residual_ln`, …) to
/// the `Graph` type, next to the canonical one.
pub(crate) fn add_aliases(graph_type: &Bound<'_, pyo3::types::PyType>) -> PyResult<()> {
    for op in rlx_bind::manifest() {
        for alias in op.py_aliases() {
            let method = graph_type.getattr(op.name)?;
            graph_type.setattr(alias, method)?;
        }
    }
    Ok(())
}

// ── JSON: structured options in, reports out ────────────────

use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rlx_bind::runtime::{Value as Json, serde_json};

/// A plain Python value (dict / list / str / number / bool / None) → JSON,
/// for the options objects the core parses.
pub(crate) fn to_json(v: &Bound<'_, PyAny>) -> PyResult<Json> {
    if v.is_none() {
        return Ok(Json::Null);
    }
    // `bool` subclasses `int`, so it is checked first.
    if let Ok(b) = v.downcast::<PyBool>() {
        return Ok(Json::Bool(b.is_true()));
    }
    if v.downcast::<PyInt>().is_ok() {
        if let Ok(i) = v.extract::<i64>() {
            return Ok(Json::from(i));
        }
        return Ok(Json::from(v.extract::<u64>()?));
    }
    if let Ok(f) = v.downcast::<PyFloat>() {
        return Ok(serde_json::Number::from_f64(f.value())
            .map(Json::Number)
            .unwrap_or(Json::Null));
    }
    if let Ok(s) = v.downcast::<PyString>() {
        return Ok(Json::String(s.to_cow()?.into_owned()));
    }
    if let Ok(d) = v.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, item) in d.iter() {
            let k: String = k
                .extract()
                .map_err(|_| PyValueError::new_err("option keys must be strings"))?;
            map.insert(k, to_json(&item)?);
        }
        return Ok(Json::Object(map));
    }
    if v.downcast::<PyList>().is_ok() || v.downcast::<PyTuple>().is_ok() {
        let items: Vec<Bound<'_, PyAny>> = v.extract()?;
        return Ok(Json::Array(
            items.iter().map(to_json).collect::<PyResult<_>>()?,
        ));
    }
    // numpy scalars / arrays and other number-likes.
    if let Ok(f) = v.extract::<f64>() {
        return Ok(serde_json::Number::from_f64(f)
            .map(Json::Number)
            .unwrap_or(Json::Null));
    }
    if let Ok(items) = v.extract::<Vec<f64>>() {
        return Ok(Json::Array(
            items
                .into_iter()
                .map(|f| {
                    serde_json::Number::from_f64(f)
                        .map(Json::Number)
                        .unwrap_or(Json::Null)
                })
                .collect(),
        ));
    }
    Err(PyValueError::new_err(format!(
        "expected plain data (dict / list / str / number / bool / None), got {}",
        v.get_type().name()?
    )))
}

/// `Option<&PyAny>` → JSON, `None` and absent both `null`.
pub(crate) fn opt_json(v: Option<&Bound<'_, PyAny>>) -> PyResult<Json> {
    match v {
        Some(v) => to_json(v),
        None => Ok(Json::Null),
    }
}

/// JSON → a plain Python value, for reports.
pub(crate) fn from_json(py: Python<'_>, v: &Json) -> PyObject {
    match v {
        Json::Null => py.None(),
        Json::Bool(b) => b.into_py(py),
        Json::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.into_py(py),
            (None, Some(u)) => u.into_py(py),
            _ => n.as_f64().unwrap_or(f64::NAN).into_py(py),
        },
        Json::String(s) => s.into_py(py),
        Json::Array(items) => {
            let list = PyList::empty_bound(py);
            for i in items {
                let _ = list.append(from_json(py, i));
            }
            list.into_py(py)
        }
        Json::Object(map) => {
            let d = PyDict::new_bound(py);
            for (k, v) in map {
                let _ = d.set_item(k, from_json(py, v));
            }
            d.into_py(py)
        }
    }
}
