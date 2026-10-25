// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! String <-> rlx_ir::DType.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rlx_ir::DType;

pub(crate) fn parse_dtype(s: &str) -> PyResult<DType> {
    rlx_bind::enums::dtype(s).map_err(crate::bind::to_py)
}

#[allow(dead_code)]
/// Encode a Python scalar as little-endian bytes for `Op::Constant`.
pub(crate) fn scalar_constant_bytes(value: f64, dtype: DType) -> PyResult<Vec<u8>> {
    rlx_ir::scalar_constant_bytes(value, dtype).map_err(PyValueError::new_err)
}
