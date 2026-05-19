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

//! String <-> rlx_ir::DType.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rlx_ir::DType;

pub(crate) fn parse_dtype(s: &str) -> PyResult<DType> {
    let dt = match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f64" | "float64" | "double" => DType::F64,
        "i8" | "int8" => DType::I8,
        "u8" | "uint8" => DType::U8,
        "i16" | "int16" => DType::I16,
        "i32" | "int32" => DType::I32,
        "u32" | "uint32" => DType::U32,
        "i64" | "int64" => DType::I64,
        "bool" => DType::Bool,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown dtype '{other}' (f32, f16, bf16, i32, ...)"
            )));
        }
    };
    Ok(dt)
}

#[allow(dead_code)]
pub(crate) fn dtype_label(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F64 => "f64",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C64 => "c64",
    }
}
