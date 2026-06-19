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

//! Python-scalar promotion rules mirrored from `pyrlx/dsl.py`.

use rlx_ir::{DType, Graph, GraphExt, NodeId};

/// Maximum integer magnitude representable exactly as f64.
pub(crate) const MAX_EXACT_INT: i128 = 1 << 53;

pub(crate) fn promote_scalar(g: &mut Graph, value: Scalar, dtype_hint: DType) -> NodeId {
    match value {
        Scalar::Bool(b) => g.constant(if b { 1.0 } else { 0.0 }, DType::Bool),
        Scalar::Int(v) => {
            let fv = int_to_f64(v).expect("integer literal exceeds exact f64 range");
            if dtype_hint.is_integral() {
                g.constant(fv, dtype_hint)
            } else {
                g.constant(fv, DType::F32)
            }
        }
        Scalar::Float(v) => {
            if dtype_hint.is_integral() {
                g.constant(v, dtype_hint)
            } else {
                g.constant(v, DType::F32)
            }
        }
    }
}

fn int_to_f64(v: i64) -> Result<f64, ()> {
    if v.unsigned_abs() > MAX_EXACT_INT as u64 {
        return Err(());
    }
    Ok(v as f64)
}

/// A literal operand in elementwise binary ops.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Self::Float(v as f64)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

trait IntegralDType {
    fn is_integral(&self) -> bool;
}

impl IntegralDType for DType {
    fn is_integral(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U32
        )
    }
}
