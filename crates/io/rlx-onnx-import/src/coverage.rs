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

//! ONNX op coverage — generated from [`crate::ops::OP_REGISTRY`].

use crate::ops::{lowered_ops, op_is_registered, rewritten_ops};

/// Rewritten to another op before lowering (count as supported in reports).
pub fn rewritten_ops_list() -> Vec<&'static str> {
    rewritten_ops()
}

/// Lowered directly in `lower.rs` / `ops/`.
pub fn lowered_ops_list() -> Vec<&'static str> {
    lowered_ops()
}

/// Legacy const aliases for callers that still reference static slices.
pub const REWRITTEN_OPS: &[&str] = &["ConvInteger", "MatMulInteger"];

pub const LOWERED_OPS: &[&str] = &[
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Max",
    "Min",
    "Mod",
    "Identity",
    "IsNaN",
    "MatMul",
    "QMatMul",
    "Gemm",
    "Relu",
    "LeakyRelu",
    "Tanh",
    "Sigmoid",
    "Sqrt",
    "Sin",
    "Cos",
    "Exp",
    "Neg",
    "Abs",
    "Atan",
    "Floor",
    "Round",
    "Erf",
    "Cast",
    "Transpose",
    "Reshape",
    "Unsqueeze",
    "Squeeze",
    "Flatten",
    "Gather",
    "Concat",
    "Softmax",
    "LayerNormalization",
    "InstanceNormalization",
    "BatchNormalization",
    "AveragePool",
    "MaxPool",
    "GlobalAveragePool",
    "Dropout",
    "Pow",
    "Clip",
    "Where",
    "Expand",
    "Equal",
    "Less",
    "Greater",
    "Not",
    "And",
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "Conv",
    "ConvTranspose",
    "Slice",
    "Shape",
    "ConstantOfShape",
    "Pad",
    "Range",
    "DynamicQuantizeLinear",
    "DynamicQuantizeLSTM",
    "Resize",
    "ScatterND",
    "ScatterElements",
    "TopK",
    "CumSum",
    "RandomNormalLike",
    "RandomUniformLike",
    "RandomNormal",
    "RandomUniform",
    "SplitToSequence",
    "ConcatFromSequence",
    "SequenceEmpty",
    "If",
    "Loop",
    "Scan",
    "ActCopy",
];

pub fn op_is_supported(op: &str) -> bool {
    op_is_registered(op)
}

pub fn registry_op_count() -> usize {
    crate::ops::OP_REGISTRY.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_covers_legacy_lists() {
        for op in REWRITTEN_OPS {
            assert!(op_is_supported(op), "missing rewritten op {op}");
        }
        for op in LOWERED_OPS {
            assert!(op_is_supported(op), "missing lowered op {op}");
        }
        assert!(registry_op_count() >= LOWERED_OPS.len());
    }
}
