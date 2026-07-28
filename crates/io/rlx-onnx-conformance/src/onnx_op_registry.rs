// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX op types — registry coverage checklist for bundled models.

use rlx_onnx_import::ops::op_is_registered;

/// Op types commonly used in quantized TTS / transformer ONNX exports.
pub const BUNDLED_ONNX_OP_TYPES: &[&str] = &[
    "Abs",
    "Add",
    "And",
    "Atan",
    "Cast",
    "Clip",
    "Concat",
    "ConcatFromSequence",
    "ConstantOfShape",
    "Conv",
    "ConvTranspose",
    "Cos",
    "CumSum",
    "Div",
    "DynamicQuantizeLinear",
    "DynamicQuantizeLSTM",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "Gather",
    "Gemm",
    "Greater",
    "If",
    "InstanceNormalization",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "Loop",
    "MatMul",
    "Mul",
    "Neg",
    "Not",
    "Pad",
    "Pow",
    "RandomNormalLike",
    "RandomUniformLike",
    "RandomNormal",
    "RandomUniform",
    "Range",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Resize",
    "Round",
    "ScatterElements",
    "ScatterND",
    "SequenceEmpty",
    "Shape",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "SplitToSequence",
    "Sqrt",
    "Sub",
    "Tanh",
    "TopK",
    "Transpose",
    "Unsqueeze",
    "Where",
];

pub fn bundled_onnx_registry_coverage() -> Vec<(&'static str, bool)> {
    BUNDLED_ONNX_OP_TYPES
        .iter()
        .map(|&op| (op, op_is_registered(op)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_bundled_onnx_op_types_registered() {
        let missing: Vec<_> = bundled_onnx_registry_coverage()
            .into_iter()
            .filter(|(_, ok)| !*ok)
            .map(|(op, _)| op)
            .collect();
        assert!(
            missing.is_empty(),
            "bundled ONNX op types missing from registry: {missing:?}"
        );
    }
}
