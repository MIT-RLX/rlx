// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX op registry — single source of truth for supported ops and coverage reports.

use std::collections::HashMap;

/// How an ONNX op is handled during import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerStrategy {
    /// Lowered to MIR via a registered handler in `lower.rs` / `ops/`.
    Mir,
    /// Rewritten to another ONNX op before lowering (e.g. `MatMulInteger` → `MatMul`).
    Rewritten,
    /// Lowered to `Op::Custom("onnx.<Name>")` with a reference kernel.
    Custom,
    /// Control-flow / sequence op (may require profile-specific handling).
    ControlFlow,
}

/// Reporting / conformance grouping for registry entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Arithmetic,
    LinearAlgebra,
    Unary,
    Shape,
    NeuralNet,
    Logic,
    Reduce,
    Quant,
    Indexing,
    Random,
    ControlFlow,
}

impl OpCategory {
    pub const ALL: &[OpCategory] = &[
        OpCategory::Arithmetic,
        OpCategory::LinearAlgebra,
        OpCategory::Unary,
        OpCategory::Shape,
        OpCategory::NeuralNet,
        OpCategory::Logic,
        OpCategory::Reduce,
        OpCategory::Quant,
        OpCategory::Indexing,
        OpCategory::Random,
        OpCategory::ControlFlow,
    ];

    pub fn name(self) -> &'static str {
        match self {
            OpCategory::Arithmetic => "arithmetic",
            OpCategory::LinearAlgebra => "linear_algebra",
            OpCategory::Unary => "unary",
            OpCategory::Shape => "shape",
            OpCategory::NeuralNet => "neural_net",
            OpCategory::Logic => "logic",
            OpCategory::Reduce => "reduce",
            OpCategory::Quant => "quant",
            OpCategory::Indexing => "indexing",
            OpCategory::Random => "random",
            OpCategory::ControlFlow => "control_flow",
        }
    }
}

/// One registry entry for an ONNX operator type.
#[derive(Debug, Clone, Copy)]
pub struct OpEntry {
    pub onnx_op: &'static str,
    pub since_opset: i64,
    pub strategy: LowerStrategy,
    pub category: OpCategory,
}

macro_rules! op {
    ($name:literal, $opset:literal, $strategy:ident, $cat:ident) => {
        OpEntry {
            onnx_op: $name,
            since_opset: $opset,
            strategy: LowerStrategy::$strategy,
            category: OpCategory::$cat,
        }
    };
}

/// Static registry of ONNX ops implemented by `rlx-onnx-import`.
pub const OP_REGISTRY: &[OpEntry] = &[
    op!("Add", 1, Mir, Arithmetic),
    op!("Sub", 1, Mir, Arithmetic),
    op!("Mul", 1, Mir, Arithmetic),
    op!("Div", 1, Mir, Arithmetic),
    op!("Pow", 1, Mir, Arithmetic),
    op!("Mod", 10, Mir, Arithmetic),
    op!("Max", 1, Mir, Arithmetic),
    op!("Min", 1, Mir, Arithmetic),
    op!("MatMul", 1, Mir, LinearAlgebra),
    op!("Gemm", 1, Mir, LinearAlgebra),
    op!("QMatMul", 6, Custom, LinearAlgebra),
    op!("MatMulInteger", 10, Rewritten, LinearAlgebra),
    op!("ConvInteger", 10, Rewritten, LinearAlgebra),
    op!("Relu", 1, Mir, Unary),
    op!("LeakyRelu", 1, Mir, Unary),
    op!("Elu", 6, Mir, Unary),
    op!("Tanh", 1, Mir, Unary),
    op!("Sigmoid", 1, Mir, Unary),
    op!("Sqrt", 1, Mir, Unary),
    op!("Sin", 7, Mir, Unary),
    op!("Cos", 7, Mir, Unary),
    op!("Exp", 1, Mir, Unary),
    op!("Neg", 1, Mir, Unary),
    op!("Abs", 1, Mir, Unary),
    op!("Atan", 7, Mir, Unary),
    op!("Floor", 1, Mir, Unary),
    op!("Round", 11, Mir, Unary),
    op!("Erf", 9, Mir, Unary),
    op!("Identity", 1, Mir, Unary),
    op!("IsNaN", 9, Custom, Unary),
    op!("Cast", 1, Mir, Shape),
    op!("Transpose", 1, Mir, Shape),
    op!("Reshape", 1, Mir, Shape),
    op!("Unsqueeze", 1, Mir, Shape),
    op!("Squeeze", 1, Mir, Shape),
    op!("Flatten", 1, Mir, Shape),
    op!("Gather", 1, Mir, Shape),
    op!("Concat", 1, Mir, Shape),
    op!("Slice", 1, Mir, Shape),
    op!("Shape", 1, Mir, Shape),
    op!("ConstantOfShape", 1, Mir, Shape),
    op!("Pad", 1, Mir, Shape),
    op!("Expand", 8, Mir, Shape),
    op!("Range", 11, Mir, Shape),
    op!("STFT", 17, Mir, NeuralNet),
    op!("Resize", 10, Mir, Shape),
    op!("Softmax", 1, Mir, NeuralNet),
    op!("LayerNormalization", 1, Mir, NeuralNet),
    op!("SimplifiedLayerNormalization", 1, Mir, NeuralNet),
    op!("SkipSimplifiedLayerNormalization", 1, Mir, NeuralNet),
    op!("GroupQueryAttention", 1, Mir, NeuralNet),
    op!("InstanceNormalization", 1, Mir, NeuralNet),
    op!("BatchNormalization", 1, Mir, NeuralNet),
    op!("Conv", 1, Mir, NeuralNet),
    op!("ConvTranspose", 1, Mir, NeuralNet),
    op!("AveragePool", 1, Mir, NeuralNet),
    op!("MaxPool", 1, Mir, NeuralNet),
    op!("GlobalAveragePool", 1, Mir, NeuralNet),
    op!("Dropout", 1, Mir, NeuralNet),
    op!("LSTM", 7, Mir, NeuralNet),
    op!("GRU", 7, Mir, NeuralNet),
    op!("Equal", 1, Mir, Logic),
    op!("Less", 1, Mir, Logic),
    op!("Greater", 1, Mir, Logic),
    op!("LessOrEqual", 7, Mir, Logic),
    op!("GreaterOrEqual", 7, Mir, Logic),
    op!("Not", 1, Mir, Logic),
    op!("And", 1, Mir, Logic),
    op!("Or", 1, Mir, Logic),
    op!("Where", 1, Mir, Logic),
    op!("Clip", 1, Mir, Logic),
    op!("ReduceMean", 1, Mir, Reduce),
    op!("ReduceSum", 1, Mir, Reduce),
    op!("ReduceMax", 1, Mir, Reduce),
    op!("ReduceMin", 1, Mir, Reduce),
    op!("ReduceProd", 1, Mir, Reduce),
    op!("ReduceL2", 1, Mir, Reduce),
    op!("ReduceL1", 1, Mir, Reduce),
    op!("ReduceSumSquare", 1, Mir, Reduce),
    op!("ReduceLogSum", 1, Mir, Reduce),
    op!("ReduceLogSumExp", 1, Mir, Reduce),
    op!("Gelu", 20, Mir, Unary),
    op!("BiasGelu", 1, Mir, Unary),
    op!("FastGelu", 1, Mir, Unary),
    op!("DynamicQuantizeLinear", 11, Custom, Quant),
    op!("DynamicQuantizeLSTM", 1, Custom, Quant),
    op!("ActCopy", 1, Custom, Quant),
    op!("ScatterND", 11, Mir, Indexing),
    op!("ScatterElements", 11, Mir, Indexing),
    op!("GatherND", 11, Mir, Indexing),
    op!("GatherElements", 1, Mir, Indexing),
    op!("OneHot", 9, Custom, Indexing),
    op!("NonZero", 9, Custom, Indexing),
    op!("CumProd", 11, Custom, Indexing),
    op!("Einsum", 12, Custom, LinearAlgebra),
    op!("TopK", 1, Mir, Indexing),
    op!("ArgMax", 1, Mir, Indexing),
    op!("ArgMin", 1, Mir, Indexing),
    op!("CumSum", 11, Mir, Indexing),
    op!("RandomNormalLike", 1, Mir, Random),
    op!("RandomUniformLike", 1, Mir, Random),
    op!("RandomNormal", 1, Mir, Random),
    op!("RandomUniform", 1, Mir, Random),
    op!("If", 1, ControlFlow, ControlFlow),
    op!("Loop", 1, ControlFlow, ControlFlow),
    op!("Scan", 1, ControlFlow, ControlFlow),
    op!("SplitToSequence", 1, ControlFlow, ControlFlow),
    op!("ConcatFromSequence", 1, ControlFlow, ControlFlow),
    op!("SequenceEmpty", 1, ControlFlow, ControlFlow),
];

pub fn registry_lookup(op: &str) -> Option<&'static OpEntry> {
    OP_REGISTRY.iter().find(|e| e.onnx_op == op)
}

pub fn op_is_registered(op: &str) -> bool {
    registry_lookup(op).is_some()
}

pub fn ops_in_category(category: OpCategory) -> impl Iterator<Item = &'static OpEntry> {
    OP_REGISTRY.iter().filter(move |e| e.category == category)
}

pub fn registry_by_category() -> HashMap<OpCategory, Vec<&'static OpEntry>> {
    let mut out: HashMap<OpCategory, Vec<&'static OpEntry>> = HashMap::new();
    for entry in OP_REGISTRY {
        out.entry(entry.category).or_default().push(entry);
    }
    out
}

pub fn lowered_ops() -> Vec<&'static str> {
    OP_REGISTRY
        .iter()
        .filter(|e| !matches!(e.strategy, LowerStrategy::Rewritten))
        .map(|e| e.onnx_op)
        .collect()
}

pub fn rewritten_ops() -> Vec<&'static str> {
    OP_REGISTRY
        .iter()
        .filter(|e| matches!(e.strategy, LowerStrategy::Rewritten))
        .map(|e| e.onnx_op)
        .collect()
}

pub fn coverage_histogram(ops: &[(&str, usize)]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (op, count) in ops {
        if op_is_registered(op) {
            *m.entry(op.to_string()).or_insert(0) += count;
        }
    }
    m
}

/// Count registered op *instances* grouped by [`OpCategory`].
pub fn bundle_coverage_by_category(
    op_histogram: &[(String, usize)],
) -> HashMap<OpCategory, CategoryCoverage> {
    let mut out: HashMap<OpCategory, CategoryCoverage> = HashMap::new();
    for (op, count) in op_histogram {
        let Some(entry) = registry_lookup(op) else {
            continue;
        };
        out.entry(entry.category).or_default().registered += count;
    }
    out
}

#[derive(Debug, Default, Clone)]
pub struct CategoryCoverage {
    pub registered: usize,
}

impl CategoryCoverage {
    pub fn total(&self) -> usize {
        self.registered
    }
}

pub fn format_registry_dashboard() -> String {
    let mut lines = vec![format!("registered_ops={}", OP_REGISTRY.len())];
    for cat in OpCategory::ALL {
        let entries: Vec<_> = ops_in_category(*cat).collect();
        lines.push(format!("[{}] count={}", cat.name(), entries.len()));
        for entry in entries {
            lines.push(format!(
                "  {} (opset>={}, {:?})",
                entry.onnx_op, entry.since_opset, entry.strategy
            ));
        }
    }
    lines.join("\n")
}

pub fn format_bundle_category_report(op_histogram: &[(String, usize)]) -> String {
    let by_cat = bundle_coverage_by_category(op_histogram);
    let total_registered: usize = by_cat.values().map(|s| s.registered).sum();
    let mut lines = vec![format!("registered_instances={total_registered}")];
    for cat in OpCategory::ALL {
        let Some(stats) = by_cat.get(cat) else {
            continue;
        };
        if stats.registered == 0 {
            continue;
        }
        let n_ops = ops_in_category(*cat).count();
        lines.push(format!(
            "[{}] instances={} registry_ops={n_ops}",
            cat.name(),
            stats.registered,
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_entry_has_category() {
        assert!(OP_REGISTRY.len() >= 80);
        for entry in OP_REGISTRY {
            assert!(
                OpCategory::ALL.contains(&entry.category),
                "{:?}",
                entry.onnx_op
            );
        }
    }

    #[test]
    fn categories_partition_registry() {
        let n: usize = OpCategory::ALL
            .iter()
            .map(|c| ops_in_category(*c).count())
            .sum();
        assert_eq!(n, OP_REGISTRY.len());
    }
}
