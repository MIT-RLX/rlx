// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Verify the MIL program before handing it to Apple's compiler.**
//!
//! CoreML is a *delegating* backend: rlx does not author its kernels, Apple's
//! compiler does. So `rlx_ir::kernel_schedule` has nothing to attach to here —
//! there is no schedule of rlx's to verify. That is a real architectural
//! exclusion, not a gap to fill with a schedule emitter.
//!
//! But CAKE's actual commitment is not "own the schedule". It is *"the compiler
//! returns localized correctness diagnostics rather than a pass/fail bit"*
//! (§3.1), and for a delegating backend the thing rlx **does** author is the IR
//! it emits. MIL is a typed IR with legality rules. Those rules can be checked
//! before the model is handed downstream, and the same contract applies:
//! findings that name the offending operation, not an abort.
//!
//! # Why this is worth a gate rather than a bug fix
//!
//! `scripts/mil-reshape-check.py` already scans a *compiled* `model.mil` for
//! reshapes whose element counts disagree, and its docstring records why:
//!
//! > rlx-ir does not verify element counts on a fully-concrete reshape target,
//! > so a mis-built graph reaches the backend intact: the CPU backend
//! > reinterprets the buffer and the model silently returns garbage, while
//! > Metal/CoreML hand the same reshape to a verifying compiler and `abort()`
//! > the process.
//!
//! `abort()` is the worst possible diagnostic — it takes the process down with
//! no finding, no operation name and no repair target. And the script can only
//! run *after* a successful compile, on a `.mlmodelc` that exists in a temp
//! directory, which means it never runs in CI and cannot report the case where
//! the compile is what died.
//!
//! This module moves that check before the compiler and generalizes it: the
//! program is walked in memory, and every finding names the operation.
//!
//! # Scope of the analysis, stated
//!
//! Deliberately narrow, and narrow in a way this file admits rather than
//! implies. It checks *structural* properties that are cheap and decidable from
//! the emitted proto: value definition before use, name collisions, concrete
//! reshape element counts, and I/O rank. It does **not** typecheck MIL, does
//! not know Apple's opset, does not model dynamic shapes, and does not predict
//! whether the ANE will accept the program. A clean report means the emitted IR
//! is self-consistent, not that it will compile — the same disclaimer
//! `verify_kernel_schedule` carries.

use std::collections::{HashMap, HashSet};

use crate::proto;

/// A defect in an emitted MIL program.
///
/// Every variant names the operation it is about, because a finding that cannot
/// be tied back to a program point is only marginally better than the `abort()`
/// it replaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MilFinding {
    /// A `reshape` whose concrete target has a different element count than its
    /// input. **The motivating case.** Nothing upstream rejects it, the CPU
    /// backend reinterprets the buffer and returns garbage, and CoreML's
    /// compiler `abort()`s.
    ReshapeElementCountMismatch {
        op: String,
        from: Vec<i64>,
        to: Vec<i64>,
    },
    /// An operation reads a value no earlier operation produced and that is not
    /// a function input. In MIL this is a malformed program; the failure
    /// downstream is a compiler diagnostic against generated text.
    UndefinedValue { op: String, value: String },
    /// Two operations produce the same value name. MIL is SSA-shaped, so the
    /// second silently shadows the first and every later read of that name
    /// resolves to the wrong tensor — wrong numbers, no crash.
    DuplicateDefinition { value: String, first: String },
    /// A model input or output with rank 0. CoreML I/O features are
    /// MLMultiArrays and require rank >= 1; `io_feature_shape` maps scalars to
    /// `[1]`, so a rank-0 survivor here means that mapping was bypassed.
    RankZeroIo { name: String, is_input: bool },
    /// An operation with no outputs. Harmless in isolation, but it means the
    /// lowering emitted work nothing consumes — usually a dropped result.
    OperationWithNoOutputs { op: String },
}

impl std::fmt::Display for MilFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReshapeElementCountMismatch { op, from, to } => write!(
                f,
                "`{op}`: reshape {from:?} ({}) -> {to:?} ({}) changes the element count",
                from.iter().product::<i64>(),
                to.iter().product::<i64>()
            ),
            Self::UndefinedValue { op, value } => {
                write!(f, "`{op}` reads `{value}`, which nothing defines")
            }
            Self::DuplicateDefinition { value, first } => write!(
                f,
                "`{value}` is defined twice (first by `{first}`); later reads take the second"
            ),
            Self::RankZeroIo { name, is_input } => write!(
                f,
                "model {} `{name}` has rank 0; CoreML I/O requires rank >= 1",
                if *is_input { "input" } else { "output" }
            ),
            Self::OperationWithNoOutputs { op } => {
                write!(f, "`{op}` produces no outputs")
            }
        }
    }
}

/// Concrete dimensions of a `NamedValueType`, or `None` when any dim is
/// symbolic/unknown.
///
/// Returning `None` rather than guessing is load-bearing: a flexible-shape
/// model legitimately has unknown dims, and reporting a "mismatch" against a
/// dimension nobody stated would be a false positive that trains people to
/// ignore the gate.
fn concrete_dims(t: &proto::NamedValueType) -> Option<Vec<i64>> {
    use proto::value_type::Type as VT;
    let vt = t.r#type.as_ref()?;
    // `ValueType` currently has one variant, so a `let ... else` here is
    // irrefutable and warns. Matching keeps this correct if a second variant
    // (list, tuple, dictionary) is ever generated.
    #[allow(irrefutable_let_patterns)]
    let VT::TensorType(tt) = vt.r#type.as_ref()? else {
        return None;
    };
    let mut dims = Vec::with_capacity(tt.dimensions.len());
    for d in &tt.dimensions {
        use proto::dimension::Dimension as D;
        match d.dimension.as_ref()? {
            D::Constant(c) => dims.push(c.size as i64),
            D::Unknown(_) => return None,
        }
    }
    Some(dims)
}

/// Every value name an operation reads.
fn read_names(op: &proto::Operation) -> Vec<String> {
    let mut out = Vec::new();
    for arg in op.inputs.values() {
        for b in &arg.arguments {
            use proto::argument::binding::Binding as B;
            if let Some(B::Name(n)) = b.binding.as_ref() {
                out.push(n.clone());
            }
        }
    }
    out
}

/// Walk `program` and report every finding.
///
/// Returns an empty vector when nothing is wrong *within the modeled domain*
/// (see the module docs). Order is deterministic: program order, then the
/// order the checks are written, so a diff of two reports is readable.
pub fn verify_program(model: &proto::Model) -> Vec<MilFinding> {
    let mut findings = Vec::new();
    let Some(proto::model::Type::MlProgram(program)) = model.r#type.as_ref().cloned() else {
        // Not an ML Program (a pipeline or a neural network). Nothing this
        // module models — reported as "no findings" would be a lie by omission,
        // but there is also nothing to name, so an empty report is honest here.
        return findings;
    };

    for func in program.functions.values() {
        let Some(block) = func
            .block_specializations
            .values()
            .next()
            .or(func.block_specializations.get(&func.opset))
        else {
            continue;
        };

        // Function inputs are defined on entry.
        let mut defined: HashMap<String, String> = HashMap::new();
        let mut input_names: HashSet<String> = HashSet::new();
        for i in &func.inputs {
            defined.insert(i.name.clone(), "<function input>".to_string());
            input_names.insert(i.name.clone());
            if concrete_dims(i).map(|d| d.is_empty()).unwrap_or(false) {
                findings.push(MilFinding::RankZeroIo {
                    name: i.name.clone(),
                    is_input: true,
                });
            }
        }

        // Shapes of values as they are produced, for the reshape check.
        //
        // Seeded with the FUNCTION INPUTS, not just op outputs. Omitting them
        // silently skipped the check for any reshape reading a model input
        // directly — which is the shape of the motivating defect, so the gate
        // would have looked green on the case it was written for.
        let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
        for i in &func.inputs {
            if let Some(d) = concrete_dims(i) {
                shapes.insert(i.name.clone(), d);
            }
        }

        for op in &block.operations {
            let label = op
                .outputs
                .first()
                .map(|o| o.name.clone())
                .unwrap_or_else(|| format!("<{}>", op.r#type));

            for name in read_names(op) {
                if !defined.contains_key(&name) {
                    findings.push(MilFinding::UndefinedValue {
                        op: label.clone(),
                        value: name,
                    });
                }
            }

            // The reshape rule, moved ahead of the compiler.
            if op.r#type == "reshape"
                && let Some(out) = op.outputs.first()
                && let Some(to) = concrete_dims(out)
                && let Some(src) = op
                    .inputs
                    .get("x")
                    .and_then(|a| a.arguments.first())
                    .and_then(|b| match b.binding.as_ref() {
                        Some(proto::argument::binding::Binding::Name(n)) => Some(n.clone()),
                        _ => None,
                    })
                && let Some(from) = shapes.get(&src)
                && from.iter().product::<i64>() != to.iter().product::<i64>()
            {
                findings.push(MilFinding::ReshapeElementCountMismatch {
                    op: label.clone(),
                    from: from.clone(),
                    to,
                });
            }

            if op.outputs.is_empty() {
                findings.push(MilFinding::OperationWithNoOutputs { op: label.clone() });
            }
            for out in &op.outputs {
                if let Some(first) = defined.get(&out.name) {
                    findings.push(MilFinding::DuplicateDefinition {
                        value: out.name.clone(),
                        first: first.clone(),
                    });
                } else {
                    defined.insert(out.name.clone(), label.clone());
                }
                if let Some(d) = concrete_dims(out) {
                    shapes.insert(out.name.clone(), d);
                }
            }
        }

        for o in &block.outputs {
            if !defined.contains_key(o) {
                findings.push(MilFinding::UndefinedValue {
                    op: "<function output>".into(),
                    value: o.clone(),
                });
            }
        }
    }

    findings
}

/// Render a report, or `None` when there is nothing to say.
pub fn report(model: &proto::Model) -> Option<String> {
    let findings = verify_program(model);
    if findings.is_empty() {
        return None;
    }
    let mut out = format!("rlx-coreml: {} MIL finding(s)\n", findings.len());
    for f in &findings {
        out.push_str(&format!("  {f}\n"));
    }
    out.push_str(
        "  (structural checks only: value definition, name collisions, concrete reshape\n\
         \x20  element counts, I/O rank. Not a typecheck and not an opset check.)\n",
    );
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto;

    fn tensor(name: &str, dims: &[i64]) -> proto::NamedValueType {
        proto::NamedValueType {
            name: name.to_string(),
            r#type: Some(proto::ValueType {
                r#type: Some(proto::value_type::Type::TensorType(proto::TensorType {
                    data_type: proto::DataType::Float32 as i32,
                    rank: dims.len() as i64,
                    dimensions: dims
                        .iter()
                        .map(|d| proto::Dimension {
                            dimension: Some(proto::dimension::Dimension::Constant(
                                proto::dimension::ConstantDimension { size: *d as u64 },
                            )),
                        })
                        .collect(),
                    attributes: Default::default(),
                })),
            }),
        }
    }

    fn name_arg(n: &str) -> proto::Argument {
        proto::Argument {
            arguments: vec![proto::argument::Binding {
                binding: Some(proto::argument::binding::Binding::Name(n.to_string())),
            }],
        }
    }

    fn op(ty: &str, inputs: &[(&str, &str)], out: proto::NamedValueType) -> proto::Operation {
        proto::Operation {
            r#type: ty.to_string(),
            inputs: inputs
                .iter()
                .map(|(k, v)| (k.to_string(), name_arg(v)))
                .collect(),
            outputs: vec![out],
            blocks: vec![],
            attributes: Default::default(),
        }
    }

    fn model_of(inputs: Vec<proto::NamedValueType>, ops: Vec<proto::Operation>) -> proto::Model {
        let outputs: Vec<String> = ops
            .last()
            .map(|o| o.outputs.iter().map(|x| x.name.clone()).collect())
            .unwrap_or_default();
        let block = proto::Block {
            inputs: vec![],
            outputs,
            operations: ops,
            attributes: Default::default(),
        };
        let func = proto::Function {
            inputs,
            opset: "CoreML6".into(),
            block_specializations: [("CoreML6".to_string(), block)].into_iter().collect(),
            attributes: Default::default(),
        };
        proto::Model {
            specification_version: 7,
            r#type: Some(proto::model::Type::MlProgram(proto::Program {
                functions: [("main".to_string(), func)].into_iter().collect(),
                doc_string: String::new(),
                version: 1,
                attributes: Default::default(),
            })),
            ..Default::default()
        }
    }

    /// THE motivating case: the defect `scripts/mil-reshape-check.py` exists to
    /// find, now caught before the compiler instead of after it.
    #[test]
    fn a_reshape_that_changes_the_element_count_is_caught() {
        let m = model_of(
            vec![tensor("x", &[2, 6])],
            vec![op("reshape", &[("x", "x")], tensor("y", &[3, 5]))],
        );
        let f = verify_program(&m);
        assert!(
            matches!(
                f.as_slice(),
                [MilFinding::ReshapeElementCountMismatch { .. }]
            ),
            "expected a reshape mismatch, got {f:?}"
        );
        // And the message names the op and both shapes — a repair target.
        let msg = f[0].to_string();
        assert!(
            msg.contains("`y`") && msg.contains("12") && msg.contains("15"),
            "{msg}"
        );
    }

    #[test]
    fn a_legal_reshape_is_accepted() {
        let m = model_of(
            vec![tensor("x", &[2, 6])],
            vec![op("reshape", &[("x", "x")], tensor("y", &[3, 4]))],
        );
        assert!(verify_program(&m).is_empty());
    }

    #[test]
    fn reading_an_undefined_value_is_caught() {
        let m = model_of(
            vec![tensor("x", &[4])],
            vec![op("relu", &[("x", "nope")], tensor("y", &[4]))],
        );
        assert!(
            verify_program(&m)
                .iter()
                .any(|f| matches!(f, MilFinding::UndefinedValue { .. })),
        );
    }

    /// Shadowing is the dangerous one: no crash, wrong tensor, wrong numbers.
    #[test]
    fn defining_the_same_value_twice_is_caught() {
        let m = model_of(
            vec![tensor("x", &[4])],
            vec![
                op("relu", &[("x", "x")], tensor("y", &[4])),
                op("relu", &[("x", "y")], tensor("y", &[4])),
            ],
        );
        assert!(
            verify_program(&m)
                .iter()
                .any(|f| matches!(f, MilFinding::DuplicateDefinition { .. })),
        );
    }

    #[test]
    fn a_rank_zero_model_input_is_caught() {
        let m = model_of(
            vec![tensor("s", &[])],
            vec![op("identity", &[("x", "s")], tensor("y", &[1]))],
        );
        assert!(
            verify_program(&m)
                .iter()
                .any(|f| matches!(f, MilFinding::RankZeroIo { is_input: true, .. })),
        );
    }

    /// A flexible-shape model has genuinely unknown dims. Reporting a mismatch
    /// against a dimension nobody stated would be a false positive, and a gate
    /// that cries wolf gets switched off.
    #[test]
    fn an_unknown_dimension_is_not_a_mismatch() {
        let mut src = tensor("x", &[2, 6]);
        if let Some(proto::value_type::Type::TensorType(tt)) =
            src.r#type.as_mut().and_then(|v| v.r#type.as_mut())
        {
            tt.dimensions[0].dimension = Some(proto::dimension::Dimension::Unknown(
                proto::dimension::UnknownDimension { variadic: false },
            ));
        }
        let m = model_of(
            vec![src],
            vec![op("reshape", &[("x", "x")], tensor("y", &[3, 5]))],
        );
        assert!(
            verify_program(&m).is_empty(),
            "a symbolic input dimension must not produce a shape finding"
        );
    }

    #[test]
    fn a_clean_program_reports_nothing() {
        let m = model_of(
            vec![tensor("x", &[2, 6])],
            vec![
                op("relu", &[("x", "x")], tensor("a", &[2, 6])),
                op("reshape", &[("x", "a")], tensor("b", &[12])),
            ],
        );
        assert_eq!(verify_program(&m), vec![]);
        assert!(report(&m).is_none());
    }
}
