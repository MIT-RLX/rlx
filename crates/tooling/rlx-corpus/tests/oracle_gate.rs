// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Numerical acceptance against an authority that is not rlx.
//!
//! Every case the independent f64 evaluator can cover is checked against it.
//! Cases it cannot cover are *counted and named* rather than quietly passing
//! on a self-comparison — the point of the exercise is that "how many results
//! were actually validated" becomes a number you can read.

use rlx_corpus::oracle::{Authority, REL_TOL, tolerance_for, validate, validate_with_packed};
use rlx_ir::{Graph, Op};
use rlx_runtime::{Device, Session};

/// Deterministic, non-degenerate values. Two properties matter: constant
/// inputs would let a kernel that ignores its indices agree with the
/// reference, and values near zero make the `Div` case blow up to infinity —
/// which the oracle then (correctly) refuses to score, so the case would go
/// unvalidated. Magnitude is held in [0.1, 1.0] with both signs present.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mag = 0.1 + (((i * 37 + seed * 101) % 90) as f32) * 0.01;
            if (i + seed).is_multiple_of(2) {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

/// Deterministic bytes for a packed quantized weight.
///
/// Any 34-byte string is a valid `block_q8_0` — an f16 scale and 32 int8
/// quants — so no encoder is needed. The scale byte pair is kept in a modest
/// exponent range so a block's values land near 1.0 rather than in f16
/// subnormals, where the whole row would decode to ~0 and the non-zero check
/// below could not tell a decode from a Nop.
fn bytes_fill(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            if i % 34 == 1 {
                // High byte of the f16 scale: exponent ~2^-3..2^-1, sign 0.
                0x2c | ((i / 34 + seed) % 3) as u8
            } else if i % 34 == 0 {
                ((i * 31 + seed * 7) % 256) as u8
            } else {
                (((i * 37 + seed * 101) % 251) as i32 - 125) as i8 as u8
            }
        })
        .collect()
}

fn elems(g: &Graph, name: &str) -> Option<usize> {
    g.nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name: nm } | Op::Param { name: nm } if nm == name))
        .and_then(|n| {
            n.shape
                .dims()
                .iter()
                .map(|d| match d {
                    rlx_ir::Dim::Static(k) => Some(*k),
                    rlx_ir::Dim::Dynamic(_) => None,
                })
                .collect::<Option<Vec<usize>>>()
        })
        .map(|d| d.iter().product())
}

#[test]
fn corpus_results_match_an_independent_oracle() {
    let mut validated = 0usize;
    let mut unvalidated: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in rlx_corpus::cases() {
        let mut inputs: Vec<(&str, Vec<f32>)> = Vec::new();
        let mut params: Vec<(&str, Vec<f32>)> = Vec::new();
        let mut packed: Vec<(&str, Vec<u8>)> = Vec::new();
        let mut ok = true;
        for (i, node) in case.graph.nodes().iter().enumerate() {
            let is_u8 = node.shape.dtype() == rlx_ir::DType::U8;
            match &node.op {
                Op::Input { name } => match elems(&case.graph, name) {
                    Some(n) => inputs.push((Box::leak(name.clone().into_boxed_str()), fill(n, i))),
                    None => ok = false,
                },
                // A `U8` param is a packed quantized weight. It goes in as raw
                // bytes: `set_param` takes `&[f32]` and would write 4x the slot.
                Op::Param { name } if is_u8 => match elems(&case.graph, name) {
                    Some(n) => packed.push((
                        Box::leak(name.clone().into_boxed_str()),
                        bytes_fill(n, i + 13),
                    )),
                    None => ok = false,
                },
                Op::Param { name } => match elems(&case.graph, name) {
                    Some(n) => {
                        params.push((Box::leak(name.clone().into_boxed_str()), fill(n, i + 7)))
                    }
                    None => ok = false,
                },
                _ => {}
            }
        }
        if !ok {
            unvalidated.push(format!("{}::{} (dynamic shape)", case.family, case.name));
            continue;
        }

        let mut exe = Session::new(Device::Cpu).compile(case.graph.clone());
        for (n, v) in &params {
            exe.set_param(n, v);
        }
        for (n, v) in &packed {
            exe.set_param_typed(n, v, rlx_ir::DType::U8);
        }
        let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let actual = exe.run(&feed);
        let actual = &actual[0];

        let tol = tolerance_for(&case.graph);
        let r = validate_with_packed(&case.graph, &inputs, &params, &packed, actual);
        match (r.authority, r.max_rel_err) {
            (a, Some(err)) if a.is_independent() => {
                if err > tol {
                    failures.push(format!(
                        "{}::{} max rel {err:.2e} > {tol:.1e} vs {}",
                        case.family,
                        case.name,
                        a.label()
                    ));
                } else {
                    validated += 1;
                }
            }
            (a, _) => unvalidated.push(format!(
                "{}::{} [{}] {}",
                case.family,
                case.name,
                a.label(),
                r.detail
            )),
        }
    }

    eprintln!(
        "oracle: {validated} case(s) validated against an INDEPENDENT reference; \
         {} case(s) had no independent authority:",
        unvalidated.len()
    );
    for u in &unvalidated {
        eprintln!("   unvalidated: {u}");
    }

    assert!(
        failures.is_empty(),
        "numerical disagreement:\n  {}",
        failures.join("\n  ")
    );
    // A run where nothing reached an independent authority would be a
    // self-comparison wearing a gate's clothes.
    assert!(validated > 0, "no case reached an independent oracle");
}

#[test]
fn self_comparison_is_not_counted_as_validation() {
    // The load-bearing distinction. If `SelfBackend` ever reports as
    // independent, every "validated" number above becomes meaningless.
    assert!(!Authority::SelfBackend.is_independent());
    assert!(Authority::Independent.is_independent());
    assert!(Authority::External("onnxruntime").is_independent());
    assert!(Authority::SelfBackend.label().contains("NOT validation"));
}

#[test]
fn the_oracle_detects_a_wrong_answer() {
    // A checker that only ever passes is indistinguishable from one that
    // inspects nothing. Perturb one element and require it to be caught.
    let case = rlx_corpus::cases()
        .into_iter()
        .find(|c| c.family == "matmul" && c.name == "prefill")
        .expect("matmul/prefill present");
    let mut inputs: Vec<(&str, Vec<f32>)> = Vec::new();
    let mut params: Vec<(&str, Vec<f32>)> = Vec::new();
    for (i, node) in case.graph.nodes().iter().enumerate() {
        match &node.op {
            Op::Input { name } => inputs.push((
                Box::leak(name.clone().into_boxed_str()),
                fill(elems(&case.graph, name).unwrap(), i),
            )),
            Op::Param { name } => params.push((
                Box::leak(name.clone().into_boxed_str()),
                fill(elems(&case.graph, name).unwrap(), i + 7),
            )),
            _ => {}
        }
    }
    let mut exe = Session::new(Device::Cpu).compile(case.graph.clone());
    for (n, v) in &params {
        exe.set_param(n, v);
    }
    let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let mut actual = exe.run(&feed)[0].clone();

    let tol = tolerance_for(&case.graph);
    let clean = validate(&case.graph, &inputs, &params, &actual);
    assert!(
        clean.max_rel_err.is_some_and(|e| e <= tol),
        "baseline should agree: {clean:?}"
    );

    actual[0] += 1.0;
    let dirty = validate(&case.graph, &inputs, &params, &actual);
    assert!(
        dirty.max_rel_err.is_some_and(|e| e > tol),
        "a corrupted element must be caught: {dirty:?}"
    );

    // A single matmul must stay near the base tolerance: the derived bound
    // scales with depth, and this pins that it does not quietly scale for a
    // shallow graph too.
    assert!(
        tol <= REL_TOL * 4.0,
        "one matmul should not need a loose bound, got {tol:.1e}"
    );
}
