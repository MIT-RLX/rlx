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

//! Op coverage report — discovers which Op variants are implemented on
//! each backend by attempting a minimal forward pass per op.
//!
//! Status semantics (per backend):
//!   ✓ ok         — output matches CPU reference within tolerance
//!   ✗ MISSING    — output is all (near-)zeros while CPU produced non-zero
//!                  (the silent Thunk::Nop pattern that bit us with Concat)
//!   ! MISMATCH   — both backends produced output, but they disagree
//!                  beyond tolerance (real numerical bug)
//!
//! cargo run -p rlx-runtime --example op_coverage --features metal
//!
//! The tool exits 0 always (it's a report), prints a tally of MISSING
//! and MISMATCH at the end. Use the parity_harness if you want CI gating.

// Shape-noting `1 *` factors (`1 * N * H * W`) are intentional in the
// per-op test inputs below — they document the rank/layout of each
// tensor in line, even though the multiplication is mathematically a
// no-op.
#![allow(clippy::identity_op)]

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    // ── helper: build, run on both, classify ────────────────────────
    #[derive(Debug)]
    // CpuMissing / BothMissing are reserved for future per-row diagnostics
    // (e.g. when a check on Metal can't run because the CPU reference
    // graph itself failed to compile). Currently every check builds CPU
    // first, so the constructors aren't reached, but the variants are
    // part of the table's UI contract.
    #[allow(dead_code)]
    enum Status {
        Ok,
        Missing,
        Mismatch(f32),
        CpuMissing,
        BothMissing,
        BuildOnly(String),
    }
    impl Status {
        fn glyph(&self) -> &'static str {
            match self {
                Status::Ok => "✓",
                Status::Missing => "✗",
                Status::Mismatch(_) => "!",
                Status::CpuMissing => "✗",
                Status::BothMissing => "✗",
                Status::BuildOnly(_) => "?",
            }
        }
        fn label(&self) -> String {
            match self {
                Status::Ok => "ok".into(),
                Status::Missing => "MISSING".into(),
                Status::Mismatch(e) => format!("MISMATCH ({e:.2e})"),
                Status::CpuMissing => "CPU MISSING".into(),
                Status::BothMissing => "BOTH MISSING".into(),
                Status::BuildOnly(s) => format!("error: {s}"),
            }
        }
    }
    type Row = (String, Status, Status);

    let f = DType::F32;
    let mut rows: Vec<Row> = Vec::new();

    let near_zero = |out: &[f32]| -> bool { out.iter().all(|v| v.abs() < 1e-6) };

    // Classify a backend's output against an optional reference. Without a
    // reference (`reference = None`) we can't tell silent-passthrough bugs
    // from real implementations — but we can still detect the all-zeros
    // pattern, which is the most common silent-Nop signature.
    let classify = |out: &[f32], reference: Option<&[f32]>, tol: f32| -> Status {
        // If we have a reference, trust it absolutely — output matching the
        // reference is correct even when both are zero (e.g. Compare::Gt with
        // x ≤ y for all elements). Only flag MISSING via the all-zeros
        // heuristic when there's no reference to disambiguate.
        if let Some(refr) = reference {
            let max_err = out
                .iter()
                .zip(refr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if max_err < tol {
                Status::Ok
            } else if near_zero(out) && !near_zero(refr) {
                Status::Missing
            } else {
                Status::Mismatch(max_err)
            }
        } else {
            if near_zero(out) {
                Status::Missing
            } else {
                Status::Ok
            }
        }
    };

    // Run a test on both backends.
    //   `reference == Some(expected)`: compare each backend independently
    //     against the hand-rolled reference (catches the case where both
    //     backends silently no-op in the same way, like ActivationInPlace
    //     passing input through when the kernel arm is missing).
    //   `reference == None`: trust CPU as ground truth and compare Metal
    //     against it. CPU passthrough/Nop on its own is still detected by
    //     the all-zeros heuristic in `classify`.
    let run_check = |name: String,
                     build: &dyn Fn() -> Graph,
                     params: &[(&str, Vec<f32>)],
                     inputs: &[(&str, Vec<f32>)],
                     reference: Option<Vec<f32>>,
                     tol: f32,
                     rows: &mut Vec<Row>| {
        let cpu_session = Session::new(Device::Cpu);
        let mut cpu = cpu_session.compile(build());
        for (n, d) in params {
            cpu.set_param(n, d);
        }
        let cpu_inputs: Vec<(&str, &[f32])> =
            inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let cpu_out = cpu.run(&cpu_inputs);

        let metal_session = Session::new(Device::Metal);
        let mut metal = metal_session.compile(build());
        for (n, d) in params {
            metal.set_param(n, d);
        }
        let metal_out = metal.run(&cpu_inputs);

        if cpu_out.is_empty()
            || metal_out.is_empty()
            || cpu_out[0].is_empty()
            || metal_out[0].is_empty()
        {
            rows.push((
                name,
                Status::BuildOnly("empty output".into()),
                Status::BuildOnly("empty output".into()),
            ));
            return;
        }

        let cpu_status;
        let metal_status;
        match reference {
            Some(refr) => {
                cpu_status = classify(&cpu_out[0], Some(&refr), tol);
                metal_status = classify(&metal_out[0], Some(&refr), tol);
            }
            None => {
                // No independent reference — fall back to all-zeros heuristic
                // for CPU, then compare Metal against CPU. If CPU itself looks
                // missing, we can't claim Metal is implemented just because
                // it produced the same zeros — both are likely silently
                // no-op'd, so propagate the missing status to both.
                let cpu_zero = near_zero(&cpu_out[0]);
                cpu_status = if cpu_zero {
                    Status::Missing
                } else {
                    Status::Ok
                };
                metal_status = if cpu_zero {
                    if near_zero(&metal_out[0]) {
                        Status::Missing
                    } else {
                        Status::Ok
                    }
                } else {
                    classify(&metal_out[0], Some(&cpu_out[0]), tol)
                };
            }
        }
        rows.push((name, cpu_status, metal_status));
    };
    // Wrapper for the common case (no independent reference, trust CPU).
    let check = |name: String,
                 build: &dyn Fn() -> Graph,
                 params: &[(&str, Vec<f32>)],
                 inputs: &[(&str, Vec<f32>)],
                 tol: f32,
                 rows: &mut Vec<Row>| {
        run_check(name, build, params, inputs, None, tol, rows);
    };

    let det = |n: usize, mul: usize, modulo: usize, offset: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * mul + 7) % modulo) as f32 / modulo as f32 + offset)
            .collect()
    };

    // ── Activations: each variant individually ──────────────────────
    // Hand-rolled scalar reference. Catching silent-no-op cases (where both
    // backends pass input through unchanged) requires comparing against this,
    // not against each other.
    // Abramowitz & Stegun 7.1.26 erf approximation — matches rlx-cpu's
    // par_gelu_inplace to ~1e-7. Used as the Gelu reference here so silent
    // no-ops are catchable independently of either backend.
    //
    // The literals carry f64-shaped precision on purpose; the published
    // A&S table prescribes them this way and rounding to f32 ulps would
    // shift the round-trip past the test tolerance.
    #[allow(clippy::excessive_precision)]
    fn erf_approx(x: f32) -> f32 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let xa = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * xa);
        let y = t
            * (0.254_829_59
                + t * (-0.284_496_74
                    + t * (1.421_413_75 + t * (-1.453_152_03 + t * 1.061_405_43))));
        sign * (1.0 - y * (-xa * xa).exp())
    }
    fn ref_activation(act: Activation, x: f32) -> f32 {
        match act {
            Activation::Gelu | Activation::GeluApprox => {
                let cdf = 0.5 * (1.0 + erf_approx(x / 2.0_f32.sqrt()));
                x * cdf
            }
            Activation::Silu => x / (1.0 + (-x).exp()),
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Exp => x.exp(),
            Activation::Log => x.ln(),
            Activation::Sqrt => x.sqrt(),
            Activation::Rsqrt => 1.0 / x.sqrt(),
            Activation::Neg => -x,
            Activation::Abs => x.abs(),
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            Activation::Round => x.round(),
        }
    }
    let act_variants = [
        ("gelu", Activation::Gelu),
        ("gelu_approx", Activation::GeluApprox),
        ("silu", Activation::Silu),
        ("relu", Activation::Relu),
        ("sigmoid", Activation::Sigmoid),
        ("tanh", Activation::Tanh),
        ("exp", Activation::Exp),
        ("log", Activation::Log),
        ("sqrt", Activation::Sqrt),
        ("rsqrt", Activation::Rsqrt),
        ("neg", Activation::Neg),
        ("abs", Activation::Abs),
        ("sin", Activation::Sin),
        ("cos", Activation::Cos),
        ("round", Activation::Round),
    ];
    for (name, act) in &act_variants {
        let act = *act;
        let needs_positive = matches!(act, Activation::Log | Activation::Sqrt | Activation::Rsqrt);
        let offset = if needs_positive { 0.5 } else { 0.1 };
        let x_data = det(32, 13, 17, offset);
        let reference: Vec<f32> = x_data.iter().map(|&v| ref_activation(act, v)).collect();
        let label = format!("Activation::{name}");
        run_check(
            label,
            &|| {
                let mut g = Graph::new("act");
                let x = g.input("x", Shape::new(&[4, 8], f));
                let y = g.activation(act, x, Shape::new(&[4, 8], f));
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("x", x_data)],
            Some(reference),
            1e-3,
            &mut rows,
        );
    }

    // ── BinaryOps: each variant individually ────────────────────────
    let bin_variants = [
        ("Add", BinaryOp::Add),
        ("Sub", BinaryOp::Sub),
        ("Mul", BinaryOp::Mul),
        ("Div", BinaryOp::Div),
        ("Max", BinaryOp::Max),
        ("Min", BinaryOp::Min),
        ("Pow", BinaryOp::Pow),
    ];
    for (name, op) in &bin_variants {
        let op = *op;
        let label = format!("Binary::{name}");
        check(
            label,
            &|| {
                let mut g = Graph::new("bin");
                let x = g.input("x", Shape::new(&[4, 8], f));
                let y = g.input("y", Shape::new(&[4, 8], f));
                let z = g.binary(op, x, y, Shape::new(&[4, 8], f));
                g.set_outputs(vec![z]);
                g
            },
            &[],
            &[("x", det(32, 7, 11, 0.5)), ("y", det(32, 13, 19, 0.3))],
            1e-4,
            &mut rows,
        );
    }

    // ── CmpOps: each variant ────────────────────────────────────────
    let cmp_variants = [
        ("Eq", CmpOp::Eq),
        ("Ne", CmpOp::Ne),
        ("Lt", CmpOp::Lt),
        ("Le", CmpOp::Le),
        ("Gt", CmpOp::Gt),
        ("Ge", CmpOp::Ge),
    ];
    for (name, op) in &cmp_variants {
        let op = *op;
        let label = format!("Compare::{name}");
        // Hand-rolled reference so we can detect both correct-but-zero
        // cases (e.g. Ne with identical inputs) and silent passthrough.
        // Mix in non-identical inputs so each comparison has some true and
        // some false outcomes — the all-zero heuristic alone would falsely
        // flag a correct comparison whose answer happens to be all-false.
        let xv = det(32, 7, 11, 0.5);
        let yv: Vec<f32> = xv
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { v } else { v + 0.1 })
            .collect();
        let reference: Vec<f32> = xv
            .iter()
            .zip(yv.iter())
            .map(|(&x, &y)| {
                let r = match op {
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                };
                if r { 1.0 } else { 0.0 }
            })
            .collect();
        run_check(
            label,
            &|| {
                let mut g = Graph::new("cmp");
                let x = g.input("x", Shape::new(&[4, 8], f));
                let y = g.input("y", Shape::new(&[4, 8], f));
                let z = g.add_node(Op::Compare(op), vec![x, y], Shape::new(&[4, 8], DType::F32));
                g.set_outputs(vec![z]);
                g
            },
            &[],
            &[("x", xv), ("y", yv)],
            Some(reference),
            1e-6,
            &mut rows,
        );
    }

    // ── Where ────────────────────────────────────────────────────────
    check(
        "Where".into(),
        &|| {
            let mut g = Graph::new("where");
            let cond = g.input("cond", Shape::new(&[4, 8], f));
            let a = g.input("a", Shape::new(&[4, 8], f));
            let b = g.input("b", Shape::new(&[4, 8], f));
            let z = g.add_node(Op::Where, vec![cond, a, b], Shape::new(&[4, 8], f));
            g.set_outputs(vec![z]);
            g
        },
        &[],
        &[
            (
                "cond",
                (0..32)
                    .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
                    .collect(),
            ),
            ("a", det(32, 7, 11, 0.5)),
            ("b", det(32, 13, 19, 0.7)),
        ],
        1e-4,
        &mut rows,
    );

    // ── Linear algebra ──────────────────────────────────────────────
    check(
        "MatMul".into(),
        &|| {
            let mut g = Graph::new("mm");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let w = g.param("w", Shape::new(&[8, 4], f));
            let y = g.matmul(x, w, Shape::new(&[6, 4], f));
            g.set_outputs(vec![y]);
            g
        },
        &[("w", det(32, 17, 31, 0.1))],
        &[("x", det(48, 13, 23, 0.1))],
        1e-4,
        &mut rows,
    );

    // DotGeneral — currently unimplemented in both backends.
    check(
        "DotGeneral (basic)".into(),
        &|| {
            let mut g = Graph::new("dg");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let w = g.param("w", Shape::new(&[8, 4], f));
            let y = g.add_node(
                Op::DotGeneral {
                    lhs_contracting: vec![1],
                    rhs_contracting: vec![0],
                    lhs_batch: vec![],
                    rhs_batch: vec![],
                },
                vec![x, w],
                Shape::new(&[6, 4], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("w", det(32, 17, 31, 0.1))],
        &[("x", det(48, 13, 23, 0.1))],
        1e-4,
        &mut rows,
    );

    // ── Normalization ───────────────────────────────────────────────
    check(
        "LayerNorm".into(),
        &|| {
            let mut g = Graph::new("ln");
            let x = g.input("x", Shape::new(&[3, 16], f));
            let gamma = g.param("g", Shape::new(&[16], f));
            let beta = g.param("b", Shape::new(&[16], f));
            let y = g.ln(x, gamma, beta, 1e-5);
            g.set_outputs(vec![y]);
            g
        },
        &[("g", vec![1.0; 16]), ("b", vec![0.0; 16])],
        &[("x", det(48, 13, 23, 0.5))],
        1e-4,
        &mut rows,
    );

    check(
        "RmsNorm".into(),
        &|| {
            let mut g = Graph::new("rms");
            let x = g.input("x", Shape::new(&[3, 16], f));
            let gamma = g.param("g", Shape::new(&[16], f));
            let beta = g.param("b", Shape::new(&[16], f));
            let y = g.rms_norm(x, gamma, beta, 1e-5);
            g.set_outputs(vec![y]);
            g
        },
        &[("g", vec![1.0; 16]), ("b", vec![0.0; 16])],
        &[("x", det(48, 13, 23, 0.5))],
        1e-4,
        &mut rows,
    );

    // ── Reductions ──────────────────────────────────────────────────
    let reduce_variants = [
        ("Sum", ReduceOp::Sum),
        ("Mean", ReduceOp::Mean),
        ("Max", ReduceOp::Max),
        ("Min", ReduceOp::Min),
        ("Prod", ReduceOp::Prod),
    ];
    for (name, op) in &reduce_variants {
        let op = *op;
        let label = format!("Reduce::{name} (axis=-1)");
        check(
            label,
            &|| {
                let mut g = Graph::new("red");
                let x = g.input("x", Shape::new(&[3, 8], f));
                let y = g.add_node(
                    Op::Reduce {
                        op,
                        axes: vec![1],
                        keep_dim: false,
                    },
                    vec![x],
                    Shape::new(&[3], f),
                );
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("x", det(24, 7, 11, 0.5))],
            1e-4,
            &mut rows,
        );
    }

    // Reduce along axis=0 (non-last). Tests the inner != 1 path of the
    // generalised Reduce thunk.
    check(
        "Reduce::Sum (axis=0)".into(),
        &|| {
            let mut g = Graph::new("red0");
            let x = g.input("x", Shape::new(&[3, 8], f));
            let y = g.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![0],
                    keep_dim: false,
                },
                vec![x],
                Shape::new(&[8], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(24, 7, 11, 0.5))],
        1e-4,
        &mut rows,
    );

    // Reduce along contiguous axes [0, 1] of a 3-D tensor — common
    // "global pooling minus channel dim" pattern after flattening.
    check(
        "Reduce::Mean (axes=[0,1])".into(),
        &|| {
            let mut g = Graph::new("redmulti");
            let x = g.input("x", Shape::new(&[2, 3, 4], f));
            let y = g.add_node(
                Op::Reduce {
                    op: ReduceOp::Mean,
                    axes: vec![0, 1],
                    keep_dim: false,
                },
                vec![x],
                Shape::new(&[4], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(24, 7, 13, 0.1))],
        1e-4,
        &mut rows,
    );

    // ── Shape ops ───────────────────────────────────────────────────
    check(
        "Reshape".into(),
        &|| {
            let mut g = Graph::new("rs");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let y = g.reshape_(x, vec![3, 16]);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(48, 7, 13, 0.5))],
        1e-6,
        &mut rows,
    );

    check(
        "Transpose (2D)".into(),
        &|| {
            let mut g = Graph::new("tr");
            let x = g.input("x", Shape::new(&[4, 6], f));
            let y = g.add_node(
                Op::Transpose { perm: vec![1, 0] },
                vec![x],
                Shape::new(&[6, 4], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(24, 7, 13, 0.5))],
        1e-6,
        &mut rows,
    );

    check(
        "Narrow (axis=last)".into(),
        &|| {
            let mut g = Graph::new("nr");
            let x = g.input("x", Shape::new(&[6, 12], f));
            let y = g.narrow_(x, 1, 3, 5);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(72, 7, 13, 0.5))],
        1e-6,
        &mut rows,
    );

    check(
        "Concat (axis=last, 2 inputs)".into(),
        &|| {
            let mut g = Graph::new("ct");
            let a = g.param("a", Shape::new(&[6, 4], f));
            let b = g.param("b", Shape::new(&[6, 5], f));
            let y = g.add_node(Op::Concat { axis: 1 }, vec![a, b], Shape::new(&[6, 9], f));
            g.set_outputs(vec![y]);
            g
        },
        &[("a", det(24, 7, 13, 0.5)), ("b", det(30, 11, 17, 0.5))],
        &[],
        1e-6,
        &mut rows,
    );

    check(
        "Expand (broadcast)".into(),
        &|| {
            let mut g = Graph::new("ex");
            let x = g.input("x", Shape::new(&[1, 8], f));
            let y = g.add_node(
                Op::Expand {
                    target_shape: vec![4, 8],
                },
                vec![x],
                Shape::new(&[4, 8], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(8, 7, 11, 0.5))],
        1e-6,
        &mut rows,
    );

    check(
        "Gather (axis=0)".into(),
        &|| {
            let mut g = Graph::new("g0");
            let table = g.param("t", Shape::new(&[10, 8], f));
            let idx = g.input("idx", Shape::new(&[4], f));
            let y = g.gather_(table, idx, 0);
            g.set_outputs(vec![y]);
            g
        },
        &[("t", det(80, 7, 13, 0.5))],
        &[("idx", vec![3.0, 1.0, 7.0, 0.0])],
        1e-6,
        &mut rows,
    );

    check(
        "Gather (axis=1)".into(),
        &|| {
            let mut g = Graph::new("g1");
            let table = g.param("t", Shape::new(&[8, 10], f));
            let idx = g.input("idx", Shape::new(&[4], f));
            let y = g.gather_(table, idx, 1);
            g.set_outputs(vec![y]);
            g
        },
        &[("t", det(80, 7, 13, 0.5))],
        &[("idx", vec![3.0, 1.0, 7.0, 0.0])],
        1e-6,
        &mut rows,
    );

    // ── Cast ─────────────────────────────────────────────────────────
    check(
        "Cast f32→f16→f32 (round-trip)".into(),
        &|| {
            let mut g = Graph::new("c");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let h = g.cast(x, DType::F16);
            let y = g.cast(h, DType::F32);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(48, 7, 13, 0.5))],
        1e-3,
        &mut rows,
    );

    // ── Softmax ──────────────────────────────────────────────────────
    check(
        "Softmax (axis=-1)".into(),
        &|| {
            let mut g = Graph::new("sm");
            let x = g.input("x", Shape::new(&[3, 8], f));
            let y = g.add_node(Op::Softmax { axis: -1 }, vec![x], Shape::new(&[3, 8], f));
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", det(24, 7, 11, 0.5))],
        1e-5,
        &mut rows,
    );

    // ── Attention / RoPE ─────────────────────────────────────────────
    check(
        "RoPE".into(),
        &|| {
            let mut g = Graph::new("rope");
            let x = g.input("x", Shape::new(&[2, 4, 16], f));
            let cos = g.param("cos", Shape::new(&[8, 8], f));
            let sin = g.param("sin", Shape::new(&[8, 8], f));
            let y = g.rope(x, cos, sin, 16);
            g.set_outputs(vec![y]);
            g
        },
        &[("cos", det(64, 7, 13, 0.5)), ("sin", det(64, 11, 17, 0.5))],
        &[("x", det(128, 5, 19, 0.5))],
        1e-4,
        &mut rows,
    );

    check(
        "Attention (SDPA)".into(),
        &|| {
            let mut g = Graph::new("attn");
            let nh = 2;
            let dh = 4;
            let h = nh * dh;
            let q = g.input("q", Shape::new(&[1, 4, h], f));
            let k = g.input("k", Shape::new(&[1, 4, h], f));
            let v = g.input("v", Shape::new(&[1, 4, h], f));
            let mask = g.input("mask", Shape::new(&[1, 4], f));
            let y = g.attention_(q, k, v, mask, nh, dh);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[
            ("q", det(32, 5, 13, 0.5)),
            ("k", det(32, 7, 17, 0.5)),
            ("v", det(32, 11, 19, 0.5)),
            ("mask", vec![1.0; 4]),
        ],
        1e-3,
        &mut rows,
    );

    // ── TopK (Phase D — MoE gating primitive) ──────────────────────
    // Hand-rolled reference: argmax-with-masking, ties → smaller index.
    // Validates both backends produce the same selection ordering.
    {
        let outer = 4usize;
        let axis_dim = 8usize;
        let k = 3usize;
        // Mix of values across rows so each top-k pick differs.
        let x = det(outer * axis_dim, 7, 11, 0.1);
        let mut reference = vec![0f32; outer * k];
        for o in 0..outer {
            let mut row = x[o * axis_dim..(o + 1) * axis_dim].to_vec();
            for ki in 0..k {
                let mut best_i = 0usize;
                let mut best_v = row[0];
                for i in 1..axis_dim {
                    if row[i] > best_v {
                        best_v = row[i];
                        best_i = i;
                    }
                }
                reference[o * k + ki] = best_i as f32;
                row[best_i] = f32::NEG_INFINITY;
            }
        }
        run_check(
            "TopK (k=3, axis=last)".into(),
            &|| {
                let mut g = Graph::new("topk");
                let xv = g.input("x", Shape::new(&[outer, axis_dim], f));
                let y = g.add_node(Op::TopK { k }, vec![xv], Shape::new(&[outer, k], f));
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("x", x)],
            Some(reference),
            0.5,
            &mut rows,
        );
    }

    // ── GroupedMatMul (Phase D — MoE GEMM) ──────────────────────────
    // Indexed batched matmul: out[i] = input[i] @ weight[expert_idx[i]].
    // Build a hand-rolled reference to validate the indexing path.
    {
        let m = 6usize;
        let k_dim = 4usize;
        let n = 3usize;
        let num_experts = 4usize;
        let input = det(m * k_dim, 7, 13, 0.05);
        let weight = det(num_experts * k_dim * n, 11, 17, 0.05);
        // Each token routes to a different expert (round-robin).
        let expert_idx: Vec<f32> = (0..m).map(|i| (i % num_experts) as f32).collect();
        let mut reference = vec![0f32; m * n];
        for i in 0..m {
            let e = expert_idx[i] as usize;
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k_dim {
                    acc += input[i * k_dim + kk] * weight[(e * k_dim + kk) * n + j];
                }
                reference[i * n + j] = acc;
            }
        }
        run_check(
            "GroupedMatMul (M=6, E=4)".into(),
            &|| {
                let mut g = Graph::new("gmm");
                let xv = g.input("x", Shape::new(&[m, k_dim], f));
                let w = g.param("w", Shape::new(&[num_experts, k_dim, n], f));
                let idx = g.input("idx", Shape::new(&[m], f));
                let y = g.add_node(Op::GroupedMatMul, vec![xv, w, idx], Shape::new(&[m, n], f));
                g.set_outputs(vec![y]);
                g
            },
            &[("w", weight)],
            &[("x", input), ("idx", expert_idx)],
            Some(reference),
            1e-4,
            &mut rows,
        );
    }

    // ── ScatterAdd (Phase D — MoE unpermute) ────────────────────────
    // Multiple updates target row 1 to exercise the atomic-add path.
    {
        let num_updates = 5usize;
        let out_dim = 4usize;
        let trailing = 3usize;
        let updates = det(num_updates * trailing, 11, 19, 0.05);
        // Indices: rows {0, 1, 2, 1, 1} — row 1 receives 3 contributions.
        let indices: Vec<f32> = vec![0.0, 1.0, 2.0, 1.0, 1.0];
        let mut reference = vec![0f32; out_dim * trailing];
        for i in 0..num_updates {
            let row = indices[i] as usize;
            for j in 0..trailing {
                reference[row * trailing + j] += updates[i * trailing + j];
            }
        }
        run_check(
            "ScatterAdd (5 updates, 4 rows)".into(),
            &|| {
                let mut g = Graph::new("scatter");
                let upd = g.input("upd", Shape::new(&[num_updates, trailing], f));
                let idx = g.input("idx", Shape::new(&[num_updates], f));
                let y = g.add_node(
                    Op::ScatterAdd,
                    vec![upd, idx],
                    Shape::new(&[out_dim, trailing], f),
                );
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("upd", updates), ("idx", indices)],
            Some(reference),
            1e-5,
            &mut rows,
        );
    }

    // ── Pool 2D ─────────────────────────────────────────────────────
    for (name, kind) in &[("MaxPool2D", ReduceOp::Max), ("MeanPool2D", ReduceOp::Mean)] {
        let kind = *kind;
        let label = format!("Pool::{name} (2x2, stride 2)");
        check(
            label,
            &|| {
                let mut g = Graph::new("pool");
                let x = g.input("x", Shape::new(&[1, 2, 4, 4], f));
                let y = g.add_node(
                    Op::Pool {
                        kind,
                        kernel_size: vec![2, 2],
                        stride: vec![2, 2],
                        padding: vec![0, 0],
                    },
                    vec![x],
                    Shape::new(&[1, 2, 2, 2], f),
                );
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("x", det(32, 7, 17, 0.1))],
            1e-5,
            &mut rows,
        );
    }

    // ── Conv 2D ─────────────────────────────────────────────────────
    check(
        "Conv (1×1, simple)".into(),
        &|| {
            let mut g = Graph::new("conv");
            let x = g.input("x", Shape::new(&[1, 4, 8, 8], f));
            let w = g.param("w", Shape::new(&[6, 4, 1, 1], f));
            let y = g.add_node(
                Op::Conv {
                    kernel_size: vec![1, 1],
                    stride: vec![1, 1],
                    padding: vec![0, 0],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![x, w],
                Shape::new(&[1, 6, 8, 8], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("w", det(24, 7, 13, 0.1))],
        &[("x", det(256, 11, 17, 0.1))],
        1e-4,
        &mut rows,
    );

    // 3×3 with same-padding — exercises the bound checks + indexing.
    check(
        "Conv (3×3, pad=1)".into(),
        &|| {
            let mut g = Graph::new("conv");
            let x = g.input("x", Shape::new(&[1, 3, 8, 8], f));
            let w = g.param("w", Shape::new(&[5, 3, 3, 3], f));
            let y = g.add_node(
                Op::Conv {
                    kernel_size: vec![3, 3],
                    stride: vec![1, 1],
                    padding: vec![1, 1],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![x, w],
                Shape::new(&[1, 5, 8, 8], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("w", det(5 * 3 * 3 * 3, 7, 13, 0.05))],
        &[("x", det(1 * 3 * 8 * 8, 11, 17, 0.05))],
        1e-4,
        &mut rows,
    );

    // Strided + grouped — depthwise-style: groups = c_in = c_out = 4.
    check(
        "Conv (3×3, groups=4, stride=2)".into(),
        &|| {
            let mut g = Graph::new("conv_dw");
            let x = g.input("x", Shape::new(&[1, 4, 8, 8], f));
            let w = g.param("w", Shape::new(&[4, 1, 3, 3], f));
            let y = g.add_node(
                Op::Conv {
                    kernel_size: vec![3, 3],
                    stride: vec![2, 2],
                    padding: vec![1, 1],
                    dilation: vec![1, 1],
                    groups: 4,
                },
                vec![x, w],
                Shape::new(&[1, 4, 4, 4], f),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("w", det(4 * 1 * 3 * 3, 7, 13, 0.1))],
        &[("x", det(1 * 4 * 8 * 8, 11, 17, 0.1))],
        1e-4,
        &mut rows,
    );

    // ── Print report ────────────────────────────────────────────────
    println!("{:<40} {:>14} {:>22}", "Op", "CPU", "Metal");
    println!("{:-<78}", "");
    let (mut cpu_ok, mut cpu_missing, mut metal_ok, mut metal_missing, mut mismatch) =
        (0, 0, 0, 0, 0);
    for (name, cpu, metal) in &rows {
        match cpu {
            Status::Ok => cpu_ok += 1,
            Status::Missing | Status::CpuMissing | Status::BothMissing => cpu_missing += 1,
            _ => {}
        }
        match metal {
            Status::Ok => metal_ok += 1,
            Status::Missing | Status::BothMissing => metal_missing += 1,
            Status::Mismatch(_) => mismatch += 1,
            _ => {}
        }
        println!(
            "{name:<40} {} {:<12} {} {:<20}",
            cpu.glyph(),
            cpu.label(),
            metal.glyph(),
            metal.label()
        );
    }
    println!();
    let total = rows.len();
    println!("Summary across {total} ops:");
    println!("  CPU   : {cpu_ok} ok, {cpu_missing} missing");
    println!("  Metal : {metal_ok} ok, {metal_missing} missing, {mismatch} mismatch");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("op_coverage requires --features metal on macOS");
}
