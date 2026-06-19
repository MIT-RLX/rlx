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

//! CPU↔Metal parity harness — sweeps every Op variant we expect a real
//! model to hit and asserts the two backends produce numerically equivalent
//! results. Catches the Concat-Nop class of bug (where a backend silently
//! returns zeros for an unhandled op) immediately, before it propagates.
//!
//! Add a new check by writing a `check(...)` call: a tiny graph + inputs +
//! params, plus a tolerance. The harness prints PASS/FAIL per case and
//! exits non-zero if any failed.
//!
//! cargo run -p rlx-runtime --example parity_harness --features metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let mut failures: Vec<String> = Vec::new();

    // ── helper: build, run on both backends, compare ────────────────
    let run_check = |name: &str,
                     build: &dyn Fn() -> Graph,
                     params: &[(&str, Vec<f32>)],
                     inputs: &[(&str, Vec<f32>)],
                     tol: f32,
                     failures: &mut Vec<String>| {
        let g_cpu = build();
        let g_metal = build();

        let cpu_session = Session::new(Device::Cpu);
        let mut cpu = cpu_session.compile(g_cpu);
        for (n, d) in params {
            cpu.set_param(n, d);
        }
        let cpu_inputs: Vec<(&str, &[f32])> =
            inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let cpu_out = cpu.run(&cpu_inputs);

        let metal_session = Session::new(Device::Metal);
        let mut metal = metal_session.compile(g_metal);
        for (n, d) in params {
            metal.set_param(n, d);
        }
        let metal_out = metal.run(&cpu_inputs);

        if cpu_out.len() != metal_out.len() {
            failures.push(format!(
                "{name}: output count mismatch ({} vs {})",
                cpu_out.len(),
                metal_out.len()
            ));
            println!("{name:35} FAIL  output count mismatch");
            return;
        }

        let mut max_err = 0f32;
        let mut max_rel = 0f32;
        let mut nan_cpu = 0usize;
        let mut nan_metal = 0usize;
        for (c, m) in cpu_out[0].iter().zip(metal_out[0].iter()) {
            if c.is_nan() {
                nan_cpu += 1;
                continue;
            }
            if m.is_nan() {
                nan_metal += 1;
                continue;
            }
            let abs = (c - m).abs();
            max_err = max_err.max(abs);
            let denom = c.abs().max(m.abs()).max(1e-6);
            max_rel = max_rel.max(abs / denom);
        }

        let pass = nan_cpu == 0 && nan_metal == 0 && max_err < tol;
        let status = if pass { "PASS" } else { "FAIL" };
        println!(
            "{name:35} {status}  max_err={max_err:.3e}  rel={max_rel:.3e}  nan(cpu/metal)={nan_cpu}/{nan_metal}"
        );
        if !pass {
            failures.push(format!(
                "{name}: max_err={max_err:.3e}, nan(cpu/metal)={nan_cpu}/{nan_metal}"
            ));
        }
    };

    let f = DType::F32;

    // ── 1. MatMul ────────────────────────────────────────────────────
    run_check(
        "matmul",
        &|| {
            let mut g = Graph::new("matmul");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let w = g.param("w", Shape::new(&[8, 4], f));
            let y = g.matmul(x, w, Shape::new(&[6, 4], f));
            g.set_outputs(vec![y]);
            g
        },
        &[("w", deterministic(8 * 4, 17, 31))],
        &[("x", deterministic(6 * 8, 13, 23))],
        1e-5,
        &mut failures,
    );

    // ── 2. MatMul + bias + activation (fused) ───────────────────────
    run_check(
        "matmul+bias+gelu (fused)",
        &|| {
            let mut g = Graph::new("mm_bias_gelu");
            let x = g.input("x", Shape::new(&[4, 8], f));
            let w = g.param("w", Shape::new(&[8, 6], f));
            let b = g.param("b", Shape::new(&[6], f));
            let mm = g.matmul(x, w, Shape::new(&[4, 6], f));
            let add = g.binary(BinaryOp::Add, mm, b, Shape::new(&[4, 6], f));
            let y = g.activation(Activation::Gelu, add, Shape::new(&[4, 6], f));
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("w", deterministic(48, 11, 19)),
            ("b", deterministic(6, 7, 13)),
        ],
        &[("x", deterministic(32, 5, 11))],
        1e-5,
        &mut failures,
    );

    // ── 3. Activations standalone ───────────────────────────────────
    for (name, act) in &[
        ("gelu", Activation::Gelu),
        ("silu", Activation::Silu),
        ("relu", Activation::Relu),
    ] {
        let act = *act;
        let label = format!("activation_{name}");
        run_check(
            &label,
            &|| {
                let mut g = Graph::new("act");
                let x = g.input("x", Shape::new(&[4, 8], f));
                let y = g.activation(act, x, Shape::new(&[4, 8], f));
                g.set_outputs(vec![y]);
                g
            },
            &[],
            &[("x", deterministic(32, 13, 17))],
            1e-5,
            &mut failures,
        );
    }

    // ── 4. Binary ops ───────────────────────────────────────────────
    for (name, op) in &[
        ("add", BinaryOp::Add),
        ("mul", BinaryOp::Mul),
        ("sub", BinaryOp::Sub),
    ] {
        let op = *op;
        let label = format!("binary_{name}");
        run_check(
            &label,
            &|| {
                let mut g = Graph::new("bin");
                let x = g.input("x", Shape::new(&[4, 8], f));
                let y = g.input("y", Shape::new(&[4, 8], f));
                let z = g.binary(op, x, y, Shape::new(&[4, 8], f));
                g.set_outputs(vec![z]);
                g
            },
            &[],
            &[
                ("x", deterministic(32, 7, 11)),
                ("y", deterministic(32, 13, 19)),
            ],
            1e-5,
            &mut failures,
        );
    }

    // ── 5. LayerNorm ────────────────────────────────────────────────
    run_check(
        "layer_norm",
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
        &[("x", deterministic(48, 13, 23))],
        1e-4,
        &mut failures,
    );

    // ── 6. Fused residual + LN ──────────────────────────────────────
    run_check(
        "residual+LN (fused)",
        &|| {
            let mut g = Graph::new("res_ln");
            let x = g.input("x", Shape::new(&[3, 16], f));
            let r = g.input("r", Shape::new(&[3, 16], f));
            let gamma = g.param("g", Shape::new(&[16], f));
            let beta = g.param("b", Shape::new(&[16], f));
            let added = g.binary(BinaryOp::Add, x, r, Shape::new(&[3, 16], f));
            let y = g.ln(added, gamma, beta, 1e-5);
            g.set_outputs(vec![y]);
            g
        },
        &[("g", vec![1.0; 16]), ("b", vec![0.0; 16])],
        &[
            ("x", deterministic(48, 13, 23)),
            ("r", deterministic(48, 17, 29)),
        ],
        1e-4,
        &mut failures,
    );

    // ── 7. Cast (f32→f16→f32 round-trip via mixed precision boundary) ─
    run_check(
        "cast_f32_to_f16_to_f32",
        &|| {
            let mut g = Graph::new("cast");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let h = g.cast(x, DType::F16);
            let y = g.cast(h, DType::F32);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", deterministic(48, 7, 13))],
        1e-3,
        &mut failures,
    ); // f16 has ~3 decimal digits

    // ── 8. Reshape ──────────────────────────────────────────────────
    run_check(
        "reshape",
        &|| {
            let mut g = Graph::new("reshape");
            let x = g.input("x", Shape::new(&[6, 8], f));
            let y = g.reshape_(x, vec![3, 16]);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", deterministic(48, 7, 13))],
        1e-6,
        &mut failures,
    );

    // ── 9. Narrow ───────────────────────────────────────────────────
    run_check(
        "narrow_lastax",
        &|| {
            let mut g = Graph::new("narrow");
            let x = g.input("x", Shape::new(&[6, 12], f));
            let y = g.narrow_(x, 1, 3, 5);
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", deterministic(72, 7, 13))],
        1e-6,
        &mut failures,
    );

    // ── 10. Concat (the bug we just fixed) ──────────────────────────
    run_check(
        "concat_lastax_2inputs",
        &|| {
            let mut g = Graph::new("concat");
            // Use params (not inputs) for both halves: matches the
            // FuseSharedInputMatMul pattern that produces concat-of-weights.
            let a = g.param("a", Shape::new(&[6, 4], f));
            let b = g.param("b", Shape::new(&[6, 5], f));
            let y = g.add_node(Op::Concat { axis: 1 }, vec![a, b], Shape::new(&[6, 9], f));
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("a", deterministic(24, 7, 13)),
            ("b", deterministic(30, 11, 17)),
        ],
        &[],
        1e-6,
        &mut failures,
    );

    // ── 11. Gather (axis=0, embedding lookup) ──────────────────────
    run_check(
        "gather_axis0",
        &|| {
            let mut g = Graph::new("gather");
            let table = g.param("table", Shape::new(&[10, 8], f));
            let idx = g.input("idx", Shape::new(&[4], f));
            let y = g.gather_(table, idx, 0);
            g.set_outputs(vec![y]);
            g
        },
        &[("table", deterministic(80, 7, 13))],
        &[("idx", vec![3.0, 1.0, 7.0, 0.0])],
        1e-6,
        &mut failures,
    );

    // ── 12. RoPE ────────────────────────────────────────────────────
    run_check(
        "rope",
        &|| {
            let mut g = Graph::new("rope");
            let x = g.input("x", Shape::new(&[2, 4, 16], f));
            let cos = g.param("cos", Shape::new(&[8, 8], f));
            let sin = g.param("sin", Shape::new(&[8, 8], f));
            let y = g.rope(x, cos, sin, 16);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("cos", deterministic(64, 7, 13)),
            ("sin", deterministic(64, 11, 17)),
        ],
        &[("x", deterministic(128, 5, 19))],
        1e-4,
        &mut failures,
    );

    // ── 13. Softmax ─────────────────────────────────────────────────
    run_check(
        "softmax_lastax",
        &|| {
            let mut g = Graph::new("softmax");
            let x = g.input("x", Shape::new(&[3, 8], f));
            let y = g.add_node(Op::Softmax { axis: -1 }, vec![x], Shape::new(&[3, 8], f));
            g.set_outputs(vec![y]);
            g
        },
        &[],
        &[("x", deterministic(24, 7, 11))],
        1e-5,
        &mut failures,
    );

    // ── 14. SDPA Attention ──────────────────────────────────────────
    run_check(
        "attention_sdpa",
        &|| {
            let mut g = Graph::new("attn");
            let nh = 2usize;
            let dh = 4usize;
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
            ("q", deterministic(32, 5, 13)),
            ("k", deterministic(32, 7, 17)),
            ("v", deterministic(32, 11, 19)),
            ("mask", vec![1.0, 1.0, 1.0, 1.0]),
        ],
        1e-3,
        &mut failures,
    );

    // ── 15. FusedSwiGLU (fired via FuseSharedInputMatMul pipeline) ─
    run_check(
        "swiglu_full_pipeline",
        &|| {
            let mut g = Graph::new("swiglu");
            let m = 4usize;
            let k = 8usize;
            let n = 6usize;
            let x = g.input("x", Shape::new(&[m, k], f));
            let w_up = g.param("w_up", Shape::new(&[k, n], f));
            let w_gate = g.param("w_gate", Shape::new(&[k, n], f));
            let up_mm = g.matmul(x, w_up, Shape::new(&[m, n], f));
            let gate_mm = g.matmul(x, w_gate, Shape::new(&[m, n], f));
            let gate = g.activation(Activation::Silu, gate_mm, Shape::new(&[m, n], f));
            let y = g.binary(BinaryOp::Mul, up_mm, gate, Shape::new(&[m, n], f));
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("w_up", deterministic(48, 11, 23)),
            ("w_gate", deterministic(48, 13, 19)),
        ],
        &[("x", deterministic(32, 7, 17))],
        1e-5,
        &mut failures,
    );

    // ── Summary ─────────────────────────────────────────────────────
    println!();
    if failures.is_empty() {
        println!("ALL PASS — CPU and Metal agree across {} cases", 19);
    } else {
        eprintln!("{} FAIL(s):", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn deterministic(n: usize, mul: usize, modulo: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * mul + 7) % modulo) as f32 / modulo as f32 - 0.5)
        .collect()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("parity_harness requires --features metal on macOS");
}
