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

//! Toy 4-expert Mixture-of-Experts forward pass — end-to-end parity test
//! for the Phase D primitives (TopK, GroupedMatMul, ScatterAdd).
//!
//! Builds the canonical k=1 MoE pattern from raw rlx-ir ops and verifies
//! CPU and Metal produce bit-identical outputs against a hand-rolled f32
//! reference. Validates the primitives compose correctly even without a
//! real MoE checkpoint to test against.
//!
//! Pipeline:
//!   logits  = x @ gate_w                       # gating logits per expert
//!   probs   = softmax(logits, axis=-1)         # routing distribution
//!   top_idx = TopK(probs, k=1)                 # winning expert per token
//!   gate    = Reduce::Max(probs, axis=-1)      # the winning probability
//!   expert_out = GroupedMatMul(x, W, top_idx)  # the chosen expert's GEMM
//!   out     = expert_out * gate                # weighted contribution
//!
//! cargo run --release -p rlx-runtime --example moe_demo --features metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::{BinaryOp, ReduceOp};
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    // Tiny shapes — enough to exercise routing, small enough to verify by hand.
    let m = 4usize; // tokens
    let n_experts = 4usize;
    let h = 8usize; // hidden / output dim

    let det = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
            .collect()
    };
    let x_data = det(0, m * h, 0.5);
    let gate_w = det(1, h * n_experts, 0.3);
    let expert_w = det(2, n_experts * h * h, 0.2);

    // ── Build the IR graph ───────────────────────────────────────────
    let build = || {
        let f = DType::F32;
        let mut g = Graph::new("moe");
        let x = g.input("x", Shape::new(&[m, h], f));
        let gw = g.param("gate_w", Shape::new(&[h, n_experts], f));
        let ew = g.param("expert_w", Shape::new(&[n_experts, h, h], f));

        let logits = g.matmul(x, gw, Shape::new(&[m, n_experts], f));
        let probs = g.add_node(
            Op::Softmax { axis: -1 },
            vec![logits],
            Shape::new(&[m, n_experts], f),
        );
        // TopK indices, k=1: [M, 1].
        let top_idx_2d = g.add_node(Op::TopK { k: 1 }, vec![probs], Shape::new(&[m, 1], f));
        // Flatten to [M] so GroupedMatMul accepts it.
        let top_idx = g.reshape_(top_idx_2d, vec![m as i64]);
        // The winning probability is the row max — equivalent to
        // gathering by top_idx because top_idx points at the argmax.
        let gate_val = g.add_node(
            Op::Reduce {
                op: ReduceOp::Max,
                axes: vec![1],
                keep_dim: false,
            },
            vec![probs],
            Shape::new(&[m], f),
        );
        // Broadcast [M] gate to [M, h] for the element-wise multiply.
        let gate_2d = g.reshape_(gate_val, vec![m as i64, 1]);
        let gate_b = g.add_node(
            Op::Expand {
                target_shape: vec![m as i64, h as i64],
            },
            vec![gate_2d],
            Shape::new(&[m, h], f),
        );
        // Each token's chosen expert dot-products against x[i].
        let expert_out = g.add_node(
            Op::GroupedMatMul,
            vec![x, ew, top_idx],
            Shape::new(&[m, h], f),
        );
        // Final weighted output.
        let out = g.binary(BinaryOp::Mul, expert_out, gate_b, Shape::new(&[m, h], f));
        g.set_outputs(vec![out]);
        g
    };

    // ── Hand-rolled f32 reference ────────────────────────────────────
    let mut reference = vec![0f32; m * h];
    for i in 0..m {
        // logits[e] = sum_k x[i,k] * gate_w[k,e]
        let mut logits = vec![0f32; n_experts];
        for e in 0..n_experts {
            for k in 0..h {
                logits[e] += x_data[i * h + k] * gate_w[k * n_experts + e];
            }
        }
        // softmax(logits)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        // argmax — winning expert
        let mut best_e = 0usize;
        let mut best_p = probs[0];
        for e in 1..n_experts {
            if probs[e] > best_p {
                best_p = probs[e];
                best_e = e;
            }
        }
        // expert_out[i, j] = sum_k x[i,k] * expert_w[best_e, k, j]
        for j in 0..h {
            let mut acc = 0f32;
            for k in 0..h {
                acc += x_data[i * h + k] * expert_w[(best_e * h + k) * h + j];
            }
            reference[i * h + j] = acc * best_p;
        }
    }

    // ── Run on both backends and compare ─────────────────────────────
    let cpu_session = Session::new(Device::Cpu);
    let mut cpu = cpu_session.compile(build());
    cpu.set_param("gate_w", &gate_w);
    cpu.set_param("expert_w", &expert_w);
    let cpu_out = cpu.run(&[("x", &x_data)]);

    let metal_session = Session::new(Device::Metal);
    let mut metal = metal_session.compile(build());
    metal.set_param("gate_w", &gate_w);
    metal.set_param("expert_w", &expert_w);
    let metal_out = metal.run(&[("x", &x_data)]);

    // Numerical comparisons against the reference.
    let max_err = |actual: &[f32], r: &[f32]| -> f32 {
        actual
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };
    let cpu_err = max_err(&cpu_out[0], &reference);
    let metal_err = max_err(&metal_out[0], &reference);
    let cm_err = max_err(&cpu_out[0], &metal_out[0]);

    println!("Toy 4-expert MoE forward (M={m}, hidden={h}, experts={n_experts}, k=1)");
    println!("  CPU vs reference   : max_err = {cpu_err:.3e}");
    println!("  Metal vs reference : max_err = {metal_err:.3e}");
    println!("  CPU vs Metal       : max_err = {cm_err:.3e}");
    println!();
    let pass = cpu_err < 1e-5 && metal_err < 1e-5 && cm_err < 1e-6;
    if pass {
        println!("PASS — Phase D primitives compose into a working MoE forward.");
    } else {
        eprintln!("FAIL — toy MoE diverged from the hand-rolled reference.");
        std::process::exit(1);
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("moe_demo requires --features metal on macOS");
}
