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

//! End-to-end PJRT round-trip tests for rlx-tpu.
//!
//! All tests in this file are gated on `LIBTPU_PATH` being set —
//! they're meant to run inside the Docker harness (which points
//! at `libpjrt_c_cpu.so`) or on a real TPU VM. On hosts without
//! a plugin they skip cleanly.
//!
//! Each test compiles a small `rlx_ir::Graph` through rlx-tpu,
//! executes against the loaded PJRT plugin, and validates the
//! result against rlx-cpu's reference output.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};

fn skip_without_plugin() -> bool {
    if std::env::var("LIBTPU_PATH").is_err() {
        eprintln!("[pjrt_roundtrip] LIBTPU_PATH not set — skipping");
        return true;
    }
    if !rlx_tpu::is_available() {
        eprintln!(
            "[pjrt_roundtrip] LIBTPU_PATH set but plugin failed to \
                   initialize — skipping"
        );
        return true;
    }
    false
}

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Trivial element-wise add — exercises the smallest possible HLO
/// path: parameter, parameter, add, root.
#[test]
fn add_two_vectors() {
    if skip_without_plugin() {
        return;
    }

    let mut g = Graph::new("add_vec");
    let s = Shape::new(&[6], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.binary(BinaryOp::Add, x, y, s);
    g.set_outputs(vec![z]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ys = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let outs = exec.run(&[("x", &xs), ("y", &ys)]);
    assert_eq!(outs.len(), 1);
    let expected = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0];
    for (i, (a, b)) in outs[0].iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(*a, *b, 1e-5), "elem {i}: got {a}, want {b}");
    }
}

/// 2-D matmul: [4, 3] × [3, 5] → [4, 5]. Validates the dot_general
/// emission with no batch dims.
#[test]
fn matmul_2d() {
    if skip_without_plugin() {
        return;
    }

    let mut g = Graph::new("mm_2d");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[4, 3], f));
    let w = g.param("w", Shape::new(&[3, 5], f));
    let y = g.matmul(x, w, Shape::new(&[4, 5], f));
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let xs: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let ws: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1).collect();
    exec.set_param("w", &ws);
    let outs = exec.run(&[("x", &xs)]);
    assert_eq!(outs[0].len(), 20);

    // Reference: row-major matmul on host.
    let mut want = [0.0f32; 20];
    for i in 0..4 {
        for j in 0..5 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += xs[i * 3 + k] * ws[k * 5 + j];
            }
            want[i * 5 + j] = s;
        }
    }
    for (n, (a, b)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!(approx_eq(*a, *b, 1e-4), "mm_2d[{n}]: got {a}, want {b}");
    }
}

/// Activation chain: relu(x) then sigmoid(y) — verifies the
/// Activation lowering for two distinct kinds end-to-end.
#[test]
fn activations_relu_sigmoid() {
    if skip_without_plugin() {
        return;
    }

    let mut g = Graph::new("act_chain");
    let s = Shape::new(&[5], DType::F32);
    let x = g.input("x", s.clone());
    let r = g.activation(Activation::Relu, x, s.clone());
    let s2 = g.activation(Activation::Sigmoid, r, s);
    g.set_outputs(vec![s2]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let xs = vec![-2.0_f32, -0.5, 0.0, 0.5, 2.0];
    let outs = exec.run(&[("x", &xs)]);
    let expect = |x: f32| {
        let r = x.max(0.0);
        1.0 / (1.0 + (-r).exp())
    };
    for (i, (a, x)) in outs[0].iter().zip(xs.iter()).enumerate() {
        let want = expect(*x);
        assert!(approx_eq(*a, want, 1e-4), "elem {i}: got {a}, want {want}");
    }
}

/// LayerNorm over a [2, 4] tensor with axis=-1. Validates the
/// composite mean / var / rsqrt / scale + bias decomposition.
#[test]
fn layernorm_minus1() {
    if skip_without_plugin() {
        return;
    }

    let mut g = Graph::new("ln");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 4], f));
    let gamma = g.param("g", Shape::new(&[4], f));
    let beta = g.param("b", Shape::new(&[4], f));
    let y = g.layer_norm(x, gamma, beta, -1, 1e-5, Shape::new(&[2, 4], f));
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let xs = vec![1.0_f32, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let g_v = vec![1.0_f32, 1.0, 1.0, 1.0];
    let b_v = vec![0.0_f32, 0.0, 0.0, 0.0];
    exec.set_param("g", &g_v);
    exec.set_param("b", &b_v);
    let outs = exec.run(&[("x", &xs)]);

    // Each row should be zero-mean unit-variance after layernorm with
    // gamma=1, beta=0. Tolerate ~1e-3 for fp32 reduction differences.
    for row in 0..2 {
        let r: &[f32] = &outs[0][row * 4..row * 4 + 4];
        let mean: f32 = r.iter().sum::<f32>() / 4.0;
        let var: f32 = r.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3, "row {row} mean {mean} not ~ 0");
        assert!((var - 1.0).abs() < 1e-2, "row {row} var {var} not ~ 1");
    }
}

// ── Tier-3 op roundtrip tests (parity targets) ───────────────────

/// TopK along the last axis. Verifies the sort+slice+convert chain.
#[test]
fn topk_descending() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("topk_rt");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 6], f));
    let y = g.add_node(rlx_ir::Op::TopK { k: 3 }, vec![x], Shape::new(&[2, 3], f));
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Row 0: top-3 are at indices 5, 2, 4 (values 9, 7, 5)
    // Row 1: top-3 are at indices 0, 3, 1 (values 8, 6, 4)
    let xs = vec![
        1.0_f32, 3.0, 7.0, 2.0, 5.0, 9.0, 8.0, 4.0, 0.0, 6.0, 1.0, 2.0,
    ];
    let outs = exec.run(&[("x", &xs)]);
    let want = [5.0, 2.0, 4.0, 0.0, 3.0, 1.0];
    for (i, (a, b)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!((*a - *b).abs() < 1e-5, "topk[{i}]: got {a}, want {b}");
    }
}

/// GroupedMatMul: per-token expert dispatch. Two tokens, two experts.
#[test]
fn grouped_matmul() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("gmm_rt");
    let f = DType::F32;
    let m = 2;
    let k = 3;
    let n = 2;
    let e = 2;
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[e, k, n], f));
    let exp = g.input("exp", Shape::new(&[m], f));
    let y = g.add_node(
        rlx_ir::Op::GroupedMatMul,
        vec![x, w, exp],
        Shape::new(&[m, n], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Two experts. Expert 0 = identity-ish, expert 1 = doubles.
    let xs = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let ws = vec![
        // expert 0
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // expert 1
        2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
    ];
    let exps = vec![0.0_f32, 1.0];
    exec.set_param("w", &ws);
    let outs = exec.run(&[("x", &xs), ("exp", &exps)]);
    // Row 0 = x[0] @ expert_0 = [1,2,3] @ [[1,0],[0,1],[1,1]] = [4, 5]
    // Row 1 = x[1] @ expert_1 = [0.5,1.5,2.5] @ 2*[[1,0],[0,1],[1,1]]
    //                          = [6, 8]
    let want = [4.0, 5.0, 6.0, 8.0];
    for (i, (a, b)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!((*a - *b).abs() < 1e-4, "gmm[{i}]: got {a}, want {b}");
    }
}

/// Sample greedy = argmax. Deterministic.
#[test]
fn sample_greedy_argmax() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("sample_rt");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[2, 5], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 0.0,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[2], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Row 0: argmax = 2; row 1: argmax = 4.
    let xs = vec![0.1_f32, 0.2, 0.9, 0.3, 0.0, 0.0, 0.5, 0.0, 0.5, 0.6];
    let outs = exec.run(&[("logits", &xs)]);
    assert!((outs[0][0] - 2.0).abs() < 1e-5, "row 0: got {}", outs[0][0]);
    assert!((outs[0][1] - 4.0).abs() < 1e-5, "row 1: got {}", outs[0][1]);
}

/// Sample(temperature > 0) on logits sharply peaked at index 1 — the
/// multinomial path almost-always returns 1, regardless of RNG draw,
/// because softmax([0, 50, 0, 0, 0]) ≈ [0, 1, 0, 0, 0].
#[test]
fn sample_temperature_peaked() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("sample_temp_rt");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[2, 5], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[2], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Row 0: peaked at index 1.  Row 1: peaked at index 3.
    let xs = vec![0.0_f32, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0];
    let outs = exec.run(&[("logits", &xs)]);
    assert!(
        (outs[0][0] - 1.0).abs() < 1e-5,
        "peaked-at-1 row 0: got {}",
        outs[0][0]
    );
    assert!(
        (outs[0][1] - 3.0).abs() < 1e-5,
        "peaked-at-3 row 1: got {}",
        outs[0][1]
    );
}

/// Sample with top_k=1 (effectively argmax via the multinomial
/// path's top_k filter, not the temperature==0 fast path). Forces
/// the top_k branch to compile and execute.
#[test]
fn sample_top_k_one() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("sample_topk_rt");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[1, 5], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 1,
            top_p: 1.0,
            temperature: 1.0,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[1], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Argmax = 2.
    let xs = vec![0.1_f32, 0.4, 0.9, 0.2, 0.0];
    let outs = exec.run(&[("logits", &xs)]);
    assert!(
        (outs[0][0] - 2.0).abs() < 1e-5,
        "top_k=1 should return argmax, got {}",
        outs[0][0]
    );
}

/// Sample with top_p — exercise the nucleus-filter path. Logits
/// chosen so softmax = [0.5, 0.3, 0.1, 0.06, 0.04]; top_p = 0.79
/// keeps {0, 1} (cum 0.8 reaches threshold at i=1). For a peaked
/// follow-up where we crank temperature down post-filter, the
/// argmax of the kept set is index 0.
#[test]
fn sample_top_p_threshold() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("sample_topp_rt");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[1, 5], f));
    // top_p=0.5 with very low temperature ≈ greedy on the kept set.
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 0.5,
            temperature: 0.01,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[1], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // logits sorted: [3.0(@2), 2.0(@4), 1.0(@0), 0.5(@1), 0.0(@3)]
    // softmax (descending): ~[0.62, 0.23, 0.08, 0.05, 0.02]
    // cumsum:                 [0.62, 0.85, ...]
    // top_p=0.5: first cum >= 0.5 is at position 0 → keep only @2.
    // After temperature scaling at 0.01, the residual mass on @2
    // dominates → argmax = 2.
    let xs = vec![1.0_f32, 0.5, 3.0, 0.0, 2.0];
    let outs = exec.run(&[("logits", &xs)]);
    assert!(
        (outs[0][0] - 2.0).abs() < 1e-5,
        "top_p=0.5 should isolate index 2, got {}",
        outs[0][0]
    );
}

/// DequantMatMul with Int8BlockAsym, block_size=4. Weights and
/// scale/zp are baked in as Constants so we don't need an i8 host
/// upload path. Reference is computed by dequantizing on the host.
#[test]
fn dequant_matmul_int8_block_asym() {
    if skip_without_plugin() {
        return;
    }

    // Shapes: m=2, k=8, n=4, block=4 → kb=2.
    let m = 2;
    let k = 8;
    let n = 4;
    let block = 4;
    let kb = k / block;

    // Pick a deterministic w_q, scale, zp.
    let w_q: Vec<i8> = (0..(k * n))
        .map(|i| ((i % 5) as i8) - 2) // {-2..2}
        .collect();
    // scale[kb, n] — one scale per (block, output_col).
    let scale: Vec<f32> = (0..(kb * n)).map(|i| 0.1 + (i as f32) * 0.05).collect();
    // zp[kb, n] — small biases.
    let zp: Vec<f32> = (0..(kb * n)).map(|i| (i as f32) * 0.5).collect();

    let mut g = Graph::new("dq_rt");
    let f = DType::F32;
    let i8t = DType::I8;
    let x = g.input("x", Shape::new(&[m, k], f));
    let wq_bytes: Vec<u8> = w_q.iter().map(|&b| b as u8).collect();
    let wq_node = g.add_node(
        rlx_ir::Op::Constant { data: wq_bytes },
        vec![],
        Shape::new(&[k, n], i8t),
    );
    let scale_bytes: Vec<u8> = scale.iter().flat_map(|f| f.to_le_bytes()).collect();
    let scale_node = g.add_node(
        rlx_ir::Op::Constant { data: scale_bytes },
        vec![],
        Shape::new(&[kb, n], f),
    );
    let zp_bytes: Vec<u8> = zp.iter().flat_map(|f| f.to_le_bytes()).collect();
    let zp_node = g.add_node(
        rlx_ir::Op::Constant { data: zp_bytes },
        vec![],
        Shape::new(&[kb, n], f),
    );
    let y = g.add_node(
        rlx_ir::Op::DequantMatMul {
            scheme: rlx_ir::quant::QuantScheme::Int8BlockAsym {
                block_size: block as u32,
            },
        },
        vec![x, wq_node, scale_node, zp_node],
        Shape::new(&[m, n], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let xs: Vec<f32> = (0..(m * k)).map(|i| 0.1 * (i as f32)).collect();
    let outs = exec.run(&[("x", &xs)]);

    // Reference: dequantize w_q on host, then matmul.
    let mut w_dq = vec![0.0_f32; k * n];
    for ki in 0..k {
        let block_idx = ki / block;
        for ni in 0..n {
            let s = scale[block_idx * n + ni];
            let z = zp[block_idx * n + ni];
            let q = w_q[ki * n + ni] as f32;
            w_dq[ki * n + ni] = (q - z) * s;
        }
    }
    let mut want = vec![0.0_f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0.0_f32;
            for ki in 0..k {
                s += xs[mi * k + ki] * w_dq[ki * n + ni];
            }
            want[mi * n + ni] = s;
        }
    }
    for (i, (got, w)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!((*got - *w).abs() < 5e-4, "dq_mm[{i}]: got {got}, want {w}");
    }
}

/// Truncating f32 → bf16 (drop the low 16 bits of mantissa). Avoids
/// pulling in the `half` crate just for tests.
fn f32_to_bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// bf16 (raw u16) → f32 by left-padding with zeros. Same truncation
/// path is used by `f32_to_bf16_bits`, so a round-trip is the
/// identity on any value already representable in bf16.
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// bf16 round-trip: cast f32 → BF16 → f32. Verifies that HLO accepts
/// BF16 in lower.rs, and that the runtime upload/download paths
/// handle PJRT_BUFFER_TYPE_BF16 (download_buffer widens BF16 → f32).
#[test]
fn bf16_cast_roundtrip() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("bf16_cast_rt");
    let f32t = DType::F32;
    let bf16t = DType::BF16;
    let x = g.input("x", Shape::new(&[8], f32t));
    let xb = g.add_node(
        rlx_ir::Op::Cast { to: bf16t },
        vec![x],
        Shape::new(&[8], bf16t),
    );
    let xback = g.add_node(
        rlx_ir::Op::Cast { to: f32t },
        vec![xb],
        Shape::new(&[8], f32t),
    );
    g.set_outputs(vec![xback]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // 3.14159 chosen as a "near-PI" non-round float to exercise the
    // bf16 mantissa truncation, not as the actual PI constant.
    #[allow(clippy::approx_constant)]
    let xs: Vec<f32> = vec![1.0, -2.5, 3.14159, 0.0, 1e-5, 1e5, -7.0, 0.5];
    let outs = exec.run(&[("x", &xs)]);

    // Each output equals the bf16-truncated input.
    for (i, (got, want_f)) in outs[0].iter().zip(xs.iter()).enumerate() {
        let want = bf16_bits_to_f32(f32_to_bf16_bits(*want_f));
        // Up to ~0.4% relative error per bf16 quantum.
        let tol = (want.abs() * 4e-3).max(1e-6);
        assert!(
            (got - want).abs() <= tol,
            "bf16_cast[{i}]: got {got}, want {want} (tol {tol})"
        );
    }
}

/// bf16 element-wise multiply with a parameter uploaded directly as
/// bf16 bytes via `set_param_typed`. Skips the f32-widening that
/// `set_param` would do — i.e., the param goes onto the device as
/// `PJRT_BUFFER_TYPE_BF16`, which is the right perf primitive for
/// real LLM weights.
#[test]
fn bf16_param_upload_and_multiply() {
    if skip_without_plugin() {
        return;
    }
    let mut g = Graph::new("bf16_mul_rt");
    let f32t = DType::F32;
    let bf16t = DType::BF16;
    let n = 8usize;

    let x = g.input("x", Shape::new(&[n], f32t));
    let w = g.param("w", Shape::new(&[n], bf16t));
    // Cast x to BF16 so the multiply's operands match dtype.
    let xb = g.add_node(
        rlx_ir::Op::Cast { to: bf16t },
        vec![x],
        Shape::new(&[n], bf16t),
    );
    let prod = g.binary(rlx_ir::op::BinaryOp::Mul, xb, w, Shape::new(&[n], bf16t));
    let out = g.add_node(
        rlx_ir::Op::Cast { to: f32t },
        vec![prod],
        Shape::new(&[n], f32t),
    );
    g.set_outputs(vec![out]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);

    let xs: Vec<f32> = (0..n).map(|i| (i as f32) + 0.5).collect();
    let ws: Vec<f32> = (0..n).map(|i| 0.1 * ((i + 1) as f32)).collect();
    // Upload w as bf16 bytes — two bytes per element, little-endian.
    let w_bf16: Vec<u8> = ws
        .iter()
        .flat_map(|&v| f32_to_bf16_bits(v).to_le_bytes())
        .collect();
    exec.set_param_typed("w", &w_bf16, bf16t);

    let outs = exec.run(&[("x", &xs)]);

    // Reference: multiply in bf16 (truncate inputs first).
    for (i, got) in outs[0].iter().enumerate() {
        let xb = bf16_bits_to_f32(f32_to_bf16_bits(xs[i]));
        let wb = bf16_bits_to_f32(f32_to_bf16_bits(ws[i]));
        let want = bf16_bits_to_f32(f32_to_bf16_bits(xb * wb));
        let tol = (want.abs() * 5e-3).max(1e-5);
        assert!(
            (got - want).abs() <= tol,
            "bf16_mul[{i}]: got {got}, want {want} (tol {tol})"
        );
    }
}

/// Distribution check for non-greedy `Sample`. Draws B=1000 samples
/// from a flat 4-way logits and checks each token's count is within
/// ±50% of the uniform expected value (250). Catches RNG miswiring
/// (constant output, wrong distribution shape) that the deterministic
/// peaked-logits test misses.
#[test]
fn sample_temperature_distribution() {
    if skip_without_plugin() {
        return;
    }
    let b = 1000_usize;
    let v = 4_usize;

    let mut g = Graph::new("sample_dist_rt");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[b, v], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            seed: 42,
        },
        vec![logits],
        Shape::new(&[b], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // Uniform logits — softmax gives 1/V mass on each token.
    let xs: Vec<f32> = vec![0.0; b * v];
    let outs = exec.run(&[("logits", &xs)]);
    assert_eq!(outs[0].len(), b);

    let mut counts = vec![0usize; v];
    for &t in outs[0].iter() {
        let i = t.round() as i64;
        assert!(i >= 0 && (i as usize) < v, "sample out of range: got {t}");
        counts[i as usize] += 1;
    }
    let expected = (b / v) as i64;
    for (i, &c) in counts.iter().enumerate() {
        let dev = (c as i64 - expected).abs();
        // 50% slack — chi-squared p<0.01 for 3 dof needs ~|dev|>~32
        // out of 250, so 125 (50%) is generously loose. We only want
        // to catch "always returns the same token" / "off-by-one
        // wraparound" / etc.
        assert!(
            dev < expected / 2,
            "sample bin {i}: count {c}, expected ~{expected}, \
                 deviation {dev} too large"
        );
    }
}

/// HLO's `round-nearest-even` rounds halves to the nearest even
/// integer (banker's rounding); Rust's `f32::round()` rounds halves
/// away from zero. Use this helper in test references for any
/// quantization op so they don't disagree at exact-half boundaries.
fn round_ties_even_ref(x: f32) -> f32 {
    x.round_ties_even()
}

/// QMatMul (real INT8 path). All operands baked as Constants because
/// the runtime's `run()` only takes f32 host slices. Output is i8;
/// `download_buffer` widens to f32 for the test API.
#[test]
fn qmatmul_int8() {
    if skip_without_plugin() {
        return;
    }
    let m = 2;
    let k = 4;
    let n = 3;
    // x s8, w s8, bias s32 — all Constants.
    let x_q: Vec<i8> = vec![1, 2, 3, 4, -1, 0, 1, 2];
    let w_q: Vec<i8> = vec![1, 0, 1, 0, 1, 1, 1, 1, 0, 2, 0, 1];
    let bias_q: Vec<i32> = vec![0, 0, 0];
    let mult: f32 = 0.5;

    let mut g = Graph::new("qmm_rt");
    let i8t = DType::I8;
    let i32t = DType::I32;
    let x_bytes: Vec<u8> = x_q.iter().map(|&b| b as u8).collect();
    let xn = g.add_node(
        rlx_ir::Op::Constant { data: x_bytes },
        vec![],
        Shape::new(&[m, k], i8t),
    );
    let w_bytes: Vec<u8> = w_q.iter().map(|&b| b as u8).collect();
    let wn = g.add_node(
        rlx_ir::Op::Constant { data: w_bytes },
        vec![],
        Shape::new(&[k, n], i8t),
    );
    let bias_bytes: Vec<u8> = bias_q.iter().flat_map(|i| i.to_le_bytes()).collect();
    let bn = g.add_node(
        rlx_ir::Op::Constant { data: bias_bytes },
        vec![],
        Shape::new(&[n], i32t),
    );
    let y = g.add_node(
        rlx_ir::Op::QMatMul {
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            mult,
        },
        vec![xn, wn, bn],
        Shape::new(&[m, n], i8t),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let outs = exec.run(&[]);

    // Reference: round((x @ w) * mult), clamp to [-128,127].
    let mut want = vec![0_i8; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc: i32 = 0;
            for ki in 0..k {
                acc += (x_q[mi * k + ki] as i32) * (w_q[ki * n + ni] as i32);
            }
            let v = round_ties_even_ref(acc as f32 * mult);
            let clamped = v.clamp(-128.0, 127.0) as i8;
            want[mi * n + ni] = clamped;
        }
    }
    for (i, (got, w)) in outs[0].iter().zip(want.iter()).enumerate() {
        // got is f32-widened; want is i8 → f32.
        assert!(
            (got - (*w as f32)).abs() < 1.0,
            "qmm[{i}]: got {got}, want {w}"
        );
    }
}

/// QConv2d. Same Constants-only pattern as QMatMul.
#[test]
fn qconv2d_int8() {
    if skip_without_plugin() {
        return;
    }
    let n_b = 1;
    let c_in = 1;
    let h = 4;
    let w = 4;
    let c_out = 1;
    let kh = 3;
    let kw = 3;
    let h_out = h;
    let w_out = w; // padding=1, stride=1.

    let x_q: Vec<i8> = (0..(n_b * c_in * h * w)).map(|i| (i as i8) % 8).collect();
    let w_q: Vec<i8> = vec![1, 0, 1, 0, 1, 0, 1, 0, 1];
    let bias_q: Vec<i32> = vec![0];
    let mult: f32 = 0.25;

    let mut g = Graph::new("qcv_rt");
    let i8t = DType::I8;
    let i32t = DType::I32;

    let x_bytes: Vec<u8> = x_q.iter().map(|&b| b as u8).collect();
    let xn = g.add_node(
        rlx_ir::Op::Constant { data: x_bytes },
        vec![],
        Shape::new(&[n_b, c_in, h, w], i8t),
    );
    let w_bytes: Vec<u8> = w_q.iter().map(|&b| b as u8).collect();
    let wn = g.add_node(
        rlx_ir::Op::Constant { data: w_bytes },
        vec![],
        Shape::new(&[c_out, c_in, kh, kw], i8t),
    );
    let bias_bytes: Vec<u8> = bias_q.iter().flat_map(|i| i.to_le_bytes()).collect();
    let bn = g.add_node(
        rlx_ir::Op::Constant { data: bias_bytes },
        vec![],
        Shape::new(&[c_out], i32t),
    );
    let y = g.add_node(
        rlx_ir::Op::QConv2d {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            mult,
        },
        vec![xn, wn, bn],
        Shape::new(&[n_b, c_out, h_out, w_out], i8t),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    let outs = exec.run(&[]);

    // Reference NCHW conv with padding=1, stride=1.
    let h_us = h;
    let w_us = w;
    let kh_us = kh;
    let kw_us = kw;
    let mut want = vec![0_i8; h * w];
    for oh in 0..h_us {
        for ow in 0..w_us {
            let mut acc: i32 = 0;
            for ki in 0..kh_us {
                for kj in 0..kw_us {
                    let ih = oh as isize + ki as isize - 1;
                    let iw = ow as isize + kj as isize - 1;
                    if ih < 0 || iw < 0 || ih >= h_us as isize || iw >= w_us as isize {
                        continue;
                    }
                    let xv = x_q[(ih as usize) * w_us + iw as usize] as i32;
                    let wv = w_q[ki * kw_us + kj] as i32;
                    acc += xv * wv;
                }
            }
            let v = round_ties_even_ref(acc as f32 * mult);
            want[oh * w_us + ow] = v.clamp(-128.0, 127.0) as i8;
        }
    }
    for (i, (got, w)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!(
            (got - (*w as f32)).abs() < 1.0,
            "qcv[{i}]: got {got}, want {w}"
        );
    }
}

/// SelectiveScan with a trivial setup that we can hand-simulate:
/// a = 0 (decay = 1), b = 1, c = 1, x\[t\] = 1 → state accumulates,
/// y\[t\] = state\[t\] (sum-of-ones up to t).
#[test]
fn selective_scan_accumulate() {
    if skip_without_plugin() {
        return;
    }
    let f = DType::F32;
    let bsz = 1;
    let l = 4;
    let d = 2;
    let n = 1;
    let mut g = Graph::new("ssm_rt");
    let x = g.input("x", Shape::new(&[bsz, l, d], f));
    let delta = g.input("delta", Shape::new(&[bsz, l, d], f));
    let a = g.param("a", Shape::new(&[d, n], f));
    let bb = g.input("b", Shape::new(&[bsz, l, n], f));
    let cc = g.input("c", Shape::new(&[bsz, l, n], f));
    let y = g.add_node(
        rlx_ir::Op::SelectiveScan { state_size: n },
        vec![x, delta, a, bb, cc],
        Shape::new(&[bsz, l, d], f),
    );
    g.set_outputs(vec![y]);

    let mut exec = rlx_tpu::TpuExecutable::compile(g);
    // x[t] = 1 for all positions. delta = 1. a = 0 (decay = e^0 = 1).
    // b[t] = 1, c[t] = 1, state_size n = 1.
    //   state[t] = state[t-1] * 1 + 1 * 1 * 1 = state[t-1] + 1
    //   y[t] = state[t] * 1 = t + 1
    let xs = vec![1.0_f32; bsz * l * d];
    let deltas = vec![1.0_f32; bsz * l * d];
    let as_ = vec![0.0_f32; d * n];
    let bs = vec![1.0_f32; bsz * l * n];
    let cs = vec![1.0_f32; bsz * l * n];
    exec.set_param("a", &as_);
    let outs = exec.run(&[("x", &xs), ("delta", &deltas), ("b", &bs), ("c", &cs)]);
    // Output per (B=0, t, d): t+1, for both d's.
    let want = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    for (i, (got, w)) in outs[0].iter().zip(want.iter()).enumerate() {
        assert!((*got - *w).abs() < 1e-3, "ssm[{i}]: got {got}, want {w}");
    }
}
