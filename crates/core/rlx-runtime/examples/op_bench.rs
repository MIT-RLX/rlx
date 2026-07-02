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

//! Per-op microbenchmark — times each implemented op on CPU and Metal at
//! model-realistic shapes. Reports median and p95 per op + per backend, so
//! after each kernel/fusion change you can spot regressions immediately.
//!
//! Methodology: warmup runs are excluded; for each (op, backend), we run
//! `iters` forward passes back-to-back through one compiled graph (so we
//! measure pure kernel-dispatch + execute time, not graph compile time).
//! Inputs are pre-loaded once and reused; outputs are read once at the
//! end so we don't double-count the read-back.
//!
//! cargo run --release -p rlx-runtime --example op_bench --features metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};
    use std::time::Instant;

    let warmup = 5usize;
    let iters: usize = rlx_ir::env::var("RLX_BENCH_ITERS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    fn time_one(
        device: Device,
        build: impl Fn() -> Graph,
        params: &[(&str, Vec<f32>)],
        inputs: &[(&str, Vec<f32>)],
        warmup: usize,
        iters: usize,
    ) -> (f64, f64) {
        let session = Session::new(device);
        let mut compiled = session.compile(build());
        for (n, d) in params {
            compiled.set_param(n, d);
        }
        let cpu_inputs: Vec<(&str, &[f32])> =
            inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        for _ in 0..warmup {
            let _ = compiled.run(&cpu_inputs);
        }
        let mut samples_us: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = compiled.run(&cpu_inputs);
            samples_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples_us[samples_us.len() / 2];
        let p95 = samples_us[(samples_us.len() * 95 / 100).min(samples_us.len() - 1)];
        (median, p95)
    }

    let f = DType::F32;
    let det = |n: usize, mul: usize, modulo: usize, offset: f32| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * mul + 7) % modulo) as f32 / modulo as f32 + offset)
            .collect()
    };

    println!(
        "{:<40} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "Op (shape)", "CPU med (µs)", "CPU p95", "Metal med", "Metal p95", "speedup"
    );
    println!("{:-<116}", "");

    let bench = |name: String,
                 build: Box<dyn Fn() -> Graph>,
                 params: Vec<(&str, Vec<f32>)>,
                 inputs: Vec<(&str, Vec<f32>)>| {
        let (cpu_med, cpu_p95) = time_one(Device::Cpu, &build, &params, &inputs, warmup, iters);
        let (m_med, m_p95) = time_one(Device::Metal, &build, &params, &inputs, warmup, iters);
        let speedup = cpu_med / m_med;
        println!(
            "{name:<40} {cpu_med:>14.2} {cpu_p95:>14.2} {m_med:>14.2} {m_p95:>14.2} {speedup:>9.2}x"
        );
    };

    // ── Linear algebra ──────────────────────────────────────────────
    // BERT-base shape: m=60 (b=4, s=15) × k=768 × n=2304 (QKV).
    bench(
        "MatMul (60, 768) × (768, 2304)".into(),
        Box::new(|| {
            let mut g = Graph::new("mm");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let w = g.param("w", Shape::new(&[768, 2304], f));
            let y = g.matmul(x, w, Shape::new(&[60, 2304], f));
            g.set_outputs(vec![y]);
            g
        }),
        vec![("w", det(768 * 2304, 17, 31, 0.001))],
        vec![("x", det(60 * 768, 13, 23, 0.001))],
    );

    // Nomic FFN down: m=12 × k=3072 × n=768
    bench(
        "MatMul (12, 3072) × (3072, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("mm");
            let x = g.input("x", Shape::new(&[12, 3072], f));
            let w = g.param("w", Shape::new(&[3072, 768], f));
            let y = g.matmul(x, w, Shape::new(&[12, 768], f));
            g.set_outputs(vec![y]);
            g
        }),
        vec![("w", det(3072 * 768, 17, 31, 0.001))],
        vec![("x", det(12 * 3072, 13, 23, 0.001))],
    );

    // ── Element-wise (BERT-typical) ─────────────────────────────────
    let elem_shape = [60usize, 768];
    let elem_n = elem_shape[0] * elem_shape[1];
    for (act_name, act) in &[
        ("gelu", Activation::Gelu),
        ("silu", Activation::Silu),
        ("relu", Activation::Relu),
    ] {
        let act = *act;
        bench(
            format!("Activation::{act_name} (60, 768)"),
            Box::new(move || {
                let mut g = Graph::new("act");
                let x = g.input("x", Shape::new(&[60, 768], f));
                let y = g.activation(act, x, Shape::new(&[60, 768], f));
                g.set_outputs(vec![y]);
                g
            }),
            vec![],
            vec![("x", det(elem_n, 13, 23, 0.1))],
        );
    }

    bench(
        "Binary::Add (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("bin");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let y = g.input("y", Shape::new(&[60, 768], f));
            let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[60, 768], f));
            g.set_outputs(vec![z]);
            g
        }),
        vec![],
        vec![
            ("x", det(elem_n, 7, 11, 0.1)),
            ("y", det(elem_n, 13, 19, 0.1)),
        ],
    );

    // ── Norms ───────────────────────────────────────────────────────
    bench(
        "LayerNorm (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("ln");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let gamma = g.param("g", Shape::new(&[768], f));
            let beta = g.param("b", Shape::new(&[768], f));
            let y = g.ln(x, gamma, beta, 1e-5);
            g.set_outputs(vec![y]);
            g
        }),
        vec![("g", vec![1.0; 768]), ("b", vec![0.0; 768])],
        vec![("x", det(elem_n, 13, 23, 0.5))],
    );

    bench(
        "RmsNorm (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("rms");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let gamma = g.param("g", Shape::new(&[768], f));
            let beta = g.param("b", Shape::new(&[768], f));
            let y = g.rms_norm(x, gamma, beta, 1e-5);
            g.set_outputs(vec![y]);
            g
        }),
        vec![("g", vec![1.0; 768]), ("b", vec![0.0; 768])],
        vec![("x", det(elem_n, 13, 23, 0.5))],
    );

    // ── Reduce / Softmax ────────────────────────────────────────────
    bench(
        "Reduce::Sum (60, 768) → (60,)".into(),
        Box::new(|| {
            let mut g = Graph::new("red");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let y = g.add_node(
                Op::Reduce {
                    op: ReduceOp::Sum,
                    axes: vec![1],
                    keep_dim: false,
                },
                vec![x],
                Shape::new(&[60], f),
            );
            g.set_outputs(vec![y]);
            g
        }),
        vec![],
        vec![("x", det(elem_n, 7, 17, 0.1))],
    );

    bench(
        "Softmax (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("sm");
            let x = g.input("x", Shape::new(&[60, 768], f));
            let y = g.add_node(Op::Softmax { axis: -1 }, vec![x], Shape::new(&[60, 768], f));
            g.set_outputs(vec![y]);
            g
        }),
        vec![],
        vec![("x", det(elem_n, 7, 17, 0.1))],
    );

    // ── Data movement ───────────────────────────────────────────────
    bench(
        "Transpose (768, 60) → (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("tr");
            let x = g.input("x", Shape::new(&[768, 60], f));
            let y = g.add_node(
                Op::Transpose { perm: vec![1, 0] },
                vec![x],
                Shape::new(&[60, 768], f),
            );
            g.set_outputs(vec![y]);
            g
        }),
        vec![],
        vec![("x", det(elem_n, 7, 17, 0.1))],
    );

    bench(
        "Gather axis=0 (vocab=30k, idx=60)".into(),
        Box::new(|| {
            let mut g = Graph::new("g0");
            let table = g.param("t", Shape::new(&[30000, 768], f));
            let idx = g.input("idx", Shape::new(&[60], f));
            let y = g.gather_(table, idx, 0);
            g.set_outputs(vec![y]);
            g
        }),
        vec![("t", det(30000 * 768, 7, 31, 0.001))],
        vec![("idx", (0..60).map(|i| (i * 17 % 30000) as f32).collect())],
    );

    bench(
        "Concat last-axis (60, 768) ⊕ (60, 768)".into(),
        Box::new(|| {
            let mut g = Graph::new("ct");
            let a = g.param("a", Shape::new(&[60, 768], f));
            let b = g.param("b", Shape::new(&[60, 768], f));
            let y = g.add_node(
                Op::Concat { axis: 1 },
                vec![a, b],
                Shape::new(&[60, 1536], f),
            );
            g.set_outputs(vec![y]);
            g
        }),
        vec![
            ("a", det(elem_n, 7, 17, 0.1)),
            ("b", det(elem_n, 11, 19, 0.1)),
        ],
        vec![],
    );

    // ── Attention block ─────────────────────────────────────────────
    // Realistic small-batch SDPA: b=1, s=15, h=768 = 12 heads × 64.
    bench(
        "Attention SDPA (b=1, s=15, h=768)".into(),
        Box::new(|| {
            let mut g = Graph::new("attn");
            let nh = 12;
            let dh = 64;
            let h = nh * dh;
            let q = g.input("q", Shape::new(&[1, 15, h], f));
            let k = g.input("k", Shape::new(&[1, 15, h], f));
            let v = g.input("v", Shape::new(&[1, 15, h], f));
            let mask = g.input("mask", Shape::new(&[1, 15], f));
            let y = g.attention_(q, k, v, mask, nh, dh);
            g.set_outputs(vec![y]);
            g
        }),
        vec![],
        vec![
            ("q", det(15 * 768, 5, 13, 0.1)),
            ("k", det(15 * 768, 7, 17, 0.1)),
            ("v", det(15 * 768, 11, 19, 0.1)),
            ("mask", vec![1.0; 15]),
        ],
    );

    // ── Fused SwiGLU pipeline ───────────────────────────────────────
    // Two matmuls sharing input → FuseSharedInputMatMul + FuseSwiGLU.
    bench(
        "SwiGLU (60, 768) → (60, 2048)".into(),
        Box::new(|| {
            let m = 60usize;
            let k = 768usize;
            let n = 2048usize;
            let mut g = Graph::new("swiglu");
            let x = g.input("x", Shape::new(&[m, k], f));
            let w_up = g.param("w_up", Shape::new(&[k, n], f));
            let w_gate = g.param("w_gate", Shape::new(&[k, n], f));
            let up_mm = g.matmul(x, w_up, Shape::new(&[m, n], f));
            let gate_mm = g.matmul(x, w_gate, Shape::new(&[m, n], f));
            let gate = g.activation(Activation::Silu, gate_mm, Shape::new(&[m, n], f));
            let y = g.binary(BinaryOp::Mul, up_mm, gate, Shape::new(&[m, n], f));
            g.set_outputs(vec![y]);
            g
        }),
        vec![
            ("w_up", det(768 * 2048, 11, 31, 0.001)),
            ("w_gate", det(768 * 2048, 13, 29, 0.001)),
        ],
        vec![("x", det(60 * 768, 7, 23, 0.1))],
    );

    println!();
    println!("(all timings include the run() overhead — graph dispatch, encoder");
    println!(" setup on Metal, output read-back. Use RLX_BENCH_ITERS=N to vary.)");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("op_bench requires --features metal on macOS");
}
