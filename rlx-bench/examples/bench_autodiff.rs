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

//! Autodiff backward-op characterization: correctness + speed + precision
//! between CPU (reference) and MLX, on realistic training-step shapes.
//!
//! Run on Apple Silicon:
//! ```sh
//! scripts/check-throttle.sh && \
//!   cargo run -p rlx-bench --release --example bench_autodiff --features mlx
//! ```
//!
//! Reports per (op × shape × device):
//!   - mean / median / min run time (`rlx_ir::Tick`-based)
//!   - max-abs and RMS divergence vs CPU output (precision)
//!
//! The CPU output is the reference. MLX divergence ≤ ~1e-5 over the
//! full output is the baseline for "precise enough"; bigger gaps point
//! at numerical drift (e.g. erf approximation, scale recomputation
//! ordering).

use rlx_driver::Device;
use rlx_ir::op::{Activation, ReduceOp, ScaleMode, SteKind};
use rlx_ir::{DType, Graph, Op, Shape, Tick};
use rlx_runtime::Session;

#[derive(Clone, Copy, Debug)]
struct DiffStats {
    max_abs: f32,
    rms: f32,
    n: usize,
}

fn diff_stats(a: &[f32], b: &[f32]) -> DiffStats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut sq = 0f64;
    let mut nan_count = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.is_nan() || y.is_nan() {
            nan_count += 1;
            continue;
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sq += (d as f64) * (d as f64);
    }
    if nan_count > 0 {
        eprintln!(
            "  WARNING: {nan_count}/{} values were NaN — excluded from stats",
            a.len()
        );
    }
    let counted = a.len() - nan_count;
    let rms = if counted == 0 {
        0.0
    } else {
        (sq / counted as f64).sqrt() as f32
    };
    DiffStats {
        max_abs,
        rms,
        n: counted,
    }
}

fn run_once(
    g: Graph,
    params: &[(&str, &[f32])],
    inputs: &[(&str, &[f32])],
    device: Device,
    warmup: usize,
    runs: usize,
) -> (Vec<f32>, Vec<u64>) {
    let mut compiled = Session::new(device).compile(g);
    for (k, v) in params {
        compiled.set_param(k, v);
    }

    // Warm-ups untimed.
    for _ in 0..warmup {
        let _ = compiled.run(inputs);
    }

    let mut samples = Vec::with_capacity(runs);
    let mut last: Vec<f32> = Vec::new();
    for _ in 0..runs {
        let t0 = Tick::now();
        let outs = compiled.run(inputs);
        let elapsed = Tick::now().elapsed_ns(t0);
        samples.push(elapsed);
        last = outs.into_iter().next().unwrap();
    }
    (last, samples)
}

fn stat(samples: &[u64]) -> (u64, u64, u64) {
    let mut s = samples.to_vec();
    s.sort_unstable();
    let mean = (s.iter().map(|&v| v as u128).sum::<u128>() / s.len() as u128) as u64;
    let median = s[s.len() / 2];
    let min = *s.first().unwrap();
    (mean, median, min)
}

struct Case<'a> {
    label: &'a str,
    build: Box<dyn Fn() -> (Graph, Vec<(String, Vec<f32>)>, Vec<(String, Vec<f32>)>)>,
}

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2654435761)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Mask to mantissa bits, force exponent=127 → value in [1.0, 2.0).
            // Sign bit comes from the next-higher bit so we get the full
            // [-2.0, -1.0) ∪ [1.0, 2.0) range.
            let mantissa = ((s >> 9) as u32) & 0x007F_FFFF;
            let sign = ((s >> 32) as u32) & 0x8000_0000;
            let bits = sign | 0x3f80_0000 | mantissa;
            f32::from_bits(bits) - 1.5 // shift to [-2.5, -1.5) ∪ [-0.5, 0.5)
        })
        .collect()
}

fn case_relu_backward(n: usize, h: usize, w: usize) -> Case<'static> {
    let label: &'static str = Box::leak(format!("ReluBackward [{n}, {h}, {w}]").into_boxed_str());
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("relu_bwd");
            let x = g.input("x", Shape::new(&[n, h, w], DType::F32));
            let dy = g.input("dy", Shape::new(&[n, h, w], DType::F32));
            let r = g.relu_backward(x, dy);
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("x".into(), pseudo(n * h * w, 1)),
                ("dy".into(), pseudo(n * h * w, 2)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_layernorm_backward_input(b: usize, s: usize, d: usize) -> Case<'static> {
    let label: &'static str =
        Box::leak(format!("LayerNormBackwardInput [{b}, {s}, {d}]").into_boxed_str());
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("ln_bwd_in");
            let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
            let gm = g.input("gamma", Shape::new(&[d], DType::F32));
            let dy = g.input("dy", Shape::new(&[b, s, d], DType::F32));
            let r = g.layer_norm_backward_input(x, gm, dy, -1, 1e-5);
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("x".into(), pseudo(b * s * d, 11)),
                ("gamma".into(), pseudo(d, 12)),
                ("dy".into(), pseudo(b * s * d, 13)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_softmax_ce_backward(n: usize, c: usize) -> Case<'static> {
    let label: &'static str =
        Box::leak(format!("SoftmaxCrossEntropyBackward [{n}, {c}]").into_boxed_str());
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("sce_bwd");
            let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
            let labels = g.input("labels", Shape::new(&[n], DType::F32));
            let dl = g.input("d_loss", Shape::new(&[n], DType::F32));
            let r = g.softmax_cross_entropy_backward(logits, labels, dl);
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("logits".into(), pseudo(n * c, 21)),
                ("labels".into(), (0..n).map(|i| (i % c) as f32).collect()),
                ("d_loss".into(), vec![1.0f32 / n as f32; n]),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_conv2d_backward_input(n: usize, ci: usize, h: usize, co: usize, k: usize) -> Case<'static> {
    let label: &'static str = Box::leak(
        format!("Conv2dBackwardInput [{n}, {ci}, {h}, {h}], k={k}, c_out={co}").into_boxed_str(),
    );
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("conv_bwd_in");
            let h_out = h; // s=1, p=1, k=3 → h_out=h
            let dy = g.input("dy", Shape::new(&[n, co, h_out, h_out], DType::F32));
            let w = g.input("w", Shape::new(&[co, ci, k, k], DType::F32));
            let r = g.conv2d_backward_input(
                dy,
                w,
                Shape::new(&[n, ci, h, h], DType::F32),
                vec![k, k],
                vec![1, 1],
                vec![1, 1],
                vec![1, 1],
                1,
            );
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("dy".into(), pseudo(n * co * h_out * h_out, 31)),
                ("w".into(), pseudo(co * ci * k * k, 32)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_conv2d_backward_weight(
    n: usize,
    ci: usize,
    h: usize,
    co: usize,
    k: usize,
) -> Case<'static> {
    let label: &'static str = Box::leak(
        format!("Conv2dBackwardWeight [{n}, {ci}, {h}, {h}], k={k}, c_out={co}").into_boxed_str(),
    );
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("conv_bwd_w");
            let h_out = h;
            let x = g.input("x", Shape::new(&[n, ci, h, h], DType::F32));
            let dy = g.input("dy", Shape::new(&[n, co, h_out, h_out], DType::F32));
            let r = g.conv2d_backward_weight(
                x,
                dy,
                Shape::new(&[co, ci, k, k], DType::F32),
                vec![k, k],
                vec![1, 1],
                vec![1, 1],
                vec![1, 1],
                1,
            );
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("x".into(), pseudo(n * ci * h * h, 41)),
                ("dy".into(), pseudo(n * co * h_out * h_out, 42)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_maxpool_backward(n: usize, c: usize, h: usize) -> Case<'static> {
    let label: &'static str =
        Box::leak(format!("MaxPool2dBackward [{n}, {c}, {h}, {h}], k=2, s=2").into_boxed_str());
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("pool_bwd");
            let h_out = h / 2;
            let x = g.input("x", Shape::new(&[n, c, h, h], DType::F32));
            let dy = g.input("dy", Shape::new(&[n, c, h_out, h_out], DType::F32));
            let r = g.maxpool2d_backward(x, dy, vec![2, 2], vec![2, 2], vec![0, 0]);
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("x".into(), pseudo(n * c * h * h, 51)),
                ("dy".into(), pseudo(n * c * h_out * h_out, 52)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_fakequant_perbatch(n: usize, c: usize, h: usize, w: usize, bits: u8) -> Case<'static> {
    let label: &'static str = Box::leak(
        format!("FakeQuantize PerBatch [{n}, {c}, {h}, {w}], bits={bits}").into_boxed_str(),
    );
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("fq");
            let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
            let q = g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(1),
                    ste: SteKind::Identity,
                    scale_mode: ScaleMode::PerBatch,
                },
                vec![x],
                Shape::new(&[n, c, h, w], DType::F32),
            );
            g.set_outputs(vec![q]);
            let inputs = vec![("x".into(), pseudo(n * c * h * w, 61))];
            (g, vec![], inputs)
        }),
    }
}

fn case_fakequant_backward(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    ste: SteKind,
    bits: u8,
) -> Case<'static> {
    let label: &'static str = Box::leak(
        format!("FakeQuantizeBackward({ste:?}) [{n}, {c}, {h}, {w}], bits={bits}").into_boxed_str(),
    );
    Case {
        label,
        build: Box::new(move || {
            let mut g = Graph::new("fq_bwd");
            let x = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
            let dy = g.input("dy", Shape::new(&[n, c, h, w], DType::F32));
            let r = g.add_node(
                Op::FakeQuantizeBackward {
                    bits,
                    axis: Some(1),
                    ste,
                },
                vec![x, dy],
                Shape::new(&[n, c, h, w], DType::F32),
            );
            g.set_outputs(vec![r]);
            let inputs = vec![
                ("x".into(), pseudo(n * c * h * w, 71)),
                ("dy".into(), pseudo(n * c * h * w, 72)),
            ];
            (g, vec![], inputs)
        }),
    }
}

fn case_end_to_end_conv_relu(n: usize, ci: usize, h: usize, co: usize) -> Case<'static> {
    let label: &'static str = Box::leak(
        format!("end-to-end conv→relu→mean grad [{n}, {ci}, {h}, {h}]→{co}").into_boxed_str(),
    );
    Case {
        label,
        build: Box::new(move || {
            let h_out = h; // s=1, p=1, k=3
            let mut fwd = Graph::new("e2e");
            let x = fwd.input("x", Shape::new(&[n, ci, h, h], DType::F32));
            let w = fwd.param("w", Shape::new(&[co, ci, 3, 3], DType::F32));
            let y = fwd.add_node(
                Op::Conv {
                    kernel_size: vec![3, 3],
                    stride: vec![1, 1],
                    padding: vec![1, 1],
                    dilation: vec![1, 1],
                    groups: 1,
                },
                vec![x, w],
                Shape::new(&[n, co, h_out, h_out], DType::F32),
            );
            let a = fwd.activation(
                Activation::Relu,
                y,
                Shape::new(&[n, co, h_out, h_out], DType::F32),
            );
            let loss = fwd.add_node(
                Op::Reduce {
                    op: ReduceOp::Mean,
                    axes: vec![0, 1, 2, 3],
                    keep_dim: false,
                },
                vec![a],
                Shape::new(&[], DType::F32),
            );
            fwd.set_outputs(vec![loss]);
            let bwd = rlx_opt::autodiff::grad_with_loss(&fwd, &[w]);
            let params = vec![("w".into(), pseudo(co * ci * 9, 81))];
            let inputs = vec![
                ("x".into(), pseudo(n * ci * h * h, 82)),
                ("d_output".into(), vec![1.0f32]),
            ];
            (bwd, params, inputs)
        }),
    }
}

fn run_case(case: &Case, devs: &[(&str, Device)], warmup: usize, runs: usize) {
    println!("\n## {}", case.label);
    let mut cpu_out: Option<Vec<f32>> = None;
    for &(label, dev) in devs {
        let (g, params_owned, inputs_owned) = (case.build)();
        let params: Vec<(&str, &[f32])> = params_owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let inputs: Vec<(&str, &[f32])> = inputs_owned
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let (out, samples) = run_once(g, &params, &inputs, dev, warmup, runs);
        let (mean, median, min) = stat(&samples);
        let to_us = |ns: u64| ns as f64 / 1000.0;
        let diff = match (label, &cpu_out) {
            ("cpu", _) => "(reference)".to_string(),
            (_, Some(cpu)) => {
                let d = diff_stats(cpu, &out);
                format!("max_abs={:.3e} rms={:.3e}  (n={})", d.max_abs, d.rms, d.n)
            }
            _ => "(no cpu reference)".to_string(),
        };
        println!(
            "  {:5}  mean={:>9.2}µs  median={:>9.2}µs  min={:>9.2}µs   {}",
            label,
            to_us(mean),
            to_us(median),
            to_us(min),
            diff
        );
        if label == "cpu" {
            cpu_out = Some(out);
        }
    }
}

fn main() {
    let devs: Vec<(&str, Device)> = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "mlx")]
    devs.push(("mlx", Device::Mlx));

    println!(
        "rlx autodiff bench / devices: {:?}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );
    println!("All shapes are realistic for ViT/ResNet-class training steps.");

    let warmup = 5;
    let runs = 50;

    let cases: Vec<Case> = vec![
        // Element-wise.
        case_relu_backward(16, 56, 56),
        // LayerNorm — a representative ViT block shape.
        case_layernorm_backward_input(16, 196, 768),
        // Softmax cross entropy on imagenet-class logits.
        case_softmax_ce_backward(64, 1000),
        // Conv backward — early/middle/late ResNet-ish stages.
        case_conv2d_backward_input(8, 64, 56, 64, 3),
        case_conv2d_backward_input(8, 256, 14, 256, 3),
        case_conv2d_backward_weight(8, 64, 56, 64, 3),
        case_conv2d_backward_weight(8, 256, 14, 256, 3),
        // Pool backward.
        case_maxpool_backward(8, 64, 56),
        // QAT.
        case_fakequant_perbatch(8, 64, 28, 28, 8),
        case_fakequant_perbatch(8, 64, 28, 28, 4),
        case_fakequant_backward(8, 64, 28, 28, SteKind::Identity, 8),
        case_fakequant_backward(8, 64, 28, 28, SteKind::ClippedIdentity, 4),
        case_fakequant_backward(8, 64, 28, 28, SteKind::Tanh, 4),
        case_fakequant_backward(8, 64, 28, 28, SteKind::HardTanh, 4),
        // Full grad_with_loss round-trip through a conv layer.
        case_end_to_end_conv_relu(8, 32, 28, 64),
    ];

    for case in &cases {
        run_case(case, &devs, warmup, runs);
    }
}
