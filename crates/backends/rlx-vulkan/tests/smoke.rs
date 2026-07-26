// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Smoke tests for the native Vulkan backend. They run real compute when a
//! Vulkan device is reachable and otherwise assert the graceful-unavailable
//! path, so the suite is green on hosts with no driver (e.g. macOS without
//! MoltenVK, CI runners).

use rlx_ir::op::{Activation, BinaryOp, CmpOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_vulkan::backend::VulkanExecutable;

fn s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

#[test]
fn unavailable_is_graceful() {
    // Must never panic regardless of host. On a driverless host it returns
    // false; with a driver it returns true and names the device.
    let avail = rlx_vulkan::is_available();
    if avail {
        assert!(rlx_vulkan::device_name().is_some());
    } else {
        assert!(rlx_vulkan::device_name().is_none());
        eprintln!("rlx-vulkan: no Vulkan device on this host — skipping compute tests");
    }
}

#[test]
fn elementwise_add_and_relu() {
    if !rlx_vulkan::is_available() {
        return; // covered by `unavailable_is_graceful`
    }
    eprintln!(
        "[rlx-vulkan] elementwise validation on device: {:?}",
        rlx_vulkan::device_name()
    );
    let mut g = Graph::new("add_relu");
    let a = g.input("a", s(&[4]));
    let b = g.input("b", s(&[4]));
    let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], s(&[4]));
    let out = g.add_node(Op::Activation(Activation::Relu), vec![sum], s(&[4]));
    g.set_outputs(vec![out]);

    let mut exe = VulkanExecutable::compile(g);
    let res = exe.run(&[
        ("a", &[1.0, -5.0, 3.0, -2.0]),
        ("b", &[0.5, 1.0, -1.0, -1.0]),
    ]);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0], vec![1.5, 0.0, 2.0, 0.0]);
}

#[test]
fn matmul_2x3_3x2() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let mut g = Graph::new("matmul");
    let a = g.input("a", s(&[2, 3]));
    let b = g.input("b", s(&[3, 2]));
    let out = g.add_node(Op::MatMul, vec![a, b], s(&[2, 2]));
    g.set_outputs(vec![out]);

    let mut exe = VulkanExecutable::compile(g);
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // AB = [[58,64],[139,154]]
    let res = exe.run(&[
        ("a", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("b", &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    ]);
    eprintln!(
        "[rlx-vulkan] matmul on {:?} → {:?}",
        rlx_vulkan::device_name(),
        res[0]
    );
    assert_eq!(res[0], vec![58.0, 64.0, 139.0, 154.0]);
}

/// Matmul throughput micro-benchmark, scalar vs tiled (pick the kernel with
/// `RLX_VULKAN_MATMUL=scalar|tiled`; default `auto` = tiled on native drivers).
/// `#[ignore]`d so it never runs in the normal suite — invoke explicitly:
/// `RLX_VULKAN_MATMUL=tiled cargo test -p rlx-vulkan --test smoke bench_matmul \
///  -- --ignored --nocapture`. Per-call overhead (input upload + fence) is
/// included and identical for both kernels, so the A/B ratio is fair.
#[test]
#[ignore]
fn bench_matmul() {
    if !rlx_vulkan::is_available() {
        eprintln!("[bench] no Vulkan device — skip");
        return;
    }
    let label = std::env::var("RLX_VULKAN_MATMUL").unwrap_or_else(|_| "auto".into());
    eprintln!(
        "[bench] device={:?} kernel={label}",
        rlx_vulkan::device_name()
    );
    let shapes: &[(usize, usize, usize)] = &[
        (96, 384, 1152), // BERT QKV
        (512, 512, 512),
        (1024, 1024, 1024), // compute-dominated
    ];
    let iters = 50usize;
    for &(m, k, n) in shapes {
        let mut g = Graph::new("mm");
        let a = g.input("a", s(&[m, k]));
        let b = g.input("b", s(&[k, n]));
        let out = g.add_node(Op::MatMul, vec![a, b], s(&[m, n]));
        g.set_outputs(vec![out]);
        let mut exe = VulkanExecutable::compile(g);
        let av: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 0.01 - 0.5).collect();
        for _ in 0..3 {
            let _ = exe.run(&[("a", &av), ("b", &bv)]); // warmup
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = exe.run(&[("a", &av), ("b", &bv)]);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (ms * 1e6);
        eprintln!("[bench] {m:>5}x{k:>5}x{n:>5}  {ms:>8.3} ms/iter  {gflops:>8.1} GFLOP/s");
    }
}

/// Matmul vs a CPU fp32 reference on 16-aligned dims. Runs the selected kernel
/// (default `tiled`, exact → tol 1e-3); under `RLX_VULKAN_MATMUL=coop` on a
/// cooperative-matrix device it validates the tensor-core kernel with an
/// f16-operand-appropriate tolerance. Green in both modes, so it's safe in the
/// normal suite and doubles as the coop correctness check when forced.
#[test]
fn matmul_matches_cpu_reference() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let kernel = std::env::var("RLX_VULKAN_MATMUL").unwrap_or_else(|_| "auto".into());
    // Shape matrix: 16-aligned (coop-eligible), K-unaligned (coop zero-pads its
    // last K-tile), and non-square + M/N-unaligned (coop routes back to the
    // fully general tiled kernel). Exercises every kernel's edge handling.
    let shapes: &[(usize, usize, usize)] = &[(32, 48, 64), (32, 50, 64), (30, 17, 45)];
    for &(m, k, n) in shapes {
        let mut g = Graph::new("mm");
        let a = g.input("a", s(&[m, k]));
        let b = g.input("b", s(&[k, n]));
        let out = g.add_node(Op::MatMul, vec![a, b], s(&[m, n]));
        g.set_outputs(vec![out]);
        let av: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();

        let mut exe = VulkanExecutable::compile(g);
        let got = exe.run(&[("a", &av), ("b", &bv)]);

        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += av[i * k + p] * bv[p * n + j];
                }
                want[i * n + j] = acc;
            }
        }
        // coop (f16 operands) only runs on 16-aligned M/N; otherwise the exact
        // fp32 tiled/scalar kernel runs, so tighten the tolerance to match.
        let coop_used = kernel == "coop" && m % 16 == 0 && n % 16 == 0;
        let tol = if coop_used { 5e-2 } else { 1e-3 };
        let maxerr = got[0]
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[matmul-check] kernel={kernel} {m}x{k}x{n} coop_used={coop_used} max|Δ|={maxerr:.5} tol={tol}"
        );
        assert!(
            maxerr < tol,
            "matmul {m}x{k}x{n} vs cpu max|Δ|={maxerr} exceeds {tol}"
        );
    }
}

/// Exact-value checks for the shape / reduce / softmax kernels (no CPU oracle
/// needed), so the suite is meaningful as a standalone device validation —
/// this is what the Linux/lavapipe Docker container runs.
#[test]
fn transpose_reduce_narrow_softmax() {
    if !rlx_vulkan::is_available() {
        return;
    }

    // transpose [2,3] -> [3,2]
    let mut g = Graph::new("tr");
    let x = g.input("x", s(&[2, 3]));
    let o = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![x], s(&[3, 2]));
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(r[0], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // reduce-sum [2,3] over last axis -> [2]
    let mut g = Graph::new("red");
    let x = g.input("x", s(&[2, 3]));
    let o = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![1],
            keep_dim: false,
        },
        vec![x],
        s(&[2]),
    );
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(r[0], vec![6.0, 15.0]);

    // narrow [2,4] axis 1 start 1 len 2 -> [2,2]
    let mut g = Graph::new("nar");
    let x = g.input("x", s(&[2, 4]));
    let o = g.add_node(
        Op::Narrow {
            axis: 1,
            start: 1,
            len: 2,
        },
        vec![x],
        s(&[2, 2]),
    );
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &[0., 1., 2., 3., 4., 5., 6., 7.])]);
    assert_eq!(r[0], vec![1.0, 2.0, 5.0, 6.0]);

    // softmax of equal logits -> uniform
    let mut g = Graph::new("sm");
    let x = g.input("x", s(&[3]));
    let o = g.add_node(Op::Softmax { axis: -1 }, vec![x], s(&[3]));
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &[0.0, 0.0, 0.0])]);
    for v in &r[0] {
        assert!((v - 1.0 / 3.0).abs() < 1e-6, "softmax uniform: {v}");
    }
    eprintln!(
        "[rlx-vulkan] shape/reduce/softmax kernels OK on {:?}",
        rlx_vulkan::device_name()
    );
}

/// Run a single-output graph and return its output vec (device required).
fn run1(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    VulkanExecutable::compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .unwrap()
}

/// Broader exact-value coverage for the elementwise / indexing / shape kernels,
/// so the Linux/lavapipe Docker run validates more than the hot path.
#[test]
fn more_ops_exact() {
    if !rlx_vulkan::is_available() {
        return;
    }

    // binary sub: [5,5,5] - [1,2,3]
    let mut g = Graph::new("sub");
    let a = g.input("a", s(&[3]));
    let b = g.input("b", s(&[3]));
    let o = g.add_node(Op::Binary(BinaryOp::Sub), vec![a, b], s(&[3]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("a", &[5., 5., 5.]), ("b", &[1., 2., 3.])]),
        vec![4., 3., 2.]
    );

    // binary max
    let mut g = Graph::new("max");
    let a = g.input("a", s(&[3]));
    let b = g.input("b", s(&[3]));
    let o = g.add_node(Op::Binary(BinaryOp::Max), vec![a, b], s(&[3]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("a", &[1., 5., 2.]), ("b", &[3., 1., 4.])]),
        vec![3., 5., 4.]
    );

    // unary abs
    let mut g = Graph::new("abs");
    let x = g.input("x", s(&[3]));
    let o = g.add_node(Op::Activation(Activation::Abs), vec![x], s(&[3]));
    g.set_outputs(vec![o]);
    assert_eq!(run1(g, &[("x", &[-1., 2., -3.])]), vec![1., 2., 3.]);

    // where: cond ? a : b
    let mut g = Graph::new("where");
    let c = g.input("c", s(&[3]));
    let a = g.input("a", s(&[3]));
    let b = g.input("b", s(&[3]));
    let o = g.add_node(Op::Where, vec![c, a, b], s(&[3]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(
            g,
            &[
                ("c", &[1., 0., 1.]),
                ("a", &[10., 20., 30.]),
                ("b", &[7., 8., 9.])
            ]
        ),
        vec![10., 8., 30.]
    );

    // compare lt -> cast to f32
    let mut g = Graph::new("cmp");
    let a = g.input("a", s(&[3]));
    let b = g.input("b", s(&[3]));
    let cmp = g.add_node(
        Op::Compare(CmpOp::Lt),
        vec![a, b],
        Shape::new(&[3], DType::Bool),
    );
    let o = g.add_node(Op::Cast { to: DType::F32 }, vec![cmp], s(&[3]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("a", &[1., 5., 2.]), ("b", &[3., 3., 3.])]),
        vec![1., 0., 1.]
    );

    // gather rows: table[3,2], idx [2,0]
    let mut g = Graph::new("gat");
    let t = g.input("t", s(&[3, 2]));
    let i = g.input("i", s(&[2]));
    let o = g.add_node(Op::Gather { axis: 0 }, vec![t, i], s(&[2, 2]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("t", &[1., 2., 3., 4., 5., 6.]), ("i", &[2., 0.])]),
        vec![5., 6., 1., 2.]
    );

    // expand [1,3] -> [2,3]
    let mut g = Graph::new("exp");
    let x = g.input("x", s(&[1, 3]));
    let o = g.add_node(
        Op::Expand {
            target_shape: vec![2, 3],
        },
        vec![x],
        s(&[2, 3]),
    );
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("x", &[7., 8., 9.])]),
        vec![7., 8., 9., 7., 8., 9.]
    );

    // concat axis 0: [1,2] ++ [1,2] -> [2,2]
    let mut g = Graph::new("cat");
    let a = g.input("a", s(&[1, 2]));
    let b = g.input("b", s(&[1, 2]));
    let o = g.add_node(Op::Concat { axis: 0 }, vec![a, b], s(&[2, 2]));
    g.set_outputs(vec![o]);
    assert_eq!(
        run1(g, &[("a", &[1., 2.]), ("b", &[3., 4.])]),
        vec![1., 2., 3., 4.]
    );

    eprintln!(
        "[rlx-vulkan] binary/unary/where/compare/gather/expand/concat OK on {:?}",
        rlx_vulkan::device_name()
    );
}

/// Native `cum_scan` (CumProd / CumMax) last-axis parity against a hand-rolled
/// reference. Runs on lavapipe in the Docker container.
#[test]
fn cum_scan_matches_reference() {
    if !rlx_vulkan::is_available() {
        return;
    }
    let x = vec![1.5f32, 0.5, 2.0, 0.8, -1.0, 3.0, 2.0, 0.5];

    // Inclusive cumprod over last axis [2,4].
    let mut g = Graph::new("cumprod");
    let inp = g.input("x", s(&[2, 4]));
    let o = g.add_node(
        Op::CumProd {
            axis: -1,
            exclusive: false,
        },
        vec![inp],
        s(&[2, 4]),
    );
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &x)]);
    assert_eq!(r[0], vec![1.5, 0.75, 1.5, 1.2, -1.0, -3.0, -6.0, -3.0]);

    // Inclusive cummax over last axis [2,4].
    let mut g = Graph::new("cummax");
    let inp = g.input("x", s(&[2, 4]));
    let o = g.add_node(
        Op::CumMax {
            axis: -1,
            exclusive: false,
        },
        vec![inp],
        s(&[2, 4]),
    );
    g.set_outputs(vec![o]);
    let r = VulkanExecutable::compile(g).run(&[("x", &x)]);
    assert_eq!(r[0], vec![1.5, 1.5, 2.0, 2.0, -1.0, 3.0, 3.0, 3.0]);
}
