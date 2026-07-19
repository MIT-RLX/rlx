//! Metal `x * x` (same input twice) vs CPU — Kokoro AdaIN variance pattern.
#![cfg(all(target_os = "macos", feature = "metal"))]
use rlx_ir::op::{BinaryOp, Op};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn run_self_mul(n0: usize, n1: usize, n2: usize, scale: f32) {
    if !is_available(Device::Metal) {
        eprintln!("skip: no Metal device");
        return;
    }
    let n = n0 * n1 * n2;
    let x: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * scale - 0.5).collect();
    let mut g = Graph::new("self_mul");
    let a = g.input("x", Shape::new(&[n0, n1, n2], DType::F32));
    let y = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![a, a],
        Shape::new(&[n0, n1, n2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let slots = [("x", x.as_slice())];
    let run = |dev| {
        Session::new(dev)
            .compile(g.clone())
            .run(&slots)
            .pop()
            .unwrap()
    };
    let cpu = run(Device::Cpu);
    let metal = run(Device::Metal);
    let maxd = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let peak = metal.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!("self-mul [{n0},{n1},{n2}] scale={scale} maxd={maxd:.3e} metal_peak={peak:.3e}");
    assert!(peak > 0.0, "Metal self-mul produced silence");
    assert!(maxd < 1e-4, "Metal self-mul mismatch {maxd}");
}

#[test]
fn metal_self_mul_small() {
    run_self_mul(1, 32, 64, 0.01);
}

#[test]
fn metal_self_mul_adain_shape() {
    // Kokoro generator AdaIN variance: 128 channels × ~6k frames
    run_self_mul(1, 128, 6241, 0.01);
}

#[test]
fn metal_self_mul_large_values() {
    // Observed Sub magnitudes near the silent Mul (~1e5)
    run_self_mul(1, 128, 512, 100.0);
}
