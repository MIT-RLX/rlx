// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The graph adapters against the host definitions, on whichever backend is
//! selected.
//!
//! `RLX_FORCE_DEVICE=metal|mlx|gpu|coreml|cuda` moves the graph. The host
//! functions in `adapters` are the reference — they are what the parity tests
//! check — so agreement here means the accelerated path computes the same
//! update, which is the claim that matters for using these in training.

use rlx_peft::{
    LoraConfig, adapters::dora_init_magnitude, dora_weight, graph, ia3_apply, lora_delta,
};
use rlx_runtime::{Device, Session};

fn device() -> Device {
    rlx_ir::env::var("RLX_FORCE_DEVICE")
        .and_then(|s| rlx_runtime::parse_device(&s).ok())
        .unwrap_or(Device::Cpu)
}

fn mat(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s >> 30;
            s = s.wrapping_mul(0xBF58476D1CE4E5B9);
            s ^= s >> 27;
            s = s.wrapping_mul(0x94D049BB133111EB);
            s ^= s >> 31;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0f64, f64::max);
    assert!(
        worst < tol,
        "{what}: max |delta| {worst:.3e} (tol {tol:.0e})"
    );
    eprintln!("  {what:<22} ok  max |delta| {worst:.2e}");
}

#[test]
fn lora_graph_matches_the_host_definition() {
    let dev = device();
    eprintln!("[rlx-peft] device = {dev:?}");
    let (i, o, r) = (12usize, 8usize, 4usize);
    let cfg = LoraConfig { r, alpha: 16.0 };
    let (a, b) = (mat(r * i, 1), mat(o * r, 2));

    let want = lora_delta(&f64s(&a), &f64s(&b), i, o, cfg).unwrap();
    let mut c = Session::new(dev).compile(graph::lora_delta_graph(i, o, cfg));
    let out = c.run(&[("a", &a), ("b", &b)]);
    close(&out[0], &want, 2e-5, "lora delta");
}

#[test]
fn ia3_graph_matches_the_host_definition() {
    let dev = device();
    let (rows, o) = (6usize, 9usize);
    let (y, l) = (mat(rows * o, 3), mat(o, 4));
    let want = ia3_apply(&f64s(&y), &f64s(&l), rows, o).unwrap();
    let mut c = Session::new(dev).compile(graph::ia3_graph(rows, o));
    let out = c.run(&[("y", &y), ("l", &l)]);
    close(&out[0], &want, 2e-6, "ia3 rescale");
}

#[test]
fn adalora_graph_matches_the_host_definition() {
    let dev = device();
    let (i, o, r) = (10usize, 6usize, 3usize);
    let cfg = LoraConfig { r, alpha: 12.0 };
    let (p, q) = (mat(o * r, 5), mat(r * i, 6));
    // A zeroed singular value: the normal state during rank annealing.
    let lam = vec![0.8f32, 0.0, 1.3];
    let want = rlx_peft::adalora_delta(&f64s(&p), &f64s(&lam), &f64s(&q), i, o, cfg).unwrap();
    let mut c = Session::new(dev).compile(graph::adalora_delta_graph(i, o, cfg));
    let out = c.run(&[("p", &p), ("lambda", &lam), ("q", &q)]);
    close(&out[0], &want, 2e-5, "adalora delta");
}

#[test]
fn dora_graph_matches_the_host_definition() {
    let dev = device();
    let (i, o) = (11usize, 7usize);
    let w = mat(o * i, 7);
    let d = mat(o * i, 8);
    let m: Vec<f32> = dora_init_magnitude(&f64s(&w), i, o)
        .unwrap()
        .iter()
        .map(|v| *v as f32)
        .collect();
    let want = dora_weight(&f64s(&w), &f64s(&d), &f64s(&m), i, o).unwrap();
    let mut c = Session::new(dev).compile(graph::dora_graph(i, o));
    let out = c.run(&[("w", &w), ("delta", &d), ("m", &m)]);
    close(&out[0], &want, 2e-5, "dora weight");
}

/// The identity-at-initialisation property must survive the graph path too:
/// `B = 0` gives a zero update on every backend, so an adapted model starts
/// byte-identical to its base.
#[test]
fn the_graph_path_is_also_identity_at_initialisation() {
    let dev = device();
    let (i, o, r) = (9usize, 5usize, 3usize);
    let cfg = LoraConfig { r, alpha: 8.0 };
    let a = mat(r * i, 9);
    let b = vec![0f32; o * r];
    let mut c = Session::new(dev).compile(graph::lora_delta_graph(i, o, cfg));
    let out = c.run(&[("a", &a), ("b", &b)]);
    assert!(
        out[0].iter().all(|v| *v == 0.0),
        "B=0 must give an exactly zero delta"
    );
}
