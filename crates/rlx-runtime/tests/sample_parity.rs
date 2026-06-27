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

//! Cross-backend parity for `Op::Sample` (top-k / top-p / temperature
//! token sampling). CPU is the reference; with a fixed seed the sampled
//! index is deterministic, so GPU backends must return the identical row.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_sample_graph(b: usize, v: usize, top_k: usize, top_p: f32, temp: f32, seed: u64) -> Graph {
    let mut g = Graph::new("sample");
    let logits = g.input("logits", Shape::new(&[b, v], DType::F32));
    let y = g.sample(
        logits,
        top_k,
        top_p,
        temp,
        seed,
        Shape::new(&[b], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn logits_data(b: usize, v: usize) -> Vec<f32> {
    // Deterministic, non-uniform logits so top-k/top-p actually bite.
    (0..b * v)
        .map(|i| {
            let r = (i % v) as f32;
            ((r * 0.37).sin() * 3.0) + ((i / v) as f32) * 0.1
        })
        .collect()
}

#[allow(dead_code)]
fn run_on(
    device: Device,
    b: usize,
    v: usize,
    top_k: usize,
    top_p: f32,
    temp: f32,
    seed: u64,
) -> Vec<f32> {
    let logits = logits_data(b, v);
    let mut exe = Session::new(device).compile(build_sample_graph(b, v, top_k, top_p, temp, seed));
    exe.run(&[("logits", logits.as_slice())]).pop().unwrap()
}

#[allow(dead_code)]
fn cases() -> Vec<(&'static str, usize, usize, usize, f32, f32, u64)> {
    vec![
        // (name, batch, vocab, top_k, top_p, temperature, seed)
        ("argmax-temp0", 4, 256, 0, 1.0, 0.0, 0), // temperature 0 → argmax
        ("topk1", 3, 1024, 1, 1.0, 1.0, 42),      // top_k=1 → deterministic argmax
        ("topk40", 2, 32000, 40, 1.0, 0.8, 7),    // realistic LLM decode
        ("topp", 2, 4096, 0, 0.9, 1.0, 123),      // nucleus
        ("topk-topp", 5, 8192, 50, 0.95, 0.7, 999), // combined
    ]
}

#[test]
fn sample_cpu_runs() {
    for (name, b, v, k, p, t, s) in cases() {
        let out = run_on(Device::Cpu, b, v, k, p, t, s);
        assert_eq!(out.len(), b, "{name}: expected {b} sampled indices");
        for &idx in &out {
            assert!(
                idx >= 0.0 && (idx as usize) < v,
                "{name}: index {idx} out of range"
            );
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn sample_metal_matches_cpu() {
    for (name, b, v, k, p, t, s) in cases() {
        let metal = run_on(Device::Metal, b, v, k, p, t, s);
        let cpu = run_on(Device::Cpu, b, v, k, p, t, s);
        assert_eq!(
            metal, cpu,
            "metal {name}: sampled indices diverge {metal:?} vs {cpu:?}"
        );
        eprintln!("metal {name}: {metal:?} == cpu");
    }
}
