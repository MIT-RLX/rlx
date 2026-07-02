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

//! Native WGSL `softmax_cross_entropy` kernel parity vs the CPU fused
//! thunk + a closed-form reference. `loss[n] = lse(logits[n]) - Σ_c
//! targets[n,c]·logits[n,c]`.

#![cfg(feature = "gpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 6;
const C: usize = 7;

fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("wgpu_sce");
    let logits = g.input("logits", Shape::new(&[N, C], f));
    let targets = g.input("targets", Shape::new(&[N, C], f));
    let loss = g.softmax_cross_entropy(logits, targets);
    g.set_outputs(vec![loss]);
    g
}

fn sample_logits() -> Vec<f32> {
    (0..N * C)
        .map(|i| 0.4 * (i as f32) - 1.3 * ((i % 3) as f32) - 0.7)
        .collect()
}

fn sample_targets() -> Vec<f32> {
    let mut t = vec![0f32; N * C];
    for n in 0..N {
        let raw: Vec<f32> = (0..C).map(|c| 0.2 + 0.5 * (((n + c) % 4) as f32)).collect();
        let s: f32 = raw.iter().sum();
        for c in 0..C {
            t[n * C + c] = raw[c] / s;
        }
    }
    t
}

fn ref_loss(logits: &[f32], targets: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|n| {
            let row = &logits[n * C..(n + 1) * C];
            let trow = &targets[n * C..(n + 1) * C];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let lse = m + sum.ln();
            let dot: f32 = (0..C).map(|c| trow[c] * row[c]).sum();
            lse - dot
        })
        .collect()
}

#[test]
fn wgpu_softmax_cross_entropy_matches_cpu_and_reference() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_softmax_cross_entropy: no adapter");
        return;
    }
    let logits = sample_logits();
    let targets = sample_targets();
    let inputs: &[(&str, &[f32])] = &[("logits", &logits), ("targets", &targets)];

    // CPU fused thunk reference.
    let cpu = {
        let mut c = Session::new(Device::Cpu).compile(build());
        c.run(inputs)[0].clone()
    };

    // Native WGSL kernel.
    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(build());
    let got = exe.run(inputs).remove(0);

    let want = ref_loss(&logits, &targets);
    assert_eq!(got.len(), N);
    for n in 0..N {
        assert!(
            (got[n] - want[n]).abs() < 1e-4,
            "row {n}: wgpu {} vs ref {}",
            got[n],
            want[n]
        );
        assert!(
            (got[n] - cpu[n]).abs() < 1e-4,
            "row {n}: wgpu {} vs cpu {}",
            got[n],
            cpu[n]
        );
    }
}
