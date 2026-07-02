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

//! End-to-end EEG preprocessing frontend, exercised across backends.
//!
//! Chains the DSP graph helpers into one graph — a 50 Hz notch (`biquad`, an
//! `Op::Scan` recurrence) → windowed `spectrogram` → per-band `band_power` →
//! `differential_entropy` — and checks it (a) is numerically sane on CPU and
//! (b) matches CPU on Metal/wgpu. The biquad forces a host-fallback *inside* a
//! GPU graph, so this also covers the Scan sync path mid-schedule.

#![cfg(feature = "cpu")]

use rlx_ir::ops::spectral::WindowKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

const SR: f32 = 128.0;
const T: usize = 512;
const C: usize = 4;
// delta, theta, alpha, beta, gamma
const BANDS: [(f32, f32); 5] = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)];

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut b = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(Op::Constant { data: b }, vec![], Shape::new(dims, DType::F32))
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// RBJ notch biquad at `f0` (Hz), quality `q`, normalized by a0 downstream.
fn notch(f0: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * std::f32::consts::PI * f0 / SR;
    let (sw, cw) = (w0.sin(), w0.cos());
    let alpha = sw / (2.0 * q);
    ([1.0, -2.0 * cw, 1.0], [1.0 + alpha, -2.0 * cw, 1.0 - alpha])
}

/// A 10 Hz (alpha) sinusoid + a 50 Hz line-noise component per channel.
fn eeg_signal() -> Vec<f32> {
    let mut v = vec![0f32; C * T];
    for c in 0..C {
        for t in 0..T {
            let tt = t as f32 / SR;
            let phase = c as f32 * 0.4;
            v[c * T + t] = (2.0 * std::f32::consts::PI * 10.0 * tt + phase).sin()
                + 0.8 * (2.0 * std::f32::consts::PI * 50.0 * tt).sin();
        }
    }
    v
}

/// Build the frontend graph; returns `[differential_entropy, band_power, spectrogram]`.
fn build(g: &mut Graph) -> Vec<NodeId> {
    let sig = const_f32(g, &eeg_signal(), &[C, T]);
    let (b, a) = notch(50.0, 5.0);
    let clean = g.biquad(sig, b, a); // remove 50 Hz line noise
    let de = g.differential_entropy(clean, SR, &BANDS);
    let bp = g.band_power(clean, SR, &BANDS);
    let spec = g.spectrogram(clean, 128, 64, WindowKind::Hann, true, true);
    vec![de, bp, spec]
}

fn run(dev: Device) -> Vec<Vec<f32>> {
    let mut g = Graph::new("eeg");
    let outs = build(&mut g);
    g.set_outputs(outs);
    Session::new(dev).compile(g).run_typed(&[]).iter().map(|o| bytes_to_f32s(&o.0)).collect()
}

#[test]
fn cpu_pipeline_is_sane() {
    let out = run(Device::Cpu);
    let (de, bp, spec) = (&out[0], &out[1], &out[2]);
    assert_eq!(de.len(), C * BANDS.len());
    assert_eq!(bp.len(), C * BANDS.len());
    assert!(de.iter().all(|v| v.is_finite()), "DE finite");
    assert!(spec.iter().all(|v| v.is_finite()), "spectrogram finite");

    // 10 Hz content ⇒ the alpha band (index 2) dominates every channel after
    // the 50 Hz notch.
    for c in 0..C {
        let row = &bp[c * BANDS.len()..(c + 1) * BANDS.len()];
        let alpha = row[2];
        assert!(
            row.iter().enumerate().all(|(i, &p)| i == 2 || alpha >= p),
            "ch{c}: alpha band should dominate, got {row:?}"
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_cpu() {
    assert_parity(Device::Metal);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_matches_cpu() {
    assert_parity(Device::Gpu);
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "gpu"))]
fn assert_parity(dev: Device) {
    let cpu = run(Device::Cpu);
    let gpu = run(dev);
    assert_eq!(cpu.len(), gpu.len());
    for (oi, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(c.len(), g.len(), "output {oi} length");
        for (i, (a, b)) in c.iter().zip(g.iter()).enumerate() {
            let d = (a - b).abs();
            let rel = d / a.abs().max(b.abs()).max(1e-4);
            assert!(d < 5e-2 || rel < 5e-2, "output {oi}[{i}] cpu={a} gpu={b}");
        }
    }
}
