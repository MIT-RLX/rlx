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

//! Cross-backend parity for the exg signal-processing graph helpers.
//!
//! Each `check_*` builds one helper's graph, runs it on the CPU backend and on
//! a GPU backend, and asserts the outputs match. The per-backend `mod`s below
//! instantiate them for every Apple-Silicon backend compiled in (`metal`,
//! `gpu`/wgpu). Run e.g.:
//!   cargo test -p rlx-runtime --features metal --test gpu_exg_parity
//!   cargo test -p rlx-runtime --features gpu   --test gpu_exg_parity
//!
//! `biquad`/`sosfilt` are intentionally absent — they lower via `Op::Scan`,
//! which is CPU-only today; they are covered by `cpu_exg_kernels.rs`.

#![cfg(feature = "cpu")]

use rlx_ir::ops::spectral::WindowKind;
use rlx_ir::ops::upsample::InterpMode;
use rlx_ir::ops::vq::VqMetric;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(dims, DType::F32),
    )
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run_on(dev: Device, build: &dyn Fn(&mut Graph) -> Vec<NodeId>) -> Vec<Vec<f32>> {
    let mut g = Graph::new("parity");
    let outs = build(&mut g);
    g.set_outputs(outs);
    Session::new(dev)
        .compile(g)
        .run_typed(&[])
        .iter()
        .map(|o| bytes_to_f32s(&o.0))
        .collect()
}

/// Build on CPU and on `dev`, assert element-wise agreement.
fn parity(name: &str, dev: Device, tol: f32, build: &dyn Fn(&mut Graph) -> Vec<NodeId>) {
    let cpu = run_on(Device::Cpu, build);
    let gpu = run_on(dev, build);
    assert_eq!(cpu.len(), gpu.len(), "{name}: output count mismatch");
    for (oi, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(c.len(), g.len(), "{name}: output {oi} length mismatch");
        for (i, (a, b)) in c.iter().zip(g.iter()).enumerate() {
            let d = (a - b).abs();
            let rel = d / (a.abs().max(b.abs()).max(1e-4));
            assert!(
                d < tol || rel < tol,
                "{name}: output {oi}[{i}] cpu={a} gpu={b} (Δ={d})"
            );
        }
    }
}

// ── Device-agnostic checks ──────────────────────────────────────

fn check_vector_quantize(dev: Device) {
    parity("vector_quantize", dev, 1e-3, &|g| {
        let cb = const_f32(g, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0], &[3, 2]);
        let x = const_f32(g, &[9.0, 1.0, 1.0, 9.0, 0.2, 0.1], &[3, 2]);
        let (idx, q) = g.vector_quantize(x, cb, VqMetric::L2);
        vec![idx, q]
    });
}

fn check_interpolate1d(dev: Device) {
    parity("interpolate1d", dev, 1e-3, &|g| {
        let data: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect(); // [2, 5] = 10 elems
        let x = const_f32(g, &data, &[2, 5]);
        let y = g.interpolate1d(x, 11, InterpMode::Linear);
        vec![y]
    });
}

fn check_conv_transpose1d(dev: Device) {
    parity("conv_transpose1d", dev, 1e-3, &|g| {
        let x = const_f32(g, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let w = const_f32(g, &[0.5, 1.0, 0.5], &[1, 1, 3]);
        let y = g.conv_transpose1d(x, w, 3, 2, 0, 1, 0, 1);
        vec![y]
    });
}

fn check_spectrogram(dev: Device) {
    parity("spectrogram", dev, 5e-2, &|g| {
        let x: Vec<f32> = (0..128)
            .map(|t| (2.0 * std::f32::consts::PI * 12.0 * t as f32 / 64.0).sin())
            .collect();
        let xn = const_f32(g, &x, &[1, 128]);
        let y = g.spectrogram(xn, 64, 32, WindowKind::Hann, true, false);
        vec![y]
    });
}

fn check_band_power(dev: Device) {
    parity("band_power", dev, 5e-2, &|g| {
        let x: Vec<f32> = (0..128)
            .map(|t| (2.0 * std::f32::consts::PI * 10.0 * t as f32 / 64.0).sin())
            .collect();
        let xn = const_f32(g, &x, &[1, 128]);
        let y = g.band_power(xn, 64.0, &[(0.0, 5.0), (8.0, 12.0), (20.0, 32.0)]);
        vec![y]
    });
}

fn check_envelope(dev: Device) {
    parity("envelope", dev, 5e-2, &|g| {
        let x: Vec<f32> = (0..64)
            .map(|t| 2.0 * (2.0 * std::f32::consts::PI * 8.0 * t as f32 / 64.0).cos())
            .collect();
        let xn = const_f32(g, &x, &[64]);
        let y = g.envelope(xn);
        vec![y]
    });
}

fn check_fir_filtfilt(dev: Device) {
    parity("fir_filtfilt", dev, 5e-2, &|g| {
        let x: Vec<f32> = (0..96).map(|t| (t as f32 * 0.3).sin() + 0.5).collect();
        let xn = const_f32(g, &x, &[96]);
        let y = g.fir_filtfilt(xn, &[0.25, 0.5, 0.25]);
        vec![y]
    });
}

fn check_resample_poly(dev: Device) {
    parity("resample_poly", dev, 5e-2, &|g| {
        let x: Vec<f32> = (0..48).map(|t| (t as f32 * 0.5).sin()).collect();
        let xn = const_f32(g, &x, &[48]);
        let y = g.resample_poly(xn, 3, 2, &[0.25, 0.5, 0.25]);
        vec![y]
    });
}

#[allow(dead_code)] // used by the metal `all` test; unused under gpu-only builds
fn check_biquad(dev: Device) {
    parity("biquad", dev, 2e-3, &|g| {
        let x: Vec<f32> = (0..64).map(|t| (t as f32 * 0.3).sin() + 0.2).collect();
        let xn = const_f32(g, &x, &[2, 32]); // [2 channels, 32 samples]
        let y = g.biquad(xn, [0.2, 0.4, 0.2], [1.0, -0.3, 0.1]);
        vec![y]
    });
}

#[allow(dead_code)] // used by the metal `all` test; unused under gpu-only builds
fn run_all(dev: Device) {
    check_vector_quantize(dev);
    check_interpolate1d(dev);
    check_conv_transpose1d(dev);
    check_spectrogram(dev);
    check_band_power(dev);
    check_envelope(dev);
    check_fir_filtfilt(dev);
    check_resample_poly(dev);
}

// ── Per-backend instantiations ──────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    #[test]
    fn vector_quantize() {
        check_vector_quantize(Device::Metal);
    }
    #[test]
    fn interpolate1d() {
        check_interpolate1d(Device::Metal);
    }
    #[test]
    fn conv_transpose1d() {
        check_conv_transpose1d(Device::Metal);
    }
    #[test]
    fn spectrogram() {
        check_spectrogram(Device::Metal);
    }
    #[test]
    fn band_power() {
        check_band_power(Device::Metal);
    }
    #[test]
    fn envelope() {
        check_envelope(Device::Metal);
    }
    #[test]
    fn fir_filtfilt() {
        check_fir_filtfilt(Device::Metal);
    }
    #[test]
    fn resample_poly() {
        check_resample_poly(Device::Metal);
    }
    #[test]
    fn biquad() {
        check_biquad(Device::Metal); // Op::Scan host-fallback
    }
    #[test]
    fn all() {
        run_all(Device::Metal);
    }
}

// All eight helpers pass on MLX. (`conv_transpose1d` needed an MLX-backend fix:
// `conv_general`'s C++ shim drops `input_dilation`, so the 1-D transposed-conv
// path now inflates the input manually — see rlx-mlx/src/lower.rs.)
#[cfg(feature = "mlx")]
mod mlx {
    use super::*;
    #[test]
    fn vector_quantize() {
        check_vector_quantize(Device::Mlx);
    }
    #[test]
    fn interpolate1d() {
        check_interpolate1d(Device::Mlx);
    }
    #[test]
    fn conv_transpose1d() {
        check_conv_transpose1d(Device::Mlx);
    }
    #[test]
    fn spectrogram() {
        check_spectrogram(Device::Mlx);
    }
    #[test]
    fn band_power() {
        check_band_power(Device::Mlx);
    }
    #[test]
    fn envelope() {
        check_envelope(Device::Mlx);
    }
    #[test]
    fn fir_filtfilt() {
        check_fir_filtfilt(Device::Mlx);
    }
    #[test]
    fn resample_poly() {
        check_resample_poly(Device::Mlx);
    }
    #[test]
    fn biquad() {
        check_biquad(Device::Mlx); // MLX lowers Op::Scan natively
    }
}

#[cfg(feature = "gpu")]
mod wgpu {
    use super::*;
    #[test]
    fn vector_quantize() {
        check_vector_quantize(Device::Gpu);
    }
    #[test]
    fn interpolate1d() {
        check_interpolate1d(Device::Gpu);
    }
    #[test]
    fn conv_transpose1d() {
        check_conv_transpose1d(Device::Gpu);
    }
    #[test]
    fn spectrogram() {
        check_spectrogram(Device::Gpu);
    }
    #[test]
    fn band_power() {
        check_band_power(Device::Gpu);
    }
    #[test]
    fn envelope() {
        check_envelope(Device::Gpu);
    }
    #[test]
    fn fir_filtfilt() {
        check_fir_filtfilt(Device::Gpu);
    }
    #[test]
    fn resample_poly() {
        check_resample_poly(Device::Gpu);
    }
    #[test]
    fn biquad() {
        check_biquad(Device::Gpu); // Op::Scan readback host-fallback
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    #[test]
    fn vector_quantize() {
        check_vector_quantize(Device::Cuda);
    }
    #[test]
    fn interpolate1d() {
        check_interpolate1d(Device::Cuda);
    }
    #[test]
    fn conv_transpose1d() {
        check_conv_transpose1d(Device::Cuda);
    }
    #[test]
    fn spectrogram() {
        check_spectrogram(Device::Cuda);
    }
    #[test]
    fn band_power() {
        check_band_power(Device::Cuda);
    }
    #[test]
    fn envelope() {
        check_envelope(Device::Cuda);
    }
    #[test]
    fn fir_filtfilt() {
        check_fir_filtfilt(Device::Cuda);
    }
    #[test]
    fn resample_poly() {
        check_resample_poly(Device::Cuda);
    }
    #[test]
    fn biquad() {
        check_biquad(Device::Cuda); // Op::Scan D2H→CPU→H2D host fallback
    }
}
