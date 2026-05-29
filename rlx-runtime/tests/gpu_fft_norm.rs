// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! GPU FFT normalization, real-input helpers, and non-pow2 host fallback.

#![cfg(all(
    feature = "cpu",
    any(
        feature = "cuda",
        feature = "gpu",
        all(feature = "metal", target_os = "macos")
    )
))]

use rlx_ir::FftNorm;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F32),
    )
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn complex_block(n: usize) -> Vec<f32> {
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut x = Vec::with_capacity(2 * n);
    x.extend_from_slice(&re);
    x.extend_from_slice(&im);
    x
}

fn assert_fft_norm_matches_cpu(device: Device, n: usize, norm: FftNorm, inverse: bool) {
    let x = complex_block(n);
    let build = || {
        let mut g = Graph::new("gpu_fft_norm");
        let xc = const_f32(&mut g, &x);
        let y = g.fft_norm(xc, inverse, norm);
        g.set_outputs(vec![y]);
        g
    };
    let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
    let gpu = bytes_to_f32s(&Session::new(device).compile(build()).run_typed(&[])[0].0);
    assert_eq!(cpu.len(), gpu.len(), "n={n} norm={norm:?} inv={inverse}");
    let tol = 1e-4 * (n as f32).sqrt();
    for k in 0..cpu.len() {
        assert!(
            (cpu[k] - gpu[k]).abs() < tol,
            "n={n} norm={norm:?} inv={inverse} k={k}: cpu={} gpu={}",
            cpu[k],
            gpu[k]
        );
    }
}

fn assert_fft_matches_cpu(device: Device, n: usize) {
    let x = complex_block(n);
    let build = || {
        let mut g = Graph::new("gpu_fft_fallback");
        let xc = const_f32(&mut g, &x);
        let y = g.fft(xc, false);
        g.set_outputs(vec![y]);
        g
    };
    let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
    let gpu = bytes_to_f32s(&Session::new(device).compile(build()).run_typed(&[])[0].0);
    assert_eq!(cpu.len(), gpu.len(), "n={n}");
    let tol = 1e-4 * (n as f32).sqrt();
    for k in 0..cpu.len() {
        assert!(
            (cpu[k] - gpu[k]).abs() < tol,
            "n={n} k={k}: cpu={} gpu={}",
            cpu[k],
            gpu[k]
        );
    }
}

macro_rules! gpu_fft_norm_tests {
    ($mod_name:ident, $device:expr, $feature:meta) => {
        mod $mod_name {
            #![cfg($feature)]

            use super::*;

            #[test]
            fn forward_and_ortho_norm_pow2() {
                for &n in &[16usize, 64, 256] {
                    for norm in [FftNorm::Forward, FftNorm::Ortho] {
                        assert_fft_norm_matches_cpu($device, n, norm, false);
                        assert_fft_norm_matches_cpu($device, n, norm, true);
                    }
                }
            }

            #[test]
            fn non_pow2_host_fallback() {
                for &n in &[15usize, 12, 20] {
                    assert_fft_matches_cpu($device, n);
                }
            }
        }
    };
}

fn assert_fft_real_and_psd(device: Device) {
    let signal = [1.0_f32, 2.0, 3.0];
    let build = || {
        let mut g = Graph::new("gpu_fft_real");
        let x = const_f32(&mut g, &signal);
        let (re, im) = g.fft_real(x, FftNorm::Backward);
        let p = g.psd(re, im);
        g.set_outputs(vec![re, im, p]);
        g
    };
    let cpu_outs = Session::new(Device::Cpu).compile(build()).run_typed(&[]);
    let gpu_outs = Session::new(device).compile(build()).run_typed(&[]);
    for (i, (cpu, gpu)) in cpu_outs.iter().zip(gpu_outs.iter()).enumerate() {
        let cpu_f = bytes_to_f32s(&cpu.0);
        let gpu_f = bytes_to_f32s(&gpu.0);
        assert_eq!(cpu_f.len(), gpu_f.len(), "out[{i}]");
        for k in 0..cpu_f.len() {
            assert!(
                (cpu_f[k] - gpu_f[k]).abs() < 1e-4,
                "out[{i}] k={k}: cpu={} gpu={}",
                cpu_f[k],
                gpu_f[k]
            );
        }
    }
}

gpu_fft_norm_tests!(cuda, Device::Cuda, all(feature = "cuda"));
gpu_fft_norm_tests!(wgpu, Device::Gpu, all(feature = "gpu"));
gpu_fft_norm_tests!(
    metal,
    Device::Metal,
    all(feature = "metal", target_os = "macos")
);

#[cfg(feature = "cuda")]
#[test]
fn cuda_fft_real_and_psd() {
    assert_fft_real_and_psd(Device::Cuda);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_fft_real_and_psd() {
    assert_fft_real_and_psd(Device::Gpu);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_fft_real_and_psd() {
    assert_fft_real_and_psd(Device::Metal);
}
