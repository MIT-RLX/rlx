// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! native-gpu-fft real→complex fusion (Metal): a forward FFT of
//! `Concat([signal_input, zeros])` reads `signal` directly with im=0, dropping
//! the Concat + zeros. Validates the fused on-chip radix-4/8 kernels against the
//! CPU reference (which runs the un-fused graph), for both the fused path
//! (default) and the un-fused path (`RLX_FFT_FUSE_REAL=0`).

#![cfg(all(
    feature = "cpu",
    feature = "metal",
    feature = "native-gpu-fft",
    target_os = "macos"
))]

use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// signal (resident Input) → fft_real → (re, im). fft_real builds
// `Concat([signal, zeros])` then `Fft`, which the Metal fusion rewrites.
fn build(n: usize) -> Graph {
    let mut g = Graph::new("fft_real_fuse");
    let sig = g.input("sig", Shape::new(&[1, n], DType::F32));
    let (re, im) = g.fft_real(sig, FftNorm::Backward);
    g.set_outputs(vec![re, im]);
    g
}

fn run(dev: Device, n: usize, sig_bytes: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let inputs: [(&str, &[u8], DType); 1] = [("sig", sig_bytes, DType::F32)];
    let out = Session::new(dev).compile(build(n)).run_typed(&inputs);
    (bytes_to_f32(&out[0].0), bytes_to_f32(&out[1].0))
}

#[test]
fn fft_real_fusion_matches_cpu() {
    for &n in &[2048usize, 4096] {
        let sig: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
        let sig_bytes: Vec<u8> = sig.iter().flat_map(|v| v.to_le_bytes()).collect();

        let (cpu_re, cpu_im) = run(Device::Cpu, n, &sig_bytes);
        let (mtl_re, mtl_im) = run(Device::Metal, n, &sig_bytes);

        assert_eq!(cpu_re.len(), mtl_re.len(), "n={n}");
        let tol = 1e-4 * (n as f32).sqrt();
        for k in 0..cpu_re.len() {
            assert!(
                (cpu_re[k] - mtl_re[k]).abs() < tol && (cpu_im[k] - mtl_im[k]).abs() < tol,
                "n={n} k={k}: cpu=({},{}) mtl=({},{})",
                cpu_re[k],
                cpu_im[k],
                mtl_re[k],
                mtl_im[k],
            );
        }
    }
}
