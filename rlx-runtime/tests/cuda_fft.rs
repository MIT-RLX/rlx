// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! CUDA-backend `Op::Fft` native multi-kernel dispatch test.

#![cfg(all(feature = "cpu", feature = "cuda"))]

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

#[test]
fn fft_cuda_native_matches_cpu_pow2() {
    for &n in &[2usize, 4, 8, 16, 64, 256, 1024, 2048, 4096] {
        let mut re: Vec<f32> = Vec::with_capacity(n);
        let mut im: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            re.push((i as f32 * 0.31 - 1.0).sin());
            im.push((i as f32 * 0.71).cos() * 0.5);
        }
        let mut x = Vec::with_capacity(2 * n);
        x.extend_from_slice(&re);
        x.extend_from_slice(&im);

        let build = || {
            let mut g = Graph::new("fft_cuda_native");
            let xc = const_f32(&mut g, &x);
            let y = g.fft(xc, false);
            g.set_outputs(vec![y]);
            g
        };
        let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let cuda = bytes_to_f32s(&Session::new(Device::Cuda).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), cuda.len(), "N={n}");
        let tol = 1e-4 * (n as f32).sqrt();
        for k in 0..cpu.len() {
            assert!(
                (cpu[k] - cuda[k]).abs() < tol,
                "N={n} k={k}: cpu={} cuda={} diff={}",
                cpu[k],
                cuda[k],
                (cpu[k] - cuda[k]).abs()
            );
        }
    }
}

#[test]
fn fft_cuda_round_trip_f32_pow2() {
    let n: usize = 32;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
    let mut x = Vec::with_capacity(2 * n);
    x.extend_from_slice(&re);
    x.extend_from_slice(&im);

    let mut g = Graph::new("cuda_round_trip");
    let xc = const_f32(&mut g, &x);
    let y = g.fft(xc, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let cuda = bytes_to_f32s(&Session::new(Device::Cuda).compile(g).run_typed(&[])[0].0);
    let nf = n as f32;
    let tol = 1e-3;
    for k in 0..n {
        assert!(
            (cuda[k] - nf * re[k]).abs() < tol,
            "re[{k}]: {} vs {}",
            cuda[k],
            nf * re[k]
        );
        assert!(
            (cuda[n + k] - nf * im[k]).abs() < tol,
            "im[{k}]: {} vs {}",
            cuda[n + k],
            nf * im[k]
        );
    }
}
