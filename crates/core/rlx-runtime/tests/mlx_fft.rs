// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! MLX-backend `Op::Fft` Session parity (2N real-block layout).

#![cfg(all(feature = "cpu", feature = "mlx", target_os = "macos"))]

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
fn fft_mlx_matches_cpu_pow2() {
    for &n in &[2usize, 4, 8, 16, 32, 64, 256] {
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
            let mut g = Graph::new("fft_mlx_native");
            let xc = const_f32(&mut g, &x);
            let y = g.fft(xc, false);
            g.set_outputs(vec![y]);
            g
        };
        let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let mlx = bytes_to_f32s(&Session::new(Device::Mlx).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), mlx.len(), "N={n}");
        assert_eq!(mlx.len(), 2 * n, "N={n}");
        let tol = 1e-3 * (n as f32).sqrt();
        for k in 0..cpu.len() {
            assert!(
                (cpu[k] - mlx[k]).abs() < tol,
                "N={n} k={k}: cpu={} mlx={}",
                cpu[k],
                mlx[k]
            );
        }
    }
}

#[test]
fn fft_mlx_forward_ortho_norm() {
    use rlx_ir::FftNorm;

    let n = 16usize;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.4).cos()).collect();
    let mut x = Vec::with_capacity(2 * n);
    x.extend_from_slice(&re);
    x.extend_from_slice(&im);

    for norm in [FftNorm::Forward, FftNorm::Ortho] {
        let build = || {
            let mut g = Graph::new("fft_mlx_norm");
            let xc = const_f32(&mut g, &x);
            let y = g.fft_norm(xc, false, norm);
            g.set_outputs(vec![y]);
            g
        };
        let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let mlx = bytes_to_f32s(&Session::new(Device::Mlx).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), mlx.len());
        let tol = 1e-3 * (n as f32).sqrt();
        for k in 0..cpu.len() {
            assert!(
                (cpu[k] - mlx[k]).abs() < tol,
                "norm={norm:?} k={k}: cpu={} mlx={}",
                cpu[k],
                mlx[k]
            );
        }
    }
}
