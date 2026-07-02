// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! WGPU-backend `Op::Fft` native multi-kernel compute-shader dispatch test.
//!
//! Validates f32 + power-of-two N (including N > 1024) against CPU reference.

#![cfg(all(feature = "cpu", feature = "gpu"))]

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
fn fft_wgpu_native_matches_cpu_pow2() {
    // Includes n > 4096 (multi-kernel path with >=2 outer stages) — this used to
    // silently corrupt due to a shared FFT uniform buffer aliasing across stages.
    for &n in &[
        2usize, 4, 8, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ] {
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
            let mut g = Graph::new("fft_wgpu_native");
            let xc = const_f32(&mut g, &x);
            let y = g.fft(xc, false);
            g.set_outputs(vec![y]);
            g
        };
        let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let wgpu = bytes_to_f32s(&Session::new(Device::Gpu).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), wgpu.len(), "N={n}");
        let tol = 1e-4 * (n as f32).sqrt();
        for k in 0..cpu.len() {
            assert!(
                (cpu[k] - wgpu[k]).abs() < tol,
                "N={n} k={k}: cpu={} wgpu={} diff={}",
                cpu[k],
                wgpu[k],
                (cpu[k] - wgpu[k]).abs()
            );
        }
    }
}

// native-gpu-fft: batched small-n FFT exercises the multi-row on-chip kernel
// (outer>=2, n<=1024 => rows=floor(2048/n) rows packed per workgroup). Verifies
// the multi-row packing/bit-reversal/store indexing against CPU.
#[test]
fn fft_wgpu_multirow_batched_matches_cpu() {
    // Multi-row is opt-in (default off — it regresses on wgpu); force it on so
    // this test still validates the kernel's packing/indexing.
    // SAFETY: test process sets a process-local gate before compiling.
    unsafe {
        std::env::set_var("RLX_FFT_MULTIROW", "1");
    }
    for &n in &[8usize, 64, 256, 512, 1024] {
        for &batch in &[2usize, 3, 7, 16] {
            // Row layout is a 2N real-block: [re[0..n], im[0..n]] per row.
            let mut x = Vec::with_capacity(batch * 2 * n);
            for b in 0..batch {
                for i in 0..n {
                    x.push(((i + b) as f32 * 0.31 - 1.0).sin());
                }
                for i in 0..n {
                    x.push(((i + 2 * b) as f32 * 0.71).cos() * 0.5);
                }
            }
            let build = || {
                let mut g = Graph::new("fft_wgpu_multirow");
                let mut bytes = Vec::with_capacity(x.len() * 4);
                for &v in &x {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                let xc = g.add_node(
                    Op::Constant { data: bytes },
                    vec![],
                    Shape::new(&[batch, 2 * n], DType::F32),
                );
                let y = g.fft(xc, false);
                g.set_outputs(vec![y]);
                g
            };
            let cpu =
                bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
            let wgpu =
                bytes_to_f32s(&Session::new(Device::Gpu).compile(build()).run_typed(&[])[0].0);
            assert_eq!(cpu.len(), wgpu.len(), "N={n} batch={batch}");
            let tol = 1e-4 * (n as f32).sqrt();
            for k in 0..cpu.len() {
                assert!(
                    (cpu[k] - wgpu[k]).abs() < tol,
                    "N={n} batch={batch} k={k}: cpu={} wgpu={} diff={}",
                    cpu[k],
                    wgpu[k],
                    (cpu[k] - wgpu[k]).abs()
                );
            }
        }
    }
}

// native-gpu-fft: batch (`outer`) exceeding wgpu's 65535 per-dimension grid cap
// must NOT panic (copy kernel) or silently drop rows (stage kernels). Two regimes:
//   - n=2, batch>65535: single_inner radix2_full path, row spilled across y/z.
//   - n=2048, batch*n*2/64 > 65535: multi-kernel path, copy dispatch spilled to 2D.
// Multirow is left default-off so these hit the row_grid / dispatch_dims fixes.
#[test]
fn fft_wgpu_grid_overflow_matches_cpu() {
    for &(n, batch) in &[(2usize, 70000usize), (2048usize, 1100usize)] {
        let mut x = Vec::with_capacity(batch * 2 * n);
        for b in 0..batch {
            for i in 0..n {
                x.push(((i + b) as f32 * 0.017).sin());
            }
            for i in 0..n {
                x.push(((i * 2 + b) as f32 * 0.011).cos() * 0.5);
            }
        }
        let build = || {
            let mut g = Graph::new("fft_wgpu_overflow");
            let mut bytes = Vec::with_capacity(x.len() * 4);
            for &v in &x {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let xc = g.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&[batch, 2 * n], DType::F32),
            );
            let y = g.fft(xc, false);
            g.set_outputs(vec![y]);
            g
        };
        let cpu = bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let wgpu = bytes_to_f32s(&Session::new(Device::Gpu).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), wgpu.len(), "N={n} batch={batch}");
        let tol = 1e-4 * (n as f32).sqrt();
        // Check the tail rows explicitly — those are the ones a dropped-row bug loses.
        let mut max_diff = 0f32;
        for k in 0..cpu.len() {
            max_diff = max_diff.max((cpu[k] - wgpu[k]).abs());
        }
        assert!(max_diff < tol, "N={n} batch={batch}: max|Δ|={max_diff} tol={tol}");
    }
}

#[test]
fn fft_wgpu_round_trip_f32_pow2() {
    let n: usize = 32;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
    let mut x = Vec::with_capacity(2 * n);
    x.extend_from_slice(&re);
    x.extend_from_slice(&im);

    let mut g = Graph::new("wgpu_round_trip");
    let xc = const_f32(&mut g, &x);
    let y = g.fft(xc, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let wgpu = bytes_to_f32s(&Session::new(Device::Gpu).compile(g).run_typed(&[])[0].0);
    let nf = n as f32;
    let tol = 1e-3;
    for k in 0..n {
        assert!(
            (wgpu[k] - nf * re[k]).abs() < tol,
            "re[{k}]: {} vs {}",
            wgpu[k],
            nf * re[k]
        );
        assert!(
            (wgpu[n + k] - nf * im[k]).abs() < tol,
            "im[{k}]: {} vs {}",
            wgpu[n + k],
            nf * im[k]
        );
    }
}
