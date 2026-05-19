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

//! Metal-backend `Op::Fft` host-fallback dispatch test.
//!
//! Validates the chain:
//!   1. `Op::Fft` is in `METAL_SUPPORTED_OPS`, so legalization accepts it.
//!   2. MPSGraph lowering returns None for `Op::Fft`, triggering the
//!      thunk-schedule fallback.
//!   3. `compile_thunks` lowers `Op::Fft` → `Thunk::Fft1d`.
//!   4. The executor flushes the cmd buffer, runs
//!      `rlx_cpu::thunk::execute_fft1d_f64` against the unified-memory
//!      arena pointer, and restarts the cmd buffer.
//!
//! Result must match the CPU backend element-wise — same kernel runs
//! in both cases, just routed through the Metal arena on this side.

#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]

use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn const_f64(g: &mut Graph, xs: &[f64]) -> NodeId {
    // Metal's run_typed widens host inputs to f32. F64 stays clean
    // when baked as Op::Constant — see metal_sparse_ops.rs for the
    // same workaround.
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F64),
    )
}

fn build_fft_round_trip_graph(n: usize, re: &[f64], im: &[f64]) -> Graph {
    let mut x_block = Vec::with_capacity(2 * n);
    x_block.extend_from_slice(re);
    x_block.extend_from_slice(im);

    let mut g = Graph::new("metal_fft_round_trip");
    let x = const_f64(&mut g, &x_block);
    let y = g.fft(x, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);
    g
}

#[test]
fn fft_round_trip_runs_on_metal_via_host_fallback() {
    // Round-trip identity ifft(fft(x)) = N·x must hold when the FFT
    // runs through Metal's host fallback. If the cmd_buf sync /
    // restart path regresses, this test catches it — either compile
    // legalization fails ("Op::Fft not supported") or the executor
    // panics ("Thunk::Fft1d not handled").
    let n: usize = 8;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5, -0.75, 3.0];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25, 1.0, -1.5];

    let mut compiled = Session::new(Device::Metal).compile(build_fft_round_trip_graph(n, &re, &im));
    let outs = compiled.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F64);
    let z_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(z_got.len(), 2 * n);

    let nf = n as f64;
    for k in 0..n {
        let want_re = nf * re[k];
        let want_im = nf * im[k];
        assert!(
            (z_got[k] - want_re).abs() < 1e-9,
            "metal round-trip re[{k}]: got {} vs N·x = {}",
            z_got[k],
            want_re
        );
        assert!(
            (z_got[n + k] - want_im).abs() < 1e-9,
            "metal round-trip im[{k}]: got {} vs N·x = {}",
            z_got[n + k],
            want_im
        );
    }
}

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
fn fft_metal_native_kernel_f32_pow2_matches_cpu() {
    // f32 + power-of-two + N ≤ 2048 → native MSL `fft_radix2_f32`
    // kernel runs. Compare against CPU reference at f32 tolerance.
    for &n in &[2usize, 4, 8, 16, 64, 256, 1024, 2048] {
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
            let mut g = Graph::new("fft_metal_native");
            let xc = const_f32(&mut g, &x);
            let y = g.fft(xc, false);
            g.set_outputs(vec![y]);
            g
        };
        let cpu =
            bytes_to_f32s(&Session::new(Device::Cpu).compile(build()).run_typed(&[])[0].0);
        let mtl =
            bytes_to_f32s(&Session::new(Device::Metal).compile(build()).run_typed(&[])[0].0);
        assert_eq!(cpu.len(), mtl.len());
        // f32 trig-per-butterfly accumulates more error than the CPU
        // path's f64 recurrence — scale tolerance with N (each stage
        // contributes ~1 ulp of rotation drift; log2(N) stages).
        let tol = 1e-4 * (n as f32).sqrt();
        for k in 0..cpu.len() {
            assert!(
                (cpu[k] - mtl[k]).abs() < tol,
                "N={n} k={k}: cpu={} mtl={} diff={}",
                cpu[k],
                mtl[k],
                (cpu[k] - mtl[k]).abs()
            );
        }
    }
}

#[test]
fn fft_metal_native_round_trip_f32_pow2() {
    // ifft(fft(x)) = N·x on the native Metal kernel path.
    let n: usize = 32;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
    let mut x = Vec::with_capacity(2 * n);
    x.extend_from_slice(&re);
    x.extend_from_slice(&im);

    let mut g = Graph::new("native_round_trip");
    let xc = const_f32(&mut g, &x);
    let y = g.fft(xc, false);
    let z = g.fft(y, true);
    g.set_outputs(vec![z]);

    let mtl = bytes_to_f32s(&Session::new(Device::Metal).compile(g).run_typed(&[])[0].0);
    let nf = n as f32;
    let tol = 1e-3;
    for k in 0..n {
        assert!(
            (mtl[k] - nf * re[k]).abs() < tol,
            "re[{k}]: {} vs {}",
            mtl[k],
            nf * re[k]
        );
        assert!(
            (mtl[n + k] - nf * im[k]).abs() < tol,
            "im[{k}]: {} vs {}",
            mtl[n + k],
            nf * im[k]
        );
    }
}

#[test]
fn fft_metal_matches_cpu_bitwise_for_non_pow2() {
    // Same input through Metal and CPU should produce identical
    // results — both backends run the same Bluestein kernel, just
    // dispatched against different arenas. N=6 forces the Bluestein
    // path on both sides.
    let n: usize = 6;
    let re = [1.0_f64, 0.5, -2.0, 0.25, 0.0, 1.5];
    let im = [0.5_f64, -1.0, 0.0, 2.0, -0.5, 0.25];
    let mut x_block = Vec::with_capacity(2 * n);
    x_block.extend_from_slice(&re);
    x_block.extend_from_slice(&im);

    let build = || {
        let mut g = Graph::new("metal_fft_vs_cpu");
        let x = const_f64(&mut g, &x_block);
        let y = g.fft(x, false);
        g.set_outputs(vec![y]);
        g
    };

    let mut cpu = Session::new(Device::Cpu).compile(build());
    let mut mtl = Session::new(Device::Metal).compile(build());
    let cpu_out = cpu.run_typed(&[]);
    let mtl_out = mtl.run_typed(&[]);
    assert_eq!(cpu_out[0].1, DType::F64);
    assert_eq!(mtl_out[0].1, DType::F64);
    let cpu_v = bytes_to_f64s(&cpu_out[0].0);
    let mtl_v = bytes_to_f64s(&mtl_out[0].0);
    assert_eq!(
        cpu_v, mtl_v,
        "Metal host-fallback FFT must be bit-identical to CPU FFT (same kernel)"
    );
}
