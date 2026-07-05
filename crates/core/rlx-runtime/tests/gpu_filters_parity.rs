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

//! Cross-backend parity for the FIR / RIR / IIR filter builders.
//!
//! Each `check_*` builds one filter's graph, runs it on the CPU backend and on
//! a GPU backend, and asserts the outputs match. The per-backend `mod`s below
//! instantiate them for every backend compiled in. Run e.g.:
//!   cargo test -p rlx-runtime --features metal --test gpu_filters_parity
//!   cargo test -p rlx-runtime --features gpu   --test gpu_filters_parity
//!
//! `fir_conv1d` / `conv_reverb` / `iir_as_fir` are pure `Op::Fft` + elementwise
//! compositions (native everywhere); `iirfilt` lowers via `Op::Scan` (native on
//! CPU/MLX, host-fallback on Metal/wgpu).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, FirMode, Graph, NodeId, Op, Shape};
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

// rlx-vulkan's arena / command pool / descriptor state is a process-global
// singleton documented as single-executable-at-a-time (not `Sync`); the default
// parallel test harness would let concurrent GPU Sessions corrupt it. Serialize
// GPU parity runs. (Cheap — each test is milliseconds.)
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Build on CPU and on `dev`, assert element-wise agreement.
fn parity(name: &str, dev: Device, tol: f32, build: &dyn Fn(&mut Graph) -> Vec<NodeId>) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cpu = run_on(Device::Cpu, build);
    let gpu = run_on(dev, build);
    assert_eq!(cpu.len(), gpu.len(), "{name}: output count mismatch");
    for (oi, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(c.len(), g.len(), "{name}: output {oi} length mismatch");
        for (i, (a, b)) in c.iter().zip(g.iter()).enumerate() {
            let d = (a - b).abs();
            let rel = d / a.abs().max(b.abs()).max(1e-4);
            assert!(
                d < tol || rel < tol,
                "{name}: output {oi}[{i}] cpu={a} gpu={b} (Δ={d})"
            );
        }
    }
}

fn signal(g: &mut Graph, n: usize, dims: &[usize]) -> NodeId {
    let xs: Vec<f32> = (0..n)
        .map(|t| (t as f32 * 0.3).sin() + 0.4 * (t as f32 * 0.11).cos())
        .collect();
    const_f32(g, &xs, dims)
}

// ── Device-agnostic checks ──────────────────────────────────────

fn check_fir_direct(dev: Device) {
    parity("fir_direct", dev, 1e-3, &|g| {
        let x = signal(g, 96, &[96]);
        let taps: Vec<f32> = (0..9).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        vec![g.fir_conv1d(x, &taps, FirMode::Causal)]
    });
}

fn check_fir_fft(dev: Device) {
    parity("fir_fft", dev, 5e-2, &|g| {
        let x = signal(g, 130, &[130]);
        let taps: Vec<f32> = (0..100)
            .map(|i| ((i as f32) * 0.02).cos() / (i as f32 + 1.0))
            .collect();
        vec![g.fir_conv1d(x, &taps, FirMode::Full)]
    });
}

fn check_conv_reverb(dev: Device) {
    parity("conv_reverb", dev, 5e-2, &|g| {
        let x = signal(g, 300, &[2, 150]);
        let ir: Vec<f32> = (0..200).map(|i| (-(i as f32) * 0.03).exp()).collect();
        vec![g.conv_reverb(x, &ir, 128)]
    });
}

#[allow(dead_code)] // only referenced by CPU/MLX/Metal/wgpu modules (Op::Scan)
fn check_iirfilt(dev: Device) {
    parity("iirfilt", dev, 1e-2, &|g| {
        let x = signal(g, 192, &[2, 96]);
        vec![g.iirfilt(x, &[0.2, 0.4, 0.2], &[1.0, -0.3, 0.1])]
    });
}

fn check_iir_as_fir(dev: Device) {
    parity("iir_as_fir", dev, 5e-2, &|g| {
        let x = signal(g, 120, &[120]);
        vec![g.iir_as_fir(
            x,
            &[0.15, 0.3, 0.15],
            &[1.0, -0.5, 0.2],
            96,
            FirMode::Causal,
        )]
    });
}

/// `Op::PartitionedConv` (fused node) — decomposes to the batched-GEMM
/// frequency-domain path (cuBLAS/rocBLAS/MPS) and must match CPU.
fn check_partitioned_conv_op(dev: Device) {
    parity("partitioned_conv_op", dev, 5e-2, &|g| {
        let x = signal(g, 300, &[2, 150]);
        let ir: Vec<f32> = (0..200).map(|i| (-(i as f32) * 0.03).exp()).collect();
        let irn = const_f32(g, &ir, &[ir.len()]);
        vec![g.partitioned_conv(x, irn, 128)]
    });
}

#[allow(dead_code)] // only referenced by some per-backend modules
fn run_all(dev: Device) {
    check_fir_direct(dev);
    check_fir_fft(dev);
    check_conv_reverb(dev);
    check_iir_as_fir(dev);
    check_partitioned_conv_op(dev);
}

// ── Per-backend instantiations ──────────────────────────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Metal);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Metal);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Metal);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Metal);
    }
    #[test]
    fn partitioned_conv_op() {
        check_partitioned_conv_op(Device::Metal);
    }
    #[test]
    fn iirfilt() {
        check_iirfilt(Device::Metal); // Op::Scan host-fallback
    }
}

#[cfg(feature = "mlx")]
mod mlx {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Mlx);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Mlx);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Mlx);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Mlx);
    }
    #[test]
    fn partitioned_conv_op() {
        check_partitioned_conv_op(Device::Mlx);
    }
    #[test]
    fn iirfilt() {
        check_iirfilt(Device::Mlx); // MLX lowers Op::Scan natively
    }
}

#[cfg(feature = "gpu")]
mod wgpu {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Gpu);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Gpu);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Gpu);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Gpu);
    }
    #[test]
    fn iirfilt() {
        check_iirfilt(Device::Gpu); // Op::Scan readback host-fallback
    }
    #[test]
    fn all() {
        run_all(Device::Gpu);
    }
}

// CUDA/ROCm/Vulkan cover the FFT-composition filters (cuFFT/rocFFT native,
// Vulkan host-fallback). `iirfilt` is omitted: it lowers via `Op::Scan`, whose
// host-fallback is only wired for CPU/MLX/Metal/wgpu — matching how
// `gpu_exg_parity` leaves `biquad` off these backends.
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Cuda);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Cuda);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Cuda);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Cuda);
    }
    #[test]
    fn partitioned_conv_op() {
        check_partitioned_conv_op(Device::Cuda);
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Rocm);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Rocm);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Rocm);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Rocm);
    }
    #[test]
    fn partitioned_conv_op() {
        check_partitioned_conv_op(Device::Rocm);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    #[test]
    fn fir_direct() {
        check_fir_direct(Device::Vulkan);
    }
    #[test]
    fn fir_fft() {
        check_fir_fft(Device::Vulkan);
    }
    #[test]
    fn conv_reverb() {
        check_conv_reverb(Device::Vulkan);
    }
    #[test]
    fn iir_as_fir() {
        check_iir_as_fir(Device::Vulkan);
    }
    #[test]
    fn partitioned_conv_op() {
        check_partitioned_conv_op(Device::Vulkan);
    }
}
