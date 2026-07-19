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

//! DiT modulation ops (`AdaLayerNorm` / `GatedResidual`) — CPU Session + Metal parity.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn ada_graph(norm: AdaNormKind) -> (Graph, usize, usize, usize) {
    let (b, s, d) = (2usize, 5usize, 8usize);
    let mut g = Graph::new("ada");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let scale = g.input("scale", Shape::new(&[b, 1, d], DType::F32));
    let shift = g.input("shift", Shape::new(&[b, 1, d], DType::F32));
    let out = g.ada_layer_norm(x, scale, shift, norm, 1e-5);
    g.set_outputs(vec![out]);
    (g, b, s, d)
}

fn gate_graph() -> (Graph, usize, usize, usize) {
    let (b, s, d) = (2usize, 5usize, 8usize);
    let mut g = Graph::new("gate");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let y = g.input("y", Shape::new(&[b, s, d], DType::F32));
    let gate = g.input("gate", Shape::new(&[b, 1, d], DType::F32));
    let out = g.gated_residual(x, y, gate);
    g.set_outputs(vec![out]);
    (g, b, s, d)
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn cpu_session_ada_layer_norm_and_gated_residual() {
    let (g, b, s, d) = ada_graph(AdaNormKind::LayerNorm);
    let x = fill(b * s * d, 0x1111);
    let scale = fill(b * d, 0x2222);
    let shift = fill(b * d, 0x3333);
    let mut sess = Session::new(Device::Cpu).compile(g);
    let out = sess
        .run(&[("x", &x), ("scale", &scale), ("shift", &shift)])
        .remove(0);
    assert_eq!(out.len(), b * s * d);
    assert!(out.iter().all(|v| v.is_finite()));

    let (g, b, s, d) = gate_graph();
    let x = fill(b * s * d, 0x4444);
    let y = fill(b * s * d, 0x5555);
    let gate = fill(b * d, 0x6666);
    let mut sess = Session::new(Device::Cpu).compile(g);
    let out = sess.run(&[("x", &x), ("y", &y), ("gate", &gate)]).remove(0);
    assert_eq!(out.len(), b * s * d);
    // Spot-check: out[0] ≈ x[0] + gate[0]*y[0] (row 0, col 0).
    let expected = x[0] + gate[0] * y[0];
    assert!(
        (out[0] - expected).abs() < 1e-5,
        "{} vs {}",
        out[0],
        expected
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn metal_ada_layer_norm_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        for &norm in &[AdaNormKind::LayerNorm, AdaNormKind::RmsNorm] {
            let (g, b, s, d) = ada_graph(norm);
            let x = fill(b * s * d, 0xA11A);
            let scale = fill(b * d, 0xB22B);
            let shift = fill(b * d, 0xC33C);
            let opts = rlx_runtime::CompileOptions::default();
            let mut cpu = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
            let cpu_out = cpu
                .run(&[("x", &x), ("scale", &scale), ("shift", &shift)])
                .remove(0);
            let mut metal = Session::new(Device::Metal).compile_with(g, &opts);
            let metal_out = metal
                .run(&[("x", &x), ("scale", &scale), ("shift", &shift)])
                .remove(0);
            let err = max_abs(&cpu_out, &metal_out);
            eprintln!("AdaLayerNorm {norm:?} max_abs={err:.6e}");
            assert!(err < 1e-4, "AdaLayerNorm {norm:?} max_abs={err}");
        }
    }

    #[test]
    fn metal_gated_residual_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        let (g, b, s, d) = gate_graph();
        let x = fill(b * s * d, 0xD44D);
        let y = fill(b * s * d, 0xE55E);
        let gate = fill(b * d, 0xF66F);
        let opts = rlx_runtime::CompileOptions::default();
        let mut cpu = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
        let cpu_out = cpu.run(&[("x", &x), ("y", &y), ("gate", &gate)]).remove(0);
        let mut metal = Session::new(Device::Metal).compile_with(g, &opts);
        let metal_out = metal
            .run(&[("x", &x), ("y", &y), ("gate", &gate)])
            .remove(0);
        let err = max_abs(&cpu_out, &metal_out);
        eprintln!("GatedResidual max_abs={err:.6e}");
        assert!(err < 1e-5, "GatedResidual max_abs={err}");
    }

    #[test]
    fn metal_dit_modulation_grad_matches_cpu() {
        if !rlx_runtime::is_available(Device::Metal) {
            return;
        }
        use rlx_ir::Op;
        use rlx_opt::autodiff::grad_with_loss;

        let (b, s, d) = (2usize, 4usize, 6usize);
        let mut g = Graph::new("ada_grad_parity");
        let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let scale = g.input("scale", Shape::new(&[b, 1, d], DType::F32));
        let shift = g.input("shift", Shape::new(&[b, 1, d], DType::F32));
        let out = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, 1e-5);
        let loss = g.sum(out, vec![0, 1, 2], false);
        g.set_outputs(vec![loss]);
        let bwd = grad_with_loss(&g, &[x, scale, shift]);
        assert!(
            bwd.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. }))
        );

        let x0 = fill(b * s * d, 0x111);
        let scale0 = fill(b * d, 0x222);
        let shift0 = fill(b * d, 0x333);
        let d_output = [1.0f32];
        let feeds = [
            ("x", x0.as_slice()),
            ("scale", scale0.as_slice()),
            ("shift", shift0.as_slice()),
            ("d_output", d_output.as_slice()),
        ];
        let opts = rlx_runtime::CompileOptions::default();
        let mut cpu = Session::new(Device::Cpu).compile_with(bwd.clone(), &opts);
        let cpu_outs = cpu.run(&feeds);
        let mut metal = Session::new(Device::Metal).compile_with(bwd, &opts);
        let metal_outs = metal.run(&feeds);
        for (i, (c, m)) in cpu_outs.iter().zip(metal_outs.iter()).enumerate() {
            let err = max_abs(c, m);
            eprintln!("ada grad out[{i}] max_abs={err:.6e}");
            assert!(err < 2e-4, "ada grad out[{i}] max_abs={err}");
        }

        let mut g = Graph::new("gate_grad_parity");
        let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let y = g.input("y", Shape::new(&[b, s, d], DType::F32));
        let gate = g.input("gate", Shape::new(&[b, 1, d], DType::F32));
        let out = g.gated_residual(x, y, gate);
        let loss = g.sum(out, vec![0, 1, 2], false);
        g.set_outputs(vec![loss]);
        let bwd = grad_with_loss(&g, &[x, y, gate]);
        let x0 = fill(b * s * d, 0x444);
        let y0 = fill(b * s * d, 0x555);
        let gate0 = fill(b * d, 0x666);
        let feeds = [
            ("x", x0.as_slice()),
            ("y", y0.as_slice()),
            ("gate", gate0.as_slice()),
            ("d_output", d_output.as_slice()),
        ];
        let mut cpu = Session::new(Device::Cpu).compile_with(bwd.clone(), &opts);
        let cpu_outs = cpu.run(&feeds);
        let mut metal = Session::new(Device::Metal).compile_with(bwd, &opts);
        let metal_outs = metal.run(&feeds);
        for (i, (c, m)) in cpu_outs.iter().zip(metal_outs.iter()).enumerate() {
            let err = max_abs(c, m);
            eprintln!("gate grad out[{i}] max_abs={err:.6e}");
            assert!(err < 2e-4, "gate grad out[{i}] max_abs={err}");
        }
    }
}

/// Shared packed-backward parity helper (CPU vs another device).
#[cfg(any(feature = "cuda", feature = "gpu", feature = "rocm", feature = "mlx"))]
fn packed_bwd_parity(device: Device, label: &str) {
    if !rlx_runtime::is_available(device) {
        return;
    }
    use rlx_ir::Op;
    use rlx_opt::autodiff::grad_with_loss;

    let (b, s, d) = (2usize, 4usize, 6usize);
    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let mut g = Graph::new("ada_grad_parity");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let scale = g.input("scale", Shape::new(&[b, 1, d], DType::F32));
    let shift = g.input("shift", Shape::new(&[b, 1, d], DType::F32));
    let out = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, 1e-5);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[x, scale, shift]);
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. }))
    );

    let x0 = fill(b * s * d, 0x111);
    let scale0 = fill(b * d, 0x222);
    let shift0 = fill(b * d, 0x333);
    let d_output = [1.0f32];
    let feeds = [
        ("x", x0.as_slice()),
        ("scale", scale0.as_slice()),
        ("shift", shift0.as_slice()),
        ("d_output", d_output.as_slice()),
    ];
    let opts = rlx_runtime::CompileOptions::default();
    let mut cpu = Session::new(Device::Cpu).compile_with(bwd.clone(), &opts);
    let cpu_outs = cpu.run(&feeds);
    let mut other = Session::new(device).compile_with(bwd, &opts);
    let other_outs = other.run(&feeds);
    for (i, (c, o)) in cpu_outs.iter().zip(other_outs.iter()).enumerate() {
        let err = max_abs(c, o);
        eprintln!("{label} ada grad out[{i}] max_abs={err:.6e}");
        assert!(err < 2e-4, "{label} ada grad out[{i}] max_abs={err}");
    }

    let mut g = Graph::new("gate_grad_parity");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let y = g.input("y", Shape::new(&[b, s, d], DType::F32));
    let gate = g.input("gate", Shape::new(&[b, 1, d], DType::F32));
    let out = g.gated_residual(x, y, gate);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[x, y, gate]);
    let x0 = fill(b * s * d, 0x444);
    let y0 = fill(b * s * d, 0x555);
    let gate0 = fill(b * d, 0x666);
    let feeds = [
        ("x", x0.as_slice()),
        ("y", y0.as_slice()),
        ("gate", gate0.as_slice()),
        ("d_output", d_output.as_slice()),
    ];
    let mut cpu = Session::new(Device::Cpu).compile_with(bwd.clone(), &opts);
    let cpu_outs = cpu.run(&feeds);
    let mut other = Session::new(device).compile_with(bwd, &opts);
    let other_outs = other.run(&feeds);
    for (i, (c, o)) in cpu_outs.iter().zip(other_outs.iter()).enumerate() {
        let err = max_abs(c, o);
        eprintln!("{label} gate grad out[{i}] max_abs={err:.6e}");
        assert!(err < 2e-4, "{label} gate grad out[{i}] max_abs={err}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_dit_modulation_grad_matches_cpu() {
    packed_bwd_parity(Device::Cuda, "cuda");
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_dit_modulation_grad_matches_cpu() {
    packed_bwd_parity(Device::Gpu, "wgpu");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_dit_modulation_grad_matches_cpu() {
    packed_bwd_parity(Device::Rocm, "rocm");
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_dit_modulation_grad_matches_cpu() {
    packed_bwd_parity(Device::Mlx, "mlx");
}
