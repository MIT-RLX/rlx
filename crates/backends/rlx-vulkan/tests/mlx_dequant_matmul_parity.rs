//! MLX `Op::DequantMatMul` Vulkan vs CPU. No-ops when Vulkan is unavailable.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};
use rlx_vulkan::backend::VulkanExecutable;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn assert_close(cpu: &[f32], gpu: &[f32], label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label} len");
    let max_abs = cpu
        .iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cmax = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = (1e-4 * (1.0 + cmax)).max(2e-3);
    assert!(
        max_abs < tol,
        "{label}: max_abs={max_abs} tol={tol} cmax={cmax}"
    );
}

fn run_affine(m: usize, k: usize, n: usize, bits: u8, group_size: u32) {
    if !rlx_vulkan::is_available() {
        return;
    }
    let gs = group_size as usize;
    let n_groups = k / gs;
    let pf = match bits {
        2 | 4 | 8 => 8 / bits,
        3 | 5 => 8,
        6 => 4,
        _ => panic!("bits"),
    } as usize;
    let bpp = match bits {
        2 | 4 | 8 => 1,
        3 | 6 => 3,
        5 => 5,
        _ => 1,
    };
    let packs_in_group = gs / pf;
    let w_bytes: Vec<u8> = (0..n * n_groups * packs_in_group * bpp)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let scales: Vec<f32> = (0..n * n_groups)
        .map(|i| 0.02 + 0.001 * (i % 7) as f32)
        .collect();
    let biases: Vec<f32> = (0..n * n_groups)
        .map(|i| -0.05 + 0.001 * (i % 5) as f32)
        .collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let scheme = QuantScheme::MlxAffine { bits, group_size };
    let mut g = Graph::new("vk_mlx_affine");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[w_bytes.len()], DType::U8));
    let s = g.param("scale", Shape::new(&[n, n_groups], DType::F32));
    let z = g.param("zp", Shape::new(&[n, n_groups], DType::F32));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w, s, z],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut cpu_c = Session::new(Device::Cpu).compile(g.clone());
    cpu_c.set_param_typed("w", &w_bytes, DType::U8);
    cpu_c.set_param_typed("scale", &f32_bytes(&scales), DType::F32);
    cpu_c.set_param_typed("zp", &f32_bytes(&biases), DType::F32);
    let want = cpu_c.run(&[("x", x.as_slice())]).remove(0);

    let mut exe = VulkanExecutable::compile(g);
    exe.set_param_bytes("w", &w_bytes);
    exe.set_param_bytes("scale", &f32_bytes(&scales));
    exe.set_param_bytes("zp", &f32_bytes(&biases));
    let got = exe.run(&[("x", x.as_slice())]).remove(0);

    assert_close(&want, &got, &format!("vulkan mlx affine{bits} m={m}"));
}

fn run_mxfp(m: usize, k: usize, n: usize, group_size: u32, mxfp8: bool) {
    if !rlx_vulkan::is_available() {
        return;
    }
    let gs = group_size as usize;
    let n_groups = k / gs;
    let w_len = if mxfp8 { n * k } else { n * k / 2 };
    let w_bytes: Vec<u8> = (0..w_len)
        .map(|i| {
            let mut b = ((i * 41 + 7) % 256) as u8;
            if b == 0x7f || b == 0xff {
                b = 0x38;
            }
            b
        })
        .collect();
    let scales_u8: Vec<u8> = (0..n * n_groups).map(|i| (120 + (i % 20)) as u8).collect();
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let zp = vec![0u8; 4];
    let scheme = if mxfp8 {
        QuantScheme::MlxMxfp8 { group_size }
    } else {
        QuantScheme::MlxMxfp4 { group_size }
    };
    let mut g = Graph::new("vk_mlx_mxfp");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[w_bytes.len()], DType::U8));
    let s = g.param("scale", Shape::new(&[scales_u8.len()], DType::U8));
    let z = g.param("zp", Shape::new(&[1], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul { scheme },
        vec![x_in, w, s, z],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut cpu_c = Session::new(Device::Cpu).compile(g.clone());
    cpu_c.set_param_typed("w", &w_bytes, DType::U8);
    cpu_c.set_param_typed("scale", &scales_u8, DType::U8);
    cpu_c.set_param_typed("zp", &zp, DType::U8);
    let want = cpu_c.run(&[("x", x.as_slice())]).remove(0);

    let mut exe = VulkanExecutable::compile(g);
    exe.set_param_bytes("w", &w_bytes);
    exe.set_param_bytes("scale", &scales_u8);
    exe.set_param_bytes("zp", &zp);
    let got = exe.run(&[("x", x.as_slice())]).remove(0);
    let tag = if mxfp8 { "mxfp8" } else { "mxfp4" };
    assert_close(&want, &got, &format!("vulkan mlx {tag} m={m}"));
}

#[test]
fn vulkan_mlx_affine4_parity() {
    run_affine(2, 64, 8, 4, 64);
    run_affine(1, 64, 8, 4, 64);
}

#[test]
fn vulkan_mlx_affine_odd_bits_parity() {
    run_affine(2, 64, 4, 3, 64);
}

#[test]
fn vulkan_mlx_mxfp_parity() {
    run_mxfp(2, 64, 8, 32, false);
    run_mxfp(1, 64, 4, 32, true);
}
