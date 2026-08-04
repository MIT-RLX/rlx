// RLX — versatile ML compiler + runtime.
//! CPU-vs-Metal parity for the native grouped MLX-MXFP4 MoE GEMV kernel
//! (`grouped_dequant_matmul_mlx_gemv`). The same packed codes + BF16 scales — the
//! DeepSeek-V4 per-expert format (`scale_bf16`) — pushed through both backends must
//! agree, proving the on-GPU expert dequant+matmul matches the CPU host path it
//! replaces.

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

// f32 → bf16 bytes (truncate; both backends decode identically → parity holds).
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| (((x.to_bits()) >> 16) as u16).to_le_bytes())
        .collect()
}

// Run the grouped MLX-MXFP4 op for `m` rows with the given per-row experts on both
// backends; assert the native Metal kernel matches the CPU host path.
fn parity(m: usize, eidx_data: Vec<f32>) {
    let (e, n, k, gs) = (3usize, 4usize, 32usize, 32usize);
    let ng = k / gs;

    let build = || {
        let mut g = Graph::new("grp");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let wq = g.param("wq", Shape::new(&[e * n * (k / 2)], DType::U8));
        let sc = g.param("sc", Shape::new(&[e, n, ng], DType::BF16));
        let bi = g.param("bi", Shape::new(&[e, n, ng], DType::BF16));
        let eidx = g.input("eidx", Shape::new(&[m], DType::F32));
        let out = g.add_node(
            Op::DequantGroupedMatMulMlx {
                scheme: QuantScheme::MlxMxfp4 {
                    group_size: gs as u32,
                },
            },
            vec![x, wq, sc, bi, eidx],
            Shape::new(&[m, n], DType::F32),
        );
        g.set_outputs(vec![out]);
        g
    };

    // Deterministic synthetic packed data (arbitrary bytes are valid e2m1 codes).
    let wq_bytes: Vec<u8> = (0..e * n * (k / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let sc_f: Vec<f32> = (0..e * n * ng)
        .map(|i| 0.5 + 0.1 * (i % 5) as f32)
        .collect();
    let sc_b = bf16_bytes(&sc_f);
    let bi_b = bf16_bytes(&vec![0f32; e * n * ng]);
    let x_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.031).sin()).collect();

    let run = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(build());
        c.set_param_typed("wq", &wq_bytes, DType::U8);
        c.set_param_typed("sc", &sc_b, DType::BF16);
        c.set_param_typed("bi", &bi_b, DType::BF16);
        c.run(&[("x", x_data.as_slice()), ("eidx", eidx_data.as_slice())])[0].clone()
    };

    let cpu = run(Device::Cpu);
    let met = run(Device::Metal);
    let err = cpu
        .iter()
        .zip(&met)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        err < 1e-3,
        "grouped MLX-MXFP4 (m={m}) Metal must match CPU host path: err {err:e}\ncpu={cpu:?}\nmetal={met:?}"
    );
}

#[test]
fn grouped_mlx_mxfp4_gemv_cpu_metal_parity() {
    parity(1, vec![1.0]); // decode GEMV, expert 1
}

#[test]
fn grouped_mlx_mxfp4_gemm_cpu_metal_parity() {
    // Prefill GEMM: 5 rows routing to mixed experts (crosses the TM=8 tile), so
    // per-row expert offsetting is exercised.
    parity(5, vec![1.0, 0.0, 2.0, 1.0, 2.0]);
}
