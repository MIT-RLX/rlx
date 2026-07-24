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

//
// CPU vs GPU parity for 2nd-order AD through decomposed *Backward ops
// (RmsNorm, LayerNorm, Attention, Conv2d/3d, MaxPool2d/3d).
//
//! ```sh
//! cargo test -p rlx-runtime --features cpu,apple --test higher_order_decompose_parity
//! cargo test -p rlx-runtime --features cpu,gpu --test higher_order_decompose_parity
//! ```

#![cfg(all(
    feature = "cpu",
    any(
        feature = "cuda",
        feature = "rocm",
        feature = "gpu",
        all(feature = "metal", target_os = "macos"),
        all(feature = "mlx", target_os = "macos")
    )
))]

use rlx_autodiff::nth_order_grad;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{MaskKind, ReduceOp, SteKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session, is_available};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_guard {
    use std::sync::{Mutex, MutexGuard};

    /// Metal command queues and MPS caches are process-global; serialize GPU tests.
    static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

    pub struct MetalTestGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl MetalTestGuard {
        pub fn new() -> Self {
            Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
        }
    }

    impl Drop for MetalTestGuard {
        fn drop(&mut self) {
            rlx_metal::device::drain_command_queue();
            rlx_metal::mps_blas::invalidate_caches();
        }
    }
}

fn bind_dynamic_conv_graph(g: Graph, batch: usize) -> Graph {
    use rlx_ir::DimBinding;
    use rlx_ir::dynamic::{bind_graph, sym};
    let spatial_out = 4 * 4;
    bind_graph(
        &g,
        &DimBinding::from_pairs(&[(sym::BATCH, batch), (sym::ROWS, batch * spatial_out)]),
    )
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn eval_f32_vec(device: Device, g: Graph, inputs: &[(&str, &[u8], DType)]) -> Vec<f32> {
    let outs = Session::new(device).compile(g).run_typed(inputs);
    outs[0]
        .0
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn assert_vec_matches_cpu(
    device: Device,
    g: Graph,
    inputs: &[(&str, &[u8], DType)],
    tol: f32,
    label: &str,
) {
    if !is_available(device) {
        eprintln!("skip higher_order_decompose_parity {label} on {device:?} (unavailable)");
        return;
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let _metal_guard = (device == Device::Metal).then(metal_guard::MetalTestGuard::new);
    let cpu = eval_f32_vec(Device::Cpu, g.clone(), inputs);
    let gpu = eval_f32_vec(device, g, inputs);
    assert_eq!(cpu.len(), gpu.len(), "{label}: len mismatch");
    let max = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max < tol,
        "{label} {device:?}: max_abs_diff={max} tol={tol}"
    );
    assert!(cpu.iter().all(|v| v.is_finite()), "{label}: cpu non-finite");
}

fn build_rms_norm_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("rms_ho");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let gamma = g.input("gamma", Shape::new(&[4], f));
    let beta = g.input("beta", Shape::new(&[4], f));
    let y = g.rms_norm(x, gamma, beta, 1e-5);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_layer_norm_loss() -> Graph {
    let f = DType::F32;
    let shape = Shape::new(&[2, 4], f);
    let mut g = Graph::new("ln_ho");
    let x = g.input("x", shape.clone());
    let gamma = g.input("gamma", Shape::new(&[4], f));
    let beta = g.input("beta", Shape::new(&[4], f));
    let y = g.layer_norm(x, gamma, beta, -1, 1e-5, shape);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_conv2d_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_ho");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
    let w = g.input("w", Shape::new(&[1, 1, 3, 3], f));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_dynamic_conv2d_loss() -> Graph {
    use rlx_ir::dynamic::sym;
    use rlx_ir::shape::Dim;

    let f = DType::F32;
    let nchw = [
        Dim::Dynamic(sym::BATCH),
        Dim::Static(1),
        Dim::Static(4),
        Dim::Static(4),
    ];
    let mut g = Graph::new("dyn_conv_ho");
    let x = g.input("x", Shape::from_dims(&nchw, f));
    let w = g.input("w", Shape::new(&[1, 1, 3, 3], f));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_group_norm_loss() -> Graph {
    let f = DType::F32;
    let shape = Shape::new(&[1, 4, 2, 2], f);
    let mut g = Graph::new("gn_ho");
    let x = g.input("x", shape.clone());
    let gamma = g.input("gamma", Shape::new(&[4], f));
    let beta = g.input("beta", Shape::new(&[4], f));
    let y = g.group_norm(x, gamma, beta, 2, 1e-5);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_rope_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("rope_ho");
    let x = g.input("x", Shape::new(&[1, 2, 4], f));
    let cos = g.input("cos", Shape::new(&[2], f));
    let sin = g.input("sin", Shape::new(&[2], f));
    let y = g.rope(x, cos, sin, 4);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_maxpool_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("pool_ho");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2], f),
    );
    let loss = g.reduce(p, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_conv3d_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv3d_ho");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], f));
    let w = g.input("w", Shape::new(&[1, 1, 3, 3, 3], f));
    let y = g.conv3d(x, w, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1);
    let loss = g.reduce(
        y,
        ReduceOp::Sum,
        vec![0, 1, 2, 3, 4],
        false,
        Shape::scalar(f),
    );
    g.set_outputs(vec![loss]);
    g
}

fn build_maxpool3d_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("pool3d_ho");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4, 4], f));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2, 2],
            stride: vec![2, 2, 2],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2, 2], f),
    );
    let loss = g.reduce(
        p,
        ReduceOp::Sum,
        vec![0, 1, 2, 3, 4],
        false,
        Shape::scalar(f),
    );
    g.set_outputs(vec![loss]);
    g
}

fn build_conv2d_groups_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_grp_ho");
    let x = g.input("x", Shape::new(&[1, 2, 4, 4], f));
    let w = g.input("w", Shape::new(&[2, 1, 3, 3], f));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 2);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_cumsum_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("cumsum_ho");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let y = g.cumsum(x, -1, false, Shape::new(&[2, 4], f));
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_gather_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gather_ho");
    let table = g.input("table", Shape::new(&[4, 2], f));
    let indices = g.input("indices", Shape::new(&[3], f));
    let y = g.gather_(table, indices, 0);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_sce_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("sce_ho");
    let logits = g.input("logits", Shape::new(&[2, 4], f));
    let labels = g.input("labels", Shape::new(&[2], f));
    let per_ex = g.add_node(
        Op::SoftmaxCrossEntropyWithLogits,
        vec![logits, labels],
        Shape::new(&[2], f),
    );
    let loss = g.reduce(per_ex, ReduceOp::Sum, vec![0], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_fake_quant_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("fq_ho");
    let x = g.input("x", Shape::new(&[8], f));
    let y = g.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: None,
            ste: SteKind::Tanh,
            scale_mode: rlx_ir::op::ScaleMode::PerBatch,
        },
        vec![x],
        Shape::new(&[8], f),
    );
    let loss = g.reduce(y, ReduceOp::Sum, vec![0], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_scan_checkpoint_loss() -> Graph {
    let f = DType::F32;
    let n = 2usize;
    let length = 4u32;
    let k = 2u32;
    let carry = Shape::new(&[n], f);
    let mut body = Graph::new("scan_ckpt_body_ho");
    let bc = body.input("carry", carry.clone());
    let bx = body.input("x_t", carry.clone());
    let by = body.binary(rlx_ir::op::BinaryOp::Add, bc, bx, carry.clone());
    body.set_outputs(vec![by]);
    let mut g = Graph::new("scan_ckpt_ho");
    let init = g.input("init", carry.clone());
    let xs = g.input("xs", Shape::new(&[length as usize, n], f));
    let y = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: k,
        },
        vec![init, xs],
        Shape::new(&[k as usize, n], f),
    );
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_scan_loss() -> Graph {
    let f = DType::F32;
    let n = 2usize;
    let length = 3u32;
    let carry = Shape::new(&[n], f);
    let mut body = Graph::new("scan_body_ho");
    let bc = body.input("carry", carry.clone());
    let bx = body.input("x_t", carry.clone());
    let by = body.binary(rlx_ir::op::BinaryOp::Add, bc, bx, carry.clone());
    body.set_outputs(vec![by]);
    let mut g = Graph::new("scan_ho");
    let init = g.input("init", carry.clone());
    let xs = g.input("xs", Shape::new(&[length as usize, n], f));
    let y = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        Shape::new(&[length as usize, n], f),
    );
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_scan_long_loss() -> Graph {
    let f = DType::F32;
    let n = 2usize;
    let length = 130u32;
    let carry = Shape::new(&[n], f);
    let mut body = Graph::new("scan_body_long_ho");
    let bc = body.input("carry", carry.clone());
    let bx = body.input("x_t", carry.clone());
    let by = body.binary(rlx_ir::op::BinaryOp::Add, bc, bx, carry.clone());
    body.set_outputs(vec![by]);
    let mut g = Graph::new("scan_long_ho");
    let init = g.input("init", carry.clone());
    let xs = g.input("xs", Shape::new(&[length as usize, n], f));
    let y = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        Shape::new(&[length as usize, n], f),
    );
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_attention_rank3_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_r3_ho");
    let q = g.input("q", Shape::new(&[1, 3, 4], f));
    let k = g.input("k", Shape::new(&[1, 3, 4], f));
    let v = g.input("v", Shape::new(&[1, 3, 4], f));
    let out = g.add_node(
        Op::Attention {
            num_heads: 2,
            head_dim: 2,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, 3, 4], f),
    );
    let loss = g.reduce(out, ReduceOp::Sum, vec![0, 1, 2], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_attention_loss(mask_kind: MaskKind) -> Graph {
    const B: usize = 1;
    const H: usize = 2;
    const S: usize = 3;
    const D: usize = 2;
    let f = DType::F32;
    let mut g = Graph::new("attn_ho");
    let q = g.input("q", Shape::new(&[B, H, S, D], f));
    let k = g.input("k", Shape::new(&[B, H, S, D], f));
    let v = g.input("v", Shape::new(&[B, H, S, D], f));
    let mut inputs = vec![q, k, v];
    if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
        let mask = g.input("mask", Shape::new(&[B, H, S, S], f));
        inputs.push(mask);
    }
    let out = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: H,
            head_dim: D,
            mask_kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        inputs,
        Shape::new(&[B, H, S, D], f),
    );
    let loss = g.reduce(
        out,
        ReduceOp::Sum,
        vec![0, 1, 2, 3],
        false,
        Shape::scalar(f),
    );
    g.set_outputs(vec![loss]);
    g
}

fn attn_mask_bytes(mask_kind: MaskKind) -> Option<Vec<f32>> {
    const B: usize = 1;
    const H: usize = 2;
    const S: usize = 3;
    let n = B * H * S * S;
    match mask_kind {
        MaskKind::Custom => Some(vec![1.0; n]),
        MaskKind::Bias => Some((0..n).map(|i| 0.01 * (i as f32 - 4.0)).collect()),
        _ => None,
    }
}

fn attn_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 2 * 3 * 2;
    let q: Vec<f32> = (0..n).map(|i| 0.1 * i as f32).collect();
    let k: Vec<f32> = (0..n).map(|i| (0.07 * i as f32).sin()).collect();
    let v: Vec<f32> = (0..n).map(|i| (0.05 * i as f32).cos()).collect();
    (q, k, v)
}

fn rms_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..8).map(|i| 0.1 * (i as f32 - 3.0)).collect();
    let gamma: Vec<f32> = (0..4).map(|i| 0.5 + 0.2 * i as f32).collect();
    let beta = vec![0.01; 4];
    (x, gamma, beta)
}

fn conv_inputs() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..16).map(|i| 0.05 * i as f32 - 0.2).collect();
    let w: Vec<f32> = (0..9).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    (x, w)
}

fn dynamic_conv_inputs(batch: usize) -> (Vec<f32>, Vec<f32>) {
    let n = batch * 16;
    let x: Vec<f32> = (0..n).map(|i| 0.05 * i as f32 - 0.2).collect();
    let w: Vec<f32> = (0..9).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    (x, w)
}

fn conv_groups_inputs() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..32).map(|i| 0.04 * i as f32 - 0.3).collect();
    let w: Vec<f32> = (0..18).map(|i| 0.08 * (i as f32 + 1.0)).collect();
    (x, w)
}

fn maxpool_inputs() -> Vec<f32> {
    (0..16).map(|i| 0.06 * i as f32 - 0.1).collect()
}

fn conv3d_inputs() -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..64).map(|i| 0.05 * i as f32 - 0.2).collect();
    let w: Vec<f32> = (0..27).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    (x, w)
}

fn maxpool3d_inputs() -> Vec<f32> {
    (0..64).map(|i| 0.06 * i as f32 - 0.1).collect()
}

fn gn_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..16).map(|i| 0.1 * i as f32 - 0.5).collect();
    let gamma: Vec<f32> = (0..4).map(|i| 0.8 + 0.05 * i as f32).collect();
    let beta = vec![0.02; 4];
    (x, gamma, beta)
}

fn rope_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..8).map(|i| 0.2 * (i as f32).sin()).collect();
    let cos: Vec<f32> = (0..2).map(|i| (0.5 * i as f32).cos()).collect();
    let sin: Vec<f32> = (0..2).map(|i| (0.5 * i as f32).sin()).collect();
    (x, cos, sin)
}

fn cumsum_inputs() -> Vec<f32> {
    (0..8).map(|i| 0.08 * i as f32 - 0.2).collect()
}

fn gather_inputs() -> (Vec<f32>, Vec<f32>) {
    let table: Vec<f32> = (0..8).map(|i| 0.1 * i as f32).collect();
    let indices = vec![0.0, 2.0, 1.0];
    (table, indices)
}

fn sce_inputs() -> (Vec<f32>, Vec<f32>) {
    let logits: Vec<f32> = (0..8).map(|i| 0.15 * i as f32 - 0.4).collect();
    let labels = vec![0.0, 1.0];
    (logits, labels)
}

fn fake_quant_inputs() -> Vec<f32> {
    (0..8).map(|i| 0.12 * i as f32 - 0.3).collect()
}

fn scan_inputs() -> (Vec<f32>, Vec<f32>) {
    let init = vec![0.1, -0.2];
    let xs: Vec<f32> = (0..6).map(|i| 0.05 * i as f32).collect();
    (init, xs)
}

fn scan_long_inputs() -> (Vec<f32>, Vec<f32>) {
    let init = vec![0.1, -0.2];
    let xs: Vec<f32> = (0..260).map(|i| 0.01 * (i as f32).sin()).collect();
    (init, xs)
}

fn scan_checkpoint_inputs() -> (Vec<f32>, Vec<f32>) {
    let init = vec![0.1, -0.2];
    let xs: Vec<f32> = (0..8).map(|i| 0.04 * i as f32 - 0.1).collect();
    (init, xs)
}

fn attn_rank3_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..12).map(|i| 0.07 * i as f32).collect();
    let k: Vec<f32> = (0..12).map(|i| (0.05 * i as f32).sin()).collect();
    let v: Vec<f32> = (0..12).map(|i| (0.04 * i as f32).cos()).collect();
    (q, k, v)
}

fn second_order_rms_norm(device: Device) {
    let forward = build_rms_norm_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, gamma, beta) = rms_inputs();
    let x_b = f32_bytes(&x);
    let g_b = f32_bytes(&gamma);
    let b_b = f32_bytes(&beta);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("gamma", g_b.as_slice(), DType::F32),
        ("beta", b_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "rms_norm 2nd");
}

fn second_order_layer_norm(device: Device) {
    let forward = build_layer_norm_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, gamma, beta) = rms_inputs();
    let x_b = f32_bytes(&x);
    let g_b = f32_bytes(&gamma);
    let b_b = f32_bytes(&beta);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("gamma", g_b.as_slice(), DType::F32),
        ("beta", b_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "layer_norm 2nd");
}

fn second_order_conv2d(device: Device) {
    let forward = build_conv2d_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, w) = conv_inputs();
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "conv2d 2nd");
}

fn second_order_conv2d_w(device: Device) {
    let forward = build_conv2d_loss();
    let hg = nth_order_grad(&forward, "w", 2);
    let (x, w) = conv_inputs();
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "conv2d w 2nd");
}

fn second_order_dynamic_conv_w(device: Device) {
    let forward = build_dynamic_conv2d_loss();
    let hg = bind_dynamic_conv_graph(nth_order_grad(&forward, "w", 2), 2);
    let (x, w) = dynamic_conv_inputs(2);
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "dynamic conv w 2nd");
}

fn second_order_dynamic_conv_x(device: Device) {
    let forward = build_dynamic_conv2d_loss();
    let hg = bind_dynamic_conv_graph(nth_order_grad(&forward, "x", 2), 2);
    let (x, w) = dynamic_conv_inputs(2);
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "dynamic conv x 2nd");
}

fn second_order_conv2d_groups_w(device: Device) {
    let forward = build_conv2d_groups_loss();
    let hg = nth_order_grad(&forward, "w", 2);
    let (x, w) = conv_groups_inputs();
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "conv2d groups w 2nd");
}

fn second_order_maxpool(device: Device) {
    let forward = build_maxpool_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let x = maxpool_inputs();
    let x_b = f32_bytes(&x);
    let inputs = [("x", x_b.as_slice(), DType::F32)];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "maxpool 2nd");
}

fn second_order_conv3d(device: Device) {
    let forward = build_conv3d_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, w) = conv3d_inputs();
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "conv3d 2nd");
}

fn second_order_conv3d_w(device: Device) {
    let forward = build_conv3d_loss();
    let hg = nth_order_grad(&forward, "w", 2);
    let (x, w) = conv3d_inputs();
    let x_b = f32_bytes(&x);
    let w_b = f32_bytes(&w);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("w", w_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "conv3d w 2nd");
}

fn second_order_maxpool3d(device: Device) {
    let forward = build_maxpool3d_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let x = maxpool3d_inputs();
    let x_b = f32_bytes(&x);
    let inputs = [("x", x_b.as_slice(), DType::F32)];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "maxpool3d 2nd");
}

fn second_order_group_norm(device: Device) {
    let forward = build_group_norm_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, gamma, beta) = gn_inputs();
    let x_b = f32_bytes(&x);
    let g_b = f32_bytes(&gamma);
    let b_b = f32_bytes(&beta);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("gamma", g_b.as_slice(), DType::F32),
        ("beta", b_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "group_norm 2nd");
}

fn second_order_group_norm_gamma(device: Device) {
    let forward = build_group_norm_loss();
    let hg = nth_order_grad(&forward, "gamma", 2);
    let (x, gamma, beta) = gn_inputs();
    let x_b = f32_bytes(&x);
    let g_b = f32_bytes(&gamma);
    let b_b = f32_bytes(&beta);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("gamma", g_b.as_slice(), DType::F32),
        ("beta", b_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "group_norm gamma 2nd");
}

fn second_order_group_norm_beta(device: Device) {
    let forward = build_group_norm_loss();
    let hg = nth_order_grad(&forward, "beta", 2);
    let (x, gamma, beta) = gn_inputs();
    let x_b = f32_bytes(&x);
    let g_b = f32_bytes(&gamma);
    let b_b = f32_bytes(&beta);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("gamma", g_b.as_slice(), DType::F32),
        ("beta", b_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "group_norm beta 2nd");
}

fn second_order_rope(device: Device) {
    let forward = build_rope_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let (x, cos, sin) = rope_inputs();
    let x_b = f32_bytes(&x);
    let c_b = f32_bytes(&cos);
    let s_b = f32_bytes(&sin);
    let inputs = [
        ("x", x_b.as_slice(), DType::F32),
        ("cos", c_b.as_slice(), DType::F32),
        ("sin", s_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "rope 2nd");
}

fn second_order_cumsum(device: Device) {
    let forward = build_cumsum_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let x = cumsum_inputs();
    let x_b = f32_bytes(&x);
    let inputs = [("x", x_b.as_slice(), DType::F32)];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "cumsum 2nd");
}

fn second_order_gather(device: Device) {
    let forward = build_gather_loss();
    let hg = nth_order_grad(&forward, "table", 2);
    let (table, indices) = gather_inputs();
    let t_b = f32_bytes(&table);
    let i_b = f32_bytes(&indices);
    let inputs = [
        ("table", t_b.as_slice(), DType::F32),
        ("indices", i_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "gather 2nd");
}

fn second_order_sce(device: Device) {
    let forward = build_sce_loss();
    let (logits, labels) = sce_inputs();
    let l_b = f32_bytes(&logits);
    let y_b = f32_bytes(&labels);
    let inputs = [
        ("logits", l_b.as_slice(), DType::F32),
        ("labels", y_b.as_slice(), DType::F32),
    ];
    let g1 = nth_order_grad(&forward, "logits", 1);
    assert_vec_matches_cpu(device, g1, &inputs, 5e-3, "sce 1st");
    let hg = nth_order_grad(&forward, "logits", 2);
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "sce 2nd");
}

fn second_order_fake_quant(device: Device) {
    let forward = build_fake_quant_loss();
    let hg = nth_order_grad(&forward, "x", 2);
    let x = fake_quant_inputs();
    let x_b = f32_bytes(&x);
    let inputs = [("x", x_b.as_slice(), DType::F32)];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "fake_quant 2nd");
}

fn second_order_scan_checkpoint(device: Device) {
    let forward = build_scan_checkpoint_loss();
    let hg = nth_order_grad(&forward, "xs", 2);
    let (init, xs) = scan_checkpoint_inputs();
    let i_b = f32_bytes(&init);
    let x_b = f32_bytes(&xs);
    let inputs = [
        ("init", i_b.as_slice(), DType::F32),
        ("xs", x_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "scan_ckpt 2nd");
}

fn second_order_scan(device: Device) {
    let forward = build_scan_loss();
    let hg = nth_order_grad(&forward, "xs", 2);
    let (init, xs) = scan_inputs();
    let i_b = f32_bytes(&init);
    let x_b = f32_bytes(&xs);
    let inputs = [
        ("init", i_b.as_slice(), DType::F32),
        ("xs", x_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "scan 2nd");
}

fn second_order_scan_long(device: Device) {
    let forward = build_scan_long_loss();
    let hg = nth_order_grad(&forward, "xs", 2);
    let (init, xs) = scan_long_inputs();
    let i_b = f32_bytes(&init);
    let x_b = f32_bytes(&xs);
    let inputs = [
        ("init", i_b.as_slice(), DType::F32),
        ("xs", x_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 5e-3, "scan_long 2nd");
}

fn second_order_attention_rank3(device: Device) {
    let forward = build_attention_rank3_loss();
    let hg = nth_order_grad(&forward, "q", 2);
    let (q, k, v) = attn_rank3_inputs();
    let q_b = f32_bytes(&q);
    let k_b = f32_bytes(&k);
    let v_b = f32_bytes(&v);
    let inputs = [
        ("q", q_b.as_slice(), DType::F32),
        ("k", k_b.as_slice(), DType::F32),
        ("v", v_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, 1e-2, "attention rank3 2nd");
}

fn second_order_attention_q(device: Device, mask_kind: MaskKind, tol: f32, label: &str) {
    let forward = build_attention_loss(mask_kind);
    let hg = nth_order_grad(&forward, "q", 2);
    let (q, k, v) = attn_inputs();
    let q_b = f32_bytes(&q);
    let k_b = f32_bytes(&k);
    let v_b = f32_bytes(&v);
    if let Some(mask) = attn_mask_bytes(mask_kind) {
        let m_b = f32_bytes(&mask);
        let inputs = [
            ("q", q_b.as_slice(), DType::F32),
            ("k", k_b.as_slice(), DType::F32),
            ("v", v_b.as_slice(), DType::F32),
            ("mask", m_b.as_slice(), DType::F32),
        ];
        assert_vec_matches_cpu(device, hg, &inputs, tol, label);
        return;
    }
    let inputs = [
        ("q", q_b.as_slice(), DType::F32),
        ("k", k_b.as_slice(), DType::F32),
        ("v", v_b.as_slice(), DType::F32),
    ];
    assert_vec_matches_cpu(device, hg, &inputs, tol, label);
}

fn second_order_attention_none_q(device: Device) {
    second_order_attention_q(device, MaskKind::None, 5e-2, "attention q 2nd");
}

fn second_order_attention_causal_q(device: Device) {
    second_order_attention_q(device, MaskKind::Causal, 5e-2, "attention causal q 2nd");
}

fn second_order_attention_sliding_window_q(device: Device) {
    second_order_attention_q(
        device,
        MaskKind::SlidingWindow(1),
        5e-2,
        "attention sliding-window q 2nd",
    );
}

fn second_order_attention_custom_q(device: Device) {
    second_order_attention_q(device, MaskKind::Custom, 5e-2, "attention custom q 2nd");
}

fn second_order_attention_bias_q(device: Device) {
    second_order_attention_q(device, MaskKind::Bias, 5e-2, "attention bias q 2nd");
}

macro_rules! decompose_parity_suite {
    ($mod_name:ident, $device:expr, $($cfg:meta),+) => {
        $(#[$cfg])*
        mod $mod_name {
            use super::*;
            #[test]
            fn rms_norm_second_derivative() {
                second_order_rms_norm($device);
            }
            #[test]
            fn layer_norm_second_derivative() {
                second_order_layer_norm($device);
            }
            #[test]
            fn conv2d_second_derivative() {
                second_order_conv2d($device);
            }
            #[test]
            fn conv2d_w_second_derivative() {
                second_order_conv2d_w($device);
            }
            #[test]
            fn dynamic_conv_w_second_derivative() {
                second_order_dynamic_conv_w($device);
            }
            #[test]
            fn dynamic_conv_x_second_derivative() {
                second_order_dynamic_conv_x($device);
            }
            #[test]
            fn conv2d_groups_w_second_derivative() {
                second_order_conv2d_groups_w($device);
            }
            #[test]
            fn maxpool_second_derivative() {
                second_order_maxpool($device);
            }
            #[test]
            fn conv3d_second_derivative() {
                second_order_conv3d($device);
            }
            #[test]
            fn conv3d_w_second_derivative() {
                second_order_conv3d_w($device);
            }
            #[test]
            fn maxpool3d_second_derivative() {
                second_order_maxpool3d($device);
            }
            #[test]
            fn group_norm_second_derivative() {
                second_order_group_norm($device);
            }
            #[test]
            fn group_norm_gamma_second_derivative() {
                second_order_group_norm_gamma($device);
            }
            #[test]
            fn group_norm_beta_second_derivative() {
                second_order_group_norm_beta($device);
            }
            #[test]
            fn rope_second_derivative() {
                second_order_rope($device);
            }
            #[test]
            fn cumsum_second_derivative() {
                second_order_cumsum($device);
            }
            #[test]
            fn gather_second_derivative() {
                second_order_gather($device);
            }
            #[test]
            fn sce_second_derivative() {
                second_order_sce($device);
            }
            #[test]
            fn fake_quant_second_derivative() {
                second_order_fake_quant($device);
            }
            #[test]
            fn scan_second_derivative() {
                second_order_scan($device);
            }
            #[test]
            fn scan_long_second_derivative() {
                second_order_scan_long($device);
            }
            #[test]
            fn scan_checkpoint_second_derivative() {
                second_order_scan_checkpoint($device);
            }
            #[test]
            fn attention_rank3_second_derivative() {
                second_order_attention_rank3($device);
            }
            #[test]
            fn attention_second_derivative() {
                second_order_attention_none_q($device);
            }
            #[test]
            fn attention_causal_second_derivative() {
                second_order_attention_causal_q($device);
            }
            #[test]
            fn attention_sliding_window_second_derivative() {
                second_order_attention_sliding_window_q($device);
            }
            #[test]
            fn attention_custom_second_derivative() {
                second_order_attention_custom_q($device);
            }
            #[test]
            fn attention_bias_second_derivative() {
                second_order_attention_bias_q($device);
            }
        }
    };
}

decompose_parity_suite!(cuda, Device::Cuda, cfg(feature = "cuda"));
decompose_parity_suite!(rocm, Device::Rocm, cfg(feature = "rocm"));
decompose_parity_suite!(wgpu, Device::Gpu, cfg(feature = "gpu"));
decompose_parity_suite!(
    metal,
    Device::Metal,
    cfg(all(feature = "metal", target_os = "macos"))
);
decompose_parity_suite!(
    mlx,
    Device::Mlx,
    cfg(all(feature = "mlx", target_os = "macos"))
);
