// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gather on axis 0 of a tensor a rank-changing `Reshape` produced.
//!
//! `fuse_rms_norm_reshape` folds `rms_norm([…, H]) → reshape([∏leading, H])`
//! into one `RmsNorm` whose output has a DIFFERENT RANK than its input. MLX's
//! lowering returned `ops::rms_norm(x, …)`, i.e. `x`'s shape, so the flattening
//! was silently dropped: the following `gather(axis=0)` then indexed the batch
//! axis and handed a rank-3 result to a matmul, which surfaced far away as
//! "[reshape] Cannot reshape array of size 984 into shape (1,3,8)". That is why
//! MLX could not compile an `rlx-kev` training graph at all.
//!
//! The reshape's parent has to be a COMPUTED node for the fusion to fire — with
//! a graph `Input` there the pass declines and this passes vacuously, which is
//! exactly how the first attempt at this test missed the bug.
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build(batch: usize, seq: usize, d: usize, k: usize, n_out: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gather_after_reshape");
    let x = g.input("x", Shape::new(&[batch, seq, d], f));
    // The reshape's parent is a COMPUTED rank-3 node, as in kev (the trunk's
    // normed hidden states), not a graph input.
    let gam = g.param("gam", Shape::new(&[d], f));
    let bet = g.param("bet", Shape::new(&[d], f));
    let h = g.rms_norm(x, gam, bet, 1e-6);
    // Drops the leading 1-dim: the shape kev's readout flattens hidden states to.
    let flat = g.reshape_(h, vec![(batch * seq) as i64, d as i64]);
    let idx = g.input("idx", Shape::new(&[k], f));
    let rows = g.gather_(flat, idx, 0);
    let w = g.param("w", Shape::new(&[d, n_out], f));
    let mm = g.matmul(rows, w, Shape::new(&[k, n_out], f));
    // matmul + bias fuses to `FusedMatMulBiasAct`, which is what kev's pointer
    // head emits and where the shape went wrong.
    let b = g.param("b", Shape::new(&[n_out], f));
    let y = g.add(mm, b);
    // ...then reshaped back to rank 3, which is the node MLX refused to lower.
    let y3 = g.reshape_(y, vec![1, k as i64, n_out as i64]);
    g.set_outputs(vec![y3]);
    g
}

#[test]
fn gather_after_rank_changing_reshape_matches_cpu() {
    let (batch, seq, d, k, n_out) = (1usize, 41usize, 16usize, 3usize, 8usize);
    let g = build(batch, seq, d, k, n_out);
    let xv: Vec<f32> = (0..batch * seq * d)
        .map(|i| 0.01 * ((i % 23) as f32 - 11.0))
        .collect();
    let idxv: Vec<f32> = vec![0.0, 7.0, 40.0];
    let wv: Vec<f32> = (0..d * n_out)
        .map(|i| 0.05 * ((i % 9) as f32 - 4.0))
        .collect();
    let bv: Vec<f32> = (0..n_out).map(|i| 0.01 * i as f32).collect();
    let gamv: Vec<f32> = vec![1.0; d];
    let betv: Vec<f32> = vec![0.0; d];
    let run = |dev: Device| {
        let mut s = Session::new(dev).compile(g.clone());
        s.set_param("w", &wv);
        s.set_param("b", &bv);
        s.set_param("gam", &gamv);
        s.set_param("bet", &betv);
        s.run(&[("x", &xv[..]), ("idx", &idxv[..])])[0].clone()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    println!("cpu {} elems, mlx {} elems", cpu.len(), mlx.len());
    assert_eq!(
        mlx.len(),
        k * n_out,
        "MLX produced {} elements for a [{k},{n_out}] output — the reshape that \
         dropped the leading 1-dim did not take, so the gather indexed the batch axis",
        mlx.len()
    );
    let err = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-5, "mlx differs by {err}");
}
