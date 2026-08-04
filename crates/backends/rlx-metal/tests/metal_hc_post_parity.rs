// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Regression for the DeepSeek-V4 `build_hc_post` output-tail zeroing on the
//! Metal **MPSGraph** path (`dsv4_paged_generate --attn-gpu`). The hyper-connection
//! tail — a 4D-broadcast `Mul`, a mid-axis `Reduce{Sum, keep_dim:false}` over a
//! `[rows,hc,hc,d]` tensor, and an `Add`, consumed only by a graph-output
//! `Reshape` — read back as ZEROS under MPSGraph's whole-graph optimizer even
//! though every input was correct (the same op mid-graph, and the whole graph on
//! the per-op thunk path, are exact). `graph_has_mps_hostile_reduce` detects the
//! interior rank≥4 reduction so both the full plan and the hybrid segmenter fall
//! back to the always-correct thunk path.

#![cfg(target_os = "macos")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// hc_post tail as the DEEP output of a high-FLOP graph. Two outputs:
///   0: hidden = reshape(hc_post(...))   — deep tail (the one that zeroed)
///   1: shallow = rms_norm(matmul(x,w))  — the "moe_in" analog (stays correct)
fn build_graph(rows: usize, hc: usize, d: usize) -> Graph {
    let mut g = Graph::new("hc_post_mps_repro");
    let f = DType::F32;
    let (r, h, dd) = (rows as i64, hc as i64, d as i64);

    let x = g.input("x", Shape::new(&[rows, d], f));
    let w = g.input("w", Shape::new(&[d, d], f));
    let moe_out = g.input("moe_out", Shape::new(&[rows, d], f));
    let norm_w = g.input("norm_w", Shape::new(&[d], f));
    let zb = g.input("zb", Shape::new(&[d], f));

    // High-FLOP matmul so estimated_max_flops() >= 1e6 → MPSGraph path.
    let mm = g.matmul(x, w, Shape::new(&[rows, d], f)); // [rows,d]
    let shallow = g.rms_norm(mm, norm_w, zb, 1e-6); // [rows,d]  (shallow output)

    // residual [rows,hc,d]: broadcast mm across the hc streams (computed node).
    let mm3 = g.reshape_(mm, vec![r, 1, dd]);
    let ones = g.input("ones_hc", Shape::new(&[1, hc, 1], f));
    let residual = g.mul(mm3, ones); // [rows,hc,d]

    // hc_post inline (mirrors build_hc_post exactly).
    let post = g.input("post", Shape::new(&[rows * hc], f));
    let comb = g.input("comb", Shape::new(&[rows * hc * hc], f));
    let post3 = g.reshape_(post, vec![r, h, 1]);
    let xo3 = g.reshape_(moe_out, vec![r, 1, dd]);
    let term1 = g.mul(post3, xo3); // [r,hc,d]
    let comb4 = g.reshape_(comb, vec![r, h, h, 1]);
    let res4 = g.reshape_(residual, vec![r, 1, h, dd]);
    let prod = g.mul(comb4, res4); // [r,hc,hc,d]
    let term2 = g.sum(prod, vec![2], false); // interior rank-4 reduce → [r,hc,d]
    let hh = g.add(term1, term2); // [r,hc,d]
    let hidden = g.reshape_(hh, vec![r, h, dd]);

    g.set_outputs(vec![hidden, shallow]);
    g
}

fn feed(rows: usize, hc: usize, d: usize) -> Vec<(&'static str, Vec<f32>)> {
    let n = |len: usize, seed: f32, step: f32, m: usize| {
        (0..len)
            .map(|i| seed + step * (i % m) as f32)
            .collect::<Vec<f32>>()
    };
    vec![
        ("x", n(rows * d, 0.5, 0.01, 13)),
        ("moe_out", n(rows * d, 1.0, 0.02, 11)),
        ("w", n(d * d, 0.01, 0.003, 7)),
        ("norm_w", vec![1.0; d]),
        ("zb", vec![0.0; d]),
        ("ones_hc", vec![1.0; hc]),
        ("post", n(rows * hc, 0.3, 0.1, 5)),
        ("comb", n(rows * hc * hc, 0.2, 0.05, 5)),
    ]
}

fn run_on(dev: Device, rows: usize, hc: usize, d: usize) -> Vec<Vec<f32>> {
    let mut s = Session::new(dev).compile(build_graph(rows, hc, d));
    let f = feed(rows, hc, d);
    let refs: Vec<(&str, &[f32])> = f.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    s.run(&refs)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

/// The detector must flag the hc_post interior rank-4 reduce (so MPSGraph is
/// skipped) but NOT a trailing / low-rank reduction (which MPSGraph handles).
#[test]
fn hostile_reduce_detector_flags_interior_rank4_only() {
    use rlx_metal::mps_graph_lower::graph_has_mps_hostile_reduce;
    let f = DType::F32;

    // hc_post `Σ_k`: rank-4 reduce over axis 2 with axis 3 kept → interior.
    assert!(
        graph_has_mps_hostile_reduce(&build_graph(1, 4, 32)),
        "hc_post interior rank-4 reduce must be flagged"
    );

    // Trailing reduce: [N,C,H,W] over [2,3] (both trailing) → NOT interior.
    let mut gt = Graph::new("trailing");
    let x = gt.input("x", Shape::new(&[2, 3, 4, 5], f));
    let y = gt.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![2, 3],
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[2, 3], f),
    );
    gt.set_outputs(vec![y]);
    assert!(
        !graph_has_mps_hostile_reduce(&gt),
        "trailing [2,3] reduce of a rank-4 tensor must NOT be flagged"
    );

    // Last-axis reduce (rank 3, the RmsNorm/softmax shape) → NOT interior.
    let mut gl = Graph::new("lastaxis");
    let x = gl.input("x", Shape::new(&[1, 4, 8], f));
    let y = gl.sum(x, vec![2], false);
    gl.set_outputs(vec![y]);
    assert!(
        !graph_has_mps_hostile_reduce(&gl),
        "rank-3 last-axis reduce must NOT be flagged"
    );
}

/// End-to-end parity: with the guard active the hc_post graph runs on the thunk
/// path and matches CPU (and is non-zero — the symptom of the original bug).
#[test]
fn hc_post_metal_matches_cpu() {
    let (rows, hc, d) = (1usize, 4usize, 256usize);
    let cpu = run_on(Device::Cpu, rows, hc, d);
    let metal = run_on(Device::Metal, rows, hc, d);
    for (i, name) in ["hidden(deep)", "shallow"].iter().enumerate() {
        eprintln!(
            "  {name}: cpu_rms={:.5} metal_rms={:.5}",
            rms(&cpu[i]),
            rms(&metal[i])
        );
    }
    assert!(
        rms(&metal[0]) > 1e-4,
        "hidden output must be non-zero (bug symptom was all-zero)"
    );
    for (i, name) in ["hidden(deep)", "shallow"].iter().enumerate() {
        let md = cpu[i]
            .iter()
            .zip(&metal[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            md < 1e-3,
            "output '{name}' diverges: max|cpu-metal|={md:.5} cpu_rms={:.5} metal_rms={:.5}",
            rms(&cpu[i]),
            rms(&metal[i])
        );
    }
}
