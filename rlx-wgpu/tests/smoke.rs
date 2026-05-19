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

//! End-to-end matmul smoke test for the wgpu backend.
//!
//! v1 only lowers MatMul (plus leaf nodes), so the test is a single
//! 2D matmul against a hand-computed reference.

use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn build_graph() -> Graph {
    let mut g = Graph::new("smoke");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn matmul_ref(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += x[i * k + kk] * w[kk * n + j];
            }
            y[i * n + j] = s;
        }
    }
    y
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn binary_add_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);
    let mut exe = WgpuExecutable::compile(g);
    let out = exe.run(&[
        ("x", &[1.0, 2.0, 3.0, 4.0]),
        ("y", &[10.0, 20.0, 30.0, 40.0]),
    ]);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn binary_max_min_pow_match_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    for (op, want) in [
        (BinaryOp::Max, vec![3.0, 4.0, 3.0, 4.0]),
        (BinaryOp::Min, vec![1.0, 2.0, 1.0, 2.0]),
        (BinaryOp::Pow, vec![1.0, 16.0, 3.0, 16.0]), // a^b where a=[1,2,3,4], b=[3,4,1,2]
    ] {
        let mut g = Graph::new("bin");
        let a = g.input("a", Shape::new(&[4], DType::F32));
        let b = g.input("b", Shape::new(&[4], DType::F32));
        let c = g.binary(op, a, b, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![c]);
        let mut exe = WgpuExecutable::compile(g);
        let out = exe.run(&[("a", &[1.0, 2.0, 3.0, 4.0]), ("b", &[3.0, 4.0, 1.0, 2.0])]);
        assert!(
            close(&out[0], &want, 1e-4),
            "Binary({op:?}) mismatch: got {:?} want {want:?}",
            out[0]
        );
    }
}

#[test]
fn activations_relu_silu_match_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("act");
    let x = g.input("x", Shape::new(&[5], DType::F32));
    let r = g.activation(Activation::Relu, x, Shape::new(&[5], DType::F32));
    let s = g.activation(Activation::Silu, r, Shape::new(&[5], DType::F32));
    g.set_outputs(vec![s]);
    let mut exe = WgpuExecutable::compile(g);
    let xs = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
    let out = exe.run(&[("x", &xs)]);
    // relu([-2, -0.5, 0, 1, 3]) = [0, 0, 0, 1, 3]
    // silu(0)=0, silu(1)=1/(1+e^-1)≈0.7311, silu(3)=3/(1+e^-3)≈2.857
    let want = vec![0.0, 0.0, 0.0, 0.7311, 2.857];
    assert!(
        close(&out[0], &want, 1e-2),
        "Relu+Silu mismatch: got {:?} want {want:?}",
        out[0]
    );
}

#[test]
fn compare_then_where_implements_abs() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("cw");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let z = g.input("z", Shape::new(&[4], DType::F32));
    let nx = g.activation(Activation::Neg, x, Shape::new(&[4], DType::F32));
    let cond = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![x, z],
        Shape::new(&[4], DType::Bool),
    );
    let out = g.add_node(Op::Where, vec![cond, x, nx], Shape::new(&[4], DType::F32));
    g.set_outputs(vec![out]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, -2.0, 3.0, -4.0]), ("z", &[0.0, 0.0, 0.0, 0.0])]);
    assert_eq!(r[0], vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn reduce_sum_last_axis_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("rsum");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.reduce(
        x,
        ReduceOp::Sum,
        vec![1],
        false,
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(r[0], vec![6.0, 15.0]);
}

#[test]
fn reduce_mean_max_min_match_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    for (op, want) in [
        (ReduceOp::Mean, vec![2.0, 5.0]),
        (ReduceOp::Max, vec![3.0, 6.0]),
        (ReduceOp::Min, vec![1.0, 4.0]),
    ] {
        let mut g = Graph::new("red");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        let y = g.reduce(x, op, vec![1], false, Shape::new(&[2], DType::F32));
        g.set_outputs(vec![y]);
        let mut exe = WgpuExecutable::compile(g);
        let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        assert!(
            close(&r[0], &want, 1e-5),
            "Reduce({op:?}) mismatch: got {:?} want {want:?}",
            r[0]
        );
    }
}

#[test]
fn softmax_last_axis_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("smx");
    let x = g.input("x", Shape::new(&[1, 3], DType::F32));
    let y = g.softmax(x, -1, Shape::new(&[1, 3], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // softmax([1, 2, 3]):
    //   shift by max=3: [-2, -1, 0]
    //   exp:           [0.1353, 0.3679, 1.0]
    //   sum:           1.5032
    //   normalized:    [0.0900, 0.2447, 0.6652]
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0])]);
    let want = vec![0.0900, 0.2447, 0.6652];
    assert!(
        close(&r[0], &want, 1e-3),
        "softmax mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn layer_norm_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("ln");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let ga = g.param("g", Shape::new(&[4], DType::F32));
    let be = g.param("b", Shape::new(&[4], DType::F32));
    let y = g.layer_norm(x, ga, be, -1, 1e-5, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("g", &[1.0, 1.0, 1.0, 1.0]);
    exe.set_param("b", &[0.0, 0.0, 0.0, 0.0]);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0];
    let r = exe.run(&[("x", &xs)]);
    // Per row: subtract mean, divide by sqrt(var + eps)
    let mut want = vec![0f32; 8];
    for row in 0..2 {
        let off = row * 4;
        let mean = (0..4).map(|i| xs[off + i]).sum::<f32>() / 4.0;
        let var = (0..4).map(|i| (xs[off + i] - mean).powi(2)).sum::<f32>() / 4.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..4 {
            want[off + i] = (xs[off + i] - mean) * inv;
        }
    }
    assert!(
        close(&r[0], &want, 1e-3),
        "layer_norm mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn cumsum_inclusive_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("cs");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.cumsum(x, 0, false, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    assert_eq!(r[0], vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn reshape_passes_data_through() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("rs");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.reshape(x, vec![3, 2], Shape::new(&[3, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(r[0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_2x3_to_3x2_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("tr");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![x],
        Shape::new(&[3, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // Input layout (row-major):
    //   1 2 3
    //   4 5 6
    // Transpose:
    //   1 4
    //   2 5
    //   3 6
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    assert_eq!(r[0], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn transpose_bhsd_layout_swap_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // [B=1, H=2, S=2, D=2] → perm [0, 2, 1, 3] → [B, S, H, D]
    let mut g = Graph::new("tr4");
    let x = g.input("x", Shape::new(&[1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![x],
        Shape::new(&[1, 2, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // Input [B, H, S, D]: indexed (h, s, d). With H=2,S=2,D=2:
    //   h0s0d0=1  h0s0d1=2  h0s1d0=3  h0s1d1=4
    //   h1s0d0=5  h1s0d1=6  h1s1d0=7  h1s1d1=8
    // After perm to [B, S, H, D]:
    //   s=0,h=0,d=0..1 = 1,2  ; s=0,h=1,d=0..1 = 5,6
    //   s=1,h=0,d=0..1 = 3,4  ; s=1,h=1,d=0..1 = 7,8
    let xs: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xs)]);
    let want = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    assert_eq!(r[0], want);
}

#[test]
fn narrow_axis2_slice_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // [1, 1, 4] → narrow(axis=2, start=1, len=2) → [1, 1, 2]
    let mut g = Graph::new("nrw");
    let x = g.input("x", Shape::new(&[1, 1, 4], DType::F32));
    let y = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 1,
            len: 2,
        },
        vec![x],
        Shape::new(&[1, 1, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[10.0, 20.0, 30.0, 40.0])]);
    assert_eq!(r[0], vec![20.0, 30.0]);
}

#[test]
fn concat_axis_minus_one_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("cat");
    let a = g.input("a", Shape::new(&[2, 2], DType::F32));
    let b = g.input("b", Shape::new(&[2, 3], DType::F32));
    let y = g.concat(vec![a, b], 1, Shape::new(&[2, 5], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[
        ("a", &[1.0, 2.0, 3.0, 4.0]),
        ("b", &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    ]);
    // Row-major output: row0 = [1, 2, 10, 20, 30]; row1 = [3, 4, 40, 50, 60]
    assert_eq!(
        r[0],
        vec![1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn gather_embedding_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("gat");
    let table = g.param("t", Shape::new(&[3, 2], DType::F32));
    let idx = g.input("i", Shape::new(&[2], DType::F32));
    let y = g.gather(table, idx, 0, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("t", &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let r = exe.run(&[("i", &[2.0, 0.0])]);
    // Index 2 → row 2 = [30, 31]; Index 0 → row 0 = [10, 11]
    assert_eq!(r[0], vec![30.0, 31.0, 10.0, 11.0]);
}

#[test]
fn attention_no_mask_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Tiny attention: B=1, H=1, S=2, D=2. Same hand-computed
    // reference we used for MLX, since the op semantics are
    // identical.
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[1, 1, 2, 2], DType::F32));
    let k = g.input("k", Shape::new(&[1, 1, 2, 2], DType::F32));
    let v = g.input("v", Shape::new(&[1, 1, 2, 2], DType::F32));
    let o = g.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: 2,
            mask_kind: MaskKind::None,
        },
        vec![q, k, v],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![o]);
    let mut exe = WgpuExecutable::compile(g);
    let qd = vec![1.0, 0.0, 0.0, 1.0];
    let kd = vec![1.0, 0.0, 0.0, 1.0];
    let vd = vec![10.0, 20.0, 30.0, 40.0];
    let r = exe.run(&[("q", &qd), ("k", &kd), ("v", &vd)]);
    // scale = 1/sqrt(2), softmax([0.7071, 0]) ≈ [0.6698, 0.3302]
    // row0 = 0.6698 * (10, 20) + 0.3302 * (30, 40) = (16.605, 26.605)
    // row1 = 0.3302 * (10, 20) + 0.6698 * (30, 40) = (23.395, 33.395)
    let want = vec![16.605, 26.605, 23.395, 33.395];
    assert!(
        close(&r[0], &want, 5e-3),
        "attention mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn rope_identity_passes_through() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // cos = 1, sin = 0 → rope is identity.
    let mut g = Graph::new("rope");
    let x = g.input("x", Shape::new(&[1, 1, 1, 4], DType::F32));
    let cos = g.input("cos", Shape::new(&[1, 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[1, 2], DType::F32));
    let y = g.add_node(
        Op::Rope { head_dim: 4 },
        vec![x, cos, sin],
        Shape::new(&[1, 1, 1, 4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[
        ("x", &[1.0, 2.0, 3.0, 4.0]),
        ("cos", &[1.0, 1.0]),
        ("sin", &[0.0, 0.0]),
    ]);
    assert_eq!(r[0], vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rope_90_degree_rotation_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // cos = 0, sin = 1 at all positions → 90° rotation.
    // y_first  = x_first*0 - x_second*1 = -x_second
    // y_second = x_second*0 + x_first*1 = x_first
    let mut g = Graph::new("rope90");
    let x = g.input("x", Shape::new(&[1, 1, 1, 4], DType::F32));
    let cos = g.input("cos", Shape::new(&[1, 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[1, 2], DType::F32));
    let y = g.add_node(
        Op::Rope { head_dim: 4 },
        vec![x, cos, sin],
        Shape::new(&[1, 1, 1, 4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[
        ("x", &[1.0, 2.0, 3.0, 4.0]),
        ("cos", &[0.0, 0.0]),
        ("sin", &[1.0, 1.0]),
    ]);
    // x = [1, 2, 3, 4]; first half=(1,2), second=(3,4)
    // y_first = -second = (-3, -4); y_second = first = (1, 2)
    let want = vec![-3.0, -4.0, 1.0, 2.0];
    assert!(
        close(&r[0], &want, 1e-5),
        "rope90 mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn expand_broadcast_replicates_values() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Input [1, 3] → expand to [2, 3]. Each row of the output is a
    // replica of the single input row.
    let mut g = Graph::new("expand");
    let x = g.input("x", Shape::new(&[1, 3], DType::F32));
    let y = g.add_node(
        Op::Expand {
            target_shape: vec![2, 3],
        },
        vec![x],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0])]);
    assert_eq!(r[0], vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn dot_general_canonical_matches_matmul() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("dg");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.add_node(
        Op::DotGeneral {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        },
        vec![x, w],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    // [(1+0+1.5, 0+2+1.5), (4+0+3, 0+5+3)] = [(2.5, 3.5), (7.0, 8.0)]
    assert!(close(&r[0], &[2.5, 3.5, 7.0, 8.0], 1e-5));
}

#[test]
fn sample_argmax_picks_dominant_logit() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("samp");
    let logits = g.input("l", Shape::new(&[1, 5], DType::F32));
    let id = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![id]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("l", &[0.0, 0.0, 100.0, 0.0, 0.0])]);
    assert_eq!(r[0][0] as i32, 2);
}

#[test]
fn pool_2x2_max_stride_2_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("pool");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![p]);
    let mut exe = WgpuExecutable::compile(g);
    let xs: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xs)]);
    assert_eq!(r[0], vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn conv2d_1x1_identity_matches_input() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // 1x1 conv with weight=1 and groups=1 → identity copy of input.
    let mut g = Graph::new("conv");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1, 1, 1], DType::F32));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![1, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &[1.0]);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    assert_eq!(r[0], vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pool1d_max_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("pool1d");
    let x = g.input("x", Shape::new(&[1, 1, 4], DType::F32));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2],
            stride: vec![2],
            padding: vec![0],
        },
        vec![x],
        Shape::new(&[1, 1, 2], DType::F32),
    );
    g.set_outputs(vec![p]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0, 3.0, 2.0, 4.0])]);
    assert_eq!(r[0], vec![3.0, 4.0]);
}

#[test]
fn pool3d_max_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("pool3d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], DType::F32),
    );
    g.set_outputs(vec![p]);
    let mut exe = WgpuExecutable::compile(g);
    let xs: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xs)]);
    assert_eq!(r[0], vec![8.0]);
}

#[test]
fn conv1d_simple_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("conv1d");
    let x = g.input("x", Shape::new(&[1, 1, 4], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1, 2], DType::F32));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![2],
            stride: vec![1],
            padding: vec![0],
            dilation: vec![1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 1, 3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &[1.0, -1.0]);
    let r = exe.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
    // diff: [1-2, 2-3, 3-4] = [-1, -1, -1]
    assert_eq!(r[0], vec![-1.0, -1.0, -1.0]);
}

#[test]
fn conv3d_1x1x1_identity_matches_input() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("conv3d");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1, 1, 1, 1], DType::F32));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![1, 1, 1],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
            dilation: vec![1, 1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 1, 2, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &[1.0]);
    let xs: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xs)]);
    assert_eq!(r[0], xs);
}

#[test]
fn fused_matmul_bias_act_matches_unfused_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // FusedMatMulBiasAct → MatMul + Add(bias) + Relu via the unfusion pass.
    let mut g = Graph::new("fmb");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2], DType::F32));
    let y = g.fused_matmul_bias_act(
        x,
        w,
        b,
        Some(Activation::Relu),
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wv = vec![0.1, 0.2, 0.3, 0.4, -0.5, 0.6];
    let bv = vec![-2.0, 0.5];
    exe.set_param("w", &wv);
    exe.set_param("b", &bv);
    let r = exe.run(&[("x", &xv)]);
    let mm = matmul_ref(&xv, &wv, 2, 3, 2);
    let want: Vec<f32> = mm
        .iter()
        .enumerate()
        .map(|(i, &v)| (v + bv[i % 2]).max(0.0))
        .collect();
    assert!(
        close(&r[0], &want, 1e-4),
        "FMB mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn fused_residual_ln_matches_unfused_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // FusedResidualLN(x, residual, bias=None, gamma, beta, eps) → Add + LN.
    let mut g = Graph::new("frln");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let r = g.input("r", Shape::new(&[2, 4], DType::F32));
    let ga = g.param("g", Shape::new(&[4], DType::F32));
    let be = g.param("b", Shape::new(&[4], DType::F32));
    let y = g.fused_residual_ln(x, r, None, ga, be, 1e-5, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("g", &[1.0, 1.0, 1.0, 1.0]);
    exe.set_param("b", &[0.0, 0.0, 0.0, 0.0]);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0];
    let rv = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let out = exe.run(&[("x", &xv), ("r", &rv)]);
    // Reference: layer_norm(x + r) row-wise.
    let mut want = vec![0f32; 8];
    for row in 0..2 {
        let off = row * 4;
        let s: Vec<f32> = (0..4).map(|i| xv[off + i] + rv[off + i]).collect();
        let mean = s.iter().sum::<f32>() / 4.0;
        let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..4 {
            want[off + i] = (s[i] - mean) * inv;
        }
    }
    assert!(
        close(&out[0], &want, 1e-3),
        "FusedResidualLN mismatch: got {:?} want {want:?}",
        out[0]
    );
}

#[test]
fn fused_swiglu_matches_unfused_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // FusedSwiGLU([..., 2N]) → up * silu(gate) where up = first N, gate = next N.
    let mut g = Graph::new("swg");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.add_node(
        Op::FusedSwiGLU { cast_to: None },
        vec![x],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xv: Vec<f32> = vec![
        1.0, 2.0, 0.5, 1.5, // up=[1,2], gate=[0.5,1.5]
        3.0, 4.0, 1.0, 2.0, // up=[3,4], gate=[1.0,2.0]
    ];
    let r = exe.run(&[("x", &xv)]);
    let silu = |z: f32| z / (1.0 + (-z).exp());
    let want = vec![
        1.0 * silu(0.5),
        2.0 * silu(1.5),
        3.0 * silu(1.0),
        4.0 * silu(2.0),
    ];
    assert!(
        close(&r[0], &want, 1e-4),
        "FusedSwiGLU mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn lora_matmul_matches_unfused_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // LoraMatMul: out = x@W + scale * (x@A) @ B.
    let mut g = Graph::new("lora");
    let m = 2;
    let k = 3;
    let n = 2;
    let r = 2;
    let scale = 0.5f32;
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let a = g.param("a", Shape::new(&[k, r], DType::F32));
    let b = g.param("b", Shape::new(&[r, n], DType::F32));
    let y = g.lora_matmul(x, w, a, b, scale, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let wv = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let av = vec![0.1, 0.0, 0.0, 0.1, 0.1, 0.1];
    let bv = vec![1.0, 0.0, 0.0, 1.0];
    exe.set_param("w", &wv);
    exe.set_param("a", &av);
    exe.set_param("b", &bv);
    let r_out = exe.run(&[("x", &xv)]);
    // Reference: x@W + scale * (x@A) @ B
    let xw = matmul_ref(&xv, &wv, m, k, n);
    let xa = matmul_ref(&xv, &av, m, k, r);
    let xab = matmul_ref(&xa, &bv, m, r, n);
    let want: Vec<f32> = xw.iter().zip(&xab).map(|(&a, &b)| a + scale * b).collect();
    assert!(
        close(&r_out[0], &want, 1e-4),
        "LoRA mismatch: got {:?} want {want:?}",
        r_out[0]
    );
}

#[test]
fn gelu_finite_for_large_inputs() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Regression: naive tanh on the GELU inner expansion overflows
    // f32 exp past x ≈ 88, yielding NaN from inf/inf. We clamp.
    let mut g = Graph::new("gelu-large");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let y = g.activation(Activation::Gelu, x, Shape::new(&[6], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xs = vec![-25.0_f32, -17.0, -5.0, 5.0, 17.0, 25.0];
    let r = exe.run(&[("x", &xs)]);
    let nans = r[0].iter().filter(|v| v.is_nan()).count();
    let infs = r[0].iter().filter(|v| v.is_infinite()).count();
    assert_eq!(nans, 0, "GELU produced NaN: r={:?}", r[0]);
    assert_eq!(infs, 0, "GELU produced Inf: r={:?}", r[0]);
    // Asymptotic checks: gelu(very_negative) ≈ 0, gelu(very_positive) ≈ x.
    assert!(
        r[0][0].abs() < 1e-3,
        "gelu(-25) should ≈ 0, got {}",
        r[0][0]
    );
    assert!(
        (r[0][5] - 25.0).abs() < 1e-2,
        "gelu(25) should ≈ 25, got {}",
        r[0][5]
    );
}

#[test]
fn attention_rank3_with_2d_mask_produces_finite_output() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // BERT-flavored: rank-3 [B, S, H*D] inputs, [B, S] padding mask.
    // Tests the unfuse rank-3 attention promotion + mask broadcast pre-pass.
    let mut g = Graph::new("attn-rank3");
    let b = 1;
    let s = 3;
    let h = 2;
    let d = 4;
    let inner = h * d;
    let q = g.input("q", Shape::new(&[b, s, inner], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, inner], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, inner], DType::F32));
    let m = g.input("m", Shape::new(&[b, s], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, m],
        Shape::new(&[b, s, inner], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let qv: Vec<f32> = (0..b * s * inner).map(|i| (i as f32) * 0.01).collect();
    let kv = qv.clone();
    let vv = qv.clone();
    let mv = vec![0.0; b * s]; // additive zero — no masking
    let r = exe.run(&[("q", &qv), ("k", &kv), ("v", &vv), ("m", &mv)]);
    let nans = r[0].iter().filter(|v| v.is_nan()).count();
    let infs = r[0].iter().filter(|v| v.is_infinite()).count();
    assert_eq!(
        nans,
        0,
        "rank-3 attention produced {nans} NaN values; \
        first 8 = {:?}",
        &r[0][..8.min(r[0].len())]
    );
    assert_eq!(infs, 0, "rank-3 attention produced {infs} Inf values");
}

#[test]
fn fused_attention_block_end_to_end() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // FusedAttentionBlock through the unfusion pass exercises the full
    // chain: 3D@2D matmul, narrow×3, reshape+transpose to BHSD, attention,
    // transpose+reshape, output 3D@2D matmul. Identity weights so the
    // path is easy to reason about without full reference math.
    let mut g = Graph::new("fab");
    let b = 1;
    let s = 2;
    let h = 2;
    let d = 2;
    let inner = h * d;
    let hidden_shape = Shape::new(&[b, s, inner], DType::F32);
    let qkv_w_shape = Shape::new(&[inner, 3 * inner], DType::F32);
    let out_w_shape = Shape::new(&[inner, inner], DType::F32);
    // Attention kernel expects [B, H, S_q, S_k]; the FAB unfuse passes the
    // mask straight through, so we shape it that way at the input boundary.
    let mask_shape = Shape::new(&[b, h, s, s], DType::F32);

    let hidden = g.input("h", hidden_shape.clone());
    let qkv_w = g.param("qkv_w", qkv_w_shape);
    let out_w = g.param("out_w", out_w_shape);
    let mask = g.input("mask", mask_shape);
    let y = g.add_node(
        Op::FusedAttentionBlock {
            num_heads: h,
            head_dim: d,
            has_bias: false,
            has_rope: false,
        },
        vec![hidden, qkv_w, out_w, mask],
        hidden_shape,
    );
    g.set_outputs(vec![y]);

    let mut exe = WgpuExecutable::compile(g);
    // QKV weight: identity for Q (cols 0..4), zero for K (4..8), zero for V (8..12).
    // With K=0 and V=0, attention output is 0 (softmax over zeros = uniform, but V is zero).
    // out_w = identity → output equals attention output (0).
    let mut qkv_w = vec![0f32; inner * 3 * inner];
    for i in 0..inner {
        qkv_w[i * 3 * inner + i] = 1.0;
    }
    // V columns are 8..12, set them to identity so attention output = mean(V_rows) which equals
    // mean of hidden rows for each batch (since softmax of zeros is uniform).
    for i in 0..inner {
        qkv_w[i * 3 * inner + 2 * inner + i] = 1.0;
    }
    let mut out_w = vec![0f32; inner * inner];
    for i in 0..inner {
        out_w[i * inner + i] = 1.0;
    }
    exe.set_param("qkv_w", &qkv_w);
    exe.set_param("out_w", &out_w);

    // hidden = [[1,2,3,4], [5,6,7,8]]; mask all zeros (no masking).
    let hidden_v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // [B=1, H=2, S=2, S=2] all-zero (no masking).
    let mask_v = vec![0.0; b * h * s * s];
    let r = exe.run(&[("h", &hidden_v), ("mask", &mask_v)]);

    // K is all zeros → softmax(QK^T / sqrt(d)) = uniform 1/S.
    // V is identity-projected hidden.
    // Output for each token = mean of V rows = mean of hidden rows.
    // Mean per channel: ((1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2) = (3, 4, 5, 6).
    let want = vec![3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0];
    assert!(
        close(&r[0], &want, 1e-3),
        "FAB mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn selective_scan_minimum_config_matches_cpu_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // B=1, S=2, H=2, N=2. Hand-checkable values.
    let mut g = Graph::new("ssm");
    let b = 1;
    let s = 2;
    let h = 2;
    let n = 2;
    let x = g.input("x", Shape::new(&[b, s, h], DType::F32));
    let dt = g.input("dt", Shape::new(&[b, s, h], DType::F32));
    let a = g.param("a", Shape::new(&[h, n], DType::F32));
    let bb = g.input("b", Shape::new(&[b, s, n], DType::F32));
    let cc = g.input("c", Shape::new(&[b, s, n], DType::F32));
    let y = g.add_node(
        Op::SelectiveScan { state_size: n },
        vec![x, dt, a, bb, cc],
        Shape::new(&[b, s, h], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // A = -1 everywhere → exp(d * -1) = exp(-d) decay.
    let av = vec![-1.0; h * n];
    exe.set_param("a", &av);
    let xv = vec![1.0, 1.0, 1.0, 1.0]; // [B, S, H]
    let dtv = vec![1.0, 1.0, 1.0, 1.0];
    let bv = vec![1.0, 0.0, 0.0, 1.0]; // [B, S, N]
    let cv = vec![1.0, 1.0, 1.0, 1.0];
    let r = exe.run(&[("x", &xv), ("dt", &dtv), ("b", &bv), ("c", &cv)]);

    // Reference: same loop as the CPU thunk.
    let mut want = vec![0f32; b * s * h];
    let mut state = vec![0f32; h * n];
    for bi in 0..b {
        for v in state.iter_mut() {
            *v = 0.0;
        }
        for si in 0..s {
            for ci in 0..h {
                let d = dtv[bi * s * h + si * h + ci];
                let xv_ = xv[bi * s * h + si * h + ci];
                let mut acc = 0.0;
                for ni in 0..n {
                    let da = (d * av[ci * n + ni]).exp();
                    state[ci * n + ni] =
                        da * state[ci * n + ni] + d * bv[bi * s * n + si * n + ni] * xv_;
                    acc += cv[bi * s * n + si * n + ni] * state[ci * n + ni];
                }
                want[bi * s * h + si * h + ci] = acc;
            }
        }
    }
    assert!(
        close(&r[0], &want, 1e-4),
        "SelectiveScan mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dequant_matmul_int8_symmetric_matches_dequant_then_matmul() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::QuantScheme;
    // x[2, 4] @ w_q[4, 3] (int8, block_size=2, symmetric) → [2, 3].
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let block_size: u32 = 2;
    let n_blocks = (k as u32).div_ceil(block_size);
    let mut g = Graph::new("dq");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let wq = g.param("wq", Shape::new(&[k, n], DType::I8));
    let sc = g.param("sc", Shape::new(&[n_blocks as usize, n], DType::F32));
    let zp = g.param("zp", Shape::new(&[n_blocks as usize, n], DType::F32));
    let y = g.dequant_matmul(
        x,
        wq,
        sc,
        zp,
        QuantScheme::Int8Block { block_size },
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);

    // Hand-pick weights and scales.
    let w_i8: Vec<i8> = vec![
        1, 2, 3, // k=0
        -1, 0, 4, // k=1
        5, -2, 1, // k=2
        2, 3, -1, // k=3
    ];
    let w_bytes: Vec<u8> = w_i8.iter().map(|&b| b as u8).collect();
    exe.set_param_bytes("wq", &w_bytes);
    let scales = vec![
        0.1, 0.2, 0.3, // block 0 (k=0,1)
        0.4, 0.5, 0.6, // block 1 (k=2,3)
    ];
    let zps = vec![0.0; (n_blocks as usize) * n]; // symmetric — zp ignored
    exe.set_param("sc", &scales);
    exe.set_param("zp", &zps);

    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = exe.run(&[("x", &xv)]);

    // Reference: dequantize first, then plain matmul.
    let mut w_dq = vec![0f32; k * n];
    for ki in 0..k {
        let block = ki / (block_size as usize);
        for ni in 0..n {
            let q = w_i8[ki * n + ni] as f32;
            w_dq[ki * n + ni] = q * scales[block * n + ni];
        }
    }
    let want = matmul_ref(&xv, &w_dq, m, k, n);
    assert!(
        close(&r[0], &want, 1e-3),
        "DequantMatMul mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dequant_matmul_int4_symmetric_matches_dequant_then_matmul() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::QuantScheme;
    // x[2, 4] @ w_q[4, 4] (int4, block_size=2, symmetric, signed [-8,7]) → [2, 4].
    let m = 2usize;
    let k = 4usize;
    let n = 4usize;
    let block_size: u32 = 2;
    let n_blocks = (k as u32).div_ceil(block_size);
    let mut g = Graph::new("dq4");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let wq = g.param("wq", Shape::new(&[k, n], DType::I8));
    let sc = g.param("sc", Shape::new(&[n_blocks as usize, n], DType::F32));
    let zp = g.param("zp", Shape::new(&[n_blocks as usize, n], DType::F32));
    let y = g.dequant_matmul(
        x,
        wq,
        sc,
        zp,
        QuantScheme::Int4Block { block_size },
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);

    // Int4 weights: 16 elements, each in [-8, 7]. Pack two per byte (low first).
    let w_i4: Vec<i8> = vec![
        1, 2, -3, 4, // k=0
        -1, 0, 5, -6, // k=1
        3, -2, 1, 7, // k=2
        -4, 6, -5, 2, // k=3
    ];
    let mut packed = vec![0u8; (k * n) / 2];
    for (i, chunk) in w_i4.chunks(2).enumerate() {
        let lo = (chunk[0] as i32 & 0xf) as u8;
        let hi = (chunk[1] as i32 & 0xf) as u8;
        packed[i] = lo | (hi << 4);
    }
    exe.set_param_bytes("wq", &packed);

    let scales = vec![
        0.1, 0.2, 0.3, 0.4, // block 0
        0.5, 0.6, 0.7, 0.8, // block 1
    ];
    let zps = vec![0.0; (n_blocks as usize) * n];
    exe.set_param("sc", &scales);
    exe.set_param("zp", &zps);

    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = exe.run(&[("x", &xv)]);

    // Reference.
    let mut w_dq = vec![0f32; k * n];
    for ki in 0..k {
        let block = ki / (block_size as usize);
        for ni in 0..n {
            let q = w_i4[ki * n + ni] as f32;
            w_dq[ki * n + ni] = q * scales[block * n + ni];
        }
    }
    let want = matmul_ref(&xv, &w_dq, m, k, n);
    assert!(
        close(&r[0], &want, 1e-3),
        "DequantMatMul Int4 mismatch: got {:?} want {want:?}",
        r[0]
    );
}

/// OCP E4M3 reference decoder (1 sign + 4 exp + 3 mantissa, exp bias 7,
/// no infinity, exp=15 + mant=7 reserved for NaN).
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0xf;
    let mant = byte & 0x7;
    let v = if exp == 0 {
        (mant as f32 / 8.0) * (-6f32).exp2()
    } else if exp == 15 && mant == 7 {
        0.0
    } else {
        let m = 1.0 + mant as f32 / 8.0;
        m * ((exp as i32 - 7) as f32).exp2()
    };
    if sign != 0 { -v } else { v }
}

/// OCP E5M2 reference (1 sign + 5 exp + 2 mantissa, exp bias 15, has inf/NaN).
fn e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 2) & 0x1f;
    let mant = byte & 0x3;
    let v = if exp == 0 {
        (mant as f32 / 4.0) * (-14f32).exp2()
    } else if exp == 31 {
        0.0
    } else {
        let m = 1.0 + mant as f32 / 4.0;
        m * ((exp as i32 - 15) as f32).exp2()
    };
    if sign != 0 { -v } else { v }
}

#[test]
fn dequant_matmul_fp8_e4m3_matches_decode_then_matmul() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::QuantScheme;
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let mut g = Graph::new("dq-e4m3");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let wq = g.param("wq", Shape::new(&[k, n], DType::U8));
    // FP8 has no scale/zp; the IR contract still requires the slots.
    let sc = g.param("sc", Shape::new(&[1, n], DType::F32));
    let zp = g.param("zp", Shape::new(&[1, n], DType::F32));
    let y = g.dequant_matmul(
        x,
        wq,
        sc,
        zp,
        QuantScheme::Fp8E4m3,
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // 12 fp8 weights covering a mix of normals, sub-normals, and signs.
    let w_bytes: Vec<u8> = vec![
        0x38, 0x40, 0x48, // +1.0, +2.0, +4.0
        0x01, 0xC0, 0x44, // tiny subnormal, -2.0, +3.0
        0xB8, 0x00, 0x70, // -1.0,  0,  +384 (large)
        0x30, 0x21, 0x68, // +0.5, mid subnormal, +112
    ];
    exe.set_param_bytes("wq", &w_bytes);
    exe.set_param("sc", &vec![0.0; n]);
    exe.set_param("zp", &vec![0.0; n]);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = exe.run(&[("x", &xv)]);
    // Reference: decode FP8 → matmul.
    let w_dq: Vec<f32> = w_bytes.iter().map(|&b| e4m3_to_f32(b)).collect();
    let want = matmul_ref(&xv, &w_dq, m, k, n);
    assert!(
        close(&r[0], &want, 1e-3),
        "FP8 E4M3 mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dequant_matmul_fp8_e5m2_matches_decode_then_matmul() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::QuantScheme;
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;
    let mut g = Graph::new("dq-e5m2");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let wq = g.param("wq", Shape::new(&[k, n], DType::U8));
    let sc = g.param("sc", Shape::new(&[1, n], DType::F32));
    let zp = g.param("zp", Shape::new(&[1, n], DType::F32));
    let y = g.dequant_matmul(
        x,
        wq,
        sc,
        zp,
        QuantScheme::Fp8E5m2,
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // E5M2 byte patterns: bias=15, mantissa=2 bits.
    //   0x3C = 0_01111_00 = 1.0
    //   0x40 = 0_10000_00 = 2.0
    //   0x44 = 0_10001_00 = 4.0
    //   0xBC = 1_01111_00 = -1.0
    //   0x00 = 0
    //   0x4C = 0_10011_00 = 16.0
    let w_bytes: Vec<u8> = vec![
        0x3C, 0x40, 0x44, 0xBC, 0x00, 0x4C, 0x3C, 0xC4, 0x3C, 0x40, 0x44, 0x40,
    ];
    exe.set_param_bytes("wq", &w_bytes);
    exe.set_param("sc", &vec![0.0; n]);
    exe.set_param("zp", &vec![0.0; n]);
    let xv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r = exe.run(&[("x", &xv)]);
    let w_dq: Vec<f32> = w_bytes.iter().map(|&b| e5m2_to_f32(b)).collect();
    let want = matmul_ref(&xv, &w_dq, m, k, n);
    assert!(
        close(&r[0], &want, 1e-3),
        "FP8 E5M2 mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dynamic_shape_auto_infers_at_run_time() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::shape::Dim;
    // Same graph as compile_with_bindings, but compile() infers
    // Dim::Dynamic(0) → 3 from the input data length on first run.
    let mut g = Graph::new("dyn-auto");
    let dyn_shape = rlx_ir::Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32);
    let x = g.input("x", dyn_shape);
    let two = g.add_node(
        Op::Constant {
            data: 2.0_f32.to_le_bytes().to_vec(),
        },
        vec![],
        rlx_ir::Shape::from_dims(&[Dim::Static(1), Dim::Static(1)], DType::F32),
    );
    let two_b = g.add_node(
        Op::Expand {
            target_shape: vec![3, 4],
        },
        vec![two],
        Shape::new(&[3, 4], DType::F32),
    );
    let dyn_out = rlx_ir::Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32);
    let y = g.binary(BinaryOp::Mul, x, two_b, dyn_out);
    g.set_outputs(vec![y]);

    let mut exe = WgpuExecutable::compile(g); // no explicit binding!
    let xv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xv)]);
    let want: Vec<f32> = xv.iter().map(|v| v * 2.0).collect();
    assert!(
        close(&r[0], &want, 1e-4),
        "auto-infer DimBinding mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dynamic_shape_resolves_via_compile_with_bindings() {
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::shape::{Dim, DimBinding};
    // Graph with shape [Dynamic(0), 4]: a Param with that dynamic dim,
    // multiplied by 2.
    let mut g = Graph::new("dyn");
    let dyn_shape = rlx_ir::Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32);
    let x = g.input("x", dyn_shape.clone());
    let two = g.add_node(
        Op::Constant {
            data: 2.0_f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1, 1], DType::F32),
    );
    let two_b = g.add_node(
        Op::Expand {
            target_shape: vec![3, 4],
        },
        vec![two],
        Shape::new(&[3, 4], DType::F32),
    );
    let y = g.binary(BinaryOp::Mul, x, two_b, Shape::new(&[3, 4], DType::F32));
    g.set_outputs(vec![y]);

    // Bind the dynamic symbol 0 → 3.
    let mut bindings = DimBinding::new();
    bindings.set(0, 3);
    let mut exe = WgpuExecutable::compile_with_bindings(g, &bindings);
    let xv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let r = exe.run(&[("x", &xv)]);
    let want: Vec<f32> = xv.iter().map(|v| v * 2.0).collect();
    assert!(
        close(&r[0], &want, 1e-4),
        "DimBinding mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn op_if_picks_branch_per_predicate() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Build two trivial branches: then = x + 1, else = x * 2.
    // With pred = bool, expected output is per-element select.
    let then_branch = {
        let mut g = Graph::new("then");
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: 1.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let cb = g.add_node(
            Op::Expand {
                target_shape: vec![3],
            },
            vec![c],
            Shape::new(&[3], DType::F32),
        );
        let y = g.binary(BinaryOp::Add, x, cb, Shape::new(&[3], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let else_branch = {
        let mut g = Graph::new("else");
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: 2.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let cb = g.add_node(
            Op::Expand {
                target_shape: vec![3],
            },
            vec![c],
            Shape::new(&[3], DType::F32),
        );
        let y = g.binary(BinaryOp::Mul, x, cb, Shape::new(&[3], DType::F32));
        g.set_outputs(vec![y]);
        g
    };

    let mut g = Graph::new("ifx");
    let pred = g.input("pred", Shape::new(&[3], DType::Bool));
    let xv = g.input("x", Shape::new(&[3], DType::F32));
    let y = g.add_node(
        Op::If {
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        vec![pred, xv],
        Shape::new(&[3], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xs = vec![1.0f32, 2.0, 3.0];
    let pv = vec![1.0f32, 0.0, 1.0]; // bool encoded as f32 in our arena
    let r = exe.run(&[("x", &xs), ("pred", &pv)]);
    let want = vec![
        1.0 + 1.0, // pred=true  → x + 1
        2.0 * 2.0, // pred=false → x * 2
        3.0 + 1.0,
    ];
    assert!(
        close(&r[0], &want, 1e-4),
        "Op::If mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn op_while_unrolls_until_cond_false() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // body: x = x * 2.   cond: x[0] < 16.
    // Starting at x=1: 1 → 2 → 4 → 8 → 16 (cond false, freeze) → 16 → ...
    // With max_iterations = 6, we expect 16 (cond goes false at the 5th iter).
    let body = {
        let mut g = Graph::new("body");
        let x = g.input("x", Shape::new(&[1], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: 2.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let y = g.binary(BinaryOp::Mul, x, c, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let cond = {
        let mut g = Graph::new("cond");
        let x = g.input("x", Shape::new(&[1], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: 16.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let y = g.add_node(
            Op::Compare(CmpOp::Lt),
            vec![x, c],
            Shape::new(&[1], DType::Bool),
        );
        g.set_outputs(vec![y]);
        g
    };

    let mut g = Graph::new("loopy");
    let x = g.input("x", Shape::new(&[1], DType::F32));
    let y = g.add_node(
        Op::While {
            cond: Box::new(cond),
            body: Box::new(body),
            max_iterations: Some(6),
        },
        vec![x],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let r = exe.run(&[("x", &[1.0f32])]);
    assert!(
        close(&r[0], &[16.0], 1e-4),
        "Op::While mismatch: got {:?}, expected 16",
        r[0]
    );
}

#[test]
fn dot_general_batched_matches_per_batch_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Batched DotGeneral: lhs[B,M,K] · rhs[B,K,N] → [B,M,N].
    // lhs_batch=[0], lhs_contracting=[2], rhs_batch=[0], rhs_contracting=[1].
    let mut g = Graph::new("dg-batched");
    let l = g.input("l", Shape::new(&[2, 2, 3], DType::F32));
    let r = g.input("r", Shape::new(&[2, 3, 2], DType::F32));
    let y = g.add_node(
        Op::DotGeneral {
            lhs_contracting: vec![2],
            rhs_contracting: vec![1],
            lhs_batch: vec![0],
            rhs_batch: vec![0],
        },
        vec![l, r],
        Shape::new(&[2, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let lv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let rv: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let r = exe.run(&[("l", &lv), ("r", &rv)]);
    let mut want = vec![0f32; 8];
    for bi in 0..2 {
        let l_slice = &lv[bi * 6..(bi + 1) * 6];
        let r_slice = &rv[bi * 6..(bi + 1) * 6];
        let y = matmul_ref(l_slice, r_slice, 2, 3, 2);
        want[bi * 4..(bi + 1) * 4].copy_from_slice(&y);
    }
    assert!(
        close(&r[0], &want, 1e-4),
        "batched DotGeneral mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dot_general_lhs_transposed_matches_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // lhs[K, M] · rhs[K, N] → [M, N]. Contracting on axis 0 of both inputs.
    let mut g = Graph::new("dg-lhs-t");
    let m = 2;
    let k = 3;
    let n = 2;
    let l = g.input("l", Shape::new(&[k, m], DType::F32));
    let r = g.input("r", Shape::new(&[k, n], DType::F32));
    let y = g.add_node(
        Op::DotGeneral {
            lhs_contracting: vec![0],
            rhs_contracting: vec![0],
            lhs_batch: vec![],
            rhs_batch: vec![],
        },
        vec![l, r],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let lv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [K=3, M=2]
    let rv = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [K=3, N=2]
    let out = exe.run(&[("l", &lv), ("r", &rv)]);
    // Reference: lhs.T @ rhs. lhs.T is [M=2, K=3] = [[1,3,5],[2,4,6]].
    // Out[0] = [1,3,5] · [(1,0),(0,1),(1,1)] = (1+0+5, 0+3+5) = (6, 8).
    // Out[1] = [2,4,6] · same = (2+0+6, 0+4+6) = (8, 10).
    let want = vec![6.0, 8.0, 8.0, 10.0];
    assert!(
        close(&out[0], &want, 1e-4),
        "DotGeneral lhs.T mismatch: got {:?} want {want:?}",
        out[0]
    );
}

#[test]
fn sample_top_k_one_collapses_to_argmax() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("samp-k1");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.add_node(
        Op::Sample {
            top_k: 1,
            top_p: 1.0,
            temperature: 1.0,
            seed: 42,
        },
        vec![x],
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xs = vec![
        1.0, 5.0, 2.0, 3.0, // argmax = 1
        9.0, 0.0, 0.0, 0.0,
    ]; // argmax = 0
    let r = exe.run(&[("x", &xs)]);
    assert_eq!(r[0], vec![1.0, 0.0]);
}

fn threefry2x32_20_ref(c_in: [u32; 2], k_in: [u32; 2]) -> [u32; 2] {
    fn rotl32(x: u32, n: u32) -> u32 {
        x.rotate_left(n)
    }
    let ks0 = k_in[0];
    let ks1 = k_in[1];
    let ks2 = ks0 ^ ks1 ^ 0x1BD11BDA;
    let mut x0 = c_in[0].wrapping_add(ks0);
    let mut x1 = c_in[1].wrapping_add(ks1);
    let r2x32: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    for round in 0..20 {
        x0 = x0.wrapping_add(x1);
        x1 = rotl32(x1, r2x32[round % 8]);
        x1 ^= x0;
        if (round + 1) % 4 == 0 {
            let inj = (round / 4 + 1) as u32;
            // ks rotation: inj=1 → (ks1, ks2); inj=2 → (ks2, ks0); inj=3 → (ks0, ks1); ...
            let ksx = match inj % 3 {
                0 => ks0,
                1 => ks1,
                _ => ks2,
            };
            let ksy = match (inj + 1) % 3 {
                0 => ks0,
                1 => ks1,
                _ => ks2,
            };
            x0 = x0.wrapping_add(ksx);
            x1 = x1.wrapping_add(ksy);
            x1 = x1.wrapping_add(inj);
        }
    }
    [x0, x1]
}

#[test]
fn threefry_reference_distributes_uniformly() {
    let mut buckets = [0u32; 8];
    for row in 0..64u32 {
        let r = threefry2x32_20_ref([row, 0], [0xC0FFEE, 0]);
        let u = r[0] as f64 / 4294967296.0;
        let bucket = (u * 8.0) as usize;
        buckets[bucket.min(7)] += 1;
    }
    let hit = buckets.iter().filter(|&&n| n > 0).count();
    assert!(
        hit >= 6,
        "Reference Threefry only hit {hit}/8 buckets — buckets={buckets:?}"
    );
}

#[test]
fn sample_threefry_seed_is_deterministic_and_distributes() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Force the Sample kernel path (not the greedy argmax fast-path)
    // by passing top_p < 1 — that's what gates the Threefry dispatch.
    // Uniform logits + temperature=1 + top_p<1 still includes every
    // token in the nucleus so the multinomial draw covers all tokens.
    let n = 8;
    let batch = 64;
    let mut g = Graph::new("samp-threefry");
    let x = g.input("x", Shape::new(&[batch, n], DType::F32));
    let y = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 0.999,
            temperature: 1.0,
            seed: 0xC0FFEE,
        },
        vec![x],
        Shape::new(&[batch], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xs: Vec<f32> = (0..batch * n).map(|_| 0.0).collect();
    let r1 = exe.run(&[("x", &xs)]);
    let r2 = exe.run(&[("x", &xs)]);
    assert_eq!(
        r1[0], r2[0],
        "Threefry should be deterministic for same seed"
    );
    let mut hit = vec![0u32; n];
    for &v in &r1[0] {
        hit[v as usize] += 1;
    }
    let covered = hit.iter().filter(|c| **c > 0).count();
    assert!(
        covered >= 6,
        "Threefry-driven Sample only hit {covered}/{n} tokens; \
         per-token counts={hit:?}"
    );
}

#[test]
fn sample_gumbel_max_concentrates_on_dominant_logit() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Logit profile heavily favors token 3. Gumbel-max draws should
    // pick token 3 the vast majority of the time across 64 batch rows.
    let n = 8;
    let batch = 64;
    let mut g = Graph::new("samp-gumbel");
    let x = g.input("x", Shape::new(&[batch, n], DType::F32));
    let y = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 0.999,
            temperature: 1.0,
            seed: 99,
        },
        vec![x],
        Shape::new(&[batch], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // Token 3 dominates by 5 nats over the next-best.
    let mut xs = vec![0.0f32; batch * n];
    for b in 0..batch {
        xs[b * n + 3] = 5.0;
    }
    let r = exe.run(&[("x", &xs)]);
    let three_picks = r[0].iter().filter(|&&v| v == 3.0).count();
    assert!(
        three_picks >= batch * 90 / 100,
        "Gumbel-max should land on the dominant token most of the time; \
         only {three_picks}/{batch} hit token 3, picks={:?}",
        r[0]
    );
}

#[test]
fn sample_top_p_zero_collapses_to_argmax() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // top_p just barely above zero forces selection of the single largest
    // probability — which is argmax of the logits.
    let mut g = Graph::new("samp-p0");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 0.001,
            temperature: 1.0,
            seed: 7,
        },
        vec![x],
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xs = vec![
        10.0, 0.0, 0.0, 0.0, // top-1 = 0
        0.0, 0.0, 0.0, 10.0,
    ]; // top-1 = 3
    let r = exe.run(&[("x", &xs)]);
    assert_eq!(r[0], vec![0.0, 3.0]);
}

#[test]
fn attention_causal_mask_zeros_future_tokens() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // [B=1, H=1, S=2, D=2]. Causal: token 0 attends only to token 0;
    // token 1 attends to tokens 0 and 1. With Q/K = identity-row and V = [[1,2],[3,4]]:
    // Score(qi=0, ki=0) = 1*1 + 0*0 = 1, no other contributions allowed.
    // softmax([1]) = [1]; out[0] = V[0] = [1,2].
    let mut g = Graph::new("attn-causal");
    let q = g.input("q", Shape::new(&[1, 1, 2, 2], DType::F32));
    let k = g.input("k", Shape::new(&[1, 1, 2, 2], DType::F32));
    let v = g.input("v", Shape::new(&[1, 1, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: 2,
            mask_kind: MaskKind::Causal,
        },
        vec![q, k, v],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let qv = vec![1.0, 0.0, 0.0, 1.0];
    let kv = vec![1.0, 0.0, 0.0, 1.0];
    let vv = vec![1.0, 2.0, 3.0, 4.0];
    let r = exe.run(&[("q", &qv), ("k", &kv), ("v", &vv)]);
    // Token 0: attends only to 0, so out[0] = V[0] = [1,2].
    // Token 1: attends to 0 and 1. Scores: Q1·K0/sqrt(2)=0, Q1·K1/sqrt(2)=1/sqrt(2).
    //   softmax(0, 0.7071) = (~0.33, ~0.67). out[1] ≈ 0.33*[1,2] + 0.67*[3,4].
    let s = 1.0 / 2.0_f32.sqrt();
    let e0 = (0.0_f32 - s).exp();
    let e1 = 1.0_f32;
    let z = e0 + e1;
    let w0 = e0 / z;
    let w1 = e1 / z;
    let want = vec![1.0, 2.0, w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0];
    assert!(
        close(&r[0], &want, 1e-3),
        "Causal attention mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn attention_sliding_window_limits_lookback() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Window=0 means qi attends only to ki==qi (a strictly diagonal mask).
    // With identity Q/K and V=[[1,2],[3,4]], output should equal V exactly.
    let mut g = Graph::new("attn-sw");
    let q = g.input("q", Shape::new(&[1, 1, 2, 2], DType::F32));
    let k = g.input("k", Shape::new(&[1, 1, 2, 2], DType::F32));
    let v = g.input("v", Shape::new(&[1, 1, 2, 2], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: 2,
            mask_kind: MaskKind::SlidingWindow(0),
        },
        vec![q, k, v],
        Shape::new(&[1, 1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let qv = vec![1.0, 0.0, 0.0, 1.0];
    let kv = vec![1.0, 0.0, 0.0, 1.0];
    let vv = vec![1.0, 2.0, 3.0, 4.0];
    let r = exe.run(&[("q", &qv), ("k", &kv), ("v", &vv)]);
    assert!(
        close(&r[0], &vv, 1e-3),
        "SlidingWindow attention mismatch: got {:?} want {vv:?}",
        r[0]
    );
}

#[test]
fn grouped_matmul_routes_per_token_to_expert() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // 2 tokens, 2 experts. K=2, N=2.
    // Token 0 → expert 0 (weight = identity);   token 1 → expert 1 (weight = scale*identity).
    let mut g = Graph::new("gmm");
    let m = 2;
    let k = 2;
    let n = 2;
    let ne = 2;
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[ne, k, n], DType::F32));
    let idx = g.input("idx", Shape::new(&[m], DType::F32));
    let y = g.add_node(
        Op::GroupedMatMul,
        vec![x, w, idx],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    // Expert 0: identity. Expert 1: 2x identity.
    let wv = vec![
        1.0, 0.0, 0.0, 1.0, // expert 0
        2.0, 0.0, 0.0, 2.0, // expert 1
    ];
    exe.set_param("w", &wv);
    let xv = vec![3.0, 4.0, 5.0, 6.0];
    let idxv = vec![0.0, 1.0];
    let r = exe.run(&[("x", &xv), ("idx", &idxv)]);
    assert!(
        close(
            &r[0],
            &[
                3.0, 4.0, // token 0 @ expert 0
                10.0, 12.0
            ], // token 1 @ expert 1 (×2)
            1e-4
        ),
        "GroupedMatMul mismatch: got {:?}",
        r[0]
    );
}

#[test]
fn topk_picks_largest_three_indices() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let mut g = Graph::new("topk");
    let x = g.input("x", Shape::new(&[2, 5], DType::F32));
    let y = g.add_node(Op::TopK { k: 3 }, vec![x], Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xv = vec![
        5.0, 1.0, 4.0, 2.0, 3.0, // top 3: indices 0, 2, 4
        0.5, 9.0, 0.1, 7.0, 8.0,
    ]; // top 3: indices 1, 4, 3
    let r = exe.run(&[("x", &xv)]);
    assert_eq!(r[0], vec![0.0, 2.0, 4.0, 1.0, 4.0, 3.0]);
}

#[test]
fn batched_matmul_3d_by_3d_matches_per_batch_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // [B=2, M=2, K=3] @ [B=2, K=3, N=2] → [B=2, M=2, N=2].
    // Different rhs per batch so this exercises the per-batch stride path.
    let mut g = Graph::new("bmm3");
    let l = g.input("l", Shape::new(&[2, 2, 3], DType::F32));
    let r = g.input("r", Shape::new(&[2, 3, 2], DType::F32));
    let y = g.matmul(l, r, Shape::new(&[2, 2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let lv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let rv: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // batch 0
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // batch 1 (different rhs)
    ];
    let r = exe.run(&[("l", &lv), ("r", &rv)]);
    // Reference per batch.
    let mut want = vec![0f32; 2 * 2 * 2];
    for bi in 0..2 {
        let l_slice = &lv[bi * 6..(bi + 1) * 6];
        let r_slice = &rv[bi * 6..(bi + 1) * 6];
        let y = matmul_ref(l_slice, r_slice, 2, 3, 2);
        want[bi * 4..(bi + 1) * 4].copy_from_slice(&y);
    }
    assert!(
        close(&r[0], &want, 1e-4),
        "batched 3D@3D mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn batched_matmul_3d_by_2d_matches_per_row_reference() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // [B=2, S=2, K=3] @ [K=3, N=2] → [B=2, S=2, N=2]
    let mut g = Graph::new("bmm");
    let x = g.input("x", Shape::new(&[2, 2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let xv: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let wv = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    exe.set_param("w", &wv);
    let r = exe.run(&[("x", &xv)]);
    // Reference: flatten [2,2,3] to [4,3], do 2D matmul, flatten back.
    let want = matmul_ref(&xv, &wv, 4, 3, 2);
    assert!(
        close(&r[0], &want, 1e-4),
        "batched matmul mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn scatter_add_accumulates_into_destination() {
    if !rlx_wgpu::is_available() {
        return;
    }
    // Output: [3 rows, 2 trailing]. Updates: 4 rows × 2 trailing.
    // Indices route updates: row 0 → dst[1], row 1 → dst[0],
    //                       row 2 → dst[1], row 3 → dst[2].
    // So dst[0] = upd[1], dst[1] = upd[0] + upd[2], dst[2] = upd[3].
    let mut g = Graph::new("sa");
    let upd = g.input("upd", Shape::new(&[4, 2], DType::F32));
    let idx = g.input("idx", Shape::new(&[4], DType::F32));
    let y = g.add_node(
        Op::ScatterAdd,
        vec![upd, idx],
        Shape::new(&[3, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    let updv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idxv = vec![1.0, 0.0, 1.0, 2.0];
    let r = exe.run(&[("upd", &updv), ("idx", &idxv)]);
    let want = vec![
        3.0,
        4.0, // dst[0] = upd[1]
        1.0 + 5.0,
        2.0 + 6.0, // dst[1] = upd[0] + upd[2]
        7.0,
        8.0, // dst[2] = upd[3]
    ];
    assert!(
        close(&r[0], &want, 1e-4),
        "ScatterAdd mismatch: got {:?} want {want:?}",
        r[0]
    );
}

/// wgpu BERT bisect helper. Produces (max_abs_diff, has_nan) where
/// `has_nan` is true if any wgpu output element is NaN. Compares
/// against a hand-computed reference passed in.
fn wgpu_run_and_check(
    g: Graph,
    inputs: &[(&str, &[f32])],
    params: &[(&str, &[f32])],
    want: &[f32],
) -> (f32, bool) {
    let mut exe = WgpuExecutable::compile(g);
    for (n, d) in params {
        exe.set_param(n, d);
    }
    let outs = exe.run(inputs);
    let got = outs.into_iter().next().unwrap_or_default();
    let has_nan = got.iter().any(|v| v.is_nan());
    let diff = if got.len() == want.len() && !has_nan {
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    } else {
        f32::INFINITY
    };
    (diff, has_nan)
}

#[test]
fn bisect_wgpu_gather_only() {
    // Just embedding lookup — simplest BERT op. Tests if `gather`
    // alone produces NaN.
    if !rlx_wgpu::is_available() {
        return;
    }
    let f = DType::F32;
    let mut g = Graph::new("gather_only");
    let ids = g.input("ids", Shape::new(&[1, 3], f));
    let table = g.param("emb", Shape::new(&[8, 4], f));
    let out = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, ids],
        Shape::new(&[1, 3, 4], f),
    );
    g.set_outputs(vec![out]);

    let ids_v = vec![0.0f32, 2.0, 5.0];
    let table_v: Vec<f32> = (0..32).map(|i| i as f32).collect();
    // Expected: rows 0, 2, 5 of table → [0..4, 8..12, 20..24]
    let want: Vec<f32> = vec![
        0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0,
    ];
    let (diff, has_nan) = wgpu_run_and_check(g, &[("ids", &ids_v)], &[("emb", &table_v)], &want);
    eprintln!("[bisect:gather] diff={diff:e} has_nan={has_nan}");
    assert!(!has_nan, "gather produced NaN");
    assert!(diff < 1e-5, "gather diff {diff:e}");
}

#[test]
fn bisect_wgpu_gather_then_layernorm() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let f = DType::F32;
    let mut g = Graph::new("gather_ln");
    let ids = g.input("ids", Shape::new(&[1, 3], f));
    let table = g.param("emb", Shape::new(&[8, 4], f));
    let gamma = g.param("gamma", Shape::new(&[4], f));
    let beta = g.param("beta", Shape::new(&[4], f));
    let g_out = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, ids],
        Shape::new(&[1, 3, 4], f),
    );
    let ln = g.add_node(
        Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![g_out, gamma, beta],
        Shape::new(&[1, 3, 4], f),
    );
    g.set_outputs(vec![ln]);

    let ids_v = vec![0.0f32, 2.0, 5.0];
    let table_v: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let gamma_v = vec![1.0f32; 4];
    let beta_v = vec![0.0f32; 4];

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("emb", &table_v);
    exe.set_param("gamma", &gamma_v);
    exe.set_param("beta", &beta_v);
    let out = exe.run(&[("ids", &ids_v)]).into_iter().next().unwrap();
    let has_nan = out.iter().any(|v| v.is_nan());
    eprintln!(
        "[bisect:gather+ln] first={:?} has_nan={has_nan}",
        &out[..4.min(out.len())]
    );
    assert!(!has_nan, "gather+layernorm produced NaN");
}

#[test]
fn bisect_wgpu_matmul_bias_narrow() {
    // matmul + bias + 3 narrows, the QKV pattern.
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::BinaryOp;
    let f = DType::F32;
    let h = 8;
    let mut g = Graph::new("mm_bias_narrow");
    let x = g.input("x", Shape::new(&[1, 3, h], f));
    let w = g.param("w", Shape::new(&[h, 3 * h], f));
    let b = g.param("b", Shape::new(&[3 * h], f));
    let mm = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[1, 3, 3 * h], f));
    let qkv = g.binary(BinaryOp::Add, mm, b, Shape::new(&[1, 3, 3 * h], f));
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[1, 3, h], f),
    );
    g.set_outputs(vec![q]);

    let x_v: Vec<f32> = (0..(3 * h)).map(|i| i as f32 * 0.1).collect();
    let w_v: Vec<f32> = (0..(h * 3 * h)).map(|i| (i % 7) as f32 * 0.05).collect();
    let b_v: Vec<f32> = (0..(3 * h)).map(|i| i as f32 * 0.01).collect();

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &w_v);
    exe.set_param("b", &b_v);
    let out = exe.run(&[("x", &x_v)]).into_iter().next().unwrap();
    let has_nan = out.iter().any(|v| v.is_nan());
    eprintln!(
        "[bisect:mm+bias+narrow] first={:?} has_nan={has_nan}",
        &out[..4.min(out.len())]
    );
    assert!(!has_nan, "mm+bias+narrow produced NaN");
}

#[test]
fn bisect_wgpu_attention_with_qkv_chain() {
    // Full attention chain: matmul(qkv) + bias + 3 narrows + attention.
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::BinaryOp;
    let f = DType::F32;
    let (b, s, nh, dh) = (1, 3, 2, 4);
    let h = nh * dh;

    let mut g = Graph::new("attn_chain");
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));
    let w = g.param("w", Shape::new(&[h, 3 * h], f));
    let bias = g.param("b", Shape::new(&[3 * h], f));
    let mm = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[b, s, 3 * h], f));
    let qkv = g.binary(BinaryOp::Add, mm, bias, Shape::new(&[b, s, 3 * h], f));
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let k = g.add_node(
        Op::Narrow {
            axis: 2,
            start: h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let v = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![attn]);

    let x_v: Vec<f32> = (0..(b * s * h)).map(|i| ((i % 11) as f32) * 0.1).collect();
    let w_v: Vec<f32> = (0..(h * 3 * h)).map(|i| ((i % 7) as f32) * 0.05).collect();
    let b_v: Vec<f32> = (0..(3 * h)).map(|i| (i as f32) * 0.01).collect();
    let mask_v = vec![1.0f32; b * s];

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("w", &w_v);
    exe.set_param("b", &b_v);
    let out = exe
        .run(&[("x", &x_v), ("mask", &mask_v)])
        .into_iter()
        .next()
        .unwrap();
    let has_nan = out.iter().any(|v| v.is_nan());
    eprintln!(
        "[bisect:attn-chain] len={} first={:?} has_nan={has_nan}",
        out.len(),
        &out[..4.min(out.len())]
    );
    assert!(!has_nan, "attention chain produced NaN");
}

#[test]
fn bisect_wgpu_fused_residual_ln() {
    if !rlx_wgpu::is_available() {
        return;
    }
    let f = DType::F32;
    let mut g = Graph::new("frln");
    let x = g.input("x", Shape::new(&[1, 3, 4], f));
    let res = g.input("res", Shape::new(&[1, 3, 4], f));
    let gamma = g.param("gamma", Shape::new(&[4], f));
    let beta = g.param("beta", Shape::new(&[4], f));
    let frln = g.add_node(
        Op::FusedResidualLN {
            has_bias: false,
            eps: 1e-5,
        },
        vec![x, res, gamma, beta],
        Shape::new(&[1, 3, 4], f),
    );
    g.set_outputs(vec![frln]);

    let x_v: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let res_v: Vec<f32> = (0..12).map(|i| i as f32 * 0.2).collect();
    let gamma_v = vec![1.0f32; 4];
    let beta_v = vec![0.0f32; 4];

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("gamma", &gamma_v);
    exe.set_param("beta", &beta_v);
    let out = exe
        .run(&[("x", &x_v), ("res", &res_v)])
        .into_iter()
        .next()
        .unwrap();
    let has_nan = out.iter().any(|v| v.is_nan());
    eprintln!(
        "[bisect:fused_residual_ln] first={:?} has_nan={has_nan}",
        &out[..4.min(out.len())]
    );
    assert!(!has_nan, "FusedResidualLN produced NaN");
}

#[test]
fn bisect_wgpu_full_bert_layer() {
    // Full BERT layer: embedding → LN → attention block → residual+LN
    // → FFN(gelu) → residual+LN. This replicates the structure that
    // makes the 5way_parity bench output NaN.
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{Activation, BinaryOp};
    let f = DType::F32;
    let (b, s, nh, dh) = (1, 3, 2, 4);
    let h = nh * dh;
    let intermediate = h * 4;

    let mut g = Graph::new("bert_layer");
    let ids = g.input("ids", Shape::new(&[b, s], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));

    let emb = g.param("emb", Shape::new(&[16, h], f));
    let h0 = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb, ids],
        Shape::new(&[b, s, h], f),
    );

    // QKV projection
    let qkv_w = g.param("qkv_w", Shape::new(&[h, 3 * h], f));
    let qkv_b = g.param("qkv_b", Shape::new(&[3 * h], f));
    let qkv_mm = g.add_node(Op::MatMul, vec![h0, qkv_w], Shape::new(&[b, s, 3 * h], f));
    let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, Shape::new(&[b, s, 3 * h], f));
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let k = g.add_node(
        Op::Narrow {
            axis: 2,
            start: h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let v = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, s, h], f),
    );

    // Output projection + residual + LN1
    let out_w = g.param("out_w", Shape::new(&[h, h], f));
    let out_b = g.param("out_b", Shape::new(&[h], f));
    let attn_out_mm = g.add_node(Op::MatMul, vec![attn, out_w], Shape::new(&[b, s, h], f));
    let attn_out = g.binary(BinaryOp::Add, attn_out_mm, out_b, Shape::new(&[b, s, h], f));
    let ln1_g = g.param("ln1_g", Shape::new(&[h], f));
    let ln1_b = g.param("ln1_b", Shape::new(&[h], f));
    let res1 = g.add_node(
        Op::FusedResidualLN {
            has_bias: false,
            eps: 1e-5,
        },
        vec![attn_out, h0, ln1_g, ln1_b],
        Shape::new(&[b, s, h], f),
    );

    // FFN intermediate (gelu)
    let ffn1_w = g.param("ffn1_w", Shape::new(&[h, intermediate], f));
    let ffn1_b = g.param("ffn1_b", Shape::new(&[intermediate], f));
    let ffn1_mm = g.add_node(
        Op::MatMul,
        vec![res1, ffn1_w],
        Shape::new(&[b, s, intermediate], f),
    );
    let ffn1_bias = g.binary(
        BinaryOp::Add,
        ffn1_mm,
        ffn1_b,
        Shape::new(&[b, s, intermediate], f),
    );
    let ffn1_gelu = g.add_node(
        Op::Activation(Activation::Gelu),
        vec![ffn1_bias],
        Shape::new(&[b, s, intermediate], f),
    );

    // FFN output + residual + LN2
    let ffn2_w = g.param("ffn2_w", Shape::new(&[intermediate, h], f));
    let ffn2_b = g.param("ffn2_b", Shape::new(&[h], f));
    let ffn2_mm = g.add_node(
        Op::MatMul,
        vec![ffn1_gelu, ffn2_w],
        Shape::new(&[b, s, h], f),
    );
    let ffn2_out = g.binary(BinaryOp::Add, ffn2_mm, ffn2_b, Shape::new(&[b, s, h], f));
    let ln2_g = g.param("ln2_g", Shape::new(&[h], f));
    let ln2_b = g.param("ln2_b", Shape::new(&[h], f));
    let res2 = g.add_node(
        Op::FusedResidualLN {
            has_bias: false,
            eps: 1e-5,
        },
        vec![ffn2_out, res1, ln2_g, ln2_b],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![res2]);

    let ids_v = vec![1.0f32, 2.0, 3.0];
    let mask_v = vec![1.0f32; b * s];
    let emb_v: Vec<f32> = (0..(16 * h)).map(|i| (i as f32) * 0.01).collect();
    let qkv_w_v: Vec<f32> = (0..(h * 3 * h)).map(|i| ((i % 7) as f32) * 0.05).collect();
    let qkv_b_v: Vec<f32> = (0..(3 * h)).map(|i| (i as f32) * 0.001).collect();
    let out_w_v: Vec<f32> = (0..(h * h)).map(|i| ((i % 5) as f32) * 0.05).collect();
    let out_b_v: Vec<f32> = (0..h).map(|i| (i as f32) * 0.001).collect();
    let ln1_g_v = vec![1.0f32; h];
    let ln1_b_v = vec![0.0f32; h];
    let ffn1_w_v: Vec<f32> = (0..(h * intermediate))
        .map(|i| ((i % 9) as f32) * 0.02)
        .collect();
    let ffn1_b_v: Vec<f32> = (0..intermediate).map(|i| (i as f32) * 0.001).collect();
    let ffn2_w_v: Vec<f32> = (0..(intermediate * h))
        .map(|i| ((i % 11) as f32) * 0.02)
        .collect();
    let ffn2_b_v: Vec<f32> = (0..h).map(|i| (i as f32) * 0.001).collect();
    let ln2_g_v = vec![1.0f32; h];
    let ln2_b_v = vec![0.0f32; h];

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("emb", &emb_v);
    exe.set_param("qkv_w", &qkv_w_v);
    exe.set_param("qkv_b", &qkv_b_v);
    exe.set_param("out_w", &out_w_v);
    exe.set_param("out_b", &out_b_v);
    exe.set_param("ln1_g", &ln1_g_v);
    exe.set_param("ln1_b", &ln1_b_v);
    exe.set_param("ffn1_w", &ffn1_w_v);
    exe.set_param("ffn1_b", &ffn1_b_v);
    exe.set_param("ffn2_w", &ffn2_w_v);
    exe.set_param("ffn2_b", &ffn2_b_v);
    exe.set_param("ln2_g", &ln2_g_v);
    exe.set_param("ln2_b", &ln2_b_v);
    let out = exe
        .run(&[("ids", &ids_v), ("mask", &mask_v)])
        .into_iter()
        .next()
        .unwrap();
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    eprintln!(
        "[bisect:full_bert_layer] len={} nan_count={}/{} first={:?}",
        out.len(),
        nan_count,
        out.len(),
        &out[..4.min(out.len())]
    );
    assert_eq!(
        nan_count, 0,
        "full BERT layer produced {nan_count} NaN values"
    );
}

#[test]
fn bisect_wgpu_full_bert_realistic_dim() {
    // Same single-layer BERT as `bisect_wgpu_full_bert_layer` but
    // with realistic dims (h=384, the MiniLM6 hidden size). Tests
    // whether wgpu breaks at production-scale shapes.
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{Activation, BinaryOp};
    let f = DType::F32;
    let (b, s, nh, dh) = (1, 6, 12, 32);
    let h = nh * dh; // 384
    let intermediate = h * 4; // 1536

    let mut g = Graph::new("bert_real");
    let ids = g.input("ids", Shape::new(&[b, s], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));
    let emb = g.param("emb", Shape::new(&[100, h], f));
    let h0 = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb, ids],
        Shape::new(&[b, s, h], f),
    );
    let qkv_w = g.param("qkv_w", Shape::new(&[h, 3 * h], f));
    let qkv_b = g.param("qkv_b", Shape::new(&[3 * h], f));
    let qkv_mm = g.add_node(Op::MatMul, vec![h0, qkv_w], Shape::new(&[b, s, 3 * h], f));
    let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, Shape::new(&[b, s, 3 * h], f));
    let q = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let k = g.add_node(
        Op::Narrow {
            axis: 2,
            start: h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let v = g.add_node(
        Op::Narrow {
            axis: 2,
            start: 2 * h,
            len: h,
        },
        vec![qkv],
        Shape::new(&[b, s, h], f),
    );
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, s, h], f),
    );
    let out_w = g.param("out_w", Shape::new(&[h, h], f));
    let out_b = g.param("out_b", Shape::new(&[h], f));
    let attn_out_mm = g.add_node(Op::MatMul, vec![attn, out_w], Shape::new(&[b, s, h], f));
    let attn_out = g.binary(BinaryOp::Add, attn_out_mm, out_b, Shape::new(&[b, s, h], f));
    let ln1_g = g.param("ln1_g", Shape::new(&[h], f));
    let ln1_b = g.param("ln1_b", Shape::new(&[h], f));
    let res1 = g.add_node(
        Op::FusedResidualLN {
            has_bias: false,
            eps: 1e-5,
        },
        vec![attn_out, h0, ln1_g, ln1_b],
        Shape::new(&[b, s, h], f),
    );
    let ffn1_w = g.param("ffn1_w", Shape::new(&[h, intermediate], f));
    let ffn1_b = g.param("ffn1_b", Shape::new(&[intermediate], f));
    let ffn1_mm = g.add_node(
        Op::MatMul,
        vec![res1, ffn1_w],
        Shape::new(&[b, s, intermediate], f),
    );
    let ffn1_bias = g.binary(
        BinaryOp::Add,
        ffn1_mm,
        ffn1_b,
        Shape::new(&[b, s, intermediate], f),
    );
    let ffn1_gelu = g.add_node(
        Op::Activation(Activation::Gelu),
        vec![ffn1_bias],
        Shape::new(&[b, s, intermediate], f),
    );
    let ffn2_w = g.param("ffn2_w", Shape::new(&[intermediate, h], f));
    let ffn2_b = g.param("ffn2_b", Shape::new(&[h], f));
    let ffn2_mm = g.add_node(
        Op::MatMul,
        vec![ffn1_gelu, ffn2_w],
        Shape::new(&[b, s, h], f),
    );
    let ffn2_out = g.binary(BinaryOp::Add, ffn2_mm, ffn2_b, Shape::new(&[b, s, h], f));
    let ln2_g = g.param("ln2_g", Shape::new(&[h], f));
    let ln2_b = g.param("ln2_b", Shape::new(&[h], f));
    let res2 = g.add_node(
        Op::FusedResidualLN {
            has_bias: false,
            eps: 1e-5,
        },
        vec![ffn2_out, res1, ln2_g, ln2_b],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![res2]);

    let ids_v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask_v = vec![1.0f32; b * s];
    // Tiny weight magnitudes so intermediate values don't overflow f32.
    let small = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| ((i % 31) as f32 - 15.0) * scale).collect()
    };
    let emb_v = small(100 * h, 0.01);
    let qkv_w_v = small(h * 3 * h, 0.01);
    let qkv_b_v = small(3 * h, 0.001);
    let out_w_v = small(h * h, 0.01);
    let out_b_v = small(h, 0.001);
    let ln1_g_v = vec![1.0f32; h];
    let ln1_b_v = vec![0.0f32; h];
    let ffn1_w_v = small(h * intermediate, 0.01);
    let ffn1_b_v = small(intermediate, 0.001);
    let ffn2_w_v = small(intermediate * h, 0.01);
    let ffn2_b_v = small(h, 0.001);
    let ln2_g_v = vec![1.0f32; h];
    let ln2_b_v = vec![0.0f32; h];

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("emb", &emb_v);
    exe.set_param("qkv_w", &qkv_w_v);
    exe.set_param("qkv_b", &qkv_b_v);
    exe.set_param("out_w", &out_w_v);
    exe.set_param("out_b", &out_b_v);
    exe.set_param("ln1_g", &ln1_g_v);
    exe.set_param("ln1_b", &ln1_b_v);
    exe.set_param("ffn1_w", &ffn1_w_v);
    exe.set_param("ffn1_b", &ffn1_b_v);
    exe.set_param("ffn2_w", &ffn2_w_v);
    exe.set_param("ffn2_b", &ffn2_b_v);
    exe.set_param("ln2_g", &ln2_g_v);
    exe.set_param("ln2_b", &ln2_b_v);
    let out = exe
        .run(&[("ids", &ids_v), ("mask", &mask_v)])
        .into_iter()
        .next()
        .unwrap();
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    eprintln!(
        "[bisect:bert_realistic h=384] len={} nan={}/{} first={:?}",
        out.len(),
        nan_count,
        out.len(),
        &out[..4.min(out.len())]
    );
    assert_eq!(
        nan_count,
        0,
        "full BERT layer at realistic dim produced {nan_count}/{} NaN",
        out.len()
    );
}

#[test]
fn bisect_wgpu_full_bert_via_models_bert() {
    // Use the actual rlx-models BERT builder to construct the same
    // graph the bench uses, then run on wgpu and check for NaN.
    // Smaller cfg than minilm6 to keep the test fast.
    if !rlx_wgpu::is_available() {
        return;
    }
    let f = DType::F32;
    let mut g = Graph::new("bert_real_2layer");
    let (b, s, h, nh, dh, n_layers, vocab) = (1, 4, 32, 4, 8, 2, 100);
    let intermediate = h * 4;

    // Manually build a 2-layer BERT-ish graph (matches the structure
    // rlx-models::bert produces).
    use rlx_ir::op::{Activation, BinaryOp};
    let ids = g.input("ids", Shape::new(&[b, s], f));
    let mask = g.input("mask", Shape::new(&[b, s], f));
    let emb = g.param("emb", Shape::new(&[vocab, h], f));
    let mut h_id = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb, ids],
        Shape::new(&[b, s, h], f),
    );
    for l in 0..n_layers {
        let qkv_w = g.param(format!("qkv_w_{l}"), Shape::new(&[h, 3 * h], f));
        let qkv_b = g.param(format!("qkv_b_{l}"), Shape::new(&[3 * h], f));
        let qkv_mm = g.add_node(Op::MatMul, vec![h_id, qkv_w], Shape::new(&[b, s, 3 * h], f));
        let qkv = g.binary(BinaryOp::Add, qkv_mm, qkv_b, Shape::new(&[b, s, 3 * h], f));
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: h,
            },
            vec![qkv],
            Shape::new(&[b, s, h], f),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: h,
                len: h,
            },
            vec![qkv],
            Shape::new(&[b, s, h], f),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 2 * h,
                len: h,
            },
            vec![qkv],
            Shape::new(&[b, s, h], f),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: dh,
                mask_kind: MaskKind::Custom,
            },
            vec![q, k, v, mask],
            Shape::new(&[b, s, h], f),
        );
        let out_w = g.param(format!("out_w_{l}"), Shape::new(&[h, h], f));
        let out_b = g.param(format!("out_b_{l}"), Shape::new(&[h], f));
        let attn_mm = g.add_node(Op::MatMul, vec![attn, out_w], Shape::new(&[b, s, h], f));
        let attn_out = g.binary(BinaryOp::Add, attn_mm, out_b, Shape::new(&[b, s, h], f));
        let ln1_g = g.param(format!("ln1_g_{l}"), Shape::new(&[h], f));
        let ln1_b = g.param(format!("ln1_b_{l}"), Shape::new(&[h], f));
        let res1 = g.add_node(
            Op::FusedResidualLN {
                has_bias: false,
                eps: 1e-5,
            },
            vec![attn_out, h_id, ln1_g, ln1_b],
            Shape::new(&[b, s, h], f),
        );
        let ffn1_w = g.param(format!("ffn1_w_{l}"), Shape::new(&[h, intermediate], f));
        let ffn1_b = g.param(format!("ffn1_b_{l}"), Shape::new(&[intermediate], f));
        let ffn1_mm = g.add_node(
            Op::MatMul,
            vec![res1, ffn1_w],
            Shape::new(&[b, s, intermediate], f),
        );
        let ffn1_bias = g.binary(
            BinaryOp::Add,
            ffn1_mm,
            ffn1_b,
            Shape::new(&[b, s, intermediate], f),
        );
        let ffn1_gelu = g.add_node(
            Op::Activation(Activation::Gelu),
            vec![ffn1_bias],
            Shape::new(&[b, s, intermediate], f),
        );
        let ffn2_w = g.param(format!("ffn2_w_{l}"), Shape::new(&[intermediate, h], f));
        let ffn2_b = g.param(format!("ffn2_b_{l}"), Shape::new(&[h], f));
        let ffn2_mm = g.add_node(
            Op::MatMul,
            vec![ffn1_gelu, ffn2_w],
            Shape::new(&[b, s, h], f),
        );
        let ffn2_out = g.binary(BinaryOp::Add, ffn2_mm, ffn2_b, Shape::new(&[b, s, h], f));
        let ln2_g = g.param(format!("ln2_g_{l}"), Shape::new(&[h], f));
        let ln2_b = g.param(format!("ln2_b_{l}"), Shape::new(&[h], f));
        h_id = g.add_node(
            Op::FusedResidualLN {
                has_bias: false,
                eps: 1e-5,
            },
            vec![ffn2_out, res1, ln2_g, ln2_b],
            Shape::new(&[b, s, h], f),
        );
    }
    g.set_outputs(vec![h_id]);

    // Tiny weight values
    let small = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| ((i % 31) as f32 - 15.0) * scale).collect()
    };
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("emb", &small(vocab * h, 0.01));
    for l in 0..n_layers {
        exe.set_param(&format!("qkv_w_{l}"), &small(h * 3 * h, 0.01));
        exe.set_param(&format!("qkv_b_{l}"), &small(3 * h, 0.001));
        exe.set_param(&format!("out_w_{l}"), &small(h * h, 0.01));
        exe.set_param(&format!("out_b_{l}"), &small(h, 0.001));
        exe.set_param(&format!("ln1_g_{l}"), &vec![1.0f32; h]);
        exe.set_param(&format!("ln1_b_{l}"), &vec![0.0f32; h]);
        exe.set_param(&format!("ffn1_w_{l}"), &small(h * intermediate, 0.01));
        exe.set_param(&format!("ffn1_b_{l}"), &small(intermediate, 0.001));
        exe.set_param(&format!("ffn2_w_{l}"), &small(intermediate * h, 0.01));
        exe.set_param(&format!("ffn2_b_{l}"), &small(h, 0.001));
        exe.set_param(&format!("ln2_g_{l}"), &vec![1.0f32; h]);
        exe.set_param(&format!("ln2_b_{l}"), &vec![0.0f32; h]);
    }
    let ids_v = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask_v = vec![1.0f32; b * s];
    let out = exe
        .run(&[("ids", &ids_v), ("mask", &mask_v)])
        .into_iter()
        .next()
        .unwrap();
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    eprintln!(
        "[bisect:bert_2layer] len={} nan={}/{} first={:?}",
        out.len(),
        nan_count,
        out.len(),
        &out[..4.min(out.len())]
    );
    assert_eq!(
        nan_count,
        0,
        "2-layer BERT produced {nan_count}/{} NaN",
        out.len()
    );
}

#[test]
fn bisect_wgpu_bert_input_prep() {
    // 3 gathers (word + position + token_type) + 2 adds + LN.
    // This is the BERT embedding prep that the bench actually uses.
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::BinaryOp;
    let f = DType::F32;
    let (b, s, h) = (1, 4, 16);
    let vocab = 100;

    let mut g = Graph::new("bert_input_prep");
    let ids = g.input("ids", Shape::new(&[b, s], f));
    let pos_ids = g.input("pos_ids", Shape::new(&[b, s], f));
    let tt_ids = g.input("tt_ids", Shape::new(&[b, s], f));

    let word_emb = g.param("word_emb", Shape::new(&[vocab, h], f));
    let pos_emb = g.param("pos_emb", Shape::new(&[vocab, h], f));
    let tt_emb = g.param("tt_emb", Shape::new(&[2, h], f));
    let ln_g = g.param("ln_g", Shape::new(&[h], f));
    let ln_b = g.param("ln_b", Shape::new(&[h], f));

    let word_out = g.add_node(
        Op::Gather { axis: 0 },
        vec![word_emb, ids],
        Shape::new(&[b, s, h], f),
    );
    let pos_out = g.add_node(
        Op::Gather { axis: 0 },
        vec![pos_emb, pos_ids],
        Shape::new(&[b, s, h], f),
    );
    let tt_out = g.add_node(
        Op::Gather { axis: 0 },
        vec![tt_emb, tt_ids],
        Shape::new(&[b, s, h], f),
    );
    let wp = g.binary(BinaryOp::Add, word_out, pos_out, Shape::new(&[b, s, h], f));
    let sum = g.binary(BinaryOp::Add, wp, tt_out, Shape::new(&[b, s, h], f));
    let ln = g.add_node(
        Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![sum, ln_g, ln_b],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![ln]);

    let small = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| ((i % 31) as f32 - 15.0) * scale).collect()
    };
    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("word_emb", &small(vocab * h, 0.01));
    exe.set_param("pos_emb", &small(vocab * h, 0.01));
    exe.set_param("tt_emb", &small(2 * h, 0.01));
    exe.set_param("ln_g", &vec![1.0f32; h]);
    exe.set_param("ln_b", &vec![0.0f32; h]);

    let ids_v = vec![1.0f32, 2.0, 3.0, 4.0];
    let pos_ids_v = vec![0.0f32, 1.0, 2.0, 3.0];
    let tt_ids_v = vec![0.0f32, 0.0, 0.0, 0.0];
    let out = exe
        .run(&[
            ("ids", &ids_v),
            ("pos_ids", &pos_ids_v),
            ("tt_ids", &tt_ids_v),
        ])
        .into_iter()
        .next()
        .unwrap();
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    eprintln!(
        "[bisect:bert_input_prep] len={} nan={}/{} first={:?}",
        out.len(),
        nan_count,
        out.len(),
        &out[..4.min(out.len())]
    );
    assert_eq!(
        nan_count,
        0,
        "BERT input prep produced {nan_count}/{} NaN",
        out.len()
    );
}

#[test]
fn region_relu_matches_atomic() {
    // Smallest possible region: one Relu chain step. If THIS fails,
    // the storage-binding path is broken end-to-end. Tests output
    // against CPU-reference values directly (rather than against the
    // atomic graph, which would also be running on wgpu).
    if !rlx_wgpu::is_available() {
        return;
    }
    use rlx_ir::op::{ChainOperand, ChainStep};

    let mut g_reg = Graph::new("relu_region");
    let xr = g_reg.input("x", Shape::new(&[8], DType::F32));
    let chain = vec![ChainStep::Activation(
        Activation::Relu,
        ChainOperand::Input(0),
    )];
    let region = g_reg.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0u32; 16],
        },
        vec![xr],
        Shape::new(&[8], DType::F32),
    );
    g_reg.set_outputs(vec![region]);

    let xs = vec![-1.0f32, 0.0, 0.5, 1.0, -2.0, 3.0, -0.5, 2.5];
    let mut reg = WgpuExecutable::compile(g_reg);
    let got_reg = reg.run(&[("x", &xs)]).into_iter().next().unwrap();
    let want: Vec<f32> = xs.iter().map(|v| v.max(0.0)).collect();
    assert!(
        close(&got_reg, &want, 1e-5),
        "region mismatch: got {got_reg:?} want {want:?}"
    );
}

#[test]
fn matmul_2x3x2_matches_cpu_reference() {
    if !rlx_wgpu::is_available() {
        eprintln!("rlx-wgpu: no compatible adapter; skipping test");
        return;
    }
    let g = build_graph();
    let mut exe = WgpuExecutable::compile(g);

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    exe.set_param("w", &w);

    let outs = exe.run(&[("x", &x)]);
    assert_eq!(outs.len(), 1);
    let want = matmul_ref(&x, &w, 2, 3, 2);
    assert!(
        close(&outs[0], &want, 1e-4),
        "matmul mismatch: got {:?} want {want:?}",
        outs[0]
    );
}
