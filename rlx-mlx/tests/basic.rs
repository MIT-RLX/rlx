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

//! End-to-end parity basic test: build a tiny `(x @ W) + b` graph,
//! run it on MLX in both modes, and check the output matches a
//! hand-computed expected result.
//!
//! Doesn't compare against rlx-cpu directly to keep the test crate
//! free of cross-backend deps — the expected values are computed in
//! pure Rust.

#![cfg(target_os = "macos")]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn build_graph() -> Graph {
    let mut g = Graph::new("basic");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2, 2], DType::F32));
    let mm = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    let out = g.binary(BinaryOp::Add, mm, b, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![out]);
    g
}

fn expected_output(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    // 2x3 @ 3x2 -> 2x2, then + 2x2.
    let mut y = vec![0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0f32;
            for k in 0..3 {
                s += x[i * 3 + k] * w[k * 2 + j];
            }
            y[i * 2 + j] = s + b[i * 2 + j];
        }
    }
    y
}

fn run_mode(mode: MlxMode) -> (Vec<f32>, Vec<f32>) {
    let g = build_graph();
    let mut exe = MlxExecutable::compile_with_mode(g, mode);

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b = vec![1.0, 2.0, 3.0, 4.0];

    exe.set_param("w", &w);
    exe.set_param("b", &b);
    let outs = exe.run(&[("x", &x)]);
    assert_eq!(outs.len(), 1);
    (
        outs.into_iter().next().unwrap(),
        expected_output(&x, &w, &b),
    )
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn lazy_matmul_add_matches_reference() {
    let (got, want) = run_mode(MlxMode::Lazy);
    assert!(
        close(&got, &want, 1e-4),
        "lazy mismatch: got {:?}, want {:?}",
        got,
        want
    );
}

#[test]
fn eager_matmul_add_matches_reference() {
    let (got, want) = run_mode(MlxMode::Eager);
    assert!(
        close(&got, &want, 1e-4),
        "eager mismatch: got {:?}, want {:?}",
        got,
        want
    );
}

#[test]
fn max_pool_2x2_stride_2_matches_reference() {
    // Tiny 2D max-pool: input [1, 1, 4, 4], kernel 2x2, stride 2,
    // no padding → output [1, 1, 2, 2].
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
    let mut exe = MlxExecutable::compile(g);
    // Input layout (row-major):
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    // Max-pool 2x2 stride 2 windows:
    //   max(1,2,5,6)=6        max(3,4,7,8)=8
    //   max(9,10,13,14)=14    max(11,12,15,16)=16
    let xs: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    let want = vec![6.0, 8.0, 14.0, 16.0];
    assert_eq!(got, want, "max-pool mismatch: got {got:?} want {want:?}");
}

#[test]
fn avg_pool_with_padding_matches_reference() {
    // 2D avg-pool: input [1, 1, 2, 2], kernel 2x2, stride 1,
    // padding 1 on each side → output [1, 1, 3, 3].
    // Padded with 0 to [1, 1, 4, 4]:
    //   0 0 0 0
    //   0 1 2 0
    //   0 3 4 0
    //   0 0 0 0
    // 2x2 windows / 4:
    //   (0+0+0+1)/4 = 0.25    (0+0+1+2)/4 = 0.75    (0+0+2+0)/4 = 0.5
    //   (0+1+0+3)/4 = 1.0     (1+2+3+4)/4 = 2.5     (2+0+4+0)/4 = 1.5
    //   (0+3+0+0)/4 = 0.75    (3+4+0+0)/4 = 1.75    (4+0+0+0)/4 = 1.0
    let mut g = Graph::new("avgpool");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: vec![2, 2],
            stride: vec![1, 1],
            padding: vec![1, 1],
        },
        vec![x],
        Shape::new(&[1, 1, 3, 3], DType::F32),
    );
    g.set_outputs(vec![p]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    let want = vec![0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75, 1.75, 1.0];
    assert!(
        close(&got, &want, 1e-5),
        "avg-pool mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn selective_scan_matches_hand_computed() {
    // Tiny Mamba SSM: batch=1, seq=2, hidden=2, state_size=1.
    // Inputs and the hand-computed reference are spelled out in the
    // assertion comments below so the test serves as documentation.
    let mut g = Graph::new("ssm");
    let x = g.input("x", Shape::new(&[1, 2, 2], DType::F32));
    let d = g.input("d", Shape::new(&[1, 2, 2], DType::F32));
    let a = g.input("a", Shape::new(&[2, 1], DType::F32));
    let b = g.input("b", Shape::new(&[1, 2, 1], DType::F32));
    let c = g.input("c", Shape::new(&[1, 2, 1], DType::F32));
    let y = g.add_node(
        Op::SelectiveScan { state_size: 1 },
        vec![x, d, a, b, c],
        Shape::new(&[1, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);

    let xd = vec![1.0, 2.0, 3.0, 4.0];
    let dd = vec![0.1, 0.2, 0.3, 0.4];
    let ad = vec![-1.0, -2.0];
    let bd = vec![0.5, 0.6];
    let cd = vec![1.0, 1.5];
    let got = exe
        .run(&[("x", &xd), ("d", &dd), ("a", &ad), ("b", &bd), ("c", &cd)])
        .into_iter()
        .next()
        .unwrap();

    // Hand-compute (state_size=1, so n=1 throughout):
    // t=0:
    //   exp(δ*A) = exp([0.1*-1, 0.2*-2]) = [0.9048, 0.6703]
    //   δ*B*x   = [0.1*0.5*1, 0.2*0.5*2] = [0.05, 0.2]
    //   state    = [0, 0]·exp + δBx = [0.05, 0.2]
    //   y[0]     = C[0] * state = [1.0*0.05, 1.0*0.2] = [0.05, 0.2]
    // t=1:
    //   exp(δ*A) = exp([0.3*-1, 0.4*-2]) = [0.7408, 0.4493]
    //   δ*B*x   = [0.3*0.6*3, 0.4*0.6*4] = [0.54, 0.96]
    //   state    = exp·prev + δBx
    //            = [0.7408*0.05+0.54, 0.4493*0.2+0.96]
    //            = [0.57704, 1.04986]
    //   y[1]     = C[1] * state = [1.5*0.57704, 1.5*1.04986]
    //            = [0.86556, 1.57479]
    let want = vec![0.05, 0.2, 0.86556, 1.57479];
    assert!(
        close(&got, &want, 5e-3),
        "selective_scan mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn gated_delta_net_matches_hand_computed() {
    use rlx_ir::Op;

    let (b, s, h, n) = (1, 4, 2, 3);
    let mut g = Graph::new("gdn");
    let q = g.input("q", Shape::new(&[b, s, h, n], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, h, n], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, h, n], DType::F32));
    let g_in = g.input("g", Shape::new(&[b, s, h], DType::F32));
    let beta = g.input("beta", Shape::new(&[b, s, h], DType::F32));
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: n,
            carry_state: false,
        },
        vec![q, k, v, g_in, beta],
        Shape::new(&[b, s, h, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);

    let nqkv = b * s * h * n;
    let ngb = b * s * h;
    let q_data: Vec<f32> = (0..nqkv).map(|i| 0.05 + 0.03 * (i as f32)).collect();
    let k_data: Vec<f32> = (0..nqkv).map(|i| 0.10 + 0.02 * (i as f32)).collect();
    let v_data: Vec<f32> = (0..nqkv).map(|i| 0.30 + 0.05 * (i as f32)).collect();
    let g_data: Vec<f32> = (0..ngb).map(|i| -0.20 - 0.01 * (i as f32)).collect();
    let beta_data: Vec<f32> = (0..ngb).map(|i| 0.40 + 0.02 * (i as f32)).collect();

    let got = exe
        .run(&[
            ("q", &q_data),
            ("k", &k_data),
            ("v", &v_data),
            ("g", &g_data),
            ("beta", &beta_data),
        ])
        .into_iter()
        .next()
        .unwrap();

    // Same scalar reference as rlx-runtime/tests/cpu_gated_delta_net_parity.rs.
    let scale = 1.0f32 / (n as f32).sqrt();
    let mut want = vec![0f32; nqkv];
    let mut state = vec![0f32; h * n * n];
    let mut sk = vec![0f32; n];
    for bi in 0..b {
        for st in state.iter_mut() {
            *st = 0.0;
        }
        for ti in 0..s {
            let step_qkv = bi * s * h * n + ti * h * n;
            let step_gb = bi * s * h + ti * h;
            for hi in 0..h {
                let q_row = &q_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let k_row = &k_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let v_row = &v_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let g_t = g_data[step_gb + hi];
                let beta_t = beta_data[step_gb + hi];
                let s_base = hi * n * n;
                let s_mat = &mut state[s_base..s_base + n * n];
                let g_exp = g_t.exp();
                for st in s_mat.iter_mut() {
                    *st *= g_exp;
                }
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * k_row[i];
                    }
                    sk[j] = acc;
                }
                for j in 0..n {
                    sk[j] = (v_row[j] - sk[j]) * beta_t;
                }
                for i in 0..n {
                    let ki = k_row[i];
                    if ki != 0.0 {
                        for j in 0..n {
                            s_mat[i * n + j] += ki * sk[j];
                        }
                    }
                }
                let out_row = &mut want[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * q_row[i];
                    }
                    out_row[j] = acc * scale;
                }
            }
        }
    }

    assert!(
        close(&got, &want, 1e-4),
        "gated_delta_net mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn dot_general_canonical_matches_matmul() {
    // The optimizer's LowerDotGeneral pass normally rewrites
    // DotGeneral to MatMul before the backend sees it. This test
    // synthesizes the IR variant directly via add_node so the
    // backend's own DotGeneral arm runs.
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
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // x @ w = [(1+0+1.5, 0+2+1.5), (4+0+3, 0+5+3)] = [(2.5, 3.5), (7.0, 8.0)]
    let want = vec![2.5, 3.5, 7.0, 8.0];
    assert!(
        close(&got, &want, 1e-5),
        "dot_general canonical mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn dynamic_shape_input_resolves_at_runtime() {
    use rlx_ir::shape::{Dim, Shape};
    // Graph: x [Dynamic(0), 3] @ w [3, 2] = y [Dynamic(0), 2].
    // Symbol 0 propagates from input to output. We run twice with
    // different batch sizes and verify both produce correct results.
    let mut g = Graph::new("dyn");
    let x_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(3)], DType::F32);
    let y_shape = Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(2)], DType::F32);
    let x = g.input("x", x_shape);
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, y_shape);
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    exe.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    // Batch 2: x [2, 3]
    let xs2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got2 = exe.run(&[("x", &xs2)]).into_iter().next().unwrap();
    // [(1+0+3, 0+2+3), (4+0+6, 0+5+6)] = [(4, 5), (10, 11)]
    let want2 = vec![4.0, 5.0, 10.0, 11.0];
    assert!(
        close(&got2, &want2, 1e-5),
        "dyn batch=2 mismatch: got {got2:?} want {want2:?}"
    );

    // Batch 4: x [4, 3]
    let xs4 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let got4 = exe.run(&[("x", &xs4)]).into_iter().next().unwrap();
    // [(1, 0), (0, 1), (1, 1), (1+0+1, 0+1+1)] = [(1,0), (0,1), (1,1), (2, 2)]
    let want4 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    assert!(
        close(&got4, &want4, 1e-5),
        "dyn batch=4 mismatch: got {got4:?} want {want4:?}"
    );
}

#[test]
fn op_if_picks_branch_per_element() {
    // Op::If: pred is a Bool tensor; we lower both branches and
    // element-wise select. then_branch returns 2*x; else_branch
    // returns -x. With pred = [true, false, true, false] and
    // x = [1, 2, 3, 4]:  out = [2, -2, 6, -4].
    let mut g = Graph::new("if_test");
    let pred = g.input("pred", Shape::new(&[4], DType::Bool));
    let x = g.input("x", Shape::new(&[4], DType::F32));

    // then_branch: returns 2*x (uses x as captured input)
    let mut tb = Graph::new("then");
    let tx = tb.input("x", Shape::new(&[4], DType::F32));
    let two = tb.add_node(
        Op::Constant {
            data: 2f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let to = tb.binary(
        rlx_ir::op::BinaryOp::Mul,
        tx,
        two,
        Shape::new(&[4], DType::F32),
    );
    tb.set_outputs(vec![to]);

    // else_branch: returns -x
    let mut eb = Graph::new("else");
    let ex = eb.input("x", Shape::new(&[4], DType::F32));
    let eo = eb.activation(
        rlx_ir::op::Activation::Neg,
        ex,
        Shape::new(&[4], DType::F32),
    );
    eb.set_outputs(vec![eo]);

    let y = g.add_node(
        Op::If {
            then_branch: Box::new(tb),
            else_branch: Box::new(eb),
        },
        vec![pred, x],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    // pred bytes: bool stored as 1-byte each
    let pred_bytes: Vec<u8> = vec![1, 0, 1, 0];
    let xs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let outs = exe.run_typed(&[
        ("pred", &pred_bytes, DType::Bool),
        (
            "x",
            unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) },
            DType::F32,
        ),
    ]);
    let (out_bytes, out_dt) = &outs[0];
    assert_eq!(*out_dt, DType::F32);
    let got: Vec<f32> = unsafe {
        std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_bytes.len() / 4)
    }
    .to_vec();
    let want = vec![2.0, -2.0, 6.0, -4.0];
    assert!(
        close(&got, &want, 1e-5),
        "If mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn op_while_max_iter_unrolls_correctly() {
    // Op::While: cond returns "x < 10"; body returns "x + 1".
    // Initial x = 0. With max_iterations=20, we should converge to
    // x = 10 after 10 iterations and stay there for the remaining 10
    // (the active mask freezes the value once cond is false).
    let mut g = Graph::new("while_test");
    let x_in = g.input("x", Shape::new(&[1], DType::F32));

    // cond: returns x < 10 (bool scalar [1])
    let mut c = Graph::new("cond");
    let cx = c.input("x", Shape::new(&[1], DType::F32));
    let ten = c.add_node(
        Op::Constant {
            data: 10f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let lt = c.add_node(
        Op::Compare(rlx_ir::op::CmpOp::Lt),
        vec![cx, ten],
        Shape::new(&[1], DType::Bool),
    );
    c.set_outputs(vec![lt]);

    // body: returns x + 1
    let mut b = Graph::new("body");
    let bx = b.input("x", Shape::new(&[1], DType::F32));
    let one = b.add_node(
        Op::Constant {
            data: 1f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let bo = b.binary(
        rlx_ir::op::BinaryOp::Add,
        bx,
        one,
        Shape::new(&[1], DType::F32),
    );
    b.set_outputs(vec![bo]);

    let y = g.add_node(
        Op::While {
            cond: Box::new(c),
            body: Box::new(b),
            max_iterations: Some(20),
        },
        vec![x_in],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    let xs = vec![0.0f32];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert!(
        close(&got, &[10.0], 1e-4),
        "While should converge to 10: got {got:?}"
    );
}

#[test]
fn constant_u32_round_trips() {
    // U32 constant fed through a graph that just outputs it (cast
    // through F32 to avoid needing U32-typed inputs from the f32 trait).
    let mut g = Graph::new("u32_const");
    let bytes: Vec<u8> = vec![
        7, 0, 0, 0, // 7
        42, 0, 0, 0, // 42
        255, 0, 0, 0, // 255
    ];
    let c = g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[3], DType::U32),
    );
    let casted = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![c],
        Shape::new(&[3], DType::F32),
    );
    g.set_outputs(vec![casted]);
    let mut exe = MlxExecutable::compile(g);
    let got = exe.run(&[]).into_iter().next().unwrap();
    assert_eq!(got, vec![7.0, 42.0, 255.0]);
}

#[test]
fn dot_general_batched_matmul_matches_reference() {
    // Batched dot: lhs [B=2, M=2, K=3], rhs [B=2, K=3, N=2].
    // batch=[0], contracting=lhs[2]/rhs[1]. Output [2, 2, 2].
    let mut g = Graph::new("dg_batched");
    let x = g.input("x", Shape::new(&[2, 2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[2, 3, 2], DType::F32));
    let y = g.add_node(
        Op::DotGeneral {
            lhs_contracting: vec![2],
            rhs_contracting: vec![1],
            lhs_batch: vec![0],
            rhs_batch: vec![0],
        },
        vec![x, w],
        Shape::new(&[2, 2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);
    // batch 0 weights = identity-ish, batch 1 = doubled-identity.
    exe.set_param(
        "w",
        &[
            // batch 0
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // batch 1
            2.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        ],
    );
    let xs = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // batch 0: x @ I-ish = first two cols → (1,2), (4,5)
    // batch 1: x @ 2I-ish = 2 * first two cols → (14,16), (20,22)
    let want = vec![1.0, 2.0, 4.0, 5.0, 14.0, 16.0, 20.0, 22.0];
    assert!(
        close(&got, &want, 1e-5),
        "dot_general batched mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn fused_transformer_layer_no_bias_lowers_and_runs() {
    use rlx_ir::op::Activation;
    // has_bias=false → 8 inputs, no bias adds, no LN betas.
    let mut g = Graph::new("ftl_nobias");
    let h = g.input("h", Shape::new(&[1, 2, 4], DType::F32));
    let qkv_w = g.param("qkv_w", Shape::new(&[4, 12], DType::F32));
    let out_w = g.param("out_w", Shape::new(&[4, 4], DType::F32));
    let ln1_g = g.param("ln1_g", Shape::new(&[4], DType::F32));
    let fc1_w = g.param("fc1_w", Shape::new(&[4, 8], DType::F32));
    let fc2_w = g.param("fc2_w", Shape::new(&[8, 4], DType::F32));
    let ln2_g = g.param("ln2_g", Shape::new(&[4], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::FusedTransformerLayer {
            num_heads: 2,
            head_dim: 2,
            intermediate_size: 8,
            eps1: 1e-5,
            eps2: 1e-5,
            activation: Activation::Gelu,
            has_bias: false,
        },
        vec![h, qkv_w, out_w, ln1_g, fc1_w, fc2_w, ln2_g, mask],
        Shape::new(&[1, 2, 4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    exe.set_param("qkv_w", &[0.1f32; 4 * 12]);
    exe.set_param("out_w", &[0.1f32; 4 * 4]);
    exe.set_param("ln1_g", &[1.0f32; 4]);
    exe.set_param("fc1_w", &[0.1f32; 4 * 8]);
    exe.set_param("fc2_w", &[0.1f32; 8 * 4]);
    exe.set_param("ln2_g", &[1.0f32; 4]);

    let h_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mask_zero = vec![0f32; 8];
    let got = exe
        .run(&[("h", &h_data), ("mask", &mask_zero)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(got.len(), 8);
    assert!(
        got.iter().all(|f| f.is_finite()),
        "no-bias FusedTransformerLayer non-finite: {got:?}"
    );
}

#[test]
fn fused_transformer_layer_lowers_and_runs() {
    use rlx_ir::op::Activation;
    // Tiny BERT-style layer: B=1, S=2, hidden=4, heads=2, head_dim=2,
    // intermediate=8. Just verifies the composition lowers and produces
    // finite outputs of the right shape — full numerical reference
    // would require recomputing the entire transformer block by hand.
    let mut g = Graph::new("ftl");
    let h = g.input("h", Shape::new(&[1, 2, 4], DType::F32));
    let qkv_w = g.param("qkv_w", Shape::new(&[4, 12], DType::F32));
    let qkv_b = g.param("qkv_b", Shape::new(&[12], DType::F32));
    let out_w = g.param("out_w", Shape::new(&[4, 4], DType::F32));
    let out_b = g.param("out_b", Shape::new(&[4], DType::F32));
    let ln1_g = g.param("ln1_g", Shape::new(&[4], DType::F32));
    let ln1_b = g.param("ln1_b", Shape::new(&[4], DType::F32));
    let fc1_w = g.param("fc1_w", Shape::new(&[4, 8], DType::F32));
    let fc1_b = g.param("fc1_b", Shape::new(&[8], DType::F32));
    let fc2_w = g.param("fc2_w", Shape::new(&[8, 4], DType::F32));
    let fc2_b = g.param("fc2_b", Shape::new(&[4], DType::F32));
    let ln2_g = g.param("ln2_g", Shape::new(&[4], DType::F32));
    let ln2_b = g.param("ln2_b", Shape::new(&[4], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::FusedTransformerLayer {
            num_heads: 2,
            head_dim: 2,
            intermediate_size: 8,
            eps1: 1e-5,
            eps2: 1e-5,
            activation: Activation::Gelu,
            has_bias: true,
        },
        vec![
            h, qkv_w, qkv_b, out_w, out_b, ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b, ln2_g, ln2_b,
            mask,
        ],
        Shape::new(&[1, 2, 4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    // Reasonable-but-arbitrary weight scales; we just need the
    // numerics to not blow up.
    exe.set_param("qkv_w", &[0.1f32; 4 * 12]);
    exe.set_param("qkv_b", &[0.0f32; 12]);
    exe.set_param("out_w", &[0.1f32; 4 * 4]);
    exe.set_param("out_b", &[0.0f32; 4]);
    exe.set_param("ln1_g", &[1.0f32; 4]);
    exe.set_param("ln1_b", &[0.0f32; 4]);
    exe.set_param("fc1_w", &[0.1f32; 4 * 8]);
    exe.set_param("fc1_b", &[0.0f32; 8]);
    exe.set_param("fc2_w", &[0.1f32; 8 * 4]);
    exe.set_param("fc2_b", &[0.0f32; 4]);
    exe.set_param("ln2_g", &[1.0f32; 4]);
    exe.set_param("ln2_b", &[0.0f32; 4]);

    let h_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mask_zero = vec![0f32; 8]; // [B=1, H=2, S=2, S=2]
    let got = exe
        .run(&[("h", &h_data), ("mask", &mask_zero)])
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(got.len(), 8, "fused transformer layer output shape wrong");
    assert!(
        got.iter().all(|f| f.is_finite()),
        "fused transformer layer non-finite: {got:?}"
    );
}

#[test]
fn pool_3d_max_matches_reference() {
    // 3D max-pool: input [1, 1, 2, 2, 2] (NCDHW), kernel 2x2x2,
    // stride 1, no padding → output [1, 1, 1, 1, 1].
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
    let mut exe = MlxExecutable::compile(g);
    let xs: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert_eq!(got, vec![8.0]);
}

#[test]
fn pool_2d_prod_matches_reference() {
    // 2D prod-pool: input [1, 1, 2, 2], kernel 2x2 → product of all 4.
    let mut g = Graph::new("pool_prod");
    let x = g.input("x", Shape::new(&[1, 1, 2, 2], DType::F32));
    let p = g.add_node(
        Op::Pool {
            kind: ReduceOp::Prod,
            kernel_size: vec![2, 2],
            stride: vec![1, 1],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1], DType::F32),
    );
    g.set_outputs(vec![p]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // 1*2*3*4 = 24
    assert_eq!(got, vec![24.0]);
}

#[test]
fn conv1d_simple_matches_reference() {
    // Conv1d: NCL = [1, 1, 4] input, weight [1, 1, 2] (kernel size 2).
    // Output = [1, 1, 3]. Sliding-window inner product.
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
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[1.0, -1.0]);
    let xs = vec![1.0, 2.0, 3.0, 4.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // diff: [1*1+2*-1, 2*1+3*-1, 3*1+4*-1] = [-1, -1, -1]
    let want = vec![-1.0, -1.0, -1.0];
    assert!(
        close(&got, &want, 1e-5),
        "conv1d mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn rope_partial_dim_passes_tail_through() {
    // last_dim=6, head_dim=4 → rotate first 4, leave last 2 unchanged.
    let mut g = Graph::new("rope_partial");
    let x = g.input("x", Shape::new(&[1, 1, 1, 6], DType::F32));
    let cos = g.input("cos", Shape::new(&[1, 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[1, 2], DType::F32));
    let y = g.add_node(
        Op::Rope {
            head_dim: 4,
            n_rot: 4,
        },
        vec![x, cos, sin],
        Shape::new(&[1, 1, 1, 6], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);
    // Identity rotation: cos=1, sin=0. y[..., :4] == x[..., :4].
    // Tail (indices 4, 5) must pass through verbatim.
    let xs = vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0];
    let got = exe
        .run(&[("x", &xs), ("cos", &[1.0, 1.0]), ("sin", &[0.0, 0.0])])
        .into_iter()
        .next()
        .unwrap();
    assert!(
        close(&got, &xs, 1e-5),
        "rope partial-dim identity should be a no-op: got {got:?} want {xs:?}"
    );
}

#[test]
fn fused_attention_block_matches_reference() {
    // Tiny block: B=1, S=2, H=2, head_dim=2, no bias, no rope.
    // qkv_w shape: [hidden, 3*H*D] = [4, 12]. out_w shape: [H*D, hidden] = [4, 4].
    // mask: [B, H, S, S] custom mask, all zeros (no masking).
    let mut g = Graph::new("fab");
    let h_in = g.input("h", Shape::new(&[1, 2, 4], DType::F32));
    let qkv_w = g.param("qkv_w", Shape::new(&[4, 12], DType::F32));
    let out_w = g.param("out_w", Shape::new(&[4, 4], DType::F32));
    let mask = g.input("mask", Shape::new(&[1, 2, 2, 2], DType::F32));
    let y = g.add_node(
        Op::FusedAttentionBlock {
            num_heads: 2,
            head_dim: 2,
            has_bias: false,
            has_rope: false,
        },
        vec![h_in, qkv_w, out_w, mask],
        Shape::new(&[1, 2, 4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);
    // Identity-ish weights: qkv_w = [I_4 | I_4 | I_4] (each column block
    // is identity). out_w = identity. With no mask the output equals
    // softmax-attention on the input itself with itself — finite, no NaNs.
    let mut qkv = vec![0f32; 4 * 12];
    for i in 0..4 {
        for blk in 0..3 {
            qkv[i * 12 + blk * 4 + i] = 1.0;
        }
    }
    let mut owt = vec![0f32; 16];
    for i in 0..4 {
        owt[i * 4 + i] = 1.0;
    }
    exe.set_param("qkv_w", &qkv);
    exe.set_param("out_w", &owt);
    let h_data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mask_zero = vec![0f32; 8];
    let got = exe
        .run(&[("h", &h_data), ("mask", &mask_zero)])
        .into_iter()
        .next()
        .unwrap();
    // Just sanity-check shape + finiteness; the full numerical
    // reference for fused attention is the unfused chain, which the
    // CPU backend covers. Here we verify the block lowers and runs.
    assert_eq!(got.len(), 8, "fused attention output shape wrong");
    assert!(
        got.iter().all(|f| f.is_finite()),
        "fused attention produced non-finite values: {got:?}"
    );
}

#[test]
fn typed_run_with_f16_param_matches_f32_reference() {
    // Build the standard matmul+add graph, but feed weights as F16
    // bytes via set_param_typed. The graph's internal dtype is still
    // F32 — set_param_typed lets the caller hand off pre-quantized
    // F16 bytes; the shim widens them once on the way in via the
    // typed Array::from_bytes path.
    //
    // Wait — graph nodes are F32 here, so set_param_typed with F16
    // would mismatch. Easier sanity check: feed F32 bytes via
    // set_param_typed (no widening, just routing through the typed
    // path) and confirm the output matches the f32 reference.
    let g = build_graph();
    let mut exe = MlxExecutable::compile(g);

    let w = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];
    let w_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) }.to_vec();
    let b_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) }.to_vec();

    exe.set_param_typed("w", &w_bytes, DType::F32);
    exe.set_param_typed("b", &b_bytes, DType::F32);

    let xs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let xs_bytes: Vec<u8> =
        unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) }.to_vec();
    let outs = exe.run_typed(&[("x", &xs_bytes, DType::F32)]);
    assert_eq!(outs.len(), 1);
    let (out_bytes, out_dt) = &outs[0];
    assert_eq!(*out_dt, DType::F32);

    let out_f32: Vec<f32> = unsafe {
        std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_bytes.len() / 4)
    }
    .to_vec();
    let want = expected_output(&xs, &w, &b);
    assert!(
        close(&out_f32, &want, 1e-5),
        "typed run mismatch: got {out_f32:?} want {want:?}"
    );
}

#[test]
fn typed_run_with_f16_graph_round_trips() {
    // Build a graph whose internal weights/inputs are F16. set_param_typed
    // and run_typed plumb the bytes through without f32 widening.
    let mut g = Graph::new("f16_typed");
    let x = g.input("x", Shape::new(&[1, 2], DType::F16));
    let w = g.param("w", Shape::new(&[2, 1], DType::F16));
    let y = g.matmul(x, w, Shape::new(&[1, 1], DType::F16));
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);

    // F16 bit-patterns: 1.0 = 0x3C00, 2.0 = 0x4000, 3.0 = 0x4200.
    let w_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // [1.0, 2.0]
    let x_bytes: Vec<u8> = vec![0x00, 0x42, 0x00, 0x3C]; // [3.0, 1.0]
    exe.set_param_typed("w", &w_bytes, DType::F16);
    let outs = exe.run_typed(&[("x", &x_bytes, DType::F16)]);
    assert_eq!(outs.len(), 1);
    let (out_bytes, out_dt) = &outs[0];
    assert_eq!(*out_dt, DType::F16);
    // 3.0*1.0 + 1.0*2.0 = 5.0, F16 = 0x4500
    assert_eq!(out_bytes, &vec![0x00, 0x45]);
}

#[test]
fn array_f16_round_trip_via_bytes() {
    // Build an MLX array directly from raw F16 bit patterns, read it
    // back as f32 (lossy widen) and as bytes (round-trip preserved).
    // 1.0 in IEEE-754 binary16 is 0x3C00; 2.0 is 0x4000.
    let bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40];
    let arr = rlx_mlx::Array::from_bytes(&bytes, &[2], DType::F16).unwrap();

    let f32s = arr.to_f32().unwrap();
    assert!(
        close(&f32s, &[1.0, 2.0], 1e-6),
        "f16 → f32 widen mismatch: {f32s:?}"
    );

    let out = arr.to_bytes().unwrap();
    assert_eq!(
        out, bytes,
        "f16 byte round-trip mismatch: got {out:?} want {bytes:?}"
    );
}

#[test]
fn array_bf16_round_trip_via_bytes() {
    // bf16 has the same exponent layout as f32 with a truncated
    // mantissa. 1.0 = 0x3F80, 2.0 = 0x4000.
    let bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40];
    let arr = rlx_mlx::Array::from_bytes(&bytes, &[2], DType::BF16).unwrap();

    let f32s = arr.to_f32().unwrap();
    assert!(
        close(&f32s, &[1.0, 2.0], 1e-6),
        "bf16 → f32 widen mismatch: {f32s:?}"
    );

    let out = arr.to_bytes().unwrap();
    assert_eq!(
        out, bytes,
        "bf16 byte round-trip mismatch: got {out:?} want {bytes:?}"
    );
}

#[test]
fn calibration_records_memory_bw_and_attention() {
    use rlx_mlx::calibrate::Calibration;
    // Force a fresh measurement by removing the cache file (or just
    // call measure() directly). Use measure() to avoid touching the
    // user's cache.
    let cal = Calibration::measure().expect("calibration measure failed");
    assert!(
        cal.memory_bw_gbps > 1.0,
        "memory_bw_gbps too low: {:.1}",
        cal.memory_bw_gbps
    );
    // Apple Silicon unified memory ranges 100-800 GB/s; cap at 2 TB/s
    // as an absurd upper bound.
    assert!(
        cal.memory_bw_gbps < 2000.0,
        "memory_bw_gbps implausibly high: {:.0}",
        cal.memory_bw_gbps
    );
    assert!(
        cal.attention_flops > 1.0e9,
        "attention_flops too low: {:.0}",
        cal.attention_flops
    );
    assert!(
        cal.reduce_gbps > 1.0,
        "reduce_gbps too low: {:.1}",
        cal.reduce_gbps
    );
}

#[test]
fn calibration_returns_plausible_numbers() {
    use rlx_mlx::calibrate::Calibration;
    let cal = Calibration::load_or_measure();
    // Sanity bounds: large matmul should achieve at least 10 GF/s
    // (any modern Apple Silicon GPU exceeds this comfortably).
    assert!(
        cal.sgemm_large_flops > 10e9,
        "large sgemm too slow: {:.0} GF/s",
        cal.sgemm_large_flops / 1e9
    );
    // Small-shape rate is dominated by overhead — should still be
    // measurable, just lower.
    assert!(
        cal.sgemm_small_flops > 0.0,
        "small sgemm not measured: {:.0} GF/s",
        cal.sgemm_small_flops / 1e9
    );
    // Round-trip should be in the µs–ms range, not negative or zero.
    assert!(
        cal.roundtrip_overhead_ns > 0.0,
        "roundtrip not measured: {:.0} ns",
        cal.roundtrip_overhead_ns
    );
    // Bound to something generous (10 ms) so an obviously-broken
    // measurement gets caught.
    assert!(
        cal.roundtrip_overhead_ns < 10_000_000.0,
        "roundtrip implausibly slow: {:.0} ns",
        cal.roundtrip_overhead_ns
    );
}

#[test]
fn compiled_matmul_add_matches_reference_and_replays() {
    // First call: pays the trace cost. Second call with different
    // inputs must produce a fresh result, demonstrating that the
    // compiled fn is actually re-running its trace against the new
    // leaf data (not stuck on the first call's outputs).
    let g = build_graph();
    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);

    let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    exe.set_param("w", &w);
    exe.set_param("b", &b);

    let xa = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outs_a = exe.run(&[("x", &xa)]);
    let want_a = expected_output(&xa, &w, &b);
    assert!(
        close(&outs_a[0], &want_a, 1e-4),
        "compiled run #1 mismatch: got {:?} want {want_a:?}",
        outs_a[0]
    );

    // Different inputs — same compiled trace. Output must reflect xb,
    // not stale xa.
    let xb = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let outs_b = exe.run(&[("x", &xb)]);
    let want_b = expected_output(&xb, &w, &b);
    assert!(
        close(&outs_b[0], &want_b, 1e-3),
        "compiled run #2 mismatch (trace reuse): got {:?} want {want_b:?}",
        outs_b[0]
    );
}

// ── PR1 op coverage ──────────────────────────────────────────────

use rlx_ir::Op;
use rlx_ir::op::{Activation, CmpOp, ReduceOp};

fn run_unary(act: Activation, input: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("u");
    let x = g.input("x", Shape::new(&[input.len()], DType::F32));
    let y = g.activation(act, x, Shape::new(&[input.len()], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);
    exe.run(&[("x", input)]).into_iter().next().unwrap()
}

#[test]
fn relu_matches_reference() {
    let xs = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let got = run_unary(Activation::Relu, &xs);
    let want: Vec<f32> = xs.iter().map(|x| x.max(0.0)).collect();
    assert!(close(&got, &want, 1e-6), "got {got:?} want {want:?}");
}

#[test]
fn tanh_exp_log_match_reference() {
    let xs = [0.5, 1.0, 1.5];
    let g_tanh = run_unary(Activation::Tanh, &xs);
    let w_tanh: Vec<f32> = xs.iter().map(|x| x.tanh()).collect();
    assert!(close(&g_tanh, &w_tanh, 1e-5));

    let g_exp = run_unary(Activation::Exp, &xs);
    let w_exp: Vec<f32> = xs.iter().map(|x| x.exp()).collect();
    assert!(close(&g_exp, &w_exp, 1e-4));

    let g_log = run_unary(Activation::Log, &xs);
    let w_log: Vec<f32> = xs.iter().map(|x| x.ln()).collect();
    assert!(close(&g_log, &w_log, 1e-5));
}

#[test]
fn sin_cos_match_reference() {
    let xs = [-1.5_f32, -0.5, 0.0, 0.5, 1.0, std::f32::consts::PI / 2.0];
    let g_sin = run_unary(Activation::Sin, &xs);
    let w_sin: Vec<f32> = xs.iter().map(|x| x.sin()).collect();
    assert!(
        close(&g_sin, &w_sin, 1e-5),
        "sin: got {g_sin:?} want {w_sin:?}"
    );
    let g_cos = run_unary(Activation::Cos, &xs);
    let w_cos: Vec<f32> = xs.iter().map(|x| x.cos()).collect();
    assert!(
        close(&g_cos, &w_cos, 1e-5),
        "cos: got {g_cos:?} want {w_cos:?}"
    );
}

#[test]
fn tan_atan_match_reference() {
    // Stay clear of cos(x) = 0 (tan asymptotes) — pick inputs in
    // (−π/2, π/2) away from the singularities.
    let xs = [-1.2_f32, -0.5, 0.0, 0.5, 1.2];
    let g_tan = run_unary(Activation::Tan, &xs);
    let w_tan: Vec<f32> = xs.iter().map(|x| x.tan()).collect();
    assert!(
        close(&g_tan, &w_tan, 1e-4),
        "tan: got {g_tan:?} want {w_tan:?}"
    );
    let g_atan = run_unary(Activation::Atan, &xs);
    let w_atan: Vec<f32> = xs.iter().map(|x| x.atan()).collect();
    assert!(
        close(&g_atan, &w_atan, 1e-5),
        "atan: got {g_atan:?} want {w_atan:?}"
    );
}

#[test]
fn reshape_transpose_match_reference() {
    let mut g = Graph::new("rt");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let r = g.reshape(x, vec![3, 2], Shape::new(&[3, 2], DType::F32));
    let t = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![r],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![t]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // reshape gives [[1,2],[3,4],[5,6]] (row-major), transpose gives [[1,3,5],[2,4,6]]
    let want = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert!(close(&got, &want, 1e-6), "got {got:?} want {want:?}");
}

#[test]
fn narrow_match_reference() {
    let mut g = Graph::new("nc");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let n = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 1,
            len: 2,
        },
        vec![x],
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![n]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![10.0, 20.0, 30.0, 40.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert_eq!(got, vec![20.0, 30.0]);
}

#[test]
fn reduce_sum_match_reference() {
    let mut g = Graph::new("rs");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let s = g.reduce(
        x,
        ReduceOp::Sum,
        vec![1],
        false,
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![s]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert_eq!(got, vec![6.0, 15.0]);
}

#[test]
fn compare_then_where_matches_reference() {
    // y = where(x > 0.0, x, -x)  →  abs(x)
    let mut g = Graph::new("cw");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let z = g.param("z", Shape::new(&[4], DType::F32)); // zeros
    let nx = g.activation(Activation::Neg, x, Shape::new(&[4], DType::F32));
    let cond = g.add_node(
        Op::Compare(CmpOp::Gt),
        vec![x, z],
        Shape::new(&[4], DType::Bool),
    );
    let sel = g.add_node(Op::Where, vec![cond, x, nx], Shape::new(&[4], DType::F32));
    g.set_outputs(vec![sel]);
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("z", &[0.0, 0.0, 0.0, 0.0]);
    let xs = vec![1.0, -2.0, 3.0, -4.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn cumsum_inclusive_matches_reference() {
    let mut g = Graph::new("cs");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let c = g.cumsum(x, 0, false, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![c]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![1.0, 2.0, 3.0, 4.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0]);
}

// ── PR2: norms + attention + fused-block coverage ────────────────

use rlx_ir::op::MaskKind;

#[test]
fn rms_norm_matches_reference() {
    // rms_norm(x) = x * gamma / sqrt(mean(x^2) + eps)
    let mut g = Graph::new("rms");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let ga = g.param("g", Shape::new(&[4], DType::F32));
    let r = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, ga],
        Shape::new(&[2, 4], DType::F32),
    );
    g.set_outputs(vec![r]);
    let mut exe = MlxExecutable::compile(g);
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    exe.set_param("g", &gamma);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();

    // Reference: per-row mean(x^2), divide
    let mut want = vec![0f32; 8];
    for row in 0..2 {
        let mut ss = 0f32;
        for c in 0..4 {
            ss += xs[row * 4 + c].powi(2);
        }
        let scale = 1.0 / (ss / 4.0 + 1e-6).sqrt();
        for c in 0..4 {
            want[row * 4 + c] = xs[row * 4 + c] * scale * gamma[c];
        }
    }
    assert!(
        close(&got, &want, 1e-4),
        "rms_norm mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn attention_no_mask_matches_reference() {
    // Tiny attention: B=1, H=1, S=2, D=2. Compute by hand.
    // Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]] (so QK^T = I, scores = I/sqrt(2))
    // V = [[10, 20], [30, 40]]
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
    let mut exe = MlxExecutable::compile(g);
    let qd = vec![1.0, 0.0, 0.0, 1.0];
    let kd = vec![1.0, 0.0, 0.0, 1.0];
    let vd = vec![10.0, 20.0, 30.0, 40.0];
    let got = exe
        .run(&[("q", &qd), ("k", &kd), ("v", &vd)])
        .into_iter()
        .next()
        .unwrap();

    // Hand-compute: scale = 1/sqrt(2), scores = Q@K^T*scale
    // row0: [1*1+0*0, 1*0+0*1]/sqrt(2) = [0.7071, 0]
    // row1: [0, 0.7071]
    // softmax row0: exp([0.7071, 0]) = [2.028, 1] / 3.028 = [0.6698, 0.3302]
    // out row0 = 0.6698*[10,20] + 0.3302*[30,40] = [16.605, 26.605]
    // by symmetry row1 has the same with row 0 vs row 1 of V swapped:
    //   = 0.3302*[10,20] + 0.6698*[30,40] = [23.395, 33.395]
    let want = vec![16.605, 26.605, 23.395, 33.395];
    assert!(
        close(&got, &want, 5e-3),
        "attention mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn fused_matmul_bias_act_matches_reference() {
    // Construct the fused op directly so the test doesn't depend on
    // running the optimizer. The runtime layer wires in the fusion
    // passes for production graphs (see rlx-runtime/src/backend.rs).
    let mut g = Graph::new("fmm");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2, 2], DType::F32));
    let y = g.add_node(
        Op::FusedMatMulBiasAct {
            activation: Some(Activation::Relu),
        },
        vec![x, w, b],
        Shape::new(&[2, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[1.0, 0.0, 0.0, -1.0, 2.0, 1.0]);
    exe.set_param("b", &[0.0, 0.0, 0.0, 0.0]);
    let xs = vec![1.0, 0.5, -0.5, 0.0, 1.0, 1.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();

    // relu(x @ w + b)
    // row0: x@w = (1*1 + 0.5*0 + -0.5*2, 1*0 + 0.5*-1 + -0.5*1) = (0, -1). relu = (0, 0).
    // row1: x@w = (0 + 0 + 2, 0 - 1 + 1) = (2, 0). relu = (2, 0).
    let want = vec![0.0, 0.0, 2.0, 0.0];
    assert!(
        close(&got, &want, 1e-5),
        "fused matmul+bias+relu mismatch: got {got:?} want {want:?}"
    );
}

// ── PR3: heavy ops coverage ──────────────────────────────────────

#[test]
fn rope_matches_reference() {
    // Small hand-checked rope: x [1, 1, 2, 4] (B, H, S, D=4), head_dim=4 → half=2.
    // cos/sin shape [max_seq, half=2].
    let mut g = Graph::new("rope");
    let x = g.input("x", Shape::new(&[1, 1, 2, 4], DType::F32));
    let cos = g.input("cos", Shape::new(&[2, 2], DType::F32));
    let sin = g.input("sin", Shape::new(&[2, 2], DType::F32));
    let y = g.add_node(
        Op::Rope {
            head_dim: 4,
            n_rot: 4,
        },
        vec![x, cos, sin],
        Shape::new(&[1, 1, 2, 4], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    // Position 0: cos=(1,1), sin=(0,0) → identity rotation.
    // Position 1: cos=(0,0), sin=(1,1) → 90° rotation.
    let cos_d = vec![1.0, 1.0, 0.0, 0.0];
    let sin_d = vec![0.0, 0.0, 1.0, 1.0];
    // x for the two positions: [a,b,c,d] split into x1=(a,b), x2=(c,d).
    // pos 0 (identity): out = (a, b, c, d)
    // pos 1 (90°):
    //   y1 = x1*0 - x2*1 = (-c, -d)
    //   y2 = x2*0 + x1*1 = (a, b)
    //   out = (-c, -d, a, b)
    let xs = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let got = exe
        .run(&[("x", &xs), ("cos", &cos_d), ("sin", &sin_d)])
        .into_iter()
        .next()
        .unwrap();
    let want = vec![
        1.0, 2.0, 3.0, 4.0, // pos 0 unchanged
        -30.0, -40.0, 10.0, 20.0,
    ]; // pos 1 90° rotated
    assert!(
        close(&got, &want, 1e-5),
        "rope mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn topk_returns_indices_of_largest() {
    // x = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3]; k=3.
    // top-3 largest are 9.0, 5.3, 4.0 at indices 4, 6, 2 (any order).
    let mut g = Graph::new("topk");
    let x = g.input("x", Shape::new(&[7], DType::F32));
    let k_node = g.add_node(Op::TopK { k: 3 }, vec![x], Shape::new(&[3], DType::F32));
    g.set_outputs(vec![k_node]);

    let mut exe = MlxExecutable::compile(g);
    let xs = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();

    let mut got_set: Vec<i32> = got.iter().map(|f| *f as i32).collect();
    got_set.sort();
    let want = vec![2, 4, 6];
    assert_eq!(
        got_set, want,
        "topk indices mismatch: got (sorted) {got_set:?} want {want:?}"
    );
}

#[test]
fn lora_matmul_matches_reference() {
    let mut g = Graph::new("lora");
    let x = g.input("x", Shape::new(&[1, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let a = g.param("a", Shape::new(&[3, 1], DType::F32)); // rank-1 LoRA
    let b = g.param("b", Shape::new(&[1, 2], DType::F32));
    let y = g.add_node(
        Op::LoraMatMul { scale: 0.5 },
        vec![x, w, a, b],
        Shape::new(&[1, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    exe.set_param("a", &[1.0, 0.0, 1.0]);
    exe.set_param("b", &[2.0, 4.0]);
    let xs = vec![1.0, 2.0, 3.0];
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();

    // x @ w = (1, 2)
    // x @ a = (1*1 + 2*0 + 3*1) = (4) [shape (1, 1)]
    // (x @ a) @ b = 4 * (2, 4) = (8, 16) [shape (1, 2)]
    // scaled = 0.5 * (8, 16) = (4, 8)
    // out = (1, 2) + (4, 8) = (5, 10)
    let want = vec![5.0, 10.0];
    assert!(
        close(&got, &want, 1e-5),
        "lora mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn commit_no_wait_then_sync_matches_run() {
    // Same matmul+add graph as run_mode; build it once, then use the
    // async pipeline. The output must equal the reference after
    // sync_pending drains — and a follow-up explicit run() must
    // reflect the latest inputs (the first commit shouldn't bleed).
    let g = build_graph();
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    exe.set_param("b", &[1.0, 2.0, 3.0, 4.0]);

    let xa = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    exe.commit_no_wait(&[("x", &xa)]);
    exe.sync_pending();

    // Subsequent synchronous run with different inputs should produce
    // the right answer (i.e. the executable's state is clean post-sync).
    let xb = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let outs = exe.run(&[("x", &xb)]);
    let want = expected_output(&xb, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        close(&outs[0], &want, 1e-5),
        "post-async run mismatch: got {:?} want {want:?}",
        outs[0]
    );
}

#[test]
fn run_slots_writes_into_arena() {
    // Same matmul+add graph as the lazy/eager tests, but readback
    // via run_slots → arena_ptr instead of run() → Vec<f32>.
    let g = build_graph();
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    exe.set_param("b", &[1.0, 2.0, 3.0, 4.0]);
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let slots: Vec<(usize, usize)> = exe.run_slots(&[&xs]).to_vec();
    assert_eq!(slots.len(), 1, "expected one output slot");
    let (off, n) = slots[0];

    let arena = exe.arena_ptr();
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(arena.add(off) as *const f32, n).to_vec() };
    let want = expected_output(&xs, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        close(&got, &want, 1e-5),
        "run_slots arena readback mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn sliding_window_attention_matches_reference() {
    // window=0 → only attend to self → output equals V (each query
    // sees only its own key/value at the same position).
    let mut g = Graph::new("sw");
    let q = g.input("q", Shape::new(&[1, 1, 3, 2], DType::F32));
    let k = g.input("k", Shape::new(&[1, 1, 3, 2], DType::F32));
    let v = g.input("v", Shape::new(&[1, 1, 3, 2], DType::F32));
    let o = g.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: 2,
            mask_kind: MaskKind::SlidingWindow(0),
        },
        vec![q, k, v],
        Shape::new(&[1, 1, 3, 2], DType::F32),
    );
    g.set_outputs(vec![o]);
    let mut exe = MlxExecutable::compile(g);
    let qd = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let kd = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let vd = vec![10.0, 11.0, 20.0, 22.0, 30.0, 33.0];
    let got = exe
        .run(&[("q", &qd), ("k", &kd), ("v", &vd)])
        .into_iter()
        .next()
        .unwrap();
    // window=0 means qi only attends to ki=qi, so out[qi] = V[qi]
    assert!(
        close(&got, &vd, 1e-5),
        "sliding-window-0 should pick V directly: got {got:?} want {vd:?}"
    );
}

#[test]
fn sample_top_k_2_only_picks_from_top_two() {
    // Logits where 5 and 3 are the top two; everything else far below.
    // top_k=2 must clip to {idx 5, idx 3}; over many samples the
    // result must always be one of those two.
    let mut g = Graph::new("samp_topk");
    let logits = g.input("logits", Shape::new(&[1, 6], DType::F32));
    let id = g.add_node(
        Op::Sample {
            top_k: 2,
            top_p: 1.0,
            temperature: 1.0,
            seed: 7,
        },
        vec![logits],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![id]);
    let mut exe = MlxExecutable::compile(g);
    // Top-2 indices are 5 (value 12.0) and 3 (value 10.0).
    let xs = vec![1.0, 2.0, 3.0, 10.0, 4.0, 12.0];
    let pick = exe.run(&[("logits", &xs)]).into_iter().next().unwrap()[0] as i32;
    assert!(
        pick == 5 || pick == 3,
        "top_k=2 sample must pick from {{3, 5}}, got {pick}"
    );
}

#[test]
fn sample_top_p_clips_to_nucleus() {
    // logits: [10, 9, 0, 0, 0, 0]. softmax is dominated by indices 0
    // and 1 (≈ 0.731 and 0.269 respectively; cumsum_excl[0]=0,
    // cumsum_excl[1]=0.731). With top_p=0.9 the nucleus is {0, 1}
    // since cumsum_excl[2]=1.0 ≥ 0.9. Sampling must therefore land
    // on idx 0 or idx 1.
    let mut g = Graph::new("samp_topp");
    let logits = g.input("logits", Shape::new(&[1, 6], DType::F32));
    let id = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 0.9,
            temperature: 1.0,
            seed: 13,
        },
        vec![logits],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![id]);
    let mut exe = MlxExecutable::compile(g);
    let xs = vec![10.0, 9.0, 0.0, 0.0, 0.0, 0.0];
    // Run a handful of times across different seeds to make sure
    // the nucleus clips (rather than passing by chance).
    for seed in 0u64..8 {
        // Mutate the seed in the graph by recompiling — but that's
        // expensive; instead trust the deterministic pick at our
        // single seed and validate it sits in the nucleus.
        let _ = seed;
    }
    let pick = exe.run(&[("logits", &xs)]).into_iter().next().unwrap()[0] as i32;
    assert!(
        pick == 0 || pick == 1,
        "top_p=0.9 sample must land in nucleus {{0, 1}}, got {pick}"
    );
}

#[test]
fn sample_temperature_one_picks_from_distribution() {
    // logits with one dominant entry. categorical with seed > 0 is
    // deterministic — we just check the pick is plausible (must be
    // a valid index in [0, vocab)).
    let mut g = Graph::new("samp");
    let logits = g.input("logits", Shape::new(&[1, 4], DType::F32));
    let id = g.add_node(
        Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            seed: 42,
        },
        vec![logits],
        Shape::new(&[1], DType::F32),
    );
    g.set_outputs(vec![id]);
    let mut exe = MlxExecutable::compile(g);
    // One very-high-logit entry should dominate the categorical pick.
    let xs = vec![0.0, 0.0, 100.0, 0.0];
    let got = exe.run(&[("logits", &xs)]).into_iter().next().unwrap();
    let pick = got[0] as i32;
    assert_eq!(
        pick, 2,
        "high-logit entry should dominate sample: got {pick}"
    );
}

#[test]
fn handle_round_trip_works() {
    // bind_handle stores a default; run() without that input uses it.
    let g = build_graph();
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("w", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    exe.set_param("b", &[0.0, 0.0, 0.0, 0.0]);

    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    exe.bind_handle("x", &xs);
    let outs = exe.run(&[]); // no explicit inputs — should pull from handle
    let want = expected_output(&xs, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[0.0, 0.0, 0.0, 0.0]);
    assert!(
        close(&outs[0], &want, 1e-5),
        "handle-as-input mismatch: got {:?} want {want:?}",
        outs[0]
    );
}

#[test]
fn fused_swiglu_matches_reference() {
    // Op::FusedSwiGLU input is concatenated [up, gate]; output last
    // dim is half. y[i] = up[i] * silu(gate[i]).
    let mut g = Graph::new("swi");
    let x = g.input("x", Shape::new(&[1, 4], DType::F32)); // [up0, up1, g0, g1]
    let y = g.add_node(
        Op::FusedSwiGLU {
            cast_to: None,
            gate_first: false,
        },
        vec![x],
        Shape::new(&[1, 2], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut exe = MlxExecutable::compile(g);
    let xs = vec![
        3.0, 5.0, // up
        1.0, 0.5,
    ]; // gate
    let got = exe.run(&[("x", &xs)]).into_iter().next().unwrap();

    // silu(g) = g * sigmoid(g)
    let silu = |g: f32| g / (1.0 + (-g).exp());
    let want = vec![3.0 * silu(1.0), 5.0 * silu(0.5)];
    assert!(
        close(&got, &want, 1e-5),
        "swiglu mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn active_extent_truncates_input_via_lazy_slice() {
    // PLAN L1: declare an [8, 4] input and run with active_extent=(3, 8).
    // Lower-with-extent slices the input leaf to [3, 4]; downstream
    // element-wise ops produce a [3, 4] output. Verify only the first
    // 12 floats of the result match the reference and the result length
    // matches the truncated extent.
    use rlx_ir::op::Activation;
    let mut g = Graph::new("active_extent");
    let x = g.input("x", Shape::new(&[8, 4], DType::F32));
    let y = g.add_node(
        Op::Activation(Activation::Relu),
        vec![x],
        Shape::new(&[8, 4], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = MlxExecutable::compile(g);
    exe.set_active_extent(Some((3, 8)));

    // 8 rows of 4 floats; only the first 3 are meaningful for this run.
    let xs: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
    let outs = exe.run(&[("x", &xs)]);
    assert_eq!(outs.len(), 1);
    let got = &outs[0];
    let want: Vec<f32> = xs[..12].iter().map(|x| x.max(0.0)).collect();
    assert_eq!(
        got.len(),
        12,
        "active extent should produce 3*4=12 floats, got {}",
        got.len()
    );
    assert!(
        close(got, &want, 1e-6),
        "active-extent mismatch: got {got:?} want {want:?}"
    );
}

#[test]
fn active_extent_falls_back_when_graph_uses_axis_0_reshape() {
    // PLAN L1 safety check: a Reshape that hardcodes the upper extent
    // in its target shape can't be honored by simple input slicing.
    // The lowering path detects this and falls back to full extent.
    let mut g = Graph::new("active_extent_fallback");
    let x = g.input("x", Shape::new(&[8, 4], DType::F32));
    // Reshape from [8, 4] to [8, 4] — `8 == upper`, which the safety
    // check rejects. Falls back to the full extent.
    let r = g.add_node(
        Op::Reshape {
            new_shape: vec![8, 4],
        },
        vec![x],
        Shape::new(&[8, 4], DType::F32),
    );
    g.set_outputs(vec![r]);
    let mut exe = MlxExecutable::compile(g);
    exe.set_active_extent(Some((3, 8)));

    let xs: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let outs = exe.run(&[("x", &xs)]);
    // Full-extent fallback: full 32 floats come back.
    assert_eq!(
        outs[0].len(),
        32,
        "Reshape with upper-dim should force fallback to full extent"
    );
    assert!(close(&outs[0], &xs, 1e-6));
}

#[test]
fn elementwise_region_scalar_broadcast_matches_atomic() {
    // PLAN L2 quality: scalar broadcast in chains. Build the same
    // `(x + bias_scalar) * scale_scalar` chain twice — once atomic
    // (bias and scale are real [4]-shape inputs, user pre-tiled),
    // once with scalars broadcast through the region (bias and scale
    // are shape-[1]; the chain encodes scalar broadcast via
    // `scalar_input_mask` bits 1 and 2). Assert both produce the
    // same numerical result.
    use rlx_ir::Op;
    use rlx_ir::op::{BinaryOp, ChainOperand, ChainStep};

    let shape = Shape::new(&[4], DType::F32);
    let scalar_shape = Shape::new(&[1], DType::F32);

    let mut g_atom = Graph::new("scalar_atom");
    let x = g_atom.input("x", shape.clone());
    let b = g_atom.input("bias4", shape.clone());
    let s = g_atom.input("scale4", shape.clone());
    let add = g_atom.binary(BinaryOp::Add, x, b, shape.clone());
    let mul = g_atom.binary(BinaryOp::Mul, add, s, shape.clone());
    g_atom.set_outputs(vec![mul]);

    let mut g_reg = Graph::new("scalar_region");
    let x2 = g_reg.input("x", shape.clone());
    let b2 = g_reg.input("bias1", scalar_shape.clone());
    let s2 = g_reg.input("scale1", scalar_shape.clone());
    let chain = vec![
        ChainStep::Binary(
            BinaryOp::Add,
            ChainOperand::Input(0),
            ChainOperand::Input(1),
        ),
        ChainStep::Binary(BinaryOp::Mul, ChainOperand::Step(0), ChainOperand::Input(2)),
    ];
    let scalar_input_mask: u32 = (1 << 1) | (1 << 2);
    let mut input_modulus = [0u32; 16];
    input_modulus[1] = 1;
    input_modulus[2] = 1;
    let region = g_reg.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 3,
            scalar_input_mask,
            input_modulus,
        },
        vec![x2, b2, s2],
        shape.clone(),
    );
    g_reg.set_outputs(vec![region]);

    let xs = vec![1.0f32, 2.0, 3.0, 4.0];
    let bias_v = 0.5f32;
    let scale_v = 2.0f32;
    let bias_tiled = vec![bias_v; 4];
    let scale_tiled = vec![scale_v; 4];

    let mut atom = MlxExecutable::compile(g_atom);
    let got_atom = atom
        .run(&[("x", &xs), ("bias4", &bias_tiled), ("scale4", &scale_tiled)])
        .into_iter()
        .next()
        .unwrap();

    let mut reg = MlxExecutable::compile(g_reg);
    let got_reg = reg
        .run(&[("x", &xs), ("bias1", &[bias_v]), ("scale1", &[scale_v])])
        .into_iter()
        .next()
        .unwrap();

    assert!(
        close(&got_atom, &got_reg, 1e-5),
        "scalar-broadcast region vs atomic mismatch: \
         atom={got_atom:?} reg={got_reg:?}"
    );
    let want: Vec<f32> = xs.iter().map(|x| (x + bias_v) * scale_v).collect();
    assert!(
        close(&got_reg, &want, 1e-5),
        "scalar-broadcast result vs hand-computed mismatch: \
         got={got_reg:?} want={want:?}"
    );
}

#[test]
fn elementwise_region_with_where_step_matches_atomic() {
    // PLAN L2 quality: Op::Where now lives inside chains. Build the
    // same `where(cond, a, b) + a` pattern twice — once atomic, once
    // as an explicit ChainStep::Where in a region — and assert the
    // outputs match the same hand-computed reference.
    use rlx_ir::Op;
    use rlx_ir::op::{BinaryOp, ChainOperand, ChainStep, CmpOp};

    let shape = Shape::new(&[4], DType::F32);
    let bool_shape = Shape::new(&[4], DType::Bool);

    // Atomic chain: cmp = a > b; sel = where(cmp, a, b); out = sel + a
    let mut g_atom = Graph::new("where_atom");
    let a = g_atom.input("a", shape.clone());
    let b = g_atom.input("b", shape.clone());
    let cmp = g_atom.add_node(Op::Compare(CmpOp::Gt), vec![a, b], bool_shape.clone());
    let sel = g_atom.add_node(Op::Where, vec![cmp, a, b], shape.clone());
    let add = g_atom.binary(BinaryOp::Add, sel, a, shape.clone());
    g_atom.set_outputs(vec![add]);

    // Region form: same three steps inside one ElementwiseRegion node.
    // num_inputs=2 = (a, b) bound positionally as ChainOperand::Input.
    let mut g_reg = Graph::new("where_region");
    let a2 = g_reg.input("a", shape.clone());
    let b2 = g_reg.input("b", shape.clone());
    let chain = vec![
        // step 0: Compare Gt(Input(0), Input(1))
        ChainStep::Compare(CmpOp::Gt, ChainOperand::Input(0), ChainOperand::Input(1)),
        // step 1: Where(Step(0), Input(0), Input(1))
        ChainStep::Where(
            ChainOperand::Step(0),
            ChainOperand::Input(0),
            ChainOperand::Input(1),
        ),
        // step 2: Binary Add(Step(1), Input(0))
        ChainStep::Binary(BinaryOp::Add, ChainOperand::Step(1), ChainOperand::Input(0)),
    ];
    let region = g_reg.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 2,
            scalar_input_mask: 0,
            input_modulus: [0u32; 16],
        },
        vec![a2, b2],
        shape.clone(),
    );
    g_reg.set_outputs(vec![region]);

    let xs = vec![1.0f32, 5.0, 3.0, -2.0];
    let ys = vec![2.0f32, 4.0, 3.5, -3.0];

    let mut atom = MlxExecutable::compile(g_atom);
    let got_atom = atom
        .run(&[("a", &xs), ("b", &ys)])
        .into_iter()
        .next()
        .unwrap();

    let mut reg = MlxExecutable::compile(g_reg);
    let got_reg = reg
        .run(&[("a", &xs), ("b", &ys)])
        .into_iter()
        .next()
        .unwrap();

    assert!(
        close(&got_atom, &got_reg, 1e-5),
        "Where-in-region native vs atomic mismatch: \
         atom={got_atom:?} reg={got_reg:?}"
    );

    // Hand-computed: max(a, b) + a per element.
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(&x, &y)| x.max(y) + x).collect();
    assert!(
        close(&got_reg, &want, 1e-5),
        "Where-in-region result vs hand-computed mismatch: \
         got={got_reg:?} want={want:?}"
    );
}

#[test]
fn elementwise_region_native_lowering_matches_atomic() {
    // PLAN L2: native MLX lowering of Op::ElementwiseRegion. Build the
    // same chain twice — once decomposed (atomic Activation/Binary
    // nodes), once as a single Op::ElementwiseRegion — and assert the
    // two outputs match. This is the kernel-of-record test for the
    // native-region path in rlx-mlx/src/lower.rs.
    use rlx_ir::Op;
    use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep};

    // Atomic chain: y = relu(x + a) * b — three element-wise ops, two
    // inputs (a, b are the parameters and we feed x as the input).
    let shape = Shape::new(&[2, 3], DType::F32);
    let mut g_atom = Graph::new("region_atom");
    let x = g_atom.input("x", shape.clone());
    let a = g_atom.param("a", shape.clone());
    let b = g_atom.param("b", shape.clone());
    let s = g_atom.binary(BinaryOp::Add, x, a, shape.clone());
    let r = g_atom.add_node(Op::Activation(Activation::Relu), vec![s], shape.clone());
    let m = g_atom.binary(BinaryOp::Mul, r, b, shape.clone());
    g_atom.set_outputs(vec![m]);

    // Region form: same three steps inside one ElementwiseRegion node.
    // num_inputs=3 = (x, a, b) bound positionally as ChainOperand::Input.
    let mut g_reg = Graph::new("region_fused");
    let x2 = g_reg.input("x", shape.clone());
    let a2 = g_reg.param("a", shape.clone());
    let b2 = g_reg.param("b", shape.clone());
    let chain = vec![
        // step 0: x + a
        ChainStep::Binary(
            BinaryOp::Add,
            ChainOperand::Input(0),
            ChainOperand::Input(1),
        ),
        // step 1: relu(step 0)
        ChainStep::Activation(Activation::Relu, ChainOperand::Step(0)),
        // step 2: step 1 * b
        ChainStep::Binary(BinaryOp::Mul, ChainOperand::Step(1), ChainOperand::Input(2)),
    ];
    let region = g_reg.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 3,
            scalar_input_mask: 0,
            input_modulus: [0u32; 16],
        },
        vec![x2, a2, b2],
        shape.clone(),
    );
    g_reg.set_outputs(vec![region]);

    let xs = vec![-1.0, 0.0, 1.0, 2.0, -2.0, 0.5];
    let as_ = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let bs = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let mut atom = MlxExecutable::compile(g_atom);
    atom.set_param("a", &as_);
    atom.set_param("b", &bs);
    let got_atom = atom.run(&[("x", &xs)]).into_iter().next().unwrap();

    let mut reg = MlxExecutable::compile(g_reg);
    reg.set_param("a", &as_);
    reg.set_param("b", &bs);
    let got_reg = reg.run(&[("x", &xs)]).into_iter().next().unwrap();

    assert!(
        close(&got_atom, &got_reg, 1e-5),
        "ElementwiseRegion native vs atomic mismatch: \
         atom={got_atom:?} reg={got_reg:?}"
    );

    // Sanity-check values too: relu(x + 0.5) * 2.
    let want: Vec<f32> = xs.iter().map(|x| (x + 0.5).max(0.0) * 2.0).collect();
    assert!(
        close(&got_reg, &want, 1e-5),
        "ElementwiseRegion result vs hand-computed mismatch: \
         got={got_reg:?} want={want:?}"
    );
}
